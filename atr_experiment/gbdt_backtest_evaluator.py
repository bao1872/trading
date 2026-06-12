#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATR GBDT 回测质量评估器

Purpose: 用2026年数据做测试、之前数据做训练，评估回归/分类两个GBDT模型
         在 top5/top10 场景下 3/5/10/20 天的回测指标
Inputs:  atr_rope_selection 表 + stock_k_data 表
Outputs: 终端打印回测指标汇总表 + CSV

How to Run:
    python atr_experiment/gbdt_backtest_evaluator.py
    python atr_experiment/gbdt_backtest_evaluator.py --split-date 2026-01-01
    python atr_experiment/gbdt_backtest_evaluator.py --top-k 5 10 20

Examples:
    python atr_experiment/gbdt_backtest_evaluator.py
    python atr_experiment/gbdt_backtest_evaluator.py --split-date 2025-07-01

Side Effects: 只读数据库，写入 output/ 目录 CSV
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from atr_experiment.atr_gbdt_utils import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES,
    HORIZONS, PROFIT_THRESHOLD, REG_PARAMS, CLS_PARAMS, NUM_BOOST_ROUND,
    compute_derived_fields, enrich_with_future_metrics,
    prepare_features, load_selection_data,
)

OUTPUT_DIR = PROJECT_ROOT / "atr_experiment" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 模型训练 ====================

def train_model(train_df: pd.DataFrame, target_col: str, model_type: str):
    """训练单个模型"""
    import logging
    logging.getLogger("lightgbm").setLevel(logging.ERROR)

    X_train, available_features = prepare_features(train_df)
    y_train = train_df[target_col].values

    cat_features = [f for f in CATEGORICAL_FEATURES if f in available_features]
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)

    if model_type == "regression":
        model = lgb.train(REG_PARAMS, train_data, num_boost_round=NUM_BOOST_ROUND)
    else:
        params = CLS_PARAMS.copy()
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        if pos_count > 0 and neg_count > 0:
            params["scale_pos_weight"] = neg_count / pos_count
        model = lgb.train(params, train_data, num_boost_round=NUM_BOOST_ROUND)

    # 特征重要性
    gain = model.feature_importance("gain")
    gain_pct = gain / gain.sum() * 100 if gain.sum() > 0 else gain
    importance = dict(zip(available_features, gain_pct))

    return model, importance


# ==================== 回测评估 ====================

def evaluate_daily(test_df: pd.DataFrame, reg_model, cls_model, top_k_list: list):
    """对测试集每个交易日做预测并评估

    Returns:
        daily_detail: 每日每场景的指标明细 (DataFrame)
        summary: 按模型×场景×期限聚合的汇总 (DataFrame)
    """
    test_dates = sorted(test_df["selection_date"].unique())
    print(f"\n测试集: {len(test_dates)} 个交易日, {len(test_df)} 条记录")

    # 准备特征
    X_all, available_features = prepare_features(test_df)

    # 预测
    cat_features = [f for f in CATEGORICAL_FEATURES if f in available_features]
    test_df = test_df.copy()
    test_df["pred_return"] = reg_model.predict(X_all)
    test_df["pred_prob"] = cls_model.predict(X_all)
    test_df["combined_score"] = test_df["pred_return"] * test_df["pred_prob"]

    # 定义评估场景
    scenarios = []
    for k in top_k_list:
        scenarios.append(("regression", f"top{k}", "pred_return", k))
        scenarios.append(("classification", f"top{k}", "pred_prob", k))
        scenarios.append(("combined", f"top{k}", "combined_score", k))

    # 逐日评估
    daily_records = []

    for date in test_dates:
        day_df = test_df[test_df["selection_date"] == date]

        # 全候选池基线
        for h in HORIZONS:
            ret_col = f"return_{h}"
            mfe_col = f"mfe_{h}"
            mae_col = f"mae_{h}"
            valid = day_df.dropna(subset=[ret_col])
            if len(valid) > 0:
                daily_records.append({
                    "date": date, "model": "baseline", "scenario": "all",
                    "horizon": h,
                    "avg_return": valid[ret_col].mean(),
                    "win_rate": (valid[ret_col] > 0).mean(),
                    "avg_mfe": valid[mfe_col].mean() if mfe_col in valid.columns else np.nan,
                    "avg_mae": valid[mae_col].mean() if mae_col in valid.columns else np.nan,
                    "n": len(valid),
                })

        # 各模型场景
        for model_key, scenario_name, sort_col, k in scenarios:
            top_stocks = day_df.nlargest(k, sort_col)
            for h in HORIZONS:
                ret_col = f"return_{h}"
                mfe_col = f"mfe_{h}"
                mae_col = f"mae_{h}"
                valid = top_stocks.dropna(subset=[ret_col])
                if len(valid) > 0:
                    daily_records.append({
                        "date": date, "model": model_key, "scenario": scenario_name,
                        "horizon": h,
                        "avg_return": valid[ret_col].mean(),
                        "win_rate": (valid[ret_col] > 0).mean(),
                        "avg_mfe": valid[mfe_col].mean() if mfe_col in valid.columns else np.nan,
                        "avg_mae": valid[mae_col].mean() if mae_col in valid.columns else np.nan,
                        "n": len(valid),
                    })

        # 两模型交集
        for k in top_k_list:
            reg_top = set(day_df.nlargest(k, "pred_return")["ts_code"].values)
            cls_top = set(day_df.nlargest(k, "pred_prob")["ts_code"].values)
            overlap_codes = reg_top & cls_top
            if overlap_codes:
                overlap_df = day_df[day_df["ts_code"].isin(overlap_codes)]
                for h in HORIZONS:
                    ret_col = f"return_{h}"
                    mfe_col = f"mfe_{h}"
                    mae_col = f"mae_{h}"
                    valid = overlap_df.dropna(subset=[ret_col])
                    if len(valid) > 0:
                        daily_records.append({
                            "date": date, "model": "overlap", "scenario": f"top{k}",
                            "horizon": h,
                            "avg_return": valid[ret_col].mean(),
                            "win_rate": (valid[ret_col] > 0).mean(),
                            "avg_mfe": valid[mfe_col].mean() if mfe_col in valid.columns else np.nan,
                            "avg_mae": valid[mae_col].mean() if mae_col in valid.columns else np.nan,
                            "n": len(valid),
                        })

    daily_detail = pd.DataFrame(daily_records)

    # 汇总
    if daily_detail.empty:
        return daily_detail, pd.DataFrame()

    summary = daily_detail.groupby(["model", "scenario", "horizon"]).agg(
        avg_return=("avg_return", "mean"),
        win_rate=("win_rate", "mean"),
        avg_mfe=("avg_mfe", "mean"),
        avg_mae=("avg_mae", "mean"),
        n_days=("date", "count"),
        n_avg=("n", "mean"),
    ).reset_index()

    return daily_detail, summary


# ==================== 输出格式化 ====================

def print_summary(summary: pd.DataFrame):
    """打印汇总表"""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print(f"\n{'='*90}")
    print("模型×场景×期限 回测指标汇总 (2026年测试集)")
    print(f"{'='*90}")
    print(f"{'模型':<16s} {'场景':<8s} {'期限':>4s}  {'平均收益':>8s}  {'胜率':>6s}  {'平均MFE':>8s}  {'平均MAE':>8s}  {'天数':>4s}  {'均样本':>6s}")
    print("-" * 90)

    # 排序：baseline → regression → classification → combined → overlap
    model_order = {"baseline": 0, "regression": 1, "classification": 2, "combined": 3, "overlap": 4}
    summary["_model_order"] = summary["model"].map(model_order)
    summary.sort_values(["_model_order", "scenario", "horizon"], inplace=True)

    for _, row in summary.iterrows():
        print(f"{row['model']:<16s} {row['scenario']:<8s} {row['horizon']:>3d}d  "
              f"{row['avg_return']*100:>7.2f}%  {row['win_rate']*100:>5.1f}%  "
              f"{row['avg_mfe']*100:>7.2f}%  {row['avg_mae']*100:>7.2f}%  "
              f"{row['n_days']:>4d}  {row['n_avg']:>6.1f}")

    # 对比分析：各模型 vs baseline
    print(f"\n{'='*90}")
    print("对比分析：各模型 vs 全候选池基线 (收益差值)")
    print(f"{'='*90}")

    baseline = summary[summary["model"] == "baseline"].set_index("horizon")["avg_return"]
    for model_name in ["regression", "classification", "combined", "overlap"]:
        for scenario in summary[summary["model"] == model_name]["scenario"].unique():
            sub = summary[(summary["model"] == model_name) & (summary["scenario"] == scenario)]
            print(f"\n{model_name} ({scenario}):")
            for _, row in sub.iterrows():
                base_ret = baseline.get(row["horizon"], 0)
                diff = (row["avg_return"] - base_ret) * 100
                print(f"  {row['horizon']:>2d}d: {row['avg_return']*100:>7.2f}% (基线 {base_ret*100:>7.2f}%, 差值 {diff:>+6.2f}%)")

    # 交集统计
    overlap_stats = summary[summary["model"] == "overlap"]
    if not overlap_stats.empty:
        print(f"\n{'='*90}")
        print("两模型交集表现")
        print(f"{'='*90}")
        for _, row in overlap_stats.iterrows():
            print(f"  {row['scenario']} {row['horizon']:>2d}d: 收益={row['avg_return']*100:>7.2f}%, "
                  f"胜率={row['win_rate']*100:>5.1f}%, MFE={row['avg_mfe']*100:>7.2f}%, "
                  f"MAE={row['avg_mae']*100:>7.2f}%, 均样本={row['n_avg']:.1f}")


def print_feature_importance(reg_importance: dict, cls_importance: dict):
    """打印特征重要性对比"""
    print(f"\n{'='*70}")
    print("特征重要性对比 (回归 vs 分类)")
    print(f"{'='*70}")
    all_feats = set(reg_importance.keys()) | set(cls_importance.keys())
    print(f"{'特征':<30s} {'回归Gain%':>10s} {'分类Gain%':>10s}")
    print("-" * 52)
    for feat in sorted(all_feats, key=lambda f: reg_importance.get(f, 0) + cls_importance.get(f, 0), reverse=True):
        r = reg_importance.get(feat, 0)
        c = cls_importance.get(feat, 0)
        print(f"{feat:<30s} {r:>10.2f} {c:>10.2f}")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="ATR GBDT 回测质量评估器")
    parser.add_argument("--split-date", default="2026-01-01", help="训练/测试分割日期 (默认: 2026-01-01)")
    parser.add_argument("--top-k", nargs="+", type=int, default=[5, 10], help="TopK 列表 (默认: 5 10)")
    args = parser.parse_args()

    split_date = args.split_date
    top_k_list = args.top_k

    # 1. 加载全量选股数据
    print("加载选股数据...")
    df = load_selection_data()
    print(f"全量数据: {len(df)} 条, 日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")

    # 2. 计算衍生字段
    df = compute_derived_fields(df)

    # 3. 计算未来收益/MFE/MAE
    print("计算未来收益率 (入场价=open[T+1])...")
    df = enrich_with_future_metrics(df, HORIZONS)

    # 构建分类目标
    for h in HORIZONS:
        ret_col = f"return_{h}"
        profit_col = f"profit_{h}"
        if ret_col in df.columns:
            df[profit_col] = (df[ret_col] > PROFIT_THRESHOLD).astype(int)

    # 4. 划分训练/测试集
    split_dt = pd.Timestamp(split_date).date()
    dates = pd.to_datetime(df["selection_date"]).dt.date
    train_df = df[dates < split_dt].copy()
    test_df = df[dates >= split_dt].copy()

    print(f"\n训练集: {len(train_df)} 条 (日期 < {split_date})")
    print(f"测试集: {len(test_df)} 条 (日期 >= {split_date})")

    # 5. 训练回归模型
    target_ret = "return_10"
    target_profit = "profit_10"
    train_valid = train_df.dropna(subset=[target_ret])

    print(f"\n--- 训练回归模型 (target={target_ret}, 样本={len(train_valid)}) ---")
    reg_model, reg_importance = train_model(train_valid, target_ret, "regression")
    top5_reg = sorted(reg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top5 特征: {', '.join(f'{k}({v:.1f}%)' for k, v in top5_reg)}")

    # 6. 训练分类模型
    train_valid_cls = train_df.dropna(subset=[target_profit])
    print(f"\n--- 训练分类模型 (target={target_profit}, >{PROFIT_THRESHOLD*100:.0f}%, 样本={len(train_valid_cls)}) ---")
    cls_model, cls_importance = train_model(train_valid_cls, target_profit, "classification")
    top5_cls = sorted(cls_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top5 特征: {', '.join(f'{k}({v:.1f}%)' for k, v in top5_cls)}")

    # 7. 回测评估
    print(f"\n--- 回测评估 (top_k={top_k_list}) ---")
    daily_detail, summary = evaluate_daily(test_df, reg_model, cls_model, top_k_list)

    # 8. 输出结果
    if not summary.empty:
        print_summary(summary)
        print_feature_importance(reg_importance, cls_importance)

        # 保存 CSV
        summary.to_csv(OUTPUT_DIR / "gbdt_backtest_summary.csv", index=False, encoding="utf-8-sig")
        daily_detail.to_csv(OUTPUT_DIR / "gbdt_backtest_daily.csv", index=False, encoding="utf-8-sig")
        print(f"\nCSV 已保存: {OUTPUT_DIR / 'gbdt_backtest_summary.csv'}")
        print(f"CSV 已保存: {OUTPUT_DIR / 'gbdt_backtest_daily.csv'}")
    else:
        print("无回测结果")

    print(f"\n{'='*70}")
    print("评估完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
