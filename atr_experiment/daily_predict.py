#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATR GBDT 每日预测脚本（双模型交集版）

Purpose: 用回归/分类两个GBDT模型对指定日期的ATR选股结果做预测，
         标记两模型预测分数和交集，按综合得分排序输出
Inputs:  atr_rope_selection 表 + stock_k_data 表
Outputs: 终端打印排序表 + 可选CSV

How to Run:
    # 首次运行（自动训练并保存模型）
    python atr_experiment/daily_predict.py --date 2026-05-11

    # 后续运行（直接加载已保存模型，秒级完成）
    python atr_experiment/daily_predict.py --date 2026-05-21

    # 强制重新训练模型
    python atr_experiment/daily_predict.py --date 2026-05-21 --retrain

    # 指定模型版本（训练截止日期）
    python atr_experiment/daily_predict.py --date 2026-05-21 --model-version 2026-05-10

Examples:
    python atr_experiment/daily_predict.py --date 2026-05-11
    python atr_experiment/daily_predict.py --date 2026-05-11 --top-k 5 10 --output output/predict.csv
    python atr_experiment/daily_predict.py --date 2026-05-21 --retrain

Side Effects: 只读数据库，可选写CSV文件，模型缓存写入 output/models/
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
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 模型缓存 ====================

def get_model_path(model_version: str, horizon: int, model_type: str) -> Path:
    """获取模型文件路径

    命名规则: {model_version}_h{horizon}_{model_type}.txt
    例如: 2026-05-10_h10_regression.txt
    """
    return MODEL_DIR / f"{model_version}_h{horizon}_{model_type}.txt"


def load_cached_model(model_version: str, horizon: int, model_type: str):
    """加载已缓存的模型，不存在返回None"""
    path = get_model_path(model_version, horizon, model_type)
    if path.exists():
        model = lgb.Booster(model_file=str(path))
        print(f"  加载缓存模型: {path.name}")
        return model
    return None


def save_model(model, model_version: str, horizon: int, model_type: str):
    """保存模型到缓存"""
    path = get_model_path(model_version, horizon, model_type)
    model.save_model(str(path))
    print(f"  模型已保存: {path.name}")


def find_latest_model(horizon: int, model_type: str, before_date: str = None):
    """查找最新的已缓存模型版本（训练截止日期 <= before_date）"""
    pattern = f"*_h{horizon}_{model_type}.txt"
    candidates = list(MODEL_DIR.glob(pattern))
    if not candidates:
        return None

    # 提取版本日期
    versions = []
    for p in candidates:
        version_str = p.name.split("_h")[0]
        try:
            version_dt = pd.Timestamp(version_str).date()
            if before_date is None or version_dt < pd.Timestamp(before_date).date():
                versions.append((version_dt, version_str, p))
        except Exception:
            continue

    if not versions:
        return None

    # 返回最新版本
    versions.sort(reverse=True)
    return versions[0][1]


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


# ==================== 预测与标记 ====================

def predict_and_mark(pred_df: pd.DataFrame, reg_model, cls_model, top_k_list: list, horizon: int):
    """双模型预测并标记交集"""
    X_pred, available_features = prepare_features(pred_df)

    # 预测
    pred_df = pred_df.copy()
    pred_df["pred_return"] = reg_model.predict(X_pred)
    pred_df["pred_prob"] = cls_model.predict(X_pred)
    pred_df["combined_score"] = pred_df["pred_return"] * pred_df["pred_prob"]

    # 回归/分类排名
    pred_df["reg_rank"] = pred_df["pred_return"].rank(ascending=False, method="min").astype(int)
    pred_df["cls_rank"] = pred_df["pred_prob"].rank(ascending=False, method="min").astype(int)
    pred_df["combined_rank"] = pred_df["combined_score"].rank(ascending=False, method="min").astype(int)

    # 标记各TopK
    for k in top_k_list:
        reg_top = set(pred_df.nlargest(k, "pred_return")["ts_code"].values)
        cls_top = set(pred_df.nlargest(k, "pred_prob")["ts_code"].values)
        overlap = reg_top & cls_top

        pred_df[f"in_reg_top{k}"] = pred_df["ts_code"].isin(reg_top)
        pred_df[f"in_cls_top{k}"] = pred_df["ts_code"].isin(cls_top)
        pred_df[f"in_overlap_top{k}"] = pred_df["ts_code"].isin(overlap)

    return pred_df


# ==================== 输出格式化 ====================

DISPLAY_COLUMNS = [
    "ts_code", "stock_name", "signal_type",
    "dsa_vwap_dev_pct", "regime_strength", "rope_dir1_pct",
    "pred_return", "pred_prob", "combined_score",
    "reg_rank", "cls_rank", "combined_rank",
]

DISPLAY_NAMES = {
    "ts_code": "股票代码", "stock_name": "股票名称", "signal_type": "信号类型",
    "dsa_vwap_dev_pct": "VWAP偏离%", "regime_strength": "趋势强度", "rope_dir1_pct": "Rope+1占比",
    "pred_return": "预测收益", "pred_prob": ">7%概率", "combined_score": "综合得分",
    "reg_rank": "回归排名", "cls_rank": "分类排名", "combined_rank": "综合排名",
}


def print_results(pred_df: pd.DataFrame, top_k_list: list, horizon: int):
    """打印预测结果"""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    # 按综合得分降序
    sorted_df = pred_df.sort_values("combined_score", ascending=False).reset_index(drop=True)

    # 交集标记列
    overlap_cols = []
    for k in top_k_list:
        overlap_cols.extend([f"in_reg_top{k}", f"in_cls_top{k}", f"in_overlap_top{k}"])

    # 构建显示列
    show_cols = DISPLAY_COLUMNS + overlap_cols
    available_cols = [c for c in show_cols if c in sorted_df.columns]

    # 终端输出：交集标记用 ★ 替换
    display_df = sorted_df[available_cols].copy()
    for col in overlap_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map({True: "★", False: ""})

    # 重命名
    rename_map = {k: v for k, v in DISPLAY_NAMES.items() if k in display_df.columns}
    for k in top_k_list:
        rename_map[f"in_reg_top{k}"] = f"回归T{k}"
        rename_map[f"in_cls_top{k}"] = f"分类T{k}"
        rename_map[f"in_overlap_top{k}"] = f"交集T{k}"
    display_df.rename(columns=rename_map, inplace=True)

    print(f"\n{'='*120}")
    print(f"ATR GBDT 每日预测 (日期={sorted_df.iloc[0].get('selection_date', '')}, 预测{horizon}天收益)")
    print(f"按综合得分降序，★=入选该TopK")
    print(f"{'='*120}")
    print(display_df.head(50).to_string(index=False))

    # 交集汇总
    for k in top_k_list:
        overlap_col = f"in_overlap_top{k}"
        if overlap_col in sorted_df.columns:
            overlap_stocks = sorted_df[sorted_df[overlap_col]]
            if not overlap_stocks.empty:
                print(f"\n--- 两模型交集 Top{k} ({len(overlap_stocks)} 只) ---")
                overlap_display = overlap_stocks[["ts_code", "stock_name", "pred_return", "pred_prob", "combined_score"]]
                overlap_display = overlap_display.sort_values("combined_score", ascending=False)
                print(overlap_display.to_string(index=False))

    # 各模型TopK独立输出
    for k in top_k_list:
        print(f"\n--- 回归模型 Top{k} ---")
        reg_top = sorted_df[sorted_df[f"in_reg_top{k}"]].sort_values("pred_return", ascending=False)
        print(reg_top[["ts_code", "stock_name", "pred_return", "pred_prob", "combined_score"]].to_string(index=False))

        print(f"\n--- 分类模型 Top{k} ---")
        cls_top = sorted_df[sorted_df[f"in_cls_top{k}"]].sort_values("pred_prob", ascending=False)
        print(cls_top[["ts_code", "stock_name", "pred_return", "pred_prob", "combined_score"]].to_string(index=False))


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="ATR GBDT 每日预测（双模型交集版）")
    parser.add_argument("--date", required=True, help="预测日期 (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=10, help="预测期限/天 (默认: 10)")
    parser.add_argument("--top-k", nargs="+", type=int, default=[5, 10], help="TopK 列表 (默认: 5 10)")
    parser.add_argument("--output", type=str, default=None, help="输出 CSV 路径 (相对于 atr_experiment/)")
    parser.add_argument("--retrain", action="store_true", help="强制重新训练模型（忽略缓存）")
    parser.add_argument("--model-version", type=str, default=None,
                        help="模型版本/训练截止日期 (默认: 自动查找最新缓存或使用预测日期前一天)")
    args = parser.parse_args()

    target_date = args.date
    horizon = args.horizon
    top_k_list = args.top_k
    ret_col = f"return_{horizon}"
    profit_col = f"profit_{horizon}"

    # 确定模型版本（训练截止日期）
    if args.model_version:
        model_version = args.model_version
    elif not args.retrain:
        # 尝试查找已有缓存
        latest_reg = find_latest_model(horizon, "regression", before_date=target_date)
        latest_cls = find_latest_model(horizon, "classification", before_date=target_date)
        if latest_reg and latest_cls:
            # 取两者中较旧的版本（确保一致性）
            model_version = min(latest_reg, latest_cls)
            print(f"使用缓存模型版本: {model_version}")
        else:
            model_version = None  # 需要训练
    else:
        model_version = None

    # 尝试加载缓存模型
    reg_model = None
    cls_model = None
    if model_version and not args.retrain:
        reg_model = load_cached_model(model_version, horizon, "regression")
        cls_model = load_cached_model(model_version, horizon, "classification")

    # 如果没有缓存，需要训练
    if reg_model is None or cls_model is None:
        # 训练版本 = 预测日期前一天（确保不使用未来数据）
        model_version = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # 加载选股数据
        print("加载选股数据（训练用）...")
        df = load_selection_data(model_version)
        print(f"全量数据: {len(df)} 条, 日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")

        # 计算衍生字段
        df = compute_derived_fields(df)

        # 计算未来收益（训练集标签）
        print("计算未来收益率 (训练集标签)...")
        df = enrich_with_future_metrics(df, [horizon])
        df[profit_col] = (df[ret_col] > PROFIT_THRESHOLD).astype(int)

        # 训练集 = 模型版本日期之前
        version_dt = pd.Timestamp(model_version).date()
        dates = pd.to_datetime(df["selection_date"]).dt.date
        train_df = df[dates <= version_dt].dropna(subset=[ret_col]).copy()

        print(f"\n训练集: {len(train_df)} 条 (日期 <= {model_version})")

        # 训练回归模型
        if reg_model is None:
            print(f"\n--- 训练回归模型 (target={ret_col}, 样本={len(train_df)}) ---")
            reg_model, reg_importance = train_model(train_df, ret_col, "regression")
            top5_reg = sorted(reg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top5 特征: {', '.join(f'{k}({v:.1f}%)' for k, v in top5_reg)}")
            save_model(reg_model, model_version, horizon, "regression")

        # 训练分类模型
        if cls_model is None:
            train_cls = df[dates <= version_dt].dropna(subset=[profit_col]).copy()
            print(f"\n--- 训练分类模型 (target={profit_col}, >{PROFIT_THRESHOLD*100:.0f}%, 样本={len(train_cls)}) ---")
            cls_model, cls_importance = train_model(train_cls, profit_col, "classification")
            top5_cls = sorted(cls_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top5 特征: {', '.join(f'{k}({v:.1f}%)' for k, v in top5_cls)}")
            save_model(cls_model, model_version, horizon, "classification")

    # 加载预测日数据
    print(f"\n加载 {target_date} 选股数据（预测用）...")
    pred_df = load_selection_data(target_date)
    # 只取预测日当天
    target_dt = pd.Timestamp(target_date).date()
    dates = pd.to_datetime(pred_df["selection_date"]).dt.date
    pred_df = pred_df[dates == target_dt].copy()

    if pred_df.empty:
        print(f"{target_date} 无选股数据，退出")
        return

    # 计算衍生字段
    pred_df = compute_derived_fields(pred_df)
    print(f"预测集: {len(pred_df)} 条 (日期 = {target_date})")

    # 预测并标记交集
    print(f"\n--- 预测并标记交集 (模型版本={model_version}, top_k={top_k_list}) ---")
    pred_df = predict_and_mark(pred_df, reg_model, cls_model, top_k_list, horizon)

    # 输出结果
    print_results(pred_df, top_k_list, horizon)

    # 输出CSV
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = OUTPUT_DIR / output_path
        pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\nCSV 已保存: {output_path}")


if __name__ == "__main__":
    main()
