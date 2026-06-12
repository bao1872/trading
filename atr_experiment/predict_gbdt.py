#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATR Rope GBDT 模型预测脚本

Purpose: 用回归/分类两个GBDT模型对指定日期的ATR选股结果做预测排序
Inputs:  atr_rope_selection 表 + stock_k_data 表
Outputs: 终端打印排序结果

How to Run:
    python atr_experiment/predict_gbdt.py --date 2026-05-21
    python atr_experiment/predict_gbdt.py --date 2026-05-21 --horizon 5

Examples:
    python atr_experiment/predict_gbdt.py --date 2026-05-21
    python atr_experiment/predict_gbdt.py --date 2026-05-21 --horizon 20

Side Effects: 只读数据库，无写入
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


# ==================== 训练与预测 ====================

def train_and_predict(train_df: pd.DataFrame, pred_df: pd.DataFrame,
                      target_col: str, model_type: str, horizon: int):
    """训练模型并预测"""
    import logging
    logging.getLogger("lightgbm").setLevel(logging.ERROR)

    X_train, available_features = prepare_features(train_df)
    X_pred, _ = prepare_features(pred_df)

    # 确保特征一致
    common_features = [f for f in available_features if f in X_pred.columns]
    X_train = X_train[common_features]
    X_pred = X_pred[common_features]
    cat_features = [f for f in CATEGORICAL_FEATURES if f in common_features]

    if model_type == "regression":
        y_train = train_df[target_col].values
        train_data = lgb.Dataset(X_train, label=y_train,
                                 categorical_feature=cat_features, free_raw_data=False)
        model = lgb.train(REG_PARAMS, train_data, num_boost_round=NUM_BOOST_ROUND)
        pred_values = model.predict(X_pred)
        print(f"  回归模型训练完成, 训练样本: {len(train_df)}, 预测样本: {len(pred_df)}")
    else:
        y_train = train_df[target_col].values
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        params = CLS_PARAMS.copy()
        if pos_count > 0 and neg_count > 0:
            params["scale_pos_weight"] = neg_count / pos_count

        train_data = lgb.Dataset(X_train, label=y_train,
                                 categorical_feature=cat_features, free_raw_data=False)
        model = lgb.train(params, train_data, num_boost_round=NUM_BOOST_ROUND)
        pred_values = model.predict(X_pred)
        print(f"  分类模型训练完成, 正类比例: {pos_count/len(y_train)*100:.1f}%, 预测样本: {len(pred_df)}")

    # 特征重要性
    gain = model.feature_importance("gain")
    gain_pct = gain / gain.sum() * 100 if gain.sum() > 0 else gain
    importance = dict(zip(common_features, gain_pct))
    top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top5 特征: {', '.join(f'{k}({v:.1f}%)' for k, v in top5)}")

    return pred_values, model, importance


# ==================== 输出格式化 ====================

def print_ranking(pred_df: pd.DataFrame, horizon: int):
    """打印排序结果"""
    reg_col = f"pred_return_{horizon}"
    cls_col = f"pred_prob_{horizon}"

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    # 回归模型排序
    print(f"\n{'='*70}")
    print(f"回归模型排序 (预测 {horizon}天收益, 按预测收益降序)")
    print(f"{'='*70}")

    reg_sorted = pred_df.sort_values(reg_col, ascending=False).reset_index(drop=True)
    display_cols = ["ts_code", "stock_name", "signal_type", "dsa_vwap_dev_pct",
                    "regime_strength", "rope_dir1_pct", reg_col, cls_col]
    available_cols = [c for c in display_cols if c in reg_sorted.columns]
    print(reg_sorted[available_cols].head(30).to_string(index=False))

    # 分类模型排序
    print(f"\n{'='*70}")
    print(f"分类模型排序 (预测 {horizon}天收益>{PROFIT_THRESHOLD*100:.0f}%概率, 按概率降序)")
    print(f"{'='*70}")

    cls_sorted = pred_df.sort_values(cls_col, ascending=False).reset_index(drop=True)
    print(cls_sorted[available_cols].head(30).to_string(index=False))

    # 两个模型 Top10 交集
    reg_top10 = set(reg_sorted.head(10)["ts_code"].values)
    cls_top10 = set(cls_sorted.head(10)["ts_code"].values)
    overlap = reg_top10 & cls_top10
    print(f"\n两个模型 Top10 交集: {overlap if overlap else '无交集'}")

    # 综合排序：回归预测收益 * 分类概率
    pred_df["combined_score"] = pred_df[reg_col] * pred_df[cls_col]
    combined_sorted = pred_df.sort_values("combined_score", ascending=False).reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"综合排序 (回归预测 × 分类概率, 降序)")
    print(f"{'='*70}")
    combined_cols = available_cols + ["combined_score"]
    print(combined_sorted[combined_cols].head(30).to_string(index=False))


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="ATR Rope GBDT 模型预测")
    parser.add_argument("--date", required=True, help="预测日期 (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=10, help="预测期限/天 (默认: 10)")
    args = parser.parse_args()

    target_date = args.date
    horizon = args.horizon
    ret_col = f"return_{horizon}"
    profit_col = f"profit_{horizon}"

    # 1. 加载选股数据（只到目标日期，避免加载未来数据）
    print("加载选股数据...")
    df = load_selection_data(target_date)
    print(f"全量数据: {len(df)} 条, 日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")

    # 2. 计算衍生字段
    df = compute_derived_fields(df)

    # 3. 计算未来收益（仅训练集需要）
    print("计算未来收益率...")
    df = enrich_with_future_metrics(df, [horizon])

    # 构建分类目标
    df[profit_col] = (df[ret_col] > PROFIT_THRESHOLD).astype(int)

    # 4. 划分训练集和预测集
    target_dt = pd.Timestamp(target_date).date()
    dates = pd.to_datetime(df["selection_date"]).dt.date
    train_df = df[dates < target_dt].dropna(subset=[ret_col]).copy()
    pred_df = df[dates == target_dt].copy()

    print(f"\n训练集: {len(train_df)} 条 (日期 < {target_date})")
    print(f"预测集: {len(pred_df)} 条 (日期 = {target_date})")

    if pred_df.empty:
        print(f"{target_date} 无选股数据，退出")
        return

    # 5. 训练回归模型
    print(f"\n--- 训练回归模型 (target={ret_col}) ---")
    reg_pred, reg_model, reg_importance = train_and_predict(
        train_df, pred_df, ret_col, "regression", horizon)
    pred_df[f"pred_return_{horizon}"] = reg_pred

    # 6. 训练分类模型
    print(f"\n--- 训练分类模型 (target={profit_col}, >{PROFIT_THRESHOLD*100:.0f}%) ---")
    cls_pred, cls_model, cls_importance = train_and_predict(
        train_df, pred_df, profit_col, "classification", horizon)
    pred_df[f"pred_prob_{horizon}"] = cls_pred

    # 7. 排序输出
    print_ranking(pred_df, horizon)

    # 8. 特征重要性对比
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


if __name__ == "__main__":
    main()
