#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATR GBDT 公共工具模块（SSOT）

Purpose: 集中管理 ATR GBDT 实验的特征配置、模型参数、衍生字段计算、未来收益计算
Inputs:  DataFrame / price_pivot
Outputs: 处理后的 DataFrame / metrics dict

How to Run:
    本模块为工具库，不直接运行。由 predict_gbdt.py / gbdt_backtest_evaluator.py / atr_gbdt_experiment.py 引用。

Examples:
    from atr_experiment.atr_gbdt_utils import (
        compute_derived_fields, compute_future_metrics_corrected,
        prepare_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )

Side Effects: 无
"""

import numpy as np
import pandas as pd

# ==================== 特征配置 ====================

NUMERIC_FEATURES = [
    "regime_strength", "rope_dev_pct", "rope_dev_atr", "range_width_pct",
    "change_pct", "vol_zscore", "dsa_dir_bars", "rope_dir1_pct", "rope_dir_neg1_pct",
    "range_pos_01", "dsa_vwap_dev_pct", "avg_amount_20d",
    "low_rope_dev_mean_pct", "low_rope_dev_std_pct", "low_rope_dev_today_pct",
    "low_vwap_dev_mean_pct", "low_vwap_dev_std_pct", "low_vwap_dev_today_pct",
]

CATEGORICAL_FEATURES = ["regime", "bbmacd_event", "low_rope_signal", "low_vwap_signal"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

HORIZONS = [3, 5, 10, 20]
PROFIT_THRESHOLD = 0.07  # 分类目标：收益 > 7%

# ==================== 模型参数 ====================

REG_PARAMS = {
    "objective": "regression", "metric": "mae",
    "num_leaves": 16, "max_depth": 5, "min_data_in_leaf": 50,
    "learning_rate": 0.03, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "max_bin": 63, "seed": 42, "verbosity": -1,
}

CLS_PARAMS = {
    "objective": "binary", "metric": "auc",
    "num_leaves": 16, "max_depth": 5, "min_data_in_leaf": 50,
    "learning_rate": 0.03, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "max_bin": 63, "seed": 42, "verbosity": -1,
}

NUM_BOOST_ROUND = 500


# ==================== 衍生字段计算 ====================

def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """计算衍生百分比字段

    将 low_rope_dev_xxx / low_vwap_dev_xxx 从绝对值转为占 rope_value 的百分比。
    所有计算仅用选股日T日已有数据，无前视偏差。
    """
    df = df.copy()
    for src_col, base_col, dst_col in [
        ("low_rope_dev_mean", "rope_value", "low_rope_dev_mean_pct"),
        ("low_rope_dev_std", "rope_value", "low_rope_dev_std_pct"),
        ("low_rope_dev_today", "rope_value", "low_rope_dev_today_pct"),
        ("low_vwap_dev_mean", "rope_value", "low_vwap_dev_mean_pct"),
        ("low_vwap_dev_std", "rope_value", "low_vwap_dev_std_pct"),
        ("low_vwap_dev_today", "rope_value", "low_vwap_dev_today_pct"),
    ]:
        if src_col in df.columns and base_col in df.columns:
            df[dst_col] = np.where(
                df[base_col].notna() & (df[base_col] != 0),
                df[src_col] / df[base_col] * 100, np.nan,
            )
        else:
            df[dst_col] = np.nan

    # rope_dev_pct 单位修正：若最大绝对值 < 1，说明是小数形式，需 ×100
    if "rope_dev_pct" in df.columns:
        sample_max = df["rope_dev_pct"].dropna().abs().max()
        if sample_max < 1.0:
            df["rope_dev_pct"] = df["rope_dev_pct"] * 100
    return df


# ==================== 未来收益计算（SSOT，无前视偏差） ====================

def compute_future_metrics_corrected(price_pivot, horizons):
    """以 open[T+1] 为入场价计算未来收益/MFE/MAE

    入场价: open[T+1]（信号日次日开盘价）
    return_N: close[T+N] / open[T+1] - 1
    MFE_N: 持仓期(T+1~T+N)最高价 / open[T+1] - 1
    MAE_N: 持仓期(T+1~T+N)最低价 / open[T+1] - 1
    """
    open_df = price_pivot["open"]
    close_df = price_pivot["close"]
    high_df = price_pivot["high"]
    low_df = price_pivot["low"]

    metrics = {}
    for N in horizons:
        entry = open_df.shift(-1)
        ret = close_df.shift(-N) / entry - 1
        mfe = (high_df.shift(-1).rolling(N, min_periods=1).max().shift(-(N - 1))) / entry - 1
        mae = (low_df.shift(-1).rolling(N, min_periods=1).min().shift(-(N - 1))) / entry - 1
        metrics[N] = {"return": ret, "mfe": mfe, "mae": mae}
    return metrics


def enrich_with_future_metrics(df: pd.DataFrame, horizons: list = None) -> pd.DataFrame:
    """为选股DataFrame添加未来收益/MFE/MAE列

    调用 compute_future_metrics_corrected + _lookup_values 向量化挂载。
    """
    if horizons is None:
        horizons = HORIZONS

    from stop_experiment.backtest.simple_backtest import load_daily_prices, build_price_pivot
    from stop_experiment.eval.filter_quality_evaluator import _lookup_values

    price_start = str(df["selection_date"].min())[:10]
    price_end = (pd.Timestamp(df["selection_date"].max()) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    print(f"加载价格数据: {price_start} ~ {price_end}...")
    daily_prices = load_daily_prices(price_start, price_end)
    price_pivot, _, _ = build_price_pivot(daily_prices)

    future_metrics = compute_future_metrics_corrected(price_pivot, horizons)

    df = df.copy()
    df["raw_code"] = df["ts_code"].str[:6]
    df["lookup_date"] = pd.to_datetime(df["selection_date"]).dt.normalize()

    for N in horizons:
        for metric_name in ["return", "mfe", "mae"]:
            col = f"{metric_name}_{N}"
            df[col] = _lookup_values(future_metrics[N][metric_name], df["lookup_date"].values, df["raw_code"].values)
        print(f"  return_{N}: 非空 {df[f'return_{N}'].notna().sum()}/{len(df)}")

    df.drop(columns=["lookup_date"], inplace=True)
    return df


# ==================== 特征矩阵准备 ====================

def prepare_features(df: pd.DataFrame) -> tuple:
    """准备特征矩阵，返回 (X, available_features)

    自动检测可用特征，分类特征转为 category 类型。
    """
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    X = df[available_features].copy()
    # 对象类型列转为数值类型
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X, available_features


# ==================== 数据加载 ====================

def load_selection_data(target_date: str = None) -> pd.DataFrame:
    """从数据库加载选股数据，只加载到目标日期（含）"""
    from sqlalchemy import text
    from datasource.database import get_engine

    engine = get_engine()
    if target_date:
        sql = text("SELECT * FROM atr_rope_selection WHERE selection_date <= :target_date ORDER BY selection_date, ts_code")
        params = {"target_date": target_date}
    else:
        sql = text("SELECT * FROM atr_rope_selection ORDER BY selection_date, ts_code")
        params = {}
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)
    engine.dispose()
    return df
