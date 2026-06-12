# -*- coding: utf-8 -*-
"""
SR 事件因子 GBDT v2 实验全局配置

Purpose: GBDT v2 实验的路径、超参、路径标签、场景分层、实验规格统一定义
Inputs:  无（纯配置常量）
Outputs: 无（被 import 使用）
How to Run: 无需单独运行
Side Effects: 无
"""
from __future__ import annotations

import os

SR_EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))
GBDT_OUTPUT_DIR = os.path.join(SR_EXPERIMENT_ROOT, "results", "gbdt")
DATASETS_DIR = os.path.join(GBDT_OUTPUT_DIR, "datasets")
MODELS_DIR = os.path.join(GBDT_OUTPUT_DIR, "models")
EVAL_DIR = os.path.join(GBDT_OUTPUT_DIR, "evaluation")

TRAIN_END = "2022-12-31"
VAL_END = "2024-12-31"

LGB_PARAMS = {
    "num_leaves": 31,
    "max_depth": 5,
    "min_data_in_leaf": 100,
    "learning_rate": 0.03,
    "n_estimators": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_bin": 63,
    "seed": 42,
    "verbosity": -1,
}

EARLY_STOPPING_ROUNDS = 50

TP_PCT_OPP = 0.08
SL_PCT_OPP = 0.06
TP_PCT_B = 0.10
SL_PCT_B = 0.08
BAD_BREAK_HORIZON = 10
PATH_HORIZON = 20


def support_reclaim_mask(df):
    return (
        df["evt_pierce_support_reclaim"].fillna(False).astype(bool)
        | df["evt_pierce_strong_support_cluster_reclaim"].fillna(False).astype(bool)
    )


def trend_pullback_mask(df):
    base = support_reclaim_mask(df)
    above_ma60 = df.get("close_above_ma60", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    ma60_slope = df.get("ma60_slope_pct", pd.Series(0.0, index=df.index)).fillna(0)
    return base & above_ma60 & (ma60_slope >= 0)


def oversold_bounce_mask(df):
    base = support_reclaim_mask(df)
    above_ma60 = df.get("close_above_ma60", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    ret_20 = df.get("ret_20", pd.Series(0.0, index=df.index)).fillna(0)
    return base & (~above_ma60 | (ret_20 < 0))


def cluster_low_volume_mask(df):
    return df["evt_pierce_support_cluster_reclaim_low_volume"].fillna(False).astype(bool)


import pandas as pd

EXPERIMENT_SPECS = {
    "A1_trend_opp": {
        "description": "趋势回踩-机会模型",
        "sample_filter": trend_pullback_mask,
        "label_name": "label_tp8_sl6_20",
        "objective": "binary",
        "model_type": "opportunity",
    },
    "A1_trend_risk": {
        "description": "趋势回踩-风险模型",
        "sample_filter": trend_pullback_mask,
        "label_name": "label_bad_break_10",
        "objective": "binary",
        "model_type": "risk",
    },
    "A2_oversold_opp": {
        "description": "超跌反抽-机会模型",
        "sample_filter": oversold_bounce_mask,
        "label_name": "label_tp8_sl6_20",
        "objective": "binary",
        "model_type": "opportunity",
    },
    "A2_oversold_risk": {
        "description": "超跌反抽-风险模型",
        "sample_filter": oversold_bounce_mask,
        "label_name": "label_bad_break_10",
        "objective": "binary",
        "model_type": "risk",
    },
    "B_cluster_opp": {
        "description": "强簇缩量-机会模型",
        "sample_filter": cluster_low_volume_mask,
        "label_name": "label_tp10_sl8_20",
        "objective": "binary",
        "model_type": "opportunity",
    },
    "B_cluster_risk": {
        "description": "强簇缩量-风险模型",
        "sample_filter": cluster_low_volume_mask,
        "label_name": "label_bad_break_10",
        "objective": "binary",
        "model_type": "risk",
    },
}

SCENE_PAIRS = {
    "A1_trend": ("A1_trend_opp", "A1_trend_risk"),
    "A2_oversold": ("A2_oversold_opp", "A2_oversold_risk"),
    "B_cluster": ("B_cluster_opp", "B_cluster_risk"),
}
