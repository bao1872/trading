#!/usr/bin/env python3
"""
DSA V型反转 GBDT 因子组合探索实验（重设计版）

Purpose: 用 LightGBM 探索 24 因子的非线性组合，预测未来标签，含因子区间效应分析
Inputs: stock_dsa_vreversal_results (选股结果表)
Outputs: CSV (折指标、Decile分组、特征重要性、因子区间效应)、JSON (最优超参数)、LightGBM 模型文件
How to Run:
    python analysis/dsa_vreversal_gbdt_explorer.py                              # 默认3折
    python analysis/dsa_vreversal_gbdt_explorer.py --n-folds 4
    python analysis/dsa_vreversal_gbdt_explorer.py --tune                       # 超参搜索
    python analysis/dsa_vreversal_gbdt_explorer.py --sample-limit 2000          # 快速测试
    python analysis/dsa_vreversal_gbdt_explorer.py --only-reversal              # 只用有反转的
Examples:
    python analysis/dsa_vreversal_gbdt_explorer.py
    python analysis/dsa_vreversal_gbdt_explorer.py --sample-limit 2000
Side Effects: 只读操作，不写数据库；输出文件到指定目录
"""

import sys
import os
import json
import argparse
import warnings
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    log_loss,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RAW_FEATURES = [
    "dsa_dir", "prev_pivot_code", "last_confirmed_high", "last_confirmed_low",
    "dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct", "current_stage_bars", "prev_stage_bars",
    "bars_since_last_high", "bars_since_last_low", "prev_stage_amp_pct",
    "current_stage_ret_pct", "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct", "bbmacd", "bbmacd_minus_avg",
    "bbmacd_band_pos_01", "bbmacd_bandwidth_zscore",
    "bbmacd_cross_upper", "bbmacd_cross_lower",
]

CATEGORICAL_FEATURES = ["bbmacd_state", "trend_align_momo"]

DERIVED_FEATURES = [
    "high_low_range", "high_low_range_pct", "vwap_dev_x_bbmacd",
    "pivot_pos_x_trend", "stage_bars_ratio", "amp_x_pullback", "bbmacd_band_width",
]

ALL_FEATURES = RAW_FEATURES + CATEGORICAL_FEATURES + DERIVED_FEATURES

FACTOR_CATEGORIES = {
    "趋势类": {
        "factors": ["dsa_dir", "trend_align_momo"],
        "intuition": "趋势方向=1时资金效率高，趋势-动量对齐=1时每bar收益高",
    },
    "位置类": {
        "factors": ["dsa_pivot_pos_01", "bbmacd_band_pos_01", "prev_pivot_code"],
        "intuition": "位置越低(接近0)→上涨空间越大→收益越高",
    },
    "偏离类": {
        "factors": [
            "ret_to_last_high_pct", "ret_to_last_low_pct",
            "price_vs_dsa_vwap_pct", "current_pullback_from_stage_extreme_pct",
        ],
        "intuition": "离高点远/离低点近→安全边际大→收益高；VWAP偏离正→动量强",
    },
    "时长类": {
        "factors": [
            "current_stage_bars", "prev_stage_bars",
            "bars_since_last_high", "bars_since_last_low", "stage_bars_ratio",
        ],
        "intuition": "距低点越近→刚启动→每bar收益高；距高点越远→空间大",
    },
    "振幅类": {
        "factors": [
            "current_stage_amp_pct", "prev_stage_amp_pct",
            "current_stage_ret_pct", "high_low_range", "high_low_range_pct",
        ],
        "intuition": "振幅大→波动大→机会多→总收益高但可能每bar收益低",
    },
    "动量类": {
        "factors": [
            "bbmacd", "bbmacd_minus_avg", "bbmacd_state",
            "bbmacd_bandwidth_zscore", "bbmacd_cross_upper", "bbmacd_cross_lower",
        ],
        "intuition": "bbmacd正值→动量强→收益高；但SHAP显示过大反而反向",
    },
    "交互类": {
        "factors": ["vwap_dev_x_bbmacd", "pivot_pos_x_trend", "amp_x_pullback", "bbmacd_band_width"],
        "intuition": "因子交互效应",
    },
}

REG_TARGETS = [
    "high_ret_per_bar",
    "low_ret_per_bar",
    "next_reversal_high_ret",
    "interim_low_ret",
]

CLS_TARGETS = ["is_high_efficiency", "is_low_loss_per_bar", "is_profitable", "is_low_drawdown"]

FACTOR_LABELS = {
    "dsa_dir": "DSA方向", "prev_pivot_code": "前一枢轴编码",
    "last_confirmed_high": "最近确认高点", "last_confirmed_low": "最近确认低点",
    "dsa_pivot_pos_01": "DSA枢轴位置[0,1]", "ret_to_last_high_pct": "相对高点收益率",
    "ret_to_last_low_pct": "相对低点收益率", "price_vs_dsa_vwap_pct": "VWAP偏离度",
    "current_stage_bars": "当前阶段bar数", "prev_stage_bars": "前一阶段bar数",
    "bars_since_last_high": "距高点bar数", "bars_since_last_low": "距低点bar数",
    "prev_stage_amp_pct": "前一阶段振幅%", "current_stage_ret_pct": "当前阶段收益率%",
    "current_stage_amp_pct": "当前阶段振幅%", "current_pullback_from_stage_extreme_pct": "当前回撤%",
    "bbmacd": "BBMacd值", "bbmacd_minus_avg": "BBMacd减均值",
    "bbmacd_state": "BBMacd状态", "bbmacd_band_pos_01": "BBMacd带内位置[0,1]",
    "bbmacd_bandwidth_zscore": "BBMacd带宽Z-Score", "bbmacd_cross_upper": "上穿上轨信号",
    "bbmacd_cross_lower": "下穿下轨信号", "trend_align_momo": "趋势-动量对齐",
    "high_low_range": "确认高低点范围", "high_low_range_pct": "确认高低点振幅%",
    "vwap_dev_x_bbmacd": "VWAP偏离×动量", "pivot_pos_x_trend": "位置×趋势对齐",
    "stage_bars_ratio": "阶段时长比", "amp_x_pullback": "振幅×回撤",
    "bbmacd_band_width": "带内位置×带宽",
}

TARGET_LABELS = {
    "high_ret_per_bar": "每bar高点收益率(资金效率)",
    "low_ret_per_bar": "每bar低点收益率(亏损效率)",
    "next_reversal_high_ret": "反转高点收益率",
    "interim_low_ret": "中间低点收益率(回撤)",
    "is_high_efficiency": "高资金效率",
    "is_low_loss_per_bar": "低亏损效率",
    "is_profitable": "盈利>10%",
    "is_low_drawdown": "回撤<10%",
}

INTUITION_RULES = {
    "dsa_dir": {"direction": "positive", "reason": "方向=1(上涨)→每bar收益高"},
    "trend_align_momo": {"direction": "positive", "reason": "趋势-动量对齐=1→资金效率高"},
    "dsa_pivot_pos_01": {"direction": "negative", "reason": "位置低→上涨空间大→收益高"},
    "bbmacd_band_pos_01": {"direction": "nonlinear", "reason": "带内位置过高可能过热"},
    "prev_pivot_code": {"direction": "positive", "reason": "枢轴编码大(HH/HL)→趋势强"},
    "ret_to_last_high_pct": {"direction": "negative", "reason": "离高点远→空间大→收益高"},
    "ret_to_last_low_pct": {"direction": "negative", "reason": "离低点近→安全边际大"},
    "price_vs_dsa_vwap_pct": {"direction": "positive", "reason": "VWAP偏离正→动量强"},
    "current_pullback_from_stage_extreme_pct": {"direction": "nonlinear", "reason": "回撤适中最好"},
    "current_stage_bars": {"direction": "nonlinear", "reason": "阶段初期效率高，过长效率低"},
    "prev_stage_bars": {"direction": "nonlinear", "reason": "前阶段时长影响当前节奏"},
    "bars_since_last_high": {"direction": "positive", "reason": "距高点远→空间大"},
    "bars_since_last_low": {"direction": "negative", "reason": "距低点近→刚启动→效率高"},
    "stage_bars_ratio": {"direction": "nonlinear", "reason": "阶段时长比影响节奏"},
    "current_stage_amp_pct": {"direction": "positive_for_total_negative_for_per_bar", "reason": "振幅大→总收益高但时间长"},
    "prev_stage_amp_pct": {"direction": "nonlinear", "reason": "前阶段振幅影响当前"},
    "current_stage_ret_pct": {"direction": "negative", "reason": "当前涨幅大→可能透支"},
    "high_low_range": {"direction": "positive", "reason": "高低点范围大→波动大→机会多"},
    "high_low_range_pct": {"direction": "positive", "reason": "高低点振幅大→机会多"},
    "bbmacd": {"direction": "nonlinear", "reason": "SHAP发现过大反而反向"},
    "bbmacd_minus_avg": {"direction": "nonlinear", "reason": "偏离均值过大可能过热"},
    "bbmacd_state": {"direction": "positive", "reason": "带上(1)>带内(0)>带下(-1)"},
    "bbmacd_bandwidth_zscore": {"direction": "positive", "reason": "带宽扩张→波动率上升→机会多"},
    "bbmacd_cross_upper": {"direction": "positive", "reason": "突破上轨→强势信号"},
    "bbmacd_cross_lower": {"direction": "negative", "reason": "跌破下轨→弱势信号"},
}

INTUITION_RULES_FOR_LOSS = {
    "dsa_dir": {"direction": "positive", "reason": "方向=1(上涨)→下跌概率低→亏损小(值接近0)"},
    "trend_align_momo": {"direction": "positive", "reason": "趋势-动量对齐→下跌概率低→亏损小"},
    "dsa_pivot_pos_01": {"direction": "negative", "reason": "位置低→离支撑近→下跌空间小→亏损小(值接近0)"},
    "bbmacd_band_pos_01": {"direction": "nonlinear", "reason": "位置过高→过热→回调风险大→亏损大"},
    "prev_pivot_code": {"direction": "positive", "reason": "枢轴编码大(HH/HL)→趋势强→亏损小"},
    "ret_to_last_high_pct": {"direction": "nonlinear", "reason": "离高点远→可能触底也可能继续跌"},
    "ret_to_last_low_pct": {"direction": "negative", "reason": "离低点近→安全边际大→亏损小(值接近0)"},
    "price_vs_dsa_vwap_pct": {"direction": "positive", "reason": "VWAP偏离正→动量强→亏损小"},
    "current_pullback_from_stage_extreme_pct": {"direction": "nonlinear", "reason": "回撤过大→亏损大；回撤适中→亏损小"},
    "current_stage_bars": {"direction": "nonlinear", "reason": "阶段时长影响亏损节奏"},
    "prev_stage_bars": {"direction": "nonlinear", "reason": "前阶段时长影响当前亏损"},
    "bars_since_last_high": {"direction": "nonlinear", "reason": "距高点远→可能触底→亏损小"},
    "bars_since_last_low": {"direction": "negative", "reason": "距低点近→刚启动→亏损小(值接近0)"},
    "stage_bars_ratio": {"direction": "nonlinear", "reason": "阶段时长比影响亏损"},
    "current_stage_amp_pct": {"direction": "negative", "reason": "振幅大→波动大→亏损大(值远离0)"},
    "prev_stage_amp_pct": {"direction": "nonlinear", "reason": "前阶段振幅影响当前亏损"},
    "current_stage_ret_pct": {"direction": "positive", "reason": "当前涨幅大→趋势强→亏损小(值接近0)"},
    "high_low_range": {"direction": "negative", "reason": "高低点范围大→波动大→亏损大(值远离0)"},
    "high_low_range_pct": {"direction": "negative", "reason": "高低点振幅大→波动大→亏损大(值远离0)"},
    "bbmacd": {"direction": "nonlinear", "reason": "bbmacd过大→过热→回调风险大→亏损大"},
    "bbmacd_minus_avg": {"direction": "nonlinear", "reason": "偏离均值过大→过热→亏损大"},
    "bbmacd_state": {"direction": "positive", "reason": "带上→强势→亏损小(值接近0)"},
    "bbmacd_bandwidth_zscore": {"direction": "negative", "reason": "带宽扩张→波动大→亏损大(值远离0)"},
    "bbmacd_cross_upper": {"direction": "positive", "reason": "突破上轨→强势→亏损小"},
    "bbmacd_cross_lower": {"direction": "negative", "reason": "跌破下轨→弱势→亏损大(值远离0)"},
}

INTUITION_RULES_BY_TARGET = {
    "high_ret_per_bar": INTUITION_RULES,
    "next_reversal_high_ret": INTUITION_RULES,
    "low_ret_per_bar": INTUITION_RULES_FOR_LOSS,
    "interim_low_ret": INTUITION_RULES_FOR_LOSS,
}

REG_DEFAULT_PARAMS = {
    "objective": "regression", "metric": "l1",
    "learning_rate": 0.03, "num_leaves": 15, "max_depth": 4,
    "min_data_in_leaf": 50, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l1": 1.0, "lambda_l2": 2.0,
    "verbosity": -1, "seed": 42,
}

CLF_DEFAULT_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.03, "num_leaves": 15, "max_depth": 4,
    "min_data_in_leaf": 50, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l1": 1.0, "lambda_l2": 2.0,
    "verbosity": -1, "seed": 42,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
EMBARGO_DAYS = 175
MIN_TRAIN_SIZE = 200


def fetch_vreversal_records(start_date, end_date, sample_limit=None, only_reversal=False):
    where = "WHERE selection_date BETWEEN :start_date AND :end_date"
    if only_reversal:
        where += " AND bars_to_reversal IS NOT NULL"
    sql = text(f"SELECT * FROM stock_dsa_vreversal_results {where} ORDER BY selection_date, ts_code")
    df = pd.read_sql(sql, engine, params={
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    })
    if df.empty:
        return df
    if sample_limit and len(df) > sample_limit:
        n_dates = df["selection_date"].nunique()
        per_date = max(1, sample_limit // max(n_dates, 1))
        df = df.groupby("selection_date", group_keys=False).head(per_date).reset_index(drop=True)
    return df


def build_derived_targets(df):
    df = df.copy()
    df["high_ret_per_bar"] = np.where(
        (df["bars_to_reversal_high"].notna()) & (df["bars_to_reversal_high"] > 0),
        df["next_reversal_high_ret"] / df["bars_to_reversal_high"], np.nan,
    )
    df["low_ret_per_bar"] = np.where(
        (df["bars_to_interim_low"].notna()) & (df["bars_to_interim_low"] > 0),
        df["interim_low_ret"] / df["bars_to_interim_low"], np.nan,
    )
    return df


def build_labels_for_fold(df, fold):
    df = df.copy()
    train_idx = fold.train_idx
    train_hrp = df.loc[train_idx, "high_ret_per_bar"].dropna()
    median_hrp = train_hrp.median() if len(train_hrp) > 10 else df["high_ret_per_bar"].median()
    train_lrp = df.loc[train_idx, "low_ret_per_bar"].dropna()
    median_abs_lrp = train_lrp.abs().median() if len(train_lrp) > 10 else df["low_ret_per_bar"].abs().median()
    df["is_high_efficiency"] = np.where(
        df["high_ret_per_bar"].notna(),
        (df["high_ret_per_bar"] > median_hrp).astype(float), np.nan,
    )
    df["is_low_loss_per_bar"] = np.where(
        df["low_ret_per_bar"].notna(),
        (df["low_ret_per_bar"].abs() < median_abs_lrp).astype(float), np.nan,
    )
    df["is_profitable"] = np.where(
        df["next_reversal_high_ret"].notna(),
        (df["next_reversal_high_ret"] > 0.1).astype(float), np.nan,
    )
    df["is_low_drawdown"] = np.where(
        df["interim_low_ret"].notna(),
        (df["interim_low_ret"].abs() < 0.1).astype(float), np.nan,
    )
    return df


def build_features(df):
    df = df.copy()
    high = df["last_confirmed_high"].fillna(0)
    low = df["last_confirmed_low"].fillna(0)
    df["high_low_range"] = high - low
    df["high_low_range_pct"] = np.where(low > 0, (high - low) / low, 0)
    df["vwap_dev_x_bbmacd"] = df["price_vs_dsa_vwap_pct"].fillna(0) * df["bbmacd"].fillna(0)
    df["pivot_pos_x_trend"] = df["dsa_pivot_pos_01"].fillna(0) * df["trend_align_momo"].fillna(0)
    prev_bars = df["prev_stage_bars"].fillna(0).replace(0, np.nan)
    df["stage_bars_ratio"] = df["current_stage_bars"].fillna(0) / prev_bars
    df["stage_bars_ratio"] = df["stage_bars_ratio"].fillna(0)
    df["amp_x_pullback"] = df["current_stage_amp_pct"].fillna(0) * df["current_pullback_from_stage_extreme_pct"].fillna(0)
    df["bbmacd_band_width"] = df["bbmacd_band_pos_01"].fillna(0) * df["bbmacd_bandwidth_zscore"].fillna(0)
    return df


def get_feature_cols(df):
    valid = []
    for col in ALL_FEATURES:
        if col not in df.columns:
            continue
        if df[col].isna().all():
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        if df[col].isna().mean() > 0.6:
            continue
        valid.append(col)
    return valid


@dataclass
class FoldData:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    name: str
    train_end: str
    valid_end: str
    test_end: str


def build_rolling_splits(df, n_folds, embargo_days=EMBARGO_DAYS):
    df = df.copy()
    df["selection_date"] = pd.to_datetime(df["selection_date"])
    df["ym"] = df["selection_date"].dt.to_period("M")
    months = sorted(df["ym"].unique())
    n_months = len(months)
    if n_months < n_folds + 1:
        return []
    test_months_per_fold = max(1, n_months // (n_folds + 1))
    valid_months_count = max(1, test_months_per_fold)
    folds = []
    for i in range(n_folds):
        test_start_m = n_months - (n_folds - i) * test_months_per_fold
        if test_start_m < 2:
            continue
        test_end_m = min(test_start_m + test_months_per_fold, n_months)
        valid_start_m = max(0, test_start_m - valid_months_count)
        valid_end_m = test_start_m
        train_months_set = set(months[:valid_start_m])
        valid_months_set = set(months[valid_start_m:valid_end_m])
        test_months_set = set(months[test_start_m:test_end_m])
        train_idx = df.index[df["ym"].isin(train_months_set)].to_numpy()
        valid_idx = df.index[df["ym"].isin(valid_months_set)].to_numpy()
        test_idx = df.index[df["ym"].isin(test_months_set)].to_numpy()
        if len(train_idx) < MIN_TRAIN_SIZE:
            continue
        train_end_date = df.loc[train_idx, "selection_date"].max()
        valid_start_date = df.loc[valid_idx, "selection_date"].min() if len(valid_idx) > 0 else train_end_date
        valid_end_date = df.loc[valid_idx, "selection_date"].max() if len(valid_idx) > 0 else train_end_date
        test_end_date = df.loc[test_idx, "selection_date"].max() if len(test_idx) > 0 else valid_end_date
        embargo_cutoff = valid_start_date - pd.Timedelta(days=embargo_days)
        train_idx = df.index[(df["ym"].isin(train_months_set)) & (df["selection_date"] <= embargo_cutoff)].to_numpy()
        if len(train_idx) < MIN_TRAIN_SIZE:
            train_idx = df.index[df["ym"].isin(train_months_set)].to_numpy()
            if len(train_idx) < MIN_TRAIN_SIZE:
                continue
        folds.append(FoldData(
            train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx,
            name=f"fold_{i + 1}",
            train_end=str(train_end_date)[:10],
            valid_end=str(valid_end_date)[:10],
            test_end=str(test_end_date)[:10],
        ))
    return folds


def train_lightgbm(df, feature_cols, target_col, fold, params, task="regression"):
    X_train = df.loc[fold.train_idx, feature_cols]
    y_train = df.loc[fold.train_idx, target_col]
    X_valid = df.loc[fold.valid_idx, feature_cols]
    y_valid = df.loc[fold.valid_idx, target_col]
    X_test = df.loc[fold.test_idx, feature_cols]
    y_test = df.loc[fold.test_idx, target_col]
    train_mask = y_train.notna()
    valid_mask = y_valid.notna()
    test_mask = y_test.notna()
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_valid, y_valid = X_valid[valid_mask], y_valid[valid_mask]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, categorical_feature=cat_cols)
    model = lgb.train(
        params, dtrain, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=0)],
    )
    y_pred_test = model.predict(X_test[test_mask], num_iteration=model.best_iteration)
    y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
    y_true_test = y_test[test_mask]
    y_true_train = y_train
    metrics = compute_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test, task)
    metrics.update({"fold": fold.name, "train_end": fold.train_end, "test_end": fold.test_end,
                     "best_iteration": model.best_iteration, "n_train": len(y_train), "n_test": int(test_mask.sum())})
    return model, metrics


def compute_metrics(y_train, y_pred_train, y_test, y_pred_test, task):
    result = {}
    if task == "regression":
        result["train_mae"] = mean_absolute_error(y_train, y_pred_train)
        result["train_rmse"] = np.sqrt(mean_squared_error(y_train, y_pred_train))
        ss_res_test = np.sum((y_test - y_pred_test) ** 2)
        ss_tot_test = np.sum((y_test - y_test.mean()) ** 2)
        result["test_r2"] = 1 - ss_res_test / ss_tot_test if ss_tot_test > 0 else np.nan
        ss_res_train = np.sum((y_train - y_pred_train) ** 2)
        ss_tot_train = np.sum((y_train - y_train.mean()) ** 2)
        result["train_r2"] = 1 - ss_res_train / ss_tot_train if ss_tot_train > 0 else np.nan
        result["r2_gap"] = result["train_r2"] - result["test_r2"]
        result["test_mae"] = mean_absolute_error(y_test, y_pred_test)
        result["test_rmse"] = np.sqrt(mean_squared_error(y_test, y_pred_test))
        if len(y_test) >= 10:
            result["test_ic_spearman"], _ = stats.spearmanr(y_test, y_pred_test)
            result["test_ic_pearson"], _ = stats.pearsonr(y_test, y_pred_test)
        else:
            result["test_ic_spearman"] = np.nan
            result["test_ic_pearson"] = np.nan
    else:
        try:
            result["train_auc"] = roc_auc_score(y_train, y_pred_train)
        except ValueError:
            result["train_auc"] = np.nan
        try:
            result["test_auc"] = roc_auc_score(y_test, y_pred_test)
        except ValueError:
            result["test_auc"] = np.nan
        try:
            result["test_ap"] = average_precision_score(y_test, y_pred_test)
        except ValueError:
            result["test_ap"] = np.nan
        try:
            result["test_logloss"] = log_loss(y_test, y_pred_test)
        except ValueError:
            result["test_logloss"] = np.nan
        pred_pos = y_pred_test > 0.5
        result["test_win_rate"] = float(y_test[pred_pos].mean()) if pred_pos.sum() > 0 else np.nan
        result["test_pos_ratio"] = float(pred_pos.mean())
    return result


def build_decile_report(y_true, y_pred, task="regression", dates=None):
    df = pd.DataFrame({"y_true": y_true.values, "y_pred": y_pred})
    if dates is not None:
        df["date"] = dates.values
    try:
        df["decile"] = pd.qcut(df["y_pred"], q=10, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    if dates is not None and "date" in df.columns and df["date"].notna().sum() > 0:
        daily_groups = []
        for dt, grp in df.groupby("date"):
            if len(grp) < 10:
                continue
            try:
                grp_decile = pd.qcut(grp["y_pred"], q=10, labels=False, duplicates="drop")
            except ValueError:
                continue
            grp = grp.copy()
            grp["decile"] = grp_decile
            daily_groups.append(grp)
        if not daily_groups:
            return pd.DataFrame()
        combined = pd.concat(daily_groups)
        rows = []
        for d in sorted(combined["decile"].unique()):
            sub = combined[combined["decile"] == d]
            row = {"decile": int(d) + 1, "n": len(sub), "mean_pred": sub["y_pred"].mean(), "mean_actual": sub["y_true"].mean()}
            if task == "regression":
                row["win_rate"] = (sub["y_true"] > 0).mean()
            rows.append(row)
    else:
        rows = []
        for d in sorted(df["decile"].unique()):
            sub = df[df["decile"] == d]
            row = {"decile": int(d) + 1, "n": len(sub), "mean_pred": sub["y_pred"].mean(), "mean_actual": sub["y_true"].mean()}
            if task == "regression":
                row["win_rate"] = (sub["y_true"] > 0).mean()
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) >= 2:
        result["long_short"] = result.iloc[-1]["mean_actual"] - result.iloc[0]["mean_actual"]
    return result


def factor_bin_analysis(df, feature_cols, target_cols, n_bins=5):
    rows = []
    for factor in feature_cols:
        if factor not in df.columns:
            continue
        factor_data = df[factor].dropna()
        if len(factor_data) < n_bins * 10:
            continue
        try:
            bins = pd.qcut(factor_data, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        for b in sorted(bins.unique()):
            mask = bins == b
            bin_data = df.loc[factor_data.index[mask]]
            row = {"factor": factor, "factor_label": FACTOR_LABELS.get(factor, factor), "bin": int(b) + 1}
            lower = factor_data[mask].min()
            upper = factor_data[mask].max()
            row["bin_lower"] = lower
            row["bin_upper"] = upper
            row["n"] = len(bin_data)
            for target in target_cols:
                if target not in bin_data.columns:
                    continue
                vals = bin_data[target].dropna()
                row[f"{target}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                row[f"{target}_median"] = vals.median() if len(vals) > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def category_summary(bin_df, target_col):
    if bin_df.empty:
        return pd.DataFrame()
    rules = INTUITION_RULES_BY_TARGET.get(target_col, INTUITION_RULES)
    rows = []
    for cat_name, cat_info in FACTOR_CATEGORIES.items():
        cat_factors = cat_info["factors"]
        cat_bins = bin_df[bin_df["factor"].isin(cat_factors)]
        if cat_bins.empty:
            continue
        col_mean = f"{target_col}_mean"
        if col_mean not in cat_bins.columns:
            continue
        for factor in cat_factors:
            f_bins = cat_bins[cat_bins["factor"] == factor].sort_values("bin")
            if len(f_bins) < 2:
                continue
            low_val = f_bins.iloc[0][col_mean]
            high_val = f_bins.iloc[-1][col_mean]
            diff = high_val - low_val if not np.isnan(low_val) and not np.isnan(high_val) else np.nan
            rule = rules.get(factor, {})
            expected_dir = rule.get("direction", "unknown")
            if expected_dir == "positive":
                match = diff > 0
            elif expected_dir == "negative":
                match = diff < 0
            else:
                match = None
            rows.append({
                "category": cat_name,
                "factor": factor,
                "factor_label": FACTOR_LABELS.get(factor, factor),
                "target": target_col,
                "low_bin_mean": low_val,
                "high_bin_mean": high_val,
                "diff": diff,
                "expected_direction": expected_dir,
                "match_intuition": match,
                "intuition_reason": rule.get("reason", ""),
            })
    return pd.DataFrame(rows)


def model_prediction_by_bin(df, feature_cols, target_col, model, n_bins=5):
    X = df[feature_cols].fillna(0)
    y_pred = model.predict(X, num_iteration=model.best_iteration)
    result_rows = []
    for factor in feature_cols:
        if factor not in df.columns:
            continue
        factor_data = df[factor].dropna()
        if len(factor_data) < n_bins * 10:
            continue
        try:
            bins = pd.qcut(factor_data, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        for b in sorted(bins.unique()):
            mask = bins == b
            idx = factor_data.index[mask]
            actual_vals = df.loc[idx, target_col].dropna()
            pred_vals = pd.Series(y_pred, index=df.index).loc[idx]
            pred_vals = pred_vals[~pred_vals.isna()]
            result_rows.append({
                "factor": factor, "factor_label": FACTOR_LABELS.get(factor, factor),
                "bin": int(b) + 1,
                "bin_lower": factor_data[mask].min(),
                "bin_upper": factor_data[mask].max(),
                "n": len(idx),
                "actual_mean": actual_vals.mean() if len(actual_vals) > 0 else np.nan,
                "predicted_mean": pred_vals.mean() if len(pred_vals) > 0 else np.nan,
            })
    return pd.DataFrame(result_rows)


def shap_analysis(model, X, feature_cols):
    import shap
    explainer = shap.TreeExplainer(model)
    X_sample = X[feature_cols].head(min(500, len(X)))
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    result = pd.DataFrame({
        "feature": feature_cols,
        "feature_label": [FACTOR_LABELS.get(f, f) for f in feature_cols],
        "shap_mean_abs": mean_abs_shap,
        "shap_mean": mean_shap,
    }).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
    result["shap_rank"] = range(1, len(result) + 1)
    result["direction"] = result["shap_mean"].apply(
        lambda x: "正向" if x > 0 else ("反向" if x < 0 else "中性")
    )
    return result


def print_factor_bin_report(bin_df, target_col):
    if bin_df.empty:
        return
    rules = INTUITION_RULES_BY_TARGET.get(target_col, INTUITION_RULES)
    print(f"\n{'=' * 100}")
    print(f"因子区间效应分析 — {TARGET_LABELS.get(target_col, target_col)} ({target_col})")
    print(f"{'=' * 100}")
    col_mean = f"{target_col}_mean"
    if col_mean not in bin_df.columns:
        return
    for factor in bin_df["factor"].unique():
        f_bins = bin_df[bin_df["factor"] == factor].sort_values("bin")
        if len(f_bins) < 2:
            continue
        label = FACTOR_LABELS.get(factor, factor)
        rule = rules.get(factor, {})
        expected = rule.get("direction", "?")
        reason = rule.get("reason", "")
        low_val = f_bins.iloc[0][col_mean]
        high_val = f_bins.iloc[-1][col_mean]
        diff = high_val - low_val
        if expected == "positive":
            match_str = "✅" if diff > 0 else "❌"
        elif expected == "negative":
            match_str = "✅" if diff < 0 else "❌"
        else:
            match_str = "🔄"
        print(f"\n  {label} ({factor}) [预期:{expected}] {match_str}")
        print(f"    直觉: {reason}")
        print(f"    {'区间':>6} {'下界':>10} {'上界':>10} {'N':>6} {'均值':>10}")
        print(f"    {'-' * 50}")
        for _, row in f_bins.iterrows():
            print(f"    {row['bin']:>6} {row['bin_lower']:>10.4f} {row['bin_upper']:>10.4f} {row['n']:>6} {row[col_mean]:>10.4f}")
        print(f"    区间1→区间N 差异: {diff:+.4f} {match_str}")


def print_category_report(cat_df):
    if cat_df.empty:
        return
    print(f"\n{'=' * 100}")
    print(f"因子类别汇总")
    print(f"{'=' * 100}")
    for cat_name in cat_df["category"].unique():
        cat_rows = cat_df[cat_df["category"] == cat_name]
        print(f"\n  [{cat_name}]")
        for _, row in cat_rows.iterrows():
            mi = row["match_intuition"]
            if pd.isna(mi):
                match_str = "🔄非线性/待验证"
            elif bool(mi):
                match_str = "✅符合直觉"
            else:
                match_str = "🔄非线性/待验证"
            diff_str = f"{row['diff']:+.4f}" if not np.isnan(row.get('diff', np.nan)) else "N/A"
            print(f"    {row['factor_label']:<20} 低区间→高区间: {diff_str} (预期:{row['expected_direction']}) {match_str}")
            print(f"      {row['intuition_reason']}")


def print_model_report(all_metrics, task, target_col):
    if not all_metrics:
        return
    task_label = "回归" if task == "regression" else "分类"
    target_label = TARGET_LABELS.get(target_col, target_col)
    print(f"\n{'=' * 80}")
    print(f"GBDT {task_label} — {target_label} ({target_col})")
    print(f"{'=' * 80}")
    metrics_df = pd.DataFrame(all_metrics)
    for fold_name in metrics_df["fold"].unique():
        fd = metrics_df[metrics_df["fold"] == fold_name].iloc[0]
        print(f"\n  {fold_name} (train~{fd['train_end']}, test~{fd['test_end']})")
        print(f"    n_train={fd.get('n_train', '?')}, n_test={fd.get('n_test', '?')}")
        if task == "regression":
            print(f"    Train R²={fd.get('train_r2', np.nan):.4f}, Test R²={fd.get('test_r2', np.nan):.4f}, Gap={fd.get('r2_gap', np.nan):.4f}")
            print(f"    Test MAE={fd.get('test_mae', np.nan):.4f}, RMSE={fd.get('test_rmse', np.nan):.4f}")
            print(f"    Test IC(Spearman)={fd.get('test_ic_spearman', np.nan):.4f}")
            if fd.get('r2_gap', 0) > 0.3:
                print(f"    ⚠️ R² Gap > 0.3，可能过拟合！")
        else:
            print(f"    Train AUC={fd.get('train_auc', np.nan):.4f}, Test AUC={fd.get('test_auc', np.nan):.4f}")
            print(f"    Test AP={fd.get('test_ap', np.nan):.4f}")
    print(f"\n  跨折汇总:")
    if task == "regression":
        for col in ["test_r2", "test_ic_spearman", "r2_gap"]:
            if col in metrics_df.columns:
                vals = metrics_df[col].dropna()
                if len(vals) > 0:
                    print(f"    {col}: {vals.mean():.4f} ± {vals.std():.4f}")
    else:
        for col in ["test_auc", "test_ap"]:
            if col in metrics_df.columns:
                vals = metrics_df[col].dropna()
                if len(vals) > 0:
                    print(f"    {col}: {vals.mean():.4f} ± {vals.std():.4f}")


def save_results(output_dir, all_metrics=None, decile_reports=None, shap_df=None,
                 fi_df=None, best_params=None, model=None, bin_df=None, cat_df=None, pred_bin_df=None):
    os.makedirs(output_dir, exist_ok=True)
    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, "fold_metrics.csv"), index=False, encoding="utf-8-sig")
    if decile_reports:
        all_d = []
        for key, report in decile_reports.items():
            if not report.empty:
                report = report.copy()
                report["label"] = key
                all_d.append(report)
        if all_d:
            pd.concat(all_d).to_csv(os.path.join(output_dir, "decile_report.csv"), index=False, encoding="utf-8-sig")
    if shap_df is not None and not shap_df.empty:
        shap_df.to_csv(os.path.join(output_dir, "shap.csv"), index=False, encoding="utf-8-sig")
    if fi_df is not None and not fi_df.empty:
        fi_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False, encoding="utf-8-sig")
    if best_params:
        with open(os.path.join(output_dir, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=2, default=str)
    if model is not None:
        model.save_model(os.path.join(output_dir, "model.txt"))
    if bin_df is not None and not bin_df.empty:
        bin_df.to_csv(os.path.join(output_dir, "factor_bin_analysis.csv"), index=False, encoding="utf-8-sig")
    if cat_df is not None and not cat_df.empty:
        cat_df.to_csv(os.path.join(output_dir, "category_summary.csv"), index=False, encoding="utf-8-sig")
    if pred_bin_df is not None and not pred_bin_df.empty:
        pred_bin_df.to_csv(os.path.join(output_dir, "model_prediction_by_bin.csv"), index=False, encoding="utf-8-sig")
    print(f"  结果保存: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DSA V型反转 GBDT 因子组合探索（重设计版）")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--n-bins", type=int, default=5, help="因子区间数")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--only-reversal", action="store_true")
    parser.add_argument("--output-dir", type=str, default="/tmp/dsa_vreversal_gbdt")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else date(2021, 5, 1)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()

    print("=" * 80)
    print("DSA V型反转 GBDT 因子组合探索（重设计版）")
    print("=" * 80)
    print(f"  日期: {start_date} ~ {end_date}, 折数: {args.n_folds}, 区间数: {args.n_bins}")

    print("\n[1/5] 获取数据...")
    df = fetch_vreversal_records(start_date, end_date, args.sample_limit, args.only_reversal)
    if df.empty:
        print("记录为空，退出")
        return
    print(f"  {len(df)} 条记录")

    print("[2/5] 构建目标变量和特征...")
    df = build_derived_targets(df)
    df = build_features(df)
    feature_cols = get_feature_cols(df)
    print(f"  特征: {len(feature_cols)} 个")

    print("[3/5] 构建 Rolling 切分...")
    df = df.sort_values("selection_date").reset_index(drop=True)
    folds = build_rolling_splits(df, args.n_folds)
    if not folds:
        print("无法构建有效折，退出")
        return
    print(f"  {len(folds)} 折")
    for fold in folds:
        print(f"    {fold.name}: train~{fold.train_end}, test~{fold.test_end}")

    # 因子区间效应分析（全样本，用全样本中位数做分类标签仅用于展示）
    df_display = df.copy()
    median_hrp_display = df_display["high_ret_per_bar"].median()
    median_abs_lrp_display = df_display["low_ret_per_bar"].abs().median()
    df_display["is_high_efficiency"] = np.where(
        df_display["high_ret_per_bar"].notna(),
        (df_display["high_ret_per_bar"] > median_hrp_display).astype(float), np.nan,
    )
    df_display["is_low_loss_per_bar"] = np.where(
        df_display["low_ret_per_bar"].notna(),
        (df_display["low_ret_per_bar"].abs() < median_abs_lrp_display).astype(float), np.nan,
    )
    df_display["is_profitable"] = np.where(
        df_display["next_reversal_high_ret"].notna(),
        (df_display["next_reversal_high_ret"] > 0.1).astype(float), np.nan,
    )
    df_display["is_low_drawdown"] = np.where(
        df_display["interim_low_ret"].notna(),
        (df_display["interim_low_ret"].abs() < 0.1).astype(float), np.nan,
    )
    all_target_cols = REG_TARGETS + CLS_TARGETS
    available_targets = [c for c in all_target_cols if c in df_display.columns]

    print("[4/5] 因子区间效应分析...")
    bin_df = factor_bin_analysis(df_display, feature_cols, available_targets, args.n_bins)
    for target in available_targets:
        print_factor_bin_report(bin_df, target)
    cat_summaries = {}
    for target in REG_TARGETS:
        if target in df.columns:
            cat_summaries[target] = category_summary(bin_df, target)
            print_category_report(cat_summaries[target])

    # GBDT 训练
    print("\n[5/5] GBDT 训练与评估...")
    for task in ["regression", "classification"]:
        target_list = REG_TARGETS if task == "regression" else CLS_TARGETS
        base_params = REG_DEFAULT_PARAMS.copy() if task == "regression" else CLF_DEFAULT_PARAMS.copy()

        for target_col in target_list:
            if target_col not in df_display.columns:
                continue
            valid_count = df_display[target_col].notna().sum()
            if valid_count < 100:
                continue

            all_metrics = []
            decile_reports = {}
            shap_df = None
            fi_df = None
            last_model = None
            best_params = None

            if args.tune and len(folds) > 0:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def objective(trial, _df=df, _fc=feature_cols, _tc=target_col, _folds=folds, _task=task, _bp=base_params):
                    params = _bp.copy()
                    params["num_leaves"] = trial.suggest_categorical("num_leaves", [8, 16, 32])
                    params["max_depth"] = trial.suggest_int("max_depth", 3, 6)
                    params["min_data_in_leaf"] = trial.suggest_categorical("min_data_in_leaf", [30, 50, 80])
                    params["learning_rate"] = trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05])
                    params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.5, 0.9)
                    scores = []
                    for _fold in _folds:
                        try:
                            if _task == "classification":
                                _df_fold = build_labels_for_fold(_df, _fold)
                            else:
                                _df_fold = _df
                            _, m = train_lightgbm(_df_fold, _fc, _tc, _fold, params, _task)
                            score = m.get("test_ic_spearman", 0) if _task == "regression" else m.get("test_auc", 0)
                            scores.append(score)
                        except Exception:
                            scores.append(0)
                    if not scores:
                        return 999 if _task == "regression" else 0
                    return -np.mean(scores) + np.std(scores)

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
                base_params.update(study.best_params)
                best_params = {k: v for k, v in base_params.items() if k not in ("objective", "metric", "verbosity", "seed")}

            for fold in folds:
                try:
                    if task == "classification":
                        df_fold = build_labels_for_fold(df, fold)
                    else:
                        df_fold = df
                    model, metrics = train_lightgbm(df_fold, feature_cols, target_col, fold, base_params, task)
                    all_metrics.append(metrics)
                    last_model = model
                    X_test = df_fold.loc[fold.test_idx, feature_cols]
                    y_test = df_fold.loc[fold.test_idx, target_col]
                    test_mask = y_test.notna()
                    y_pred = model.predict(X_test[test_mask], num_iteration=model.best_iteration)
                    decile_reports[f"{target_col}_{fold.name}"] = build_decile_report(
                        y_test[test_mask], y_pred, task,
                        dates=df_fold.loc[fold.test_idx[test_mask], "selection_date"] if "selection_date" in df_fold.columns else None,
                    )
                    if fi_df is None:
                        gain = model.feature_importance("gain")
                        fi_df = pd.DataFrame({
                            "feature": feature_cols,
                            "feature_label": [FACTOR_LABELS.get(f, f) for f in feature_cols],
                            "importance_gain": gain / gain.sum() if gain.sum() > 0 else gain,
                        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
                except Exception as e:
                    print(f"  {fold.name} 训练失败: {e}")

            if last_model is not None and len(folds) > 0:
                last_fold = folds[-1]
                if task == "classification":
                    df_last_fold = build_labels_for_fold(df, last_fold)
                else:
                    df_last_fold = df
                X_test = df_last_fold.loc[last_fold.test_idx, feature_cols]
                try:
                    shap_df = shap_analysis(last_model, X_test, feature_cols)
                except Exception as e:
                    print(f"  SHAP 失败: {e}")
                pred_bin_df = model_prediction_by_bin(
                    df_last_fold.loc[last_fold.test_idx], feature_cols, target_col, last_model, args.n_bins
                )
            else:
                pred_bin_df = pd.DataFrame()

            print_model_report(all_metrics, task, target_col)

            if fi_df is not None and not fi_df.empty:
                print(f"\n  特征重要性 Top 10:")
                for _, row in fi_df.head(10).iterrows():
                    print(f"    {row['feature_label']:<20} ({row['feature']:<35}) gain={row['importance_gain']:.4f}")

            if shap_df is not None and not shap_df.empty:
                print(f"\n  SHAP Top 10:")
                for _, row in shap_df.head(10).iterrows():
                    print(f"    {row['feature_label']:<20} ({row['feature']:<35}) |SHAP|={row['shap_mean_abs']:.4f} {row['direction']}")

            save_results(
                os.path.join(args.output_dir, target_col),
                all_metrics, decile_reports, shap_df, fi_df,
                best_params, last_model, bin_df,
                cat_summaries.get(target_col), pred_bin_df,
            )

    print(f"\n{'=' * 80}")
    print("分析完成")
    print(f"结果: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
