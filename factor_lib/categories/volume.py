# -*- coding: utf-8 -*-
"""
factor_lib/categories/volume.py - 量能类因子

Purpose: 量能类因子（成交量Z-score/变异系数等）的批量计算与注册。
         基于 features/volume_zscore_plotly.py 的 volume_zscore 权威实现。

Public API:
    compute_volume_factors(df) -> DataFrame

Registered Factors:
    - vol_zscore_5: 5日成交量Z-score
    - vol_zscore_10: 10日成交量Z-score
    - vol_zscore_20: 20日成交量Z-score
    - vol_ratio_10: 10日成交量比率
    - days_since_vol_spike: 距上次放量天数
    - vol_stage_cv: 当前阶段成交量变异系数
    - vol_prev_stage_cv: 前一阶段成交量变异系数
    - vol_cv_ratio: 成交量CV比率
"""
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np


def compute_volume_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    批量计算全部量能类因子。

    Args:
        df: 含 volume 列的 DataFrame，index 为 datetime

    Returns:
        DataFrame 含 8 列量能因子
    """
    from features.volume_zscore_plotly import volume_zscore

    result = pd.DataFrame(index=df.index)
    vol = df["volume"].astype(float)

    z5, _, _ = volume_zscore(vol, 5)
    result["vol_zscore_5"] = z5

    z10, mu10, _ = volume_zscore(vol, 10)
    result["vol_zscore_10"] = z10
    result["vol_ratio_10"] = vol / mu10

    z20, _, _ = volume_zscore(vol, 20)
    result["vol_zscore_20"] = z20

    is_spike = z20 > 2.0
    days_since = np.full(len(df), np.nan)
    last_spike = np.nan
    for i in range(len(df)):
        if is_spike.iloc[i]:
            last_spike = 0
        elif np.isfinite(last_spike):
            last_spike += 1
        days_since[i] = last_spike
    result["days_since_vol_spike"] = pd.Series(days_since, index=df.index)

    mu20 = vol.rolling(20, min_periods=5).mean()
    sd20 = vol.rolling(20, min_periods=5).std(ddof=1)
    cv_curr = sd20 / mu20
    result["vol_stage_cv"] = cv_curr

    mu_prev = vol.shift(20).rolling(20, min_periods=5).mean()
    sd_prev = vol.shift(20).rolling(20, min_periods=5).std(ddof=1)
    cv_prev = sd_prev / mu_prev
    result["vol_prev_stage_cv"] = cv_prev

    result["vol_cv_ratio"] = cv_curr / cv_prev.replace(0.0, np.nan)
    return result


def _compute_vol_zscore_5(df):
    return compute_volume_factors(df)["vol_zscore_5"]


def _compute_vol_zscore_10(df):
    return compute_volume_factors(df)["vol_zscore_10"]


def _compute_vol_zscore_20(df):
    return compute_volume_factors(df)["vol_zscore_20"]


def _compute_vol_ratio_10(df):
    return compute_volume_factors(df)["vol_ratio_10"]


def _compute_days_since_vol_spike(df):
    return compute_volume_factors(df)["days_since_vol_spike"]


def _compute_vol_stage_cv(df):
    return compute_volume_factors(df)["vol_stage_cv"]


def _compute_vol_prev_stage_cv(df):
    return compute_volume_factors(df)["vol_prev_stage_cv"]


def _compute_vol_cv_ratio(df):
    return compute_volume_factors(df)["vol_cv_ratio"]


# 注册量能类因子
register_factor(
    name="vol_zscore_5",
    category="量能类",
    compute_func=_compute_vol_zscore_5,
    source_module="features.volume_zscore_plotly",
    source_function="volume_zscore",
    description="5日成交量Z-score",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="vol_zscore_10",
    category="量能类",
    compute_func=_compute_vol_zscore_10,
    source_module="features.volume_zscore_plotly",
    source_function="volume_zscore",
    description="10日成交量Z-score",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="vol_zscore_20",
    category="量能类",
    compute_func=_compute_vol_zscore_20,
    source_module="features.volume_zscore_plotly",
    source_function="volume_zscore",
    description="20日成交量Z-score",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="vol_ratio_10",
    category="量能类",
    compute_func=_compute_vol_ratio_10,
    source_module="features.volume_zscore_plotly",
    source_function="volume_zscore",
    description="10日成交量比率",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="days_since_vol_spike",
    category="量能类",
    compute_func=_compute_days_since_vol_spike,
    source_module="features.volume_zscore_plotly",
    source_function="volume_zscore",
    description="距上次放量天数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="vol_stage_cv",
    category="量能类",
    compute_func=_compute_vol_stage_cv,
    source_module="factor_lib.categories.volume",
    source_function="_compute_vol_stage_cv",
    description="当前阶段成交量变异系数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="vol_prev_stage_cv",
    category="量能类",
    compute_func=_compute_vol_prev_stage_cv,
    source_module="factor_lib.categories.volume",
    source_function="_compute_vol_prev_stage_cv",
    description="前一阶段成交量变异系数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="vol_cv_ratio",
    category="量能类",
    compute_func=_compute_vol_cv_ratio,
    source_module="factor_lib.categories.volume",
    source_function="_compute_vol_cv_ratio",
    description="成交量CV比率",
    direction="neutral",
    is_core=False,
)
