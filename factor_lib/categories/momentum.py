# -*- coding: utf-8 -*-
"""
factor_lib/categories/momentum.py - 动量类因子

Purpose: 动量类因子（BBMACD 及其衍生）的批量计算与注册。
         基于 features/dsa_bbmacd_24factors_viewer.py 的 compute_bbmacd 权威实现。

Public API:
    compute_momentum_factors(df, bb_result=None) -> DataFrame

Registered Factors:
    - bbmacd: BBMACD柱状线
    - bbmacd_minus_avg: BBMACD距均值偏差
    - bbmacd_state: BBMACD状态
    - bbmacd_band_pos_01: BBMACD在布林带中的位置
    - bbmacd_bandwidth_zscore: BBMACD带宽Z-score
    - bbmacd_cross_upper: 上穿布林带上轨
    - bbmacd_cross_lower: 下穿布林带下轨
    - bbmacd_sign: BBMACD符号
    - bbmacd_slope_3: BBMACD 3日斜率
"""
from factor_lib.registry import register_factor
import pandas as pd


def compute_momentum_factors(df: pd.DataFrame, bb_result: pd.DataFrame = None) -> pd.DataFrame:
    """
    批量计算全部动量类因子。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame
        bb_result: 预计算的 compute_bbmacd 结果，若为 None 则内部计算

    Returns:
        DataFrame 含 9 列动量因子
    """
    from features.dsa_bbmacd_24factors_viewer import compute_bbmacd

    if bb_result is None:
        bb_result = compute_bbmacd(df)

    result = pd.DataFrame(index=df.index)
    result["bbmacd"] = bb_result["bbmacd"]
    result["bbmacd_minus_avg"] = bb_result["bbmacd_minus_avg"]
    result["bbmacd_state"] = bb_result["bbmacd_state"]
    result["bbmacd_band_pos_01"] = bb_result["bbmacd_band_pos_01"]
    result["bbmacd_bandwidth_zscore"] = bb_result["bbmacd_bandwidth_zscore"]
    result["bbmacd_cross_upper"] = bb_result["bbmacd_cross_upper"]
    result["bbmacd_cross_lower"] = bb_result["bbmacd_cross_lower"]

    bbmacd_series = bb_result["bbmacd"]
    result["bbmacd_sign"] = bbmacd_series.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    result["bbmacd_slope_3"] = bbmacd_series.diff(3)
    return result


def _compute_bbmacd(df):
    return compute_momentum_factors(df)["bbmacd"]


def _compute_bbmacd_minus_avg(df):
    return compute_momentum_factors(df)["bbmacd_minus_avg"]


def _compute_bbmacd_state(df):
    return compute_momentum_factors(df)["bbmacd_state"]


def _compute_bbmacd_band_pos_01(df):
    return compute_momentum_factors(df)["bbmacd_band_pos_01"]


def _compute_bbmacd_bandwidth_zscore(df):
    return compute_momentum_factors(df)["bbmacd_bandwidth_zscore"]


def _compute_bbmacd_cross_upper(df):
    return compute_momentum_factors(df)["bbmacd_cross_upper"]


def _compute_bbmacd_cross_lower(df):
    return compute_momentum_factors(df)["bbmacd_cross_lower"]


def _compute_bbmacd_sign(df):
    return compute_momentum_factors(df)["bbmacd_sign"]


def _compute_bbmacd_slope_3(df):
    return compute_momentum_factors(df)["bbmacd_slope_3"]


# 注册动量类因子
register_factor(
    name="bbmacd",
    category="动量类",
    compute_func=_compute_bbmacd,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_bbmacd",
    description="BBMACD柱状线",
    direction="positive",
    is_core=True,
)

register_factor(
    name="bbmacd_minus_avg",
    category="动量类",
    compute_func=_compute_bbmacd_minus_avg,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_bbmacd",
    description="BBMACD距均值偏差",
    direction="positive",
    is_core=False,
)

register_factor(
    name="bbmacd_state",
    category="动量类",
    compute_func=_compute_bbmacd_state,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_bbmacd",
    description="BBMACD状态",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="bbmacd_band_pos_01",
    category="动量类",
    compute_func=_compute_bbmacd_band_pos_01,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_bbmacd",
    description="BBMACD在布林带中的位置（0-1）",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="bbmacd_bandwidth_zscore",
    category="动量类",
    compute_func=_compute_bbmacd_bandwidth_zscore,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_bbmacd",
    description="BBMACD带宽Z-score",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="bbmacd_cross_upper",
    category="动量类",
    compute_func=_compute_bbmacd_cross_upper,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_bbmacd",
    description="上穿布林带上轨",
    direction="positive",
    is_core=True,
)

register_factor(
    name="bbmacd_cross_lower",
    category="动量类",
    compute_func=_compute_bbmacd_cross_lower,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_bbmacd",
    description="下穿布林带下轨",
    direction="negative",
    is_core=True,
)

register_factor(
    name="bbmacd_sign",
    category="动量类",
    compute_func=_compute_bbmacd_sign,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_bbmacd",
    description="BBMACD符号：1=正，-1=负，0=零",
    direction="positive",
    is_core=False,
)

register_factor(
    name="bbmacd_slope_3",
    category="动量类",
    compute_func=_compute_bbmacd_slope_3,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_bbmacd",
    description="BBMACD 3日斜率",
    direction="positive",
    is_core=False,
)
