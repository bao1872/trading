# -*- coding: utf-8 -*-
"""
factor_lib/categories/trend.py - 趋势类因子

Purpose: 趋势类因子（DSA趋势方向/枢轴点编码/趋势动量一致性等）的批量计算与注册。
         基于 features/dsa_bbmacd_24factors_viewer.py 的 compute_dsa 权威实现。

Public API:
    compute_trend_factors(df, dsa_result=None, bb_result=None) -> DataFrame

Registered Factors:
    - dsa_dir: DSA趋势方向
    - prev_pivot_code: 前一个枢轴点编码
    - trend_align_momo: 趋势-动量一致性
    - dsa_dir_age: DSA方向持续时间
"""
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np


def compute_trend_factors(
    df: pd.DataFrame,
    dsa_result: pd.DataFrame = None,
    bb_result: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    批量计算全部趋势类因子。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame
        dsa_result: 预计算的 compute_dsa 结果，若为 None 则内部计算
        bb_result: 预计算的 compute_bbmacd 结果（trend_align_momo 需要），若为 None 则内部计算

    Returns:
        DataFrame 含列: dsa_dir, prev_pivot_code, trend_align_momo, dsa_dir_age
    """
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, compute_bbmacd, DSAConfig

    if dsa_result is None:
        cfg = DSAConfig()
        dsa_result, _, _ = compute_dsa(df, cfg)

    result = pd.DataFrame(index=df.index)
    result["dsa_dir"] = dsa_result["dsa_dir"]

    pivot_code_map = {"HH": 2.0, "HL": 1.0, "LH": -1.0, "LL": -2.0}
    result["prev_pivot_code"] = dsa_result["prev_pivot_type"].map(pivot_code_map).astype(float)

    if bb_result is None:
        bb_result = compute_bbmacd(df)
    dsa_dir = dsa_result["dsa_dir"]
    bbmacd_minus_avg = bb_result["bbmacd_minus_avg"]
    trend_align = np.zeros(len(df), dtype=float)
    long_align = (dsa_dir > 0) & (bbmacd_minus_avg > 0)
    short_align = (dsa_dir < 0) & (bbmacd_minus_avg < 0)
    trend_align[long_align.to_numpy()] = 1.0
    trend_align[short_align.to_numpy()] = -1.0
    result["trend_align_momo"] = pd.Series(trend_align, index=df.index)

    dir_series = dsa_result["dsa_dir"]
    age = np.zeros(len(dir_series), dtype=int)
    current_age = 0
    for i in range(len(dir_series)):
        if i == 0 or dir_series.iloc[i] != dir_series.iloc[i - 1]:
            current_age = 0
        else:
            current_age += 1
        age[i] = current_age
    result["dsa_dir_age"] = pd.Series(age, index=dir_series.index)
    return result


def _compute_dsa_dir(df):
    return compute_trend_factors(df)["dsa_dir"]


def _compute_prev_pivot_code(df):
    return compute_trend_factors(df)["prev_pivot_code"]


def _compute_trend_align_momo(df):
    return compute_trend_factors(df)["trend_align_momo"]


def _compute_dsa_dir_age(df):
    return compute_trend_factors(df)["dsa_dir_age"]


# 注册趋势类因子
register_factor(
    name="dsa_dir",
    category="趋势类",
    compute_func=_compute_dsa_dir,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="DSA趋势方向：1=上升阶段，-1=下降阶段",
    direction="positive",
    is_core=True,
)

register_factor(
    name="prev_pivot_code",
    category="趋势类",
    compute_func=_compute_prev_pivot_code,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="前一个枢轴点编码：HH=2, HL=1, LH=-1, LL=-2",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="trend_align_momo",
    category="趋势类",
    compute_func=_compute_trend_align_momo,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa + compute_bbmacd",
    description="趋势-动量一致性：1=一致，-1=背离",
    direction="positive",
    is_core=True,
)

register_factor(
    name="dsa_dir_age",
    category="趋势类",
    compute_func=_compute_dsa_dir_age,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="DSA方向持续时间（bars）",
    direction="neutral",
    is_core=False,
)
