# -*- coding: utf-8 -*-
"""
factor_lib/categories/coordination.py - 协同类因子

Purpose: 协同类因子（量价/多维度共振）的批量计算与注册。
         基于 DSA 和 BBMACD 计算结果。

Public API:
    compute_coordination_factors(df, dsa_result=None, bb_result=None) -> DataFrame

Registered Factors:
    - price_vol_coord: 价格-成交量协调因子
    - momo_vol_coord: 动量-成交量协调因子
    - low_pos_break_coord: 低点位置突破协调因子
    - coord_consistency: 协调性一致性
    - coord_stage_current: 当前阶段协调状态
    - coord_stage_prev: 前一阶段协调状态
    - coord_stage_ratio: 阶段协调比率
"""
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np


def compute_coordination_factors(
    df: pd.DataFrame,
    dsa_result: pd.DataFrame = None,
    bb_result: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    批量计算全部协同类因子。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame
        dsa_result: 预计算的 compute_dsa 结果，若为 None 则内部计算
        bb_result: 预计算的 compute_bbmacd 结果，若为 None 则内部计算

    Returns:
        DataFrame 含 7 列协同因子
    """
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, compute_bbmacd, DSAConfig, _run_from_dir

    if dsa_result is None:
        cfg = DSAConfig()
        dsa_result, _, _ = compute_dsa(df, cfg)
    if bb_result is None:
        bb_result = compute_bbmacd(df)

    result = pd.DataFrame(index=df.index)
    dsa_dir = dsa_result["dsa_dir"]
    bbmacd = bb_result["bbmacd"]
    vol = df["volume"]
    vol_mu = vol.rolling(20, min_periods=5).mean()
    vol_sd = vol.rolling(20, min_periods=5).std(ddof=1).replace(0.0, np.nan)
    vol_z = (vol - vol_mu) / vol_sd

    coord = np.zeros(len(df), dtype=float)
    long_coord = (dsa_dir > 0) & (vol_z > 0)
    short_coord = (dsa_dir < 0) & (vol_z < 0)
    coord[long_coord.to_numpy()] = 1.0
    coord[short_coord.to_numpy()] = -1.0
    result["price_vol_coord"] = pd.Series(coord, index=df.index)

    coord2 = np.zeros(len(df), dtype=float)
    m_long = (bbmacd > 0) & (vol_z > 0)
    m_short = (bbmacd < 0) & (vol_z < 0)
    coord2[m_long.to_numpy()] = 1.0
    coord2[m_short.to_numpy()] = -1.0
    result["momo_vol_coord"] = pd.Series(coord2, index=df.index)

    pivot_type = dsa_result["prev_pivot_type"]
    is_hl = pivot_type == "HL"
    is_hh = pivot_type == "HH"
    lb_coord = np.zeros(len(df), dtype=float)
    lb_coord[(is_hl | is_hh).to_numpy()] = 1.0
    result["low_pos_break_coord"] = pd.Series(lb_coord, index=df.index)

    pvc = result["price_vol_coord"]
    mvc = result["momo_vol_coord"]
    consistency = np.zeros(len(df), dtype=float)
    consistency[(pvc > 0) & (mvc > 0)] = 1.0
    consistency[(pvc < 0) & (mvc < 0)] = 1.0
    consistency[(pvc > 0) & (mvc < 0)] = -1.0
    consistency[(pvc < 0) & (mvc > 0)] = -1.0
    result["coord_consistency"] = pd.Series(consistency, index=df.index)

    stage = np.zeros(len(df), dtype=float)
    long_all = (dsa_dir > 0) & (bbmacd > 0) & (vol_z > 0)
    short_all = (dsa_dir < 0) & (bbmacd < 0) & (vol_z < 0)
    stage[long_all.to_numpy()] = 1.0
    stage[short_all.to_numpy()] = -1.0
    result["coord_stage_current"] = pd.Series(stage, index=df.index)

    prev_stage = np.full(len(df), np.nan)
    runs = _run_from_dir(dsa_dir.to_numpy(float))
    prev_val = np.nan
    for st, ed, run_dir in runs:
        prev_stage[st:ed + 1] = prev_val
        prev_val = stage[ed] if ed < len(stage) else np.nan
    result["coord_stage_prev"] = pd.Series(prev_stage, index=df.index)

    is_coord = result["coord_stage_current"] != 0
    result["coord_stage_ratio"] = is_coord.rolling(20, min_periods=5).mean()
    return result


def _compute_price_vol_coord(df):
    return compute_coordination_factors(df)["price_vol_coord"]


def _compute_momo_vol_coord(df):
    return compute_coordination_factors(df)["momo_vol_coord"]


def _compute_low_pos_break_coord(df):
    return compute_coordination_factors(df)["low_pos_break_coord"]


def _compute_coord_consistency(df):
    return compute_coordination_factors(df)["coord_consistency"]


def _compute_coord_stage_current(df):
    return compute_coordination_factors(df)["coord_stage_current"]


def _compute_coord_stage_prev(df):
    return compute_coordination_factors(df)["coord_stage_prev"]


def _compute_coord_stage_ratio(df):
    return compute_coordination_factors(df)["coord_stage_ratio"]


# 注册协同类因子
register_factor(
    name="price_vol_coord",
    category="协同类",
    compute_func=_compute_price_vol_coord,
    source_module="factor_lib.categories.coordination",
    source_function="_compute_price_vol_coord",
    description="价格-成交量协调因子",
    direction="positive",
    is_core=True,
)

register_factor(
    name="momo_vol_coord",
    category="协同类",
    compute_func=_compute_momo_vol_coord,
    source_module="factor_lib.categories.coordination",
    source_function="_compute_momo_vol_coord",
    description="动量-成交量协调因子",
    direction="positive",
    is_core=False,
)

register_factor(
    name="low_pos_break_coord",
    category="协同类",
    compute_func=_compute_low_pos_break_coord,
    source_module="factor_lib.categories.coordination",
    source_function="_compute_low_pos_break_coord",
    description="低点位置突破协调因子",
    direction="positive",
    is_core=False,
)

register_factor(
    name="coord_consistency",
    category="协同类",
    compute_func=_compute_coord_consistency,
    source_module="factor_lib.categories.coordination",
    source_function="_compute_coord_consistency",
    description="协调性一致性",
    direction="positive",
    is_core=False,
)

register_factor(
    name="coord_stage_current",
    category="协同类",
    compute_func=_compute_coord_stage_current,
    source_module="factor_lib.categories.coordination",
    source_function="_compute_coord_stage_current",
    description="当前阶段协调状态",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="coord_stage_prev",
    category="协同类",
    compute_func=_compute_coord_stage_prev,
    source_module="factor_lib.categories.coordination",
    source_function="_compute_coord_stage_prev",
    description="前一阶段协调状态",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="coord_stage_ratio",
    category="协同类",
    compute_func=_compute_coord_stage_ratio,
    source_module="factor_lib.categories.coordination",
    source_function="_compute_coord_stage_ratio",
    description="阶段协调比率",
    direction="neutral",
    is_core=False,
)
