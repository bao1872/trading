# -*- coding: utf-8 -*-
"""
factor_lib/categories/position.py - 位置类因子

Purpose: 位置类因子（DSA枢轴位置/VWAP偏离/回撤反弹等）的批量计算与注册。
         基于 features/dsa_bbmacd_24factors_viewer.py 的 compute_dsa 权威实现。

Public API:
    compute_position_factors(df, dsa_result=None) -> DataFrame

Registered Factors:
    - dsa_pivot_pos_01: DSA枢轴位置（0-1）
    - price_vs_dsa_vwap_pct: 价格相对DSA VWAP百分比
    - ret_to_last_high_pct: 距前高回撤百分比
    - ret_to_last_low_pct: 距前低反弹百分比
    - current_pullback_from_stage_extreme_pct: 当前阶段极端回撤
    - liquidity_range_pos_01: 流动性区间位置（0-1）
"""
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np


def compute_position_factors(
    df: pd.DataFrame,
    dsa_result: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    批量计算全部位置类因子。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame
        dsa_result: 预计算的 compute_dsa 结果，若为 None 则内部计算

    Returns:
        DataFrame 含 6 列位置因子
    """
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, DSAConfig, _run_from_dir

    if dsa_result is None:
        cfg = DSAConfig()
        dsa_result, _, _ = compute_dsa(df, cfg)

    result = pd.DataFrame(index=df.index)
    result["dsa_pivot_pos_01"] = dsa_result["dsa_pivot_pos_01"]
    result["price_vs_dsa_vwap_pct"] = df["close"] / dsa_result["DSA_VWAP"] - 1.0
    result["ret_to_last_high_pct"] = df["close"] / dsa_result["last_confirmed_high"] - 1.0
    result["ret_to_last_low_pct"] = df["close"] / dsa_result["last_confirmed_low"] - 1.0

    dsa_dir = dsa_result["dsa_dir"]
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    current_pullback = np.full(len(df), np.nan)
    runs = _run_from_dir(dsa_dir.to_numpy(float))
    for st, ed, run_dir in runs:
        if run_dir > 0:
            roll_high = df["high"].iloc[st:ed + 1].cummax().to_numpy(dtype=float)
            curr_close = close[st:ed + 1]
            current_pullback[st:ed + 1] = curr_close / roll_high - 1.0
        else:
            roll_low = df["low"].iloc[st:ed + 1].cummin().to_numpy(dtype=float)
            curr_close = close[st:ed + 1]
            current_pullback[st:ed + 1] = curr_close / roll_low - 1.0
    result["current_pullback_from_stage_extreme_pct"] = dsa_result["dsa_pivot_pos_01"].__class__(
        current_pullback, index=df.index
    )

    from factor_lib.categories.structure import _compute_sell_stop_cluster, _compute_buy_stop_cluster
    sell_cluster = _compute_sell_stop_cluster(df)
    buy_cluster = _compute_buy_stop_cluster(df)
    sell_vals = sell_cluster.to_numpy(dtype=float)
    buy_vals = buy_cluster.to_numpy(dtype=float)
    range_width = buy_vals - sell_vals
    pos = np.full(len(df), np.nan)
    valid = (range_width > 0) & np.isfinite(sell_vals) & np.isfinite(buy_vals) & np.isfinite(close)
    pos[valid] = (close[valid] - sell_vals[valid]) / range_width[valid]
    pos = np.clip(pos, 0.0, 1.0)
    result["liquidity_range_pos_01"] = pd.Series(pos, index=df.index)
    return result


def _compute_dsa_pivot_pos_01(df):
    return compute_position_factors(df)["dsa_pivot_pos_01"]


def _compute_price_vs_dsa_vwap_pct(df):
    return compute_position_factors(df)["price_vs_dsa_vwap_pct"]


def _compute_ret_to_last_high_pct(df):
    return compute_position_factors(df)["ret_to_last_high_pct"]


def _compute_ret_to_last_low_pct(df):
    return compute_position_factors(df)["ret_to_last_low_pct"]


def _compute_current_pullback_from_stage_extreme_pct(df):
    return compute_position_factors(df)["current_pullback_from_stage_extreme_pct"]


def _compute_liquidity_range_pos_01(df):
    return compute_position_factors(df)["liquidity_range_pos_01"]


# 注册位置类因子
register_factor(
    name="dsa_pivot_pos_01",
    category="位置类",
    compute_func=_compute_dsa_pivot_pos_01,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="DSA枢轴位置（0-1），0=阶段低点，1=阶段高点",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="price_vs_dsa_vwap_pct",
    category="位置类",
    compute_func=_compute_price_vs_dsa_vwap_pct,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="价格相对DSA VWAP百分比，核心因子（占48%重要性）",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="ret_to_last_high_pct",
    category="位置类",
    compute_func=_compute_ret_to_last_high_pct,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="距前高回撤百分比",
    direction="negative",
    is_core=False,
)

register_factor(
    name="ret_to_last_low_pct",
    category="位置类",
    compute_func=_compute_ret_to_last_low_pct,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="距前低反弹百分比",
    direction="positive",
    is_core=False,
)

register_factor(
    name="current_pullback_from_stage_extreme_pct",
    category="位置类",
    compute_func=_compute_current_pullback_from_stage_extreme_pct,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="当前阶段极端回撤百分比",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="liquidity_range_pos_01",
    category="位置类",
    compute_func=_compute_liquidity_range_pos_01,
    source_module="factor_lib.categories.position",
    source_function="_compute_liquidity_range_pos_01",
    description="流动性区间位置（0-1）：0=触及卖方流动性低点，1=触及买方流动性高点",
    direction="neutral",
    is_core=False,
)
