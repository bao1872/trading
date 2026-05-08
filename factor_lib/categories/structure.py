# -*- coding: utf-8 -*-
"""
factor_lib/categories/structure.py - 结构类因子

Purpose: 结构类因子（支撑阻力/流动性/止损聚类/趋势线突破）的批量计算与注册。

Public API:
    compute_structure_factors(df) -> DataFrame

Registered Factors:
    - sell_stop_cluster: 聚类卖方流动性（近期低点）
    - buy_stop_cluster: 聚类买方流动性（近期高点）
    - support_resistance_zones: 支撑阻力区域突破
    - liquidity_pools: 流动性池强度
    - trendline_upper: 趋势线上轨
    - trendline_lower: 趋势线下轨
    - upper_break: 上轨突破
    - lower_break: 下轨突破
"""
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np


def compute_structure_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    批量计算全部结构类因子。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame

    Returns:
        DataFrame 含 8 列结构因子
    """
    result = pd.DataFrame(index=df.index)

    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    n = len(df)
    window = min(20, n)
    sell_clusters = np.full(n, np.nan)
    buy_clusters = np.full(n, np.nan)
    for i in range(n):
        st = max(0, i - window + 1)
        sell_clusters[i] = np.nanmin(low[st:i + 1])
        buy_clusters[i] = np.nanmax(high[st:i + 1])
    result["sell_stop_cluster"] = pd.Series(sell_clusters, index=df.index)
    result["buy_stop_cluster"] = pd.Series(buy_clusters, index=df.index)

    from features.support_resistance_channels import compute_sr_channels, SRChannelConfig
    cfg = SRChannelConfig()
    sr_result, _ = compute_sr_channels(df, cfg)
    result["support_resistance_zones"] = (
        sr_result["supportbroken"] | sr_result["resistancebroken"]
    ).astype(float)

    from features.liquidity_zones_plotly import build_liquidity_zones, LZConfig
    lz_cfg = LZConfig()
    try:
        payload = build_liquidity_zones(df, lz_cfg)
        zones = payload.get("zones", [])
        pool_strength = np.zeros(n)
        result["liquidity_pools"] = pd.Series(pool_strength, index=df.index)
    except Exception:
        result["liquidity_pools"] = pd.Series(np.zeros(n), index=df.index)

    from features.trendlines_with_breaks_luxalgo import trendlines_with_breaks, TLBConfig
    tlb_cfg = TLBConfig()
    tlb_result = trendlines_with_breaks(df, tlb_cfg)
    result["trendline_upper"] = tlb_result["upper_plot"]
    result["trendline_lower"] = tlb_result["lower_plot"]
    result["upper_break"] = tlb_result["upper_break"]
    result["lower_break"] = tlb_result["lower_break"]
    return result


def _compute_sell_stop_cluster(df):
    return compute_structure_factors(df)["sell_stop_cluster"]


def _compute_buy_stop_cluster(df):
    return compute_structure_factors(df)["buy_stop_cluster"]


def _compute_support_resistance_zones(df):
    return compute_structure_factors(df)["support_resistance_zones"]


def _compute_liquidity_pools(df):
    return compute_structure_factors(df)["liquidity_pools"]


def _compute_trendline_upper(df):
    return compute_structure_factors(df)["trendline_upper"]


def _compute_trendline_lower(df):
    return compute_structure_factors(df)["trendline_lower"]


def _compute_upper_break(df):
    return compute_structure_factors(df)["upper_break"]


def _compute_lower_break(df):
    return compute_structure_factors(df)["lower_break"]


# 注册结构类因子
register_factor(
    name="sell_stop_cluster",
    category="结构类",
    compute_func=_compute_sell_stop_cluster,
    source_module="factor_lib.categories.structure",
    source_function="_compute_sell_stop_cluster",
    description="聚类卖方流动性：近期低点区域，代表多头止损位/卖方流动性池",
    direction="negative",
    is_core=False,
)

register_factor(
    name="buy_stop_cluster",
    category="结构类",
    compute_func=_compute_buy_stop_cluster,
    source_module="factor_lib.categories.structure",
    source_function="_compute_buy_stop_cluster",
    description="聚类买方流动性：近期高点区域，代表空头止损位/买方流动性池",
    direction="positive",
    is_core=False,
)

register_factor(
    name="support_resistance_zones",
    category="结构类",
    compute_func=_compute_support_resistance_zones,
    source_module="features.support_resistance_channels",
    source_function="compute_sr_channels",
    description="支撑阻力区域突破",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="liquidity_pools",
    category="结构类",
    compute_func=_compute_liquidity_pools,
    source_module="features.liquidity_zones_plotly",
    source_function="build_liquidity_zones",
    description="流动性池强度",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="trendline_upper",
    category="结构类",
    compute_func=_compute_trendline_upper,
    source_module="features.trendlines_with_breaks_luxalgo",
    source_function="trendlines_with_breaks",
    description="趋势线上轨",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="trendline_lower",
    category="结构类",
    compute_func=_compute_trendline_lower,
    source_module="features.trendlines_with_breaks_luxalgo",
    source_function="trendlines_with_breaks",
    description="趋势线下轨",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="upper_break",
    category="结构类",
    compute_func=_compute_upper_break,
    source_module="features.trendlines_with_breaks_luxalgo",
    source_function="trendlines_with_breaks",
    description="上轨突破信号",
    direction="positive",
    is_core=False,
)

register_factor(
    name="lower_break",
    category="结构类",
    compute_func=_compute_lower_break,
    source_module="features.trendlines_with_breaks_luxalgo",
    source_function="trendlines_with_breaks",
    description="下轨突破信号",
    direction="negative",
    is_core=False,
)
