# -*- coding: utf-8 -*-
"""
event_lib/detectors/trend_events.py - 趋势事件检测

Purpose: 基于趋势类因子列检测趋势相关事件。

Registered Events:
    - evt_dsa_dir_flip_up: DSA方向翻转为上升
    - evt_dsa_dir_flip_down: DSA方向翻转为下降
    - evt_cross_above_dsa_vwap: 价格上穿DSA VWAP
    - evt_cross_below_dsa_vwap: 价格下穿DSA VWAP
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_dsa_dir_flip_up(factors_df: pd.DataFrame) -> pd.Series:
    """DSA方向从下降/盘整翻转为上升。"""
    if "dsa_dir" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    dsa_dir = factors_df["dsa_dir"]
    return ((dsa_dir == 1) & (dsa_dir.shift(1) != 1)).astype(int)


def _detect_dsa_dir_flip_down(factors_df: pd.DataFrame) -> pd.Series:
    """DSA方向从上升/盘整翻转为下降。"""
    if "dsa_dir" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    dsa_dir = factors_df["dsa_dir"]
    return ((dsa_dir == -1) & (dsa_dir.shift(1) != -1)).astype(int)


def _detect_cross_above_dsa_vwap(factors_df: pd.DataFrame) -> pd.Series:
    """价格上穿DSA VWAP。"""
    if "price_vs_dsa_vwap_pct" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    pct = factors_df["price_vs_dsa_vwap_pct"]
    return ((pct > 0) & (pct.shift(1) <= 0)).astype(int)


def _detect_cross_below_dsa_vwap(factors_df: pd.DataFrame) -> pd.Series:
    """价格下穿DSA VWAP。"""
    if "price_vs_dsa_vwap_pct" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    pct = factors_df["price_vs_dsa_vwap_pct"]
    return ((pct < 0) & (pct.shift(1) >= 0)).astype(int)


def _detect_vreversal(factors_df: pd.DataFrame) -> pd.Series:
    """BBMACD V型反转：bbmacd[t] > bbmacd[t-1] 且 bbmacd[t-1] < bbmacd[t-2]。

    这是 dsa_experiment 管线的核心选股触发事件。
    """
    if "bbmacd" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    bbmacd = factors_df["bbmacd"]
    t_gt_t1 = bbmacd > bbmacd.shift(1)
    t1_lt_t2 = bbmacd.shift(1) < bbmacd.shift(2)
    valid = bbmacd.notna() & bbmacd.shift(1).notna() & bbmacd.shift(2).notna()
    return (t_gt_t1 & t1_lt_t2 & valid).astype(int)


# 注册趋势事件
register_event(
    name="evt_dsa_dir_flip_up",
    category="趋势事件",
    detect_func=_detect_dsa_dir_flip_up,
    required_factors=["dsa_dir"],
    description="DSA方向从下降/盘整翻转为上升",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_dsa_dir_flip_down",
    category="趋势事件",
    detect_func=_detect_dsa_dir_flip_down,
    required_factors=["dsa_dir"],
    description="DSA方向从上升/盘整翻转为下降",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_cross_above_dsa_vwap",
    category="趋势事件",
    detect_func=_detect_cross_above_dsa_vwap,
    required_factors=["price_vs_dsa_vwap_pct"],
    description="价格上穿DSA VWAP",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_cross_below_dsa_vwap",
    category="趋势事件",
    detect_func=_detect_cross_below_dsa_vwap,
    required_factors=["price_vs_dsa_vwap_pct"],
    description="价格下穿DSA VWAP",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_vreversal",
    category="趋势事件",
    detect_func=_detect_vreversal,
    required_factors=["bbmacd"],
    description="BBMACD V型反转：bbmacd在局部最低点后拐头向上（dsa_experiment核心选股事件）",
    direction="positive",
    is_core=True,
)
