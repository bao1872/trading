# -*- coding: utf-8 -*-
"""
event_lib/detectors/stage_shake_events.py - 破位收回事件检测

Purpose: 基于阶段位置/成熟度因子列检测破位与收回事件。

Registered Events:
    - evt_lower_break: 跌破阶段下沿
    - evt_break_last_reclaim_low: 跌破上一轮回收低点
    - evt_lower_break_reclaim: 跌破下沿后收回
    - evt_long_lower_shadow: 长下影线
    - evt_stop_cluster_reclaim: 打到下方止损簇后收回
    - evt_range_expand_down_reclaim: 大振幅向下假破收回
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_lower_break(factors_df: pd.DataFrame) -> pd.Series:
    required = ["low", "stage_lower_boundary"]
    if not all(c in factors_df.columns for c in required):
        return pd.Series(0, index=factors_df.index)
    lower_ref = factors_df["stage_lower_boundary"].shift(1)
    return (factors_df["low"] < lower_ref).astype(int)


def _detect_break_last_reclaim_low(factors_df: pd.DataFrame) -> pd.Series:
    if "break_last_wash_low" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["break_last_wash_low"] > 0).astype(int)


def _detect_lower_break_reclaim(factors_df: pd.DataFrame) -> pd.Series:
    required = ["low", "close", "stage_lower_boundary"]
    if not all(c in factors_df.columns for c in required):
        return pd.Series(0, index=factors_df.index)
    lower_ref = factors_df["stage_lower_boundary"].shift(1)
    return ((factors_df["low"] < lower_ref) & (factors_df["close"] >= lower_ref)).astype(int)


def _detect_long_lower_shadow(factors_df: pd.DataFrame) -> pd.Series:
    if "lower_shadow_ratio" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["lower_shadow_ratio"] > 0.6).astype(int)


def _detect_stop_cluster_reclaim(factors_df: pd.DataFrame) -> pd.Series:
    if "sell_stop_cluster" not in factors_df.columns or "close" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    low = factors_df["low"] if "low" in factors_df.columns else factors_df["close"]
    sell_cluster = factors_df["sell_stop_cluster"]
    close = factors_df["close"]
    touched = low <= sell_cluster
    reclaimed = close >= sell_cluster
    return (touched & reclaimed).astype(int)


def _detect_range_expand_down_reclaim(factors_df: pd.DataFrame) -> pd.Series:
    required = ["shake_range_atr", "low", "close", "stage_lower_boundary"]
    if not all(c in factors_df.columns for c in required):
        return pd.Series(0, index=factors_df.index)
    lower_ref = factors_df["stage_lower_boundary"].shift(1)
    big_range = factors_df["shake_range_atr"] > 2.0
    broke = factors_df["low"] < lower_ref
    reclaimed = factors_df["close"] >= lower_ref
    return (big_range & broke & reclaimed).astype(int)


register_event(
    name="evt_lower_break",
    category="破位收回事件",
    detect_func=_detect_lower_break,
    required_factors=["low", "stage_lower_boundary"],
    description="跌破阶段下沿（使用上一bar边界）",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_break_last_reclaim_low",
    category="破位收回事件",
    detect_func=_detect_break_last_reclaim_low,
    required_factors=["break_last_wash_low"],
    description="跌破上一轮回收低点",
    direction="negative",
    is_core=False,
)

register_event(
    name="evt_lower_break_reclaim",
    category="破位收回事件",
    detect_func=_detect_lower_break_reclaim,
    required_factors=["low", "close", "stage_lower_boundary"],
    description="跌破下沿后收回（使用上一bar边界）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_long_lower_shadow",
    category="破位收回事件",
    detect_func=_detect_long_lower_shadow,
    required_factors=["lower_shadow_ratio"],
    description="长下影线（lower_shadow_ratio > 0.6）",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_stop_cluster_reclaim",
    category="破位收回事件",
    detect_func=_detect_stop_cluster_reclaim,
    required_factors=["sell_stop_cluster", "close"],
    description="打到下方止损簇后收回",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_range_expand_down_reclaim",
    category="破位收回事件",
    detect_func=_detect_range_expand_down_reclaim,
    required_factors=["shake_range_atr", "low", "close", "stage_lower_boundary"],
    description="大振幅向下假破收回（shake_range_atr>2 且 破位+收回，使用上一bar边界）",
    direction="positive",
    is_core=False,
)
