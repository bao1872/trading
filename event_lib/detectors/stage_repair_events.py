# -*- coding: utf-8 -*-
"""
event_lib/detectors/stage_repair_events.py - 修复收回事件检测

Purpose: 基于阶段位置/趋势/动量因子列检测修复收回事件。

Registered Events:
    - evt_reclaim_lower: 收回下沿
    - evt_reclaim_mid: 收回中枢
    - evt_reclaim_upper: 收回上沿
    - evt_reclaim_dsa_vwap: 收回DSA VWAP
    - evt_trend_slope_turn_positive: DSA斜率由负转平/正
    - evt_bbmacd_turn_positive: BBMACD动量由弱转强
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_reclaim_lower(factors_df: pd.DataFrame) -> pd.Series:
    if "close" not in factors_df.columns or "stage_lower_boundary" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    close = factors_df["close"]
    lower_ref = factors_df["stage_lower_boundary"].shift(1)
    prev_below = close.shift(1) < lower_ref
    curr_above = close >= lower_ref
    return (prev_below & curr_above).astype(int)


def _detect_reclaim_mid(factors_df: pd.DataFrame) -> pd.Series:
    if "close" not in factors_df.columns or "stage_mid_boundary" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    close = factors_df["close"]
    mid_ref = factors_df["stage_mid_boundary"].shift(1)
    prev_below = close.shift(1) < mid_ref
    curr_above = close >= mid_ref
    return (prev_below & curr_above).astype(int)


def _detect_reclaim_upper(factors_df: pd.DataFrame) -> pd.Series:
    if "close" not in factors_df.columns or "stage_upper_boundary" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    close = factors_df["close"]
    upper_ref = factors_df["stage_upper_boundary"].shift(1)
    prev_below = close.shift(1) < upper_ref
    curr_above = close >= upper_ref
    return (prev_below & curr_above).astype(int)


def _detect_reclaim_dsa_vwap(factors_df: pd.DataFrame) -> pd.Series:
    if "close" not in factors_df.columns or "DSA_VWAP" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    close = factors_df["close"]
    vwap = factors_df["DSA_VWAP"]
    prev_below = close.shift(1) < vwap.shift(1)
    curr_above = close >= vwap
    return (prev_below & curr_above).astype(int)


def _detect_trend_slope_turn_positive(factors_df: pd.DataFrame) -> pd.Series:
    if "dsa_vwap_slope_atr_5" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    slope = factors_df["dsa_vwap_slope_atr_5"]
    prev_neg = slope.shift(1) < 0
    curr_non_neg = slope >= 0
    return (prev_neg & curr_non_neg).astype(int)


def _detect_bbmacd_turn_positive(factors_df: pd.DataFrame) -> pd.Series:
    if "bbmacd_sign" not in factors_df.columns or "bbmacd_slope_3" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    sign = factors_df["bbmacd_sign"]
    slope = factors_df["bbmacd_slope_3"]
    prev_neg = sign.shift(1) < 0
    curr_pos = sign > 0
    slope_positive = slope > 0
    return ((prev_neg & curr_pos) | (prev_neg & slope_positive)).astype(int)


register_event(
    name="evt_reclaim_lower",
    category="修复收回事件",
    detect_func=_detect_reclaim_lower,
    required_factors=["close", "stage_lower_boundary"],
    description="收回下沿（价格从下方穿越上一bar下沿）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_reclaim_mid",
    category="修复收回事件",
    detect_func=_detect_reclaim_mid,
    required_factors=["close", "stage_mid_boundary"],
    description="收回中枢（价格从下方穿越上一bar中枢）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_reclaim_upper",
    category="修复收回事件",
    detect_func=_detect_reclaim_upper,
    required_factors=["close", "stage_upper_boundary"],
    description="收回上沿（价格从下方穿越上一bar上沿）",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_reclaim_dsa_vwap",
    category="修复收回事件",
    detect_func=_detect_reclaim_dsa_vwap,
    required_factors=["close", "DSA_VWAP"],
    description="收回DSA VWAP",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_trend_slope_turn_positive",
    category="修复收回事件",
    detect_func=_detect_trend_slope_turn_positive,
    required_factors=["dsa_vwap_slope_atr_5"],
    description="DSA斜率由负转平/正",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_bbmacd_turn_positive",
    category="修复收回事件",
    detect_func=_detect_bbmacd_turn_positive,
    required_factors=["bbmacd_sign", "bbmacd_slope_3"],
    description="BBMACD动量由弱转强",
    direction="positive",
    is_core=False,
)
