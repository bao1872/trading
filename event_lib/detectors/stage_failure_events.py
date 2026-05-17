# -*- coding: utf-8 -*-
"""
event_lib/detectors/stage_failure_events.py - 失败/风险事件检测

Purpose: 基于阶段位置/成熟度因子列检测破位/出货/失败过滤事件。

Registered Events:
    - evt_fail_break_lower_no_reclaim: 跌破下沿不收回
    - evt_fail_trend_down_confirm: 趋势确认向下
    - evt_fail_weak_rebound: 反弹缩量不过中枢
    - evt_fail_distribution_risk: 出货/派发风险高
    - evt_stage_failure: 阶段假设失败
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_fail_break_lower_no_reclaim(factors_df: pd.DataFrame) -> pd.Series:
    if "close" not in factors_df.columns or "stage_lower_boundary" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    close = factors_df["close"]
    lower = factors_df["stage_lower_boundary"]
    prev_below = close.shift(1) < lower.shift(1)
    curr_still_below = close < lower
    return (prev_below & curr_still_below).astype(int)


def _detect_fail_trend_down_confirm(factors_df: pd.DataFrame) -> pd.Series:
    required = ["dsa_dir", "dsa_vwap_slope_atr_5", "bbmacd_sign"]
    if not all(c in factors_df.columns for c in required):
        return pd.Series(0, index=factors_df.index)
    dsa_down = factors_df["dsa_dir"] < 0
    slope_neg = factors_df["dsa_vwap_slope_atr_5"] < 0
    bb_neg = factors_df["bbmacd_sign"] < 0
    return (dsa_down & slope_neg & bb_neg).astype(int)


def _detect_fail_weak_rebound(factors_df: pd.DataFrame) -> pd.Series:
    required = ["close", "stage_mid_boundary"]
    if not all(c in factors_df.columns for c in required):
        return pd.Series(0, index=factors_df.index)
    close = factors_df["close"]
    mid = factors_df["stage_mid_boundary"]
    below_mid = close < mid
    vol_shrink = pd.Series(False, index=factors_df.index)
    if "vol_zscore_20" in factors_df.columns:
        vol_shrink = factors_df["vol_zscore_20"] < -1
    return (below_mid & vol_shrink).astype(int)


def _detect_fail_distribution_risk(factors_df: pd.DataFrame) -> pd.Series:
    required = ["dsa_pivot_pos_01", "vol_zscore_20", "close"]
    if not all(c in factors_df.columns for c in required):
        return pd.Series(0, index=factors_df.index)
    high_pos = factors_df["dsa_pivot_pos_01"] > 0.8
    vol_spike = factors_df["vol_zscore_20"] > 2
    price_stall = factors_df["close"].diff() <= 0
    return (high_pos & vol_spike & price_stall).astype(int)


def _detect_stage_failure(factors_df: pd.DataFrame) -> pd.Series:
    if "failure_score" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["failure_score"] > 0.6).astype(int)


register_event(
    name="evt_fail_break_lower_no_reclaim",
    category="失败事件",
    detect_func=_detect_fail_break_lower_no_reclaim,
    required_factors=["close", "stage_lower_boundary"],
    description="跌破下沿不收回（连续2 bar收盘低于下沿）",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_fail_trend_down_confirm",
    category="失败事件",
    detect_func=_detect_fail_trend_down_confirm,
    required_factors=["dsa_dir", "dsa_vwap_slope_atr_5", "bbmacd_sign"],
    description="趋势确认向下（DSA下行+斜率负+BBMACD负）",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_fail_weak_rebound",
    category="失败事件",
    detect_func=_detect_fail_weak_rebound,
    required_factors=["close", "stage_mid_boundary", "vol_zscore_20"],
    description="反弹缩量不过中枢",
    direction="negative",
    is_core=False,
)

register_event(
    name="evt_fail_distribution_risk",
    category="失败事件",
    detect_func=_detect_fail_distribution_risk,
    required_factors=["dsa_pivot_pos_01", "vol_zscore_20", "close"],
    description="出货/派发风险高（高位+放量+滞涨）",
    direction="negative",
    is_core=False,
)

register_event(
    name="evt_stage_failure",
    category="失败事件",
    detect_func=_detect_stage_failure,
    required_factors=["failure_score"],
    description="阶段假设失败（failure_score > 0.6）",
    direction="negative",
    is_core=True,
)
