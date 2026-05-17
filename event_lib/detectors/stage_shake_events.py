# -*- coding: utf-8 -*-
"""
event_lib/detectors/stage_shake_events.py - 震仓事件检测

Purpose: 基于阶段位置/成熟度因子列检测末端强洗/震仓候选事件。

Registered Events:
    - evt_shake_break_lower: 跌破阶段下沿
    - evt_shake_break_last_wash_low: 跌破上一轮洗盘低点
    - evt_shake_lower_reclaim: 跌破下沿后收回
    - evt_shake_long_lower_shadow: 长下影修复
    - evt_shake_stop_cluster_reclaim: 打到下方潜在止损簇后收回
    - evt_shake_candidate: 成熟成本区后的末端强洗候选
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_shake_break_lower(factors_df: pd.DataFrame) -> pd.Series:
    if "break_lower_intrabar" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["break_lower_intrabar"] > 0).astype(int)


def _detect_shake_break_last_wash_low(factors_df: pd.DataFrame) -> pd.Series:
    if "break_last_wash_low" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["break_last_wash_low"] > 0).astype(int)


def _detect_shake_lower_reclaim(factors_df: pd.DataFrame) -> pd.Series:
    if "break_lower_intrabar" not in factors_df.columns or "reclaim_lower_close" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (
        (factors_df["break_lower_intrabar"] > 0) & (factors_df["reclaim_lower_close"] > 0)
    ).astype(int)


def _detect_shake_long_lower_shadow(factors_df: pd.DataFrame) -> pd.Series:
    if "lower_shadow_ratio" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["lower_shadow_ratio"] > 0.6).astype(int)


def _detect_shake_stop_cluster_reclaim(factors_df: pd.DataFrame) -> pd.Series:
    if "sell_stop_cluster" not in factors_df.columns or "close" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    low = factors_df["low"] if "low" in factors_df.columns else factors_df["close"]
    sell_cluster = factors_df["sell_stop_cluster"]
    close = factors_df["close"]
    touched = low <= sell_cluster
    reclaimed = close >= sell_cluster
    return (touched & reclaimed).astype(int)


def _detect_shake_candidate(factors_df: pd.DataFrame) -> pd.Series:
    if "cost_zone_maturity_score" not in factors_df.columns or "final_shake_score" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    mature = factors_df["cost_zone_maturity_score"] > 0.5
    shake = factors_df["final_shake_score"] > 0.5
    return (mature & shake).astype(int)


register_event(
    name="evt_shake_break_lower",
    category="震仓事件",
    detect_func=_detect_shake_break_lower,
    required_factors=["break_lower_intrabar"],
    description="跌破阶段下沿",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_shake_break_last_wash_low",
    category="震仓事件",
    detect_func=_detect_shake_break_last_wash_low,
    required_factors=["break_last_wash_low"],
    description="跌破上一轮洗盘低点",
    direction="negative",
    is_core=False,
)

register_event(
    name="evt_shake_lower_reclaim",
    category="震仓事件",
    detect_func=_detect_shake_lower_reclaim,
    required_factors=["break_lower_intrabar", "reclaim_lower_close"],
    description="跌破下沿后收回",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_shake_long_lower_shadow",
    category="震仓事件",
    detect_func=_detect_shake_long_lower_shadow,
    required_factors=["lower_shadow_ratio"],
    description="长下影修复（lower_shadow_ratio > 0.6）",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_shake_stop_cluster_reclaim",
    category="震仓事件",
    detect_func=_detect_shake_stop_cluster_reclaim,
    required_factors=["sell_stop_cluster", "close"],
    description="打到下方潜在止损簇后收回",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_shake_candidate",
    category="震仓事件",
    detect_func=_detect_shake_candidate,
    required_factors=["cost_zone_maturity_score", "final_shake_score"],
    description="成熟成本区后的末端强洗候选（maturity gating）",
    direction="positive",
    is_core=True,
)
