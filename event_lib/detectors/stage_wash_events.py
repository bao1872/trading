# -*- coding: utf-8 -*-
"""
event_lib/detectors/stage_wash_events.py - 洗盘循环事件检测

Purpose: 基于阶段位置/成熟度因子列检测洗盘循环事件。

Registered Events:
    - evt_wash_pullback_to_lower: 从中上部回踩到下沿/低位
    - evt_wash_break_short_hold_long: 小破位但大结构收回
    - evt_wash_reclaim_mid: 洗盘后回中枢
    - evt_wash_cycle_complete: 一轮有效洗盘完成
    - evt_wash_multi_cycle: 多轮洗盘迭代
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_wash_pullback_to_lower(factors_df: pd.DataFrame) -> pd.Series:
    if "price_pos_in_stage_01" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    pos = factors_df["price_pos_in_stage_01"]
    prev_pos = pos.shift(1)
    return ((pos < 0.2) & (prev_pos > 0.5)).astype(int)


def _detect_wash_break_short_hold_long(factors_df: pd.DataFrame) -> pd.Series:
    if "break_lower_intrabar" not in factors_df.columns or "reclaim_lower_close" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (
        (factors_df["break_lower_intrabar"] > 0) & (factors_df["reclaim_lower_close"] > 0)
    ).astype(int)


def _detect_wash_reclaim_mid(factors_df: pd.DataFrame) -> pd.Series:
    if "close" not in factors_df.columns or "stage_mid_boundary" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    close = factors_df["close"]
    mid = factors_df["stage_mid_boundary"]
    prev_below = close.shift(1) < mid.shift(1)
    curr_above = close >= mid
    return (prev_below & curr_above).astype(int)


def _detect_wash_cycle_complete(factors_df: pd.DataFrame) -> pd.Series:
    if "wash_cycle_count_n" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    count = factors_df["wash_cycle_count_n"]
    return (count > count.shift(1)).astype(int)


def _detect_wash_multi_cycle(factors_df: pd.DataFrame) -> pd.Series:
    if "wash_cycle_count_n" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["wash_cycle_count_n"] >= 2).astype(int)


register_event(
    name="evt_wash_pullback_to_lower",
    category="洗盘事件",
    detect_func=_detect_wash_pullback_to_lower,
    required_factors=["price_pos_in_stage_01"],
    description="从中上部回踩到下沿/低位（pos从>0.5降到<0.2）",
    direction="neutral",
    is_core=True,
)

register_event(
    name="evt_wash_break_short_hold_long",
    category="洗盘事件",
    detect_func=_detect_wash_break_short_hold_long,
    required_factors=["break_lower_intrabar", "reclaim_lower_close"],
    description="小破位但大结构收回（同bar破位+收回）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_wash_reclaim_mid",
    category="洗盘事件",
    detect_func=_detect_wash_reclaim_mid,
    required_factors=["close", "stage_mid_boundary"],
    description="洗盘后回中枢（价格从下方穿越中枢）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_wash_cycle_complete",
    category="洗盘事件",
    detect_func=_detect_wash_cycle_complete,
    required_factors=["wash_cycle_count_n"],
    description="一轮有效洗盘完成（wash_cycle_count增加）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_wash_multi_cycle",
    category="洗盘事件",
    detect_func=_detect_wash_multi_cycle,
    required_factors=["wash_cycle_count_n"],
    description="多轮洗盘迭代（wash_cycle_count >= 2）",
    direction="positive",
    is_core=False,
)
