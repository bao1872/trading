# -*- coding: utf-8 -*-
"""
event_lib/detectors/sr_resistance_events.py - SR压力事件检测

Purpose: 基于压力类因子列检测SR压力相关事件。

Registered Events:
    - evt_high_break_recent_resistance: 盘中突破压力
    - evt_cross_recent_resistance: 收盘上穿压力
    - evt_wick_break_resistance_fail: 影线突破失败
    - evt_gap_up_break_resistance: 跳空突破压力
    - evt_break_resistance_bull_bar: 突破压力+阳线
    - evt_break_resistance_close_strong: 突破压力+收盘强势
    - evt_break_resistance_from_low_zone: 低位突破压力
    - evt_break_resistance_from_mid_zone: 中位突破压力
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_high_break_recent_resistance(factors_df: pd.DataFrame) -> pd.Series:
    return (factors_df["high"] > factors_df["resistance_ref"]).astype(int)


def _detect_cross_recent_resistance(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_cross_recent_resistance"].fillna(False).astype(int)


def _detect_wick_break_resistance_fail(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_wick_break_resistance_fail"].fillna(False).astype(int)


def _detect_gap_up_break_resistance(factors_df: pd.DataFrame) -> pd.Series:
    cross = factors_df["evt_cross_recent_resistance"].fillna(False)
    open_above = factors_df["open"] > factors_df["resistance_ref"].shift(1)
    return (cross & open_above).astype(int)


def _detect_break_resistance_bull_bar(factors_df: pd.DataFrame) -> pd.Series:
    cross = factors_df["evt_cross_recent_resistance"].fillna(False)
    bull = factors_df.get("is_bull_bar", pd.Series(False, index=factors_df.index)).fillna(False)
    return (cross & bull).astype(int)


def _detect_break_resistance_close_strong(factors_df: pd.DataFrame) -> pd.Series:
    cross = factors_df["evt_cross_recent_resistance"].fillna(False)
    close_pos = factors_df.get("close_pos_in_bar", pd.Series(0.0, index=factors_df.index)).fillna(0)
    return (cross & (close_pos > 0.7)).astype(int)


def _detect_break_resistance_from_low_zone(factors_df: pd.DataFrame) -> pd.Series:
    cross = factors_df["evt_cross_recent_resistance"].fillna(False)
    sr_pos = factors_df.get("sr_pos_01", pd.Series(0.5, index=factors_df.index)).shift(1).fillna(0.5)
    return (cross & (sr_pos <= 0.35)).astype(int)


def _detect_break_resistance_from_mid_zone(factors_df: pd.DataFrame) -> pd.Series:
    cross = factors_df["evt_cross_recent_resistance"].fillna(False)
    sr_pos = factors_df.get("sr_pos_01", pd.Series(0.5, index=factors_df.index)).shift(1).fillna(0.5)
    return (cross & (sr_pos <= 0.50)).astype(int)


def _detect_break_strong_resistance_cluster(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_break_strong_resistance_cluster"].fillna(False).astype(int)


def _detect_wick_break_resistance_cluster_fail(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_wick_break_resistance_cluster_fail"].fillna(False).astype(int)


def _detect_close_above_resistance_cluster_upper(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_close_above_resistance_cluster_upper"].fillna(False).astype(int)


register_event(
    name="evt_high_break_recent_resistance",
    category="SR压力事件",
    detect_func=_detect_high_break_recent_resistance,
    required_factors=["high", "resistance_ref"],
    description="盘中突破压力",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_cross_recent_resistance",
    category="SR压力事件",
    detect_func=_detect_cross_recent_resistance,
    required_factors=["close", "resistance_ref"],
    description="收盘上穿压力",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_wick_break_resistance_fail",
    category="SR压力事件",
    detect_func=_detect_wick_break_resistance_fail,
    required_factors=["high", "resistance_ref", "close"],
    description="影线突破失败",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_gap_up_break_resistance",
    category="SR压力事件",
    detect_func=_detect_gap_up_break_resistance,
    required_factors=["open", "close", "resistance_ref"],
    description="跳空突破压力",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_break_resistance_bull_bar",
    category="SR压力事件",
    detect_func=_detect_break_resistance_bull_bar,
    required_factors=["evt_cross_recent_resistance", "is_bull_bar"],
    description="突破压力+阳线",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_break_resistance_close_strong",
    category="SR压力事件",
    detect_func=_detect_break_resistance_close_strong,
    required_factors=["evt_cross_recent_resistance", "close_pos_in_bar"],
    description="突破压力+收盘强势",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_break_resistance_from_low_zone",
    category="SR压力事件",
    detect_func=_detect_break_resistance_from_low_zone,
    required_factors=["evt_cross_recent_resistance", "sr_pos_01"],
    description="低位突破压力",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_break_resistance_from_mid_zone",
    category="SR压力事件",
    detect_func=_detect_break_resistance_from_mid_zone,
    required_factors=["evt_cross_recent_resistance", "sr_pos_01"],
    description="中位突破压力",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_break_strong_resistance_cluster",
    category="SR压力事件",
    detect_func=_detect_break_strong_resistance_cluster,
    required_factors=["evt_break_strong_resistance_cluster"],
    description="突破强压力簇",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_wick_break_resistance_cluster_fail",
    category="SR压力事件",
    detect_func=_detect_wick_break_resistance_cluster_fail,
    required_factors=["evt_wick_break_resistance_cluster_fail"],
    description="强压力簇影线突破失败",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_close_above_resistance_cluster_upper",
    category="SR压力事件",
    detect_func=_detect_close_above_resistance_cluster_upper,
    required_factors=["evt_close_above_resistance_cluster_upper"],
    description="收盘站上压力簇上界",
    direction="positive",
    is_core=True,
)


if __name__ == "__main__":
    from event_lib.registry import list_by_category

    events = list_by_category("SR压力事件")
    print(f"SR压力事件已注册 {len(events)} 个:")
    for e in events:
        core_tag = "[核心]" if e["is_core"] else ""
        print(f"  {e['name']} ({e['direction']}) {core_tag} - {e['description']}")
