# -*- coding: utf-8 -*-
"""
event_lib/detectors/sr_support_events.py - SR支撑事件检测

Purpose: 基于支撑类因子列检测SR支撑相关事件。

Registered Events:
    - evt_pierce_recent_support: 盘中跌破最近支撑
    - evt_pierce_support_reclaim: 刺破有效支撑后收回
    - evt_pierce_pivot_support_reclaim: 刺破pivot支撑后收回
    - evt_pierce_flipped_support_reclaim: 刺破压力转支撑后收回
    - evt_pierce_active_support_reclaim: 刺破当前有效支撑后收回
    - evt_failed_reclaim_support: 刺破有效支撑失败
    - evt_failed_reclaim_pivot_support: 刺破pivot支撑失败
    - evt_failed_reclaim_flipped_support: 刺破压力转支撑失败
    - evt_failed_reclaim_active_support: 刺破当前有效支撑失败
    - evt_close_break_recent_support: 收盘跌破支撑
    - evt_retest_flipped_support: 回踩压力转支撑
    - evt_clean_hold_flipped_support: 压力转支撑干净守住
    - evt_breakdown_flipped_support: 跌破压力转支撑
    - evt_reclaim_support_and_bull_bar: 刺破收回+阳线
    - evt_reclaim_support_close_strong: 刺破收回+收盘强势
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_pierce_recent_support(factors_df: pd.DataFrame) -> pd.Series:
    return (factors_df["low"] < factors_df["support_ref"]).astype(int)


def _detect_pierce_support_reclaim(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_pierce_active_support_reclaim"].fillna(False).astype(int)


def _detect_pierce_pivot_support_reclaim(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_pierce_pivot_support_reclaim"].fillna(False).astype(int)


def _detect_pierce_flipped_support_reclaim(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_pierce_flipped_support_reclaim"].fillna(False).astype(int)


def _detect_pierce_active_support_reclaim(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_pierce_active_support_reclaim"].fillna(False).astype(int)


def _detect_failed_reclaim_support(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_failed_reclaim_active_support"].fillna(False).astype(int)


def _detect_failed_reclaim_pivot_support(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_failed_reclaim_pivot_support"].fillna(False).astype(int)


def _detect_failed_reclaim_flipped_support(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_failed_reclaim_flipped_support"].fillna(False).astype(int)


def _detect_failed_reclaim_active_support(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_failed_reclaim_active_support"].fillna(False).astype(int)


def _detect_close_break_recent_support(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_close_break_recent_support"].fillna(False).astype(int)


def _detect_retest_flipped_support(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_retest_flipped_support"].fillna(False).astype(int)


def _detect_clean_hold_flipped_support(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_clean_hold_flipped_support"].fillna(False).astype(int)


def _detect_breakdown_flipped_support(factors_df: pd.DataFrame) -> pd.Series:
    return factors_df["evt_breakdown_flipped_support"].fillna(False).astype(int)


def _detect_reclaim_support_and_bull_bar(factors_df: pd.DataFrame) -> pd.Series:
    pierce = factors_df["evt_pierce_active_support_reclaim"].fillna(False)
    bull = factors_df.get("is_bull_bar", pd.Series(False, index=factors_df.index)).fillna(False)
    return (pierce & bull).astype(int)


def _detect_reclaim_support_close_strong(factors_df: pd.DataFrame) -> pd.Series:
    pierce = factors_df["evt_pierce_active_support_reclaim"].fillna(False)
    close_pos = factors_df.get("close_pos_in_bar", pd.Series(0.0, index=factors_df.index)).fillna(0)
    return (pierce & (close_pos > 0.7)).astype(int)


register_event(
    name="evt_pierce_recent_support",
    category="SR支撑事件",
    detect_func=_detect_pierce_recent_support,
    required_factors=["low", "support_ref"],
    description="盘中跌破最近支撑",
    direction="negative",
    is_core=False,
)

register_event(
    name="evt_pierce_support_reclaim",
    category="SR支撑事件",
    detect_func=_detect_pierce_support_reclaim,
    required_factors=["evt_pierce_active_support_reclaim"],
    description="刺破有效支撑后收回",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_pierce_pivot_support_reclaim",
    category="SR支撑事件",
    detect_func=_detect_pierce_pivot_support_reclaim,
    required_factors=["low", "pivot_support_ref", "close"],
    description="刺破pivot支撑后收回",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_pierce_flipped_support_reclaim",
    category="SR支撑事件",
    detect_func=_detect_pierce_flipped_support_reclaim,
    required_factors=["low", "flipped_support_ref", "close"],
    description="刺破压力转支撑后收回",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_pierce_active_support_reclaim",
    category="SR支撑事件",
    detect_func=_detect_pierce_active_support_reclaim,
    required_factors=["low", "active_support_ref", "close"],
    description="刺破当前有效支撑后收回",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_failed_reclaim_support",
    category="SR支撑事件",
    detect_func=_detect_failed_reclaim_support,
    required_factors=["evt_failed_reclaim_active_support"],
    description="刺破有效支撑失败",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_failed_reclaim_pivot_support",
    category="SR支撑事件",
    detect_func=_detect_failed_reclaim_pivot_support,
    required_factors=["low", "pivot_support_ref", "close"],
    description="刺破pivot支撑失败",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_failed_reclaim_flipped_support",
    category="SR支撑事件",
    detect_func=_detect_failed_reclaim_flipped_support,
    required_factors=["low", "flipped_support_ref", "close"],
    description="刺破压力转支撑失败",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_failed_reclaim_active_support",
    category="SR支撑事件",
    detect_func=_detect_failed_reclaim_active_support,
    required_factors=["low", "active_support_ref", "close"],
    description="刺破当前有效支撑失败",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_close_break_recent_support",
    category="SR支撑事件",
    detect_func=_detect_close_break_recent_support,
    required_factors=["close", "support_ref"],
    description="收盘跌破支撑",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_retest_flipped_support",
    category="SR支撑事件",
    detect_func=_detect_retest_flipped_support,
    required_factors=["low", "flipped_support_ref", "atr_14"],
    description="回踩压力转支撑",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_clean_hold_flipped_support",
    category="SR支撑事件",
    detect_func=_detect_clean_hold_flipped_support,
    required_factors=["low", "flipped_support_ref", "atr_14", "is_support_flipped"],
    description="压力转支撑干净守住",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_breakdown_flipped_support",
    category="SR支撑事件",
    detect_func=_detect_breakdown_flipped_support,
    required_factors=["close", "flipped_support_ref", "is_support_flipped"],
    description="跌破压力转支撑",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_reclaim_support_and_bull_bar",
    category="SR支撑事件",
    detect_func=_detect_reclaim_support_and_bull_bar,
    required_factors=["evt_pierce_active_support_reclaim", "is_bull_bar"],
    description="刺破收回+阳线",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_reclaim_support_close_strong",
    category="SR支撑事件",
    detect_func=_detect_reclaim_support_close_strong,
    required_factors=["evt_pierce_active_support_reclaim", "close_pos_in_bar"],
    description="刺破收回+收盘强势",
    direction="positive",
    is_core=False,
)


if __name__ == "__main__":
    from event_lib.registry import list_by_category

    events = list_by_category("SR支撑事件")
    print(f"SR支撑事件已注册 {len(events)} 个:")
    for e in events:
        core_tag = "[核心]" if e["is_core"] else ""
        print(f"  {e['name']} ({e['direction']}) {core_tag} - {e['description']}")
