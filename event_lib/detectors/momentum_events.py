# -*- coding: utf-8 -*-
"""
event_lib/detectors/momentum_events.py - 动量事件检测

Purpose: 基于动量类因子列检测动量相关事件。

Registered Events:
    - evt_bbmacd_cross_upper: BBMACD上穿上轨
    - evt_bbmacd_cross_lower: BBMACD下穿下轨
    - evt_macd_golden_cross: MACD金叉
    - evt_macd_death_cross: MACD死叉
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_bbmacd_cross_upper(factors_df: pd.DataFrame) -> pd.Series:
    """BBMACD上穿上轨。"""
    if "bbmacd_cross_upper" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return factors_df["bbmacd_cross_upper"].fillna(0).astype(int)


def _detect_bbmacd_cross_lower(factors_df: pd.DataFrame) -> pd.Series:
    """BBMACD下穿下轨。"""
    if "bbmacd_cross_lower" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return factors_df["bbmacd_cross_lower"].fillna(0).astype(int)


def _detect_macd_golden_cross(factors_df: pd.DataFrame) -> pd.Series:
    """MACD金叉：bbmacd从负变正。"""
    if "bbmacd" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    bbmacd = factors_df["bbmacd"]
    return ((bbmacd > 0) & (bbmacd.shift(1) <= 0)).astype(int)


def _detect_macd_death_cross(factors_df: pd.DataFrame) -> pd.Series:
    """MACD死叉：bbmacd从正变负。"""
    if "bbmacd" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    bbmacd = factors_df["bbmacd"]
    return ((bbmacd < 0) & (bbmacd.shift(1) >= 0)).astype(int)


# 注册动量事件
register_event(
    name="evt_bbmacd_cross_upper",
    category="动量事件",
    detect_func=_detect_bbmacd_cross_upper,
    required_factors=["bbmacd_cross_upper"],
    description="BBMACD上穿上轨",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_bbmacd_cross_lower",
    category="动量事件",
    detect_func=_detect_bbmacd_cross_lower,
    required_factors=["bbmacd_cross_lower"],
    description="BBMACD下穿下轨",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_macd_golden_cross",
    category="动量事件",
    detect_func=_detect_macd_golden_cross,
    required_factors=["bbmacd"],
    description="MACD金叉（bbmacd从负变正）",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_macd_death_cross",
    category="动量事件",
    detect_func=_detect_macd_death_cross,
    required_factors=["bbmacd"],
    description="MACD死叉（bbmacd从正变负）",
    direction="negative",
    is_core=False,
)
