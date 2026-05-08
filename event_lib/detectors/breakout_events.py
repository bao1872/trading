# -*- coding: utf-8 -*-
"""
event_lib/detectors/breakout_events.py - 突破事件检测

Purpose: 基于位置类因子列检测突破相关事件。

Registered Events:
    - evt_cross_above_value_area_high: 价格上穿价值区域高点
    - evt_cross_below_value_area_low: 价格下穿价值区域低点
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_cross_above_value_area_high(factors_df: pd.DataFrame) -> pd.Series:
    """价格上穿价值区域高点（基于PAVP）。"""
    # 使用 dsa_pivot_pos_01 作为代理：接近1表示接近高点
    if "dsa_pivot_pos_01" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    pos = factors_df["dsa_pivot_pos_01"]
    return ((pos > 0.8) & (pos.shift(1) <= 0.8)).astype(int)


def _detect_cross_below_value_area_low(factors_df: pd.DataFrame) -> pd.Series:
    """价格下穿价值区域低点（基于PAVP）。"""
    if "dsa_pivot_pos_01" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    pos = factors_df["dsa_pivot_pos_01"]
    return ((pos < 0.2) & (pos.shift(1) >= 0.2)).astype(int)


# 注册突破事件
register_event(
    name="evt_cross_above_value_area_high",
    category="突破事件",
    detect_func=_detect_cross_above_value_area_high,
    required_factors=["dsa_pivot_pos_01"],
    description="价格上穿价值区域高点（pivot_pos > 0.8）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_cross_below_value_area_low",
    category="突破事件",
    detect_func=_detect_cross_below_value_area_low,
    required_factors=["dsa_pivot_pos_01"],
    description="价格下穿价值区域低点（pivot_pos < 0.2）",
    direction="negative",
    is_core=True,
)
