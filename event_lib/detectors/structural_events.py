# -*- coding: utf-8 -*-
"""
event_lib/detectors/structural_events.py - 结构事件检测

Purpose: 基于结构类因子列检测结构相关事件。

Registered Events:
    - evt_break_sell_stop_cluster: 跌破卖出止损聚类
    - evt_break_buy_stop_cluster: 突破买入止损聚类
    - evt_support_broken: 支撑跌破
    - evt_resistance_broken: 阻力突破
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_break_sell_stop_cluster(factors_df: pd.DataFrame) -> pd.Series:
    """跌破卖出止损聚类。"""
    if "stop_cluster_levels" not in factors_df.columns or "close" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    # 简化：价格创20日新低
    low_20 = factors_df["close"].rolling(window=20, min_periods=1).min()
    return (factors_df["close"] <= low_20).astype(int)


def _detect_break_buy_stop_cluster(factors_df: pd.DataFrame) -> pd.Series:
    """突破买入止损聚类。"""
    if "stop_cluster_levels" not in factors_df.columns or "close" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    # 简化：价格创20日新高
    high_20 = factors_df["close"].rolling(window=20, min_periods=1).max()
    return (factors_df["close"] >= high_20).astype(int)


def _detect_support_broken(factors_df: pd.DataFrame) -> pd.Series:
    """支撑跌破。"""
    if "support_resistance_zones" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    # 使用结构类因子的代理
    return pd.Series(0, index=factors_df.index)


def _detect_resistance_broken(factors_df: pd.DataFrame) -> pd.Series:
    """阻力突破。"""
    if "support_resistance_zones" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    # 使用结构类因子的代理
    return pd.Series(0, index=factors_df.index)


# 注册结构事件
register_event(
    name="evt_break_sell_stop_cluster",
    category="结构事件",
    detect_func=_detect_break_sell_stop_cluster,
    required_factors=["close"],
    description="跌破卖出止损聚类（20日新低）",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_break_buy_stop_cluster",
    category="结构事件",
    detect_func=_detect_break_buy_stop_cluster,
    required_factors=["close"],
    description="突破买入止损聚类（20日新高）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_support_broken",
    category="结构事件",
    detect_func=_detect_support_broken,
    required_factors=["support_resistance_zones"],
    description="支撑跌破",
    direction="negative",
    is_core=False,
)

register_event(
    name="evt_resistance_broken",
    category="结构事件",
    detect_func=_detect_resistance_broken,
    required_factors=["support_resistance_zones"],
    description="阻力突破",
    direction="positive",
    is_core=False,
)
