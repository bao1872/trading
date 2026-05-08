# -*- coding: utf-8 -*-
"""
event_lib/detectors/volume_events.py - 量能事件检测

Purpose: 基于量能类因子列检测量能相关事件。

Registered Events:
    - evt_up_move_with_vol_spike: 上涨+放量
    - evt_down_move_with_vol_spike: 下跌+放量
    - evt_vol_shrink: 缩量
    - evt_vol_divergence: 量价背离
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_up_move_with_vol_spike(factors_df: pd.DataFrame) -> pd.Series:
    """上涨+放量：价格上涨且成交量Z-score > 2。"""
    if "vol_zscore_20" not in factors_df.columns or "close" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    price_up = factors_df["close"] > factors_df["close"].shift(1)
    vol_spike = factors_df["vol_zscore_20"] > 2
    return (price_up & vol_spike).astype(int)


def _detect_down_move_with_vol_spike(factors_df: pd.DataFrame) -> pd.Series:
    """下跌+放量：价格下跌且成交量Z-score > 2。"""
    if "vol_zscore_20" not in factors_df.columns or "close" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    price_down = factors_df["close"] < factors_df["close"].shift(1)
    vol_spike = factors_df["vol_zscore_20"] > 2
    return (price_down & vol_spike).astype(int)


def _detect_vol_shrink(factors_df: pd.DataFrame) -> pd.Series:
    """缩量：成交量Z-score < -1。"""
    if "vol_zscore_20" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["vol_zscore_20"] < -1).astype(int)


def _detect_vol_divergence(factors_df: pd.DataFrame) -> pd.Series:
    """量价背离：价格上涨但成交量下降，或价格下跌但成交量上升。"""
    if "vol_zscore_20" not in factors_df.columns or "close" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    price_change = factors_df["close"].diff()
    vol_change = factors_df["vol_zscore_20"].diff()
    # 价涨量缩 或 价跌量增
    divergence = ((price_change > 0) & (vol_change < 0)) | ((price_change < 0) & (vol_change > 0))
    return divergence.astype(int)


# 注册量能事件
register_event(
    name="evt_up_move_with_vol_spike",
    category="量能事件",
    detect_func=_detect_up_move_with_vol_spike,
    required_factors=["vol_zscore_20", "close"],
    description="上涨+放量（vol_zscore > 2）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_down_move_with_vol_spike",
    category="量能事件",
    detect_func=_detect_down_move_with_vol_spike,
    required_factors=["vol_zscore_20", "close"],
    description="下跌+放量（vol_zscore > 2）",
    direction="negative",
    is_core=True,
)

register_event(
    name="evt_vol_shrink",
    category="量能事件",
    detect_func=_detect_vol_shrink,
    required_factors=["vol_zscore_20"],
    description="缩量（vol_zscore < -1）",
    direction="neutral",
    is_core=False,
)

register_event(
    name="evt_vol_divergence",
    category="量能事件",
    detect_func=_detect_vol_divergence,
    required_factors=["vol_zscore_20", "close"],
    description="量价背离",
    direction="neutral",
    is_core=False,
)
