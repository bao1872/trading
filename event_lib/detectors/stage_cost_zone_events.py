# -*- coding: utf-8 -*-
"""
event_lib/detectors/stage_cost_zone_events.py - 趋势位置事件检测

Purpose: 基于趋势/位置因子列检测趋势转平、中枢穿越频率、波动压缩、下沿收回、上沿拒绝、下沿触及事件。

Registered Events:
    - evt_trend_flat: 趋势进入低斜率/低方向状态
    - evt_price_cross_mid_freq_high: 价格围绕中枢穿越频率高
    - evt_volatility_compress: 波动压缩
    - evt_lower_reclaim_freq_high: 下沿多次收回
    - evt_upper_reject: 上沿试盘失败（最高价突破上沿但收盘回落）
    - evt_lower_touch: 最低价触及下沿
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_trend_flat(factors_df: pd.DataFrame) -> pd.Series:
    if "trend_flat_score" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["trend_flat_score"] > 0.7).astype(int)


def _detect_price_cross_mid_freq_high(factors_df: pd.DataFrame) -> pd.Series:
    if "price_cross_mid_count_n" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["price_cross_mid_count_n"] >= 3).astype(int)


def _detect_volatility_compress(factors_df: pd.DataFrame) -> pd.Series:
    if "bbmacd_bandwidth_zscore" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["bbmacd_bandwidth_zscore"] < -1).astype(int)


def _detect_lower_reclaim_freq_high(factors_df: pd.DataFrame) -> pd.Series:
    if "lower_reclaim_count_n" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["lower_reclaim_count_n"] >= 2).astype(int)


def _detect_upper_reject(factors_df: pd.DataFrame) -> pd.Series:
    required = ["high", "close", "stage_upper_boundary"]
    if not all(c in factors_df.columns for c in required):
        return pd.Series(0, index=factors_df.index)
    upper_ref = factors_df["stage_upper_boundary"].shift(1)
    return ((factors_df["high"] > upper_ref) & (factors_df["close"] <= upper_ref)).astype(int)


def _detect_lower_touch(factors_df: pd.DataFrame) -> pd.Series:
    required = ["low", "stage_lower_boundary"]
    if not all(c in factors_df.columns for c in required):
        return pd.Series(0, index=factors_df.index)
    lower_ref = factors_df["stage_lower_boundary"].shift(1)
    return (factors_df["low"] <= lower_ref).astype(int)


register_event(
    name="evt_trend_flat",
    category="趋势位置事件",
    detect_func=_detect_trend_flat,
    required_factors=["trend_flat_score"],
    description="趋势进入低斜率/低方向状态（trend_flat_score > 0.7）",
    direction="neutral",
    is_core=True,
)

register_event(
    name="evt_price_cross_mid_freq_high",
    category="趋势位置事件",
    detect_func=_detect_price_cross_mid_freq_high,
    required_factors=["price_cross_mid_count_n"],
    description="价格围绕中枢穿越频率高（穿越次数 >= 3）",
    direction="neutral",
    is_core=True,
)

register_event(
    name="evt_volatility_compress",
    category="趋势位置事件",
    detect_func=_detect_volatility_compress,
    required_factors=["bbmacd_bandwidth_zscore"],
    description="波动压缩（bbmacd_bandwidth_zscore < -1）",
    direction="neutral",
    is_core=False,
)

register_event(
    name="evt_lower_reclaim_freq_high",
    category="趋势位置事件",
    detect_func=_detect_lower_reclaim_freq_high,
    required_factors=["lower_reclaim_count_n"],
    description="下沿多次收回（reclaim_count >= 2）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_upper_reject",
    category="趋势位置事件",
    detect_func=_detect_upper_reject,
    required_factors=["high", "close", "stage_upper_boundary"],
    description="上沿试盘失败（最高价突破上一bar上沿但收盘回落）",
    direction="neutral",
    is_core=True,
)

register_event(
    name="evt_lower_touch",
    category="趋势位置事件",
    detect_func=_detect_lower_touch,
    required_factors=["low", "stage_lower_boundary"],
    description="最低价触及上一bar下沿",
    direction="neutral",
    is_core=True,
)
