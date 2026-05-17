# -*- coding: utf-8 -*-
"""
event_lib/detectors/stage_cost_zone_events.py - 成本区事件检测

Purpose: 基于阶段上下文/位置/成熟度因子列检测成本区形成/成熟事件。

Registered Events:
    - evt_cz_trend_flat: 趋势进入低斜率/低方向状态
    - evt_cz_price_around_mid: 价格围绕中枢反复交换
    - evt_cz_volatility_compress: 波动压缩
    - evt_cz_lower_hold: 下沿多次收回
    - evt_cz_mature: 成本区成熟综合事件
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_cz_trend_flat(factors_df: pd.DataFrame) -> pd.Series:
    if "trend_flat_score" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["trend_flat_score"] > 0.7).astype(int)


def _detect_cz_price_around_mid(factors_df: pd.DataFrame) -> pd.Series:
    if "price_cross_mid_count_n" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["price_cross_mid_count_n"] >= 3).astype(int)


def _detect_cz_volatility_compress(factors_df: pd.DataFrame) -> pd.Series:
    if "bbmacd_bandwidth_zscore" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["bbmacd_bandwidth_zscore"] < -1).astype(int)


def _detect_cz_lower_hold(factors_df: pd.DataFrame) -> pd.Series:
    if "lower_reclaim_count_n" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["lower_reclaim_count_n"] >= 2).astype(int)


def _detect_cz_mature(factors_df: pd.DataFrame) -> pd.Series:
    if "cost_zone_maturity_score" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["cost_zone_maturity_score"] > 0.6).astype(int)


register_event(
    name="evt_cz_trend_flat",
    category="成本区事件",
    detect_func=_detect_cz_trend_flat,
    required_factors=["trend_flat_score"],
    description="趋势进入低斜率/低方向状态（trend_flat_score > 0.7）",
    direction="neutral",
    is_core=True,
)

register_event(
    name="evt_cz_price_around_mid",
    category="成本区事件",
    detect_func=_detect_cz_price_around_mid,
    required_factors=["price_cross_mid_count_n"],
    description="价格围绕中枢反复交换（穿越次数 >= 3）",
    direction="neutral",
    is_core=True,
)

register_event(
    name="evt_cz_volatility_compress",
    category="成本区事件",
    detect_func=_detect_cz_volatility_compress,
    required_factors=["bbmacd_bandwidth_zscore"],
    description="波动压缩（bbmacd_bandwidth_zscore < -1）",
    direction="neutral",
    is_core=False,
)

register_event(
    name="evt_cz_lower_hold",
    category="成本区事件",
    detect_func=_detect_cz_lower_hold,
    required_factors=["lower_reclaim_count_n"],
    description="下沿多次收回（reclaim_count >= 2）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_cz_mature",
    category="成本区事件",
    detect_func=_detect_cz_mature,
    required_factors=["cost_zone_maturity_score"],
    description="成本区成熟综合事件（maturity_score > 0.6，依赖stage_engine输出）",
    direction="positive",
    is_core=True,
)
