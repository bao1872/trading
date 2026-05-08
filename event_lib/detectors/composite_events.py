# -*- coding: utf-8 -*-
"""
event_lib/detectors/composite_events.py - 复合事件检测

Purpose: 基于多个因子列检测复合事件（多因子共振）。

Registered Events:
    - evt_trend_flip_with_volume: 趋势翻转+放量确认
    - evt_low_with_vol_shrink: 低点+缩量止跌
    - evt_momo_accel_with_vol: 动量加速+量能配合
    - evt_coord_breakout: 协同向上突破
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_trend_flip_with_volume(factors_df: pd.DataFrame) -> pd.Series:
    """趋势翻转+放量确认：dsa_dir翻转且vol_zscore > 2。"""
    if "dsa_dir" not in factors_df.columns or "vol_zscore_20" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    dsa_dir = factors_df["dsa_dir"]
    vol_zscore = factors_df["vol_zscore_20"]
    flip_up = (dsa_dir == 1) & (dsa_dir.shift(1) != 1)
    flip_down = (dsa_dir == -1) & (dsa_dir.shift(1) != -1)
    vol_confirm = vol_zscore > 2
    return ((flip_up | flip_down) & vol_confirm).astype(int)


def _detect_low_with_vol_shrink(factors_df: pd.DataFrame) -> pd.Series:
    """低点+缩量止跌：pivot_pos < 0.2 且 vol_zscore < -1 且价格稳定。"""
    if "dsa_pivot_pos_01" not in factors_df.columns or "vol_zscore_20" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    low_pos = factors_df["dsa_pivot_pos_01"] < 0.2
    vol_shrink = factors_df["vol_zscore_20"] < -1
    price_stable = factors_df["close"].diff() >= 0 if "close" in factors_df.columns else pd.Series(True, index=factors_df.index)
    return (low_pos & vol_shrink & price_stable).astype(int)


def _detect_momo_accel_with_vol(factors_df: pd.DataFrame) -> pd.Series:
    """动量加速+量能配合：bbmacd_slope > 0 且 vol_zscore > 1。"""
    if "bbmacd_slope_3" not in factors_df.columns or "vol_zscore_20" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    momo_accel = factors_df["bbmacd_slope_3"] > 0
    vol_support = factors_df["vol_zscore_20"] > 1
    return (momo_accel & vol_support).astype(int)


def _detect_coord_breakout(factors_df: pd.DataFrame) -> pd.Series:
    """协同向上突破：coord_stage_current > 0 且 bbmacd_cross_upper。"""
    if "coord_stage_current" not in factors_df.columns or "bbmacd_cross_upper" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    coord_up = factors_df["coord_stage_current"] > 0
    bb_cross = factors_df["bbmacd_cross_upper"] == 1
    return (coord_up & bb_cross).astype(int)


# 注册复合事件
register_event(
    name="evt_trend_flip_with_volume",
    category="复合事件",
    detect_func=_detect_trend_flip_with_volume,
    required_factors=["dsa_dir", "vol_zscore_20"],
    description="趋势翻转+放量确认（dsa_dir翻转且vol_zscore>2）",
    direction="positive",
    is_core=True,
)

register_event(
    name="evt_low_with_vol_shrink",
    category="复合事件",
    detect_func=_detect_low_with_vol_shrink,
    required_factors=["dsa_pivot_pos_01", "vol_zscore_20", "close"],
    description="低点+缩量止跌（pivot_pos<0.2且vol_zscore<-1且价格稳定）",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_momo_accel_with_vol",
    category="复合事件",
    detect_func=_detect_momo_accel_with_vol,
    required_factors=["bbmacd_slope_3", "vol_zscore_20"],
    description="动量加速+量能配合（bbmacd_slope>0且vol_zscore>1）",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_coord_breakout",
    category="复合事件",
    detect_func=_detect_coord_breakout,
    required_factors=["coord_stage_current", "bbmacd_cross_upper"],
    description="协同向上突破（coord_stage>0且bbmacd_cross_upper）",
    direction="positive",
    is_core=False,
)
