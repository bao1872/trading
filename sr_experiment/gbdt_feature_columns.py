# -*- coding: utf-8 -*-
"""
SR 事件因子 GBDT 特征列统一定义（SSOT）

Purpose: 所有 GBDT 实验脚本统一从此文件导入特征列名，禁止在各自脚本中重复定义
Inputs:  无（纯列名定义）
Outputs: 无（被 import 使用）
How to Run: 无需单独运行
Side Effects: 无
"""
from __future__ import annotations

CLUSTER_COLS = [
    "support_cluster_count",
    "support_cluster_score",
    "support_cluster_density",
    "support_cluster_is_strong",
    "support_confluence_score",
    "support_confluence_is_strong",
    "resistance_cluster_count",
    "resistance_cluster_score",
    "resistance_cluster_is_strong",
    "resistance_confluence_score",
    "resistance_confluence_is_strong",
    "support_zone_low",
    "support_zone_high",
    "resistance_zone_low",
    "resistance_zone_high",
]

SUPPORT_R2S_COLS = [
    "is_support_flipped",
    "flipped_support_age_bars",
    "support_gap_pct",
    "support_age_bars",
    "support_is_higher_low",
    "support_touch_count_20",
    "support_touch_count_60",
    "support_is_overused",
    "support_is_fresh",
]

PIERCE_RECLAIM_COLS = [
    "support_pierce_depth_pct",
    "support_pierce_depth_atr",
    "support_pierce_depth_sr",
    "support_reclaim_strength_pct",
    "support_reclaim_strength_atr",
    "support_reclaim_strength_sr",
    "close_pos_in_bar",
    "lower_shadow_pct",
    "upper_shadow_pct",
    "body_pct",
    "bar_range_atr",
]

VOLUME_COLS = [
    "volume_ratio_20",
    "volume_z_20",
    "amount_ratio_20",
    "amount_z_20",
    "is_volume_expansion",
    "is_volume_shrink",
]

TREND_POSITION_COLS = [
    "sr_pos_01",
    "sr_pos_raw",
    "ret_5",
    "ret_10",
    "ret_20",
    "ma20_slope_pct",
    "ma60_slope_pct",
    "close_above_ma20",
    "close_above_ma60",
    "trend_ma_bull",
    "trend_ma_bear",
    "atr_pct_14",
    "realized_vol_20",
    "max_drawdown_20",
]

EVENT_BOOL_COLS = [
    "evt_pierce_support_reclaim",
    "evt_pierce_strong_support_cluster_reclaim",
    "evt_pierce_support_cluster_reclaim_low_volume",
    "evt_pierce_flipped_support_reclaim",
    "evt_break_strong_support_cluster",
    "evt_wick_break_resistance_cluster_fail",
    "evt_close_above_resistance_cluster_upper",
]

ALL_FEATURE_COLS = (
    CLUSTER_COLS + SUPPORT_R2S_COLS + PIERCE_RECLAIM_COLS
    + VOLUME_COLS + TREND_POSITION_COLS + EVENT_BOOL_COLS
)

FACTOR_CATEGORIES = {
    "结构簇": CLUSTER_COLS,
    "支撑/R2S": SUPPORT_R2S_COLS,
    "刺破/收回": PIERCE_RECLAIM_COLS,
    "量能": VOLUME_COLS,
    "趋势/位置": TREND_POSITION_COLS,
    "事件bool": EVENT_BOOL_COLS,
}

LABEL_COLS = [
    "fwd_ret_1", "fwd_ret_3", "fwd_ret_5", "fwd_ret_10", "fwd_ret_20",
    "fwd_mdd_5", "fwd_mdd_10", "fwd_mdd_20",
    "fwd_max_ret_5", "fwd_max_ret_10", "fwd_max_ret_20",
    "fwd_reward_risk_5", "fwd_reward_risk_10", "fwd_reward_risk_20",
]

PATH_LABEL_COLS = [
    "label_tp8_sl6_20",
    "label_tp10_sl8_20",
    "label_low_mdd_20",
    "label_bad_break_10",
    "tp_hit_bar",
    "sl_hit_bar",
]

PATH_INPUT_COLS = ["close", "low", "high", "active_support_ref"]

META_COLS = ["ts_code", "bar_time"]

ALL_DATASET_COLS = list(dict.fromkeys(
    META_COLS + ALL_FEATURE_COLS + LABEL_COLS + PATH_INPUT_COLS
))
