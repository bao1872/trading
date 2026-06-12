SYMBOL = "300133"
FREQ = "w"
PIVOT_LEN = 10
BARS = 300
FETCH_BARS = 1200
OUT_DIR = "sr_experiment/results"
DATA_SOURCE = "db"
AMOUNT_FILL_METHOD = "volume_close"
EVENT_COLS = [
    "evt_pierce_support_reclaim",
    "evt_pierce_pivot_support_reclaim",
    "evt_pierce_flipped_support_reclaim",
    "evt_pierce_active_support_reclaim",
    "evt_failed_reclaim_support",
    "evt_failed_reclaim_pivot_support",
    "evt_failed_reclaim_flipped_support",
    "evt_failed_reclaim_active_support",
    "evt_close_break_recent_support",
    "evt_retest_flipped_support",
    "evt_clean_hold_flipped_support",
    "evt_breakdown_flipped_support",
    "evt_cross_recent_resistance",
    "evt_wick_break_resistance_fail",
    "evt_break_resistance_from_low_zone",
    "evt_pierce_strong_support_cluster_reclaim",
    "evt_pierce_support_zone_reclaim",
    "evt_pierce_support_zone_reclaim_strong",
    "evt_break_strong_support_cluster",
    "evt_break_strong_support_cluster_high_volume",
    "evt_break_strong_resistance_cluster",
    "evt_wick_break_resistance_cluster_fail",
    "evt_close_above_resistance_cluster_upper",
    "evt_pierce_support_cluster_reclaim_low_volume",
    "evt_pierce_support_cluster_reclaim_high_volume",
]
FWD_RET_COLS = ["fwd_ret_1", "fwd_ret_3", "fwd_ret_5", "fwd_ret_10", "fwd_ret_20"]
FWD_MDD_COLS = ["fwd_mdd_5", "fwd_mdd_10", "fwd_mdd_20"]
FWD_MAX_COLS = ["fwd_max_ret_5", "fwd_max_ret_10", "fwd_max_ret_20"]
FWD_RR_COLS = ["fwd_reward_risk_5", "fwd_reward_risk_10", "fwd_reward_risk_20"]

PANEL_EVENT_STAT_COLS = (
    EVENT_COLS + FWD_RET_COLS + FWD_MDD_COLS + FWD_MAX_COLS + FWD_RR_COLS
)

PANEL_GROUP_STAT_COLS = (
    EVENT_COLS + FWD_RET_COLS + FWD_MDD_COLS + FWD_MAX_COLS + FWD_RR_COLS
    + ["sr_pos_01", "support_pierce_depth_atr", "support_reclaim_strength_atr",
       "volume_z_20", "trend_ma_bull", "trend_ma_bear", "is_support_flipped",
       "close_pos_in_bar", "is_volume_expansion", "is_volume_shrink",
       "is_long_lower_shadow", "support_is_higher_low", "flipped_support_age_bars",
       "support_cluster_count", "support_cluster_score", "support_cluster_is_strong",
       "support_confluence_score", "support_confluence_is_strong",
       "resistance_cluster_count", "resistance_cluster_score", "resistance_cluster_is_strong",
       "resistance_confluence_score", "resistance_confluence_is_strong"]
)
