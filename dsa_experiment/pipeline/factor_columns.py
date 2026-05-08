# -*- coding: utf-8 -*-
"""
因子列名统一定义文件

Purpose: 所有 DSA pipeline 脚本统一从此文件导入因子列名，禁止在各自脚本中重复定义
How to Run: 无需单独运行，被其他脚本 import

列名规范:
    ALL_DAILY_FACTOR_COLS  — 全部日线因子（42列），04_build_daily_factor_table 使用
    WEEKLY_FEATURE_COLS    — 周线模型特征列（35列），07_daily_trading_sheet 周线部分使用
    DAILY_SELECT_FACTORS   — 日线模型特征列（27列），07_daily_trading_sheet 日线部分使用
    FACTOR_COLUMNS         — V型反转表因子列（38列），01_selection_dsa 写入 stock_dsa_vreversal_results 使用
"""

# 因子24 + 量能8 + 协同7 + 派生3 = 42 列
ALL_DAILY_FACTOR_COLS = [
    "dsa_dir", "prev_pivot_code", "last_confirmed_high", "last_confirmed_low",
    "dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct", "current_stage_bars", "prev_stage_bars",
    "bars_since_last_high", "bars_since_last_low",
    "prev_stage_amp_pct", "current_stage_ret_pct", "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct",
    "bbmacd", "bbmacd_minus_avg", "bbmacd_state", "bbmacd_band_pos_01",
    "bbmacd_bandwidth_zscore", "bbmacd_cross_upper", "bbmacd_cross_lower",
    "trend_align_momo",
    "vol_zscore_5", "vol_zscore_10", "vol_zscore_20",
    "vol_ratio_10", "days_since_vol_spike",
    "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio",
    "dsa_dir_age", "bbmacd_sign", "bbmacd_slope_3",
    "price_vol_coord", "momo_vol_coord", "low_pos_break_coord", "coord_consistency",
    "coord_stage_current", "coord_stage_prev", "coord_stage_ratio",
]

# 周线模型特征（ALL_DAILY_FACTOR_COLS 去掉 dsa_dir_age, bbmacd_sign, bbmacd_slope_3,
# days_since_vol_spike = 42 - 4 = 38。包含 low_pos_break_coord（06_weekly_selector 实际使用）
WEEKLY_FEATURE_COLS = [
    "dsa_dir", "prev_pivot_code", "last_confirmed_high", "last_confirmed_low",
    "dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct", "current_stage_bars", "prev_stage_bars",
    "bars_since_last_high", "bars_since_last_low",
    "prev_stage_amp_pct", "current_stage_ret_pct", "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct",
    "bbmacd", "bbmacd_minus_avg", "bbmacd_state", "bbmacd_band_pos_01",
    "bbmacd_bandwidth_zscore", "bbmacd_cross_upper", "bbmacd_cross_lower",
    "trend_align_momo",
    "vol_zscore_5", "vol_zscore_10", "vol_zscore_20",
    "vol_ratio_10",
    "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio",
    "price_vol_coord", "momo_vol_coord", "low_pos_break_coord", "coord_consistency",
    "coord_stage_current", "coord_stage_prev", "coord_stage_ratio",
]

# 日线精选因子（27列，WEEKLY_FEATURE_COLS 的子集）
DAILY_SELECT_FACTORS = [
    "dsa_dir", "price_vs_dsa_vwap_pct", "bbmacd_sign", "bbmacd_slope_3",
    "dsa_pivot_pos_01", "ret_to_last_low_pct", "bars_since_last_high",
    "bbmacd", "bbmacd_minus_avg", "bbmacd_bandwidth_zscore",
    "vol_zscore_20", "vol_stage_cv", "vol_zscore_10", "vol_ratio_10",
    "days_since_vol_spike", "current_stage_amp_pct", "prev_stage_amp_pct",
    "current_stage_ret_pct", "prev_pivot_code", "trend_align_momo",
    "price_vol_coord", "momo_vol_coord", "low_pos_break_coord",
    "coord_consistency", "coord_stage_current", "coord_stage_prev", "coord_stage_ratio",
]

# V型反转表因子列（38列，ALL_DAILY_FACTOR_COLS 去掉 dsa_dir_age, bbmacd_sign, bbmacd_slope_3, days_since_vol_spike）
FACTOR_COLUMNS = [
    "dsa_dir", "prev_pivot_code", "last_confirmed_high", "last_confirmed_low",
    "dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct", "current_stage_bars", "prev_stage_bars",
    "bars_since_last_high", "bars_since_last_low",
    "prev_stage_amp_pct", "current_stage_ret_pct", "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct",
    "bbmacd", "bbmacd_minus_avg", "bbmacd_state", "bbmacd_band_pos_01",
    "bbmacd_bandwidth_zscore", "bbmacd_cross_upper", "bbmacd_cross_lower",
    "trend_align_momo",
    "vol_zscore_5", "vol_zscore_10", "vol_zscore_20",
    "vol_ratio_10",
    "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio",
    "price_vol_coord", "momo_vol_coord", "low_pos_break_coord", "coord_consistency",
    "coord_stage_current", "coord_stage_prev", "coord_stage_ratio",
]
