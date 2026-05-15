# -*- coding: utf-8 -*-
"""
竞价拍卖 GBDT 实验特征列名统一定义（SSOT）

Purpose:
    bid_experiment 所有脚本统一从此文件导入特征列名，禁止在各自脚本中重复定义。

Inputs:
    - 无（纯列名定义）

Outputs:
    - 无（被 import 使用）

How to Run:
    python -m bid_experiment.feature_columns

Side Effects:
    - 无
"""

# ==================== 第一组：竞价最终状态特征 ====================
AUCTION_STATE_COLS = [
    "auc_ret_close",
    "auc_ret_open_ref",
    "gap_flag_up",
    "gap_flag_down",
    "auction_vs_ma5",
    "auction_vs_ma10",
    "auction_vs_prev_high",
    "auction_vs_prev_low",
    "dist_to_prev_close_atr",
    "auc_ret_close_pct_60d",
    "auc_ret_close_z_60d",
]

# ==================== 第二组：竞价量能特征 ====================
AUCTION_VOLUME_COLS = [
    "auc_volume",
    "auc_amount",
    "auc_turnover",
    "auc_amount_vs_prev",
    "auc_amount_vs_mean_20d",
    "auc_volume_vs_mean_20d",
    "auc_amount_pct_60d",
    "auc_amount_z_60d",
    "auc_turnover_pct_60d",
    "auc_ret_x_auc_amount_z",
    "sign_auc_ret_x_auc_amount_pct",
]

# ==================== 第三组：竞价过程特征（仅价格过程） ====================
AUCTION_PROCESS_COLS = [
    "auc_price_slope_920_925",
    "auc_price_slope_last_1m",
    "auc_price_range_auction",
    "auc_price_final_vs_peak",
    "auc_price_final_vs_low",
    "auc_price_final_vs_920",
    "auc_ref_price_count",
]

# ==================== 第四组：前一日背景先验特征 ====================
BACKGROUND_COLS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "close_pos_20d",
    "distance_to_20d_high",
    "distance_to_20d_low",
    "atr_pct",
    "realized_vol_5d",
    "turnover_avg_5d",
    "amount_avg_5d",
    "prev_day_body_pct",
    "prev_day_upper_shadow_pct",
    "prev_day_lower_shadow_pct",
    "prev_day_close_in_range",
]

# ==================== 第五组：环境特征 ====================
ENV_COLS = [
    "index_auc_ret",
]

# ==================== 合并特征列表 ====================
ALL_FEATURE_COLS = (
    AUCTION_STATE_COLS
    + AUCTION_VOLUME_COLS
    + AUCTION_PROCESS_COLS
    + BACKGROUND_COLS
    + ENV_COLS
)

# ==================== 元信息列 ====================
META_COLS = [
    "stock_id",
    "trade_date",
]

# ==================== 标签列 ====================
LABEL_COLS = [
    "entry_price",
    "exit_price",
    "MFE_10m",
    "MAE_10m",
    "RET_10m",
    "RET_30m",
    "future_min_ret_15m",
    "y_buy_now",
    "y_sell_now",
]

# ==================== 特征分组映射 ====================
FACTOR_CATEGORIES = {
    "auction_state": AUCTION_STATE_COLS,
    "auction_volume": AUCTION_VOLUME_COLS,
    "auction_process": AUCTION_PROCESS_COLS,
    "background": BACKGROUND_COLS,
    "env": ENV_COLS,
}

if __name__ == "__main__":
    print("=== Feature Column Groups ===")
    for name, cols in FACTOR_CATEGORIES.items():
        print(f"  {name}: {len(cols)} columns")
    print(f"  META_COLS: {len(META_COLS)} columns")
    print(f"  LABEL_COLS: {len(LABEL_COLS)} columns")
    print(f"  ALL_FEATURE_COLS: {len(ALL_FEATURE_COLS)} columns")
