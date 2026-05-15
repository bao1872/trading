# -*- coding: utf-8 -*-
"""
竞价拍卖 GBDT 实验全局配置

Purpose:
    竞价拍卖策略GBDT实验的全局参数定义。所有脚本通过 import 引用，确保参数一致性。

Inputs:
    - 无（纯配置常量）

Outputs:
    - 无（被 import 使用）

How to Run:
    python bid_experiment/bid_config.py

Side Effects:
    - 无
"""

import os

# ==================== 路径 ====================
BID_EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BID_EXPERIMENT_ROOT, "output")
RAW_DATA_DIR = os.path.join(OUTPUT_DIR, "raw_data")
DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# ==================== 标签阈值 ====================
BUY_NOW_MFE_THRESHOLD = 0.01
BUY_NOW_MAE_THRESHOLD = -0.006
BUY_NOW_RET_THRESHOLD = 0.0
SELL_NOW_MIN_RET_THRESHOLD = -0.01
SELL_NOW_RET_30M_THRESHOLD = -0.003

# ==================== 样本筛选阈值 ====================
BUY_MIN_AUC_RET_ABS = 0.005
BUY_MIN_AUC_AMOUNT_RATIO = 0.5

# ==================== 模型训练 ====================
EMBARGO_DAYS = 25
OBS_TRAIN_END = "2025-06-30"
OBS_VAL_END = "2025-12-31"

LGB_PARAMS = {
    "num_leaves": 16,
    "max_depth": 5,
    "min_data_in_leaf": 30,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_bin": 63,
    "seed": 42,
    "verbosity": -1,
}

MODEL_SPECS = {
    "buy_now": {"target": "y_buy_now", "objective": "binary", "metric": "auc"},
    "sell_now": {"target": "y_sell_now", "objective": "binary", "metric": "auc"},
}

# ==================== 滚动窗口 ====================
ROLLING_WINDOW_60D = 60
ROLLING_WINDOW_20D = 20

# ==================== 交易成本 ====================
BUY_COST = 0.0005
SELL_COST = 0.0010

# ==================== pytdx 竞价数据标识 ====================
BUYORSELL_AUCTION_REF = 8
BUYORSELL_AUCTION_MATCH = 2
BUYORSELL_BUY = 0
BUYORSELL_SELL = 1

# ==================== 指数代码 ====================
INDEX_CODE_SH = "000001"
INDEX_CODE_SZ = "399001"


if __name__ == "__main__":
    sections = {
        "路径": [
            ("BID_EXPERIMENT_ROOT", BID_EXPERIMENT_ROOT),
            ("OUTPUT_DIR", OUTPUT_DIR),
            ("RAW_DATA_DIR", RAW_DATA_DIR),
            ("DATASET_DIR", DATASET_DIR),
            ("MODELS_DIR", MODELS_DIR),
            ("FIGURES_DIR", FIGURES_DIR),
        ],
        "标签阈值": [
            ("BUY_NOW_MFE_THRESHOLD", BUY_NOW_MFE_THRESHOLD),
            ("BUY_NOW_MAE_THRESHOLD", BUY_NOW_MAE_THRESHOLD),
            ("BUY_NOW_RET_THRESHOLD", BUY_NOW_RET_THRESHOLD),
            ("SELL_NOW_MIN_RET_THRESHOLD", SELL_NOW_MIN_RET_THRESHOLD),
            ("SELL_NOW_RET_30M_THRESHOLD", SELL_NOW_RET_30M_THRESHOLD),
        ],
        "样本筛选阈值": [
            ("BUY_MIN_AUC_RET_ABS", BUY_MIN_AUC_RET_ABS),
            ("BUY_MIN_AUC_AMOUNT_RATIO", BUY_MIN_AUC_AMOUNT_RATIO),
        ],
        "模型训练": [
            ("EMBARGO_DAYS", EMBARGO_DAYS),
            ("OBS_TRAIN_END", OBS_TRAIN_END),
            ("OBS_VAL_END", OBS_VAL_END),
            ("LGB_PARAMS", LGB_PARAMS),
            ("MODEL_SPECS", MODEL_SPECS),
        ],
        "滚动窗口": [
            ("ROLLING_WINDOW_60D", ROLLING_WINDOW_60D),
            ("ROLLING_WINDOW_20D", ROLLING_WINDOW_20D),
        ],
        "交易成本": [
            ("BUY_COST", BUY_COST),
            ("SELL_COST", SELL_COST),
        ],
        "pytdx 竞价数据标识": [
            ("BUYORSELL_AUCTION_REF", BUYORSELL_AUCTION_REF),
            ("BUYORSELL_AUCTION_MATCH", BUYORSELL_AUCTION_MATCH),
            ("BUYORSELL_BUY", BUYORSELL_BUY),
            ("BUYORSELL_SELL", BUYORSELL_SELL),
        ],
        "指数代码": [
            ("INDEX_CODE_SH", INDEX_CODE_SH),
            ("INDEX_CODE_SZ", INDEX_CODE_SZ),
        ],
    }
    for section, items in sections.items():
        print(f"\n===== {section} =====")
        for name, value in items.items() if False else [(k, v) for k, v in items]:
            print(f"  {name} = {value!r}")
