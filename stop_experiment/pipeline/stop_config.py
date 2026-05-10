# -*- coding: utf-8 -*-
"""
Stop-Loss Clustering 策略实验参数配置

Purpose:
    SLC策略GBDT实验的全局参数定义。所有脚本通过 import 引用，确保参数一致性。

Pipeline Position:
    共享配置（被所有脚本 import）。
    上游: —
    下游: 所有 pipeline/backtest/experiments 脚本

Inputs:
    - 无（纯配置常量）

Outputs:
    - 无（被 import 使用）

How to Run:
    无需单独运行，被其他脚本 import

Side Effects:
    - 无
"""

# ==================== 观察期 ====================
OBS_DAYS = 20              # 信号触发后的观察期天数（最长持有观察天数）

# ==================== 交易成本 ====================
BUY_COST = 0.0005          # 买入成本（佣金万五）
SELL_COST = 0.0010         # 卖出成本（佣金万五 + 印花税千一）

# ==================== 模型训练 ====================
EMBARGO_DAYS = 25          # 时间序列切分 embargo 天数
N_FOLDS = 3                # 滚动折数

# ==================== 分类阈值 ====================
SELL_CLS_THRESHOLD = 0.07  # 卖点分类阈值：mfe_20 > 7% → 还有上涨空间（不卖）
BUY_CLS_THRESHOLD = -0.07    # Phase 2: 从 -0.05 收紧到 -0.07

# ==================== 数据分割 ====================
# 3年数据 train/val/test 分割（基于 obs_date，2026-05 更新）
# 数据范围: 2023-03 ~ 2026-05，信号 53K+ vs 旧 14K
OBS_TRAIN_END = "2025-06-30"   # train: obs_date <= OBS_TRAIN_END - EMBARGO_DAYS  (~2023-03 ~ 2025-06)
OBS_VAL_END = "2025-12-31"     # val:   OBS_TRAIN_END < obs_date <= OBS_VAL_END   (~2025-07 ~ 2025-12)
                                # test:  obs_date > OBS_VAL_END                    (~2026-01 ~ 2026-05)

# 旧口径常量（保留用于审计对比，仅 lookahead_bias_check.py 等诊断脚本使用）
TRAIN_END = "2025-11-30"       # 旧口径: selection_date <= TRAIN_END
VAL_END = "2026-02-28"         # 旧口径: selection_date <= VAL_END

# ==================== LightGBM 超参数 ====================
LGB_PARAMS = {
    "num_leaves": 16,
    "max_depth": 5,
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "verbosity": -1,
}

# ==================== 4个模型变体 ====================
MODEL_SPECS = {
    "sell_reg": {"target": "mfe_20", "objective": "regression", "metric": "mae"},
    "sell_cls": {"target": "sell_signal", "objective": "binary", "metric": "auc"},
    "buy_reg": {"target": "mae_20", "objective": "regression", "metric": "mae"},
    "buy_cls": {"target": "buy_signal", "objective": "binary", "metric": "auc"},
}

# ==================== 路径 ====================
import os

STOP_EXPERIMENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(STOP_EXPERIMENT_ROOT, "output")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.parquet")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models_control")       # baseline_3y_control_v1
MODELS_TREATMENT_DIR = os.path.join(OUTPUT_DIR, "models")     # VSA版（研究保留）
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
BACKTEST_DIR = os.path.join(OUTPUT_DIR, "backtest")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
HOLDINGS_DIR = os.path.join(OUTPUT_DIR, "holdings")
LIVE_DIR = os.path.join(OUTPUT_DIR, "live")
DECISIONS_DIR = os.path.join(LIVE_DIR, "decisions")
EXECUTIONS_DIR = os.path.join(LIVE_DIR, "executions")

# ==================== VSA 因子开关 ====================
VSA_ENABLED = False  # 默认关闭，研究需要时改为 True

# ==================== 模型退出默认参数 (Phase 1.5 验证) ====================
MODEL_EXIT_PARAMS = {
    "buy_cls_exit_threshold": 0.7,
    "stop_loss": -0.07,
    "max_hold_days": 20,
    "candidate_obs_days": [1, 2, 3],
}

# ==================== 版本基线 (Phase 3 Frozen: 2026-05) ====================
# 冻结版本参数，后续所有实验/对比均基于此基线
# 对应 buy_signal=-7% + obs_day=1~3 + model_exit + stop_loss=-7% + max_hold=20
V1_BASELINE = "v1_frozen_202605"

V1_PARAMS = {
    "buy_signal_threshold": -0.07,
    "candidate_obs_days": [1, 2, 3],
    "buy_cls_exit_threshold": 0.70,
    "stop_loss": -0.07,
    "max_hold_days": 20,
    "max_stocks_default": 10,
    "strategy_default": "sell_score",  # 实际 = pred_sell_reg（score_stocks）
}

# ==================== 新基线 (Phase 4: 3年Control, 2026-05-10 冻结) ====================
BASELINE_3Y_CONTROL_V1 = "baseline_3y_control_v1"

BASELINE_3Y_PARAMS = {
    "dataset_version": "3y_batched_v2",
    "data_range": "2023-03 ~ 2026-05",
    "train_end": "2025-06-30",
    "val_end": "2025-12-31",
    "test_range": "2026-01 ~ 2026-05",
    "signals_count": 54678,
    "feature_set_name": "control_no_vsa",
    "vsa_enabled": False,
    "feature_version": "baseline_3y_v1",
    "model_dir": "models_control",
    "buy_signal_threshold": -0.07,
    "candidate_obs_days": [1, 2, 3],
    "buy_cls_exit_threshold": 0.70,
    "stop_loss": -0.07,
    "max_hold_days": 20,
    "max_stocks_default": 10,
    "strategy_default": "sell_score",  # 实际 = pred_sell_reg（score_stocks）
}
