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

# ==================== 候选池 obs_day 范围 ====================
CANDIDATE_OBS_DAYS = [1, 2, 3]  # 生产候选池 obs_day 范围（原 [1]，2026-05-13 扩展为 [1,2,3]）

# ==================== K线 warm-up / forward ====================
FACTOR_WARMUP_DAYS = 600        # 因子计算K线回看天数（约2.4年，覆盖DSA/BBMacd完整warm-up）
FACTOR_FORWARD_DAYS = OBS_DAYS + 50  # 标签计算前瞻天数

# ==================== 交易成本 ====================
BUY_COST = 0.0005          # 买入成本（佣金万五）
SELL_COST = 0.0010         # 卖出成本（佣金万五 + 印花税千一）

# ==================== 模型训练 ====================
EMBARGO_DAYS = 25          # 时间序列切分 embargo 天数
N_FOLDS = 3                # 滚动折数

# ==================== 分类阈值 ====================
# 训练侧标签阈值，不等于生产 Exit 阈值。
# sell_cls: mfe_20 > 7% → 还有上涨空间（不卖）
# buy_cls:  mae_20 < -7% → 买点信号（此处 -0.07 是训练标签定义，
#           生产 Exit 阈值 buy_cls_exit_threshold=0.70 见 BASELINE_V2_PARAMS）
SELL_CLS_THRESHOLD = 0.07
BUY_CLS_THRESHOLD = -0.07

# ==================== 数据分割 ====================
# 3年数据 train/val/test 分割（基于 obs_date，2026-05 更新）
# 数据范围: 2023-03 ~ 2026-05，信号 74K+，4929 只股票
OBS_TRAIN_END = "2025-06-30"   # train: obs_date <= OBS_TRAIN_END - EMBARGO_DAYS  (~2023-03 ~ 2025-06)
OBS_VAL_END = "2025-12-31"     # val:   OBS_TRAIN_END < obs_date <= OBS_VAL_END   (~2025-07 ~ 2025-12)
                                # test:  obs_date > OBS_VAL_END                    (~2026-01 ~ 2026-05)

# ==================== LightGBM 超参数 ====================
LGB_PARAMS = {
    "num_leaves": 16,
    "max_depth": 5,
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_bin": 63,
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
MODELS_TREATMENT_DIR = os.path.join(OUTPUT_DIR, "models")     # VSA版（研究保留，目录当前不存在）
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
BACKTEST_DIR = os.path.join(OUTPUT_DIR, "backtest")
BACKTEST_LEDGER_DIR = os.path.join(OUTPUT_DIR, "backtest_ledger")
REPLAY_LEDGER_DIR = os.path.join(OUTPUT_DIR, "replay_ledger")  # DEPRECATED: replay 模式已移除
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")      # DEPRECATED: 仅兼容，新数据请使用 prediction_store
HOLDINGS_DIR = os.path.join(OUTPUT_DIR, "holdings")            # DEPRECATED: 仅回测后处理兼容
LIVE_DIR = os.path.join(OUTPUT_DIR, "live")                    # DEPRECATED: live 模式已移除
DECISIONS_DIR = os.path.join(LIVE_DIR, "decisions")            # DEPRECATED: live 模式已移除
EXECUTIONS_DIR = os.path.join(LIVE_DIR, "executions")          # DEPRECATED: live 模式已移除

# ==================== VSA 因子开关 ====================
VSA_ENABLED = False  # 默认关闭，研究需要时改为 True

# ==================== 生产基线 (3年全量数据训练, 2026-05-11 冻结) ====================
BASELINE_V2 = "baseline_v2_3y"

BASELINE_V2_PARAMS = {
    "profile": BASELINE_V2,
    "description": "3年全量数据训练 (4929只股票/74374条信号) + E0 Entry + X1 Exit",
    "max_stocks": 10,
    "score_col": "pred_sell_reg",
    "exit_mode": "model_exit",
    "candidate_obs_days": [1, 2, 3],
    "buy_cls_exit_threshold": 0.70,
    "stop_loss": -0.07,
    "max_hold_days": 20,
    "buy_cost": 0.001,
    "sell_cost": 0.001,
    "entry_gate_sell_cls": None,
    "entry_gate_buy_cls": None,
    "exit_sub_mode": None,
    "buy_reg_exit_threshold": None,
    "frozen_at": "2026-05-11",
    "expected_nav": 6.1722,
    "expected_sharpe": 14.77,
    "expected_mdd": -0.0671,
    "expected_n_trades": 59,
}

PRODUCTION_PARAMS = BASELINE_V2_PARAMS


# ==================== 候选池过滤（SSOT） ====================

def filter_production_candidates(df):
    """生产候选池过滤（SSOT）：obs_day ∈ CANDIDATE_OBS_DAYS + (ts_code, obs_date) 去重保留 obs_day 最小。

    所有需要生产口径候选池的脚本（07/dynamic_exit_backtest_v2/generate_full_predictions/实验）
    必须调用此函数，禁止内联重写过滤逻辑。
    """
    import pandas as pd
    df = df[df["obs_day"].isin(CANDIDATE_OBS_DAYS)].copy()
    df = df.sort_values("obs_day").drop_duplicates(
        subset=["ts_code", "obs_date"],
        keep="first",
    )
    return df
