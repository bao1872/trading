#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量 test 集预测生成（final 模型）

Purpose:
    读取训练产出的 candidate_with_scores.parquet，筛选 test 集，
    生成 full_test_predictions.parquet 供回测引擎和模拟盘使用。

Pipeline Position:
    训练流水线第五步（离线，一次性）。
    上游: 02_train_gbdt_models.py (candidate_with_scores.parquet)
    下游: 回测引擎 (_load_data), 模拟盘 (09_paper_trading_runner)

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet

Outputs:
    - stop_experiment/output/full_test_predictions.parquet

How to Run:
    python -m stop_experiment.backtest.generate_full_predictions

Side Effects:
    - 只读 candidate_with_scores.parquet，输出 full_test_predictions.parquet
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, MODELS_DIR, OBS_VAL_END,
)


def generate_full_predictions():
    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    output_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")

    if not os.path.exists(scores_path):
        raise FileNotFoundError(
            f"{scores_path} 不存在，请先运行 02_train_gbdt_models.py"
        )

    print(f"读取: {scores_path}")
    df = pd.read_parquet(scores_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    print(f"  总行数: {len(df)}")

    val_end = pd.to_datetime(OBS_VAL_END)
    test_df = df[df["obs_date"] > val_end].copy()
    print(f"  test 集 (obs_date > {OBS_VAL_END}): {len(test_df)} 行")

    if "composite_score" not in test_df.columns:
        sell_score = test_df.get("pred_sell_reg", pd.Series(0, index=test_df.index))
        buy_score = -test_df.get("pred_buy_reg", pd.Series(0, index=test_df.index))
        test_df["composite_score"] = sell_score * 0.5 + buy_score * 0.5
        print("  composite_score 已计算 (sell_reg*0.5 + (-buy_reg)*0.5)")

    if "score" not in test_df.columns:
        test_df["score"] = test_df.get("pred_sell_reg", 0)
        print("  score 列已补充 (pred_sell_reg)")

    required_cols = [
        "signal_id", "obs_date", "obs_day", "ts_code",
        "pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls",
        "composite_score", "score",
    ]
    available_cols = [c for c in required_cols if c in test_df.columns]
    missing_cols = [c for c in required_cols if c not in test_df.columns]
    if missing_cols:
        print(f"  缺少列: {missing_cols}")

    test_df[available_cols].to_parquet(output_path, index=False)
    print(f"输出: {output_path} ({len(test_df)} 行, {len(available_cols)} 列)")

    date_range = f"{test_df['obs_date'].min().strftime('%Y-%m-%d')} ~ {test_df['obs_date'].max().strftime('%Y-%m-%d')}"
    print(f"  日期范围: {date_range}")
    print(f"  obs_day 分布: {test_df['obs_day'].value_counts().sort_index().to_dict()}")

    return test_df


if __name__ == "__main__":
    generate_full_predictions()
