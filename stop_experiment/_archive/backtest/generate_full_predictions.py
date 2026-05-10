#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量test集预测生成：基于final模型对test集所有obs_day的样本打分

Purpose:
    从 candidate_with_scores.parquet 筛选 test 集数据，
    生成 full_test_predictions.parquet 供动态退出回测使用。
    所有预测值来自 final 模型（train+val重训）。

Pipeline Position:
    训练流水线最后一步（离线，一次性）。
    上游: 04_signal_selector.py
    下游: dynamic_exit_backtest_v2.py, 06_daily_inference_replay.py, 08_daily_inference_report.py

Inputs:
    - stop_experiment/output/candidate_with_scores.parquet

Outputs:
    - stop_experiment/output/full_test_predictions.parquet

How to Run:
    python stop_experiment/backtest/generate_full_predictions.py

Side Effects:
    - 只读parquet，输出parquet
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, OBS_VAL_END,
)


def main():
    print("=" * 60)
    print("全量test集预测生成")
    print("=" * 60)

    input_path = os.path.join(OUTPUT_DIR, "candidate_with_scores.parquet")
    print(f"\n[1/3] 加载: {input_path}")
    df = pd.read_parquet(input_path)
    df["selection_date"] = pd.to_datetime(df["selection_date"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    print(f"  总行数: {len(df)}, 列数: {len(df.columns)}")
    val_end_ts = pd.Timestamp(OBS_VAL_END)
    test_mask = df["obs_date"] > val_end_ts
    test_df = df[test_mask].copy()
    print(f"  test集 (obs_date > {OBS_VAL_END}): {len(test_df)} 行")

    print(f"\n[2/3] 预测列检查...")
    pred_cols = [c for c in test_df.columns if c.startswith("pred_")]
    print(f"  预测列: {pred_cols}")
    for col in pred_cols:
        print(f"    {col}: mean={test_df[col].mean():.4f}, std={test_df[col].std():.4f}")

    print(f"\n[3/3] 保存...")
    output_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    test_df.to_parquet(output_path, index=False)
    print(f"  保存: {output_path}")

    print(f"\n  日期范围: {test_df['obs_date'].min()} ~ {test_df['obs_date'].max()}")
    print(f"  obs_day分布: {test_df['obs_day'].value_counts().sort_index().to_dict()}")
    print(f"  股票数: {test_df['ts_code'].nunique()}")
    print(f"  信号数: {test_df['signal_id'].nunique()}")


if __name__ == "__main__":
    main()
