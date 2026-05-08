#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buy_cls 模型专项诊断：回答"为什么 buy_cls 有用但预测又偏高"的4张表

Purpose:
    诊断 buy_cls 模型在 test 集的表现，回答 4 个核心问题：
    - 表1: train/val/test 的 buy_signal 正类率分布
    - 表2: pred_buy_cls 在各切片的分布对比
    - 表3: test 集按 pred_buy_cls 分10桶的校准表
    - 表4: buy_cls 按月 AUC 和 pred>0.7 命中率

Pipeline Position:
    诊断工具（按需运行）。
    上游: 02_train_gbdt_models.py
    下游: —

Inputs:
    - stop_experiment/output/candidate_with_scores.parquet

Outputs:
    - stop_experiment/output/backtest/dynamic/buy_cls_diagnosis/
      ├── table1_label_dist.csv
      ├── table2_pred_dist.csv
      ├── table3_bucket_calibration.csv
      └── table4_monthly_auc.csv

How to Run:
    python stop_experiment/backtest/buy_cls_diagnosis.py

Side Effects:
    - 只读parquet，输出csv
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score

from stop_experiment.pipeline.stop_config import OUTPUT_DIR, BACKTEST_DIR, VAL_END


DYNAMIC_DIR = os.path.join(BACKTEST_DIR, "dynamic")
DIAG_DIR = os.path.join(DYNAMIC_DIR, "buy_cls_diagnosis")


def load_full_dataset():
    path = os.path.join(OUTPUT_DIR, "candidate_with_scores.parquet")
    df = pd.read_parquet(path)
    df["selection_date"] = pd.to_datetime(df["selection_date"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    return df


def add_split_label(df):
    """按时间切分 train/val/test"""
    val_end_ts = pd.Timestamp(VAL_END)
    df = df.copy()

    # train: selection_date <= '2025-12-31'
    # val:   '2026-01-01' <= selection_date <= '2026-02-28'
    # test:  selection_date > '2026-02-28'
    train_mask = df["selection_date"] <= pd.Timestamp("2025-12-31")
    val_mask = (df["selection_date"] > pd.Timestamp("2025-12-31")) & (df["selection_date"] <= val_end_ts)
    test_mask = df["selection_date"] > val_end_ts

    df["split"] = "other"
    df.loc[train_mask, "split"] = "train"
    df.loc[val_mask, "split"] = "val"
    df.loc[test_mask, "split"] = "test"
    return df


# ---- 表1: buy_signal 正类率 ----
def table1_label_distribution(df):
    print("\n[表1] train/val/test 的 buy_signal 正类率")
    print("-" * 60)

    # 对每个 split，统计 obs_day=1 的样本（避免重复计数同一信号）
    obs1 = df[df["obs_day"] == 1]
    result = obs1.groupby("split").agg(
        n_signals=("signal_id", "count"),
        n_positive=("buy_signal", "sum"),
        positive_rate=("buy_signal", "mean"),
    ).reset_index()
    print(result.to_string(index=False))

    path = os.path.join(DIAG_DIR, "table1_label_dist.csv")
    result.to_csv(path, index=False)
    print(f"  保存: {path}")

    # 补充：也看所有 obs_day 的统计
    full_result = df.groupby("split").agg(
        n_samples=("signal_id", "count"),
        positive_rate=("buy_signal", "mean"),
    ).reset_index()
    print("\n  全量样本(所有 obs_day):")
    print(full_result.to_string(index=False))
    return result


# ---- 表2: pred_buy_cls 分布对比 ----
def table2_prediction_distribution(df):
    print("\n[表2] pred_buy_cls 在各切片的分布对比")
    print("-" * 60)

    stats_list = []
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]["pred_buy_cls"]
        if sub.empty:
            continue
        row = {
            "split": split_name,
            "n": len(sub),
            "mean": sub.mean(),
            "std": sub.std(),
            "min": sub.min(),
            "p10": sub.quantile(0.1),
            "p25": sub.quantile(0.25),
            "p50": sub.quantile(0.5),
            "p75": sub.quantile(0.75),
            "p90": sub.quantile(0.9),
            "max": sub.max(),
            "pct_gt_0.5": (sub > 0.5).mean(),
            "pct_gt_0.6": (sub > 0.6).mean(),
            "pct_gt_0.7": (sub > 0.7).mean(),
        }
        stats_list.append(row)

    result = pd.DataFrame(stats_list)
    print(result.to_string(index=False))

    path = os.path.join(DIAG_DIR, "table2_pred_dist.csv")
    result.to_csv(path, index=False)
    print(f"  保存: {path}")
    return result


# ---- 表3: 分桶校准表 ----
def table3_bucket_calibration(df):
    print("\n[表3] test 集按 pred_buy_cls 分桶校准表")
    print("-" * 60)

    test_df = df[df["split"] == "test"]
    bins = np.arange(0.0, 1.01, 0.1)
    test_df["bucket"] = pd.cut(test_df["pred_buy_cls"], bins=bins, right=False,
                                labels=[f"{b:.1f}-{b+0.1:.1f}" for b in bins[:-1]])

    result = test_df.groupby("bucket", observed=False).agg(
        n_samples=("signal_id", "count"),
        actual_positive_rate=("buy_signal", "mean"),
        pred_mean=("pred_buy_cls", "mean"),
    ).reset_index()
    result["calibration_error"] = result["pred_mean"] - result["actual_positive_rate"]
    result["n_samples_pct"] = result["n_samples"] / result["n_samples"].sum()

    pd.set_option("display.max_rows", 20)
    pd.set_option("display.width", 200)
    print(result.to_string(index=False))

    path = os.path.join(DIAG_DIR, "table3_bucket_calibration.csv")
    result.to_csv(path, index=False)
    print(f"  保存: {path}")
    return result


# ---- 表4: 月度 AUC 和命中率 ----
def table4_monthly_auc(df):
    print("\n[表4] buy_cls 月度 AUC 和 pred>0.7 命中率")
    print("-" * 60)

    test_df = df[df["split"] == "test"].copy()
    test_df["month"] = test_df["obs_date"].dt.to_period("M")

    rows = []
    for month, sub in test_df.groupby("month", observed=True):
        if sub["buy_signal"].nunique() < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(sub["buy_signal"], sub["pred_buy_cls"])

        mask_07 = sub["pred_buy_cls"] > 0.7
        n_07 = mask_07.sum()
        hit_rate = np.nan
        if n_07 > 0:
            hit_rate = sub.loc[mask_07, "buy_signal"].mean()

        rows.append({
            "month": str(month),
            "n_samples": len(sub),
            "auc": auc,
            "n_gt_0.7": n_07,
            "pct_gt_0.7": mask_07.mean(),
            "hit_rate_gt_0.7": hit_rate,
            "positive_rate": sub["buy_signal"].mean(),
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    path = os.path.join(DIAG_DIR, "table4_monthly_auc.csv")
    result.to_csv(path, index=False)
    print(f"  保存: {path}")
    return result


def main():
    print("=" * 60)
    print("buy_cls 模型专项诊断")
    print("=" * 60)

    os.makedirs(DIAG_DIR, exist_ok=True)

    df = load_full_dataset()
    df = add_split_label(df)
    print(f"  全量: {len(df)} 行")
    for split_name in ["train", "val", "test"]:
        n = (df["split"] == split_name).sum()
        print(f"    {split_name}: {n:,} 行")

    table1_label_distribution(df)
    table2_prediction_distribution(df)
    table3_bucket_calibration(df)
    table4_monthly_auc(df)

    print(f"\n诊断完成，结果保存在: {DIAG_DIR}")


if __name__ == "__main__":
    main()
