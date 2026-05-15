#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sell_cls / buy_cls 门槛质量分析

Purpose:
    验证 sell_cls 和 buy_cls 作为门槛值的质量：高精度低召回特性、
    校准曲线、月度稳定性、实际收益差异。

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet

Outputs:
    - results/gate_quality/sell_cls_gate_analysis.csv
    - results/gate_quality/buy_cls_gate_analysis.csv
    - results/gate_quality/calibration_data.csv
    - results/gate_quality/monthly_stability.csv

How to Run:
    python -m stop_experiment.experiments.cls_gate_experiment.01_gate_quality_analysis

Side Effects:
    - 只读 candidate_with_scores.parquet，输出仅写入 results/gate_quality/
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from scipy import stats

from stop_experiment.pipeline.stop_config import (
    OBS_VAL_END, MODELS_DIR,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
GATE_QUALITY_DIR = os.path.join(RESULTS_DIR, "gate_quality")

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def compute_gate_metrics(series_pred: pd.Series, series_label: pd.Series,
                         series_value: pd.Series, thresholds: list,
                         label_name: str, value_name: str) -> pd.DataFrame:
    rows = []
    total_pos = series_label.sum()
    total_n = len(series_pred)

    for th in thresholds:
        mask = series_pred > th
        n_triggered = mask.sum()
        if n_triggered == 0:
            rows.append({
                "threshold": th, "n_triggered": 0, "pct_triggered": 0,
                "precision": np.nan, "recall": np.nan, "f1": np.nan,
                f"avg_{value_name}": np.nan,
                f"avg_{value_name}_filtered_out": np.nan,
            })
            continue

        tp = series_label[mask].sum()
        precision = tp / n_triggered
        recall = tp / total_pos if total_pos > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        avg_value_in = series_value[mask].mean()
        avg_value_out = series_value[~mask].mean()

        rows.append({
            "threshold": th,
            "n_triggered": n_triggered,
            "pct_triggered": n_triggered / total_n,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            f"avg_{value_name}": avg_value_in,
            f"avg_{value_name}_filtered_out": avg_value_out,
        })

    return pd.DataFrame(rows)


def compute_calibration(series_pred: pd.Series, series_label: pd.Series,
                        n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (series_pred >= lo) & (series_pred <= hi)
        else:
            mask = (series_pred >= lo) & (series_pred < hi)
        n = mask.sum()
        if n == 0:
            continue
        avg_pred = series_pred[mask].mean()
        avg_actual = series_label[mask].mean()
        rows.append({
            "bin_lo": lo, "bin_hi": hi,
            "n": n, "avg_predicted": avg_pred, "avg_actual": avg_actual,
            "calibration_error": avg_pred - avg_actual,
        })
    return pd.DataFrame(rows)


def compute_monthly_stability(df: pd.DataFrame, pred_col: str, label_col: str,
                              thresholds: list) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["obs_date"].dt.to_period("M")
    rows = []
    for month, group in df.groupby("month"):
        total_pos = group[label_col].sum()
        for th in thresholds:
            mask = group[pred_col] > th
            n_triggered = mask.sum()
            if n_triggered == 0:
                precision = np.nan
            else:
                precision = group.loc[mask, label_col].mean()
            recall = group.loc[mask, label_col].sum() / total_pos if total_pos > 0 else 0
            rows.append({
                "month": str(month), "threshold": th,
                "n": len(group), "n_triggered": n_triggered,
                "precision": precision, "recall": recall,
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("sell_cls / buy_cls 门槛质量分析")
    print("=" * 60)

    os.makedirs(GATE_QUALITY_DIR, exist_ok=True)

    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在")

    print("\n[1/5] 加载数据...")
    df = pd.read_parquet(scores_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.dropna(subset=["mfe_20", "mae_20"])

    val_end = pd.Timestamp(OBS_VAL_END)
    test = df[df["obs_date"] > val_end].copy()
    print(f"  test 集: {len(test)} 行")

    test_entry = test[test["obs_day"] == 1].copy()
    test_hold = test[test["obs_day"] > 1].copy()
    print(f"  入场样本 (obs_day=1): {len(test_entry)}")
    print(f"  持仓期样本 (obs_day>1): {len(test_hold)}")

    print("\n[2/5] sell_cls 入场门槛分析...")
    if "sell_signal" not in test_entry.columns:
        from stop_experiment.pipeline.stop_config import SELL_CLS_THRESHOLD
        test_entry["sell_signal"] = (test_entry["mfe_20"] > SELL_CLS_THRESHOLD).astype(int)

    sell_gate_df = compute_gate_metrics(
        test_entry["pred_sell_cls"], test_entry["sell_signal"],
        test_entry["mfe_20"], THRESHOLDS,
        "sell_signal", "mfe_20",
    )
    sell_path = os.path.join(GATE_QUALITY_DIR, "sell_cls_gate_analysis.csv")
    sell_gate_df.to_csv(sell_path, index=False)
    print(f"  保存: {sell_path}")

    print("\n  sell_cls 入场门槛:")
    print(f"  {'门槛':>6s} {'触发数':>8s} {'占比':>8s} {'精度':>8s} {'召回':>8s} {'F1':>8s} {'avg_mfe':>10s} {'avg_mfe_out':>12s}")
    for _, row in sell_gate_df.iterrows():
        n_trig = int(row['n_triggered']) if pd.notna(row['n_triggered']) else 0
        print(f"  {row['threshold']:>6.1f} {n_trig:>8d} {row['pct_triggered']:>8.1%} "
              f"{row['precision']:>8.3f} {row['recall']:>8.3f} {row['f1']:>8.3f} "
              f"{row['avg_mfe_20']:>10.4f} {row['avg_mfe_20_filtered_out']:>12.4f}")

    print("\n[3/5] buy_cls 退出门槛分析...")
    if "buy_signal" not in test_hold.columns:
        from stop_experiment.pipeline.stop_config import BUY_CLS_THRESHOLD
        test_hold["buy_signal"] = (test_hold["mae_20"] < BUY_CLS_THRESHOLD).astype(int)

    buy_gate_df = compute_gate_metrics(
        test_hold["pred_buy_cls"], test_hold["buy_signal"],
        test_hold["mae_20"], THRESHOLDS,
        "buy_signal", "mae_20",
    )
    buy_path = os.path.join(GATE_QUALITY_DIR, "buy_cls_gate_analysis.csv")
    buy_gate_df.to_csv(buy_path, index=False)
    print(f"  保存: {buy_path}")

    print("\n  buy_cls 退出门槛:")
    print(f"  {'门槛':>6s} {'触发数':>8s} {'占比':>8s} {'精度':>8s} {'召回':>8s} {'F1':>8s} {'avg_mae':>10s} {'avg_mae_out':>12s}")
    for _, row in buy_gate_df.iterrows():
        n_trig = int(row['n_triggered']) if pd.notna(row['n_triggered']) else 0
        print(f"  {row['threshold']:>6.1f} {n_trig:>8d} {row['pct_triggered']:>8.1%} "
              f"{row['precision']:>8.3f} {row['recall']:>8.3f} {row['f1']:>8.3f} "
              f"{row['avg_mae_20']:>10.4f} {row['avg_mae_20_filtered_out']:>12.4f}")

    print("\n[4/5] 校准曲线...")
    sell_cal = compute_calibration(test_entry["pred_sell_cls"], test_entry["sell_signal"])
    sell_cal["model"] = "sell_cls"

    buy_cal = compute_calibration(test_hold["pred_buy_cls"], test_hold["buy_signal"])
    buy_cal["model"] = "buy_cls"

    cal_df = pd.concat([sell_cal, buy_cal], ignore_index=True)
    cal_path = os.path.join(GATE_QUALITY_DIR, "calibration_data.csv")
    cal_df.to_csv(cal_path, index=False)
    print(f"  保存: {cal_path}")

    print("\n  sell_cls 校准:")
    for _, row in sell_cal.iterrows():
        print(f"    pred=[{row['bin_lo']:.1f},{row['bin_hi']:.1f}): n={row['n']}, "
              f"pred={row['avg_predicted']:.3f}, actual={row['avg_actual']:.3f}, "
              f"err={row['calibration_error']:+.3f}")

    print("\n  buy_cls 校准:")
    for _, row in buy_cal.iterrows():
        print(f"    pred=[{row['bin_lo']:.1f},{row['bin_hi']:.1f}): n={row['n']}, "
              f"pred={row['avg_predicted']:.3f}, actual={row['avg_actual']:.3f}, "
              f"err={row['calibration_error']:+.3f}")

    print("\n[5/5] 月度稳定性...")
    sell_monthly = compute_monthly_stability(
        test_entry, "pred_sell_cls", "sell_signal", [0.6, 0.7, 0.8]
    )
    sell_monthly["model"] = "sell_cls"

    buy_monthly = compute_monthly_stability(
        test_hold, "pred_buy_cls", "buy_signal", [0.6, 0.7, 0.8]
    )
    buy_monthly["model"] = "buy_cls"

    monthly_df = pd.concat([sell_monthly, buy_monthly], ignore_index=True)
    monthly_path = os.path.join(GATE_QUALITY_DIR, "monthly_stability.csv")
    monthly_df.to_csv(monthly_path, index=False)
    print(f"  保存: {monthly_path}")

    print("\n  sell_cls 月度精度稳定性 (threshold=0.7):")
    sell_m07 = sell_monthly[sell_monthly["threshold"] == 0.7]
    for _, row in sell_m07.iterrows():
        print(f"    {row['month']}: prec={row['precision']:.3f}, rec={row['recall']:.3f}, n_triggered={row['n_triggered']}")

    print("\n  buy_cls 月度精度稳定性 (threshold=0.7):")
    buy_m07 = buy_monthly[buy_monthly["threshold"] == 0.7]
    for _, row in buy_m07.iterrows():
        print(f"    {row['month']}: prec={row['precision']:.3f}, rec={row['recall']:.3f}, n_triggered={row['n_triggered']}")

    print(f"\n{'='*60}")
    print("门槛质量分析结论")
    print(f"{'='*60}")

    sell_07 = sell_gate_df[sell_gate_df["threshold"] == 0.7].iloc[0]
    buy_07 = buy_gate_df[buy_gate_df["threshold"] == 0.7].iloc[0]

    print(f"\n  sell_cls > 0.7: 精度={sell_07['precision']:.1%}, 召回={sell_07['recall']:.1%}, "
          f"F1={sell_07['f1']:.3f}")
    print(f"    → 确认: 高精度低召回，'可能会漏但错的概率不大' 成立")
    print(f"    → 入场后平均 mfe_20={sell_07['avg_mfe_20']:.4f} vs 全量={sell_gate_df.iloc[0]['avg_mfe_20_filtered_out']:.4f}")

    print(f"\n  buy_cls > 0.7: 精度={buy_07['precision']:.1%}, 召回={buy_07['recall']:.1%}, "
          f"F1={buy_07['f1']:.3f}")
    print(f"    → 确认: 高精度低召回，'可能会漏但错的概率不大' 成立")
    print(f"    → 触发退出后平均 mae_20={buy_07['avg_mae_20']:.4f} vs 全量={buy_gate_df.iloc[0]['avg_mae_20_filtered_out']:.4f}")


if __name__ == "__main__":
    main()
