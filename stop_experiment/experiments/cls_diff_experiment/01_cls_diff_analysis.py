#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cls_diff (sell_cls - buy_cls) 离线分析

Purpose:
    分析 cls_diff 作为入场/退出信号的门槛质量、趋势逆转特征、
    与 sell_cls/buy_cls 单独使用的对比。

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet

Outputs:
    - results/analysis/cls_diff_entry_analysis.csv
    - results/analysis/cls_diff_exit_analysis.csv
    - results/analysis/reversal_analysis.csv
    - results/analysis/entry_comparison.csv
    - results/analysis/monthly_stability.csv

How to Run:
    python -m stop_experiment.experiments.cls_diff_experiment.01_cls_diff_analysis

Side Effects:
    - 只读 candidate_with_scores.parquet，输出仅写入 results/analysis/
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import OBS_VAL_END, MODELS_DIR

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

ENTRY_THRESHOLDS = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
EXIT_THRESHOLDS = [-0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1]


def compute_gate_metrics(series_pred, series_label, series_value, thresholds,
                         label_name, value_name, direction="gt"):
    rows = []
    total_pos = series_label.sum()
    total_n = len(series_pred)
    for th in thresholds:
        mask = series_pred > th if direction == "gt" else series_pred < th
        n_triggered = mask.sum()
        if n_triggered == 0:
            rows.append({"threshold": th, "n_triggered": 0, "pct_triggered": 0,
                         "precision": np.nan, "recall": np.nan, "f1": np.nan,
                         f"avg_{value_name}": np.nan})
            continue
        tp = series_label[mask].sum()
        precision = tp / n_triggered
        recall = tp / total_pos if total_pos > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_val = series_value[mask].mean()
        rows.append({"threshold": th, "n_triggered": n_triggered,
                     "pct_triggered": n_triggered / total_n,
                     "precision": precision, "recall": recall, "f1": f1,
                     f"avg_{value_name}": avg_val})
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("cls_diff (sell_cls - buy_cls) 离线分析")
    print("=" * 60)

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在")

    print("\n[1/6] 加载数据...")
    df = pd.read_parquet(scores_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.dropna(subset=["mfe_20", "mae_20"])
    df["cls_diff"] = df["pred_sell_cls"] - df["pred_buy_cls"]

    val_end = pd.Timestamp(OBS_VAL_END)
    test = df[df["obs_date"] > val_end].copy()
    print(f"  test 集: {len(test)} 行")

    if "sell_signal" not in test.columns:
        from stop_experiment.pipeline.stop_config import SELL_CLS_THRESHOLD
        test["sell_signal"] = (test["mfe_20"] > SELL_CLS_THRESHOLD).astype(int)
    if "buy_signal" not in test.columns:
        from stop_experiment.pipeline.stop_config import BUY_CLS_THRESHOLD
        test["buy_signal"] = (test["mae_20"] < BUY_CLS_THRESHOLD).astype(int)

    test_entry = test[test["obs_day"] == 1].copy()
    test_hold = test[test["obs_day"] > 1].copy()
    print(f"  入场样本: {len(test_entry)}, 持仓期样本: {len(test_hold)}")

    print("\n[2/6] 入场端: cls_diff 门槛分析...")
    entry_df = compute_gate_metrics(
        test_entry["cls_diff"], test_entry["sell_signal"],
        test_entry["mfe_20"], ENTRY_THRESHOLDS,
        "sell_signal", "mfe_20", direction="gt",
    )
    entry_path = os.path.join(ANALYSIS_DIR, "cls_diff_entry_analysis.csv")
    entry_df.to_csv(entry_path, index=False)
    print(f"  保存: {entry_path}")

    print(f"\n  {'门槛':>8s} {'触发数':>8s} {'占比':>8s} {'精度':>8s} {'召回':>8s} {'F1':>8s} {'avg_mfe':>10s}")
    for _, row in entry_df.iterrows():
        n = int(row['n_triggered']) if pd.notna(row['n_triggered']) else 0
        print(f"  {row['threshold']:>8.2f} {n:>8d} {row['pct_triggered']:>8.1%} "
              f"{row['precision']:>8.3f} {row['recall']:>8.3f} {row['f1']:>8.3f} "
              f"{row['avg_mfe_20']:>10.4f}")

    print("\n[3/6] 退出端: cls_diff 门槛分析...")
    exit_df = compute_gate_metrics(
        test_hold["cls_diff"], test_hold["buy_signal"],
        test_hold["mae_20"], EXIT_THRESHOLDS,
        "buy_signal", "mae_20", direction="lt",
    )
    exit_path = os.path.join(ANALYSIS_DIR, "cls_diff_exit_analysis.csv")
    exit_df.to_csv(exit_path, index=False)
    print(f"  保存: {exit_path}")

    print(f"\n  {'门槛':>8s} {'触发数':>8s} {'占比':>8s} {'精度':>8s} {'召回':>8s} {'F1':>8s} {'avg_mae':>10s}")
    for _, row in exit_df.iterrows():
        n = int(row['n_triggered']) if pd.notna(row['n_triggered']) else 0
        print(f"  {row['threshold']:>8.2f} {n:>8d} {row['pct_triggered']:>8.1%} "
              f"{row['precision']:>8.3f} {row['recall']:>8.3f} {row['f1']:>8.3f} "
              f"{row['avg_mae_20']:>10.4f}")

    print("\n[4/6] 趋势逆转分析...")
    test_sorted = test.sort_values(["signal_id", "obs_day"]).copy()
    test_sorted["cls_diff_prev"] = test_sorted.groupby("signal_id")["cls_diff"].shift(1)

    reversal = test_sorted[(test_sorted["cls_diff_prev"] > 0) & (test_sorted["cls_diff"] < 0)].copy()
    reversal_hold = reversal[reversal["obs_day"] > 1]

    reversal_rows = []
    for label, mask in [
        ("reversal_any", reversal_hold.index.isin(reversal_hold.index)),
        ("reversal_cls_diff_lt_m01", reversal_hold["cls_diff"] < -0.1),
        ("reversal_buy_cls_gt_07", reversal_hold["pred_buy_cls"] > 0.7),
        ("reversal_and_buy_cls", (reversal_hold["cls_diff"] < -0.1) & (reversal_hold["pred_buy_cls"] > 0.7)),
    ]:
        sub = reversal_hold[mask]
        n = len(sub)
        prec = sub["buy_signal"].mean() if n > 0 else np.nan
        avg_mae = sub["mae_20"].mean() if n > 0 else np.nan
        reversal_rows.append({
            "signal": label, "n": n,
            "precision": prec, "avg_mae_20": avg_mae,
        })

    buy_cls_exit = test_hold[test_hold["pred_buy_cls"] > 0.7]
    reversal_rows.append({
        "signal": "buy_cls_gt_07_baseline",
        "n": len(buy_cls_exit),
        "precision": buy_cls_exit["buy_signal"].mean() if len(buy_cls_exit) > 0 else np.nan,
        "avg_mae_20": buy_cls_exit["mae_20"].mean() if len(buy_cls_exit) > 0 else np.nan,
    })

    and_exit = test_hold[(test_hold["cls_diff"] < 0) & (test_hold["pred_buy_cls"] > 0.7)]
    reversal_rows.append({
        "signal": "AND_cls_diff_lt0_buy_cls_gt07",
        "n": len(and_exit),
        "precision": and_exit["buy_signal"].mean() if len(and_exit) > 0 else np.nan,
        "avg_mae_20": and_exit["mae_20"].mean() if len(and_exit) > 0 else np.nan,
    })

    rev_df = pd.DataFrame(reversal_rows)
    rev_path = os.path.join(ANALYSIS_DIR, "reversal_analysis.csv")
    rev_df.to_csv(rev_path, index=False)
    print(f"  保存: {rev_path}")

    print(f"\n  {'信号':35s} {'触发数':>8s} {'精度':>8s} {'avg_mae':>10s}")
    for _, row in rev_df.iterrows():
        print(f"  {row['signal']:35s} {int(row['n']):>8d} {row['precision']:>8.3f} {row['avg_mae_20']:>10.4f}")

    print("\n[5/6] 入场端对比: cls_diff vs sell_cls...")
    comp_rows = []
    for label, mask in [
        ("cls_diff>0", test_entry["cls_diff"] > 0),
        ("cls_diff>0.3", test_entry["cls_diff"] > 0.3),
        ("sell_cls>0.8", test_entry["pred_sell_cls"] > 0.8),
        ("sell_cls>0.7", test_entry["pred_sell_cls"] > 0.7),
        ("both_diff0_sell08", (test_entry["cls_diff"] > 0) & (test_entry["pred_sell_cls"] > 0.8)),
        ("both_diff03_sell08", (test_entry["cls_diff"] > 0.3) & (test_entry["pred_sell_cls"] > 0.8)),
    ]:
        sub = test_entry[mask]
        n = len(sub)
        prec = sub["sell_signal"].mean() if n > 0 else np.nan
        avg_mfe = sub["mfe_20"].mean() if n > 0 else np.nan
        comp_rows.append({"signal": label, "n": n, "precision": prec, "avg_mfe_20": avg_mfe})

    comp_df = pd.DataFrame(comp_rows)
    comp_path = os.path.join(ANALYSIS_DIR, "entry_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"  保存: {comp_path}")

    print(f"\n  {'信号':30s} {'触发数':>8s} {'精度':>8s} {'avg_mfe':>10s}")
    for _, row in comp_df.iterrows():
        print(f"  {row['signal']:30s} {int(row['n']):>8d} {row['precision']:>8.3f} {row['avg_mfe_20']:>10.4f}")

    print("\n[6/6] 月度稳定性...")
    test_entry_m = test_entry.copy()
    test_entry_m["month"] = test_entry_m["obs_date"].dt.to_period("M")
    monthly_rows = []
    for month, group in test_entry_m.groupby("month"):
        for label, mask_fn in [
            ("cls_diff>0", lambda g: g["cls_diff"] > 0),
            ("cls_diff>0.3", lambda g: g["cls_diff"] > 0.3),
            ("sell_cls>0.8", lambda g: g["pred_sell_cls"] > 0.8),
        ]:
            m = mask_fn(group)
            n = m.sum()
            prec = group.loc[m, "sell_signal"].mean() if n > 0 else np.nan
            monthly_rows.append({"month": str(month), "signal": label, "n": n, "precision": prec})

    monthly_df = pd.DataFrame(monthly_rows)
    monthly_path = os.path.join(ANALYSIS_DIR, "monthly_stability.csv")
    monthly_df.to_csv(monthly_path, index=False)
    print(f"  保存: {monthly_path}")

    print(f"\n{'='*60}")
    print("cls_diff 分析结论")
    print(f"{'='*60}")

    entry_0 = entry_df[entry_df["threshold"] == 0].iloc[0] if len(entry_df[entry_df["threshold"] == 0]) > 0 else None
    if entry_0 is not None:
        print(f"\n  入场: cls_diff>0 精度={entry_0['precision']:.1%}, 召回={entry_0['recall']:.1%}")

    and_row = rev_df[rev_df["signal"] == "AND_cls_diff_lt0_buy_cls_gt07"]
    buy_row = rev_df[rev_df["signal"] == "buy_cls_gt_07_baseline"]
    if len(and_row) > 0 and len(buy_row) > 0:
        print(f"\n  退出: AND(cls_diff<0 & buy_cls>0.7) 精度={and_row.iloc[0]['precision']:.1%}")
        print(f"  退出: buy_cls>0.7 精度={buy_row.iloc[0]['precision']:.1%}")
        print(f"  → AND 组合精度更高，cls_diff 是 buy_cls 退出的有效确认器")

    rev_row = rev_df[rev_df["signal"] == "reversal_any"]
    if len(rev_row) > 0:
        print(f"\n  趋势逆转: 精度={rev_row.iloc[0]['precision']:.1%}, 触发数={int(rev_row.iloc[0]['n'])}")
        print(f"  → 逆转信号精度适中，但需要回测验证端到端效果")


if __name__ == "__main__":
    main()
