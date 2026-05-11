#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer 2: 预测一致性检查 — full_test_predictions vs predictions/T

Purpose:
    对比回测预测来源 (full_test_predictions.parquet) 与模拟盘实时预测来源
    (predictions/YYYY-MM-DD.parquet) 在同一日期的逐行预测差异。

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - stop_experiment/output/predictions/*.parquet

Outputs:
    - 控制台: 逐日对比结果 + 差异最大的样本

How to Run:
    python -m stop_experiment.tests_consistency.compare_prediction_sources

Side Effects:
    无（只读）
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from glob import glob

from stop_experiment.tests_consistency import (
    OUTPUT_DIR, PREDICTIONS_DIR, PRED_MEAN_TOL, PRED_MAX_TOL, fmt_pass,
)
from stop_experiment.pipeline.stop_config import PRODUCTION_PARAMS

PRED_COLS = ["pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls"]
MERGE_KEYS = ["ts_code", "signal_id", "obs_day"]


def compare_prediction_sources() -> dict:
    print("=" * 70)
    print("Layer 2: 预测一致性检查")
    print("=" * 70)

    full_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    if not os.path.exists(full_path):
        print("  ❌ full_test_predictions.parquet 不存在")
        return {"status": "FAIL", "issues": ["full_test_predictions.parquet 不存在"]}

    full_df = pd.read_parquet(full_path)
    full_df["obs_date"] = pd.to_datetime(full_df["obs_date"])

    pred_files = sorted(glob(os.path.join(PREDICTIONS_DIR, "*.parquet")))
    if not pred_files:
        print("  ⚠ predictions/ 目录无文件，跳过 Layer 2")
        return {"status": "SKIP", "issues": ["predictions/ 目录无文件"]}

    all_issues = []
    date_results = []

    for pf in pred_files:
        date_str = os.path.basename(pf).replace(".parquet", "")
        try:
            date = pd.to_datetime(date_str)
        except ValueError:
            continue

        live_df = pd.read_parquet(pf)
        live_df["obs_date"] = pd.to_datetime(live_df["obs_date"])

        candidate_obs_days = PRODUCTION_PARAMS.get("candidate_obs_days", [1])
        if "obs_day" in live_df.columns:
            live_df = live_df[live_df["obs_day"].isin(candidate_obs_days)].copy()
        if "obs_day" in full_df.columns:
            full_day_base = full_df[full_df["obs_date"] == date]
            full_day = full_day_base[full_day_base["obs_day"].isin(candidate_obs_days)].copy()
        else:
            full_day = full_df[full_df["obs_date"] == date].copy()
        if full_day.empty:
            all_issues.append(f"{date_str}: full_test 中无该日数据")
            date_results.append({"date": date_str, "status": "SKIP", "match_rate": 0})
            continue

        merged = full_day.merge(
            live_df[MERGE_KEYS + PRED_COLS + ["score"]],
            on=MERGE_KEYS, how="inner", suffixes=("_bt", "_live"),
        )

        match_rate = len(merged) / max(len(full_day), 1)

        col_diffs = {}
        for col in PRED_COLS:
            bt_col = f"{col}_bt"
            live_col = f"{col}_live"
            if bt_col in merged.columns and live_col in merged.columns:
                abs_diff = (merged[bt_col] - merged[live_col]).abs()
                col_diffs[col] = {
                    "mean": abs_diff.mean(),
                    "max": abs_diff.max(),
                    "median": abs_diff.median(),
                }

        full_day_sorted = full_day.sort_values("pred_sell_reg", ascending=False)
        live_day_sorted = live_df.sort_values("score" if "score" in live_df.columns else "pred_sell_reg", ascending=False)
        bt_top10 = set(full_day_sorted["ts_code"].head(10).values)
        live_top10 = set(live_day_sorted["ts_code"].head(10).values)
        top10_overlap = len(bt_top10 & live_top10)

        mean_ok = all(d["mean"] < PRED_MEAN_TOL for d in col_diffs.values())
        max_ok = all(d["max"] < PRED_MAX_TOL for d in col_diffs.values())
        top10_ok = top10_overlap == 10

        status = "PASS" if (mean_ok and max_ok and top10_ok) else "FAIL"

        date_results.append({
            "date": date_str,
            "status": status,
            "match_rate": match_rate,
            "col_diffs": col_diffs,
            "top10_overlap": top10_overlap,
            "n_full": len(full_day),
            "n_live": len(live_df),
            "n_merged": len(merged),
        })

        if not mean_ok:
            worst_col = max(col_diffs.items(), key=lambda x: x[1]["mean"])
            all_issues.append(f"{date_str}: {worst_col[0]} mean_diff={worst_col[1]['mean']:.6f}")
        if not max_ok:
            worst_col = max(col_diffs.items(), key=lambda x: x[1]["max"])
            all_issues.append(f"{date_str}: {worst_col[0]} max_diff={worst_col[1]['max']:.6f}")
        if not top10_ok:
            all_issues.append(f"{date_str}: top10 overlap={top10_overlap}/10")

    for dr in date_results:
        diffs_str = ""
        if dr.get("col_diffs"):
            parts = [f"{k}={v['mean']:.6f}" for k, v in dr["col_diffs"].items()]
            diffs_str = ", ".join(parts[:2])
        print(f"  {dr['date']:<12} {dr['status']:<6} "
              f"match={dr.get('match_rate',0):.1%}, "
              f"top10={dr.get('top10_overlap','?')}/10, "
              f"{diffs_str}")

    if all_issues:
        print(f"\n  差异详情:")
        for i in all_issues:
            print(f"    ⚠ {i}")

    n_pass = sum(1 for d in date_results if d["status"] == "PASS")
    n_fail = sum(1 for d in date_results if d["status"] == "FAIL")
    n_skip = sum(1 for d in date_results if d["status"] == "SKIP")
    overall = "PASS" if n_fail == 0 else "FAIL"

    print(f"\n结果: {overall} (PASS={n_pass}, FAIL={n_fail}, SKIP={n_skip})")
    if n_fail > 0:
        print("  ⚠ 预测来源不一致 — replay 模式使用 full_test_predictions 可绕过此问题")
        print("  ⚠ live 模式需确保 07 的特征构造与训练时一致")

    return {"status": overall, "n_pass": n_pass, "n_fail": n_fail, "n_skip": n_skip,
            "issues": all_issues}


if __name__ == "__main__":
    result = compare_prediction_sources()
    sys.exit(0 if result["status"] in ("PASS", "SKIP") else 1)
