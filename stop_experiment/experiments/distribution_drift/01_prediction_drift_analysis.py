#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测值分布漂移分析

Purpose:
    检查模型预测值在不同时间段的分布漂移，判断是"绝对值漂移但排序仍有效"(A)
    还是"top10也无优势"(B)，为生产策略是否可用 rank 替代绝对值提供依据。

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet
      (含全量数据 + 4 模型预测列 + 标签列)

Outputs:
    - experiments/distribution_drift/results/drift_stats.csv
    - experiments/distribution_drift/results/cross_section_ic.csv
    - experiments/distribution_drift/results/verdict.json

How to Run:
    python -m stop_experiment.experiments.distribution_drift.01_prediction_drift_analysis

Examples:
    python -m stop_experiment.experiments.distribution_drift.01_prediction_drift_analysis
    python -m stop_experiment.experiments.distribution_drift.01_prediction_drift_analysis --top-k 10

Side Effects:
    - 只读 candidate_with_scores.parquet，输出仅写入 results/
"""

from __future__ import annotations

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from scipy import stats

from stop_experiment.pipeline.stop_config import MODELS_DIR, OBS_TRAIN_END, OBS_VAL_END
from stop_experiment.pipeline.factor_columns import REGRESSION_LABELS, CLASSIFICATION_LABELS

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")

PRED_COLS = ["pred_sell_reg", "pred_sell_cls", "pred_buy_cls"]
SCORE_COL = "pred_sell_reg"
LABEL_COL = "mfe_20"
TOP_K = 10

PERIOD_DEFS = {
    "train":      ("2020-01-01", "2025-06-05"),
    "val":        ("2025-07-01", "2025-12-31"),
    "test_early": ("2026-01-01", "2026-03-31"),
    "test_late":  ("2026-04-01", "2029-12-31"),
}


def load_data(scores_path: str) -> pd.DataFrame:
    need_cols = ["obs_date", "obs_day", "ts_code"] + PRED_COLS + [LABEL_COL, "mae_20"]
    df = pd.read_parquet(scores_path, columns=[c for c in need_cols])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.dropna(subset=[LABEL_COL])
    return df


def assign_period(obs_date: pd.Series) -> pd.Series:
    train_end = pd.Timestamp("2025-06-05")
    val_start = pd.Timestamp("2025-07-01")
    val_end = pd.Timestamp("2025-12-31")
    test_early_start = pd.Timestamp("2026-01-01")
    test_early_end = pd.Timestamp("2026-03-31")
    test_late_start = pd.Timestamp("2026-04-01")

    conditions = [
        obs_date <= train_end,
        (obs_date >= val_start) & (obs_date <= val_end),
        (obs_date >= test_early_start) & (obs_date <= test_early_end),
        obs_date >= test_late_start,
    ]
    choices = ["train", "val", "test_early", "test_late"]
    return pd.Series(np.select(conditions, choices, default="unknown"), index=obs_date.index, name="period")


def compute_drift_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period_name, group in df.groupby("period"):
        if period_name == "unknown":
            continue
        for col in PRED_COLS:
            s = group[col]
            rows.append({
                "period": period_name,
                "pred_col": col,
                "n": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "p10": s.quantile(0.10),
                "p90": s.quantile(0.90),
                "std": s.std(),
            })
    return pd.DataFrame(rows)


def compute_cross_section_ic(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    for period_name, period_df in df.groupby("period"):
        if period_name == "unknown":
            continue

        daily_spearman = []
        top10_vs_pool_rows = []

        for date, day_group in period_df.groupby("obs_date"):
            if len(day_group) < top_k + 1:
                continue

            s_score = day_group[SCORE_COL]
            s_label = day_group[LABEL_COL]

            if s_score.std() < 1e-10 or s_label.std() < 1e-10:
                continue

            ic_sp, _ = stats.spearmanr(s_score, s_label)
            daily_spearman.append(ic_sp)

            topk = day_group.nlargest(top_k, SCORE_COL)
            pool_median_label = day_group[LABEL_COL].median()
            pool_mean_label = day_group[LABEL_COL].mean()
            topk_mean_label = topk[LABEL_COL].mean()
            topk_mean_score = topk[SCORE_COL].mean()
            pool_median_score = day_group[SCORE_COL].median()

            top10_vs_pool_rows.append({
                "obs_date": date,
                "top10_mean_score": topk_mean_score,
                "pool_median_score": pool_median_score,
                "top10_mean_mfe": topk_mean_label,
                "pool_mean_mfe": pool_mean_label,
                "pool_median_mfe": pool_median_label,
                "mfe_spread": topk_mean_label - pool_mean_label,
            })

        if not daily_spearman:
            continue

        top10_df = pd.DataFrame(top10_vs_pool_rows)

        rows.append({
            "period": period_name,
            "n_days": len(daily_spearman),
            "mean_spearman_ic": np.mean(daily_spearman),
            "std_spearman_ic": np.std(daily_spearman),
            "ic_ir": np.mean(daily_spearman) / np.std(daily_spearman) if np.std(daily_spearman) > 1e-10 else 0.0,
            "pct_ic_positive": np.mean([ic > 0 for ic in daily_spearman]),
            "top10_avg_mfe": top10_df["top10_mean_mfe"].mean(),
            "pool_avg_mfe": top10_df["pool_mean_mfe"].mean(),
            "top10_avg_score": top10_df["top10_mean_score"].mean(),
            "pool_median_score": top10_df["pool_median_score"].mean(),
            "avg_mfe_spread": top10_df["mfe_spread"].mean(),
            "pct_top10_beats_pool": (top10_df["mfe_spread"] > 0).mean(),
        })

    return pd.DataFrame(rows)


def determine_verdict(drift_stats: pd.DataFrame, cross_ic: pd.DataFrame) -> dict:
    verdict = {
        "conclusion": "unknown",
        "drift_detected": False,
        "ranking_effective": False,
        "top10_advantage": False,
        "details": {},
    }

    train_sell_reg = drift_stats[
        (drift_stats["period"] == "train") & (drift_stats["pred_col"] == SCORE_COL)
    ]
    test_late_sell_reg = drift_stats[
        (drift_stats["period"] == "test_late") & (drift_stats["pred_col"] == SCORE_COL)
    ]

    if len(train_sell_reg) > 0 and len(test_late_sell_reg) > 0:
        train_mean = train_sell_reg["mean"].iloc[0]
        test_mean = test_late_sell_reg["mean"].iloc[0]
        mean_shift = test_mean - train_mean
        train_std = train_sell_reg["std"].iloc[0]
        shift_in_std = abs(mean_shift) / train_std if train_std > 1e-10 else 0

        verdict["drift_detected"] = bool(shift_in_std > 0.5)
        verdict["details"]["sell_reg_mean_shift"] = float(mean_shift)
        verdict["details"]["sell_reg_shift_in_std"] = float(shift_in_std)

    for _, row in cross_ic.iterrows():
        period = row["period"]
        verdict["details"][f"{period}_mean_ic"] = float(row["mean_spearman_ic"])
        verdict["details"][f"{period}_ic_ir"] = float(row.get("ic_ir", 0))
        verdict["details"][f"{period}_pct_ic_positive"] = float(row["pct_ic_positive"])
        verdict["details"][f"{period}_top10_avg_mfe"] = float(row["top10_avg_mfe"])
        verdict["details"][f"{period}_pool_avg_mfe"] = float(row["pool_avg_mfe"])
        verdict["details"][f"{period}_avg_mfe_spread"] = float(row["avg_mfe_spread"])
        verdict["details"][f"{period}_pct_top10_beats_pool"] = float(row["pct_top10_beats_pool"])

    test_periods = cross_ic[cross_ic["period"].isin(["test_early", "test_late"])]
    if len(test_periods) > 0:
        mean_ic = test_periods["mean_spearman_ic"].mean()
        pct_positive = test_periods["pct_ic_positive"].mean()
        pct_beats = test_periods["pct_top10_beats_pool"].mean()
        avg_spread = test_periods["avg_mfe_spread"].mean()

        verdict["ranking_effective"] = bool(mean_ic > 0.02 and pct_positive > 0.55)
        verdict["top10_advantage"] = bool(pct_beats > 0.55 and avg_spread > 0.005)

        verdict["details"]["test_mean_ic"] = float(mean_ic)
        verdict["details"]["test_pct_ic_positive"] = float(pct_positive)
        verdict["details"]["test_pct_top10_beats_pool"] = float(pct_beats)
        verdict["details"]["test_avg_mfe_spread"] = float(avg_spread)

    if not verdict["top10_advantage"]:
        verdict["conclusion"] = "B"
    elif verdict["drift_detected"] and verdict["ranking_effective"]:
        verdict["conclusion"] = "A"
    elif verdict["ranking_effective"] and verdict["top10_advantage"]:
        verdict["conclusion"] = "A_no_drift"
    else:
        verdict["conclusion"] = "inconclusive"

    return verdict


def main():
    parser = argparse.ArgumentParser(description="预测值分布漂移分析")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="每日 top-k 候选数")
    args = parser.parse_args()

    print("=" * 60)
    print("预测值分布漂移分析")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在")

    print("\n[1/4] 加载数据...")
    df = load_data(scores_path)
    df["period"] = assign_period(df["obs_date"])
    df = df[df["period"] != "unknown"]
    print(f"  总行数: {len(df)}")
    for p in ["train", "val", "test_early", "test_late"]:
        n = len(df[df["period"] == p])
        n_dates = df[df["period"] == p]["obs_date"].nunique()
        print(f"  {p}: {n} 行, {n_dates} 个交易日")

    print("\n[2/4] 四段分布对比...")
    drift_stats = compute_drift_stats(df)
    drift_path = os.path.join(RESULTS_DIR, "drift_stats.csv")
    drift_stats.to_csv(drift_path, index=False)
    print(f"  保存: {drift_path}")

    print(f"\n  {'期间':12s} {'预测列':18s} {'N':>8s} {'均值':>10s} {'中位数':>10s} {'P10':>10s} {'P90':>10s} {'标准差':>10s}")
    for _, row in drift_stats.iterrows():
        print(f"  {row['period']:12s} {row['pred_col']:18s} {int(row['n']):>8d} "
              f"{row['mean']:>10.4f} {row['median']:>10.4f} {row['p10']:>10.4f} "
              f"{row['p90']:>10.4f} {row['std']:>10.4f}")

    print("\n[3/4] 横截面排序有效性...")
    cross_ic = compute_cross_section_ic(df, top_k=args.top_k)
    ic_path = os.path.join(RESULTS_DIR, "cross_section_ic.csv")
    cross_ic.to_csv(ic_path, index=False)
    print(f"  保存: {ic_path}")

    print(f"\n  {'期间':12s} {'天数':>6s} {'均值IC':>10s} {'IC_IR':>10s} {'IC正比例':>10s} "
          f"{'Top10 MFE':>12s} {'池MFE':>10s} {'MFE差':>10s} {'Top10胜池%':>12s}")
    for _, row in cross_ic.iterrows():
        print(f"  {row['period']:12s} {int(row['n_days']):>6d} {row['mean_spearman_ic']:>10.4f} "
              f"{row.get('ic_ir', 0):>10.4f} {row['pct_ic_positive']:>10.1%} "
              f"{row['top10_avg_mfe']:>12.4f} {row['pool_avg_mfe']:>10.4f} "
              f"{row['avg_mfe_spread']:>10.4f} {row['pct_top10_beats_pool']:>12.1%}")

    print("\n[4/4] 判定结论...")
    verdict = determine_verdict(drift_stats, cross_ic)
    verdict_path = os.path.join(RESULTS_DIR, "verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print(f"  保存: {verdict_path}")

    print(f"\n{'='*60}")
    print("结论")
    print(f"{'='*60}")

    if verdict["drift_detected"]:
        print(f"  ✅ 检测到分布漂移 (sell_reg 均值偏移 {verdict['details'].get('sell_reg_mean_shift', 'N/A'):+.4f}, "
              f"{verdict['details'].get('sell_reg_shift_in_std', 'N/A'):.2f}σ)")
    else:
        print(f"  ❌ 未检测到显著分布漂移")

    if verdict["ranking_effective"]:
        print(f"  ✅ 排序仍有效 (test 均值IC={verdict['details'].get('test_mean_ic', 'N/A'):.4f}, "
              f"IC正比例={verdict['details'].get('test_pct_ic_positive', 'N/A'):.1%})")
    else:
        print(f"  ❌ 排序失效")

    if verdict["top10_advantage"]:
        print(f"  ✅ Top10 有优势 (胜池比例={verdict['details'].get('test_pct_top10_beats_pool', 'N/A'):.1%}, "
              f"MFE差={verdict['details'].get('test_avg_mfe_spread', 'N/A'):.4f})")
    else:
        print(f"  ❌ Top10 无优势")

    conclusion = verdict["conclusion"]
    if conclusion == "A":
        print(f"\n  📊 判定: 情况 A — 绝对值漂移但排序仍有效，可用 rank 替代绝对值")
    elif conclusion == "A_no_drift":
        print(f"\n  📊 判定: 优于 A — 无显著漂移且排序有效，模型稳定可用")
    elif conclusion == "B":
        print(f"\n  📊 判定: 情况 B — Top10 也无优势，模型/候选池失效")
    else:
        print(f"\n  📊 判定: 不确定 — 需进一步分析")


if __name__ == "__main__":
    main()
