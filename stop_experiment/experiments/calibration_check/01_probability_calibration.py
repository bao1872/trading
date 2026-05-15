#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
概率校准质量检查 + 固定阈值 vs 横截面分位退出对比

Purpose:
    检查 LightGBM 分类模型 (sell_cls / buy_cls) 的概率校准质量，
    以及固定阈值退出 vs 横截面分位退出的对比。

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet
      (含全量数据 + 4 模型预测列 + 标签列)

Outputs:
    - experiments/calibration_check/results/calibration_data.csv
    - experiments/calibration_check/results/threshold_sensitivity.csv
    - experiments/calibration_check/results/percentile_vs_fixed.csv
    - experiments/calibration_check/results/verdict.json

How to Run:
    python -m stop_experiment.experiments.calibration_check.01_probability_calibration

Examples:
    python -m stop_experiment.experiments.calibration_check.01_probability_calibration
    python -m stop_experiment.experiments.calibration_check.01_probability_calibration --n-bins 15

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

from stop_experiment.pipeline.stop_config import MODELS_DIR, OBS_TRAIN_END, OBS_VAL_END

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")

NEED_COLS = [
    "obs_date", "ts_code", "obs_day",
    "pred_sell_cls", "pred_buy_cls",
    "sell_signal", "buy_signal",
]

N_BINS_DEFAULT = 10
THRESHOLD_CANDIDATES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PERCENTILE_CANDIDATES = [70, 80, 90]
CURRENT_EXIT_THRESHOLD = 0.70


def load_data(scores_path: str) -> pd.DataFrame:
    df = pd.read_parquet(scores_path, columns=NEED_COLS)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.dropna(subset=["pred_sell_cls", "pred_buy_cls", "sell_signal", "buy_signal"])
    return df


def split_by_period(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_end = pd.Timestamp(OBS_TRAIN_END)
    val_end = pd.Timestamp(OBS_VAL_END)
    return {
        "train": df[df["obs_date"] <= train_end],
        "val": df[(df["obs_date"] > train_end) & (df["obs_date"] <= val_end)],
        "test": df[df["obs_date"] > val_end],
    }


def compute_calibration_curve(pred: pd.Series, actual: pd.Series, n_bins: int) -> pd.DataFrame:
    bins = pd.qcut(pred, q=n_bins, duplicates="drop")
    grouped = pd.DataFrame({"pred": pred, "actual": actual, "bin": bins})
    curve = grouped.groupby("bin", observed=True).agg(
        pred_mean=("pred", "mean"),
        actual_rate=("actual", "mean"),
        count=("pred", "size"),
    ).reset_index()
    curve["bin_label"] = curve["bin"].astype(str)
    total = curve["count"].sum()
    curve["ece_weight"] = curve["count"] / total
    curve["abs_error"] = (curve["pred_mean"] - curve["actual_rate"]).abs()
    return curve


def compute_ece(curve: pd.DataFrame) -> float:
    return float((curve["abs_error"] * curve["ece_weight"]).sum())


def run_calibration_analysis(splits: dict[str, pd.DataFrame], n_bins: int) -> pd.DataFrame:
    rows = []
    for period_name, df in splits.items():
        if period_name == "train":
            continue
        for model_name, pred_col, label_col in [
            ("sell_cls", "pred_sell_cls", "sell_signal"),
            ("buy_cls", "pred_buy_cls", "buy_signal"),
        ]:
            curve = compute_calibration_curve(df[pred_col], df[label_col], n_bins)
            ece = compute_ece(curve)
            for _, row in curve.iterrows():
                rows.append({
                    "period": period_name,
                    "model": model_name,
                    "bin_label": row["bin_label"],
                    "pred_mean": row["pred_mean"],
                    "actual_rate": row["actual_rate"],
                    "count": int(row["count"]),
                    "ece": ece,
                })
    return pd.DataFrame(rows)


def run_threshold_sensitivity(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(test_df)
    n_positive = test_df["buy_signal"].sum()
    for threshold in THRESHOLD_CANDIDATES:
        triggered = test_df["pred_buy_cls"] >= threshold
        n_triggered = int(triggered.sum())
        trigger_rate = n_triggered / total if total > 0 else 0.0
        if n_triggered > 0:
            precision = test_df.loc[triggered, "buy_signal"].mean()
        else:
            precision = 0.0
        triggered_positive = int((test_df.loc[triggered, "buy_signal"]).sum()) if n_triggered > 0 else 0
        recall = triggered_positive / n_positive if n_positive > 0 else 0.0
        rows.append({
            "threshold": threshold,
            "n_triggered": n_triggered,
            "trigger_rate": trigger_rate,
            "precision": precision,
            "recall": recall,
        })
    return pd.DataFrame(rows)


def run_percentile_vs_fixed(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fixed_mask = test_df["pred_buy_cls"] >= CURRENT_EXIT_THRESHOLD
    n_fixed = int(fixed_mask.sum())
    fixed_trigger_rate = n_fixed / len(test_df) if len(test_df) > 0 else 0.0

    if n_fixed > 0:
        fixed_precision = test_df.loc[fixed_mask, "buy_signal"].mean()
    else:
        fixed_precision = 0.0

    daily_stats = []
    for date, day_group in test_df.groupby("obs_date"):
        if len(day_group) < 5:
            continue
        for pct in PERCENTILE_CANDIDATES:
            cutoff = day_group["pred_buy_cls"].quantile(pct / 100.0)
            pct_mask = day_group["pred_buy_cls"] >= cutoff
            n_pct = int(pct_mask.sum())
            pct_precision = day_group.loc[pct_mask, "buy_signal"].mean() if n_pct > 0 else 0.0
            daily_stats.append({
                "obs_date": date,
                "percentile": pct,
                "cutoff_value": cutoff,
                "n_triggered": n_pct,
                "day_total": len(day_group),
                "trigger_rate": n_pct / len(day_group),
                "precision": pct_precision,
            })

    daily_df = pd.DataFrame(daily_stats)

    for pct in PERCENTILE_CANDIDATES:
        sub = daily_df[daily_df["percentile"] == pct]
        if len(sub) == 0:
            continue
        avg_trigger_rate = sub["trigger_rate"].mean()
        avg_cutoff = sub["cutoff_value"].mean()
        avg_precision = sub["precision"].mean()
        rows.append({
            "method": f"percentile_{pct}",
            "avg_trigger_rate": avg_trigger_rate,
            "avg_cutoff_value": avg_cutoff,
            "avg_precision": avg_precision,
            "n_days": len(sub),
        })

    rows.append({
        "method": f"fixed_{CURRENT_EXIT_THRESHOLD}",
        "avg_trigger_rate": fixed_trigger_rate,
        "avg_cutoff_value": CURRENT_EXIT_THRESHOLD,
        "avg_precision": fixed_precision,
        "n_days": test_df["obs_date"].nunique(),
    })

    return pd.DataFrame(rows), daily_df


def build_verdict(
    calibration_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    percentile_df: pd.DataFrame,
) -> dict:
    verdict = {
        "conclusion": "unknown",
        "calibration_quality": {},
        "threshold_recommendation": {},
        "percentile_vs_fixed": {},
        "details": {},
    }

    for period in ["val", "test"]:
        for model in ["sell_cls", "buy_cls"]:
            sub = calibration_df[(calibration_df["period"] == period) & (calibration_df["model"] == model)]
            if len(sub) > 0:
                ece = sub["ece"].iloc[0]
                key = f"{period}_{model}_ece"
                verdict["calibration_quality"][key] = float(ece)
                verdict["details"][key] = float(ece)

    buy_cls_val_ece = verdict["calibration_quality"].get("val_buy_cls_ece", None)
    buy_cls_test_ece = verdict["calibration_quality"].get("test_buy_cls_ece", None)
    if buy_cls_val_ece is not None and buy_cls_test_ece is not None:
        avg_ece = (buy_cls_val_ece + buy_cls_test_ece) / 2
        if avg_ece > 0.10:
            verdict["conclusion"] = "poorly_calibrated"
            verdict["calibration_quality"]["overall"] = "poor"
        elif avg_ece > 0.05:
            verdict["conclusion"] = "moderately_calibrated"
            verdict["calibration_quality"]["overall"] = "moderate"
        else:
            verdict["conclusion"] = "well_calibrated"
            verdict["calibration_quality"]["overall"] = "good"

    current_row = threshold_df[threshold_df["threshold"] == CURRENT_EXIT_THRESHOLD]
    if len(current_row) > 0:
        verdict["threshold_recommendation"]["current_threshold"] = CURRENT_EXIT_THRESHOLD
        verdict["threshold_recommendation"]["current_precision"] = float(current_row["precision"].iloc[0])
        verdict["threshold_recommendation"]["current_recall"] = float(current_row["recall"].iloc[0])

    best_f1_row = None
    best_f1 = -1.0
    for _, row in threshold_df.iterrows():
        p = row["precision"]
        r = row["recall"]
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_f1_row = row

    if best_f1_row is not None:
        p = best_f1_row["precision"]
        r = best_f1_row["recall"]
        verdict["threshold_recommendation"]["best_f1_threshold"] = float(best_f1_row["threshold"])
        verdict["threshold_recommendation"]["best_f1"] = float(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        verdict["threshold_recommendation"]["best_f1_precision"] = float(p)
        verdict["threshold_recommendation"]["best_f1_recall"] = float(r)

    fixed_row = percentile_df[percentile_df["method"] == f"fixed_{CURRENT_EXIT_THRESHOLD}"]
    p80_row = percentile_df[percentile_df["method"] == "percentile_80"]
    if len(fixed_row) > 0 and len(p80_row) > 0:
        fixed_rate = float(fixed_row["avg_trigger_rate"].iloc[0])
        p80_rate = float(p80_row["avg_trigger_rate"].iloc[0])
        rate_diff = p80_rate - fixed_rate
        verdict["percentile_vs_fixed"]["fixed_trigger_rate"] = fixed_rate
        verdict["percentile_vs_fixed"]["p80_trigger_rate"] = p80_rate
        verdict["percentile_vs_fixed"]["rate_diff"] = float(rate_diff)
        verdict["percentile_vs_fixed"]["p80_avg_cutoff"] = float(p80_row["avg_cutoff_value"].iloc[0])
        verdict["percentile_vs_fixed"]["p80_avg_precision"] = float(p80_row["avg_precision"].iloc[0])
        verdict["percentile_vs_fixed"]["fixed_avg_precision"] = float(fixed_row["avg_precision"].iloc[0])

        if abs(rate_diff) < 0.05:
            verdict["percentile_vs_fixed"]["conclusion"] = "similar_trigger_rate"
        elif rate_diff > 0:
            verdict["percentile_vs_fixed"]["conclusion"] = "p80_triggers_more"
        else:
            verdict["percentile_vs_fixed"]["conclusion"] = "p80_triggers_less"

    return verdict


def main():
    parser = argparse.ArgumentParser(description="概率校准质量检查")
    parser.add_argument("--n-bins", type=int, default=N_BINS_DEFAULT, help="校准曲线分 bin 数")
    args = parser.parse_args()

    print("=" * 60)
    print("概率校准质量检查 + 固定阈值 vs 横截面分位退出对比")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在")

    print("\n[1/5] 加载数据...")
    df = load_data(scores_path)
    splits = split_by_period(df)
    for name, sub in splits.items():
        print(f"  {name}: {len(sub)} 行, {sub['obs_date'].nunique()} 个交易日")

    print("\n[2/5] 校准曲线分析...")
    calibration_df = run_calibration_analysis(splits, n_bins=args.n_bins)
    cal_path = os.path.join(RESULTS_DIR, "calibration_data.csv")
    calibration_df.to_csv(cal_path, index=False)
    print(f"  保存: {cal_path}")

    print(f"\n  {'期间':6s} {'模型':10s} {'ECE':>8s}")
    for period in ["val", "test"]:
        for model in ["sell_cls", "buy_cls"]:
            sub = calibration_df[(calibration_df["period"] == period) & (calibration_df["model"] == model)]
            if len(sub) > 0:
                ece = sub["ece"].iloc[0]
                print(f"  {period:6s} {model:10s} {ece:>8.4f}")

    print(f"\n  校准曲线详情 (pred_mean vs actual_rate):")
    print(f"  {'期间':6s} {'模型':10s} {'Bin':20s} {'预测均值':>10s} {'实际比例':>10s} {'样本数':>8s}")
    for _, row in calibration_df.iterrows():
        print(f"  {row['period']:6s} {row['model']:10s} {row['bin_label']:20s} "
              f"{row['pred_mean']:>10.4f} {row['actual_rate']:>10.4f} {int(row['count']):>8d}")

    print("\n[3/5] 阈值敏感性分析 (buy_cls, test 集)...")
    test_df = splits["test"]
    threshold_df = run_threshold_sensitivity(test_df)
    thresh_path = os.path.join(RESULTS_DIR, "threshold_sensitivity.csv")
    threshold_df.to_csv(thresh_path, index=False)
    print(f"  保存: {thresh_path}")

    print(f"\n  {'阈值':>6s} {'触发数':>8s} {'触发率':>10s} {'精确度':>10s} {'召回率':>10s}")
    for _, row in threshold_df.iterrows():
        print(f"  {row['threshold']:>6.1f} {int(row['n_triggered']):>8d} "
              f"{row['trigger_rate']:>10.2%} {row['precision']:>10.4f} {row['recall']:>10.4f}")

    print("\n[4/5] 横截面分位 vs 固定阈值对比 (buy_cls, test 集)...")
    percentile_df, daily_pct_df = run_percentile_vs_fixed(test_df)
    pct_path = os.path.join(RESULTS_DIR, "percentile_vs_fixed.csv")
    percentile_df.to_csv(pct_path, index=False)
    print(f"  保存: {pct_path}")

    print(f"\n  {'方法':20s} {'平均触发率':>12s} {'平均截断值':>12s} {'平均精确度':>12s} {'天数':>6s}")
    for _, row in percentile_df.iterrows():
        print(f"  {row['method']:20s} {row['avg_trigger_rate']:>12.2%} "
              f"{row['avg_cutoff_value']:>12.4f} {row['avg_precision']:>12.4f} {int(row['n_days']):>6d}")

    print("\n[5/5] 生成结论...")
    verdict = build_verdict(calibration_df, threshold_df, percentile_df)
    verdict_path = os.path.join(RESULTS_DIR, "verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print(f"  保存: {verdict_path}")

    print(f"\n{'=' * 60}")
    print("结论")
    print(f"{'=' * 60}")

    cal_quality = verdict["calibration_quality"].get("overall", "unknown")
    if cal_quality == "good":
        print(f"  ✅ 概率校准良好 (ECE < 0.05)")
    elif cal_quality == "moderate":
        print(f"  ⚠️  概率校准中等 (0.05 < ECE < 0.10)")
    elif cal_quality == "poor":
        print(f"  ❌ 概率校准差 (ECE > 0.10)，scale_pos_weight 导致概率失真")
    else:
        print(f"  ❓ 校准质量未知")

    for key, val in verdict["calibration_quality"].items():
        if key != "overall" and val is not None:
            print(f"     {key}: {val:.4f}")

    thr_rec = verdict["threshold_recommendation"]
    if "best_f1_threshold" in thr_rec:
        print(f"\n  阈值推荐:")
        print(f"     当前阈值 {thr_rec.get('current_threshold', 'N/A')}: "
              f"precision={thr_rec.get('current_precision', 0):.4f}, "
              f"recall={thr_rec.get('current_recall', 0):.4f}")
        print(f"     F1 最优阈值 {thr_rec['best_f1_threshold']:.1f}: "
              f"F1={thr_rec['best_f1']:.4f}, "
              f"precision={thr_rec['best_f1_precision']:.4f}, "
              f"recall={thr_rec['best_f1_recall']:.4f}")

    pct_vs_fixed = verdict.get("percentile_vs_fixed", {})
    if pct_vs_fixed:
        print(f"\n  横截面对比:")
        print(f"     固定阈值 {CURRENT_EXIT_THRESHOLD} 触发率: {pct_vs_fixed.get('fixed_trigger_rate', 'N/A'):.2%}")
        print(f"     P80 分位触发率: {pct_vs_fixed.get('p80_trigger_rate', 'N/A'):.2%}")
        print(f"     P80 平均截断值: {pct_vs_fixed.get('p80_avg_cutoff', 'N/A'):.4f}")
        print(f"     触发率差异: {pct_vs_fixed.get('rate_diff', 'N/A'):+.2%}")
        print(f"     P80 精确度: {pct_vs_fixed.get('p80_avg_precision', 'N/A'):.4f}")
        print(f"     固定阈值精确度: {pct_vs_fixed.get('fixed_avg_precision', 'N/A'):.4f}")
        pct_conclusion = pct_vs_fixed.get("conclusion", "unknown")
        if pct_conclusion == "similar_trigger_rate":
            print(f"     📊 固定阈值与 P80 分位触发率相近，横截面分位可作为替代")
        elif pct_conclusion == "p80_triggers_more":
            print(f"     📊 P80 分位触发率更高，横截面分位更宽松")
        elif pct_conclusion == "p80_triggers_less":
            print(f"     📊 P80 分位触发率更低，横截面分位更严格")
        else:
            print(f"     📊 横截面对比结论: {pct_conclusion}")


if __name__ == "__main__":
    main()
