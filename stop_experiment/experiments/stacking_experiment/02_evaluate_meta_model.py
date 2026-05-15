#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacking 元模型离线评估：IC/MAE/AUC + 月度切片 + 显著性检验

Purpose:
    评估元模型 vs baseline 的预测质量差异，包含月度切片和统计显著性检验。

Inputs:
    - results/meta_test_predictions.parquet (01_train_meta_model.py 产出)

Outputs:
    - results/metrics/evaluation_report.csv (评估指标)
    - results/metrics/monthly_ic_comparison.csv (月度 IC 对比)
    - results/metrics/significance_tests.csv (显著性检验结果)

How to Run:
    python -m stop_experiment.experiments.stacking_experiment.02_evaluate_meta_model

Side Effects:
    - 只读 meta_test_predictions.parquet，输出仅写入 results/metrics/
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

META_PRED_COLS = [
    "meta_stack_lgb_sell_reg",
    "meta_stack_lgb_sell_cls",
    "meta_stack_lgb_composite",
]

BASELINE_COL = "pred_sell_reg"


def compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 10:
        return np.nan
    return stats.spearmanr(y_true[valid], y_pred[valid])[0]


def compute_monthly_ic(df: pd.DataFrame, pred_col: str, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["obs_date"].dt.to_period("M")
    rows = []
    for month, group in df.groupby("month"):
        ic = compute_ic(group[target_col].values, group[pred_col].values)
        rows.append({"month": str(month), "ic": ic, "n": len(group)})
    return pd.DataFrame(rows)


def run_significance_test(monthly_ic_baseline: np.ndarray, monthly_ic_meta: np.ndarray) -> dict:
    valid = ~(np.isnan(monthly_ic_baseline) | np.isnan(monthly_ic_meta))
    if valid.sum() < 3:
        return {"t_stat": np.nan, "p_value": np.nan, "significant_005": False, "n_months": int(valid.sum())}
    diff = monthly_ic_meta[valid] - monthly_ic_baseline[valid]
    t_stat, p_value = stats.ttest_1samp(diff, 0)
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "mean_diff": float(diff.mean()),
        "n_months": int(valid.sum()),
    }


def main():
    print("=" * 60)
    print("Stacking 元模型离线评估")
    print("=" * 60)

    pred_path = os.path.join(RESULTS_DIR, "meta_test_predictions.parquet")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"{pred_path} 不存在，请先运行 01_train_meta_model.py")

    df = pd.read_parquet(pred_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    print(f"  数据: {len(df)} 行, 日期范围: {df['obs_date'].min().date()} ~ {df['obs_date'].max().date()}")

    os.makedirs(METRICS_DIR, exist_ok=True)

    print("\n[1/4] 整体指标对比...")
    eval_rows = []

    baseline_mae = mean_absolute_error(
        df["mfe_20"].dropna().values,
        df.loc[df["mfe_20"].notna(), BASELINE_COL].dropna().values,
    )
    baseline_ic = compute_ic(df["mfe_20"].values, df[BASELINE_COL].values)
    eval_rows.append({
        "model": "baseline_sell_reg",
        "target": "mfe_20",
        "ic": baseline_ic,
        "mae": baseline_mae,
    })
    print(f"  baseline_sell_reg: IC={baseline_ic:.4f}, MAE={baseline_mae:.4f}")

    for meta_col in META_PRED_COLS:
        if meta_col not in df.columns:
            print(f"  {meta_col} 不存在，跳过")
            continue

        if "sell_cls" in meta_col:
            target = "sell_signal"
            valid = df[target].notna() & df[meta_col].notna()
            if valid.sum() > 10 and df.loc[valid, target].nunique() > 1:
                auc = roc_auc_score(df.loc[valid, target], df.loc[valid, meta_col])
                ap = average_precision_score(df.loc[valid, target], df.loc[valid, meta_col])
            else:
                auc, ap = np.nan, np.nan
            ic = compute_ic(df["mfe_20"].values, df[meta_col].values)
            eval_rows.append({
                "model": meta_col,
                "target": target,
                "ic": ic,
                "auc": auc,
                "ap": ap,
            })
            print(f"  {meta_col}: IC(mfe_20)={ic:.4f}, AUC={auc:.4f}, AP={ap:.4f}")
        else:
            target = "mfe_20"
            valid = df[target].notna() & df[meta_col].notna()
            mae = mean_absolute_error(df.loc[valid, target], df.loc[valid, meta_col]) if valid.sum() > 0 else np.nan
            ic = compute_ic(df[target].values, df[meta_col].values)
            eval_rows.append({
                "model": meta_col,
                "target": target,
                "ic": ic,
                "mae": mae,
            })
            print(f"  {meta_col}: IC={ic:.4f}, MAE={mae:.4f}")

    eval_df = pd.DataFrame(eval_rows)
    eval_path = os.path.join(METRICS_DIR, "evaluation_report.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"  保存: {eval_path}")

    print("\n[2/4] 月度 IC 对比...")
    monthly_rows = []
    baseline_monthly = compute_monthly_ic(df, BASELINE_COL, "mfe_20")
    for _, row in baseline_monthly.iterrows():
        monthly_rows.append({"month": row["month"], "model": "baseline_sell_reg", "ic": row["ic"], "n": row["n"]})

    for meta_col in META_PRED_COLS:
        if meta_col not in df.columns:
            continue
        monthly = compute_monthly_ic(df, meta_col, "mfe_20")
        for _, row in monthly.iterrows():
            monthly_rows.append({"month": row["month"], "model": meta_col, "ic": row["ic"], "n": row["n"]})

    monthly_df = pd.DataFrame(monthly_rows)
    monthly_path = os.path.join(METRICS_DIR, "monthly_ic_comparison.csv")
    monthly_df.to_csv(monthly_path, index=False)
    print(f"  保存: {monthly_path}")

    print("\n  月度 IC 对比表:")
    months = sorted(monthly_df["month"].unique())
    models = ["baseline_sell_reg"] + [c for c in META_PRED_COLS if c in df.columns]
    header = f"  {'月份':12s}" + "".join(f"  {m[:20]:>20s}" for m in models)
    print(header)
    for month in months:
        row_str = f"  {month:12s}"
        for model in models:
            ic_val = monthly_df[(monthly_df["month"] == month) & (monthly_df["model"] == model)]["ic"]
            if len(ic_val) > 0 and pd.notna(ic_val.iloc[0]):
                row_str += f"  {ic_val.iloc[0]:>20.4f}"
            else:
                row_str += f"  {'N/A':>20s}"
        print(row_str)

    print("\n[3/4] 显著性检验 (paired t-test: meta IC - baseline IC per month)...")
    sig_rows = []
    baseline_ic_arr = baseline_monthly["ic"].values

    for meta_col in META_PRED_COLS:
        if meta_col not in df.columns:
            continue
        meta_monthly = compute_monthly_ic(df, meta_col, "mfe_20")
        meta_ic_arr = meta_monthly["ic"].values

        n_months = min(len(baseline_ic_arr), len(meta_ic_arr))
        result = run_significance_test(baseline_ic_arr[:n_months], meta_ic_arr[:n_months])
        result["model"] = meta_col
        result["baseline_mean_ic"] = float(baseline_ic_arr[:n_months][~np.isnan(baseline_ic_arr[:n_months])].mean())
        result["meta_mean_ic"] = float(meta_ic_arr[:n_months][~np.isnan(meta_ic_arr[:n_months])].mean())
        sig_rows.append(result)

        sig_str = "✅ 显著" if result["significant_005"] else "❌ 不显著"
        print(f"  {meta_col}:")
        print(f"    baseline月均IC={result['baseline_mean_ic']:.4f}, meta月均IC={result['meta_mean_ic']:.4f}, "
              f"Δ={result['mean_diff']:.4f}")
        print(f"    t={result['t_stat']:.3f}, p={result['p_value']:.4f}, {sig_str} (n_months={result['n_months']})")

    sig_df = pd.DataFrame(sig_rows)
    sig_path = os.path.join(METRICS_DIR, "significance_tests.csv")
    sig_df.to_csv(sig_path, index=False)
    print(f"  保存: {sig_path}")

    print("\n[4/4] 结论...")
    for row in sig_rows:
        model = row["model"]
        delta = row["mean_diff"]
        p = row["p_value"]
        if pd.isna(delta) or pd.isna(p):
            verdict = "⚠️ 无法判断"
        elif delta > 0.05 and p < 0.05:
            verdict = "✅ 有效 (IC提升>5%且显著)"
        elif delta > 0.01 and p < 0.05:
            verdict = "⚠️ 微弱 (IC提升1-5%且显著)"
        elif delta > 0.01:
            verdict = "❌ 不显著"
        else:
            verdict = "❌ 无效 (IC未提升)"
        print(f"  {model}: ΔIC={delta:.4f}, p={p:.4f} → {verdict}")


if __name__ == "__main__":
    main()
