#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buy_reg 低值反转假设离线分析

Purpose:
    验证"buy_reg 很低的股票基本上涨"的假设，确认"很低"的精确含义，
    以及该信号与 sell_reg/buy_cls/sell_cls 的交互效果。

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet

Outputs:
    - results/analysis/buy_reg_distribution.csv
    - results/analysis/buy_reg_quantile_returns.csv
    - results/analysis/buy_reg_threshold_returns.csv
    - results/analysis/buy_reg_interaction.csv
    - results/analysis/buy_reg_monthly_stability.csv

How to Run:
    python -m stop_experiment.experiments.buy_reg_low_experiment.01_buy_reg_low_analysis

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

N_QUANTILES = 10

THRESHOLD_CONFIGS = [
    {"label": "buy_reg>-0.02", "filter": lambda df: df["pred_buy_reg"] > -0.02, "direction": "close_to_zero"},
    {"label": "buy_reg>-0.03", "filter": lambda df: df["pred_buy_reg"] > -0.03, "direction": "close_to_zero"},
    {"label": "buy_reg>-0.05", "filter": lambda df: df["pred_buy_reg"] > -0.05, "direction": "close_to_zero"},
    {"label": "buy_reg<-0.10", "filter": lambda df: df["pred_buy_reg"] < -0.10, "direction": "very_neg"},
    {"label": "buy_reg<-0.12", "filter": lambda df: df["pred_buy_reg"] < -0.12, "direction": "very_neg"},
    {"label": "buy_reg<-0.15", "filter": lambda df: df["pred_buy_reg"] < -0.15, "direction": "very_neg"},
    {"label": "buy_reg<-0.20", "filter": lambda df: df["pred_buy_reg"] < -0.20, "direction": "very_neg"},
]


def compute_return_metrics(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0, "avg_mfe_20": np.nan, "avg_mae_20": np.nan,
                "pct_rise": np.nan, "pct_profitable": np.nan,
                "median_mfe_20": np.nan, "median_mae_20": np.nan}
    avg_mfe = sub["mfe_20"].mean()
    avg_mae = sub["mae_20"].mean()
    pct_rise = (sub["mfe_20"] > 0).mean()
    net_ret = sub["mfe_20"] + sub["mae_20"]
    pct_profitable = (net_ret > 0).mean()
    return {"n": n, "avg_mfe_20": avg_mfe, "avg_mae_20": avg_mae,
            "pct_rise": pct_rise, "pct_profitable": pct_profitable,
            "median_mfe_20": sub["mfe_20"].median(),
            "median_mae_20": sub["mae_20"].median()}


def main():
    print("=" * 60)
    print("buy_reg 低值反转假设离线分析")
    print("=" * 60)

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在")

    print("\n[1/6] 加载数据...")
    df = pd.read_parquet(scores_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.dropna(subset=["mfe_20", "mae_20", "pred_buy_reg"])

    val_end = pd.Timestamp(OBS_VAL_END)
    test = df[df["obs_date"] > val_end].copy()
    test_entry = test[test["obs_day"] == 1].copy()
    print(f"  test 集: {len(test)} 行, 入场样本: {len(test_entry)}")

    print("\n[2/6] pred_buy_reg 分布概览...")
    desc = test_entry["pred_buy_reg"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    dist_rows = []
    for stat_name in ["count", "mean", "std", "min", "1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "max"]:
        dist_rows.append({"stat": stat_name, "value": desc[stat_name]})
    dist_df = pd.DataFrame(dist_rows)
    dist_path = os.path.join(ANALYSIS_DIR, "buy_reg_distribution.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"  保存: {dist_path}")
    print(f"\n  均值={desc['mean']:.4f}, 中位数={desc['50%']:.4f}")
    print(f"  1%={desc['1%']:.4f}, 5%={desc['5%']:.4f}, 95%={desc['95%']:.4f}, 99%={desc['99%']:.4f}")

    print("\n  校准曲线: pred_buy_reg 分箱 vs 实际 mae_20 均值")
    test_entry["br_bin"] = pd.qcut(test_entry["pred_buy_reg"], N_QUANTILES, duplicates="drop")
    calib = test_entry.groupby("br_bin", observed=True).agg(
        n=("mae_20", "size"),
        avg_pred_buy_reg=("pred_buy_reg", "mean"),
        avg_actual_mae_20=("mae_20", "mean"),
    ).reset_index()
    print(f"  {'分箱':30s} {'样本数':>8s} {'avg_pred':>10s} {'avg_actual':>10s}")
    for _, row in calib.iterrows():
        print(f"  {str(row['br_bin']):30s} {int(row['n']):>8d} {row['avg_pred_buy_reg']:>10.4f} {row['avg_actual_mae_20']:>10.4f}")

    print("\n[3/6] 分位数收益分析（核心验证）...")
    test_entry["br_q"] = pd.qcut(test_entry["pred_buy_reg"], N_QUANTILES, labels=False, duplicates="drop")
    quantile_rows = []
    for q in range(N_QUANTILES):
        sub = test_entry[test_entry["br_q"] == q]
        metrics = compute_return_metrics(sub)
        metrics["quantile"] = q
        metrics["quantile_label"] = f"Q{q}(low)" if q == 0 else (f"Q{q}(high)" if q == N_QUANTILES - 1 else f"Q{q}")
        metrics["avg_pred_buy_reg"] = sub["pred_buy_reg"].mean()
        quantile_rows.append(metrics)

    q_df = pd.DataFrame(quantile_rows)
    q_path = os.path.join(ANALYSIS_DIR, "buy_reg_quantile_returns.csv")
    q_df.to_csv(q_path, index=False)
    print(f"  保存: {q_path}")

    print(f"\n  {'分位':10s} {'样本':>6s} {'avg_pred':>10s} {'avg_mfe':>10s} {'avg_mae':>10s} {'上涨率':>8s} {'盈利率':>8s}")
    for _, row in q_df.iterrows():
        print(f"  {row['quantile_label']:10s} {int(row['n']):>6d} {row['avg_pred_buy_reg']:>10.4f} "
              f"{row['avg_mfe_20']:>10.4f} {row['avg_mae_20']:>10.4f} "
              f"{row['pct_rise']:>8.1%} {row['pct_profitable']:>8.1%}")

    q0 = q_df[q_df["quantile"] == 0].iloc[0] if len(q_df[q_df["quantile"] == 0]) > 0 else None
    q_last = q_df[q_df["quantile"] == N_QUANTILES - 1].iloc[0] if len(q_df[q_df["quantile"] == N_QUANTILES - 1]) > 0 else None
    if q0 is not None and q_last is not None:
        print(f"\n  Q0(buy_reg最低): avg_mfe={q0['avg_mfe_20']:.4f}, 上涨率={q0['pct_rise']:.1%}")
        print(f"  Q9(buy_reg最高): avg_mfe={q_last['avg_mfe_20']:.4f}, 上涨率={q_last['pct_rise']:.1%}")

    print("\n[4/6] 绝对阈值分析...")
    thresh_rows = []
    for cfg in THRESHOLD_CONFIGS:
        mask = cfg["filter"](test_entry)
        sub = test_entry[mask]
        metrics = compute_return_metrics(sub)
        metrics["label"] = cfg["label"]
        metrics["direction"] = cfg["direction"]
        thresh_rows.append(metrics)

    all_mask = pd.Series(True, index=test_entry.index)
    metrics_all = compute_return_metrics(test_entry)
    metrics_all["label"] = "all"
    metrics_all["direction"] = "all"
    thresh_rows.append(metrics_all)

    t_df = pd.DataFrame(thresh_rows)
    t_path = os.path.join(ANALYSIS_DIR, "buy_reg_threshold_returns.csv")
    t_df.to_csv(t_path, index=False)
    print(f"  保存: {t_path}")

    print(f"\n  {'阈值':20s} {'方向':12s} {'样本':>6s} {'avg_mfe':>10s} {'avg_mae':>10s} {'上涨率':>8s} {'盈利率':>8s}")
    for _, row in t_df.iterrows():
        print(f"  {row['label']:20s} {row['direction']:12s} {int(row['n']):>6d} "
              f"{row['avg_mfe_20']:>10.4f} {row['avg_mae_20']:>10.4f} "
              f"{row['pct_rise']:>8.1%} {row['pct_profitable']:>8.1%}")

    print("\n[5/6] 交互分析...")
    test_entry["br_tercile"] = pd.qcut(test_entry["pred_buy_reg"], 3, labels=["low", "mid", "high"], duplicates="drop")
    test_entry["sr_tercile"] = pd.qcut(test_entry["pred_sell_reg"], 3, labels=["low", "mid", "high"], duplicates="drop")
    test_entry["bc_tercile"] = pd.qcut(test_entry["pred_buy_cls"], 3, labels=["low", "mid", "high"], duplicates="drop")
    test_entry["sc_tercile"] = pd.qcut(test_entry["pred_sell_cls"], 3, labels=["low", "mid", "high"], duplicates="drop")

    interaction_rows = []
    for dim_name, dim_col in [("sell_reg", "sr_tercile"), ("buy_cls", "bc_tercile"), ("sell_cls", "sc_tercile")]:
        for br_val in ["low", "mid", "high"]:
            for dim_val in ["low", "mid", "high"]:
                sub = test_entry[(test_entry["br_tercile"] == br_val) & (test_entry[dim_col] == dim_val)]
                metrics = compute_return_metrics(sub)
                metrics["buy_reg_level"] = br_val
                metrics["interact_dim"] = dim_name
                metrics["interact_level"] = dim_val
                interaction_rows.append(metrics)

    i_df = pd.DataFrame(interaction_rows)
    i_path = os.path.join(ANALYSIS_DIR, "buy_reg_interaction.csv")
    i_df.to_csv(i_path, index=False)
    print(f"  保存: {i_path}")

    print(f"\n  buy_reg × sell_reg 交互:")
    print(f"  {'buy_reg':8s} {'sell_reg':8s} {'样本':>6s} {'avg_mfe':>10s} {'上涨率':>8s} {'盈利率':>8s}")
    for _, row in i_df[i_df["interact_dim"] == "sell_reg"].iterrows():
        print(f"  {row['buy_reg_level']:8s} {row['interact_level']:8s} {int(row['n']):>6d} "
              f"{row['avg_mfe_20']:>10.4f} {row['pct_rise']:>8.1%} {row['pct_profitable']:>8.1%}")

    print(f"\n  buy_reg × buy_cls 交互:")
    print(f"  {'buy_reg':8s} {'buy_cls':8s} {'样本':>6s} {'avg_mfe':>10s} {'上涨率':>8s} {'盈利率':>8s}")
    for _, row in i_df[i_df["interact_dim"] == "buy_cls"].iterrows():
        print(f"  {row['buy_reg_level']:8s} {row['interact_level']:8s} {int(row['n']):>6d} "
              f"{row['avg_mfe_20']:>10.4f} {row['pct_rise']:>8.1%} {row['pct_profitable']:>8.1%}")

    print("\n[6/6] 月度稳定性...")
    test_entry_m = test_entry.copy()
    test_entry_m["month"] = test_entry_m["obs_date"].dt.to_period("M")
    test_entry_m["br_low"] = test_entry_m["pred_buy_reg"] < test_entry_m["pred_buy_reg"].quantile(0.3)

    monthly_rows = []
    for month, group in test_entry_m.groupby("month"):
        for label, mask_fn in [
            ("all", lambda g: pd.Series(True, index=g.index)),
            ("br_low30%", lambda g: g["br_low"]),
            ("br_high70%", lambda g: ~g["br_low"]),
        ]:
            m = mask_fn(group)
            sub = group[m]
            metrics = compute_return_metrics(sub)
            metrics["month"] = str(month)
            metrics["group"] = label
            monthly_rows.append(metrics)

    m_df = pd.DataFrame(monthly_rows)
    m_path = os.path.join(ANALYSIS_DIR, "buy_reg_monthly_stability.csv")
    m_df.to_csv(m_path, index=False)
    print(f"  保存: {m_path}")

    print(f"\n  {'月份':10s} {'组':12s} {'样本':>6s} {'avg_mfe':>10s} {'上涨率':>8s}")
    for _, row in m_df.iterrows():
        if row["group"] in ("br_low30%", "br_high70%"):
            print(f"  {row['month']:10s} {row['group']:12s} {int(row['n']):>6d} "
                  f"{row['avg_mfe_20']:>10.4f} {row['pct_rise']:>8.1%}")

    print(f"\n{'='*60}")
    print("buy_reg 低值分析结论")
    print(f"{'='*60}")

    if q0 is not None:
        print(f"\n  Q0(buy_reg最低, avg_pred={q0['avg_pred_buy_reg']:.4f}):")
        print(f"    avg_mfe_20={q0['avg_mfe_20']:.4f}, 上涨率={q0['pct_rise']:.1%}, 盈利率={q0['pct_profitable']:.1%}")
    if q_last is not None:
        print(f"\n  Q9(buy_reg最高, avg_pred={q_last['avg_pred_buy_reg']:.4f}):")
        print(f"    avg_mfe_20={q_last['avg_mfe_20']:.4f}, 上涨率={q_last['pct_rise']:.1%}, 盈利率={q_last['pct_profitable']:.1%}")

    close_zero_rows = t_df[t_df["direction"] == "close_to_zero"]
    very_neg_rows = t_df[t_df["direction"] == "very_neg"]
    if len(close_zero_rows) > 0:
        best_cz = close_zero_rows.loc[close_zero_rows["avg_mfe_20"].idxmax()]
        print(f"\n  接近0方向最优: {best_cz['label']}, avg_mfe={best_cz['avg_mfe_20']:.4f}, 上涨率={best_cz['pct_rise']:.1%}")
    if len(very_neg_rows) > 0 and very_neg_rows["avg_mfe_20"].notna().any():
        best_vn = very_neg_rows.loc[very_neg_rows["avg_mfe_20"].idxmax()]
        print(f"  非常负方向最优: {best_vn['label']}, avg_mfe={best_vn['avg_mfe_20']:.4f}, 上涨率={best_vn['pct_rise']:.1%}")


if __name__ == "__main__":
    main()
