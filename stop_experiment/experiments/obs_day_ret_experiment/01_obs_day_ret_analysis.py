#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obs_day_ret（观察日当天涨幅）因子离线分析

Purpose:
    验证观察日当天涨幅 obs_day_ret = close_t / close_{t-1} - 1 的预测力和独立性。
    不修改 pipeline，仅基于已有 candidate_with_scores.parquet 做分析。

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet

Outputs:
    - results/analysis/obs_day_ret_distribution.csv
    - results/analysis/obs_day_ret_ic.csv
    - results/analysis/obs_day_ret_quantile_returns.csv
    - results/analysis/obs_day_ret_correlation.csv
    - results/analysis/obs_day_ret_conditional.csv
    - results/analysis/obs_day_ret_monthly_stability.csv

How to Run:
    python -m stop_experiment.experiments.obs_day_ret_experiment.01_obs_day_ret_analysis

Side Effects:
    - 只读 candidate_with_scores.parquet，输出仅写入 results/analysis/
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from scipy import stats

from stop_experiment.pipeline.stop_config import OBS_VAL_END, MODELS_DIR

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

N_QUANTILES = 10


def compute_obs_day_ret(df: pd.DataFrame) -> pd.Series:
    """
    计算观察日当天涨幅 obs_day_ret = close_t / close_{t-1} - 1

    - obs_day > 1: 按 signal_id 分组，用 obs_close 的 pct_change
    - obs_day = 1: 前一日收盘 = 信号日收盘 = obs_close / (1 + ret_to_trigger)
    """
    df = df.sort_values(["signal_id", "obs_day"]).copy()
    obs_day_ret = df.groupby("signal_id")["obs_close"].pct_change()

    mask_d1 = df["obs_day"] == 1
    ret_to_trigger_safe = df.loc[mask_d1, "ret_to_trigger"].replace(0, np.nan)
    signal_close = df.loc[mask_d1, "obs_close"] / (1 + ret_to_trigger_safe)
    obs_day_ret_d1 = df.loc[mask_d1, "obs_close"] / signal_close - 1
    obs_day_ret.iloc[mask_d1.values.nonzero()[0]] = obs_day_ret_d1.values

    obs_day_ret = obs_day_ret.replace([np.inf, -np.inf], np.nan)
    return obs_day_ret


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
    print("obs_day_ret（观察日当天涨幅）因子离线分析")
    print("=" * 60)

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在")

    print("\n[1/6] 加载数据 + 计算 obs_day_ret...")
    df = pd.read_parquet(scores_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.dropna(subset=["mfe_20", "mae_20"])

    required_cols = ["signal_id", "obs_day", "obs_close", "ret_to_trigger"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    df["obs_day_ret"] = compute_obs_day_ret(df)
    valid_mask = df["obs_day_ret"].notna()
    print(f"  总行数: {len(df)}, obs_day_ret 有效: {valid_mask.sum()} ({valid_mask.mean():.1%})")

    val_end = pd.Timestamp(OBS_VAL_END)
    test = df[df["obs_date"] > val_end].copy()
    test = test[test["obs_day_ret"].notna()]
    print(f"  test 集: {len(test)} 行")

    print("\n[2/6] 分布概览...")
    desc = test["obs_day_ret"].describe(
        percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )
    dist_rows = []
    for stat_name in ["count", "mean", "std", "min", "1%", "5%", "10%",
                       "25%", "50%", "75%", "90%", "95%", "99%", "max"]:
        dist_rows.append({"stat": stat_name, "value": desc[stat_name]})
    dist_df = pd.DataFrame(dist_rows)
    dist_path = os.path.join(ANALYSIS_DIR, "obs_day_ret_distribution.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"  保存: {dist_path}")
    print(f"  均值={desc['mean']:.4f}, 中位数={desc['50%']:.4f}, 标准差={desc['std']:.4f}")
    print(f"  1%={desc['1%']:.4f}, 99%={desc['99%']:.4f}")
    print(f"  涨停附近(>9%): {(test['obs_day_ret'] > 0.09).mean():.2%}")
    print(f"  跌停附近(<-9%): {(test['obs_day_ret'] < -0.09).mean():.2%}")

    print("\n[3/6] 单因子 IC 分析...")
    ic_rows = []
    for label_col in ["mfe_20", "mae_20"]:
        valid = test[test[label_col].notna() & test["obs_day_ret"].notna()]
        if len(valid) < 100:
            continue
        ic_spearman, p_spearman = stats.spearmanr(valid["obs_day_ret"], valid[label_col])
        ic_pearson, p_pearson = stats.pearsonr(valid["obs_day_ret"], valid[label_col])
        ic_rows.append({
            "label": label_col,
            "ic_spearman": ic_spearman, "p_spearman": p_spearman,
            "ic_pearson": ic_pearson, "p_pearson": p_pearson,
            "n": len(valid),
        })

    for obs_day_val in sorted(test["obs_day"].unique()):
        sub = test[test["obs_day"] == obs_day_val]
        for label_col in ["mfe_20", "mae_20"]:
            valid = sub[sub[label_col].notna() & sub["obs_day_ret"].notna()]
            if len(valid) < 50:
                continue
            ic_sp, p_sp = stats.spearmanr(valid["obs_day_ret"], valid[label_col])
            ic_rows.append({
                "label": f"{label_col}_obs_day={obs_day_val}",
                "ic_spearman": ic_sp, "p_spearman": p_sp,
                "ic_pearson": np.nan, "p_pearson": np.nan,
                "n": len(valid),
            })

    ic_df = pd.DataFrame(ic_rows)
    ic_path = os.path.join(ANALYSIS_DIR, "obs_day_ret_ic.csv")
    ic_df.to_csv(ic_path, index=False)
    print(f"  保存: {ic_path}")

    print(f"\n  {'标签':30s} {'IC(Spearman)':>14s} {'p值':>12s} {'样本':>8s}")
    for _, row in ic_df.iterrows():
        if "obs_day=" not in row["label"]:
            print(f"  {row['label']:30s} {row['ic_spearman']:>14.4f} {row['p_spearman']:>12.2e} {int(row['n']):>8d}")

    print(f"\n  按 obs_day 分层 IC (mfe_20):")
    for _, row in ic_df.iterrows():
        if "mfe_20_obs_day=" in row["label"]:
            print(f"    {row['label']:30s} IC={row['ic_spearman']:.4f}")

    print("\n[4/6] 分位数收益分析...")
    test_q = test.copy()
    test_q["ret_q"] = pd.qcut(test_q["obs_day_ret"], N_QUANTILES, labels=False, duplicates="drop")
    quantile_rows = []
    for q in range(N_QUANTILES):
        sub = test_q[test_q["ret_q"] == q]
        metrics = compute_return_metrics(sub)
        metrics["quantile"] = q
        metrics["avg_obs_day_ret"] = sub["obs_day_ret"].mean()
        quantile_rows.append(metrics)

    q_df = pd.DataFrame(quantile_rows)
    q_path = os.path.join(ANALYSIS_DIR, "obs_day_ret_quantile_returns.csv")
    q_df.to_csv(q_path, index=False)
    print(f"  保存: {q_path}")

    print(f"\n  {'分位':6s} {'样本':>6s} {'avg_ret':>10s} {'avg_mfe':>10s} {'avg_mae':>10s} {'上涨率':>8s} {'盈利率':>8s}")
    for _, row in q_df.iterrows():
        print(f"  Q{int(row['quantile']):<4d} {int(row['n']):>6d} {row['avg_obs_day_ret']:>10.4f} "
              f"{row['avg_mfe_20']:>10.4f} {row['avg_mae_20']:>10.4f} "
              f"{row['pct_rise']:>8.1%} {row['pct_profitable']:>8.1%}")

    q0 = q_df[q_df["quantile"] == 0].iloc[0] if len(q_df[q_df["quantile"] == 0]) > 0 else None
    q_last = q_df[q_df["quantile"] == N_QUANTILES - 1].iloc[0] if len(q_df[q_df["quantile"] == N_QUANTILES - 1]) > 0 else None
    if q0 is not None and q_last is not None:
        print(f"\n  Q0(涨幅最低): avg_mfe={q0['avg_mfe_20']:.4f}, avg_mae={q0['avg_mae_20']:.4f}")
        print(f"  Q9(涨幅最高): avg_mfe={q_last['avg_mfe_20']:.4f}, avg_mae={q_last['avg_mae_20']:.4f}")
        spread = q_last["avg_mfe_20"] - q0["avg_mfe_20"]
        print(f"  Q9-Q0 MFE spread: {spread:.4f}")

    print("\n[5/6] 因子独立性分析...")
    corr_candidates = ["ret_to_trigger", "change_pct", "intraday_range",
                       "range_position", "vol_ratio", "obs_day"]
    available_corr = [c for c in corr_candidates if c in test.columns]
    corr_rows = []

    for col in available_corr:
        valid = test[test["obs_day_ret"].notna() & test[col].notna()]
        if len(valid) < 100:
            continue
        sp_corr, sp_p = stats.spearmanr(valid["obs_day_ret"], valid[col])
        pe_corr, pe_p = stats.pearsonr(valid["obs_day_ret"], valid[col])
        corr_rows.append({
            "factor": col,
            "spearman_corr": sp_corr, "spearman_p": sp_p,
            "pearson_corr": pe_corr, "pearson_p": pe_p,
            "n": len(valid),
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_path = os.path.join(ANALYSIS_DIR, "obs_day_ret_correlation.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"  保存: {corr_path}")

    print(f"\n  {'因子':25s} {'Spearman':>10s} {'Pearson':>10s} {'样本':>8s}")
    for _, row in corr_df.iterrows():
        print(f"  {row['factor']:25s} {row['spearman_corr']:>10.4f} {row['pearson_corr']:>10.4f} {int(row['n']):>8d}")

    high_corr = corr_df[corr_df["spearman_corr"].abs() > 0.5]
    if len(high_corr) > 0:
        print(f"\n  ⚠ 高相关因子(>0.5): {', '.join(high_corr['factor'].tolist())}")
    else:
        print(f"\n  ✅ 无高相关因子(>0.5)，obs_day_ret 具有独立性")

    print("\n[6/6] 条件分析 + 月度稳定性...")

    test_cond = test.copy()
    test_cond["ret_tercile"] = pd.qcut(test_cond["obs_day_ret"], 3, labels=["low", "mid", "high"], duplicates="drop")

    cond_rows = []
    for dim_name, dim_col, dim_vals in [
        ("sell_cls", "pred_sell_cls", [("low", lambda g: g["pred_sell_cls"] < g["pred_sell_cls"].quantile(0.33)),
                                        ("mid", lambda g: (g["pred_sell_cls"] >= g["pred_sell_cls"].quantile(0.33)) & (g["pred_sell_cls"] < g["pred_sell_cls"].quantile(0.67))),
                                        ("high", lambda g: g["pred_sell_cls"] >= g["pred_sell_cls"].quantile(0.67))]),
        ("buy_cls", "pred_buy_cls", [("low", lambda g: g["pred_buy_cls"] < g["pred_buy_cls"].quantile(0.33)),
                                      ("mid", lambda g: (g["pred_buy_cls"] >= g["pred_buy_cls"].quantile(0.33)) & (g["pred_buy_cls"] < g["pred_buy_cls"].quantile(0.67))),
                                      ("high", lambda g: g["pred_buy_cls"] >= g["pred_buy_cls"].quantile(0.67))]),
        ("obs_day", "obs_day", [("1", lambda g: g["obs_day"] == 1),
                                 ("2-3", lambda g: g["obs_day"].isin([2, 3])),
                                 ("4+", lambda g: g["obs_day"] >= 4)]),
    ]:
        if dim_col not in test_cond.columns:
            continue
        for ret_level in ["low", "mid", "high"]:
            for dim_label, dim_mask_fn in dim_vals:
                sub = test_cond[(test_cond["ret_tercile"] == ret_level) & dim_mask_fn(test_cond)]
                metrics = compute_return_metrics(sub)
                metrics["obs_day_ret_level"] = ret_level
                metrics["interact_dim"] = dim_name
                metrics["interact_level"] = dim_label
                cond_rows.append(metrics)

    cond_df = pd.DataFrame(cond_rows)
    cond_path = os.path.join(ANALYSIS_DIR, "obs_day_ret_conditional.csv")
    cond_df.to_csv(cond_path, index=False)
    print(f"  保存: {cond_path}")

    print(f"\n  obs_day_ret × sell_cls 交互:")
    print(f"  {'ret_level':10s} {'sell_cls':10s} {'样本':>6s} {'avg_mfe':>10s} {'上涨率':>8s}")
    for _, row in cond_df[cond_df["interact_dim"] == "sell_cls"].iterrows():
        print(f"  {row['obs_day_ret_level']:10s} {row['interact_level']:10s} {int(row['n']):>6d} "
              f"{row['avg_mfe_20']:>10.4f} {row['pct_rise']:>8.1%}")

    print(f"\n  obs_day_ret × obs_day 交互:")
    print(f"  {'ret_level':10s} {'obs_day':10s} {'样本':>6s} {'avg_mfe':>10s} {'上涨率':>8s}")
    for _, row in cond_df[cond_df["interact_dim"] == "obs_day"].iterrows():
        print(f"  {row['obs_day_ret_level']:10s} {row['interact_level']:10s} {int(row['n']):>6d} "
              f"{row['avg_mfe_20']:>10.4f} {row['pct_rise']:>8.1%}")

    print("\n  月度稳定性...")
    test_m = test.copy()
    test_m["month"] = test_m["obs_date"].dt.to_period("M")
    monthly_rows = []
    for month, group in test_m.groupby("month"):
        valid = group[group["obs_day_ret"].notna() & group["mfe_20"].notna()]
        if len(valid) < 30:
            continue
        ic_sp, _ = stats.spearmanr(valid["obs_day_ret"], valid["mfe_20"])
        ic_sp_mae, _ = stats.spearmanr(valid["obs_day_ret"], valid["mae_20"])

        low_ret = valid[valid["obs_day_ret"] < valid["obs_day_ret"].quantile(0.3)]
        high_ret = valid[valid["obs_day_ret"] > valid["obs_day_ret"].quantile(0.7)]
        monthly_rows.append({
            "month": str(month), "n": len(valid),
            "ic_mfe": ic_sp, "ic_mae": ic_sp_mae,
            "avg_mfe_low_ret": low_ret["mfe_20"].mean() if len(low_ret) > 0 else np.nan,
            "avg_mfe_high_ret": high_ret["mfe_20"].mean() if len(high_ret) > 0 else np.nan,
        })

    m_df = pd.DataFrame(monthly_rows)
    m_path = os.path.join(ANALYSIS_DIR, "obs_day_ret_monthly_stability.csv")
    m_df.to_csv(m_path, index=False)
    print(f"  保存: {m_path}")

    print(f"\n  {'月份':10s} {'样本':>6s} {'IC(mfe)':>10s} {'IC(mae)':>10s} {'mfe_low':>10s} {'mfe_high':>10s}")
    for _, row in m_df.iterrows():
        print(f"  {row['month']:10s} {int(row['n']):>6d} {row['ic_mfe']:>10.4f} {row['ic_mae']:>10.4f} "
              f"{row['avg_mfe_low_ret']:>10.4f} {row['avg_mfe_high_ret']:>10.4f}")

    ic_sign_consistency = (m_df["ic_mfe"] > 0).mean() if len(m_df) > 0 else 0
    print(f"\n  IC(mfe) 月度正比例: {ic_sign_consistency:.1%}")

    print(f"\n{'='*60}")
    print("obs_day_ret 分析结论")
    print(f"{'='*60}")

    overall_ic = ic_df[ic_df["label"] == "mfe_20"]
    if len(overall_ic) > 0:
        ic_val = overall_ic.iloc[0]["ic_spearman"]
        print(f"\n  整体 IC(mfe_20): {ic_val:.4f}")
        if abs(ic_val) > 0.03:
            print(f"  ✅ IC > 0.03，有预测力")
        else:
            print(f"  ❌ IC < 0.03，预测力弱")

    if len(high_corr) == 0:
        print(f"  ✅ 与现有因子无高相关，具有独立性")
    else:
        print(f"  ⚠ 与 {', '.join(high_corr['factor'].tolist())} 高相关，独立性存疑")

    if ic_sign_consistency > 0.7:
        print(f"  ✅ IC 月度正比例 {ic_sign_consistency:.0%} > 70%，信号稳定")
    else:
        print(f"  ⚠ IC 月度正比例 {ic_sign_consistency:.0%} < 70%，信号不稳定")

    if q0 is not None and q_last is not None:
        spread = q_last["avg_mfe_20"] - q0["avg_mfe_20"]
        if abs(spread) > 0.02:
            print(f"  ✅ 分位数 MFE spread = {spread:.4f} > 2%，区分度好")
        else:
            print(f"  ⚠ 分位数 MFE spread = {spread:.4f} < 2%，区分度弱")


if __name__ == "__main__":
    main()
