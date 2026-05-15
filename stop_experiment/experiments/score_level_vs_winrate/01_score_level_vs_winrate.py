#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证"Top10 模型分数同比下降时胜率下降"的规律

Purpose:
    验证当每日 Top10 候选股的 4 个模型分数同比都在变低时，入场胜率是否也在下降。
    不修改 pipeline，仅基于 candidate_with_scores.parquet 做分析。

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet

Outputs:
    - results/analysis/daily_top10_scores.csv
    - results/analysis/score_level_vs_entry_winrate.csv
    - results/analysis/score_delta_vs_winrate_delta.csv
    - results/analysis/rolling_score_vs_winrate.csv
    - results/analysis/monthly_score_vs_winrate.csv

How to Run:
    python -m stop_experiment.experiments.score_level_vs_winrate.01_score_level_vs_winrate

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

PRED_COLS = ["pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls"]
TOP_K = 10
WINRATE_THRESHOLD = 0.0


def compute_daily_top10(df: pd.DataFrame) -> pd.DataFrame:
    """
    每个交易日按 pred_sell_reg 排序取 Top10，计算 4 模型分数均值和入场胜率。

    胜率定义：Top10 中 obs_day=1 的股票，mfe_20 + mae_20 > 0 的比例（净收益为正）。
    同时计算 avg_mfe 作为收益水平指标。
    """
    daily_rows = []
    for date, group in df.groupby("obs_date"):
        topk = group.nlargest(TOP_K, "pred_sell_reg")
        row = {"obs_date": date, "n_top10": len(topk)}
        for col in PRED_COLS:
            row[f"avg_{col}"] = topk[col].mean()
            row[f"median_{col}"] = topk[col].median()

        entry = topk[topk["obs_day"] == 1]
        row["n_entry"] = len(entry)
        if len(entry) > 0:
            net_ret = entry["mfe_20"] + entry["mae_20"]
            row["entry_winrate"] = (net_ret > WINRATE_THRESHOLD).mean()
            row["entry_avg_mfe"] = entry["mfe_20"].mean()
            row["entry_avg_mae"] = entry["mae_20"].mean()
            row["entry_avg_net"] = net_ret.mean()
            row["entry_pct_rise"] = (entry["mfe_20"] > 0).mean()
        else:
            row["entry_winrate"] = np.nan
            row["entry_avg_mfe"] = np.nan
            row["entry_avg_mae"] = np.nan
            row["entry_avg_net"] = np.nan
            row["entry_pct_rise"] = np.nan

        all_entry = group[group["obs_day"] == 1]
        if len(all_entry) > 0:
            net_ret_all = all_entry["mfe_20"] + all_entry["mae_20"]
            row["all_entry_winrate"] = (net_ret_all > WINRATE_THRESHOLD).mean()
            row["all_entry_avg_mfe"] = all_entry["mfe_20"].mean()
            row["n_all_entry"] = len(all_entry)
        else:
            row["all_entry_winrate"] = np.nan
            row["all_entry_avg_mfe"] = np.nan
            row["n_all_entry"] = 0

        daily_rows.append(row)

    return pd.DataFrame(daily_rows).sort_values("obs_date").reset_index(drop=True)


def main():
    print("=" * 60)
    print("验证: Top10 模型分数水平 vs 收益质量")
    print("=" * 60)

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在")

    print("\n[准备] 加载数据...")
    df = pd.read_parquet(scores_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.dropna(subset=["mfe_20", "mae_20"])

    for col in PRED_COLS:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    val_end = pd.Timestamp(OBS_VAL_END)
    test = df[df["obs_date"] > val_end].copy()
    test = test[test["obs_day"].isin([1, 2, 3])]
    print(f"  test 集 (obs_day 1~3): {len(test)} 行, {test['obs_date'].nunique()} 个交易日")

    print("\n[1/5] 每日 Top10 分数水平时间序列...")
    daily = compute_daily_top10(test)
    daily_path = os.path.join(ANALYSIS_DIR, "daily_top10_scores.csv")
    daily.to_csv(daily_path, index=False)
    print(f"  保存: {daily_path}")
    print(f"  交易日数: {len(daily)}")
    print(f"  avg_sell_reg 范围: [{daily['avg_pred_sell_reg'].min():.4f}, {daily['avg_pred_sell_reg'].max():.4f}]")
    print(f"  avg_sell_cls 范围: [{daily['avg_pred_sell_cls'].min():.4f}, {daily['avg_pred_sell_cls'].max():.4f}]")
    print(f"  avg_buy_reg 范围: [{daily['avg_pred_buy_reg'].min():.4f}, {daily['avg_pred_buy_reg'].max():.4f}]")
    print(f"  avg_buy_cls 范围: [{daily['avg_pred_buy_cls'].min():.4f}, {daily['avg_pred_buy_cls'].max():.4f}]")
    print(f"  entry_winrate 范围: [{daily['entry_winrate'].min():.2%}, {daily['entry_winrate'].max():.2%}]")
    print(f"  entry_avg_mfe 范围: [{daily['entry_avg_mfe'].min():.4f}, {daily['entry_avg_mfe'].max():.4f}]")
    print(f"  entry_avg_net 范围: [{daily['entry_avg_net'].min():.4f}, {daily['entry_avg_net'].max():.4f}]")

    print("\n[2/5] 分数水平 vs 收益质量（胜率 + avg_mfe + avg_net）...")
    level_rows = []
    for col in PRED_COLS:
        avg_col = f"avg_{col}"
        valid = daily[daily["entry_avg_mfe"].notna()].copy()
        if len(valid) < 10:
            continue
        valid["score_tercile"] = pd.qcut(valid[avg_col], 3, labels=["low", "mid", "high"], duplicates="drop")
        for tercile in ["low", "mid", "high"]:
            sub = valid[valid["score_tercile"] == tercile]
            if len(sub) == 0:
                continue
            level_rows.append({
                "factor": col,
                "tercile": tercile,
                "n_days": len(sub),
                "avg_winrate": sub["entry_winrate"].mean(),
                "avg_mfe": sub["entry_avg_mfe"].mean(),
                "avg_net": sub["entry_avg_net"].mean(),
                "avg_score_level": sub[avg_col].mean(),
            })

    level_df = pd.DataFrame(level_rows)
    level_path = os.path.join(ANALYSIS_DIR, "score_level_vs_entry_winrate.csv")
    level_df.to_csv(level_path, index=False)
    print(f"  保存: {level_path}")

    print(f"\n  {'因子':20s} {'三分位':8s} {'天数':>6s} {'胜率':>8s} {'avg_mfe':>10s} {'avg_net':>10s} {'分数水平':>10s}")
    for _, row in level_df.iterrows():
        print(f"  {row['factor']:20s} {row['tercile']:8s} {int(row['n_days']):>6d} "
              f"{row['avg_winrate']:>8.1%} {row['avg_mfe']:>10.4f} {row['avg_net']:>10.4f} {row['avg_score_level']:>10.4f}")

    print(f"\n  Spearman 相关 (分数水平 vs 收益指标):")
    valid = daily[daily["entry_avg_mfe"].notna()]
    for col in PRED_COLS:
        avg_col = f"avg_{col}"
        if avg_col not in valid.columns:
            continue
        for target, target_label in [("entry_winrate", "胜率"), ("entry_avg_mfe", "avg_mfe"), ("entry_avg_net", "avg_net")]:
            if valid[target].std() < 1e-10:
                print(f"    {col} vs {target_label}: 常数列，无法计算")
                continue
            corr, p = stats.spearmanr(valid[avg_col], valid[target])
            sig = "✅" if abs(corr) > 0.3 and p < 0.05 else ""
            print(f"    {col} vs {target_label}: ρ={corr:.4f}, p={p:.4f} {sig}")

    print("\n[3/5] 分数变化方向 vs 胜率变化...")
    delta = daily.copy()
    for col in PRED_COLS:
        avg_col = f"avg_{col}"
        delta[f"delta_{col}"] = delta[avg_col].diff()
    delta["delta_winrate"] = delta["entry_winrate"].diff()
    delta["delta_mfe"] = delta["entry_avg_mfe"].diff()
    delta["delta_net"] = delta["entry_avg_net"].diff()
    delta = delta.dropna(subset=["delta_mfe"])

    delta["all_declining"] = (
        (delta["delta_pred_sell_reg"] < 0) &
        (delta["delta_pred_sell_cls"] < 0) &
        (delta["delta_pred_buy_reg"] < 0) &
        (delta["delta_pred_buy_cls"] < 0)
    )
    delta["mfe_declining"] = delta["delta_mfe"] < 0
    delta["winrate_declining"] = delta["delta_winrate"] < 0

    n_all_declining = delta["all_declining"].sum()
    n_total = len(delta)
    pct_all_declining = n_all_declining / n_total if n_total > 0 else 0

    declining = delta[delta["all_declining"]]
    non_declining = delta[~delta["all_declining"]]

    avg_mfe_declining = declining["entry_avg_mfe"].mean() if len(declining) > 0 else np.nan
    avg_mfe_non_declining = non_declining["entry_avg_mfe"].mean() if len(non_declining) > 0 else np.nan
    avg_wr_declining = declining["entry_winrate"].mean() if len(declining) > 0 else np.nan
    avg_wr_non_declining = non_declining["entry_winrate"].mean() if len(non_declining) > 0 else np.nan

    if len(declining) > 0:
        cond_prob_mfe = declining["mfe_declining"].mean()
        cond_prob_wr = declining["winrate_declining"].mean()
    else:
        cond_prob_mfe = np.nan
        cond_prob_wr = np.nan

    delta_rows = []
    for label, sub in [
        ("4分数同降日", declining),
        ("非同降日", non_declining),
    ]:
        if len(sub) == 0:
            continue
        delta_rows.append({
            "group": label,
            "n_days": len(sub),
            "avg_winrate": sub["entry_winrate"].mean(),
            "avg_mfe": sub["entry_avg_mfe"].mean(),
            "avg_net": sub["entry_avg_net"].mean(),
            "mfe_declining_pct": sub["mfe_declining"].mean(),
            "winrate_declining_pct": sub["winrate_declining"].mean(),
        })

    for col in PRED_COLS:
        d_col = f"delta_{col}"
        declining_col = delta[d_col] < 0
        for label, mask in [(f"{col}_下降", declining_col), (f"{col}_上升", ~declining_col)]:
            sub = delta[mask]
            if len(sub) == 0:
                continue
            delta_rows.append({
                "group": label,
                "n_days": len(sub),
                "avg_winrate": sub["entry_winrate"].mean(),
                "avg_mfe": sub["entry_avg_mfe"].mean(),
                "avg_net": sub["entry_avg_net"].mean(),
                "mfe_declining_pct": sub["mfe_declining"].mean(),
                "winrate_declining_pct": sub["winrate_declining"].mean(),
            })

    delta_df = pd.DataFrame(delta_rows)
    delta_path = os.path.join(ANALYSIS_DIR, "score_delta_vs_winrate_delta.csv")
    delta_df.to_csv(delta_path, index=False)
    print(f"  保存: {delta_path}")

    mfe_diff = avg_mfe_declining - avg_mfe_non_declining if pd.notna(avg_mfe_declining) and pd.notna(avg_mfe_non_declining) else np.nan
    wr_diff = avg_wr_declining - avg_wr_non_declining if pd.notna(avg_wr_declining) and pd.notna(avg_wr_non_declining) else np.nan

    print(f"\n  4分数同降日占比: {pct_all_declining:.1%} ({n_all_declining}/{n_total} 天)")
    print(f"  4分数同降日 avg_mfe: {avg_mfe_declining:.4f}, 非同降日: {avg_mfe_non_declining:.4f}, 差: {mfe_diff:+.4f}")
    print(f"  4分数同降日 胜率: {avg_wr_declining:.2%}, 非同降日: {avg_wr_non_declining:.2%}, 差: {wr_diff:+.2%}")
    print(f"  4分数同降 → mfe也降的条件概率: {cond_prob_mfe:.1%}")
    print(f"  4分数同降 → 胜率也降的条件概率: {cond_prob_wr:.1%}")

    if len(declining) >= 5 and len(non_declining) >= 5:
        t_stat, p_val = stats.ttest_ind(
            declining["entry_avg_mfe"].dropna(),
            non_declining["entry_avg_mfe"].dropna(),
        )
        print(f"  t检验(avg_mfe): t={t_stat:.3f}, p={p_val:.4f} {'✅ p<0.05' if p_val < 0.05 else ''}")

    print(f"\n  {'分组':25s} {'天数':>6s} {'avg_mfe':>10s} {'avg_net':>10s} {'胜率':>8s} {'mfe降占比':>10s}")
    for _, row in delta_df.iterrows():
        print(f"  {row['group']:25s} {int(row['n_days']):>6d} {row['avg_mfe']:>10.4f} "
              f"{row['avg_net']:>10.4f} {row['avg_winrate']:>8.1%} {row['mfe_declining_pct']:>10.1%}")

    print("\n[4/5] 滚动窗口分析...")
    rolling_rows = []
    for window in [5, 10]:
        r = daily.copy()
        for col in PRED_COLS:
            avg_col = f"avg_{col}"
            r[f"roll_{window}_{avg_col}"] = r[avg_col].rolling(window, min_periods=1).mean()
            r[f"deviation_{col}"] = r[avg_col] - r[f"roll_{window}_{avg_col}"]

        r["all_below_trend"] = (
            (r[f"deviation_{PRED_COLS[0]}"] < 0) &
            (r[f"deviation_{PRED_COLS[1]}"] < 0) &
            (r[f"deviation_{PRED_COLS[2]}"] < 0) &
            (r[f"deviation_{PRED_COLS[3]}"] < 0)
        )

        valid = r[r["entry_avg_mfe"].notna()]
        for label, mask in [
            ("all_below_trend", valid["all_below_trend"]),
            ("not_all_below", ~valid["all_below_trend"]),
        ]:
            sub = valid[mask]
            if len(sub) == 0:
                continue
            rolling_rows.append({
                "window": window,
                "group": label,
                "n_days": len(sub),
                "avg_winrate": sub["entry_winrate"].mean(),
                "avg_mfe": sub["entry_avg_mfe"].mean(),
                "avg_net": sub["entry_avg_net"].mean(),
            })

        for col in PRED_COLS:
            dev_col = f"deviation_{col}"
            below = valid[valid[dev_col] < 0]
            above = valid[valid[dev_col] >= 0]
            for label, sub in [(f"{col}_below_trend", below), (f"{col}_above_trend", above)]:
                if len(sub) == 0:
                    continue
                rolling_rows.append({
                    "window": window,
                    "group": label,
                    "n_days": len(sub),
                    "avg_winrate": sub["entry_winrate"].mean(),
                    "avg_mfe": sub["entry_avg_mfe"].mean(),
                    "avg_net": sub["entry_avg_net"].mean(),
                })

    rolling_df = pd.DataFrame(rolling_rows)
    rolling_path = os.path.join(ANALYSIS_DIR, "rolling_score_vs_winrate.csv")
    rolling_df.to_csv(rolling_path, index=False)
    print(f"  保存: {rolling_path}")

    print(f"\n  {'窗口':>4s} {'分组':30s} {'天数':>6s} {'avg_mfe':>10s} {'avg_net':>10s} {'胜率':>8s}")
    for _, row in rolling_df.iterrows():
        print(f"  {int(row['window']):>4d} {row['group']:30s} {int(row['n_days']):>6d} "
              f"{row['avg_mfe']:>10.4f} {row['avg_net']:>10.4f} {row['avg_winrate']:>8.1%}")

    print("\n[5/5] 月度聚合验证...")
    daily_m = daily.copy()
    daily_m["month"] = daily_m["obs_date"].dt.to_period("M")
    monthly = daily_m.groupby("month").agg(
        n_days=("obs_date", "size"),
        avg_sell_reg=("avg_pred_sell_reg", "mean"),
        avg_sell_cls=("avg_pred_sell_cls", "mean"),
        avg_buy_reg=("avg_pred_buy_reg", "mean"),
        avg_buy_cls=("avg_pred_buy_cls", "mean"),
        avg_winrate=("entry_winrate", "mean"),
        avg_mfe=("entry_avg_mfe", "mean"),
        avg_net=("entry_avg_net", "mean"),
    ).reset_index()
    monthly["month"] = monthly["month"].astype(str)

    monthly_rows = []
    for col in PRED_COLS:
        avg_col = f"avg_{col}"
        if avg_col not in monthly.columns:
            continue
        for target, target_label in [("avg_winrate", "胜率"), ("avg_mfe", "avg_mfe"), ("avg_net", "avg_net")]:
            if monthly[target].std() < 1e-10:
                continue
            corr_sp, p_sp = stats.spearmanr(monthly[avg_col], monthly[target])
            corr_pe, p_pe = stats.pearsonr(monthly[avg_col], monthly[target])
            monthly_rows.append({
                "factor": col,
                "target": target_label,
                "spearman_corr": corr_sp, "spearman_p": p_sp,
                "pearson_corr": corr_pe, "pearson_p": p_pe,
            })

    monthly_corr_df = pd.DataFrame(monthly_rows)
    monthly_path = os.path.join(ANALYSIS_DIR, "monthly_score_vs_winrate.csv")
    monthly.to_csv(monthly_path, index=False)
    print(f"  保存: {monthly_path}")

    print(f"\n  月度数据:")
    print(f"  {'月份':10s} {'天数':>4s} {'sell_reg':>10s} {'sell_cls':>10s} {'buy_reg':>10s} {'buy_cls':>10s} {'avg_mfe':>10s} {'avg_net':>10s} {'胜率':>8s}")
    for _, row in monthly.iterrows():
        print(f"  {row['month']:10s} {int(row['n_days']):>4d} {row['avg_sell_reg']:>10.4f} "
              f"{row['avg_sell_cls']:>10.4f} {row['avg_buy_reg']:>10.4f} {row['avg_buy_cls']:>10.4f} "
              f"{row['avg_mfe']:>10.4f} {row['avg_net']:>10.4f} {row['avg_winrate']:>8.1%}")

    print(f"\n  月度相关系数 (分数水平 vs 收益指标):")
    print(f"  {'因子':20s} {'目标':10s} {'Spearman':>10s} {'p值':>10s} {'Pearson':>10s} {'p值':>10s}")
    for _, row in monthly_corr_df.iterrows():
        sig = "✅" if abs(row["spearman_corr"]) > 0.3 and row["spearman_p"] < 0.05 else ""
        print(f"  {row['factor']:20s} {row['target']:10s} {row['spearman_corr']:>10.4f} {row['spearman_p']:>10.4f} "
              f"{row['pearson_corr']:>10.4f} {row['pearson_p']:>10.4f} {sig}")

    print(f"\n{'='*60}")
    print("验证结论")
    print(f"{'='*60}")

    valid = daily[daily["entry_avg_mfe"].notna()]
    any_significant = False

    print(f"\n  1. 分数水平 vs 收益指标 (日度 Spearman):")
    for col in PRED_COLS:
        avg_col = f"avg_{col}"
        for target, target_label in [("entry_avg_mfe", "avg_mfe"), ("entry_avg_net", "avg_net")]:
            if valid[target].std() < 1e-10:
                continue
            corr, p = stats.spearmanr(valid[avg_col], valid[target])
            sig = "✅ 显著" if abs(corr) > 0.3 and p < 0.05 else "❌ 不显著"
            print(f"     {col} vs {target_label}: ρ={corr:.4f}, p={p:.4f} {sig}")
            if abs(corr) > 0.3 and p < 0.05:
                any_significant = True

    print(f"\n  2. 4分数同降日 vs 非同降日:")
    if pd.notna(mfe_diff):
        print(f"     avg_mfe 差: {mfe_diff:+.4f}")
        if abs(mfe_diff) > 0.02:
            print(f"     ✅ avg_mfe 差异 > 2%")
            any_significant = True
        else:
            print(f"     ❌ avg_mfe 差异 < 2%")
    if pd.notna(wr_diff):
        print(f"     胜率差: {wr_diff:+.2%}")

    print(f"\n  3. 月度相关性:")
    for _, row in monthly_corr_df.iterrows():
        if abs(row["spearman_corr"]) > 0.3 and row["spearman_p"] < 0.05:
            print(f"     ✅ {row['factor']} vs {row['target']}: ρ={row['spearman_corr']:.4f}")
            any_significant = True

    if any_significant:
        print(f"\n  📊 结论: 规律部分成立 — 模型分数水平与收益质量存在正相关，分数低时收益确实偏低")
    else:
        print(f"\n  📊 结论: 规律不成立 — 模型分数水平与收益质量无显著关联")

    print(f"\n  补充说明:")
    print(f"  - 4分数同降日占比: {pct_all_declining:.1%}")
    print(f"  - 4分数同降→mfe也降的条件概率: {cond_prob_mfe:.1%}")
    print(f"  - 4分数同降→胜率也降的条件概率: {cond_prob_wr:.1%}")
    print(f"  - 因果方向注意: 分数低可能是市场环境差的反映，而非分数本身导致收益低")


if __name__ == "__main__":
    main()
