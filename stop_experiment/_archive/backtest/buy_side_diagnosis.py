#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[DEPRECATED] buy侧专项诊断：标签漂移+分层表现+校准分析

⚠️ 已废弃：核心诊断能力（标签漂移+分层+校准）已由 buy_cls_diagnosis.py 覆盖。
   替代: buy_cls_diagnosis.py

Purpose:
    (历史保留) 诊断buy_reg/buy_cls在test集泛化衰减的根因。
         (2) 分层表现：按价位/有无buy_cluster/波动率/月份拆解buy模型IC/AUC
         (3) 校准分析：预测概率分桶vs真实命中率

Inputs:
    - stop_experiment/output/dataset.parquet (全量数据集，含train/val/test标签)
    - stop_experiment/output/test_predictions.parquet (test集预测)
    - stop_experiment/output/fold_metrics.csv (模型指标)

Outputs:
    - stop_experiment/output/backtest/buy_diagnosis/ 目录:
      - label_drift.csv           (标签漂移统计)
      - stratified_performance.csv (分层表现)
      - calibration.csv           (校准分析)
      - figures/                  (可视化)

Pipeline Position:
    诊断工具（已废弃）。
    替代: buy_cls_diagnosis.py

How to Run:
    # ⚠️ 请勿运行，已废弃。请使用:
    python stop_experiment/backtest/buy_cls_diagnosis.py

Side Effects:
    - (历史保留，不建议运行)
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, BACKTEST_DIR, TRAIN_END, VAL_END,
    BUY_CLS_THRESHOLD,
)

DIAGNOSIS_DIR = os.path.join(BACKTEST_DIR, "buy_diagnosis")


# ==================== 1. 标签漂移分析 ====================
def analyze_label_drift(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    分析train/val/test三段的标签分布差异

    检查项：
    - buy_signal 正类率
    - mae_20 分布（均值/中位数/分位数）
    - 按月份的正类率
    - 按价位的正类率
    - 按行业/波动率的正类率
    """
    dataset = dataset.copy()
    dataset["selection_date"] = pd.to_datetime(dataset["selection_date"])

    # 数据分割
    train_end = pd.Timestamp(TRAIN_END)
    val_end = pd.Timestamp(VAL_END)

    dataset["split"] = "test"
    dataset.loc[dataset["selection_date"] <= train_end, "split"] = "train"
    dataset.loc[(dataset["selection_date"] > train_end) & (dataset["selection_date"] <= val_end), "split"] = "val"

    # 只用obs_day=1
    obs1 = dataset[dataset["obs_day"] == 1].copy()
    obs1["month"] = obs1["selection_date"].dt.to_period("M").astype(str)

    # 价位分层
    obs1["price_tier"] = pd.cut(
        obs1["obs_close"],
        bins=[0, 10, 20, 50, 100, 2000],
        labels=["<10", "10-20", "20-50", "50-100", ">100"],
    )

    # 波动率分层
    if "volatility_20d" in obs1.columns:
        obs1["vol_tier"] = pd.qcut(obs1["volatility_20d"].fillna(0), q=3, labels=["low", "mid", "high"], duplicates="drop")
    else:
        obs1["vol_tier"] = "unknown"

    results = []

    # 1.1 总体标签分布
    for split_name, grp in obs1.groupby("split"):
        results.append({
            "dimension": "总体",
            "segment": split_name,
            "n_samples": len(grp),
            "buy_signal_rate": grp["buy_signal"].mean(),
            "mae_20_mean": grp["mae_20"].mean(),
            "mae_20_median": grp["mae_20"].median(),
            "mae_20_p25": grp["mae_20"].quantile(0.25),
            "mae_20_p75": grp["mae_20"].quantile(0.75),
            "mae_20_std": grp["mae_20"].std(),
        })

    # 1.2 按月份
    for (split_name, month), grp in obs1.groupby(["split", "month"]):
        results.append({
            "dimension": "月份",
            "segment": f"{split_name}_{month}",
            "n_samples": len(grp),
            "buy_signal_rate": grp["buy_signal"].mean(),
            "mae_20_mean": grp["mae_20"].mean(),
            "mae_20_median": grp["mae_20"].median(),
            "mae_20_p25": grp["mae_20"].quantile(0.25),
            "mae_20_p75": grp["mae_20"].quantile(0.75),
            "mae_20_std": grp["mae_20"].std(),
        })

    # 1.3 按价位
    for (split_name, tier), grp in obs1.groupby(["split", "price_tier"]):
        if pd.isna(tier):
            continue
        results.append({
            "dimension": "价位",
            "segment": f"{split_name}_{tier}",
            "n_samples": len(grp),
            "buy_signal_rate": grp["buy_signal"].mean(),
            "mae_20_mean": grp["mae_20"].mean(),
            "mae_20_median": grp["mae_20"].median(),
            "mae_20_p25": grp["mae_20"].quantile(0.25),
            "mae_20_p75": grp["mae_20"].quantile(0.75),
            "mae_20_std": grp["mae_20"].std(),
        })

    # 1.4 按波动率
    for (split_name, tier), grp in obs1.groupby(["split", "vol_tier"]):
        results.append({
            "dimension": "波动率",
            "segment": f"{split_name}_{tier}",
            "n_samples": len(grp),
            "buy_signal_rate": grp["buy_signal"].mean(),
            "mae_20_mean": grp["mae_20"].mean(),
            "mae_20_median": grp["mae_20"].median(),
            "mae_20_p25": grp["mae_20"].quantile(0.25),
            "mae_20_p75": grp["mae_20"].quantile(0.75),
            "mae_20_std": grp["mae_20"].std(),
        })

    # 1.5 按有无buy_cluster
    if "has_buy_cluster" in obs1.columns:
        for (split_name, has_bc), grp in obs1.groupby(["split", "has_buy_cluster"]):
            results.append({
                "dimension": "buy_cluster",
                "segment": f"{split_name}_has_bc={has_bc}",
                "n_samples": len(grp),
                "buy_signal_rate": grp["buy_signal"].mean(),
                "mae_20_mean": grp["mae_20"].mean(),
                "mae_20_median": grp["mae_20"].median(),
                "mae_20_p25": grp["mae_20"].quantile(0.25),
                "mae_20_p75": grp["mae_20"].quantile(0.75),
                "mae_20_std": grp["mae_20"].std(),
            })

    return pd.DataFrame(results)


# ==================== 2. 分层表现分析 ====================
def analyze_stratified_performance(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    按维度拆解buy模型在test集的表现

    维度：
    - 价位（<10 / 10-20 / 20-50 / 50-100 / >100）
    - 有无buy_cluster
    - 波动率分组
    - 月份
    """
    df = test_df[test_df["obs_day"] == 1].copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"])

    # 价位分层
    df["price_tier"] = pd.cut(
        df["obs_close"],
        bins=[0, 10, 20, 50, 100, 2000],
        labels=["<10", "10-20", "20-50", "50-100", ">100"],
    )

    # 波动率分层
    if "volatility_20d" in df.columns:
        df["vol_tier"] = pd.qcut(df["volatility_20d"].fillna(0), q=3, labels=["low", "mid", "high"], duplicates="drop")
    else:
        df["vol_tier"] = "unknown"

    # 月份
    df["month"] = df["obs_date"].dt.to_period("M").astype(str)

    results = []

    def _compute_metrics(sub_df: pd.DataFrame, dimension: str, segment: str) -> dict:
        """计算一组子集的IC/AUC"""
        n = len(sub_df)
        if n < 20:
            return {"dimension": dimension, "segment": segment, "n_samples": n}

        metrics = {"dimension": dimension, "segment": segment, "n_samples": n}

        # buy_reg IC (Spearman)
        if "pred_buy_reg" in sub_df.columns and "mae_20" in sub_df.columns:
            valid = sub_df[["pred_buy_reg", "mae_20"]].dropna()
            if len(valid) >= 20:
                from scipy.stats import spearmanr
                ic, p_val = spearmanr(valid["pred_buy_reg"], valid["mae_20"])
                metrics["buy_reg_ic"] = ic
                metrics["buy_reg_ic_pval"] = p_val

        # buy_cls AUC
        if "pred_buy_cls" in sub_df.columns and "buy_signal" in sub_df.columns:
            valid = sub_df[["pred_buy_cls", "buy_signal"]].dropna()
            if len(valid) >= 20 and valid["buy_signal"].nunique() >= 2:
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(valid["buy_signal"], valid["pred_buy_cls"])
                    metrics["buy_cls_auc"] = auc
                except ValueError:
                    pass

        # sell_reg IC (参照)
        if "pred_sell_reg" in sub_df.columns and "mfe_20" in sub_df.columns:
            valid = sub_df[["pred_sell_reg", "mfe_20"]].dropna()
            if len(valid) >= 20:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(valid["pred_sell_reg"], valid["mfe_20"])
                metrics["sell_reg_ic"] = ic

        # buy_signal 正类率
        if "buy_signal" in sub_df.columns:
            metrics["buy_signal_rate"] = sub_df["buy_signal"].mean()

        # mae_20 分布
        if "mae_20" in sub_df.columns:
            metrics["mae_20_mean"] = sub_df["mae_20"].mean()
            metrics["mae_20_median"] = sub_df["mae_20"].median()

        return metrics

    # 总体
    results.append(_compute_metrics(df, "总体", "all"))

    # 按价位
    for tier, grp in df.groupby("price_tier"):
        if pd.isna(tier):
            continue
        results.append(_compute_metrics(grp, "价位", str(tier)))

    # 按有无buy_cluster
    if "has_buy_cluster" in df.columns:
        for has_bc, grp in df.groupby("has_buy_cluster"):
            results.append(_compute_metrics(grp, "buy_cluster", f"has={has_bc}"))

    # 按波动率
    for tier, grp in df.groupby("vol_tier"):
        results.append(_compute_metrics(grp, "波动率", str(tier)))

    # 按月份
    for month, grp in df.groupby("month"):
        results.append(_compute_metrics(grp, "月份", month))

    return pd.DataFrame(results)


# ==================== 3. 校准分析 ====================
def analyze_calibration(test_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    buy_cls 预测概率 vs 真实命中率校准分析

    分桶: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    每桶统计: 样本数、预测均值、真实命中率、偏差
    """
    df = test_df[test_df["obs_day"] == 1].copy()

    if "pred_buy_cls" not in df.columns or "buy_signal" not in df.columns:
        print("  跳过校准分析：缺少pred_buy_cls或buy_signal列")
        return pd.DataFrame()

    valid = df[["pred_buy_cls", "buy_signal", "obs_close", "obs_date"]].dropna()
    print(f"  校准分析样本数: {len(valid)}")

    # 分桶
    valid["prob_bin"] = pd.cut(valid["pred_buy_cls"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)

    calibration = []
    for prob_bin, grp in valid.groupby("prob_bin"):
        n = len(grp)
        if n == 0:
            continue
        calibration.append({
            "prob_bin": str(prob_bin),
            "n_samples": n,
            "pred_mean": grp["pred_buy_cls"].mean(),
            "actual_rate": grp["buy_signal"].mean(),
            "bias": grp["buy_signal"].mean() - grp["pred_buy_cls"].mean(),
            "n_positive": grp["buy_signal"].sum(),
            "n_negative": n - grp["buy_signal"].sum(),
        })

    # 分层校准：按价位
    valid["price_tier"] = pd.cut(
        valid["obs_close"],
        bins=[0, 10, 20, 50, 100, 2000],
        labels=["<10", "10-20", "20-50", "50-100", ">100"],
    )

    stratified_cal = []
    for tier, tier_grp in valid.groupby("price_tier"):
        if pd.isna(tier) or len(tier_grp) < 30:
            continue
        tier_grp = tier_grp.copy()
        tier_grp["prob_bin"] = pd.cut(tier_grp["pred_buy_cls"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
        for prob_bin, grp in tier_grp.groupby("prob_bin"):
            n = len(grp)
            if n == 0:
                continue
            stratified_cal.append({
                "price_tier": str(tier),
                "prob_bin": str(prob_bin),
                "n_samples": n,
                "pred_mean": grp["pred_buy_cls"].mean(),
                "actual_rate": grp["buy_signal"].mean(),
                "bias": grp["buy_signal"].mean() - grp["pred_buy_cls"].mean(),
            })

    return pd.DataFrame(calibration), pd.DataFrame(stratified_cal)


# ==================== 可视化 ====================
def plot_label_drift(drift_df: pd.DataFrame, output_dir: str):
    """绘制标签漂移图"""
    os.makedirs(output_dir, exist_ok=True)

    # 只画总体和月份的buy_signal_rate
    overall = drift_df[drift_df["dimension"] == "总体"].copy()
    monthly = drift_df[drift_df["dimension"] == "月份"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Buy-Side Label Drift Analysis", fontsize=14, fontweight="bold")

    # 1. 总体正类率对比
    ax = axes[0]
    splits = ["train", "val", "test"]
    rates = [overall[overall["segment"] == s]["buy_signal_rate"].values[0] for s in splits if len(overall[overall["segment"] == s]) > 0]
    ax.bar(range(len(rates)), rates, color=["steelblue", "orange", "green"][:len(rates)], alpha=0.7)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(splits[:len(rates)])
    ax.set_ylabel("buy_signal positive rate")
    ax.set_title("Positive Rate by Split")
    for i, r in enumerate(rates):
        ax.text(i, r + 0.01, f"{r:.1%}", ha="center", fontsize=10)

    # 2. mae_20分布对比
    ax = axes[1]
    for s, color in zip(splits, ["steelblue", "orange", "green"]):
        row = overall[overall["segment"] == s]
        if len(row) > 0:
            row = row.iloc[0]
            ax.barh(s, row["mae_20_mean"], xerr=row["mae_20_std"], color=color, alpha=0.7, capsize=5)
    ax.set_xlabel("mae_20")
    ax.set_title("MAE_20 Distribution by Split")

    # 3. 月度正类率趋势
    ax = axes[2]
    if not monthly.empty:
        for s, color, marker in zip(splits, ["steelblue", "orange", "green"], ["o", "s", "^"]):
            sub = monthly[monthly["segment"].str.startswith(s)].copy()
            if not sub.empty:
                sub["month"] = sub["segment"].str.replace(f"{s}_", "")
                sub = sub.sort_values("month")
                ax.plot(sub["month"], sub["buy_signal_rate"], marker=marker, color=color, label=s, alpha=0.7)
        ax.set_xlabel("Month")
        ax.set_ylabel("buy_signal positive rate")
        ax.set_title("Monthly Positive Rate Trend")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(output_dir, "label_drift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_stratified_performance(strat_df: pd.DataFrame, output_dir: str):
    """绘制分层表现图"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Buy-Side Stratified Performance", fontsize=14, fontweight="bold")

    for ax, dim in zip(axes.flat, ["价位", "buy_cluster", "波动率", "月份"]):
        sub = strat_df[strat_df["dimension"] == dim].copy()
        if sub.empty or "buy_reg_ic" not in sub.columns:
            ax.set_title(f"{dim} - no data")
            continue

        sub = sub.dropna(subset=["buy_reg_ic"])
        if sub.empty:
            ax.set_title(f"{dim} - no valid IC")
            continue

        x = range(len(sub))
        width = 0.35
        if "sell_reg_ic" in sub.columns:
            ax.bar([i - width/2 for i in x], sub["sell_reg_ic"].values, width, label="sell_reg IC", color="steelblue", alpha=0.7)
        ax.bar([i + width/2 for i in x], sub["buy_reg_ic"].values, width, label="buy_reg IC", color="orange", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["segment"].values, rotation=45, ha="right", fontsize=8)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.set_ylabel("IC (Spearman)")
        ax.set_title(f"IC by {dim}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "stratified_performance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_calibration(cal_df: pd.DataFrame, strat_cal_df: pd.DataFrame, output_dir: str):
    """绘制校准图"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Buy-CLS Calibration Analysis", fontsize=14, fontweight="bold")

    # 1. 总体校准
    ax = axes[0]
    if not cal_df.empty:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
        ax.plot(cal_df["pred_mean"], cal_df["actual_rate"], "o-", color="steelblue", label="buy_cls", markersize=6)
        ax.fill_between(cal_df["pred_mean"],
                        cal_df["actual_rate"] - 0.02,
                        cal_df["actual_rate"] + 0.02,
                        alpha=0.2, color="steelblue")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual positive rate")
        ax.set_title("Overall Calibration")
        ax.legend()

    # 2. 按价位分层校准
    ax = axes[1]
    if not strat_cal_df.empty:
        for tier in strat_cal_df["price_tier"].unique():
            sub = strat_cal_df[strat_cal_df["price_tier"] == tier]
            ax.plot(sub["pred_mean"], sub["actual_rate"], "o-", label=tier, markersize=4, alpha=0.7)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual positive rate")
        ax.set_title("Calibration by Price Tier")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "calibration.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


# ==================== 主函数 ====================
def main(args):
    print("=" * 60)
    print("Buy侧专项诊断")
    print("=" * 60)

    os.makedirs(DIAGNOSIS_DIR, exist_ok=True)
    fig_dir = os.path.join(DIAGNOSIS_DIR, "figures")

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    dataset_path = os.path.join(OUTPUT_DIR, "dataset.parquet")
    test_path = os.path.join(OUTPUT_DIR, "test_predictions.parquet")

    dataset = pd.read_parquet(dataset_path)
    print(f"  dataset: {len(dataset)} 行")

    test_df = pd.read_parquet(test_path)
    test_df["obs_date"] = pd.to_datetime(test_df["obs_date"])
    print(f"  test_predictions: {len(test_df)} 行")

    # 2. 标签漂移分析
    print("\n[2/4] 标签漂移分析...")
    drift_df = analyze_label_drift(dataset)
    drift_path = os.path.join(DIAGNOSIS_DIR, "label_drift.csv")
    drift_df.to_csv(drift_path, index=False)
    print(f"  保存: {drift_path}")

    # 打印关键发现
    overall = drift_df[drift_df["dimension"] == "总体"]
    print("\n  === 总体标签分布 ===")
    for _, row in overall.iterrows():
        print(f"    {row['segment']:6s}: n={row['n_samples']:>7.0f}, "
              f"buy_signal_rate={row['buy_signal_rate']:.2%}, "
              f"mae_20_mean={row['mae_20_mean']:.4f}, "
              f"mae_20_median={row['mae_20_median']:.4f}")

    monthly = drift_df[drift_df["dimension"] == "月份"]
    if not monthly.empty:
        print("\n  === 月度正类率趋势 ===")
        for _, row in monthly.iterrows():
            print(f"    {row['segment']:20s}: rate={row['buy_signal_rate']:.2%}, mae_20_mean={row['mae_20_mean']:.4f}")

    # 3. 分层表现分析
    print("\n[3/4] 分层表现分析...")
    strat_df = analyze_stratified_performance(test_df)
    strat_path = os.path.join(DIAGNOSIS_DIR, "stratified_performance.csv")
    strat_df.to_csv(strat_path, index=False)
    print(f"  保存: {strat_path}")

    # 打印关键发现
    print("\n  === buy_reg IC 分层 ===")
    for dim in ["总体", "价位", "buy_cluster", "波动率", "月份"]:
        sub = strat_df[strat_df["dimension"] == dim]
        if sub.empty:
            continue
        print(f"\n    [{dim}]")
        for _, row in sub.iterrows():
            ic_str = f"buy_IC={row.get('buy_reg_ic', np.nan):.3f}" if pd.notna(row.get("buy_reg_ic")) else "buy_IC=N/A"
            sell_str = f"sell_IC={row.get('sell_reg_ic', np.nan):.3f}" if pd.notna(row.get("sell_reg_ic")) else ""
            auc_str = f"buy_AUC={row.get('buy_cls_auc', np.nan):.3f}" if pd.notna(row.get("buy_cls_auc")) else ""
            print(f"      {row['segment']:20s}: n={row['n_samples']:>5}, {ic_str} {sell_str} {auc_str}")

    # 4. 校准分析
    print("\n[4/4] 校准分析...")
    cal_df, strat_cal_df = analyze_calibration(test_df)
    cal_path = os.path.join(DIAGNOSIS_DIR, "calibration.csv")
    cal_df.to_csv(cal_path, index=False)
    print(f"  保存: {cal_path}")

    strat_cal_path = os.path.join(DIAGNOSIS_DIR, "calibration_stratified.csv")
    strat_cal_df.to_csv(strat_cal_path, index=False)
    print(f"  保存: {strat_cal_path}")

    # 打印校准结果
    if not cal_df.empty:
        print("\n  === 校准分析 ===")
        for _, row in cal_df.iterrows():
            print(f"    {row['prob_bin']:15s}: n={row['n_samples']:>5}, "
                  f"pred={row['pred_mean']:.3f}, actual={row['actual_rate']:.3f}, "
                  f"bias={row['bias']:+.3f}")

    # 5. 可视化
    if not args.no_plot:
        print("\n[可视化]...")
        plot_label_drift(drift_df, fig_dir)
        plot_stratified_performance(strat_df, fig_dir)
        if not cal_df.empty:
            plot_calibration(cal_df, strat_cal_df, fig_dir)

    # 6. 诊断结论
    print(f"\n{'='*60}")
    print("Buy侧诊断结论")
    print(f"{'='*60}")

    # 标签漂移
    train_rate = overall[overall["segment"] == "train"]["buy_signal_rate"].values
    test_rate = overall[overall["segment"] == "test"]["buy_signal_rate"].values
    if len(train_rate) > 0 and len(test_rate) > 0:
        rate_shift = test_rate[0] - train_rate[0]
        print(f"  标签漂移: buy_signal正类率 train={train_rate[0]:.2%} → test={test_rate[0]:.2%}, "
              f"偏移={rate_shift:+.2%}")

    # buy IC衰减
    overall_strat = strat_df[strat_df["dimension"] == "总体"]
    if not overall_strat.empty and "buy_reg_ic" in overall_strat.columns:
        buy_ic = overall_strat["buy_reg_ic"].values[0]
        sell_ic = overall_strat.get("sell_reg_ic", pd.Series([np.nan])).values[0]
        if pd.notna(sell_ic) and pd.notna(buy_ic):
            ratio = buy_ic / sell_ic if sell_ic != 0 else 0
            print(f"  IC对比: buy_reg={buy_ic:.3f} vs sell_reg={sell_ic:.3f} (比值={ratio:.2f})")

    # 校准
    if not cal_df.empty and "bias" in cal_df.columns:
        avg_bias = cal_df["bias"].mean()
        max_bias = cal_df["bias"].abs().max()
        print(f"  校准偏差: 平均={avg_bias:+.3f}, 最大绝对={max_bias:.3f}")
        if avg_bias > 0.05:
            print(f"  ⚠️  模型整体低估正类概率，实际命中率高于预测")
        elif avg_bias < -0.05:
            print(f"  ⚠️  模型整体高估正类概率，实际命中率低于预测")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buy侧专项诊断")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    args = parser.parse_args()
    main(args)
