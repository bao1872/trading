#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子重要性分析与可视化

Purpose:
    分析4个GBDT模型的因子重要性，输出排名、跨折稳定性、因子类别贡献度。

Pipeline Position:
    训练流水线第三步（离线，一次性）。
    上游: 02_train_gbdt_models.py
    下游: —

Inputs:
    - stop_experiment/output/feature_importance.csv
    - stop_experiment/output/models_control/ (模型文件)
    - stop_experiment/output/dataset.parquet

Outputs:
    - stop_experiment/output/factor_importance_summary.csv
    - stop_experiment/output/figures/ (可视化图表)

How to Run:
    python stop_experiment/pipeline/03_factor_importance.py
    python stop_experiment/pipeline/03_factor_importance.py --no-shap  # 跳过SHAP（耗时）

Side Effects:
    - 只读操作，输出文件
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

from stop_experiment.pipeline.stop_config import OUTPUT_DIR, MODELS_DIR, FIGURES_DIR, DATASET_PATH, MODEL_SPECS
from stop_experiment.pipeline.factor_columns import ALL_FEATURE_COLS, FACTOR_CATEGORIES


def compute_importance_summary(imp_df: pd.DataFrame) -> pd.DataFrame:
    """因子重要性汇总 + 排名（v2: 无fold维度，直接按model_name聚合）"""
    # v2: 无fold维度，imp_df已经只有model_name+feature+gain
    summary = (
        imp_df.groupby(["model_name", "feature"])["gain"]
        .agg(["mean"])
        .reset_index()
        .rename(columns={"mean": "avg_gain"})
    )

    # 排名
    summary["rank"] = summary.groupby("model_name")["avg_gain"].rank(ascending=False).astype(int)

    # 因子类别
    category_map = {}
    for cat, cols in FACTOR_CATEGORIES.items():
        for col in cols:
            category_map[col] = cat
    summary["category"] = summary["feature"].map(category_map).fillna("other")

    # 归一化 gain（每个模型内 gain 占比）
    summary["gain_pct"] = summary.groupby("model_name")["avg_gain"].transform(
        lambda x: x / x.sum() * 100
    )

    return summary


def plot_top_features(summary: pd.DataFrame, top_n: int = 20):
    """Top N 因子重要性条形图（4个子图）"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Top 20 Feature Importance (Gain)", fontsize=14, fontweight="bold")

    for ax, (model_name, _) in zip(axes.flat, MODEL_SPECS.items()):
        model_data = summary[summary["model_name"] == model_name].nlargest(top_n, "avg_gain")
        colors = [CATEGORY_COLORS.get(c, "#888888") for c in model_data["category"]]
        ax.barh(range(top_n), model_data["avg_gain"].values, color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(model_data["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Avg Gain")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "top_features.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_category_contribution(summary: pd.DataFrame):
    """因子类别贡献度饼图"""
    cat_contrib = (
        summary.groupby(["model_name", "category"])["gain_pct"]
        .sum()
        .reset_index()
    )

    categories = sorted(cat_contrib["category"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Feature Category Contribution (%)", fontsize=14, fontweight="bold")

    for ax, model_name in zip(axes, MODEL_SPECS.keys()):
        model_data = cat_contrib[cat_contrib["model_name"] == model_name]
        values = [model_data[model_data["category"] == c]["gain_pct"].sum() for c in categories]
        colors = [CATEGORY_COLORS.get(c, "#888888") for c in categories]
        ax.pie(values, labels=categories, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title(model_name, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "category_contribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_sell_vs_buy(summary: pd.DataFrame):
    """卖点模型 vs 买点模型 因子重要性对比"""
    # 取 sell_reg 和 buy_reg 的 top 30
    sell_imp = summary[summary["model_name"] == "sell_reg"][["feature", "avg_gain", "rank"]].rename(
        columns={"avg_gain": "sell_gain", "rank": "sell_rank"}
    )
    buy_imp = summary[summary["model_name"] == "buy_reg"][["feature", "avg_gain", "rank"]].rename(
        columns={"avg_gain": "buy_gain", "rank": "buy_rank"}
    )
    merged = sell_imp.merge(buy_imp, on="feature", how="outer").fillna(0)

    # Top 20 by max rank
    merged["max_rank"] = merged[["sell_rank", "buy_rank"]].min(axis=1)
    top = merged.nsmallest(25, "max_rank")

    fig, ax = plt.subplots(figsize=(10, 10))
    y_pos = range(len(top))
    bar_height = 0.35
    ax.barh([y - bar_height/2 for y in y_pos], top["sell_gain"], bar_height, label="sell_reg", color="#e74c3c", alpha=0.8)
    ax.barh([y + bar_height/2 for y in y_pos], top["buy_gain"], bar_height, label="buy_reg", color="#3498db", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Gain")
    ax.set_title("Sell Model vs Buy Model Feature Importance", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "sell_vs_buy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def plot_stability(summary: pd.DataFrame, top_n: int = 20):
    """因子重要性条形图（v2: 无fold维度，改用gain分布）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Feature Importance (Gain) by Category", fontsize=14, fontweight="bold")

    for ax, model_name in zip(axes.flat, MODEL_SPECS.keys()):
        model_data = summary[summary["model_name"] == model_name].copy()
        model_data = model_data[model_data["avg_gain"] > 0]

        # 按类别着色
        colors = [CATEGORY_COLORS.get(c, "#888888") for c in model_data["category"]]
        ax.barh(range(len(model_data)), model_data["avg_gain"].values, color=colors)
        ax.set_yticks(range(min(20, len(model_data))))
        top = model_data.nlargest(20, "avg_gain")
        ax.set_yticklabels(top["feature"].values, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Gain")
        ax.set_title(model_name, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "importance_by_category.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def compute_shap(model_name: str, model_path: str, dataset_path: str, n_samples: int = 5000):
    """计算 SHAP 值并绘制 beeswarm 图"""
    try:
        import shap
    except ImportError:
        print(f"  shap 库未安装，跳过 SHAP 分析")
        return

    import lightgbm as lgb

    model = lgb.Booster(model_file=model_path)
    df = pd.read_parquet(dataset_path)
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]

    # 采样
    df_valid = df.dropna(subset=["mfe_20", "mae_20"])
    if len(df_valid) > n_samples:
        sample = df_valid.sample(n_samples, random_state=42)
    else:
        sample = df_valid

    X = sample[feature_cols]

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # beeswarm plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, max_display=20, show=False)
    plt.title(f"SHAP - {model_name}", fontweight="bold")
    path = os.path.join(FIGURES_DIR, f"shap_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


# 类别颜色映射
CATEGORY_COLORS = {
    "slc": "#e74c3c",
    "trend": "#3498db",
    "position": "#2ecc71",
    "momentum": "#f39c12",
    "volume": "#9b59b6",
    "risk": "#1abc9c",
    "rhythm": "#e67e22",
    "dynamic": "#34495e",
    "derived": "#95a5a6",
}


def main(args):
    print("=" * 60)
    print("因子重要性分析")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. 加载重要性数据
    print("\n[1/4] 加载因子重要性...")
    imp_path = os.path.join(MODELS_DIR, "feature_importance.csv")
    imp_df = pd.read_csv(imp_path)
    print(f"  记录数: {len(imp_df)}, 模型数: {imp_df['model_name'].nunique()}")

    # 2. 汇总
    print("\n[2/4] 计算因子重要性汇总...")
    summary = compute_importance_summary(imp_df)

    summary_path = os.path.join(OUTPUT_DIR, "factor_importance_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  保存: {summary_path}")

    # 打印各模型 Top 10
    for model_name in MODEL_SPECS:
        top = summary[summary["model_name"] == model_name].nsmallest(10, "rank")
        print(f"\n  {model_name} Top 10:")
        for _, row in top.iterrows():
            print(f"    {row['rank']:2d}. {row['feature']:40s} gain={row['avg_gain']:.1f} "
                  f"pct={row['gain_pct']:.1f}% [{row['category']}]")

    # 因子类别贡献度
    print(f"\n  因子类别贡献度:")
    cat_contrib = summary.groupby(["model_name", "category"])["gain_pct"].sum().unstack("model_name")
    print(cat_contrib.round(1).to_string())

    # 3. 可视化
    print("\n[3/4] 生成可视化...")
    plot_top_features(summary)
    plot_category_contribution(summary)
    plot_sell_vs_buy(summary)
    plot_stability(summary)

    # 4. SHAP（可选，使用final模型）
    if not args.no_shap:
        print("\n[4/4] SHAP 分析...")
        for model_name in MODEL_SPECS:
            # 优先使用final模型
            model_path = os.path.join(MODELS_DIR, f"{model_name}_final.txt")
            if not os.path.exists(model_path):
                model_path = os.path.join(MODELS_DIR, f"{model_name}.txt")
            if os.path.exists(model_path):
                compute_shap(model_name, model_path, DATASET_PATH, n_samples=5000)
    else:
        print("\n[4/4] SHAP 分析已跳过")

    print(f"\n完成！所有输出保存在 {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="因子重要性分析")
    parser.add_argument("--no-shap", action="store_true", help="跳过SHAP分析（耗时）")
    args = parser.parse_args()
    main(args)
