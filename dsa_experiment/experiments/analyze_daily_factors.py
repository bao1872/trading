#!/usr/bin/env python3
"""
日线因子单因子分析：IC/分组/相关性/增量分析

Purpose: 分析日线因子与买卖点标签的相关性，评估因子有效性
Inputs: output/daily_factor_table.parquet
Outputs: 终端报告, output/daily_factor_analysis/
How to Run:
    python dsa_experiment/analyze_daily_factors.py
Examples:
    python dsa_experiment/analyze_daily_factors.py
Side Effects: 只读操作，输出 CSV 文件
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

FACTOR_COLS_24 = [
    "dsa_dir", "prev_pivot_code", "last_confirmed_high", "last_confirmed_low",
    "dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct", "current_stage_bars", "prev_stage_bars",
    "bars_since_last_high", "bars_since_last_low",
    "prev_stage_amp_pct", "current_stage_ret_pct", "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct",
    "bbmacd", "bbmacd_minus_avg", "bbmacd_state", "bbmacd_band_pos_01",
    "bbmacd_bandwidth_zscore", "bbmacd_cross_upper", "bbmacd_cross_lower",
    "trend_align_momo",
]

EXTRA_FACTOR_COLS = [
    "vol_zscore_5", "vol_zscore_10", "vol_zscore_20",
    "vol_ratio_10", "days_since_vol_spike",
    "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio",
    "dsa_dir_age", "bbmacd_sign", "bbmacd_slope_3",
]

ALL_FACTOR_COLS = FACTOR_COLS_24 + EXTRA_FACTOR_COLS

LABEL_COLS = [
    "future_high_ret_5", "future_low_ret_5",
    "future_high_ret_10", "future_low_ret_10",
]

FACTOR_CATEGORIES = {
    "趋势类": ["dsa_dir", "price_vs_dsa_vwap_pct", "dsa_dir_age", "bbmacd_sign", "bbmacd_slope_3"],
    "位置类": ["dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct",
              "bars_since_last_high", "bars_since_last_low", "current_pullback_from_stage_extreme_pct"],
    "动量类": ["bbmacd", "bbmacd_minus_avg", "bbmacd_state", "bbmacd_band_pos_01", "bbmacd_bandwidth_zscore"],
    "成交量类": ["vol_zscore_5", "vol_zscore_10", "vol_zscore_20", "vol_ratio_10",
                "days_since_vol_spike", "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio"],
    "节奏类": ["current_stage_bars", "current_stage_ret_pct", "current_stage_amp_pct"],
    "交互类": ["prev_pivot_code", "prev_stage_amp_pct", "bbmacd_cross_upper", "bbmacd_cross_lower", "trend_align_momo"],
}


def compute_ic(df: pd.DataFrame, factor_col: str, label_col: str) -> dict:
    valid = df[[factor_col, label_col]].dropna()
    if len(valid) < 30:
        return {"ic": np.nan, "p_value": np.nan, "n": len(valid)}
    ic, p = stats.spearmanr(valid[factor_col], valid[label_col])
    return {"ic": ic, "p_value": p, "n": len(valid)}


def compute_daily_ic(df: pd.DataFrame, factor_col: str, label_col: str) -> dict:
    daily_ics = []
    for sel_date, day_df in df.groupby("selection_date"):
        valid = day_df[[factor_col, label_col]].dropna()
        if len(valid) < 10:
            continue
        ic, _ = stats.spearmanr(valid[factor_col], valid[label_col])
        daily_ics.append(ic)
    if not daily_ics:
        return {"mean_ic": np.nan, "std_ic": np.nan, "icir": np.nan, "n_days": 0}
    mean_ic = np.mean(daily_ics)
    std_ic = np.std(daily_ics)
    icir = mean_ic / std_ic if std_ic > 0 else 0
    return {"mean_ic": mean_ic, "std_ic": std_ic, "icir": icir, "n_days": len(daily_ics)}


def compute_quintile_report(df: pd.DataFrame, factor_col: str, label_col: str) -> pd.DataFrame:
    valid = df[[factor_col, label_col, "selection_date"]].dropna()
    if len(valid) < 50:
        return pd.DataFrame()

    def qcut_daily(g):
        if len(g) < 5:
            return g
        g = g.copy()
        g["quintile"] = pd.qcut(g[factor_col], 5, labels=False, duplicates="drop") + 1
        return g

    valid = valid.groupby("selection_date", group_keys=False).apply(qcut_daily)
    if "quintile" not in valid.columns:
        return pd.DataFrame()

    report = valid.groupby("quintile").agg(
        n=(label_col, "count"),
        mean_label=(label_col, "mean"),
        std_label=(label_col, "std"),
    ).reset_index()
    return report


def main():
    print("=" * 80)
    print("日线因子单因子分析")
    print("=" * 80)

    input_path = os.path.join(OUTPUT_DIR, "daily_factor_table.parquet")
    df = pd.read_parquet(input_path)
    print(f"\n  记录数: {len(df)}")
    print(f"  日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")

    out_dir = os.path.join(OUTPUT_DIR, "daily_factor_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. 全量 IC 分析 ──
    print("\n" + "=" * 80)
    print("1. 全量 IC 分析（Spearman）")
    print("=" * 80)

    ic_results = []
    for factor in ALL_FACTOR_COLS:
        if factor not in df.columns:
            continue
        row = {"factor": factor}
        for label in LABEL_COLS:
            ic_info = compute_ic(df, factor, label)
            row[f"ic_{label}"] = ic_info["ic"]
            row[f"p_{label}"] = ic_info["p_value"]
        ic_results.append(row)

    ic_df = pd.DataFrame(ic_results)

    print(f"\n  {'因子':<40} {'IC_high5':>9} {'IC_low5':>9} {'IC_high10':>9} {'IC_low10':>9}")
    print(f"  {'-'*80}")
    for _, row in ic_df.iterrows():
        print(f"  {row['factor']:<40} {row.get('ic_future_high_ret_5', np.nan):>8.4f} {row.get('ic_future_low_ret_5', np.nan):>8.4f} {row.get('ic_future_high_ret_10', np.nan):>8.4f} {row.get('ic_future_low_ret_10', np.nan):>8.4f}")

    ic_df.to_csv(os.path.join(out_dir, "ic_summary.csv"), index=False)

    # ── 2. 日频 IC 分析（ICIR） ──
    print("\n" + "=" * 80)
    print("2. 日频 IC 分析（ICIR）")
    print("=" * 80)

    daily_ic_results = []
    for factor in ALL_FACTOR_COLS:
        if factor not in df.columns:
            continue
        row = {"factor": factor}
        for label in ["future_high_ret_5", "future_low_ret_5"]:
            ic_info = compute_daily_ic(df, factor, label)
            row[f"mean_ic_{label}"] = ic_info["mean_ic"]
            row[f"icir_{label}"] = ic_info["icir"]
        daily_ic_results.append(row)

    daily_ic_df = pd.DataFrame(daily_ic_results)

    print(f"\n  {'因子':<40} {'IC_high5':>8} {'ICIR_h5':>8} {'IC_low5':>8} {'ICIR_l5':>8}")
    print(f"  {'-'*80}")
    for _, row in daily_ic_df.iterrows():
        print(f"  {row['factor']:<40} {row.get('mean_ic_future_high_ret_5', np.nan):>7.4f} {row.get('icir_future_high_ret_5', np.nan):>7.2f} {row.get('mean_ic_future_low_ret_5', np.nan):>7.4f} {row.get('icir_future_low_ret_5', np.nan):>7.2f}")

    daily_ic_df.to_csv(os.path.join(out_dir, "daily_ic_summary.csv"), index=False)

    # ── 3. 分类别 IC 汇总 ──
    print("\n" + "=" * 80)
    print("3. 分类别 IC 汇总")
    print("=" * 80)

    for cat_name, cat_factors in FACTOR_CATEGORIES.items():
        cat_rows = daily_ic_df[daily_ic_df["factor"].isin(cat_factors)]
        if cat_rows.empty:
            continue
        avg_ic_high = cat_rows["mean_ic_future_high_ret_5"].mean()
        avg_icir_high = cat_rows["icir_future_high_ret_5"].mean()
        avg_ic_low = cat_rows["mean_ic_future_low_ret_5"].mean()
        avg_icir_low = cat_rows["icir_future_low_ret_5"].mean()
        best_factor = cat_rows.loc[cat_rows["icir_future_high_ret_5"].abs().idxmax(), "factor"] if not cat_rows.empty else ""
        print(f"  {cat_name}: avg_IC_high5={avg_ic_high:.4f}, avg_ICIR_high5={avg_icir_high:.2f}, avg_IC_low5={avg_ic_low:.4f}, best={best_factor}")

    # ── 4. 成交量因子增量分析 ──
    print("\n" + "=" * 80)
    print("4. 成交量因子增量分析（日线特有 vs 周线因子）")
    print("=" * 80)

    vol_factors = ["vol_zscore_5", "vol_zscore_10", "vol_zscore_20", "vol_ratio_10",
                   "days_since_vol_spike", "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio"]
    weekly_factors = ["bbmacd", "price_vs_dsa_vwap_pct", "bars_since_last_low", "ret_to_last_low_pct"]

    print(f"\n  {'成交量因子':<25} {'IC_high5':>9} {'ICIR_h5':>8} {'IC_low5':>9} {'ICIR_l5':>8}")
    print(f"  {'-'*65}")
    for factor in vol_factors:
        row = daily_ic_df[daily_ic_df["factor"] == factor]
        if row.empty:
            continue
        row = row.iloc[0]
        print(f"  {factor:<25} {row.get('mean_ic_future_high_ret_5', np.nan):>8.4f} {row.get('icir_future_high_ret_5', np.nan):>7.2f} {row.get('mean_ic_future_low_ret_5', np.nan):>8.4f} {row.get('icir_future_low_ret_5', np.nan):>7.2f}")

    print(f"\n  对比：周线核心因子")
    print(f"  {'周线因子':<25} {'IC_high5':>9} {'ICIR_h5':>8} {'IC_low5':>9} {'ICIR_l5':>8}")
    print(f"  {'-'*65}")
    for factor in weekly_factors:
        row = daily_ic_df[daily_ic_df["factor"] == factor]
        if row.empty:
            continue
        row = row.iloc[0]
        print(f"  {factor:<25} {row.get('mean_ic_future_high_ret_5', np.nan):>8.4f} {row.get('icir_future_high_ret_5', np.nan):>7.2f} {row.get('mean_ic_future_low_ret_5', np.nan):>8.4f} {row.get('icir_future_low_ret_5', np.nan):>7.2f}")

    # ── 5. day_offset 效应分析 ──
    print("\n" + "=" * 80)
    print("5. day_offset 效应（周线触发后第几天买入效果最好）")
    print("=" * 80)

    print(f"\n  {'day_offset':>10} {'avg_high5':>10} {'avg_low5':>10} {'胜率(>0)':>10} {'样本数':>8}")
    print(f"  {'-'*55}")
    for offset in range(1, 6):
        sub = df[df["day_offset"] == offset]
        if sub.empty:
            continue
        avg_high = sub["future_high_ret_5"].mean()
        avg_low = sub["future_low_ret_5"].mean()
        win_rate = (sub["future_high_ret_5"] > 0).mean()
        n = len(sub)
        print(f"  {offset:>10} {avg_high:>9.2%} {avg_low:>9.2%} {win_rate:>9.0%} {n:>8}")

    # ── 6. 关键因子分组报告 ──
    print("\n" + "=" * 80)
    print("6. 关键因子分组报告（Top 5 ICIR 因子）")
    print("=" * 80)

    top_factors = daily_ic_df.nlargest(5, "icir_future_high_ret_5")["factor"].tolist()
    for factor in top_factors:
        print(f"\n  --- {factor} vs future_high_ret_5 ---")
        report = compute_quintile_report(df, factor, "future_high_ret_5")
        if report.empty:
            print("    (样本不足)")
            continue
        print(f"    {'分组':>4} {'样本':>6} {'均值':>8} {'标准差':>8}")
        print(f"    {'-'*30}")
        for _, row in report.iterrows():
            print(f"    {int(row['quintile']):>4} {int(row['n']):>6} {row['mean_label']:>7.2%} {row['std_label']:>7.2%}")

    print("\n" + "=" * 80)
    print("日线因子分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
