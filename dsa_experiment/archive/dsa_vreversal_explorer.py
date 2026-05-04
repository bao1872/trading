#!/usr/bin/env python3
"""
DSA V型反转触发点因子探索实验

Purpose: 探索 stock_dsa_vreversal_results 中 24 因子与未来标签的相关性
Inputs: stock_dsa_vreversal_results (选股结果表)
Outputs: CSV (IC汇总、分组收益、目标分布、因子相关性、时间距离分析)
How to Run:
    python analysis/dsa_vreversal_explorer.py                                # 默认全部
    python analysis/dsa_vreversal_explorer.py --start 2023-01-01 --end 2025-12-31
    python analysis/dsa_vreversal_explorer.py --sample-limit 500             # 快速测试
    python analysis/dsa_vreversal_explorer.py --only-reversal                # 只分析有反转的
    python analysis/dsa_vreversal_explorer.py --output-dir /tmp/dsa_analysis
Examples:
    python analysis/dsa_vreversal_explorer.py
    python analysis/dsa_vreversal_explorer.py --sample-limit 200
Side Effects: 只读操作，不写数据库；输出 CSV 到指定目录
"""
import sys
import os
import argparse
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

NUMERIC_FACTORS = [
    "dsa_dir",
    "prev_pivot_code",
    "last_confirmed_high",
    "last_confirmed_low",
    "dsa_pivot_pos_01",
    "ret_to_last_high_pct",
    "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct",
    "current_stage_bars",
    "prev_stage_bars",
    "bars_since_last_high",
    "bars_since_last_low",
    "prev_stage_amp_pct",
    "current_stage_ret_pct",
    "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct",
    "bbmacd",
    "bbmacd_minus_avg",
    "bbmacd_band_pos_01",
    "bbmacd_bandwidth_zscore",
    "bbmacd_cross_upper",
    "bbmacd_cross_lower",
]

CATEGORICAL_FACTORS = ["bbmacd_state", "trend_align_momo"]

TARGET_COLUMNS = [
    "high_ret_per_bar",
    "next_reversal_high_ret",
    "interim_low_ret",
    "bars_to_reversal_high",
    "bars_to_interim_low",
    "bars_to_reversal",
]

DERIVED_TARGETS = [
    "interim_low_ret_per_bar",
    "risk_reward_ratio",
    "is_profitable",
    "is_low_risk",
]

FACTOR_LABELS = {
    "dsa_dir": "DSA方向",
    "prev_pivot_code": "前一枢轴编码",
    "last_confirmed_high": "最近确认高点",
    "last_confirmed_low": "最近确认低点",
    "dsa_pivot_pos_01": "DSA枢轴位置[0,1]",
    "ret_to_last_high_pct": "相对高点收益率",
    "ret_to_last_low_pct": "相对低点收益率",
    "price_vs_dsa_vwap_pct": "VWAP偏离度",
    "current_stage_bars": "当前阶段bar数",
    "prev_stage_bars": "前一阶段bar数",
    "bars_since_last_high": "距高点bar数",
    "bars_since_last_low": "距低点bar数",
    "prev_stage_amp_pct": "前一阶段振幅%",
    "current_stage_ret_pct": "当前阶段收益率%",
    "current_stage_amp_pct": "当前阶段振幅%",
    "current_pullback_from_stage_extreme_pct": "当前回撤%",
    "bbmacd": "BBMacd值",
    "bbmacd_minus_avg": "BBMacd减均值",
    "bbmacd_state": "BBMacd状态",
    "bbmacd_band_pos_01": "BBMacd带内位置[0,1]",
    "bbmacd_bandwidth_zscore": "BBMacd带宽Z-Score",
    "bbmacd_cross_upper": "上穿上轨信号",
    "bbmacd_cross_lower": "下穿下轨信号",
    "trend_align_momo": "趋势-动量对齐",
}

TARGET_LABELS = {
    "high_ret_per_bar": "每bar高点收益率(资金效率)",
    "next_reversal_high_ret": "反转高点收益率",
    "interim_low_ret": "中间低点收益率",
    "bars_to_reversal_high": "距反转高点bar数",
    "bars_to_interim_low": "距中间低点bar数",
    "bars_to_reversal": "距dsa_dir反转bar数",
    "interim_low_ret_per_bar": "每bar低点收益率",
    "risk_reward_ratio": "风险收益比",
    "is_profitable": "是否盈利>10%",
    "is_low_risk": "是否低风险(回撤<5%)",
}

MIN_CROSS_SECTION_SIZE = 5
MIN_IC_PERIODS = 10


def fetch_vreversal_records(
    start_date: date,
    end_date: date,
    sample_limit: Optional[int] = None,
    only_reversal: bool = False,
) -> pd.DataFrame:
    """从 stock_dsa_vreversal_results 读取触发点记录"""
    where = "WHERE selection_date BETWEEN :start_date AND :end_date"
    if only_reversal:
        where += " AND bars_to_reversal IS NOT NULL"

    sql = text(f"""
        SELECT *
        FROM stock_dsa_vreversal_results
        {where}
        ORDER BY selection_date, ts_code
    """)
    df = pd.read_sql(
        sql,
        engine,
        params={
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        },
    )
    if df.empty:
        return df
    if sample_limit and len(df) > sample_limit:
        n_dates = df["selection_date"].nunique()
        per_date = max(1, sample_limit // max(n_dates, 1))
        df = df.groupby("selection_date", group_keys=False).head(per_date).reset_index(drop=True)
    return df


def build_derived_targets(df: pd.DataFrame) -> pd.DataFrame:
    """构建衍生目标变量（含每bar收益率，反映资金效率）"""
    df = df.copy()
    df["high_ret_per_bar"] = np.where(
        (df["bars_to_reversal_high"].notna()) & (df["bars_to_reversal_high"] > 0),
        df["next_reversal_high_ret"] / df["bars_to_reversal_high"],
        np.nan,
    )
    df["interim_low_ret_per_bar"] = np.where(
        (df["bars_to_interim_low"].notna()) & (df["bars_to_interim_low"] > 0),
        df["interim_low_ret"] / df["bars_to_interim_low"],
        np.nan,
    )
    df["risk_reward_ratio"] = np.where(
        (df["next_reversal_high_ret"].notna()) & (df["interim_low_ret"].notna()) & (df["interim_low_ret"] != 0),
        df["next_reversal_high_ret"] / df["interim_low_ret"].abs(),
        np.nan,
    )
    df["is_profitable"] = np.where(
        df["next_reversal_high_ret"].notna(),
        (df["next_reversal_high_ret"] > 0.1).astype(float),
        np.nan,
    )
    df["is_low_risk"] = np.where(
        df["interim_low_ret"].notna(),
        (df["interim_low_ret"].abs() < 0.05).astype(float),
        np.nan,
    )
    return df


def run_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """目标变量分布统计"""
    all_targets = TARGET_COLUMNS + DERIVED_TARGETS
    rows = []
    for col in all_targets:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        row = {
            "target": col,
            "label": TARGET_LABELS.get(col, col),
            "n": len(series),
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "p25": series.quantile(0.25),
            "p75": series.quantile(0.75),
            "max": series.max(),
            "skew": series.skew(),
            "kurt": series.kurtosis(),
        }
        if col in ("is_profitable", "is_low_risk"):
            row["positive_ratio"] = series.mean()
        elif col in ("next_reversal_high_ret", "interim_low_ret", "high_ret_per_bar", "risk_reward_ratio"):
            row["positive_ratio"] = (series > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def run_yearly_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """按年份的目标变量分布"""
    if "selection_date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["year"] = pd.to_datetime(df["selection_date"]).dt.year
    rows = []
    for year, group in df.groupby("year"):
        for col in TARGET_COLUMNS:
            if col not in group.columns:
                continue
            series = group[col].dropna()
            if series.empty:
                continue
            rows.append({
                "year": year,
                "target": col,
                "n": len(series),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
            })
    return pd.DataFrame(rows)


def calc_cross_section_ic(
    df: pd.DataFrame, factor_name: str, target_col: str
) -> Optional[Dict]:
    """计算截面 IC"""
    sub = df[["selection_date", factor_name, target_col]].dropna()
    if len(sub) < MIN_CROSS_SECTION_SIZE:
        return None

    ic_list = []
    for dt, group in sub.groupby("selection_date"):
        if len(group) < MIN_CROSS_SECTION_SIZE:
            continue
        spearman_ic, _ = stats.spearmanr(group[factor_name], group[target_col])
        pearson_ic, _ = stats.pearsonr(group[factor_name], group[target_col])
        if not np.isnan(spearman_ic):
            ic_list.append({
                "selection_date": dt,
                "spearman_ic": spearman_ic,
                "pearson_ic": pearson_ic,
                "n": len(group),
            })

    if len(ic_list) < MIN_IC_PERIODS:
        return None

    ic_df = pd.DataFrame(ic_list)
    ic_mean = ic_df["spearman_ic"].mean()
    ic_std = ic_df["spearman_ic"].std()

    return {
        "factor": factor_name,
        "factor_label": FACTOR_LABELS.get(factor_name, factor_name),
        "target": target_col,
        "target_label": TARGET_LABELS.get(target_col, target_col),
        "n_periods": len(ic_df),
        "avg_n": ic_df["n"].mean(),
        "ic_mean": ic_mean,
        "ic_median": ic_df["spearman_ic"].median(),
        "ic_std": ic_std,
        "icir": ic_mean / ic_std if ic_std > 0 else np.nan,
        "t_stat": ic_mean / (ic_std / np.sqrt(len(ic_df))) if ic_std > 0 else np.nan,
        "ic_positive_ratio": (ic_df["spearman_ic"] > 0).mean(),
        "ic_abs_mean": ic_df["spearman_ic"].abs().mean(),
        "pearson_ic_mean": ic_df["pearson_ic"].mean(),
        "ic_detail": ic_df,
    }


def run_ic_analysis(df: pd.DataFrame, target_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """IC 分析"""
    ic_summary_rows = []
    ic_ts_rows = []

    for factor in NUMERIC_FACTORS:
        for target in target_cols:
            if target not in df.columns:
                continue
            result = calc_cross_section_ic(df, factor, target)
            if result is None:
                continue
            ic_summary_rows.append({k: v for k, v in result.items() if k != "ic_detail"})
            for _, row in result["ic_detail"].iterrows():
                ic_ts_rows.append({
                    "factor": factor,
                    "target": target,
                    "selection_date": row["selection_date"],
                    "spearman_ic": row["spearman_ic"],
                    "pearson_ic": row["pearson_ic"],
                    "n": row["n"],
                })

    return pd.DataFrame(ic_summary_rows), pd.DataFrame(ic_ts_rows)


def run_group_analysis(
    df: pd.DataFrame, target_cols: List[str], n_groups: int = 5
) -> pd.DataFrame:
    """分组收益分析（全局分组，不逐日期迭代，提升性能）"""
    results = []
    for factor in NUMERIC_FACTORS:
        for target in target_cols:
            if target not in df.columns:
                continue
            sub = df[[factor, target]].dropna()
            if len(sub) < n_groups * MIN_CROSS_SECTION_SIZE:
                continue
            try:
                sub = sub.copy()
                sub["group"] = pd.qcut(
                    sub[factor], q=n_groups, labels=False, duplicates="drop"
                )
            except ValueError:
                continue
            actual_groups = sorted(sub["group"].unique())
            if len(actual_groups) < 2:
                continue
            for g in actual_groups:
                g_data = sub[sub["group"] == g]
                if len(g_data) == 0:
                    continue
                results.append({
                    "factor": factor,
                    "factor_label": FACTOR_LABELS.get(factor, factor),
                    "target": target,
                    "target_label": TARGET_LABELS.get(target, target),
                    "group": int(g),
                    "group_label": f"Q{int(g) + 1}",
                    "mean_val": g_data[target].mean(),
                    "median_val": g_data[target].median(),
                    "win_rate": (g_data[target] > 0).mean() if target not in ("bars_to_reversal_high", "bars_to_interim_low", "bars_to_reversal") else np.nan,
                    "n": len(g_data),
                })
    return pd.DataFrame(results)


def run_group_summary(group_df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """分组汇总"""
    if group_df.empty:
        return pd.DataFrame()
    agg = (
        group_df.groupby(["factor", "factor_label", "target", "target_label", "group_label"])
        .agg(
            mean_val=("mean_val", "mean"),
            median_val=("median_val", "mean"),
            win_rate=("win_rate", "mean"),
            avg_n=("n", "mean"),
        )
        .reset_index()
    )
    q1 = agg[agg["group_label"] == "Q1"][["factor", "target", "mean_val"]].rename(
        columns={"mean_val": "Q1_val"}
    )
    q_last = agg.groupby(["factor", "target"]).last().reset_index()[["factor", "target", "mean_val"]].rename(
        columns={"mean_val": "Q_last_val"}
    )
    ls = q1.merge(q_last, on=["factor", "target"], how="inner")
    ls["long_short"] = ls["Q_last_val"] - ls["Q1_val"]
    result = agg.merge(
        ls[["factor", "target", "long_short"]],
        on=["factor", "target"],
        how="left",
    )
    return result


def run_categorical_analysis(
    df: pd.DataFrame, target_cols: List[str]
) -> pd.DataFrame:
    """分类型因子分析"""
    results = []
    for cat_factor in CATEGORICAL_FACTORS:
        if cat_factor not in df.columns:
            continue
        for target in target_cols:
            if target not in df.columns:
                continue
            sub = df[[cat_factor, target]].dropna()
            if sub.empty:
                continue
            for cat_val, group in sub.groupby(cat_factor):
                if len(group) < 3:
                    continue
                results.append({
                    "factor": cat_factor,
                    "factor_label": FACTOR_LABELS.get(cat_factor, cat_factor),
                    "target": target,
                    "target_label": TARGET_LABELS.get(target, target),
                    "category": cat_val,
                    "mean_val": group[target].mean(),
                    "median_val": group[target].median(),
                    "win_rate": (group[target] > 0).mean() if target not in ("bars_to_reversal_high", "bars_to_interim_low", "bars_to_reversal") else np.nan,
                    "n": len(group),
                })
    return pd.DataFrame(results)


def run_factor_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """因子间相关性"""
    available = [f for f in NUMERIC_FACTORS if f in df.columns]
    if not available:
        return pd.DataFrame()
    sub = df[available].dropna()
    if len(sub) < 20:
        return pd.DataFrame()
    return sub.corr(method="spearman")


def run_time_distance_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """时间距离与收益关系分析"""
    rows = []

    pairs = [
        ("bars_to_reversal_high", "next_reversal_high_ret"),
        ("bars_to_interim_low", "interim_low_ret"),
        ("bars_to_reversal", "next_reversal_high_ret"),
    ]
    for dist_col, ret_col in pairs:
        if dist_col not in df.columns or ret_col not in df.columns:
            continue
        sub = df[[dist_col, ret_col]].dropna()
        if len(sub) < 20:
            continue
        spearman_r, spearman_p = stats.spearmanr(sub[dist_col], sub[ret_col])
        pearson_r, pearson_p = stats.pearsonr(sub[dist_col], sub[ret_col])
        rows.append({
            "distance_col": dist_col,
            "return_col": ret_col,
            "n": len(sub),
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
        })

    if "bars_to_reversal_high" in df.columns and "next_reversal_high_ret" in df.columns:
        sub = df[["bars_to_reversal_high", "next_reversal_high_ret"]].dropna()
        if len(sub) >= 30:
            for threshold in [5, 10, 20, 50]:
                fast = sub[sub["bars_to_reversal_high"] <= threshold]
                slow = sub[sub["bars_to_reversal_high"] > threshold]
                if len(fast) < 5 or len(slow) < 5:
                    continue
                rows.append({
                    "distance_col": f"bars<={threshold}",
                    "return_col": "next_reversal_high_ret",
                    "n": len(fast),
                    "spearman_r": np.nan,
                    "spearman_p": np.nan,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "mean_ret": fast["next_reversal_high_ret"].mean(),
                    "median_ret": fast["next_reversal_high_ret"].median(),
                    "win_rate": (fast["next_reversal_high_ret"] > 0).mean(),
                })
                rows.append({
                    "distance_col": f"bars>{threshold}",
                    "return_col": "next_reversal_high_ret",
                    "n": len(slow),
                    "spearman_r": np.nan,
                    "spearman_p": np.nan,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "mean_ret": slow["next_reversal_high_ret"].mean(),
                    "median_ret": slow["next_reversal_high_ret"].median(),
                    "win_rate": (slow["next_reversal_high_ret"] > 0).mean(),
                })

    return pd.DataFrame(rows)


def print_summary_report(
    target_dist: pd.DataFrame,
    yearly_dist: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    categorical_result: pd.DataFrame,
    factor_corr: pd.DataFrame,
    time_dist: pd.DataFrame,
    df: pd.DataFrame,
):
    print("=" * 80)
    print("DSA V型反转因子探索实验报告")
    print("=" * 80)

    print(f"\n样本概览:")
    print(f"  总记录数: {len(df)}")
    print(f"  唯一股票数: {df['ts_code'].nunique()}")
    print(f"  日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")
    print(f"  唯一选股日数: {df['selection_date'].nunique()}")
    has_rev = df["bars_to_reversal"].notna().sum() if "bars_to_reversal" in df.columns else 0
    print(f"  有反转记录: {has_rev}, 无反转记录: {len(df) - has_rev}")

    if not target_dist.empty:
        print("\n" + "=" * 80)
        print("目标变量分布统计")
        print("=" * 80)
        for _, row in target_dist.iterrows():
            label = row["label"]
            print(f"\n  {label} ({row['target']}): n={row['n']}")
            print(f"    均值={row['mean']:.4f}, 中位={row['median']:.4f}, 标准差={row['std']:.4f}")
            print(f"    min={row['min']:.4f}, P25={row['p25']:.4f}, P75={row['p75']:.4f}, max={row['max']:.4f}")
            print(f"    偏度={row['skew']:.2f}, 峰度={row['kurt']:.2f}")
            if "positive_ratio" in row and not np.isnan(row.get("positive_ratio", np.nan)):
                print(f"    正值比例={row['positive_ratio']:.2%}")

    if not yearly_dist.empty:
        print("\n" + "=" * 80)
        print("按年份目标变量分布")
        print("=" * 80)
        for target in TARGET_COLUMNS:
            sub = yearly_dist[yearly_dist["target"] == target]
            if sub.empty:
                continue
            label = TARGET_LABELS.get(target, target)
            print(f"\n  {label}:")
            print(f"    {'年份':>6} {'N':>6} {'均值':>10} {'中位':>10} {'标准差':>10}")
            for _, row in sub.iterrows():
                print(f"    {int(row['year']):>6} {row['n']:>6} {row['mean']:>10.4f} {row['median']:>10.4f} {row['std']:>10.4f}")

    if not ic_summary.empty:
        print("\n" + "=" * 80)
        print("IC 分析汇总")
        print("=" * 80)
        for target in TARGET_COLUMNS + DERIVED_TARGETS:
            sub = ic_summary[ic_summary["target"] == target].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("ic_abs_mean", ascending=False)
            label = TARGET_LABELS.get(target, target)
            print(f"\n--- {label} ({target}) ---")
            print(
                f"  {'因子':<20} {'IC均值':>8} {'IC中位':>8} {'ICIR':>8} "
                f"{'t值':>8} {'IC>0比':>8} {'|IC|均值':>8} {'截面数':>6}"
            )
            print("  " + "-" * 80)
            for _, row in sub.head(10).iterrows():
                fl = row.get("factor_label", row["factor"])
                print(
                    f"  {fl:<20} {row['ic_mean']:>8.4f} {row['ic_median']:>8.4f} "
                    f"{row['icir']:>8.4f} {row['t_stat']:>8.2f} "
                    f"{row['ic_positive_ratio']:>8.2f} {row['ic_abs_mean']:>8.4f} "
                    f"{row['n_periods']:>6d}"
                )
            effective = sub[(sub["ic_abs_mean"] > 0.03) & (sub["icir"].abs() > 0.5)]
            significant = sub[sub["t_stat"].abs() > 2.0]
            print(f"\n  弱有效因子(|IC|>0.03 & |ICIR|>0.5): {len(effective)} 个")
            print(f"  统计显著因子(|t|>2.0): {len(significant)} 个")

    if not group_summary.empty:
        print("\n" + "=" * 80)
        print("分组收益汇总（Q1~Q5 均值）")
        print("=" * 80)
        for target in TARGET_COLUMNS:
            sub = group_summary[group_summary["target"] == target]
            if sub.empty:
                continue
            label = TARGET_LABELS.get(target, target)
            print(f"\n--- {label} ---")
            for factor in sub["factor_label"].unique():
                fsub = sub[sub["factor_label"] == factor]
                q_row = fsub.sort_values("group_label")
                q_str = "  ".join(
                    [f"{row['group_label']}:{row['mean_val']:+.4f}" for _, row in q_row.iterrows()]
                )
                ls_val = fsub["long_short"].iloc[0] if fsub["long_short"].notna().any() else np.nan
                ls_str = f"  L-S:{ls_val:+.4f}" if not np.isnan(ls_val) else ""
                print(f"  {factor:<20} {q_str}{ls_str}")

    if not categorical_result.empty:
        print("\n" + "=" * 80)
        print("分类型因子分析")
        print("=" * 80)
        for cat_factor in CATEGORICAL_FACTORS:
            for target in TARGET_COLUMNS:
                sub = categorical_result[
                    (categorical_result["factor"] == cat_factor) & (categorical_result["target"] == target)
                ]
                if sub.empty:
                    continue
                fl = FACTOR_LABELS.get(cat_factor, cat_factor)
                tl = TARGET_LABELS.get(target, target)
                print(f"\n  {fl} → {tl}:")
                for _, row in sub.iterrows():
                    wr_str = f" 胜率:{row['win_rate']:.1%}" if not np.isnan(row.get("win_rate", np.nan)) else ""
                    print(f"    类别={row['category']:<6} 均值:{row['mean_val']:+.4f} 中位:{row['median_val']:+.4f} n={row['n']}{wr_str}")

    if not time_dist.empty:
        print("\n" + "=" * 80)
        print("时间距离与收益关系")
        print("=" * 80)
        corr_rows = time_dist[time_dist["spearman_r"].notna()]
        if not corr_rows.empty:
            print("\n  相关性:")
            for _, row in corr_rows.iterrows():
                sig = "***" if row["spearman_p"] < 0.001 else ("**" if row["spearman_p"] < 0.01 else ("*" if row["spearman_p"] < 0.05 else ""))
                print(f"    {row['distance_col']} ↔ {row['return_col']}: ρ={row['spearman_r']:.4f} (p={row['spearman_p']:.4f}){sig}")
        thresh_rows = time_dist[time_dist["mean_ret"].notna()]
        if not thresh_rows.empty:
            print("\n  时间阈值分组:")
            for _, row in thresh_rows.iterrows():
                wr_str = f" 胜率:{row['win_rate']:.1%}" if not np.isnan(row.get("win_rate", np.nan)) else ""
                print(f"    {row['distance_col']:<15} → {row['return_col']}: 均值={row['mean_ret']:+.4f} 中位={row['median_ret']:+.4f} n={row['n']}{wr_str}")

    if not factor_corr.empty:
        print("\n" + "=" * 80)
        print("因子间高相关对（|ρ|>0.6）")
        print("=" * 80)
        high_corr_pairs = []
        for i in range(len(factor_corr)):
            for j in range(i + 1, len(factor_corr)):
                val = factor_corr.iloc[i, j]
                if abs(val) > 0.6:
                    high_corr_pairs.append((factor_corr.index[i], factor_corr.columns[j], val))
        if high_corr_pairs:
            for f1, f2, val in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
                l1 = FACTOR_LABELS.get(f1, f1)
                l2 = FACTOR_LABELS.get(f2, f2)
                print(f"  {l1} ↔ {l2}: ρ = {val:.3f}")
        else:
            print("  无高相关因子对（|ρ|均<=0.6）")


def save_results(
    target_dist: pd.DataFrame,
    yearly_dist: pd.DataFrame,
    ic_summary: pd.DataFrame,
    ic_ts: pd.DataFrame,
    group_summary: pd.DataFrame,
    categorical_result: pd.DataFrame,
    factor_corr: pd.DataFrame,
    time_dist: pd.DataFrame,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    files = {
        "dsa_target_distribution.csv": target_dist,
        "dsa_yearly_distribution.csv": yearly_dist,
        "dsa_ic_summary.csv": ic_summary,
        "dsa_ic_timeseries.csv": ic_ts,
        "dsa_group_summary.csv": group_summary,
        "dsa_categorical_result.csv": categorical_result,
        "dsa_factor_correlation.csv": factor_corr,
        "dsa_time_distance.csv": time_dist,
    }
    for fname, df in files.items():
        if df.empty:
            continue
        save_cols = [c for c in df.columns if c != "ic_detail"]
        df[save_cols].to_csv(
            os.path.join(output_dir, fname), index=False, encoding="utf-8-sig"
        )
        print(f"  保存: {os.path.join(output_dir, fname)}")


def main():
    parser = argparse.ArgumentParser(
        description="DSA V型反转因子探索实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analysis/dsa_vreversal_explorer.py
  python analysis/dsa_vreversal_explorer.py --start 2023-01-01 --end 2025-12-31
  python analysis/dsa_vreversal_explorer.py --sample-limit 500
  python analysis/dsa_vreversal_explorer.py --only-reversal
        """,
    )
    parser.add_argument("--start", type=str, default=None, help="起始日期 (YYYY-MM-DD)，默认 2021-05-01")
    parser.add_argument("--end", type=str, default=None, help="结束日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--sample-limit", type=int, default=None, help="限制样本量（快速测试）")
    parser.add_argument("--output-dir", type=str, default="/tmp/dsa_vreversal_analysis", help="输出目录")
    parser.add_argument("--only-reversal", action="store_true", help="只分析有 dsa_dir 反转的记录")

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else date(2021, 5, 1)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()

    print("=" * 80)
    print("DSA V型反转因子探索实验")
    print("=" * 80)
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  只看有反转: {'是' if args.only_reversal else '否'}")
    if args.sample_limit:
        print(f"  样本限制: {args.sample_limit}")
    print()

    print("[1/7] 获取触发点记录...")
    df = fetch_vreversal_records(start_date, end_date, args.sample_limit, args.only_reversal)
    if df.empty:
        print("记录为空，退出")
        return
    print(f"  获取 {len(df)} 条记录，{df['ts_code'].nunique()} 只股票")

    print("[2/7] 构建衍生目标变量...")
    df = build_derived_targets(df)
    all_target_cols = [c for c in TARGET_COLUMNS + DERIVED_TARGETS if c in df.columns]
    print(f"  目标变量: {len(all_target_cols)} 个")

    print("[3/7] 目标变量分布统计...")
    target_dist = run_target_distribution(df)
    yearly_dist = run_yearly_distribution(df)

    print("[4/7] IC 分析...")
    ic_summary, ic_ts = run_ic_analysis(df, all_target_cols)
    print(f"  IC 汇总: {len(ic_summary)} 条")

    print("[5/7] 分组收益分析...")
    group_df = run_group_analysis(df, all_target_cols)
    group_summary = run_group_summary(group_df, all_target_cols)
    print(f"  分组汇总: {len(group_summary)} 条")

    print("[6/7] 分类型因子 + 因子相关性 + 时间距离分析...")
    categorical_result = run_categorical_analysis(df, all_target_cols)
    factor_corr = run_factor_correlation(df)
    time_dist = run_time_distance_analysis(df)

    print("[7/7] 输出报告...")
    print()
    print_summary_report(
        target_dist, yearly_dist, ic_summary, group_summary,
        categorical_result, factor_corr, time_dist, df
    )

    print("\n" + "-" * 80)
    print("保存结果...")
    save_results(
        target_dist, yearly_dist, ic_summary, ic_ts, group_summary,
        categorical_result, factor_corr, time_dist, args.output_dir
    )
    print("-" * 80)
    print("\n分析完成")


if __name__ == "__main__":
    main()
