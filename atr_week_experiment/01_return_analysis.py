#!/usr/bin/env python3
"""
ATR Rope 周线突破选股 — 持股收益与成交量相关性分析

Purpose:
    分析 atr_week_selection 表中信号的持股3/5/10/20天收益，
    并探索 vol_zscore 与收益的相关性。

Inputs:
    atr_week_selection 表 + stock_k_data 表

Outputs:
    atr_week_experiment/output/ 下 CSV 文件

How to Run:
    python atr_week_experiment/01_return_analysis.py
    python atr_week_experiment/01_return_analysis.py --start 2025-01-01 --end 2026-05-26
    python atr_week_experiment/01_return_analysis.py --start 2026-04-01 --end 2026-04-30

Examples:
    python atr_week_experiment/01_return_analysis.py
    python atr_week_experiment/01_return_analysis.py --start 2026-04-01 --end 2026-04-30

Side Effects: 只读数据库，写入 output/ 目录下的 CSV 文件

================================================================================
【分析维度】

A. 整体收益统计：各持股期的均值/中位数/胜率/MFE/MAE
B. 按 vol_zscore 分组：低(<0)/中(0~1)/高(>=1)，看收益差异
C. Spearman 相关性：vol_zscore 与各持股期收益的 rank correlation
D. 分位数桶分析：5分位桶，看收益随 zscore 递增/递减趋势
E. IC 时间序列稳定性：按月计算 IC，看 ICIR 和 t-stat

【核心计算】
- 未来收益：引用 atr_experiment/atr_gbdt_utils.py 的 enrich_with_future_metrics()
  入场价 open[T+1]，无前视偏差
================================================================================
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

# 引用 SSOT 的未来收益计算（open[T+1]入场，无前视偏差）
from atr_experiment.atr_gbdt_utils import enrich_with_future_metrics

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [3, 5, 10, 20]

VOL_ZSCORE_BINS = [
    ("低(<0)", lambda x: x < 0),
    ("中(0~1)", lambda x: (x >= 0) & (x < 1)),
    ("高(>=1)", lambda x: x >= 1),
]


def load_selection_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """从 atr_week_selection 表加载选股记录"""
    conditions = []
    params = {}
    if start_date:
        conditions.append("selection_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append("selection_date <= :end_date")
        params["end_date"] = end_date

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    sql = text(f"""
        SELECT selection_date, signal_date, ts_code, stock_name,
               rope_dir, rope_value, c_hi, c_lo, atr_value,
               rope_dev_pct, rope_dev_atr, range_width_pct, range_pos_01,
               dsa_dir, dsa_vwap, dsa_vwap_dev_pct,
               change_pct, vol_zscore, avg_amount_5w
        FROM atr_week_selection
        WHERE {where_clause}
        ORDER BY selection_date, ts_code
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)
    print(f"加载选股记录: {len(df)} 条")
    return df


def compute_return_stats(series: pd.Series) -> dict:
    """计算收益统计指标"""
    valid = series.dropna()
    if len(valid) == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "win_rate": np.nan, "std": np.nan}
    return {
        "count": len(valid),
        "mean": valid.mean(),
        "median": valid.median(),
        "win_rate": (valid > 0).mean(),
        "std": valid.std(),
    }


def analysis_overall(df: pd.DataFrame):
    """A. 整体收益统计"""
    print("\n" + "=" * 80)
    print("A. 整体收益统计")
    print("=" * 80)

    rows = []
    for N in HORIZONS:
        ret_col = f"return_{N}"
        mfe_col = f"mfe_{N}"
        mae_col = f"mae_{N}"

        ret_stats = compute_return_stats(df[ret_col])
        mfe_stats = compute_return_stats(df[mfe_col])
        mae_stats = compute_return_stats(df[mae_col])

        rows.append({
            "持股天数": N,
            "信号数": ret_stats["count"],
            "收益均值": f"{ret_stats['mean']*100:.2f}%",
            "收益中位数": f"{ret_stats['median']*100:.2f}%",
            "胜率": f"{ret_stats['win_rate']*100:.1f}%",
            "收益标准差": f"{ret_stats['std']*100:.2f}%",
            "MFE均值": f"{mfe_stats['mean']*100:.2f}%",
            "MAE均值": f"{mae_stats['mean']*100:.2f}%",
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    return result_df


def analysis_by_vol_zscore(df: pd.DataFrame):
    """B. 按 vol_zscore 分组看收益"""
    print("\n" + "=" * 80)
    print("B. 按 vol_zscore 分组收益统计")
    print("=" * 80)

    df = df.copy()
    df["vol_group"] = "未知"
    for label, cond_fn in VOL_ZSCORE_BINS:
        mask = df["vol_zscore"].notna() & cond_fn(df["vol_zscore"])
        df.loc[mask, "vol_group"] = label

    rows = []
    for N in HORIZONS:
        ret_col = f"return_{N}"
        for group_name in ["低(<0)", "中(0~1)", "高(>=1)"]:
            sub = df[df["vol_group"] == group_name]
            s = compute_return_stats(sub[ret_col])
            rows.append({
                "持股天数": N,
                "vol_zscore组": group_name,
                "信号数": s["count"],
                "收益均值": f"{s['mean']*100:.2f}%",
                "收益中位数": f"{s['median']*100:.2f}%",
                "胜率": f"{s['win_rate']*100:.1f}%",
            })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    return result_df


def analysis_spearman(df: pd.DataFrame):
    """C. vol_zscore 与收益的 Spearman 相关性"""
    print("\n" + "=" * 80)
    print("C. vol_zscore 与收益的 Spearman 相关性")
    print("=" * 80)

    rows = []
    for N in HORIZONS:
        ret_col = f"return_{N}"
        valid = df[["vol_zscore", ret_col]].dropna()
        if len(valid) < 10:
            rows.append({"持股天数": N, "Spearman_rho": "N/A", "p_value": "N/A", "显著": "N/A"})
            continue

        rho, pval = stats.spearmanr(valid["vol_zscore"], valid[ret_col])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        rows.append({
            "持股天数": N,
            "Spearman_rho": f"{rho:.4f}",
            "p_value": f"{pval:.4f}",
            "显著": sig,
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    return result_df


def analysis_quantile_buckets(df: pd.DataFrame, n_buckets: int = 5):
    """D. vol_zscore 分位数桶分析"""
    print("\n" + "=" * 80)
    print(f"D. vol_zscore {n_buckets}分位桶分析")
    print("=" * 80)

    df = df.copy()
    valid = df["vol_zscore"].dropna()
    if len(valid) < n_buckets * 10:
        print("有效数据不足，跳过")
        return pd.DataFrame()

    df["zscore_bucket"] = pd.qcut(df["vol_zscore"], n_buckets, labels=False, duplicates="drop")

    rows = []
    for N in HORIZONS:
        ret_col = f"return_{N}"
        for b in sorted(df["zscore_bucket"].dropna().unique()):
            sub = df[df["zscore_bucket"] == b]
            s = compute_return_stats(sub[ret_col])
            # 取该桶的zscore范围
            zscore_min = sub["vol_zscore"].min()
            zscore_max = sub["vol_zscore"].max()
            rows.append({
                "持股天数": N,
                "桶": f"Q{int(b)+1}",
                "zscore范围": f"[{zscore_min:.2f}, {zscore_max:.2f}]",
                "信号数": s["count"],
                "收益均值": f"{s['mean']*100:.2f}%",
                "收益中位数": f"{s['median']*100:.2f}%",
                "胜率": f"{s['win_rate']*100:.1f}%",
            })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    return result_df


def analysis_ic_stability(df: pd.DataFrame):
    """E. IC 时间序列稳定性（按月计算 Spearman IC）"""
    print("\n" + "=" * 80)
    print("E. IC 时间序列稳定性（按月）")
    print("=" * 80)

    df = df.copy()
    df["month"] = pd.to_datetime(df["selection_date"]).dt.to_period("M")

    rows = []
    for N in HORIZONS:
        ret_col = f"return_{N}"
        monthly_ics = []
        for month, group in df.groupby("month"):
            valid = group[["vol_zscore", ret_col]].dropna()
            if len(valid) < 5:
                continue
            rho, _ = stats.spearmanr(valid["vol_zscore"], valid[ret_col])
            monthly_ics.append({"month": str(month), "IC": rho})

        if len(monthly_ics) < 3:
            rows.append({"持股天数": N, "IC均值": "N/A", "IC标准差": "N/A", "ICIR": "N/A", "t_stat": "N/A", "月数": len(monthly_ics)})
            continue

        ic_series = pd.Series([m["IC"] for m in monthly_ics])
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_series))) if ic_std > 0 else 0

        rows.append({
            "持股天数": N,
            "IC均值": f"{ic_mean:.4f}",
            "IC标准差": f"{ic_std:.4f}",
            "ICIR": f"{icir:.4f}",
            "t_stat": f"{t_stat:.2f}",
            "月数": len(monthly_ics),
        })

        # 打印月度IC明细
        print(f"\n  持股{N}天 - 月度IC:")
        for m in monthly_ics:
            bar = "█" * int(abs(m["IC"]) * 50) if not np.isnan(m["IC"]) else ""
            direction = "+" if m["IC"] > 0 else "-"
            print(f"    {m['month']}: IC={m['IC']:+.4f} {direction}{bar}")

    result_df = pd.DataFrame(rows)
    print(f"\n  IC汇总:")
    print(result_df.to_string(index=False))
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="ATR Rope 周线突破选股 — 持股收益与成交量相关性分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python atr_week_experiment/01_return_analysis.py
  python atr_week_experiment/01_return_analysis.py --start 2025-01-01 --end 2026-05-26
  python atr_week_experiment/01_return_analysis.py --start 2026-04-01 --end 2026-04-30
        """
    )
    parser.add_argument("--start", help="起始日期 (YYYY-MM-DD)，默认2023-05-26")
    parser.add_argument("--end", help="结束日期 (YYYY-MM-DD)，默认2026-05-26")
    args = parser.parse_args()

    start_date = args.start or "2023-05-26"
    end_date = args.end or "2026-05-26"

    print("=" * 80)
    print("ATR Rope 周线突破选股 — 持股收益与成交量相关性分析")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  持股期: {HORIZONS} 天")
    print("=" * 80)

    # 1. 加载选股数据
    df = load_selection_data(start_date, end_date)
    if df.empty:
        print("无选股记录，退出")
        return

    # 2. 计算未来收益（引用SSOT，open[T+1]入场）
    print("\n计算未来收益...")
    df = enrich_with_future_metrics(df, horizons=HORIZONS)

    # 保存含收益的完整数据集
    output_csv = OUTPUT_DIR / "return_analysis.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n完整数据集已保存: {output_csv}")

    # 3. 各维度分析
    stats_a = analysis_overall(df)
    stats_b = analysis_by_vol_zscore(df)
    stats_c = analysis_spearman(df)
    stats_d = analysis_quantile_buckets(df)
    stats_e = analysis_ic_stability(df)

    # 4. 保存统计汇总
    summary_rows = []
    for name, sdf in [("整体统计", stats_a), ("vol分组", stats_b), ("Spearman", stats_c), ("分位桶", stats_d), ("IC稳定性", stats_e)]:
        if not sdf.empty:
            sdf = sdf.copy()
            sdf.insert(0, "分析维度", name)
            summary_rows.append(sdf)

    if summary_rows:
        summary_df = pd.concat(summary_rows, ignore_index=True)
        summary_csv = OUTPUT_DIR / "stats_summary.csv"
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"\n统计汇总已保存: {summary_csv}")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
