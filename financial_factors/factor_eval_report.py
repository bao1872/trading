#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    财务因子预测力评估：IC分析 + 分组回测 + 因子衰减 + 相关性 + 汇总报告。
    从 factor_return_dataset 表读取数据，输出评估结果到控制台和CSV。

Inputs:
    - 数据库表: factor_return_dataset

Outputs:
    - CSV: out/factor_ic_summary.csv（IC汇总）
    - CSV: out/factor_group_result.csv（分组回测结果）
    - CSV: out/factor_correlation.csv（因子相关性矩阵）
    - 控制台: 汇总报告

How to Run:
    python financial_factors/factor_eval_report.py
    python financial_factors/factor_eval_report.py --horizon 20d
    python financial_factors/factor_eval_report.py --out-dir ./out

Examples:
    python financial_factors/factor_eval_report.py
    python financial_factors/factor_eval_report.py --horizon 5d --out-dir ./results

Side Effects:
    - 写入CSV文件（不写数据库）
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

try:
    from config import DATABASE_URL
except Exception:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FACTOR_COLS = [
    "q_rev_yoy", "q_op_yoy", "q_np_parent_yoy", "ytd_rev_yoy", "ytd_np_parent_yoy",
    "q_ebit_yoy", "q_rev_qoq", "q_op_qoq",
    "q_gross_margin", "q_gm_yoy_change", "q_op_margin", "q_np_parent_margin",
    "q_gm_qoq_change", "op_margin_change", "q_ebit_margin",
    "q_cfo_to_np_parent", "ttm_cfo_to_np_parent", "q_accruals_to_assets",
    "ttm_cfo_to_ebit", "q_np_parent_to_np",
    "q_cfo_to_rev", "q_cfo_yoy", "ytd_cfo_yoy", "ttm_fcf_to_np_parent",
    "capex_to_cfo", "cash_sales_ratio", "cash_sales_yoy",
    "roa_parent", "cfo_to_assets", "asset_turnover", "ccc", "contract_liab_to_rev",
    "q_rev_yoy_delta", "q_np_parent_yoy_delta", "trend_consistency",
    "profit_cash_sync", "margin_profit_sync", "cfo_to_np_change",
]

DIMENSION_SCORE_COLS = [
    "规模与增长_score", "盈利能力_score", "利润质量_score",
    "现金创造能力_score", "资产效率与资金占用_score", "边际变化与持续性_score",
]

ALL_FACTOR_COLS = FACTOR_COLS + DIMENSION_SCORE_COLS + ["total_score"]

FACTOR_LABELS = {
    "q_rev_yoy": "单季营收同比", "q_op_yoy": "单季营业利润同比",
    "q_np_parent_yoy": "单季归母净利同比", "ytd_rev_yoy": "累计营收同比",
    "ytd_np_parent_yoy": "累计归母净利同比", "q_ebit_yoy": "单季EBIT同比",
    "q_rev_qoq": "单季营收环比", "q_op_qoq": "单季营业利润环比",
    "q_gross_margin": "单季毛利率", "q_gm_yoy_change": "毛利率同比变化",
    "q_op_margin": "单季营业利润率", "q_np_parent_margin": "单季归母净利率",
    "q_gm_qoq_change": "毛利率环比变化", "op_margin_change": "营业利润率同比变化",
    "q_ebit_margin": "EBIT利润率",
    "q_cfo_to_np_parent": "经营现金流/归母净利", "ttm_cfo_to_np_parent": "TTM经营现金流/归母净利",
    "q_accruals_to_assets": "应计项/总资产", "ttm_cfo_to_ebit": "TTM经营现金流/EBIT",
    "q_np_parent_to_np": "归母净利/净利润",
    "q_cfo_to_rev": "经营现金流/收入", "q_cfo_yoy": "经营现金流同比",
    "ytd_cfo_yoy": "累计经营现金流同比", "ttm_fcf_to_np_parent": "TTM自由现金流/归母净利",
    "capex_to_cfo": "资本开支/经营现金流", "cash_sales_ratio": "销售收现比",
    "cash_sales_yoy": "销售收现同比",
    "roa_parent": "归母ROA", "cfo_to_assets": "经营现金流/总资产",
    "asset_turnover": "总资产周转率", "ccc": "现金转换周期",
    "contract_liab_to_rev": "合同负债/收入",
    "q_rev_yoy_delta": "营收同比变化", "q_np_parent_yoy_delta": "归母净利同比变化",
    "trend_consistency": "趋势连续性", "profit_cash_sync": "利润现金流同步",
    "margin_profit_sync": "毛利率利润率同步", "cfo_to_np_change": "经营现金流/归母净利变化",
    "规模与增长_score": "规模与增长", "盈利能力_score": "盈利能力",
    "利润质量_score": "利润质量", "现金创造能力_score": "现金创造能力",
    "资产效率与资金占用_score": "资产效率", "边际变化与持续性_score": "边际变化",
    "total_score": "综合评分",
}


def get_db_engine():
    return create_engine(DATABASE_URL, pool_size=5, max_overflow=10, pool_recycle=3600)


def load_dataset(engine) -> pd.DataFrame:
    sql = "SELECT * FROM factor_return_dataset"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    logger.info(f"加载数据集: {len(df)} 条, {df['ts_code'].nunique()} 只股票")
    return df


def compute_ic(df: pd.DataFrame, factor_col: str, return_col: str) -> pd.DataFrame:
    ic_records = []
    for report_date, group in df.groupby("report_date"):
        valid = group[[factor_col, return_col]].dropna()
        if len(valid) < 20:
            continue
        ic_val, _ = stats.spearmanr(valid[factor_col], valid[return_col])
        ic_records.append({"report_date": report_date, "IC": ic_val, "n": len(valid)})
    return pd.DataFrame(ic_records)


def run_ic_analysis(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    results = []
    for factor in ALL_FACTOR_COLS:
        ic_df = compute_ic(df, factor, return_col)
        if ic_df.empty:
            continue
        ic_mean = ic_df["IC"].mean()
        ic_std = ic_df["IC"].std()
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ic_pos_ratio = (ic_df["IC"] > 0).mean()
        results.append({
            "factor": factor,
            "label": FACTOR_LABELS.get(factor, factor),
            "IC_mean": round(ic_mean, 4),
            "IC_std": round(ic_std, 4),
            "ICIR": round(icir, 4),
            "IC_positive_ratio": round(ic_pos_ratio, 4),
            "n_periods": len(ic_df),
            "avg_n": round(ic_df["n"].mean(), 0),
        })
    result_df = pd.DataFrame(results).sort_values("ICIR", key=abs, ascending=False)
    return result_df


def run_group_analysis(df: pd.DataFrame, factor_col: str, return_col: str,
                       n_groups: int = 5) -> pd.DataFrame:
    results = []
    for report_date, group in df.groupby("report_date"):
        valid = group[[factor_col, return_col, "ts_code"]].dropna()
        if len(valid) < n_groups * 10:
            continue
        valid = valid.copy()
        valid["group"] = pd.qcut(valid[factor_col], n_groups, labels=False, duplicates="drop")
        for g, g_df in valid.groupby("group"):
            results.append({
                "report_date": report_date,
                "group": int(g) + 1,
                "mean_ret": g_df[return_col].mean(),
                "median_ret": g_df[return_col].median(),
                "win_rate": (g_df[return_col] > 0).mean(),
                "n": len(g_df),
            })
    return pd.DataFrame(results)


def run_all_group_analysis(df: pd.DataFrame, return_col: str,
                           n_groups: int = 5) -> pd.DataFrame:
    all_results = []
    for factor in ALL_FACTOR_COLS:
        group_df = run_group_analysis(df, factor, return_col, n_groups)
        if group_df.empty:
            continue
        summary = group_df.groupby("group").agg(
            mean_ret=("mean_ret", "mean"),
            median_ret=("median_ret", "mean"),
            win_rate=("win_rate", "mean"),
            avg_n=("n", "mean"),
        ).reset_index()

        g5_ret = summary.loc[summary["group"] == n_groups, "mean_ret"].values
        g1_ret = summary.loc[summary["group"] == 1, "mean_ret"].values
        long_short = float(g5_ret[0] - g1_ret[0]) if len(g5_ret) > 0 and len(g1_ret) > 0 else np.nan

        is_monotone = True
        rets = summary["mean_ret"].values
        for i in range(len(rets) - 1):
            if rets[i] >= rets[i + 1]:
                is_monotone = False
                break

        for _, row in summary.iterrows():
            all_results.append({
                "factor": factor,
                "label": FACTOR_LABELS.get(factor, factor),
                "group": int(row["group"]),
                "mean_ret": round(row["mean_ret"], 6),
                "win_rate": round(row["win_rate"], 4),
                "long_short": round(long_short, 6) if not np.isnan(long_short) else None,
                "monotone": is_monotone,
            })
    return pd.DataFrame(all_results)


def compute_factor_correlation(df: pd.DataFrame) -> pd.DataFrame:
    valid_cols = [c for c in ALL_FACTOR_COLS if c in df.columns and df[c].notna().sum() > 100]
    corr_df = df[valid_cols].corr(method="spearman")
    return corr_df


def print_summary(ic_df: pd.DataFrame, group_df: pd.DataFrame, corr_df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("财务因子预测力评估汇总报告")
    print("=" * 80)

    print("\n【有效因子】(ICIR > 0.3 且 |IC_mean| > 0.02)")
    effective = ic_df[(ic_df["ICIR"].abs() > 0.3) & (ic_df["IC_mean"].abs() > 0.02)]
    if effective.empty:
        print("  无有效因子")
    else:
        for _, row in effective.iterrows():
            direction = "正向" if row["IC_mean"] > 0 else "反向"
            print(f"  {row['label']:20s} IC_mean={row['IC_mean']:+.4f} ICIR={row['ICIR']:+.4f} "
                  f"IC正比={row['IC_positive_ratio']:.1%} ({direction})")

    print("\n【弱有效因子】(0.15 < |ICIR| <= 0.3 且 |IC_mean| > 0.015)")
    weak = ic_df[(ic_df["ICIR"].abs() > 0.15) & (ic_df["ICIR"].abs() <= 0.3) &
                 (ic_df["IC_mean"].abs() > 0.015)]
    if weak.empty:
        print("  无弱有效因子")
    else:
        for _, row in weak.iterrows():
            direction = "正向" if row["IC_mean"] > 0 else "反向"
            print(f"  {row['label']:20s} IC_mean={row['IC_mean']:+.4f} ICIR={row['ICIR']:+.4f} "
                  f"IC正比={row['IC_positive_ratio']:.1%} ({direction})")

    print("\n【无效因子】(|ICIR| <= 0.15)")
    ineffective = ic_df[ic_df["ICIR"].abs() <= 0.15]
    if not ineffective.empty:
        for _, row in ineffective.iterrows():
            print(f"  {row['label']:20s} IC_mean={row['IC_mean']:+.4f} ICIR={row['ICIR']:+.4f}")

    print("\n【分组单调性】(5组收益单调递增/递减)")
    factor_monotone = group_df.groupby("factor")["monotone"].first()
    mono_factors = factor_monotone[factor_monotone == True].index.tolist()
    if mono_factors:
        for f in mono_factors:
            label = FACTOR_LABELS.get(f, f)
            ls = group_df[group_df["factor"] == f]["long_short"].iloc[0]
            print(f"  {label:20s} 多空差={ls:+.4f}")
    else:
        print("  无单调因子")

    print("\n【高共线因子组】(|Spearman corr| > 0.7)")
    high_corr_pairs = []
    cols = corr_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_df.iloc[i, j]
            if abs(val) > 0.7:
                high_corr_pairs.append((cols[i], cols[j], val))
    if high_corr_pairs:
        for f1, f2, val in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
            l1, l2 = FACTOR_LABELS.get(f1, f1), FACTOR_LABELS.get(f2, f2)
            print(f"  {l1:20s} <-> {l2:20s} corr={val:+.3f}")
    else:
        print("  无高共线因子")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="财务因子预测力评估")
    parser.add_argument("--horizon", type=str, default="20d",
                        choices=["5d", "20d", "60d"], help="收益持有期")
    parser.add_argument("--out-dir", type=str, default="out", help="输出目录")
    args = parser.parse_args()

    engine = get_db_engine()
    df = load_dataset(engine)

    return_col = f"ret_{args.horizon}"
    if return_col not in df.columns:
        logger.error(f"收益列 {return_col} 不存在")
        return 1

    valid = df[df[return_col].notna()]
    logger.info(f"有效数据: {len(valid)} 条 (收益列 {return_col})")

    logger.info("计算IC分析...")
    ic_df = run_ic_analysis(valid, return_col)

    logger.info("计算分组回测...")
    group_df = run_all_group_analysis(valid, return_col)

    logger.info("计算因子相关性...")
    corr_df = compute_factor_correlation(valid)

    os.makedirs(args.out_dir, exist_ok=True)

    ic_path = os.path.join(args.out_dir, "factor_ic_summary.csv")
    ic_df.to_csv(ic_path, index=False)
    logger.info(f"IC汇总已保存: {ic_path}")

    group_path = os.path.join(args.out_dir, "factor_group_result.csv")
    group_df.to_csv(group_path, index=False)
    logger.info(f"分组回测已保存: {group_path}")

    corr_path = os.path.join(args.out_dir, "factor_correlation.csv")
    corr_df.to_csv(corr_path)
    logger.info(f"相关性矩阵已保存: {corr_path}")

    print_summary(ic_df, group_df, corr_df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
