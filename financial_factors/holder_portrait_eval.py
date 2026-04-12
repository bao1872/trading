# -*- coding: utf-8 -*-
"""
Purpose:
    股东投资质量画像模型 - 因子评估脚本
    评估画像评分的预测力: IC分析 + 分组回测 + 衰减分析

Inputs:
    - 数据库表: stock_holder_quality_portrait
    - 数据库表: stock_top10_holders_tushare
    - 数据库表: stock_k_data (行情)

Outputs:
    - 控制台: 评估报告
    - CSV: holder_portrait_ic_summary.csv, holder_portrait_group_result.csv

How to Run:
    python financial_factors/holder_portrait_eval.py --horizon 60
    python financial_factors/holder_portrait_eval.py --horizon 20 --min-stocks 3

Examples:
    python financial_factors/holder_portrait_eval.py --horizon 60
    python financial_factors/holder_portrait_eval.py --horizon 20 --min-stocks 5

Side Effects:
    - 写 CSV 文件到 financial_factors/ 目录
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_session, query_df
from financial_factors.holder_quality_portrait import (
    normalize_ts_code,
    normalize_holder_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PORTRAIT_TABLE = "stock_holder_quality_portrait"
HOLDER_TABLE = "stock_top10_holders_tushare"


def load_portrait(min_stocks: int = 1) -> pd.DataFrame:
    with get_session() as session:
        df = query_df(session, PORTRAIT_TABLE)
    if df.empty:
        return df
    if min_stocks > 1:
        df = df[pd.to_numeric(df["sample_stocks"], errors="coerce").fillna(0) >= min_stocks]
    return df


def load_top10_with_industry() -> pd.DataFrame:
    with get_session() as session:
        df = query_df(session, HOLDER_TABLE)
    if df.empty:
        return df
    df = df.copy()
    df["ts_code"] = df["ts_code"].astype(str).map(normalize_ts_code)
    df["holder_name_std"] = df["holder_name"].map(normalize_holder_name)
    df["report_date"] = pd.to_datetime(df["report_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    df["ann_date"] = pd.to_datetime(df["ann_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    for col in ["hold_float_ratio", "hold_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_industry_info() -> pd.DataFrame:
    cols = ["ts_code", "industry_l2"]
    try:
        with get_session() as session:
            df = query_df(session, "stock_pools", columns=cols)
    except Exception:
        return pd.DataFrame(columns=cols)
    if df.empty:
        return df
    df["ts_code"] = df["ts_code"].astype(str).map(normalize_ts_code)
    df = df.drop_duplicates(subset=["ts_code"], keep="last")
    return df


def load_stock_returns(start_date: str, end_date: str) -> pd.DataFrame:
    from sqlalchemy import text
    sql = """
        SELECT ts_code, bar_time as date, close
        FROM stock_k_data
        WHERE freq = 'd' AND bar_time >= :start AND bar_time <= :end
        ORDER BY ts_code, bar_time
    """
    with get_session() as session:
        result = session.execute(text(sql), {"start": start_date, "end": end_date})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    if df.empty:
        return df
    df["ts_code"] = df["ts_code"].astype(str).map(normalize_ts_code)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def build_stock_holder_score(top10_df: pd.DataFrame, portrait_df: pd.DataFrame) -> pd.DataFrame:
    if top10_df.empty or portrait_df.empty:
        return pd.DataFrame()

    quality_map = dict(zip(
        portrait_df["holder_name_std"],
        pd.to_numeric(portrait_df["composite_score"], errors="coerce")
    ))

    score_records = []
    for (ts_code, report_date), group in top10_df.groupby(["ts_code", "report_date"]):
        scores = []
        weights = []
        for _, row in group.iterrows():
            name_std = row["holder_name_std"]
            if name_std in quality_map and pd.notna(quality_map[name_std]):
                ratio = safe_float(row.get("hold_float_ratio"))
                if ratio is None or ratio <= 0:
                    ratio = 1.0
                scores.append(quality_map[name_std])
                weights.append(ratio)

        if scores:
            weights_arr = np.array(weights)
            total_w = weights_arr.sum()
            if total_w > 0:
                weighted_score = float(np.average(scores, weights=weights_arr))
            else:
                weighted_score = float(np.mean(scores))
            score_records.append({
                "ts_code": ts_code,
                "report_date": report_date,
                "holder_quality_score": weighted_score,
                "holder_count": len(scores),
            })

    return pd.DataFrame(score_records)


def safe_float(v: object) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def calc_ic(factor_series: pd.Series, return_series: pd.Series) -> Tuple[float, float, float]:
    valid = pd.DataFrame({"factor": factor_series, "ret": return_series}).dropna()
    if len(valid) < 5:
        return np.nan, np.nan, np.nan
    ic, _ = stats.spearmanr(valid["factor"], valid["ret"])
    ic_mean = ic
    ic_std = 1.0 / np.sqrt(len(valid) - 2) if len(valid) > 2 else np.nan
    icir = ic_mean / ic_std if abs(ic_std) > 1e-12 else np.nan
    return ic_mean, ic_std, icir


def run_ic_analysis(score_df: pd.DataFrame, return_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    logger.info("IC 分析: horizon=%s", horizon)

    return_pivot = return_df.pivot(index="date", columns="ts_code", values="close")
    ret_forward = return_pivot.pct_change(horizon).shift(-horizon)

    if hasattr(ret_forward.index, "tz") and ret_forward.index.tz:
        ret_forward.index = ret_forward.index.tz_localize(None)

    score_df = score_df.copy()
    score_df["report_date"] = pd.to_datetime(score_df["report_date"], errors="coerce")
    if hasattr(score_df["report_date"].dtype, "tz") and score_df["report_date"].dt.tz is not None:
        score_df["report_date"] = score_df["report_date"].dt.tz_localize(None)

    score_by_date = score_df.set_index("report_date")
    report_dates = sorted(score_by_date.index.unique())

    factor_cols = ["holder_quality_score"]
    for dim in ["picking_score", "style_score", "expertise_score", "adapt_score", "risk_score", "scale_score"]:
        if dim in score_df.columns:
            factor_cols.append(dim)

    ic_results = []
    for factor_col in factor_cols:
        if factor_col not in score_df.columns:
            continue
        ics = []
        for date in report_dates:
            date_scores = score_by_date.loc[[date]] if date in score_by_date.index else pd.DataFrame()
            if date_scores.empty:
                continue
            factor_vals = pd.to_numeric(date_scores[factor_col], errors="coerce")
            if "ts_code" in date_scores.columns:
                factor_vals.index = date_scores["ts_code"].values
            ret_date = pd.Timestamp(date)
            if ret_date.tzinfo:
                ret_date = ret_date.tz_localize(None)
            ret_date = ret_date + pd.Timedelta(days=horizon * 2)

            if ret_date not in ret_forward.index:
                nearest_idx = ret_forward.index.searchsorted(ret_date)
                if nearest_idx >= len(ret_forward.index):
                    continue
                ret_date = ret_forward.index[nearest_idx]
            ret_vals = ret_forward.loc[ret_date] if ret_date in ret_forward.index else pd.Series(dtype=float)
            common = factor_vals.index.intersection(ret_vals.dropna().index)
            if len(common) < 5:
                continue
            ic, _, _ = calc_ic(factor_vals.loc[common], ret_vals.loc[common])
            if not np.isnan(ic):
                ics.append(ic)

        if ics:
            ic_mean = np.mean(ics)
            ic_std = np.std(ics)
            icir = ic_mean / ic_std if abs(ic_std) > 1e-12 else np.nan
            ic_positive = np.mean([1 if x > 0 else 0 for x in ics])
            ic_results.append({
                "factor": factor_col,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "ic_positive_ratio": ic_positive,
                "n_periods": len(ics),
            })

    return pd.DataFrame(ic_results)


def run_group_backtest(score_df: pd.DataFrame, return_df: pd.DataFrame, horizon: int, n_groups: int = 5) -> pd.DataFrame:
    logger.info("分组回测: horizon=%s, n_groups=%s", horizon, n_groups)

    return_pivot = return_df.pivot(index="date", columns="ts_code", values="close")
    ret_forward = return_pivot.pct_change(horizon).shift(-horizon)

    if hasattr(ret_forward.index, "tz") and ret_forward.index.tz:
        ret_forward.index = ret_forward.index.tz_localize(None)

    score_df = score_df.copy()
    score_df["report_date"] = pd.to_datetime(score_df["report_date"], errors="coerce")
    if hasattr(score_df["report_date"].dtype, "tz") and score_df["report_date"].dt.tz is not None:
        score_df["report_date"] = score_df["report_date"].dt.tz_localize(None)

    score_by_date = score_df.set_index("report_date")
    report_dates = sorted(score_by_date.index.unique())

    group_results = []
    for date in report_dates:
        date_scores = score_by_date.loc[[date]] if date in score_by_date.index else pd.DataFrame()
        if date_scores.empty:
            continue
        factor_vals = pd.to_numeric(date_scores["holder_quality_score"], errors="coerce").dropna()
        if "ts_code" in date_scores.columns:
            factor_vals.index = date_scores["ts_code"].values
        if len(factor_vals) < n_groups * 2:
            continue

        ret_date = pd.Timestamp(date)
        if ret_date.tzinfo:
            ret_date = ret_date.tz_localize(None)
        ret_date = ret_date + pd.Timedelta(days=horizon * 2)
        if ret_date not in ret_forward.index:
            nearest_idx = ret_forward.index.searchsorted(ret_date)
            if nearest_idx >= len(ret_forward.index):
                continue
            ret_date = ret_forward.index[nearest_idx]
        ret_vals = ret_forward.loc[ret_date] if ret_date in ret_forward.index else pd.Series(dtype=float)

        common = factor_vals.index.intersection(ret_vals.dropna().index)
        if len(common) < n_groups * 2:
            continue

        factor_common = factor_vals.loc[common]
        ret_common = ret_vals.loc[common]

        quantiles = pd.qcut(factor_common.rank(method="first"), n_groups, labels=False, duplicates="drop")
        for g in range(n_groups):
            mask = quantiles == g
            if mask.sum() == 0:
                continue
            group_ret = ret_common[mask].mean()
            group_results.append({
                "date": date,
                "group": g + 1,
                "avg_return": group_ret,
                "n_stocks": mask.sum(),
            })

    if not group_results:
        return pd.DataFrame()

    df = pd.DataFrame(group_results)
    summary = df.groupby("group").agg(
        avg_return=("avg_return", "mean"),
        std_return=("avg_return", "std"),
        n_periods=("date", "count"),
    ).reset_index()

    if len(summary) >= 2:
        long_short = summary.iloc[-1]["avg_return"] - summary.iloc[0]["avg_return"]
        logger.info("多空收益 (G%d - G1): %.4f", n_groups, long_short)

    return summary


def run_evaluation(horizon: int = 60, min_stocks: int = 1) -> None:
    logger.info("=" * 60)
    logger.info("股东投资质量画像 - 因子评估")
    logger.info("horizon=%s, min_stocks=%s", horizon, min_stocks)
    logger.info("=" * 60)

    portrait_df = load_portrait(min_stocks=min_stocks)
    if portrait_df.empty:
        logger.warning("画像数据为空")
        return
    logger.info("画像数据: %s 个股东 (min_stocks=%s)", len(portrait_df), min_stocks)

    top10_df = load_top10_with_industry()
    if top10_df.empty:
        logger.warning("top10 数据为空")
        return
    logger.info("top10 数据: %s 行", len(top10_df))

    industry_df = load_industry_info()
    if not industry_df.empty:
        top10_df = top10_df.merge(industry_df, on="ts_code", how="left")

    score_df = build_stock_holder_score(top10_df, portrait_df)
    if score_df.empty:
        logger.warning("无法构建个股股东质量评分")
        return
    logger.info("个股评分: %s 条", len(score_df))

    start_date = (top10_df["report_date"].min() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    return_df = load_stock_returns(start_date, end_date)
    if return_df.empty:
        logger.warning("行情数据为空")
        return
    logger.info("行情数据: %s 行", len(return_df))

    ic_df = run_ic_analysis(score_df, return_df, horizon)
    if not ic_df.empty:
        logger.info("")
        logger.info("=== IC 分析结果 ===")
        for _, row in ic_df.iterrows():
            logger.info("  %s: IC=%.4f, ICIR=%.4f, IC正比=%.1f%%, 期数=%d",
                        row["factor"], row["ic_mean"], row["icir"],
                        row["ic_positive_ratio"] * 100, row["n_periods"])

        output_path = os.path.join(os.path.dirname(__file__), "holder_portrait_ic_summary.csv")
        ic_df.to_csv(output_path, index=False)
        logger.info("IC 结果已保存: %s", output_path)

    group_df = run_group_backtest(score_df, return_df, horizon)
    if not group_df.empty:
        logger.info("")
        logger.info("=== 分组回测结果 ===")
        for _, row in group_df.iterrows():
            logger.info("  G%d: avg_ret=%.4f, std=%.4f, 期数=%d",
                        int(row["group"]), row["avg_return"], row["std_return"], row["n_periods"])

        output_path = os.path.join(os.path.dirname(__file__), "holder_portrait_group_result.csv")
        group_df.to_csv(output_path, index=False)
        logger.info("分组结果已保存: %s", output_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("评估完成")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="股东投资质量画像 - 因子评估")
    parser.add_argument("--horizon", type=int, default=60, help="持有期(交易日)")
    parser.add_argument("--min-stocks", type=int, default=1, help="最少出现股票数过滤")
    args = parser.parse_args()
    run_evaluation(horizon=args.horizon, min_stocks=args.min_stocks)


if __name__ == "__main__":
    main()
