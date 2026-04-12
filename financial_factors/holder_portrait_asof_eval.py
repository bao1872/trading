# -*- coding: utf-8 -*-
"""
Purpose:
    股东投资质量画像模型 - as-of 评估脚本
    使用 as-of 画像避免前视偏差, 逐期构建画像并评估预测力

Inputs:
    - 数据库表: stock_top10_holders_tushare
    - 数据库表: stock_k_data (行情)
    - 数据库表: stock_pools (行业信息)

Outputs:
    - 控制台: as-of 评估报告
    - CSV: holder_portrait_asof_eval.csv

How to Run:
    python financial_factors/holder_portrait_asof_eval.py --horizon 60
    python financial_factors/holder_portrait_asof_eval.py --horizon 20 --min-realized 60

Examples:
    python financial_factors/holder_portrait_asof_eval.py --horizon 60
    python financial_factors/holder_portrait_asof_eval.py --horizon 20 --min-realized 40

Side Effects:
    - 写 CSV 文件到 financial_factors/ 目录
    - 无写库操作
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_session, query_df
from financial_factors.holder_quality_portrait import (
    PORTRAIT_TABLE,
    build_change_events,
    build_event_metrics_cache,
    build_holder_tenure,
    compute_composite_score,
    compute_holder_portrait,
    filter_events_asof,
    load_bench_data,
    load_industry_info,
    load_market_data_from_db,
    load_stock_pool,
    load_top10_data,
    normalize_holder_name,
    normalize_ts_code,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_all_events() -> pd.DataFrame:
    with get_session() as session:
        top10 = query_df(session, "stock_top10_holders_tushare")
    if top10.empty:
        return top10
    top10 = top10.copy()
    top10["ts_code"] = top10["ts_code"].astype(str).map(normalize_ts_code)
    top10["report_date"] = pd.to_datetime(top10["report_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    top10["ann_date"] = pd.to_datetime(top10["ann_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    top10["holder_name_std"] = top10["holder_name"].map(normalize_holder_name)
    for col in ["hold_amount", "hold_ratio", "hold_float_ratio"]:
        if col in top10.columns:
            top10[col] = pd.to_numeric(top10[col], errors="coerce")
    if "holder_rank" in top10.columns:
        top10["holder_rank"] = pd.to_numeric(top10["holder_rank"], errors="coerce").astype("Int64")

    industry_df = load_industry_info()
    if not industry_df.empty:
        top10 = top10.merge(industry_df.drop(columns=["name"], errors="ignore"), on="ts_code", how="left")

    top10 = build_holder_tenure(top10)

    stock_df_map: Dict[str, pd.DataFrame] = {}
    unique_codes = top10["ts_code"].drop_duplicates().tolist()
    start_date_global = top10["ann_date"].dropna().min()
    if pd.isna(start_date_global):
        start_date_global = top10["report_date"].dropna().min()
    end_date = pd.Timestamp.today().normalize()

    from tqdm import tqdm
    for ts_code in tqdm(unique_codes, desc="加载行情", unit="股票"):
        g = top10[top10["ts_code"] == ts_code]
        start_date = g["ann_date"].dropna().min()
        if pd.isna(start_date):
            start_date = g["report_date"].dropna().min()
        if pd.isna(start_date):
            continue
        stock_df = load_market_data_from_db(ts_code, start_date - pd.Timedelta(days=250), end_date)
        if not stock_df.empty:
            stock_df_map[ts_code] = stock_df

    valid_codes = [c for c in unique_codes if c in stock_df_map]
    top10 = top10[top10["ts_code"].isin(valid_codes)].copy()

    bench_df = load_bench_data(start_date_global - pd.Timedelta(days=250), end_date)

    event_cache = build_event_metrics_cache(top10, stock_df_map, bench_df)
    events_df = build_change_events(top10, event_cache)
    return events_df


def run_asof_evaluation(horizon: int = 60, min_realized: int = 60) -> None:
    logger.info("=" * 60)
    logger.info("as-of 画像评估")
    logger.info("horizon=%s, min_realized=%s", horizon, min_realized)
    logger.info("=" * 60)

    events_df = load_all_events()
    if events_df.empty:
        logger.warning("无事件数据")
        return
    logger.info("事件数据: %s 条", len(events_df))

    from sqlalchemy import text
    with get_session() as session:
        ret_df = pd.DataFrame(session.execute(text(
            "SELECT ts_code, bar_time as date, close FROM stock_k_data WHERE freq = 'd' ORDER BY ts_code, bar_time"
        )).fetchall(), columns=["ts_code", "date", "close"])
    ret_df["ts_code"] = ret_df["ts_code"].astype(str).map(normalize_ts_code)
    ret_df["date"] = pd.to_datetime(ret_df["date"], utc=True).dt.tz_localize(None)
    ret_df["close"] = pd.to_numeric(ret_df["close"], errors="coerce")
    pivot = ret_df.pivot(index="date", columns="ts_code", values="close")
    ret_fwd = pivot.pct_change(horizon).shift(-horizon)

    report_dates = sorted(events_df["report_date"].dropna().unique())
    logger.info("报告期数: %s", len(report_dates))

    dims = ["picking_score", "adapt_score", "composite_score"]
    all_ic_results: List[Dict] = []
    all_group_results: List[Dict] = []

    for ri, report_date in enumerate(report_dates):
        ann_date = events_df[events_df["report_date"] == report_date]["ann_date"].dropna().min()
        if pd.isna(ann_date):
            continue
        effective_date = pd.Timestamp(ann_date) + pd.Timedelta(days=1)

        hist_events = filter_events_asof(events_df, effective_date, min_realized_bdays=min_realized)
        if hist_events.empty:
            continue

        portrait = compute_holder_portrait(hist_events, hist_events, None)
        if portrait.empty:
            continue
        portrait = compute_composite_score(portrait)

        quality_map = dict(zip(
            portrait["holder_name_std"],
            pd.to_numeric(portrait["composite_score"], errors="coerce")
        ))
        dim_maps = {}
        for dim in dims:
            dim_maps[dim] = dict(zip(portrait["holder_name_std"], pd.to_numeric(portrait[dim], errors="coerce")))

        current_top10 = events_df[events_df["report_date"] == report_date]
        score_records = []
        for ts_code, group in current_top10.groupby("ts_code"):
            rec = {"ts_code": ts_code}
            for dim in dims:
                vals, wts = [], []
                for _, row in group.iterrows():
                    name_std = row["holder_name_std"]
                    if name_std in dim_maps[dim] and pd.notna(dim_maps[dim][name_std]):
                        ratio = float(row.get("hold_float_ratio_curr", 1.0)) if pd.notna(row.get("hold_float_ratio_curr")) else 1.0
                        if ratio <= 0:
                            ratio = 1.0
                        vals.append(dim_maps[dim][name_std])
                        wts.append(ratio)
                if vals:
                    w = np.array(wts)
                    rec[dim] = float(np.average(vals, weights=w)) if w.sum() > 0 else float(np.mean(vals))
            score_records.append(rec)

        if not score_records:
            continue

        score_df = pd.DataFrame(score_records)
        factor_idx = score_df["ts_code"].values

        target_date = pd.Timestamp(report_date) + pd.Timedelta(days=horizon * 2)
        idx = ret_fwd.index
        pos = idx.searchsorted(target_date)
        if pos >= len(idx):
            continue
        actual_date = idx[pos]
        ret_vals = ret_fwd.loc[actual_date].dropna()

        for dim in dims:
            if dim not in score_df.columns:
                continue
            factor = pd.to_numeric(score_df[dim], errors="coerce")
            factor.index = factor_idx
            factor = factor.dropna()
            common = factor.index.intersection(ret_vals.index)
            if len(common) < 20:
                continue
            ic, _ = stats.spearmanr(factor.loc[common], ret_vals.loc[common])
            if not np.isnan(ic):
                all_ic_results.append({
                    "report_date": report_date,
                    "dim": dim,
                    "ic": ic,
                    "n_stocks": len(common),
                })

        if "composite_score" in score_df.columns:
            factor = pd.to_numeric(score_df["composite_score"], errors="coerce")
            factor.index = factor_idx
            factor = factor.dropna()
            common = factor.index.intersection(ret_vals.index)
            if len(common) >= 10:
                try:
                    quantiles = pd.qcut(factor.loc[common].rank(method="first"), 5, labels=False, duplicates="drop")
                    for g in range(5):
                        mask = quantiles == g
                        if mask.sum() == 0:
                            continue
                        all_group_results.append({
                            "report_date": report_date,
                            "group": g + 1,
                            "ret": ret_vals.loc[common].loc[mask].mean(),
                        })
                except ValueError:
                    pass

        if (ri + 1) % 5 == 0 or ri == len(report_dates) - 1:
            logger.info("进度: %s/%s 报告期", ri + 1, len(report_dates))

    if not all_ic_results:
        logger.warning("无 IC 结果")
        return

    ic_df = pd.DataFrame(all_ic_results)
    logger.info("")
    logger.info("=== as-of IC 分析结果 (horizon=%s) ===", horizon)
    for dim in dims:
        dim_ics = ic_df[ic_df["dim"] == dim]["ic"]
        if len(dim_ics) > 0:
            ic_mean = dim_ics.mean()
            ic_std = dim_ics.std()
            icir = ic_mean / ic_std if abs(ic_std) > 1e-12 else np.nan
            ic_pos = (dim_ics > 0).mean()
            logger.info("  %s: IC=%+.4f, ICIR=%+.4f, IC正比=%.1f%%, 期数=%d",
                        dim, ic_mean, icir, ic_pos * 100, len(dim_ics))

    if not all_group_results:
        return

    grp_df = pd.DataFrame(all_group_results)
    logger.info("")
    logger.info("=== as-of 分组回测 (5分组, horizon=%s) ===", horizon)
    for g in range(1, 6):
        g_rets = grp_df[grp_df["group"] == g]["ret"]
        if len(g_rets) > 0:
            logger.info("  G%d: avg_ret=%+.4f, 期数=%d", g, g_rets.mean(), len(g_rets))
    g1 = grp_df[grp_df["group"] == 1]["ret"].mean()
    g5 = grp_df[grp_df["group"] == 5]["ret"].mean()
    logger.info("  多空(G5-G1): %+.4f", g5 - g1)

    output_path = os.path.join(os.path.dirname(__file__), "holder_portrait_asof_eval.csv")
    ic_df.to_csv(output_path, index=False)
    logger.info("as-of 评估结果已保存: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="股东投资质量画像 - as-of 评估")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--min-realized", type=int, default=60, help="as-of 最少已实现交易日数")
    args = parser.parse_args()
    run_asof_evaluation(horizon=args.horizon, min_realized=args.min_realized)


if __name__ == "__main__":
    main()
