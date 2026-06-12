#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过滤器质量评估引擎

Purpose:
    评估模型筛选 TopK 的质量——验证筛出来的 TopK 是否比全候选池更值得花时间看。
    从"账户NAV回测"转向"过滤器质量评估"，不模拟交易，只看信号预测能力。

Inputs:
    - prediction_store: 每日预测 parquet (通过 read_prediction_store_range 读取)
    - DB: stock_k_data (K线价格)

Outputs:
    - stop_experiment/output/filter_quality/ 目录下 5 个 CSV:
      1. daily_topk_quality.csv
      2. score_bucket_quality.csv
      3. good_stock_coverage.csv
      4. bad_stock_risk.csv
      5. filter_quality_summary.csv

How to Run:
    python -m stop_experiment.eval.filter_quality_evaluator --start-date 2026-01-05 --end-date 2026-05-15
    python -m stop_experiment.eval.filter_quality_evaluator --start-date 2026-01-05 --end-date 2026-05-15 --force

Examples:
    python -m stop_experiment.eval.filter_quality_evaluator --start-date 2026-01-05 --end-date 2026-05-15
    python -m stop_experiment.eval.filter_quality_evaluator --start-date 2026-03-01 --end-date 2026-03-31 --force

Side Effects:
    - 只读 DB 和 prediction_store
    - 写入 filter_quality/ 目录下的 CSV 文件
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from stop_experiment.pipeline.stop_config import FILTER_QUALITY_DIR
from stop_experiment.registries.prediction_store import read_prediction_store_range
from stop_experiment.backtest.simple_backtest import load_daily_prices, build_price_pivot

TOPK_LIST = [10, 20, 30, 50]
HORIZONS_DAILY = [1, 3, 5, 10]
HORIZONS_BUCKET = [5, 10]
HORIZONS_ALL = [1, 3, 5, 10, 20]
N_BUCKETS = 5

GOOD_CRITERIA = [
    ("5日涨幅>5%", 5, "return", 0.05, ">"),
    ("10日涨幅>8%", 10, "return", 0.08, ">"),
    ("20日MFE>10%", 20, "mfe", 0.10, ">"),
]

BAD_CRITERIA = [
    ("5日MAE<-5%", 5, "mae", -0.05, "<"),
    ("10日MAE<-8%", 10, "mae", -0.08, "<"),
]

OUTPUT_FILES = [
    "daily_topk_quality.csv",
    "score_bucket_quality.csv",
    "good_stock_coverage.csv",
    "bad_stock_risk.csv",
    "filter_quality_summary.csv",
]


def compute_future_metrics(price_pivot, horizons):
    """向量化计算所有股票在指定horizons下的未来收益/MFE/MAE。

    以 close[T] 为基准价（与回测引擎以 buy_price=open[T+1] 为基准不同）。
    return_N = close[T+N] / close[T] - 1
    MFE_N    = max(high[T:T+N+1]) / close[T] - 1
    MAE_N    = min(low[T:T+N+1]) / close[T] - 1
    """
    close_df = price_pivot["close"]
    high_df = price_pivot["high"]
    low_df = price_pivot["low"]

    metrics = {}
    for N in horizons:
        ret = close_df.shift(-N) / close_df - 1
        mfe = high_df.rolling(N + 1, min_periods=1).max().shift(-N) / close_df - 1
        mae = low_df.rolling(N + 1, min_periods=1).min().shift(-N) / close_df - 1
        metrics[N] = {"return": ret, "mfe": mfe, "mae": mae}

    return metrics


def _lookup_values(metric_df, dates, raw_codes):
    """向量化查找: 从metric_df中按(dates, raw_codes)提取值，找不到返回NaN。"""
    dates_arr = pd.to_datetime(dates)
    date_idx = metric_df.index.get_indexer(dates_arr)
    code_idx = metric_df.columns.get_indexer(raw_codes)
    valid = (date_idx >= 0) & (code_idx >= 0)
    result = np.full(len(dates), np.nan)
    if valid.any():
        result[valid] = metric_df.values[date_idx[valid], code_idx[valid]]
    return result


def enrich_with_future_metrics(pred_df, future_metrics, horizons):
    """为预测DataFrame添加未来收益/MFE/MAE列。"""
    pred_df = pred_df.copy()
    pred_df["raw_code"] = pred_df["ts_code"].str[:6]
    pred_df["obs_date"] = pd.to_datetime(pred_df["obs_date"]).dt.normalize()

    for N in horizons:
        for metric_name in ["return", "mfe", "mae"]:
            col = f"{metric_name}_{N}"
            pred_df[col] = _lookup_values(
                future_metrics[N][metric_name],
                pred_df["obs_date"].values,
                pred_df["raw_code"].values,
            )

    return pred_df


def _group_stats(group_df, horizon):
    """计算单组在指定horizon下的统计指标，无有效数据返回None。"""
    ret_col = f"return_{horizon}"
    valid = group_df[ret_col].notna()
    n_valid = valid.sum()
    if n_valid == 0:
        return None

    ret = group_df.loc[valid, ret_col]
    mfe = group_df.loc[valid, f"mfe_{horizon}"]
    mae = group_df.loc[valid, f"mae_{horizon}"]

    return {
        "n_stocks": int(n_valid),
        "avg_return": ret.mean(),
        "med_return": ret.median(),
        "win_rate": (ret > 0).mean(),
        "avg_mfe": mfe.mean(),
        "avg_mae": mae.mean(),
        "max_dd_rate": (mae < -0.05).mean(),
    }


def evaluate_all(pred_df):
    """单次遍历所有交易日，计算4类评估指标。"""
    dates = sorted(pred_df["obs_date"].unique())
    topk_rows, bucket_rows, good_rows, bad_rows = [], [], [], []

    for date in tqdm(dates, desc="Evaluating"):
        day_df = pred_df[pred_df["obs_date"] == date].sort_values("score", ascending=False)
        if day_df.empty:
            continue

        # ---- daily_topk_quality ----
        groups = {"全候选池": day_df}
        for k in TOPK_LIST:
            groups[f"Top{k}"] = day_df.head(k)

        for group_name, gdf in groups.items():
            for h in HORIZONS_DAILY:
                s = _group_stats(gdf, h)
                if s:
                    topk_rows.append({"date": date, "group": group_name, "horizon": h, **s})

        # ---- score_bucket_quality ----
        if len(day_df) >= N_BUCKETS:
            try:
                tmp = day_df.copy()
                tmp["bucket"] = pd.qcut(
                    tmp["score"], N_BUCKETS, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
                )
                for bucket in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
                    bdf = tmp[tmp["bucket"] == bucket]
                    for h in HORIZONS_BUCKET:
                        s = _group_stats(bdf, h)
                        if s:
                            bucket_rows.append({"date": date, "bucket": bucket, "horizon": h, **s})
            except ValueError:
                pass

        # ---- good_stock_coverage ----
        top30 = day_df.head(30)
        top50 = day_df.head(50)

        for cname, horizon, metric, threshold, cmp in GOOD_CRITERIA:
            col = f"{metric}_{horizon}"
            if col not in day_df.columns:
                continue
            pool_v = day_df[col].dropna()
            t30_v = top30[col].dropna()
            t50_v = top50[col].dropna()
            if cmp == ">":
                total_good = (pool_v > threshold).sum()
                t30_c = (t30_v > threshold).sum()
                t50_c = (t50_v > threshold).sum()
            else:
                total_good = (pool_v < threshold).sum()
                t30_c = (t30_v < threshold).sum()
                t50_c = (t50_v < threshold).sum()
            good_rows.append({
                "date": date,
                "criterion": cname,
                "total_good": int(total_good),
                "top30_covered": int(t30_c),
                "top50_covered": int(t50_c),
                "top30_rate": t30_c / max(total_good, 1),
                "top50_rate": t50_c / max(total_good, 1),
            })

        # ---- bad_stock_risk ----
        for cname, horizon, metric, threshold, _cmp in BAD_CRITERIA:
            col = f"{metric}_{horizon}"
            if col not in day_df.columns:
                continue
            pool_v = day_df[col].dropna()
            t30_v = top30[col].dropna()
            t50_v = top50[col].dropna()
            pool_bad = (pool_v < threshold).sum()
            t30_bad = (t30_v < threshold).sum()
            t50_bad = (t50_v < threshold).sum()
            bad_rows.append({
                "date": date,
                "criterion": cname,
                "pool_bad_rate": pool_bad / max(len(pool_v), 1),
                "top30_bad_rate": t30_bad / max(len(t30_v), 1),
                "top50_bad_rate": t50_bad / max(len(t50_v), 1),
                "top30_count": int(t30_bad),
                "top50_count": int(t50_bad),
                "pool_count": int(pool_bad),
            })

    return (
        pd.DataFrame(topk_rows),
        pd.DataFrame(bucket_rows),
        pd.DataFrame(good_rows),
        pd.DataFrame(bad_rows),
    )


def build_summary(daily_topk, good_coverage, bad_risk):
    """构建汇总表: 一行总结整个评估期间的统计。"""
    groups = ["Top10", "Top20", "Top30", "Top50", "全候选池"]
    rows = []

    for horizon in [5, 10]:
        h_df = daily_topk[daily_topk["horizon"] == horizon]
        for mname, col in [(f"{horizon}日平均收益", "avg_return"), (f"{horizon}日胜率", "win_rate")]:
            row = {"metric": mname}
            for g in groups:
                vals = h_df.loc[h_df["group"] == g, col]
                row[g] = vals.mean() if len(vals) > 0 else np.nan
            rows.append(row)
        if horizon == 5:
            for mname, col in [("5日MFE", "avg_mfe"), ("5日MAE", "avg_mae")]:
                row = {"metric": mname}
                for g in groups:
                    vals = h_df.loc[h_df["group"] == g, col]
                    row[g] = vals.mean() if len(vals) > 0 else np.nan
                rows.append(row)

    for cname, _, _, _, _ in GOOD_CRITERIA:
        c_df = good_coverage[good_coverage["criterion"] == cname]
        row = {"metric": f"好票覆盖率({cname})"}
        for g in groups:
            if g == "Top30":
                row[g] = c_df["top30_rate"].mean() if len(c_df) > 0 else np.nan
            elif g == "Top50":
                row[g] = c_df["top50_rate"].mean() if len(c_df) > 0 else np.nan
            else:
                row[g] = np.nan
        rows.append(row)

    for cname, _, _, _, _ in BAD_CRITERIA:
        c_df = bad_risk[bad_risk["criterion"] == cname]
        row = {"metric": f"坏票比例({cname})"}
        for g in groups:
            if g == "Top30":
                row[g] = c_df["top30_bad_rate"].mean() if len(c_df) > 0 else np.nan
            elif g == "Top50":
                row[g] = c_df["top50_bad_rate"].mean() if len(c_df) > 0 else np.nan
            elif g == "全候选池":
                row[g] = c_df["pool_bad_rate"].mean() if len(c_df) > 0 else np.nan
            else:
                row[g] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)[["metric", "Top10", "Top20", "Top30", "Top50", "全候选池"]]


def run_evaluation(start_date, end_date, force=False):
    """运行过滤器质量评估主流程。"""
    os.makedirs(FILTER_QUALITY_DIR, exist_ok=True)

    if not force and all(
        os.path.exists(os.path.join(FILTER_QUALITY_DIR, f)) for f in OUTPUT_FILES
    ):
        print("  输出文件已存在，使用 --force 重新计算")
        return

    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)

    print("[1/4] 加载K线数据...")
    price_end = (ed + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    daily_prices = load_daily_prices(start_date, price_end)
    price_pivot, trading_days_all, _ = build_price_pivot(daily_prices)
    trading_days = [d for d in trading_days_all if sd <= d <= ed]
    print(
        f"  交易日: {len(trading_days)}, "
        f"价格覆盖: {trading_days_all[0].strftime('%Y-%m-%d')} ~ {trading_days_all[-1].strftime('%Y-%m-%d')}"
    )

    print("[2/4] 加载预测数据...")
    df_all, _ = read_prediction_store_range("production", trading_days)
    if df_all.empty:
        print("  无预测数据，退出")
        return
    print(
        f"  预测: {len(df_all)} 行, "
        f"日期: {df_all['obs_date'].min().strftime('%Y-%m-%d')} ~ {df_all['obs_date'].max().strftime('%Y-%m-%d')}"
    )

    print("[3/4] 计算未来收益指标...")
    future_metrics = compute_future_metrics(price_pivot, HORIZONS_ALL)
    pred_df = enrich_with_future_metrics(df_all, future_metrics, HORIZONS_ALL)

    print("[4/4] 评估过滤器质量...")
    daily_topk, score_bucket, good_coverage, bad_risk = evaluate_all(pred_df)
    summary = build_summary(daily_topk, good_coverage, bad_risk)

    for fname, df in zip(
        OUTPUT_FILES, [daily_topk, score_bucket, good_coverage, bad_risk, summary]
    ):
        path = os.path.join(FILTER_QUALITY_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  {fname}: {len(df)} 行")

    print("\n  汇总表:")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="过滤器质量评估引擎")
    parser.add_argument("--start-date", required=True, help="评估开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="评估结束日期 (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    run_evaluation(args.start_date, args.end_date, force=args.force)


if __name__ == "__main__":
    main()
