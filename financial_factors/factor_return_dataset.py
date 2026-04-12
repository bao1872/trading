#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    构建财务因子-未来收益数据集，用于评估因子对公告后股价涨跌的预测力。
    以 ann_date（公告日期）为时间锚点，计算公告后 5d/20d/60d 收益率及超额收益。

Inputs:
    - 数据库表: stock_financial_score_pool（因子原始值 + 维度分 + total_score）
    - 数据库表: stock_k_data freq='d'（日线行情）
    - 数据库表: stock_pools（股票池，用于行业信息）

Outputs:
    - 数据库表: factor_return_dataset
    - CSV文件（可选）

How to Run:
    # 小批量验证（50只股票）
    python financial_factors/factor_return_dataset.py --limit 50

    # 全量构建
    python financial_factors/factor_return_dataset.py

    # 导出CSV
    python financial_factors/factor_return_dataset.py --csv out/factor_return.csv

Examples:
    python financial_factors/factor_return_dataset.py --limit 50
    python financial_factors/factor_return_dataset.py --csv out/factor_return.csv

Side Effects:
    - 写入/更新数据库表 factor_return_dataset
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

try:
    from config import DATABASE_URL
except Exception:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_TABLE = "factor_return_dataset"

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

RETURN_HORIZONS = [5, 20, 60]

BENCHMARK_CODE = "000300.SH"


def get_db_engine():
    return create_engine(DATABASE_URL, pool_size=5, max_overflow=10, pool_recycle=3600)


def load_score_data(engine, limit: int = None) -> pd.DataFrame:
    cols = ["ts_code", "stock_name", "report_date", "ann_date"] + ALL_FACTOR_COLS
    col_clause = ", ".join(cols)
    sql = f"SELECT {col_clause} FROM stock_financial_score_pool"
    if limit:
        sql += f" ORDER BY ts_code, report_date LIMIT {limit}"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    logger.info(f"加载评分数据: {len(df)} 条, {df['ts_code'].nunique()} 只股票")
    return df


def load_daily_prices(engine) -> pd.DataFrame:
    sql = "SELECT ts_code, bar_time, close FROM stock_k_data WHERE freq = 'd'"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    df["trade_date"] = pd.to_datetime(df["bar_time"]).dt.strftime("%Y%m%d")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    logger.info(f"加载日线数据: {len(df)} 条, {df['ts_code'].nunique()} 只股票")
    return df[["ts_code", "trade_date", "close"]]


def load_benchmark_prices(engine) -> pd.DataFrame:
    from tushare_data.fetcher import fetch_market_data
    end_date = pd.Timestamp.now().strftime("%Y%m%d")
    start_date = "20230101"
    df = fetch_market_data(BENCHMARK_CODE, start_date, end_date)
    if df.empty:
        logger.warning("基准指数数据为空，超额收益将无法计算")
        return pd.DataFrame(columns=["trade_date", "close"])
    df = df.reset_index()
    df["trade_date"] = df["date"].dt.strftime("%Y%m%d")
    logger.info(f"加载基准指数数据: {len(df)} 条")
    return df[["trade_date", "close"]].rename(columns={"close": "bench_close"})


def build_trade_date_map(daily_df: pd.DataFrame) -> Dict[str, Set[str]]:
    all_dates = sorted(daily_df["trade_date"].unique())
    return set(all_dates), all_dates


def find_next_trade_date(ann_date_str: str, trade_dates_set: Set[str],
                         trade_dates_sorted: List[str]) -> Optional[str]:
    if ann_date_str in trade_dates_set:
        idx = trade_dates_sorted.index(ann_date_str)
        if idx + 1 < len(trade_dates_sorted):
            return trade_dates_sorted[idx + 1]
    for d in trade_dates_sorted:
        if d > ann_date_str:
            return d
    return None


def compute_forward_returns(
    score_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    bench_df: pd.DataFrame,
) -> pd.DataFrame:
    trade_dates_set, trade_dates_sorted = build_trade_date_map(daily_df)

    price_pivot = daily_df.pivot_table(index="trade_date", columns="ts_code", values="close")
    price_pivot = price_pivot.sort_index()

    bench_map = {}
    if not bench_df.empty:
        bench_map = dict(zip(bench_df["trade_date"], bench_df["bench_close"]))

    results = []
    grouped = score_df.groupby("ts_code")

    for ts_code, group in tqdm(grouped, desc="计算未来收益", ncols=100):
        if ts_code not in price_pivot.columns:
            continue

        stock_prices = price_pivot[ts_code].dropna()
        stock_dates = stock_prices.index.tolist()

        for _, row in group.iterrows():
            ann_date = row.get("ann_date")
            if pd.isna(ann_date) or str(ann_date).strip() == "" or str(ann_date) == "NaN":
                continue

            ann_str = str(int(ann_date)) if isinstance(ann_date, (int, float)) else str(ann_date).strip()
            if len(ann_str) != 8:
                continue

            entry_date = find_next_trade_date(ann_str, trade_dates_set, trade_dates_sorted)
            if entry_date is None:
                continue

            if entry_date not in stock_dates:
                continue

            entry_price = stock_prices.get(entry_date)
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            entry_idx = stock_dates.index(entry_date)
            record = {
                "ts_code": ts_code,
                "stock_name": row.get("stock_name", ""),
                "report_date": row.get("report_date", ""),
                "ann_date": ann_str,
                "entry_date": entry_date,
                "entry_price": entry_price,
            }

            for factor_col in ALL_FACTOR_COLS:
                if factor_col in row.index:
                    record[factor_col] = row[factor_col]

            for horizon in RETURN_HORIZONS:
                target_idx = entry_idx + horizon
                if target_idx < len(stock_dates):
                    exit_date = stock_dates[target_idx]
                    exit_price = stock_prices.iloc[target_idx]
                    if pd.notna(exit_price) and exit_price > 0:
                        ret = (exit_price - entry_price) / entry_price
                        record[f"ret_{horizon}d"] = ret

                        entry_bench = bench_map.get(entry_date)
                        exit_bench = bench_map.get(exit_date)
                        if entry_bench and exit_bench and entry_bench > 0:
                            bench_ret = (exit_bench - entry_bench) / entry_bench
                            record[f"ret_{horizon}d_excess"] = ret - bench_ret
                        else:
                            record[f"ret_{horizon}d_excess"] = np.nan
                    else:
                        record[f"ret_{horizon}d"] = np.nan
                        record[f"ret_{horizon}d_excess"] = np.nan
                else:
                    record[f"ret_{horizon}d"] = np.nan
                    record[f"ret_{horizon}d_excess"] = np.nan

            results.append(record)

    result_df = pd.DataFrame(results)
    logger.info(f"构建完成: {len(result_df)} 条记录, {result_df['ts_code'].nunique()} 只股票")
    return result_df


def _qc(col: str) -> str:
    return f'"{col}"'


def ensure_output_table(engine, df: pd.DataFrame):
    type_map = {
        "object": "VARCHAR",
        "float64": "DOUBLE PRECISION",
        "int64": "BIGINT",
    }
    col_defs = ["id SERIAL PRIMARY KEY"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        pg_type = type_map.get(dtype, "VARCHAR")
        col_defs.append(f"{_qc(col)} {pg_type}")
    col_defs.append("created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP")
    col_defs.append("updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP")

    sql = f'CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} ({", ".join(col_defs)})'
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info(f"确保表 {OUTPUT_TABLE} 存在")


def upsert_to_db(df: pd.DataFrame, engine):
    if df.empty:
        logger.warning("数据为空，跳过写入")
        return

    ensure_output_table(engine, df)

    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            df_copy[col] = df_copy[col].where(df_copy[col].notna(), None)

    df_copy.to_sql(
        OUTPUT_TABLE, engine, if_exists="append", index=False, method="multi", chunksize=5000
    )
    logger.info(f"写入 {len(df_copy)} 条记录到 {OUTPUT_TABLE}")


def main():
    parser = argparse.ArgumentParser(description="构建财务因子-未来收益数据集")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数量（用于测试）")
    parser.add_argument("--csv", type=str, default=None, help="导出CSV路径")
    args = parser.parse_args()

    engine = get_db_engine()

    score_df = load_score_data(engine, limit=args.limit)
    daily_df = load_daily_prices(engine)
    bench_df = load_benchmark_prices(engine)

    result_df = compute_forward_returns(score_df, daily_df, bench_df)

    if result_df.empty:
        logger.error("结果为空，请检查数据")
        return 1

    for horizon in RETURN_HORIZONS:
        ret_col = f"ret_{horizon}d"
        if ret_col in result_df.columns:
            valid = result_df[ret_col].notna().sum()
            logger.info(f"  {ret_col}: {valid}/{len(result_df)} 有效 ({valid/len(result_df)*100:.1f}%)")

    upsert_to_db(result_df, engine)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        result_df.to_csv(args.csv, index=False)
        logger.info(f"已导出CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
