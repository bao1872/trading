# -*- coding: utf-8 -*-
"""
sr_experiment/db_adapter.py - 数据库适配层

Purpose: 薄封装 datasource/k_data_loader，为 sr_experiment 提供统一数据入口。
         处理 ts_code 格式转换、amount 缺失填充、列名标准化。

Public API:
    get_stock_pool(pool_name=None) -> DataFrame
    load_kline(ts_code, freq, start_date=None, end_date=None) -> DataFrame
    load_kline_panel(ts_codes, freq) -> dict[str, DataFrame]

How to Run:
    python sr_experiment/db_adapter.py --ts-code 300133.SZ --freq w
    python sr_experiment/db_adapter.py --list-pools

Side Effects: 只读数据库，无写入
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasource.k_data_loader import (
    build_name_map,
    get_all_codes,
    iter_k_data_with_names,
    load_k_data,
)
from datasource.database import query_df, get_session

_DB_COLUMNS_TO_DROP = {"id", "ts_code", "freq", "created_at"}
_STANDARD_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def _code_to_ts_code(code: str) -> str:
    if "." in code:
        return code
    if code.startswith(("6", "5")):
        return f"{code}.SH"
    return f"{code}.SZ"


def _standardize_kline(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in _DB_COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    if "amount" not in df.columns and "volume" in df.columns and "close" in df.columns:
        df["amount"] = df["volume"] * df["close"]
    keep = [c for c in _STANDARD_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in _STANDARD_COLUMNS]
    df = df[keep + extra]
    df = df.sort_index()
    return df


def get_stock_pool(pool_name: Optional[str] = None) -> pd.DataFrame:
    with get_session() as session:
        filters = {}
        if pool_name:
            filters["pool_name"] = pool_name
        df = query_df(
            session,
            table_name="stock_pools",
            columns=["ts_code", "name", "industry_l2", "industry_l3"],
            filters=filters,
        )
    return df


def load_kline(
    ts_code: str,
    freq: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    db_code = _code_to_ts_code(ts_code)
    df = load_k_data(db_code, freq=freq, start_date=start_date, end_date=end_date)
    if df.empty:
        return pd.DataFrame(columns=_STANDARD_COLUMNS)
    return _standardize_kline(df)


def load_kline_panel(
    ts_codes: list[str],
    freq: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    result = {}
    for code in ts_codes:
        df = load_kline(code, freq, start_date=start_date, end_date=end_date)
        if not df.empty:
            result[_code_to_ts_code(code)] = df
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DB adapter test")
    parser.add_argument("--ts-code", type=str, default="300133.SZ")
    parser.add_argument("--freq", type=str, default="w")
    parser.add_argument("--list-pools", action="store_true")
    args = parser.parse_args()

    if args.list_pools:
        pool = get_stock_pool()
        print(f"Stock pool size: {len(pool)}")
        print(pool.head())
    else:
        df = load_kline(args.ts_code, args.freq)
        print(f"Loaded {args.ts_code} [{args.freq}]: {len(df)} bars")
        if not df.empty:
            print(f"Columns: {list(df.columns)}")
            print(df.head(3))
