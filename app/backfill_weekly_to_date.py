# -*- coding: utf-8 -*-
"""回补周线数据到指定日期

用途：
    回补股票池全量股票的周线行情数据到指定日期

运行命令：
    python app/backfill_weekly_to_date.py

说明：
    - 获取500根周线数据（约10年）
    - 只保留到指定日期（2026-04-24）的数据
    - 保存到数据库
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict

import pandas as pd
from tqdm import tqdm

# 确保项目根目录在 Python 路径最前面
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir in sys.path:
    sys.path.remove(base_dir)
sys.path.insert(0, base_dir)

# 切换到项目根目录
os.chdir(base_dir)

import config
from datasource.pytdx_client import connect_pytdx, get_kline_data
from datasource.database import get_session, bulk_upsert

# 目标日期
TARGET_DATE = "2026-04-24"


def _code_to_symbol(code: str) -> str:
    """code 如 600489 -> 600489.SH"""
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"


def save_to_database(session, data_cache: Dict[str, Dict], freq: str):
    """将数据缓存保存到数据库"""
    db_freq = freq

    all_dfs = []
    for code, stock_data in tqdm(data_cache.items(), desc=f"整理{freq}数据"):
        df = stock_data.get("data")
        if df is None or df.empty:
            continue

        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.isna()]
        df = df.sort_index()

        ts_code = _code_to_symbol(code)
        df["ts_code"] = ts_code
        df["freq"] = db_freq
        df["bar_time"] = df.index
        df["name"] = stock_data.get("name", "")

        df_for_db = df.reset_index(drop=True)
        df_for_db = df_for_db[["ts_code", "freq", "bar_time", "open", "high", "low", "close", "volume"]]
        df_for_db["bar_time"] = pd.to_datetime(df_for_db["bar_time"])

        # 周线只保存日期（不带时间）
        df_for_db["bar_time"] = df_for_db["bar_time"].dt.strftime("%Y-%m-%d")

        all_dfs.append(df_for_db)

    if not all_dfs:
        print(f"⚠️  无数据需要保存")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    total_rows = len(combined_df)
    print(f"\n正在保存 {total_rows} 条数据到数据库...")

    batch_size = 50000
    total_batches = (total_rows + batch_size - 1) // batch_size
    saved_count = 0

    try:
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_rows)
            batch_df = combined_df.iloc[start_idx:end_idx]
            bulk_upsert(session, "stock_k_data", batch_df, unique_keys=["ts_code", "freq", "bar_time"])
            saved_count += len(batch_df)
            print(f"  批次 {i+1}/{total_batches}: 已保存 {saved_count}/{total_rows} 条")
        print(f"✅ 成功保存 {saved_count} 条数据")
    except Exception as e:
        print(f"保存失败: {e}")


def backfill_weekly_to_date(cache_df: pd.DataFrame, target_date: str):
    """
    回补周线数据到指定日期

    参数：
        cache_df: 股票池DataFrame
        target_date: 目标日期字符串，如 "2026-04-24"
    """
    target_dt = pd.to_datetime(target_date).normalize()

    print(f"\n{'='*70}")
    print(f"📥 回补周线数据到 {target_date}")
    print(f"{'='*70}")
    print(f"股票池: {len(cache_df)} 只")
    print(f"{'='*70}\n")

    api = connect_pytdx()
    data_cache: Dict[str, Dict] = {}

    try:
        for _, row in tqdm(cache_df.iterrows(), total=len(cache_df), desc="获取周线数据"):
            symbol = row['ts_code'].split('.')[0]
            name = row['name']

            try:
                # 获取500根周线数据（约10年）
                df = get_kline_data(api, symbol, "w", 500)
                if df.empty or len(df) < 10:
                    continue

                df = df.set_index("datetime")

                # 只保留到目标日期的数据
                df = df[df.index <= target_dt]

                if len(df) < 10:
                    continue

                data_cache[symbol] = {
                    "name": name,
                    "data": df,
                }

            except Exception as e:
                continue

    finally:
        api.disconnect()

    print(f"\n✅ 数据获取完成: {len(data_cache)} 只股票")

    if data_cache:
        total_bars = sum(len(d["data"]) for d in data_cache.values())
        print(f"总计 {total_bars} 条周线数据")
        print(f"正在保存到数据库...")
        with get_session() as session:
            save_to_database(session, data_cache, "w")
        print(f"✅ 数据已保存到数据库")
    else:
        print(f"⚠️  无数据需要保存")

    return len(data_cache)


def main():
    from datasource.database import query_df

    with get_session() as session:
        cache_df = query_df(session, "stock_pools", columns=["ts_code", "name"])
    print(f"从数据库 stock_pools 读取股票池: {len(cache_df)} 只")

    # 回补周线数据到4月24号
    backfill_weekly_to_date(cache_df, TARGET_DATE)

    print(f"\n{'='*70}")
    print(f"✅ 周线数据回补完成")
    print(f"目标日期: {TARGET_DATE}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
