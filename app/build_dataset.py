# -*- coding: utf-8 -*-
"""
构建多周期数据集

用途：
    获取股票池全量股票的多周期行情数据并保存到数据库

数据规格：
    - 日线：500 bar
    - 周线：500 bar
    - 60分钟：2000 bar

保存路径：
    PostgreSQL 远程数据库 (config.DATABASE_URL)

运行命令：
    python app/build_dataset.py
    python app/build_dataset.py --update

注意：
    这是一个耗时操作，只需要运行一次
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Dict

import pandas as pd
from tqdm import tqdm

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from datasource.pytdx_client import connect_pytdx, get_kline_data
from datasource.database import get_session, bulk_upsert


def resample_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """将日线数据转换为周线"""
    df = df_daily.copy()
    df['week'] = df.index.to_period('W')

    weekly = df.groupby('week').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })

    weekly.index = weekly.index.to_timestamp()
    return weekly


def _symbol_to_code(ts_code: str) -> str:
    """ts_code 如 600489.SH -> 600489"""
    return ts_code.split(".")[0]


def _code_to_symbol(code: str) -> str:
    """code 如 600489 -> 600489.SH"""
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"


def save_to_database(session, data_cache: Dict[str, Dict], freq: str):
    """
    将数据缓存保存到数据库

    Args:
        session: 数据库会话
        data_cache: 数据字典 {code: {'name': xxx, 'data': DataFrame}}
        freq: 周期 ('d', 'w', '60m')
    """
    db_freq_map = {'d': 'd', 'w': 'w', '60': '60m'}
    db_freq = db_freq_map.get(freq, freq)

    all_dfs = []
    for code, stock_data in tqdm(data_cache.items(), desc=f"整理{freq}"):
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
        df_for_db["bar_time"] = pd.to_datetime(df_for_db["bar_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        all_dfs.append(df_for_db)

    if not all_dfs:
        print(f"⚠️  无数据需要保存")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n正在保存 {len(combined_df)} 条数据到数据库...")

    try:
        bulk_upsert(session, "stock_k_data", combined_df, unique_keys=["ts_code", "freq", "bar_time"])
        print(f"✅ 成功保存 {len(combined_df)} 条数据")
    except Exception as e:
        print(f"保存失败: {e}")


def fetch_and_save_data(cache_df: pd.DataFrame, bar_type: str, bar_count: int, save_to_db: bool = True):
    """
    获取数据并保存

    参数：
        bar_type: 'd'日线, 'w'周线, '60'60分钟
        bar_count: K线数量
        save_to_db: 是否保存到数据库，默认True
    """
    print(f"\n{'='*70}")
    print(f"📥 获取 {bar_type} 数据 ({bar_count} bar)")
    print(f"{'='*70}")
    print(f"股票池: {len(cache_df)} 只")
    print(f"{'='*70}\n")

    api = connect_pytdx()
    data_cache: Dict[str, Dict] = {}

    try:
        for _, row in tqdm(cache_df.iterrows(), total=len(cache_df), desc=f"获取{bar_type}数据"):
            symbol = row['ts_code'].split('.')[0]
            name = row['name']

            try:
                if bar_type == 'w':
                    df = get_kline_data(api, symbol, "w", bar_count)
                    if df.empty or len(df) < 100:
                        continue
                    df = df.set_index("datetime")
                elif bar_type == '60':
                    df = get_kline_data(api, symbol, "60m", bar_count)
                    if df.empty or len(df) < 100:
                        continue
                    df = df.set_index("datetime")
                else:
                    df = get_kline_data(api, symbol, bar_type, bar_count)
                    if df.empty or len(df) < 100:
                        continue
                    df = df.set_index("datetime")

                if len(df) < 50:
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

    if save_to_db and data_cache:
        print(f"正在保存到数据库...")
        with get_session() as session:
            save_to_database(session, data_cache, bar_type)
        print(f"✅ 数据已保存到数据库")
    else:
        print(f"⚠️  未保存到数据库（save_to_db=False）")

    return len(data_cache)


def _is_market_close_for_weekly() -> bool:
    """判断当前是否已到周五收盘后（可以更新周线）"""
    now = datetime.now()
    is_friday = now.weekday() == 4
    is_after_3pm = now.hour > 15 or (now.hour == 15 and now.minute >= 5)
    return is_friday and is_after_3pm


def _cleanup_incomplete_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理周线中未收盘的本周数据。
    若最新周线的时间窗口起始日位于本周（且今天非周五15:00后），则删除该条，
    后续更新时会从日线重新聚合出完整的本周周K。
    """
    if df.empty:
        return df
    latest = df.index.max()
    latest_week_start = latest.date() - pd.Timedelta(days=latest.weekday())
    today = datetime.today()
    today_week_start = today.date() - pd.Timedelta(days=today.weekday())
    if latest_week_start == today_week_start and today.weekday() < 4:
        df = df[df.index < latest]
    return df


def update_dataset(cache_df: pd.DataFrame, bar_type: str, bar_count: int):
    """
    增量更新数据集：从数据库读取最新时间戳，只拉取新增数据

    参数：
        bar_type: 'd'日线, 'w'周线, '60'60分钟
        bar_count: K 线数量上限
    """
    if bar_type == "w" and not _is_market_close_for_weekly():
        print(f"\n⏭️  周线更新跳过：仅在周五 15:00 后更新（当前 {datetime.now().strftime('%Y-%m-%d %H:%M')} 不满足条件）")
        return

    from datasource.database import query_df
    import pandas as pd

    db_freq_map = {'d': 'd', 'w': 'w', '60': '60m'}
    db_freq = db_freq_map.get(bar_type, bar_type)

    print(f"\n{'='*70}")
    print(f"🔄 增量更新 {bar_type} 数据 ({bar_count} bar)")
    print(f"{'='*70}\n")

    with get_session() as session:
        existing_df = query_df(session, "stock_k_data", filters={"freq": db_freq}, columns=["ts_code", "bar_time"])
        if not existing_df.empty:
            existing_df["bar_time"] = pd.to_datetime(existing_df["bar_time"])
            existing_dates = existing_df.groupby("ts_code")["bar_time"].max()
        else:
            existing_dates = pd.Series(dtype="datetime64[ns]")

    existing_codes = set(existing_dates.index)
    all_codes = set(cache_df['ts_code'].apply(lambda x: x.split('.')[0]))

    missing_codes = all_codes - existing_codes
    need_update_codes = existing_codes

    print(f"已有股票: {len(existing_codes)} 只")
    print(f"缺失股票: {len(missing_codes)} 只")
    print(f"需要检查更新的: {len(need_update_codes)} 只")

    api = connect_pytdx()
    data_cache: Dict[str, Dict] = {}

    def should_update(symbol, newest_time):
        if newest_time is None:
            return True
        today = pd.Timestamp.today().normalize()
        return newest_time < today

    try:
        for _, row in tqdm(cache_df.iterrows(), total=len(cache_df), desc=f"更新{bar_type}"):
            symbol = row["ts_code"].split(".")[0]
            name = row["name"]

            if symbol in missing_codes:
                newest_time = None
            elif symbol in need_update_codes:
                newest_time = existing_dates.get(symbol, None)
                if not should_update(symbol, newest_time):
                    continue
            else:
                continue

            try:
                if bar_type == "w":
                    df = get_kline_data(api, symbol, "d", bar_count)
                    if df.empty or len(df) < 100:
                        continue
                    df = df.set_index("datetime")
                    if newest_time:
                        df = df[df.index > newest_time]
                    if df.empty:
                        continue
                    df = resample_to_weekly(df)
                else:
                    period_map = {"60": "60m", "d": "d"}
                    pytdx_period = period_map.get(bar_type, bar_type)
                    df = get_kline_data(api, symbol, pytdx_period, bar_count)
                    if df.empty or len(df) < 100:
                        continue
                    df = df.set_index("datetime")
                    if newest_time:
                        df = df[df.index > newest_time]

                if df.empty or len(df) < 50:
                    continue

                data_cache[symbol] = {
                    "name": name,
                    "data": df,
                }

            except Exception:
                continue

    finally:
        api.disconnect()

    if data_cache:
        print(f"\n获取到 {len(data_cache)} 只股票的新数据，正在保存到数据库...")
        with get_session() as session:
            save_to_database(session, data_cache, bar_type)
        print(f"\n✅ 更新完成")
    else:
        print(f"\n⚠️  无新数据需要更新")


def main():
    parser = argparse.ArgumentParser(description="构建多周期数据集（保存到数据库）")
    parser.add_argument("--update", action="store_true", help="增量更新到当天")
    parser.add_argument("--limit", type=int, default=None, help="限制处理的股票数量（用于测试）")
    args = parser.parse_args()

    from datasource.database import query_df
    with get_session() as session:
        cache_df = query_df(session, "stock_pools", columns=["ts_code", "name"])
    print(f"从数据库 stock_pools 读取股票池: {len(cache_df)} 只")

    if args.limit:
        cache_df = cache_df.head(args.limit)
        print(f"⚠️  测试模式：仅处理 {len(cache_df)} 只股票")

    print(f"股票池总数: {len(cache_df)}")

    def do_daily():
        if args.update:
            update_dataset(cache_df, "d", 500)
        else:
            fetch_and_save_data(cache_df, "d", 500)

    def do_weekly():
        if args.update:
            update_dataset(cache_df, "w", 500)
        else:
            fetch_and_save_data(cache_df, "w", 500)

    def do_min60():
        if args.update:
            update_dataset(cache_df, "60", 2000)
        else:
            fetch_and_save_data(cache_df, "60", 2000)

    if args.update:
        print(f"\n{'='*70}")
        print(f"🔄 增量更新模式：将数据集更新到当天")
        print(f"{'='*70}\n")
        do_daily()
        do_weekly()
        do_min60()
        print(f"\n{'='*70}")
        print(f"✅ 增量更新完成")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"📊 构建多周期数据集（保存到数据库）")
        print(f"{'='*70}\n")
        do_daily()
        do_weekly()
        do_min60()
        print(f"\n{'='*70}")
        print(f"✅ 数据集构建完成")
        print(f"保存位置: PostgreSQL 远程数据库 (config.DATABASE_URL)")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
