# -*- coding: utf-8 -*-
"""构建多周期数据集

用途：
    获取股票池全量股票的多周期行情数据并保存到数据库

数据规格（在 PERIOD_CONFIG 中配置）：
    - 日线：1300 bar
    - 周线：500 bar
    - 15分钟线：1500 bar（仅自选股）

保存路径：
    PostgreSQL 远程数据库 (config.DATABASE_URL)

运行命令：
    # 构建所有周期数据（首次运行）
    python app/build_dataset.py

    # 增量更新日线数据到当天
    python app/build_dataset.py --update --period d

    # 增量更新周线数据（每天可运行，自动删除本周之前的数据）
    python app/build_dataset.py --update --period w

    # 只回补日线数据（1300 bar）
    python app/build_dataset.py --period d

    # 只回补周线数据（500 bar）
    python app/build_dataset.py --period w

    # 回补自选股15分钟K线数据（1500 bar）
    python app/build_dataset.py --period 15m

    # 增量更新自选股15分钟K线
    python app/build_dataset.py --update --period 15m

    # 刷新自选股Tick缓存（汇总+PVDI因子）
    python app/build_dataset.py --period tick

    # 测试模式（只处理前10只股票）
    python app/build_dataset.py --period d --limit 10

周线更新逻辑：
    - 每天都可以运行更新
    - 更新时先删除该股票本周的数据（本周一到下周一之前）
    - 然后从pytdx获取最新周线数据，只保留本周及之后的数据
    - 插入最新的本周数据，确保周线数据始终最新且不会重复

15分钟线更新逻辑：
    - 仅处理自选股（stock_watchlist 表）
    - 增量更新：查询每只股票最新bar_time，拉取缺失部分
    - 首次回补：拉取1500根15分钟K线

Tick缓存更新逻辑：
    - 仅处理自选股（stock_watchlist 表）
    - 刷新最近30个交易日的tick汇总+PVDI因子到tick_cache表
    - 复用 selection_tick.py 的 refresh_tick_cache() 逻辑

注意：
    这是一个耗时操作，只需要运行一次
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm

# 确保项目根目录在 Python 路径最前面（确保导入根目录 config.py）
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir in sys.path:
    sys.path.remove(base_dir)
sys.path.insert(0, base_dir)

# 切换到项目根目录（确保相对路径正确）
os.chdir(base_dir)

# 先导入根目录的 config，确保后续导入使用正确的配置
import config

from datasource.pytdx_client import connect_pytdx, get_kline_data
from datasource.database import get_session, bulk_upsert

# 各周期数据配置
PERIOD_CONFIG = {
    "d": {"bars": 1300, "desc": "日线"},
    "w": {"bars": 500, "desc": "周线"},
    "15m": {"bars": 1500, "desc": "15分钟线"},
}


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


def save_to_database(session, data_cache: Dict[str, Dict], freq: str,
                     delete_before_week_start: bool = False,
                     confirmed_codes: Optional[set] = None,
                     quiet: bool = False):
    """
    将数据缓存保存到数据库

    Args:
        session: 数据库会话
        data_cache: 数据字典 {code: {'name': xxx, 'data': DataFrame}}
        freq: 周期 ('d', 'w', '60m')
        delete_before_week_start: 是否删除本周的数据（本周一到下周一之前，仅对周线有效）
        confirmed_codes: 确认有数据的 code 集合（None 表示遍历 data_cache 内部判断）
        quiet: 为 True 时禁用 tqdm 进度条（Streamlit 环境需设为 True）
    """
    from datetime import timedelta
    from sqlalchemy import text

    db_freq_map = {'d': 'd', 'w': 'w', '60': '60m'}
    db_freq = db_freq_map.get(freq, freq)

    # 先收集有数据的 code，避免 DELETE 后 INSERT 失败导致数据丢失
    codes_with_data = confirmed_codes
    if codes_with_data is None:
        codes_with_data = set()
        for code, stock_data in data_cache.items():
            df = stock_data.get("data")
            if df is not None and not df.empty:
                codes_with_data.add(code)

    # 周线更新：只对确认有数据的股票删除本周数据
    if delete_before_week_start and db_freq == 'w' and codes_with_data:
        today = pd.Timestamp.now().normalize()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=7)
        week_start_str = week_start.strftime("%Y-%m-%d")
        week_end_str = week_end.strftime("%Y-%m-%d")

        for code in codes_with_data:
            ts_code = _code_to_symbol(code)
            try:
                delete_sql = text("""
                    DELETE FROM stock_k_data 
                    WHERE ts_code = :ts_code AND freq = 'w' 
                    AND bar_time >= :week_start AND bar_time < :week_end
                """)
                result = session.execute(delete_sql, {
                    "ts_code": ts_code,
                    "week_start": week_start_str,
                    "week_end": week_end_str
                })
                if result.rowcount > 0:
                    print(f"    删除 {ts_code} 本周数据: {result.rowcount} 条")
                session.commit()
            except Exception as e:
                print(f"    删除 {ts_code} 本周数据失败: {e}")
                session.rollback()

    all_dfs = []
    for code, stock_data in tqdm(data_cache.items(), desc=f"整理{freq}", disable=quiet):
        if code not in codes_with_data:
            continue

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

        if db_freq in ('d', 'w'):
            df_for_db["bar_time"] = df_for_db["bar_time"].dt.strftime("%Y-%m-%d")
        else:
            df_for_db["bar_time"] = df_for_db["bar_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

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


def update_dataset(cache_df: pd.DataFrame, bar_type: str, bar_count: int, force: bool = False):
    """
    增量更新数据集：拉取从数据库最新日期到今天的所有缺失数据

    日线更新逻辑：
    - 批量查询所有股票的最新 bar_time
    - 获取最近30根日线数据
    - 只保留比数据库最新的数据
    - 使用 upsert 保存

    周线更新逻辑（每天可运行）：
    - 批量查询所有股票的最新 bar_time
    - 获取最近20根周线数据
    - 只保留本周开始日期（周一）及之后的数据
    - 先删除该股票本周的数据（本周一到下周一之前）
    - 插入从pytdx获取的最新本周数据
    - 确保周线数据始终最新

    参数：
        bar_type: 'd'日线, 'w'周线
        bar_count: K 线数量上限（仅用于首次获取）
        force: 是否强制更新，忽略时间限制
    """
    from datasource.database import query_df
    from sqlalchemy import text
    import pandas as pd
    from datetime import timedelta

    db_freq_map = {'d': 'd', 'w': 'w', '60': '60m'}
    db_freq = db_freq_map.get(bar_type, bar_type)
    freq_label = "日线" if bar_type == "d" else "周线"

    print(f"\n{'='*70}")
    print(f"🔄 增量更新 {bar_type} 数据")
    print(f"{'='*70}\n")

    # 批量查询所有股票的最新 bar_time（一次查询替代 N 次）
    with get_session() as session:
        results = session.execute(
            text("SELECT ts_code, MAX(bar_time) as max_time FROM stock_k_data WHERE freq = :freq GROUP BY ts_code"),
            {"freq": db_freq}
        ).fetchall()
    max_time_map = {}
    for r in results:
        if r[1]:
            code = r[0].split(".")[0]
            max_time_map[code] = pd.to_datetime(r[1]).normalize()

    api = connect_pytdx()
    data_cache: Dict[str, Dict] = {}

    try:
        for _, row in tqdm(cache_df.iterrows(), total=len(cache_df), desc=f"更新{bar_type}"):
            symbol = row["ts_code"].split(".")[0]
            name = row["name"]
            stock_max_time = max_time_map.get(symbol)

            try:
                if bar_type == "w":
                    df = get_kline_data(api, symbol, "w", 20)
                    if df.empty:
                        continue
                    df = df.set_index("datetime")

                    today = pd.Timestamp.now().normalize()
                    week_start = today - timedelta(days=today.weekday())

                    df = df[df.index >= week_start]

                    if df.empty:
                        continue

                elif bar_type == "d":
                    df = get_kline_data(api, symbol, "d", 30)
                    if df.empty:
                        continue
                    df = df.set_index("datetime")

                    if stock_max_time:
                        df = df[df.index > stock_max_time]

                    if df.empty:
                        continue

                data_cache[symbol] = {
                    "name": name,
                    "data": df,
                }

            except Exception:
                continue

    finally:
        api.disconnect()

    total_stocks = len(cache_df)
    success_count = len(data_cache)
    success_rate = success_count / max(total_stocks, 1)

    if data_cache:
        total_new_bars = sum(len(d["data"]) for d in data_cache.values())
        print(f"\n获取到 {success_count} 只股票的新数据，共 {total_new_bars} 条K线")
        print(f"正在保存到数据库...")
        with get_session() as session:
            delete_before_week = (bar_type == "w")
            save_to_database(session, data_cache, bar_type, delete_before_week_start=delete_before_week)
        print(f"\n✅ 更新完成")
    else:
        print(f"\n⚠️  无新数据需要更新")

    if success_rate < 0.8:
        print(f"\n⚠️  WARNING: {freq_label}更新成功率过低: {success_rate:.1%} ({success_count}/{total_stocks})")
    else:
        print(f"{freq_label}更新成功率: {success_rate:.1%} ({success_count}/{total_stocks})")


def _load_watchlist_df() -> pd.DataFrame:
    """从 stock_watchlist 表加载自选股列表

    Returns:
        DataFrame，列：ts_code, stock_name
    """
    from datasource.database import query_df
    with get_session() as session:
        wl_df = query_df(session, "stock_watchlist", columns=["ts_code", "stock_name"])
    if wl_df.empty:
        print("⚠️  自选股列表为空")
    return wl_df


def update_watchlist_15m(watchlist_df: pd.DataFrame = None, bar_count: int = 1500, quiet: bool = False):
    """增量更新自选股15分钟K线数据

    从 stock_k_data 表查询每只股票 freq='15m' 的最新 bar_time，
    增量拉取缺失数据并入库。

    Args:
        watchlist_df: 自选股 DataFrame（列：ts_code, stock_name）。
                      为 None 时自动从数据库加载。
        bar_count: 首次回补时拉取的K线数量
        quiet: 为 True 时禁用 tqdm 进度条（Streamlit 环境需设为 True）
    """
    from sqlalchemy import text

    if watchlist_df is None:
        watchlist_df = _load_watchlist_df()
    if watchlist_df.empty:
        return

    print(f"\n{'='*70}")
    print(f"🔄 增量更新自选股15分钟K线数据")
    print(f"{'='*70}")
    print(f"自选股: {len(watchlist_df)} 只\n")

    # 批量查询每只股票15分钟K线的最新bar_time
    with get_session() as session:
        results = session.execute(
            text("SELECT ts_code, MAX(bar_time) as max_time "
                 "FROM stock_k_data WHERE freq = '15m' GROUP BY ts_code"),
        ).fetchall()
    max_time_map = {}
    for r in results:
        if r[1]:
            code = r[0].split(".")[0]
            max_time_map[code] = pd.to_datetime(r[1])

    api = connect_pytdx()
    data_cache: Dict[str, Dict] = {}

    try:
        for _, row in tqdm(watchlist_df.iterrows(), total=len(watchlist_df), desc="更新15m", disable=quiet):
            symbol = row["ts_code"].split(".")[0]
            name = row.get("stock_name", "")
            stock_max_time = max_time_map.get(symbol)

            try:
                if stock_max_time:
                    # 增量：只拉取最新30根，过滤已有数据
                    df = get_kline_data(api, symbol, "15m", 30)
                    if df.empty:
                        continue
                    df = df.set_index("datetime")
                    df = df[df.index > stock_max_time]
                    if df.empty:
                        continue
                else:
                    # 首次：拉取全量
                    df = get_kline_data(api, symbol, "15m", bar_count)
                    if df.empty or len(df) < 10:
                        continue
                    df = df.set_index("datetime")

                data_cache[symbol] = {"name": name, "data": df}

            except Exception:
                continue

    finally:
        api.disconnect()

    if data_cache:
        total_new_bars = sum(len(d["data"]) for d in data_cache.values())
        print(f"\n获取到 {len(data_cache)} 只股票的新数据，共 {total_new_bars} 条K线")
        with get_session() as session:
            save_to_database(session, data_cache, "15m", quiet=quiet)
        print(f"✅ 15分钟K线更新完成")
    else:
        print(f"⚠️  无新数据需要更新")


def update_watchlist_tick(watchlist_df: pd.DataFrame = None, n_days: int = 30, quiet: bool = False):
    """刷新自选股Tick缓存（汇总+PVDI因子）

    复用 selection_tick.py 的 refresh_tick_cache() 逻辑，
    仅处理自选股的 tick_cache 数据。

    Args:
        watchlist_df: 自选股 DataFrame（列：ts_code, stock_name）。
                      为 None 时自动从数据库加载。
        n_days: 回补最近 n_days 个交易日的数据
        quiet: 为 True 时禁用 tqdm 进度条（Streamlit 环境需设为 True）
    """
    from datasource.pytdx_client import get_tick_data_for_dates, compute_pvdi_for_dates
    from selection.selection_tick import (
        ensure_cache_table_exists, get_cached_tick_data, cache_tick_data,
    )

    if watchlist_df is None:
        watchlist_df = _load_watchlist_df()
    if watchlist_df.empty:
        return

    print(f"\n{'='*70}")
    print(f"🔄 刷新自选股Tick缓存（最近{n_days}天）")
    print(f"{'='*70}")
    print(f"自选股: {len(watchlist_df)} 只\n")

    ensure_cache_table_exists()

    # 获取最近n_days个交易日的日期列表
    with get_session() as session:
        from sqlalchemy import text
        result = session.execute(text(
            "SELECT DISTINCT bar_time FROM stock_k_data "
            "WHERE freq = 'd' ORDER BY bar_time DESC LIMIT :limit"
        ), {"limit": n_days})
        trade_dates_df = pd.DataFrame(result.fetchall(), columns=["bar_time"])
    if trade_dates_df.empty:
        print("⚠️  无法获取交易日列表")
        return

    trade_dates_df["bar_time"] = pd.to_datetime(trade_dates_df["bar_time"])
    recent_dates = trade_dates_df.head(n_days)["bar_time"]
    date_ints = [int(d.strftime("%Y%m%d")) for d in recent_dates]

    if not date_ints:
        print("⚠️  无交易日数据")
        return

    api = connect_pytdx()
    success_count = 0

    try:
        for _, row in tqdm(watchlist_df.iterrows(), total=len(watchlist_df), desc="刷新Tick缓存", disable=quiet):
            ts_code = row["ts_code"]
            symbol = ts_code.split(".")[0]

            try:
                # 检查缓存缺失日期
                cached = get_cached_tick_data(ts_code)
                if not cached.empty:
                    cached_date_ints = set(
                        int(pd.Timestamp(d).strftime("%Y%m%d"))
                        for d in cached["trade_date"]
                    )
                else:
                    cached_date_ints = set()

                missing_date_ints = sorted(set(date_ints) - cached_date_ints)

                if not missing_date_ints:
                    continue

                # 拉取缺失日期的tick和PVDI数据
                tick_df = get_tick_data_for_dates(api, symbol, missing_date_ints)
                pvdi_df = compute_pvdi_for_dates(api, symbol, missing_date_ints)

                # 写入缓存
                saved = cache_tick_data(ts_code, tick_df, pvdi_df)
                if saved > 0:
                    success_count += 1

            except Exception as e:
                print(f"  {ts_code} Tick缓存刷新失败: {e}")
                continue

    finally:
        api.disconnect()

    print(f"\n✅ Tick缓存刷新完成: {success_count}/{len(watchlist_df)} 只股票更新")


def main():
    parser = argparse.ArgumentParser(description="构建多周期数据集（保存到数据库）")
    parser.add_argument("--update", action="store_true", help="增量更新到当天")
    parser.add_argument("--force", action="store_true", help="强制更新，忽略时间限制（如周五15:00限制）")
    parser.add_argument("--limit", type=int, default=None, help="限制处理的股票数量（用于测试）")
    parser.add_argument("--period", type=str, choices=["d", "w", "15m", "tick"], default=None,
                        help="只回补指定周期: d=日线, w=周线, 15m=15分钟线, tick=Tick缓存")
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
        cfg = PERIOD_CONFIG["d"]
        if args.update:
            update_dataset(cache_df, "d", cfg["bars"], force=args.force)
        else:
            fetch_and_save_data(cache_df, "d", cfg["bars"])

    def do_weekly():
        cfg = PERIOD_CONFIG["w"]
        if args.update:
            update_dataset(cache_df, "w", cfg["bars"], force=args.force)
        else:
            fetch_and_save_data(cache_df, "w", cfg["bars"])



    # 如果只回补指定周期
    if args.period:
        if args.period == "tick":
            # Tick缓存刷新（不走PERIOD_CONFIG）
            print(f"\n{'='*70}")
            print(f"📊 刷新自选股Tick缓存")
            print(f"{'='*70}\n")
            update_watchlist_tick()
            print(f"\n{'='*70}")
            print(f"✅ Tick缓存刷新完成")
            print(f"{'='*70}\n")
            return

        cfg = PERIOD_CONFIG[args.period]
        print(f"\n{'='*70}")
        print(f"📊 单独回补 {cfg['desc']} 数据 ({cfg['bars']} bar)")
        print(f"{'='*70}\n")

        if args.period == "d":
            do_daily()
        elif args.period == "w":
            do_weekly()
        elif args.period == "15m":
            update_watchlist_15m(bar_count=cfg["bars"])

        print(f"\n{'='*70}")
        print(f"✅ {cfg['desc']} 回补完成")
        print(f"{'='*70}\n")
        return

    # 全量回补所有周期
    if args.update:
        # 增量更新模式：必须指定 --period 参数
        if args.period == "d":
            print(f"\n{'='*70}")
            print(f"🔄 增量更新日线数据")
            print(f"{'='*70}\n")
            do_daily()
            print(f"\n{'='*70}")
            print(f"✅ 日线增量更新完成")
            print(f"{'='*70}\n")
        elif args.period == "w":
            print(f"\n{'='*70}")
            print(f"🔄 增量更新周线数据")
            print(f"{'='*70}\n")
            do_weekly()
            print(f"\n{'='*70}")
            print(f"✅ 周线增量更新完成")
            print(f"{'='*70}\n")
        elif args.period == "15m":
            update_watchlist_15m()
        elif args.period == "tick":
            update_watchlist_tick()
        else:
            print("\n❌ 错误：--update 模式必须指定 --period 参数")
            print("示例：")
            print("  python app/build_dataset.py --update --period d    # 更新日线")
            print("  python app/build_dataset.py --update --period w    # 更新周线")
            print("  python app/build_dataset.py --update --period 15m  # 更新15分钟线")
            print("  python app/build_dataset.py --period tick          # 刷新Tick缓存")
            return
    else:
        print(f"\n{'='*70}")
        print(f"📊 构建多周期数据集（保存到数据库）")
        print(f"{'='*70}\n")
        do_daily()
        do_weekly()
        print(f"\n{'='*70}")
        print(f"✅ 数据集构建完成")
        print(f"保存位置: PostgreSQL 远程数据库 (config.DATABASE_URL)")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
