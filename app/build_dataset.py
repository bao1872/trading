# -*- coding: utf-8 -*-
"""构建多周期数据集

用途：
    获取股票池全量股票的多周期行情数据并保存到数据库

数据规格（在 PERIOD_CONFIG 中配置）：
    - 日线：1300 bar
    - 周线：500 bar

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

    # 测试模式（只处理前10只股票）
    python app/build_dataset.py --period d --limit 10

周线更新逻辑：
    - 每天都可以运行更新
    - 更新时先删除该股票本周开始日期（周一）之前的数据
    - 然后插入本周及之后的最新数据
    - 确保周线数据始终最新且不会重复

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

# 确保项目根目录在 Python 路径中（支持 systemd 服务运行）
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# 切换到项目根目录（确保相对路径正确）
os.chdir(base_dir)

from datasource.pytdx_client import connect_pytdx, get_kline_data
from datasource.database import get_session, bulk_upsert

# 各周期数据配置
PERIOD_CONFIG = {
    "d": {"bars": 1300, "desc": "日线"},
    "w": {"bars": 500, "desc": "周线"},
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


def save_to_database(session, data_cache: Dict[str, Dict], freq: str, delete_before_week_start: bool = False):
    """
    将数据缓存保存到数据库

    Args:
        session: 数据库会话
        data_cache: 数据字典 {code: {'name': xxx, 'data': DataFrame}}
        freq: 周期 ('d', 'w', '60m')
        delete_before_week_start: 是否删除本周开始日期之前的数据（仅对周线有效）
    """
    from datetime import timedelta
    from sqlalchemy import text

    db_freq_map = {'d': 'd', 'w': 'w', '60': '60m'}
    db_freq = db_freq_map.get(freq, freq)

    # 周线更新：先删除本周的数据（用于重新插入本周最新数据）
    if delete_before_week_start and db_freq == 'w':
        today = pd.Timestamp.now().normalize()
        week_start = today - timedelta(days=today.weekday())  # 本周一
        week_end = week_start + timedelta(days=7)  # 下周一
        week_start_str = week_start.strftime("%Y-%m-%d")
        week_end_str = week_end.strftime("%Y-%m-%d")

        for code in data_cache.keys():
            ts_code = _code_to_symbol(code)
            try:
                # 只删除本周的数据（本周一到下周一之前）
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
        df_for_db["bar_time"] = pd.to_datetime(df_for_db["bar_time"])

        # 日线和周线只保存日期（不带时间）
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
    - 获取最近30根日线数据
    - 只保留比数据库最新的数据
    - 使用 upsert 保存

    周线更新逻辑（每天可运行）：
    - 获取最近20根周线数据
    - 只保留本周开始日期（周一）及之后的数据
    - 先删除该股票本周之前的数据，再插入新数据
    - 确保周线数据始终最新

    参数：
        bar_type: 'd'日线, 'w'周线
        bar_count: K 线数量上限（仅用于首次获取）
        force: 是否强制更新，忽略时间限制
    """
    from datasource.database import query_df
    from sqlalchemy import text
    import pandas as pd

    db_freq_map = {'d': 'd', 'w': 'w', '60': '60m'}
    db_freq = db_freq_map.get(bar_type, bar_type)

    print(f"\n{'='*70}")
    print(f"🔄 增量更新 {bar_type} 数据")
    print(f"{'='*70}\n")

    api = connect_pytdx()
    data_cache: Dict[str, Dict] = {}

    try:
        for _, row in tqdm(cache_df.iterrows(), total=len(cache_df), desc=f"更新{bar_type}"):
            symbol = row["ts_code"].split(".")[0]
            name = row["name"]

            try:
                # 查询该股票最后的数据日期
                with get_session() as session:
                    result = session.execute(
                        text("SELECT MAX(bar_time) as max_time FROM stock_k_data WHERE ts_code = :ts_code AND freq = :freq"),
                        {"ts_code": row["ts_code"], "freq": db_freq}
                    ).fetchone()
                    stock_max_time = result[0] if result and result[0] else None
                    if stock_max_time:
                        stock_max_time = pd.to_datetime(stock_max_time).tz_localize(None).normalize()

                if bar_type == "w":
                    # 周线：获取最近20根数据，保留本周及之后的数据
                    df = get_kline_data(api, symbol, "w", 20)
                    if df.empty:
                        continue
                    df = df.set_index("datetime")

                    # 计算本周开始日期（周一）
                    from datetime import timedelta
                    today = pd.Timestamp.now().normalize()
                    week_start = today - timedelta(days=today.weekday())  # 本周一

                    # 只保留本周及之后的数据（删除本周之前的数据）
                    df = df[df.index >= week_start]

                    if df.empty:
                        continue

                elif bar_type == "d":
                    # 日线：获取最近30根
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

    if data_cache:
        total_new_bars = sum(len(d["data"]) for d in data_cache.values())
        print(f"\n获取到 {len(data_cache)} 只股票的新数据，共 {total_new_bars} 条K线")
        print(f"正在保存到数据库...")
        with get_session() as session:
            # 周线更新时，先删除本周之前的数据
            delete_before_week = (bar_type == "w")
            save_to_database(session, data_cache, bar_type, delete_before_week_start=delete_before_week)
        print(f"\n✅ 更新完成")
    else:
        print(f"\n⚠️  无新数据需要更新")


def main():
    parser = argparse.ArgumentParser(description="构建多周期数据集（保存到数据库）")
    parser.add_argument("--update", action="store_true", help="增量更新到当天")
    parser.add_argument("--force", action="store_true", help="强制更新，忽略时间限制（如周五15:00限制）")
    parser.add_argument("--limit", type=int, default=None, help="限制处理的股票数量（用于测试）")
    parser.add_argument("--period", type=str, choices=["d", "w"], default=None,
                        help="只回补指定周期: d=日线, w=周线")
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
        cfg = PERIOD_CONFIG[args.period]
        print(f"\n{'='*70}")
        print(f"📊 单独回补 {cfg['desc']} 数据 ({cfg['bars']} bar)")
        print(f"{'='*70}\n")

        if args.period == "d":
            do_daily()
        elif args.period == "w":
            do_weekly()

        print(f"\n{'='*70}")
        print(f"✅ {cfg['desc']} 回补完成")
        print(f"{'='*70}\n")
        return

    # 全量回补所有周期
    if args.update:
        # 增量更新模式：必须指定 --period 参数，日线和周线分开更新
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
        else:
            print("\n❌ 错误：--update 模式必须指定 --period 参数（d=日线 或 w=周线）")
            print("示例：")
            print("  python app/build_dataset.py --update --period d  # 更新日线")
            print("  python app/build_dataset.py --update --period w  # 更新周线")
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
