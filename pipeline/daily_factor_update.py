#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[DEPRECATED] 每日因子/事件增量更新脚本 — 已停用，请勿在生产环境中使用

因子/事件不再每日写入数据库。实验请直接引用 factor_lib 批量计算函数：
    from factor_lib import compute_all_factors_v2
    from event_lib import detect_panel

Purpose: (历史遗留) 每日收盘后，从 stock_k_data 读取行情，计算所有因子和事件，写入 factor_value / event_trigger
Inputs: stock_k_data (DB), factor_lib, event_lib
Outputs: factor_value (DB), event_trigger (DB)
How to Run: [已停用，仅供参考]
    python pipeline/daily_factor_update.py --date 2024-01-15
Examples: 不再提供
Side Effects: 写入/更新 factor_value 和 event_trigger 表（请勿实际执行）
"""
import argparse
import sys
import os
import warnings
from datetime import datetime, date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_engine, table_exists, bulk_upsert
from factor_lib import compute_panel, FACTOR_REGISTRY
from event_lib import detect_panel, EVENT_REGISTRY
from pipeline.factor_utils import (
    get_recommended_lookback,
    get_stock_bars,
    check_coverage,
    compute_factors_for_stock,
    compute_events_for_stock,
)

warnings.filterwarnings("ignore")

# 写入批次大小
WRITE_BATCH_SIZE = 5000


def get_stock_list(conn, trade_date: date, freq_db: str = "d") -> List[str]:
    """获取指定日期的股票列表"""
    sql = text("""
        SELECT DISTINCT ts_code FROM stock_k_data
        WHERE freq = :freq AND bar_time = :trade_date
    """)
    result = conn.execute(sql, {"freq": freq_db, "trade_date": trade_date.strftime("%Y-%m-%d")})
    return [row[0] for row in result.fetchall()]


def factors_to_long_df(ts_code: str, as_of_date: date, freq: str, factors_df: pd.DataFrame,
                       coverage_ratio: float = None, coverage_status: str = None,
                       for_new_table: bool = False) -> pd.DataFrame:
    """将因子宽表转为长表格式

    Args:
        for_new_table: 是否为新表格式（不含 freq 列）
    """
    if factors_df.empty:
        return pd.DataFrame()

    # 只取最后一行（当前日期）
    last_row = factors_df.iloc[-1]
    base_cols = {"open", "high", "low", "close", "volume"}
    factor_cols = [c for c in factors_df.columns if c not in base_cols]

    records = []
    for col in factor_cols:
        val = last_row[col]
        # 处理非数值类型
        if pd.isna(val):
            val = None
        elif isinstance(val, (bool, np.bool_)):
            val = float(val)
        elif not np.isscalar(val):
            val = None
        else:
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = None

        record = {
            "ts_code": ts_code,
            "as_of_date": as_of_date,
            "factor_name": col,
            "factor_value": val,
            "factor_version": "v1",
            "source_table": "factor_lib",
        }
        if coverage_status is not None:
            record["coverage_status"] = coverage_status
        if not for_new_table:
            record["freq"] = freq
        records.append(record)

    return pd.DataFrame(records)


def events_to_long_df(ts_code: str, as_of_date: date, freq: str, events_df: pd.DataFrame,
                      for_new_table: bool = False) -> pd.DataFrame:
    """将事件宽表转为长表格式

    Args:
        for_new_table: 是否为新表格式（不含 freq 列）
    """
    if events_df.empty:
        return pd.DataFrame()

    last_row = events_df.iloc[-1]
    evt_cols = [c for c in events_df.columns if c.startswith("evt_")]

    records = []
    for col in evt_cols:
        val = last_row[col]
        triggered = bool(val) if not pd.isna(val) else False
        record = {
            "ts_code": ts_code,
            "as_of_date": as_of_date,
            "event_name": col,
            "triggered": triggered,
            "event_strength": float(val) if triggered and np.isscalar(val) else None,
            "event_direction": "up" if "up" in col else ("down" if "down" in col else "neutral"),
            "event_version": "v1",
        }
        if not for_new_table:
            record["freq"] = freq
        records.append(record)

    return pd.DataFrame(records)


def create_partition_if_needed(conn, ts_code: str):
    """为指定股票创建分区（如果不存在）"""
    # 分区名规范化：去掉 .SH/.SZ 后缀中的点
    partition_name = f"factor_value_{ts_code.replace('.', '_')}"
    sql_check = text("""
        SELECT EXISTS (
            SELECT FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = :partition_name AND c.relkind = 'r'
        )
    """)
    result = conn.execute(sql_check, {"partition_name": partition_name})
    exists = result.scalar()

    if not exists:
        # 创建分区
        sql_create = text(f"""
            CREATE TABLE IF NOT EXISTS {partition_name}
            PARTITION OF factor_value
            FOR VALUES IN (:ts_code)
        """)
        try:
            conn.execute(sql_create, {"ts_code": ts_code})
            conn.commit()
        except Exception as e:
            conn.rollback()
            # 分区可能已存在（并发）
            pass

    # 同样为 event_trigger 创建分区
    evt_partition_name = f"event_trigger_{ts_code.replace('.', '_')}"
    sql_check_evt = text("""
        SELECT EXISTS (
            SELECT FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = :partition_name AND c.relkind = 'r'
        )
    """)
    result_evt = conn.execute(sql_check_evt, {"partition_name": evt_partition_name})
    exists_evt = result_evt.scalar()

    if not exists_evt:
        sql_create_evt = text(f"""
            CREATE TABLE IF NOT EXISTS {evt_partition_name}
            PARTITION OF event_trigger
            FOR VALUES IN (:ts_code)
        """)
        try:
            conn.execute(sql_create_evt, {"ts_code": ts_code})
            conn.commit()
        except Exception:
            conn.rollback()
            pass


def get_factor_table(freq: str, use_new: bool = False) -> str:
    """根据频率获取因子表名"""
    if use_new:
        return f"factor_value_{freq}"
    return "factor_value"


def get_event_table(freq: str, use_new: bool = False) -> str:
    """根据频率获取事件表名"""
    if use_new:
        return f"event_trigger_{freq}"
    return "event_trigger"


def process_date(conn, trade_date: date, freq: str = "1d", stock_list: Optional[List[str]] = None,
                 skip_factors: bool = False, skip_events: bool = False, target_table: str = "old"):
    """处理单日的因子/事件计算和入库

    Args:
        freq: 频率 ("1d" 或 "1w")，映射到 stock_k_data.freq 列值 ("d" 或 "w")
        target_table: 写入目标表，'old'=旧表, 'new'=新表, 'both'=双写
    """
    freq_db_map = {"1d": "d", "1w": "w"}
    freq_db = freq_db_map.get(freq, "d")

    # 统一从 factor_definition 取 recommended_lookback_bars
    lookback_bars = get_recommended_lookback(conn, freq)

    if stock_list is None:
        stock_list = get_stock_list(conn, trade_date, freq_db)

    if not stock_list:
        print(f"  {trade_date} 无股票数据")
        return 0, 0

    all_factor_records_old = []
    all_event_records_old = []
    all_factor_records_new = []
    all_event_records_new = []

    for ts_code in tqdm(stock_list, desc=f"{trade_date} 计算因子/事件", leave=False):
        if not skip_factors or not skip_events:
            create_partition_if_needed(conn, ts_code)

        # 统一使用公共 helper 获取 K 线
        df = get_stock_bars(conn, ts_code, trade_date, freq_db)
        if df.empty:
            continue

        # 统一 coverage 检查
        is_ok, coverage_ratio, coverage_status = check_coverage(df, lookback_bars)
        if not is_ok:
            continue

        if not skip_factors:
            factors_df = compute_factors_for_stock(df)
            if not factors_df.empty:
                if target_table in ["old", "both"]:
                    factor_long = factors_to_long_df(ts_code, trade_date, freq, factors_df, coverage_ratio, coverage_status, for_new_table=False)
                    if not factor_long.empty:
                        all_factor_records_old.append(factor_long)

                if target_table in ["new", "both"]:
                    factor_long_new = factors_to_long_df(ts_code, trade_date, freq, factors_df, coverage_ratio, coverage_status, for_new_table=True)
                    if not factor_long_new.empty:
                        all_factor_records_new.append(factor_long_new)

                if not skip_events:
                    events_df = compute_events_for_stock(factors_df)
                    if not events_df.empty:
                        if target_table in ["old", "both"]:
                            event_long = events_to_long_df(ts_code, trade_date, freq, events_df, for_new_table=False)
                            if not event_long.empty:
                                all_event_records_old.append(event_long)

                        if target_table in ["new", "both"]:
                            event_long_new = events_to_long_df(ts_code, trade_date, freq, events_df, for_new_table=True)
                            if not event_long_new.empty:
                                all_event_records_new.append(event_long_new)
        elif not skip_events:
            factors_df = compute_factors_for_stock(df)
            if not factors_df.empty:
                events_df = compute_events_for_stock(factors_df)
                if not events_df.empty:
                    if target_table in ["old", "both"]:
                        event_long = events_to_long_df(ts_code, trade_date, freq, events_df, for_new_table=False)
                        if not event_long.empty:
                            all_event_records_old.append(event_long)

                    if target_table in ["new", "both"]:
                        event_long_new = events_to_long_df(ts_code, trade_date, freq, events_df, for_new_table=True)
                        if not event_long_new.empty:
                            all_event_records_new.append(event_long_new)

    # 批量写入
    total_factors = 0
    total_events = 0

    # 写入旧表
    if target_table in ["old", "both"]:
        if all_factor_records_old:
            factor_df = pd.concat(all_factor_records_old, ignore_index=True)
            for i in range(0, len(factor_df), WRITE_BATCH_SIZE):
                batch = factor_df.iloc[i:i + WRITE_BATCH_SIZE]
                bulk_upsert(conn, "factor_value", batch, unique_keys=["ts_code", "as_of_date", "freq", "factor_name", "factor_version"])
                total_factors += len(batch)

        if all_event_records_old:
            event_df = pd.concat(all_event_records_old, ignore_index=True)
            for i in range(0, len(event_df), WRITE_BATCH_SIZE):
                batch = event_df.iloc[i:i + WRITE_BATCH_SIZE]
                bulk_upsert(conn, "event_trigger", batch, unique_keys=["ts_code", "as_of_date", "freq", "event_name", "event_version"])
                total_events += len(batch)

    # 写入新表
    if target_table in ["new", "both"]:
        factor_table = get_factor_table(freq, use_new=True)
        event_table = get_event_table(freq, use_new=True)

        if all_factor_records_new:
            factor_df = pd.concat(all_factor_records_new, ignore_index=True)
            for i in range(0, len(factor_df), WRITE_BATCH_SIZE):
                batch = factor_df.iloc[i:i + WRITE_BATCH_SIZE]
                bulk_upsert(conn, factor_table, batch, unique_keys=["ts_code", "as_of_date", "factor_name", "factor_version"])
                total_factors += len(batch)

        if all_event_records_new:
            event_df = pd.concat(all_event_records_new, ignore_index=True)
            for i in range(0, len(event_df), WRITE_BATCH_SIZE):
                batch = event_df.iloc[i:i + WRITE_BATCH_SIZE]
                bulk_upsert(conn, event_table, batch, unique_keys=["ts_code", "as_of_date", "event_name", "event_version"])
                total_events += len(batch)

    return total_factors, total_events


def main():
    parser = argparse.ArgumentParser(description="每日因子/事件增量更新")
    parser.add_argument("--date", type=str, help="目标日期 (YYYY-MM-DD)")
    parser.add_argument("--freq", type=str, default="1d", choices=["1d", "1w"], help="频率")
    parser.add_argument("--backfill", action="store_true", help="回补模式")
    parser.add_argument("--start", type=str, help="回补开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="回补结束日期 (YYYY-MM-DD)")
    parser.add_argument("--batch-size", type=int, default=5000, help="写入批次大小")
    parser.add_argument("--stock-list", type=str, help="指定股票列表，逗号分隔")
    parser.add_argument("--skip-factors", action="store_true", help="跳过因子计算，只做事件检测")
    parser.add_argument("--skip-events", action="store_true", help="跳过事件检测，只计算因子")
    parser.add_argument("--factor-names", type=str, help="只计算指定因子，逗号分隔（用于选择性回补）")
    parser.add_argument("--target-table", type=str, default="new", choices=["old", "new", "both"],
                        help="写入目标表：old=旧表, new=新表, both=双写（默认 new）")
    args = parser.parse_args()

    global WRITE_BATCH_SIZE
    WRITE_BATCH_SIZE = args.batch_size

    if args.factor_names:
        from factor_lib import compute_panel
        factor_names = [n.strip() for n in args.factor_names.split(",")]
        print(f"选择性模式: 只计算 {factor_names}")
    else:
        factor_names = None

    engine = get_engine()
    conn = engine.connect()

    try:
        if args.backfill:
            start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
            end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
            dates = pd.date_range(start=start_date, end=end_date, freq="B")

            print(f"=" * 60)
            print(f"回补模式: {start_date} ~ {end_date} ({len(dates)} 个交易日)")
            print(f"=" * 60)

            for trade_date in tqdm(dates, desc="回补进度"):
                n_factors, n_events = process_date(
                    conn, trade_date.date(), args.freq,
                    skip_factors=args.skip_factors, skip_events=args.skip_events,
                    target_table=args.target_table,
                )
                tqdm.write(f"  {trade_date.date()}: {n_factors} 因子, {n_events} 事件")

        else:
            if args.date:
                target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
            else:
                target_date = date.today() - timedelta(days=1)

            stock_list = None
            if args.stock_list:
                stock_list = [s.strip() for s in args.stock_list.split(",")]

            print(f"=" * 60)
            print(f"单日更新: {target_date}, freq={args.freq}")
            print(f"=" * 60)

            n_factors, n_events = process_date(
                conn, target_date, args.freq, stock_list,
                skip_factors=args.skip_factors, skip_events=args.skip_events,
                target_table=args.target_table,
            )
            print(f"[OK] 写入 {n_factors} 条因子, {n_events} 条事件")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
