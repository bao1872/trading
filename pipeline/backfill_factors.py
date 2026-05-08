#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[DEPRECATED] 因子/事件历史回补脚本 — 已停用，请勿在生产环境中使用

因子/事件不再写入数据库。实验请直接引用 factor_lib 批量计算函数：
    from factor_lib import compute_all_factors_v2
    from event_lib import detect_panel

Purpose: (历史遗留) 按股票遍历，一次性拉取全历史 K 线，计算全历史因子/事件，批量写入 DB
Inputs: stock_k_data (DB), factor_lib, event_lib
Outputs: factor_value (DB), event_trigger (DB)
How to Run: [已停用，仅供参考]
    python pipeline/backfill_factors.py --start 2024-01-01 --end 2026-04-30 --freq 1d
Examples: 不再提供
Side Effects: 写入/更新 factor_value 和 event_trigger 表（请勿实际执行）
"""
import argparse
import sys
import os
import warnings
import logging
from datetime import datetime, date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_engine, bulk_upsert
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

logger = logging.getLogger(__name__)

WRITE_BATCH_SIZE = 20000


def get_all_bars(conn, ts_code: str, start_date: date, end_date: date, freq_db: str = "d") -> pd.DataFrame:
    """获取单只股票在指定日期范围内的全部 K 线（兼容旧接口，实际调用公共 helper）

    Args:
        ts_code: 股票代码（带后缀，如 000001.SZ）
        start_date: 开始日期
        end_date: 结束日期
        freq_db: 数据库 freq 列值 ('d' 或 'w')

    Returns:
        DataFrame: index=bar_time, columns=open/high/low/close/volume
    """
    # 公共 helper 的 get_stock_bars 以 end_date 为基准计算 query_start
    # 回补场景需要确保 query_start <= start_date，所以传入 end_date
    df = get_stock_bars(conn, ts_code, end_date, freq_db)
    if df.empty:
        return df
    # 过滤到目标范围
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    return df.loc[mask]


def get_stock_list(conn, start_date: date, end_date: date, freq_db: str = "d") -> List[str]:
    """获取在指定日期范围内有数据的全部股票列表"""
    sql = text("""
        SELECT DISTINCT ts_code FROM stock_k_data
        WHERE freq = :freq AND bar_time >= :start AND bar_time <= :end
    """)
    result = conn.execute(sql, {
        "freq": freq_db,
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
    })
    return [row[0] for row in result.fetchall()]


def wide_to_long_factors(factors_df: pd.DataFrame, ts_code: str, freq: str, for_new_table: bool = False) -> pd.DataFrame:
    """将因子宽表（多日期）转为长表格式

    Args:
        factors_df: index=bar_time, columns=open/high/low/close/volume + 因子列
        ts_code: 股票代码
        freq: 频率 '1d' / '1w'
        for_new_table: 是否为新表格式（不含 freq 列）

    Returns:
        DataFrame: 长表格式，每行一条因子记录
    """
    if factors_df.empty:
        return pd.DataFrame()

    base_cols = {"open", "high", "low", "close", "volume"}
    factor_cols = [c for c in factors_df.columns if c not in base_cols]

    records = []
    for as_of_date, row in factors_df.iterrows():
        as_of_date_str = as_of_date.strftime("%Y-%m-%d") if hasattr(as_of_date, "strftime") else str(as_of_date)[:10]
        for col in factor_cols:
            val = row[col]
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
                "as_of_date": as_of_date_str,
                "factor_name": col,
                "factor_value": val,
                "factor_version": "v1",
                "source_table": "factor_lib",
            }
            if not for_new_table:
                record["freq"] = freq
            records.append(record)

    return pd.DataFrame(records)


def wide_to_long_events(events_df: pd.DataFrame, ts_code: str, freq: str, for_new_table: bool = False) -> pd.DataFrame:
    """将事件宽表（多日期）转为长表格式

    Args:
        events_df: index=bar_time, columns=open/high/low/close/volume + evt_* 列
        ts_code: 股票代码
        freq: 频率 '1d' / '1w'
        for_new_table: 是否为新表格式（不含 freq 列）

    Returns:
        DataFrame: 长表格式，每行一条事件记录
    """
    if events_df.empty:
        return pd.DataFrame()

    evt_cols = [c for c in events_df.columns if c.startswith("evt_")]

    records = []
    for as_of_date, row in events_df.iterrows():
        as_of_date_str = as_of_date.strftime("%Y-%m-%d") if hasattr(as_of_date, "strftime") else str(as_of_date)[:10]
        for col in evt_cols:
            val = row[col]
            triggered = bool(val) if not pd.isna(val) else False
            record = {
                "ts_code": ts_code,
                "as_of_date": as_of_date_str,
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
    """为指定股票创建分区（如果不存在）

    WARNING: 当前 per-stock LIST 分区已积累 4507 个，触发 out of shared memory。
    此函数已废弃，保留仅作兼容。新架构使用频率分表+日期分区，无需 per-stock 分区。
    """
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
        sql_create = text(f"""
            CREATE TABLE IF NOT EXISTS {partition_name}
            PARTITION OF factor_value
            FOR VALUES IN (:ts_code)
        """)
        try:
            conn.execute(sql_create, {"ts_code": ts_code})
            conn.commit()
        except Exception:
            conn.rollback()

    evt_partition_name = f"event_trigger_{ts_code.replace('.', '_')}"
    result_evt = conn.execute(sql_check, {"partition_name": evt_partition_name})
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


def record_batch_log(
    conn,
    batch_id: str,
    ts_code: str,
    as_of_date: date,
    freq: str,
    data_type: str,
    version: str,
    record_count: int,
    coverage_ratio: float = None,
    coverage_status: str = None,
):
    """记录回补批次日志，用于后续校验和断点续传"""
    sql = text("""
        INSERT INTO backfill_batch_log
        (batch_id, ts_code, as_of_date, freq, data_type, version, record_count, coverage_ratio, coverage_status)
        VALUES (:batch_id, :ts_code, :as_of_date, :freq, :data_type, :version, :record_count, :coverage_ratio, :coverage_status)
        ON CONFLICT (batch_id, ts_code, as_of_date, freq, data_type, version) DO UPDATE SET
            record_count = EXCLUDED.record_count,
            coverage_ratio = EXCLUDED.coverage_ratio,
            coverage_status = EXCLUDED.coverage_status,
            created_at = CURRENT_TIMESTAMP
    """)
    conn.execute(sql, {
        "batch_id": batch_id,
        "ts_code": ts_code,
        "as_of_date": as_of_date,
        "freq": freq,
        "data_type": data_type,
        "version": version,
        "record_count": record_count,
        "coverage_ratio": coverage_ratio,
        "coverage_status": coverage_status,
    })


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


def backfill_stock(
    conn,
    ts_code: str,
    start_date: date,
    end_date: date,
    freq: str,
    batch_id: str,
    skip_factors: bool = False,
    skip_events: bool = False,
    target_table: str = "old",
) -> tuple:
    """单只股票全历史回补

    Args:
        target_table: 写入目标表，'old'=旧表, 'new'=新表, 'both'=双写

    Returns:
        (factor_count, event_count)
    """
    freq_db_map = {"1d": "d", "1w": "w"}
    freq_db = freq_db_map.get(freq, "d")

    # 1. 统一从 factor_definition 取 recommended_lookback_bars
    lookback_bars = get_recommended_lookback(conn, freq)

    # 2. 创建分区（兼容旧架构，新架构下此操作无实际作用）
    if not skip_factors or not skip_events:
        create_partition_if_needed(conn, ts_code)

    # 3. 统一使用公共 helper 获取 K 线（以 end_date 为基准）
    bars = get_stock_bars(conn, ts_code, end_date, freq_db)
    if bars.empty:
        return 0, 0

    # 4. 统一 coverage 检查
    is_ok, coverage_ratio, coverage_status = check_coverage(bars, lookback_bars)
    if not is_ok:
        return 0, 0

    # 5. 过滤到目标日期范围（只保留 start_date ~ end_date 的数据用于输出）
    target_mask = (bars.index >= pd.Timestamp(start_date)) & (bars.index <= pd.Timestamp(end_date))
    if not target_mask.any():
        return 0, 0

    # 6. 一次性计算全历史因子
    factors_df = compute_factors_for_stock(bars)
    if factors_df.empty:
        return 0, 0

    # 7. 一次性检测全历史事件
    events_df = None
    if not skip_events:
        events_df = compute_events_for_stock(factors_df)

    # 8. 宽表转长表并写入
    total_factors = 0
    total_events = 0

    target_dates = bars.index[target_mask]

    if not skip_factors:
        factors_target = factors_df.loc[factors_df.index.isin(target_dates)]

        # 写入旧表
        if target_table in ["old", "both"]:
            factor_long_old = wide_to_long_factors(factors_target, ts_code, freq, for_new_table=False)
            if not factor_long_old.empty:
                for i in range(0, len(factor_long_old), WRITE_BATCH_SIZE):
                    batch = factor_long_old.iloc[i:i + WRITE_BATCH_SIZE]
                    bulk_upsert(conn, "factor_value", batch,
                                unique_keys=["ts_code", "as_of_date", "freq", "factor_name", "factor_version"],
                                auto_commit=True)
                    total_factors += len(batch)

        # 写入新表
        if target_table in ["new", "both"]:
            factor_long_new = wide_to_long_factors(factors_target, ts_code, freq, for_new_table=True)
            if not factor_long_new.empty:
                factor_table = get_factor_table(freq, use_new=True)
                for i in range(0, len(factor_long_new), WRITE_BATCH_SIZE):
                    batch = factor_long_new.iloc[i:i + WRITE_BATCH_SIZE]
                    bulk_upsert(conn, factor_table, batch,
                                unique_keys=["ts_code", "as_of_date", "factor_name", "factor_version"],
                                auto_commit=True)
                    total_factors += len(batch)

    if not skip_events and events_df is not None and not events_df.empty:
        events_target = events_df.loc[events_df.index.isin(target_dates)]

        # 写入旧表
        if target_table in ["old", "both"]:
            event_long_old = wide_to_long_events(events_target, ts_code, freq, for_new_table=False)
            if not event_long_old.empty:
                for i in range(0, len(event_long_old), WRITE_BATCH_SIZE):
                    batch = event_long_old.iloc[i:i + WRITE_BATCH_SIZE]
                    bulk_upsert(conn, "event_trigger", batch,
                                unique_keys=["ts_code", "as_of_date", "freq", "event_name", "event_version"],
                                auto_commit=True)
                    total_events += len(batch)

        # 写入新表
        if target_table in ["new", "both"]:
            event_long_new = wide_to_long_events(events_target, ts_code, freq, for_new_table=True)
            if not event_long_new.empty:
                event_table = get_event_table(freq, use_new=True)
                for i in range(0, len(event_long_new), WRITE_BATCH_SIZE):
                    batch = event_long_new.iloc[i:i + WRITE_BATCH_SIZE]
                    bulk_upsert(conn, event_table, batch,
                                unique_keys=["ts_code", "as_of_date", "event_name", "event_version"],
                                auto_commit=True)
                    total_events += len(batch)

    # 9. 记录 batch_log（按日期汇总，包含 coverage_ratio 和 coverage_status）
    factor_long = wide_to_long_factors(factors_target, ts_code, freq, for_new_table=False) if not skip_factors else pd.DataFrame()
    event_long = wide_to_long_events(events_target, ts_code, freq, for_new_table=False) if (not skip_events and events_df is not None and not events_df.empty) else pd.DataFrame()

    if total_factors > 0 and not factor_long.empty:
        for as_of_date in target_dates:
            date_str = as_of_date.strftime("%Y-%m-%d")
            day_count = len(factor_long[factor_long["as_of_date"] == date_str])
            if day_count > 0:
                record_batch_log(conn, batch_id, ts_code, date_str, freq, "factor", "v1", day_count, coverage_ratio, coverage_status)
    if total_events > 0 and not event_long.empty:
        for as_of_date in target_dates:
            date_str = as_of_date.strftime("%Y-%m-%d")
            day_count = len(event_long[event_long["as_of_date"] == date_str])
            if day_count > 0:
                record_batch_log(conn, batch_id, ts_code, date_str, freq, "event", "v1", day_count, coverage_ratio, coverage_status)

    return total_factors, total_events


def main():
    parser = argparse.ArgumentParser(description="因子/事件历史回补（按股票遍历）")
    parser.add_argument("--start", type=str, required=True, help="回补开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="回补结束日期 (YYYY-MM-DD)")
    parser.add_argument("--freq", type=str, default="1d", choices=["1d", "1w"], help="频率")
    parser.add_argument("--batch-size", type=int, default=20000, help="写入批次大小")
    parser.add_argument("--stock-list", type=str, help="指定股票列表，逗号分隔")
    parser.add_argument("--mode", type=str, default="all", choices=["factors", "events", "all"],
                        help="回补模式：factors=只写因子, events=只写事件, all=因子+事件（默认 all）")
    parser.add_argument("--batch-id", type=str, help="回补批次 ID（用于追踪和校验）")
    parser.add_argument("--target-table", type=str, default="new", choices=["old", "new", "both"],
                        help="写入目标表：old=旧表, new=新表, both=双写（默认 new）")
    args = parser.parse_args()

    # 根据 mode 设置 skip 标志
    skip_factors = args.mode == "events"
    skip_events = args.mode == "factors"

    global WRITE_BATCH_SIZE
    WRITE_BATCH_SIZE = args.batch_size

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    freq_db_map = {"1d": "d", "1w": "w"}
    freq_db = freq_db_map.get(args.freq, "d")

    # 生成批次 ID
    batch_id = args.batch_id or f"bf_{args.freq}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}"

    engine = get_engine()
    conn = engine.connect()

    try:
        # 获取股票列表
        if args.stock_list:
            stock_list = [s.strip() for s in args.stock_list.split(",")]
        else:
            print(f"获取股票列表 ({args.freq})...")
            stock_list = get_stock_list(conn, start_date, end_date, freq_db)

        print(f"=" * 60)
        print(f"回补模式: {start_date} ~ {end_date}, freq={args.freq}")
        print(f"批次 ID: {batch_id}")
        print(f"股票数量: {len(stock_list)}")
        print(f"=" * 60)

        total_factors = 0
        total_events = 0
        failed_stocks = []

        for ts_code in tqdm(stock_list, desc="回补进度"):
            try:
                n_f, n_e = backfill_stock(
                    conn, ts_code, start_date, end_date, args.freq,
                    batch_id=batch_id,
                    skip_factors=skip_factors, skip_events=skip_events,
                    target_table=args.target_table,
                )
                total_factors += n_f
                total_events += n_e
            except Exception as e:
                logger.error(f"[{ts_code}] 回补失败: {e}")
                failed_stocks.append(ts_code)

        print(f"\n[OK] 总计写入 {total_factors} 条因子, {total_events} 条事件")
        print(f"[OK] 批次 ID: {batch_id}")
        if failed_stocks:
            print(f"[WARN] {len(failed_stocks)} 只股票失败: {failed_stocks[:10]}...")

    finally:
        conn.close()
        engine.dispose()


if __name__ == "__main__":
    main()
