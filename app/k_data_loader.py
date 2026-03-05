# -*- coding: utf-8 -*-
"""
K线数据加载脚本 - 从数据库读取K线数据
支持多周期聚合（最小周期为5m）

python -m src.k_data_loader --code 600489.SH --start 2025-01-01 --end 2026-02-15 --freq 60m
"""
import os
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import DATABASE_URL
from app.db import get_session, query_sql
from app.logger import get_logger

logger = get_logger(__name__)


def load_k_data_from_db(ts_code: str,
                         freq: str = "5m",
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
    """
    从数据库加载K线数据
    
    Args:
        ts_code: 股票代码，如 "600489.SH"
        freq: 周期，支持 5m, 15m, 30m, 60m, d, w
        start_date: 开始日期，如 "2025-01-01"
        end_date: 结束日期，如 "2026-02-15"
    
    Returns:
        K线数据DataFrame，索引为datetime
    """
    sql = """
        SELECT bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = '5m'
    """
    params = {"ts_code": ts_code}
    
    if start_date:
        sql += " AND bar_time >= :start_date"
        params["start_date"] = f"{start_date} 00:00:00"
    if end_date:
        sql += " AND bar_time <= :end_date"
        params["end_date"] = f"{end_date} 23:59:59"
    
    sql += " ORDER BY bar_time"
    
    with get_session() as session:
        df = query_sql(session, sql, params)
    
    if df.empty:
        return pd.DataFrame()
    
    df["bar_time"] = pd.to_datetime(df["bar_time"])
    df = df.set_index("bar_time")
    
    freq_lower = freq.lower()
    if freq_lower not in ("5m", "5min"):
        df = resample_ohlc(df, freq)
    
    return df


def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """将K线数据重采样到更大周期"""
    if df.empty:
        return df
    
    freq_map = {
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "60min",
        "d": "D",
        "day": "D",
        "w": "W",
        "week": "W",
    }
    
    target_freq = freq_map.get(freq.lower(), freq)
    
    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    
    resampled = df.resample(target_freq).agg(ohlc_dict).dropna()
    return resampled


def get_available_dates(ts_code: str) -> List[str]:
    """获取股票可用的日期列表"""
    sql = """
        SELECT DISTINCT DATE(bar_time) as trade_date
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = '5m'
        ORDER BY trade_date
    """
    with get_session() as session:
        result = query_sql(session, sql, {"ts_code": ts_code})
    
    if result.empty:
        return []
    
    return [str(d) for d in result["trade_date"]]


def get_data_range(ts_code: str) -> Tuple[Optional[str], Optional[str]]:
    """获取股票数据的日期范围"""
    sql = """
        SELECT MIN(bar_time) as min_time, MAX(bar_time) as max_time
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = '5m'
    """
    with get_session() as session:
        result = query_sql(session, sql, {"ts_code": ts_code})
    
    if result.empty or result["min_time"].isna().all():
        return None, None
    
    min_time = pd.to_datetime(result["min_time"].iloc[0])
    max_time = pd.to_datetime(result["max_time"].iloc[0])
    
    return min_time.strftime("%Y-%m-%d"), max_time.strftime("%Y-%m-%d")


def list_available_stocks() -> List[str]:
    """列出所有有数据的股票"""
    sql = """
        SELECT DISTINCT ts_code
        FROM stock_k_data
        ORDER BY ts_code
    """
    with get_session() as session:
        result = query_sql(session, sql)
    
    if result.empty:
        return []
    
    return result["ts_code"].tolist()


def get_stock_bar_count(ts_code: str, freq: str = "5m") -> int:
    """获取股票K线数量"""
    sql = """
        SELECT COUNT(*) as cnt
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = :freq
    """
    with get_session() as session:
        result = query_sql(session, sql, {"ts_code": ts_code, "freq": freq})
    
    if result.empty:
        return 0
    return int(result["cnt"].iloc[0]) or 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="K线数据加载脚本")
    ap.add_argument("--code", type=str, default=None, help="股票代码，如 600489.SH")
    ap.add_argument("--freq", type=str, default="5m", help="周期: 5m, 15m, 30m, 60m, d, w")
    ap.add_argument("--start", type=str, default=None, help="开始日期，如 2025-01-01")
    ap.add_argument("--end", type=str, default=None, help="结束日期，如 2026-02-15")
    ap.add_argument("--list", action="store_true", help="列出所有可用股票")
    ap.add_argument("--range", action="store_true", help="显示数据日期范围")
    args = ap.parse_args()
    
    if args.list:
        stocks = list_available_stocks()
        print(f"可用股票数: {len(stocks)}")
        for s in stocks[:20]:
            print(f"  {s}")
        if len(stocks) > 20:
            print(f"  ... 还有 {len(stocks) - 20} 只")
        exit(0)
    
    if not args.code:
        print("[ERR] 请指定 --code 参数")
        exit(1)
    
    if args.range:
        start, end = get_data_range(args.code)
        if start:
            print(f"{args.code} 数据范围: {start} ~ {end}")
            count = get_stock_bar_count(args.code)
            print(f"  K线数量: {count}")
        else:
            print(f"{args.code} 无数据")
        exit(0)
    
    df = load_k_data_from_db(
        ts_code=args.code,
        freq=args.freq,
        start_date=args.start,
        end_date=args.end
    )
    
    if df.empty:
        print(f"[WARN] 无数据: {args.code}")
        exit(1)
    
    print(f"[OK] {args.code} {args.freq} 数据:")
    print(f"  日期范围: {df.index.min()} ~ {df.index.max()}")
    print(f"  记录数: {len(df)}")
    print()
    print(df.head(10))
    print("...")
    print(df.tail(5))
