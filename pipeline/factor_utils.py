#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算公共工具模块

Purpose: 统一 daily_factor_update.py 和 backfill_factors.py 的取数口径、
         coverage 规则、因子/事件计算逻辑
Inputs: stock_k_data (DB), factor_definition (DB)
Outputs: K线 DataFrame, 因子 DataFrame, 事件 DataFrame
How to Run:
    from pipeline.factor_utils import get_stock_bars, check_coverage, compute_factors_for_stock
Examples:
    bars = get_stock_bars(conn, '000001.SZ', date(2024,3,15), 'd')
    is_ok, cov, status = check_coverage(bars, 120)
Side Effects: 无（只读操作）
"""
import sys
import os
from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_lib import compute_panel
from event_lib import detect_panel

# 缓存 lookback，避免重复查询
_LOOKBACK_CACHE: dict = {}


def get_recommended_lookback(conn, freq: str) -> int:
    """从 factor_definition 查询该频率下所有因子的最大 recommended_lookback_bars"""
    cache_key = freq
    if cache_key in _LOOKBACK_CACHE:
        return _LOOKBACK_CACHE[cache_key]

    sql = text("""
        SELECT COALESCE(MAX(recommended_lookback_bars), 120)
        FROM factor_definition
        WHERE is_active = TRUE
    """)
    result = conn.execute(sql)
    max_bars = result.scalar() or 120
    # 额外加 10 bars 缓冲
    max_bars = int(max_bars) + 10
    _LOOKBACK_CACHE[cache_key] = max_bars
    return max_bars


def get_query_start(end_date: date, freq: str, lookback_bars: int) -> date:
    """统一计算查询起始日期

    Args:
        end_date: 查询结束日期
        freq: 频率 '1d' 或 '1w'
        lookback_bars: 需要的 K 线根数

    Returns:
        query_start: 查询起始日期
    """
    if freq == "1w":
        # 周线：lookback_bars 周 ≈ lookback_bars * 7 天，再加 60 天缓冲
        delta_days = lookback_bars * 7 + 60
    else:
        # 日线：lookback_bars 天，再加 60 天缓冲（应对停牌/节假日/新股）
        delta_days = lookback_bars + 60
    return end_date - pd.Timedelta(days=delta_days)


def get_stock_bars(conn, ts_code: str, end_date: date, freq_db: str = "d") -> pd.DataFrame:
    """统一获取 K 线（日更和回补共用）

    Args:
        conn: 数据库连接
        ts_code: 股票代码
        end_date: 结束日期（日更=target_date，回补=end_date）
        freq_db: 数据库 freq 列值 ('d' 或 'w')

    Returns:
        DataFrame: index=bar_time, columns=open/high/low/close/volume
    """
    freq = "1w" if freq_db == "w" else "1d"
    lookback_bars = get_recommended_lookback(conn, freq)
    query_start = get_query_start(end_date, freq, lookback_bars)

    sql = text("""
        SELECT bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = :freq
          AND bar_time >= :start AND bar_time <= :end
        ORDER BY bar_time
    """)
    df = pd.read_sql(sql, conn, params={
        "ts_code": ts_code,
        "freq": freq_db,
        "start": query_start.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
    })
    if df.empty:
        return df
    df["bar_time"] = pd.to_datetime(df["bar_time"])
    df = df.set_index("bar_time")
    df.columns = [c.lower() for c in df.columns]
    return df


def check_coverage(bars: pd.DataFrame, required_bars: int) -> Tuple[bool, float, str]:
    """统一 coverage 检查

    Args:
        bars: K 线 DataFrame
        required_bars: 需要的 K 线根数

    Returns:
        (is_ok, coverage_ratio, coverage_status)
        is_ok: 是否满足最低 coverage 要求
        coverage_ratio: 实际 bars /  required bars
        coverage_status: 'sufficient' / 'marginal' / 'insufficient'
    """
    if bars.empty:
        return False, 0.0, "insufficient"

    coverage = len(bars) / required_bars if required_bars > 0 else 1.0

    if coverage >= 1.0:
        return True, coverage, "sufficient"
    elif coverage >= 0.95:
        return True, coverage, "marginal"
    else:
        return False, coverage, "insufficient"


def compute_factors_for_stock(df: pd.DataFrame) -> pd.DataFrame:
    """为单只股票计算所有因子"""
    if len(df) < 30:
        return pd.DataFrame()
    try:
        return compute_panel(df)
    except Exception as e:
        print(f"    因子计算失败: {e}")
        return pd.DataFrame()


def compute_events_for_stock(df: pd.DataFrame) -> pd.DataFrame:
    """为单只股票计算所有事件"""
    if len(df) < 30:
        return pd.DataFrame()
    try:
        return detect_panel(df)
    except Exception as e:
        print(f"    事件检测失败: {e}")
        return pd.DataFrame()
