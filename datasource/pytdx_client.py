# -*- coding: utf-8 -*-
"""
数据获取模块 - 统一数据获取入口

目标：保证数据来源一致、可追溯、可维护；避免不同脚本各自拉数据导致口径不一致与重复代码。

所有行情数据必须通过本模块获取，禁止其他脚本直接使用 pytdx。

Usage:
    from datasource.pytdx_client import connect_pytdx, get_kline_data, get_stock_code_by_name
    
    # 连接服务器
    api = connect_pytdx()
    
    # 获取 K 线数据
    df = get_kline_data(api, '000001', 'd', 255)
    
    # 断开连接
    api.disconnect()
"""
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pytdx.errors import TdxConnectionError
from pytdx.hq import TdxHq_API


# pytdx 服务器列表
PYTDX_SERVERS = [
    ("119.147.212.81", 7709),
    ("119.147.164.60", 7709),
    ("14.215.128.18", 7709),
    ("14.215.128.116", 7709),
    ("101.133.156.38", 7709),
    ("114.80.149.19", 7709),
    ("115.238.90.165", 7709),
    ("123.125.108.23", 7709),
    ("180.153.18.170", 7709),
    ("202.108.253.131", 7709),
]

# 周期映射
PERIOD_MAP = {
    '1m': 8,
    '5m': 0,
    '15m': 1,
    '30m': 2,
    '60m': 3,
    'd': 4,
    'w': 5,
    'm': 6,
}


def market_from_code(code: str) -> int:
    """根据股票代码判断市场"""
    c = str(code)
    return 1 if c.startswith("6") else 0


def connect_pytdx() -> TdxHq_API:
    """连接 pytdx 服务器"""
    last_errors = []
    for host, port in PYTDX_SERVERS:
        try:
            api = TdxHq_API(raise_exception=True, auto_retry=True)
            if api.connect(host, port):
                print(f"✅ 连接成功：{host}:{port}")
                return api
        except TdxConnectionError as exc:
            last_errors.append(f"{host}:{port} {exc}")
        except Exception as exc:
            last_errors.append(f"{host}:{port} {exc}")
    err = "; ".join(last_errors[-5:])
    raise RuntimeError(f"pytdx 连接失败：{err}")


def get_stock_code_by_name(api: TdxHq_API, name: str, stock_cache_path: str = None) -> Optional[str]:
    """通过股票名称查找股票代码"""
    if stock_cache_path and os.path.exists(stock_cache_path):
        cache_df = pd.read_excel(stock_cache_path)
        if 'name' in cache_df.columns and 'ts_code' in cache_df.columns:
            match = cache_df[cache_df['name'] == name]
            if not match.empty:
                ts_code = match.iloc[0]['ts_code']
                code = ts_code.split('.')[0]
                return code
    
    for market in [1, 0]:
        for start in range(0, 1000, 100):
            data = api.get_security_list(market, start)
            if not data:
                break
            for item in data:
                if item['name'] == name:
                    return item['code']
    return None


def get_stock_name(api: TdxHq_API, symbol: str) -> str:
    """获取股票名称"""
    market = market_from_code(symbol)
    data = api.get_security_list(market, 0)
    if data:
        for item in data:
            if item['code'] == symbol:
                return item['name']
    return ""


def get_kline_data(api: TdxHq_API, symbol: str, period: str, count: int = 255) -> pd.DataFrame:
    """获取指定周期的 K 线数据"""
    if period not in PERIOD_MAP:
        raise ValueError(f"不支持的周期：{period}，支持的周期：{list(PERIOD_MAP.keys())}")
    
    market = market_from_code(symbol)
    cat = PERIOD_MAP[period]
    
    all_bars = []
    start = 0
    fetch_count = 800
    
    while len(all_bars) < count:
        data = api.get_security_bars(cat, market, symbol, start, fetch_count)
        if not data:
            break
        
        all_bars.extend(data)
        
        if len(data) < fetch_count:
            break
        
        start += fetch_count
    
    if not all_bars:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_bars)
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif {'year', 'month', 'day', 'hour', 'minute'}.issubset(df.columns):
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']].astype(int))
    
    df = df[['datetime', 'open', 'high', 'low', 'close', 'vol', 'amount']]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']
    
    df = df.sort_values('datetime', ascending=True).tail(count).reset_index(drop=True)
    
    return df


def get_multi_period_kline(api: TdxHq_API, symbol: str, periods: List[str] = None, count: int = 255) -> Dict[str, pd.DataFrame]:
    """获取多周期 K 线数据"""
    if periods is None:
        periods = ['1m', '5m', '15m', '60m', 'd', 'w']
    
    result = {}
    
    for period in periods:
        df = get_kline_data(api, symbol, period, count)
        
        if df.empty:
            print(f"  {period}: ❌ 无数据")
        else:
            print(f"  {period}: ✅ {len(df)} 条")
            result[period] = df
    
    return result


def get_daily_bars(api: TdxHq_API, symbol: str, days: int = 30) -> pd.DataFrame:
    """获取日线数据"""
    return get_kline_data(api, symbol, 'd', days)


def get_tick_data_for_date(api: TdxHq_API, symbol: str, date_int: int) -> Optional[Dict]:
    """获取指定日期的 tick 数据并汇总"""
    market = market_from_code(symbol)
    try:
        all_ticks = []
        start = 0
        count = 2000
        
        while True:
            data = api.get_history_transaction_data(market, symbol, start, count, date_int)
            
            if not data:
                break
            
            all_ticks.extend(data)
            
            if len(data) < count:
                break
            
            start += count
            
            if start > 100000:
                break
        
        if not all_ticks:
            return None
        
        df = pd.DataFrame(all_ticks)
        
        buy_df = df[df['buyorsell'] == 0]
        sell_df = df[df['buyorsell'] == 1]
        
        buy_volume = buy_df['vol'].sum() * 100
        sell_volume = sell_df['vol'].sum() * 100
        
        buy_amount = (buy_df['price'] * buy_df['vol'] * 100).sum()
        sell_amount = (sell_df['price'] * sell_df['vol'] * 100).sum()
        
        return {
            'buy_trades': len(buy_df),
            'sell_trades': len(sell_df),
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_amount': buy_amount,
            'sell_amount': sell_amount,
            'total_trades': len(df),
        }
    except Exception:
        return None


def get_raw_tick_data(api: TdxHq_API, symbol: str, date_int: int, page_size: int = 2000) -> pd.DataFrame:
    """
    获取指定日期的原始逐笔成交数据（DataFrame格式）
    
    Args:
        api: TdxHq_API 实例
        symbol: 股票代码
        date_int: 日期整数，如 20260312
        page_size: 每次请求数量，默认2000
    
    Returns:
        DataFrame，包含原始逐笔数据
    """
    market = market_from_code(symbol)
    all_ticks = []
    start = 0
    
    while True:
        data = api.get_history_transaction_data(market, symbol, start, page_size, date_int)
        if not data:
            break
        all_ticks.extend(data)
        if len(data) < page_size:
            break
        start += page_size
        if start > 50000:
            break
    
    if not all_ticks:
        return pd.DataFrame()
    
    return pd.DataFrame(all_ticks).drop_duplicates()


if __name__ == "__main__":
    # 自测入口
    api = connect_pytdx()
    try:
        # 测试获取股票名称
        symbol = "000001"
        name = get_stock_name(api, symbol)
        print(f"股票名称：{name}")
        
        # 测试获取 K 线数据
        df = get_kline_data(api, symbol, 'd', 10)
        print(f"\n获取 K 线数据：{len(df)} 条")
        print(df[['datetime', 'close']].tail())
        
        # 测试获取多周期数据
        print("\n获取多周期数据:")
        multi_df = get_multi_period_kline(api, symbol, ['d', 'w'], 5)
        for period, data in multi_df.items():
            print(f"  {period}: {len(data)} 条")
        
    finally:
        api.disconnect()
