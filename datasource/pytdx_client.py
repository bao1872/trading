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
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    elif {'year', 'month', 'day', 'hour', 'minute'}.issubset(df.columns):
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']].astype(int)).dt.tz_localize(None)
    
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


def get_tick_data_for_dates(api: TdxHq_API, symbol: str, date_ints: List[int]) -> pd.DataFrame:
    """批量获取多日tick汇总数据

    内部循环调用 get_tick_data_for_date，跳过无数据的日期。

    Args:
        api: TdxHq_API 实例
        symbol: 股票代码（如 '000001'）
        date_ints: 日期整数列表，如 [20260301, 20260302]

    Returns:
        DataFrame，列：date_int, buy_volume, sell_volume, buy_trades, sell_trades
    """
    records = []
    for date_int in sorted(date_ints):
        result = get_tick_data_for_date(api, symbol, date_int)
        if result is not None:
            records.append({
                'date_int': date_int,
                'buy_volume': result['buy_volume'],
                'sell_volume': result['sell_volume'],
                'buy_trades': result['buy_trades'],
                'sell_trades': result['sell_trades'],
            })
    if not records:
        return pd.DataFrame(columns=['date_int', 'buy_volume', 'sell_volume', 'buy_trades', 'sell_trades'])
    return pd.DataFrame(records)


def compute_pvdi_for_date(api: TdxHq_API, symbol: str, date_int: int) -> Optional[Dict]:
    """计算指定日期的PVDI（Price-Volume Distribution Imbalance）三因子

    基于原始逐笔tick数据，分别计算买入/卖出集合的成交量加权价格分布指标，
    构造三个子因子：F_center(重心偏移)、F_spread(离散度差异)、F_skew(偏度差异)。

    仅使用 buyorsell=0(主动买入) 和 1(主动卖出) 的数据，排除竞价数据(2/8)。

    Args:
        api: TdxHq_API 实例
        symbol: 股票代码（如 '002916'）
        date_int: 日期整数，如 20260603

    Returns:
        {'f_center': float, 'f_spread': float, 'f_skew': float} 或 None（无数据时）
    """
    import numpy as np

    df = get_raw_tick_data(api, symbol, date_int)
    if df.empty:
        return None

    # 过滤：仅保留连续竞价的买入(0)和卖出(1)
    df = df[df['buyorsell'].isin([0, 1])].copy()
    if df.empty:
        return None

    # vol 从手转为股数
    df['vol'] = df['vol'] * 100

    buy_df = df[df['buyorsell'] == 0]
    sell_df = df[df['buyorsell'] == 1]

    # 若买入或卖出集合为空，三个因子均返回0
    if buy_df.empty or sell_df.empty:
        return {'f_center': 0.0, 'f_spread': 0.0, 'f_skew': 0.0, 'skew_b': 0.0, 'skew_s': 0.0, 'pvdi_weighted': 0.0}

    # 窗口内价格范围
    p_max = df['price'].max()
    p_min = df['price'].min()
    price_range = p_max - p_min

    # --- 成交量加权平均价格（重心） ---
    p_bar_b = (buy_df['vol'] * buy_df['price']).sum() / buy_df['vol'].sum()
    p_bar_s = (sell_df['vol'] * sell_df['price']).sum() / sell_df['vol'].sum()

    # 整体VWAP和日中价
    vwap = (df['vol'] * df['price']).sum() / df['vol'].sum()
    p_mid = (p_max + p_min) / 2

    # --- 成交量加权价格标准差（离散度） ---
    var_b = (buy_df['vol'] * (buy_df['price'] - p_bar_b) ** 2).sum() / buy_df['vol'].sum()
    var_s = (sell_df['vol'] * (sell_df['price'] - p_bar_s) ** 2).sum() / sell_df['vol'].sum()
    sigma_b = np.sqrt(var_b)
    sigma_s = np.sqrt(var_s)

    # --- 成交量加权偏度 ---
    if sigma_b > 0:
        skew_b = (buy_df['vol'] * (buy_df['price'] - p_bar_b) ** 3).sum() / (buy_df['vol'].sum() * sigma_b ** 3)
    else:
        skew_b = 0.0

    if sigma_s > 0:
        skew_s = (sell_df['vol'] * (sell_df['price'] - p_bar_s) ** 3).sum() / (sell_df['vol'].sum() * sigma_s ** 3)
    else:
        skew_s = 0.0

    # --- 构造三个子因子 ---
    # F_center: VWAP偏离日中价，反映成交量在价格区间内的真实分布方向
    # 旧公式 (p̄_B - p̄_S)/(p_max-p_min) 因bid-ask价差存在结构性正偏，已弃用
    f_center = 2 * (vwap - p_mid) / price_range if price_range > 0 else 0.0

    # F_spread: 价格离散度差异
    f_spread = (sigma_s - sigma_b) / (sigma_s + sigma_b) if (sigma_s + sigma_b) > 0 else 0.0

    # F_skew: 偏度差异，除以2压缩至[-1,1]，截断
    f_skew = float(np.clip((skew_b - skew_s) / 2, -1, 1))

    # PVDI加权和：重心0.5 + 偏度0.3 + 离散度0.2
    pvdi_weighted = 0.5 * f_center + 0.3 * f_skew + 0.2 * f_spread

    return {'f_center': f_center, 'f_spread': f_spread, 'f_skew': f_skew, 'skew_b': float(skew_b), 'skew_s': float(skew_s), 'pvdi_weighted': pvdi_weighted}


def classify_pvdi_pattern(f_center: float, f_spread: float, f_skew: float) -> Dict:
    """根据PVDI三因子符号组合判断市场微观结构状态

    8种模式基于 F_center(重心偏移)、F_spread(离散度差异)、F_skew(偏度差异) 的符号组合。
    当所有因子绝对值均<0.1时，视为中性无信号，返回pattern=0。

    Args:
        f_center: 重心偏移因子
        f_spread: 离散度差异因子
        f_skew: 偏度差异因子

    Returns:
        {'pattern': int, 'label': str, 'signal': str, 'strength': str}
    """
    # 中性判断：所有因子绝对值均<0.1
    if abs(f_center) < 0.1 and abs(f_spread) < 0.1 and abs(f_skew) < 0.1:
        return {'pattern': 0, 'label': '中性', 'signal': 'neutral', 'strength': 'weak'}

    # 符号判断
    c = '+' if f_center >= 0 else '-'
    s = '+' if f_spread >= 0 else '-'
    k = '+' if f_skew >= 0 else '-'

    # 8种模式映射
    patterns = {
        ('+', '-', '+'): {'pattern': 1, 'label': '吸筹式上涨', 'signal': 'bullish'},
        ('+', '-', '-'): {'pattern': 2, 'label': '谨慎推升', 'signal': 'bullish'},
        ('+', '+', '+'): {'pattern': 3, 'label': '追涨狂热', 'signal': 'extreme_bull'},
        ('+', '+', '-'): {'pattern': 4, 'label': '虚涨派发', 'signal': 'neutral'},
        ('-', '-', '+'): {'pattern': 5, 'label': '恐慌下跌', 'signal': 'bearish'},
        ('-', '-', '-'): {'pattern': 6, 'label': '承接式下跌', 'signal': 'bearish'},
        ('-', '+', '+'): {'pattern': 7, 'label': '多杀多', 'signal': 'extreme_bear'},
        ('-', '+', '-'): {'pattern': 8, 'label': '震荡出货', 'signal': 'neutral'},
    }

    key = (c, s, k)
    result = patterns.get(key, {'pattern': 0, 'label': '中性', 'signal': 'neutral'})

    # 强度判断：取三个因子绝对值的最大值
    max_abs = max(abs(f_center), abs(f_spread), abs(f_skew))
    if max_abs >= 0.5:
        result['strength'] = 'strong'
    elif max_abs >= 0.2:
        result['strength'] = 'medium'
    else:
        result['strength'] = 'weak'

    return result


def compute_pvdi_for_dates(api: TdxHq_API, symbol: str, date_ints: List[int]) -> pd.DataFrame:
    """批量计算多日PVDI因子

    Args:
        api: TdxHq_API 实例
        symbol: 股票代码（如 '002916'）
        date_ints: 日期整数列表，如 [20260529, 20260602]

    Returns:
        DataFrame，列：date_int, f_center, f_spread, f_skew, skew_b, skew_s,
                       pvdi_weighted, pattern, label, signal, strength
    """
    records = []
    for date_int in sorted(date_ints):
        result = compute_pvdi_for_date(api, symbol, date_int)
        if result is not None:
            cls = classify_pvdi_pattern(result['f_center'], result['f_spread'], result['f_skew'])
            records.append({
                'date_int': date_int,
                'f_center': result['f_center'],
                'f_spread': result['f_spread'],
                'f_skew': result['f_skew'],
                'skew_b': result['skew_b'],
                'skew_s': result['skew_s'],
                'pvdi_weighted': result['pvdi_weighted'],
                'pattern': cls['pattern'],
                'label': cls['label'],
                'signal': cls['signal'],
                'strength': cls['strength'],
            })
    if not records:
        return pd.DataFrame(columns=['date_int', 'f_center', 'f_spread', 'f_skew', 'skew_b', 'skew_s',
                                     'pvdi_weighted', 'pattern', 'label', 'signal', 'strength'])
    return pd.DataFrame(records)


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
