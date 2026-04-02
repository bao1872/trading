# -*- coding: utf-8 -*-
"""
分钟级行情分析核心计算模块

功能：
1. 计算日内累计成交量
2. 计算日内累计资金净流入（涨=流入，跌=流出）
3. 计算当前值 vs 历史同时间段的 ZScore

Usage:
    from features.minute_flow import analyze_minute_flow
    
    result = analyze_minute_flow(df_1m)
    print(f"成交量 ZScore: {result['volume_zscore']:.2f}")
    print(f"资金净流入 ZScore: {result['flow_zscore']:.2f}")
"""
import sys
import os
from datetime import datetime, time
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_intraday_cumulative_volume(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个交易日的日内累计成交量（按时间点）

    Args:
        df_1m: 1分钟K线数据，必须包含 datetime, volume 列

    Returns:
        DataFrame: 包含 date, time, cumulative_volume 列
        - date: 交易日期
        - time: 时间点 (HH:MM 格式)
        - cumulative_volume: 截止到该时间点的累计成交量
    """
    if df_1m.empty:
        return pd.DataFrame(columns=['date', 'time', 'cumulative_volume'])

    df = df_1m.copy()
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    df['cumulative_volume'] = df.groupby('date')['volume'].cumsum()

    result = df[['date', 'time', 'cumulative_volume']].copy()
    return result


def compute_intraday_cumulative_flow(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个交易日的日内累计资金净流入

    资金流向判断：
    - 涨（close > open）：流入金额 = amount
    - 跌（close < open）：流出金额 = amount
    - 平（close == open）：不计入

    Args:
        df_1m: 1分钟K线数据，必须包含 datetime, open, close, amount 列

    Returns:
        DataFrame: 包含 date, time, cumulative_flow, inflow, outflow 列
        - date: 交易日期
        - time: 时间点 (HH:MM 格式)
        - cumulative_flow: 截止到该时间点的累计净流入
        - inflow: 该分钟的流入金额
        - outflow: 该分钟的流出金额
    """
    if df_1m.empty:
        return pd.DataFrame(columns=['date', 'time', 'cumulative_flow', 'inflow', 'outflow'])

    df = df_1m.copy()
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    df['inflow'] = np.where(df['close'] > df['open'], df['amount'], 0.0)
    df['outflow'] = np.where(df['close'] < df['open'], df['amount'], 0.0)
    df['net_flow'] = df['inflow'] - df['outflow']

    df['cumulative_flow'] = df.groupby('date')['net_flow'].cumsum()

    result = df[['date', 'time', 'cumulative_flow', 'inflow', 'outflow']].copy()
    return result


def compute_zscore_against_history(
    current_value: float,
    history_values: List[float]
) -> Dict:
    """
    计算当前值相对于历史值的 zscore

    公式: z = (current - mean) / std

    Args:
        current_value: 当前值
        history_values: 历史同期值列表

    Returns:
        dict: {
            'zscore': float,      # ZScore 值
            'mean': float,        # 历史均值
            'std': float,         # 历史标准差
            'current': float,     # 当前值
            'history_count': int  # 历史数据点数
        }
    """
    history_arr = np.array(history_values, dtype=float)
    history_arr = history_arr[~np.isnan(history_arr)]

    if len(history_arr) == 0:
        return {
            'zscore': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'current': current_value,
            'history_count': 0
        }

    mean_val = np.mean(history_arr)
    std_val = np.std(history_arr, ddof=0)

    if std_val == 0 or np.isnan(std_val):
        zscore = np.nan
    else:
        zscore = (current_value - mean_val) / std_val

    return {
        'zscore': float(zscore) if np.isfinite(zscore) else np.nan,
        'mean': float(mean_val),
        'std': float(std_val),
        'current': float(current_value),
        'history_count': len(history_arr)
    }


def analyze_minute_flow(
    df_1m: pd.DataFrame,
    current_time: Optional[datetime] = None
) -> Dict:
    """
    分析分钟级资金流向，返回成交量 zscore 和资金净流入 zscore

    Args:
        df_1m: 1分钟K线数据，必须包含 datetime, open, high, low, close, volume, amount 列
        current_time: 当前时间（默认使用最新数据时间）

    Returns:
        dict: {
            'volume_zscore': float,           # 成交量 ZScore
            'flow_zscore': float,             # 资金净流入 ZScore
            'current_volume': float,          # 当前交易日截止到现在的累计成交量
            'current_flow': float,            # 当前交易日截止到现在的累计净流入
            'history_volume_mean': float,     # 历史同期成交量均值
            'history_flow_mean': float,       # 历史同期资金净流入均值
            'history_volume_std': float,      # 历史同期成交量标准差
            'history_flow_std': float,        # 历史同期资金净流入标准差
            'analysis_time': datetime,        # 分析时间点
            'trade_date': date,               # 当前交易日期
            'history_count': int,             # 历史数据天数
        }
    """
    if df_1m.empty:
        return {
            'volume_zscore': np.nan,
            'flow_zscore': np.nan,
            'current_volume': np.nan,
            'current_flow': np.nan,
            'history_volume_mean': np.nan,
            'history_flow_mean': np.nan,
            'history_volume_std': np.nan,
            'history_flow_std': np.nan,
            'analysis_time': current_time or datetime.now(),
            'trade_date': None,
            'history_count': 0,
        }

    df = df_1m.copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    if current_time is None:
        current_time = df['datetime'].max()

    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    current_date = current_time.date() if hasattr(current_time, 'date') else current_time
    current_time_str = current_time.strftime('%H:%M') if hasattr(current_time, 'strftime') else str(current_time)

    vol_df = compute_intraday_cumulative_volume(df)
    flow_df = compute_intraday_cumulative_flow(df)

    current_vol_row = vol_df[(vol_df['date'] == current_date) & (vol_df['time'] <= current_time_str)]
    if not current_vol_row.empty:
        current_volume = current_vol_row['cumulative_volume'].iloc[-1]
    else:
        current_volume = 0.0

    current_flow_row = flow_df[(flow_df['date'] == current_date) & (flow_df['time'] <= current_time_str)]
    if not current_flow_row.empty:
        current_flow = current_flow_row['cumulative_flow'].iloc[-1]
    else:
        current_flow = 0.0

    history_dates = sorted(vol_df['date'].unique().tolist())
    if current_date in history_dates:
        history_dates.remove(current_date)

    history_volume_values = []
    history_flow_values = []

    for hist_date in history_dates:
        hist_vol_row = vol_df[(vol_df['date'] == hist_date) & (vol_df['time'] <= current_time_str)]
        if not hist_vol_row.empty:
            history_volume_values.append(hist_vol_row['cumulative_volume'].iloc[-1])

        hist_flow_row = flow_df[(flow_df['date'] == hist_date) & (flow_df['time'] <= current_time_str)]
        if not hist_flow_row.empty:
            history_flow_values.append(hist_flow_row['cumulative_flow'].iloc[-1])

    vol_result = compute_zscore_against_history(current_volume, history_volume_values)
    flow_result = compute_zscore_against_history(current_flow, history_flow_values)

    return {
        'volume_zscore': vol_result['zscore'],
        'flow_zscore': flow_result['zscore'],
        'current_volume': vol_result['current'],
        'current_flow': flow_result['current'],
        'history_volume_mean': vol_result['mean'],
        'history_flow_mean': flow_result['mean'],
        'history_volume_std': vol_result['std'],
        'history_flow_std': flow_result['std'],
        'analysis_time': current_time,
        'trade_date': current_date,
        'history_count': vol_result['history_count'],
    }


def analyze_minute_flow_batch(
    df_1m: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    批量分析：计算每个交易日每个时间点的 ZScore

    Args:
        df_1m: 1分钟K线数据

    Returns:
        Tuple[DataFrame, DataFrame]:
        - 成交量分析结果: date, time, cumulative_volume, zscore, mean, std
        - 资金流入分析结果: date, time, cumulative_flow, zscore, mean, std
    """
    if df_1m.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df_1m.copy()
    df = df.sort_values('datetime').reset_index(drop=True)
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    vol_df = compute_intraday_cumulative_volume(df)
    flow_df = compute_intraday_cumulative_flow(df)

    all_dates = sorted(vol_df['date'].unique())
    all_times = sorted(vol_df['time'].unique())

    vol_results = []
    flow_results = []

    for target_time in all_times:
        history_vol_by_date = {}
        history_flow_by_date = {}

        for d in all_dates:
            vol_row = vol_df[(vol_df['date'] == d) & (vol_df['time'] == target_time)]
            if not vol_row.empty:
                history_vol_by_date[d] = vol_row['cumulative_volume'].iloc[-1]

            flow_row = flow_df[(flow_df['date'] == d) & (flow_df['time'] == target_time)]
            if not flow_row.empty:
                history_flow_by_date[d] = flow_row['cumulative_flow'].iloc[-1]

        for d in all_dates:
            if d not in history_vol_by_date:
                continue

            current_vol = history_vol_by_date[d]
            hist_vols = [v for date_key, v in history_vol_by_date.items() if date_key != d]
            vol_zscore_result = compute_zscore_against_history(current_vol, hist_vols)

            vol_results.append({
                'date': d,
                'time': target_time,
                'cumulative_volume': current_vol,
                'zscore': vol_zscore_result['zscore'],
                'mean': vol_zscore_result['mean'],
                'std': vol_zscore_result['std'],
            })

            if d not in history_flow_by_date:
                continue

            current_flow = history_flow_by_date[d]
            hist_flows = [v for date_key, v in history_flow_by_date.items() if date_key != d]
            flow_zscore_result = compute_zscore_against_history(current_flow, hist_flows)

            flow_results.append({
                'date': d,
                'time': target_time,
                'cumulative_flow': current_flow,
                'zscore': flow_zscore_result['zscore'],
                'mean': flow_zscore_result['mean'],
                'std': flow_zscore_result['std'],
            })

    vol_result_df = pd.DataFrame(vol_results)
    flow_result_df = pd.DataFrame(flow_results)

    return vol_result_df, flow_result_df


if __name__ == "__main__":
    print("=" * 60)
    print("分钟级行情分析模块自测")
    print("=" * 60)

    np.random.seed(42)
    dates = pd.date_range('2026-03-01 09:30', '2026-03-11 15:00', freq='1min')
    dates = dates[(dates.time >= time(9, 30)) & (dates.time <= time(11, 30)) | 
                  (dates.time >= time(13, 0)) & (dates.time <= time(15, 0))]

    dates = dates[dates.dayofweek < 5]

    n = len(dates)
    test_df = pd.DataFrame({
        'datetime': dates,
        'open': 10.0 + np.random.randn(n) * 0.1,
        'high': 10.1 + np.random.randn(n) * 0.1,
        'low': 9.9 + np.random.randn(n) * 0.1,
        'close': 10.0 + np.random.randn(n) * 0.1,
        'volume': np.random.randint(1000, 10000, n).astype(float),
        'amount': np.random.randint(10000, 100000, n).astype(float),
    })
    test_df['high'] = test_df[['open', 'close', 'high']].max(axis=1)
    test_df['low'] = test_df[['open', 'close', 'low']].min(axis=1)

    print(f"\n测试数据: {len(test_df)} 条 1 分钟 K 线")
    print(f"日期范围: {test_df['datetime'].min()} ~ {test_df['datetime'].max()}")

    print("\n" + "-" * 40)
    print("测试 analyze_minute_flow()")
    print("-" * 40)

    result = analyze_minute_flow(test_df)
    print(f"交易日期: {result['trade_date']}")
    print(f"分析时间: {result['analysis_time']}")
    print(f"历史天数: {result['history_count']}")
    print(f"\n成交量分析:")
    print(f"  当前累计成交量: {result['current_volume']:,.0f}")
    print(f"  历史均值: {result['history_volume_mean']:,.0f}")
    print(f"  历史标准差: {result['history_volume_std']:,.0f}")
    print(f"  ZScore: {result['volume_zscore']:.3f}")
    print(f"\n资金流入分析:")
    print(f"  当前累计净流入: {result['current_flow']:,.0f}")
    print(f"  历史均值: {result['history_flow_mean']:,.0f}")
    print(f"  历史标准差: {result['history_flow_std']:,.0f}")
    print(f"  ZScore: {result['flow_zscore']:.3f}")

    print("\n" + "-" * 40)
    print("测试 compute_intraday_cumulative_volume()")
    print("-" * 40)
    vol_df = compute_intraday_cumulative_volume(test_df)
    print(f"结果行数: {len(vol_df)}")
    print(f"唯一日期数: {vol_df['date'].nunique()}")
    print("\n最后一天的数据:")
    last_date = vol_df['date'].max()
    print(vol_df[vol_df['date'] == last_date].tail())

    print("\n" + "-" * 40)
    print("测试 compute_intraday_cumulative_flow()")
    print("-" * 40)
    flow_df = compute_intraday_cumulative_flow(test_df)
    print(f"结果行数: {len(flow_df)}")
    print(f"唯一日期数: {flow_df['date'].nunique()}")
    print("\n最后一天的数据:")
    print(flow_df[flow_df['date'] == last_date].tail())

    print("\n" + "=" * 60)
    print("✅ 自测完成")
    print("=" * 60)
