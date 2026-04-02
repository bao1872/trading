"""
Purpose: 主题投资放量异动检测 - Z-Score方法
Inputs: 数据字典 {code: {'name': xxx, 'data': DataFrame}}
Outputs: 包含Z-Score的DataFrame
"""
import os
import sys
import numpy as np
import pandas as pd


PERIOD_CONFIG = {
    'daily': {'window': 51, 'freq': 'd'},
    'weekly': {'window': 9, 'freq': 'w'},
    'min60': {'window': 52 * 4, 'freq': '60m'},
    'min15': {'window': 42 * 16, 'freq': '15m'},
}


def calculate_volume_zscore(
    data: dict,
    window: int = 20,
    snapshot_date: str = None,
    filter_rising: bool = True
) -> pd.DataFrame:
    """
    计算个股成交量的Z-Score

    Args:
        data: 数据字典 {code: {'name': xxx, 'data': DataFrame}}
        window: 滚动窗口大小
        snapshot_date: 指定日期截面，None则取最新
        filter_rising: 是否过滤下跌股票，默认True

    Returns:
        DataFrame with columns: code, name, zscore, volume, rolling_mean, date
    """
    results = []

    for code, info in data.items():
        df = info['data'].copy()
        if 'volume' not in df.columns or len(df) < window:
            continue

        df['rolling_mean'] = df['volume'].rolling(window=window).mean()
        df['rolling_std'] = df['volume'].rolling(window=window).std()
        df['zscore'] = (df['volume'] - df['rolling_mean']) / df['rolling_std']

        if 'close' in df.columns:
            df['prev_close'] = df['close'].shift(1)
            df['price_change'] = (df['close'] - df['prev_close']) / df['prev_close']
            df['body_down'] = df['close'] < df['open']

        if snapshot_date is None:
            target_idx = len(df) - 1
            target_date = df.index[-1]
        else:
            parsed_date = pd.Timestamp(snapshot_date)
            parsed_date_start = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
            parsed_date_end = parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)

            has_exact_date = False
            for i, idx in enumerate(df.index):
                if parsed_date_start <= idx <= parsed_date_end:
                    target_idx = i
                    target_date = idx
                    has_exact_date = True
                    break

            if not has_exact_date:
                continue

        if target_idx < window:
            continue

        latest = df.iloc[target_idx]

        if filter_rising and 'price_change' in df.columns:
            if pd.isna(latest['price_change']) or latest['price_change'] <= 0:
                continue
            if latest.get('body_down', False):
                continue
            if latest['price_change'] > 0.30:
                continue

        results.append({
            'code': code,
            'name': info['name'],
            'zscore': latest['zscore'],
            'volume': latest['volume'],
            'rolling_mean': latest['rolling_mean'],
            'date': target_date
        })

    return pd.DataFrame(results)


def load_data_for_period(period: str) -> dict:
    """
    加载指定周期的数据（从数据库）

    Args:
        period: 'daily', 'weekly', 'min60', 'min15'

    Returns:
        数据字典 {code: {'name': xxx, 'data': DataFrame}}
    """
    config = PERIOD_CONFIG.get(period)
    if config is None:
        raise ValueError(f"Unknown period: {period}. Available: {list(PERIOD_CONFIG.keys())}")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from datasource.k_data_loader import load_k_data_as_dict
    return load_k_data_as_dict(freq=period)


def get_window_for_period(period: str) -> int:
    """获取指定周期的默认窗口"""
    return PERIOD_CONFIG[period]['window']


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from datasource.k_data_loader import load_k_data_as_dict

    data = load_k_data_as_dict(freq='daily')
    df = calculate_volume_zscore(data, window=42)
    print(df.sort_values('zscore', ascending=False).head(10))
