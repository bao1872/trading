"""
Purpose: 日线滚动窗口异动选股过滤器
过滤条件:
1. 5日滚动zscore >= 2.0
2. CV <= 0.4 或 Spearman >= 0.7

Usage:
    python daily_rolling_filter.py --date 2026-03-20
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from utils.volume_anomaly import load_data_for_period


def calculate_5day_rolling(daily_data: dict) -> dict:
    """计算每日5日滚动成交量"""
    result = {}
    for code, info in daily_data.items():
        df = info['data'].copy()
        df['volume_5d'] = df['volume'].rolling(window=5).sum()
        df = df.dropna(subset=['volume_5d'])
        result[code] = {
            'name': info['name'],
            'data': df
        }
    return result


def calculate_5day_zscore(daily_data: dict, window: int = 51) -> pd.DataFrame:
    """计算5日滚动成交量的zscore"""
    rolling_data = calculate_5day_rolling(daily_data)
    results = []

    for code, info in rolling_data.items():
        df = info['data']
        if len(df) < window + 1:
            continue

        df = df.copy()
        if 'close' in df.columns and len(df) >= 2:
            df['close_prev'] = df['close'].shift(1)
            df['price_change'] = (df['close'] - df['close_prev']) / df['close_prev'] * 100
        else:
            df['price_change'] = None

        df['rolling_mean'] = df['volume_5d'].rolling(window=window).mean()
        df['rolling_std'] = df['volume_5d'].rolling(window=window).std()
        df['zscore'] = (df['volume_5d'] - df['rolling_mean']) / df['rolling_std']
        df = df.dropna(subset=['zscore'])

        for idx, row in df.iterrows():
            results.append({
                'code': code,
                'name': row['name'],
                'zscore': row['zscore'],
                'volume': row['volume'],
                'rolling_mean': row['rolling_mean'],
                'date': idx,
                'price_change': round(row['price_change'], 2) if pd.notna(row.get('price_change')) else None,
            })

    return pd.DataFrame(results)


def calculate_volume_cv(daily_data: dict, current_date: pd.Timestamp, code: str) -> float:
    """计算周内日均成交量变异系数(CV)"""
    if code not in daily_data:
        return None

    df = daily_data[code]['data']
    week_start = current_date - pd.Timedelta(days=6)
    week_end = current_date + pd.Timedelta(days=1)

    week_volumes = []
    for idx in df.index:
        if week_start <= idx <= week_end:
            if 'volume' in df.columns:
                vol = df.loc[idx, 'volume']
                if pd.notna(vol) and vol > 0:
                    week_volumes.append(vol)

    if len(week_volumes) < 2:
        return None

    mean_vol = np.mean(week_volumes)
    std_vol = np.std(week_volumes, ddof=1)

    if mean_vol == 0:
        return None

    cv = std_vol / mean_vol
    return round(cv, 3)


def calculate_volume_spearman(daily_data: dict, current_date: pd.Timestamp, code: str) -> float:
    """计算周内日均成交量放大顺序的Spearman等级相关系数"""
    if code not in daily_data:
        return None

    df = daily_data[code]['data']
    week_start = current_date - pd.Timedelta(days=6)
    week_end = current_date + pd.Timedelta(days=1)

    week_volumes = []
    for idx in df.index:
        if week_start <= idx <= week_end:
            if 'volume' in df.columns:
                vol = df.loc[idx, 'volume']
                if pd.notna(vol) and vol > 0:
                    week_volumes.append((idx, vol))

    if len(week_volumes) < 2:
        return None

    days = list(range(len(week_volumes)))
    volumes = [v[1] for v in week_volumes]

    rho, _ = spearmanr(days, volumes)
    return round(rho, 3)


def filter_rolling_signals(
    daily_data: dict,
    snapshot_date: str,
    cv_threshold: float = 0.4,
    spearman_threshold: float = 0.7,
    zscore_threshold: float = 2.5
) -> pd.DataFrame:
    """
    过滤选股条件：
    1. 5日滚动zscore >= zscore_threshold
    2. CV <= cv_threshold 或 Spearman >= spearman_threshold

    Args:
        daily_data: 日线数据
        snapshot_date: 截面日期
        cv_threshold: CV阈值（越小越均匀）
        spearman_threshold: Spearman阈值（越大越递增）
        zscore_threshold: zscore阈值

    Returns:
        DataFrame with filtered signals
    """
    rolling_data = calculate_5day_rolling(daily_data)
    volume_df = calculate_5day_zscore(daily_data)

    if volume_df.empty:
        return pd.DataFrame()

    snapshot_ts = pd.Timestamp(snapshot_date)
    volume_df = volume_df[volume_df['date'].dt.date == snapshot_ts.date()]

    if volume_df.empty:
        return pd.DataFrame()

    filtered_results = []

    for _, row in volume_df.iterrows():
        code = row['code']
        current_date = row['date']

        if row['zscore'] < zscore_threshold:
            continue

        cv = calculate_volume_cv(daily_data, current_date, code)
        spearman = calculate_volume_spearman(daily_data, current_date, code)

        if cv is not None and spearman is not None:
            if not (cv <= cv_threshold or spearman >= spearman_threshold):
                continue
        elif cv is not None:
            if not (cv <= cv_threshold):
                continue
        elif spearman is not None:
            if not (spearman >= spearman_threshold):
                continue
        else:
            continue

        filtered_results.append({
            'code': code,
            'name': row['name'],
            'zscore': row['zscore'],
            'volume': row['volume'],
            'rolling_mean': row['rolling_mean'],
            'date': row['date'],
            'cv': cv,
            'spearman': spearman,
            'price_change': row.get('price_change'),
        })

    return pd.DataFrame(filtered_results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='日线滚动窗口异动选股过滤器')
    parser.add_argument('--date', type=str, default='latest', help='截面日期')
    parser.add_argument('--cv', type=float, default=0.4, help='CV阈值')
    parser.add_argument('--spearman', type=float, default=0.7, help='Spearman阈值')
    args = parser.parse_args()

    print("加载日线数据...")
    daily_data = load_data_for_period('daily')

    if args.date == 'latest':
        snapshot_date = None
        dates = sorted(daily_data[list(daily_data.keys())[0]]['data'].index)
        snapshot_date = str(dates[-1])[:10]
    else:
        snapshot_date = args.date

    print(f"过滤选股: {snapshot_date}")
    signals = filter_rolling_signals(
        daily_data=daily_data,
        snapshot_date=snapshot_date,
        cv_threshold=args.cv,
        spearman_threshold=args.spearman
    )

    print(f"过滤后股票数: {len(signals)}")
    if not signals.empty:
        print(signals.head(10))