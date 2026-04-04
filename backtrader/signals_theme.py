"""
Purpose: 主题投资信号输出 - 整合放量检测+概念映射+主题聚合
Inputs: dataset/, stock_concepts_cache.xlsx, theme_mapping.json
Outputs: 主题强度表、概念强度表、个股异动表、交叉信号表
"""
import pandas as pd
import numpy as np
import os
import re

from utils.concept_mapping import load_concept_data, build_stock_to_concepts, build_concept_to_stocks
from utils.theme_aggregation import (
    aggregate_theme_with_concept_detail,
    load_theme_mapping,
    build_concept_to_theme_map,
    get_excluded_concepts,
    get_stock_themes
)
from datasource.k_data_loader import iter_k_data_with_names, load_all_stock_data
from datasource.database import get_session, bulk_upsert


def ensure_signal_tables(session) -> None:
    """确保信号表存在，不存在则创建（PostgreSQL）"""
    from datasource.database import execute_sql

    tables_sql = {
        'theme_signals': """
            CREATE TABLE IF NOT EXISTS theme_signals (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                period VARCHAR(20),
                theme VARCHAR(100) NOT NULL,
                total_zscore FLOAT,
                avg_zscore FLOAT,
                stock_count INTEGER,
                strength FLOAT,
                anomalous_concept_count INTEGER,
                total_concept_count INTEGER,
                concept_coverage FLOAT,
                top_stocks TEXT,
                top_concepts TEXT,
                UNIQUE(snapshot_date, theme)
            )
        """,
        'concept_signals': """
            CREATE TABLE IF NOT EXISTS concept_signals (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                concept VARCHAR(100) NOT NULL,
                theme VARCHAR(100),
                total_zscore FLOAT,
                avg_zscore FLOAT,
                anomalous_stock_count INTEGER,
                total_stock_count INTEGER,
                normalized_strength FLOAT,
                intensity FLOAT,
                top_stocks TEXT,
                UNIQUE(snapshot_date, concept)
            )
        """,
        'stock_anomaly_signals': """
            CREATE TABLE IF NOT EXISTS stock_anomaly_signals (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                code VARCHAR(20) NOT NULL,
                name VARCHAR(50),
                zscore FLOAT,
                date DATE,
                price_change FLOAT,
                themes_str TEXT,
                concepts_str TEXT,
                UNIQUE(snapshot_date, code)
            )
        """,
        'limit_up_signals': """
            CREATE TABLE IF NOT EXISTS limit_up_signals (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                theme VARCHAR(100),
                streak_key VARCHAR(20),
                ts_code VARCHAR(20) NOT NULL,
                stock_name VARCHAR(50),
                streak_count INTEGER,
                streak_trading_days INTEGER,
                signal_type VARCHAR(20) NOT NULL,
                UNIQUE(snapshot_date, ts_code, signal_type)
            )
        """
    }
    for table_name, sql in tables_sql.items():
        execute_sql(sql)


def save_signals_to_database(signals: dict) -> None:
    """保存信号结果到数据库"""
    from datasource.database import get_session, bulk_upsert
    import pandas as pd

    snapshot_date = signals['snapshot_date']

    with get_session() as session:
        ensure_signal_tables(session)

        theme_df = signals['theme_ranking']
        if not theme_df.empty:
            theme_df = theme_df.copy()
            theme_df['snapshot_date'] = snapshot_date
            theme_df['period'] = signals['period']
            if 'top_stocks' in theme_df.columns:
                theme_df['top_stocks'] = theme_df['top_stocks'].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else x
                )
            theme_cols = ['snapshot_date', 'period', 'theme', 'total_zscore', 'stock_count',
                         'strength', 'anomalous_concept_count', 'total_concept_count',
                         'concept_coverage', 'top_stocks']
            theme_cols = [c for c in theme_cols if c in theme_df.columns]
            bulk_upsert(session, 'theme_signals', theme_df[theme_cols], ['snapshot_date', 'theme'])

        concept_df = signals['concept_ranking']
        if not concept_df.empty:
            concept_df = concept_df.copy()
            concept_df['snapshot_date'] = snapshot_date
            if 'top_stocks' in concept_df.columns:
                concept_df['top_stocks'] = concept_df['top_stocks'].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else x
                )
            concept_cols = ['snapshot_date', 'concept', 'theme', 'total_zscore', 'avg_zscore',
                          'anomalous_stock_count', 'total_stock_count', 'normalized_strength',
                          'intensity', 'top_stocks']
            concept_cols = [c for c in concept_cols if c in concept_df.columns]
            bulk_upsert(session, 'concept_signals', concept_df[concept_cols], ['snapshot_date', 'concept'])

        stock_df = signals['stock_anomaly']
        if not stock_df.empty:
            stock_df = stock_df.copy()
            stock_df['snapshot_date'] = snapshot_date
            if 'date' in stock_df.columns:
                stock_df['date'] = stock_df['date'].apply(lambda x: str(x)[:10] if hasattr(x, 'strftime') else x)
            stock_cols = ['snapshot_date', 'code', 'name', 'zscore', 'date',
                         'price_change', 'themes_str', 'concepts_str']
            stock_cols = [c for c in stock_cols if c in stock_df.columns]
            bulk_upsert(session, 'stock_anomaly_signals', stock_df[stock_cols], ['snapshot_date', 'code'])

        limit_up_df = signals['theme_limit_up']
        if not limit_up_df.empty:
            limit_up_rows = []
            for _, row in limit_up_df.iterrows():
                theme = row['theme']
                for key, stocks in row.get('streak_groups', {}).items():
                    for s in stocks:
                        limit_up_rows.append({
                            'snapshot_date': snapshot_date,
                            'theme': theme,
                            'streak_key': key,
                            'ts_code': s['code'],
                            'stock_name': s['name'],
                            'streak_count': s['streak_count'],
                            'streak_trading_days': s['streak_trading_days'],
                            'signal_type': 'limit_up'
                        })
            if limit_up_rows:
                limit_up_db_df = pd.DataFrame(limit_up_rows)
                bulk_upsert(session, 'limit_up_signals', limit_up_db_df, ['snapshot_date', 'ts_code', 'signal_type'])

        limit_down_df = signals['theme_limit_down']
        if not limit_down_df.empty:
            limit_down_rows = []
            for _, row in limit_down_df.iterrows():
                theme = row['theme']
                for key, stocks in row.get('streak_groups', {}).items():
                    for s in stocks:
                        limit_down_rows.append({
                            'snapshot_date': snapshot_date,
                            'theme': theme,
                            'streak_key': key,
                            'ts_code': s['code'],
                            'stock_name': s['name'],
                            'streak_count': s['streak_count'],
                            'streak_trading_days': s['streak_trading_days'],
                            'signal_type': 'limit_down'
                        })
            if limit_down_rows:
                limit_down_db_df = pd.DataFrame(limit_down_rows)
                bulk_upsert(session, 'limit_up_signals', limit_down_db_df, ['snapshot_date', 'ts_code', 'signal_type'])


def calculate_consecutive_limit(df: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """
    向量化计算连板天数（涨停/跌停）

    "几天几板" 定义:
    - 几板 = 涨停天数（涨停日计数）
    - 几天 = 从第一板到当日的日历天数（含首尾）

    中间最多允许连续2天非涨停日，否则断开重新计算

    Args:
        df: 含 is_limit_up_close, is_limit_down_close 列的DataFrame
        max_gap: 最大允许的连续非涨停日数

    Returns:
        添加了 is_limit_up, is_limit_down, streak_count, streak_trading_days, streak_key 列的DataFrame
    """
    df = df.sort_values(['ts_code', 'bar_time']).reset_index(drop=True)

    df['is_limit_up'] = df['is_limit_up_close']
    df['is_limit_down'] = df['is_limit_down_close']

    streak_up_count = np.zeros(len(df), dtype=int)
    streak_up_days = np.zeros(len(df), dtype=int)
    streak_down_count = np.zeros(len(df), dtype=int)
    streak_down_days = np.zeros(len(df), dtype=int)

    codes = df['ts_code'].unique()

    for code in codes:
        mask = df['ts_code'] == code
        idx_arr = df.index[mask].values
        bar_times = df.loc[mask, 'bar_time'].values
        is_lu = df.loc[mask, 'is_limit_up_close'].values
        is_ld = df.loc[mask, 'is_limit_down_close'].values
        n = len(idx_arr)

        cur_up_count = 0
        last_up_idx = -9999
        gap_count_up = 0
        streak_start_i = -1

        cur_down_count = 0
        last_down_idx = -9999
        gap_count_down = 0
        streak_start_i_down = -1

        for i in range(n):
            bar_time_i = bar_times[i]

            if is_lu[i]:
                gap = i - last_up_idx - 1
                if gap <= max_gap:
                    cur_up_count += 1
                else:
                    cur_up_count = 1
                    streak_start_i = i

                last_up_idx = i
                gap_count_up = 0
            else:
                gap_count_up += 1
                if gap_count_up > max_gap:
                    cur_up_count = 0
                    last_up_idx = -9999
                    streak_start_i = -1

            streak_up_count[idx_arr[i]] = cur_up_count
            streak_up_days[idx_arr[i]] = i - streak_start_i + 1 if streak_start_i >= 0 else 0

            if is_ld[i]:
                gap = i - last_down_idx - 1
                if gap <= max_gap:
                    cur_down_count += 1
                else:
                    cur_down_count = 1
                    streak_start_i_down = i

                last_down_idx = i
                gap_count_down = 0
            else:
                gap_count_down += 1
                if gap_count_down > max_gap:
                    cur_down_count = 0
                    last_down_idx = -9999
                    streak_start_i_down = -1

            streak_down_count[idx_arr[i]] = cur_down_count
            streak_down_days[idx_arr[i]] = i - streak_start_i_down + 1 if streak_start_i_down >= 0 else 0

    streak_key_up = [
        f"{sd}天{sc}板" if sc > 0 else ''
        for sd, sc in zip(streak_up_days, streak_up_count)
    ]
    df['streak_count'] = streak_up_count
    df['streak_trading_days'] = streak_up_days
    df['streak_key'] = streak_key_up

    return df


def _streak_sort_key(key):
    m = re.match(r'(\d+)天(\d+)板', key)
    if m:
        return (-int(m.group(2)), int(m.group(1)))
    return (0, 0)


def process_all_stocks_vectorized(
    freq: str,
    window: int,
    snapshot_date: str,
    lookback: int = 51,
    filter_rising: bool = True,
    limit_stocks: int = None,
    preloaded_data: tuple = None
) -> pd.DataFrame:
    """
    向量化一次性计算所有股票的Z-Score和涨跌停信息

    Args:
        preloaded_data: 可选，(DataFrame, name_map) 元组，若传入则直接使用，不重复加载

    Returns:
        DataFrame with columns: code, name, zscore, volume, rolling_mean, date,
                                limit_up_info, limit_down_info, price_change
    """
    if preloaded_data is not None:
        all_data, name_map = preloaded_data
        all_data = all_data.copy()
    else:
        target_date = None
        if snapshot_date:
            target_date = pd.Timestamp(snapshot_date)

        start_date = None
        if target_date is not None:
            start_date = (target_date - pd.Timedelta(days=lookback + 60)).strftime('%Y-%m-%d')

        print(f"  加载数据范围: {start_date} 到 {snapshot_date or '最新'}...")
        all_data, name_map = load_all_stock_data(freq, start_date=start_date, end_date=None, limit_stocks=limit_stocks)

    if all_data.empty:
        return pd.DataFrame()

    print(f"  数据量: {len(all_data)} 行, {len(name_map)} 只股票")

    target_date = None
    target_date_start = None
    target_date_end = None
    if snapshot_date:
        target_date = pd.Timestamp(snapshot_date)
        target_date_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        target_date_end = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        # 统一移除时区，避免比较问题
        all_data['bar_time'] = pd.to_datetime(all_data['bar_time']).dt.tz_localize(None)

    all_data = all_data[all_data['ts_code'].isin(name_map.keys())]

    print(f"  计算Z-Score (向量化)...")
    all_data['volume_5d'] = all_data.groupby('ts_code')['volume'].transform(
        lambda x: x.rolling(window=5).sum()
    )

    all_data['rolling_mean'] = all_data.groupby('ts_code')['volume_5d'].transform(
        lambda x: x.rolling(window=window).mean()
    )
    all_data['rolling_std'] = all_data.groupby('ts_code')['volume_5d'].transform(
        lambda x: x.rolling(window=window).std()
    )
    all_data['zscore'] = (all_data['volume_5d'] - all_data['rolling_mean']) / all_data['rolling_std']

    all_data['prev_close'] = all_data.groupby('ts_code')['close'].shift(1)
    all_data['price_change'] = (all_data['close'] - all_data['prev_close']) / all_data['prev_close']
    all_data['body_down'] = all_data['close'] < all_data['open']

    code_str = all_data['ts_code'].astype(str)
    conditions_limit_up = [
        code_str.str.contains('ST') | code_str.str.startswith('4'),
        code_str.str.startswith('688') | code_str.str.startswith('30'),
        code_str.str.startswith('8'),
    ]
    choices_limit_up = [1.05, 1.20, 1.30]
    all_data['limit_mult'] = np.select(conditions_limit_up, choices_limit_up, default=1.10)
    all_data['limit_up_price'] = (all_data['prev_close'] * all_data['limit_mult']).round(2)

    conditions_limit_down = [
        code_str.str.contains('ST') | code_str.str.startswith('4'),
        code_str.str.startswith('688') | code_str.str.startswith('30'),
        code_str.str.startswith('8'),
    ]
    choices_limit_down = [0.95, 0.80, 0.70]
    all_data['limit_down_price'] = (all_data['prev_close'] * np.select(conditions_limit_down, choices_limit_down, default=0.90)).round(2)

    all_data['is_limit_up_close'] = all_data['close'] >= all_data['limit_up_price']
    all_data['is_limit_down_close'] = all_data['close'] <= all_data['limit_down_price']

    print(f"  计算连板天数...")
    all_data = calculate_consecutive_limit(all_data)

    if target_date:
        snapshot_mask = (all_data['bar_time'] >= target_date_start) & (all_data['bar_time'] <= target_date_end)
        snapshot_data = all_data[snapshot_mask].copy()
    else:
        snapshot_data = all_data.groupby('ts_code').last().reset_index()

    if filter_rising:
        snapshot_data = snapshot_data[
            (snapshot_data['price_change'] > 0) &
            (~snapshot_data['body_down']) &
            (snapshot_data['price_change'] <= 0.30)
        ]

    results = []
    for _, row in snapshot_data.iterrows():
        code = row['ts_code']
        results.append({
            'code': code,
            'name': row['name'],
            'zscore': row['zscore'],
            'volume': row['volume'],
            'rolling_mean': row['rolling_mean'],
            'date': row['bar_time'],
            'price_change': row.get('price_change'),
            'streak_count': row.get('streak_count', 0),
            'streak_trading_days': row.get('streak_trading_days', 0),
            'streak_key': row.get('streak_key', ''),
            'is_limit_up': row.get('is_limit_up', False),
            'is_limit_down': row.get('is_limit_down', False),
        })

    return pd.DataFrame(results)


def aggregate_limit_up_by_theme(
    limit_up_df: pd.DataFrame,
    stock_to_concepts: dict,
    concept_to_theme: dict,
    excluded: set,
    theme_ranking: list = None
) -> pd.DataFrame:
    """按主题聚合涨停统计"""
    if theme_ranking is None:
        theme_ranking = []

    theme_limit_up = {}
    assigned_stocks = set()

    for _, row in limit_up_df.iterrows():
        code = row['code']
        if code in assigned_stocks:
            continue
        if not row.get('is_limit_up', False):
            continue

        concepts = stock_to_concepts.get(code, [])
        stock_themes = []
        for concept in concepts:
            if concept in excluded:
                continue
            theme = concept_to_theme.get(concept)
            if theme and theme not in stock_themes:
                stock_themes.append(theme)

        if not stock_themes:
            continue

        best_theme = None
        for theme in theme_ranking:
            if theme in stock_themes:
                best_theme = theme
                break
        if best_theme is None:
            best_theme = stock_themes[0]

        if best_theme not in theme_limit_up:
            theme_limit_up[best_theme] = {'limit_up_stocks': [], 'streak_groups': {}}

        assigned_stocks.add(code)

        streak_count = int(row.get('streak_count', 0))
        streak_trading_days = int(row.get('streak_trading_days', 0))

        theme_limit_up[best_theme]['limit_up_stocks'].append({
            'code': code,
            'name': row['name'],
            'limit_up_count': streak_count,
            'streak_count': streak_count,
            'streak_trading_days': streak_trading_days,
        })

        streak_key = f"{streak_trading_days}天{streak_count}板" if streak_count > 0 else ''
        if streak_key and streak_count > 0:
            if streak_key not in theme_limit_up[best_theme]['streak_groups']:
                theme_limit_up[best_theme]['streak_groups'][streak_key] = []
            theme_limit_up[best_theme]['streak_groups'][streak_key].append({
                'code': code,
                'name': row['name'],
                'streak_count': streak_count,
                'streak_trading_days': streak_trading_days,
            })

    results = []
    for theme, info in theme_limit_up.items():
        results.append({
            'theme': theme,
            'limit_up_stock_count': len(info['limit_up_stocks']),
            'streak_groups': info['streak_groups'],
            'stocks': info['limit_up_stocks']
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('limit_up_stock_count', ascending=False).reset_index(drop=True)
    return df


def aggregate_limit_down_by_theme(
    limit_down_df: pd.DataFrame,
    stock_to_concepts: dict,
    concept_to_theme: dict,
    excluded: set,
    theme_ranking: list = None
) -> pd.DataFrame:
    """按主题聚合跌停统计"""
    if theme_ranking is None:
        theme_ranking = []

    theme_limit_down = {}
    assigned_stocks = set()

    for _, row in limit_down_df.iterrows():
        code = row['code']
        if code in assigned_stocks:
            continue
        if not row.get('is_limit_down', False):
            continue

        concepts = stock_to_concepts.get(code, [])
        stock_themes = []
        for concept in concepts:
            if concept in excluded:
                continue
            theme = concept_to_theme.get(concept)
            if theme and theme not in stock_themes:
                stock_themes.append(theme)

        if not stock_themes:
            continue

        best_theme = None
        for theme in theme_ranking:
            if theme in stock_themes:
                best_theme = theme
                break
        if best_theme is None:
            best_theme = stock_themes[0]

        if best_theme not in theme_limit_down:
            theme_limit_down[best_theme] = {'limit_down_stocks': [], 'streak_groups': {}}

        assigned_stocks.add(code)

        streak_count = int(row.get('limit_down_count', 1))
        streak_trading_days = 1

        theme_limit_down[best_theme]['limit_down_stocks'].append({
            'code': code,
            'name': row['name'],
            'limit_down_count': int(row.get('limit_down_count', 1)),
            'streak_count': streak_count,
            'streak_trading_days': streak_trading_days,
        })

        key = f"{streak_trading_days}天{streak_count}板"
        if key not in theme_limit_down[best_theme]['streak_groups']:
            theme_limit_down[best_theme]['streak_groups'][key] = []
        theme_limit_down[best_theme]['streak_groups'][key].append({
            'code': code,
            'name': row['name'],
            'streak_count': streak_count,
            'streak_trading_days': streak_trading_days,
        })

    results = []
    for theme, info in theme_limit_down.items():
        results.append({
            'theme': theme,
            'limit_down_stock_count': len(info['limit_down_stocks']),
            'streak_groups': info['streak_groups'],
            'stocks': info['limit_down_stocks']
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('limit_down_stock_count', ascending=False).reset_index(drop=True)
    return df


def generate_signals(
    period: str = 'daily',
    snapshot_date: str = None,
    zscore_threshold: float = 2.0,
    top_n_themes: int = 10,
    top_n_stocks: int = 20,
    top_n_concepts: int = 20,
    concept_xlsx_path: str = '../stock_concepts_cache.xlsx',
    limit_stocks: int = None,
    preloaded_data: tuple = None
) -> dict:
    """
    生成主题投资信号

    Args:
        preloaded_data: 可选，(DataFrame, name_map) 元组，若传入则直接使用，不重复加载
    """
    from utils.volume_anomaly import get_window_for_period
    window = get_window_for_period(period)

    print("加载概念和主题映射...")
    concept_df = load_concept_data(concept_xlsx_path)
    theme_mapping = load_theme_mapping()

    s2c = build_stock_to_concepts(concept_df)
    c2s = build_concept_to_stocks(concept_df)
    concept_to_theme = build_concept_to_theme_map(theme_mapping)
    excluded = get_excluded_concepts(theme_mapping)

    print("向量化计算所有指标...")
    volume_df = process_all_stocks_vectorized(period, window=window, snapshot_date=snapshot_date, lookback=51, limit_stocks=limit_stocks, preloaded_data=preloaded_data)

    if volume_df.empty:
        return {
            'period': period,
            'snapshot_date': snapshot_date,
            'theme_ranking': pd.DataFrame(),
            'concept_ranking': pd.DataFrame(),
            'theme_concept_detail': pd.DataFrame(),
            'stock_anomaly': pd.DataFrame(),
            'cross_signals': pd.DataFrame(),
            'theme_limit_up': pd.DataFrame(),
            'theme_limit_down': pd.DataFrame()
        }

    print(f"  处理完成: {len(volume_df)} 只股票, 涨停={volume_df['is_limit_up'].sum()}, 跌停={volume_df['is_limit_down'].sum()}")

    volume_df = volume_df.sort_values('zscore', ascending=False)

    print("聚合主题和概念...")
    theme_df, concept_df_out, detail_df = aggregate_theme_with_concept_detail(
        volume_df, s2c, c2s, zscore_threshold=zscore_threshold
    )

    top_themes_list = theme_df.head(top_n_themes)['theme'].tolist() if not theme_df.empty else []

    cross_df = volume_df[volume_df['zscore'] > zscore_threshold].copy() if 'zscore' in volume_df.columns else pd.DataFrame()
    if not cross_df.empty:
        cross_df['themes'] = cross_df['code'].map(
            lambda x: get_stock_themes(s2c.get(x, []), concept_to_theme, excluded)
        )
        cross_df['themes_str'] = cross_df['themes'].map(lambda x: ','.join(x) if x else '')
        cross_df = cross_df[cross_df['themes'].map(lambda x: any(t in top_themes_list for t in x))]
        cross_df = cross_df.sort_values('zscore', ascending=False)

    anomalous_df = volume_df[volume_df['zscore'] > zscore_threshold].copy() if 'zscore' in volume_df.columns else pd.DataFrame()
    if not anomalous_df.empty:
        anomalous_df['themes'] = anomalous_df['code'].map(
            lambda x: get_stock_themes(s2c.get(x, []), concept_to_theme, excluded)
        )
        anomalous_df['themes_str'] = anomalous_df['themes'].map(lambda x: ','.join(x) if x else '')
        anomalous_df['concepts_str'] = anomalous_df['code'].map(
            lambda x: ','.join(s2c.get(x, [])) if s2c.get(x) else ''
        )
        anomalous_df = anomalous_df.sort_values('zscore', ascending=False)
    else:
        anomalous_df = pd.DataFrame(columns=['code', 'name', 'zscore', 'volume', 'rolling_mean', 'date', 'price_change', 'themes', 'themes_str', 'concepts_str'])

    print("聚合涨跌停...")
    limit_up_df = volume_df[volume_df['is_limit_up'] == True].copy()
    limit_down_df = volume_df[volume_df['is_limit_down'] == True].copy()

    theme_limit_up_df = aggregate_limit_up_by_theme(
        limit_up_df, s2c, concept_to_theme, excluded, theme_ranking=top_themes_list
    )
    theme_limit_down_df = aggregate_limit_down_by_theme(
        limit_down_df, s2c, concept_to_theme, excluded, theme_ranking=top_themes_list
    )

    return {
        'period': period,
        'snapshot_date': snapshot_date or str(volume_df['date'].iloc[0]),
        'theme_ranking': theme_df if not theme_df.empty else pd.DataFrame(),
        'concept_ranking': concept_df_out if not concept_df_out.empty else pd.DataFrame(),
        'theme_concept_detail': detail_df if not detail_df.empty else pd.DataFrame(),
        'stock_anomaly': anomalous_df,
        'cross_signals': cross_df.head(top_n_stocks),
        'theme_limit_up': theme_limit_up_df,
        'theme_limit_down': theme_limit_down_df
    }


def print_signals(signals: dict):
    """打印信号报告"""
    print('=' * 80)
    print(f"主题投资超短线策略 - 信号报告 [{signals['period']}] [{signals['snapshot_date']}]")
    print('=' * 80)

    print('\n【主题强度排名 TOP 10 - 含驱动概念】')
    print('-' * 100)
    theme_df = signals['theme_ranking']
    detail_df = signals['theme_concept_detail']
    if theme_df.empty:
        print("无数据")
    else:
        print(f"{'排名':<4} {'主题':<12} {'总Z':>8} {'股数':>6} {'强度':>8} {'覆盖':>8} {'驱动概念(异动股/总股,强度)'}")
        print('-' * 100)
        for i, row in theme_df.iterrows():
            theme_name = row['theme']
            coverage = f"{row['anomalous_concept_count']}/{row['total_concept_count']}"
            if not detail_df.empty:
                theme_concepts = detail_df[detail_df['theme'] == theme_name].sort_values('normalized_strength', ascending=False)
                driver_concepts = []
                for _, cr in theme_concepts.head(3).iterrows():
                    pct = cr['anomalous_stock_count'] / cr['total_stock_count'] * 100 if cr['total_stock_count'] > 0 else 0
                    driver_concepts.append(f"{cr['concept']}({cr['anomalous_stock_count']}/{cr['total_stock_count']},{cr['normalized_strength']:.1f})")
                drivers_str = ' '.join(driver_concepts) if driver_concepts else ''
            else:
                drivers_str = ''
            print(f"{i+1:<4} {row['theme']:<12} {row['total_zscore']:>8.1f} {row['stock_count']:>6} {row['strength']:>8.2f} {coverage:>8} {drivers_str}")

    print('\n【概念强度排名 TOP 20 - 归一化】')
    print('-' * 80)
    concept_df = signals['concept_ranking']
    if concept_df.empty:
        print("无数据")
    else:
        print(f"{'排名':<4} {'概念':<16} {'主题':<12} {'总Z-Score':>10} {'异动股':>8} {'总股数':>8} {'均Z-Score':>10} {'强度':>8} {'占比':>8}")
        print('-' * 80)
        for i, row in concept_df.head(20).iterrows():
            print(f"{i+1:<4} {row['concept']:<16} {row['theme']:<12} {row['total_zscore']:>10.2f} {row['anomalous_stock_count']:>8} {row['total_stock_count']:>8} {row['avg_zscore']:>10.2f} {row['normalized_strength']:>8.2f} {row['intensity']:>8.1%}")

    print('\n【个股异动详情 TOP 20】')
    print('-' * 80)
    stock_df = signals['stock_anomaly']
    if stock_df.empty:
        print("无数据")
    else:
        print(f"{'代码':<12} {'名称':<10} {'Z-Score':>10} {'所属主题'}")
        print('-' * 80)
        for _, row in stock_df.iterrows():
            themes_str = row.get('themes_str', '')
            print(f"{row['code']:<12} {row['name']:<10} {row['zscore']:>10.2f} {themes_str}")

    print('\n【交叉信号：强势主题 + 异常放量】')
    print('-' * 80)
    cross_df = signals['cross_signals']
    if cross_df.empty:
        print("无数据")
    else:
        print(f"{'代码':<12} {'名称':<10} {'Z-Score':>10} {'所属主题'}")
        print('-' * 80)
        for _, row in cross_df.iterrows():
            themes_str = row.get('themes_str', '')
            print(f"{row['code']:<12} {row['name']:<10} {row['zscore']:>10.2f} {themes_str}")

    print('\n【涨停金字塔 - 近10日】')
    print('=' * 80)
    limit_df = signals['theme_limit_up']
    if limit_df.empty:
        print("无涨停")
    else:
        for _, row in limit_df.iterrows():
            print(f"\n主题: {row['theme']}")
            print(f"涨停股数: {row['limit_up_stock_count']} 只")
            print('-' * 80)
            ordered_keys = sorted(row['streak_groups'].keys(), key=_streak_sort_key)
            for key in ordered_keys:
                if key in row['streak_groups'] and row['streak_groups'][key]:
                    stocks = row['streak_groups'][key]
                    names = ', '.join([f"{s['name']}({s['code']})" for s in stocks])
                    print(f"{key:<12} {names}")

    print('\n【跌停金字塔 - 近10日】')
    print('=' * 80)
    limit_down_df = signals['theme_limit_down']
    if limit_down_df.empty:
        print("无跌停")
    else:
        for _, row in limit_down_df.iterrows():
            print(f"\n主题: {row['theme']}")
            print(f"跌停股数: {row['limit_down_stock_count']} 只")
            print('-' * 80)
            ordered_keys = sorted(row['streak_groups'].keys(), key=_streak_sort_key)
            for key in ordered_keys:
                if key in row['streak_groups'] and row['streak_groups'][key]:
                    stocks = row['streak_groups'][key]
                    names = ', '.join([f"{s['name']}({s['code']})" for s in stocks])
                    print(f"{key:<12} {names}")

    print('\n' + '=' * 80)


def export_signals_excel(signals: dict, output_dir: str = '../review') -> str:
    """导出信号报告为Excel文件"""
    from datasource.k_data_loader import build_name_map

    period = signals['period']
    snapshot_date = signals['snapshot_date']
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'theme_{snapshot_date}.xlsx')

    code_to_name = build_name_map()

    column_rename = {
        'theme': '主题', 'total_zscore': '总Z分数', 'stock_count': '股票数量',
        'strength': '强度', 'anomalous_concept_count': '异动概念数',
        'total_concept_count': '总概念数', 'concept_coverage': '概念覆盖率',
        'top_stocks': 'TOP股票', 'concept': '概念',
        'anomalous_stock_count': '异动股数', 'total_stock_count': '总股数',
        'avg_zscore': '平均Z分数', 'normalized_strength': '标准化强度',
        'intensity': '强度', 'zscore': 'Z分数', 'volume': '成交量',
        'rolling_mean': '滚动均值', 'date': '日期',
        'themes': '主题列表', 'themes_str': '主题',
    }

    def rename_columns(df):
        return df.rename(columns=column_rename)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        theme_df = signals['theme_ranking']
        if not theme_df.empty:
            theme_export = theme_df.head(21).copy()
            if 'top_stocks' in theme_export.columns:
                def format_stocks(codes):
                    if isinstance(codes, list):
                        return ', '.join([f"{code_to_name.get(c, c)}({c})" for c in codes])
                    return codes
                theme_export['top_stocks'] = theme_export['top_stocks'].apply(format_stocks)
            rename_columns(theme_export).to_excel(writer, sheet_name='主题排名', index=False)

        concept_df = signals['concept_ranking']
        if not concept_df.empty:
            concept_export = concept_df.head(20).copy()
            if 'top_stocks' in concept_export.columns:
                def format_stocks(codes):
                    if isinstance(codes, list):
                        return ', '.join([f"{code_to_name.get(c, c)}({c})" for c in codes])
                    return codes
                concept_export['top_stocks'] = concept_export['top_stocks'].apply(format_stocks)
            rename_columns(concept_export).to_excel(writer, sheet_name='概念排名', index=False)

        stock_df = signals['stock_anomaly']
        if not stock_df.empty:
            stock_export = stock_df[stock_df['zscore'] > 2.5].copy()
            if not stock_export.empty:
                if 'name' in stock_export.columns and 'code' in stock_export.columns:
                    stock_export['股票'] = stock_export['name'] + '(' + stock_export['code'] + ')'
                    cols = ['股票'] + [c for c in stock_export.columns if c not in ['name', 'code', '股票']]
                    stock_export = stock_export[cols]
                rename_columns(stock_export).to_excel(writer, sheet_name='个股异动', index=False)

        limit_up_df = signals['theme_limit_up']
        if not limit_up_df.empty:
            rows = []
            for _, row in limit_up_df.iterrows():
                theme = row['theme']
                ordered_keys = sorted(row['streak_groups'].keys(), key=_streak_sort_key)
                for key in ordered_keys:
                    if key in row['streak_groups'] and row['streak_groups'][key]:
                        stocks = row['streak_groups'][key]
                        names = '、'.join([f"{s['name']}({s['code']})" for s in stocks])
                        rows.append({'主题': theme, '几天几板': key, '股票名称': names})
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name='涨停金字塔', index=False)

        limit_down_df = signals['theme_limit_down']
        if not limit_down_df.empty:
            rows = []
            for _, row in limit_down_df.iterrows():
                theme = row['theme']
                ordered_keys = sorted(row['streak_groups'].keys(), key=_streak_sort_key)
                for key in ordered_keys:
                    if key in row['streak_groups'] and row['streak_groups'][key]:
                        stocks = row['streak_groups'][key]
                        names = '、'.join([f"{s['name']}({s['code']})" for s in stocks])
                        rows.append({'主题': theme, '几天几板': key, '股票名称': names})
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name='跌停金字塔', index=False)

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except Exception:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    return output_path


def print_signals_md(signals: dict) -> str:
    """生成 Markdown 格式信号报告"""
    lines = []
    period = signals['period']
    snapshot_date = signals['snapshot_date']

    lines.append(f"# 主题投资超短线策略 - 信号报告")
    lines.append(f"**周期**: {period} | **日期**: {snapshot_date}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 一、主题强度排名 TOP 10")
    lines.append("")
    theme_df = signals['theme_ranking']
    detail_df = signals['theme_concept_detail']
    if theme_df.empty:
        lines.append("*无数据*")
    else:
        lines.append(f"| 排名 | 主题 | 总Z | 股数 | 强度 | 覆盖 | 驱动概念 |")
        lines.append(f"| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for i, row in theme_df.head(10).iterrows():
            theme_name = row['theme']
            coverage = f"{row['anomalous_concept_count']}/{row['total_concept_count']}"

            if not detail_df.empty:
                theme_concepts = detail_df[detail_df['theme'] == theme_name].sort_values('normalized_strength', ascending=False)
                driver_concepts = []
                for _, cr in theme_concepts.head(3).iterrows():
                    pct = cr['anomalous_stock_count'] / cr['total_stock_count'] * 100 if cr['total_stock_count'] > 0 else 0
                    driver_concepts.append(f"{cr['concept']}({cr['anomalous_stock_count']}/{cr['total_stock_count']},{cr['normalized_strength']:.1f})")
                drivers_str = ' '.join(driver_concepts) if driver_concepts else ''
            else:
                drivers_str = ''

            lines.append(f"| {i+1} | {theme_name} | {row['total_zscore']:.1f} | {row['stock_count']} | {row['strength']:.2f} | {coverage} | {drivers_str} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 二、概念强度排名 TOP 20")
    lines.append("")
    concept_df = signals['concept_ranking']
    if concept_df.empty:
        lines.append("*无数据*")
    else:
        lines.append(f"| 排名 | 概念 | 主题 | 总Z | 异动股 | 总股 | 均Z | 强度 | 占比 |")
        lines.append(f"| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for i, row in concept_df.head(20).iterrows():
            pct = row['intensity'] * 100 if pd.notna(row.get('intensity')) else 0
            lines.append(f"| {i+1} | {row['concept']} | {row['theme']} | {row['total_zscore']:.2f} | {row['anomalous_stock_count']} | {row['total_stock_count']} | {row['avg_zscore']:.2f} | {row['normalized_strength']:.2f} | {pct:.1f}% |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 三、个股异动详情 TOP 20")
    lines.append("")
    stock_df = signals['stock_anomaly']
    if stock_df.empty:
        lines.append("*无数据*")
    else:
        lines.append(f"| 代码 | 名称 | Z-Score | 所属主题 |")
        lines.append(f"| --- | --- | ---: | --- |")
        for _, row in stock_df.iterrows():
            themes_str = row.get('themes_str', '')
            lines.append(f"| {row['code']} | {row['name']} | {row['zscore']:.2f} | {themes_str} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 四、交叉信号：强势主题 + 异常放量")
    lines.append("")
    cross_df = signals['cross_signals']
    if cross_df.empty:
        lines.append("*无数据*")
    else:
        lines.append(f"| 代码 | 名称 | Z-Score | 所属主题 |")
        lines.append(f"| --- | --- | ---: | --- |")
        for _, row in cross_df.iterrows():
            themes_str = row.get('themes_str', '')
            lines.append(f"| {row['code']} | {row['name']} | {row['zscore']:.2f} | {themes_str} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 五、涨停金字塔")
    lines.append("")
    limit_df = signals['theme_limit_up']
    if limit_df.empty:
        lines.append("*无涨停*")
    else:
        for _, row in limit_df.iterrows():
            theme = row['theme']
            stock_count = row['limit_up_stock_count']
            streak_groups = row['streak_groups']
            lines.append(f"### {theme}")
            lines.append(f"涨停股数: **{stock_count}** 只")
            lines.append("")
            ordered_keys = sorted(streak_groups.keys(), key=_streak_sort_key)
            has_content = False
            table_rows = []
            for key in ordered_keys:
                if key in streak_groups and streak_groups[key]:
                    stocks = streak_groups[key]
                    names = '、'.join([f"{s['name']}({s['code']})" for s in stocks])
                    table_rows.append((key, names))
                    has_content = True
            if has_content:
                lines.append(f"| 几天几板 | 股票名称 |")
                lines.append(f"| --- | --- |")
                for td, names in table_rows:
                    lines.append(f"| {td} | {names} |")
            else:
                lines.append("*无连板数据*")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 六、跌停金字塔")
    lines.append("")
    limit_down_df = signals['theme_limit_down']
    if limit_down_df.empty:
        lines.append("*无跌停*")
    else:
        for _, row in limit_down_df.iterrows():
            theme = row['theme']
            stock_count = row['limit_down_stock_count']
            streak_groups = row['streak_groups']
            lines.append(f"### {theme}")
            lines.append(f"跌停股数: **{stock_count}** 只")
            lines.append("")
            ordered_keys = sorted(streak_groups.keys(), key=_streak_sort_key)
            has_content = False
            table_rows = []
            for key in ordered_keys:
                if key in streak_groups and streak_groups[key]:
                    stocks = streak_groups[key]
                    names = '、'.join([f"{s['name']}({s['code']})" for s in stocks])
                    table_rows.append((key, names))
                    has_content = True
            if has_content:
                lines.append(f"| 几天几板 | 股票名称 |")
                lines.append(f"| --- | --- |")
                for td, names in table_rows:
                    lines.append(f"| {td} | {names} |")
            else:
                lines.append("*无连板数据*")
            lines.append("")

    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='主题投资信号')
    parser.add_argument('--period', type=str, default='daily', choices=['daily', 'weekly', 'min60', 'min15'])
    parser.add_argument('--date', type=str, default=None, help='日期截面，如 2026-03-10')
    parser.add_argument('--zscore', type=float, default=2.0)
    parser.add_argument('--top-themes', type=int, default=10)
    parser.add_argument('--top-stocks', type=int, default=20)
    parser.add_argument('--top-concepts', type=int, default=20)
    parser.add_argument('--md', action='store_true', help='输出Markdown格式')
    parser.add_argument('--excel', action='store_true', help='输出Excel格式')
    parser.add_argument('--db', action='store_true', help='输出到数据库')
    parser.add_argument('--output', type=str, default='../review', help='输出目录')
    args = parser.parse_args()

    signals = generate_signals(
        period=args.period,
        snapshot_date=args.date,
        zscore_threshold=args.zscore,
        top_n_themes=args.top_themes,
        top_n_stocks=args.top_stocks,
        top_n_concepts=args.top_concepts
    )

    if args.db:
        save_signals_to_database(signals)
        print("数据库写入完成")

    if args.md:
        print(print_signals_md(signals))
    elif args.excel:
        output_path = export_signals_excel(signals, output_dir=args.output)
        print(f"Excel输出完成: {output_path}")
    else:
        print_signals(signals)