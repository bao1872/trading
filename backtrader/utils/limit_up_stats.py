"""
Purpose: 涨跌停统计 - 涨停/跌停判定与连板计算
Inputs: 日线数据
Outputs: 涨停/跌停标记、连板统计
"""
import pandas as pd
import numpy as np


def get_limit_up_price(close_price: float, code: str = None) -> float:
    """
    计算涨停价（基于前收盘），取2位小数避免浮点数精度问题
    """
    if pd.isna(close_price) or close_price <= 0:
        return np.nan
    if code:
        code_str = str(code)
        if 'ST' in code_str or code_str.startswith('4'):
            return round(close_price * 1.05, 2)
        if code_str.startswith('688') or code_str.startswith('30'):
            return round(close_price * 1.20, 2)
        if code_str.startswith('8') or code_str.startswith('4'):
            return round(close_price * 1.30, 2)
    return round(close_price * 1.10, 2)


def get_limit_down_price(close_price: float, code: str = None) -> float:
    """
    计算跌停价（基于前收盘），取2位小数避免浮点数精度问题
    """
    if pd.isna(close_price) or close_price <= 0:
        return np.nan
    if code:
        code_str = str(code)
        if 'ST' in code_str or code_str.startswith('4'):
            return round(close_price * 0.95, 2)
        if code_str.startswith('688') or code_str.startswith('30'):
            return round(close_price * 0.80, 2)
        if code_str.startswith('8') or code_str.startswith('4'):
            return round(close_price * 0.70, 2)
    return round(close_price * 0.90, 2)


def calculate_consecutive_details(limit_up_indices, full_indices, max_gap: int = 4) -> dict:
    """
    计算连板详细信息，允许中间断开最多max_gap-1天

    Args:
        limit_up_indices: 涨停日在统计窗口内的index列表（已排序）
        full_indices: 统计窗口内的所有index（已排序）
        max_gap: 允许的最大间隔天数，默认4（即断3天仍连续，断4天才断开）

    Returns:
        dict with streak_count, trading_days, all_streaks
    """
    if len(limit_up_indices) == 0:
        return {'streak_count': 0, 'trading_days': 0, 'all_streaks': []}

    full_indices_list = list(full_indices)
    limit_up_positions = []
    for lu_idx in limit_up_indices:
        if lu_idx in full_indices_list:
            limit_up_positions.append(full_indices_list.index(lu_idx))

    if not limit_up_positions:
        return {'streak_count': 0, 'trading_days': 0, 'all_streaks': []}

    streaks = []
    current_streak_start_pos = limit_up_positions[0]
    current_streak_count = 1
    last_limit_up_pos = limit_up_positions[0]

    for i in limit_up_positions[1:]:
        gap = i - last_limit_up_pos - 1
        if gap < max_gap:
            current_streak_count += 1
        else:
            trading_days = last_limit_up_pos - current_streak_start_pos + 1
            streaks.append({
                'count': current_streak_count,
                'trading_days': trading_days,
                'start_idx': current_streak_start_pos,
                'end_idx': last_limit_up_pos
            })
            current_streak_start_pos = i
            current_streak_count = 1
        last_limit_up_pos = i

    if current_streak_count > 0:
        trading_days = last_limit_up_pos - current_streak_start_pos + 1
        streaks.append({
            'count': current_streak_count,
            'trading_days': trading_days,
            'start_idx': current_streak_start_pos,
            'end_idx': last_limit_up_pos
        })

    streaks.sort(key=lambda x: -x['count'])

    return {
        'streak_count': streaks[0]['count'] if streaks else 0,
        'trading_days': streaks[0]['trading_days'] if streaks else 0,
        'all_streaks': streaks
    }


def calculate_limit_up_info_streaming(freq: str, window: int = 10, lookback: int = 30, snapshot_date: str = None) -> pd.DataFrame:
    """
    迭代器逐股票计算涨停/连板信息

    Args:
        freq: 周期 ('daily', 'weekly', 'min60', 'min15')
        window: 统计窗口
        lookback: 回看天数
        snapshot_date: 指定日期截面，默认None取最新
    """
    from datasource.k_data_loader import iter_k_data_with_names
    results = []

    target_date = None
    target_date_start = None
    target_date_end = None
    if snapshot_date:
        target_date = pd.Timestamp(snapshot_date)
        target_date_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        target_date_end = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    for code, name, df in iter_k_data_with_names(freq=freq):
        if len(df) < lookback + 1:
            continue

        df = df.tail(lookback + 1).copy()
        df['prev_close'] = df['close'].shift(1)

        code_str = str(code)
        if 'ST' in code_str or code_str.startswith('4'):
            limit_mult = 1.05
        elif code_str.startswith('688') or code_str.startswith('30'):
            limit_mult = 1.20
        elif code_str.startswith('8') or code_str.startswith('4'):
            limit_mult = 1.30
        else:
            limit_mult = 1.10

        df['limit_up_price'] = (df['prev_close'] * limit_mult).round(2)
        df['is_limit_up'] = df['high'] >= df['limit_up_price']
        df['is_limit_up_close'] = df['close'] >= df['limit_up_price']

        recent = df.tail(window)
        recent_indices = recent.index.tolist()

        if target_date:
            actual_date = None
            for idx in recent_indices:
                if target_date_start <= idx <= target_date_end:
                    actual_date = idx
                    break
            if actual_date is None:
                continue
            if not recent.loc[actual_date, 'is_limit_up_close']:
                continue
            limit_up_in_window = [d for d in recent.index if d <= actual_date and recent.loc[d, 'is_limit_up_close']]
        else:
            limit_up_in_window = recent[recent['is_limit_up_close']].index.tolist()

        if len(limit_up_in_window) == 0:
            continue

        all_recent_dates = sorted([str(d) for d in limit_up_in_window])

        if target_date:
            actual_date_idx = recent_indices.index(actual_date) if actual_date in recent_indices else -1
            if actual_date_idx >= 0:
                window_indices = recent_indices[:actual_date_idx + 1]
                limit_up_before_target = [x for x in limit_up_in_window if x <= actual_date]
                consecutive_info = calculate_consecutive_details(
                    limit_up_before_target, window_indices, max_gap=4
                )
            else:
                consecutive_info = {'streak_count': 0, 'trading_days': 0, 'all_streaks': []}
        else:
            consecutive_info = calculate_consecutive_details(
                limit_up_in_window, recent_indices, max_gap=4
            )

        streak_count = consecutive_info['streak_count']
        streak_trading_days = consecutive_info['trading_days']

        results.append({
            'code': code,
            'name': name,
            'limit_up_count': len(limit_up_in_window),
            'streak_count': streak_count,
            'streak_trading_days': streak_trading_days,
            'consecutive_details': consecutive_info,
            'last_limit_up_date': str(limit_up_in_window[-1]),
            'limit_up_dates': all_recent_dates
        })

    return pd.DataFrame(results)


def calculate_limit_down_info_streaming(freq: str, window: int = 10, lookback: int = 30, snapshot_date: str = None) -> pd.DataFrame:
    """
    迭代器逐股票计算跌停信息

    Args:
        freq: 周期 ('daily', 'weekly', 'min60', 'min15')
        window: 统计窗口
        lookback: 回看天数
        snapshot_date: 指定日期截面，默认None取最新
    """
    from datasource.k_data_loader import iter_k_data_with_names
    results = []

    target_date = None
    target_date_start = None
    target_date_end = None
    if snapshot_date:
        target_date = pd.Timestamp(snapshot_date)
        target_date_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        target_date_end = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    for code, name, df in iter_k_data_with_names(freq=freq):
        if len(df) < lookback + 1:
            continue

        df = df.tail(lookback + 1).copy()
        df['prev_close'] = df['close'].shift(1)

        code_str = str(code)
        if 'ST' in code_str or code_str.startswith('4'):
            limit_mult = 0.95
        elif code_str.startswith('688') or code_str.startswith('30'):
            limit_mult = 0.80
        elif code_str.startswith('8') or code_str.startswith('4'):
            limit_mult = 0.70
        else:
            limit_mult = 0.90

        df['limit_down_price'] = (df['prev_close'] * limit_mult).round(2)
        df['is_limit_down'] = df['low'] <= df['limit_down_price']
        df['is_limit_down_close'] = df['close'] <= df['limit_down_price']

        recent = df.tail(window)
        recent_indices = recent.index.tolist()

        if target_date:
            actual_date = None
            for idx in recent_indices:
                if target_date_start <= idx <= target_date_end:
                    actual_date = idx
                    break
            if actual_date is None:
                continue
            if not recent.loc[actual_date, 'is_limit_down_close']:
                continue
            limit_down_in_window = [d for d in recent.index if d <= actual_date and recent.loc[d, 'is_limit_down_close']]
        else:
            limit_down_in_window = recent[recent['is_limit_down_close']].index.tolist()

        if len(limit_down_in_window) == 0:
            continue

        all_recent_dates = sorted([str(d) for d in limit_down_in_window])

        if target_date:
            actual_date_idx = recent_indices.index(actual_date) if actual_date in recent_indices else -1
            if actual_date_idx >= 0:
                window_indices = recent_indices[:actual_date_idx + 1]
                limit_down_before_target = [x for x in limit_down_in_window if x <= actual_date]
                consecutive_info = calculate_consecutive_details(
                    limit_down_before_target, window_indices, max_gap=4
                )
            else:
                consecutive_info = {'streak_count': 0, 'trading_days': 0, 'all_streaks': []}
        else:
            consecutive_info = calculate_consecutive_details(
                limit_down_in_window, recent_indices, max_gap=4
            )

        streak_count = consecutive_info['streak_count']
        streak_trading_days = consecutive_info['trading_days']

        results.append({
            'code': code,
            'name': name,
            'limit_down_count': len(limit_down_in_window),
            'streak_count': streak_count,
            'streak_trading_days': streak_trading_days,
            'consecutive_details': consecutive_info,
            'last_limit_down_date': str(limit_down_in_window[-1]),
            'limit_down_dates': all_recent_dates
        })

    return pd.DataFrame(results)


def calculate_limit_down_info(daily_data: dict, window: int = 10, lookback: int = 30, snapshot_date: str = None) -> pd.DataFrame:
    """
    计算所有股票的跌停信息

    Args:
        daily_data: 数据字典
        window: 统计窗口
        lookback: 回看天数
        snapshot_date: 指定日期截面，默认None取最新
    """
    results = []

    target_date = None
    target_date_start = None
    target_date_end = None
    if snapshot_date:
        target_date = pd.Timestamp(snapshot_date)
        target_date_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        target_date_end = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    for code, info in daily_data.items():
        df = info['data'].copy()
        if len(df) < lookback + 1:
            continue

        df = df.tail(lookback + 1)
        df['prev_close'] = df['close'].shift(1)
        df['limit_down_price'] = df.apply(
            lambda x: get_limit_down_price(x['prev_close'], code), axis=1
        )
        df['is_limit_down'] = df['low'] <= df['limit_down_price']
        df['is_limit_down_close'] = df['close'] <= df['limit_down_price']

        recent = df.tail(window)
        recent_indices = recent.index.tolist()

        if target_date:
            actual_date = None
            for idx in recent_indices:
                if target_date_start <= idx <= target_date_end:
                    actual_date = idx
                    break
            if actual_date is None:
                continue
            if not recent.loc[actual_date, 'is_limit_down_close']:
                continue
            limit_down_in_window = [d for d in recent.index if d <= actual_date and recent.loc[d, 'is_limit_down_close']]
        else:
            limit_down_in_window = recent[recent['is_limit_down_close']].index.tolist()

        if len(limit_down_in_window) == 0:
            continue

        all_recent_dates = sorted([str(d) for d in limit_down_in_window])

        if target_date:
            actual_date_idx = recent_indices.index(actual_date) if actual_date in recent_indices else -1
            if actual_date_idx >= 0:
                window_indices = recent_indices[:actual_date_idx + 1]
                limit_down_before_target = [x for x in limit_down_in_window if x <= actual_date]
                consecutive_info = calculate_consecutive_details(
                    limit_down_before_target, window_indices, max_gap=4
                )
            else:
                consecutive_info = {'streak_count': 0, 'trading_days': 0, 'all_streaks': []}
        else:
            consecutive_info = calculate_consecutive_details(
                limit_down_in_window, recent_indices, max_gap=4
            )

        streak_count = consecutive_info['streak_count']
        streak_trading_days = consecutive_info['trading_days']

        results.append({
            'code': code,
            'name': info['name'],
            'limit_down_count': len(limit_down_in_window),
            'streak_count': streak_count,
            'streak_trading_days': streak_trading_days,
            'consecutive_details': consecutive_info,
            'last_limit_down_date': str(limit_down_in_window[-1]),
            'limit_down_dates': all_recent_dates
        })

    return pd.DataFrame(results)


def aggregate_limit_up_by_theme(
    limit_up_df: pd.DataFrame,
    stock_to_concepts: dict,
    concept_to_theme: dict,
    excluded: set,
    theme_ranking: list = None
) -> pd.DataFrame:
    """
    按主题聚合涨停统计，每只股票只归类到最强主题

    Args:
        limit_up_df: 涨停股票DataFrame
        stock_to_concepts: 股票到概念的映射
        concept_to_theme: 概念到主题的映射
        excluded: 排除的概念集合
        theme_ranking: 主题排名列表，按强度从强到弱排序
    """
    if limit_up_df.empty:
        return pd.DataFrame()

    if theme_ranking is None:
        theme_ranking = []

    theme_limit_up = {}
    assigned_stocks = set()

    for _, row in limit_up_df.iterrows():
        code = row['code']
        if code in assigned_stocks:
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
            theme_limit_up[best_theme] = {
                'limit_up_stocks': [],
                'streak_groups': {}
            }

        assigned_stocks.add(code)

        theme_limit_up[best_theme]['limit_up_stocks'].append({
            'code': code,
            'name': row['name'],
            'limit_up_count': row['limit_up_count'],
            'streak_count': row['streak_count'],
            'streak_trading_days': row.get('streak_trading_days', 0),
            'last_limit_up_date': row['last_limit_up_date'],
            'limit_up_dates': row.get('limit_up_dates', [])
        })

        td = row.get('streak_trading_days', 0)
        sc = row['streak_count']
        key = f"{td}天{sc}板"

        if key not in theme_limit_up[best_theme]['streak_groups']:
            theme_limit_up[best_theme]['streak_groups'][key] = []
        theme_limit_up[best_theme]['streak_groups'][key].append({
            'code': code,
            'name': row['name'],
            'streak_count': sc,
            'streak_trading_days': td,
            'dates': row.get('limit_up_dates', [])
        })

    results = []
    for theme, info in theme_limit_up.items():
        total_stocks = len(info['limit_up_stocks'])

        streak_groups = {}
        for key, stocks in info['streak_groups'].items():
            streak_groups[key] = sorted(stocks, key=lambda x: -x['streak_count'])

        results.append({
            'theme': theme,
            'limit_up_stock_count': total_stocks,
            'streak_groups': streak_groups,
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
    """
    按主题聚合跌停统计，每只股票只归类到最强主题

    Args:
        limit_down_df: 跌停股票DataFrame
        stock_to_concepts: 股票到概念的映射
        concept_to_theme: 概念到主题的映射
        excluded: 排除的概念集合
        theme_ranking: 主题排名列表，按强度从强到弱排序
    """
    if limit_down_df.empty:
        return pd.DataFrame()

    if theme_ranking is None:
        theme_ranking = []

    theme_limit_down = {}
    assigned_stocks = set()

    for _, row in limit_down_df.iterrows():
        code = row['code']
        if code in assigned_stocks:
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
            theme_limit_down[best_theme] = {
                'limit_down_stocks': [],
                'streak_groups': {}
            }

        assigned_stocks.add(code)

        theme_limit_down[best_theme]['limit_down_stocks'].append({
            'code': code,
            'name': row['name'],
            'limit_down_count': row['limit_down_count'],
            'streak_count': row['streak_count'],
            'streak_trading_days': row.get('streak_trading_days', 0),
            'last_limit_down_date': row['last_limit_down_date'],
            'limit_down_dates': row.get('limit_down_dates', [])
        })

        td = row.get('streak_trading_days', 0)
        sc = row['streak_count']
        key = f"{td}天{sc}板"

        if key not in theme_limit_down[best_theme]['streak_groups']:
            theme_limit_down[best_theme]['streak_groups'][key] = []
        theme_limit_down[best_theme]['streak_groups'][key].append({
            'code': code,
            'name': row['name'],
            'streak_count': sc,
            'streak_trading_days': td,
            'dates': row.get('limit_down_dates', [])
        })

    results = []
    for theme, info in theme_limit_down.items():
        total_stocks = len(info['limit_down_stocks'])

        streak_groups = {}
        for key, stocks in info['streak_groups'].items():
            streak_groups[key] = sorted(stocks, key=lambda x: -x['streak_count'])

        results.append({
            'theme': theme,
            'limit_down_stock_count': total_stocks,
            'streak_groups': streak_groups,
            'stocks': info['limit_down_stocks']
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('limit_down_stock_count', ascending=False).reset_index(drop=True)
    return df


if __name__ == '__main__':
    from volume_anomaly import load_data_for_period

    daily_data = load_data_for_period('daily')

    limit_df = calculate_limit_up_info(daily_data, window=10, lookback=30)
    print(f"近期有涨停的股票: {len(limit_df)}")
    print(limit_df.sort_values('streak_count', ascending=False).head(10))
