"""
Purpose: 基于异动信号 + CHoCH 买点的回测
条件:
1. 5日滚动窗口zscore>=2.5
2. CV<=0.4 或 ρ>=0.8
3. CHoCH买点：异动日后60个日线bar内出现 bullish CHoCH（15m或60m）
4. 买入: CHoCH出现日的下一bar开盘价(滑点0.5%)
5. 计算未来3/5/10/20/30个bar的完整收益统计
6. 过滤涨跌停、停牌

Usage:
    python trading_entry.py --start 2025-11-03 --end 2026-03-20
"""
import os
import glob
import pickle
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

SLIPPAGE = 0.005
COMMISSION = 0.0003
LOCK_PERIOD_BARS = 60


def load_rolling_signals(signal_dir: str = '../review') -> pd.DataFrame:
    """加载所有 signal_ 异动结果"""
    all_signals = []
    pattern = os.path.join(signal_dir, 'signal_*.xlsx')

    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        date_str = fname.replace('signal_', '').replace('.xlsx', '')

        try:
            xl = pd.ExcelFile(fpath)
            if '个股异动' not in xl.sheet_names:
                continue

            df = pd.read_excel(fpath, sheet_name='个股异动')
            if df.empty:
                continue

            df['signal_date'] = date_str
            all_signals.append(df)
        except Exception as e:
            print(f"Warning: 读取{fname}失败: {e}")

    if not all_signals:
        return pd.DataFrame()

    combined = pd.concat(all_signals, ignore_index=True)
    combined['代码'] = combined['代码'].astype(str).str.zfill(6)
    return combined


def load_minute_data(period: str = '60m') -> dict:
    """加载分钟线数据集（从数据库）"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datasource.k_data_loader import load_k_data_as_dict

    freq_map = {'15m': 'min15', '60m': 'min60'}
    freq = freq_map.get(period)
    if not freq:
        raise ValueError(f"不支持的周期: {period}")

    return load_k_data_as_dict(freq=freq)


def get_smc_choch_times(df: pd.DataFrame, period: str) -> dict:
    """对 DataFrame 运行 SMC 分析，获取 CHoCH 时间点"""
    import argparse

    if df is None or len(df) < 50:
        return {'swing': [], 'internal': [], 'df': None}

    df = df.rename(columns={
        'open': 'open', 'high': 'high', 'low': 'low',
        'close': 'close', 'volume': 'volume'
    })
    if 'date' not in df.columns and df.index.name != 'date':
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'
    df = df.sort_index()

    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', default=period)
    parser.add_argument('--swings_length', type=int, default=20)
    parser.add_argument('--internal_filter_confluence', action='store_true', default=False)
    parser.add_argument('--show_internals', action='store_true', default=False)
    parser.add_argument('--show_structure', action='store_true', default=False)
    parser.add_argument('--show_swing_bull', default='All')
    parser.add_argument('--show_swing_bear', default='All')
    parser.add_argument('--show_internal_bull', default='All')
    parser.add_argument('--show_internal_bear', default='All')
    parser.add_argument('--show_equal_hl', action='store_true', default=False)
    parser.add_argument('--show_internal_order_blocks', action='store_true', default=False)
    parser.add_argument('--show_swing_order_blocks', action='store_true', default=False)
    parser.add_argument('--show_order_blocks', action='store_true', default=False)
    parser.add_argument('--show_fvg', action='store_true', default=False)
    parser.add_argument('--show_trend', action='store_true', default=False)
    parser.add_argument('--show_high_low_swings', action='store_true', default=False)
    parser.add_argument('--order_block_filter', default='Cumulative Mean Range')
    parser.add_argument('--style', default='Colored')
    parser.add_argument('--mode', default='Historical')
    parser.add_argument('--show_daily_levels', action='store_true', default=False)
    parser.add_argument('--show_weekly_levels', action='store_true', default=False)
    parser.add_argument('--show_monthly_levels', action='store_true', default=False)
    parser.add_argument('--show_zones', action='store_true', default=False)
    parser.add_argument('--swing_bull_color', default='#089981')
    parser.add_argument('--swing_bear_color', default='#F23645')
    parser.add_argument('--internal_bull_color', default='#089981')
    parser.add_argument('--internal_bear_color', default='#F23645')
    parser.add_argument('--fvg_bull_color', default='rgba(8,153,129,0.30)')
    parser.add_argument('--fvg_bear_color', default='rgba(242,54,69,0.30)')
    parser.add_argument('--premium_zone_color', default='#F23645')
    parser.add_argument('--discount_zone_color', default='#089981')
    parser.add_argument('--daily_levels_style', default='solid')
    parser.add_argument('--weekly_levels_style', default='dash')
    parser.add_argument('--monthly_levels_style', default='dotted')
    parser.add_argument('--daily_levels_color', default='#878991')
    parser.add_argument('--weekly_levels_color', default='#878991')
    parser.add_argument('--monthly_levels_color', default='#878991')
    parser.add_argument('--fvg_extend', type=int, default=3)
    parser.add_argument('--equal_length', type=int, default=5)
    parser.add_argument('--equal_threshold', type=float, default=0.0001)
    args = parser.parse_args([])

    args.show_internals = True
    args.show_structure = True
    args.show_swings = True
    args.show_internal_order_blocks = False
    args.show_swing_order_blocks = False
    args.show_order_blocks = False
    args.show_fvg = False
    args.show_trend = False
    args.show_high_low_swings = False
    args.show_equal_hl = False

    sys.path.insert(0, str(Path(__file__).parent.parent / 'features'))
    from smc_luxalgo_pytdx_plotly import SMCIndicatorPineCloser

    indicator = SMCIndicatorPineCloser(df, args)
    indicator.run()

    return {
        'swing': indicator.swing_bullish_choch_times,
        'internal': indicator.internal_bullish_choch_times,
        'df': df
    }


def find_first_choch_since(code: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
                           period: str, minute_data: dict, choch_cache: dict) -> tuple:
    """
    查找从start_date到end_date之间的第一个CHoCH
    返回: (choch_date, choch_type) 或 (None, None)
    """
    cache_key = f"{code}_{period}"
    if cache_key not in choch_cache:
        if code not in minute_data:
            choch_cache[cache_key] = {'swing': [], 'internal': []}
        else:
            df = minute_data[code]['data'].copy()
            result = get_smc_choch_times(df, period)
            choch_cache[cache_key] = {
                'swing': result.get('swing', []),
                'internal': result.get('internal', [])
            }

    swing_times = choch_cache[cache_key]['swing']
    internal_times = choch_cache[cache_key]['internal']

    for t in swing_times:
        if start_date < t <= end_date:
            return (t, 'swing')

    for t in internal_times:
        if start_date < t <= end_date:
            return (t, 'internal')

    return (None, None)


def get_nth_trading_date(dates: list, anchor_date: pd.Timestamp, n: int, direction: int = 1) -> pd.Timestamp:
    """获取anchor_date之后（或之前）第n个交易日"""
    anchor_str = str(anchor_date)[:10]
    date_strs = [str(d)[:10] for d in dates]

    if anchor_str not in date_strs:
        return None

    anchor_idx = date_strs.index(anchor_str)
    target_idx = anchor_idx + (n * direction)

    if 0 <= target_idx < len(dates):
        return dates[target_idx]
    return None


def backtest_with_choch_entry(
    daily_data: dict,
    signals: pd.DataFrame,
    start_date: str,
    end_date: str,
    cv_threshold: float = 0.4,
    spearman_threshold: float = 0.8,
    zscore_threshold: float = 2.5,
    holding_days_list: list = [3, 5, 10, 20, 30]
) -> pd.DataFrame:
    """基于信号 + CHoCH买点进行回测"""
    print("加载分钟线数据...")
    min15_data = load_minute_data('15m')
    min60_data = load_minute_data('60m')
    print(f"  15m 股票数: {len(min15_data)}")
    print(f"  60m 股票数: {len(min60_data)}")

    choch_cache = {}

    results = []
    sorted_dates = sorted(daily_data[list(daily_data.keys())[0]]['data'].index)
    start_date_str = start_date[:10] if isinstance(start_date, str) else str(start_date)[:10]
    end_date_str = end_date[:10] if isinstance(end_date, str) else str(end_date)[:10]

    filtered_dates = [d for d in sorted_dates if start_date_str <= str(d)[:10] <= end_date_str]

    total_signals = 0
    choch_triggered = 0
    no_choch_count = 0

    for i, current_date in enumerate(filtered_dates):
        snapshot_date = str(current_date)[:10]

        day_signals = signals[signals['signal_date'] == snapshot_date]
        if day_signals.empty:
            continue

        for _, row in day_signals.iterrows():
            code = row['代码']

            zscore = row.get('Z分数', None)
            if zscore is None or zscore < zscore_threshold:
                continue

            cv = row.get('成交量均匀度(CV)', None)
            spearman = row.get('放量顺序(ρ)', None)

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

            total_signals += 1
            signal_date = pd.Timestamp(snapshot_date)

            lock_end_date = get_nth_trading_date(sorted_dates, signal_date, LOCK_PERIOD_BARS, direction=1)
            if lock_end_date is None:
                no_choch_count += 1
                continue

            choch_date = None
            choch_type = None

            for period, minute_data in [('15m', min15_data), ('60m', min60_data)]:
                cd, ct = find_first_choch_since(
                    code, signal_date, lock_end_date, period, minute_data, choch_cache
                )
                if cd is not None:
                    choch_date = cd
                    choch_type = f"{period}_{ct}"
                    break

            if choch_date is None:
                no_choch_count += 1
                continue

            choch_triggered += 1

            choch_date_normalized = choch_date.normalize()
            buy_bar_date = get_nth_trading_date(sorted_dates, choch_date_normalized, 1, direction=1)
            if buy_bar_date is None:
                continue

            if code not in daily_data:
                continue

            df_daily = daily_data[code]['data']

            buy_price = None
            limit_up_today = False
            limit_down_today = False
            price_percentile = 50.0

            for idx in df_daily.index:
                if buy_bar_date.date() == idx.date():
                    prev_close = df_daily.loc[idx, 'close'] if idx > df_daily.index[0] else None

                    if 'open' in df_daily.columns:
                        open_price = df_daily.loc[idx, 'open']
                        high_price = df_daily.loc[idx, 'high']
                        low_price = df_daily.loc[idx, 'low']

                        limit_up_price = prev_close * 1.20 if prev_close else None
                        limit_down_price = prev_close * 0.80 if prev_close else None

                        if limit_up_price and high_price >= limit_up_price:
                            limit_up_today = True
                        if limit_down_price and low_price <= limit_down_price:
                            limit_down_today = True

                        if limit_up_today or limit_down_today:
                            break

                        buy_price = open_price * (1 + SLIPPAGE)

                        past_closes = df_daily.loc[:idx, 'close'].iloc[-256:-1]
                        if len(past_closes) > 0:
                            price_percentile = (past_closes < buy_price).sum() / len(past_closes) * 100
                    break

            if buy_price is None or buy_price <= 0:
                continue

            future_data = df_daily[df_daily.index > buy_bar_date]

            for holding_days in holding_days_list:
                if len(future_data) < holding_days:
                    continue

                period_closes = []
                period_lows = []

                for j, (idx, row_data) in enumerate(future_data.iterrows()):
                    if j >= holding_days:
                        break
                    period_closes.append(row_data['close'])
                    period_lows.append(row_data['low'])

                sell_price = period_closes[-1] * (1 - SLIPPAGE - COMMISSION)
                net_return = (sell_price - buy_price) / buy_price * 100

                max_price = max(period_closes)
                min_price = min(period_lows)
                max_return_val = (max_price - buy_price) / buy_price * 100
                max_drawdown = (buy_price - min_price) / buy_price * 100

                sell_date = future_data.index[holding_days - 1] if holding_days <= len(future_data) else future_data.index[-1]

                results.append({
                    'signal_date': snapshot_date,
                    'choch_date': str(choch_date)[:10],
                    'buy_date': str(buy_bar_date)[:10],
                    'sell_date': str(sell_date)[:10],
                    'holding_days': holding_days,
                    'code': code,
                    'name': row.get('名称', ''),
                    'zscore': zscore,
                    'cv': cv,
                    'spearman': spearman,
                    'choch_type': choch_type,
                    'theme': row.get('主题', ''),
                    'buy_price': round(buy_price, 2),
                    'price_percentile': round(price_percentile, 2),
                    'sell_price': round(sell_price, 2),
                    'net_return': round(net_return, 2),
                    'max_return': round(max_return_val, 2),
                    'max_drawdown': round(max_drawdown, 2),
                })

    print(f"\nCHoCH统计: 总信号={total_signals}, 触发CHoCH={choch_triggered}, 未触发={no_choch_count}")

    return pd.DataFrame(results)


def print_full_stats(results: pd.DataFrame, cv_th: float, spearman_th: float, zscore_th: float):
    """打印完整回测统计"""
    print("\n" + "=" * 80)
    print("回测条件")
    print("=" * 80)
    print(f"  总交易数: {len(results)}")
    print(f"  条件1: 5日滚动zscore>={zscore_th}")
    print(f"  条件2: CV<={cv_th} 或 ρ>={spearman_th}")
    print(f"  买点: 异动日后60日内CHoCH出现时买入(15m或60m)")
    print(f"  买入: CHoCH出现日下一bar开盘价(滑点0.5%)")
    print(f"  卖出: 持有期末收盘价(滑点0.5%+手续费0.03%)")
    print(f"  过滤: 涨跌停、停牌")

    for holding_days in [3, 5, 10, 20, 30]:
        period_results = results[results['holding_days'] == holding_days]

        print("\n" + "=" * 80)
        print(f"持有{holding_days}天统计")
        print("=" * 80)

        if period_results.empty:
            print("  无数据")
            continue

        print(f"\n【净收益 (扣除滑点和手续费)】")
        print(period_results['net_return'].describe())
        print(f"\n  盈利交易数: {(period_results['net_return'] > 0).sum()}")
        print(f"  亏损交易数: {(period_results['net_return'] <= 0).sum()}")
        print(f"  胜率: {(period_results['net_return'] > 0).mean() * 100:.2f}%")

        avg_win = period_results[period_results['net_return'] > 0]['net_return'].mean()
        avg_loss = period_results[period_results['net_return'] < 0]['net_return'].mean()
        print(f"  盈亏比: {abs(avg_win / avg_loss):.2f}" if avg_loss != 0 else "  盈亏比: N/A")

        print(f"\n  分位数:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            val = period_results['net_return'].quantile(p/100)
            print(f"    {p}%: {val:.2f}%")

        print(f"\n【最大涨幅 (Max Return)】")
        print(period_results['max_return'].describe())

        print(f"\n【最大回撤 (Max Drawdown)】")
        print(period_results['max_drawdown'].describe())


if __name__ == '__main__':
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / 'features'))

    parser = argparse.ArgumentParser(description='基于异动信号+CHoCH买点的回测')
    parser.add_argument('--start', type=str, default='2026-03-10', help='回测开始日期')
    parser.add_argument('--end', type=str, default='2026-03-20', help='回测结束日期')
    parser.add_argument('--cv', type=float, default=0.4, help='CV阈值')
    parser.add_argument('--spearman', type=float, default=0.8, help='Spearman阈值')
    parser.add_argument('--zscore', type=float, default=2.5, help='zscore阈值')
    args = parser.parse_args()

    print("加载日线异动信号...")
    from utils.volume_anomaly import load_data_for_period

    signals = load_rolling_signals()
    print(f"原始信号数: {len(signals)}")

    if signals.empty:
        print("没有找到异动信号文件")
        sys.exit(1)

    print("加载日线数据...")
    daily_data = load_data_for_period('daily')

    print("运行回测 (带CHoCH买点)...")
    results = backtest_with_choch_entry(
        daily_data=daily_data,
        signals=signals,
        start_date=args.start,
        end_date=args.end,
        cv_threshold=args.cv,
        spearman_threshold=args.spearman,
        zscore_threshold=args.zscore
    )

    if results.empty:
        print("没有符合条件的交易")
    else:
        print_full_stats(results, args.cv, args.spearman, args.zscore)

        output_path = '../review/trading_entry_results.xlsx'
        results.to_excel(output_path, index=False)
        print(f"\n详细结果已保存到: {output_path}")