#!/usr/bin/env python3
"""
ATR Rope 选股动态退出回测统计

Purpose:
    对 ATR Rope 选股结果做动态退出回测，统计收益率分布、方差等指标。
    多头买入破趋势线止盈/止损，震荡买入 c_lo 止损 / 破趋势线止盈。

Inputs:
    atr_rope_selection (选股信号表)
    atr_rope_factors (因子数据表，由 00_build_factor_table.py 构建)
    stock_k_data (日线K线数据)

Outputs:
    atr_experiment/results/ 下的 CSV 文件 + 控制台报告

How to Run:
    python atr_experiment/01_backtest_stats.py
    python atr_experiment/01_backtest_stats.py --start-date 2024-01-01
    python atr_experiment/01_backtest_stats.py --start-date 2023-01-01 --end-date 2026-05-19

Examples:
    python atr_experiment/01_backtest_stats.py
    python atr_experiment/01_backtest_stats.py --start-date 2025-01-01

Side Effects:
    只写 CSV 文件到 atr_experiment/results/，不写数据库

================================================================================
【回测规则】

买入：
  - 买入价格 = 信号日次日开盘价
  - 信号来源：atr_rope_selection 表

卖出（动态退出）：
  - 多头场景（regime=多头）：
    close < rope → 次日开盘卖出（跌破趋势线）
  - 震荡场景（regime=震荡）：
    close < c_lo → 次日开盘卖出（跌破蓝色箱体下轨止损）
    close < rope → 次日开盘卖出（跌破趋势线止盈）
    震荡转多头后止损参考从 c_lo 切换为 rope
  - 无最大持有期限制

【分批次读取策略】
  - 信号按月份分批加载
  - 因子数据按 ts_code 列表分批查询（每批 100 只），处理完释放
  - 单次内存最多持有 100 只股票的因子数据
================================================================================
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from tabulate import tabulate

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100  # 每批查询的股票数量


def normalize_ts_code(ts_code: str) -> str:
    """标准化股票代码为6位纯数字"""
    return str(ts_code).strip().upper().split('.')[0]


def load_signals(start_date: str, end_date: str) -> pd.DataFrame:
    """从 atr_rope_selection 加载选股信号"""
    sql = text("""
        SELECT selection_date, signal_date, ts_code, stock_name,
               regime, regime_value, regime_strength, signal_type,
               rope_value, lower_value, upper_value,
               rope_dir, rope_dev_pct, rope_dev_atr,
               change_pct, vol_zscore, bbmacd_event
        FROM atr_rope_selection
        WHERE selection_date >= :start_date AND selection_date <= :end_date
        ORDER BY selection_date, ts_code
    """)
    df = pd.read_sql(sql, engine, params={'start_date': start_date, 'end_date': end_date})
    df['raw_code'] = df['ts_code'].apply(normalize_ts_code)
    df['selection_date'] = pd.to_datetime(df['selection_date'])
    return df


def load_factors_batch(ts_codes: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    分批加载因子数据（每批 100 只股票）

    Returns:
        DataFrame with columns: bar_time, ts_code, atr_rope_rope, atr_rope_c_lo,
        factor_atr_rope_regime, ...
    """
    all_dfs = []
    raw_codes = [normalize_ts_code(c) for c in ts_codes]
    unique_codes = list(set(raw_codes))

    for i in range(0, len(unique_codes), BATCH_SIZE):
        batch = unique_codes[i:i + BATCH_SIZE]
        placeholders = ', '.join([f"'{c}'" for c in batch])
        sql = text(f"""
            SELECT bar_time, ts_code,
                   atr_rope_rope, atr_rope_c_lo, atr_rope_c_hi,
                   factor_atr_rope_regime
            FROM atr_rope_factors
            WHERE ts_code IN ({placeholders})
            AND DATE(bar_time) >= :start_date AND DATE(bar_time) <= :end_date
            ORDER BY bar_time, ts_code
        """)
        # end_date 需要往后推 120 天以覆盖持有期
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=120)
        batch_df = pd.read_sql(sql, engine, params={
            'start_date': start_date,
            'end_date': end_dt.strftime('%Y-%m-%d')
        })
        all_dfs.append(batch_df)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df['bar_time'] = pd.to_datetime(df['bar_time'])
    df['ts_code'] = df['ts_code'].apply(normalize_ts_code)
    return df


def load_price_batch(ts_codes: list, start_date: str, end_date: str) -> pd.DataFrame:
    """分批加载价格数据"""
    all_dfs = []
    raw_codes = [normalize_ts_code(c) for c in ts_codes]
    unique_codes = list(set(raw_codes))

    for i in range(0, len(unique_codes), BATCH_SIZE):
        batch = unique_codes[i:i + BATCH_SIZE]
        code_conditions = []
        for code in batch:
            code_conditions.append(f"ts_code = '{code}' OR ts_code = '{code}.SH' OR ts_code = '{code}.SZ'")

        or_clause = " OR ".join([f"({c})" for c in code_conditions])
        sql = text(f"""
            SELECT bar_time, ts_code, open, high, low, close
            FROM stock_k_data
            WHERE freq = 'd' AND ({or_clause})
            AND DATE(bar_time) >= :start_date AND DATE(bar_time) <= :end_date
            ORDER BY bar_time, ts_code
        """)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=120)
        batch_df = pd.read_sql(sql, engine, params={
            'start_date': start_date,
            'end_date': end_dt.strftime('%Y-%m-%d')
        })
        all_dfs.append(batch_df)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df['bar_time'] = pd.to_datetime(df['bar_time'])
    df['ts_code'] = df['ts_code'].apply(normalize_ts_code)
    return df


def simulate_trade(signal: pd.Series, price_df: pd.DataFrame, factors_df: pd.DataFrame) -> dict:
    """
    模拟单笔交易的动态退出回测

    Args:
        signal: 选股信号（一行数据）
        price_df: 该股票的价格数据（已按日期排序）
        factors_df: 该股票的因子数据（已按日期排序）

    Returns:
        交易结果字典
    """
    raw_code = signal['raw_code']
    signal_date = signal['selection_date']

    # 筛选该股票的数据
    stock_price = price_df[price_df['ts_code'] == raw_code].sort_values('bar_time').set_index('bar_time')
    stock_factors = factors_df[factors_df['ts_code'] == raw_code].sort_values('bar_time').set_index('bar_time')

    if stock_price.empty or stock_factors.empty:
        return None

    # 确定买入日 = signal_date 的下一个交易日
    future_dates = stock_price.index[stock_price.index > signal_date]
    if future_dates.empty:
        return None
    buy_date = future_dates[0]
    buy_price = float(stock_price.loc[buy_date, 'open'])

    if buy_price <= 0:
        return None

    # 从买入日的下一个交易日开始逐日检查退出条件
    holding_dates = stock_price.index[stock_price.index > buy_date]

    sell_date = None
    sell_price = None
    exit_reason = None
    original_regime = signal['regime']

    for dt in holding_dates:
        if dt not in stock_factors.index:
            continue

        close = float(stock_price.loc[dt, 'close'])
        rope = float(stock_factors.loc[dt, 'atr_rope_rope']) if pd.notna(stock_factors.loc[dt, 'atr_rope_rope']) else None
        c_lo = float(stock_factors.loc[dt, 'atr_rope_c_lo']) if pd.notna(stock_factors.loc[dt, 'atr_rope_c_lo']) else None
        regime_val = stock_factors.loc[dt, 'factor_atr_rope_regime']
        current_regime = int(regime_val) if pd.notna(regime_val) else None

        if rope is None:
            continue

        if original_regime == '多头':
            # 多头：close < rope 则退出
            if close < rope:
                # 次日开盘卖出
                next_dates = stock_price.index[stock_price.index > dt]
                if next_dates.empty:
                    sell_date = dt
                    sell_price = close
                else:
                    sell_date = next_dates[0]
                    sell_price = float(stock_price.loc[sell_date, 'open'])
                exit_reason = '跌破趋势线'
                break

        elif original_regime == '震荡':
            if c_lo is None:
                continue

            # 震荡转多头：止损参考从 c_lo 切换为 rope
            if current_regime == 1:  # 已转为多头
                if close < rope:
                    next_dates = stock_price.index[stock_price.index > dt]
                    if next_dates.empty:
                        sell_date = dt
                        sell_price = close
                    else:
                        sell_date = next_dates[0]
                        sell_price = float(stock_price.loc[sell_date, 'open'])
                    exit_reason = '趋势转多头后跌破趋势线'
                    break
            else:
                # 仍为震荡：先检查 c_lo 止损，再检查 rope 止盈
                if close < c_lo:
                    next_dates = stock_price.index[stock_price.index > dt]
                    if next_dates.empty:
                        sell_date = dt
                        sell_price = close
                    else:
                        sell_date = next_dates[0]
                        sell_price = float(stock_price.loc[sell_date, 'open'])
                    exit_reason = '跌破c_lo止损'
                    break
                elif close < rope:
                    next_dates = stock_price.index[stock_price.index > dt]
                    if next_dates.empty:
                        sell_date = dt
                        sell_price = close
                    else:
                        sell_date = next_dates[0]
                        sell_price = float(stock_price.loc[sell_date, 'open'])
                    exit_reason = '跌破趋势线止盈'
                    break

    if sell_date is None:
        # 未触发退出（数据截止），用最后一天收盘价平仓
        last_date = stock_price.index[-1]
        sell_date = last_date
        sell_price = float(stock_price.loc[last_date, 'close'])
        exit_reason = '数据截止'

    return_pct = (sell_price - buy_price) / buy_price
    holding_days = (sell_date - buy_date).days

    return {
        'ts_code': signal['ts_code'],
        'stock_name': signal.get('stock_name', ''),
        'selection_date': signal['selection_date'],
        'regime': original_regime,
        'signal_type': signal['signal_type'],
        'regime_strength': signal.get('regime_strength'),
        'buy_date': buy_date,
        'buy_price': buy_price,
        'sell_date': sell_date,
        'sell_price': sell_price,
        'return_pct': return_pct,
        'holding_days': holding_days,
        'exit_reason': exit_reason,
    }


def run_backtest(signals_df: pd.DataFrame, price_df: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
    """遍历所有信号执行回测"""
    results = []

    for idx, signal in signals_df.iterrows():
        result = simulate_trade(signal, price_df, factors_df)
        if result is not None:
            results.append(result)

    return pd.DataFrame(results)


def compute_statistics(trades_df: pd.DataFrame) -> dict:
    """计算回测统计指标"""
    if trades_df.empty:
        return {}

    stats = {}

    # 整体统计
    returns = trades_df['return_pct']
    stats['整体'] = _compute_group_stats(returns, trades_df)

    # 按 regime 分组
    for regime, group in trades_df.groupby('regime'):
        stats[f'Regime={regime}'] = _compute_group_stats(group['return_pct'], group)

    # 按 signal_type 分组
    for st, group in trades_df.groupby('signal_type'):
        stats[f'信号={st}'] = _compute_group_stats(group['return_pct'], group)

    # 按 regime × signal_type 交叉分组
    for (regime, st), group in trades_df.groupby(['regime', 'signal_type']):
        stats[f'{regime}-{st}'] = _compute_group_stats(group['return_pct'], group)

    # 退出原因分布
    exit_dist = trades_df['exit_reason'].value_counts()
    stats['退出原因分布'] = exit_dist.to_dict()

    return stats


def _compute_group_stats(returns: pd.Series, group: pd.DataFrame) -> dict:
    """计算单组的统计指标"""
    n = len(returns)
    if n == 0:
        return {}

    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    holding_days = group['holding_days']

    # 年化收益率（按平均持有天数折算，一年 242 个交易日）
    avg_holding = holding_days.mean()
    annualized = (1 + returns.mean()) ** (242 / avg_holding) - 1 if avg_holding > 0 else 0

    return {
        '样本数': n,
        '平均收益率': f"{returns.mean()*100:.2f}%",
        '中位数收益率': f"{returns.median()*100:.2f}%",
        '收益率标准差': f"{returns.std()*100:.2f}%",
        '收益率方差': f"{returns.var()*100:.4f}",
        '胜率': f"{(returns > 0).mean()*100:.1f}%",
        '盈亏比': f"{wins.mean()/abs(losses.mean()):.2f}" if len(losses) > 0 and len(wins) > 0 else "N/A",
        '平均持有天数': f"{avg_holding:.1f}",
        '中位持有天数': f"{holding_days.median():.0f}",
        'P10': f"{returns.quantile(0.1)*100:.2f}%",
        'P25': f"{returns.quantile(0.25)*100:.2f}%",
        'P50': f"{returns.quantile(0.5)*100:.2f}%",
        'P75': f"{returns.quantile(0.75)*100:.2f}%",
        'P90': f"{returns.quantile(0.9)*100:.2f}%",
        '最大盈利': f"{returns.max()*100:.2f}%",
        '最大亏损': f"{returns.min()*100:.2f}%",
        '年化收益率': f"{annualized*100:.2f}%",
        # raw values for CSV
        '_mean': returns.mean(),
        '_median': returns.median(),
        '_std': returns.std(),
        '_var': returns.var(),
        '_winrate': (returns > 0).mean(),
        '_annualized': annualized,
    }


def print_report(stats: dict):
    """打印回测报告"""
    print("\n" + "=" * 100)
    print("ATR Rope 选股动态退出回测报告")
    print("=" * 100)

    # 各组统计
    for group_name, group_stats in stats.items():
        if group_name == '退出原因分布':
            continue
        if isinstance(group_stats, dict) and '样本数' in group_stats:
            print(f"\n--- {group_name} ---")
            for k, v in group_stats.items():
                if not k.startswith('_'):
                    print(f"  {k}: {v}")

    # 退出原因分布
    if '退出原因分布' in stats:
        print("\n--- 退出原因分布 ---")
        total = sum(stats['退出原因分布'].values())
        for reason, count in stats['退出原因分布'].items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {reason}: {count} ({pct:.1f}%)")


def save_results(trades_df: pd.DataFrame, stats: dict, start_date: str, end_date: str):
    """保存回测结果到 CSV"""
    # 交易明细
    trades_df.to_csv(RESULTS_DIR / "backtest_trades.csv", index=False, encoding="utf-8-sig")

    # 统计摘要
    rows = []
    for group_name, group_stats in stats.items():
        if group_name == '退出原因分布':
            continue
        if isinstance(group_stats, dict) and '样本数' in group_stats:
            row = {'分组': group_name}
            for k, v in group_stats.items():
                if not k.startswith('_'):
                    row[k] = v
            rows.append(row)

    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False, encoding="utf-8-sig")

    # 退出原因分布
    if '退出原因分布' in stats:
        exit_df = pd.DataFrame([
            {'退出原因': k, '次数': v} for k, v in stats['退出原因分布'].items()
        ])
        exit_df.to_csv(RESULTS_DIR / "exit_reason_dist.csv", index=False, encoding="utf-8-sig")

    print(f"\n结果已保存到 {RESULTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description='ATR Rope 选股动态退出回测统计',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--start-date', default='2023-01-01', help='回测开始日期 (默认 2023-01-01)')
    parser.add_argument('--end-date', default='2026-05-19', help='回测结束日期 (默认 2026-05-19)')
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date

    print("=" * 100)
    print("ATR Rope 选股动态退出回测统计")
    print(f"  回测区间: {start_date} ~ {end_date}")
    print(f"  买入: 信号日次日开盘价")
    print(f"  多头退出: close < rope → 次日开盘卖出")
    print(f"  震荡退出: close < c_lo 止损 / close < rope 止盈")
    print(f"  震荡转多头后止损参考切为 rope")
    print(f"  无最大持有期限制")
    print(f"  分批策略: 按月份分批加载信号，每批独立加载价格和因子数据")
    print("=" * 100)

    # 1. 加载全部选股信号（信号数据量小，约4.5万行，可一次加载）
    print("\n[1/3] 加载选股信号...")
    signals_df = load_signals(start_date, end_date)
    print(f"  共 {len(signals_df)} 条信号, {signals_df['selection_date'].nunique()} 个交易日")

    if signals_df.empty:
        print("无选股信号数据，退出")
        return

    # 信号分布
    print(f"  Regime 分布: {dict(signals_df['regime'].value_counts())}")
    print(f"  信号类型分布: {dict(signals_df['signal_type'].value_counts())}")

    # 2. 按月份分批处理回测（控制内存）
    print("\n[2/3] 按月份分批执行回测...")
    signals_df['year_month'] = signals_df['selection_date'].dt.to_period('M')
    months = sorted(signals_df['year_month'].unique())

    all_trades = []

    for i, ym in enumerate(months):
        month_signals = signals_df[signals_df['year_month'] == ym]
        ts_codes = month_signals['ts_code'].unique().tolist()

        # 每月独立加载价格和因子数据（处理完释放）
        ym_start = ym.start_time.strftime('%Y-%m-%d')
        ym_end = ym.end_time.strftime('%Y-%m-%d')

        price_df = load_price_batch(ts_codes, ym_start, end_date)
        factors_df = load_factors_batch(ts_codes, ym_start, end_date)

        if factors_df.empty:
            print(f"  {ym}: 因子数据为空，跳过")
            continue

        month_trades = run_backtest(month_signals, price_df, factors_df)
        all_trades.append(month_trades)

        # 显式释放内存
        del price_df, factors_df

        if (i + 1) % 6 == 0 or i == len(months) - 1:
            print(f"  进度: {i+1}/{len(months)} 个月份, 累计 {sum(len(t) for t in all_trades)} 笔交易")

    if not all_trades:
        print("无回测结果")
        return

    trades_df = pd.concat(all_trades, ignore_index=True)
    print(f"  回测完成: {len(trades_df)} 笔交易")

    # 3. 统计分析
    print("\n[3/3] 统计分析...")
    stats = compute_statistics(trades_df)

    # 打印报告
    print_report(stats)

    # 保存结果
    save_results(trades_df, stats, start_date, end_date)


if __name__ == '__main__':
    main()
