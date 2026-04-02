"""
Purpose: 筛选最近N个交易日内涨停且财务总分大于阈值的股票，输出到自选股表
Inputs: 数据库表(limit_up_signals, stock_financial_score_pool, stock_k_data)
Outputs: stock_watchlist 表
Side Effects: 清空并写入 stock_watchlist 表

Usage:
    # 使用指定日期，筛选51个交易日内涨停且总分>65的股票
    python limited_up_seletion.py --date 2026-03-20

    # 使用最新日期
    python limited_up_seletion.py --date latest

    # 自定义参数
    python limited_up_seletion.py --date 2026-03-20 --trading-days 51 --score-threshold 65

Examples:
    python limited_up_seletion.py --date 2026-03-20
    python limited_up_seletion.py --date latest --trading-days 30 --score-threshold 70
"""
import sys
import os
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text

from datasource.database import get_session, truncate_table, bulk_insert


def get_latest_snapshot_date(session) -> str:
    """获取 limit_up_signals 表中最新的 snapshot_date"""
    sql = "SELECT MAX(snapshot_date) as max_date FROM limit_up_signals"
    result = session.execute(text(sql))
    row = result.fetchone()
    return row[0] if row and row[0] else None


def get_trading_dates(session, end_date: str, n: int = 51) -> list:
    """获取指定日期前 n 个交易日的日期列表"""
    sql = """
        SELECT DISTINCT DATE(bar_time) as trade_date
        FROM stock_k_data
        WHERE freq = 'd'
        AND DATE(bar_time) <= :end_date
        ORDER BY DATE(bar_time) DESC
        LIMIT :n
    """
    result = session.execute(text(sql), {"end_date": end_date, "n": n})
    dates = [row[0] for row in result.fetchall()]
    return sorted(dates) if dates else []


def select_limit_up_stocks(session, start_date: str, end_date: str) -> pd.DataFrame:
    """筛选指定日期范围内出现过涨停的股票"""
    sql = """
        SELECT DISTINCT ts_code, stock_name
        FROM limit_up_signals
        WHERE signal_type = 'limit_up'
        AND snapshot_date >= :start_date
        AND snapshot_date <= :end_date
    """
    df = pd.read_sql(text(sql), session.bind, params={"start_date": start_date, "end_date": end_date})
    return df


def get_latest_financial_scores(session, ts_codes: list) -> pd.DataFrame:
    """获取指定股票的最新一期财务评分"""
    if not ts_codes:
        return pd.DataFrame()

    placeholders = ','.join([f":code_{i}" for i in range(len(ts_codes))])
    params = {f"code_{i}": code for i, code in enumerate(ts_codes)}

    sql = f"""
        SELECT ts_code, stock_name, total_score, report_date
        FROM stock_financial_score_pool
        WHERE (ts_code, report_date) IN (
            SELECT ts_code, MAX(report_date)
            FROM stock_financial_score_pool
            GROUP BY ts_code
        )
        AND ts_code IN ({placeholders})
    """
    df = pd.read_sql(text(sql), session.bind, params=params)
    return df


def select_stocks_by_score(df: pd.DataFrame, threshold: float = 65.0) -> pd.DataFrame:
    """筛选财务总分大于阈值的股票"""
    if df.empty:
        return df
    return df[df['total_score'] > threshold].copy()


def save_to_watchlist(session, df: pd.DataFrame) -> int:
    """清空并写入自选股表，返回写入数量"""
    if df.empty:
        truncate_table(session, 'stock_watchlist')
        return 0

    truncate_table(session, 'stock_watchlist')

    df = df.sort_values('total_score', ascending=False).reset_index(drop=True)

    watchlist_df = pd.DataFrame({
        'ts_code': df['ts_code'],
        'name': df['stock_name'],
        'sort_order': range(len(df)),
        'added_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    bulk_insert(session, 'stock_watchlist', watchlist_df)
    return len(watchlist_df)


def main():
    parser = argparse.ArgumentParser(description='涨停股票筛选器')
    parser.add_argument('--date', type=str, default='latest',
                        help='截面日期，如 2026-03-20 或 latest')
    parser.add_argument('--trading-days', type=int, default=51,
                        help='回看的交易日数量，默认51')
    parser.add_argument('--score-threshold', type=float, default=65.0,
                        help='财务总分阈值，默认65')

    args = parser.parse_args()

    with get_session() as session:
        if args.date == 'latest':
            snapshot_date = get_latest_snapshot_date(session)
            if not snapshot_date:
                print("错误：limit_up_signals 表中没有数据")
                return
            print(f"使用最新截面日期: {snapshot_date}")
        else:
            snapshot_date = args.date

        print("=" * 60)
        print(f"Step 1/4: 获取截面日期前 {args.trading_days} 个交易日")
        print("=" * 60)
        trading_dates = get_trading_dates(session, snapshot_date, n=args.trading_days)
        if len(trading_dates) < args.trading_days:
            print(f"警告：只找到 {len(trading_dates)} 个交易日（请求 {args.trading_days} 个）")
        if not trading_dates:
            print("错误：未找到交易日数据")
            return
        start_date = trading_dates[0]
        end_date = trading_dates[-1]
        print(f"  交易日范围: {start_date} ~ {end_date} (共 {len(trading_dates)} 天)")

        print("=" * 60)
        print("Step 2/4: 筛选涨停股票")
        print("=" * 60)
        limit_up_df = select_limit_up_stocks(session, start_date, end_date)
        print(f"  找到 {len(limit_up_df)} 只在近 {args.trading_days} 个交易日涨停过的股票")

        print("=" * 60)
        print("Step 3/4: 获取财务评分并筛选")
        print("=" * 60)
        if limit_up_df.empty:
            print("  无涨停股票，跳过")
            watchlist_count = save_to_watchlist(session, pd.DataFrame())
        else:
            ts_codes = limit_up_df['ts_code'].tolist()
            scores_df = get_latest_financial_scores(session, ts_codes)
            print(f"  获取到 {len(scores_df)} 只股票的财务评分")

            filtered_df = select_stocks_by_score(scores_df, threshold=args.score_threshold)
            print(f"  财务总分 > {args.score_threshold}: {len(filtered_df)} 只")

            print("=" * 60)
            print("Step 4/4: 写入自选股表")
            print("=" * 60)
            watchlist_count = save_to_watchlist(session, filtered_df)

        print(f"\n完成！自选股表已更新，共 {watchlist_count} 只股票")

        if watchlist_count > 0:
            print("\n自选股列表:")
            result = session.execute(text("SELECT ts_code, name FROM stock_watchlist ORDER BY sort_order"))
            for row in result.fetchall():
                print(f"  {row[0]} - {row[1]}")


if __name__ == '__main__':
    main()