"""
Purpose: 回填 stock_anomaly_signals 和 stock_anomaly_signals_rolling 表的 price_change 字段

Usage:
    python signals_backfill_price_change.py --table stock_anomaly_signals_rolling
    python signals_backfill_price_change.py --table stock_anomaly_signals
    python signals_backfill_price_change.py --all

Examples:
    python signals_backfill_price_change.py --all --batch 500
"""
import sys
import os

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base_dir)
os.chdir(_base_dir)

from datasource.database import get_session
from sqlalchemy import text


def get_stats(session, table_name: str) -> dict:
    total = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    filled = session.execute(
        text(f"SELECT COUNT(*) FROM {table_name} WHERE price_change IS NOT NULL")
    ).scalar()
    return {'total': total, 'filled': filled, 'null': total - filled}


def backfill_price_change(table_name: str, batch_size: int = 500) -> dict:
    """
    批量回填 price_change
    每次取 batch_size 个 code，从 stock_k_data 批量取出所有历史 close，
    再用 Python 计算前一交易日，批量 UPDATE
    """
    updated = 0
    errors = 0

    with get_session() as session:
        stats = get_stats(session, table_name)
        print(f"  表 {table_name}: 总 {stats['total']} 条, 已填充 {stats['filled']} 条, 待填充 {stats['null']} 条")
        if stats['null'] == 0:
            print("  无需回填")
            return {'updated': 0, 'errors': 0}

        iteration = 0
        while True:
            rows = session.execute(
                text(f"""
                    SELECT DISTINCT code, snapshot_date
                    FROM {table_name}
                    WHERE price_change IS NULL
                    LIMIT :batch
                """),
                {'batch': batch_size}
            ).fetchall()

            if not rows:
                break

            iteration += 1
            code_dates = [(r.code, str(r.snapshot_date)[:10]) for r in rows]
            codes = list({r[0] for r in code_dates})
            date_min = min(r[1] for r in code_dates)
            date_max = max(r[1] for r in code_dates)

            placeholders = ', '.join([f':c{i}' for i in range(len(codes))])
            kdata = session.execute(
                text(f"""
                    SELECT ts_code, date(bar_time) as dt, close
                    FROM stock_k_data
                    WHERE freq = 'd'
                    AND ts_code IN ({placeholders})
                    ORDER BY ts_code, bar_time
                """),
                {f'c{i}': c for i, c in enumerate(codes)}
            ).fetchall()

            close_map = {}
            for (ts_code, dt, close) in kdata:
                key = (ts_code, dt)
                if key not in close_map:
                    close_map[key] = close

            prev_close_map = {}
            ts_dates = {}
            for (ts_code, dt) in close_map.keys():
                if ts_code not in ts_dates:
                    ts_dates[ts_code] = []
                ts_dates[ts_code].append(dt)

            for ts_code, dates in ts_dates.items():
                dates_sorted = sorted(dates)
                for i, dt in enumerate(dates_sorted):
                    if i > 0:
                        prev_close_map[(ts_code, dt)] = close_map.get((ts_code, dates_sorted[i - 1]))

            batch_updated = 0
            for code, dt in code_dates:
                today_close = close_map.get((code, dt))
                prev_close = prev_close_map.get((code, dt))
                if today_close is None or prev_close is None or prev_close <= 0:
                    errors += 1
                    continue
                price_change = round((today_close - prev_close) / prev_close * 100, 2)
                result = session.execute(
                    text(f"""
                        UPDATE {table_name}
                        SET price_change = :pc
                        WHERE code = :code
                          AND date(snapshot_date) = :dt
                          AND price_change IS NULL
                    """),
                    {'pc': price_change, 'code': code, 'dt': dt}
                )
                if result.rowcount > 0:
                    batch_updated += 1

            session.commit()
            updated += batch_updated
            print(f"  批次 {iteration}: {len(rows)} 组合, 更新 {batch_updated} 条")

    return {'updated': updated, 'errors': errors}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='回填 price_change 字段')
    parser.add_argument('--table', type=str,
                        choices=['stock_anomaly_signals', 'stock_anomaly_signals_rolling'],
                        help='指定要回填的表')
    parser.add_argument('--all', action='store_true', help='同时回填两个表')
    parser.add_argument('--batch', type=int, default=500, help='每批处理组合数')
    args = parser.parse_args()

    if not args.table and not args.all:
        print("请指定 --table 或 --all")
        parser.print_help()
        sys.exit(1)

    tables = ['stock_anomaly_signals', 'stock_anomaly_signals_rolling'] if args.all else [args.table]

    for tbl in tables:
        print(f"\n{'='*50}\n开始回填: {tbl}\n{'='*50}")
        result = backfill_price_change(tbl, batch_size=args.batch)
        print(f"  完成: 更新 {result['updated']} 条, 错误 {result['errors']} 条")

    print("\n全部完成")
