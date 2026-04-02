#!/usr/bin/env python3
"""
Purpose
-------
批量回补 first_day_launch_scores 数据库：从起始日期到结束日期逐日计算并写入DB。

How to Run
----------
    python backtrader/backfill_launch_scores.py --start 2025-11-01 --end 2026-03-30
    python backtrader/backfill_launch_scores.py --start 2025-11-01 --end 2026-03-30 --dry-run  # 只查日期不跑
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from datasource.database import get_session
from sqlalchemy import text


def main():
    parser = argparse.ArgumentParser(description="批量回补首日启动评分数据")
    parser.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="只查日期，不执行计算")
    parser.add_argument("--pool", default="stock_concepts_cache.xlsx", help="股票池文件")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from first_day_launch_pytdx import main as score_main

    # 使用统一的 database 模块查询
    with get_session() as session:
        result = session.execute(
            text("""
                SELECT DISTINCT DATE(bar_time) as d
                FROM stock_k_data
                WHERE freq='d' AND bar_time >= :start AND bar_time <= :end || ' 23:59:59'
                ORDER BY d
            """),
            {"start": args.start, "end": args.end}
        )
        dates = [r[0] for r in result.fetchall()]

    print(f"待处理交易日数: {len(dates)}")
    print(f"起始: {dates[0] if dates else 'N/A'}")
    print(f"终止: {dates[-1] if dates else 'N/A'}")

    if args.dry_run:
        print("[Dry-run] 退出")
        return

    total = len(dates)
    ok_count = 0
    fail_count = 0

    for idx, date in enumerate(dates, start=1):
        argv = [
            "--date", date,
            "--pool", args.pool,
            "--output", f"/tmp/launch_{date}.csv",
            "--log-level", "WARNING",
        ]
        try:
            ret = score_main(argv)
            if ret == 0:
                ok_count += 1
                print(f"[{idx}/{total}] {date} OK")
            else:
                fail_count += 1
                print(f"[{idx}/{total}] {date} FAIL (exit={ret})")
        except Exception as exc:
            fail_count += 1
            print(f"[{idx}/{total}] {date} EXCEPTION: {exc}")

    print(f"\n完成: 成功 {ok_count}, 失败 {fail_count}, 总计 {total}")


if __name__ == "__main__":
    raise SystemExit(main())
