#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票池财务评分全历史回补脚本

目的：
- 基于 batch_score.py 的核心计算逻辑（SSOT），对 stock_concepts_cache 中的全股票池
  计算并回填所有可用报告期的财务评分
- 结果写入 stock_financial_score_pool 表（按 ts_code + report_date upsert）

与 batch_score.py 的区别：
- batch_score.py 只输出每只股票的最新一期评分（用于日常增量更新）
- 本脚本输出所有历史报告期评分（用于历史回补）

使用方法：
1) 先清空现有数据（可选，或用 --clean 逐报告期清空）
   python -c "
   from sqlalchemy import create_engine, text
   from config import DATABASE_URL
   engine = create_engine(DATABASE_URL)
   with engine.connect() as conn:
       conn.execute(text('DELETE FROM stock_financial_score_pool'))
       conn.commit()
   print('已清空 stock_financial_score_pool')
   "

2) 全量回补（默认从 20220101 起始，读足够窗口）
   python financial_factors/batch_score_backfill.py

3) 测试模式（少量股票）
   python financial_factors/batch_score_backfill.py --limit 5

4) 断点续算（跳过已有报告期的股票）
   python financial_factors/batch_score_backfill.py --resume

输出：
- 终端打印每批次（每个报告期）的进度和统计
- 结果写入 stock_financial_score_pool 表（upsert，不重复则跳过）
"""

import os
import sys
import time
import argparse
import logging
import importlib.util

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasource.models import FINANCIAL_SCORE_POOL_TABLE
from config import DATABASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_URL = DATABASE_URL

SCORE_COLS = [
    "total_score",
    "规模与增长_score",
    "盈利能力_score",
    "利润质量_score",
    "现金创造能力_score",
    "资产效率与资金占用_score",
    "边际变化与持续性_score",
]
FACTOR_COLS = None


def _load_db_score():
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "financial_factors", "sample_score.py"
    )
    spec = importlib.util.spec_from_file_location("sample_score", db_path)
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    return sm


sm = _load_db_score()
fetch_financial_statements_from_db = sm.fetch_financial_from_db
prepare_base_dataframe = sm.prepare_base_dataframe
dedup_latest = sm.dedup_latest
add_ytd_and_ttm = sm.add_ytd_and_ttm
add_factors = sm.add_factors
score_dataframe = sm.score_dataframe
FACTOR_CONFIG = sm.FACTOR_CONFIG
FACTOR_COLS = [cfg["factor_name"] for cfg in FACTOR_CONFIG]


def clean_all(engine):
    with engine.connect() as conn:
        result = conn.execute(text("DELETE FROM stock_financial_score_pool"))
        conn.commit()
        return result.rowcount


def clean_report_date(engine, report_date: str) -> int:
    with engine.connect() as conn:
        result = conn.execute(
            text("DELETE FROM stock_financial_score_pool WHERE report_date = :rd"),
            {"rd": report_date}
        )
        conn.commit()
        return result.rowcount


def ensure_table_exists(engine):
    with engine.connect() as conn:
        if engine.dialect.name == "postgresql":
            result = conn.execute(
                text("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'stock_financial_score_pool'")
            )
            exists = result.fetchone()[0] > 0
        else:
            result = conn.execute(
                text("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = 'stock_financial_score_pool'")
            )
            exists = result.fetchone()[0] > 0

        if not exists:
            logger.info("创建表 stock_financial_score_pool ...")
            conn.execute(text(FINANCIAL_SCORE_POOL_TABLE.split(";")[0]))
            conn.commit()
            logger.info("表创建完成")
        else:
            logger.info("表 stock_financial_score_pool 已存在")


def fetch_stock_pool(engine, limit=None):
    sql = "SELECT ts_code, name FROM stock_pools ORDER BY ts_code"
    if limit:
        sql += f" LIMIT {limit}"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    logger.info(f"股票池共 {len(df)} 只")
    return df


def fetch_already_processed(engine, report_date):
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT ts_code FROM stock_financial_score_pool WHERE report_date = :rd"),
            conn, params={"rd": report_date}
        )
    return set(df["ts_code"].tolist())


def compute_all_scores(
    ts_code: str,
    name: str,
    start_date: str,
    lookback: int,
    limit_n_quarters: int = 20,
):
    try:
        df = prepare_base_dataframe(
            ts_code=ts_code,
            start_date=start_date,
        )

        if df.empty:
            return pd.DataFrame()

        df = add_ytd_and_ttm(df)
        df = add_factors(df)
        df = score_dataframe(df, lookback=lookback)

        return df

    except Exception as e:
        logger.warning(f"[{ts_code}] 计算异常: {e}")
        return pd.DataFrame()


def build_output_rows(scored: pd.DataFrame, ts_code: str, name: str) -> list:
    rows = []
    for _, latest in scored.iterrows():
        row = {
            "ts_code": ts_code,
            "stock_name": name,
            "report_date": latest["end_date"].strftime("%Y%m%d"),
        }
        for col in SCORE_COLS:
            if col in latest.index:
                v = latest[col]
                row[col] = None if pd.isna(v) else float(v)
        for col in FACTOR_COLS:
            if col in latest.index:
                v = latest[col]
                row[col] = None if pd.isna(v) else float(v)
        rows.append(row)
    return rows


def upsert_rows(engine, rows):
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    cols = list(df.columns)
    uniq = ["ts_code", "report_date"]
    non_key = [c for c in cols if c not in uniq]
    update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in non_key])
    col_clause = ", ".join(cols)
    placeholders = ", ".join([f":{c}" for c in cols])
    sql = f"""
        INSERT INTO stock_financial_score_pool ({col_clause})
        VALUES ({placeholders})
        ON CONFLICT ({', '.join(uniq)}) DO UPDATE SET {update_clause}
    """
    with engine.connect() as conn:
        for _, r in df.iterrows():
            conn.execute(text(sql), r.to_dict())
        conn.commit()
    return len(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="股票池财务评分全历史回补")
    parser.add_argument("--start", type=str, default="20120101",
                        help="财报起始日期，默认 20120101（需足够回看 lookback 窗口）")
    parser.add_argument("--lookback", type=int, default=12,
                        help="时序打分回看季度数，默认 12（3年）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制股票数量（用于测试），默认 None（全部）")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已有数据的股票，从断点继续")
    parser.add_argument("--clean", type=str, default=None,
                        help="清空指定报告期的已有数据后重新计算，如 --clean 20221231")
    return parser.parse_args()


def main():
    args = parse_args()
    engine = create_engine(DB_URL, pool_pre_ping=True)
    ensure_table_exists(engine)

    if args.clean:
        n = clean_report_date(engine, args.clean)
        logger.info(f"已清空 report_date={args.clean} 的 {n} 条旧数据")

    stock_pool = fetch_stock_pool(engine, limit=args.limit)
    logger.info(f"开始全历史回补，共 {len(stock_pool)} 只股票")

    if args.resume:
        with engine.connect() as conn:
            done = pd.read_sql(
                text("SELECT ts_code, report_date FROM stock_financial_score_pool"), conn
            )
        done_map = done.groupby("ts_code")["report_date"].apply(set).to_dict()
        logger.info(f"断点续算模式，已处理 {len(done_map)} 只股票")

    total_rows = 0
    success, skipped = 0, 0

    pbar = tqdm(stock_pool.iterrows(), total=len(stock_pool), desc="全历史评分")
    for _, row in pbar:
        ts_code = row["ts_code"]
        name = row["name"] if pd.notna(row["name"]) else ts_code

        scored = compute_all_scores(ts_code, name, args.start, args.lookback)
        if scored.empty:
            skipped += 1
            pbar.write(f"[{ts_code}] 无数据，已跳过")
            time.sleep(0.2)
            continue

        if args.resume and ts_code in done_map:
            done_dates = done_map[ts_code]
            scored = scored[~scored["end_date"].apply(lambda x: x.strftime("%Y%m%d")).isin(done_dates)]
            if scored.empty:
                tqdm.write(f"[{ts_code}] 所有报告期已存在，跳过")
                time.sleep(0.05)
                continue

        out_rows = build_output_rows(scored, ts_code, name)
        n = upsert_rows(engine, out_rows)
        total_rows += n
        success += 1

        pbar.set_postfix({"成功": success, "跳过": skipped, "已写入": total_rows})
        time.sleep(0.2)

    logger.info(f"完成！成功 {success} 只，跳过 {skipped} 只，总写入 {total_rows} 条")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM stock_financial_score_pool")).fetchone()[0]
        latest_date = conn.execute(
            text("SELECT MAX(report_date) FROM stock_financial_score_pool")
        ).fetchone()[0]
        oldest_date = conn.execute(
            text("SELECT MIN(report_date) FROM stock_financial_score_pool")
        ).fetchone()[0]
        date_dist = pd.read_sql(
            text("SELECT report_date, COUNT(*) as cnt FROM stock_financial_score_pool GROUP BY report_date ORDER BY report_date"),
            conn
        )
    logger.info(f"当前表共 {total} 条，报告期范围: {oldest_date} ~ {latest_date}")
    logger.info("各报告期分布：")
    for _, r in date_dist.iterrows():
        logger.info(f"  {r['report_date']}: {r['cnt']} 条")


if __name__ == "__main__":
    main()
