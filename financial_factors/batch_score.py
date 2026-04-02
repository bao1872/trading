#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票池财务评分批量计算脚本

目的：
- 引用 db_score.py 的核心计算逻辑（SSOT）
- 对 stock_concepts_cache 中的全股票池计算最近一期财务因子和评分
- 结果写入 stock_financial_score_pool 表

使用方法：
1) 在项目根目录运行
   python financial_factors/batch_score.py
2) 可选参数
   python financial_factors/batch_score.py --limit 3 --resume

输出：
- 终端打印进度和统计信息
- 结果写入 stock_financial_score_pool 表

说明：
- 写入语义为 upsert（按 ts_code + report_date），重复运行会覆盖
- 单只股票数据异常或无数据时跳过该只，不中断整体流程
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
        "financial_factors", "db_score.py"
    )
    spec = importlib.util.spec_from_file_location("db_score", db_path)
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    return sm


sm = _load_db_score()
fetch_financial_statements_from_db = sm.fetch_financial_statements_from_db
prepare_base_dataframe = sm.prepare_base_dataframe
dedup_latest = sm.dedup_latest
add_ytd_and_ttm = sm.add_ytd_and_ttm
add_factors = sm.add_factors
score_dataframe = sm.score_dataframe
FACTOR_CONFIG = sm.FACTOR_CONFIG
FACTOR_COLS = [cfg["factor_name"] for cfg in FACTOR_CONFIG]


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
            pk_def = "id SERIAL PRIMARY KEY"
        else:
            result = conn.execute(
                text("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name = 'stock_financial_score_pool'")
            )
            exists = result.fetchone()[0] > 0
            pk_def = "id INTEGER PRIMARY KEY AUTOINCREMENT"

        if not exists:
            logger.info("创建表 stock_financial_score_pool ...")
            create_sql = f"""
CREATE TABLE IF NOT EXISTS stock_financial_score_pool (
    {pk_def},
    ts_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    report_date VARCHAR(8) NOT NULL,
    total_score FLOAT,
    规模与增长_score FLOAT,
    盈利能力_score FLOAT,
    利润质量_score FLOAT,
    现金创造能力_score FLOAT,
    资产效率与资金占用_score FLOAT,
    边际变化与持续性_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, report_date)
)"""
            conn.execute(text(create_sql))
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


def fetch_already_processed(engine):
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT ts_code FROM stock_financial_score_pool"), conn)
    return set(df["ts_code"].tolist())


def compute_single_stock(
    ts_code: str,
    name: str,
    start_date: str,
    lookback: int,
    limit_n_quarters: int = 16,
):
    try:
        df = prepare_base_dataframe(
            ts_code=ts_code,
            start_date=start_date,
            limit_n_quarters=limit_n_quarters,
        )

        if df.empty:
            return None

        df = add_ytd_and_ttm(df)
        df = add_factors(df)
        df = score_dataframe(df, lookback=lookback)

        latest = df.iloc[[-1]].copy()
        return latest

    except Exception as e:
        logger.warning(f"[{ts_code}] 计算异常: {e}")
        return None


def build_output_row(latest: pd.DataFrame, ts_code: str, name: str) -> dict:
    row = {
        "ts_code": ts_code,
        "stock_name": name,
        "report_date": latest["end_date"].iloc[0].strftime("%Y%m%d"),
    }
    for col in SCORE_COLS:
        if col in latest.columns:
            v = latest[col].iloc[0]
            row[col] = None if pd.isna(v) else float(v)
    for col in FACTOR_COLS:
        if col in latest.columns:
            v = latest[col].iloc[0]
            row[col] = None if pd.isna(v) else float(v)
    return row


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
    parser = argparse.ArgumentParser(description="股票池财务评分批量计算")
    parser.add_argument("--start", type=str, default="20120101",
                        help="财报起始日期（控制数据库读取范围，建议使用早期日期配合 --recent-n 使用）")
    parser.add_argument("--lookback", type=int, default=12,
                        help="时序打分回看季度数，默认 12（3年）")
    parser.add_argument("--recent-n", type=int, default=16,
                        help="从数据库最多读取最近 N 个季度，默认16（约4年）；需足够覆盖最新报告期和 lookback 窗口")
    parser.add_argument("--clean", type=str, default=None,
                        help="清空指定报告期的已有数据后重新计算，如 --clean 20251231")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制股票数量（用于测试），默认 None（全部）")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已入库的股票，从断点继续")
    return parser.parse_args()


def main():
    args = parse_args()
    engine = create_engine(DB_URL, pool_pre_ping=True)
    ensure_table_exists(engine)

    if args.clean:
        n = clean_report_date(engine, args.clean)
        logger.info(f"已清空 report_date={args.clean} 的 {n} 条旧数据")

    stock_pool = fetch_stock_pool(engine, limit=args.limit)
    logger.info(f"开始批量计算，共 {len(stock_pool)} 只股票")

    if args.resume:
        already = fetch_already_processed(engine)
        stock_pool = stock_pool[~stock_pool["ts_code"].isin(already)]
        logger.info(f"断点续算模式，跳过 {len(already)} 只已有数据，待处理 {len(stock_pool)} 只")

    results = []
    success, skipped = 0, 0

    for _, row in tqdm(stock_pool.iterrows(), total=len(stock_pool), desc="财务评分"):
        ts_code = row["ts_code"]
        name = row["name"] if pd.notna(row["name"]) else ts_code

        latest = compute_single_stock(ts_code, name, args.start, args.lookback, limit_n_quarters=args.recent_n)
        if latest is None or latest.empty:
            skipped += 1
            tqdm.write(f"[{ts_code}] 无数据，已跳过")
            time.sleep(0.2)
            continue

        out_row = build_output_row(latest, ts_code, name)
        n = upsert_rows(engine, [out_row])
        success += 1

        time.sleep(0.2)

    logger.info(f"完成！成功 {success} 只，跳过 {skipped} 只")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM stock_financial_score_pool")).fetchone()[0]
        latest_date = conn.execute(
            text("SELECT MAX(report_date) FROM stock_financial_score_pool")
        ).fetchone()[0]
    logger.info(f"当前表共 {total} 条，最新报告期: {latest_date}")


if __name__ == "__main__":
    main()
