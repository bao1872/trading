#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行情数据预拉取脚本

用途：
    将所有股票的日线行情一次性从 Tushare 拉取到本地缓存表
    后续回补计算时直接读缓存，不再调用 Tushare API

与 backfill_top10_holder.py 的关系：
    - Phase 1: 预拉取行情（本脚本）
    - Phase 2: 回补计算（backfill_top10_holder.py --mode compute）

用法：
    # 全量预拉取
    python financial_factors/prefetch_market_data.py

    # 测试模式（10只）
    python financial_factors/prefetch_market_data.py --limit 10

    # 断点续算（如中断后）
    python financial_factors/prefetch_market_data.py --resume

    # 查看进度
    python financial_factors/prefetch_market_data.py --status

    # 强制重拉（跳过缓存）
    python financial_factors/prefetch_market_data.py --force

副作用：
    - 写入 stock_market_data_cache 表
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import bulk_upsert, get_session, query_df
from tushare_data.fetcher import fetch_market_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

CACHE_TABLE = "stock_market_data_cache"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "market_data_prefetch.json"
BATCH_SIZE = 50
MAX_RETRIES = 5
RETRY_DELAY = 5
LOOKBACK_YEARS = 5
DEFAULT_BENCH = "000300.SH"
INDEX_CODES = {"000001.SH", "000300.SH", "000905.SH", "000016.SH", "000688.SH",
               "399001.SZ", "399006.SZ", "399005.SZ", "399300.SH", "399673.SZ"}


def ensure_checkpoint_dir() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint() -> Dict:
    if not CHECKPOINT_FILE.exists():
        return {"last_updated": None, "status": "pending", "total": 0, "done": [], "failed": {}}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(status: str, total: int, done: List[str], failed: Dict) -> None:
    ensure_checkpoint_dir()
    data = {
        "last_updated": datetime.now().isoformat(),
        "status": status,
        "total": total,
        "done": done,
        "failed": failed,
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_cache_table() -> None:
    sql = f"""
    CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
        ts_code TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        vol REAL,
        turnover REAL,
        turnover_rate REAL,
        PRIMARY KEY (ts_code, trade_date)
    )
    """
    with get_session() as session:
        session.execute(text(sql))
        session.commit()


def get_cached_codes() -> Set[str]:
    sql = f"SELECT DISTINCT ts_code FROM {CACHE_TABLE}"
    try:
        with get_session() as session:
            df = pd.read_sql(text(sql), session.bind)
        return set(df["ts_code"].astype(str).tolist())
    except Exception:
        return set()


def get_stock_pool(limit: Optional[int] = None) -> List[str]:
    with get_session() as session:
        df = query_df(session, "stock_pools", columns=["ts_code"])
    if df.empty:
        return []
    codes = df["ts_code"].astype(str).tolist()
    if limit:
        codes = codes[:limit]
    return codes


def save_market_data_to_cache(ts_code: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    df = df.copy()
    df["ts_code"] = ts_code
    df["trade_date"] = df.index.astype(str).str[:10]
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "turnover", "turnover_rate"]
    available = [c for c in cols if c in df.columns]
    df = df[available]
    if df.empty:
        return 0
    unique_keys = ["ts_code", "trade_date"]
    non_key = [c for c in available if c not in unique_keys]
    if not non_key:
        return 0
    update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in non_key])
    col_clause = ", ".join(available)
    placeholders = ", ".join([f":{c}" for c in available])
    sql = f"""
        INSERT INTO {CACHE_TABLE} ({col_clause})
        VALUES ({placeholders})
        ON CONFLICT ({', '.join(unique_keys)}) DO UPDATE SET {update_clause}
    """
    records = df.to_dict(orient="records")
    total = 0
    with get_session() as session:
        for batch in (records[i:i+1000] for i in range(0, len(records), 1000)):
            session.execute(text(sql), batch)
            total += len(batch)
        session.commit()
    return total


def fetch_and_cache(
    ts_code: str,
    start_date: str,
    end_date: str,
    max_retries: int = MAX_RETRIES,
) -> bool:
    for attempt in range(max_retries):
        try:
            df = fetch_market_data(ts_code, start_date, end_date)
            if df.empty:
                if attempt < max_retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return False
            save_market_data_to_cache(ts_code, df)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            logger.warning(f"{ts_code} 最终失败: {e}")
            return False
    return False


def run_prefetch(
    limit: Optional[int],
    resume: bool,
    force: bool,
) -> None:
    ensure_cache_table()

    all_codes = get_stock_pool(limit=limit)
    if not all_codes:
        logger.warning("股票池为空")
        return

    cached_codes = get_cached_codes() if not force else set()
    total = len(all_codes)

    if force:
        logger.info(f"强制模式：跳过缓存检查，共 {total} 只股票")
    else:
        logger.info(f"股票池共 {total} 只，缓存中已有 {len(cached_codes)} 只")

    to_fetch = [c for c in all_codes if c not in cached_codes]
    logger.info(f"需要拉取 {len(to_fetch)} 只")

    if not to_fetch:
        logger.info("全部已缓存，无需拉取")
        return

    checkpoint = load_checkpoint() if resume else {}
    done: List[str] = checkpoint.get("done", [])
    failed: Dict[str, Dict] = checkpoint.get("failed", {})

    done_set = set(done)
    to_fetch = [c for c in to_fetch if c not in done_set]

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y%m%d")

    failed_count = 0
    success_count = len(done)
    interrupted = False

    def handle_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        logger.info("收到中断信号，正在保存进度...")

    old_handler = signal.signal(signal.SIGINT, handle_interrupt)

    pbar = tqdm(to_fetch, desc="预拉取行情", unit="只")
    for ts_code in pbar:
        if interrupted:
            break

        pbar.set_postfix_str(f"成功 {success_count} | 失败 {failed_count}")

        ok = fetch_and_cache(ts_code, start_date, end_date)

        if ok:
            done.append(ts_code)
            success_count += 1
            if ts_code in failed:
                del failed[ts_code]
        else:
            failed[ts_code] = {"error": "fetch failed", "retries": MAX_RETRIES}
            failed_count += 1

        if len(done) % 100 == 0:
            save_checkpoint("running", total, done, failed)

        time.sleep(1)

    signal.signal(signal.SIGINT, old_handler)

    final_status = "done" if not failed else "partial"
    save_checkpoint(final_status, total, done, failed)

    logger.info(f"预拉取完成：成功 {success_count}，失败 {failed_count}")

    if failed:
        logger.info("失败列表（前10只）：")
        for code, info in list(failed.items())[:10]:
            logger.info(f"  {code}: {info}")


def show_status() -> None:
    checkpoint = load_checkpoint()
    cached_codes = get_cached_codes()

    if not checkpoint.get("done") and not checkpoint.get("failed"):
        logger.info(f"无断点信息。缓存中已有 {len(cached_codes)} 只股票。")
        return

    total = checkpoint.get("total", 0)
    done_count = len(checkpoint.get("done", []))
    failed = checkpoint.get("failed", {})

    logger.info(f"状态: {checkpoint.get('status', 'unknown')}")
    logger.info(f"进度: {done_count}/{total} ({done_count/total*100:.1f}%)" if total else f"完成 {done_count} 只")
    logger.info(f"缓存中: {len(cached_codes)} 只")
    logger.info(f"失败: {len(failed)} 只")

    if failed:
        logger.info("失败列表（前10只）：")
        for code, info in list(failed.items())[:10]:
            logger.info(f"  {code}: {info}")


def main() -> None:
    parser = argparse.ArgumentParser(description="行情数据预拉取脚本")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数量（测试用）")
    parser.add_argument("--resume", action="store_true", help="断点续算")
    parser.add_argument("--force", action="store_true", help="强制重拉（跳过缓存）")
    parser.add_argument("--status", action="store_true", help="查看进度")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    run_prefetch(
        limit=args.limit,
        resume=args.resume,
        force=args.force,
    )


if __name__ == "__main__":
    main()