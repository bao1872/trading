#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前十大流通股东评价体系回补脚本（两阶段架构）

用途：
    Phase 1: 计算全局股东画像（一次性）
    Phase 2: 基于全局画像计算个股评分（从缓存读，很快）

核心逻辑（SSOT）：
    - 行情数据读取：load_market_data_from_cache（只读缓存，不调 Tushare）
    - 评价计算：top10_holder_eval_factors.py 的 compute_stock_scores / build_holder_profiles

用法：
    # Phase 1: 计算全局股东画像
    python financial_factors/backfill_top10_holder.py --mode profiles --clean

    # Phase 2: 计算个股评分
    python financial_factors/backfill_top10_holder.py --mode compute --clean

    # 一次性完成（profiles + scores）
    python financial_factors/backfill_top10_holder.py --mode full --clean

    # 测试模式
    python financial_factors/backfill_top10_holder.py --mode full --limit 5 --dry_run

    # 断点续算
    python financial_factors/backfill_top10_holder.py --mode resume

    # 查看进度
    python financial_factors/backfill_top10_holder.py --mode status

副作用：
    - 写入 stock_top10_holder_profiles_tushare
    - 写入 stock_top10_holder_eval_scores_tushare
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import bulk_upsert, get_session, query_df
from financial_factors.top10_holder_eval_factors import (
    DEFAULT_BENCH,
    INPUT_TABLE,
    PROFILE_TABLE,
    SCORE_TABLE,
    build_holder_profiles,
    build_holder_tenure,
    build_holder_industry_scores,
    compute_stock_scores,
    ensure_output_tables,
    get_stock_pool,
    load_existing_data,
    load_industry_info,
    load_top10_data,
    normalize_ts_code,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

CACHE_TABLE = "stock_market_data_cache"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "top10_holder_backfill.json"
EVENT_CACHE_FILE = CHECKPOINT_DIR / "event_cache.pkl"
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_INTERVAL = 5
LOOKBACK_YEARS = 5


def ensure_checkpoint_dir() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint() -> Dict:
    if not CHECKPOINT_FILE.exists():
        return {"last_updated": None, "status": "pending", "total": 0, "processed": 0, "done": [], "failed": {}}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    status: str,
    total: int,
    done: List[str],
    failed: Dict[str, Dict],
    processed: int = 0,
) -> None:
    ensure_checkpoint_dir()
    data = {
        "last_updated": datetime.now().isoformat(),
        "status": status,
        "total": total,
        "processed": processed,
        "done": done,
        "failed": failed,
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_output_tables() -> Tuple[int, int]:
    n_scores = 0
    n_profiles = 0
    with get_session() as session:
        r1 = session.execute(text(f"DELETE FROM {SCORE_TABLE}"))
        n_scores = r1.rowcount
        r2 = session.execute(text(f"DELETE FROM {PROFILE_TABLE}"))
        n_profiles = r2.rowcount
        session.commit()
    return n_scores, n_profiles


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


def load_market_data_from_cache(ts_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    sql = f"""
        SELECT trade_date, open, high, low, close, vol, turnover, turnover_rate
        FROM {CACHE_TABLE}
        WHERE ts_code = :ts_code AND trade_date BETWEEN :start AND :end
        ORDER BY trade_date
    """
    try:
        with get_session() as session:
            df = pd.read_sql(text(sql), session.bind, params={"ts_code": ts_code, "start": start_str, "end": end_str})
        if not df.empty:
            df = df.set_index("trade_date")
            df.index = pd.to_datetime(df.index)
            df.index.name = None
        return df
    except Exception as e:
        logger.warning(f"读取缓存失败 {ts_code}: {e}")
        return pd.DataFrame()


def load_profiles_from_db() -> pd.DataFrame:
    try:
        with get_session() as session:
            df = pd.read_sql(text(f"SELECT * FROM {PROFILE_TABLE}"), session.bind)
        return df
    except Exception as e:
        logger.warning(f"读取 profiles 失败: {e}")
        return pd.DataFrame()


def load_event_cache() -> Dict:
    if not EVENT_CACHE_FILE.exists():
        return {}
    try:
        with open(EVENT_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"读取 event_cache 失败: {e}")
        return {}


def save_event_cache(cache: Dict) -> None:
    ensure_checkpoint_dir()
    with open(EVENT_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def process_single_stock_build_events(
    ts_code: str,
    lookback_years: int,
    industry_file: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Optional[str]]:
    from financial_factors.top10_holder_eval_factors import (
        build_change_events,
        build_event_metrics_cache,
        EventMarketMetrics,
    )

    top10_df = load_top10_data(ts_codes=[ts_code], lookback_years=lookback_years)
    if top10_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, f"无前十大流通股东数据"

    industry_df = load_industry_info(industry_file)
    if not industry_df.empty:
        top10_df = top10_df.merge(industry_df.drop(columns=["name"], errors="ignore"), on="ts_code", how="left")

    ann_date_min = top10_df["ann_date"].dropna().min()
    report_date_min = top10_df["report_date"].dropna().min()
    if pd.isna(ann_date_min) or pd.isna(report_date_min):
        return pd.DataFrame(), pd.DataFrame(), {}, "无有效日期"

    start_date = min(ann_date_min, report_date_min)
    end_date = pd.Timestamp.today().normalize()

    stock_df = load_market_data_from_cache(ts_code, start_date - pd.Timedelta(days=250), end_date)
    if stock_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, "行情缓存为空"

    bench_code = DEFAULT_BENCH
    bench_df = load_market_data_from_cache(bench_code, start_date - pd.Timedelta(days=250), end_date)
    if bench_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, f"基准指数 {bench_code} 行情缓存为空"

    top10_df = build_holder_tenure(top10_df)

    event_cache = build_event_metrics_cache(top10_df, {ts_code: stock_df}, {ts_code: bench_df}, {})

    events_df = build_change_events(top10_df.copy(), event_cache)

    return top10_df, events_df, event_cache, None


def process_single_stock_scores(
    ts_code: str,
    name: str,
    lookback_years: int,
    industry_file: Optional[str],
    profiles_df: pd.DataFrame,
    event_cache: Dict,
) -> Tuple[pd.DataFrame, Optional[str]]:
    from financial_factors.top10_holder_eval_factors import (
        build_change_events,
        build_event_metrics_cache,
        compute_stock_scores,
    )

    top10_df = load_top10_data(ts_codes=[ts_code], lookback_years=lookback_years)
    if top10_df.empty:
        return pd.DataFrame(), f"无前十大流通股东数据"

    industry_df = load_industry_info(industry_file)
    if not industry_df.empty:
        top10_df = top10_df.merge(industry_df.drop(columns=["name"], errors="ignore"), on="ts_code", how="left")

    ann_date_min = top10_df["ann_date"].dropna().min()
    report_date_min = top10_df["report_date"].dropna().min()
    if pd.isna(ann_date_min) or pd.isna(report_date_min):
        return pd.DataFrame(), "无有效日期"

    start_date = min(ann_date_min, report_date_min)
    end_date = pd.Timestamp.today().normalize()

    stock_df = load_market_data_from_cache(ts_code, start_date - pd.Timedelta(days=250), end_date)
    if stock_df.empty:
        return pd.DataFrame(), "行情缓存为空"

    bench_code = DEFAULT_BENCH
    bench_df = load_market_data_from_cache(bench_code, start_date - pd.Timedelta(days=250), end_date)
    if bench_df.empty:
        return pd.DataFrame(), f"基准指数 {bench_code} 行情缓存为空"

    top10_df = build_holder_tenure(top10_df)

    local_event_cache = build_event_metrics_cache(top10_df, {ts_code: stock_df}, {ts_code: bench_df}, {})

    events_df = build_change_events(top10_df.copy(), local_event_cache)
    if events_df.empty:
        return pd.DataFrame(), "无有效事件"

    existing_scores, _ = load_existing_data([ts_code])

    holder_industry_quality_map = build_holder_industry_scores(events_df, profiles_df) if not profiles_df.empty else {}

    scores_df = compute_stock_scores(
        top10_df,
        profiles_df,
        {ts_code: stock_df},
        local_event_cache,
        holder_industry_quality_map,
        existing_scores=existing_scores,
    )

    return scores_df, None


def run_compute_profiles(
    limit: Optional[int],
    lookback_years: int,
    dry_run: bool,
    industry_file: Optional[str],
    clean: bool,
) -> None:
    ensure_output_tables()
    ensure_cache_table()

    if clean:
        n_scores, n_profiles = clean_output_tables()
        logger.info(f"已清理：scores {n_scores} 行, profiles {n_profiles} 行")

    cached_codes = get_cached_codes()
    logger.info(f"行情缓存中已有 {len(cached_codes)} 只股票")

    pool = get_stock_pool(limit=limit)
    if not pool:
        logger.warning("股票池为空")
        return

    pool = [(ts, name) for ts, name in pool if ts in cached_codes]
    total = len(pool)
    logger.info(f"Phase 1: 计算全局股东画像，共 {total} 只股票...")

    all_events_list: List[pd.DataFrame] = []
    all_event_caches: List[Dict] = []

    pbar = tqdm(pool, desc="Phase1-构建事件", unit="股票")
    interrupted = False

    def handle_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        logger.info("收到中断信号...")

    old_handler = signal.signal(signal.SIGINT, handle_interrupt)

    try:
        for i, (ts_code, name) in enumerate(pbar):
            if interrupted:
                break

            retry_count = 0
            error_msg = None
            top10_df = pd.DataFrame()
            events_df = pd.DataFrame()
            event_cache: Dict = {}

            while retry_count < MAX_RETRIES:
                try:
                    top10_df, events_df, event_cache, error_msg = process_single_stock_build_events(
                        ts_code=ts_code,
                        lookback_years=lookback_years,
                        industry_file=industry_file,
                    )
                    if error_msg is None:
                        break
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"{ts_code} 第 {retry_count + 1} 次尝试失败: {e}")

                retry_count += 1
                if retry_count < MAX_RETRIES:
                    time.sleep(RETRY_INTERVAL)

            if error_msg is not None:
                logger.warning(f"{ts_code} 最终失败: {error_msg}")
                continue

            if not events_df.empty:
                events_df["_source_ts_code"] = ts_code
                all_events_list.append(events_df)
                all_event_caches.append(event_cache)

            pbar.set_postfix_str(f"已收集 {len(all_events_list)} 只")

            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{total} 只，收集事件 {sum(len(e) for e in all_events_list)} 条")

    finally:
        signal.signal(signal.SIGINT, old_handler)

    if not all_events_list:
        logger.warning("无有效事件数据")
        return

    logger.info(f"Phase 1: 合并所有事件，共 {sum(len(e) for e in all_events_list)} 条...")
    all_events_df = pd.concat(all_events_list, ignore_index=True)
    all_events_df.drop(columns=["_source_ts_code"], inplace=True, errors="ignore")

    logger.info("Phase 1: 计算全局股东画像...")
    global_profiles = build_holder_profiles(all_events_df, existing_profiles=None)

    if global_profiles.empty:
        logger.warning("全局股东画像为空")
        return

    logger.info(f"全局股东画像: {len(global_profiles)} 个股东")

    if dry_run:
        logger.info(f"[DRY RUN] profiles rows={len(global_profiles)}")
    else:
        with get_session() as session:
            n = bulk_upsert(session, PROFILE_TABLE, global_profiles, ["holder_name_std"])
            session.commit()
        logger.info(f"写入 profiles {n} 行")

    merged_event_cache: Dict = {}
    for ec in all_event_caches:
        merged_event_cache.update(ec)
    logger.info(f"保存 event_cache ({len(merged_event_cache)} 条)...")
    save_event_cache(merged_event_cache)

    logger.info("Phase 1 完成！请运行 Phase 2: --mode compute")


def run_compute_scores(
    limit: Optional[int],
    lookback_years: int,
    dry_run: bool,
    industry_file: Optional[str],
    clean: bool,
) -> None:
    ensure_output_tables()
    ensure_cache_table()

    if clean:
        with get_session() as session:
            r1 = session.execute(text(f"DELETE FROM {SCORE_TABLE}"))
            session.commit()
        logger.info("已清理 scores 表")

    profiles_df = load_profiles_from_db()
    if profiles_df.empty:
        logger.warning("profiles 表为空，请先运行 --mode profiles")
        return

    logger.info(f"从数据库加载 {len(profiles_df)} 个股东画像")

    cached_codes = get_cached_codes()
    pool = get_stock_pool(limit=limit)
    if not pool:
        logger.warning("股票池为空")
        return

    pool = [(ts, name) for ts, name in pool if ts in cached_codes]
    total = len(pool)
    logger.info(f"Phase 2: 计算个股评分，共 {total} 只股票...")

    checkpoint = load_checkpoint()
    done: Set[str] = set(checkpoint.get("done", []))
    failed: Dict[str, Dict] = checkpoint.get("failed", {})
    scores_batch: List[pd.DataFrame] = []
    total_scores = 0

    pbar = tqdm(pool, desc="Phase2-计算评分", unit="股票")
    interrupted = False

    def handle_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        logger.info("收到中断信号...")

    old_handler = signal.signal(signal.SIGINT, handle_interrupt)

    try:
        for i, (ts_code, name) in enumerate(pbar):
            if interrupted:
                break

            if ts_code in done:
                pbar.set_postfix_str(f"跳过 {len(done)} | 失败 {len(failed)}")
                continue

            pbar.set_postfix_str(f"成功 {len(done)} | 失败 {len(failed)}")

            retry_count = 0
            error_msg = None
            scores_df = pd.DataFrame()

            while retry_count < MAX_RETRIES:
                try:
                    scores_df, error_msg = process_single_stock_scores(
                        ts_code=ts_code,
                        name=name,
                        lookback_years=lookback_years,
                        industry_file=industry_file,
                        profiles_df=profiles_df,
                        event_cache={},
                    )
                    if error_msg is None:
                        break
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"{ts_code} 第 {retry_count + 1} 次尝试失败: {e}")

                retry_count += 1
                if retry_count < MAX_RETRIES:
                    time.sleep(RETRY_INTERVAL)

            if error_msg is not None:
                failed[ts_code] = {"error": error_msg, "retries": retry_count}
                continue

            done.add(ts_code)
            if ts_code in failed:
                del failed[ts_code]

            if not scores_df.empty:
                scores_batch.append(scores_df)

            if len(scores_batch) >= BATCH_SIZE or i == len(pool) - 1:
                if scores_batch:
                    scores_df_total = pd.concat(scores_batch, ignore_index=True)
                    if dry_run:
                        logger.info(f"[DRY RUN] scores rows={len(scores_df_total)}")
                    else:
                        with get_session() as session:
                            n = bulk_upsert(session, SCORE_TABLE, scores_df_total, ["ts_code", "report_date"])
                            session.commit()
                        total_scores += n
                    scores_batch.clear()

            if (i + 1) % 50 == 0:
                save_checkpoint("running", total, list(done), failed, i + 1)

            time.sleep(0.3)

    finally:
        signal.signal(signal.SIGINT, old_handler)

    if scores_batch:
        scores_df_total = pd.concat(scores_batch, ignore_index=True)
        if dry_run:
            logger.info(f"[DRY RUN] scores rows={len(scores_df_total)}")
        else:
            with get_session() as session:
                n = bulk_upsert(session, SCORE_TABLE, scores_df_total, ["ts_code", "report_date"])
                session.commit()
            total_scores += n

    final_status = "failed" if failed else "done"
    save_checkpoint(final_status, total, list(done), failed, len(done))
    logger.info(f"Phase 2 完成：成功 {len(done)} 只，失败 {len(failed)} 只")
    logger.info(f"写入 scores {total_scores} 行")


def run_full(
    limit: Optional[int],
    lookback_years: int,
    dry_run: bool,
    industry_file: Optional[str],
    clean: bool,
) -> None:
    logger.info("=" * 50)
    logger.info("开始全量回补（Phase 1 + Phase 2）")
    logger.info("=" * 50)

    run_compute_profiles(limit, lookback_years, dry_run, industry_file, clean)
    logger.info("=" * 50)
    run_compute_scores(limit, lookback_years, dry_run, industry_file, clean=True)
    logger.info("=" * 50)
    logger.info("全量回补完成！")


def show_status() -> None:
    checkpoint = load_checkpoint()
    cached_codes = get_cached_codes()

    profiles_count = 0
    try:
        with get_session() as session:
            df = pd.read_sql(text(f"SELECT COUNT(*) as cnt FROM {PROFILE_TABLE}"), session.bind)
            profiles_count = int(df["cnt"].iloc[0]) if not df.empty else 0
    except Exception:
        pass

    if not checkpoint.get("done") and not checkpoint.get("failed"):
        logger.info("无断点信息")
        logger.info(f"行情缓存：{len(cached_codes)} 只")
        logger.info(f"股东画像：{profiles_count} 个")
        return

    total = checkpoint.get("total", 0)
    done_count = len(checkpoint.get("done", []))
    failed = checkpoint.get("failed", {})

    logger.info(f"当前状态：{checkpoint.get('status', 'unknown')}")
    logger.info(f"Phase 2 进度：{done_count}/{total} ({done_count/total*100:.1f}%)" if total else f"完成 {done_count} 只")
    logger.info(f"Phase 2 失败：{len(failed)} 只")
    logger.info(f"行情缓存：{len(cached_codes)} 只")
    logger.info(f"股东画像：{profiles_count} 个")

    if failed:
        logger.info("失败列表（前10只）：")
        for code, info in list(failed.items())[:10]:
            logger.info(f"  {code}: {info['error']}")

    logger.info(f"最后更新：{checkpoint.get('last_updated', 'N/A')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="前十大流通股东评价体系回补脚本（两阶段架构）")
    parser.add_argument("--mode", choices=["profiles", "compute", "full", "resume", "status"], default="full")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数量（测试用）")
    parser.add_argument("--lookback_years", type=int, default=LOOKBACK_YEARS)
    parser.add_argument("--dry_run", action="store_true", help="只预览，不写库")
    parser.add_argument("--industry_file", type=str, default=None)
    parser.add_argument("--clean", action="store_true", help="计算前清理已有数据")

    args = parser.parse_args()

    if args.mode == "status":
        show_status()
        return

    if args.mode == "profiles":
        run_compute_profiles(
            limit=args.limit,
            lookback_years=args.lookback_years,
            dry_run=args.dry_run,
            industry_file=args.industry_file,
            clean=args.clean,
        )
        return

    if args.mode == "compute":
        run_compute_scores(
            limit=args.limit,
            lookback_years=args.lookback_years,
            dry_run=args.dry_run,
            industry_file=args.industry_file,
            clean=args.clean,
        )
        return

    if args.mode == "full":
        run_full(
            limit=args.limit,
            lookback_years=args.lookback_years,
            dry_run=args.dry_run,
            industry_file=args.industry_file,
            clean=args.clean,
        )
        return

    if args.mode == "resume":
        run_compute_scores(
            limit=args.limit,
            lookback_years=args.lookback_years,
            dry_run=args.dry_run,
            industry_file=args.industry_file,
            clean=False,
        )
        return


if __name__ == "__main__":
    main()
