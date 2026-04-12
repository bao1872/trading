# -*- coding: utf-8 -*-
"""
Purpose:
    股东投资质量画像模型 - 批量回补脚本
    两阶段架构: Phase1 构建变动事件, Phase2 计算画像并入库

Inputs:
    - 数据库表: stock_top10_holders_tushare
    - 数据库表: stock_k_data (行情)
    - 数据库表: stock_pools (行业信息)

Outputs:
    - 数据库表: stock_holder_quality_portrait
    - 临时表: _holder_portrait_events_temp (Phase1 中间结果)

How to Run:
    python financial_factors/backfill_holder_portrait.py --mode full
    python financial_factors/backfill_holder_portrait.py --mode full --limit 50
    python financial_factors/backfill_holder_portrait.py --mode full --resume

Examples:
    python financial_factors/backfill_holder_portrait.py --mode full --limit 100
    python financial_factors/backfill_holder_portrait.py --mode full --resume

Side Effects:
    - 写入 stock_holder_quality_portrait 表
    - 写入/删除 _holder_portrait_events_temp 临时表
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import bulk_upsert, get_engine, get_session, query_df
from financial_factors.holder_quality_portrait import (
    PORTRAIT_TABLE,
    EPS,
    build_change_events,
    build_event_metrics_cache,
    build_holder_tenure,
    compute_composite_score,
    compute_holder_portrait,
    ensure_portrait_table,
    load_bench_data,
    load_industry_info,
    load_market_data_from_db,
    load_stock_pool,
    load_top10_data,
    normalize_ts_code,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EVENTS_TEMP_TABLE = "_holder_portrait_events_temp"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), ".backfill_checkpoints")
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5


class GracefulExiter:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        logger.info("收到终止信号 (%s), 完成当前批次后退出...", signum)
        self.shutdown = True

    def should_exit(self):
        return self.shutdown


def ensure_events_temp_table() -> None:
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {EVENTS_TEMP_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        ts_code TEXT NOT NULL,
        report_date TEXT NOT NULL,
        ann_date TEXT,
        effective_date TEXT,
        holder_name_std TEXT NOT NULL,
        holder_type TEXT,
        holder_rank_curr INTEGER,
        hold_float_ratio_curr REAL,
        delta_hold_float_ratio REAL,
        action_type TEXT NOT NULL,
        tenure_curr INTEGER,
        industry_l2 TEXT,
        future_ret_20 REAL,
        future_ret_60 REAL,
        future_ret_120 REAL,
        future_excess_ret_20 REAL,
        future_excess_ret_60 REAL,
        future_excess_ret_120 REAL,
        future_mdd_60 REAL,
        ret_120 REAL,
        mom_20 REAL,
        mom_60 REAL,
        turnover_value_mean_20 REAL,
        vol_20 REAL,
        drawdown_from_high_120 REAL
    )
    """
    idx_sql = f"""
    CREATE INDEX IF NOT EXISTS idx_events_temp_holder ON {EVENTS_TEMP_TABLE} (holder_name_std);
    CREATE INDEX IF NOT EXISTS idx_events_temp_ts_code ON {EVENTS_TEMP_TABLE} (ts_code);
    """
    with get_engine().begin() as conn:
        conn.execute(text(create_sql))
        for stmt in idx_sql.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    logger.info("Events temp table ensured")


def drop_events_temp_table() -> None:
    with get_engine().begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {EVENTS_TEMP_TABLE}"))
    logger.info("Events temp table dropped")


def save_checkpoint(phase: str, processed_codes: List[str], total_codes: int) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"backfill_{phase}.json")
    data = {"phase": phase, "processed_codes": processed_codes, "total_codes": total_codes, "timestamp": time.time()}
    with open(path, "w") as f:
        json.dump(data, f)
    logger.info("Checkpoint saved: phase=%s, processed=%s/%s", phase, len(processed_codes), total_codes)


def load_checkpoint(phase: str) -> Optional[Dict]:
    path = os.path.join(CHECKPOINT_DIR, f"backfill_{phase}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    logger.info("Checkpoint loaded: phase=%s, processed=%s/%s", data["phase"], len(data["processed_codes"]), data["total_codes"])
    return data


def clear_checkpoint(phase: str) -> None:
    path = os.path.join(CHECKPOINT_DIR, f"backfill_{phase}.json")
    if os.path.exists(path):
        os.remove(path)


def retry_with_backoff(func, *args, max_retries=MAX_RETRIES, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("重试 %s/%s: %s, 等待 %ss...", attempt + 1, max_retries, e, delay)
            time.sleep(delay)


def run_phase1(limit: Optional[int] = None, resume: bool = False, exiter: Optional[GracefulExiter] = None) -> int:
    logger.info("=" * 60)
    logger.info("Phase 1: 构建变动事件")
    logger.info("=" * 60)

    ensure_events_temp_table()

    pool = load_stock_pool(limit)
    if not pool:
        logger.warning("股票池为空")
        return 0
    ts_codes = [x[0] for x in pool]
    logger.info("股票池: %s 只", len(ts_codes))

    processed_codes: List[str] = []
    if resume:
        ckpt = load_checkpoint("phase1")
        if ckpt:
            processed_codes = ckpt["processed_codes"]
            logger.info("从断点恢复: 已处理 %s 只股票", len(processed_codes))
    else:
        with get_engine().begin() as conn:
            conn.execute(text(f"TRUNCATE {EVENTS_TEMP_TABLE}"))

    remaining_codes = [c for c in ts_codes if c not in set(processed_codes)]
    logger.info("待处理: %s 只股票", len(remaining_codes))

    top10_df = load_top10_data(ts_codes=remaining_codes, lookback_years=5)
    if top10_df.empty:
        logger.warning("top10 数据为空")
        return 0
    logger.info("top10 数据: %s 行", len(top10_df))

    industry_df = load_industry_info()
    if not industry_df.empty:
        top10_df = top10_df.merge(industry_df.drop(columns=["name"], errors="ignore"), on="ts_code", how="left")

    top10_df = build_holder_tenure(top10_df)

    stock_df_map: Dict[str, pd.DataFrame] = {}
    unique_codes = top10_df["ts_code"].drop_duplicates().tolist()
    start_date_global = top10_df["ann_date"].dropna().min()
    if pd.isna(start_date_global):
        start_date_global = top10_df["report_date"].dropna().min()
    end_date = pd.Timestamp.today().normalize()

    from tqdm import tqdm
    for ts_code in tqdm(unique_codes, desc="加载行情", unit="股票"):
        g = top10_df[top10_df["ts_code"] == ts_code]
        start_date = g["ann_date"].dropna().min()
        if pd.isna(start_date):
            start_date = g["report_date"].dropna().min()
        if pd.isna(start_date):
            continue
        stock_df = load_market_data_from_db(ts_code, start_date - pd.Timedelta(days=250), end_date)
        if not stock_df.empty:
            stock_df_map[ts_code] = stock_df

    valid_codes = [c for c in unique_codes if c in stock_df_map]
    top10_df = top10_df[top10_df["ts_code"].isin(valid_codes)].copy()
    logger.info("有效股票: %s 只", len(valid_codes))

    bench_df = load_bench_data(start_date_global - pd.Timedelta(days=250), end_date)

    event_cache = build_event_metrics_cache(top10_df, stock_df_map, bench_df)
    events_df = build_change_events(top10_df, event_cache)
    logger.info("变动事件: %s 条", len(events_df))

    if events_df.empty:
        logger.warning("无变动事件")
        return 0

    for col in ["report_date", "ann_date", "effective_date"]:
        if col in events_df.columns:
            events_df[col] = events_df[col].astype(str).str[:10]

    batch_size = 5000
    total_batches = (len(events_df) + batch_size - 1) // batch_size
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(events_df))
        batch = events_df.iloc[start_idx:end_idx].copy()

        def _write_batch(b=batch):
            with get_engine().begin() as conn:
                b.to_sql(EVENTS_TEMP_TABLE, conn, if_exists="append", index=False)

        retry_with_backoff(_write_batch)
        logger.info("  写入批次 %s/%s (%s-%s)", i + 1, total_batches, start_idx, end_idx)

        if exiter and exiter.should_exit():
            processed_codes.extend(valid_codes)
            save_checkpoint("phase1", processed_codes, len(ts_codes))
            logger.info("Phase 1 中断, 已保存断点")
            return len(events_df)

    processed_codes.extend(valid_codes)
    save_checkpoint("phase1", processed_codes, len(ts_codes))
    logger.info("Phase 1 完成: %s 条事件", len(events_df))
    return len(events_df)


def run_phase2(exiter: Optional[GracefulExiter] = None) -> int:
    logger.info("=" * 60)
    logger.info("Phase 2: 计算画像并入库")
    logger.info("=" * 60)

    ensure_portrait_table()

    with get_session() as session:
        events_df = query_df(session, EVENTS_TEMP_TABLE)
    if events_df.empty:
        logger.warning("临时事件表为空, 请先运行 Phase 1")
        return 0
    logger.info("加载事件: %s 条", len(events_df))

    for col in ["report_date", "ann_date", "effective_date"]:
        if col in events_df.columns:
            events_df[col] = pd.to_datetime(events_df[col], errors="coerce")

    for col in ["future_ret_20", "future_ret_60", "future_ret_120",
                "future_excess_ret_20", "future_excess_ret_60", "future_excess_ret_120",
                "future_mdd_60", "ret_120", "mom_20", "mom_60",
                "turnover_value_mean_20", "vol_20", "drawdown_from_high_120",
                "hold_float_ratio_curr", "delta_hold_float_ratio"]:
        if col in events_df.columns:
            events_df[col] = pd.to_numeric(events_df[col], errors="coerce")

    for col in ["holder_rank_curr", "tenure_curr"]:
        if col in events_df.columns:
            events_df[col] = pd.to_numeric(events_df[col], errors="coerce").astype("Int64")

    with get_session() as session:
        top10_df = query_df(session, "stock_top10_holders_tushare")
    if not top10_df.empty:
        top10_df["ts_code"] = top10_df["ts_code"].astype(str).map(normalize_ts_code)
        top10_df["holder_name_std"] = top10_df["holder_name"].map(
            lambda x: str(x).strip().upper() if x and not (isinstance(x, float) and np.isnan(x)) else ""
        )
        for col in ["hold_amount", "hold_ratio", "hold_float_ratio"]:
            if col in top10_df.columns:
                top10_df[col] = pd.to_numeric(top10_df[col], errors="coerce")

    industry_df = load_industry_info()

    portrait = compute_holder_portrait(events_df, top10_df, industry_df)
    if portrait.empty:
        logger.warning("画像为空")
        return 0

    portrait = compute_composite_score(portrait)
    logger.info("画像: %s 个股东", len(portrait))

    for col in ["report_date", "ann_date", "effective_date", "profile_asof_date"]:
        if col in portrait.columns:
            portrait[col] = portrait[col].astype(str).replace({"NaT": None, "nan": None, "None": None})

    records = portrait.to_dict(orient="records")
    cleaned_records = []
    for rec in records:
        clean_rec = {}
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                clean_rec[k] = None
            elif v is pd.NaT:
                clean_rec[k] = None
            else:
                clean_rec[k] = v
        cleaned_records.append(clean_rec)

    if not cleaned_records:
        logger.warning("无有效画像数据")
        return 0

    columns = list(cleaned_records[0].keys())
    non_key_columns = [c for c in columns if c != "holder_name_std"]
    col_clause = ", ".join(columns)
    placeholders = ", ".join([f":{c}" for c in columns])
    update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in non_key_columns])
    upsert_sql = f"""
        INSERT INTO {PORTRAIT_TABLE} ({col_clause})
        VALUES ({placeholders})
        ON CONFLICT (holder_name_std) DO UPDATE SET {update_clause}
    """

    batch_size = 500
    total_batches = (len(cleaned_records) + batch_size - 1) // batch_size
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(cleaned_records))
        batch_records = cleaned_records[start_idx:end_idx]

        def _write_batch(b=batch_records):
            with get_engine().begin() as conn:
                conn.execute(text(upsert_sql), b)
            return len(b)

        n = retry_with_backoff(_write_batch)
        logger.info("  写入批次 %s/%s (%s-%s): %s 行", i + 1, total_batches, start_idx, end_idx, n)

        if exiter and exiter.should_exit():
            logger.info("Phase 2 中断")
            return len(portrait)

    logger.info("Phase 2 完成: %s 个股东入库", len(portrait))

    grade_dist = portrait["quality_grade"].value_counts()
    logger.info("质量等级分布:")
    for grade, cnt in grade_dist.items():
        logger.info("  %s: %s (%.1f%%)", grade, cnt, cnt / len(portrait) * 100)

    style_dist = portrait["style_label"].value_counts()
    logger.info("风格标签分布:")
    for label, cnt in style_dist.items():
        logger.info("  %s: %s (%.1f%%)", label, cnt, cnt / len(portrait) * 100)

    drop_events_temp_table()
    clear_checkpoint("phase1")
    logger.info("临时表和断点已清理")

    return len(portrait)


def main() -> None:
    parser = argparse.ArgumentParser(description="股东投资质量画像模型 - 批量回补")
    parser.add_argument("--mode", choices=["full", "phase1", "phase2"], default="full")
    parser.add_argument("--limit", type=int, default=None, help="限制股票数量")
    parser.add_argument("--resume", action="store_true", help="从断点恢复")
    args = parser.parse_args()

    exiter = GracefulExiter()

    if args.mode in ("full", "phase1"):
        n_events = run_phase1(limit=args.limit, resume=args.resume, exiter=exiter)
        if exiter.should_exit():
            return

    if args.mode in ("full", "phase2"):
        n_portrait = run_phase2(exiter=exiter)

    logger.info("=" * 60)
    logger.info("回补完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
