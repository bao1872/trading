# -*- coding: utf-8 -*-
"""
Breakout Events 扫描器

Purpose:
    扫描股票池，基于 ATR Rope + Breakout Volume Delta 因子引擎，
    识别并记录两类独立事件：
    1. rope_dir 翻多事件 (breakout_dir_turn_events)
    2. 回踩买点事件 (breakout_pullback_buy_events)

Inputs:
    - 数据库股票池: stock_concepts_cache 表
    - K线数据: 从 stock_k_data 数据库读取

Outputs:
    - breakout_dir_turn_events 表: 翻多事件记录
    - breakout_pullback_buy_events 表: 回踩买点记录

How to Run:
    # 扫描今天（日线 + 60分钟线）
    python backtrader/scan_breakout_events.py

    # 扫描指定日期
    python backtrader/scan_breakout_events.py --date 2026-03-30

    # 只扫描日线
    python backtrader/scan_breakout_events.py --date 2026-03-30 --freqs d

    # 只扫描指定股票
    python backtrader/scan_breakout_events.py --date 2026-03-30 --codes 600519,000001

    # 使用自定义数据库
    python backtrader/scan_breakout_events.py --date 2026-03-30 --db-path ./my_data.db

    # 设置日志级别
    python backtrader/scan_breakout_events.py --date 2026-03-30 --log-level DEBUG

    # 批量回补（从开始日期到上一交易日，自动跳过已扫描的日期，只扫日线）
    python backtrader/scan_breakout_events.py --batch-backfill 2026-02-01

    # 批量回补指定日期范围
    python backtrader/scan_breakout_events.py --batch-backfill 2026-02-01 --end-date 2026-03-15

Side Effects:
    - 写入 breakout_dir_turn_events 表
    - 写入 breakout_pullback_buy_events 表
    - 默认强制更新（重复扫描会替换旧数据）
    - --batch-backfill 会自动跳过已扫描的日期
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATABASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("breakout_events")


def get_engine():
    url = os.environ.get("DATABASE_URL", DATABASE_URL)
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True, pool_recycle=1800)


def infer_market(ts_code: str) -> Optional[int]:
    code = ts_code.split(".")[0] if "." in ts_code else ts_code
    if code.startswith(("000", "001", "002", "003", "300")):
        return 0
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return 1
    return None


def ts_code_to_code(ts_code: str) -> str:
    return ts_code.split(".")[0] if "." in ts_code else ts_code


def load_stock_pool_from_db() -> List[Dict]:
    sql = """
        SELECT ts_code, name
        FROM stock_pools
        ORDER BY ts_code
    """
    engine = get_engine()
    df = pd.read_sql_query(text(sql), engine)
    engine.dispose()
    return df.to_dict("records")


def normalize_freq(freq: str) -> str:
    f = freq.lower().strip()
    if f in {"d", "1d", "day", "daily"}:
        return "d"
    if f in {"60", "60m", "60min", "1h"}:
        return "60m"
    raise ValueError(f"Unsupported freq: {freq}")


def load_kline_data_from_db(
    ts_code: str,
    freq: str,
    bars: int,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    engine = get_engine()
    if end_date:
        sql = """
            SELECT bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE ts_code = :ts_code AND freq = :freq AND bar_time <= :end_date
            ORDER BY bar_time DESC
            LIMIT :limit
        """
        params = {"ts_code": ts_code, "freq": freq, "limit": bars, "end_date": end_date + " 23:59:59"}
    else:
        sql = """
            SELECT bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE ts_code = :ts_code AND freq = :freq
            ORDER BY bar_time DESC
            LIMIT :limit
        """
        params = {"ts_code": ts_code, "freq": freq, "limit": bars}
    try:
        df = pd.read_sql_query(text(sql), engine, params=params)
    except Exception as exc:
        logger.warning(f"查询 {ts_code} {freq} 数据失败: {exc}")
        return None
    finally:
        engine.dispose()

    if df.empty:
        return None

    df = df.sort_values("bar_time", ascending=True)
    df["datetime"] = pd.to_datetime(df["bar_time"])
    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.rename(columns={"volume": "vol"})
    df["amount"] = df["vol"]
    return df


def build_engine_args(
    symbol: str,
    freq: str,
    bars: int,
    len_: int = 14,
    multi: float = 1.5,
) -> argparse.Namespace:
    args = argparse.Namespace()
    args.symbol = symbol
    args.freq = normalize_freq(freq)
    args.bars = bars
    args.len = len_
    args.multi = multi
    args.source = "close"
    args.show_ranges = False
    args.show_atr_channel = False
    args.show_break_markers = False
    args.show_factor_panels = False
    args.swing_left = 10
    args.swing_right = 10
    args.breakout_by = "Close"
    args.max_levels = 5
    args.br_vol_filter = False
    args.br_vol_filter_pct = 60.0
    args.liq_lookback = 20
    return args


def extract_dir_turn_events(
    df: pd.DataFrame,
    ts_code: str,
    name: str,
    freq: str,
    target_date: Optional[str] = None,
) -> List[Dict]:
    if "dir_turn_long_flag" not in df.columns:
        return []
    events = []
    for idx, row in df.iterrows():
        if row.get("dir_turn_long_flag", 0) == 1.0 and pd.notna(row.get("breakout_quality_score")):
            event_time = idx.strftime("%Y-%m-%d %H:%M:%S") if hasattr(idx, "strftime") else str(idx)
            event_date = event_time.split(" ")[0]
            if target_date and event_date != target_date:
                continue
            events.append({
                "ts_code": ts_code,
                "name": name,
                "event_time": event_time,
                "freq": freq,
                "breakout_quality_score": row.get("breakout_quality_score", None),
                "breakout_quality_grade": row.get("breakout_quality_grade", ""),
                "score_trend_total": row.get("score_trend_total", None),
                "score_candle_total": row.get("score_candle_total", None),
                "score_volume_total": row.get("score_volume_total", None),
                "score_freshness_total": row.get("score_freshness_total", None),
                "score_bg_rope_slope": row.get("score_bg_rope_slope", None),
                "score_bg_dist_to_rope": row.get("score_bg_dist_to_rope", None),
                "score_bg_consolidation": row.get("score_bg_consolidation", None),
                "score_candle_close_pos": row.get("score_candle_close_pos", None),
                "score_candle_body_to_range": row.get("score_candle_body_to_range", None),
                "score_candle_upper_wick": row.get("score_candle_upper_wick", None),
                "score_volume_vol_z": row.get("score_volume_vol_z", None),
                "score_volume_vol_record": row.get("score_volume_vol_record", None),
                "score_freshness_count": row.get("score_freshness_count", None),
                "score_freshness_cum_gain": row.get("score_freshness_cum_gain", None),
                "breakout_action": row.get("breakout_action", ""),
                "breakout_freshness_count": row.get("breakout_freshness_count", None),
                "breakout_freshness_cum_gain": row.get("breakout_freshness_cum_gain", None),
                "rope_slope_atr_5": row.get("rope_slope_atr_5", None),
                "dist_to_rope_atr": row.get("dist_to_rope_atr", None),
                "consolidation_bars": row.get("consolidation_bars", None),
                "vol_zscore": row.get("vol_zscore", None),
                "vol_record_days": row.get("vol_record_days", None),
                "dir_turn_upper_price": row.get("dir_turn_upper_price", None),
                "dir_turn_atr_raw": row.get("dir_turn_atr_raw", None),
                "dir_turn_tol_price": row.get("dir_turn_tol_price", None),
                "dir_turn_band_low": row.get("dir_turn_band_low", None),
                "dir_turn_band_high": row.get("dir_turn_band_high", None),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
    return events


def extract_pullback_buy_events(
    df: pd.DataFrame,
    ts_code: str,
    name: str,
    freq: str,
    target_date: Optional[str] = None,
) -> List[Dict]:
    if "pullback_buy_flag" not in df.columns:
        return []
    events = []
    for i, (idx, row) in enumerate(df.iterrows()):
        if row.get("pullback_buy_flag", 0) == 1.0:
            buy_time = idx.strftime("%Y-%m-%d %H:%M:%S") if hasattr(idx, "strftime") else str(idx)
            buy_date = buy_time.split(" ")[0]
            if target_date and buy_date != target_date:
                continue
            src_idx = row.get("source_breakout_index")
            breakout_to_buy_bars = i - int(src_idx) if pd.notna(src_idx) else None
            dir_turn_upper_price = None
            dir_turn_atr_raw = None
            dir_turn_tol_price = None
            dir_turn_band_low = None
            dir_turn_band_high = None
            if pd.notna(src_idx) and int(src_idx) < len(df):
                src_row = df.iloc[int(src_idx)]
                dir_turn_upper_price = src_row.get("dir_turn_upper_price", None)
                dir_turn_atr_raw = src_row.get("dir_turn_atr_raw", None)
                dir_turn_tol_price = src_row.get("dir_turn_tol_price", None)
                dir_turn_band_low = src_row.get("dir_turn_band_low", None)
                dir_turn_band_high = src_row.get("dir_turn_band_high", None)
            events.append({
                "ts_code": ts_code,
                "name": name,
                "buy_time": buy_time,
                "freq": freq,
                "buy_type": row.get("buy_type", ""),
                "source_breakout_time": row.get("source_breakout_time", ""),
                "source_breakout_index": src_idx,
                "breakout_to_buy_bars": breakout_to_buy_bars,
                "breakout_quality_score": row.get("breakout_quality_score", None),
                "score_trend_total": row.get("score_trend_total", None),
                "score_candle_total": row.get("score_candle_total", None),
                "score_volume_total": row.get("score_volume_total", None),
                "score_freshness_total": row.get("score_freshness_total", None),
                "signal_note": row.get("signal_note", ""),
                "pullback_touch_support_flag": row.get("pullback_touch_support_flag", None),
                "pullback_hhhl_seen_flag": row.get("pullback_hhhl_seen_flag", None),
                "dir_turn_upper_price": dir_turn_upper_price,
                "dir_turn_atr_raw": dir_turn_atr_raw,
                "dir_turn_tol_price": dir_turn_tol_price,
                "dir_turn_band_low": dir_turn_band_low,
                "dir_turn_band_high": dir_turn_band_high,
                "rope": row.get("rope", None),
                "close": row.get("close", None),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
    return events


def process_stock(
    symbol: str,
    name: str,
    freqs: List[str],
    bars: int,
    target_date: Optional[str],
) -> Tuple[int, int]:
    from features.merged_atr_rope_breakout_volume_delta import MergedEngine

    total_dir_turn = 0
    total_pullback = 0

    for freq in freqs:
        df = load_kline_data_from_db(symbol, freq, bars, target_date)
        if df is None or len(df) < 50:
            return (-2, 0)

        if target_date:
            last_date = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1])[:10]
            if last_date != target_date:
                return (-1, 0)

        try:
            args = build_engine_args(symbol, freq, bars)
            engine = MergedEngine(df, args)
            engine.run()
            result_df = engine.df
        except Exception as e:
            logger.warning("计算 %s %s 因子失败: %s", symbol, freq, e)
            continue

        dir_turn_events = extract_dir_turn_events(result_df, symbol, name, freq, target_date)
        pullback_events = extract_pullback_buy_events(result_df, symbol, name, freq, target_date)

        with get_engine().connect() as conn:
            conn.execute(
                text("DELETE FROM breakout_dir_turn_events WHERE ts_code = :ts_code AND freq = :freq"),
                {"ts_code": symbol, "freq": freq},
            )
            conn.commit()
        with get_engine().connect() as conn:
            conn.execute(
                text("DELETE FROM breakout_pullback_buy_events WHERE ts_code = :ts_code AND freq = :freq"),
                {"ts_code": symbol, "freq": freq},
            )
            conn.commit()

        if dir_turn_events:
            from datasource.database import bulk_upsert
            with get_engine().connect() as conn:
                bulk_upsert(conn, "breakout_dir_turn_events", pd.DataFrame(dir_turn_events), ["ts_code", "freq", "event_time"])
            total_dir_turn += len(dir_turn_events)

        if pullback_events:
            from datasource.database import bulk_upsert
            with get_engine().connect() as conn:
                bulk_upsert(conn, "breakout_pullback_buy_events", pd.DataFrame(pullback_events), ["ts_code", "freq", "buy_time"])
            total_pullback += len(pullback_events)

    return total_dir_turn, total_pullback


def process_stock_no_write(
    symbol: str,
    name: str,
    freq: str,
    bars: int,
    target_date: Optional[str],
) -> Tuple[int, int, List[Dict], List[Dict]]:
    from features.merged_atr_rope_breakout_volume_delta import MergedEngine

    df = load_kline_data_from_db(symbol, freq, bars, target_date)
    if df is None or len(df) < 50:
        return (-2, 0, [], [])

    if target_date:
        last_date = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1])[:10]
        if last_date != target_date:
            return (-1, 0, [], [])

    try:
        args = build_engine_args(symbol, freq, bars)
        engine = MergedEngine(df, args)
        engine.run()
        result_df = engine.df
    except Exception as e:
        logger.warning("计算 %s %s 因子失败: %s", symbol, freq, e)
        return (0, 0, [], [])

    dir_turn_events = extract_dir_turn_events(result_df, symbol, name, freq, target_date)
    pullback_events = extract_pullback_buy_events(result_df, symbol, name, freq, target_date)

    return (0, len(pullback_events), dir_turn_events, pullback_events)


def process_stock_for_dates(
    preloaded_data: pd.DataFrame,
    ts_code: str,
    name: str,
    freq: str,
    bars: int,
    dates: List[str],
) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """
    对已加载的行情数据，按日期列表逐个计算事件

    Args:
        preloaded_data: 从DB加载的行情数据
        ts_code: 股票代码
        name: 股票名称
        freq: 周期 ('d', '60m')
        bars: K线数量（用于验证数据充足性）
        dates: 要计算的日期列表

    Returns:
        {
            '2026-03-01': ([dir_events], [pullback_events]),
            ...
        }
    """
    from features.merged_atr_rope_breakout_volume_delta import MergedEngine

    results: Dict[str, Tuple[List[Dict], List[Dict]]] = {}

    if preloaded_data is None or len(preloaded_data) < 50:
        return results

    for target_date in dates:
        df = preloaded_data[preloaded_data.index <= target_date].copy()
        if len(df) < 50:
            results[target_date] = ([], [])
            continue

        try:
            args = build_engine_args(ts_code, freq, bars)
            engine = MergedEngine(df, args)
            engine.run()
            result_df = engine.df
        except Exception as e:
            logger.warning("计算 %s %s %s 因子失败: %s", ts_code, freq, target_date, e)
            results[target_date] = ([], [])
            continue

        dir_events = extract_dir_turn_events(result_df, ts_code, name, freq, target_date)
        pullback_events = extract_pullback_buy_events(result_df, ts_code, name, freq, target_date)
        results[target_date] = (dir_events, pullback_events)

    return results


def ensure_tables_exist():
    from datasource.database import DATABASE_URL
    is_postgres = not DATABASE_URL.startswith("sqlite")

    def pk_sql():
        if is_postgres:
            return "id SERIAL PRIMARY KEY"
        return "id INTEGER PRIMARY KEY AUTOINCREMENT"

    dir_turn_table = f"""
    CREATE TABLE IF NOT EXISTS breakout_dir_turn_events (
        {pk_sql()},
        ts_code VARCHAR(20) NOT NULL,
        name VARCHAR(50),
        event_time VARCHAR(30) NOT NULL,
        freq VARCHAR(10) NOT NULL,
        breakout_quality_score REAL,
        breakout_quality_grade VARCHAR(5),
        score_trend_total REAL,
        score_candle_total REAL,
        score_volume_total REAL,
        score_freshness_total REAL,
        score_bg_rope_slope REAL,
        score_bg_dist_to_rope REAL,
        score_bg_consolidation REAL,
        score_candle_close_pos REAL,
        score_candle_body_to_range REAL,
        score_candle_upper_wick REAL,
        score_volume_vol_z REAL,
        score_volume_vol_record REAL,
        score_freshness_count REAL,
        score_freshness_cum_gain REAL,
        breakout_action VARCHAR(30),
        breakout_freshness_count REAL,
        breakout_freshness_cum_gain REAL,
        rope_slope_atr_5 REAL,
        dist_to_rope_atr REAL,
        consolidation_bars REAL,
        vol_zscore REAL,
        vol_record_days REAL,
        dir_turn_upper_price REAL,
        dir_turn_atr_raw REAL,
        dir_turn_tol_price REAL,
        dir_turn_band_low REAL,
        dir_turn_band_high REAL,
        created_at VARCHAR(30),
        UNIQUE(ts_code, freq, event_time)
    );
    """
    pullback_table = f"""
    CREATE TABLE IF NOT EXISTS breakout_pullback_buy_events (
        {pk_sql()},
        ts_code VARCHAR(20) NOT NULL,
        name VARCHAR(50),
        buy_time VARCHAR(30) NOT NULL,
        freq VARCHAR(10) NOT NULL,
        buy_type VARCHAR(30),
        source_breakout_time VARCHAR(30),
        source_breakout_index REAL,
        breakout_to_buy_bars INTEGER,
        breakout_quality_score REAL,
        score_trend_total REAL,
        score_candle_total REAL,
        score_volume_total REAL,
        score_freshness_total REAL,
        signal_note TEXT,
        pullback_touch_support_flag REAL,
        pullback_hhhl_seen_flag REAL,
        dir_turn_upper_price REAL,
        dir_turn_atr_raw REAL,
        dir_turn_tol_price REAL,
        dir_turn_band_low REAL,
        dir_turn_band_high REAL,
        rope REAL,
        close REAL,
        created_at VARCHAR(30),
        UNIQUE(ts_code, freq, buy_time)
    );
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(dir_turn_table))
        conn.execute(text(pullback_table))
    engine.dispose()
    logger.info("数据库表已确认存在")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Breakout Events Scanner")
    p.add_argument("--date", default="", help="指定扫描日期，格式 YYYY-MM-DD，默认今天")
    p.add_argument("--bars", type=int, default=255, help="回溯K线数量，默认255")
    p.add_argument("--freqs", default="d,60m", help="扫描周期，逗号分隔，默认 d,60m")
    p.add_argument("--codes", default="", help="逗号分隔的股票代码，如 600519,000001")
    p.add_argument("--len", type=int, default=14, dest="len", help="ATR周期，默认14")
    p.add_argument("--multi", type=float, default=1.5, help="ATR倍数，默认1.5")
    p.add_argument("--log-level", default="INFO", help="日志级别: DEBUG/INFO/WARNING/ERROR")
    p.add_argument("--batch-backfill", default="", help="批量回补开始日期（到上一交易日），格式 YYYY-MM-DD，如 2026-02-01")
    p.add_argument("--end-date", default="", help="批量回补结束日期，格式 YYYY-MM-DD，如 2026-03-15")
    return p


def resolve_stock_list(args: argparse.Namespace) -> List[Dict]:
    if args.codes:
        return [{"ts_code": c.strip(), "name": ""} for c in args.codes.split(",") if c.strip()]
    return load_stock_pool_from_db()


def get_scanned_dates() -> set:
    engine = get_engine()
    sql = "SELECT DISTINCT substr(event_time, 1, 10) as dt FROM breakout_dir_turn_events"
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn)
    engine.dispose()
    return set(df["dt"].tolist())


def batch_backfill(start_date: str, bars: int, log_level: str, end_date: str = None) -> None:
    import qstock as qs

    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    today = datetime.now().strftime("%Y-%m-%d")
    end_date = end_date or today

    trade_dates = qs.get_dates(start_date, end_date)
    trade_dates = [d[:4] + "-" + d[4:6] + "-" + d[6:8] for d in trade_dates]

    scanned = get_scanned_dates()
    to_scan = [d for d in trade_dates if d not in scanned]

    logger.info("批量回补范围: %s ~ %s", start_date, end_date)
    logger.info("交易日总数: %d，已扫描跳过: %d，待扫描: %d",
                 len(trade_dates), len(scanned), len(to_scan))

    if not to_scan:
        logger.info("没有需要回补的日期，退出")
        return

    logger.info("待扫描日期列表: %s", to_scan)

    total_dir_turn = 0
    total_pullback = 0
    processed = 0

    stock_list = load_stock_pool_from_db()
    logger.info("股票池: %d 只", len(stock_list))

    freq = "d"

    for stock in tqdm(stock_list, desc=f"批量回补 {start_date}~{end_date}", unit="只"):
        ts_code = stock["ts_code"]
        name = stock.get("name", "") or ""
        market = infer_market(ts_code)
        if market is None:
            continue

        preloaded_data = load_kline_data_from_db(ts_code, freq, bars, to_scan[-1])
        if preloaded_data is None:
            continue

        stock_results = process_stock_for_dates(
            preloaded_data, ts_code, name, freq, bars, to_scan
        )

        for scan_date in to_scan:
            dir_events, pullback_events = stock_results.get(scan_date, ([], []))

            if dir_events or pullback_events:
                from datasource.database import bulk_upsert
                engine = get_engine()
                with engine.connect() as conn:
                    if dir_events:
                        bulk_upsert(conn, "breakout_dir_turn_events", pd.DataFrame(dir_events), ["ts_code", "freq", "event_time"])
                    if pullback_events:
                        bulk_upsert(conn, "breakout_pullback_buy_events", pd.DataFrame(pullback_events), ["ts_code", "freq", "buy_time"])
                engine.dispose()

                total_dir_turn += len(dir_events)
                total_pullback += len(pullback_events)

        processed += 1

        if processed % 100 == 0:
            logger.info("已处理 %d / %d 只股票", processed, len(stock_list))

    logger.info("=" * 60)
    logger.info("批量回补完成！共处理 %d 只股票", processed)
    logger.info("翻多事件: %d 个，回踩买点: %d 个", total_dir_turn, total_pullback)


def main():
    args = build_parser().parse_args()

    if args.batch_backfill:
        batch_backfill(
            start_date=args.batch_backfill,
            bars=args.bars,
            log_level=args.log_level,
            end_date=args.end_date if args.end_date else None,
        )
        return

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    target_date = args.date if args.date else datetime.now().strftime("%Y-%m-%d")
    freqs = [normalize_freq(f.strip()) for f in args.freqs.split(",")]

    ensure_tables_exist()

    stock_list = resolve_stock_list(args)
    logger.info("待扫描股票数量: %s", len(stock_list))
    logger.info("目标日期: %s", target_date)

    # 按天收集结果，最后批量写入
    day_dir_events: List[Dict] = []
    day_pullback_events: List[Dict] = []
    processed = 0
    skipped_no_data = 0
    skipped_insufficient = 0

    for idx, stock in enumerate(tqdm(stock_list, desc=f"扫描 {target_date}", unit="只"), start=1):
        ts_code = stock["ts_code"]
        name = stock.get("name", "") or ""
        market = infer_market(ts_code)
        if market is None:
            continue

        # 遍历所有周期
        for freq in freqs:
            # 使用不写入的版本收集事件
            code, _, dir_events, pull_events = process_stock_no_write(ts_code, name, freq, args.bars, target_date)
            if code == -1:
                skipped_no_data += 1
                continue
            if code == -2:
                skipped_insufficient += 1
                continue

            day_dir_events.extend(dir_events)
            day_pullback_events.extend(pull_events)
            processed += 1

        if idx % 500 == 0:
            logger.info("已处理 %s / %s", idx, len(stock_list))

    # 批量写入当天所有结果
    if day_dir_events or day_pullback_events:
        from datasource.database import bulk_upsert
        engine = get_engine()
        try:
            # 使用 engine.begin() 自动管理事务
            with engine.begin() as conn:
                if day_dir_events:
                    bulk_upsert(conn, "breakout_dir_turn_events", pd.DataFrame(day_dir_events), ["ts_code", "freq", "event_time"], auto_commit=False)
                if day_pullback_events:
                    bulk_upsert(conn, "breakout_pullback_buy_events", pd.DataFrame(day_pullback_events), ["ts_code", "freq", "buy_time"], auto_commit=False)
            logger.info("成功写入数据库: 翻多事件 %s 个, 回踩买点 %s 个", len(day_dir_events), len(day_pullback_events))
        except Exception as e:
            logger.error("数据库连接或写入异常: %s", e)
            # 保存到本地文件作为备份
            import json
            from datetime import datetime
            backup_file = f"breakout_events_backup_{target_date}_{datetime.now().strftime('%H%M%S')}.json"
            backup_data = {
                "target_date": target_date,
                "dir_turn_events": day_dir_events,
                "pullback_events": day_pullback_events,
                "error": str(e)
            }
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            logger.warning("事件数据已备份到文件: %s", backup_file)
        finally:
            engine.dispose()

    logger.info("扫描完成！共处理 %s 只股票 (截止 %s)", processed, target_date)
    logger.info("翻多事件: %s 个，回踩买点: %s 个", len(day_dir_events), len(day_pullback_events))
    logger.info("跳过（无当天数据）: %s", skipped_no_data)
    logger.info("跳过（数据不足）: %s", skipped_insufficient)


if __name__ == "__main__":
    main()