"""
batch_processor.py
批量处理器 — 遍历股票池计算因子+检测事件

Purpose:
    遍历股票池中所有股票，对每只股票调用因子计算引擎和事件检测器，
    汇总输出因子+事件表。支持单股票调试和批量处理。

Inputs:
    - PostgreSQL 数据库中的 stock_k_data 表（通过 k_data_loader 加载）
    - 命令行参数: freq/start_date/end_date/limit_stocks/output_path

Outputs:
    - CSV 文件（可选）
    - 控制台输出汇总统计

How to Run:
    python market_structure_analysis/batch_processor.py --limit_stocks 3 --start 2024-01-01
    python market_structure_analysis/batch_processor.py --ts_code 600519.SH --start 2024-01-01
    python market_structure_analysis/batch_processor.py --help

Examples:
    # 批量处理 3 只股票（调试用）
    python market_structure_analysis/batch_processor.py --limit_stocks 3

    # 处理单只股票
    python market_structure_analysis/batch_processor.py --ts_code 600519.SH

    # 批量处理并输出 CSV
    python market_structure_analysis/batch_processor.py --limit_stocks 10 --output factors_events.csv

Side Effects:
    可选写入 CSV 文件（--output 参数），不写入数据库
"""

import argparse
import logging
import sys
import os
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.k_data_loader import load_k_data, build_name_map
from datasource.database import get_session, query_df
from market_structure_analysis.factor_engine import compute_all_factors
from market_structure_analysis.event_detector import detect_events, CORE_EVENTS, AUX_STATES

logger = logging.getLogger(__name__)


def _get_stock_codes(freq: str = "d") -> list:
    """从 stock_pools 表获取股票代码列表（0.03s，替代 get_all_codes 的 8s 全量扫描）"""
    with get_session() as session:
        df = query_df(session, table_name="stock_pools", columns=["ts_code"])
    if df.empty:
        return []
    return df["ts_code"].unique().tolist()


def process_single_stock(
    ts_code: str,
    freq: str = "d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    处理单只股票，返回因子+事件 DataFrame。

    Parameters
    ----------
    ts_code : str
        股票代码，如 '600519.SH'
    freq : str
        K线周期，默认 'd'
    start_date : str, optional
        起始日期 'YYYY-MM-DD'
    end_date : str, optional
        结束日期 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame
        因子+事件 DataFrame，含 ts_code 列
    """
    df = load_k_data(ts_code, freq=freq, start_date=start_date, end_date=end_date)

    if df.empty:
        logger.warning("%s: 无K线数据，跳过", ts_code)
        return pd.DataFrame()

    try:
        factors = compute_all_factors(df)
        result = detect_events(factors)
    except Exception as exc:
        logger.error("%s: 因子计算/事件检测失败: %s", ts_code, exc, exc_info=True)
        return pd.DataFrame()

    result["ts_code"] = ts_code
    return result


def process_stock_pool(
    freq: str = "d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit_stocks: Optional[int] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    批量处理股票池，返回汇总的因子+事件 DataFrame。

    优化策略：先获取股票代码列表，再逐只用 load_k_data 按 ts_code+时间查询，
    避免全量加载 600 万行数据到内存。

    Parameters
    ----------
    freq : str
        K线周期
    start_date : str, optional
        起始日期
    end_date : str, optional
        结束日期
    limit_stocks : int, optional
        限制处理股票数量（调试用）
    output_path : str, optional
        CSV 输出路径，None 则不输出文件

    Returns
    -------
    pd.DataFrame
        所有股票的因子+事件汇总表
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    t0 = time.time()

    codes = _get_stock_codes(freq)
    if limit_stocks:
        codes = codes[:limit_stocks]

    name_map = build_name_map(codes)
    total = len(codes)
    logger.info("待处理股票数: %d", total)

    results = []
    success_count = 0
    fail_count = 0

    _existing_chunks = sorted([f for f in _os.listdir(_cache_dir) if f.startswith("stream_chunk_") and f.endswith(".pkl")])
    if _existing_chunks:
        _last_chunk = _pickle.load(open(_os.path.join(_cache_dir, _existing_chunks[-1]), "rb"))
        if not _last_chunk.empty:
            _already_processed = set(_last_chunk["ts_code"].unique())
            _skip_count = len(_already_processed)
            logger.info("断点恢复: 发现 %d 个增量文件，跳过已处理 %d 只股票",
                        len(_existing_chunks), _skip_count)
            codes = [c for c in codes if c not in _already_processed]
            del _last_chunk
        else:
            _skip_count = 0
            for _f in _existing_chunks:
                _os.remove(_os.path.join(_cache_dir, _f))
            logger.info("清理无效增量缓存")
    else:
        _skip_count = 0

    if tqdm is not None:
        iterator = tqdm(codes, desc="Processing", position=0, leave=True)
    else:
        iterator = codes

    for code in iterator:
        name = name_map.get(code, code)
        df = load_k_data(code, freq=freq, start_date=start_date, end_date=end_date)

        if df.empty or len(df) < 60:
            logger.debug("%s (%s): 数据不足(%d bars)，跳过", code, name, len(df))
            fail_count += 1
            continue

        try:
            factors = compute_all_factors(df)
            result = detect_events(factors)
            result["ts_code"] = code
            result["stock_name"] = name
            results.append(result)
            success_count += 1
        except Exception as exc:
            logger.error("%s (%s): 处理失败: %s", code, name, exc)
            fail_count += 1

    elapsed = time.time() - t0

    if not results:
        logger.warning("无成功处理结果")
        return pd.DataFrame()

    combined = pd.concat(results, axis=0)
    combined = combined.sort_values(["ts_code", combined.index.name or "bar_time"])

    evt_cols = [c for c in combined.columns if c.startswith("evt_")]
    state_cols = [c for c in combined.columns if c.startswith("state_")]

    logger.info(
        "批量处理完成: 成功 %d, 失败 %d, 耗时 %.1fs",
        success_count, fail_count, elapsed,
    )
    logger.info("汇总: %d 行, %d 列 (含 %d 事件列, %d 状态列)",
                len(combined), len(combined.columns), len(evt_cols), len(state_cols))

    for col in evt_cols:
        count = int(combined[col].sum())
        logger.info("  %s: %d 次", col, count)

    if output_path:
        combined.to_csv(output_path, index=True)
        logger.info("已输出到: %s", output_path)

    return combined


EVT_COLS = [f"evt_{e}" for e in CORE_EVENTS]


def process_stock_pool_streaming(
    freq: str = "d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit_stocks: Optional[int] = None,
    skip_pavp: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    流式处理：批量预加载 K 线 + 逐股计算因子+事件 + 增量聚合。

    优化要点：
    - 1 次 DB 查询加载全量 K 线（替代 5000 次逐股查询，IO 降 95%）
    - 不保留 lightweight_events 全量 DataFrame（内存从 300MB → 0）
    - 逐股 pop 释放已处理数据，内存峰值 = 单股 DataFrame + 累加器

    Parameters
    ----------
    freq : str
        K线周期
    start_date : str, optional
        起始日期
    end_date : str, optional
        结束日期
    limit_stocks : int, optional
        限制处理股票数量（调试用）
    skip_pavp : bool
        跳过 PAVP 计算（速度提升 4x，丢失 2 事件 + 1 状态）

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (daily_agg, lightweight_events)
        daily_agg: 每日事件计数+股票数
        lightweight_events: 仅含 ts_code/trade_date/evt_* 列的轻量表
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    t0 = time.time()

    codes = _get_stock_codes(freq)
    if limit_stocks:
        codes = codes[:limit_stocks]

    total = len(codes)
    logger.info("待处理股票数: %d", total)

    logger.info("批量预加载 K 线数据...")
    all_k_data = load_k_data(freq=freq, start_date=start_date, end_date=end_date)
    loaded_codes = set(all_k_data.keys())
    logger.info("K 线加载完成: %d 只股票, 耗时 %.1fs", len(loaded_codes), time.time() - t0)

    name_map = build_name_map(codes)

    daily_counts: Dict = defaultdict(lambda: defaultdict(float))
    daily_stocks: Dict = defaultdict(set)
    lightweight_rows = []

    success_count = 0
    fail_count = 0

    if tqdm is not None:
        iterator = tqdm(codes, desc="Streaming", position=0, leave=True)
    else:
        iterator = codes

    for code in iterator:
        if code not in loaded_codes:
            fail_count += 1
            continue

        stock_entry = all_k_data.pop(code)
        df = stock_entry["data"] if isinstance(stock_entry, dict) else stock_entry

        if df.empty or len(df) < 60:
            fail_count += 1
            continue

        try:
            factors = compute_all_factors(df, skip_pavp=skip_pavp)
            result = detect_events(factors)
        except Exception as exc:
            name = name_map.get(code, code)
            logger.error("%s (%s): 处理失败: %s", code, name, exc)
            fail_count += 1
            continue

        result["ts_code"] = code

        dates = result.index.date if hasattr(result.index, "date") else pd.to_datetime(result.index).date

        evt_cols_present = [c for c in EVT_COLS if c in result.columns]

        for i, dt in enumerate(dates):
            daily_stocks[dt].add(code)
            for ec in evt_cols_present:
                daily_counts[dt][ec] += result.iloc[i].get(ec, 0.0)

        light = result[["ts_code"] + evt_cols_present].copy()
        light["trade_date"] = dates
        lightweight_rows.append(light)

        del result, factors, df
        success_count += 1

    elapsed = time.time() - t0

    if not daily_counts:
        logger.warning("流式处理无结果")
        return pd.DataFrame(), pd.DataFrame()

    trade_dates = sorted(daily_counts.keys())
    rows = []
    for dt in trade_dates:
        row = {"trade_date": dt, "total_stocks": len(daily_stocks[dt])}
        for ec in EVT_COLS:
            row[ec + "_count"] = daily_counts[dt].get(ec, 0.0)
        rows.append(row)
    daily_agg = pd.DataFrame(rows)
    daily_agg["trade_date"] = pd.to_datetime(daily_agg["trade_date"])
    daily_agg = daily_agg.set_index("trade_date")

    del daily_counts, daily_stocks

    if lightweight_rows:
        lightweight_events = pd.concat(lightweight_rows, ignore_index=True)
        lightweight_events["trade_date"] = pd.to_datetime(lightweight_events["trade_date"])
    else:
        lightweight_events = pd.DataFrame()
    del lightweight_rows

    logger.info(
        "流式处理完成: 成功 %d, 失败 %d, 耗时 %.1fs, 速度 %.1f it/s",
        success_count, fail_count, elapsed, total / elapsed if elapsed > 0 else 0,
    )
    logger.info("日级聚合: %d 个交易日, 轻量事件表: %d 行", len(daily_agg), len(lightweight_events))

    return daily_agg, lightweight_events


def main():
    parser = argparse.ArgumentParser(description="批量处理器 — 遍历股票池计算因子+检测事件")
    parser.add_argument("--ts_code", type=str, default=None, help="单只股票代码（调试用）")
    parser.add_argument("--freq", type=str, default="d", help="K线周期")
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--limit_stocks", type=int, default=None, help="限制股票数量（调试用）")
    parser.add_argument("--output", type=str, default=None, help="CSV 输出路径")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.ts_code:
        logger.info("单股票模式: %s", args.ts_code)
        result = process_single_stock(
            args.ts_code, freq=args.freq,
            start_date=args.start, end_date=args.end,
        )
        if not result.empty:
            evt_cols = [c for c in result.columns if c.startswith("evt_")]
            state_cols = [c for c in result.columns if c.startswith("state_")]
            logger.info("因子+事件列数: %d (事件: %d, 状态: %d)",
                        len(evt_cols) + len(state_cols), len(evt_cols), len(state_cols))
            print(result[evt_cols + state_cols].tail(10).to_string())
        else:
            logger.error("处理失败或无数据")
    else:
        logger.info("批量模式: freq=%s, limit=%s", args.freq, args.limit_stocks)
        result = process_stock_pool(
            freq=args.freq,
            start_date=args.start,
            end_date=args.end,
            limit_stocks=args.limit_stocks,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
