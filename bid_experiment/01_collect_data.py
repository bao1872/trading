# -*- coding: utf-8 -*-
"""
竞价数据采集脚本

Purpose:
    从 pytdx 和 tushare 采集竞价拍卖原始数据，保存为 parquet 文件

Inputs:
    - pytdx: 逐笔成交数据（含竞价标识 buyorsell=8/2/0/1）
    - pytdx: 日线 K 线、1 分钟 K 线
    - tushare: stk_auction 竞价数据

Outputs:
    - raw_data/{symbol}/tick/{date_int}.parquet       全天逐笔成交
    - raw_data/{symbol}/auction_ref/{date_int}.parquet 竞价参考价 (buyorsell=8)
    - raw_data/{symbol}/auction_match/{date_int}.parquet 竞价撮合结果 (buyorsell=2)
    - raw_data/{symbol}/daily.parquet                  日线 K 线
    - raw_data/{symbol}/minute_1m.parquet              1 分钟 K 线
    - raw_data/{symbol}/tushare_auction.parquet        tushare stk_auction

How to Run:
    python bid_experiment/01_collect_data.py --symbol 000001 --start-date 20260508 --end-date 20260512
    python bid_experiment/01_collect_data.py --symbol 600000 --start-date 20260501 --end-date 20260512 --sample-limit 3

Side Effects:
    - 写文件（parquet），不写数据库
    - 连接 pytdx 服务器和 tushare API
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bid_experiment.bid_config import (
    RAW_DATA_DIR,
    BUYORSELL_AUCTION_REF,
    BUYORSELL_AUCTION_MATCH,
    BUYORSELL_BUY,
    BUYORSELL_SELL,
)
from datasource.pytdx_client import connect_pytdx, get_raw_tick_data, get_kline_data, market_from_code
from tushare_data.fetcher import get_pro

logger = logging.getLogger(__name__)

MAX_RETRY = 5
RETRY_BASE_DELAY = 2.0


def get_trading_dates(start_date: int, end_date: int) -> List[int]:
    """获取交易日列表

    优先使用 tushare trade_cal，失败则降级为 weekday 判断。

    Args:
        start_date: 起始日期整数，如 20260501
        end_date: 结束日期整数，如 20260512

    Returns:
        交易日整数列表，如 [20260508, 20260509, ...]
    """
    start_str = str(start_date)
    end_str = str(end_date)

    try:
        pro = get_pro()
        df = pro.trade_cal(
            exchange="SSE",
            start_date=start_str,
            end_date=end_str,
            is_open="1",
        )
        if df is not None and not df.empty:
            dates = sorted(df["cal_date"].astype(int).tolist())
            logger.info(f"tushare trade_cal 返回 {len(dates)} 个交易日")
            return dates
    except Exception as e:
        logger.warning(f"tushare trade_cal 查询失败，降级为 weekday: {e}")

    start_dt = datetime.strptime(start_str, "%Y%m%d")
    end_dt = datetime.strptime(end_str, "%Y%m%d")
    dates = []
    current = start_dt
    while current <= end_dt:
        if current.weekday() < 5:
            dates.append(int(current.strftime("%Y%m%d")))
        current += timedelta(days=1)
    logger.warning(f"weekday 降级模式返回 {len(dates)} 个日期，可能含节假日")
    return dates


def _symbol_to_ts_code(symbol: str) -> str:
    """将纯数字代码转为 tushare 格式，如 000001 -> 000001.SZ, 600000 -> 600000.SH"""
    symbol = str(symbol).strip()
    if "." in symbol:
        return symbol
    market = market_from_code(symbol)
    suffix = "SH" if market == 1 else "SZ"
    return f"{symbol}.{suffix}"


def collect_tushare_auction(symbol: str, start_date: int, end_date: int) -> pd.DataFrame:
    """采集 tushare stk_auction 数据

    Args:
        symbol: 股票代码，如 '000001'
        start_date: 起始日期整数
        end_date: 结束日期整数

    Returns:
        DataFrame，tushare stk_auction 原始数据
    """
    ts_code = _symbol_to_ts_code(symbol)
    start_str = str(start_date)
    end_str = str(end_date)

    for attempt in range(MAX_RETRY):
        try:
            pro = get_pro()
            df = pro.stk_auction(
                ts_code=ts_code,
                start_date=start_str,
                end_date=end_str,
            )
            if df is not None and not df.empty:
                logger.info(f"tushare stk_auction 返回 {len(df)} 条记录 ({ts_code})")
                return df
            logger.warning(f"tushare stk_auction 返回空数据 ({ts_code})")
            return pd.DataFrame()
        except Exception as e:
            if attempt < MAX_RETRY - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"tushare stk_auction 第 {attempt + 1} 次失败，{delay:.0f}s 后重试: {e}")
                time.sleep(delay)
            else:
                logger.error(f"tushare stk_auction 采集失败，已重试 {MAX_RETRY} 次: {e}")
                raise


def _extract_auction_ref(tick_df: pd.DataFrame) -> pd.DataFrame:
    """从逐笔数据中提取竞价参考价 (buyorsell=8, 09:15-09:24)"""
    if tick_df.empty:
        return pd.DataFrame()
    mask = (
        (tick_df["buyorsell"] == BUYORSELL_AUCTION_REF)
        & (tick_df["time"] >= "09:15")
        & (tick_df["time"] <= "09:24:59")
    )
    return tick_df.loc[mask].reset_index(drop=True)


def _extract_auction_match(tick_df: pd.DataFrame) -> pd.DataFrame:
    """从逐笔数据中提取竞价撮合结果 (buyorsell=2, 09:25)"""
    if tick_df.empty:
        return pd.DataFrame()
    mask = (
        (tick_df["buyorsell"] == BUYORSELL_AUCTION_MATCH)
        & (tick_df["time"] >= "09:25")
        & (tick_df["time"] < "09:30")
    )
    return tick_df.loc[mask].reset_index(drop=True)


def _extract_post_open(tick_df: pd.DataFrame) -> pd.DataFrame:
    """从逐笔数据中提取开盘后成交 (buyorsell=0/1, 09:30-10:00)"""
    if tick_df.empty:
        return pd.DataFrame()
    mask = (
        (tick_df["buyorsell"].isin([BUYORSELL_BUY, BUYORSELL_SELL]))
        & (tick_df["time"] >= "09:30")
        & (tick_df["time"] <= "10:00")
    )
    return tick_df.loc[mask].reset_index(drop=True)


def _save_parquet(df: pd.DataFrame, filepath: str) -> None:
    """保存 DataFrame 为 parquet，自动创建目录"""
    if df.empty:
        logger.warning(f"数据为空，跳过保存: {filepath}")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, index=False)
    logger.info(f"已保存: {filepath} ({len(df)} 条)")


def collect_stock_data(
    symbol: str,
    start_date: int,
    end_date: int,
    output_dir: Optional[str] = None,
    sample_limit: Optional[int] = None,
) -> dict:
    """采集单只股票的竞价原始数据

    Args:
        symbol: 股票代码，如 '000001'
        start_date: 起始日期整数，如 20260501
        end_date: 结束日期整数，如 20260512
        output_dir: 输出根目录，默认 RAW_DATA_DIR
        sample_limit: 限制采集交易日数量，用于小批量验证

    Returns:
        采集结果摘要 dict
    """
    if output_dir is None:
        output_dir = RAW_DATA_DIR

    trading_dates = get_trading_dates(start_date, end_date)
    if sample_limit is not None and sample_limit > 0:
        trading_dates = trading_dates[:sample_limit]
        logger.info(f"sample_limit={sample_limit}，截取前 {len(trading_dates)} 个交易日")

    if not trading_dates:
        logger.warning("无交易日可采集")
        return {"symbol": symbol, "days_collected": 0, "days_failed": 0, "details": []}

    api = connect_pytdx()
    summary = {
        "symbol": symbol,
        "days_collected": 0,
        "days_failed": 0,
        "details": [],
    }

    try:
        daily_kline = get_kline_data(api, symbol, "d", 500)
        if not daily_kline.empty:
            daily_path = os.path.join(output_dir, symbol, "daily.parquet")
            _save_parquet(daily_kline, daily_path)

        minute_kline = get_kline_data(api, symbol, "1m", 5000)
        if not minute_kline.empty:
            minute_path = os.path.join(output_dir, symbol, "minute_1m.parquet")
            _save_parquet(minute_kline, minute_path)

        for date_int in tqdm(trading_dates, desc=f"采集 {symbol}", ncols=100):
            day_detail = {"date": date_int, "tick_rows": 0, "ref_rows": 0, "match_rows": 0, "post_open_rows": 0}

            tick_df = pd.DataFrame()
            for attempt in range(MAX_RETRY):
                try:
                    tick_df = get_raw_tick_data(api, symbol, date_int)
                    break
                except Exception as e:
                    if attempt < MAX_RETRY - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(f"{symbol} {date_int} tick 第 {attempt + 1} 次失败，{delay:.0f}s 后重试: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"{symbol} {date_int} tick 采集失败，已重试 {MAX_RETRY} 次: {e}")

            if tick_df.empty:
                logger.warning(f"{symbol} {date_int} 无逐笔数据，跳过")
                summary["days_failed"] += 1
                day_detail["error"] = "no_tick_data"
                summary["details"].append(day_detail)
                continue

            tick_path = os.path.join(output_dir, symbol, "tick", f"{date_int}.parquet")
            _save_parquet(tick_df, tick_path)
            day_detail["tick_rows"] = len(tick_df)

            auction_ref = _extract_auction_ref(tick_df)
            if not auction_ref.empty:
                ref_path = os.path.join(output_dir, symbol, "auction_ref", f"{date_int}.parquet")
                _save_parquet(auction_ref, ref_path)
            day_detail["ref_rows"] = len(auction_ref)

            auction_match = _extract_auction_match(tick_df)
            if not auction_match.empty:
                match_path = os.path.join(output_dir, symbol, "auction_match", f"{date_int}.parquet")
                _save_parquet(auction_match, match_path)
            day_detail["match_rows"] = len(auction_match)

            post_open = _extract_post_open(tick_df)
            day_detail["post_open_rows"] = len(post_open)

            summary["days_collected"] += 1
            summary["details"].append(day_detail)

            logger.info(
                f"{symbol} {date_int}: tick={len(tick_df)}, "
                f"ref={len(auction_ref)}, match={len(auction_match)}, "
                f"post_open={len(post_open)}"
            )
    finally:
        api.disconnect()

    try:
        tushare_auc = collect_tushare_auction(symbol, start_date, end_date)
        if not tushare_auc.empty:
            tushare_path = os.path.join(output_dir, symbol, "tushare_auction.parquet")
            _save_parquet(tushare_auc, tushare_path)
        summary["tushare_auction_rows"] = len(tushare_auc)
    except Exception as e:
        logger.error(f"tushare stk_auction 采集失败: {e}")
        summary["tushare_auction_rows"] = 0

    return summary


def _print_summary(summary: dict) -> None:
    """打印采集结果摘要"""
    print(f"\n{'='*60}")
    print(f"采集摘要: {summary['symbol']}")
    print(f"  成功天数: {summary['days_collected']}")
    print(f"  失败天数: {summary['days_failed']}")
    print(f"  tushare_auction: {summary.get('tushare_auction_rows', 'N/A')} 条")
    print(f"{'-'*60}")
    for d in summary["details"]:
        date_str = d["date"]
        tick = d["tick_rows"]
        ref = d["ref_rows"]
        match = d["match_rows"]
        post = d["post_open_rows"]
        err = d.get("error", "")
        line = f"  {date_str}: tick={tick}, ref={ref}, match={match}, post_open={post}"
        if err:
            line += f"  [ERROR: {err}]"
        print(line)
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="竞价数据采集脚本")
    parser.add_argument("--symbol", type=str, default="000001", help="股票代码（如 000001）")
    parser.add_argument("--start-date", type=str, default="20260501", help="起始日期 YYYYMMDD")
    parser.add_argument("--end-date", type=str, default="20260512", help="结束日期 YYYYMMDD")
    parser.add_argument("--sample-limit", type=int, default=None, help="限制采集交易日数量")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbol = args.symbol
    start_date = int(args.start_date)
    end_date = int(args.end_date)
    sample_limit = args.sample_limit

    logger.info(f"开始采集: symbol={symbol}, start={start_date}, end={end_date}, limit={sample_limit}")

    summary = collect_stock_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        sample_limit=sample_limit,
    )

    _print_summary(summary)


if __name__ == "__main__":
    main()
