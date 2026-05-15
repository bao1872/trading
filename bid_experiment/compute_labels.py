# -*- coding: utf-8 -*-
"""
标签计算模块

Purpose:
    基于逐笔成交数据（优先）或1分钟K线数据，计算每日交易标签

Inputs:
    - raw_data/{symbol}/tick/{date_int}.parquet: 逐笔成交数据（优先，历史覆盖广）
      columns: [time, price, vol, buyorsell], buyorsell=0/1 for 09:30+ trades
    - raw_data/{symbol}/minute_1m.parquet: 1分钟K线数据（备选，仅最近~20天）
      columns: [datetime, open, high, low, close, volume, amount], starts from 09:31

Outputs:
    - DataFrame with columns: stock_id, trade_date, + LABEL_COLS

How to Run:
    python -m bid_experiment.compute_labels
    python -m bid_experiment.compute_labels --symbol 000001

Side Effects:
    - 无（纯计算，不写库表/文件）
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from bid_experiment.bid_config import (
    BUY_NOW_MAE_THRESHOLD,
    BUY_NOW_MFE_THRESHOLD,
    BUY_NOW_RET_THRESHOLD,
    RAW_DATA_DIR,
    SELL_NOW_MIN_RET_THRESHOLD,
    SELL_NOW_RET_30M_THRESHOLD,
)
from bid_experiment.feature_columns import LABEL_COLS

logger = logging.getLogger(__name__)


def _compute_labels_from_tick(
    symbol: str, raw_data_dir: str
) -> pd.DataFrame:
    tick_dir = os.path.join(raw_data_dir, symbol, "tick")
    if not os.path.isdir(tick_dir):
        return pd.DataFrame()

    rows = []
    for f in sorted(os.listdir(tick_dir)):
        if not f.endswith(".parquet"):
            continue
        date_int = int(f.replace(".parquet", ""))
        tick_df = pd.read_parquet(os.path.join(tick_dir, f))
        if tick_df.empty:
            continue

        post_open = tick_df[
            (tick_df["buyorsell"].isin([0, 1])) & (tick_df["time"] >= "09:30")
        ].copy()
        if post_open.empty:
            continue

        post_open = post_open.sort_values("time").reset_index(drop=True)

        entry_price = post_open.iloc[0]["price"]

        first_10m = post_open[post_open["time"] < "09:40"]
        first_15m = post_open[post_open["time"] < "09:45"]
        first_30m = post_open[post_open["time"] < "10:00"]

        if first_10m.empty:
            continue

        mfe_10m = (first_10m["price"].max() / entry_price) - 1
        mae_10m = (first_10m["price"].min() / entry_price) - 1
        ret_10m = (first_10m.iloc[-1]["price"] / entry_price) - 1

        if first_30m.empty:
            ret_30m = ret_10m
        else:
            ret_30m = (first_30m.iloc[-1]["price"] / entry_price) - 1

        if first_15m.empty:
            future_min_ret_15m = mae_10m
        else:
            future_min_ret_15m = (first_15m["price"].min() / entry_price) - 1

        rows.append({
            "stock_id": symbol,
            "trade_date": date_int,
            "entry_price": entry_price,
            "exit_price": entry_price,
            "MFE_10m": mfe_10m,
            "MAE_10m": mae_10m,
            "RET_10m": ret_10m,
            "RET_30m": ret_30m,
            "future_min_ret_15m": future_min_ret_15m,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    result["y_buy_now"] = (
        (result["MFE_10m"] >= BUY_NOW_MFE_THRESHOLD)
        & (result["MAE_10m"] > BUY_NOW_MAE_THRESHOLD)
        & (result["RET_10m"] > BUY_NOW_RET_THRESHOLD)
    ).astype(int)

    result["y_sell_now"] = (
        (result["future_min_ret_15m"] <= SELL_NOW_MIN_RET_THRESHOLD)
        & (result["RET_30m"] <= SELL_NOW_RET_30M_THRESHOLD)
    ).astype(int)

    return result[["stock_id", "trade_date"] + LABEL_COLS]


def _compute_labels_from_minute(
    symbol: str, raw_data_dir: str
) -> pd.DataFrame:
    minute_path = os.path.join(raw_data_dir, symbol, "minute_1m.parquet")
    if not os.path.exists(minute_path):
        return pd.DataFrame()

    df = pd.read_parquet(minute_path)
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("datetime").reset_index(drop=True)
    df["trade_date"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)
    df["bar_idx"] = df.groupby("trade_date").cumcount()

    first_bar = df[df["bar_idx"] == 0].copy()
    first_bar["entry_price"] = np.where(
        first_bar["volume"] > 0,
        first_bar["amount"] / first_bar["volume"],
        first_bar["close"],
    )
    price_df = first_bar[["trade_date", "entry_price"]].copy()
    price_df["exit_price"] = price_df["entry_price"]

    df = df.merge(price_df, on="trade_date", how="left")

    df["high_ratio"] = df["high"] / df["entry_price"] - 1
    df["low_ratio_entry"] = df["low"] / df["entry_price"] - 1
    df["low_ratio_exit"] = df["low"] / df["exit_price"] - 1

    bars_10 = df[df["bar_idx"] < 10]
    bars_15 = df[df["bar_idx"] < 15]
    bars_30 = df[df["bar_idx"] < 30]

    mfe_10m = bars_10.groupby("trade_date")["high_ratio"].max()
    mae_10m = bars_10.groupby("trade_date")["low_ratio_entry"].min()

    bars_10_last = bars_10.groupby("trade_date").last()
    ret_10m = bars_10_last["close"] / bars_10_last["entry_price"] - 1

    bars_30_last = bars_30.groupby("trade_date").last()
    ret_30m = bars_30_last["close"] / bars_30_last["entry_price"] - 1

    future_min_ret_15m = bars_15.groupby("trade_date")["low_ratio_exit"].min()

    result = price_df.copy()
    result["stock_id"] = symbol
    result["MFE_10m"] = result["trade_date"].map(mfe_10m)
    result["MAE_10m"] = result["trade_date"].map(mae_10m)
    result["RET_10m"] = result["trade_date"].map(ret_10m)
    result["RET_30m"] = result["trade_date"].map(ret_30m)
    result["future_min_ret_15m"] = result["trade_date"].map(future_min_ret_15m)

    result["y_buy_now"] = (
        (result["MFE_10m"] >= BUY_NOW_MFE_THRESHOLD)
        & (result["MAE_10m"] > BUY_NOW_MAE_THRESHOLD)
        & (result["RET_10m"] > BUY_NOW_RET_THRESHOLD)
    ).astype(int)

    result["y_sell_now"] = (
        (result["future_min_ret_15m"] <= SELL_NOW_MIN_RET_THRESHOLD)
        & (result["RET_30m"] <= SELL_NOW_RET_30M_THRESHOLD)
    ).astype(int)

    return result[["stock_id", "trade_date"] + LABEL_COLS]


def compute_labels_for_stock(
    symbol: str, raw_data_dir: Optional[str] = None
) -> pd.DataFrame:
    """计算单只股票的每日标签

    优先使用逐笔成交数据（历史覆盖广），备选1分钟K线数据

    Args:
        symbol: 股票代码，如 '000001'
        raw_data_dir: 原始数据根目录，默认 RAW_DATA_DIR

    Returns:
        DataFrame with columns: stock_id, trade_date, + LABEL_COLS
    """
    if raw_data_dir is None:
        raw_data_dir = RAW_DATA_DIR

    tick_labels = _compute_labels_from_tick(symbol, raw_data_dir)
    if not tick_labels.empty:
        logger.info(f"从逐笔成交数据计算标签: {len(tick_labels)} 天")
        return tick_labels.reset_index(drop=True)

    minute_labels = _compute_labels_from_minute(symbol, raw_data_dir)
    if not minute_labels.empty:
        logger.info(f"从1分钟K线数据计算标签: {len(minute_labels)} 天")
        return minute_labels.reset_index(drop=True)

    logger.warning(f"无可用数据计算标签: {symbol}")
    return pd.DataFrame(columns=["stock_id", "trade_date"] + LABEL_COLS)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbol = "000001"
    labels_df = compute_labels_for_stock(symbol)

    if labels_df.empty:
        print(f"无标签数据，请先采集 {symbol} 的数据")
    else:
        print(f"\n=== {symbol} 标签数据 (前10行) ===")
        print(labels_df.head(10).to_string(index=False))

        print(f"\n=== 标签分布 ===")
        n = len(labels_df)
        print(f"总交易日: {n}")
        buy_pos = labels_df["y_buy_now"].sum()
        sell_pos = labels_df["y_sell_now"].sum()
        print(f"y_buy_now  正样本率: {buy_pos / n:.4f} ({buy_pos}/{n})")
        print(f"y_sell_now 正样本率: {sell_pos / n:.4f} ({sell_pos}/{n})")
