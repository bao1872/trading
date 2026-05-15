# -*- coding: utf-8 -*-
"""
竞价特征计算模块

Purpose:
    从原始竞价数据（pytdx tick + tushare stk_auction + 日线）计算5组共44个特征

Inputs:
    - raw_data/{symbol}/auction_ref/{date_int}.parquet: 竞价参考价 (buyorsell=8)
    - raw_data/{symbol}/auction_match/{date_int}.parquet: 竞价撮合结果 (buyorsell=2)
    - raw_data/{symbol}/daily.parquet: 日线数据
    - raw_data/{symbol}/tushare_auction.parquet: tushare竞价补充数据

Outputs:
    - DataFrame with columns: stock_id, trade_date, + ALL_FEATURE_COLS

How to Run:
    python -m bid_experiment.compute_features
    python -m bid_experiment.compute_features --symbol 000001

Side Effects:
    - 无（纯计算，不写库表/文件）
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from bid_experiment.bid_config import (
    RAW_DATA_DIR,
    ROLLING_WINDOW_60D,
    ROLLING_WINDOW_20D,
)
from bid_experiment.feature_columns import (
    ALL_FEATURE_COLS,
    AUCTION_STATE_COLS,
    AUCTION_VOLUME_COLS,
    AUCTION_PROCESS_COLS,
    BACKGROUND_COLS,
    ENV_COLS,
    FACTOR_CATEGORIES,
)

logger = logging.getLogger(__name__)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std()
    return (series - mean) / std.replace(0, np.nan)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _load_auction_data(symbol: str, raw_data_dir: str):
    ref_dir = os.path.join(raw_data_dir, symbol, "auction_ref")
    match_dir = os.path.join(raw_data_dir, symbol, "auction_match")

    ref_frames = {}
    match_frames = {}

    if os.path.isdir(ref_dir):
        for f in sorted(os.listdir(ref_dir)):
            if f.endswith(".parquet"):
                date_int = int(f.replace(".parquet", ""))
                df = pd.read_parquet(os.path.join(ref_dir, f))
                if not df.empty:
                    ref_frames[date_int] = df

    if os.path.isdir(match_dir):
        for f in sorted(os.listdir(match_dir)):
            if f.endswith(".parquet"):
                date_int = int(f.replace(".parquet", ""))
                df = pd.read_parquet(os.path.join(match_dir, f))
                if not df.empty:
                    match_frames[date_int] = df

    return ref_frames, match_frames


def _compute_auction_state_features(
    auction_prices: pd.Series,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    features = pd.DataFrame(index=auction_prices.index)

    prev_close = daily_df["close"].shift(1)
    prev_open = daily_df["open"].shift(1)
    ma5 = prev_close.rolling(5, min_periods=1).mean()
    ma10 = prev_close.rolling(10, min_periods=1).mean()
    prev_high = daily_df["high"].shift(1)
    prev_low = daily_df["low"].shift(1)
    atr = compute_atr(daily_df["high"], daily_df["low"], daily_df["close"], 14)

    features["auc_ret_close"] = auction_prices / prev_close - 1
    features["auc_ret_open_ref"] = auction_prices / prev_open - 1
    features["gap_flag_up"] = (features["auc_ret_close"] > 0).astype(int)
    features["gap_flag_down"] = (features["auc_ret_close"] < 0).astype(int)
    features["auction_vs_ma5"] = auction_prices / ma5 - 1
    features["auction_vs_ma10"] = auction_prices / ma10 - 1
    features["auction_vs_prev_high"] = auction_prices / prev_high - 1
    features["auction_vs_prev_low"] = auction_prices / prev_low - 1
    features["dist_to_prev_close_atr"] = (auction_prices - prev_close) / atr
    features["auc_ret_close_pct_60d"] = rolling_percentile(
        features["auc_ret_close"], ROLLING_WINDOW_60D
    )
    features["auc_ret_close_z_60d"] = rolling_zscore(
        features["auc_ret_close"], ROLLING_WINDOW_60D
    )

    return features


def _compute_auction_volume_features(
    auction_volumes: pd.Series,
    auction_amounts: pd.Series,
    auction_prices: pd.Series,
    daily_df: pd.DataFrame,
    tushare_df: Optional[pd.DataFrame],
    float_shares: Optional[pd.Series],
) -> pd.DataFrame:
    features = pd.DataFrame(index=auction_volumes.index)

    features["auc_volume"] = auction_volumes
    features["auc_amount"] = auction_amounts

    if float_shares is not None:
        features["auc_turnover"] = auction_volumes / float_shares
    elif tushare_df is not None and "turnover_rate" in tushare_df.columns:
        features["auc_turnover"] = tushare_df["turnover_rate"]
    else:
        features["auc_turnover"] = np.nan

    prev_amount = daily_df["amount"].shift(1)
    features["auc_amount_vs_prev"] = auction_amounts / prev_amount - 1
    features["auc_amount_vs_mean_20d"] = auction_amounts / auction_amounts.rolling(
        ROLLING_WINDOW_20D, min_periods=1
    ).mean()
    features["auc_volume_vs_mean_20d"] = auction_volumes / auction_volumes.rolling(
        ROLLING_WINDOW_20D, min_periods=1
    ).mean()
    features["auc_amount_pct_60d"] = rolling_percentile(
        auction_amounts, ROLLING_WINDOW_60D
    )
    features["auc_amount_z_60d"] = rolling_zscore(
        auction_amounts, ROLLING_WINDOW_60D
    )
    features["auc_turnover_pct_60d"] = rolling_percentile(
        features["auc_turnover"], ROLLING_WINDOW_60D
    )

    auc_ret_close = auction_prices / daily_df["close"].shift(1) - 1
    features["auc_ret_x_auc_amount_z"] = auc_ret_close * features["auc_amount_z_60d"]
    features["sign_auc_ret_x_auc_amount_pct"] = (
        np.sign(auc_ret_close) * features["auc_amount_pct_60d"]
    )

    return features


def _compute_auction_process_features(
    ref_frames: dict,
    match_frames: dict,
    trade_dates: list,
) -> pd.DataFrame:
    rows = []
    for date_int in trade_dates:
        row = {"trade_date": date_int}

        ref_df = ref_frames.get(date_int, pd.DataFrame())
        match_df = match_frames.get(date_int, pd.DataFrame())

        if ref_df.empty:
            for col in [
                "auc_price_slope_920_925", "auc_price_slope_last_1m",
                "auc_price_range_auction", "auc_price_final_vs_peak",
                "auc_price_final_vs_low", "auc_price_final_vs_920",
                "auc_ref_price_count",
            ]:
                row[col] = np.nan
            rows.append(row)
            continue

        ref_prices = ref_df["price"].values
        ref_times = ref_df["time"].values
        final_price = match_df.iloc[0]["price"] if not match_df.empty else ref_prices[-1]

        row["auc_ref_price_count"] = len(ref_prices)

        if len(ref_prices) >= 2:
            row["auc_price_range_auction"] = (
                (ref_prices.max() - ref_prices.min()) / ref_prices.min()
            )
            row["auc_price_final_vs_peak"] = final_price / ref_prices.max() - 1
            row["auc_price_final_vs_low"] = final_price / ref_prices.min() - 1
        else:
            row["auc_price_range_auction"] = 0.0
            row["auc_price_final_vs_peak"] = 0.0
            row["auc_price_final_vs_low"] = 0.0

        mask_920 = np.array([t >= "09:20" for t in ref_times])
        if mask_920.any():
            prices_920 = ref_prices[mask_920]
            idx = np.arange(len(prices_920))
            if len(prices_920) >= 2:
                slope = np.polyfit(idx, prices_920, 1)[0]
                row["auc_price_slope_920_925"] = slope / prices_920[0]
            else:
                row["auc_price_slope_920_925"] = 0.0

            price_at_920 = prices_920[0]
            row["auc_price_final_vs_920"] = final_price / price_at_920 - 1
        else:
            row["auc_price_slope_920_925"] = np.nan
            row["auc_price_final_vs_920"] = np.nan

        if len(ref_prices) >= 2:
            last_n = min(len(ref_prices), 3)
            last_prices = ref_prices[-last_n:]
            idx = np.arange(last_n)
            if last_n >= 2:
                slope = np.polyfit(idx, last_prices, 1)[0]
                row["auc_price_slope_last_1m"] = slope / last_prices[0]
            else:
                row["auc_price_slope_last_1m"] = 0.0
        else:
            row["auc_price_slope_last_1m"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows).set_index("trade_date")


def _compute_background_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=daily_df.index)

    close = daily_df["close"]
    prev_close = close.shift(1)

    features["ret_1d"] = prev_close / close.shift(2) - 1
    features["ret_3d"] = prev_close / close.shift(4) - 1
    features["ret_5d"] = prev_close / close.shift(6) - 1

    high_20d = daily_df["high"].rolling(20, min_periods=1).max()
    low_20d = daily_df["low"].rolling(20, min_periods=1).min()
    range_20d = high_20d - low_20d

    features["close_pos_20d"] = np.where(
        range_20d > 0, (prev_close - low_20d) / range_20d, 0.5
    )
    features["distance_to_20d_high"] = prev_close / high_20d - 1
    features["distance_to_20d_low"] = prev_close / low_20d - 1

    atr = compute_atr(daily_df["high"], daily_df["low"], close, 14)
    features["atr_pct"] = atr / prev_close

    daily_ret = close.pct_change()
    features["realized_vol_5d"] = daily_ret.rolling(5, min_periods=1).std()

    features["amount_avg_5d"] = daily_df["amount"].rolling(5, min_periods=1).mean()

    if "turnover" in daily_df.columns:
        features["turnover_avg_5d"] = daily_df["turnover"].rolling(5, min_periods=1).mean()
    else:
        features["turnover_avg_5d"] = np.nan

    prev_open = daily_df["open"].shift(1)
    prev_high = daily_df["high"].shift(1)
    prev_low = daily_df["low"].shift(1)
    hl_range = prev_high - prev_low

    features["prev_day_body_pct"] = np.where(
        hl_range > 0, (prev_close - prev_open).abs() / hl_range, 0.0
    )
    features["prev_day_upper_shadow_pct"] = np.where(
        hl_range > 0,
        (prev_high - pd.concat([prev_open, prev_close], axis=1).max(axis=1)) / hl_range,
        0.0,
    )
    features["prev_day_lower_shadow_pct"] = np.where(
        hl_range > 0,
        (pd.concat([prev_open, prev_close], axis=1).min(axis=1) - prev_low) / hl_range,
        0.0,
    )
    features["prev_day_close_in_range"] = np.where(
        hl_range > 0, (prev_close - prev_low) / hl_range, 0.5
    )

    return features


def _compute_env_features(
    index_auc_ret: Optional[pd.Series],
) -> pd.DataFrame:
    features = pd.DataFrame(index=index_auc_ret.index if index_auc_ret is not None else [0])
    if index_auc_ret is not None:
        features["index_auc_ret"] = index_auc_ret
    else:
        features["index_auc_ret"] = np.nan
    return features


def compute_features_for_stock(
    symbol: str, raw_data_dir: Optional[str] = None
) -> pd.DataFrame:
    """计算单只股票的全部竞价特征

    Args:
        symbol: 股票代码
        raw_data_dir: 原始数据根目录

    Returns:
        DataFrame with columns: stock_id, trade_date, + ALL_FEATURE_COLS
    """
    if raw_data_dir is None:
        raw_data_dir = RAW_DATA_DIR

    daily_path = os.path.join(raw_data_dir, symbol, "daily.parquet")
    if not os.path.exists(daily_path):
        logger.warning(f"日线数据不存在: {daily_path}")
        return pd.DataFrame(columns=["stock_id", "trade_date"] + ALL_FEATURE_COLS)

    daily_df = pd.read_parquet(daily_path)
    if daily_df.empty:
        return pd.DataFrame(columns=["stock_id", "trade_date"] + ALL_FEATURE_COLS)

    daily_df = daily_df.sort_values("datetime").reset_index(drop=True)
    daily_df["trade_date"] = daily_df["datetime"].dt.strftime("%Y%m%d").astype(int)

    ref_frames, match_frames = _load_auction_data(symbol, raw_data_dir)

    tushare_path = os.path.join(raw_data_dir, symbol, "tushare_auction.parquet")
    tushare_df = None
    if os.path.exists(tushare_path):
        tushare_df = pd.read_parquet(tushare_path)

    trade_dates = sorted(set(ref_frames.keys()) | set(match_frames.keys()))
    if not trade_dates:
        logger.warning(f"无竞价数据: {symbol}")
        return pd.DataFrame(columns=["stock_id", "trade_date"] + ALL_FEATURE_COLS)

    auction_prices = pd.Series(index=daily_df.index, dtype=float)
    auction_volumes = pd.Series(index=daily_df.index, dtype=float)
    auction_amounts = pd.Series(index=daily_df.index, dtype=float)

    for date_int in trade_dates:
        mask = daily_df["trade_date"] == date_int
        if not mask.any():
            continue
        idx = mask.idxmax()

        match_df = match_frames.get(date_int, pd.DataFrame())
        if not match_df.empty:
            row = match_df.iloc[0]
            auction_prices.iloc[idx] = row["price"]
            auction_volumes.iloc[idx] = row["vol"] * 100
            auction_amounts.iloc[idx] = row["price"] * row["vol"] * 100

    valid_mask = auction_prices.notna()
    if valid_mask.sum() == 0:
        return pd.DataFrame(columns=["stock_id", "trade_date"] + ALL_FEATURE_COLS)

    float_shares = None
    if tushare_df is not None and "float_share" in tushare_df.columns:
        ts_map = dict(zip(
            tushare_df["trade_date"].astype(int),
            tushare_df["float_share"] * 10000
        ))
        float_shares = daily_df["trade_date"].map(ts_map)

    state_feats = _compute_auction_state_features(auction_prices, daily_df)
    vol_feats = _compute_auction_volume_features(
        auction_volumes, auction_amounts, auction_prices, daily_df,
        tushare_df, float_shares
    )
    process_feats = _compute_auction_process_features(ref_frames, match_frames, trade_dates)
    bg_feats = _compute_background_features(daily_df)
    env_feats = _compute_env_features(None)
    if env_feats.shape[0] == 1 and len(daily_df) > 1:
        env_feats = pd.DataFrame(
            {"index_auc_ret": np.nan}, index=daily_df.index
        )

    result = pd.DataFrame(index=daily_df.index)
    result["stock_id"] = symbol
    result["trade_date"] = daily_df["trade_date"]

    for col in AUCTION_STATE_COLS:
        if col in state_feats.columns:
            result[col] = state_feats[col].values
        else:
            result[col] = np.nan

    for col in AUCTION_VOLUME_COLS:
        if col in vol_feats.columns:
            result[col] = vol_feats[col].values
        else:
            result[col] = np.nan

    process_aligned = process_feats.reindex(daily_df["trade_date"].values)
    for col in AUCTION_PROCESS_COLS:
        if col in process_aligned.columns:
            result[col] = process_aligned[col].values
        else:
            result[col] = np.nan

    for col in BACKGROUND_COLS:
        if col in bg_feats.columns:
            result[col] = bg_feats[col].values
        else:
            result[col] = np.nan

    for col in ENV_COLS:
        if col in env_feats.columns:
            result[col] = env_feats[col].values
        else:
            result[col] = np.nan

    result = result[valid_mask].reset_index(drop=True)
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbol = "000001"
    features_df = compute_features_for_stock(symbol)

    if features_df.empty:
        print(f"无特征数据，请先采集 {symbol} 的竞价数据")
    else:
        print(f"\n=== {symbol} 特征数据 (前5行) ===")
        print(features_df.head(5).to_string(index=False))

        print(f"\n=== 特征统计 ===")
        print(f"总行数: {len(features_df)}")
        for cat_name, cat_cols in FACTOR_CATEGORIES.items():
            existing = [c for c in cat_cols if c in features_df.columns]
            nan_counts = features_df[existing].isna().sum()
            print(f"\n{cat_name} ({len(existing)} cols):")
            for col in existing:
                nan_pct = nan_counts[col] / len(features_df) * 100
                print(f"  {col}: NaN={nan_pct:.1f}%")
