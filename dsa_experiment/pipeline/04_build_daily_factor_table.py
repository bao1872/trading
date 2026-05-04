#!/usr/bin/env python3
"""
日线因子表构建：在周线候选池内提取日线因子+买卖点标签

Purpose: 对周线触发点后的每个交易日提取日线因子快照，计算买卖点标签
Inputs: candidate_with_scores.parquet, stock_k_data (DB)
Outputs: output/daily_factor_table.parquet
How to Run:
    python dsa_experiment/pipeline/04_build_daily_factor_table.py
    python dsa_experiment/pipeline/04_build_daily_factor_table.py --sample 100
Examples:
    python dsa_experiment/pipeline/04_build_daily_factor_table.py
    python dsa_experiment/pipeline/04_build_daily_factor_table.py --sample 100
Side Effects: 只读操作，输出 parquet 文件

管线位置: Step 4/7 — 日线因子表构建（T+1~T+5因子快照+买卖点标签）

因子来源（与 selection_dsa.py 一致）:
  compute_dsa / compute_bbmacd / compute_24_factors → from features.dsa_bbmacd_24factors_viewer
  volume_zscore → from features.volume_zscore_plotly
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

from features.dsa_bbmacd_24factors_viewer import (
    compute_dsa,
    compute_bbmacd,
    compute_24_factors,
    DSAConfig,
)
from features.volume_zscore_plotly import volume_zscore

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

DSA_CFG = DSAConfig(prd=50, base_apt=20.0, use_adapt=False, vol_bias=10.0, atr_len=50)

DAILY_LOOKBACK = 150
DAILY_LOOKAHEAD = 15
FUTURE_WINDOWS = [5, 10]


def append_suffix(code: str) -> str:
    if code.startswith(("6", "5")):
        return f"{code}.SH"
    return f"{code}.SZ"


def load_daily_prices(ts_codes: list, min_date: str, max_date: str) -> pd.DataFrame:
    if not ts_codes:
        return pd.DataFrame()
    db_codes = [append_suffix(c) for c in ts_codes]
    all_dfs = []
    batch_size = 500
    for i in range(0, len(db_codes), batch_size):
        batch = db_codes[i:i + batch_size]
        placeholders = ", ".join([f"'{c}'" for c in batch])
        sql = text(f"""
            SELECT ts_code, bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE ts_code IN ({placeholders}) AND freq = 'd'
              AND bar_time >= :min_date AND bar_time <= :max_date
            ORDER BY ts_code, bar_time
        """)
        with engine.connect() as conn:
            batch_df = pd.read_sql(sql, conn, params={"min_date": min_date, "max_date": max_date})
        all_dfs.append(batch_df)
    if not all_dfs:
        return pd.DataFrame()
    result = pd.concat(all_dfs, ignore_index=True)
    result["bar_time"] = pd.to_datetime(result["bar_time"])
    result["raw_code"] = result["ts_code"].str.replace(r"\.(SH|SZ)", "", regex=True)
    return result


def compute_daily_factors_for_stock(stock_daily: pd.DataFrame) -> pd.DataFrame:
    if len(stock_daily) < 60:
        return pd.DataFrame()

    df = stock_daily.copy()
    df = df.sort_values("bar_time").reset_index(drop=True)

    try:
        dsa_df, _, _ = compute_dsa(df, DSA_CFG)
        bbmacd_df = compute_bbmacd(df)
        merged = pd.concat([df, dsa_df, bbmacd_df], axis=1)
        factors_df = compute_24_factors(merged)
        for col in factors_df.columns:
            merged[col] = factors_df[col]
    except Exception:
        return pd.DataFrame()

    if "volume" in merged.columns and merged["volume"].notna().sum() > 20:
        vol = merged["volume"].fillna(0)
        for win in [5, 10, 20]:
            z, mu, sd = volume_zscore(vol, win)
            merged[f"vol_zscore_{win}"] = z
        merged["vol_ratio_10"] = np.where(
            vol.rolling(10, min_periods=10).mean() > 0,
            vol / vol.rolling(10, min_periods=10).mean().replace(0, np.nan),
            np.nan,
        )

        vol_z10 = merged.get("vol_zscore_10", pd.Series(dtype=float))
        if vol_z10.notna().sum() > 0:
            spike_mask = vol_z10 > 2
            days_since = np.full(len(merged), np.nan)
            last_spike = -1
            for i in range(len(merged)):
                if spike_mask.iloc[i]:
                    last_spike = i
                if last_spike >= 0:
                    days_since[i] = i - last_spike
            merged["days_since_vol_spike"] = days_since

    if "dsa_dir" in merged.columns and "volume" in merged.columns:
        dsa_dir = merged["dsa_dir"].ffill()
        if isinstance(dsa_dir, pd.DataFrame):
            dsa_dir = dsa_dir.iloc[:, 0]
        vol = merged["volume"].fillna(0)
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        stage_changes = dsa_dir.diff().fillna(0) != 0
        stage_ids = stage_changes.cumsum()

        vol_cv_df = pd.DataFrame({"vol": vol.values, "stage": stage_ids.values})
        vol_cv = vol_cv_df.groupby("stage")["vol"].agg(["mean", "std"])
        vol_cv["cv"] = np.where(
            vol_cv["mean"] > 0,
            vol_cv["std"] / vol_cv["mean"],
            np.nan,
        )
        cv_map = vol_cv["cv"].to_dict()
        merged["vol_stage_cv"] = stage_ids.map(cv_map)

        prev_cv = stage_ids.map(
            lambda x: cv_map.get(x - 1, np.nan) if x > 0 else np.nan
        )
        merged["vol_prev_stage_cv"] = prev_cv
        merged["vol_cv_ratio"] = np.where(
            merged["vol_prev_stage_cv"].notna() & (merged["vol_prev_stage_cv"] > 0),
            merged["vol_stage_cv"] / merged["vol_prev_stage_cv"],
            np.nan,
        )

    if "dsa_dir" in merged.columns:
        dsa_dir = merged["dsa_dir"].ffill()
        if isinstance(dsa_dir, pd.DataFrame):
            dsa_dir = dsa_dir.iloc[:, 0]
        changes = dsa_dir.diff().fillna(0) != 0
        changes_vals = changes.values
        age = np.zeros(len(merged))
        for i in range(1, len(merged)):
            age[i] = 0 if changes_vals[i] else age[i - 1] + 1
        merged["dsa_dir_age"] = age

    if "bbmacd" in merged.columns:
        merged["bbmacd_sign"] = np.sign(merged["bbmacd"])
        merged["bbmacd_slope_3"] = merged["bbmacd"].diff(3) / 3

    if all(c in merged.columns for c in ["price_vs_dsa_vwap_pct", "vol_zscore_10"]):
        p_z = (merged["price_vs_dsa_vwap_pct"] - merged["price_vs_dsa_vwap_pct"].rolling(20, min_periods=5).mean()) / merged["price_vs_dsa_vwap_pct"].rolling(20, min_periods=5).std().replace(0, np.nan)
        v_z = merged["vol_zscore_10"]
        m_z = (merged["bbmacd_minus_avg"] - merged["bbmacd_minus_avg"].rolling(20, min_periods=5).mean()) / merged["bbmacd_minus_avg"].rolling(20, min_periods=5).std().replace(0, np.nan)
        merged["price_vol_coord"] = p_z * v_z
        merged["momo_vol_coord"] = m_z * v_z

    if all(c in merged.columns for c in ["dsa_pivot_pos_01", "bbmacd_minus_avg", "vol_zscore_10"]):
        merged["low_pos_break_coord"] = (1 - merged["dsa_pivot_pos_01"]) * merged.get("momo_vol_coord", np.nan)

    if all(c in merged.columns for c in ["price_vs_dsa_vwap_pct", "bbmacd_minus_avg", "vol_zscore_10"]):
        p_z2 = (merged["price_vs_dsa_vwap_pct"] - merged["price_vs_dsa_vwap_pct"].rolling(20, min_periods=5).mean()) / merged["price_vs_dsa_vwap_pct"].rolling(20, min_periods=5).std().replace(0, np.nan)
        m_z2 = (merged["bbmacd_minus_avg"] - merged["bbmacd_minus_avg"].rolling(20, min_periods=5).mean()) / merged["bbmacd_minus_avg"].rolling(20, min_periods=5).std().replace(0, np.nan)
        v_z2 = merged["vol_zscore_10"]
        three_z = pd.DataFrame({"p": p_z2, "m": m_z2, "v": v_z2})
        merged["coord_consistency"] = three_z.mean(axis=1) - three_z.std(axis=1)

    if all(c in merged.columns for c in ["dsa_dir", "price_vs_dsa_vwap_pct", "vol_zscore_10"]):
        dsa_dir = merged["dsa_dir"].ffill()
        if isinstance(dsa_dir, pd.DataFrame):
            dsa_dir = dsa_dir.iloc[:, 0]
        stage_changes = dsa_dir.diff().fillna(0) != 0
        stage_ids = stage_changes.cumsum()
        pvc = merged.get("price_vol_coord", pd.Series(dtype=float))
        if pvc.notna().sum() > 0:
            coord_df = pd.DataFrame({"pvc": pvc.values, "stage": stage_ids.values})
            stage_coord = coord_df.groupby("stage")["pvc"].mean()
            coord_map = stage_coord.to_dict()
            merged["coord_stage_current"] = stage_ids.map(coord_map)
            prev_coord = stage_ids.map(
                lambda x: coord_map.get(x - 1, np.nan) if x > 0 else np.nan
            )
            merged["coord_stage_prev"] = prev_coord
            merged["coord_stage_ratio"] = np.where(
                merged["coord_stage_prev"].notna() & (merged["coord_stage_prev"].abs() > 1e-8),
                merged["coord_stage_current"] / merged["coord_stage_prev"],
                np.nan,
            )

    return merged


def compute_future_labels(df: pd.DataFrame, trigger_idx: int) -> dict:
    labels = {}
    close_price = df.iloc[trigger_idx]["close"]
    if close_price <= 0 or np.isnan(close_price):
        return labels

    for n in FUTURE_WINDOWS:
        end_idx = min(trigger_idx + n + 1, len(df))
        if end_idx <= trigger_idx + 1:
            continue
        future = df.iloc[trigger_idx + 1:end_idx]

        if future.empty:
            continue

        high_prices = future["high"].values
        low_prices = future["low"].values

        max_high_idx = np.nanargmax(high_prices)
        min_low_idx = np.nanargmin(low_prices)

        labels[f"future_high_ret_{n}"] = (high_prices[max_high_idx] - close_price) / close_price
        labels[f"future_high_bars_{n}"] = max_high_idx + 1
        labels[f"future_low_ret_{n}"] = (low_prices[min_low_idx] - close_price) / close_price
        labels[f"future_low_bars_{n}"] = min_low_idx + 1

    return labels


def compute_tradeable_labels(df: pd.DataFrame, trigger_idx: int) -> dict:
    """计算可交易口径标签：当日收盘买入，未来N日收盘卖出"""
    labels = {}
    close_price = df.iloc[trigger_idx]["close"]
    if close_price <= 0 or np.isnan(close_price):
        return labels

    for n in FUTURE_WINDOWS:
        sell_idx = trigger_idx + n
        if sell_idx >= len(df):
            continue
        sell_close = df.iloc[sell_idx]["close"]
        if np.isnan(sell_close) or sell_close <= 0:
            continue
        labels[f"ret_{n}_close_to_close"] = (sell_close - close_price) / close_price

        # 同时计算持有期内的最大回撤（MAE）和最大浮盈（MFE）
        hold_period = df.iloc[trigger_idx + 1:sell_idx + 1]
        if not hold_period.empty:
            lowest_low = hold_period["low"].min()
            highest_high = hold_period["high"].max()
            labels[f"mae_{n}"] = (lowest_low - close_price) / close_price
            labels[f"mfe_{n}"] = (highest_high - close_price) / close_price

    return labels


FACTOR_COLS_24 = [
    "dsa_dir", "prev_pivot_code", "last_confirmed_high", "last_confirmed_low",
    "dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct", "current_stage_bars", "prev_stage_bars",
    "bars_since_last_high", "bars_since_last_low",
    "prev_stage_amp_pct", "current_stage_ret_pct", "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct",
    "bbmacd", "bbmacd_minus_avg", "bbmacd_state", "bbmacd_band_pos_01",
    "bbmacd_bandwidth_zscore", "bbmacd_cross_upper", "bbmacd_cross_lower",
    "trend_align_momo",
]

EXTRA_FACTOR_COLS = [
    "vol_zscore_5", "vol_zscore_10", "vol_zscore_20",
    "vol_ratio_10", "days_since_vol_spike",
    "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio",
    "dsa_dir_age", "bbmacd_sign", "bbmacd_slope_3",
    "price_vol_coord", "momo_vol_coord", "low_pos_break_coord", "coord_consistency",
    "coord_stage_current", "coord_stage_prev", "coord_stage_ratio",
]

ALL_FACTOR_COLS = FACTOR_COLS_24 + EXTRA_FACTOR_COLS


def main():
    parser = argparse.ArgumentParser(description="日线因子表构建")
    parser.add_argument("--sample", type=int, default=0, help="抽样数量（0=全量）")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 80)
    print("日线因子表构建")
    print("=" * 80)

    input_path = os.path.join(args.output_dir, "candidate_with_scores.parquet")
    print(f"\n  加载周线触发点: {input_path}")
    candidates = pd.read_parquet(input_path)
    tradeable = candidates[candidates["can_buy_next_open"] == True].copy()
    tradeable = tradeable[tradeable["return_score"].notna()]
    print(f"  可交易且有预测分: {len(tradeable)}")

    if args.sample > 0:
        tradeable = tradeable.head(args.sample)
        print(f"  抽样: {len(tradeable)}")

    print("\n  [1/4] 确定日线数据范围...")
    sel_dates = pd.to_datetime(tradeable["selection_date"])
    min_date = (sel_dates.min() - pd.Timedelta(days=DAILY_LOOKBACK + 30)).strftime("%Y-%m-%d")
    max_date = (sel_dates.max() + pd.Timedelta(days=DAILY_LOOKAHEAD + 30)).strftime("%Y-%m-%d")
    print(f"  日线范围: {min_date} ~ {max_date}")

    ts_code_col = "ts_code_raw" if "ts_code_raw" in tradeable.columns else "ts_code"
    unique_stocks = tradeable[ts_code_col].unique().tolist()
    print(f"  股票数: {len(unique_stocks)}")

    print("\n  [2/4] 加载日线行情...")
    daily = load_daily_prices(unique_stocks, min_date, max_date)
    print(f"  日线记录: {len(daily)}")

    print("\n  [3/4] 逐股计算日线因子+标签...")
    all_records = []
    stock_groups = tradeable.groupby(ts_code_col)
    n_stocks = len(stock_groups)
    FLUSH_INTERVAL = 500

    out_path = os.path.join(args.output_dir, "daily_factor_table.parquet")
    total_saved = 0

    for stock_idx, (ts_code, stock_candidates) in enumerate(stock_groups):
        stock_daily = daily[daily["raw_code"] == ts_code].copy()
        if stock_daily.empty:
            continue

        factor_df = compute_daily_factors_for_stock(stock_daily)
        if factor_df.empty:
            continue

        bar_time_idx = factor_df.set_index("bar_time")

        for _, cand in stock_candidates.iterrows():
            sel_date = pd.Timestamp(cand["selection_date"])
            trigger_bar = bar_time_idx.index.get_indexer([sel_date], method="ffill")
            if len(trigger_bar) == 0 or trigger_bar[0] < 0:
                continue
            trigger_idx = trigger_bar[0]

            for day_offset in range(1, 6):
                target_idx = trigger_idx + day_offset
                if target_idx >= len(factor_df) or target_idx < 0:
                    continue

                row = factor_df.iloc[target_idx]
                record = {
                    "selection_date": cand["selection_date"],
                    "ts_code": ts_code,
                    "trigger_bar_time": cand.get("trigger_bar_time", ""),
                    "day_offset": day_offset,
                    "bar_time": row.get("bar_time", ""),
                    "open": row.get("open", np.nan),
                    "high": row.get("high", np.nan),
                    "low": row.get("low", np.nan),
                    "close": row.get("close", np.nan),
                    "volume": row.get("volume", np.nan),
                    "weekly_return_score": cand.get("return_score", np.nan),
                    "weekly_risk_score": cand.get("risk_score", np.nan),
                }

                for col in ALL_FACTOR_COLS:
                    record[col] = row.get(col, np.nan)

                labels = compute_future_labels(factor_df, target_idx)
                record.update(labels)

                tradeable_labels = compute_tradeable_labels(factor_df, target_idx)
                record.update(tradeable_labels)

                all_records.append(record)

        if (stock_idx + 1) % FLUSH_INTERVAL == 0 and all_records:
            chunk_df = pd.DataFrame(all_records)
            if total_saved == 0:
                chunk_df.to_parquet(out_path, index=False)
            else:
                existing = pd.read_parquet(out_path)
                pd.concat([existing, chunk_df], ignore_index=True).to_parquet(out_path, index=False)
                del existing
            total_saved += len(all_records)
            print(f"  已保存 {total_saved} 条 (处理 {stock_idx + 1}/{n_stocks} 股)")
            del chunk_df
            all_records = []

    if all_records:
        chunk_df = pd.DataFrame(all_records)
        if total_saved == 0:
            chunk_df.to_parquet(out_path, index=False)
        else:
            existing = pd.read_parquet(out_path)
            pd.concat([existing, chunk_df], ignore_index=True).to_parquet(out_path, index=False)
            del existing
        total_saved += len(all_records)
        del chunk_df
        all_records = []

    print(f"\n  日线记录总数: {total_saved}")

    print("\n  [4/4] 验证结果...")
    result_df = pd.read_parquet(out_path)
    print(f"  最终记录数: {len(result_df)}")
    print(f"  列数: {len(result_df.columns)}")

    for offset in range(1, 6):
        n = len(result_df[result_df["day_offset"] == offset])
        print(f"  T+{offset}: {n} 条")

    label_cols = [c for c in result_df.columns if "future_" in c or ("ret_" in c and "close_to_close" in c)]
    print(f"\n  标签统计:")
    for col in label_cols:
        if result_df[col].notna().sum() > 0:
            print(f"    {col}: mean={result_df[col].mean():.4f}, std={result_df[col].std():.4f}")
    del result_df

    print("\n" + "=" * 80)
    print("日线因子表构建完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
