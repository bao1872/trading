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

因子来源: 从 factor_value 表读取（单事实来源）
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

from dsa_experiment.pipeline.dsa_config import DSAConfig
from dsa_experiment.pipeline.factor_columns import ALL_DAILY_FACTOR_COLS

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
        sql = text("""
            SELECT ts_code, bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE ts_code = ANY(:codes) AND freq = 'd'
              AND bar_time >= :min_date AND bar_time <= :max_date
            ORDER BY ts_code, bar_time
        """)
        with engine.connect() as conn:
            batch_df = pd.read_sql(sql, conn, params={
                "codes": batch,
                "min_date": min_date,
                "max_date": max_date,
            })
        all_dfs.append(batch_df)
    if not all_dfs:
        return pd.DataFrame()
    result = pd.concat(all_dfs, ignore_index=True)
    result["bar_time"] = pd.to_datetime(result["bar_time"])
    result["raw_code"] = result["ts_code"].str.replace(r"\.(SH|SZ)", "", regex=True)
    return result


def load_factors_batch(ts_codes: list, min_date: str, max_date: str) -> dict:
    """从 factor_value 批量加载因子（返回 {code: wide_df} 字典）"""
    if not ts_codes:
        return {}
    db_codes = [append_suffix(c) for c in ts_codes]
    result = {}
    batch_size = 500
    for i in range(0, len(db_codes), batch_size):
        batch = db_codes[i:i + batch_size]
        sql = text("""
            SELECT ts_code, factor_name, factor_value, as_of_date
            FROM factor_value
            WHERE ts_code = ANY(:codes) AND freq = '1d'
              AND as_of_date >= :min_date AND as_of_date <= :max_date
            ORDER BY ts_code, as_of_date, factor_name
        """)
        with engine.connect() as conn:
            batch_df = pd.read_sql(sql, conn, params={
                "codes": batch,
                "min_date": min_date,
                "max_date": max_date,
            })
        if batch_df.empty:
            continue
        batch_df["raw_code"] = batch_df["ts_code"].str.replace(r"\.(SH|SZ)", "", regex=True)
        for raw_code, grp in batch_df.groupby("raw_code"):
            wide = grp.pivot(index="as_of_date", columns="factor_name", values="factor_value")
            wide = wide.reset_index()
            wide.columns.name = None
            result[raw_code] = wide
    return result


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

    print("\n  [3/4] 从 factor_value 读取日线因子...")
    factor_data = load_factors_batch(unique_stocks, min_date, max_date)
    n_loaded = sum(len(v) for v in factor_data.values())
    print(f"  因子宽表记录: {n_loaded} (覆盖 {len(factor_data)} 只股票)")

    all_records = []
    stock_groups = tradeable.groupby(ts_code_col)
    FLUSH_INTERVAL = 500

    out_path = os.path.join(args.output_dir, "daily_factor_table.parquet")
    total_saved = 0

    for stock_idx, (ts_code, stock_candidates) in enumerate(stock_groups):
        if ts_code not in factor_data:
            continue

        factors_wide = factor_data[ts_code]
        stock_daily = daily[daily["raw_code"] == ts_code].copy()
        if stock_daily.empty:
            continue

        stock_daily["as_of_date"] = pd.to_datetime(stock_daily["bar_time"]).dt.date
        factor_df_full = stock_daily.merge(factors_wide, on="as_of_date", how="left")
        factor_df = factor_df_full.sort_values("bar_time").reset_index(drop=True)

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
                    "bar_time": str(row.get("bar_time", "")),
                    "open": row.get("open", np.nan),
                    "high": row.get("high", np.nan),
                    "low": row.get("low", np.nan),
                    "close": row.get("close", np.nan),
                    "volume": row.get("volume", np.nan),
                    "weekly_return_score": cand.get("return_score", np.nan),
                    "weekly_risk_score": cand.get("risk_score", np.nan),
                }

                for col in ALL_DAILY_FACTOR_COLS:
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
