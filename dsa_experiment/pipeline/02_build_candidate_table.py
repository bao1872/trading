#!/usr/bin/env python3
"""
构建候选池底表：从选股结果拼接日线价格路径，生成交易收益/风险/执行约束标签

Purpose: 读取选股结果 + 拼接日线价格 + 生成交易标签 → 保存候选池底表
Inputs: stock_dsa_vreversal_results, stock_k_data(freq='d'), stock_pools
Outputs: candidate_table.parquet
How to Run:
    python dsa_experiment/pipeline/02_build_candidate_table.py
    python dsa_experiment/pipeline/02_build_candidate_table.py --sample-limit 100
    python dsa_experiment/pipeline/02_build_candidate_table.py --start-date 2024-01-01
Side Effects: 只读数据库，输出文件到 dsa_experiment/output/

管线位置: Step 2/7 — 候选池底表构建（拼接日线价格路径+交易标签）
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

FACTOR_COLS = [
    "dsa_dir", "prev_pivot_code", "last_confirmed_high", "last_confirmed_low",
    "dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct", "current_stage_bars", "prev_stage_bars",
    "bars_since_last_high", "bars_since_last_low", "prev_stage_amp_pct",
    "current_stage_ret_pct", "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct", "bbmacd", "bbmacd_minus_avg",
    "bbmacd_state", "bbmacd_band_pos_01", "bbmacd_bandwidth_zscore",
    "bbmacd_cross_upper", "bbmacd_cross_lower", "trend_align_momo",
    "vol_zscore_5", "vol_zscore_10", "vol_zscore_20", "vol_ratio_10",
    "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio",
    "price_vol_coord", "momo_vol_coord", "low_pos_break_coord", "coord_consistency",
    "coord_stage_current", "coord_stage_prev", "coord_stage_ratio",
]


def build_trade_calendar() -> pd.DatetimeIndex:
    sql = text("SELECT DISTINCT bar_time FROM stock_k_data WHERE freq='d' ORDER BY bar_time")
    with engine.connect() as conn:
        dates = pd.read_sql(sql, conn)["bar_time"]
    return pd.DatetimeIndex(dates.sort_values())


def find_next_trade_day(cal: pd.DatetimeIndex, target_date) -> pd.Timestamp:
    target = pd.Timestamp(target_date)
    future = cal[cal > target]
    return future[0] if len(future) > 0 else pd.NaT


def load_selection_records(start_date=None, sample_limit=0) -> pd.DataFrame:
    sql = text(f"""
        SELECT selection_date, ts_code, stock_name, trigger_bar_time, trigger_close,
               {', '.join(FACTOR_COLS)}
        FROM stock_dsa_vreversal_results
        WHERE 1=1
        {"AND selection_date >= :start_date" if start_date else ""}
        ORDER BY selection_date, ts_code
        {"LIMIT :limit" if sample_limit > 0 else ""}
    """)
    params = {}
    if start_date:
        params["start_date"] = start_date
    if sample_limit > 0:
        params["limit"] = sample_limit
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params=params)
    df["ts_code_raw"] = df["ts_code"]
    df["ts_code"] = df["ts_code"].apply(_append_suffix)
    return df


def _append_suffix(code: str) -> str:
    if "." in code:
        return code
    num = code[:6]
    if num.startswith(("6", "5")):
        return f"{num}.SH"
    return f"{num}.SZ"


def load_daily_prices(ts_codes: list, min_date: str, max_date: str) -> pd.DataFrame:
    if not ts_codes:
        return pd.DataFrame()
    chunk_size = 500
    frames = []
    for i in range(0, len(ts_codes), chunk_size):
        chunk = ts_codes[i : i + chunk_size]
        codes_str = ",".join(f"'{c}'" for c in chunk)
        sql = text(f"""
            SELECT ts_code, bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE freq='d'
              AND ts_code IN ({codes_str})
              AND bar_time >= :min_date
              AND bar_time <= :max_date
            ORDER BY ts_code, bar_time
        """)
        with engine.connect() as conn:
            chunk_df = pd.read_sql(sql, conn, params={"min_date": min_date, "max_date": max_date})
        frames.append(chunk_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_stock_pools() -> pd.DataFrame:
    sql = text("""
        SELECT ts_code, industry_l2, industry_l3, market_cap, total_market_cap
        FROM stock_pools
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def compute_trade_labels(df_sel: pd.DataFrame, df_daily: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    if df_daily.empty:
        return df_sel

    df_daily = df_daily.copy()
    df_daily["bar_time"] = pd.to_datetime(df_daily["bar_time"])
    df_daily = df_daily.sort_values(["ts_code", "bar_time"]).reset_index(drop=True)
    df_daily["prev_close"] = df_daily.groupby("ts_code")["close"].shift(1)

    cal_list = list(cal)
    cal_map = {d: i for i, d in enumerate(cal_list)}

    def _next_trade_day(d):
        ts = pd.Timestamp(d)
        for c in cal_list:
            if c > ts:
                return c
        return pd.NaT

    df_sel = df_sel.copy()
    df_sel["t1"] = df_sel["selection_date"].apply(_next_trade_day)
    valid_mask = df_sel["t1"].notna()
    df_sel = df_sel[valid_mask].copy()

    t1_idx = df_sel["t1"].map(cal_map)
    df_sel["t1_idx"] = t1_idx

    daily_idx = df_daily.set_index(["ts_code", "bar_time"])

    def _batch_get(field, ts_codes, bar_times):
        lookup = pd.DataFrame({"ts_code": ts_codes, "bar_time": [pd.Timestamp(t) for t in bar_times]})
        lookup = lookup.set_index(["ts_code", "bar_time"])
        merged = lookup.join(daily_idx[[field]], how="left")
        return merged[field].values

    t1_opens = _batch_get("open", df_sel["ts_code"].values, df_sel["t1"].values)
    t1_prev_closes = _batch_get("prev_close", df_sel["ts_code"].values, df_sel["t1"].values)

    df_sel["buy_open"] = t1_opens
    df_sel["prev_close"] = t1_prev_closes

    t1_highs = _batch_get("high", df_sel["ts_code"].values, df_sel["t1"].values)
    t1_lows = _batch_get("low", df_sel["ts_code"].values, df_sel["t1"].values)
    t1_closes = _batch_get("close", df_sel["ts_code"].values, df_sel["t1"].values)

    is_limit_up = (
        (np.abs(t1_opens - t1_highs) < 0.001)
        & (np.abs(t1_highs - t1_lows) < 0.001)
        & (np.abs(t1_lows - t1_closes) < 0.001)
        & (t1_closes >= t1_prev_closes * 1.09)
        & (~np.isnan(t1_prev_closes))
    )
    df_sel["can_buy_next_open"] = (~is_limit_up) & (~np.isnan(t1_opens)) & (t1_opens > 0)
    df_sel["limit_up_next"] = is_limit_up.astype(float)

    for hold_days in [3, 5, 10, 20]:
        sell_dates = []
        for idx_val in df_sel["t1_idx"].values:
            if np.isnan(idx_val):
                sell_dates.append(pd.NaT)
            else:
                target_idx = int(idx_val) + hold_days
                sell_dates.append(cal_list[target_idx] if target_idx < len(cal_list) else pd.NaT)
        sell_opens = _batch_get("open", df_sel["ts_code"].values, sell_dates)
        ret_key = f"ret_{hold_days}_open_to_open"
        df_sel[ret_key] = np.where(
            (~np.isnan(sell_opens)) & (t1_opens > 0) & (~np.isnan(t1_opens)),
            (sell_opens - t1_opens) / t1_opens,
            np.nan,
        )

    for hold_days in [3, 5, 10, 20]:
        mae_vals = np.full(len(df_sel), np.nan)
        mfe_vals = np.full(len(df_sel), np.nan)
        for offset in range(1, hold_days + 1):
            day_dates = []
            for idx_val in df_sel["t1_idx"].values:
                if np.isnan(idx_val):
                    day_dates.append(pd.NaT)
                else:
                    target_idx = int(idx_val) + offset
                    day_dates.append(cal_list[target_idx] if target_idx < len(cal_list) else pd.NaT)
            day_lows = _batch_get("low", df_sel["ts_code"].values, day_dates)
            day_highs = _batch_get("high", df_sel["ts_code"].values, day_dates)
            mae_cand = np.where((~np.isnan(day_lows)) & (t1_opens > 0), (day_lows - t1_opens) / t1_opens, np.nan)
            mfe_cand = np.where((~np.isnan(day_highs)) & (t1_opens > 0), (day_highs - t1_opens) / t1_opens, np.nan)
            valid_mae = ~np.isnan(mae_cand)
            valid_mfe = ~np.isnan(mfe_cand)
            mae_vals = np.where(valid_mae & (np.isnan(mae_vals) | (mae_cand < mae_vals)), mae_cand, mae_vals)
            mfe_vals = np.where(valid_mfe & (np.isnan(mfe_vals) | (mfe_cand > mfe_vals)), mfe_cand, mfe_vals)
        df_sel[f"mae_{hold_days}"] = mae_vals
        df_sel[f"mfe_{hold_days}"] = mfe_vals

    df_sel["stop_hit_5"] = np.where(
        ~np.isnan(df_sel["mae_5"]), (df_sel["mae_5"] <= -0.05).astype(float), np.nan,
    )

    drop_cols = ["t1", "t1_idx", "buy_open", "prev_close"]
    df_sel = df_sel.drop(columns=[c for c in drop_cols if c in df_sel.columns], errors="ignore")
    return df_sel


def main():
    parser = argparse.ArgumentParser(description="构建候选池底表")
    parser.add_argument("--start-date", type=str, default=None, help="起始日期")
    parser.add_argument("--sample-limit", type=int, default=0, help="限制样本量（快速测试）")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("构建候选池底表")
    print("=" * 80)

    print("\n[1/4] 加载交易日历...")
    cal = build_trade_calendar()
    print(f"  交易日历: {cal[0].date()} ~ {cal[-1].date()}, 共 {len(cal)} 个交易日")

    print("\n[2/4] 加载选股结果...")
    df_sel = load_selection_records(start_date=args.start_date, sample_limit=args.sample_limit)
    print(f"  选股记录: {len(df_sel)} 条")
    if df_sel.empty:
        print("  无数据，退出")
        return
    print(f"  日期范围: {df_sel['selection_date'].min()} ~ {df_sel['selection_date'].max()}")
    print(f"  股票数: {df_sel['ts_code'].nunique()}")

    print("\n[3/4] 拼接日线价格并计算交易标签...")
    min_date = str(df_sel["selection_date"].min() - pd.Timedelta(days=10))
    max_date = str(df_sel["selection_date"].max() + pd.Timedelta(days=30))
    ts_codes = df_sel["ts_code"].unique().tolist()

    print(f"  加载日线数据: {len(ts_codes)} 只股票, {min_date} ~ {max_date}")
    df_daily = load_daily_prices(ts_codes, min_date, max_date)
    print(f"  日线记录: {len(df_daily)} 条")

    df_result = compute_trade_labels(df_sel, df_daily, cal)

    print("\n[4/4] 拼接行业/市值信息...")
    df_pools = load_stock_pools()
    if not df_pools.empty:
        df_result = df_result.merge(
            df_pools[["ts_code", "industry_l2", "industry_l3", "market_cap", "total_market_cap"]],
            on="ts_code", how="left",
        )
        print(f"  行业覆盖率: {df_result['industry_l2'].notna().mean():.1%}")

    can_buy_count = df_result["can_buy_next_open"].sum() if "can_buy_next_open" in df_result.columns else 0
    total_count = len(df_result)
    print(f"\n  可交易记录: {can_buy_count}/{total_count} ({can_buy_count/total_count:.1%})")

    for col in ["ret_3_open_to_open", "ret_5_open_to_open", "ret_10_open_to_open", "mae_3", "mae_5", "mfe_3", "mfe_5"]:
        if col in df_result.columns:
            valid = df_result[col].notna().sum()
            mean_val = df_result[col].mean()
            print(f"  {col}: valid={valid}, mean={mean_val:.4f}")

    if "stop_hit_5" in df_result.columns:
        valid = df_result[df_result["stop_hit_5"].notna()]
        if len(valid) > 0:
            print(f"  stop_hit_5: hit_rate={valid['stop_hit_5'].mean():.2%}")

    output_path = os.path.join(args.output_dir, "candidate_table.parquet")
    df_result.to_parquet(output_path, index=False)
    print(f"\n  保存: {output_path}")
    print(f"  总记录: {len(df_result)}, 列数: {len(df_result.columns)}")

    print("\n" + "=" * 80)
    print("候选池底表构建完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
