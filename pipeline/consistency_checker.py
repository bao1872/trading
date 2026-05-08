#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一致性抽检器：比对 daily_factor_update.py 和 backfill_factors.py 的输出

Purpose: 验证回补脚本和日更脚本的因子/事件计算口径是否一致
Inputs: stock_k_data (DB)
Outputs: 一致性报告（控制台）
How to Run:
    python pipeline/consistency_checker.py --stock-list 000001.SZ,000002.SZ --date 2024-03-15
    python pipeline/consistency_checker.py --sample-size 30 --date 2024-03-15
Examples:
    python pipeline/consistency_checker.py --stock-list 000001.SZ --date 2024-03-15 --verbose
Side Effects: 无（只读操作，不写入数据库）
"""
import argparse
import sys
import os
import warnings
from datetime import datetime, date
from typing import List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_engine
from pipeline.factor_utils import (
    get_recommended_lookback,
    get_stock_bars,
    check_coverage,
    compute_factors_for_stock,
    compute_events_for_stock,
)

warnings.filterwarnings("ignore")

# DSA 主链因子列表
DSA_CORE_FACTORS = [
    "dsa_dir", "prev_pivot_code", "trend_align_momo", "dsa_dir_age",
    "dsa_pivot_pos_01", "price_vs_dsa_vwap_pct", "ret_to_last_high_pct",
    "ret_to_last_low_pct", "current_pullback_from_stage_extreme_pct",
    "current_stage_bars", "current_stage_ret_pct", "current_stage_amp_pct",
    "prev_stage_bars", "prev_stage_amp_pct",
    "DSA_VWAP", "prev_pivot_type", "last_confirmed_high", "last_confirmed_low",
    "bars_since_last_high", "bars_since_last_low", "dsa_atr", "dsa_ratio",
    "liquidity_range_pos_01"
]


def compute_daily_style(conn, ts_code: str, target_date: date, freq: str = "1d") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按 daily_factor_update.py 风格计算因子和事件（只取最后一行）"""
    freq_db_map = {"1d": "d", "1w": "w"}
    freq_db = freq_db_map.get(freq, "d")

    # 统一使用公共 helper 获取 K 线
    df = get_stock_bars(conn, ts_code, target_date, freq_db)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 统一 coverage 检查
    lookback_bars = get_recommended_lookback(conn, freq)
    is_ok, coverage_ratio, coverage_status = check_coverage(df, lookback_bars)
    if not is_ok:
        return pd.DataFrame(), pd.DataFrame()

    factors_df = compute_factors_for_stock(df)
    if factors_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 只取最后一行
    last_factor_row = factors_df.iloc[-1:]

    events_df = compute_events_for_stock(factors_df)
    if events_df.empty:
        return last_factor_row, pd.DataFrame()

    last_event_row = events_df.iloc[-1:]
    return last_factor_row, last_event_row


def compute_backfill_style(conn, ts_code: str, target_date: date, freq: str = "1d") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按 backfill_factors.py 风格计算因子和事件（取目标日期所在行）"""
    freq_db_map = {"1d": "d", "1w": "w"}
    freq_db = freq_db_map.get(freq, "d")

    # 统一使用公共 helper 获取 K 线（以 target_date 为 end_date）
    bars = get_stock_bars(conn, ts_code, target_date, freq_db)
    if bars.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 统一 coverage 检查
    lookback_bars = get_recommended_lookback(conn, freq)
    is_ok, coverage_ratio, coverage_status = check_coverage(bars, lookback_bars)
    if not is_ok:
        return pd.DataFrame(), pd.DataFrame()

    factors_df = compute_factors_for_stock(bars)
    if factors_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    events_df = compute_events_for_stock(factors_df)

    # 过滤到目标日期
    target_ts = pd.Timestamp(target_date)
    if target_ts in factors_df.index:
        factor_target = factors_df.loc[[target_ts]]
    else:
        return pd.DataFrame(), pd.DataFrame()

    if events_df is not None and not events_df.empty and target_ts in events_df.index:
        event_target = events_df.loc[[target_ts]]
    else:
        event_target = pd.DataFrame()

    return factor_target, event_target


def compare_outputs(daily_factor: pd.DataFrame, backfill_factor: pd.DataFrame,
                    daily_event: pd.DataFrame, backfill_event: pd.DataFrame,
                    ts_code: str, target_date: date, verbose: bool = False) -> dict:
    """比对两个脚本的输出，返回差异统计"""
    result = {
        "ts_code": ts_code,
        "date": target_date,
        "factor_match": True,
        "event_match": True,
        "dsa_match": True,
        "factor_diffs": [],
        "event_diffs": [],
        "dsa_diffs": [],
        "missing_factors": [],
        "missing_events": [],
    }

    # 比对因子
    base_cols = {"open", "high", "low", "close", "volume"}
    daily_factor_cols = set(daily_factor.columns) - base_cols
    backfill_factor_cols = set(backfill_factor.columns) - base_cols

    # 检查因子名集合是否一致
    if daily_factor_cols != backfill_factor_cols:
        result["factor_match"] = False
        result["missing_factors"] = list(daily_factor_cols.symmetric_difference(backfill_factor_cols))

    # 比对因子值
    common_factor_cols = daily_factor_cols & backfill_factor_cols
    for col in common_factor_cols:
        daily_val = daily_factor[col].iloc[0] if not daily_factor.empty else np.nan
        backfill_val = backfill_factor[col].iloc[0] if not backfill_factor.empty else np.nan

        if pd.isna(daily_val) and pd.isna(backfill_val):
            continue
        if pd.isna(daily_val) or pd.isna(backfill_val):
            result["factor_match"] = False
            if col in DSA_CORE_FACTORS:
                result["dsa_match"] = False
                result["dsa_diffs"].append({
                    "factor": col,
                    "daily": daily_val,
                    "backfill": backfill_val,
                    "diff": np.nan,
                })
            else:
                result["factor_diffs"].append({
                    "factor": col,
                    "daily": daily_val,
                    "backfill": backfill_val,
                    "diff": np.nan,
                })
            continue

        # 处理字符串类型（如 prev_pivot_type）
        if isinstance(daily_val, str) or isinstance(backfill_val, str):
            if str(daily_val) != str(backfill_val):
                result["factor_match"] = False
                if col in DSA_CORE_FACTORS:
                    result["dsa_match"] = False
                    result["dsa_diffs"].append({
                        "factor": col,
                        "daily": daily_val,
                        "backfill": backfill_val,
                        "diff": None,
                    })
                else:
                    result["factor_diffs"].append({
                        "factor": col,
                        "daily": daily_val,
                        "backfill": backfill_val,
                        "diff": None,
                    })
            continue

        diff = abs(float(daily_val) - float(backfill_val))
        if diff > 1e-6:
            result["factor_match"] = False
            if col in DSA_CORE_FACTORS:
                result["dsa_match"] = False
                result["dsa_diffs"].append({
                    "factor": col,
                    "daily": float(daily_val),
                    "backfill": float(backfill_val),
                    "diff": diff,
                })
            else:
                result["factor_diffs"].append({
                    "factor": col,
                    "daily": float(daily_val),
                    "backfill": float(backfill_val),
                    "diff": diff,
                })

    # 比对事件
    daily_event_cols = set([c for c in daily_event.columns if c.startswith("evt_")])
    backfill_event_cols = set([c for c in backfill_event.columns if c.startswith("evt_")])

    if daily_event_cols != backfill_event_cols:
        result["event_match"] = False
        result["missing_events"] = list(daily_event_cols.symmetric_difference(backfill_event_cols))

    common_event_cols = daily_event_cols & backfill_event_cols
    for col in common_event_cols:
        daily_val = daily_event[col].iloc[0] if not daily_event.empty else np.nan
        backfill_val = backfill_event[col].iloc[0] if not backfill_event.empty else np.nan

        daily_triggered = bool(daily_val) if not pd.isna(daily_val) else False
        backfill_triggered = bool(backfill_val) if not pd.isna(backfill_val) else False

        if daily_triggered != backfill_triggered:
            result["event_match"] = False
            result["event_diffs"].append({
                "event": col,
                "daily_triggered": daily_triggered,
                "backfill_triggered": backfill_triggered,
            })

    if verbose and (not result["factor_match"] or not result["event_match"]):
        print(f"\n[{ts_code} @ {target_date}] 差异详情:")
        if result["missing_factors"]:
            print(f"  缺失因子: {result['missing_factors']}")
        for d in result["factor_diffs"]:
            if d["diff"] is None:
                print(f"  因子 {d['factor']}: daily={d['daily']}, backfill={d['backfill']} (字符串差异)")
            elif np.isnan(d["diff"]):
                print(f"  因子 {d['factor']}: daily={d['daily']}, backfill={d['backfill']} (NaN差异)")
            else:
                print(f"  因子 {d['factor']}: daily={d['daily']:.6f}, backfill={d['backfill']:.6f}, diff={d['diff']:.6f}")
        if result["missing_events"]:
            print(f"  缺失事件: {result['missing_events']}")
        for d in result["event_diffs"]:
            print(f"  事件 {d['event']}: daily={d['daily_triggered']}, backfill={d['backfill_triggered']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="一致性抽检器")
    parser.add_argument("--date", type=str, required=True, help="目标日期 (YYYY-MM-DD)")
    parser.add_argument("--stock-list", type=str, help="指定股票列表，逗号分隔")
    parser.add_argument("--sample-size", type=int, default=30, help="随机抽样数量")
    parser.add_argument("--freq", type=str, default="1d", choices=["1d", "1w"], help="频率")
    parser.add_argument("--verbose", action="store_true", help="显示详细差异")
    args = parser.parse_args()

    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    freq_db_map = {"1d": "d", "1w": "w"}
    freq_db = freq_db_map.get(args.freq, "d")

    engine = get_engine()
    conn = engine.connect()

    try:
        # 获取股票列表
        if args.stock_list:
            stock_list = [s.strip() for s in args.stock_list.split(",")]
        else:
            sql = text("""
                SELECT DISTINCT ts_code FROM stock_k_data
                WHERE freq = :freq AND bar_time = :date
            """)
            result = conn.execute(sql, {"freq": freq_db, "date": target_date.strftime("%Y-%m-%d")})
            all_stocks = [row[0] for row in result.fetchall()]
            # 随机抽样
            import random
            random.seed(42)
            stock_list = random.sample(all_stocks, min(args.sample_size, len(all_stocks)))

        print(f"=" * 60)
        print(f"一致性抽检: {target_date}, freq={args.freq}")
        print(f"股票数量: {len(stock_list)}")
        print(f"=" * 60)

        results = []
        for ts_code in tqdm(stock_list, desc="抽检进度"):
            daily_factor, daily_event = compute_daily_style(conn, ts_code, target_date, args.freq)
            backfill_factor, backfill_event = compute_backfill_style(conn, ts_code, target_date, args.freq)

            if daily_factor.empty and backfill_factor.empty:
                continue

            result = compare_outputs(
                daily_factor, backfill_factor,
                daily_event, backfill_event,
                ts_code, target_date, args.verbose
            )
            results.append(result)

        # 统计报告
        total = len(results)
        factor_matches = sum(1 for r in results if r["factor_match"])
        event_matches = sum(1 for r in results if r["event_match"])
        dsa_matches = sum(1 for r in results if r["dsa_match"])
        full_matches = sum(1 for r in results if r["factor_match"] and r["event_match"])

        print(f"\n{'=' * 60}")
        print(f"抽检结果汇总")
        print(f"{'=' * 60}")
        print(f"总样本数: {total}")
        print(f"因子完全匹配: {factor_matches}/{total} ({factor_matches/total*100:.1f}%)")
        print(f"事件完全匹配: {event_matches}/{total} ({event_matches/total*100:.1f}%)")
        print(f"DSA 主链完全匹配: {dsa_matches}/{total} ({dsa_matches/total*100:.1f}%)")
        print(f"完全匹配: {full_matches}/{total} ({full_matches/total*100:.1f}%)")

        if full_matches < total:
            print(f"\n差异样本:")
            for r in results:
                if not r["factor_match"] or not r["event_match"]:
                    issues = []
                    if not r["dsa_match"]:
                        issues.append(f"{len(r['dsa_diffs'])}个DSA差异")
                    elif not r["factor_match"]:
                        issues.append(f"{len(r['factor_diffs'])}个因子差异")
                    if not r["event_match"]:
                        issues.append(f"{len(r['event_diffs'])}个事件差异")
                    print(f"  {r['ts_code']}: {', '.join(issues)}")

        # DSA 主链详细报告
        print(f"\n{'=' * 60}")
        print(f"DSA 主链核对详情")
        print(f"{'=' * 60}")
        dsa_total_checks = 0
        dsa_failed_checks = 0
        for r in results:
            for d in r["dsa_diffs"]:
                dsa_total_checks += 1
                dsa_failed_checks += 1
                print(f"  {r['ts_code']} @ {r['date']} | {d['factor']}: daily={d['daily']}, backfill={d['backfill']}")
        if dsa_failed_checks == 0:
            print(f"  DSA 主链全部一致 ✓")
        else:
            print(f"  DSA 主链差异: {dsa_failed_checks}/{dsa_total_checks}")

    finally:
        conn.close()
        engine.dispose()


if __name__ == "__main__":
    main()
