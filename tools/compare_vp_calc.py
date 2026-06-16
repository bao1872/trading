#!/usr/bin/env python3
"""
对比 selection_limit_up.py 与 luxalgo CLI --db-code 模式的 Volume Profile 计算差异

Purpose:
    逐层对比两种方式的日线数据、15m数据、compute_volume_profile 输出，
    定位差异根因。

Inputs:
    数据库 stock_k_data 表 + stock_adj_factor 表

Outputs:
    终端打印对比结果

How to Run:
    python tools/compare_vp_calc.py 000969.SZ 2026-06-12
    python tools/compare_vp_calc.py 000969.SZ 2026-06-12 --verbose

Examples:
    python tools/compare_vp_calc.py 000969.SZ 2026-06-12
    python tools/compare_vp_calc.py 000969.SZ 2026-06-12 --verbose

Side Effects:
    无（只读数据库，不写入）
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import date

# 方式1：selection_limit_up.py 的数据加载方式
from selection.selection_dsa import get_kline_data_db
from datasource.k_data_loader import load_k_data
from datasource.adj_factor import apply_adj_factor_intraday

# 方式2：luxalgo CLI --db-code 的数据加载方式（直接用 load_k_data + adj='qfq'）

# 共用计算
from features.luxalgo_volume_profile_pytdx_15m_aligned import (
    compute_volume_profile,
    VolumeProfileConfig,
)
from selection.selection_limit_up import extract_node_clusters, VP_LOOKBACK, VP_ROWS


def compare(ts_code: str, end_date: date, verbose: bool = False):
    print("=" * 80)
    print(f"对比 Volume Profile 计算: {ts_code}, 截止日期: {end_date}")
    print("=" * 80)

    # =========================================================================
    # 1. 日线数据对比
    # =========================================================================
    print("\n--- 1. 日线数据对比 ---")

    # 方式1: get_kline_data_db (selection_limit_up.py 使用)
    daily_method1 = get_kline_data_db(ts_code, bars=800, end_date=end_date)
    print(f"方式1 (get_kline_data_db): {len(daily_method1)} 条")

    # 方式2: load_k_data(adj='qfq') (luxalgo CLI --db-code 使用)
    daily_method2 = load_k_data(ts_code, freq='d', adj='qfq')
    # 截取到 end_date
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
    daily_method2 = daily_method2[daily_method2.index <= end_ts]
    print(f"方式2 (load_k_data qfq):   {len(daily_method2)} 条")

    # 对比最后N条数据
    compare_len = min(len(daily_method1), len(daily_method2), 10)
    if compare_len > 0:
        m1_tail = daily_method1.tail(compare_len)[['open', 'high', 'low', 'close', 'volume']]
        m2_tail = daily_method2.tail(compare_len)[['open', 'high', 'low', 'close', 'volume']]

        # 检查数值差异
        diff_mask = (m1_tail.values != m2_tail.values)
        if not diff_mask.any():
            print(f"  最后 {compare_len} 条日线数据: 完全一致")
        else:
            print(f"  最后 {compare_len} 条日线数据: 存在差异!")
            if verbose:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    d1 = m1_tail[col].values
                    d2 = m2_tail[col].values
                    diff = np.abs(d1 - d2)
                    max_diff = diff.max()
                    if max_diff > 0:
                        print(f"    {col}: 最大差异={max_diff:.6f}")

        # 对比最后一条的 close
        last_close_1 = float(daily_method1['close'].iloc[-1])
        last_close_2 = float(daily_method2['close'].iloc[-1])
        print(f"  最后收盘价: 方式1={last_close_1:.4f}, 方式2={last_close_2:.4f}, 差异={abs(last_close_1-last_close_2):.6f}")

    # =========================================================================
    # 2. 15m数据对比
    # =========================================================================
    print("\n--- 2. 15m数据对比 ---")

    # 方式1: load_k_data(freq='15m') + apply_adj_factor_intraday (selection_limit_up.py)
    ltf_method1 = load_k_data(ts_code, freq='15m')
    ltf_method1 = apply_adj_factor_intraday(ltf_method1, ts_code)
    if not isinstance(ltf_method1.index, pd.DatetimeIndex):
        ltf_method1.index = pd.to_datetime(ltf_method1.index)
    ltf_method1 = ltf_method1[ltf_method1.index <= end_ts]
    print(f"方式1 (load_k_data 15m + adj_intraday): {len(ltf_method1)} 条")

    # 方式2: load_k_data(freq='15m', adj=None) + apply_adj_factor_intraday (luxalgo CLI --db-code)
    ltf_method2 = load_k_data(ts_code, freq='15m', adj=None)
    ltf_method2 = apply_adj_factor_intraday(ltf_method2, ts_code)
    if not isinstance(ltf_method2.index, pd.DatetimeIndex):
        ltf_method2.index = pd.to_datetime(ltf_method2.index)
    ltf_method2 = ltf_method2[ltf_method2.index <= end_ts]
    print(f"方式2 (load_k_data 15m raw + adj_intraday): {len(ltf_method2)} 条")

    # 对比15m数据
    if len(ltf_method1) == len(ltf_method2):
        compare_cols = ['open', 'high', 'low', 'close', 'volume']
        m1_vals = ltf_method1[compare_cols].values
        m2_vals = ltf_method2[compare_cols].values
        diff_mask = (m1_vals != m2_vals)
        if not diff_mask.any():
            print(f"  15m数据: 完全一致 ({len(ltf_method1)} 条)")
        else:
            diff_count = diff_mask.sum()
            print(f"  15m数据: 存在差异! 差异单元格数={diff_count}")
            if verbose:
                for col in compare_cols:
                    d1 = ltf_method1[col].values
                    d2 = ltf_method2[col].values
                    diff = np.abs(d1 - d2)
                    max_diff = diff.max()
                    if max_diff > 0:
                        print(f"    {col}: 最大差异={max_diff:.6f}")
    else:
        print(f"  15m数据条数不同! 方式1={len(ltf_method1)}, 方式2={len(ltf_method2)}")

    # =========================================================================
    # 3. compute_volume_profile 输出对比
    # =========================================================================
    print("\n--- 3. compute_volume_profile 输出对比 ---")

    cfg = VolumeProfileConfig(
        peaks_show="peaks",
        profile_lookback_length=VP_LOOKBACK,
        profile_number_of_rows=VP_ROWS,
        peaks_detection_percent=0.05,  # 与 UI/monitoring 保持一致
    )

    # 方式1: selection_limit_up.py 的数据准备方式
    daily1_vp = daily_method1.copy()
    if 'datetime' not in daily1_vp.columns:
        daily1_vp['datetime'] = daily1_vp.index
    ltf1_vp = ltf_method1.copy()
    if 'datetime' not in ltf1_vp.columns:
        ltf1_vp['datetime'] = ltf1_vp.index

    vp_result1 = compute_volume_profile(daily1_vp, cfg, profile_df=ltf1_vp, main_period="day")

    # 方式2: luxalgo CLI --db-code 的数据准备方式
    daily2_vp = daily_method2.copy()
    if 'datetime' not in daily2_vp.columns:
        daily2_vp['datetime'] = daily2_vp.index
    ltf2_vp = ltf_method2.copy()
    if 'datetime' not in ltf2_vp.columns:
        ltf2_vp['datetime'] = ltf2_vp.index

    vp_result2 = compute_volume_profile(daily2_vp, cfg, profile_df=ltf2_vp, main_period="day")

    # 对比 profile_df
    pdf1 = vp_result1.profile_df
    pdf2 = vp_result2.profile_df

    print(f"  方式1 profile_df: {len(pdf1)} 行, peaks={len(vp_result1.all_peak_prices)}")
    print(f"  方式2 profile_df: {len(pdf2)} 行, peaks={len(vp_result2.all_peak_prices)}")

    # 对比 POC/VAH/VAL
    print(f"\n  POC: 方式1={vp_result1.poc_price:.4f}, 方式2={vp_result2.poc_price:.4f}, 差异={abs(vp_result1.poc_price-vp_result2.poc_price):.6f}")
    print(f"  VAH: 方式1={vp_result1.vah_price:.4f}, 方式2={vp_result2.vah_price:.4f}, 差异={abs(vp_result1.vah_price-vp_result2.vah_price):.6f}")
    print(f"  VAL: 方式1={vp_result1.val_price:.4f}, 方式2={vp_result2.val_price:.4f}, 差异={abs(vp_result1.val_price-vp_result2.val_price):.6f}")

    # 对比价格范围
    print(f"  最低价: 方式1={vp_result1.lowest_price:.4f}, 方式2={vp_result2.lowest_price:.4f}")
    print(f"  最高价: 方式1={vp_result1.highest_price:.4f}, 方式2={vp_result2.highest_price:.4f}")

    # 对比 peak 行（使用 VolumeProfileResult.peak_df，SSOT）
    peaks1 = vp_result1.peak_df
    peaks2 = vp_result2.peak_df

    print(f"\n  Peak行对比:")
    print(f"    方式1 peak价格: {vp_result1.all_peak_prices}")
    print(f"    方式2 peak价格: {vp_result2.all_peak_prices}")

    if verbose:
        if peaks1 is not None and not peaks1.empty:
            print(f"\n  方式1 peak详情:")
            for _, row in peaks1.iterrows():
                print(f"    price_mid={row['price_mid']:.4f}, total_vol={row['total_volume']:.2f}, bull={row['bullish_volume']:.2f}, bear={row['bearish_volume']:.2f}")
        if peaks2 is not None and not peaks2.empty:
            print(f"\n  方式2 peak详情:")
            for _, row in peaks2.iterrows():
                print(f"    price_mid={row['price_mid']:.4f}, total_vol={row['total_volume']:.2f}, bull={row['bullish_volume']:.2f}, bear={row['bearish_volume']:.2f}")

    # =========================================================================
    # 4. extract_node_clusters 结果对比
    # =========================================================================
    print("\n--- 4. extract_node_clusters 结果对比 ---")

    current_price = float(daily_method1['close'].iloc[-1])
    print(f"  当前价格: {current_price}")

    nodes1 = extract_node_clusters(vp_result1, current_price)
    nodes2 = extract_node_clusters(vp_result2, current_price)

    print(f"\n  方式1 节点信息:")
    for k, v in nodes1.items():
        print(f"    {k}: {v}")

    print(f"\n  方式2 节点信息:")
    for k, v in nodes2.items():
        print(f"    {k}: {v}")

    # 逐字段对比
    print(f"\n  差异汇总:")
    has_diff = False
    for k in nodes1:
        v1 = nodes1[k]
        v2 = nodes2[k]
        if v1 != v2:
            has_diff = True
            if v1 is not None and v2 is not None:
                diff = abs(float(v1) - float(v2))
                print(f"    {k}: 方式1={v1}, 方式2={v2}, 差异={diff:.6f}")
            else:
                print(f"    {k}: 方式1={v1}, 方式2={v2} (None差异)")
    if not has_diff:
        print("    无差异")

    # =========================================================================
    # 5. luxalgo CLI 不截取 end_date 时的结果（模拟 CLI 行为）
    # =========================================================================
    print("\n--- 5. luxalgo CLI 行为模拟（不截取 end_date，使用全部数据）---")

    daily_full = load_k_data(ts_code, freq='d', adj='qfq')
    ltf_full = load_k_data(ts_code, freq='15m', adj=None)
    ltf_full = apply_adj_factor_intraday(ltf_full, ts_code)
    if not isinstance(ltf_full.index, pd.DatetimeIndex):
        ltf_full.index = pd.to_datetime(ltf_full.index)

    daily_full_vp = daily_full.copy()
    if 'datetime' not in daily_full_vp.columns:
        daily_full_vp['datetime'] = daily_full_vp.index
    ltf_full_vp = ltf_full.copy()
    if 'datetime' not in ltf_full_vp.columns:
        ltf_full_vp['datetime'] = ltf_full_vp.index

    vp_result_full = compute_volume_profile(daily_full_vp, cfg, profile_df=ltf_full_vp, main_period="day")
    pdf_full = vp_result_full.profile_df

    current_price_full = float(daily_full['close'].iloc[-1])
    nodes_full = extract_node_clusters(vp_result_full, current_price_full)

    print(f"  日线数据量: {len(daily_full)} 条")
    print(f"  15m数据量: {len(ltf_full)} 条")
    print(f"  当前价格: {current_price_full}")
    print(f"  POC: {vp_result_full.poc_price:.4f}")
    print(f"  VAH: {vp_result_full.vah_price:.4f}")
    print(f"  VAL: {vp_result_full.val_price:.4f}")
    print(f"  Peak行数: {len(vp_result_full.all_peak_prices)}")
    print(f"  Peak价格: {vp_result_full.all_peak_prices}")

    print(f"\n  CLI模式节点信息:")
    for k, v in nodes_full.items():
        print(f"    {k}: {v}")

    # 与方式1对比
    print(f"\n  与 selection_limit_up.py 方式的差异:")
    has_diff = False
    for k in nodes1:
        v1 = nodes1[k]
        v_full = nodes_full[k]
        if v1 != v_full:
            has_diff = True
            if v1 is not None and v_full is not None:
                diff = abs(float(v1) - float(v_full))
                print(f"    {k}: 选股={v1}, CLI={v_full}, 差异={diff:.6f}")
            else:
                print(f"    {k}: 选股={v1}, CLI={v_full} (None差异)")
    if not has_diff:
        print("    无差异")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比两种 Volume Profile 计算方式")
    parser.add_argument("ts_code", help="股票代码，如 000969.SZ")
    parser.add_argument("end_date", help="截止日期，如 2026-06-12")
    parser.add_argument("--verbose", action="store_true", help="打印详细差异")
    args = parser.parse_args()

    from datetime import datetime
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    compare(args.ts_code, end_date, verbose=args.verbose)
