#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer 4: 状态与账户一致性检查 — 回测 vs 模拟盘

Purpose:
    逐日对比持仓、逐笔对比交易、逐日对比净值曲线。

Inputs:
    - stop_experiment/output/backtest/baseline_e0_x1_v1_nav.csv
    - stop_experiment/output/backtest/baseline_e0_x1_v1_trades.csv
    - stop_experiment/output/backtest/baseline_e0_x1_v1_equity_curve.csv
    - stop_experiment/output/live/live_equity_curve.csv
    - stop_experiment/output/live/live_trade_report.csv
    - stop_experiment/output/holdings/*.parquet

Outputs:
    - 控制台: 对比结果汇总

How to Run:
    python -m stop_experiment.tests_consistency.compare_equity_curves

Side Effects:
    无（只读）
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from stop_experiment.tests_consistency import (
    NAV_TOL, HOLDINGS_DIR,
    load_backtest_nav, load_backtest_trades,
    load_live_equity_curve, load_live_trade_report,
    fmt_pass, fmt_diff,
)
from stop_experiment.pipeline.stop_config import PRODUCTION_PARAMS

MAX_STOCKS = PRODUCTION_PARAMS.get("max_stocks", 10)


def _compare_nav_curves() -> dict:
    bt_nav = load_backtest_nav()
    live_nav = load_live_equity_curve()

    # 检查起始日期是否一致（用第一个有持仓的日期）
    bt_first = bt_nav["date"].min()
    live_first_with_pos = live_nav[live_nav["n_positions"] > 0]["date"].min()
    if pd.isna(live_first_with_pos):
        same_start = True
    else:
        same_start = abs((bt_first - live_first_with_pos).days) <= 3

    merged = bt_nav[["date", "nav"]].merge(
        live_nav[["date", "nav_live"]], on="date", how="inner"
    )

    if merged.empty:
        return {"name": "NAV曲线", "status": "FAIL", "issues": ["无共同日期"]}

    issues = []

    if same_start:
        # 起始日期一致：严格对比绝对NAV
        merged["nav_diff"] = (merged["nav_live"] - merged["nav"]).abs()
        max_diff = merged["nav_diff"].max()
        mean_diff = merged["nav_diff"].mean()
        if max_diff > NAV_TOL:
            issues.append(f"最大NAV偏差: {max_diff:.8f} > {NAV_TOL}")
    else:
        # 起始日期不同：只检查模拟盘自身完整性
        max_diff = 0
        mean_diff = 0
        if live_nav["nav_live"].iloc[0] != 1.0:
            issues.append(f"首日NAV≠1.0: {live_nav['nav_live'].iloc[0]:.6f}")
        if (live_nav["nav_live"] <= 0).any():
            issues.append("NAV出现负值")

    max_diff_val = max_diff if same_start else 0

    bt_daily_ret = bt_nav.set_index("date")["nav"].pct_change()
    live_daily_ret = live_nav.set_index("date")["nav_live"].pct_change()
    common_idx = bt_daily_ret.index.intersection(live_daily_ret.index)

    if same_start:
        ret_diff = (bt_daily_ret.loc[common_idx] - live_daily_ret.loc[common_idx]).abs()
        max_ret_diff = ret_diff.max()
        if max_ret_diff > NAV_TOL:
            issues.append(f"最大日收益偏差: {max_ret_diff:.8f}")
    else:
        max_ret_diff = 0

    bt_final = bt_nav["nav"].iloc[-1]
    live_final = live_nav["nav_live"].iloc[-1]

    status = "PASS" if not issues else "FAIL"
    return {
        "name": "NAV曲线",
        "status": status,
        "issues": issues,
        "common_dates": len(merged),
        "max_nav_diff": max_diff,
        "mean_nav_diff": mean_diff,
        "max_daily_ret_diff": max_ret_diff,
        "bt_final": bt_final,
        "live_final": live_final,
    }


def _compare_trades() -> dict:
    bt_trades = load_backtest_trades()
    live_trades = load_live_trade_report()

    issues = []

    bt_trades["buy_date"] = pd.to_datetime(bt_trades["buy_date"])
    live_trades["buy_dt"] = pd.to_datetime(live_trades["买入日期"])
    bt_first = bt_trades["buy_date"].min()
    live_first = live_trades["buy_dt"].min()
    same_start = abs((bt_first - live_first).days) <= 3

    if same_start and len(bt_trades) != len(live_trades):
        issues.append(f"交易笔数: bt={len(bt_trades)} vs live={len(live_trades)}")

    bt_trades["key"] = bt_trades["ts_code"].astype(str) + "_" + bt_trades["buy_date"].astype(str).str[:10]
    live_trades["key"] = live_trades["股票代码"].astype(str) + "_" + live_trades["买入日期"].astype(str).str[:10]

    bt_set = set(bt_trades["key"])
    live_set = set(live_trades["key"])
    only_bt = bt_set - live_set
    only_live = live_set - bt_set

    if same_start:
        if only_bt:
            issues.append(f"仅回测有: {sorted(only_bt)[:5]}")
        if only_live:
            issues.append(f"仅模拟盘有: {sorted(only_live)[:5]}")

    common_keys = bt_set & live_set
    bt_common = bt_trades[bt_trades["key"].isin(common_keys)].set_index("key")
    live_common = live_trades[live_trades["key"].isin(common_keys)].set_index("key")

    price_mismatch = 0
    for k in common_keys:
        if k not in bt_common.index or k not in live_common.index:
            continue
        bt_row = bt_common.loc[k]
        live_row = live_common.loc[k]
        bt_buy = float(bt_row["buy_price"])
        live_buy = float(live_row["买入价"])
        bt_sell = float(bt_row["sell_price"])
        live_sell = float(live_row["卖出价"])
        if abs(bt_buy - live_buy) > 0.01 or abs(bt_sell - live_sell) > 0.01:
            price_mismatch += 1

    if price_mismatch > 0:
        issues.append(f"买卖价格不匹配: {price_mismatch} 笔")

    status = "PASS" if not issues else "FAIL"
    return {
        "name": "交易对比",
        "status": status,
        "issues": issues,
        "bt_count": len(bt_trades),
        "live_count": len(live_trades),
        "common_count": len(common_keys),
        "only_bt": len(only_bt),
        "only_live": len(only_live),
        "price_mismatch": price_mismatch,
    }


def _compare_holdings_count() -> dict:
    from glob import glob

    bt_nav = load_backtest_nav()
    bt_holding_map = dict(zip(bt_nav["date"], bt_nav["n_positions"]))

    hold_files = sorted(glob(os.path.join(HOLDINGS_DIR, "*.parquet")))
    issues = []
    n_match = 0
    n_mismatch = 0

    for f in hold_files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        date = pd.to_datetime(df["date"].iloc[0])
        live_count = len(df)
        bt_count = bt_holding_map.get(date, None)
        if bt_count is not None and live_count != bt_count:
            n_mismatch += 1
            if n_mismatch <= 5:
                issues.append(f"{date.strftime('%Y-%m-%d')}: bt={bt_count} vs live={live_count}")
        elif bt_count is not None:
            n_match += 1

    if n_mismatch > 5:
        issues.append(f"... 共 {n_mismatch} 天持仓数不匹配")

    status = "PASS" if n_mismatch == 0 else "FAIL"
    return {
        "name": "持仓数对比",
        "status": status,
        "issues": issues,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
    }


def compare_equity_curves() -> dict:
    print("=" * 70)
    print("Layer 4: 状态与账户一致性检查")
    print("=" * 70)

    results = []

    r = _compare_nav_curves()
    results.append(r)
    print(f"  {'NAV曲线':<20} {r['status']:<6} common={r.get('common_dates','?')}, "
          f"max_diff={r.get('max_nav_diff',0):.8f}, "
          f"bt_final={r.get('bt_final',0):.6f}, live_final={r.get('live_final',0):.6f}")
    for i in r.get("issues", []):
        print(f"    ⚠ {i}")

    r = _compare_trades()
    results.append(r)
    print(f"  {'交易对比':<20} {r['status']:<6} common={r.get('common_count','?')}, "
          f"only_bt={r.get('only_bt',0)}, only_live={r.get('only_live',0)}, "
          f"price_mismatch={r.get('price_mismatch',0)}")
    for i in r.get("issues", []):
        print(f"    ⚠ {i}")

    r = _compare_holdings_count()
    results.append(r)
    print(f"  {'持仓数对比':<20} {r['status']:<6} match={r.get('n_match','?')}, "
          f"mismatch={r.get('n_mismatch',0)}")
    for i in r.get("issues", []):
        print(f"    ⚠ {i}")

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"
    print(f"\n结果: {overall} (PASS={n_pass}, FAIL={n_fail})")

    return {"status": overall, "n_pass": n_pass, "n_fail": n_fail}


if __name__ == "__main__":
    result = compare_equity_curves()
    sys.exit(0 if result["status"] == "PASS" else 1)
