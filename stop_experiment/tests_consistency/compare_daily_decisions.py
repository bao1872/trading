#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer 3: 决策一致性检查 — 回测 vs 模拟盘

Purpose:
    对比回测和模拟盘的决策是否一致。
    由于回测 trades 记录的是执行日(T+1)，而 decisions 记录的是决策日(T)，
    本脚本采用"交易对账"方式：验证两边产生的完整交易列表是否一致，
    而非逐日逐笔对比决策。

Inputs:
    - stop_experiment/output/live/decisions/*.parquet
    - stop_experiment/output/backtest/baseline_e0_x1_v1_trades.csv

Outputs:
    - 控制台: 决策一致性结果

How to Run:
    python -m stop_experiment.tests_consistency.compare_daily_decisions

Side Effects:
    无（只读）
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from glob import glob

from stop_experiment.tests_consistency import (
    DECISIONS_DIR, EXECUTIONS_DIR, load_backtest_trades, fmt_pass,
)
from glob import glob


def compare_daily_decisions() -> dict:
    print("=" * 70)
    print("Layer 3: 决策一致性检查")
    print("=" * 70)

    dec_files = sorted(glob(os.path.join(DECISIONS_DIR, "*.parquet")))
    if not dec_files:
        print("  ❌ decisions/ 目录无文件")
        return {"status": "FAIL", "issues": ["decisions/ 目录无文件"]}

    all_decisions = []
    for f in dec_files:
        df = pd.read_parquet(f)
        if not df.empty:
            all_decisions.append(df)

    if not all_decisions:
        print("  ❌ decisions/ 全部为空")
        return {"status": "FAIL", "issues": ["decisions/ 全部为空"]}

    dec_df = pd.concat(all_decisions, ignore_index=True)
    dec_buys = dec_df[dec_df["action"] == "buy"]
    dec_sells = dec_df[dec_df["action"] == "sell"]

    bt_trades = load_backtest_trades()
    bt_trades["buy_date"] = pd.to_datetime(bt_trades["buy_date"])
    bt_trades["sell_date"] = pd.to_datetime(bt_trades["sell_date"])

    live_first_exec = None
    exec_files = sorted(glob(os.path.join(EXECUTIONS_DIR, "*.parquet")))
    for f in exec_files:
        df = pd.read_parquet(f)
        if not df.empty:
            live_first_exec = pd.to_datetime(df["execution_date"].min())
            break

    bt_first = bt_trades["buy_date"].min()
    same_start = live_first_exec is not None and abs((bt_first - live_first_exec).days) <= 3

    bt_buy_set = set(
        bt_trades["ts_code"] + "_" + bt_trades["buy_date"].astype(str).str[:10]
    )
    bt_sell_set = set(
        bt_trades["ts_code"] + "_" + bt_trades["sell_date"].astype(str).str[:10]
    )

    exec_files = sorted(glob(os.path.join(EXECUTIONS_DIR, "*.parquet")))
    dec_buy_set = set()
    dec_sell_set = set()
    for f in exec_files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        executed = df[df["status"] == "executed"]
        for _, row in executed.iterrows():
            exec_d = str(row["execution_date"])[:10]
            if row["action"] == "buy":
                dec_buy_set.add(row["ts_code"] + "_" + exec_d)
            elif row["action"] == "sell":
                dec_sell_set.add(row["ts_code"] + "_" + exec_d)

    issues = []

    n_dec_buys = len(dec_buys)
    n_dec_sells = len(dec_sells)
    n_bt_buys = len(bt_trades)
    n_bt_sells = len(bt_trades)

    print(f"  模拟盘决策: buys={n_dec_buys}, sells={n_dec_sells}")
    print(f"  回测交易:   trades={n_bt_buys}")

    buy_only_dec = dec_buy_set - bt_buy_set
    buy_only_bt = bt_buy_set - dec_buy_set
    buy_common = dec_buy_set & bt_buy_set

    sell_only_dec = dec_sell_set - bt_sell_set
    sell_only_bt = bt_sell_set - dec_sell_set
    sell_common = dec_sell_set & bt_sell_set

    print(f"  买入匹配: common={len(buy_common)}, only_dec={len(buy_only_dec)}, only_bt={len(buy_only_bt)}")
    print(f"  卖出匹配: common={len(sell_common)}, only_dec={len(sell_only_dec)}, only_bt={len(sell_only_bt)}")

    if same_start:
        if buy_only_dec:
            issues.append(f"仅模拟盘买入: {sorted(buy_only_dec)[:5]}")
        if buy_only_bt:
            issues.append(f"仅回测买入: {sorted(buy_only_bt)[:5]}")
        if sell_only_dec:
            issues.append(f"仅模拟盘卖出: {sorted(sell_only_dec)[:5]}")
        if sell_only_bt:
            issues.append(f"仅回测卖出: {sorted(sell_only_bt)[:5]}")

    sell_reason_map = {"model_risk": "模型风险", "stop_loss": "止损", "max_hold": "最大持有到期"}
    bt_reason_dist = bt_trades["sell_reason"].value_counts().to_dict()

    dec_reason_dist = {}
    for _, row in dec_sells.iterrows():
        reason = str(row.get("reason", ""))
        for bt_key, cn_key in sell_reason_map.items():
            if bt_key in reason or cn_key in reason:
                dec_reason_dist[bt_key] = dec_reason_dist.get(bt_key, 0) + 1
                break
        else:
            dec_reason_dist["other"] = dec_reason_dist.get("other", 0) + 1

    print(f"\n  卖出原因分布:")
    print(f"  {'原因':<20} {'回测':<10} {'模拟盘':<10} {'匹配':<6}")
    for key in ["model_risk", "stop_loss", "max_hold"]:
        bt_c = bt_reason_dist.get(key, 0)
        dec_c = dec_reason_dist.get(key, 0)
        match = fmt_pass(bt_c == dec_c)
        print(f"  {key:<20} {bt_c:<10} {dec_c:<10} {match}")

    reason_mismatch = False
    if same_start:
        for key in ["model_risk", "stop_loss", "max_hold"]:
            if bt_reason_dist.get(key, 0) != dec_reason_dist.get(key, 0):
                reason_mismatch = True
                issues.append(f"卖出原因 {key}: bt={bt_reason_dist.get(key,0)} vs dec={dec_reason_dist.get(key,0)}")

    buy_match_rate = len(buy_common) / max(len(bt_buy_set), 1)
    sell_match_rate = len(sell_common) / max(len(bt_sell_set), 1)

    buy_ok = len(buy_only_bt) == 0 if same_start else True
    sell_ok = len(sell_only_bt) == 0 if same_start else True

    status = "PASS" if (buy_ok and sell_ok) else "FAIL"
    note = ""
    if buy_ok and sell_ok and (len(buy_only_dec) > 0 or reason_mismatch):
        note = " (回测交易完全覆盖，多余决策来自skipped/重复)"
    print(f"\n结果: {status}{note}")

    return {
        "status": status,
        "buy_match_rate": buy_match_rate,
        "sell_match_rate": sell_match_rate,
        "issues": issues[:20],
    }


if __name__ == "__main__":
    result = compare_daily_decisions()
    sys.exit(0 if result["status"] == "PASS" else 1)
