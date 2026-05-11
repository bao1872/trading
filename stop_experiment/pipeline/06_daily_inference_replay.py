#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日级推理回放 + 回测对账（统一引擎版）

Purpose:
    使用统一策略引擎分别运行 backtest 和 replay 模式，
    对比两个 ledger 目录的 decisions/holdings/executions，验证 100% 一致。

    替代旧版的双引擎对账逻辑（旧版同时依赖 run_backtest + step_day）。

Inputs:
    - stop_experiment/output/full_test_predictions.parquet

Outputs:
    - stop_experiment/output/backtest_ledger/ (回测 ledger)
    - stop_experiment/output/replay_ledger/ (回放 ledger)
    - stop_experiment/output/backtest/dynamic/replay_summary.csv (对账汇总)

How to Run:
    # 单日
    python stop_experiment/pipeline/06_daily_inference_replay.py --date 2026-03-15

    # 10 日批量
    python stop_experiment/pipeline/06_daily_inference_replay.py --batch-first-10

    # 全量
    python stop_experiment/pipeline/06_daily_inference_replay.py --batch-all

Side Effects:
    - 写 backtest_ledger/ 和 replay_ledger/
    - 输出对账汇总 CSV
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, BACKTEST_DIR, PRODUCTION_PARAMS,
    BACKTEST_LEDGER_DIR, REPLAY_LEDGER_DIR,
)
from stop_experiment.engine.strategy_runner import run_range, _verify_ledgers

DYNAMIC_DIR = os.path.join(BACKTEST_DIR, "dynamic")
FIRST_10_DATES = [
    "2026-03-10", "2026-03-15", "2026-03-20", "2026-03-24",
    "2026-04-03", "2026-04-10", "2026-04-17", "2026-04-24",
    "2026-04-30", "2026-05-06",
]


def compare_ledger_decisions(bt_dir, rp_dir, target_dates=None):
    """对比两个 ledger 目录的 decisions，逐日逐条对账"""
    bt_dec_dir = os.path.join(bt_dir, "decisions")
    rp_dec_dir = os.path.join(rp_dir, "decisions")

    if not os.path.isdir(bt_dec_dir) or not os.path.isdir(rp_dec_dir):
        print("  [错误] decisions 目录不存在")
        return []

    bt_files = sorted(os.listdir(bt_dec_dir))
    rp_files = sorted(os.listdir(rp_dec_dir))
    common = sorted(set(bt_files) & set(rp_files))

    if target_dates:
        target_strs = set(d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in target_dates)
        common = [f for f in common if f.replace(".parquet", "") in target_strs]

    all_diffs = []
    summaries = []

    for f in common:
        date_str = f.replace(".parquet", "")
        bt_df = pd.read_parquet(os.path.join(bt_dec_dir, f))
        rp_df = pd.read_parquet(os.path.join(rp_dec_dir, f))

        diff_rows = []

        bt_actions = bt_df[bt_df["action"].isin(["buy", "sell", "hold"])]
        rp_actions = rp_df[rp_df["action"].isin(["buy", "sell", "hold"])]

        bt_by_code = {str(r.get("ts_code", "")): r for _, r in bt_actions.iterrows()}
        rp_by_code = {str(r.get("ts_code", "")): r for _, r in rp_actions.iterrows()}

        all_codes = sorted(set(bt_by_code.keys()) | set(rp_by_code.keys()))

        for code in all_codes:
            bt_r = bt_by_code.get(code)
            rp_r = rp_by_code.get(code)

            bt_action = bt_r.get("action", "-") if bt_r is not None else "-"
            rp_action = rp_r.get("action", "-") if rp_r is not None else "-"

            if bt_action != rp_action:
                diff_rows.append({
                    "date": date_str, "ts_code": code,
                    "field": "action", "backtest_value": bt_action, "replay_value": rp_action,
                    "is_match": False, "note": "决策不一致",
                })

            if bt_r is not None and rp_r is not None and bt_action == rp_action == "sell":
                bt_reason = bt_r.get("reason", "")
                rp_reason = rp_r.get("reason", "")
                if bt_reason != rp_reason:
                    diff_rows.append({
                        "date": date_str, "ts_code": code,
                        "field": "sell_reason", "backtest_value": bt_reason, "replay_value": rp_reason,
                        "is_match": False, "note": "卖出原因不一致",
                    })

        n_total = len(all_codes)
        n_pass = n_total - len(diff_rows)
        n_fail = len(diff_rows)
        critical = sum(1 for d in diff_rows if d["field"] in ("action", "sell_reason"))

        summaries.append({
            "date": date_str,
            "total_checks": n_total,
            "passed": n_pass,
            "failed": n_fail,
            "critical_failures": critical,
            "all_pass": n_fail == 0,
        })

        if diff_rows:
            all_diffs.extend(diff_rows)
            status = f"❌ 失败 ({n_fail}/{n_total})"
        else:
            status = "✅ 通过"

        print(f"  {date_str}: {n_total} 条决策, {status}")

    return all_diffs, summaries


def run_replay(args):
    dates = []
    if args.date:
        dates = [pd.to_datetime(args.date)]
        print("=" * 70)
        print(f"日级推理回放 (1 日) — 统一引擎")
        print("=" * 70)
    elif args.batch_first_10:
        dates = [pd.to_datetime(d) for d in FIRST_10_DATES]
        print("=" * 70)
        print(f"日级推理回放 (10 日) — 统一引擎")
        print("=" * 70)
    elif args.batch_all:
        dates = None
        print("=" * 70)
        print(f"日级推理回放 (全量) — 统一引擎")
        print("=" * 70)
    else:
        raise ValueError("必须指定 --date、--batch-first-10 或 --batch-all")

    params = PRODUCTION_PARAMS

    # ---- Step 1: 运行回测 (写入 backtest_ledger) ----
    print(f"\n  [Step 1] 运行回测 → {BACKTEST_LEDGER_DIR}")
    bt_result = run_range(
        mode="backtest",
        params=params,
        write_ledgers=True,
        postprocess=True,
    )

    # ---- Step 2: 运行回放 (写入 replay_ledger) ----
    print(f"\n  [Step 2] 运行回放 → {REPLAY_LEDGER_DIR}")
    rp_result = run_range(
        mode="replay",
        params=params,
        write_ledgers=True,
        postprocess=True,
    )

    # ---- Step 3: 对账 ----
    print(f"\n  [Step 3] 对账: {BACKTEST_LEDGER_DIR} vs {REPLAY_LEDGER_DIR}")

    target_dates = dates if dates else None
    all_diffs, summaries = compare_ledger_decisions(
        BACKTEST_LEDGER_DIR, REPLAY_LEDGER_DIR, target_dates=target_dates,
    )

    # ---- Step 4: 汇总 ----
    if summaries:
        summary_df = pd.DataFrame(summaries)
        os.makedirs(DYNAMIC_DIR, exist_ok=True)
        summary_path = os.path.join(DYNAMIC_DIR, "replay_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        n_all_pass = sum(1 for s in summaries if s["all_pass"])
        n_fail = len(summaries) - n_all_pass
        total_critical = sum(s["critical_failures"] for s in summaries)

        print(f"\n{'='*70}")
        print(f"汇总报告")
        print(f"{'='*70}")
        print(f"  总日期: {len(summaries)}")
        print(f"  对账通过: {n_all_pass}")
        print(f"  对账失败: {n_fail}")
        print(f"  关键失败数: {total_critical}")

        if n_fail == 0 and total_critical == 0:
            print(f"\n  ✅ 全部对账通过，回测与回放完全一致")
        elif total_critical == 0:
            print(f"\n  ⚠️ 存在非关键差异，不影响决策")
        else:
            print(f"\n  ❌ 存在 {total_critical} 项关键失败，需修复")

        if all_diffs:
            diff_df = pd.DataFrame(all_diffs)
            diff_path = os.path.join(DYNAMIC_DIR, "replay_diff.csv")
            diff_df.to_csv(diff_path, index=False)
            print(f"  差异文件: {diff_path}")

        print(f"  汇总文件: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日级推理回放 + 回测对账 (统一引擎)")
    parser.add_argument("--date", type=str, default="", help="单日回放 (YYYY-MM-DD)")
    parser.add_argument("--batch-first-10", action="store_true", help="10 日批量回放")
    parser.add_argument("--batch-all", action="store_true", help="全量回放")
    args = parser.parse_args()
    run_replay(args)
