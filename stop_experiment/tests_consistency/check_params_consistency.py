#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer 1: 参数一致性检查 — 回测 vs 模拟盘

Purpose:
    验证回测基线参数与模拟盘运行时参数完全一致。

Inputs:
    - stop_experiment/pipeline/stop_config.py (BASELINE_E0_X1_V1_PARAMS, PRODUCTION_PARAMS)
    - stop_experiment/pipeline/09_paper_trading_runner.py (运行时参数)
    - stop_experiment/pipeline/07_generate_daily_predictions.py (模型路径)
    - stop_experiment/backtest/daily_state_machine.py (运行时参数)

Outputs:
    - 控制台: 参数对比表格 + PASS/FAIL

How to Run:
    python -m stop_experiment.tests_consistency.check_params_consistency

Side Effects:
    无（只读代码中的参数定义）
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from stop_experiment.pipeline.stop_config import (
    BASELINE_E0_X1_V1_PARAMS,
    PRODUCTION_PARAMS,
    MODELS_DIR,
)
from stop_experiment.tests_consistency import fmt_pass


def check_params_consistency() -> dict:
    bt = BASELINE_E0_X1_V1_PARAMS
    live = PRODUCTION_PARAMS

    checks = [
        ("candidate_obs_days", bt.get("candidate_obs_days"), live.get("candidate_obs_days")),
        ("score_col", bt.get("score_col"), live.get("score_col")),
        ("strategy_default", bt.get("strategy_default"), live.get("strategy_default")),
        ("buy_cls_exit_threshold", bt.get("buy_cls_exit_threshold"), live.get("buy_cls_exit_threshold")),
        ("stop_loss", bt.get("stop_loss"), live.get("stop_loss")),
        ("max_hold_days", bt.get("max_hold_days"), live.get("max_hold_days")),
        ("max_stocks", bt.get("max_stocks"), live.get("max_stocks")),
        ("exit_mode", bt.get("exit_mode"), live.get("exit_mode")),
        ("buy_signal_threshold", bt.get("buy_signal_threshold"), live.get("buy_signal_threshold")),
        ("profile", bt.get("profile"), live.get("profile")),
    ]

    print("=" * 70)
    print("Layer 1: 参数一致性检查")
    print("=" * 70)
    print(f"{'参数':<30} {'回测(SSOT)':<20} {'模拟盘':<20} {'匹配':<6}")
    print("-" * 70)

    n_pass = 0
    n_fail = 0
    failed = []
    for name, bt_val, live_val in checks:
        passed = bt_val == live_val
        if passed:
            n_pass += 1
        else:
            n_fail += 1
            failed.append(name)
        bt_str = str(bt_val)[:18]
        live_str = str(live_val)[:18]
        print(f"{name:<30} {bt_str:<20} {live_str:<20} {fmt_pass(passed)}")

    print("-" * 70)

    models_dir_check = os.path.isdir(MODELS_DIR)
    print(f"{'models_dir 存在':<30} {MODELS_DIR:<20} {'':<20} {fmt_pass(models_dir_check)}")

    if models_dir_check:
        n_pass += 1
    else:
        n_fail += 1
        failed.append("models_dir")

    total = n_pass + n_fail
    status = "PASS" if n_fail == 0 else "FAIL"
    print(f"\n结果: {status} ({n_pass}/{total})")
    if failed:
        print(f"失败项: {failed}")

    return {"status": status, "n_pass": n_pass, "n_fail": n_fail, "failed": failed}


if __name__ == "__main__":
    result = check_params_consistency()
    sys.exit(0 if result["status"] == "PASS" else 1)
