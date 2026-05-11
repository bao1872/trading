#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线脚本：统一引擎版

Purpose:
    调用统一策略引擎 (strategy_runner) 运行回测基线。
    不再直接调用 dynamic_exit_backtest_v2.run_backtest。

Inputs:
    - stop_experiment/pipeline/stop_config.py (PRODUCTION_PARAMS)
    - output/full_test_predictions.parquet

Outputs:
    - output/backtest_ledger/ (统一 ledger)
    - output/backtest/ (兼容旧格式)

How to Run:
    python stop_experiment/backtest/run_baseline.py

Side Effects:
    - 写 backtest_ledger/ 和 backtest/ 目录
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stop_experiment.engine.strategy_runner import run_range
from stop_experiment.pipeline.stop_config import PRODUCTION_PARAMS, BACKTEST_LEDGER_DIR


def main():
    p = PRODUCTION_PARAMS

    print("=" * 60)
    print(f"基线回测 (统一引擎)")
    print(f"  obs_day={p.get('candidate_obs_days', [1])}, max_stocks={p.get('max_stocks_default', 10)}, "
          f"exit_th={p.get('buy_cls_exit_threshold', 0.70)}")
    print(f"  ledger: {BACKTEST_LEDGER_DIR}")
    print("=" * 60)

    result = run_range(
        mode="backtest",
        params=p,
        write_ledgers=True,
        postprocess=True,
    )

    eq_path = os.path.join(BACKTEST_LEDGER_DIR, "live_equity_curve.csv")
    report_path = os.path.join(BACKTEST_LEDGER_DIR, "live_trade_report.csv")

    import pandas as pd
    if os.path.exists(eq_path):
        eq_df = pd.read_csv(eq_path)
        if not eq_df.empty:
            final_nav = eq_df["nav_live"].iloc[-1]
            max_dd = eq_df["drawdown"].max()
            print(f"\n  验收: NAV={final_nav:.4f}, max_dd={max_dd:.4f}")

    if os.path.exists(report_path):
        report_df = pd.read_csv(report_path)
        print(f"  交易: {len(report_df)} 笔")

    print(f"\n  输出: {BACKTEST_LEDGER_DIR}/")


if __name__ == "__main__":
    main()
