#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线脚本：E0+X1+0.70 冻结版

Purpose:
    加载 BASELINE_E0_X1_V1_PARAMS，运行单次回测，输出标准文件。
    不承载任何新逻辑 — 纯参数编排。

Pipeline Position:
    基线冻结（Phase 0-3 完结后）。
    下游：entry_recheck_entries.py, exit_recheck_exits.py, out_of_sample_validator.py

Inputs:
    - stop_experiment/pipeline/stop_config.py (BASELINE_E0_X1_V1_PARAMS)

Outputs:
    - output/backtest/baseline_e0_x1_v1_nav.csv
    - output/backtest/baseline_e0_x1_v1_trades.csv
    - output/backtest/baseline_e0_x1_v1_summary.csv
    - output/backtest/baseline_e0_x1_v1_equity_curve.csv
    - output/backtest/baseline_e0_x1_v1_trade_report.csv
    - output/backtest/baseline_e0_x1_v1_trade_summary.csv

How to Run:
    python stop_experiment/backtest/run_baseline.py

Side Effects:
    - 读取预测数据和K线数据，输出CSV
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np

from stop_experiment.pipeline.stop_config import BASELINE_E0_X1_V1_PARAMS, BACKTEST_DIR
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest, compute_summary,
)
from stop_experiment.backtest.build_equity_curve import save_equity_curve
from stop_experiment.backtest.trade_report import save_trade_report


def main():
    p = BASELINE_E0_X1_V1_PARAMS

    print("=" * 60)
    print(f"基线冻结: {p['profile']}")
    print(f"  {p['description']}")
    print(f"  obs_day={p['candidate_obs_days']}, top_k={p['max_stocks']}, "
          f"exit_th={p['buy_cls_exit_threshold']}")
    print(f"  预期: NAV={p['expected_nav']:.4f}, Sharpe={p['expected_sharpe']:.2f}, "
          f"MDD={p['expected_mdd']:.4f}, n_trades={p['expected_n_trades']}")
    print("=" * 60)

    test_df, price, td, prev, pred = _load_data(
        candidate_obs_days=p["candidate_obs_days"]
    )
    test_df["trading_date"] = test_df["obs_date"]
    test_df["score"] = test_df[p["score_col"]]
    test_df["sell_score"] = test_df[p["score_col"]]

    print(f"  数据: {len(test_df)} 条, {len(td)} 交易日")
    print("  运行回测...", flush=True)

    result = run_backtest(
        test_df, price, td, prev, pred,
        max_stocks=p["max_stocks"],
        strategy="sell_score",
        exit_mode=p["exit_mode"],
        stop_loss=p["stop_loss"],
        max_hold_days=p["max_hold_days"],
        buy_cls_exit_threshold=p["buy_cls_exit_threshold"],
        exit_sub_mode=p["exit_sub_mode"],
        buy_reg_exit_threshold=p["buy_reg_exit_threshold"],
        buy_cost=p["buy_cost"],
        sell_cost=p["sell_cost"],
        strict=True,
    )

    s = compute_summary(result)

    # 输出 NAV
    nav_df = result["nav_df"]
    nav_path = os.path.join(BACKTEST_DIR, f"{p['profile']}_nav.csv")
    nav_df.to_csv(nav_path, index=False)

    # 输出 trades
    trades_df = result["trades_df"]
    trades_path = os.path.join(BACKTEST_DIR, f"{p['profile']}_trades.csv")
    trades_df.to_csv(trades_path, index=False)

    # 输出 summary
    summary = {
        "profile": p["profile"],
        "final_nav": s.get("final_nav"),
        "sharpe": s.get("sharpe"),
        "max_dd": s.get("max_dd"),
        "win_rate": s.get("win_rate"),
        "avg_hold_days": s.get("avg_hold_days"),
        "n_trades": s.get("n_trades"),
        "model_exits": int((trades_df["sell_reason"] == "model_risk").sum()),
        "stop_exits": int((trades_df["sell_reason"] == "stop_loss").sum()),
        "max_hold_exits": int((trades_df["sell_reason"] == "max_hold").sum()),
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(BACKTEST_DIR, f"{p['profile']}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # 输出净值曲线
    eq_path = os.path.join(BACKTEST_DIR, f"{p['profile']}_equity_curve.csv")
    eq_df = save_equity_curve(result, price, td, eq_path, label="基线")

    # 输出交易报告
    report_path = os.path.join(BACKTEST_DIR, f"{p['profile']}_trade_report.csv")
    report_summary_path = os.path.join(BACKTEST_DIR, f"{p['profile']}_trade_summary.csv")
    _, trade_summary = save_trade_report(trades_df, report_path, report_summary_path, label="基线")

    # 验收
    nav_val = s.get("final_nav", 0)
    sharpe_val = s.get("sharpe", 0)
    mdd_val = s.get("max_dd", 0)
    n_tr = s.get("n_trades", 0)

    print(f"\n  实际: NAV={nav_val:.4f}, Sharpe={sharpe_val:.2f}, "
          f"MDD={mdd_val:.4f}, n_trades={n_tr}")
    print(f"  预期: NAV={p['expected_nav']:.4f}, Sharpe={p['expected_sharpe']:.2f}, "
          f"MDD={p['expected_mdd']:.4f}, n_trades={p['expected_n_trades']}")

    nav_ok = abs(nav_val - p["expected_nav"]) < 0.01
    mdd_ok = abs(mdd_val - p["expected_mdd"]) < 0.0005
    n_ok = n_tr == p["expected_n_trades"]

    if nav_ok and mdd_ok and n_ok:
        print("  ✅ 基线验收通过")
    else:
        print(f"  ⚠️ 基线偏离: NAV={'OK' if nav_ok else 'MISMATCH'}, "
              f"MDD={'OK' if mdd_ok else 'MISMATCH'}, n_trades={'OK' if n_ok else 'MISMATCH'}")

    # 净值曲线验收
    nav_first = eq_df["nav"].iloc[0]
    nav_last = eq_df["nav"].iloc[-1]
    dd_min = eq_df["drawdown"].max()
    print(f"\n  净值曲线验收:")
    print(f"    nav[0]={nav_first:.4f}, final_nav={nav_last:.4f}, max_dd={dd_min:.4f}")
    print(f"    ✅ nav[0] == 1.0: {abs(nav_first - 1.0) < 0.001}")
    print(f"    ✅ final_nav ≈ summary: {abs(nav_last - nav_val) < 0.005} (Δ={abs(nav_last - nav_val):.4f})")

    print(f"\n  输出: {nav_path}")
    print(f"        {trades_path}")
    print(f"        {summary_path}")
    print(f"        {eq_path}")
    print(f"        {report_path}")
    print(f"        {report_summary_path}")


if __name__ == "__main__":
    main()