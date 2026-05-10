#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 输出完整性检查 — 验证 output/ 下所有 CSV/Parquet 输出的完整性和一致性

Purpose:
    检查模拟盘所有输出文件的完整性、列完整性、数据合理性，
    以及与回测基线的一致性。

Inputs:
    - stop_experiment/output/live/live_equity_curve.csv
    - stop_experiment/output/live/live_trade_report.csv
    - stop_experiment/output/daily_trading_sheet.csv
    - stop_experiment/output/live/decisions/*.parquet
    - stop_experiment/output/live/executions/*.parquet
    - stop_experiment/output/holdings/*.parquet
    - stop_experiment/output/backtest/baseline_e0_x1_v1_*.csv (基线对比)

Outputs:
    - 控制台: 检查结果汇总

How to Run:
    python -m stop_experiment.tests_consistency.check_csv_outputs

Side Effects:
    无（只读）
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from glob import glob

from stop_experiment.tests_consistency import (
    OUTPUT_DIR, LIVE_DIR, BACKTEST_DIR, HOLDINGS_DIR,
    DECISIONS_DIR, EXECUTIONS_DIR, NAV_TOL, WEIGHT_SUM_TOL,
    load_backtest_nav, load_backtest_trades,
    load_live_equity_curve, load_live_trade_report,
    fmt_pass, check_no_nan_inf,
    BASELINE_PREFIX,
)
from stop_experiment.pipeline.stop_config import BASELINE_E0_X1_V1_PARAMS

MAX_STOCKS = BASELINE_E0_X1_V1_PARAMS.get("max_stocks", 10)


def _check_live_equity_curve() -> dict:
    issues = []
    path = os.path.join(LIVE_DIR, "live_equity_curve.csv")
    if not os.path.exists(path):
        return {"name": "live_equity_curve.csv", "status": "FAIL", "issues": ["文件不存在"]}

    df = load_live_equity_curve()
    required_cols = ["date", "cash", "market_value", "equity", "nav_live",
                     "n_positions", "daily_return", "drawdown"]
    for c in required_cols:
        if c not in df.columns:
            issues.append(f"缺列: {c}")

    if issues:
        return {"name": "live_equity_curve.csv", "status": "FAIL", "issues": issues}

    if len(df) == 0:
        issues.append("空 DataFrame")
    if abs(df["nav_live"].iloc[0] - 1.0) > 0.001:
        issues.append(f"首日 nav_live={df['nav_live'].iloc[0]:.4f} ≠ 1.0")

    nan_inf = check_no_nan_inf(df, ["nav_live", "daily_return", "drawdown", "equity"])
    issues.extend(nan_inf)

    if (df["drawdown"] < -0.001).any():
        issues.append(f"drawdown < 0: {(df['drawdown'] < -0.001).sum()} 行")

    bt_nav = load_backtest_nav()
    merged = df.merge(bt_nav[["date", "nav"]], on="date", how="inner")
    if not merged.empty:
        bt_first = pd.to_datetime(bt_nav["date"].min())
        live_first_with_pos = pd.to_datetime(df[df["n_positions"] > 0]["date"].min()) if "n_positions" in df.columns else None
        if live_first_with_pos is not None and not pd.isna(live_first_with_pos):
            same_start = abs((bt_first - live_first_with_pos).days) <= 3
        else:
            same_start = True
        if same_start:
            max_diff = (merged["nav_live"] - merged["nav"]).abs().max()
            if max_diff > NAV_TOL:
                issues.append(f"NAV vs 回测最大偏差: {max_diff:.8f} > {NAV_TOL}")

    status = "PASS" if not issues else "FAIL"
    return {"name": "live_equity_curve.csv", "status": status, "issues": issues,
            "rows": len(df), "final_nav": df["nav_live"].iloc[-1] if len(df) > 0 else None}


def _check_live_trade_report() -> dict:
    issues = []
    path = os.path.join(LIVE_DIR, "live_trade_report.csv")
    if not os.path.exists(path):
        return {"name": "live_trade_report.csv", "status": "FAIL", "issues": ["文件不存在"]}

    df = load_live_trade_report()
    required_cols = ["序号", "股票代码", "股票名称", "买入日期", "卖出日期",
                     "买入价", "卖出价", "持仓天数", "毛盈亏%", "净盈亏%",
                     "盈亏标签", "退出原因", "入场评分"]
    for c in required_cols:
        if c not in df.columns:
            issues.append(f"缺列: {c}")

    if issues:
        return {"name": "live_trade_report.csv", "status": "FAIL", "issues": issues}

    bt_trades = load_backtest_trades()
    bt_trades["buy_date"] = pd.to_datetime(bt_trades["buy_date"])
    bt_march = bt_trades[bt_trades["buy_date"] >= pd.Timestamp("2026-03-02")]
    if len(df) != len(bt_march):
        issues.append(f"交易笔数: live={len(df)} vs backtest_march={len(bt_march)} (起始不同可接受)")

    n_unknown = (df["退出原因"] == "unknown").sum()
    if n_unknown > 0:
        issues.append(f"unknown 退出原因: {n_unknown} 笔")

    if n_unknown > 0:
        status = "WARN"
    elif len(df) != len(bt_march):
        status = "WARN"
    else:
        status = "PASS"
    return {"name": "live_trade_report.csv", "status": status, "issues": issues,
            "rows": len(df), "unknown_count": n_unknown}


def _check_daily_trading_sheet() -> dict:
    issues = []
    path = os.path.join(OUTPUT_DIR, "daily_trading_sheet.csv")
    if not os.path.exists(path):
        return {"name": "daily_trading_sheet.csv", "status": "SKIP", "issues": ["文件不存在（非必须）"]}

    df = pd.read_csv(path)
    required_cols = ["ts_code", "obs_date", "obs_day", "composite_score",
                     "sell_score", "action", "grade"]
    for c in required_cols:
        if c not in df.columns:
            issues.append(f"缺列: {c}")

    if issues:
        return {"name": "daily_trading_sheet.csv", "status": "FAIL", "issues": issues}

    valid_actions = {"buy", "sell", "hold"}
    invalid_actions = set(df["action"].dropna().unique()) - valid_actions
    if invalid_actions:
        issues.append(f"非法 action: {invalid_actions}")

    valid_grades = {"A", "B", "C", "D"}
    invalid_grades = set(df["grade"].dropna().unique()) - valid_grades
    if invalid_grades:
        issues.append(f"非法 grade: {invalid_grades}")

    status = "PASS" if not issues else "FAIL"
    return {"name": "daily_trading_sheet.csv", "status": status, "issues": issues, "rows": len(df)}


def _check_parquet_dir(dir_path: str, name: str, required_cols: list[str],
                       extra_checks=None) -> dict:
    issues = []
    if not os.path.isdir(dir_path):
        return {"name": name, "status": "FAIL", "issues": [f"目录不存在: {dir_path}"]}

    files = sorted(glob(os.path.join(dir_path, "*.parquet")))
    if len(files) == 0:
        issues.append("无 parquet 文件")

    n_total = len(files)
    n_valid = 0
    n_empty = 0
    col_issues = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if df.empty:
                n_empty += 1
                n_valid += 1
                continue
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                col_issues.append(f"{os.path.basename(f)}: 缺列 {missing}")
                continue
            n_valid += 1
            if extra_checks:
                extra_issues = extra_checks(df, os.path.basename(f))
                issues.extend(extra_issues)
        except Exception as e:
            issues.append(f"{os.path.basename(f)}: 读取失败 {e}")

    issues.extend(col_issues)
    if n_valid < n_total:
        issues.append(f"有效文件: {n_valid}/{n_total}")

    status = "PASS" if not issues and n_valid > 0 else "FAIL"
    return {"name": name, "status": status, "issues": issues, "files": n_total, "valid": n_valid}


def _holdings_extra_checks(df: pd.DataFrame, fname: str) -> list:
    issues = []
    if (df["weight"] <= 0).any():
        issues.append(f"{fname}: weight<=0: {(df['weight']<=0).sum()}")
    if len(df) > MAX_STOCKS + 1:
        issues.append(f"{fname}: 持仓数 {len(df)} > max_stocks+1={MAX_STOCKS+1}")
    w_sum = df["weight"].sum()
    if w_sum > 1.0 + WEIGHT_SUM_TOL:
        issues.append(f"{fname}: weight之和={w_sum:.4f} > 1.0")
    return issues


def _executions_extra_checks(df: pd.DataFrame, fname: str) -> list:
    issues = []
    executed = df[df["status"] == "executed"]
    if not executed.empty:
        bad_price = executed["executed_price"].isna() | (executed["executed_price"] <= 0)
        if bad_price.any():
            issues.append(f"{fname}: executed_price<=0或NaN: {bad_price.sum()}")
    return issues


def check_csv_outputs() -> dict:
    print("=" * 70)
    print("CSV 输出完整性检查")
    print("=" * 70)

    results = []

    r = _check_live_equity_curve()
    results.append(r)
    detail = f" rows={r.get('rows','?')}"
    if r.get("final_nav"):
        detail += f" final_nav={r['final_nav']:.6f}"
    print(f"  {r['name']:<35} {r['status']:<6}{detail}")
    for i in r.get("issues", []):
        print(f"    ⚠ {i}")

    r = _check_live_trade_report()
    results.append(r)
    detail = f" rows={r.get('rows','?')}"
    if r.get("unknown_count", 0) > 0:
        detail += f" unknown={r['unknown_count']}"
    print(f"  {r['name']:<35} {r['status']:<6}{detail}")
    for i in r.get("issues", []):
        print(f"    ⚠ {i}")

    r = _check_daily_trading_sheet()
    results.append(r)
    detail = f" rows={r.get('rows','?')}" if r.get("rows") else ""
    print(f"  {r['name']:<35} {r['status']:<6}{detail}")
    for i in r.get("issues", []):
        print(f"    ⚠ {i}")

    r = _check_parquet_dir(
        DECISIONS_DIR, "decisions/",
        ["decision_date", "ts_code", "signal_id", "action", "reason", "score"],
    )
    results.append(r)
    detail = f" files={r.get('files','?')} valid={r.get('valid','?')}"
    print(f"  {r['name']:<35} {r['status']:<6}{detail}")
    for i in r.get("issues", []):
        print(f"    ⚠ {i}")

    r = _check_parquet_dir(
        EXECUTIONS_DIR, "executions/",
        ["execution_date", "decision_date", "ts_code", "signal_id",
         "action", "planned_price", "executed_price", "status"],
        extra_checks=_executions_extra_checks,
    )
    results.append(r)
    detail = f" files={r.get('files','?')} valid={r.get('valid','?')}"
    print(f"  {r['name']:<35} {r['status']:<6}{detail}")
    for i in r.get("issues", []):
        print(f"    ⚠ {i}")

    r = _check_parquet_dir(
        HOLDINGS_DIR, "holdings/",
        ["date", "code", "ts_code", "entry_date", "entry_price",
         "weight", "days_held", "signal_id", "score"],
        extra_checks=_holdings_extra_checks,
    )
    results.append(r)
    detail = f" files={r.get('files','?')} valid={r.get('valid','?')}"
    print(f"  {r['name']:<35} {r['status']:<6}{detail}")
    for i in r.get("issues", []):
        print(f"    ⚠ {i}")

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    total = len(results)

    overall = "PASS" if n_fail == 0 else "FAIL"
    print(f"\n结果: {overall} (PASS={n_pass}, WARN={n_warn}, FAIL={n_fail}, SKIP={n_skip})")

    return {"status": overall, "n_pass": n_pass, "n_warn": n_warn,
            "n_fail": n_fail, "n_skip": n_skip, "total": total}


if __name__ == "__main__":
    result = check_csv_outputs()
    sys.exit(0 if result["status"] in ("PASS", "WARN") else 1)
