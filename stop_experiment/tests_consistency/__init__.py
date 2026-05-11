#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测 vs 模拟盘一致性验收框架 — 公共常量与工具

Purpose:
    定义四层一致性验收框架的公共常量、路径、容差和工具函数，
    供 check_params / compare_predictions / compare_decisions /
    compare_equity / check_csv_outputs 共享。

Inputs/Outputs:
    无直接 I/O，仅提供常量和函数。

How to Run:
    from stop_experiment.tests_consistency import (
        OUTPUT_DIR, BACKTEST_DIR, LIVE_DIR, HOLDINGS_DIR,
        NAV_TOL, PRED_TOL, load_backtest_nav, load_backtest_trades,
    )

Side Effects:
    无
"""

from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "stop_experiment", "output")
BACKTEST_DIR = os.path.join(OUTPUT_DIR, "backtest")
LIVE_DIR = os.path.join(OUTPUT_DIR, "live")
HOLDINGS_DIR = os.path.join(OUTPUT_DIR, "holdings")
DECISIONS_DIR = os.path.join(LIVE_DIR, "decisions")
EXECUTIONS_DIR = os.path.join(LIVE_DIR, "executions")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

NAV_TOL = 1e-6
PRED_MEAN_TOL = 1e-6
PRED_MAX_TOL = 1e-5
WEIGHT_SUM_TOL = 0.01

BASELINE_PREFIX = "baseline_v2_3y"


def load_backtest_nav() -> pd.DataFrame:
    path = os.path.join(BACKTEST_DIR, f"{BASELINE_PREFIX}_nav.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_backtest_trades() -> pd.DataFrame:
    path = os.path.join(BACKTEST_DIR, f"{BASELINE_PREFIX}_trades.csv")
    return pd.read_csv(path)


def load_backtest_equity_curve() -> pd.DataFrame:
    path = os.path.join(BACKTEST_DIR, f"{BASELINE_PREFIX}_equity_curve.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_live_equity_curve() -> pd.DataFrame:
    path = os.path.join(LIVE_DIR, "live_equity_curve.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_live_trade_report() -> pd.DataFrame:
    path = os.path.join(LIVE_DIR, "live_trade_report.csv")
    return pd.read_csv(path)


def fmt_pass(passed: bool) -> str:
    return "✅" if passed else "❌"


def fmt_diff(val: float, tol: float) -> str:
    if abs(val) < tol:
        return f"{val:+.8f} ✅"
    return f"{val:+.8f} ❌"


def check_no_nan_inf(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    issues = []
    check_cols = cols if cols else df.select_dtypes(include=[np.number]).columns.tolist()
    for col in check_cols:
        if col not in df.columns:
            continue
        n_nan = df[col].isna().sum()
        n_inf = np.isinf(df[col]).sum() if df[col].dtype != object else 0
        if n_nan > 0:
            issues.append(f"{col}: {n_nan} NaN")
        if n_inf > 0:
            issues.append(f"{col}: {n_inf} Inf")
    return issues
