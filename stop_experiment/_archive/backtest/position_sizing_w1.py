#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓位实验 W0+W1: 排名分层仓位

Purpose:
    在固定 E0+X1+0.70 基线上，测试排名分层仓位分配：
    W0: 等权基线 (weight_mode=None)
    W1a: top1-3=1.25, top4-6=1.0, top7-10=0.75 → 归一化
    W1b: top1-2=1.40, top3-5=1.05, top6-10=0.80 → 归一化

Pipeline Position:
    Part B 仓位实验 Phase W1。
    上游：B0 (run_backtest 支持 weight_mode)
    下游：B4 (position_sizing_summary)

Inputs:
    - full_test_predictions.parquet

Outputs:
    - output/backtest/w1_results.csv
    - output/backtest/w0_trades.csv, w0_nav.csv, w0_equity_curve.csv, w0_trade_report.csv
    - output/backtest/w1a_trades.csv, w1a_nav.csv, w1a_equity_curve.csv, w1a_trade_report.csv
    - output/backtest/w1b_trades.csv, w1b_nav.csv

How to Run:
    python stop_experiment/backtest/position_sizing_w1.py

Side Effects:
    - 写 CSV
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import BASELINE_E0_X1_V1_PARAMS, BACKTEST_DIR
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest, compute_summary,
)
from stop_experiment.backtest.build_equity_curve import save_equity_curve
from stop_experiment.backtest.trade_report import save_trade_report


def monthly_win_rate(nav_df):
    """计算月胜率 (月末 NAV > 1.0 的月份占比)"""
    nav_df = nav_df.copy()
    nav_df.index = pd.to_datetime(nav_df.index)
    monthly = nav_df["nav"].resample("ME").last()
    monthly = monthly.dropna()
    if len(monthly) == 0:
        return np.nan, [], []
    win = (monthly > 1.0).sum()
    return win / len(monthly), monthly.tolist(), list(monthly.index)


def compute_concentration(snapshots, trades_df):
    """从 snapshots 计算集中度指标"""
    max_weights = []
    top3_shares = []
    hhis = []

    for snap in snapshots:
        holdings = snap.get("holdings_after", {})
        weights = [h["weight"] for h in holdings.values()]
        if not weights:
            continue
        max_weights.append(max(weights))
        sorted_w = sorted(weights, reverse=True)
        top3 = sum(sorted_w[:3])
        top3_shares.append(top3)
        hhi = sum(w * w for w in weights)
        hhis.append(hhi)

    return {
        "max_single_weight": np.mean(max_weights) if max_weights else np.nan,
        "max_single_weight_max": max(max_weights) if max_weights else np.nan,
        "avg_top3_share": np.mean(top3_shares) if top3_shares else np.nan,
        "avg_hhi": np.mean(hhis) if hhis else np.nan,
    }


def run_w1_variant(label, weight_mode, weight_params, save_outputs=True):
    """运行单个 W1 变体"""
    p = BASELINE_E0_X1_V1_PARAMS
    test_df, price, td, prev, pred = _load_data(candidate_obs_days=p["candidate_obs_days"])

    test_df["trading_date"] = test_df["obs_date"]
    test_df["score"] = test_df[p["score_col"]]
    test_df["sell_score"] = test_df[p["score_col"]]

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
        weight_mode=weight_mode,
        weight_params=weight_params,
        debug_snapshots=True,
    )

    s = compute_summary(result)
    nav = s["final_nav"]
    sharpe = s["sharpe"]
    mdd = s["max_dd"]
    calmar = s.get("calmar", np.nan)
    n_trades = s["n_trades"]
    win_rate = s["win_rate"]

    mwr, monthly_navs, monthly_idx = monthly_win_rate(result["nav_df"])
    conc = compute_concentration(result["snapshots"], result["trades_df"])

    if save_outputs:
        variant_tag = label.split(" ")[0].lower().replace("(", "").replace(")", "")
        trades_path = os.path.join(BACKTEST_DIR, f"{variant_tag}_trades.csv")
        nav_path = os.path.join(BACKTEST_DIR, f"{variant_tag}_nav.csv")
        result["trades_df"].to_csv(trades_path, index=False)
        result["nav_df"].to_csv(nav_path, index=False)

        if variant_tag in ("w0", "w1a"):
            eq_path = os.path.join(BACKTEST_DIR, f"{variant_tag}_equity_curve.csv")
            save_equity_curve(result, price, td, eq_path, label=label)
            report_path = os.path.join(BACKTEST_DIR, f"{variant_tag}_trade_report.csv")
            save_trade_report(result["trades_df"], report_path, label=label)

    return {
        "label": label,
        "weight_mode": weight_mode,
        "weight_params": str(weight_params) if weight_params else "",
        "nav": nav,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "monthly_win_rate": mwr,
        "monthly_navs": str(monthly_navs),
        "max_single_weight": conc["max_single_weight"],
        "max_single_weight_max": conc["max_single_weight_max"],
        "avg_top3_share": conc["avg_top3_share"],
        "avg_hhi": conc["avg_hhi"],
    }


def judge(label, row, w0):
    """根据通过标准判定"""
    nav = row["nav"]
    sharpe = row["sharpe"]
    mdd = row["mdd"]
    mwr = row["monthly_win_rate"]

    # 否决检查
    if nav < w0["nav"]:
        return "❌ 否决: NAV < W0"
    if mdd < w0["mdd"] - 0.02:
        return "❌ 否决: MDD 恶化 > 2pp"
    if not np.isnan(mwr) and not np.isnan(w0["monthly_win_rate"]) and mwr < w0["monthly_win_rate"]:
        return "❌ 否决: 月度胜率 < W0"

    # 基础通过: NAV≥W0, Sharpe↑, MDD不恶化>2pp, 月度胜率≥W0
    basic = (
        nav >= w0["nav"]
        and sharpe > w0["sharpe"]
        and mdd >= w0["mdd"] - 0.02
        and (np.isnan(mwr) or np.isnan(w0["monthly_win_rate"]) or mwr >= w0["monthly_win_rate"])
    )

    if not basic:
        return "⚠️ 未通过: 不满足基础条件"

    # 强通过: NAV↑>5%, MDD持平或↓, 无极端依赖单月
    strength = (
        nav > w0["nav"] * 1.05
        and mdd >= w0["mdd"]
    )
    if strength:
        return "🏆 强通过"

    return "✅ 基础通过"


def main():
    p = BASELINE_E0_X1_V1_PARAMS
    print("=" * 60)
    print(f"  仓位实验 W0+W1: 排名分层仓位")
    print(f"  固定基线: E0+X1+{p['buy_cls_exit_threshold']}, obs_day={p['candidate_obs_days']}")
    print("=" * 60)

    variants = [
        ("W0 (等权基线)", None, None),
        ("W1a (1.25/1.0/0.75)", "by_rank", {"tiers": [(1, 3, 1.25), (4, 6, 1.0), (7, 10, 0.75)]}),
        ("W1b (1.40/1.05/0.80)", "by_rank", {"tiers": [(1, 2, 1.40), (3, 5, 1.05), (6, 10, 0.80)]}),
    ]

    rows = []
    w0 = None

    for label, wm, wp in variants:
        print(f"\n  运行: {label} ...")
        row = run_w1_variant(label, wm, wp)
        rows.append(row)
        print(f"    NAV={row['nav']:.4f}, Sharpe={row['sharpe']:.2f}, MDD={row['mdd']:.4f}, "
              f"n_trades={row['n_trades']}, 月胜率={row['monthly_win_rate']:.1%}")
        print(f"    集中度: max_w={row['max_single_weight']:.3f}, top3={row['avg_top3_share']:.3f}, "
              f"HHI={row['avg_hhi']:.3f}")
        if label.startswith("W0"):
            w0 = row

    # 判定
    print("\n" + "─" * 60)
    print("  通过判定")
    print("─" * 60)
    for row in rows:
        if row["label"].startswith("W0"):
            print(f"  W0: 等权基线 (对照组)")
            continue
        verdict = judge(row["label"], row, w0)
        print(f"  {row['label']}: {verdict}")

    # 输出
    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "w1_results.csv")
    df_out.to_csv(path, index=False)
    print(f"\n  输出: {path}")

    # 判断是否有任何一组基础通过 → 决定 B3 是否执行
    any_pass = False
    for row in rows:
        if row["label"].startswith("W0"):
            continue
        verdict = judge(row["label"], row, w0)
        if "基础通过" in verdict or "强通过" in verdict:
            any_pass = True

    if any_pass:
        print("\n  ✅ 至少一组基础通过 → W3 (联合仓位) 可以进入")
    else:
        print("\n  ❌ 无组基础通过 → W3 (联合仓位) 不进入")

    return any_pass


if __name__ == "__main__":
    main()