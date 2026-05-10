#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓位实验 W2: sell_reg 分段仓位
    [已关闭] 2026-05-10: by_score 线性映射线已关闭。
    by_rank 排名分层 (W1a) 是唯一保留的仓位方向。

Purpose:
    在固定 E0+X1+0.70 基线上，测试 sell_reg 分段映射仓位：
    W2a: top10内按 pred_sell_reg 分位分三档 (上1/3=1.25, 中1/3=1.0, 下1/3=0.75)
    W2b-1: 线性映射到 [0.75, 1.25] (轻微倾斜)
    W2b-2: 线性映射到 [0.50, 1.50] (明显拉开)

Pipeline Position:
    Part B 仓位实验 Phase W2。
    上游：B0 (run_backtest 支持 weight_mode)
    下游：B4 (position_sizing_summary)

Inputs:
    - full_test_predictions.parquet

Outputs:
    - output/backtest/w2_results.csv

How to Run:
    python stop_experiment/backtest/position_sizing_w2.py

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


def monthly_win_rate(nav_df):
    nav_df = nav_df.copy()
    nav_df.index = pd.to_datetime(nav_df.index)
    monthly = nav_df["nav"].resample("ME").last()
    monthly = monthly.dropna()
    if len(monthly) == 0:
        return np.nan, [], []
    win = (monthly > 1.0).sum()
    return win / len(monthly), monthly.tolist(), list(monthly.index)


def compute_concentration(snapshots):
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


def run_w2_variant(label, weight_mode, weight_params):
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

    mwr, monthly_navs, _ = monthly_win_rate(result["nav_df"])
    conc = compute_concentration(result["snapshots"])

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
    nav = row["nav"]
    sharpe = row["sharpe"]
    mdd = row["mdd"]
    mwr = row["monthly_win_rate"]

    if nav < w0["nav"]:
        return "❌ 否决: NAV < W0"
    if mdd < w0["mdd"] - 0.02:
        return "❌ 否决: MDD 恶化 > 2pp"
    if not np.isnan(mwr) and not np.isnan(w0["monthly_win_rate"]) and mwr < w0["monthly_win_rate"]:
        return "❌ 否决: 月度胜率 < W0"

    basic = (
        nav >= w0["nav"]
        and sharpe > w0["sharpe"]
        and mdd >= w0["mdd"] - 0.02
        and (np.isnan(mwr) or np.isnan(w0["monthly_win_rate"]) or mwr >= w0["monthly_win_rate"])
    )

    if not basic:
        return "⚠️ 未通过: 不满足基础条件"

    strength = nav > w0["nav"] * 1.05 and mdd >= w0["mdd"]
    if strength:
        return "🏆 强通过"
    return "✅ 基础通过"


def main():
    p = BASELINE_E0_X1_V1_PARAMS
    print("=" * 60)
    print(f"  仓位实验 W2: sell_reg 分段仓位")
    print(f"  固定基线: E0+X1+{p['buy_cls_exit_threshold']}, obs_day={p['candidate_obs_days']}")
    print("=" * 60)

    # W2a: 分位三档 — by_rank tiers dynamically computed for n=10: top3=1.25, mid3=1.0, bottom4=0.75
    # (same as W1a for n=10, but conceptually different: quantile-based vs fixed-rank)
    variants = [
        ("W2a (分位 1.25/1.0/0.75)", "by_rank",
         {"tiers": [(1, 3, 1.25), (4, 6, 1.0), (7, 10, 0.75)]}),
        ("W2b-1 [0.75,1.25]", "by_score",
         {"lo": 0.75, "hi": 1.25}),
        ("W2b-2 [0.50,1.50]", "by_score",
         {"lo": 0.50, "hi": 1.50}),
    ]

    # 加载 W0 基线作为对照组
    w0_path = os.path.join(BACKTEST_DIR, "w1_results.csv")
    w0_row = None
    if os.path.exists(w0_path):
        w1_df = pd.read_csv(w0_path)
        w0 = w1_df[w1_df["label"] == "W0 (等权基线)"]
        if len(w0) > 0:
            w0_row = w0.iloc[0]
            print(f"\n  W0 基线: NAV={w0_row['nav']:.4f}, Sharpe={w0_row['sharpe']:.2f}, "
                  f"MDD={w0_row['mdd']:.4f}")

    # Fallback: run W0 if not cached
    if w0_row is None:
        print("  运行 W0 基线 ...")
        w0_row = run_w2_variant("W0 (等权基线)", None, None)
        print(f"    NAV={w0_row['nav']:.4f}, Sharpe={w0_row['sharpe']:.2f}, "
              f"MDD={w0_row['mdd']:.4f}")

    rows = []
    for label, wm, wp in variants:
        print(f"\n  运行: {label} ...")
        row = run_w2_variant(label, wm, wp)
        rows.append(row)
        print(f"    NAV={row['nav']:.4f}, Sharpe={row['sharpe']:.2f}, MDD={row['mdd']:.4f}, "
              f"n_trades={row['n_trades']}, 月胜率={row['monthly_win_rate']:.1%}")
        print(f"    集中度: max_w={row['max_single_weight']:.3f}, top3={row['avg_top3_share']:.3f}, "
              f"HHI={row['avg_hhi']:.3f}")

    # 判定
    print("\n" + "─" * 60)
    print("  通过判定")
    print("─" * 60)
    print(f"  W2a vs W1a: {'完全相同' if abs(rows[0]['nav'] - 5.6474) < 1e-4 else '差异 {}'.format(rows[0]['nav'])}")
    print()

    for row in rows:
        verdict = judge(row["label"], row, w0_row)
        print(f"  {row['label']}: {verdict}")

    # 对比：轻微 vs 明显倾斜
    if len(rows) >= 3:
        r_b1 = rows[1]  # W2b-1
        r_b2 = rows[2]  # W2b-2
        print(f"\n  W2b-1 (轻微倾斜): NAV={r_b1['nav']:.4f}, MDD={r_b1['mdd']:.4f}")
        print(f"  W2b-2 (明显倾斜): NAV={r_b2['nav']:.4f}, MDD={r_b2['mdd']:.4f}")
        if r_b1["nav"] >= w0_row["nav"] * 1.01:
            print("  → 轻微倾斜已经有效")
        if r_b2["nav"] > r_b1["nav"]:
            print("  → 明显倾斜进一步提升 NAV")
        else:
            print("  → 轻微/明显倾斜差异不大")

    # 输出
    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "w2_results.csv")
    df_out.to_csv(path, index=False)
    print(f"\n  输出: {path}")


if __name__ == "__main__":
    main()