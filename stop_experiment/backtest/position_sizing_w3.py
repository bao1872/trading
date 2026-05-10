#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓位实验 W3: sell_reg + buy_cls 联合仓位
    [已关闭] 2026-05-10: 联合仓位线已关闭。W1a 是唯一保留方向。

Purpose:
    在固定 E0+X1+0.70 基线上，测试 sell_reg + buy_cls 联合仓位分配：
    W3a: size_score = rank(pred_sell_reg) - 0.5 * rank(pred_buy_cls)
    W3b: size_score = rank(pred_sell_reg) - 1.0 * rank(pred_buy_cls)
    将 size_score 线性映射为权重

    rank 定义:
    - rank_sell_reg: groupby obs_date, ascending=False (最高=1)
    - rank_buy_cls:  groupby obs_date, ascending=True (最低风险=1)
    - size_score = rank_sell_reg - λ * rank_buy_cls (值越大越好)

Pipeline Position:
    Part B 仓位实验 Phase W3 (仅当 W1/W2 至少一组基础通过时执行)。
    上游：B0 + B1/B2
    下游：B4 (position_sizing_summary)

Inputs:
    - full_test_predictions.parquet

Outputs:
    - output/backtest/w3_results.csv

How to Run:
    python stop_experiment/backtest/position_sizing_w3.py

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


def run_w3_variant(label, lam):
    """运行单个 W3 变体: size_score = rank_sell - lam * rank_buy"""
    p = BASELINE_E0_X1_V1_PARAMS
    test_df, price, td, prev, pred = _load_data(candidate_obs_days=p["candidate_obs_days"])

    test_df["trading_date"] = test_df["obs_date"]
    test_df["sell_score"] = test_df[p["score_col"]]

    # 计算联合仓位得分（存入 weight_score 列，不覆盖 score）
    test_df["rank_sell"] = test_df.groupby("obs_date")["pred_sell_reg"].rank(ascending=False)
    test_df["rank_buy"] = test_df.groupby("obs_date")["pred_buy_cls"].rank(ascending=True)
    test_df["size_score"] = test_df["rank_sell"] - lam * test_df["rank_buy"]

    test_df["weight_score"] = test_df["size_score"]

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
        weight_mode="by_score",
        weight_params={"lo": 0.50, "hi": 1.50},
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
        "lam": lam,
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


def judge(label, row, w_row, w1a_row):
    nav = row["nav"]
    sharpe = row["sharpe"]
    mdd = row["mdd"]
    mwr = row["monthly_win_rate"]

    # 对 W3 用 W0 判定 + 额外看 vs W1a
    if nav < w_row["nav"]:
        return "❌ 否决: NAV < W0"
    if mdd < w_row["mdd"] - 0.02:
        return "❌ 否决: MDD 恶化 > 2pp vs W0"
    if not np.isnan(mwr) and not np.isnan(w_row["monthly_win_rate"]) and mwr < w_row["monthly_win_rate"]:
        return "❌ 否决: 月度胜率 < W0"

    basic = (
        nav >= w_row["nav"]
        and sharpe > w_row["sharpe"]
        and mdd >= w_row["mdd"] - 0.02
    )

    if not basic:
        vs_w1 = ""
        if w1a_row is not None:
            vs_w1 = f" (vs W1a: {'超越' if nav > w1a_row['nav'] else '弱于'} W1a NAV={w1a_row['nav']:.4f})"
        return f"⚠️ 未通过{vs_w1}"

    # 额外: vs W1a
    vs_w1a = ""
    if w1a_row is not None:
        if nav > w1a_row["nav"] and mdd >= w1a_row["mdd"]:
            vs_w1a = ", 同时超越 W1a"
        elif nav > w1a_row["nav"]:
            vs_w1a = ", NAV超W1a但MDD更差"

    strength = nav > w_row["nav"] * 1.05 and mdd >= w_row["mdd"]
    if strength:
        return f"🏆 强通过{vs_w1a}"
    return f"✅ 基础通过{vs_w1a}"


def main():
    p = BASELINE_E0_X1_V1_PARAMS
    print("=" * 60)
    print(f"  仓位实验 W3: sell_reg + buy_cls 联合仓位")
    print(f"  固定基线: E0+X1+{p['buy_cls_exit_threshold']}, obs_day={p['candidate_obs_days']}")
    print("=" * 60)

    # 检查 W1/W2 是否有基础通过
    w1_path = os.path.join(BACKTEST_DIR, "w1_results.csv")
    w2_path = os.path.join(BACKTEST_DIR, "w2_results.csv")

    w1_df = pd.read_csv(w1_path) if os.path.exists(w1_path) else None
    w2_df = pd.read_csv(w2_path) if os.path.exists(w2_path) else None

    w0_row = w1_df[w1_df["label"] == "W0 (等权基线)"].iloc[0] if w1_df is not None else None
    w1a_row = w1_df[w1_df["label"] == "W1a (1.25/1.0/0.75)"].iloc[0] if w1_df is not None else None

    print(f"\n  W0 基线: NAV={w0_row['nav']:.4f}, Sharpe={w0_row['sharpe']:.2f}, "
          f"MDD={w0_row['mdd']:.4f}")
    print(f"  W1a 最佳: NAV={w1a_row['nav']:.4f}, Sharpe={w1a_row['sharpe']:.2f}, "
          f"MDD={w1a_row['mdd']:.4f}")
    print()

    variants = [
        ("W3a (λ=0.5)", 0.5),
        ("W3b (λ=1.0)", 1.0),
    ]

    rows = []
    for label, lam in variants:
        print(f"  运行: {label} (size_score = rank_sell - {lam} * rank_buy) ...")
        row = run_w3_variant(label, lam)
        rows.append(row)
        print(f"    NAV={row['nav']:.4f}, Sharpe={row['sharpe']:.2f}, MDD={row['mdd']:.4f}, "
              f"n_trades={row['n_trades']}, 月胜率={row['monthly_win_rate']:.1%}")
        print(f"    集中度: max_w={row['max_single_weight']:.3f}, top3={row['avg_top3_share']:.3f}, "
              f"HHI={row['avg_hhi']:.3f}")

    # 判定
    print("\n" + "─" * 60)
    print("  通过判定")
    print("─" * 60)
    for row in rows:
        verdict = judge(row["label"], row, w0_row, w1a_row)
        print(f"  {row['label']}: {verdict}")

    # buy_cls risk penalty 分析
    print("\n" + "─" * 60)
    print("  buy_cls risk penalty 效果分析")
    print("─" * 60)
    for row in rows:
        vs_w0_mdd = row["mdd"] - w0_row["mdd"]
        vs_w0_nav = (row["nav"] - w0_row["nav"]) / w0_row["nav"] * 100
        vs_w1a_nav = (row["nav"] - w1a_row["nav"]) / w1a_row["nav"] * 100 if w1a_row is not None else 0
        print(f"  {row['label']}: ΔNAV={vs_w0_nav:+.1f}% vs W0, ΔMDD={vs_w0_mdd:+.4f}, "
              f"vs W1a ΔNAV={vs_w1a_nav:+.1f}%")

    # 结论
    print("\n  buy_cls 作为 risk penalty:")
    any_mdd_improve = any(r["mdd"] >= w0_row["mdd"] for r in rows)
    any_nav_compare = any(r["nav"] >= w1a_row["nav"] * 0.9 for r in rows) if w1a_row is not None else False

    if any_mdd_improve:
        print("  → buy_cls 作为权重 penalty 可以改善 MDD")
    else:
        print("  → buy_cls 作为权重 penalty 未改善 MDD")

    if any_nav_compare and w1a_row is not None:
        print(f"  → W3 vs W1a: 收益变化幅度不足以证明联合仓位优于纯 sell_reg 分层")

    # 输出
    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "w3_results.csv")
    df_out.to_csv(path, index=False)
    print(f"\n  输出: {path}")


if __name__ == "__main__":
    main()