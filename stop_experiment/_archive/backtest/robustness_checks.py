#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三类局部稳健性测试: 阈值 / 交易成本 / 容量

Purpose:
    在 frozen V1 基线上执行三类局部稳健性验证，确认模型退出 (M > F) 不是单点最优偶然现象。

    验证维度:
    - thresholds: buy_cls 阈值扫描（0.6~0.8）
    - cost: 交易成本变化（0.05%~0.20%）
    - capacity: 持仓容量变化（5~15只）

Pipeline Position:
    诊断工具（按需运行）。
    上游: dynamic_exit_backtest_v2.py
    下游: —

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - DB: stock_k_data

Outputs:
    - stop_experiment/output/backtest/robustness_thresholds.csv
    - stop_experiment/output/backtest/robustness_cost.csv
    - stop_experiment/output/backtest/robustness_capacity.csv

How to Run:
    python stop_experiment/backtest/robustness_checks.py --mode all
    python stop_experiment/backtest/robustness_checks.py --mode thresholds
    python stop_experiment/backtest/robustness_checks.py --mode cost
    python stop_experiment/backtest/robustness_checks.py --mode capacity

Side Effects:
    - 只读DB和parquet，输出csv
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, BACKTEST_DIR, BUY_COST, SELL_COST,
    V1_PARAMS,
)
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest, compute_summary,
)

DYNAMIC_DIR = os.path.join(BACKTEST_DIR, "dynamic")


def run_threshold_robustness(args):
    """3a: 阈值局部稳健性 (buy_cls_exit_threshold × stop_loss)"""
    print("=" * 60)
    print("阈值局部稳健性")
    print("=" * 60)

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    stops = [-0.05, -0.07, -0.09]

    test_df, price, td, prev, pred = _load_data(
        candidate_obs_days=V1_PARAMS["candidate_obs_days"]
    )

    rows = []
    for th in thresholds:
        for sl in stops:
            print(f"  th={th}, sl={sl:.2f}")
            result = run_backtest(
                test_df, price, td, prev, pred,
                max_stocks=args.top_k, strategy=args.strategy,
                exit_mode="model_exit", stop_loss=sl,
                buy_cls_exit_threshold=th,
                strict=True,
            )
            s = compute_summary(result)
            rows.append({
                "threshold": th,
                "stop_loss": sl,
                "final_nav": s.get("final_nav", np.nan),
                "max_dd": s.get("max_dd", np.nan),
                "sharpe": s.get("sharpe", np.nan),
                "win_rate": s.get("win_rate", np.nan),
                "n_trades": s.get("n_trades", 0),
            })

    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "robustness_thresholds.csv")
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    df_out.to_csv(path, index=False)

    print(f"\n  输出: {path}")
    print(f"\n  NAV 矩阵 (threshold × stop_loss):")
    pivot = df_out.pivot_table(values="final_nav", index="stop_loss", columns="threshold")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    nav_values = df_out["final_nav"].values
    print(f"\n  NAV: min={nav_values.min():.4f}, max={nav_values.max():.4f}, "
          f"range={nav_values.max()-nav_values.min():.4f}")
    if nav_values.max() - nav_values.min() < 0.3:
        print(f"  ✅ 结论: 局部稳健，NAV 波动 < 0.3")
    else:
        print(f"  ⚠️ 结论: 对阈值敏感，需进一步验证")


def run_cost_sensitivity(args):
    """3b: 交易成本敏感性 (1.0× / 1.5× / 2.0×)"""
    print("=" * 60)
    print("交易成本敏感性")
    print("=" * 60)

    test_df, price, td, prev, pred = _load_data(
        candidate_obs_days=V1_PARAMS["candidate_obs_days"]
    )

    rows = []
    for cost_mult in [1.0, 1.5, 2.0]:
        bc = BUY_COST * cost_mult
        sc = SELL_COST * cost_mult
        print(f"\n  cost={cost_mult:.1f}× (buy={bc:.3%}, sell={sc:.3%})")

        for exit_mode in ["fixed_hold", "model_exit"]:
            result = run_backtest(
                test_df, price, td, prev, pred,
                max_stocks=args.top_k, strategy=args.strategy,
                exit_mode=exit_mode, hold_days=5,
                buy_cost=bc, sell_cost=sc,
                stop_loss=V1_PARAMS["stop_loss"],
                buy_cls_exit_threshold=V1_PARAMS["buy_cls_exit_threshold"],
                strict=True,
            )
            s = compute_summary(result)
            rows.append({
                "cost_mult": cost_mult,
                "buy_cost": bc,
                "sell_cost": sc,
                "exit_mode": exit_mode,
                "final_nav": s.get("final_nav", np.nan),
                "max_dd": s.get("max_dd", np.nan),
                "sharpe": s.get("sharpe", np.nan),
                "win_rate": s.get("win_rate", np.nan),
                "n_trades": s.get("n_trades", 0),
            })
            label = "M" if exit_mode == "model_exit" else "F"
            print(f"    {label}: NAV={s.get('final_nav', 0):.4f} "
                  f"win={s.get('win_rate', 0):.2%} sharpe={s.get('sharpe', 0):.2f}")

    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "robustness_cost.csv")
    df_out.to_csv(path, index=False)

    print(f"\n  输出: {path}")
    print(f"\n  M-F 差值随成本变化:")
    for cm in [1.0, 1.5, 2.0]:
        f_nav = df_out[(df_out["cost_mult"] == cm) & (df_out["exit_mode"] == "fixed_hold")]["final_nav"].values
        m_nav = df_out[(df_out["cost_mult"] == cm) & (df_out["exit_mode"] == "model_exit")]["final_nav"].values
        if len(f_nav) and len(m_nav):
            diff = m_nav[0] - f_nav[0]
            status = "✅" if diff > 0 else "⚠️"
            print(f"    {cm:.1f}×: M-F = {diff:+.4f} {status}")

    # 结论
    all_mf_positive = all(
        (df_out[(df_out["cost_mult"] == cm) & (df_out["exit_mode"] == "model_exit")]["final_nav"].values[0] -
         df_out[(df_out["cost_mult"] == cm) & (df_out["exit_mode"] == "fixed_hold")]["final_nav"].values[0]) > 0
        for cm in [1.0, 1.5, 2.0]
    )
    if all_mf_positive:
        print(f"\n  ✅ 结论: 成本 1.0×~2.0× 范围内 M > F 持续成立")
    else:
        print(f"\n  ⚠️ 结论: 高成本下模型退出优势减弱或消失")


def run_buy_cls_calibration(args):
    """buy_cls 阈值专项重标定：固定 stop_loss=-0.07，纯扫 buy_cls 阈值"""
    print("=" * 60)
    print("buy_cls 阈值专项重标定")
    print("=" * 60)

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    fixed_stop = -0.07

    test_df, price, td, prev, pred = _load_data(
        candidate_obs_days=V1_PARAMS["candidate_obs_days"]
    )

    rows = []
    for th in thresholds:
        print(f"  buy_cls_threshold={th}, stop_loss={fixed_stop:.2f}")
        result = run_backtest(
            test_df, price, td, prev, pred,
            max_stocks=args.top_k, strategy=args.strategy,
            exit_mode="model_exit", stop_loss=fixed_stop,
            buy_cls_exit_threshold=th,
            strict=True,
        )
        s = compute_summary(result)
        rows.append({
            "buy_cls_threshold": th,
            "stop_loss": fixed_stop,
            "final_nav": s.get("final_nav", np.nan),
            "max_dd": s.get("max_dd", np.nan),
            "sharpe": s.get("sharpe", np.nan),
            "win_rate": s.get("win_rate", np.nan),
            "n_trades": s.get("n_trades", 0),
            "avg_hold_days": s.get("avg_hold_days", np.nan),
        })

    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "buy_cls_calibration.csv")
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    df_out.to_csv(path, index=False)

    print(f"\n  输出: {path}")
    print(f"\n  buy_cls 阈值扫描结果:")
    for _, row in df_out.iterrows():
        print(f"    th={row['buy_cls_threshold']:.2f}: "
              f"NAV={row['final_nav']:.4f}, Sharpe={row['sharpe']:.2f}, "
              f"MDD={row['max_dd']:.4f}, win={row['win_rate']:.2%}, "
              f"avg_hold={row['avg_hold_days']:.1f}d")

    print(f"\n  NAV: min={df_out['final_nav'].min():.4f}, max={df_out['final_nav'].max():.4f}, "
          f"range={df_out['final_nav'].max()-df_out['final_nav'].min():.4f}")

    best = df_out.loc[df_out["final_nav"].idxmax()]
    print(f"  ✅ 最优 buy_cls 阈值: {best['buy_cls_threshold']:.2f} (NAV={best['final_nav']:.4f})")


def run_capacity_sensitivity(args):
    """3c: 容量敏感性 (top_k=5/10/20)"""
    print("=" * 60)
    print("容量敏感性 (top_k)")
    print("=" * 60)

    test_df, price, td, prev, pred = _load_data(
        candidate_obs_days=V1_PARAMS["candidate_obs_days"]
    )

    rows = []
    for top_k in [5, 10, 20]:
        print(f"\n  top_k={top_k}")
        for exit_mode in ["fixed_hold", "model_exit"]:
            result = run_backtest(
                test_df, price, td, prev, pred,
                max_stocks=top_k, strategy=args.strategy,
                exit_mode=exit_mode, hold_days=5,
                stop_loss=V1_PARAMS["stop_loss"],
                buy_cls_exit_threshold=V1_PARAMS["buy_cls_exit_threshold"],
                strict=True,
            )
            s = compute_summary(result)
            rows.append({
                "top_k": top_k,
                "exit_mode": exit_mode,
                "final_nav": s.get("final_nav", np.nan),
                "max_dd": s.get("max_dd", np.nan),
                "sharpe": s.get("sharpe", np.nan),
                "win_rate": s.get("win_rate", np.nan),
                "n_trades": s.get("n_trades", 0),
            })
            label = "M" if exit_mode == "model_exit" else "F"
            print(f"    {label}: NAV={s.get('final_nav', 0):.4f} "
                  f"win={s.get('win_rate', 0):.2%} sharpe={s.get('sharpe', 0):.2f}")

    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "robustness_capacity.csv")
    df_out.to_csv(path, index=False)

    print(f"\n  输出: {path}")
    print(f"\n  M-F 差值随容量变化:")
    for tk in [5, 10, 20]:
        f_nav = df_out[(df_out["top_k"] == tk) & (df_out["exit_mode"] == "fixed_hold")]["final_nav"].values
        m_nav = df_out[(df_out["top_k"] == tk) & (df_out["exit_mode"] == "model_exit")]["final_nav"].values
        if len(f_nav) and len(m_nav):
            diff = m_nav[0] - f_nav[0]
            status = "✅" if diff > 0 else "⚠️"
            print(f"    k={tk:2d}: M-F = {diff:+.4f} {status}")

    diff_5 = df_out[(df_out["top_k"] == 5) & (df_out["exit_mode"] == "model_exit")]["final_nav"].values[0] - \
             df_out[(df_out["top_k"] == 5) & (df_out["exit_mode"] == "fixed_hold")]["final_nav"].values[0]
    diff_20 = df_out[(df_out["top_k"] == 20) & (df_out["exit_mode"] == "model_exit")]["final_nav"].values[0] - \
              df_out[(df_out["top_k"] == 20) & (df_out["exit_mode"] == "fixed_hold")]["final_nav"].values[0]

    if diff_5 > 0 and diff_20 > 0:
        print(f"\n  ✅ 结论: k=5~20 范围内 M > F 持续成立，容量不敏感")
    elif diff_5 > 0 and diff_20 < 0:
        print(f"\n  ⚠️ 结论: 模型退出优势随容量扩大而消失 (k5={diff_5:+.4f}, k20={diff_20:+.4f})")
    else:
        print(f"\n  ⚠️ 结论: 需进一步验证")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="三类局部稳健性测试")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "thresholds", "cost", "capacity", "buy_cls_calibration"])
    parser.add_argument("--top-k", type=int, default=10, help="最大持仓数")
    parser.add_argument("--strategy", type=str, default="sell_score",
                        choices=["sell_score", "low_risk", "composite"])
    args = parser.parse_args()

    if args.mode in ("all", "thresholds"):
        run_threshold_robustness(args)
        print()

    if args.mode in ("all", "cost"):
        run_cost_sensitivity(args)
        print()

    if args.mode in ("all", "capacity"):
        run_capacity_sensitivity(args)
        print()

    if args.mode in ("all", "buy_cls_calibration"):
        run_buy_cls_calibration(args)
