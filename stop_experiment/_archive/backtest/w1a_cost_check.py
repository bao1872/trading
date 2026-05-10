#!/usr/bin/env python3
"""
W1a 验证 — 成本敏感性: W0 vs W1a 在 1.0x/1.5x/2.0x 成本下对比

Purpose:
    验证 W1a 在更高交易成本下仍不弱于 W0。
    对 W0 和 W1a 分别用 3 档成本重跑回测，对比 NAV/Sharpe/MDD。

Pipeline Position:
    Phase C 仓位验证 Step 3。
    上游: w1a_monthly_check.py, w1a_rolling_check.py
    下游: w1a_final_judge.py

Outputs:
    - output/backtest/w1a_cost_sensitivity.csv

How to Run:
    python stop_experiment/backtest/w1a_cost_check.py

Side Effects:
    - 运行 6 次回测 (2方案×3档成本), 写 CSV
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd


def main():
    from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data, run_backtest, compute_summary
    from stop_experiment.pipeline.stop_config import BASELINE_E0_X1_V1_PARAMS, BACKTEST_DIR

    p = BASELINE_E0_X1_V1_PARAMS
    base_buy, base_sell = p["buy_cost"], p["sell_cost"]

    print("loading data...")
    test_df, price, td, prev, pred = _load_data(candidate_obs_days=p["candidate_obs_days"])
    test_df["trading_date"] = test_df["obs_date"]
    test_df["score"] = test_df[p["score_col"]]
    test_df["sell_score"] = test_df[p["score_col"]]

    cost_mult = [1.0, 1.5, 2.0]
    wp = {"tiers": [(1, 3, 1.25), (4, 7, 1.0), (8, 10, 0.75)]}

    rows = []
    for mult in cost_mult:
        bc = base_buy * mult
        sc = base_sell * mult
        print(f"\n  ── 成本 ×{mult:.1f} (buy={bc:.4f}, sell={sc:.4f}) ──")

        # W0
        r0 = run_backtest(test_df, price, td, prev, pred,
                          max_stocks=p["max_stocks"], strategy="sell_score", exit_mode=p["exit_mode"],
                          stop_loss=p["stop_loss"], max_hold_days=p["max_hold_days"],
                          buy_cls_exit_threshold=p["buy_cls_exit_threshold"],
                          exit_sub_mode=p["exit_sub_mode"],
                          buy_reg_exit_threshold=p["buy_reg_exit_threshold"],
                          buy_cost=bc, sell_cost=sc, strict=True,
                          weight_mode=None)
        s0 = compute_summary(r0)

        # W1a
        r1 = run_backtest(test_df, price, td, prev, pred,
                          max_stocks=p["max_stocks"], strategy="sell_score", exit_mode=p["exit_mode"],
                          stop_loss=p["stop_loss"], max_hold_days=p["max_hold_days"],
                          buy_cls_exit_threshold=p["buy_cls_exit_threshold"],
                          exit_sub_mode=p["exit_sub_mode"],
                          buy_reg_exit_threshold=p["buy_reg_exit_threshold"],
                          buy_cost=bc, sell_cost=sc, strict=True,
                          weight_mode="by_rank", weight_params=wp)
        s1 = compute_summary(r1)

        rows.append({"cost_mult": mult, "label": "W0_等权",
                     "nav": s0["final_nav"], "sharpe": s0["sharpe"], "mdd": s0["max_dd"]})
        rows.append({"cost_mult": mult, "label": "W1a_分层",
                     "nav": s1["final_nav"], "sharpe": s1["sharpe"], "mdd": s1["max_dd"]})

        print(f"    W0:  NAV={s0['final_nav']:.4f}, Sharpe={s0['sharpe']:.2f}, MDD={s0['max_dd']:.4f}")
        print(f"    W1a: NAV={s1['final_nav']:.4f}, Sharpe={s1['sharpe']:.2f}, MDD={s1['max_dd']:.4f}")

    df = pd.DataFrame(rows)

    # Judge
    print("\n  ── 判定 ──")
    verdicts = []
    for mult in [1.5, 2.0]:
        w0_ = df[(df["cost_mult"] == mult) & (df["label"] == "W0_等权")]
        w1_ = df[(df["cost_mult"] == mult) & (df["label"] == "W1a_分层")]
        if not w0_.empty and not w1_.empty:
            w0_nav = w0_["nav"].values[0]
            w1_nav = w1_["nav"].values[0]
            ok = w1_nav >= w0_nav
            print(f"    ×{mult:.1f}: W1a NAV={w1_nav:.4f} ≥ W0 NAV={w0_nav:.4f} → {'✅ PASS' if ok else '❌ FAIL'}")
            verdicts.append(ok)

    overall = all(verdicts)
    print(f"\n  成本敏感性总判定: {'PASS' if overall else 'FAIL'}")

    out = os.path.join(BACKTEST_DIR, "w1a_cost_sensitivity.csv")
    df.to_csv(out, index=False)
    print(f"  输出: {out}")

    return overall


if __name__ == "__main__":
    main()