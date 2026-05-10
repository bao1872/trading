#!/usr/bin/env python3
"""
W1a 验证 — 20/40日滚动窗对比: W0 vs W1a

Purpose:
    验证 W1a 在短窗口上持续优于 W0。
    输出每窗 NAV/Sharpe/MDD 对比。

Pipeline Position:
    Phase C 仓位验证 Step 2。
    上游: w1a_monthly_check.py 的回测结果可复用
    下游: w1a_final_judge.py

Inputs:
    - W0 回测 NAV (来自 run_backtest weight_mode=None)
    - W1a 回测 NAV (来自 run_backtest weight_mode=by_rank)

Outputs:
    - output/backtest/w1a_rolling_comparison.csv

How to Run:
    python stop_experiment/backtest/w1a_rolling_check.py

Side Effects:
    - 运行回测, 写 CSV
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd


def rolling_metrics(nav_series, window):
    nav = nav_series.values
    n = len(nav)
    rows = []
    for i in range(window, n + 1):
        w_nav = nav[i - window:i]
        rets = np.diff(w_nav) / w_nav[:-1]
        if len(rets) < 2:
            continue
        avg_ret = np.mean(rets)
        std_ret = np.std(rets, ddof=1)
        sharpe = avg_ret / std_ret * np.sqrt(252) if std_ret > 1e-12 else 0

        peak = w_nav[0]
        mdd = 0.0
        for v in w_nav:
            peak = max(peak, v)
            dd = (v - peak) / peak
            mdd = min(mdd, dd)

        rows.append({
            "window": window, "end_idx": i - 1,
            "nav": w_nav[-1], "sharpe": sharpe, "mdd": mdd,
        })
    return pd.DataFrame(rows)


def judge(df_merged):
    w0_better = (df_merged["nav_w1a"] >= df_merged["nav_w0"] * 0.99).sum()
    total = len(df_merged)
    pct_better = w0_better / total

    worst_mdd_diff = (df_merged["mdd_w1a"] - df_merged["mdd_w0"]).min()

    result = {
        "total_windows": total,
        "w1a_better_or_even_pct": round(pct_better, 3),
        "pct_pass": pct_better >= 0.60,
        "worst_mdd_diff": round(worst_mdd_diff, 4),
        "verdict": "PASS" if (pct_better >= 0.60) else "FAIL",
    }
    return result


def main():
    from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data, run_backtest
    from stop_experiment.pipeline.stop_config import BASELINE_E0_X1_V1_PARAMS, BACKTEST_DIR

    p = BASELINE_E0_X1_V1_PARAMS
    print("loading data...")
    test_df, price, td, prev, pred = _load_data(candidate_obs_days=p["candidate_obs_days"])
    test_df["trading_date"] = test_df["obs_date"]
    test_df["score"] = test_df[p["score_col"]]
    test_df["sell_score"] = test_df[p["score_col"]]

    # W0
    print("running W0...")
    r0 = run_backtest(test_df, price, td, prev, pred,
                      max_stocks=p["max_stocks"], strategy="sell_score", exit_mode=p["exit_mode"],
                      stop_loss=p["stop_loss"], max_hold_days=p["max_hold_days"],
                      buy_cls_exit_threshold=p["buy_cls_exit_threshold"],
                      exit_sub_mode=p["exit_sub_mode"], buy_reg_exit_threshold=p["buy_reg_exit_threshold"],
                      buy_cost=p["buy_cost"], sell_cost=p["sell_cost"], strict=True,
                      weight_mode=None)

    # W1a
    print("running W1a...")
    wp = {"tiers": [(1, 3, 1.25), (4, 7, 1.0), (8, 10, 0.75)]}
    r1 = run_backtest(test_df, price, td, prev, pred,
                      max_stocks=p["max_stocks"], strategy="sell_score", exit_mode=p["exit_mode"],
                      stop_loss=p["stop_loss"], max_hold_days=p["max_hold_days"],
                      buy_cls_exit_threshold=p["buy_cls_exit_threshold"],
                      exit_sub_mode=p["exit_sub_mode"], buy_reg_exit_threshold=p["buy_reg_exit_threshold"],
                      buy_cost=p["buy_cost"], sell_cost=p["sell_cost"], strict=True,
                      weight_mode="by_rank", weight_params=wp)

    all_results = {}
    for window in [20, 40]:
        print(f"\n  window={window} 日:")
        m0 = rolling_metrics(r0["nav_df"]["nav"], window)
        m1 = rolling_metrics(r1["nav_df"]["nav"], window)

        merged = pd.merge(m0, m1, on=["window", "end_idx"], suffixes=("_w0", "_w1a"))
        result = judge(merged)
        all_results[f"{window}d"] = result

        print(f"    W1a >= W0: {result['w1a_better_or_even_pct']:.0%} (≥60% → {'✅' if result['pct_pass'] else '❌'})")
        print(f"    判定: {result['verdict']}")

    # Save
    rows = []
    for k, v in all_results.items():
        v["window"] = k
        rows.append(v)
    df = pd.DataFrame(rows)
    out = os.path.join(BACKTEST_DIR, "w1a_rolling_comparison.csv")
    df.to_csv(out, index=False)

    print(f"\n  输出: {out}")

    overall = all(v["verdict"] == "PASS" for v in all_results.values())
    print(f"\n  滚动窗总判定: {'PASS' if overall else 'FAIL'}")

    return all_results


if __name__ == "__main__":
    main()