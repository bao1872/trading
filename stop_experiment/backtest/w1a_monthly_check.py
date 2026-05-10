#!/usr/bin/env python3
"""
W1a 验证 — 月度对比: W0 vs W1a 逐月四维指标

Purpose:
    验证 W1a 在月度切面上不弱于 W0。
    输出逐月 NAV/收益/胜率/MDD 对比表。

Pipeline Position:
    Phase C 仓位验证 Step 1。
    上游: position_sizing_w1.py 的 W0/W1a 回测产物
    下游: w1a_final_judge.py

Inputs:
    - output/backtest/w0_nav.csv (W0 等权逐日NAV)
    - output/backtest/w1a_nav.csv (W1a 分层逐日NAV)

Outputs:
    - output/backtest/w1a_monthly_comparison.csv

How to Run:
    python stop_experiment/backtest/w1a_monthly_check.py

Side Effects:
    - 读 CSV, 写 CSV
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd


def load_w0_w1a(w0_dir, w1a_dir):
    df0 = pd.read_csv(os.path.join(w0_dir, "nav.csv"))
    df1 = pd.read_csv(os.path.join(w1a_dir, "nav.csv"))
    return df0, df1


def monthly_metrics(nav_df, label):
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.sort_values("date")
    nav_df["month"] = nav_df["date"].dt.to_period("M")
    nav_df["daily_ret"] = nav_df["nav"].pct_change()

    rows = []
    for month, grp in nav_df.groupby("month"):
        month_navs = grp["nav"].values
        rets = grp["daily_ret"].dropna().values
        month_start_nav = grp["nav"].iloc[0]
        prev_month_end = nav_df[nav_df["date"] < grp["date"].iloc[0]]["nav"].iloc[-1] if len(nav_df[nav_df["date"] < grp["date"].iloc[0]]) > 0 else month_start_nav
        month_return = (grp["nav"].iloc[-1] - month_start_nav) / month_start_nav

        peak = month_navs[0]
        mdd = 0.0
        for v in month_navs:
            peak = max(peak, v)
            dd = (v - peak) / peak
            mdd = min(mdd, dd)

        rows.append({
            "label": label, "month": str(month),
            "nav_end": grp["nav"].iloc[-1],
            "month_ret": month_return,
            "month_max_dd": mdd,
            "n_days": len(grp),
            "win": 1 if month_return > 0 else 0,
        })
    return pd.DataFrame(rows)


def judge(df_merged):
    w0 = df_merged[df_merged["label"] == "W0_等权"]
    w1 = df_merged[df_merged["label"] == "W1a_分层"]

    w0_win_rate = w0["win"].mean()
    w1_win_rate = w1["win"].mean()

    merged = pd.merge(w0, w1, on="month", suffixes=("_w0", "_w1a"))
    worst_diff = (merged["month_max_dd_w1a"] - merged["month_max_dd_w0"]).min()

    result = {
        "w0_win_rate": round(w0_win_rate, 3),
        "w1a_win_rate": round(w1_win_rate, 3),
        "win_rate_pass": w1_win_rate >= w0_win_rate,
        "worst_mdd_diff": round(worst_diff, 4),
        "mdd_pass": worst_diff > -0.03,
        "verdict": "PASS" if (w1_win_rate >= w0_win_rate and worst_diff > -0.03) else "FAIL",
    }
    return result


def main():
    from stop_experiment.pipeline.stop_config import BACKTEST_DIR

    w0_p = os.path.join(BACKTEST_DIR, "w0")
    w1a_p = os.path.join(BACKTEST_DIR, "w1a")
    if not os.path.exists(w0_p):
        w0_p = BACKTEST_DIR  # fallback to shared baseline

    # For this implementation, we use the run_backtest to re-extract monthly data
    from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data, run_backtest, compute_summary
    from stop_experiment.pipeline.stop_config import BASELINE_E0_X1_V1_PARAMS

    p = BASELINE_E0_X1_V1_PARAMS
    print("loading data...")
    test_df, price, td, prev, pred = _load_data(candidate_obs_days=p["candidate_obs_days"])
    test_df["trading_date"] = test_df["obs_date"]
    test_df["score"] = test_df[p["score_col"]]
    test_df["sell_score"] = test_df[p["score_col"]]

    # W0: 等权
    print("running W0 (equal-weight)...")
    r0 = run_backtest(test_df, price, td, prev, pred,
                      max_stocks=p["max_stocks"], strategy="sell_score", exit_mode=p["exit_mode"],
                      stop_loss=p["stop_loss"], max_hold_days=p["max_hold_days"],
                      buy_cls_exit_threshold=p["buy_cls_exit_threshold"],
                      exit_sub_mode=p["exit_sub_mode"], buy_reg_exit_threshold=p["buy_reg_exit_threshold"],
                      buy_cost=p["buy_cost"], sell_cost=p["sell_cost"], strict=True,
                      weight_mode=None, weight_params=None)
    s0 = compute_summary(r0)

    # W1a: 排名分层 1.25/1.0/0.75
    print("running W1a (rank-tier 1.25/1.0/0.75)...")
    weight_params = {"tiers": [(1, 3, 1.25), (4, 7, 1.0), (8, 10, 0.75)]}
    r1 = run_backtest(test_df, price, td, prev, pred,
                      max_stocks=p["max_stocks"], strategy="sell_score", exit_mode=p["exit_mode"],
                      stop_loss=p["stop_loss"], max_hold_days=p["max_hold_days"],
                      buy_cls_exit_threshold=p["buy_cls_exit_threshold"],
                      exit_sub_mode=p["exit_sub_mode"], buy_reg_exit_threshold=p["buy_reg_exit_threshold"],
                      buy_cost=p["buy_cost"], sell_cost=p["sell_cost"], strict=True,
                      weight_mode="by_rank", weight_params=weight_params)
    s1 = compute_summary(r1)

    # Build monthly comparison
    df0 = r0["nav_df"].copy()
    df0["label"] = "W0_等权"
    df1 = r1["nav_df"].copy()
    df1["label"] = "W1a_分层"

    m0 = monthly_metrics(df0.rename(columns={"date": "date", "nav": "nav"}) if "date" in df0.columns else df0.rename(columns={"trading_date": "date"}), "W0_等权")
    m1 = monthly_metrics(df1.rename(columns={"date": "date", "nav": "nav"}) if "date" in df1.columns else df1.rename(columns={"trading_date": "date"}), "W1a_分层")

    all_m = pd.concat([m0, m1], ignore_index=True)

    # Judge
    result = judge(all_m)

    # Output
    print()
    print("=" * 60)
    print(f"  W1a 月度对比结果: {result['verdict']}")
    print("=" * 60)
    print(f"  W0 月胜率:   {result['w0_win_rate']:.1%}")
    print(f"  W1a 月胜率:  {result['w1a_win_rate']:.1%}")
    print(f"  月胜率通过:  {'✅' if result['win_rate_pass'] else '❌'}")
    print(f"  最差月MDD差: {result['worst_mdd_diff']:.4f} (阈值 > -0.03)")
    print(f"  MDD通过:     {'✅' if result['mdd_pass'] else '❌'}")

    print()
    print("  逐月明细:")
    pivot = all_m.pivot(index="month", columns="label", values=["nav_end", "month_ret", "month_max_dd"])
    print(pivot.to_string())

    out = os.path.join(BACKTEST_DIR, "w1a_monthly_comparison.csv")
    all_m.to_csv(out, index=False)
    print(f"\n  输出: {out}")

    return result


if __name__ == "__main__":
    main()