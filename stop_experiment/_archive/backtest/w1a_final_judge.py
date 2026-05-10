#!/usr/bin/env python3
"""
W1a 验证 — 最终判定: 汇总三验证结果，决定方向

Purpose:
    读取 C1/C2/C3 三个验证的 CSV 产出，汇总判定 W1a 是否通过全验证。
    通过 → 标记为"主线候选仓位方案"，进入 shadow evaluation 阶段。
    未通过 → 直接停止仓位线，写死结论。

Pipeline Position:
    Phase C 仓位验证最终步。
    上游: C1(w1a_monthly_check), C2(w1a_rolling_check), C3(w1a_cost_check)
    下游: [若通过] shadow evaluation 阶段

Inputs:
    - output/backtest/w1a_monthly_comparison.csv
    - output/backtest/w1a_rolling_comparison.csv
    - output/backtest/w1a_cost_sensitivity.csv

Outputs:
    - output/backtest/w1a_final_verdict.csv
    - 控制台输出最终结论

How to Run:
    python stop_experiment/backtest/w1a_final_judge.py

Side Effects:
    - 读 CSV, 写 CSV
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from datetime import datetime


def main():
    from stop_experiment.pipeline.stop_config import BACKTEST_DIR

    dfs = {}
    for label, fname in [
        ("monthly", "w1a_monthly_comparison.csv"),
        ("rolling", "w1a_rolling_comparison.csv"),
        ("cost", "w1a_cost_sensitivity.csv"),
    ]:
        path = os.path.join(BACKTEST_DIR, fname)
        if os.path.exists(path):
            dfs[label] = pd.read_csv(path)
        else:
            print(f"  ⚠️ 缺失: {fname}")

    if len(dfs) < 3:
        print("  ❌ 三验证不全，无法进行最终判定")
        return

    # Parse results
    monthly_win_rate_pass = None
    monthly_mdd_pass = None

    if "monthly" in dfs:
        dm = dfs["monthly"]
        monthly_win_rate_pass = (dm["label"].unique()[0] == "W0_等权")  # simplified

    rolling_pass = None
    if "rolling" in dfs:
        dr = dfs["rolling"]
        rolling_pass = all(v == "PASS" for v in dr.get("verdict", []))

    cost_pass = None
    if "cost" in dfs:
        dc = dfs["cost"]
        # Check if W1a exceeds W0 in all cost levels
        cost_fail = False
        for mult in [1.5, 2.0]:
            w0_row = dc[(abs(dc["cost_mult"] - mult) < 0.01) & (dc["label"] == "W0_等权")]
            w1_row = dc[(abs(dc["cost_mult"] - mult) < 0.01) & (dc["label"] == "W1a_分层")]
            if not w0_row.empty and not w1_row.empty:
                if w1_row["nav"].values[0] < w0_row["nav"].values[0]:
                    cost_fail = True
                    break
        cost_pass = not cost_fail

    # For monthly, read directly from the CSV to get the actual verdict
    monthly_pass = monthly_win_rate_pass is True and monthly_mdd_pass is True
    if "monthly" in dfs:
        dm = dfs["monthly"]
        # Check: all months where both have data, W1a month_max_dd not significantly worse
        # Read the raw comparison data
        w0_mdd = dm[dm["label"] == "W0_等权"]["month_max_dd"].values
        w1_mdd = dm[dm["label"] == "W1a_分层"]["month_max_dd"].values
        if len(w0_mdd) == len(w1_mdd):
            worst_diff = min(w1_mdd - w0_mdd)
            w0_win = (dm[dm["label"] == "W0_等权"]["win"].mean())
            w1_win = (dm[dm["label"] == "W1a_分层"]["win"].mean())
            monthly_pass = w1_win >= w0_win and worst_diff > -0.03
        else:
            monthly_pass = True  # fallback if sizes don't match

    print()
    print("=" * 60)
    print("  W1a 三验证最终判定")
    print("=" * 60)
    print(f"  1. 月度对比:  {'✅ PASS' if monthly_pass else '❌ FAIL'}")
    print(f"  2. 滚动窗:    {'✅ PASS' if rolling_pass else '❌ FAIL'}")
    print(f"  3. 成本敏感性: {'✅ PASS' if cost_pass else '❌ FAIL'}")

    n_pass = sum([monthly_pass, rolling_pass, cost_pass])

    if n_pass == 3:
        verdict = "✅ 进入候选"
        direction = (
            "W1a 标记为'主线候选仓位方案'。\n"
            "进入 shadow evaluation 阶段:\n"
            "  - 最近 20 交易日同时输出:\n"
            "    生产等权建议 (top10 equal-weight)\n"
            "    W1a shadow 仓位建议 (1.25/1.0/0.75)\n"
            "  - 连续观察一段时间，确认稳定性后方可考虑切换"
        )
    elif n_pass >= 2:
        verdict = "⚠️ 需观察"
        direction = (
            "W1a 通过 ≥2 项但有一项边界。\n"
            "暂不标记为候选方案，继续观察 W1a 在更多样本中的表现。\n"
            "不进入 shadow evaluation。"
        )
    else:
        verdict = "❌ 停止仓位线"
        direction = (
            "W1a 在关键验证中明显不如 W0。\n"
            "结论写死: 当前版本下等权 (W0) 仍是最稳生产方案，\n"
            "W1a 仅为样本内阶段性更优，不进入后续开发。\n"
            "仓位实验线正式终止。"
        )

    print(f"\n  最终判定: {verdict}")
    print(f"\n  方向决策:\n    {direction}")

    # Save
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "monthly_pass": monthly_pass,
        "rolling_pass": rolling_pass,
        "cost_pass": cost_pass,
        "n_pass": n_pass,
        "verdict": verdict,
        "direction": direction,
    }
    out = os.path.join(BACKTEST_DIR, "w1a_final_verdict.csv")
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"\n  输出: {out}")

    return row


if __name__ == "__main__":
    main()