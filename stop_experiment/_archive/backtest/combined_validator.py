#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组合验证实验：对最优策略 E0+X1(th=0.70) 进行稳健性检验

Purpose:
    固定最优结构 (E0 Entry + X1 Exit, buy_cls_exit_threshold=0.70)，
    进行 4 组稳健性检验：
    1. Slice 分片分析：将回测区间等分为 3 段，每段独立回测
    2. Capacity 容量测试：k ∈ [5, 8, 10, 12, 15]
    3. Cost sensitivity：正常费率 vs 2x 费率
    4. obs_day 稳健性：obs_day=[1] vs [3]

Pipeline Position:
    实验脚本（Phase 3）。
    上游: exit_experiment.py (X1 th=0.70 胜出)
    下游: 最终对比报告

Inputs:
    - stop_experiment/output/full_test_predictions.parquet

Outputs:
    - stop_experiment/output/backtest/combined_validation_3.csv

How to Run:
    python stop_experiment/backtest/combined_validator.py

Side Effects:
    - 读取预测数据和K线数据，输出CSV
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import OUTPUT_DIR, BACKTEST_DIR
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest, compute_summary,
)
from stop_experiment.backtest.exit_experiment import (
    compute_trade_mfe_mae, compute_exit_quality,
)


def run_one(label, test_df, price, td, prev, pred, max_stocks=10,
            exit_th=0.70, buy_cost=None, sell_cost=None):
    """运行单次回测，返回汇总 dict"""
    result = run_backtest(
        test_df, price, td, prev, pred,
        max_stocks=max_stocks, strategy="sell_score",
        exit_mode="model_exit", stop_loss=-0.07,
        buy_cls_exit_threshold=exit_th,
        buy_cost=buy_cost, sell_cost=sell_cost,
        strict=True,
    )
    s = compute_summary(result)
    trades_df = result.get("trades_df", pd.DataFrame())
    mfe_vals, mae_vals = compute_trade_mfe_mae(trades_df, price)
    mfe_mean = np.nanmean(mfe_vals) if mfe_vals else np.nan
    mae_mean = np.nanmean(mae_vals) if mae_vals else np.nan
    mfe_mae_diff = mfe_mean - abs(mae_mean) if not (np.isnan(mfe_mean) or np.isnan(mae_mean)) else np.nan
    post5_rets, _ = compute_exit_quality(trades_df, price)
    post5_mean = np.nanmean(post5_rets) if post5_rets else np.nan

    n_model = int((trades_df["sell_reason"] == "model_risk").sum()) if not trades_df.empty else 0
    n_stop = int((trades_df["sell_reason"] == "stop_loss").sum()) if not trades_df.empty else 0
    n_max_hold = int((trades_df["sell_reason"] == "max_hold").sum()) if not trades_df.empty else 0

    return {
        "label": label,
        "nav": s.get("final_nav", np.nan),
        "sharpe": s.get("sharpe", np.nan),
        "mdd": s.get("max_dd", np.nan),
        "win_rate": s.get("win_rate", np.nan),
        "avg_hold_days": s.get("avg_hold_days", np.nan),
        "n_trades": s.get("n_trades", 0),
        "mfe_mae_diff_pct": mfe_mae_diff * 100 if not np.isnan(mfe_mae_diff) else np.nan,
        "post5_ret_pct": post5_mean * 100 if not np.isnan(post5_mean) else np.nan,
        "exits_model": n_model,
        "exits_stop": n_stop,
        "exits_max_hold": n_max_hold,
    }


def main():
    EXIT_TH = 0.70

    print("=" * 70)
    print(f"Phase 3: 组合验证 (E0+X1, exit_th={EXIT_TH})")
    print("=" * 70)

    # 加载基准数据
    test_df, price, td, prev, pred = _load_data(candidate_obs_days=[1])
    test_df["trading_date"] = test_df["obs_date"]
    test_df["score"] = test_df["pred_sell_reg"]
    test_df["sell_score"] = test_df["pred_sell_reg"]
    print(f"  基准数据: {len(test_df)} 条, {len(td)} 交易日")

    rows = []

    # === 1. Slice 分片分析 ===
    print("\n--- 1. Slice 分片分析 ---")
    total_days = len(td)
    thirds = [td[:total_days//3], td[total_days//3:2*total_days//3], td[2*total_days//3:]]
    slice_names = ["Slice1 (早期)", "Slice2 (中期)", "Slice3 (近期)"]

    for name, slice_td_list in zip(slice_names, thirds):
        slice_td = pd.to_datetime(slice_td_list)
        # 过滤候选信号到该分片区间
        slice_df = test_df[test_df["trading_date"].isin(slice_td)].copy()
        if len(slice_df) == 0:
            print(f"  {name}: 无候选!")
            continue
        print(f"  {name}: {len(slice_df)} 候选, {slice_df['trading_date'].nunique()} 交易日...", end=" ", flush=True)
        r = run_one(name, slice_df, price, slice_td, prev, pred, exit_th=EXIT_TH)
        rows.append(r)
        print(f"NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, n_trades={r['n_trades']}")

    # === 2. Capacity 容量测试 ===
    print("\n--- 2. Capacity 容量测试 ---")
    for k in [5, 8, 10, 12, 15]:
        label = f"k={k}"
        print(f"  {label}...", end=" ", flush=True)
        r = run_one(label, test_df, price, td, prev, pred, max_stocks=k, exit_th=EXIT_TH)
        r["label"] = label
        rows.append(r)
        print(f"NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, nav_per_stock={r['nav']/k:.2f}")

    # === 3. Cost sensitivity ===
    print("\n--- 3. Cost sensitivity ---")
    # 正常费率 (基准 k=10)
    print("  Normal cost...", end=" ", flush=True)
    r = run_one("cost_normal", test_df, price, td, prev, pred, exit_th=EXIT_TH,
                 buy_cost=0.001, sell_cost=0.001)  # 含0.1%印花+交易费
    rows.append(r)
    print(f"NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}")

    # 无费率
    print("  No cost...", end=" ", flush=True)
    r = run_one("cost_zero", test_df, price, td, prev, pred, exit_th=EXIT_TH,
                 buy_cost=0.0, sell_cost=0.0)
    rows.append(r)
    print(f"NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}")

    # === 4. obs_day 稳健性 ===
    print("\n--- 4. obs_day 稳健性 === (验证 obs_day=[1] vs [3]) ---")
    for od in [1, 3]:
        od_test_df, od_price, od_td, od_prev, od_pred = _load_data(candidate_obs_days=[od])
        od_test_df["trading_date"] = od_test_df["obs_date"]
        od_test_df["score"] = od_test_df["pred_sell_reg"]
        od_test_df["sell_score"] = od_test_df["pred_sell_reg"]
        label = f"obs_day=[{od}]"
        print(f"  {label}: {len(od_test_df)} 条, {len(od_td)} 交易日...", end=" ", flush=True)
        r = run_one(label, od_test_df, od_price, od_td, od_prev, od_pred, exit_th=EXIT_TH)
        r["label"] = label
        rows.append(r)
        print(f"NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}")

    # 汇总输出
    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "combined_validation_3.csv")
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    df_out.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print("Phase 3 结果汇总")
    print("=" * 70)
    key_cols = ["label", "nav", "sharpe", "mdd", "win_rate", "avg_hold_days",
                "n_trades", "mfe_mae_diff_pct", "post5_ret_pct",
                "exits_model", "exits_stop", "exits_max_hold"]
    print(df_out[key_cols].to_string(index=False))

    # 稳定性评估
    print("\n--- 稳定性评估 ---")
    # Slice: 检查各分片 NAV 极差
    slice_rows = [r for r in rows if "Slice" in r["label"]]
    if slice_rows:
        navs = [r["nav"] for r in slice_rows]
        max_nav = max(navs)
        min_nav = min(navs)
        spread = (max_nav - min_nav) / max_nav * 100 if max_nav > 0 else 0
        print(f"  分片 NAV 极差: {min_nav:.2f} - {max_nav:.2f}, 相对分散度={spread:.1f}%")
        if spread > 50:
            print("  ⚠️ 分片分散度 > 50%，策略时序不稳定！")

    # obs_day: 验证 [1] 不劣于 [3]
    od1 = [r for r in rows if r["label"] == "obs_day=[1]"]
    od3 = [r for r in rows if r["label"] == "obs_day=[3]"]
    if od1 and od3:
        ratio = od1[0]["nav"] / od3[0]["nav"] if od3[0]["nav"] > 0 else 0
        print(f"  obs_day[1]/[3] NAV 比率: {ratio:.3f}")
        if ratio < 0.80:
            print("  ⚠️ obs_day=[1] 显著劣于 [3]！")
        else:
            print(f"  ✅ obs_day=[1] 相对 [3] 表现可接受")

    print(f"\n  输出: {path}")


if __name__ == "__main__":
    main()