#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样本外验证：双窗口滚动 + 月度分层验证

Purpose:
    对 E0+X1+0.70 进行样本外稳健性检验，参数全部固定：
    1. 双窗口滚动：20 日 / 步长 5 日 + 40 日 / 步长 10 日
    2. 月度切分：自然月独立回测，分层判定

Pipeline Position:
    Step 4 验证。
    上游：exit_recheck_exits.py
    下游：最终总结

Inputs:
    - stop_experiment/output/full_test_predictions.parquet

Outputs:
    - output/backtest/rolling_20d.csv / rolling_40d.csv
    - output/backtest/monthly_validation.csv

How to Run:
    python stop_experiment/backtest/out_of_sample_validator.py

Side Effects:
    - 读取预测数据和K线数据，输出CSV
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import BACKTEST_DIR, BASELINE_E0_X1_V1_PARAMS
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest, compute_summary,
)


def run_rolling_window(test_df, price_pivot, all_td, pred_lookup, prev_close_map,
                       window_size=20, step=5):
    """滚动窗口验证

    返回 DataFrame: [window_id, start_date, end_date, n_days, nav, sharpe, mdd, n_trades, win_rate]
    """
    rows = []
    n_windows = max(1, (len(all_td) - window_size) // step + 1)

    for w in range(n_windows):
        start_idx = w * step
        end_idx = start_idx + window_size
        if end_idx > len(all_td):
            end_idx = len(all_td)
            start_idx = end_idx - window_size

        window_td = all_td[start_idx:end_idx]
        start_date = window_td[0]
        end_date = window_td[-1]

        # 过滤候选
        win_df = test_df[test_df["trading_date"].isin(window_td)].copy()
        if len(win_df) == 0:
            continue
        if win_df["trading_date"].nunique() < 5:
            continue  # too few trading days

        try:
            result = run_backtest(
                win_df, price_pivot, window_td, prev_close_map, pred_lookup,
                max_stocks=10, strategy="sell_score",
                exit_mode="model_exit", stop_loss=-0.07,
                buy_cls_exit_threshold=0.70,
                strict=True,
            )
            s = compute_summary(result)
        except Exception as e:
            print(f"    ⚠️ 窗口 {w} ({start_date}→{end_date}) 异常: {e}")
            continue

        rows.append({
            "window_id": w,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "n_days": len(window_td),
            "nav": s.get("final_nav", np.nan),
            "sharpe": s.get("sharpe", np.nan),
            "mdd": s.get("max_dd", np.nan),
            "n_trades": s.get("n_trades", 0),
            "win_rate": s.get("win_rate", np.nan),
        })

    return pd.DataFrame(rows)


def run_monthly_validation(test_df, price_pivot, all_td, pred_lookup, prev_close_map):
    """自然月切分验证

    返回 DataFrame: [month, n_days, nav, sharpe, mdd, n_trades, win_rate]
    """
    td_series = pd.to_datetime(all_td)
    months = td_series.to_period("M").unique()

    rows = []
    for month in months:
        month_mask = td_series.to_period("M") == month
        month_td = td_series[month_mask].tolist()
        month_df = test_df[test_df["trading_date"].isin(month_td)].copy()

        if len(month_df) == 0 or len(month_td) < 5:
            continue

        try:
            result = run_backtest(
                month_df, price_pivot, month_td, prev_close_map, pred_lookup,
                max_stocks=10, strategy="sell_score",
                exit_mode="model_exit", stop_loss=-0.07,
                buy_cls_exit_threshold=0.70,
                strict=True,
            )
            s = compute_summary(result)
        except Exception as e:
            print(f"    ⚠️ 月 {month} 异常: {e}")
            continue

        rows.append({
            "month": str(month),
            "n_days": len(month_td),
            "nav": s.get("final_nav", np.nan),
            "sharpe": s.get("sharpe", np.nan),
            "mdd": s.get("max_dd", np.nan),
            "n_trades": s.get("n_trades", 0),
            "win_rate": s.get("win_rate", np.nan),
        })

    return pd.DataFrame(rows)


def main():
    EXIT_TH = 0.70

    print("=" * 70)
    print(f"Step 4: 样本外验证 (E0+X1, exit_th={EXIT_TH}, 参数固定)")
    print("=" * 70)

    test_df, price, td, prev, pred = _load_data(candidate_obs_days=[1])
    test_df["trading_date"] = test_df["obs_date"]
    test_df["score"] = test_df["pred_sell_reg"]
    test_df["sell_score"] = test_df["pred_sell_reg"]
    print(f"  数据: {len(test_df)} 条, {len(td)} 交易日")

    all_td = pd.to_datetime(td).sort_values()
    all_td_list = all_td.tolist()
    print(f"  区间: {all_td[0].date()} → {all_td[-1].date()} "
          f"({len(all_td)} 个交易日)\n")

    # === 4a. 双窗口滚动验证 ===
    print("--- 4a. 滚动窗口验证 ---")

    for ws, step, name in [(20, 5, "20d"), (40, 10, "40d")]:
        print(f"  {name} 窗口 (size={ws}, step={step})...", end=" ", flush=True)
        df_roll = run_rolling_window(
            test_df, price, all_td_list, pred, prev,
            window_size=ws, step=step,
        )
        path = os.path.join(BACKTEST_DIR, f"rolling_{name}.csv")
        df_roll.to_csv(path, index=False)

        n_wins = len(df_roll)
        nav_above_1 = (df_roll["nav"] > 1.0).sum()
        pct = nav_above_1 / n_wins * 100 if n_wins > 0 else 0
        nav_mean = df_roll["nav"].mean() if n_wins > 0 else np.nan
        nav_std = df_roll["nav"].std() if n_wins > 0 else np.nan

        print(f"{n_wins} 窗口, NAV>1: {nav_above_1}/{n_wins} ({pct:.1f}%), "
              f"avg NAV={nav_mean:.2f}±{nav_std:.2f}")

        # 判定
        threshold = 80 if ws == 20 else 90
        if pct >= threshold:
            print(f"    ✅ {name} 通过 (≥{threshold}%)")
        else:
            print(f"    ❌ {name} 未通过 ({pct:.1f}% < {threshold}%)")

        print(f"    → {path}")

    # === 4b. 月度验证 ===
    print("\n--- 4b. 月度切分验证 ---")
    df_monthly = run_monthly_validation(test_df, price, all_td_list, pred, prev)
    path_monthly = os.path.join(BACKTEST_DIR, "monthly_validation.csv")
    df_monthly.to_csv(path_monthly, index=False)

    n_months = len(df_monthly)
    nav_data = df_monthly["nav"].values
    nav_above_1 = (nav_data > 1.0).sum()
    monthly_win_rate = nav_above_1 / n_months * 100 if n_months > 0 else 0
    nav_min = nav_data.min() if n_months > 0 else np.nan

    print(f"  月数: {n_months}")
    print(f"  月末 NAV:")
    for _, row in df_monthly.iterrows():
        flag = " ✓" if row["nav"] > 1.0 else " ✗"
        print(f"    {row['month']}: NAV={row['nav']:.4f}, {row['n_trades']}笔{flag}")
    print(f"  月胜率: {nav_above_1}/{n_months} = {monthly_win_rate:.1f}%")
    print(f"  最差月 NAV: {nav_min:.4f}")

    # 分层判定
    no_extreme = nav_min > 0.85 if not np.isnan(nav_min) else False
    if monthly_win_rate >= 2/3 * 100 and no_extreme:
        # 检查是否极端依赖单月
        nav_no_top = np.sort(nav_data)[:-1] if n_months > 1 else nav_data
        top_contribution = (nav_data.max() - 1) / (np.sum(nav_data - 1)) if n_months > 1 and np.sum(nav_data - 1) > 0 else 0
        if top_contribution < 0.5:
            print(f"\n  🏆 很强通过: 月胜率={monthly_win_rate:.1f}% ≥ 2/3, "
                  f"无单月NAV<0.85, 收益不极端依赖单月")
        else:
            print(f"\n  ⚠️ 较强通过: 月胜率={monthly_win_rate:.1f}% ≥ 2/3, "
                  f"但单月贡献={top_contribution:.0%} > 50%, 不达很强")
    elif monthly_win_rate >= 60 and no_extreme:
        print(f"\n  ✅ 较强通过: 月胜率={monthly_win_rate:.1f}% ≥ 60%, "
              f"无单月 NAV<0.85")
    elif monthly_win_rate > 50 and no_extreme:
        print(f"\n  ⚠️ 基础通过: 月胜率={monthly_win_rate:.1f}% > 50%, "
              f"无单月 NAV<0.85, 信号偏弱")
    elif monthly_win_rate > 50:
        print(f"\n  ⚠️ 基础通过（有保留）: 月胜率={monthly_win_rate:.1f}% > 50%, "
              f"但最差月 NAV={nav_min:.4f} < 0.85")
    else:
        print(f"\n  ❌ 未通过: 月胜率={monthly_win_rate:.1f}% ≤ 50%")

    print(f"\n  → {path_monthly}")


if __name__ == "__main__":
    main()