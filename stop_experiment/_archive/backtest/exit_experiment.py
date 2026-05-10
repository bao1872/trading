#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exit 聚合实验：比较 4 种 Exit 结构（用 E0 Entry）

Purpose:
    固定 Entry（E0: 无gate, sell_reg 排序, obs_day=[1]），比较 4 种退出结构：
    X0: 仅 stop_loss + max_hold（规则退出，无模型）
    X1: buy_cls > 0.70 + 规则退出
    X3: buy_cls > 0.70 且 sell_reg 较前日下降（衰减联动）
    X4: buy_cls > 0.70 或 buy_reg < -0.05（双风险模型联合）

Pipeline Position:
    实验脚本（Phase 2A）。
    上游: entry_experiment.py (E0 胜出)
    下游: combined_validator.py

Inputs:
    - stop_experiment/output/full_test_predictions.parquet

Outputs:
    - stop_experiment/output/backtest/exit_experiment_2a.csv

How to Run:
    python stop_experiment/backtest/exit_experiment.py

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


def compute_trade_mfe_mae(trades_df, price_pivot):
    """从 price_pivot 计算每笔交易的 MFE/MAE"""
    if trades_df.empty:
        return [], []

    mfe_vals, mae_vals = [], []
    price_close = price_pivot.xs("close", level=0, axis=1)

    for _, trade in trades_df.iterrows():
        code = trade["ts_code"]
        code_clean = code.split(".")[0] if "." in code else code
        buy_date = trade["buy_date"]
        sell_date = trade["sell_date"]
        buy_price = trade["buy_price"]

        if code_clean not in price_close.columns:
            mfe_vals.append(np.nan)
            mae_vals.append(np.nan)
            continue

        mask = (price_close.index >= buy_date) & (price_close.index <= sell_date)
        codes_prices = price_close.loc[mask, code_clean].dropna()

        if len(codes_prices) == 0 or buy_price <= 0:
            mfe_vals.append(np.nan)
            mae_vals.append(np.nan)
            continue

        rets = (codes_prices - buy_price) / buy_price
        mfe_vals.append(rets.max())
        mae_vals.append(rets.min())

    return mfe_vals, mae_vals


def compute_exit_quality(trades_df, price_pivot):
    """计算退出质量指标：卖出后 5/10 日收益"""
    if trades_df.empty:
        return [], []

    post5_rets, post10_rets = [], []
    price_close = price_pivot.xs("close", level=0, axis=1)
    max_idx = price_close.index.max()

    for _, trade in trades_df.iterrows():
        code = trade["ts_code"]
        code_clean = code.split(".")[0] if "." in code else code
        sell_date = trade["sell_date"]
        sell_price = trade["sell_price"]

        if code_clean not in price_close.columns:
            post5_rets.append(np.nan)
            post10_rets.append(np.nan)
            continue

        for horizon, rets_list in [(5, post5_rets), (10, post10_rets)]:
            end_date = sell_date + pd.Timedelta(days=horizon)
            end_date = min(end_date, max_idx)
            mask = (price_close.index > sell_date) & (price_close.index <= end_date)
            future_prices = price_close.loc[mask, code_clean].dropna()

            if len(future_prices) == 0 or sell_price <= 0:
                rets_list.append(np.nan)
            else:
                rets_list.append((future_prices.iloc[-1] - sell_price) / sell_price)

    return post5_rets, post10_rets


def run_exit_group(label, test_df, price, td, prev, pred,
                   exit_mode, exit_sub_mode=None,
                   exit_th=0.70, buy_reg_th=0.05,
                   max_stocks=10):
    """运行一组 Exit 实验

    返回 {label, nav, sharpe, mdd, win_rate, avg_hold, n_trades,
           mfe_mean, mae_mean, mfe_mae_diff,
           post5_ret, post10_ret, fp5_rate, fn5_rate,
           exits_model, exits_stop, exits_max_hold}
    """
    result = run_backtest(
        test_df, price, td, prev, pred,
        max_stocks=max_stocks, strategy="sell_score",
        exit_mode=exit_mode, stop_loss=-0.07,
        buy_cls_exit_threshold=exit_th,
        exit_sub_mode=exit_sub_mode,
        buy_reg_exit_threshold=buy_reg_th,
        strict=True,
    )

    s = compute_summary(result)
    trades_df = result.get("trades_df", pd.DataFrame())

    # MFE/MAE
    mfe_vals, mae_vals = compute_trade_mfe_mae(trades_df, price)
    mfe_mean = np.nanmean(mfe_vals) if mfe_vals else np.nan
    mae_mean = np.nanmean(mae_vals) if mae_vals else np.nan
    mfe_mae_diff = mfe_mean - abs(mae_mean) if not (np.isnan(mfe_mean) or np.isnan(mae_mean)) else np.nan

    # 退出后收益（越负越好，说明卖得对）
    post5_rets, post10_rets = compute_exit_quality(trades_df, price)
    post5_mean = np.nanmean(post5_rets) if post5_rets else np.nan
    post10_mean = np.nanmean(post10_rets) if post10_rets else np.nan

    # False Positive: 卖出后 5 日 mfe > 5%（误杀）
    fp5 = np.nanmean([v > 0.05 for v in post5_rets]) if post5_rets else np.nan
    # False Negative: 没卖出的持仓中 mae < -5%（漏杀）— 用 mae_lt_neg7 近似
    mae_vals_all = mae_vals
    fn5 = np.nanmean([v < -0.05 for v in mae_vals_all]) if mae_vals_all else np.nan

    # 退出原因分布
    n_model = int((trades_df["sell_reason"] == "model_risk").sum()) if not trades_df.empty else 0
    n_stop = int((trades_df["sell_reason"] == "stop_loss").sum()) if not trades_df.empty else 0
    n_max_hold = int((trades_df["sell_reason"] == "max_hold").sum()) if not trades_df.empty else 0

    return {
        "label": label,
        "exit_mode": exit_mode,
        "exit_sub_mode": str(exit_sub_mode),
        "nav": s.get("final_nav", np.nan),
        "sharpe": s.get("sharpe", np.nan),
        "mdd": s.get("max_dd", np.nan),
        "win_rate": s.get("win_rate", np.nan),
        "avg_hold_days": s.get("avg_hold_days", np.nan),
        "n_trades": s.get("n_trades", 0),
        "mfe_mae_diff_pct": mfe_mae_diff * 100 if not np.isnan(mfe_mae_diff) else np.nan,
        "mfe_mean_pct": mfe_mean * 100 if not np.isnan(mfe_mean) else np.nan,
        "mae_mean_pct": mae_mean * 100 if not np.isnan(mae_mean) else np.nan,
        "post5_ret_pct": post5_mean * 100 if not np.isnan(post5_mean) else np.nan,
        "post10_ret_pct": post10_mean * 100 if not np.isnan(post10_mean) else np.nan,
        "fp5_rate_pct": fp5 * 100 if not np.isnan(fp5) else np.nan,
        "fn5_rate_pct": fn5 * 100 if not np.isnan(fn5) else np.nan,
        "exits_model": n_model,
        "exits_stop": n_stop,
        "exits_max_hold": n_max_hold,
    }


def main():
    EXIT_TH = 0.70       # 控制变量
    BUY_REG_TH = 0.05    # X4 的 buy_reg 阈值

    print("=" * 70)
    print(f"Phase 2A: Exit 聚合实验 (E0 Entry, obs_day=[1], exit_th={EXIT_TH})")
    print("=" * 70)

    # 加载数据（E0 Entry: 无gate, sell_reg 排序）
    test_df, price, td, prev, pred = _load_data(candidate_obs_days=[1])
    test_df["trading_date"] = test_df["obs_date"]
    test_df["score"] = test_df["pred_sell_reg"]
    test_df["sell_score"] = test_df["pred_sell_reg"]
    print(f"  候选池: {len(test_df)} 条, {test_df['trading_date'].nunique()} 交易日")
    print(f"  exit_th={EXIT_TH}, buy_reg_th={BUY_REG_TH} (控制变量)\n")

    rows = []

    # X0: 仅 stop_loss + max_hold（规则退出）
    print("  运行 X0 (规则退出，无模型)...")
    r = run_exit_group("X0", test_df, price, td, prev, pred,
                       exit_mode="rule_exit", exit_th=EXIT_TH)
    rows.append(r)
    print(f"    NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, n_trades={r['n_trades']}, "
          f"avg_hold={r['avg_hold_days']:.1f}d")

    # X1: buy_cls > 0.70 模型退出
    print("  运行 X1 (buy_cls > 0.70 单阈值)...")
    r = run_exit_group("X1", test_df, price, td, prev, pred,
                       exit_mode="model_exit", exit_sub_mode=None,
                       exit_th=EXIT_TH)
    rows.append(r)
    print(f"    NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, "
          f"model={r['exits_model']}, stop={r['exits_stop']}, max={r['exits_max_hold']}")

    # X3: buy_cls > 0.70 且 sell_reg 衰减
    print("  运行 X3 (buy_cls + sell_reg decay)...")
    r = run_exit_group("X3", test_df, price, td, prev, pred,
                       exit_mode="model_exit", exit_sub_mode="sell_decay",
                       exit_th=EXIT_TH)
    rows.append(r)
    print(f"    NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, "
          f"model={r['exits_model']}, stop={r['exits_stop']}, max={r['exits_max_hold']}")

    # X4: buy_cls > 0.70 或 buy_reg < -0.05
    print("  运行 X4 (buy_cls or buy_reg)...")
    r = run_exit_group("X4", test_df, price, td, prev, pred,
                       exit_mode="model_exit", exit_sub_mode="or_buy_reg",
                       exit_th=EXIT_TH, buy_reg_th=BUY_REG_TH)
    rows.append(r)
    print(f"    NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, "
          f"model={r['exits_model']}, stop={r['exits_stop']}, max={r['exits_max_hold']}")

    # 汇总输出
    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "exit_experiment_2a.csv")
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    df_out.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    key_cols = ["label", "nav", "sharpe", "mdd", "win_rate", "avg_hold_days",
                "n_trades", "mfe_mae_diff_pct", "post5_ret_pct", "post10_ret_pct",
                "fp5_rate_pct", "fn5_rate_pct",
                "exits_model", "exits_stop", "exits_max_hold"]
    print(df_out[key_cols].to_string(index=False))

    # 胜出判定
    x0 = df_out[df_out["label"] == "X0"].iloc[0]
    print(f"\nX0 (规则退出) 基线: NAV={x0['nav']:.4f}, avg_hold={x0['avg_hold_days']:.1f}d, "
          f"post5={x0['post5_ret_pct']:.2f}%")

    winners = []
    for _, row in df_out.iterrows():
        if row["label"] == "X0":
            continue

        results = []
        # 1. 卖出后收益低于 X0（越负越好，说明卖得对）
        c1 = (not np.isnan(row["post5_ret_pct"]) and not np.isnan(x0["post5_ret_pct"]) and
              row["post5_ret_pct"] < x0["post5_ret_pct"])
        results.append(f"post5={'YES' if c1 else 'NO'}")

        # 2. false negative 低于 X0
        c2 = (not np.isnan(row["fn5_rate_pct"]) and not np.isnan(x0["fn5_rate_pct"]) and
              row["fn5_rate_pct"] < x0["fn5_rate_pct"])
        results.append(f"fn5={'YES' if c2 else 'NO'}")

        # 3. NAV 或 MDD 改善
        c3 = row["nav"] > x0["nav"] or row["mdd"] > x0["mdd"]
        results.append(f"nav_mdd={'YES' if c3 else 'NO'}")

        # 4. 不靠拉长持仓天数
        c4 = row["avg_hold_days"] < x0["avg_hold_days"] * 1.3
        results.append(f"hold={'YES' if c4 else 'NO'}")

        passed = sum([c1, c2, c3, c4])
        status = "✅ WIN" if passed >= 3 else f"❌ FAIL ({passed}/4)"
        if passed >= 3:
            winners.append(row["label"])
        print(f"  {row['label']}: {', '.join(results)} → {status}")

    if winners:
        print(f"\n  胜出组: {winners}")
    else:
        print("\n  ⚠️ 无 Exit 结构显著优于 X0 基线")

    print(f"\n  输出: {path}")


if __name__ == "__main__":
    main()


def phase_2b_scan():
    """Phase 2B: X1 buy_cls_exit_threshold 精细扫描"""
    import numpy as np, pandas as pd

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    print("=" * 70)
    print("Phase 2B: X1 buy_cls_exit_threshold 扫描 (E0 Entry, obs_day=[1])")
    print("=" * 70)

    test_df, price, td, prev, pred = _load_data(candidate_obs_days=[1])
    test_df["trading_date"] = test_df["obs_date"]
    test_df["score"] = test_df["pred_sell_reg"]
    test_df["sell_score"] = test_df["pred_sell_reg"]

    rows = []
    for th in thresholds:
        print(f"  th={th:.2f}...", end=" ", flush=True)
        result = run_backtest(
            test_df, price, td, prev, pred,
            max_stocks=10, strategy="sell_score",
            exit_mode="model_exit", stop_loss=-0.07,
            buy_cls_exit_threshold=th,
            strict=True,
        )
        s = compute_summary(result)
        trades_df = result.get("trades_df", pd.DataFrame())
        mfe_vals, mae_vals = compute_trade_mfe_mae(trades_df, price)
        mfe_mean = np.nanmean(mfe_vals) if mfe_vals else np.nan
        mae_mean = np.nanmean(mae_vals) if mae_vals else np.nan
        mfe_mae_diff = mfe_mean - abs(mae_mean) if not (np.isnan(mfe_mean) or np.isnan(mae_mean)) else np.nan

        post5_rets, post10_rets = compute_exit_quality(trades_df, price)
        post5_mean = np.nanmean(post5_rets) if post5_rets else np.nan

        n_model = int((trades_df["sell_reason"] == "model_risk").sum()) if not trades_df.empty else 0
        n_stop = int((trades_df["sell_reason"] == "stop_loss").sum()) if not trades_df.empty else 0
        n_max_hold = int((trades_df["sell_reason"] == "max_hold").sum()) if not trades_df.empty else 0

        rows.append({
            "threshold": th,
            "nav": s.get("final_nav"),
            "sharpe": s.get("sharpe"),
            "mdd": s.get("max_dd"),
            "win_rate": s.get("win_rate"),
            "avg_hold_days": s.get("avg_hold_days"),
            "n_trades": s.get("n_trades", 0),
            "mfe_mae_diff_pct": mfe_mae_diff * 100 if not np.isnan(mfe_mae_diff) else np.nan,
            "post5_ret_pct": post5_mean * 100 if not np.isnan(post5_mean) else np.nan,
            "exits_model": n_model,
            "exits_stop": n_stop,
            "exits_max_hold": n_max_hold,
        })
        print(f"NAV={rows[-1]['nav']:.4f}, Sharpe={rows[-1]['sharpe']:.2f}, "
              f"n_trades={rows[-1]['n_trades']}")

    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "exit_scan_2b.csv")
    df_out.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print("Phase 2B 结果汇总 (buy_cls_exit_threshold 扫描)")
    print("=" * 70)
    key_cols = ["threshold", "nav", "sharpe", "mdd", "win_rate", "avg_hold_days",
                "n_trades", "mfe_mae_diff_pct", "post5_ret_pct",
                "exits_model", "exits_stop", "exits_max_hold"]
    print(df_out[key_cols].to_string(index=False))

    best_nav = df_out.loc[df_out["nav"].idxmax()]
    best_sharpe = df_out.loc[df_out["sharpe"].idxmax()]
    print(f"\n  Best NAV: th={best_nav['threshold']:.2f}, NAV={best_nav['nav']:.4f}")
    print(f"  Best Sharpe: th={best_sharpe['threshold']:.2f}, Sharpe={best_sharpe['sharpe']:.2f}")
    print(f"\n  输出: {path}")