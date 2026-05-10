#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry 聚合实验：比较 6 种 Entry 结构（Gate × 排序）

Purpose:
    固定退出（model_exit, buy_cls_threshold=0.65），obs_day=[1]，比较：
    E0: 无gate, 排序=sell_reg（基线）
    E1: gate=sell_cls>0.60, 排序=sell_reg
    E2: gate=sell_cls>0.60, 排序=sell_reg × sell_cls
    E3: gate=sell_cls>0.60 AND buy_cls<0.40, 排序=sell_reg
    E4: gate=sell_cls>0.60 AND buy_cls<0.40, 排序=sell_reg × sell_cls
    E5: gate=sell_cls>0.60 AND buy_cls<0.40, 排序=rank(sell_reg)+rank(sell_cls)-0.5×rank(buy_cls)

Pipeline Position:
    实验脚本（Phase 1A）。
    上游: generate_full_predictions.py
    下游: exit_experiment.py

Inputs:
    - stop_experiment/output/full_test_predictions.parquet

Outputs:
    - stop_experiment/output/backtest/entry_experiment_1a.csv

How to Run:
    python stop_experiment/backtest/entry_experiment.py

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


def apply_entry_gate(candidates_df, gate_sell_cls=None, gate_buy_cls=None):
    """对单日候选池应用 gate 过滤，返回过滤后的 DataFrame"""
    df = candidates_df.copy()
    if gate_sell_cls is not None:
        df = df[df["pred_sell_cls"] > gate_sell_cls]
    if gate_buy_cls is not None:
        df = df[df["pred_buy_cls"] < gate_buy_cls]
    return df


def compute_rank_score(candidates_df, lamb=0.5):
    """计算 rank 联合评分（每日候选池内排名后聚合）"""
    df = candidates_df.copy()
    if len(df) < 2:
        df["entry_score"] = 0
        return df

    df["r_sell_reg"] = df["pred_sell_reg"].rank(pct=True)
    df["r_sell_cls"] = df["pred_sell_cls"].rank(pct=True)
    df["r_buy_cls"] = (1 - df["pred_buy_cls"]).rank(pct=True)

    df["entry_score"] = df["r_sell_reg"] + df["r_sell_cls"] - lamb * (1 - df["r_buy_cls"])
    return df


def compute_trade_mfe_mae(trades_df, price_pivot):
    """从 price_pivot 计算每笔交易的 MFE/MAE

    MFE (Maximum Favorable Excursion): 持仓期间最高未实现收益
    MAE (Maximum Adverse Excursion): 持仓期间最低未实现收益（最负的）
    """
    if trades_df.empty:
        return [], []

    mfe_vals, mae_vals = [], []
    price_close = price_pivot.xs("close", level=0, axis=1)  # MultiIndex level 0 = field (close/open/high/low/volume)

    for _, trade in trades_df.iterrows():
        code = trade["ts_code"]
        code_clean = code.split(".")[0] if "." in code else code  # remove .SZ/.SH suffix
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


def run_entry_group(label, test_df, price, td, prev, pred, gate_sell_cls, gate_buy_cls,
                    score_col="pred_sell_reg", exit_th=0.65, max_stocks=10):
    """运行一组 Entry 实验

    返回 {label, nav, sharpe, mdd, win_rate, avg_hold, n_trades,
           mfe_mean, mae_mean, mfe_mae_diff, mfe_gt_7pct, mae_lt_neg7pct,
           n_days_under_k, exits_model, exits_stop, exits_max, gate_pass_pct}
    """
    # 对每个交易日应用 gate
    test_gated = test_df.copy()
    test_gated["_pass_gate"] = True

    n_days_total = test_gated["trading_date"].nunique()
    n_days_under_k = 0

    for date_key in test_gated["trading_date"].unique():
        date_mask = test_gated["trading_date"] == date_key
        day_candidates = test_gated.loc[date_mask].copy()

        gated = apply_entry_gate(day_candidates, gate_sell_cls, gate_buy_cls)
        test_gated.loc[date_mask, "_pass_gate"] = test_gated.loc[date_mask].index.isin(gated.index)
        if len(gated) < max_stocks:
            n_days_under_k += 1

    # 计算排序分
    if score_col == "product":
        test_gated["_score"] = test_gated["pred_sell_reg"] * test_gated["pred_sell_cls"]
    elif score_col == "rank":
        scored_dfs = []
        for date_key in test_gated["trading_date"].unique():
            day = test_gated[test_gated["trading_date"] == date_key].copy()
            day = compute_rank_score(day, lamb=0.5)
            scored_dfs.append(day)
        test_gated = pd.concat(scored_dfs)
        test_gated["_score"] = test_gated["entry_score"]
    elif score_col == "rank_no_buy_reg":
        # E6: add buy_reg penalty
        scored_dfs = []
        for date_key in test_gated["trading_date"].unique():
            day = test_gated[test_gated["trading_date"] == date_key].copy()
            day = compute_rank_score(day, lamb=0.5)
            day["r_abs_buy_reg"] = day["pred_buy_reg"].abs().rank(pct=True)
            day["entry_score"] = day["entry_score"] - 0.3 * day["r_abs_buy_reg"]
            scored_dfs.append(day)
        test_gated = pd.concat(scored_dfs)
        test_gated["_score"] = test_gated["entry_score"]
    else:
        test_gated["_score"] = test_gated[score_col]

    # 只保留通过 gate 的候选
    test_gated = test_gated[test_gated["_pass_gate"]].copy()

    gate_pass_pct = 100 - (n_days_under_k / n_days_total * 100) if n_days_total > 0 else 100

    # 临时替换 score 列名以兼容 run_backtest
    test_gated["_orig_score"] = test_gated.get("score", np.nan)
    test_gated["_orig_sell_score"] = test_gated.get("sell_score", np.nan)
    test_gated["score"] = test_gated["_score"]
    test_gated["sell_score"] = test_gated["_score"]

    result = run_backtest(
        test_gated, price, td, prev, pred,
        max_stocks=max_stocks, strategy="sell_score",
        exit_mode="model_exit", stop_loss=-0.07,
        buy_cls_exit_threshold=exit_th,
        strict=True,
    )

    s = compute_summary(result)

    # MFE/MAE 从 price 数据计算（引擎未实现此功能）
    trades_df = result.get("trades_df", pd.DataFrame())
    mfe_vals, mae_vals = compute_trade_mfe_mae(trades_df, price)
    mfe_mean = np.nanmean(mfe_vals) if mfe_vals else np.nan
    mae_mean = np.nanmean(mae_vals) if mae_vals else np.nan
    mfe_mae_diff = mfe_mean - abs(mae_mean) if not (np.isnan(mfe_mean) or np.isnan(mae_mean)) else np.nan
    mfe_gt_7 = np.nanmean([v > 0.07 for v in mfe_vals]) if mfe_vals else np.nan
    mae_lt_neg7 = np.nanmean([v < -0.07 for v in mae_vals]) if mae_vals else np.nan

    # 退出原因分布（从 trades_df 统计）
    n_model = int((trades_df["sell_reason"] == "model_risk").sum()) if not trades_df.empty else 0
    n_stop = int((trades_df["sell_reason"] == "stop_loss").sum()) if not trades_df.empty else 0
    n_max_hold = int((trades_df["sell_reason"] == "max_hold").sum()) if not trades_df.empty else 0

    return {
        "label": label,
        "gate": f"sell_cls>{gate_sell_cls}" if gate_sell_cls else "none",
        "gate_buy_cls": f"buy_cls<{gate_buy_cls}" if gate_buy_cls else "none",
        "score": score_col,
        "nav": s.get("final_nav", np.nan),
        "sharpe": s.get("sharpe", np.nan),
        "mdd": s.get("max_dd", np.nan),
        "win_rate": s.get("win_rate", np.nan),
        "avg_hold_days": s.get("avg_hold_days", np.nan),
        "n_trades": s.get("n_trades", 0),
        "mfe_mean_pct": mfe_mean * 100 if not np.isnan(mfe_mean) else np.nan,
        "mae_mean_pct": mae_mean * 100 if not np.isnan(mae_mean) else np.nan,
        "mfe_mae_diff_pct": mfe_mae_diff * 100 if not np.isnan(mfe_mae_diff) else np.nan,
        "mfe_gt_7pct": mfe_gt_7 * 100 if not np.isnan(mfe_gt_7) else np.nan,
        "mae_lt_neg7pct": mae_lt_neg7 * 100 if not np.isnan(mae_lt_neg7) else np.nan,
        "gate_pass_pct": gate_pass_pct,
        "n_days_under_k": int(n_days_under_k),
        "exits_model": n_model,
        "exits_stop": n_stop,
        "exits_max_hold": n_max_hold,
    }


def main():
    EXIT_TH = 0.70  # 控制变量（非冻结最优），放大 Entry 质量差异

    print("=" * 70)
    print(f"Phase 1A revised: Entry 聚合实验 (obs_day=[1], exit_th={EXIT_TH}, relaxed gates)")
    print("=" * 70)

    # 加载数据
    test_df, price, td, prev, pred = _load_data(candidate_obs_days=[1])
    test_df["trading_date"] = test_df["obs_date"]
    print(f"  候选池: {len(test_df)} 条, {test_df['trading_date'].nunique()} 交易日")
    print(f"  exit_th={EXIT_TH} (控制变量，非冻结最优值)\n")

    rows = []

    # E0: 基线（无gate, 排序=sell_reg）
    print("  运行 E0 (基线)...")
    r = run_entry_group("E0", test_df, price, td, prev, pred,
                        gate_sell_cls=None, gate_buy_cls=None,
                        score_col="pred_sell_reg", exit_th=EXIT_TH)
    rows.append(r)
    print(f"    NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, "
          f"n_trades={r['n_trades']}, mfe_mae_diff={r['mfe_mae_diff_pct']:.2f}%")

    # E1: gate=sell_cls>0.60, sort=sell_reg
    print("  运行 E1...")
    r = run_entry_group("E1", test_df, price, td, prev, pred,
                        gate_sell_cls=0.60, gate_buy_cls=None,
                        score_col="pred_sell_reg", exit_th=EXIT_TH)
    rows.append(r)
    print(f"    NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, "
          f"gate_pass={r['gate_pass_pct']:.1f}%, n_trades={r['n_trades']}")

    # E3: gate=sell_cls>0.60 AND buy_cls<0.50 (比原0.40放宽), sort=sell_reg
    print("  运行 E3 (buy_cls<0.50)...")
    r = run_entry_group("E3", test_df, price, td, prev, pred,
                        gate_sell_cls=0.60, gate_buy_cls=0.50,
                        score_col="pred_sell_reg", exit_th=EXIT_TH)
    rows.append(r)
    print(f"    NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, "
          f"gate_pass={r['gate_pass_pct']:.1f}%, n_trades={r['n_trades']}")

    # E5: gate=sell_cls>0.60 AND buy_cls<0.50, sort=rank联合
    print("  运行 E5 (buy_cls<0.50 + rank)...")
    r = run_entry_group("E5", test_df, price, td, prev, pred,
                        gate_sell_cls=0.60, gate_buy_cls=0.50,
                        score_col="rank", exit_th=EXIT_TH)
    rows.append(r)
    print(f"    NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, "
          f"mfe_mae_diff={r['mfe_mae_diff_pct']:.2f}%")

    # E6: gate=sell_cls>0.60 AND buy_cls<0.50, sort=rank+buy_reg penalty
    print("  运行 E6 (buy_cls<0.50 + rank+buy_reg)...")
    r = run_entry_group("E6", test_df, price, td, prev, pred,
                        gate_sell_cls=0.60, gate_buy_cls=0.50,
                        score_col="rank_no_buy_reg", exit_th=EXIT_TH)
    rows.append(r)
    print(f"    NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, "
          f"mfe_mae_diff={r['mfe_mae_diff_pct']:.2f}%")

    # E3b: buy_cls<0.60 (更宽松), sort=sell_reg
    print("  运行 E3b (buy_cls<0.60)...")
    r = run_entry_group("E3b", test_df, price, td, prev, pred,
                        gate_sell_cls=0.60, gate_buy_cls=0.60,
                        score_col="pred_sell_reg", exit_th=EXIT_TH)
    rows.append(r)
    print(f"    NAV={r['nav']:.4f}, Sharpe={r['sharpe']:.2f}, "
          f"gate_pass={r['gate_pass_pct']:.1f}%, n_trades={r['n_trades']}")

    # 汇总输出
    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "entry_experiment_1a.csv")
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    df_out.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print(f"结果汇总 (exit_th={EXIT_TH}, obs_day=[1])")
    print("=" * 70)
    key_cols = ["label", "nav", "sharpe", "mdd", "win_rate", "avg_hold_days",
                "n_trades", "mfe_mae_diff_pct", "mfe_gt_7pct", "mae_lt_neg7pct",
                "gate_pass_pct", "exits_model", "exits_stop", "exits_max_hold"]
    print(df_out[key_cols].to_string(index=False))

    # 胜出判定
    e0 = df_out[df_out["label"] == "E0"].iloc[0]
    print(f"\nE0 基线: NAV={e0['nav']:.4f}, mfe_mae_diff={e0['mfe_mae_diff_pct']:.2f}%, "
          f"MDD={e0['mdd']:.4f}")

    winners = []
    for _, row in df_out.iterrows():
        if row["label"] == "E0":
            continue
        c1 = row["mfe_mae_diff_pct"] >= e0["mfe_mae_diff_pct"] + 1.0
        c2 = row["nav"] >= e0["nav"]
        c3 = row["mdd"] >= e0["mdd"] - 0.02
        passed = sum([c1, c2, c3])
        status = "✅ WIN" if passed == 3 else f"❌ FAIL ({passed}/3)"
        if passed == 3:
            winners.append(row["label"])
        print(f"  {row['label']}: mfe_mae+={c1}, nav_ok={c2}, mdd_ok={c3} → {status}")

    if winners:
        print(f"\n  胜出组: {winners}")
    else:
        print("\n  ⚠️ 无 Entry 结构严格优于 E0 基线，检查近似胜出组...")
        # 放宽判定：仅看 mfe_mae_diff + MDD 改善
        for _, row in df_out.iterrows():
            if row["label"] == "E0":
                continue
            if row["mfe_mae_diff_pct"] >= e0["mfe_mae_diff_pct"] + 0.5 and row["mdd"] > e0["mdd"]:
                print(f"  {row['label']}: mfe_mae={row['mfe_mae_diff_pct']:.2f}% (+{row['mfe_mae_diff_pct']-e0['mfe_mae_diff_pct']:.2f}), "
                      f"MDD={row['mdd']:.4f} (基线={e0['mdd']:.4f}) — 近似改善")

    print(f"\n  输出: {path}")


if __name__ == "__main__":
    main()