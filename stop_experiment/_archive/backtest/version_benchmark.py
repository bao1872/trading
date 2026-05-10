#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[DEPRECATED] 版本验收模板: old vs new 固定对照

⚠️ 已废弃：版本对比实验，已融入 dynamic_exit_backtest_v2.py。
   替代: dynamic_exit_backtest_v2.py

Purpose:
    (历史保留) 固定输出 old(-5%+obs_day=1+fixed_hold) vs new(-7%+obs_day=1~3+model_exit) 全量对比。

Pipeline Position:
    实验逻辑（已废弃）。
    替代: dynamic_exit_backtest_v2.py

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - DB: stock_k_data

Outputs:
    - stop_experiment/output/backtest/benchmark_old_new.csv

How to Run:
    # ⚠️ 请勿运行，已废弃。请使用:
    python stop_experiment/backtest/dynamic_exit_backtest_v2.py --mode three-way

Side Effects:
    - (历史保留，不建议运行)
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, BACKTEST_DIR,
    V1_BASELINE, V1_PARAMS,
)
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest, compute_summary,
)
from stop_experiment.backtest.performance_report import compute_full_metrics

DYNAMIC_DIR = os.path.join(OUTPUT_DIR, "backtest", "dynamic")


def _recompute_buy_signal(df, threshold=-0.05):
    """用原始 mae_20 重新推导 buy_signal (用于模拟旧版阈值)"""
    df = df.copy()
    if "mae_20" not in df.columns:
        raise KeyError("需要 mae_20 列来重算 buy_signal")
    df["buy_signal"] = (df["mae_20"] < threshold).astype(float)
    return df


def _compute_calibration_error(df, pred_col="pred_buy_cls", label_col="buy_signal", n_buckets=10):
    """计算分桶校准误差: |positive_rate - mean_pred| 的加权平均"""
    df = df.copy()
    df = df.dropna(subset=[pred_col, label_col])
    if df.empty:
        return np.nan, None

    df["bucket"] = pd.qcut(df[pred_col], n_buckets, duplicates="drop")
    bucket_stats = df.groupby("bucket", observed=False).agg(
        actual_rate=(label_col, "mean"),
        mean_pred=(pred_col, "mean"),
        count=(label_col, "count"),
    ).reset_index()
    bucket_stats["abs_error"] = (bucket_stats["actual_rate"] - bucket_stats["mean_pred"]).abs()
    cal_error = (bucket_stats["abs_error"] * bucket_stats["count"]).sum() / bucket_stats["count"].sum()
    return cal_error, bucket_stats


def run_version_benchmark(args):
    print("=" * 70)
    print(f"版本验收: old vs new (V1 baseline: {V1_BASELINE})")
    print("=" * 70)

    # ---- 加载数据 ----
    full_pred_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    full_pred = pd.read_parquet(full_pred_path)
    full_pred["obs_date"] = pd.to_datetime(full_pred["obs_date"])

    # ---- 构建 new 版本 ----
    new_df = full_pred[full_pred["obs_day"].isin(V1_PARAMS["candidate_obs_days"])].copy()
    new_df = _recompute_buy_signal(new_df, threshold=V1_PARAMS["buy_signal_threshold"])
    print(f"\n  NEW: buy_signal=-7% + obs_day=1~3, 候选行数: {len(new_df)}")

    test_df_n, price_n, td_n, prev_n, pred_n = _load_data(
        candidate_obs_days=V1_PARAMS["candidate_obs_days"]
    )

    # ---- 构建 old 版本 (模拟) ----
    old_df = full_pred[full_pred["obs_day"] == 1].copy()
    old_df = _recompute_buy_signal(old_df, threshold=-0.05)
    print(f"  OLD: buy_signal=-5% + obs_day=1, 候选行数: {len(old_df)}")

    # old 版用 obs_day=1 路径，重新加载数据
    test_df_o, price_o, td_o, prev_o, pred_o = _load_data(
        candidate_obs_days=[1]
    )

    # ---- 运行回测 ----
    all_results = []

    # OLD: buy_signal=-5%, obs_day=1, fixed_hold(h=5)
    print(f"\n  --- OLD ---")
    result_old = run_backtest(
        test_df_o, price_o, td_o, prev_o, pred_o,
        max_stocks=args.top_k, strategy=args.strategy,
        exit_mode="fixed_hold", hold_days=5,
        strict=True,
    )
    metrics_old = compute_full_metrics(result_old)
    trades_old = result_old.get("trades_df", pd.DataFrame())
    sell_reasons_old = {}
    if not trades_old.empty and "sell_reason" in trades_old.columns:
        sell_reasons_old = trades_old["sell_reason"].value_counts().to_dict()

    row_old = {
        "version": "old",
        "exit_mode": "fixed_hold",
        "final_nav": metrics_old.get("final_nav", np.nan),
        "annual_return": metrics_old.get("annual_ret", np.nan),
        "max_drawdown": metrics_old.get("max_dd", np.nan),
        "sharpe": metrics_old.get("sharpe", np.nan),
        "win_rate": metrics_old.get("win_rate", np.nan),
        "profit_factor": metrics_old.get("profit_factor", np.nan),
        "calmar": metrics_old.get("calmar", np.nan),
        "volatility": metrics_old.get("volatility", np.nan),
        "n_trades": metrics_old.get("n_trades", 0),
        "avg_win": metrics_old.get("avg_win", np.nan),
        "avg_loss": metrics_old.get("avg_loss", np.nan),
        "max_win": metrics_old.get("max_win", np.nan),
        "max_loss": metrics_old.get("max_loss", np.nan),
        "n_fixed": sell_reasons_old.get("fixed", 0),
        "n_model_risk": sell_reasons_old.get("model_risk", 0),
        "n_stop_loss": sell_reasons_old.get("stop_loss", 0),
        "n_max_hold": sell_reasons_old.get("max_hold", 0),
    }
    all_results.append(row_old)
    print(f"    NAV={row_old['final_nav']:.4f} DD={row_old['max_drawdown']:.4f} "
          f"sharpe={row_old['sharpe']:.2f} win={row_old['win_rate']:.2%} pf={row_old['profit_factor']:.2f}")

    # NEW: buy_signal=-7%, obs_day=1~3, model_exit (V1_PARAMS)
    print(f"\n  --- NEW ---")
    result_new = run_backtest(
        test_df_n, price_n, td_n, prev_n, pred_n,
        max_stocks=args.top_k, strategy=args.strategy,
        exit_mode="model_exit", hold_days=5,
        strict=True,
    )
    metrics_new = compute_full_metrics(result_new)
    trades_new = result_new.get("trades_df", pd.DataFrame())
    sell_reasons_new = {}
    if not trades_new.empty and "sell_reason" in trades_new.columns:
        sell_reasons_new = trades_new["sell_reason"].value_counts().to_dict()

    row_new = {
        "version": "new",
        "exit_mode": "model_exit",
        "final_nav": metrics_new.get("final_nav", np.nan),
        "annual_return": metrics_new.get("annual_ret", np.nan),
        "max_drawdown": metrics_new.get("max_dd", np.nan),
        "sharpe": metrics_new.get("sharpe", np.nan),
        "win_rate": metrics_new.get("win_rate", np.nan),
        "profit_factor": metrics_new.get("profit_factor", np.nan),
        "calmar": metrics_new.get("calmar", np.nan),
        "volatility": metrics_new.get("volatility", np.nan),
        "n_trades": metrics_new.get("n_trades", 0),
        "avg_win": metrics_new.get("avg_win", np.nan),
        "avg_loss": metrics_new.get("avg_loss", np.nan),
        "max_win": metrics_new.get("max_win", np.nan),
        "max_loss": metrics_new.get("max_loss", np.nan),
        "n_fixed": sell_reasons_new.get("fixed", 0),
        "n_model_risk": sell_reasons_new.get("model_risk", 0),
        "n_stop_loss": sell_reasons_new.get("stop_loss", 0),
        "n_max_hold": sell_reasons_new.get("max_hold", 0),
    }
    all_results.append(row_new)
    print(f"    NAV={row_new['final_nav']:.4f} DD={row_new['max_drawdown']:.4f} "
          f"sharpe={row_new['sharpe']:.2f} win={row_new['win_rate']:.2%} pf={row_new['profit_factor']:.2f}")

    # ---- 模型指标 (用 obs_days_df 计算 AUC / calibration) ----
    from sklearn.metrics import roc_auc_score

    for label, obs_days_df in [("old", old_df), ("new", new_df)]:
        sub = obs_days_df.dropna(subset=["buy_signal", "pred_buy_cls"])
        if sub.empty:
            continue
        y = sub["buy_signal"].values
        yp = sub["pred_buy_cls"].values
        if len(np.unique(y)) < 2:
            continue

        # AUC
        auc_val = roc_auc_score(y, yp)
        # calibration
        cal_err, _ = _compute_calibration_error(sub)
        # pos_rate
        pos_rate = y.mean()
        # pred_buy_cls > 0.7 占比
        gt07_pct = (yp > 0.7).mean()

        for r in all_results:
            if r["version"] == label:
                r["auc"] = auc_val
                r["calibration_error"] = cal_err
                r["pos_rate"] = pos_rate
                r["pred_buy_cls_gt_0.7_pct"] = gt07_pct

    # ---- 汇总对比 ----
    df_out = pd.DataFrame(all_results)
    path = os.path.join(BACKTEST_DIR, "benchmark_old_new.csv")
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    df_out.to_csv(path, index=False)
    print(f"\n  输出: {path}")

    # ---- Console 对比表 ----
    print(f"\n{'='*80}")
    print("版本对照 (old vs new)")
    print(f"{'='*80}")
    headers = ["指标", "old (-5%+fixed)", "new (-7%+model)", "Δ"]
    rows = []

    def _fmt_float(v, fmt=".4f"):
        if pd.isna(v):
            return "N/A"
        return f"{v:{fmt}}"

    def _fmt_pct(v):
        if pd.isna(v):
            return "N/A"
        return f"{v:.2%}"

    old_r = next((r for r in all_results if r["version"] == "old"), None)
    new_r = next((r for r in all_results if r["version"] == "new"), None)

    if old_r and new_r:
        pairs = [
            ("正类率", "pos_rate", _fmt_pct),
            ("AUC", "auc", lambda v: _fmt_float(v, ".4f")),
            ("校准误差", "calibration_error", lambda v: _fmt_float(v, ".4f")),
            ("pred>0.7占比", "pred_buy_cls_gt_0.7_pct", _fmt_pct),
            ("终值NAV", "final_nav", lambda v: _fmt_float(v, ".4f")),
            ("年化收益", "annual_return", _fmt_pct),
            ("最大回撤", "max_drawdown", _fmt_pct),
            ("夏普率", "sharpe", lambda v: _fmt_float(v, ".2f")),
            ("胜率", "win_rate", _fmt_pct),
            ("盈亏比", "profit_factor", lambda v: _fmt_float(v, ".2f")),
            ("交易数", "n_trades", lambda v: str(int(v)) if pd.notna(v) else "N/A"),
        ]

        for name, key, fmt in pairs:
            ov = old_r.get(key, np.nan)
            nv = new_r.get(key, np.nan)
            delta = ""
            if pd.notna(ov) and pd.notna(nv) and key not in ("n_trades",):
                try:
                    diff = nv - ov
                    if key in ("pos_rate", "win_rate", "max_drawdown", "annual_return", "pred_buy_cls_gt_0.7_pct"):
                        delta = f"{diff:+.2%}"
                    else:
                        delta = f"{diff:+.4f}"
                except (TypeError, ValueError):
                    delta = ""
            rows.append([name, fmt(ov), fmt(nv), delta])

        # 卖出原因分布
        for reason in ["fixed", "model_risk", "stop_loss", "max_hold"]:
            o_key = f"n_{reason}" if reason != "fixed" else "n_fixed"
            ov = old_r.get(o_key, 0)
            nv = new_r.get(o_key, 0)
            rows.append([
                f"sell: {reason}",
                str(int(ov)) if pd.notna(ov) else "0",
                str(int(nv)) if pd.notna(nv) else "0",
                "",
            ])

        # 打印表格
        col_w = [18, 20, 20, 12]
        fmt_line = f"  {{:<{col_w[0]}}} {{:<{col_w[1]}}} {{:<{col_w[2]}}} {{:<{col_w[3]}}}"
        print(fmt_line.format(*headers))
        print("  " + "-" * (sum(col_w)))
        for row in rows:
            print(fmt_line.format(*[str(x)[:cw-1] for x, cw in zip(row, col_w)]))

    # ---- 结论 ----
    print(f"\n结论:")
    if old_r and new_r:
        o_nav = old_r.get("final_nav", 0)
        n_nav = new_r.get("final_nav", 0)
        diff = n_nav - o_nav if pd.notna(o_nav) and pd.notna(n_nav) else np.nan
        if pd.notna(diff):
            print(f"  新版 (V1) NAV: {n_nav:.4f}, 旧版 NAV: {o_nav:.4f}, Δ={diff:+.4f}")
            if diff > 0:
                print(f"  ✅ V1 优于旧版，版本升级有效")
            else:
                print(f"  ⚠️ V1 未优于旧版，需重新评估参数")
        o_wr = old_r.get("win_rate", 0)
        n_wr = new_r.get("win_rate", 0)
        print(f"  胜率: {o_wr:.2%} → {n_wr:.2%}")
        o_sh = old_r.get("sharpe", 0)
        n_sh = new_r.get("sharpe", 0)
        print(f"  夏普: {o_sh:.2f} → {n_sh:.2f}")
        print(f"  Frozen baseline: {V1_BASELINE} ({V1_PARAMS})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="版本验收: old vs new")
    parser.add_argument("--top-k", type=int, default=10, help="最大持仓数")
    parser.add_argument("--strategy", type=str, default="sell_score",
                        choices=["sell_score", "low_risk", "composite"])
    args = parser.parse_args()
    run_version_benchmark(args)
