#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[DEPRECATED] 环境分层回测: 验证模型退出是否只应在"高风险环境"强启用

⚠️ 已废弃：环境门控实验，未实际使用，无明确结论。

Purpose:
    (历史保留) 将 trading days 按风险环境分层，分别回测 fixed_hold 和 model_exit。

Pipeline Position:
    实验逻辑（已废弃，无替代）。

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - stop_experiment/output/candidate_with_scores.parquet
    - DB: stock_k_data

Outputs:
    - stop_experiment/output/backtest/dynamic/environment_gating.csv

How to Run:
    # ⚠️ 不建议运行，未验证
    python stop_experiment/backtest/environment_gating.py --top-k 10 --strategy sell_score

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
    OUTPUT_DIR, TRAIN_END, VAL_END,
)
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest, compute_summary,
)

DYNAMIC_DIR = os.path.join(OUTPUT_DIR, "backtest", "dynamic")


def build_risk_index(candidate_df, pred_col="pred_buy_cls", threshold=0.7):
    """
    构建每日风险指标: risk_score = fraction of (pred_buy_cls > threshold)
    """
    df = candidate_df.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    daily_risk = df.groupby("obs_date")[pred_col].apply(
        lambda x: (x > threshold).mean()
    ).rename("risk_score")
    return daily_risk


def main(args):
    print("=" * 60)
    print("环境分层回测")
    print("=" * 60)

    # 1. 加载数据
    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data(
        candidate_obs_days=args.candidate_obs_days if hasattr(args, 'candidate_obs_days') else [1]
    )

    candidate_path = os.path.join(OUTPUT_DIR, "candidate_with_scores.parquet")
    candidate = pd.read_parquet(candidate_path)
    candidate["obs_date"] = pd.to_datetime(candidate["obs_date"])
    print(f"  test候选: {len(test_df)}, 交易日: {len(trading_days)}")

    # 2. 构建风险指标
    # 用 train 期的数据计算阈值
    train_end_ts = pd.Timestamp(TRAIN_END)
    train_candidate = candidate[candidate["selection_date"] <= train_end_ts]
    risk_index = build_risk_index(train_candidate)
    median_risk = risk_index.median()
    print(f"  train期 risk_score median: {median_risk:.4f}")
    print(f"  risk_score 范围: {risk_index.min():.4f} ~ {risk_index.max():.4f}")

    # 3. 对 test 期的每个日期打上环境标签
    # 使用 test 期内部的 risk_score 中位数分层
    test_risk = build_risk_index(candidate[candidate["selection_date"] > pd.Timestamp(VAL_END)])
    test_median = test_risk.median()
    print(f"  test期 risk_score median: {test_median:.4f}")
    print(f"  test期 risk_score 范围: {test_risk.min():.4f} ~ {test_risk.max():.4f}")

    date_env = {}
    for d in sorted(test_df["obs_date"].unique()):
        risk = test_risk.get(d, np.nan)
        if pd.isna(risk):
            env = "unknown"
        elif risk > test_median:
            env = "high_risk"
        else:
            env = "normal"
        date_env[d] = env

    n_high = sum(1 for v in date_env.values() if v == "high_risk")
    n_normal = sum(1 for v in date_env.values() if v == "normal")
    print(f"\n  test期环境分布: high_risk={n_high}, normal={n_normal}, unknown={sum(1 for v in date_env.values() if v == 'unknown')}")

    # 4. 分层回测
    print(f"\n[分层回测] {args.strategy}, k={args.top_k}")
    print("-" * 70)

    all_rows = []
    for env_label in ["high_risk", "normal", "all"]:
        if env_label == "all":
            env_signals = test_df
            env_td = trading_days
        else:
            env_dates = [d for d, e in date_env.items() if e == env_label]
            if not env_dates:
                continue
            env_signals = test_df[test_df["obs_date"].isin(env_dates)]
            start_d = min(env_dates)
            end_d = max(env_dates) + pd.Timedelta(days=60)
            env_td = [d for d in trading_days if start_d <= d <= end_d]

        if env_signals.empty or not env_td:
            continue

        for exit_mode in ["fixed_hold", "model_exit"]:
            for hd in [5, 10]:
                if exit_mode == "model_exit" and hd != 5:
                    continue
                result = run_backtest(
                    env_signals, price_pivot, env_td, prev_close_map, pred_lookup,
                    max_stocks=args.top_k, strategy=args.strategy,
                    exit_mode=exit_mode, hold_days=hd, strict=True,
                )
                s = compute_summary(result)
                tdf = result.get("trades_df", pd.DataFrame())
                reasons = {}
                if not tdf.empty and "sell_reason" in tdf.columns:
                    reasons = tdf["sell_reason"].value_counts().to_dict()

                all_rows.append({
                    "environment": env_label,
                    "exit_mode": exit_mode,
                    "hold_days": hd,
                    "final_nav": s.get("final_nav", np.nan),
                    "sharpe": s.get("sharpe", np.nan),
                    "max_dd": s.get("max_dd", np.nan),
                    "win_rate": s.get("win_rate", np.nan),
                    "avg_net_ret": s.get("avg_net_ret", np.nan),
                    "avg_hold_days": s.get("avg_hold_days", np.nan),
                    "n_trades": s.get("n_trades", 0),
                    "n_model_risk": reasons.get("model_risk", 0),
                    "n_stop_loss": reasons.get("stop_loss", 0),
                    "n_max_hold": reasons.get("max_hold", 0),
                    "n_fixed": reasons.get("fixed", 0),
                })

                mrk = "M" if exit_mode == "model_exit" else "F"
                print(f"  {env_label:12s} {mrk} h={hd:2d} nav={s.get('final_nav', 0):.4f} "
                      f"sharpe={s.get('sharpe', 0):.2f} win={s.get('win_rate', 0):.2%} "
                      f"n={s.get('n_trades', 0)} avg_hold={s.get('avg_hold_days', 0):.1f}d")

    gate_df = pd.DataFrame(all_rows)
    path = os.path.join(DYNAMIC_DIR, "environment_gating.csv")
    gate_df.to_csv(path, index=False)
    print(f"\n  保存: {path}")

    # 5. 关键对比
    print(f"\n{'='*80}")
    print("环境分层对比: 模型退出的边际价值 (M - F)")
    print(f"{'='*80}")
    print(f"  {'环境':12s} {'F_nav':>8s} {'M_nav':>8s} {'M-F':>8s} {'F_sharpe':>8s} {'M_sharpe':>8s}")
    for env_label in ["high_risk", "normal", "all"]:
        f_row = [r for r in all_rows if r["environment"] == env_label and r["exit_mode"] == "fixed_hold" and r["hold_days"] == 5]
        m_row = [r for r in all_rows if r["environment"] == env_label and r["exit_mode"] == "model_exit"]
        fn = f_row[0]["final_nav"] if f_row else np.nan
        mn = m_row[0]["final_nav"] if m_row else np.nan
        diff = mn - fn if pd.notna(fn) and pd.notna(mn) else np.nan
        fs = f_row[0]["sharpe"] if f_row else np.nan
        ms = m_row[0]["sharpe"] if m_row else np.nan
        print(f"  {env_label:12s} {fn:.4f}   {mn:.4f}   {diff:+.4f}    {fs:.2f}        {ms:.2f}")

    # 结论
    f_high = [r for r in all_rows if r["environment"] == "high_risk" and r["exit_mode"] == "fixed_hold" and r["hold_days"] == 5]
    m_high = [r for r in all_rows if r["environment"] == "high_risk" and r["exit_mode"] == "model_exit"]
    f_norm = [r for r in all_rows if r["environment"] == "normal" and r["exit_mode"] == "fixed_hold" and r["hold_days"] == 5]
    m_norm = [r for r in all_rows if r["environment"] == "normal" and r["exit_mode"] == "model_exit"]

    mf_high = m_high[0]["final_nav"] - f_high[0]["final_nav"] if f_high and m_high else np.nan
    mf_norm = m_norm[0]["final_nav"] - f_norm[0]["final_nav"] if f_norm and m_norm else np.nan

    print(f"\n结论:")
    if pd.notna(mf_high) and pd.notna(mf_norm):
        if mf_high > mf_norm:
            print(f"  ✅ high_risk 环境 M-F={mf_high:+.4f} > normal 环境 M-F={mf_norm:+.4f}")
            print(f"  模型退出在风险环境中价值更大，环境分层有效")
        else:
            print(f"  ⚠️  high_risk M-F={mf_high:+.4f} <= normal M-F={mf_norm:+.4f}")
            print(f"  环境分层未发现显著差异")
        if mf_norm > 0:
            print(f"  normal 环境中 M 仍 > F (+{mf_norm:+.4f})，模型退出全域有效")
        else:
            print(f"  normal 环境中 M-F={mf_norm:+.4f}，可考虑在低风险环境关闭模型退出")

    # ====== 月度切片固定报告 ======
    print(f"\n{'='*60}")
    print("月度切片报告")
    print(f"{'='*60}")

    monthly_rows = []
    test_df["year_month"] = test_df["obs_date"].dt.to_period("M")

    for ym in sorted(test_df["year_month"].unique()):
        ym_signals = test_df[test_df["year_month"] == ym]
        ym_start = ym.start_time
        ym_end = ym.end_time + pd.Timedelta(days=60)
        ym_td = [d for d in trading_days if ym_start <= d <= ym_end]

        if ym_signals.empty or not ym_td:
            continue

        f_result = run_backtest(
            ym_signals, price_pivot, ym_td, prev_close_map, pred_lookup,
            max_stocks=args.top_k, strategy=args.strategy,
            exit_mode="fixed_hold", hold_days=5,
            strict=True,
        )
        f_summary = compute_summary(f_result)

        m_result = run_backtest(
            ym_signals, price_pivot, ym_td, prev_close_map, pred_lookup,
            max_stocks=args.top_k, strategy=args.strategy,
            exit_mode="model_exit", hold_days=5,
            strict=True,
        )
        m_summary = compute_summary(m_result)

        m_trades = m_result.get("trades_df", pd.DataFrame())
        n_model_risk = 0
        n_stop_loss = 0
        n_max_hold = 0
        if not m_trades.empty and "sell_reason" in m_trades.columns:
            reasons = m_trades["sell_reason"].value_counts().to_dict()
            n_model_risk = reasons.get("model_risk", 0)
            n_stop_loss = reasons.get("stop_loss", 0)
            n_max_hold = reasons.get("max_hold", 0)

        monthly_rows.append({
            "year_month": str(ym),
            "fixed_nav": f_summary.get("final_nav", np.nan),
            "model_nav": m_summary.get("final_nav", np.nan),
            "m_minus_f": m_summary.get("final_nav", np.nan) - f_summary.get("final_nav", np.nan),
            "pos_rate": ym_signals["buy_signal"].mean(),
            "n_model_risk": n_model_risk,
            "n_stop_loss": n_stop_loss,
            "n_max_hold": n_max_hold,
            "f_trades": f_summary.get("n_trades", 0),
            "m_trades": m_summary.get("n_trades", 0),
        })
        print(f"  {ym} F={f_summary.get('final_nav', 0):.3f} M={m_summary.get('final_nav', 0):.3f} "
              f"M-F={m_summary.get('final_nav', 0) - f_summary.get('final_nav', 0):+.3f}")

    if monthly_rows:
        monthly_df = pd.DataFrame(monthly_rows)
        slice_path = os.path.join(DYNAMIC_DIR, "monthly_slice_report.csv")
        monthly_df.to_csv(slice_path, index=False)
        print(f"\n  月度切片保存: {slice_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="环境分层回测")
    parser.add_argument("--top-k", type=int, default=10, help="最大持仓数")
    parser.add_argument("--strategy", type=str, default="sell_score",
                        choices=["sell_score", "low_risk", "composite"])
    parser.add_argument("--candidate-obs-days", type=int, nargs="+", default=[1],
                        help="候选池 obs_day 列表")
    args = parser.parse_args()
    main(args)
