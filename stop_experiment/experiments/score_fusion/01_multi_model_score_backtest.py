#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型 rank 融合评分 vs 单一 pred_sell_reg 排序回测对比

Purpose:
    对比 4 种评分方案在回测中的效果:
    A: pred_sell_reg (基线，当前生产)
    B: rank(pred_sell_reg) 横截面 rank 归一化到 0~1
    C: 双门控 + rank 融合 (sell_reg 前30% + buy_cls 非高风险区 → 加权 rank)
    D: composite_score (sell_reg*0.5 + (-buy_reg)*0.5)

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - DB: stock_k_data

Outputs:
    - experiments/score_fusion/results/score_comparison.csv
    - experiments/score_fusion/results/gate_stats.csv
    - experiments/score_fusion/results/verdict.json

How to Run:
    python -m stop_experiment.experiments.score_fusion.01_multi_model_score_backtest
    python -m stop_experiment.experiments.score_fusion.01_multi_model_score_backtest --top-k 5

Examples:
    python -m stop_experiment.experiments.score_fusion.01_multi_model_score_backtest
    python -m stop_experiment.experiments.score_fusion.01_multi_model_score_backtest --top-k 5

Side Effects:
    - 只读 DB 和 parquet，输出仅写入 results/ 目录
"""

from __future__ import annotations

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data, run_backtest
from stop_experiment.backtest.simple_backtest import compute_summary

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")

STRATEGY_VAL_START = pd.Timestamp("2026-01-01")
STRATEGY_VAL_END = pd.Timestamp("2026-03-31")
FINAL_TEST_START = pd.Timestamp("2026-04-01")
FINAL_TEST_END = pd.Timestamp("2026-05-31")

EXIT_BUFFER_DAYS = 60
PRE_BUFFER_DAYS = 5

SELL_REG_PCT_GATE = 0.70
BUY_CLS_PCT_GATE = 0.70

FUSION_WEIGHTS = {
    "pred_sell_reg": 0.45,
    "pred_sell_cls": 0.25,
    "pred_buy_reg": 0.20,
    "pred_buy_cls": -0.10,
}

SCHEME_LABELS = {
    "A": "A_sell_reg_baseline",
    "B": "B_rank_sell_reg",
    "C": "C_dual_gate_rank_fusion",
    "D": "D_composite",
}


def load_and_split_data():
    """加载全量数据并按 obs_date 切分为 strategy_val / final_test。

    Returns
    -------
    dict : test_df, sv_signals, sv_td, ft_signals, ft_td,
           price_pivot, prev_close_map, pred_lookup
    """
    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data()
    print(f"  全量数据: {len(test_df)} 信号, {len(trading_days)} 交易日")
    print(f"  信号日期范围: {test_df['obs_date'].min().date()} ~ {test_df['obs_date'].max().date()}")

    sv_signals = test_df[
        (test_df["obs_date"] >= STRATEGY_VAL_START)
        & (test_df["obs_date"] <= STRATEGY_VAL_END)
    ].copy()
    ft_signals = test_df[
        (test_df["obs_date"] >= FINAL_TEST_START)
        & (test_df["obs_date"] <= FINAL_TEST_END)
    ].copy()

    print(f"  strategy_val: {len(sv_signals)} 信号")
    print(f"  final_test:   {len(ft_signals)} 信号")

    if sv_signals.empty:
        raise ValueError("strategy_val 期间无信号数据")
    if ft_signals.empty:
        raise ValueError("final_test 期间无信号数据")

    sv_td_start = sv_signals["obs_date"].min() - pd.Timedelta(days=PRE_BUFFER_DAYS)
    sv_td_end = sv_signals["obs_date"].max() + pd.Timedelta(days=EXIT_BUFFER_DAYS)
    sv_td = [d for d in trading_days if sv_td_start <= d <= sv_td_end]

    ft_td_start = ft_signals["obs_date"].min() - pd.Timedelta(days=PRE_BUFFER_DAYS)
    ft_td_end = ft_signals["obs_date"].max() + pd.Timedelta(days=EXIT_BUFFER_DAYS)
    ft_td = [d for d in trading_days if ft_td_start <= d <= ft_td_end]

    print(f"  strategy_val 交易日: {len(sv_td)}")
    print(f"  final_test   交易日: {len(ft_td)}")

    return {
        "test_df": test_df,
        "sv_signals": sv_signals,
        "sv_td": sv_td,
        "ft_signals": ft_signals,
        "ft_td": ft_td,
        "price_pivot": price_pivot,
        "prev_close_map": prev_close_map,
        "pred_lookup": pred_lookup,
    }


def prepare_scheme(signals_df, scheme_key):
    """按方案准备信号数据，注入 sell_score / composite_score 使 score_stocks 选用自定义分数。

    利用 score_stocks 的 "if col not in df.columns" 逻辑：预设列不会被覆盖。

    Parameters
    ----------
    signals_df : 全量 obs_day=1 信号
    scheme_key : "A" / "B" / "C" / "D"

    Returns
    -------
    (prepared_df, strategy, gate_info_or_none)
    """
    if scheme_key == "A":
        return signals_df.copy(), "sell_score", None

    if scheme_key == "B":
        df = signals_df.copy()
        df["sell_score"] = df.groupby("obs_date")["pred_sell_reg"].rank(pct=True)
        return df, "sell_score", None

    if scheme_key == "C":
        df = signals_df.copy()

        df["pred_sell_reg_pct"] = df.groupby("obs_date")["pred_sell_reg"].rank(pct=True)
        df["pred_buy_cls_pct"] = df.groupby("obs_date")["pred_buy_cls"].rank(pct=True)

        before_count = len(df)
        gate_mask = (
            (df["pred_sell_reg_pct"] >= SELL_REG_PCT_GATE)
            & (df["pred_buy_cls_pct"] <= BUY_CLS_PCT_GATE)
        )
        df = df[gate_mask].copy()
        after_count = len(df)

        df["rank_sell_reg"] = df.groupby("obs_date")["pred_sell_reg"].rank(pct=True)
        df["rank_sell_cls"] = df.groupby("obs_date")["pred_sell_cls"].rank(pct=True)
        df["rank_buy_reg"] = df.groupby("obs_date")["pred_buy_reg"].rank(pct=True)
        df["rank_buy_cls"] = df.groupby("obs_date")["pred_buy_cls"].rank(pct=True)

        df["composite_score"] = (
            FUSION_WEIGHTS["pred_sell_reg"] * df["rank_sell_reg"]
            + FUSION_WEIGHTS["pred_sell_cls"] * df["rank_sell_cls"]
            + FUSION_WEIGHTS["pred_buy_reg"] * df["rank_buy_reg"]
            + FUSION_WEIGHTS["pred_buy_cls"] * df["rank_buy_cls"]
        )

        gate_info = {
            "total_before": before_count,
            "total_after": after_count,
            "filter_rate": 1.0 - after_count / before_count if before_count > 0 else 0.0,
        }
        return df, "composite", gate_info

    if scheme_key == "D":
        return signals_df.copy(), "composite", None

    raise ValueError(f"未知方案: {scheme_key}")


def compute_gate_stats_daily(signals_df):
    """逐日统计方案 C 的门控过滤情况。"""
    df = signals_df.copy()
    df["pred_sell_reg_pct"] = df.groupby("obs_date")["pred_sell_reg"].rank(pct=True)
    df["pred_buy_cls_pct"] = df.groupby("obs_date")["pred_buy_cls"].rank(pct=True)

    gate_mask = (
        (df["pred_sell_reg_pct"] >= SELL_REG_PCT_GATE)
        & (df["pred_buy_cls_pct"] <= BUY_CLS_PCT_GATE)
    )

    daily_before = df.groupby("obs_date").size().rename("n_before")
    daily_after = df[gate_mask].groupby("obs_date").size().rename("n_after")

    stats = pd.concat([daily_before, daily_after], axis=1).fillna(0)
    stats["n_before"] = stats["n_before"].astype(int)
    stats["n_after"] = stats["n_after"].astype(int)
    stats["filter_rate"] = 1.0 - stats["n_after"] / stats["n_before"]
    stats = stats.reset_index()
    return stats


def run_all_comparisons(data, max_stocks=10):
    """对 4 种方案 × 2 个时间段运行回测，返回对比结果。"""
    rows = []
    gate_info = None
    gate_stats_daily = None

    for scheme_key, label in SCHEME_LABELS.items():
        print(f"\n  --- {label} ---")

        prepared_df, strategy, gate_info = prepare_scheme(data["test_df"], scheme_key)

        if scheme_key == "C":
            gate_stats_daily = compute_gate_stats_daily(data["test_df"])
            print(f"    门控过滤率: {gate_info['filter_rate']:.2%} "
                  f"({gate_info['total_before']} -> {gate_info['total_after']})")

        for period, period_start, period_end, td_key in [
            ("strategy_val", STRATEGY_VAL_START, STRATEGY_VAL_END, "sv_td"),
            ("final_test", FINAL_TEST_START, FINAL_TEST_END, "ft_td"),
        ]:
            period_mask = (
                (prepared_df["obs_date"] >= period_start)
                & (prepared_df["obs_date"] <= period_end)
            )
            period_signals = prepared_df[period_mask].copy()

            n_signals = len(period_signals)
            print(f"    {period}: {n_signals} 信号", end="")

            if period_signals.empty:
                rows.append({
                    "scheme": label, "period": period,
                    "final_nav": np.nan, "sharpe": np.nan, "max_dd": np.nan,
                    "n_trades": 0, "win_rate": np.nan,
                    "avg_net_ret": np.nan, "avg_hold_days": np.nan,
                })
                print(" -> 无信号")
                continue

            result = run_backtest(
                period_signals,
                data["price_pivot"],
                data[td_key],
                data["prev_close_map"],
                data["pred_lookup"],
                max_stocks=max_stocks,
                strategy=strategy,
                exit_mode="model_exit",
                strict=True,
            )
            summary = compute_summary(result)

            rows.append({
                "scheme": label,
                "period": period,
                "final_nav": summary.get("final_nav", np.nan),
                "sharpe": summary.get("sharpe", np.nan),
                "max_dd": summary.get("max_dd", np.nan),
                "n_trades": summary.get("n_trades", 0),
                "win_rate": summary.get("win_rate", np.nan),
                "avg_net_ret": summary.get("avg_net_ret", np.nan),
                "avg_hold_days": summary.get("avg_hold_days", np.nan),
            })

            nav = summary.get("final_nav", np.nan)
            sharpe = summary.get("sharpe", np.nan)
            n_trades = summary.get("n_trades", 0)
            nav_s = f"{nav:.4f}" if not np.isnan(nav) else "N/A"
            sharpe_s = f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"
            print(f" -> NAV={nav_s}, Sharpe={sharpe_s}, n={n_trades}")

    return pd.DataFrame(rows), gate_info, gate_stats_daily


def determine_verdict(comp_df, gate_info):
    """综合判定各方案优劣。"""
    verdict = {
        "best_scheme_sv": None,
        "best_scheme_ft": None,
        "scheme_c_vs_baseline": {},
        "recommendation": "",
        "details": {},
    }

    sv_rows = comp_df[comp_df["period"] == "strategy_val"]
    ft_rows = comp_df[comp_df["period"] == "final_test"]

    if not sv_rows.empty and sv_rows["final_nav"].notna().any():
        best_sv = sv_rows.loc[sv_rows["final_nav"].idxmax()]
        verdict["best_scheme_sv"] = best_sv["scheme"]
        verdict["details"]["sv_best_nav"] = float(best_sv["final_nav"])

    if not ft_rows.empty and ft_rows["final_nav"].notna().any():
        best_ft = ft_rows.loc[ft_rows["final_nav"].idxmax()]
        verdict["best_scheme_ft"] = best_ft["scheme"]
        verdict["details"]["ft_best_nav"] = float(best_ft["final_nav"])

    baseline_label = SCHEME_LABELS["A"]
    scheme_c_label = SCHEME_LABELS["C"]

    for period_key, rows in [("sv", sv_rows), ("ft", ft_rows)]:
        bl = rows[rows["scheme"] == baseline_label]
        sc = rows[rows["scheme"] == scheme_c_label]
        if not bl.empty and not sc.empty:
            bl_nav = bl["final_nav"].iloc[0]
            sc_nav = sc["final_nav"].iloc[0]
            if not np.isnan(bl_nav) and not np.isnan(sc_nav):
                verdict["scheme_c_vs_baseline"][f"{period_key}_nav_diff"] = float(sc_nav - bl_nav)

    if gate_info:
        verdict["details"]["gate_filter_rate"] = gate_info["filter_rate"]
        verdict["details"]["gate_total_before"] = gate_info["total_before"]
        verdict["details"]["gate_total_after"] = gate_info["total_after"]

    c_sv_diff = verdict["scheme_c_vs_baseline"].get("sv_nav_diff", 0)
    c_ft_diff = verdict["scheme_c_vs_baseline"].get("ft_nav_diff", 0)

    if c_sv_diff > 0 and c_ft_diff > 0:
        verdict["recommendation"] = "scheme_C_improves_both_periods"
    elif c_sv_diff > 0 or c_ft_diff > 0:
        verdict["recommendation"] = "scheme_C_mixed_results"
    else:
        verdict["recommendation"] = "baseline_sufficient"

    return verdict


def main(max_stocks=10):
    print("=" * 60)
    print("多模型 rank 融合评分 vs 单一 pred_sell_reg 回测对比")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/4] 加载并切分数据...")
    data = load_and_split_data()

    print("\n[2/4] 运行 4 种评分方案回测...")
    comp_df, gate_info, gate_stats_daily = run_all_comparisons(data, max_stocks)

    print("\n[3/4] 保存结果...")
    comp_path = os.path.join(RESULTS_DIR, "score_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"  保存: {comp_path}")

    if gate_stats_daily is not None:
        gate_path = os.path.join(RESULTS_DIR, "gate_stats.csv")
        gate_stats_daily.to_csv(gate_path, index=False)
        print(f"  保存: {gate_path}")

    print("\n[4/4] 综合判定...")
    verdict = determine_verdict(comp_df, gate_info)
    verdict_path = os.path.join(RESULTS_DIR, "verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print(f"  保存: {verdict_path}")

    print(f"\n{'=' * 80}")
    print("回测对比汇总")
    print(f"{'=' * 80}")
    header = (f"  {'方案':30s} {'期间':15s} {'NAV':>8s} {'Sharpe':>8s} "
              f"{'MDD':>8s} {'胜率':>8s} {'交易数':>6s}")
    print(header)
    for _, row in comp_df.iterrows():
        nav = f"{row['final_nav']:.4f}" if not np.isnan(row.get('final_nav', np.nan)) else "N/A"
        sharpe = f"{row['sharpe']:.2f}" if not np.isnan(row.get('sharpe', np.nan)) else "N/A"
        mdd = f"{row['max_dd']:.4f}" if not np.isnan(row.get('max_dd', np.nan)) else "N/A"
        wr = f"{row['win_rate']:.2%}" if not np.isnan(row.get('win_rate', np.nan)) else "N/A"
        print(f"  {row['scheme']:30s} {row['period']:15s} {nav:>8s} "
              f"{sharpe:>8s} {mdd:>8s} {wr:>8s} {row['n_trades']:>6d}")

    print(f"\n  结论: {verdict['recommendation']}")
    if verdict["scheme_c_vs_baseline"]:
        for k, v in verdict["scheme_c_vs_baseline"].items():
            print(f"    {k}: {v:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模型 rank 融合评分回测对比")
    parser.add_argument("--top-k", type=int, default=10, help="最大持仓数")
    args = parser.parse_args()
    main(max_stocks=args.top_k)
