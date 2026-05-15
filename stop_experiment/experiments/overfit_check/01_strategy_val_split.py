#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy-Val 时间切分过拟合验证

Purpose:
    检查回测参数是否在 test 集上过拟合。通过 strategy_val (2026-01~03) 调参，
    在 final_test (2026-04~05) 上验证参数泛化能力，并 bootstrap 评估低交易次数稳健性。

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - DB: stock_k_data

Outputs:
    - experiments/overfit_check/results/param_search.csv
    - experiments/overfit_check/results/final_test_validation.csv
    - experiments/overfit_check/results/bootstrap_ci.csv
    - experiments/overfit_check/results/verdict.json

How to Run:
    python -m stop_experiment.experiments.overfit_check.01_strategy_val_split
    python -m stop_experiment.experiments.overfit_check.01_strategy_val_split --bootstrap-only

Examples:
    python -m stop_experiment.experiments.overfit_check.01_strategy_val_split
    python -m stop_experiment.experiments.overfit_check.01_strategy_val_split --bootstrap-only

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
from tqdm import tqdm

from stop_experiment.pipeline.stop_config import OUTPUT_DIR
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data,
    run_backtest,
)
from stop_experiment.backtest.simple_backtest import compute_summary

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")

STRATEGY_VAL_START = pd.Timestamp("2026-01-01")
STRATEGY_VAL_END = pd.Timestamp("2026-02-28")
STRATEGY_EVAL_END = pd.Timestamp("2026-03-31")
FINAL_TEST_START = pd.Timestamp("2026-04-01")
FINAL_TEST_END = pd.Timestamp("2026-05-31")

BUY_CLS_EXIT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
TOP_K_LIST = [5, 10, 15]
STOP_LOSS_LIST = [-0.05, -0.07, -0.10]

BASELINE_PARAMS = {
    "buy_cls_exit_threshold": 0.7,
    "top_k": 10,
    "stop_loss": -0.07,
}

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
EXIT_BUFFER_DAYS = 60
PRE_BUFFER_DAYS = 5


def load_and_split_data():
    """加载全量数据并按 obs_date 切分为 strategy_val / final_test 两个子集。

    Returns
    -------
    dict with keys:
        sv_signals, sv_td, ft_signals, ft_td,
        price_pivot, prev_close_map, pred_lookup
    """
    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data()
    print(f"  全量数据: {len(test_df)} 信号, {len(trading_days)} 交易日")
    print(f"  信号日期范围: {test_df['obs_date'].min().date()} ~ {test_df['obs_date'].max().date()}")

    sv_signals = test_df[
        (test_df["obs_date"] >= STRATEGY_VAL_START) & (test_df["obs_date"] <= STRATEGY_VAL_END)
    ].copy()
    ft_signals = test_df[
        (test_df["obs_date"] >= FINAL_TEST_START) & (test_df["obs_date"] <= FINAL_TEST_END)
    ].copy()

    print(f"  strategy_val: {len(sv_signals)} 信号, 日期 {STRATEGY_VAL_START.date()} ~ {STRATEGY_VAL_END.date()}")
    print(f"  final_test:   {len(ft_signals)} 信号, 日期 {FINAL_TEST_START.date()} ~ {FINAL_TEST_END.date()}")

    if sv_signals.empty:
        raise ValueError("strategy_val 期间无信号数据")
    if ft_signals.empty:
        raise ValueError("final_test 期间无信号数据")

    sv_td_start = sv_signals["obs_date"].min() - pd.Timedelta(days=PRE_BUFFER_DAYS)
    sv_td_end = STRATEGY_EVAL_END + pd.Timedelta(days=EXIT_BUFFER_DAYS)
    sv_td = [d for d in trading_days if sv_td_start <= d <= sv_td_end]
    sv_eval_td = [d for d in sv_td if d <= STRATEGY_EVAL_END]

    ft_td_start = ft_signals["obs_date"].min() - pd.Timedelta(days=PRE_BUFFER_DAYS)
    ft_td_end = ft_signals["obs_date"].max() + pd.Timedelta(days=EXIT_BUFFER_DAYS)
    ft_td = [d for d in trading_days if ft_td_start <= d <= ft_td_end]

    print(f"  strategy_val 交易日: {len(sv_td)} ({sv_td[0].date() if sv_td else 'N/A'} ~ {sv_td[-1].date() if sv_td else 'N/A'})")
    print(f"  strategy_val 评估截止: {STRATEGY_EVAL_END.date()} ({len(sv_eval_td)} 交易日)")
    print(f"  final_test   交易日: {len(ft_td)} ({ft_td[0].date() if ft_td else 'N/A'} ~ {ft_td[-1].date() if ft_td else 'N/A'})")

    return {
        "sv_signals": sv_signals,
        "sv_td": sv_td,
        "sv_eval_td": sv_eval_td,
        "ft_signals": ft_signals,
        "ft_td": ft_td,
        "price_pivot": price_pivot,
        "prev_close_map": prev_close_map,
        "pred_lookup": pred_lookup,
    }


def run_param_search(data):
    """在 strategy_val 上搜索 4×3×3=36 组参数，返回按 NAV 降序排列的结果。"""
    rows = []
    combos = [
        (th, tk, sl)
        for th in BUY_CLS_EXIT_THRESHOLDS
        for tk in TOP_K_LIST
        for sl in STOP_LOSS_LIST
    ]
    print(f"\n[参数搜索] strategy_val 上搜索 {len(combos)} 组参数...")

    for buy_cls_th, top_k, stop_loss in tqdm(combos, desc="param_search", ncols=80):
        result = run_backtest(
            data["sv_signals"],
            data["price_pivot"],
            data["sv_td"],
            data["prev_close_map"],
            data["pred_lookup"],
            max_stocks=top_k,
            strategy="sell_score",
            exit_mode="model_exit",
            buy_cls_exit_threshold=buy_cls_th,
            stop_loss=stop_loss,
            strict=True,
        )
        nav_df = result["nav_df"]
        eval_end = STRATEGY_EVAL_END
        nav_df_eval = nav_df[nav_df["date"] <= eval_end]
        result_eval = {**result, "nav_df": nav_df_eval}
        s = compute_summary(result_eval)
        rows.append({
            "buy_cls_exit_threshold": buy_cls_th,
            "top_k": top_k,
            "stop_loss": stop_loss,
            "final_nav": s.get("final_nav", np.nan),
            "sharpe": s.get("sharpe", np.nan),
            "max_dd": s.get("max_dd", np.nan),
            "win_rate": s.get("win_rate", np.nan),
            "n_trades": s.get("n_trades", 0),
            "avg_net_ret": s.get("avg_net_ret", np.nan),
            "avg_hold_days": s.get("avg_hold_days", np.nan),
        })

    df = pd.DataFrame(rows).sort_values("final_nav", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def validate_on_final_test(data, param_search_df, top_n=3):
    """用 strategy_val 上 NAV 最优的 top_n 组参数在 final_test 上验证。"""
    top_params = param_search_df.head(top_n)
    rows = []

    baseline_result = run_backtest(
        data["ft_signals"],
        data["price_pivot"],
        data["ft_td"],
        data["prev_close_map"],
        data["pred_lookup"],
        max_stocks=BASELINE_PARAMS["top_k"],
        strategy="sell_score",
        exit_mode="model_exit",
        buy_cls_exit_threshold=BASELINE_PARAMS["buy_cls_exit_threshold"],
        stop_loss=BASELINE_PARAMS["stop_loss"],
        strict=True,
    )
    baseline_s = compute_summary(baseline_result)
    baseline_nav = baseline_s.get("final_nav", np.nan)
    print(f"\n  基线参数在 final_test 上: NAV={baseline_nav:.4f}")

    rows.append({
        "source": "baseline",
        "sv_rank": 0,
        "buy_cls_exit_threshold": BASELINE_PARAMS["buy_cls_exit_threshold"],
        "top_k": BASELINE_PARAMS["top_k"],
        "stop_loss": BASELINE_PARAMS["stop_loss"],
        "sv_nav": np.nan,
        "ft_nav": baseline_nav,
        "nav_decay": np.nan,
        "ft_sharpe": baseline_s.get("sharpe", np.nan),
        "ft_n_trades": baseline_s.get("n_trades", 0),
        "ft_max_dd": baseline_s.get("max_dd", np.nan),
        "ft_win_rate": baseline_s.get("win_rate", np.nan),
    })

    print(f"\n[final_test 验证] strategy_val top{top_n} 参数在 final_test 上的表现:")
    for _, row in top_params.iterrows():
        th = row["buy_cls_exit_threshold"]
        tk = int(row["top_k"])
        sl = row["stop_loss"]
        sv_nav = row["final_nav"]
        sv_rank = int(row["rank"])

        ft_result = run_backtest(
            data["ft_signals"],
            data["price_pivot"],
            data["ft_td"],
            data["prev_close_map"],
            data["pred_lookup"],
            max_stocks=tk,
            strategy="sell_score",
            exit_mode="model_exit",
            buy_cls_exit_threshold=th,
            stop_loss=sl,
            strict=True,
        )
        ft_s = compute_summary(ft_result)
        ft_nav = ft_s.get("final_nav", np.nan)
        nav_decay = (ft_nav - sv_nav) / sv_nav if sv_nav > 0 else np.nan

        print(f"    rank={sv_rank} th={th} k={tk} sl={sl}  sv_nav={sv_nav:.4f} ft_nav={ft_nav:.4f} decay={nav_decay:+.2%}")

        rows.append({
            "source": "sv_top",
            "sv_rank": sv_rank,
            "buy_cls_exit_threshold": th,
            "top_k": tk,
            "stop_loss": sl,
            "sv_nav": sv_nav,
            "ft_nav": ft_nav,
            "nav_decay": nav_decay,
            "ft_sharpe": ft_s.get("sharpe", np.nan),
            "ft_n_trades": ft_s.get("n_trades", 0),
            "ft_max_dd": ft_s.get("max_dd", np.nan),
            "ft_win_rate": ft_s.get("win_rate", np.nan),
        })

    return pd.DataFrame(rows)


def bootstrap_analysis(data, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """对基线参数的 59 笔交易做 bootstrap 稳健性分析。"""
    baseline_result = run_backtest(
        data["sv_signals"],
        data["price_pivot"],
        data["sv_td"],
        data["prev_close_map"],
        data["pred_lookup"],
        max_stocks=BASELINE_PARAMS["top_k"],
        strategy="sell_score",
        exit_mode="model_exit",
        buy_cls_exit_threshold=BASELINE_PARAMS["buy_cls_exit_threshold"],
        stop_loss=BASELINE_PARAMS["stop_loss"],
        strict=True,
    )
    trades_df = baseline_result.get("trades_df", pd.DataFrame())
    if trades_df.empty or "net_ret" not in trades_df.columns:
        print("  ⚠ 基线参数在 strategy_val 上无交易，跳过 bootstrap")
        return pd.DataFrame()

    net_rets = trades_df["net_ret"].values
    n_trades = len(net_rets)
    print(f"\n[Bootstrap] 基线参数在 strategy_val 上: {n_trades} 笔交易")

    trade_nav = float(np.prod(1 + net_rets))
    print(f"  交易 NAV (逐笔累积): {trade_nav:.4f}")

    rng = np.random.RandomState(seed)
    bootstrap_navs = []
    for _ in range(n_bootstrap):
        sample_idx = rng.randint(0, n_trades, size=n_trades)
        sample_rets = net_rets[sample_idx]
        bootstrap_navs.append(float(np.prod(1 + sample_rets)))

    bootstrap_navs = np.array(bootstrap_navs)
    ci_lower = float(np.percentile(bootstrap_navs, 2.5))
    ci_upper = float(np.percentile(bootstrap_navs, 97.5))
    boot_mean = float(bootstrap_navs.mean())
    boot_std = float(bootstrap_navs.std())

    print(f"  Bootstrap NAV: mean={boot_mean:.4f}, std={boot_std:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    sorted_rets = np.sort(net_rets)
    nav_remove_worst3 = float(np.prod(1 + sorted_rets[3:]))
    nav_remove_best3 = float(np.prod(1 + sorted_rets[:-3]))
    print(f"  去最差3笔: NAV={nav_remove_worst3:.4f} (原 {trade_nav:.4f})")
    print(f"  去最佳3笔: NAV={nav_remove_best3:.4f} (原 {trade_nav:.4f})")

    rows = [{
        "metric": "trade_nav",
        "value": trade_nav,
    }, {
        "metric": "n_trades",
        "value": n_trades,
    }, {
        "metric": "bootstrap_mean",
        "value": boot_mean,
    }, {
        "metric": "bootstrap_std",
        "value": boot_std,
    }, {
        "metric": "ci_2.5",
        "value": ci_lower,
    }, {
        "metric": "ci_97.5",
        "value": ci_upper,
    }, {
        "metric": "nav_remove_worst3",
        "value": nav_remove_worst3,
    }, {
        "metric": "nav_remove_best3",
        "value": nav_remove_best3,
    }, {
        "metric": "worst3_avg_ret",
        "value": float(sorted_rets[:3].mean()),
    }, {
        "metric": "best3_avg_ret",
        "value": float(sorted_rets[-3:].mean()),
    }]

    return pd.DataFrame(rows)


def determine_verdict(param_search_df, final_test_df, bootstrap_df):
    """综合判定过拟合风险。"""
    verdict = {
        "overfitting_detected": False,
        "overfitting_severity": "none",
        "param_sensitivity": "unknown",
        "bootstrap_robust": False,
        "details": {},
    }

    sv_top = param_search_df.head(3)
    if len(sv_top) > 0:
        sv_nav_range = sv_top["final_nav"].max() - sv_top["final_nav"].min()
        sv_nav_mean = sv_top["final_nav"].mean()
        sensitivity = sv_nav_range / sv_nav_mean if sv_nav_mean > 0 else 0
        verdict["details"]["sv_top3_nav_range"] = float(sv_nav_range)
        verdict["details"]["sv_top3_nav_mean"] = float(sv_nav_mean)
        verdict["details"]["param_sensitivity_ratio"] = float(sensitivity)

        if sensitivity < 0.05:
            verdict["param_sensitivity"] = "low"
        elif sensitivity < 0.15:
            verdict["param_sensitivity"] = "medium"
        else:
            verdict["param_sensitivity"] = "high"

    sv_top_ft = final_test_df[final_test_df["source"] == "sv_top"]
    if not sv_top_ft.empty:
        decays = sv_top_ft["nav_decay"].dropna()
        if len(decays) > 0:
            max_decay = float(decays.min())
            mean_decay = float(decays.mean())
            verdict["details"]["ft_max_nav_decay"] = max_decay
            verdict["details"]["ft_mean_nav_decay"] = mean_decay

            if max_decay < -0.30:
                verdict["overfitting_detected"] = True
                verdict["overfitting_severity"] = "strong"
            elif max_decay < -0.15:
                verdict["overfitting_detected"] = True
                verdict["overfitting_severity"] = "moderate"
            elif max_decay < -0.05:
                verdict["overfitting_detected"] = False
                verdict["overfitting_severity"] = "mild"
            else:
                verdict["overfitting_detected"] = False
                verdict["overfitting_severity"] = "none"

    if not bootstrap_df.empty:
        ci_lower_row = bootstrap_df[bootstrap_df["metric"] == "ci_2.5"]
        trade_nav_row = bootstrap_df[bootstrap_df["metric"] == "trade_nav"]
        remove_worst3_row = bootstrap_df[bootstrap_df["metric"] == "nav_remove_worst3"]

        if not ci_lower_row.empty:
            ci_lower = float(ci_lower_row["value"].iloc[0])
            verdict["details"]["bootstrap_ci_lower"] = ci_lower
            verdict["bootstrap_robust"] = ci_lower > 1.0

        if not trade_nav_row.empty and not remove_worst3_row.empty:
            orig_nav = float(trade_nav_row["value"].iloc[0])
            worst3_nav = float(remove_worst3_row["value"].iloc[0])
            verdict["details"]["worst3_nav_drop_pct"] = float((worst3_nav - orig_nav) / orig_nav) if orig_nav > 0 else 0

    return verdict


def main(bootstrap_only=False):
    print("=" * 60)
    print("Strategy-Val 时间切分过拟合验证")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/5] 加载并切分数据...")
    data = load_and_split_data()

    if bootstrap_only:
        print("\n[bootstrap-only] 跳过参数搜索和 final_test 验证")
        print("\n[2/5] Bootstrap 稳健性分析...")
        bootstrap_df = bootstrap_analysis(data)
        if not bootstrap_df.empty:
            bootstrap_path = os.path.join(RESULTS_DIR, "bootstrap_ci.csv")
            bootstrap_df.to_csv(bootstrap_path, index=False)
            print(f"  保存: {bootstrap_path}")
        return

    print("\n[2/5] strategy_val 参数搜索 (36 组)...")
    param_search_df = run_param_search(data)
    param_search_path = os.path.join(RESULTS_DIR, "param_search.csv")
    param_search_df.to_csv(param_search_path, index=False)
    print(f"\n  保存: {param_search_path}")
    print(f"\n  Top5 参数组:")
    for _, row in param_search_df.head(5).iterrows():
        print(f"    rank={int(row['rank'])} th={row['buy_cls_exit_threshold']} "
              f"k={int(row['top_k'])} sl={row['stop_loss']} "
              f"NAV={row['final_nav']:.4f} Sharpe={row['sharpe']:.2f} n={int(row['n_trades'])}")

    print("\n[3/5] final_test 验证...")
    final_test_df = validate_on_final_test(data, param_search_df, top_n=3)
    final_test_path = os.path.join(RESULTS_DIR, "final_test_validation.csv")
    final_test_df.to_csv(final_test_path, index=False)
    print(f"  保存: {final_test_path}")

    print("\n[4/5] Bootstrap 稳健性分析...")
    bootstrap_df = bootstrap_analysis(data)
    bootstrap_path = os.path.join(RESULTS_DIR, "bootstrap_ci.csv")
    if not bootstrap_df.empty:
        bootstrap_df.to_csv(bootstrap_path, index=False)
        print(f"  保存: {bootstrap_path}")

    print("\n[5/5] 综合判定...")
    verdict = determine_verdict(param_search_df, final_test_df, bootstrap_df)
    verdict_path = os.path.join(RESULTS_DIR, "verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print(f"  保存: {verdict_path}")

    print(f"\n{'='*60}")
    print("过拟合验证结论")
    print(f"{'='*60}")
    severity = verdict["overfitting_severity"]
    if verdict["overfitting_detected"]:
        print(f"  ⚠ 检测到过拟合 (严重度: {severity})")
    else:
        print(f"  ✅ 未检测到显著过拟合 (严重度: {severity})")
    print(f"  参数敏感度: {verdict['param_sensitivity']}")
    print(f"  Bootstrap 稳健: {'是' if verdict['bootstrap_robust'] else '否'}")
    if "ft_max_nav_decay" in verdict["details"]:
        print(f"  final_test 最大 NAV 衰减: {verdict['details']['ft_max_nav_decay']:+.2%}")
    if "bootstrap_ci_lower" in verdict["details"]:
        print(f"  Bootstrap 95% CI 下界: {verdict['details']['bootstrap_ci_lower']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy-Val 时间切分过拟合验证")
    parser.add_argument("--bootstrap-only", action="store_true",
                        help="仅运行 bootstrap 稳健性分析")
    args = parser.parse_args()
    main(bootstrap_only=args.bootstrap_only)
