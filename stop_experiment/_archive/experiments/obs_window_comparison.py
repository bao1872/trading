#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
候选观察窗口对照回测实验

Purpose:
    对比 4 种 candidate_obs_days 窗口的回测表现：
    C3=[1,2,3] (基线), C5=[1..5], C10=[1..10], C20=[1..20]。
    量化"窗口扩展是否有交易价值"。

    实验结论:
    - C3 最优: NAV 2.74, 胜率 82.9%
    - C20 最差: NAV 2.53, 胜率 75.8%
    - 结论: obs_day [1,2,3] 已固化到 stop_config.py，无需再扩展

Pipeline Position:
    实验逻辑（已完成，结论已固化）。
    上游: dynamic_exit_backtest_v2.py
    下游: —

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - DB: stock_k_data

Outputs:
    - Console: 对照指标表 + obs_day 分布
    - stop_experiment/output/obs_window_comparison/
      ├── metrics_comparison.csv
      ├── trades_C3/C5/C10/C20.csv
      └── nav_comparison.png

How to Run:
    python stop_experiment/experiments/obs_window_comparison.py

Side Effects:
    - 只读 parquet/DB，写 CSV/PNG 到 output 目录
"""

from __future__ import annotations

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

from stop_experiment.pipeline.stop_config import OUTPUT_DIR, V1_PARAMS
from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data, run_backtest


COMPARISON_DIR = os.path.join(OUTPUT_DIR, "obs_window_comparison")
FIXED_PARAMS = {
    "max_stocks": 10,
    "strategy": "sell_score",
    "exit_mode": "model_exit",
    "stop_loss": -0.07,
    "buy_cls_exit_threshold": 0.70,
    "max_hold_days": 20,
}


def compute_metrics_from_trades(trades_df, nav_df, initial_cash=1_000_000):
    """从 trades_df + nav_df 计算核心指标"""
    if trades_df.empty:
        return {
            "final_nav": nav_df["nav"].iloc[-1] if not nav_df.empty else 1.0,
            "annual_return": 0, "max_drawdown": 0,
            "win_rate": 0, "avg_hold_days": 0, "total_trades": 0,
            "avg_net_return": 0, "total_sells": 0, "nav_high": 1.0,
        }
    sells = trades_df.copy()
    total_sells = len(sells)
    win_rate = (sells["net_ret"] > 0).mean()
    avg_net_return = sells["net_ret"].mean()
    avg_hold_days = sells["hold_days"].mean() if "hold_days" in sells.columns else 0

    final_nav = 1.0
    max_drawdown = 0.0
    annual_return = 0.0
    nav_high = 1.0

    if not nav_df.empty and "nav" in nav_df.columns:
        nav_series = nav_df["nav"]
        final_nav = nav_series.iloc[-1]
        nav_high = nav_series.max()
        peak = nav_series.iloc[0]
        for v in nav_series.iloc[1:]:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_drawdown:
                max_drawdown = dd
        n = len(nav_series)
        years = max(n / 252, 0.01)
        if final_nav > 0:
            annual_return = (final_nav / nav_series.iloc[0]) ** (1 / years) - 1
    else:
        total_ret = sells["net_ret"].sum()
        final_nav = 1.0 + total_ret / initial_cash
        nav_high = max(final_nav, 1.0)

    return {
        "final_nav": final_nav,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_hold_days": avg_hold_days,
        "avg_net_return": avg_net_return,
        "total_trades": len(trades_df),
        "total_sells": total_sells,
        "nav_high": nav_high,
    }


def compute_obs_day_distribution(trades_df, signals_df):
    """计算买入交易中各 obs_day 的占比。trades_df 每行为已完成的交易，含 buy_date + ts_code。"""
    if trades_df.empty:
        return pd.Series(dtype=float)

    obs_day_counts = {}
    for _, trade in trades_df.iterrows():
        code = str(trade.get("ts_code", ""))
        code_clean = code[:6] if "." in code else code
        bdate = pd.to_datetime(trade.get("buy_date"))

        cand = signals_df[
            (signals_df["ts_code"].str[:6] == code_clean) &
            (signals_df["obs_date"] == bdate)
        ]
        if cand.empty:
            cand = signals_df[
                (signals_df["ts_code"].str.contains(code_clean, na=False)) &
                (signals_df["obs_date"] == bdate)
            ]
        if not cand.empty:
            od = cand.iloc[0].get("obs_day", "?")
        else:
            od = "?"
        obs_day_counts[od] = obs_day_counts.get(od, 0) + 1

    result = {}
    for k, v in sorted(obs_day_counts.items(),
                       key=lambda x: (not str(x[0]).isdigit(),
                                      int(x[0]) if str(x[0]).isdigit() else 999)):
        result[k] = v
    return pd.Series(result)


def plot_nav_comparison(nav_data, output_path):
    """绘制 4 条 NAV 曲线"""
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"C3": "#1f77b4", "C5": "#ff7f0e", "C10": "#2ca02c", "C20": "#d62728"}
    labels = {"C3": "基线 [1,2,3]", "C5": "实验A [1..5]",
              "C10": "实验B [1..10]", "C20": "实验C [1..20]"}

    for label, (nav_df, label_full) in nav_data.items():
        if nav_df.empty:
            continue
        x = np.arange(len(nav_df))
        ax.plot(x, nav_df["nav"], color=colors.get(label, "#333"),
                linewidth=1.5, label=labels.get(label, label), alpha=0.85)

    ax.set_title("候选观察窗口 NAV 对比 (C3 vs C5 vs C10 vs C20)", fontsize=14)
    ax.set_xlabel("交易日序号")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\n  NAV 对比图已保存: {output_path}")


def run_single_backtest(candidate_obs_days, label):
    """运行单个窗口的回测"""
    print(f"\n  [{label}] candidate_obs_days={candidate_obs_days} ...")
    test_df, price, td, prev_close, pred_lookup = _load_data(
        candidate_obs_days=candidate_obs_days
    )
    bt_result = run_backtest(
        test_df, price, td, prev_close, pred_lookup,
        max_stocks=FIXED_PARAMS["max_stocks"],
        strategy=FIXED_PARAMS["strategy"],
        exit_mode=FIXED_PARAMS["exit_mode"],
        stop_loss=FIXED_PARAMS["stop_loss"],
        buy_cls_exit_threshold=FIXED_PARAMS["buy_cls_exit_threshold"],
        debug_snapshots=False,
        strict=True,
    )
    trades_df = bt_result.get("trades_df", pd.DataFrame())
    nav_df = bt_result.get("nav_df", pd.DataFrame())
    signals_df = test_df.copy()
    metrics = compute_metrics_from_trades(trades_df, nav_df)
    metrics["obs_days"] = str(candidate_obs_days)
    metrics["label"] = label

    obs_dist = compute_obs_day_distribution(trades_df, signals_df)
    print(f"    交易: {metrics['total_trades']} 笔, 卖出: {metrics['total_sells']} 笔")
    print(f"    最终 NAV: {metrics['final_nav']:.4f}, 胜率: {metrics['win_rate']:.2%}, "
          f"最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"    平均持有: {metrics['avg_hold_days']:.1f} 天, 平均净收益: {metrics['avg_net_return']:+.4f}")

    return metrics, trades_df, nav_df, obs_dist


def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    windows = [
        ([1, 2, 3], "C3"),
        (list(range(1, 6)), "C5"),
        (list(range(1, 11)), "C10"),
        (list(range(1, 21)), "C20"),
    ]

    all_metrics = []
    all_trades = {}
    all_nav = {}
    all_obs_dist = {}

    for obs_days, label in windows:
        m, tdf, ndf, od = run_single_backtest(obs_days, label)
        all_metrics.append(m)
        all_trades[label] = tdf
        all_nav[label] = ndf
        all_obs_dist[label] = od

        tdf_path = os.path.join(COMPARISON_DIR, f"trades_{label}.csv")
        if not tdf.empty:
            tdf.to_csv(tdf_path, index=False)
            print(f"    交易明细已保存: {tdf_path}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(COMPARISON_DIR, "metrics_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 80)
    print("  候选观察窗口对照回测报告")
    print("=" * 80)
    print(f"\n  基准参数: max_stocks=10, strategy=sell_score, exit_mode=model_exit")
    print(f"             stop_loss=-7%, buy_cls_exit=0.70, max_hold=20d")
    print()
    header = f"  {'窗口':<6} {'obs_days':<20} {'最终NAV':>10} {'年化':>8} {'最大回撤':>8} {'胜率':>8} {'均持天数':>8} {'均净收益':>10} {'交易笔数':>8}"
    print(header)
    print(f"  {'-'*95}")
    for _, r in metrics_df.iterrows():
        ann = f"{r['annual_return']:+.2%}" if abs(r['annual_return']) < 100 else "N/A"
        print(f"  {r['label']:<6} {r['obs_days']:<20} {r['final_nav']:>10.4f} {ann:>8} "
              f"{r['max_drawdown']:>8.2%} {r['win_rate']:>8.2%} "
              f"{r['avg_hold_days']:>8.1f} {r['avg_net_return']:>10.4f} "
              f"{r['total_trades']:>8}")

    print(f"\n  {'─'*60}")
    print(f"  买入 obs_day 分布 (各窗口下买入的 obs_day 分布)")
    for label, dist in all_obs_dist.items():
        if dist.empty:
            print(f"  {label}: (无买入)")
            continue
        parts = [f"od={k}:{v}" for k, v in dist.items()]
        print(f"  {label}: {', '.join(parts)}")

    nav_data = {}
    for label, ndf in all_nav.items():
        nav_data[label] = (ndf, label)

    if any(not ndf.empty for ndf, _ in nav_data.values()):
        png_path = os.path.join(COMPARISON_DIR, "nav_comparison.png")
        plot_nav_comparison(nav_data, png_path)

    csv_path = os.path.join(COMPARISON_DIR, "metrics_comparison.csv")
    print(f"\n  指标 CSV: {csv_path}")
    print(f"  交易明细: {COMPARISON_DIR}/trades_C*.csv")
    print(f"\n  {'─'*60}")
    print(f"  结论:")
    c3_nav = metrics_df.loc[metrics_df["label"] == "C3", "final_nav"].values[0]
    c3_wr = metrics_df.loc[metrics_df["label"] == "C3", "win_rate"].values[0]
    best_nav = metrics_df.loc[metrics_df["final_nav"].idxmax()]
    best_wr = metrics_df.loc[metrics_df["win_rate"].idxmax()]

    print(f"    基线 C3 [1,2,3]: NAV={c3_nav:.4f}, 胜率={c3_wr:.2%}")
    print(f"    最高 NAV   {best_nav['label']} [{best_nav['obs_days']}]: "
          f"NAV={best_nav['final_nav']:.4f}, 胜率={best_nav['win_rate']:.2%}")
    print(f"    最高胜率   {best_wr['label']} [{best_wr['obs_days']}]: "
          f"NAV={best_wr['final_nav']:.4f}, 胜率={best_wr['win_rate']:.2%}")

    if best_nav["label"] == "C3":
        print(f"    → 当前窗口 [1,2,3] 已最优，无需扩展")
    else:
        diff = (best_nav["final_nav"] - c3_nav) / c3_nav
        print(f"    → {best_nav['label']} 相对基线提升 {diff:+.2%}，建议切换到 {best_nav['obs_days']}")

    print(f"\n  报告生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
