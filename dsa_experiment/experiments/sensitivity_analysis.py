#!/usr/bin/env python3
"""
参数敏感性分析 + 精选回测验证

Purpose: 测试 top_n/veto/档位策略的参数敏感性，验证精选回测分年稳定性
Inputs: candidate_with_scores.parquet
Outputs: 终端报告, portfolio/sensitivity_report.csv
How to Run:
    python dsa_experiment/sensitivity_analysis.py
Examples:
    python dsa_experiment/sensitivity_analysis.py
Side Effects: 只读操作，输出文件到 dsa_experiment/output/portfolio/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

COST_BUY = 0.002
COST_SELL = 0.002


def compute_stats(df: pd.DataFrame, strategy_fn, **kwargs) -> dict:
    daily_groups = df.groupby("selection_date")
    nav_values = [1.0]
    all_rets = []
    all_mae = []
    all_mfe = []
    all_stop = []
    n_stocks_list = []

    for sel_date, day_df in daily_groups:
        if len(day_df) < 3:
            continue
        selected = strategy_fn(day_df, **kwargs)
        if selected.empty:
            continue
        n = len(selected)
        weight = 1.0 / n
        ret = (selected["ret_5_open_to_open"] * weight).sum()
        net_ret = ret - COST_BUY - COST_SELL
        nav_values.append(nav_values[-1] * (1 + net_ret))
        all_rets.append(ret)
        all_mae.append(selected["mae_5"].mean())
        all_mfe.append(selected["mfe_5"].mean())
        all_stop.append(selected["stop_hit_5"].mean())
        n_stocks_list.append(n)

    if not all_rets:
        return {}

    rets = np.array(all_rets)
    nav = np.array(nav_values)
    n_periods = len(rets)
    total_ret = nav[-1] / nav[0] - 1
    ann_ret = (1 + total_ret) ** (52 / n_periods) - 1 if n_periods > 0 else 0
    max_dd = (nav / np.maximum.accumulate(nav) - 1).min()
    win_rate = (rets > 0).mean()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        "n_periods": n_periods,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_ret_5": rets.mean(),
        "avg_mae_5": np.mean(all_mae),
        "avg_mfe_5": np.mean(all_mfe),
        "avg_stop_rate": np.mean(all_stop),
        "avg_n_stocks": np.mean(n_stocks_list),
    }


def select_top_n_veto(day_df, top_n=20, veto_pct=0.20):
    sorted_df = day_df.sort_values("return_score", ascending=False)
    if veto_pct > 0:
        veto_cutoff = sorted_df["risk_score"].quantile(1 - veto_pct)
        candidates = sorted_df[sorted_df["risk_score"] <= veto_cutoff]
        if len(candidates) < top_n // 2:
            candidates = sorted_df
        return candidates.head(top_n)
    return sorted_df.head(top_n)


def select_grade(day_df, grade="A"):
    if "grade" not in day_df.columns:
        return pd.DataFrame()
    return day_df[day_df["grade"] == grade]


def assign_grade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    opp_p80 = df["return_score"].quantile(0.80)
    risk_p50 = df["risk_score"].quantile(0.50)
    df["grade"] = "D"
    df.loc[(df["return_score"] >= opp_p80) & (df["risk_score"] <= risk_p50), "grade"] = "A"
    df.loc[(df["return_score"] >= opp_p80) & (df["risk_score"] > risk_p50), "grade"] = "B"
    df.loc[(df["return_score"] < opp_p80) & (df["risk_score"] <= risk_p50), "grade"] = "C"
    return df


def main():
    print("=" * 80)
    print("参数敏感性分析 + 精选回测验证")
    print("=" * 80)

    input_path = os.path.join(OUTPUT_DIR, "candidate_with_scores.parquet")
    df = pd.read_parquet(input_path)
    tradeable = df[df["can_buy_next_open"] == True].copy()
    tradeable = tradeable[tradeable["return_score"].notna() & tradeable["risk_score"].notna()]

    tradeable = assign_grade(tradeable)

    print(f"\n  记录数: {len(tradeable)}")
    print(f"  日期范围: {tradeable['selection_date'].min()} ~ {tradeable['selection_date'].max()}")
    grade_counts = tradeable["grade"].value_counts().to_dict()
    for g in ["A", "B", "C", "D"]:
        print(f"  {g}档: {grade_counts.get(g, 0)}")

    # ── 1. top_n 敏感性 ──
    print("\n" + "=" * 80)
    print("1. top_n 敏感性（veto=20%）")
    print("=" * 80)
    print(f"\n  {'top_n':>6} {'年化':>8} {'回撤':>8} {'卡玛':>8} {'胜率':>6} {'收益':>8} {'止损率':>6} {'持仓':>5}")
    print(f"  {'-'*65}")

    top_n_results = []
    for top_n in [10, 15, 20, 30, 50]:
        stats = compute_stats(tradeable, select_top_n_veto, top_n=top_n, veto_pct=0.20)
        if stats:
            print(f"  {top_n:>6} {stats['ann_return']:>7.0%} {stats['max_drawdown']:>7.0%} {stats['calmar']:>7.2f} {stats['win_rate']:>5.0%} {stats['avg_ret_5']:>7.2%} {stats['avg_stop_rate']:>5.0%} {stats['avg_n_stocks']:>5.0f}")
            stats["strategy"] = f"top{top_n}_veto20"
            top_n_results.append(stats)

    # ── 2. veto 阈值敏感性 ──
    print("\n" + "=" * 80)
    print("2. veto 阈值敏感性（top_n=20）")
    print("=" * 80)
    print(f"\n  {'veto':>6} {'年化':>8} {'回撤':>8} {'卡玛':>8} {'胜率':>6} {'收益':>8} {'止损率':>6} {'持仓':>5}")
    print(f"  {'-'*65}")

    veto_results = []
    for veto in [0.0, 0.10, 0.20, 0.30]:
        stats = compute_stats(tradeable, select_top_n_veto, top_n=20, veto_pct=veto)
        if stats:
            label = f"veto{int(veto*100)}%"
            print(f"  {label:>6} {stats['ann_return']:>7.0%} {stats['max_drawdown']:>7.0%} {stats['calmar']:>7.2f} {stats['win_rate']:>5.0%} {stats['avg_ret_5']:>7.2%} {stats['avg_stop_rate']:>5.0%} {stats['avg_n_stocks']:>5.0f}")
            stats["strategy"] = f"top20_veto{int(veto*100)}"
            veto_results.append(stats)

    # ── 3. 档位策略回测 ──
    print("\n" + "=" * 80)
    print("3. 档位策略回测")
    print("=" * 80)
    print(f"\n  {'档位':>8} {'年化':>8} {'回撤':>8} {'卡玛':>8} {'胜率':>6} {'收益':>8} {'止损率':>6} {'持仓':>5}")
    print(f"  {'-'*65}")

    grade_results = []
    for grade in ["A", "A+B", "A+B+C"]:
        if "+" in grade:
            grades = grade.split("+")
            fn = lambda day_df, gs=grades: day_df[day_df["grade"].isin(gs)] if "grade" in day_df.columns else pd.DataFrame()
        else:
            fn = lambda day_df, g=grade: select_grade(day_df, g)
        stats = compute_stats(tradeable, fn)
        if stats:
            print(f"  {grade:>8} {stats['ann_return']:>7.0%} {stats['max_drawdown']:>7.0%} {stats['calmar']:>7.2f} {stats['win_rate']:>5.0%} {stats['avg_ret_5']:>7.2%} {stats['avg_stop_rate']:>5.0%} {stats['avg_n_stocks']:>5.0f}")
            stats["strategy"] = f"grade_{grade.replace('+', '')}"
            grade_results.append(stats)

    # ── 4. 最优策略分年稳定性 ──
    print("\n" + "=" * 80)
    print("4. 最优策略分年稳定性（top20_veto20%）")
    print("=" * 80)

    tradeable["year"] = pd.to_datetime(tradeable["selection_date"]).dt.year
    print(f"\n  {'年份':>6} {'年化':>8} {'回撤':>8} {'胜率':>6} {'收益':>8} {'止损率':>6} {'期数':>5}")
    print(f"  {'-'*50}")

    for year, year_df in tradeable.groupby("year"):
        stats = compute_stats(year_df, select_top_n_veto, top_n=20, veto_pct=0.20)
        if stats and stats["n_periods"] >= 5:
            print(f"  {year:>6} {stats['ann_return']:>7.0%} {stats['max_drawdown']:>7.0%} {stats['win_rate']:>5.0%} {stats['avg_ret_5']:>7.2%} {stats['avg_stop_rate']:>5.0%} {stats['n_periods']:>5}")

    # ── 5. 综合判断 ──
    print("\n" + "=" * 80)
    print("5. 综合判断")
    print("=" * 80)

    all_results = top_n_results + veto_results + grade_results
    if all_results:
        best = max(all_results, key=lambda x: x.get("calmar", 0))
        print(f"\n  最优策略（卡玛比率最高）: {best['strategy']}")
        print(f"    年化: {best['ann_return']:.0%}, 回撤: {best['max_drawdown']:.0%}, 卡玛: {best['calmar']:.2f}")

        best_sharpe_like = max(all_results, key=lambda x: x.get("win_rate", 0) * abs(x.get("ann_return", 0)))
        print(f"  最优策略（胜率×年化最高）: {best_sharpe_like['strategy']}")
        print(f"    年化: {best_sharpe_like['ann_return']:.0%}, 胜率: {best_sharpe_like['win_rate']:.0%}")

        sensitivity_df = pd.DataFrame(all_results)
        out_cols = ["strategy", "ann_return", "max_drawdown", "calmar", "win_rate", "avg_ret_5", "avg_stop_rate", "avg_n_stocks", "n_periods"]
        avail = [c for c in out_cols if c in sensitivity_df.columns]
        portfolio_dir = os.path.join(OUTPUT_DIR, "portfolio")
        os.makedirs(portfolio_dir, exist_ok=True)
        sensitivity_df[avail].to_csv(os.path.join(portfolio_dir, "sensitivity_report.csv"), index=False)
        print(f"\n  已保存: {os.path.join(portfolio_dir, 'sensitivity_report.csv')}")

    print("\n" + "=" * 80)
    print("参数敏感性分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
