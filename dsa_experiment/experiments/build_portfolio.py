#!/usr/bin/env python3
"""
组合构建与回测：按增量价值验证方案构建7组策略组合

Purpose: 基于 advicement.txt 建议的渐进角色测试，构建7组策略对比 GBDT 增量价值
Inputs: candidate_with_scores.parquet
Outputs: portfolio/portfolio_nav.csv, portfolio/portfolio_trade_log.csv
How to Run:
    python dsa_experiment/build_portfolio.py
    python dsa_experiment/build_portfolio.py --top-n 10
Examples:
    python dsa_experiment/build_portfolio.py
    python dsa_experiment/build_portfolio.py --top-n 15
Side Effects: 只读操作，输出文件到 dsa_experiment/output/portfolio/

策略说明（对齐 advicement.txt 第七节）:
  S1 baseline_all:          全量 BBMACD 触发股，等权（第一步 baseline）
  S2 simple_bbmacd_top20:   按 bbmacd 原始值排序取 top20，等权（组4：简单规则）
  S3 gbdt_record_only:      全量持仓+记录GBDT分数，等权（角色1：被动观察）
  S4 gbdt_weighted_all:     全量持仓，按 return_score 加权（角色2：排序微调）
  S5 gbdt_return_top20:     按 return_score 排序取 top20，等权（角色2：强筛选版）
  S6 veto_only_20:          全量持仓，剔除 risk_score 前20%，剩余等权（角色3：独立veto）
  S7 gbdt_return_top20_veto20: top20 + veto 20%（角色3：组合版）
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

COST_BUY = 0.002
COST_SELL = 0.002
HOLD_DAYS = 5

PORTFOLIO_NAMES = [
    "baseline_all",
    "simple_bbmacd_top20",
    "gbdt_record_only",
    "gbdt_weighted_all",
    "gbdt_return_top20",
    "veto_only_20",
    "gbdt_return_top20_veto20",
]


def select_stocks_daily(day_df: pd.DataFrame, strategy: str, top_n: int = 20) -> pd.DataFrame:
    if strategy == "baseline_all":
        return day_df.copy()

    if strategy == "simple_bbmacd_top20":
        if "bbmacd" not in day_df.columns:
            return pd.DataFrame()
        return day_df.sort_values("bbmacd", ascending=False).head(top_n)

    if strategy == "gbdt_record_only":
        return day_df.copy()

    if strategy == "gbdt_weighted_all":
        return day_df.copy()

    if strategy == "gbdt_return_top20":
        return day_df.sort_values("return_score", ascending=False).head(top_n)

    if strategy == "veto_only_20":
        veto_cutoff = day_df["risk_score"].quantile(0.8)
        candidates = day_df[day_df["risk_score"] <= veto_cutoff]
        if len(candidates) < 10:
            candidates = day_df
        return candidates

    if strategy == "gbdt_return_top20_veto20":
        sorted_df = day_df.sort_values("return_score", ascending=False)
        veto_cutoff = sorted_df["risk_score"].quantile(0.8)
        candidates = sorted_df[sorted_df["risk_score"] <= veto_cutoff]
        if len(candidates) < top_n // 2:
            candidates = sorted_df
        return candidates.head(top_n)

    return day_df.head(top_n)


def compute_weights(selected: pd.DataFrame, strategy: str) -> pd.Series:
    n = len(selected)
    if n == 0:
        return pd.Series(dtype=float)

    if strategy == "gbdt_weighted_all":
        scores = selected["return_score"].values
        scores_min = scores.min()
        scores_shifted = scores - scores_min + 1e-8
        weights = scores_shifted / scores_shifted.sum()
        return pd.Series(weights, index=selected.index)

    return pd.Series([1.0 / n] * n, index=selected.index)


def build_portfolio(df: pd.DataFrame, strategy: str, top_n: int = 20) -> pd.DataFrame:
    daily_groups = df.groupby("selection_date")
    records = []

    for sel_date, day_df in daily_groups:
        if len(day_df) < 5:
            continue

        selected = select_stocks_daily(day_df, strategy, top_n)
        if selected.empty:
            continue

        weights = compute_weights(selected, strategy)

        for idx, (row_idx, row) in enumerate(selected.iterrows()):
            ret_5 = row.get("ret_5_open_to_open", np.nan)
            if np.isnan(ret_5):
                continue
            net_ret = ret_5 - COST_BUY - COST_SELL
            records.append({
                "selection_date": sel_date,
                "ts_code": row.get("ts_code_raw", row.get("ts_code", "")),
                "stock_name": row.get("stock_name", ""),
                "strategy": strategy,
                "return_score": row.get("return_score", np.nan),
                "risk_score": row.get("risk_score", np.nan),
                "bbmacd": row.get("bbmacd", np.nan),
                "ret_5_gross": ret_5,
                "ret_5_net": net_ret,
                "weight": weights.iloc[idx],
                "mae_5": row.get("mae_5", np.nan),
                "mfe_5": row.get("mfe_5", np.nan),
                "stop_hit_5": row.get("stop_hit_5", np.nan),
            })

    return pd.DataFrame(records)


def compute_nav(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()

    daily = trade_log.groupby("selection_date").apply(
        lambda g: pd.Series({
            "daily_ret_gross": np.average(g["ret_5_gross"], weights=g["weight"]),
            "daily_ret_net": np.average(g["ret_5_net"], weights=g["weight"]),
            "n_stocks": len(g),
            "avg_mae_5": g["mae_5"].mean(),
            "avg_mfe_5": g["mfe_5"].mean(),
            "stop_hit_rate": g["stop_hit_5"].mean(),
        })
    ).reset_index()

    daily = daily.sort_values("selection_date")
    daily["nav_gross"] = (1 + daily["daily_ret_gross"]).cumprod()
    daily["nav_net"] = (1 + daily["daily_ret_net"]).cumprod()

    return daily


def main():
    parser = argparse.ArgumentParser(description="组合构建与回测（增量价值验证版）")
    parser.add_argument("--top-n", type=int, default=20, help="每期选股数量")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "portfolio"), exist_ok=True)

    print("=" * 80)
    print("组合构建与回测（增量价值验证版）")
    print("=" * 80)

    print("\n[1/3] 加载预测分数据...")
    input_path = os.path.join(args.output_dir, "candidate_with_scores.parquet")
    df = pd.read_parquet(input_path)
    print(f"  记录数: {len(df)}")
    print(f"  日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")
    print(f"  有收益分: {df['return_score'].notna().sum()}")
    print(f"  有风险分: {df['risk_score'].notna().sum()}")

    tradeable = df[df["can_buy_next_open"] == True].copy()
    tradeable = tradeable[tradeable["return_score"].notna() & tradeable["risk_score"].notna()]
    print(f"  可交易且有预测分: {len(tradeable)}")

    print("\n[2/3] 构建组合...")
    all_logs = []
    all_navs = []

    for strategy in PORTFOLIO_NAMES:
        print(f"\n  --- {strategy} ---")
        log = build_portfolio(tradeable, strategy, args.top_n)
        if log.empty:
            print(f"    无交易记录")
            continue

        nav = compute_nav(log)
        log["strategy"] = strategy
        nav["strategy"] = strategy
        all_logs.append(log)
        all_navs.append(nav)

        if not nav.empty:
            final_nav = nav["nav_net"].iloc[-1]
            n_days = len(nav)
            avg_ret = nav["daily_ret_net"].mean()
            avg_n = nav["n_stocks"].mean()
            stop_rate = nav["stop_hit_rate"].mean()
            max_dd = (nav["nav_net"] / nav["nav_net"].cummax() - 1).min()
            print(f"    交易天数: {n_days}, 终值: {final_nav:.4f}")
            print(f"    日均收益: {avg_ret:.4%}, 平均持仓: {avg_n:.1f}")
            print(f"    止损率: {stop_rate:.2%}, 最大回撤: {max_dd:.2%}")

    print("\n[3/3] 保存结果...")
    if all_logs:
        trade_log = pd.concat(all_logs, ignore_index=True)
        trade_log.to_csv(os.path.join(args.output_dir, "portfolio", "portfolio_trade_log.csv"), index=False)
        print(f"  交易日志: {len(trade_log)} 条")

    if all_navs:
        nav_all = pd.concat(all_navs, ignore_index=True)
        nav_all.to_csv(os.path.join(args.output_dir, "portfolio", "portfolio_nav.csv"), index=False)
        print(f"  净值数据: {len(nav_all)} 条")

    print("\n" + "=" * 80)
    print("组合构建完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
