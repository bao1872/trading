#!/usr/bin/env python3
"""
日线最小交易协议回测

Purpose: 验证日线买点模型在最简单交易协议下能否稳定跑出正的交易质量
Inputs: output/daily_factor_with_scores.parquet or daily_factor_with_scores_strict.parquet
Outputs: output/daily_backtest_results.csv, 控制台报告
How to Run:
    python dsa_experiment/daily_backtest.py
    python dsa_experiment/daily_backtest.py --opp-threshold 0.7 --risk-threshold 0.3
    python dsa_experiment/daily_backtest.py --strict
Examples:
    python dsa_experiment/daily_backtest.py
    python dsa_experiment/daily_backtest.py --opp-threshold 0.8
    python dsa_experiment/daily_backtest.py --strict
Side Effects: 只读操作，输出CSV到 dsa_experiment/output/

交易协议说明：
  入场: daily_opportunity_score >= q{opp_threshold} AND daily_risk_score >= q{risk_threshold}
  出场: 3组对比 - F1(5天固定)/F2(10天固定)/F3(止损+5天)
  组合: 每天最多N只，等权，不加仓不减仓

严格模式（--strict）：
  使用 daily_factor_with_scores_strict.parquet
  阈值来自训练集OOF分位数（无泄漏）
"""

import sys
import os
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def run_backtest(df, hold_days, use_stopsell=False, max_daily_stocks=10, stop_threshold=-0.05):
    trades = []
    df = df.sort_values(["selection_date", "bar_time"]).reset_index(drop=True)

    buy_signals = df.copy()
    for date, group in buy_signals.groupby("selection_date"):
        if len(group) > max_daily_stocks:
            top_n = group.nlargest(max_daily_stocks, "daily_opportunity_score")
            for idx in top_n.index:
                row = df.loc[idx]
                trade = {
                    "buy_date": row["bar_time"],
                    "ts_code": row["ts_code"],
                    "buy_close": row["close"],
                    "opp_score": row["daily_opportunity_score"],
                    "risk_score": row["daily_risk_score"],
                    "day_offset": row["day_offset"],
                    "selection_date": row["selection_date"],
                }

                if use_stopsell:
                    mae_col = f"mae_{hold_days}"
                    if mae_col in df.columns and pd.notna(row.get(mae_col)):
                        if row.get(mae_col, 0) < stop_threshold:
                            trade["sell_reason"] = "stop"
                            trade["sell_date"] = row["bar_time"]
                            trade["ret"] = stop_threshold
                            trade["hold_days"] = 0
                            trades.append(trade)
                            continue

                ret_col = f"ret_{hold_days}_close_to_close"
                if ret_col in df.columns and pd.notna(row.get(ret_col)):
                    trade["ret"] = row[ret_col]
                    trade["sell_reason"] = "fixed"
                    trade["hold_days"] = hold_days
                    mfe_col = f"mfe_{hold_days}"
                    mae_col = f"mae_{hold_days}"
                    trade["mfe"] = row.get(mfe_col, np.nan)
                    trade["mae"] = row.get(mae_col, np.nan)
                    trades.append(trade)

    if not trades:
        return {}

    trades_df = pd.DataFrame(trades)
    n_trades = len(trades_df)
    avg_ret = trades_df["ret"].mean()
    avg_mae = trades_df["mae"].mean()
    avg_mfe = trades_df["mfe"].mean()
    win_rate = (trades_df["ret"] > 0).mean()
    rr = avg_ret / abs(avg_mae) if abs(avg_mae) > 1e-6 else np.nan

    trades_df["year"] = pd.to_datetime(trades_df["selection_date"]).dt.year
    yearly = {}
    for year, grp in trades_df.groupby("year"):
        yearly[year] = {
            "n": len(grp),
            "avg_ret": grp["ret"].mean(),
            "win_rate": (grp["ret"] > 0).mean(),
            "avg_mae": grp["mae"].mean(),
        }

    return {
        "hold_days": hold_days,
        "use_stopsell": use_stopsell,
        "n_trades": n_trades,
        "avg_ret": avg_ret,
        "avg_mae": avg_mae,
        "avg_mfe": avg_mfe,
        "win_rate": win_rate,
        "rr": rr,
        "yearly": yearly,
        "trades_df": trades_df,
    }


def main():
    parser = argparse.ArgumentParser(description="日线最小交易协议回测")
    parser.add_argument("--opp-threshold", type=float, default=0.7, help="opportunity score分位阈值")
    parser.add_argument("--risk-threshold", type=float, default=0.3, help="risk score分位阈值(低风险)")
    parser.add_argument("--max-daily", type=int, default=10, help="每天最多买几只")
    parser.add_argument("--stop-threshold", type=float, default=-0.05, help="止损阈值")
    parser.add_argument("--strict", action="store_true", help="严格模式：使用OOS数据+训练集阈值")
    parser.add_argument("--regime-filter", action="store_true", help="启用市场环境过滤")
    parser.add_argument("--regime-csv", type=str, default="/tmp/market_daily_summary.csv",
                        help="市场环境标签 CSV 路径")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("日线最小交易协议回测" + (" [严格模式]" if args.strict else ""))
    print("=" * 80)

    if args.strict:
        strict_path = os.path.join(OUTPUT_DIR, "daily_factor_with_scores_strict.parquet")
        if os.path.exists(strict_path):
            df = pd.read_parquet(strict_path)
            print(f"  使用严格OOS数据: {strict_path}")
        else:
            print(f"  错误: 未找到严格OOS数据 {strict_path}")
            return
    else:
        df = pd.read_parquet(os.path.join(OUTPUT_DIR, "daily_factor_with_scores.parquet"))

    print(f"  总记录数: {len(df)}")

    if args.strict and "opp_train_q70" in df.columns:
        opp_q = df["opp_train_q70"].iloc[0]
        risk_q = df["risk_train_q30"].iloc[0]
        print(f"  使用训练集OOF阈值（无泄漏）: opp_q70={opp_q:.4f}, risk_q30={risk_q:.4f}")
    else:
        opp_q = df["daily_opportunity_score"].quantile(args.opp_threshold)
        risk_q = df["daily_risk_score"].quantile(args.risk_threshold)
        print(f"  入场条件: opp_score >= {opp_q:.4f} (q{args.opp_threshold}), risk_score >= {risk_q:.4f} (q{args.risk_threshold})")

    buy_df = df[(df["daily_opportunity_score"] >= opp_q) & (df["daily_risk_score"] >= risk_q)].copy()
    print(f"  符合条件的记录: {len(buy_df)}")

    if args.regime_filter:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from market_structure_analysis.regime_filter import load_regime_series, get_regime_rules

        try:
            regime_series = load_regime_series(csv_path=args.regime_csv, use_cache=False)
        except (FileNotFoundError, KeyError) as e:
            print(f"  警告: 无法加载环境标签 ({e})，跳过过滤")
            regime_series = None

        if regime_series is not None and not regime_series.empty:
            buy_df["buy_date_dt"] = pd.to_datetime(buy_df["buy_date"]).dt.date
            buy_df["_regime"] = buy_df["buy_date_dt"].apply(
                lambda d: regime_series.get(pd.Timestamp(d), "中性")
            )
            before_filter = len(buy_df)
            buy_df = buy_df[buy_df["_regime"] != "强退潮"]
            after_filter = len(buy_df)
            filtered_out = before_filter - after_filter
            print(f"  [regime filter] 过滤前: {before_filter}, 过滤后: {after_filter}, 滤除: {filtered_out} ({filtered_out/before_filter*100:.1f}%)")

            regime_stats = buy_df["_regime"].value_counts()
            print(f"  信号环境分布: {dict(regime_stats)}")

    all_results = []

    # F1: 固定5天
    print("\n[F1] 固定持有5天...")
    f1 = run_backtest(buy_df, hold_days=5, max_daily_stocks=args.max_daily)
    if f1:
        print(f"  交易数={f1['n_trades']}, avg_ret={f1['avg_ret']:.2%}, win={f1['win_rate']:.0%}, RR={f1['rr']:.2f}")
        all_results.append({"group": "F1_hold5", **{k: v for k, v in f1.items() if k not in ["yearly", "trades_df"]}})

    # F2: 固定10天
    print("\n[F2] 固定持有10天...")
    f2 = run_backtest(buy_df, hold_days=10, max_daily_stocks=args.max_daily)
    if f2:
        print(f"  交易数={f2['n_trades']}, avg_ret={f2['avg_ret']:.2%}, win={f2['win_rate']:.0%}, RR={f2['rr']:.2f}")
        all_results.append({"group": "F2_hold10", **{k: v for k, v in f2.items() if k not in ["yearly", "trades_df"]}})

    # F3: 止损+5天
    print("\n[F3] 止损+5天...")
    f3 = run_backtest(buy_df, hold_days=5, use_stopsell=True, max_daily_stocks=args.max_daily, stop_threshold=args.stop_threshold)
    if f3:
        print(f"  交易数={f3['n_trades']}, avg_ret={f3['avg_ret']:.2%}, win={f3['win_rate']:.0%}, RR={f3['rr']:.2f}")
        all_results.append({"group": "F3_stop+hold5", **{k: v for k, v in f3.items() if k not in ["yearly", "trades_df"]}})

    # 汇总
    print(f"\n{'=' * 80}")
    print("交易协议回测汇总")
    print(f"{'=' * 80}")
    print(f"{'组':<20} {'交易数':>6} {'avg_ret':>8} {'avg_mae':>8} {'avg_mfe':>8} {'win':>5} {'RR':>5}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['group']:<20} {r['n_trades']:>6} {r['avg_ret']:>7.2%} {r['avg_mae']:>7.2%} {r['avg_mfe']:>7.2%} {r['win_rate']:>4.0%} {r['rr']:>5.2f}")

    # 按年表现
    print(f"\n按年表现:")
    for f_result, name in [(f1, "F1"), (f2, "F2"), (f3, "F3")]:
        if not f_result or "yearly" not in f_result:
            continue
        print(f"\n  {name}:")
        for year, yd in sorted(f_result["yearly"].items()):
            print(f"    {year}: n={yd['n']}, ret={yd['avg_ret']:.2%}, win={yd['win_rate']:.0%}, mae={yd['avg_mae']:.2%}")

    # 买点质量二维分层
    print(f"\n{'=' * 80}")
    print("买点质量分析（opp_score × risk_score 二维分层）")
    print(f"{'=' * 80}")
    df_opp_bins = pd.qcut(df["daily_opportunity_score"], 3, labels=["低机会", "中机会", "高机会"], duplicates="drop")
    df_risk_bins = pd.qcut(df["daily_risk_score"], 3, labels=["高风险", "中风险", "低风险"], duplicates="drop")
    cross = pd.DataFrame({"opp": df_opp_bins, "risk": df_risk_bins,
                           "ret": df["ret_5_close_to_close"], "mae": df["mae_5"],
                           "mfe": df["mfe_5"]})
    print(f"  {'区域':<20} {'n':>6} {'ret5':>8} {'mae5':>8} {'mfe5':>8} {'win':>5}")
    print("  " + "-" * 55)
    for opp_level in ["高机会", "中机会", "低机会"]:
        for risk_level in ["低风险", "中风险", "高风险"]:
            sub = cross[(cross["opp"] == opp_level) & (cross["risk"] == risk_level)]
            if len(sub) > 100:
                avg_ret = sub["ret"].mean()
                avg_mae = sub["mae"].mean()
                avg_mfe = sub["mfe"].mean()
                win = (sub["ret"] > 0).mean()
                print(f"  {opp_level}+{risk_level:<6} {len(sub):>6} {avg_ret:>7.2%} {avg_mae:>7.2%} {avg_mfe:>7.2%} {win:>4.0%}")

    pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_DIR, "daily_backtest_results.csv"), index=False)
    print(f"\n已保存: {os.path.join(OUTPUT_DIR, 'daily_backtest_results.csv')}")


if __name__ == "__main__":
    main()
