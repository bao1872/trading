#!/usr/bin/env python3
"""
Regime分段验证：验证模型在不同市场环境下的增量能力

Purpose: 按市场环境（下行/震荡/强趋势）分段评估模型排序能力
Inputs: output/daily_factor_with_scores.parquet
Outputs: output/regime_validation_report.csv, 控制台报告
How to Run:
    python dsa_experiment/regime_validation.py
Examples:
    python dsa_experiment/regime_validation.py
Side Effects: 只读操作，输出CSV

Regime定义:
  下行段: 2022-01 ~ 2023-12（沪深300跌33%）
  震荡段: 2024-01 ~ 2024-12（沪深300涨16%但最大回撤14%）
  强趋势段: 2025-01 ~ 2026-04（中证500涨47%）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

REGIMES = {
    "下行(2022-2023)": ("2022-01-01", "2023-12-31"),
    "震荡(2024)": ("2024-01-01", "2024-12-31"),
    "强趋势(2025-2026)": ("2025-01-01", "2026-12-31"),
}


def compute_regime_ic(df, pred_col, label_col):
    dedup = df[df["day_offset"] == 3].copy()
    valid = dedup[[pred_col, label_col, "selection_date"]].dropna()
    if len(valid) < 30:
        return {"total_ic": np.nan, "daily_ic_mean": np.nan, "daily_ic_std": np.nan, "icir": np.nan, "n": len(valid)}

    total_ic = stats.spearmanr(valid[pred_col], valid[label_col])[0]

    daily_ics = []
    for dt, grp in valid.groupby("selection_date"):
        if len(grp) >= 5:
            ic, _ = stats.spearmanr(grp[pred_col], grp[label_col])
            daily_ics.append(ic)

    if not daily_ics:
        return {"total_ic": total_ic, "daily_ic_mean": np.nan, "daily_ic_std": np.nan, "icir": np.nan, "n": len(valid)}

    ic_mean = np.mean(daily_ics)
    ic_std = np.std(daily_ics)
    icir = ic_mean / ic_std if ic_std > 0 else 0
    return {"total_ic": total_ic, "daily_ic_mean": ic_mean, "daily_ic_std": ic_std, "icir": icir, "n": len(valid)}


def compute_regime_quintile(df, pred_col, label_col):
    dedup = df[df["day_offset"] == 3].copy()
    valid = dedup[[pred_col, label_col]].dropna()
    if len(valid) < 50:
        return {}

    valid = valid.copy()
    valid["quintile"] = pd.qcut(valid[pred_col], 5, labels=False, duplicates="drop") + 1

    report = {}
    for q in sorted(valid["quintile"].unique()):
        sub = valid[valid["quintile"] == q]
        report[f"Q{int(q)}"] = {"n": len(sub), "mean": sub[label_col].mean()}

    if all(f"Q{i}" in report for i in range(1, 6)):
        report["spread"] = report["Q5"]["mean"] - report["Q1"]["mean"]
        report["monotonic"] = all(
            report[f"Q{i}"]["mean"] <= report[f"Q{i + 1}"]["mean"]
            for i in range(1, 5)
        ) or all(
            report[f"Q{i}"]["mean"] >= report[f"Q{i + 1}"]["mean"]
            for i in range(1, 5)
        )
    return report


def main():
    print("=" * 80)
    print("Regime分段验证")
    print("=" * 80)

    input_path = os.path.join(OUTPUT_DIR, "daily_factor_with_scores.parquet")
    df = pd.read_parquet(input_path)
    df["selection_date"] = pd.to_datetime(df["selection_date"])
    print(f"  总记录: {len(df)}")
    print(f"  日期范围: {df['selection_date'].min().date()} ~ {df['selection_date'].max().date()}")

    all_results = []

    for regime_name, (start, end) in REGIMES.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        regime_df = df[(df["selection_date"] >= start_ts) & (df["selection_date"] <= end_ts)].copy()

        if regime_df.empty:
            print(f"\n  {regime_name}: 无数据")
            continue

        print(f"\n{'=' * 80}")
        print(f"Regime: {regime_name}")
        print(f"{'=' * 80}")
        print(f"  记录数: {len(regime_df)}, 触发点: {regime_df['trigger_bar_time'].nunique()}, 股票: {regime_df['ts_code'].nunique()}")

        dedup = regime_df[regime_df["day_offset"] == 3]
        avg_ret5 = dedup["ret_5_close_to_close"].mean()
        avg_mae5 = dedup["mae_5"].mean()
        win_rate = (dedup["ret_5_close_to_close"] > 0).mean()
        print(f"  基准: avg_ret5={avg_ret5:.2%}, avg_mae5={avg_mae5:.2%}, win={win_rate:.0%}")

        opp_ic = compute_regime_ic(regime_df, "daily_opportunity_score", "ret_5_close_to_close")
        risk_ic = compute_regime_ic(regime_df, "daily_risk_score", "mae_5")

        print(f"\n  机会模型IC（去重后）:")
        print(f"    总IC={opp_ic['total_ic']:.4f}, 日频IC={opp_ic['daily_ic_mean']:.4f} ± {opp_ic['daily_ic_std']:.4f}, ICIR={opp_ic['icir']:.2f}")

        print(f"\n  风险模型IC（去重后）:")
        print(f"    总IC={risk_ic['total_ic']:.4f}, 日频IC={risk_ic['daily_ic_mean']:.4f} ± {risk_ic['daily_ic_std']:.4f}, ICIR={risk_ic['icir']:.2f}")

        opp_q = compute_regime_quintile(regime_df, "daily_opportunity_score", "ret_5_close_to_close")
        if opp_q:
            print(f"\n  机会模型5分组（去重后）:")
            for q in range(1, 6):
                if f"Q{q}" in opp_q:
                    print(f"    Q{q}: n={opp_q[f'Q{q}']['n']}, avg_ret5={opp_q[f'Q{q}']['mean']:.2%}")
            if "spread" in opp_q:
                print(f"    Spread: {opp_q['spread']:.2%}, 单调: {'是' if opp_q.get('monotonic') else '否'}")

        risk_q = compute_regime_quintile(regime_df, "daily_risk_score", "mae_5")
        if risk_q:
            print(f"\n  风险模型5分组（去重后）:")
            for q in range(1, 6):
                if f"Q{q}" in risk_q:
                    print(f"    Q{q}: n={risk_q[f'Q{q}']['n']}, avg_mae5={risk_q[f'Q{q}']['mean']:.2%}")
            if "spread" in risk_q:
                print(f"    Spread: {risk_q['spread']:.2%}, 单调: {'是' if risk_q.get('monotonic') else '否'}")

        opp_q70 = regime_df["daily_opportunity_score"].quantile(0.7)
        risk_q30 = regime_df["daily_risk_score"].quantile(0.3)
        selected = regime_df[
            (regime_df["daily_opportunity_score"] >= opp_q70)
            & (regime_df["daily_risk_score"] >= risk_q30)
        ]
        sel_dedup = selected[selected["day_offset"] == 3]
        if not sel_dedup.empty:
            sel_ret5 = sel_dedup["ret_5_close_to_close"].mean()
            sel_win = (sel_dedup["ret_5_close_to_close"] > 0).mean()
            delta = sel_ret5 - avg_ret5
            print(f"\n  模型筛选增量:")
            print(f"    筛选后: avg_ret5={sel_ret5:.2%}, win={sel_win:.0%}, 增量={delta:+.2%}")

        all_results.append({
            "regime": regime_name,
            "n_records": len(regime_df),
            "n_triggers": regime_df["trigger_bar_time"].nunique(),
            "base_ret5": avg_ret5,
            "base_mae5": avg_mae5,
            "base_win": win_rate,
            "opp_total_ic": opp_ic["total_ic"],
            "opp_daily_ic": opp_ic["daily_ic_mean"],
            "opp_icir": opp_ic["icir"],
            "risk_total_ic": risk_ic["total_ic"],
            "risk_daily_ic": risk_ic["daily_ic_mean"],
            "risk_icir": risk_ic["icir"],
            "opp_spread": opp_q.get("spread", np.nan),
            "opp_monotonic": opp_q.get("monotonic", np.nan),
            "risk_spread": risk_q.get("spread", np.nan),
            "risk_monotonic": risk_q.get("monotonic", np.nan),
        })

    print(f"\n{'=' * 80}")
    print("Regime分段汇总")
    print(f"{'=' * 80}")
    summary = pd.DataFrame(all_results)
    print(f"  {'Regime':<22} {'基准ret5':>9} {'opp_ICIR':>9} {'opp_spread':>11} {'单调':>4} {'risk_ICIR':>9} {'risk_spread':>11}")
    print(f"  {'-' * 80}")
    for _, row in summary.iterrows():
        mono = "是" if row.get("opp_monotonic") else "否"
        print(f"  {row['regime']:<22} {row['base_ret5']:>8.2%} {row['opp_icir']:>9.2f} {row['opp_spread']:>10.2%} {mono:>4} {row['risk_icir']:>9.2f} {row['risk_spread']:>10.2%}")

    summary.to_csv(os.path.join(OUTPUT_DIR, "regime_validation_report.csv"), index=False)
    print(f"\n已保存: {os.path.join(OUTPUT_DIR, 'regime_validation_report.csv')}")

    print(f"\n{'=' * 80}")
    print("Regime分段验证完成")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
