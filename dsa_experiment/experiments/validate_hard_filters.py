#!/usr/bin/env python3
"""
硬过滤效果验证：对比有/无硬过滤的 GBDT top20 效果

Purpose: 验证5项硬过滤是否有效，是否在"错杀"
Inputs: candidate_with_scores.parquet
Outputs: 终端报告
How to Run:
    python dsa_experiment/validate_hard_filters.py
Examples:
    python dsa_experiment/validate_hard_filters.py
Side Effects: 只读操作
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

HARD_FILTERS = {
    "trend_align_momo == 1": lambda df: df["trend_align_momo"] == 1,
    "ret_to_last_high_pct < -0.05": lambda df: df["ret_to_last_high_pct"] < -0.05,
    "bbmacd_state >= 0": lambda df: df["bbmacd_state"] >= 0,
    "ret_to_last_low_pct < 0.5": lambda df: df["ret_to_last_low_pct"] < 0.5,
    "current_stage_amp_pct < 2.0": lambda df: df["current_stage_amp_pct"] < 2.0,
}

COST_BUY = 0.002
COST_SELL = 0.002


def apply_all_hard_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for name, fn in HARD_FILTERS.items():
        mask &= fn(df)
    return df[mask]


def compute_stats_for_group(df: pd.DataFrame, top_n: int = 20) -> dict:
    if df.empty:
        return {}

    daily_groups = df.groupby("selection_date")
    all_rets = []
    all_mae = []
    all_mfe = []
    all_stop = []
    n_periods = 0
    nav_values = [1.0]

    for sel_date, day_df in daily_groups:
        if len(day_df) < 5:
            continue
        top = day_df.sort_values("return_score", ascending=False).head(top_n)
        if top.empty:
            continue

        ret = top["ret_5_open_to_open"].mean()
        net_ret = ret - COST_BUY - COST_SELL
        all_rets.append(ret)
        all_mae.append(top["mae_5"].mean())
        all_mfe.append(top["mfe_5"].mean())
        all_stop.append(top["stop_hit_5"].mean())
        nav_values.append(nav_values[-1] * (1 + net_ret))
        n_periods += 1

    if not all_rets:
        return {}

    rets = np.array(all_rets)
    nav = np.array(nav_values)
    total_ret = nav[-1] / nav[0] - 1
    ann_ret = (1 + total_ret) ** (52 / n_periods) - 1 if n_periods > 0 else 0
    max_dd = (nav / np.maximum.accumulate(nav) - 1).min()
    win_rate = (rets > 0).mean()
    avg_ret = rets.mean()

    return {
        "n_periods": n_periods,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "avg_ret_5": avg_ret,
        "avg_mae_5": np.mean(all_mae),
        "avg_mfe_5": np.mean(all_mfe),
        "avg_stop_rate": np.mean(all_stop),
    }


def main():
    print("=" * 80)
    print("硬过滤效果验证")
    print("=" * 80)

    input_path = os.path.join(OUTPUT_DIR, "candidate_with_scores.parquet")
    df = pd.read_parquet(input_path)
    print(f"\n  原始记录: {len(df)}")

    tradeable = df[df["can_buy_next_open"] == True].copy()
    tradeable = tradeable[tradeable["return_score"].notna()]
    print(f"  可交易且有预测分: {len(tradeable)}")

    print("\n" + "=" * 80)
    print("1. 逐项硬过滤效果")
    print("=" * 80)

    print(f"\n  {'过滤条件':<35} {'过滤前':>8} {'过滤后':>8} {'保留率':>7} {'过滤后胜率':>9} {'过滤后收益':>9}")
    print(f"  {'-'*80}")

    baseline_stats = compute_stats_for_group(tradeable)
    print(f"  {'(无过滤) baseline':<35} {len(tradeable):>8} {len(tradeable):>8} {'100%':>7} {baseline_stats['win_rate']:>8.0%} {baseline_stats['avg_ret_5']:>8.2%}")

    for name, fn in HARD_FILTERS.items():
        filtered = tradeable[fn(tradeable)]
        if len(filtered) < 100:
            print(f"  {name:<35} {len(tradeable):>8} {len(filtered):>8} {len(filtered)/len(tradeable):>6.0%}    (样本不足)")
            continue
        stats = compute_stats_for_group(filtered)
        print(f"  {name:<35} {len(tradeable):>8} {len(filtered):>8} {len(filtered)/len(tradeable):>6.0%} {stats['win_rate']:>8.0%} {stats['avg_ret_5']:>8.2%}")

    print("\n" + "=" * 80)
    print("2. 组合硬过滤效果（逐步叠加）")
    print("=" * 80)

    cumulative = tradeable.copy()
    print(f"\n  {'步骤':<40} {'保留数':>8} {'保留率':>7} {'胜率':>6} {'收益':>8} {'年化':>8} {'回撤':>8}")
    print(f"  {'-'*90}")

    stats = compute_stats_for_group(cumulative)
    print(f"  {'(0) 无过滤':<40} {len(cumulative):>8} {'100%':>7} {stats['win_rate']:>5.0%} {stats['avg_ret_5']:>7.2%} {stats['ann_return']:>7.0%} {stats['max_drawdown']:>7.0%}")

    for i, (name, fn) in enumerate(HARD_FILTERS.items(), 1):
        cumulative = cumulative[fn(cumulative)]
        if len(cumulative) < 100:
            print(f"  ({i}) + {name:<37} {len(cumulative):>8} {len(cumulative)/len(tradeable):>6.0%}    (样本不足)")
            break
        stats = compute_stats_for_group(cumulative)
        print(f"  ({i}) + {name:<37} {len(cumulative):>8} {len(cumulative)/len(tradeable):>6.0%} {stats['win_rate']:>5.0%} {stats['avg_ret_5']:>7.2%} {stats['ann_return']:>7.0%} {stats['max_drawdown']:>7.0%}")

    print("\n" + "=" * 80)
    print("3. 关键对比：有硬过滤 vs 无硬过滤（GBDT top20）")
    print("=" * 80)

    no_filter = tradeable.copy()
    with_filter = apply_all_hard_filters(tradeable)

    stats_no = compute_stats_for_group(no_filter)
    stats_with = compute_stats_for_group(with_filter)

    print(f"\n  {'指标':<20} {'无硬过滤':>12} {'有硬过滤':>12} {'差异':>12}")
    print(f"  {'-'*60}")
    print(f"  {'样本数':<20} {len(no_filter):>12} {len(with_filter):>12} {len(with_filter)-len(no_filter):>+12}")
    print(f"  {'保留率':<20} {'100%':>12} {len(with_filter)/len(no_filter):>11.0%} {len(with_filter)/len(no_filter)-1:>+11.0%}")
    print(f"  {'胜率':<20} {stats_no['win_rate']:>11.0%} {stats_with['win_rate']:>11.0%} {stats_with['win_rate']-stats_no['win_rate']:>+11.0%}")
    print(f"  {'平均收益':<20} {stats_no['avg_ret_5']:>11.2%} {stats_with['avg_ret_5']:>11.2%} {stats_with['avg_ret_5']-stats_no['avg_ret_5']:>+11.2%}")
    print(f"  {'年化收益':<20} {stats_no['ann_return']:>11.0%} {stats_with['ann_return']:>11.0%} {stats_with['ann_return']-stats_no['ann_return']:>+11.0%}")
    print(f"  {'最大回撤':<20} {stats_no['max_drawdown']:>11.0%} {stats_with['max_drawdown']:>11.0%} {stats_with['max_drawdown']-stats_no['max_drawdown']:>+11.0%}")
    print(f"  {'止损率':<20} {stats_no['avg_stop_rate']:>11.0%} {stats_with['avg_stop_rate']:>11.0%} {stats_with['avg_stop_rate']-stats_no['avg_stop_rate']:>+11.0%}")
    print(f"  {'MFE':<20} {stats_no['avg_mfe_5']:>11.2%} {stats_with['avg_mfe_5']:>11.2%} {stats_with['avg_mfe_5']-stats_no['avg_mfe_5']:>+11.2%}")
    print(f"  {'MAE':<20} {stats_no['avg_mae_5']:>11.2%} {stats_with['avg_mae_5']:>11.2%} {stats_with['avg_mae_5']-stats_no['avg_mae_5']:>+11.2%}")

    print("\n  判断:")
    if stats_with['ann_return'] > stats_no['ann_return'] and stats_with['win_rate'] > stats_no['win_rate']:
        print("  ✅ 硬过滤有效：提高胜率和年化收益，应保留")
    elif stats_with['ann_return'] > stats_no['ann_return']:
        print("  ⚠️ 硬过滤部分有效：提高年化但胜率未提升")
    elif stats_with['avg_stop_rate'] < stats_no['avg_stop_rate']:
        print("  ⚠️ 硬过滤主要降低风险：降低止损率但牺牲收益")
    else:
        print("  ❌ 硬过滤无效或有害：降低收益且未改善风险，应移除")

    print("\n" + "=" * 80)
    print("4. 错杀分析：被硬过滤删除但实际收益好的样本")
    print("=" * 80)

    filtered = apply_all_hard_filters(tradeable)
    removed = tradeable[~tradeable.index.isin(filtered.index)]
    removed_good = removed[removed["ret_5_open_to_open"] > 0.05]
    removed_bad = removed[removed["stop_hit_5"] == 1]

    print(f"\n  被硬过滤删除: {len(removed)} 条")
    print(f"  其中收益>5%: {len(removed_good)} 条 ({len(removed_good)/len(removed):.1%})")
    print(f"  其中触发止损: {len(removed_bad)} 条 ({len(removed_bad)/len(removed):.1%})")

    if len(removed) > 0:
        removed_top20_rets = []
        for sel_date, day_df in removed.groupby("selection_date"):
            top = day_df.sort_values("return_score", ascending=False).head(20)
            if not top.empty:
                removed_top20_rets.append(top["ret_5_open_to_open"].mean())

        if removed_top20_rets:
            avg_removed_top20 = np.mean(removed_top20_rets)
            print(f"  被删除样本的GBDT top20平均收益: {avg_removed_top20:.2%}")
            if avg_removed_top20 > stats_no['avg_ret_5']:
                print("  ⚠️ 被删除样本的top20收益高于全量，硬过滤可能在错杀")
            else:
                print("  ✅ 被删除样本的top20收益低于全量，硬过滤合理")

    print("\n" + "=" * 80)
    print("硬过滤效果验证完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
