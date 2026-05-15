#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新概念涌现因子离线分析

Purpose:
    验证 stop_loss 候选池中"新冒出来的概念"是否对股票未来收益有预测力。
    不修改 pipeline，仅基于已有 candidate_with_scores.parquet + stock_pools 概念映射做分析。

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet
    - DB: stock_pools (via theme_aggregator.load_concept_map)

Outputs:
    - results/analysis/concept_coverage.csv
    - results/analysis/new_concept_events.csv
    - results/analysis/return_comparison.csv
    - results/analysis/ic_analysis.csv
    - results/analysis/independence_analysis.csv
    - results/analysis/monthly_stability.csv

How to Run:
    python -m stop_experiment.experiments.new_concept_emergence.01_new_concept_analysis

Examples:
    python -m stop_experiment.experiments.new_concept_emergence.01_new_concept_analysis
    python -m stop_experiment.experiments.new_concept_emergence.01_new_concept_analysis --lookback 10

Side Effects:
    - 只读 candidate_with_scores.parquet 和 DB stock_pools
    - 输出仅写入 results/analysis/

Limitations:
    - stock_pools.concepts 是当前快照，非历史数据，存在前视偏差风险
    - 涌现定义基于候选池历史统计，对概念归属快照的依赖较弱
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from scipy import stats

from stop_experiment.pipeline.stop_config import OBS_VAL_END, MODELS_DIR

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

EMERGENCE_DEFS = {
    "A_absent_5": {"method": "absent", "lookback": 5},
    "A_absent_10": {"method": "absent", "lookback": 10},
    "A_absent_20": {"method": "absent", "lookback": 20},
    "B_surge_2x": {"method": "surge", "lookback": 10, "surge_k": 2},
    "B_surge_3x": {"method": "surge", "lookback": 10, "surge_k": 3},
    "C_count_2": {"method": "count_threshold", "lookback": 5, "min_count": 2},
    "C_count_3": {"method": "count_threshold", "lookback": 5, "min_count": 3},
}


def load_concept_map_safe() -> pd.DataFrame:
    from market_structure_analysis.theme_aggregator import load_concept_map
    return load_concept_map()


def build_daily_concept_stats(df_entry: pd.DataFrame, concept_map: pd.DataFrame) -> pd.DataFrame:
    df_with_concept = df_entry.merge(concept_map, on="ts_code", how="inner")
    daily_concept = (
        df_with_concept
        .groupby(["obs_date", "concept"])
        .agg(
            stock_count=("ts_code", "nunique"),
            avg_sell_reg=("pred_sell_reg", "mean"),
            avg_sell_cls=("pred_sell_cls", "mean"),
            avg_mfe_20=("mfe_20", "mean"),
        )
        .reset_index()
    )
    return daily_concept


def identify_emergent_concepts(
    daily_concept: pd.DataFrame,
    method: str,
    lookback: int = 10,
    surge_k: float = 2.0,
    min_count: int = 2,
) -> pd.DataFrame:
    all_dates = sorted(daily_concept["obs_date"].unique())
    if len(all_dates) < lookback + 1:
        return pd.DataFrame()

    concept_daily_pivot = daily_concept.pivot_table(
        index="obs_date", columns="concept", values="stock_count", fill_value=0,
    )
    concept_daily_pivot = concept_daily_pivot.reindex(all_dates, fill_value=0)

    lb_stock_mean = concept_daily_pivot.rolling(window=lookback, min_periods=1).mean()
    lb_stock_mean = lb_stock_mean.shift(1)
    lb_days_present = (concept_daily_pivot > 0).rolling(window=lookback, min_periods=1).sum()
    lb_days_present = lb_days_present.shift(1)

    results = []
    for current_date in all_dates:
        if current_date not in concept_daily_pivot.index:
            continue
        current_counts = concept_daily_pivot.loc[current_date]
        current_counts = current_counts[current_counts > 0]
        if current_counts.empty:
            continue

        lb_mean = lb_stock_mean.loc[current_date] if current_date in lb_stock_mean.index else pd.Series(0, index=current_counts.index)
        lb_days = lb_days_present.loc[current_date] if current_date in lb_days_present.index else pd.Series(0, index=current_counts.index)

        common_concepts = current_counts.index.intersection(lb_mean.index)
        if len(common_concepts) == 0:
            continue

        curr = current_counts.loc[common_concepts]
        mean_lb = lb_mean.loc[common_concepts].fillna(0)
        days_lb = lb_days.loc[common_concepts].fillna(0)

        if method == "absent":
            mask = days_lb == 0
        elif method == "surge":
            mask = (mean_lb > 0) & (curr >= mean_lb * surge_k)
        elif method == "count_threshold":
            mask = (curr >= min_count) & (mean_lb < 1)
        else:
            continue

        emergent_concepts_list = common_concepts[mask]
        if len(emergent_concepts_list) == 0:
            continue

        emergent_df = pd.DataFrame({
            "obs_date": current_date,
            "concept": emergent_concepts_list,
            "stock_count": curr.loc[emergent_concepts_list].values,
            "lb_stock_count_mean": mean_lb.loc[emergent_concepts_list].values,
            "lb_days_present": days_lb.loc[emergent_concepts_list].values,
            "method": method,
            "lookback": lookback,
        })
        results.append(emergent_df)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def tag_new_concept_stocks(
    df_entry: pd.DataFrame,
    concept_map: pd.DataFrame,
    emergent_concepts: pd.DataFrame,
) -> pd.DataFrame:
    if emergent_concepts.empty:
        df_entry = df_entry.copy()
        df_entry["is_new_concept"] = False
        return df_entry

    emergent_flag = emergent_concepts[["obs_date", "concept"]].drop_duplicates()
    emergent_flag["is_emergent"] = True

    df_with_concept = df_entry[["signal_id", "obs_date", "ts_code"]].merge(
        concept_map, on="ts_code", how="left"
    )
    df_with_concept = df_with_concept.merge(
        emergent_flag, on=["obs_date", "concept"], how="left"
    )
    df_with_concept["is_emergent"] = df_with_concept["is_emergent"].fillna(False)

    stock_level = (
        df_with_concept
        .groupby(["signal_id", "obs_date", "ts_code"])
        .agg(is_new_concept=("is_emergent", "any"))
        .reset_index()
    )

    result = df_entry.merge(
        stock_level[["signal_id", "obs_date", "ts_code", "is_new_concept"]],
        on=["signal_id", "obs_date", "ts_code"],
        how="left",
    )
    result["is_new_concept"] = result["is_new_concept"].fillna(False)
    return result


def compute_return_metrics(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0, "avg_mfe_20": np.nan, "avg_mae_20": np.nan,
                "pct_rise": np.nan, "pct_profitable": np.nan,
                "median_mfe_20": np.nan, "median_mae_20": np.nan}
    avg_mfe = sub["mfe_20"].mean()
    avg_mae = sub["mae_20"].mean()
    pct_rise = (sub["mfe_20"] > 0).mean()
    net_ret = sub["mfe_20"] + sub["mae_20"]
    pct_profitable = (net_ret > 0).mean()
    return {"n": n, "avg_mfe_20": avg_mfe, "avg_mae_20": avg_mae,
            "pct_rise": pct_rise, "pct_profitable": pct_profitable,
            "median_mfe_20": sub["mfe_20"].median(),
            "median_mae_20": sub["mae_20"].median()}


def main():
    parser = argparse.ArgumentParser(description="新概念涌现因子离线分析")
    parser.add_argument("--lookback", type=int, default=None, help="仅运行指定 lookback 的涌现定义")
    args = parser.parse_args()

    print("=" * 60)
    print("新概念涌现因子离线分析")
    print("=" * 60)
    print()
    print("⚠ 数据限制: stock_pools.concepts 是当前快照，非历史数据")
    print("  涌现定义基于候选池历史统计，对概念归属快照的依赖较弱")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("\n[1/6] 加载数据 + 概念映射...")
    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在")

    df = pd.read_parquet(scores_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.dropna(subset=["mfe_20", "mae_20"])
    print(f"  总行数: {len(df)}, 股票数: {df['ts_code'].nunique()}")

    concept_map = load_concept_map_safe()
    print(f"  概念映射: {concept_map['concept'].nunique()} 个概念, {concept_map['ts_code'].nunique()} 只股票")

    covered = df[df["ts_code"].isin(concept_map["ts_code"])]
    coverage = len(covered) / len(df)
    print(f"  候选池概念覆盖率: {len(covered)}/{len(df)} = {coverage:.1%}")

    val_end = pd.Timestamp(OBS_VAL_END)
    test = df[df["obs_date"] > val_end].copy()
    test_entry = test[test["obs_day"].isin([1, 2, 3])].copy()
    print(f"  test 集: {len(test)} 行, 入场样本(obs_day 1~3): {len(test_entry)}")

    coverage_rows = [
        {"metric": "total_rows", "value": len(df)},
        {"metric": "total_stocks", "value": df["ts_code"].nunique()},
        {"metric": "concept_count", "value": concept_map["concept"].nunique()},
        {"metric": "concept_covered_rows", "value": len(covered)},
        {"metric": "concept_coverage", "value": coverage},
        {"metric": "test_entry_rows", "value": len(test_entry)},
    ]
    pd.DataFrame(coverage_rows).to_csv(
        os.path.join(ANALYSIS_DIR, "concept_coverage.csv"), index=False
    )

    print("\n[2/6] 新概念涌现事件识别...")
    daily_concept = build_daily_concept_stats(test_entry, concept_map)
    print(f"  候选池每日概念统计: {len(daily_concept)} 行, {daily_concept['concept'].nunique()} 个概念")

    all_events = []
    event_summary = []
    for def_name, def_params in EMERGENCE_DEFS.items():
        if args.lookback is not None and def_params.get("lookback") != args.lookback:
            continue

        emergent = identify_emergent_concepts(daily_concept, **def_params)
        n_events = len(emergent)
        n_concepts = emergent["concept"].nunique() if n_events > 0 else 0
        n_dates = emergent["obs_date"].nunique() if n_events > 0 else 0
        n_stocks_total = 0
        if n_events > 0:
            emergent_flag_df = emergent[["obs_date", "concept"]].drop_duplicates()
            candidate_concepts = concept_map.merge(emergent_flag_df, on="concept", how="inner")
            matched_ts = test_entry.merge(candidate_concepts, on=["ts_code", "obs_date"], how="inner")
            n_stocks_total = matched_ts["ts_code"].nunique()

        event_summary.append({
            "definition": def_name,
            "n_events": n_events,
            "n_concepts": n_concepts,
            "n_dates": n_dates,
            "n_unique_stocks": n_stocks_total,
        })

        if n_events > 0:
            emergent["definition"] = def_name
            all_events.append(emergent)

        print(f"  {def_name}: {n_events} 事件, {n_concepts} 概念, {n_dates} 天, {n_stocks_total} 只股票")

    event_df = pd.DataFrame(event_summary)
    event_df.to_csv(os.path.join(ANALYSIS_DIR, "new_concept_events.csv"), index=False)

    if all_events:
        all_events_df = pd.concat(all_events, ignore_index=True)
        all_events_df.to_csv(os.path.join(ANALYSIS_DIR, "new_concept_events_detail.csv"), index=False)

    print("\n[3/6] 新概念股票 vs 非新概念股票收益对比...")
    comparison_rows = []
    best_def = None
    best_spread = -999

    for def_name, def_params in EMERGENCE_DEFS.items():
        if args.lookback is not None and def_params.get("lookback") != args.lookback:
            continue

        emergent = identify_emergent_concepts(daily_concept, **def_params)
        tagged = tag_new_concept_stocks(test_entry, concept_map, emergent)

        new_concept = tagged[tagged["is_new_concept"]]
        old_concept = tagged[~tagged["is_new_concept"]]

        new_metrics = compute_return_metrics(new_concept)
        old_metrics = compute_return_metrics(old_concept)

        comparison_rows.append({"definition": def_name, "group": "new_concept", **new_metrics})
        comparison_rows.append({"definition": def_name, "group": "other", **old_metrics})

        spread = new_metrics["avg_mfe_20"] - old_metrics["avg_mfe_20"]
        if spread > best_spread and new_metrics["n"] >= 30:
            best_spread = spread
            best_def = def_name

        print(f"  {def_name}:")
        print(f"    新概念: n={new_metrics['n']}, avg_mfe={new_metrics['avg_mfe_20']:.4f}, "
              f"avg_mae={new_metrics['avg_mae_20']:.4f}, 上涨率={new_metrics['pct_rise']:.1%}")
        print(f"    其他:   n={old_metrics['n']}, avg_mfe={old_metrics['avg_mfe_20']:.4f}, "
              f"avg_mae={old_metrics['avg_mae_20']:.4f}, 上涨率={old_metrics['pct_rise']:.1%}")
        print(f"    MFE spread: {spread:+.4f}")

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(ANALYSIS_DIR, "return_comparison.csv"), index=False)

    if best_def:
        print(f"\n  最优涌现定义: {best_def} (MFE spread={best_spread:+.4f})")
    else:
        print(f"\n  ⚠ 无满足最小样本(30)的涌现定义")

    print("\n[4/6] 新概念信号 IC 分析...")
    ic_rows = []

    for def_name, def_params in EMERGENCE_DEFS.items():
        if args.lookback is not None and def_params.get("lookback") != args.lookback:
            continue

        emergent = identify_emergent_concepts(daily_concept, **def_params)
        tagged = tag_new_concept_stocks(test_entry, concept_map, emergent)

        valid = tagged[tagged["is_new_concept"].notna() & tagged["mfe_20"].notna()]
        if len(valid) < 100 or valid["is_new_concept"].sum() < 10:
            ic_rows.append({
                "definition": def_name, "label": "mfe_20",
                "ic_spearman": np.nan, "p_spearman": np.nan, "n": len(valid),
                "n_new_concept": int(valid["is_new_concept"].sum()),
            })
            continue

        ic_sp, p_sp = stats.spearmanr(valid["is_new_concept"].astype(float), valid["mfe_20"])
        ic_rows.append({
            "definition": def_name, "label": "mfe_20",
            "ic_spearman": ic_sp, "p_spearman": p_sp, "n": len(valid),
            "n_new_concept": int(valid["is_new_concept"].sum()),
        })

        valid_mae = valid[valid["mae_20"].notna()]
        if len(valid_mae) >= 100:
            ic_sp_mae, p_sp_mae = stats.spearmanr(
                valid_mae["is_new_concept"].astype(float), valid_mae["mae_20"]
            )
            ic_rows.append({
                "definition": def_name, "label": "mae_20",
                "ic_spearman": ic_sp_mae, "p_spearman": p_sp_mae, "n": len(valid_mae),
                "n_new_concept": int(valid_mae["is_new_concept"].sum()),
            })

    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(os.path.join(ANALYSIS_DIR, "ic_analysis.csv"), index=False)

    print(f"\n  {'定义':15s} {'标签':10s} {'IC(Spearman)':>14s} {'p值':>12s} {'样本':>8s} {'新概念':>8s}")
    for _, row in ic_df.iterrows():
        ic_val = row["ic_spearman"]
        p_val = row["p_spearman"]
        n_new = int(row.get("n_new_concept", 0))
        if pd.notna(ic_val):
            print(f"  {row['definition']:15s} {row['label']:10s} {ic_val:>14.4f} {p_val:>12.2e} "
                  f"{int(row['n']):>8d} {n_new:>8d}")

    print("\n[5/6] 因子独立性分析...")
    if best_def is None:
        best_def = "A_absent_10"

    best_params = EMERGENCE_DEFS[best_def]
    emergent_best = identify_emergent_concepts(daily_concept, **best_params)
    tagged_best = tag_new_concept_stocks(test_entry, concept_map, emergent_best)

    corr_candidates = [
        "pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls",
        "ret_to_trigger", "change_pct", "intraday_range",
        "range_position", "vol_ratio", "obs_day",
    ]
    available_corr = [c for c in corr_candidates if c in tagged_best.columns]
    corr_rows = []

    for col in available_corr:
        valid = tagged_best[tagged_best["is_new_concept"].notna() & tagged_best[col].notna()]
        if len(valid) < 100:
            continue
        sp_corr, sp_p = stats.spearmanr(valid["is_new_concept"].astype(float), valid[col])
        corr_rows.append({
            "factor": col,
            "spearman_corr": sp_corr, "spearman_p": sp_p,
            "n": len(valid),
            "definition": best_def,
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(ANALYSIS_DIR, "independence_analysis.csv"), index=False)

    print(f"\n  最优定义: {best_def}")
    print(f"  {'因子':25s} {'Spearman':>10s} {'p值':>12s} {'样本':>8s}")
    for _, row in corr_df.iterrows():
        print(f"  {row['factor']:25s} {row['spearman_corr']:>10.4f} {row['spearman_p']:>12.2e} {int(row['n']):>8d}")

    high_corr = corr_df[corr_df["spearman_corr"].abs() > 0.3]
    if len(high_corr) > 0:
        print(f"\n  ⚠ 中高相关因子(|r|>0.3): {', '.join(high_corr['factor'].tolist())}")
    else:
        print(f"\n  ✅ 无中高相关因子(|r|>0.3)，is_new_concept 具有独立性")

    print("\n[6/6] 月度稳定性验证...")
    test_m = tagged_best.copy()
    test_m["month"] = test_m["obs_date"].dt.to_period("M")
    monthly_rows = []

    for month, group in test_m.groupby("month"):
        valid = group[group["is_new_concept"].notna() & group["mfe_20"].notna()]
        if len(valid) < 30:
            continue

        n_new = valid["is_new_concept"].sum()
        if n_new < 5:
            monthly_rows.append({
                "month": str(month), "n": len(valid), "n_new_concept": int(n_new),
                "ic_mfe": np.nan, "ic_mae": np.nan,
                "avg_mfe_new": np.nan, "avg_mfe_other": np.nan,
                "mfe_spread": np.nan,
            })
            continue

        ic_sp, _ = stats.spearmanr(valid["is_new_concept"].astype(float), valid["mfe_20"])

        new_c = valid[valid["is_new_concept"]]
        other_c = valid[~valid["is_new_concept"]]

        avg_mfe_new = new_c["mfe_20"].mean() if len(new_c) > 0 else np.nan
        avg_mfe_other = other_c["mfe_20"].mean() if len(other_c) > 0 else np.nan

        ic_sp_mae = np.nan
        valid_mae = valid[valid["mae_20"].notna()]
        if len(valid_mae) >= 30 and valid_mae["is_new_concept"].sum() >= 5:
            ic_sp_mae, _ = stats.spearmanr(valid_mae["is_new_concept"].astype(float), valid_mae["mae_20"])

        monthly_rows.append({
            "month": str(month), "n": len(valid), "n_new_concept": int(n_new),
            "ic_mfe": ic_sp, "ic_mae": ic_sp_mae,
            "avg_mfe_new": avg_mfe_new, "avg_mfe_other": avg_mfe_other,
            "mfe_spread": avg_mfe_new - avg_mfe_other if pd.notna(avg_mfe_new) and pd.notna(avg_mfe_other) else np.nan,
        })

    m_df = pd.DataFrame(monthly_rows)
    m_df.to_csv(os.path.join(ANALYSIS_DIR, "monthly_stability.csv"), index=False)

    print(f"\n  {'月份':10s} {'样本':>6s} {'新概念':>6s} {'IC(mfe)':>10s} {'mfe_new':>10s} {'mfe_other':>10s} {'spread':>10s}")
    for _, row in m_df.iterrows():
        ic_str = f"{row['ic_mfe']:.4f}" if pd.notna(row["ic_mfe"]) else "N/A"
        spread_str = f"{row['mfe_spread']:+.4f}" if pd.notna(row["mfe_spread"]) else "N/A"
        mfe_new_str = f"{row['avg_mfe_new']:.4f}" if pd.notna(row["avg_mfe_new"]) else "N/A"
        mfe_other_str = f"{row['avg_mfe_other']:.4f}" if pd.notna(row["avg_mfe_other"]) else "N/A"
        print(f"  {row['month']:10s} {int(row['n']):>6d} {int(row['n_new_concept']):>6d} "
              f"{ic_str:>10s} {mfe_new_str:>10s} {mfe_other_str:>10s} {spread_str:>10s}")

    valid_months = m_df[m_df["ic_mfe"].notna()]
    ic_sign_consistency = (valid_months["ic_mfe"] > 0).mean() if len(valid_months) > 0 else 0
    spread_positive = (valid_months["mfe_spread"] > 0).mean() if len(valid_months) > 0 else 0
    print(f"\n  IC(mfe) 月度正比例: {ic_sign_consistency:.1%}")
    print(f"  MFE spread 月度正比例: {spread_positive:.1%}")

    print(f"\n{'='*60}")
    print("新概念涌现分析结论")
    print(f"{'='*60}")

    overall_ic = ic_df[ic_df["label"] == "mfe_20"].dropna(subset=["ic_spearman"])
    if len(overall_ic) > 0:
        best_ic_row = overall_ic.loc[overall_ic["ic_spearman"].abs().idxmax()]
        ic_val = best_ic_row["ic_spearman"]
        best_ic_def = best_ic_row["definition"]
        print(f"\n  最优 IC(mfe_20): {ic_val:.4f} (定义: {best_ic_def})")
        if abs(ic_val) > 0.03:
            print(f"  ✅ IC > 0.03，有预测力")
        else:
            print(f"  ❌ IC < 0.03，预测力弱")

    if best_def and best_spread > 0:
        print(f"  ✅ 新概念股票 MFE 高于非新概念 (spread={best_spread:+.4f})")
    elif best_def:
        print(f"  ❌ 新概念股票 MFE 未高于非新概念 (spread={best_spread:+.4f})")

    if len(high_corr) == 0:
        print(f"  ✅ 与现有因子无中高相关，具有独立性")
    else:
        print(f"  ⚠ 与 {', '.join(high_corr['factor'].tolist())} 中高相关，独立性存疑")

    if ic_sign_consistency > 0.7:
        print(f"  ✅ IC 月度正比例 {ic_sign_consistency:.0%} > 70%，信号稳定")
    else:
        print(f"  ⚠ IC 月度正比例 {ic_sign_consistency:.0%} < 70%，信号不稳定")

    print(f"\n  ⚠ 前视偏差风险: 概念映射使用当前快照，非历史数据")
    print(f"     若结果显著，建议实现 instrument_snapshot 每日写入后重新验证")


if __name__ == "__main__":
    main()
