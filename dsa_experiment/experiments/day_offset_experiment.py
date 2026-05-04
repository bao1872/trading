#!/usr/bin/env python3
"""
day_offset 价值验证实验

Purpose: 验证 day_offset 作为特征 vs 不作为特征 vs 分组训练的效果差异
Inputs: output/daily_factor_table.parquet
Outputs: output/day_offset_experiment_results.csv, 控制台报告
How to Run:
    python dsa_experiment/day_offset_experiment.py
    python dsa_experiment/day_offset_experiment.py --sample-limit 50000
Examples:
    python dsa_experiment/day_offset_experiment.py
    python dsa_experiment/day_offset_experiment.py --sample-limit 50000
Side Effects: 只读操作，输出CSV到 dsa_experiment/output/
"""

import sys
import os
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

FULL_FEATURES = [
    "dsa_dir", "prev_pivot_code", "dsa_pivot_pos_01",
    "ret_to_last_low_pct", "bars_since_last_high",
    "price_vs_dsa_vwap_pct", "current_stage_amp_pct", "prev_stage_amp_pct",
    "current_stage_ret_pct", "bbmacd", "bbmacd_minus_avg",
    "bbmacd_bandwidth_zscore", "bbmacd_slope_3", "bbmacd_sign",
    "trend_align_momo",
    "vol_zscore_5", "vol_zscore_10", "vol_zscore_20",
    "vol_ratio_10", "vol_stage_cv",
    "price_vol_coord", "coord_stage_current",
    "momo_vol_coord", "coord_consistency",
    "weekly_return_score", "weekly_risk_score",
]

DERIVED = {
    "pivot_pos_x_trend": lambda df: df["dsa_pivot_pos_01"] * df["dsa_dir"],
    "amp_x_pullback": lambda df: df["current_stage_amp_pct"] * df.get("current_pullback_from_stage_extreme_pct", 0),
    "vol_x_stage_amp": lambda df: df["vol_zscore_20"] * df["current_stage_amp_pct"],
}

OPP_TARGET = "ret_5_close_to_close"
RISK_TARGET = "mae_5"
EMBARGO_DAYS = 25

REG_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "seed": 42,
    "num_leaves": 16,
    "max_depth": 5,
    "min_data_in_leaf": 100,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}


def build_rolling_splits(df, n_folds=3):
    trigger_groups = df.groupby("trigger_bar_time")
    trigger_dates = df.groupby("trigger_bar_time")["selection_date"].first()
    sorted_triggers = trigger_dates.sort_values()
    unique_triggers = sorted_triggers.index.tolist()
    n = len(unique_triggers)
    fold_size = n // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_end_idx = fold_size * (i + 1)
        test_end_idx = fold_size * (i + 2) if i < n_folds - 1 else n
        if train_end_idx >= n or test_end_idx > n:
            continue
        train_triggers = set(unique_triggers[:train_end_idx])
        test_triggers = set(unique_triggers[train_end_idx:test_end_idx])
        embargo_cutoff = pd.Timestamp(sorted_triggers.iloc[train_end_idx]) - pd.Timedelta(days=EMBARGO_DAYS)
        train_mask = df["trigger_bar_time"].isin(train_triggers) & (pd.to_datetime(df["selection_date"]) <= embargo_cutoff)
        test_mask = df["trigger_bar_time"].isin(test_triggers)
        train_idx = df.index[train_mask].values
        test_idx = df.index[test_mask].values
        if len(train_idx) < 200 or len(test_idx) < 100:
            continue
        folds.append((train_idx, test_idx))
    return folds


def train_and_eval(df, feature_cols, target_col, folds):
    ics = []
    maes = []
    for train_idx, test_idx in folds:
        train_data = lgb.Dataset(df.loc[train_idx, feature_cols], df.loc[train_idx, target_col], free_raw_data=False)
        test_data = lgb.Dataset(df.loc[test_idx, feature_cols], df.loc[test_idx, target_col], reference=train_data, free_raw_data=False)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(REG_PARAMS, train_data, num_boost_round=1000, valid_sets=[test_data], callbacks=callbacks)
        y_test = df.loc[test_idx, target_col]
        y_pred = model.predict(df.loc[test_idx, feature_cols], num_iteration=model.best_iteration)
        valid = y_test.notna()
        y_v, p_v = y_test[valid].values, y_pred[valid]
        if len(y_v) > 10:
            ics.append(stats.spearmanr(y_v, p_v)[0])
        maes.append(mean_absolute_error(y_v, p_v))
    ic_mean = np.mean(ics) if ics else np.nan
    ic_std = np.std(ics) if ics else np.nan
    return {"ic": ic_mean, "icir": ic_mean / ic_std if ic_std > 0 else np.nan, "mae": np.mean(maes) if maes else np.nan}


def main():
    parser = argparse.ArgumentParser(description="day_offset价值验证实验")
    parser.add_argument("--sample-limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("day_offset 价值验证实验")
    print("=" * 80)

    df = pd.read_parquet(os.path.join(OUTPUT_DIR, "daily_factor_table.parquet"))
    if args.sample_limit > 0:
        df = df.head(args.sample_limit)
    print(f"  记录数: {len(df)}")

    for name, func in DERIVED.items():
        try:
            df[name] = func(df)
        except Exception:
            df[name] = np.nan

    all_features_with_offset = FULL_FEATURES + ["day_offset"] + list(DERIVED.keys())
    all_features_no_offset = FULL_FEATURES + list(DERIVED.keys())
    available_with = [c for c in all_features_with_offset if c in df.columns]
    available_no = [c for c in all_features_no_offset if c in df.columns]

    df = df.sort_values("selection_date").reset_index(drop=True)
    folds = build_rolling_splits(df)
    print(f"  有效折数: {len(folds)}")

    results = []

    # E1: day_offset作为特征
    print("\n[E1] day_offset作为特征...")
    e1_opp = train_and_eval(df, available_with, OPP_TARGET, folds)
    e1_risk = train_and_eval(df, available_with, RISK_TARGET, folds)
    results.append({"group": "E1_with_offset", "n_features": len(available_with),
                     "opp_ic": e1_opp["ic"], "opp_icir": e1_opp["icir"],
                     "risk_ic": e1_risk["ic"], "risk_icir": e1_risk["icir"]})
    print(f"  机会IC={e1_opp['ic']:.4f}, 风险IC={e1_risk['ic']:.4f}")

    # E2: 去掉day_offset
    print("\n[E2] 去掉day_offset...")
    e2_opp = train_and_eval(df, available_no, OPP_TARGET, folds)
    e2_risk = train_and_eval(df, available_no, RISK_TARGET, folds)
    results.append({"group": "E2_no_offset", "n_features": len(available_no),
                     "opp_ic": e2_opp["ic"], "opp_icir": e2_opp["icir"],
                     "risk_ic": e2_risk["ic"], "risk_icir": e2_risk["icir"]})
    print(f"  机会IC={e2_opp['ic']:.4f}, 风险IC={e2_risk['ic']:.4f}")

    # E3: 按day_offset分组训练
    print("\n[E3] 按day_offset分组训练...")
    for offset in range(1, 6):
        sub = df[df["day_offset"] == offset].copy().reset_index(drop=True)
        sub_features = [c for c in available_no if c in sub.columns]
        sub_folds = build_rolling_splits(sub)
        if len(sub_folds) < 2:
            print(f"  T+{offset}: 折数不足，跳过")
            continue
        e3_opp = train_and_eval(sub, sub_features, OPP_TARGET, sub_folds)
        e3_risk = train_and_eval(sub, sub_features, RISK_TARGET, sub_folds)
        results.append({"group": f"E3_offset_{offset}", "n_features": len(sub_features),
                         "opp_ic": e3_opp["ic"], "opp_icir": e3_opp["icir"],
                         "risk_ic": e3_risk["ic"], "risk_icir": e3_risk["icir"]})
        print(f"  T+{offset}: 机会IC={e3_opp['ic']:.4f}, 风险IC={e3_risk['ic']:.4f}")

    # 汇总
    print(f"\n{'=' * 80}")
    print("day_offset 实验汇总")
    print(f"{'=' * 80}")
    print(f"{'组':<20} {'特征数':>4} {'机会IC':>8} {'机会ICIR':>8} {'风险IC':>8} {'风险ICIR':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['group']:<20} {r['n_features']:>4} {r['opp_ic']:>8.4f} {r['opp_icir']:>8.2f} {r['risk_ic']:>8.4f} {r['risk_icir']:>8.2f}")

    if len(results) >= 2:
        delta_opp = results[0]["opp_ic"] - results[1]["opp_ic"]
        delta_risk = results[0]["risk_ic"] - results[1]["risk_ic"]
        print(f"\n  day_offset增量: 机会IC={delta_opp:+.4f}, 风险IC={delta_risk:+.4f}")
        if abs(delta_opp) < 0.005 and abs(delta_risk) < 0.005:
            print("  结论: day_offset贡献极小，可考虑移除")
        elif abs(delta_opp) < 0.01 and abs(delta_risk) < 0.01:
            print("  结论: day_offset有微弱贡献，保留但不依赖")
        else:
            print("  结论: day_offset有显著贡献，建议保留")

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "day_offset_experiment_results.csv"), index=False)
    print(f"\n已保存: {os.path.join(OUTPUT_DIR, 'day_offset_experiment_results.csv')}")


if __name__ == "__main__":
    main()
