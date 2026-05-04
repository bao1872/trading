#!/usr/bin/env python3
"""
因子删减对比实验（ablation）：量化量能因子和协同因子的增量价值

Purpose: 4组因子删减实验，对比基线/量能/协同/全量的模型效果
Inputs: output/daily_factor_table.parquet
Outputs: output/ablation_results.csv, 控制台报告
How to Run:
    python dsa_experiment/ablation_experiment.py
    python dsa_experiment/ablation_experiment.py --sample-limit 50000
Examples:
    python dsa_experiment/ablation_experiment.py
    python dsa_experiment/ablation_experiment.py --sample-limit 50000
Side Effects: 只读操作，输出CSV到 dsa_experiment/output/
"""

import sys
import os
import argparse
import warnings
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

BASE_FACTORS = [
    "dsa_dir", "prev_pivot_code", "dsa_pivot_pos_01",
    "ret_to_last_low_pct", "bars_since_last_high",
    "price_vs_dsa_vwap_pct", "current_stage_amp_pct", "prev_stage_amp_pct",
    "current_stage_ret_pct", "bbmacd", "bbmacd_minus_avg",
    "bbmacd_bandwidth_zscore", "bbmacd_slope_3", "bbmacd_sign",
    "trend_align_momo",
]

VOLUME_FACTORS = [
    "vol_zscore_5", "vol_zscore_10", "vol_zscore_20",
    "vol_ratio_10", "vol_stage_cv",
]

CORE_COORD_FACTORS = [
    "price_vol_coord", "coord_stage_current",
]

ALL_COORD_FACTORS = [
    "price_vol_coord", "momo_vol_coord", "low_pos_break_coord",
    "coord_consistency", "coord_stage_current", "coord_stage_prev",
    "coord_stage_ratio",
]

EXTRA_FEATURES = ["weekly_return_score", "weekly_risk_score", "day_offset"]

DERIVED = {
    "pivot_pos_x_trend": lambda df: df["dsa_pivot_pos_01"] * df["dsa_dir"],
    "amp_x_pullback": lambda df: df["current_stage_amp_pct"] * df.get("current_pullback_from_stage_extreme_pct", 0),
    "vol_x_stage_amp": lambda df: df["vol_zscore_20"] * df["current_stage_amp_pct"],
}

FACTOR_SETS = {
    "A_base": BASE_FACTORS + EXTRA_FEATURES,
    "B_base+volume": BASE_FACTORS + VOLUME_FACTORS + EXTRA_FEATURES,
    "C_base+volume+core_coord": BASE_FACTORS + VOLUME_FACTORS + CORE_COORD_FACTORS + EXTRA_FEATURES,
    "D_full": BASE_FACTORS + VOLUME_FACTORS + ALL_COORD_FACTORS + EXTRA_FEATURES,
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


@dataclass
class Fold:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray


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
        folds.append(Fold(name=f"fold_{i+1}", train_idx=train_idx, test_idx=test_idx))
    return folds


def train_and_evaluate(df, feature_cols, target_col, folds):
    all_ics = []
    all_maes = []
    for fold in folds:
        train_data = lgb.Dataset(
            df.loc[fold.train_idx, feature_cols],
            df.loc[fold.train_idx, target_col],
            free_raw_data=False,
        )
        test_data = lgb.Dataset(
            df.loc[fold.test_idx, feature_cols],
            df.loc[fold.test_idx, target_col],
            reference=train_data,
            free_raw_data=False,
        )
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(REG_PARAMS, train_data, num_boost_round=1000, valid_sets=[test_data], callbacks=callbacks)

        y_test = df.loc[fold.test_idx, target_col]
        y_pred = model.predict(df.loc[fold.test_idx, feature_cols], num_iteration=model.best_iteration)
        valid_mask = y_test.notna()
        y_test_v = y_test[valid_mask].values
        y_pred_v = y_pred[valid_mask]

        if len(y_test_v) > 10:
            ic = stats.spearmanr(y_test_v, y_pred_v)[0]
            all_ics.append(ic)
        mae = mean_absolute_error(y_test_v, y_pred_v)
        all_maes.append(mae)

    ic_mean = np.mean(all_ics) if all_ics else np.nan
    ic_std = np.std(all_ics) if all_ics else np.nan
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    mae_mean = np.mean(all_maes) if all_maes else np.nan
    return {"ic_mean": ic_mean, "ic_std": ic_std, "icir": icir, "mae": mae_mean}


def compute_quintile_spread(df, feature_cols, target_col, folds, last_model=None):
    if last_model is None:
        return np.nan
    last_fold = folds[-1]
    y_pred = last_model.predict(df.loc[last_fold.test_idx, feature_cols], num_iteration=last_model.best_iteration)
    test_df = df.loc[last_fold.test_idx].copy()
    test_df["pred"] = y_pred
    valid = test_df[[target_col, "pred", "selection_date"]].dropna()
    if len(valid) < 100:
        return np.nan

    def qcut_daily(g):
        if len(g) < 5:
            return g
        g = g.copy()
        g["quintile"] = pd.qcut(g["pred"], 5, labels=False, duplicates="drop") + 1
        return g

    valid = valid.groupby("selection_date", group_keys=False).apply(qcut_daily)
    if "quintile" not in valid.columns:
        return np.nan
    report = valid.groupby("quintile")[target_col].mean()
    return report.iloc[-1] - report.iloc[0] if len(report) >= 2 else np.nan


def main():
    parser = argparse.ArgumentParser(description="因子删减对比实验")
    parser.add_argument("--sample-limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("因子删减对比实验（ablation）")
    print("=" * 80)

    print("\n加载数据...")
    df = pd.read_parquet(os.path.join(OUTPUT_DIR, "daily_factor_table.parquet"))
    if args.sample_limit > 0:
        df = df.head(args.sample_limit)
    print(f"  记录数: {len(df)}")

    for name, func in DERIVED.items():
        try:
            df[name] = func(df)
        except Exception:
            df[name] = np.nan

    for set_name, factor_cols in FACTOR_SETS.items():
        factor_cols_with_derived = factor_cols.copy()
        for d in DERIVED:
            if d not in factor_cols_with_derived:
                if d == "vol_x_stage_amp" and "vol_zscore_20" in factor_cols:
                    factor_cols_with_derived.append(d)
                elif d in ["pivot_pos_x_trend", "amp_x_pullback"]:
                    factor_cols_with_derived.append(d)
        FACTOR_SETS[set_name] = factor_cols_with_derived

    df = df.sort_values("selection_date").reset_index(drop=True)
    folds = build_rolling_splits(df)
    print(f"  有效折数: {len(folds)}")

    results = []
    for set_name, factor_cols in FACTOR_SETS.items():
        available = [c for c in factor_cols if c in df.columns]
        print(f"\n{'=' * 60}")
        print(f"实验组: {set_name} (特征数: {len(available)})")
        print(f"{'=' * 60}")

        opp_result = train_and_evaluate(df, available, OPP_TARGET, folds)
        risk_result = train_and_evaluate(df, available, RISK_TARGET, folds)

        result = {
            "group": set_name,
            "n_features": len(available),
            "opp_ic": opp_result["ic_mean"],
            "opp_icir": opp_result["icir"],
            "opp_mae": opp_result["mae"],
            "risk_ic": risk_result["ic_mean"],
            "risk_icir": risk_result["icir"],
            "risk_mae": risk_result["mae"],
        }
        results.append(result)

        print(f"  机会模型: IC={opp_result['ic_mean']:.4f}, ICIR={opp_result['icir']:.2f}, MAE={opp_result['mae']:.4f}")
        print(f"  风险模型: IC={risk_result['ic_mean']:.4f}, ICIR={risk_result['icir']:.2f}, MAE={risk_result['mae']:.4f}")

    results_df = pd.DataFrame(results)

    print(f"\n{'=' * 80}")
    print("因子删减对比汇总")
    print(f"{'=' * 80}")
    print(f"{'组':<30} {'特征数':>4} {'机会IC':>8} {'机会ICIR':>8} {'风险IC':>8} {'风险ICIR':>8}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['group']:<30} {row['n_features']:>4} {row['opp_ic']:>8.4f} {row['opp_icir']:>8.2f} {row['risk_ic']:>8.4f} {row['risk_icir']:>8.2f}")

    if len(results_df) >= 2:
        base_opp_ic = results_df.iloc[0]["opp_ic"]
        base_risk_ic = results_df.iloc[0]["risk_ic"]
        print(f"\n增量分析（相对A基线）:")
        for _, row in results_df.iterrows():
            opp_delta = row["opp_ic"] - base_opp_ic
            risk_delta = row["risk_ic"] - base_risk_ic
            print(f"  {row['group']:<30} 机会IC增量={opp_delta:+.4f}, 风险IC增量={risk_delta:+.4f}")

    results_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"), index=False)
    print(f"\n已保存: {os.path.join(OUTPUT_DIR, 'ablation_results.csv')}")


if __name__ == "__main__":
    main()
