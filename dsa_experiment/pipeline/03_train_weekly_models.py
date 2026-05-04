#!/usr/bin/env python3
"""
训练收益排序模型 + 风险veto模型，输出预测分

Purpose: 基于候选池底表训练 LightGBM 模型，输出收益分和风险分
Inputs: candidate_table.parquet
Outputs: return_model/, risk_model/, candidate_with_scores.parquet
How to Run:
    python dsa_experiment/pipeline/03_train_weekly_models.py
    python dsa_experiment/pipeline/03_train_weekly_models.py --sample-limit 5000
Side Effects: 只读操作，输出文件到 dsa_experiment/output/

管线位置: Step 3/7 — 周线模型训练（收益排序+风险veto）
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
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

FEATURE_COLS = [
    "dsa_dir", "prev_pivot_code", "last_confirmed_high", "last_confirmed_low",
    "dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct", "current_stage_bars", "prev_stage_bars",
    "bars_since_last_high", "bars_since_last_low", "prev_stage_amp_pct",
    "current_stage_ret_pct", "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct", "bbmacd", "bbmacd_minus_avg",
    "bbmacd_state", "bbmacd_band_pos_01", "bbmacd_bandwidth_zscore",
    "bbmacd_cross_upper", "bbmacd_cross_lower", "trend_align_momo",
    "vol_zscore_5", "vol_zscore_10", "vol_zscore_20", "vol_ratio_10",
    "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio",
    "price_vol_coord", "momo_vol_coord", "low_pos_break_coord", "coord_consistency",
    "coord_stage_current", "coord_stage_prev", "coord_stage_ratio",
]

DERIVED_FEATURES = {
    "high_low_range": lambda df: df["last_confirmed_high"] - df["last_confirmed_low"],
    "high_low_range_pct": lambda df: (df["last_confirmed_high"] - df["last_confirmed_low"]) / df["last_confirmed_low"].replace(0, np.nan),
    "pivot_pos_x_trend": lambda df: df["dsa_pivot_pos_01"] * df["trend_align_momo"],
    "stage_bars_ratio": lambda df: df["current_stage_bars"] / df["prev_stage_bars"].replace(0, np.nan),
    "amp_x_pullback": lambda df: df["current_stage_amp_pct"] * df["current_pullback_from_stage_extreme_pct"],
    "bbmacd_band_width": lambda df: df["bbmacd_band_pos_01"] * df["bbmacd_bandwidth_zscore"],
}

RETURN_TARGET = "ret_5_open_to_open"
RISK_TARGET = "stop_hit_5"
EMBARGO_DAYS = 25

REG_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "seed": 42,
    "num_leaves": 16,
    "max_depth": 5,
    "min_data_in_leaf": 50,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

CLS_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "seed": 42,
    "num_leaves": 16,
    "max_depth": 5,
    "min_data_in_leaf": 50,
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
    train_end: str
    test_end: str


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, func in DERIVED_FEATURES.items():
        df[name] = func(df)
    return df


def get_all_feature_cols(df: pd.DataFrame) -> list:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    for name in DERIVED_FEATURES:
        if name in df.columns:
            cols.append(name)
    return cols


def build_rolling_splits(df: pd.DataFrame, n_folds: int = 3) -> list:
    dates = sorted(df["selection_date"].unique())
    n = len(dates)
    fold_size = n // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_end_idx = fold_size * (i + 1)
        test_end_idx = fold_size * (i + 2) if i < n_folds - 1 else n
        train_end_date = dates[train_end_idx - 1]
        test_start_date = dates[train_end_idx]
        test_end_date = dates[test_end_idx - 1]

        embargo_cutoff = pd.Timestamp(test_start_date) - pd.Timedelta(days=EMBARGO_DAYS)
        train_mask = pd.to_datetime(df["selection_date"]) <= embargo_cutoff
        test_mask = (pd.to_datetime(df["selection_date"]) >= pd.Timestamp(test_start_date)) & (pd.to_datetime(df["selection_date"]) <= pd.Timestamp(test_end_date))

        train_idx = df.index[train_mask].values
        test_idx = df.index[test_mask].values
        if len(train_idx) < 100 or len(test_idx) < 50:
            continue
        folds.append(Fold(
            name=f"fold_{i + 1}",
            train_idx=train_idx,
            test_idx=test_idx,
            train_end=str(train_end_date)[:10],
            test_end=str(test_end_date)[:10],
        ))
    return folds


def train_lightgbm(df, feature_cols, target_col, fold, params, task="regression"):
    train_data = lgb.Dataset(
        df.loc[fold.train_idx, feature_cols],
        df.loc[fold.train_idx, target_col],
        categorical_feature=["bbmacd_state", "trend_align_momo"],
        free_raw_data=False,
    )
    test_data = lgb.Dataset(
        df.loc[fold.test_idx, feature_cols],
        df.loc[fold.test_idx, target_col],
        reference=train_data,
        categorical_feature=["bbmacd_state", "trend_align_momo"],
        free_raw_data=False,
    )
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data], callbacks=callbacks)

    y_test = df.loc[fold.test_idx, target_col]
    y_pred = model.predict(df.loc[fold.test_idx, feature_cols], num_iteration=model.best_iteration)
    valid_mask = y_test.notna()
    y_test_v = y_test[valid_mask].values
    y_pred_v = y_pred[valid_mask]

    metrics = {"fold": fold.name, "train_end": fold.train_end, "test_end": fold.test_end,
               "n_train": len(fold.train_idx), "n_test": len(fold.test_idx),
               "best_iteration": model.best_iteration}

    if task == "regression":
        metrics["test_mae"] = mean_absolute_error(y_test_v, y_pred_v)
        ic_spearman = stats.spearmanr(y_test_v, y_pred_v)[0] if len(y_test_v) > 10 else 0
        metrics["test_ic_spearman"] = ic_spearman
    else:
        if len(np.unique(y_test_v)) > 1:
            metrics["test_auc"] = roc_auc_score(y_test_v, y_pred_v)
            metrics["test_ap"] = average_precision_score(y_test_v, y_pred_v)
        else:
            metrics["test_auc"] = np.nan
            metrics["test_ap"] = np.nan

    return model, metrics


def compute_daily_ic(df, feature_cols, target_col, fold, model) -> pd.DataFrame:
    test_df = df.loc[fold.test_idx].copy()
    test_df["pred"] = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)
    valid = test_df[[target_col, "pred", "selection_date"]].dropna()
    if valid.empty:
        return pd.DataFrame()
    rows = []
    for dt, grp in valid.groupby("selection_date"):
        if len(grp) < 10:
            continue
        ic = stats.spearmanr(grp[target_col], grp["pred"])[0]
        rows.append({"date": dt, "ic": ic, "n": len(grp)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="训练收益+风险模型")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("训练收益排序模型 + 风险veto模型")
    print("=" * 80)

    print("\n[1/5] 加载候选池底表...")
    input_path = os.path.join(args.output_dir, "candidate_table.parquet")
    df = pd.read_parquet(input_path)
    if args.sample_limit > 0:
        df = df.head(args.sample_limit)
    print(f"  记录数: {len(df)}")

    print("\n[2/5] 特征工程...")
    df = build_features(df)
    feature_cols = get_all_feature_cols(df)
    print(f"  特征数: {len(feature_cols)}")

    tradeable = df[df["can_buy_next_open"] == True].copy()
    tradeable = tradeable.sort_values("selection_date").reset_index(drop=True)
    print(f"  可交易记录: {len(tradeable)}")

    print("\n[3/5] 时间序列切分...")
    folds = build_rolling_splits(tradeable, args.n_folds)
    print(f"  有效折数: {len(folds)}")
    for f in folds:
        print(f"    {f.name}: train~{f.train_end}, test~{f.test_end}, n_train={len(f.train_idx)}, n_test={len(f.test_idx)}")

    all_return_ic = []
    all_risk_ic = []

    print("\n[4/5] 训练收益排序模型...")
    ret_metrics = []
    ret_models = []
    for fold in folds:
        valid_ret = tradeable.loc[fold.train_idx, RETURN_TARGET].notna().sum()
        if valid_ret < 50:
            print(f"  {fold.name}: 训练集有效标签不足({valid_ret})，跳过")
            continue
        model, m = train_lightgbm(tradeable, feature_cols, RETURN_TARGET, fold, REG_PARAMS, "regression")
        ret_metrics.append(m)
        ret_models.append(model)
        print(f"  {fold.name}: IC={m['test_ic_spearman']:.4f}, MAE={m['test_mae']:.4f}")

        daily_ic = compute_daily_ic(tradeable, feature_cols, RETURN_TARGET, fold, model)
        if not daily_ic.empty:
            daily_ic["fold"] = fold.name
            all_return_ic.append(daily_ic)

    if ret_metrics:
        ic_mean = np.mean([m["test_ic_spearman"] for m in ret_metrics])
        ic_std = np.std([m["test_ic_spearman"] for m in ret_metrics])
        print(f"\n  跨折汇总: IC={ic_mean:.4f} ± {ic_std:.4f}, ICIR={ic_mean/ic_std:.2f}" if ic_std > 0 else f"  IC={ic_mean:.4f}")

    print("\n[5/5] 训练风险veto模型...")
    risk_metrics = []
    risk_models = []
    for fold in folds:
        valid_risk = tradeable.loc[fold.train_idx, RISK_TARGET].notna().sum()
        pos_rate = tradeable.loc[fold.train_idx, RISK_TARGET].mean()
        if valid_risk < 50 or pos_rate < 0.01 or pos_rate > 0.99:
            print(f"  {fold.name}: 训练集有效标签不足或正例比例异常({pos_rate:.2%})，跳过")
            continue
        model, m = train_lightgbm(tradeable, feature_cols, RISK_TARGET, fold, CLS_PARAMS, "classification")
        risk_metrics.append(m)
        risk_models.append(model)
        print(f"  {fold.name}: AUC={m.get('test_auc', 0):.4f}, AP={m.get('test_ap', 0):.4f}")

    if risk_metrics:
        auc_mean = np.mean([m["test_auc"] for m in risk_metrics])
        print(f"\n  跨折汇总: AUC={auc_mean:.4f}")

    print("\n保存模型和结果...")
    ret_dir = os.path.join(args.output_dir, "return_model")
    risk_dir = os.path.join(args.output_dir, "risk_model")
    os.makedirs(ret_dir, exist_ok=True)
    os.makedirs(risk_dir, exist_ok=True)

    if ret_metrics:
        pd.DataFrame(ret_metrics).to_csv(os.path.join(ret_dir, "fold_metrics.csv"), index=False)
        if ret_models:
            ret_models[-1].save_model(os.path.join(ret_dir, "model.txt"))
            gain = ret_models[-1].feature_importance("gain")
            fi = pd.DataFrame({"feature": feature_cols, "importance_gain": gain / gain.sum() if gain.sum() > 0 else gain})
            fi = fi.sort_values("importance_gain", ascending=False).reset_index(drop=True)
            fi.to_csv(os.path.join(ret_dir, "feature_importance.csv"), index=False)
            print(f"  收益模型特征重要性 Top5:")
            for _, row in fi.head(5).iterrows():
                print(f"    {row['feature']:<35} gain={row['importance_gain']:.4f}")

    if all_return_ic:
        pd.concat(all_return_ic).to_csv(os.path.join(ret_dir, "daily_ic.csv"), index=False)

    if risk_metrics:
        pd.DataFrame(risk_metrics).to_csv(os.path.join(risk_dir, "fold_metrics.csv"), index=False)
        if risk_models:
            risk_models[-1].save_model(os.path.join(risk_dir, "model.txt"))
            gain = risk_models[-1].feature_importance("gain")
            fi = pd.DataFrame({"feature": feature_cols, "importance_gain": gain / gain.sum() if gain.sum() > 0 else gain})
            fi = fi.sort_values("importance_gain", ascending=False).reset_index(drop=True)
            fi.to_csv(os.path.join(risk_dir, "feature_importance.csv"), index=False)
            print(f"  风险模型特征重要性 Top5:")
            for _, row in fi.head(5).iterrows():
                print(f"    {row['feature']:<35} gain={row['importance_gain']:.4f}")

    print("\n生成预测分...")
    if ret_models and risk_models and folds:
        last_fold = folds[-1]
        tradeable["return_score"] = np.nan
        tradeable["risk_score"] = np.nan
        for i, fold in enumerate(folds):
            if i < len(ret_models):
                test_mask = tradeable.index.isin(fold.test_idx)
                tradeable.loc[test_mask, "return_score"] = ret_models[i].predict(
                    tradeable.loc[test_mask, feature_cols], num_iteration=ret_models[i].best_iteration
                )
            if i < len(risk_models):
                test_mask = tradeable.index.isin(fold.test_idx)
                tradeable.loc[test_mask, "risk_score"] = risk_models[i].predict(
                    tradeable.loc[test_mask, feature_cols], num_iteration=risk_models[i].best_iteration
                )

        scored = tradeable[tradeable["return_score"].notna()].copy()
        output_path = os.path.join(args.output_dir, "candidate_with_scores.parquet")
        scored.to_parquet(output_path, index=False)
        print(f"  保存: {output_path}")
        print(f"  有预测分的记录: {len(scored)}")
    else:
        print("  模型不足，跳过预测分生成")

    print("\n" + "=" * 80)
    print("模型训练完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
