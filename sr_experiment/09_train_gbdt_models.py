# -*- coding: utf-8 -*-
"""
Purpose: 训练 LightGBM 模型（3个实验，时间切分，val 早停）
Inputs:  results/gbdt/datasets/{experiment_name}.parquet
Outputs: results/gbdt/models/{experiment_name}.txt + feature_importance.csv + test_predictions.parquet
How to Run:
    python sr_experiment/09_train_gbdt_models.py
    python sr_experiment/09_train_gbdt_models.py --experiment A_support_reclaim_quality
Examples:
    python sr_experiment/09_train_gbdt_models.py
    python sr_experiment/09_train_gbdt_models.py --experiment B_cluster_low_volume_refine
Side Effects: 写模型文件和预测结果到 sr_experiment/results/gbdt/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sr_experiment.gbdt_config import (
    DATASETS_DIR,
    EARLY_STOPPING_ROUNDS,
    EVAL_DIR,
    EXPERIMENT_SPECS,
    LGB_PARAMS,
    MODELS_DIR,
)
from sr_experiment.gbdt_feature_columns import ALL_FEATURE_COLS, FACTOR_CATEGORIES


def train_experiment(exp_name: str, spec: dict):
    dataset_path = Path(DATASETS_DIR) / f"{exp_name}.parquet"
    if not dataset_path.exists():
        print(f"数据集不存在: {dataset_path}")
        return

    df = pd.read_parquet(dataset_path)
    print(f"\n{'='*60}")
    print(f"实验 {exp_name}: {spec['description']}")
    print(f"总样本: {len(df)}")

    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    print(f"可用特征: {len(feature_cols)}/{len(ALL_FEATURE_COLS)}")

    label_col = spec["label_name"]
    if label_col not in df.columns:
        print(f"标签列不存在: {label_col}")
        return

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, label_col]
    X_val = df.loc[val_mask, feature_cols]
    y_val = df.loc[val_mask, label_col]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, label_col]

    print(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"train正样本率: {y_train.mean():.4f}, val正样本率: {y_val.mean():.4f}")

    if len(X_train) < 200 or y_train.sum() < 50:
        print("训练样本不足，跳过")
        return

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {**LGB_PARAMS, "objective": spec["objective"], "metric": "auc"}

    model = lgb.train(
        params,
        train_data,
        num_boost_round=LGB_PARAMS["n_estimators"],
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    model_path = Path(MODELS_DIR) / f"{exp_name}.txt"
    model.save_model(str(model_path))
    print(f"模型已保存: {model_path}")

    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "experiment": exp_name,
        "feature": feature_cols,
        "gain": importance,
    }).sort_values("gain", ascending=False)

    category_map = {}
    for cat, cols in FACTOR_CATEGORIES.items():
        for col in cols:
            category_map[col] = cat
    imp_df["category"] = imp_df["feature"].map(category_map).fillna("other")
    imp_df["rank"] = range(1, len(imp_df) + 1)
    imp_df["gain_pct"] = imp_df["gain"] / imp_df["gain"].sum() * 100

    Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)
    imp_path = Path(EVAL_DIR) / f"feature_importance_{exp_name}.csv"
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
    print(f"\nTop 10 特征重要性:")
    print(imp_df.head(10)[["rank", "feature", "category", "gain_pct"]].to_string(index=False))

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    for name, y_true, y_pred in [("train", y_train, pred_train), ("val", y_val, pred_val), ("test", y_test, pred_test)]:
        if len(y_true) > 0 and y_true.nunique() > 1:
            auc = roc_auc_score(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
            print(f"{name}: AUC={auc:.4f}, AP={ap:.4f}")

    test_idx = df.loc[test_mask].index
    pred_df = df.loc[test_idx, ["ts_code", "bar_time", label_col] + feature_cols].copy()
    pred_df["pred_score"] = pred_test
    for col in ["fwd_ret_20", "fwd_max_ret_20", "fwd_mdd_20", "fwd_reward_risk_20"]:
        if col in df.columns:
            pred_df[col] = df.loc[test_idx, col].values

    pred_path = Path(EVAL_DIR) / f"test_predictions_{exp_name}.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"预测结果已保存: {pred_path}")


def main():
    parser = argparse.ArgumentParser(description="训练 GBDT 模型")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=list(EXPERIMENT_SPECS.keys()))
    args = parser.parse_args()

    specs = EXPERIMENT_SPECS
    if args.experiment:
        specs = {args.experiment: EXPERIMENT_SPECS[args.experiment]}

    for exp_name, spec in specs.items():
        train_experiment(exp_name, spec)


if __name__ == "__main__":
    main()
