#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四模型预测 Stacking 元模型训练

Purpose:
    将 4 个子模型 (sell_reg/sell_cls/buy_reg/buy_cls) 的预测值作为特征，
    训练元模型，对比是否优于单模型 sell_reg 打分。

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet
    - stop_experiment/output/models_control/feature_importance.csv

Outputs:
    - results/meta_models/ (元模型文件)
    - results/metrics/train_metrics.csv (训练指标)
    - results/meta_test_predictions.parquet (test 集预测)

How to Run:
    python -m stop_experiment.experiments.stacking_experiment.01_train_meta_model
    python -m stop_experiment.experiments.stacking_experiment.01_train_meta_model --sample-limit 5000

Side Effects:
    - 只读 candidate_with_scores.parquet 和 feature_importance.csv
    - 输出仅写入 results/ 子目录，不影响生产
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score

from stop_experiment.pipeline.stop_config import (
    EMBARGO_DAYS, OBS_TRAIN_END, OBS_VAL_END,
    OUTPUT_DIR, MODELS_DIR,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
META_MODELS_DIR = os.path.join(RESULTS_DIR, "meta_models")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

PRED_COLS = ["pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls"]

TOP10_FEATURES = [
    "price_vs_dsa_vwap_pct",
    "ret_to_last_low_pct",
    "current_stage_amp_pct",
    "ret_to_last_high_pct",
    "current_pullback_from_stage_extreme_pct",
    "current_stage_ret_pct",
    "dsa_pivot_pos_01",
    "beta",
    "range_position",
    "intraday_range",
]

META_LGB_PARAMS = {
    "num_leaves": 8,
    "max_depth": 3,
    "min_data_in_leaf": 100,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_bin": 31,
    "seed": 42,
    "verbosity": -1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
}

META_SPECS = {
    "stack_lgb_sell_reg": {
        "target": "mfe_20",
        "objective": "regression",
        "metric": "mae",
        "feature_groups": ["pred_only"],
    },
    "stack_lgb_sell_cls": {
        "target": "sell_signal",
        "objective": "binary",
        "metric": "auc",
        "feature_groups": ["pred_only"],
    },
    "stack_lgb_composite": {
        "target": "mfe_20",
        "objective": "regression",
        "metric": "mae",
        "feature_groups": ["pred_only", "top10_raw"],
    },
}


def build_split(df: pd.DataFrame) -> dict:
    obs_train_end_ts = pd.Timestamp(OBS_TRAIN_END)
    obs_val_end_ts = pd.Timestamp(OBS_VAL_END)
    embargo_td = pd.Timedelta(days=EMBARGO_DAYS)
    train_cutoff = obs_train_end_ts - embargo_td

    train_mask = df["obs_date"] <= train_cutoff
    val_mask = (df["obs_date"] > obs_train_end_ts) & (df["obs_date"] <= obs_val_end_ts)
    test_mask = df["obs_date"] > obs_val_end_ts

    train_idx = df.index[train_mask].values
    val_idx = df.index[val_mask].values
    test_idx = df.index[test_mask].values

    print(f"  train: obs_date <= {train_cutoff.date()}, n={len(train_idx)}")
    print(f"  val:   {obs_train_end_ts.date()} < obs_date <= {obs_val_end_ts.date()}, n={len(val_idx)}")
    print(f"  test:  obs_date > {obs_val_end_ts.date()}, n={len(test_idx)}")

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def get_feature_cols(spec: dict, df: pd.DataFrame) -> list:
    cols = []
    for group in spec["feature_groups"]:
        if group == "pred_only":
            cols.extend([c for c in PRED_COLS if c in df.columns])
        elif group == "top10_raw":
            cols.extend([c for c in TOP10_FEATURES if c in df.columns])
    return cols


def train_meta_model(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    params: dict,
    task: str,
    model_name: str,
) -> tuple:
    train_data = lgb.Dataset(
        df.loc[train_idx, feature_cols],
        df.loc[train_idx, target_col],
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        df.loc[val_idx, feature_cols],
        df.loc[val_idx, target_col],
        reference=train_data,
        free_raw_data=False,
    )

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    y_val = df.loc[val_idx, target_col]
    y_val_pred = model.predict(df.loc[val_idx, feature_cols], num_iteration=model.best_iteration)
    valid_mask = y_val.notna()
    y_val_v = y_val[valid_mask].values
    y_val_pred_v = y_val_pred[valid_mask]

    val_metrics = {
        "model_name": model_name,
        "split": "val",
        "best_iteration": model.best_iteration,
        "n_features": len(feature_cols),
    }

    if task == "regression":
        val_metrics["eval_mae"] = mean_absolute_error(y_val_v, y_val_pred_v) if len(y_val_v) > 0 else np.nan
        val_metrics["eval_ic_spearman"] = stats.spearmanr(y_val_v, y_val_pred_v)[0] if len(y_val_v) > 10 else np.nan
    else:
        if len(np.unique(y_val_v)) > 1 and len(y_val_v) > 0:
            val_metrics["eval_auc"] = roc_auc_score(y_val_v, y_val_pred_v)
            val_metrics["eval_ap"] = average_precision_score(y_val_v, y_val_pred_v)
        else:
            val_metrics["eval_auc"] = np.nan
            val_metrics["eval_ap"] = np.nan

    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "gain": importance,
        "model_name": model_name,
    })

    return model, val_metrics, imp_df


def evaluate_on_test(
    model,
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_idx: np.ndarray,
    task: str,
    model_name: str,
) -> dict:
    y_test = df.loc[test_idx, target_col]
    y_pred = model.predict(df.loc[test_idx, feature_cols], num_iteration=model.best_iteration)
    valid_mask = y_test.notna()
    y_test_v = y_test[valid_mask].values
    y_pred_v = y_pred[valid_mask]

    test_metrics = {
        "model_name": model_name,
        "split": "test",
        "n_test": len(test_idx),
        "n_valid": len(y_test_v),
    }

    if task == "regression":
        test_metrics["test_mae"] = mean_absolute_error(y_test_v, y_pred_v) if len(y_test_v) > 0 else np.nan
        test_metrics["test_ic_spearman"] = stats.spearmanr(y_test_v, y_pred_v)[0] if len(y_test_v) > 10 else np.nan
    else:
        if len(np.unique(y_test_v)) > 1 and len(y_test_v) > 0:
            test_metrics["test_auc"] = roc_auc_score(y_test_v, y_pred_v)
            test_metrics["test_ap"] = average_precision_score(y_test_v, y_pred_v)
        else:
            test_metrics["test_auc"] = np.nan
            test_metrics["test_ap"] = np.nan

    return test_metrics


def main(args):
    print("=" * 60)
    print("Stacking 元模型训练实验")
    print("=" * 60)

    os.makedirs(META_MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在，请先运行 02_train_gbdt_models.py")

    print("\n[1/4] 加载数据...")
    df = pd.read_parquet(scores_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    if args.sample_limit > 0:
        df = df.head(args.sample_limit)
        print(f"  sample_limit: {args.sample_limit}")
    df = df.dropna(subset=["mfe_20", "mae_20"])
    df = df.sort_values("obs_date").reset_index(drop=True)
    print(f"  总行数: {len(df)}")

    for col in PRED_COLS:
        if col not in df.columns:
            raise ValueError(f"缺少预测列: {col}")
    missing_pred = df[PRED_COLS].isna().any(axis=1).sum()
    if missing_pred > 0:
        print(f"  预测列有 {missing_pred} 行缺失，将剔除")
        df = df.dropna(subset=PRED_COLS).reset_index(drop=True)
        print(f"  剔除后: {len(df)} 行")

    if "sell_signal" not in df.columns:
        from stop_experiment.pipeline.stop_config import SELL_CLS_THRESHOLD
        df["sell_signal"] = (df["mfe_20"] > SELL_CLS_THRESHOLD).astype(int)
        print(f"  sell_signal 已构造 (mfe_20 > {SELL_CLS_THRESHOLD})")

    print("\n[2/4] 数据分割...")
    split = build_split(df)

    print("\n[3/4] 训练元模型...")
    all_metrics = []
    all_importance = []
    test_preds = df.loc[split["test"]].copy()

    baseline_ic = np.nan
    baseline_mae = np.nan
    if len(split["test"]) > 10:
        valid_mask = df.loc[split["test"], "mfe_20"].notna() & df.loc[split["test"], "pred_sell_reg"].notna()
        if valid_mask.sum() > 10:
            baseline_ic = stats.spearmanr(
                df.loc[split["test"], "mfe_20"][valid_mask].values,
                df.loc[split["test"], "pred_sell_reg"][valid_mask].values,
            )[0]
            baseline_mae = mean_absolute_error(
                df.loc[split["test"], "mfe_20"][valid_mask].values,
                df.loc[split["test"], "pred_sell_reg"][valid_mask].values,
            )
    all_metrics.append({
        "model_name": "baseline_sell_reg",
        "split": "test",
        "n_test": len(split["test"]),
        "test_ic_spearman": baseline_ic,
        "test_mae": baseline_mae,
    })
    print(f"  baseline (sell_reg): IC={baseline_ic:.4f}, MAE={baseline_mae:.4f}")

    for model_name, spec in META_SPECS.items():
        target = spec["target"]
        objective = spec["objective"]
        metric_name = spec["metric"]
        task = "classification" if objective == "binary" else "regression"

        feature_cols = get_feature_cols(spec, df)
        print(f"\n  --- {model_name} ---")
        print(f"    target={target}, task={task}, features={feature_cols}")

        params = META_LGB_PARAMS.copy()
        params["objective"] = objective
        params["metric"] = metric_name

        if task == "classification":
            pos_rate = df.loc[split["train"], target].mean()
            if pos_rate > 0:
                params["scale_pos_weight"] = (1 - pos_rate) / pos_rate
            print(f"    正类比例(train): {pos_rate:.3f}")

        model, val_metrics, imp_df = train_meta_model(
            df, feature_cols, target,
            split["train"], split["val"],
            params, task, model_name,
        )
        all_metrics.append(val_metrics)
        all_importance.append(imp_df)

        if task == "regression":
            print(f"    val: MAE={val_metrics.get('eval_mae', np.nan):.4f}, "
                  f"IC={val_metrics.get('eval_ic_spearman', np.nan):.4f}, "
                  f"best_iter={val_metrics['best_iteration']}")
        else:
            print(f"    val: AUC={val_metrics.get('eval_auc', np.nan):.4f}, "
                  f"AP={val_metrics.get('eval_ap', np.nan):.4f}, "
                  f"best_iter={val_metrics['best_iteration']}")

        test_metrics = evaluate_on_test(
            model, df, feature_cols, target,
            split["test"], task, model_name,
        )
        all_metrics.append(test_metrics)

        test_preds[f"meta_{model_name}"] = model.predict(
            df.loc[split["test"], feature_cols], num_iteration=model.best_iteration
        )

        if task == "regression":
            print(f"    test: MAE={test_metrics.get('test_mae', np.nan):.4f}, "
                  f"IC={test_metrics.get('test_ic_spearman', np.nan):.4f}")
        else:
            print(f"    test: AUC={test_metrics.get('test_auc', np.nan):.4f}, "
                  f"AP={test_metrics.get('test_ap', np.nan):.4f}")

        model_path = os.path.join(META_MODELS_DIR, f"{model_name}.txt")
        model.save_model(model_path)
        print(f"    模型保存: {model_path}")

    print("\n[4/4] 保存结果...")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(METRICS_DIR, "train_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  指标: {metrics_path}")

    if all_importance:
        imp_df = pd.concat(all_importance, ignore_index=True)
        imp_path = os.path.join(METRICS_DIR, "meta_feature_importance.csv")
        imp_df.to_csv(imp_path, index=False)
        print(f"  重要性: {imp_path}")

    test_preds_path = os.path.join(RESULTS_DIR, "meta_test_predictions.parquet")
    test_preds.to_parquet(test_preds_path, index=False)
    print(f"  test预测: {test_preds_path}")

    print(f"\n{'='*60}")
    print("指标汇总")
    print(f"{'='*60}")
    for row in all_metrics:
        name = row["model_name"]
        split_name = row["split"]
        if split_name == "test":
            if "test_ic_spearman" in row:
                print(f"  {name}: test_IC={row['test_ic_spearman']:.4f}, test_MAE={row.get('test_mae', np.nan):.4f}")
            elif "test_auc" in row:
                print(f"  {name}: test_AUC={row['test_auc']:.4f}, test_AP={row.get('test_ap', np.nan):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacking 元模型训练")
    parser.add_argument("--sample-limit", type=int, default=0, help="限制样本数（调试用）")
    args = parser.parse_args()
    main(args)
