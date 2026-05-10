#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 Stop-Loss Clustering GBDT 模型（4个变体，v2: train/val/test分割）

Purpose:
    读取离线 dataset.parquet，训练4个 LightGBM 模型：
    sell_reg (mfe_20回归) + sell_cls (mfe_20>7%分类) + buy_reg (mae_20回归) + buy_cls (mae_20<-5%分类)。

    v2改动:
    - 数据分割从3-fold expanding window → train/val/test
    - val用于早停，test用于最终评估
    - 最终模型用train+val重训
    - 剔除8个无效特征，新增bbmacd_slope_3_pct和has_buy_cluster

Pipeline Position:
    训练流水线第二步（离线，一次性）。
    上游: 01_build_dataset.py
    下游: 03_factor_importance.py, 04_signal_selector.py

Inputs:
    - stop_experiment/output/dataset.parquet

Outputs:
    - stop_experiment/output/models_control/ (4个模型txt + 4个final模型txt)
    - stop_experiment/output/fold_metrics.csv
    - stop_experiment/output/feature_importance.csv
    - stop_experiment/output/candidate_with_scores.parquet
    - stop_experiment/output/test_predictions.parquet

How to Run:
    python stop_experiment/pipeline/02_train_gbdt_models.py
    python stop_experiment/pipeline/02_train_gbdt_models.py --sample-limit 10000

Side Effects:
    - 只读 parquet，输出模型和指标文件
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import warnings
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

from stop_experiment.pipeline.stop_config import (
    EMBARGO_DAYS, LGB_PARAMS, MODEL_SPECS,
    OUTPUT_DIR, DATASET_PATH, MODELS_DIR, MODELS_TREATMENT_DIR,
    OBS_TRAIN_END, OBS_VAL_END,
)
from stop_experiment.pipeline.factor_columns import ALL_FEATURE_COLS

BATCHES_DIR = os.path.join(OUTPUT_DIR, "dataset_batches")


def load_dataset() -> pd.DataFrame:
    """加载数据集，自动兼容分批/单文件模式"""
    manifest_path = os.path.join(BATCHES_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  分批数据集: {manifest['total_batches']} 批, {manifest['total_rows']} 行")
        dfs = []
        for batch_file in tqdm(manifest["batch_files"], desc="加载分批数据集"):
            batch_path = os.path.join(BATCHES_DIR, batch_file)
            dfs.append(pd.read_parquet(batch_path))
        return pd.concat(dfs, ignore_index=True)
    elif os.path.exists(DATASET_PATH):
        print(f"  单文件数据集: {DATASET_PATH}")
        return pd.read_parquet(DATASET_PATH)
    else:
        raise FileNotFoundError(f"数据集不存在: {BATCHES_DIR} 或 {DATASET_PATH}")


from datetime import datetime


@dataclass
class DataSplit:
    """train/val/test 数据分割"""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_end: str
    val_end: str


def build_train_val_test_split(df: pd.DataFrame) -> DataSplit:
    """
    基于 obs_date 的 train/val/test 分割（修复前视偏差）。

    旧版按 selection_date 切分，但特征来自 obs_date，导致 train 中混入 obs_date 已进入 val/test 期的样本。
    新版按 obs_date 切分，确保训练时不使用未来信息。

    - train: obs_date <= OBS_TRAIN_END - EMBARGO_DAYS
    - val:   OBS_TRAIN_END < obs_date <= OBS_VAL_END
    - test:  obs_date > OBS_VAL_END
    - 中间加 EMBARGO_DAYS 隔离防止数据泄漏
    """
    df["obs_date"] = pd.to_datetime(df["obs_date"])
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

    print(f"  train: obs_date <= {train_cutoff.date()} (embargo), n={len(train_idx)}")
    print(f"  val:   {obs_train_end_ts.date()} < obs_date <= {obs_val_end_ts.date()}, n={len(val_idx)}")
    print(f"  test:  obs_date > {obs_val_end_ts.date()}, n={len(test_idx)}")

    if obs_train_end_ts in df["obs_date"].values:
        gap_count = ((df["obs_date"] > train_cutoff) & (df["obs_date"] <= obs_train_end_ts)).sum()
        print(f"  embargo_gap (rejected): {gap_count} 样本")

    if len(train_idx) < 100:
        raise ValueError(f"训练集过小: {len(train_idx)}")
    if len(val_idx) < 50:
        raise ValueError(f"验证集过小: {len(val_idx)}")
    if len(test_idx) < 50:
        raise ValueError(f"测试集过小: {len(test_idx)}")

    return DataSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_end=OBS_TRAIN_END,
        val_end=OBS_VAL_END,
    )


def train_single_model(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    params: dict,
    task: str = "regression",
    model_name_tag: str = "",
) -> tuple:
    """训练单个 LightGBM 模型（用val早停）"""
    # 分类特征
    cat_cols = [c for c in ["trend_align_momo", "dsa_dir",
                             "bbmacd_sign", "prev_pivot_code", "price_vol_coord",
                             "has_buy_cluster"]
                if c in feature_cols]

    train_data = lgb.Dataset(
        df.loc[train_idx, feature_cols],
        df.loc[train_idx, target_col],
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        df.loc[val_idx, feature_cols],
        df.loc[val_idx, target_col],
        reference=train_data,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params, train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # 在val上评估
    y_val = df.loc[val_idx, target_col]
    y_val_pred = model.predict(df.loc[val_idx, feature_cols], num_iteration=model.best_iteration)
    valid_mask = y_val.notna()
    y_val_v = y_val[valid_mask].values
    y_val_pred_v = y_val_pred[valid_mask]

    val_metrics = {
        "split": "val",
        "n_train": len(train_idx),
        "n_eval": len(val_idx),
        "best_iteration": model.best_iteration,
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

    # 因子重要性
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "gain": importance,
        "model_name": model_name_tag,
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
    """在测试集上评估模型"""
    y_test = df.loc[test_idx, target_col]
    y_pred = model.predict(df.loc[test_idx, feature_cols], num_iteration=model.best_iteration)
    valid_mask = y_test.notna()
    y_test_v = y_test[valid_mask].values
    y_pred_v = y_pred[valid_mask]

    test_metrics = {
        "model_name": model_name,
        "target": target_col,
        "task": task,
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


def retrain_final_model(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    train_val_idx: np.ndarray,
    params: dict,
    best_iteration: int,
    task: str = "regression",
    model_name_tag: str = "",
) -> lgb.Booster:
    """
    用 train+val 重训最终模型。
    使用之前从val早停得到的 best_iteration 作为固定迭代次数。
    """
    cat_cols = [c for c in ["trend_align_momo", "dsa_dir",
                             "bbmacd_sign", "prev_pivot_code", "price_vol_coord",
                             "has_buy_cluster"]
                if c in feature_cols]

    train_data = lgb.Dataset(
        df.loc[train_val_idx, feature_cols],
        df.loc[train_val_idx, target_col],
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    # 用固定迭代次数训练，不做早停
    final_params = params.copy()
    final_params["verbosity"] = -1
    model = lgb.train(
        final_params, train_data,
        num_boost_round=best_iteration,
    )

    return model


def main(args):
    print("=" * 60)
    print("Stop-Loss Clustering GBDT 模型训练 (v2: train/val/test)")
    print("=" * 60)

    # 确定模型输出目录和特征集
    if args.exclude_vsa:
        models_dir = MODELS_DIR                        # models_control（当前默认基线）
        feature_set_tag = "Control (56特征, 不含VSA)"
    else:
        models_dir = MODELS_TREATMENT_DIR              # models（VSA版，研究保留）
        feature_set_tag = "Treatment (68特征, 含VSA)"

    # 1. 加载数据
    print("\n[1/5] 加载数据集...")
    df = load_dataset()
    if args.sample_limit > 0:
        df = df.head(args.sample_limit)
        print(f"  sample_limit: {args.sample_limit}")
    print(f"  总行数: {len(df)}, 列数: {len(df.columns)}")

    # 过滤可交易 + 有效标签
    df = df[df["can_buy"] == 1].copy()
    df = df.dropna(subset=["mfe_20", "mae_20"])
    df = df.sort_values("obs_date").reset_index(drop=True)
    print(f"  可交易+有效标签: {len(df)} 行")

    # 特征列
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    missing_features = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    print(f"  特征数: {len(feature_cols)}/{len(ALL_FEATURE_COLS)}")
    if missing_features:
        print(f"  缺少特征: {missing_features}")

    # 排除/保留 VSA 因子
    if args.exclude_vsa:
        feature_cols = [c for c in feature_cols if not c.startswith("vsa_")]
        print(f"  排除 VSA 后特征数: {len(feature_cols)} ({feature_set_tag})")

    # 2. 构建 train/val/test 分割
    print("\n[2/5] 构建 train/val/test 分割...")
    split = build_train_val_test_split(df)

    # 3. 训练4个模型（用val早停）
    print("\n[3/5] 训练4个模型变体（val早停）...")
    os.makedirs(models_dir, exist_ok=True)

    all_metrics = []
    all_importance = []
    models = {}  # 保存early-stopping模型
    best_iterations = {}  # 保存最佳迭代次数

    for model_name, spec in MODEL_SPECS.items():
        target = spec["target"]
        objective = spec["objective"]
        metric_name = spec["metric"]
        task = "classification" if objective == "binary" else "regression"

        print(f"\n  --- {model_name} (target={target}, task={task}) ---")

        # 构建参数
        params = LGB_PARAMS.copy()
        params["objective"] = objective
        params["metric"] = metric_name

        # 分类任务检查正类比例
        if task == "classification":
            pos_rate = df.loc[split.train_idx, target].mean()
            if pos_rate > 0:
                params["scale_pos_weight"] = (1 - pos_rate) / pos_rate
            print(f"    正类比例(train): {pos_rate:.3f}, scale_pos_weight: {params.get('scale_pos_weight', 1.0):.2f}")

        model, val_metrics, imp_df = train_single_model(
            df, feature_cols, target,
            split.train_idx, split.val_idx,
            params, task,
            model_name_tag=model_name,
        )
        val_metrics["model_name"] = model_name
        val_metrics["target"] = target
        val_metrics["task"] = task
        all_metrics.append(val_metrics)
        all_importance.append(imp_df)
        models[model_name] = model
        best_iterations[model_name] = model.best_iteration

        # 打印val指标
        if task == "regression":
            print(f"    val: MAE={val_metrics.get('eval_mae', np.nan):.4f}, "
                  f"IC={val_metrics.get('eval_ic_spearman', np.nan):.4f}, "
                  f"best_iter={val_metrics['best_iteration']}")
        else:
            print(f"    val: AUC={val_metrics.get('eval_auc', np.nan):.4f}, "
                  f"AP={val_metrics.get('eval_ap', np.nan):.4f}, "
                  f"best_iter={val_metrics['best_iteration']}")

        # 保存early-stopping模型
        model_path = os.path.join(models_dir, f"{model_name}.txt")
        model.save_model(model_path)
        print(f"    模型保存: {model_path}")

    # 4. 在test集上评估
    print("\n[4/5] 测试集评估...")
    test_metrics_all = []
    test_preds = df.loc[split.test_idx].copy()

    for model_name, spec in MODEL_SPECS.items():
        target = spec["target"]
        task = "classification" if spec["objective"] == "binary" else "regression"
        model = models[model_name]

        test_metrics = evaluate_on_test(
            model, df, feature_cols, target,
            split.test_idx, task, model_name,
        )
        test_metrics_all.append(test_metrics)

        # test集预测
        test_preds[f"pred_{model_name}"] = model.predict(
            df.loc[split.test_idx, feature_cols], num_iteration=model.best_iteration
        )

        if task == "regression":
            print(f"  {model_name}: test_MAE={test_metrics.get('test_mae', np.nan):.4f}, "
                  f"test_IC={test_metrics.get('test_ic_spearman', np.nan):.4f}")
        else:
            print(f"  {model_name}: test_AUC={test_metrics.get('test_auc', np.nan):.4f}, "
                  f"test_AP={test_metrics.get('test_ap', np.nan):.4f}")

    all_metrics.extend(test_metrics_all)

    # 5. 用 train+val 重训最终模型
    print("\n[5/5] 重训最终模型（train+val）...")
    train_val_idx = np.concatenate([split.train_idx, split.val_idx])

    for model_name, spec in MODEL_SPECS.items():
        target = spec["target"]
        objective = spec["objective"]
        metric_name = spec["metric"]
        task = "classification" if objective == "binary" else "regression"

        print(f"\n  --- {model_name} final ---")

        params = LGB_PARAMS.copy()
        params["objective"] = objective
        params["metric"] = metric_name

        if task == "classification":
            pos_rate = df.loc[train_val_idx, target].mean()
            if pos_rate > 0:
                params["scale_pos_weight"] = (1 - pos_rate) / pos_rate

        final_model = retrain_final_model(
            df, feature_cols, target,
            train_val_idx, params,
            best_iterations[model_name],
            task, model_name_tag=model_name,
        )

        # 保存最终模型
        final_model_path = os.path.join(models_dir, f"{model_name}_final.txt")
        final_model.save_model(final_model_path)
        print(f"    最终模型保存: {final_model_path} (iter={best_iterations[model_name]})")

    # 保存指标
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(models_dir, "fold_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  指标: {metrics_path}")

    # 保存重要性
    imp_df = pd.concat(all_importance, ignore_index=True)
    imp_path = os.path.join(models_dir, "feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"  重要性: {imp_path}")

    # 保存test集预测
    test_preds_path = os.path.join(models_dir, "test_predictions.parquet")
    test_preds.to_parquet(test_preds_path, index=False)
    print(f"  test预测: {test_preds_path}")

    # 生成全量带评分数据集（用final模型）
    print("\n生成全量带评分数据集（final模型）...")
    for model_name, spec in MODEL_SPECS.items():
        final_model_path = os.path.join(models_dir, f"{model_name}_final.txt")
        final_model = lgb.Booster(model_file=final_model_path)
        df[f"pred_{model_name}"] = final_model.predict(df[feature_cols])

    scores_path = os.path.join(models_dir, "candidate_with_scores.parquet")
    df.to_parquet(scores_path, index=False)
    print(f"  保存: {scores_path}")

    # 汇总指标
    print(f"\n{'='*60}")
    print("模型指标汇总")
    print(f"{'='*60}")
    for model_name in MODEL_SPECS:
        # val指标
        val_row = metrics_df[(metrics_df["model_name"] == model_name) & (metrics_df["split"] == "val")]
        test_row = metrics_df[(metrics_df["model_name"] == model_name) & (metrics_df["split"] == "test")]
        task = val_row["task"].iloc[0] if len(val_row) > 0 else "unknown"

        if task == "regression":
            val_mae = val_row["eval_mae"].iloc[0] if len(val_row) > 0 else np.nan
            val_ic = val_row["eval_ic_spearman"].iloc[0] if len(val_row) > 0 else np.nan
            test_mae = test_row["test_mae"].iloc[0] if len(test_row) > 0 else np.nan
            test_ic = test_row["test_ic_spearman"].iloc[0] if len(test_row) > 0 else np.nan
            print(f"  {model_name}: val_MAE={val_mae:.4f}, val_IC={val_ic:.4f} | "
                  f"test_MAE={test_mae:.4f}, test_IC={test_ic:.4f}")
        else:
            val_auc = val_row["eval_auc"].iloc[0] if len(val_row) > 0 else np.nan
            val_ap = val_row["eval_ap"].iloc[0] if len(val_row) > 0 else np.nan
            test_auc = test_row["test_auc"].iloc[0] if len(test_row) > 0 else np.nan
            test_ap = test_row["test_ap"].iloc[0] if len(test_row) > 0 else np.nan
            print(f"  {model_name}: val_AUC={val_auc:.4f}, val_AP={val_ap:.4f} | "
                  f"test_AUC={test_auc:.4f}, test_AP={test_ap:.4f}")

    # ===== monitoring =====
    if split and "pred_buy_cls" in df.columns and "buy_signal" in df.columns:
        monitor = {
            "timestamp": datetime.now().isoformat(),
            "train_pos_rate": df.loc[split.train_idx, "buy_signal"].mean(),
            "val_pos_rate": df.loc[split.val_idx, "buy_signal"].mean(),
            "test_pos_rate": df.loc[split.test_idx, "buy_signal"].mean(),
            "pred_buy_cls_mean": df["pred_buy_cls"].mean(),
            "pred_buy_cls_median": df["pred_buy_cls"].median(),
            "pred_buy_cls_p75": df["pred_buy_cls"].quantile(0.75),
            "pred_buy_cls_p90": df["pred_buy_cls"].quantile(0.90),
            "pred_buy_cls_gt_0.7_pct": (df["pred_buy_cls"] > 0.7).mean(),
        }
        if "obs_day" in df.columns:
            monitor["obs_day_1_pct"] = (df["obs_day"] == 1).mean()
            monitor["obs_day_2_pct"] = (df["obs_day"] == 2).mean()
            monitor["obs_day_3_pct"] = (df["obs_day"] == 3).mean()

        mon_path = os.path.join(models_dir, "monitoring.csv")
        monitor_df = pd.DataFrame([monitor])
        if os.path.exists(mon_path):
            monitor_df.to_csv(mon_path, mode="a", header=False, index=False)
        else:
            monitor_df.to_csv(mon_path, index=False)
        print(f"\n  monitoring: {mon_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 Stop-Loss Clustering GBDT 模型 (v2)")
    parser.add_argument("--sample-limit", type=int, default=0, help="限制样本数（调试用）")
    parser.add_argument("--exclude-vsa", action="store_true", help="排除VSA因子（Control组，57旧特征）")
    args = parser.parse_args()
    main(args)
