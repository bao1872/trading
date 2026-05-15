# -*- coding: utf-8 -*-
"""
GBDT 模型训练脚本

Purpose:
    使用 LightGBM 训练 buy_now / sell_now 二分类模型，含早停、验证指标、特征重要性输出

Inputs:
    - dataset parquet 文件，含列: stock_id, trade_date, ALL_FEATURE_COLS, LABEL_COLS, split ('train'/'val'/'test')

Outputs:
    - {output_dir}/{model_name}.lgb: LightGBM 模型文件
    - {output_dir}/{model_name}_feature_importance.csv: 特征重要性 CSV

How to Run:
    python bid_experiment/03_train_gbdt.py
    python bid_experiment/03_train_gbdt.py --dataset-path bid_experiment/output/dataset/dataset.parquet --output-dir bid_experiment/output/models

Side Effects:
    - 写文件（模型 .lgb + 特征重要性 .csv），不写数据库
"""

import argparse
import logging
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bid_experiment.bid_config import DATASET_DIR, LGB_PARAMS, MODELS_DIR, MODEL_SPECS
from bid_experiment.feature_columns import ALL_FEATURE_COLS

logger = logging.getLogger(__name__)

NUM_BOOST_ROUND = 1000


def _compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def _compute_val_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pos_count = y_true.sum()
    if pos_count == 0 or pos_count == len(y_true):
        logger.warning(
            f"验证集正样本数={pos_count}/{len(y_true)}，无法计算 AUC/PR-AUC"
        )
        return {"AUC": np.nan, "PR-AUC": np.nan, "LogLoss": _compute_log_loss(y_true, y_pred)}

    auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    log_loss = _compute_log_loss(y_true, y_pred)
    return {"AUC": auc, "PR-AUC": pr_auc, "LogLoss": log_loss}


def train_models(dataset_path: str, output_dir: str) -> dict:
    """训练全部 MODEL_SPECS 中的模型

    Args:
        dataset_path: 数据集 parquet 文件路径
        output_dir: 模型输出目录

    Returns:
        各模型训练结果摘要 dict
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"加载数据集: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    logger.info(f"数据集形状: {df.shape}, split 分布: {df['split'].value_counts().to_dict()}")

    results = {}

    for model_name, spec in MODEL_SPECS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"开始训练模型: {model_name}")
        logger.info(f"{'='*60}")

        target_col = spec["target"]
        objective = spec["objective"]
        metric = spec["metric"]

        feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
        missing_features = set(ALL_FEATURE_COLS) - set(feature_cols)
        if missing_features:
            logger.warning(f"数据集缺少特征列: {missing_features}")

        all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
        if all_nan_cols:
            logger.warning(f"特征列全为 NaN，将排除: {all_nan_cols}")
            feature_cols = [c for c in feature_cols if c not in all_nan_cols]

        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()

        train_df = train_df.dropna(subset=[target_col])
        val_df = val_df.dropna(subset=[target_col])

        logger.info(f"训练集: {len(train_df)} 行, 验证集: {len(val_df)} 行")

        train_pos = train_df[target_col].sum()
        val_pos = val_df[target_col].sum()
        logger.info(
            f"训练集正样本: {train_pos}/{len(train_df)} ({train_pos/len(train_df):.4f})"
        )
        logger.info(
            f"验证集正样本: {val_pos}/{len(val_df)} ({val_pos/len(val_df):.4f})"
        )

        if train_pos == 0:
            logger.error(f"训练集无正样本，跳过模型 {model_name}")
            results[model_name] = {"status": "skipped", "reason": "no positive samples in train"}
            continue

        if val_pos == 0:
            logger.error(f"验证集无正样本，跳过模型 {model_name}")
            results[model_name] = {"status": "skipped", "reason": "no positive samples in val"}
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)

        params = dict(LGB_PARAMS)
        params["objective"] = objective
        params["metric"] = metric

        callbacks = [
            lgb.log_evaluation(100),
            lgb.early_stopping(50),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        model_path = os.path.join(output_dir, f"{model_name}.lgb")
        model.save_model(model_path)
        logger.info(f"模型已保存: {model_path}")

        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        metrics = _compute_val_metrics(y_val.values, y_val_pred)

        logger.info(f"验证集指标: AUC={metrics['AUC']:.4f}, PR-AUC={metrics['PR-AUC']:.4f}, LogLoss={metrics['LogLoss']:.4f}")

        importance = model.feature_importance(importance_type="gain")
        feat_imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance_gain": importance,
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

        imp_path = os.path.join(output_dir, f"{model_name}_feature_importance.csv")
        feat_imp_df.to_csv(imp_path, index=False)
        logger.info(f"特征重要性已保存: {imp_path}")

        results[model_name] = {
            "status": "ok",
            "best_iteration": model.best_iteration,
            "val_metrics": metrics,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "train_pos_rate": train_pos / len(train_df),
            "val_pos_rate": val_pos / len(val_df),
            "model_path": model_path,
            "importance_path": imp_path,
        }

    return results


def _print_summary(results: dict) -> None:
    print(f"\n{'='*70}")
    print("模型训练摘要")
    print(f"{'='*70}")
    for name, info in results.items():
        print(f"\n--- {name} ---")
        if info["status"] != "ok":
            print(f"  状态: {info['status']} ({info['reason']})")
            continue
        print(f"  最佳迭代: {info['best_iteration']}")
        print(f"  训练样本: {info['train_samples']} (正样本率={info['train_pos_rate']:.4f})")
        print(f"  验证样本: {info['val_samples']} (正样本率={info['val_pos_rate']:.4f})")
        m = info["val_metrics"]
        print(f"  AUC={m['AUC']:.4f}, PR-AUC={m['PR-AUC']:.4f}, LogLoss={m['LogLoss']:.4f}")
        print(f"  模型: {info['model_path']}")
        print(f"  特征重要性: {info['importance_path']}")
    print(f"\n{'='*70}")


def parse_args():
    parser = argparse.ArgumentParser(description="GBDT 模型训练脚本")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="数据集 parquet 文件路径（默认自动检测 DATASET_DIR）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="模型输出目录（默认 MODELS_DIR）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dataset_path = args.dataset_path
    if dataset_path is None:
        default_path = os.path.join(DATASET_DIR, "dataset.parquet")
        if os.path.exists(default_path):
            dataset_path = default_path
        else:
            print(f"未找到默认数据集: {default_path}，请通过 --dataset-path 指定")
            sys.exit(1)

    output_dir = args.output_dir if args.output_dir is not None else MODELS_DIR

    results = train_models(dataset_path, output_dir)
    _print_summary(results)


if __name__ == "__main__":
    main()
