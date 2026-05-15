#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证集泄漏实验: 对比 train-only 重训 vs train+val 重训的模型效果差异

Purpose:
    评估验证集数据通过"train+val 重训"间接影响最终模型的程度。
    策略 A (当前): train 早停 → best_iteration → train+val 重训 (_final.txt)
    策略 B (纯 train): train 早停 → best_iteration → 仅 train 重训

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet
      (含全量数据 + 4 模型预测列 + 特征列 + 标签列)

Outputs:
    - experiments/val_leakage/results/model_metrics_comparison.csv
    - experiments/val_leakage/results/prediction_distribution.csv
    - experiments/val_leakage/results/verdict.json
    - experiments/val_leakage/results/tmp_models/ (临时模型文件，实验后可删)

How to Run:
    python -m stop_experiment.experiments.val_leakage.01_train_only_vs_trainval
    python -m stop_experiment.experiments.val_leakage.01_train_only_vs_trainval --sample-limit 5000

Examples:
    # 完整运行（约 2~3 分钟）
    python -m stop_experiment.experiments.val_leakage.01_train_only_vs_trainval

    # 快速调试（限制样本数）
    python -m stop_experiment.experiments.val_leakage.01_train_only_vs_trainval --sample-limit 5000

Side Effects:
    - 只读 candidate_with_scores.parquet
    - 输出写入 experiments/val_leakage/results/（含临时模型目录）
    - 不覆盖现有模型文件
"""

from __future__ import annotations

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import importlib

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, roc_auc_score

from stop_experiment.pipeline.stop_config import (
    LGB_PARAMS, MODEL_SPECS, MODELS_DIR,
    OBS_TRAIN_END, OBS_VAL_END, EMBARGO_DAYS,
)
from stop_experiment.pipeline.factor_columns import ALL_FEATURE_COLS

_train_module = importlib.import_module("stop_experiment.pipeline.02_train_gbdt_models")
build_train_val_test_split = _train_module.build_train_val_test_split
train_single_model = _train_module.train_single_model
retrain_final_model = _train_module.retrain_final_model

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
TMP_MODELS_DIR = os.path.join(RESULTS_DIR, "tmp_models")

STRATEGY_A = "trainval_retrain"
STRATEGY_B = "trainonly_retrain"


def load_data(scores_path: str, sample_limit: int = 0) -> pd.DataFrame:
    df = pd.read_parquet(scores_path)
    if sample_limit > 0:
        df = df.head(sample_limit)
    df = df.dropna(subset=["mfe_20", "mae_20"])
    df = df.sort_values("obs_date").reset_index(drop=True)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  缺少特征: {missing}")
    return feature_cols


def compute_scale_pos_weight(df: pd.DataFrame, idx: np.ndarray, target: str) -> float | None:
    pos_rate = df.loc[idx, target].mean()
    if pos_rate > 0 and pos_rate < 1:
        return (1 - pos_rate) / pos_rate
    return None


def evaluate_model(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_idx: np.ndarray,
    task: str,
) -> dict:
    y_test = df.loc[test_idx, target_col]
    y_pred = model.predict(df.loc[test_idx, feature_cols], num_iteration=model.best_iteration)
    valid_mask = y_test.notna()
    y_v = y_test[valid_mask].values
    p_v = y_pred[valid_mask]

    metrics = {"n_valid": len(y_v)}
    if task == "regression":
        metrics["MAE"] = mean_absolute_error(y_v, p_v) if len(y_v) > 0 else np.nan
        metrics["IC"] = stats.spearmanr(y_v, p_v)[0] if len(y_v) > 10 else np.nan
    else:
        if len(np.unique(y_v)) > 1 and len(y_v) > 0:
            metrics["AUC"] = roc_auc_score(y_v, p_v)
        else:
            metrics["AUC"] = np.nan
    return metrics


def build_metrics_comparison(
    results: dict,
) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_SPECS:
        spec = MODEL_SPECS[model_name]
        task = "classification" if spec["objective"] == "binary" else "regression"
        for strategy in [STRATEGY_A, STRATEGY_B]:
            entry = {
                "model": model_name,
                "task": task,
                "strategy": strategy,
                "best_iteration": results[model_name]["best_iteration"],
            }
            m = results[model_name][strategy]["test_metrics"]
            entry.update(m)
            rows.append(entry)
    return pd.DataFrame(rows)


def build_prediction_distribution(
    df: pd.DataFrame,
    results: dict,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_SPECS:
        pred_a = results[model_name][STRATEGY_A]["test_pred"]
        pred_b = results[model_name][STRATEGY_B]["test_pred"]
        valid = ~np.isnan(pred_a) & ~np.isnan(pred_b)
        pa = pred_a[valid]
        pb = pred_b[valid]

        for strategy, pred in [(STRATEGY_A, pa), (STRATEGY_B, pb)]:
            rows.append({
                "model": model_name,
                "strategy": strategy,
                "mean": float(np.mean(pred)),
                "median": float(np.median(pred)),
                "std": float(np.std(pred)),
                "p10": float(np.percentile(pred, 10)),
                "p90": float(np.percentile(pred, 90)),
                "min": float(np.min(pred)),
                "max": float(np.max(pred)),
            })

        if len(pa) > 10 and len(pb) > 10:
            spearman_r, spearman_p = stats.spearmanr(pa, pb)
            rows.append({
                "model": model_name,
                "strategy": "spearman_corr",
                "mean": spearman_r,
                "median": spearman_p,
                "std": np.nan,
                "p10": np.nan,
                "p90": np.nan,
                "min": np.nan,
                "max": np.nan,
            })
    return pd.DataFrame(rows)


def build_verdict(metrics_df: pd.DataFrame, dist_df: pd.DataFrame) -> dict:
    verdict = {
        "conclusion": "unknown",
        "val_leakage_risk": "unknown",
        "details": {},
    }

    model_diffs = {}
    for model_name in MODEL_SPECS:
        spec = MODEL_SPECS[model_name]
        task = "classification" if spec["objective"] == "binary" else "regression"

        a_row = metrics_df[(metrics_df["model"] == model_name) & (metrics_df["strategy"] == STRATEGY_A)]
        b_row = metrics_df[(metrics_df["model"] == model_name) & (metrics_df["strategy"] == STRATEGY_B)]

        if len(a_row) == 0 or len(b_row) == 0:
            continue

        diff = {}
        if task == "regression":
            mae_a = float(a_row["MAE"].iloc[0])
            mae_b = float(b_row["MAE"].iloc[0])
            ic_a = float(a_row["IC"].iloc[0])
            ic_b = float(b_row["IC"].iloc[0])
            mae_diff_pct = (mae_a - mae_b) / abs(mae_b) if abs(mae_b) > 1e-10 else 0.0
            ic_diff = ic_a - ic_b
            diff = {
                "MAE_A": mae_a, "MAE_B": mae_b, "MAE_diff_pct": mae_diff_pct,
                "IC_A": ic_a, "IC_B": ic_b, "IC_diff": ic_diff,
            }
        else:
            auc_a = float(a_row["AUC"].iloc[0])
            auc_b = float(b_row["AUC"].iloc[0])
            auc_diff = auc_a - auc_b
            diff = {"AUC_A": auc_a, "AUC_B": auc_b, "AUC_diff": auc_diff}

        spearman_row = dist_df[(dist_df["model"] == model_name) & (dist_df["strategy"] == "spearman_corr")]
        if len(spearman_row) > 0:
            diff["spearman_corr"] = float(spearman_row["mean"].iloc[0])

        model_diffs[model_name] = diff

    verdict["details"]["per_model"] = model_diffs

    max_mae_diff_pct = 0.0
    max_auc_diff = 0.0
    min_spearman = 1.0
    for model_name, d in model_diffs.items():
        if "MAE_diff_pct" in d:
            max_mae_diff_pct = max(max_mae_diff_pct, abs(d["MAE_diff_pct"]))
        if "AUC_diff" in d:
            max_auc_diff = max(max_auc_diff, abs(d["AUC_diff"]))
        if "spearman_corr" in d:
            min_spearman = min(min_spearman, d["spearman_corr"])

    verdict["details"]["max_mae_diff_pct"] = max_mae_diff_pct
    verdict["details"]["max_auc_diff"] = max_auc_diff
    verdict["details"]["min_spearman_corr"] = min_spearman

    if max_mae_diff_pct > 0.05 or max_auc_diff > 0.02:
        verdict["val_leakage_risk"] = "high"
        verdict["conclusion"] = (
            "train+val 重训带来显著性能提升，验证集存在间接泄漏风险，"
            "建议改用 train-only 重训或增加 embargo"
        )
    elif max_mae_diff_pct > 0.02 or max_auc_diff > 0.01:
        verdict["val_leakage_risk"] = "medium"
        verdict["conclusion"] = (
            "train+val 重训带来轻微性能提升，验证集泄漏风险中等，"
            "两种策略预测高度相关，当前做法可接受但需关注"
        )
    else:
        verdict["val_leakage_risk"] = "low"
        verdict["conclusion"] = (
            "train+val 重训与 train-only 重训效果几乎一致，"
            "验证集泄漏风险低，当前做法无显著问题"
        )

    return verdict


def main():
    parser = argparse.ArgumentParser(description="验证集泄漏实验: train-only vs train+val 重训")
    parser.add_argument("--sample-limit", type=int, default=0, help="限制样本数（调试用，0=不限）")
    args = parser.parse_args()

    print("=" * 60)
    print("验证集泄漏实验: train-only vs train+val 重训")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TMP_MODELS_DIR, exist_ok=True)

    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在，请先运行 02_train_gbdt_models.py")

    print("\n[1/4] 加载数据...")
    df = load_data(scores_path, args.sample_limit)
    print(f"  总行数: {len(df)}")

    feature_cols = get_feature_cols(df)
    print(f"  特征数: {len(feature_cols)}")

    print("\n[2/4] 构建 train/val/test 分割...")
    split = build_train_val_test_split(df)

    train_val_idx = np.concatenate([split.train_idx, split.val_idx])

    print("\n[3/4] 训练模型 (4 模型 × 2 策略)...")
    results = {}

    for model_name, spec in MODEL_SPECS.items():
        target = spec["target"]
        objective = spec["objective"]
        metric_name = spec["metric"]
        task = "classification" if objective == "binary" else "regression"

        print(f"\n  --- {model_name} (target={target}, task={task}) ---")

        params = LGB_PARAMS.copy()
        params["objective"] = objective
        params["metric"] = metric_name

        if task == "classification":
            spw = compute_scale_pos_weight(df, split.train_idx, target)
            if spw is not None:
                params["scale_pos_weight"] = spw
            print(f"    正类比例(train): {df.loc[split.train_idx, target].mean():.3f}, "
                  f"scale_pos_weight: {params.get('scale_pos_weight', 1.0):.2f}")

        model_es, val_metrics, _ = train_single_model(
            df, feature_cols, target,
            split.train_idx, split.val_idx,
            params, task,
            model_name_tag=f"{model_name}_es",
        )
        best_iter = model_es.best_iteration
        print(f"    best_iteration: {best_iter}")
        print(f"    val 指标: {val_metrics}")

        results[model_name] = {"best_iteration": best_iter}

        # 策略 A: train+val 重训
        params_a = LGB_PARAMS.copy()
        params_a["objective"] = objective
        params_a["metric"] = metric_name
        if task == "classification":
            spw_a = compute_scale_pos_weight(df, train_val_idx, target)
            if spw_a is not None:
                params_a["scale_pos_weight"] = spw_a

        model_a = retrain_final_model(
            df, feature_cols, target,
            train_val_idx, params_a, best_iter,
            task, model_name_tag=f"{model_name}_A",
        )
        model_a.save_model(os.path.join(TMP_MODELS_DIR, f"{model_name}_A.txt"))

        test_metrics_a = evaluate_model(model_a, df, feature_cols, target, split.test_idx, task)
        test_pred_a = model_a.predict(df.loc[split.test_idx, feature_cols], num_iteration=model_a.best_iteration)
        results[model_name][STRATEGY_A] = {
            "test_metrics": test_metrics_a,
            "test_pred": test_pred_a,
        }
        print(f"    策略A (train+val): {test_metrics_a}")

        # 策略 B: 仅 train 重训
        params_b = LGB_PARAMS.copy()
        params_b["objective"] = objective
        params_b["metric"] = metric_name
        if task == "classification":
            spw_b = compute_scale_pos_weight(df, split.train_idx, target)
            if spw_b is not None:
                params_b["scale_pos_weight"] = spw_b

        model_b = retrain_final_model(
            df, feature_cols, target,
            split.train_idx, params_b, best_iter,
            task, model_name_tag=f"{model_name}_B",
        )
        model_b.save_model(os.path.join(TMP_MODELS_DIR, f"{model_name}_B.txt"))

        test_metrics_b = evaluate_model(model_b, df, feature_cols, target, split.test_idx, task)
        test_pred_b = model_b.predict(df.loc[split.test_idx, feature_cols], num_iteration=model_b.best_iteration)
        results[model_name][STRATEGY_B] = {
            "test_metrics": test_metrics_b,
            "test_pred": test_pred_b,
        }
        print(f"    策略B (train-only): {test_metrics_b}")

    print("\n[4/4] 生成对比结果...")

    metrics_df = build_metrics_comparison(results)
    metrics_path = os.path.join(RESULTS_DIR, "model_metrics_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  模型指标对比: {metrics_path}")

    dist_df = build_prediction_distribution(df, results, split.test_idx)
    dist_path = os.path.join(RESULTS_DIR, "prediction_distribution.csv")
    dist_df.to_csv(dist_path, index=False)
    print(f"  预测分布对比: {dist_path}")

    verdict = build_verdict(metrics_df, dist_df)
    verdict_path = os.path.join(RESULTS_DIR, "verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print(f"  结论: {verdict_path}")

    print(f"\n{'=' * 60}")
    print("模型指标对比汇总")
    print(f"{'=' * 60}")
    print(f"  {'模型':12s} {'策略':20s} {'指标':>10s}")
    for _, row in metrics_df.iterrows():
        model = row["model"]
        strategy = row["strategy"]
        task = row["task"]
        if task == "regression":
            mae = row.get("MAE", np.nan)
            ic = row.get("IC", np.nan)
            print(f"  {model:12s} {strategy:20s} MAE={mae:.4f}  IC={ic:.4f}")
        else:
            auc = row.get("AUC", np.nan)
            print(f"  {model:12s} {strategy:20s} AUC={auc:.4f}")

    print(f"\n{'=' * 60}")
    print("预测排序相关性 (Spearman)")
    print(f"{'=' * 60}")
    for model_name in MODEL_SPECS:
        pred_a = results[model_name][STRATEGY_A]["test_pred"]
        pred_b = results[model_name][STRATEGY_B]["test_pred"]
        valid = ~np.isnan(pred_a) & ~np.isnan(pred_b)
        if valid.sum() > 10:
            r, p = stats.spearmanr(pred_a[valid], pred_b[valid])
            print(f"  {model_name}: ρ={r:.4f} (p={p:.2e})")

    print(f"\n{'=' * 60}")
    print(f"结论: {verdict['conclusion']}")
    print(f"验证集泄漏风险: {verdict['val_leakage_risk']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
