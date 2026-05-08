#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[DEPRECATED] buy_signal 标签收紧对照实验: -5% vs -7%

⚠️ 已废弃：标签收紧实验（-0.05 → -0.07）已完成，结论已固化到 stop_config.py。
   BUY_CLS_THRESHOLD 已从 -0.05 收紧到 -0.07。

Purpose:
    (历史保留) 验证收紧 buy_signal 阈值是否能降低标签偏斜、改善 buy_cls 校准和回测表现。
    实验结论: -7% 优于 -5%，标签偏斜降低，回测表现提升。

Pipeline Position:
    实验逻辑（已完成，结论已固化）。

Inputs:
    - stop_experiment/output/dataset.parquet

Outputs:
    - stop_experiment/output/backtest/dynamic/label_tightening/

How to Run:
    # ⚠️ 已完成，不建议重跑
    python stop_experiment/backtest/label_tightening_experiment.py

Side Effects:
    - 训练临时 buy_cls 模型，不覆盖原始模型
    - (历史保留，已完成)
"""

from __future__ import annotations

import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, BACKTEST_DIR, TRAIN_END, VAL_END,
    LGB_PARAMS,
)

DYNAMIC_DIR = os.path.join(BACKTEST_DIR, "dynamic")
LABEL_DIR = os.path.join(DYNAMIC_DIR, "label_tightening")


def load_and_prepare():
    """加载数据集，过滤有效标签"""
    path = os.path.join(OUTPUT_DIR, "dataset.parquet")
    df = pd.read_parquet(path)
    df["selection_date"] = pd.to_datetime(df["selection_date"])

    # 过滤有效标签
    mask = df["mae_20"].notna() & (df["mae_20"] < 0.99) & (df["mae_20"] > -10)
    df = df[mask].copy()

    return df


def split_data(df):
    """按时间分割 train/val/test"""
    train_end = pd.Timestamp(TRAIN_END) - pd.Timedelta(days=25)  # embargo
    val_end = pd.Timestamp(VAL_END)
    test_end = df["selection_date"].max()

    train = df[df["selection_date"] <= train_end].copy()
    val = df[(df["selection_date"] > train_end) & (df["selection_date"] <= val_end)].copy()
    test = df[df["selection_date"] > val_end].copy()

    return train, val, test


def train_buy_cls(train_df, val_df, feature_cols, label_col, pos_label=None):
    """训练 buy_cls 模型并返回评估结果"""
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(int)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[label_col].values.astype(int)

    pos = y_train.sum()
    neg = (1 - y_train).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    params = LGB_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(0),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = lgb.train(
            params,
            lgb.Dataset(X_train, y_train),
            num_boost_round=500,
            valid_sets=[lgb.Dataset(X_val, y_val)],
            valid_names=["val"],
            callbacks=callbacks,
        )

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    return {
        "model": model,
        "pred_val": y_pred_val,
        "y_val": y_val,
        "auc_val": roc_auc_score(y_val, y_pred_val),
        "ap_val": average_precision_score(y_val, y_pred_val),
        "pos_rate_train": y_train.mean(),
        "pos_rate_val": y_val.mean(),
    }


def bucket_calibration(pred, y_true, n_buckets=10):
    """分桶校准: 返回每桶的 pred_mean vs actual_rate"""
    bins = np.linspace(0, 1, n_buckets + 1)
    bucket_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_buckets)]
    bucket_idx = np.digitize(pred, bins) - 1
    bucket_idx = np.clip(bucket_idx, 0, n_buckets - 1)

    rows = []
    for i, label in enumerate(bucket_labels):
        mask = bucket_idx == i
        if not mask.any():
            continue
        rows.append({
            "bucket": label,
            "n": mask.sum(),
            "pred_mean": pred[mask].mean(),
            "actual_rate": y_true[mask].mean(),
        })
    cal_df = pd.DataFrame(rows)
    cal_df["calibration_error"] = cal_df["pred_mean"] - cal_df["actual_rate"]
    return cal_df


def main():
    print("=" * 60)
    print("buy_signal 标签收紧对照实验: -5% vs -7%")
    print("=" * 60)

    os.makedirs(LABEL_DIR, exist_ok=True)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    df = load_and_prepare()
    train, val, test = split_data(df)
    print(f"  train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    # 特征列
    exclude = [c for c in ["mfe_20", "mae_20", "sell_signal", "buy_signal",
                           "ts_code", "signal_id", "obs_date", "trigger_date",
                           "selection_date", "split"]
               if c in df.columns]
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "float32", "int64", "int32")]
    print(f"  特征数: {len(feature_cols)}")

    # 2. 计算两种标签
    print(f"\n[2/5] 构建两种标签...")
    train["buy_signal_v5"] = (train["mae_20"] < -0.05).astype(int)
    train["buy_signal_v7"] = (train["mae_20"] < -0.07).astype(int)
    val["buy_signal_v5"] = (val["mae_20"] < -0.05).astype(int)
    val["buy_signal_v7"] = (val["mae_20"] < -0.07).astype(int)
    test["buy_signal_v5"] = (test["mae_20"] < -0.05).astype(int)
    test["buy_signal_v7"] = (test["mae_20"] < -0.07).astype(int)

    print(f"  Control   (-5%): train pos={train['buy_signal_v5'].mean():.1%}, "
          f"val={val['buy_signal_v5'].mean():.1%}, test={test['buy_signal_v5'].mean():.1%}")
    print(f"  Treatment (-7%): train pos={train['buy_signal_v7'].mean():.1%}, "
          f"val={val['buy_signal_v7'].mean():.1%}, test={test['buy_signal_v7'].mean():.1%}")

    # 3. 训练两组 buy_cls
    print(f"\n[3/5] 训练两组 buy_cls...")

    print("  --- Control (-5%) ---")
    res_v5 = train_buy_cls(train, val, feature_cols, "buy_signal_v5")
    print(f"    AUC={res_v5['auc_val']:.4f}, AP={res_v5['ap_val']:.4f}, "
          f"pos_rate(train)={res_v5['pos_rate_train']:.1%}, pos_rate(val)={res_v5['pos_rate_val']:.1%}")

    print("  --- Treatment (-7%) ---")
    res_v7 = train_buy_cls(train, val, feature_cols, "buy_signal_v7")
    print(f"    AUC={res_v7['auc_val']:.4f}, AP={res_v7['ap_val']:.4f}, "
          f"pos_rate(train)={res_v7['pos_rate_train']:.1%}, pos_rate(val)={res_v7['pos_rate_val']:.1%}")

    # 4. test 集评估
    print(f"\n[4/5] test 集评估...")
    X_test = test[feature_cols].values.astype(np.float32)

    test_pred_v5 = res_v5["model"].predict(X_test)
    test_pred_v7 = res_v7["model"].predict(X_test)
    test_y_v5 = test["buy_signal_v5"].values
    test_y_v7 = test["buy_signal_v7"].values

    auc_v5 = roc_auc_score(test_y_v5, test_pred_v5)
    auc_v7 = roc_auc_score(test_y_v7, test_pred_v7)

    print(f"  test AUC: v5={auc_v5:.4f}, v7={auc_v7:.4f}")

    # 5. 分桶校准对比
    print(f"\n[5/5] 分桶校准对比...")
    cal_v5 = bucket_calibration(test_pred_v5, test_y_v5)
    cal_v7 = bucket_calibration(test_pred_v7, test_y_v7)

    # 合并对比
    cal_compare = cal_v5[["bucket", "n", "pred_mean", "actual_rate", "calibration_error"]].copy()
    cal_compare.columns = ["bucket", "n_v5", "pred_v5", "actual_v5", "cal_err_v5"]
    cal_v7_r = cal_v7[["n", "pred_mean", "actual_rate", "calibration_error"]].copy()
    cal_v7_r.columns = ["n_v7", "pred_v7", "actual_v7", "cal_err_v7"]
    cal_compare = pd.concat([cal_compare, cal_v7_r], axis=1)

    # 全局校准误差 (MAE)
    mae_cal_v5 = abs(test_pred_v5.mean() - test_y_v5.mean())
    mae_cal_v7 = abs(test_pred_v7.mean() - test_y_v7.mean())

    print(f"\n  校准误差 (|pred_mean - actual|):")
    print(f"    Control (-5%):   pred_mean={test_pred_v5.mean():.4f}, actual={test_y_v5.mean():.4f}, "
          f"error={mae_cal_v5:.4f}")
    print(f"    Treatment (-7%): pred_mean={test_pred_v7.mean():.4f}, actual={test_y_v7.mean():.4f}, "
          f"error={mae_cal_v7:.4f}")

    print(f"\n  分桶校准详情:")
    print(cal_compare.to_string(index=False))

    # 保存
    path_compare = os.path.join(LABEL_DIR, "calibration_comparison.csv")
    cal_compare.to_csv(path_compare, index=False)

    # 汇总
    summary = pd.DataFrame([
        {"label": "buy_signal (mae_20 < -5%)", "label_short": "v5",
         "train_pos_rate": train["buy_signal_v5"].mean(),
         "val_pos_rate": val["buy_signal_v5"].mean(),
         "test_pos_rate": test["buy_signal_v5"].mean(),
         "val_AUC": res_v5["auc_val"], "test_AUC": auc_v5,
         "global_cal_error": mae_cal_v5,
         "pred_mean_test": test_pred_v5.mean(), "actual_mean_test": test_y_v5.mean()},
        {"label": "buy_signal (mae_20 < -7%)", "label_short": "v7",
         "train_pos_rate": train["buy_signal_v7"].mean(),
         "val_pos_rate": val["buy_signal_v7"].mean(),
         "test_pos_rate": test["buy_signal_v7"].mean(),
         "val_AUC": res_v7["auc_val"], "test_AUC": auc_v7,
         "global_cal_error": mae_cal_v7,
         "pred_mean_test": test_pred_v7.mean(), "actual_mean_test": test_y_v7.mean()},
    ])

    print(f"\n{'='*80}")
    print("汇总对比")
    print(f"{'='*80}")
    print(summary[["label_short", "train_pos_rate", "val_pos_rate", "test_pos_rate",
                   "val_AUC", "test_AUC", "global_cal_error"]].to_string(index=False))

    path_summary = os.path.join(LABEL_DIR, "summary.csv")
    summary.to_csv(path_summary, index=False)
    print(f"\n结果保存在: {LABEL_DIR}")

    # 结论
    improvement = mae_cal_v5 - mae_cal_v7
    auc_loss = auc_v5 - auc_v7
    print(f"\n结论:")
    print(f"  校准改善: {improvement:+.4f} (正=改善)")
    print(f"  AUC损失:   {auc_loss:+.4f}  (负=改善)")
    print(f"  正类率:    {train['buy_signal_v5'].mean():.1%} → {train['buy_signal_v7'].mean():.1%}")
    if improvement > 0:
        print(f"  ✅ 标签收紧改善了校准 (误差降低 {improvement:.4f})")
    else:
        print(f"  ❌ 标签收紧未能改善校准")


if __name__ == "__main__":
    main()
