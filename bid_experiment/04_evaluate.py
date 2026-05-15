# -*- coding: utf-8 -*-
"""
竞价拍卖 GBDT 模型评估报告

Purpose:
    加载训练好的 LightGBM 模型（buy_now / sell_now），在测试集上生成预测并输出完整评估报告，
    包括分类指标、分数桶分析、交易指标、稳定性指标。

Inputs:
    - 数据集 parquet（含 stock_id, trade_date, ALL_FEATURE_COLS, LABEL_COLS, split）
    - MODELS_DIR 下的 LightGBM 模型文件（buy_now.txt, sell_now.txt）

Outputs:
    - output_dir/evaluation_report.txt   完整文本评估报告
    - output_dir/bucket_analysis.csv     分数桶统计

How to Run:
    python bid_experiment/04_evaluate.py --dataset-path bid_experiment/output/dataset/dataset.parquet
    python bid_experiment/04_evaluate.py --dataset-path bid_experiment/output/dataset/dataset.parquet --models-dir bid_experiment/output/models

Side Effects:
    - 写文件（evaluation_report.txt, bucket_analysis.csv），不写数据库
"""

import argparse
import logging
import os
import sys
from io import StringIO

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bid_experiment.bid_config import MODELS_DIR, OUTPUT_DIR
from bid_experiment.feature_columns import ALL_FEATURE_COLS, LABEL_COLS

logger = logging.getLogger(__name__)

MODEL_NAMES = ["buy_now", "sell_now"]
N_BUCKETS = 10
HIGH_QUANTILE = 0.80
LOW_QUANTILE = 0.20


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    logger.info(f"加载数据集: {len(df)} 行, {len(df.columns)} 列")
    return df


def _load_model(model_name: str, models_dir: str) -> lgb.Booster:
    for ext in [".lgb", ".txt"]:
        model_path = os.path.join(models_dir, f"{model_name}{ext}")
        if os.path.exists(model_path):
            model = lgb.Booster(model_file=model_path)
            logger.info(f"加载模型: {model_path}")
            return model
    raise FileNotFoundError(f"模型文件不存在: {models_dir}/{model_name}.lgb 或 .txt")


def _get_test_data(df: pd.DataFrame) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise ValueError("测试集为空，无法评估")
    logger.info(f"测试集样本数: {len(test)}")
    return test


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(np.unique(y_true)) < 2:
        logger.warning("标签只有单一类别，AUC/PR-AUC 不可计算")
        return {
            "auc": float("nan"),
            "pr_auc": float("nan"),
            "log_loss": float("nan"),
            "accuracy": float("nan"),
        }

    auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    log_loss = -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
    y_hat = (y_pred >= 0.5).astype(int)
    accuracy = np.mean(y_hat == y_true)

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "log_loss": log_loss,
        "accuracy": accuracy,
    }


def compute_bucket_analysis(y_true: np.ndarray, y_pred: np.ndarray, n_buckets: int = N_BUCKETS) -> pd.DataFrame:
    buckets = pd.qcut(y_pred, n_buckets, duplicates="drop")
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "bucket": buckets})
    stats = (
        df.groupby("bucket", observed=True)
        .agg(count=("y_true", "size"), positive_rate=("y_true", "mean"), mean_score=("y_pred", "mean"))
        .reset_index()
    )
    stats["bucket"] = stats["bucket"].astype(str)
    return stats


def compute_monotonicity(bucket_stats: pd.DataFrame) -> dict:
    if len(bucket_stats) < 3:
        return {"spearman_corr": float("nan"), "spearman_pvalue": float("nan"), "is_monotonic": False}
    corr, pval = spearmanr(range(len(bucket_stats)), bucket_stats["positive_rate"].values)
    is_mono = bucket_stats["positive_rate"].is_monotonic_increasing or bucket_stats["positive_rate"].is_monotonic_decreasing
    return {"spearman_corr": corr, "spearman_pvalue": pval, "is_monotonic": is_mono}


def compute_buy_trading_metrics(test: pd.DataFrame, y_pred: np.ndarray) -> dict:
    high_thresh = np.quantile(y_pred, HIGH_QUANTILE)
    low_thresh = np.quantile(y_pred, LOW_QUANTILE)

    high_mask = y_pred >= high_thresh
    low_mask = y_pred <= low_thresh

    ret_cols = ["RET_10m", "RET_30m", "MFE_10m", "MAE_10m"]
    for col in ret_cols:
        if col not in test.columns:
            logger.warning(f"缺少列 {col}，buy 交易指标不完整")
            return {"high_score_count": int(high_mask.sum()), "low_score_count": int(low_mask.sum())}

    high = test.loc[high_mask]
    low = test.loc[low_mask]

    high_ret_10m = high["RET_10m"].values
    win_mask = high_ret_10m > 0
    win_rate = win_mask.mean() if len(high_ret_10m) > 0 else float("nan")
    pos_mean = high_ret_10m[win_mask].mean() if win_mask.any() else float("nan")
    neg_mean = high_ret_10m[~win_mask].mean() if (~win_mask).any() else float("nan")
    pl_ratio = pos_mean / abs(neg_mean) if (neg_mean != 0 and not np.isnan(neg_mean)) else float("nan")

    result = {
        "high_score_count": int(high_mask.sum()),
        "low_score_count": int(low_mask.sum()),
        "high_mean_RET_10m": high["RET_10m"].mean(),
        "high_mean_RET_30m": high["RET_30m"].mean(),
        "high_mean_MFE_10m": high["MFE_10m"].mean(),
        "high_mean_MAE_10m": high["MAE_10m"].mean(),
        "high_win_rate": win_rate,
        "high_profit_loss_ratio": pl_ratio,
        "low_mean_RET_10m": low["RET_10m"].mean(),
        "low_mean_RET_30m": low["RET_30m"].mean(),
        "high_vs_low_RET_10m_diff": high["RET_10m"].mean() - low["RET_10m"].mean(),
    }
    return result


def compute_sell_trading_metrics(test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    high_thresh = np.quantile(y_pred, HIGH_QUANTILE)
    low_thresh = np.quantile(y_pred, LOW_QUANTILE)

    high_mask = y_pred >= high_thresh
    low_mask = y_pred <= low_thresh

    if "future_min_ret_15m" not in test.columns:
        logger.warning("缺少列 future_min_ret_15m，sell 交易指标不完整")
        return {"high_score_count": int(high_mask.sum()), "low_score_count": int(low_mask.sum())}

    high = test.loc[high_mask]
    low = test.loc[low_mask]

    sell_miss_mask = (y_true == 0) & high_mask
    sell_miss_rate = sell_miss_mask.sum() / high_mask.sum() if high_mask.sum() > 0 else float("nan")

    result = {
        "high_score_count": int(high_mask.sum()),
        "low_score_count": int(low_mask.sum()),
        "high_mean_future_min_ret_15m": high["future_min_ret_15m"].mean(),
        "high_avoided_drawdown": -high["future_min_ret_15m"].mean(),
        "low_mean_future_min_ret_15m": low["future_min_ret_15m"].mean(),
        "sell_miss_rate": sell_miss_rate,
        "high_vs_low_future_min_ret_15m_diff": high["future_min_ret_15m"].mean() - low["future_min_ret_15m"].mean(),
    }
    return result


def compute_stability_metrics(test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if "trade_date" not in test.columns:
        return {"auc_by_year": {}, "note": "缺少 trade_date 列"}

    test_copy = test.copy()
    test_copy["_y_true"] = y_true
    test_copy["_y_pred"] = y_pred
    test_copy["year"] = pd.to_datetime(test_copy["trade_date"].astype(str)).dt.year

    auc_by_year = {}
    for year, grp in test_copy.groupby("year"):
        yt = grp["_y_true"].values
        yp = grp["_y_pred"].values
        if len(np.unique(yt)) < 2:
            auc_by_year[year] = float("nan")
            logger.warning(f"年份 {year} 标签单一，AUC 不可计算")
        else:
            auc_by_year[year] = roc_auc_score(yt, yp)

    return {"auc_by_year": auc_by_year}


def _format_report(
    model_name: str,
    cls_metrics: dict,
    bucket_stats: pd.DataFrame,
    mono: dict,
    trading_metrics: dict,
    stability: dict,
    test_count: int,
) -> str:
    buf = StringIO()
    sep = "=" * 70

    buf.write(f"\n{sep}\n")
    buf.write(f"  模型评估报告: {model_name}\n")
    buf.write(f"  测试集样本数: {test_count}\n")
    buf.write(f"{sep}\n\n")

    buf.write("--- 分类指标 ---\n")
    buf.write(f"  AUC:       {cls_metrics['auc']:.4f}\n")
    buf.write(f"  PR-AUC:    {cls_metrics['pr_auc']:.4f}\n")
    buf.write(f"  LogLoss:   {cls_metrics['log_loss']:.4f}\n")
    buf.write(f"  Accuracy:  {cls_metrics['accuracy']:.4f}\n\n")

    buf.write("--- 分数桶分析 ---\n")
    buf.write(bucket_stats.to_string(index=False))
    buf.write("\n\n")

    buf.write("--- 单调性 ---\n")
    buf.write(f"  Spearman 相关系数: {mono['spearman_corr']:.4f}  (p={mono['spearman_pvalue']:.4f})\n")
    buf.write(f"  严格单调: {'是' if mono['is_monotonic'] else '否'}\n\n")

    buf.write("--- 交易指标 ---\n")
    for k, v in trading_metrics.items():
        if isinstance(v, float):
            buf.write(f"  {k}: {v:.6f}\n")
        else:
            buf.write(f"  {k}: {v}\n")
    buf.write("\n")

    buf.write("--- 稳定性指标 ---\n")
    auc_by_year = stability.get("auc_by_year", {})
    if auc_by_year:
        for year, auc_val in sorted(auc_by_year.items()):
            buf.write(f"  {year} AUC: {auc_val:.4f}\n")
    else:
        buf.write("  无年份维度数据\n")
    buf.write("\n")

    return buf.getvalue()


def evaluate_models(dataset_path: str, models_dir: str, output_dir: str) -> dict:
    df = _load_dataset(dataset_path)
    test = _get_test_data(df)

    feature_cols = [c for c in ALL_FEATURE_COLS if c in test.columns]
    all_nan_cols = [c for c in feature_cols if test[c].isna().all()]
    if all_nan_cols:
        logger.warning(f"特征列全为 NaN，将排除: {all_nan_cols}")
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]
    missing = set(ALL_FEATURE_COLS) - set(feature_cols)
    if missing:
        logger.warning(f"测试集缺少特征列: {missing}")

    X_test = test[feature_cols]
    all_reports = {}
    all_bucket_dfs = []

    for model_name in MODEL_NAMES:
        try:
            model = _load_model(model_name, models_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        target_col = f"y_{model_name}"
        if target_col not in test.columns:
            logger.error(f"测试集缺少标签列: {target_col}")
            continue

        y_true = test[target_col].values.astype(float)
        y_pred = model.predict(X_test)

        if np.all(y_pred == y_pred[0]):
            logger.warning(f"{model_name} 所有预测值相同: {y_pred[0]}")

        cls_metrics = compute_classification_metrics(y_true, y_pred)
        bucket_stats = compute_bucket_analysis(y_true, y_pred)
        mono = compute_monotonicity(bucket_stats)
        stability = compute_stability_metrics(test, y_true, y_pred)

        if model_name == "buy_now":
            trading_metrics = compute_buy_trading_metrics(test, y_pred)
        elif model_name == "sell_now":
            trading_metrics = compute_sell_trading_metrics(test, y_true, y_pred)
        else:
            trading_metrics = {}

        report_text = _format_report(
            model_name, cls_metrics, bucket_stats, mono, trading_metrics, stability, len(test)
        )
        all_reports[model_name] = report_text

        bucket_stats["model"] = model_name
        all_bucket_dfs.append(bucket_stats)

        logger.info(f"{model_name} 评估完成: AUC={cls_metrics['auc']:.4f}")

    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("竞价拍卖 GBDT 模型评估报告\n")
        f.write(f"数据集: {dataset_path}\n")
        f.write(f"模型目录: {models_dir}\n")
        for text in all_reports.values():
            f.write(text)
    logger.info(f"评估报告已保存: {report_path}")

    if all_bucket_dfs:
        bucket_df = pd.concat(all_bucket_dfs, ignore_index=True)
        bucket_path = os.path.join(output_dir, "bucket_analysis.csv")
        bucket_df.to_csv(bucket_path, index=False)
        logger.info(f"分数桶分析已保存: {bucket_path}")

    return all_reports


def parse_args():
    parser = argparse.ArgumentParser(description="竞价拍卖 GBDT 模型评估")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.path.join(OUTPUT_DIR, "dataset", "dataset.parquet"),
        help="数据集 parquet 文件路径",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=MODELS_DIR,
        help="模型文件目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="评估报告输出目录（默认 OUTPUT_DIR/evaluation）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = args.output_dir or os.path.join(OUTPUT_DIR, "evaluation")

    logger.info(f"开始评估: dataset={args.dataset_path}, models={args.models_dir}, output={output_dir}")

    reports = evaluate_models(
        dataset_path=args.dataset_path,
        models_dir=args.models_dir,
        output_dir=output_dir,
    )

    for name, text in reports.items():
        print(text)


if __name__ == "__main__":
    main()
