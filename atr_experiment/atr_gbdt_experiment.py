#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATR Rope 选股因子 GBDT 实验脚本

Purpose: 用 LightGBM 回归/分类模型评估 ATR 选股因子重要程度，分析收益分布
Inputs:  atr_factor_return_dataset.csv 或从数据库加载
Outputs: 终端输出 + CSV + PNG 图表

How to Run:
    python atr_experiment/atr_gbdt_experiment.py
    python atr_experiment/atr_gbdt_experiment.py --mode regression
    python atr_experiment/atr_gbdt_experiment.py --mode classification
    python atr_experiment/atr_gbdt_experiment.py --dataset atr_experiment/output/atr_factor_return_dataset.csv

Examples:
    python atr_experiment/atr_gbdt_experiment.py
    python atr_experiment/atr_gbdt_experiment.py --mode regression --dataset atr_experiment/output/atr_factor_return_dataset.csv

Side Effects: 只读数据，写入 output/ 目录
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from atr_experiment.atr_gbdt_utils import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES,
    HORIZONS, PROFIT_THRESHOLD, REG_PARAMS, CLS_PARAMS,
    compute_derived_fields, compute_future_metrics_corrected,
    enrich_with_future_metrics,
)

OUTPUT_DIR = PROJECT_ROOT / "atr_experiment" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_BOOST_ROUND = 1000
EARLY_STOPPING = 50
N_FOLDS = 3
EMBARGO_DAYS = 25


# ==================== 数据准备 ====================

def load_dataset(dataset_path: str = None) -> pd.DataFrame:
    """加载数据集"""
    if dataset_path:
        print(f"加载数据集: {dataset_path}")
        df = pd.read_csv(dataset_path, low_memory=False)
        # 确保衍生字段存在
        for col in ["low_rope_dev_mean_pct", "low_rope_dev_std_pct", "low_rope_dev_today_pct",
                    "low_vwap_dev_mean_pct", "low_vwap_dev_std_pct", "low_vwap_dev_today_pct"]:
            if col not in df.columns:
                df[col] = np.nan
        return df

    # 从数据库加载
    from sqlalchemy import text
    from datasource.database import get_engine

    engine = get_engine()
    sql = text("SELECT * FROM atr_rope_selection WHERE selection_date >= '2023-01-01' ORDER BY selection_date, ts_code")
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    engine.dispose()

    # 计算衍生字段（调用SSOT）
    df = compute_derived_fields(df)

    # 计算未来收益（修正版，调用SSOT）
    print("计算未来收益率 (入场价=open[T+1])...")
    df = enrich_with_future_metrics(df, HORIZONS)

    return df


# ==================== 时间序列切分 ====================

def build_rolling_splits(df, n_folds=N_FOLDS, embargo_days=EMBARGO_DAYS):
    """3折滚动时间序列切分 + embargo"""
    dates = sorted(df["selection_date"].unique())
    fold_size = len(dates) // (n_folds + 1)
    splits = []

    for i in range(n_folds):
        train_end_idx = fold_size * (i + 1)
        test_start_idx = train_end_idx + 1
        test_end_idx = fold_size * (i + 2) if i < n_folds - 1 else len(dates)

        train_dates = set(dates[:train_end_idx])
        embargo_cutoff = pd.Timestamp(dates[train_end_idx]) - pd.Timedelta(days=embargo_days)
        train_dates = {d for d in train_dates if pd.Timestamp(d) <= embargo_cutoff}

        test_dates = set(dates[test_start_idx:test_end_idx])

        train_mask = df["selection_date"].isin(train_dates)
        test_mask = df["selection_date"].isin(test_dates)
        splits.append((train_mask, test_mask))

        print(f"  Fold {i+1}: train={min(train_dates)}~{max(train_dates)} ({train_mask.sum()}), "
              f"test={min(test_dates)}~{max(test_dates)} ({test_mask.sum()})")

    return splits


# ==================== 训练与评估 ====================

def train_and_evaluate(df, target_col, model_type, fold_idx, train_mask, test_mask):
    """训练单个模型并评估"""
    available_features = [f for f in ALL_FEATURES if f in df.columns]

    train_df = df.loc[train_mask].dropna(subset=[target_col])
    test_df = df.loc[test_mask].dropna(subset=[target_col])

    if len(train_df) < 100 or len(test_df) < 50:
        return None

    X_train = train_df[available_features].copy()
    y_train = train_df[target_col]
    X_test = test_df[available_features].copy()
    y_test = test_df[target_col]

    # 分类特征转为 category 类型
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

    # 分类任务：计算 scale_pos_weight
    params = (CLS_PARAMS.copy() if model_type == "classification" else REG_PARAMS.copy())
    if model_type == "classification":
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        if pos_count > 0 and neg_count > 0:
            params["scale_pos_weight"] = neg_count / pos_count

    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
    test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)

    # 训练
    callbacks = [lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(0)]
    model = lgb.train(params, train_data, num_boost_round=NUM_BOOST_ROUND,
                      valid_sets=[test_data], callbacks=callbacks)

    # 预测
    y_pred = model.predict(X_test)

    # 评估
    results = {"fold": fold_idx, "n_train": len(train_df), "n_test": len(test_df),
               "best_iteration": model.best_iteration}

    if model_type == "regression":
        from sklearn.metrics import mean_absolute_error
        results["mae"] = mean_absolute_error(y_test, y_pred)
        ic, _ = stats.spearmanr(y_test, y_pred)
        results["ic"] = ic
    else:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if y_test.nunique() > 1:
            results["auc"] = roc_auc_score(y_test, y_pred)
            results["ap"] = average_precision_score(y_test, y_pred)
        else:
            results["auc"] = np.nan
            results["ap"] = np.nan
        results["pos_rate"] = y_test.mean()
        # 也计算 IC（概率排序 vs 实际收益）
        ret_col = target_col.replace("profit_", "return_")
        if ret_col in test_df.columns:
            ic, _ = stats.spearmanr(test_df[ret_col], y_pred)
            results["ic"] = ic

    # 特征重要性
    gain = model.feature_importance("gain")
    gain_pct = gain / gain.sum() * 100 if gain.sum() > 0 else gain
    importance = dict(zip(available_features, gain_pct))
    results["importance"] = importance

    # 五分组报告
    quintile_report = compute_quintile_report(y_test, y_pred, test_df, model_type)
    results["quintile_report"] = quintile_report

    # 保存预测值用于后续分析
    results["y_test"] = y_test.values
    results["y_pred"] = y_pred

    return results


def compute_quintile_report(y_test, y_pred, test_df, model_type):
    """按预测值分5组，计算每组统计"""
    try:
        buckets = pd.qcut(y_pred, 5, labels=False, duplicates="drop")
    except ValueError:
        return None

    report = []
    for b in sorted(pd.Series(buckets).unique()):
        mask = buckets == b
        y_true_bucket = pd.Series(y_test).iloc[mask]

        row = {"bucket": int(b), "n": mask.sum()}
        if model_type == "regression":
            row["mean_return"] = y_true_bucket.mean()
            row["std_return"] = y_true_bucket.std()
            row["win_rate"] = (y_true_bucket > 0).mean()
        else:
            row["pos_rate"] = y_true_bucket.mean()
            # 也计算实际收益
            ret_col_name = None
            for h in HORIZONS:
                rc = f"return_{h}"
                if rc in test_df.columns:
                    ret_col_name = rc
                    break
            if ret_col_name:
                ret_bucket = test_df.iloc[mask][ret_col_name]
                row["mean_return"] = ret_bucket.mean()
                row["win_rate"] = (ret_bucket > 0).mean()

        report.append(row)

    return pd.DataFrame(report)


# ==================== 收益分布分析 ====================

def analyze_return_distribution(all_results, df, horizons, model_type):
    """分析各期限收益分布"""
    print(f"\n{'='*60}")
    print(f"收益分布分析 ({model_type})")
    print(f"{'='*60}")

    for h in horizons:
        ret_col = f"return_{h}"
        if ret_col not in df.columns:
            continue
        valid = df[ret_col].dropna()
        print(f"\n--- {ret_col} ---")
        print(f"  全样本: mean={valid.mean()*100:.2f}%, std={valid.std()*100:.2f}%, "
              f"median={valid.median()*100:.2f}%, win_rate={(valid>0).mean()*100:.1f}%")
        if model_type == "classification":
            profit_col = f"profit_{h}"
            if profit_col in df.columns:
                pos_rate = df[profit_col].mean()
                print(f"  正类比例(>{PROFIT_THRESHOLD*100}%): {pos_rate*100:.1f}%")


# ==================== 图表 ====================

def plot_feature_importance(importance_df, output_path):
    """绘制特征重要性条形图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if importance_df.empty:
        return

    # 跨折平均 gain%
    avg_importance = importance_df.groupby("feature")["gain_pct"].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(avg_importance) * 0.35)))
    avg_importance.plot.barh(ax=ax, color="steelblue", alpha=0.8)
    ax.set_xlabel("Gain Importance (%)")
    ax.set_title("ATR因子 GBDT 特征重要性 (跨折平均)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_quintile_returns(quintile_df, output_path):
    """绘制五分组收益对比图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if quintile_df.empty:
        return

    models = quintile_df["model"].unique()
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        sub = quintile_df[quintile_df["model"] == model_name]
        if "mean_return" in sub.columns:
            ax.bar(sub["bucket"], sub["mean_return"] * 100, color="steelblue", alpha=0.8)
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        elif "pos_rate" in sub.columns:
            ax.bar(sub["bucket"], sub["pos_rate"] * 100, color="steelblue", alpha=0.8)
        ax.set_title(model_name, fontsize=9)
        ax.set_xlabel("Bucket")
        ax.set_ylabel("Mean Return (%)" if "mean_return" in sub.columns else "Positive Rate (%)")

    plt.suptitle("五分组收益对比")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="ATR Rope 选股因子 GBDT 实验")
    parser.add_argument("--mode", choices=["regression", "classification", "both"], default="both",
                        help="模型类型 (默认: both)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集 CSV 路径 (默认: 从数据库加载)")
    args = parser.parse_args()

    # 加载数据
    df = load_dataset(args.dataset)
    print(f"数据: {len(df)} 条, 日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")

    # 确保衍生字段存在
    for col in ["low_rope_dev_mean_pct", "low_rope_dev_std_pct", "low_rope_dev_today_pct",
                "low_vwap_dev_mean_pct", "low_vwap_dev_std_pct", "low_vwap_dev_today_pct"]:
        if col not in df.columns:
            df[col] = np.nan

    # 构建分类目标
    for h in HORIZONS:
        ret_col = f"return_{h}"
        profit_col = f"profit_{h}"
        if ret_col in df.columns:
            df[profit_col] = (df[ret_col] > PROFIT_THRESHOLD).astype(int)

    # 时间序列切分
    print(f"\n时间序列切分 ({N_FOLDS}折, embargo={EMBARGO_DAYS}天):")
    splits = build_rolling_splits(df)

    # 定义模型任务
    tasks = []
    if args.mode in ("regression", "both"):
        for h in HORIZONS:
            tasks.append({"target": f"return_{h}", "type": "regression", "name": f"reg_return_{h}"})
    if args.mode in ("classification", "both"):
        for h in HORIZONS:
            tasks.append({"target": f"profit_{h}", "type": "classification", "name": f"cls_profit_{h}"})

    # 训练所有模型
    all_results = {}
    all_importance = []
    all_quintile = []

    for task in tasks:
        target = task["target"]
        model_type = task["type"]
        model_name = task["name"]

        print(f"\n{'='*60}")
        print(f"训练: {model_name} (target={target}, type={model_type})")
        print(f"{'='*60}")

        fold_results = []
        for fold_idx, (train_mask, test_mask) in enumerate(splits):
            result = train_and_evaluate(df, target, model_type, fold_idx, train_mask, test_mask)
            if result is None:
                print(f"  Fold {fold_idx+1}: 样本不足，跳过")
                continue

            # 打印评估结果
            if model_type == "regression":
                print(f"  Fold {fold_idx+1}: MAE={result['mae']:.4f}, IC={result['ic']:.4f}, "
                      f"best_iter={result['best_iteration']}")
            else:
                print(f"  Fold {fold_idx+1}: AUC={result.get('auc', np.nan):.4f}, "
                      f"AP={result.get('ap', np.nan):.4f}, pos_rate={result['pos_rate']:.3f}, "
                      f"IC={result.get('ic', np.nan):.4f}, best_iter={result['best_iteration']}")

            # 收集特征重要性
            for feat, gain in result["importance"].items():
                all_importance.append({
                    "model": model_name, "fold": fold_idx,
                    "feature": feat, "gain_pct": gain,
                })

            # 收集五分组报告
            if result["quintile_report"] is not None:
                qr = result["quintile_report"].copy()
                qr["model"] = model_name
                qr["fold"] = fold_idx
                all_quintile.append(qr)

            fold_results.append(result)

        # 跨折汇总
        if fold_results:
            if model_type == "regression":
                ics = [r["ic"] for r in fold_results if "ic" in r]
                maes = [r["mae"] for r in fold_results]
                ic_mean = np.mean(ics) if ics else 0
                ic_std = np.std(ics) if len(ics) > 1 else 0
                icir = ic_mean / ic_std if ic_std > 0 else 0
                print(f"\n  汇总: MAE={np.mean(maes):.4f}, IC_mean={ic_mean:.4f}, "
                      f"IC_std={ic_std:.4f}, ICIR={icir:.3f}")
            else:
                aucs = [r["auc"] for r in fold_results if not np.isnan(r.get("auc", np.nan))]
                aps = [r["ap"] for r in fold_results if not np.isnan(r.get("ap", np.nan))]
                ics = [r["ic"] for r in fold_results if "ic" in r and not np.isnan(r.get("ic", np.nan))]
                ic_mean = np.mean(ics) if ics else 0
                ic_std = np.std(ics) if len(ics) > 1 else 0
                icir = ic_mean / ic_std if ic_std > 0 else 0
                print(f"\n  汇总: AUC={np.mean(aucs):.4f}, AP={np.mean(aps):.4f}, "
                      f"IC_mean={ic_mean:.4f}, ICIR={icir:.3f}")

        all_results[model_name] = fold_results

    # 收益分布分析
    if args.mode in ("regression", "both"):
        analyze_return_distribution(all_results, df, HORIZONS, "regression")
    if args.mode in ("classification", "both"):
        analyze_return_distribution(all_results, df, HORIZONS, "classification")

    # 特征重要性排名
    importance_df = pd.DataFrame(all_importance)
    if not importance_df.empty:
        print(f"\n{'='*60}")
        print("特征重要性排名 (跨模型跨折平均)")
        print(f"{'='*60}")
        avg_imp = importance_df.groupby("feature")["gain_pct"].mean().sort_values(ascending=False)
        for feat, gain in avg_imp.items():
            print(f"  {feat:30s}  {gain:.2f}%")

        importance_df.to_csv(OUTPUT_DIR / "atr_gbdt_feature_importance.csv", index=False)
        plot_feature_importance(importance_df, OUTPUT_DIR / "atr_gbdt_feature_importance.png")

    # 五分组报告
    quintile_df = pd.concat(all_quintile, ignore_index=True) if all_quintile else pd.DataFrame()
    if not quintile_df.empty:
        print(f"\n{'='*60}")
        print("五分组收益报告 (最后一折)")
        print(f"{'='*60}")
        last_fold = quintile_df["fold"].max()
        for model_name in quintile_df["model"].unique():
            sub = quintile_df[(quintile_df["model"] == model_name) & (quintile_df["fold"] == last_fold)]
            if sub.empty:
                continue
            print(f"\n--- {model_name} ---")
            print(sub.to_string(index=False))

        quintile_df.to_csv(OUTPUT_DIR / "atr_gbdt_quintile_report.csv", index=False)
        plot_quintile_returns(quintile_df[quintile_df["fold"] == last_fold],
                              OUTPUT_DIR / "atr_gbdt_quintile_returns.png")

    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
