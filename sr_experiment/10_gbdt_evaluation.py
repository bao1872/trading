# -*- coding: utf-8 -*-
"""
Purpose: GBDT v2 模型评估：AUC/AP + 分层收益表 + quality_score 分层 + 交易指标 + 特征重要性
Inputs:  results/gbdt/evaluation/test_predictions_{exp}.parquet + feature_importance_{exp}.csv
Outputs: 控制台报告 + results/gbdt/evaluation/stratified_returns_{exp}.csv
How to Run:
    python sr_experiment/10_gbdt_evaluation.py
    python sr_experiment/10_gbdt_evaluation.py --experiment A1_trend_opp
Examples:
    python sr_experiment/10_gbdt_evaluation.py
    python sr_experiment/10_gbdt_evaluation.py --scene A1_trend
Side Effects: 写 CSV 到 sr_experiment/results/gbdt/evaluation/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sr_experiment.gbdt_config import EVAL_DIR, EXPERIMENT_SPECS, SCENE_PAIRS
from sr_experiment.gbdt_feature_columns import FACTOR_CATEGORIES

BASELINE_FWD_RET_20 = 0.0532
BASELINE_WIN_RATE = 0.4959

QUANTILE_BINS = [0, 0.05, 0.10, 0.20, 0.50, 1.0]
QUANTILE_LABELS = ["top5%", "5-10%", "10-20%", "20-50%", "bottom50%"]


def _stratify(df: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    if score_col not in df.columns or df[score_col].isna().all():
        return pd.DataFrame()

    df = df.dropna(subset=[score_col]).copy()
    if len(df) < 10:
        return pd.DataFrame()

    df["score_group"] = pd.qcut(df[score_col], q=QUANTILE_BINS, labels=QUANTILE_LABELS, duplicates="drop")

    rows = []
    for grp_label in QUANTILE_LABELS:
        sub = df[df["score_group"] == grp_label]
        if sub.empty:
            continue
        row = {"score_group": grp_label, "count": len(sub)}
        for col in ["fwd_ret_20", "fwd_max_ret_20", "fwd_mdd_20", "fwd_reward_risk_20"]:
            if col in sub.columns:
                vals = sub[col].dropna()
                row[col] = vals.mean() if len(vals) > 0 else np.nan
        if "fwd_ret_20" in sub.columns:
            vals = sub["fwd_ret_20"].dropna()
            row["win_rate_20"] = (vals > 0).mean() if len(vals) > 0 else np.nan
        if "tp_hit_bar" in sub.columns:
            tp_mask = sub["tp_hit_bar"].notna()
            row["tp_rate"] = tp_mask.mean()
            if tp_mask.any():
                row["avg_tp_bars"] = sub.loc[tp_mask, "tp_hit_bar"].mean()
        if "sl_hit_bar" in sub.columns:
            sl_mask = sub["sl_hit_bar"].notna()
            row["sl_rate"] = sl_mask.mean()
            if sl_mask.any():
                row["avg_sl_bars"] = sub.loc[sl_mask, "sl_hit_bar"].mean()
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_single(exp_name: str, spec: dict):
    pred_path = Path(EVAL_DIR) / f"test_predictions_{exp_name}.parquet"
    if not pred_path.exists():
        print(f"预测文件不存在: {pred_path}")
        return

    df = pd.read_parquet(pred_path)
    label_col = spec["label_name"]

    print(f"\n{'='*60}")
    print(f"实验 {exp_name}: {spec['description']} ({spec['model_type']})")
    print(f"测试集样本: {len(df)}")

    if "pred_score" not in df.columns or label_col not in df.columns:
        print("缺少 pred_score 或标签列")
        return

    from sklearn.metrics import average_precision_score, roc_auc_score
    if df[label_col].nunique() > 1:
        auc = roc_auc_score(df[label_col], df["pred_score"])
        ap = average_precision_score(df[label_col], df["pred_score"])
        print(f"AUC: {auc:.4f}, AP: {ap:.4f}")

    strat_df = _stratify(df, "pred_score", label_col)
    if strat_df.empty:
        return

    strat_path = Path(EVAL_DIR) / f"stratified_returns_{exp_name}.csv"
    strat_df.to_csv(strat_path, index=False, encoding="utf-8-sig")

    print(f"\n分层收益表:")
    display_cols = [c for c in ["score_group", "count", "fwd_ret_20", "fwd_max_ret_20",
                                 "fwd_mdd_20", "fwd_reward_risk_20", "win_rate_20",
                                 "tp_rate", "sl_rate", "avg_tp_bars", "avg_sl_bars"]
                    if c in strat_df.columns]
    print(tabulate(strat_df[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    imp_csv_path = Path(EVAL_DIR) / f"feature_importance_{exp_name}.csv"
    if imp_csv_path.exists():
        imp_df = pd.read_csv(imp_csv_path)
        print(f"\nTop 5 特征:")
        print(tabulate(imp_df.head(5)[["rank", "feature", "category", "gain_pct"]],
                       headers="keys", tablefmt="grid", showindex=False, floatfmt=".2f"))


def evaluate_scene(scene_name: str):
    """对场景的 opp + risk 模型做 quality_score 分层评估。"""
    if scene_name not in SCENE_PAIRS:
        print(f"场景 {scene_name} 不存在")
        return

    opp_name, risk_name = SCENE_PAIRS[scene_name]
    opp_path = Path(EVAL_DIR) / f"test_predictions_{opp_name}.parquet"
    risk_path = Path(EVAL_DIR) / f"test_predictions_{risk_name}.parquet"

    if not opp_path.exists() or not risk_path.exists():
        print(f"场景 {scene_name} 的预测文件不完整")
        return

    opp_df = pd.read_parquet(opp_path, columns=["ts_code", "bar_time", "pred_score"])
    risk_df = pd.read_parquet(risk_path, columns=["ts_code", "bar_time", "pred_score"])

    opp_df = opp_df.rename(columns={"pred_score": "opp_score"})
    risk_df = risk_df.rename(columns={"pred_score": "risk_score"})

    merged = opp_df.merge(risk_df, on=["ts_code", "bar_time"], how="inner")
    merged["quality_score"] = merged["opp_score"] - merged["risk_score"]

    full_opp = pd.read_parquet(opp_path)
    full_opp = full_opp.merge(merged[["ts_code", "bar_time", "quality_score"]], on=["ts_code", "bar_time"], how="left")

    test_mask = full_opp["split"] == "test" if "split" in full_opp.columns else pd.Series(True, index=full_opp.index)
    test_df = full_opp[test_mask].copy()

    print(f"\n{'='*60}")
    print(f"场景 {scene_name}: quality_score = opp_score - risk_score")
    print(f"测试集样本: {len(test_df)}")

    strat_df = _stratify(test_df, "quality_score", EXPERIMENT_SPECS[opp_name]["label_name"])
    if strat_df.empty:
        return

    strat_path = Path(EVAL_DIR) / f"stratified_quality_{scene_name}.csv"
    strat_df.to_csv(strat_path, index=False, encoding="utf-8-sig")

    print(f"\nquality_score 分层收益表:")
    display_cols = [c for c in ["score_group", "count", "fwd_ret_20", "fwd_max_ret_20",
                                 "fwd_mdd_20", "fwd_reward_risk_20", "win_rate_20",
                                 "tp_rate", "sl_rate"]
                    if c in strat_df.columns]
    print(tabulate(strat_df[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    top10 = strat_df[strat_df["score_group"].isin(["top5%", "5-10%"])]
    if not top10.empty and "fwd_ret_20" in top10.columns:
        top10_ret = top10["fwd_ret_20"].mean()
        print(f"\nvs 基准(强支撑簇+缩量): top10% fwd_ret_20={top10_ret:.4f}, 基准={BASELINE_FWD_RET_20:.4f}")
        if top10_ret > BASELINE_FWD_RET_20:
            print("✅ quality_score top10% 优于基准")
        else:
            print("❌ quality_score top10% 未超越基准")


def main():
    parser = argparse.ArgumentParser(description="GBDT v2 模型评估")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=list(EXPERIMENT_SPECS.keys()))
    parser.add_argument("--scene", type=str, default=None,
                        choices=list(SCENE_PAIRS.keys()),
                        help="评估场景的 quality_score 分层")
    args = parser.parse_args()

    if args.scene:
        evaluate_scene(args.scene)
        return

    specs = EXPERIMENT_SPECS
    if args.experiment:
        specs = {args.experiment: EXPERIMENT_SPECS[args.experiment]}

    for exp_name, spec in specs.items():
        evaluate_single(exp_name, spec)

    if not args.experiment:
        print("\n\n========== 场景 quality_score 分层 ==========")
        for scene_name in SCENE_PAIRS:
            evaluate_scene(scene_name)


if __name__ == "__main__":
    main()
