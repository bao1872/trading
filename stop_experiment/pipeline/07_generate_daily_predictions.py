#!/usr/bin/env python3
"""
预测账本生成器 — 唯一负责生成 output/predictions/{T}.parquet 的正式入口。

Purpose:
    从 candidate_with_scores.parquet（已含 final 模型预测）提取指定交易日的候选样本，
    计算 score 列，写入预测账本。幂等：已存在则跳过。

用法：
    python -m stop_experiment.pipeline.07_generate_daily_predictions --date 2026-05-08
    python -m stop_experiment.pipeline.07_generate_daily_predictions --date 2026-05-08 --force

参数：
    --date  目标交易日 YYYY-MM-DD
    --force 强制覆盖已存在的预测账本

输入：
    stop_experiment/output/candidate_with_scores.parquet (由 02_train_gbdt_models 生成)

输出：
    stop_experiment/output/predictions/YYYY-MM-DD.parquet
    列: ts_code, signal_id, obs_date, obs_day, pred_sell_reg, pred_sell_cls,
         pred_buy_reg, pred_buy_cls, score

副作用：只读 parquet，写 parquet（幂等）
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from stop_experiment.pipeline.stop_config import OUTPUT_DIR, PREDICTIONS_DIR
from stop_experiment.backtest.simple_backtest import score_stocks


CANDIDATE_PATH = os.path.join(OUTPUT_DIR, "candidate_with_scores.parquet")
REQUIRED_COLS = ["ts_code", "signal_id", "obs_date", "obs_day",
                 "pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls"]


def check_candidate_coverage(target_date):
    """检查 candidate_with_scores.parquet 是否覆盖目标日期。返回 (ok, message, df_day_or_none)。"""
    if not os.path.exists(CANDIDATE_PATH):
        return False, (
            f"candidate_with_scores.parquet 不存在: {CANDIDATE_PATH}\n"
            "请先运行上游流水线: 01_build_dataset.py → 02_train_gbdt_models.py"
        ), None

    df_all = pd.read_parquet(CANDIDATE_PATH, columns=["obs_date"])
    df_all["obs_date"] = pd.to_datetime(df_all["obs_date"])
    min_date = df_all["obs_date"].min()
    max_date = df_all["obs_date"].max()
    n_target = (df_all["obs_date"] == target_date).sum()

    if n_target == 0:
        return False, (
            f"candidate_with_scores.parquet 未覆盖目标交易日 {target_date.strftime('%Y-%m-%d')}\n"
            f"  数据范围: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}\n"
            f"  目标行数: 0\n"
            f"请先运行上游流水线更新候选数据: 01_build_dataset.py → 02_train_gbdt_models.py"
        ), None

    return True, (
        f"candidate_with_scores.parquet 覆盖 {target_date.strftime('%Y-%m-%d')}"
        f" ({n_target} 行，范围 {min_date.strftime('%Y-%m-%d')}~{max_date.strftime('%Y-%m-%d')})"
    ), None


def generate_predictions(target_date, force=False):
    """
    生成 predictions/{T}.parquet。

    返回 (success, message, output_path)。
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    date_str = target_date.strftime("%Y-%m-%d")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_path = os.path.join(PREDICTIONS_DIR, f"{date_str}.parquet")

    if os.path.exists(out_path) and not force:
        existing = pd.read_parquet(out_path)
        return True, (
            f"预测账本已存在: {out_path} ({len(existing)} 行)，跳过生成。"
            f"使用 --force 强制覆盖"
        ), out_path

    ok, msg, _ = check_candidate_coverage(target_date)
    if not ok:
        return False, msg, out_path

    df_all = pd.read_parquet(CANDIDATE_PATH)
    df_all["obs_date"] = pd.to_datetime(df_all["obs_date"])
    df_day = df_all[df_all["obs_date"] == target_date].copy()

    available_cols = [c for c in REQUIRED_COLS if c in df_day.columns]
    if len(available_cols) < len(REQUIRED_COLS):
        missing = set(REQUIRED_COLS) - set(available_cols)
        return False, f"candidate_with_scores.parquet 缺少必要列: {missing}", out_path

    df_day = df_day[available_cols].copy()

    if "score" not in df_day.columns:
        df_day = score_stocks(df_day, strategy="sell_score")

    df_day.to_parquet(out_path, index=False)

    return True, (
        f"预测账本已生成: {out_path} ({len(df_day)} 行, "
        f"obs_day={sorted(df_day['obs_day'].unique())})"
    ), out_path


def main():
    parser = argparse.ArgumentParser(description="生成当日预测账本 predictions/{T}.parquet")
    parser.add_argument("--date", required=True, help="目标交易日 YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的预测账本")
    args = parser.parse_args()

    target_date = pd.to_datetime(args.date)
    success, msg, out_path = generate_predictions(target_date, force=args.force)

    print(msg)
    if success:
        print(f"  ✅ 完成: {out_path}")
    else:
        print(f"  ❌ 失败")
        sys.exit(1)


if __name__ == "__main__":
    main()