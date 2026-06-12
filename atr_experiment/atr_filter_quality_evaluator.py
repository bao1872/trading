#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATR GBDT 过滤器质量评估引擎

Purpose: 评估 ATR GBDT 双模型（回归/分类）的 TopK 过滤质量
         沿用 gbdt_backtest_evaluator.py 的评估逻辑，输出 CSV 供前端展示
Inputs:  atr_rope_features 表 + stock_k_data 表
Outputs: atr_experiment/output/filter_quality/ 目录下 2 个 CSV

How to Run:
    python atr_experiment/atr_filter_quality_evaluator.py --start-date 2026-01-05 --end-date 2026-05-25
    python atr_experiment/atr_filter_quality_evaluator.py --start-date 2026-01-05 --end-date 2026-05-25 --force

Examples:
    python atr_experiment/atr_filter_quality_evaluator.py --start-date 2026-01-05 --end-date 2026-05-25
    python atr_experiment/atr_filter_quality_evaluator.py --start-date 2026-03-01 --end-date 2026-03-31 --force

Side Effects:
    - 只读 DB
    - 写入 atr_experiment/output/filter_quality/ 目录下的 CSV 文件
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from atr_experiment.atr_gbdt_utils import (
    HORIZONS, compute_future_metrics_corrected,
)
from stop_experiment.backtest.simple_backtest import load_daily_prices, build_price_pivot
from stop_experiment.eval.filter_quality_evaluator import _lookup_values

# ==================== 配置 ====================

OUTPUT_DIR = PROJECT_ROOT / "atr_experiment" / "output" / "filter_quality"

TOPK_LIST = [5, 10]
HORIZONS_EVAL = [3, 5, 10, 20]

# 评估场景: (model_key, scenario_name, sort_col_or_none)
def _build_scenarios():
    scenarios = [("baseline", "all", None)]
    for k in TOPK_LIST:
        scenarios.append(("regression", f"top{k}", "pred_return"))
        scenarios.append(("classification", f"top{k}", "pred_prob"))
        scenarios.append(("combined", f"top{k}", "combined_score"))
        scenarios.append(("overlap", f"top{k}", f"overlap_top{k}"))
    return scenarios

SCENARIOS = _build_scenarios()

OUTPUT_FILES = ["atr_daily_detail.csv", "atr_summary.csv"]


# ==================== 数据加载 ====================

def load_features_data(start_date: str, end_date: str) -> pd.DataFrame:
    """从 atr_rope_features 表加载选股预测数据"""
    from sqlalchemy import text
    from datasource.database import get_engine

    engine = get_engine()
    sql = text("""
        SELECT feature_date, ts_code, stock_name, source,
               pred_return, pred_prob, combined_score,
               reg_rank, cls_rank, combined_rank,
               in_reg_top5, in_cls_top5, in_overlap_top5,
               in_reg_top10, in_cls_top10, in_overlap_top10
        FROM atr_rope_features
        WHERE source = 'selection'
          AND feature_date >= :start
          AND feature_date <= :end
        ORDER BY feature_date, ts_code
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"start": start_date, "end": end_date})
    engine.dispose()
    return df


# ==================== 评估逻辑 ====================

def _group_stats(group_df: pd.DataFrame, horizon: int) -> dict | None:
    """计算单组在指定 horizon 下的统计指标"""
    ret_col = f"return_{horizon}"
    mfe_col = f"mfe_{horizon}"
    mae_col = f"mae_{horizon}"

    valid = group_df.dropna(subset=[ret_col])
    if len(valid) == 0:
        return None

    ret = valid[ret_col]
    mfe = valid[mfe_col] if mfe_col in valid.columns else pd.Series(dtype=float)
    mae = valid[mae_col] if mae_col in valid.columns else pd.Series(dtype=float)

    result = {
        "avg_return": ret.mean(),
        "win_rate": (ret > 0).mean(),
        "n": len(valid),
    }
    if len(mfe.dropna()) > 0:
        result["avg_mfe"] = mfe.mean()
    if len(mae.dropna()) > 0:
        result["avg_mae"] = mae.mean()
    return result


def evaluate_daily(features_df: pd.DataFrame) -> pd.DataFrame:
    """逐日逐场景评估，沿用 gbdt_backtest_evaluator.py 的 evaluate_daily 逻辑"""
    dates = sorted(features_df["feature_date"].unique())
    records = []

    for date in tqdm(dates, desc="Evaluating"):
        day_df = features_df[features_df["feature_date"] == date].copy()
        if day_df.empty:
            continue

        for model_key, scenario_name, sort_col in SCENARIOS:
            # 选择子集
            if sort_col is None:
                # baseline: 全候选池
                subset = day_df
            elif sort_col.startswith("overlap_top"):
                # 交集: reg_top ∩ cls_top
                k = int(sort_col.replace("overlap_top", ""))
                overlap_col = f"in_overlap_top{k}"
                if overlap_col in day_df.columns:
                    subset = day_df[day_df[overlap_col] == True]  # noqa: E712
                else:
                    # 回退：手动计算交集
                    reg_top = set(day_df.nlargest(k, "pred_return")["ts_code"].values)
                    cls_top = set(day_df.nlargest(k, "pred_prob")["ts_code"].values)
                    overlap_codes = reg_top & cls_top
                    subset = day_df[day_df["ts_code"].isin(overlap_codes)]
            else:
                # 单模型 TopK
                k = int(scenario_name.replace("top", ""))
                subset = day_df.nlargest(k, sort_col)

            for h in HORIZONS_EVAL:
                stats = _group_stats(subset, h)
                if stats:
                    records.append({
                        "date": date,
                        "model": model_key,
                        "scenario": scenario_name,
                        "horizon": h,
                        **stats,
                    })

    return pd.DataFrame(records)


def build_summary(daily_detail: pd.DataFrame) -> pd.DataFrame:
    """按模型×场景×期限聚合汇总"""
    if daily_detail.empty:
        return pd.DataFrame()

    summary = daily_detail.groupby(["model", "scenario", "horizon"]).agg(
        avg_return=("avg_return", "mean"),
        win_rate=("win_rate", "mean"),
        avg_mfe=("avg_mfe", "mean"),
        avg_mae=("avg_mae", "mean"),
        n_days=("date", "count"),
        n_avg=("n", "mean"),
    ).reset_index()

    return summary


# ==================== 主流程 ====================

def run_evaluation(start_date: str, end_date: str, force: bool = False):
    """运行 ATR GBDT 过滤器质量评估主流程"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not force and all(
        os.path.exists(os.path.join(OUTPUT_DIR, f)) for f in OUTPUT_FILES
    ):
        print("输出文件已存在，使用 --force 重新计算")
        return

    # 1. 加载预测数据
    print("[1/4] 加载 ATR 预测数据...")
    features_df = load_features_data(start_date, end_date)
    if features_df.empty:
        print("无预测数据，退出")
        return
    print(f"  预测数据: {len(features_df)} 条, "
          f"日期: {features_df['feature_date'].min()} ~ {features_df['feature_date'].max()}")

    # 2. 加载价格数据并计算未来收益
    print("[2/4] 加载价格数据并计算未来收益 (入场价=open[T+1])...")
    price_start = start_date
    price_end = (pd.Timestamp(end_date) + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    daily_prices = load_daily_prices(price_start, price_end)
    price_pivot, trading_days_all, _ = build_price_pivot(daily_prices)
    print(f"  交易日: {len(trading_days_all)}, "
          f"价格覆盖: {trading_days_all[0].strftime('%Y-%m-%d')} ~ {trading_days_all[-1].strftime('%Y-%m-%d')}")

    # 计算未来收益/MFE/MAE（入场价=open[T+1]，无前视偏差）
    future_metrics = compute_future_metrics_corrected(price_pivot, HORIZONS_EVAL)

    # 挂载到 features_df
    features_df["raw_code"] = features_df["ts_code"].str[:6]
    features_df["lookup_date"] = pd.to_datetime(features_df["feature_date"]).dt.normalize()

    for N in HORIZONS_EVAL:
        for metric_name in ["return", "mfe", "mae"]:
            col = f"{metric_name}_{N}"
            features_df[col] = _lookup_values(
                future_metrics[N][metric_name],
                features_df["lookup_date"].values,
                features_df["raw_code"].values,
            )
        print(f"  return_{N}: 非空 {features_df[f'return_{N}'].notna().sum()}/{len(features_df)}")

    features_df.drop(columns=["lookup_date"], inplace=True)

    # 3. 逐日评估
    print("[3/4] 评估过滤器质量...")
    daily_detail = evaluate_daily(features_df)

    # 4. 汇总并输出
    print("[4/4] 汇总并输出...")
    summary = build_summary(daily_detail)

    for fname, df in zip(OUTPUT_FILES, [daily_detail, summary]):
        path = os.path.join(OUTPUT_DIR, fname)
        df.to_csv(path, index=False)
        print(f"  {fname}: {len(df)} 行")

    if not summary.empty:
        print("\n汇总表:")
        print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="ATR GBDT 过滤器质量评估引擎")
    parser.add_argument("--start-date", required=True, help="评估开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="评估结束日期 (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="强制重新计算")
    args = parser.parse_args()

    run_evaluation(args.start_date, args.end_date, force=args.force)


if __name__ == "__main__":
    main()
