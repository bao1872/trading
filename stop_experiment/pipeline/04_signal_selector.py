#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号精选：基于双模型评分的综合排序

Purpose:
    利用4个模型评分对观察期内的每日样本进行精选排序：
    sell_reg + sell_cls (mfe>7%) + buy_reg + buy_cls (mae<-5%)。

Pipeline Position:
    训练流水线第四步（离线，一次性）。
    上游: 02_train_gbdt_models.py
    下游: generate_full_predictions.py (在 backtest/)

Inputs:
    - stop_experiment/output/candidate_with_scores.parquet

Outputs:
    - stop_experiment/output/selected_signals.parquet

How to Run:
    python stop_experiment/pipeline/04_signal_selector.py
    python stop_experiment/pipeline/04_signal_selector.py --top-k 50

Side Effects:
    - 只读parquet，输出精选信号文件

Note:
    BUY_CLS_THRESHOLD=-0.07 是训练侧标签阈值（mae_20<-7%→buy_cls=1），
    不等于生产 Exit 阈值 (buy_cls_exit_threshold=0.70)。
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import OUTPUT_DIR, BUY_CLS_THRESHOLD, PRODUCTION_PARAMS


def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算综合评分：
    - sell_score: 卖点模型认为上涨空间大 → 不卖（sell_reg 预测的 mfe_20 越大越好）
    - buy_score: 买点模型认为下跌风险小 → 可买（buy_reg 预测的 mae_20 越大越好）
    - veto: buy_cls 预测为正类（还有下跌风险）→ 剔除
    """
    df = df.copy()

    # sell_reg 预测的 mfe_20（越大越好，上涨空间大）
    if "pred_sell_reg" in df.columns:
        df["sell_score"] = df["pred_sell_reg"]

    # buy_reg 预测的 mae_20（越大越好，下跌风险小；mae_20 越接近0越好）
    if "pred_buy_reg" in df.columns:
        df["buy_score"] = -df["pred_buy_reg"]  # 翻转：mae_20 负值越小风险越大，取负后越大越好

    # sell_cls: P(mfe_20 > 7%) → 越大越说明还有上涨空间
    if "pred_sell_cls" in df.columns:
        df["sell_confidence"] = df["pred_sell_cls"]

    # buy_cls: P(mae_20 < -5%) → 越大说明还有下跌风险 → veto
    if "pred_buy_cls" in df.columns:
        df["downside_risk"] = df["pred_buy_cls"]

    # 综合评分：sell_score + buy_score
    df["composite_score"] = (
        df.get("sell_score", pd.Series(0, index=df.index)) * 0.5
        + df.get("buy_score", pd.Series(0, index=df.index)) * 0.5
    )

    return df


def assign_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    档位分配：
    A: 高收益 + 低风险（sell_score高 + downside_risk低）
    B: 高收益 + 高风险
    C: 低收益 + 低风险
    D: 低收益 + 高风险
    """
    df = df.copy()

    sell_median = df["sell_score"].median() if "sell_score" in df.columns else 0
    risk_median = df["downside_risk"].median() if "downside_risk" in df.columns else 0.5

    high_return = df.get("sell_score", pd.Series(0, index=df.index)) >= sell_median
    low_risk = df.get("downside_risk", pd.Series(1, index=df.index)) <= risk_median

    df["grade"] = "D"
    df.loc[high_return & low_risk, "grade"] = "A"
    df.loc[high_return & ~low_risk, "grade"] = "B"
    df.loc[~high_return & low_risk, "grade"] = "C"

    return df


def select_signals(df: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
    """
    精选信号：
    1. 排除 veto（downside_risk > 0.6 表示还有较大下跌风险）
    2. 按 composite_score 排序
    3. 取 top_k
    """
    # 过滤 veto
    if "downside_risk" in df.columns:
        veto_threshold = 0.6
        df = df[df["downside_risk"] <= veto_threshold].copy()
        print(f"  veto过滤: downside_risk<={veto_threshold}, 剩余 {len(df)} 行")

    # 按日期分组，每日取 top_k
    if "obs_date" in df.columns:
        selected = (
            df.sort_values("composite_score", ascending=False)
            .groupby("obs_date")
            .head(top_k)
        )
    else:
        selected = df.nlargest(top_k, "composite_score")

    return selected


def main(args):
    print("=" * 60)
    print("信号精选")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/3] 加载带评分数据集...")
    input_path = os.path.join(OUTPUT_DIR, "candidate_with_scores.parquet")
    df = pd.read_parquet(input_path)
    print(f"  总行数: {len(df)}")

    # 只保留可交易的
    if "can_buy" in df.columns:
        df = df[df["can_buy"] == 1].copy()
    print(f"  可交易: {len(df)}")

    # 候选池过滤: 只保留 obs_day=1（生产口径）
    candidate_obs_days = PRODUCTION_PARAMS.get("candidate_obs_days", [1])
    df = df[df["obs_day"].isin(candidate_obs_days)].copy()
    print(f"  obs_day in {candidate_obs_days}: {len(df)}")

    df = df.sort_values(["signal_id", "obs_date", "obs_day"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["signal_id", "obs_date"], keep="first")
    print(f"  去重后: {len(df)}")

    # 2. 计算综合评分
    print("\n[2/3] 计算综合评分...")
    df = compute_composite_score(df)
    df = assign_grade(df)

    # 打印档位分布
    grade_dist = df["grade"].value_counts().sort_index()
    print(f"  档位分布:")
    for grade, cnt in grade_dist.items():
        print(f"    {grade}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # 3. 精选
    print(f"\n[3/3] 精选 top_{args.top_k}...")
    selected = select_signals(df, top_k=args.top_k)

    # 保存
    selected_path = os.path.join(OUTPUT_DIR, "selected_signals.parquet")
    selected.to_parquet(selected_path, index=False)
    print(f"  精选后: {len(selected)} 行")
    print(f"  保存: {selected_path}")

    # 摘要
    if "obs_date" in selected.columns:
        print(f"\n  日期范围: {selected['obs_date'].min()} ~ {selected['obs_date'].max()}")
        print(f"  日期数: {selected['obs_date'].nunique()}")
        print(f"  每日信号数: 均值={len(selected)/max(selected['obs_date'].nunique(),1):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="信号精选")
    parser.add_argument("--top-k", type=int, default=100, help="每日选取信号数")
    args = parser.parse_args()
    main(args)
