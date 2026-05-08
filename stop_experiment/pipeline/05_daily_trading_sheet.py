#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日决策：生成交易清单 (DEPRECATED)

Purpose:
    基于精选信号生成每日交易决策表。
    ⚠️ 已废弃：被 08_daily_inference_report.py 替代，新生产环境请使用日报系统。

Pipeline Position:
    生产流水线（旧版，已废弃）。
    上游: 04_signal_selector.py
    下游: —

Inputs:
    - stop_experiment/output/selected_signals.parquet

Outputs:
    - stop_experiment/output/daily_trading_sheet.csv

How to Run:
    python stop_experiment/pipeline/05_daily_trading_sheet.py
    python stop_experiment/pipeline/05_daily_trading_sheet.py --date 2026-05-06

Side Effects:
    - 只读parquet，输出交易清单CSV
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import OUTPUT_DIR, MODEL_EXIT_PARAMS


def generate_trading_sheet(selected: pd.DataFrame, target_date: str | None = None) -> pd.DataFrame:
    """
    生成每日交易清单。

    Phase 2 升级：模型驱动退出 + obs_day=1~3 候选池
    买入: obs_day in [1,2,3] 的候选信号，按 composite_score 排序
    卖出: pred_buy_cls > 0.7 (model_risk) / stop_loss=-7% / max_hold=20

    输出列增加 sell_reason，区别于旧版 action 字段
    """
    buy_cls_threshold = MODEL_EXIT_PARAMS.get("buy_cls_exit_threshold", 0.7)
    stop_loss = MODEL_EXIT_PARAMS.get("stop_loss", -0.07)
    max_hold = MODEL_EXIT_PARAMS.get("max_hold_days", 20)

    if target_date:
        target_date = pd.Timestamp(target_date)
        sheet = selected[selected["obs_date"] == target_date].copy()
    else:
        latest_date = selected["obs_date"].max()
        sheet = selected[selected["obs_date"] == latest_date].copy()
        target_date = latest_date

    if sheet.empty:
        print(f"  {target_date} 无精选信号")
        return sheet

    sheet["action"] = "hold"
    sheet["sell_reason"] = ""

    # 买入建议：按 composite_score 排序取前 N（兼容旧 grade 逻辑）
    if "composite_score" in sheet.columns:
        buy_threshold = sheet["composite_score"].quantile(0.7) if len(sheet) > 1 else -999
        sheet.loc[sheet["composite_score"] >= buy_threshold, "action"] = "buy"

    # 卖出建议 (模型驱动)
    if "pred_buy_cls" in sheet.columns:
        model_risk_mask = sheet["pred_buy_cls"] > buy_cls_threshold
        sheet.loc[model_risk_mask, "action"] = "sell"
        sheet.loc[model_risk_mask, "sell_reason"] = "model_risk"

    # 止损（需要在回测中跟踪持仓收益，此处仅标注模型风险+提示止损阈值）
    if "pred_buy_cls" in sheet.columns:
        high_concern = sheet["pred_buy_cls"] > (buy_cls_threshold - 0.1)
        sheet.loc[high_concern & (sheet["sell_reason"] == ""), "sell_reason"] = "watch"

    return sheet


def main(args):
    print("=" * 60)
    print("每日交易清单")
    print("=" * 60)

    # 1. 加载精选信号
    print("\n[1/2] 加载精选信号...")
    input_path = os.path.join(OUTPUT_DIR, "selected_signals.parquet")
    selected = pd.read_parquet(input_path)
    print(f"  总行数: {len(selected)}")

    # 2. 生成交易清单
    print("\n[2/2] 生成交易清单...")
    sheet = generate_trading_sheet(selected, target_date=args.date)

    if sheet.empty:
        print("  无交易信号")
        return

    # 输出列
    output_cols = [
        "ts_code", "stock_name", "obs_date", "obs_close",
        "obs_day", "composite_score", "sell_score", "buy_score",
        "sell_confidence", "pred_buy_cls", "downside_risk", "grade",
        "action", "sell_reason",
    ]
    output_cols = [c for c in output_cols if c in sheet.columns]

    # 保存
    output_path = os.path.join(OUTPUT_DIR, "daily_trading_sheet.csv")
    sheet[output_cols].to_csv(output_path, index=False)
    print(f"  保存: {output_path}")

    # 摘要
    print(f"\n  日期: {sheet['obs_date'].iloc[0] if len(sheet) > 0 else 'N/A'}")
    print(f"  总信号数: {len(sheet)}")
    if "action" in sheet.columns:
        action_dist = sheet["action"].value_counts()
        print(f"  操作分布:")
        for action, cnt in action_dist.items():
            print(f"    {action}: {cnt}")
    if "sell_reason" in sheet.columns:
        reason_dist = sheet[sheet["sell_reason"] != ""]["sell_reason"].value_counts()
        if len(reason_dist) > 0:
            print(f"  卖出原因分布:")
            for reason, cnt in reason_dist.items():
                print(f"    {reason}: {cnt}")

    if "grade" in sheet.columns:
        grade_dist = sheet["grade"].value_counts().sort_index()
        print(f"  档位分布:")
        for grade, cnt in grade_dist.items():
            print(f"    {grade}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="每日交易清单")
    parser.add_argument("--date", type=str, default=None, help="目标日期 (YYYY-MM-DD)")
    args = parser.parse_args()
    main(args)
