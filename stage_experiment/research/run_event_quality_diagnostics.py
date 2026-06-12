# -*- coding: utf-8 -*-
"""
stage_experiment/research/run_event_quality_diagnostics.py - 事件质量诊断

Purpose: 检查事件触发频率、簇级统计、路径分数分布，特别检查 S 簇是否触发过于频繁。

Public API:
    run_diagnostics(result_df, config=None) -> dict

Inputs:
    result_df: run_event_path_analysis 的输出 DataFrame
    config: 配置字典（用于获取事件簇映射）

Outputs: 诊断结果字典 + 控制台打印
How to Run:
    python -m stage_experiment.research.run_event_quality_diagnostics --input result.parquet
    python -m stage_experiment.research.run_event_quality_diagnostics --symbols 000001 600036
Examples:
    python -m stage_experiment.research.run_event_quality_diagnostics --symbols 000001 600036
Side Effects: 无
"""
import argparse
import sys
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from stage_experiment.core.event_groups import load_config


_CLUSTER_NAMES = {
    "cost": "C(成本交换)",
    "wash": "W(回踩收回)",
    "shake": "S(下沿假破)",
    "repair": "R(修复)",
    "failure": "F(失败风险)",
    "auxiliary": "辅助观察",
}


def _build_event_cluster_map(config: dict) -> dict:
    group_defs = config.get("event_groups", {})
    evt_to_cluster = {}
    for cluster_name, evt_list in group_defs.items():
        for evt_name in evt_list:
            evt_to_cluster[evt_name] = cluster_name
    return evt_to_cluster


def _compute_event_stats(result_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    evt_to_cluster = _build_event_cluster_map(config)
    evt_cols = [c for c in result_df.columns if c in evt_to_cluster]
    total = len(result_df)

    rows = []
    for evt_name in sorted(evt_cols):
        cluster = evt_to_cluster.get(evt_name, "unknown")
        series = result_df[evt_name].fillna(0).astype(int)
        trigger_count = (series > 0).sum()
        trigger_rate = trigger_count / total if total > 0 else 0.0
        rows.append({
            "event": evt_name,
            "cluster": _CLUSTER_NAMES.get(cluster, cluster),
            "cluster_key": cluster,
            "trigger_count": trigger_count,
            "trigger_rate": trigger_rate,
            "mean": series.mean(),
        })

    return pd.DataFrame(rows).sort_values(["cluster_key", "event"])


def _compute_cluster_stats(result_df: pd.DataFrame) -> pd.DataFrame:
    cluster_cols = {
        "cost": "cost_event_count",
        "wash": "wash_event_count",
        "shake": "shake_event_count",
        "repair": "repair_event_count",
        "failure": "failure_event_count",
    }

    rows = []
    for cluster_key, col in cluster_cols.items():
        if col not in result_df.columns:
            continue
        series = result_df[col].fillna(0).astype(float)
        total = len(series)
        nonzero = (series > 0).sum()
        rows.append({
            "cluster": _CLUSTER_NAMES.get(cluster_key, cluster_key),
            "cluster_key": cluster_key,
            "nonzero_count": nonzero,
            "nonzero_rate": nonzero / total if total > 0 else 0.0,
            "mean": series.mean(),
            "p25": series.quantile(0.25),
            "p50": series.quantile(0.50),
            "p75": series.quantile(0.75),
            "max": series.max(),
        })

    return pd.DataFrame(rows)


def _compute_score_stats(result_df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "cw_context_score",
        "cws_path_score",
        "cwsr_path_score",
        "cws_failure_adjusted_score",
    ]
    existing = [c for c in score_cols if c in result_df.columns]
    if not existing:
        return pd.DataFrame()
    return result_df[existing].describe().T


def _compute_boundary_stats(result_df: pd.DataFrame) -> pd.DataFrame:
    boundary_cols = [
        "stage_lower_boundary",
        "stage_mid_boundary",
        "stage_upper_boundary",
        "price_pos_in_stage_01",
        "dist_to_stage_lower_atr",
        "dist_to_stage_upper_atr",
    ]
    existing = [c for c in boundary_cols if c in result_df.columns]
    if not existing:
        return pd.DataFrame()
    return result_df[existing].describe().T


def run_diagnostics(result_df: pd.DataFrame, config: dict = None) -> dict:
    """
    事件质量诊断。

    Args:
        result_df: run_event_path_analysis 的输出 DataFrame
        config: 配置字典

    Returns:
        诊断结果字典，含 event_stats, cluster_stats, score_stats, shake_overtrigger
    """
    if config is None:
        config = load_config()

    event_stats = _compute_event_stats(result_df, config)
    cluster_stats = _compute_cluster_stats(result_df)
    score_stats = _compute_score_stats(result_df)
    boundary_stats = _compute_boundary_stats(result_df)

    shake_nonzero_rate = 0.0
    if "shake_event_count" in result_df.columns:
        shake_nonzero_rate = (result_df["shake_event_count"] > 0).mean()
    shake_overtrigger = shake_nonzero_rate > 0.5

    print("=" * 70)
    print("事件质量诊断报告")
    print("=" * 70)
    print(f"数据总行数: {len(result_df)}")
    print(f"股票数: {result_df['ts_code'].nunique() if 'ts_code' in result_df.columns else 'N/A'}")
    print()

    print("--- 1. 单事件触发统计 ---")
    print(event_stats[["event", "cluster", "trigger_count", "trigger_rate", "mean"]].to_string(index=False))
    print()

    print("--- 2. 簇级统计 ---")
    print(cluster_stats[["cluster", "nonzero_count", "nonzero_rate", "mean", "p25", "p50", "p75", "max"]].to_string(index=False))
    print()

    print("--- 3. 路径分数分布 ---")
    if not score_stats.empty:
        print(score_stats.to_string())
    else:
        print("无路径分数列")
    print()

    print("--- 4. S 簇过触发检查 ---")
    print(f"shake_event_count 非零率: {shake_nonzero_rate:.2%}")
    if shake_overtrigger:
        print("⚠ 警告: S 簇触发率超过 50%，可能存在过触发问题")
    else:
        print("✅ S 簇触发率在合理范围")
    print()

    print("--- 5. 边界因子分布 ---")
    if not boundary_stats.empty:
        print(boundary_stats.to_string())
    else:
        print("无边界因子列")
    print()

    return {
        "event_stats": event_stats,
        "cluster_stats": cluster_stats,
        "score_stats": score_stats,
        "boundary_stats": boundary_stats,
        "shake_overtrigger": shake_overtrigger,
        "shake_nonzero_rate": shake_nonzero_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="事件质量诊断")
    parser.add_argument("--input", type=str, default=None, help="输入文件路径（.csv 或 .parquet）")
    parser.add_argument("--symbols", nargs="+", default=None, help="股票代码列表（若无 input 文件则在线获取）")
    parser.add_argument("--count", type=int, default=255, help="拉取K线数量")
    args = parser.parse_args()

    if args.input:
        if args.input.endswith(".parquet"):
            result_df = pd.read_parquet(args.input)
        else:
            result_df = pd.read_csv(args.input)
    elif args.symbols:
        from stage_experiment.research.run_event_path_analysis import run_analysis
        result_df = run_analysis(args.symbols, count=args.count)
    else:
        print("请指定 --input 或 --symbols")
        sys.exit(1)

    if result_df.empty:
        print("无有效数据")
        sys.exit(1)

    run_diagnostics(result_df)


if __name__ == "__main__":
    main()
