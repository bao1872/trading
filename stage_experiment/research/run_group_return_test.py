# -*- coding: utf-8 -*-
"""
stage_experiment/research/run_group_return_test.py - 分组收益验证

Purpose: 基于 run_event_path_analysis 输出做 MVP 分组收益验证。
         使用分位数分组（非固定阈值），不输出 stage_guess/buy_signal。

Public API:
    run_group_return_test(result_df, config=None, high_quantile=0.8, low_quantile=0.5) -> DataFrame

Inputs:
    result_df: run_event_path_analysis 的输出 DataFrame
    config: 配置字典
    high_quantile: "高"的分位数阈值（默认 P80 = top 20%）
    low_quantile: "低"的分位数阈值（默认 P50 = below median）

Outputs: 分组收益统计 DataFrame
How to Run:
    python -m stage_experiment.research.run_group_return_test --symbols 000001 600036
    python -m stage_experiment.research.run_group_return_test --input result.parquet
Examples:
    python -m stage_experiment.research.run_group_return_test --symbols 000001 600036
    python -m stage_experiment.research.run_group_return_test --input result.parquet --high_q 0.75 --low_q 0.4
Side Effects: 无
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from stage_experiment.core.event_groups import load_config


_GROUP_DEFINITIONS = {
    "G1_S高_C低_W低": {
        "shake_density_mid": "high",
        "cost_density_mid": "low",
        "wash_density_mid": "low",
    },
    "G2_C高_W低_S高": {
        "cost_density_mid": "high",
        "wash_density_mid": "low",
        "shake_density_mid": "high",
    },
    "G3_C高_W高_S高": {
        "cost_density_mid": "high",
        "wash_density_mid": "high",
        "shake_density_mid": "high",
    },
    "G4_C高_W高_S高_R高": {
        "cost_density_mid": "high",
        "wash_density_mid": "high",
        "shake_density_mid": "high",
        "repair_density_mid": "high",
    },
    "G5_F高": {
        "failure_density_mid": "high",
    },
}


def _assign_group(row: pd.Series, thresholds: dict, group_defs: dict) -> str:
    groups = []
    for group_name, conditions in group_defs.items():
        match = True
        for col, level in conditions.items():
            if col not in row.index or pd.isna(row[col]):
                match = False
                break
            val = row[col]
            if level == "high" and val < thresholds[col]["high"]:
                match = False
                break
            if level == "low" and val >= thresholds[col]["low"]:
                match = False
                break
        if match:
            groups.append(group_name)
    return "|".join(groups) if groups else "other"


def _compute_group_stats(group_df: pd.DataFrame) -> dict:
    n = len(group_df)
    if n == 0:
        return {"sample_count": 0}

    ret_cols = ["future_ret_5", "future_ret_10", "future_ret_20", "future_ret_40"]
    mdd_cols = ["future_mdd_20", "future_mdd_40"]

    result = {"sample_count": n}

    for col in ret_cols:
        if col in group_df.columns:
            valid = group_df[col].dropna()
            result[f"{col}_mean"] = valid.mean() if len(valid) > 0 else np.nan
            result[f"{col}_median"] = valid.median() if len(valid) > 0 else np.nan

    if "future_ret_20" in group_df.columns:
        valid = group_df["future_ret_20"].dropna()
        result["win_rate_20"] = (valid > 0).mean() if len(valid) > 0 else np.nan

    for col in mdd_cols:
        if col in group_df.columns:
            valid = group_df[col].dropna()
            result[f"{col}_mean"] = valid.mean() if len(valid) > 0 else np.nan

    ret_20_mean = result.get("future_ret_20_mean", np.nan)
    mdd_20_mean = result.get("future_mdd_20_mean", np.nan)
    if pd.notna(ret_20_mean) and pd.notna(mdd_20_mean) and mdd_20_mean != 0:
        result["return_mdd_ratio"] = ret_20_mean / abs(mdd_20_mean)
    else:
        result["return_mdd_ratio"] = np.nan

    return result


def run_group_return_test(
    result_df: pd.DataFrame,
    config: dict = None,
    high_quantile: float = 0.8,
    low_quantile: float = 0.5,
) -> pd.DataFrame:
    """
    分组收益验证。

    Args:
        result_df: run_event_path_analysis 的输出 DataFrame
        config: 配置字典
        high_quantile: "高"的分位数阈值
        low_quantile: "低"的分位数阈值

    Returns:
        分组收益统计 DataFrame
    """
    if config is None:
        config = load_config()

    density_cols_needed = set()
    for conditions in _GROUP_DEFINITIONS.values():
        density_cols_needed.update(conditions.keys())

    thresholds = {}
    for col in density_cols_needed:
        if col in result_df.columns:
            valid = result_df[col].dropna()
            thresholds[col] = {
                "high": valid.quantile(high_quantile) if len(valid) > 0 else np.nan,
                "low": valid.quantile(low_quantile) if len(valid) > 0 else np.nan,
            }

    print("=" * 70)
    print("分组收益验证")
    print("=" * 70)
    print(f"数据总行数: {len(result_df)}")
    print(f"股票数: {result_df['ts_code'].nunique() if 'ts_code' in result_df.columns else 'N/A'}")
    print(f"高阈值分位数: P{int(high_quantile * 100)}, 低阈值分位数: P{int(low_quantile * 100)}")
    print()
    print("--- 分位数阈值 ---")
    for col, th in sorted(thresholds.items()):
        print(f"  {col}: 高>={th['high']:.4f}, 低<{th['low']:.4f}")
    print()

    group_series = result_df.apply(
        lambda row: _assign_group(row, thresholds, _GROUP_DEFINITIONS), axis=1
    )

    all_groups = {}
    for group_name in list(_GROUP_DEFINITIONS.keys()) + ["other"]:
        mask = group_series.str.contains(group_name.replace("|", "\\|")) if group_name != "other" else group_series == "other"
        if mask.any():
            group_data = result_df[mask]
            stats = _compute_group_stats(group_data)
            stats["group"] = group_name
            all_groups[group_name] = stats

    if not all_groups:
        print("无有效分组")
        return pd.DataFrame()

    stats_df = pd.DataFrame(all_groups.values())

    col_order = ["group", "sample_count"]
    for suffix in ["_mean", "_median"]:
        for ret in ["future_ret_5", "future_ret_10", "future_ret_20", "future_ret_40"]:
            c = f"{ret}{suffix}"
            if c in stats_df.columns:
                col_order.append(c)
    for extra in ["win_rate_20", "future_mdd_20_mean", "future_mdd_40_mean", "return_mdd_ratio"]:
        if extra in stats_df.columns:
            col_order.append(extra)

    existing_cols = [c for c in col_order if c in stats_df.columns]
    remaining = [c for c in stats_df.columns if c not in existing_cols]
    stats_df = stats_df[existing_cols + remaining]

    print("--- 分组收益统计 ---")
    print(stats_df.to_string(index=False))
    print()

    for group_name in _GROUP_DEFINITIONS.keys():
        if group_name in all_groups:
            n = all_groups[group_name]["sample_count"]
            rate = n / len(result_df) * 100
            print(f"  {group_name}: {n} 样本 ({rate:.1f}%)")

    return stats_df


def main():
    parser = argparse.ArgumentParser(description="分组收益验证")
    parser.add_argument("--input", type=str, default=None, help="输入文件路径（.csv 或 .parquet）")
    parser.add_argument("--symbols", nargs="+", default=None, help="股票代码列表")
    parser.add_argument("--count", type=int, default=255, help="拉取K线数量")
    parser.add_argument("--high_q", type=float, default=0.8, help="高阈值分位数（默认0.8）")
    parser.add_argument("--low_q", type=float, default=0.5, help="低阈值分位数（默认0.5）")
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

    run_group_return_test(result_df, high_quantile=args.high_q, low_quantile=args.low_q)


if __name__ == "__main__":
    main()
