# -*- coding: utf-8 -*-
"""
stage_experiment/core/event_context.py - 上下文特征整合

Purpose: 整合衰减强度和序列统计为上下文特征，计算 pre_shake_context_strength。

Public API:
    compute_event_context(decay_df, sequence_df, config=None) -> DataFrame

Output Columns (6列):
    cost_context_strength: 成本交换背景强度（复用 decay 输出）
    wash_context_strength: 回踩收回背景强度（复用 decay 输出）
    shake_context_strength: 下沿假破背景强度（复用 decay 输出）
    repair_context_strength: 修复背景强度（复用 decay 输出）
    pre_shake_context_strength: 前置震仓上下文强度
    failure_context_strength: 失败风险背景强度（复用 decay 输出）

Formula:
    pre_shake_context_strength = cost_context_strength * wash_context_strength * (1 - failure_context_strength)

Inputs:
    decay_df (compute_event_decay 输出)
    sequence_df (compute_event_sequence 输出)
Outputs: DataFrame with 4 columns
How to Run:
    python -m stage_experiment.core.event_context
Examples:
    python -m stage_experiment.core.event_context
Side Effects: None
"""
import pandas as pd
import numpy as np


def compute_event_context(
    decay_df: pd.DataFrame,
    sequence_df: pd.DataFrame = None,
    config: dict = None,
) -> pd.DataFrame:
    """
    整合衰减强度为上下文特征。

    Args:
        decay_df: compute_event_decay() 输出
        sequence_df: compute_event_sequence() 输出（当前未使用，保留接口）
        config: 未使用，保留接口一致性

    Returns:
        DataFrame 含 4 列上下文特征
    """
    result = pd.DataFrame(index=decay_df.index)

    cost_strength = decay_df.get("cost_context_strength", pd.Series(0.0, index=decay_df.index))
    wash_strength = decay_df.get("wash_context_strength", pd.Series(0.0, index=decay_df.index))
    shake_strength = decay_df.get("shake_context_strength", pd.Series(0.0, index=decay_df.index))
    repair_strength = decay_df.get("repair_context_strength", pd.Series(0.0, index=decay_df.index))
    failure_strength = decay_df.get("failure_context_strength", pd.Series(0.0, index=decay_df.index))

    result["cost_context_strength"] = cost_strength
    result["wash_context_strength"] = wash_strength
    result["shake_context_strength"] = shake_strength
    result["repair_context_strength"] = repair_strength
    result["failure_context_strength"] = failure_strength

    pre_shake = cost_strength * wash_strength * (1.0 - failure_strength)
    result["pre_shake_context_strength"] = pre_shake.clip(0.0, 1.0)

    return result


if __name__ == "__main__":
    from stage_experiment.core.event_groups import map_event_groups
    from stage_experiment.core.event_decay import compute_event_decay
    from stage_experiment.core.event_sequence import compute_event_sequence

    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    fake_events = pd.DataFrame(
        {
            "evt_trend_flat": np.random.choice([0, 1], n, p=[0.8, 0.2]),
            "evt_price_cross_mid_freq_high": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "evt_volatility_compress": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "evt_upper_reject": np.random.choice([0, 1], n, p=[0.85, 0.15]),
            "evt_lower_touch": np.random.choice([0, 1], n, p=[0.8, 0.2]),
            "evt_pullback_to_lower": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "evt_price_cross_above_mid": np.random.choice([0, 1], n, p=[0.85, 0.15]),
            "evt_range_pullback_reclaim_cycle": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "evt_lower_break": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "evt_lower_break_reclaim": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "evt_long_lower_shadow": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "evt_stop_cluster_reclaim": np.random.choice([0, 1], n, p=[0.95, 0.05]),
            "evt_range_expand_down_reclaim": np.random.choice([0, 1], n, p=[0.95, 0.05]),
            "evt_reclaim_lower": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "evt_reclaim_mid": np.random.choice([0, 1], n, p=[0.85, 0.15]),
            "evt_reclaim_dsa_vwap": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "evt_trend_slope_turn_positive": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "evt_bbmacd_turn_positive": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "evt_lower_break_no_reclaim": np.random.choice([0, 1], n, p=[0.95, 0.05]),
            "evt_trend_down_confirm": np.random.choice([0, 1], n, p=[0.95, 0.05]),
            "evt_weak_rebound": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        },
        index=dates,
    )

    group_df = map_event_groups(fake_events)
    decay_df = compute_event_decay(group_df)
    seq_df = compute_event_sequence(group_df)
    context_df = compute_event_context(decay_df, seq_df)
    print("=== 上下文特征输出 ===")
    print(context_df.describe())
