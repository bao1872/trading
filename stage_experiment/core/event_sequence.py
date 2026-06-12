# -*- coding: utf-8 -*-
"""
stage_experiment/core/event_sequence.py - 事件顺序与距离统计

Purpose: 统计事件顺序（S之前是否有C/W）和事件距离（最近一次事件距今多少bar）。

Public API:
    compute_event_sequence(group_df, config=None) -> DataFrame

Output Columns (7列):
    has_cost_before_shake: 当前S事件之前是否已有C事件
    has_wash_before_shake: 当前S事件之前是否已有W事件
    has_failure_before_shake: 当前S事件之前是否已有F事件
    bars_since_last_cost: 最近一次C事件距今bar数
    bars_since_last_wash: 最近一次W事件距今bar数
    bars_since_last_shake: 最近一次S事件距今bar数
    bars_since_last_failure: 最近一次F事件距今bar数

Inputs: group_df (map_event_groups 输出)
Outputs: DataFrame with 7 columns
How to Run:
    python -m stage_experiment.core.event_sequence
Examples:
    python -m stage_experiment.core.event_sequence
Side Effects: None
"""
import pandas as pd
import numpy as np


def _bars_since_last(series: pd.Series) -> pd.Series:
    n = len(series)
    result = np.full(n, np.nan, dtype=float)
    last_idx = -1
    vals = series.fillna(0).astype(int).to_numpy()
    for i in range(n):
        if vals[i] > 0:
            last_idx = i
        if last_idx >= 0:
            result[i] = i - last_idx
    return pd.Series(result, index=series.index)


def _has_before(shake_series: pd.Series, other_series: pd.Series) -> pd.Series:
    n = len(shake_series)
    result = np.zeros(n, dtype=int)
    shake_vals = shake_series.fillna(0).astype(int).to_numpy()
    other_vals = other_series.fillna(0).astype(int).to_numpy()
    had_other = False
    for i in range(n):
        if other_vals[i] > 0:
            had_other = True
        if shake_vals[i] > 0 and had_other:
            result[i] = 1
    return pd.Series(result, index=shake_series.index)


def compute_event_sequence(
    group_df: pd.DataFrame,
    config: dict = None,
) -> pd.DataFrame:
    """
    统计事件顺序与距离。

    Args:
        group_df: map_event_groups() 输出的事件计数 DataFrame
        config: 未使用，保留接口一致性

    Returns:
        DataFrame 含 7 列顺序/距离统计
    """
    result = pd.DataFrame(index=group_df.index)

    cost_col = group_df.get("cost_event_count", pd.Series(0, index=group_df.index))
    wash_col = group_df.get("wash_event_count", pd.Series(0, index=group_df.index))
    shake_col = group_df.get("shake_event_count", pd.Series(0, index=group_df.index))
    failure_col = group_df.get("failure_event_count", pd.Series(0, index=group_df.index))

    shake_any = (shake_col > 0).astype(int)
    result["has_cost_before_shake"] = _has_before(shake_any, cost_col)
    result["has_wash_before_shake"] = _has_before(shake_any, wash_col)
    result["has_failure_before_shake"] = _has_before(shake_any, failure_col)

    result["bars_since_last_cost"] = _bars_since_last(cost_col)
    result["bars_since_last_wash"] = _bars_since_last(wash_col)
    result["bars_since_last_shake"] = _bars_since_last(shake_col)
    result["bars_since_last_failure"] = _bars_since_last(failure_col)

    return result


if __name__ == "__main__":
    from stage_experiment.core.event_groups import map_event_groups

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
    seq_df = compute_event_sequence(group_df)
    print("=== 事件序列统计输出 ===")
    print(seq_df.describe())
    print()
    print("has_cost_before_shake 分布:")
    print(seq_df["has_cost_before_shake"].value_counts())
