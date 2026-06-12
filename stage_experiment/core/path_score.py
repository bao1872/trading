# -*- coding: utf-8 -*-
"""
stage_experiment/core/path_score.py - 路径分数计算

Purpose: 构造研究用路径分数（cw/cws/cwsr/cws_failure_adjusted）。
         只用于研究，不进入 factor_lib 和 event_lib。

Public API:
    compute_path_score(context_df, group_df, config=None) -> DataFrame

Output Columns (4列):
    cw_context_score: C*W 上下文分数
    cws_path_score: C*W*S 路径分数（含失败过滤）
    cwsr_path_score: C*W*S*R 路径分数
    cws_failure_adjusted_score: C*W*S 失败调整分数

Formula:
    cw_context_score = cost_context_strength * wash_context_strength
    cws_path_score = cw_context_score * shake_event_today * (1 - failure_context_strength)
    cwsr_path_score = cws_path_score * repair_context_strength
    cws_failure_adjusted_score = cw_context_score * (1 - failure_context_strength)

Inputs:
    context_df (compute_event_context 输出)
    group_df (map_event_groups 输出)
Outputs: DataFrame with 4 columns
How to Run:
    python -m stage_experiment.core.path_score
Examples:
    python -m stage_experiment.core.path_score
Side Effects: None
"""
import pandas as pd
import numpy as np


def compute_path_score(
    context_df: pd.DataFrame,
    group_df: pd.DataFrame,
    config: dict = None,
) -> pd.DataFrame:
    """
    构造研究用路径分数。

    Args:
        context_df: compute_event_context() 输出
        group_df: map_event_groups() 输出
        config: 未使用，保留接口一致性

    Returns:
        DataFrame 含 4 列路径分数
    """
    result = pd.DataFrame(index=context_df.index)

    cost_strength = context_df.get("cost_context_strength", pd.Series(0.0, index=context_df.index))
    wash_strength = context_df.get("wash_context_strength", pd.Series(0.0, index=context_df.index))
    failure_strength = context_df.get("failure_context_strength", pd.Series(0.0, index=context_df.index))

    shake_today = group_df.get("shake_event_count", pd.Series(0, index=group_df.index))
    shake_today = (shake_today > 0).astype(float)

    repair_strength = context_df.get("repair_context_strength", pd.Series(0.0, index=context_df.index))

    cw = cost_strength * wash_strength
    result["cw_context_score"] = cw.clip(0.0, 1.0)

    cws = cw * shake_today * (1.0 - failure_strength)
    result["cws_path_score"] = cws.clip(0.0, 1.0)

    cwsr = cws * repair_strength
    result["cwsr_path_score"] = cwsr.clip(0.0, 1.0)

    cws_fail_adj = cw * (1.0 - failure_strength)
    result["cws_failure_adjusted_score"] = cws_fail_adj.clip(0.0, 1.0)

    return result


if __name__ == "__main__":
    from stage_experiment.core.event_groups import map_event_groups
    from stage_experiment.core.event_decay import compute_event_decay
    from stage_experiment.core.event_context import compute_event_context

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
    context_df = compute_event_context(decay_df)
    score_df = compute_path_score(context_df, group_df)
    print("=== 路径分数输出 ===")
    print(score_df.describe())
