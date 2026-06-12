# -*- coding: utf-8 -*-
"""
stage_experiment/core/event_decay.py - 事件衰减背景强度计算

Purpose: 使用指数衰减代替硬窗口，计算各事件簇的上下文背景强度。

Public API:
    compute_event_decay(group_df, config=None) -> DataFrame

Output Columns (5列):
    cost_context_strength
    wash_context_strength
    shake_context_strength
    repair_context_strength
    failure_context_strength

Formula:
    context_strength[t] = context_strength[t-1] * decay + event_count[t]
    decay = exp(-ln(2) / half_life)
    归一化到 [0, 1]

Inputs: group_df (map_event_groups 输出)
Outputs: DataFrame with 5 columns
How to Run:
    python -m stage_experiment.core.event_decay
Examples:
    python -m stage_experiment.core.event_decay
Side Effects: None
"""
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "stage_event_config.yaml"


def load_config(config_path=None):
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _exponential_decay_series(event_count_series: pd.Series, half_life: int) -> pd.Series:
    decay_factor = np.exp(-np.log(2) / half_life)
    n = len(event_count_series)
    result = np.zeros(n, dtype=float)
    vals = event_count_series.fillna(0).astype(float).to_numpy()
    for i in range(1, n):
        result[i] = result[i - 1] * decay_factor + vals[i]
    max_val = result.max()
    if max_val > 0:
        result = result / max_val
    return pd.Series(result, index=event_count_series.index)


def compute_event_decay(
    group_df: pd.DataFrame,
    config: dict = None,
) -> pd.DataFrame:
    """
    计算各事件簇的指数衰减背景强度。

    Args:
        group_df: map_event_groups() 输出的事件计数 DataFrame
        config: 配置字典，含 decay.half_life_mid 键

    Returns:
        DataFrame 含 5 列衰减背景强度（归一化到 [0, 1]）
    """
    if config is None:
        config = load_config()

    decay_cfg = config.get("decay", {"half_life_mid": 10})
    half_life = decay_cfg.get("half_life_mid", 10)
    result = pd.DataFrame(index=group_df.index)

    group_names = ["cost", "wash", "shake", "repair", "failure"]
    for group_name in group_names:
        col = f"{group_name}_event_count"
        if col not in group_df.columns:
            result[f"{group_name}_context_strength"] = 0.0
            continue
        result[f"{group_name}_context_strength"] = _exponential_decay_series(
            group_df[col], half_life
        )

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
    decay_df = compute_event_decay(group_df)
    print("=== 事件衰减背景强度输出 ===")
    print(decay_df.describe())
