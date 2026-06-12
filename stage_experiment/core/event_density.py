# -*- coding: utf-8 -*-
"""
stage_experiment/core/event_density.py - 事件密度计算

Purpose: 计算每类事件在短/中/长窗口内的密度。

Public API:
    compute_event_density(group_df, config=None) -> DataFrame

Output Columns (15列):
    cost_density_short/mid/long
    wash_density_short/mid/long
    shake_density_short/mid/long
    repair_density_short/mid/long
    failure_density_short/mid/long

Inputs: group_df (map_event_groups 输出)
Outputs: DataFrame with 15 columns
How to Run:
    python -m stage_experiment.core.event_density
Examples:
    python -m stage_experiment.core.event_density
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


def compute_event_density(
    group_df: pd.DataFrame,
    config: dict = None,
) -> pd.DataFrame:
    """
    计算每类事件在短/中/长窗口内的密度。

    Args:
        group_df: map_event_groups() 输出的事件计数 DataFrame
        config: 配置字典，含 density_windows 键

    Returns:
        DataFrame 含 15 列密度值
    """
    if config is None:
        config = load_config()

    windows = config.get("density_windows", {"short": 10, "mid": 20, "long": 40})
    result = pd.DataFrame(index=group_df.index)

    group_names = ["cost", "wash", "shake", "repair", "failure"]
    for group_name in group_names:
        col = f"{group_name}_event_count"
        if col not in group_df.columns:
            for w_name in windows:
                result[f"{group_name}_density_{w_name}"] = np.nan
            continue
        series = group_df[col].fillna(0).astype(float)
        for w_name, w_size in windows.items():
            result[f"{group_name}_density_{w_name}"] = series.rolling(
                w_size, min_periods=max(1, w_size // 2)
            ).mean()

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
    density_df = compute_event_density(group_df)
    print("=== 事件密度输出 ===")
    print(f"列数: {len(density_df.columns)}")
    print(density_df.describe())
