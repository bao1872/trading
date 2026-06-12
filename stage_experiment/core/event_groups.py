# -*- coding: utf-8 -*-
"""
stage_experiment/core/event_groups.py - 事件分组映射

Purpose: 读取配置，把 event_lib 输出的事件列映射成 C/W/S/R/F 五类事件矩阵。

Public API:
    map_event_groups(events_df, config=None) -> DataFrame

Output Columns:
    - cost_event_count: 成本交换簇当日事件计数
    - wash_event_count: 回踩收回簇当日事件计数
    - shake_event_count: 下沿假破簇当日事件计数
    - repair_event_count: 修复簇当日事件计数
    - failure_event_count: 失败风险簇当日事件计数

Inputs: events_df (detect_panel 输出)
Outputs: DataFrame with 5 columns
How to Run:
    python -m stage_experiment.core.event_groups
Examples:
    python -m stage_experiment.core.event_groups
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


def map_event_groups(events_df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    把 event_lib 输出的事件列映射成 C/W/S/R/F 五类事件矩阵。

    Args:
        events_df: detect_panel() 输出的事件 DataFrame
        config: 配置字典，含 event_groups 键。若为 None 则从默认 yaml 加载。

    Returns:
        DataFrame 含 5 列事件计数
    """
    if config is None:
        config = load_config()

    group_defs = config.get("event_groups", {})
    result = pd.DataFrame(index=events_df.index)

    group_names = ["cost", "wash", "shake", "repair", "failure"]
    for group_name in group_names:
        event_list = group_defs.get(group_name, [])
        count = pd.Series(0, index=events_df.index, dtype=int)
        for evt_name in event_list:
            if evt_name in events_df.columns:
                count = count + events_df[evt_name].fillna(0).astype(int)
        result[f"{group_name}_event_count"] = count

    return result


if __name__ == "__main__":
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

    result = map_event_groups(fake_events)
    print("=== 事件分组映射输出 ===")
    print(result.describe())
    print()
    for col in result.columns:
        print(f"{col}: 非零比例 = {(result[col] > 0).mean():.2%}")
