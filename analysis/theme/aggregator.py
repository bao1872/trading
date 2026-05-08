# -*- coding: utf-8 -*-
"""
analysis/theme/aggregator.py - 题材聚合

Purpose: 从 theme_aggregator.py 迁移，保持现有功能。

Usage:
    from analysis.theme.aggregator import aggregate_by_theme
    themes = aggregate_by_theme(events_df, concept_df)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd


def aggregate_by_theme(events_df: pd.DataFrame, concept_df: pd.DataFrame) -> pd.DataFrame:
    """
    按题材聚合事件和因子。

    Args:
        events_df: 包含事件和因子列的 DataFrame
        concept_df: 概念映射 DataFrame（ts_code -> concept_name）

    Returns:
        题材聚合结果 DataFrame
    """
    # 合并概念映射
    merged = events_df.merge(concept_df, on="ts_code", how="left")

    # 按概念分组聚合
    grouped = merged.groupby("concept_name").agg({
        "ts_code": "count",
        "evt_dsa_dir_flip_up": "sum",
        "evt_dsa_dir_flip_down": "sum",
        "evt_up_move_with_vol_spike": "sum",
        "evt_down_move_with_vol_spike": "sum",
    }).rename(columns={"ts_code": "stock_count"})

    return grouped.reset_index()
