# -*- coding: utf-8 -*-
"""
analysis/theme/rotation.py - 轮动检测

Purpose: 检测大小盘轮动、行业轮动、题材接力。

Usage:
    from analysis.theme.rotation import detect_rotation
    rotation = detect_rotation(market_df)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from typing import Dict, Any


def detect_rotation(market_df: pd.DataFrame, window: int = 5) -> Dict[str, Any]:
    """
    检测市场轮动状态。

    Args:
        market_df: 市场聚合数据（包含 cap_tier/industry 分组）
        window: 观察窗口

    Returns:
        轮动检测结果
    """
    result = {
        "cap_rotation": "unknown",
        "industry_rotation": "unknown",
        "theme_handoff": "unknown",
    }

    # 大小盘轮动
    if "cap_tier" in market_df.columns and "structure_score" in market_df.columns:
        cap_groups = market_df.groupby("cap_tier")["structure_score"].mean()
        if len(cap_groups) >= 2:
            large_cap = cap_groups.get("large", 0)
            small_cap = cap_groups.get("small", 0)
            if small_cap > large_cap * 1.2:
                result["cap_rotation"] = "small_cap_dominant"
            elif large_cap > small_cap * 1.2:
                result["cap_rotation"] = "large_cap_dominant"
            else:
                result["cap_rotation"] = "balanced"

    # 行业轮动（简化：用结构分变化速度）
    if "industry" in market_df.columns and "structure_score" in market_df.columns:
        industry_scores = market_df.groupby("industry")["structure_score"].mean().sort_values(ascending=False)
        if len(industry_scores) >= 3:
            top3 = industry_scores.head(3)
            result["industry_rotation"] = {
                "top_industries": top3.index.tolist(),
                "top_scores": top3.values.tolist(),
            }

    return result
