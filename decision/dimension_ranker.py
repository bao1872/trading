# -*- coding: utf-8 -*-
"""
decision/dimension_ranker.py - 维度排序器

Purpose: 按不同维度对股票进行排序。

Usage:
    from decision.dimension_ranker import rank_by_dimensions
    rankings = rank_by_dimensions(research_panel, dimensions=['momentum', 'value'])
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from typing import Dict, Any, List, Optional


# 维度定义：列名 -> 排序方向（True=升序，False=降序）
DIMENSION_MAP = {
    "momentum": {"columns": ["bbmacd", "bbmacd_slope_3"], "ascending": False},
    "value": {"columns": ["price_vs_dsa_vwap_pct"], "ascending": True},
    "volume": {"columns": ["vol_zscore_20"], "ascending": False},
    "trend": {"columns": ["dsa_dir", "trend_align_momo"], "ascending": False},
    "risk": {"columns": ["atr_pct", "volatility_20d"], "ascending": True},
}


def rank_by_dimensions(
    research_panel: pd.DataFrame,
    dimensions: Optional[List[str]] = None,
    top_n: int = 20,
) -> Dict[str, List[tuple]]:
    """
    按多个维度对股票排序。

    Args:
        research_panel: 标准个股底表
        dimensions: 维度列表，None表示全部
        top_n: 每个维度返回前N名

    Returns:
        各维度排名结果
    """
    if dimensions is None:
        dimensions = list(DIMENSION_MAP.keys())

    result = {}
    for dim in dimensions:
        if dim not in DIMENSION_MAP:
            continue

        config = DIMENSION_MAP[dim]
        cols = [c for c in config["columns"] if c in research_panel.columns]
        if not cols:
            continue

        # 计算综合得分（各列标准化后求和）
        scores = pd.Series(0.0, index=research_panel.index)
        for col in cols:
            col_data = research_panel[col].fillna(0)
            std = col_data.std()
            if std > 0:
                normalized = (col_data - col_data.mean()) / std
            else:
                normalized = pd.Series(0.0, index=research_panel.index)
            scores += normalized

        if not config["ascending"]:
            scores = -scores

        # 排序
        ranked = research_panel.copy()
        ranked["_score"] = scores
        ranked = ranked.sort_values("_score", ascending=True)

        # 取前N
        top = ranked.head(top_n)[["ts_code", "_score"]].values.tolist()
        result[dim] = [(code, round(float(score), 4)) for code, score in top]

    return result
