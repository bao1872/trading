# -*- coding: utf-8 -*-
"""
财务因子模块 - A股季度财务因子计算框架

功能：
1. 从 financial_quarterly_data 表获取个股季度财务数据（单季度值）
2. 计算37个财务因子（增长、盈利、质量、现金、效率、边际）
3. 时序分位数打分

Usage:
    from financial_factors import financial_quarterly_score
    # 或直接调用核心函数：
    from financial_factors.sample_score import (
        FACTOR_CONFIG, DIMENSION_WEIGHTS, dedup_latest,
        add_ytd_and_ttm, add_factors, score_dataframe,
    )
"""
from .sample_score import (
    FACTOR_CONFIG,
    DIMENSION_WEIGHTS,
    dedup_latest,
    add_ytd_and_ttm,
    add_factors,
    score_dataframe,
)

__all__ = [
    'FACTOR_CONFIG',
    'DIMENSION_WEIGHTS',
    'dedup_latest',
    'add_ytd_and_ttm',
    'add_factors',
    'score_dataframe',
]
