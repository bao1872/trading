# -*- coding: utf-8 -*-
"""
财务因子模块 - A股季度财务因子计算框架

功能：
1. 从 PostgreSQL 数据库获取个股财务数据（季度历史，YTD累计格式）
2. 计算37个财务因子（增长、盈利、质量、现金、效率、边际）
3. 时序分位数打分

Usage:
    from financial_factors.sample_score import score_dataframe, add_factors, add_ytd_and_ttm

    from financial_factors import batch_score
"""
from .sample_score import (
    FACTOR_CONFIG,
    DIMENSION_WEIGHTS,
    fetch_financial_from_db,
    dedup_latest,
    add_ytd_and_ttm,
    add_factors,
    score_dataframe,
)

__all__ = [
    'FACTOR_CONFIG',
    'DIMENSION_WEIGHTS',
    'fetch_financial_from_db',
    'dedup_latest',
    'add_ytd_and_ttm',
    'add_factors',
    'score_dataframe',
]
