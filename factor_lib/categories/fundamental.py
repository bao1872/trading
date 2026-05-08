# -*- coding: utf-8 -*-
"""
factor_lib/categories/fundamental.py - 财务类因子

Purpose: 财务类因子（营收/利润/ROE/现金流/偿债能力等）的批量计算与注册。

Important:
    财务因子需要财务数据，compute_func 接收的 df 需包含对应财务列。
    实验前应先从 financial_factors/sample_score.py 调用 add_factors() 预加载财务数据，
    或从 factor_value 表读取已有财务因子值。

Public API:
    compute_fundamental_factors(df) -> DataFrame

Registered Factors:
    - q_rev_yoy: 季度营收同比增长率
    - q_gross_margin: 季度毛利率
    - q_np_yoy: 季度净利润同比增长率
    - roe_weighted: 加权ROE
    - roa_parent: 归母ROA
    - cfo_to_np_parent: 经营现金流/归母净利润
    - cfo_to_asset: 经营现金流/总资产
    - cash_cov_interest: 现金利息保障倍数
    - debt_to_equity: 产权比率
    - current_ratio: 流动比率
    - quick_ratio: 速动比率
    - q_rev_delta: 季度营收环比变化
    - q_np_delta: 季度净利润环比变化
    - q_total_asset_yoy: 季度总资产同比增长率
"""
import warnings
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np

FUNDAMENTAL_FACTORS = [
    ("q_rev_yoy", "季度营收同比增长率", "positive"),
    ("q_gross_margin", "季度毛利率", "positive"),
    ("q_np_yoy", "季度净利润同比增长率", "positive"),
    ("roe_weighted", "加权ROE", "positive"),
    ("roa_parent", "归母ROA", "positive"),
    ("cfo_to_np_parent", "经营现金流/归母净利润", "positive"),
    ("cfo_to_asset", "经营现金流/总资产", "positive"),
    ("cash_cov_interest", "现金利息保障倍数", "positive"),
    ("debt_to_equity", "产权比率", "negative"),
    ("current_ratio", "流动比率", "positive"),
    ("quick_ratio", "速动比率", "positive"),
    ("q_rev_delta", "季度营收环比变化", "positive"),
    ("q_np_delta", "季度净利润环比变化", "positive"),
    ("q_total_asset_yoy", "季度总资产同比增长率", "positive"),
]


def compute_fundamental_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    批量提取全部财务类因子。

    若 df 中已有财务列，直接返回对应列；否则返回含 NaN 的 DataFrame
    并提示需从 financial_factors 或数据库预加载财务数据。

    Args:
        df: DataFrame，如已含财务列则直接提取

    Returns:
        DataFrame 含 14 列财务因子
    """
    result = pd.DataFrame(index=df.index)
    missing = []
    for name, _, _ in FUNDAMENTAL_FACTORS:
        if name in df.columns:
            result[name] = df[name]
        else:
            result[name] = np.nan
            missing.append(name)
    if missing:
        warnings.warn(
            f"财务因子缺失 {len(missing)} 列，请先调用 financial_factors.sample_score.add_factors() "
            f"或从 factor_value 表加载财务数据。缺失列: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return result


def _make_fundamental_factor(factor_name: str):
    def _compute(df):
        if factor_name in df.columns:
            return df[factor_name]
        return pd.Series(np.nan, index=df.index, name=factor_name)
    return _compute

for name, desc, direction in FUNDAMENTAL_FACTORS:
    register_factor(
        name=name,
        category="财务类",
        compute_func=_make_fundamental_factor(name),
        source_module="financial_factors.sample_score",
        source_function="add_factors",
        description=desc,
        direction=direction,
        is_core=False,
    )
