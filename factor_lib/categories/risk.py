# -*- coding: utf-8 -*-
"""
factor_lib/categories/risk.py - 风险类因子

Purpose: 风险类因子（波动率/回撤/尾部风险）的批量计算与注册。
         提供公开的 compute_risk_factors() 供实验直接调用。

Public API:
    compute_risk_factors(df) -> DataFrame

Registered Factors:
    - atr_pct: ATR百分比
    - volatility_20d: 20日波动率
    - max_drawdown_60d: 60日最大回撤
    - beta: Beta系数
"""
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np


def compute_risk_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    批量计算全部风险类因子。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame，index 为 datetime

    Returns:
        DataFrame 含列: atr_pct, volatility_20d, max_drawdown_60d, beta
    """
    result = pd.DataFrame(index=df.index)

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean()
    result["atr_pct"] = atr / close * 100

    returns = close.pct_change()
    result["volatility_20d"] = returns.rolling(window=20, min_periods=1).std() * np.sqrt(252) * 100

    rolling_max = close.rolling(window=60, min_periods=1).max()
    drawdown = (close - rolling_max) / rolling_max * 100
    result["max_drawdown_60d"] = drawdown.rolling(window=60, min_periods=1).min()

    result["beta"] = returns.rolling(window=60, min_periods=1).std() * np.sqrt(252)
    return result


def _compute_atr_pct(df):
    """ATR百分比。"""
    return compute_risk_factors(df)["atr_pct"]


def _compute_volatility_20d(df):
    """20日收益率波动率（年化）。"""
    return compute_risk_factors(df)["volatility_20d"]


def _compute_max_drawdown_60d(df):
    """60日最大回撤。"""
    return compute_risk_factors(df)["max_drawdown_60d"]


def _compute_beta(df):
    """Beta系数（相对于市场，简化版）。"""
    return compute_risk_factors(df)["beta"]


# 注册风险类因子
register_factor(
    name="atr_pct",
    category="风险类",
    compute_func=_compute_atr_pct,
    source_module="factor_lib.categories.risk",
    source_function="_compute_atr_pct",
    description="ATR百分比（14日）",
    direction="negative",
    is_core=False,
)

register_factor(
    name="volatility_20d",
    category="风险类",
    compute_func=_compute_volatility_20d,
    source_module="factor_lib.categories.risk",
    source_function="_compute_volatility_20d",
    description="20日收益率波动率（年化%）",
    direction="negative",
    is_core=False,
)

register_factor(
    name="max_drawdown_60d",
    category="风险类",
    compute_func=_compute_max_drawdown_60d,
    source_module="factor_lib.categories.risk",
    source_function="_compute_max_drawdown_60d",
    description="60日最大回撤（%）",
    direction="negative",
    is_core=False,
)

register_factor(
    name="beta",
    category="风险类",
    compute_func=_compute_beta,
    source_module="factor_lib.categories.risk",
    source_function="_compute_beta",
    description="Beta系数（简化版）",
    direction="neutral",
    is_core=False,
)
