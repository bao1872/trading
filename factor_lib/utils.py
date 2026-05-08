# -*- coding: utf-8 -*-
"""
factor_lib/utils.py - 因子计算通用工具

Purpose: 提供因子计算中常用的辅助函数，如标准化、分位计算、横截面处理等。
         这些工具函数不依赖具体因子，可被所有因子复用。

Usage:
    from factor_lib.utils import zscore, rank_pct, winsorize
"""
from typing import Optional
import numpy as np
import pandas as pd


def zscore(series: pd.Series, window: Optional[int] = None) -> pd.Series:
    """
    计算 Z-score 标准化。

    Args:
        series: 输入序列
        window: 滚动窗口，None 表示用全序列

    Returns:
        Z-score 序列
    """
    if window is None:
        mean = series.mean()
        std = series.std()
    else:
        mean = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()

    std = std.replace(0, np.nan)
    return (series - mean) / std


def rank_pct(series: pd.Series) -> pd.Series:
    """
    计算横截面百分位排名（0-1）。

    Args:
        series: 输入序列

    Returns:
        百分位排名序列
    """
    return series.rank(pct=True)


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    缩尾处理（Winsorize）。

    Args:
        series: 输入序列
        lower: 下尾分位数
        upper: 上尾分位数

    Returns:
        缩尾后的序列
    """
    q_low = series.quantile(lower)
    q_high = series.quantile(upper)
    return series.clip(lower=q_low, upper=q_high)


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """滚动 Z-score。"""
    return zscore(series, window=window)


def cross_sectional_zscore(df: pd.DataFrame, factor_col: str) -> pd.Series:
    """
    横截面 Z-score（对某一日的所有股票计算）。

    Args:
        df: 包含多股票的 DataFrame
        factor_col: 因子列名

    Returns:
        横截面 Z-score 序列
    """
    mean = df[factor_col].mean()
    std = df[factor_col].std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=df.index)
    return (df[factor_col] - mean) / std


def neutralize(series: pd.Series, group: pd.Series) -> pd.Series:
    """
    行业/市值中性化（分组去均值）。

    Args:
        series: 因子序列
        group: 分组标签（如行业）

    Returns:
        中性化后的序列
    """
    return series - series.groupby(group).transform("mean")
