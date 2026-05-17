# -*- coding: utf-8 -*-
"""
factor_lib/categories/sr_trend.py - SR趋势类因子

Purpose: SR趋势类因子的批量计算与注册。
         基于 features.sr_event_factor_lab.compute_sr_factor_lab() 权威实现。

Public API:
    compute_sr_trend_factors(df) -> DataFrame

Registered Factors:
    - ma20: 20期均线
    - ma60: 60期均线
    - close_above_ma20: 收盘在MA20上方
    - close_above_ma60: 收盘在MA60上方
    - trend_ma_bull: 多头均线环境
    - trend_ma_bear: 空头均线环境
    - ret_5: 5期收益
    - ret_10: 10期收益
    - ret_20: 20期收益
"""
from factor_lib.registry import register_factor
import pandas as pd


_FACTOR_NAMES = [
    "ma20",
    "ma60",
    "close_above_ma20",
    "close_above_ma60",
    "trend_ma_bull",
    "trend_ma_bear",
    "ret_5",
    "ret_10",
    "ret_20",
]


def compute_sr_trend_factors(df: pd.DataFrame) -> pd.DataFrame:
    from features.sr_event_factor_lab import compute_sr_factor_lab, LabConfig

    lab = compute_sr_factor_lab(df, LabConfig())
    return lab[_FACTOR_NAMES].copy()


def _compute_ma20(df):
    return compute_sr_trend_factors(df)["ma20"]


def _compute_ma60(df):
    return compute_sr_trend_factors(df)["ma60"]


def _compute_close_above_ma20(df):
    return compute_sr_trend_factors(df)["close_above_ma20"]


def _compute_close_above_ma60(df):
    return compute_sr_trend_factors(df)["close_above_ma60"]


def _compute_trend_ma_bull(df):
    return compute_sr_trend_factors(df)["trend_ma_bull"]


def _compute_trend_ma_bear(df):
    return compute_sr_trend_factors(df)["trend_ma_bear"]


def _compute_ret_5(df):
    return compute_sr_trend_factors(df)["ret_5"]


def _compute_ret_10(df):
    return compute_sr_trend_factors(df)["ret_10"]


def _compute_ret_20(df):
    return compute_sr_trend_factors(df)["ret_20"]


register_factor(
    name="ma20",
    category="SR趋势类",
    compute_func=_compute_ma20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="20期均线",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="ma60",
    category="SR趋势类",
    compute_func=_compute_ma60,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="60期均线",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="close_above_ma20",
    category="SR趋势类",
    compute_func=_compute_close_above_ma20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="收盘在MA20上方",
    direction="positive",
    is_core=False,
)

register_factor(
    name="close_above_ma60",
    category="SR趋势类",
    compute_func=_compute_close_above_ma60,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="收盘在MA60上方",
    direction="positive",
    is_core=False,
)

register_factor(
    name="trend_ma_bull",
    category="SR趋势类",
    compute_func=_compute_trend_ma_bull,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="多头均线环境",
    direction="positive",
    is_core=True,
)

register_factor(
    name="trend_ma_bear",
    category="SR趋势类",
    compute_func=_compute_trend_ma_bear,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="空头均线环境",
    direction="negative",
    is_core=True,
)

register_factor(
    name="ret_5",
    category="SR趋势类",
    compute_func=_compute_ret_5,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="5期收益",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="ret_10",
    category="SR趋势类",
    compute_func=_compute_ret_10,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="10期收益",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="ret_20",
    category="SR趋势类",
    compute_func=_compute_ret_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="20期收益",
    direction="neutral",
    is_core=False,
)


if __name__ == "__main__":
    import numpy as np

    n = 80
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.3)
    df = pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.1,
            "high": close + np.abs(np.random.randn(n)) * 0.2,
            "low": close - np.abs(np.random.randn(n)) * 0.2,
            "close": close,
            "volume": np.random.randint(1000, 5000, n).astype(float),
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.01
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.01
    result = compute_sr_trend_factors(df)
    print(result.tail(5))
    print(f"\n列数: {len(result.columns)}, 预期: {len(_FACTOR_NAMES)}")
