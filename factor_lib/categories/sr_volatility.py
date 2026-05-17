# -*- coding: utf-8 -*-
"""
factor_lib/categories/sr_volatility.py - SR波动类因子

Purpose: SR波动类因子的批量计算与注册。
         基于 features.sr_event_factor_lab.compute_sr_factor_lab() 权威实现。

Public API:
    compute_sr_volatility_factors(df) -> DataFrame

Registered Factors:
    - atr_14: 14期ATR
    - atr_pct_14: ATR/close
    - realized_vol_20: 20期收益波动
    - max_drawdown_20: 20期最大回撤
"""
from factor_lib.registry import register_factor
import pandas as pd


_FACTOR_NAMES = [
    "atr_14",
    "atr_pct_14",
    "realized_vol_20",
    "max_drawdown_20",
]


def compute_sr_volatility_factors(df: pd.DataFrame) -> pd.DataFrame:
    from features.sr_event_factor_lab import compute_sr_factor_lab, LabConfig

    lab = compute_sr_factor_lab(df, LabConfig())
    return lab[_FACTOR_NAMES].copy()


def _compute_atr_14(df):
    return compute_sr_volatility_factors(df)["atr_14"]


def _compute_atr_pct_14(df):
    return compute_sr_volatility_factors(df)["atr_pct_14"]


def _compute_realized_vol_20(df):
    return compute_sr_volatility_factors(df)["realized_vol_20"]


def _compute_max_drawdown_20(df):
    return compute_sr_volatility_factors(df)["max_drawdown_20"]


register_factor(
    name="atr_14",
    category="SR波动类",
    compute_func=_compute_atr_14,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="14期ATR",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="atr_pct_14",
    category="SR波动类",
    compute_func=_compute_atr_pct_14,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="ATR/close",
    direction="negative",
    is_core=False,
)

register_factor(
    name="realized_vol_20",
    category="SR波动类",
    compute_func=_compute_realized_vol_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="20期收益波动",
    direction="negative",
    is_core=False,
)

register_factor(
    name="max_drawdown_20",
    category="SR波动类",
    compute_func=_compute_max_drawdown_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="20期最大回撤",
    direction="negative",
    is_core=False,
)


if __name__ == "__main__":
    import numpy as np

    n = 60
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
    result = compute_sr_volatility_factors(df)
    print(result.tail(5))
    print(f"\n列数: {len(result.columns)}, 预期: {len(_FACTOR_NAMES)}")
