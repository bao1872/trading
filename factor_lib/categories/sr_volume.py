# -*- coding: utf-8 -*-
"""
factor_lib/categories/sr_volume.py - SR量能类因子

Purpose: SR量能类因子的批量计算与注册。
         基于 features.sr_event_factor_lab.compute_sr_factor_lab() 权威实现。

Public API:
    compute_sr_volume_factors(df) -> DataFrame

Registered Factors:
    - volume_ratio_20: 成交量/20期均量
    - volume_z_20: 20期成交量Z-score
    - amount_ratio_20: 成交额/20期均额
    - amount_z_20: 20期成交额Z-score
    - is_volume_expansion: 是否放量
    - is_volume_shrink: 是否缩量
"""
from factor_lib.registry import register_factor
import pandas as pd


_FACTOR_NAMES = [
    "volume_ratio_20",
    "volume_z_20",
    "amount_ratio_20",
    "amount_z_20",
    "is_volume_expansion",
    "is_volume_shrink",
]


def compute_sr_volume_factors(df: pd.DataFrame) -> pd.DataFrame:
    from features.sr_event_factor_lab import compute_sr_factor_lab, LabConfig

    lab = compute_sr_factor_lab(df, LabConfig())
    return lab[_FACTOR_NAMES].copy()


def _compute_volume_ratio_20(df):
    return compute_sr_volume_factors(df)["volume_ratio_20"]


def _compute_volume_z_20(df):
    return compute_sr_volume_factors(df)["volume_z_20"]


def _compute_amount_ratio_20(df):
    return compute_sr_volume_factors(df)["amount_ratio_20"]


def _compute_amount_z_20(df):
    return compute_sr_volume_factors(df)["amount_z_20"]


def _compute_is_volume_expansion(df):
    return compute_sr_volume_factors(df)["is_volume_expansion"]


def _compute_is_volume_shrink(df):
    return compute_sr_volume_factors(df)["is_volume_shrink"]


register_factor(
    name="volume_ratio_20",
    category="SR量能类",
    compute_func=_compute_volume_ratio_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="成交量/20期均量",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="volume_z_20",
    category="SR量能类",
    compute_func=_compute_volume_z_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="20期成交量Z-score",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="amount_ratio_20",
    category="SR量能类",
    compute_func=_compute_amount_ratio_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="成交额/20期均额",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="amount_z_20",
    category="SR量能类",
    compute_func=_compute_amount_z_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="20期成交额Z-score",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="is_volume_expansion",
    category="SR量能类",
    compute_func=_compute_is_volume_expansion,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="是否放量",
    direction="positive",
    is_core=True,
)

register_factor(
    name="is_volume_shrink",
    category="SR量能类",
    compute_func=_compute_is_volume_shrink,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="是否缩量",
    direction="neutral",
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
            "amount": np.random.randint(10000, 50000, n).astype(float),
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.01
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.01
    result = compute_sr_volume_factors(df)
    print(result.tail(5))
    print(f"\n列数: {len(result.columns)}, 预期: {len(_FACTOR_NAMES)}")
