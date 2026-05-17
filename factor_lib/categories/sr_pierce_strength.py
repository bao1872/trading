# -*- coding: utf-8 -*-
"""
factor_lib/categories/sr_pierce_strength.py - SR刺破强度类因子

Purpose: SR刺破强度类因子（含R2S刺破）的批量计算与注册。
         基于 features.sr_event_factor_lab.compute_sr_factor_lab() 权威实现。

Public API:
    compute_sr_pierce_strength_factors(df) -> DataFrame

Registered Factors:
    - support_pierce_depth_pct: 跌破支撑百分比深度
    - support_pierce_depth_atr: 跌破深度/ATR
    - support_pierce_depth_sr: 跌破深度/区间宽度
    - support_reclaim_strength_pct: 收回强度百分比
    - support_reclaim_strength_atr: 收回强度/ATR
    - support_reclaim_strength_sr: 收回强度/区间宽度
    - is_shallow_support_pierce: 是否浅刺破
    - is_mid_support_pierce: 是否中刺破
    - is_deep_support_pierce: 是否深刺破
    - pivot_support_pierce_depth_atr: 刺破pivot支撑深度/ATR
    - pivot_support_reclaim_strength_atr: pivot支撑收回强度/ATR
    - flipped_support_pierce_depth_atr: 刺破压力转支撑深度/ATR
    - flipped_support_reclaim_strength_atr: 压力转支撑收回强度/ATR
"""
from factor_lib.registry import register_factor
import pandas as pd


_FACTOR_NAMES = [
    "support_pierce_depth_pct",
    "support_pierce_depth_atr",
    "support_pierce_depth_sr",
    "support_reclaim_strength_pct",
    "support_reclaim_strength_atr",
    "support_reclaim_strength_sr",
    "is_shallow_support_pierce",
    "is_mid_support_pierce",
    "is_deep_support_pierce",
    "pivot_support_pierce_depth_atr",
    "pivot_support_reclaim_strength_atr",
    "flipped_support_pierce_depth_atr",
    "flipped_support_reclaim_strength_atr",
]


def compute_sr_pierce_strength_factors(df: pd.DataFrame) -> pd.DataFrame:
    from features.sr_event_factor_lab import compute_sr_factor_lab, LabConfig

    lab = compute_sr_factor_lab(df, LabConfig())
    return lab[_FACTOR_NAMES].copy()


def _compute_support_pierce_depth_pct(df):
    return compute_sr_pierce_strength_factors(df)["support_pierce_depth_pct"]


def _compute_support_pierce_depth_atr(df):
    return compute_sr_pierce_strength_factors(df)["support_pierce_depth_atr"]


def _compute_support_pierce_depth_sr(df):
    return compute_sr_pierce_strength_factors(df)["support_pierce_depth_sr"]


def _compute_support_reclaim_strength_pct(df):
    return compute_sr_pierce_strength_factors(df)["support_reclaim_strength_pct"]


def _compute_support_reclaim_strength_atr(df):
    return compute_sr_pierce_strength_factors(df)["support_reclaim_strength_atr"]


def _compute_support_reclaim_strength_sr(df):
    return compute_sr_pierce_strength_factors(df)["support_reclaim_strength_sr"]


def _compute_is_shallow_support_pierce(df):
    return compute_sr_pierce_strength_factors(df)["is_shallow_support_pierce"]


def _compute_is_mid_support_pierce(df):
    return compute_sr_pierce_strength_factors(df)["is_mid_support_pierce"]


def _compute_is_deep_support_pierce(df):
    return compute_sr_pierce_strength_factors(df)["is_deep_support_pierce"]


def _compute_pivot_support_pierce_depth_atr(df):
    return compute_sr_pierce_strength_factors(df)["pivot_support_pierce_depth_atr"]


def _compute_pivot_support_reclaim_strength_atr(df):
    return compute_sr_pierce_strength_factors(df)["pivot_support_reclaim_strength_atr"]


def _compute_flipped_support_pierce_depth_atr(df):
    return compute_sr_pierce_strength_factors(df)["flipped_support_pierce_depth_atr"]


def _compute_flipped_support_reclaim_strength_atr(df):
    return compute_sr_pierce_strength_factors(df)["flipped_support_reclaim_strength_atr"]


register_factor(
    name="support_pierce_depth_pct",
    category="SR刺破强度类",
    compute_func=_compute_support_pierce_depth_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="跌破支撑百分比深度",
    direction="negative",
    is_core=True,
)

register_factor(
    name="support_pierce_depth_atr",
    category="SR刺破强度类",
    compute_func=_compute_support_pierce_depth_atr,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="跌破深度/ATR",
    direction="negative",
    is_core=True,
)

register_factor(
    name="support_pierce_depth_sr",
    category="SR刺破强度类",
    compute_func=_compute_support_pierce_depth_sr,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="跌破深度/区间宽度",
    direction="negative",
    is_core=False,
)

register_factor(
    name="support_reclaim_strength_pct",
    category="SR刺破强度类",
    compute_func=_compute_support_reclaim_strength_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="收回强度百分比",
    direction="positive",
    is_core=True,
)

register_factor(
    name="support_reclaim_strength_atr",
    category="SR刺破强度类",
    compute_func=_compute_support_reclaim_strength_atr,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="收回强度/ATR",
    direction="positive",
    is_core=True,
)

register_factor(
    name="support_reclaim_strength_sr",
    category="SR刺破强度类",
    compute_func=_compute_support_reclaim_strength_sr,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="收回强度/区间宽度",
    direction="positive",
    is_core=False,
)

register_factor(
    name="is_shallow_support_pierce",
    category="SR刺破强度类",
    compute_func=_compute_is_shallow_support_pierce,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="是否浅刺破",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="is_mid_support_pierce",
    category="SR刺破强度类",
    compute_func=_compute_is_mid_support_pierce,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="是否中刺破",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="is_deep_support_pierce",
    category="SR刺破强度类",
    compute_func=_compute_is_deep_support_pierce,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="是否深刺破",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="pivot_support_pierce_depth_atr",
    category="SR刺破强度类",
    compute_func=_compute_pivot_support_pierce_depth_atr,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="刺破pivot支撑深度/ATR",
    direction="negative",
    is_core=False,
)

register_factor(
    name="pivot_support_reclaim_strength_atr",
    category="SR刺破强度类",
    compute_func=_compute_pivot_support_reclaim_strength_atr,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="pivot支撑收回强度/ATR",
    direction="positive",
    is_core=False,
)

register_factor(
    name="flipped_support_pierce_depth_atr",
    category="SR刺破强度类",
    compute_func=_compute_flipped_support_pierce_depth_atr,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="刺破压力转支撑深度/ATR",
    direction="negative",
    is_core=True,
)

register_factor(
    name="flipped_support_reclaim_strength_atr",
    category="SR刺破强度类",
    compute_func=_compute_flipped_support_reclaim_strength_atr,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="压力转支撑收回强度/ATR",
    direction="positive",
    is_core=True,
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
    result = compute_sr_pierce_strength_factors(df)
    print(result.tail(5))
    print(f"\n列数: {len(result.columns)}, 预期: {len(_FACTOR_NAMES)}")
