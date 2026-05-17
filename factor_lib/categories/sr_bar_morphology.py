# -*- coding: utf-8 -*-
"""
factor_lib/categories/sr_bar_morphology.py - SR K线形态类因子

Purpose: SR K线形态类因子的批量计算与注册。
         基于 features.sr_event_factor_lab.compute_sr_factor_lab() 权威实现。

Public API:
    compute_sr_bar_morphology_factors(df) -> DataFrame

Registered Factors:
    - close_pos_in_bar: 收盘在K线中的位置
    - body_pct: 实体占比
    - upper_shadow_pct: 上影线比例
    - lower_shadow_pct: 下影线比例
    - is_bull_bar: 是否阳线
    - is_long_lower_shadow: 是否长下影
    - is_long_upper_shadow: 是否长上影
    - bar_range_atr: K线振幅/ATR
    - is_wide_range_bar: 是否宽幅K线
"""
from factor_lib.registry import register_factor
import pandas as pd


_FACTOR_NAMES = [
    "close_pos_in_bar",
    "body_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "is_bull_bar",
    "is_long_lower_shadow",
    "is_long_upper_shadow",
    "bar_range_atr",
    "is_wide_range_bar",
]


def compute_sr_bar_morphology_factors(df: pd.DataFrame) -> pd.DataFrame:
    from features.sr_event_factor_lab import compute_sr_factor_lab, LabConfig

    lab = compute_sr_factor_lab(df, LabConfig())
    return lab[_FACTOR_NAMES].copy()


def _compute_close_pos_in_bar(df):
    return compute_sr_bar_morphology_factors(df)["close_pos_in_bar"]


def _compute_body_pct(df):
    return compute_sr_bar_morphology_factors(df)["body_pct"]


def _compute_upper_shadow_pct(df):
    return compute_sr_bar_morphology_factors(df)["upper_shadow_pct"]


def _compute_lower_shadow_pct(df):
    return compute_sr_bar_morphology_factors(df)["lower_shadow_pct"]


def _compute_is_bull_bar(df):
    return compute_sr_bar_morphology_factors(df)["is_bull_bar"]


def _compute_is_long_lower_shadow(df):
    return compute_sr_bar_morphology_factors(df)["is_long_lower_shadow"]


def _compute_is_long_upper_shadow(df):
    return compute_sr_bar_morphology_factors(df)["is_long_upper_shadow"]


def _compute_bar_range_atr(df):
    return compute_sr_bar_morphology_factors(df)["bar_range_atr"]


def _compute_is_wide_range_bar(df):
    return compute_sr_bar_morphology_factors(df)["is_wide_range_bar"]


register_factor(
    name="close_pos_in_bar",
    category="SR K线形态类",
    compute_func=_compute_close_pos_in_bar,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="收盘在K线中的位置",
    direction="positive",
    is_core=True,
)

register_factor(
    name="body_pct",
    category="SR K线形态类",
    compute_func=_compute_body_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="实体占比",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="upper_shadow_pct",
    category="SR K线形态类",
    compute_func=_compute_upper_shadow_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="上影线比例",
    direction="negative",
    is_core=False,
)

register_factor(
    name="lower_shadow_pct",
    category="SR K线形态类",
    compute_func=_compute_lower_shadow_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="下影线比例",
    direction="positive",
    is_core=False,
)

register_factor(
    name="is_bull_bar",
    category="SR K线形态类",
    compute_func=_compute_is_bull_bar,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="是否阳线",
    direction="positive",
    is_core=False,
)

register_factor(
    name="is_long_lower_shadow",
    category="SR K线形态类",
    compute_func=_compute_is_long_lower_shadow,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="是否长下影",
    direction="positive",
    is_core=True,
)

register_factor(
    name="is_long_upper_shadow",
    category="SR K线形态类",
    compute_func=_compute_is_long_upper_shadow,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="是否长上影",
    direction="negative",
    is_core=False,
)

register_factor(
    name="bar_range_atr",
    category="SR K线形态类",
    compute_func=_compute_bar_range_atr,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="K线振幅/ATR",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="is_wide_range_bar",
    category="SR K线形态类",
    compute_func=_compute_is_wide_range_bar,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="是否宽幅K线",
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
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.01
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.01
    result = compute_sr_bar_morphology_factors(df)
    print(result.tail(5))
    print(f"\n列数: {len(result.columns)}, 预期: {len(_FACTOR_NAMES)}")
