# -*- coding: utf-8 -*-
"""
factor_lib/categories/sr_breakout_strength.py - SR突破强度类因子

Purpose: SR突破强度类因子的批量计算与注册。
         基于 features.sr_event_factor_lab.compute_sr_factor_lab() 权威实现。

Public API:
    compute_sr_breakout_strength_factors(df) -> DataFrame

Registered Factors:
    - resistance_break_strength_pct: 突破压力百分比幅度
    - resistance_break_strength_atr: 突破幅度/ATR
    - resistance_break_strength_sr: 突破幅度/区间宽度
    - high_break_strength_pct: 盘中突破压力幅度
    - breakout_body_pct: 突破K线实体占比
    - breakout_close_pos_in_bar: 突破K线收盘位置
"""
from factor_lib.registry import register_factor
import pandas as pd


_FACTOR_NAMES = [
    "resistance_break_strength_pct",
    "resistance_break_strength_atr",
    "resistance_break_strength_sr",
    "high_break_strength_pct",
    "breakout_body_pct",
    "breakout_close_pos_in_bar",
]


def compute_sr_breakout_strength_factors(df: pd.DataFrame) -> pd.DataFrame:
    from features.sr_event_factor_lab import compute_sr_factor_lab, LabConfig

    lab = compute_sr_factor_lab(df, LabConfig())
    return lab[_FACTOR_NAMES].copy()


def _compute_resistance_break_strength_pct(df):
    return compute_sr_breakout_strength_factors(df)["resistance_break_strength_pct"]


def _compute_resistance_break_strength_atr(df):
    return compute_sr_breakout_strength_factors(df)["resistance_break_strength_atr"]


def _compute_resistance_break_strength_sr(df):
    return compute_sr_breakout_strength_factors(df)["resistance_break_strength_sr"]


def _compute_high_break_strength_pct(df):
    return compute_sr_breakout_strength_factors(df)["high_break_strength_pct"]


def _compute_breakout_body_pct(df):
    return compute_sr_breakout_strength_factors(df)["breakout_body_pct"]


def _compute_breakout_close_pos_in_bar(df):
    return compute_sr_breakout_strength_factors(df)["breakout_close_pos_in_bar"]


register_factor(
    name="resistance_break_strength_pct",
    category="SR突破强度类",
    compute_func=_compute_resistance_break_strength_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="突破压力百分比幅度",
    direction="positive",
    is_core=True,
)

register_factor(
    name="resistance_break_strength_atr",
    category="SR突破强度类",
    compute_func=_compute_resistance_break_strength_atr,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="突破幅度/ATR",
    direction="positive",
    is_core=True,
)

register_factor(
    name="resistance_break_strength_sr",
    category="SR突破强度类",
    compute_func=_compute_resistance_break_strength_sr,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="突破幅度/区间宽度",
    direction="positive",
    is_core=False,
)

register_factor(
    name="high_break_strength_pct",
    category="SR突破强度类",
    compute_func=_compute_high_break_strength_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="盘中突破压力幅度",
    direction="positive",
    is_core=False,
)

register_factor(
    name="breakout_body_pct",
    category="SR突破强度类",
    compute_func=_compute_breakout_body_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="突破K线实体占比",
    direction="positive",
    is_core=False,
)

register_factor(
    name="breakout_close_pos_in_bar",
    category="SR突破强度类",
    compute_func=_compute_breakout_close_pos_in_bar,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="突破K线收盘位置",
    direction="positive",
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
    result = compute_sr_breakout_strength_factors(df)
    print(result.tail(5))
    print(f"\n列数: {len(result.columns)}, 预期: {len(_FACTOR_NAMES)}")
