# -*- coding: utf-8 -*-
"""
factor_lib/categories/sr_position.py - SR位置类因子

Purpose: SR位置类因子（支撑/压力区间位置/距离/空间）的批量计算与注册。
         基于 features.sr_event_factor_lab.compute_sr_factor_lab 权威实现。

Public API:
    compute_sr_position_factors(df, pivot_len=10) -> DataFrame

Registered Factors:
    - sr_pos_01: SR区间位置(0-1)
    - sr_pos_raw: SR区间原始位置(可越界)
    - sr_pos_open: 开盘SR位置(0-1)
    - sr_pos_high: 最高SR位置(0-1)
    - sr_pos_low: 最低SR位置(0-1)
    - sr_range_pct: SR区间宽度百分比
    - close_to_support_pct: 收盘距支撑百分比
    - close_to_resistance_pct: 收盘距压力百分比
    - upside_to_resistance_pct: 到压力的上涨空间
    - downside_to_support_pct: 到支撑的下跌空间
"""
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np

_SR_FACTOR_COLUMNS = [
    "sr_pos_01",
    "sr_pos_raw",
    "sr_pos_open",
    "sr_pos_high",
    "sr_pos_low",
    "sr_range_pct",
    "close_to_support_pct",
    "close_to_resistance_pct",
    "upside_to_resistance_pct",
    "downside_to_support_pct",
]


def compute_sr_position_factors(
    df: pd.DataFrame,
    pivot_len: int = 10,
) -> pd.DataFrame:
    from features.sr_event_factor_lab import compute_sr_factor_lab, LabConfig

    cfg = LabConfig(pivot_len=pivot_len)
    sr_result = compute_sr_factor_lab(df, cfg)
    result = pd.DataFrame(index=df.index)
    for col in _SR_FACTOR_COLUMNS:
        result[col] = sr_result[col]
    return result


def _compute_sr_pos_01(df):
    return compute_sr_position_factors(df)["sr_pos_01"]


def _compute_sr_pos_raw(df):
    return compute_sr_position_factors(df)["sr_pos_raw"]


def _compute_sr_pos_open(df):
    return compute_sr_position_factors(df)["sr_pos_open"]


def _compute_sr_pos_high(df):
    return compute_sr_position_factors(df)["sr_pos_high"]


def _compute_sr_pos_low(df):
    return compute_sr_position_factors(df)["sr_pos_low"]


def _compute_sr_range_pct(df):
    return compute_sr_position_factors(df)["sr_range_pct"]


def _compute_close_to_support_pct(df):
    return compute_sr_position_factors(df)["close_to_support_pct"]


def _compute_close_to_resistance_pct(df):
    return compute_sr_position_factors(df)["close_to_resistance_pct"]


def _compute_upside_to_resistance_pct(df):
    return compute_sr_position_factors(df)["upside_to_resistance_pct"]


def _compute_downside_to_support_pct(df):
    return compute_sr_position_factors(df)["downside_to_support_pct"]


register_factor(
    name="sr_pos_01",
    category="SR位置类",
    compute_func=_compute_sr_pos_01,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="SR区间位置(0-1)，0=支撑位，1=压力位",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="sr_pos_raw",
    category="SR位置类",
    compute_func=_compute_sr_pos_raw,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="SR区间原始位置(可越界)，未裁剪到0-1",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="sr_pos_open",
    category="SR位置类",
    compute_func=_compute_sr_pos_open,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="开盘SR位置(0-1)",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="sr_pos_high",
    category="SR位置类",
    compute_func=_compute_sr_pos_high,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="最高SR位置(0-1)",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="sr_pos_low",
    category="SR位置类",
    compute_func=_compute_sr_pos_low,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="最低SR位置(0-1)",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="sr_range_pct",
    category="SR位置类",
    compute_func=_compute_sr_range_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="SR区间宽度百分比(压力-支撑)/支撑",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="close_to_support_pct",
    category="SR位置类",
    compute_func=_compute_close_to_support_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="收盘距支撑百分比(收盘-支撑)/支撑",
    direction="positive",
    is_core=False,
)

register_factor(
    name="close_to_resistance_pct",
    category="SR位置类",
    compute_func=_compute_close_to_resistance_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="收盘距压力百分比(收盘-压力)/压力",
    direction="negative",
    is_core=False,
)

register_factor(
    name="upside_to_resistance_pct",
    category="SR位置类",
    compute_func=_compute_upside_to_resistance_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="到压力的上涨空间(压力-收盘)/收盘",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="downside_to_support_pct",
    category="SR位置类",
    compute_func=_compute_downside_to_support_pct,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="到支撑的下跌空间(收盘-支撑)/收盘",
    direction="neutral",
    is_core=False,
)


if __name__ == "__main__":
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.3)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    open_ = low + np.random.rand(n) * (high - low)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": np.random.randint(1000, 10000, n)},
        index=dates,
    )
    result = compute_sr_position_factors(df)
    print(result.dropna().head(10).to_string())
    print(f"\n列数: {len(result.columns)}, 期望: {len(_SR_FACTOR_COLUMNS)}")
    assert list(result.columns) == _SR_FACTOR_COLUMNS, f"列名不匹配: {list(result.columns)}"
    print("自测通过")
