# -*- coding: utf-8 -*-
"""
factor_lib/categories/sr_future_label.py - SR未来标签类因子

Purpose: SR未来标签类因子的批量计算与注册。
         基于 features.sr_event_factor_lab.compute_sr_factor_lab() 权威实现。

Public API:
    compute_sr_future_label_factors(df) -> DataFrame

Registered Factors:
    - fwd_ret_1: 未来1根收益
    - fwd_ret_3: 未来3根收益
    - fwd_ret_5: 未来5根收益
    - fwd_ret_10: 未来10根收益
    - fwd_ret_20: 未来20根收益
    - fwd_max_ret_5: 未来5根最大上涨
    - fwd_max_ret_10: 未来10根最大上涨
    - fwd_max_ret_20: 未来20根最大上涨
    - fwd_mdd_5: 未来5根最大回撤
    - fwd_mdd_10: 未来10根最大回撤
    - fwd_mdd_20: 未来20根最大回撤
    - fwd_reward_risk_5: 未来5根收益回撤比
    - fwd_reward_risk_10: 未来10根收益回撤比
    - fwd_reward_risk_20: 未来20根收益回撤比
"""
from factor_lib.registry import register_factor
import pandas as pd


_FACTOR_NAMES = [
    "fwd_ret_1",
    "fwd_ret_3",
    "fwd_ret_5",
    "fwd_ret_10",
    "fwd_ret_20",
    "fwd_max_ret_5",
    "fwd_max_ret_10",
    "fwd_max_ret_20",
    "fwd_mdd_5",
    "fwd_mdd_10",
    "fwd_mdd_20",
    "fwd_reward_risk_5",
    "fwd_reward_risk_10",
    "fwd_reward_risk_20",
]


def compute_sr_future_label_factors(df: pd.DataFrame) -> pd.DataFrame:
    from features.sr_event_factor_lab import compute_sr_factor_lab, LabConfig

    lab = compute_sr_factor_lab(df, LabConfig())
    return lab[_FACTOR_NAMES].copy()


def _compute_fwd_ret_1(df):
    return compute_sr_future_label_factors(df)["fwd_ret_1"]


def _compute_fwd_ret_3(df):
    return compute_sr_future_label_factors(df)["fwd_ret_3"]


def _compute_fwd_ret_5(df):
    return compute_sr_future_label_factors(df)["fwd_ret_5"]


def _compute_fwd_ret_10(df):
    return compute_sr_future_label_factors(df)["fwd_ret_10"]


def _compute_fwd_ret_20(df):
    return compute_sr_future_label_factors(df)["fwd_ret_20"]


def _compute_fwd_max_ret_5(df):
    return compute_sr_future_label_factors(df)["fwd_max_ret_5"]


def _compute_fwd_max_ret_10(df):
    return compute_sr_future_label_factors(df)["fwd_max_ret_10"]


def _compute_fwd_max_ret_20(df):
    return compute_sr_future_label_factors(df)["fwd_max_ret_20"]


def _compute_fwd_mdd_5(df):
    return compute_sr_future_label_factors(df)["fwd_mdd_5"]


def _compute_fwd_mdd_10(df):
    return compute_sr_future_label_factors(df)["fwd_mdd_10"]


def _compute_fwd_mdd_20(df):
    return compute_sr_future_label_factors(df)["fwd_mdd_20"]


def _compute_fwd_reward_risk_5(df):
    return compute_sr_future_label_factors(df)["fwd_reward_risk_5"]


def _compute_fwd_reward_risk_10(df):
    return compute_sr_future_label_factors(df)["fwd_reward_risk_10"]


def _compute_fwd_reward_risk_20(df):
    return compute_sr_future_label_factors(df)["fwd_reward_risk_20"]


register_factor(
    name="fwd_ret_1",
    category="SR未来标签类",
    compute_func=_compute_fwd_ret_1,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来1根收益",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="fwd_ret_3",
    category="SR未来标签类",
    compute_func=_compute_fwd_ret_3,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来3根收益",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="fwd_ret_5",
    category="SR未来标签类",
    compute_func=_compute_fwd_ret_5,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来5根收益",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="fwd_ret_10",
    category="SR未来标签类",
    compute_func=_compute_fwd_ret_10,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来10根收益",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="fwd_ret_20",
    category="SR未来标签类",
    compute_func=_compute_fwd_ret_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来20根收益",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="fwd_max_ret_5",
    category="SR未来标签类",
    compute_func=_compute_fwd_max_ret_5,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来5根最大上涨",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="fwd_max_ret_10",
    category="SR未来标签类",
    compute_func=_compute_fwd_max_ret_10,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来10根最大上涨",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="fwd_max_ret_20",
    category="SR未来标签类",
    compute_func=_compute_fwd_max_ret_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来20根最大上涨",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="fwd_mdd_5",
    category="SR未来标签类",
    compute_func=_compute_fwd_mdd_5,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来5根最大回撤",
    direction="negative",
    is_core=False,
)

register_factor(
    name="fwd_mdd_10",
    category="SR未来标签类",
    compute_func=_compute_fwd_mdd_10,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来10根最大回撤",
    direction="negative",
    is_core=False,
)

register_factor(
    name="fwd_mdd_20",
    category="SR未来标签类",
    compute_func=_compute_fwd_mdd_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来20根最大回撤",
    direction="negative",
    is_core=False,
)

register_factor(
    name="fwd_reward_risk_5",
    category="SR未来标签类",
    compute_func=_compute_fwd_reward_risk_5,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来5根收益回撤比",
    direction="positive",
    is_core=False,
)

register_factor(
    name="fwd_reward_risk_10",
    category="SR未来标签类",
    compute_func=_compute_fwd_reward_risk_10,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来10根收益回撤比",
    direction="positive",
    is_core=False,
)

register_factor(
    name="fwd_reward_risk_20",
    category="SR未来标签类",
    compute_func=_compute_fwd_reward_risk_20,
    source_module="features.sr_event_factor_lab",
    source_function="compute_sr_factor_lab",
    description="未来20根收益回撤比",
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
    result = compute_sr_future_label_factors(df)
    print(result.tail(5))
    print(f"\n列数: {len(result.columns)}, 预期: {len(_FACTOR_NAMES)}")
