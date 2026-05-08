# -*- coding: utf-8 -*-
"""
event_lib/detectors/fundamental_events.py - 基本面事件检测

Purpose: 基于财务类因子列检测基本面相关事件。

Registered Events:
    - evt_earnings_acceleration: 业绩加速（净利润同比增长率上升）
    - evt_earnings_deceleration: 业绩减速（净利润同比增长率下降）
    - evt_cashflow_improvement: 现金流改善
    - evt_cashflow_deterioration: 现金流恶化
    - evt_roe_inflection: ROE拐点

Note: 基本面事件需要财务数据，如果 factors_df 中缺少财务列，返回全0。
"""
from event_lib.registry import register_event
import pandas as pd


def _detect_earnings_acceleration(factors_df: pd.DataFrame) -> pd.Series:
    """业绩加速：q_np_yoy 上升。"""
    if "q_np_yoy" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["q_np_yoy"].diff() > 0).astype(int)


def _detect_earnings_deceleration(factors_df: pd.DataFrame) -> pd.Series:
    """业绩减速：q_np_yoy 下降。"""
    if "q_np_yoy" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["q_np_yoy"].diff() < 0).astype(int)


def _detect_cashflow_improvement(factors_df: pd.DataFrame) -> pd.Series:
    """现金流改善：cfo_to_np_parent 上升。"""
    if "cfo_to_np_parent" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["cfo_to_np_parent"].diff() > 0).astype(int)


def _detect_cashflow_deterioration(factors_df: pd.DataFrame) -> pd.Series:
    """现金流恶化：cfo_to_np_parent 下降。"""
    if "cfo_to_np_parent" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    return (factors_df["cfo_to_np_parent"].diff() < 0).astype(int)


def _detect_roe_inflection(factors_df: pd.DataFrame) -> pd.Series:
    """ROE拐点：roe_weighted 方向改变。"""
    if "roe_weighted" not in factors_df.columns:
        return pd.Series(0, index=factors_df.index)
    roe = factors_df["roe_weighted"]
    return ((roe.diff() > 0) & (roe.diff().shift(1) <= 0)).astype(int)


# 注册基本面事件
register_event(
    name="evt_earnings_acceleration",
    category="基本面事件",
    detect_func=_detect_earnings_acceleration,
    required_factors=["q_np_yoy"],
    description="业绩加速（净利润同比增长率上升）",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_earnings_deceleration",
    category="基本面事件",
    detect_func=_detect_earnings_deceleration,
    required_factors=["q_np_yoy"],
    description="业绩减速（净利润同比增长率下降）",
    direction="negative",
    is_core=False,
)

register_event(
    name="evt_cashflow_improvement",
    category="基本面事件",
    detect_func=_detect_cashflow_improvement,
    required_factors=["cfo_to_np_parent"],
    description="现金流改善",
    direction="positive",
    is_core=False,
)

register_event(
    name="evt_cashflow_deterioration",
    category="基本面事件",
    detect_func=_detect_cashflow_deterioration,
    required_factors=["cfo_to_np_parent"],
    description="现金流恶化",
    direction="negative",
    is_core=False,
)

register_event(
    name="evt_roe_inflection",
    category="基本面事件",
    detect_func=_detect_roe_inflection,
    required_factors=["roe_weighted"],
    description="ROE拐点",
    direction="positive",
    is_core=False,
)
