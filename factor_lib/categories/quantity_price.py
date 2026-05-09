# -*- coding: utf-8 -*-
"""
factor_lib/categories/quantity_price.py - 量价类因子（VSA Effort vs Result）

Purpose: VSA IQ 量价分析因子的批量计算与注册。
         基于 features/vsa_iq_er_viewer.py 的 compute_vsa_iq 权威实现。

Public API:
    compute_quantity_price_factors(df) -> DataFrame

Registered Factors:
    - vsa_er_factor: 量价努力-结果差值（Effort Rank - Result Rank）
    - vsa_er_factor_ma: ER 因子的 MA 平滑值
    - vsa_effort_rank: 量能努力等级 1-10
    - vsa_result_rank: 价差结果等级 1-10
    - vsa_vol_rank: 成交量滚动百分位排名 0-100
    - vsa_spread_rank: 价差滚动百分位排名 0-100
    - vsa_net_score: 多空背景净得分
    - vsa_strength_score: 多头背景得分
    - vsa_weakness_score: 空头背景得分
    - vsa_bull_score: 单根 K 线多头信号得分 (0/1/2/3)
    - vsa_bear_score: 单根 K 线空头信号得分 (0/1/2/3)
    - vsa_strong_move_z: 强移动 Z-score

Rules:
    - 本文件不实现任何计算公式，只做薄封装
    - 权威实现位于 features/vsa_iq_er_viewer.py 的 compute_vsa_iq 函数
"""
from factor_lib.registry import register_factor
import pandas as pd

_VSA_DEFAULT_PARAMS = dict(
    ma_length=14,
    ma_type="SMA",
    bg_lookback=10,
    trend_rule=True,
)


def compute_quantity_price_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    批量计算全部量价类因子。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame

    Returns:
        DataFrame 含 12 列量价因子
    """
    from features.vsa_iq_er_viewer import compute_vsa_iq

    if df.empty:
        return pd.DataFrame(index=df.index)

    vsa_result = compute_vsa_iq(
        df,
        ma_length=_VSA_DEFAULT_PARAMS["ma_length"],
        ma_type=_VSA_DEFAULT_PARAMS["ma_type"],
        bg_lookback=_VSA_DEFAULT_PARAMS["bg_lookback"],
        trend_rule=_VSA_DEFAULT_PARAMS["trend_rule"],
    )

    result = pd.DataFrame(index=df.index)
    result["vsa_er_factor"] = vsa_result["er_factor"]
    result["vsa_er_factor_ma"] = vsa_result["er_factor_ma"]
    result["vsa_effort_rank"] = vsa_result["effort_rank_easy"]
    result["vsa_result_rank"] = vsa_result["result_rank_easy"]
    result["vsa_vol_rank"] = vsa_result["vol_rank"]
    result["vsa_spread_rank"] = vsa_result["spread_rank"]
    result["vsa_net_score"] = vsa_result["net_score"]
    result["vsa_strength_score"] = vsa_result["strength_score"]
    result["vsa_weakness_score"] = vsa_result["weakness_score"]
    result["vsa_bull_score"] = vsa_result["bull_score"]
    result["vsa_bear_score"] = vsa_result["bear_score"]
    result["vsa_strong_move_z"] = vsa_result["strong_move_z"]
    return result


def _compute_vsa_er_factor(df):
    return compute_quantity_price_factors(df)["vsa_er_factor"]


def _compute_vsa_er_factor_ma(df):
    return compute_quantity_price_factors(df)["vsa_er_factor_ma"]


def _compute_vsa_effort_rank(df):
    return compute_quantity_price_factors(df)["vsa_effort_rank"]


def _compute_vsa_result_rank(df):
    return compute_quantity_price_factors(df)["vsa_result_rank"]


def _compute_vsa_vol_rank(df):
    return compute_quantity_price_factors(df)["vsa_vol_rank"]


def _compute_vsa_spread_rank(df):
    return compute_quantity_price_factors(df)["vsa_spread_rank"]


def _compute_vsa_net_score(df):
    return compute_quantity_price_factors(df)["vsa_net_score"]


def _compute_vsa_strength_score(df):
    return compute_quantity_price_factors(df)["vsa_strength_score"]


def _compute_vsa_weakness_score(df):
    return compute_quantity_price_factors(df)["vsa_weakness_score"]


def _compute_vsa_bull_score(df):
    return compute_quantity_price_factors(df)["vsa_bull_score"]


def _compute_vsa_bear_score(df):
    return compute_quantity_price_factors(df)["vsa_bear_score"]


def _compute_vsa_strong_move_z(df):
    return compute_quantity_price_factors(df)["vsa_strong_move_z"]


register_factor(
    name="vsa_er_factor",
    category="量价类",
    compute_func=_compute_vsa_er_factor,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="量价努力-结果差值（Effort Rank - Result Rank），VSA 核心 ER 因子，正值表示量能努力 > 价差结果",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="vsa_er_factor_ma",
    category="量价类",
    compute_func=_compute_vsa_er_factor_ma,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="ER 因子的 MA 平滑值（默认 SMA14）",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="vsa_effort_rank",
    category="量价类",
    compute_func=_compute_vsa_effort_rank,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="量能努力等级 1-10，基于成交量滚动百分位分档",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="vsa_result_rank",
    category="量价类",
    compute_func=_compute_vsa_result_rank,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="价差结果等级 1-10，基于价差滚动百分位分档",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="vsa_vol_rank",
    category="量价类",
    compute_func=_compute_vsa_vol_rank,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="成交量滚动百分位排名 0-100（窗口 100）",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="vsa_spread_rank",
    category="量价类",
    compute_func=_compute_vsa_spread_rank,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="价差滚动百分位排名 0-100（窗口 100）",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="vsa_net_score",
    category="量价类",
    compute_func=_compute_vsa_net_score,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="多空背景净得分（strength_score - weakness_score），正值偏多",
    direction="positive",
    is_core=False,
)

register_factor(
    name="vsa_strength_score",
    category="量价类",
    compute_func=_compute_vsa_strength_score,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="多头背景得分（bull_score 滚动求和 + 趋势加权）",
    direction="positive",
    is_core=False,
)

register_factor(
    name="vsa_weakness_score",
    category="量价类",
    compute_func=_compute_vsa_weakness_score,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="空头背景得分（bear_score 滚动求和 + 趋势加权）",
    direction="negative",
    is_core=False,
)

register_factor(
    name="vsa_bull_score",
    category="量价类",
    compute_func=_compute_vsa_bull_score,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="单根 K 线多头信号得分（Bullish Efficiency=3, Constructive Upward=2, Easy Upward=1）",
    direction="positive",
    is_core=False,
)

register_factor(
    name="vsa_bear_score",
    category="量价类",
    compute_func=_compute_vsa_bear_score,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="单根 K 线空头信号得分（Bearish Efficiency=3, Constructive Downward=2, Easy Downward=1）",
    direction="negative",
    is_core=False,
)

register_factor(
    name="vsa_strong_move_z",
    category="量价类",
    compute_func=_compute_vsa_strong_move_z,
    source_module="features.vsa_iq_er_viewer",
    source_function="compute_vsa_iq",
    description="强移动 Z-score，3日对数收益标准化后的偏离度",
    direction="neutral",
    is_core=False,
)