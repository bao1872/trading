# -*- coding: utf-8 -*-
"""
factor_lib/ - 因子计算库（纯计算底座）

Purpose: 因子计算方法的权威出口。不写数据库，只提供批量计算函数供实验引入。

Usage:
    # 全部因子一键计算（推荐）
    from factor_lib import compute_all_factors_v2
    factors = compute_all_factors_v2(df_kline)

    # 按类别计算
    from factor_lib import compute_trend_factors, compute_risk_factors
    trend = compute_trend_factors(df_kline)
    risk = compute_risk_factors(df_kline)

    # 单个因子
    from factor_lib import get_factor, compute_panel_v2
    meta = get_factor("dsa_dir")
    selected = compute_panel_v2(df_kline, factor_names=["dsa_dir", "bbmacd"])

    # 诊断因子列表
    from factor_lib import list_all, list_by_category
    all_meta = list_all()
    trend_meta = list_by_category("趋势类")

Public API:
    compute_all_factors_v2(df, categories=None, factor_names=None, exclude=None) -> DataFrame
    compute_panel(df, ...)            # 旧版兼容（委托给 v2）
    compute_panel_v2(df, ...)         # 批量优化版

Category-level batch functions:
    compute_trend_factors(df, dsa_result=None, bb_result=None) -> DataFrame
    compute_position_factors(df, dsa_result=None) -> DataFrame
    compute_momentum_factors(df, bb_result=None) -> DataFrame
    compute_volume_factors(df) -> DataFrame
    compute_coordination_factors(df, dsa_result=None, bb_result=None) -> DataFrame
    compute_rhythm_factors(df, dsa_result=None) -> DataFrame
    compute_structure_factors(df) -> DataFrame
    compute_risk_factors(df) -> DataFrame
    compute_fundamental_factors(df) -> DataFrame
    compute_raw_features(df, dsa_result=None, bb_result=None) -> DataFrame
    compute_quantity_price_factors(df) -> DataFrame
"""
from factor_lib.registry import (
    FACTOR_REGISTRY,
    register_factor,
    list_all,
    list_by_category,
    get_factor,
    compute_panel,
    compute_panel_v2,
)

from factor_lib.categories.trend import compute_trend_factors
from factor_lib.categories.position import compute_position_factors
from factor_lib.categories.momentum import compute_momentum_factors
from factor_lib.categories.volume import compute_volume_factors
from factor_lib.categories.coordination import compute_coordination_factors
from factor_lib.categories.rhythm import compute_rhythm_factors
from factor_lib.categories.structure import compute_structure_factors
from factor_lib.categories.risk import compute_risk_factors
from factor_lib.categories.fundamental import compute_fundamental_factors
from factor_lib.categories.raw_features import compute_raw_features
from factor_lib.categories.quantity_price import compute_quantity_price_factors
from factor_lib.categories.stage_context import compute_stage_context_factors
from factor_lib.categories.stage_position import compute_stage_position_factors
from factor_lib.categories.stage_maturity import compute_stage_maturity_factors
from factor_lib.categories.sr_structure import compute_sr_structure_factors
from factor_lib.categories.sr_position import compute_sr_position_factors
from factor_lib.categories.sr_bar_morphology import compute_sr_bar_morphology_factors
from factor_lib.categories.sr_volume import compute_sr_volume_factors
from factor_lib.categories.sr_trend import compute_sr_trend_factors
from factor_lib.categories.sr_volatility import compute_sr_volatility_factors
from factor_lib.categories.sr_pierce_strength import compute_sr_pierce_strength_factors
from factor_lib.categories.sr_breakout_strength import compute_sr_breakout_strength_factors
from factor_lib.categories.sr_future_label import compute_sr_future_label_factors

compute_all_factors_v2 = compute_panel_v2

__all__ = [
    "FACTOR_REGISTRY",
    "register_factor",
    "list_all",
    "list_by_category",
    "get_factor",
    "compute_panel",
    "compute_panel_v2",
    "compute_all_factors_v2",
    "compute_trend_factors",
    "compute_position_factors",
    "compute_momentum_factors",
    "compute_volume_factors",
    "compute_coordination_factors",
    "compute_rhythm_factors",
    "compute_structure_factors",
    "compute_risk_factors",
    "compute_fundamental_factors",
    "compute_raw_features",
    "compute_quantity_price_factors",
    "compute_stage_context_factors",
    "compute_stage_position_factors",
    "compute_stage_maturity_factors",
    "compute_sr_structure_factors",
    "compute_sr_position_factors",
    "compute_sr_bar_morphology_factors",
    "compute_sr_volume_factors",
    "compute_sr_trend_factors",
    "compute_sr_volatility_factors",
    "compute_sr_pierce_strength_factors",
    "compute_sr_breakout_strength_factors",
    "compute_sr_future_label_factors",
]
