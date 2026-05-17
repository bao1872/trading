# -*- coding: utf-8 -*-
"""
factor_lib/categories/ - 因子分类批量计算模块

Usage:
    from factor_lib.categories import compute_all_category_factors

    # 单独计算某类因子
    from factor_lib.categories.trend import compute_trend_factors
    from factor_lib.categories.risk import compute_risk_factors
    ...

    # 全部因子（通过 compute_panel_v2 编排）
    from factor_lib import compute_all_factors_v2
    factors = compute_all_factors_v2(df)
"""

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
from factor_lib.categories.sr_position import compute_sr_position_factors
from factor_lib.categories.sr_structure import compute_sr_structure_factors
from factor_lib.categories.sr_bar_morphology import compute_sr_bar_morphology_factors
from factor_lib.categories.sr_volume import compute_sr_volume_factors
from factor_lib.categories.sr_trend import compute_sr_trend_factors
from factor_lib.categories.sr_volatility import compute_sr_volatility_factors
from factor_lib.categories.sr_pierce_strength import compute_sr_pierce_strength_factors
from factor_lib.categories.sr_breakout_strength import compute_sr_breakout_strength_factors
from factor_lib.categories.sr_future_label import compute_sr_future_label_factors

__all__ = [
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
    "compute_sr_position_factors",
    "compute_sr_structure_factors",
    "compute_sr_bar_morphology_factors",
    "compute_sr_volume_factors",
    "compute_sr_trend_factors",
    "compute_sr_volatility_factors",
    "compute_sr_pierce_strength_factors",
    "compute_sr_breakout_strength_factors",
    "compute_sr_future_label_factors",
]
