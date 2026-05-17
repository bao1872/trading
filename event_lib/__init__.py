# -*- coding: utf-8 -*-
"""
event_lib - 事件库

Purpose: 事件注册表 + 统一检测入口。检测逻辑基于因子列触发，不重算因子。

Usage:
    from event_lib import list_all, detect_panel, get_event
    events = detect_panel(factors_df, categories=['趋势事件', '量能事件'])
"""

from event_lib.registry import (
    EVENT_REGISTRY,
    register_event,
    list_all,
    list_by_category,
    get_event,
    detect_panel,
)

# 自动导入并注册所有事件检测器（触发注册）
from event_lib.detectors import (
    trend_events,
    breakout_events,
    volume_events,
    momentum_events,
    structural_events,
    fundamental_events,
    composite_events,
    stage_cost_zone_events,
    stage_wash_events,
    stage_shake_events,
    stage_repair_events,
    stage_failure_events,
    sr_support_events,
    sr_resistance_events,
)

__all__ = [
    "EVENT_REGISTRY",
    "register_event",
    "list_all",
    "list_by_category",
    "get_event",
    "detect_panel",
]
