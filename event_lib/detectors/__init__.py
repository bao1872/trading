# -*- coding: utf-8 -*-
"""
event_lib/detectors/ - 事件检测器

Purpose: 每个文件负责一类事件的检测逻辑，基于标准化因子列触发。

Categories:
    - trend_events.py        趋势事件
    - breakout_events.py     突破事件
    - volume_events.py       量能事件
    - momentum_events.py     动量事件
    - structural_events.py   结构事件
    - fundamental_events.py  基本面事件
    - composite_events.py    复合事件

Usage:
    import event_lib.detectors  # 自动注册所有事件
"""

# 自动导入并注册所有检测器
from event_lib.detectors import (
    trend_events,
    breakout_events,
    volume_events,
    momentum_events,
    structural_events,
    fundamental_events,
    composite_events,
)
