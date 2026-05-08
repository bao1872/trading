# -*- coding: utf-8 -*-
"""
event_lib/base.py - 事件基类（可选面向对象封装）

Purpose: 提供 Event 基类，用于需要面向对象封装的场景。
         简单场景可直接使用 register_event() 函数式注册。

Usage:
    from event_lib.base import Event

    class MyEvent(Event):
        name = "my_event"
        category = "趋势事件"
        description = "自定义事件"
        required_factors = ["dsa_dir"]

        def detect(self, factors_df: pd.DataFrame) -> pd.Series:
            return factors_df["dsa_dir"] == 1
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd


class Event(ABC):
    """事件基类。子类只需实现 detect() 方法。"""

    name: str = ""
    category: str = ""
    description: str = ""
    direction: str = "neutral"
    is_core: bool = False
    required_factors: List[str] = []
    outputs_strength: bool = False

    @abstractmethod
    def detect(self, factors_df: pd.DataFrame) -> pd.Series:
        """检测事件。子类必须实现。"""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于注册表。"""
        return {
            "name": self.name,
            "category": self.category,
            "detect": self.detect,
            "required_factors": self.required_factors,
            "description": self.description,
            "direction": self.direction,
            "is_core": self.is_core,
            "outputs_strength": self.outputs_strength,
        }
