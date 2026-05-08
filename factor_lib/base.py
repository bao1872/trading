# -*- coding: utf-8 -*-
"""
factor_lib/base.py - 因子基类（可选面向对象封装）

Purpose: 提供 Factor 基类，用于需要面向对象封装的场景。
         简单场景可直接使用 register_factor() 函数式注册。

Usage:
    from factor_lib.base import Factor

    class MyFactor(Factor):
        name = "my_factor"
        category = "趋势类"
        description = "自定义因子"

        def compute(self, df: pd.DataFrame) -> pd.Series:
            return df["close"].rolling(20).mean()
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class Factor(ABC):
    """因子基类。子类只需实现 compute() 方法。"""

    name: str = ""
    category: str = ""
    description: str = ""
    direction: str = "neutral"
    is_core: bool = False
    params: Dict[str, Any] = {}
    source_module: str = ""
    source_function: str = ""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """计算因子值。子类必须实现。"""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于注册表。"""
        return {
            "name": self.name,
            "category": self.category,
            "compute": self.compute,
            "source_module": self.source_module,
            "source_function": self.source_function,
            "description": self.description,
            "direction": self.direction,
            "is_core": self.is_core,
            "params": self.params,
        }
