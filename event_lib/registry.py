# -*- coding: utf-8 -*-
"""
event_lib/registry.py - 事件注册表

Purpose: 全局事件注册中心。新增事件只需调用 register_event()，无需修改现有代码。

Rules:
    - 本文件只存元数据和检测逻辑，不重新计算因子。
    - 事件检测只能基于标准化因子列触发（接收 factors_df）。
    - 禁止在本文件中写因子计算公式。

Usage:
    from event_lib.registry import register_event, list_all, detect_panel
"""
from typing import Callable, Dict, Any, List, Optional
import pandas as pd

EVENT_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_event(
    name: str,
    category: str,
    detect_func: Callable[[pd.DataFrame], pd.Series],
    required_factors: List[str],
    description: str,
    direction: str = "neutral",
    is_core: bool = False,
    outputs_strength: bool = False,
) -> None:
    """
    注册一个事件到全局注册表。

    Args:
        name: 事件列名（唯一标识，建议前缀 evt_）
        category: 事件类别，如'趋势事件'、'量能事件'等
        detect_func: 检测函数，接收 factors_df 返回 Series（0/1 或布尔值）
        required_factors: 依赖的因子列名列表
        description: 事件描述
        direction: 事件方向，'positive'/'negative'/'neutral'
        is_core: 是否核心事件
        outputs_strength: 是否输出强度列（evt_*_strength）
    """
    if name in EVENT_REGISTRY:
        raise ValueError(f"事件 '{name}' 已注册")

    EVENT_REGISTRY[name] = {
        "name": name,
        "category": category,
        "detect": detect_func,
        "required_factors": required_factors,
        "description": description,
        "direction": direction,
        "is_core": is_core,
        "outputs_strength": outputs_strength,
    }


def list_all() -> List[Dict[str, Any]]:
    """返回所有已注册事件的元数据列表。"""
    return list(EVENT_REGISTRY.values())


def list_by_category(category: str) -> List[Dict[str, Any]]:
    """按类别返回事件元数据列表。"""
    return [e for e in EVENT_REGISTRY.values() if e["category"] == category]


def get_event(name: str) -> Dict[str, Any]:
    """获取单个事件的元数据。"""
    if name not in EVENT_REGISTRY:
        raise KeyError(f"事件 '{name}' 未注册")
    return EVENT_REGISTRY[name]


def detect_panel(
    factors_df: pd.DataFrame,
    categories: Optional[List[str]] = None,
    event_names: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    统一检测入口。遍历注册表，基于因子列触发事件。

    Args:
        factors_df: 包含因子列的 DataFrame
        categories: 按类别筛选，如 ['趋势事件', '量能事件']
        event_names: 按名称筛选，如 ['evt_dsa_dir_flip_up']
        exclude: 排除的事件名列表

    Returns:
        包含所有事件标记列的 DataFrame
    """
    result = factors_df.copy()
    exclude = exclude or []

    for name, meta in EVENT_REGISTRY.items():
        if name in exclude:
            continue
        if categories and meta["category"] not in categories:
            continue
        if event_names and name not in event_names:
            continue

        # 检查依赖的因子列是否存在
        missing = [f for f in meta["required_factors"] if f not in result.columns]
        if missing:
            # 依赖缺失，输出全0
            result[name] = 0
            if meta["outputs_strength"]:
                result[f"{name}_strength"] = 0.0
            continue

        try:
            detected = meta["detect"](result)
            if isinstance(detected, pd.Series):
                result[name] = detected.astype(int)
            elif isinstance(detected, pd.DataFrame):
                for col in detected.columns:
                    result[col] = detected[col]
        except Exception as e:
            raise RuntimeError(f"检测事件 '{name}' 失败: {e}") from e

    return result
