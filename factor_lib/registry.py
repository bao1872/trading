# -*- coding: utf-8 -*-
"""
factor_lib/registry.py - 因子注册表

Purpose: 全局因子注册中心。新增因子只需调用 register_factor()，无需修改现有代码。

Rules:
    - 本文件只存元数据，不实现计算。
    - 计算函数必须来自 features/ 或 financial_factors/ 的权威实现。
    - 禁止在本文件中写计算公式。

Usage:
    from factor_lib.registry import register_factor, list_all, compute_panel, compute_panel_v2
"""
from typing import Callable, Dict, Any, List, Optional
import pandas as pd

FACTOR_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_factor(
    name: str,
    category: str,
    compute_func: Callable[[pd.DataFrame], pd.Series],
    source_module: str,
    source_function: str,
    description: str,
    direction: str = "neutral",
    is_core: bool = False,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    注册一个因子到全局注册表。

    Args:
        name: 因子列名（唯一标识）
        category: 因子类别，如'趋势类'、'位置类'等
        compute_func: 计算函数，接收 DataFrame 返回 Series
        source_module: 权威实现所在模块，如'features.dsa_bbmacd_24factors_viewer'
        source_function: 权威实现所在函数名
        description: 因子描述
        direction: 因子方向，'positive'/'negative'/'neutral'
        is_core: 是否核心因子
        params: 因子参数配置
    """
    if name in FACTOR_REGISTRY:
        raise ValueError(f"因子 '{name}' 已注册，来源: {FACTOR_REGISTRY[name]['source_module']}")

    FACTOR_REGISTRY[name] = {
        "name": name,
        "category": category,
        "compute": compute_func,
        "source_module": source_module,
        "source_function": source_function,
        "description": description,
        "direction": direction,
        "is_core": is_core,
        "params": params or {},
    }


def list_all() -> List[Dict[str, Any]]:
    """返回所有已注册因子的元数据列表。"""
    return list(FACTOR_REGISTRY.values())


def list_by_category(category: str) -> List[Dict[str, Any]]:
    """按类别返回因子元数据列表。"""
    return [f for f in FACTOR_REGISTRY.values() if f["category"] == category]


def get_factor(name: str) -> Dict[str, Any]:
    """获取单个因子的元数据。"""
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"因子 '{name}' 未注册")
    return FACTOR_REGISTRY[name]


def compute_panel(
    df: pd.DataFrame,
    categories: Optional[List[str]] = None,
    factor_names: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    统一计算入口（旧版，兼容保留）。

    Args:
        df: 输入行情 DataFrame（必须包含 open/high/low/close/volume）
        categories: 按类别筛选，如 ['趋势类', '位置类']
        factor_names: 按名称筛选，如 ['dsa_dir', 'bbmacd']
        exclude: 排除的因子名列表

    Returns:
        包含所有计算因子列的 DataFrame
    """
    return compute_panel_v2(df, categories=categories, factor_names=factor_names, exclude=exclude)


def compute_panel_v2(
    df: pd.DataFrame,
    categories: Optional[List[str]] = None,
    factor_names: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    统一计算入口（V2，批量优化版）。

    通过批量计算函数消除重复计算：
    DSA 算 1 次、BBMACD 算 1 次，供各 category 共享。

    Args:
        df: 输入行情 DataFrame（必须包含 open/high/low/close/volume）
        categories: 按类别筛选，如 ['趋势类', '位置类']
        factor_names: 按名称筛选，如 ['dsa_dir', 'bbmacd']
        exclude: 排除的因子名列表

    Returns:
        包含所有计算因子列的 DataFrame
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
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, compute_bbmacd, DSAConfig

    result = df.copy()
    exclude = set(exclude or [])
    factor_name_set = set(factor_names) if factor_names else None

    CATEGORY_MAP = {
        "趋势类": ("trend", compute_trend_factors),
        "位置类": ("position", compute_position_factors),
        "动量类": ("momentum", compute_momentum_factors),
        "量能类": ("volume", compute_volume_factors),
        "协同类": ("coordination", compute_coordination_factors),
        "节奏类": ("rhythm", compute_rhythm_factors),
        "结构类": ("structure", compute_structure_factors),
        "风险类": ("risk", compute_risk_factors),
        "财务类": ("fundamental", compute_fundamental_factors),
        "原始特征": ("raw_features", compute_raw_features),
    }

    dsanf, bbnf, volnf = False, False, False
    needed_cats = set()
    for name, meta in FACTOR_REGISTRY.items():
        if name in exclude:
            continue
        if categories and meta["category"] not in categories:
            continue
        if factor_name_set and name not in factor_name_set:
            continue
        needed_cats.add(meta["category"])
        src = meta.get("source_module", "")
        if "compute_dsa" in src:
            dsanf = True
        if "compute_bbmacd" in src:
            bbnf = True
        if "volume" in src.lower():
            volnf = True

    cfg = DSAConfig()
    dsa_result = None
    bb_result = None
    if dsanf and need_dsa(needed_cats):
        dsa_result, _, _ = compute_dsa(df, cfg)
    if bbnf and need_bb(needed_cats):
        bb_result = compute_bbmacd(df)

    for cat_name, (cat_key, cat_func) in CATEGORY_MAP.items():
        if cat_name not in needed_cats:
            continue
        kwargs = {}
        if cat_key in ("trend", "position", "rhythm", "coordination", "raw_features"):
            kwargs["dsa_result"] = dsa_result
        if cat_key in ("trend", "momentum", "coordination", "raw_features"):
            kwargs["bb_result"] = bb_result
        try:
            cat_result = cat_func(df, **kwargs)
            if isinstance(cat_result, pd.DataFrame):
                for col in cat_result.columns:
                    if col not in exclude:
                        if factor_name_set and col not in factor_name_set:
                            continue
                        result[col] = cat_result[col]
        except Exception as e:
            raise RuntimeError(f"计算类别 '{cat_name}' 因子失败: {e}") from e

    return result


def need_dsa(needed_cats: set) -> bool:
    dsa_cats = {"趋势类", "位置类", "节奏类", "协同类", "原始特征"}
    return bool(needed_cats & dsa_cats)


def need_bb(needed_cats: set) -> bool:
    bb_cats = {"趋势类", "动量类", "协同类", "原始特征"}
    return bool(needed_cats & bb_cats)
