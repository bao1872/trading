# -*- coding: utf-8 -*-
"""
decision/registry.py - 信号规则注册表

Purpose: 注册决策规则，支持插件化扩展。

Usage:
    from decision.registry import register_rule, evaluate_rules
"""
from typing import Callable, Dict, Any, List

RULE_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_rule(
    name: str,
    rule_func: Callable[[Dict[str, Any]], bool],
    description: str,
    category: str = "general",
) -> None:
    """
    注册一个决策规则。

    Args:
        name: 规则名称
        rule_func: 规则函数，接收 context 字典返回 bool
        description: 规则描述
        category: 规则类别
    """
    RULE_REGISTRY[name] = {
        "name": name,
        "rule": rule_func,
        "description": description,
        "category": category,
    }


def evaluate_rules(context: Dict[str, Any]) -> Dict[str, bool]:
    """
    评估所有已注册规则。

    Args:
        context: 决策上下文

    Returns:
        各规则评估结果
    """
    return {name: meta["rule"](context) for name, meta in RULE_REGISTRY.items()}


# ---------- 默认决策规则 ----------

def _rule_trend_aligned(context: Dict[str, Any]) -> bool:
    """趋势一致：dsa_dir > 0 且 trend_align_momo > 0"""
    return context.get("dsa_dir", 0) > 0 and context.get("trend_align_momo", 0) > 0


def _rule_momentum_positive(context: Dict[str, Any]) -> bool:
    """动量正向：bbmacd > 0 且 bbmacd_slope_3 > 0"""
    return context.get("bbmacd", 0) > 0 and context.get("bbmacd_slope_3", 0) > 0


def _rule_volume_confirm(context: Dict[str, Any]) -> bool:
    """量能确认：vol_zscore_20 > 1"""
    return context.get("vol_zscore_20", 0) > 1.0


def _rule_not_overextended(context: Dict[str, Any]) -> bool:
    """非超买：dsa_pivot_pos_01 < 0.8"""
    return context.get("dsa_pivot_pos_01", 1.0) < 0.8


def _rule_low_risk(context: Dict[str, Any]) -> bool:
    """低风险：atr_pct < 5%"""
    return context.get("atr_pct", 100.0) < 5.0


def _rule_earnings_good(context: Dict[str, Any]) -> bool:
    """基本面良好：q_rev_yoy > 0"""
    return context.get("q_rev_yoy", 0) > 0


# 注册默认规则
register_rule(
    name="trend_aligned",
    rule_func=_rule_trend_aligned,
    description="趋势一致：dsa_dir > 0 且 trend_align_momo > 0",
    category="趋势",
)
register_rule(
    name="momentum_positive",
    rule_func=_rule_momentum_positive,
    description="动量正向：bbmacd > 0 且斜率向上",
    category="动量",
)
register_rule(
    name="volume_confirm",
    rule_func=_rule_volume_confirm,
    description="量能确认：成交量Z-score > 1",
    category="量能",
)
register_rule(
    name="not_overextended",
    rule_func=_rule_not_overextended,
    description="非超买：位置 < 0.8",
    category="位置",
)
register_rule(
    name="low_risk",
    rule_func=_rule_low_risk,
    description="低风险：ATR < 5%",
    category="风险",
)
register_rule(
    name="earnings_good",
    rule_func=_rule_earnings_good,
    description="基本面良好：营收同比 > 0",
    category="基本面",
)
