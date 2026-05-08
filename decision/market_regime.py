# -*- coding: utf-8 -*-
"""
decision/market_regime.py - 市场环境过滤

Purpose: 根据市场状态过滤允许的操作类型。

Output:
    {
        "regime": "expansion",
        "allowed_operations": ["buy", "hold"],
        "risk_level": "medium",
        "position_limit": 0.8
    }
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List


# 市场环境定义
REGIME_CONFIG = {
    "expansion": {
        "allowed_operations": ["buy", "hold"],
        "risk_level": "medium",
        "position_limit": 0.8,
    },
    "repair": {
        "allowed_operations": ["hold", "sell"],
        "risk_level": "medium-high",
        "position_limit": 0.5,
    },
    "retreat": {
        "allowed_operations": ["sell", "hold"],
        "risk_level": "high",
        "position_limit": 0.2,
    },
    "defense": {
        "allowed_operations": ["sell"],
        "risk_level": "very_high",
        "position_limit": 0.0,
    },
}


def filter_by_regime(market_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据市场状态过滤操作。

    Args:
        market_state: 市场状态字典（包含 regime 字段）

    Returns:
        过滤结果
    """
    regime = market_state.get("regime", "unknown")
    config = REGIME_CONFIG.get(regime, REGIME_CONFIG["defense"])

    return {
        "regime": regime,
        "allowed_operations": config["allowed_operations"],
        "risk_level": config["risk_level"],
        "position_limit": config["position_limit"],
    }
