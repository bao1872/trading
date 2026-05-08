# -*- coding: utf-8 -*-
"""
analysis/market/daily_report.py - 盘面日报生成

Purpose: 生成每日盘面解读报告。

Output Format:
    日期: 2024-01-01
    市场状态: 扩张期 | 置信度: 0.82
    指数背景: 上证强扩张 | 沪深300跟随
    量能流向: 小盘放量率 > 大盘 | 资金向成长扩散
    主线题材:
      1. AI算力 (持续性: 78/100, 置信度: 0.91, 龙头: xxx)
      2. 新能源 (持续性: 65/100, 置信度: 0.75, 风险: 龙头走弱)
    风格偏好: 小盘成长
    板块扩散度: 高
    风险提示: 题材1龙头换手异常，注意退潮

Usage:
    from analysis.market.daily_report import generate_daily_report
    report = generate_daily_report(market_state, themes, index_ctx, vol_ctx)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any, List
from datetime import datetime


def generate_daily_report(
    trade_date: str,
    market_state: Dict[str, Any],
    themes: List[Dict[str, Any]],
    index_context: Dict[str, Any],
    volume_context: Dict[str, Any],
) -> str:
    """
    生成盘面日报。

    Args:
        trade_date: 交易日期
        market_state: 市场状态字典
        themes: 题材列表
        index_context: 指数背景
        volume_context: 成交量背景

    Returns:
        日报文本
    """
    lines = []
    lines.append(f"日期: {trade_date}")
    lines.append("")

    # 市场状态
    regime = market_state.get("regime", "unknown")
    confidence = market_state.get("confidence", 0)
    lines.append(f"市场状态: {regime} | 置信度: {confidence:.2f}")

    # 指数背景
    sse_state = index_context.get("sse_state", "unknown")
    hs300_state = index_context.get("hs300_state", "unknown")
    lines.append(f"指数背景: 上证{sse_state} | 沪深300{hs300_state}")

    # 量能流向
    vol_flow = volume_context.get("flow_direction", "unknown")
    lines.append(f"量能流向: {vol_flow}")

    # 主线题材
    lines.append("主线题材:")
    for i, theme in enumerate(themes[:5], 1):
        name = theme.get("theme_name", "unknown")
        score = theme.get("sustainability_score", 0)
        conf = theme.get("confidence_score", 0)
        risk = ", ".join(theme.get("risk_flags", [])) or "无"
        leader = theme.get("leader", "N/A")
        lines.append(f"  {i}. {name} (持续性: {score}/100, 置信度: {conf:.2f}, 龙头: {leader})")
        if risk != "无":
            lines.append(f"     风险: {risk}")

    # 风格偏好
    style = market_state.get("style", "unknown")
    lines.append(f"风格偏好: {style}")

    # 板块扩散度
    diffusion = market_state.get("diffusion", "unknown")
    lines.append(f"板块扩散度: {diffusion}")

    # 风险提示
    risks = []
    for theme in themes[:3]:
        risks.extend(theme.get("risk_flags", []))
    if risks:
        lines.append(f"风险提示: {', '.join(set(risks))}")
    else:
        lines.append("风险提示: 无")

    return "\n".join(lines)
