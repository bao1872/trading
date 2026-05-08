# -*- coding: utf-8 -*-
"""
analysis/theme/sustainability.py - 题材持续性判断

Purpose: 判断题材的持续性，输出可持续性评分和可信度。

Output:
    {
        "theme_name": "xxx",
        "sustainability_score": 75,      # 0-100
        "confidence_score": 0.85,        # 0-1
        "event_diffusion_trend": "expanding",
        "strength_acceleration": 0.12,
        "risk_flags": ["龙头走弱", "跟风不足"]
    }

Rules:
    - theme_stock_count < 3 → confidence_score = 0（样本不足）
    - top3_ret_pct / theme_avg_ret > 3 → confidence_score 降权（过度集中）
    - 必须有至少 1 个龙头 + 2 个跟风 → 才算有效题材
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from typing import List, Dict, Any


def assess_sustainability(
    theme_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window: int = 5,
) -> Dict[str, Any]:
    """
    评估题材的持续性。

    Args:
        theme_df: 题材聚合结果（单题材）
        events_df: 该题材下所有股票的事件数据
        window: 观察窗口

    Returns:
        持续性评估结果字典
    """
    stock_count = len(events_df)

    # 样本数检查
    if stock_count < 3:
        return {
            "theme_name": theme_df.get("concept_name", "unknown"),
            "sustainability_score": 0,
            "confidence_score": 0.0,
            "event_diffusion_trend": "insufficient_sample",
            "strength_acceleration": 0.0,
            "risk_flags": ["样本不足（<3只）"],
        }

    # 计算集中度
    if "close" in events_df.columns:
        returns = events_df.groupby("ts_code")["close"].apply(lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 1 else 0)
        top3_avg_ret = returns.nlargest(3).mean()
        theme_avg_ret = returns.mean()
        concentration_ratio = top3_avg_ret / theme_avg_ret if theme_avg_ret != 0 else 1
    else:
        concentration_ratio = 1

    # 可信度计算
    confidence = min(1.0, stock_count / 10)  # 样本数越多越可信
    if concentration_ratio > 3:
        confidence *= 0.5  # 过度集中降权

    # 事件扩散度趋势
    recent_events = events_df.tail(window)
    up_events = recent_events.get("evt_up_move_with_vol_spike", pd.Series(0)).sum()
    down_events = recent_events.get("evt_down_move_with_vol_spike", pd.Series(0)).sum()

    if up_events > down_events * 2:
        diffusion_trend = "expanding"
    elif down_events > up_events * 2:
        diffusion_trend = "contracting"
    else:
        diffusion_trend = "stable"

    # 强度加速度（简化：用事件数变化率）
    if len(events_df) >= window * 2:
        prev_window = events_df.iloc[-window * 2 : -window]
        curr_window = events_df.iloc[-window:]
        prev_events = prev_window.get("evt_up_move_with_vol_spike", pd.Series(0)).sum()
        curr_events = curr_window.get("evt_up_move_with_vol_spike", pd.Series(0)).sum()
        strength_accel = (curr_events - prev_events) / max(prev_events, 1)
    else:
        strength_accel = 0.0

    # 可持续性评分
    sustainability = min(100, max(0, int(
        stock_count * 3 +                    # 样本数贡献
        up_events * 5 +                      # 上涨事件贡献
        (strength_accel * 20) +              # 加速度贡献
        (confidence * 30)                    # 可信度贡献
    )))

    # 风险标记
    risk_flags = []
    if concentration_ratio > 3:
        risk_flags.append("过度集中")
    if down_events > up_events:
        risk_flags.append("空头事件占优")
    if stock_count < 5:
        risk_flags.append("样本偏少")

    return {
        "theme_name": theme_df.get("concept_name", "unknown"),
        "sustainability_score": sustainability,
        "confidence_score": round(confidence, 2),
        "event_diffusion_trend": diffusion_trend,
        "strength_acceleration": round(strength_accel, 2),
        "risk_flags": risk_flags,
    }
