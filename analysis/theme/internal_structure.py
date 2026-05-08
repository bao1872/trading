# -*- coding: utf-8 -*-
"""
analysis/theme/internal_structure.py - 题材内部结构

Purpose: 识别题材内部的龙头/跟风/补涨股。

Usage:
    from analysis.theme.internal_structure import analyze_internal_structure
    structure = analyze_internal_structure(theme_stocks_df)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from typing import Dict, Any, List


def analyze_internal_structure(theme_stocks_df: pd.DataFrame) -> Dict[str, Any]:
    """
    分析题材内部结构。

    Args:
        theme_stocks_df: 某题材下所有股票的 DataFrame

    Returns:
        内部结构分析结果
    """
    if len(theme_stocks_df) < 3:
        return {
            "leader": None,
            "followers": [],
            "laggards": [],
            "health_score": 0,
            "risk_flags": ["样本不足"],
        }

    # 计算每只股票的近期涨幅和量能
    stock_metrics = []
    for ts_code, group in theme_stocks_df.groupby("ts_code"):
        if len(group) < 2:
            continue
        ret = group["close"].iloc[-1] / group["close"].iloc[0] - 1 if "close" in group.columns else 0
        vol = group["vol"].mean() if "vol" in group.columns else 0
        stock_metrics.append({
            "ts_code": ts_code,
            "return": ret,
            "volume": vol,
            "score": ret * vol,  # 涨幅*量能 = 活跃度
        })

    if not stock_metrics:
        return {
            "leader": None,
            "followers": [],
            "laggards": [],
            "health_score": 0,
            "risk_flags": ["数据不足"],
        }

    metrics_df = pd.DataFrame(stock_metrics).sort_values("score", ascending=False)

    # 龙头：涨幅最大 + 量能最活跃
    leader = metrics_df.iloc[0]["ts_code"] if len(metrics_df) > 0 else None

    # 跟风股：与龙头同向但幅度较小（前50%除龙头外）
    mid_point = max(1, len(metrics_df) // 2)
    followers = metrics_df.iloc[1:mid_point]["ts_code"].tolist() if len(metrics_df) > 1 else []

    # 补涨股：题材内滞涨但结构良好（后50%）
    laggards = metrics_df.iloc[mid_point:]["ts_code"].tolist() if len(metrics_df) > mid_point else []

    # 健康度：龙头是否持续强于跟风
    health_score = 50
    if leader and followers:
        leader_ret = metrics_df[metrics_df["ts_code"] == leader]["return"].values[0]
        follower_avg_ret = metrics_df[metrics_df["ts_code"].isin(followers)]["return"].mean()
        if leader_ret > follower_avg_ret:
            health_score += 25
        if len(laggards) > 0:
            health_score += 25  # 有补涨空间

    risk_flags = []
    if leader and followers:
        leader_ret = metrics_df[metrics_df["ts_code"] == leader]["return"].values[0]
        follower_avg_ret = metrics_df[metrics_df["ts_code"].isin(followers)]["return"].mean()
        if leader_ret < follower_avg_ret:
            risk_flags.append("龙头弱于跟风")
    if len(followers) < 2:
        risk_flags.append("跟风不足")

    return {
        "leader": leader,
        "followers": followers,
        "laggards": laggards,
        "health_score": health_score,
        "risk_flags": risk_flags,
    }
