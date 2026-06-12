# -*- coding: utf-8 -*-
"""
Purpose: 人工交易候选表评分+分层公共逻辑（SSOT）
Inputs:  被 14_manual_trade_candidate_report.py 和 15_manual_filter_validation.py 引用
Outputs: 无（被 import 使用）
How to Run: 无需单独运行
Side Effects: 无
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RISK_THRESHOLD = 0.8351
HARD_EXCLUDE_RISK_THRESHOLD = 0.84

TIER_ORDER = ["S_low_buy", "S_watch_breakout", "A_observe", "B_watch_pullback", "B_weak_ignore", "C_exclude"]


def compute_score(row: pd.Series, reclaim_median: float) -> tuple:
    score = 0
    reasons = []
    warnings = []

    if bool(row.get("support_cluster_is_strong", False)):
        score += 30
        reasons.append("强支撑簇+30")

    if bool(row.get("is_volume_shrink", False)):
        score += 20
        reasons.append("缩量+20")

    conf_score = row.get("support_confluence_score", 0)
    if pd.notna(conf_score) and conf_score >= 3.0:
        score += 10
        reasons.append("共振分高+10")

    if bool(row.get("is_support_flipped", False)):
        score += 5
        reasons.append("R2S已验证+5")

    reclaim = row.get("support_reclaim_strength_atr", np.nan)
    if pd.notna(reclaim) and reclaim >= reclaim_median:
        score += 15
        reasons.append("收回强度高+15")

    close_pos = row.get("close_pos_in_bar", np.nan)
    if pd.notna(close_pos) and close_pos >= 0.7:
        score += 10
        reasons.append("收盘上半部+10")

    if bool(row.get("is_long_lower_shadow", False)):
        score += 5
        reasons.append("长下影+5")

    risk = row.get("risk_score", np.nan)
    if pd.notna(risk) and risk < RISK_THRESHOLD:
        score += 20
        reasons.append("risk_score低+20")
    elif pd.notna(risk) and risk >= RISK_THRESHOLD:
        score -= 30
        warnings.append("risk_score高-30")

    broken = bool(row.get("daily_broken_weekly_low", False))
    if not broken:
        score += 15
        reasons.append("未跌破周线low+15")
    else:
        score -= 40
        warnings.append("日线跌破周线low-40")

    dist_pct = row.get("distance_to_weekly_low_pct", np.nan)
    if pd.notna(dist_pct) and dist_pct < 5:
        score += 10
        reasons.append("距支撑近+10")
    elif pd.notna(dist_pct) and dist_pct > 8:
        score -= 15
        warnings.append("距支撑远-15")

    if bool(row.get("evt_pierce_support_cluster_reclaim_high_volume", False)):
        score -= 20
        warnings.append("放量刺破-20")

    ret_since = row.get("ret_since_signal", np.nan)
    if pd.notna(ret_since) and ret_since > 0.10:
        score -= 10
        warnings.append("涨幅过大-10")

    return score, "; ".join(reasons), "; ".join(warnings)


def compute_manual_rr(row: pd.Series) -> tuple:
    close = row.get("latest_daily_close", row.get("close", np.nan))
    if pd.isna(close) or close <= 0:
        return np.nan, np.nan, np.nan

    resistance = row.get("resistance_ref", np.nan)
    if pd.isna(resistance):
        resistance = row.get("resistance_active", np.nan)
    if pd.isna(resistance) or resistance <= 0:
        return np.nan, np.nan, np.nan

    weekly_low = row.get("low", np.nan)
    if pd.isna(weekly_low) or weekly_low <= 0:
        return np.nan, np.nan, np.nan

    upside_pct = resistance / close - 1
    downside_pct = close / weekly_low - 1

    if downside_pct <= 0:
        return np.nan, upside_pct, downside_pct

    manual_rr = upside_pct / downside_pct
    return manual_rr, upside_pct, downside_pct


def compute_entry_status(row: pd.Series) -> str:
    if bool(row.get("daily_broken_weekly_low", False)):
        return "已破位剔除"

    ret_since = row.get("ret_since_signal", np.nan)
    if pd.notna(ret_since) and ret_since > 10:
        return "已反弹过多"

    dist_resistance = row.get("current_to_resistance_pct", np.nan)
    if pd.notna(dist_resistance) and dist_resistance < 3:
        return "接近压力不追"

    dist_low = row.get("distance_to_weekly_low_pct", np.nan)
    manual_rr = row.get("manual_rr", np.nan)
    if pd.notna(dist_low) and dist_low < 5 and pd.notna(manual_rr) and manual_rr >= 1.5:
        return "可低吸"

    return "可观察"


def compute_tags(row: pd.Series, reclaim_median: float) -> tuple:
    positive = []
    negative = []

    if bool(row.get("support_cluster_is_strong", False)):
        positive.append("强支撑簇")
    if bool(row.get("is_volume_shrink", False)):
        positive.append("缩量")
    reclaim = row.get("support_reclaim_strength_atr", np.nan)
    if pd.notna(reclaim) and reclaim >= reclaim_median:
        positive.append("收回强")
    risk = row.get("risk_score", np.nan)
    if pd.notna(risk) and risk < RISK_THRESHOLD:
        positive.append("risk低")
    if not bool(row.get("daily_broken_weekly_low", False)):
        positive.append("未破周线low")
    if bool(row.get("is_support_flipped", False)):
        positive.append("R2S已验证")

    dist_pct = row.get("distance_to_weekly_low_pct", np.nan)
    if pd.notna(dist_pct) and dist_pct > 8:
        negative.append(f"距支撑远({dist_pct:.1f}%)")
    dist_resistance = row.get("current_to_resistance_pct", np.nan)
    if pd.notna(dist_resistance) and dist_resistance < 3:
        negative.append("接近压力")
    ret_since = row.get("ret_since_signal", np.nan)
    if pd.notna(ret_since) and ret_since > 5:
        negative.append(f"涨幅过大({ret_since:.1f}%)")
    if pd.notna(risk) and risk >= RISK_THRESHOLD:
        negative.append("risk高")
    if bool(row.get("evt_pierce_support_cluster_reclaim_high_volume", False)):
        negative.append("放量刺破")

    entry_status = row.get("entry_status", "")
    tier = row.get("tier", "")
    if entry_status == "可低吸":
        dist_str = f"{dist_pct:.1f}%" if pd.notna(dist_pct) else "?"
        action = f"可低吸，距支撑{dist_str}"
    elif tier == "B_watch_pullback":
        action = "等回踩再观察"
    elif entry_status == "已反弹过多":
        action = "不追，等回踩"
    elif entry_status == "接近压力不追":
        action = "接近压力，不追"
    elif entry_status == "已破位剔除":
        action = "已破位，剔除"
    else:
        action = "观察，等待信号"

    return "; ".join(positive), "; ".join(negative), action


def is_hard_exclude(row: pd.Series) -> bool:
    if bool(row.get("daily_broken_weekly_low", False)):
        return True
    risk = row.get("risk_score", np.nan)
    if pd.notna(risk) and risk >= HARD_EXCLUDE_RISK_THRESHOLD:
        return True
    if bool(row.get("evt_pierce_support_cluster_reclaim_high_volume", False)):
        return True
    return False


def assign_tier(df: pd.DataFrame, reclaim_median: float) -> pd.DataFrame:
    df = df.copy()

    scores = df.apply(lambda r: compute_score(r, reclaim_median), axis=1)
    df["final_score"] = [s[0] for s in scores]
    df["reason"] = [s[1] for s in scores]
    df["risk_warning"] = [s[2] for s in scores]

    df["is_hard_exclude"] = df.apply(is_hard_exclude, axis=1)

    rr_results = df.apply(compute_manual_rr, axis=1)
    df["manual_rr"] = [r[0] for r in rr_results]
    df["upside_to_resistance_pct"] = [r[1] for r in rr_results]
    df["downside_to_weekly_low_pct"] = [r[2] for r in rr_results]

    df["entry_status"] = df.apply(compute_entry_status, axis=1)

    tag_results = df.apply(lambda r: compute_tags(r, reclaim_median), axis=1)
    df["positive_tags"] = [t[0] for t in tag_results]
    df["negative_tags"] = [t[1] for t in tag_results]
    df["action_suggestion"] = [t[2] for t in tag_results]

    df["tier"] = "B_ignore"

    non_exclude = df[~df["is_hard_exclude"]].copy()
    if len(non_exclude) > 0:
        n = len(non_exclude)
        s_threshold = non_exclude["final_score"].quantile(0.85)
        a_threshold = non_exclude["final_score"].quantile(0.65)

        df.loc[~df["is_hard_exclude"] & (df["final_score"] >= s_threshold), "tier"] = "S"
        df.loc[
            ~df["is_hard_exclude"]
            & (df["final_score"] < s_threshold)
            & (df["final_score"] >= a_threshold),
            "tier",
        ] = "A"

    s_mask = df["tier"] == "S"
    df.loc[s_mask & (df["entry_status"] == "可低吸"), "tier"] = "S_low_buy"
    df.loc[s_mask & (df["entry_status"] != "可低吸"), "tier"] = "S_watch_breakout"

    a_mask = df["tier"] == "A"
    df.loc[a_mask, "tier"] = "A_observe"

    b_mask = df["tier"] == "B_ignore"
    manual_rr_low = df["manual_rr"].notna() & (df["manual_rr"] < 1.0)

    s_low_mask = df["tier"] == "S_low_buy"
    df.loc[s_low_mask & manual_rr_low, "tier"] = "A_observe"

    s_watch_mask = df["tier"] == "S_watch_breakout"
    df.loc[s_watch_mask & manual_rr_low, "tier"] = "A_observe"

    a_obs_mask = df["tier"] == "A_observe"
    df.loc[a_obs_mask & manual_rr_low, "tier"] = "B_ignore"

    b_mask = df["tier"] == "B_ignore"
    pullback_mask = b_mask & df["entry_status"].isin(["已反弹过多", "接近压力不追"])
    df.loc[pullback_mask, "tier"] = "B_watch_pullback"
    df.loc[b_mask & ~pullback_mask, "tier"] = "B_weak_ignore"

    df.loc[df["is_hard_exclude"], "tier"] = "C_exclude"

    return df


if __name__ == "__main__":
    print("manual_filter_scoring.py 是公共模块，被 14/15 脚本引用，无需单独运行")
    print(f"RISK_THRESHOLD={RISK_THRESHOLD}, HARD_EXCLUDE_RISK_THRESHOLD={HARD_EXCLUDE_RISK_THRESHOLD}")
    print(f"分层顺序: {TIER_ORDER}")
