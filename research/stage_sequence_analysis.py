# -*- coding: utf-8 -*-
"""
research/stage_sequence_analysis.py - 阶段序列分析（研究层）

Purpose: 读取事件列，滚动统计事件密度，计算阶段评分。
         这是研究层代码，不在 factor_lib/event_lib 底座主流程中调用。
         底座只输出因子值和 0/1 事件，阶段判断在此研究层通过事件序列组合来做。

Public API:
    compute_stage_scores(events_df, factors_df=None, config=None) -> DataFrame

Output Columns:
    - cost_zone_forming_score: 成本区形成度
    - cost_zone_maturity_score: 成本区成熟度
    - wash_cycle_count: 区间循环数
    - wash_cycle_quality_score: 区间循环质量
    - final_shake_score: 末端强洗候选强度
    - repair_score: 修复度
    - failure_score: 破位风险
    - stage_chain_score: 阶段链条综合分
    - stage_guess: 仅供研究展示的阶段推断

Inputs:
    events_df: detect_panel() 输出的事件 DataFrame
    factors_df: 可选的因子 DataFrame
    config: 可选配置字典

Outputs: DataFrame with 9 columns
How to Run:
    python -m research.stage_sequence_analysis
Examples:
    python -m research.stage_sequence_analysis
Side Effects: None
"""
import pandas as pd
import numpy as np

_DEFAULT_CONFIG = {
    "density_window": 20,
    "cz_forming_weights": {
        "evt_trend_flat": 0.3,
        "evt_price_cross_mid_freq_high": 0.3,
        "evt_volatility_compress": 0.2,
        "evt_lower_reclaim_freq_high": 0.2,
    },
    "cz_maturity_weights": {
        "evt_upper_reject_count_n": 0.0,
        "evt_lower_reclaim_freq_high": 0.3,
        "evt_price_cross_mid_freq_high": 0.2,
        "evt_lower_reclaim_freq_high": 0.2,
        "evt_range_pullback_reclaim_cycle": 0.3,
    },
    "shake_weights": {
        "evt_lower_break": 0.25,
        "evt_break_last_reclaim_low": 0.2,
        "evt_lower_break_reclaim": 0.25,
        "evt_long_lower_shadow": 0.15,
        "evt_stop_cluster_reclaim": 0.15,
    },
    "repair_weights": {
        "evt_reclaim_lower": 0.2,
        "evt_reclaim_mid": 0.25,
        "evt_reclaim_upper": 0.1,
        "evt_reclaim_dsa_vwap": 0.15,
        "evt_trend_slope_turn_positive": 0.15,
        "evt_bbmacd_turn_positive": 0.15,
    },
    "failure_weights": {
        "evt_lower_break_no_reclaim": 0.3,
        "evt_trend_down_confirm": 0.25,
        "evt_weak_rebound": 0.2,
        "evt_distribution_risk": 0.25,
    },
    "failure_threshold": 0.5,
    "shake_mature_threshold": 0.5,
    "repair_threshold": 0.5,
    "shake_threshold": 0.5,
    "forming_threshold": 0.4,
    "maturity_threshold": 0.5,
}


def _weighted_event_density(
    events_df: pd.DataFrame,
    weights: dict,
    window: int,
) -> pd.Series:
    weighted_sum = pd.Series(0.0, index=events_df.index)
    total_weight = sum(weights.values())
    for evt_name, w in weights.items():
        if evt_name in events_df.columns:
            evt_col = events_df[evt_name].fillna(0).astype(float)
            density = evt_col.rolling(window, min_periods=1).mean()
            weighted_sum += density * w
    if total_weight > 0:
        weighted_sum /= total_weight
    return weighted_sum


def compute_stage_scores(
    events_df: pd.DataFrame,
    factors_df: pd.DataFrame = None,
    config: dict = None,
) -> pd.DataFrame:
    """
    阶段序列分析：读取事件列，滚动统计事件密度，计算阶段评分。

    Args:
        events_df: detect_panel() 输出的事件 DataFrame
        factors_df: 可选的因子 DataFrame
        config: 可选配置字典，覆盖默认阈值

    Returns:
        DataFrame 含 9 列阶段评分
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    window = cfg["density_window"]
    result = pd.DataFrame(index=events_df.index)

    cost_zone_forming_score = _weighted_event_density(
        events_df, cfg["cz_forming_weights"], window
    )
    result["cost_zone_forming_score"] = cost_zone_forming_score

    cost_zone_maturity_score = _weighted_event_density(
        events_df, cfg["cz_maturity_weights"], window
    )
    result["cost_zone_maturity_score"] = cost_zone_maturity_score

    if "evt_range_pullback_reclaim_cycle" in events_df.columns:
        wash_cycle_events = events_df["evt_range_pullback_reclaim_cycle"].fillna(0).astype(int)
        wash_cycle_count = wash_cycle_events.cumsum()
        result["wash_cycle_count"] = wash_cycle_count
    else:
        result["wash_cycle_count"] = 0

    wash_quality = pd.Series(0.0, index=events_df.index)
    if "evt_lower_break_short_hold_long" in events_df.columns:
        wash_quality += events_df["evt_lower_break_short_hold_long"].fillna(0).astype(float).rolling(window, min_periods=1).mean() * 0.4
    if "evt_price_cross_above_mid" in events_df.columns:
        wash_quality += events_df["evt_price_cross_above_mid"].fillna(0).astype(float).rolling(window, min_periods=1).mean() * 0.3
    if "evt_pullback_to_lower" in events_df.columns:
        wash_quality += events_df["evt_pullback_to_lower"].fillna(0).astype(float).rolling(window, min_periods=1).mean() * 0.3
    result["wash_cycle_quality_score"] = wash_quality.clip(0, 1)

    raw_shake = _weighted_event_density(
        events_df, cfg["shake_weights"], window
    )
    shake_mature_gate = np.where(
        cost_zone_maturity_score > cfg["shake_mature_threshold"],
        1.0,
        cost_zone_maturity_score / max(cfg["shake_mature_threshold"], 1e-9),
    )
    final_shake_score = raw_shake * shake_mature_gate
    result["final_shake_score"] = final_shake_score.clip(0, 1)

    repair = _weighted_event_density(
        events_df, cfg["repair_weights"], window
    )
    result["repair_score"] = repair.clip(0, 1)

    failure = _weighted_event_density(
        events_df, cfg["failure_weights"], window
    )
    result["failure_score"] = failure.clip(0, 1)

    chain_score = (
        cost_zone_forming_score * 0.15
        + cost_zone_maturity_score * 0.2
        + wash_quality * 0.1
        + final_shake_score * 0.2
        + repair * 0.2
        + (1.0 - failure) * 0.15
    )
    result["stage_chain_score"] = chain_score.clip(0, 1)

    stage_guess = pd.Series("无有效阶段结构", index=events_df.index)
    failure_mask = failure > cfg["failure_threshold"]
    repair_shake_mature = (
        (repair > cfg["repair_threshold"])
        & (final_shake_score > cfg["shake_threshold"])
        & (cost_zone_maturity_score > cfg["maturity_threshold"])
    )
    shake_mature = (
        (final_shake_score > cfg["shake_threshold"])
        & (cost_zone_maturity_score > cfg["maturity_threshold"])
        & ~repair_shake_mature
    )
    wash_forming = (
        (result["wash_cycle_count"] > result["wash_cycle_count"].shift(1))
        & (cost_zone_forming_score > cfg["forming_threshold"])
        & ~shake_mature
        & ~repair_shake_mature
    )
    forming = (
        (cost_zone_forming_score > cfg["forming_threshold"])
        & ~wash_forming
        & ~shake_mature
        & ~repair_shake_mature
    )

    stage_guess[failure_mask] = "结构失败/风险"
    stage_guess[repair_shake_mature & ~failure_mask] = "疑似震仓末期/修复确认"
    stage_guess[shake_mature & ~failure_mask] = "疑似末端强洗/震仓中"
    stage_guess[wash_forming & ~failure_mask] = "成本区成熟/洗盘迭代"
    stage_guess[forming & ~failure_mask] = "成本区形成"
    result["stage_guess"] = stage_guess

    return result


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.05)
    low = close - np.abs(np.random.randn(n) * 0.05)
    opn = close + np.random.randn(n) * 0.02
    vol = np.random.randint(1000, 5000, n).astype(float)
    df = pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )

    from factor_lib import compute_all_factors_v2
    from event_lib import detect_panel

    factors_df = compute_all_factors_v2(df)
    events_df = detect_panel(factors_df)
    stage_df = compute_stage_scores(events_df, factors_df)

    print("=== 阶段序列分析输出 ===")
    print(stage_df.describe())
    print()
    print("stage_guess 分布:")
    print(stage_df["stage_guess"].value_counts())
