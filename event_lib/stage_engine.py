# -*- coding: utf-8 -*-
"""
event_lib/stage_engine.py - 阶段事件统计引擎

Purpose: 读取事件列，滚动统计事件密度，计算阶段评分。
         输出可供回测脚本使用的纯计算结果，不写数据库。

Public API:
    compute_stage_scores(events_df, factors_df=None, config=None) -> DataFrame

Output Columns:
    - cost_zone_forming_score: 成本区形成度
    - cost_zone_maturity_score: 成本区成熟度
    - wash_cycle_count: 有效洗盘循环数
    - wash_cycle_quality_score: 最近洗盘质量
    - final_shake_score: 末端强洗候选强度
    - repair_score: 震仓后修复度
    - failure_score: 结构失败/出货风险
    - stage_chain_score: 阶段链条综合分
    - stage_guess: 仅供研究展示的阶段推断

Inputs:
    events_df: detect_panel() 输出的事件 DataFrame
    factors_df: 可选的因子 DataFrame（用于 maturity gating）
    config: 可选配置字典

Outputs: DataFrame with 9 columns
How to Run:
    python -m event_lib.stage_engine
Examples:
    python -m event_lib.stage_engine
Side Effects: None
"""
import pandas as pd
import numpy as np

_DEFAULT_CONFIG = {
    "density_window": 20,
    "cz_forming_weights": {
        "evt_cz_trend_flat": 0.3,
        "evt_cz_price_around_mid": 0.3,
        "evt_cz_volatility_compress": 0.2,
        "evt_cz_lower_hold": 0.2,
    },
    "cz_maturity_weights": {
        "evt_cz_upper_test_fail": 0.0,
        "evt_cz_lower_reclaim": 0.3,
        "evt_cz_range_reentry": 0.0,
        "evt_cz_mature": 0.3,
        "evt_cz_price_around_mid": 0.2,
        "evt_cz_lower_hold": 0.2,
    },
    "shake_weights": {
        "evt_shake_break_lower": 0.25,
        "evt_shake_break_last_wash_low": 0.2,
        "evt_shake_lower_reclaim": 0.25,
        "evt_shake_long_lower_shadow": 0.15,
        "evt_shake_stop_cluster_reclaim": 0.15,
    },
    "repair_weights": {
        "evt_repair_reclaim_lower": 0.2,
        "evt_repair_reclaim_mid": 0.25,
        "evt_repair_reclaim_upper": 0.1,
        "evt_repair_dsa_vwap": 0.15,
        "evt_repair_trend_slope": 0.15,
        "evt_repair_bbmacd": 0.15,
    },
    "failure_weights": {
        "evt_fail_break_lower_no_reclaim": 0.3,
        "evt_fail_trend_down_confirm": 0.25,
        "evt_fail_weak_rebound": 0.2,
        "evt_fail_distribution_risk": 0.25,
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
    阶段引擎：读取事件列，滚动统计事件密度，计算阶段评分。

    Args:
        events_df: detect_panel() 输出的事件 DataFrame
        factors_df: 可选的因子 DataFrame（用于 maturity gating）
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
    maturity_gate = cost_zone_maturity_score.clip(lower=0)
    result["cost_zone_maturity_score"] = cost_zone_maturity_score

    if "evt_wash_cycle_complete" in events_df.columns:
        wash_cycle_events = events_df["evt_wash_cycle_complete"].fillna(0).astype(int)
        wash_cycle_count = wash_cycle_events.cumsum()
        result["wash_cycle_count"] = wash_cycle_count
    else:
        result["wash_cycle_count"] = 0

    wash_quality = pd.Series(0.0, index=events_df.index)
    if "evt_wash_break_short_hold_long" in events_df.columns:
        wash_quality += events_df["evt_wash_break_short_hold_long"].fillna(0).astype(float).rolling(window, min_periods=1).mean() * 0.4
    if "evt_wash_reclaim_mid" in events_df.columns:
        wash_quality += events_df["evt_wash_reclaim_mid"].fillna(0).astype(float).rolling(window, min_periods=1).mean() * 0.3
    if "evt_wash_pullback_to_lower" in events_df.columns:
        wash_quality += events_df["evt_wash_pullback_to_lower"].fillna(0).astype(float).rolling(window, min_periods=1).mean() * 0.3
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

    print("=== 阶段引擎输出 ===")
    print(stage_df.describe())
    print()
    print("stage_guess 分布:")
    print(stage_df["stage_guess"].value_counts())
    print()
    print("最后5行:")
    print(stage_df.tail())
