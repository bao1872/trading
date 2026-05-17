# -*- coding: utf-8 -*-
"""
factor_lib/categories/stage_context.py - 阶段上下文因子

Purpose: 趋势转平/成本区形成相关因子的批量计算与注册。
         基于 features/dsa_bbmacd_24factors_viewer.py 的 compute_dsa / compute_bbmacd 权威实现。

Public API:
    compute_stage_context_factors(df, dsa_result=None, bb_result=None) -> DataFrame

Registered Factors:
    - dsa_vwap_slope_atr_3: DSA VWAP 3-bar斜率，ATR归一
    - dsa_vwap_slope_atr_5: DSA VWAP 5-bar斜率，ATR归一
    - dsa_vwap_slope_abs_rank: 斜率绝对值滚动分位，越低越横盘
    - dsa_dir_flip_count_n: 滚动窗口内方向切换次数
    - trend_flat_score: 趋势转平综合评分

Inputs: df with open/high/low/close/volume
Outputs: DataFrame with 5 factor columns
How to Run:
    python -m factor_lib.categories.stage_context
Examples:
    python -m factor_lib.categories.stage_context
Side Effects: None
"""
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np

_SLOPE_RANK_WINDOW = 60
_DIR_FLIP_WINDOW = 20


def compute_stage_context_factors(
    df: pd.DataFrame,
    dsa_result: pd.DataFrame = None,
    bb_result: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    批量计算全部阶段上下文因子。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame
        dsa_result: 预计算的 compute_dsa 结果，若为 None 则内部计算
        bb_result: 预计算的 compute_bbmacd 结果，若为 None 则内部计算

    Returns:
        DataFrame 含 5 列阶段上下文因子
    """
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, compute_bbmacd, DSAConfig

    if dsa_result is None:
        cfg = DSAConfig()
        dsa_result, _, _ = compute_dsa(df, cfg)
    if bb_result is None:
        bb_result = compute_bbmacd(df)

    result = pd.DataFrame(index=df.index)
    dsa_vwap = dsa_result["DSA_VWAP"].to_numpy(dtype=float)
    dsa_atr = dsa_result["dsa_atr"].to_numpy(dtype=float)
    dsa_dir = dsa_result["dsa_dir"]

    slope3 = np.full(len(df), np.nan)
    slope5 = np.full(len(df), np.nan)
    safe_atr = np.where(dsa_atr > 0, dsa_atr, np.nan)
    slope3[3:] = (dsa_vwap[3:] - dsa_vwap[:-3]) / (3.0 * safe_atr[3:])
    slope5[5:] = (dsa_vwap[5:] - dsa_vwap[:-5]) / (5.0 * safe_atr[5:])
    result["dsa_vwap_slope_atr_3"] = pd.Series(slope3, index=df.index)
    result["dsa_vwap_slope_atr_5"] = pd.Series(slope5, index=df.index)

    slope_abs = np.abs(slope5)
    slope_abs_s = pd.Series(slope_abs, index=df.index)
    result["dsa_vwap_slope_abs_rank"] = slope_abs_s.rolling(
        _SLOPE_RANK_WINDOW, min_periods=5
    ).rank(pct=True)

    is_flip = (dsa_dir != dsa_dir.shift(1)).astype(int)
    result["dsa_dir_flip_count_n"] = is_flip.rolling(
        _DIR_FLIP_WINDOW, min_periods=3
    ).sum()

    slope_rank = result["dsa_vwap_slope_abs_rank"].fillna(0.5)
    flip_count = result["dsa_dir_flip_count_n"].fillna(0)
    flip_norm = flip_count / max(_DIR_FLIP_WINDOW * 0.5, 1)
    flip_norm = np.clip(flip_norm, 0, 1)
    bw_z = bb_result["bbmacd_bandwidth_zscore"].fillna(0).to_numpy(dtype=float)
    bw_norm = np.clip(1.0 - (bw_z + 2.0) / 4.0, 0, 1)
    trend_flat = (
        (1.0 - slope_rank) * 0.4
        + flip_norm * 0.3
        + bw_norm * 0.3
    )
    result["trend_flat_score"] = pd.Series(np.clip(trend_flat, 0, 1), index=df.index)
    return result


def _compute_dsa_vwap_slope_atr_3(df):
    return compute_stage_context_factors(df)["dsa_vwap_slope_atr_3"]


def _compute_dsa_vwap_slope_atr_5(df):
    return compute_stage_context_factors(df)["dsa_vwap_slope_atr_5"]


def _compute_dsa_vwap_slope_abs_rank(df):
    return compute_stage_context_factors(df)["dsa_vwap_slope_abs_rank"]


def _compute_dsa_dir_flip_count_n(df):
    return compute_stage_context_factors(df)["dsa_dir_flip_count_n"]


def _compute_trend_flat_score(df):
    return compute_stage_context_factors(df)["trend_flat_score"]


register_factor(
    name="dsa_vwap_slope_atr_3",
    category="阶段上下文",
    compute_func=_compute_dsa_vwap_slope_atr_3,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="DSA VWAP 3-bar斜率，ATR归一",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="dsa_vwap_slope_atr_5",
    category="阶段上下文",
    compute_func=_compute_dsa_vwap_slope_atr_5,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="DSA VWAP 5-bar斜率，ATR归一",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="dsa_vwap_slope_abs_rank",
    category="阶段上下文",
    compute_func=_compute_dsa_vwap_slope_abs_rank,
    source_module="factor_lib.categories.stage_context",
    source_function="_compute_dsa_vwap_slope_abs_rank",
    description="斜率绝对值滚动分位，越低越横盘",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="dsa_dir_flip_count_n",
    category="阶段上下文",
    compute_func=_compute_dsa_dir_flip_count_n,
    source_module="factor_lib.categories.stage_context",
    source_function="_compute_dsa_dir_flip_count_n",
    description="滚动窗口内方向切换次数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="trend_flat_score",
    category="阶段上下文",
    compute_func=_compute_trend_flat_score,
    source_module="factor_lib.categories.stage_context",
    source_function="_compute_trend_flat_score",
    description="趋势转平综合评分（0-1），越高越横盘",
    direction="neutral",
    is_core=True,
)


if __name__ == "__main__":
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, DSAConfig
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

    result = compute_stage_context_factors(df)
    print(result.describe())
    print("\ntrend_flat_score 分布:")
    print(result["trend_flat_score"].value_counts(bins=5, sort=False))
