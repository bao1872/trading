# -*- coding: utf-8 -*-
"""
factor_lib/categories/stage_position.py - 阶段位置因子

Purpose: 统一下沿/中枢/上沿边界与价格位置因子的批量计算与注册。
         基于 features/dsa_bbmacd_24factors_viewer.py 的 compute_dsa 权威实现。

Public API:
    compute_stage_position_factors(df, dsa_result=None) -> DataFrame

Registered Factors:
    - stage_lower_boundary: 阶段下沿统一抽象
    - stage_mid_boundary: 阶段中枢统一抽象
    - stage_upper_boundary: 阶段上沿统一抽象
    - price_pos_in_stage_01: 当前价在阶段区间内位置（0-1）
    - dist_to_stage_lower_atr: 距下沿ATR归一距离
    - dist_to_stage_mid_atr: 距中枢ATR归一距离
    - dist_to_stage_upper_atr: 距上沿ATR归一距离

Inputs: df with open/high/low/close/volume
Outputs: DataFrame with 7 factor columns
How to Run:
    python -m factor_lib.categories.stage_position
Examples:
    python -m factor_lib.categories.stage_position
Side Effects: None
"""
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np


def compute_stage_position_factors(
    df: pd.DataFrame,
    dsa_result: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    批量计算全部阶段位置因子。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame
        dsa_result: 预计算的 compute_dsa 结果，若为 None 则内部计算

    Returns:
        DataFrame 含 7 列阶段位置因子
    """
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, DSAConfig

    if dsa_result is None:
        cfg = DSAConfig()
        dsa_result, _, _ = compute_dsa(df, cfg)

    result = pd.DataFrame(index=df.index)
    close = df["close"].to_numpy(dtype=float)
    dsa_atr = dsa_result["dsa_atr"].to_numpy(dtype=float)

    last_low = dsa_result["last_confirmed_low"].to_numpy(dtype=float)
    last_high = dsa_result["last_confirmed_high"].to_numpy(dtype=float)
    dsa_vwap = dsa_result["DSA_VWAP"].to_numpy(dtype=float)

    from factor_lib.categories.structure import _compute_sell_stop_cluster, _compute_buy_stop_cluster
    sell_cluster = _compute_sell_stop_cluster(df).to_numpy(dtype=float)
    buy_cluster = _compute_buy_stop_cluster(df).to_numpy(dtype=float)

    lower = np.where(
        np.isfinite(sell_cluster) & np.isfinite(last_low),
        np.minimum(sell_cluster, last_low),
        np.where(np.isfinite(last_low), last_low, sell_cluster),
    )
    upper = np.where(
        np.isfinite(buy_cluster) & np.isfinite(last_high),
        np.maximum(buy_cluster, last_high),
        np.where(np.isfinite(last_high), last_high, buy_cluster),
    )
    mid = dsa_vwap

    result["stage_lower_boundary"] = pd.Series(lower, index=df.index)
    result["stage_mid_boundary"] = pd.Series(mid, index=df.index)
    result["stage_upper_boundary"] = pd.Series(upper, index=df.index)

    range_width = upper - lower
    valid_range = range_width > 0
    pos = np.full(len(df), np.nan)
    pos[valid_range] = (close[valid_range] - lower[valid_range]) / range_width[valid_range]
    pos = np.clip(pos, 0.0, 1.0)
    result["price_pos_in_stage_01"] = pd.Series(pos, index=df.index)

    safe_atr = np.where(dsa_atr > 0, dsa_atr, np.nan)
    result["dist_to_stage_lower_atr"] = pd.Series(
        (close - lower) / safe_atr, index=df.index
    )
    result["dist_to_stage_mid_atr"] = pd.Series(
        (close - mid) / safe_atr, index=df.index
    )
    result["dist_to_stage_upper_atr"] = pd.Series(
        (close - upper) / safe_atr, index=df.index
    )
    return result


def _compute_stage_lower_boundary(df):
    return compute_stage_position_factors(df)["stage_lower_boundary"]


def _compute_stage_mid_boundary(df):
    return compute_stage_position_factors(df)["stage_mid_boundary"]


def _compute_stage_upper_boundary(df):
    return compute_stage_position_factors(df)["stage_upper_boundary"]


def _compute_price_pos_in_stage_01(df):
    return compute_stage_position_factors(df)["price_pos_in_stage_01"]


def _compute_dist_to_stage_lower_atr(df):
    return compute_stage_position_factors(df)["dist_to_stage_lower_atr"]


def _compute_dist_to_stage_mid_atr(df):
    return compute_stage_position_factors(df)["dist_to_stage_mid_atr"]


def _compute_dist_to_stage_upper_atr(df):
    return compute_stage_position_factors(df)["dist_to_stage_upper_atr"]


register_factor(
    name="stage_lower_boundary",
    category="阶段位置",
    compute_func=_compute_stage_lower_boundary,
    source_module="factor_lib.categories.stage_position",
    source_function="_compute_stage_lower_boundary",
    description="阶段下沿统一抽象：min(last_confirmed_low, sell_stop_cluster)",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="stage_mid_boundary",
    category="阶段位置",
    compute_func=_compute_stage_mid_boundary,
    source_module="factor_lib.categories.stage_position",
    source_function="_compute_stage_mid_boundary",
    description="阶段中枢统一抽象：DSA VWAP",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="stage_upper_boundary",
    category="阶段位置",
    compute_func=_compute_stage_upper_boundary,
    source_module="factor_lib.categories.stage_position",
    source_function="_compute_stage_upper_boundary",
    description="阶段上沿统一抽象：max(last_confirmed_high, buy_stop_cluster)",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="price_pos_in_stage_01",
    category="阶段位置",
    compute_func=_compute_price_pos_in_stage_01,
    source_module="factor_lib.categories.stage_position",
    source_function="_compute_price_pos_in_stage_01",
    description="当前价在阶段区间内位置（0-1），0=下沿，1=上沿",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="dist_to_stage_lower_atr",
    category="阶段位置",
    compute_func=_compute_dist_to_stage_lower_atr,
    source_module="factor_lib.categories.stage_position",
    source_function="_compute_dist_to_stage_lower_atr",
    description="距下沿ATR归一距离",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="dist_to_stage_mid_atr",
    category="阶段位置",
    compute_func=_compute_dist_to_stage_mid_atr,
    source_module="factor_lib.categories.stage_position",
    source_function="_compute_dist_to_stage_mid_atr",
    description="距中枢ATR归一距离",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="dist_to_stage_upper_atr",
    category="阶段位置",
    compute_func=_compute_dist_to_stage_upper_atr,
    source_module="factor_lib.categories.stage_position",
    source_function="_compute_dist_to_stage_upper_atr",
    description="距上沿ATR归一距离",
    direction="neutral",
    is_core=False,
)


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

    result = compute_stage_position_factors(df)
    print(result.describe())
    print("\nprice_pos_in_stage_01 范围检查:")
    print(f"  min={result['price_pos_in_stage_01'].min():.4f}, max={result['price_pos_in_stage_01'].max():.4f}")
