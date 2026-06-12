# -*- coding: utf-8 -*-
"""
factor_lib/categories/stage_maturity.py - 阶段成熟度因子

Purpose: 价格穿越/触边/收回/循环计数等纯连续型因子的批量计算与注册。
         依赖 stage_position 因子提供统一边界（lower/mid/upper）。
         不输出阶段判断类因子（repair_score/failure_score 已移至 research 层）。

Public API:
    compute_stage_maturity_factors(df, dsa_result=None, bb_result=None) -> DataFrame

Registered Factors:
    - price_cross_mid_count_n: 价格围绕中枢穿越次数
    - upper_touch_count_n: 上沿测试次数
    - lower_touch_count_n: 下沿测试次数
    - lower_reclaim_count_n: 下沿假破/收回次数
    - upper_reject_count_n: 上沿试盘失败次数
    - wash_cycle_count_n: 区间回踩收回循环次数
    - break_lower_intrabar: 当bar跌破下沿
    - reclaim_lower_close: 收盘收回下沿
    - break_last_wash_low: 跌破最近收回低点
    - lower_shadow_ratio: 下影线占比
    - shake_range_atr: bar振幅ATR归一

Inputs: df with open/high/low/close/volume
Outputs: DataFrame with 11 factor columns
How to Run:
    python -m factor_lib.categories.stage_maturity
Examples:
    python -m factor_lib.categories.stage_maturity
Side Effects: None
"""
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np

_TOUCH_WINDOW = 20


def compute_stage_maturity_factors(
    df: pd.DataFrame,
    dsa_result: pd.DataFrame = None,
    bb_result: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    批量计算全部阶段成熟度因子。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame
        dsa_result: 预计算的 compute_dsa 结果，若为 None 则内部计算
        bb_result: 预计算的 compute_bbmacd 结果，若为 None 则内部计算

    Returns:
        DataFrame 含 11 列阶段成熟度因子
    """
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, compute_bbmacd, DSAConfig
    from factor_lib.categories.stage_position import compute_stage_position_factors

    if dsa_result is None:
        cfg = DSAConfig()
        dsa_result, _, _ = compute_dsa(df, cfg)
    if bb_result is None:
        bb_result = compute_bbmacd(df)

    pos_result = compute_stage_position_factors(df, dsa_result=dsa_result)

    result = pd.DataFrame(index=df.index)
    n = len(df)

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    opn = df["open"].to_numpy(dtype=float)
    dsa_atr = dsa_result["dsa_atr"].to_numpy(dtype=float)
    safe_atr = np.where(dsa_atr > 0, dsa_atr, np.nan)

    lower = pos_result["stage_lower_boundary"].to_numpy(dtype=float)
    mid = pos_result["stage_mid_boundary"].to_numpy(dtype=float)
    upper = pos_result["stage_upper_boundary"].to_numpy(dtype=float)

    above_mid = pd.Series(close > mid, index=df.index)
    cross_mid = (above_mid != above_mid.shift(1)).astype(int)
    result["price_cross_mid_count_n"] = cross_mid.rolling(
        _TOUCH_WINDOW, min_periods=3
    ).sum()

    touch_upper = pd.Series(high >= upper, index=df.index)
    result["upper_touch_count_n"] = touch_upper.astype(int).rolling(
        _TOUCH_WINDOW, min_periods=3
    ).sum()

    touch_lower = pd.Series(low <= lower, index=df.index)
    result["lower_touch_count_n"] = touch_lower.astype(int).rolling(
        _TOUCH_WINDOW, min_periods=3
    ).sum()

    break_lower_bar = pd.Series(low < lower, index=df.index)
    reclaim_same_bar = pd.Series(close >= lower, index=df.index)
    lower_reclaim = (break_lower_bar & reclaim_same_bar).astype(int)
    result["lower_reclaim_count_n"] = lower_reclaim.to_frame(
        "x"
    )["x"].rolling(_TOUCH_WINDOW, min_periods=3).sum()

    touch_upper_bar = pd.Series(high > upper, index=df.index)
    reject_upper = (touch_upper_bar & (close <= upper)).astype(int)
    result["upper_reject_count_n"] = pd.Series(
        reject_upper.values, index=df.index
    ).rolling(_TOUCH_WINDOW, min_periods=3).sum()

    upper_touch_s = touch_upper.astype(int)
    lower_reclaim_s = lower_reclaim
    wash_cycle = np.zeros(n, dtype=int)
    cycle_count = 0
    had_upper = False
    had_reclaim = False
    for i in range(n):
        if upper_touch_s.iat[i] > 0:
            had_upper = True
        if lower_reclaim_s.iat[i] > 0 and had_upper:
            had_reclaim = True
        if had_upper and had_reclaim:
            cycle_count += 1
            had_upper = False
            had_reclaim = False
        wash_cycle[i] = cycle_count
    result["wash_cycle_count_n"] = pd.Series(wash_cycle, index=df.index)

    result["break_lower_intrabar"] = break_lower_bar.astype(int)
    result["reclaim_lower_close"] = reclaim_same_bar.astype(int)

    last_wash_low = np.full(n, np.nan)
    wash_low_val = np.nan
    lower_reclaim_vals = lower_reclaim.values
    low_vals = low
    for i in range(n):
        if lower_reclaim_vals[i] > 0 and np.isfinite(low_vals[i]):
            wash_low_val = low_vals[i]
        last_wash_low[i] = wash_low_val
    break_wash_low = np.zeros(n, dtype=int)
    valid_wash = np.isfinite(last_wash_low)
    break_wash_low[valid_wash] = (low_vals[valid_wash] < last_wash_low[valid_wash]).astype(int)
    result["break_last_wash_low"] = pd.Series(break_wash_low, index=df.index)

    bar_range = high - low
    valid_range = bar_range > 0
    shadow = np.full(n, np.nan)
    is_bullish = close >= opn
    shadow[valid_range & is_bullish] = (close[valid_range & is_bullish] - low[valid_range & is_bullish]) / bar_range[valid_range & is_bullish]
    shadow[valid_range & ~is_bullish] = (opn[valid_range & ~is_bullish] - low[valid_range & ~is_bullish]) / bar_range[valid_range & ~is_bullish]
    result["lower_shadow_ratio"] = pd.Series(shadow, index=df.index)

    result["shake_range_atr"] = pd.Series(bar_range / safe_atr, index=df.index)

    return result


def _compute_price_cross_mid_count_n(df):
    return compute_stage_maturity_factors(df)["price_cross_mid_count_n"]


def _compute_upper_touch_count_n(df):
    return compute_stage_maturity_factors(df)["upper_touch_count_n"]


def _compute_lower_touch_count_n(df):
    return compute_stage_maturity_factors(df)["lower_touch_count_n"]


def _compute_lower_reclaim_count_n(df):
    return compute_stage_maturity_factors(df)["lower_reclaim_count_n"]


def _compute_upper_reject_count_n(df):
    return compute_stage_maturity_factors(df)["upper_reject_count_n"]


def _compute_wash_cycle_count_n(df):
    return compute_stage_maturity_factors(df)["wash_cycle_count_n"]


def _compute_break_lower_intrabar(df):
    return compute_stage_maturity_factors(df)["break_lower_intrabar"]


def _compute_reclaim_lower_close(df):
    return compute_stage_maturity_factors(df)["reclaim_lower_close"]


def _compute_break_last_wash_low(df):
    return compute_stage_maturity_factors(df)["break_last_wash_low"]


def _compute_lower_shadow_ratio(df):
    return compute_stage_maturity_factors(df)["lower_shadow_ratio"]


def _compute_shake_range_atr(df):
    return compute_stage_maturity_factors(df)["shake_range_atr"]


register_factor(
    name="price_cross_mid_count_n",
    category="阶段成熟度",
    compute_func=_compute_price_cross_mid_count_n,
    source_module="factor_lib.categories.stage_maturity",
    source_function="_compute_price_cross_mid_count_n",
    description="滚动20日价格围绕中枢穿越次数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="upper_touch_count_n",
    category="阶段成熟度",
    compute_func=_compute_upper_touch_count_n,
    source_module="factor_lib.categories.stage_maturity",
    source_function="_compute_upper_touch_count_n",
    description="滚动20日上沿测试次数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="lower_touch_count_n",
    category="阶段成熟度",
    compute_func=_compute_lower_touch_count_n,
    source_module="factor_lib.categories.stage_maturity",
    source_function="_compute_lower_touch_count_n",
    description="滚动20日下沿测试次数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="lower_reclaim_count_n",
    category="阶段成熟度",
    compute_func=_compute_lower_reclaim_count_n,
    source_module="factor_lib.categories.stage_maturity",
    source_function="_compute_lower_reclaim_count_n",
    description="滚动20日下沿假破/收回次数",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="upper_reject_count_n",
    category="阶段成熟度",
    compute_func=_compute_upper_reject_count_n,
    source_module="factor_lib.categories.stage_maturity",
    source_function="_compute_upper_reject_count_n",
    description="滚动20日上沿试盘失败次数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="wash_cycle_count_n",
    category="阶段成熟度",
    compute_func=_compute_wash_cycle_count_n,
    source_module="factor_lib.categories.stage_maturity",
    source_function="_compute_wash_cycle_count_n",
    description="区间回踩收回循环次数（上沿测试+下沿收回为一轮）",
    direction="neutral",
    is_core=True,
)

register_factor(
    name="break_lower_intrabar",
    category="阶段成熟度",
    compute_func=_compute_break_lower_intrabar,
    source_module="factor_lib.categories.stage_maturity",
    source_function="_compute_break_lower_intrabar",
    description="当bar最低价跌破阶段下沿",
    direction="negative",
    is_core=True,
)

register_factor(
    name="reclaim_lower_close",
    category="阶段成熟度",
    compute_func=_compute_reclaim_lower_close,
    source_module="factor_lib.categories.stage_maturity",
    source_function="_compute_reclaim_lower_close",
    description="收盘价收回阶段下沿",
    direction="positive",
    is_core=True,
)

register_factor(
    name="break_last_wash_low",
    category="阶段成熟度",
    compute_func=_compute_break_last_wash_low,
    source_module="factor_lib.categories.stage_maturity",
    source_function="_compute_break_last_wash_low",
    description="跌破最近收回低点",
    direction="negative",
    is_core=False,
)

register_factor(
    name="lower_shadow_ratio",
    category="阶段成熟度",
    compute_func=_compute_lower_shadow_ratio,
    source_module="factor_lib.categories.stage_maturity",
    source_function="_compute_lower_shadow_ratio",
    description="下影线占比",
    direction="positive",
    is_core=False,
)

register_factor(
    name="shake_range_atr",
    category="阶段成熟度",
    compute_func=_compute_shake_range_atr,
    source_module="factor_lib.categories.stage_maturity",
    source_function="_compute_shake_range_atr",
    description="bar振幅ATR归一",
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

    result = compute_stage_maturity_factors(df)
    print(f"因子列数: {len(result.columns)}")
    print(result.describe())
    print("\nwash_cycle_count_n 最大值:", result["wash_cycle_count_n"].max())
