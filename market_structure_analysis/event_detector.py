"""
event_detector.py
事件检测器 — 基于 advicement.txt 定义的 12 核心事件 + 6 辅助状态

Purpose:
    基于因子 DataFrame 检测结构性事件，包括：
    - 12 个核心事件（触发型，0/1 标记）
    - 6 个辅助状态（背景型，0/1 或分桶值）
    所有检测逻辑为纯 pandas 向量化计算，无外部依赖。

Inputs:
    - factors_df: pd.DataFrame, 由 factor_engine.compute_all_factors() 输出

Outputs:
    - pd.DataFrame, 原始列 + evt_* 事件列 + state_* 状态列

How to Run:
    python market_structure_analysis/event_detector.py

Examples:
    python market_structure_analysis/event_detector.py

Side Effects:
    无（纯计算，不读写数据库或文件）
"""

import logging
import os
import sys
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VOL_SPIKE_THRESHOLD = 2.0
NEAR_STOP_ATR_THRESHOLD = 1.0
PIVOT_POS_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PIVOT_POS_LABELS = [1, 2, 3, 4, 5]

CORE_EVENTS: List[str] = [
    "dsa_dir_flip_up",
    "dsa_dir_flip_down",
    "cross_above_dsa_vwap",
    "cross_below_dsa_vwap",
    "bbmacd_cross_upper",
    "bbmacd_cross_lower",
    "up_move_with_vol_spike",
    "down_move_with_vol_spike",
    "cross_above_value_area_high",
    "cross_below_value_area_low",
    "break_sell_stop_cluster",
    "break_buy_stop_cluster",
]

AUX_STATES: List[str] = [
    "dsa_above_vwap",
    "dsa_below_vwap",
    "dsa_pivot_pos_01_bucket",
    "inside_value_area",
    "near_sell_stop_cluster",
    "near_buy_stop_cluster",
]


def _crossover(series: pd.Series, threshold: pd.Series) -> pd.Series:
    """series 从下方穿越 threshold（前一根 ≤，当前 >）"""
    prev_le = series.shift(1) <= threshold.shift(1)
    curr_gt = series > threshold
    return (prev_le & curr_gt).astype(float)


def _crossunder(series: pd.Series, threshold: pd.Series) -> pd.Series:
    """series 从上方穿越 threshold（前一根 ≥，当前 <）"""
    prev_ge = series.shift(1) >= threshold.shift(1)
    curr_lt = series < threshold
    return (prev_ge & curr_lt).astype(float)


def _flip_up(dir_series: pd.Series) -> pd.Series:
    """方向序列从非多头(≤0)翻转为多头(1)"""
    prev_le0 = dir_series.shift(1) <= 0
    curr_is1 = dir_series == 1
    return (prev_le0 & curr_is1).astype(float)


def _flip_down(dir_series: pd.Series) -> pd.Series:
    """方向序列从非空头(≥0)翻转为空头(-1)"""
    prev_ge0 = dir_series.shift(1) >= 0
    curr_is_neg1 = dir_series == -1
    return (prev_ge0 & curr_is_neg1).astype(float)


def _detect_dsa_events(df: pd.DataFrame) -> pd.DataFrame:
    """检测 DSA 相关事件和状态"""
    out = pd.DataFrame(index=df.index)

    if "dsa_dir" in df.columns:
        out["evt_dsa_dir_flip_up"] = _flip_up(df["dsa_dir"])
        out["evt_dsa_dir_flip_down"] = _flip_down(df["dsa_dir"])

    if "DSA_VWAP" in df.columns and "close" in df.columns:
        vwap = df["DSA_VWAP"]
        close = df["close"]
        out["evt_cross_above_dsa_vwap"] = _crossover(close, vwap)
        out["evt_cross_below_dsa_vwap"] = _crossunder(close, vwap)
        out["state_dsa_above_vwap"] = (close > vwap).astype(float)
        out["state_dsa_below_vwap"] = (close < vwap).astype(float)

    if "dsa_pivot_pos_01" in df.columns:
        out["state_dsa_pivot_pos_01_bucket"] = pd.cut(
            df["dsa_pivot_pos_01"],
            bins=PIVOT_POS_BINS,
            labels=PIVOT_POS_LABELS,
            include_lowest=True,
        ).astype(float)

    return out


def _detect_bbmacd_events(df: pd.DataFrame) -> pd.DataFrame:
    """检测 BBMACD 相关事件"""
    out = pd.DataFrame(index=df.index)

    if "bbmacd_cross_upper" in df.columns:
        out["evt_bbmacd_cross_upper"] = df["bbmacd_cross_upper"].fillna(0)

    if "bbmacd_cross_lower" in df.columns:
        out["evt_bbmacd_cross_lower"] = df["bbmacd_cross_lower"].fillna(0)

    return out


def _detect_volume_events(df: pd.DataFrame) -> pd.DataFrame:
    """检测 Volume Z-score 相关事件"""
    out = pd.DataFrame(index=df.index)

    if "vol_zscore" in df.columns and "close" in df.columns:
        close_up = df["close"] > df["close"].shift(1)
        close_down = df["close"] < df["close"].shift(1)
        vol_spike = df["vol_zscore"] > VOL_SPIKE_THRESHOLD

        out["evt_up_move_with_vol_spike"] = (close_up & vol_spike).astype(float)
        out["evt_down_move_with_vol_spike"] = (close_down & vol_spike).astype(float)

    return out


def _detect_pavp_events(df: pd.DataFrame) -> pd.DataFrame:
    """检测 PAVP 相关事件和状态"""
    out = pd.DataFrame(index=df.index)

    has_vah = "vah_price" in df.columns
    has_val = "val_price" in df.columns
    has_close = "close" in df.columns

    if has_vah and has_close:
        vah = df["vah_price"]
        close = df["close"]
        out["evt_cross_above_value_area_high"] = _crossover(close, vah)

    if has_val and has_close:
        val = df["val_price"]
        close = df["close"]
        out["evt_cross_below_value_area_low"] = _crossunder(close, val)

    if has_vah and has_val and has_close:
        vah = df["vah_price"]
        val = df["val_price"]
        close = df["close"]
        out["state_inside_value_area"] = (
            (close >= val) & (close <= vah)
        ).astype(float)

    return out


def _detect_stop_cluster_events(df: pd.DataFrame) -> pd.DataFrame:
    """检测 Stop Cluster 相关事件和状态"""
    out = pd.DataFrame(index=df.index)

    if "sell_stop_triggered" in df.columns:
        out["evt_break_sell_stop_cluster"] = (df["sell_stop_triggered"] == 1).astype(float)

    if "buy_stop_triggered" in df.columns:
        out["evt_break_buy_stop_cluster"] = (df["buy_stop_triggered"] == 1).astype(float)

    if "dist_to_nearest_sell_stop_atr" in df.columns:
        out["state_near_sell_stop_cluster"] = (
            df["dist_to_nearest_sell_stop_atr"] < NEAR_STOP_ATR_THRESHOLD
        ).astype(float)

    if "dist_to_nearest_buy_stop_atr" in df.columns:
        out["state_near_buy_stop_cluster"] = (
            df["dist_to_nearest_buy_stop_atr"] < NEAR_STOP_ATR_THRESHOLD
        ).astype(float)

    return out


def detect_events(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    检测所有事件和状态，返回事件/状态列追加到 factors_df。

    Parameters
    ----------
    factors_df : pd.DataFrame
        由 factor_engine.compute_all_factors() 输出的因子 DataFrame

    Returns
    -------
    pd.DataFrame
        原始因子列 + evt_* 事件列 + state_* 状态列

    Notes
    -----
    事件列命名: evt_{event_name}, 值为 0.0/1.0
    状态列命名: state_{state_name}, 值为 0.0/1.0 或分桶整数
    缺少依赖因子列时，对应事件/状态列不会生成（而非填 NaN）
    """
    dsa_events = _detect_dsa_events(factors_df)
    bbmacd_events = _detect_bbmacd_events(factors_df)
    volume_events = _detect_volume_events(factors_df)
    pavp_events = _detect_pavp_events(factors_df)
    stop_events = _detect_stop_cluster_events(factors_df)

    all_events = pd.concat(
        [dsa_events, bbmacd_events, volume_events, pavp_events, stop_events],
        axis=1,
    )

    result = pd.concat([factors_df, all_events], axis=1)

    evt_cols = [c for c in result.columns if c.startswith("evt_")]
    state_cols = [c for c in result.columns if c.startswith("state_")]
    logger.info("检测完成: %d 个事件列, %d 个状态列", len(evt_cols), len(state_cols))

    return result


def _build_synthetic_test_data(n: int = 200) -> pd.DataFrame:
    """构建合成测试数据，用于自测事件检测逻辑"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 50.0)

    df = pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.2,
            "high": close + np.abs(np.random.randn(n)) * 0.5,
            "low": close - np.abs(np.random.randn(n)) * 0.5,
            "close": close,
            "volume": np.random.randint(1000, 100000, n).astype(float),
        },
        index=dates,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def main():
    """自测入口：用合成数据验证事件检测逻辑"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from market_structure_analysis.factor_engine import compute_all_factors

    logger.info("构建合成测试数据...")
    df = _build_synthetic_test_data(300)

    logger.info("计算因子...")
    factors = compute_all_factors(df)

    logger.info("检测事件...")
    result = detect_events(factors)

    evt_cols = [c for c in result.columns if c.startswith("evt_")]
    state_cols = [c for c in result.columns if c.startswith("state_")]

    logger.info("事件列 (%d): %s", len(evt_cols), evt_cols)
    logger.info("状态列 (%d): %s", len(state_cols), state_cols)

    for col in evt_cols:
        count = int(result[col].sum())
        if count > 0:
            first_idx = result[result[col] == 1].index[0]
            logger.info("  %s: %d 次, 首次 %s", col, count, first_idx)

    print("\n=== 最近 10 行事件/状态 ===")
    print(result[evt_cols + state_cols].tail(10).to_string())


if __name__ == "__main__":
    main()
