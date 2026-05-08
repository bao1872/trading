"""
event_detector.py
事件检测器 — 基于 event_lib 注册表统一检测 + 本地状态列补充

Purpose:
    1. 通过 event_lib.detect_panel() 统一检测所有已注册事件（evt_* 列）
    2. 补充本地状态列：PAVP inside_value_area、StopCluster near_stop、DSA above/below VWAP
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
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_lib import detect_panel as event_detect_panel
from event_lib.registry import list_all as list_registered_events

logger = logging.getLogger(__name__)

from market_structure_analysis._config import (
    VOL_ZSCORE_SPIKE_THRESHOLD,
    NEAR_STOP_ATR_THRESHOLD,
)

PIVOT_POS_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PIVOT_POS_LABELS = [1, 2, 3, 4, 5]


def _get_core_events() -> List[str]:
    """从 event_lib 动态获取所有 is_core=True 的事件名列表。"""
    core = []
    for meta in list_registered_events():
        if meta.get("is_core", False):
            core.append(meta["name"])
    return sorted(core)


CORE_EVENTS = _get_core_events()

AUX_STATES: List[str] = [
    "state_dsa_above_vwap",
    "state_dsa_below_vwap",
    "state_dsa_pivot_pos_01_bucket",
    "state_inside_value_area",
    "state_near_sell_stop_cluster",
    "state_near_buy_stop_cluster",
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


def _compute_pavp_states(df: pd.DataFrame) -> pd.DataFrame:
    """计算 PAVP 相关状态列（仅状态列，不产出 evt_*）。"""
    out = pd.DataFrame(index=df.index)

    has_vah = "vah_price" in df.columns
    has_val = "val_price" in df.columns
    has_close = "close" in df.columns

    if has_vah and has_val and has_close:
        vah = df["vah_price"]
        val = df["val_price"]
        close = df["close"]
        out["state_inside_value_area"] = (
            (close >= val) & (close <= vah)
        ).astype(float)

    return out


def _compute_stop_cluster_states(df: pd.DataFrame) -> pd.DataFrame:
    """计算 Stop Cluster 相关状态列（仅状态列，不产出 evt_*）。"""
    out = pd.DataFrame(index=df.index)

    if "dist_to_nearest_sell_stop_atr" in df.columns:
        out["state_near_sell_stop_cluster"] = (
            df["dist_to_nearest_sell_stop_atr"] < NEAR_STOP_ATR_THRESHOLD
        ).astype(float)

    if "dist_to_nearest_buy_stop_atr" in df.columns:
        out["state_near_buy_stop_cluster"] = (
            df["dist_to_nearest_buy_stop_atr"] < NEAR_STOP_ATR_THRESHOLD
        ).astype(float)

    return out


def _compute_dsa_states(df: pd.DataFrame) -> pd.DataFrame:
    """计算 DSA 相关状态列（仅状态列，不产出 evt_*）。"""
    out = pd.DataFrame(index=df.index)

    if "DSA_VWAP" in df.columns and "close" in df.columns:
        vwap = df["DSA_VWAP"]
        close = df["close"]
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
    事件列命名: evt_{event_name}, 值为 0/1
    状态列命名: state_{state_name}, 值为 0.0/1.0 或分桶整数
    检测逻辑分层：
      1. event_lib.detect_panel() 检测所有注册事件
      2. 本地计算状态列（PAVP/StopCluster/DSA）
    """
    result = factors_df.copy()

    logger.debug("Step 1: 通过 event_lib 检测注册事件...")
    result = event_detect_panel(result)

    logger.debug("Step 2: 计算本地状态列...")
    pavp_states = _compute_pavp_states(result)
    stop_states = _compute_stop_cluster_states(result)
    dsa_states = _compute_dsa_states(result)

    all_states = pd.concat([pavp_states, stop_states, dsa_states], axis=1)
    new_cols = [c for c in all_states.columns if c not in result.columns]
    if new_cols:
        result = pd.concat([result, all_states[new_cols]], axis=1)

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
