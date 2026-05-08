# -*- coding: utf-8 -*-
"""
factor_lib/categories/rhythm.py - 节奏类因子

Purpose: 节奏类因子（时间/幅度/阶段）的批量计算与注册。
         基于 features/dsa_bbmacd_24factors_viewer.py 的 compute_dsa 权威实现。

Public API:
    compute_rhythm_factors(df, dsa_result=None) -> DataFrame

Registered Factors:
    - current_stage_bars: 当前阶段K线数
    - current_stage_ret_pct: 当前阶段收益率
    - current_stage_amp_pct: 当前阶段振幅
    - prev_stage_bars: 前一阶段K线数
    - prev_stage_amp_pct: 前一阶段振幅
"""
from factor_lib.registry import register_factor
import pandas as pd
import numpy as np


def compute_rhythm_factors(
    df: pd.DataFrame,
    dsa_result: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    批量计算全部节奏类因子。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame
        dsa_result: 预计算的 compute_dsa 结果，若为 None 则内部计算

    Returns:
        DataFrame 含 5 列节奏因子
    """
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, DSAConfig, _run_from_dir

    if dsa_result is None:
        cfg = DSAConfig()
        dsa_result, _, _ = compute_dsa(df, cfg)

    result = pd.DataFrame(index=df.index)
    result["current_stage_bars"] = dsa_result["current_stage_bars"]
    result["prev_stage_bars"] = dsa_result["prev_stage_bars"]

    dsa_dir = dsa_result["dsa_dir"]
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    current_stage_ret = np.full(len(df), np.nan)
    runs = _run_from_dir(dsa_dir.to_numpy(float))
    for st, ed, run_dir in runs:
        start_close = float(close[st]) if np.isfinite(close[st]) and close[st] != 0 else np.nan
        if np.isfinite(start_close):
            current_stage_ret[st:ed + 1] = close[st:ed + 1] / start_close - 1.0
    result["current_stage_ret_pct"] = pd.Series(current_stage_ret, index=df.index)

    current_stage_amp = np.full(len(df), np.nan)
    for st, ed, run_dir in runs:
        if run_dir > 0:
            run_low = float(np.nanmin(low[st:ed + 1]))
            roll_high = df["high"].iloc[st:ed + 1].cummax().to_numpy(dtype=float)
            current_stage_amp[st:ed + 1] = (
                (roll_high - run_low) / run_low
                if np.isfinite(run_low) and run_low != 0
                else np.nan
            )
        else:
            run_high = float(np.nanmax(high[st:ed + 1]))
            roll_low = df["low"].iloc[st:ed + 1].cummin().to_numpy(dtype=float)
            current_stage_amp[st:ed + 1] = (
                (run_high - roll_low) / run_high
                if np.isfinite(run_high) and run_high != 0
                else np.nan
            )
    result["current_stage_amp_pct"] = pd.Series(current_stage_amp, index=df.index)

    prev_stage_amp = np.full(len(df), np.nan)
    prev_amp_val = np.nan
    for st, ed, run_dir in runs:
        seg_high = float(np.nanmax(high[st:ed + 1]))
        seg_low = float(np.nanmin(low[st:ed + 1]))
        seg_amp = (seg_high - seg_low) / seg_low if np.isfinite(seg_low) and seg_low != 0 else np.nan
        prev_stage_amp[st:ed + 1] = prev_amp_val
        prev_amp_val = seg_amp
    result["prev_stage_amp_pct"] = pd.Series(prev_stage_amp, index=df.index)
    return result


def _compute_current_stage_bars(df):
    return compute_rhythm_factors(df)["current_stage_bars"]


def _compute_current_stage_ret_pct(df):
    return compute_rhythm_factors(df)["current_stage_ret_pct"]


def _compute_current_stage_amp_pct(df):
    return compute_rhythm_factors(df)["current_stage_amp_pct"]


def _compute_prev_stage_bars(df):
    return compute_rhythm_factors(df)["prev_stage_bars"]


def _compute_prev_stage_amp_pct(df):
    return compute_rhythm_factors(df)["prev_stage_amp_pct"]


# 注册节奏类因子
register_factor(
    name="current_stage_bars",
    category="节奏类",
    compute_func=_compute_current_stage_bars,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="当前阶段K线数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="current_stage_ret_pct",
    category="节奏类",
    compute_func=_compute_current_stage_ret_pct,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="当前阶段收益率",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="current_stage_amp_pct",
    category="节奏类",
    compute_func=_compute_current_stage_amp_pct,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="当前阶段振幅",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="prev_stage_bars",
    category="节奏类",
    compute_func=_compute_prev_stage_bars,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="前一阶段K线数",
    direction="neutral",
    is_core=False,
)

register_factor(
    name="prev_stage_amp_pct",
    category="节奏类",
    compute_func=_compute_prev_stage_amp_pct,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="前一阶段振幅",
    direction="neutral",
    is_core=False,
)
