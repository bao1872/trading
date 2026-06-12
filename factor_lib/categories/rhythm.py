# -*- coding: utf-8 -*-
"""
factor_lib/categories/rhythm.py - 节奏类因子

Purpose: 节奏类因子（时间/阶段）的批量计算与注册。
         基于 features/dsa_bbmacd_24factors_viewer.py 的 compute_dsa 权威实现。

Public API:
    compute_rhythm_factors(df, dsa_result=None) -> DataFrame

Registered Factors:
    - current_stage_bars: 当前阶段K线数
    - current_stage_ret_pct: 当前阶段收益率
    - prev_stage_bars: 前一阶段K线数
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
        DataFrame 含 3 列节奏因子
    """
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, DSAConfig, _run_from_dir

    if dsa_result is None:
        cfg = DSAConfig()
        dsa_result, _, _ = compute_dsa(df, cfg)

    result = pd.DataFrame(index=df.index)
    result["current_stage_bars"] = dsa_result["current_stage_bars"]
    result["prev_stage_bars"] = dsa_result["prev_stage_bars"]

    dsa_dir = dsa_result["dsa_dir"]
    close = df["close"].to_numpy(dtype=float)

    current_stage_ret = np.full(len(df), np.nan)
    runs = _run_from_dir(dsa_dir.to_numpy(float))
    for st, ed, run_dir in runs:
        start_close = float(close[st]) if np.isfinite(close[st]) and close[st] != 0 else np.nan
        if np.isfinite(start_close):
            current_stage_ret[st:ed + 1] = close[st:ed + 1] / start_close - 1.0
    result["current_stage_ret_pct"] = pd.Series(current_stage_ret, index=df.index)

    return result


def _compute_current_stage_bars(df):
    return compute_rhythm_factors(df)["current_stage_bars"]


def _compute_current_stage_ret_pct(df):
    return compute_rhythm_factors(df)["current_stage_ret_pct"]


def _compute_prev_stage_bars(df):
    return compute_rhythm_factors(df)["prev_stage_bars"]


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
    name="prev_stage_bars",
    category="节奏类",
    compute_func=_compute_prev_stage_bars,
    source_module="features.dsa_bbmacd_24factors_viewer",
    source_function="compute_dsa",
    description="前一阶段K线数",
    direction="neutral",
    is_core=False,
)
