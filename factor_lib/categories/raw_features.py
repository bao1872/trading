# -*- coding: utf-8 -*-
"""
factor_lib/categories/raw_features.py - 原始特征列

Purpose: features/ 函数直接输出的原始列批量提取与注册。
         保证向后兼容，统一通过 factor_lib 暴露。

Public API:
    compute_raw_features(df, dsa_result=None, bb_result=None) -> DataFrame

Registered Factors:
    DSA原始列: DSA_VWAP, prev_pivot_type, last_confirmed_high, last_confirmed_low,
              bars_since_last_high, bars_since_last_low, dsa_atr, dsa_ratio
    BBMACD原始列: m_rapida, m_lenta, bbmacd_avg, bbmacd_std, bbmacd_upper,
                 bbmacd_lower, bbmacd_bandwidth, bbmacd_cross_up_lower,
                 bbmacd_cross_down_upper
    Volume原始列: vol_zscore, vol_zscore_mu, vol_zscore_sd
"""
from factor_lib.registry import register_factor


def compute_raw_features(
    df,
    dsa_result=None,
    bb_result=None,
) -> "pd.DataFrame":
    """
    批量提取全部原始特征列。

    Args:
        df: 含 open/high/low/close/volume 列的 DataFrame
        dsa_result: 预计算的 compute_dsa 结果，若为 None 则内部计算
        bb_result: 预计算的 compute_bbmacd 结果，若为 None 则内部计算

    Returns:
        DataFrame 含 22 列原始特征
    """
    import pandas as pd
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, compute_bbmacd, DSAConfig
    from features.volume_zscore_plotly import volume_zscore

    if dsa_result is None:
        cfg = DSAConfig()
        dsa_result, _, _ = compute_dsa(df, cfg)
    if bb_result is None:
        bb_result = compute_bbmacd(df)

    result = pd.DataFrame(index=df.index)
    dsa_cols = [
        "DSA_VWAP", "prev_pivot_type", "last_confirmed_high", "last_confirmed_low",
        "bars_since_last_high", "bars_since_last_low", "dsa_atr", "dsa_ratio",
    ]
    for col in dsa_cols:
        result[col] = dsa_result[col]

    bb_cols = [
        "m_rapida", "m_lenta", "bbmacd_avg", "bbmacd_std",
        "bbmacd_upper", "bbmacd_lower", "bbmacd_bandwidth",
        "bbmacd_cross_up_lower", "bbmacd_cross_down_upper",
    ]
    for col in bb_cols:
        result[col] = bb_result[col]

    z, mu, sd = volume_zscore(df["volume"], 14)
    result["vol_zscore"] = z
    result["vol_zscore_mu"] = mu
    result["vol_zscore_sd"] = sd
    return result


def _compute_DSA_VWAP(df):
    return compute_raw_features(df)["DSA_VWAP"]


def _compute_prev_pivot_type(df):
    return compute_raw_features(df)["prev_pivot_type"]


def _compute_last_confirmed_high(df):
    return compute_raw_features(df)["last_confirmed_high"]


def _compute_last_confirmed_low(df):
    return compute_raw_features(df)["last_confirmed_low"]


def _compute_bars_since_last_high(df):
    return compute_raw_features(df)["bars_since_last_high"]


def _compute_bars_since_last_low(df):
    return compute_raw_features(df)["bars_since_last_low"]


def _compute_dsa_atr(df):
    return compute_raw_features(df)["dsa_atr"]


def _compute_dsa_ratio(df):
    return compute_raw_features(df)["dsa_ratio"]


def _compute_m_rapida(df):
    return compute_raw_features(df)["m_rapida"]


def _compute_m_lenta(df):
    return compute_raw_features(df)["m_lenta"]


def _compute_bbmacd_avg(df):
    return compute_raw_features(df)["bbmacd_avg"]


def _compute_bbmacd_std(df):
    return compute_raw_features(df)["bbmacd_std"]


def _compute_bbmacd_upper(df):
    return compute_raw_features(df)["bbmacd_upper"]


def _compute_bbmacd_lower(df):
    return compute_raw_features(df)["bbmacd_lower"]


def _compute_bbmacd_bandwidth(df):
    return compute_raw_features(df)["bbmacd_bandwidth"]


def _compute_bbmacd_cross_up_lower(df):
    return compute_raw_features(df)["bbmacd_cross_up_lower"]


def _compute_bbmacd_cross_down_upper(df):
    return compute_raw_features(df)["bbmacd_cross_down_upper"]


def _compute_vol_zscore(df):
    return compute_raw_features(df)["vol_zscore"]


def _compute_vol_zscore_mu(df):
    return compute_raw_features(df)["vol_zscore_mu"]


def _compute_vol_zscore_sd(df):
    return compute_raw_features(df)["vol_zscore_sd"]


# 注册所有原始列
register_factor(name="DSA_VWAP", category="原始列", compute_func=_compute_DSA_VWAP,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_dsa",
                description="DSA动态VWAP", direction="neutral", is_core=False)
register_factor(name="prev_pivot_type", category="原始列", compute_func=_compute_prev_pivot_type,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_dsa",
                description="前一个枢轴点类型", direction="neutral", is_core=False)
register_factor(name="last_confirmed_high", category="原始列", compute_func=_compute_last_confirmed_high,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_dsa",
                description="最近确认高点", direction="neutral", is_core=False)
register_factor(name="last_confirmed_low", category="原始列", compute_func=_compute_last_confirmed_low,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_dsa",
                description="最近确认低点", direction="neutral", is_core=False)
register_factor(name="bars_since_last_high", category="原始列", compute_func=_compute_bars_since_last_high,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_dsa",
                description="距上次高点bars", direction="neutral", is_core=False)
register_factor(name="bars_since_last_low", category="原始列", compute_func=_compute_bars_since_last_low,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_dsa",
                description="距上次低点bars", direction="neutral", is_core=False)
register_factor(name="dsa_atr", category="原始列", compute_func=_compute_dsa_atr,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_dsa",
                description="DSA ATR", direction="neutral", is_core=False)
register_factor(name="dsa_ratio", category="原始列", compute_func=_compute_dsa_ratio,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_dsa",
                description="DSA ATR比率", direction="neutral", is_core=False)

register_factor(name="m_rapida", category="原始列", compute_func=_compute_m_rapida,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_bbmacd",
                description="BBMACD快线", direction="neutral", is_core=False)
register_factor(name="m_lenta", category="原始列", compute_func=_compute_m_lenta,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_bbmacd",
                description="BBMACD慢线", direction="neutral", is_core=False)
register_factor(name="bbmacd_avg", category="原始列", compute_func=_compute_bbmacd_avg,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_bbmacd",
                description="BBMACD信号线", direction="neutral", is_core=False)
register_factor(name="bbmacd_std", category="原始列", compute_func=_compute_bbmacd_std,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_bbmacd",
                description="BBMACD标准差", direction="neutral", is_core=False)
register_factor(name="bbmacd_upper", category="原始列", compute_func=_compute_bbmacd_upper,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_bbmacd",
                description="布林带上轨", direction="neutral", is_core=False)
register_factor(name="bbmacd_lower", category="原始列", compute_func=_compute_bbmacd_lower,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_bbmacd",
                description="布林带下轨", direction="neutral", is_core=False)
register_factor(name="bbmacd_bandwidth", category="原始列", compute_func=_compute_bbmacd_bandwidth,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_bbmacd",
                description="布林带带宽", direction="neutral", is_core=False)
register_factor(name="bbmacd_cross_up_lower", category="原始列", compute_func=_compute_bbmacd_cross_up_lower,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_bbmacd",
                description="BBMACD上穿下轨", direction="neutral", is_core=False)
register_factor(name="bbmacd_cross_down_upper", category="原始列", compute_func=_compute_bbmacd_cross_down_upper,
                source_module="features.dsa_bbmacd_24factors_viewer", source_function="compute_bbmacd",
                description="BBMACD下穿上轨", direction="neutral", is_core=False)

register_factor(name="vol_zscore", category="原始列", compute_func=_compute_vol_zscore,
                source_module="features.volume_zscore_plotly", source_function="volume_zscore",
                description="成交量Z-score（14日）", direction="neutral", is_core=False)
register_factor(name="vol_zscore_mu", category="原始列", compute_func=_compute_vol_zscore_mu,
                source_module="features.volume_zscore_plotly", source_function="volume_zscore",
                description="成交量均值", direction="neutral", is_core=False)
register_factor(name="vol_zscore_sd", category="原始列", compute_func=_compute_vol_zscore_sd,
                source_module="features.volume_zscore_plotly", source_function="volume_zscore",
                description="成交量标准差", direction="neutral", is_core=False)
