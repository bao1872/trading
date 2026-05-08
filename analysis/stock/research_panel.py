# -*- coding: utf-8 -*-
"""
analysis/stock/research_panel.py - 标准个股底表

Purpose: 生成统一的标准个股底表，列名统一，因子/事件/状态/属性全部可追踪。

Schema:
    基础行情列: ts_code, trade_date, open, high, low, close, vol, amount
    趋势类因子: dsa_dir, prev_pivot_code, trend_align_momo, dsa_dir_age
    位置类因子: dsa_pivot_pos_01, price_vs_dsa_vwap_pct, ...
    动量类因子: bbmacd, bbmacd_minus_avg, ...
    量能类因子: vol_zscore_20, vol_stage_cv, ...
    协同类因子: price_vol_coord, coord_consistency, ...
    节奏类因子: current_stage_bars, current_stage_ret_pct, ...
    财务类因子: q_rev_yoy, roe_weighted, ...
    结构类因子: stop_cluster_levels, ...
    风险类因子: atr_pct, volatility, ...
    事件标记: evt_dsa_dir_flip_up, evt_break_sell_stop_cluster, ...
    事件强度: evt_dsa_dir_flip_up_strength, ...
    状态标记: state_in_uptrend, state_strong_momentum, ...
    属性标签: cap_tier, industry, concept_list

Usage:
    from analysis.stock.research_panel import build_research_panel
    panel = build_research_panel(df)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from typing import Optional


def load_factors_from_db(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: str = "1d",
    factor_names: Optional[list] = None,
) -> pd.DataFrame:
    """
    从 factor_value 长表读取因子，转为宽表格式。

    Args:
        ts_code: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        freq: 频率 ('1d'/'1w'/'1mo'/'1q')
        factor_names: 指定因子列表，None表示全部

    Returns:
        因子宽表 DataFrame (index=as_of_date, columns=factor_names)
    """
    from sqlalchemy import text
    from datasource.database import get_engine

    engine = get_engine()

    sql = """
        SELECT as_of_date, factor_name, factor_value
        FROM factor_value
        WHERE ts_code = :ts_code AND freq = :freq
    """
    params = {"ts_code": ts_code, "freq": freq}

    if start_date:
        sql += " AND as_of_date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        sql += " AND as_of_date <= :end_date"
        params["end_date"] = end_date
    if factor_names:
        placeholders = ", ".join([f"'{f}'" for f in factor_names])
        sql += f" AND factor_name IN ({placeholders})"

    sql += " ORDER BY as_of_date"

    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        return pd.DataFrame()

    wide = df.pivot(index="as_of_date", columns="factor_name", values="factor_value")
    wide.index = pd.to_datetime(wide.index)
    return wide


def load_events_from_db(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: str = "1d",
    event_names: Optional[list] = None,
) -> pd.DataFrame:
    """
    从 event_trigger 长表读取事件，转为宽表格式。

    Args:
        ts_code: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        freq: 频率
        event_names: 指定事件列表，None表示全部

    Returns:
        事件宽表 DataFrame (index=as_of_date, columns=event_names)
    """
    from sqlalchemy import text
    from datasource.database import get_engine

    engine = get_engine()

    sql = """
        SELECT as_of_date, event_name, triggered, event_strength
        FROM event_trigger
        WHERE ts_code = :ts_code AND freq = :freq
    """
    params = {"ts_code": ts_code, "freq": freq}

    if start_date:
        sql += " AND as_of_date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        sql += " AND as_of_date <= :end_date"
        params["end_date"] = end_date
    if event_names:
        placeholders = ", ".join([f"'{e}'" for e in event_names])
        sql += f" AND event_name IN ({placeholders})"

    sql += " ORDER BY as_of_date"

    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        return pd.DataFrame()

    # 转为 0/1 宽表
    wide = df.pivot(index="as_of_date", columns="event_name", values="triggered")
    wide = wide.fillna(0).astype(float)
    wide.index = pd.to_datetime(wide.index)
    return wide


def build_research_panel_from_db(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: str = "1d",
    factor_names: Optional[list] = None,
    event_names: Optional[list] = None,
) -> pd.DataFrame:
    """
    从数据库构建标准个股底表（替代全量计算）。

    Args:
        ts_code: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        freq: 频率
        factor_names: 指定因子列表，None表示全部
        event_names: 指定事件列表，None表示全部

    Returns:
        标准个股底表 DataFrame
    """
    factors = load_factors_from_db(ts_code, start_date, end_date, freq, factor_names)
    events = load_events_from_db(ts_code, start_date, end_date, freq, event_names)

    # 合并因子和事件
    if not factors.empty and not events.empty:
        panel = factors.join(events, how="outer")
    elif not factors.empty:
        panel = factors.copy()
    elif not events.empty:
        panel = events.copy()
    else:
        return pd.DataFrame()

    panel["ts_code"] = ts_code
    panel = panel.reset_index().rename(columns={"as_of_date": "trade_date"})
    return panel


def build_research_panel(
    df: pd.DataFrame,
    compute_factors: bool = True,
    compute_events: bool = True,
    categories: Optional[list] = None,
    skip_pavp: bool = False,
) -> pd.DataFrame:
    """
    生成标准个股底表。

    Args:
        df: 输入行情 DataFrame（必须包含 open/high/low/close/volume）
        compute_factors: 是否计算因子
        compute_events: 是否检测事件
        categories: 指定计算的因子类别，None表示全部
        skip_pavp: 是否跳过 PAVP 计算（数据不足时可跳过）

    Returns:
        标准个股底表 DataFrame
    """
    import argparse
    import numpy as np

    result = df.copy()

    if compute_factors:
        from factor_lib import compute_panel
        result = compute_panel(result, categories=categories)

        # 补充 PAVP（不在 factor_lib 中）
        if not skip_pavp:
            try:
                from features.pavp_tv_fixed_params_factors import compute_pavp
                pavp_out, _, _ = compute_pavp(result)
                pavp_factor_cols = [c for c in pavp_out.columns if c not in result.columns]
                if pavp_factor_cols:
                    result = pd.concat([result, pavp_out[pavp_factor_cols]], axis=1)
            except Exception:
                pass  # PAVP 数据不足时跳过

        # 补充 Stop Cluster（不在 factor_lib 中）
        try:
            from features.stop_loss_clustering_with_factors import StopLossClusteringEngine
            df_for_stop = result.rename(columns={"volume": "vol"})
            stop_args = argparse.Namespace(
                freq="d", model="dbscan",
                show_historical_triggers=False, max_lines=20,
            )
            engine = StopLossClusteringEngine(df_for_stop, stop_args)
            engine.run()
            stop_df = engine.df
            stop_factor_cols = [c for c in stop_df.columns if c not in result.columns and c != "vol"]
            if stop_factor_cols:
                result = pd.concat([result, stop_df[stop_factor_cols]], axis=1)
        except Exception:
            pass  # Stop Cluster 数据不足时跳过

    if compute_events:
        from event_lib import detect_panel
        result = detect_panel(result)

        # 补充状态列（不在 event_lib 中）
        if "dsa_dir" in result.columns:
            dsa_dir = result["dsa_dir"]
            result["evt_dsa_dir_flip_up"] = ((dsa_dir > 0) & (dsa_dir.shift(1) < 0)).astype(float).fillna(0.0)
            result["evt_dsa_dir_flip_down"] = ((dsa_dir < 0) & (dsa_dir.shift(1) > 0)).astype(float).fillna(0.0)
        if "DSA_VWAP" in result.columns and "close" in result.columns:
            vwap = result["DSA_VWAP"]
            close = result["close"]
            result["state_dsa_above_vwap"] = (close > vwap).astype(float)
            result["state_dsa_below_vwap"] = (close < vwap).astype(float)
        if "dsa_pivot_pos_01" in result.columns:
            result["state_dsa_pivot_pos_01_bucket"] = pd.cut(
                result["dsa_pivot_pos_01"],
                bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=[1, 2, 3, 4, 5],
                include_lowest=True,
            ).astype(float)

        # PAVP 状态列
        if "vah_price" in result.columns and "val_price" in result.columns and "close" in result.columns:
            vah = result["vah_price"]
            val = result["val_price"]
            close = result["close"]
            result["state_inside_value_area"] = ((close >= val) & (close <= vah)).astype(float)

        # Stop Cluster 状态列
        if "dist_to_nearest_sell_stop_atr" in result.columns:
            result["state_near_sell_stop_cluster"] = (result["dist_to_nearest_sell_stop_atr"] < 1.0).astype(float)
        if "dist_to_nearest_buy_stop_atr" in result.columns:
            result["state_near_buy_stop_cluster"] = (result["dist_to_nearest_buy_stop_atr"] < 1.0).astype(float)

    return result


# 标准列名清单（用于验证）
RESEARCH_PANEL_COLUMNS = {
    "基础行情": ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
    "趋势类": ["dsa_dir", "prev_pivot_code", "trend_align_momo", "dsa_dir_age"],
    "位置类": ["dsa_pivot_pos_01", "price_vs_dsa_vwap_pct", "ret_to_last_high_pct", "ret_to_last_low_pct", "current_pullback_from_stage_extreme_pct"],
    "动量类": ["bbmacd", "bbmacd_minus_avg", "bbmacd_state", "bbmacd_band_pos_01", "bbmacd_bandwidth_zscore", "bbmacd_cross_upper", "bbmacd_cross_lower", "bbmacd_sign", "bbmacd_slope_3"],
    "量能类": ["vol_zscore_5", "vol_zscore_10", "vol_zscore_20", "vol_ratio_10", "days_since_vol_spike", "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio"],
    "协同类": ["price_vol_coord", "momo_vol_coord", "low_pos_break_coord", "coord_consistency", "coord_stage_current", "coord_stage_prev", "coord_stage_ratio"],
    "节奏类": ["current_stage_bars", "current_stage_ret_pct", "current_stage_amp_pct", "prev_stage_bars", "prev_stage_amp_pct"],
    "结构类": ["stop_cluster_levels", "support_resistance_zones", "liquidity_pools", "trendline_upper", "trendline_lower", "upper_break", "lower_break"],
    "风险类": ["atr_pct", "volatility_20d", "max_drawdown_60d", "beta"],
    "趋势事件": ["evt_dsa_dir_flip_up", "evt_dsa_dir_flip_down", "evt_cross_above_dsa_vwap", "evt_cross_below_dsa_vwap"],
    "突破事件": ["evt_cross_above_value_area_high", "evt_cross_below_value_area_low"],
    "量能事件": ["evt_up_move_with_vol_spike", "evt_down_move_with_vol_spike", "evt_vol_shrink", "evt_vol_divergence"],
    "动量事件": ["evt_bbmacd_cross_upper", "evt_bbmacd_cross_lower", "evt_macd_golden_cross", "evt_macd_death_cross"],
    "结构事件": ["evt_break_sell_stop_cluster", "evt_break_buy_stop_cluster", "evt_support_broken", "evt_resistance_broken"],
    "基本面事件": ["evt_earnings_acceleration", "evt_earnings_deceleration", "evt_cashflow_improvement", "evt_cashflow_deterioration", "evt_roe_inflection"],
    "复合事件": ["evt_trend_flip_with_volume", "evt_low_with_vol_shrink", "evt_momo_accel_with_vol", "evt_coord_breakout"],
}


def validate_research_panel(df: pd.DataFrame) -> dict:
    """
    验证底表是否包含所有标准列。

    Args:
        df: 底表 DataFrame

    Returns:
        验证结果字典
    """
    result = {"missing": {}, "present": {}, "total_expected": 0, "total_present": 0}

    for category, columns in RESEARCH_PANEL_COLUMNS.items():
        present = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]
        result["present"][category] = present
        result["missing"][category] = missing
        result["total_expected"] += len(columns)
        result["total_present"] += len(present)

    result["coverage"] = result["total_present"] / result["total_expected"] if result["total_expected"] > 0 else 0
    return result
