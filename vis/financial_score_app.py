# -*- coding: utf-8 -*-
"""
财务评分与选股结果可视化分析（Streamlit 应用）

Purpose: 展示个股财务评分、选股结果、BSM指标等多维度数据
Inputs:
    - stock_financial_score_pool 表（财务评分）
    - stock_selection_results 表（选股结果：BSM买点信号、Z-Score指标）
    - stock_k_data 表（K线数据，用于BSM指标计算）
    - stock_pools 表（股票名称、概念）
Outputs: Streamlit Web 界面
How to Run: streamlit run vis/financial_score_app.py
Examples:
    streamlit run vis/financial_score_app.py
Side Effects:
    - 仅读取数据库，无写入操作
    - filter_manager 使用 session_state 存储筛选条件
"""
import os
import sys
import importlib.util
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_session, query_sql
from datasource.k_data_loader import load_k_data
from datasource.pytdx_client import connect_pytdx, get_kline_data
from features.dynamic_swing_anchored_vwap import dynamic_swing_anchored_vwap, DSAConfig
from features.atr_rope_with_factors_pytdx_plotly import (
    ATRRopeEngine, UP_COL, DOWN_COL, FLAT_COL, CHANNEL_LINE_COL, CHANNEL_FILL_COL
)
import argparse
import tushare as ts

_kline_cache = {}
_cache_ttl = 300


def init_saved_filters_table(session):
    """确保 saved_filters 表存在"""
    from sqlalchemy import text
    sql = text("""
        CREATE TABLE IF NOT EXISTS saved_filters (
            id SERIAL PRIMARY KEY,
            user_id TEXT DEFAULT 'default',
            page_name TEXT NOT NULL,
            filter_name TEXT NOT NULL,
            conditions TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    session.execute(sql)
    session.commit()


def save_filter(session, page_name: str, filter_name: str, conditions: list):
    """保存筛选条件"""
    import json
    from sqlalchemy import text
    sql = text("""
        INSERT INTO saved_filters (page_name, filter_name, conditions)
        VALUES (:page_name, :filter_name, :conditions)
    """)
    session.execute(sql, {
        "page_name": page_name,
        "filter_name": filter_name,
        "conditions": json.dumps(conditions)
    })
    session.commit()


def load_filters(session, page_name: str) -> pd.DataFrame:
    """加载指定页面的所有筛选配置"""
    import json
    sql = f"""
        SELECT id, filter_name, conditions FROM saved_filters
        WHERE page_name = '{page_name}'
        ORDER BY updated_at DESC
    """
    return query_sql(session, sql, {})


def delete_filter(session, filter_id: int):
    """删除筛选配置"""
    from sqlalchemy import text
    sql = text("DELETE FROM saved_filters WHERE id = :id")
    session.execute(sql, {"id": filter_id})
    session.commit()


def update_filter_name(session, filter_id: int, new_name: str):
    """更新筛选配置名称"""
    from sqlalchemy import text
    sql = text("""
        UPDATE saved_filters SET filter_name = :name, updated_at = CURRENT_TIMESTAMP
        WHERE id = :id
    """)
    session.execute(sql, {"id": filter_id, "name": new_name})
    session.commit()


def render_filter_manager(session, page_name: str, current_conditions: list) -> list:
    """渲染筛选条件管理组件，返回加载的条件或 None"""
    import json

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        with st.expander("💾 保存当前筛选"):
            filter_name = st.text_input("配置名称", key=f"save_name_{page_name}")
            if st.button("保存", key=f"save_btn_{page_name}") and filter_name:
                save_filter(session, page_name, filter_name, current_conditions)
                st.success(f"已保存: {filter_name}")
                st.rerun()

    with col2:
        filters_df = load_filters(session, page_name)
        if filters_df is not None and not filters_df.empty:
            options = [f"{row['filter_name']}" for _, row in filters_df.iterrows()]
            selected = st.selectbox("📂 加载已保存", options=["(选择)"] + options, key=f"load_{page_name}")
            if selected != "(选择)":
                row = filters_df[filters_df["filter_name"] == selected].iloc[0]
                loaded = json.loads(row["conditions"])
                st.info(f"已加载: {selected}")
                return loaded
        else:
            st.info("暂无已保存的筛选")

    with col3:
        if filters_df is not None and not filters_df.empty:
            with st.expander("🗑️ 删除已保存"):
                for _, row in filters_df.iterrows():
                    col_del, col_name = st.columns([1, 3])
                    with col_del:
                        if st.button("删除", key=f"del_{row['id']}"):
                            delete_filter(session, row["id"])
                            st.rerun()
                    with col_name:
                        st.text(row["filter_name"])

    return None

def get_cached_kline(ts_code: str, period: str, count: int = 250) -> pd.DataFrame:
    """获取K线数据，带5分钟缓存"""
    import time

    cache_key = (ts_code, period)
    if cache_key in _kline_cache:
        cached_data, cached_time = _kline_cache[cache_key]
        if time.time() - cached_time < _cache_ttl:
            return cached_data

    df = None

    # 分钟数据用 pytdx（无复权），其他周期用 tushare（前复权）
    if period in ['60m', '30m', '15m', '5m', '1m']:
        # 分钟数据用 pytdx
        try:
            api = connect_pytdx()
            df = get_kline_data(api, ts_code, period, count=count)
            if df is not None and not df.empty:
                df = df.rename(columns={'datetime': 'bar_time'})
                df = df[['bar_time', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"pytdx 获取失败: {e}")
    else:
        # 日/周/月线用 tushare 前复权
        freq_map = {'d': 'D', 'w': 'W', 'm': 'M'}
        freq = freq_map.get(period, 'D')
        try:
            df = ts.pro_bar(ts_code=ts_code, adj='qfq', freq=freq, limit=count)
            if df is not None and not df.empty:
                df = df.rename(columns={'trade_date': 'bar_time', 'vol': 'volume'})
                df = df[['bar_time', 'open', 'high', 'low', 'close', 'volume']]
                df = df.sort_values('bar_time', ascending=True).reset_index(drop=True)
        except Exception as e:
            print(f"tushare 获取失败: {e}")

    _kline_cache[cache_key] = (df, time.time())
    return df


from financial_factors.financial_quarterly_score import FACTOR_CONFIG, DIMENSION_WEIGHTS

DARK_THEME = {
    "bg_color": "#131722",
    "paper_bgcolor": "#131722",
    "grid_color": "rgba(255,255,255,0.06)",
    "text_color": "#d1d4dc",
    "up_color": "#26a69a",
    "down_color": "#ef5350",
}

INDICATOR_CONFIG = {
    "vwap": {
        "name": "VWAP",
        "periods": ["w", "m"],
        "default": True,
    },
    "atr_rope": {
        "name": "ATR Rope",
        "periods": ["d", "60m"],
        "default": True,
    },
}

DIMENSION_COLORS = {
    "边际变化与持续性": "#2962ff",
    "利润质量": "#26a69a",
    "现金创造能力": "#ff9800",
    "资产效率与资金占用": "#e91e63",
    "规模与增长": "#9c27b0",
    "盈利能力": "#00bcd4",
}

DIMENSION_COLS = [
    "边际变化与持续性_score",
    "利润质量_score",
    "现金创造能力_score",
    "资产效率与资金占用_score",
    "规模与增长_score",
    "盈利能力_score",
]


def match_stocks_by_pinyin(pinyin_input: str, stock_list: pd.DataFrame) -> pd.DataFrame:
    """根据拼音首字母匹配股票"""
    if not pinyin_input:
        return pd.DataFrame()

    pinyin_lower = pinyin_input.lower()

    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        return stock_list[stock_list["name"].str.contains(pinyin_input, case=False, na=False)]

    def get_pinyin_initials(name: str) -> str:
        try:
            py_list = lazy_pinyin(name)
            return "".join([p[0].lower() for p in py_list if p])
        except Exception:
            return ""

    stock_list = stock_list.copy()
    stock_list["pinyin_initials"] = stock_list["name"].apply(get_pinyin_initials)

    matched = stock_list[
        stock_list["pinyin_initials"].str.startswith(pinyin_lower, na=False) |
        stock_list["name"].str.contains(pinyin_input, case=False, na=False)
    ]

    return matched


def load_stock_list(session) -> pd.DataFrame:
    """加载股票列表（从股票池）"""
    sql = """
        SELECT ts_code, name
        FROM stock_pools
        ORDER BY name
    """
    try:
        return query_sql(session, sql)
    except Exception:
        return pd.DataFrame()


def load_score_data(session, ts_code: str) -> pd.DataFrame:
    """从 stock_financial_score_pool 加载股票评分数据，缺失季度前向填充保证趋势图连续"""
    sql = """
        SELECT ts_code, stock_name, report_date, ann_date, total_score,
               "边际变化与持续性_score", "利润质量_score", "现金创造能力_score",
               "资产效率与资金占用_score", "规模与增长_score", "盈利能力_score"
        FROM stock_financial_score_pool
        WHERE ts_code = :ts_code
        ORDER BY report_date ASC
    """
    try:
        df = query_sql(session, sql, {"ts_code": ts_code})
        if df.empty:
            return df
        score_cols = [
            "total_score", "边际变化与持续性_score", "利润质量_score", "现金创造能力_score",
            "资产效率与资金占用_score", "规模与增长_score", "盈利能力_score"
        ]
        present_cols = [c for c in score_cols if c in df.columns]
        df[present_cols] = df[present_cols].ffill()
        return df
    except Exception:
        return pd.DataFrame()


def load_factor_scores(session, ts_code: str, report_date: str) -> pd.DataFrame:
    """加载指定报告期的因子分数"""
    sql = f"""
        SELECT * FROM stock_financial_score_pool
        WHERE ts_code = :ts_code AND report_date = :report_date
    """
    try:
        df = query_sql(session, sql, {"ts_code": ts_code, "report_date": report_date})
        return df
    except Exception:
        return pd.DataFrame()


def load_all_scores(session, report_date: Optional[str] = None) -> pd.DataFrame:
    """加载全股池评分数据"""
    try:
        if report_date:
            sql = """
                SELECT * FROM stock_financial_score_pool
                WHERE report_date = :report_date
                ORDER BY CASE WHEN total_score = 'NaN' THEN 1 ELSE 0 END, total_score DESC
            """
            df = query_sql(session, sql, {"report_date": report_date})
        else:
            sql = """
                SELECT * FROM stock_financial_score_pool
                ORDER BY ts_code, report_date DESC
            """
            df = query_sql(session, sql, {})
            df = df.groupby("ts_code").first().reset_index()
            df = df.sort_values("total_score", ascending=False, na_position="last").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def get_available_report_dates(session) -> List[str]:
    """获取所有可用的报告期"""
    sql = "SELECT DISTINCT report_date FROM stock_financial_score_pool ORDER BY report_date DESC"
    try:
        df = query_sql(session, sql, {})
        return df["report_date"].tolist()
    except Exception:
        return []


def get_available_signal_dates(session) -> List[str]:
    """获取信号数据的所有可用日期（合并当日表和rolling表）"""
    sql = """
        SELECT DISTINCT snapshot_date FROM (
            SELECT snapshot_date FROM theme_signals
            UNION
            SELECT snapshot_date FROM theme_signals_rolling
        ) ORDER BY snapshot_date DESC
    """
    try:
        df = query_sql(session, sql, {})
        return df["snapshot_date"].tolist() if not df.empty else []
    except Exception:
        return []


def load_signal_table(session, table_name: str, snapshot_date: str) -> pd.DataFrame:
    """通用信号表加载函数"""
    sql = f"SELECT * FROM {table_name} WHERE snapshot_date = :snapshot_date"
    try:
        return query_sql(session, sql, {"snapshot_date": snapshot_date})
    except Exception:
        return pd.DataFrame()


def load_limit_up_signals(session, snapshot_date: str) -> pd.DataFrame:
    """加载涨停信号"""
    sql = "SELECT * FROM limit_up_signals WHERE snapshot_date = :snapshot_date"
    try:
        return query_sql(session, sql, {"snapshot_date": snapshot_date})
    except Exception:
        return pd.DataFrame()


def colorize_numeric_columns(df: pd.DataFrame, numeric_cols: list, custom_formatters: dict = None) -> pd.DataFrame:
    """对数值列应用颜色渐变渲染（基于每列的极值）"""
    if df.empty or not numeric_cols:
        return df
    available_cols = [c for c in numeric_cols if c in df.columns]
    if not available_cols:
        return df

    def format_with_backup(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        try:
            return f"{float(val):.2f}"
        except (ValueError, TypeError):
            return str(val)

    styler = df.style.background_gradient(
        cmap="RdYlGn",
        subset=available_cols,
        low=0.3,
        high=1.0
    )
    for col in available_cols:
        fmt = custom_formatters.get(col, format_with_backup) if custom_formatters else format_with_backup
        styler = styler.format({col: fmt}, subset=[col])
    return styler


def render_paginated_dataframe(df: pd.DataFrame, numeric_cols: list = None, page_size: int = 100, key: str = "", custom_formatters: dict = None):
    """分页渲染DataFrame，支持颜色渲染"""
    if df.empty:
        st.info("暂无数据")
        return

    total_rows = len(df)
    total_pages = max(1, (total_rows - 1) // page_size + 1)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        page = st.number_input(f"页码 (1-{total_pages})", min_value=1, max_value=total_pages, value=1, key=f"page_{key}")
    with col2:
        st.write(f"共 {total_rows} 条, 第 {page}/{total_pages} 页")
    with col3:
        page_size_select = st.selectbox("每页条数", [50, 100, 200, 500], index=1, key=f"size_{key}")

    start_idx = (page - 1) * page_size_select
    end_idx = min(start_idx + page_size_select, total_rows)
    page_df = df.iloc[start_idx:end_idx]

    if numeric_cols and page_df is not None and not page_df.empty:
        st.dataframe(colorize_numeric_columns(page_df, numeric_cols, custom_formatters), use_container_width=True, hide_index=True)
    else:
        st.dataframe(page_df, use_container_width=True, hide_index=True)


TAB_FIELD_CONFIGS = {
    "主题_当日": {
        "fields": ["theme", "avg_zscore", "strength", "concept_coverage", "stock_count", "total_zscore"],
        "types": {"theme": "string", "avg_zscore": "numeric", "strength": "numeric",
                  "concept_coverage": "numeric", "stock_count": "numeric", "total_zscore": "numeric"},
        "string_fields": ["theme"],
        "display_names": {"theme": "主题", "avg_zscore": "平均Z分", "strength": "强度",
                         "concept_coverage": "概念覆盖率", "stock_count": "股票数", "total_zscore": "总Z分"}
    },
    "主题_rolling": {
        "fields": ["theme", "avg_zscore", "strength", "concept_coverage", "stock_count", "total_zscore"],
        "types": {"theme": "string", "avg_zscore": "numeric", "strength": "numeric",
                  "concept_coverage": "numeric", "stock_count": "numeric", "total_zscore": "numeric"},
        "string_fields": ["theme"],
        "display_names": {"theme": "主题", "avg_zscore": "平均Z分", "strength": "强度",
                         "concept_coverage": "概念覆盖率", "stock_count": "股票数", "total_zscore": "总Z分"}
    },
    "概念_当日": {
        "fields": ["concept", "theme", "avg_zscore", "normalized_strength", "intensity", "anomalous_stock_count", "total_stock_count", "total_zscore"],
        "types": {"concept": "string", "theme": "string", "avg_zscore": "numeric",
                  "normalized_strength": "numeric", "intensity": "numeric",
                  "anomalous_stock_count": "numeric", "total_stock_count": "numeric", "total_zscore": "numeric"},
        "string_fields": ["concept", "theme"],
        "display_names": {"concept": "概念", "theme": "主题", "avg_zscore": "平均Z分",
                         "normalized_strength": "归一化强度", "intensity": "集中度",
                         "anomalous_stock_count": "异动股数", "total_stock_count": "成分股总数", "total_zscore": "总Z分"}
    },
    "概念_rolling": {
        "fields": ["concept", "theme", "avg_zscore", "normalized_strength", "intensity", "anomalous_stock_count", "total_stock_count", "total_zscore"],
        "types": {"concept": "string", "theme": "string", "avg_zscore": "numeric",
                  "normalized_strength": "numeric", "intensity": "numeric",
                  "anomalous_stock_count": "numeric", "total_stock_count": "numeric", "total_zscore": "numeric"},
        "string_fields": ["concept", "theme"],
        "display_names": {"concept": "概念", "theme": "主题", "avg_zscore": "平均Z分",
                         "normalized_strength": "归一化强度", "intensity": "集中度",
                         "anomalous_stock_count": "异动股数", "total_stock_count": "成分股总数", "total_zscore": "总Z分"}
    },
    "个股异动_当日": {
        "fields": ["code", "name", "zscore", "price_change", "themes_str", "concepts_str", "total_score",
                   "边际变化与持续性", "利润质量", "资产效率", "规模与增长", "现金创造能力", "report_date"],
        "types": {"code": "string", "name": "string", "zscore": "numeric", "price_change": "numeric", "themes_str": "string", "concepts_str": "string",
                  "total_score": "numeric", "边际变化与持续性": "numeric", "利润质量": "numeric",
                  "资产效率": "numeric", "规模与增长": "numeric", "现金创造能力": "numeric",
                  "report_date": "string"},
        "string_fields": ["code", "name", "themes_str", "report_date", "概念"],
        "display_names": {"code": "股票代码", "name": "股票名称", "zscore": "Z分数", "price_change": "涨跌幅", "themes_str": "主题", "concepts_str": "概念",
                         "total_score": "总分", "边际变化与持续性": "边际变化与持续性", "利润质量": "利润质量",
                         "资产效率": "资产效率", "规模与增长": "规模与增长", "现金创造能力": "现金创造能力",
                         "report_date": "财报日期"}
    },
    "个股异动_rolling": {
        "fields": ["code", "name", "zscore", "volume_cv", "volume_spearman", "price_change", "themes_str", "concepts_str", "total_score",
                   "边际变化与持续性", "利润质量", "资产效率", "规模与增长", "现金创造能力", "report_date"],
        "types": {"code": "string", "name": "string", "zscore": "numeric", "volume_cv": "numeric",
                  "volume_spearman": "numeric", "price_change": "numeric", "themes_str": "string", "concepts_str": "string", "total_score": "numeric",
                  "边际变化与持续性": "numeric", "利润质量": "numeric",
                  "资产效率": "numeric", "规模与增长": "numeric", "现金创造能力": "numeric",
                  "report_date": "string"},
        "string_fields": ["code", "name", "themes_str", "report_date", "概念"],
        "display_names": {"code": "股票代码", "name": "股票名称", "zscore": "Z分数", "volume_cv": "CV",
                         "volume_spearman": "Spearman", "price_change": "涨跌幅", "themes_str": "主题", "concepts_str": "概念", "total_score": "总分",
                         "边际变化与持续性": "边际变化与持续性", "利润质量": "利润质量",
                         "资产效率": "资产效率", "规模与增长": "规模与增长", "现金创造能力": "现金创造能力",
                         "report_date": "财报日期"}
    },
    "涨停追踪": {
        "fields": ["股票代码", "股票名称", "概念", "连板天数", "连板交易日", "total_score",
                   "边际变化与持续性", "利润质量", "资产效率", "规模与增长", "现金创造能力", "主题", "财报日期"],
        "types": {"股票代码": "string", "股票名称": "string", "概念": "string", "连板天数": "numeric",
                  "连板交易日": "numeric", "total_score": "numeric", "边际变化与持续性": "numeric",
                  "利润质量": "numeric", "资产效率": "numeric", "规模与增长": "numeric",
                  "现金创造能力": "numeric", "主题": "string", "财报日期": "string"},
        "string_fields": ["股票代码", "股票名称", "概念", "主题", "财报日期"],
        "display_names": {"股票代码": "股票代码", "股票名称": "股票名称", "概念": "概念", "连板天数": "几板",
                         "连板交易日": "几天", "total_score": "总分",
                         "边际变化与持续性": "边际变化与持续性", "利润质量": "利润质量", "资产效率": "资产效率",
                         "规模与增长": "规模与增长", "现金创造能力": "现金创造能力",
                         "主题": "主题", "财报日期": "财报日期"}
    },
    "全股池因子表": {
        "fields": ["股票代码", "股票名称", "公告日期", "概念", "total_score",
                   "边际变化与持续性", "利润质量", "资产效率", "规模与增长", "现金创造能力", "盈利能力"],
        "types": {"股票代码": "string", "股票名称": "string", "公告日期": "string", "概念": "string",
                  "total_score": "numeric", "边际变化与持续性": "numeric", "利润质量": "numeric",
                  "资产效率": "numeric", "规模与增长": "numeric", "现金创造能力": "numeric", "盈利能力": "numeric"},
        "string_fields": ["股票代码", "股票名称", "公告日期", "概念"],
        "display_names": {"股票代码": "股票代码", "股票名称": "股票名称", "公告日期": "公告日期", "概念": "概念",
                         "total_score": "总分", "边际变化与持续性": "边际变化与持续性", "利润质量": "利润质量",
                         "资产效率": "资产效率", "规模与增长": "规模与增长", "现金创造能力": "现金创造能力", "盈利能力": "盈利能力"}
    },
    "C2策略选股": {
        "fields": ["symbol", "name", "concepts", "close", "dsa_pivot_pos_01", "signed_vwap_dev_pct",
                   "w_dsa_pivot_pos_01", "bars_since_dir_change", "rope_dir", "rope_slope_atr_5",
                   "bb_pos_01", "bb_width_percentile", "total_score",
                   "边际变化与持续性_score", "利润质量_score", "资产效率与资金占用_score",
                   "规模与增长_score", "现金创造能力_score"],
        "types": {"symbol": "string", "name": "string", "concepts": "string", "close": "numeric",
                  "dsa_pivot_pos_01": "numeric", "signed_vwap_dev_pct": "numeric",
                  "w_dsa_pivot_pos_01": "numeric", "bars_since_dir_change": "numeric",
                  "rope_dir": "numeric", "rope_slope_atr_5": "numeric",
                  "bb_pos_01": "numeric", "bb_width_percentile": "numeric", "total_score": "numeric",
                  "边际变化与持续性_score": "numeric", "利润质量_score": "numeric", "资产效率与资金占用_score": "numeric",
                  "规模与增长_score": "numeric", "现金创造能力_score": "numeric"},
        "string_fields": ["symbol", "name", "concepts"],
        "display_names": {"symbol": "股票代码", "name": "股票名称", "concepts": "概念", "close": "收盘价",
                         "dsa_pivot_pos_01": "日线DSA位置", "signed_vwap_dev_pct": "VWAP偏离度(%)",
                         "w_dsa_pivot_pos_01": "周线DSA位置", "bars_since_dir_change": "趋势转变Bar数",
                         "rope_dir": "Rope方向", "rope_slope_atr_5": "Rope斜率",
                         "bb_pos_01": "布林带位置", "bb_width_percentile": "布林带宽度分位", "total_score": "财务总分",
                         "边际变化与持续性_score": "边际变化与持续性", "利润质量_score": "利润质量", "资产效率与资金占用_score": "资产效率",
                         "规模与增长_score": "规模与增长", "现金创造能力_score": "现金创造能力"}
    },
    "选股主页": {
        "fields": ["ts_code", "stock_name", "concepts", "daily_reversal_buy", "daily_breakout_buy",
                   "weekly_reversal_buy", "weekly_breakout_buy",
                   "daily_bb_width_zscore", "weekly_bb_width_zscore",
                   "daily_vol_zscore", "weekly_vol_zscore", "weekly_vwap_deviation"],
        "types": {"ts_code": "string", "stock_name": "string", "concepts": "string",
                  "daily_reversal_buy": "numeric", "daily_breakout_buy": "numeric",
                  "weekly_reversal_buy": "numeric", "weekly_breakout_buy": "numeric",
                  "daily_bb_width_zscore": "numeric", "weekly_bb_width_zscore": "numeric",
                  "daily_vol_zscore": "numeric", "weekly_vol_zscore": "numeric",
                  "weekly_vwap_deviation": "numeric"},
        "string_fields": ["ts_code", "stock_name", "concepts"],
        "display_names": {"ts_code": "代码", "stock_name": "名称", "concepts": "概念",
                         "daily_reversal_buy": "日线反转", "daily_breakout_buy": "日线突破",
                         "weekly_reversal_buy": "周线反转", "weekly_breakout_buy": "周线突破",
                         "daily_bb_width_zscore": "日线Z分", "weekly_bb_width_zscore": "周线Z分",
                         "daily_vol_zscore": "日线VolZ", "weekly_vol_zscore": "周线VolZ",
                         "weekly_vwap_deviation": "VWAP偏移"}
    },
    "股东画像": {
        "fields": ["holder_name_std", "holder_type", "quality_grade",
                   "sample_stocks", "total_trades", "total_profit_pct", "total_profit_amount", "avg_win_rate",
                   "sample_entry", "entry_win_rate_60", "sample_add", "add_win_rate_60",
                   "total_entry_amount", "total_exit_amount",
                   "style_label", "period_label", "industry_label", "ability_label"],
        "types": {"holder_name_std": "string", "holder_type": "string", "quality_grade": "string",
                  "sample_stocks": "numeric", "total_trades": "numeric", "total_profit_pct": "numeric",
                  "total_profit_amount": "numeric", "avg_win_rate": "numeric",
                  "sample_entry": "numeric", "entry_win_rate_60": "numeric",
                  "sample_add": "numeric", "add_win_rate_60": "numeric",
                  "total_entry_amount": "numeric", "total_exit_amount": "numeric",
                  "style_label": "string", "period_label": "string",
                  "industry_label": "string", "ability_label": "string"},
        "string_fields": ["holder_name_std", "holder_type", "quality_grade", "style_label", "period_label", "industry_label", "ability_label"],
        "display_names": {"holder_name_std": "股东名称", "holder_type": "类型", "quality_grade": "质量等级",
                         "sample_stocks": "涉及股票", "total_trades": "总交易次数",
                         "total_profit_pct": "累计盈亏比例(%)", "total_profit_amount": "累计盈亏金额(万)",
                         "avg_win_rate": "平均胜率(%)",
                         "sample_entry": "入场事件", "entry_win_rate_60": "入场胜率(%)",
                         "sample_add": "加仓事件", "add_win_rate_60": "加仓胜率(%)",
                         "total_entry_amount": "总入场金额(万)", "total_exit_amount": "总出场金额(万)",
                         "style_label": "风格标签", "period_label": "周期标签",
                         "industry_label": "行业标签", "ability_label": "能力标签"}
    },
    "股东变化评价": {
        "fields": ["ts_code", "stock_name", "industry_l2", "label_primary", "total_score",
                   "score_change_neutral", "score_stability_neutral", "score_quality_neutral", "cr10", "delta_cr10"],
        "types": {"ts_code": "string", "stock_name": "string", "industry_l2": "string", "label_primary": "string",
                  "total_score": "numeric", "score_change_neutral": "numeric", "score_stability_neutral": "numeric",
                  "score_quality_neutral": "numeric", "cr10": "numeric", "delta_cr10": "numeric"},
        "string_fields": ["ts_code", "stock_name", "industry_l2", "label_primary"],
        "display_names": {"ts_code": "股票代码", "stock_name": "股票名称", "industry_l2": "行业",
                         "label_primary": "风向标", "total_score": "总分", "score_change_neutral": "结构分",
                         "score_stability_neutral": "稳定分", "score_quality_neutral": "质量分",
                         "cr10": "cr10", "delta_cr10": "delta_cr10"}
    },
    "形态选股": {
        "fields": ["ts_code", "stock_name", "close_chg",
                   "final_launch_score", "launch_type",
                   "quiet_score", "breakout_score", "volume_score",
                   "strength_score", "quality_score", "freshness_score",
                   "total_score",
                   "边际变化与持续性_score", "利润质量_score", "资产效率与资金占用_score",
                   "规模与增长_score", "现金创造能力_score",
                   "amt_ratio_20", "risk_tags", "concepts",
                   "score_report_date"],
        "types": {"final_launch_score": "numeric",
                  "quiet_score": "numeric", "breakout_score": "numeric",
                  "volume_score": "numeric", "strength_score": "numeric",
                  "quality_score": "numeric", "freshness_score": "numeric",
                  "close_chg": "numeric", "amt_ratio_20": "numeric",
                  "total_score": "numeric",
                  "边际变化与持续性_score": "numeric", "利润质量_score": "numeric",
                  "资产效率与资金占用_score": "numeric", "规模与增长_score": "numeric",
                  "现金创造能力_score": "numeric",
                  "ts_code": "string", "stock_name": "string",
                  "launch_type": "string", "risk_tags": "string", "concepts": "string",
                  "score_report_date": "string"},
        "string_fields": ["ts_code", "stock_name", "启动类型", "风险标签", "概念", "财报日期"],
        "display_names": {
            "ts_code": "股票代码", "stock_name": "股票名称",
            "close_chg": "涨跌幅", "final_launch_score": "最终评分", "launch_type": "启动类型",
            "quiet_score": "沉寂", "breakout_score": "突破",
            "volume_score": "量能", "strength_score": "强度",
            "quality_score": "质量", "freshness_score": "新鲜度",
            "amt_ratio_20": "成交额比", "risk_tags": "风险标签", "concepts": "概念",
            "score_report_date": "财报日期", "total_score": "总分",
            "边际变化与持续性_score": "边际变化与持续性", "利润质量_score": "利润质量",
            "资产效率与资金占用_score": "资产效率", "规模与增长_score": "规模与增长",
            "现金创造能力_score": "现金创造能力"
        }
    },
    "翻多事件": {
        "fields": ["ts_code", "name", "event_time", "freq",
                   "breakout_quality_score",
                   "total_score",
                   "边际变化与持续性_score",
                   "price_chg",
                   "breakout_quality_grade",
                   "score_trend_total", "score_candle_total", "score_volume_total", "score_freshness_total",
                   "rope_slope_atr_5", "dist_to_rope_atr", "consolidation_bars",
                   "vol_zscore", "vol_record_days",
                   "利润质量_score", "资产效率与资金占用_score", "规模与增长_score", "现金创造能力_score",
                   "concepts", "score_report_date"],
        "types": {"ts_code": "string", "name": "string", "event_time": "string", "freq": "string",
                  "breakout_quality_score": "numeric", "breakout_quality_grade": "string",
                  "score_trend_total": "numeric", "score_candle_total": "numeric",
                  "score_volume_total": "numeric", "score_freshness_total": "numeric",
                  "rope_slope_atr_5": "numeric", "dist_to_rope_atr": "numeric", "consolidation_bars": "numeric",
                  "vol_zscore": "numeric", "vol_record_days": "numeric",
                  "total_score": "numeric",
                  "边际变化与持续性_score": "numeric", "利润质量_score": "numeric",
                  "资产效率与资金占用_score": "numeric", "规模与增长_score": "numeric",
                  "现金创造能力_score": "numeric",
                  "price_chg": "numeric",
                  "concepts": "string", "score_report_date": "string"},
        "string_fields": ["ts_code", "name", "event_time", "freq", "breakout_quality_grade", "concepts", "score_report_date"],
        "display_names": {"ts_code": "股票代码", "name": "股票名称", "event_time": "事件时间", "freq": "周期",
                         "breakout_quality_score": "突破质量分",
                         "total_score": "财务总分",
                         "边际变化与持续性_score": "边际变化与持续性",
                         "price_chg": "涨跌幅",
                         "breakout_quality_grade": "等级",
                         "score_trend_total": "趋势评分", "score_candle_total": "K线质量评分",
                         "score_volume_total": "量能评分", "score_freshness_total": "新鲜度评分",
                         "rope_slope_atr_5": "rope斜率", "dist_to_rope_atr": "距rope(ATR)",
                         "consolidation_bars": "盘整周期", "vol_zscore": "量能Z分", "vol_record_days": "量能记录日",
                         "利润质量_score": "利润质量", "资产效率与资金占用_score": "资产效率",
                         "规模与增长_score": "规模与增长", "现金创造能力_score": "现金创造能力",
                         "concepts": "概念", "score_report_date": "财报日期"}
    },
    "回踩买点": {
        "fields": ["ts_code", "name", "buy_time", "freq", "buy_type",
                   "breakout_quality_score",
                   "total_score",
                   "边际变化与持续性_score",
                   "price_chg",
                   "breakout_to_buy_bars",
                   "score_trend_total", "score_candle_total", "score_volume_total", "score_freshness_total",
                   "pullback_touch_support_flag", "pullback_hhhl_seen_flag",
                   "lower", "rope", "close",
                   "利润质量_score", "资产效率与资金占用_score", "规模与增长_score", "现金创造能力_score",
                   "concepts", "score_report_date"],
        "types": {"ts_code": "string", "name": "string", "buy_time": "string", "freq": "string", "buy_type": "string",
                  "breakout_quality_score": "numeric", "breakout_to_buy_bars": "numeric",
                  "score_trend_total": "numeric", "score_candle_total": "numeric",
                  "score_volume_total": "numeric", "score_freshness_total": "numeric",
                  "pullback_touch_support_flag": "numeric", "pullback_hhhl_seen_flag": "numeric",
                  "lower": "numeric", "rope": "numeric", "close": "numeric",
                  "total_score": "numeric",
                  "边际变化与持续性_score": "numeric", "利润质量_score": "numeric",
                  "资产效率与资金占用_score": "numeric", "规模与增长_score": "numeric",
                  "现金创造能力_score": "numeric",
                  "price_chg": "numeric",
                  "concepts": "string", "score_report_date": "string"},
        "string_fields": ["ts_code", "name", "buy_time", "freq", "buy_type", "concepts", "score_report_date"],
        "display_names": {"ts_code": "股票代码", "name": "股票名称", "buy_time": "买入时间", "freq": "周期", "buy_type": "买入类型",
                         "breakout_quality_score": "突破质量分",
                         "total_score": "财务总分",
                         "边际变化与持续性_score": "边际变化与持续性",
                         "price_chg": "涨跌幅",
                         "breakout_to_buy_bars": "间隔bar数",
                         "score_trend_total": "趋势评分", "score_candle_total": "K线质量评分",
                         "score_volume_total": "量能评分", "score_freshness_total": "新鲜度评分",
                         "pullback_touch_support_flag": "回踩支撑", "pullback_hhhl_seen_flag": "HH/HL确认",
                         "lower": "lower", "rope": "rope", "close": "收盘价",
                         "利润质量_score": "利润质量", "资产效率与资金占用_score": "资产效率",
                         "规模与增长_score": "规模与增长", "现金创造能力_score": "现金创造能力",
                         "concepts": "概念", "score_report_date": "财报日期"}
    },
}


def apply_filters(df: pd.DataFrame, conditions: list, config: dict) -> pd.DataFrame:
    """根据条件列表筛选DataFrame，支持中英文列名映射"""
    if not conditions or df.empty:
        return df
    result = df.copy()
    display_names = config.get("display_names", {})
    field_to_display = display_names
    display_to_field = {v: k for k, v in display_names.items()}

    for cond in conditions:
        field = cond["field"]
        operator = cond["operator"]
        value = cond["value"]
        value2 = cond.get("value2")

        actual_field = field
        if field not in result.columns:
            if field in field_to_display:
                actual_field = field_to_display[field]
            elif field in display_to_field:
                actual_field = display_to_field[field]

        if actual_field not in result.columns:
            continue

        if config["types"].get(field) == "string":
            if operator == "等于" and value:
                result = result[result[actual_field].astype(str) == str(value)]
            elif operator == "包含" and value:
                result = result[result[actual_field].astype(str).str.contains(str(value), case=False, na=False)]
            elif operator == "不包含" and value:
                result = result[~result[actual_field].astype(str).str.contains(str(value), case=False, na=False)]
        else:
            result[actual_field] = pd.to_numeric(result[actual_field], errors="coerce")
            if operator == "大于":
                result = result[result[actual_field] > value]
            elif operator == "小于":
                result = result[result[actual_field] < value]
            elif operator == "大于等于":
                result = result[result[actual_field] >= value]
            elif operator == "小于等于":
                result = result[result[actual_field] <= value]
            elif operator == "等于":
                result = result[result[actual_field] == value]
            elif operator == "区间" and value is not None and value2 is not None:
                result = result[(result[actual_field] >= value) & (result[actual_field] <= value2)]

    return result


# ==================== 选股结果数据层函数 ====================

def get_available_selection_dates(session) -> List[str]:
    """获取所有可用的选股日期"""
    sql = """
        SELECT DISTINCT selection_date FROM stock_selection_results
        ORDER BY selection_date DESC
    """
    try:
        df = query_sql(session, sql, {})
        return df["selection_date"].tolist() if not df.empty else []
    except Exception:
        return []


def load_selection_results(session, selection_date: str = None) -> pd.DataFrame:
    """
    加载选股结果数据

    Args:
        session: 数据库会话
        selection_date: 选股日期，为None时取最新日期

    Returns:
        DataFrame包含选股结果
    """
    # 如果没有指定日期，获取最新日期
    if selection_date is None:
        dates = get_available_selection_dates(session)
        if not dates:
            return pd.DataFrame()
        selection_date = dates[0]

    sql = """
        WITH current_selection AS (
            SELECT
                selection_date,
                ts_code,
                stock_name,
                report_date,
                total_score,
                margin_score,
                scale_growth_score,
                profitability_score,
                profit_quality_score,
                cash_creation_score,
                asset_efficiency_score,
                q_rev_yoy_delta,
                q_np_parent_yoy_delta,
                trend_consistency,
                ann_date,
                daily_reversal_buy,
                daily_breakout_buy,
                weekly_reversal_buy,
                weekly_breakout_buy,
                daily_bb_width_zscore,
                weekly_bb_width_zscore,
                daily_vol_zscore,
                weekly_vol_zscore,
                created_at
            FROM stock_selection_results
            WHERE selection_date = :selection_date
        )
        SELECT
            c.selection_date,
            c.ts_code,
            c.stock_name,
            c.report_date,
            c.total_score,
            c.margin_score,
            c.scale_growth_score,
            c.profitability_score,
            c.profit_quality_score,
            c.cash_creation_score,
            c.asset_efficiency_score,
            c.q_rev_yoy_delta,
            c.q_np_parent_yoy_delta,
            c.trend_consistency,
            c.ann_date,
            c.daily_reversal_buy,
            c.daily_breakout_buy,
            c.weekly_reversal_buy,
            c.weekly_breakout_buy,
            c.daily_bb_width_zscore,
            c.daily_vol_zscore,
            c.created_at,
            -- 如果当前日期没有周线数据，用最近一次周线买点的值填充
            COALESCE(c.weekly_bb_width_zscore, lw.weekly_bb_width_zscore) as weekly_bb_width_zscore,
            COALESCE(c.weekly_vol_zscore, lw.weekly_vol_zscore) as weekly_vol_zscore
        FROM current_selection c
        LEFT JOIN LATERAL (
            SELECT weekly_bb_width_zscore, weekly_vol_zscore
            FROM stock_selection_results
            WHERE ts_code = c.ts_code
            AND selection_date < :selection_date
            AND (weekly_reversal_buy = TRUE OR weekly_breakout_buy = TRUE)
            ORDER BY selection_date DESC
            LIMIT 1
        ) lw ON TRUE
        ORDER BY c.margin_score DESC, c.total_score DESC
    """
    try:
        df = query_sql(session, sql, {"selection_date": selection_date})
        return df
    except Exception as e:
        st.error(f"加载选股结果失败: {e}")
        return pd.DataFrame()


def load_stock_bsm_data(ts_code: str, period: str = 'd', bars: int = 120) -> pd.DataFrame:
    """
    加载并计算单只股票的BSM指标数据

    Args:
        ts_code: 股票代码
        period: 周期 ('d'=日线, 'w'=周线)
        bars: K线数量

    Returns:
        DataFrame包含K线和BSM指标
    """
    from sqlalchemy import create_engine, text
    from config import DATABASE_URL

    engine = create_engine(DATABASE_URL)

    sql = """
        SELECT bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = :freq
        ORDER BY bar_time DESC
        LIMIT :bars
    """
    try:
        df = pd.read_sql(text(sql), engine, params={
            'ts_code': ts_code,
            'freq': period,
            'bars': bars
        })
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values('bar_time').reset_index(drop=True)

        # 计算BSM指标
        df = compute_bbmacd_for_chart(df)
        return df
    except Exception as e:
        return pd.DataFrame()


def pine_ema(src: pd.Series, length: int) -> pd.Series:
    """Pine Script风格的EMA计算"""
    arr = src.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if length <= 0 or len(arr) == 0:
        return pd.Series(out, index=src.index)
    alpha = 2.0 / (length + 1.0)
    prev = np.nan
    valid_count = 0
    seed_vals = []
    seeded = False
    for i, val in enumerate(arr):
        if np.isnan(val):
            continue
        valid_count += 1
        if not seeded:
            seed_vals.append(val)
            if valid_count >= length:
                prev = float(np.mean(seed_vals[-length:]))
                out[i] = prev
                seeded = True
            continue
        prev = alpha * val + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=src.index)


def compute_bbmacd_for_chart(df: pd.DataFrame, rapida: int = 8, lenta: int = 26,
                              stdv: float = 0.8, signal_len: int = 9) -> pd.DataFrame:
    """计算BSM指标（Bollinguer sobre Macd）"""
    out = df.copy()
    out["m_rapida"] = pine_ema(out["close"], rapida)
    out["m_lenta"] = pine_ema(out["close"], lenta)
    out["bbmacd"] = out["m_rapida"] - out["m_lenta"]
    out["avg"] = pine_ema(out["bbmacd"], signal_len)
    out["sdev"] = out["bbmacd"].rolling(signal_len, min_periods=signal_len).std(ddof=1)
    out["banda_supe"] = out["avg"] + stdv * out["sdev"]
    out["banda_inf"] = out["avg"] - stdv * out["sdev"]
    return out


# ==================== 原自选股功能（保留） ====================

def load_watchlist(session) -> list:
    """从数据库加载自选股列表"""
    sql = "SELECT ts_code, name FROM stock_watchlist ORDER BY sort_order, added_date DESC"
    return query_sql(session, sql)


def add_to_watchlist(session, ts_code: str, name: str) -> bool:
    """添加股票到自选股"""
    from sqlalchemy import text
    sql = """
        INSERT INTO stock_watchlist (ts_code, name)
        VALUES (:ts_code, :name)
        ON CONFLICT (ts_code) DO NOTHING
    """
    try:
        session.execute(text(sql), {"ts_code": ts_code, "name": name})
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False


def remove_from_watchlist(session, ts_code: str) -> bool:
    """从自选股删除股票"""
    from datasource.database import delete_by_filter
    try:
        delete_by_filter(session, "stock_watchlist", {"ts_code": ts_code})
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False


DARK_THEME = {
    "bg_color": "#131722",
    "paper_bgcolor": "#131722",
    "grid_color": "rgba(255,255,255,0.06)",
    "text_color": "#d1d4dc",
    "up_color": "#26a69a",
    "down_color": "#ef5350",
}


def render_stock_profile_page(session, ts_code: str, name: str):
    """渲染个股属性页面"""
    st.subheader(f"{name} 个股属性")

    fs_df = get_latest_financial_scores(session)
    stock_fs = fs_df[fs_df["ts_code"] == ts_code]

    if stock_fs.empty:
        st.warning("暂无财务评分数据")
    else:
        latest = stock_fs.iloc[0]
        total_score = latest.get("total_score", 0)
        st.metric("财务总分", f"{total_score:.1f}" if pd.notna(total_score) else "N/A")

        dim_fig = render_dimension_chart(stock_fs, latest["report_date"])
        if dim_fig:
            st.plotly_chart(dim_fig, use_container_width=True)

    sql = "SELECT concepts FROM stock_pools WHERE ts_code = :ts_code"
    try:
        concept_df = query_sql(session, sql, {"ts_code": ts_code})
        if concept_df is not None and not concept_df.empty:
            concepts_str = concept_df.iloc[0]["concepts"]
            stock_concepts = [c.strip() for c in concepts_str.split(";") if c.strip()] if concepts_str else []
            if stock_concepts:
                st.markdown("**概念:**")
                st.write(", ".join(stock_concepts))
            else:
                st.info("暂无概念数据")
        else:
            st.info("暂无概念数据")
    except Exception as e:
        st.error(f"概念查询失败: {e}")

    st.markdown("---")
    st.markdown("### 十大流通股东评价")

    try:
        sql = "SELECT * FROM stock_top10_holder_eval_scores_tushare WHERE ts_code = :ts_code ORDER BY report_date DESC LIMIT 1"
        holder_df = query_sql(session, sql, {"ts_code": ts_code})

        if holder_df is not None and not holder_df.empty:
            row = holder_df.iloc[0]

            cols = st.columns(5)
            total = row.get("total_score")
            cols[0].metric("总分", f"{total:.1f}" if pd.notna(total) else "N/A")

            score_change = row.get("score_change_neutral")
            cols[1].metric("结构分", f"{score_change:.1f}" if pd.notna(score_change) else "N/A")

            score_stability = row.get("score_stability_neutral")
            cols[2].metric("稳定分", f"{score_stability:.1f}" if pd.notna(score_stability) else "N/A")

            score_quality = row.get("score_quality_neutral")
            cols[3].metric("质量分", f"{score_quality:.1f}" if pd.notna(score_quality) else "N/A")

            label = row.get("label_primary", "N/A")
            cols[4].metric("风向标", str(label) if pd.notna(label) else "N/A")

            cr10 = row.get("cr10")
            delta_cr10 = row.get("delta_cr10")
            if pd.notna(cr10) and pd.notna(delta_cr10):
                st.markdown(f"**前十股东持股比例**: {cr10:.1%} (环比 {delta_cr10:+.1%})")

            entry = row.get("entry_count", 0)
            exit = row.get("exit_count", 0)
            add = row.get("add_count", 0)
            reduce = row.get("reduce_count", 0)
            st.markdown(f"**股东进出**: 新进{int(entry) if pd.notna(entry) else 0} 退出{int(exit) if pd.notna(exit) else 0} 加仓{int(add) if pd.notna(add) else 0} 减仓{int(reduce) if pd.notna(reduce) else 0}")
        else:
            st.info("暂无股东评价数据")
    except Exception as e:
        st.error(f"股东评价查询失败: {e}")


def build_kline_chart(df: pd.DataFrame, stock_name: str, show_vwap: bool = False, show_atr_rope: bool = False) -> go.Figure:
    """构建K线图表"""
    if df.empty:
        return go.Figure()

    x_values = df.index.strftime('%Y-%m-%d %H:%M').tolist()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False,
        vertical_spacing=0.03, row_heights=[0.7, 0.3],
        subplot_titles=(f"{stock_name} K线", "成交量")
    )

    colors = ["#FF0000" if df["close"].iloc[i] >= df["open"].iloc[i] else "#00FF00" for i in range(len(df))]

    fig.add_trace(go.Candlestick(
        x=x_values,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing_line_color="#FF0000",
        decreasing_line_color="#00FF00",
        increasing_fillcolor="rgba(255,0,0,0)",
        decreasing_fillcolor="rgba(0,255,0,1)",
        name="K线"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=x_values,
        y=df["volume"],
        marker_color=colors,
        opacity=1.0,
        name="成交量"
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        plot_bgcolor=DARK_THEME["bg_color"],
        hovermode='x unified'
    )

    fig.update_xaxes(type='category', showspikes=True, spikecolor="rgba(255,255,255,0.3)",
                     spikethickness=1, spikemode="across", showticklabels=True, row=1, col=1)
    fig.update_xaxes(type='category', showspikes=True, spikecolor="rgba(255,255,255,0.3)",
                     spikethickness=1, spikemode="across", showticklabels=False, row=2, col=1)

    if show_vwap:
        cfg = DSAConfig(prd=50, baseAPT=20, useAdapt=False, volBias=10)
        try:
            vwap_series, dir_series, pivot_labels, segments = dynamic_swing_anchored_vwap(df, cfg)
            for seg in segments:
                seg_dir = seg["dir"]
                color = "#ff1744" if seg_dir > 0 else "#00e676"
                x_str = [pd.Timestamp(t).strftime('%Y-%m-%d %H:%M') for t in seg["x"]]
                fig.add_trace(
                    go.Scatter(x=x_str, y=seg["y"], mode="lines",
                               line=dict(width=2, color=color),
                               showlegend=False),
                    row=1, col=1
                )
            for lab in pivot_labels:
                txt = lab["text"]
                if not txt or lab["x"] not in df.index:
                    continue
                is_up = lab["dir"] > 0
                bgcolor = "rgba(0,230,118,0.85)" if is_up else "rgba(255,23,68,0.85)"
                ay = 25 if is_up else -25
                x_str = pd.Timestamp(lab["x"]).strftime('%Y-%m-%d %H:%M')
                fig.add_annotation(
                    x=x_str, y=lab["y"], xref="x", yref="y",
                    text=txt, showarrow=True, arrowhead=2, ax=0, ay=ay,
                    bgcolor=bgcolor, font=dict(color="white", size=12),
                    bordercolor=bgcolor, borderwidth=1, row=1, col=1
                )
        except Exception as e:
            print(f"VWAP计算失败: {e}")

    if show_atr_rope:
        freq_type = 'd'
        if len(df) > 0:
            idx = df.index[0]
            if hasattr(idx, 'hour') and idx.hour != 0:
                freq_type = '60m'
        args = argparse.Namespace(
            len=14, multi=1.5, source='close', freq=freq_type,
            show_factor_panels=False, show_atr_channel=True, show_ranges=True,
            show_break_markers=False
        )
        try:
            engine = ATRRopeEngine(df, args)
            engine.run()

            rope_colors = {
                "rope_up": "rgba(61, 170, 69, 0.6)",
                "rope_down": "rgba(255, 3, 62, 0.6)",
                "rope_flat": "rgba(0, 77, 146, 0.6)"
            }
            for col_name, color in rope_colors.items():
                fig.add_trace(go.Scatter(
                    x=x_values, y=engine.df[col_name], mode="lines",
                    line=dict(width=2.5, color=color, dash="dash"),
                    hoverinfo="skip", showlegend=False
                ), row=1, col=1)

            if engine.args.show_atr_channel:
                dir_arr = engine.df["rope_dir"].to_numpy()
                upper_arr = engine.df["upper"].to_numpy()
                lower_arr = engine.df["lower"].to_numpy()

                cons_mask = (dir_arr == 0) & np.isfinite(upper_arr) & np.isfinite(lower_arr)
                trend_mask = (dir_arr != 0) & np.isfinite(upper_arr) & np.isfinite(lower_arr)

                cons_ranges = []
                start = None
                for i, m in enumerate(cons_mask):
                    if m and start is None:
                        start = i
                    elif not m and start is not None:
                        cons_ranges.append((start, i - 1))
                        start = None
                if start is not None:
                    cons_ranges.append((start, len(cons_mask) - 1))

                trend_ranges = []
                start = None
                for i, m in enumerate(trend_mask):
                    if m and start is None:
                        start = i
                    elif not m and start is None:
                        start = i
                    elif not m and start is not None:
                        trend_ranges.append((start, i - 1))
                        start = None
                if start is not None:
                    trend_ranges.append((start, len(trend_mask) - 1))

                for s, e in trend_ranges:
                    fig.add_trace(go.Scatter(
                        x=x_values[s:e+1], y=upper_arr[s:e+1], mode="lines",
                        line=dict(width=1.5, color=CHANNEL_LINE_COL, dash="dash"),
                        hoverinfo="skip", showlegend=False
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=x_values[s:e+1], y=lower_arr[s:e+1], mode="lines",
                        line=dict(width=1.5, color=CHANNEL_LINE_COL, dash="dash"),
                        hoverinfo="skip", showlegend=False
                    ), row=1, col=1)

                for s, e in cons_ranges:
                    fig.add_trace(go.Scatter(
                        x=x_values[s:e+1], y=upper_arr[s:e+1], mode="lines",
                        line=dict(width=3, color="rgba(255, 255, 0, 1)"),
                        hoverinfo="skip", showlegend=False
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=x_values[s:e+1], y=lower_arr[s:e+1], mode="lines",
                        line=dict(width=3, color="rgba(255, 255, 0, 1)"),
                        hoverinfo="skip", showlegend=False
                    ), row=1, col=1)
        except Exception as e:
            print(f"ATR Rope计算失败: {e}")

    return fig


# ==================== BSM指标图表组件 ====================

BUY_SIGNAL_COLORS = {
    "daily_reversal_buy": "#26a69a",    # 绿色 - 日线一类
    "daily_breakout_buy": "#2962ff",    # 蓝色 - 日线二类
    "weekly_reversal_buy": "#9c27b0",   # 紫色 - 周线一类
    "weekly_breakout_buy": "#ff9800",   # 橙色 - 周线二类
}

BUY_SIGNAL_NAMES = {
    "daily_reversal_buy": "日线反转",
    "daily_breakout_buy": "日线突破",
    "weekly_reversal_buy": "周线反转",
    "weekly_breakout_buy": "周线突破",
}


def build_kline_chart_with_bsm(df: pd.DataFrame, stock_name: str,
                                show_vwap: bool = False,
                                show_atr_rope: bool = False,
                                show_bsm: bool = True) -> go.Figure:
    """
    构建带BSM指标的K线图表

    Args:
        df: K线数据DataFrame（需包含BSM计算后的列）
        stock_name: 股票名称
        show_vwap: 是否显示VWAP
        show_atr_rope: 是否显示ATR Rope
        show_bsm: 是否显示BSM指标副图

    Returns:
        Plotly Figure对象
    """
    if df.empty:
        return go.Figure()

    has_bsm = show_bsm and 'bbmacd' in df.columns and 'banda_supe' in df.columns

    # 确定子图布局
    if has_bsm:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=(f"{stock_name} K线", "成交量", "BSM指标")
        )
    else:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False,
            vertical_spacing=0.03, row_heights=[0.7, 0.3],
            subplot_titles=(f"{stock_name} K线", "成交量")
        )

    x_values = df.index.strftime('%Y-%m-%d %H:%M').tolist() if hasattr(df.index, 'strftime') else list(range(len(df)))
    if hasattr(df.index, 'strftime'):
        x_values = df.index.strftime('%Y-%m-%d %H:%M').tolist()
    else:
        x_values = df.index.astype(str).tolist()

    colors = ["#FF0000" if df["close"].iloc[i] >= df["open"].iloc[i] else "#00FF00" for i in range(len(df))]

    # K线图
    fig.add_trace(go.Candlestick(
        x=x_values,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing_line_color="#FF0000",
        decreasing_line_color="#00FF00",
        increasing_fillcolor="rgba(255,0,0,0)",
        decreasing_fillcolor="rgba(0,255,0,1)",
        name="K线"
    ), row=1, col=1)

    # 成交量
    fig.add_trace(go.Bar(
        x=x_values,
        y=df["volume"],
        marker_color=colors,
        opacity=1.0,
        name="成交量"
    ), row=2, col=1)

    # BSM指标副图
    if has_bsm:
        # bbmacd线
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df["bbmacd"],
            mode="lines",
            line=dict(width=1.5, color="#2962ff"),
            name="BBMACD"
        ), row=3, col=1)

        # 上轨
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df["banda_supe"],
            mode="lines",
            line=dict(width=1, color="#26a69a", dash="dash"),
            name="上轨"
        ), row=3, col=1)

        # 下轨
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df["banda_inf"],
            mode="lines",
            line=dict(width=1, color="#ef5350", dash="dash"),
            name="下轨"
        ), row=3, col=1)

        # 零轴
        fig.add_trace(go.Scatter(
            x=x_values,
            y=[0] * len(x_values),
            mode="lines",
            line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
            name="零轴"
        ), row=3, col=1)

    # 添加VWAP
    if show_vwap:
        cfg = DSAConfig(prd=50, baseAPT=20, useAdapt=False, volBias=10)
        try:
            vwap_series, dir_series, pivot_labels, segments = dynamic_swing_anchored_vwap(df, cfg)
            for seg in segments:
                seg_dir = seg["dir"]
                color = "#ff1744" if seg_dir > 0 else "#00e676"
                x_str = [pd.Timestamp(t).strftime('%Y-%m-%d %H:%M') for t in seg["x"]]
                fig.add_trace(
                    go.Scatter(x=x_str, y=seg["y"], mode="lines",
                               line=dict(width=2, color=color),
                               showlegend=False),
                    row=1, col=1
                )
            for lab in pivot_labels:
                txt = lab["text"]
                if not txt or lab["x"] not in df.index:
                    continue
                is_up = lab["dir"] > 0
                bgcolor = "rgba(0,230,118,0.85)" if is_up else "rgba(255,23,68,0.85)"
                ay = 25 if is_up else -25
                x_str = pd.Timestamp(lab["x"]).strftime('%Y-%m-%d %H:%M')
                fig.add_annotation(
                    x=x_str, y=lab["y"], xref="x", yref="y",
                    text=txt, showarrow=True, arrowhead=2, ax=0, ay=ay,
                    bgcolor=bgcolor, font=dict(color="white", size=12),
                    bordercolor=bgcolor, borderwidth=1, row=1, col=1
                )
        except Exception as e:
            print(f"VWAP计算失败: {e}")

    # 添加ATR Rope
    if show_atr_rope:
        freq_type = 'd'
        if len(df) > 0:
            idx = df.index[0]
            if hasattr(idx, 'hour') and idx.hour != 0:
                freq_type = '60m'
        args = argparse.Namespace(
            len=14, multi=1.5, source='close', freq=freq_type,
            show_factor_panels=False, show_atr_channel=True, show_ranges=True,
            show_break_markers=False
        )
        try:
            engine = ATRRopeEngine(df, args)
            engine.run()

            rope_colors = {
                "rope_up": "rgba(61, 170, 69, 0.6)",
                "rope_down": "rgba(255, 3, 62, 0.6)",
                "rope_flat": "rgba(0, 77, 146, 0.6)"
            }
            for col_name, color in rope_colors.items():
                fig.add_trace(go.Scatter(
                    x=x_values, y=engine.df[col_name], mode="lines",
                    line=dict(width=2.5, color=color, dash="dash"),
                    hoverinfo="skip", showlegend=False
                ), row=1, col=1)
        except Exception as e:
            print(f"ATR Rope计算失败: {e}")

    # 布局设置
    height = 800 if has_bsm else 600
    fig.update_layout(
        template="plotly_dark",
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        plot_bgcolor=DARK_THEME["bg_color"],
        hovermode='x unified'
    )

    fig.update_xaxes(type='category', showspikes=True, spikecolor="rgba(255,255,255,0.3)",
                     spikethickness=1, spikemode="across", showticklabels=False, row=1, col=1)
    fig.update_xaxes(type='category', showspikes=True, spikecolor="rgba(255,255,255,0.3)",
                     spikethickness=1, spikemode="across", showticklabels=False, row=2, col=1)
    if has_bsm:
        fig.update_xaxes(type='category', showspikes=True, spikecolor="rgba(255,255,255,0.3)",
                         spikethickness=1, spikemode="across", showticklabels=True, row=3, col=1)

    return fig


def add_buy_signals_to_chart(fig: go.Figure, df: pd.DataFrame, signals: Dict, period: str = 'd') -> go.Figure:
    """
    在K线图上添加买点标记

    Args:
        fig: Plotly Figure对象
        df: K线数据
        signals: 买点信号字典
        period: 周期 ('d'=日线, 'w'=周线)

    Returns:
        添加了标记的Figure对象
    """
    if df.empty:
        return fig

    x_values = df.index.strftime('%Y-%m-%d %H:%M').tolist() if hasattr(df.index, 'strftime') else df.index.astype(str).tolist()

    # 根据周期确定要检查的买点类型
    if period == 'd':
        signal_keys = ['daily_reversal_buy', 'daily_breakout_buy']
    elif period == 'w':
        signal_keys = ['weekly_reversal_buy', 'weekly_breakout_buy']
    else:
        return fig

    # 检查最后一根K线是否有买点信号
    for signal_key in signal_keys:
        if signals.get(signal_key, False):
            color = BUY_SIGNAL_COLORS.get(signal_key, "#ffffff")
            name = BUY_SIGNAL_NAMES.get(signal_key, signal_key)

            # 在最后一根K线下方添加标记
            if len(df) > 0:
                last_idx = len(x_values) - 1
                last_low = df['low'].iloc[-1]

                fig.add_annotation(
                    x=x_values[last_idx],
                    y=last_low,
                    xref="x", yref="y",
                    text="▲",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    ax=0,
                    ay=30,
                    font=dict(size=16, color=color),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor=color,
                    borderwidth=1,
                    row=1, col=1
                )

                # 添加文字标签
                fig.add_annotation(
                    x=x_values[last_idx],
                    y=last_low,
                    xref="x", yref="y",
                    text=name,
                    showarrow=False,
                    ax=0,
                    ay=50,
                    font=dict(size=10, color=color),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor=color,
                    borderwidth=1,
                    row=1, col=1
                )

    return fig


# ==================== 新的自选股页面（基于选股结果） ====================

def render_stock_picker_page(session, sidebar):
    """选股结果展示页面 - 仅显示股票列表"""
    from datetime import date

    # 获取可用日期列表
    available_dates = get_available_selection_dates(session)
    if not available_dates:
        st.warning("暂无选股数据，请先运行选股脚本")
        return

    # 日期选择（日历控件）
    # 处理日期格式：数据库可能是字符串或date对象
    def _to_date_obj(d):
        if isinstance(d, date):
            return d
        # 字符串格式 '20250415'
        return date(int(d[:4]), int(d[4:6]), int(d[6:8]))

    available_date_objs = [_to_date_obj(d) for d in available_dates]
    min_date = min(available_date_objs)
    max_date = max(available_date_objs)
    default_date = available_date_objs[0]  # 最新日期

    selected_date_obj = st.date_input(
        "选股日期",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        key="sel_date_calendar"
    )

    # 转换回字符串格式 '20250415'
    selected_date = selected_date_obj.strftime('%Y%m%d')

    # 如果选中日期无数据，自动切换到最近的有效日期
    available_date_strs = [d.strftime('%Y%m%d') if isinstance(d, date) else str(d) for d in available_dates]
    if selected_date not in available_date_strs:
        st.warning(f"{selected_date} 暂无选股数据，自动切换到最近的有效日期")
        # 找到最近的有效日期（小于等于选中日期的最大有效日期）
        valid_dates = [d for d in available_date_strs if d <= selected_date]
        if valid_dates:
            selected_date = max(valid_dates)
        else:
            selected_date = available_date_strs[0]  # 如果没有更早的日期，使用最新日期

    # 加载选股结果
    selection_df = load_selection_results(session, selected_date)

    if selection_df.empty:
        st.warning("该日期暂无选股数据")
        return

    # 显示统计
    st.markdown(f"**选股日期: {selected_date} | 共 {len(selection_df)} 只股票**")

    # 从 stock_pools 表获取概念数据并合并
    ts_codes = selection_df['ts_code'].tolist()
    if ts_codes:
        concept_sql = f"""
            SELECT ts_code, concepts 
            FROM stock_pools 
            WHERE ts_code IN ({','.join([f"'{code}'" for code in ts_codes])})
        """
        concept_df = query_sql(session, concept_sql, {})
        if concept_df is not None and not concept_df.empty:
            concept_map = dict(zip(concept_df['ts_code'], concept_df['concepts']))
            selection_df['concepts'] = selection_df['ts_code'].map(concept_map)
        else:
            selection_df['concepts'] = ''
    else:
        selection_df['concepts'] = ''

    # 重命名列以匹配配置
    selection_df = selection_df.rename(columns={
        'stock_name': 'stock_name',
        'ts_code': 'ts_code'
    })

    # 买点类型快速筛选
    st.markdown("**快速筛选**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_daily_reversal = st.checkbox("日线反转", value=False, key="filter_dr")
    with col2:
        filter_daily_breakout = st.checkbox("日线突破", value=False, key="filter_db")
    with col3:
        filter_weekly_reversal = st.checkbox("周线反转", value=False, key="filter_wr")
    with col4:
        filter_weekly_breakout = st.checkbox("周线突破", value=False, key="filter_wb")

    # 应用快速筛选
    filtered_df = selection_df.copy()
    if filter_daily_reversal:
        filtered_df = filtered_df[filtered_df['daily_reversal_buy'] == True]
    if filter_daily_breakout:
        filtered_df = filtered_df[filtered_df['daily_breakout_buy'] == True]
    if filter_weekly_reversal:
        filtered_df = filtered_df[filtered_df['weekly_reversal_buy'] == True]
    if filter_weekly_breakout:
        filtered_df = filtered_df[filtered_df['weekly_breakout_buy'] == True]

    st.markdown("---")

    # 多条件筛选
    conditions = render_filter_bar(filtered_df, "选股主页")
    if conditions:
        filtered_df = apply_filters(filtered_df, conditions, TAB_FIELD_CONFIGS["选股主页"])

    loaded_conditions = render_filter_manager(session, "选股主页", conditions)
    if loaded_conditions:
        filtered_df = selection_df.copy()
        # 重新应用快速筛选
        if filter_daily_reversal:
            filtered_df = filtered_df[filtered_df['daily_reversal_buy'] == True]
        if filter_daily_breakout:
            filtered_df = filtered_df[filtered_df['daily_breakout_buy'] == True]
        if filter_weekly_reversal:
            filtered_df = filtered_df[filtered_df['weekly_reversal_buy'] == True]
        if filter_weekly_breakout:
            filtered_df = filtered_df[filtered_df['weekly_breakout_buy'] == True]
        filtered_df = apply_filters(filtered_df, loaded_conditions, TAB_FIELD_CONFIGS["选股主页"])

    st.markdown(f"**筛选结果: {len(filtered_df)} 只**")
    st.markdown("---")

    # 显示股票列表表格
    if filtered_df.empty:
        st.info("无符合条件的股票")
    else:
        # 准备显示数据
        display_data = []
        for _, row in filtered_df.iterrows():
            # 买点标记
            buy_signals = []
            if row.get('daily_reversal_buy'):
                buy_signals.append("🟢日反")
            if row.get('daily_breakout_buy'):
                buy_signals.append("🔵日突")
            if row.get('weekly_reversal_buy'):
                buy_signals.append("🟣周反")
            if row.get('weekly_breakout_buy'):
                buy_signals.append("🟠周突")
            buy_signal_str = " ".join(buy_signals) if buy_signals else "-"

            # 获取概念
            concepts = row.get('concepts', '')
            if concepts and isinstance(concepts, str) and concepts.strip():
                concept_list = [c.strip() for c in concepts.split(';') if c.strip()]
                concepts_display = '、'.join(concept_list)
            else:
                concepts_display = '-'

            display_data.append({
                '代码': row['ts_code'],
                '名称': row.get('stock_name', ''),
                '概念': concepts_display,
                '买点': buy_signal_str,
                '日线Z分': row.get('daily_bb_width_zscore') if pd.notna(row.get('daily_bb_width_zscore')) else None,
                '周线Z分': row.get('weekly_bb_width_zscore') if pd.notna(row.get('weekly_bb_width_zscore')) else None,
                '日线VolZ': row.get('daily_vol_zscore') if pd.notna(row.get('daily_vol_zscore')) else None,
                '周线VolZ': row.get('weekly_vol_zscore') if pd.notna(row.get('weekly_vol_zscore')) else None,
            })

        display_df = pd.DataFrame(display_data)

        # 配置列格式：数值列保持数值类型用于正确排序，同时控制显示精度
        column_config = {
            '日线Z分': st.column_config.NumberColumn(format="%.2f"),
            '周线Z分': st.column_config.NumberColumn(format="%.2f"),
            '日线VolZ': st.column_config.NumberColumn(format="%.2f"),
            '周线VolZ': st.column_config.NumberColumn(format="%.2f"),
        }

        st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=column_config)


# ==================== 原自选股功能（保留备用） ====================

def render_old_watchlist_page(session, sidebar):
    """原自选股管理页面（保留备用）"""
    watchlist_df = load_watchlist(session)

    with sidebar:
        st.subheader("📌 我的自选股")

        if watchlist_df.empty:
            st.info("自选股为空，请添加股票")
        else:
            stock_options = [f"{row['name']} ({row['ts_code']})" for _, row in watchlist_df.iterrows()]

            if "wl_idx" not in st.session_state:
                st.session_state.wl_idx = 0

            nav_cols = st.columns([1, 1, 8])
            with nav_cols[0]:
                if st.button("◀", key="wl_prev", help="上一只"):
                    st.session_state.wl_idx = max(0, st.session_state.wl_idx - 1)
                    st.rerun()
            with nav_cols[1]:
                if st.button("▶", key="wl_next", help="下一只"):
                    st.session_state.wl_idx = min(len(stock_options) - 1, st.session_state.wl_idx + 1)
                    st.rerun()

            selected = st.selectbox(
                "选择股票",
                options=stock_options,
                index=st.session_state.wl_idx,
                key="wl_sel"
            )
            selected_idx = stock_options.index(selected)
            st.session_state.wl_idx = selected_idx

            cols = st.columns([11, 1])
            with cols[1]:
                ts_code = watchlist_df.iloc[selected_idx]["ts_code"]
                if st.button("×", key=f"wl_del_{selected_idx}"):
                    remove_from_watchlist(session, ts_code)
                    st.rerun()

        st.markdown("---")
        st.subheader("+ 添加自选股")

        pinyin_input = st.text_input("输入股票名称/代码", placeholder="例如: pag -> 平安银行", key="wl_search")

        if pinyin_input:
            stock_list = load_stock_list(session)
            matched = match_stocks_by_pinyin(pinyin_input, stock_list)

            if not matched.empty:
                selected = st.selectbox(
                    "选择股票",
                    options=range(len(matched)),
                    format_func=lambda i: f"{matched.iloc[i]['name']} ({matched.iloc[i]['ts_code']})",
                    key="wl_select"
                )
                if st.button("添加", key="wl_add_btn"):
                    ts_code = matched.iloc[selected]["ts_code"]
                    name = matched.iloc[selected]["name"]
                    if add_to_watchlist(session, ts_code, name):
                        st.success(f"已添加 {name}")
                        st.rerun()
                    else:
                        st.warning(f"{name} 已在自选股中")
            else:
                st.info("未找到匹配的股票")

    st.header("⭐ 自选股")

    if watchlist_df.empty:
        st.info("请先添加自选股")
        return

    ts_code = watchlist_df.iloc[selected_idx]["ts_code"]
    name = watchlist_df.iloc[selected_idx]["name"]

    tabs = st.tabs(["月线", "周线", "日线", "60分钟", "📊 个股属性"])
    period_map = {"月线": "m", "周线": "w", "日线": "d", "60分钟": "60m"}

    for i, tab_name in enumerate(["月线", "周线", "日线", "60分钟"]):
        with tabs[i]:
            period = period_map[tab_name]

            with st.expander("⚙️ 指标设置", expanded=False):
                cols = st.columns(2)
                enabled_indicators = {}
                period_code = period
                for j, (key, cfg) in enumerate(INDICATOR_CONFIG.items()):
                    with cols[j % 2]:
                        is_applicable = period_code in cfg["periods"]
                        enabled = st.checkbox(
                            cfg["name"],
                            value=cfg["default"] if is_applicable else False,
                            disabled=not is_applicable,
                            key=f"ind_{key}_{tab_name}"
                        )
                        enabled_indicators[key] = enabled

            with st.spinner("加载K线数据..."):
                try:
                    kline_df = get_cached_kline(ts_code, period, count=250)
                    if kline_df is None or kline_df.empty:
                        st.warning("暂无数据")
                    else:
                        kline_df['bar_time'] = pd.to_datetime(kline_df['bar_time']).dt.tz_localize(None)
                        kline_df = kline_df.set_index('bar_time')

                        fig = build_kline_chart(
                            kline_df, name,
                            show_vwap=enabled_indicators.get("vwap", False),
                            show_atr_rope=enabled_indicators.get("atr_rope", False)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        latest = kline_df.iloc[-1]
                        prev = kline_df.iloc[-2] if len(kline_df) > 1 else latest

                        cols = st.columns(4)
                        close = latest.get("close", 0)
                        prev_close = prev.get("close", close)
                        pct_change = ((close - prev_close) / prev_close * 100) if prev_close != 0 else 0

                        cols[0].metric("最新价", f"{close:.2f}", f"{pct_change:.2f}%")
                        cols[1].metric("最高", f"{latest.get('high', 0):.2f}")
                        cols[2].metric("最低", f"{latest.get('low', 0):.2f}")
                        cols[3].metric("成交量", f"{latest.get('volume', 0)/10000:.2f}万")
                except Exception as e:
                    st.error(f"获取K线数据失败: {e}")

    with tabs[4]:
        render_stock_profile_page(session, ts_code, name)


def render_filter_bar(df: pd.DataFrame, tab_name: str) -> list:
    """渲染多条件筛选栏，支持动态添加/删除条件"""
    if tab_name not in TAB_FIELD_CONFIGS:
        return []

    if "filter_conditions" not in st.session_state:
        st.session_state.filter_conditions = {}

    if tab_name not in st.session_state.filter_conditions:
        st.session_state.filter_conditions[tab_name] = []

    config = TAB_FIELD_CONFIGS[tab_name]
    conditions = st.session_state.filter_conditions[tab_name]

    st.markdown("**🔍 筛选条件**")

    to_remove = []
    for i, cond in enumerate(conditions):
        cols = st.columns([3, 2, 3, 1])
        fields = list(config["fields"])

        with cols[0]:
            default_idx = fields.index(cond["field"]) if cond.get("field") in fields else 0
            selected_field = st.selectbox(
                "字段",
                options=fields,
                format_func=lambda x: config["display_names"].get(x, x),
                key=f"cond_{tab_name}_{i}_field",
                index=default_idx
            )

        field_type = config["types"].get(selected_field, "string")
        operators = ["包含", "等于", "不包含"] if field_type == "string" else ["大于", "小于", "大于等于", "小于等于", "区间"]

        with cols[1]:
            default_op = operators.index(cond["operator"]) if cond.get("operator") in operators else 0
            selected_op = st.selectbox(
                "运算符",
                options=operators,
                key=f"cond_{tab_name}_{i}_op",
                index=default_op
            )

        with cols[2]:
            if field_type == "string":
                value = st.text_input("值", value=cond.get("value") or "", key=f"cond_{tab_name}_{i}_val", label_visibility="collapsed")
                value2 = None
            else:
                if selected_op == "区间":
                    col_sub = st.columns(2)
                    with col_sub[0]:
                        value = st.number_input("起始", value=float(cond.get("value") or 0.0), key=f"cond_{tab_name}_{i}_val", format="%.2f")
                    with col_sub[1]:
                        value2 = st.number_input("结束", value=float(cond.get("value2") or 100.0), key=f"cond_{tab_name}_{i}_val2", format="%.2f")
                else:
                    value = st.number_input("值", value=float(cond.get("value") or 0.0), key=f"cond_{tab_name}_{i}_val", format="%.2f")
                    value2 = None

        with cols[3]:
            if st.button("×", key=f"cond_{tab_name}_{i}_remove"):
                to_remove.append(i)

        st.session_state.filter_conditions[tab_name][i] = {
            "field": selected_field,
            "operator": selected_op,
            "value": value,
            "value2": value2
        }

    for i in sorted(to_remove, reverse=True):
        st.session_state.filter_conditions[tab_name].pop(i)

    btn_cols = st.columns([1, 1, 4])
    with btn_cols[0]:
        if st.button("+ 添加条件", key=f"add_cond_{tab_name}"):
            first_field = config["fields"][0]
            default_op = "大于" if config["types"].get(first_field) == "numeric" else "包含"
            default_val = 0.0 if config["types"].get(first_field) == "numeric" else ""
            st.session_state.filter_conditions[tab_name].append({
                "field": first_field,
                "operator": default_op,
                "value": default_val,
                "value2": None
            })
            st.rerun()

    with btn_cols[1]:
        if st.button("清除全部", key=f"clear_cond_{tab_name}"):
            st.session_state.filter_conditions[tab_name] = []
            st.rerun()

    return st.session_state.filter_conditions.get(tab_name, [])


def get_latest_financial_scores(session) -> pd.DataFrame:
    """获取每只股票的最新一期财务评分（用于join）"""
    sql = """
        SELECT ts_code, stock_name, total_score, report_date,
               边际变化与持续性_score, 利润质量_score, 资产效率与资金占用_score,
               规模与增长_score, 现金创造能力_score, 盈利能力_score
        FROM stock_financial_score_pool
        WHERE (ts_code, report_date) IN (
            SELECT ts_code, MAX(report_date)
            FROM stock_financial_score_pool
            GROUP BY ts_code
        )
    """
    try:
        return query_sql(session, sql, {})
    except Exception:
        return pd.DataFrame()


def format_factor_value(val) -> str:
    """格式化因子值显示"""
    if pd.isna(val):
        return "N/A"
    if abs(val) > 100:
        return f"{val:.2f}"
    elif abs(val) > 1:
        return f"{val:.4f}"
    else:
        return f"{val:.2%}"
    return str(val)


def format_score(val) -> str:
    """格式化分数显示"""
    if pd.isna(val):
        return "N/A"
    return f"{val:.1f}"


def format_report_date(date_str: str) -> str:
    """格式化报告期显示"""
    if pd.isna(date_str) or date_str is None or len(str(date_str)) != 8:
        return str(date_str) if date_str else "N/A"
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def render_dimension_chart(df: pd.DataFrame, report_date: str) -> go.Figure:
    """渲染5维度柱状图"""
    if df.empty:
        return None

    latest = df[df["report_date"] == report_date]
    if latest.empty:
        latest = df.iloc[[-1]]
    latest = latest.iloc[0]

    dimensions = []
    scores = []
    colors = []

    for dim_col in DIMENSION_COLS:
        dim_name = dim_col.replace("_score", "")
        score = latest.get(dim_col)
        if pd.notna(score):
            dimensions.append(dim_name)
            scores.append(float(score))
            colors.append(DIMENSION_COLORS.get(dim_name, "#888888"))

    fig = go.Figure(go.Bar(
        x=scores,
        y=dimensions,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}" for s in scores],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>分数: %{x:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="5维度评分",
            font=dict(size=14, color=DARK_THEME["text_color"]),
        ),
        plot_bgcolor=DARK_THEME["bg_color"],
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        font=dict(color=DARK_THEME["text_color"]),
        height=280,
        margin=dict(l=20, r=40, t=40, b=20),
        xaxis=dict(
            range=[0, 100],
            title="分数",
            gridcolor=DARK_THEME["grid_color"],
        ),
        yaxis=dict(
            title="",
            gridcolor=DARK_THEME["grid_color"],
        ),
    )

    return fig


def render_trend_chart(df: pd.DataFrame) -> go.Figure:
    """渲染历史趋势折线图"""
    if df.empty or len(df) < 2:
        return None

    df = df.copy()
    df["report_date_display"] = df["report_date"].apply(format_report_date)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["report_date_display"],
        y=df["total_score"],
        mode="lines+markers",
        name="总分",
        line=dict(color="#ffffff", width=3),
        marker=dict(size=8),
        hovertemplate="<b>总分</b><br>报告期: %{x}<br>分数: %{y:.1f}<extra></extra>",
    ))

    for dim_col in DIMENSION_COLS:
        dim_name = dim_col.replace("_score", "")
        color = DIMENSION_COLORS.get(dim_name, "#888888")
        fig.add_trace(go.Scatter(
            x=df["report_date_display"],
            y=df[dim_col],
            mode="lines+markers",
            name=dim_name,
            line=dict(color=color, width=1.5),
            marker=dict(size=6),
            hovertemplate=f"<b>{dim_name}</b><br>报告期: %{{x}}<br>分数: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="评分历史趋势",
            font=dict(size=14, color=DARK_THEME["text_color"]),
        ),
        plot_bgcolor=DARK_THEME["bg_color"],
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        font=dict(color=DARK_THEME["text_color"]),
        height=350,
        margin=dict(l=50, r=30, t=40, b=50),
        xaxis=dict(
            title="报告期",
            gridcolor=DARK_THEME["grid_color"],
            tickangle=-45,
        ),
        yaxis=dict(
            range=[0, 100],
            title="分数",
            gridcolor=DARK_THEME["grid_color"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1.0,
            font=dict(size=9),
        ),
        hovermode="x unified",
    )

    return fig


def render_factor_table(factor_data: pd.DataFrame, dim_filter: Optional[str] = None) -> pd.DataFrame:
    """构建因子明细表格数据"""
    rows = []
    for cfg in FACTOR_CONFIG:
        fname = cfg["factor_name"]

        if fname not in factor_data.columns:
            continue

        raw_val = factor_data[fname].iloc[0] if fname in factor_data.columns else np.nan

        if dim_filter and cfg["dimension"] != dim_filter:
            continue

        if isinstance(raw_val, (int, float)) and pd.notna(raw_val):
            if abs(raw_val) > 100:
                raw_display = f"{raw_val:.2f}"
            elif abs(raw_val) > 1:
                raw_display = f"{raw_val:.4f}"
            else:
                raw_display = f"{raw_val:.2%}"
        else:
            raw_display = "N/A"

        rows.append({
            "维度": cfg["dimension"],
            "因子名": fname,
            "中文名": cfg["label"],
            "原始值": raw_display,
            "方向": cfg["direction"],
            "权重": cfg["weight"],
            "核心": "是" if cfg["is_core"] else "否",
        })

    return pd.DataFrame(rows)


def render_pool_page(session):
    """渲染全股池因子表页面"""
    st.header("📊 全股池财务因子表")

    report_dates = get_available_report_dates(session)
    if not report_dates:
        report_dates = []

    if report_dates:
        selected_report = st.selectbox("报告期", options=report_dates, format_func=format_report_date, index=0)
    else:
        selected_report = None
        st.info("暂无报告期数据")

    with st.spinner("加载数据..."):
        df = load_all_scores(session, selected_report) if selected_report else pd.DataFrame()

    concept_sql = "SELECT ts_code, concepts FROM stock_pools"
    concept_df = query_sql(session, concept_sql, {})
    if concept_df is not None and not concept_df.empty:
        df = df.merge(concept_df, on="ts_code", how="left")

    display_cols = ["ts_code", "stock_name", "report_date", "ann_date", "total_score"] + DIMENSION_COLS
    display_cols = [c for c in display_cols if c in df.columns]

    key_factors = ["q_rev_yoy_delta", "q_np_parent_yoy_delta", "q_gm_yoy_change", "q_cfo_to_np_parent", "roa_parent"]
    key_factors = [f for f in key_factors if f in df.columns]
    display_cols.extend(key_factors)

    if "concepts" in df.columns:
        display_cols.append("concepts")

    display_df = df[display_cols].copy()

    col_names = ["股票代码", "股票名称", "报告期", "公告日期", "总分",
                 "边际变化与持续性", "利润质量", "资产效率", "规模与增长", "现金创造能力", "盈利能力"]
    col_names = col_names[:len(display_cols) - len(key_factors) - (1 if "concepts" in df.columns else 0)]
    display_df.columns = col_names + [f for f in key_factors] + (["概念"] if "concepts" in df.columns else [])

    conditions = render_filter_bar(display_df, "全股池因子表")
    if conditions:
        display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["全股池因子表"])

    loaded_conditions = render_filter_manager(session, "全股池因子表", conditions)
    if loaded_conditions:
        display_df = df.copy()
        display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["全股池因子表"])

    # 默认按边际变化与持续性降序排序（确保分页前已排序）
    if "边际变化与持续性" in display_df.columns:
        display_df = display_df.sort_values("边际变化与持续性", ascending=False, na_position="last").reset_index(drop=True)
    elif "边际变化与持续性_score" in display_df.columns:
        display_df = display_df.sort_values("边际变化与持续性_score", ascending=False, na_position="last").reset_index(drop=True)

    total_rows = len(display_df)
    st.markdown(f"**股票数: {total_rows}** | **报告期: {format_report_date(selected_report)}**")

    # 分页
    page_size = 20
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page = st.number_input("页码", min_value=1, max_value=total_pages, value=1, step=1, key="pool_page")
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    page_df = display_df.iloc[start_idx:end_idx].copy()
    st.caption(f"第 {page}/{total_pages} 页，显示第 {start_idx + 1}-{end_idx} 条")

    color_cols = ["总分", "边际变化与持续性", "利润质量", "资产效率", "规模与增长", "现金创造能力"]
    st.dataframe(
        colorize_numeric_columns(page_df, color_cols),
        use_container_width=True,
        hide_index=True,
        height=600,
    )


def _render_stock_page(session, matched_stocks: pd.DataFrame, selected_option: int):
    """渲染个股分析页面"""
    if selected_option is None:
        return

    selected_stock = matched_stocks.iloc[selected_option]
    ts_code = selected_stock["ts_code"]
    stock_name = selected_stock["name"]

    st.markdown(f"### 当前股票: **{stock_name}** ({ts_code})")

    # 加载财务评分数据
    with st.spinner("加载评分数据..."):
        score_df = load_score_data(session, ts_code)

    has_score_data = not score_df.empty

    if has_score_data:
        report_dates = score_df["report_date"].unique().tolist()
        latest_report = report_dates[-1] if report_dates else None

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            total_score = score_df[score_df["report_date"] == latest_report]["total_score"].values[0]
            st.metric("总分", format_score(total_score))

        with col2:
            score = score_df[score_df["report_date"] == latest_report]["边际变化与持续性_score"].values[0]
            st.metric("边际变化与持续性", format_score(score))

        with col3:
            score = score_df[score_df["report_date"] == latest_report]["利润质量_score"].values[0]
            st.metric("利润质量", format_score(score))

        with col4:
            score = score_df[score_df["report_date"] == latest_report]["资产效率与资金占用_score"].values[0]
            st.metric("资产效率与资金占用", format_score(score))

        with col5:
            score = score_df[score_df["report_date"] == latest_report]["规模与增长_score"].values[0]
            st.metric("规模与增长", format_score(score))

        with col6:
            score = score_df[score_df["report_date"] == latest_report]["现金创造能力_score"].values[0]
            st.metric("现金创造能力", format_score(score))

        st.markdown("---")

        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.markdown("#### 最新报告期评分")
            # 获取最新报告期的公告日期
            latest_ann_date = score_df[score_df["report_date"] == latest_report]["ann_date"].values[0]
            ann_date_str = str(latest_ann_date) if pd.notna(latest_ann_date) else "N/A"
            st.markdown(f"**报告期: {format_report_date(latest_report)}**  |  **公告日: {ann_date_str}**")
            dim_chart = render_dimension_chart(score_df, latest_report)
            if dim_chart:
                st.plotly_chart(dim_chart, use_container_width=True)

        with chart_col2:
            st.markdown("#### 评分历史趋势")
            trend_chart = render_trend_chart(score_df.tail(8))
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)

        st.markdown("---")

        # 股东质量画像
        st.markdown("#### 股东质量画像")
        render_stock_holder_quality_section(session, ts_code)

        st.markdown("---")

        with st.expander("📋 因子明细表"):
            dim_options = ["全部"] + sorted(list(DIMENSION_WEIGHTS.keys()))
            selected_dim = st.selectbox("按维度筛选", options=dim_options, key="dim_filter")

            factor_df = load_factor_scores(session, ts_code, latest_report)
            table_data = render_factor_table(factor_df)

            if selected_dim != "全部":
                table_data = table_data[table_data["维度"] == selected_dim]

            if not table_data.empty:
                table_data = table_data.sort_values(["核心", "权重"], ascending=[False, False])
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("无因子数据")

        st.markdown("---")

        with st.expander("📈 评分数据概览"):
            display_df = score_df[["report_date", "total_score"] + DIMENSION_COLS].copy()
            display_df["report_date"] = display_df["report_date"].apply(format_report_date)
            display_df.columns = ["报告期", "总分", "边际变化与持续性", "利润质量", "现金创造能力",
                                   "资产效率与资金占用", "规模与增长", "盈利能力"]
            for col in display_df.columns[1:]:
                display_df[col] = display_df[col].apply(format_score)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")
    else:
        st.warning(f"该股票暂无财务评分数据")
        st.markdown("---")

        # 股东质量画像（无财务评分数据时也显示）
        st.markdown("#### 股东质量画像")
        render_stock_holder_quality_section(session, ts_code)


def render_stock_holder_quality_section(session, ts_code: str):
    """渲染个股股东质量画像板块"""
    # 查询该股票最新报告期的十大流通股东及其画像
    holder_sql = """
        SELECT h.holder_rank, h.holder_name, h.holder_type, h.hold_ratio, h.hold_change, h.hold_amount,
               p.composite_score, p.quality_grade, p.picking_score, p.style_score,
               p.expertise_score, p.adapt_score, p.risk_score, p.scale_score,
               p.style_label, p.period_label, p.industry_label, p.ability_label,
               p.sample_stocks, p.sample_entry, p.entry_win_rate_60,
               p.sample_add, p.add_win_rate_60
        FROM stock_top10_holders_tushare h
        LEFT JOIN stock_holder_quality_portrait p ON h.holder_name = p.holder_name_std
        WHERE h.ts_code = :ts_code
        AND h.report_date = (SELECT MAX(report_date) FROM stock_top10_holders_tushare WHERE ts_code = :ts_code)
        ORDER BY h.holder_rank
    """
    holder_df = query_sql(session, holder_sql, {"ts_code": ts_code})

    if holder_df is None or holder_df.empty:
        st.info("暂无股东质量画像数据")
        return

    # 计算加权综合分
    total_weight = 0
    weighted_score = 0
    dim_scores = {"picking_score": [], "style_score": [], "expertise_score": [],
                  "adapt_score": [], "risk_score": [], "scale_score": []}

    for _, row in holder_df.iterrows():
        ratio = row.get("hold_ratio", 0) or 0
        if ratio <= 0:
            ratio = 1.0
        composite = row.get("composite_score")
        if pd.notna(composite):
            weighted_score += composite * ratio
            total_weight += ratio
        for dim in dim_scores.keys():
            val = row.get(dim)
            if pd.notna(val):
                dim_scores[dim].append((val, ratio))

    avg_score = weighted_score / total_weight if total_weight > 0 else 50

    # 获取市场平均分
    market_sql = "SELECT AVG(composite_score) as avg_score FROM stock_holder_quality_portrait"
    market_df = query_sql(session, market_sql, {})
    market_avg = market_df["avg_score"].iloc[0] if market_df is not None and not market_df.empty else 50

    # 计算各维度平均分
    dim_avgs = {}
    for dim, vals in dim_scores.items():
        if vals:
            total_w = sum(w for _, w in vals)
            dim_avgs[dim] = sum(v * w for v, w in vals) / total_w if total_w > 0 else 50
        else:
            dim_avgs[dim] = 50

    # 确定质量等级
    if avg_score >= 75:
        grade = "A"
        grade_color = "#28a745"
    elif avg_score >= 60:
        grade = "B"
        grade_color = "#007bff"
    elif avg_score >= 45:
        grade = "C"
        grade_color = "#6c757d"
    else:
        grade = "D"
        grade_color = "#dc3545"

    # 展示加权综合分
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown(f"**加权综合分**: <span style='font-size:24px;font-weight:bold;color:{grade_color}'>{avg_score:.1f}</span> <span style='background-color:{grade_color};color:white;padding:2px 8px;border-radius:4px'>{grade}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**市场平均**: {market_avg:.1f}")
    with col3:
        delta = avg_score - market_avg
        delta_color = "green" if delta > 0 else "red" if delta < 0 else "gray"
        st.markdown(f"**相对市场**: <span style='color:{delta_color};font-weight:bold'>{delta:+.1f}</span>", unsafe_allow_html=True)

    # 股东明细表格 (带质量分)
    st.markdown("**股东明细**:")

    # 计算持股变化比例
    def calc_change_ratio(row):
        if pd.isna(row['hold_change']) or pd.isna(row['hold_amount']) or row['hold_amount'] == 0:
            return None
        prev_amount = row['hold_amount'] - row['hold_change']
        if prev_amount == 0:
            return None
        return row['hold_change'] / prev_amount * 100

    holder_df['change_ratio'] = holder_df.apply(calc_change_ratio, axis=1)

    # 构建展示表格
    display_df = holder_df[["holder_rank", "holder_name", "holder_type", "hold_ratio", "change_ratio",
                            "composite_score", "sample_stocks", "entry_win_rate_60", "add_win_rate_60",
                            "quality_grade", "style_label"]].copy()
    display_df.columns = ["排名", "股东名称", "类型", "持股比例", "变化比例", "质量分",
                          "涉及股票", "入场胜率", "加仓胜率", "等级", "风格"]

    # 格式化列
    display_df["持股比例"] = display_df["持股比例"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

    # 变化比例带颜色标记
    def format_change_ratio(x):
        if pd.isna(x):
            return "N/A"
        color = "#28a745" if x > 0 else "#dc3545" if x < 0 else "#6c757d"
        return f"<span style='color:{color};font-weight:bold'>{x:+.2f}%</span>"

    display_df["变化比例"] = display_df["变化比例"].apply(format_change_ratio)
    display_df["质量分"] = display_df["质量分"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    display_df["涉及股票"] = display_df["涉及股票"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")
    display_df["入场胜率"] = display_df["入场胜率"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    display_df["加仓胜率"] = display_df["加仓胜率"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    # 等级颜色标记
    def grade_badge(grade):
        color = get_quality_grade_color(str(grade))
        return f"<span style='background-color:{color};color:white;padding:2px 6px;border-radius:4px;font-size:12px'>{grade}</span>"

    display_df["等级"] = display_df["等级"].apply(grade_badge)
    st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)


def render_holder_radar_chart(portrait_row: pd.Series) -> go.Figure:
    """渲染股东盈亏雷达图（基于实际盈亏数据）"""
    # 使用新的盈亏指标替代原有评分
    categories = ["累计盈亏", "平均胜率", "入场胜率", "加仓胜率", "涉及股票", "交易次数"]

    # 归一化数值到0-100范围
    total_profit = min(portrait_row.get("total_profit_pct", 0) * 2, 100)  # 盈亏比例*2，上限100
    avg_win_rate = portrait_row.get("avg_win_rate", 0)
    entry_win_rate = portrait_row.get("entry_win_rate_60", 0)
    add_win_rate = portrait_row.get("add_win_rate_60", 0)
    sample_stocks = min(portrait_row.get("sample_stocks", 0) * 10, 100)  # 股票数*10，上限100
    total_trades = min(portrait_row.get("total_trades", 0) * 5, 100)  # 交易次数*5，上限100

    values = [total_profit, avg_win_rate, entry_win_rate, add_win_rate, sample_stocks, total_trades]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(66, 133, 244, 0.3)',
        line=dict(color='rgb(66, 133, 244)', width=2),
        name='盈亏指标'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            angularaxis=dict(direction="clockwise")
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=350
    )
    return fig


def get_quality_grade_color(grade: str) -> str:
    """获取质量等级对应的颜色"""
    color_map = {
        "A": "#28a745",  # 绿色
        "B": "#007bff",  # 蓝色
        "C": "#6c757d",  # 灰色
        "D": "#dc3545",  # 红色
    }
    return color_map.get(grade, "#6c757d")


def load_holder_detail_from_db(session, holder_name: str) -> Dict:
    """
    从数据库加载股东详情数据（包含交易记录和统计）

    Returns:
        {
            'portrait': DataFrame,  # 基本信息
            'trade_records': DataFrame,  # 交易记录
            'trade_stats': DataFrame,  # 按股票统计
            'current_holdings': DataFrame  # 当前持股
        }
    """
    result = {}

    # 1. 查询基本信息
    portrait_sql = """
        SELECT * FROM stock_holder_quality_portrait 
        WHERE holder_name_std = :holder_name
    """
    result['portrait'] = query_sql(session, portrait_sql, {"holder_name": holder_name})

    # 2. 查询交易记录
    records_sql = """
        SELECT 
            ts_code, stock_name, report_date, operation,
            hold_ratio, hold_amount, current_price,
            entry_date, entry_price, entry_shares, entry_amount,
            exit_date, exit_price, exit_shares, exit_amount,
            profit_pct, profit_amount
        FROM holder_trade_records 
        WHERE holder_name = :holder_name 
        ORDER BY report_date DESC, ts_code
    """
    result['trade_records'] = query_sql(session, records_sql, {"holder_name": holder_name})

    # 3. 查询按股票统计
    stats_sql = """
        SELECT 
            ts_code, stock_name, total_profit, win_rate,
            operation_count, win_count, loss_count,
            total_entry_amount, total_exit_amount, total_profit_amount,
            first_entry_date, last_exit_date
        FROM holder_trade_stats 
        WHERE holder_name = :holder_name 
        ORDER BY total_profit DESC
    """
    result['trade_stats'] = query_sql(session, stats_sql, {"holder_name": holder_name})

    # 4. 查询当前持股
    holdings_sql = """
        SELECT 
            h.ts_code, 
            h.stock_name, 
            h.hold_ratio, 
            h.hold_change, 
            h.report_date, 
            h.holder_rank,
            h.hold_amount
        FROM stock_top10_holders_tushare h
        WHERE h.holder_name = :holder_name
        AND h.report_date = (
            SELECT MAX(report_date) 
            FROM stock_top10_holders_tushare 
            WHERE holder_name = :holder_name
        )
        ORDER BY h.hold_ratio DESC
    """
    result['current_holdings'] = query_sql(session, holdings_sql, {"holder_name": holder_name})

    return result


def format_amount(amount):
    """格式化金额显示"""
    if amount is None or pd.isna(amount):
        return "N/A"
    try:
        amount = float(amount)
        if abs(amount) >= 100000000:
            return f"{amount/100000000:+.2f}亿"
        elif abs(amount) >= 10000:
            return f"{amount/10000:+.2f}万"
        else:
            return f"{amount:+.2f}元"
    except:
        return "N/A"


def get_operation_icon(operation: str) -> str:
    """获取操作类型对应的图标"""
    icon_map = {
        "入场": "⬆️",
        "出场": "⬇️",
        "加仓": "📈",
        "减仓": "📉",
        "持仓": "➡️"
    }
    return icon_map.get(operation, "➡️")


def load_holder_portrait_paginated(session, page: int = 1, page_size: int = 100,
                                    filters: dict = None, order_by: str = "total_profit_pct", 
                                    order_desc: bool = True) -> tuple:
    """分页加载股东画像数据（集成实际盈亏数据）
    
    Returns:
        (df, total_count): 当前页数据框和总记录数
    """
    # 先获取总数
    count_sql = "SELECT COUNT(*) as cnt FROM stock_holder_quality_portrait"
    count_df = query_sql(session, count_sql, {})
    total_count = count_df["cnt"].iloc[0] if count_df is not None and not count_df.empty else 0
    
    # 构建查询SQL
    offset = (page - 1) * page_size
    order_direction = "DESC" if order_desc else "ASC"
    
    # 基础字段 - 删除原有评分字段，新增实际盈亏字段
    base_sql = """SELECT 
                        p.holder_name_std, p.holder_type, p.quality_grade,
                        p.sample_stocks, p.sample_entry, p.sample_add, p.sample_reduce, p.sample_exit,
                        p.style_label, p.period_label, p.industry_label, p.ability_label,
                        p.entry_excess_ret_60, p.add_excess_ret_60, p.entry_win_rate_60, p.add_win_rate_60,
                        p.avg_mdd_60, p.avg_tenure, p.contrarian_ratio,
                        -- 新增实际盈亏数据（来自 holder_trade_stats）
                        COALESCE(s.total_trades, 0) as total_trades,
                        COALESCE(s.total_profit_pct, 0) as total_profit_pct,
                        COALESCE(s.total_profit_amount, 0) as total_profit_amount,
                        COALESCE(s.win_rate, 0) as avg_win_rate,
                        COALESCE(s.total_entry_amount, 0) as total_entry_amount,
                        COALESCE(s.total_exit_amount, 0) as total_exit_amount
                 FROM stock_holder_quality_portrait p
                 LEFT JOIN (
                     SELECT 
                         holder_name,
                         SUM(operation_count) as total_trades,
                         SUM(total_profit) as total_profit_pct,
                         SUM(total_profit_amount) as total_profit_amount,
                         AVG(win_rate) as win_rate,
                         SUM(total_entry_amount) as total_entry_amount,
                         SUM(total_exit_amount) as total_exit_amount
                     FROM holder_trade_stats
                     GROUP BY holder_name
                 ) s ON p.holder_name_std = s.holder_name"""
    
    # 添加过滤条件
    where_clause = ""
    params = {}
    if filters:
        conditions = []
        if filters.get("holder_name"):
            conditions.append("p.holder_name_std ILIKE :holder_name")
            params["holder_name"] = f"%{filters['holder_name']}%"
        if filters.get("holder_type"):
            conditions.append("p.holder_type = :holder_type")
            params["holder_type"] = filters['holder_type']
        if filters.get("quality_grade"):
            conditions.append("p.quality_grade = :quality_grade")
            params["quality_grade"] = filters['quality_grade']
        if filters.get("min_stocks"):
            conditions.append("p.sample_stocks >= :min_stocks")
            params["min_stocks"] = filters['min_stocks']
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
    
    # 完整的分页SQL
    sql = f"""{base_sql}
              {where_clause}
              ORDER BY {order_by} {order_direction}
              LIMIT :limit OFFSET :offset"""
    params["limit"] = page_size
    params["offset"] = offset
    
    df = query_sql(session, sql, params)
    return df, total_count


def render_holder_profile_page(session):
    """渲染股东画像页面 (新系统) - 支持分页"""
    st.header("📊 股东画像 (新系统)")
    
    # 统计概览 - 只查询一次
    with st.spinner("加载统计..."):
        stats_sql = """SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN quality_grade = 'A' THEN 1 ELSE 0 END) as a_count,
            SUM(CASE WHEN quality_grade = 'B' THEN 1 ELSE 0 END) as b_count,
            SUM(CASE WHEN quality_grade = 'C' THEN 1 ELSE 0 END) as c_count,
            SUM(CASE WHEN quality_grade = 'D' THEN 1 ELSE 0 END) as d_count
        FROM stock_holder_quality_portrait"""
        stats_df = query_sql(session, stats_sql, {})
        
        if stats_df is None or stats_df.empty:
            st.warning("暂无股东画像数据，请先运行回补脚本")
            return
        
        total_count = int(stats_df["total"].iloc[0])
        a_count = int(stats_df["a_count"].iloc[0])
        b_count = int(stats_df["b_count"].iloc[0])
        c_count = int(stats_df["c_count"].iloc[0])
        d_count = int(stats_df["d_count"].iloc[0])

    # 统计概览卡片
    cols = st.columns(5)
    cols[0].metric("总股东数", f"{total_count:,}")
    cols[1].metric("A级股东", f"{a_count:,}", delta=f"{a_count/total_count*100:.1f}%")
    cols[2].metric("B级股东", f"{b_count:,}", delta=f"{b_count/total_count*100:.1f}%")
    cols[3].metric("C级股东", f"{c_count:,}", delta=f"{c_count/total_count*100:.1f}%")
    cols[4].metric("D级股东", f"{d_count:,}", delta=f"{d_count/total_count*100:.1f}%")

    # 质量等级说明
    with st.expander("📊 质量等级说明（基于实际盈亏数据）"):
        st.markdown("""
        **A级**: 累计盈亏 > +50% 且 胜率 > 60%  
        **B级**: 累计盈亏 > +20% 且 胜率 > 50%  
        **C级**: 累计盈亏 > -10% 且 胜率 > 40%  
        **D级**: 累计盈亏 ≤ -10% 或 胜率 ≤ 40%
        """)

    st.markdown("---")

    # 固定每页条数
    page_size = 100

    # 先加载所有数据用于多条件筛选
    with st.spinner("加载股东数据..."):
        all_df, _ = load_holder_portrait_paginated(
            session,
            page=1,
            page_size=100000,  # 加载所有数据
            filters=None,
            order_by="total_profit_pct",
            order_desc=True
        )

    if all_df.empty:
        st.info("暂无股东画像数据")
        return

    # 筛选栏 - 多条件筛选（作用于所有数据）
    # 列顺序：基本信息 -> 盈亏统计 -> 事件/胜率 -> 标签
    cols_to_show = ["holder_name_std", "holder_type", "quality_grade",
                    "sample_stocks", "total_trades", "total_profit_pct", "total_profit_amount", "avg_win_rate",
                    "sample_entry", "entry_win_rate_60", "sample_add", "add_win_rate_60",
                    "total_entry_amount", "total_exit_amount",
                    "style_label", "period_label"]

    display_df = all_df.copy()
    display_df = display_df[cols_to_show]

    # 默认按累计盈亏比例降序排序
    if "total_profit_pct" in display_df.columns:
        display_df = display_df.sort_values("total_profit_pct", ascending=False)

    # 应用多条件筛选器
    conditions = render_filter_bar(display_df, "股东画像")
    if conditions:
        display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["股东画像"])

    loaded_conditions = render_filter_manager(session, "股东画像", conditions)
    if loaded_conditions:
        display_df = all_df.copy()
        display_df = display_df[cols_to_show]
        # 重新应用默认排序
        if "total_profit_pct" in display_df.columns:
            display_df = display_df.sort_values("total_profit_pct", ascending=False)
        display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["股东画像"])

    # 筛选后的总数
    filtered_total = len(display_df)

    # 重命名列
    display_df.columns = ["股东名称", "类型", "质量等级",
                          "涉及股票", "总交易次数", "累计盈亏比例(%)", "累计盈亏金额(万)", "平均胜率(%)",
                          "入场事件", "入场胜率(%)", "加仓事件", "加仓胜率(%)",
                          "总入场金额(万)", "总出场金额(万)",
                          "风格标签", "周期标签"]

    # 分页计算
    total_pages = max(1, (filtered_total + page_size - 1) // page_size)

    # 分页控制
    page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
    with page_col1:
        if st.button("⏮ 首页", key="holder_first_page"):
            st.session_state.holder_page = 1
    with page_col2:
        current_page = st.session_state.get("holder_page", 1)
        current_page = st.number_input(f"页码 (共{total_pages}页)", min_value=1, max_value=total_pages,
                                       value=current_page, step=1, key="holder_page_input")
        st.session_state.holder_page = current_page
    with page_col3:
        if st.button("末页 ⏭", key="holder_last_page"):
            st.session_state.holder_page = total_pages

    # 对筛选后的数据进行分页
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, filtered_total)
    display_df = display_df.iloc[start_idx:end_idx]

    st.markdown(f"**当前展示: 第{current_page}/{total_pages}页，共{filtered_total:,}个股东 (本页{len(display_df)}个)**")
    
    # 颜色标记的质量等级
    def colorize_grade(val):
        color = get_quality_grade_color(str(val))
        return f'background-color: {color}; color: white; font-weight: bold; border-radius: 4px; padding: 2px 8px;'
    
    numeric_cols = ["涉及股票", "总交易次数", "累计盈亏比例(%)", "累计盈亏金额(万)", "平均胜率(%)",
                    "入场事件", "入场胜率(%)", "加仓事件", "加仓胜率(%)",
                    "总入场金额(万)", "总出场金额(万)"]

    # 股东选择器
    clicked_holder = None
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown("点击股东名称查看详情:")
    with cols[1]:
        holder_names = display_df["股东名称"].tolist()
        clicked_holder = st.selectbox("选择股东", options=["(未选择)"] + holder_names, label_visibility="collapsed", key="holder_select")

    # 展示表格 - 带颜色渲染
    def colorize_grade(val):
        color = get_quality_grade_color(str(val))
        return f'background-color: {color}; color: white; font-weight: bold; border-radius: 4px; padding: 2px 8px;'

    def colorize_profit_pct(val):
        """根据盈亏比例返回颜色"""
        try:
            pct = float(val)
            if pct >= 50:
                return 'background-color: #28a745; color: white; font-weight: bold;'  # 深绿
            elif pct >= 20:
                return 'background-color: #d4edda; color: #155724;'  # 浅绿
            elif pct >= -10:
                return 'background-color: #fff3cd; color: #856404;'  # 黄色
            else:
                return 'background-color: #dc3545; color: white; font-weight: bold;'  # 红色
        except:
            return ''

    def colorize_win_rate(val):
        """根据胜率返回颜色"""
        try:
            rate = float(val)
            if rate >= 60:
                return 'background-color: #28a745; color: white; font-weight: bold;'  # 深绿
            elif rate >= 50:
                return 'background-color: #d4edda; color: #155724;'  # 浅绿
            elif rate >= 40:
                return 'background-color: #fff3cd; color: #856404;'  # 黄色
            else:
                return 'background-color: #f8d7da; color: #721c24;'  # 红色
        except:
            return ''

    def colorize_count(val):
        """根据数量返回颜色（热力图效果）"""
        try:
            count = float(val)
            if count >= 10:
                return 'background-color: #c3e6cb; color: #155724;'  # 深绿
            elif count >= 5:
                return 'background-color: #d4edda; color: #155724;'  # 浅绿
            elif count >= 2:
                return 'background-color: #fff3cd; color: #856404;'  # 黄色
            else:
                return 'background-color: #f8d7da; color: #721c24;'  # 红色
        except:
            return ''

    # 应用样式
    styled_df = display_df.style\
        .map(colorize_grade, subset=["质量等级"])\
        .map(colorize_profit_pct, subset=["累计盈亏比例(%)"])\
        .map(colorize_win_rate, subset=["平均胜率(%)", "入场胜率(%)", "加仓胜率(%)"])\
        .map(colorize_count, subset=["涉及股票", "总交易次数", "入场事件", "加仓事件"])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # 详情面板
    if clicked_holder and clicked_holder != "(未选择)":
        st.markdown("---")
        st.markdown(f"### {clicked_holder} - 画像详情")

        # 从数据库加载股东详情数据（包含交易记录和统计）
        detail_data = load_holder_detail_from_db(session, clicked_holder)

        if detail_data['portrait'].empty:
            st.info("未找到该股东数据")
            return

        portrait_row = detail_data['portrait'].iloc[0]
        trade_records = detail_data['trade_records']
        trade_stats = detail_data['trade_stats']
        current_holdings = detail_data['current_holdings']

        # 计算汇总数据
        total_trades = len(trade_records) if not trade_records.empty else 0
        total_profit_pct = trade_stats['total_profit'].sum() if not trade_stats.empty else 0
        total_profit_amount = trade_stats['total_profit_amount'].sum() if not trade_stats.empty else 0
        avg_win_rate = trade_stats['win_rate'].mean() if not trade_stats.empty else 0
        total_entry_amount = trade_stats['total_entry_amount'].sum() if not trade_stats.empty else 0
        total_exit_amount = trade_stats['total_exit_amount'].sum() if not trade_stats.empty else 0

        # 基本信息和雷达图
        col1, col2 = st.columns([1, 1])

        with col1:
            grade = portrait_row.get("quality_grade", "C")
            grade_color = get_quality_grade_color(grade)
            st.markdown(f"**质量等级**: <span style='background-color:{grade_color};color:white;padding:4px 12px;border-radius:4px;font-weight:bold'>{grade}</span>", unsafe_allow_html=True)
            st.markdown(f"**股东类型**: {portrait_row.get('holder_type', 'N/A')}")
            st.markdown(f"**涉及股票**: {int(portrait_row.get('sample_stocks', 0))} 只")
            st.markdown(f"**总交易次数**: {total_trades} 次")

            # 累计盈亏展示
            profit_color = "#28a745" if total_profit_pct > 0 else "#dc3545" if total_profit_pct < 0 else "#6c757d"
            st.markdown(f"**累计盈亏**: <span style='color:{profit_color};font-size:1.2em;font-weight:bold'>{total_profit_pct:+.2f}%</span>", unsafe_allow_html=True)
            st.markdown(f"**盈亏金额**: {format_amount(total_profit_amount)}")
            st.markdown(f"**平均胜率**: {avg_win_rate:.1f}%")

            # 标签展示
            labels = []
            for label_col in ["style_label", "period_label", "industry_label", "ability_label"]:
                val = portrait_row.get(label_col)
                if val and str(val) not in ["nan", "None", ""]:
                    labels.append(str(val))
            if labels:
                st.markdown("**标签**: " + " ".join([f"<span style='background:#e9ecef;padding:2px 8px;border-radius:4px;margin-right:4px'>{l}</span>" for l in labels]), unsafe_allow_html=True)

        with col2:
            # 构建雷达图数据
            radar_data = pd.Series({
                'total_profit_pct': total_profit_pct,
                'avg_win_rate': avg_win_rate,
                'entry_win_rate_60': portrait_row.get('entry_win_rate_60', 0),
                'add_win_rate_60': portrait_row.get('add_win_rate_60', 0),
                'sample_stocks': portrait_row.get('sample_stocks', 0),
                'total_trades': total_trades
            })
            fig = render_holder_radar_chart(radar_data)
            st.plotly_chart(fig, use_container_width=True)

        # 关键指标卡片
        st.markdown("#### 关键指标")
        metric_cols = st.columns(4)
        metric_cols[0].metric("累计盈亏", f"{total_profit_pct:+.2f}%")
        metric_cols[1].metric("平均胜率", f"{avg_win_rate:.1f}%")
        metric_cols[2].metric("总交易次数", f"{total_trades}")
        metric_cols[3].metric("涉及股票", f"{int(portrait_row.get('sample_stocks', 0))}")

        # 交易记录明细
        st.markdown("---")
        st.markdown("#### 交易记录明细")

        if not trade_records.empty:
            # 添加操作图标
            trade_records['操作'] = trade_records['operation'].apply(lambda x: f"{get_operation_icon(x)} {x}")

            # 格式化金额
            trade_records['入场金额'] = trade_records['entry_amount'].apply(format_amount)
            trade_records['出场金额'] = trade_records['exit_amount'].apply(format_amount)

            # 格式化盈亏
            trade_records['盈亏比例'] = trade_records['profit_pct'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
            trade_records['盈亏金额'] = trade_records['profit_amount'].apply(format_amount)

            # 选择显示列
            display_records = trade_records[[
                'ts_code', 'stock_name', '操作', 'report_date',
                'entry_date', 'entry_price', 'entry_amount',
                'exit_date', 'exit_price', 'exit_amount',
                '盈亏比例', '盈亏金额'
            ]].copy()

            display_records.columns = [
                '股票代码', '股票名称', '操作', '报告期',
                '入场日期', '入场价格', '入场金额',
                '出场日期', '出场价格', '出场金额',
                '盈亏比例', '盈亏金额'
            ]

            st.dataframe(display_records, use_container_width=True, hide_index=True)
        else:
            st.info("暂无交易记录")

        # 按股票统计汇总
        st.markdown("---")
        st.markdown("#### 按股票统计汇总")

        if not trade_stats.empty:
            # 格式化金额和比例
            trade_stats['累计盈亏'] = trade_stats['total_profit'].apply(lambda x: f"{x:+.2f}%")
            trade_stats['胜率'] = trade_stats['win_rate'].apply(lambda x: f"{x:.1f}%")
            trade_stats['盈亏金额'] = trade_stats['total_profit_amount'].apply(format_amount)

            display_stats = trade_stats[[
                'ts_code', 'stock_name', 'operation_count', 'win_count', 'loss_count',
                '累计盈亏', '胜率', '盈亏金额'
            ]].copy()

            display_stats.columns = [
                '股票代码', '股票名称', '交易次数', '盈利次数', '亏损次数',
                '累计盈亏', '胜率', '盈亏金额'
            ]

            st.dataframe(display_stats, use_container_width=True, hide_index=True)
        else:
            st.info("暂无统计数据")

        # 当前持股明细
        st.markdown("---")
        st.markdown("#### 当前持股明细")

        if not current_holdings.empty:
            current_holdings['持股比例'] = current_holdings['hold_ratio'].apply(lambda x: f"{x:.2f}%")
            current_holdings['持股数量'] = current_holdings['hold_amount'].apply(lambda x: f"{x:,.0f}")

            display_holdings = current_holdings[[
                'ts_code', 'stock_name', '持股比例', '持股数量', 'holder_rank', 'report_date'
            ]].copy()

            display_holdings.columns = [
                '股票代码', '股票名称', '持股比例', '持股数量', '排名', '报告期'
            ]

            st.dataframe(display_holdings, use_container_width=True, hide_index=True)
        else:
            st.info("暂无持股明细")


def render_holder_eval_page(session):
    """渲染股东变化评价页面"""
    st.header("📊 股东变化评价")

    with st.spinner("加载数据..."):
        sql = "SELECT DISTINCT report_date FROM stock_top10_holder_eval_scores_tushare ORDER BY report_date DESC"
        dates_df = query_sql(session, sql, {})
        available_dates = dates_df["report_date"].tolist() if dates_df is not None and not dates_df.empty else []

    if not available_dates:
        st.warning("暂无股东变化评价数据")
        return

    with st.sidebar:
        selected_date = st.selectbox("报告期", available_dates, index=0)

    with st.spinner("加载数据..."):
        df = query_sql(session, f"SELECT * FROM stock_top10_holder_eval_scores_tushare WHERE report_date = '{selected_date}'", {})

    if df.empty:
        st.warning("该报告期暂无数据")
        return

    display_df = df.copy()

    conditions = render_filter_bar(display_df, "股东变化评价")
    if conditions:
        display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["股东变化评价"])

    loaded_conditions = render_filter_manager(session, "股东变化评价", conditions)
    if loaded_conditions:
        display_df = df.copy()
        display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["股东变化评价"])

    display_df = display_df[["ts_code", "stock_name", "industry_l2", "report_date",
                             "cr10", "delta_cr10", "hhi", "turnover_struct",
                             "total_score", "score_change_neutral", "score_stability_neutral", "score_quality_neutral",
                             "label_primary", "label_secondary"]].copy()

    display_df.columns = ["股票代码", "股票名称", "行业", "报告期",
                         "cr10", "delta_cr10", "hhi", "换手结构",
                         "总分", "结构分", "稳定分", "质量分",
                         "风向标", "次级标签"]

    st.markdown(f"**股票数量: {len(display_df)}**")

    numeric_cols = ["总分", "结构分", "稳定分", "质量分"]
    st.dataframe(colorize_numeric_columns(display_df, numeric_cols), use_container_width=True, hide_index=True)


def _render_breakout_tab(session, tab_name: str, config_key: str, sql: str, display_cols: list, col_mapping: dict, numeric_cols: list):
    """通用渲染函数：翻多事件/回踩买点"""
    with st.spinner("加载数据..."):
        df = query_sql(session, sql, {})

    if df is None or df.empty:
        st.warning("该日期暂无数据")
        return

    _render_breakout_tab_df(session, tab_name, config_key, df, display_cols, col_mapping, numeric_cols)


def _render_breakout_tab_df(session, tab_name: str, config_key: str, df: pd.DataFrame,
                             display_cols: list, col_mapping: dict, numeric_cols: list):
    """
    display_cols: 英文字段名列表
    col_mapping: {英文字段名: 中文显示名}
    numeric_cols: 英文字段名列表（要着色的列）
    """
    display_df = df.copy()
    conditions = render_filter_bar(display_df, config_key)
    if conditions:
        display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS[config_key])

    loaded_conditions = render_filter_manager(session, config_key, conditions)
    if loaded_conditions:
        display_df = apply_filters(df.copy(), loaded_conditions, TAB_FIELD_CONFIGS[config_key])

    available_cols = [c for c in display_cols if c in display_df.columns]
    display_df = display_df[available_cols].copy()
    display_df.rename(columns=col_mapping, inplace=True)

    numeric_cols_display = [col_mapping.get(c, c) for c in numeric_cols if c in available_cols]

    st.markdown(f"**股票数量: {len(display_df)}**")

    st.markdown("""
    <style>
    .element-container[data-stale="false"] .stDataFrame thead th:nth-child(1),
    .element-container[data-stale="false"] .stDataFrame tbody td:nth-child(1),
    .element-container[data-stale="false"] .stDataFrame thead th:nth-child(2),
    .element-container[data-stale="false"] .stDataFrame tbody td:nth-child(2) {
        position: sticky;
        left: 0;
        background: white;
        z-index: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    st.dataframe(colorize_numeric_columns(display_df, numeric_cols_display), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="财务评分分析",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #131722;
        }
        .stTextInput > div > div > input {
            background-color: #1e222d;
            color: #d1d4dc;
        }
        .stSelectbox > div > div > select {
            background-color: #1e222d;
            color: #d1d4dc;
        }
        .stMetric {
            background-color: #1e222d;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with get_session() as session:
        init_saved_filters_table(session)
        stock_list = load_stock_list(session)

        with st.sidebar:
            st.header("📊 财务评分分析")
            page = st.radio("页面", ["个股分析", "全股池因子表", "选股主页", "股东画像"], index=0)

        if page == "个股分析":
            if stock_list.empty:
                st.info("暂无评分数据（stock_financial_score_pool 表不存在或为空）")
            else:
                with st.sidebar:
                    st.markdown("---")
                    st.subheader("⚙️ 股票选择")

                    pinyin_input = st.text_input(
                        "输入股票名称首字母",
                        value="",
                        placeholder="例如: zgsy -> 中国石油",
                        help="输入股票名称的拼音首字母进行模糊匹配",
                    )

                    matched_stocks = match_stocks_by_pinyin(pinyin_input, stock_list)

                    selected_option = None
                    if matched_stocks.empty:
                        st.info("请输入股票名称首字母进行搜索")
                    else:
                        selected_option = st.selectbox(
                            "选择股票",
                            options=range(len(matched_stocks)),
                            format_func=lambda i: f"{matched_stocks.iloc[i]['name']} ({matched_stocks.iloc[i]['ts_code']})",
                            index=0,
                        )

                        st.markdown("---")
                        st.markdown("**维度权重说明**")
                        for dim, weight in DIMENSION_WEIGHTS.items():
                            color = DIMENSION_COLORS.get(dim, "#888888")
                            st.markdown(f"<span style='color:{color}'>●</span> {dim}: {weight*100:.0f}%", unsafe_allow_html=True)

                if selected_option is not None:
                    _render_stock_page(session, matched_stocks, selected_option)
        elif page == "全股池因子表":
            with st.sidebar:
                st.markdown("---")
                st.markdown("**维度权重说明**")
                for dim, weight in DIMENSION_WEIGHTS.items():
                    color = DIMENSION_COLORS.get(dim, "#888888")
                    st.markdown(f"<span style='color:{color}'>●</span> {dim}: {weight*100:.0f}%", unsafe_allow_html=True)

            render_pool_page(session)
        elif page == "选股主页":
            render_stock_picker_page(session, st.sidebar)

        elif page == "股东画像":
            render_holder_profile_page(session)

if __name__ == "__main__":
    main()
