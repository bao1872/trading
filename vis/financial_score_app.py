# -*- coding: utf-8 -*-
"""
个股财务评分可视化分析（Streamlit 应用）

Purpose: 展示个股近期财务数据，基于6维度评分体系进行可视化分析
Inputs: stock_financial_score_pool 表（由 financial_quarterly_score.py 批量计算生成）
Outputs: Streamlit Web 界面
How to Run: streamlit run vis/financial_score_app.py
Examples:
    streamlit run vis/financial_score_app.py
Side Effects: 仅读取数据库，无写入操作
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


from financial_factors.sample_score import FACTOR_CONFIG, DIMENSION_WEIGHTS

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
    "规模与增长": "#2962ff",
    "盈利能力": "#26a69a",
    "利润质量": "#ff9800",
    "现金创造能力": "#e91e63",
    "资产效率与资金占用": "#9c27b0",
    "边际变化与持续性": "#00bcd4",
}

DIMENSION_COLS = [
    "规模与增长_score",
    "盈利能力_score",
    "利润质量_score",
    "现金创造能力_score",
    "资产效率与资金占用_score",
    "边际变化与持续性_score",
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
    """加载股票列表（从评分池）"""
    sql = """
        SELECT DISTINCT ts_code, stock_name as name
        FROM stock_financial_score_pool
        ORDER BY stock_name
    """
    try:
        return query_sql(session, sql)
    except Exception:
        return pd.DataFrame()


def load_score_data(session, ts_code: str) -> pd.DataFrame:
    """从 stock_financial_score_pool 加载股票评分数据，缺失季度前向填充保证趋势图连续"""
    sql = """
        SELECT ts_code, stock_name, report_date, total_score,
               规模与增长_score, 盈利能力_score, 利润质量_score,
               现金创造能力_score, 资产效率与资金占用_score, 边际变化与持续性_score
        FROM stock_financial_score_pool
        WHERE ts_code = :ts_code
        ORDER BY report_date ASC
    """
    try:
        df = query_sql(session, sql, {"ts_code": ts_code})
        if df.empty:
            return df
        score_cols = [
            "total_score", "规模与增长_score", "盈利能力_score", "利润质量_score",
            "现金创造能力_score", "资产效率与资金占用_score", "边际变化与持续性_score"
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
                ORDER BY total_score DESC
            """
            df = query_sql(session, sql, {"report_date": report_date})
        else:
            sql = """
                SELECT * FROM stock_financial_score_pool
                ORDER BY ts_code, report_date DESC
            """
            df = query_sql(session, sql, {})
            df = df.groupby("ts_code").first().reset_index()
            df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
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
                   "规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化", "report_date"],
        "types": {"code": "string", "name": "string", "zscore": "numeric", "price_change": "numeric", "themes_str": "string", "concepts_str": "string",
                  "total_score": "numeric", "规模与增长": "numeric", "盈利能力": "numeric",
                  "利润质量": "numeric", "现金创造能力": "numeric", "资产效率": "numeric",
                  "边际变化": "numeric", "report_date": "string"},
        "string_fields": ["code", "name", "themes_str", "report_date", "概念"],
        "display_names": {"code": "股票代码", "name": "股票名称", "zscore": "Z分数", "price_change": "涨跌幅", "themes_str": "主题", "concepts_str": "概念",
                         "total_score": "总分", "规模与增长": "规模与增长", "盈利能力": "盈利能力",
                         "利润质量": "利润质量", "现金创造能力": "现金创造能力", "资产效率": "资产效率",
                         "边际变化": "边际变化", "report_date": "财报日期"}
    },
    "个股异动_rolling": {
        "fields": ["code", "name", "zscore", "volume_cv", "volume_spearman", "price_change", "themes_str", "concepts_str", "total_score",
                   "规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化", "report_date"],
        "types": {"code": "string", "name": "string", "zscore": "numeric", "volume_cv": "numeric",
                  "volume_spearman": "numeric", "price_change": "numeric", "themes_str": "string", "concepts_str": "string", "total_score": "numeric",
                  "规模与增长": "numeric", "盈利能力": "numeric", "利润质量": "numeric",
                  "现金创造能力": "numeric", "资产效率": "numeric", "边际变化": "numeric",
                  "report_date": "string"},
        "string_fields": ["code", "name", "themes_str", "report_date", "概念"],
        "display_names": {"code": "股票代码", "name": "股票名称", "zscore": "Z分数", "volume_cv": "CV",
                         "volume_spearman": "Spearman", "price_change": "涨跌幅", "themes_str": "主题", "concepts_str": "概念", "total_score": "总分",
                         "规模与增长": "规模与增长", "盈利能力": "盈利能力", "利润质量": "利润质量",
                         "现金创造能力": "现金创造能力", "资产效率": "资产效率", "边际变化": "边际变化",
                         "report_date": "财报日期"}
    },
    "涨停追踪": {
        "fields": ["股票代码", "股票名称", "概念", "连板天数", "连板交易日", "total_score",
                   "规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化", "主题", "财报日期"],
        "types": {"股票代码": "string", "股票名称": "string", "概念": "string", "连板天数": "numeric",
                  "连板交易日": "numeric", "total_score": "numeric", "规模与增长": "numeric",
                  "盈利能力": "numeric", "利润质量": "numeric", "现金创造能力": "numeric",
                  "资产效率": "numeric", "边际变化": "numeric", "主题": "string", "财报日期": "string"},
        "string_fields": ["股票代码", "股票名称", "概念", "主题", "财报日期"],
        "display_names": {"股票代码": "股票代码", "股票名称": "股票名称", "概念": "概念", "连板天数": "几板",
                         "连板交易日": "几天", "total_score": "总分",
                         "规模与增长": "规模与增长", "盈利能力": "盈利能力", "利润质量": "利润质量",
                         "现金创造能力": "现金创造能力", "资产效率": "资产效率", "边际变化": "边际变化",
                         "主题": "主题", "财报日期": "财报日期"}
    },
    "全股池因子表": {
        "fields": ["股票代码", "股票名称", "公告日期", "概念", "total_score",
                   "规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化"],
        "types": {"股票代码": "string", "股票名称": "string", "公告日期": "string", "概念": "string",
                  "total_score": "numeric", "规模与增长": "numeric", "盈利能力": "numeric",
                  "利润质量": "numeric", "现金创造能力": "numeric",
                  "资产效率": "numeric", "边际变化": "numeric"},
        "string_fields": ["股票代码", "股票名称", "公告日期", "概念"],
        "display_names": {"股票代码": "股票代码", "股票名称": "股票名称", "公告日期": "公告日期", "概念": "概念",
                         "total_score": "总分", "规模与增长": "规模与增长", "盈利能力": "盈利能力",
                         "利润质量": "利润质量", "现金创造能力": "现金创造能力",
                         "资产效率": "资产效率", "边际变化": "边际变化"}
    },
    "C2策略选股": {
        "fields": ["symbol", "name", "concepts", "close", "dsa_pivot_pos_01", "signed_vwap_dev_pct",
                   "w_dsa_pivot_pos_01", "bars_since_dir_change", "rope_dir", "rope_slope_atr_5",
                   "bb_pos_01", "bb_width_percentile", "total_score",
                   "规模与增长_score", "盈利能力_score", "利润质量_score",
                   "现金创造能力_score", "资产效率与资金占用_score", "边际变化与持续性_score"],
        "types": {"symbol": "string", "name": "string", "concepts": "string", "close": "numeric",
                  "dsa_pivot_pos_01": "numeric", "signed_vwap_dev_pct": "numeric",
                  "w_dsa_pivot_pos_01": "numeric", "bars_since_dir_change": "numeric",
                  "rope_dir": "numeric", "rope_slope_atr_5": "numeric",
                  "bb_pos_01": "numeric", "bb_width_percentile": "numeric", "total_score": "numeric",
                  "规模与增长_score": "numeric", "盈利能力_score": "numeric", "利润质量_score": "numeric",
                  "现金创造能力_score": "numeric", "资产效率与资金占用_score": "numeric", "边际变化与持续性_score": "numeric"},
        "string_fields": ["symbol", "name", "concepts"],
        "display_names": {"symbol": "股票代码", "name": "股票名称", "concepts": "概念", "close": "收盘价",
                         "dsa_pivot_pos_01": "日线DSA位置", "signed_vwap_dev_pct": "VWAP偏离度(%)",
                         "w_dsa_pivot_pos_01": "周线DSA位置", "bars_since_dir_change": "趋势转变Bar数",
                         "rope_dir": "Rope方向", "rope_slope_atr_5": "Rope斜率",
                         "bb_pos_01": "布林带位置", "bb_width_percentile": "布林带宽度分位", "total_score": "财务总分",
                         "规模与增长_score": "规模与增长", "盈利能力_score": "盈利能力", "利润质量_score": "利润质量",
                         "现金创造能力_score": "现金创造能力", "资产效率与资金占用_score": "资产效率", "边际变化与持续性_score": "边际变化"}
    },
    "股东画像": {
        "fields": ["holder_name_std", "holder_type", "final_holder_quality", "sample_total",
                   "win_rate_60_entry", "win_rate_60_add", "avg_excess_ret_60_entry", "avg_excess_ret_60_add"],
        "types": {"holder_name_std": "string", "holder_type": "string", "final_holder_quality": "numeric",
                  "sample_total": "numeric", "win_rate_60_entry": "numeric", "win_rate_60_add": "numeric",
                  "avg_excess_ret_60_entry": "numeric", "avg_excess_ret_60_add": "numeric"},
        "string_fields": ["holder_name_std", "holder_type"],
        "display_names": {"holder_name_std": "股东名称", "holder_type": "类型", "final_holder_quality": "最终质量分",
                         "sample_total": "总样本", "win_rate_60_entry": "入场胜率", "win_rate_60_add": "加仓胜率",
                         "avg_excess_ret_60_entry": "入场超额收益", "avg_excess_ret_60_add": "加仓超额收益"}
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
                   "规模与增长_score", "盈利能力_score", "利润质量_score",
                   "现金创造能力_score", "资产效率与资金占用_score", "边际变化与持续性_score",
                   "amt_ratio_20", "risk_tags", "concepts",
                   "score_report_date"],
        "types": {"final_launch_score": "numeric",
                  "quiet_score": "numeric", "breakout_score": "numeric",
                  "volume_score": "numeric", "strength_score": "numeric",
                  "quality_score": "numeric", "freshness_score": "numeric",
                  "close_chg": "numeric", "amt_ratio_20": "numeric",
                  "total_score": "numeric",
                  "规模与增长_score": "numeric", "盈利能力_score": "numeric",
                  "利润质量_score": "numeric", "现金创造能力_score": "numeric",
                  "资产效率与资金占用_score": "numeric", "边际变化与持续性_score": "numeric",
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
            "规模与增长_score": "规模与增长", "盈利能力_score": "盈利能力",
            "利润质量_score": "利润质量", "现金创造能力_score": "现金创造能力",
            "资产效率与资金占用_score": "资产效率", "边际变化与持续性_score": "边际变化"
        }
    },
    "翻多事件": {
        "fields": ["ts_code", "name", "event_time", "freq",
                   "breakout_quality_score",
                   "total_score",
                   "规模与增长_score",
                   "边际变化与持续性_score",
                   "price_chg",
                   "breakout_quality_grade",
                   "score_trend_total", "score_candle_total", "score_volume_total", "score_freshness_total",
                   "rope_slope_atr_5", "dist_to_rope_atr", "consolidation_bars",
                   "vol_zscore", "vol_record_days",
                   "盈利能力_score", "利润质量_score", "现金创造能力_score", "资产效率与资金占用_score",
                   "concepts", "score_report_date"],
        "types": {"ts_code": "string", "name": "string", "event_time": "string", "freq": "string",
                  "breakout_quality_score": "numeric", "breakout_quality_grade": "string",
                  "score_trend_total": "numeric", "score_candle_total": "numeric",
                  "score_volume_total": "numeric", "score_freshness_total": "numeric",
                  "rope_slope_atr_5": "numeric", "dist_to_rope_atr": "numeric", "consolidation_bars": "numeric",
                  "vol_zscore": "numeric", "vol_record_days": "numeric",
                  "total_score": "numeric",
                  "规模与增长_score": "numeric", "盈利能力_score": "numeric",
                  "利润质量_score": "numeric", "现金创造能力_score": "numeric",
                  "资产效率与资金占用_score": "numeric", "边际变化与持续性_score": "numeric",
                  "price_chg": "numeric",
                  "concepts": "string", "score_report_date": "string"},
        "string_fields": ["ts_code", "name", "event_time", "freq", "breakout_quality_grade", "concepts", "score_report_date"],
        "display_names": {"ts_code": "股票代码", "name": "股票名称", "event_time": "事件时间", "freq": "周期",
                         "breakout_quality_score": "突破质量分",
                         "total_score": "财务总分",
                         "规模与增长_score": "规模与增长", "边际变化与持续性_score": "边际变化",
                         "price_chg": "涨跌幅",
                         "breakout_quality_grade": "等级",
                         "score_trend_total": "趋势评分", "score_candle_total": "K线质量评分",
                         "score_volume_total": "量能评分", "score_freshness_total": "新鲜度评分",
                         "rope_slope_atr_5": "rope斜率", "dist_to_rope_atr": "距rope(ATR)",
                         "consolidation_bars": "盘整周期", "vol_zscore": "量能Z分", "vol_record_days": "量能记录日",
                         "盈利能力_score": "盈利能力", "利润质量_score": "利润质量",
                         "现金创造能力_score": "现金创造能力", "资产效率与资金占用_score": "资产效率",
                         "concepts": "概念", "score_report_date": "财报日期"}
    },
    "回踩买点": {
        "fields": ["ts_code", "name", "buy_time", "freq", "buy_type",
                   "breakout_quality_score",
                   "total_score",
                   "规模与增长_score",
                   "边际变化与持续性_score",
                   "price_chg",
                   "breakout_to_buy_bars",
                   "score_trend_total", "score_candle_total", "score_volume_total", "score_freshness_total",
                   "pullback_touch_support_flag", "pullback_hhhl_seen_flag",
                   "lower", "rope", "close",
                   "盈利能力_score", "利润质量_score", "现金创造能力_score", "资产效率与资金占用_score",
                   "concepts", "score_report_date"],
        "types": {"ts_code": "string", "name": "string", "buy_time": "string", "freq": "string", "buy_type": "string",
                  "breakout_quality_score": "numeric", "breakout_to_buy_bars": "numeric",
                  "score_trend_total": "numeric", "score_candle_total": "numeric",
                  "score_volume_total": "numeric", "score_freshness_total": "numeric",
                  "pullback_touch_support_flag": "numeric", "pullback_hhhl_seen_flag": "numeric",
                  "lower": "numeric", "rope": "numeric", "close": "numeric",
                  "total_score": "numeric",
                  "规模与增长_score": "numeric", "盈利能力_score": "numeric",
                  "利润质量_score": "numeric", "现金创造能力_score": "numeric",
                  "资产效率与资金占用_score": "numeric", "边际变化与持续性_score": "numeric",
                  "price_chg": "numeric",
                  "concepts": "string", "score_report_date": "string"},
        "string_fields": ["ts_code", "name", "buy_time", "freq", "buy_type", "concepts", "score_report_date"],
        "display_names": {"ts_code": "股票代码", "name": "股票名称", "buy_time": "买入时间", "freq": "周期", "buy_type": "买入类型",
                         "breakout_quality_score": "突破质量分",
                         "total_score": "财务总分",
                         "规模与增长_score": "规模与增长", "边际变化与持续性_score": "边际变化",
                         "price_chg": "涨跌幅",
                         "breakout_to_buy_bars": "间隔bar数",
                         "score_trend_total": "趋势评分", "score_candle_total": "K线质量评分",
                         "score_volume_total": "量能评分", "score_freshness_total": "新鲜度评分",
                         "pullback_touch_support_flag": "回踩支撑", "pullback_hhhl_seen_flag": "HH/HL确认",
                         "lower": "lower", "rope": "rope", "close": "收盘价",
                         "盈利能力_score": "盈利能力", "利润质量_score": "利润质量",
                         "现金创造能力_score": "现金创造能力", "资产效率与资金占用_score": "资产效率",
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


def render_watchlist_page(session, sidebar):
    """自选股管理 + K线展示页面"""
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
               规模与增长_score, 盈利能力_score, 利润质量_score,
               现金创造能力_score, 资产效率与资金占用_score, 边际变化与持续性_score
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


def render_signal_page(session):
    """渲染选股主页（6个Tab：主题_当日/rolling、概念_当日/rolling、个股异动_当日/rolling）"""

    st.header("📊 选股信号")

    available_dates = get_available_signal_dates(session)
    if not available_dates:
        st.warning("数据库中没有信号数据，请先运行 signals_theme.py 或 signals_abnormal.py 生成数据")
        return

    snapshot_date = st.selectbox(
            "选择日期",
            options=available_dates,
            format_func=lambda x: str(x)[:10] if len(str(x)) > 10 else str(x),
            index=0
        )
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "主题_当日", "主题_rolling",
        "概念_当日", "概念_rolling",
        "个股异动_当日", "个股异动_rolling",
        "涨停追踪"
    ])

    with st.spinner("加载数据..."):
        theme_df = load_signal_table(session, "theme_signals", snapshot_date)
        theme_rolling_df = load_signal_table(session, "theme_signals_rolling", snapshot_date)
        concept_df = load_signal_table(session, "concept_signals", snapshot_date)
        concept_rolling_df = load_signal_table(session, "concept_signals_rolling", snapshot_date)
        anomaly_df = load_signal_table(session, "stock_anomaly_signals", snapshot_date)
        anomaly_rolling_df = load_signal_table(session, "stock_anomaly_signals_rolling", snapshot_date)
        limit_up_df = load_limit_up_signals(session, snapshot_date)

    concept_cache_sql = "SELECT ts_code, concepts FROM stock_pools"
    concept_cache_df = query_sql(session, concept_cache_sql, {})

    financial_scores = get_latest_financial_scores(session)
    if not limit_up_df.empty:
        limit_up_df = limit_up_df.rename(columns={"ts_code": "code"})
        if not financial_scores.empty:
            fs_copy = financial_scores.rename(columns={"ts_code": "code"})
            score_cols = ["code", "total_score", "report_date"] + DIMENSION_COLS
            score_cols = [c for c in score_cols if c in fs_copy.columns]
            limit_up_df = limit_up_df.merge(
                fs_copy[score_cols],
                on="code",
                how="left",
                suffixes=("", "_dup")
            )
            dup_cols = [c for c in limit_up_df.columns if c.endswith("_dup")]
            if dup_cols:
                limit_up_df = limit_up_df.drop(columns=dup_cols)
    if not financial_scores.empty and not anomaly_df.empty:
        fs_copy = financial_scores.rename(columns={"ts_code": "code"})
        score_cols = ["code", "total_score", "report_date"] + DIMENSION_COLS
        score_cols = [c for c in score_cols if c in fs_copy.columns]
        anomaly_df = anomaly_df.merge(fs_copy[score_cols], on="code", how="left", suffixes=("", "_dup"))
        dup_cols = [c for c in anomaly_df.columns if c.endswith("_dup")]
        if dup_cols:
            anomaly_df = anomaly_df.drop(columns=dup_cols)

    if not financial_scores.empty and not anomaly_rolling_df.empty:
        fs_copy = financial_scores.rename(columns={"ts_code": "code"})
        score_cols = ["code", "total_score", "report_date"] + DIMENSION_COLS
        score_cols = [c for c in score_cols if c in fs_copy.columns]
        anomaly_rolling_df = anomaly_rolling_df.merge(fs_copy[score_cols], on="code", how="left", suffixes=("", "_dup"))
        dup_cols = [c for c in anomaly_rolling_df.columns if c.endswith("_dup")]
        if dup_cols:
            anomaly_rolling_df = anomaly_rolling_df.drop(columns=dup_cols)

    with tab1:
        st.markdown("### 主题信号（当日）")
        if not theme_df.empty:
            conditions = render_filter_bar(theme_df, "主题_当日")
            display_cols = ["theme", "avg_zscore", "strength", "concept_coverage",
                           "anomalous_concept_count", "total_concept_count", "stock_count", "total_zscore"]
            display_cols = [c for c in display_cols if c in theme_df.columns]
            display_df = theme_df[display_cols].copy()
            rename_map = {"theme": "主题", "avg_zscore": "平均Z分", "strength": "强度",
                         "concept_coverage": "概念覆盖率", "anomalous_concept_count": "异动概念数",
                         "total_concept_count": "概念总数", "stock_count": "股票数", "total_zscore": "总Z分"}
            display_df = display_df.rename(columns=rename_map)
            FINAL_COLS = ["主题", "平均Z分", "强度", "概念覆盖率", "异动概念数", "概念总数", "股票数", "总Z分"]
            display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]
            if conditions:
                display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["主题_当日"])
            loaded_conditions = render_filter_manager(session, "主题_当日", conditions)
            if loaded_conditions:
                display_df = theme_df[display_cols].copy()
                display_df = display_df.rename(columns=rename_map)
                display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]
                display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["主题_当日"])
            st.markdown(f"共 **{len(display_df)}** 条")
            numeric_cols = ["平均Z分", "强度", "概念覆盖率", "总Z分"]
            st.dataframe(colorize_numeric_columns(display_df, numeric_cols), use_container_width=True, hide_index=True, height=500)
        else:
            st.info("当日主题信号暂无数据")

    with tab2:
        st.markdown("### 主题信号（Rolling）")
        if not theme_rolling_df.empty:
            conditions = render_filter_bar(theme_rolling_df, "主题_rolling")
            display_cols = ["theme", "avg_zscore", "strength", "concept_coverage",
                           "anomalous_concept_count", "total_concept_count", "stock_count", "total_zscore"]
            display_cols = [c for c in display_cols if c in theme_rolling_df.columns]
            display_df = theme_rolling_df[display_cols].copy()
            rename_map = {"theme": "主题", "avg_zscore": "平均Z分", "strength": "强度",
                         "concept_coverage": "概念覆盖率", "anomalous_concept_count": "异动概念数",
                         "total_concept_count": "概念总数", "stock_count": "股票数", "total_zscore": "总Z分"}
            display_df = display_df.rename(columns=rename_map)
            FINAL_COLS = ["主题", "平均Z分", "强度", "概念覆盖率", "异动概念数", "概念总数", "股票数", "总Z分"]
            display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]
            if conditions:
                display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["主题_rolling"])
            loaded_conditions = render_filter_manager(session, "主题_rolling", conditions)
            if loaded_conditions:
                display_df = theme_rolling_df[display_cols].copy()
                display_df = display_df.rename(columns=rename_map)
                display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]
                display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["主题_rolling"])
            st.markdown(f"共 **{len(display_df)}** 条")
            numeric_cols = ["平均Z分", "强度", "概念覆盖率", "总Z分"]
            st.dataframe(colorize_numeric_columns(display_df, numeric_cols), use_container_width=True, hide_index=True, height=500)
        else:
            st.info("Rolling主题信号暂无数据")

    with tab3:
        st.markdown("### 概念信号（当日）")
        if not concept_df.empty:
            conditions = render_filter_bar(concept_df, "概念_当日")
            display_cols = ["concept", "theme", "avg_zscore", "normalized_strength", "intensity",
                          "anomalous_stock_count", "total_stock_count", "total_zscore"]
            display_cols = [c for c in display_cols if c in concept_df.columns]
            display_df = concept_df[display_cols].copy()
            rename_map = {"concept": "概念", "theme": "主题", "avg_zscore": "平均Z分",
                         "normalized_strength": "归一化强度", "intensity": "集中度",
                         "anomalous_stock_count": "异动股数", "total_stock_count": "成分股总数", "total_zscore": "总Z分"}
            display_df = display_df.rename(columns=rename_map)
            FINAL_COLS = ["概念", "主题", "平均Z分", "归一化强度", "集中度", "异动股数", "成分股总数", "总Z分"]
            display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]
            if conditions:
                display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["概念_当日"])
            loaded_conditions = render_filter_manager(session, "概念_当日", conditions)
            if loaded_conditions:
                display_df = concept_df[display_cols].copy()
                display_df = display_df.rename(columns=rename_map)
                display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]
                display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["概念_当日"])
            st.markdown(f"共 **{len(display_df)}** 条")
            numeric_cols = ["平均Z分", "归一化强度", "集中度", "总Z分"]
            st.dataframe(colorize_numeric_columns(display_df, numeric_cols), use_container_width=True, hide_index=True, height=500)
        else:
            st.info("当日概念信号暂无数据")

    with tab4:
        st.markdown("### 概念信号（Rolling）")
        if not concept_rolling_df.empty:
            conditions = render_filter_bar(concept_rolling_df, "概念_rolling")
            display_cols = ["concept", "theme", "avg_zscore", "normalized_strength", "intensity",
                          "anomalous_stock_count", "total_stock_count", "total_zscore"]
            display_cols = [c for c in display_cols if c in concept_rolling_df.columns]
            display_df = concept_rolling_df[display_cols].copy()
            rename_map = {"concept": "概念", "theme": "主题", "avg_zscore": "平均Z分",
                         "normalized_strength": "归一化强度", "intensity": "集中度",
                         "anomalous_stock_count": "异动股数", "total_stock_count": "成分股总数", "total_zscore": "总Z分"}
            display_df = display_df.rename(columns=rename_map)
            FINAL_COLS = ["概念", "主题", "平均Z分", "归一化强度", "集中度", "异动股数", "成分股总数", "总Z分"]
            display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]
            if conditions:
                display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["概念_rolling"])
            loaded_conditions = render_filter_manager(session, "概念_rolling", conditions)
            if loaded_conditions:
                display_df = concept_rolling_df[display_cols].copy()
                display_df = display_df.rename(columns=rename_map)
                display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]
                display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["概念_rolling"])
            st.markdown(f"共 **{len(display_df)}** 条")
            numeric_cols = ["平均Z分", "归一化强度", "集中度", "总Z分"]
            st.dataframe(colorize_numeric_columns(display_df, numeric_cols), use_container_width=True, hide_index=True, height=500)
        else:
            st.info("Rolling概念信号暂无数据")

    with tab5:
        st.markdown("### 个股异动（当日）")
        if not anomaly_df.empty:
            if not concept_cache_df.empty:
                anomaly_df = anomaly_df.merge(
                    concept_cache_df.rename(columns={"ts_code": "code"}),
                    on="code",
                    how="left",
                    suffixes=("", "_full")
                )
                if "concepts" in anomaly_df.columns:
                    anomaly_df["concepts"] = anomaly_df["concepts"].fillna(anomaly_df.get("concepts_full", ""))
                    anomaly_df = anomaly_df.drop(columns=[c for c in anomaly_df.columns if c.endswith("_full")], errors="ignore")

            conditions = render_filter_bar(anomaly_df, "个股异动_当日")
            score_cols = ["total_score"] + DIMENSION_COLS + ["report_date"] if "total_score" in anomaly_df.columns else []
            score_cols = [c for c in score_cols if c in anomaly_df.columns]
            base_cols = ["code", "name", "zscore", "concepts_str", "themes_str"]

            display_df = anomaly_df.copy()
            score_col_names = ["规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化"]
            rename_map = dict(zip([c for c in DIMENSION_COLS if c in display_df.columns], score_col_names))
            rename_map["code"] = "股票代码"
            rename_map["name"] = "股票名称"
            rename_map["total_score"] = "总分"
            rename_map["report_date"] = "财报日期"
            rename_map["themes_str"] = "主题"
            rename_map["concepts"] = "概念"
            rename_map["zscore"] = "Z分数"
            rename_map["price_change"] = "涨跌幅"
            display_df = display_df.rename(columns=rename_map)
            FINAL_COLS = ["股票代码", "股票名称", "Z分数", "涨跌幅", "总分",
                          "规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化",
                          "概念", "主题", "财报日期"]
            display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]

            if conditions:
                display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["个股异动_当日"])
            loaded_conditions = render_filter_manager(session, "个股异动_当日", conditions)
            if loaded_conditions:
                display_df = anomaly_df[[c for c in anomaly_df.columns if c not in ["themes_str", "concepts"]]].copy()
                display_df = display_df.rename(columns=rename_map)
                display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]
                display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["个股异动_当日"])
            st.markdown(f"共 **{len(display_df)}** 条")

            with st.expander("⭐ 添加到自选股"):
                options = display_df["股票代码"].tolist()
                labels = [f"{code} - {name}" for code, name in zip(display_df["股票代码"], display_df["股票名称"])]
                selected = st.multiselect("选择股票", options=options, format_func=lambda x: f"{x} - {display_df[display_df['股票代码']==x]['股票名称'].values[0]}", key="sel_tab5")
                if st.button("⭐ 添加选中到自选股", key="add_sel_tab5", disabled=len(selected) == 0):
                    count = 0
                    for code in selected:
                        row = display_df[display_df["股票代码"] == code].iloc[0]
                        if add_to_watchlist(session, code, row["股票名称"]):
                            count += 1
                    if count > 0:
                        st.success(f"已添加 {count} 只股票到自选股")

            numeric_cols = ["Z分数", "涨跌幅", "总分"] + [c for c in display_df.columns if any(x in c for x in ["规模", "盈利", "利润", "现金", "资产", "边际"])]
            pct_fmt = lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            st.dataframe(colorize_numeric_columns(display_df, numeric_cols, {"涨跌幅": pct_fmt}), use_container_width=True, hide_index=True, height=500)
        else:
            st.info("当日个股异动暂无数据")

    with tab6:
        st.markdown("### 个股异动（Rolling）")
        if not anomaly_rolling_df.empty:
            if not concept_cache_df.empty:
                anomaly_rolling_df = anomaly_rolling_df.merge(
                    concept_cache_df.rename(columns={"ts_code": "code"}),
                    on="code",
                    how="left",
                    suffixes=("", "_full")
                )
                if "concepts" in anomaly_rolling_df.columns:
                    anomaly_rolling_df["concepts"] = anomaly_rolling_df["concepts"].fillna(anomaly_rolling_df.get("concepts_full", ""))
                    anomaly_rolling_df = anomaly_rolling_df.drop(columns=[c for c in anomaly_rolling_df.columns if c.endswith("_full")], errors="ignore")

            conditions = render_filter_bar(anomaly_rolling_df, "个股异动_rolling")
            score_cols = ["total_score"] + DIMENSION_COLS + ["report_date"] if "total_score" in anomaly_rolling_df.columns else []
            score_cols = [c for c in score_cols if c in anomaly_rolling_df.columns]
            base_cols = ["code", "name", "zscore", "volume_cv", "volume_spearman", "concepts_str", "themes_str"]

            display_df = anomaly_rolling_df.copy()
            score_col_names = ["规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化"]
            rename_map = dict(zip([c for c in DIMENSION_COLS if c in display_df.columns], score_col_names))
            rename_map["code"] = "股票代码"
            rename_map["name"] = "股票名称"
            rename_map["total_score"] = "总分"
            rename_map["report_date"] = "财报日期"
            rename_map["themes_str"] = "主题"
            rename_map["concepts"] = "概念"
            rename_map["zscore"] = "Z分数"
            rename_map["volume_cv"] = "CV"
            rename_map["volume_spearman"] = "Spearman"
            rename_map["price_change"] = "涨跌幅"
            display_df = display_df.rename(columns=rename_map)
            FINAL_COLS = ["股票代码", "股票名称", "Z分数", "CV", "Spearman", "涨跌幅", "总分",
                          "规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化",
                          "概念", "主题", "财报日期"]
            display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]

            if conditions:
                display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["个股异动_rolling"])
            loaded_conditions = render_filter_manager(session, "个股异动_rolling", conditions)
            if loaded_conditions:
                display_df = anomaly_rolling_df[[c for c in anomaly_rolling_df.columns if c not in ["themes_str", "concepts"]]].copy()
                display_df = display_df.rename(columns=rename_map)
                display_df = display_df[[c for c in FINAL_COLS if c in display_df.columns]]
                display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["个股异动_rolling"])
            st.markdown(f"共 **{len(display_df)}** 条")

            with st.expander("⭐ 添加到自选股"):
                options = display_df["股票代码"].tolist()
                selected = st.multiselect("选择股票", options=options, format_func=lambda x: f"{x} - {display_df[display_df['股票代码']==x]['股票名称'].values[0]}", key="sel_tab6")
                if st.button("⭐ 添加选中到自选股", key="add_sel_tab6", disabled=len(selected) == 0):
                    count = 0
                    for code in selected:
                        row = display_df[display_df["股票代码"] == code].iloc[0]
                        if add_to_watchlist(session, code, row["股票名称"]):
                            count += 1
                    if count > 0:
                        st.success(f"已添加 {count} 只股票到自选股")

            numeric_cols = ["Z分数", "涨跌幅", "总分", "CV", "Spearman"] + [c for c in display_df.columns if any(x in c for x in ["规模", "盈利", "利润", "现金", "资产", "边际"])]
            pct_fmt = lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            st.dataframe(colorize_numeric_columns(display_df, numeric_cols, {"涨跌幅": pct_fmt}), use_container_width=True, hide_index=True, height=500)
        else:
            st.info("Rolling个股异动暂无数据")

    with tab7:
        st.markdown("### 涨停追踪")
        if not limit_up_df.empty:
            if not concept_cache_df.empty:
                limit_up_df = limit_up_df.merge(
                    concept_cache_df.rename(columns={"ts_code": "code"}),
                    on="code",
                    how="left",
                    suffixes=("", "_full")
                )
                if "concepts" in limit_up_df.columns:
                    limit_up_df["concepts"] = limit_up_df["concepts"].fillna(limit_up_df.get("concepts_full", ""))
                    limit_up_df = limit_up_df.drop(columns=[c for c in limit_up_df.columns if c.endswith("_full")], errors="ignore")

            conditions = render_filter_bar(limit_up_df, "涨停追踪")
            st.markdown("#### 涨停")
            limit_up = limit_up_df[limit_up_df["signal_type"] == "limit_up"]
            if not limit_up.empty:
                limit_up_display = limit_up[["code", "stock_name", "streak_count", "streak_trading_days", "theme", "concepts", "total_score", "report_date"] + DIMENSION_COLS].copy()
                score_col_map = {"total_score": "总分", "report_date": "财报日期", **{c: n for c, n in zip(DIMENSION_COLS, ["规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化"])}}
                limit_up_display = limit_up_display.rename(columns={
                    "code": "股票代码", "stock_name": "股票名称", "streak_count": "几板",
                    "streak_trading_days": "几天", "theme": "主题", "concepts": "概念"
                })
                limit_up_display = limit_up_display.rename(columns=score_col_map)
                FINAL_COLS = ["股票代码", "股票名称", "几板", "几天", "总分",
                              "规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化",
                              "概念", "主题", "财报日期"]
                limit_up_display = limit_up_display[[c for c in FINAL_COLS if c in limit_up_display.columns]]

                if conditions:
                    limit_up_display = apply_filters(limit_up_display, conditions, TAB_FIELD_CONFIGS["涨停追踪"])
                loaded_conditions = render_filter_manager(session, "涨停追踪", conditions)
                if loaded_conditions:
                    limit_up_display = limit_up_df.copy()
                    limit_up_display = limit_up_display.rename(columns=score_col_map)
                    limit_up_display = limit_up_display[[c for c in FINAL_COLS if c in limit_up_display.columns]]
                    limit_up_display = apply_filters(limit_up_display, loaded_conditions, TAB_FIELD_CONFIGS["涨停追踪"])
                st.markdown(f"共 **{len(limit_up_display)}** 条")

                with st.expander("⭐ 添加涨停到自选股"):
                    options = limit_up_display["股票代码"].tolist()
                    selected = st.multiselect("选择股票", options=options, format_func=lambda x: f"{x} - {limit_up_display[limit_up_display['股票代码']==x]['股票名称'].values[0]}", key="sel_tab7_up")
                    if st.button("⭐ 添加选中到自选股", key="add_sel_tab7_up", disabled=len(selected) == 0):
                        count = 0
                        for code in selected:
                            row = limit_up_display[limit_up_display["股票代码"] == code].iloc[0]
                            if add_to_watchlist(session, code, row["股票名称"]):
                                count += 1
                        if count > 0:
                            st.success(f"已添加 {count} 只股票到自选股")

                numeric_cols = ["几板", "几天", "总分", "规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化"]
                st.dataframe(colorize_numeric_columns(limit_up_display, numeric_cols), use_container_width=True, hide_index=True, height=300)
            else:
                st.info("当日无涨停")

            st.markdown("#### 跌停")
            limit_down = limit_up_df[limit_up_df["signal_type"] == "limit_down"]
            if not limit_down.empty:
                limit_down_display = limit_down[["code", "stock_name", "streak_count", "streak_trading_days", "theme", "total_score", "report_date"] + DIMENSION_COLS].copy()
                score_col_map = {"total_score": "总分", "report_date": "财报日期", **{c: n for c, n in zip(DIMENSION_COLS, ["规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化"])}}
                limit_down_display = limit_down_display.rename(columns={
                    "code": "股票代码", "stock_name": "股票名称", "streak_count": "几板",
                    "streak_trading_days": "几天", "theme": "主题"
                })
                limit_down_display = limit_down_display.rename(columns=score_col_map)
                FINAL_COLS = ["股票代码", "股票名称", "几板", "几天", "总分",
                              "规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化",
                              "主题", "财报日期"]
                limit_down_display = limit_down_display[[c for c in FINAL_COLS if c in limit_down_display.columns]]

                if conditions:
                    limit_down_display = apply_filters(limit_down_display, conditions, TAB_FIELD_CONFIGS["涨停追踪"])
                st.markdown(f"共 **{len(limit_down_display)}** 条")

                with st.expander("⭐ 添加跌停到自选股"):
                    options = limit_down_display["股票代码"].tolist()
                    selected = st.multiselect("选择股票", options=options, format_func=lambda x: f"{x} - {limit_down_display[limit_down_display['股票代码']==x]['股票名称'].values[0]}", key="sel_tab7_down")
                    if st.button("⭐ 添加选中到自选股", key="add_sel_tab7_down", disabled=len(selected) == 0):
                        count = 0
                        for code in selected:
                            row = limit_down_display[limit_down_display["股票代码"] == code].iloc[0]
                            if add_to_watchlist(session, code, row["股票名称"]):
                                count += 1
                        if count > 0:
                            st.success(f"已添加 {count} 只股票到自选股")

                st.dataframe(colorize_numeric_columns(limit_down_display, numeric_cols), use_container_width=True, hide_index=True, height=300)
            else:
                st.info("当日无跌停")
        else:
            st.info("涨停追踪暂无数据")


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
    """渲染6维度柱状图"""
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
            text="6维度评分",
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

    key_factors = ["q_rev_yoy", "q_np_parent_yoy", "q_gross_margin", "q_cfo_to_np_parent", "roa_parent"]
    key_factors = [f for f in key_factors if f in df.columns]
    display_cols.extend(key_factors)

    if "concepts" in df.columns:
        display_cols.append("concepts")

    display_df = df[display_cols].copy()

    col_names = ["股票代码", "股票名称", "报告期", "公告日期", "总分",
                 "规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化"]
    col_names = col_names[:len(display_cols) - len(key_factors) - (1 if "concepts" in df.columns else 0)]
    display_df.columns = col_names + [f for f in key_factors] + (["概念"] if "concepts" in df.columns else [])

    conditions = render_filter_bar(display_df, "全股池因子表")
    if conditions:
        display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["全股池因子表"])

    loaded_conditions = render_filter_manager(session, "全股池因子表", conditions)
    if loaded_conditions:
        display_df = df.copy()
        display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["全股池因子表"])

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

    color_cols = ["总分", "规模与增长", "盈利能力", "利润质量", "现金创造能力", "资产效率", "边际变化"]
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

    with st.spinner("加载评分数据..."):
        score_df = load_score_data(session, ts_code)

    if score_df.empty:
        st.warning(f"该股票暂无评分数据")
        return

    report_dates = score_df["report_date"].unique().tolist()
    latest_report = report_dates[-1] if report_dates else None

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        total_score = score_df[score_df["report_date"] == latest_report]["total_score"].values[0]
        st.metric("总分", format_score(total_score))

    with col2:
        score = score_df[score_df["report_date"] == latest_report]["规模与增长_score"].values[0]
        st.metric("规模与增长", format_score(score))

    with col3:
        score = score_df[score_df["report_date"] == latest_report]["盈利能力_score"].values[0]
        st.metric("盈利能力", format_score(score))

    with col4:
        score = score_df[score_df["report_date"] == latest_report]["利润质量_score"].values[0]
        st.metric("利润质量", format_score(score))

    with col5:
        score = score_df[score_df["report_date"] == latest_report]["现金创造能力_score"].values[0]
        st.metric("现金创造能力", format_score(score))

    with col6:
        score = score_df[score_df["report_date"] == latest_report]["资产效率与资金占用_score"].values[0]
        st.metric("资产效率", format_score(score))

    with col7:
        score = score_df[score_df["report_date"] == latest_report]["边际变化与持续性_score"].values[0]
        st.metric("边际变化", format_score(score))

    st.markdown("---")

    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown("#### 最新报告期评分")
        st.markdown(f"**{format_report_date(latest_report)}**")
        dim_chart = render_dimension_chart(score_df, latest_report)
        if dim_chart:
            st.plotly_chart(dim_chart, use_container_width=True)

    with chart_col2:
        st.markdown("#### 评分历史趋势")
        trend_chart = render_trend_chart(score_df.tail(8))
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True)

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
        display_df.columns = ["报告期", "总分", "规模与增长", "盈利能力", "利润质量",
                               "现金创造能力", "资产效率", "边际变化"]
        for col in display_df.columns[1:]:
            display_df[col] = display_df[col].apply(format_score)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 十大流通股东评价")

    holder_sql = """
        SELECT h.holder_rank, h.holder_name, h.holder_type, h.hold_ratio, h.hold_change, h.hold_amount,
               p.final_holder_quality, h.report_date, h.ann_date
        FROM stock_top10_holders_tushare h
        LEFT JOIN stock_top10_holder_profiles_tushare p ON h.holder_name = p.holder_name_std
        WHERE h.ts_code = :ts_code
        AND h.report_date = (SELECT MAX(report_date) FROM stock_top10_holders_tushare WHERE ts_code = :ts_code)
        ORDER BY h.holder_rank
    """
    holder_detail_df = query_sql(session, holder_sql, {"ts_code": ts_code})

    eval_sql = "SELECT * FROM stock_top10_holder_eval_scores_tushare WHERE ts_code = :ts_code ORDER BY report_date DESC LIMIT 1"
    holder_eval_df = query_sql(session, eval_sql, {"ts_code": ts_code})

    if holder_detail_df is not None and not holder_detail_df.empty:
        eval_row = holder_eval_df.iloc[0] if holder_eval_df is not None and not holder_eval_df.empty else {}

        h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns(5)
        total = eval_row.get("total_score", 0)
        h_col1.metric("总分", f"{total:.1f}" if pd.notna(total) else "N/A")

        score_change = eval_row.get("score_change_neutral", 0)
        h_col2.metric("结构分", f"{score_change:.1f}" if pd.notna(score_change) else "N/A")

        score_stability = eval_row.get("score_stability_neutral", 0)
        h_col3.metric("稳定分", f"{score_stability:.1f}" if pd.notna(score_stability) else "N/A")

        score_quality = eval_row.get("score_quality_neutral", 0)
        h_col4.metric("质量分", f"{score_quality:.1f}" if pd.notna(score_quality) else "N/A")

        label = eval_row.get("label_primary", "N/A")
        h_col5.metric("风向标", str(label) if pd.notna(label) else "N/A")

        cr10 = eval_row.get("cr10")
        delta_cr10 = eval_row.get("delta_cr10")
        if pd.notna(cr10) and pd.notna(delta_cr10):
            st.markdown(f"**前十股东持股**: {cr10:.1%} (环比 {delta_cr10:+.1%})")

        entry = eval_row.get("entry_count", 0)
        exit = eval_row.get("exit_count", 0)
        add = eval_row.get("add_count", 0)
        reduce = eval_row.get("reduce_count", 0)
        st.markdown(f"**股东进出**: 新进{int(entry) if pd.notna(entry) else 0} 退出{int(exit) if pd.notna(exit) else 0} 加仓{int(add) if pd.notna(add) else 0} 减仓{int(reduce) if pd.notna(reduce) else 0}")

        st.markdown("##### 股东明细")
        detail_display = holder_detail_df[["holder_rank", "holder_name", "holder_type", "hold_ratio", "hold_change", "hold_amount", "final_holder_quality", "report_date", "ann_date"]].copy()
        detail_display.columns = ["排名", "股东名称", "类型", "持股比例", "持股变化(股)", "持股金额", "质量分", "报告日期", "公告日期"]
        detail_display["持股比例"] = detail_display["持股比例"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        detail_display["持股变化(股)"] = detail_display["持股变化(股)"].apply(lambda x: f"{int(x):+,}股" if pd.notna(x) else "N/A")

        def calc_change_ratio(row):
            try:
                change = row["持股变化(股)"]
                amount = row["持股金额"]
                if pd.isna(change) or pd.isna(amount) or amount == 0:
                    return "N/A"
                change_val = int(str(change).replace(",", "").replace("股", "")) if isinstance(change, str) else change
                prev_amount = amount - change_val
                if prev_amount == 0:
                    return "N/A"
                return f"{change_val / prev_amount * 100:.2f}%"
            except:
                return "N/A"

        detail_display["变化比例"] = detail_display.apply(calc_change_ratio, axis=1)
        detail_display["质量分"] = detail_display["质量分"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        st.dataframe(detail_display[["排名", "股东名称", "类型", "持股比例", "持股变化(股)", "变化比例", "质量分", "报告日期", "公告日期"]], use_container_width=True, hide_index=True)
    else:
        st.info("暂无股东明细数据")


def render_holder_profile_page(session):
    """渲染股东画像页面"""
    st.header("📊 股东画像")

    with st.spinner("加载数据..."):
        sql = "SELECT * FROM stock_top10_holder_profiles_tushare"
        df = query_sql(session, sql, {})

    if df.empty:
        st.warning("暂无股东画像数据")
        return

    cols_to_show = ["holder_name_std", "holder_type", "sample_total", "sample_entry",
                    "sample_add", "sample_reduce", "sample_hold",
                    "win_rate_60_entry", "win_rate_60_add",
                    "avg_excess_ret_60_entry", "avg_excess_ret_60_add",
                    "posterior_score_shrink", "final_holder_quality"]

    display_df = df.copy()
    display_df = display_df[cols_to_show]

    conditions = render_filter_bar(display_df, "股东画像")
    if conditions:
        display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["股东画像"])

    loaded_conditions = render_filter_manager(session, "股东画像", conditions)
    if loaded_conditions:
        display_df = df.copy()
        display_df = display_df[cols_to_show]
        display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["股东画像"])

    display_df.columns = ["股东名称", "类型", "总样本", "入场", "加仓", "减仓", "持有",
                         "入场胜率", "加仓胜率", "入场超额收益", "加仓超额收益",
                         "后验质量分", "最终质量分"]

    st.markdown(f"**股东数量: {len(display_df)}**")

    numeric_cols = ["入场胜率", "加仓胜率", "入场超额收益", "加仓超额收益", "后验质量分", "最终质量分"]

    clicked_holder = None
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown("点击股东名称查看持股明细:")
    with cols[1]:
        holder_names = display_df["股东名称"].tolist()
        clicked_holder = st.selectbox("选择股东", options=["(未选择)"] + holder_names, label_visibility="collapsed")

    if len(display_df) > 1000:
        st.info("数据量较大，仅对当前页应用颜色渲染")
        render_paginated_dataframe(display_df, numeric_cols=numeric_cols, page_size=100, key="holder_profile")
    else:
        st.dataframe(colorize_numeric_columns(display_df, numeric_cols), use_container_width=True, hide_index=True)

    if clicked_holder and clicked_holder != "(未选择)":
        st.markdown(f"### {clicked_holder} 持股明细")

        holder_stock_sql = """
            SELECT h.ts_code, h.stock_name, h.hold_ratio, h.hold_change, h.report_date,
                   s.total_score, s.total_score_industry_neutral
            FROM stock_top10_holders_tushare h
            LEFT JOIN stock_top10_holder_eval_scores_tushare s
                   ON h.ts_code = s.ts_code AND h.report_date = s.report_date
            WHERE h.holder_name = :holder_name
            AND h.report_date = (SELECT MAX(report_date) FROM stock_top10_holders_tushare WHERE holder_name = :holder_name)
            ORDER BY h.hold_ratio DESC
        """
        holder_stock_df = query_sql(session, holder_stock_sql, {"holder_name": clicked_holder})

        if holder_stock_df is not None and not holder_stock_df.empty:
            holder_stock_df.columns = ["股票代码", "股票名称", "持股比例", "持股变化", "报告期", "总分", "行业中性分"]
            st.dataframe(holder_stock_df, use_container_width=True, hide_index=True)
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


def render_first_day_launch_page(session):
    """渲染形态选股页面"""
    st.header("📊 形态选股")

    def get_trading_days(days=90):
        from datetime import datetime, timedelta
        dates = []
        today = datetime.now()
        for i in range(days):
            date = today - timedelta(days=i)
            if date.weekday() < 5:
                dates.append(date.strftime('%Y-%m-%d'))
        return dates

    def get_prev_trading_day(date_str):
        from datetime import datetime, timedelta
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        for i in range(1, 10):
            prev = dt - timedelta(days=i)
            if prev.weekday() < 5:
                return prev.strftime('%Y-%m-%d')
        return date_str

    def load_breakout_events_with_price_chg(session, table_name, date_col, selected_date, config_key):
        """分步加载事件数据 + Python 计算涨跌幅"""
        date_filter = f"{date_col} LIKE :date || '%'"
        event_sql = f"""
            SELECT d.*, c.concepts,
                   SUBSTR(d.{date_col}, 1, 10) as trade_date
            FROM {table_name} d
            LEFT JOIN stock_pools c ON d.ts_code = c.ts_code
            WHERE {date_filter}
        """
        df_events = query_sql(session, event_sql, {"date": selected_date})

        if df_events is None or df_events.empty:
            return None

        ts_codes = df_events['ts_code'].unique().tolist()
        ts_codes_str = "','".join(ts_codes)

        score_sql = f"""
            SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY report_date DESC) as rn
                FROM stock_financial_score_pool
            ) WHERE rn = 1 AND ts_code IN ('{ts_codes_str}')
        """
        df_score = query_sql(session, score_sql, {})

        today_k_sql = f"""
            SELECT ts_code, close
            FROM stock_k_data
            WHERE freq = 'd' AND bar_time::date = :date
            AND ts_code IN ('{ts_codes_str}')
        """
        df_today = query_sql(session, today_k_sql, {"date": selected_date})

        prev_date = get_prev_trading_day(selected_date)
        prev_k_sql = f"""
            SELECT ts_code, close as prev_close
            FROM stock_k_data
            WHERE freq = 'd' AND bar_time::date = :date
            AND ts_code IN ('{ts_codes_str}')
        """
        df_prev = query_sql(session, prev_k_sql, {"date": prev_date})

        if df_score is not None and not df_score.empty:
            score_cols = ['ts_code', 'report_date', 'total_score',
                         '规模与增长_score', '盈利能力_score', '利润质量_score',
                         '现金创造能力_score', '资产效率与资金占用_score', '边际变化与持续性_score']
            score_cols_exist = [c for c in score_cols if c in df_score.columns]
            df_events = df_events.merge(df_score[score_cols_exist], on='ts_code', how='left')

        if df_today is not None and not df_today.empty:
            df_events = df_events.merge(
                df_today[['ts_code', 'close']].rename(columns={'close': 'kd_close'}),
                on='ts_code',
                how='left'
            )
        if df_prev is not None and not df_prev.empty:
            df_events = df_events.merge(
                df_prev[['ts_code', 'prev_close']],
                on='ts_code',
                how='left'
            )
            if 'kd_close' in df_events.columns and 'prev_close' in df_events.columns:
                df_events['price_chg'] = (df_events['kd_close'] - df_events['prev_close']) / df_events['prev_close'] * 100

        return df_events

    def load_c2_selections(session, select_date: str) -> pd.DataFrame:
        """加载指定日期的 C2 策略选股结果（关联股票名称、财务分数、概念）"""
        sql = """
        SELECT
            c2.symbol,
            c2.select_date,
            c2.signal_date,
            c2.close,
            c2.dsa_pivot_pos_01,
            c2.signed_vwap_dev_pct,
            c2.w_dsa_pivot_pos_01,
            c2.bars_since_dir_change,
            c2.rope_dir,
            c2.rope_slope_atr_5,
            c2.bb_pos_01,
            c2.bb_width_percentile,
            sp.name,
            sp.concepts,
            fs.total_score,
            fs."规模与增长_score",
            fs."盈利能力_score",
            fs."利润质量_score",
            fs."现金创造能力_score",
            fs."资产效率与资金占用_score",
            fs."边际变化与持续性_score"
        FROM c2_strategy_selections c2
        LEFT JOIN stock_pools sp ON c2.symbol = sp.ts_code
        LEFT JOIN stock_financial_score_pool fs ON c2.symbol = fs.ts_code
            AND fs.report_date = (
                SELECT MAX(report_date) FROM stock_financial_score_pool
                WHERE ts_code = c2.symbol AND report_date::date <= c2.select_date
            )
        WHERE c2.select_date = :select_date
        ORDER BY c2.signed_vwap_dev_pct ASC
        """
        return query_sql(session, sql, {"select_date": select_date})

    with st.sidebar:
        available_dates = get_trading_days(90)
        selected_date = st.selectbox("日期", available_dates)

    tabs = st.tabs(["翻多事件", "回踩买点", "C2策略选股"])

    with tabs[0]:
        st.markdown("### 翻多事件")
        df_dir_turn = load_breakout_events_with_price_chg(
            session, "breakout_dir_turn_events", "event_time", selected_date, "翻多事件"
        )
        if df_dir_turn is None or df_dir_turn.empty:
            st.warning("该日期暂无翻多事件数据")
        else:
            _render_breakout_tab_df(session, "翻多事件", "翻多事件", df_dir_turn,
                ["ts_code", "name", "event_time", "freq",
                 "breakout_quality_score",
                 "total_score",
                 "规模与增长_score",
                 "边际变化与持续性_score",
                 "price_chg",
                 "breakout_quality_grade",
                 "score_trend_total", "score_candle_total", "score_volume_total", "score_freshness_total",
                 "rope_slope_atr_5", "dist_to_rope_atr", "consolidation_bars",
                 "vol_zscore", "vol_record_days",
                 "盈利能力_score", "利润质量_score", "现金创造能力_score", "资产效率与资金占用_score",
                 "concepts", "score_report_date"],
                {"ts_code": "股票代码", "name": "股票名称", "event_time": "事件时间", "freq": "周期",
                 "breakout_quality_score": "突破质量分",
                 "total_score": "财务总分",
                 "规模与增长_score": "规模与增长",
                 "边际变化与持续性_score": "边际变化",
                 "price_chg": "涨跌幅",
                 "breakout_quality_grade": "等级",
                 "score_trend_total": "趋势评分", "score_candle_total": "K线质量评分", "score_volume_total": "量能评分", "score_freshness_total": "新鲜度评分",
                 "rope_slope_atr_5": "rope斜率", "dist_to_rope_atr": "距rope(ATR)", "consolidation_bars": "盘整周期",
                 "vol_zscore": "量能Z分", "vol_record_days": "量能记录日",
                 "盈利能力_score": "盈利能力", "利润质量_score": "利润质量", "现金创造能力_score": "现金创造能力", "资产效率与资金占用_score": "资产效率",
                 "concepts": "概念", "score_report_date": "财报日期"},
                ["breakout_quality_score", "total_score", "规模与增长_score", "边际变化与持续性_score", "price_chg",
                 "score_trend_total", "score_candle_total", "score_volume_total", "score_freshness_total",
                 "rope_slope_atr_5", "dist_to_rope_atr", "consolidation_bars", "vol_zscore", "vol_record_days",
                 "盈利能力_score", "利润质量_score", "现金创造能力_score", "资产效率与资金占用_score"]
            )

    with tabs[1]:
        st.markdown("### 回踩买点")
        df_pullback = load_breakout_events_with_price_chg(
            session, "breakout_pullback_buy_events", "buy_time", selected_date, "回踩买点"
        )
        if df_pullback is None or df_pullback.empty:
            st.warning("该日期暂无回踩买点数据")
        else:
            _render_breakout_tab_df(session, "回踩买点", "回踩买点", df_pullback,
                ["ts_code", "name", "buy_time", "freq", "buy_type",
                 "breakout_quality_score",
                 "total_score",
                 "规模与增长_score",
                 "边际变化与持续性_score",
                 "price_chg",
                 "breakout_to_buy_bars",
                 "score_trend_total", "score_candle_total", "score_volume_total", "score_freshness_total",
                 "pullback_touch_support_flag", "pullback_hhhl_seen_flag",
                 "lower", "rope", "close",
                 "盈利能力_score", "利润质量_score", "现金创造能力_score", "资产效率与资金占用_score",
                 "concepts", "score_report_date"],
                {"ts_code": "股票代码", "name": "股票名称", "buy_time": "买入时间", "freq": "周期", "buy_type": "买入类型",
                 "breakout_quality_score": "突破质量分",
                 "total_score": "财务总分",
                 "规模与增长_score": "规模与增长",
                 "边际变化与持续性_score": "边际变化",
                 "price_chg": "涨跌幅",
                 "breakout_to_buy_bars": "间隔bar数",
                 "score_trend_total": "趋势评分", "score_candle_total": "K线质量评分", "score_volume_total": "量能评分", "score_freshness_total": "新鲜度评分",
                 "pullback_touch_support_flag": "回踩支撑", "pullback_hhhl_seen_flag": "HH/HL确认",
                 "lower": "lower", "rope": "rope", "close": "收盘价",
                 "盈利能力_score": "盈利能力", "利润质量_score": "利润质量", "现金创造能力_score": "现金创造能力", "资产效率与资金占用_score": "资产效率",
                 "concepts": "概念", "score_report_date": "财报日期"},
                ["breakout_quality_score", "total_score", "规模与增长_score", "边际变化与持续性_score", "price_chg",
                 "breakout_to_buy_bars", "score_trend_total", "score_candle_total", "score_volume_total", "score_freshness_total",
                 "lower", "rope", "close",
                 "盈利能力_score", "利润质量_score", "现金创造能力_score", "资产效率与资金占用_score"]
            )

    with tabs[2]:
        st.markdown("### C2 策略选股结果")

        df_c2 = load_c2_selections(session, selected_date)

        if df_c2 is None or df_c2.empty:
            st.warning("该日期暂无 C2 策略选股数据")
        else:
            _render_breakout_tab_df(
                session,
                "C2策略选股",
                "C2策略选股",
                df_c2,
                # 显示列（参考回踩买点 Tab 的字段顺序）
                ["symbol", "name", "close",
                 "dsa_pivot_pos_01", "signed_vwap_dev_pct",
                 "w_dsa_pivot_pos_01", "bars_since_dir_change", "rope_dir",
                 "rope_slope_atr_5", "bb_pos_01", "bb_width_percentile",
                 "total_score",
                 "规模与增长_score", "盈利能力_score", "利润质量_score",
                 "现金创造能力_score", "资产效率与资金占用_score", "边际变化与持续性_score",
                 "concepts"],
                # 列名映射（中文）
                {"symbol": "股票代码", "name": "股票名称", "close": "收盘价",
                 "dsa_pivot_pos_01": "日线DSA位置", "signed_vwap_dev_pct": "VWAP偏离度(%)",
                 "w_dsa_pivot_pos_01": "周线DSA位置", "bars_since_dir_change": "趋势转变Bar数",
                 "rope_dir": "Rope方向", "rope_slope_atr_5": "Rope斜率",
                 "bb_pos_01": "布林带位置", "bb_width_percentile": "布林带宽度分位",
                 "total_score": "财务总分",
                 "规模与增长_score": "规模与增长", "盈利能力_score": "盈利能力",
                 "利润质量_score": "利润质量", "现金创造能力_score": "现金创造能力",
                 "资产效率与资金占用_score": "资产效率", "边际变化与持续性_score": "边际变化",
                 "concepts": "概念"},
                # 数值列（用于颜色标记）
                ["close", "dsa_pivot_pos_01", "signed_vwap_dev_pct", "w_dsa_pivot_pos_01",
                 "bb_pos_01", "bb_width_percentile", "total_score",
                 "规模与增长_score", "盈利能力_score", "利润质量_score",
                 "现金创造能力_score", "资产效率与资金占用_score", "边际变化与持续性_score"]
            )


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
            page = st.radio("页面", ["个股分析", "全股池因子表", "选股主页", "自选股", "股东画像", "股东变化评价", "形态选股"], index=0)

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
            with st.sidebar:
                st.markdown("---")
                st.markdown("**维度权重说明**")
                for dim, weight in DIMENSION_WEIGHTS.items():
                    color = DIMENSION_COLORS.get(dim, "#888888")
                    st.markdown(f"<span style='color:{color}'>●</span> {dim}: {weight*100:.0f}%", unsafe_allow_html=True)

            render_signal_page(session)
        elif page == "自选股":
            render_watchlist_page(session, st.sidebar)

        elif page == "股东画像":
            render_holder_profile_page(session)

        elif page == "股东变化评价":
            render_holder_eval_page(session)

        elif page == "形态选股":
            render_first_day_launch_page(session)

if __name__ == "__main__":
    main()
