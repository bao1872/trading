# -*- coding: utf-8 -*-
"""
财务评分与选股结果可视化分析（Streamlit 应用）

Purpose: 展示个股财务评分、选股结果、ATR Rope指标等多维度数据
Inputs:
    - financial_scores 表（新财务评分系统 - 5维度14指标）
    - stock_financial_score_pool 表（旧财务评分系统 - 6维度，用于其他页面）
    - stock_selection_results 表（选股结果：ATR Rope震荡上轨信号、POC斜率、Z-Score指标）
    - stock_k_data 表（K线数据，用于ATR Rope和BSM指标计算）
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
import datetime
from typing import Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from decimal import Decimal, ROUND_HALF_UP
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

# 新财务评分系统维度配置 (financial_statement_scorecard.py)
# 五大维度，满分100分
DIMENSION_CONFIG = {
    "main_profit_score": {"name": "主业盈利改善", "weight": 45, "color": "#2962ff"},
    "revenue_expense_efficiency_score": {"name": "收入与费用效率", "weight": 20, "color": "#26a69a"},
    "cashflow_validation_score": {"name": "现金流验证", "weight": 15, "color": "#ff9800"},
    "working_capital_quality_score": {"name": "营运资本质量", "weight": 15, "color": "#e91e63"},
    "investment_asset_efficiency_score": {"name": "投入与资产效率", "weight": 5, "color": "#9c27b0"},
}

DIMENSION_COLS = list(DIMENSION_CONFIG.keys())

DIMENSION_COLORS = {v["name"]: v["color"] for k, v in DIMENSION_CONFIG.items()}

# 14项核心指标配置
METRIC_CONFIG = {
    # 主业盈利改善 (45分)
    "F1_core_profit_improve_strength": {"name": "主业经营利润改善强度", "weight": 20, "dimension": "main_profit_score"},
    "F2_core_profit_acceleration": {"name": "主业经营利润加速度", "weight": 15, "dimension": "main_profit_score"},
    "F3_parent_np_improve_strength": {"name": "归母净利润改善强度", "weight": 10, "dimension": "main_profit_score"},
    # 收入与费用效率 (20分)
    "F4_revenue_momentum": {"name": "收入动量", "weight": 8, "dimension": "revenue_expense_efficiency_score"},
    "F5_gross_margin_expansion": {"name": "毛利率扩张", "weight": 7, "dimension": "revenue_expense_efficiency_score"},
    "F6_expense_ratio_improvement": {"name": "期间费用率改善", "weight": 5, "dimension": "revenue_expense_efficiency_score"},
    # 现金流验证 (15分)
    "F7_sales_cash_collection_improvement": {"name": "销售收现率改善", "weight": 5, "dimension": "cashflow_validation_score"},
    "F8_ocf_improve_strength": {"name": "经营现金流改善强度", "weight": 5, "dimension": "cashflow_validation_score"},
    "F9_profit_cash_gap_improvement": {"name": "利润现金背离改善", "weight": 5, "dimension": "cashflow_validation_score"},
    # 营运资本质量 (15分)
    "F10_ar_pressure_improvement": {"name": "应收压力改善", "weight": 5, "dimension": "working_capital_quality_score"},
    "F11_inventory_pressure_improvement": {"name": "存货压力改善", "weight": 5, "dimension": "working_capital_quality_score"},
    "F12_contract_liab_improvement": {"name": "合同负债改善", "weight": 5, "dimension": "working_capital_quality_score"},
    # 投入与资产效率 (5分)
    "F13_capex_efficiency_improvement": {"name": "资本开支效率改善", "weight": 2, "dimension": "investment_asset_efficiency_score"},
    "F14_asset_turnover_improvement": {"name": "资产周转率改善", "weight": 3, "dimension": "investment_asset_efficiency_score"},
}

# 风险标记配置
RISK_FLAG_CONFIG = {
    "cashflow_red_flag": {"name": "现金流红灯", "desc": "利润正但现金流负", "color": "#ef5350"},
    "ar_pressure_worsen_flag": {"name": "应收压力恶化", "desc": "应收账款压力上升", "color": "#ff9800"},
    "inventory_pressure_worsen_flag": {"name": "存货压力恶化", "desc": "存货压力上升", "color": "#ff9800"},
    "core_profit_worsen_flag": {"name": "核心利润恶化", "desc": "核心利润下滑", "color": "#ef5350"},
}


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
    """从 financial_scores 加载股票评分数据，缺失季度前向填充保证趋势图连续"""
    sql = """
        SELECT ts_code, end_date as report_date, ann_date, total_score,
               main_profit_score, revenue_expense_efficiency_score, cashflow_validation_score,
               working_capital_quality_score, investment_asset_efficiency_score,
               cashflow_red_flag, ar_pressure_worsen_flag, inventory_pressure_worsen_flag, core_profit_worsen_flag,
               F1_core_profit_improve_strength, F2_core_profit_acceleration, F3_parent_np_improve_strength,
               F4_revenue_momentum, F5_gross_margin_expansion, F6_expense_ratio_improvement,
               F7_sales_cash_collection_improvement, F8_ocf_improve_strength, F9_profit_cash_gap_improvement,
               F10_ar_pressure_improvement, F11_inventory_pressure_improvement, F12_contract_liab_improvement,
               F13_capex_efficiency_improvement, F14_asset_turnover_improvement,
               F1_score, F2_score, F3_score, F4_score, F5_score, F6_score, F7_score, F8_score,
               F9_score, F10_score, F11_score, F12_score, F13_score, F14_score
        FROM financial_scores
        WHERE ts_code = :ts_code
        ORDER BY end_date ASC
    """
    try:
        df = query_sql(session, sql, {"ts_code": ts_code})
        if df.empty:
            return df
        # 前向填充评分数据保证趋势图连续
        score_cols = ["total_score"] + DIMENSION_COLS
        present_cols = [c for c in score_cols if c in df.columns]
        if present_cols:
            df[present_cols] = df[present_cols].ffill()
        return df
    except Exception:
        return pd.DataFrame()


def load_factor_scores(session, ts_code: str, report_date: str) -> pd.DataFrame:
    """加载指定报告期的因子分数（从 financial_scores 表）"""
    sql = """
        SELECT ts_code, end_date as report_date, ann_date, total_score,
               main_profit_score, revenue_expense_efficiency_score, cashflow_validation_score,
               working_capital_quality_score, investment_asset_efficiency_score,
               cashflow_red_flag, ar_pressure_worsen_flag, inventory_pressure_worsen_flag, core_profit_worsen_flag,
               F1_core_profit_improve_strength, F2_core_profit_acceleration, F3_parent_np_improve_strength,
               F4_revenue_momentum, F5_gross_margin_expansion, F6_expense_ratio_improvement,
               F7_sales_cash_collection_improvement, F8_ocf_improve_strength, F9_profit_cash_gap_improvement,
               F10_ar_pressure_improvement, F11_inventory_pressure_improvement, F12_contract_liab_improvement,
               F13_capex_efficiency_improvement, F14_asset_turnover_improvement,
               F1_score, F2_score, F3_score, F4_score, F5_score, F6_score, F7_score, F8_score,
               F9_score, F10_score, F11_score, F12_score, F13_score, F14_score
        FROM financial_scores
        WHERE ts_code = :ts_code AND end_date = :report_date
    """
    try:
        df = query_sql(session, sql, {"ts_code": ts_code, "report_date": report_date})
        return df
    except Exception:
        return pd.DataFrame()


def load_all_scores(session, report_date: Optional[str] = None) -> pd.DataFrame:
    """加载全股池评分数据（从 financial_scores 表）"""
    try:
        if report_date:
            sql = """
                SELECT ts_code, end_date as report_date, ann_date, total_score,
                       main_profit_score, revenue_expense_efficiency_score, cashflow_validation_score,
                       working_capital_quality_score, investment_asset_efficiency_score,
                       cashflow_red_flag, ar_pressure_worsen_flag, inventory_pressure_worsen_flag, core_profit_worsen_flag
                FROM financial_scores
                WHERE end_date = :report_date
                ORDER BY CASE WHEN total_score IS NULL THEN 1 ELSE 0 END, total_score DESC
            """
            df = query_sql(session, sql, {"report_date": report_date})
        else:
            sql = """
                SELECT ts_code, end_date as report_date, ann_date, total_score,
                       main_profit_score, revenue_expense_efficiency_score, cashflow_validation_score,
                       working_capital_quality_score, investment_asset_efficiency_score,
                       cashflow_red_flag, ar_pressure_worsen_flag, inventory_pressure_worsen_flag, core_profit_worsen_flag
                FROM financial_scores
                ORDER BY ts_code, end_date DESC
            """
            df = query_sql(session, sql, {})
            df = df.groupby("ts_code").first().reset_index()
            df = df.sort_values("total_score", ascending=False, na_position="last").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def load_financial_score_overview(session) -> pd.DataFrame:
    """加载财务评分总览数据（关联 stock_pools 获取股票名称和概念）
    
    显示股票池中所有股票，每只股票取最近一期的评分数据
    """
    try:
        sql = """
            SELECT 
                f.ts_code,
                COALESCE(p.name, '') as stock_name,
                COALESCE(p.concepts, '') as concept,
                f.end_date as report_date,
                f.ann_date,
                f.total_score,
                f.main_profit_score,
                f.revenue_expense_efficiency_score,
                f.cashflow_validation_score,
                f.working_capital_quality_score,
                f.investment_asset_efficiency_score
            FROM (
                SELECT DISTINCT ON (ts_code) 
                    ts_code,
                    end_date,
                    ann_date,
                    total_score,
                    main_profit_score,
                    revenue_expense_efficiency_score,
                    cashflow_validation_score,
                    working_capital_quality_score,
                    investment_asset_efficiency_score
                FROM financial_scores
                ORDER BY ts_code, end_date DESC
            ) f
            INNER JOIN stock_pools p ON f.ts_code = p.ts_code
            ORDER BY CASE WHEN f.total_score IS NULL THEN 1 ELSE 0 END, f.total_score DESC
        """
        df = query_sql(session, sql, {})
        return df
    except Exception as e:
        print(f"加载财务评分总览数据失败: {e}")
        return pd.DataFrame()


def get_available_report_dates(session) -> List[str]:
    """获取所有可用的报告期（从 financial_scores 表）"""
    sql = "SELECT DISTINCT end_date as report_date FROM financial_scores ORDER BY end_date DESC"
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
    """对数值列应用颜色渐变渲染（基于每列的极值），始终返回 Styler 对象"""
    if df.empty or not numeric_cols:
        return df.style
    available_cols = [c for c in numeric_cols if c in df.columns]

    def format_with_backup(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        try:
            return f"{float(val):.2f}"
        except (ValueError, TypeError):
            return str(val)

    styler = df.style
    if available_cols:
        styler = styler.background_gradient(
            cmap="RdYlGn",
            subset=available_cols,
            low=0.3,
            high=1.0
        )
        for col in available_cols:
            fmt = custom_formatters.get(col, format_with_backup) if custom_formatters else format_with_backup
            styler = styler.format({col: fmt}, subset=[col])
    return styler


def colorize_numeric_columns_simple(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """对数值列应用颜色渐变渲染（每列单独取极值，简化版），始终返回 Styler 对象

    Args:
        df: DataFrame数据
        numeric_cols: 需要渲染的数值列列表

    Returns:
        带颜色样式的 Styler 对象
    """
    if df.empty or not numeric_cols:
        return df.style

    available_cols = [c for c in numeric_cols if c in df.columns]

    # 为每列单独计算颜色（基于该列自身的极值，而非全局）
    def get_color_for_value(val, col_min, col_max, col_name):
        """根据值在列中的位置返回颜色"""
        if val is None or pd.isna(val):
            return ''
        if col_max == col_min:
            return 'background-color: #e0e0e0; color: #333333'  # 灰色（所有值相同）

        # 归一化到 0-1
        ratio = (val - col_min) / (col_max - col_min)

        # 使用更深的颜色，避免白色/淡黄色背景看不清文字
        # 颜色映射：深红(低) -> 橙色(中低) -> 黄色(中) -> 浅绿(中高) -> 深绿(高)
        if ratio < 0.25:
            # 深红色 (低值)
            r, g, b = 220, 80, 80
            text_color = '#ffffff'  # 白色文字
        elif ratio < 0.5:
            # 橙色 (中低值)
            r, g, b = 255, 160, 80
            text_color = '#333333'  # 深色文字
        elif ratio < 0.75:
            # 黄色 (中值)
            r, g, b = 255, 220, 100
            text_color = '#333333'  # 深色文字
        else:
            # 深绿色 (高值)
            r, g, b = 80, 180, 80
            text_color = '#ffffff'  # 白色文字

        return f'background-color: rgb({r}, {g}, {b}); color: {text_color}; font-weight: 500'

    # 应用样式：每列单独计算 min/max，确保颜色仅基于该列自身分布
    styler = df.style
    for col in available_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            col_min, col_max = 0, 0
        else:
            col_min, col_max = col_data.min(), col_data.max()

        def make_style_func(cmin, cmax, cname):
            def style_func(val):
                return get_color_for_value(val, cmin, cmax, cname)
            return style_func

        styler = styler.map(make_style_func(col_min, col_max, col), subset=[col])

    # 格式化数值显示（处理None值）
    format_dict = {}
    for col in available_cols:
        if col == 'VWAP偏离%':
            format_dict[col] = lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "-"
        else:
            format_dict[col] = lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else "-"

    styler = styler.format(format_dict, subset=available_cols)
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
    "Stop-Loss选股结果": {
        "fields": ["ts_code", "stock_name", "total_market_cap",
                   "change_pct", "obs_day",
                   "pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls",
                   "sell_trigger_max_vol_price", "sell_stop_scale",
                   "dist_to_nearest_sell_stop_atr",
                   "last_event_volume", "last_event_bars_ago",
                   "vol_zscore", "bbmacd_event",
                   "daily_bb_width_zscore", "concepts"],
        "types": {"ts_code": "string", "stock_name": "string", "total_market_cap": "numeric",
                  "change_pct": "numeric", "obs_day": "numeric", "score": "numeric",
                  "pred_sell_reg": "numeric", "pred_sell_cls": "numeric",
                  "pred_buy_reg": "numeric", "pred_buy_cls": "numeric",
                  "sell_stop_triggered": "numeric",
                  "sell_trigger_volume": "numeric",
                  "active_sell_cluster_count": "numeric",
                  "sum_sells_active": "numeric", "sell_trigger_max_vol_price": "numeric", "sell_stop_scale": "numeric",
                  "nearest_sell_stop_price": "numeric",
                  "dist_to_nearest_sell_stop_atr": "numeric",
                  "last_event_type": "string", "last_event_volume": "numeric", "last_event_bars_ago": "numeric",
                  "vol_zscore": "numeric", "bbmacd_event": "string",
                  "daily_bb_width_zscore": "numeric", "concepts": "string"},
        "string_fields": ["ts_code", "stock_name", "last_event_type", "bbmacd_event", "concepts"],
        "display_names": {"ts_code": "股票代码", "stock_name": "股票名称", "total_market_cap": "当前市值(亿)",
                         "change_pct": "涨跌幅%", "obs_day": "观察天数", "score": "综合评分",
                         "pred_sell_reg": "Sell回归", "pred_sell_cls": "Sell分类",
                         "pred_buy_reg": "Buy回归", "pred_buy_cls": "Buy分类",
                         "sell_stop_triggered": "突破Sell",
                         "sell_trigger_volume": "Sell释放量",
                         "active_sell_cluster_count": "活跃Sell数",
                         "sum_sells_active": "Sell规模", "sell_trigger_max_vol_price": "突破价格", "sell_stop_scale": "Sell规模×价格",
                         "nearest_sell_stop_price": "最近Sell价",
                         "dist_to_nearest_sell_stop_atr": "Sell距离ATR",
                         "last_event_type": "上次事件", "last_event_volume": "上次量", "last_event_bars_ago": "上次距今天",
                         "vol_zscore": "成交量Z分", "bbmacd_event": "BBMACD事件",
                         "daily_bb_width_zscore": "日BB宽度Z分", "concepts": "概念"}
    },
    "自选股": {
        "fields": ["ts_code", "stock_name", "priority", "weighted_score", "core_driver",
                   "total_market_cap", "change_pct",
                   "pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls",
                   "second_deriv_type", "key_strength", "main_risk",
                   "concepts", "is_monitored"],
        "types": {"ts_code": "string", "stock_name": "string",
                  "priority": "string", "weighted_score": "numeric", "core_driver": "string",
                  "total_market_cap": "numeric", "change_pct": "numeric",
                  "pred_sell_reg": "numeric", "pred_sell_cls": "numeric",
                  "pred_buy_reg": "numeric", "pred_buy_cls": "numeric",
                  "second_deriv_type": "string", "key_strength": "string", "main_risk": "string",
                  "concepts": "string", "is_monitored": "string"},
        "string_fields": ["ts_code", "stock_name", "priority", "core_driver",
                          "second_deriv_type", "key_strength", "main_risk",
                          "concepts", "is_monitored"],
        "display_names": {"ts_code": "股票代码", "stock_name": "股票名称",
                         "priority": "跟踪优先级", "weighted_score": "加权总分", "core_driver": "核心驱动",
                         "total_market_cap": "当前市值(亿)", "change_pct": "涨跌幅%",
                         "pred_sell_reg": "Sell回归", "pred_sell_cls": "Sell分类",
                         "pred_buy_reg": "Buy回归", "pred_buy_cls": "Buy分类",
                         "second_deriv_type": "二阶导类型", "key_strength": "主要强点", "main_risk": "主要扣分/风险",
                         "concepts": "概念", "is_monitored": "监控"}
    }
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

        # 确定实际字段名（用于DataFrame列访问）
        actual_field = field
        if field not in result.columns:
            if field in field_to_display:
                actual_field = field_to_display[field]
            elif field in display_to_field:
                actual_field = display_to_field[field]

        if actual_field not in result.columns:
            continue

        # 确定字段类型：优先使用原始字段名查询types，如果找不到则尝试显示名称
        field_type = config["types"].get(field)
        if field_type is None and actual_field != field:
            field_type = config["types"].get(actual_field)

        if field_type == "string":
            if operator == "等于" and value:
                result = result[result[actual_field].astype(str) == str(value)]
            elif operator == "包含" and value:
                result = result[result[actual_field].astype(str).str.contains(str(value), case=False, na=False)]
            elif operator == "不包含" and value:
                result = result[~result[actual_field].astype(str).str.contains(str(value), case=False, na=False)]
        elif field_type == "enum":
            # 枚举类型：将显示值映射回实际值
            enum_mapping = {"向上": 1, "向下": -1, "走平": 0}
            actual_value = enum_mapping.get(value)
            if actual_value is not None:
                result = result[result[actual_field] == actual_value]
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


def get_available_stop_loss_dates(session) -> List[str]:
    """获取所有可用的Stop-Loss选股日期"""
    sql = """
        SELECT DISTINCT prediction_date as date FROM stop_loss_predictions
        WHERE profile = 'production'
        ORDER BY prediction_date DESC
    """
    try:
        df = query_sql(session, sql, {})
        return df["date"].tolist() if not df.empty else []
    except Exception:
        return []


def load_stop_loss_results(session, selection_date: str = None) -> pd.DataFrame:
    """
    加载Stop-Loss Clustering选股结果数据

    Args:
        session: 数据库会话
        selection_date: 选股日期，为None时取最新日期

    Returns:
        DataFrame包含Stop-Loss选股结果
    """
    if selection_date is None:
        dates = get_available_stop_loss_dates(session)
        if not dates:
            return pd.DataFrame()
        selection_date = dates[0]

    sql = """
        SELECT
            pred.ts_code,
            s.selection_date,
            s.signal_date,
            s.stock_name,
            p.concepts,
            p.total_market_cap / 100000000.0 AS total_market_cap,
            s.sell_stop_triggered,
            s.sell_trigger_volume,
            s.active_sell_cluster_count,
            s.sum_sells_active,
            s.sell_trigger_max_vol_price,
            s.sell_stop_scale,
            s.nearest_sell_stop_price,
            s.dist_to_nearest_sell_stop_atr,
            s.last_event_type,
            s.last_event_volume,
            s.last_event_bars_ago,
            s.vol_zscore,
            s.bbmacd_event,
            s.daily_bb_width_zscore,
            pred.pred_sell_reg,
            pred.pred_sell_cls,
            pred.pred_buy_reg,
            pred.pred_buy_cls,
            pred.score,
            pred.obs_date,
            pred.obs_day,
            pred.signal_id,
            s.created_at
        FROM stop_loss_predictions pred
        LEFT JOIN stop_loss_selection s ON pred.signal_id = s.id
        LEFT JOIN stock_pools p ON pred.ts_code = p.ts_code
        WHERE pred.prediction_date = :prediction_date
            AND pred.profile = 'production'
        ORDER BY pred.pred_sell_reg DESC NULLS LAST
    """
    try:
        df = query_sql(session, sql, {"prediction_date": selection_date})
        if not df.empty and "ts_code" in df.columns:
            change_map = compute_change_pct_for_date(session, selection_date, df["ts_code"].tolist())
            df["change_pct"] = df["ts_code"].map(change_map).fillna(0.0)
        return df
    except Exception as e:
        st.error(f"加载Stop-Loss选股结果失败: {e}")
        return pd.DataFrame()


def compute_change_pct_for_date(session, prediction_date: str, ts_codes: list) -> dict:
    """
    批量计算指定日期所有股票的涨跌幅

    Args:
        session: 数据库会话
        prediction_date: 预测日期 YYYY-MM-DD
        ts_codes: 股票代码列表

    Returns:
        {ts_code: change_pct} 字典
    """
    if not ts_codes:
        return {}
    codes_str = ",".join([f"'{c}'" for c in ts_codes])
    sql = f"""
        SELECT ts_code, bar_time, close
        FROM stock_k_data
        WHERE freq = 'd'
          AND ts_code IN ({codes_str})
          AND bar_time <= CAST(:prediction_date AS DATE)
          AND bar_time >= (CAST(:prediction_date AS DATE) - INTERVAL '10 days')
        ORDER BY ts_code, bar_time
    """
    try:
        kline = query_sql(session, sql, {"prediction_date": prediction_date})
        if kline.empty:
            return {}
        kline = kline.sort_values(["ts_code", "bar_time"])
        kline["prev_close"] = kline.groupby("ts_code")["close"].shift(1)
        latest = kline.dropna(subset=["prev_close"]).groupby("ts_code").last().reset_index()
        latest["change_pct"] = np.where(
            (latest["prev_close"] > 0) & (latest["close"] > 0),
            np.round((latest["close"] - latest["prev_close"]) / latest["prev_close"] * 100, 2),
            0.0,
        )
        return dict(zip(latest["ts_code"], latest["change_pct"]))
    except Exception:
        return {}


def load_watchlist_data(session) -> pd.DataFrame:
    """加载自选股数据，关联 stock_pools 和 stop_loss_predictions

    Returns:
        DataFrame: 自选股列表（含市值、涨跌幅、4模型评分等）
    """
    sql = """
        SELECT
            w.ts_code,
            w.stock_name,
            w.added_date,
            w.sort_order,
            w.priority,
            w.weighted_score,
            w.core_driver,
            w.second_deriv_type,
            w.key_strength,
            w.main_risk,
            w.is_monitored,
            p.total_market_cap / 100000000.0 AS total_market_cap,
            p.concepts,
            pred.pred_sell_reg,
            pred.pred_sell_cls,
            pred.pred_buy_reg,
            pred.pred_buy_cls
        FROM stock_watchlist w
        LEFT JOIN stock_pools p ON p.ts_code = w.ts_code
            OR p.ts_code = w.ts_code || '.SZ'
            OR p.ts_code = w.ts_code || '.SH'
            OR p.ts_code = w.ts_code || '.BJ'
        LEFT JOIN LATERAL (
            SELECT pred_sell_reg, pred_sell_cls, pred_buy_reg, pred_buy_cls
            FROM stop_loss_predictions
            WHERE (ts_code = w.ts_code
                OR ts_code = w.ts_code || '.SZ'
                OR ts_code = w.ts_code || '.SH'
                OR ts_code = w.ts_code || '.BJ')
              AND profile IN ('production', 'position')
              AND prediction_date = (
                  SELECT MAX(prediction_date) FROM stop_loss_predictions
                  WHERE profile IN ('production', 'position')
              )
            ORDER BY obs_date DESC
            LIMIT 1
        ) pred ON TRUE
        ORDER BY w.sort_order, w.added_date DESC
    """
    try:
        df = query_sql(session, sql, {})
        if not df.empty and "ts_code" in df.columns:
            full_codes = [_format_ts_code(c) for c in df["ts_code"].tolist()]
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            change_map = compute_change_pct_for_date(session, today_str, full_codes)
            df["change_pct"] = df["ts_code"].map(
                {c: change_map.get(_format_ts_code(c), 0.0) for c in df["ts_code"].tolist()}
            ).fillna(0.0)
        if not df.empty and "is_monitored" in df.columns:
            df["is_monitored"] = df["is_monitored"].map({True: "是", False: "否", None: "否"}).fillna("否")
        return df
    except Exception as e:
        st.error(f"加载自选股数据失败: {e}")
        return pd.DataFrame()


def add_to_watchlist(session, ts_code: str, stock_name: str):
    """添加股票到自选股列表（幂等，已存在则跳过）"""
    from sqlalchemy import text
    sql = text("""
        INSERT INTO stock_watchlist (ts_code, stock_name, is_monitored)
        VALUES (:ts_code, :stock_name, TRUE)
        ON CONFLICT (ts_code) DO NOTHING
    """)
    session.execute(sql, {"ts_code": ts_code, "stock_name": stock_name})
    session.commit()


def remove_from_watchlist(session, ts_code: str):
    """从自选股列表移除股票"""
    from sqlalchemy import text
    sql = text("DELETE FROM stock_watchlist WHERE ts_code = :ts_code")
    session.execute(sql, {"ts_code": ts_code})
    session.commit()


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



def pine_ema(series: pd.Series, length: int) -> pd.Series:
    """Pine Script风格的EMA计算"""
    alpha = 2 / (length + 1)
    return series.ewm(span=length, adjust=False).mean()


def cross_over(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """判断series1上穿series2"""
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def cross_under(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """判断series1下穿series2"""
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def get_kline_data_from_db(ts_code: str, freq: str = 'd', bars: int = 60) -> pd.DataFrame:
    """从数据库获取K线数据，返回包含 bar_time 列的 DataFrame。"""
    try:
        df = load_k_data(ts_code, freq=freq)
        if df.empty:
            return pd.DataFrame()
        df = df.tail(bars).copy()
        # load_k_data 单只股票将 bar_time 设为 index，需恢复为列
        if 'bar_time' not in df.columns:
            df = df.reset_index()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                col_lower = col.lower()
                if col_lower in df.columns:
                    df[col] = df[col_lower]
        return df
    except Exception as e:
        print(f"获取K线数据失败 {ts_code} {freq}: {e}")
        return pd.DataFrame()


def compute_bbmacd_for_chart(df: pd.DataFrame, rapida: int = 8, lenta: int = 26,
                              signal_length: int = 9) -> pd.DataFrame:
    """计算BBMACD指标用于图表展示"""
    if df.empty or len(df) < lenta:
        return df

    close = df['close']

    ema_rapida = pine_ema(close, rapida)
    ema_lenta = pine_ema(close, lenta)
    macd_line = ema_rapida - ema_lenta
    signal_line = pine_ema(macd_line, signal_length)
    histogram = macd_line - signal_line

    df = df.copy()
    df['macd_line'] = macd_line
    df['signal_line'] = signal_line
    df['histogram'] = histogram
    df['ema_rapida'] = ema_rapida
    df['ema_lenta'] = ema_lenta

    return df


def calculate_bsm_indicators(df: pd.DataFrame, rapida: int = 8, lenta: int = 26,
                              signal_length: int = 9) -> pd.DataFrame:
    """计算BSM指标（EMA/BBMACD/信号线）"""
    if df.empty:
        return df

    df = compute_bbmacd_for_chart(df, rapida, lenta, signal_length)

    if 'macd_line' in df.columns and len(df) > 0:
        df['bsm_signal'] = 'neutral'
        df.loc[cross_over(df['macd_line'], df['signal_line']), 'bsm_signal'] = 'bullish'
        df.loc[cross_under(df['macd_line'], df['signal_line']), 'bsm_signal'] = 'bearish'

    return df


def build_kline_chart_with_bsm(df: pd.DataFrame, stock_name: str,
                                show_vwap: bool = True, show_bsm: bool = True,
                                show_pavp: bool = True, ts_code: str = ""):
    """构建K线图+BSM副图"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if df.empty:
        return go.Figure()

    df = df.copy()
    if 'macd_line' not in df.columns:
        df = calculate_bsm_indicators(df)

    has_bsm = 'macd_line' in df.columns and df['macd_line'].notna().any()

    rows = 2 if has_bsm else 1
    row_heights = [0.7, 0.3] if has_bsm else [1.0]
    subplot_titles = []

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=row_heights,
                        subplot_titles=subplot_titles)

    fig.add_trace(go.Candlestick(
        x=df['bar_time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='K线',
        increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
        increasing_fillcolor='#ef5350', decreasing_fillcolor='#26a69a',
    ), row=1, col=1)

    if 'ema_rapida' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['bar_time'], y=df['ema_rapida'],
            name=f'EMA8', line=dict(color='#FF9800', width=1),
        ), row=1, col=1)

    if 'ema_lenta' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['bar_time'], y=df['ema_lenta'],
            name=f'EMA26', line=dict(color='#2196F3', width=1),
        ), row=1, col=1)

    if has_bsm:
        colors = ['#ef5350' if v >= 0 else '#26a69a' for v in df['histogram'].fillna(0)]
        fig.add_trace(go.Bar(
            x=df['bar_time'], y=df['histogram'], name='Histogram',
            marker_color=colors, showlegend=False,
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df['bar_time'], y=df['macd_line'],
            name='MACD', line=dict(color='#FF9800', width=1.5),
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df['bar_time'], y=df['signal_line'],
            name='Signal', line=dict(color='#2196F3', width=1.5),
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{stock_name} K线图", font=dict(size=16)),
        height=600 if has_bsm else 400,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(type='category', row=1, col=1)
    if has_bsm:
        fig.update_xaxes(type='category', row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)

    return fig


SELL_CLS_THRESHOLD = 0.7
BUY_CLS_THRESHOLD = 0.5


def build_factor_tracking_chart(df_kline: pd.DataFrame, timeline_data: pd.DataFrame,
                                 stock_name: str, ts_code: str, entry_date=None) -> go.Figure:
    """构建双面板图：主图=K线+EMA+入场标记，副图=4因子追踪+阈值参考线

    Args:
        df_kline: 日线K线数据（含 bar_time/open/high/low/close）
        timeline_data: 评分时间线数据（含 date/pred_sell_reg/pred_sell_cls/pred_buy_reg/pred_buy_cls）
        stock_name: 股票名称
        ts_code: 股票代码
        entry_date: 入场日期（用于标记）

    Returns:
        Plotly Figure 对象
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if df_kline.empty and timeline_data.empty:
        return go.Figure()

    # 统一日期格式：转为字符串，保持 category 类型以避免非交易日断层
    df_kline = df_kline.copy()
    if 'bar_time' in df_kline.columns:
        df_kline['date'] = pd.to_datetime(df_kline['bar_time']).dt.strftime('%Y-%m-%d')
    else:
        df_kline['date'] = pd.to_datetime(df_kline.iloc[:, 0]).dt.strftime('%Y-%m-%d')

    timeline_data = timeline_data.copy()
    timeline_data['date'] = pd.to_datetime(timeline_data['date']).dt.strftime('%Y-%m-%d')

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.6, 0.4],
        subplot_titles=[f"{stock_name} ({ts_code}) 日线", "4因子追踪"],
    )

    # --- 主图：K线 ---
    if not df_kline.empty:
        fig.add_trace(go.Candlestick(
            x=df_kline['date'], open=df_kline['open'], high=df_kline['high'],
            low=df_kline['low'], close=df_kline['close'], name='K线',
            increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
            increasing_fillcolor='#ef5350', decreasing_fillcolor='#26a69a',
        ), row=1, col=1)

        if 'ema_rapida' in df_kline.columns:
            fig.add_trace(go.Scatter(
                x=df_kline['date'], y=df_kline['ema_rapida'],
                name='EMA8', line=dict(color='#FF9800', width=1),
            ), row=1, col=1)

        if 'ema_lenta' in df_kline.columns:
            fig.add_trace(go.Scatter(
                x=df_kline['date'], y=df_kline['ema_lenta'],
                name='EMA26', line=dict(color='#2196F3', width=1),
            ), row=1, col=1)

        # 入场标记
        if entry_date is not None and not pd.isna(entry_date):
            entry_dt = pd.to_datetime(entry_date)
            entry_bar = df_kline[df_kline['date'] == entry_dt]
            entry_price = entry_bar['close'].iloc[0] if not entry_bar.empty else None

            # 绿色竖线
            fig.add_vline(
                x=entry_dt, line_color='#4CAF50', line_width=2, line_dash='dash',
                row=1, col=1,
            )
            if entry_price is not None:
                fig.add_annotation(
                    x=entry_dt, y=entry_price,
                    text=f'入场 {entry_price:.2f}',
                    showarrow=False, font=dict(size=10, color='#4CAF50'),
                    bgcolor='white', bordercolor='#4CAF50', borderwidth=1,
                    xshift=-25, yshift=15, row=1, col=1,
                )

    # --- 副图：4因子追踪 ---
    factor_colors = [
        ('pred_sell_reg', '#FF9800', '卖出回归'),
        ('pred_sell_cls', '#E91E63', '卖出分类'),
        ('pred_buy_reg', '#2196F3', '买入回归'),
        ('pred_buy_cls', '#4CAF50', '买入分类'),
    ]
    for col_name, color, label in factor_colors:
        if col_name in timeline_data.columns:
            sub = timeline_data.dropna(subset=[col_name])
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub['date'], y=sub[col_name],
                    name=label, line=dict(color=color, width=1.5),
                    hovertemplate='%{x|%Y-%m-%d}<br>' + label + ': %{y:.4f}<extra></extra>',
                ), row=2, col=1)

    # 阈值参考线
    fig.add_hline(
        y=SELL_CLS_THRESHOLD, line_dash='dash', line_color='rgba(128,128,128,0.5)',
        row=2, col=1,
    )
    fig.add_annotation(
        x=0.02, y=SELL_CLS_THRESHOLD, xref='x domain', yref='y',
        text=f'卖出阈值={SELL_CLS_THRESHOLD}', showarrow=False,
        font=dict(size=9, color='gray'), row=2, col=1,
    )

    fig.add_hline(
        y=BUY_CLS_THRESHOLD, line_dash='dash', line_color='rgba(128,128,128,0.5)',
        row=2, col=1,
    )
    fig.add_annotation(
        x=0.02, y=BUY_CLS_THRESHOLD, xref='x domain', yref='y',
        text=f'买入阈值={BUY_CLS_THRESHOLD}', showarrow=False,
        font=dict(size=9, color='gray'), row=2, col=1,
    )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
    )

    fig.update_xaxes(type='category', row=1, col=1)
    fig.update_xaxes(type='category', row=2, col=1)
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='评分', row=2, col=1)

    return fig


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

    x_values = df.index.strftime('%Y-%m-%d').tolist()

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
                x_str = [pd.Timestamp(t).strftime('%Y-%m-%d') for t in seg["x"]]
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
                x_str = pd.Timestamp(lab["x"]).strftime('%Y-%m-%d')
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
    "weekly_reversal_buy": "#9c27b0",   # 紫色 - 周线一类
    "weekly_breakout_buy": "#ff9800",   # 橙色 - 周线二类
}

BUY_SIGNAL_NAMES = {
    "weekly_reversal_buy": "周线反转",
    "weekly_breakout_buy": "周线突破",
}



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

    x_values = df.index.strftime('%Y-%m-%d').tolist() if hasattr(df.index, 'strftime') else df.index.astype(str).tolist()

    # 根据周期确定要检查的买点类型
    if period == 'w':
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

        # 根据字段类型确定运算符
        if field_type == "string":
            operators = ["包含", "等于", "不包含"]
        elif field_type == "enum":
            operators = ["等于"]
        else:
            operators = ["大于", "小于", "大于等于", "小于等于", "区间"]

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
            elif field_type == "enum":
                # 枚举类型显示下拉选择
                enum_values = config.get("enum_fields", {}).get(selected_field, [])
                default_val = cond.get("value") if cond.get("value") in enum_values else (enum_values[0] if enum_values else "")
                value = st.selectbox("值", options=enum_values, index=enum_values.index(default_val) if default_val in enum_values else 0, key=f"cond_{tab_name}_{i}_val", label_visibility="collapsed")
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
    """获取每只股票的最新一期财务评分（用于join，从 financial_scores 表）"""
    sql = """
        SELECT ts_code, total_score, end_date as report_date,
               main_profit_score, revenue_expense_efficiency_score, cashflow_validation_score,
               working_capital_quality_score, investment_asset_efficiency_score,
               cashflow_red_flag, ar_pressure_worsen_flag, inventory_pressure_worsen_flag, core_profit_worsen_flag
        FROM financial_scores
        WHERE (ts_code, end_date) IN (
            SELECT ts_code, MAX(end_date)
            FROM financial_scores
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


def render_risk_flags(row: pd.Series):
    """渲染风险警示区域"""
    active_flags = []

    for flag_col, flag_config in RISK_FLAG_CONFIG.items():
        if row.get(flag_col, False):
            active_flags.append(flag_config)

    if not active_flags:
        return

    st.markdown("#### ⚠️ 风险警示")
    cols = st.columns(len(active_flags))
    for i, flag in enumerate(active_flags):
        with cols[i]:
            st.markdown(
                f"<div style='background-color: {flag['color']}22; border-left: 4px solid {flag['color']}; "
                f"padding: 10px; border-radius: 4px;'>"
                f"<span style='color: {flag['color']}; font-weight: bold;'>⚠️ {flag['name']}</span><br>"
                f"<span style='color: #888; font-size: 12px;'>{flag['desc']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    st.markdown("---")


def render_metric_detail_table(row: pd.Series):
    """渲染14项核心指标明细表格"""
    # 按维度分组展示
    for dim_col, dim_config in DIMENSION_CONFIG.items():
        dim_name = dim_config["name"]
        dim_weight = dim_config["weight"]
        dim_color = dim_config["color"]

        # 获取该维度下的指标
        dim_metrics = {k: v for k, v in METRIC_CONFIG.items() if v["dimension"] == dim_col}

        if not dim_metrics:
            continue

        st.markdown(f"**{dim_name}** (权重{dim_weight}分)")

        # 构建指标数据
        metric_data = []
        for metric_col, metric_config in dim_metrics.items():
            metric_name = metric_config["name"]
            metric_weight = metric_config["weight"]

            # 原始值
            raw_value = row.get(metric_col)
            raw_display = f"{raw_value:.4f}" if pd.notna(raw_value) else "N/A"

            # 得分
            score_col = f"{metric_col.split('_')[0]}_score"  # F1_score, F2_score, etc.
            score_value = row.get(score_col)
            score_display = f"{score_value:.2f}" if pd.notna(score_value) else "N/A"

            # 得分率
            if pd.notna(score_value) and metric_weight > 0:
                score_ratio = score_value / metric_weight * 100
                score_bar = f"<div style='background: linear-gradient(90deg, {dim_color} {score_ratio:.0f}%, transparent {score_ratio:.0f}%); " \
                           f"height: 8px; border-radius: 4px; width: 100px;'></div>"
            else:
                score_bar = "<div style='color: #888;'>N/A</div>"

            metric_data.append({
                "指标": metric_name,
                "权重": f"{metric_weight}分",
                "原始值": raw_display,
                "得分": score_display,
                "得分率": score_bar,
            })

        if metric_data:
            metric_df = pd.DataFrame(metric_data)
            st.markdown(
                metric_df.to_html(escape=False, index=False, classes="metric-table"),
                unsafe_allow_html=True
            )
        st.markdown("")


def render_dimension_chart(df: pd.DataFrame, report_date: str) -> go.Figure:
    """渲染5维度雷达图"""
    if df.empty:
        return None

    latest = df[df["report_date"] == report_date]
    if latest.empty:
        latest = df.iloc[[-1]]
    latest = latest.iloc[0]

    dimensions = []
    scores = []
    colors = []
    max_weights = []

    for dim_col in DIMENSION_COLS:
        dim_config = DIMENSION_CONFIG.get(dim_col, {})
        dim_name = dim_config.get("name", dim_col)
        max_weight = dim_config.get("weight", 0)
        score = latest.get(dim_col)
        if pd.notna(score):
            dimensions.append(dim_name)
            scores.append(float(score))
            colors.append(dim_config.get("color", "#888888"))
            max_weights.append(max_weight)

    # 使用雷达图展示5维度评分
    fig = go.Figure()

    # 添加实际得分
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],  # 闭合
        theta=dimensions + [dimensions[0]],
        fill='toself',
        fillcolor='rgba(41, 98, 255, 0.2)',
        line=dict(color='#2962ff', width=2),
        name='实际得分',
        hovertemplate="<b>%{theta}</b><br>得分: %{r:.1f}<extra></extra>",
    ))

    # 添加满分参考线
    fig.add_trace(go.Scatterpolar(
        r=max_weights + [max_weights[0]],
        theta=dimensions + [dimensions[0]],
        fill='none',
        line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'),
        name='维度满分',
        hovertemplate="<b>%{theta}</b><br>满分: %{r:.1f}<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max_weights) * 1.1],
                gridcolor='rgba(255,255,255,0.1)',
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color=DARK_THEME["text_color"]),
        ),
        plot_bgcolor=DARK_THEME["bg_color"],
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        font=dict(color=DARK_THEME["text_color"]),
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
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
        dim_config = DIMENSION_CONFIG.get(dim_col, {})
        dim_name = dim_config.get("name", dim_col)
        color = dim_config.get("color", "#888888")
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


def init_selected_picks_table(session):
    """确保 stock_selected_picks 表存在"""
    from sqlalchemy import text
    sql = text("""
        CREATE TABLE IF NOT EXISTS stock_selected_picks (
            id SERIAL PRIMARY KEY,
            pick_date DATE NOT NULL,
            rank INTEGER NOT NULL,
            ts_code VARCHAR(20) NOT NULL,
            stock_name VARCHAR(50),
            recommend_score DECIMAL(5,2),
            recommend_grade VARCHAR(50),
            action_advice VARCHAR(100),
            theme_tags VARCHAR(200),
            valuation_judge VARCHAR(100),
            positioning VARCHAR(50),
            sustainability VARCHAR(20),
            profit_trend VARCHAR(50),
            upside_pct DECIMAL(6,2),
            weekly_z DECIMAL(8,4),
            weekly_vol_z DECIMAL(8,4),
            daily_vol_z DECIMAL(8,4),
            vwap_deviation DECIMAL(8,4),
            sort_reason TEXT,
            sectors VARCHAR(200),
            logic TEXT,
            risk_factors TEXT,
            score INTEGER,
            theme VARCHAR(100),
            position VARCHAR(50),
            report_period VARCHAR(50),
            detail TEXT,
            signal_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    session.execute(sql)
    session.commit()
    
    sql = text("CREATE INDEX IF NOT EXISTS idx_selected_picks_date ON stock_selected_picks(pick_date)")
    session.execute(sql)
    sql = text("CREATE INDEX IF NOT EXISTS idx_selected_picks_date_rank ON stock_selected_picks(pick_date, rank)")
    session.execute(sql)
    session.commit()







def load_selected_picks(session, pick_date):
    """加载指定日期的精选股票池数据
    Args:
        session: 数据库会话
        pick_date: 选股日期（date或str）
    Returns:
        DataFrame: 按 rank 升序排序的精选股票池数据（含逐笔成交额统计）
    """
    sql = f"""
        SELECT rank, stock_name, ts_code, recommend_score, recommend_grade, action_advice,
               theme_tags, valuation_judge, positioning, sustainability, profit_trend,
               upside_pct, weekly_z, weekly_vol_z, daily_vol_z, vwap_deviation,
               sort_reason, sectors, logic, risk_factors, score, signal_date,
               avg_tick_amount, max_tick_amount, min_tick_amount
        FROM stock_selected_picks
        WHERE pick_date = '{pick_date}'
        ORDER BY rank ASC
    """
    return query_sql(session, sql, {})


def _format_ts_code(ts_code: str) -> str:
    """添加交易所后缀"""
    if "." not in ts_code:
        if ts_code.startswith("6") or ts_code.startswith("9"):
            return ts_code + ".SH"
        elif ts_code.startswith("8") or ts_code.startswith("4"):
            return ts_code + ".BJ"
        else:
            return ts_code + ".SZ"
    return ts_code





def render_stop_loss_selection_page(session, sidebar):
    """渲染Stop-Loss Clustering选股结果页面

    根据日期获取 selection_stop.py 的选股结果，
    只展示突破 sell-stop cluster 的股票，
    支持搜索、单选展开详情、添加到自选股。
    """
    st.header("🛑 Stop-Loss Clustering 选股结果（突破Sell-Stop）")

    # 获取可用选股日期
    available_dates = get_available_stop_loss_dates(session)
    if not available_dates:
        st.info("暂无Stop-Loss选股结果数据，请先运行 selection/selection_stop.py 生成数据")
        return

    # 日期选择器放在 sidebar
    with sidebar:
        st.markdown("---")
        st.subheader("⚙️ Stop-Loss选股配置")
        available_dates_dt = []
        for d in available_dates:
            if isinstance(d, str):
                available_dates_dt.append(datetime.datetime.strptime(d, "%Y-%m-%d").date())
            elif isinstance(d, datetime.datetime):
                available_dates_dt.append(d.date())
            elif isinstance(d, datetime.date):
                available_dates_dt.append(d)
        default_date = available_dates_dt[0] if available_dates_dt else datetime.date.today()
        selected_date_dt = st.date_input(
            "选股日期",
            value=default_date,
            min_value=min(available_dates_dt) if available_dates_dt else None,
            max_value=max(available_dates_dt) if available_dates_dt else None,
            key="stop_loss_selection_date"
        )
        selected_date = selected_date_dt.strftime("%Y-%m-%d")

    # 加载选股数据
    with st.spinner("加载Stop-Loss选股数据..."):
        df = load_stop_loss_results(session, selected_date)

    if df.empty:
        st.info(f"{selected_date} 暂无Stop-Loss选股结果数据")
        return

    # 统计概览
    total_count = len(df)
    sell_count = int(df["sell_stop_triggered"].sum()) if "sell_stop_triggered" in df.columns else 0

    st.markdown(
        f"**选股总数**: {total_count} 只 | "
        f"**突破Sell-Stop**: {sell_count} 只"
    )

    # 多条件筛选
    display_df = df.copy()
    conditions = render_filter_bar(display_df, "Stop-Loss选股结果")
    if conditions:
        display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["Stop-Loss选股结果"])

    loaded_conditions = render_filter_manager(session, "Stop-Loss选股结果", conditions)
    if loaded_conditions:
        display_df = df.copy()
        display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["Stop-Loss选股结果"])

    filtered_count = len(display_df)
    st.markdown(f"**共 {total_count} 只** | 筛选后: {filtered_count} 只")

    if display_df.empty:
        st.info("没有匹配的股票")
        return

    # 准备展示数据
    # display_df 已经在多条件筛选中准备好

    # 重命名列为中文（用于显示）
    config = TAB_FIELD_CONFIGS["Stop-Loss选股结果"]
    display_names = config["display_names"]
    rename_map = {k: v for k, v in display_names.items() if k in display_df.columns}
    display_df_renamed = display_df.rename(columns=rename_map)

    # 确定要展示的列（按配置顺序）
    available_cols = [display_names.get(f, f) for f in config["fields"] if display_names.get(f, f) in display_df_renamed.columns]
    display_df_renamed = display_df_renamed[available_cols]

    # 需要颜色渲染的数值列（中文列名）
    numeric_cols_display = [
        display_names.get("total_market_cap", "当前市值(亿)"),
        display_names.get("obs_day", "观察天数"),
        display_names.get("sell_trigger_max_vol_price", "突破价格"),
        display_names.get("sell_stop_scale", "Sell规模×价格"),
        display_names.get("dist_to_nearest_sell_stop_atr", "Sell距离ATR"),
        display_names.get("last_event_volume", "上次量"),
        display_names.get("last_event_bars_ago", "上次距今天"),
        display_names.get("change_pct", "涨跌幅%"),
        display_names.get("pred_sell_reg", "Sell回归"),
        display_names.get("pred_sell_cls", "Sell分类"),
        display_names.get("pred_buy_reg", "Buy回归"),
        display_names.get("pred_buy_cls", "Buy分类"),
        display_names.get("vol_zscore", "成交量Z分"),
        display_names.get("daily_bb_width_zscore", "日BB宽度Z分"),
    ]
    numeric_cols_display = [c for c in numeric_cols_display if c in display_df_renamed.columns]

    # 重置索引以确保与event.selection.rows匹配
    display_df_renamed_reset = display_df_renamed.reset_index(drop=True)

    # 数据表格（支持单选，带颜色渲染）
    event = st.dataframe(
        colorize_numeric_columns_simple(display_df_renamed_reset, numeric_cols_display),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=500,
    )

    # 处理选中行 - 展开详情
    if event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_row_display = display_df_renamed_reset.iloc[selected_idx]
        selected_row = display_df.iloc[selected_idx]
        ts_code = selected_row["ts_code"]
        stock_name = selected_row["stock_name"]

        st.markdown("---")
        st.markdown(f"### 📊 {stock_name} ({ts_code}) 详情")

        # 复用个股详情渲染函数
        ts_code_full = _format_ts_code(ts_code)
        _render_stock_detail(session, ts_code_full, stock_name, key_prefix=f"stop_{ts_code}_")





def _render_stock_detail(session, ts_code: str, stock_name: str, key_prefix: str = ""):
    """渲染个股详情（K线+财务评分+股东画像）
    
    此函数被个股分析页面和自选股页面复用，保持两处显示一致。
    当修改个股分析页面布局时，只需修改此函数即可。
    
    Args:
        session: 数据库会话
        ts_code: 带交易所后缀的股票代码（如 000001.SZ）
        stock_name: 股票名称
        key_prefix: 用于区分不同页面的 key 前缀，避免冲突
    """
    # K线图标签页切换（日线/周线）
    tab_daily, tab_weekly = st.tabs(["📈 日线", "📈 周线"])

    with tab_daily:
        with st.spinner("加载日线数据..."):
            daily_kline_df = get_kline_data_from_db(ts_code, 'd', bars=250)
            if not daily_kline_df.empty:
                daily_kline_df = calculate_bsm_indicators(daily_kline_df)
                daily_fig = build_kline_chart_with_bsm(
                    daily_kline_df, f"{stock_name} 日线",
                    show_vwap=True, show_bsm=True, show_pavp=True, ts_code=ts_code
                )
                st.plotly_chart(daily_fig, use_container_width=True, key=f"{key_prefix}daily_{ts_code}")
            else:
                st.warning("⚠️ 无日线数据")

    with tab_weekly:
        with st.spinner("加载周线数据..."):
            weekly_kline_df = get_kline_data_from_db(ts_code, 'w', bars=250)
            if not weekly_kline_df.empty:
                weekly_kline_df = calculate_bsm_indicators(weekly_kline_df)
                weekly_fig = build_kline_chart_with_bsm(
                    weekly_kline_df, f"{stock_name} 周线",
                    show_vwap=True, show_bsm=True, show_pavp=True, ts_code=ts_code
                )
                st.plotly_chart(weekly_fig, use_container_width=True, key=f"{key_prefix}weekly_{ts_code}")
            else:
                st.warning("⚠️ 无周线数据")

    st.markdown("---")

    # 加载财务评分数据
    with st.spinner("加载评分数据..."):
        score_df = load_score_data(session, ts_code)

    has_score_data = not score_df.empty

    if has_score_data:
        report_dates = score_df["report_date"].unique().tolist()
        latest_report = report_dates[-1] if report_dates else None
        latest_row = score_df[score_df["report_date"] == latest_report].iloc[0]

        # ========== 财务评分概览卡片 ==========
        st.markdown("#### 📊 财务评分概览")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            total_score = latest_row["total_score"]
            st.metric("总分", format_score(total_score), help="满分100分，基于14项指标综合计算")

        with col2:
            score = latest_row["main_profit_score"]
            weight = DIMENSION_CONFIG["main_profit_score"]["weight"]
            st.metric(f"主业盈利改善", format_score(score), help=f"权重{weight}分：核心利润改善强度、加速度、归母净利润改善")

        with col3:
            score = latest_row["revenue_expense_efficiency_score"]
            weight = DIMENSION_CONFIG["revenue_expense_efficiency_score"]["weight"]
            st.metric(f"收入费用效率", format_score(score), help=f"权重{weight}分：收入动量、毛利率扩张、费用率改善")

        with col4:
            score = latest_row["cashflow_validation_score"]
            weight = DIMENSION_CONFIG["cashflow_validation_score"]["weight"]
            st.metric(f"现金流验证", format_score(score), help=f"权重{weight}分：销售收现、经营现金流、利润现金背离")

        with col5:
            score = latest_row["working_capital_quality_score"]
            weight = DIMENSION_CONFIG["working_capital_quality_score"]["weight"]
            st.metric(f"营运资本质量", format_score(score), help=f"权重{weight}分：应收、存货、合同负债改善")

        with col6:
            score = latest_row["investment_asset_efficiency_score"]
            weight = DIMENSION_CONFIG["investment_asset_efficiency_score"]["weight"]
            st.metric(f"投入资产效率", format_score(score), help=f"权重{weight}分：资本开支效率、资产周转率改善")

        st.markdown("---")

        # ========== 风险警示区域 ==========
        render_risk_flags(latest_row)

        # ========== 评分可视化区域 ==========
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.markdown("#### 最新报告期评分")
            # 获取最新报告期的公告日期
            latest_ann_date = latest_row.get("ann_date")
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

        # ========== 14项核心指标明细 ==========
        with st.expander("📋 14项核心指标明细"):
            render_metric_detail_table(latest_row)

        st.markdown("---")

        # ========== 历史评分数据表 ==========
        with st.expander("📈 历史评分数据"):
            display_df = score_df[["report_date", "total_score"] + DIMENSION_COLS].copy()
            display_df["report_date"] = display_df["report_date"].apply(format_report_date)
            display_df.columns = ["报告期", "总分", "主业盈利改善", "收入费用效率", "现金流验证",
                                   "营运资本质量", "投入资产效率"]
            for col in display_df.columns[1:]:
                display_df[col] = display_df[col].apply(format_score)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ========== 股东质量画像 ==========
        st.markdown("#### 股东质量画像")
        render_stock_holder_quality_section(session, ts_code)

        st.markdown("---")
    else:
        st.warning(f"该股票暂无财务评分数据")
        st.markdown("---")

        # 股东质量画像（无财务评分数据时也显示）
        st.markdown("#### 股东质量画像")
        render_stock_holder_quality_section(session, ts_code)


def _get_score_color(value: float, min_val: float = 0, max_val: float = 100) -> str:
    """根据分数值返回对应的颜色（红到绿的渐变）"""
    if pd.isna(value):
        return "#808080"  # 灰色
    if max_val == min_val:
        return "#808080"
    
    # 归一化到 0-1
    ratio = (value - min_val) / (max_val - min_val)
    ratio = max(0, min(1, ratio))
    
    # 红色 (低分) -> 黄色 (中分) -> 绿色 (高分)
    if ratio < 0.5:
        # 红色到黄色
        r = 255
        g = int(255 * ratio * 2)
        b = 0
    else:
        # 黄色到绿色
        r = int(255 * (1 - ratio) * 2)
        g = 255
        b = 0
    
    return f"#{r:02x}{g:02x}{b:02x}"


def _load_latest_predictions(session, ts_codes: list) -> tuple[pd.DataFrame, dict]:
    """从 stop_loss_predictions 加载每只持仓股各自的最新4模型评分。

    对每只 ts_code 取各自最新的预测记录（DISTINCT ON + ORDER BY prediction_date DESC），
    确保即使不同股票预测日期不同，也能拿到各自的最新数据。

    Args:
        session: DB session
        ts_codes: 持仓股票代码列表（格式如 000001.SZ）

    Returns:
        (pred_df, stats) 其中 stats 为:
        - found: 查到预测数据的股票数
        - missing: 查不到预测数据的股票数
        - missing_codes: 查不到预测数据的股票代码列表
        - date_distribution: {date_str: count} 各预测日期的股票数量分布
    """
    if not ts_codes:
        return pd.DataFrame(), {"found": 0, "missing": 0, "missing_codes": [], "date_distribution": {}}

    placeholders = ", ".join([f":tc{i}" for i in range(len(ts_codes))])
    params = {f"tc{i}": code for i, code in enumerate(ts_codes)}

    # 对每只股票取各自最新的预测记录
    sql = f"""
        SELECT DISTINCT ON (ts_code)
            ts_code, pred_sell_reg, pred_sell_cls,
            pred_buy_reg, pred_buy_cls, score, obs_date, prediction_date
        FROM stop_loss_predictions
        WHERE profile IN ('production', 'position')
        AND ts_code IN ({placeholders})
        ORDER BY ts_code, prediction_date DESC, obs_date DESC
    """
    try:
        df = query_sql(session, sql, params)
    except Exception as e:
        st.warning(f"加载模型评分失败: {e}")
        return pd.DataFrame(), {"found": 0, "missing": len(ts_codes), "missing_codes": list(ts_codes), "date_distribution": {}}

    # 统计日期分布和缺失股票
    found_codes = set(df["ts_code"].unique()) if not df.empty else set()
    missing_codes = [c for c in ts_codes if c not in found_codes]

    date_dist = {}
    if not df.empty and "prediction_date" in df.columns:
        date_counts = df["prediction_date"].value_counts().sort_index(ascending=False)
        date_dist = {str(d): int(c) for d, c in date_counts.items()}

    stats = {
        "found": len(found_codes),
        "missing": len(missing_codes),
        "missing_codes": missing_codes,
        "date_distribution": date_dist,
    }

    return df, stats


def _fetch_qmt_positions():
    """从 QMT 获取实时持仓，返回 list[Position]。连接失败时降级到 mock。"""
    try:
        from qmt_trader.client import QmtClient
        from qmt_trader.session import SessionManager
        from qmt_trader.query import QueryAPI

        client = QmtClient()
        if not client.health():
            raise ConnectionError("QMT 服务不可达")

        mgr = SessionManager(client)
        sess = mgr.open()
        try:
            api = QueryAPI(client)
            positions = api.get_positions(sess.session_id)
            return positions, "live"
        finally:
            mgr.close(sess.session_id)
    except Exception as e:
        try:
            from qmt_trader.mock import MockQmtClient
            from qmt_trader.query import QueryAPI
            client = MockQmtClient()
            api = QueryAPI(client)
            positions = api.get_positions("mock_session")
            return positions, "mock"
        except Exception:
            return [], "error"


BACKTEST_LEDGER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stop_experiment", "output", "backtest_ledger",
)


@st.cache_data(ttl=300)
def _load_backtest_data() -> dict:
    """加载回测产物CSV，返回 {equity_curve, trade_report, daily_holdings, daily_audit}"""
    result = {}
    csv_map = {
        "equity_curve": "live_equity_curve.csv",
        "trade_report": "live_trade_report.csv",
        "daily_holdings": "daily_holdings_report.csv",
        "daily_audit": "daily_trade_audit.csv",
    }
    for key, fname in csv_map.items():
        fpath = os.path.join(BACKTEST_LEDGER_DIR, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                for date_col in ["date", "decision_date"]:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col])
                result[key] = df
            except Exception:
                result[key] = pd.DataFrame()
        else:
            result[key] = pd.DataFrame()
    return result


def _calc_backtest_metrics(eq_df: pd.DataFrame) -> dict:
    """从净值曲线计算核心回测指标（基于实际有持仓的交易日）"""
    if eq_df.empty:
        return {}
    nav = eq_df["nav_live"]
    final_nav = nav.iloc[-1]
    n_days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days
    ann_ret = (final_nav ** (365 / max(n_days, 1)) - 1) if n_days > 0 else 0
    mdd = eq_df["drawdown"].max()
    daily_ret = eq_df["daily_return"].dropna()
    daily_ret = daily_ret[daily_ret != 0]
    sharpe = 0.0
    if len(daily_ret) > 5 and daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * (252 ** 0.5)
    calmar = ann_ret / mdd if mdd > 0 else 0.0
    return {
        "final_nav": final_nav,
        "ann_ret": ann_ret,
        "mdd": mdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "trade_start": eq_df["date"].iloc[0],
        "trade_end": eq_df["date"].iloc[-1],
        "trade_days": n_days,
        "n_trading_rows": len(eq_df),
    }


def _render_backtest_tab1(eq_df: pd.DataFrame):
    """Tab1: 净值曲线与核心指标（仅展示有持仓的交易期）"""
    if eq_df.empty:
        st.info("暂无净值曲线数据")
        return

    total_rows = len(eq_df)
    active_df = eq_df[eq_df["n_positions"] > 0].copy() if "n_positions" in eq_df.columns else eq_df.copy()
    if active_df.empty:
        st.info("回测期间无持仓记录")
        return

    idle_rows = total_rows - len(active_df)
    if idle_rows > 0:
        idle_start = eq_df["date"].iloc[0].strftime("%Y-%m-%d")
        idle_end = active_df["date"].iloc[0].strftime("%Y-%m-%d")
        st.caption(f"📊 交易区间 {active_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {active_df['date'].iloc[-1].strftime('%Y-%m-%d')}，共 {len(active_df)} 个交易日"
                   f"（已过滤空仓期 {idle_rows} 天：{idle_start} ~ {idle_end}）")

    metrics = _calc_backtest_metrics(active_df)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("累计净值", f"{metrics['final_nav']:.4f}")
    with c2:
        st.metric("年化收益率", f"{metrics['ann_ret']:.2%}")
    with c3:
        st.metric("最大回撤", f"{metrics['mdd']:.2%}", delta=f"-{metrics['mdd']:.2%}", delta_color="inverse")
    with c4:
        st.metric("夏普比率", f"{metrics['sharpe']:.2f}")
    with c5:
        st.metric("Calmar比率", f"{metrics['calmar']:.2f}")

    st.markdown("---")

    nav_chart_data = active_df[["date", "nav_live", "drawdown"]].copy()
    nav_chart_data["drawdown_neg"] = -nav_chart_data["drawdown"]

    base = alt.Chart(nav_chart_data).encode(x=alt.X("date:T", title="日期"))

    nav_line = base.mark_line(color="#2196F3", strokeWidth=2).encode(
        y=alt.Y("nav_live:Q", title="净值", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", title="日期"), alt.Tooltip("nav_live:Q", title="净值", format=".4f")],
    )

    dd_area = base.mark_area(color="#FF5252", opacity=0.3).encode(
        y=alt.Y("drawdown_neg:Q", title="回撤"),
    )

    st.subheader("净值曲线 & 回撤")
    st.altair_chart((nav_line + dd_area).resolve_scale(y="independent").properties(height=400, width="container"), use_container_width=True)

    st.markdown("---")
    st.subheader("月度收益率")

    eq_monthly = active_df[["date", "nav_live"]].copy()
    eq_monthly["year"] = eq_monthly["date"].dt.year
    eq_monthly["month"] = eq_monthly["date"].dt.month
    monthly_nav = eq_monthly.groupby(["year", "month"])["nav_live"].last()
    monthly_ret = monthly_nav / monthly_nav.shift(1) - 1
    monthly_ret = monthly_ret.dropna().reset_index()
    monthly_ret.columns = ["year", "month", "ret"]

    if not monthly_ret.empty:
        heatmap = alt.Chart(monthly_ret).mark_rect().encode(
            x=alt.X("month:O", title="月份", scale=alt.Scale(domain=list(range(1, 13)))),
            y=alt.Y("year:O", title="年份"),
            color=alt.Color("ret:Q", title="月度收益率",
                            scale=alt.Scale(scheme="redyellowgreen", domainMid=0)),
            tooltip=[alt.Tooltip("year:O", title="年"), alt.Tooltip("month:O", title="月"),
                     alt.Tooltip("ret:Q", title="收益率", format=".2%")],
        ).properties(height=200, width="container")
        st.altair_chart(heatmap, use_container_width=True)


def _render_backtest_tab2(trade_df: pd.DataFrame):
    """Tab2: 交易明细"""
    if trade_df.empty:
        st.info("暂无交易数据")
        return

    c1, c2, c3, c4 = st.columns(4)
    n_total = len(trade_df)
    n_win = (trade_df["盈亏标签"] == "盈利").sum()
    n_loss = (trade_df["盈亏标签"] == "亏损").sum()
    win_rate = n_win / n_total if n_total > 0 else 0

    with c1:
        st.metric("总交易数", f"{n_total}")
    with c2:
        st.metric("胜率", f"{win_rate:.1%}", delta=f"{n_win}胜/{n_loss}亏")
    with c3:
        st.metric("盈利次数", f"{n_win}")
    with c4:
        st.metric("亏损次数", f"{n_loss}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("盈亏比", trade_df.attrs.get("wl_ratio", "-"))
    with c6:
        avg_hold = trade_df["持仓天数"].mean() if "持仓天数" in trade_df.columns else 0
        st.metric("平均持仓天数", f"{avg_hold:.1f}")
    with c7:
        max_win = trade_df["净盈亏%"].max() if "净盈亏%" in trade_df.columns else "-"
        st.metric("最大单笔盈利", f"{max_win}" if isinstance(max_win, str) else f"{max_win}")
    with c8:
        max_loss = trade_df["净盈亏%"].min() if "净盈亏%" in trade_df.columns else "-"
        st.metric("最大单笔亏损", f"{max_loss}" if isinstance(max_loss, str) else f"{max_loss}")

    st.markdown("---")

    reason_counts = trade_df["退出原因"].value_counts() if "退出原因" in trade_df.columns else pd.Series()
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.metric("模型风险退出", f"{reason_counts.get('模型风险退出', 0)}")
    with rc2:
        st.metric("止损退出", f"{reason_counts.get('止损退出', 0)}")
    with rc3:
        st.metric("最大持有到期", f"{reason_counts.get('最大持有到期', 0)}")

    st.markdown("---")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        label_filter = st.multiselect("盈亏标签", options=["盈利", "亏损", "持平"],
                                       default=["盈利", "亏损", "持平"], key="bt_label_filter")
    with filter_col2:
        reason_options = trade_df["退出原因"].unique().tolist() if "退出原因" in trade_df.columns else []
        reason_filter = st.multiselect("退出原因", options=reason_options,
                                        default=reason_options, key="bt_reason_filter")

    filtered = trade_df[
        trade_df["盈亏标签"].isin(label_filter) &
        trade_df["退出原因"].isin(reason_filter)
    ].copy()

    if filtered.empty:
        st.info("无匹配交易")
        return

    display_cols = ["序号", "股票代码", "股票名称", "买入日期", "卖出日期",
                    "买入价", "卖出价", "持仓天数", "毛盈亏%", "净盈亏%", "盈亏标签", "退出原因", "入场评分"]
    avail_cols = [c for c in display_cols if c in filtered.columns]
    df_display = filtered[avail_cols].reset_index(drop=True)

    event = st.dataframe(df_display, use_container_width=True, hide_index=True,
                          on_select="rerun", selection_mode="single-row")

    if event.selection and event.selection.rows:
        idx = event.selection.rows[0]
        row = filtered.iloc[idx] if idx < len(filtered) else None
        if row is not None:
            st.markdown("---")
            st.subheader(f"📊 {row.get('股票名称', '')} ({row.get('股票代码', '')}) 交易详情")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("买入价", f"¥{row.get('买入价', '-')}")
            with d2:
                st.metric("卖出价", f"¥{row.get('卖出价', '-')}")
            with d3:
                st.metric("净盈亏", row.get("净盈亏%", "-"))
            with d4:
                st.metric("退出原因", row.get("退出原因", "-"))


def _render_backtest_tab3(holdings_df: pd.DataFrame):
    """Tab3: 每日持仓"""
    if holdings_df.empty:
        st.info("暂无每日持仓数据")
        return

    available_dates = sorted(holdings_df["date"].unique())
    latest_date = available_dates[-1]

    selected_date = st.date_input("选择日期", value=latest_date, key="bt_holdings_date")
    selected_dt = pd.to_datetime(selected_date)

    day_data = holdings_df[holdings_df["date"] == selected_dt].copy()
    if day_data.empty:
        st.info(f"{selected_date} 无持仓数据")
        return

    st.caption(f"📅 {selected_date} 共 {len(day_data)} 只持仓")

    rename_map = {
        "ts_code": "股票代码", "stock_name": "名称", "entry_date": "入场日期",
        "entry_price": "入场价", "days_held": "持仓天数", "cur_ret": "当日收益",
        "score": "综合评分",
        "pred_sell_reg": "卖出回归", "pred_sell_cls": "卖出分类",
        "pred_buy_reg": "买入回归", "pred_buy_cls": "买入分类",
    }
    df_display = day_data.rename(columns=rename_map)
    display_cols = ["股票代码", "名称", "入场日期", "入场价", "持仓天数", "当日收益", "综合评分",
                    "卖出回归", "卖出分类", "买入回归", "买入分类"]
    avail_cols = [c for c in display_cols if c in df_display.columns]
    df_display = df_display[avail_cols].reset_index(drop=True)

    numeric_cols = ["卖出回归", "卖出分类", "买入回归", "买入分类", "当日收益", "综合评分"]
    numeric_cols = [c for c in numeric_cols if c in df_display.columns]
    styled = colorize_numeric_columns_simple(df_display, numeric_cols)

    event = st.dataframe(styled, use_container_width=True, hide_index=True,
                          on_select="rerun", selection_mode="single-row")

    if event.selection and event.selection.rows:
        idx = event.selection.rows[0]
        row = day_data.iloc[idx] if idx < len(day_data) else None
        if row is not None:
            ts_code = row.get("ts_code", "")
            stock_name = row.get("stock_name", "")
            st.markdown("---")
            st.subheader(f"📊 {stock_name} ({ts_code}) 评分时间线")

            stock_timeline = holdings_df[holdings_df["ts_code"] == ts_code].sort_values("date").copy()
            if stock_timeline.empty:
                st.info("无历史数据")
                return

            entry_date = stock_timeline["entry_date"].min() if "entry_date" in stock_timeline.columns else None

            df_kline = get_kline_data_from_db(ts_code, freq='d', bars=120)

            fig = build_factor_tracking_chart(df_kline, stock_timeline, stock_name, ts_code, entry_date)
            st.plotly_chart(fig, use_container_width=True)


def _render_backtest_tab4(audit_df: pd.DataFrame):
    """Tab4: 决策审计"""
    if audit_df.empty:
        st.info("暂无决策审计数据")
        return

    available_dates = sorted(audit_df["decision_date"].unique())
    latest_date = available_dates[-1]

    c1, c2 = st.columns([1, 2])
    with c1:
        selected_date = st.date_input("选择日期", value=latest_date, key="bt_audit_date")
    with c2:
        action_options = audit_df["action"].unique().tolist() if "action" in audit_df.columns else []
        selected_actions = st.multiselect("动作筛选", options=action_options,
                                           default=action_options, key="bt_audit_action")

    selected_dt = pd.to_datetime(selected_date)
    day_data = audit_df[
        (audit_df["decision_date"] == selected_dt) &
        (audit_df["action"].isin(selected_actions))
    ].copy()

    if day_data.empty:
        st.info(f"{selected_date} 无匹配决策")
        return

    st.caption(f"📅 {selected_date} 共 {len(day_data)} 条决策")

    rename_map = {
        "ts_code": "股票代码", "stock_name": "名称", "action": "动作",
        "reason": "原因", "score": "评分", "days_held": "持仓天数",
        "cur_ret": "当日收益", "rank": "排名",
        "pred_sell_reg": "卖出回归", "pred_sell_cls": "卖出分类",
        "pred_buy_reg": "买入回归", "pred_buy_cls": "买入分类",
    }
    df_display = day_data.rename(columns=rename_map)
    display_cols = ["股票代码", "名称", "动作", "评分", "持仓天数", "当日收益", "排名",
                    "卖出回归", "卖出分类", "买入回归", "买入分类"]
    avail_cols = [c for c in display_cols if c in df_display.columns]
    df_display = df_display[avail_cols].reset_index(drop=True)

    numeric_cols = ["卖出回归", "卖出分类", "买入回归", "买入分类", "评分", "当日收益"]
    numeric_cols = [c for c in numeric_cols if c in df_display.columns]
    styled = colorize_numeric_columns_simple(df_display, numeric_cols)

    event = st.dataframe(styled, use_container_width=True, hide_index=True,
                          on_select="rerun", selection_mode="single-row")

    if event.selection and event.selection.rows:
        idx = event.selection.rows[0]
        row = day_data.iloc[idx] if idx < len(day_data) else None
        if row is not None:
            ts_code = row.get("ts_code", "")
            stock_name = row.get("stock_name", "")
            action = row.get("action", "")
            why = row.get("why", "")
            st.markdown("---")
            st.subheader(f"🔍 {stock_name} ({ts_code}) - {action}")

            if why:
                st.markdown("**决策推理:**")
                st.text(why)

            scores = {
                "卖出回归": row.get("pred_sell_reg"),
                "卖出分类": row.get("pred_sell_cls"),
                "买入回归": row.get("pred_buy_reg"),
                "买入分类": row.get("pred_buy_cls"),
            }
            valid_scores = {k: v for k, v in scores.items() if pd.notna(v)}
            if valid_scores:
                score_df = pd.DataFrame([valid_scores])
                score_melt = score_df.melt(var_name="model", value_name="score")
                chart = alt.Chart(score_melt).mark_bar().encode(
                    x=alt.X("model:N", title="模型"),
                    y=alt.Y("score:Q", title="评分"),
                    color=alt.condition(
                        alt.datum.score > 0,
                        alt.value("#4CAF50"),
                        alt.value("#F44336"),
                    ),
                    tooltip=[alt.Tooltip("model:N", title="模型"), alt.Tooltip("score:Q", title="评分", format=".4f")],
                ).properties(height=250, width=400)
                st.altair_chart(chart, use_container_width=True)

            detail_cols = ["threshold_value", "planned_price", "planned_weight",
                           "signal_id", "is_held", "is_pending_buy", "obs_day"]
            details = {c: row.get(c) for c in detail_cols if c in row.index and pd.notna(row.get(c))}
            if details:
                st.markdown("**交易参数:**")
                st.json(details)


def render_watchlist_page(session, sidebar):
    """渲染自选股页面：展示自选股列表 + 添加/移除 + 筛选 + 个股详情展开"""
    st.header("⭐ 自选股")

    with st.expander("➕ 添加自选股", expanded=False):
        stock_list = load_stock_list(session)
        if stock_list.empty:
            st.info("暂无股票数据")
        else:
            pinyin_input = st.text_input(
                "输入股票名称首字母",
                value="",
                placeholder="例如: zgsy -> 中国石油",
                key="watchlist_add_pinyin",
            )
            matched = match_stocks_by_pinyin(pinyin_input, stock_list)
            if matched.empty:
                if pinyin_input:
                    st.info("未匹配到股票")
            else:
                selected_option = st.selectbox(
                    "选择股票",
                    options=range(len(matched)),
                    format_func=lambda i: f"{matched.iloc[i]['name']} ({matched.iloc[i]['ts_code']})",
                    index=0,
                    key="watchlist_add_select",
                )
                col_add, col_spacer = st.columns([1, 4])
                with col_add:
                    if st.button("添加", key="watchlist_add_btn"):
                        sel = matched.iloc[selected_option]
                        add_to_watchlist(session, sel["ts_code"], sel["name"])
                        st.success(f"已添加 {sel['name']} ({sel['ts_code']})")
                        st.rerun()

    with st.spinner("加载自选股数据..."):
        df = load_watchlist_data(session)

    if df.empty:
        st.info("暂无自选股，请通过上方添加")
        return

    total_count = len(df)
    monitored_count = int((df["is_monitored"] == "是").sum()) if "is_monitored" in df.columns else 0

    priority_order = ["S", "S-", "A", "A-", "A-/B+", "B+", "B"]
    priority_counts = {}
    if "priority" in df.columns:
        for p in priority_order:
            cnt = int((df["priority"] == p).sum())
            if cnt > 0:
                priority_counts[p] = cnt

    cols_spec = [1, 1] + [0.7] * len(priority_counts) + [0.5]
    cols = st.columns(cols_spec)
    with cols[0]:
        st.metric("自选股数", f"{total_count} 只")
    with cols[1]:
        st.metric("监控中", f"{monitored_count} 只")
    for i, (p, cnt) in enumerate(priority_counts.items()):
        with cols[2 + i]:
            st.metric(f"{p}级", f"{cnt} 只")
    with cols[-1]:
        if st.button("🔄", key="watchlist_refresh_btn"):
            st.rerun()

    st.markdown("---")

    display_df = df.copy()
    conditions = render_filter_bar(display_df, "自选股")
    if conditions:
        display_df = apply_filters(display_df, conditions, TAB_FIELD_CONFIGS["自选股"])

    loaded_conditions = render_filter_manager(session, "自选股", conditions)
    if loaded_conditions:
        display_df = df.copy()
        display_df = apply_filters(display_df, loaded_conditions, TAB_FIELD_CONFIGS["自选股"])

    filtered_count = len(display_df)
    st.markdown(f"**共 {total_count} 只** | 筛选后: {filtered_count} 只")

    if display_df.empty:
        st.info("没有匹配的自选股")
        return

    config = TAB_FIELD_CONFIGS["自选股"]
    display_names = config["display_names"]
    rename_map = {k: v for k, v in display_names.items() if k in display_df.columns}
    display_df_renamed = display_df.rename(columns=rename_map)

    available_cols = [display_names.get(f, f) for f in config["fields"] if display_names.get(f, f) in display_df_renamed.columns]
    display_df_renamed = display_df_renamed[available_cols]

    numeric_cols_display = [
        display_names.get("weighted_score", "加权总分"),
        display_names.get("total_market_cap", "当前市值(亿)"),
        display_names.get("change_pct", "涨跌幅%"),
        display_names.get("pred_sell_reg", "Sell回归"),
        display_names.get("pred_sell_cls", "Sell分类"),
        display_names.get("pred_buy_reg", "Buy回归"),
        display_names.get("pred_buy_cls", "Buy分类"),
    ]
    numeric_cols_display = [c for c in numeric_cols_display if c in display_df_renamed.columns]

    display_df_renamed_reset = display_df_renamed.reset_index(drop=True)

    col_remove, col_spacer = st.columns([1, 5])
    with col_remove:
        remove_code = st.text_input("移除股票代码", placeholder="如 000001.SZ", key="watchlist_remove_input")

    if remove_code:
        if st.button("移除", key="watchlist_remove_btn"):
            remove_from_watchlist(session, remove_code.strip())
            st.success(f"已移除 {remove_code.strip()}")
            st.rerun()

    event = st.dataframe(
        colorize_numeric_columns_simple(display_df_renamed_reset, numeric_cols_display),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=500,
    )

    if event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_row = display_df.iloc[selected_idx]
        ts_code = selected_row["ts_code"]
        stock_name = selected_row.get("stock_name", ts_code)

        st.markdown("---")
        st.markdown(f"### 📊 {stock_name} ({ts_code}) 详情")

        ts_code_full = _format_ts_code(ts_code)
        _render_stock_detail(session, ts_code_full, stock_name, key_prefix=f"wl_{ts_code}_")


def render_backtest_page():
    """渲染回测评估页面：净值曲线、交易明细、每日持仓、决策审计"""
    data = _load_backtest_data()

    eq_df = data["equity_curve"]
    trade_df = data["trade_report"]
    holdings_df = data["daily_holdings"]
    audit_df = data["daily_audit"]

    if all(df.empty for df in data.values()):
        st.warning("暂无回测数据，请先运行回测引擎生成产物")
        return

    n_trades = len(trade_df)
    n_holdings_dates = holdings_df["date"].nunique() if not holdings_df.empty and "date" in holdings_df.columns else 0
    n_audit_dates = audit_df["decision_date"].nunique() if not audit_df.empty and "decision_date" in audit_df.columns else 0
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("交易笔数", f"{n_trades}")
    with s2:
        st.metric("持仓天数", f"{n_holdings_dates}")
    with s3:
        st.metric("决策天数", f"{n_audit_dates}")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 净值曲线", "💰 交易明细", "📋 每日持仓", "🔍 决策审计"])

    with tab1:
        _render_backtest_tab1(eq_df)
    with tab2:
        _render_backtest_tab2(trade_df)
    with tab3:
        _render_backtest_tab3(holdings_df)
    with tab4:
        _render_backtest_tab4(audit_df)


def render_positions_page(session, sidebar):
    """渲染实盘持仓页面：展示 QMT 持仓列表 + 汇总 + 个股详情展开。"""
    st.header("📋 实盘持仓")

    positions, mode = _fetch_qmt_positions()

    if mode == "mock":
        st.warning("⚠️ QMT 连接失败，显示模拟数据")
    elif mode == "error":
        st.error("❌ 无法获取持仓数据（QMT 连接失败且 Mock 不可用）")
        return
    elif mode == "live":
        st.success(f"✅ 已连接 QMT 实盘")

    if not positions:
        st.info("当前无持仓")
        return

    rows = []
    for p in positions:
        rows.append({
            "股票代码": p.stock_code,
            "名称": p.instrument_name,
            "持仓量": p.volume,
            "可用量": p.can_use_volume,
            "均价": p.avg_price,
            "最新价": p.last_price,
            "市值": p.market_value,
            "盈亏率": p.profit_rate,
        })
    df = pd.DataFrame(rows)

    ts_codes = [p.stock_code for p in positions]
    pred_df, pred_stats = _load_latest_predictions(session, ts_codes)

    # 显示模型评分日期分布
    if pred_df.empty:
        st.warning("⚠️ 所有持仓股均未找到模型评分数据（盘后任务可能未执行或数据格式不匹配）")
    else:
        dist = pred_stats.get("date_distribution", {})
        if dist:
            latest_date = list(dist.keys())[0]
            latest_count = dist[latest_date]
            total_found = pred_stats["found"]
            if len(dist) == 1 and latest_count == total_found:
                st.caption(f"📊 模型评分日期: {latest_date}（全部 {total_found} 只）")
            else:
                dist_parts = [f"{d}: {c}只" for d, c in dist.items()]
                st.caption(f"📊 模型评分日期分布: {', '.join(dist_parts)}")

        if pred_stats["missing"] > 0:
            missing_str = ", ".join(pred_stats["missing_codes"])
            st.warning(f"⚠️ {pred_stats['missing']} 只持仓股未找到预测数据: {missing_str}")

    if not pred_df.empty:
        df = df.merge(pred_df, left_on="股票代码", right_on="ts_code", how="left")
        df = df.drop(columns=["ts_code"], errors="ignore")
    for col_name in ["pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls", "score"]:
        if col_name not in df.columns:
            df[col_name] = np.nan

    # 按市值降序排序，同步重排 positions 保持索引一致
    df["__orig_idx"] = range(len(df))
    df = df.sort_values("市值", ascending=False).reset_index(drop=True)
    positions = [positions[i] for i in df["__orig_idx"]]
    df = df.drop(columns=["__orig_idx"])

    total_mv = df["市值"].sum()
    total_cost = (df["市值"] / (1 + df["盈亏率"])).sum()
    total_pnl = total_mv - total_cost
    total_pnl_rate = total_pnl / total_cost if total_cost > 0 else 0

    col1, col2, col3, col_refresh = st.columns([1, 1, 1, 0.5])
    with col1:
        st.metric("总市值", f"¥{total_mv:,.0f}")
    with col2:
        st.metric("总盈亏", f"¥{total_pnl:,.0f}",
                   delta=f"{total_pnl_rate:.2%}",
                   delta_color="normal" if total_pnl >= 0 else "inverse")
    with col3:
        st.metric("持仓数", f"{len(positions)} 只")
    with col_refresh:
        if st.button("🔄 刷新", key="pos_refresh_btn"):
            st.rerun()

    st.markdown("---")

    df_display = df.copy()

    score_col_map = {
        "pred_sell_reg": "卖出回归",
        "pred_sell_cls": "卖出分类",
        "pred_buy_reg": "买入回归",
        "pred_buy_cls": "买入分类",
    }
    for src, dst in score_col_map.items():
        df_display[dst] = df_display[src]

    display_cols = ["股票代码", "名称", "持仓量", "可用量", "均价", "最新价", "市值", "盈亏率",
                    "卖出回归", "卖出分类", "买入回归", "买入分类"]
    df_display = df_display[display_cols]

    numeric_cols = ["卖出回归", "卖出分类", "买入回归", "买入分类", "盈亏率"]

    df_display_reset = df_display.reset_index(drop=True)
    styled = colorize_numeric_columns_simple(df_display_reset, numeric_cols)

    format_dict = {}
    for col in ["盈亏率", "卖出回归", "卖出分类", "买入回归", "买入分类"]:
        format_dict[col] = lambda x: f"{x:.2%}" if pd.notna(x) and x is not None else "-"
    for col in ["均价", "最新价"]:
        format_dict[col] = lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else "-"
    format_dict["市值"] = lambda x: f"¥{x:,.0f}" if pd.notna(x) and x is not None else "-"
    format_dict["持仓量"] = lambda x: f"{x:,}" if pd.notna(x) and x is not None else "-"
    format_dict["可用量"] = lambda x: f"{x:,}" if pd.notna(x) and x is not None else "-"
    styled = styled.format(format_dict)

    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    st.markdown("---")

    if event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        p = positions[selected_idx]
        ts_code = p.stock_code
        stock_name = p.instrument_name

        st.subheader(f"📊 {stock_name} ({ts_code}) 详情")

        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric("持仓量", f"{p.volume:,}")
        with col_info2:
            st.metric("可用量", f"{p.can_use_volume:,}")
        with col_info3:
            st.metric("均价", f"¥{p.avg_price:.2f}")
        with col_info4:
            st.metric("盈亏率", f"{p.profit_rate:.2%}",
                       delta=f"¥{p.market_value - p.avg_price * p.volume:,.0f}",
                       delta_color="normal" if p.profit_rate >= 0 else "inverse")

        ts_code_full = _format_ts_code(ts_code)
        _render_stock_detail(session, ts_code_full, stock_name, key_prefix=f"pos_{ts_code}_")


def render_financial_score_overview_page(session):
    """渲染财务评分总览页面
    
    以表格形式展示所有股票的最近一期财务评分，支持：
    - 颜色渲染（按数值从小到大渐变，红->黄->绿）
    - 筛选/搜索功能
    - 点击行跳转到个股详情
    """
    st.header("📊 财务评分总览")
    
    # 加载数据
    with st.spinner("加载财务评分数据..."):
        df = load_financial_score_overview(session)
    
    if df.empty:
        st.info("暂无财务评分数据")
        return
    
    # 统计数据分布
    scored_count = df["total_score"].notna().sum()
    unscored_count = df["total_score"].isna().sum()
    report_date_counts = df["report_date"].value_counts().sort_index(ascending=False)
    
    # 显示数据概览
    st.markdown(f"**股票池总数**: {len(df)} 只 | **有评分**: {scored_count} 只 | **无评分**: {unscored_count} 只")
    
    # 显示各报告期分布
    with st.expander("📅 查看各报告期数据分布"):
        report_dist = report_date_counts.head(10).reset_index()
        report_dist.columns = ["报告期", "股票数量"]
        st.dataframe(report_dist, use_container_width=True, hide_index=True)
    
    # 筛选区域
    st.markdown("---")
    col_search, col_score_min, col_score_max = st.columns([3, 1, 1])
    
    with col_search:
        search_term = st.text_input(
            "🔍 搜索",
            placeholder="输入股票代码、名称或概念",
            key="overview_search"
        )
    
    with col_score_min:
        min_score = st.number_input(
            "最低分",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0,
            key="overview_min_score"
        )
    
    with col_score_max:
        max_score = st.number_input(
            "最高分",
            min_value=0.0,
            max_value=100.0,
            value=100.0,
            step=5.0,
            key="overview_max_score"
        )
    
    # 应用筛选
    filtered_df = df.copy()
    
    if search_term:
        mask = (
            filtered_df["ts_code"].str.contains(search_term, case=False, na=False) |
            filtered_df["stock_name"].str.contains(search_term, case=False, na=False) |
            filtered_df["concept"].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    filtered_df = filtered_df[
        (filtered_df["total_score"] >= min_score) &
        (filtered_df["total_score"] <= max_score)
    ]
    
    st.markdown(f"**筛选结果**: {len(filtered_df)} 只股票")
    st.markdown("---")
    
    # 准备显示数据 - 调整列顺序和列名
    display_df = filtered_df[[
        "ts_code", "stock_name", "concept", "total_score",
        "main_profit_score", "revenue_expense_efficiency_score", "cashflow_validation_score",
        "working_capital_quality_score", "investment_asset_efficiency_score",
        "report_date", "ann_date"
    ]].copy()
    
    display_df.columns = [
        "股票代码", "股票名称", "概念", "总分",
        "主业盈利改善", "收入费用效率", "现金流验证",
        "营运资本质量", "投入资产效率",
        "报告期", "公告日"
    ]
    
    # 格式化日期
    display_df["报告期"] = display_df["报告期"].apply(format_report_date)
    display_df["公告日"] = display_df["公告日"].apply(lambda x: str(int(float(x))) if pd.notna(x) and str(x).lower() not in ['nan', 'nat', 'none'] else "")
    
    # 使用 st.dataframe 显示，支持点击行（不显示颜色，保持交互性）
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=600,
        column_config={
            "股票代码": st.column_config.TextColumn("股票代码", width="small"),
            "股票名称": st.column_config.TextColumn("股票名称", width="small"),
            "概念": st.column_config.TextColumn("概念", width="medium"),
            "总分": st.column_config.NumberColumn("总分", width="small", format="%.1f"),
            "主业盈利改善": st.column_config.NumberColumn("主业盈利改善", width="small", format="%.1f"),
            "收入费用效率": st.column_config.NumberColumn("收入费用效率", width="small", format="%.1f"),
            "现金流验证": st.column_config.NumberColumn("现金流验证", width="small", format="%.1f"),
            "营运资本质量": st.column_config.NumberColumn("营运资本质量", width="small", format="%.1f"),
            "投入资产效率": st.column_config.NumberColumn("投入资产效率", width="small", format="%.1f"),
            "报告期": st.column_config.TextColumn("报告期", width="small"),
            "公告日": st.column_config.TextColumn("公告日", width="small"),
        }
    )
    
    # 处理点击行 - 跳转到个股详情
    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_row = filtered_df.iloc[selected_idx]
        ts_code = selected_row["ts_code"]
        stock_name = selected_row["stock_name"]
        
        st.markdown("---")
        st.markdown(f"### 📈 {stock_name} ({ts_code}) 详情")
        
        # 复用个股详情渲染函数
        ts_code_full = _format_ts_code(ts_code)
        _render_stock_detail(session, ts_code_full, stock_name, key_prefix=f"overview_{ts_code}_")


def _render_stock_page(session, matched_stocks: pd.DataFrame, selected_option: int):
    """渲染个股分析页面"""
    if selected_option is None:
        return

    selected_stock = matched_stocks.iloc[selected_option]
    ts_code = selected_stock["ts_code"]
    stock_name = selected_stock["name"]

    # 股票标题
    st.markdown(f"### 当前股票: **{stock_name}** ({ts_code})")

    # 复用通用的详情渲染函数
    _render_stock_detail(session, ts_code, stock_name, key_prefix="analysis_")


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
        init_selected_picks_table(session)
        stock_list = load_stock_list(session)

        with st.sidebar:
            st.header("📊 财务评分分析")
            page = st.radio("页面", ["个股分析", "自选股", "实盘持仓", "Stop-Loss选股结果", "回测评估", "财务评分总览", "股东画像"], index=0)

        if page == "个股分析":
            if stock_list.empty:
                st.info("暂无股票数据（stock_pools 表不存在或为空）")
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
                        for dim_col, dim_config in DIMENSION_CONFIG.items():
                            dim_name = dim_config["name"]
                            weight = dim_config["weight"]
                            color = dim_config["color"]
                            st.markdown(f"<span style='color:{color}'>●</span> {dim_name}: {weight}分", unsafe_allow_html=True)

                if selected_option is not None:
                    _render_stock_page(session, matched_stocks, selected_option)
        elif page == "自选股":
            render_watchlist_page(session, st.sidebar)
        elif page == "实盘持仓":
            render_positions_page(session, st.sidebar)
        elif page == "Stop-Loss选股结果":
            render_stop_loss_selection_page(session, st.sidebar)
        elif page == "回测评估":
            render_backtest_page()
        elif page == "财务评分总览":
            render_financial_score_overview_page(session)
        elif page == "股东画像":
            render_holder_profile_page(session)

if __name__ == "__main__":
    main()
