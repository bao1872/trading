# -*- coding: utf-8 -*-
"""
Purpose:
    通用 UI 组件，供交易工作台各页面共享。
    状态灯、指标卡片、风险标签、分页表格、日期选择器。

Inputs:
    - Streamlit session_state

Outputs:
    - 渲染到 Streamlit 页面的 UI 组件

How to Run:
    python -m vis.common.components

Examples:
    from vis.common.components import render_status_light, render_date_selector
    date = render_date_selector()

Side Effects:
    - 渲染 Streamlit UI 组件
    - render_date_selector() 更新 st.session_state["selected_date"]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd

from vis.common.theme import STATUS_COLORS
from vis.common.data_loader import load_available_dates, load_latest_date


def render_status_light(label: str, status: str):
    """渲染状态灯

    Args:
        label: 状态名称
        status: "ok" / "warn" / "error" / "unknown"
    """
    color_map = {
        "ok": "#26a69a",
        "warn": "#ffc107",
        "error": "#ef5350",
        "unknown": "#9e9e9e",
    }
    color = color_map.get(status, "#9e9e9e")
    st.markdown(
        f'<span style="display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;background-color:{color};margin-right:6px;"></span>'
        f'{label}',
        unsafe_allow_html=True,
    )


def render_risk_tag(tag_type: str, text: str = None):
    """渲染风险标签

    Args:
        tag_type: sell/risk/hold/watch/buy/neutral
        text: 标签文本（默认取 tag_type 的中文映射）
    """
    label_map = {
        "sell": "建议卖出",
        "risk": "高风险",
        "near_max_hold": "临近最大持仓",
        "near_stop_loss": "接近止损",
        "strong_hold": "强势持有",
        "hold": "继续持有",
        "watch": "观察",
        "buy": "建议买入",
        "neutral": "中性",
    }
    display_text = text or label_map.get(tag_type, tag_type)
    color = STATUS_COLORS.get(tag_type, STATUS_COLORS["neutral"])
    st.markdown(
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'font-size:12px;font-weight:500;background-color:{color}22;'
        f'color:{color};border:1px solid {color}44;">{display_text}</span>',
        unsafe_allow_html=True,
    )


def render_date_selector(key: str = "date_selector") -> str:
    """渲染日期选择器，返回选中的日期

    Args:
        key: Streamlit widget key

    Returns:
        选中的日期字符串 (YYYY-MM-DD)
    """
    dates = load_available_dates()
    if not dates:
        st.warning("未找到任何交易日数据")
        return ""

    if "selected_date" not in st.session_state:
        st.session_state.selected_date = dates[0]

    selected = st.selectbox(
        "选择交易日",
        dates,
        index=dates.index(st.session_state.selected_date) if st.session_state.selected_date in dates else 0,
        key=key,
    )
    st.session_state.selected_date = selected
    return selected


def render_paginated_table(
    df: pd.DataFrame,
    page_size: int = 50,
    key: str = "paginated_table",
    height: int = 400,
):
    """渲染分页表格

    Args:
        df: 要展示的 DataFrame
        page_size: 每页行数
        key: Streamlit widget key
        height: 表格高度（像素）
    """
    if df.empty:
        st.info("无数据")
        return

    total_rows = len(df)
    total_pages = max(1, (total_rows - 1) // page_size + 1)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        page = st.number_input(
            "页码", min_value=1, max_value=total_pages,
            value=1, key=f"{key}_page",
        )
    with col2:
        st.markdown(f"<div style='text-align:center;padding-top:28px;'>第 {page}/{total_pages} 页，共 {total_rows} 条</div>", unsafe_allow_html=True)
    with col3:
        ps = st.selectbox("每页条数", [20, 50, 100, 200], index=1, key=f"{key}_ps")
        page_size = ps

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    page_df = df.iloc[start_idx:end_idx]

    st.dataframe(page_df, use_container_width=True, height=height)


def render_metric_cards(metrics: list):
    """渲染指标卡片组

    Args:
        metrics: list of dict, 每个含 label, value, delta(可选), color(可选)
    """
    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        with cols[i]:
            delta_str = m.get("delta", None)
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=delta_str,
                delta_color=m.get("delta_color", "normal"),
            )


def format_pct(val) -> str:
    """格式化百分比"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if isinstance(val, (int, float)):
        return f"{val * 100:.2f}%" if abs(val) < 1 else f"{val:.2f}%"
    return str(val)


def format_price(val) -> str:
    """格式化价格"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    return f"{val:.2f}"


def format_money(val) -> str:
    """格式化金额（带千分位）"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if isinstance(val, (int, float)):
        return f"¥{val:,.2f}"
    return str(val)


if __name__ == "__main__":
    print("=" * 60)
    print("  components 自测（非 Streamlit 环境，仅验证工具函数）")
    print("=" * 60)

    print(f"  format_pct(0.05): {format_pct(0.05)}")
    print(f"  format_pct(-0.071): {format_pct(-0.071)}")
    print(f"  format_pct(None): {format_pct(None)}")
    print(f"  format_price(12.5): {format_price(12.5)}")
    print(f"  format_price(None): {format_price(None)}")

    dates = load_available_dates()
    print(f"  可用交易日: {len(dates)}")
    if dates:
        print(f"  最新: {dates[0]}")

    print(f"\n  ✅ components 自测完成")
