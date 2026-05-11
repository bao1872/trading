# -*- coding: utf-8 -*-
"""
Purpose:
    半自动实盘交易决策工作台 — Streamlit Multipage App 主入口。
    设置页面配置、注入暗色主题、初始化 session_state。

Inputs:
    - 无（页面配置）

Outputs:
    - Streamlit 页面框架

How to Run:
    streamlit run vis/trading_workbench/app.py

Examples:
    streamlit run vis/trading_workbench/app.py --server.port 8502

Side Effects:
    - 注入全局暗色 CSS
    - 初始化 st.session_state
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from vis.common.theme import inject_dark_css
from vis.common.data_loader import load_latest_date


st.set_page_config(
    page_title="半自动实盘交易决策工作台",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_dark_css()

if "selected_date" not in st.session_state:
    st.session_state.selected_date = load_latest_date()

st.sidebar.title("🎯 交易工作台")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "半自动实盘交易决策工作台\n\n"
    "系统给建议，人工出最终计划\n\n"
    f"📅 当前日期: `{st.session_state.get('selected_date', '-')}`"
)
