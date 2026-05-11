# -*- coding: utf-8 -*-
"""
Purpose:
    暗色主题配置 SSOT，供 vis/ 下所有 Streamlit 应用共享。

Inputs:
    - 无

Outputs:
    - DARK_THEME: dict，Plotly 图表配色
    - STATUS_COLORS: dict，状态标签配色
    - inject_dark_css(): 函数，注入全局暗色 CSS

How to Run:
    无需单独运行，被其他模块 import

Examples:
    from vis.common.theme import DARK_THEME, inject_dark_css
    inject_dark_css()

Side Effects:
    - inject_dark_css() 会调用 st.markdown 注入全局 CSS（仅影响当前 Streamlit 页面）
"""

DARK_THEME = {
    "bg_color": "#131722",
    "paper_bgcolor": "#131722",
    "grid_color": "rgba(255,255,255,0.06)",
    "text_color": "#d1d4dc",
    "up_color": "#26a69a",
    "down_color": "#ef5350",
}

STATUS_COLORS = {
    "sell": "#ef5350",
    "risk": "#ff9800",
    "hold": "#26a69a",
    "watch": "#ffc107",
    "buy": "#2196f3",
    "neutral": "#9e9e9e",
}

_DARK_CSS = """
<style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .stSidebar {{
        background-color: #1e222d;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #1e222d;
    }}
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {{
        background-color: #1e222d;
        color: {text_color};
    }}
    .stDataFrame {{
        background-color: #1e222d;
    }}
    .stMetric {{
        background-color: #1e222d;
        border-radius: 8px;
        padding: 12px;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #1e222d;
        border-radius: 4px 4px 0 0;
        color: {text_color};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #2a2e39;
        color: #ffffff;
    }}
    .status-light {{
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }}
    .risk-tag {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 4px;
    }}
</style>
"""


def inject_dark_css():
    """注入暗色主题 CSS 到 Streamlit 页面"""
    import streamlit as st
    st.markdown(
        _DARK_CSS.format(
            bg_color=DARK_THEME["bg_color"],
            text_color=DARK_THEME["text_color"],
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    print("DARK_THEME:", DARK_THEME)
    print("STATUS_COLORS:", STATUS_COLORS)
    print("CSS snippet length:", len(_DARK_CSS))
