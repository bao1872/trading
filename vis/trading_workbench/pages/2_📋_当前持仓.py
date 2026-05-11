# -*- coding: utf-8 -*-
"""
Purpose:
    当前持仓页 — 快速识别需要卖出、继续持有、减仓观察的持仓

Inputs:
    - holdings/T.parquet
    - decisions/T.parquet

Outputs:
    - Streamlit 页面渲染

How to Run:
    通过 trading_workbench/app.py 自动加载

Side Effects:
    - 无（只读数据）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import streamlit as st
import pandas as pd

from vis.common.data_loader import (
    load_holdings_df, load_decisions_df, get_stock_name_map,
    compute_risk_tags,
)
from vis.common.components import (
    render_date_selector, render_paginated_table, format_pct, format_price,
)
from vis.common.theme import STATUS_COLORS


def render():
    st.title("📋 当前持仓")

    date = render_date_selector("holdings_date")
    if not date:
        st.error("未找到交易日数据")
        return

    holdings_df = load_holdings_df(date)
    decisions_df = load_decisions_df(date)

    if holdings_df.empty:
        st.warning(f"{date} 无持仓数据")
        return

    name_map = get_stock_name_map()
    tagged = compute_risk_tags(holdings_df, decisions_df)

    st.markdown("---")

    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)
    with filter_col1:
        show_all = st.button("全部", use_container_width=True, key="filter_all")
    with filter_col2:
        show_sell = st.button("🔴 建议卖出", use_container_width=True, key="filter_sell")
    with filter_col3:
        show_risk = st.button("🟠 高风险", use_container_width=True, key="filter_risk")
    with filter_col4:
        show_loss = st.button("🔵 浮亏", use_container_width=True, key="filter_loss")
    with filter_col5:
        show_strong = st.button("🟢 强势持有", use_container_width=True, key="filter_strong")

    if show_sell:
        filtered = tagged[tagged["risk_tags"].str.contains("sell", na=False)]
    elif show_risk:
        filtered = tagged[tagged["risk_tags"].str.contains("risk|near_max_hold", na=False)]
    elif show_loss:
        if "cur_ret" in tagged.columns:
            filtered = tagged[pd.to_numeric(tagged.get("cur_ret", 0), errors="coerce") < 0]
        else:
            filtered = tagged
    elif show_strong:
        filtered = tagged[tagged["risk_tags"].str.contains("strong_hold", na=False)]
    else:
        filtered = tagged

    display_rows = []
    for _, row in filtered.iterrows():
        ts = row.get("ts_code", row.get("code", ""))
        name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
        cur_ret = row.get("cur_ret", None)
        if cur_ret is None and "entry_price" in row and row.get("entry_price", 0) > 0:
            pass

        tag = row.get("risk_tags", "")
        tag_color = STATUS_COLORS.get(tag.split(",")[0], STATUS_COLORS["neutral"])
        tag_label_map = {
            "sell": "建议卖出", "risk": "高风险", "near_max_hold": "临近max_hold",
            "near_stop_loss": "接近止损", "strong_hold": "强势持有",
        }
        tag_display = " / ".join(tag_label_map.get(t, t) for t in tag.split(","))

        display_rows.append({
            "代码": ts,
            "名称": name,
            "仓位": format_pct(row.get("weight", 0)),
            "成本价": format_price(row.get("entry_price")),
            "持仓天数": int(row.get("days_held", 0)) if pd.notna(row.get("days_held")) else "-",
            "入场评分": f"{row.get('score', 0):.3f}" if pd.notna(row.get("score")) else "-",
            "状态标签": tag_display,
        })

    if display_rows:
        display_df = pd.DataFrame(display_rows)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "状态标签": st.column_config.TextColumn(width="medium"),
            },
        )
    else:
        st.info("无匹配持仓")

    st.markdown("---")

    if st.button("🎯 前往操作计划 →"):
        st.switch_page("pages/4_🎯_操作计划.py")


render()
