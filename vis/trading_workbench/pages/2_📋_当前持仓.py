# -*- coding: utf-8 -*-
"""
Purpose:
    当前持仓页 — 展示 QMT 真实持仓 + AI 风险标签

Inputs:
    - QMT 实盘持仓数据
    - AI decisions parquet

Outputs:
    - Streamlit 页面渲染

How to Run:
    通过 trading_workbench/app.py 自动加载

Side Effects:
    - 无（只读数据，不下单）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import streamlit as st
import pandas as pd

from vis.common.data_loader import (
    load_decisions_df, load_latest_date, get_stock_name_map,
    compute_risk_tags, refresh_qmt_connection,
    load_qmt_positions_df, load_real_holdings_with_risk_tags,
)
from vis.common.components import (
    render_date_selector, render_paginated_table, format_pct, format_price,
    format_money,
)
from vis.common.theme import STATUS_COLORS


def render():
    st.title("📋 当前持仓（QMT 实盘）")

    if st.button("🔄 刷新 QMT 持仓数据"):
        refresh_qmt_connection()
        st.rerun()

    tagged = load_real_holdings_with_risk_tags()

    if tagged.empty:
        st.warning("无法获取 QMT 持仓数据。请检查 QMT 服务连接。")
        # 回退到本地数据
        date = load_latest_date()
        if date:
            from vis.common.data_loader import load_holdings_df
            holdings = load_holdings_df(date)
            if not holdings.empty:
                st.info(f"降级使用本地历史数据 ({date})")
                tagged = holdings.copy()
                if "risk_tags" not in tagged.columns:
                    tagged["risk_tags"] = ""
            else:
                st.error("无本地持仓数据，请确保 QMT 服务运行中")
                return
        else:
            return

    name_map = get_stock_name_map()

    st.markdown("---")

    # 筛选按钮
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

    risk_col = "risk_tags" if "risk_tags" in tagged.columns else None

    if show_sell and risk_col:
        filtered = tagged[tagged[risk_col].str.contains("sell", na=False)]
    elif show_risk and risk_col:
        filtered = tagged[tagged[risk_col].str.contains("risk|near_max_hold", na=False)]
    elif show_loss:
        if "profit_rate" in tagged.columns:
            filtered = tagged[pd.to_numeric(tagged["profit_rate"], errors="coerce") < 0]
        else:
            filtered = tagged
    elif show_strong and risk_col:
        filtered = tagged[tagged[risk_col].str.contains("strong_hold", na=False)]
    else:
        filtered = tagged

    display_rows = []
    for _, row in filtered.iterrows():
        ts = row.get("ts_code", "")
        name = row.get("instrument_name", "")
        if not name:
            name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)

        profit_rate = row.get("profit_rate", None)
        if profit_rate is not None and pd.notna(profit_rate):
            profit_str = format_pct(profit_rate)
        else:
            profit_str = "-"

        vol = int(row.get("volume", 0)) if pd.notna(row.get("volume", 0)) else 0
        can_use = int(row.get("can_use_volume", 0)) if pd.notna(row.get("can_use_volume", 0)) else 0
        mkt_val = row.get("market_value", 0)
        avg_price = row.get("avg_price", 0)
        last_price = row.get("last_price", 0)

        tag = str(row.get(risk_col, "")) if risk_col else ""
        tag_label_map = {
            "sell": "建议卖出", "risk": "高风险", "near_max_hold": "临近max_hold",
            "near_stop_loss": "接近止损", "strong_hold": "强势持有",
        }
        tag_display = " / ".join(tag_label_map.get(t, t) for t in tag.split(",") if t)

        display_rows.append({
            "代码": ts,
            "名称": name,
            "持仓量": vol,
            "可用量": can_use,
            "成本价": format_price(avg_price),
            "现价": format_price(last_price),
            "市值": format_money(mkt_val),
            "盈亏": profit_str,
            "风险标签": tag_display or "-",
        })

    if display_rows:
        display_df = pd.DataFrame(display_rows)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "风险标签": st.column_config.TextColumn(width="medium"),
            },
        )
    else:
        st.info("无匹配持仓")

    st.markdown("---")

    if st.button("🎯 前往操作计划 →"):
        st.switch_page("pages/4_🎯_操作计划.py")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("**数据来源:** QMT 实盘（通过 quant-qmt-proxy）")
    with col_info2:
        st.markdown("**风险标签:** 由 AI 模型计算")


render()