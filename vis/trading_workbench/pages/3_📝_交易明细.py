# -*- coding: utf-8 -*-
"""
Purpose:
    交易明细页 — QMT 实盘委托 + 成交记录 + 持仓变化

Inputs:
    - QMT 实盘数据（委托/成交）
    - 本地 historical executions（回退）

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
    load_qmt_orders_df, load_qmt_trades_df,
    load_qmt_positions_df, load_executions_df,
    load_latest_date, refresh_qmt_connection,
    get_stock_name_map,
)
from vis.common.components import format_pct, format_price, format_money
from vis.common.theme import STATUS_COLORS


def render():
    st.title("📝 交易明细（QMT 实盘）")

    if st.button("🔄 刷新数据"):
        refresh_qmt_connection()
        st.rerun()

    name_map = get_stock_name_map()

    tab1, tab2, tab3 = st.tabs(["📋 当前委托", "✅ 今日成交", "📊 持仓变化"])

    with tab1:
        st.subheader("当前委托")
        orders_df = load_qmt_orders_df()

        if orders_df.empty:
            st.info("当前无委托")
        else:
            display_rows = []
            for _, row in orders_df.iterrows():
                code = row.get("stock_code", "")
                order_type = row.get("order_type", 0)
                side_label = "买入" if order_type > 0 else "卖出"
                status_map = {50: "已提交", 56: "已成交", 57: "已撤单"}
                status = status_map.get(row.get("order_status_code"), row.get("status_msg", ""))
                display_rows.append({
                    "代码": code,
                    "名称": name_map.get(code, ""),
                    "方向": side_label,
                    "委托价": format_price(row.get("price", 0)),
                    "委托量": int(row.get("order_volume", 0)),
                    "已成交": int(row.get("traded_volume", 0)),
                    "状态": status,
                })

            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("今日成交")
        trades_df = load_qmt_trades_df()

        if trades_df.empty:
            st.info("今日无成交")
        else:
            display_rows = []
            for _, row in trades_df.iterrows():
                code = row.get("stock_code", "")
                display_rows.append({
                    "代码": code,
                    "名称": name_map.get(code, ""),
                    "方向": row.get("side", ""),
                    "成交价": format_price(row.get("traded_price", 0)),
                    "成交量": int(row.get("traded_volume", 0)),
                })

            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("当前持仓概览")
        pos_df = load_qmt_positions_df()
        if not pos_df.empty:
            n_total = len(pos_df)
            total_mkt = pos_df["market_value"].sum() if "market_value" in pos_df.columns else 0
            profit_pos = len(pos_df[pos_df["profit_rate"] > 0].index) if "profit_rate" in pos_df.columns else 0
            profit_pct = profit_pos / n_total * 100 if n_total > 0 else 0

            st.markdown(f"**持仓数:** {n_total} 只 | **总市值:** {format_money(total_mkt)} | **盈利占比:** {profit_pct:.0f}%")
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("无持仓数据")


render()