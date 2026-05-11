# -*- coding: utf-8 -*-
"""
Purpose:
    交易明细页 — 回答"今天已经发生了什么"

Inputs:
    - executions/T.parquet
    - holdings/T.parquet + holdings/T-1.parquet

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
    load_executions_df, load_holdings_df, get_stock_name_map,
    get_prev_date,
)
from vis.common.components import render_date_selector, format_price


def render():
    st.title("📝 交易明细")

    date = render_date_selector("trades_date")
    if not date:
        st.error("未找到交易日数据")
        return

    executions_df = load_executions_df(date)
    holdings_df = load_holdings_df(date)
    prev_date = get_prev_date(date)
    prev_holdings_df = load_holdings_df(prev_date) if prev_date else pd.DataFrame()

    name_map = get_stock_name_map()

    tab_a, tab_b, tab_c = st.tabs(["✅ 已执行交易", "⚠️ 未执行/部分执行", "📊 持仓变化"])

    with tab_a:
        st.subheader("今日已执行交易")
        if executions_df.empty:
            st.info(f"{date} 无执行记录")
        else:
            executed = executions_df[executions_df["status"] == "executed"] if "status" in executions_df.columns else executions_df
            if executed.empty:
                st.info("今日无已执行交易")
            else:
                rows = []
                for _, row in executed.iterrows():
                    ts = row.get("ts_code", "")
                    name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                    action = row.get("action", "")
                    action_label = "买入" if action == "buy" else "卖出"
                    rows.append({
                        "代码": ts,
                        "名称": name,
                        "动作": action_label,
                        "执行价": format_price(row.get("executed_price")),
                        "计划价": format_price(row.get("planned_price")),
                        "信号ID": row.get("signal_id", "-"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_b:
        st.subheader("今日未执行/部分执行")
        if executions_df.empty:
            st.info(f"{date} 无执行记录")
        else:
            skipped = executions_df[executions_df["status"] == "skipped"] if "status" in executions_df.columns else pd.DataFrame()
            if skipped.empty:
                st.info("今日无未执行交易")
            else:
                rows = []
                for _, row in skipped.iterrows():
                    ts = row.get("ts_code", "")
                    name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                    action = row.get("action", "")
                    action_label = "买入" if action == "buy" else "卖出"
                    rows.append({
                        "代码": ts,
                        "名称": name,
                        "原计划动作": action_label,
                        "计划价": format_price(row.get("planned_price")),
                        "未执行原因": row.get("skip_reason", "-"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_c:
        st.subheader("今日持仓变化")
        if holdings_df.empty and prev_holdings_df.empty:
            st.info("无持仓数据")
        else:
            cur_codes = set()
            if not holdings_df.empty and "ts_code" in holdings_df.columns:
                cur_codes = set(holdings_df["ts_code"].tolist())
            elif not holdings_df.empty and "code" in holdings_df.columns:
                cur_codes = set(holdings_df["code"].tolist())

            prev_codes = set()
            if not prev_holdings_df.empty and "ts_code" in prev_holdings_df.columns:
                prev_codes = set(prev_holdings_df["ts_code"].tolist())
            elif not prev_holdings_df.empty and "code" in prev_holdings_df.columns:
                prev_codes = set(prev_holdings_df["code"].tolist())

            all_codes = cur_codes | prev_codes
            rows = []
            for ts in sorted(all_codes):
                name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                in_cur = ts in cur_codes
                in_prev = ts in prev_codes

                if in_cur and not in_prev:
                    direction = "🆕 新开仓"
                elif in_prev and not in_cur:
                    direction = "❌ 清仓"
                else:
                    direction = "➡️ 不变"

                cur_weight = 0
                if in_cur and not holdings_df.empty:
                    h = holdings_df[holdings_df.get("ts_code", holdings_df.get("code", pd.Series())) == ts]
                    if not h.empty:
                        cur_weight = h.iloc[0].get("weight", 0)

                prev_weight = 0
                if in_prev and not prev_holdings_df.empty:
                    h = prev_holdings_df[prev_holdings_df.get("ts_code", prev_holdings_df.get("code", pd.Series())) == ts]
                    if not h.empty:
                        prev_weight = h.iloc[0].get("weight", 0)

                rows.append({
                    "代码": ts,
                    "名称": name,
                    "昨日仓位": f"{prev_weight * 100:.1f}%" if prev_weight else "-",
                    "今日仓位": f"{cur_weight * 100:.1f}%" if cur_weight else "-",
                    "变化方向": direction,
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("无持仓变化")


render()
