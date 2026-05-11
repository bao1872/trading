# -*- coding: utf-8 -*-
"""
Purpose:
    候选与建议页 — 回答"系统建议关注什么"

Inputs:
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

from vis.common.data_loader import load_decisions_df, get_stock_name_map
from vis.common.components import render_date_selector, format_pct


def render():
    st.title("💡 候选与建议")

    date = render_date_selector("candidate_date")
    if not date:
        st.error("未找到交易日数据")
        return

    decisions_df = load_decisions_df(date)
    if decisions_df.empty:
        st.warning(f"{date} 无决策数据")
        return

    name_map = get_stock_name_map()

    buy_dec = decisions_df[decisions_df["action"] == "buy"] if "action" in decisions_df.columns else pd.DataFrame()
    cand_dec = decisions_df[decisions_df["action"] == "candidate"] if "action" in decisions_df.columns else pd.DataFrame()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 新买入候选")
        if buy_dec.empty:
            st.info("系统无新买入候选")
        else:
            rows = []
            for _, row in buy_dec.iterrows():
                ts = row.get("ts_code", "")
                name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                rows.append({
                    "排名": row.get("rank", "-"),
                    "代码": ts,
                    "名称": name,
                    "sell_reg": f"{row.get('score', 0):.3f}",
                    "计划价": f"{row.get('planned_price', 0):.2f}" if pd.notna(row.get("planned_price")) else "-",
                    "入选理由": row.get("why", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("🔄 持仓延申候选")
        if cand_dec.empty:
            st.info("无持仓延申候选")
        else:
            rows = []
            for _, row in cand_dec.iterrows():
                ts = row.get("ts_code", "")
                name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                is_held = row.get("is_held", False)
                is_pending = row.get("is_pending_buy", False)
                tag = "持仓延申" if is_held else ("待买入" if is_pending else "新候选")

                pred_buy_cls = row.get("pred_buy_cls", None)
                pred_sell_reg = row.get("pred_sell_reg", None)

                rows.append({
                    "排名": row.get("rank", "-"),
                    "代码": ts,
                    "名称": name,
                    "类型": tag,
                    "buy_cls": f"{pred_buy_cls:.3f}" if pd.notna(pred_buy_cls) else "-",
                    "sell_reg": f"{pred_sell_reg:.3f}" if pd.notna(pred_sell_reg) else "-",
                    "score": f"{row.get('score', 0):.3f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.button("🎯 前往制定操作计划 →"):
        st.switch_page("pages/4_🎯_操作计划.py")


render()
