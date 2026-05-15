# -*- coding: utf-8 -*-
"""
Purpose:
    总览 Dashboard — 30秒内回答"账户现在怎么样、明天大概怎么做"
    使用 QMT 实盘数据作为主数据源，本地 CSV 作为回退。

Inputs:
    - QMT 实盘数据（资产/持仓/委托/成交）通过 qmt-proxy
    - AI 模型输出（decisions parquet, predictions parquet, action_plans JSON）
    - 本地历史记录（live_equity_curve.csv 回退用）

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
    load_equity_curve, load_decisions_df,
    load_latest_date, get_stock_name_map,
    compute_risk_tags, refresh_qmt_connection,
    load_qmt_asset, load_qmt_positions_df,
    load_real_holdings_with_risk_tags,
)
from vis.common.components import (
    render_date_selector, render_metric_cards, render_status_light,
    format_pct, format_money,
)
from vis.common.chart_builder import build_equity_curve_chart, build_daily_return_bar
from vis.common.theme import STATUS_COLORS


def render():
    st.title("📊 总览 Dashboard")

    # ==================== QMT 连接状态面板 ====================
    conn = refresh_qmt_connection()
    qmt_connected = conn["connected"]

    st.markdown("### 🔌 QMT 实盘连接")
    cols = st.columns(4)
    with cols[0]:
        icon = "🟢" if qmt_connected else "🔴"
        st.metric("QMT 服务", f"{icon} {'已连接' if qmt_connected else '不可达'}",
                  delta=conn.get("error", ""))
    with cols[1]:
        has_sid = bool(conn.get("session_id"))
        st.metric("交易会话", "🟢 就绪" if has_sid else "🔴 未创建")
    with cols[2]:
        asset_data = load_qmt_asset()
        st.metric("实盘订单", "🟢 启用" if asset_data else "🟡 待获取")
    with cols[3]:
        last_update = "实时" if asset_data else "无数据"
        st.metric("数据新鲜度", last_update)

    st.markdown("---")

    # ==================== 账户概览 ====================
    st.subheader("💰 实盘账户概览")

    if qmt_connected and asset_data:
        total_asset = asset_data["total_asset"]
        cash_val = asset_data["cash"]
        mkt_val = asset_data["market_value"]

        pos_df = load_qmt_positions_df()
        n_pos = len(pos_df)

        eq_df = load_equity_curve()
        if not eq_df.empty:
            prev_nav = eq_df.iloc[-1].get("nav_live", total_asset)
        else:
            prev_nav = total_asset
        daily_ret_pct = ((total_asset - prev_nav) / prev_nav * 100) if prev_nav != total_asset else 0

        cash_pct = (cash_val / total_asset * 100) if total_asset > 0 else 0

        render_metric_cards([
            {"label": "总资产", "value": format_money(total_asset)},
            {"label": "现金", "value": format_money(cash_val)},
            {"label": "市值", "value": format_money(mkt_val)},
            {"label": "可用仓位", "value": f"{cash_pct:.1f}%"},
            {"label": "持仓数", "value": f"{n_pos} 只"},
        ])
    else:
        st.warning("QMT 连接不可用，尝试加载本地数据...")
        eq_df = load_equity_curve()
        if not eq_df.empty:
            latest = eq_df.iloc[-1]
            nav = latest.get("nav_live", 0)
            n_pos = int(latest.get("n_positions", 0)) if pd.notna(latest.get("n_positions")) else 0
            cash_pct_val = max(0, (1 - n_pos / 10)) * 100
            render_metric_cards([
                {"label": "账户净值(历史)", "value": f"{nav:.4f}"},
                {"label": "持仓数量(历史)", "value": f"{n_pos}/10"},
                {"label": "可用仓位(历史)", "value": f"{cash_pct_val:.0f}%"},
            ])
        else:
            st.error("无历史数据，请确保 QMT 服务在运行")

    st.markdown("---")

    # ==================== 风险面板 + 候选摘要 ====================
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📋 持仓风险面板")
        tagged = load_real_holdings_with_risk_tags()
        if not tagged.empty and "risk_tags" in tagged.columns:
            name_map = get_stock_name_map()

            sell_count = sum(1 for t in tagged["risk_tags"] if "sell" in str(t))
            risk_count = sum(1 for t in tagged["risk_tags"] if "risk" in str(t))
            near_max = sum(1 for t in tagged["risk_tags"] if "near_max_hold" in str(t))
            near_sl = sum(1 for t in tagged["risk_tags"] if "near_stop_loss" in str(t))

            st.markdown(
                f'<p><span style="color:{STATUS_COLORS["sell"]}">🔴 建议卖出: {sell_count}</span></p>'
                f'<p><span style="color:{STATUS_COLORS["risk"]}">🟠 高风险: {risk_count}</span></p>'
                f'<p><span style="color:{STATUS_COLORS["watch"]}">🟡 临近max_hold: {near_max}</span></p>'
                f'<p><span style="color:{STATUS_COLORS["buy"]}">🔵 浮亏近止损: {near_sl}</span></p>',
                unsafe_allow_html=True,
            )

            if sell_count > 0:
                sell_codes = []
                for _, row in tagged.iterrows():
                    if "sell" in str(row["risk_tags"]):
                        ts = row.get("ts_code", "")
                        name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                        sell_codes.append(name)
                st.markdown(f"卖出标的: {', '.join(sell_codes)}")
        else:
            st.info("无持仓或 AI 决策数据")

        if st.button("📋 查看持仓详情 →"):
            st.switch_page("pages/2_📋_当前持仓.py")

    with col_right:
        st.subheader("💡 AI 候选摘要")
        date = load_latest_date()
        decisions_df = load_decisions_df(date) if date else pd.DataFrame()

        if not decisions_df.empty and "action" in decisions_df.columns:
            buy_count = len(decisions_df[decisions_df["action"] == "buy"])
            sell_count = len(decisions_df[decisions_df["action"] == "sell"])
            hold_count = len(decisions_df[decisions_df["action"] == "hold"])
            cand_count = len(decisions_df[decisions_df["action"] == "candidate"])

            st.markdown(
                f'<p><span style="color:{STATUS_COLORS["buy"]}">📈 建议买入: {buy_count}</span></p>'
                f'<p><span style="color:{STATUS_COLORS["sell"]}">📉 建议卖出: {sell_count}</span></p>'
                f'<p><span style="color:{STATUS_COLORS["hold"]}">🔄 继续持有: {hold_count}</span></p>'
                f'<p><span style="color:{STATUS_COLORS["neutral"]}">🆕 候选池: {cand_count}</span></p>',
                unsafe_allow_html=True,
            )

            buy_dec = decisions_df[decisions_df["action"] == "buy"].head(5)
            if not buy_dec.empty and "ts_code" in buy_dec.columns:
                name_map = get_stock_name_map()
                st.markdown("**新候选 TOP5:**")
                for _, row in buy_dec.iterrows():
                    ts = row.get("ts_code", "")
                    name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                    score = row.get("score", 0)
                    st.markdown(f"  {row.get('rank', '-')}. {name} (score={score:.3f})")
        else:
            st.info("无决策数据")

        if st.button("🎯 制定操作计划 →"):
            st.switch_page("pages/4_🎯_操作计划.py")

    st.markdown("---")

    # ==================== 系统状态灯 ====================
    st.subheader("🔔 系统状态灯")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        has_dec = not decisions_df.empty if 'decisions_df' in dir() else False
        render_status_light("决策链运行", "ok" if has_dec else "error")
    with col_d2:
        render_status_light("QMT 连接", "ok" if qmt_connected else "error")
    with col_d3:
        from vis.common.data_loader import load_action_plan_json
        plan = load_action_plan_json(date) if date else {}
        render_status_light("操作包生成", "ok" if plan else "warn")
    with col_d4:
        render_status_light("实盘交易就绪", "ok" if qmt_connected and asset_data else "warn")

    st.markdown("---")

    # ==================== 净值走势 ====================
    st.subheader("📈 净值走势")
    eq_df = load_equity_curve()
    if not eq_df.empty:
        period = st.radio("区间", ["最近20天", "最近60天", "全部"],
                          index=2, horizontal=True, key="eq_period")
        period_map = {"最近20天": "20d", "最近60天": "60d", "全部": "all"}
        fig = build_equity_curve_chart(eq_df, period=period_map[period])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无历史净值数据（需积累 QMT 每日资产快照）")


render()