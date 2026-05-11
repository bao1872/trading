# -*- coding: utf-8 -*-
"""
Purpose:
    总览 Dashboard — 30秒内回答"账户现在怎么样、明天大概怎么做"

Inputs:
    - live_equity_curve.csv
    - holdings/T.parquet
    - decisions/T.parquet
    - executions/T.parquet

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
    load_equity_curve, load_holdings_df, load_decisions_df,
    load_executions_df, load_latest_date, get_stock_name_map,
    compute_risk_tags,
)
from vis.common.components import (
    render_date_selector, render_metric_cards, render_status_light,
    format_pct,
)
from vis.common.chart_builder import build_equity_curve_chart, build_daily_return_bar
from vis.common.theme import STATUS_COLORS


def render():
    st.title("📊 总览 Dashboard")

    date = render_date_selector("overview_date")
    if not date:
        st.error("未找到交易日数据，请先运行 run_daily.py")
        return

    eq_df = load_equity_curve()
    holdings_df = load_holdings_df(date)
    decisions_df = load_decisions_df(date)
    executions_df = load_executions_df(date)

    st.markdown("---")

    st.subheader("当前账户")
    if not eq_df.empty:
        latest = eq_df.iloc[-1]
        nav = latest.get("nav_live", 0)
        daily_ret = latest.get("daily_return", 0)
        drawdown = latest.get("drawdown", 0)
        n_pos = int(latest.get("n_positions", 0)) if pd.notna(latest.get("n_positions")) else 0

        if abs(daily_ret) < 1:
            daily_ret_pct = daily_ret * 100
        else:
            daily_ret_pct = daily_ret

        if abs(drawdown) < 1:
            dd_pct = drawdown * 100
        else:
            dd_pct = drawdown

        cash_pct = max(0, (1 - n_pos / 10)) * 100

        render_metric_cards([
            {"label": "账户净值", "value": f"{nav:.4f}"},
            {"label": "当日收益", "value": f"{daily_ret_pct:.2f}%", "delta": f"{daily_ret_pct:.2f}%", "delta_color": "normal"},
            {"label": "当前回撤", "value": f"{dd_pct:.2f}%", "delta": f"{dd_pct:.2f}%", "delta_color": "inverse"},
            {"label": "持仓数量", "value": f"{n_pos}/10"},
            {"label": "可用仓位", "value": f"{cash_pct:.0f}%"},
        ])
    else:
        st.warning("净值曲线数据为空")

    st.markdown("---")

    col_b, col_c = st.columns(2)

    with col_b:
        st.subheader("持仓风险面板")
        if not holdings_df.empty and not decisions_df.empty:
            tagged = compute_risk_tags(holdings_df, decisions_df)
            name_map = get_stock_name_map()

            sell_count = sum(1 for t in tagged["risk_tags"] if "sell" in t)
            risk_count = sum(1 for t in tagged["risk_tags"] if "risk" in t)
            near_max = sum(1 for t in tagged["risk_tags"] if "near_max_hold" in t)
            near_sl = sum(1 for t in tagged["risk_tags"] if "near_stop_loss" in t)

            st.markdown(
                f'<p><span style="color:{STATUS_COLORS["sell"]}">🔴 建议卖出: {sell_count}</span></p>'
                f'<p><span style="color:{STATUS_COLORS["risk"]}">🟠 高风险持仓: {risk_count}</span></p>'
                f'<p><span style="color:{STATUS_COLORS["watch"]}">🟡 临近 max_hold: {near_max}</span></p>'
                f'<p><span style="color:{STATUS_COLORS["buy"]}">🔵 浮亏接近止损: {near_sl}</span></p>',
                unsafe_allow_html=True,
            )

            if sell_count > 0:
                sell_codes = []
                for _, row in tagged.iterrows():
                    if "sell" in row["risk_tags"]:
                        ts = row.get("ts_code", row.get("code", ""))
                        name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                        sell_codes.append(name)
                st.markdown(f"卖出标的: {', '.join(sell_codes)}")
        else:
            st.info("无持仓或决策数据")

        if st.button("📋 查看持仓详情 →"):
            st.switch_page("pages/2_📋_当前持仓.py")

    with col_c:
        st.subheader("明日候选摘要")
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

    st.subheader("系统状态灯")
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        has_dec = not decisions_df.empty
        render_status_light("决策链运行", "ok" if has_dec else "error")
    with col_d2:
        has_hold = not holdings_df.empty
        has_exec = not executions_df.empty
        render_status_light("账本完整性", "ok" if (has_hold and has_exec) else "warn")
    with col_d3:
        from vis.common.data_loader import load_action_plan_json
        plan = load_action_plan_json(date)
        render_status_light("操作包生成", "ok" if plan else "warn")
    with col_d4:
        render_status_light("一致性检查", "ok")

    st.markdown("---")

    st.subheader("净值走势")
    if not eq_df.empty:
        period = st.radio("区间", ["最近20天", "最近60天", "全部"], index=2, horizontal=True, key="eq_period")
        period_map = {"最近20天": "20d", "最近60天": "60d", "全部": "all"}
        fig = build_equity_curve_chart(eq_df, period=period_map[period])
        st.plotly_chart(fig, use_container_width=True)


render()
