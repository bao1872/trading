# -*- coding: utf-8 -*-
"""
Purpose:
    历史净值与复盘页 — 回答"过去表现如何"

Inputs:
    - live_equity_curve.csv
    - live_trade_report.csv

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

from vis.common.data_loader import load_equity_curve, load_trade_report
from vis.common.chart_builder import build_equity_curve_chart, build_daily_return_bar
from vis.common.components import render_metric_cards, format_pct


def render():
    st.title("📈 历史净值与复盘")

    eq_df = load_equity_curve()
    tr_df = load_trade_report()

    if eq_df.empty:
        st.warning("净值曲线数据为空")
        return

    st.subheader("净值曲线")
    period = st.radio("区间", ["最近20天", "最近60天", "全部"], index=2, horizontal=True, key="history_period")
    period_map = {"最近20天": "20d", "最近60天": "60d", "全部": "all"}
    fig = build_equity_curve_chart(eq_df, period=period_map[period])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("日收益率")
    fig_bar = build_daily_return_bar(eq_df, n_days=30)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    st.subheader("交易统计")
    if not tr_df.empty:
        col_names = tr_df.columns.tolist()
        pnl_col = None
        for c in col_names:
            if "净盈亏" in c or "pnl" in c.lower():
                pnl_col = c
                break

        if pnl_col:
            pnl_values = pd.to_numeric(tr_df[pnl_col].astype(str).str.replace("%", ""), errors="coerce")
            total_trades = len(tr_df)
            wins = (pnl_values > 0).sum()
            losses = (pnl_values < 0).sum()
            win_rate = wins / total_trades * 100 if total_trades > 0 else 0
            avg_pnl = pnl_values.mean()
            avg_win = pnl_values[pnl_values > 0].mean() if wins > 0 else 0
            avg_loss = pnl_values[pnl_values < 0].mean() if losses > 0 else 0
            pnl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

            render_metric_cards([
                {"label": "总交易数", "value": str(total_trades)},
                {"label": "胜率", "value": f"{win_rate:.1f}%"},
                {"label": "盈亏比", "value": f"{pnl_ratio:.2f}"},
                {"label": "平均盈亏%", "value": f"{avg_pnl:.2f}%"},
            ])
        else:
            st.dataframe(tr_df.describe(), use_container_width=True)
    else:
        st.info("交易报告数据为空")

    st.markdown("---")

    st.subheader("历史交易表")
    if not tr_df.empty:
        display_cols = []
        for c in tr_df.columns:
            if c in ["序号", "股票代码", "股票名称", "买入日期", "卖出日期",
                      "持仓天数", "净盈亏%", "退出原因", "入场评分"]:
                display_cols.append(c)

        if display_cols:
            st.dataframe(tr_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.dataframe(tr_df, use_container_width=True, hide_index=True)
    else:
        st.info("无历史交易数据")


render()
