# -*- coding: utf-8 -*-
"""
Purpose:
    具体操作计划页（最核心页面）— 明日 T+1 交易指令编辑器
    系统建议 vs 人工最终计划双列对比

Inputs:
    - holdings/T.parquet
    - decisions/T.parquet
    - predictions/T.parquet
    - manual_plans/T.parquet（已有计划）

Outputs:
    - manual_plans/YYYY-MM-DD.parquet（保存人工计划）

How to Run:
    通过 trading_workbench/app.py 自动加载

Side Effects:
    - 保存人工计划到 output/live/manual_plans/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import streamlit as st
import pandas as pd
from datetime import datetime

from vis.common.data_loader import (
    load_holdings_df, load_decisions_df, load_predictions_df,
    load_manual_plan_df, save_manual_plan, get_stock_name_map,
    compute_risk_tags,
)
from vis.common.components import (
    render_date_selector, render_metric_cards, format_pct, format_price,
)
from vis.common.theme import STATUS_COLORS


SELL_OPTIONS = ["全卖", "部分卖", "不卖"]
BUY_OPTIONS = ["买入", "不买", "延后观察"]
HOLD_OPTIONS = ["继续", "转减仓观察", "转卖出"]
WATCH_OPTIONS = ["无需操作", "盘中关注"]


def render():
    st.title("🎯 具体操作计划")

    date = render_date_selector("plan_date")
    if not date:
        st.error("未找到交易日数据")
        return

    col_load, col_save = st.columns(2)
    with col_load:
        load_clicked = st.button("📥 载入今日持仓与策略建议", use_container_width=True, type="primary")
    with col_save:
        save_clicked = st.button("💾 生成明日 T+1 交易计划", use_container_width=True)

    if load_clicked:
        st.session_state.plan_loaded = True
        st.rerun()

    if "plan_loaded" not in st.session_state:
        st.session_state.plan_loaded = False

    if not st.session_state.plan_loaded:
        existing = load_manual_plan_df(date)
        if not existing.empty:
            st.info(f"已存在 {date} 的人工计划（{len(existing)} 条），点击载入可重新生成")
            st.dataframe(existing, use_container_width=True, hide_index=True)
        else:
            st.warning("请先点击「载入今日持仓与策略建议」")
        return

    holdings_df = load_holdings_df(date)
    decisions_df = load_decisions_df(date)
    predictions_df = load_predictions_df(date)
    name_map = get_stock_name_map()

    if decisions_df.empty:
        st.error(f"{date} 无决策数据，请先运行 run_daily.py")
        return

    sell_dec = decisions_df[decisions_df["action"] == "sell"] if "action" in decisions_df.columns else pd.DataFrame()
    buy_dec = decisions_df[decisions_df["action"] == "buy"] if "action" in decisions_df.columns else pd.DataFrame()
    hold_dec = decisions_df[decisions_df["action"] == "hold"] if "action" in decisions_df.columns else pd.DataFrame()
    cand_dec = decisions_df[decisions_df["action"] == "candidate"] if "action" in decisions_df.columns else pd.DataFrame()

    tab_sell, tab_buy, tab_hold, tab_watch = st.tabs([
        f"📉 卖出计划 ({len(sell_dec)})",
        f"📈 买入计划 ({len(buy_dec)})",
        f"🔄 继续持有 ({len(hold_dec)})",
        f"👁️ 观察清单 ({len(cand_dec)})",
    ])

    plan_rows = []

    with tab_sell:
        st.subheader("明日卖出计划")
        if sell_dec.empty:
            st.info("系统无卖出建议")
        else:
            for i, (_, row) in enumerate(sell_dec.iterrows()):
                ts = row.get("ts_code", "")
                name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                reason = row.get("reason", "-")
                days = int(row.get("days_held", 0)) if pd.notna(row.get("days_held")) else "-"
                cur_ret = row.get("cur_ret", None)
                cur_ret_str = format_pct(cur_ret) if cur_ret is not None else "-"
                score = row.get("score", 0)
                why = row.get("why", "")

                col_info, col_plan = st.columns([3, 1])
                with col_info:
                    reason_color = STATUS_COLORS.get("sell", "#ef5350")
                    st.markdown(
                        f"**{name}** ({ts}) | "
                        f"持仓{days}天 | 收益率 {cur_ret_str} | "
                        f"系统原因: <span style='color:{reason_color}'>{reason}</span>"
                        + (f" | {why}" if why else ""),
                        unsafe_allow_html=True,
                    )
                with col_plan:
                    plan_action = st.selectbox(
                        "人工计划", SELL_OPTIONS, index=0,
                        key=f"sell_plan_{i}",
                    )
                    note = st.text_input("备注", "", key=f"sell_note_{i}")

                plan_rows.append({
                    "trade_date": date,
                    "ts_code": ts,
                    "system_action": "sell",
                    "plan_action": plan_action,
                    "planned_weight": 0,
                    "approved_weight": 0 if plan_action == "全卖" else None,
                    "priority_rank": i,
                    "score": score,
                    "reason": reason,
                    "signal_id": row.get("signal_id", ""),
                    "approved_by": "",
                    "note": note,
                    "created_at": datetime.now().isoformat(),
                })

    with tab_buy:
        st.subheader("明日买入计划")
        if buy_dec.empty:
            st.info("系统无买入建议")
        else:
            for i, (_, row) in enumerate(buy_dec.iterrows()):
                ts = row.get("ts_code", "")
                name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                rank = row.get("rank", i)
                score = row.get("score", 0)
                why = row.get("why", "")
                planned_price = row.get("planned_price", None)
                planned_weight = row.get("planned_weight", 0.1)

                pred_sell_reg = row.get("pred_sell_reg", None)
                pred_buy_cls = row.get("pred_buy_cls", None)

                col_info, col_plan = st.columns([3, 1])
                with col_info:
                    st.markdown(
                        f"**{name}** ({ts}) | 排名 #{rank} | "
                        f"sell_reg={pred_sell_reg:.3f} " if pd.notna(pred_sell_reg) else f"**{name}** ({ts}) | 排名 #{rank} | ",
                        unsafe_allow_html=True,
                    )
                    if pd.notna(pred_buy_cls):
                        st.markdown(f"buy_cls={pred_buy_cls:.3f}", unsafe_allow_html=True)
                    if why:
                        st.markdown(f"入选理由: {why}")
                    if pd.notna(planned_price):
                        st.markdown(f"计划买入价: {format_price(planned_price)}")

                with col_plan:
                    plan_action = st.selectbox(
                        "人工计划", BUY_OPTIONS, index=0,
                        key=f"buy_plan_{i}",
                    )
                    if plan_action == "买入":
                        approved_weight = st.number_input(
                            "仓位(%)", min_value=1, max_value=100,
                            value=int(planned_weight * 100) if pd.notna(planned_weight) else 10,
                            key=f"buy_weight_{i}",
                        )
                    else:
                        approved_weight = 0
                    note = st.text_input("备注", "", key=f"buy_note_{i}")

                plan_rows.append({
                    "trade_date": date,
                    "ts_code": ts,
                    "system_action": "buy",
                    "plan_action": plan_action,
                    "planned_weight": planned_weight,
                    "approved_weight": approved_weight / 100 if plan_action == "买入" else 0,
                    "priority_rank": rank,
                    "score": score,
                    "reason": why,
                    "signal_id": row.get("signal_id", ""),
                    "approved_by": "",
                    "note": note,
                    "created_at": datetime.now().isoformat(),
                })

    with tab_hold:
        st.subheader("明日继续持有")
        if hold_dec.empty:
            st.info("无继续持有建议")
        else:
            for i, (_, row) in enumerate(hold_dec.iterrows()):
                ts = row.get("ts_code", "")
                name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                days = int(row.get("days_held", 0)) if pd.notna(row.get("days_held")) else "-"
                cur_ret = row.get("cur_ret", None)
                cur_ret_str = format_pct(cur_ret) if cur_ret is not None else "-"
                why = row.get("why", "")

                col_info, col_plan = st.columns([3, 1])
                with col_info:
                    st.markdown(
                        f"**{name}** ({ts}) | 持仓{days}天 | 收益率 {cur_ret_str}"
                        + (f" | {why}" if why else ""),
                    )
                with col_plan:
                    plan_action = st.selectbox(
                        "人工计划", HOLD_OPTIONS, index=0,
                        key=f"hold_plan_{i}",
                    )
                    note = st.text_input("备注", "", key=f"hold_note_{i}")

                plan_rows.append({
                    "trade_date": date,
                    "ts_code": ts,
                    "system_action": "hold",
                    "plan_action": plan_action,
                    "planned_weight": row.get("weight", 0.1),
                    "approved_weight": row.get("weight", 0.1),
                    "priority_rank": i,
                    "score": row.get("score", 0),
                    "reason": why,
                    "signal_id": row.get("signal_id", ""),
                    "approved_by": "",
                    "note": note,
                    "created_at": datetime.now().isoformat(),
                })

    with tab_watch:
        st.subheader("明日观察清单")
        if cand_dec.empty:
            st.info("无候选观察标的")
        else:
            for i, (_, row) in enumerate(cand_dec.iterrows()):
                ts = row.get("ts_code", "")
                name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
                rank = row.get("rank", i)
                score = row.get("score", 0)
                is_held = row.get("is_held", False)
                is_pending = row.get("is_pending_buy", False)
                pred_buy_cls = row.get("pred_buy_cls", None)
                pred_sell_reg = row.get("pred_sell_reg", None)

                tag = "持仓延申" if is_held else ("待买入" if is_pending else "新候选")

                col_info, col_plan = st.columns([3, 1])
                with col_info:
                    st.markdown(
                        f"**{name}** ({ts}) | 排名 #{rank} | "
                        f"类型: {tag} | score={score:.3f}"
                    )
                    if pd.notna(pred_buy_cls):
                        st.markdown(f"buy_cls={pred_buy_cls:.3f} | sell_reg={pred_sell_reg:.3f}" if pd.notna(pred_sell_reg) else f"buy_cls={pred_buy_cls:.3f}")

                with col_plan:
                    plan_action = st.selectbox(
                        "人工计划", WATCH_OPTIONS, index=0,
                        key=f"watch_plan_{i}",
                    )
                    note = st.text_input("备注", "", key=f"watch_note_{i}")

                plan_rows.append({
                    "trade_date": date,
                    "ts_code": ts,
                    "system_action": "candidate",
                    "plan_action": plan_action,
                    "planned_weight": 0,
                    "approved_weight": 0,
                    "priority_rank": rank,
                    "score": score,
                    "reason": "candidate",
                    "signal_id": row.get("signal_id", ""),
                    "approved_by": "",
                    "note": note,
                    "created_at": datetime.now().isoformat(),
                })

    if save_clicked and plan_rows:
        plan_df = pd.DataFrame(plan_rows)
        path = save_manual_plan(date, plan_df)

        st.success(f"✅ T+1 交易计划已保存: {path}")

        st.subheader("计划摘要")
        buy_plans = plan_df[plan_df["system_action"] == "buy"]
        sell_plans = plan_df[plan_df["system_action"] == "sell"]
        hold_plans = plan_df[plan_df["system_action"] == "hold"]

        render_metric_cards([
            {"label": "计划买入", "value": str(len(buy_plans[buy_plans["plan_action"] == "买入"]))},
            {"label": "计划卖出", "value": str(len(sell_plans[sell_plans["plan_action"] == "全卖"]))},
            {"label": "继续持有", "value": str(len(hold_plans[hold_plans["plan_action"] == "继续"]))},
            {"label": "总条目", "value": str(len(plan_df))},
        ])

        st.dataframe(plan_df, use_container_width=True, hide_index=True)
    elif save_clicked and not plan_rows:
        st.warning("无计划条目可保存")


render()
