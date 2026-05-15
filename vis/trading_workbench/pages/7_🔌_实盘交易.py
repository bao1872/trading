# -*- coding: utf-8 -*-
"""
Purpose:
    实盘交易控制台 — 加载操作计划、校验、Dry-Run、实盘执行

Inputs:
    - AI 生成的 manual_plan (parquet)
    - QMT 实盘数据（资产/持仓/委托/成交）
    - 人工确认

Outputs:
    - 真实 QMT 下单/撤单
    - 执行结果展示

How to Run:
    通过 trading_workbench/app.py 自动加载

Examples:
    点击「加载操作计划」→ 校验 → Dry-Run → 确认 → 实盘执行

Side Effects:
    - dry_run=True 时无副作用
    - dry_run=False 时会实际向 QMT 发起下单/撤单
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import streamlit as st
import pandas as pd
from datetime import datetime

from vis.common.data_loader import (
    load_manual_plan_df, load_latest_date, get_stock_name_map,
    refresh_qmt_connection, load_qmt_asset, load_qmt_positions_df,
    load_qmt_orders_df, load_qmt_trades_df, get_qmt_client,
)
from vis.common.components import format_pct, format_price, format_money
from vis.common.theme import STATUS_COLORS

from qmt_trader.executor import TradeExecutor
from qmt_trader.session import SessionManager
import qmt_trader.config as qmt_cfg


MODE_LABELS = {
    "prod": "生产环境 (QMT 实盘)",
    "mock": "Mock 模式 (本地测试)",
}

ORDER_STATUS_MAP = {
    48: "未报",
    49: "待报",
    50: "已报",
    51: "已报待撤",
    52: "部成待撤",
    53: "部撤",
    54: "已撤",
    55: "部成",
    56: "已成",
    57: "废单",
}


def render():
    st.title("🔌 实盘交易控制台")

    # ==================== 顶部状态栏 ====================
    conn = refresh_qmt_connection()
    qmt_online = conn["connected"]
    sid = conn["session_id"]

    icon = "🟢" if qmt_online else "🔴"
    st.markdown(
        f"{icon} QMT: {'已连接' if qmt_online else '不可达'} | "
        f"会话: {sid or '无'} | "
        f"模式: {MODE_LABELS.get(qmt_cfg.QMT_MODE, qmt_cfg.QMT_MODE)} | "
        f"镜像: {qmt_cfg.QMT_PROXY_URL}"
    )

    if not qmt_online:
        st.error("QMT 服务不可达，请检查服务是否运行在 " + qmt_cfg.QMT_PROXY_URL)
        return

    col_actions = st.columns(4)
    with col_actions[0]:
        if st.button("🔄 刷新连接", use_container_width=True):
            refresh_qmt_connection()
            st.rerun()

    with col_actions[1]:
        if st.button("📊 加载操作计划", use_container_width=True):
            st.rerun()

    with col_actions[2]:
        if st.button("📋 刷新持仓", use_container_width=True):
            st.rerun()

    with col_actions[3]:
        if st.button("📝 刷新委托", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # ==================== 账户概览 ====================
    st.subheader("📊 实盘账户概览")
    asset_data = load_qmt_asset()
    if asset_data:
        cols = st.columns(5)
        with cols[0]:
            st.metric("总资产", format_money(asset_data["total_asset"]))
        with cols[1]:
            st.metric("现金", format_money(asset_data["cash"]))
        with cols[2]:
            st.metric("市值", format_money(asset_data["market_value"]))
        with cols[3]:
            pos = load_qmt_positions_df()
            st.metric("持仓数", f"{len(pos)} 只")
        with cols[4]:
            pct = (asset_data["cash"] / asset_data["total_asset"] * 100
                   if asset_data["total_asset"] > 0 else 0)
            st.metric("可用仓位", f"{pct:.1f}%")
    else:
        st.warning("无法获取账户资产")

    st.markdown("---")

    # ==================== 待执行计划 ====================
    st.subheader("📋 待执行计划")
    date = load_latest_date()
    plan_df = load_manual_plan_df(date) if date else pd.DataFrame()

    if plan_df.empty:
        st.info("暂无已保存的操作计划。请先在「操作计划」页制定计划。")
        if st.button("🎯 前往操作计划 →"):
            st.switch_page("pages/4_🎯_操作计划.py")
        return

    name_map = get_stock_name_map()
    display_rows = []
    for _, row in plan_df.iterrows():
        ts = row.get("ts_code", "")
        name = name_map.get(ts, ts.split(".")[0] if "." in ts else ts)
        action = row.get("plan_action", row.get("system_action", ""))
        display_rows.append({
            "ts_code": ts,
            "名称": name,
            "AI建议": row.get("system_action", ""),
            "人工计划": action,
            "计划权重": format_pct(row.get("approved_weight", 0)),
            "评分": f"{row.get('score', 0):.3f}" if pd.notna(row.get("score")) else "-",
        })

    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    # ==================== 计划校验 ====================
    st.markdown("---")
    st.subheader("🔍 计划校验")

    executor = TradeExecutor(get_qmt_client())
    valid, msg, detail = executor.validate_plan(sid, plan_df)

    if valid:
        if detail.get("warnings"):
            st.warning(f"校验通过但有警告: {'; '.join(detail['warnings'])}")
        else:
            st.success(f"✅ {msg}")
    else:
        st.error(f"❌ {msg}")
        for err in detail.get("errors", []):
            st.error(f"  - {err}")

    for warn in detail.get("warnings", []):
        st.warning(f"⚠️ {warn}")

    # ==================== 执行操作 ====================
    st.markdown("---")
    st.subheader("⚡ 执行操作")

    col_dry, col_real = st.columns(2)

    with col_dry:
        st.markdown("### 🔍 Dry-Run 模拟")
        st.caption("仅校验不实际下单，安全预览预期结果")
        if st.button("模拟执行", use_container_width=True, type="secondary", key="dry_run_btn"):
            with st.spinner("Dry-Run 模拟执行中..."):
                results = executor.execute_plan(plan_df, dry_run=True, session_id=sid)
                st.session_state["exec_results"] = results
                st.session_state["exec_mode"] = "dry_run"
            st.rerun()

    with col_real:
        st.markdown("### ⚠️ 实盘执行")
        st.caption("⚠️ 将真实向 QMT 下单，请谨慎操作")

        confirm_col1, confirm_col2 = st.columns(2)
        with confirm_col1:
            confirmed = st.checkbox("我确认要执行实盘交易", value=False, key="real_confirm")
        with confirm_col2:
            cancel_before = st.checkbox("先撤销所有挂单", value=False, key="cancel_before")

        real_disabled = not confirmed

        if st.button("⚠️ 实盘下单", use_container_width=True, type="primary",
                     disabled=real_disabled, key="real_exec_btn"):
            with st.spinner("实盘执行中...本操作不可撤销"):
                results = executor.execute_plan(
                    plan_df,
                    dry_run=False,
                    session_id=sid,
                    cancel_existing=cancel_before,
                )
                st.session_state["exec_results"] = results
                st.session_state["exec_mode"] = "real"
                st.session_state["exec_time"] = datetime.now().isoformat()
            st.rerun()

    # ==================== 执行结果 ====================
    if "exec_results" in st.session_state:
        st.markdown("---")
        st.subheader("✅ 执行结果")

        mode_label = "Dry-Run 模拟" if st.session_state.get("exec_mode") == "dry_run" else "实盘"
        exec_time = st.session_state.get("exec_time", datetime.now().isoformat())
        st.caption(f"执行模式: {mode_label} | 时间: {exec_time}")

        results = st.session_state["exec_results"]
        result_rows = []
        for r in results:
            result_rows.append({
                "ts_code": r.ts_code,
                "计划动作": r.plan_action,
                "结果": r.result,
                "价格": format_price(r.price) if r.price else "-",
                "数量": r.volume if r.volume else "-",
                "委托ID": r.order_id if r.order_id else "-",
                "错误": r.error if r.error else "",
            })

        st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)

        success = sum(1 for r in results if r.result == "success")
        failed = sum(1 for r in results if r.result == "failed")
        dry = sum(1 for r in results if r.result == "dry_run")

        cols_summary = st.columns(3)
        with cols_summary[0]:
            st.metric("成功", success)
        with cols_summary[1]:
            st.metric("失败", failed, delta=None)
        with cols_summary[2]:
            st.metric("Dry-Run", dry)

    # ==================== 当前挂单（可撤单） ====================
    st.markdown("---")
    st.subheader("📝 当前挂单")

    orders_df = load_qmt_orders_df()
    if not orders_df.empty:
        display_rows = []
        for _, row in orders_df.iterrows():
            sts = ORDER_STATUS_MAP.get(row.get("order_status_code"), str(row.get("order_status_code")))
            display_rows.append({
                "代码": row.get("stock_code", ""),
                "委托ID": row.get("order_id", ""),
                "系统ID": row.get("order_sysid", ""),
                "委托量": int(row.get("order_volume", 0)),
                "已成交": int(row.get("traded_volume", 0)),
                "价格": format_price(row.get("price", 0)),
                "状态": sts,
            })
        orders_display = pd.DataFrame(display_rows)
        st.dataframe(orders_display, use_container_width=True, hide_index=True)

        uncompleted = orders_display[
            ~orders_display["状态"].isin(["已成", "已撤", "废单"])
        ]
        if not uncompleted.empty:
            st.markdown(f"**{len(uncompleted)} 条未完成委托**")
            cancel_order_id = st.selectbox(
                "选择要撤销的委托",
                uncompleted["委托ID"].tolist(),
                format_func=lambda x: f"{x}",
                key="cancel_select",
            )
            if st.button("🧹 撤单", type="secondary", key="cancel_single"):
                from qmt_trader.order import OrderAPI
                api = OrderAPI(get_qmt_client())

                row = orders_display[orders_display["委托ID"] == cancel_order_id].iloc[0]
                stock_code = row["代码"]
                market = OrderAPI.infer_market(stock_code)
                ok = api.cancel(sid, cancel_order_id, market=market,
                                order_sysid=row["系统ID"])
                if ok:
                    st.success(f"撤单成功: {cancel_order_id}")
                else:
                    st.error(f"撤单失败: {cancel_order_id}")
                st.rerun()

        if st.button("🧹 一键撤所有挂单", type="secondary", key="cancel_all"):
            executor._batch_cancel_all(sid)
            st.success("已提交撤单请求")
            st.rerun()
    else:
        st.info("当前无委托")

    st.markdown("---")
    st.caption(f"QMT 代理: {qmt_cfg.QMT_PROXY_URL} | 账户: {qmt_cfg.QMT_ACCOUNT_ID} | 模式: {qmt_cfg.QMT_MODE}")


render()