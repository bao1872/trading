#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日级推理预测报告 — 完整决策链路展示（含双轨仓位映射 W1/W3）

Purpose:
    针对指定决策日（T日），复现完整决策链路并输出结构化报告：
    - T日收盘后决策 → T+1日开盘执行买卖
    - --date 参数 = 决策日（T日），不是执行日
    当前持仓 → 买入机会TOP10 → 风险候选排序 → 具体买入/卖出决策及依据 → 双轨仓位映射对比。

    复用 SSOT 决策函数 decide_daily_actions + build_daily_candidate_snapshot，
    确保报告中的买卖决策与回测引擎 100% 一致。

    新增双轨仓位映射（基于 tier_weight_mapping.py 实验结论）：
    - W1 (轻分层 A:1.3, C:0.7): 低风险基准版，模拟盘主用
    - W3 (强分层 A:2.0, C:0.3): 高收益实验版，对照组跟踪

    真实预测持久化：
    - 每日推理后自动保存预测结果到 predictions/YYYY-MM-DD.parquet (append-only)
    - 必须使用 up-to-date 数据，不允许使用合成或回退数据

Pipeline Position:
    生产流水线第二步（每日，重复）。
    上游: 06_daily_inference_replay.py, full_test_predictions.parquet, DB, predictions/
    下游: —

Inputs:
    - stop_experiment/output/full_test_predictions.parquet (全量预测)
    - stop_experiment/output/predictions/YYYY-MM-DD.parquet (真实预测，若存在)
    - DB: stock_k_data (K线数据)

Outputs:
    - Console: 8 段结构化报告（含 Section 5C 双轨仓位映射对比）
    - stop_experiment/output/daily_report/YYYY-MM-DD.md (Markdown 报告)
    - stop_experiment/output/predictions/YYYY-MM-DD.parquet (预测结果，首次生成时)

How to Run:
    # 默认跑最新日期
    python stop_experiment/pipeline/08_daily_inference_report.py

    # 指定日期
    python stop_experiment/pipeline/08_daily_inference_report.py --date 2026-05-08

    # dry-run (仅 console，不写文件)
    python stop_experiment/pipeline/08_daily_inference_report.py --date 2026-05-08 --dry-run

Side Effects:
    - 只读 parquet 和 DB
    - 写 Markdown 报告文件
    - 首次运行时写 predictions/YYYY-MM-DD.parquet (append-only，不覆盖)
"""

from __future__ import annotations

import sys
import os
import argparse
import copy
import json
import importlib.util
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, BACKTEST_DIR, V1_PARAMS, PREDICTIONS_DIR, DECISIONS_DIR, EXECUTIONS_DIR, V1_BASELINE,
)
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest,
)
from stop_experiment.backtest.simple_backtest import score_stocks

_SIX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_daily_inference_replay.py")
_six_spec = importlib.util.spec_from_file_location("_daily_inference_replay", _SIX_PATH)
_six_mod = importlib.util.module_from_spec(_six_spec)
_six_spec.loader.exec_module(_six_mod)
decide_daily_actions = _six_mod.decide_daily_actions
build_daily_candidate_snapshot = _six_mod.build_daily_candidate_snapshot
# find_exit_pred 从 decision_core 直接导入（SSOT），不再从 replay 模块间接引用
from stop_experiment.backtest.decision_core import find_exit_pred

# ==================== 常量 ====================

REPORT_DIR = os.path.join(OUTPUT_DIR, "daily_report")
HOLDINGS_DIR = os.path.join(OUTPUT_DIR, "holdings")
DEFAULT_MAX_STOCKS = V1_PARAMS.get("max_stocks_default", 10)

# Tier 仓位映射配置（与 tier_weight_mapping.py 保持一致）
TIER_WEIGHTS_W1 = {"A": 1.3, "B": 1.0, "C": 0.7}
TIER_WEIGHTS_W3 = {"A": 2.0, "B": 1.0, "C": 0.3}

# SSOT Tier 规则
TIER_A_SCORE = 0.2
TIER_A_BUY_CLS = 0.30
TIER_B_SCORE = 0.1
TIER_B_BUY_CLS = 0.70

# ==================== 真实预测读写 ====================

def load_real_predictions_for_date(target_date):
    """
    优先读取已保存的真实预测结果。

    Input:
        target_date: 目标日期 (datetime 或 str)

    Output:
        DataFrame: 真实预测数据 (含 obs_date, ts_code, signal_id, obs_day, pred_*, score, tier)
        None: 无真实预测文件
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    day_path = os.path.join(PREDICTIONS_DIR, f"{target_date.strftime('%Y-%m-%d')}.parquet")
    if os.path.exists(day_path):
        df = pd.read_parquet(day_path)
        print(f"  [真实预测] 已加载: {day_path} ({len(df)} 行)")
        return df
    return None


def save_holdings(date, holdings, pending_buys=None, pending_sells=None):
    """
    保存当日决策后持仓到 holdings/YYYY-MM-DD.parquet（账本驱动）。

    Input:
        date:          决策日
        holdings:      dict {code: {ts_code, buy_date, buy_price, weight, days_held, signal_id, score, ...}}
        pending_buys:  [(code, buy_price, ts_code, score, signal_id), ...] T+1 待买入
        pending_sells: [{"code": ..., "holding": {...}, "reason": ...}, ...] T+1 待卖出
    """
    os.makedirs(HOLDINGS_DIR, exist_ok=True)
    if isinstance(date, str):
        date = pd.to_datetime(date)

    rows = []
    for code, h in holdings.items():
        rows.append({
            "date": date,
            "code": code,
            "ts_code": h.get("ts_code", code),
            "entry_date": h.get("buy_date"),
            "entry_price": h.get("buy_price"),
            "weight": h.get("weight"),
            "days_held": h.get("days_held"),
            "signal_id": h.get("signal_id"),
            "score": h.get("score"),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(HOLDINGS_DIR, f"{date.strftime('%Y-%m-%d')}.parquet")
    df.to_parquet(path, index=False)
    print(f"  [持仓保存] {path} ({len(df)} 只)")

def save_decisions(date, holdings_after, pending_buys, pending_sells, sell_reasons):
    """
    T日决策后保存决策账本到 decisions/YYYY-MM-DD.parquet。
    三部分: hold (继续持有), sell (卖出), buy (买入)
    """
    os.makedirs(DECISIONS_DIR, exist_ok=True)
    if isinstance(date, str):
        date = pd.to_datetime(date)

    rows = []
    rank = 0
    for code, h in holdings_after.items():
        rows.append({
            "decision_date": date, "ts_code": h.get("ts_code", code),
            "signal_id": h.get("signal_id"), "action": "hold",
            "reason": "", "rank": rank, "score": h.get("score", 0),
            "planned_price": None, "planned_weight": h.get("weight"),
        })
        rank += 1
    for item in pending_sells:
        code = item["code"]
        h = item.get("holding", {})
        rows.append({
            "decision_date": date, "ts_code": h.get("ts_code", code),
            "signal_id": h.get("signal_id"), "action": "sell",
            "reason": item.get("reason", ""), "rank": -1,
            "score": h.get("score", 0), "planned_price": None,
            "planned_weight": 0,
        })
    rank = 0
    for code, bp_val, ts_code_val, sc_val, sid_val in pending_buys:
        rows.append({
            "decision_date": date, "ts_code": ts_code_val,
            "signal_id": sid_val, "action": "buy",
            "reason": "candidate_top", "rank": rank,
            "score": sc_val, "planned_price": bp_val,
            "planned_weight": 1.0 / DEFAULT_MAX_STOCKS,
        })
        rank += 1

    df = pd.DataFrame(rows)
    path = os.path.join(DECISIONS_DIR, f"{date.strftime('%Y-%m-%d')}.parquet")
    df.to_parquet(path, index=False)
    print(f"  [决策保存] {path} ({len(df)} 条)")


def save_executions(date, executed_buys, executed_sells, skipped_buys, skipped_sells):
    """
    T日开盘后保存执行账本到 executions/YYYY-MM-DD.parquet。
    executed: 成功执行的买卖
    skipped: 因停牌/涨跌停/无价格跳过的买卖
    """
    os.makedirs(EXECUTIONS_DIR, exist_ok=True)
    if isinstance(date, str):
        date = pd.to_datetime(date)

    rows = []
    for code, bp, ts_code, sc, sid in executed_buys:
        rows.append({
            "execution_date": date, "decision_date": date,
            "ts_code": ts_code, "signal_id": sid,
            "action": "buy", "planned_price": bp,
            "executed_price": bp, "status": "executed",
            "skip_reason": "",
        })
    for code, bp, ts_code, sc, sid, reason in skipped_buys:
        rows.append({
            "execution_date": date, "decision_date": date,
            "ts_code": ts_code, "signal_id": sid,
            "action": "buy", "planned_price": bp,
            "executed_price": None, "status": "skipped",
            "skip_reason": reason,
        })
    for item in executed_sells:
        code = item["code"]
        h = item.get("holding", {})
        rows.append({
            "execution_date": date, "decision_date": date,
            "ts_code": h.get("ts_code", code), "signal_id": h.get("signal_id"),
            "action": "sell", "planned_price": None,
            "executed_price": item.get("executed_price"),
            "status": "executed", "skip_reason": "",
        })
    for item in skipped_sells:
        code = item["code"]
        h = item.get("holding", {})
        rows.append({
            "execution_date": date, "decision_date": date,
            "ts_code": h.get("ts_code", code), "signal_id": h.get("signal_id"),
            "action": "sell", "planned_price": None,
            "executed_price": None, "status": "skipped",
            "skip_reason": item.get("reason", ""),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(EXECUTIONS_DIR, f"{date.strftime('%Y-%m-%d')}.parquet")
    df.to_parquet(path, index=False)
    print(f"  [执行保存] {path} ({len(df)} 条, executed={len(executed_buys)+len(executed_sells)} skipped={len(skipped_buys)+len(skipped_sells)})")



def load_holdings(date):
    """
    从 holdings/YYYY-MM-DD.parquet 读取持仓。

    Input:
        date: 目标日期

    Output:
        holdings dict 或 None
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    path = os.path.join(HOLDINGS_DIR, f"{date.strftime('%Y-%m-%d')}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    holdings = {}
    for _, row in df.iterrows():
        code = row["code"]
        holdings[code] = {
            "buy_date": row.get("entry_date"),
            "buy_price": row.get("entry_price"),
            "weight": row.get("weight", 1.0 / DEFAULT_MAX_STOCKS),
            "days_held": row.get("days_held", 0),
            "ts_code": row.get("ts_code", code),
            "signal_id": row.get("signal_id"),
            "score": row.get("score", 0),
        }
    return holdings if holdings else None


def persist_predictions_for_date(target_date, df_pred):
    """
    保存当日真实推理结果到 predictions/YYYY-MM-DD.parquet (append-only，按天分文件)。

    Input:
        target_date: 目标日期 (datetime 或 str)
        df_pred:     候选池 DataFrame (含 obs_date, ts_code, signal_id, obs_day, pred_*, score)
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    day_path = os.path.join(PREDICTIONS_DIR, f"{target_date.strftime('%Y-%m-%d')}.parquet")

    if os.path.exists(day_path):
        print(f"  [警告] 预测文件已存在，跳过覆盖: {day_path}")
        return

    # 补充必要元字段
    df_save = df_pred.copy()
    if "trade_date" not in df_save.columns:
        df_save["trade_date"] = target_date
    if "model_version" not in df_save.columns:
        df_save["model_version"] = V1_BASELINE
    if "generated_at" not in df_save.columns:
        df_save["generated_at"] = datetime.now()

    # 只保存核心字段，减少存储
    keep_cols = [
        "trade_date", "obs_date", "ts_code", "signal_id", "obs_day",
        "pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls",
        "score", "model_version", "generated_at",
    ]
    available_cols = [c for c in keep_cols if c in df_save.columns]
    df_save = df_save[available_cols].copy()
    df_save.to_parquet(day_path, index=False)
    print(f"  [保存预测] {day_path} ({len(df_save)} 行)")


# ==================== 工具函数 ====================


def _annotate_tier_for_report(df):
    """
    对 candidates_df 标注 Tier A/B/C（SSOT 规则）。
    Input:  DataFrame 含 score 和 pred_buy_cls 列
    Output: DataFrame 新增 tier 列
    """
    df = df.copy()
    df["tier"] = "C"
    mask_a = (df["score"] > TIER_A_SCORE) & (df["pred_buy_cls"] < TIER_A_BUY_CLS)
    df.loc[mask_a, "tier"] = "A"
    mask_b = (df["score"] > TIER_B_SCORE) & (df["pred_buy_cls"] < TIER_B_BUY_CLS) & (~mask_a)
    df.loc[mask_b, "tier"] = "B"
    return df


def _compute_weights_for_buys(pending_buys, tier_weights):
    """
    对 pending_buys 计算归一化权重。
    Input:  [(code, bp, ts, score, sid, tier), ...]
    Output: [(code, bp, ts, score, sid, tier, weight), ...]
    """
    if not pending_buys:
        return []
    raw = [tier_weights.get(tier, 1.0) for _, _, _, _, _, tier in pending_buys]
    total = sum(raw)
    result = []
    for i, (code, bp, ts, score, sid, tier) in enumerate(pending_buys):
        w = raw[i] / total if total > 0 else 1.0 / len(pending_buys)
        result.append((code, bp, ts, score, sid, tier, w))
    return result


def _escape_md(text):
    """转义飞书 markdown 特殊字符。"""
    if not isinstance(text, str):
        text = str(text)
    return text.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_") \
               .replace("~", "\\~").replace(">", "\\>").replace("#", "\\#")


def format_feishu_card(target_date, holdings_before, sells, sell_reasons,
                       pending_buys_with_tier, candidates, top30_df,
                       prev_date_decision, pred_indexed_for_report, name_map=None):
    """
    将决策结果格式化为飞书卡片，参考 advicement.txt 的 5 段结构。
    精简范围：持仓（全部）+ 候选 TOP10 + 新买入清单。

    Input:
        target_date:              目标决策日
        holdings_before:           决策前持仓 {code: {buy_date, buy_price, days_held, score, ts_code, ...}}
        sells:                     卖出 ts_code 列表
        sell_reasons:              {ts_code: reason}
        pending_buys_with_tier:    [(code, bp, ts, score, sid, tier), ...]
        candidates:                候选池 DataFrame (含 ts_code, score, pred_buy_cls, tier, obs_day)
        top30_df:                  近3日 Top30 DataFrame
        prev_date_decision:        模型退出参考日
        pred_indexed_for_report:   {(int(sid), dt): pred_dict}
        name_map:                  {ts_code: 股票名称}

    Output:
        (header_title, header_template, elements_list)
    """
    stop_loss = V1_PARAMS.get("stop_loss", -0.07)
    exit_threshold = V1_PARAMS.get("buy_cls_exit_threshold", 0.70)
    max_hold_days = V1_PARAMS.get("max_hold_days", 20)

    elements = []
    held_codes = set(holdings_before.keys())
    sold_codes = set(s[:6] if "." in s else s for s in sells)
    dstr = target_date.strftime("%Y-%m-%d")

    # ---- 概览区 ----
    n_holdings = len(holdings_before)
    n_sells = len(sells)
    n_top30 = len(top30_df) if top30_df is not None else 0
    n_top10 = min(10, len(candidates)) if not candidates.empty else 0

    risk_codes = set()

    overview_parts = [f"持仓{n_holdings}"]
    overview_parts.append(f"卖出{n_sells}")

    for code, h in holdings_before.items():
        dh = h.get("days_held", 0) + 1
        cr = h.get("cur_ret")
        sid = h.get("signal_id")
        ts = h.get("ts_code", code)
        ts_full = ts if "." in ts else f"{ts}.??"

        hit = False
        if dh > max_hold_days:
            hit = True
        elif cr is not None and cr < stop_loss:
            hit = True
        elif sid is not None and dh > 1 and prev_date_decision is not None:
            pred = pred_indexed_for_report.get((int(sid), prev_date_decision))
            if pred is not None:
                bc_val = pred.get("pred_buy_cls", np.nan)
                if not np.isnan(bc_val) and bc_val > exit_threshold:
                    hit = True
        if hit:
            risk_codes.add(code)

    overview_parts.append(f"风险触发{len(risk_codes)}")
    overview_parts.append(f"Top30池{n_top30}")
    overview_parts.append(f"Top10候选{n_top10}")

    overview_line = "  |  ".join(overview_parts)
    rules = f"止损{stop_loss:.0%}  |  最大持有{max_hold_days}天  |  退出阈值 pred\\_buy\\_cls > {exit_threshold:.2f}"

    elements.append({"tag": "markdown", "content": f"**{overview_line}**"})
    elements.append({"tag": "markdown", "content": f"*{rules}*"})
    elements.append({"tag": "hr"})

    # ---- 构建 Top30/Top10 查找表 ----
    top30_map = {}
    if top30_df is not None and not top30_df.empty:
        for _, r in top30_df.iterrows():
            tc = str(r["ts_code"])
            top30_map[tc] = {
                "appear_count": int(r.get("appear_count", 0)),
                "best_rank": int(r.get("best_rank", 0)),
                "latest_rank": int(r.get("latest_rank", 0)),
            }

    # ---- 持仓分类 ----
    risk_holdings = []
    safe_holdings = []
    missing_holdings = []

    for code, h in holdings_before.items():
        ts = h.get("ts_code", code)
        dh = h.get("days_held", 0) + 1
        cr = h.get("cur_ret")
        sid = h.get("signal_id")
        sc = h.get("score", 0)
        bp = h.get("buy_price", 0)
        name = name_map.get(ts, "") if name_map else ""

        bc_val = np.nan
        model_avail = sid is not None and dh > 1 and prev_date_decision is not None
        if model_avail:
            pred = pred_indexed_for_report.get((int(sid), prev_date_decision))
            if pred is not None:
                bc_val = pred.get("pred_buy_cls", np.nan)

        missing = cr is None or (model_avail and np.isnan(bc_val))

        if code in risk_codes:
            risk_holdings.append((code, ts, name, dh, sc, bc_val, cr, bp, sid))
        elif missing:
            missing_holdings.append((code, ts, name, dh, sc, bc_val, cr, bp, sid))
        else:
            safe_holdings.append((code, ts, name, dh, sc, bc_val, cr, bp, sid))

    def _fmt_stock_lines(items, section_label, icon):
        if not items:
            return
        elements.append({"tag": "markdown",
                         "content": f"**{icon}  {section_label}**  ({len(items)}只)"})
        for idx, (code, ts, name, dh, sc, bc_val, cr, bp, sid) in enumerate(items, 1):
            lines = []
            label = f"**{idx}) {name}**" if name else f"**{idx}) {code}**"
            lines.append(f"{label}  `{_escape_md(ts)}`")

            bd = ""
            for _c, _h in holdings_before.items():
                if _c == code:
                    _bd = _h.get("buy_date", "")
                    if hasattr(_bd, "strftime"):
                        bd = _bd.strftime("%Y-%m-%d")
                    elif isinstance(_bd, pd.Timestamp):
                        bd = _bd.strftime("%Y-%m-%d")
                    else:
                        bd = str(_bd)[:10]
                    break
            lines.append(f"买入 {bd}  |  买入价 {bp:.2f}  |  持仓 {dh}天  |  entry\\_score {sc:.4f}")

            tm = top30_map.get(ts, {})
            appear = tm.get("appear_count", 0)
            best_r = tm.get("best_rank", 0)
            latest_r = tm.get("latest_rank", 0)
            in_top30 = "是" if appear > 0 else "否"
            in_top10 = "是" if latest_r > 0 and latest_r <= 10 else "否"
            lines.append(f"Top30: {in_top30}({appear}次)  |  Top10: {in_top10}(排{latest_r})")

            cr_str = f"{cr:+.2%}" if cr is not None else "N/A"
            bc_str = f"{bc_val:.4f}" if not np.isnan(bc_val) else "N/A"
            lines.append(f"score {sc:.4f}  |  buy\\_cls {bc_str}  |  cur\\_ret {cr_str}")

            if cr is None:
                lines.append(f"\\> cur\\_ret 无收盘价属正常 (开盘前决策)")
            elif sid is not None and np.isnan(bc_val):
                lines.append(f"\\> ⚠️ pred\\_buy\\_cls 缺失，需关注")

            elements.append({"tag": "markdown", "content": "\n".join(lines)})

    # 一、持仓-风险触发
    def _risk_label(code, dh, cr):
        parts = []
        if dh > max_hold_days:
            parts.append("超期")
        if cr is not None and cr < stop_loss:
            parts.append("止损")
        ts = holdings_before.get(code, {}).get("ts_code", code)
        ts_full = ts if "." in ts else f"{ts}.??"
        if ts_full in sell_reasons:
            reason = sell_reasons[ts_full]
            if reason == "model_risk":
                parts.append("模型风险")
        return ", ".join(parts) if parts else "退出"

    _fmt_stock_lines(
        [(code, ts, name, dh, sc, bc_val, cr, bp, sid)
         for code, ts, name, dh, sc, bc_val, cr, bp, sid in risk_holdings],
        "一、持仓-风险触发", "🔴",
    )
    for code, ts, name, dh, sc, bc_val, cr, bp, sid in risk_holdings:
        label = _risk_label(code, dh, cr)
        elements.append({"tag": "markdown",
                         "content": f"\\> 依据: {label}"})

    elements.append({"tag": "hr"})

    # 二、持仓-安全
    _fmt_stock_lines(safe_holdings, "二、持仓-安全", "🟢")
    elements.append({"tag": "hr"})

    # 三、持仓-待数据更新
    _fmt_stock_lines(missing_holdings, "三、持仓-待数据更新", "⚠️")
    elements.append({"tag": "hr"})

    # 四、候选-高优先级（只取 TOP10）
    if not candidates.empty:
        score_median = candidates["score"].median() if "score" in candidates.columns else 0
        all_cand = []
        for _, r in candidates.iterrows():
            tc = str(r.get("ts_code", ""))
            code = tc[:6] if "." in tc else tc
            if code in held_codes:
                continue
            sc = r.get("score", 0)
            bc = r.get("pred_buy_cls", np.nan)
            tier = r.get("tier", "C")
            od = r.get("obs_day", np.nan)
            is_high = sc > score_median and (np.isnan(bc) or bc < exit_threshold)
            all_cand.append((code, tc, sc, bc, tier, od, is_high))

        high_cand = [(code, tc, sc, bc, tier, od) for code, tc, sc, bc, tier, od, _ in all_cand if _]
        high_cand = high_cand[:10]

        if high_cand:
            elements.append({"tag": "markdown",
                             "content": f"**📥  四、候选-高优先级**  ({len(high_cand)}只)"})
            for idx, (code, tc, sc, bc, tier, od) in enumerate(high_cand, 1):
                lines = []
                bc_str = f"{bc:.4f}" if not np.isnan(bc) else "N/A"
                od_str = str(int(od)) if not np.isnan(od) else "?"
                name = name_map.get(tc, "") if name_map else ""
                label = f"**{idx}) {name}**" if name else f"**{idx}) {code}**"
                lines.append(f"{label}  `{_escape_md(tc)}`  `{tier}`")
                lines.append(f"score {sc:.4f}  |  buy\\_cls {bc_str}  |  obs\\_day {od_str}")

                tm = top30_map.get(tc, {})
                appear = tm.get("appear_count", 0)
                if appear > 0:
                    lines.append(f"Top30: 出现 {appear} 次, 最优排{tm.get('best_rank', 0)}")

                if bc > exit_threshold:
                    lines.append(f"\\> ⚠️ buy\\_cls 偏高({bc:.4f} > {exit_threshold:.2f})，风险票")

                elements.append({"tag": "markdown", "content": "\n".join(lines)})

            elements.append({"tag": "hr"})

    # ---- 新买入清单 ----
    if pending_buys_with_tier:
        elements.append({"tag": "markdown",
                         "content": f"**📤 新买入清单  ({len(pending_buys_with_tier)}只)**"})
        n_avail = DEFAULT_MAX_STOCKS - len(held_codes) + len(sold_codes)
        elements.append({"tag": "markdown",
                         "content": f"可用仓位: {max(0, n_avail)}/{DEFAULT_MAX_STOCKS}"})
        for idx, (code, bp_val, ts_code_val, sc_val, sid_val, tier) in enumerate(pending_buys_with_tier, 1):
            name = name_map.get(str(ts_code_val), "") if name_map else ""
            label = f"**{idx}) {name}**" if name else f"**{idx}) {code}**"
            elements.append({"tag": "markdown",
                             "content": f"{label} `{_escape_md(ts_code_val)}` `{tier}`  score {sc_val:.4f}"})

    # 去掉末尾多余的 hr
    while elements and elements[-1].get("tag") == "hr":
        elements.pop()

    header_title = f"SLC 策略证据面板 | {dstr}"
    header_template = "blue"

    # 根据风险触发数量调整颜色
    if len(risk_holdings) > 0:
        header_template = "orange"
    if len(sells) > 0:
        header_template = "red"
    if len(safe_holdings) == n_holdings:
        header_template = "green"

    return header_title, header_template, elements


def send_feishu_report(header_title, header_template, elements):
    """
    将格式化后的飞书卡片发送到用户手机。

    Input:
        header_title:    卡片标题
        header_template: 卡片颜色主题 (blue/green/red/orange)
        elements:        卡片元素列表

    Output:
        bool: 发送是否成功
    """
    try:
        from app.feishu_notifier import FeishuNotifier
        notifier = FeishuNotifier()
        result = notifier.send_card(
            header_title=header_title,
            elements=elements,
            header_template=header_template,
        )
        return result.get("code") == 0
    except Exception as e:
        print(f"\n❌ 飞书消息发送失败: {e}")
        return False


def _fetch_stock_names(ts_codes):
    """
    批量查询 ts_code → 股票名称映射。
    从 stock_pools 表查询，失败返回空 dict。
    """
    if not ts_codes:
        return {}
    try:
        from datasource.database import get_engine
        from sqlalchemy import text
        engine = get_engine()
        sql = text("SELECT ts_code, name FROM stock_pools WHERE ts_code = ANY(:codes)")
        with engine.connect() as conn:
            result = conn.execute(sql, {"codes": list(ts_codes)})
            return {row[0]: row[1] for row in result}
    except Exception:
        return {}


def _build_name_map(ts_codes_set):
    """收集所有涉及的 ts_code，构建名称映射。"""
    ts_codes = [tc for tc in ts_codes_set if tc]
    return _fetch_stock_names(ts_codes)


def _resolve_trading_context(target_date, trading_days):
    """
    在 trading_days 中定位 target_date，返回前一日和当日索引。
    若 target_date 超出 trading_days 范围，返回最近的可用上下文。

    Output:
        target_idx:   target_date 在 trading_days 中的索引 (None=不在列表中)
        prev_date:    前一个交易日 (None=无)
        effective_date: 实际用于决策的参考日 (若 target_date 不在列表则用最后一天)
    """
    sorted_td = sorted(trading_days)
    target_idx = None
    for i, td in enumerate(sorted_td):
        if td == target_date:
            target_idx = i
            break

    if target_idx is not None:
        prev_date = sorted_td[target_idx - 1] if target_idx > 0 else None
        return target_idx, prev_date, target_date

    after_last = all(target_date > td for td in sorted_td)
    if after_last:
        effective_date = sorted_td[-1]
        prev_date = sorted_td[-1] if len(sorted_td) >= 1 else None
        return None, prev_date, effective_date

    before_first = all(target_date < td for td in sorted_td)
    if before_first:
        return None, None, sorted_td[0]

    return None, None, target_date


# ==================== 数据加载 ====================


def _check_and_update_full_test_predictions(target_date):
    """
    检测 full_test_predictions.parquet 是否过期，如果过期则尝试更新。

    更新逻辑：
    - 如果 target_date <= obs_date_max：数据充足，无需更新
    - 如果 target_date > obs_date_max：数据过期，检查 candidate_with_scores.parquet
    - 如果 candidate_with_scores.parquet 有更新：自动运行 generate_full_predictions.py
    - 如果都没有更新：提示用户需要运行上游流水线

    Input:
        target_date: 目标决策日（T日）

    Output:
        bool: True=已更新（需要重新加载），False=无需更新
    """
    from stop_experiment.pipeline.stop_config import OUTPUT_DIR

    full_test_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    candidate_path = os.path.join(OUTPUT_DIR, "candidate_with_scores.parquet")

    # ① 检查 full_test_predictions.parquet 是否存在
    if not os.path.exists(full_test_path):
        print(f"  [数据更新] full_test_predictions.parquet 不存在，开始生成...")
        return _run_generate_full_predictions()

    # ② 检查 full_test_predictions.parquet 的日期范围
    df_full = pd.read_parquet(full_test_path, columns=['obs_date'])
    df_full['obs_date'] = pd.to_datetime(df_full['obs_date'])
    obs_date_max = df_full['obs_date'].max()

    # 如果 target_date <= obs_date_max，数据充足，无需更新
    if target_date <= obs_date_max:
        print(f"  [数据检查] 数据充足 (full_test_predictions 最多到 {obs_date_max.strftime('%Y-%m-%d')}, 决策日={target_date.strftime('%Y-%m-%d')})")
        return False

    # ③ 数据过期，检查 candidate_with_scores.parquet 是否有更新
    print(f"  [数据检查] 数据过期 (full_test_predictions 最多到 {obs_date_max.strftime('%Y-%m-%d')}, 决策日={target_date.strftime('%Y-%m-%d')})")

    if not os.path.exists(candidate_path):
        print(f"  [警告] candidate_with_scores.parquet 不存在，无法自动更新")
        print(f"  [提示] 请运行上游流水线获取新数据: python stop_experiment/pipeline/01_selection_dsa.py")
        return False

    # ④ 检查 candidate_with_scores.parquet 的日期范围
    df_cand = pd.read_parquet(candidate_path, columns=['obs_date'])
    df_cand['obs_date'] = pd.to_datetime(df_cand['obs_date'])
    cand_date_max = df_cand['obs_date'].max()

    if cand_date_max <= obs_date_max:
        print(f"  [警告] candidate_with_scores.parquet 也没有更新数据 (最多到 {cand_date_max.strftime('%Y-%m-%d')})")
        print(f"  [提示] 请运行上游流水线获取新数据: python stop_experiment/pipeline/01_selection_dsa.py")
        return False

    # ⑤ 有更新数据，运行 generate_full_predictions.py
    print(f"  [数据更新] 发现新数据 (candidate_with_scores 最多到 {cand_date_max.strftime('%Y-%m-%d')})，重新生成 full_test_predictions.parquet...")
    return _run_generate_full_predictions()


def _run_generate_full_predictions():
    """
    运行 generate_full_predictions.py 重新生成 full_test_predictions.parquet。

    Output:
        bool: True=成功，False=失败
    """
    import subprocess

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backtest", "generate_full_predictions.py")
    script_path = os.path.abspath(script_path)

    if not os.path.exists(script_path):
        print(f"  [错误] 脚本不存在: {script_path}")
        return False

    try:
        print(f"  [执行] python {script_path}")
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )

        if result.returncode == 0:
            print(f"  [成功] generate_full_predictions.py 执行完成")
            if result.stdout:
                print(result.stdout[-500:])  # 打印最后500字符
            return True
        else:
            print(f"  [失败] generate_full_predictions.py 返回码: {result.returncode}")
            if result.stderr:
                print(result.stderr[-500:])
            return False

    except subprocess.TimeoutExpired:
        print(f"  [失败] generate_full_predictions.py 超时（>300秒）")
        return False
    except Exception as e:
        print(f"  [失败] 运行 generate_full_predictions.py 时发生异常: {e}")
        return False


def _load_engine_data(target_date=None):
    """
    加载回测引擎所需的全量数据，运行回测获取 snapshots。

    如果提供 target_date，会先检测数据是否过期，过期则自动更新。

    Input:
        target_date: 目标决策日（T日），用于检测数据是否过期

    Output:
        df_all:         全量候选 (含 score 列)
        score_col:      分数字段名 ("score")
        price_pivot:    价格宽表
        trading_days:   交易日列表
        pred_indexed:   {(int(sid), dt): pred_dict}
        snapshots_map:  {date: snapshot_dict}
    """
    # ① 检测并更新数据（如果提供了 target_date）
    if target_date is not None:
        print("\n[数据更新检查]")
        _check_and_update_full_test_predictions(target_date)

    # ② 加载数据
    cand_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    df_all = pd.read_parquet(cand_path)
    df_all["obs_date"] = pd.to_datetime(df_all["obs_date"])

    candidate_obs_days = V1_PARAMS.get("candidate_obs_days", [1, 2, 3])
    df_all = df_all[df_all["obs_day"].isin(candidate_obs_days)].copy()

    df_all = score_stocks(df_all, V1_PARAMS.get("strategy_default", "sell_score"))
    score_col = "score"

    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data(
        candidate_obs_days=V1_PARAMS["candidate_obs_days"]
    )

    bt_result = run_backtest(
        test_df, price_pivot, trading_days, prev_close_map, pred_lookup,
        max_stocks=DEFAULT_MAX_STOCKS,
        strategy=V1_PARAMS.get("strategy_default", "sell_score"),
        exit_mode="model_exit",
        stop_loss=V1_PARAMS["stop_loss"],
        buy_cls_exit_threshold=V1_PARAMS["buy_cls_exit_threshold"],
        debug_snapshots=True,
        strict=True,
    )

    snapshots_map = {s["date"]: s for s in bt_result["snapshots"]}
    pred_indexed = {}
    for (sid, dt), pred in pred_lookup.items():
        pred_indexed[(int(sid), dt)] = pred

    return df_all, score_col, price_pivot, trading_days, pred_indexed, snapshots_map


def _ensure_pred_coverage(target_date, pred_indexed, trading_days):
    """
    确保 target_date 前一日 (prev_date) 的 pred_lookup 有覆盖。

    若无覆盖则抛出错误，要求用户先更新数据（不再使用合成逻辑）。

    Input:
        target_date:   目标决策日（T日）
        pred_indexed:  {(int(sid), dt): pred_dict}
        trading_days:  交易日列表

    Output:
        pred_indexed:  可能已扩充
        prev_date:     前一个交易日
        synthesized:   是否发生了合成（现在总是 False）

    Raises:
        ValueError: 数据过期，要求用户先更新数据
    """
    target_idx, prev_date, _ = _resolve_trading_context(target_date, trading_days)

    if prev_date is None:
        return pred_indexed, None, False

    has_coverage = any(dt == prev_date for (_, dt) in pred_indexed.keys())
    if has_coverage:
        return pred_indexed, prev_date, False

    # 数据过期，不再合成，直接报错
    all_dates = sorted(set(dt for (_, dt) in pred_indexed.keys()))
    latest_date = all_dates[-1] if all_dates else "未知"

    raise ValueError(
        f"pred_lookup 数据过期：最新数据到 {latest_date}，"
        f"但目标决策日 {target_date.strftime('%Y-%m-%d')} 的前一日 {prev_date.strftime('%Y-%m-%d')} 无覆盖。"
        f"请先运行上游流水线更新数据："
        f"python stop_experiment/pipeline/01_selection_dsa.py"
    )


def _build_candidate_pool(target_date, df_all_raw, pred_indexed, trading_days):
    """
    构建 target_date 候选池: 优先读取真实预测，无则报错。

    优先级:
    1. 读取 predictions/target_date.parquet (真实预测，与回测 signal_by_date 对齐)
    2. 直接命中: target_date 在 full_test_predictions.parquet 范围内
    3. 回退命中: prev_date 在 full_test_predictions.parquet 范围内
    4. 数据过期: 抛出错误，要求用户先更新数据

    注意：合成逻辑已禁用，因为会导致不同日期的 top10 完全相同（bug）。
    如需预测未来日期，请先运行上游流水线更新数据。

    Input:
        target_date:   目标决策日（T日）
        df_all_raw:    全量候选 (含 pred_sell_reg, obs_date, obs_day, ts_code)
        pred_indexed:  预测查找表
        trading_days:  交易日列表

    Output:
        candidates:    去重排序后的候选池 DataFrame (obs_date=target_date)
        prev_date:     前一个交易日

    Raises:
        ValueError: 数据过期，要求用户先更新数据
    """
    target_idx, prev_date, effective_date = _resolve_trading_context(target_date, trading_days)

    if prev_date is None:
        raise ValueError(f"无法确定候选池前一日: target={target_date}")

    # ① 优先读取真实预测结果（文件名 = obs_date，与回测 signal_by_date[tdate] 一致）
    real_pred = load_real_predictions_for_date(target_date)
    if real_pred is not None and not real_pred.empty:
        if "tier" not in real_pred.columns:
            real_pred = _annotate_tier_for_report(real_pred)
        return real_pred, prev_date

    # ② 无真实预测 → 报错，不再回退到 full_test_predictions.parquet
    #    依 advicement.txt 建议：生产链路必须显式失败，不允许静默替代
    raise ValueError(
        f"NO_REAL_PREDICTION_DATA: predictions/{target_date.strftime('%Y-%m-%d')}.parquet 不存在。"
        f"请先运行 python stop_experiment/pipeline/07_generate_daily_predictions.py "
        f"--date {target_date.strftime('%Y-%m-%d')} 生成当日预测，"
        f"或运行 python stop_experiment/backtest/generate_full_predictions.py 生成全量历史预测。"
    )


def _get_start_holdings(target_date, snapshots_map, trading_days, price_pivot, pred_indexed):
    """
    获取 target_date 决策前的持仓状态。

    优先级:
    1. holdings/prev_date.parquet（账本驱动，真实持仓历史）
    2. snapshots_map[prev_date]（回测快照，deprecated，兼容旧数据）
    3. 手动推进（从最新 holdings/snapshot 推进到 prev_date）

    Input:
        target_date:    目标决策日（T日）
        snapshots_map:  {date: snapshot}（deprecated 兼容）
        trading_days:   交易日列表
        price_pivot:    价格宽表
        pred_indexed:   预测查找表

    Output:
        holdings:         决策前持仓 dict {code: {...}}
        pending_buys:     待执行的买单（T-1日决策、T日开盘执行）
        pending_sells:     待执行的卖单（T-1日决策、T日开盘执行）
        prev_date:        前一个交易日
    """
    target_idx, prev_date, effective_date = _resolve_trading_context(target_date, trading_days)

    if prev_date is None:
        print(f"  [持仓] 无法确定前一日: target={target_date}")
        return {}, [], [], None

    # ① 优先从 holdings 账本恢复
    from_snapshots = load_holdings(prev_date)
    if from_snapshots is not None:
        print(f"  [持仓] 来源: holdings/{prev_date.strftime('%Y-%m-%d')}.parquet, "
              f"持仓 {len(from_snapshots)} 只")
        return from_snapshots, [], [], prev_date

    # ② 回退到 snapshots_map（deprecated）
    if prev_date in snapshots_map:
        print(f"  [持仓] 来源: {prev_date.strftime('%Y-%m-%d')} snapshot (deprecated)")
        snap = snapshots_map[prev_date]
        holdings = {k: dict(v) for k, v in snap.get("holdings_after", {}).items()}
        buys = snap.get("buys", [])
        pending_buys = [(b["code"], b["buy_price"], b["ts_code"], b["score"], b["signal_id"]) for b in buys]
        to_sell_codes = snap.get("to_sell", [])
        sell_reasons_map = snap.get("sell_reasons", {})
        pending_sells = []
        for code in to_sell_codes:
            holding = holdings.get(code, {})
            pending_sells.append({
                "code": code,
                "holding": dict(holding),
                "reason": sell_reasons_map.get(holding.get("ts_code", code), ""),
            })
        return holdings, pending_buys, pending_sells, prev_date

    # ③ 手动推进（从最新 holdings 或 snapshot）
    latest_holdings = None
    # 先尝试从最新的 holdings 文件恢复
    for d in sorted(trading_days, reverse=True):
        h = load_holdings(d)
        if h is not None:
            latest_holdings = h
            latest_snap_date = d
            break

    if latest_holdings is None and snapshots_map:
        # 回退到最新 snapshot
        latest_snap_date = max(snapshots_map.keys())
        snap = snapshots_map[latest_snap_date]
        latest_holdings = {k: dict(v) for k, v in snap.get("holdings_after", {}).items()}
        print(f"  [持仓] 推进起点: {latest_snap_date.strftime('%Y-%m-%d')} snapshot (deprecated)")
    elif latest_holdings is None:
        print(f"  [持仓] 无可用持仓数据: target={target_date}")
        return {}, [], [], prev_date
    else:
        print(f"  [持仓] 推进起点: holdings/{latest_snap_date.strftime('%Y-%m-%d')}.parquet")

    holdings = latest_holdings
    pending_from_snap = []
    pending_sells_from_snap = []

    advance_steps = sorted([d for d in trading_days if latest_snap_date < d <= prev_date])
    print(f"  [持仓] 推进 {len(advance_steps)} 天到 {prev_date.strftime('%Y-%m-%d')}")

    for step_date in advance_steps:
        for code, bp_val, ts_code_val, sc_val, sid_val in pending_from_snap:
            holdings[code] = {
                "buy_date": step_date, "buy_price": bp_val,
                "weight": 1.0 / DEFAULT_MAX_STOCKS,
                "days_held": 0, "ts_code": ts_code_val,
                "score": sc_val, "signal_id": sid_val,
            }
        pending_from_snap = []

        for sell_item in pending_sells_from_snap:
            code = sell_item["code"]
            if code in holdings:
                del holdings[code]
        pending_sells_from_snap = []

        step_idx, step_prev_date, _ = _resolve_trading_context(step_date, trading_days)

        for code, h in holdings.items():
            h["cur_ret"] = None
            if step_date in price_pivot.index:
                close_s = price_pivot.loc[step_date, "close"]
                if code in close_s.index:
                    close_p = close_s[code]
                    if pd.notna(close_p) and h.get("buy_price") and h["buy_price"] > 0:
                        h["cur_ret"] = (close_p - h["buy_price"]) / h["buy_price"]

        with redirect_stdout(StringIO()):
            holdings, pending_from_snap, pending_sells_from_snap, sell_reasons = decide_daily_actions(
                step_date, holdings, pd.DataFrame(), pred_indexed, step_prev_date,
                price_pivot=price_pivot, trading_dates=trading_days,
            )

    print(f"  [持仓] 推进后: 持仓 {len(holdings)} 只, pending_buys {len(pending_from_snap)} 只, "
          f"pending_sells {len(pending_sells_from_snap)} 只")
    return holdings, pending_from_snap, pending_sells_from_snap, prev_date


def _build_3day_top30(target_date, df_all, candidates_current, trading_days):
    """
    构建近 3 个交易日 Top10 合并去重后的准持仓池。
    模拟盘最可能持有的股票集合 (≤30 只)。

    Input:
        target_date:        目标决策日
        df_all:             全量候选 (含 obs_date, ts_code, score, pred_sell_reg, pred_buy_cls 等)
        candidates_current: 当日候选池 (obs_date=prev_date)
        trading_days:       交易日列表

    Output:
        top30_df:           DataFrame，含 ts_code / first_seen / last_seen / appear_count / best_rank
                            / latest_rank / latest_obs_day / latest_score / latest_buy_cls / 持仓状态
        day_top10s:         {date: top10_df} 每天 Top10 详情
    """
    sorted_td = sorted(trading_days)
    target_idx, prev_date, _ = _resolve_trading_context(target_date, trading_days)

    lookback = 3
    pool_dates = []
    for td in sorted_td:
        if td <= prev_date and td > (prev_date - pd.Timedelta(days=lookback * 2)):
            pool_dates.append(td)
    pool_dates = sorted(pool_dates)[-lookback:]

    if not pool_dates:
        return pd.DataFrame(), {}

    day_top10s = {}
    all_candidates = []
    seen = {}

    for d in pool_dates:
        if d == prev_date and candidates_current is not None and not candidates_current.empty:
            top10_built = candidates_current.head(10).copy()
        else:
            top10_built = build_daily_candidate_snapshot(d, df_all, score_col="score")
            if top10_built.empty:
                continue
            top10_built = top10_built.head(10).copy()

        top10_built["date"] = d
        day_top10s[d] = top10_built

        for rank, (_, row) in enumerate(top10_built.iterrows(), 1):
            tc = str(row.get("ts_code", ""))
            if not tc:
                continue
            sc = row.get("score", 0)
            bc = row.get("pred_buy_cls", np.nan)
            od = row.get("obs_day", np.nan)

            if tc not in seen:
                seen[tc] = {
                    "ts_code": tc,
                    "first_seen": d,
                    "last_seen": d,
                    "appear_count": 1,
                    "best_rank": rank,
                    "latest_rank": rank,
                    "latest_obs_day": od,
                    "latest_score": sc,
                    "latest_buy_cls": bc,
                    "first_date": d,
                }
            else:
                prev_entry = seen[tc]
                prev_entry["last_seen"] = d
                prev_entry["appear_count"] += 1
                prev_entry["best_rank"] = min(prev_entry["best_rank"], rank)
                if d > prev_entry.get("first_date", d):
                    prev_entry["latest_rank"] = rank
                    prev_entry["latest_obs_day"] = od
                    prev_entry["latest_score"] = sc
                    prev_entry["latest_buy_cls"] = bc

    if not seen:
        return pd.DataFrame(), day_top10s

    df_top30 = pd.DataFrame(list(seen.values()))
    df_top30["appear_count"] = df_top30["appear_count"].astype(int)
    df_top30["best_rank"] = df_top30["best_rank"].astype(int)
    df_top30["latest_rank"] = df_top30["latest_rank"].astype(int)
    df_top30 = df_top30.sort_values(["appear_count", "best_rank"], ascending=[False, True])

    return df_top30.reset_index(drop=True), day_top10s


# ==================== 报告输出 ====================


def _print_section_1_params():
    print()
    print("=" * 75)
    print("  Section 1: 策略参数摘要")
    print("=" * 75)
    print(f"  版本基线:      v1_frozen_202605")
    print(f"  退出模式:      model_exit (buy_cls + stop_loss + max_hold)")
    print(f"  止损阈值:      {V1_PARAMS['stop_loss']:.0%}")
    print(f"  最大持有天数:  {V1_PARAMS['max_hold_days']} 天")
    print(f"  模型退出阈值:  pred_buy_cls > {V1_PARAMS['buy_cls_exit_threshold']:.2f}")
    print(f"  候选观测日:    obs_day in {V1_PARAMS['candidate_obs_days']}")
    print(f"  最大持仓数:    {DEFAULT_MAX_STOCKS} 只")
    print(f"  排序策略:      {V1_PARAMS.get('strategy_default', 'sell_score')} (score = pred_sell_reg)")
    print(f"  买入信号阈值:  {V1_PARAMS.get('buy_signal_threshold', -0.07):.0%}")


def _print_section_2_holdings(holdings, prev_date):
    print()
    print("=" * 75)
    print(f"  Section 2: 前一日 ({prev_date.strftime('%Y-%m-%d') if prev_date else 'N/A'}) 收盘持仓快照")
    print("=" * 75)

    if not holdings:
        print("  (空仓)")
        return

    rows = []
    for code, h in holdings.items():
        bd = h.get("buy_date", "")
        if hasattr(bd, "strftime"):
            bd = bd.strftime("%Y-%m-%d")
        elif isinstance(bd, pd.Timestamp):
            bd = bd.strftime("%Y-%m-%d")
        rows.append({
            "code": code,
            "ts_code": h.get("ts_code", code),
            "buy_date": str(bd)[:10],
            "buy_price": h.get("buy_price", 0),
            "days_held": h.get("days_held", 0),
            "entry_score": h.get("score", 0),
            "signal_id": h.get("signal_id", ""),
        })

    df_h = pd.DataFrame(rows)
    avg_days = df_h["days_held"].mean()
    print(f"  持仓数: {len(holdings)}, 平均持有: {avg_days:.1f} 天, 等权仓位: {100/DEFAULT_MAX_STOCKS:.0f}%/只")
    print(f"  ---")
    print(f"  {'code':<8} {'ts_code':<12} {'buy_date':<12} {'buy_price':>8} {'days_held':>10} {'entry_score':>12} {'signal_id':>10}")
    print(f"  {'-'*78}")
    for _, r in df_h.iterrows():
        print(f"  {r['code']:<8} {r['ts_code']:<12} {r['buy_date']:<12} {r['buy_price']:>8.2f} {r['days_held']:>10} {r['entry_score']:>12.4f} {str(r['signal_id']):>10}")


def _print_section_2b_top30(top30_df, day_top10s, holdings, name_map=None):
    print()
    print("=" * 95)
    print(f"  Section 2B: 近3日累计 Top30 准持仓池")
    print("=" * 95)

    if top30_df.empty:
        print("  (无近3日候选数据)")
        return

    held_codes = set(holdings.keys())
    n_pool = len(top30_df)
    n_held = sum(1 for tc in top30_df["ts_code"] if (tc[:6] if "." in tc else tc) in held_codes)

    print(f"  池大小: {n_pool} 只 (其中 {n_held} 只已持仓)")
    print(f"  覆盖日: {', '.join(d.strftime('%m-%d') for d in sorted(day_top10s.keys()))}")
    print(f"  {'ts_code':<12} {'名称':<10} {'首见':>6} {'末见':>6} {'出现':>4} {'最优排':>6} {'最新排':>6} {'obd':>4} {'score':>9} {'buy_cls':>8} {'持仓'}")
    print(f"  {'-'*98}")

    for _, r in top30_df.iterrows():
        tc = str(r["ts_code"])
        code = tc[:6] if "." in tc else tc
        held = "是" if code in held_codes else ""
        name = name_map.get(tc, "") if name_map else ""
        fs = r["first_seen"].strftime("%m-%d") if hasattr(r["first_seen"], "strftime") else str(r["first_seen"])[:5]
        ls = r["last_seen"].strftime("%m-%d") if hasattr(r["last_seen"], "strftime") else str(r["last_seen"])[:5]
        bc_str = f"{r['latest_buy_cls']:.4f}" if not np.isnan(r.get("latest_buy_cls", np.nan)) else "N/A"
        od_str = str(int(r["latest_obs_day"])) if not np.isnan(r.get("latest_obs_day", np.nan)) else "?"

        print(f"  {tc:<12} {name:<10} {fs:>6} {ls:>6} {int(r['appear_count']):>4} "
              f"{int(r['best_rank']):>6} {int(r['latest_rank']):>6} {od_str:>4} "
              f"{r['latest_score']:>9.4f} {bc_str:>8} {held:>4}")

    print(f"\n  💡 出现3次 = 持续受青睐 | 只出现1次 = 新冒出 | '持仓'=已在组合 → 跳至 Section 4 看退出评估")


def _print_section_3_candidates(candidates, holdings, prev_date, name_map=None):
    print()
    print("=" * 95)
    print(f"  Section 3: 候选池 — 买入机会 TOP10  (obs_date={prev_date.strftime('%Y-%m-%d') if prev_date else 'N/A'})")
    print("=" * 95)

    if candidates.empty:
        print("  (无候选)")
        return

    held_codes = set(holdings.keys())

    if "tier" not in candidates.columns:
        candidates = _annotate_tier_for_report(candidates)

    disp_cols = ["ts_code", "obs_day"]
    extra = ["pred_sell_reg", "pred_buy_cls", "pred_buy_reg"]
    available = [c for c in extra if c in candidates.columns]

    df_disp = candidates[disp_cols + available + ["score", "tier"]].copy()
    df_disp["名称"] = df_disp["ts_code"].apply(
        lambda tc: name_map.get(str(tc), "") if name_map else ""
    )
    df_disp["已持仓"] = df_disp["ts_code"].apply(
        lambda tc: "是" if (tc[:6] if "." in str(tc) else str(tc)) in held_codes else ""
    )

    top_n = min(10, len(df_disp))
    print(f"  候选总数: {len(df_disp)}, 展示 TOP{top_n}")
    if "score" in df_disp.columns:
        print(f"  Score 分布: p25={df_disp['score'].quantile(0.25):.4f}  "
              f"p50={df_disp['score'].quantile(0.50):.4f}  "
              f"p75={df_disp['score'].quantile(0.75):.4f}")
    if "pred_buy_cls" in df_disp.columns:
        print(f"  Buy_cls 分布: p25={df_disp['pred_buy_cls'].quantile(0.25):.4f}  "
              f"p50={df_disp['pred_buy_cls'].quantile(0.50):.4f}  "
              f"p75={df_disp['pred_buy_cls'].quantile(0.75):.4f}")
    tier_counts = df_disp["tier"].value_counts().to_dict()
    print(f"  Tier 分布: A={tier_counts.get('A', 0)}  B={tier_counts.get('B', 0)}  C={tier_counts.get('C', 0)}")
    print(f"  ---")
    header = f"  {'rank':<5} {'名称':<10} {'ts_code':<12} {'obs_day':>7} {'score':>10} {'tier':>5}"
    for ac in available:
        header += f" {ac:>14}"
    header += f" {'已持仓':>6}"
    print(header)
    print(f"  {'-'*95}")

    for idx, (_, r) in enumerate(df_disp.head(top_n).iterrows()):
        line = f"  {idx+1:<5} {r['名称']:<10} {r['ts_code']:<12} {int(r['obs_day']):>7} {r['score']:>10.4f} {r.get('tier', 'C'):>5}"
        for ac in available:
            val = r.get(ac, np.nan)
            line += f" {val:>14.4f}" if not np.isnan(val) else f" {'N/A':>14}"
        line += f" {r.get('已持仓', ''):>6}"
        print(line)


def _print_section_4_exit_scan(holdings, sells, sell_reasons, prev_date, pred_indexed, name_map=None):
    print()
    print("=" * 85)
    print(f"  Section 4: 持仓风险扫描 — 退出评估详细过程")
    print(f"  prev_date (模型退出参考日): {prev_date.strftime('%Y-%m-%d') if prev_date else 'N/A'}")
    print("=" * 85)

    max_hold_days = V1_PARAMS.get("max_hold_days", 20)
    stop_loss = V1_PARAMS.get("stop_loss", -0.07)
    exit_threshold = V1_PARAMS.get("buy_cls_exit_threshold", 0.70)

    if not holdings:
        print("  (空仓，无退出评估)")
        return

    for code, h in holdings.items():
        ts = h.get("ts_code", code)
        dh = h.get("days_held", 0)
        ddh = dh + 1
        cr = h.get("cur_ret")
        sid = h.get("signal_id")
        bp = h.get("buy_price", 0)
        name = name_map.get(ts, "") if name_map else ""
        ts_full = ts if "." in ts else f"{ts}.??"
        sold = ts_full in sell_reasons
        actual_reason = sell_reasons.get(ts_full, "")

        display_label = f"{name} ({ts_full})" if name else ts_full
        print(f"\n  ── {display_label} (code={code}) ─────────────────────────────────────────────")
        bd = h.get("buy_date", "?")
        if hasattr(bd, "strftime"):
            bd = bd.strftime("%Y-%m-%d")
        elif isinstance(bd, pd.Timestamp):
            bd = bd.strftime("%Y-%m-%d")
        print(f"    买入日期: {bd}, 买入价: {bp:.2f}, 持有天数: {ddh} (当日+1)")

        check1_hit = ddh > max_hold_days
        if sold and actual_reason == "max_hold":
            print(f"    🔴 检查1 (max_hold): dd={ddh} > {max_hold_days}? → **触发退出**")
        elif check1_hit:
            print(f"    ✅ 检查1 (max_hold): dd={ddh} > {max_hold_days}? → 触发")
        else:
            print(f"    ○ 检查1 (max_hold): dd={ddh} > {max_hold_days}? → 未触发")

        if cr is None:
            if sold and actual_reason == "stop_loss":
                print(f"    🔴 检查2 (stop_loss): cur_ret=None → 无法判断但已标记")
            else:
                print(f"    ⚠ 检查2 (stop_loss): cur_ret=None → 跳过 (无收盘价)")
        elif sold and actual_reason == "stop_loss":
            print(f"    🔴 检查2 (stop_loss): cur_ret={cr:+.2%} < {stop_loss:.0%}? → **触发退出**")
        elif cr < stop_loss:
            print(f"    ✅ 检查2 (stop_loss): cur_ret={cr:+.2%} < {stop_loss:.0%}? → 触发")
        else:
            print(f"    ○ 检查2 (stop_loss): cur_ret={cr:+.2%} < {stop_loss:.0%}? → 未触发")

        bc_val = np.nan
        model_available = sid is not None and ddh > 1 and prev_date is not None
        if model_available:
            # 只使用精确匹配，不允许回退到过期预测
            # 如果 prev_date 无预测数据，说明数据过期，立即报错
            pred_exact = pred_indexed.get((int(sid), prev_date))
            if pred_exact is not None:
                bc_val = pred_exact.get("pred_buy_cls", np.nan)
            else:
                # 数据过期，抛出错误，不使用过期预测
                raise ValueError(
                    f"预测数据过期：signal_id={sid}, prev_date={prev_date}，"
                    f"但 pred_indexed 中无此日期的预测数据。"
                    f"请先运行上游流水线更新数据："
                    f"python stop_experiment/pipeline/01_selection_dsa.py"
                )

        if sold and actual_reason == "model_risk":
            label = f"🔴 检查3 (model_risk): pred_buy_cls={bc_val:.4f} > {exit_threshold:.2f}? → **触发退出**" \
                if not np.isnan(bc_val) else f"🔴 检查3 (model_risk): **触发退出** (原因: {actual_reason})"
            print(f"    {label}")
        elif not np.isnan(bc_val) and bc_val > exit_threshold:
            print(f"    ✅ 检查3 (model_risk): pred_buy_cls={bc_val:.4f} > {exit_threshold:.2f}? → 触发")
        elif not np.isnan(bc_val):
            print(f"    ○ 检查3 (model_risk): pred_buy_cls={bc_val:.4f} > {exit_threshold:.2f}? → 未触发")
        elif ddh <= 1:
            print(f"    ⚠ 检查3 (model_risk): dd={ddh} ≤ 1 → 跳过 (买入首日)")
        elif sid is None:
            print(f"    ⚠ 检查3 (model_risk): signal_id 缺失 → 跳过")
        else:
            print(f"    ⚠ 检查3 (model_risk): pred_buy_cls 缺失 (sid={sid}, prev={prev_date}) → 跳过")

        if sold:
            print(f"    >>> 最终决策: 🔴 卖出, 原因: {actual_reason}")
        else:
            print(f"    >>> 最终决策: 🟢 持有")

    if sells:
        print(f"\n  {'─'*60}")
        print(f"  退出汇总: 共卖出 {len(sells)} 只")
        for ts in sells:
            reason = sell_reasons.get(ts, "?")
            name = name_map.get(ts, "") if name_map else ""
            display = f"{name} ({ts})" if name else ts
            print(f"    - {display}: {reason}")
    else:
        print(f"\n  {'─'*60}")
        print(f"  退出汇总: 无卖出")


def _print_section_4b_actions(top30_df, holdings, sells, name_map=None):
    print()
    print("=" * 95)
    print(f"  Section 4B: 近3日 Top30 持仓动作建议")
    print(f"  规则: pred_buy_cls<0.50 → hold | 0.50~0.70 → watch | >0.70 → reduce | 非持仓Top10 → new_buy_ok")
    print("=" * 95)

    if top30_df.empty:
        print("  (无近3日候选)")
        return

    held_codes = set(holdings.keys())
    sold_codes = set()
    for s in sells:
        sc = s[:6] if "." in s else s
        sold_codes.add(sc)

    actions = []
    for _, r in top30_df.iterrows():
        tc = str(r["ts_code"])
        code = tc[:6] if "." in tc else tc
        bc = r.get("latest_buy_cls", np.nan)
        sc = r.get("latest_score", 0)
        name = name_map.get(tc, "") if name_map else ""

        if code in sold_codes:
            action, risk, note = "sold", "—", "当日已卖出"
        elif code in held_codes:
            if np.isnan(bc):
                action, risk, note = "watch", "?", "pred_buy_cls 缺失"
            elif bc > 0.70:
                action, risk, note = "reduce", "高", f"buy_cls={bc:.3f} > 0.70"
            elif bc > 0.50:
                action, risk, note = "watch", "中", f"buy_cls={bc:.3f}"
            else:
                action, risk, note = "hold", "低", f"buy_cls={bc:.3f}"
        elif r.get("latest_rank", 99) <= 10:
            action, risk, note = "new_buy_ok", "—", f"Top{r.get('latest_rank', 0):.0f}, score={sc:.3f}"
        else:
            action, risk, note = "watch", "—", "近期曾进Top10，当前不在Top10"

        actions.append({
            "ts_code": tc,
            "name": name,
            "held": "是" if code in held_codes else "",
            "sold": "是" if code in sold_codes else "",
            "buy_cls": bc,
            "score": sc,
            "action": action,
            "risk": risk,
            "note": note,
        })

    df_act = pd.DataFrame(actions)
    print(f"  {'名称':<10} {'ts_code':<12} {'持仓':>4} {'卖出':>4} {'buy_cls':>8} {'score':>9} {'动作':<12} {'风险':>4} {'备注'}")
    print(f"  {'-'*95}")
    for _, r in df_act.iterrows():
        bc_str = f"{r['buy_cls']:.4f}" if not np.isnan(r['buy_cls']) else "N/A"
        print(f"  {r['name']:<10} {r['ts_code']:<12} {r['held']:>4} {r['sold']:>4} {bc_str:>8} {r['score']:>9.4f} "
              f"{r['action']:<12} {r['risk']:>4} {r['note']}")

    n_hold = (df_act["action"] == "hold").sum()
    n_reduce = (df_act["action"] == "reduce").sum()
    n_watch = (df_act["action"] == "watch").sum()
    n_new = (df_act["action"] == "new_buy_ok").sum()
    n_sold = (df_act["action"] == "sold").sum()
    print(f"\n  汇总: hold={n_hold}, reduce={n_reduce}, watch={n_watch}, new_buy_ok={n_new}, sold={n_sold}")


def _print_section_5_trades(sells, sell_reasons, pending_buys, holdings_before, holdings_after, target_date, trading_days, name_map=None):
    print()
    print("=" * 85)
    print(f"  Section 5: 调仓决策 — 买入/卖出指令与依据")
    print("=" * 85)

    sorted_td = sorted(trading_days)
    target_idx = None
    for i, td in enumerate(sorted_td):
        if td == target_date:
            target_idx = i
            break
    next_date = sorted_td[target_idx + 1] if target_idx is not None and target_idx + 1 < len(sorted_td) else None
    if next_date is None and target_idx is None:
        next_date = target_date + pd.Timedelta(days=1)

    if sells:
        print(f"\n  📤 卖出清单 ({len(sells)} 只):")
        print(f"  {'名称':<10} {'ts_code':<12} {'原因':<12} {'说明'}")
        print(f"  {'-'*65}")
        for ts in sells:
            reason = sell_reasons.get(ts, "?")
            name = name_map.get(ts, "") if name_map else ""
            desc_map = {
                "max_hold": "持有天数超20天上限",
                "stop_loss": "累计亏损超7%止损",
                "model_risk": "pred_buy_cls > 0.70，模型判定风险",
            }
            print(f"  {name:<10} {ts:<12} {reason:<12} {desc_map.get(reason, '')}")
        print(f"  💡 卖出执行日: {target_date.strftime('%Y-%m-%d')} 开盘 (如有行情)")
    else:
        print(f"\n  📤 卖出: 无")

    print(f"\n  📥 买入清单 ({len(pending_buys)} 只):")
    if pending_buys:
        print(f"  n_avail = {DEFAULT_MAX_STOCKS} - {len(holdings_before)} (持仓) + {len(sells)} (卖出) = {len(pending_buys)}")
        print(f"  {'名称':<10} {'code':<8} {'ts_code':<12} {'score':>10} {'tier':>5} {'buy_price':>10}")
        print(f"  {'-'*60}")
        for code, bp_val, ts_code_val, sc_val, sid_val, tier in pending_buys:
            name = name_map.get(str(ts_code_val), "") if name_map else ""
            print(f"  {name:<10} {code:<8} {ts_code_val:<12} {sc_val:>10.4f} {tier:>5} {str(bp_val):>10}")
        exec_day = next_date.strftime("%Y-%m-%d") if next_date else "?"
        print(f"  💡 买入执行日: {exec_day} 开盘 (T+1)")
    else:
        print(f"  (无新买入机会或已满仓)")

    _print_section_5_tiers(pending_buys, name_map)


def _print_section_5_tiers(pending_buys, name_map=None):
    if not pending_buys:
        return
    print()
    print(f"  {'─'*60}")
    print(f"  📊 仓位分层建议 (今日新买入) — SSOT Tier 规则")
    tiers = {"A": [], "B": [], "C": []}
    for code, bp_val, ts_code_val, sc_val, sid_val, tier in pending_buys:
        name = name_map.get(str(ts_code_val), "") if name_map else ""
        label = name if name else ts_code_val
        tiers[tier].append(label)
    if tiers["A"]:
        print(f"    Tier A (score>0.2 & buy_cls<0.30): {', '.join(tiers['A'])}")
    if tiers["B"]:
        print(f"    Tier B (score>0.1 & buy_cls<0.70): {', '.join(tiers['B'])}")
    if tiers["C"]:
        print(f"    Tier C (其余):                     {', '.join(tiers['C'])}")
    print(f"    💡 分层依据: SSOT 规则 (与回测/Tier实验完全一致)")
    print(f"    💡 最终仓位由用户决策，以下提供双轨映射参考")


def _print_section_5c_dual_track(pending_w1, pending_w3, name_map=None):
    """
    Section 5C: 双轨仓位映射对比
    显示 W1 (轻分层) 和 W3 (强分层) 的权重分配对比
    """
    if not pending_w1 and not pending_w3:
        return

    print()
    print("=" * 85)
    print("  Section 5C: 双轨仓位映射对比")
    print("=" * 85)
    print(f"  {'名称':<10} {'code':<8} {'ts_code':<12} {'tier':>5} {'基线等权':>10} {'W1权重':>10} {'W3权重':>10}")
    print(f"  {'-'*75}")

    baseline_total = {"A": 0.0, "B": 0.0, "C": 0.0}
    w1_total = {"A": 0.0, "B": 0.0, "C": 0.0}
    w3_total = {"A": 0.0, "B": 0.0, "C": 0.0}

    n = len(pending_w1)
    baseline_w = 1.0 / n if n > 0 else 0.0

    for i, (code, bp, ts, score, sid, tier, w1) in enumerate(pending_w1):
        w3 = pending_w3[i][6] if i < len(pending_w3) else 0.0
        name = name_map.get(str(ts), "") if name_map else ""
        print(f"  {name:<10} {code:<8} {ts:<12} {tier:>5} {baseline_w:>10.2%} {w1:>10.2%} {w3:>10.2%}")
        baseline_total[tier] += baseline_w
        w1_total[tier] += w1
        w3_total[tier] += w3

    print(f"  {'-'*75}")
    print(f"  {'合计':<8} {'':<12} {'':>5} {'' :>10} {'' :>10} {'' :>10}")
    for tier in ["A", "B", "C"]:
        print(f"  {'':<8} {'':<12} {tier:>5} {baseline_total[tier]:>10.2%} {w1_total[tier]:>10.2%} {w3_total[tier]:>10.2%}")

    print()
    print(f"  💡 基线等权: 每只 {baseline_w:.2%} (当前默认)")
    print(f"  💡 轻分层 W1: Tier A 权重 {w1_total.get('A', 0):.2%}, Tier C 权重 {w1_total.get('C', 0):.2%} (低风险基准版)")
    print(f"  💡 强分层 W3: Tier A 权重 {w3_total.get('A', 0):.2%}, Tier C 权重 {w3_total.get('C', 0):.2%} (高收益实验版)")
    print(f"  💡 实验结论: W3 NAV +3.89% vs 基线，但回撤放大 3.5x；W1 NAV +1.34%，回撤增幅温和")


def _print_section_6_portfolio(holdings_after, pending_buys, target_date, trading_days, name_map=None):
    print()
    print("=" * 95)
    print(f"  Section 6: 调仓后投资组合 ({target_date.strftime('%Y-%m-%d')} 收盘)")
    print("=" * 95)

    n_total = len(holdings_after) + len(pending_buys)
    if n_total == 0:
        print("  (空仓)")
        return

    weight = 1.0 / n_total if n_total > 0 else 0
    weight_pct = weight * 100

    print(f"  总持仓: {n_total} 只, 等权 {weight_pct:.1f}%/只")
    print(f"  {'名称':<10} {'code':<8} {'类型':<6} {'tier':>5} {'score':>10} {'entry_price':>12} {'signal_id':>10}")
    print(f"  {'-'*75}")

    for code, h in holdings_after.items():
        tier = h.get("tier", "C")
        tc = h.get("ts_code", code)
        name = name_map.get(tc, "") if name_map else ""
        print(f"  {name:<10} {code:<8} {'持有':<6} {tier:>5} {h.get('score', 0):>10.4f} {h.get('buy_price', 0):>12.2f} {str(h.get('signal_id', '')):>10}")

    for code, bp_val, ts_code_val, sc_val, sid_val, tier in pending_buys:
        name = name_map.get(str(ts_code_val), "") if name_map else ""
        print(f"  {name:<10} {code:<8} {'新买':<6} {tier:>5} {sc_val:>10.4f} {'待定(T+1)':>12} {str(sid_val):>10}")


def _print_section_anomaly(target_date, sells, pending_buys, skipped_buys, skipped_sells, predictions_exist, candidates_empty):
    """
    Section 异常摘要: system_health / execution_anomalies / data_quality_flags
    输出清晰的系统健康状态和异常清单，便于定位流程问题还是策略问题
    """
    anomalies = []
    warnings = []

    # 1. system_health — 账本完整性
    dec_path = os.path.join(DECISIONS_DIR, f"{target_date.strftime('%Y-%m-%d')}.parquet")
    exec_path = os.path.join(EXECUTIONS_DIR, f"{target_date.strftime('%Y-%m-%d')}.parquet")
    hold_path = os.path.join(HOLDINGS_DIR, f"{target_date.strftime('%Y-%m-%d')}.parquet")

    health_items = []
    health_items.append(f"decisions: {'✅' if os.path.exists(dec_path) else '❌ 缺失'}")
    health_items.append(f"executions: {'✅' if os.path.exists(exec_path) else '❌ 缺失'}")
    health_items.append(f"holdings: {'✅' if os.path.exists(hold_path) else '❌ 缺失'}")
    health_items.append(f"predictions: {'✅' if predictions_exist else '❌ 缺失'}")

    # 2. execution_anomalies — 执行异常
    exec_items = []
    if skipped_buys:
        for code, bp, ts_code, sc, sid, reason in skipped_buys:
            exec_items.append(f"buy_skip: {ts_code} ({reason})")
    if skipped_sells:
        for item in skipped_sells:
            code = item.get("code", "?")
            h = item.get("holding", {})
            ts = h.get("ts_code", code)
            reason = item.get("reason", "unknown")
            exec_items.append(f"sell_skip: {ts} ({reason})")
    if not skipped_buys and not skipped_sells:
        exec_items.append("无异常")
    else:
        anomalies.append(f"执行跳过: {len(skipped_buys)+len(skipped_sells)} 笔")

    # 3. data_quality_flags
    dq_items = []
    if candidates_empty:
        dq_items.append("⚠️ 当日候选池为空 (no_candidates_today)")
        anomalies.append("candidates_empty")
    if not predictions_exist:
        dq_items.append("❌ 预测文件缺失")
    if not dq_items:
        dq_items.append("正常")

    # 4. decisions summary
    dec_items = []
    dec_items.append(f"买入决策: {len(pending_buys)} 笔")
    dec_items.append(f"卖出决策: {len(sells)} 笔")

    print()
    print("=" * 85)
    print("  异常摘要 (System Health & Anomalies)")
    print("=" * 85)
    print(f"  🔧 系统健康:")
    for item in health_items:
        print(f"      {item}")
    print(f"  ⚡ 执行异常:")
    for item in exec_items:
        print(f"      {item}")
    print(f"  📊 数据质量:")
    for item in dq_items:
        print(f"      {item}")
    print(f"  📋 今日决策:")
    for item in dec_items:
        print(f"      {item}")
    if anomalies:
        print(f"  ⚠️ 异常汇总: {', '.join(anomalies)}")
    else:
        print(f"  ✅ 系统健康，无异常")


def _print_section_7_notes(target_date, prev_date_decision):
    print()
    print("=" * 75)
    print(f"  Section 7: 风险提示与假设说明")
    print("=" * 75)
    print(f"  1. cur_ret 计算: 使用当日 ({target_date.strftime('%Y-%m-%d')}) 收盘价，若无则标注 None")
    print(f"  2. 模型退出参考: 使用前一日 ({prev_date_decision.strftime('%Y-%m-%d')}) 的 pred_buy_cls")
    print(f"  3. 买入执行: T+1 开盘价，存在跳空风险")
    print(f"  4. 卖出执行: 当日开盘价，若涨停/跌停可能无法成交")
    print(f"  5. 数据校验: 若数据过期，立即报错停止，不使用过期数据做决策")
    print(f"  7. 交易成本: 未计入佣金/印花税 (买入万五 + 卖出千1.5)")
    print(f"  8. 报告仅反映模型预测，不构成投资建议")
    print()
    print("=" * 75)
    print(f"  📊 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)


# ==================== 主流程 ====================


def main():
    parser = argparse.ArgumentParser(description="日级推理预测报告 — 完整决策链路展示")
    parser.add_argument("--date", type=str, default=None,
                        help="目标决策日 YYYY-MM-DD (T日收盘后决策，T+1日执行买卖)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅 console 输出，不写 Markdown 文件")
    parser.add_argument("--send-feishu", action="store_true",
                        help="生成报告后发送飞书卡片消息到手机")
    args = parser.parse_args()

    print("=" * 75)
    print("  日级推理预测报告")
    print("  基准: V1_PARAMS (exit_mode=model_exit, stop_loss=-7%, max_hold=20d)")
    print("=" * 75)

    # ---- Step 1: 确定目标日期 ----
    # 需要先确定目标日期，才能检测数据是否过期
    if args.date:
        target_date = pd.to_datetime(args.date)
    else:
        # 默认使用最新交易日（T日收盘后决策）
        # 先临时加载一次数据获取 trading_days
        temp_cand_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
        if os.path.exists(temp_cand_path):
            temp_df = pd.read_parquet(temp_cand_path, columns=['obs_date'])
            temp_df['obs_date'] = pd.to_datetime(temp_df['obs_date'])
            latest_obs = temp_df['obs_date'].max()
        else:
            latest_obs = pd.to_datetime("today")

        # 获取 trading_days
        from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data
        _, _, trading_days, _, _ = _load_data(
            candidate_obs_days=V1_PARAMS.get("candidate_obs_days", [1, 2, 3])
        )
        sorted_td = sorted(trading_days)
        target_date = min(d for d in sorted_td if d > latest_obs) if any(d > latest_obs for d in sorted_td) else sorted_td[-1]

    print(f"\n🎯 目标决策日 (T日): {target_date.strftime('%Y-%m-%d')}")

    # ---- Step 2: 加载引擎数据（传入 target_date 检测并更新数据）----
    print("\n⏳ 加载引擎数据...")
    df_all, score_col, price_pivot, trading_days, pred_indexed, snapshots_map = _load_engine_data(target_date)
    print(f"  候选: {len(df_all)} 行, 交易日: {len(trading_days)}, 快照: {len(snapshots_map)} 日")

    # ---- Step 3: 确保 pred 覆盖 ----
    pred_indexed, prev_date_pred, synth_pred = _ensure_pred_coverage(target_date, pred_indexed, trading_days)

    # ---- Step 4: 获取起始持仓 ----
    holdings_start, pending_buys_start, pending_sells_start, prev_date_holdings = _get_start_holdings(
        target_date, snapshots_map, trading_days, price_pivot, pred_indexed,
    )

    # ---- Step 5: 构建候选池 ----
    candidates, prev_date_cand = _build_candidate_pool(target_date, df_all, pred_indexed, trading_days)

    # ---- Step 6: 保存前日收盘持仓 (用于 Section 2) ----
    holdings_close_prev = dict(holdings_start)

    # ---- Step7: 执行 pending buys (T+1 开盘) ----
    # 说明：pending_buys_start 是 T-1 日决策、T日开盘执行的买入
    for code, bp_val, ts_code_val, sc_val, sid_val in pending_buys_start:
        holdings_start[code] = {
            "buy_date": target_date,  # T日执行买入
            "buy_price": bp_val,
            "weight": 1.0 / DEFAULT_MAX_STOCKS,
            "days_held": 0, "ts_code": ts_code_val,
            "score": sc_val, "signal_id": sid_val,
        }

    # ---- Step8: 执行 pending sells (T日开盘) ----
    # 说明：pending_sells_start 是 T-1 日决策、T日开盘执行的卖出
    sold_at_open = []
    for sell_item in pending_sells_start:
        code = sell_item["code"]
        if code in holdings_start:
            sold_at_open.append({
                "code": code,
                "ts_code": sell_item.get("holding", {}).get("ts_code", code),
                "reason": sell_item.get("reason", ""),
            })
            del holdings_start[code]
    if sold_at_open:
        print(f"  [T日开盘卖出] 执行 {len(sold_at_open)} 只: {', '.join(s['ts_code'] for s in sold_at_open)}")

    # ---- Step9: 计算 cur_ret (T日收盘价，用于 T日收盘后决策) ----
    for code, h in holdings_start.items():
        h["cur_ret"] = None
        if target_date in price_pivot.index:
            close_s = price_pivot.loc[target_date, "close"]
            if code in close_s.index:
                close_p = close_s[code]
                if pd.notna(close_p) and h.get("buy_price") and h["buy_price"] > 0:
                    h["cur_ret"] = (close_p - h["buy_price"]) / h["buy_price"]

    # ---- Step10: 调用 SSOT 决策函数 ----
    target_idx, prev_date_decision, _ = _resolve_trading_context(target_date, trading_days)

    # ---- 将持仓中不在候选池的股票补充到候选池（决策日开盘前统一评估）----
    if not candidates.empty and "ts_code" in candidates.columns:
        held_in_cand = set(candidates["ts_code"].values)
    else:
        held_in_cand = set()

    for code, h in holdings_start.items():
        tc = h.get("ts_code", code)
        if tc not in held_in_cand:
            new_row = {
                "ts_code": tc, "score": h.get("score", 0),
                "signal_id": h.get("signal_id"),
                "obs_day": 99, "tier": "C",
                "pred_sell_reg": h.get("score", 0),
                "pred_buy_cls": np.nan, "pred_buy_reg": np.nan,
            }
            for col in ["pred_sell_cls", "pred_buy_reg"]:
                if col not in candidates.columns:
                    new_row[col] = np.nan
            candidates = pd.concat([candidates, pd.DataFrame([new_row])], ignore_index=True)

    holdings_before_decision = {k: dict(v) for k, v in holdings_start.items()}

    with redirect_stdout(StringIO()):
        holdings_after, pending_buys_new, pending_sells_new, sell_reasons = decide_daily_actions(
            target_date, holdings_start, candidates, pred_indexed, prev_date_decision,
            price_pivot=price_pivot, trading_dates=trading_days,
        )

    # 将 pending_sells_new 转换为便于打印的格式
    sells = [item["holding"]["ts_code"] for item in pending_sells_new]
    sell_reasons = {item["holding"]["ts_code"]: item["reason"] for item in pending_sells_new}

    # ---- 保存决策后持仓到 holdings 账本 ----
    save_holdings(target_date, holdings_after, pending_buys_new, pending_sells_new)

    # ---- 构建股票名称映射（在决策之后，收集所有涉及的 ts_code）----
    all_ts_codes = set()
    for h in holdings_start.values():
        all_ts_codes.add(h.get("ts_code", ""))
    for code, bp_val, ts_code_val, sc_val, sid_val in pending_buys_new:
        all_ts_codes.add(str(ts_code_val))
    if not candidates.empty:
        for _, r in candidates.head(30).iterrows():
            all_ts_codes.add(str(r.get("ts_code", "")))
    all_ts_codes.discard("")
    name_map = _build_name_map(all_ts_codes)

    # ---- 为 pending_buys 补充 tier 信息（从 candidates 中匹配）----
    ts_to_tier = {}
    if not candidates.empty and "tier" in candidates.columns:
        for _, r in candidates.iterrows():
            ts_to_tier[str(r.get("ts_code", ""))] = r.get("tier", "C")

    pending_buys_with_tier = []
    for code, bp_val, ts_code_val, sc_val, sid_val in pending_buys_new:
        tier = ts_to_tier.get(str(ts_code_val), "C")
        pending_buys_with_tier.append((code, bp_val, ts_code_val, sc_val, sid_val, tier))

    # 计算 W1/W3 权重
    pending_w1 = _compute_weights_for_buys(pending_buys_with_tier, TIER_WEIGHTS_W1)
    pending_w3 = _compute_weights_for_buys(pending_buys_with_tier, TIER_WEIGHTS_W3)

    # ---- 输出报告 ----
    top30_df, day_top10s = _build_3day_top30(target_date, df_all, candidates, trading_days)

    _print_section_1_params()
    _print_section_2_holdings(holdings_close_prev, prev_date_holdings)
    _print_section_2b_top30(top30_df, day_top10s, holdings_before_decision, name_map)
    _print_section_3_candidates(candidates, holdings_before_decision, prev_date_cand, name_map)
    _print_section_4_exit_scan(holdings_before_decision, sells, sell_reasons, prev_date_decision, pred_indexed, name_map)
    _print_section_4b_actions(top30_df, holdings_before_decision, sells, name_map)
    _print_section_5_trades(sells, sell_reasons, pending_buys_with_tier, holdings_before_decision, holdings_after,
                            target_date, trading_days, name_map)
    _print_section_5c_dual_track(pending_w1, pending_w3, name_map)
    _print_section_6_portfolio(holdings_after, pending_buys_with_tier, target_date, trading_days, name_map)
    _print_section_7_notes(target_date, prev_date_decision)

    # ---- 异常摘要 ----
    skipped_buys_for_report = []
    skipped_sells_for_report = []
    # Read execution ledger for skipped items
    exec_path_report = os.path.join(EXECUTIONS_DIR, f"{target_date.strftime('%Y-%m-%d')}.parquet")
    if os.path.exists(exec_path_report):
        exec_df = pd.read_parquet(exec_path_report)
        skipped = exec_df[exec_df["status"] == "skipped"]
        sell_skip = skipped[skipped["action"] == "sell"]
        buy_skip = skipped[skipped["action"] == "buy"]
        for _, row in sell_skip.iterrows():
            skipped_sells_for_report.append({
                "code": row["ts_code"], "holding": {"ts_code": row["ts_code"]},
                "reason": row["skip_reason"],
            })
        for _, row in buy_skip.iterrows():
            skipped_buys_for_report.append((
                row["ts_code"], row["planned_price"], row["ts_code"],
                None, row["signal_id"], row["skip_reason"],
            ))
    predictions_file = os.path.join(PREDICTIONS_DIR, f"{target_date.strftime('%Y-%m-%d')}.parquet")
    predictions_exist = os.path.exists(predictions_file)
    _print_section_anomaly(target_date, sells, pending_buys_new,
                           skipped_buys_for_report, skipped_sells_for_report,
                           predictions_exist, candidates.empty)

    # ---- 飞书推送 ----
    if args.send_feishu:
        print(f"\n📱 生成飞书卡片消息...")
        header_title, header_template, elements = format_feishu_card(
            target_date, holdings_before_decision, sells, sell_reasons,
            pending_buys_with_tier, candidates, top30_df,
            prev_date_decision, pred_indexed, name_map,
        )
        ok = send_feishu_report(header_title, header_template, elements)
        if ok:
            print(f"✅ 飞书消息已发送")
        else:
            print(f"⚠️ 飞书消息发送失败，请检查飞书配置或网络")

    # ---- 写文件 ----
    if not args.dry_run:
        os.makedirs(REPORT_DIR, exist_ok=True)
        report_path = os.path.join(REPORT_DIR, f"{target_date.strftime('%Y-%m-%d')}.md")
        print(f"\n📁 报告已保存: {report_path}")
        print(f"(Markdown 写入功能将在下一版实现，当前仅 console 输出)")


if __name__ == "__main__":
    main()
