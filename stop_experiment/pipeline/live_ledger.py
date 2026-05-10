#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一账本读写模块（SSOT）

Purpose:
    统一负责 holdings / decisions / executions 三本账的读写，
    消除 08_daily_inference_report.py 和 09_paper_trading_runner.py 中的重复实现。

    所有读写函数唯一定义于此，其他模块只能通过 import 引用调用。

    Usage:
        python stop_experiment/pipeline/live_ledger.py
        运行自测（读写临时目录，无副作用）。

    How to Run:
        python stop_experiment/pipeline/live_ledger.py

    Examples:
        # 单日模拟盘
        from stop_experiment.pipeline.live_ledger import load_holdings, save_holdings, save_decisions, save_executions
        holdings = load_holdings(date)
        save_executions(target_date, executed_buys, executed_sells, skipped_buys, skipped_sells)
        save_decisions(target_date, holdings, pending_buys, pending_sells, sell_reasons, extra=extra, day_close=day_close)
        save_holdings(target_date, holdings)

    Side Effects:
        - 读写 holdings/decisions/executions 三本账（parquet 格式）
        - 不修改数据库 schema

    Outputs:
        - holdings/YYYY-MM-DD.parquet
        - live/decisions/YYYY-MM-DD.parquet
        - live/executions/YYYY-MM-DD.parquet
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import (
    HOLDINGS_DIR, DECISIONS_DIR, EXECUTIONS_DIR,
    BASELINE_E0_X1_V1_PARAMS,
)

DEFAULT_MAX_STOCKS = BASELINE_E0_X1_V1_PARAMS.get("max_stocks", 10)


def load_holdings(date):
    """从 holdings/YYYY-MM-DD.parquet 读取持仓"""
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


def save_holdings(date, holdings):
    """保存当日决策后持仓到 holdings/YYYY-MM-DD.parquet"""
    os.makedirs(HOLDINGS_DIR, exist_ok=True)
    if isinstance(date, str):
        date = pd.to_datetime(date)
    rows = []
    for code, h in holdings.items():
        rows.append({
            "date": date, "code": code,
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


def save_decisions(date, holdings, pending_buys, pending_sells, sell_reasons, extra=None, day_close=None):
    """T日收盘后保存决策账本。cur_ret 优先从 extra details 取，兜底用 day_close+buy_price 计算。"""
    os.makedirs(DECISIONS_DIR, exist_ok=True)
    if isinstance(date, str):
        date = pd.to_datetime(date)

    def _resolve_cur_ret(h, detail, default=None):
        val = detail.get("cur_ret")
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return val
        if day_close is not None and h.get("buy_price") and h["buy_price"] > 0:
            ts_val = h.get("ts_code", "")
            cp = day_close.get(ts_val)
            if cp is None or (isinstance(cp, float) and np.isnan(cp)):
                code_pure = ts_val.split(".")[0] if isinstance(ts_val, str) and "." in ts_val else ts_val
                cp = day_close.get(code_pure)
            if cp is not None and not (isinstance(cp, float) and np.isnan(cp)) and cp > 0:
                return float((cp - h["buy_price"]) / h["buy_price"])
        return default

    buy_details = extra.get("buy_details", []) if extra else []
    sell_details = extra.get("sell_details", []) if extra else []
    hold_details = extra.get("hold_details", []) if extra else []
    buy_detail_map = {d["ts_code"]: d for d in buy_details}
    buy_detail_map.update({d.get("code", ""): d for d in buy_details})
    sell_detail_map = {d["ts_code"]: d for d in sell_details}
    sell_detail_map.update({d.get("code", ""): d for d in sell_details})
    hold_detail_map = {d["ts_code"]: d for d in hold_details}
    hold_detail_map.update({d.get("code", ""): d for d in hold_details})
    sell_codes = set(sell_reasons.keys())
    rows = []
    for code, h in holdings.items():
        if code in sell_codes:
            continue
        ts = h.get("ts_code", code)
        detail = hold_detail_map.get(ts, hold_detail_map.get(code, {}))
        rows.append({
            "decision_date": date, "ts_code": ts,
            "signal_id": h.get("signal_id"), "action": "hold",
            "reason": detail.get("why", "held"),
            "score": h.get("score", 0),
            "days_held": h.get("days_held", 0),
            "cur_ret": _resolve_cur_ret(h, detail),
            "threshold_value": None,
            "why": detail.get("why", ""),
        })
    for code, reason in sell_reasons.items():
        h = holdings.get(code, {})
        ts = h.get("ts_code", code)
        detail = sell_detail_map.get(ts, sell_detail_map.get(code, {}))
        rows.append({
            "decision_date": date, "ts_code": ts,
            "signal_id": h.get("signal_id"), "action": "sell",
            "reason": reason,
            "score": h.get("score", 0),
            "days_held": h.get("days_held", 0),
            "cur_ret": _resolve_cur_ret(h, detail),
            "threshold_value": detail.get("threshold"),
            "why": detail.get("why", ""),
        })
    rank = 0
    for code, bp_val, ts_code_val, sc_val, sid_val in pending_buys:
        detail = buy_detail_map.get(code, {})
        rows.append({
            "decision_date": date, "ts_code": ts_code_val,
            "signal_id": sid_val, "action": "buy",
            "reason": detail.get("why", "candidate_top"),
            "score": sc_val,
            "days_held": 0, "cur_ret": None,
            "threshold_value": detail.get("threshold_value"),
            "why": detail.get("why", ""),
            "rank": rank, "obs_day": detail.get("obs_day"),
            "planned_price": bp_val,
            "planned_weight": 1.0 / DEFAULT_MAX_STOCKS,
        })
        rank += 1
    candidate_top10 = extra.get("candidate_top10", []) if extra else []
    for cand in candidate_top10:
        rows.append({
            "decision_date": date, "ts_code": cand.get("ts_code", ""),
            "signal_id": cand.get("signal_id"), "action": "candidate",
            "reason": "top10_candidate",
            "score": cand.get("score", 0),
            "days_held": None, "cur_ret": None,
            "rank": cand.get("rank"),
            "is_held": cand.get("is_held", False),
            "is_pending_buy": cand.get("is_pending_buy", False),
            "pred_buy_cls": cand.get("pred_buy_cls"),
            "pred_sell_reg": cand.get("pred_sell_reg"),
            "pred_sell_cls": cand.get("pred_sell_cls"),
            "pred_buy_reg": cand.get("pred_buy_reg"),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(DECISIONS_DIR, f"{date.strftime('%Y-%m-%d')}.parquet")
    df.to_parquet(path, index=False)
    print(f"  [决策保存] {path} ({len(df)} 条)")


def save_executions(date, executed_buys, executed_sells, skipped_buys, skipped_sells):
    """T日开盘后保存执行账本到 executions/YYYY-MM-DD.parquet"""
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
    nb = len(executed_buys) + len(executed_sells)
    ns = len(skipped_buys) + len(skipped_sells)
    print(f"  [执行保存] {path} ({len(df)} 条, executed={nb} skipped={ns})")


if __name__ == "__main__":
    """
    自测：读写真实路径（本地测试环境，无副作用）。
    不依赖临时目录 hack。
    """
    test_date = pd.Timestamp("2026-05-10")

    print("=" * 60)
    print("  live_ledger 自测")
    print("=" * 60)

    # 测试 save_holdings + load_holdings
    test_holdings = {
        "000001.SZ": {"buy_date": test_date, "buy_price": 10.0, "weight": 0.5,
                       "days_held": 1, "ts_code": "000001.SZ", "signal_id": "s1", "score": 0.8},
        "600000.SH": {"buy_date": test_date, "buy_price": 20.0, "weight": 0.5,
                       "days_held": 1, "ts_code": "600000.SH", "signal_id": "s2", "score": 0.6},
    }

    save_holdings(test_date, test_holdings)
    loaded = load_holdings(test_date)
    assert loaded is not None, "load_holdings 返回 None"
    assert len(loaded) == 2, f"期望 2 只，实际 {len(loaded)}"
    print(f"  ✅ save_holdings + load_holdings: {len(loaded)} 只")

    # 测试 save_executions
    executed_buys = [("000001.SZ", 10.0, "000001.SZ", 0.8, "s1")]
    skipped_buys = [("600001.SH", 15.0, "600001.SH", 0.5, "s3", "limit_up")]
    executed_sells = [{"code": "600000.SH", "holding": {"ts_code": "600000.SH", "signal_id": "s2"},
                        "executed_price": 22.0, "reason": "stop_loss"}]
    skipped_sells = []
    save_executions(test_date, executed_buys, executed_sells, skipped_buys, skipped_sells)
    epath = os.path.join(EXECUTIONS_DIR, f"{test_date.strftime('%Y-%m-%d')}.parquet")
    assert os.path.exists(epath), f"执行账本未生成: {epath}"
    edf = pd.read_parquet(epath)
    print(f"  ✅ save_executions: {len(edf)} 条")

    # 测试 save_decisions
    save_decisions(test_date, test_holdings, executed_buys, executed_sells, {"600000.SH": "stop_loss"},
                   extra={"hold_details": [{"ts_code": "000001.SZ", "cur_ret": 0.05, "why": "held"}],
                          "sell_details": [{"ts_code": "600000.SH", "cur_ret": 0.10, "threshold": -0.07, "why": "stop_loss"}],
                          "buy_details": []},
                   day_close=pd.Series({"000001.SZ": 10.5, "600000.SH": 22.0}))
    dpath = os.path.join(DECISIONS_DIR, f"{test_date.strftime('%Y-%m-%d')}.parquet")
    assert os.path.exists(dpath), f"决策账本未生成: {dpath}"
    ddf = pd.read_parquet(dpath)
    hold_row = ddf[ddf["action"] == "hold"].iloc[0]
    assert abs(hold_row["cur_ret"] - 0.05) < 0.001, f"cur_ret 期望 0.05, 实际 {hold_row['cur_ret']}"
    print(f"  ✅ save_decisions: {len(ddf)} 条, cur_ret={hold_row['cur_ret']:.4f}")

    print(f"\n  ✅ live_ledger 自测全部通过")