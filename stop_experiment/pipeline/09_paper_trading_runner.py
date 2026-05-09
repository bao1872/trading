#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟盘主入口 — 单入口串联 5 步日流程，四本账全部落地

Purpose:
    作为模拟盘唯一入口，每天跑一遍完整流程：
    Step 1: 加载前日持仓 + 待执行订单
    Step 2: T日开盘执行买卖 → 生成 executions/T.parquet (含 skip_reason)
    Step 3: 更新开盘后持仓
    Step 4: 收盘后预测 → decide_eod() → 生成 decisions/T.parquet
    Step 5: 保存收盘后持仓 → holdings/T.parquet

Inputs:
    - stop_experiment/output/predictions/T.parquet (当日真实预测)
    - stop_experiment/output/holdings/T-1.parquet (前日持仓)
    - stop_experiment/output/full_test_predictions.parquet (全量预测，用于候选池构建)
    - DB: stock_k_data (K线数据)

Outputs:
    - stop_experiment/output/live/executions/T.parquet
    - stop_experiment/output/live/decisions/T.parquet
    - stop_experiment/output/holdings/T.parquet

How to Run:
    # 单日运行
    python stop_experiment/pipeline/09_paper_trading_runner.py --date 2026-05-08

    # 10 日批量回放
    python stop_experiment/pipeline/09_paper_trading_runner.py --batch-first-10

    # 全量
    python stop_experiment/pipeline/09_paper_trading_runner.py --batch-all

    # 故障注入
    python stop_experiment/pipeline/09_paper_trading_runner.py --date 2026-03-20 --fault-inject MISSING_PREDICTION

Side Effects:
    - 写 decisions/executions/holdings 三本账
    - 只读 predictions 和 DB
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, V1_PARAMS, PREDICTIONS_DIR, HOLDINGS_DIR, DECISIONS_DIR, EXECUTIONS_DIR,
)
from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data
from stop_experiment.backtest.simple_backtest import score_stocks, is_limit_down, is_limit_up, is_suspended
from stop_experiment.backtest.decision_core import decide_eod

DEFAULT_MAX_STOCKS = V1_PARAMS.get("max_stocks_default", 10)

FIRST_10_DATES = [
    "2026-03-20", "2026-03-24", "2026-04-03", "2026-04-10", "2026-04-17",
    "2026-04-24", "2026-04-30", "2026-05-06",
]


# ==================== 前日持仓读写 ====================

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


# ==================== 决策/执行账本 ====================

def save_decisions(date, holdings_after, pending_buys, pending_sells):
    """T日决策后保存决策账本到 decisions/YYYY-MM-DD.parquet"""
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


# ==================== 统一校验 ====================

def validate_daily_inputs(target_date, holdings, price_pivot, pred_indexed, fault_inject=None):
    """
    统一校验当日输入完整性，缺少硬性条件直接 raise。
    返回 (anomalies: list[str]) 非致命异常列表。
    """
    anomalies = []

    # 故障注入
    if fault_inject == "MISSING_PREDICTION":
        raise RuntimeError(f"[FAULT_INJECT] 模拟缺预测文件: {target_date.strftime('%Y-%m-%d')}")

    if fault_inject == "MISSING_HOLDINGS":
        raise RuntimeError(f"[FAULT_INJECT] 模拟缺持仓文件: {target_date.strftime('%Y-%m-%d')}")

    # 1. 前日持仓存在（首日除外：起始日无持仓不报错）
    if holdings is None:
        trading_days_sorted = sorted(price_pivot.index)
        if len(trading_days_sorted) > 0 and target_date != trading_days_sorted[0]:
            raise RuntimeError(
                f"[VALIDATE] 缺少前日持仓: holdings/{target_date.strftime('%Y-%m-%d')}.parquet 不存在"
                f"，且当日不是首个交易日"
            )

    # 2. 价格数据存在
    if target_date not in price_pivot.index:
        raise RuntimeError(f"[VALIDATE] 缺少当日价格数据: {target_date.strftime('%Y-%m-%d')}")

    day_open = price_pivot.loc[target_date, "open"]
    day_close = price_pivot.loc[target_date, "close"]
    if day_open.isna().all():
        anomalies.append("all_open_missing")
    if day_close.isna().all():
        raise RuntimeError(f"[VALIDATE] 当日收盘价全部缺失: {target_date.strftime('%Y-%m-%d')}")

    # 3. 决策用 pred_lookup 非空（非硬性，记录 anomaly）
    if pred_indexed is None or len(pred_indexed) == 0:
        anomalies.append("pred_lookup_empty")

    return anomalies


def validate_daily_outputs(target_date, decisions_path, executions_path, holdings):
    """校验当日输出完整性"""
    issues = []
    if not os.path.exists(decisions_path):
        issues.append("missing_decisions_file")
    if not os.path.exists(executions_path):
        issues.append("missing_executions_file")
    return issues


# ==================== 主流程 ====================

def _build_candidate_pool(date, df_all, trading_days, pred_lookup):
    """从全量候选构建当日候选池"""
    df_date = df_all[df_all["obs_date"] == date].copy()
    if df_date.empty:
        return pd.DataFrame()

    score_col = "score" if "score" in df_date.columns else "composite_score"
    if score_col in df_date.columns:
        df_date = df_date.sort_values(score_col, ascending=False)
    df_date = df_date.drop_duplicates(subset=["ts_code"], keep="first")
    return df_date


def run_single_day(target_date, df_all, price_pivot, trading_days, prev_close_map, pred_lookup,
                   holdings_input=None, pending_buys_input=None, pending_sells_input=None,
                   fault_inject=None):
    """单日模拟盘运行，返回当日结果"""

    # Step 0: 校验
    anomalies = validate_daily_inputs(target_date, holdings_input, price_pivot, pred_lookup, fault_inject)

    # Step 1: 加载前日持仓 + 待执行订单
    holdings = dict(holdings_input) if holdings_input else {}
    pending_buys_prev = list(pending_buys_input) if pending_buys_input else []
    pending_sells_prev = list(pending_sells_input) if pending_sells_input else []
    if holdings_input is None:
        prev_dt = _prev_trading_day(target_date, trading_days)
        if prev_dt is not None:
            holdings = load_holdings(prev_dt) or {}
        # 尝试从 decisions 账本加载前日 pending
        if prev_dt is not None:
            dec_path = os.path.join(DECISIONS_DIR, f"{prev_dt.strftime('%Y-%m-%d')}.parquet")
            if os.path.exists(dec_path):
                dec_df = pd.read_parquet(dec_path)
                buys = dec_df[dec_df["action"] == "buy"]
                sells = dec_df[dec_df["action"] == "sell"]
                for _, row in buys.iterrows():
                    pending_buys_prev.append((
                        row["ts_code"], row["planned_price"], row["ts_code"],
                        row["score"], row["signal_id"],
                    ))
                for _, row in sells.iterrows():
                    pending_sells_prev.append({
                        "code": row["ts_code"],
                        "holding": {"ts_code": row["ts_code"], "signal_id": row["signal_id"]},
                        "reason": row["reason"],
                    })

    # Step 2: 执行 pending orders
    day_open = price_pivot.loc[target_date, "open"]
    executed_buys, skipped_buys = [], []
    executed_sells, skipped_sells = [], []

    for code, bp_val, ts_code_val, sc_val, sid_val in pending_buys_prev:
        if code in holdings:
            skipped_buys.append((code, bp_val, ts_code_val, sc_val, sid_val, "already_held"))
            continue
        buy_price = bp_val
        if np.isnan(buy_price) or buy_price <= 0:
            skipped_buys.append((code, bp_val, ts_code_val, sc_val, sid_val, "missing_price"))
            continue
        if "volume" in price_pivot.columns.levels[0] and code in price_pivot["volume"].columns:
            vol_val = price_pivot["volume"][code].get(target_date, np.nan)
            if is_suspended(vol_val):
                skipped_buys.append((code, bp_val, ts_code_val, sc_val, sid_val, "suspended_skip"))
                continue
        if code in prev_close_map and target_date in prev_close_map[code].index:
            prev_c = prev_close_map[code].get(target_date, np.nan)
            if not np.isnan(prev_c) and prev_c > 0:
                if "high" in price_pivot.columns.levels[0] and code in price_pivot["high"].columns:
                    high_val = price_pivot["high"][code].get(target_date, np.nan)
                    if not np.isnan(high_val) and is_limit_up(buy_price, high_val, prev_c):
                        skipped_buys.append((code, bp_val, ts_code_val, sc_val, sid_val, "limit_up_skip"))
                        continue
        holdings[code] = {
            "buy_date": target_date, "buy_price": buy_price,
            "weight": 1.0 / DEFAULT_MAX_STOCKS,
            "days_held": 0, "ts_code": ts_code_val,
            "score": sc_val, "signal_id": sid_val,
        }
        executed_buys.append((code, bp_val, ts_code_val, sc_val, sid_val))

    for sell_item in pending_sells_prev:
        code = sell_item["code"]
        if code not in holdings:
            continue
        sell_price = np.nan
        if code in day_open.index and not np.isnan(day_open[code]):
            sell_price = day_open[code]
        if np.isnan(sell_price) or sell_price <= 0:
            sell_item["reason"] = "missing_price"
            skipped_sells.append(dict(sell_item))
            anomalies.append(f"skip_sell_no_price:{code}")
            continue
        if "volume" in price_pivot.columns.levels[0] and code in price_pivot["volume"].columns:
            vol_val = price_pivot["volume"][code].get(target_date, np.nan)
            if is_suspended(vol_val):
                sell_item["reason"] = "suspended_skip"
                skipped_sells.append(dict(sell_item))
                anomalies.append(f"skip_sell_suspended:{code}")
                continue
        if code in prev_close_map and target_date in prev_close_map[code].index:
            prev_c = prev_close_map[code].get(target_date, np.nan)
            if not np.isnan(prev_c) and prev_c > 0:
                if "low" in price_pivot.columns.levels[0] and code in price_pivot["low"].columns:
                    low_val = price_pivot["low"][code].get(target_date, np.nan)
                    if not np.isnan(low_val) and is_limit_down(sell_price, low_val, prev_c):
                        sell_item["reason"] = "limit_down_skip"
                        skipped_sells.append(dict(sell_item))
                        anomalies.append(f"skip_sell_limit_down:{code}")
                        continue
        sell_item["executed_price"] = sell_price
        executed_sells.append(dict(sell_item))
        del holdings[code]

    save_executions(target_date, executed_buys, executed_sells, skipped_buys, skipped_sells)

    # Step 3: 开盘后持仓已在 holdings 中（已执行买卖）

    # Step 4: 收盘后决策
    candidates = _build_candidate_pool(target_date, df_all, trading_days, pred_lookup)
    if candidates.empty:
        anomalies.append("no_candidates_today")
        print(f"  [候选池] 空 (no_candidates_today)")

    prev_dt = _prev_trading_day(target_date, trading_days)
    day_close = price_pivot.loc[target_date, "close"]

    next_idx = None
    for i, d in enumerate(sorted(trading_days)):
        if d == target_date:
            next_idx = i + 1 if i + 1 < len(trading_days) else None
            break
    day_open_next = pd.Series(dtype=float)
    if next_idx is not None:
        nd = sorted(trading_days)[next_idx]
        if nd in price_pivot.index:
            day_open_next = price_pivot.loc[nd, "open"]

    holdings, pending_buys, pending_sells, sell_reasons, _extra = decide_eod(
        decision_date=target_date, holdings=holdings,
        candidates=candidates, pred_lookup=pred_lookup,
        prev_date=prev_dt, day_close=day_close,
        day_open_next=day_open_next,
        max_stocks=DEFAULT_MAX_STOCKS,
        max_hold_days=V1_PARAMS.get("max_hold_days", 20),
        stop_loss=V1_PARAMS.get("stop_loss", -0.07),
        exit_threshold=V1_PARAMS.get("buy_cls_exit_threshold", 0.70),
    )

    save_decisions(target_date, holdings, pending_buys, pending_sells)

    # Step 5: 保存持仓
    save_holdings(target_date, holdings)

    return {
        "date": target_date,
        "holdings": holdings,
        "pending_buys": pending_buys,
        "pending_sells": pending_sells,
        "sell_reasons": sell_reasons,
        "executed_buys": executed_buys,
        "executed_sells": executed_sells,
        "skipped_buys": skipped_buys,
        "skipped_sells": skipped_sells,
        "anomalies": anomalies,
        "candidates_count": len(candidates),
    }


def _prev_trading_day(date, trading_days):
    sorted_td = sorted(trading_days)
    for i, d in enumerate(sorted_td):
        if d == date and i > 0:
            return sorted_td[i - 1]
    return None


# ==================== 批量运行 ====================

def run_batch(dates, df_all, price_pivot, trading_days, prev_close_map, pred_lookup):
    """批量运行，状态在日期间传递"""
    results = []
    holdings = {}
    pending_buys = []
    pending_sells = []

    for tdate in sorted(dates):
        res = run_single_day(
            tdate, df_all, price_pivot, trading_days, prev_close_map, pred_lookup,
            holdings_input=holdings, pending_buys_input=pending_buys,
            pending_sells_input=pending_sells,
        )
        holdings = res["holdings"]
        pending_buys = res["pending_buys"]
        pending_sells = res["pending_sells"]
        results.append(res)

        summary = f"  {tdate.strftime('%Y-%m-%d')}: holding={len(holdings)} buy={len(res['pending_buys'])} sell={len(res['pending_sells'])}"
        if res["anomalies"]:
            summary += f" anomalies={res['anomalies']}"
        print(summary)

    return results


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(description="模拟盘主入口 — 单入口 5 步日流程")
    parser.add_argument("--date", type=str, default=None, help="单日 YYYY-MM-DD")
    parser.add_argument("--batch-first-10", action="store_true", help="10 日批量回放")
    parser.add_argument("--batch-all", action="store_true", help="全量回放")
    parser.add_argument("--fault-inject", type=str, default=None,
                        choices=["MISSING_PREDICTION", "MISSING_HOLDINGS", "MISSING_OPEN_PRICE"],
                        help="故障注入测试")
    args = parser.parse_args()

    print("=" * 70)
    print("  模拟盘主入口 — 5 步日流程")
    print("=" * 70)

    # 加载数据
    cand_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    df_all = pd.read_parquet(cand_path)
    df_all["obs_date"] = pd.to_datetime(df_all["obs_date"])
    candidate_obs_days = V1_PARAMS.get("candidate_obs_days", [1, 2, 3])
    df_all = df_all[df_all["obs_day"].isin(candidate_obs_days)].copy()
    df_all = score_stocks(df_all, V1_PARAMS.get("strategy_default", "sell_score"))

    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data(
        candidate_obs_days=V1_PARAMS["candidate_obs_days"]
    )
    print(f"  候选: {len(df_all)} 行, 交易日: {len(trading_days)}")

    dates = []
    if args.date:
        dates = [pd.to_datetime(args.date)]
    elif args.batch_first_10:
        dates = [pd.to_datetime(d) for d in FIRST_10_DATES]
    elif args.batch_all:
        dates = sorted(trading_days)
    else:
        print("请指定 --date, --batch-first-10 或 --batch-all")
        return

    print(f"  运行日期: {len(dates)} 日\n")

    if len(dates) == 1 and args.fault_inject:
        # 单日故障注入
        res = run_single_day(
            dates[0], df_all, price_pivot, trading_days, prev_close_map, pred_lookup,
            fault_inject=args.fault_inject,
        )
    else:
        results = run_batch(dates, df_all, price_pivot, trading_days, prev_close_map, pred_lookup)

        # 汇总
        total_exec = sum(len(r["executed_buys"]) + len(r["executed_sells"]) for r in results)
        total_skip = sum(len(r["skipped_buys"]) + len(r["skipped_sells"]) for r in results)
        total_anomalies = sum(len(r["anomalies"]) for r in results)
        no_candidate_days = sum(1 for r in results if "no_candidates_today" in r["anomalies"])

        print(f"\n{'='*70}")
        print(f"汇总:")
        print(f"  运行: {len(results)} 日")
        print(f"  执行成功: {total_exec} 笔")
        print(f"  跳过: {total_skip} 笔")
        print(f"  异常: {total_anomalies} 条")
        print(f"  无候选: {no_candidate_days} 日")
        files_ok = sum(1 for r in results if len(validate_daily_outputs(
            r["date"],
            os.path.join(DECISIONS_DIR, f"{r['date'].strftime('%Y-%m-%d')}.parquet"),
            os.path.join(EXECUTIONS_DIR, f"{r['date'].strftime('%Y-%m-%d')}.parquet"),
            r["holdings"],
        )) == 0)
        print(f"  文件完整: {files_ok}/{len(results)} 日")


if __name__ == "__main__":
    main()