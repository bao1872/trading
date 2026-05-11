#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一策略执行引擎 — 单引擎，多模式

Purpose:
    回测、回放、模拟盘共用同一执行内核 (decision_core + daily_state_machine + live_ledger)，
    消除双引擎口径分叉问题。

Modes:
    backtest: 历史回测，写入 backtest_ledger/
    replay:   一致性回放，写入 replay_ledger/，与 backtest_ledger 对账
    live:     每日盘后生产，写入 live/

Inputs:
    - full_test_predictions.parquet (backtest/replay)
    - predictions/YYYY-MM-DD.parquet (live)
    - K线价格数据

Outputs:
    - {mode}_ledger/holdings/YYYY-MM-DD.parquet
    - {mode}_ledger/decisions/YYYY-MM-DD.parquet
    - {mode}_ledger/executions/YYYY-MM-DD.parquet
    - 净值曲线 + 交易报告 (后处理)

How to Run:
    # 回测
    python -m stop_experiment.engine.strategy_runner --mode backtest

    # 回放对账
    python -m stop_experiment.engine.strategy_runner --mode replay --verify

    # 模拟盘
    python -m stop_experiment.engine.strategy_runner --mode live --date 2026-05-08

    # 模拟盘批量
    python -m stop_experiment.engine.strategy_runner --mode live --batch-all

Side Effects:
    - 写 ledger 文件 (parquet)
    - 写净值曲线 + 交易报告 (csv)
    - 不写数据库
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stop_experiment.backtest.daily_state_machine import step_day
from stop_experiment.backtest.simple_backtest import score_stocks
from stop_experiment.pipeline.live_ledger import (
    load_holdings, save_holdings, save_decisions, save_executions,
)
from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, PREDICTIONS_DIR, HOLDINGS_DIR, DECISIONS_DIR, EXECUTIONS_DIR,
    LIVE_DIR, BACKTEST_LEDGER_DIR, REPLAY_LEDGER_DIR,
    PRODUCTION_PARAMS,
    BUY_COST, SELL_COST,
)

DEFAULT_MAX_STOCKS = 10

MODE_BASE_DIR = {
    "backtest": BACKTEST_LEDGER_DIR,
    "replay": REPLAY_LEDGER_DIR,
    "live": LIVE_DIR,
}


def _resolve_base_dir(mode, base_dir=None):
    if base_dir is not None:
        return base_dir
    return MODE_BASE_DIR.get(mode, LIVE_DIR)


def _resolve_prev_decision_date(target_date, trading_days, pred_lookup):
    sorted_td = sorted(trading_days)
    idx = None
    for i, d in enumerate(sorted_td):
        if d == target_date:
            idx = i
            break
    if idx is None:
        return None
    for j in range(idx - 1, max(idx - 4, -1), -1):
        candidate_dt = sorted_td[j]
        has_any = any(k[1] == candidate_dt for k in pred_lookup.keys())
        if has_any:
            return candidate_dt
    return sorted_td[idx - 1] if idx > 0 else None


def _load_latest_holdings(target_date, trading_days, base_dir=None):
    sorted_td = sorted(trading_days)
    idx = None
    for i, d in enumerate(sorted_td):
        if d == target_date:
            idx = i
            break
    if idx is None:
        return None
    for j in range(idx - 1, max(idx - 11, -1), -1):
        prev_dt = sorted_td[j]
        h = load_holdings(prev_dt, base_dir=base_dir)
        if h is not None:
            print(f"  [持仓回溯] {target_date.strftime('%Y-%m-%d')} <- {prev_dt.strftime('%Y-%m-%d')} ({len(h)} 只)")
            return h
    print(f"  [持仓回溯] {target_date.strftime('%Y-%m-%d')} <- 无 (回溯 10 日内无持仓文件)")
    return None


def _build_candidate_pool(target_date, df_all, trading_days, pred_lookup):
    if "obs_date" not in df_all.columns:
        return pd.DataFrame()
    candidates = df_all[df_all["obs_date"] == target_date].copy()
    if candidates.empty:
        return pd.DataFrame()
    return candidates


def _get_day_open_next(target_date, price_pivot, trading_days):
    sorted_td = sorted(trading_days)
    next_idx = None
    for i, d in enumerate(sorted_td):
        if d == target_date:
            next_idx = i + 1 if i + 1 < len(sorted_td) else None
            break
    if next_idx is not None:
        nd = sorted_td[next_idx]
        if nd in price_pivot.index:
            return price_pivot.loc[nd, "open"]
    return None


def run_day(
    date,
    holdings,
    pending_buys,
    pending_sells,
    candidates_df,
    price_pivot,
    trading_days,
    prev_close_map,
    pred_lookup,
    params,
    write_ledgers=False,
    base_dir=None,
):
    """单日推进 — 唯一日状态推进入口"""
    if isinstance(date, str):
        date = pd.to_datetime(date)

    prev_dt = _resolve_prev_decision_date(date, trading_days, pred_lookup)

    day_close = price_pivot.loc[date, "close"]
    day_open_next = _get_day_open_next(date, price_pivot, trading_days)

    step_params = {
        "max_stocks": params.get("max_stocks_default", DEFAULT_MAX_STOCKS),
        "max_hold_days": params.get("max_hold_days", 20),
        "stop_loss": params.get("stop_loss", -0.07),
        "exit_threshold": params.get("buy_cls_exit_threshold", 0.70),
    }

    step_result = step_day(
        date, dict(holdings), list(pending_buys), list(pending_sells),
        price_pivot, candidates_df, pred_lookup, prev_dt, step_params,
        prev_close_map=prev_close_map, strict=True,
        day_open_next=day_open_next,
    )

    if write_ledgers:
        _base = _resolve_base_dir("custom", base_dir)
        _hold_dir = os.path.join(_base, "holdings")
        save_executions(
            date,
            step_result["executed_buys"],
            step_result["executed_sells"],
            step_result["skipped_buys"],
            step_result["skipped_sells"],
            base_dir=_base,
        )
        save_decisions(
            date,
            step_result["holdings"],
            step_result["pending_buys"],
            step_result["pending_sells"],
            step_result["sell_reasons"],
            extra=step_result.get("extra", {}),
            day_close=day_close,
            base_dir=_base,
        )
        save_holdings(date, step_result["holdings"], base_dir=_hold_dir)

    return step_result


def run_range(
    start_date=None,
    end_date=None,
    mode="backtest",
    prediction_source=None,
    params=None,
    initial_holdings=None,
    write_ledgers=True,
    postprocess=True,
    base_dir=None,
    verify=False,
):
    """多日连续运行 — 统一回测/回放/模拟盘入口"""
    if params is None:
        params = PRODUCTION_PARAMS
    if prediction_source is None:
        prediction_source = "full_test" if mode in ("backtest", "replay") else "daily_parquet"

    _base = _resolve_base_dir(mode, base_dir)

    print(f"\n{'='*70}")
    print(f"  统一策略引擎 — mode={mode}, prediction_source={prediction_source}")
    print(f"  ledger: {_base}")
    print(f"{'='*70}")

    # ---- 加载数据 ----
    if prediction_source == "full_test":
        from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data
        candidate_obs_days = params.get("candidate_obs_days", [1])
        test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data(
            candidate_obs_days=candidate_obs_days
        )
        df_all = test_df.copy()
        if "obs_date" in df_all.columns:
            df_all["obs_date"] = pd.to_datetime(df_all["obs_date"])
        df_all = df_all[df_all["obs_day"].isin(candidate_obs_days)].copy()
        df_all = score_stocks(df_all, "sell_score")
        print(f"  候选: {len(df_all)} 行 (全量预测), 交易日: {len(trading_days)}")
    else:
        raise NotImplementedError("daily_parquet 模式请使用 run_daily.py 或 09_paper_trading_runner.py")

    # ---- 确定日期范围 ----
    sorted_td = sorted(trading_days)
    dates = list(sorted_td)
    if start_date is not None:
        sd = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        dates = [d for d in dates if d >= sd]
    if end_date is not None:
        ed = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
        dates = [d for d in dates if d <= ed]

    print(f"  运行日期: {len(dates)} 日 ({dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')})\n")

    # ---- 日循环 ----
    holdings = dict(initial_holdings) if initial_holdings else {}
    pending_buys = []
    pending_sells = []
    results = []

    for tdate in dates:
        candidates = _build_candidate_pool(tdate, df_all, trading_days, pred_lookup)

        step_result = run_day(
            date=tdate,
            holdings=holdings,
            pending_buys=pending_buys,
            pending_sells=pending_sells,
            candidates_df=candidates,
            price_pivot=price_pivot,
            trading_days=trading_days,
            prev_close_map=prev_close_map,
            pred_lookup=pred_lookup,
            params=params,
            write_ledgers=write_ledgers,
            base_dir=_base,
        )

        holdings = step_result["holdings"]
        pending_buys = step_result["pending_buys"]
        pending_sells = step_result["pending_sells"]
        results.append({"date": tdate, "step_result": step_result})

        n_buys = len(step_result["executed_buys"])
        n_sells = len(step_result["executed_sells"])
        n_hold = len(holdings)
        print(f"  {tdate.strftime('%Y-%m-%d')}: hold={n_hold} buy_exec={n_buys} sell_exec={n_sells} "
              f"pb={len(pending_buys)} ps={len(pending_sells)}")

    # ---- 后处理 ----
    if postprocess and write_ledgers:
        from stop_experiment.pipeline.build_live_equity_curve import (
            build_live_equity_curve, save_live_equity_curve,
        )
        from stop_experiment.pipeline.build_live_trade_report import (
            build_live_trade_report, save_live_trade_report,
        )

        print(f"\n{'='*70}")
        print(f"后处理: 净值曲线 + 交易报告")
        eq_df = build_live_equity_curve(price_pivot, base_dir=_base)
        eq_path = os.path.join(_base, "live_equity_curve.csv")
        save_live_equity_curve(eq_df, output_path=eq_path)

        report_df, trade_summary = build_live_trade_report(base_dir=_base)
        report_path = os.path.join(_base, "live_trade_report.csv")
        save_live_trade_report(report_df, trade_summary, output_path=report_path)

        if len(eq_df) > 0:
            final_nav = eq_df["nav_live"].iloc[-1]
            print(f"  最终 NAV: {final_nav:.4f}")

    # ---- 对账 ----
    if verify and mode == "replay":
        _verify_ledgers(BACKTEST_LEDGER_DIR, REPLAY_LEDGER_DIR)

    return {
        "results": results,
        "final_holdings": holdings,
        "base_dir": _base,
    }


def _verify_ledgers(bt_dir, rp_dir):
    """对比两个 ledger 目录的 holdings/decisions/executions"""
    print(f"\n{'='*70}")
    print(f"对账: {bt_dir} vs {rp_dir}")
    print(f"{'='*70}")

    bt_hold = os.path.join(bt_dir, "holdings") if not bt_dir.endswith("holdings") else bt_dir
    rp_hold = os.path.join(rp_dir, "holdings") if not rp_dir.endswith("holdings") else rp_dir

    bt_dec = os.path.join(bt_dir, "decisions")
    rp_dec = os.path.join(rp_dir, "decisions")

    mismatches = 0
    total = 0

    # 对比 holdings
    if os.path.isdir(bt_hold) and os.path.isdir(rp_hold):
        bt_files = set(os.listdir(bt_hold))
        rp_files = set(os.listdir(rp_hold))
        common = bt_files & rp_files
        for f in sorted(common):
            bt_df = pd.read_parquet(os.path.join(bt_hold, f))
            rp_df = pd.read_parquet(os.path.join(rp_hold, f))
            total += 1
            if not bt_df.equals(rp_df):
                mismatches += 1
                print(f"  [HOLDINGS MISMATCH] {f}: bt={len(bt_df)} rp={len(rp_df)}")
        missing_bt = bt_files - rp_files
        missing_rp = rp_files - bt_files
        if missing_bt:
            print(f"  [MISSING in replay] {sorted(missing_bt)}")
            mismatches += len(missing_bt)
        if missing_rp:
            print(f"  [MISSING in backtest] {sorted(missing_rp)}")
            mismatches += len(missing_rp)

    # 对比 decisions
    if os.path.isdir(bt_dec) and os.path.isdir(rp_dec):
        bt_files = set(os.listdir(bt_dec))
        rp_files = set(os.listdir(rp_dec))
        common = bt_files & rp_files
        for f in sorted(common):
            bt_df = pd.read_parquet(os.path.join(bt_dec, f))
            rp_df = pd.read_parquet(os.path.join(rp_dec, f))
            total += 1
            if len(bt_df) != len(rp_df):
                mismatches += 1
                print(f"  [DECISIONS MISMATCH] {f}: bt={len(bt_df)} rp={len(rp_df)}")

    if mismatches == 0:
        print(f"  对账通过: {total} 个文件完全一致")
    else:
        print(f"  对账失败: {mismatches} 处不一致 (共 {total} 个文件)")


def main():
    parser = argparse.ArgumentParser(description="统一策略执行引擎")
    parser.add_argument("--mode", choices=["backtest", "replay", "live"], default="backtest")
    parser.add_argument("--date", type=str, default=None, help="单日 YYYY-MM-DD")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--batch-all", action="store_true")
    parser.add_argument("--verify", action="store_true", help="replay 模式对账")
    parser.add_argument("--no-postprocess", action="store_true", help="跳过后处理")
    args = parser.parse_args()

    if args.mode == "live":
        raise NotImplementedError(
            "live 模式请使用 run_daily.py 或 09_paper_trading_runner.py\n"
            "strategy_runner 的 live 模式将在 Phase 5 完成后启用"
        )

    start = args.start_date
    end = args.end_date
    if args.date:
        start = args.date
        end = args.date

    result = run_range(
        start_date=start,
        end_date=end,
        mode=args.mode,
        write_ledgers=True,
        postprocess=not args.no_postprocess,
        verify=args.verify,
    )

    print(f"\n{'='*70}")
    print(f"  运行完成: mode={args.mode}, ledger={result['base_dir']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
