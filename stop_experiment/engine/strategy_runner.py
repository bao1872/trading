#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一策略执行引擎 — 回测模式

Purpose:
    回测执行引擎，使用 prediction_store 作为唯一预测源，
    写入 backtest_ledger/ 产出持仓/决策/执行/净值/交易报告。

Inputs:
    - prediction_store (唯一预测源)
    - K线价格数据

Outputs:
    - backtest_ledger/holdings/YYYY-MM-DD.parquet
    - backtest_ledger/decisions/YYYY-MM-DD.parquet
    - backtest_ledger/executions/YYYY-MM-DD.parquet
    - backtest_ledger/live_equity_curve.csv
    - backtest_ledger/live_trade_report.csv

How to Run:
    python -m stop_experiment.engine.strategy_runner --profile production --start-date 2026-01-05 --end-date 2026-05-08

Side Effects:
    - 写 ledger 文件 (parquet)
    - 写净值曲线 + 交易报告 (csv)
    - 不写数据库
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stop_experiment.backtest.daily_state_machine import step_day
from stop_experiment.backtest.simple_backtest import score_stocks, load_daily_prices, build_price_pivot
from stop_experiment.pipeline.live_ledger import (
    load_holdings, save_holdings, save_decisions, save_executions,
)
from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, BACKTEST_LEDGER_DIR,
    PRODUCTION_PARAMS,
    BUY_COST, SELL_COST, CANDIDATE_OBS_DAYS,
)

DEFAULT_MAX_STOCKS = 10


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


def validate_daily_inputs(target_date, holdings, price_pivot, pred_lookup):
    if target_date not in price_pivot.index:
        raise RuntimeError(f"[VALIDATE] 缺少当日价格数据: {target_date.strftime('%Y-%m-%d')}")
    day_close = price_pivot.loc[target_date, "close"]
    if day_close.isna().all():
        raise RuntimeError(f"[VALIDATE] 当日收盘价全部缺失: {target_date.strftime('%Y-%m-%d')}")


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
    holdings_base_dir=None,
):
    if isinstance(date, str):
        date = pd.to_datetime(date)

    prev_dt = _resolve_prev_decision_date(date, trading_days, pred_lookup)
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
        _base = base_dir or BACKTEST_LEDGER_DIR
        _hold_dir = holdings_base_dir or os.path.join(_base, "holdings")
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
            day_close=price_pivot.loc[date, "close"],
            base_dir=_base,
        )
        save_holdings(date, step_result["holdings"], base_dir=_hold_dir)

    return step_result


def run_range(
    start_date=None,
    end_date=None,
    profile=None,
    params=None,
    initial_holdings=None,
    write_ledgers=True,
    postprocess=True,
    base_dir=None,
):
    if profile is not None:
        try:
            from stop_experiment.registries import resolve_profile_params
            resolved = resolve_profile_params(profile)
            print(f"  [profile] {profile} → strategy_version={resolved.get('strategy_version')}")
        except Exception as e:
            print(f"  [profile] 无法解析 profile '{profile}': {e}, 使用默认 params")
            resolved = {}
        if params is None:
            params = resolved
        elif isinstance(params, dict) and isinstance(resolved, dict):
            merged = dict(resolved)
            merged.update(params)
            params = merged
    if params is None:
        params = PRODUCTION_PARAMS

    _base = base_dir or BACKTEST_LEDGER_DIR

    print(f"\n{'='*70}")
    print(f"  策略引擎 — 回测模式")
    print(f"  ledger: {_base}")
    print(f"{'='*70}")

    daily = load_daily_prices("2024-01-01", "2027-01-01")
    price_pivot, trading_days, prev_close_map = build_price_pivot(daily)
    sorted_td = sorted(trading_days)

    if start_date is not None:
        sd = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
    else:
        sd = sorted_td[0]
    if end_date is not None:
        ed = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
    else:
        ed = sorted_td[-1]
    target_dates = [d for d in sorted_td if sd <= d <= ed]

    pname = profile or "production"
    from stop_experiment.registries.prediction_store import read_prediction_store_range
    df_all, pred_lookup = read_prediction_store_range(pname, target_dates)

    if df_all.empty:
        print("  [LEGACY] prediction_store 无数据，fallback 到 full_test_predictions.parquet (已过时)")
        from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data
        test_df, price_pivot2, trading_days2, prev_close_map2, pred_lookup2 = _load_data()
        if not (not price_pivot.empty and price_pivot2.empty):
            price_pivot, trading_days, prev_close_map = price_pivot2, trading_days2, prev_close_map2
        pred_lookup = pred_lookup2
        df_all = test_df.copy()
        if "obs_date" in df_all.columns:
            df_all["obs_date"] = pd.to_datetime(df_all["obs_date"])
        df_all = df_all[df_all["obs_day"].isin(CANDIDATE_OBS_DAYS)].copy()
        if "ts_code" in df_all.columns and "obs_date" in df_all.columns:
            df_all = df_all.sort_values("obs_day").drop_duplicates(subset=["ts_code", "obs_date"], keep="first")
        df_all = score_stocks(df_all, "sell_score")
        print(f"  候选: {len(df_all)} 行 (LEGACY full_test), 交易日: {len(trading_days)}")
    else:
        print(f"  候选: {len(df_all)} 行 (prediction_store), 交易日: {len(trading_days)}, pred_lookup: {len(pred_lookup)}")

    sorted_td = sorted(trading_days)
    dates = list(sorted_td)
    if start_date is not None:
        sd = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        dates = [d for d in dates if d >= sd]
    if end_date is not None:
        ed = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
        dates = [d for d in dates if d <= ed]

    print(f"  运行日期: {len(dates)} 日 ({dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')})\n")

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

    return {
        "results": results,
        "final_holdings": holdings,
        "base_dir": _base,
    }


def main():
    parser = argparse.ArgumentParser(description="策略执行引擎 — 回测模式")
    parser.add_argument("--profile", type=str, default=None,
                        help="运行 profile (如 production)")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--no-postprocess", action="store_true", help="跳过后处理")
    args = parser.parse_args()

    result = run_range(
        start_date=args.start_date,
        end_date=args.end_date,
        profile=args.profile,
        write_ledgers=True,
        postprocess=not args.no_postprocess,
    )

    print(f"\n{'='*70}")
    print(f"  运行完成: ledger={result['base_dir']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
