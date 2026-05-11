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
    后处理: 生成净值曲线 (live_equity_curve.csv) + 交易报告 (live_trade_report.csv)

Inputs:
    - stop_experiment/output/predictions/T.parquet (当日真实预测)
    - stop_experiment/output/holdings/T-1.parquet (前日持仓)
    - stop_experiment/output/full_test_predictions.parquet (全量预测，replay 模式)
    - DB: stock_k_data (K线数据)

Outputs:
    - stop_experiment/output/live/executions/T.parquet
    - stop_experiment/output/live/decisions/T.parquet
    - stop_experiment/output/holdings/T.parquet
    - stop_experiment/output/live/live_equity_curve.csv (后处理: 净值曲线)
    - stop_experiment/output/live/live_trade_report.csv (后处理: 交易盈亏报告)

How to Run:
    # 单日运行 (live 模式：只读当日预测账本)
    python stop_experiment/pipeline/09_paper_trading_runner.py --date 2026-05-08 --mode live

    # 单日运行 (replay 模式：读全量预测)
    python stop_experiment/pipeline/09_paper_trading_runner.py --date 2026-05-08

    # 10 日批量回放
    python stop_experiment/pipeline/09_paper_trading_runner.py --batch-first-10

    # 全量
    python stop_experiment/pipeline/09_paper_trading_runner.py --batch-all

    # 故障注入
    python stop_experiment/pipeline/09_paper_trading_runner.py --date 2026-03-20 --fault-inject MISSING_PREDICTION

Side Effects:
    - 写 decisions/executions/holdings 三本账
    - 写 live_equity_curve.csv (净值曲线)
    - 写 live_trade_report.csv (交易报告)
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
    OUTPUT_DIR, PRODUCTION_PARAMS,
    PREDICTIONS_DIR, HOLDINGS_DIR, DECISIONS_DIR, EXECUTIONS_DIR,
)
from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data
from stop_experiment.backtest.daily_state_machine import step_day
from stop_experiment.backtest.simple_backtest import score_stocks
from stop_experiment.pipeline.live_ledger import load_holdings, save_holdings, save_decisions, save_executions
from stop_experiment.pipeline.build_live_equity_curve import build_live_equity_curve, save_live_equity_curve
from stop_experiment.pipeline.build_live_trade_report import build_live_trade_report, save_live_trade_report

DEFAULT_MAX_STOCKS = PRODUCTION_PARAMS.get("max_stocks", 10)

FIRST_10_DATES = [
    "2026-03-20", "2026-03-24", "2026-04-03", "2026-04-10", "2026-04-17",
    "2026-04-24", "2026-04-30", "2026-05-06",
]


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
    #    holdings=None 时由后续 _load_latest_holdings 回溯加载，不在此 raise
    if holdings is None:
        trading_days_sorted = sorted(price_pivot.index)
        if len(trading_days_sorted) > 0 and target_date != trading_days_sorted[0]:
            anomalies.append("holdings_none_will_backfill")

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
    if holdings_input is None and pending_buys_input is None:
        prev_dt = _resolve_prev_decision_date(target_date, trading_days, pred_lookup)
        if prev_dt is not None:
            holdings = _load_latest_holdings(target_date, trading_days) or {}
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

    # Step 2: 执行 pending orders（统一日状态推进器 SSOT step_day）
    # 构建候选人池
    candidates = _build_candidate_pool(target_date, df_all, trading_days, pred_lookup)
    if candidates.empty:
        anomalies.append("no_candidates_today")
        print(f"  [候选池] 空 (no_candidates_today)")

    # 计算前日决策日
    prev_dt = _resolve_prev_decision_date(target_date, trading_days, pred_lookup)

    # 计算 day_close 和 day_open_next
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
    if day_open_next.empty or day_open_next.isna().all():
        day_open_next = None
        print(f"  [信息] T+1 开盘价不可用（最后一日或数据缺失），不生成买入单")

    # 统一日状态推进（SSOT）
    step_params = {
        "max_stocks": DEFAULT_MAX_STOCKS,
        "max_hold_days": PRODUCTION_PARAMS.get("max_hold_days", 20),
        "stop_loss": PRODUCTION_PARAMS.get("stop_loss", -0.07),
        "exit_threshold": PRODUCTION_PARAMS.get("buy_cls_exit_threshold", 0.70),
    }
    step_result = step_day(
        target_date, holdings, pending_buys_prev, pending_sells_prev,
        price_pivot, candidates, pred_lookup, prev_dt, step_params,
        prev_close_map=prev_close_map, strict=True,
        day_open_next=day_open_next,
    )

    executed_buys = step_result["executed_buys"]
    executed_sells = step_result["executed_sells"]
    skipped_buys = step_result["skipped_buys"]
    skipped_sells = step_result["skipped_sells"]
    holdings = step_result["holdings"]
    pending_buys = step_result["pending_buys"]
    pending_sells = step_result["pending_sells"]
    sell_reasons = step_result["sell_reasons"]

    # 采集 skip 异常
    for _sb in skipped_buys:
        anomalies.append(f"skip_buy:{_sb[5]}:{_sb[0]}")
    for _ss in skipped_sells:
        anomalies.append(f"skip_sell:{_ss.get('reason', 'unknown')}:{_ss.get('code', '?')}")

    # Step 2+: 写执行账本
    save_executions(target_date, executed_buys, executed_sells, skipped_buys, skipped_sells)

    # Step 4+: 写决策账本
    save_decisions(target_date, holdings, pending_buys, pending_sells, sell_reasons,
                   extra=step_result.get("extra", {}), day_close=day_close)

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


def _resolve_prev_decision_date(target_date, trading_days, pred_lookup):
    """
    向前回溯最多 3 个交易日，找到 pred_lookup 有预测的最近交易日期。
    确保 decide_eod 中 find_exit_pred(sid, prev_date) 精确匹配，不会因某日缺预测文件而全跳过退出检查。
    """
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


def _load_latest_holdings(target_date, trading_days):
    """
    从 target_date 向前回溯（最多 10 个交易日），查找最近存在的持仓文件
    """
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
        h = load_holdings(prev_dt)
        if h is not None:
            print(f"  [持仓回溯] {target_date.strftime('%Y-%m-%d')} ← {prev_dt.strftime('%Y-%m-%d')} ({len(h)} 只)")
            return h
    print(f"  [持仓回溯] {target_date.strftime('%Y-%m-%d')} ← 无 (回溯 10 日内无持仓文件)")
    return None


# ==================== 批量运行 ====================

def run_batch(dates, df_all, price_pivot, trading_days, prev_close_map, pred_lookup):
    """批量运行，状态在日期间传递"""
    results = []
    holdings = None
    pending_buys = None
    pending_sells = None

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
    parser.add_argument("--batch-all", action="store_true", help="遍历全部交易日")
    parser.add_argument("--mode", choices=["replay", "live"], default="replay",
                        help="replay=读全量预测(默认) | live=只读真实预测账本")
    parser.add_argument("--fault-inject", type=str, default=None,
                        choices=["MISSING_PREDICTION", "MISSING_HOLDINGS", "MISSING_OPEN_PRICE"],
                        help="故障注入测试")
    parser.add_argument("--start-date", type=str, default=None,
                        help="批量运行起始日期 (YYYY-MM-DD)")
    args = parser.parse_args()

    print("=" * 70)
    print("  模拟盘主入口 — 5 步日流程")
    print(f"  模式: {args.mode}")
    print("=" * 70)

    # 加载候选池数据
    if args.mode == "replay":
        cand_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
        if not os.path.exists(cand_path):
            raise FileNotFoundError(f"replay 模式需要 {cand_path}，请先生成全量预测")
        df_all = pd.read_parquet(cand_path)
        df_all["obs_date"] = pd.to_datetime(df_all["obs_date"])
        candidate_obs_days = PRODUCTION_PARAMS.get("candidate_obs_days", [1])
        df_all = df_all[df_all["obs_day"].isin(candidate_obs_days)].copy()
        df_all = score_stocks(df_all, "sell_score")
        print(f"  候选: {len(df_all)} 行 (全量预测)")

        test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data(
            candidate_obs_days=PRODUCTION_PARAMS["candidate_obs_days"]
        )
    else:
        # live 模式：只读当日预测账本，pred_lookup 从实时预测构建
        if not args.date:
            raise RuntimeError("live 模式必须指定 --date (单日)")
        pred_path = os.path.join(PREDICTIONS_DIR, f"{args.date}.parquet")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"live 模式缺少预测账本: {pred_path}")
        df_all = pd.read_parquet(pred_path)
        if "obs_date" not in df_all.columns and "pred_date" in df_all.columns:
            df_all["obs_date"] = pd.to_datetime(df_all["pred_date"])
        elif "obs_date" not in df_all.columns:
            df_all["obs_date"] = pd.to_datetime(args.date)
        if "obs_day" not in df_all.columns:
            df_all["obs_day"] = 1
        candidate_obs_days = PRODUCTION_PARAMS.get("candidate_obs_days", [1])
        df_all = df_all[df_all["obs_day"].isin(candidate_obs_days)].copy()
        if "score" not in df_all.columns:
            df_all = score_stocks(df_all, "sell_score")
        print(f"  候选: {len(df_all)} 行 (实时预测账本)")

        # 从实时预测构建 pred_lookup（替代 full_test_predictions.parquet）
        pred_lookup = {}
        for _, row in df_all.iterrows():
            key = (int(row["signal_id"]), row["obs_date"])
            pred_lookup[key] = {
                "pred_buy_cls": float(row.get("pred_buy_cls", np.nan)),
                "pred_sell_reg": float(row.get("pred_sell_reg", np.nan)),
                "composite_score": float(row.get("score", np.nan)),
            }

        # 价格数据直接从 DB 加载（不依赖 full_test_predictions.parquet）
        from stop_experiment.backtest.simple_backtest import load_daily_prices, build_price_pivot
        daily = load_daily_prices("2024-01-01", "2027-01-01")
        price_pivot, trading_days, prev_close_map = build_price_pivot(daily)
        print(f"  交易日: {len(trading_days)}, pred_lookup: {len(pred_lookup)} 条")

        # 补充前日预测账本，保证 decide_eod 的 find_exit_pred(sid, prev_date) 可匹配
        target_dt = pd.to_datetime(args.date)
        if target_dt in trading_days:
            tidx = trading_days.index(target_dt)
            for lookback in range(1, 4):
                if tidx - lookback < 0:
                    break
                prev_tday = trading_days[tidx - lookback]
                prev_pred_path = os.path.join(PREDICTIONS_DIR, f"{prev_tday.strftime('%Y-%m-%d')}.parquet")
                if os.path.exists(prev_pred_path):
                    prev_pred = pd.read_parquet(prev_pred_path)
                    if "obs_date" not in prev_pred.columns and "pred_date" in prev_pred.columns:
                        prev_pred["obs_date"] = pd.to_datetime(prev_pred["pred_date"])
                    merged = 0
                    for _, row in prev_pred.iterrows():
                        key = (int(row["signal_id"]), row["obs_date"])
                        if key not in pred_lookup:
                            pred_lookup[key] = {
                                "pred_buy_cls": float(row.get("pred_buy_cls", np.nan)),
                                "pred_sell_reg": float(row.get("pred_sell_reg", np.nan)),
                                "composite_score": float(row.get("score", np.nan)),
                            }
                            merged += 1
                    print(f"  [pred_lookup] 已合并 {prev_tday.strftime('%Y-%m-%d')} 预测: {prev_pred_path} (合并 {merged} 条)")
                    break
            else:
                print(f"  [pred_lookup] 回溯 3 个交易日内无前日预测文件 (退出检查可能缺精度)")
        else:
            print(f"  [pred_lookup] {args.date} 非交易日，跳过前日回溯")

    dates = []
    if args.date:
        dates = [pd.to_datetime(args.date)]
    elif args.batch_first_10:
        dates = [pd.to_datetime(d) for d in FIRST_10_DATES]
    elif args.batch_all:
        dates = sorted(trading_days)
        if args.start_date:
            start = pd.to_datetime(args.start_date)
            dates = [d for d in dates if d >= start]
            print(f"  起始日期: {start.strftime('%Y-%m-%d')}")
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

    # === 后处理：生成净值曲线 + 交易报告 ===
    print(f"\n{'='*70}")
    print(f"后处理: 净值曲线 + 交易报告")
    eq_df = build_live_equity_curve(price_pivot)
    save_live_equity_curve(eq_df)
    report_df, trade_summary = build_live_trade_report()
    save_live_trade_report(report_df, trade_summary)


if __name__ == "__main__":
    main()