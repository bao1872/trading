#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态退出稳健性收口验证引擎 v2 (SSOT)

Purpose:
    研究用回测引擎（实验参数探索）。
    buy-only 动态退出（buy_cls > 0.7）+ 月度切片 + 三类退出对比 + 多日试探。

    引擎特性:
    - 退出模式: model_exit（pred_buy_cls > 0.70）+ stop_loss（-7%）+ max_hold（20d）
    - 决策: T 日收盘后决策，T+1 日开盘执行买卖
    - 买入: T 日决策 → T+1 开盘买入，等权分配
    - 卖出: T 日决策卖出 → T+1 开盘执行（不再 T 日执行）
    - 成本: 买入 0.05% + 卖出 0.10%
    - 候选池: obs_day=1（生产口径，经聚合实验验证最优）

    退出版本:
    - fixed_hold : 固定持有 hold_days 天后卖出 (baseline A)
    - rule_exit  : 止盈+10% / 止损-7% / max_hold=20 (baseline B)
    - model_exit : pred_buy_cls_prev > 0.7 / 止损-7% / max_hold=20 (C)

    运行模式 (--mode):
    - single    : 单组回测对比
    - slice     : 月度切片验证
    - three-way : 三类退出全参数对比 (A vs B vs C)
    - multi-day : obs_day=1 only vs 1~3 候选池试探

    重要声明:
    - run_backtest 为本模块内部函数，仅供 single/slice/three-way/multi-day 研究模式使用
    - 生产链路（回测/回放/模拟盘）请使用 stop_experiment.engine.strategy_runner.run_range()
    - 禁止从外部模块 import run_backtest

Pipeline Position:
    研究用回测引擎（实验参数探索，非生产链路）。
    上游: full_test_predictions.parquet, DB
    下游: 无（研究脚本自含）

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - DB: stock_k_data

Outputs:
    - stop_experiment/output/backtest/dynamic/
      ├── slice_validation.csv
      ├── three_way_comparison.csv
      ├── multi_day_candidate.csv
      └── figures/

How to Run:
    python stop_experiment/backtest/dynamic_exit_backtest_v2.py --mode single --top-k 10
    python stop_experiment/backtest/dynamic_exit_backtest_v2.py --mode slice --top-k 10
    python stop_experiment/backtest/dynamic_exit_backtest_v2.py --mode three-way
    python stop_experiment/backtest/dynamic_exit_backtest_v2.py --mode multi-day

Side Effects:
    - 只读DB和parquet，输出回测结果
"""

from __future__ import annotations

import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, BACKTEST_DIR, BUY_COST, SELL_COST,
)
from stop_experiment.backtest.simple_backtest import (
    load_daily_prices, build_price_pivot,
    is_limit_up, is_limit_down, is_suspended,
    score_stocks, compute_summary,
)

from stop_experiment.backtest.decision_core import (
    find_exit_pred,
    evaluate_model_exit,
    decide_eod,
)

DYNAMIC_DIR = os.path.join(BACKTEST_DIR, "dynamic")

# ---- 固定参数 (网格搜索确定) ----
BUY_CLS_EXIT_THRESHOLD = 0.7
STOP_LOSS = -0.07
MAX_HOLD_DAYS = 20
TAKE_PROFIT = 0.12


# ---- 权重分配辅助函数 ----

def _calc_weights(codes, scores, mode, params):
    """计算买入时的目标权重（归一化，不做 cap/floor）

    mode: None(等权), by_rank(排名分层), by_score(线性映射, 已关闭)
    生产仅用 None 和 by_rank(W1a研究)。
    by_score 线已于 2026-05-10 关闭，仅保留代码不删除。

    codes: 本次买入的 code 列表（已排序，索引0=rank1）
    scores: 对应 score 列表（用于 by_score 模式）
    mode: None | "by_rank" | "by_score"
    params: dict，格式取决于 mode

    Returns: {code: weight} dict，总和=1.0
    """
    n = len(codes)
    if n == 0:
        return {}

    if mode is None:
        w = 1.0 / n
        return {c: w for c in codes}

    if mode == "by_rank":
        tiers = params.get("tiers", []) if params else []
        raw = {}
        for i, code in enumerate(codes):
            rank = i + 1
            mult = 1.0
            for lo, hi, m in tiers:
                if lo <= rank <= hi:
                    mult = m
                    break
            raw[code] = mult
        total = sum(raw.values())
        return {c: w / total for c, w in raw.items()} if total > 0 else {c: 1.0 / n for c in codes}

    if mode == "by_score":
        lo = (params or {}).get("lo", 0.75)
        hi = (params or {}).get("hi", 1.25)
        s = np.array(scores)
        s_min, s_max = s.min(), s.max()
        if s_max - s_min < 1e-12:
            w = 1.0 / n
            return {c: w for c in codes}
        normalized = (s - s_min) / (s_max - s_min)
        mapped = lo + (hi - lo) * normalized
        mapped = np.clip(mapped, 0.01, None)  # floor at 1% to avoid zero weight
        total = mapped.sum()
        return {c: float(w / total) for c, w in zip(codes, mapped)} if total > 0 else {c: 1.0 / n for c in codes}

    return {c: 1.0 / n for c in codes}


# ---- 数据加载 ----
def _load_data(candidate_obs_days=None):
    """加载全量预测和K线"""
    full_pred_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    if not os.path.exists(full_pred_path):
        raise FileNotFoundError(f"{full_pred_path} 不存在，请先运行 generate_full_predictions.py")

    full_pred = pd.read_parquet(full_pred_path)
    full_pred["obs_date"] = pd.to_datetime(full_pred["obs_date"])

    # 构建预测查找键
    prediction_lookup = {}
    for _, row in full_pred.iterrows():
        key = (int(row["signal_id"]), row["obs_date"])
        prediction_lookup[key] = {
            "pred_buy_cls": float(row.get("pred_buy_cls", np.nan)),
            "pred_sell_reg": float(row.get("pred_sell_reg", np.nan)),
            "pred_sell_cls": float(row.get("pred_sell_cls", np.nan)),
            "pred_buy_reg": float(row.get("pred_buy_reg", np.nan)),
            "composite_score": float(row.get("composite_score", np.nan)) if "composite_score" in row else np.nan,
        }

    # 买入候选池
    if candidate_obs_days is None:
        candidate_obs_days = [1]
    test_df = full_pred[full_pred["obs_day"].isin(candidate_obs_days)].copy()

    if test_df.empty:
        raise ValueError("无可回测数据")

    signal_start = str(test_df["obs_date"].min().date())
    signal_end_dt = test_df["obs_date"].max() + pd.Timedelta(days=60)
    signal_end = str(signal_end_dt.date())
    daily_prices = load_daily_prices(signal_start, signal_end)
    price_pivot, trading_days, prev_close_map = build_price_pivot(daily_prices)

    return test_df, price_pivot, trading_days, prev_close_map, prediction_lookup


# ---- 退出评估函数 ----
# find_exit_pred / evaluate_model_exit 从 decision_core.py 导入
# 严格精确匹配（无回退），保证回测和 replay 口径一致


def evaluate_rule_exit(holding, code, day_close, take_profit):
    """
    普通规则退出:
    - current_ret >= take_profit → take_profit
    - current_ret < stop_loss → stop_loss (由调用方处理)
    - days_held > max_hold → max_hold (由调用方处理)
    """
    if code in day_close.index:
        cp = day_close[code]
        if not np.isnan(cp) and cp > 0:
            ret = (cp - holding["buy_price"]) / holding["buy_price"]
            if ret >= take_profit:
                return True, "take_profit"
    return False, ""


# ---- 通用回测引擎 ----

def run_backtest(
    signals_df, price_pivot, trading_days, prev_close_map,
    pred_lookup, max_stocks=20, strategy="sell_score",
    exit_mode="fixed_hold", hold_days=5,
    max_hold_days=MAX_HOLD_DAYS, stop_loss=STOP_LOSS,
    take_profit=TAKE_PROFIT, strict=True,
    buy_cost=None, sell_cost=None,
    buy_cls_exit_threshold=0.70,
    exit_sub_mode=None, buy_reg_exit_threshold=None,
    debug_snapshots=False,
    weight_mode=None, weight_params=None,
):
    """
    [内部专用] 研究用回测引擎 — 仅供本模块 single/slice/three-way/multi-day 模式调用。
    生产链路请使用 stop_experiment.engine.strategy_runner.run_range(mode="backtest")。
    禁止从外部模块 import 此函数。

    exit_mode 控制退出策略:
    - fixed_hold
    - rule_exit
    - model_exit (支持 exit_sub_mode: None/X1 / "sell_decay"/X3 / "or_buy_reg"/X4)

    weight_mode: None(等权) | "by_rank"(排名分层) | "by_score"(分数线性映射)
    weight_params: dict，格式取决于 weight_mode
    debug_snapshots=True 时返回逐日快照，用于与逐日推理脚本对账。
    """
    signals_df = score_stocks(signals_df, strategy)
    signals_sorted = signals_df.sort_values(["obs_date", "score"], ascending=[True, False])

    signal_dates = sorted(signals_df["obs_date"].unique())
    signal_by_date = {}
    for date in signal_dates:
        day_sigs = signals_sorted[signals_sorted["obs_date"] == date]
        signal_by_date[date] = day_sigs.drop_duplicates(subset=["ts_code"], keep="first")

    holdings = {}
    pending_orders = []       # T日决策买入，T+1执行
    pending_sells = []        # T日决策卖出，T+1执行
    trade_details = []
    nav_records = []
    skipped = defaultdict(int)
    snapshots = []
    holdings_beginning_of_day = {}

    for t_idx, current_date in enumerate(trading_days):
        if current_date not in price_pivot.index:
            continue

        # day_open/day_close 对应当前交易日（执行日）的行情
        day_open = price_pivot.loc[current_date, "open"] if "open" in price_pivot else pd.Series(dtype=float)
        day_close = price_pivot.loc[current_date, "close"] if "close" in price_pivot else pd.Series(dtype=float)

        # ========== 执行阶段（T-1日决策，T日执行） ==========

        # Step 1: 执行pending买单（T-1日决策买入，今日开盘执行）
        if pending_orders:
            executed = []
            for code, bp, ts_code, sc, sid in pending_orders:
                if code in holdings:
                    skipped["already_held"] += 1
                    continue
                if strict:
                    if "volume" in price_pivot and code in price_pivot["volume"].columns:
                        vol_c = price_pivot["volume"][code].get(current_date, np.nan)
                        if is_suspended(vol_c):
                            skipped["suspended"] += 1
                            continue
                    if code in prev_close_map and current_date in prev_close_map[code].index:
                        prev_c = prev_close_map[code].get(current_date, np.nan)
                        if not np.isnan(prev_c) and prev_c > 0:
                            if "high" in price_pivot:
                                dh = price_pivot["high"][code]
                                if current_date in dh.index:
                                    if is_limit_up(bp, dh.get(current_date, np.nan), prev_c):
                                        skipped["limit_up"] += 1
                                        continue
                executed.append((code, bp, ts_code, sc, sid))

            if executed:
                all_codes = list(holdings.keys()) + [code for code, _, _, _, _ in executed]
                all_scores = [holdings[code_h].get("weight_score", holdings[code_h]["score"])
                              for code_h in holdings]
                all_scores += [ws_map.get(sid, sc) for _, _, _, sc, sid in executed]
                scored = sorted(zip(all_codes, all_scores), key=lambda x: x[1], reverse=True)
                sorted_codes = [c for c, _ in scored]
                sorted_scores = [s for _, s in scored]
                weights = _calc_weights(sorted_codes, sorted_scores, weight_mode, weight_params)
                for code_h in holdings:
                    holdings[code_h]["weight"] = weights.get(code_h, 0)
                for code, bp, ts_code, sc, sid in executed:
                    ws = ws_map.get(sid, sc)
                    holdings[code] = {
                        "buy_date": current_date, "buy_price": bp,
                        "weight": weights.get(code, 0), "days_held": 0,
                        "ts_code": ts_code, "score": sc, "signal_id": sid,
                        "weight_score": ws,
                        "entry_weight": weights.get(code, 0),
                    }
            pending_orders = []

        # Step 1.5: 执行pending卖单（T-1日决策卖出，今日开盘执行）
        if pending_sells:
            for sell_item in pending_sells:
                code = sell_item["code"]
                h = sell_item["holding"]
                sell_price = np.nan
                if code in day_open.index and not np.isnan(day_open[code]):
                    sell_price = day_open[code]
                if np.isnan(sell_price) or sell_price <= 0:
                    skipped["no_sell_price"] += 1
                    continue

                if strict:
                    if "volume" in price_pivot and code in price_pivot["volume"].columns:
                        vol_c = price_pivot["volume"][code].get(current_date, np.nan)
                        if is_suspended(vol_c):
                            skipped["suspended_sell"] += 1
                            continue
                    if code in prev_close_map and current_date in prev_close_map[code].index:
                        prev_c = prev_close_map[code].get(current_date, np.nan)
                        if not np.isnan(prev_c) and prev_c > 0:
                            if "low" in price_pivot:
                                dl = price_pivot["low"][code]
                                if current_date in dl.index and not np.isnan(dl[current_date]):
                                    if is_limit_down(sell_price, dl[current_date], prev_c):
                                        skipped["limit_down"] += 1
                                        continue

                gross_ret = (sell_price - h["buy_price"]) / h["buy_price"]
                _buy_cost = buy_cost if buy_cost is not None else BUY_COST
                _sell_cost = sell_cost if sell_cost is not None else SELL_COST
                net_ret = gross_ret - _buy_cost - _sell_cost

                trade_details.append({
                    "ts_code": h["ts_code"], "buy_date": h["buy_date"],
                    "sell_date": current_date, "buy_price": h["buy_price"],
                    "sell_price": sell_price, "hold_days": h["days_held"],
                    "gross_ret": gross_ret, "net_ret": net_ret,
                    "sell_reason": sell_item["reason"],
                    "score": h.get("score", 0),
                    "strategy": strategy, "exit_mode": exit_mode,
                    "weight": h.get("entry_weight", h.get("weight", 0)),
                })
                if code in holdings:
                    del holdings[code]

            pending_sells = []

        # Step 2.5: 补充 MAE/MFE 到所有已完成交易（卖出执行后补充）
        for trade in trade_details:
            if "mae" not in trade:
                trade["mae"] = np.nan
            if "mfe" not in trade:
                trade["mfe"] = np.nan

        # ---- debug: 捕获当日开盘后持仓 (pending_orders/pending_sells 已执行) ----
        if debug_snapshots:
            holdings_beginning_of_day = {k: dict(v) for k, v in holdings.items()}

        # ========== 决策阶段（T日收盘后，决策T+1日操作） ==========
        # Step 2~4: 统一日终决策（SSOT decide_eod）
        prev_date = trading_days[t_idx - 1] if t_idx > 0 else None
        next_idx = t_idx + 1
        day_open_next = price_pivot.loc[trading_days[next_idx], "open"] if next_idx < len(trading_days) else pd.Series(dtype=float)

        holdings, pending_buys_new, pending_sells_new, sell_reasons, _extra_new = decide_eod(
            decision_date=current_date,
            holdings=holdings,
            candidates=signal_by_date.get(current_date, pd.DataFrame()),
            pred_lookup=pred_lookup,
            prev_date=prev_date,
            day_close=day_close,
            day_open_next=day_open_next,
            max_stocks=max_stocks,
            max_hold_days=max_hold_days,
            stop_loss=stop_loss,
            exit_threshold=buy_cls_exit_threshold,
            exit_sub_mode=exit_sub_mode,
            buy_reg_exit_threshold=buy_reg_exit_threshold,
            prev_decision_date=trading_days[t_idx - 2] if t_idx >= 2 else None,
        )

        pending_sells = pending_sells_new
        pending_orders = pending_buys_new

        # 构建 weight_score 映射（用于 by_score 权重模式下的替代分数）
        ws_map = {}
        if weight_mode == "by_score" and "weight_score" in signals_df.columns:
            for _, row in signals_df.iterrows():
                ws_map[row["signal_id"]] = row["weight_score"]

        # 按 exit_mode 过滤卖出类型（修复：之前 decide_eod 总是评估全部三种退出）
        if exit_mode == "fixed_hold":
            pending_sells = [s for s in pending_sells if s["reason"] == "max_hold"]
        elif exit_mode == "rule_exit":
            pending_sells = [s for s in pending_sells if s["reason"] != "model_risk"]
        # else: model_exit — 保持全部 pending_sells

        # Step 3: NAV（与 decide_eod 前完全一致，holdings 中 days_held 已递增）
        daily_ret = 0.0
        for code, h in holdings.items():
            if code in day_close.index and not np.isnan(day_close[code]):
                if h["days_held"] == 1:
                    if code in day_open.index and not np.isnan(day_open[code]):
                        sr = (day_close[code] - day_open[code]) / day_open[code]
                    else:
                        sr = 0
                elif t_idx > 0:
                    prev_d = trading_days[t_idx - 1]
                    if prev_d in price_pivot.index:
                        prev_c = price_pivot.loc[prev_d, "close"]
                        if code in prev_c.index and not np.isnan(prev_c[code]):
                            sr = (day_close[code] - prev_c[code]) / prev_c[code]
                        else:
                            sr = 0
                    else:
                        sr = 0
                else:
                    sr = 0
                daily_ret += h["weight"] * sr

        prev_nav = nav_records[-1]["nav"] if nav_records else 1.0
        nav = prev_nav * (1 + daily_ret)

        nav_records.append({
            "date": current_date, "nav": nav,
            "daily_ret": daily_ret, "n_positions": len(holdings),
        })

        # ---- debug: 记录当日快照 (decide_eod 之后, 含当日新 pending buys/sells) ----
        if debug_snapshots:
            day_candidates = signal_by_date.get(current_date, pd.DataFrame())
            day_candidate_codes = list(day_candidates["ts_code"]) if not day_candidates.empty else []

            snapshots.append({
                "date": current_date,
                "candidate_codes": day_candidate_codes,
                "candidates_df": day_candidates.reset_index(drop=True) if not day_candidates.empty else day_candidates,
                "holdings_before": holdings_beginning_of_day,
                "to_sell": [item["code"] for item in pending_sells],  # 决策卖出，下一日执行
                "sell_reasons": sell_reasons,  # decide_eod 当日决策的卖出原因
                "buys": [{
                    "ts_code": o[2], "code": o[0], "buy_price": o[1],
                    "score": o[3], "signal_id": o[4],
                } for o in pending_orders],
                "holdings_after": {k: dict(v) for k, v in holdings.items()},
                "nav": nav,
            })

    nav_df = pd.DataFrame(nav_records)
    trades_df = pd.DataFrame(trade_details)
    return {
        "nav_df": nav_df, "trades_df": trades_df, "skipped_stats": dict(skipped),
        "params": {
            "max_stocks": max_stocks, "hold_days": hold_days,
            "strategy": strategy, "exit_mode": exit_mode,
            "max_hold_days": max_hold_days, "stop_loss": stop_loss,
            "take_profit": take_profit, "strict": strict,
        },
        "snapshots": snapshots,
    }


# ---- 模式1: 单组回测 ----

def run_single(args):
    print("单组回测对比")
    print("=" * 50)

    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data([1])
    print(f"  候选: {len(test_df)}, 交易日: {len(trading_days)}")

    exit_opts = [
        ("fixed_hold", args.hold_days, 0.0),
        ("rule_exit", 0, args.take_profit),
        ("model_exit", 0, 0.0),
    ]

    for exit_mode, hd, tp in exit_opts:
        result = run_backtest(
            test_df, price_pivot, trading_days, prev_close_map, pred_lookup,
            max_stocks=args.top_k, strategy=args.strategy,
            exit_mode=exit_mode, hold_days=hd, take_profit=tp,
            strict=True,
        )
        s = compute_summary(result)

        reasons_str = ""
        tdf = result.get("trades_df", pd.DataFrame())
        if not tdf.empty and "sell_reason" in tdf.columns:
            rc = tdf["sell_reason"].value_counts().to_dict()
            reasons_str = " | " + " ".join(f"{k}:{v}" for k, v in rc.items())

        print(f"  {exit_mode:12s} nav={s.get('final_nav', 0):.4f} "
              f"sharpe={s.get('sharpe', 0):.2f} win={s.get('win_rate', 0):.2%} "
              f"avg_net={s.get('avg_net_ret', 0):.2%} "
              f"avg_hold={s.get('avg_hold_days', 0):.1f}d "
              f"n={s.get('n_trades', 0)}{reasons_str}")


# ---- 模式2: 月度切片验证 ----

def run_time_slice_validation(args):
    print("=" * 60)
    print("月度切片验证: 固定持有 vs 模型退出")
    print("=" * 60)

    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data([1])
    test_df["month"] = test_df["obs_date"].dt.to_period("M")

    months = sorted(test_df["month"].unique())
    print(f"  月份: {[str(m) for m in months]}, 交易日: {len(trading_days)}")

    rows = []
    # 全窗口
    all_windows = list(months) + ["__ALL__"]

    for month in all_windows:
        if month == "__ALL__":
            sub_signals = test_df
            label = "全窗口"
            sub_td = trading_days
        else:
            month_mask = test_df["month"] == month
            sub_signals = test_df[month_mask]
            label = str(month)

            m_start = sub_signals["obs_date"].min()
            m_end_plus = sub_signals["obs_date"].max() + pd.Timedelta(days=60)
            sub_td = [d for d in trading_days if m_start <= d <= m_end_plus]

        if sub_signals.empty or not sub_td:
            continue

        for exit_mode in ["fixed_hold", "model_exit"]:
            for hd in [5, 10]:
                if exit_mode == "model_exit" and hd != 5:
                    continue  # hold_days only matters for fixed

                result = run_backtest(
                    sub_signals, price_pivot, sub_td, prev_close_map, pred_lookup,
                    max_stocks=args.top_k, strategy=args.strategy,
                    exit_mode=exit_mode, hold_days=hd, strict=True,
                )
                s = compute_summary(result)

                tdf = result.get("trades_df", pd.DataFrame())
                reasons = {}
                if not tdf.empty and "sell_reason" in tdf.columns:
                    reasons = tdf["sell_reason"].value_counts().to_dict()

                rows.append({
                    "window": label, "exit_mode": exit_mode,
                    "hold_days": hd,
                    "final_nav": s.get("final_nav", np.nan),
                    "sharpe": s.get("sharpe", np.nan),
                    "max_dd": s.get("max_dd", np.nan),
                    "win_rate": s.get("win_rate", np.nan),
                    "avg_net_ret": s.get("avg_net_ret", np.nan),
                    "avg_hold_days": s.get("avg_hold_days", np.nan),
                    "n_trades": s.get("n_trades", 0),
                    "n_model_risk": reasons.get("model_risk", 0),
                    "n_stop_loss": reasons.get("stop_loss", 0),
                    "n_max_hold": reasons.get("max_hold", 0),
                    "n_take_profit": reasons.get("take_profit", 0),
                    "n_fixed": reasons.get("fixed", 0),
                })

                mrk = "M" if exit_mode == "model_exit" else "F"
                print(f"  {label:10s} {mrk} h={hd:2d} nav={s.get('final_nav', 0):.4f} "
                      f"sharpe={s.get('sharpe', 0):.2f} win={s.get('win_rate', 0):.2%} "
                      f"n={s.get('n_trades', 0)} avg_hold={s.get('avg_hold_days', 0):.1f}d")

    slice_df = pd.DataFrame(rows)
    path = os.path.join(DYNAMIC_DIR, "slice_validation.csv")
    slice_df.to_csv(path, index=False)
    print(f"\n  保存: {path}")

    # 打印对比摘要
    print(f"\n{'='*80}")
    print("月度切片对比: 固定 h=5 vs 动态退出 (nav)")
    print(f"{'='*80}")
    for month in months:
        f_row = [r for r in rows if r["window"] == str(month) and r["exit_mode"] == "fixed_hold" and r["hold_days"] == 5]
        m_row = [r for r in rows if r["window"] == str(month) and r["exit_mode"] == "model_exit"]
        fn = f_row[0]["final_nav"] if f_row else np.nan
        mn = m_row[0]["final_nav"] if m_row else np.nan
        diff = mn - fn if pd.notna(fn) and pd.notna(mn) else np.nan
        fn_s = f"{fn:.4f}" if pd.notna(fn) else "N/A"
        mn_s = f"{mn:.4f}" if pd.notna(mn) else "N/A"
        d_s = f"{diff:+.4f}" if pd.notna(diff) else "N/A"
        print(f"  {str(month):10s} fixed={fn_s}  model={mn_s}  diff={d_s}")

    # 箱线图
    _plot_slice_boxplots(slice_df, args)


def _plot_slice_boxplots(slice_df, args):
    fig_dir = os.path.join(DYNAMIC_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # nav bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Time Slice Validation ({args.strategy}, k={args.top_k})", fontsize=13)

    windows = sorted(slice_df["window"].unique())
    for idx, exit_mode in enumerate(["fixed_hold", "model_exit"]):
        ax = axes[idx]
        sub = slice_df[slice_df["exit_mode"] == exit_mode]
        nav_by_win = {w: sub[(sub["window"] == w) & (sub["hold_days"] == 5)]["final_nav"].iloc[0]
                       if len(sub[(sub["window"] == w) & (sub["hold_days"] == 5)]) > 0 else np.nan
                       for w in windows}

        names, vals = zip(*[(w, v) for w, v in nav_by_win.items() if pd.notna(v)])
        colors = ["g" if v >= 1 else "r" for v in vals]
        ax.bar(list(names), list(vals), color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"{exit_mode}")
        ax.set_ylabel("Final NAV")
        for i, (n, v) in enumerate(zip(names, vals)):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 1 else "top", fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(fig_dir, "slice_nav.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


# ---- 模式3: 三类退出全参数对比 ----

def run_three_way_comparison(args):
    print("=" * 60)
    print("三类退出基线对比: fixed_hold / rule_exit / model_exit")
    print("=" * 60)

    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data([1])
    print(f"  候选: {len(test_df)}, 交易日: {len(trading_days)}")

    top_k_list = [5, 10, 20]
    strategy_list = ["sell_score", "low_risk", "composite"]
    hold_days_list = [3, 5, 10]
    tp_list = [0.10, 0.12, 0.15]

    all_rows = []
    total = len(top_k_list) * len(strategy_list) * (3 + 3 + 1)  # fixed×3 + rule×3 + model×1
    done = 0

    for strategy in strategy_list:
        for top_k in top_k_list:
            for hd in hold_days_list:
                done += 1
                result = run_backtest(
                    test_df, price_pivot, trading_days, prev_close_map, pred_lookup,
                    max_stocks=top_k, strategy=strategy,
                    exit_mode="fixed_hold", hold_days=hd, strict=True,
                )
                s = compute_summary(result)
                all_rows.append(_make_row(strategy, top_k, hd, 0, "fixed_hold", "fixed_hold", s, result))

            for tp in tp_list:
                done += 1
                result = run_backtest(
                    test_df, price_pivot, trading_days, prev_close_map, pred_lookup,
                    max_stocks=top_k, strategy=strategy,
                    exit_mode="rule_exit", take_profit=tp, strict=True,
                )
                s = compute_summary(result)
                all_rows.append(_make_row(strategy, top_k, 0, tp, "rule_exit", "rule_exit", s, result))

            done += 1
            result = run_backtest(
                test_df, price_pivot, trading_days, prev_close_map, pred_lookup,
                max_stocks=top_k, strategy=strategy,
                exit_mode="model_exit", strict=True,
            )
            s = compute_summary(result)
            all_rows.append(_make_row(strategy, top_k, MAX_HOLD_DAYS, 0, "model_exit", "model_exit", s, result))

            nav_f = [r["final_nav"] for r in all_rows[-4:-1]]  # last 3 fixed
            nav_r = all_rows[-2]["final_nav"]  # last rule
            nav_m = all_rows[-1]["final_nav"]  # last model
            best_f = max(nav_f) if nav_f else 0
            print(f"  {strategy:12s} k={top_k:2d} "
                  f"F(best)={best_f:.4f} R={nav_r:.4f} M={nav_m:.4f} "
                  f"R-F={nav_r-best_f:+.4f} M-R={nav_m-nav_r:+.4f}")

    comp_df = pd.DataFrame(all_rows)
    path = os.path.join(DYNAMIC_DIR, "three_way_comparison.csv")
    comp_df.to_csv(path, index=False)
    print(f"\n  保存: {path}")

    # 关键对比表
    print(f"\n{'='*100}")
    print("关键对比: R(Rule) vs F(Fixed best) vs M(Model)   (R-F < 0 → rule worse, M-R > 0→ model better)")
    print(f"{'='*100}")
    print(f"  {'strategy':12s} {'k':>3s}  {'F_best':>8s} {'R_mean':>8s} {'M':>8s}  "
          f"{'R-F':>8s} {'M-R':>8s} {'M-F':>8s}")

    for strategy in strategy_list:
        for top_k in top_k_list:
            sub = comp_df[(comp_df["strategy"] == strategy) & (comp_df["top_k"] == top_k)]
            f_sub = sub[sub["exit_mode"] == "fixed_hold"]
            r_sub = sub[(sub["exit_mode"] == "rule_exit") & (sub["take_profit"] == TAKE_PROFIT)]
            m_sub = sub[sub["exit_mode"] == "model_exit"]

            fn = f_sub["final_nav"].max() if not f_sub.empty else np.nan
            # R mean across all tp_list
            r_all = sub[sub["exit_mode"] == "rule_exit"]
            r_best = r_all["final_nav"].max() if not r_all.empty else np.nan
            mn = m_sub["final_nav"].iloc[0] if not m_sub.empty else np.nan

            rf = r_best - fn if pd.notna(r_best) and pd.notna(fn) else np.nan
            mr = mn - r_best if pd.notna(mn) and pd.notna(r_best) else np.nan
            mf = mn - fn if pd.notna(mn) and pd.notna(fn) else np.nan

            fn_s = f"{fn:.4f}" if pd.notna(fn) else "N/A"
            r_s = f"{r_best:.4f}" if pd.notna(r_best) else "N/A"
            m_s = f"{mn:.4f}" if pd.notna(mn) else "N/A"
            print(f"  {strategy:12s} {top_k:3d}  {fn_s:>8s} {r_s:>8s} {m_s:>8s}  "
                  f"{rf:+.4f}  {mr:+.4f}  {mf:+.4f}")


def _make_row(strategy, top_k, hold_days, take_profit, exit_mode, exit_label, summary, result):
    tdf = result.get("trades_df", pd.DataFrame())
    reasons = {}
    if not tdf.empty and "sell_reason" in tdf.columns:
        reasons = tdf["sell_reason"].value_counts().to_dict()

    return {
        "strategy": strategy, "top_k": top_k,
        "hold_days": hold_days, "take_profit": take_profit,
        "exit_mode": exit_mode, "exit_label": exit_label,
        **{k: v for k, v in summary.items() if not isinstance(v, dict)},
        "n_model_risk": reasons.get("model_risk", 0),
        "n_stop_loss": reasons.get("stop_loss", 0),
        "n_max_hold": reasons.get("max_hold", 0),
        "n_take_profit": reasons.get("take_profit", 0),
        "n_fixed": reasons.get("fixed", 0),
    }


# ---- 模式4: 多日候选池试探 ----

def run_multi_day_candidate(args):
    print("=" * 60)
    print("买入端试探: obs_day=1 only vs obs_day=1~3")
    print("=" * 60)

    # obs_day=1 only
    df1, pp, td, pm, pl = _load_data([1])
    print(f"  obs_day=1 only: {len(df1)} 候选, {len(td)} 交易日")

    for exit_mode in ["fixed_hold", "model_exit"]:
        for top_k in [5, 10]:
            result = run_backtest(
                df1, pp, td, pm, pl,
                max_stocks=top_k, strategy=args.strategy,
                exit_mode=exit_mode, hold_days=5, strict=True,
            )
            s = compute_summary(result)
            mrk = "M" if exit_mode == "model_exit" else "F"
            print(f"  obs_day=1 {exit_mode:12s} k={top_k:2d} nav={s.get('final_nav', 0):.4f} "
                  f"sharpe={s.get('sharpe', 0):.2f} n={s.get('n_trades', 0)}")

    # obs_day=1~3
    df3, pp3, td3, pm3, pl3 = _load_data([1, 2, 3])
    # 同一 signal_id 每天只保留最新的 obs_day
    df3 = df3.sort_values(["signal_id", "obs_day"], ascending=[True, False])
    df3 = df3.drop_duplicates(subset=["signal_id", "obs_date"], keep="first")
    print(f"\n  obs_day=1~3 (去重后): {len(df3)} 候选, {len(td3)} 交易日")

    for exit_mode in ["fixed_hold", "model_exit"]:
        for top_k in [5, 10]:
            result = run_backtest(
                df3, pp3, td3, pm3, pl3,
                max_stocks=top_k, strategy=args.strategy,
                exit_mode=exit_mode, hold_days=5, strict=True,
            )
            s = compute_summary(result)
            mrk = "M" if exit_mode == "model_exit" else "F"
            tdf = result.get("trades_df", pd.DataFrame())
            # 统计 obs_day 分布
            od_counts = {}
            if not tdf.empty and "obs_day" not in tdf.columns:
                pass
            print(f"  obs_day=1~3 {exit_mode:12s} k={top_k:2d} nav={s.get('final_nav', 0):.4f} "
                  f"sharpe={s.get('sharpe', 0):.2f} n={s.get('n_trades', 0)}")


# ---- main ----

def main(args):
    os.makedirs(os.path.join(DYNAMIC_DIR, "figures"), exist_ok=True)

    mode = args.mode
    if mode == "single":
        run_single(args)
    elif mode == "slice":
        run_time_slice_validation(args)
    elif mode == "three-way":
        run_three_way_comparison(args)
    elif mode == "multi-day":
        run_multi_day_candidate(args)
    else:
        print(f"未知模式: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1.5: 动态退出稳健性收口验证 v2")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "slice", "three-way", "multi-day"],
                        help="运行模式")
    parser.add_argument("--top-k", type=int, default=10, help="最大持仓数")
    parser.add_argument("--hold-days", type=int, default=5, help="固定持有天数")
    parser.add_argument("--strategy", type=str, default="sell_score",
                        choices=["sell_score", "low_risk", "composite"])
    parser.add_argument("--take-profit", type=float, default=TAKE_PROFIT,
                        help="止盈线 (仅 rule_exit)")
    args = parser.parse_args()
    main(args)
