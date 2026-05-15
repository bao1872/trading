#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日状态推进器（SSOT）

Purpose:
    将"执行 pending buys/sells → 推进持仓 → 调用 decide_eod"抽象为公共函数，
    供回测 / replay / 模拟盘 / 盘后报告统一调用，消除三套日状态推进实现。

    Usage:
        python stop_experiment/backtest/daily_state_machine.py
        运行自测（使用回测引擎内置样例）。

    How to Run:
        # 模块自测（依赖回测引擎内置样例）
        python stop_experiment/backtest/daily_state_machine.py

    Examples:
        # 1. 在回测循环中使用
        from stop_experiment.backtest.daily_state_machine import execute_pending_buys, execute_pending_sells
        holdings, executed, skipped = execute_pending_buys(holdings, pending_orders, date, price_pivot, prev_close_map)

        # 2. 在单日模拟盘中使用 step_day
        from stop_experiment.backtest.daily_state_machine import step_day
        result = step_day(date, holdings, pending_buys, pending_sells,
                          price_pivot, candidates, pred_lookup, prev_date, params)

    Side Effects:
        无（纯函数，不读写文件/数据库）

Outputs:
        无（纯计算模块）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.backtest.simple_backtest import is_suspended, is_limit_up, is_limit_down
from stop_experiment.backtest.decision_core import decide_eod


def execute_pending_buys(holdings, pending_buys, date, price_pivot, prev_close_map,
                         default_weight=0.1, strict=True, max_slots=None):
    """
    执行 T-1 日 pending buys：T 日开盘买入，含停牌/涨停检查。

    Args:
        holdings: 当日开盘前持仓 {code: {buy_date, buy_price, weight, days_held, ts_code, score, signal_id, ...}}
        pending_buys: [(code, planned_price, ts_code, score, signal_id), ...]
        date: 执行日（交易日）
        price_pivot: MultiIndex DataFrame，至少含 open/volume/(high for strict) 子面板
        prev_close_map: {code: Series indexed by date}，前收盘价
        default_weight: 新买入默认权重（等权 = 1/max_stocks）
        strict: 是否严格检查停牌/涨停
        max_slots: 最大成功买入数（None=不限制），用于替补机制：pending_buys 含 buffer 候选，
                   成功买入数达到 max_slots 后停止，跳过剩余候选

    Returns:
        (holdings, executed_buys, skipped_buys)
        - holdings: 更新后的持仓（新买入已加入）
        - executed_buys: [(code, buy_price, ts_code, score, signal_id), ...]
        - skipped_buys: [(code, buy_price, ts_code, score, signal_id, reason), ...]
    """
    executed = []
    skipped = []

    day_open = price_pivot.loc[date, "open"] if date in price_pivot.index else pd.Series(dtype=float)

    for code, bp, ts_code, sc, sid in pending_buys:
        if max_slots is not None and len(executed) >= max_slots:
            break
        if code in holdings:
            skipped.append((code, bp, ts_code, sc, sid, "already_held"))
            continue
        if np.isnan(bp) or bp <= 0:
            skipped.append((code, bp, ts_code, sc, sid, "missing_price"))
            continue
        if strict:
            if "volume" in price_pivot.columns.levels[0] and code in price_pivot["volume"].columns:
                vol_val = price_pivot["volume"][code].get(date, np.nan)
                if is_suspended(vol_val):
                    skipped.append((code, bp, ts_code, sc, sid, "suspended"))
                    continue
            if code in prev_close_map and date in prev_close_map[code].index:
                prev_c = prev_close_map[code].get(date, np.nan)
                if not np.isnan(prev_c) and prev_c > 0:
                    if "high" in price_pivot.columns.levels[0] and code in price_pivot["high"].columns:
                        high_val = price_pivot["high"][code].get(date, np.nan)
                        if not np.isnan(high_val) and is_limit_up(bp, high_val, prev_c):
                            skipped.append((code, bp, ts_code, sc, sid, "limit_up"))
                            continue
        holdings[code] = {
            "buy_date": date, "buy_price": bp,
            "weight": default_weight, "days_held": 0,
            "ts_code": ts_code, "score": sc, "signal_id": sid,
        }
        executed.append((code, bp, ts_code, sc, sid))

    return holdings, executed, skipped


def execute_pending_sells(holdings, pending_sells, date, price_pivot, prev_close_map,
                          strict=True):
    """
    执行 T-1 日 pending sells：T 日开盘卖出，含停牌/跌停检查。

    Args:
        holdings: 当日开盘前持仓（含已执行的买入）
        pending_sells: [{"code": ..., "holding": {...}, "reason": ..., ...}, ...]
        date: 执行日（交易日）
        price_pivot: MultiIndex DataFrame
        prev_close_map: {code: Series indexed by date}
        strict: 是否严格检查停牌/跌停

    Returns:
        (holdings, executed_sells, skipped_sells)
        - holdings: 更新后的持仓（已卖出已删除）
        - executed_sells: [dict(sell_item with "executed_price"), ...]
        - skipped_sells: [dict(sell_item with "reason" updated), ...]
    """
    executed = []
    skipped = []

    day_open = price_pivot.loc[date, "open"] if date in price_pivot.index else pd.Series(dtype=float)

    for sell_item in pending_sells:
        code = sell_item["code"]
        if code not in holdings:
            skipped.append(dict(sell_item, reason="not_held"))
            continue
        sell_price = np.nan
        if code in day_open.index and not np.isnan(day_open[code]):
            sell_price = day_open[code]
        if np.isnan(sell_price) or sell_price <= 0:
            skipped.append(dict(sell_item, reason="missing_price"))
            continue
        if strict:
            if "volume" in price_pivot.columns.levels[0] and code in price_pivot["volume"].columns:
                vol_val = price_pivot["volume"][code].get(date, np.nan)
                if is_suspended(vol_val):
                    skipped.append(dict(sell_item, reason="suspended"))
                    continue
            if code in prev_close_map and date in prev_close_map[code].index:
                prev_c = prev_close_map[code].get(date, np.nan)
                if not np.isnan(prev_c) and prev_c > 0:
                    if "low" in price_pivot.columns.levels[0] and code in price_pivot["low"].columns:
                        low_val = price_pivot["low"][code].get(date, np.nan)
                        if not np.isnan(low_val) and is_limit_down(sell_price, low_val, prev_c):
                            skipped.append(dict(sell_item, reason="limit_down"))
                            continue
        sell_item = dict(sell_item)
        sell_item["executed_price"] = sell_price
        executed.append(sell_item)
        del holdings[code]

    return holdings, executed, skipped


def step_day(date, holdings, pending_buys, pending_sells,
             price_pivot, candidates, pred_lookup, prev_date, params,
             prev_close_map=None, strict=True,
             day_open_next=None, exit_sub_mode=None,
             buy_reg_exit_threshold=None, prev_decision_date=None,
             weight_mode=None, weight_params=None):
    """
    单日状态推进（SSOT）：执行 pending orders → 调用 decide_eod → 返回新状态。

    回测 / replay / 模拟盘 / 盘后报告统一调用，确保"日状态推进"只此一处实现。

    Args:
        date: 交易日（执行日）
        holdings: 当日开盘前持仓 {code: {...}}
        pending_buys: [(code, planned_price, ts_code, score, signal_id), ...]
        pending_sells: [{"code": ..., "holding": {...}, "reason": ..., ...}, ...]
        price_pivot: MultiIndex DataFrame
        candidates: DataFrame，当日候选（含 score, ts_code, signal_id 列）
        pred_lookup: {(signal_id, obs_date): pred_dict}
        prev_date: 上一交易日（用于模型退出参考）
        params: 参数 dict，含:
            - max_stocks: int
            - max_hold_days: int
            - stop_loss: float
            - exit_threshold: float (buy_cls_exit_threshold)
        prev_close_map: {code: Series}，前收盘价（strict 检查用）
        strict: 是否严格检查停牌/涨跌停
        day_open_next: Series，下一交易日开盘价（decide_eod 用）
        exit_sub_mode: 退出子模式（decide_eod 透传）
        buy_reg_exit_threshold: buy_reg 退出阈值（decide_eod 透传）
        prev_decision_date: 前日决策日（decide_eod 透传）

    Returns:
        {
            "holdings": {},            # 收盘后持仓（已含决策买入/卖出）
            "pending_buys": [],        # 新决策买入（T+1 执行）
            "pending_sells": [],       # 新决策卖出（T+1 执行）
            "sell_reasons": {},        # {code: reason}
            "extra": {},               # decide_eod extra diagnostics
            "executed_buys": [],       # 今日实际执行的买入
            "executed_sells": [],      # 今日实际执行的卖出
            "skipped_buys": [],        # 今日跳过的买入
            "skipped_sells": [],       # 今日跳过的卖出
        }
    """
    max_stocks = params.get("max_stocks", 10)
    max_hold_days = params.get("max_hold_days", 20)
    stop_loss = params.get("stop_loss", -0.07)
    exit_threshold = params.get("exit_threshold", params.get("buy_cls_exit_threshold", 0.70))
    default_weight = 1.0 / max_stocks

    # Step 1: 先执行 T-1 pending sells（卖出先于买入，确保仓位释放后再买入）
    holdings, executed_sells, skipped_sells = execute_pending_sells(
        holdings, pending_sells, date, price_pivot, prev_close_map or {},
        strict=strict,
    )

    # Step 1+: 卖出后重新平衡权重
    if executed_sells:
        from stop_experiment.backtest.dynamic_exit_backtest_v2 import _calc_weights
        all_codes = list(holdings.keys())
        if all_codes:
            all_scores = [holdings[c].get("weight_score", holdings[c].get("score", 0)) for c in all_codes]
            weights = _calc_weights(all_codes, all_scores, weight_mode, weight_params)
            for code in holdings:
                holdings[code]["weight"] = weights.get(code, 0)

    # max_slots: 卖出已执行，基于实际持仓计算可用买入数
    max_slots = max_stocks - len(holdings)
    if max_slots < 0:
        max_slots = 0

    # Step 1.5: 执行 T-1 pending buys（含 buffer 候选替补机制）
    holdings, executed_buys, skipped_buys = execute_pending_buys(
        holdings, pending_buys, date, price_pivot, prev_close_map or {},
        default_weight=default_weight, strict=strict,
        max_slots=max_slots,
    )

    # Step 1.5+: 买入后重新平衡所有持仓权重
    if executed_buys:
        from stop_experiment.backtest.dynamic_exit_backtest_v2 import _calc_weights
        all_codes = list(holdings.keys())
        all_scores = [holdings[c].get("weight_score", holdings[c].get("score", 0)) for c in all_codes]
        weights = _calc_weights(all_codes, all_scores, weight_mode, weight_params)
        for code in holdings:
            holdings[code]["weight"] = weights.get(code, 0)

    # Step 2~4: 统一日终决策（SSOT decide_eod）
    day_close = price_pivot.loc[date, "close"] if date in price_pivot.index else pd.Series(dtype=float)

    holdings, pending_buys_new, pending_sells_new, sell_reasons, extra = decide_eod(
        decision_date=date,
        holdings=holdings,
        candidates=candidates,
        pred_lookup=pred_lookup,
        prev_date=prev_date,
        day_close=day_close,
        day_open_next=day_open_next,
        max_stocks=max_stocks,
        max_hold_days=max_hold_days,
        stop_loss=stop_loss,
        exit_threshold=exit_threshold,
        exit_sub_mode=exit_sub_mode,
        buy_reg_exit_threshold=buy_reg_exit_threshold,
        prev_decision_date=prev_decision_date,
    )

    return {
        "holdings": holdings,
        "pending_buys": pending_buys_new,
        "pending_sells": pending_sells_new,
        "sell_reasons": sell_reasons,
        "extra": extra,
        "executed_buys": executed_buys,
        "executed_sells": executed_sells,
        "skipped_buys": skipped_buys,
        "skipped_sells": skipped_sells,
    }


if __name__ == "__main__":
    """
    自测：使用回测引擎内置样例验证 step_day 核心流程可运行。
    不写库表，无副作用。
    """
    from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data
    from stop_experiment.pipeline.stop_config import PRODUCTION_PARAMS
    from stop_experiment.backtest.simple_backtest import score_stocks

    print("=" * 60)
    print("  daily_state_machine 自测")
    print("=" * 60)

    # 加载回测数据
    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data()
    print(f"  交易日数: {len(trading_days)}")
    print(f"  信号数: {len(test_df)}")

    # 取前两个交易日测试 step_day
    test_date = sorted(trading_days)[0]
    next_date = sorted(trading_days)[1]

    candidates_all = score_stocks(test_df, "sell_score")
    candidates = candidates_all[candidates_all["obs_date"] == test_date].copy()
    if candidates.empty:
        print("  ⚠ 无候选数据，跳过测试")
        sys.exit(0)

    params = {
        "max_stocks": PRODUCTION_PARAMS.get("max_stocks", 10),
        "max_hold_days": PRODUCTION_PARAMS.get("max_hold_days", 20),
        "stop_loss": PRODUCTION_PARAMS.get("stop_loss", -0.07),
        "exit_threshold": PRODUCTION_PARAMS.get("buy_cls_exit_threshold", 0.70),
    }

    # Day 1: 空仓起步
    print(f"\n  [Day 1] {test_date.strftime('%Y-%m-%d')}: 空仓起步")
    result = step_day(
        test_date, {}, [], [],
        price_pivot, candidates, pred_lookup, None, params,
        prev_close_map=prev_close_map,
    )
    print(f"    持仓: {len(result['holdings'])} 只")
    print(f"    买入: {len(result['executed_buys'])}, 卖出: {len(result['executed_sells'])}")
    print(f"    新 pending buys: {len(result['pending_buys'])}, sells: {len(result['pending_sells'])}")

    # Day 2: 带 pending buys 推进
    if not result["pending_buys"]:
        print("  ⚠ Day 1 无 pending buys，跳过 Day 2 测试")
        sys.exit(0)

    print(f"\n  [Day 2] {next_date.strftime('%Y-%m-%d')}: 带 pending buys 推进")
    candidates2 = candidates_all[candidates_all["obs_date"] == next_date].copy()
    result2 = step_day(
        next_date, dict(result["holdings"]),
        list(result["pending_buys"]), list(result["pending_sells"]),
        price_pivot, candidates2, pred_lookup, test_date, params,
        prev_close_map=prev_close_map,
    )
    print(f"    持仓: {len(result2['holdings'])} 只")
    print(f"    买入: {len(result2['executed_buys'])}, 卖出: {len(result2['executed_sells'])}")
    print(f"    skipped buys: {len(result2['skipped_buys'])}, sells: {len(result2['skipped_sells'])}")

    print("\n  ✅ step_day 自测通过")