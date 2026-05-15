#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
账户净值曲线构建器（后处理模块）

Purpose:
    从引擎 nav_df 和 trades_df + price_pivot 构建账户净值曲线。
    equity/nav/daily_return 直接取自引擎 nav_df（精确），
    cash/market_value 从 trades 重建（标准化份额，近似分解），
    drawdown 从 equity 计算。

Pipeline Position:
    后处理模块，位于 run_backtest() 之后。
    上游：dynamic_exit_backtest_v2.run_backtest()
    下游：run_baseline.py, position_sizing_w1.py

Inputs:
    - result: run_backtest() 返回的 dict (含 nav_df, trades_df)
    - price_pivot: MultiIndex DataFrame, close 用于盯市
    - trading_days: 完整交易日列表

Outputs:
    - equity_curve.csv: date, cash, market_value, equity, nav,
                        n_positions, daily_return, drawdown

How to Run:
    python stop_experiment/backtest/build_equity_curve.py

Side Effects:
    - 写 CSV
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd


def build_equity_curve(result, price_pivot, trading_days,
                       initial_cash=1.0, buy_cost=0.0005, sell_cost=0.0010):
    """
    从回测产物构建账户净值曲线

    equity/nav/daily_return 直接取自引擎 nav_df（精确，与 summary 一致）。
    cash/market_value 从 trades_df + price_pivot 重建（标准化份额，近似分解）。

    Args:
        result: run_backtest() 返回的 dict (含 nav_df, trades_df)
        price_pivot: MultiIndex DataFrame
        trading_days: 完整交易日列表
        initial_cash: 初始现金 (默认 1.0)
        buy_cost/sell_cost: 交易成本率

    Returns:
        DataFrame: date, cash, market_value, equity, nav,
                  n_positions, daily_return, drawdown
    """
    nav_df = result["nav_df"].copy()
    trades_df = result.get("trades_df", pd.DataFrame())

    if nav_df.empty:
        return pd.DataFrame()

    nav_df["date"] = pd.to_datetime(nav_df["date"])

    eq_df = nav_df.copy()
    eq_df["equity"] = nav_df["nav"]
    eq_df["cummax_equity"] = nav_df["nav"].cummax()
    eq_df["drawdown"] = 1.0 - nav_df["nav"] / eq_df["cummax_equity"]

    if trades_df.empty:
        eq_df["cash"] = 1.0
        eq_df["market_value"] = 0.0
    else:
        shares_map = _derive_shares(
            trades_df, nav_df, price_pivot, trading_days,
            initial_cash, buy_cost, sell_cost,
        )
        eq_df = _fill_cash_market_value(eq_df, shares_map, price_pivot)

    eq_df = eq_df.rename(columns={"nav": "nav", "daily_ret": "daily_return"})
    eq_df = eq_df[["date", "cash", "market_value", "equity", "nav",
                    "n_positions", "daily_return", "drawdown"]]

    for col in ["cash", "market_value", "equity", "nav", "daily_return", "drawdown"]:
        eq_df[col] = eq_df[col].round(8)

    return eq_df


def _derive_shares(trades_df, nav_df, price_pivot, trading_days,
                   initial_cash, buy_cost, sell_cost):
    """
    从 trades 重建逐日持仓份额

    使用 notional 法: shares_i = (equity_before_buy × entry_weight_i) / buy_price_i
    其中 equity_before_buy 取上一日收盘 equity（引擎 nav_df.nav）。

    Returns:
        dict: date -> {code: shares}
    """
    if trades_df.empty:
        return {}

    trades = trades_df.copy()
    trades["buy_date"] = pd.to_datetime(trades["buy_date"])
    trades["sell_date"] = pd.to_datetime(trades["sell_date"])

    td_sorted = sorted(trading_days)

    trades_by_buy = {}
    trades_by_sell = {}
    for _, t in trades.iterrows():
        bd = t["buy_date"]
        sd = t["sell_date"]
        trades_by_buy.setdefault(bd, []).append(t)
        trades_by_sell.setdefault(sd, []).append(t)

    nav_map = {}
    nav_df_idx = nav_df.copy()
    nav_df_idx.index = pd.to_datetime(nav_df_idx["date"])
    for idx, row in nav_df_idx.iterrows():
        nav_map[idx] = row["nav"]

    positions = {}       # code -> shares
    shares_by_day = {}

    for i, current_date in enumerate(td_sorted):
        prev_date = td_sorted[i - 1] if i > 0 else None
        prev_nav = nav_map.get(prev_date, 1.0) if prev_date else 1.0

        # 卖出处理
        for t in trades_by_sell.get(current_date, []):
            code = _extract_code(t["ts_code"])
            if code in positions:
                del positions[code]

        # 买入处理
        for t in trades_by_buy.get(current_date, []):
            code = _extract_code(t["ts_code"])
            bp = t["buy_price"]
            if np.isnan(bp) or bp <= 0:
                continue
            weight = t.get("weight", 0.0)
            if weight <= 0:
                continue
            notional = prev_nav * weight
            shares = notional / bp
            positions[code] = shares

        shares_by_day[current_date] = dict(positions)

    return shares_by_day


def _fill_cash_market_value(eq_df, shares_map, price_pivot):
    """根据逐日份额和收盘价计算市值与现金"""
    cash_vals = []
    mv_vals = []

    for _, row in eq_df.iterrows():
        d = row["date"]
        if isinstance(d, str):
            d = pd.Timestamp(d)
        equity = row["equity"]

        if d not in shares_map or not shares_map[d]:
            cash_vals.append(equity)
            mv_vals.append(0.0)
            continue

        positions = shares_map[d]

        if d in price_pivot.index:
            day_close = price_pivot.loc[d, "close"]
        else:
            day_close = pd.Series(dtype=float)

        market_value = 0.0
        for code, shares in positions.items():
            if code in day_close.index:
                cp = day_close[code]
                if not np.isnan(cp) and cp > 0:
                    market_value += shares * cp

        if market_value > equity:
            market_value = equity
        cash = max(equity - market_value, 0.0)

        cash_vals.append(cash)
        mv_vals.append(market_value)

    eq_df["cash"] = cash_vals
    eq_df["market_value"] = mv_vals
    return eq_df


def save_equity_curve(result, price_pivot, trading_days, output_path, label="基线"):
    """
    从 run_backtest() 结果直接生成并保存净值曲线

    Args:
        result: run_backtest() 返回的 dict
        price_pivot: MultiIndex DataFrame
        trading_days: 交易日列表
        output_path: 输出 CSV 路径
        label: 标签（用于打印）
    """
    eq_df = build_equity_curve(result, price_pivot, trading_days)
    eq_df.to_csv(output_path, index=False)

    final_equity = eq_df["equity"].iloc[-1]
    max_dd = eq_df["drawdown"].max()
    print(f"  [{label}] 净值曲线: final_equity={final_equity:.4f}, max_dd={max_dd:.4f}")
    return eq_df


def _extract_code(ts_code):
    """从 ts_code (如 002361.SZ) 提取纯代码 (002361)"""
    if isinstance(ts_code, str) and "." in ts_code:
        return ts_code.split(".")[0]
    return ts_code


if __name__ == "__main__":
    from stop_experiment.pipeline.stop_config import BACKTEST_DIR, PRODUCTION_PARAMS
    from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data

    p = PRODUCTION_PARAMS
    print(f"冒烟测试: build_equity_curve ({p['profile']})")

    trades_path = os.path.join(BACKTEST_DIR, f"{p['profile']}_trades.csv")
    nav_path = os.path.join(BACKTEST_DIR, f"{p['profile']}_nav.csv")
    if not os.path.exists(trades_path) or not os.path.exists(nav_path):
        print(f"  跳过: 缺少 trades 或 nav 文件")
        sys.exit(0)

    trades_df = pd.read_csv(trades_path)
    nav_df = pd.read_csv(nav_path)
    _, price, _, _, _ = _load_data()
    td = sorted(price.index.unique())

    result = {"nav_df": nav_df, "trades_df": trades_df}
    eq = build_equity_curve(result, price, td)

    print(f"  行数: {len(eq)}")
    print(f"  nav[0]: {eq['nav'].iloc[0]:.4f}")
    print(f"  final_nav: {eq['nav'].iloc[-1]:.4f}")
    print(f"  max_dd: {eq['drawdown'].max():.4f}")

    out = os.path.join(BACKTEST_DIR, f"{p['profile']}_equity_curve.csv")
    eq.to_csv(out, index=False)
    print(f"  输出: {out}")