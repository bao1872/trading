#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘净值曲线构建器 — 从模拟盘执行账本重放交易重建净值

Purpose:
    从 live 的 executions + holdings 账本 + price_pivot 重放逐日交易，
    重建账户净值曲线，输出格式与回测 equity_curve.csv 一致。

Inputs:
    - stop_experiment/output/live/executions/*.parquet (执行账本)
    - stop_experiment/output/holdings/*.parquet (持仓账本，用于交叉验证)
    - price_pivot: MultiIndex DataFrame (close)

Outputs:
    - stop_experiment/output/live/live_equity_curve.csv
      columns: date, cash, market_value, equity, nav, n_positions, daily_return, drawdown

How to Run:
    # 单独运行（需要先有 executions 和 holdings 数据）
    python stop_experiment/pipeline/build_live_equity_curve.py

    # 从其他脚本导入调用
    from stop_experiment.pipeline.build_live_equity_curve import build_live_equity_curve

Side Effects:
    - 只读 executions/holdings/price_pivot，不写数据库
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from glob import glob

from stop_experiment.pipeline.stop_config import EXECUTIONS_DIR, HOLDINGS_DIR, OUTPUT_DIR

LIVE_DIR = os.path.join(OUTPUT_DIR, "live")
BUY_COST = 0.0005
SELL_COST = 0.0010


def _extract_code(ts_code):
    """从 ts_code (如 002361.SZ) 提取纯代码 (002361)"""
    if isinstance(ts_code, str) and "." in ts_code:
        return ts_code.split(".")[0]
    return ts_code


def _load_all_executions():
    """加载所有执行账本文件，返回合并 DataFrame"""
    files = sorted(glob(os.path.join(EXECUTIONS_DIR, "*.parquet")))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df_all = pd.concat(dfs, ignore_index=True)
    df_all["execution_date"] = pd.to_datetime(df_all["execution_date"])
    df_all["code"] = df_all["ts_code"].apply(_extract_code)
    # 只保留 executed 的买卖
    df_all = df_all[df_all["status"] == "executed"]
    return df_all.sort_values("execution_date")


def _derive_latest_nav(eq_df):
    """从净值曲线 DataFrame 提取最新净值/日收益/回撤，未覆盖日期返回 (None, None, None)"""
    if eq_df.empty:
        return None, None, None
    latest = eq_df.iloc[-1]
    nav = latest.get("nav")
    daily_ret = latest.get("daily_return")
    dd = latest.get("drawdown")
    return nav, daily_ret, dd


def build_live_equity_curve(price_pivot, initial_cash=1.0,
                            buy_cost=BUY_COST, sell_cost=SELL_COST):
    """
    重放执行账本中的所有交易，重建账户净值曲线。

    Args:
        price_pivot: MultiIndex DataFrame (含 close)
        initial_cash: 初始现金 (默认 1.0)
        buy_cost: 买入成本率 (默认 0.0005)
        sell_cost: 卖出成本率 (默认 0.0010)

    Returns:
        DataFrame: date, cash, market_value, equity, nav, n_positions, daily_return, drawdown
    """
    exec_df = _load_all_executions()
    if exec_df.empty:
        print("  [净值曲线] 无执行记录，返回空 DataFrame")
        return pd.DataFrame()

    td_sorted = sorted(price_pivot.index)

    cash = initial_cash
    positions = {}  # code -> {"shares": float, "buy_price": float}
    prev_equity = initial_cash
    records = []

    for date in td_sorted:
        if date not in price_pivot.index:
            continue
        day_close = price_pivot.loc[date, "close"]

        # 当日执行（仅 executed）
        day_exec = exec_df[exec_df["execution_date"] == date]

        # Step A: 执行卖出（先卖释放现金）
        day_sells = day_exec[day_exec["action"] == "sell"]
        for _, row in day_sells.iterrows():
            code = row["code"]
            if code not in positions:
                continue
            sell_price = row["executed_price"]
            if pd.isna(sell_price) or sell_price <= 0:
                sell_price = day_close.get(code, np.nan)
            if pd.isna(sell_price) or sell_price <= 0:
                continue
            shares = positions[code]["shares"]
            cash += shares * sell_price * (1.0 - sell_cost)
            del positions[code]

        # Step B: 执行买入（使用当前权益分配份额）
        day_buys = day_exec[day_exec["action"] == "buy"]
        for _, row in day_buys.iterrows():
            code = row["code"]
            if code in positions:
                continue  # 已持有，跳过
            buy_price = row["executed_price"]
            if pd.isna(buy_price) or buy_price <= 0:
                buy_price = day_close.get(code, np.nan)
            if pd.isna(buy_price) or buy_price <= 0:
                continue
            # 计算买入前权益
            mv = 0.0
            for c, p in positions.items():
                cp_val = day_close.get(c, np.nan)
                if not pd.isna(cp_val) and cp_val > 0:
                    mv += p["shares"] * cp_val
            equity_pre_buy = cash + mv
            # 等权买入：position_value = equity_pre_buy / max(N, 1)
            n_total = max(len(positions) + 1, 1)
            weight = 1.0 / max(10, n_total)  # 最多10只，但初始可能不到10只
            # 实际用等权：目标每只占 1/max_stocks
            weight = 1.0 / 10.0
            notional = equity_pre_buy * weight
            shares = notional / buy_price
            actual_cost = shares * buy_price * (1.0 + buy_cost)
            if actual_cost > cash:
                # 现金不足，按可用现金调整份额
                shares = (cash / (1.0 + buy_cost)) / buy_price
                actual_cost = shares * buy_price * (1.0 + buy_cost)
            cash -= actual_cost
            positions[code] = {"shares": shares, "buy_price": buy_price}

        # Step C: 盯市计算市值
        market_value = 0.0
        for code, pos in positions.items():
            cp = day_close.get(code, np.nan)
            if not pd.isna(cp) and cp > 0:
                market_value += pos["shares"] * cp

        equity = cash + market_value
        n_positions = len(positions)

        # 计算日收益率（仅在 equity > 0 时）
        if prev_equity > 1e-8:
            daily_return = (equity - prev_equity) / prev_equity
        else:
            daily_return = 0.0

        records.append({
            "date": date,
            "cash": round(cash, 8),
            "market_value": round(market_value, 8),
            "equity": round(equity, 8),
            "nav": round(equity, 8),
            "n_positions": n_positions,
            "daily_return": round(daily_return, 8),
        })
        prev_equity = equity

    if not records:
        return pd.DataFrame()

    eq_df = pd.DataFrame(records)
    eq_df["date"] = pd.to_datetime(eq_df["date"])

    # 计算回撤
    eq_df["cummax_equity"] = eq_df["equity"].cummax()
    eq_df["drawdown"] = 1.0 - eq_df["equity"] / eq_df["cummax_equity"]
    eq_df["drawdown"] = eq_df["drawdown"].clip(0.0, None)

    eq_df = eq_df[["date", "cash", "market_value", "equity", "nav",
                    "n_positions", "daily_return", "drawdown"]]
    return eq_df


def save_live_equity_curve(eq_df, output_path=None):
    """保存净值曲线到 live/live_equity_curve.csv"""
    if output_path is None:
        os.makedirs(LIVE_DIR, exist_ok=True)
        output_path = os.path.join(LIVE_DIR, "live_equity_curve.csv")
    if eq_df.empty:
        print("  [净值曲线] 无数据，跳过保存")
        return eq_df
    eq_df.to_csv(output_path, index=False)
    final_nav = eq_df["nav"].iloc[-1]
    max_dd = eq_df["drawdown"].max()
    print(f"  [净值曲线] 已保存 {output_path} ({len(eq_df)} 日)")
    print(f"    final_nav={final_nav:.4f}, max_dd={max_dd:.4%}")
    return eq_df


def get_latest_nav(eq_df):
    """从净值曲线提取最新值，未覆盖返回 (None, None, None)"""
    if eq_df.empty:
        return None, None, None
    latest = eq_df.iloc[-1]
    return latest.get("nav"), latest.get("daily_return"), latest.get("drawdown")


# ==================== 自测入口 ====================
if __name__ == "__main__":
    from stop_experiment.backtest.dynamic_exit_backtest_v2 import _load_data
    from stop_experiment.pipeline.stop_config import BASELINE_E0_X1_V1_PARAMS

    p = BASELINE_E0_X1_V1_PARAMS
    print(f"冒烟测试: build_live_equity_curve")
    _, price, _, _, _ = _load_data(candidate_obs_days=p["candidate_obs_days"])
    eq_df = build_live_equity_curve(price)
    if eq_df.empty:
        print("  无执行记录，跳过")
    else:
        print(f"  行数: {len(eq_df)}")
        print(f"  nav[0]: {eq_df['nav'].iloc[0]:.4f}")
        print(f"  final_nav: {eq_df['nav'].iloc[-1]:.4f}")
        print(f"  max_dd: {eq_df['drawdown'].max():.4%}")
        save_live_equity_curve(eq_df)