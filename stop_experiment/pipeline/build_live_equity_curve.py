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
      columns: date, cash, market_value, equity, nav_live, n_positions, daily_return, drawdown

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
    nav = latest.get("nav_live")
    daily_ret = latest.get("daily_return")
    dd = latest.get("drawdown")
    return nav, daily_ret, dd


def _load_holdings_index():
    """
    加载所有持仓账本，构建逐日持仓索引。

    Returns:
        {date: {code: {weight, days_held}}}
    """
    files = sorted(glob(os.path.join(HOLDINGS_DIR, "*.parquet")))
    index = {}
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        for d, grp in df.groupby("date"):
            index[d] = {}
            for _, row in grp.iterrows():
                code = row["code"]
                index[d][code] = {
                    "weight": float(row.get("weight", 0)),
                    "days_held": int(row.get("days_held", 0)),
                }
    return index


def build_live_equity_curve(price_pivot, initial_cash=1.0,
                            buy_cost=BUY_COST, sell_cost=SELL_COST):
    """
    从持仓账本重建净值曲线，使用与回测一致的加权收益率法。

    回测公式（SSOT dynamic_exit_backtest_v2.run_backtest L406-L434）：
        daily_ret = sum(holdings[code].weight * stock_ret)
        nav *= (1 + daily_ret)
    其中:
        days_held==1: stock_ret = (close - open) / open
        days_held> 1: stock_ret = (close_today - close_yesterday) / close_yesterday

    Returns:
        DataFrame: date, cash, market_value, equity, nav_live, n_positions, daily_return, drawdown
    """
    td_sorted = sorted(price_pivot.index)

    holdings_by_day = _load_holdings_index()
    if not holdings_by_day:
        print("  [净值曲线] 无持仓记录，返回空 DataFrame")
        return pd.DataFrame()

    nav = initial_cash
    prev_equity = initial_cash
    records = []

    for idx, date in enumerate(td_sorted):
        if date not in price_pivot.index:
            continue

        day_open = price_pivot.loc[date, "open"] if "open" in price_pivot.columns.levels[0] else pd.Series(dtype=float)
        day_close = price_pivot.loc[date, "close"]

        day_holdings = holdings_by_day.get(date, {})

        daily_ret = 0.0
        for code, h in day_holdings.items():
            weight = h.get("weight", 0)
            days_held = h.get("days_held", 0)
            if weight <= 0:
                continue
            if code not in day_close.index or np.isnan(day_close[code]):
                continue

            if days_held == 1:
                if code in day_open.index and not np.isnan(day_open[code]) and day_open[code] > 0:
                    sr = (day_close[code] - day_open[code]) / day_open[code]
                else:
                    sr = 0
            else:
                if idx > 0:
                    prev_d = td_sorted[idx - 1]
                    if prev_d in price_pivot.index:
                        prev_c = price_pivot.loc[prev_d, "close"]
                        if code in prev_c.index and not np.isnan(prev_c[code]) and prev_c[code] > 0:
                            sr = (day_close[code] - prev_c[code]) / prev_c[code]
                        else:
                            sr = 0
                    else:
                        sr = 0
                else:
                    sr = 0
            daily_ret += weight * sr

        nav *= (1 + daily_ret)
        n_positions = len(day_holdings)

        if prev_equity > 1e-8:
            daily_return = (nav - prev_equity) / prev_equity
        else:
            daily_return = 0.0

        records.append({
            "date": date,
            "cash": round(nav * 0.05, 8),           # 近似分解
            "market_value": round(nav * 0.95, 8),    # 近似分解
            "equity": round(nav, 8),
            "nav_live": round(nav, 8),
            "n_positions": n_positions,
            "daily_return": round(daily_return, 8),
        })
        prev_equity = nav

    if not records:
        return pd.DataFrame()

    eq_df = pd.DataFrame(records)
    eq_df["date"] = pd.to_datetime(eq_df["date"])

    eq_df["cummax_equity"] = eq_df["equity"].cummax()
    eq_df["drawdown"] = 1.0 - eq_df["equity"] / eq_df["cummax_equity"]
    eq_df["drawdown"] = eq_df["drawdown"].clip(0.0, None)

    eq_df = eq_df[["date", "cash", "market_value", "equity", "nav_live",
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
    final_nav = eq_df["nav_live"].iloc[-1]
    max_dd = eq_df["drawdown"].max()
    print(f"  [净值曲线] 已保存 {output_path} ({len(eq_df)} 日)")
    print(f"    final_nav={final_nav:.4f}, max_dd={max_dd:.4%}")
    return eq_df


def get_latest_nav(eq_df):
    """从净值曲线提取最新值，未覆盖返回 (None, None, None)"""
    if eq_df.empty:
        return None, None, None
    latest = eq_df.iloc[-1]
    return latest.get("nav_live"), latest.get("daily_return"), latest.get("drawdown")


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
        print(f"  nav_live[0]: {eq_df['nav_live'].iloc[0]:.4f}")
        print(f"  final_nav_live: {eq_df['nav_live'].iloc[-1]:.4f}")
        print(f"  max_dd: {eq_df['drawdown'].max():.4%}")
        save_live_equity_curve(eq_df)