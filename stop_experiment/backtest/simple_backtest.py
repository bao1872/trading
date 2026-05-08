#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[DEPRECATED] 真实持仓回测引擎：基于模型评分的选股回测

⚠️ 已废弃：早期 baseline，固定持有期回测，已被 dynamic_exit_backtest_v2.py 覆盖。
   替代: dynamic_exit_backtest_v2.py

Purpose:
    (历史保留) 在test集上做真实交易口径回测：
    T日obs_day=1信号 → T+1 open买入 → T+N open卖出。
    成本落地（买0.05%+卖0.10%）、涨跌停/停牌约束、持仓去重。

Pipeline Position:
    实验逻辑（已废弃）。
    替代: dynamic_exit_backtest_v2.py

Inputs:
    - stop_experiment/output/test_predictions.parquet
    - DB: stock_k_data

Outputs:
    - stop_experiment/output/backtest/ 目录

How to Run:
    # ⚠️ 请勿运行，已废弃。请使用:
    python stop_experiment/backtest/dynamic_exit_backtest_v2.py --mode single --top-k 10

Side Effects:
    - (历史保留，不建议运行)
"""

from __future__ import annotations

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, BACKTEST_DIR, BUY_COST, SELL_COST,
)

# ==================== 涨跌停/停牌判断 ====================
LIMIT_UP_THRESHOLD = 0.095    # 涨停阈值（9.5%，考虑ST股）
LIMIT_DOWN_THRESHOLD = -0.095  # 跌停阈值


def is_limit_up(buy_open: float, day_high: float, prev_close: float) -> bool:
    """涨停判断：开盘=最高 且 开盘涨幅>=9.5%"""
    if pd.isna(buy_open) or pd.isna(day_high) or pd.isna(prev_close) or prev_close <= 0:
        return False
    return abs(buy_open - day_high) < 1e-6 and (buy_open - prev_close) / prev_close >= LIMIT_UP_THRESHOLD


def is_limit_down(sell_open: float, day_low: float, prev_close: float) -> bool:
    """跌停判断：开盘=最低 且 开盘跌幅<=-9.5%"""
    if pd.isna(sell_open) or pd.isna(day_low) or pd.isna(prev_close) or prev_close <= 0:
        return False
    return abs(sell_open - day_low) < 1e-6 and (sell_open - prev_close) / prev_close <= LIMIT_DOWN_THRESHOLD


def is_suspended(volume: float) -> bool:
    """停牌判断：成交量为0"""
    return pd.isna(volume) or volume <= 0


# ==================== K线数据加载 ====================
def load_daily_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """从DB加载日线K线数据，pivot为宽表"""
    from sqlalchemy import create_engine, text
    from config import DATABASE_URL
    engine = create_engine(DATABASE_URL)
    sql = text("""
        SELECT ts_code, bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE freq = 'd' AND bar_time >= :start AND bar_time <= :end
        ORDER BY ts_code, bar_time
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"start": start_date, "end": end_date})
    df["raw_code"] = df["ts_code"].str[:6]
    df["bar_time"] = pd.to_datetime(df["bar_time"])
    return df


def build_price_pivot(daily_prices: pd.DataFrame) -> tuple:
    """
    构建价格宽表 + 交易日列表 + prev_close映射

    Returns
    -------
    price_pivot : pd.DataFrame (multi-index columns: open/high/low/close/volume, index=bar_time)
    trading_days : list of dates
    prev_close_map : dict {raw_code: Series(prev_close, index=bar_time)}
    """
    price_pivot = daily_prices.pivot_table(
        index="bar_time", columns="raw_code",
        values=["open", "high", "low", "close", "volume"],
        aggfunc="first",
    )
    trading_days = sorted(daily_prices["bar_time"].unique())
    all_codes = daily_prices["raw_code"].unique()

    prev_close_map = {}
    if "close" in price_pivot:
        for code in all_codes:
            if code in price_pivot["close"].columns:
                prev_close_map[code] = price_pivot["close"][code].shift(1)

    return price_pivot, trading_days, prev_close_map


# ==================== 选股策略 ====================
def score_stocks(test_df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    为每只股票打分，添加 score 列

    Parameters
    ----------
    test_df : 仅obs_day=1的数据
    strategy : sell_score / low_risk / composite
    """
    df = test_df.copy()

    if "sell_score" not in df.columns:
        df["sell_score"] = df.get("pred_sell_reg", 0)
    if "downside_risk" not in df.columns:
        df["downside_risk"] = df.get("pred_buy_cls", 0)
    if "composite_score" not in df.columns:
        sell_score = df.get("pred_sell_reg", pd.Series(0, index=df.index))
        buy_score = -df.get("pred_buy_reg", pd.Series(0, index=df.index))
        df["composite_score"] = sell_score * 0.5 + buy_score * 0.5

    if strategy == "sell_score":
        df["score"] = df["sell_score"]
    elif strategy == "low_risk":
        # 低风险策略：downside_risk < 0.4 的候选中按sell_score排序
        # 评分 = sell_score - downside_risk * 2（惩罚下行风险）
        df["score"] = df["sell_score"] - df["downside_risk"] * 2
    elif strategy == "composite":
        df["score"] = df["composite_score"]
    else:
        df["score"] = df["sell_score"]

    return df


# ==================== 核心回测引擎 ====================
def run_backtest(
    signals_df: pd.DataFrame,
    price_pivot: pd.DataFrame,
    trading_days: list,
    prev_close_map: dict,
    max_stocks: int = 20,
    hold_days: int = 5,
    strategy: str = "sell_score",
    strict: bool = True,
) -> dict:
    """
    真实持仓回测引擎

    时序：
    - T日 obs_day=1 出现信号 → T+1日 open 买入
    - 持仓 hold_days 天后 → T+N+1日 open 卖出
    - 同一股票持仓未结束前不重复开仓（续持不加仓）

    Parameters
    ----------
    signals_df : 仅obs_day=1的test集数据，含pred_*列
    price_pivot : 宽表 (index=bar_time, columns=raw_code)
    trading_days : 排序后的交易日列表
    prev_close_map : {raw_code: Series(prev_close)}
    max_stocks : 每日最大持仓数（即top_k）
    hold_days : 持仓天数
    strategy : 选股策略名
    strict : 是否启用涨跌停/停牌约束

    Returns
    -------
    dict with keys: nav_df, trades_df, skipped_stats, params
    """
    # 打分
    signals_df = score_stocks(signals_df, strategy)
    signals_sorted = signals_df.sort_values(["obs_date", "score"], ascending=[True, False])

    # 信号按日期索引，方便查找
    signal_dates = sorted(signals_df["obs_date"].unique())
    signal_by_date = {}
    for date in signal_dates:
        day_sigs = signals_sorted[signals_sorted["obs_date"] == date]
        # 同一股票只取评分最高的信号
        signal_by_date[date] = day_sigs.drop_duplicates(subset=["ts_code"], keep="first")

    # 持仓状态: {raw_code: {buy_date, buy_price, weight, days_held, ...}}
    holdings = {}
    # 待执行买单: [(code, buy_price, ts_code, score), ...]
    # 信号日产生 → 下一个交易日开盘执行
    pending_orders = []
    nav = 1.0
    trade_details = []
    nav_records = []
    skipped_limit_up = 0
    skipped_limit_down = 0
    skipped_suspended = 0
    skipped_already_held = 0

    for t_idx, current_date in enumerate(trading_days):
        if current_date not in price_pivot.index:
            continue

        # ---- 当日价格数据 ----
        day_open = price_pivot.loc[current_date, "open"] if "open" in price_pivot else pd.Series(dtype=float)
        day_high = price_pivot.loc[current_date, "high"] if "high" in price_pivot else pd.Series(dtype=float)
        day_low = price_pivot.loc[current_date, "low"] if "low" in price_pivot else pd.Series(dtype=float)
        day_close = price_pivot.loc[current_date, "close"] if "close" in price_pivot else pd.Series(dtype=float)

        # ---- Step 1: 执行pending买单（T+1 open买入） ----
        if pending_orders:
            executed = []
            for order in pending_orders:
                code, buy_price, ts_code, score = order

                # 持仓去重：同股持仓未结束前不重复开仓
                if code in holdings:
                    skipped_already_held += 1
                    continue

                # 涨停检查（在买入日执行）
                if strict:
                    if "volume" in price_pivot and code in price_pivot["volume"].columns:
                        vol_series = price_pivot["volume"][code]
                        if current_date in vol_series.index:
                            if is_suspended(vol_series.get(current_date, np.nan)):
                                skipped_suspended += 1
                                continue
                    if code in prev_close_map and current_date in prev_close_map[code].index:
                        prev_c = prev_close_map[code].get(current_date, np.nan)
                        if not np.isnan(prev_c) and prev_c > 0:
                            day_high_series = price_pivot["high"][code]
                            if current_date in day_high_series.index:
                                if is_limit_up(buy_price, day_high_series.get(current_date, np.nan), prev_c):
                                    skipped_limit_up += 1
                                    continue

                executed.append((code, buy_price, ts_code, score))

            # 等权分配（新买入+已有持仓）
            if executed:
                n_total = len(holdings) + len(executed)
                w = 1.0 / n_total
                # 重新分配已有持仓权重
                for code in holdings:
                    holdings[code]["weight"] = w
                for code, bp, ts_code, score in executed:
                    holdings[code] = {
                        "buy_date": current_date,
                        "buy_price": bp,
                        "weight": w,
                        "days_held": 0,
                        "ts_code": ts_code,
                        "score": score,
                    }

            pending_orders = []

        # ---- Step 2: 递增持仓天数 + 卖出逻辑 ----
        to_sell = []
        for code, h in list(holdings.items()):
            h["days_held"] += 1

            should_sell = h["days_held"] > hold_days
            if not should_sell:
                continue

            # 卖出：T+N+1 open卖出（当前日即卖出日）
            sell_price = np.nan
            if code in day_open.index and not np.isnan(day_open[code]):
                sell_price = day_open[code]

            if np.isnan(sell_price) or sell_price <= 0:
                # 无法卖出（停牌等），继续持有
                continue

            # 跌停检查
            if strict:
                if "volume" in price_pivot and code in price_pivot["volume"].columns:
                    vol_series = price_pivot["volume"][code]
                    if current_date in vol_series.index:
                        if is_suspended(vol_series.get(current_date, np.nan)):
                            skipped_suspended += 1
                            continue
                if code in prev_close_map and current_date in prev_close_map[code].index:
                    prev_c = prev_close_map[code].get(current_date, np.nan)
                    if not np.isnan(prev_c) and prev_c > 0:
                        if code in day_low.index and not np.isnan(day_low[code]):
                            if is_limit_down(sell_price, day_low[code], prev_c):
                                skipped_limit_down += 1
                                continue

            # 计算收益
            gross_ret = (sell_price - h["buy_price"]) / h["buy_price"]
            net_ret = gross_ret - BUY_COST - SELL_COST

            # 计算持仓期MAE/MFE
            mae_val = np.nan
            mfe_val = np.nan
            for d_back in range(1, h["days_held"] + 1):
                check_idx = t_idx - d_back
                if check_idx < 0 or check_idx >= len(trading_days):
                    continue
                check_date = trading_days[check_idx]
                if check_date not in price_pivot.index:
                    continue
                check_low = price_pivot.loc[check_date, "low"]
                check_high = price_pivot.loc[check_date, "high"]
                if code in check_low.index and not np.isnan(check_low[code]):
                    c_mae = (check_low[code] - h["buy_price"]) / h["buy_price"]
                    if np.isnan(mae_val) or c_mae < mae_val:
                        mae_val = c_mae
                if code in check_high.index and not np.isnan(check_high[code]):
                    c_mfe = (check_high[code] - h["buy_price"]) / h["buy_price"]
                    if np.isnan(mfe_val) or c_mfe > mfe_val:
                        mfe_val = c_mfe

            trade_details.append({
                "raw_code": code,
                "ts_code": h["ts_code"],
                "buy_date": h["buy_date"],
                "sell_date": current_date,
                "buy_price": h["buy_price"],
                "sell_price": sell_price,
                "hold_days": h["days_held"],
                "gross_ret": gross_ret,
                "net_ret": net_ret,
                "mae": mae_val,
                "mfe": mfe_val,
                "sell_reason": "fixed",
                "score": h.get("score", 0),
                "strategy": strategy,
            })
            to_sell.append(code)

        for code in to_sell:
            del holdings[code]

        # ---- Step 3: 计算当日NAV（仅已有持仓参与，不含pending） ----
        daily_ret = 0.0
        for code, h in holdings.items():
            if code in day_close.index and not np.isnan(day_close[code]):
                if h["days_held"] == 1:
                    # 买入当天（days_held在Step2已+1）：从开盘到收盘
                    if code in day_open.index and not np.isnan(day_open[code]):
                        stock_ret = (day_close[code] - day_open[code]) / day_open[code]
                    else:
                        stock_ret = 0
                else:
                    # 非买入当天：前日收盘→今日收盘
                    if t_idx > 0:
                        prev_date = trading_days[t_idx - 1]
                        if prev_date in price_pivot.index:
                            prev_day_close = price_pivot.loc[prev_date, "close"]
                            if code in prev_day_close.index and not np.isnan(prev_day_close[code]):
                                stock_ret = (day_close[code] - prev_day_close[code]) / prev_day_close[code]
                            else:
                                stock_ret = 0
                        else:
                            stock_ret = 0
                    else:
                        stock_ret = 0
                daily_ret += h["weight"] * stock_ret

        nav *= (1 + daily_ret)
        cash_ratio = max(0, 1.0 - sum(h["weight"] for h in holdings.values()))

        nav_records.append({
            "date": current_date,
            "nav": nav,
            "daily_ret": daily_ret,
            "n_positions": len(holdings),
            "cash_ratio": cash_ratio,
        })

        # ---- Step 4: 收集当日信号 → 加入pending_orders（T+1执行） ----
        if current_date in signal_by_date:
            day_signals = signal_by_date[current_date]
        else:
            day_signals = pd.DataFrame()

        n_available = max_stocks - len(holdings) - len(pending_orders)

        if n_available > 0 and len(day_signals) > 0:
            # 买入在下一个交易日执行
            next_idx = t_idx + 1
            if next_idx < len(trading_days):
                buy_date = trading_days[next_idx]

                if buy_date in price_pivot.index:
                    buy_day_open = price_pivot.loc[buy_date, "open"]

                    candidates = day_signals.head(n_available * 3)
                    for _, sig in candidates.iterrows():
                        if len(holdings) + len(pending_orders) >= max_stocks:
                            break

                        code = sig["ts_code"][:6] if "." in sig["ts_code"] else sig["ts_code"]

                        # 去重：已在持仓或已在pending中的不重复
                        if code in holdings or any(o[0] == code for o in pending_orders):
                            skipped_already_held += 1
                            continue

                        if code not in buy_day_open.index or np.isnan(buy_day_open[code]):
                            continue

                        buy_price = buy_day_open[code]
                        if buy_price <= 0:
                            continue

                        pending_orders.append((code, buy_price, sig["ts_code"], sig.get("score", 0)))

    nav_df = pd.DataFrame(nav_records)
    trades_df = pd.DataFrame(trade_details)
    skipped_stats = {
        "limit_up": skipped_limit_up,
        "limit_down": skipped_limit_down,
        "suspended": skipped_suspended,
        "already_held": skipped_already_held,
    }
    return {
        "nav_df": nav_df,
        "trades_df": trades_df,
        "skipped_stats": skipped_stats,
        "params": {
            "max_stocks": max_stocks,
            "hold_days": hold_days,
            "strategy": strategy,
            "strict": strict,
        },
    }


# ==================== 汇总指标 ====================
def compute_summary(result: dict) -> dict:
    """计算回测核心指标"""
    nav_df = result["nav_df"]
    trades_df = result["trades_df"]
    params = result["params"]

    if nav_df.empty:
        return {**params, "n_trades": 0}

    nav_df = nav_df.copy()
    nav_df["cummax"] = nav_df["nav"].cummax()
    nav_df["drawdown"] = (nav_df["nav"] - nav_df["cummax"]) / nav_df["cummax"]

    total_days = len(nav_df)
    total_years = total_days / 252
    final_nav = nav_df["nav"].iloc[-1]
    annual_ret = (final_nav ** (1 / total_years) - 1) if total_years > 0 and final_nav > 0 else 0
    max_dd = nav_df["drawdown"].min()
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 1e-6 else 0
    daily_rets = nav_df["daily_ret"]
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 1e-6 else 0

    n_trades = len(trades_df)
    win_rate = (trades_df["net_ret"] > 0).mean() if n_trades > 0 else 0
    avg_gross_ret = trades_df["gross_ret"].mean() if n_trades > 0 else 0
    avg_net_ret = trades_df["net_ret"].mean() if n_trades > 0 else 0
    avg_mae = trades_df["mae"].mean() if n_trades > 0 else 0
    avg_mfe = trades_df["mfe"].mean() if n_trades > 0 else 0
    avg_hold = trades_df["hold_days"].mean() if n_trades > 0 else 0

    return {
        **params,
        "n_trades": n_trades,
        "final_nav": final_nav,
        "annual_ret": annual_ret,
        "max_dd": max_dd,
        "calmar": calmar,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "avg_gross_ret": avg_gross_ret,
        "avg_net_ret": avg_net_ret,
        "avg_mae": avg_mae,
        "avg_mfe": avg_mfe,
        "avg_hold_days": avg_hold,
        "skipped": result["skipped_stats"],
    }


# ==================== 可视化 ====================
def plot_backtest(result: dict, output_dir: str):
    """绘制回测图表"""
    nav_df = result["nav_df"]
    trades_df = result["trades_df"]
    params = result["params"]

    if nav_df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    tag = f"{params['strategy']}_k{params['max_stocks']}_h{params['hold_days']}"

    # 计算回撤列
    nav_plot = nav_df.copy()
    nav_plot["cummax"] = nav_plot["nav"].cummax()
    nav_plot["drawdown"] = (nav_plot["nav"] - nav_plot["cummax"]) / nav_plot["cummax"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Backtest: {tag}", fontsize=14, fontweight="bold")

    # 1. 净值曲线
    ax = axes[0, 0]
    ax.plot(nav_plot["date"], nav_plot["nav"], label="NAV", color="steelblue", linewidth=1.5)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3)
    ax.set_title("NAV Curve")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    # 2. 回撤
    ax = axes[0, 1]
    ax.fill_between(nav_plot["date"], nav_plot["drawdown"], 0, color="red", alpha=0.3)
    ax.set_title("Drawdown")
    ax.tick_params(axis="x", rotation=45)

    # 3. 每日收益分布
    ax = axes[1, 0]
    ax.bar(nav_plot["date"], nav_plot["daily_ret"], color=np.where(nav_plot["daily_ret"] >= 0, "green", "red"), alpha=0.6, width=1)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_title("Daily Returns")
    ax.tick_params(axis="x", rotation=45)

    # 4. 逐笔收益分布
    ax = axes[1, 1]
    if not trades_df.empty:
        ax.hist(trades_df["net_ret"], bins=50, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.axvline(x=trades_df["net_ret"].mean(), color="green", linestyle="--", alpha=0.8, label=f"mean={trades_df['net_ret'].mean():.2%}")
        ax.legend()
    ax.set_title("Trade Returns (net)")

    plt.tight_layout()
    path = os.path.join(output_dir, f"backtest_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


# ==================== 稳定性矩阵 ====================
def run_stability_matrix(
    test_df: pd.DataFrame,
    price_pivot: pd.DataFrame,
    trading_days: list,
    prev_close_map: dict,
) -> pd.DataFrame:
    """
    稳定性矩阵：遍历 top_k × hold_days × strategy，输出成本前后对比

    Returns
    -------
    pd.DataFrame with columns: strategy, top_k, hold_days, n_trades,
        final_nav, annual_ret, max_dd, calmar, sharpe, win_rate,
        avg_gross_ret, avg_net_ret, avg_mae, avg_mfe
    """
    top_k_list = [5, 10, 20]
    hold_days_list = [3, 5, 10, 20]
    strategy_list = ["sell_score", "low_risk", "composite"]

    all_summaries = []
    total = len(top_k_list) * len(hold_days_list) * len(strategy_list)
    done = 0

    for strategy in strategy_list:
        for top_k in top_k_list:
            for hold_days in hold_days_list:
                done += 1
                print(f"  [{done}/{total}] strategy={strategy}, top_k={top_k}, hold_days={hold_days}")

                result = run_backtest(
                    test_df, price_pivot, trading_days, prev_close_map,
                    max_stocks=top_k, hold_days=hold_days,
                    strategy=strategy, strict=True,
                )
                summary = compute_summary(result)
                all_summaries.append(summary)

                # 保存逐笔交易和净值
                tag = f"{strategy}_k{top_k}_h{hold_days}"
                if not result["nav_df"].empty:
                    result["nav_df"].to_csv(
                        os.path.join(BACKTEST_DIR, f"nav_{tag}.csv"), index=False
                    )
                if not result["trades_df"].empty:
                    result["trades_df"].to_csv(
                        os.path.join(BACKTEST_DIR, f"trades_{tag}.csv"), index=False
                    )

    return pd.DataFrame(all_summaries)


def plot_stability_matrix(matrix_df: pd.DataFrame, output_dir: str):
    """绘制稳定性矩阵热力图"""
    os.makedirs(output_dir, exist_ok=True)

    # 统一列名：max_stocks → top_k
    if "max_stocks" in matrix_df.columns and "top_k" not in matrix_df.columns:
        matrix_df = matrix_df.rename(columns={"max_stocks": "top_k"})
    if "hold_days" not in matrix_df.columns and "hold_days" in matrix_df.columns:
        pass  # already correct

    for metric in ["avg_net_ret", "sharpe", "win_rate", "calmar"]:
        if metric not in matrix_df.columns:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Stability Matrix: {metric}", fontsize=14, fontweight="bold")

        for ax, strategy in zip(axes, ["sell_score", "low_risk", "composite"]):
            sub = matrix_df[matrix_df["strategy"] == strategy].copy()
            if sub.empty:
                continue

            pivot = sub.pivot_table(index="hold_days", columns="top_k", values=metric)
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("top_k")
            ax.set_ylabel("hold_days")
            ax.set_title(strategy)

            # 标注数值
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

            fig.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        path = os.path.join(output_dir, f"matrix_{metric}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")


# ==================== 主函数 ====================
def main(args):
    print("=" * 60)
    print("真实持仓回测引擎")
    print("=" * 60)

    os.makedirs(BACKTEST_DIR, exist_ok=True)

    # 1. 加载test集预测
    print("\n[1/3] 加载test集数据...")
    test_path = os.path.join(OUTPUT_DIR, "test_predictions.parquet")
    if not os.path.exists(test_path):
        print(f"  错误: {test_path} 不存在，请先运行 02_train_gbdt_models.py")
        return

    test_df = pd.read_parquet(test_path)
    test_df["obs_date"] = pd.to_datetime(test_df["obs_date"])
    print(f"  test集: {len(test_df)} 行, 日期范围: {test_df['obs_date'].min()} ~ {test_df['obs_date'].max()}")

    # 只用obs_day=1的样本（买入决策点）
    test_df = test_df[test_df["obs_day"] == 1].copy()
    print(f"  obs_day=1: {len(test_df)} 行")

    if test_df.empty:
        print("  无可回测数据")
        return

    # 2. 加载K线数据
    print("\n[2/3] 加载K线数据...")
    # 需要覆盖信号日到信号日+hold_days的范围
    signal_start = str(test_df["obs_date"].min().date())
    # 最多持有20天，需要多加载30天确保有足够交易日
    signal_end_dt = test_df["obs_date"].max() + pd.Timedelta(days=50)
    signal_end = str(signal_end_dt.date())
    daily_prices = load_daily_prices(signal_start, signal_end)
    print(f"  K线记录: {len(daily_prices)}, 日期范围: {daily_prices['bar_time'].min()} ~ {daily_prices['bar_time'].max()}")

    price_pivot, trading_days, prev_close_map = build_price_pivot(daily_prices)
    print(f"  交易日数: {len(trading_days)}, 股票数: {len(prev_close_map)}")

    # 3. 运行回测
    print("\n[3/3] 运行回测...")
    fig_dir = os.path.join(BACKTEST_DIR, "figures")

    if args.matrix:
        # 稳定性矩阵模式
        print("\n  === 稳定性矩阵 ===")
        matrix_df = run_stability_matrix(test_df, price_pivot, trading_days, prev_close_map)

        # 保存矩阵
        matrix_path = os.path.join(BACKTEST_DIR, "stability_matrix.csv")
        matrix_df.to_csv(matrix_path, index=False)
        print(f"\n  稳定性矩阵: {matrix_path}")

        # 可视化
        plot_stability_matrix(matrix_df, fig_dir)

        # 打印矩阵
        print(f"\n{'='*80}")
        print("稳定性矩阵汇总")
        print(f"{'='*80}")
        key_cols = ["strategy", "top_k", "hold_days", "n_trades", "avg_gross_ret", "avg_net_ret",
                     "win_rate", "sharpe", "calmar", "max_dd"]
        existing_cols = [c for c in key_cols if c in matrix_df.columns]
        print(matrix_df[existing_cols].to_string(index=False))
    else:
        # 单次回测模式
        result = run_backtest(
            test_df, price_pivot, trading_days, prev_close_map,
            max_stocks=args.top_k, hold_days=args.hold_days,
            strategy=args.strategy, strict=True,
        )
        summary = compute_summary(result)

        # 打印结果
        print(f"\n  策略: {args.strategy}, top_k={args.top_k}, hold_days={args.hold_days}")
        print(f"  交易数: {summary.get('n_trades', 0)}")
        print(f"  最终净值: {summary.get('final_nav', 0):.4f}")
        print(f"  年化收益: {summary.get('annual_ret', 0):.2%}")
        print(f"  最大回撤: {summary.get('max_dd', 0):.2%}")
        print(f"  夏普比率: {summary.get('sharpe', 0):.2f}")
        print(f"  卡玛比率: {summary.get('calmar', 0):.2f}")
        print(f"  胜率: {summary.get('win_rate', 0):.2%}")
        print(f"  平均毛收益: {summary.get('avg_gross_ret', 0):.2%}")
        print(f"  平均净收益: {summary.get('avg_net_ret', 0):.2%}")
        print(f"  平均MAE: {summary.get('avg_mae', 0):.2%}")
        print(f"  平均MFE: {summary.get('avg_mfe', 0):.2%}")
        print(f"  跳过统计: {result['skipped_stats']}")

        # 保存
        tag = f"{args.strategy}_k{args.top_k}_h{args.hold_days}"
        if not result["nav_df"].empty:
            result["nav_df"].to_csv(os.path.join(BACKTEST_DIR, f"nav_{tag}.csv"), index=False)
        if not result["trades_df"].empty:
            result["trades_df"].to_csv(os.path.join(BACKTEST_DIR, f"trades_{tag}.csv"), index=False)

        # 画图
        plot_backtest(result, fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="真实持仓回测引擎")
    parser.add_argument("--top-k", type=int, default=20, help="每日最大持仓数")
    parser.add_argument("--hold-days", type=int, default=5, help="持仓天数")
    parser.add_argument("--strategy", type=str, default="sell_score",
                        choices=["sell_score", "low_risk", "composite"],
                        help="选股策略")
    parser.add_argument("--matrix", action="store_true",
                        help="运行稳定性矩阵（遍历top_k×hold_days×strategy）")
    args = parser.parse_args()
    main(args)
