#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[DEPRECATED] 模型预测驱动的动态交易回测引擎

⚠️ 已废弃：被 dynamic_exit_backtest_v2.py 替代。
   废弃原因: v2 移除 sell_reg exit（已被证伪无边际贡献），改为 buy-only exit（buy_cls > threshold）。
   新实验请使用 v2 引擎。

Purpose:
    (历史保留) 对比两种退出机制的回测效果：
    1) fixed_hold — 固定持有期
    2) dynamic — 基于模型实时预测的动态退出（含 sell_reg exit，已证伪）

Pipeline Position:
    实验逻辑（已废弃）。
    替代: dynamic_exit_backtest_v2.py

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - DB: stock_k_data

Outputs:
    - stop_experiment/output/backtest/dynamic/ 目录

How to Run:
    python stop_experiment/backtest/dynamic_exit_backtest.py --compare
    python stop_experiment/backtest/dynamic_exit_backtest.py --exit-mode dynamic --top-k 10 --max-hold 20
    python stop_experiment/backtest/dynamic_exit_backtest.py --matrix --exit-mode dynamic

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
    score_stocks, compute_summary, plot_backtest,
)

DYNAMIC_DIR = os.path.join(BACKTEST_DIR, "dynamic")


def build_prediction_lookup(full_pred_df: pd.DataFrame) -> dict:
    """
    构建预测值快速查找字典。
    Key: (signal_id, obs_date) → dict of predictions

    同时预计算 composite_score。
    """
    df = full_pred_df.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"])

    if "pred_sell_reg" not in df.columns:
        raise ValueError("full_test_predictions 缺少 pred_sell_reg 列")

    if "composite_score" not in df.columns:
        sell_score = df.get("pred_sell_reg", pd.Series(0, index=df.index))
        buy_score = -df.get("pred_buy_reg", pd.Series(0, index=df.index))
        df["composite_score"] = sell_score * 0.5 + buy_score * 0.5

    lookup = {}
    for _, row in df.iterrows():
        key = (int(row["signal_id"]), row["obs_date"])
        lookup[key] = {
            "pred_sell_reg": float(row.get("pred_sell_reg", np.nan)),
            "pred_buy_reg": float(row.get("pred_buy_reg", np.nan)),
            "pred_sell_cls": float(row.get("pred_sell_cls", np.nan)),
            "pred_buy_cls": float(row.get("pred_buy_cls", np.nan)),
            "composite_score": float(row.get("composite_score", np.nan)),
        }
    return lookup


def evaluate_dynamic_exit(
    holding: dict,
    code: str,
    current_date,
    prev_date,
    prediction_lookup: dict,
    sell_exit_threshold: float,
    buy_cls_exit_threshold: float,
    max_hold_days: int,
    stop_loss: float,
    day_close: pd.Series,
) -> tuple[bool, str]:
    """
    评估是否触发动态退出条件。

    前瞻偏差修正：
    - 模型预测使用前一交易日(prev_date)的特征，该特征在当日开盘前已确定
    - 退出决策在当日(current_date)执行
    - 买入当天(days_held==1)不检查模型预测（买入决策已足够筛选）

    Parameters
    ----------
    holding : dict with keys: signal_id, buy_price, buy_date, days_held
    code : 股票raw_code（6位）
    current_date : 当前交易日（卖出决策执行日）
    prev_date : 前一交易日（用于获取不含同日前瞻的预测值）
    prediction_lookup : (signal_id, date) → prediction dict
    sell_exit_threshold : pred_sell_reg 低于此值则上涨空间耗尽
    buy_cls_exit_threshold : pred_buy_cls 高于此值则下跌风险加大
    max_hold_days : 最大持有天数
    stop_loss : 止损线（负值，如 -0.07）
    day_close : 当日收盘价 Series（用于止损检查）

    Returns
    -------
    (should_exit: bool, reason: str)
    """
    days_held = holding["days_held"]
    signal_id = holding.get("signal_id")

    # 1. 强制最大持有期
    if days_held > max_hold_days:
        return True, "max_hold"

    # 2. 止损检查（用当日收盘价，盘中可能更低但收盘价是保守估计）
    if code in day_close.index:
        current_price = day_close[code]
        if not np.isnan(current_price) and current_price > 0:
            current_ret = (current_price - holding["buy_price"]) / holding["buy_price"]
            if current_ret < stop_loss:
                return True, "stop_loss"

    # 3. 模型预测退出（买入当天不检查，留给模型至少一天观察期）
    if signal_id is not None and days_held > 1 and prev_date is not None:
        pred_key = (int(signal_id), prev_date)
        pred = prediction_lookup.get(pred_key)
        if pred is not None:
            pred_sell_reg = pred.get("pred_sell_reg", np.nan)
            pred_buy_cls = pred.get("pred_buy_cls", np.nan)

            if not np.isnan(pred_buy_cls) and pred_buy_cls > buy_cls_exit_threshold:
                return True, "downside_risk"
            if not np.isnan(pred_sell_reg) and pred_sell_reg < sell_exit_threshold:
                return True, "upside_exhausted"

    return False, ""


def run_dynamic_backtest(
    signals_df: pd.DataFrame,
    price_pivot: pd.DataFrame,
    trading_days: list,
    prev_close_map: dict,
    prediction_lookup: dict,
    max_stocks: int = 20,
    strategy: str = "sell_score",
    exit_mode: str = "dynamic",
    hold_days: int = 5,
    sell_exit_threshold: float = -0.02,
    buy_cls_exit_threshold: float = 0.6,
    max_hold_days: int = 20,
    stop_loss: float = -0.07,
    strict: bool = True,
) -> dict:
    """
    动态退出回测引擎。

    exit_mode='fixed_hold': 固定持有hold_days天后退出（baseline）
    exit_mode='dynamic': 基于模型预测动态退出

    时序：
    - T日 obs_day=1 出现信号 → T+1日 open 买入
    - 持仓每日评估退出条件
    - 同一股票不重复开仓
    """
    signals_df = score_stocks(signals_df, strategy)
    signals_sorted = signals_df.sort_values(["obs_date", "score"], ascending=[True, False])

    signal_dates = sorted(signals_df["obs_date"].unique())
    signal_by_date = {}
    for date in signal_dates:
        day_sigs = signals_sorted[signals_sorted["obs_date"] == date]
        signal_by_date[date] = day_sigs.drop_duplicates(subset=["ts_code"], keep="first")

    holdings = {}
    pending_orders = []
    trade_details = []
    nav_records = []
    skipped_counts = defaultdict(int)

    for t_idx, current_date in enumerate(trading_days):
        if current_date not in price_pivot.index:
            continue

        day_open = price_pivot.loc[current_date, "open"] if "open" in price_pivot else pd.Series(dtype=float)
        day_high = price_pivot.loc[current_date, "high"] if "high" in price_pivot else pd.Series(dtype=float)
        day_low = price_pivot.loc[current_date, "low"] if "low" in price_pivot else pd.Series(dtype=float)
        day_close = price_pivot.loc[current_date, "close"] if "close" in price_pivot else pd.Series(dtype=float)

        # ---- Step 1: 执行pending买单 ----
        if pending_orders:
            executed = []
            for order in pending_orders:
                code, buy_price, ts_code, score, signal_id = order
                if code in holdings:
                    skipped_counts["already_held"] += 1
                    continue
                if strict:
                    if "volume" in price_pivot and code in price_pivot["volume"].columns:
                        vol_c = price_pivot["volume"][code].get(current_date, np.nan)
                        if is_suspended(vol_c):
                            skipped_counts["suspended"] += 1
                            continue
                    if code in prev_close_map and current_date in prev_close_map[code].index:
                        prev_c = prev_close_map[code].get(current_date, np.nan)
                        if not np.isnan(prev_c) and prev_c > 0:
                            dh = price_pivot["high"][code]
                            if current_date in dh.index:
                                if is_limit_up(buy_price, dh.get(current_date, np.nan), prev_c):
                                    skipped_counts["limit_up"] += 1
                                    continue
                executed.append((code, buy_price, ts_code, score, signal_id))

            if executed:
                n_total = len(holdings) + len(executed)
                w = 1.0 / n_total
                for code_h in holdings:
                    holdings[code_h]["weight"] = w
                for code, bp, ts_code, score, signal_id in executed:
                    holdings[code] = {
                        "buy_date": current_date,
                        "buy_price": bp,
                        "weight": w,
                        "days_held": 0,
                        "ts_code": ts_code,
                        "score": score,
                        "signal_id": signal_id,
                    }
            pending_orders = []

        # ---- Step 2: 递增天数 + 卖出逻辑 ----
        to_sell = []
        for code, h in list(holdings.items()):
            h["days_held"] += 1

            if exit_mode == "fixed_hold":
                should_sell = h["days_held"] > hold_days
                sell_reason = "fixed"
            else:
                prev_date = trading_days[t_idx - 1] if t_idx > 0 else None
                should_sell, sell_reason = evaluate_dynamic_exit(
                    h, code, current_date, prev_date, prediction_lookup,
                    sell_exit_threshold, buy_cls_exit_threshold,
                    max_hold_days, stop_loss, day_close,
                )

            if not should_sell:
                continue

            sell_price = np.nan
            if code in day_open.index and not np.isnan(day_open[code]):
                sell_price = day_open[code]
            if np.isnan(sell_price) or sell_price <= 0:
                continue

            if strict:
                if "volume" in price_pivot and code in price_pivot["volume"].columns:
                    vol_c = price_pivot["volume"][code].get(current_date, np.nan)
                    if is_suspended(vol_c):
                        skipped_counts["suspended_sell"] += 1
                        continue
                if code in prev_close_map and current_date in prev_close_map[code].index:
                    prev_c = prev_close_map[code].get(current_date, np.nan)
                    if not np.isnan(prev_c) and prev_c > 0:
                        if code in day_low.index and not np.isnan(day_low[code]):
                            if is_limit_down(sell_price, day_low[code], prev_c):
                                skipped_counts["limit_down"] += 1
                                continue

            gross_ret = (sell_price - h["buy_price"]) / h["buy_price"]
            net_ret = gross_ret - BUY_COST - SELL_COST

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
                "sell_reason": sell_reason,
                "score": h.get("score", 0),
                "strategy": strategy,
                "exit_mode": exit_mode,
            })
            to_sell.append(code)

        for code in to_sell:
            del holdings[code]

        # ---- Step 3: 计算NAV ----
        daily_ret = 0.0
        for code, h in holdings.items():
            if code in day_close.index and not np.isnan(day_close[code]):
                if h["days_held"] == 1:
                    if code in day_open.index and not np.isnan(day_open[code]):
                        stock_ret = (day_close[code] - day_open[code]) / day_open[code]
                    else:
                        stock_ret = 0
                else:
                    if t_idx > 0:
                        prev_date = trading_days[t_idx - 1]
                        if prev_date in price_pivot.index:
                            prev_c = price_pivot.loc[prev_date, "close"]
                            if code in prev_c.index and not np.isnan(prev_c[code]):
                                stock_ret = (day_close[code] - prev_c[code]) / prev_c[code]
                            else:
                                stock_ret = 0
                        else:
                            stock_ret = 0
                    else:
                        stock_ret = 0
                daily_ret += h["weight"] * stock_ret

        prev_nav = nav_records[-1]["nav"] if nav_records else 1.0
        nav = prev_nav * (1 + daily_ret)

        nav_records.append({
            "date": current_date,
            "nav": nav,
            "daily_ret": daily_ret,
            "n_positions": len(holdings),
            "cash_ratio": max(0, 1.0 - sum(h["weight"] for h in holdings.values())),
        })

        # ---- Step 4: 收集信号生成pending ----
        if current_date in signal_by_date:
            day_signals = signal_by_date[current_date]
        else:
            day_signals = pd.DataFrame()

        n_available = max_stocks - len(holdings) - len(pending_orders)
        if n_available > 0 and len(day_signals) > 0:
            next_idx = t_idx + 1
            if next_idx < len(trading_days):
                buy_date_ts = trading_days[next_idx]
                if buy_date_ts in price_pivot.index:
                    buy_day_open = price_pivot.loc[buy_date_ts, "open"]
                    for _, sig in day_signals.iterrows():
                        if len(holdings) + len(pending_orders) >= max_stocks:
                            break
                        code = sig["ts_code"][:6] if "." in sig["ts_code"] else sig["ts_code"]
                        if code in holdings or any(o[0] == code for o in pending_orders):
                            skipped_counts["already_held"] += 1
                            continue
                        if code not in buy_day_open.index or np.isnan(buy_day_open[code]):
                            continue
                        bp = buy_day_open[code]
                        if bp <= 0:
                            continue
                        sid = sig.get("signal_id", None)
                        pending_orders.append((code, bp, sig["ts_code"], sig.get("score", 0), sid))

    nav_df = pd.DataFrame(nav_records)
    trades_df = pd.DataFrame(trade_details)
    return {
        "nav_df": nav_df,
        "trades_df": trades_df,
        "skipped_stats": dict(skipped_counts),
        "params": {
            "max_stocks": max_stocks,
            "hold_days": hold_days,
            "strategy": strategy,
            "exit_mode": exit_mode,
            "sell_exit_threshold": sell_exit_threshold,
            "buy_cls_exit_threshold": buy_cls_exit_threshold,
            "max_hold_days": max_hold_days,
            "stop_loss": stop_loss,
            "strict": strict,
        },
    }


def plot_comparison(results: dict, output_dir: str):
    """
    对比固定持有 vs 动态退出的净值曲线和回撤。
    results: {"fixed": result_dict, "dynamic": result_dict}
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Fixed Hold vs Dynamic Exit Comparison", fontsize=14, fontweight="bold")

    colors = {"fixed": "steelblue", "dynamic": "crimson"}

    # 1. NAV对比
    ax = axes[0, 0]
    for mode, res in results.items():
        ndf = res["nav_df"]
        if ndf.empty:
            continue
        ndf_plot = ndf.copy()
        ax.plot(ndf_plot["date"], ndf_plot["nav"], label=f"{mode}", color=colors[mode], linewidth=1.5)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3)
    ax.set_title("NAV Curve")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    # 2. 回撤对比
    ax = axes[0, 1]
    for mode, res in results.items():
        ndf_plot = res["nav_df"].copy()
        if ndf_plot.empty:
            continue
        ndf_plot["cummax"] = ndf_plot["nav"].cummax()
        ndf_plot["drawdown"] = (ndf_plot["nav"] - ndf_plot["cummax"]) / ndf_plot["cummax"]
        ax.fill_between(ndf_plot["date"], ndf_plot["drawdown"], 0,
                         color=colors[mode], alpha=0.3, label=f"{mode}")
    ax.set_title("Drawdown")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    # 3. 持仓天数直方图
    ax = axes[1, 0]
    for mode, res in results.items():
        tdf = res["trades_df"]
        if tdf.empty:
            continue
        ax.hist(tdf["hold_days"], bins=20, alpha=0.5, label=f"{mode} (mean={tdf['hold_days'].mean():.1f})",
                color=colors[mode], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Hold Days")
    ax.set_ylabel("Count")
    ax.set_title("Hold Days Distribution")
    ax.legend()

    # 4. 退出原因分布（仅dynamic）
    ax = axes[1, 1]
    dyn_trades = results.get("dynamic", {}).get("trades_df", pd.DataFrame())
    if not dyn_trades.empty and "sell_reason" in dyn_trades.columns:
        reason_counts = dyn_trades["sell_reason"].value_counts()
        ax.pie(reason_counts.values, labels=reason_counts.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Dynamic Exit Reasons")
    else:
        ax.set_title("Exit Reasons - no data")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center")

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def print_comparison_table(fixed_summary: dict, dynamic_summary: dict):
    """打印固定 vs 动态对比表"""
    key_metrics = ["n_trades", "final_nav", "annual_ret", "max_dd",
                   "calmar", "sharpe", "win_rate", "avg_net_ret",
                   "avg_mae", "avg_mfe", "avg_hold_days"]

    print(f"\n{'='*80}")
    print(f"{'指标':20s} {'固定持有(baseline)':>20s} {'动态退出':>20s} {'改善':>15s}")
    print(f"{'='*80}")

    for m in key_metrics:
        f_val = fixed_summary.get(m, np.nan)
        d_val = dynamic_summary.get(m, np.nan)

        # 改善：越高越好 (nav/ret/sharpe/win_rate) 或 越低越好 (mdd/mae)
        higher_better = m in ["final_nav", "annual_ret", "calmar", "sharpe", "win_rate",
                              "avg_net_ret", "avg_gross_ret", "avg_mfe", "n_trades"]
        if higher_better:
            improvement = d_val - f_val if pd.notna(f_val) and pd.notna(d_val) else np.nan
        else:
            improvement = f_val - d_val if pd.notna(f_val) and pd.notna(d_val) else np.nan

        f_str = f"{f_val:.4f}" if pd.notna(f_val) else "N/A"
        d_str = f"{d_val:.4f}" if pd.notna(d_val) else "N/A"
        imp_str = f"{improvement:+.4f}" if pd.notna(improvement) else "N/A"

        direction = "↑" if higher_better else "↓"
        print(f"  {m:20s} {f_str:>20s} {d_str:>20s} {imp_str:>12s}  {direction}")

    print(f"{'='*80}")


def _load_data():
    """加载全量预测和K线数据，返回公共数据结构"""
    full_pred_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    if not os.path.exists(full_pred_path):
        raise FileNotFoundError(f"{full_pred_path} 不存在，请先运行 generate_full_predictions.py")

    full_pred = pd.read_parquet(full_pred_path)
    full_pred["obs_date"] = pd.to_datetime(full_pred["obs_date"])
    prediction_lookup = build_prediction_lookup(full_pred)

    test_df = full_pred[full_pred["obs_day"] == 1].copy()
    if test_df.empty:
        raise ValueError("无可回测数据")

    signal_start = str(test_df["obs_date"].min().date())
    signal_end_dt = test_df["obs_date"].max() + pd.Timedelta(days=50)
    signal_end = str(signal_end_dt.date())
    daily_prices = load_daily_prices(signal_start, signal_end)
    price_pivot, trading_days, prev_close_map = build_price_pivot(daily_prices)

    return test_df, price_pivot, trading_days, prev_close_map, prediction_lookup


def run_bulk_compare(args):
    """批量对比模式：遍历 top_k × hold_days × strategy × exit_mode"""
    print("=" * 60)
    print("批量对比：固定持有 vs 动态退出")
    print("=" * 60)

    os.makedirs(DYNAMIC_DIR, exist_ok=True)

    print("\n[1/2] 加载数据...")
    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data()
    print(f"  obs_day=1: {len(test_df)} 行, 交易日: {len(trading_days)}, 股票: {len(prev_close_map)}")

    top_k_list = [5, 10, 20]
    hold_days_list = [3, 5, 10]
    strategy_list = ["sell_score", "low_risk", "composite"]
    exit_mode_list = ["fixed_hold", "dynamic"]

    all_rows = []
    total = len(top_k_list) * len(hold_days_list) * len(strategy_list) * len(exit_mode_list)
    done = 0

    print(f"\n[2/2] 运行 {total} 个组合...")
    for strategy in strategy_list:
        for top_k in top_k_list:
            for hold_days in hold_days_list:
                for exit_mode in exit_mode_list:
                    done += 1

                    if exit_mode == "fixed_hold":
                        result = run_dynamic_backtest(
                            test_df, price_pivot, trading_days, prev_close_map,
                            pred_lookup,
                            max_stocks=top_k, strategy=strategy,
                            exit_mode="fixed_hold", hold_days=hold_days,
                            strict=True,
                        )
                    else:
                        result = run_dynamic_backtest(
                            test_df, price_pivot, trading_days, prev_close_map,
                            pred_lookup,
                            max_stocks=top_k, strategy=strategy,
                            exit_mode="dynamic",
                            sell_exit_threshold=args.sell_exit_threshold,
                            buy_cls_exit_threshold=args.buy_cls_exit_threshold,
                            max_hold_days=args.max_hold_days,
                            stop_loss=args.stop_loss,
                            strict=True,
                        )

                    summary = compute_summary(result)
                    row = {
                        "strategy": strategy,
                        "top_k": top_k,
                        "hold_days": hold_days if exit_mode == "fixed_hold" else args.max_hold_days,
                        "exit_mode": exit_mode,
                        **{k: v for k, v in summary.items() if not isinstance(v, dict)},
                    }
                    all_rows.append(row)

                    # 保存
                    tag = f"{strategy}_{exit_mode}_k{top_k}_h{hold_days}"
                    if not result["nav_df"].empty:
                        result["nav_df"].to_csv(os.path.join(DYNAMIC_DIR, f"nav_{tag}.csv"), index=False)
                    if not result["trades_df"].empty:
                        result["trades_df"].to_csv(os.path.join(DYNAMIC_DIR, f"trades_{tag}.csv"), index=False)

                    fixed_flag = "F" if exit_mode == "fixed_hold" else "D"
                    print(f"  [{done:3d}/{total}] {strategy:12s} k={top_k:2d} h={hold_days:2d} {fixed_flag} "
                          f"nav={summary.get('final_nav', 0):.4f} win={summary.get('win_rate', 0):.2%} "
                          f"sharpe={summary.get('sharpe', 0):.2f} n={summary.get('n_trades', 0)}")

    bulk_df = pd.DataFrame(all_rows)

    # 保存汇总
    bulk_path = os.path.join(DYNAMIC_DIR, "bulk_comparison.csv")
    bulk_df.to_csv(bulk_path, index=False)
    print(f"\n  批量对比汇总: {bulk_path}")

    # 打印对比表：按 (strategy, top_k, hold_days) pivot
    print(f"\n{'='*120}")
    print("对比总表 (fixed vs dynamic 关键指标)")
    print(f"{'='*120}")

    key_metrics = ["final_nav", "sharpe", "win_rate", "avg_net_ret", "avg_hold_days", "n_trades"]
    for metric in key_metrics:
        if metric not in bulk_df.columns:
            continue
        print(f"\n  === {metric} ===")
        print(f"  {'strategy':12s} {'top_k':>5s} {'fixed':>10s} {'dynamic':>10s} {'diff':>10s}")

        for strategy in strategy_list:
            for top_k in top_k_list:
                for hold_days in hold_days_list:
                    f_row = bulk_df[(bulk_df["strategy"] == strategy) &
                                    (bulk_df["top_k"] == top_k) &
                                    (bulk_df["hold_days"] == hold_days) &
                                    (bulk_df["exit_mode"] == "fixed_hold")]
                    d_row = bulk_df[(bulk_df["strategy"] == strategy) &
                                    (bulk_df["top_k"] == top_k) &
                                    (bulk_df["hold_days"] == hold_days) &
                                    (bulk_df["exit_mode"] == "dynamic")]

                    f_val = f_row[metric].iloc[0] if len(f_row) > 0 else np.nan
                    d_val = d_row[metric].iloc[0] if len(d_row) > 0 else np.nan

                    if pd.isna(f_val) or pd.isna(d_val):
                        continue
                    diff = d_val - f_val

                    f_str = f"{f_val:.4f}" if abs(f_val) < 100 else f"{f_val:.1f}"
                    d_str = f"{d_val:.4f}" if abs(d_val) < 100 else f"{d_val:.1f}"
                    diff_str = f"{diff:+.4f}" if abs(diff) < 100 else f"{diff:+.1f}"

                    print(f"  {strategy:12s} k={top_k:2d} h={hold_days:2d}  {f_str:>10s} {d_str:>10s} {diff_str:>10s}")

    # 绘图：热力图对比
    _plot_bulk_heatmap(bulk_df, top_k_list, hold_days_list, strategy_list)


def _plot_bulk_heatmap(bulk_df, top_k_list, hold_days_list, strategy_list):
    """绘制批量对比热力图"""
    fig_dir = os.path.join(DYNAMIC_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    for metric in ["final_nav", "sharpe", "win_rate", "avg_net_ret"]:
        if metric not in bulk_df.columns:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Bulk Comparison: Fixed vs Dynamic ({metric})", fontsize=14, fontweight="bold")

        for row_idx, exit_mode in enumerate(["fixed_hold", "dynamic"]):
            for col_idx, strategy in enumerate(strategy_list):
                ax = axes[row_idx, col_idx]
                sub = bulk_df[(bulk_df["exit_mode"] == exit_mode) &
                              (bulk_df["strategy"] == strategy)]
                if sub.empty:
                    ax.set_title(f"{exit_mode[:5]}-{strategy[:6]} (no data)")
                    continue

                pivot = sub.pivot_table(index="hold_days", columns="top_k", values=metric)
                if pivot.empty:
                    continue

                im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([f"k={c}" for c in pivot.columns])
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([f"h={r}" for r in pivot.index])
                ax.set_title(f"{exit_mode[:5]} | {strategy[:8]}")
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        val = pivot.values[i, j]
                        if not np.isnan(val):
                            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)
                fig.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        path = os.path.join(fig_dir, f"bulk_{metric}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  保存: {path}")


def run_grid_search(args):
    """网格搜索模式：遍历 exit 阈值参数找最优组合"""
    print("=" * 60)
    print("动态退出阈值参数网格搜索")
    print("=" * 60)

    os.makedirs(DYNAMIC_DIR, exist_ok=True)

    print("\n[1/2] 加载数据...")
    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data()
    print(f"  obs_day=1: {len(test_df)} 行, 交易日: {len(trading_days)}")

    # 网格参数
    sell_exit_list = [-0.03, -0.02, -0.01, 0.0, 0.01]
    buy_cls_list = [0.4, 0.5, 0.6, 0.7]
    max_hold_list = [10, 15, 20]
    stop_loss_list = [-0.05, -0.07, -0.10]

    all_rows = []
    total = len(sell_exit_list) * len(buy_cls_list) * len(max_hold_list) * len(stop_loss_list)
    done = 0

    print(f"\n[2/2] 网格搜索: sell_exit×{len(sell_exit_list)} × buy_cls×{len(buy_cls_list)}"
          f" × max_hold×{len(max_hold_list)} × stop_loss×{len(stop_loss_list)} = {total}")

    for sell_exit in sell_exit_list:
        for buy_cls in buy_cls_list:
            for max_hold in max_hold_list:
                for stop_loss in stop_loss_list:
                    done += 1
                    result = run_dynamic_backtest(
                        test_df, price_pivot, trading_days, prev_close_map,
                        pred_lookup,
                        max_stocks=args.top_k, strategy=args.strategy,
                        exit_mode="dynamic",
                        sell_exit_threshold=sell_exit,
                        buy_cls_exit_threshold=buy_cls,
                        max_hold_days=max_hold,
                        stop_loss=stop_loss,
                        strict=True,
                    )
                    summary = compute_summary(result)
                    row = {
                        "sell_exit_threshold": sell_exit,
                        "buy_cls_exit_threshold": buy_cls,
                        "max_hold_days": max_hold,
                        "stop_loss": stop_loss,
                        **{k: v for k, v in summary.items() if not isinstance(v, dict)},
                    }

                    # 退出原因统计
                    tdf = result.get("trades_df", pd.DataFrame())
                    if not tdf.empty and "sell_reason" in tdf.columns:
                        reasons = tdf["sell_reason"].value_counts().to_dict()
                        for reason in ["upside_exhausted", "downside_risk", "stop_loss", "max_hold"]:
                            row[f"n_{reason}"] = reasons.get(reason, 0)

                    all_rows.append(row)

                    se_str = f"se={sell_exit:.2f}" if sell_exit != 0 else "se=0.00"
                    print(f"  [{done:4d}/{total}] {se_str} bc={buy_cls:.1f} mh={max_hold:2d} sl={stop_loss:.2f}"
                          f" nav={summary.get('final_nav', 0):.4f} sharpe={summary.get('sharpe', 0):.2f}"
                          f" win={summary.get('win_rate', 0):.2%} n={summary.get('n_trades', 0)}")

    grid_df = pd.DataFrame(all_rows)

    # 保存
    grid_path = os.path.join(DYNAMIC_DIR, "grid_search_results.csv")
    grid_df.to_csv(grid_path, index=False)
    print(f"\n  网格搜索结果: {grid_path}")

    # 找出最优
    print(f"\n{'='*80}")
    print("最优参数组合 (按 final_nav 排序 Top 10)")
    print(f"{'='*80}")
    key_cols = ["sell_exit_threshold", "buy_cls_exit_threshold", "max_hold_days", "stop_loss",
                "final_nav", "sharpe", "win_rate", "avg_net_ret", "avg_hold_days", "n_trades",
                "n_downside_risk", "n_stop_loss", "n_max_hold", "n_upside_exhausted"]
    avail_cols = [c for c in key_cols if c in grid_df.columns]
    top10 = grid_df.nlargest(10, "final_nav")[avail_cols]
    print(top10.to_string(index=False))

    # 绘图
    fig_dir = os.path.join(DYNAMIC_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Grid Search Heatmap ({args.strategy}, k={args.top_k})", fontsize=14, fontweight="bold")

    for idx, (param, ax) in enumerate(zip(
        ["max_hold_days", "buy_cls_exit_threshold", "sell_exit_threshold", "stop_loss"],
        axes.flatten(),
    )):
        if param not in grid_df.columns:
            continue
        grouped = grid_df.groupby(param)["final_nav"].agg(["mean", "std"])
        ax.errorbar(grouped.index.astype(str), grouped["mean"], yerr=grouped["std"],
                    fmt="o-", capsize=5, color=f"C{idx}", linewidth=2, markersize=8)
        ax.set_title(f"NAV vs {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("Mean Final NAV")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(fig_dir, "grid_sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {path}")


def main(args):
    print("=" * 60)
    print("模型预测驱动的动态交易回测引擎")
    print("=" * 60)

    os.makedirs(DYNAMIC_DIR, exist_ok=True)

    if args.bulk_compare:
        run_bulk_compare(args)
        return

    if args.grid_search:
        run_grid_search(args)
        return

    fig_dir = os.path.join(DYNAMIC_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data()
    print(f"  obs_day=1: {len(test_df)} 行, 交易日: {len(trading_days)}, 股票: {len(prev_close_map)}")

    # 2. 运行回测
    print("\n[2/4] 运行回测...")
    results = {}

    # 固定持有 baseline
    print("\n  --- 固定持有 (baseline) ---")
    fixed_result = run_dynamic_backtest(
        test_df, price_pivot, trading_days, prev_close_map,
        pred_lookup,
        max_stocks=args.top_k, strategy=args.strategy,
        exit_mode="fixed_hold", hold_days=args.hold_days,
        strict=True,
    )
    fixed_summary = compute_summary(fixed_result)
    results["fixed"] = fixed_result
    _print_summary("固定持有", fixed_summary)

    # 动态退出
    print(f"\n  --- 动态退出 (sell_threshold={args.sell_exit_threshold}, "
          f"buy_cls_threshold={args.buy_cls_exit_threshold}, max_hold={args.max_hold_days}) ---")
    dynamic_result = run_dynamic_backtest(
        test_df, price_pivot, trading_days, prev_close_map,
        pred_lookup,
        max_stocks=args.top_k, strategy=args.strategy,
        exit_mode="dynamic",
        sell_exit_threshold=args.sell_exit_threshold,
        buy_cls_exit_threshold=args.buy_cls_exit_threshold,
        max_hold_days=args.max_hold_days,
        stop_loss=args.stop_loss,
        strict=True,
    )
    dynamic_summary = compute_summary(dynamic_result)
    results["dynamic"] = dynamic_result
    _print_summary("动态退出", dynamic_summary)

    # 3. 对比分析
    print("\n[3/4] 对比分析...")
    print_comparison_table(fixed_summary, dynamic_summary)

    # 4. 保存与可视化
    print("\n[4/4] 保存...")
    tag_fixed = f"{args.strategy}_fixed_k{args.top_k}_h{args.hold_days}"
    tag_dynamic = f"{args.strategy}_dynamic_k{args.top_k}_mh{args.max_hold_days}"
    for mode, res in results.items():
        tag = tag_fixed if mode == "fixed" else tag_dynamic
        if not res["nav_df"].empty:
            res["nav_df"].to_csv(os.path.join(DYNAMIC_DIR, f"nav_{tag}.csv"), index=False)
        if not res["trades_df"].empty:
            res["trades_df"].to_csv(os.path.join(DYNAMIC_DIR, f"trades_{tag}.csv"), index=False)

    comparison_rows = [
        {"exit_mode": "fixed", **{k: v for k, v in fixed_summary.items() if not isinstance(v, dict)}},
        {"exit_mode": "dynamic", **{k: v for k, v in dynamic_summary.items() if not isinstance(v, dict)}},
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(DYNAMIC_DIR, "comparison_summary.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"  对比汇总: {comparison_path}")

    plot_comparison(results, fig_dir)

    dyn_trades = dynamic_result.get("trades_df", pd.DataFrame())
    if not dyn_trades.empty and "sell_reason" in dyn_trades.columns:
        print(f"\n  动态退出原因分布:")
        for reason, cnt in dyn_trades["sell_reason"].value_counts().items():
            print(f"    {reason}: {cnt} ({cnt/len(dyn_trades)*100:.1f}%)")


def _print_summary(label: str, summary: dict):
    print(f"    [{label}] n_trades={summary.get('n_trades', 0)}, "
          f"nav={summary.get('final_nav', 0):.4f}, "
          f"ann_ret={summary.get('annual_ret', 0):.2%}, "
          f"mdd={summary.get('max_dd', 0):.2%}, "
          f"sharpe={summary.get('sharpe', 0):.2f}, "
          f"calmar={summary.get('calmar', 0):.2f}, "
          f"win_rate={summary.get('win_rate', 0):.2%}, "
          f"avg_net={summary.get('avg_net_ret', 0):.2%}, "
          f"avg_hold={summary.get('avg_hold_days', 0):.1f}d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型预测驱动的动态交易回测引擎")
    parser.add_argument("--top-k", type=int, default=20, help="每日最大持仓数")
    parser.add_argument("--hold-days", type=int, default=5, help="固定持有天数")
    parser.add_argument("--strategy", type=str, default="sell_score",
                        choices=["sell_score", "low_risk", "composite"])
    parser.add_argument("--bulk-compare", action="store_true",
                        help="批量对比模式：遍历top_k×hold_days×strategy×exit_mode")
    parser.add_argument("--grid-search", action="store_true",
                        help="网格搜索模式：遍历exit阈值找最优参数")
    # 动态退出参数
    parser.add_argument("--sell-exit-threshold", type=float, default=-0.02,
                        help="sell_reg退出阈值（低于此值→上涨耗尽）")
    parser.add_argument("--buy-cls-exit-threshold", type=float, default=0.6,
                        help="buy_cls退出阈值（高于此值→下跌风险）")
    parser.add_argument("--max-hold-days", type=int, default=20,
                        help="最大持有天数")
    parser.add_argument("--stop-loss", type=float, default=-0.07,
                        help="止损线（负值）")
    args = parser.parse_args()
    main(args)
