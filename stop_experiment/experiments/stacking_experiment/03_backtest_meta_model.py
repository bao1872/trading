#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacking 元模型回测对比：接入 dynamic_exit_backtest_v2 引擎

Purpose:
    将元模型预测作为 score 列，接入回测引擎，对比 NAV/Sharpe/MDD/胜率。

Inputs:
    - results/meta_test_predictions.parquet (01_train_meta_model.py 产出)
    - DB: stock_k_data (K线数据)

Outputs:
    - results/backtest/backtest_comparison.csv (回测对比结果)

How to Run:
    python -m stop_experiment.experiments.stacking_experiment.03_backtest_meta_model

Side Effects:
    - 只读 meta_test_predictions.parquet 和 DB
    - 输出仅写入 results/backtest/
"""

from __future__ import annotations

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, BUY_COST, SELL_COST,
)
from stop_experiment.backtest.simple_backtest import (
    load_daily_prices, build_price_pivot,
    is_limit_up, is_limit_down, is_suspended,
)
from stop_experiment.backtest.decision_core import (
    find_exit_pred,
    decide_eod,
)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
BACKTEST_DIR = os.path.join(RESULTS_DIR, "backtest")

META_PRED_COLS = [
    ("meta_stack_lgb_sell_reg", "stack_lgb_sell_reg"),
    ("meta_stack_lgb_sell_cls", "stack_lgb_sell_cls"),
    ("meta_stack_lgb_composite", "stack_lgb_composite"),
]

MAX_HOLD_DAYS = 20
STOP_LOSS = -0.07
BUY_CLS_EXIT_THRESHOLD = 0.70


def compute_backtest_summary(result: dict) -> dict:
    nav_df = result["nav_df"].copy()
    trades_df = result["trades_df"]

    if nav_df.empty:
        return {"n_trades": 0, "final_nav": 1.0, "sharpe": 0, "max_dd": 0,
                "win_rate": 0, "avg_net_ret": 0, "avg_hold_days": 0}

    nav_df["cummax"] = nav_df["nav"].cummax()
    nav_df["drawdown"] = (nav_df["nav"] - nav_df["cummax"]) / nav_df["cummax"]

    total_days = len(nav_df)
    total_years = total_days / 252
    final_nav = nav_df["nav"].iloc[-1]
    annual_ret = (final_nav ** (1 / total_years) - 1) if total_years > 0 and final_nav > 0 else 0
    max_dd = nav_df["drawdown"].min()
    daily_rets = nav_df["daily_ret"]
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 1e-6 else 0

    n_trades = len(trades_df)
    win_rate = (trades_df["net_ret"] > 0).mean() if n_trades > 0 else 0
    avg_net_ret = trades_df["net_ret"].mean() if n_trades > 0 else 0
    avg_hold = trades_df["hold_days"].mean() if n_trades > 0 else 0

    return {
        "n_trades": n_trades,
        "final_nav": final_nav,
        "annual_ret": annual_ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "avg_net_ret": avg_net_ret,
        "avg_hold_days": avg_hold,
    }


def build_pred_lookup(df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in df.iterrows():
        pred_dict = {
            "pred_buy_cls": float(row.get("pred_buy_cls", np.nan)),
            "pred_sell_reg": float(row.get("pred_sell_reg", np.nan)),
            "pred_sell_cls": float(row.get("pred_sell_cls", np.nan)),
            "pred_buy_reg": float(row.get("pred_buy_reg", np.nan)),
        }
        sid_key = (int(row["signal_id"]), row["obs_date"])
        lookup[sid_key] = pred_dict
        ts_code = row.get("ts_code")
        if ts_code:
            ts_key = (ts_code, row["obs_date"])
            lookup[ts_key] = pred_dict
    return lookup


def run_backtest(
    signals_df, price_pivot, trading_days, prev_close_map,
    pred_lookup, max_stocks=10, score_col="score",
    buy_cost=0.001, sell_cost=0.001,
):
    signals_sorted = signals_df.sort_values([score_col, "ts_code"], ascending=[False, True])

    signal_dates = sorted(signals_df["obs_date"].unique())
    signal_by_date = {}
    for date in signal_dates:
        day_sigs = signals_sorted[signals_sorted["obs_date"] == date]
        signal_by_date[date] = day_sigs.drop_duplicates(subset=["ts_code"], keep="first")

    holdings = {}
    pending_orders = []
    pending_sells = []
    trade_details = []
    nav_records = []
    skipped = defaultdict(int)

    for t_idx, current_date in enumerate(trading_days):
        if current_date not in price_pivot.index:
            continue

        day_open = price_pivot.loc[current_date, "open"] if "open" in price_pivot else pd.Series(dtype=float)
        day_close = price_pivot.loc[current_date, "close"] if "close" in price_pivot else pd.Series(dtype=float)

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
                net_ret = gross_ret - buy_cost - sell_cost
                trade_details.append({
                    "ts_code": h["ts_code"], "buy_date": h["buy_date"],
                    "sell_date": current_date, "buy_price": h["buy_price"],
                    "sell_price": sell_price, "hold_days": h["days_held"],
                    "gross_ret": gross_ret, "net_ret": net_ret,
                    "sell_reason": sell_item["reason"],
                    "score": h.get("score", 0),
                })
                if code in holdings:
                    del holdings[code]

            pending_sells = []

        _buy_max_slots = max_stocks - len(holdings)
        if _buy_max_slots < 0:
            _buy_max_slots = 0
        if pending_orders:
            executed = []
            for code, bp, ts_code, sc, sid in pending_orders:
                if len(executed) >= _buy_max_slots:
                    break
                if code in holdings:
                    skipped["already_held"] += 1
                    continue
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
                n = len(holdings) + len(executed)
                w = 1.0 / n
                for code_h in holdings:
                    holdings[code_h]["weight"] = w
                for code, bp, ts_code, sc, sid in executed:
                    holdings[code] = {
                        "buy_date": current_date, "buy_price": bp,
                        "weight": w, "days_held": 0,
                        "ts_code": ts_code, "score": sc, "signal_id": sid,
                    }
            pending_orders = []

        prev_date = trading_days[t_idx - 1] if t_idx > 0 else None
        next_idx = t_idx + 1
        day_open_next = price_pivot.loc[trading_days[next_idx], "open"] if next_idx < len(trading_days) else pd.Series(dtype=float)

        holdings, pending_buys_new, pending_sells_new, sell_reasons, _ = decide_eod(
            decision_date=current_date,
            holdings=holdings,
            candidates=signal_by_date.get(current_date, pd.DataFrame()),
            pred_lookup=pred_lookup,
            prev_date=prev_date,
            day_close=day_close,
            day_open_next=day_open_next,
            max_stocks=max_stocks,
            max_hold_days=MAX_HOLD_DAYS,
            stop_loss=STOP_LOSS,
            exit_threshold=BUY_CLS_EXIT_THRESHOLD,
        )

        pending_sells = pending_sells_new
        pending_orders = pending_buys_new

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

    nav_df = pd.DataFrame(nav_records)
    trades_df = pd.DataFrame(trade_details)
    return {
        "nav_df": nav_df, "trades_df": trades_df, "skipped_stats": dict(skipped),
        "params": {
            "max_stocks": max_stocks, "score_col": score_col,
            "max_hold_days": MAX_HOLD_DAYS, "stop_loss": STOP_LOSS,
            "exit_mode": "model_exit",
        },
    }


def main():
    print("=" * 60)
    print("Stacking 元模型回测对比")
    print("=" * 60)

    os.makedirs(BACKTEST_DIR, exist_ok=True)

    pred_path = os.path.join(RESULTS_DIR, "meta_test_predictions.parquet")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"{pred_path} 不存在，请先运行 01_train_meta_model.py")

    df = pd.read_parquet(pred_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    print(f"  数据: {len(df)} 行")

    test_df = df[df["obs_day"] == 1].copy()
    print(f"  obs_day=1: {len(test_df)} 行")

    if test_df.empty:
        raise ValueError("无可回测数据")

    signal_start = str(test_df["obs_date"].min().date())
    signal_end_dt = test_df["obs_date"].max() + pd.Timedelta(days=60)
    signal_end = str(signal_end_dt.date())

    print(f"\n  加载K线数据: {signal_start} ~ {signal_end}")
    daily_prices = load_daily_prices(signal_start, signal_end)
    price_pivot, trading_days, prev_close_map = build_price_pivot(daily_prices)
    print(f"  交易日: {len(trading_days)}")

    pred_lookup = build_pred_lookup(test_df)

    score_configs = [
        ("pred_sell_reg", "baseline_sell_reg"),
    ]
    for meta_col, label in META_PRED_COLS:
        if meta_col in test_df.columns:
            score_configs.append((meta_col, label))

    print(f"\n  回测方案: {len(score_configs)} 组")
    for sc, label in score_configs:
        print(f"    - {label}: score_col={sc}")

    results_rows = []
    for score_col, label in score_configs:
        print(f"\n  --- 回测: {label} (score_col={score_col}) ---")

        run_df = test_df.copy()
        run_df["score"] = run_df[score_col]

        valid = run_df["score"].notna() & (run_df["score"] != 0)
        run_df = run_df[valid].copy()
        if run_df.empty:
            print(f"    无有效数据，跳过")
            continue

        result = run_backtest(
            run_df, price_pivot, trading_days, prev_close_map,
            pred_lookup, max_stocks=10, score_col="score",
        )
        s = compute_backtest_summary(result)

        tdf = result.get("trades_df", pd.DataFrame())
        reasons = {}
        if not tdf.empty and "sell_reason" in tdf.columns:
            reasons = tdf["sell_reason"].value_counts().to_dict()

        row = {
            "label": label,
            "score_col": score_col,
            **{k: v for k, v in s.items() if not isinstance(v, dict)},
            "n_model_risk": reasons.get("model_risk", 0),
            "n_stop_loss": reasons.get("stop_loss", 0),
            "n_max_hold": reasons.get("max_hold", 0),
        }
        results_rows.append(row)

        print(f"    NAV={s.get('final_nav', 0):.4f}, Sharpe={s.get('sharpe', 0):.2f}, "
              f"MDD={s.get('max_dd', 0):.4f}, 胜率={s.get('win_rate', 0):.2%}, "
              f"交易数={s.get('n_trades', 0)}")

    comp_df = pd.DataFrame(results_rows)
    comp_path = os.path.join(BACKTEST_DIR, "backtest_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\n  保存: {comp_path}")

    print(f"\n{'='*80}")
    print("回测对比汇总")
    print(f"{'='*80}")
    print(f"  {'方案':25s} {'NAV':>8s} {'Sharpe':>8s} {'MDD':>8s} {'胜率':>8s} {'交易数':>6s}")
    for row in results_rows:
        nav = row.get('final_nav', np.nan)
        sharpe = row.get('sharpe', np.nan)
        mdd = row.get('max_dd', np.nan)
        wr = row.get('win_rate', np.nan)
        nt = row.get('n_trades', 0)
        print(f"  {row['label']:25s} {nav:>8.4f} {sharpe:>8.2f} {mdd:>8.4f} {wr:>8.2%} {nt:>6d}")


if __name__ == "__main__":
    main()
