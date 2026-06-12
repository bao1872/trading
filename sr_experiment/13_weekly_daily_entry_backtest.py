# -*- coding: utf-8 -*-
"""
Purpose: 周线低吸回测 v2——周线信号为主买点，日线做辅助（风险过滤/加仓确认）
Inputs:  周线分片 parquet (w_pv10) + 日线分片 parquet (d_pv20)
Outputs: 控制台汇总指标 + 按年份统计 + 逐笔交易 CSV
How to Run:
    python sr_experiment/13_weekly_daily_entry_backtest.py --weekly-pv 10 --daily-pv 20 --rule W2
    python sr_experiment/13_weekly_daily_entry_backtest.py --weekly-pv 10 --daily-pv 20 --rule W0
    python sr_experiment/13_weekly_daily_entry_backtest.py --weekly-pv 10 --daily-pv 20 --rule W1
Examples:
    python sr_experiment/13_weekly_daily_entry_backtest.py --weekly-pv 10 --daily-pv 20 --rule W2
    python sr_experiment/13_weekly_daily_entry_backtest.py --weekly-pv 10 --daily-pv 20 --rule W0 --hold-days 40
Side Effects: 写入 results/backtest_v2_trades_*.csv
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sr_experiment.gbdt_config import MODELS_DIR, TRAIN_END, VAL_END
from sr_experiment.gbdt_feature_columns import ALL_FEATURE_COLS

_build_mod = importlib.import_module("sr_experiment.00_build_factor_panel_from_db")
iter_shards = _build_mod.iter_shards

WEEKLY_COLS = [
    "ts_code", "bar_time", "open", "close", "low", "high",
    "evt_pierce_support_cluster_reclaim_low_volume",
    "support_reclaim_strength_atr",
    "fwd_ret_20", "fwd_max_ret_20", "fwd_mdd_20",
]

DAILY_COLS = [
    "ts_code", "bar_time", "open", "high", "low", "close",
    "evt_cross_recent_resistance",
]


def _load_weekly_signals(weekly_pv: int, rule: str) -> pd.DataFrame:
    import lightgbm as lgb

    print(f"加载周线信号 (w_pv{weekly_pv}, 规则={rule})...")
    cols_to_load = list(dict.fromkeys(WEEKLY_COLS + ALL_FEATURE_COLS))
    parts = []
    for shard_df in iter_shards("w", weekly_pv, columns=cols_to_load, shard_type="panel"):
        parts.append(shard_df)
    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    if "bar_time" in df.columns:
        df["bar_time"] = pd.to_datetime(df["bar_time"])

    base_mask = df["evt_pierce_support_cluster_reclaim_low_volume"].fillna(False).astype(bool)
    if isinstance(base_mask, pd.DataFrame):
        base_mask = base_mask.iloc[:, 0]
    signals = df.loc[base_mask].copy()
    print(f"  周线强簇缩量事件: {len(signals)}")

    if rule in ("W1", "W2"):
        if "bar_time" in signals.columns:
            train_mask = signals["bar_time"] <= TRAIN_END
            if rule == "W1":
                reclaim_col = "support_reclaim_strength_atr"
                if reclaim_col in signals.columns:
                    threshold = signals.loc[train_mask, reclaim_col].median()
                    signals = signals[signals[reclaim_col] > threshold].copy()
                    print(f"  W1 过滤后: {len(signals)} (reclaim_threshold={threshold:.4f})")
            elif rule == "W2":
                model_path = Path(MODELS_DIR) / "B_cluster_risk.txt"
                if model_path.exists():
                    model = lgb.Booster(model_file=str(model_path))
                    feature_cols = [c for c in ALL_FEATURE_COLS if c in signals.columns]
                    signals["risk_score"] = model.predict(signals[feature_cols])
                    threshold = signals.loc[train_mask, "risk_score"].quantile(0.70)
                    signals = signals[signals["risk_score"] < threshold].copy()
                    print(f"  W2 过滤后: {len(signals)} (risk_threshold={threshold:.4f})")

    return signals


def _load_daily_panel(daily_pv: int) -> pd.DataFrame:
    print(f"加载日线面板 (d_pv{daily_pv})...")
    parts = []
    for shard_df in iter_shards("d", daily_pv, columns=DAILY_COLS, shard_type="panel"):
        parts.append(shard_df)
    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    if "bar_time" in df.columns:
        df["bar_time"] = pd.to_datetime(df["bar_time"])
    if "evt_cross_recent_resistance" in df.columns:
        df["evt_cross_recent_resistance"] = df["evt_cross_recent_resistance"].fillna(False).astype(bool)
    print(f"  日线总行数: {len(df)}")
    return df


def _find_weekly_entries(
    weekly_signals: pd.DataFrame,
    daily_df: pd.DataFrame,
    add_window: int = 10,
) -> pd.DataFrame:
    print("匹配周线入场点...")
    daily_sorted = daily_df.sort_values(["ts_code", "bar_time"]).copy()
    daily_by_code = {code: group for code, group in daily_sorted.groupby("ts_code")}

    results = []
    n_matched = 0

    for _, w_row in weekly_signals.iterrows():
        ts_code = w_row["ts_code"]
        w_time = w_row["bar_time"]
        w_low = w_row.get("low", np.nan)

        d_group = daily_by_code.get(ts_code)
        if d_group is None:
            continue

        d_bartimes = d_group["bar_time"].values
        idx_start = np.searchsorted(d_bartimes, w_time, side="right")
        if idx_start >= len(d_bartimes):
            continue

        entry_bar = d_group.iloc[idx_start]
        entry_price = entry_bar.get("open", np.nan)
        entry_date = entry_bar["bar_time"]

        if pd.isna(entry_price):
            continue

        if pd.notna(entry_bar.get("open")) and pd.notna(entry_bar.get("high")) and pd.notna(entry_bar.get("low")):
            if entry_bar["open"] == entry_bar["high"] == entry_bar["low"]:
                continue

        add_idx = None
        add_price = np.nan
        add_date = pd.NaT
        broken_before_add = False

        idx_end = min(idx_start + add_window, len(d_bartimes))
        for i in range(idx_start, idx_end):
            d_row = d_group.iloc[i]
            if pd.notna(w_low) and pd.notna(d_row.get("low")) and d_row["low"] < w_low:
                broken_before_add = True
                break
            if d_row.get("evt_cross_recent_resistance", False) and add_idx is None:
                add_idx = i
                if i + 1 < len(d_group):
                    next_bar = d_group.iloc[i + 1]
                    add_price = next_bar.get("open", np.nan)
                    add_date = next_bar["bar_time"]
                    if pd.notna(add_price) and pd.notna(next_bar.get("high")) and pd.notna(next_bar.get("low")):
                        if next_bar["open"] == next_bar["high"] == next_bar["low"]:
                            add_price = np.nan
                            add_date = pd.NaT
                            add_idx = None

        n_matched += 1
        results.append({
            "ts_code": ts_code,
            "weekly_bar_time": w_time,
            "weekly_close": w_row.get("close", np.nan),
            "weekly_low": w_low,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "stop_price": w_low,
            "has_daily_breakout": add_idx is not None and not broken_before_add,
            "add_date": add_date,
            "add_price": add_price,
            "broken_before_add": broken_before_add,
        })

    print(f"  匹配: {n_matched} 个周线信号找到日线数据")
    print(f"  有效入场点: {len(results)}")
    df = pd.DataFrame(results)
    if not df.empty:
        n_add = df["has_daily_breakout"].sum()
        n_broken = df["broken_before_add"].sum()
        print(f"  有日线突破加仓机会: {n_add}")
        print(f"  加仓前已跌破周线low: {n_broken}")
    return df


def _simulate_plan_a(
    entries: pd.DataFrame,
    daily_df: pd.DataFrame,
    hold_days: int,
) -> pd.DataFrame:
    print(f"模拟方案A: 纯周线低吸+固定持有{hold_days}日...")
    daily_sorted = daily_df.sort_values(["ts_code", "bar_time"]).copy()
    daily_by_code = {code: group for code, group in daily_sorted.groupby("ts_code")}

    trades = []
    for _, entry in entries.iterrows():
        ts_code = entry["ts_code"]
        entry_price = entry["entry_price"]
        entry_date = entry["entry_date"]
        stop_price = entry["stop_price"]

        if pd.isna(entry_price) or pd.isna(entry_date):
            continue

        d_group = daily_by_code.get(ts_code)
        if d_group is None:
            continue

        d_bartimes = d_group["bar_time"].values
        entry_idx = np.searchsorted(d_bartimes, entry_date, side="left")
        if entry_idx >= len(d_bartimes):
            continue

        exit_price = np.nan
        exit_date = pd.NaT
        is_stopped = False
        actual_hold = 0

        for day_offset in range(hold_days):
            bar_idx = entry_idx + day_offset
            if bar_idx >= len(d_group):
                break

            bar = d_group.iloc[bar_idx]
            actual_hold = day_offset + 1

            if day_offset == hold_days - 1 or bar_idx == len(d_group) - 1:
                exit_price = bar.get("close", np.nan)
                exit_date = bar["bar_time"]
                break

        if pd.isna(exit_price) or pd.isna(exit_date):
            continue

        pnl_pct = (exit_price - entry_price) / entry_price

        trades.append({
            "ts_code": ts_code,
            "weekly_bar_time": entry["weekly_bar_time"],
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "hold_days": actual_hold,
            "pnl_pct": pnl_pct,
            "is_stopped": is_stopped,
            "is_winner": pnl_pct > 0,
            "plan": "A",
            "hold_days_target": hold_days,
        })

    print(f"  有效交易: {len(trades)}")
    return pd.DataFrame(trades)


def _simulate_plan_b(
    entries: pd.DataFrame,
    daily_df: pd.DataFrame,
    hold_days: int,
) -> pd.DataFrame:
    print(f"模拟方案B: 周线低吸+日线风险过滤+持有{hold_days}日...")
    daily_sorted = daily_df.sort_values(["ts_code", "bar_time"]).copy()
    daily_by_code = {code: group for code, group in daily_sorted.groupby("ts_code")}

    trades = []
    for _, entry in entries.iterrows():
        ts_code = entry["ts_code"]
        entry_price = entry["entry_price"]
        entry_date = entry["entry_date"]
        stop_price = entry["stop_price"]

        if pd.isna(entry_price) or pd.isna(entry_date):
            continue

        d_group = daily_by_code.get(ts_code)
        if d_group is None:
            continue

        d_bartimes = d_group["bar_time"].values
        entry_idx = np.searchsorted(d_bartimes, entry_date, side="left")
        if entry_idx >= len(d_bartimes):
            continue

        exit_price = np.nan
        exit_date = pd.NaT
        is_stopped = False
        actual_hold = 0

        for day_offset in range(hold_days):
            bar_idx = entry_idx + day_offset
            if bar_idx >= len(d_group):
                break

            bar = d_group.iloc[bar_idx]
            actual_hold = day_offset + 1

            if pd.notna(stop_price) and pd.notna(bar.get("low")):
                if bar["low"] <= stop_price:
                    exit_price = stop_price
                    exit_date = bar["bar_time"]
                    is_stopped = True
                    break

            if day_offset == hold_days - 1 or bar_idx == len(d_group) - 1:
                exit_price = bar.get("close", np.nan)
                exit_date = bar["bar_time"]
                break

        if pd.isna(exit_price) or pd.isna(exit_date):
            continue

        pnl_pct = (exit_price - entry_price) / entry_price

        trades.append({
            "ts_code": ts_code,
            "weekly_bar_time": entry["weekly_bar_time"],
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "hold_days": actual_hold,
            "pnl_pct": pnl_pct,
            "is_stopped": is_stopped,
            "is_winner": pnl_pct > 0,
            "plan": "B",
            "hold_days_target": hold_days,
        })

    print(f"  有效交易: {len(trades)}")
    return pd.DataFrame(trades)


def _simulate_plan_c(
    entries: pd.DataFrame,
    daily_df: pd.DataFrame,
    hold_days: int,
) -> pd.DataFrame:
    print(f"模拟方案C: 周线先买50%+日线突破加仓50%+持有{hold_days}日...")
    daily_sorted = daily_df.sort_values(["ts_code", "bar_time"]).copy()
    daily_by_code = {code: group for code, group in daily_sorted.groupby("ts_code")}

    trades = []
    for _, entry in entries.iterrows():
        ts_code = entry["ts_code"]
        first_entry_price = entry["entry_price"]
        first_entry_date = entry["entry_date"]
        stop_price = entry["stop_price"]

        if pd.isna(first_entry_price) or pd.isna(first_entry_date):
            continue

        d_group = daily_by_code.get(ts_code)
        if d_group is None:
            continue

        d_bartimes = d_group["bar_time"].values
        first_entry_idx = np.searchsorted(d_bartimes, first_entry_date, side="left")
        if first_entry_idx >= len(d_bartimes):
            continue

        first_exit_price = np.nan
        first_exit_date = pd.NaT
        first_is_stopped = False
        first_actual_hold = 0

        second_entry_price = np.nan
        second_entry_date = pd.NaT
        second_entry_idx = -1
        second_exit_price = np.nan
        second_exit_date = pd.NaT
        second_is_stopped = False
        second_actual_hold = 0
        has_second = False

        if entry["has_daily_breakout"] and pd.notna(entry.get("add_price")) and pd.notna(entry.get("add_date")):
            second_entry_price = entry["add_price"]
            second_entry_date = entry["add_date"]
            second_entry_idx = np.searchsorted(d_bartimes, second_entry_date, side="left")
            has_second = second_entry_idx < len(d_bartimes)

        for day_offset in range(hold_days):
            bar_idx = first_entry_idx + day_offset
            if bar_idx >= len(d_group):
                break

            bar = d_group.iloc[bar_idx]

            if pd.notna(stop_price) and pd.notna(bar.get("low")):
                if bar["low"] <= stop_price:
                    if not first_is_stopped:
                        first_exit_price = stop_price
                        first_exit_date = bar["bar_time"]
                        first_is_stopped = True
                        first_actual_hold = day_offset + 1
                    if has_second and bar_idx >= second_entry_idx and not second_is_stopped:
                        second_exit_price = stop_price
                        second_exit_date = bar["bar_time"]
                        second_is_stopped = True
                        second_actual_hold = bar_idx - second_entry_idx + 1
                    break

            if not first_is_stopped:
                if day_offset == hold_days - 1 or bar_idx == len(d_group) - 1:
                    first_exit_price = bar.get("close", np.nan)
                    first_exit_date = bar["bar_time"]
                    first_actual_hold = day_offset + 1
                    first_is_stopped = False

            if has_second and bar_idx >= second_entry_idx and not second_is_stopped:
                remaining = hold_days - (second_entry_idx - first_entry_idx)
                if day_offset - (second_entry_idx - first_entry_idx) >= remaining - 1 or bar_idx == len(d_group) - 1:
                    second_exit_price = bar.get("close", np.nan)
                    second_exit_date = bar["bar_time"]
                    second_actual_hold = bar_idx - second_entry_idx + 1
                    second_is_stopped = False

        if pd.isna(first_exit_price) or pd.isna(first_exit_date):
            continue

        first_pnl = (first_exit_price - first_entry_price) / first_entry_price
        combined_pnl = first_pnl * 0.5

        if has_second and pd.notna(second_exit_price) and pd.notna(second_exit_date):
            second_pnl = (second_exit_price - second_entry_price) / second_entry_price
            combined_pnl += second_pnl * 0.5
        else:
            combined_pnl = first_pnl

        trades.append({
            "ts_code": ts_code,
            "weekly_bar_time": entry["weekly_bar_time"],
            "entry_date": first_entry_date,
            "entry_price": first_entry_price,
            "exit_date": first_exit_date,
            "exit_price": first_exit_price,
            "hold_days": first_actual_hold,
            "pnl_pct": combined_pnl,
            "first_pnl": first_pnl,
            "second_pnl": (second_exit_price - second_entry_price) / second_entry_price if has_second and pd.notna(second_exit_price) and pd.notna(second_entry_price) else np.nan,
            "has_second": has_second,
            "is_stopped": first_is_stopped or second_is_stopped,
            "is_winner": combined_pnl > 0,
            "plan": "C",
            "hold_days_target": hold_days,
        })

    print(f"  有效交易: {len(trades)}")
    return pd.DataFrame(trades)


def _compute_metrics(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"version": label, "count": 0}

    pnls = trades["pnl_pct"].dropna()
    if pnls.empty:
        return {"version": label, "count": 0}

    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]

    years_span = 1
    if "entry_date" in trades.columns:
        dates = pd.to_datetime(trades["entry_date"].dropna())
        if len(dates) > 1:
            years_span = max((dates.max() - dates.min()).days / 365.25, 1)

    avg_winner = winners.mean() if len(winners) > 0 else 0
    avg_loser = abs(losers.mean()) if len(losers) > 0 else 1e-9

    top10_sum = pnls.nlargest(min(10, len(pnls))).sum() if len(pnls) > 0 else 0
    total_sum = pnls.sum() if len(pnls) > 0 else 1e-9

    return {
        "version": label,
        "count": len(pnls),
        "avg_per_year": round(len(pnls) / years_span, 1),
        "mean_ret": pnls.mean(),
        "median_ret": pnls.median(),
        "win_rate": (pnls > 0).mean(),
        "profit_loss_ratio": avg_winner / avg_loser if avg_loser > 0 else np.nan,
        "max_loss": pnls.min(),
        "max_gain": pnls.max(),
        "stop_rate": trades["is_stopped"].mean() if "is_stopped" in trades.columns else np.nan,
        "top10_concentration": top10_sum / total_sum if total_sum != 0 else np.nan,
    }


def _report_by_year(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    if trades.empty or "entry_date" not in trades.columns:
        return pd.DataFrame()

    df = trades.copy()
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year

    rows = []
    for year in sorted(df["year"].dropna().unique()):
        ydf = df[df["year"] == year]
        pnls = ydf["pnl_pct"].dropna()
        if pnls.empty:
            continue
        rows.append({
            "version": label,
            "year": int(year),
            "count": len(pnls),
            "mean_ret": pnls.mean(),
            "win_rate": (pnls > 0).mean(),
            "stop_rate": ydf["is_stopped"].mean() if "is_stopped" in ydf.columns else np.nan,
        })

    return pd.DataFrame(rows)


def _report_by_stock(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {}
    stock_pnl = trades.groupby("ts_code")["pnl_pct"].sum().sort_values(ascending=False)
    top5 = stock_pnl.head(5)
    top5_sum = top5.sum()
    total_sum = stock_pnl.sum()
    return {
        "top5_stocks": dict(top5),
        "top5_concentration": top5_sum / total_sum if total_sum != 0 else np.nan,
        "n_positive_stocks": (stock_pnl > 0).sum(),
        "n_negative_stocks": (stock_pnl <= 0).sum(),
        "total_stocks": len(stock_pnl),
    }


def main():
    parser = argparse.ArgumentParser(description="周线低吸回测 v2")
    parser.add_argument("--weekly-pv", type=int, default=10)
    parser.add_argument("--daily-pv", type=int, default=20)
    parser.add_argument("--rule", type=str, default="W2", choices=["W0", "W1", "W2"])
    parser.add_argument("--hold-days", type=int, nargs="+", default=[20, 40, 60])
    args = parser.parse_args()

    weekly_signals = _load_weekly_signals(args.weekly_pv, args.rule)
    if weekly_signals.empty:
        print("无周线信号")
        return

    daily_df = _load_daily_panel(args.daily_pv)
    if daily_df.empty:
        print("无日线数据")
        return

    entries = _find_weekly_entries(weekly_signals, daily_df)
    if entries.empty:
        print("无有效入场点")
        return

    all_metrics = []
    all_year_rows = []

    for hd in args.hold_days:
        trades_a = _simulate_plan_a(entries, daily_df, hd)
        trades_b = _simulate_plan_b(entries, daily_df, hd)
        trades_c = _simulate_plan_c(entries, daily_df, hd)

        label_a = f"A:纯周线_H{hd}"
        label_b = f"B:周线+日线止损_H{hd}"
        label_c = f"C:50%+50%加仓_H{hd}"

        all_metrics.append(_compute_metrics(trades_a, label_a))
        all_metrics.append(_compute_metrics(trades_b, label_b))
        all_metrics.append(_compute_metrics(trades_c, label_c))

        all_year_rows.append(_report_by_year(trades_a, label_a))
        all_year_rows.append(_report_by_year(trades_b, label_b))
        all_year_rows.append(_report_by_year(trades_c, label_c))

        out_dir = Path("sr_experiment/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        for trades, ver in [(trades_a, f"A_H{hd}"), (trades_b, f"B_H{hd}"), (trades_c, f"C_H{hd}")]:
            if not trades.empty:
                path = out_dir / f"backtest_v2_trades_{args.rule}_{ver}.csv"
                trades.to_csv(path, index=False)

    print(f"\n{'='*120}")
    print(f"回测结果 v2: 规则={args.rule}, 周线pv={args.weekly_pv}, 日线pv={args.daily_pv}")
    print(f"{'='*120}")

    metrics_df = pd.DataFrame(all_metrics)
    display_cols = [c for c in ["version", "count", "avg_per_year", "mean_ret", "median_ret",
                                 "win_rate", "profit_loss_ratio", "max_loss", "max_gain",
                                 "stop_rate", "top10_concentration"]
                    if c in metrics_df.columns]
    print("\n--- 汇总指标 ---")
    print(tabulate(metrics_df[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    year_all = pd.concat(all_year_rows, ignore_index=True)
    if not year_all.empty:
        print("\n--- 按年份统计 ---")
        print(tabulate(year_all, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    print("\n--- 收益集中度（方案A_H20）---")
    trades_a20 = _simulate_plan_a(entries, daily_df, 20)
    stock_info = _report_by_stock(trades_a20, "A")
    if stock_info:
        print(f"  参与股票数: {stock_info.get('total_stocks', 0)}")
        print(f"  盈利股票数: {stock_info.get('n_positive_stocks', 0)}")
        print(f"  亏损股票数: {stock_info.get('n_negative_stocks', 0)}")
        if stock_info.get("top5_stocks"):
            for code, pnl in stock_info["top5_stocks"].items():
                print(f"    {code}: {pnl:.4f}")


if __name__ == "__main__":
    main()
