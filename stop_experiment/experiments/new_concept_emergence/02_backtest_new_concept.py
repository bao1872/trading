#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新概念涌现因子回测对比实验

Purpose:
    在端到端回测中验证"新概念涌现"信号对策略表现的实际提升。
    对照组 A: 当前基线（sell_score 选股）
    实验组 B1: 新概念股票 score × 1.2
    实验组 B2: 新概念股票 score × 1.5
    实验组 C: 只买新概念股票（若无可买则空仓）
    实验组 D: 新概念优先（先选新概念，不足再补非新概念）

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet
    - stop_experiment/output/models_control/ (final models)
    - DB: stock_k_data, stock_pools

Outputs:
    - results/backtest/backtest_comparison.csv
    - results/backtest/new_concept_trades_detail.csv

How to Run:
    python -m stop_experiment.experiments.new_concept_emergence.02_backtest_new_concept

Examples:
    python -m stop_experiment.experiments.new_concept_emergence.02_backtest_new_concept
    python -m stop_experiment.experiments.new_concept_emergence.02_backtest_new_concept --emergence-def A_absent_10

Side Effects:
    - 只读 DB 和 parquet
    - 输出仅写入 results/backtest/

Limitations:
    - stock_pools.concepts 是当前快照，非历史数据，存在前视偏差风险
"""

from __future__ import annotations

import sys
import os
import argparse
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OBS_VAL_END, MODELS_DIR, BUY_COST, SELL_COST,
)
from stop_experiment.backtest.simple_backtest import (
    load_daily_prices, build_price_pivot,
    is_limit_up, is_limit_down, is_suspended,
)
from stop_experiment.backtest.decision_core import decide_eod

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
BACKTEST_DIR = os.path.join(RESULTS_DIR, "backtest")

MAX_HOLD_DAYS = 20
STOP_LOSS = -0.07
EXIT_THRESHOLD = 0.70
MAX_STOCKS = 10

EMERGENCE_DEFS = {
    "A_absent_5": {"method": "absent", "lookback": 5},
    "A_absent_10": {"method": "absent", "lookback": 10},
    "A_absent_20": {"method": "absent", "lookback": 20},
    "C_count_2": {"method": "count_threshold", "lookback": 5, "min_count": 2},
    "C_count_3": {"method": "count_threshold", "lookback": 5, "min_count": 3},
}


def load_concept_map_safe() -> pd.DataFrame:
    from market_structure_analysis.theme_aggregator import load_concept_map
    return load_concept_map()


def build_daily_concept_stats(df_entry: pd.DataFrame, concept_map: pd.DataFrame) -> pd.DataFrame:
    df_with_concept = df_entry.merge(concept_map, on="ts_code", how="inner")
    daily_concept = (
        df_with_concept
        .groupby(["obs_date", "concept"])
        .agg(stock_count=("ts_code", "nunique"))
        .reset_index()
    )
    return daily_concept


def identify_emergent_concepts(
    daily_concept: pd.DataFrame,
    method: str,
    lookback: int = 10,
    surge_k: float = 2.0,
    min_count: int = 2,
) -> pd.DataFrame:
    all_dates = sorted(daily_concept["obs_date"].unique())
    if len(all_dates) < lookback + 1:
        return pd.DataFrame()

    concept_daily_pivot = daily_concept.pivot_table(
        index="obs_date", columns="concept", values="stock_count", fill_value=0,
    )
    concept_daily_pivot = concept_daily_pivot.reindex(all_dates, fill_value=0)

    lb_stock_mean = concept_daily_pivot.rolling(window=lookback, min_periods=1).mean()
    lb_stock_mean = lb_stock_mean.shift(1)
    lb_days_present = (concept_daily_pivot > 0).rolling(window=lookback, min_periods=1).sum()
    lb_days_present = lb_days_present.shift(1)

    results = []
    for current_date in all_dates:
        if current_date not in concept_daily_pivot.index:
            continue
        current_counts = concept_daily_pivot.loc[current_date]
        current_counts = current_counts[current_counts > 0]
        if current_counts.empty:
            continue

        lb_mean = lb_stock_mean.loc[current_date] if current_date in lb_stock_mean.index else pd.Series(0, index=current_counts.index)
        lb_days = lb_days_present.loc[current_date] if current_date in lb_days_present.index else pd.Series(0, index=current_counts.index)

        common_concepts = current_counts.index.intersection(lb_mean.index)
        if len(common_concepts) == 0:
            continue

        curr = current_counts.loc[common_concepts]
        mean_lb = lb_mean.loc[common_concepts].fillna(0)
        days_lb = lb_days.loc[common_concepts].fillna(0)

        if method == "absent":
            mask = days_lb == 0
        elif method == "surge":
            mask = (mean_lb > 0) & (curr >= mean_lb * surge_k)
        elif method == "count_threshold":
            mask = (curr >= min_count) & (mean_lb < 1)
        else:
            continue

        emergent_concepts_list = common_concepts[mask]
        if len(emergent_concepts_list) == 0:
            continue

        emergent_df = pd.DataFrame({
            "obs_date": current_date,
            "concept": emergent_concepts_list,
        })
        results.append(emergent_df)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def tag_new_concept_stocks(
    df_entry: pd.DataFrame,
    concept_map: pd.DataFrame,
    emergent_concepts: pd.DataFrame,
) -> pd.DataFrame:
    if emergent_concepts.empty:
        df_entry = df_entry.copy()
        df_entry["is_new_concept"] = False
        return df_entry

    emergent_flag = emergent_concepts[["obs_date", "concept"]].drop_duplicates()
    emergent_flag["is_emergent"] = True

    df_with_concept = df_entry[["signal_id", "obs_date", "ts_code"]].merge(
        concept_map, on="ts_code", how="left"
    )
    df_with_concept = df_with_concept.merge(
        emergent_flag, on=["obs_date", "concept"], how="left"
    )
    df_with_concept["is_emergent"] = df_with_concept["is_emergent"].fillna(False)

    stock_level = (
        df_with_concept
        .groupby(["signal_id", "obs_date", "ts_code"])
        .agg(is_new_concept=("is_emergent", "any"))
        .reset_index()
    )

    result = df_entry.merge(
        stock_level[["signal_id", "obs_date", "ts_code", "is_new_concept"]],
        on=["signal_id", "obs_date", "ts_code"],
        how="left",
    )
    result["is_new_concept"] = result["is_new_concept"].fillna(False)
    return result


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


def compute_summary(result: dict) -> dict:
    nav_df = result["nav_df"].copy()
    trades_df = result["trades_df"]
    if nav_df.empty:
        return {"n_trades": 0, "final_nav": 1.0, "annual_ret": 0, "sharpe": 0,
                "max_dd": 0, "win_rate": 0, "avg_net_ret": 0, "avg_hold_days": 0,
                "profit_loss_ratio": 0}
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
    profit_loss_ratio = 0
    if (trades_df["net_ret"] > 0).any() and (trades_df["net_ret"] < 0).any():
        profit_loss_ratio = (trades_df[trades_df["net_ret"] > 0]["net_ret"].mean() /
                             abs(trades_df[trades_df["net_ret"] < 0]["net_ret"].mean()))
    return {"n_trades": n_trades, "final_nav": final_nav, "annual_ret": annual_ret,
            "max_dd": max_dd, "sharpe": sharpe, "win_rate": win_rate,
            "avg_net_ret": avg_net_ret, "avg_hold_days": avg_hold,
            "profit_loss_ratio": profit_loss_ratio}


def apply_score_mode(candidates: pd.DataFrame, mode: str, boost_factor: float = 1.0) -> pd.DataFrame:
    candidates = candidates.copy()
    if "score" not in candidates.columns:
        candidates["score"] = candidates["pred_sell_reg"]

    if mode == "sell_score":
        candidates["score"] = candidates["pred_sell_reg"]
    elif mode == "sell_score_boost":
        candidates["score"] = candidates["pred_sell_reg"]
        mask = candidates["is_new_concept"]
        candidates.loc[mask, "score"] = candidates.loc[mask, "score"] * boost_factor
    elif mode == "new_concept_only":
        candidates = candidates[candidates["is_new_concept"]]
        candidates["score"] = candidates["pred_sell_reg"]
    elif mode == "new_concept_prefer":
        new_c = candidates[candidates["is_new_concept"]].copy()
        new_c["score"] = new_c["pred_sell_reg"] + 1.0
        other_c = candidates[~candidates["is_new_concept"]].copy()
        other_c["score"] = other_c["pred_sell_reg"]
        remaining = MAX_STOCKS - len(new_c)
        if remaining > 0:
            candidates = pd.concat([new_c, other_c.head(remaining)], ignore_index=True)
        else:
            candidates = new_c.head(MAX_STOCKS)
    return candidates


def run_backtest(signals_df, price_pivot, trading_days, prev_close_map, pred_lookup,
                 score_mode="sell_score", boost_factor=1.0):
    signals_sorted = signals_df.sort_values(["obs_date", "pred_sell_reg"], ascending=[True, False])
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
    empty_pool_days = 0

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
                net_ret = gross_ret - BUY_COST - SELL_COST
                trade_details.append({
                    "ts_code": h["ts_code"], "buy_date": h["buy_date"],
                    "sell_date": current_date, "buy_price": h["buy_price"],
                    "sell_price": sell_price, "hold_days": h["days_held"],
                    "gross_ret": gross_ret, "net_ret": net_ret,
                    "sell_reason": sell_item["reason"],
                    "score": h.get("score", 0),
                    "is_new_concept": h.get("is_new_concept", False),
                })
                if code in holdings:
                    del holdings[code]
            pending_sells = []

        _buy_max = MAX_STOCKS - len(holdings)
        if _buy_max < 0:
            _buy_max = 0
        if pending_orders:
            executed = []
            for code, bp, ts_code, sc, sid, is_nc in pending_orders:
                if len(executed) >= _buy_max:
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
                executed.append((code, bp, ts_code, sc, sid, is_nc))
            if executed:
                n = len(holdings) + len(executed)
                w = 1.0 / n
                for code_h in holdings:
                    holdings[code_h]["weight"] = w
                for code, bp, ts_code, sc, sid, is_nc in executed:
                    holdings[code] = {
                        "buy_date": current_date, "buy_price": bp,
                        "weight": w, "days_held": 0,
                        "ts_code": ts_code, "score": sc, "signal_id": sid,
                        "is_new_concept": is_nc,
                    }
            pending_orders = []

        prev_date = trading_days[t_idx - 1] if t_idx > 0 else None
        next_idx = t_idx + 1
        day_open_next = (price_pivot.loc[trading_days[next_idx], "open"]
                         if next_idx < len(trading_days) else pd.Series(dtype=float))

        candidates = signal_by_date.get(current_date, pd.DataFrame())
        if candidates.empty:
            empty_pool_days += 1
        else:
            candidates = apply_score_mode(candidates, score_mode, boost_factor)

        holdings, pending_buys_new, pending_sells_new, sell_reasons, _ = decide_eod(
            decision_date=current_date,
            holdings=holdings,
            candidates=candidates,
            pred_lookup=pred_lookup,
            prev_date=prev_date,
            day_close=day_close,
            day_open_next=day_open_next,
            max_stocks=MAX_STOCKS,
            max_hold_days=MAX_HOLD_DAYS,
            stop_loss=STOP_LOSS,
            exit_threshold=EXIT_THRESHOLD,
        )

        pending_sells = pending_sells_new
        pending_orders = []
        for item in pending_buys_new:
            code = item[0]
            bp = item[1]
            ts_code = item[2] if len(item) > 2 else code
            sc = item[3] if len(item) > 3 else 0
            sid = item[4] if len(item) > 4 else 0
            is_nc = False
            if not candidates.empty and ts_code in candidates["ts_code"].values:
                nc_vals = candidates.loc[candidates["ts_code"] == ts_code, "is_new_concept"]
                if len(nc_vals) > 0:
                    is_nc = bool(nc_vals.iloc[0])
            pending_orders.append((code, bp, ts_code, sc, sid, is_nc))

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
        nav_records.append({"date": current_date, "nav": nav, "daily_ret": daily_ret,
                            "n_positions": len(holdings)})

    nav_df = pd.DataFrame(nav_records)
    trades_df = pd.DataFrame(trade_details)
    return {"nav_df": nav_df, "trades_df": trades_df,
            "skipped_stats": dict(skipped), "empty_pool_days": empty_pool_days}


def main():
    parser = argparse.ArgumentParser(description="新概念涌现因子回测对比实验")
    parser.add_argument("--emergence-def", type=str, default="A_absent_10",
                        help="涌现定义名称 (默认 A_absent_10)")
    args = parser.parse_args()

    print("=" * 60)
    print("新概念涌现因子回测对比实验")
    print("=" * 60)
    print(f"  涌现定义: {args.emergence_def}")

    os.makedirs(BACKTEST_DIR, exist_ok=True)

    print("\n[1/4] 加载数据 + 概念映射...")
    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在")

    df = pd.read_parquet(scores_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.dropna(subset=["mfe_20", "mae_20"])

    concept_map = load_concept_map_safe()
    print(f"  总行数: {len(df)}, 概念映射: {concept_map['concept'].nunique()} 个概念")

    val_end = pd.Timestamp(OBS_VAL_END)
    test = df[df["obs_date"] > val_end].copy()
    test_entry = test[test["obs_day"].isin([1, 2, 3])].copy()
    print(f"  test 入场样本: {len(test_entry)}")

    print("\n[2/4] 识别新概念涌现...")
    if args.emergence_def not in EMERGENCE_DEFS:
        raise ValueError(f"未知涌现定义: {args.emergence_def}, 可选: {list(EMERGENCE_DEFS.keys())}")

    def_params = EMERGENCE_DEFS[args.emergence_def]
    daily_concept = build_daily_concept_stats(test_entry, concept_map)
    emergent = identify_emergent_concepts(daily_concept, **def_params)
    print(f"  涌现事件: {len(emergent)}")

    tagged = tag_new_concept_stocks(test_entry, concept_map, emergent)
    n_new = tagged["is_new_concept"].sum()
    print(f"  新概念股票样本: {n_new} ({n_new/len(tagged):.1%})")

    print("\n[3/4] 加载K线数据...")
    signal_start = str(test_entry["obs_date"].min().date())
    signal_end_dt = test_entry["obs_date"].max() + pd.Timedelta(days=60)
    signal_end = str(signal_end_dt.date())
    print(f"  K线范围: {signal_start} ~ {signal_end}")

    daily_prices = load_daily_prices(signal_start, signal_end)
    price_pivot, trading_days, prev_close_map = build_price_pivot(daily_prices)
    print(f"  交易日: {len(trading_days)}")

    print("\n[4/4] 回测对比...")

    EXPERIMENT_CONFIGS = [
        {"label": "A_baseline", "score_mode": "sell_score", "boost_factor": 1.0},
        {"label": "B1_boost_1.2", "score_mode": "sell_score_boost", "boost_factor": 1.2},
        {"label": "B2_boost_1.5", "score_mode": "sell_score_boost", "boost_factor": 1.5},
        {"label": "C_filter_only", "score_mode": "new_concept_only", "boost_factor": 1.0},
        {"label": "D_prefer", "score_mode": "new_concept_prefer", "boost_factor": 1.0},
    ]

    results_rows = []
    all_trades = []

    for config in EXPERIMENT_CONFIGS:
        label = config["label"]
        score_mode = config["score_mode"]
        boost_factor = config["boost_factor"]

        print(f"\n  --- {label} (mode={score_mode}, boost={boost_factor}) ---")

        pred_lookup = build_pred_lookup(tagged)

        result = run_backtest(
            tagged, price_pivot, trading_days, prev_close_map, pred_lookup,
            score_mode=score_mode, boost_factor=boost_factor,
        )
        s = compute_summary(result)

        row = {"label": label, "score_mode": score_mode, "boost_factor": boost_factor,
               "emergence_def": args.emergence_def, **s,
               "empty_pool_days": result.get("empty_pool_days", 0)}
        results_rows.append(row)

        print(f"    NAV={s['final_nav']:.4f}, Sharpe={s['sharpe']:.2f}, "
              f"MDD={s['max_dd']:.4f}, 胜率={s['win_rate']:.2%}, "
              f"盈亏比={s['profit_loss_ratio']:.2f}, 交易数={s['n_trades']}")

        if not result["trades_df"].empty:
            trades_copy = result["trades_df"].copy()
            trades_copy["experiment"] = label
            all_trades.append(trades_copy)

    comp_df = pd.DataFrame(results_rows)
    comp_path = os.path.join(BACKTEST_DIR, "backtest_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\n  保存: {comp_path}")

    if all_trades:
        all_trades_df = pd.concat(all_trades, ignore_index=True)
        trades_path = os.path.join(BACKTEST_DIR, "new_concept_trades_detail.csv")
        all_trades_df.to_csv(trades_path, index=False)
        print(f"  保存: {trades_path}")

    print(f"\n{'='*80}")
    print("回测对比汇总")
    print(f"{'='*80}")
    print(f"  涌现定义: {args.emergence_def}")
    print(f"  {'方案':20s} {'NAV':>8s} {'Sharpe':>8s} {'MDD':>8s} {'胜率':>8s} {'盈亏比':>8s} {'交易':>5s}")
    for row in results_rows:
        print(f"  {row['label']:20s} {row['final_nav']:>8.4f} {row['sharpe']:>8.2f} "
              f"{row['max_dd']:>8.4f} {row['win_rate']:>8.2%} {row['profit_loss_ratio']:>8.2f} "
              f"{row['n_trades']:>5d}")

    if len(results_rows) >= 2:
        baseline_nav = results_rows[0]["final_nav"]
        for row in results_rows[1:]:
            delta = row["final_nav"] - baseline_nav
            delta_pct = delta / baseline_nav * 100 if baseline_nav > 0 else 0
            print(f"\n  {row['label']} vs baseline: ΔNAV = {delta:+.4f} ({delta_pct:+.1f}%)")

    print(f"\n  ⚠ 前视偏差风险: 概念映射使用当前快照，非历史数据")


if __name__ == "__main__":
    main()
