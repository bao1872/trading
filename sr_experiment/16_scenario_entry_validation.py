# -*- coding: utf-8 -*-
"""
Purpose: 场景化入场验证——比较三种交易场景(R2S刺破收回/动量回踩/低吸止跌)的不同入场方式
Inputs:  周线 events 分片 (w_pv10) + 日线 panel 分片 (d_pv20) + GBDT风险模型
Outputs: 控制台对比表 + scenario_entry_validation.csv
How to Run:
    python sr_experiment/16_scenario_entry_validation.py --rule W2
    python sr_experiment/16_scenario_entry_validation.py --rule W0
Examples:
    python sr_experiment/16_scenario_entry_validation.py --rule W2
    python sr_experiment/16_scenario_entry_validation.py --rule W0
Side Effects: 写入 sr_experiment/results/scenario_entry_validation.csv
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
from sr_experiment.manual_filter_scoring import (
    TIER_ORDER,
    assign_tier,
)

_build_mod = importlib.import_module("sr_experiment.00_build_factor_panel_from_db")
iter_shards = _build_mod.iter_shards

EVENTS_COLS = [
    "ts_code", "bar_time", "open", "close", "low", "high",
    "evt_retest_flipped_support", "evt_clean_hold_flipped_support",
    "evt_pierce_flipped_support_reclaim", "evt_breakdown_flipped_support",
    "evt_pierce_support_cluster_reclaim_low_volume",
    "is_support_flipped", "flipped_support_ref", "flipped_support_age_bars",
    "flipped_support_pierce_depth_atr", "flipped_support_reclaim_strength_atr",
    "support_cluster_is_strong", "support_cluster_score",
    "support_confluence_score", "support_confluence_is_strong",
    "is_volume_shrink", "is_volume_expansion", "volume_z_20",
    "support_reclaim_strength_atr", "close_pos_in_bar",
    "is_long_lower_shadow", "is_shallow_support_pierce", "is_deep_support_pierce",
    "active_support_ref", "resistance_ref", "resistance_active",
    "fwd_ret_20", "fwd_max_ret_20", "fwd_mdd_20",
    "fwd_reward_risk_20",
]

DAILY_COLS = [
    "ts_code", "bar_time", "open", "high", "low", "close",
    "close_above_ma20", "close_pos_in_bar", "is_volume_shrink",
    "resistance_zone_high",
]


def _load_weekly_events(weekly_pv: int, rule: str) -> pd.DataFrame:
    import lightgbm as lgb

    print(f"加载周线 events (w_pv{weekly_pv}, 规则={rule})...")
    cols_to_load = list(dict.fromkeys(EVENTS_COLS + ALL_FEATURE_COLS))
    parts = []
    for shard_df in iter_shards("w", weekly_pv, columns=cols_to_load, shard_type="event"):
        parts.append(shard_df)
    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    if "bar_time" in df.columns:
        df["bar_time"] = pd.to_datetime(df["bar_time"])

    print(f"  周线总事件行数: {len(df)}")

    if rule in ("W1", "W2"):
        train_mask = df["bar_time"] <= TRAIN_END
        base_mask = df["evt_pierce_support_cluster_reclaim_low_volume"].fillna(False).astype(bool)
        if isinstance(base_mask, pd.DataFrame):
            base_mask = base_mask.iloc[:, 0]
        signals = df.loc[base_mask].copy()
        print(f"  周线强簇缩量事件: {len(signals)}")

        if rule == "W1":
            reclaim_col = "support_reclaim_strength_atr"
            if reclaim_col in signals.columns:
                threshold = signals.loc[train_mask & base_mask, reclaim_col].median()
                signals = signals[signals[reclaim_col] > threshold].copy()
                print(f"  W1 过滤后: {len(signals)}")
        elif rule == "W2":
            model_path = Path(MODELS_DIR) / "B_cluster_risk.txt"
            if model_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                feature_cols = [c for c in ALL_FEATURE_COLS if c in signals.columns]
                signals["risk_score"] = model.predict(signals[feature_cols])
                threshold = signals.loc[train_mask & base_mask, "risk_score"].quantile(0.70)
                signals = signals[signals["risk_score"] < threshold].copy()
                print(f"  W2 过滤后: {len(signals)}")
        return signals

    return df


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
    print(f"  日线总行数: {len(df)}")
    return df


def _compute_group_stats(df: pd.DataFrame, group_col: str, group_name: str) -> dict:
    if df.empty:
        return {"group": group_name, "count": 0}
    ret = df["fwd_ret_20"].dropna() if "fwd_ret_20" in df.columns else pd.Series()
    mdd = df["fwd_mdd_20"].dropna() if "fwd_mdd_20" in df.columns else pd.Series()
    max_ret = df["fwd_max_ret_20"].dropna() if "fwd_max_ret_20" in df.columns else pd.Series()
    rr = df["fwd_reward_risk_20"].dropna() if "fwd_reward_risk_20" in df.columns else pd.Series()
    return {
        "group": group_name,
        "count": len(df),
        "mean_ret_20": ret.mean() if len(ret) > 0 else np.nan,
        "median_ret_20": ret.median() if len(ret) > 0 else np.nan,
        "win_rate": (ret > 0).mean() if len(ret) > 0 else np.nan,
        "mean_mdd_20": mdd.mean() if len(mdd) > 0 else np.nan,
        "mean_max_ret_20": max_ret.mean() if len(max_ret) > 0 else np.nan,
        "mean_rr_20": rr.mean() if len(rr) > 0 else np.nan,
    }


def experiment1_r2s_retest(df: pd.DataFrame) -> pd.DataFrame:
    print("\n实验1: R2S刺破收回确认池验证")
    print("=" * 80)

    pierce_reclaim = df[df["evt_pierce_flipped_support_reclaim"].fillna(False).astype(bool)].copy()
    print(f"  R2S刺破收回事件总数: {len(pierce_reclaim)}")

    if pierce_reclaim.empty:
        print("  无R2S刺破收回事件")
        return pd.DataFrame()

    retest = df[df["evt_retest_flipped_support"].fillna(False).astype(bool)].copy()
    breakdown = df[df["evt_breakdown_flipped_support"].fillna(False).astype(bool)].copy()

    groups = [
        ("R2S刺破收回全部(基线)", pierce_reclaim),
        ("R2S刺破收回+缩量", pierce_reclaim[pierce_reclaim["is_volume_shrink"].fillna(False).astype(bool)]),
        ("R2S刺破收回+放量", pierce_reclaim[pierce_reclaim["is_volume_expansion"].fillna(False).astype(bool)]),
        ("R2S刺破收回+强支撑簇", pierce_reclaim[pierce_reclaim["support_cluster_is_strong"].fillna(False).astype(bool)]),
        ("R2S刺破收回+收盘上半部", pierce_reclaim[pierce_reclaim["close_pos_in_bar"].fillna(0) >= 0.6]),
        ("R2S刺破收回+老R2S(>20bar)", pierce_reclaim[pierce_reclaim["flipped_support_age_bars"].fillna(0) > 20]),
        ("R2S刺破收回+新R2S(<=20bar)", pierce_reclaim[pierce_reclaim["flipped_support_age_bars"].fillna(999) <= 20]),
        ("R2S刺破收回+浅刺破", pierce_reclaim[pierce_reclaim["is_shallow_support_pierce"].fillna(False).astype(bool)] if "is_shallow_support_pierce" in pierce_reclaim.columns else pd.DataFrame()),
        ("R2S刺破收回+深刺破", pierce_reclaim[pierce_reclaim["is_deep_support_pierce"].fillna(False).astype(bool)] if "is_deep_support_pierce" in pierce_reclaim.columns else pd.DataFrame()),
        ("[对照]R2S回踩不破", retest),
        ("[对照]R2S回踩不破+缩量", retest[retest["is_volume_shrink"].fillna(False).astype(bool)]),
        ("[风险]R2S跌破", breakdown),
    ]

    rows = []
    for name, sub in groups:
        rows.append(_compute_group_stats(sub, "group", name))

    result = pd.DataFrame(rows)
    display_cols = [c for c in ["group", "count", "mean_ret_20", "median_ret_20",
                                 "win_rate", "mean_mdd_20", "mean_max_ret_20", "mean_rr_20"]
                    if c in result.columns]
    print(tabulate(result[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    return result


def experiment2_momentum_pullback(
    combined: pd.DataFrame,
    daily_df: pd.DataFrame,
    hold_days: int = 20,
) -> pd.DataFrame:
    print("\n实验2: 动量回踩确认池验证")
    print("=" * 80)

    bp = combined[combined["tier"] == "B_watch_pullback"].copy()
    print(f"  B_watch_pullback 样本数: {len(bp)}")

    if bp.empty:
        print("  无B_watch_pullback样本")
        return pd.DataFrame()

    daily_sorted = daily_df.sort_values(["ts_code", "bar_time"]).copy()
    daily_by_code = {code: group for code, group in daily_sorted.groupby("ts_code")}

    entry_methods = {
        "直接追(次日开盘)": _simulate_direct_entry,
        "等回踩周线low附近(3%内)": _simulate_weekly_low_pullback,
        "等回踩不破active_support": _simulate_support_pullback,
        "等缩量转阳再买": _simulate_shrink_positive_entry,
    }

    rows = []
    for method_name, sim_func in entry_methods.items():
        trades = sim_func(bp, daily_by_code, hold_days)
        rows.append(_compute_trade_stats(trades, method_name))

    result = pd.DataFrame(rows)
    display_cols = [c for c in ["method", "count", "mean_ret", "median_ret", "win_rate",
                                 "stop_rate", "plr", "max_loss", "max_gain"]
                    if c in result.columns]
    print(tabulate(result[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    return result


def experiment3_low_buy_stop_confirm(
    combined: pd.DataFrame,
    daily_df: pd.DataFrame,
    hold_days: int = 20,
) -> pd.DataFrame:
    print("\n实验3: 低吸止跌确认池验证")
    print("=" * 80)

    sl = combined[combined["tier"] == "S_low_buy"].copy()
    print(f"  S_low_buy 样本数: {len(sl)}")

    if sl.empty:
        print("  无S_low_buy样本")
        return pd.DataFrame()

    daily_sorted = daily_df.sort_values(["ts_code", "bar_time"]).copy()
    daily_by_code = {code: group for code, group in daily_sorted.groupby("ts_code")}

    entry_methods = {
        "直接买(次日开盘)": _simulate_direct_entry,
        "等3日不破周线low": _simulate_3day_no_break,
        "等日线收盘上半部": _simulate_close_upper_half,
        "等缩量转阳": _simulate_shrink_positive,
    }

    rows = []
    for method_name, sim_func in entry_methods.items():
        trades = sim_func(sl, daily_by_code, hold_days)
        rows.append(_compute_trade_stats(trades, method_name))

    result = pd.DataFrame(rows)
    display_cols = [c for c in ["method", "count", "mean_ret", "median_ret", "win_rate",
                                 "stop_rate", "plr", "max_loss", "max_gain"]
                    if c in result.columns]
    print(tabulate(result[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    return result


def _simulate_direct_entry(signals, daily_by_code, hold_days):
    trades = []
    for _, w_row in signals.iterrows():
        result = _execute_trade(w_row, daily_by_code, hold_days, entry_offset=0)
        if result is not None:
            trades.append(result)
    return trades


def _simulate_weekly_low_pullback(signals, daily_by_code, hold_days):
    trades = []
    for _, w_row in signals.iterrows():
        result = _wait_and_execute(w_row, daily_by_code, hold_days,
                                    wait_days=15,
                                    condition_func=_weekly_low_near_condition)
        if result is not None:
            trades.append(result)
    return trades


def _simulate_shrink_positive_entry(signals, daily_by_code, hold_days):
    trades = []
    for _, w_row in signals.iterrows():
        result = _wait_and_execute(w_row, daily_by_code, hold_days,
                                    wait_days=15,
                                    condition_func=_shrink_positive_condition)
        if result is not None:
            trades.append(result)
    return trades


def _simulate_support_pullback(signals, daily_by_code, hold_days):
    trades = []
    for _, w_row in signals.iterrows():
        result = _wait_and_execute(w_row, daily_by_code, hold_days,
                                    wait_days=15,
                                    condition_func=_support_condition)
        if result is not None:
            trades.append(result)
    return trades


def _simulate_3day_no_break(signals, daily_by_code, hold_days):
    trades = []
    for _, w_row in signals.iterrows():
        result = _wait_and_execute(w_row, daily_by_code, hold_days,
                                    wait_days=5,
                                    condition_func=_no_break_3day_condition)
        if result is not None:
            trades.append(result)
    return trades


def _simulate_close_upper_half(signals, daily_by_code, hold_days):
    trades = []
    for _, w_row in signals.iterrows():
        result = _wait_and_execute(w_row, daily_by_code, hold_days,
                                    wait_days=10,
                                    condition_func=_close_upper_condition)
        if result is not None:
            trades.append(result)
    return trades


def _simulate_shrink_positive(signals, daily_by_code, hold_days):
    trades = []
    for _, w_row in signals.iterrows():
        result = _wait_and_execute(w_row, daily_by_code, hold_days,
                                    wait_days=10,
                                    condition_func=_shrink_positive_condition)
        if result is not None:
            trades.append(result)
    return trades


def _get_daily_group(w_row, daily_by_code):
    ts_code = w_row["ts_code"]
    return daily_by_code.get(ts_code)


def _get_entry_index(w_row, d_group):
    if d_group is None:
        return None
    w_time = w_row["bar_time"]
    d_bartimes = d_group["bar_time"].values
    entry_idx = np.searchsorted(d_bartimes, w_time, side="right")
    if entry_idx >= len(d_bartimes):
        return None
    return entry_idx


def _execute_trade(w_row, daily_by_code, hold_days, entry_offset=0):
    d_group = _get_daily_group(w_row, daily_by_code)
    if d_group is None:
        return None

    base_idx = _get_entry_index(w_row, d_group)
    if base_idx is None:
        return None

    entry_idx = base_idx + entry_offset
    if entry_idx >= len(d_group):
        return None

    entry_bar = d_group.iloc[entry_idx]
    entry_price = entry_bar.get("open", np.nan)
    if pd.isna(entry_price) or entry_price <= 0:
        return None

    w_low = w_row.get("low", np.nan)
    exit_price = np.nan
    is_stopped = False
    actual_hold = 0

    for day_offset in range(hold_days):
        bar_idx = entry_idx + day_offset
        if bar_idx >= len(d_group):
            break
        bar = d_group.iloc[bar_idx]
        actual_hold = day_offset + 1

        if pd.notna(w_low) and pd.notna(bar.get("low")) and bar["low"] <= w_low:
            exit_price = w_low
            is_stopped = True
            break

        if day_offset == hold_days - 1 or bar_idx == len(d_group) - 1:
            exit_price = bar.get("close", np.nan)
            break

    if pd.isna(exit_price):
        return None

    pnl_pct = (exit_price - entry_price) / entry_price
    return {"pnl_pct": pnl_pct, "is_stopped": is_stopped, "hold_days": actual_hold}


def _wait_and_execute(w_row, daily_by_code, hold_days, wait_days, condition_func):
    d_group = _get_daily_group(w_row, daily_by_code)
    if d_group is None:
        return None

    base_idx = _get_entry_index(w_row, d_group)
    if base_idx is None:
        return None

    w_low = w_row.get("low", np.nan)

    for wait_offset in range(wait_days):
        check_idx = base_idx + wait_offset
        if check_idx >= len(d_group):
            return None
        bar = d_group.iloc[check_idx]

        if pd.notna(w_low) and pd.notna(bar.get("low")) and bar["low"] <= w_low:
            return {"pnl_pct": (w_low / w_row.get("close", np.nan) - 1) if pd.notna(w_row.get("close")) else np.nan,
                    "is_stopped": True, "hold_days": wait_offset + 1}

        if condition_func(bar, w_row):
            return _execute_trade(w_row, daily_by_code, hold_days, entry_offset=wait_offset + 1)

    return None


def _weekly_low_near_condition(bar, w_row):
    w_low = w_row.get("low", np.nan)
    low = bar.get("low", np.nan)
    close = bar.get("close", np.nan)
    if pd.isna(w_low) or pd.isna(low) or pd.isna(close) or w_low <= 0:
        return False
    dist_pct = (low - w_low) / w_low
    return dist_pct >= -0.01 and dist_pct <= 0.03 and close > w_low


def _support_condition(bar, w_row):
    active_support = w_row.get("active_support_ref", np.nan)
    low = bar.get("low", np.nan)
    close = bar.get("close", np.nan)
    if pd.isna(active_support) or pd.isna(low) or pd.isna(close):
        return False
    dist_pct = (low - active_support) / active_support
    return dist_pct >= -0.02 and dist_pct <= 0.05 and close > active_support


def _no_break_3day_condition(bar, w_row):
    return True


def _close_upper_condition(bar, w_row):
    close_pos = bar.get("close_pos_in_bar", np.nan)
    if pd.isna(close_pos):
        close = bar.get("close", np.nan)
        low = bar.get("low", np.nan)
        high = bar.get("high", np.nan)
        if pd.notna(close) and pd.notna(low) and pd.notna(high) and high > low:
            close_pos = (close - low) / (high - low)
    if pd.isna(close_pos):
        return False
    return close_pos >= 0.6


def _shrink_positive_condition(bar, w_row):
    is_shrink = bool(bar.get("is_volume_shrink", False))
    close = bar.get("close", np.nan)
    open_ = bar.get("open", np.nan)
    is_positive = pd.notna(close) and pd.notna(open_) and close > open_
    return is_shrink and is_positive


def _compute_trade_stats(trades, method_name):
    if not trades:
        return {"method": method_name, "count": 0}

    trades_df = pd.DataFrame(trades)
    pnls = trades_df["pnl_pct"].dropna()
    if pnls.empty:
        return {"method": method_name, "count": len(trades)}

    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 1e-9

    return {
        "method": method_name,
        "count": len(trades),
        "mean_ret": pnls.mean(),
        "median_ret": pnls.median(),
        "win_rate": (pnls > 0).mean(),
        "stop_rate": trades_df["is_stopped"].mean(),
        "plr": avg_win / avg_loss if avg_loss > 0 else np.nan,
        "max_loss": pnls.min(),
        "max_gain": pnls.max(),
    }


def main():
    parser = argparse.ArgumentParser(description="场景化入场验证")
    parser.add_argument("--weekly-pv", type=int, default=10)
    parser.add_argument("--daily-pv", type=int, default=20)
    parser.add_argument("--rule", type=str, default="W2", choices=["W0", "W1", "W2"])
    args = parser.parse_args()

    df = _load_weekly_events(args.weekly_pv, args.rule)
    if df.empty:
        print("无周线数据")
        return

    df_full = _load_weekly_events(args.weekly_pv, "W0")
    if df_full.empty:
        print("无周线全量数据")
        return

    daily_df = _load_daily_panel(args.daily_pv)
    if daily_df.empty:
        print("无日线数据")
        return

    if "bar_time" in df.columns:
        df["split"] = "train"
        df.loc[df["bar_time"] > TRAIN_END, "split"] = "val"
        df.loc[df["bar_time"] > VAL_END, "split"] = "test"

    if "bar_time" in df_full.columns:
        df_full["split"] = "train"
        df_full.loc[df_full["bar_time"] > TRAIN_END, "split"] = "val"
        df_full.loc[df_full["bar_time"] > VAL_END, "split"] = "test"

    print(f"\n{'='*100}")
    print(f"场景化入场验证报告: 规则={args.rule}")
    print(f"{'='*100}")

    r2s_result = experiment1_r2s_retest(df_full)

    print("\n计算评分+分层(用于实验2/3)...")
    daily_sorted = daily_df.sort_values(["ts_code", "bar_time"]).copy()
    daily_by_code = {code: group for code, group in daily_sorted.groupby("ts_code")}

    base_mask = df["evt_pierce_support_cluster_reclaim_low_volume"].fillna(False).astype(bool)
    if isinstance(base_mask, pd.DataFrame):
        base_mask = base_mask.iloc[:, 0]
    signals = df.loc[base_mask].copy()
    print(f"  周线强簇缩量事件: {len(signals)}")

    daily_status_rows = []
    for _, w_row in signals.iterrows():
        ts_code = w_row["ts_code"]
        w_time = w_row["bar_time"]
        w_low = w_row.get("low", np.nan)
        w_close = w_row.get("close", np.nan)

        d_group = daily_by_code.get(ts_code)
        if d_group is None:
            daily_status_rows.append({"daily_broken_weekly_low": False, "distance_to_weekly_low_pct": np.nan,
                                      "current_to_resistance_pct": np.nan, "ret_since_signal": np.nan,
                                      "latest_daily_close": np.nan})
            continue

        d_bartimes = d_group["bar_time"].values
        idx_start = np.searchsorted(d_bartimes, w_time, side="right")
        idx_end = min(idx_start + 10, len(d_bartimes))
        window_df = d_group.iloc[idx_start:idx_end]

        broken = False
        latest_close = np.nan
        if not window_df.empty:
            latest_close = window_df.iloc[-1].get("close", np.nan)
            for _, d_row in window_df.iterrows():
                if pd.notna(w_low) and pd.notna(d_row.get("low")) and d_row["low"] < w_low:
                    broken = True

        dist_pct = (latest_close - w_low) / w_low * 100 if pd.notna(latest_close) and pd.notna(w_low) and w_low > 0 else np.nan
        ret_since = (latest_close - w_close) / w_close * 100 if pd.notna(latest_close) and pd.notna(w_close) and w_close > 0 else np.nan

        resistance = w_row.get("resistance_ref", np.nan)
        if pd.isna(resistance):
            resistance = w_row.get("resistance_active", np.nan)
        dist_resistance = (resistance - latest_close) / latest_close * 100 if pd.notna(resistance) and pd.notna(latest_close) and latest_close > 0 else np.nan

        daily_status_rows.append({
            "daily_broken_weekly_low": broken,
            "distance_to_weekly_low_pct": dist_pct,
            "current_to_resistance_pct": dist_resistance,
            "ret_since_signal": ret_since,
            "latest_daily_close": latest_close,
        })

    daily_status = pd.DataFrame(daily_status_rows)
    combined = pd.concat([signals.reset_index(drop=True), daily_status.reset_index(drop=True)], axis=1)

    reclaim_median = combined["support_reclaim_strength_atr"].median() if "support_reclaim_strength_atr" in combined.columns else 0
    combined = assign_tier(combined, reclaim_median)

    if "bar_time" in combined.columns:
        combined["split"] = "train"
        combined.loc[combined["bar_time"] > TRAIN_END, "split"] = "val"
        combined.loc[combined["bar_time"] > VAL_END, "split"] = "test"

    momentum_result = experiment2_momentum_pullback(combined, daily_df)
    low_buy_result = experiment3_low_buy_stop_confirm(combined, daily_df)

    print(f"\n{'='*100}")
    print("四、三条线对比汇总")
    print(f"{'='*100}")

    summary_rows = []

    if not r2s_result.empty:
        best_r2s = r2s_result.loc[r2s_result["mean_ret_20"].idxmax()] if "mean_ret_20" in r2s_result.columns else None
        if best_r2s is not None:
            summary_rows.append({
                "scenario": "R2S刺破收回(最优分组)",
                "best_group": best_r2s["group"],
                "count": best_r2s["count"],
                "mean_ret_20": best_r2s["mean_ret_20"],
                "win_rate": best_r2s.get("win_rate", np.nan),
                "mean_mdd_20": best_r2s.get("mean_mdd_20", np.nan),
                "mean_rr_20": best_r2s.get("mean_rr_20", np.nan),
            })

    if not momentum_result.empty:
        best_mom = momentum_result.loc[momentum_result["mean_ret"].idxmax()] if "mean_ret" in momentum_result.columns else None
        if best_mom is not None:
            summary_rows.append({
                "scenario": "动量回踩(最优入场)",
                "best_group": best_mom["method"],
                "count": best_mom["count"],
                "mean_ret_20": best_mom.get("mean_ret", np.nan),
                "win_rate": best_mom.get("win_rate", np.nan),
                "mean_mdd_20": np.nan,
                "mean_rr_20": best_mom.get("plr", np.nan),
            })

    if not low_buy_result.empty:
        best_lb = low_buy_result.loc[low_buy_result["mean_ret"].idxmax()] if "mean_ret" in low_buy_result.columns else None
        if best_lb is not None:
            summary_rows.append({
                "scenario": "低吸止跌(最优入场)",
                "best_group": best_lb["method"],
                "count": best_lb["count"],
                "mean_ret_20": best_lb.get("mean_ret", np.nan),
                "win_rate": best_lb.get("win_rate", np.nan),
                "mean_mdd_20": np.nan,
                "mean_rr_20": best_lb.get("plr", np.nan),
            })

    summary_df = pd.DataFrame(summary_rows)
    print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    out_dir = Path("sr_experiment/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scenario_entry_validation.csv"

    all_results = combined.copy()
    all_results.to_csv(out_path, index=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
