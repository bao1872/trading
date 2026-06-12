# -*- coding: utf-8 -*-
"""
Purpose: 周线信号后的日线入场确认实验——周线选环境，日线找入场
Inputs:  周线分片 parquet (w_pv10) + 日线分片 parquet (d_pv10/d_pv20/d_pv30)
Outputs: 控制台对比表（有/无日线确认信号的收益对比）
How to Run:
    python sr_experiment/12_weekly_signal_daily_entry.py --weekly-pv 10 --daily-pv 20 --entry-window 10
    python sr_experiment/12_weekly_signal_daily_entry.py --weekly-pv 10 --daily-pv 10 --entry-window 5
Examples:
    python sr_experiment/12_weekly_signal_daily_entry.py --weekly-pv 10 --daily-pv 20
    python sr_experiment/12_weekly_signal_daily_entry.py --weekly-pv 10 --daily-pv 10 --entry-window 20
Side Effects: 无（纯计算+输出）
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
from sr_experiment.sr_config import PANEL_EVENT_STAT_COLS

_build_mod = importlib.import_module("sr_experiment.00_build_factor_panel_from_db")
iter_shards = _build_mod.iter_shards


WEEKLY_SIGNAL_COLS = [
    "ts_code", "bar_time", "close", "low", "high",
    "evt_pierce_support_cluster_reclaim_low_volume",
    "support_reclaim_strength_atr",
    "fwd_ret_20", "fwd_max_ret_20", "fwd_mdd_20", "fwd_reward_risk_20",
]

DAILY_ENTRY_COLS = [
    "ts_code", "bar_time", "close", "low", "high",
    "evt_pierce_support_reclaim",
    "evt_pierce_strong_support_cluster_reclaim",
    "evt_pierce_support_cluster_reclaim_low_volume",
    "is_volume_shrink",
    "close_above_ma20",
    "close_above_ma60",
    "support_reclaim_strength_atr",
    "sr_pos_01",
    "evt_cross_recent_resistance",
    "evt_close_above_resistance_cluster_upper",
    "evt_wick_break_resistance_cluster_fail",
]


def _load_weekly_signals(weekly_pv: int, rule: str) -> pd.DataFrame:
    import lightgbm as lgb

    print(f"加载周线信号 (w_pv{weekly_pv})...")
    cols_to_load = list(dict.fromkeys(WEEKLY_SIGNAL_COLS + ALL_FEATURE_COLS))
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
    for shard_df in iter_shards("d", daily_pv, columns=DAILY_ENTRY_COLS, shard_type="panel"):
        parts.append(shard_df)
    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    if "bar_time" in df.columns:
        df["bar_time"] = pd.to_datetime(df["bar_time"])
    print(f"  日线总行数: {len(df)}")
    return df


def _find_daily_confirmations(
    weekly_signals: pd.DataFrame,
    daily_df: pd.DataFrame,
    entry_window: int,
) -> pd.DataFrame:
    print(f"匹配日线入场确认 (window={entry_window}个交易日)...")
    daily_sorted = daily_df.sort_values(["ts_code", "bar_time"]).copy()

    bool_cols = [
        "evt_pierce_support_reclaim", "evt_pierce_strong_support_cluster_reclaim",
        "evt_pierce_support_cluster_reclaim_low_volume", "evt_cross_recent_resistance",
        "close_above_ma20",
    ]
    for col in bool_cols:
        if col in daily_sorted.columns:
            daily_sorted[col] = daily_sorted[col].fillna(False).astype(bool)

    daily_by_code = {code: group for code, group in daily_sorted.groupby("ts_code")}

    results = []
    n_matched = 0
    n_total = 0

    for _, w_row in weekly_signals.iterrows():
        n_total += 1
        ts_code = w_row["ts_code"]
        w_time = w_row["bar_time"]
        w_low = w_row.get("low", np.nan)

        d_group = daily_by_code.get(ts_code)
        if d_group is None:
            continue

        d_bartimes = d_group["bar_time"].values
        idx_start = np.searchsorted(d_bartimes, w_time, side="left")
        if idx_start >= len(d_bartimes):
            continue

        idx_end = min(idx_start + entry_window, len(d_bartimes))
        window_df = d_group.iloc[idx_start:idx_end]
        if window_df.empty:
            continue

        n_matched += 1

        has_pierce_reclaim = bool(window_df["evt_pierce_support_reclaim"].any()) if "evt_pierce_support_reclaim" in window_df.columns else False
        has_cluster_reclaim = bool(window_df["evt_pierce_strong_support_cluster_reclaim"].any()) if "evt_pierce_strong_support_cluster_reclaim" in window_df.columns else False
        has_volume_shrink_reclaim = bool(window_df["evt_pierce_support_cluster_reclaim_low_volume"].any()) if "evt_pierce_support_cluster_reclaim_low_volume" in window_df.columns else False
        has_resistance_break = bool(window_df["evt_cross_recent_resistance"].any()) if "evt_cross_recent_resistance" in window_df.columns else False
        has_close_above_ma20 = bool(window_df["close_above_ma20"].any()) if "close_above_ma20" in window_df.columns else False

        has_broken_weekly_low = False
        if "low" in window_df.columns and pd.notna(w_low):
            has_broken_weekly_low = bool((window_df["low"].dropna() < w_low).any())

        has_any_confirm = has_pierce_reclaim or has_cluster_reclaim or has_volume_shrink_reclaim or has_resistance_break

        result = {
            "ts_code": ts_code,
            "weekly_bar_time": w_time,
            "weekly_close": w_row.get("close", np.nan),
            "weekly_low": w_low,
            "fwd_ret_20": w_row.get("fwd_ret_20", np.nan),
            "fwd_max_ret_20": w_row.get("fwd_max_ret_20", np.nan),
            "fwd_mdd_20": w_row.get("fwd_mdd_20", np.nan),
            "has_pierce_reclaim": has_pierce_reclaim,
            "has_cluster_reclaim": has_cluster_reclaim,
            "has_volume_shrink_reclaim": has_volume_shrink_reclaim,
            "has_resistance_break": has_resistance_break,
            "has_close_above_ma20": has_close_above_ma20,
            "has_broken_weekly_low": has_broken_weekly_low,
            "has_any_confirm": has_any_confirm,
        }
        results.append(result)

    print(f"  匹配: {n_matched}/{n_total} 个周线信号找到日线数据")
    return pd.DataFrame(results)


def _eval_group(sub: pd.DataFrame, label: str) -> dict:
    if sub.empty:
        return {"group": label, "count": 0}
    row = {"group": label, "count": len(sub)}
    for col in ["fwd_ret_20", "fwd_max_ret_20", "fwd_mdd_20"]:
        if col in sub.columns:
            vals = sub[col].dropna()
            row[col] = vals.mean() if len(vals) > 0 else np.nan
    if "fwd_ret_20" in sub.columns:
        vals = sub["fwd_ret_20"].dropna()
        row["win_rate"] = (vals > 0).mean() if len(vals) > 0 else np.nan
    if "fwd_max_ret_20" in sub.columns and "fwd_mdd_20" in sub.columns:
        max_r = sub["fwd_max_ret_20"].dropna()
        mdd = sub["fwd_mdd_20"].dropna().abs()
        if len(max_r) > 0 and len(mdd) > 0 and mdd.mean() > 0:
            row["reward_risk"] = max_r.mean() / mdd.mean()
    return row


def main():
    parser = argparse.ArgumentParser(description="周线信号后的日线入场确认实验")
    parser.add_argument("--weekly-pv", type=int, default=10, help="周线 pivot_len")
    parser.add_argument("--daily-pv", type=int, default=20, help="日线 pivot_len")
    parser.add_argument("--entry-window", type=int, default=10, help="入场确认窗口（交易日数）")
    parser.add_argument("--rule", type=str, default="W0", choices=["W0", "W1", "W2"], help="周线规则")
    args = parser.parse_args()

    weekly_signals = _load_weekly_signals(args.weekly_pv, args.rule)
    if weekly_signals.empty:
        print("无周线信号")
        return

    daily_df = _load_daily_panel(args.daily_pv)
    if daily_df.empty:
        print("无日线数据")
        return

    matched = _find_daily_confirmations(weekly_signals, daily_df, args.entry_window)
    if matched.empty:
        print("无匹配结果")
        return

    if "weekly_bar_time" in matched.columns:
        matched["split"] = "train"
        matched.loc[matched["weekly_bar_time"] > TRAIN_END, "split"] = "val"
        matched.loc[matched["weekly_bar_time"] > VAL_END, "split"] = "test"

    print(f"\n{'='*80}")
    print(f"周线规则: {args.rule}, 日线pv: {args.daily_pv}, 入场窗口: {args.entry_window}个交易日")
    print(f"{'='*80}")

    confirm_cols = [
        ("has_any_confirm", "任一确认信号"),
        ("has_pierce_reclaim", "日线刺破收回"),
        ("has_cluster_reclaim", "日线强簇收回"),
        ("has_volume_shrink_reclaim", "日线缩量收回"),
        ("has_resistance_break", "日线突破压力"),
        ("has_close_above_ma20", "日线站上MA20"),
        ("has_broken_weekly_low", "日线跌破周线low"),
    ]

    for split_name in ["train", "val", "test"]:
        split_df = matched[matched["split"] == split_name]
        if split_df.empty:
            continue

        print(f"\n--- {split_name} (N={len(split_df)}) ---")

        rows = [_eval_group(split_df, "全部周线信号")]

        for col, label in confirm_cols:
            if col in split_df.columns:
                yes = split_df[split_df[col] == True]
                no = split_df[split_df[col] == False]
                rows.append(_eval_group(yes, f"有{label}"))
                rows.append(_eval_group(no, f"无{label}"))

        result_df = pd.DataFrame(rows)
        display_cols = [c for c in ["group", "count", "fwd_ret_20", "fwd_max_ret_20",
                                     "fwd_mdd_20", "reward_risk", "win_rate"]
                        if c in result_df.columns]
        print(tabulate(result_df[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    print(f"\n\n========== Test 集总结 ==========")
    test_df = matched[matched["split"] == "test"]
    if not test_df.empty:
        base = _eval_group(test_df, "全部周线信号")
        print(f"全部: N={base.get('count',0)}, ret_20={base.get('fwd_ret_20',np.nan):.4f}, "
              f"win_rate={base.get('win_rate',np.nan):.4f}, reward_risk={base.get('reward_risk',np.nan):.4f}")

        for col, label in confirm_cols:
            if col in test_df.columns:
                yes = test_df[test_df[col] == True]
                no = test_df[test_df[col] == False]
                y = _eval_group(yes, label)
                n = _eval_group(no, f"无{label}")
                print(f"有{label}: N={y.get('count',0)}, ret_20={y.get('fwd_ret_20',np.nan):.4f}, "
                      f"win_rate={y.get('win_rate',np.nan):.4f}")
                print(f"无{label}: N={n.get('count',0)}, ret_20={n.get('fwd_ret_20',np.nan):.4f}, "
                      f"win_rate={n.get('win_rate',np.nan):.4f}")


if __name__ == "__main__":
    main()
