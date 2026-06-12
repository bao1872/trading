# -*- coding: utf-8 -*-
"""
Purpose: 人工交易机会过滤器 v2——周线信号评分+分位数分层+买点状态+原因标签
Inputs:  周线分片 parquet (w_pv10) + 日线分片 parquet (d_pv20) + GBDT风险模型
Outputs: sr_manual_candidates_YYYYMMDD.xlsx (分sheet: S低吸/S观察/A观察/B弱/C剔除/汇总)
How to Run:
    python sr_experiment/14_manual_trade_candidate_report.py
    python sr_experiment/14_manual_trade_candidate_report.py --date 20250518
    python sr_experiment/14_manual_trade_candidate_report.py --rule W2
Examples:
    python sr_experiment/14_manual_trade_candidate_report.py
    python sr_experiment/14_manual_trade_candidate_report.py --date 20250518 --rule W2
Side Effects: 写入 sr_experiment/results/sr_manual_candidates_YYYYMMDD.xlsx
"""
from __future__ import annotations

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sr_experiment.gbdt_config import MODELS_DIR, TRAIN_END
from sr_experiment.gbdt_feature_columns import ALL_FEATURE_COLS
from sr_experiment.manual_filter_scoring import (
    TIER_ORDER,
    assign_tier,
)

_build_mod = importlib.import_module("sr_experiment.00_build_factor_panel_from_db")
iter_shards = _build_mod.iter_shards

WEEKLY_COLS = [
    "ts_code", "bar_time", "open", "close", "low", "high",
    "evt_pierce_support_cluster_reclaim_low_volume",
    "evt_pierce_support_cluster_reclaim_high_volume",
    "support_cluster_is_strong", "support_cluster_score",
    "support_confluence_score", "support_confluence_is_strong",
    "is_volume_shrink", "volume_z_20",
    "support_reclaim_strength_atr", "close_pos_in_bar",
    "is_long_lower_shadow", "is_support_flipped",
    "active_support_ref", "resistance_ref", "resistance_active",
    "fwd_ret_20", "fwd_max_ret_20", "fwd_mdd_20",
]

DAILY_COLS = [
    "ts_code", "bar_time", "open", "high", "low", "close",
    "close_above_ma20", "close_pos_in_bar", "is_volume_shrink",
    "resistance_zone_high",
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

    model_path = Path(MODELS_DIR) / "B_cluster_risk.txt"
    if model_path.exists():
        model = lgb.Booster(model_file=str(model_path))
        feature_cols = [c for c in ALL_FEATURE_COLS if c in signals.columns]
        signals["risk_score"] = model.predict(signals[feature_cols])
        print(f"  risk_score 已预测")

    train_mask = signals["bar_time"] <= TRAIN_END
    if rule == "W1":
        reclaim_col = "support_reclaim_strength_atr"
        if reclaim_col in signals.columns:
            threshold = signals.loc[train_mask, reclaim_col].median()
            signals = signals[signals[reclaim_col] > threshold].copy()
            print(f"  W1 过滤后: {len(signals)}")
    elif rule == "W2":
        if "risk_score" in signals.columns:
            threshold = signals.loc[train_mask, "risk_score"].quantile(0.70)
            signals = signals[signals["risk_score"] < threshold].copy()
            print(f"  W2 过滤后: {len(signals)}")

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
    print(f"  日线总行数: {len(df)}")
    return df


def _get_daily_status(
    weekly_signals: pd.DataFrame,
    daily_df: pd.DataFrame,
    lookback: int = 10,
) -> pd.DataFrame:
    print("获取日线状态...")
    daily_sorted = daily_df.sort_values(["ts_code", "bar_time"]).copy()
    daily_by_code = {code: group for code, group in daily_sorted.groupby("ts_code")}

    results = []
    for _, w_row in weekly_signals.iterrows():
        ts_code = w_row["ts_code"]
        w_time = w_row["bar_time"]
        w_low = w_row.get("low", np.nan)
        w_close = w_row.get("close", np.nan)

        d_group = daily_by_code.get(ts_code)
        if d_group is None:
            results.append({"daily_broken_weekly_low": False, "daily_above_ma20": False,
                           "daily_repair_signal": False, "distance_to_weekly_low_pct": np.nan,
                           "current_to_active_support_pct": np.nan, "current_to_resistance_pct": np.nan,
                           "ret_since_signal": np.nan, "bars_since_signal": 0,
                           "latest_daily_close": np.nan, "latest_daily_date": pd.NaT})
            continue

        d_bartimes = d_group["bar_time"].values
        idx_start = np.searchsorted(d_bartimes, w_time, side="right")

        broken = False
        above_ma20 = False
        repair = False
        latest_close = np.nan
        latest_date = pd.NaT

        idx_end = min(idx_start + lookback, len(d_bartimes))
        window_df = d_group.iloc[idx_start:idx_end]

        if not window_df.empty:
            latest_bar = window_df.iloc[-1]
            latest_close = latest_bar.get("close", np.nan)
            latest_date = latest_bar["bar_time"]

            for _, d_row in window_df.iterrows():
                if pd.notna(w_low) and pd.notna(d_row.get("low")) and d_row["low"] < w_low:
                    broken = True
                if bool(d_row.get("close_above_ma20", False)):
                    above_ma20 = True
                if bool(d_row.get("is_volume_shrink", False)) and pd.notna(d_row.get("close")) and pd.notna(w_low) and d_row["close"] > w_low:
                    repair = True

        dist_to_low = (latest_close - w_low) / w_low if pd.notna(latest_close) and pd.notna(w_low) and w_low > 0 else np.nan

        active_support = w_row.get("active_support_ref", np.nan)
        dist_to_support = (latest_close - active_support) / active_support if pd.notna(latest_close) and pd.notna(active_support) and active_support > 0 else np.nan

        resistance = w_row.get("resistance_ref", np.nan)
        if pd.isna(resistance):
            resistance = w_row.get("resistance_active", np.nan)
        dist_to_resistance = (resistance - latest_close) / latest_close if pd.notna(resistance) and pd.notna(latest_close) and latest_close > 0 else np.nan

        ret_since = (latest_close - w_close) / w_close if pd.notna(latest_close) and pd.notna(w_close) and w_close > 0 else np.nan

        results.append({
            "daily_broken_weekly_low": broken,
            "daily_above_ma20": above_ma20,
            "daily_repair_signal": repair,
            "distance_to_weekly_low_pct": dist_to_low * 100 if pd.notna(dist_to_low) else np.nan,
            "current_to_active_support_pct": dist_to_support * 100 if pd.notna(dist_to_support) else np.nan,
            "current_to_resistance_pct": dist_to_resistance * 100 if pd.notna(dist_to_resistance) else np.nan,
            "ret_since_signal": ret_since * 100 if pd.notna(ret_since) else np.nan,
            "bars_since_signal": len(window_df),
            "latest_daily_close": latest_close,
            "latest_daily_date": latest_date,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="人工交易机会过滤器 v2")
    parser.add_argument("--weekly-pv", type=int, default=10)
    parser.add_argument("--daily-pv", type=int, default=20)
    parser.add_argument("--rule", type=str, default="all", choices=["W0", "W1", "W2", "all"])
    parser.add_argument("--date", type=str, default=None, help="指定日期 YYYYMMDD，默认最新")
    args = parser.parse_args()

    rules = ["W0", "W1", "W2"] if args.rule == "all" else [args.rule]

    all_candidates = []

    for rule in rules:
        weekly_signals = _load_weekly_signals(args.weekly_pv, rule)
        if weekly_signals.empty:
            continue

        daily_df = _load_daily_panel(args.daily_pv)
        if daily_df.empty:
            continue

        daily_status = _get_daily_status(weekly_signals, daily_df)
        combined = pd.concat([weekly_signals.reset_index(drop=True), daily_status.reset_index(drop=True)], axis=1)

        reclaim_median = combined["support_reclaim_strength_atr"].median() if "support_reclaim_strength_atr" in combined.columns else 0

        combined = assign_tier(combined, reclaim_median)
        combined["weekly_rule"] = rule

        all_candidates.append(combined)

    if not all_candidates:
        print("无候选信号")
        return

    df = pd.concat(all_candidates, ignore_index=True)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    target_date = args.date or datetime.now().strftime("%Y%m%d")

    output_cols = [
        "rank", "ts_code", "bar_time", "tier", "final_score",
        "entry_status", "manual_rr", "weekly_rule",
        "support_cluster_score", "support_confluence_score",
        "volume_z_20", "support_reclaim_strength_atr", "risk_score",
        "distance_to_weekly_low_pct", "daily_broken_weekly_low",
        "daily_repair_signal", "daily_above_ma20",
        "current_to_active_support_pct", "current_to_resistance_pct",
        "upside_to_resistance_pct", "downside_to_weekly_low_pct",
        "ret_since_signal", "bars_since_signal",
        "fwd_ret_20", "fwd_max_ret_20", "fwd_mdd_20",
        "positive_tags", "negative_tags", "action_suggestion",
        "reason", "risk_warning",
    ]
    existing_cols = [c for c in output_cols if c in df.columns]
    df_out = df[existing_cols].copy()

    sheet_map = {
        "S_low_buy": "S低吸候选",
        "S_watch_breakout": "S观察候选",
        "A_observe": "A观察池",
        "B_watch_pullback": "B等回踩",
        "B_weak_ignore": "B弱忽略",
        "C_exclude": "C硬剔除",
    }

    summary_rows = []
    for tier in TIER_ORDER:
        sub = df_out[df_out["tier"] == tier]
        n = len(sub)
        summary_rows.append({
            "tier": tier,
            "count": n,
            "pct": f"{n/len(df)*100:.1f}%",
            "avg_score": sub["final_score"].mean() if n > 0 else np.nan,
            "avg_fwd_ret_20": sub["fwd_ret_20"].mean() if "fwd_ret_20" in sub.columns and n > 0 else np.nan,
            "avg_risk_score": sub["risk_score"].mean() if "risk_score" in sub.columns and n > 0 else np.nan,
            "avg_manual_rr": sub["manual_rr"].mean() if "manual_rr" in sub.columns and n > 0 else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows)

    out_dir = Path("sr_experiment/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sr_manual_candidates_{target_date}.xlsx"

    with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
        for tier, sheet_name in sheet_map.items():
            sub = df_out[df_out["tier"] == tier]
            if not sub.empty:
                sub.to_excel(writer, sheet_name=sheet_name, index=False)
        summary_df.to_excel(writer, sheet_name="统计汇总", index=False)

    print(f"\n{'='*80}")
    print(f"人工交易候选清单 v2: {target_date}")
    print(f"{'='*80}")
    print(f"\n--- 分层统计 ---")
    for tier in TIER_ORDER:
        sub = df_out[df_out["tier"] == tier]
        n = len(sub)
        print(f"  {tier}: {n} ({n/len(df)*100:.1f}%)")
    print(f"\n  总候选: {len(df)}")
    print(f"  输出: {out_path}")

    s_low = df_out[df_out["tier"] == "S_low_buy"]
    if not s_low.empty:
        print(f"\n--- S低吸候选 (Top 10) ---")
        display = s_low.head(10)
        for _, row in display.iterrows():
            print(f"  {row['ts_code']} | score={row['final_score']:.0f} | "
                  f"rr={row.get('manual_rr', np.nan):.2f} | "
                  f"risk={row.get('risk_score', np.nan):.3f} | "
                  f"dist_low={row.get('distance_to_weekly_low_pct', np.nan):.1f}% | "
                  f"entry={row.get('entry_status', '')} | "
                  f"action={row.get('action_suggestion', '')}")


if __name__ == "__main__":
    main()
