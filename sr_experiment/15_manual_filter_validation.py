# -*- coding: utf-8 -*-
"""
Purpose: 验证评分分层效果 v3——新分层+实际交易收益模拟(方案B)+自然收益对比
Inputs:  周线分片 parquet (w_pv10) + 日线分片 parquet (d_pv20) + GBDT风险模型
Outputs: 控制台对比表 + manual_filter_validation_v3.csv
How to Run:
    python sr_experiment/15_manual_filter_validation.py --weekly-pv 10 --daily-pv 20
    python sr_experiment/15_manual_filter_validation.py --weekly-pv 10 --daily-pv 20 --rule W2
Examples:
    python sr_experiment/15_manual_filter_validation.py --weekly-pv 10 --daily-pv 20
    python sr_experiment/15_manual_filter_validation.py --weekly-pv 10 --daily-pv 20 --rule W2
Side Effects: 写入 sr_experiment/results/manual_filter_validation_v3.csv
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


def _simulate_plan_b_by_tier(
    combined: pd.DataFrame,
    daily_df: pd.DataFrame,
    hold_days: int = 20,
) -> dict:
    print(f"模拟方案B实际交易收益 (持有{hold_days}日)...")
    daily_sorted = daily_df.sort_values(["ts_code", "bar_time"]).copy()
    daily_by_code = {code: group for code, group in daily_sorted.groupby("ts_code")}

    tier_metrics = {}
    for tier in TIER_ORDER:
        tier_signals = combined[combined["tier"] == tier]
        if tier_signals.empty:
            tier_metrics[tier] = {"count": 0}
            continue

        trades = []
        for _, w_row in tier_signals.iterrows():
            ts_code = w_row["ts_code"]
            w_time = w_row["bar_time"]
            w_close = w_row.get("close", np.nan)
            w_low = w_row.get("low", np.nan)

            d_group = daily_by_code.get(ts_code)
            if d_group is None:
                continue

            d_bartimes = d_group["bar_time"].values
            entry_idx = np.searchsorted(d_bartimes, w_time, side="right")
            if entry_idx >= len(d_bartimes):
                continue

            entry_bar = d_group.iloc[entry_idx]
            entry_price = entry_bar.get("open", np.nan)
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            if pd.notna(entry_bar.get("open")) and pd.notna(entry_bar.get("high")) and pd.notna(entry_bar.get("low")):
                if entry_bar["open"] == entry_bar["high"] == entry_bar["low"]:
                    continue

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
                continue

            pnl_pct = (exit_price - entry_price) / entry_price
            trades.append({
                "pnl_pct": pnl_pct,
                "is_stopped": is_stopped,
                "is_winner": pnl_pct > 0,
                "hold_days": actual_hold,
            })

        if trades:
            trades_df = pd.DataFrame(trades)
            pnls = trades_df["pnl_pct"]
            winners = pnls[pnls > 0]
            losers = pnls[pnls <= 0]
            avg_win = winners.mean() if len(winners) > 0 else 0
            avg_loss = abs(losers.mean()) if len(losers) > 0 else 1e-9
            tier_metrics[tier] = {
                "count": len(trades),
                "mean_ret": pnls.mean(),
                "median_ret": pnls.median(),
                "win_rate": (pnls > 0).mean(),
                "stop_rate": trades_df["is_stopped"].mean(),
                "profit_loss_ratio": avg_win / avg_loss if avg_loss > 0 else np.nan,
                "max_loss": pnls.min(),
                "max_gain": pnls.max(),
            }
        else:
            tier_metrics[tier] = {"count": 0}

    return tier_metrics


def main():
    parser = argparse.ArgumentParser(description="验证评分分层效果 v3")
    parser.add_argument("--weekly-pv", type=int, default=10)
    parser.add_argument("--daily-pv", type=int, default=20)
    parser.add_argument("--rule", type=str, default="W2", choices=["W0", "W1", "W2"])
    args = parser.parse_args()

    import lightgbm as lgb

    print(f"加载周线信号 (w_pv{args.weekly_pv}, 规则={args.rule})...")
    cols_to_load = list(dict.fromkeys(WEEKLY_COLS + ALL_FEATURE_COLS))
    parts = []
    for shard_df in iter_shards("w", args.weekly_pv, columns=cols_to_load, shard_type="panel"):
        parts.append(shard_df)
    if not parts:
        print("无周线数据")
        return

    df = pd.concat(parts, ignore_index=True)
    if "bar_time" in df.columns:
        df["bar_time"] = pd.to_datetime(df["bar_time"])

    base_mask = df["evt_pierce_support_cluster_reclaim_low_volume"].fillna(False).astype(bool)
    if isinstance(base_mask, pd.DataFrame):
        base_mask = base_mask.iloc[:, 0]
    signals = df.loc[base_mask].copy()
    print(f"  周线强簇缩量事件: {len(signals)}")

    if args.rule in ("W1", "W2"):
        train_mask = signals["bar_time"] <= TRAIN_END
        if args.rule == "W1":
            reclaim_col = "support_reclaim_strength_atr"
            if reclaim_col in signals.columns:
                threshold = signals.loc[train_mask, reclaim_col].median()
                signals = signals[signals[reclaim_col] > threshold].copy()
                print(f"  W1 过滤后: {len(signals)}")
        elif args.rule == "W2":
            model_path = Path(MODELS_DIR) / "B_cluster_risk.txt"
            if model_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                feature_cols = [c for c in ALL_FEATURE_COLS if c in signals.columns]
                signals["risk_score"] = model.predict(signals[feature_cols])
                threshold = signals.loc[train_mask, "risk_score"].quantile(0.70)
                signals = signals[signals["risk_score"] < threshold].copy()
                print(f"  W2 过滤后: {len(signals)}")

    print(f"加载日线面板 (d_pv{args.daily_pv})...")
    daily_parts = []
    for shard_df in iter_shards("d", args.daily_pv, columns=DAILY_COLS, shard_type="panel"):
        daily_parts.append(shard_df)
    if not daily_parts:
        print("无日线数据")
        return

    daily_df = pd.concat(daily_parts, ignore_index=True)
    if "bar_time" in daily_df.columns:
        daily_df["bar_time"] = pd.to_datetime(daily_df["bar_time"])
    print(f"  日线总行数: {len(daily_df)}")

    print("获取日线状态...")
    daily_sorted = daily_df.sort_values(["ts_code", "bar_time"]).copy()
    daily_by_code = {code: group for code, group in daily_sorted.groupby("ts_code")}

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

    print("计算评分+分层...")
    combined = assign_tier(combined, reclaim_median)

    if "bar_time" in combined.columns:
        combined["split"] = "train"
        combined.loc[combined["bar_time"] > TRAIN_END, "split"] = "val"
        combined.loc[combined["bar_time"] > VAL_END, "split"] = "test"

    tier_trading_metrics = _simulate_plan_b_by_tier(combined, daily_df, hold_days=20)

    print(f"\n{'='*100}")
    print(f"评分分层验证 v3: 规则={args.rule}")
    print(f"{'='*100}")

    print(f"\n{'='*100}")
    print("一、自然收益 (fwd_ret_20) 对比")
    print(f"{'='*100}")

    for split_name in ["train", "val", "test", "all"]:
        if split_name == "all":
            sub = combined
        else:
            sub = combined[combined["split"] == split_name]
        if sub.empty:
            continue

        print(f"\n--- {split_name} (N={len(sub)}) ---")

        rows = []
        for tier in TIER_ORDER:
            tier_df = sub[sub["tier"] == tier]
            n = len(tier_df)
            if n == 0:
                rows.append({"tier": tier, "count": 0, "pct": "0%", "mean_score": np.nan,
                            "mean_ret": np.nan, "median_ret": np.nan, "win_rate": np.nan,
                            "mean_mdd": np.nan, "mean_manual_rr": np.nan})
                continue

            ret = tier_df["fwd_ret_20"].dropna() if "fwd_ret_20" in tier_df.columns else pd.Series()
            mdd = tier_df["fwd_mdd_20"].dropna() if "fwd_mdd_20" in tier_df.columns else pd.Series()
            manual_rr = tier_df["manual_rr"].dropna() if "manual_rr" in tier_df.columns else pd.Series()

            rows.append({
                "tier": tier,
                "count": n,
                "pct": f"{n/len(sub)*100:.1f}%",
                "mean_score": tier_df["final_score"].mean(),
                "mean_ret": ret.mean() if len(ret) > 0 else np.nan,
                "median_ret": ret.median() if len(ret) > 0 else np.nan,
                "win_rate": (ret > 0).mean() if len(ret) > 0 else np.nan,
                "mean_mdd": mdd.mean() if len(mdd) > 0 else np.nan,
                "mean_manual_rr": manual_rr.mean() if len(manual_rr) > 0 else np.nan,
            })

        result_df = pd.DataFrame(rows)
        display_cols = [c for c in ["tier", "count", "pct", "mean_score", "mean_ret", "median_ret",
                                     "win_rate", "mean_mdd", "mean_manual_rr"]
                        if c in result_df.columns]
        print(tabulate(result_df[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    print(f"\n{'='*100}")
    print("二、实际交易收益 (方案B: 周线低吸+日线止损+持有20日) 对比")
    print(f"{'='*100}")

    for split_name in ["train", "val", "test", "all"]:
        if split_name == "all":
            sub = combined
        else:
            sub = combined[combined["split"] == split_name]
        if sub.empty:
            continue

        print(f"\n--- {split_name} (N={len(sub)}) ---")

        split_metrics = _simulate_plan_b_by_tier(sub, daily_df, hold_days=20)

        rows = []
        for tier in TIER_ORDER:
            m = split_metrics.get(tier, {"count": 0})
            if m["count"] == 0:
                rows.append({"tier": tier, "count": 0, "mean_ret": np.nan, "median_ret": np.nan,
                            "win_rate": np.nan, "stop_rate": np.nan, "profit_loss_ratio": np.nan,
                            "max_loss": np.nan, "max_gain": np.nan})
            else:
                rows.append({
                    "tier": tier,
                    "count": m["count"],
                    "mean_ret": m.get("mean_ret", np.nan),
                    "median_ret": m.get("median_ret", np.nan),
                    "win_rate": m.get("win_rate", np.nan),
                    "stop_rate": m.get("stop_rate", np.nan),
                    "profit_loss_ratio": m.get("profit_loss_ratio", np.nan),
                    "max_loss": m.get("max_loss", np.nan),
                    "max_gain": m.get("max_gain", np.nan),
                })

        result_df = pd.DataFrame(rows)
        display_cols = [c for c in ["tier", "count", "mean_ret", "median_ret", "win_rate",
                                     "stop_rate", "profit_loss_ratio", "max_loss", "max_gain"]
                        if c in result_df.columns]
        print(tabulate(result_df[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    print(f"\n{'='*100}")
    print("三、自然收益 vs 实际交易收益 对比汇总 (all)")
    print(f"{'='*100}")

    compare_rows = []
    for tier in TIER_ORDER:
        tier_df = combined[combined["tier"] == tier]
        n = len(tier_df)
        if n == 0:
            compare_rows.append({"tier": tier, "count": 0,
                                "nat_mean_ret": np.nan, "nat_win_rate": np.nan,
                                "trade_mean_ret": np.nan, "trade_win_rate": np.nan,
                                "trade_stop_rate": np.nan, "trade_plr": np.nan})
            continue

        nat_ret = tier_df["fwd_ret_20"].dropna() if "fwd_ret_20" in tier_df.columns else pd.Series()
        m = tier_trading_metrics.get(tier, {"count": 0})

        compare_rows.append({
            "tier": tier,
            "count": n,
            "nat_mean_ret": nat_ret.mean() if len(nat_ret) > 0 else np.nan,
            "nat_win_rate": (nat_ret > 0).mean() if len(nat_ret) > 0 else np.nan,
            "trade_mean_ret": m.get("mean_ret", np.nan) if m["count"] > 0 else np.nan,
            "trade_win_rate": m.get("win_rate", np.nan) if m["count"] > 0 else np.nan,
            "trade_stop_rate": m.get("stop_rate", np.nan) if m["count"] > 0 else np.nan,
            "trade_plr": m.get("profit_loss_ratio", np.nan) if m["count"] > 0 else np.nan,
        })

    compare_df = pd.DataFrame(compare_rows)
    print(tabulate(compare_df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    print(f"\n--- entry_status 分布 ---")
    if "entry_status" in combined.columns:
        for split_name in ["train", "val", "test", "all"]:
            if split_name == "all":
                sub = combined
            else:
                sub = combined[combined["split"] == split_name]
            if sub.empty:
                continue
            print(f"\n  {split_name}:")
            status_counts = sub["entry_status"].value_counts()
            for status, cnt in status_counts.items():
                mean_ret = sub.loc[sub["entry_status"] == status, "fwd_ret_20"].mean() if "fwd_ret_20" in sub.columns else np.nan
                print(f"    {status}: {cnt} ({cnt/len(sub)*100:.1f}%), mean_ret={mean_ret:.4f}" if pd.notna(mean_ret) else f"    {status}: {cnt} ({cnt/len(sub)*100:.1f}%)")

    out_dir = Path("sr_experiment/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manual_filter_validation_v3.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
