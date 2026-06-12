# -*- coding: utf-8 -*-
"""
Purpose: 日线独立场景验证——完全不看周线，验证日线自身的R2S回踩和动量回踩是否有效
Inputs:  日线 events 分片 (d_pv20)
Outputs: 控制台三张对比表 + daily_standalone_validation.csv
How to Run:
    python sr_experiment/17_daily_standalone_scenario_validation.py
Examples:
    python sr_experiment/17_daily_standalone_scenario_validation.py
Side Effects: 写入 sr_experiment/results/daily_standalone_validation.csv

注意：本脚本不使用任何周线字段（weekly_low, W0/W1/W2, weekly risk_score 等），
      只使用日线自身的支撑/压力/簇/量能/趋势/前向标签。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sr_experiment.gbdt_config import TRAIN_END, VAL_END

_build_mod = importlib.import_module("sr_experiment.00_build_factor_panel_from_db")
iter_shards = _build_mod.iter_shards

COLS_TO_LOAD = [
    "ts_code", "bar_time", "open", "close", "low", "high",
    "evt_retest_flipped_support", "evt_clean_hold_flipped_support",
    "evt_pierce_flipped_support_reclaim", "evt_breakdown_flipped_support",
    "evt_pierce_support_cluster_reclaim_low_volume",
    "evt_pierce_support_cluster_reclaim_high_volume",
    "is_support_flipped", "flipped_support_ref", "flipped_support_age_bars",
    "flipped_support_reclaim_strength_atr",
    "active_support_ref", "resistance_ref", "resistance_active",
    "support_cluster_is_strong", "support_cluster_score",
    "support_confluence_score", "support_confluence_is_strong",
    "is_volume_shrink", "is_volume_expansion", "volume_z_20",
    "close_pos_in_bar", "close_above_ma20",
    "close_to_support_pct", "upside_to_resistance_pct",
    "is_shallow_support_pierce", "is_deep_support_pierce",
    "fwd_ret_3", "fwd_ret_5", "fwd_ret_10", "fwd_ret_20",
    "fwd_mdd_3", "fwd_mdd_5", "fwd_mdd_10", "fwd_mdd_20",
    "fwd_max_ret_3", "fwd_max_ret_5", "fwd_max_ret_10", "fwd_max_ret_20",
    "fwd_reward_risk_3", "fwd_reward_risk_5", "fwd_reward_risk_10", "fwd_reward_risk_20",
    "fwd_win_10_3pct", "fwd_win_10_5pct",
    "fwd_win_5_3pct", "fwd_win_5_5pct",
    "ret_5", "ret_10",
]


def _load_daily_events(pivot_len: int = 20) -> pd.DataFrame:
    print(f"加载日线 events (d_pv{pivot_len})...")
    parts = []
    for shard_df in iter_shards("d", pivot_len, columns=COLS_TO_LOAD, shard_type="event"):
        parts.append(shard_df)
    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    if "bar_time" in df.columns:
        df["bar_time"] = pd.to_datetime(df["bar_time"])
    print(f"  日线总事件行数: {len(df)}")
    return df


def _group_stats(df: pd.DataFrame, group_name: str) -> dict:
    if df.empty:
        return {"group": group_name, "count": 0}
    ret_cols = {}
    for h in [3, 5, 10, 20]:
        col = f"fwd_ret_{h}"
        ret_cols[h] = df[col].dropna() if col in df.columns else pd.Series()

    mdd_10 = df["fwd_mdd_10"].dropna() if "fwd_mdd_10" in df.columns else pd.Series()
    rr_10 = df["fwd_reward_risk_10"].dropna() if "fwd_reward_risk_10" in df.columns else pd.Series()
    max_ret_10 = df["fwd_max_ret_10"].dropna() if "fwd_max_ret_10" in df.columns else pd.Series()
    win_10_3pct = df["fwd_win_10_3pct"].mean() if "fwd_win_10_3pct" in df.columns else np.nan
    win_5_3pct = df["fwd_win_5_3pct"].mean() if "fwd_win_5_3pct" in df.columns else np.nan

    result = {"group": group_name, "count": len(df)}
    for h, s in ret_cols.items():
        result[f"ret_{h}"] = s.mean() if len(s) > 0 else np.nan
    result["win_5"] = win_5_3pct
    result["win_10"] = win_10_3pct
    result["mdd_10"] = mdd_10.mean() if len(mdd_10) > 0 else np.nan
    result["rr_10"] = rr_10.mean() if len(rr_10) > 0 else np.nan
    result["max_ret_10"] = max_ret_10.mean() if len(max_ret_10) > 0 else np.nan
    return result


def table1_r2s(df: pd.DataFrame) -> pd.DataFrame:
    print("\n表1: 日线 R2S 刺破收回场景表")
    print("=" * 100)

    pierce_reclaim = df[df["evt_pierce_flipped_support_reclaim"].fillna(False).astype(bool)].copy()
    retest = df[df["evt_retest_flipped_support"].fillna(False).astype(bool)].copy()
    breakdown = df[df["evt_breakdown_flipped_support"].fillna(False).astype(bool)].copy()

    groups = [
        ("R2S刺破收回全部(基线)", pierce_reclaim),
        ("R2S刺破收回+缩量", pierce_reclaim[pierce_reclaim["is_volume_shrink"].fillna(False).astype(bool)]),
        ("R2S刺破收回+放量", pierce_reclaim[pierce_reclaim["is_volume_expansion"].fillna(False).astype(bool)]),
        ("R2S刺破收回+老R2S(>20bar)", pierce_reclaim[pierce_reclaim["flipped_support_age_bars"].fillna(0) > 20]),
        ("R2S刺破收回+新R2S(<=20bar)", pierce_reclaim[pierce_reclaim["flipped_support_age_bars"].fillna(999) <= 20]),
        ("R2S刺破收回+浅刺破", pierce_reclaim[pierce_reclaim["is_shallow_support_pierce"].fillna(False).astype(bool)] if "is_shallow_support_pierce" in pierce_reclaim.columns else pd.DataFrame()),
        ("R2S刺破收回+深刺破", pierce_reclaim[pierce_reclaim["is_deep_support_pierce"].fillna(False).astype(bool)] if "is_deep_support_pierce" in pierce_reclaim.columns else pd.DataFrame()),
        ("[对照]R2S回踩不破", retest),
        ("[对照]R2S回踩不破+缩量", retest[retest["is_volume_shrink"].fillna(False).astype(bool)]),
        ("[风险]R2S跌破", breakdown),
    ]

    rows = [_group_stats(sub, name) for name, sub in groups]
    result = pd.DataFrame(rows)
    display_cols = [c for c in ["group", "count", "ret_3", "ret_5", "ret_10", "ret_20",
                                 "win_5", "win_10", "mdd_10", "rr_10", "max_ret_10"]
                    if c in result.columns]
    print(tabulate(result[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    return result


def table2_momentum(df: pd.DataFrame) -> pd.DataFrame:
    print("\n表2: 日线动量回踩表")
    print("=" * 100)

    base = df[df["evt_pierce_support_cluster_reclaim_low_volume"].fillna(False).astype(bool)].copy()
    ret5 = base["ret_5"] if "ret_5" in base.columns else pd.Series(dtype=float)
    momentum = base[ret5 > 0.03].copy() if "ret_5" in base.columns else base.copy()
    print(f"  强簇缩量收回事件: {len(base)}, 已反弹>3%: {len(momentum)}")

    groups = [
        ("动量回踩全部(基线)", momentum),
        ("回踩不破active_support", momentum[momentum["close"] > momentum["active_support_ref"]] if "active_support_ref" in momentum.columns and "close" in momentum.columns else pd.DataFrame()),
        ("回踩强支撑簇", momentum[momentum["support_cluster_is_strong"].fillna(False).astype(bool)]),
        ("回踩缩量", momentum[momentum["is_volume_shrink"].fillna(False).astype(bool)]),
        ("回踩后收盘上半部", momentum[momentum["close_pos_in_bar"].fillna(0) >= 0.6]),
        ("回踩后站上MA20", momentum[momentum["close_above_ma20"].fillna(False).astype(bool)] if "close_above_ma20" in momentum.columns else pd.DataFrame()),
    ]

    rows = [_group_stats(sub, name) for name, sub in groups]
    result = pd.DataFrame(rows)
    display_cols = [c for c in ["group", "count", "ret_3", "ret_5", "ret_10", "ret_20",
                                 "win_5", "win_10", "mdd_10", "rr_10", "max_ret_10"]
                    if c in result.columns]
    print(tabulate(result[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    return result


def table3_risk(df: pd.DataFrame) -> pd.DataFrame:
    print("\n表3: 日线风险剔除表")
    print("=" * 100)

    below_support = df[df["close"] < df["active_support_ref"]] if "active_support_ref" in df.columns and "close" in df.columns else pd.DataFrame()
    vol_break_cluster = df[
        df["evt_pierce_support_cluster_reclaim_high_volume"].fillna(False).astype(bool)
    ] if "evt_pierce_support_cluster_reclaim_high_volume" in df.columns else pd.DataFrame()
    r2s_breakdown = df[df["evt_breakdown_flipped_support"].fillna(False).astype(bool)]
    high_vol_drop = df[
        (df["close_pos_in_bar"].fillna(1) < 0.3) & (df["is_volume_expansion"].fillna(False).astype(bool))
    ]

    groups = [
        ("跌破active_support", below_support),
        ("放量刺破强支撑簇", vol_break_cluster),
        ("R2S breakdown", r2s_breakdown),
        ("高位放量回落", high_vol_drop),
    ]

    rows = [_group_stats(sub, name) for name, sub in groups]
    result = pd.DataFrame(rows)
    display_cols = [c for c in ["group", "count", "ret_3", "ret_5", "ret_10", "ret_20", "mdd_10"]
                    if c in result.columns]
    print(tabulate(result[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    return result


def _yearly_stats(df: pd.DataFrame, event_col: str, split_col: str) -> pd.DataFrame:
    if event_col not in df.columns or split_col not in df.columns:
        return pd.DataFrame()

    event_df = df[df[event_col].fillna(False).astype(bool)]
    if event_df.empty:
        return pd.DataFrame()

    rows = []
    for year in sorted(event_df["year"].unique()):
        sub = event_df[event_df["year"] == year]
        ret_10 = sub["fwd_ret_10"].mean() if "fwd_ret_10" in sub.columns else np.nan
        win_10 = sub["fwd_win_10_3pct"].mean() if "fwd_win_10_3pct" in sub.columns else np.nan
        mdd_10 = sub["fwd_mdd_10"].mean() if "fwd_mdd_10" in sub.columns else np.nan
        rows.append({"year": year, "count": len(sub), "ret_10": ret_10, "win_10": win_10, "mdd_10": mdd_10})

    return pd.DataFrame(rows)


def main():
    df = _load_daily_events(pivot_len=20)
    if df.empty:
        print("无日线数据")
        return

    if "bar_time" in df.columns:
        df["year"] = df["bar_time"].dt.year
        df["split"] = "train"
        df.loc[df["bar_time"] > TRAIN_END, "split"] = "val"
        df.loc[df["bar_time"] > VAL_END, "split"] = "test"

    print(f"\n{'='*100}")
    print("日线独立场景验证报告 (d_pv20, pivot_len=20)")
    print("注意：本报告不使用任何周线字段")
    print(f"{'='*100}")

    r2s_result = table1_r2s(df)
    momentum_result = table2_momentum(df)
    risk_result = table3_risk(df)

    print(f"\n{'='*100}")
    print("按split统计")
    print(f"{'='*100}")

    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        if sub.empty:
            continue
        print(f"\n--- {split_name} (N={len(sub)}) ---")

        retest = sub[sub["evt_retest_flipped_support"].fillna(False).astype(bool)]
        retest_shrink = retest[retest["is_volume_shrink"].fillna(False).astype(bool)]
        retest_expand = retest[retest["is_volume_expansion"].fillna(False).astype(bool)]

        pierce_reclaim = sub[sub["evt_pierce_flipped_support_reclaim"].fillna(False).astype(bool)]
        pierce_shrink = pierce_reclaim[pierce_reclaim["is_volume_shrink"].fillna(False).astype(bool)]
        pierce_expand = pierce_reclaim[pierce_reclaim["is_volume_expansion"].fillna(False).astype(bool)]

        rows = [
            _group_stats(pierce_reclaim, "R2S刺破收回全部"),
            _group_stats(pierce_shrink, "R2S刺破收回+缩量"),
            _group_stats(pierce_expand, "R2S刺破收回+放量"),
            _group_stats(retest, "[对照]R2S回踩不破"),
            _group_stats(retest_shrink, "[对照]R2S回踩不破+缩量"),
        ]
        result = pd.DataFrame(rows)
        display_cols = [c for c in ["group", "count", "ret_5", "ret_10", "ret_20", "win_10", "mdd_10", "rr_10"]
                        if c in result.columns]
        print(tabulate(result[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    print(f"\n{'='*100}")
    print("R2S刺破收回+缩量 按年统计")
    print(f"{'='*100}")

    pierce_shrink_yearly = _yearly_stats(
        df[df["is_volume_shrink"].fillna(False).astype(bool)],
        "evt_pierce_flipped_support_reclaim",
        "year"
    )
    if not pierce_shrink_yearly.empty:
        print(tabulate(pierce_shrink_yearly, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    print(f"\n{'='*100}")
    print("动量回踩 按年统计")
    print(f"{'='*100}")

    base = df[df["evt_pierce_support_cluster_reclaim_low_volume"].fillna(False).astype(bool)]
    if "ret_5" in base.columns:
        momentum = base[base["ret_5"] > 0.03]
    else:
        momentum = base
    momentum_yearly = _yearly_stats(momentum, "evt_pierce_support_cluster_reclaim_low_volume", "year")
    if not momentum_yearly.empty:
        print(tabulate(momentum_yearly, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

    out_dir = Path("sr_experiment/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "daily_standalone_validation.csv"
    df.to_csv(out_path, index=False)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
