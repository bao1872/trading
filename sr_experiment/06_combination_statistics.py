# -*- coding: utf-8 -*-
"""
Purpose: 组合事件统计分析（刺破收回+形态/量能、低位突破+量能、影线突破失败、刺破失败等）
Inputs:  分片 parquet（freq + pivot_len 定位分片目录）或 单股票 db/pytdx 数据
Outputs: results/combination_statistics.csv / results/combination_statistics_panel.csv
How to Run:
    python sr_experiment/06_combination_statistics.py --freq w --pivot-len 10
    python sr_experiment/06_combination_statistics.py --symbol 300133 --freq w --pivot-len 10
    python sr_experiment/06_combination_statistics.py --symbol 300133 --freq w --data-source db
Examples:
    python sr_experiment/06_combination_statistics.py --freq w --pivot-len 10
    python sr_experiment/06_combination_statistics.py --symbol 300133 --freq w --pivot-len 10 --fetch-bars 1200
    python sr_experiment/06_combination_statistics.py --symbol 300133 --freq w --data-source db
Side Effects: 写 CSV 文件到 sr_experiment/results/ 目录
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

from features.sr_event_factor_lab import (
    LabConfig,
    compute_sr_factor_lab,
    fetch_kline_pytdx,
)
from sr_experiment.db_adapter import _code_to_ts_code, load_kline as load_kline_db
from sr_experiment.sr_config import FWD_RET_COLS, OUT_DIR, PANEL_GROUP_STAT_COLS

_build_mod = importlib.import_module("sr_experiment.00_build_factor_panel_from_db")
iter_shards = _build_mod.iter_shards

_COMBO_DEFS = [
    ("刺破收回+长下影", ["evt_pierce_support_reclaim", "is_long_lower_shadow"]),
    ("刺破收回+收盘上半部", ["evt_pierce_support_reclaim", "is_close_upper_half"]),
    ("刺破收回+支撑抬高", ["evt_pierce_support_reclaim", "support_is_higher_low"]),
    ("刺破收回+放量", ["evt_pierce_support_reclaim", "is_volume_expansion"]),
    ("刺破收回+缩量", ["evt_pierce_support_reclaim", "is_volume_shrink"]),
    ("低位突破+放量", ["evt_break_resistance_from_low_zone", "is_volume_expansion"]),
    ("低位突破+收盘强势", ["evt_break_resistance_from_low_zone", "evt_break_resistance_close_strong"]),
    ("影线突破失败", ["evt_wick_break_resistance_fail"]),
    ("刺破失败", ["evt_failed_reclaim_support"]),
]


def _combo_stats_shard(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = df.loc[mask]
    n = len(sub)
    row = {"combination": label, "count": n}
    if n == 0:
        for col in FWD_RET_COLS:
            row[f"{col}_sum"] = 0.0
            row[f"{col}_valid_count"] = 0
            row[f"{col}_win_count"] = 0
        return row
    for col in FWD_RET_COLS:
        if col in sub.columns:
            vals = sub[col].dropna()
            row[f"{col}_sum"] = vals.sum() if len(vals) > 0 else 0.0
            row[f"{col}_valid_count"] = len(vals)
            row[f"{col}_win_count"] = int((vals > 0).sum()) if len(vals) > 0 else 0
    return row


def _compute_combination_stats_shard(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def _col_mask(col: str) -> pd.Series:
        if col in df.columns:
            return df[col].fillna(False).astype(bool)
        return pd.Series(False, index=df.index)

    for label, cols in _COMBO_DEFS:
        mask = _col_mask(cols[0])
        for c in cols[1:]:
            mask = mask & _col_mask(c)
        rows.append(_combo_stats_shard(df, mask, label))

    return pd.DataFrame(rows)


def _merge_combination_stats(shard_stats: list[pd.DataFrame]) -> pd.DataFrame:
    if not shard_stats:
        return pd.DataFrame()

    all_stats = pd.concat(shard_stats, ignore_index=True)
    result_rows = []

    for combo, grp in all_stats.groupby("combination"):
        row = {"combination": combo, "count": grp["count"].sum()}
        for col in FWD_RET_COLS:
            s = grp[f"{col}_sum"].sum() if f"{col}_sum" in grp.columns else 0.0
            c = grp[f"{col}_valid_count"].sum() if f"{col}_valid_count" in grp.columns else 0
            wc = grp[f"{col}_win_count"].sum() if f"{col}_win_count" in grp.columns else 0
            row[f"{col}_mean"] = s / c if c > 0 else np.nan
            row[f"{col}_win_rate"] = wc / c if c > 0 else np.nan
        result_rows.append(row)

    return pd.DataFrame(result_rows)


def _combo_stats(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = df.loc[mask]
    n = len(sub)
    row = {"combination": label, "count": n}
    if n == 0:
        for col in FWD_RET_COLS:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_win_rate"] = np.nan
        return row
    for col in FWD_RET_COLS:
        if col in sub.columns:
            vals = sub[col].dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) > 0 else np.nan
            row[f"{col}_win_rate"] = (vals > 0).mean() if len(vals) > 0 else np.nan
    return row


def compute_combination_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def _col_mask(col: str) -> pd.Series:
        if col in df.columns:
            return df[col].fillna(False).astype(bool)
        return pd.Series(False, index=df.index)

    pierce_reclaim = _col_mask("evt_pierce_support_reclaim")
    long_lower = _col_mask("is_long_lower_shadow")
    close_upper_half = _col_mask("is_close_upper_half")
    support_is_higher = _col_mask("support_is_higher_low")
    vol_expansion = _col_mask("is_volume_expansion")
    vol_shrink = _col_mask("is_volume_shrink")
    low_break = _col_mask("evt_break_resistance_from_low_zone")
    close_strong = _col_mask("evt_break_resistance_close_strong")
    wick_fail = _col_mask("evt_wick_break_resistance_fail")
    pierce_fail = _col_mask("evt_failed_reclaim_support")

    rows.append(_combo_stats(df, pierce_reclaim & long_lower, "刺破收回+长下影"))
    rows.append(_combo_stats(df, pierce_reclaim & close_upper_half, "刺破收回+收盘上半部"))
    rows.append(_combo_stats(df, pierce_reclaim & support_is_higher, "刺破收回+支撑抬高"))
    rows.append(_combo_stats(df, pierce_reclaim & vol_expansion, "刺破收回+放量"))
    rows.append(_combo_stats(df, pierce_reclaim & vol_shrink, "刺破收回+缩量"))
    rows.append(_combo_stats(df, low_break & vol_expansion, "低位突破+放量"))
    rows.append(_combo_stats(df, low_break & close_strong, "低位突破+收盘强势"))
    rows.append(_combo_stats(df, wick_fail, "影线突破失败"))
    rows.append(_combo_stats(df, pierce_fail, "刺破失败"))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="组合事件统计分析")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--freq", default="w")
    parser.add_argument("--pivot-len", type=int, default=10)
    parser.add_argument("--fetch-bars", type=int, default=1200)
    parser.add_argument("--data-source", type=str, default="db", choices=["db", "pytdx"])
    args = parser.parse_args()

    use_shards = args.symbol is None

    if use_shards:
        cols_to_load = [c for c in PANEL_GROUP_STAT_COLS if c != "ts_code"]
        if "ts_code" not in cols_to_load:
            cols_to_load.append("ts_code")
        for extra in ["bar_time", "is_close_upper_half", "evt_break_resistance_close_strong"]:
            if extra not in cols_to_load:
                cols_to_load.append(extra)
        shard_stats = []
        for shard_df in iter_shards(args.freq, args.pivot_len, columns=cols_to_load, shard_type="panel"):
            stats = _compute_combination_stats_shard(shard_df)
            if not stats.empty:
                shard_stats.append(stats)
            del shard_df
        stats = _merge_combination_stats(shard_stats)
    else:
        symbol = args.symbol
        if args.data_source == "db":
            raw = load_kline_db(symbol, args.freq)
        else:
            fetch_count = max(args.fetch_bars, 2 * args.pivot_len + 200)
            raw = fetch_kline_pytdx(symbol, args.freq, fetch_count)
        cfg = LabConfig(pivot_len=args.pivot_len)
        out = compute_sr_factor_lab(raw, cfg)
        stats = compute_combination_statistics(out)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_panel" if use_shards else ""
    csv_path = out_dir / f"combination_statistics{suffix}.csv"
    stats.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(tabulate(stats, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
