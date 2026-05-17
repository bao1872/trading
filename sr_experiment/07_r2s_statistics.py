# -*- coding: utf-8 -*-
"""
Purpose: 压力转支撑（R2S）专项统计分析
Inputs:  分片 parquet（freq + pivot_len 定位分片目录）或 单股票 db/pytdx 数据
Outputs: results/r2s_statistics.csv / results/r2s_statistics_panel.csv
How to Run:
    python sr_experiment/07_r2s_statistics.py --freq w --pivot-len 10
    python sr_experiment/07_r2s_statistics.py --symbol 300133 --freq w --pivot-len 10
    python sr_experiment/07_r2s_statistics.py --symbol 300133 --freq w --data-source db
Examples:
    python sr_experiment/07_r2s_statistics.py --freq w --pivot-len 10
    python sr_experiment/07_r2s_statistics.py --symbol 300133 --freq w --pivot-len 10 --fetch-bars 1200
    python sr_experiment/07_r2s_statistics.py --symbol 300133 --freq w --data-source db
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


def _sub_stats_shard(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = df.loc[mask]
    n = len(sub)
    row = {"section": label, "count": n}
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


def _compute_r2s_stats_shard(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def _col_mask(col: str) -> pd.Series:
        if col in df.columns:
            return df[col].fillna(False).astype(bool)
        return pd.Series(False, index=df.index)

    retest = _col_mask("evt_retest_flipped_support")
    flipped_reclaim = _col_mask("evt_pierce_flipped_support_reclaim")
    pivot_reclaim = _col_mask("evt_pierce_pivot_support_reclaim")
    breakdown = _col_mask("evt_breakdown_flipped_support")
    vol_expansion = _col_mask("is_volume_expansion")
    vol_shrink = _col_mask("is_volume_shrink")

    rows.append(_sub_stats_shard(df, retest, "R2S回踩"))
    rows.append(_sub_stats_shard(df, flipped_reclaim, "R2S刺破收回"))
    rows.append(_sub_stats_shard(df, pivot_reclaim, "Pivot刺破收回"))
    rows.append(_sub_stats_shard(df, breakdown, "R2S失败(跌破)"))

    age_col = "flipped_support_age_bars"
    if age_col in df.columns:
        retest_with_age = df.loc[retest & df[age_col].notna()].copy()
        if not retest_with_age.empty:
            retest_with_age["_age_group"] = pd.cut(
                retest_with_age[age_col],
                bins=[-np.inf, 5, 20, np.inf],
                labels=["新<=5", "中5-20", "老>20"],
            )
            for grp_val, sub in retest_with_age.groupby("_age_group", dropna=True):
                row = {"section": f"R2S新鲜度_{grp_val}", "count": len(sub)}
                for col in FWD_RET_COLS:
                    if col in sub.columns:
                        vals = sub[col].dropna()
                        row[f"{col}_sum"] = vals.sum() if len(vals) > 0 else 0.0
                        row[f"{col}_valid_count"] = len(vals)
                        row[f"{col}_win_count"] = int((vals > 0).sum()) if len(vals) > 0 else 0
                rows.append(row)

    rows.append(_sub_stats_shard(df, retest & vol_expansion, "R2S回踩+放量"))
    rows.append(_sub_stats_shard(df, retest & vol_shrink, "R2S回踩+缩量"))
    rows.append(_sub_stats_shard(df, flipped_reclaim & vol_expansion, "R2S刺破收回+放量"))
    rows.append(_sub_stats_shard(df, flipped_reclaim & vol_shrink, "R2S刺破收回+缩量"))

    return pd.DataFrame(rows)


def _merge_r2s_stats(shard_stats: list[pd.DataFrame]) -> pd.DataFrame:
    if not shard_stats:
        return pd.DataFrame()

    all_stats = pd.concat(shard_stats, ignore_index=True)
    result_rows = []

    for section, grp in all_stats.groupby("section"):
        row = {"section": section, "count": grp["count"].sum()}
        for col in FWD_RET_COLS:
            s = grp[f"{col}_sum"].sum() if f"{col}_sum" in grp.columns else 0.0
            c = grp[f"{col}_valid_count"].sum() if f"{col}_valid_count" in grp.columns else 0
            wc = grp[f"{col}_win_count"].sum() if f"{col}_win_count" in grp.columns else 0
            row[f"{col}_mean"] = s / c if c > 0 else np.nan
            row[f"{col}_win_rate"] = wc / c if c > 0 else np.nan
        result_rows.append(row)

    return pd.DataFrame(result_rows)


def _sub_stats(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = df.loc[mask]
    n = len(sub)
    row = {"section": label, "count": n}
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


def compute_r2s_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def _col_mask(col: str) -> pd.Series:
        if col in df.columns:
            return df[col].fillna(False).astype(bool)
        return pd.Series(False, index=df.index)

    retest = _col_mask("evt_retest_flipped_support")
    flipped_reclaim = _col_mask("evt_pierce_flipped_support_reclaim")
    pivot_reclaim = _col_mask("evt_pierce_pivot_support_reclaim")
    breakdown = _col_mask("evt_breakdown_flipped_support")
    vol_expansion = _col_mask("is_volume_expansion")
    vol_shrink = _col_mask("is_volume_shrink")

    rows.append(_sub_stats(df, retest, "R2S回踩"))

    r2s_sub = df.loc[retest] if retest.any() else df.iloc[0:0]
    pivot_sub = df.loc[pivot_reclaim] if pivot_reclaim.any() else df.iloc[0:0]
    rows.append(_sub_stats(df, flipped_reclaim, "R2S刺破收回"))
    rows.append(_sub_stats(df, pivot_reclaim, "Pivot刺破收回"))

    rows.append(_sub_stats(df, breakdown, "R2S失败(跌破)"))

    age_col = "flipped_support_age_bars"
    if age_col in df.columns:
        retest_with_age = df.loc[retest & df[age_col].notna()].copy()
        if not retest_with_age.empty:
            retest_with_age["_age_group"] = pd.cut(
                retest_with_age[age_col],
                bins=[-np.inf, 5, 20, np.inf],
                labels=["新<=5", "中5-20", "老>20"],
            )
            for grp_val, sub in retest_with_age.groupby("_age_group", dropna=True):
                row = {"section": f"R2S新鲜度_{grp_val}", "count": len(sub)}
                for col in FWD_RET_COLS:
                    if col in sub.columns:
                        vals = sub[col].dropna()
                        row[f"{col}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                        row[f"{col}_win_rate"] = (vals > 0).mean() if len(vals) > 0 else np.nan
                rows.append(row)

    rows.append(_sub_stats(df, retest & vol_expansion, "R2S回踩+放量"))
    rows.append(_sub_stats(df, retest & vol_shrink, "R2S回踩+缩量"))
    rows.append(_sub_stats(df, flipped_reclaim & vol_expansion, "R2S刺破收回+放量"))
    rows.append(_sub_stats(df, flipped_reclaim & vol_shrink, "R2S刺破收回+缩量"))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="压力转支撑（R2S）专项统计分析")
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
        if "bar_time" not in cols_to_load:
            cols_to_load.append("bar_time")
        shard_stats = []
        for shard_df in iter_shards(args.freq, args.pivot_len, columns=cols_to_load, shard_type="panel"):
            stats = _compute_r2s_stats_shard(shard_df)
            if not stats.empty:
                shard_stats.append(stats)
            del shard_df
        stats = _merge_r2s_stats(shard_stats)
    else:
        symbol = args.symbol
        if args.data_source == "db":
            raw = load_kline_db(symbol, args.freq)
        else:
            fetch_count = max(args.fetch_bars, 2 * args.pivot_len + 200)
            raw = fetch_kline_pytdx(symbol, args.freq, fetch_count)
        cfg = LabConfig(pivot_len=args.pivot_len)
        out = compute_sr_factor_lab(raw, cfg)
        stats = compute_r2s_statistics(out)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_panel" if use_shards else ""
    csv_path = out_dir / f"r2s_statistics{suffix}.csv"
    stats.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(tabulate(stats, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
