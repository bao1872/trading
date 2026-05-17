# -*- coding: utf-8 -*-
"""
Purpose: 分组统计分析（位置/刺破深度/收回强度/量能/趋势/支撑类型分组）
Inputs:  分片 parquet（freq + pivot_len 定位分片目录）或 单股票 db/pytdx 数据
Outputs: results/grouped_statistics.csv / results/grouped_statistics_panel.csv
How to Run:
    python sr_experiment/05_grouped_statistics.py --freq w --pivot-len 10
    python sr_experiment/05_grouped_statistics.py --symbol 300133 --freq w --pivot-len 10
    python sr_experiment/05_grouped_statistics.py --symbol 300133 --freq w --data-source db
Examples:
    python sr_experiment/05_grouped_statistics.py --freq w --pivot-len 10
    python sr_experiment/05_grouped_statistics.py --symbol 300133 --freq w --pivot-len 10 --fetch-bars 1200
    python sr_experiment/05_grouped_statistics.py --symbol 300133 --freq w --data-source db
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
from sr_experiment.sr_config import (
    EVENT_COLS,
    FWD_RET_COLS,
    OUT_DIR,
    PANEL_GROUP_STAT_COLS,
)

_build_mod = importlib.import_module("sr_experiment.00_build_factor_panel_from_db")
iter_shards = _build_mod.iter_shards


def _group_stats(df: pd.DataFrame, group_col: str, group_label: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    valid = df[df[group_col].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    rows = []
    for grp_val, sub in valid.groupby(group_col, dropna=True):
        row = {"group_type": group_label, "group_value": grp_val, "count": len(sub)}
        for col in FWD_RET_COLS:
            if col in sub.columns:
                vals = sub[col].dropna()
                row[f"{col}_sum"] = vals.sum() if len(vals) > 0 else 0.0
                row[f"{col}_valid_count"] = len(vals)
                row[f"{col}_win_count"] = int((vals > 0).sum()) if len(vals) > 0 else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_grouped_stats_shard(df: pd.DataFrame) -> pd.DataFrame:
    """对单个分片计算分组统计，返回可合并的中间结果。"""
    parts = []

    pos_col = "sr_pos_01"
    if pos_col in df.columns:
        df["_pos_group"] = pd.cut(
            df[pos_col],
            bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
            labels=["0-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"],
        )
        parts.append(_group_stats(df, "_pos_group", "sr_pos_01"))

    depth_col = "support_pierce_depth_atr"
    if depth_col in df.columns:
        pierce_mask = df["evt_pierce_support_reclaim"].fillna(False).astype(bool) if "evt_pierce_support_reclaim" in df.columns else pd.Series(False, index=df.index)
        sub = df.loc[pierce_mask].copy()
        if not sub.empty:
            sub["_depth_group"] = pd.cut(
                sub[depth_col],
                bins=[-np.inf, 0.3, 0.8, np.inf],
                labels=["0-0.3", "0.3-0.8", "0.8+"],
            )
            parts.append(_group_stats(sub, "_depth_group", "support_pierce_depth_atr"))

    reclaim_col = "support_reclaim_strength_atr"
    if reclaim_col in df.columns:
        reclaim_mask = df["evt_pierce_support_reclaim"].fillna(False).astype(bool) if "evt_pierce_support_reclaim" in df.columns else pd.Series(False, index=df.index)
        sub = df.loc[reclaim_mask].copy()
        if not sub.empty:
            sub["_reclaim_group"] = pd.cut(
                sub[reclaim_col],
                bins=[-np.inf, 0.2, 0.5, np.inf],
                labels=["0-0.2", "0.2-0.5", "0.5+"],
            )
            parts.append(_group_stats(sub, "_reclaim_group", "support_reclaim_strength_atr"))

    vol_col = "volume_z_20"
    if vol_col in df.columns:
        df["_vol_group"] = pd.cut(
            df[vol_col],
            bins=[-np.inf, -1, 1, 2, np.inf],
            labels=["<-1", "-1~1", "1~2", ">2"],
        )
        parts.append(_group_stats(df, "_vol_group", "volume_z_20"))

    trend_col = "_trend_group"
    df[trend_col] = "neutral"
    if "trend_ma_bull" in df.columns:
        df.loc[df["trend_ma_bull"].fillna(False).astype(bool), trend_col] = "bull"
    if "trend_ma_bear" in df.columns:
        df.loc[df["trend_ma_bear"].fillna(False).astype(bool), trend_col] = "bear"
    parts.append(_group_stats(df, trend_col, "trend"))

    support_type_col = "_support_type"
    df[support_type_col] = "pivot"
    if "is_support_flipped" in df.columns:
        df.loc[df["is_support_flipped"].fillna(False).astype(bool), support_type_col] = "flipped"
    if "evt_pierce_active_support_reclaim" in df.columns:
        active_mask = df["evt_pierce_active_support_reclaim"].fillna(False).astype(bool)
        df.loc[active_mask, support_type_col] = "active"
    parts.append(_group_stats(df, support_type_col, "support_type"))

    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame()


def _merge_grouped_stats(shard_stats: list[pd.DataFrame]) -> pd.DataFrame:
    """合并多个分片的分组统计结果，按 (group_type, group_value) 加权平均。"""
    if not shard_stats:
        return pd.DataFrame()

    all_stats = pd.concat(shard_stats, ignore_index=True)
    result_rows = []

    for (gtype, gval), grp in all_stats.groupby(["group_type", "group_value"], dropna=False):
        row = {
            "group_type": gtype,
            "group_value": gval,
            "count": grp["count"].sum(),
        }
        for col in FWD_RET_COLS:
            s = grp[f"{col}_sum"].sum() if f"{col}_sum" in grp.columns else 0.0
            c = grp[f"{col}_valid_count"].sum() if f"{col}_valid_count" in grp.columns else 0
            wc = grp[f"{col}_win_count"].sum() if f"{col}_win_count" in grp.columns else 0
            row[f"{col}_mean"] = s / c if c > 0 else np.nan
            row[f"{col}_win_rate"] = wc / c if c > 0 else np.nan
        result_rows.append(row)

    return pd.DataFrame(result_rows)


def compute_grouped_statistics(df: pd.DataFrame) -> pd.DataFrame:
    parts = []

    pos_col = "sr_pos_01"
    if pos_col in df.columns:
        df["_pos_group"] = pd.cut(
            df[pos_col],
            bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
            labels=["0-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"],
        )
        parts.append(_group_stats(df, "_pos_group", "sr_pos_01"))

    depth_col = "support_pierce_depth_atr"
    if depth_col in df.columns:
        pierce_mask = df["evt_pierce_support_reclaim"].fillna(False).astype(bool) if "evt_pierce_support_reclaim" in df.columns else pd.Series(False, index=df.index)
        sub = df.loc[pierce_mask].copy()
        if not sub.empty:
            sub["_depth_group"] = pd.cut(
                sub[depth_col],
                bins=[-np.inf, 0.3, 0.8, np.inf],
                labels=["0-0.3", "0.3-0.8", "0.8+"],
            )
            parts.append(_group_stats(sub, "_depth_group", "support_pierce_depth_atr"))

    reclaim_col = "support_reclaim_strength_atr"
    if reclaim_col in df.columns:
        reclaim_mask = df["evt_pierce_support_reclaim"].fillna(False).astype(bool) if "evt_pierce_support_reclaim" in df.columns else pd.Series(False, index=df.index)
        sub = df.loc[reclaim_mask].copy()
        if not sub.empty:
            sub["_reclaim_group"] = pd.cut(
                sub[reclaim_col],
                bins=[-np.inf, 0.2, 0.5, np.inf],
                labels=["0-0.2", "0.2-0.5", "0.5+"],
            )
            parts.append(_group_stats(sub, "_reclaim_group", "support_reclaim_strength_atr"))

    vol_col = "volume_z_20"
    if vol_col in df.columns:
        df["_vol_group"] = pd.cut(
            df[vol_col],
            bins=[-np.inf, -1, 1, 2, np.inf],
            labels=["<-1", "-1~1", "1~2", ">2"],
        )
        parts.append(_group_stats(df, "_vol_group", "volume_z_20"))

    trend_col = "_trend_group"
    df[trend_col] = "neutral"
    if "trend_ma_bull" in df.columns:
        df.loc[df["trend_ma_bull"].fillna(False).astype(bool), trend_col] = "bull"
    if "trend_ma_bear" in df.columns:
        df.loc[df["trend_ma_bear"].fillna(False).astype(bool), trend_col] = "bear"
    parts.append(_group_stats(df, trend_col, "trend"))

    support_type_col = "_support_type"
    df[support_type_col] = "pivot"
    if "is_support_flipped" in df.columns:
        df.loc[df["is_support_flipped"].fillna(False).astype(bool), support_type_col] = "flipped"
    if "evt_pierce_active_support_reclaim" in df.columns:
        active_mask = df["evt_pierce_active_support_reclaim"].fillna(False).astype(bool)
        df.loc[active_mask, support_type_col] = "active"
    parts.append(_group_stats(df, support_type_col, "support_type"))

    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="分组统计分析")
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
            stats = _compute_grouped_stats_shard(shard_df)
            if not stats.empty:
                shard_stats.append(stats)
            del shard_df
        stats = _merge_grouped_stats(shard_stats)
    else:
        symbol = args.symbol
        if args.data_source == "db":
            raw = load_kline_db(symbol, args.freq)
        else:
            fetch_count = max(args.fetch_bars, 2 * args.pivot_len + 200)
            raw = fetch_kline_pytdx(symbol, args.freq, fetch_count)
        cfg = LabConfig(pivot_len=args.pivot_len)
        out = compute_sr_factor_lab(raw, cfg)
        stats = compute_grouped_statistics(out)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_panel" if use_shards else ""
    csv_path = out_dir / f"grouped_statistics{suffix}.csv"
    stats.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(tabulate(stats, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
