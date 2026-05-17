# -*- coding: utf-8 -*-
"""
Purpose: 核心事件统计分析（样本数、触发率、未来收益/回撤/胜率/收益回撤比）
Inputs:  分片 parquet（freq + pivot_len 定位分片目录）或 单股票 db/pytdx 数据
Outputs: results/event_statistics.csv / results/event_statistics_panel.csv
How to Run:
    python sr_experiment/04_event_statistics.py --freq w --pivot-len 10
    python sr_experiment/04_event_statistics.py --symbol 300133 --freq w --pivot-len 10
    python sr_experiment/04_event_statistics.py --symbol 300133 --freq w --data-source db
Examples:
    python sr_experiment/04_event_statistics.py --freq w --pivot-len 10
    python sr_experiment/04_event_statistics.py --symbol 300133 --freq w --pivot-len 10 --fetch-bars 1200
    python sr_experiment/04_event_statistics.py --symbol 300133 --freq w --data-source db
Side Effects: 写 CSV 文件到 sr_experiment/results/ 目录
"""
from __future__ import annotations

import argparse
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
    FWD_MAX_COLS,
    FWD_MDD_COLS,
    FWD_RET_COLS,
    FWD_RR_COLS,
    OUT_DIR,
    PANEL_EVENT_STAT_COLS,
)

import importlib

_build_mod = importlib.import_module("sr_experiment.00_build_factor_panel_from_db")
iter_shards = _build_mod.iter_shards


def _compute_event_stats_shard(df: pd.DataFrame) -> pd.DataFrame:
    """对单个分片计算事件统计，返回每事件的 count/sum/mean/win_count。"""
    total = len(df)
    rows = []
    for evt in EVENT_COLS:
        if evt not in df.columns:
            continue
        mask = df[evt].fillna(False).astype(bool)
        n = int(mask.sum())
        if n == 0:
            continue
        sub = df.loc[mask]
        row = {"event": evt, "count": n, "total_in_shard": total}
        for col in FWD_RET_COLS:
            if col in sub.columns:
                vals = sub[col].dropna()
                row[f"{col}_sum"] = vals.sum() if len(vals) > 0 else 0.0
                row[f"{col}_count"] = len(vals)
                row[f"{col}_median"] = vals.median() if len(vals) > 0 else np.nan
                row[f"{col}_win_count"] = int((vals > 0).sum()) if len(vals) > 0 else 0
        for col in FWD_MAX_COLS:
            if col in sub.columns:
                vals = sub[col].dropna()
                row[f"{col}_sum"] = vals.sum() if len(vals) > 0 else 0.0
                row[f"{col}_count"] = len(vals)
        for col in FWD_MDD_COLS:
            if col in sub.columns:
                vals = sub[col].dropna()
                row[f"{col}_sum"] = vals.sum() if len(vals) > 0 else 0.0
                row[f"{col}_count"] = len(vals)
        for col in FWD_RR_COLS:
            if col in sub.columns:
                vals = sub[col].dropna()
                row[f"{col}_sum"] = vals.sum() if len(vals) > 0 else 0.0
                row[f"{col}_count"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_shard_stats(shard_stats: list[pd.DataFrame]) -> pd.DataFrame:
    """合并多个分片的统计结果，用加权平均计算 mean/win_rate。"""
    if not shard_stats:
        return pd.DataFrame()

    all_stats = pd.concat(shard_stats, ignore_index=True)

    result_rows = []
    for evt, grp in all_stats.groupby("event"):
        total_count = grp["count"].sum()
        total_rows_in_panel = grp["total_in_shard"].sum()
        row = {
            "event": evt,
            "count": total_count,
            "trigger_rate": total_count / total_rows_in_panel if total_rows_in_panel > 0 else np.nan,
        }
        for col in FWD_RET_COLS:
            s = grp[f"{col}_sum"].sum() if f"{col}_sum" in grp.columns else 0.0
            c = grp[f"{col}_count"].sum() if f"{col}_count" in grp.columns else 0
            wc = grp[f"{col}_win_count"].sum() if f"{col}_win_count" in grp.columns else 0
            row[f"{col}_mean"] = s / c if c > 0 else np.nan
            row[f"{col}_median"] = np.nan
            row[f"{col}_win_rate"] = wc / c if c > 0 else np.nan
        for col in FWD_MAX_COLS:
            s = grp[f"{col}_sum"].sum() if f"{col}_sum" in grp.columns else 0.0
            c = grp[f"{col}_count"].sum() if f"{col}_count" in grp.columns else 0
            row[f"{col}_mean"] = s / c if c > 0 else np.nan
        for col in FWD_MDD_COLS:
            s = grp[f"{col}_sum"].sum() if f"{col}_sum" in grp.columns else 0.0
            c = grp[f"{col}_count"].sum() if f"{col}_count" in grp.columns else 0
            row[f"{col}_mean"] = s / c if c > 0 else np.nan
        for col in FWD_RR_COLS:
            s = grp[f"{col}_sum"].sum() if f"{col}_sum" in grp.columns else 0.0
            c = grp[f"{col}_count"].sum() if f"{col}_count" in grp.columns else 0
            row[f"{col}_mean"] = s / c if c > 0 else np.nan
        result_rows.append(row)

    return pd.DataFrame(result_rows)


def compute_event_statistics(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    for evt in EVENT_COLS:
        if evt not in df.columns:
            continue
        mask = df[evt].fillna(False).astype(bool)
        n = int(mask.sum())
        if n == 0:
            continue
        sub = df.loc[mask]
        row = {
            "event": evt,
            "count": n,
            "trigger_rate": n / total,
        }
        for col in FWD_RET_COLS:
            if col in sub.columns:
                vals = sub[col].dropna()
                row[f"{col}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                row[f"{col}_median"] = vals.median() if len(vals) > 0 else np.nan
                row[f"{col}_win_rate"] = (vals > 0).mean() if len(vals) > 0 else np.nan
        for col in FWD_MAX_COLS:
            if col in sub.columns:
                vals = sub[col].dropna()
                row[f"{col}_mean"] = vals.mean() if len(vals) > 0 else np.nan
        for col in FWD_MDD_COLS:
            if col in sub.columns:
                vals = sub[col].dropna()
                row[f"{col}_mean"] = vals.mean() if len(vals) > 0 else np.nan
        for col in FWD_RR_COLS:
            if col in sub.columns:
                vals = sub[col].dropna()
                row[f"{col}_mean"] = vals.mean() if len(vals) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="核心事件统计分析")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--freq", default="w")
    parser.add_argument("--pivot-len", type=int, default=10)
    parser.add_argument("--fetch-bars", type=int, default=1200)
    parser.add_argument("--data-source", type=str, default="db", choices=["db", "pytdx"])
    args = parser.parse_args()

    use_shards = args.symbol is None

    if use_shards:
        cols_to_load = [c for c in PANEL_EVENT_STAT_COLS if c != "ts_code"]
        if "ts_code" not in cols_to_load:
            cols_to_load.append("ts_code")
        shard_stats = []
        for shard_df in iter_shards(args.freq, args.pivot_len, columns=cols_to_load, shard_type="panel"):
            stats = _compute_event_stats_shard(shard_df)
            if not stats.empty:
                shard_stats.append(stats)
            del shard_df
        stats = _merge_shard_stats(shard_stats)
    else:
        symbol = args.symbol
        if args.data_source == "db":
            raw = load_kline_db(symbol, args.freq)
        else:
            fetch_count = max(args.fetch_bars, 2 * args.pivot_len + 200)
            raw = fetch_kline_pytdx(symbol, args.freq, fetch_count)
        cfg = LabConfig(pivot_len=args.pivot_len)
        out = compute_sr_factor_lab(raw, cfg)
        stats = compute_event_statistics(out)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_panel" if use_shards else ""
    csv_path = out_dir / f"event_statistics{suffix}.csv"
    stats.to_csv(csv_path, index=False, encoding="utf-8-sig")

    display_cols = ["event", "count", "trigger_rate"]
    for col in FWD_RET_COLS:
        if f"{col}_mean" in stats.columns:
            display_cols.extend([f"{col}_mean", f"{col}_win_rate"])
    print(tabulate(stats[display_cols], headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
