# -*- coding: utf-8 -*-
"""
Purpose: 核心事件统计分析（样本数、触发率、未来收益/回撤/胜率/收益回撤比）
Inputs:  pytdx/db 拉取 K 线数据（symbol, freq, pivot_len, fetch_bars, data_source）
         或 parquet panel 文件（panel_file）
Outputs: results/event_statistics.csv / results/event_statistics_panel.csv
How to Run:
    python sr_experiment/04_event_statistics.py --symbol 300133 --freq w --pivot-len 10 --fetch-bars 1200
    python sr_experiment/04_event_statistics.py --symbol 300133 --freq w --data-source db
    python sr_experiment/04_event_statistics.py --panel-file sr_experiment/results/sr_factor_panel_w_pv10.parquet
Examples:
    python sr_experiment/04_event_statistics.py
    python sr_experiment/04_event_statistics.py --symbol 300133 --freq w --pivot-len 10 --fetch-bars 1200
    python sr_experiment/04_event_statistics.py --data-source db --symbol 300133 --freq w
    python sr_experiment/04_event_statistics.py --panel-file sr_experiment/results/sr_factor_panel_w_pv10.parquet
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
)


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
    parser.add_argument("--symbol", default="300133")
    parser.add_argument("--freq", default="w")
    parser.add_argument("--pivot-len", type=int, default=10)
    parser.add_argument("--fetch-bars", type=int, default=1200)
    parser.add_argument("--data-source", type=str, default="db", choices=["db", "pytdx"])
    parser.add_argument("--panel-file", type=str, default=None, help="parquet panel 文件路径")
    args = parser.parse_args()

    use_panel = args.panel_file is not None

    if use_panel:
        panel = pd.read_parquet(args.panel_file)
        if "ts_code" in panel.columns and args.symbol:
            ts_code = _code_to_ts_code(args.symbol)
            if ts_code in panel["ts_code"].values:
                out = panel[panel["ts_code"] == ts_code].copy()
            else:
                out = panel.copy()
        else:
            out = panel.copy()
        if "bar_time" in out.columns:
            out = out.set_index("bar_time")
    else:
        if args.data_source == "db":
            raw = load_kline_db(args.symbol, args.freq)
        else:
            fetch_count = max(args.fetch_bars, 2 * args.pivot_len + 200)
            raw = fetch_kline_pytdx(args.symbol, args.freq, fetch_count)
        cfg = LabConfig(pivot_len=args.pivot_len)
        out = compute_sr_factor_lab(raw, cfg)

    stats = compute_event_statistics(out)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_panel" if use_panel else ""
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
