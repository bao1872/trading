# -*- coding: utf-8 -*-
"""
Purpose: 单票周线支撑/压力事件因子实验
Inputs:  K 线数据（--data-source db 从数据库加载，pytdx 从通达信拉取）
Outputs: results/{symbol}_{freq}_sr_factor_lab_pv{pivot_len}.csv
         results/{symbol}_{freq}_sr_factor_lab_pv{pivot_len}_events_only.csv
         results/{symbol}_{freq}_sr_factor_lab_pv{pivot_len}.html
How to Run:
    python sr_experiment/01_single_stock_weekly.py --symbol 300133 --freq w --pivot-len 10 --bars 300 --fetch-bars 1200
    python sr_experiment/01_single_stock_weekly.py --symbol 300133 --freq w --data-source pytdx
Examples:
    python sr_experiment/01_single_stock_weekly.py
    python sr_experiment/01_single_stock_weekly.py --symbol 300133 --freq w --pivot-len 10 --bars 300 --fetch-bars 1200
    python sr_experiment/01_single_stock_weekly.py --data-source pytdx
Side Effects: 写 CSV/HTML 文件到 sr_experiment/results/ 目录
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.sr_event_factor_lab import (
    LabConfig,
    build_html,
    compute_sr_factor_lab,
    fetch_kline_pytdx,
)
from sr_experiment.db_adapter import load_kline as load_kline_db
from sr_experiment.sr_config import EVENT_COLS, OUT_DIR


def run_single_stock(symbol: str, freq: str, pivot_len: int, bars: int, fetch_bars: int, data_source: str = "db") -> dict:
    if data_source == "db":
        raw = load_kline_db(symbol, freq)
    else:
        fetch_count = max(fetch_bars, bars, 2 * pivot_len + 200)
        raw = fetch_kline_pytdx(symbol, freq, fetch_count)
    cfg = LabConfig(pivot_len=pivot_len)
    out_full = compute_sr_factor_lab(raw, cfg)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol}_{freq}_sr_factor_lab_pv{pivot_len}"
    csv_path = out_dir / f"{stem}.csv"
    html_path = out_dir / f"{stem}.html"
    events_csv_path = out_dir / f"{stem}_events_only.csv"

    out_full.to_csv(csv_path, encoding="utf-8-sig")

    mask_any_event = pd.Series(False, index=out_full.index)
    for col in EVENT_COLS:
        if col in out_full.columns:
            mask_any_event = mask_any_event | out_full[col].fillna(False).astype(bool)
    out_full.loc[mask_any_event].to_csv(events_csv_path, encoding="utf-8-sig")

    out_plot = out_full.tail(bars).copy()
    title = f"{symbol} [{freq}] SR Event Factor Lab | pivot_len={pivot_len}"
    build_html(out_full, out_plot, str(html_path), title, pivot_len)

    stats = {}
    for col in EVENT_COLS:
        if col in out_full.columns:
            stats[col] = int(out_full[col].sum())
    stats["total_rows"] = len(out_full)
    stats["events_rows"] = int(mask_any_event.sum())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="单票周线支撑/压力事件因子实验")
    parser.add_argument("--symbol", default="300133")
    parser.add_argument("--freq", default="w")
    parser.add_argument("--pivot-len", type=int, default=10)
    parser.add_argument("--bars", type=int, default=300)
    parser.add_argument("--fetch-bars", type=int, default=1200)
    parser.add_argument("--data-source", default="db", choices=["db", "pytdx"])
    args = parser.parse_args()

    stats = run_single_stock(args.symbol, args.freq, args.pivot_len, args.bars, args.fetch_bars, data_source=args.data_source)

    print(f"总K线数: {stats['total_rows']}")
    print(f"事件K线数: {stats['events_rows']}")
    print("\n事件计数:")
    for col in EVENT_COLS:
        if col in stats:
            print(f"  {col}: {stats[col]}")


if __name__ == "__main__":
    main()
