# -*- coding: utf-8 -*-
"""
Purpose: 多周期支撑/压力事件对比实验（d/w/60m 等）
Inputs:  K 线数据（--data-source db 从数据库加载，pytdx 从通达信拉取）
Outputs: results/multi_freq_comparison.csv
How to Run:
    python sr_experiment/02_multi_freq_experiment.py --symbol 300133 --freqs d,w,60m --pivot-len 10 --bars 300 --fetch-bars 1200
    python sr_experiment/02_multi_freq_experiment.py --symbol 300133 --freqs d,w,60m --data-source pytdx
Examples:
    python sr_experiment/02_multi_freq_experiment.py
    python sr_experiment/02_multi_freq_experiment.py --symbol 300133 --freqs d,w,60m --pivot-len 10 --bars 300 --fetch-bars 1200
    python sr_experiment/02_multi_freq_experiment.py --data-source pytdx
Side Effects: 写 CSV 文件到 sr_experiment/results/ 目录
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.sr_event_factor_lab import (
    LabConfig,
    compute_sr_factor_lab,
    fetch_kline_pytdx,
)
from sr_experiment.db_adapter import load_kline as load_kline_db
from sr_experiment.sr_config import EVENT_COLS, OUT_DIR


def run_multi_freq(symbol: str, freqs: list[str], pivot_len: int, bars: int, fetch_bars: int, data_source: str = "db") -> pd.DataFrame:
    rows = []
    for freq in freqs:
        if data_source == "db":
            raw = load_kline_db(symbol, freq)
        else:
            fetch_count = max(fetch_bars, bars, 2 * pivot_len + 200)
            raw = fetch_kline_pytdx(symbol, freq, fetch_count)
        cfg = LabConfig(pivot_len=pivot_len)
        out = compute_sr_factor_lab(raw, cfg)

        row = {"freq": freq, "total_rows": len(out)}
        for col in EVENT_COLS:
            if col in out.columns:
                row[col] = int(out[col].sum())
            else:
                row[col] = 0
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="多周期支撑/压力事件对比实验")
    parser.add_argument("--symbol", default="300133")
    parser.add_argument("--freqs", default="d,w,60m")
    parser.add_argument("--pivot-len", type=int, default=10)
    parser.add_argument("--bars", type=int, default=300)
    parser.add_argument("--fetch-bars", type=int, default=1200)
    parser.add_argument("--data-source", default="db", choices=["db", "pytdx"])
    args = parser.parse_args()

    freqs = [f.strip() for f in args.freqs.split(",") if f.strip()]
    df = run_multi_freq(args.symbol, freqs, args.pivot_len, args.bars, args.fetch_bars, data_source=args.data_source)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "multi_freq_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
