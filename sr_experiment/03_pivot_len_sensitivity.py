# -*- coding: utf-8 -*-
"""
Purpose: pivot_len 参数敏感性实验（5/10/20 等）
Inputs:  K 线数据（--data-source db 从数据库加载，pytdx 从通达信拉取）
Outputs: results/pivot_len_sensitivity.csv
How to Run:
    python sr_experiment/03_pivot_len_sensitivity.py --symbol 300133 --freq w --pivot-lens 5,10,20 --bars 300 --fetch-bars 1200
    python sr_experiment/03_pivot_len_sensitivity.py --symbol 300133 --freq w --data-source pytdx
Examples:
    python sr_experiment/03_pivot_len_sensitivity.py
    python sr_experiment/03_pivot_len_sensitivity.py --symbol 300133 --freq w --pivot-lens 5,10,20 --bars 300 --fetch-bars 1200
    python sr_experiment/03_pivot_len_sensitivity.py --data-source pytdx
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
from sr_experiment.db_adapter import load_kline as load_kline_db
from sr_experiment.sr_config import EVENT_COLS, OUT_DIR


def run_pivot_len_sensitivity(symbol: str, freq: str, pivot_lens: list[int], bars: int, fetch_bars: int, data_source: str = "db") -> pd.DataFrame:
    rows = []
    for pl in pivot_lens:
        if data_source == "db":
            raw = load_kline_db(symbol, freq)
        else:
            fetch_count = max(fetch_bars, bars, 2 * pl + 200)
            raw = fetch_kline_pytdx(symbol, freq, fetch_count)
        cfg = LabConfig(pivot_len=pl)
        out = compute_sr_factor_lab(raw, cfg)

        row = {"pivot_len": pl, "total_rows": len(out)}
        for col in EVENT_COLS:
            if col in out.columns:
                row[col] = int(out[col].sum())
            else:
                row[col] = 0

        support_valid = out["support_ref"].notna().sum() if "support_ref" in out.columns else 0
        resistance_valid = out["resistance_ref"].notna().sum() if "resistance_ref" in out.columns else 0
        row["support_valid_bars"] = int(support_valid)
        row["resistance_valid_bars"] = int(resistance_valid)

        if "sr_range_pct" in out.columns:
            row["sr_range_pct_mean"] = out["sr_range_pct"].mean()
            row["sr_range_pct_median"] = out["sr_range_pct"].median()
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="pivot_len 参数敏感性实验")
    parser.add_argument("--symbol", default="300133")
    parser.add_argument("--freq", default="w")
    parser.add_argument("--pivot-lens", default="5,10,20")
    parser.add_argument("--bars", type=int, default=300)
    parser.add_argument("--fetch-bars", type=int, default=1200)
    parser.add_argument("--data-source", default="db", choices=["db", "pytdx"])
    args = parser.parse_args()

    pivot_lens = [int(x.strip()) for x in args.pivot_lens.split(",") if x.strip()]
    df = run_pivot_len_sensitivity(args.symbol, args.freq, pivot_lens, args.bars, args.fetch_bars, data_source=args.data_source)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pivot_len_sensitivity.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
