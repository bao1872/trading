# -*- coding: utf-8 -*-
"""
Purpose: 压力转支撑（R2S）专项统计分析
Inputs:  pytdx/db 拉取 K 线数据（symbol, freq, pivot_len, fetch_bars, data_source）
         或 parquet panel 文件（panel_file）
Outputs: results/r2s_statistics.csv / results/r2s_statistics_panel.csv
How to Run:
    python sr_experiment/07_r2s_statistics.py --symbol 300133 --freq w --pivot-len 10 --fetch-bars 1200
    python sr_experiment/07_r2s_statistics.py --symbol 300133 --freq w --data-source db
    python sr_experiment/07_r2s_statistics.py --panel-file sr_experiment/results/sr_factor_panel_w_pv10.parquet
Examples:
    python sr_experiment/07_r2s_statistics.py
    python sr_experiment/07_r2s_statistics.py --symbol 300133 --freq w --pivot-len 10 --fetch-bars 1200
    python sr_experiment/07_r2s_statistics.py --data-source db --symbol 300133 --freq w
    python sr_experiment/07_r2s_statistics.py --panel-file sr_experiment/results/sr_factor_panel_w_pv10.parquet
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
from sr_experiment.sr_config import FWD_RET_COLS, OUT_DIR


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

    stats = compute_r2s_statistics(out)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_panel" if use_panel else ""
    csv_path = out_dir / f"r2s_statistics{suffix}.csv"
    stats.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(tabulate(stats, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
