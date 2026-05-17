# -*- coding: utf-8 -*-
"""
Purpose: 从数据库批量构建 SR 因子面板，输出 parquet
Inputs:  数据库股票池 + stock_k_data 行情表
Outputs: results/sr_factor_panel_{freq}_pv{pivot_len}.parquet
         results/sr_events_only_{freq}_pv{pivot_len}.parquet
How to Run:
    python sr_experiment/00_build_factor_panel_from_db.py --freq w --pivot-len 10 --limit-stocks 10
    python sr_experiment/00_build_factor_panel_from_db.py --freq d --pivot-len 10
Examples:
    python sr_experiment/00_build_factor_panel_from_db.py --freq w --limit-stocks 3
    python sr_experiment/00_build_factor_panel_from_db.py --freq w --pivot-len 10
Side Effects: 写 parquet 文件到 sr_experiment/results/ 目录
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.sr_event_factor_lab import LabConfig, compute_sr_factor_lab
from sr_experiment.db_adapter import get_stock_pool, load_kline
from sr_experiment.sr_config import EVENT_COLS, OUT_DIR


def build_panel(freq: str, pivot_len: int, limit_stocks: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = get_stock_pool()
    codes = pool["ts_code"].tolist()
    if limit_stocks:
        codes = codes[:limit_stocks]

    cfg = LabConfig(pivot_len=pivot_len)
    all_frames = []
    event_frames = []
    skipped = []

    for ts_code in tqdm(codes, desc=f"Building panel [{freq}] pv{pivot_len}"):
        try:
            raw = load_kline(ts_code, freq)
        except Exception:
            skipped.append(ts_code)
            continue

        if raw.empty or len(raw) < 2 * pivot_len + 50:
            skipped.append(ts_code)
            continue

        try:
            out = compute_sr_factor_lab(raw, cfg)
        except Exception:
            skipped.append(ts_code)
            continue

        out["ts_code"] = ts_code
        all_frames.append(out.reset_index())

        mask_any = pd.Series(False, index=out.index)
        for col in EVENT_COLS:
            if col in out.columns:
                mask_any = mask_any | out[col].fillna(False).astype(bool)
        if mask_any.any():
            evt_df = out.loc[mask_any].copy()
            event_frames.append(evt_df.reset_index())

    if not all_frames:
        return pd.DataFrame(), pd.DataFrame()

    panel = pd.concat(all_frames, ignore_index=True)
    if "index" in panel.columns:
        panel = panel.rename(columns={"index": "bar_time"})
    if "bar_time" in panel.columns:
        panel["bar_time"] = pd.to_datetime(panel["bar_time"])

    events_only = pd.DataFrame()
    if event_frames:
        events_only = pd.concat(event_frames, ignore_index=True)
        if "index" in events_only.columns:
            events_only = events_only.rename(columns={"index": "bar_time"})
        if "bar_time" in events_only.columns:
            events_only["bar_time"] = pd.to_datetime(events_only["bar_time"])

    print(f"完成: {len(all_frames)}/{len(codes)} 只股票, 跳过: {len(skipped)}")
    if skipped and len(skipped) <= 10:
        print(f"跳过: {skipped}")

    return panel, events_only


def main():
    parser = argparse.ArgumentParser(description="从数据库批量构建 SR 因子面板")
    parser.add_argument("--freq", type=str, default="w", help="周期 (d/w/60m)")
    parser.add_argument("--pivot-len", type=int, default=10, help="pivot 左右确认长度")
    parser.add_argument("--limit-stocks", type=int, default=None, help="限制股票数量（调试用）")
    args = parser.parse_args()

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    panel, events_only = build_panel(args.freq, args.pivot_len, args.limit_stocks)
    elapsed = time.time() - t0

    if panel.empty:
        print("面板为空，无输出")
        return

    freq_tag = args.freq
    pv_tag = args.pivot_len
    panel_path = out_dir / f"sr_factor_panel_{freq_tag}_pv{pv_tag}.parquet"
    events_path = out_dir / f"sr_events_only_{freq_tag}_pv{pv_tag}.parquet"

    panel.to_parquet(panel_path, index=False)
    print(f"面板已保存: {panel_path} ({len(panel)} rows, {len(panel.columns)} cols)")

    if not events_only.empty:
        events_only.to_parquet(events_path, index=False)
        print(f"事件行已保存: {events_path} ({len(events_only)} rows)")

    print(f"耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
