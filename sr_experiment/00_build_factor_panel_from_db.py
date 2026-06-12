# -*- coding: utf-8 -*-
"""
Purpose: 从数据库批量构建 SR 因子面板，输出分片 parquet（内存友好，支持断点续跑）
Inputs:  数据库股票池 + stock_k_data 行情表
Outputs: results/shards/{freq}_pv{pivot_len}/panel_XXXX.parquet (分片面板)
         results/shards/{freq}_pv{pivot_len}/events_XXXX.parquet (分片事件)
         results/shards/{freq}_pv{pivot_len}/manifest.json (分片清单)
How to Run:
    python sr_experiment/00_build_factor_panel_from_db.py --freq w --pivot-len 10 --limit-stocks 10
    python sr_experiment/00_build_factor_panel_from_db.py --freq w --pivot-len 10 --batch-size 200
    python sr_experiment/00_build_factor_panel_from_db.py --freq d --pivot-len 10
Examples:
    python sr_experiment/00_build_factor_panel_from_db.py --freq w --limit-stocks 3
    python sr_experiment/00_build_factor_panel_from_db.py --freq w --pivot-len 10 --batch-size 500
Side Effects: 写 parquet 文件到 sr_experiment/results/shards/ 目录
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.sr_event_factor_lab import LabConfig, compute_sr_factor_lab
from sr_experiment.db_adapter import get_stock_pool, load_kline
from sr_experiment.sr_config import EVENT_COLS, OUT_DIR


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if "index" in df.columns:
        df = df.rename(columns={"index": "bar_time"})
    if "bar_time" in df.columns:
        df["bar_time"] = pd.to_datetime(df["bar_time"])
    return df


def _shard_dir(freq: str, pivot_len: int) -> Path:
    return Path(OUT_DIR) / "shards" / f"{freq}_pv{pivot_len}"


def _scan_done_stocks(sdir: Path) -> set[str]:
    """扫描已有分片文件，提取已完成的 ts_code 集合（用于断点续跑）。"""
    done = set()
    for p in sorted(sdir.glob("panel_*.parquet")):
        try:
            df = pd.read_parquet(p, columns=["ts_code"])
            done.update(df["ts_code"].unique().tolist())
        except Exception:
            pass
    return done


def build_panel(
    freq: str,
    pivot_len: int,
    limit_stocks: int | None = None,
    batch_size: int = 200,
    resume: bool = True,
) -> tuple[int, int]:
    pool = get_stock_pool()
    codes = pool["ts_code"].tolist()
    if limit_stocks:
        codes = codes[:limit_stocks]

    cfg = LabConfig(pivot_len=pivot_len)
    sdir = _shard_dir(freq, pivot_len)
    sdir.mkdir(parents=True, exist_ok=True)

    done_stocks: set[str] = set()
    if resume:
        done_stocks = _scan_done_stocks(sdir)
        if done_stocks:
            print(f"断点续跑: 已有 {len(done_stocks)} 只股票的分片，跳过")

    existing_panels = sorted(sdir.glob("panel_*.parquet"))
    existing_events = sorted(sdir.glob("events_*.parquet"))
    batch_idx = len(existing_panels)

    manifest = {"panel_shards": [], "event_shards": [], "total_rows": 0, "total_event_rows": 0}
    for p in existing_panels:
        try:
            df = pd.read_parquet(p, columns=["ts_code"])
            manifest["panel_shards"].append(p.name)
            manifest["total_rows"] += len(df)
        except Exception:
            pass
    for p in existing_events:
        try:
            df = pd.read_parquet(p, columns=["ts_code"])
            manifest["event_shards"].append(p.name)
            manifest["total_event_rows"] += len(df)
        except Exception:
            pass

    total_done = len(done_stocks)
    total_events = manifest["total_event_rows"]
    skipped = []
    batch_frames = []
    batch_event_frames = []

    remaining = [c for c in codes if c not in done_stocks]
    print(f"待处理: {len(remaining)} 只股票 (总池: {len(codes)}, 已完成: {len(done_stocks)})")

    for ts_code in tqdm(remaining, desc=f"Building panel [{freq}] pv{pivot_len}"):
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
        batch_frames.append(out.reset_index())

        mask_any = pd.Series(False, index=out.index)
        for col in EVENT_COLS:
            if col in out.columns:
                mask_any = mask_any | out[col].fillna(False).astype(bool)
        if mask_any.any():
            evt_df = out.loc[mask_any].copy()
            batch_event_frames.append(evt_df.reset_index())
            total_events += len(evt_df)

        total_done += 1

        if len(batch_frames) >= batch_size:
            _flush_batch(batch_frames, batch_event_frames, sdir, batch_idx, manifest)
            batch_idx += 1
            batch_frames = []
            batch_event_frames = []
            gc.collect()

    if batch_frames:
        _flush_batch(batch_frames, batch_event_frames, sdir, batch_idx, manifest)

    manifest_path = sdir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"完成: {total_done}/{len(codes)} 只股票, 跳过: {len(skipped)}, 事件行: {total_events}")
    print(f"分片目录: {sdir}")
    print(f"面板分片: {len(manifest['panel_shards'])} 个, 总行数: {manifest['total_rows']}")
    print(f"事件分片: {len(manifest['event_shards'])} 个, 总行数: {manifest['total_event_rows']}")
    if skipped and len(skipped) <= 20:
        print(f"跳过前20: {skipped[:20]}")

    return total_done, total_events


def _flush_batch(
    frames: list[pd.DataFrame],
    event_frames: list[pd.DataFrame],
    sdir: Path,
    batch_idx: int,
    manifest: dict,
) -> None:
    if not frames:
        return

    panel_name = f"panel_{batch_idx:04d}.parquet"
    batch = _normalize_df(pd.concat(frames, ignore_index=True))
    batch.to_parquet(sdir / panel_name, index=False)
    manifest["panel_shards"].append(panel_name)
    manifest["total_rows"] += len(batch)

    if event_frames:
        events_name = f"events_{batch_idx:04d}.parquet"
        evt_batch = _normalize_df(pd.concat(event_frames, ignore_index=True))
        evt_batch.to_parquet(sdir / events_name, index=False)
        manifest["event_shards"].append(events_name)
        manifest["total_event_rows"] += len(evt_batch)


def iter_shards(
    freq: str,
    pivot_len: int,
    columns: list[str] | None = None,
    shard_type: str = "panel",
) -> pd.DataFrame:
    """逐片读取 parquet 分片，yield 每片 DataFrame，避免全量加载。"""
    sdir = _shard_dir(freq, pivot_len)
    manifest_path = sdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    key = f"{shard_type}_shards"
    for shard_name in manifest.get(key, []):
        path = sdir / shard_name
        if path.exists():
            df = pd.read_parquet(path, columns=columns)
            yield df


def main():
    parser = argparse.ArgumentParser(description="从数据库批量构建 SR 因子面板（分片保存，支持断点续跑）")
    parser.add_argument("--freq", type=str, default="w", help="周期 (d/w/60m)")
    parser.add_argument("--pivot-len", type=int, default=10, help="pivot 左右确认长度")
    parser.add_argument("--limit-stocks", type=int, default=None, help="限制股票数量（调试用）")
    parser.add_argument("--batch-size", type=int, default=200, help="每批保存的股票数（默认200）")
    parser.add_argument("--no-resume", action="store_true", help="不从断点续跑，清空重来")
    args = parser.parse_args()

    t0 = time.time()
    total_done, total_events = build_panel(
        args.freq, args.pivot_len, args.limit_stocks, args.batch_size,
        resume=not args.no_resume,
    )
    elapsed = time.time() - t0
    print(f"耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
