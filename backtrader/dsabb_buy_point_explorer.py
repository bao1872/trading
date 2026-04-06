# -*- coding: utf-8 -*-
"""
基于 DSA VWAP + ATR Rope + Bollinger 因子的 DSABB 买点质量研究脚本

Purpose
  - 参考 buy_point_explorer.py 的实验组织方式
  - 研究“DSA 多头 + 已出现 HL/HH + 首次进入布林下轨区”的回踩低吸质量
  - 支持日线/周线 DSA 过滤，并输出事件表、候选汇总、参数扫描、时间切片、固定样本验证、邻域稳健性

Core event
  - DSA_DIR == 1
  - last_pivot_type in {HL, HH}
  - bb_pos_01 首次进入下轨区域
  - 可选：日线 VWAP 偏离、日线 DSA 位置、Rope 方向、周线 DSA 方向/位置过滤

Outputs
  - dsabb_buy_point_explorer/*.csv

How to Run
  python dsabb_buy_point_explorer.py --n-stocks 300
  python dsabb_buy_point_explorer.py --n-stocks 1000 --dsabb-bb-thr 0.15 --dsabb-weekly-dir-required
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from datasource.database import get_engine
from features.merged_dsa_atr_rope_bb_factors import (
    DSAConfig,
    RopeConfig,
    compute_atr_rope,
    compute_bollinger,
    compute_dsa,
)

warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = "dsabb_buy_point_explorer"
os.makedirs(OUT_DIR, exist_ok=True)

RET_WINDOWS = [5, 10, 20]
EVENT_DEDUP_BARS = 20


# =========================
# Generic helpers
# =========================
def rr_from_ret_dd(ret: float, dd: float) -> float:
    if pd.isna(ret) or pd.isna(dd) or dd == 0:
        return np.nan
    return float(ret / abs(dd))


def summarize_events(events: pd.DataFrame, windows: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {"n": int(len(events))}
    if len(events) == 0:
        for w in windows:
            out[f"ret_{w}"] = np.nan
            out[f"mae_{w}"] = np.nan
            out[f"wr_{w}"] = np.nan
            out[f"rr_{w}"] = np.nan
        return out
    for w in windows:
        ret_col = f"ret_{w}"
        dd_col = f"max_dd_{w}"
        win_col = f"win_{w}"
        ret = events[ret_col].mean() if ret_col in events.columns else np.nan
        dd = events[dd_col].mean() if dd_col in events.columns else np.nan
        wr = events[win_col].mean() if win_col in events.columns else np.nan
        out[f"ret_{w}"] = round(float(ret), 5) if pd.notna(ret) else np.nan
        out[f"mae_{w}"] = round(float(dd), 5) if pd.notna(dd) else np.nan
        out[f"wr_{w}"] = round(float(wr), 4) if pd.notna(wr) else np.nan
        out[f"rr_{w}"] = round(rr_from_ret_dd(ret, dd), 3) if pd.notna(ret) and pd.notna(dd) else np.nan
    return out


def dedup_events(df: pd.DataFrame, cooldown_bars: int = EVENT_DEDUP_BARS) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "symbol" not in df.columns:
        return df.copy()
    work = df.sort_values(["symbol", "datetime"]).copy()
    keep_idx: List[int] = []
    for _, g in work.groupby("symbol", sort=False):
        accepted_positions: List[int] = []
        idxs = g.index.tolist()
        for pos, idx in enumerate(idxs):
            if not accepted_positions or pos - accepted_positions[-1] >= cooldown_bars:
                accepted_positions.append(pos)
                keep_idx.append(idx)
    return work.loc[keep_idx].sort_values(["datetime", "symbol"]).reset_index(drop=True)


def slice_equal_time_buckets(df: pd.DataFrame, n_slices: int = 3, time_col: str = "datetime") -> pd.DataFrame:
    out = df.sort_values(time_col).copy()
    if len(out) == 0:
        out["time_slice"] = []
        return out
    labels = [f"T{i+1}" for i in range(n_slices)]
    ranks = np.linspace(0, n_slices, len(out), endpoint=False)
    out["time_slice"] = [labels[min(int(x), n_slices - 1)] for x in ranks]
    return out


# =========================
# Data loaders
# =========================
def load_kline(ts_code: str, freq: str = "d", bars: int = 1000) -> Optional[pd.DataFrame]:
    from sqlalchemy import text

    engine = get_engine()
    sql = """
        SELECT bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = :freq
        ORDER BY bar_time DESC LIMIT :limit
    """
    try:
        df = pd.read_sql_query(text(sql), engine, params={"ts_code": ts_code, "freq": freq, "limit": bars})
    except Exception as exc:
        print(f"  [WARN] {ts_code} 查询失败: {exc}")
        return None
    finally:
        engine.dispose()
    if df.empty:
        return None
    df = df.sort_values("bar_time").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["bar_time"]).dt.tz_localize(None)
    df = df.set_index("datetime")
    df["vol"] = df["volume"].astype(float).replace(0, np.nan).ffill().fillna(1.0)
    return df[["open", "high", "low", "close", "vol"]].astype(float)


def get_stock_pool(n: int = 100, seed: int = 42) -> List[str]:
    from sqlalchemy import text

    engine = get_engine()
    sql = "SELECT ts_code FROM stock_pools ORDER BY ts_code"
    try:
        df = pd.read_sql_query(text(sql), engine)
    finally:
        engine.dispose()
    if df.empty:
        raise ValueError("stock_pools表为空")
    codes = df["ts_code"].tolist()
    rng = np.random.default_rng(seed)
    return rng.choice(codes, size=min(n, len(codes)), replace=False).tolist()


# =========================
# Factor computation
# =========================
def aggregate_weekly_strict(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    if "volume" not in d.columns and "vol" in d.columns:
        d["volume"] = d["vol"]
    wk = pd.DataFrame(index=d.resample("W-FRI").last().index)
    wk["open"] = d["open"].resample("W-FRI").first()
    wk["high"] = d["high"].resample("W-FRI").max()
    wk["low"] = d["low"].resample("W-FRI").min()
    wk["close"] = d["close"].resample("W-FRI").last()
    wk["volume"] = d["volume"].resample("W-FRI").sum()
    return wk.dropna()


def map_weekly_dsa_strict_to_daily(daily: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=daily.index)
    wk = aggregate_weekly_strict(daily)
    if len(wk) < 10:
        return out
    w_dsa, _, _ = compute_dsa(wk, DSAConfig(prd=50, base_apt=20.0))
    prev_friday = (daily.index.to_period("W-FRI").start_time + pd.offsets.Week(weekday=4) - pd.offsets.Week(1)).normalize()
    mapped = w_dsa.copy()
    mapped.index = mapped.index.normalize()
    out["w_DSA_DIR"] = mapped["DSA_DIR"].reindex(prev_friday).to_numpy()
    out["w_dsa_pivot_pos_01"] = mapped["dsa_pivot_pos_01"].reindex(prev_friday).to_numpy()
    out["w_signed_vwap_dev_pct"] = mapped["signed_vwap_dev_pct"].reindex(prev_friday).to_numpy()
    return out


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "vol" in d.columns and "volume" not in d.columns:
        d["volume"] = d["vol"]
    dsa_df, _, _ = compute_dsa(d, DSAConfig(prd=50, base_apt=20.0))
    rope_df = compute_atr_rope(d, RopeConfig(length=14, multi=1.5))
    bb_df = compute_bollinger(d, length=20, mult=2.0, pct_lookback=120)
    weekly_df = map_weekly_dsa_strict_to_daily(d)
    merged = pd.concat(
        [
            d,
            dsa_df,
            rope_df.drop(columns=d.columns, errors="ignore"),
            bb_df.drop(columns=d.columns, errors="ignore"),
            weekly_df,
        ],
        axis=1,
    )
    return merged


def add_forward_returns(df: pd.DataFrame, windows: Sequence[int] = RET_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].to_numpy(float)
    low = out["low"].to_numpy(float)
    high = out["high"].to_numpy(float)
    n = len(out)
    for w in windows:
        fut = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        max_up = np.full(n, np.nan)
        for i in range(n - w):
            entry = close[i]
            fut[i] = (close[i + w] - entry) / entry
            mae[i] = (np.nanmin(low[i + 1 : i + w + 1]) - entry) / entry
            max_up[i] = (np.nanmax(high[i + 1 : i + w + 1]) - entry) / entry
        out[f"ret_{w}"] = fut
        out[f"max_dd_{w}"] = mae
        out[f"max_up_{w}"] = max_up
        out[f"win_{w}"] = (out[f"ret_{w}"] > 0).astype(float)
    return out


def add_structure_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hlhh = out.get("last_pivot_type", pd.Series(index=out.index, dtype=object)).isin(["HL", "HH"])
    out["has_hl_hh_context"] = hlhh.astype(float)
    bars = np.full(len(out), np.nan)
    last_seen: Optional[int] = None
    for i, flag in enumerate(hlhh.to_numpy(dtype=bool)):
        if flag:
            last_seen = i
            bars[i] = 0.0
        elif last_seen is not None:
            bars[i] = float(i - last_seen)
    out["bars_since_last_hl_hh"] = bars
    out["enter_lower_zone_015"] = ((out.get("bb_pos_01") <= 0.15) & (out.get("bb_pos_01").shift(1) > 0.15)).astype(float)
    return out


# =========================
# DSABB events
# =========================
EVENT_COLS_BASE = [
    "symbol", "datetime", "open", "high", "low", "close",
    "DSA_DIR", "DSA_VWAP", "dsa_pivot_pos_01", "signed_vwap_dev_pct", "trend_aligned_vwap_dev_pct",
    "last_pivot_type", "has_hl_hh_context", "bars_since_last_hl_hh",
    "bb_mid", "bb_upper", "bb_lower", "bb_pos_01", "bb_width_norm", "bb_width_percentile", "bb_width_change_5",
    "bb_expand_streak", "bb_contract_streak",
    "rope_dir", "bars_since_dir_change", "dist_to_rope_atr", "dist_to_lower_atr", "channel_pos_01", "rope_pivot_pos_01",
    "w_DSA_DIR", "w_dsa_pivot_pos_01", "w_signed_vwap_dev_pct",
]


def build_dsabb_events(
    df: pd.DataFrame,
    bb_pos_thr: float = 0.15,
    require_hl_hh: bool = True,
    vwap_dev_lo: Optional[float] = -1.0,
    dsa_pos_hi: Optional[float] = 0.55,
    rope_dir_min: Optional[int] = 0,
    weekly_dir_required: bool = False,
    weekly_pos_hi: Optional[float] = None,
    cooldown: int = EVENT_DEDUP_BARS,
) -> pd.DataFrame:
    need = EVENT_COLS_BASE + [f"ret_{w}" for w in RET_WINDOWS] + [f"max_dd_{w}" for w in RET_WINDOWS] + [f"max_up_{w}" for w in RET_WINDOWS] + [f"win_{w}" for w in RET_WINDOWS]
    need = [c for c in need if c in df.columns]
    sub = df[need].dropna(subset=[c for c in ["ret_5", "ret_10", "ret_20", "bb_pos_01", "DSA_DIR"] if c in need]).copy()
    enter_lower_zone = (sub["bb_pos_01"] <= bb_pos_thr) & (sub["bb_pos_01"].shift(1) > bb_pos_thr)
    mask = (sub["DSA_DIR"] == 1) & enter_lower_zone
    if require_hl_hh and "last_pivot_type" in sub.columns:
        mask &= sub["last_pivot_type"].isin(["HL", "HH"])
    if vwap_dev_lo is not None and "signed_vwap_dev_pct" in sub.columns:
        mask &= sub["signed_vwap_dev_pct"] >= vwap_dev_lo
    if dsa_pos_hi is not None and "dsa_pivot_pos_01" in sub.columns:
        mask &= sub["dsa_pivot_pos_01"] <= dsa_pos_hi
    if rope_dir_min is not None and "rope_dir" in sub.columns:
        mask &= sub["rope_dir"] >= rope_dir_min
    if weekly_dir_required and "w_DSA_DIR" in sub.columns:
        mask &= sub["w_DSA_DIR"] == 1
    if weekly_pos_hi is not None and "w_dsa_pivot_pos_01" in sub.columns:
        mask &= sub["w_dsa_pivot_pos_01"] <= weekly_pos_hi
    events = dedup_events(sub[mask].copy(), cooldown)
    events["bb_pos_thr"] = bb_pos_thr
    events["require_hl_hh"] = int(require_hl_hh)
    events["vwap_dev_lo"] = np.nan if vwap_dev_lo is None else vwap_dev_lo
    events["dsa_pos_hi"] = np.nan if dsa_pos_hi is None else dsa_pos_hi
    events["rope_dir_min"] = np.nan if rope_dir_min is None else rope_dir_min
    events["weekly_dir_required"] = int(weekly_dir_required)
    events["weekly_pos_hi"] = np.nan if weekly_pos_hi is None else weekly_pos_hi
    for w in RET_WINDOWS:
        events[f"rr_{w}"] = events.apply(lambda r: rr_from_ret_dd(r[f"ret_{w}"], r[f"max_dd_{w}"]), axis=1)
    return events.reset_index(drop=True)


def build_dsabb_events_base(
    df: pd.DataFrame,
    bb_pos_thr: float,
    vwap_dev_lo: Optional[float],
    dsa_pos_hi: Optional[float],
    rope_dir_min: Optional[int],
    weekly_dir_required: bool,
    weekly_pos_hi: Optional[float],
    cooldown: int,
) -> pd.DataFrame:
    return build_dsabb_events(
        df,
        bb_pos_thr=bb_pos_thr,
        require_hl_hh=True,
        vwap_dev_lo=vwap_dev_lo,
        dsa_pos_hi=dsa_pos_hi,
        rope_dir_min=rope_dir_min,
        weekly_dir_required=weekly_dir_required,
        weekly_pos_hi=weekly_pos_hi,
        cooldown=cooldown,
    )


# =========================
# Analyses
# =========================
def analyze_03_dsabb_candidate_events(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次③: DSABB 候选策略事件表")
    print("=" * 70)
    variants = {
        "D1_HLHH后纯下轨低吸": dict(bb_pos_thr=0.20, vwap_dev_lo=None, dsa_pos_hi=None, rope_dir_min=None, weekly_dir_required=False, weekly_pos_hi=None),
        "D2_HLHH后健康回踩": dict(bb_pos_thr=0.15, vwap_dev_lo=-1.0, dsa_pos_hi=0.55, rope_dir_min=0, weekly_dir_required=False, weekly_pos_hi=None),
        "D3_HLHH后健康回踩_周线过滤": dict(bb_pos_thr=0.15, vwap_dev_lo=-1.0, dsa_pos_hi=0.55, rope_dir_min=0, weekly_dir_required=True, weekly_pos_hi=0.70),
    }
    rows = []
    all_events = []
    for name, cfg in variants.items():
        events = build_dsabb_events(df, require_hl_hh=True, **cfg)
        if len(events) == 0:
            continue
        row = {"strategy": name}
        row.update(summarize_events(events, RET_WINDOWS))
        row = {("event_n" if k == "n" else k): v for k, v in row.items()}
        rows.append(row)
        tmp = events.copy()
        tmp["strategy"] = name
        all_events.append(tmp)
    rdf = pd.DataFrame(rows).sort_values("rr_20", ascending=False)
    if not rdf.empty:
        print(rdf[["strategy", "event_n", "ret_5", "rr_5", "ret_10", "rr_10", "ret_20", "rr_20"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/dsabb_03_candidate_summary.csv", index=False)
        pd.concat(all_events, ignore_index=True).to_csv(f"{OUT_DIR}/dsabb_03_candidate_events.csv", index=False)
    return rdf


def analyze_04_dsabb_param_scan(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次④: DSABB 参数扫描")
    print("=" * 70)
    rows = []
    for bb_pos_thr in [0.10, 0.15, 0.20, 0.25]:
        for vwap_dev_lo in [None, -1.5, -1.0, -0.5]:
            for dsa_pos_hi in [None, 0.45, 0.55, 0.65]:
                for rope_dir_min in [None, 0, 1]:
                    for weekly_dir_required in [False, True]:
                        for weekly_pos_hi in [None, 0.70, 0.50]:
                            events = build_dsabb_events(
                                df,
                                bb_pos_thr=bb_pos_thr,
                                require_hl_hh=True,
                                vwap_dev_lo=vwap_dev_lo,
                                dsa_pos_hi=dsa_pos_hi,
                                rope_dir_min=rope_dir_min,
                                weekly_dir_required=weekly_dir_required,
                                weekly_pos_hi=weekly_pos_hi,
                            )
                            if len(events) < 20:
                                continue
                            row = {
                                "bb_pos_thr": bb_pos_thr,
                                "vwap_dev_lo": "none" if vwap_dev_lo is None else vwap_dev_lo,
                                "dsa_pos_hi": "none" if dsa_pos_hi is None else dsa_pos_hi,
                                "rope_dir_min": "none" if rope_dir_min is None else rope_dir_min,
                                "weekly_dir_required": int(weekly_dir_required),
                                "weekly_pos_hi": "none" if weekly_pos_hi is None else weekly_pos_hi,
                                "event_n": len(events),
                            }
                            row.update(summarize_events(events, RET_WINDOWS))
                            rows.append(row)
    rdf = pd.DataFrame(rows).sort_values("rr_20", ascending=False)
    if not rdf.empty:
        print(rdf.head(20)[["bb_pos_thr", "vwap_dev_lo", "dsa_pos_hi", "rope_dir_min", "weekly_dir_required", "weekly_pos_hi", "event_n", "ret_20", "rr_20"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/dsabb_04_param_scan.csv", index=False)
    return rdf


def analyze_05_dsabb_time_slices(events: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑤: DSABB 时间切片验证")
    print("=" * 70)
    sliced = slice_equal_time_buckets(events, 3)
    rows = []
    for name, g in sliced.groupby("time_slice", observed=True):
        row = {"time_slice": str(name), "start": g["datetime"].min().strftime("%Y-%m"), "end": g["datetime"].max().strftime("%Y-%m")}
        row.update(summarize_events(g, RET_WINDOWS))
        rows.append(row)
    rdf = pd.DataFrame(rows).sort_values("time_slice")
    if not rdf.empty:
        print(rdf[["time_slice", "start", "end", "n", "ret_5", "rr_5", "ret_10", "rr_10", "ret_20", "rr_20"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/dsabb_05_time_slices.csv", index=False)
    return rdf


def analyze_06_dsabb_fixed_validation(df: pd.DataFrame, base_events: pd.DataFrame, stock_counts: Sequence[int]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑥: DSABB 固定参数大样本验证")
    print("=" * 70)
    rows = []
    syms = sorted(df["symbol"].dropna().unique().tolist())
    for n_stocks in stock_counts:
        keep = set(syms[: min(n_stocks, len(syms))])
        g = base_events[base_events["symbol"].isin(keep)]
        if len(g) < 20:
            continue
        row = {"n_stocks": min(n_stocks, len(syms)), "event_n": len(g), "coverage_pct": round(len(g) / len(base_events) * 100, 1) if len(base_events) else np.nan}
        row.update(summarize_events(g, RET_WINDOWS))
        rows.append(row)
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        print(rdf[["n_stocks", "event_n", "coverage_pct", "ret_5", "rr_5", "ret_10", "rr_10", "ret_20", "rr_20"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/dsabb_06_fixed_validation.csv", index=False)
    return rdf


def analyze_07_dsabb_local_grid(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑦: DSABB 邻域稳健性测试")
    print("=" * 70)
    rows = []
    for bb_pos_thr in [0.12, 0.15, 0.18]:
        for vwap_dev_lo in [-1.5, -1.0, -0.5]:
            for dsa_pos_hi in [0.45, 0.55, 0.65]:
                events = build_dsabb_events(
                    df,
                    bb_pos_thr=bb_pos_thr,
                    require_hl_hh=True,
                    vwap_dev_lo=vwap_dev_lo,
                    dsa_pos_hi=dsa_pos_hi,
                    rope_dir_min=0,
                    weekly_dir_required=True,
                    weekly_pos_hi=0.70,
                )
                if len(events) < 20:
                    continue
                row = {
                    "bb_pos_thr": bb_pos_thr,
                    "vwap_dev_lo": vwap_dev_lo,
                    "dsa_pos_hi": dsa_pos_hi,
                    "event_n": len(events),
                }
                row.update(summarize_events(events, RET_WINDOWS))
                rows.append(row)
    if not rows:
        print("[WARN] 邻域稳健性测试无有效事件")
        return pd.DataFrame()
    rdf = pd.DataFrame(rows).sort_values("rr_20", ascending=False)
    if not rdf.empty:
        print(rdf.head(20).to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/dsabb_07_local_grid.csv", index=False)
    return rdf


def analyze_08_dsabb_weekly_filter_compare(events: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑧: DSABB 周线 DSA 过滤对比")
    print("=" * 70)
    variants: List[Tuple[str, pd.Series]] = [("Base", pd.Series(True, index=events.index))]
    if "w_dsa_pivot_pos_01" in events.columns:
        variants += [
            ("W1_wpos_lt_0.7", events["w_dsa_pivot_pos_01"] < 0.7),
            ("W2_wpos_le_0.5", events["w_dsa_pivot_pos_01"] <= 0.5),
            ("W3_wpos_le_0.4", events["w_dsa_pivot_pos_01"] <= 0.4),
        ]
    if "w_DSA_DIR" in events.columns:
        variants += [("W4_wdir_up", events["w_DSA_DIR"] == 1)]
    if {"w_dsa_pivot_pos_01", "w_DSA_DIR"}.issubset(events.columns):
        variants += [
            ("W5_wpos_le_0.7_and_wdir_up", (events["w_dsa_pivot_pos_01"] <= 0.7) & (events["w_DSA_DIR"] == 1)),
            ("W6_wpos_le_0.5_and_wdir_up", (events["w_dsa_pivot_pos_01"] <= 0.5) & (events["w_DSA_DIR"] == 1)),
        ]
    rows = []
    for name, mask in variants:
        g = events[mask.fillna(False)] if hasattr(mask, "fillna") else events[mask]
        if len(g) < 10:
            continue
        row = {"variant": name, "event_n": len(g)}
        row.update(summarize_events(g, RET_WINDOWS))
        rows.append(row)
    rdf = pd.DataFrame(rows).sort_values("rr_20", ascending=False)
    if not rdf.empty:
        print(rdf.to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/dsabb_08_weekly_filter_compare.csv", index=False)
    return rdf


# =========================
# Main
# =========================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DSABB buy point explorer")
    p.add_argument("--n-stocks", type=int, default=300, help="随机抽样股票数")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--freq", default="d", help="K线频率，默认 d")
    p.add_argument("--bars", type=int, default=1200, help="每只股票最多读取 bars 数")
    p.add_argument("--dsabb-bb-thr", type=float, default=0.15, help="默认 DSABB 布林下轨阈值")
    p.add_argument("--dsabb-vwap-dev-lo", type=float, default=-1.0, help="默认 DSABB 日线 signed_vwap_dev_pct 下限")
    p.add_argument("--dsabb-dsa-pos-hi", type=float, default=0.55, help="默认 DSABB 日线 dsa_pivot_pos_01 上限")
    p.add_argument("--dsabb-rope-dir-min", type=int, default=0, help="默认 DSABB rope_dir 最低值")
    p.add_argument("--dsabb-weekly-dir-required", action="store_true", help="默认 DSABB 是否要求 w_DSA_DIR == 1")
    p.add_argument("--dsabb-weekly-pos-hi", type=float, default=0.70, help="默认 DSABB 周线 w_dsa_pivot_pos_01 上限")
    return p


def main() -> None:
    args = build_parser().parse_args()
    stocks = get_stock_pool(args.n_stocks, args.seed)
    all_dfs: List[pd.DataFrame] = []
    print(f"开始处理股票数: {len(stocks)}")
    for i, ts_code in enumerate(stocks, 1):
        raw = load_kline(ts_code, freq=args.freq, bars=args.bars)
        if raw is None or len(raw) < 200:
            continue
        try:
            fac = compute_factors(raw)
            fac = add_structure_context(fac)
            fac = add_forward_returns(fac, RET_WINDOWS)
            fac = fac.reset_index().rename(columns={"index": "datetime"})
            fac["symbol"] = ts_code
            all_dfs.append(fac)
        except Exception as exc:
            print(f"  [WARN] {ts_code} 因子计算失败: {exc}")
        if i % 50 == 0:
            print(f"  已处理 {i}/{len(stocks)}")

    if not all_dfs:
        print("无有效数据，退出")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n总样本: {len(df)}, 股票数: {df['symbol'].nunique()}")
    print("\n基线统计:")
    for w in RET_WINDOWS:
        ret = df[f"ret_{w}"].mean()
        dd = df[f"max_dd_{w}"].mean()
        wr = df[f"win_{w}"].mean()
        print(f"  ret_{w}: 均值={ret:+.5f}, max_dd均值={dd:.5f}, 胜率={wr:.4f}, rr={rr_from_ret_dd(ret, dd):.3f}")

    analyze_03_dsabb_candidate_events(df)
    analyze_04_dsabb_param_scan(df)

    base_events = build_dsabb_events_base(
        df,
        bb_pos_thr=args.dsabb_bb_thr,
        vwap_dev_lo=args.dsabb_vwap_dev_lo,
        dsa_pos_hi=args.dsabb_dsa_pos_hi,
        rope_dir_min=args.dsabb_rope_dir_min,
        weekly_dir_required=bool(args.dsabb_weekly_dir_required),
        weekly_pos_hi=args.dsabb_weekly_pos_hi,
        cooldown=EVENT_DEDUP_BARS,
    )
    if len(base_events) > 0:
        analyze_05_dsabb_time_slices(base_events)
        stock_counts = sorted(set([100, 300, 500, 1000, args.n_stocks]))
        analyze_06_dsabb_fixed_validation(df, base_events, stock_counts)
        analyze_07_dsabb_local_grid(df)
        analyze_08_dsabb_weekly_filter_compare(base_events)
        base_events.to_csv(f"{OUT_DIR}/dsabb_base_events.csv", index=False)
    else:
        print("\n[WARN] 默认 DSABB 参数下无事件，跳过基于 base_events 的分析")

    print(f"\n{'='*70}")
    print(f"# 全部完成! 结果保存在 {OUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
