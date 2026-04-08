# -*- coding: utf-8 -*-
"""
合并版：Dynamic Swing Anchored VWAP + ATR Rope + Bollinger 因子研究脚本（独立可运行）

Purpose
- 直接通过 pytdx 拉取 A 股 K 线（不依赖项目内 datasource 模块）
- 计算 DSA VWAP、ATR Rope、Bollinger 三套指标及其核心因子
- 输出交互式 HTML：主图 K 线 + DSA/ATR/BB，副图逐 bar 显示因子
- 可选输出 CSV，便于后续样本研究/回测

How to Run
    python merged_dsa_atr_rope_bb_factors.py --symbol 600547 --freq d --bars 300 \
        --out merged_600547.html --csv-out merged_600547.csv

    python merged_dsa_atr_rope_bb_factors.py --symbol 300750 --freq 60m --bars 500 \
        --out merged_300750_60m.html

Dependencies
    pip install pandas numpy plotly pytdx

Outputs
- HTML: 主图 + 多个副图逐 bar 显示因子
- CSV : 含所有原始指标和因子字段
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from pytdx.hq import TdxHq_API
    from pytdx.params import TDXParams
except Exception as exc:  # pragma: no cover
    raise RuntimeError("请先安装 pytdx: pip install pytdx") from exc


# =========================
# pytdx data fetch (standalone)
# =========================
SERVERS = [
    ("119.147.212.81", 7709),
    ("119.147.164.60", 7709),
    ("14.215.128.18", 7709),
    ("14.215.128.116", 7709),
    ("101.133.156.38", 7709),
    ("114.80.149.19", 7709),
    ("115.238.90.165", 7709),
    ("123.125.108.23", 7709),
    ("180.153.18.170", 7709),
    ("202.108.253.131", 7709),
]


def normalize_freq(freq: str) -> str:
    f = str(freq).strip().lower()
    if f in {"d", "1d", "day", "daily", "101"}:
        return "d"
    if f in {"w", "1w", "week", "weekly"}:
        return "w"
    if f in {"m", "mo", "month", "monthly"}:
        return "mo"
    if f in {"60", "60m", "1h"}:
        return "60m"
    if f in {"30", "30m"}:
        return "30m"
    if f in {"15", "15m"}:
        return "15m"
    if f in {"5", "5m"}:
        return "5m"
    if f in {"1", "1m"}:
        return "1m"
    raise ValueError(f"不支持的频率: {freq}")


def _category_from_freq(freq: str) -> int:
    f = normalize_freq(freq)
    return {
        "d": TDXParams.KLINE_TYPE_RI_K,
        "w": TDXParams.KLINE_TYPE_WEEKLY,
        "mo": TDXParams.KLINE_TYPE_MONTHLY,
        "60m": TDXParams.KLINE_TYPE_1HOUR,
        "30m": TDXParams.KLINE_TYPE_30MIN,
        "15m": TDXParams.KLINE_TYPE_15MIN,
        "5m": TDXParams.KLINE_TYPE_5MIN,
        "1m": TDXParams.KLINE_TYPE_1MIN,
    }[f]


def _market_from_symbol(symbol: str) -> int:
    return 1 if str(symbol).startswith(("5", "6", "9")) else 0


def connect_pytdx() -> TdxHq_API:
    errors: List[str] = []
    for host, port in SERVERS:
        try:
            api = TdxHq_API(raise_exception=True, auto_retry=True)
            if api.connect(host, port):
                return api
        except Exception as exc:  # pragma: no cover
            errors.append(f"{host}:{port} {exc}")
    raise RuntimeError("pytdx 连接失败: " + "; ".join(errors[-5:]))


def fetch_kline_pytdx(symbol: str, freq: str, count: int) -> pd.DataFrame:
    api = connect_pytdx()
    try:
        cat = _category_from_freq(freq)
        mkt = _market_from_symbol(symbol)
        size = 800
        frames: List[pd.DataFrame] = []
        start = 0
        target = max(int(count), 300)
        while start < target + size:
            recs = api.get_security_bars(cat, mkt, symbol, start, size)
            if not recs:
                break
            d = pd.DataFrame(recs)
            if "datetime" in d.columns:
                d["datetime"] = pd.to_datetime(d["datetime"]).dt.tz_localize(None)
            else:
                d["datetime"] = pd.to_datetime(
                    d[["year", "month", "day", "hour", "minute"]].astype(int)
                )
            if "vol" in d.columns:
                d = d.rename(columns={"vol": "volume"})
            if "amount" not in d.columns:
                d["amount"] = np.nan
            keep = ["datetime", "open", "high", "low", "close", "volume", "amount"]
            frames.append(d[keep].sort_values("datetime"))
            if len(recs) < size:
                break
            start += size
        if not frames:
            raise RuntimeError("pytdx 无数据")
        out = (
            pd.concat(frames)
            .sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
            .tail(count)
            .set_index("datetime")
        )
        return out.astype(float)
    finally:
        try:
            api.disconnect()
        except Exception:
            pass


# =========================
# Common helpers
# =========================
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def pine_rma(src: pd.Series, length: int) -> pd.Series:
    arr = src.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if length <= 0 or len(arr) == 0:
        return pd.Series(out, index=src.index)
    first = min(length, len(arr))
    init = np.nanmean(arr[:first])
    out[first - 1] = init
    prev = init
    alpha = 1.0 / length
    for i in range(first, len(arr)):
        prev = alpha * arr[i] + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=src.index)


def atr_pine(df: pd.DataFrame, length: int) -> pd.Series:
    return pine_rma(true_range(df), length)


def rolling_percentile_last(values: np.ndarray) -> float:
    if len(values) == 0 or np.isnan(values[-1]):
        return np.nan
    arr = values[np.isfinite(values)]
    if len(arr) == 0:
        return np.nan
    last = values[-1]
    return float(np.sum(arr <= last) / len(arr))


def clip01(x: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(np.clip(np.asarray(x, dtype=float), 0.0, 1.0))


# =========================
# DSA VWAP
# =========================
@dataclass
class DSAConfig:
    prd: int = 50
    base_apt: float = 20.0
    use_adapt: bool = False
    vol_bias: float = 10.0
    atr_len: int = 50


def hlc3(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def alpha_from_apt(apt: float) -> float:
    apt = max(1.0, float(apt))
    decay = np.exp(-np.log(2.0) / apt)
    return float(1.0 - decay)


def compute_dsa(df: pd.DataFrame, cfg: DSAConfig) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    d = df.copy()
    d["hlc3"] = hlc3(d)
    atr = atr_pine(d, cfg.atr_len)
    atr_avg = pine_rma(atr, cfg.atr_len)
    ratio = np.where(atr_avg.to_numpy() > 0, atr.to_numpy() / atr_avg.to_numpy(), 1.0)
    if cfg.use_adapt:
        apt_raw = cfg.base_apt / np.power(ratio, cfg.vol_bias)
    else:
        apt_raw = np.full(len(d), cfg.base_apt, dtype=float)
    apt_series = np.rint(np.clip(apt_raw, 5.0, 300.0)).astype(int)

    n = len(d)
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    close = d["close"].to_numpy(float)
    vol = d["volume"].replace(0, np.nan).ffill().fillna(1.0).to_numpy(float)
    h3 = d["hlc3"].to_numpy(float)

    ph = np.nan
    pl = np.nan
    phL = 0
    plL = 0
    prev = np.nan
    ph_prev_store = np.nan
    pl_prev_store = np.nan
    p = h3[0] * vol[0]
    v = vol[0]
    last_dir: Optional[int] = None

    vwap_out = np.full(n, np.nan)
    dir_out = np.full(n, np.nan)
    dsa_pivot_high = np.full(n, np.nan)
    dsa_pivot_low = np.full(n, np.nan)
    dsa_pivot_pos = np.full(n, np.nan)
    signed_vwap_dev_pct = np.full(n, np.nan)
    bull_vwap_dev_pct = np.zeros(n, dtype=float)
    bear_vwap_dev_pct = np.zeros(n, dtype=float)
    trend_aligned_vwap_dev_pct = np.zeros(n, dtype=float)
    lh_hh_low_pos = np.full(n, np.nan)
    last_pivot_type = np.array([""] * n, dtype=object)

    segments: List[Dict] = []
    cur_points_x: List[pd.Timestamp] = []
    cur_points_y: List[float] = []
    pivot_labels: List[Dict] = []

    last_lh_hh_idx: Optional[int] = None
    last_lh_hh_price = np.nan
    recent_pivot_high_price = np.nan
    recent_pivot_low_price = np.nan
    recent_pivot_high_idx: Optional[int] = None
    recent_pivot_low_idx: Optional[int] = None
    latest_label_type = ""

    for t in range(n):
        st = max(0, t - cfg.prd + 1)
        win_h = high[st:t + 1]
        win_l = low[st:t + 1]

        if np.isfinite(high[t]) and high[t] == np.nanmax(win_h):
            ph = high[t]
            phL = t
            recent_pivot_high_price = ph
            recent_pivot_high_idx = t
        if np.isfinite(low[t]) and low[t] == np.nanmin(win_l):
            pl = low[t]
            plL = t
            recent_pivot_low_price = pl
            recent_pivot_low_idx = t

        dir_ = 1 if phL > plL else -1
        dir_out[t] = dir_
        dsa_pivot_high[t] = recent_pivot_high_price
        dsa_pivot_low[t] = recent_pivot_low_price

        if last_dir is None:
            last_dir = dir_

        if dir_ != last_dir:
            if len(cur_points_x) >= 2:
                segments.append({"dir": int(last_dir), "x": np.array(cur_points_x), "y": np.array(cur_points_y, dtype=float)})

            x_anchor = plL if dir_ > 0 else phL
            y_anchor = pl if dir_ > 0 else ph

            txt = ""
            if dir_ > 0:
                if np.isfinite(prev):
                    if np.isfinite(y_anchor) and y_anchor < prev:
                        txt = "LL"
                    elif np.isfinite(y_anchor) and y_anchor > prev:
                        txt = "HL"
            else:
                if np.isfinite(prev):
                    if np.isfinite(y_anchor) and y_anchor < prev:
                        txt = "LH"
                    elif np.isfinite(y_anchor) and y_anchor > prev:
                        txt = "HH"

            latest_label_type = txt
            if txt in {"LH", "HH"}:
                last_lh_hh_idx = x_anchor
                last_lh_hh_price = y_anchor

            pivot_labels.append({"t": int(x_anchor), "x": d.index[x_anchor], "y": float(y_anchor), "text": txt, "dir": int(dir_)})
            prev = ph_prev_store if dir_ > 0 else pl_prev_store

            p = y_anchor * vol[x_anchor]
            v = vol[x_anchor]
            cur_points_x = []
            cur_points_y = []
            for k in range(x_anchor, t + 1):
                alpha = alpha_from_apt(float(apt_series[k]))
                p = (1.0 - alpha) * p + alpha * (h3[k] * vol[k])
                v = (1.0 - alpha) * v + alpha * vol[k]
                vv = (p / v) if v > 0 else np.nan
                vwap_out[k] = vv
                cur_points_x.append(d.index[k])
                cur_points_y.append(vv)
            last_dir = dir_
        else:
            alpha = alpha_from_apt(float(apt_series[t]))
            p = (1.0 - alpha) * p + alpha * (h3[t] * vol[t])
            v = (1.0 - alpha) * v + alpha * vol[t]
            vv = (p / v) if v > 0 else np.nan
            vwap_out[t] = vv
            cur_points_x.append(d.index[t])
            cur_points_y.append(vv)

        ph_prev_store = ph
        pl_prev_store = pl
        last_pivot_type[t] = latest_label_type

        if np.isfinite(recent_pivot_high_price) and np.isfinite(recent_pivot_low_price):
            lo = min(recent_pivot_low_price, recent_pivot_high_price)
            hi = max(recent_pivot_low_price, recent_pivot_high_price)
            if hi > lo:
                dsa_pivot_pos[t] = np.clip((close[t] - lo) / (hi - lo), 0.0, 1.0)

        if np.isfinite(vwap_out[t]) and vwap_out[t] != 0:
            signed_vwap_dev_pct[t] = (close[t] - vwap_out[t]) / vwap_out[t] * 100.0
            if dir_ > 0:
                bull_vwap_dev_pct[t] = signed_vwap_dev_pct[t]
                trend_aligned_vwap_dev_pct[t] = signed_vwap_dev_pct[t]
            elif dir_ < 0:
                bear_vwap_dev_pct[t] = signed_vwap_dev_pct[t]
                trend_aligned_vwap_dev_pct[t] = -signed_vwap_dev_pct[t]

        if last_lh_hh_idx is not None and np.isfinite(last_lh_hh_price):
            window_low = float(np.nanmin(low[last_lh_hh_idx:t + 1]))
            denom = last_lh_hh_price - window_low
            if np.isfinite(window_low) and denom > 0:
                lh_hh_low_pos[t] = np.clip((close[t] - window_low) / denom, 0.0, 1.0)

    if len(cur_points_x) >= 2:
        segments.append({"dir": int(last_dir if last_dir is not None else -1), "x": np.array(cur_points_x), "y": np.array(cur_points_y, dtype=float)})

    # 回看确认标签：
    # 1) 先完整计算出全样本 DSA_DIR；
    # 2) 再按连续方向段回看确认该段的最终极值；
    # 3) 将结果回填到原来的 pivot_labels / last_pivot_type，
    #    保持 HTML 输出层完全不改。
    hindsight_labels: List[Dict] = []
    hindsight_last_pivot_type = np.array([""] * n, dtype=object)
    prev_confirmed_up_bars_arr = np.full(n, np.nan)
    prev_confirmed_down_bars_arr = np.full(n, np.nan)
    last_confirmed_run_bars_arr = np.full(n, np.nan)
    current_run_bars_arr = np.full(n, np.nan)
    runs: List[Tuple[int, int, int]] = []
    if n > 0:
        run_start = 0
        for i in range(1, n):
            curr_dir = int(dir_out[i]) if np.isfinite(dir_out[i]) else 0
            prev_dir2 = int(dir_out[i - 1]) if np.isfinite(dir_out[i - 1]) else 0
            if curr_dir != prev_dir2:
                if prev_dir2 != 0:
                    runs.append((run_start, i - 1, prev_dir2))
                run_start = i
        tail_dir = int(dir_out[n - 1]) if np.isfinite(dir_out[n - 1]) else 0
        if tail_dir != 0:
            runs.append((run_start, n - 1, tail_dir))

    prev_hindsight_high = np.nan
    prev_hindsight_low = np.nan
    latest_hindsight_type = ""
    latest_confirmed_up_bars = np.nan
    latest_confirmed_down_bars = np.nan
    latest_confirmed_run_bars = np.nan
    for st_run, ed_run, run_dir in runs:
        if st_run > ed_run:
            continue

        if run_dir > 0:
            seg = high[st_run:ed_run + 1]
            if len(seg) == 0 or np.all(~np.isfinite(seg)):
                continue
            rel_idx = int(np.nanargmax(seg))
            pivot_idx = st_run + rel_idx
            pivot_price = float(high[pivot_idx])
            txt = ""
            if np.isfinite(prev_hindsight_high):
                txt = "HH" if pivot_price > prev_hindsight_high else "LH"
            prev_hindsight_high = pivot_price
        else:
            seg = low[st_run:ed_run + 1]
            if len(seg) == 0 or np.all(~np.isfinite(seg)):
                continue
            rel_idx = int(np.nanargmin(seg))
            pivot_idx = st_run + rel_idx
            pivot_price = float(low[pivot_idx])
            txt = ""
            if np.isfinite(prev_hindsight_low):
                txt = "HL" if pivot_price > prev_hindsight_low else "LL"
            prev_hindsight_low = pivot_price

        if txt:
            hindsight_labels.append({
                "t": int(pivot_idx),
                "x": d.index[pivot_idx],
                "y": pivot_price,
                "text": txt,
                "dir": int(run_dir),
            })
            latest_hindsight_type = txt

        run_bars = float(ed_run - st_run + 1)
        current_run_bars_arr[st_run:ed_run + 1] = np.arange(1, ed_run - st_run + 2, dtype=float)
        prev_confirmed_up_bars_arr[st_run:ed_run + 1] = latest_confirmed_up_bars
        prev_confirmed_down_bars_arr[st_run:ed_run + 1] = latest_confirmed_down_bars
        last_confirmed_run_bars_arr[st_run:ed_run + 1] = latest_confirmed_run_bars

        if latest_hindsight_type:
            hindsight_last_pivot_type[st_run:ed_run + 1] = latest_hindsight_type

        if run_dir > 0:
            latest_confirmed_up_bars = run_bars
        else:
            latest_confirmed_down_bars = run_bars
        latest_confirmed_run_bars = run_bars

    # 关键：回填到原来的变量名，保持 build_figure() 完全不需要改。
    pivot_labels = hindsight_labels
    last_pivot_type = hindsight_last_pivot_type

    # 使用回看确认的枢轴点重新计算 dsa_pivot_high/low 和 dsa_pivot_pos_01
    # 删除 50 日窗口限制，使用已确认的 HH/HL/LH/LL
    hindsight_high_arr = np.full(n, np.nan)
    hindsight_low_arr = np.full(n, np.nan)
    hindsight_pos_arr = np.full(n, np.nan)

    # 从 hindsight_labels 中提取已确认的高点和低点
    confirmed_highs: List[Tuple[int, float]] = []  # (index, price)
    confirmed_lows: List[Tuple[int, float]] = []   # (index, price)

    for label in hindsight_labels:
        if label["text"] in {"HH", "LH"}:
            confirmed_highs.append((label["t"], label["y"]))
        elif label["text"] in {"HL", "LL"}:
            confirmed_lows.append((label["t"], label["y"]))

    # 对每个时间点，找到最近的已确认高点和低点
    for t in range(n):
        # 找到最近的已确认高点（在当前或之前）
        recent_high = np.nan
        for idx, price in reversed(confirmed_highs):
            if idx <= t:
                recent_high = price
                break

        # 找到最近的已确认低点（在当前或之前）
        recent_low = np.nan
        for idx, price in reversed(confirmed_lows):
            if idx <= t:
                recent_low = price
                break

        hindsight_high_arr[t] = recent_high
        hindsight_low_arr[t] = recent_low

        # 重新计算位置因子
        if np.isfinite(recent_high) and np.isfinite(recent_low):
            lo = min(recent_low, recent_high)
            hi = max(recent_low, recent_high)
            if hi > lo:
                hindsight_pos_arr[t] = np.clip((close[t] - lo) / (hi - lo), 0.0, 1.0)

    out = pd.DataFrame(index=df.index)
    out["DSA_VWAP"] = vwap_out
    out["DSA_DIR"] = dir_out
    out["dsa_pivot_high"] = hindsight_high_arr  # 使用回看确认的高点
    out["dsa_pivot_low"] = hindsight_low_arr    # 使用回看确认的低点
    out["dsa_pivot_pos_01"] = hindsight_pos_arr  # 使用回看确认的位置因子
    out["signed_vwap_dev_pct"] = signed_vwap_dev_pct
    out["bull_vwap_dev_pct"] = bull_vwap_dev_pct
    out["bear_vwap_dev_pct"] = bear_vwap_dev_pct
    out["trend_aligned_vwap_dev_pct"] = trend_aligned_vwap_dev_pct
    out["lh_hh_low_pos"] = lh_hh_low_pos
    out["last_pivot_type"] = last_pivot_type
    out["prev_confirmed_up_bars"] = prev_confirmed_up_bars_arr
    out["prev_confirmed_down_bars"] = prev_confirmed_down_bars_arr
    out["last_confirmed_run_bars"] = last_confirmed_run_bars_arr
    out["current_run_bars"] = current_run_bars_arr
    return out, pivot_labels, segments


# =========================
# ATR Rope
# =========================
@dataclass
class RopeConfig:
    length: int = 14
    multi: float = 1.5
    source: str = "close"


def pine_cross(curr_a: float, curr_b: float, prev_a: float, prev_b: float) -> bool:
    if any(pd.isna(v) for v in (curr_a, curr_b, prev_a, prev_b)):
        return False
    curr_rel = curr_a - curr_b
    prev_rel = prev_a - prev_b
    return bool((curr_rel > 0 and prev_rel <= 0) or (curr_rel < 0 and prev_rel >= 0))


def segment_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def compute_atr_rope(df: pd.DataFrame, cfg: RopeConfig) -> pd.DataFrame:
    out = df.copy()
    out["src"] = out[cfg.source].astype(float)
    out["atr_raw"] = atr_pine(out, cfg.length)
    out["atr"] = out["atr_raw"] * cfg.multi

    src = out["src"].to_numpy(float)
    atr = out["atr"].to_numpy(float)
    close = out["close"].to_numpy(float)
    high = out["high"].to_numpy(float)
    low = out["low"].to_numpy(float)
    n = len(out)

    rope_arr = np.full(n, np.nan)
    upper_arr = np.full(n, np.nan)
    lower_arr = np.full(n, np.nan)
    dir_arr = np.full(n, np.nan)
    ff_arr = np.full(n, np.nan)
    chi_arr = np.full(n, np.nan)
    clo_arr = np.full(n, np.nan)
    bars_since_dir_change = np.full(n, np.nan)
    cons_bars_arr = np.full(n, np.nan)
    is_cons_arr = np.full(n, np.nan)
    dist_rope_arr = np.full(n, np.nan)
    dist_upper_arr = np.full(n, np.nan)
    dist_lower_arr = np.full(n, np.nan)
    slope_atr_arr = np.full(n, np.nan)
    width_atr_arr = np.full(n, np.nan)
    break_up_arr = np.zeros(n, dtype=float)
    break_dn_arr = np.zeros(n, dtype=float)
    break_up_strength_arr = np.full(n, np.nan)
    break_dn_strength_arr = np.full(n, np.nan)
    channel_pos_arr = np.full(n, np.nan)
    range_pos_arr = np.full(n, np.nan)
    rope_pivot_pos_arr = np.full(n, np.nan)
    last_pivot_low_arr = np.full(n, np.nan)
    last_pivot_high_arr = np.full(n, np.nan)

    rope = math.nan
    dir_val = 0
    last_dir_change_idx: Optional[int] = None
    segment_start_idx = 0
    c_hi = math.nan
    c_lo = math.nan
    h_sum = 0.0
    l_sum = 0.0
    c_count = 0
    ff = True
    last_pivot_low = np.nan
    last_pivot_high = np.nan
    highest_since_pivot = np.nan
    lowest_since_pivot = np.nan

    for i in range(n):
        src_i = src[i]
        atr_i = atr[i]
        if pd.isna(src_i):
            continue
        if pd.isna(rope):
            rope = float(src_i)

        move = float(src_i) - rope
        rope = rope + max(abs(move) - (0.0 if pd.isna(atr_i) else atr_i), 0.0) * float(np.sign(move))
        upper = rope + atr_i if pd.notna(atr_i) else np.nan
        lower = rope - atr_i if pd.notna(atr_i) else np.nan

        prev_dir = dir_val
        prev_rope = rope_arr[i - 1] if i > 0 else np.nan
        if i > 0 and pd.notna(prev_rope):
            if rope > prev_rope:
                dir_val = 1
            elif rope < prev_rope:
                dir_val = -1
        if i > 0 and pine_cross(src_i, rope, src[i - 1], prev_rope):
            dir_val = 0

        if i == 0 or dir_val != prev_dir:
            if i > 0:
                if dir_val == 1:
                    last_pivot_low = float(np.nanmin(low[segment_start_idx:i + 1]))
                    highest_since_pivot = high[i]
                    lowest_since_pivot = np.nan
                elif dir_val == -1:
                    last_pivot_high = float(np.nanmax(high[segment_start_idx:i + 1]))
                    lowest_since_pivot = low[i]
                    highest_since_pivot = np.nan
            segment_start_idx = i
            last_dir_change_idx = i

        if dir_val == 0:
            if prev_dir != 0:
                h_sum = 0.0
                l_sum = 0.0
                c_count = 0
                ff = not ff
            if pd.notna(upper) and pd.notna(lower):
                h_sum += upper
                l_sum += lower
                c_count += 1
                c_hi = h_sum / c_count
                c_lo = l_sum / c_count
        
        if dir_val == 1:
            highest_since_pivot = np.nanmax([highest_since_pivot, high[i]]) if np.isfinite(highest_since_pivot) else high[i]
        elif dir_val == -1:
            lowest_since_pivot = np.nanmin([lowest_since_pivot, low[i]]) if np.isfinite(lowest_since_pivot) else low[i]

        rope_arr[i] = rope
        upper_arr[i] = upper
        lower_arr[i] = lower
        dir_arr[i] = dir_val
        ff_arr[i] = 1.0 if ff else 0.0
        chi_arr[i] = c_hi
        clo_arr[i] = c_lo
        is_cons_arr[i] = 1.0 if dir_val == 0 else 0.0
        cons_bars_arr[i] = float(c_count) if dir_val == 0 else 0.0
        bars_since_dir_change[i] = float(i - last_dir_change_idx) if last_dir_change_idx is not None else np.nan
        last_pivot_low_arr[i] = last_pivot_low
        last_pivot_high_arr[i] = last_pivot_high

        if pd.notna(atr_i) and atr_i != 0:
            dist_rope_arr[i] = (close[i] - rope) / atr_i
            dist_upper_arr[i] = (close[i] - upper) / atr_i if pd.notna(upper) else np.nan
            dist_lower_arr[i] = (close[i] - lower) / atr_i if pd.notna(lower) else np.nan
            if i >= 5 and pd.notna(rope_arr[i - 5]):
                slope_atr_arr[i] = (rope - rope_arr[i - 5]) / atr_i
            if pd.notna(c_hi) and pd.notna(c_lo):
                width_atr_arr[i] = (c_hi - c_lo) / atr_i
                if i > 0 and pd.notna(chi_arr[i - 1]) and pd.notna(clo_arr[i - 1]):
                    up_break = close[i] > c_hi and close[i - 1] <= chi_arr[i - 1]
                    dn_break = close[i] < c_lo and close[i - 1] >= clo_arr[i - 1]
                    if up_break:
                        break_up_arr[i] = 1.0
                        break_up_strength_arr[i] = (close[i] - c_hi) / atr_i
                    if dn_break:
                        break_dn_arr[i] = 1.0
                        break_dn_strength_arr[i] = (c_lo - close[i]) / atr_i
            if pd.notna(upper) and pd.notna(lower) and upper > lower:
                channel_pos_arr[i] = np.clip((close[i] - lower) / (upper - lower), 0.0, 1.0)
        if pd.notna(c_hi) and pd.notna(c_lo) and c_hi > c_lo:
            range_pos_arr[i] = np.clip((close[i] - c_lo) / (c_hi - c_lo), 0.0, 1.0)

        if dir_val == 1 and np.isfinite(last_pivot_low) and np.isfinite(highest_since_pivot) and highest_since_pivot > last_pivot_low:
            rope_pivot_pos_arr[i] = np.clip((close[i] - last_pivot_low) / (highest_since_pivot - last_pivot_low), 0.0, 1.0)
        elif dir_val == -1 and np.isfinite(last_pivot_high) and np.isfinite(lowest_since_pivot) and last_pivot_high > lowest_since_pivot:
            rope_pivot_pos_arr[i] = np.clip((last_pivot_high - close[i]) / (last_pivot_high - lowest_since_pivot), 0.0, 1.0)

    out["rope"] = rope_arr
    out["upper"] = upper_arr
    out["lower"] = lower_arr
    out["rope_dir"] = dir_arr
    out["ff"] = ff_arr
    out["c_hi"] = chi_arr
    out["c_lo"] = clo_arr
    out["is_consolidating"] = is_cons_arr
    out["bars_since_dir_change"] = bars_since_dir_change
    out["consolidation_bars"] = cons_bars_arr
    out["dist_to_rope_atr"] = dist_rope_arr
    out["dist_to_upper_atr"] = dist_upper_arr
    out["dist_to_lower_atr"] = dist_lower_arr
    out["rope_slope_atr_5"] = slope_atr_arr
    out["range_width_atr"] = width_atr_arr
    out["range_break_up"] = break_up_arr
    out["range_break_down"] = break_dn_arr
    out["range_break_up_strength"] = break_up_strength_arr
    out["range_break_down_strength"] = break_dn_strength_arr
    out["channel_pos_01"] = channel_pos_arr
    out["range_pos_01"] = range_pos_arr
    out["rope_pivot_pos_01"] = rope_pivot_pos_arr
    out["last_rope_pivot_low"] = last_pivot_low_arr
    out["last_rope_pivot_high"] = last_pivot_high_arr
    out["range_width_atr_ffill"] = out["range_width_atr"].ffill()
    out["c_hi_ffill"] = out["c_hi"].ffill()
    out["c_lo_ffill"] = out["c_lo"].ffill()

    out["rope_up"] = np.where(out["rope_dir"] > 0, out["rope"], np.nan)
    out["rope_down"] = np.where(out["rope_dir"] < 0, out["rope"], np.nan)
    out["rope_flat"] = np.where(out["rope_dir"] == 0, out["rope"], np.nan)
    out["range_high_1"] = np.where(out["ff"] == 1.0, out["c_hi"], np.nan)
    out["range_low_1"] = np.where(out["ff"] == 1.0, out["c_lo"], np.nan)
    out["range_high_2"] = np.where(out["ff"] == 0.0, out["c_hi"], np.nan)
    out["range_low_2"] = np.where(out["ff"] == 0.0, out["c_lo"], np.nan)
    return out


# =========================
# Bollinger
# =========================
def _run_streak(mask: pd.Series) -> pd.Series:
    vals = mask.fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(vals), dtype=float)
    streak = 0
    for i, flag in enumerate(vals):
        if flag:
            streak += 1
        else:
            streak = 0
        out[i] = float(streak)
    return pd.Series(out, index=mask.index)


def compute_bollinger(df: pd.DataFrame, length: int = 20, mult: float = 2.0, pct_lookback: int = 120) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    mid = df["close"].rolling(length, min_periods=length).mean()
    std = df["close"].rolling(length, min_periods=length).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    width_norm = (upper - lower) / mid.replace(0, np.nan)
    pos = (df["close"] - lower) / (upper - lower)
    pos = pos.clip(0.0, 1.0)
    width_pctile = width_norm.rolling(pct_lookback, min_periods=max(20, pct_lookback // 4)).apply(rolling_percentile_last, raw=True)
    width_change_5 = width_norm / width_norm.shift(5) - 1.0
    width_vs_ma = width_norm / width_norm.rolling(20, min_periods=10).mean()
    bb_expanding = (width_change_5 > 0).astype(float)
    bb_contracting = (width_change_5 < 0).astype(float)
    bb_expand_streak = _run_streak(width_change_5 > 0)
    bb_contract_streak = _run_streak(width_change_5 < 0)

    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bb_pos_01"] = pos
    out["bb_width_norm"] = width_norm
    out["bb_width_percentile"] = width_pctile
    out["bb_width_change_5"] = width_change_5
    out["bb_width_vs_ma"] = width_vs_ma
    out["bb_expanding"] = bb_expanding
    out["bb_contracting"] = bb_contracting
    out["bb_expand_streak"] = bb_expand_streak
    out["bb_contract_streak"] = bb_contract_streak
    return out


# =========================
# plotting
# =========================
UP_COL = "#3daa45"
DOWN_COL = "#ff033e"
FLAT_COL = "#004d92"
RANGE_LINE_COL = "rgba(0,114,230,0.95)"
RANGE_FILL_COL = "rgba(0,77,146,0.18)"
BB_COL = "#f5c542"
ZERO_COL = "#888888"


def add_range_segments(fig: go.Figure, x_num: np.ndarray, df: pd.DataFrame, hi_col: str, lo_col: str, row: int) -> None:
    hi_mask = df[hi_col].notna().to_numpy()
    first = True
    for st, ed in segment_runs(hi_mask):
        x = x_num[st:ed + 1]
        hi_y = df[hi_col].iloc[st:ed + 1]
        lo_y = df[lo_col].iloc[st:ed + 1]
        fig.add_trace(go.Scatter(x=x, y=hi_y, mode="lines", line=dict(width=1.2, color=RANGE_LINE_COL), name="range", legendgroup="range", showlegend=first, connectgaps=False, hovertemplate="range_high=%{y:.4f}<extra></extra>"), row=row, col=1)
        fig.add_trace(go.Scatter(x=x, y=lo_y, mode="lines", line=dict(width=1.2, color=RANGE_LINE_COL), fill="tonexty", fillcolor=RANGE_FILL_COL, name="range_fill", legendgroup="range", showlegend=False, connectgaps=False, hovertemplate="range_low=%{y:.4f}<extra></extra>"), row=row, col=1)
        first = False


def build_figure(df: pd.DataFrame, dsa_segments: List[Dict], dsa_labels: List[Dict], out_html: str, title: str) -> None:
    x_num = np.arange(len(df), dtype=float)
    tick_text = [ts.strftime("%Y-%m-%d %H:%M") if (len(df.index) > 1 and (df.index[1] - df.index[0]) < pd.Timedelta("20H")) else ts.strftime("%Y-%m-%d") for ts in df.index]

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.40, 0.10, 0.12, 0.12, 0.13, 0.13],
        subplot_titles=(
            title,
            "趋势状态",
            "位置因子(0-1)",
            "偏离/强弱",
            "波动/宽度",
            "触发/阶段",
        ),
    )

    # main price panel
    fig.add_trace(go.Candlestick(x=x_num, open=df["open"], high=df["high"], low=df["low"], close=df["close"], increasing_line_color="#00c2a0", decreasing_line_color="#ff4d5a", increasing_fillcolor="#00c2a0", decreasing_fillcolor="#ff4d5a", name="K线", showlegend=False), row=1, col=1)
    for seg in dsa_segments:
        seg_x = pd.to_datetime(seg["x"])
        mask = (seg_x >= df.index.min()) & (seg_x <= df.index.max())
        if mask.sum() < 2:
            continue
        seg_idx = [df.index.get_indexer([pd.Timestamp(x)])[0] for x in seg_x[mask] if x in df.index]
        if len(seg_idx) < 2:
            continue
        seg_y = np.asarray(seg["y"])[mask]
        color = "#ff1744" if seg["dir"] > 0 else "#00e676"
        fig.add_trace(go.Scatter(x=np.array(seg_idx, dtype=float), y=seg_y[:len(seg_idx)], mode="lines", line=dict(width=2, color=color), name="DSA_VWAP", showlegend=False, hoverinfo="skip"), row=1, col=1)

    for col_name, color, name, lg in [("rope_up", UP_COL, "ATR Rope", True), ("rope_down", DOWN_COL, "ATR Rope", False), ("rope_flat", FLAT_COL, "ATR Rope", False)]:
        fig.add_trace(go.Scatter(x=x_num, y=df[col_name], mode="lines", line=dict(width=2.4, color=color), connectgaps=False, name=name, legendgroup="rope", showlegend=lg), row=1, col=1)

    fig.add_trace(go.Scatter(x=x_num, y=df["bb_upper"], mode="lines", line=dict(width=1.0, color=BB_COL, dash="dot"), name="BB", legendgroup="bb", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["bb_lower"], mode="lines", line=dict(width=1.0, color=BB_COL, dash="dot"), fill="tonexty", fillcolor="rgba(245,197,66,0.08)", name="BB", legendgroup="bb", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["bb_mid"], mode="lines", line=dict(width=1.0, color=BB_COL), name="BB_mid", showlegend=False), row=1, col=1)

    add_range_segments(fig, x_num, df, "range_high_1", "range_low_1", 1)
    add_range_segments(fig, x_num, df, "range_high_2", "range_low_2", 1)

    up_mask = df["range_break_up"] == 1
    dn_mask = df["range_break_down"] == 1
    fig.add_trace(go.Scatter(x=x_num[up_mask.to_numpy()], y=df.loc[up_mask, "close"], mode="markers", marker=dict(symbol="circle-open", size=8, color="#2f7cff", line=dict(width=1.5)), name="break_up", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_num[dn_mask.to_numpy()], y=df.loc[dn_mask, "close"], mode="markers", marker=dict(symbol="circle-open", size=8, color="#ffaa00", line=dict(width=1.5)), name="break_down", showlegend=True), row=1, col=1)

    for lab in dsa_labels:
        if lab["x"] not in df.index or not lab.get("text"):
            continue
        idx = df.index.get_loc(lab["x"])
        is_up = lab["dir"] > 0
        fig.add_annotation(x=idx, y=lab["y"], xref="x", yref="y", text=lab["text"], showarrow=True, arrowhead=2, ax=0, ay=25 if is_up else -25, bgcolor="rgba(0,230,118,0.85)" if is_up else "rgba(255,23,68,0.85)", font=dict(color="white", size=11), row=1, col=1)

    # state panel
    for nm in ["DSA_DIR", "rope_dir", "bars_since_dir_change", "is_consolidating", "consolidation_bars"]:
        fig.add_trace(go.Scatter(x=x_num, y=df[nm], mode="lines", line=dict(width=1.3), name=nm, showlegend=False), row=2, col=1)

    # positions panel
    for nm in ["dsa_pivot_pos_01", "lh_hh_low_pos", "channel_pos_01", "range_pos_01", "rope_pivot_pos_01", "bb_pos_01"]:
        fig.add_trace(go.Scatter(x=x_num, y=df[nm], mode="lines", line=dict(width=1.35), name=nm, showlegend=False), row=3, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=3, col=1)
    fig.add_hline(y=0.5, line_width=1, line_dash="dot", line_color=ZERO_COL, row=3, col=1)
    fig.add_hline(y=1.0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=3, col=1)

    # deviations panel
    for nm in ["signed_vwap_dev_pct", "bull_vwap_dev_pct", "bear_vwap_dev_pct", "trend_aligned_vwap_dev_pct", "dist_to_rope_atr"]:
        fig.add_trace(go.Scatter(x=x_num, y=df[nm], mode="lines", line=dict(width=1.35), name=nm, showlegend=False), row=4, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=4, col=1)

    # volatility panel
    for nm in ["range_width_atr_ffill", "bb_width_norm", "bb_width_percentile", "bb_width_change_5", "bb_width_vs_ma", "bb_expand_streak", "bb_contract_streak"]:
        fig.add_trace(go.Scatter(x=x_num, y=df[nm], mode="lines", line=dict(width=1.35), name=nm, showlegend=False), row=5, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=5, col=1)
    fig.add_hline(y=0.5, line_width=1, line_dash="dot", line_color=ZERO_COL, row=5, col=1)
    fig.add_hline(y=1.0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=5, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["bb_expanding"], mode="lines", line=dict(width=1.1, dash="dot"), name="bb_expanding", showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["bb_contracting"], mode="lines", line=dict(width=1.1, dash="dot"), name="bb_contracting", showlegend=False), row=5, col=1)

    # trigger/stage panel
    trigger_cols = ["rope_slope_atr_5", "range_break_up", "range_break_up_strength"]
    for nm in trigger_cols:
        fig.add_trace(go.Scatter(x=x_num, y=df[nm], mode="lines", line=dict(width=1.35), name=nm, showlegend=False), row=6, col=1)
    pivot_code_map = {"": np.nan, "HH": 2.0, "HL": 1.0, "LH": -1.0, "LL": -2.0}
    pivot_code = df["last_pivot_type"].map(pivot_code_map).astype(float)
    hover_text = [f"last_pivot_type={v}" if isinstance(v, str) and v else "last_pivot_type=" for v in df["last_pivot_type"].tolist()]
    fig.add_trace(
        go.Scatter(
            x=x_num,
            y=pivot_code,
            mode="lines+markers",
            line=dict(width=1.1, dash="dot"),
            marker=dict(size=5),
            name="last_pivot_type",
            showlegend=False,
            text=hover_text,
            hovertemplate="%{text}<br>pivot_code=%{y}<extra></extra>",
        ),
        row=6,
        col=1,
    )
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=6, col=1)

    tick_step = max(1, len(df) // 10)
    tickvals = list(range(0, len(df), tick_step))
    if tickvals[-1] != len(df) - 1:
        tickvals.append(len(df) - 1)
    ticktext = [tick_text[i] for i in tickvals]

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=40, r=20, t=80, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01), height=1750)
    for r in [1, 2, 3, 4, 5, 6]:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True, zeroline=False, row=r, col=1)
        fig.update_yaxes(showgrid=True, zeroline=False, row=r, col=1)
    fig.update_yaxes(range=[-2.4, 2.4], tickmode="array", tickvals=[-2,-1,0,1,2], ticktext=["LL","LH","0","HL","HH"], row=6, col=1)
    fig.write_html(out_html, include_plotlyjs="cdn")


# =========================
# main
# =========================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merged DSA VWAP + ATR Rope + Bollinger factor viewer")
    p.add_argument("--symbol", required=True, help="A股代码，如 600547")
    p.add_argument("--freq", default="d", help="d/w/mo/1m/5m/15m/30m/60m")
    p.add_argument("--bars", type=int, default=300, help="展示最近 N 根K线")
    p.add_argument("--fetch-bars", type=int, default=1200, help="实际抓取并用于计算的K线数量，建议 > bars")
    p.add_argument("--out", default="merged_factors.html", help="输出HTML文件")
    p.add_argument("--csv-out", default="", help="可选：输出CSV文件")
    p.add_argument("--dsa-prd", type=int, default=50)
    p.add_argument("--dsa-base-apt", type=float, default=20.0)
    p.add_argument("--dsa-use-adapt", action="store_true")
    p.add_argument("--dsa-vol-bias", type=float, default=10.0)
    p.add_argument("--rope-len", type=int, default=14)
    p.add_argument("--rope-multi", type=float, default=1.5)
    p.add_argument("--rope-source", default="close", choices=["open", "high", "low", "close"])
    p.add_argument("--bb-len", type=int, default=20)
    p.add_argument("--bb-mult", type=float, default=2.0)
    p.add_argument("--bb-pct-lookback", type=int, default=120)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.freq = normalize_freq(args.freq)
    fetch_bars = max(args.fetch_bars, args.bars, 300)
    df = fetch_kline_pytdx(args.symbol, args.freq, fetch_bars)

    dsa_df, dsa_labels, dsa_segments = compute_dsa(
        df,
        DSAConfig(
            prd=args.dsa_prd,
            base_apt=args.dsa_base_apt,
            use_adapt=bool(args.dsa_use_adapt),
            vol_bias=args.dsa_vol_bias,
        ),
    )
    rope_df = compute_atr_rope(df, RopeConfig(length=args.rope_len, multi=args.rope_multi, source=args.rope_source))
    bb_df = compute_bollinger(df, length=args.bb_len, mult=args.bb_mult, pct_lookback=args.bb_pct_lookback)

    merged = pd.concat([df, dsa_df, rope_df.drop(columns=df.columns, errors="ignore"), bb_df], axis=1)
    merged = merged.tail(args.bars).copy()

    title = f"{args.symbol} [{args.freq}] DSA VWAP + ATR Rope + Bollinger Factors"
    build_figure(merged, dsa_segments, dsa_labels, args.out, title)

    if args.csv_out:
        merged.to_csv(args.csv_out, encoding="utf-8-sig")
        print(f"CSV 已生成: {args.csv_out}")
    print(f"HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()
