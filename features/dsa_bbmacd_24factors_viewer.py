# -*- coding: utf-8 -*-
"""
DSA + BBMacd 24因子可视化脚本（独立可运行）

Purpose
- 直接通过 pytdx 拉取 A 股 K 线
- 尽量按用户给定的 DSA 参考实现计算 DSA 结构与线段
- 计算 24 个核心因子，并输出 HTML 便于核对计算是否正确
- 可选输出 CSV

How to run
    python dsa_bbmacd_24factors_viewer.py --symbol 600547 --freq d --bars 300 \
        --out dsa_bbmacd_24factors_600547.html --csv-out dsa_bbmacd_24factors_600547.csv

Dependencies
    pip install pandas numpy plotly pytdx
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
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
                d["datetime"] = pd.to_datetime(d[["year", "month", "day", "hour", "minute"]].astype(int))
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


def pine_ema(src: pd.Series, length: int) -> pd.Series:
    arr = src.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if length <= 0 or len(arr) == 0:
        return pd.Series(out, index=src.index)
    alpha = 2.0 / (length + 1.0)
    prev = np.nan
    valid_count = 0
    seed_vals: List[float] = []
    seeded = False
    for i, val in enumerate(arr):
        if np.isnan(val):
            continue
        valid_count += 1
        if not seeded:
            seed_vals.append(val)
            if valid_count >= length:
                prev = float(np.mean(seed_vals[-length:]))
                out[i] = prev
                seeded = True
            continue
        prev = alpha * val + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=src.index)


def pine_rma(src: pd.Series, length: int) -> pd.Series:
    arr = src.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if length <= 0 or len(arr) == 0:
        return pd.Series(out, index=src.index)
    alpha = 1.0 / float(length)
    prev = np.nan
    valid_count = 0
    seed_vals: List[float] = []
    seeded = False
    for i, val in enumerate(arr):
        if np.isnan(val):
            continue
        valid_count += 1
        if not seeded:
            seed_vals.append(val)
            if valid_count >= length:
                prev = float(np.mean(seed_vals[-length:]))
                out[i] = prev
                seeded = True
            continue
        prev = alpha * val + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=src.index)


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


def atr_pine(df: pd.DataFrame, length: int) -> pd.Series:
    return pine_rma(true_range(df), length)


def cross_over(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_under(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def alpha_from_apt(apt: float) -> float:
    apt = max(1.0, float(apt))
    decay = math.exp(-math.log(2.0) / apt)
    return float(1.0 - decay)


# DSAConfig 已迁移到 dsa_experiment.pipeline.dsa_config
# 此处保留重导出以兼容现有引用（04/07 后续改为直接导入 dsa_config）
from dsa_experiment.pipeline.dsa_config import DSAConfig  # noqa: F401


@dataclass
class PivotLabel:
    t: int
    x: pd.Timestamp
    y: float
    text: str
    dir: int


def hlc3(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def compute_bbmacd(
    df: pd.DataFrame,
    rapida: int = 8,
    lenta: int = 26,
    stdv: float = 0.8,
    signal_len: int = 9,
    width_z_window: int = 60,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["m_rapida"] = pine_ema(df["close"], rapida)
    out["m_lenta"] = pine_ema(df["close"], lenta)
    out["bbmacd"] = out["m_rapida"] - out["m_lenta"]
    out["bbmacd_avg"] = pine_ema(out["bbmacd"], signal_len)
    out["bbmacd_std"] = out["bbmacd"].rolling(signal_len, min_periods=signal_len).std(ddof=1)
    out["bbmacd_upper"] = out["bbmacd_avg"] + stdv * out["bbmacd_std"]
    out["bbmacd_lower"] = out["bbmacd_avg"] - stdv * out["bbmacd_std"]
    out["bbmacd_minus_avg"] = out["bbmacd"] - out["bbmacd_avg"]
    out["bbmacd_state"] = np.select(
        [out["bbmacd"] < out["bbmacd_lower"], out["bbmacd"] > out["bbmacd_upper"]],
        [-1, 1],
        default=0,
    )
    denom = (out["bbmacd_upper"] - out["bbmacd_lower"]).replace(0.0, np.nan)
    out["bbmacd_band_pos_01"] = ((out["bbmacd"] - out["bbmacd_lower"]) / denom).clip(0.0, 1.0)
    out["bbmacd_bandwidth"] = out["bbmacd_upper"] - out["bbmacd_lower"]
    width_mean = out["bbmacd_bandwidth"].rolling(width_z_window, min_periods=width_z_window).mean()
    width_std = out["bbmacd_bandwidth"].rolling(width_z_window, min_periods=width_z_window).std(ddof=1)
    out["bbmacd_bandwidth_zscore"] = (out["bbmacd_bandwidth"] - width_mean) / width_std.replace(0.0, np.nan)
    out["bbmacd_cross_upper"] = cross_over(out["bbmacd"], out["bbmacd_upper"]).astype(float)
    out["bbmacd_cross_lower"] = cross_under(out["bbmacd"], out["bbmacd_lower"]).astype(float)
    out["bbmacd_cross_up_lower"] = cross_over(out["bbmacd"], out["bbmacd_lower"]).astype(float)
    out["bbmacd_cross_down_upper"] = cross_under(out["bbmacd"], out["bbmacd_upper"]).astype(float)
    return out


def _run_from_dir(dir_out: np.ndarray) -> List[Tuple[int, int, int]]:
    runs: List[Tuple[int, int, int]] = []
    n = len(dir_out)
    if n == 0:
        return runs
    run_start = 0
    for i in range(1, n):
        curr_dir = int(dir_out[i]) if np.isfinite(dir_out[i]) else 0
        prev_dir = int(dir_out[i - 1]) if np.isfinite(dir_out[i - 1]) else 0
        if curr_dir != prev_dir:
            if prev_dir != 0:
                runs.append((run_start, i - 1, prev_dir))
            run_start = i
    tail_dir = int(dir_out[n - 1]) if np.isfinite(dir_out[n - 1]) else 0
    if tail_dir != 0:
        runs.append((run_start, n - 1, tail_dir))
    return runs


def compute_dsa(df: pd.DataFrame, cfg: DSAConfig) -> Tuple[pd.DataFrame, List[PivotLabel], List[Dict]]:
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

    segments: List[Dict] = []
    cur_points_x: List[pd.Timestamp] = []
    cur_points_y: List[float] = []
    pivot_labels: List[PivotLabel] = []

    for t in range(n):
        st = max(0, t - cfg.prd + 1)
        win_h = high[st:t + 1]
        win_l = low[st:t + 1]

        if np.isfinite(high[t]) and high[t] == np.nanmax(win_h):
            ph = high[t]
            phL = t
        if np.isfinite(low[t]) and low[t] == np.nanmin(win_l):
            pl = low[t]
            plL = t

        dir_ = 1 if phL > plL else -1
        dir_out[t] = dir_
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
            pivot_labels.append(PivotLabel(t=int(x_anchor), x=d.index[x_anchor], y=float(y_anchor), text=txt, dir=int(dir_)))
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

    if len(cur_points_x) >= 2:
        segments.append({"dir": int(last_dir if last_dir is not None else -1), "x": np.array(cur_points_x), "y": np.array(cur_points_y, dtype=float)})

    # hindsight确认标签与最近确认高低点
    runs = _run_from_dir(dir_out)
    hindsight_labels: List[PivotLabel] = []
    last_pivot_type = np.array([""] * n, dtype=object)
    prev_confirmed_run_bars = np.full(n, np.nan)
    current_run_bars = np.full(n, np.nan)

    prev_hindsight_high = np.nan
    prev_hindsight_low = np.nan
    latest_type = ""
    latest_run_bars = np.nan
    for st_run, ed_run, run_dir in runs:
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
            hindsight_labels.append(PivotLabel(t=int(pivot_idx), x=d.index[pivot_idx], y=pivot_price, text=txt, dir=int(run_dir)))
            latest_type = txt

        run_bars = float(ed_run - st_run + 1)
        current_run_bars[st_run:ed_run + 1] = np.arange(1, ed_run - st_run + 2, dtype=float)
        prev_confirmed_run_bars[st_run:ed_run + 1] = latest_run_bars
        if latest_type:
            last_pivot_type[st_run:ed_run + 1] = latest_type
        latest_run_bars = run_bars

    hindsight_high = np.full(n, np.nan)
    hindsight_low = np.full(n, np.nan)
    hindsight_pos = np.full(n, np.nan)
    bars_since_last_high = np.full(n, np.nan)
    bars_since_last_low = np.full(n, np.nan)

    confirmed_highs: List[Tuple[int, float]] = []
    confirmed_lows: List[Tuple[int, float]] = []
    for lab in hindsight_labels:
        if lab.text in {"HH", "LH"}:
            confirmed_highs.append((lab.t, lab.y))
        elif lab.text in {"HL", "LL"}:
            confirmed_lows.append((lab.t, lab.y))

    for t in range(n):
        recent_high = np.nan
        recent_high_idx: Optional[int] = None
        for idx, price in reversed(confirmed_highs):
            if idx <= t:
                recent_high = price
                recent_high_idx = idx
                break
        recent_low = np.nan
        recent_low_idx: Optional[int] = None
        for idx, price in reversed(confirmed_lows):
            if idx <= t:
                recent_low = price
                recent_low_idx = idx
                break
        hindsight_high[t] = recent_high
        hindsight_low[t] = recent_low
        if recent_high_idx is not None:
            bars_since_last_high[t] = float(t - recent_high_idx)
        if recent_low_idx is not None:
            bars_since_last_low[t] = float(t - recent_low_idx)
        if np.isfinite(recent_high) and np.isfinite(recent_low):
            lo = min(recent_low, recent_high)
            hi = max(recent_low, recent_high)
            if hi > lo:
                hindsight_pos[t] = np.clip((close[t] - lo) / (hi - lo), 0.0, 1.0)

    out = pd.DataFrame(index=df.index)
    out["DSA_VWAP"] = vwap_out
    out["dsa_dir"] = dir_out
    out["prev_pivot_type"] = last_pivot_type
    out["last_confirmed_high"] = hindsight_high
    out["last_confirmed_low"] = hindsight_low
    out["dsa_pivot_pos_01"] = hindsight_pos
    out["current_stage_bars"] = current_run_bars
    out["prev_stage_bars"] = prev_confirmed_run_bars
    out["bars_since_last_high"] = bars_since_last_high
    out["bars_since_last_low"] = bars_since_last_low
    out["dsa_atr"] = atr.to_numpy(dtype=float)
    out["dsa_ratio"] = ratio
    return out, hindsight_labels, segments


def compute_24_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pivot_code_map = {"HH": 2.0, "HL": 1.0, "LH": -1.0, "LL": -2.0}
    out["prev_pivot_code"] = out["prev_pivot_type"].map(pivot_code_map).astype(float)

    out["ret_to_last_high_pct"] = out["close"] / out["last_confirmed_high"] - 1.0
    out["ret_to_last_low_pct"] = out["close"] / out["last_confirmed_low"] - 1.0
    out["price_vs_dsa_vwap_pct"] = out["close"] / out["DSA_VWAP"] - 1.0

    runs = _run_from_dir(out["dsa_dir"].to_numpy(float))
    prev_stage_amp = np.full(len(out), np.nan)
    current_stage_ret = np.full(len(out), np.nan)
    current_stage_amp = np.full(len(out), np.nan)
    current_pullback = np.full(len(out), np.nan)

    prev_amp_val = np.nan
    for st, ed, run_dir in runs:
        seg_high = float(np.nanmax(out["high"].iloc[st:ed + 1]))
        seg_low = float(np.nanmin(out["low"].iloc[st:ed + 1]))
        seg_amp = (seg_high - seg_low) / seg_low if np.isfinite(seg_low) and seg_low != 0 else np.nan
        prev_stage_amp[st:ed + 1] = prev_amp_val

        start_close = float(out["close"].iloc[st])
        if np.isfinite(start_close) and start_close != 0:
            current_stage_ret[st:ed + 1] = out["close"].iloc[st:ed + 1] / start_close - 1.0

        if run_dir > 0:
            run_low = float(np.nanmin(out["low"].iloc[st:ed + 1]))
            roll_high = out["high"].iloc[st:ed + 1].cummax().to_numpy(dtype=float)
            current_stage_amp[st:ed + 1] = (roll_high - run_low) / run_low if np.isfinite(run_low) and run_low != 0 else np.nan
            curr_close = out["close"].iloc[st:ed + 1].to_numpy(dtype=float)
            current_pullback[st:ed + 1] = curr_close / roll_high - 1.0
        else:
            run_high = float(np.nanmax(out["high"].iloc[st:ed + 1]))
            roll_low = out["low"].iloc[st:ed + 1].cummin().to_numpy(dtype=float)
            current_stage_amp[st:ed + 1] = (run_high - roll_low) / run_high if np.isfinite(run_high) and run_high != 0 else np.nan
            curr_close = out["close"].iloc[st:ed + 1].to_numpy(dtype=float)
            current_pullback[st:ed + 1] = curr_close / roll_low - 1.0

        prev_amp_val = seg_amp

    out["prev_stage_amp_pct"] = prev_stage_amp
    out["current_stage_ret_pct"] = current_stage_ret
    out["current_stage_amp_pct"] = current_stage_amp
    out["current_pullback_from_stage_extreme_pct"] = current_pullback

    trend_align = np.zeros(len(out), dtype=float)
    long_align = (out["dsa_dir"] > 0) & (out["bbmacd_minus_avg"] > 0)
    short_align = (out["dsa_dir"] < 0) & (out["bbmacd_minus_avg"] < 0)
    trend_align[long_align.to_numpy()] = 1.0
    trend_align[short_align.to_numpy()] = -1.0
    out["trend_align_momo"] = trend_align

    # 最终核心24字段 + 辅助文本字段 prev_pivot_type
    field_order = [
        "dsa_dir",
        "prev_pivot_code",
        "last_confirmed_high",
        "last_confirmed_low",
        "dsa_pivot_pos_01",
        "ret_to_last_high_pct",
        "ret_to_last_low_pct",
        "price_vs_dsa_vwap_pct",
        "current_stage_bars",
        "prev_stage_bars",
        "bars_since_last_high",
        "bars_since_last_low",
        "prev_stage_amp_pct",
        "current_stage_ret_pct",
        "current_stage_amp_pct",
        "current_pullback_from_stage_extreme_pct",
        "bbmacd",
        "bbmacd_minus_avg",
        "bbmacd_state",
        "bbmacd_band_pos_01",
        "bbmacd_bandwidth_zscore",
        "bbmacd_cross_upper",
        "bbmacd_cross_lower",
        "trend_align_momo",
    ]
    out["prev_pivot_type"] = out["prev_pivot_type"].astype(object)
    return out[["prev_pivot_type"] + field_order]


def _format_factor_hover(df: pd.DataFrame) -> List[str]:
    cols = [
        "prev_pivot_type", "dsa_dir", "prev_pivot_code", "last_confirmed_high", "last_confirmed_low",
        "dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct", "price_vs_dsa_vwap_pct",
        "current_stage_bars", "prev_stage_bars", "bars_since_last_high", "bars_since_last_low",
        "prev_stage_amp_pct", "current_stage_ret_pct", "current_stage_amp_pct",
        "current_pullback_from_stage_extreme_pct", "bbmacd", "bbmacd_minus_avg", "bbmacd_state",
        "bbmacd_band_pos_01", "bbmacd_bandwidth_zscore", "bbmacd_cross_upper", "bbmacd_cross_lower",
        "trend_align_momo",
    ]

    def _scalarize(v: object) -> object:
        if isinstance(v, pd.Series):
            return v.iloc[0] if len(v) else np.nan
        if isinstance(v, np.ndarray):
            return v.flat[0] if v.size else np.nan
        if isinstance(v, (list, tuple)):
            return v[0] if len(v) else np.nan
        return v

    texts: List[str] = []
    view = df.loc[:, cols]
    for _, row in view.iterrows():
        parts: List[str] = []
        for c in cols:
            v = _scalarize(row[c])
            if isinstance(v, str):
                parts.append(f"{c}={v}")
            elif pd.isna(v):
                parts.append(f"{c}=nan")
            else:
                parts.append(f"{c}={float(v):.6f}")
        texts.append("<br>".join(parts))
    return texts


def build_figure(df: pd.DataFrame, dsa_segments: List[Dict], dsa_labels: List[PivotLabel], out_html: str, title: str) -> None:
    x_num = np.arange(len(df), dtype=float)
    intraday = len(df.index) > 1 and (df.index[1] - df.index[0]) < pd.Timedelta("20h")
    tick_text = [ts.strftime("%Y-%m-%d %H:%M") if intraday else ts.strftime("%Y-%m-%d") for ts in df.index]

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.40, 0.22, 0.12, 0.13, 0.13],
        subplot_titles=(
            title,
            "BBMacd",
            "结构 / 位置",
            "时间 / 振幅",
            "组合状态",
        ),
    )

    hover_text = _format_factor_hover(df)

    fig.add_trace(
        go.Candlestick(
            x=x_num,
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color="#00c2a0", decreasing_line_color="#ff4d5a",
            increasing_fillcolor="#00c2a0", decreasing_fillcolor="#ff4d5a",
            name="K线", showlegend=False,
        ),
        row=1, col=1,
    )

    for seg in dsa_segments:
        seg_x = pd.to_datetime(seg["x"])
        mask = (seg_x >= df.index.min()) & (seg_x <= df.index.max())
        if mask.sum() < 2:
            continue
        seg_idx: List[int] = []
        seg_y: List[float] = []
        for xv, yv in zip(seg_x[mask], np.asarray(seg["y"])[mask]):
            loc = df.index.get_indexer([pd.Timestamp(xv)])[0]
            if loc >= 0:
                seg_idx.append(int(loc))
                seg_y.append(float(yv))
        if len(seg_idx) < 2:
            continue
        color = "#ff1744" if seg["dir"] > 0 else "#00e676"
        fig.add_trace(
            go.Scatter(
                x=np.array(seg_idx, dtype=float), y=np.array(seg_y, dtype=float),
                mode="lines", line=dict(width=2.2, color=color),
                name="DSA_VWAP", showlegend=False, hoverinfo="skip",
            ),
            row=1, col=1,
        )

    for lab in dsa_labels:
        if lab.x not in df.index or not lab.text:
            continue
        idx = df.index.get_loc(lab.x)
        is_up = lab.dir > 0
        fig.add_annotation(
            x=idx, y=lab.y, xref="x", yref="y", text=lab.text,
            showarrow=True, arrowhead=2, ax=0, ay=25 if is_up else -25,
            bgcolor="rgba(0,230,118,0.85)" if is_up else "rgba(255,23,68,0.85)",
            font=dict(color="white", size=11), row=1, col=1,
        )

    # 价格轨迹上的因子hover锚点
    fig.add_trace(
        go.Scatter(
            x=x_num, y=df["close"], mode="lines", line=dict(color="rgba(0,0,0,0)", width=8),
            name="factor_hover", showlegend=False, text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1, col=1,
    )

    # BBMacd panel
    fig.add_trace(go.Scatter(x=x_num, y=df["bbmacd_upper"], mode="lines", line=dict(color="rgba(0,191,255,0.95)", width=1), name="上轨", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["bbmacd_lower"], mode="lines", line=dict(color="rgba(0,191,255,0.95)", width=1), fill="tonexty", fillcolor="rgba(0,191,255,0.16)", name="下轨", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["bbmacd_avg"], mode="lines", line=dict(color="#ffd54f", width=1.0, dash="dot"), name="avg", showlegend=False), row=2, col=1)

    state = df["bbmacd_state"].fillna(0).astype(int)
    start = 0
    shown = set()
    for i in range(1, len(df)):
        if state.iloc[i] != state.iloc[i - 1]:
            rng = (start, i - 1, int(state.iloc[i - 1]))
            start = i - 1
            s, e, st = rng
            xs = x_num[s:e + 1]
            ys = df["bbmacd"].iloc[s:e + 1]
            color = {1: "#008000", -1: "#FF0000", 0: "#1e88e5"}[st]
            name = {1: "BBMacd>上轨", -1: "BBMacd<下轨", 0: "BBMacd带内"}[st]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=2), name=name, showlegend=name not in shown), row=2, col=1)
            shown.add(name)
    if len(df) > 0:
        s, e, st = start, len(df) - 1, int(state.iloc[-1])
        xs = x_num[s:e + 1]
        ys = df["bbmacd"].iloc[s:e + 1]
        color = {1: "#008000", -1: "#FF0000", 0: "#1e88e5"}[st]
        name = {1: "BBMacd>上轨", -1: "BBMacd<下轨", 0: "BBMacd带内"}[st]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=2), name=name, showlegend=name not in shown), row=2, col=1)

    buy_mask = df["bbmacd_cross_upper"] == 1
    sell_mask = df["bbmacd_cross_lower"] == 1
    fig.add_trace(go.Scatter(x=x_num[buy_mask.to_numpy()], y=df.loc[buy_mask, "bbmacd"], mode="markers", marker=dict(symbol="triangle-up", size=10, color="#00e676"), name="cross_upper", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_num[sell_mask.to_numpy()], y=df.loc[sell_mask, "bbmacd"], mode="markers", marker=dict(symbol="triangle-down", size=10, color="#ff4d5a"), name="cross_lower", showlegend=False), row=2, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#90a4ae", row=2, col=1)

    # structure / position
    for nm in ["dsa_dir", "prev_pivot_code", "dsa_pivot_pos_01", "bbmacd_band_pos_01"]:
        fig.add_trace(go.Scatter(x=x_num, y=df[nm], mode="lines", line=dict(width=1.35), name=nm, showlegend=False), row=3, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#888888", row=3, col=1)
    fig.add_hline(y=1.0, line_width=1, line_dash="dot", line_color="#888888", row=3, col=1)

    # time / amplitude
    for nm in ["current_stage_bars", "prev_stage_bars", "bars_since_last_high", "bars_since_last_low", "prev_stage_amp_pct", "current_stage_ret_pct", "current_stage_amp_pct", "current_pullback_from_stage_extreme_pct"]:
        fig.add_trace(go.Scatter(x=x_num, y=df[nm], mode="lines", line=dict(width=1.25), name=nm, showlegend=False), row=4, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#888888", row=4, col=1)

    # combo / momentum state
    for nm in ["bbmacd_minus_avg", "bbmacd_bandwidth_zscore", "price_vs_dsa_vwap_pct", "ret_to_last_high_pct", "ret_to_last_low_pct", "trend_align_momo"]:
        fig.add_trace(go.Scatter(x=x_num, y=df[nm], mode="lines", line=dict(width=1.35), name=nm, showlegend=False), row=5, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#888888", row=5, col=1)

    tick_step = max(1, len(df) // 10)
    tickvals = list(range(0, len(df), tick_step))
    if tickvals and tickvals[-1] != len(df) - 1:
        tickvals.append(len(df) - 1)
    ticktext = [tick_text[i] for i in tickvals] if tickvals else []

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        height=1700,
    )
    for r in [1, 2, 3, 4, 5]:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True, zeroline=False, row=r, col=1)
        fig.update_yaxes(showgrid=True, zeroline=False, row=r, col=1)

    fig.write_html(out_html, include_plotlyjs="cdn")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DSA + BBMacd 24因子可视化脚本")
    p.add_argument("--symbol", required=True, help="A股代码，如 600547")
    p.add_argument("--freq", default="d", help="d/w/mo/1m/5m/15m/30m/60m")
    p.add_argument("--bars", type=int, default=300, help="展示最近 N 根K线")
    p.add_argument("--fetch-bars", type=int, default=1200, help="实际抓取并用于计算的K线数量，建议 > bars")
    p.add_argument("--out", default="dsa_bbmacd_24factors.html", help="输出HTML文件")
    p.add_argument("--csv-out", default="", help="可选：输出CSV文件")
    p.add_argument("--dsa-prd", type=int, default=50)
    p.add_argument("--dsa-base-apt", type=float, default=20.0)
    p.add_argument("--dsa-use-adapt", action="store_true")
    p.add_argument("--dsa-vol-bias", type=float, default=10.0)
    p.add_argument("--bb-rapida", type=int, default=8)
    p.add_argument("--bb-lenta", type=int, default=26)
    p.add_argument("--bb-stdv", type=float, default=0.8)
    p.add_argument("--bb-signal-len", type=int, default=9)
    p.add_argument("--bb-width-z-window", type=int, default=60)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.freq = normalize_freq(args.freq)
    fetch_bars = max(args.fetch_bars, args.bars, 300)
    price_df = fetch_kline_pytdx(args.symbol, args.freq, fetch_bars)

    dsa_df, dsa_labels, dsa_segments = compute_dsa(
        price_df,
        DSAConfig(
            prd=args.dsa_prd,
            base_apt=args.dsa_base_apt,
            use_adapt=bool(args.dsa_use_adapt),
            vol_bias=args.dsa_vol_bias,
        ),
    )
    bb_df = compute_bbmacd(
        price_df,
        rapida=args.bb_rapida,
        lenta=args.bb_lenta,
        stdv=args.bb_stdv,
        signal_len=args.bb_signal_len,
        width_z_window=args.bb_width_z_window,
    )

    merged = pd.concat([price_df, dsa_df, bb_df], axis=1)
    factors_df = compute_24_factors(merged)
    view_cols = ["open", "high", "low", "close", "bbmacd_upper", "bbmacd_lower", "bbmacd_avg"]
    merged = pd.concat([merged[view_cols], factors_df], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
    merged = merged.loc[~merged.index.duplicated(keep="last")]
    merged = merged.tail(args.bars).copy()

    title = f"{args.symbol} [{args.freq}] DSA + BBMacd 24 Factors"
    build_figure(merged, dsa_segments, dsa_labels, args.out, title)

    if args.csv_out:
        merged.to_csv(args.csv_out, encoding="utf-8-sig")
        print(f"CSV 已生成: {args.csv_out}")
    print(f"HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()
