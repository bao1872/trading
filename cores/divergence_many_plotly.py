# -*- coding: utf-8 -*-
"""
Divergence for Many Indicators v4 (TradingView Pine by LonesomeTheBlue) -> Plotly HTML

Purpose:
    检测多种指标的背离信号并生成 Plotly HTML 图表

Inputs:
    - symbol: A 股代码 (如 000426)
    - freq: 周期 (5m/15m/30m/60m/d/w/month)
    - start/end: 日期时间范围
    - prd: Pivot Period
    - searchdiv: 背离类型 (Regular/Hidden/Regular/Hidden)

Outputs:
    - HTML 图表文件 (包含 K 线、成交量、背离线和标签)
    - 可选 PNG 图片

How to Run:
    python cores/divergence_many_plotly.py --symbol 600489 --freq 60m --start "2025-11-01 09:30" --end "2026-01-20 15:00" --prd 5 --searchdiv Regular --out div.html
    python cores/divergence_many_plotly.py --symbol 002099 --years 3 --out div.html

Side Effects:
    - 写文件：输出 HTML/PNG 文件到指定路径
"""

from __future__ import annotations

import argparse
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datasource.pytdx_client import connect_pytdx, PERIOD_MAP


# =========================
# Config (match Pine inputs)
# =========================

@dataclass
class DivConfig:
    prd: int = 5
    source: str = "Close"
    searchdiv: str = "Regular"
    showindis: str = "Full"
    showlimit: int = 1
    maxpp: int = 10
    maxbars: int = 100
    shownum: bool = True
    showlast: bool = False
    dontconfirm: bool = False
    showlines: bool = True
    showpivot: bool = False

    calcmacd: bool = True
    calcmacda: bool = True
    calcrsi: bool = False
    calcstoc: bool = False
    calccci: bool = False
    calcmom: bool = False
    calcobv: bool = True
    calcvwmacd: bool = False
    calccmf: bool = False
    calcmfi: bool = False
    calcext: bool = False

    pos_reg_div_col: str = "rgba(255, 235, 59, 1.0)"
    neg_reg_div_col: str = "rgba(0, 32, 96, 1.0)"
    pos_hid_div_col: str = "rgba(0, 255, 0, 1.0)"
    neg_hid_div_col: str = "rgba(255, 0, 0, 1.0)"
    pos_div_text_col: str = "rgba(0,0,0,1.0)"
    neg_div_text_col: str = "rgba(255,255,255,1.0)"

    reg_div_l_style: str = "Solid"
    hid_div_l_style: str = "Dashed"
    reg_div_l_width: int = 2
    hid_div_l_width: int = 1

    showmas: bool = False
    ma50_col: str = "rgba(0,255,0,0.9)"
    ma200_col: str = "rgba(255,0,0,0.9)"


def fetch_bars_pytdx(symbol: str, freq: str, start: str, end: str, *, max_pages: int = 200) -> pd.DataFrame:
    """Fetch bars via pytdx and filter by datetime range (inclusive)."""
    api = connect_pytdx()
    try:
        cat = PERIOD_MAP.get(freq.lower())
        if cat is None:
            raise ValueError(f"不支持的 freq: {freq}. 可选：{list(PERIOD_MAP.keys())}")
        mkt = 1 if symbol.startswith("6") else 0

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        page = 0
        size = 800
        frames: List[pd.DataFrame] = []

        while page < max_pages:
            recs = api.get_security_bars(cat, mkt, symbol, page * size, size)
            if not recs:
                break
            df = pd.DataFrame(recs)
            if df.empty:
                break
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            elif {"year", "month", "day", "hour", "minute"}.issubset(df.columns):
                df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]].astype(int))
                df = df.set_index("datetime")
            if "vol" in df.columns:
                df = df.rename(columns={"vol": "volume"})
            frames.append(df)

            if len(recs) < size:
                break
            page += 1

        if not frames:
            raise RuntimeError("pytdx 无数据")

        all_df = pd.concat(frames).sort_index()
        out = all_df[(all_df.index >= start_dt) & (all_df.index <= end_dt)]
        return out
    finally:
        api.disconnect()


# =========================
# Indicators (Pine-aligned)
# =========================

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def vwma(price: pd.Series, vol: pd.Series, n: int) -> pd.Series:
    pv = (price * vol).rolling(n, min_periods=n).sum()
    vv = vol.rolling(n, min_periods=n).sum()
    return pv / vv


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 10) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = sma(tp, n)
    md = (tp - ma).abs().rolling(n, min_periods=n).mean()
    return (tp - ma) / (0.015 * md)


def stoch_k(close: pd.Series, high: pd.Series, low: pd.Series, n: int = 14) -> pd.Series:
    ll = low.rolling(n, min_periods=n).min()
    hh = high.rolling(n, min_periods=n).max()
    denom = (hh - ll).replace(0.0, np.nan)
    return 100.0 * (close - ll) / denom


def obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    d = close.diff()
    direction = np.sign(d).fillna(0.0)
    return (direction * vol).cumsum()


def cmf(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, n: int = 21) -> pd.Series:
    denom = (high - low).replace(0.0, np.nan)
    cmfm = ((close - low) - (high - close)) / denom
    cmfv = cmfm * vol
    return sma(cmfv, n) / sma(vol, n)


def mfi_pine(src: pd.Series, vol: pd.Series, n: int = 14) -> pd.Series:
    tp = src
    rmf = tp * vol
    delta = tp.diff()
    pos = rmf.where(delta > 0, 0.0)
    neg = rmf.where(delta < 0, 0.0).abs()
    pos_sum = pos.rolling(n, min_periods=n).sum()
    neg_sum = neg.rolling(n, min_periods=n).sum()
    ratio = pos_sum / neg_sum
    return 100 - (100 / (1 + ratio))


def compute_indicators(df: pd.DataFrame) -> dict[str, pd.Series]:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    macd_line = ema(close, 12) - ema(close, 26)
    signal = ema(macd_line, 9)
    hist = macd_line - signal

    moment = close.diff(10)
    cci10 = cci(high, low, close, 10)
    obv_ = obv(close, vol)
    stk = sma(stoch_k(close, high, low, 14), 3)

    vw_fast = vwma(close, vol, 12)
    vw_slow = vwma(close, vol, 26)
    vwmacd_ = vw_fast - vw_slow

    cmf_ = cmf(high, low, close, vol, 21)
    mfi_ = mfi_pine(close, vol, 14)
    rsi_ = rsi_wilder(close, 14)

    return {
        "macd": macd_line,
        "hist": hist,
        "rsi": rsi_,
        "stoch": stk,
        "cci": cci10,
        "mom": moment,
        "obv": obv_,
        "vwmacd": vwmacd_,
        "cmf": cmf_,
        "mfi": mfi_,
        "external": close,
    }


# =========================
# Pivot confirmation (Pine-like)
# =========================

def pivots_confirmed(src: np.ndarray, prd: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(src)
    ph = np.full(n, np.nan, dtype=float)
    pl = np.full(n, np.nan, dtype=float)
    w = prd

    for t in range(2*w, n):
        i = t - w
        if i - w < 0 or i + w >= n:
            continue
        window = src[i - w : i + w + 1]
        v = src[i]
        if np.isfinite(v) and np.nanmax(window) == v:
            ph[t] = v
        if np.isfinite(v) and np.nanmin(window) == v:
            pl[t] = v
    return ph, pl


# =========================
# Divergence core (ported from Pine)
# =========================

def _line_dash(style: str) -> str:
    if style == "Solid":
        return "solid"
    if style == "Dashed":
        return "dash"
    return "dot"


def pos_reg_or_neg_hid(src: np.ndarray, close: np.ndarray, low: np.ndarray, pl_positions: list[int], pl_vals: list[float],
                       t: int, cfg: DivConfig, cond: int) -> int:
    divlen = 0
    prsc = close if cfg.source == "Close" else low

    if cfg.dontconfirm or (t-1 >= 0 and (src[t] > src[t-1] or close[t] > close[t-1])):
        startpoint = 0 if cfg.dontconfirm else 1

        for x in range(min(cfg.maxpp, len(pl_positions))):
            conf_idx = pl_positions[x]
            if conf_idx == 0:
                break
            length = t - conf_idx + cfg.prd
            if length > cfg.maxbars:
                break
            if length <= 5:
                continue
            if t - length < 0:
                continue

            src_sp = src[t - startpoint]
            src_len = src[t - length]
            pr_sp = prsc[t - startpoint]
            plv = pl_vals[x]

            ok = False
            if cond == 1:
                ok = (src_sp > src_len) and (pr_sp < plv)
            else:
                ok = (src_sp < src_len) and (pr_sp > plv)

            if not ok:
                continue

            slope1 = (src_sp - src_len) / (length - startpoint)
            virtual1 = src_sp - slope1

            c_sp = close[t - startpoint]
            c_len = close[t - length]
            slope2 = (c_sp - c_len) / (length - startpoint)
            virtual2 = c_sp - slope2

            arrived = True
            for y in range(1 + startpoint, length):
                idx = t - y
                if idx < 0:
                    arrived = False
                    break
                if src[idx] < virtual1 or close[idx] < virtual2:
                    arrived = False
                    break
                virtual1 -= slope1
                virtual2 -= slope2

            if arrived:
                divlen = length
                break

    return divlen


def neg_reg_or_pos_hid(src: np.ndarray, close: np.ndarray, high: np.ndarray, ph_positions: list[int], ph_vals: list[float],
                       t: int, cfg: DivConfig, cond: int) -> int:
    divlen = 0
    prsc = close if cfg.source == "Close" else high

    if cfg.dontconfirm or (t-1 >= 0 and (src[t] < src[t-1] or close[t] < close[t-1])):
        startpoint = 0 if cfg.dontconfirm else 1

        for x in range(min(cfg.maxpp, len(ph_positions))):
            conf_idx = ph_positions[x]
            if conf_idx == 0:
                break
            length = t - conf_idx + cfg.prd
            if length > cfg.maxbars:
                break
            if length <= 5:
                continue
            if t - length < 0:
                continue

            src_sp = src[t - startpoint]
            src_len = src[t - length]
            pr_sp = prsc[t - startpoint]
            phv = ph_vals[x]

            ok = False
            if cond == 1:
                ok = (src_sp < src_len) and (pr_sp > phv)
            else:
                ok = (src_sp > src_len) and (pr_sp < phv)

            if not ok:
                continue

            slope1 = (src_sp - src_len) / (length - startpoint)
            virtual1 = src_sp - slope1

            c_sp = close[t - startpoint]
            c_len = close[t - length]
            slope2 = (c_sp - c_len) / (length - startpoint)
            virtual2 = c_sp - slope2

            arrived = True
            for y in range(1 + startpoint, length):
                idx = t - y
                if idx < 0:
                    arrived = False
                    break
                if src[idx] > virtual1 or close[idx] > virtual2:
                    arrived = False
                    break
                virtual1 -= slope1
                virtual2 -= slope2

            if arrived:
                divlen = length
                break

    return divlen


def calculate_divs(enabled: bool, src: np.ndarray, close: np.ndarray, high: np.ndarray, low: np.ndarray,
                   ph_positions: list[int], ph_vals: list[float], pl_positions: list[int], pl_vals: list[float],
                   t: int, cfg: DivConfig) -> list[int]:
    out = [0, 0, 0, 0]
    if not enabled:
        return out

    if cfg.searchdiv in ("Regular", "Regular/Hidden"):
        out[0] = pos_reg_or_neg_hid(src, close, low, pl_positions, pl_vals, t, cfg, cond=1)
        out[1] = neg_reg_or_pos_hid(src, close, high, ph_positions, ph_vals, t, cfg, cond=1)

    if cfg.searchdiv in ("Hidden", "Regular/Hidden"):
        out[2] = neg_reg_or_pos_hid(src, close, high, ph_positions, ph_vals, t, cfg, cond=2)
        out[3] = pos_reg_or_neg_hid(src, close, low, pl_positions, pl_vals, t, cfg, cond=2)

    return out


# =========================
# Engine: per-bar playback + "deletion" semantics
# =========================

def build_indicator_names(cfg: DivConfig) -> list[str]:
    if cfg.showindis == "Don't Show":
        return [""] * 11
    if cfg.showindis == "Full":
        return ["MACD","Hist","RSI","Stoch","CCI","MOM","OBV","VWMACD","CMF","MFI","Extrn"]
    return ["M","H","E","S","C","M","O","V","C","M","X"]


def build_div_colors(cfg: DivConfig) -> list[str]:
    return [cfg.pos_reg_div_col, cfg.neg_reg_div_col, cfg.pos_hid_div_col, cfg.neg_hid_div_col]


def compute_div_features_from_events(
    events: list[tuple[int, str]],
    t_now: int,
    n_last: int = 5,
    rate_M: int = 100,
    age_cap: int = 100
) -> dict:
    ev = [(t, d) for (t, d) in events if d in ("top", "bottom") and t <= t_now]

    if not ev:
        return {
            "div_bias": 0.0,
            "div_last_type": "none",
            "div_last_age": 999,
            "div_last_age_norm": 1.0,
            "div_rate": 0.0,
        }

    last_n = ev[-n_last:] if len(ev) >= n_last else ev
    top_cnt = sum(1 for _, d in last_n if d == "top")
    bottom_cnt = sum(1 for _, d in last_n if d == "bottom")
    denom = max(len(last_n), 1)
    div_bias = (bottom_cnt - top_cnt) / denom

    last_t = ev[-1][0]
    dirs_at_last_t = {d for (t, d) in ev if t == last_t}
    if dirs_at_last_t == {"top"}:
        last_type = "top"
    elif dirs_at_last_t == {"bottom"}:
        last_type = "bottom"
    else:
        last_type = "conflict"

    last_age = int(t_now - last_t)
    cap = max(int(age_cap), 1)
    last_age_norm = min(last_age, cap) / cap

    M = max(int(rate_M), 1)
    win_start = max(t_now - M + 1, 0)
    ev_in_window = sum(1 for (t, _) in ev if win_start <= t <= t_now)
    div_rate = ev_in_window / M

    return {
        "div_bias": float(div_bias),
        "div_last_type": last_type,
        "div_last_age": int(last_age),
        "div_last_age_norm": float(last_age_norm),
        "div_rate": float(div_rate),
    }


def run_divergence_engine(df: pd.DataFrame, cfg: DivConfig, feat_n: int = 5, feat_M: int = 100, age_cap: int = 100) -> tuple[list[dict], list[dict], list[tuple[str,float,str]], list[tuple[str,float,str]], dict]:
    ind = compute_indicators(df)
    names = build_indicator_names(cfg)
    div_colors = build_div_colors(cfg)

    div_events: list[tuple[int, str]] = []
    last_feat: dict = {}

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    pivot_src_high = (df["close"] if cfg.source == "Close" else df["high"]).to_numpy(dtype=float)
    pivot_src_low = (df["close"] if cfg.source == "Close" else df["low"]).to_numpy(dtype=float)

    ph_conf, _ = pivots_confirmed(pivot_src_high, cfg.prd)
    _, pl_conf = pivots_confirmed(pivot_src_low, cfg.prd)

    maxarraysize = 20
    ph_positions: list[int] = []
    pl_positions: list[int] = []
    ph_vals: list[float] = []
    pl_vals: list[float] = []

    pos_div_lines: list[dict] = []
    neg_div_lines: list[dict] = []
    pos_div_labels: list[tuple[str,float,str,str,str]] = []
    neg_div_labels: list[tuple[str,float,str,str,str]] = []

    last_pos_div_lines = 0
    last_neg_div_lines = 0
    remove_last_pos_divs = False
    remove_last_neg_divs = False

    def delete_last_pos_div_lines_label(n: int):
        nonlocal pos_div_lines, pos_div_labels
        if n > 0 and len(pos_div_lines) >= n:
            for _ in range(n):
                pos_div_lines.pop()
            if len(pos_div_labels) > 0:
                pos_div_labels.pop()

    def delete_last_neg_div_lines_label(n: int):
        nonlocal neg_div_lines, neg_div_labels
        if n > 0 and len(neg_div_lines) >= n:
            for _ in range(n):
                neg_div_lines.pop()
            if len(neg_div_labels) > 0:
                neg_div_labels.pop()

    startpoint = 0 if cfg.dontconfirm else 1

    xcat = [pd.Timestamp(x).strftime('%Y-%m-%d %H:%M') for x in df.index]
    idx = xcat
    n = len(df)

    debug_ts = getattr(cfg, "debug_ts", None)
    if isinstance(debug_ts, str) and debug_ts.strip() == "":
        debug_ts = None
    debug_indicator = getattr(cfg, "debug_indicator", "hist")

    def _debug_print(msg: str):
        if debug_ts is not None:
            print(msg)

    def _debug_eval_indicator(t: int):
        key = str(debug_indicator).lower()
        if key not in ind:
            _debug_print(f"[DEBUG] unknown indicator={key}; available={list(ind.keys())}")
            return
        src_arr = ind[key].to_numpy(dtype=float)
        _debug_print(f"\n[DEBUG] t={t} ts={idx[t]} indicator={key} close={close[t]:.4f} src={src_arr[t]:.6f}")
        _debug_print(f"[DEBUG] pivots: ph_conf={int(np.isfinite(ph_conf[t]))} pl_conf={int(np.isfinite(pl_conf[t]))} ph_stored={len(ph_positions)} pl_stored={len(pl_positions)}")

        def _scan_pl(cond: int):
            label = "pos_reg" if cond == 1 else "neg_hid"
            prsc = close if cfg.source == "Close" else low
            if not (cfg.dontconfirm or (t-1 >= 0 and (src_arr[t] > src_arr[t-1] or close[t] > close[t-1]))):
                _debug_print(f"[DEBUG] {label}: precondition failed")
                return
            sp = 0 if cfg.dontconfirm else 1
            hit = False
            for xi in range(min(cfg.maxpp, len(pl_positions))):
                conf_idx = pl_positions[xi]
                length = t - conf_idx + cfg.prd
                if conf_idx == 0:
                    break
                if length > cfg.maxbars:
                    break
                if length <= 5 or t-length < 0:
                    continue
                src_sp = src_arr[t-sp]
                src_len = src_arr[t-length]
                pr_sp = prsc[t-sp]
                plv = pl_vals[xi]
                if cond == 1:
                    ok = (src_sp > src_len) and (pr_sp < plv)
                else:
                    ok = (src_sp < src_len) and (pr_sp > plv)
                if not ok:
                    continue
                slope1 = (src_sp - src_len) / (length - sp)
                v1 = src_sp - slope1
                c_sp = close[t-sp]
                c_len = close[t-length]
                slope2 = (c_sp - c_len) / (length - sp)
                v2 = c_sp - slope2
                arrived = True
                for y in range(1+sp, length):
                    ii = t - y
                    if src_arr[ii] < v1 or close[ii] < v2:
                        arrived = False
                        break
                    v1 -= slope1
                    v2 -= slope2
                if arrived:
                    hit = True
                    break
            if not hit:
                _debug_print(f"[DEBUG] {label}: no valid PL candidate")

        def _scan_ph(cond: int):
            label = "neg_reg" if cond == 1 else "pos_hid"
            prsc = close if cfg.source == "Close" else high
            if not (cfg.dontconfirm or (t-1 >= 0 and (src_arr[t] < src_arr[t-1] or close[t] < close[t-1]))):
                _debug_print(f"[DEBUG] {label}: precondition failed")
                return
            sp = 0 if cfg.dontconfirm else 1
            hit = False
            for xi in range(min(cfg.maxpp, len(ph_positions))):
                conf_idx = ph_positions[xi]
                length = t - conf_idx + cfg.prd
                if conf_idx == 0:
                    break
                if length > cfg.maxbars:
                    break
                if length <= 5 or t-length < 0:
                    continue
                src_sp = src_arr[t-sp]
                src_len = src_arr[t-length]
                pr_sp = prsc[t-sp]
                phv = ph_vals[xi]
                if cond == 1:
                    ok = (src_sp < src_len) and (pr_sp > phv)
                else:
                    ok = (src_sp > src_len) and (pr_sp < phv)
                if not ok:
                    continue
                slope1 = (src_sp - src_len) / (length - sp)
                v1 = src_sp - slope1
                c_sp = close[t-sp]
                c_len = close[t-length]
                slope2 = (c_sp - c_len) / (length - sp)
                v2 = c_sp - slope2
                arrived = True
                for y in range(1+sp, length):
                    ii = t - y
                    if src_arr[ii] > v1 or close[ii] > v2:
                        arrived = False
                        break
                    v1 -= slope1
                    v2 -= slope2
                if arrived:
                    hit = True
                    break
            if not hit:
                _debug_print(f"[DEBUG] {label}: no valid PH candidate")

        if cfg.searchdiv in ("Regular", "Regular/Hidden"):
            _scan_pl(cond=1)
            _scan_ph(cond=1)
        if cfg.searchdiv in ("Hidden", "Regular/Hidden"):
            _scan_ph(cond=2)
            _scan_pl(cond=2)

    for t in range(n):
        if np.isfinite(ph_conf[t]):
            ph_positions.insert(0, t)
            ph_vals.insert(0, float(ph_conf[t]))
            if len(ph_positions) > maxarraysize:
                ph_positions.pop()
                ph_vals.pop()

        if np.isfinite(pl_conf[t]):
            pl_positions.insert(0, t)
            pl_vals.insert(0, float(pl_conf[t]))
            if len(pl_positions) > maxarraysize:
                pl_positions.pop()
                pl_vals.pop()

        if np.isfinite(pl_conf[t]):
            remove_last_pos_divs = False
            last_pos_div_lines = 0
        if np.isfinite(ph_conf[t]):
            remove_last_neg_divs = False
            last_neg_div_lines = 0

        if debug_ts and idx[t] == debug_ts:
            _debug_eval_indicator(t)

        series_list = [
            ("macd", cfg.calcmacd),
            ("hist", cfg.calcmacda),
            ("rsi", cfg.calcrsi),
            ("stoch", cfg.calcstoc),
            ("cci", cfg.calccci),
            ("mom", cfg.calcmom),
            ("obv", cfg.calcobv),
            ("vwmacd", cfg.calcvwmacd),
            ("cmf", cfg.calccmf),
            ("mfi", cfg.calcmfi),
            ("external", cfg.calcext),
        ]

        all_divs = [0] * 44

        for i, (key, enabled) in enumerate(series_list):
            if not enabled:
                divs4 = [0, 0, 0, 0]
            else:
                src_arr = ind[key].to_numpy(dtype=float)
                divs4 = calculate_divs(
                    enabled=True,
                    src=src_arr,
                    close=close, high=high, low=low,
                    ph_positions=ph_positions, ph_vals=ph_vals,
                    pl_positions=pl_positions, pl_vals=pl_vals,
                    t=t, cfg=cfg
                )
            for k in range(4):
                all_divs[i*4 + k] = int(divs4[k])

        total_div = sum(1 for v in all_divs if v > 0)
        if total_div < cfg.showlimit:
            all_divs = [0] * 44

        divergence_text_top = ""
        divergence_text_bottom = ""
        distances: set[int] = set()
        dnumdiv_top = 0
        dnumdiv_bottom = 0
        top_label_col = "rgba(255,255,255,1.0)"
        bottom_label_col = "rgba(255,255,255,1.0)"

        for x in range(11):
            div_type = -1
            for y in range(4):
                dist = all_divs[x*4 + y]
                if dist > 0:
                    div_type = y
                    if (y % 2) == 1:
                        dnumdiv_top += 1
                        top_label_col = div_colors[y]
                    else:
                        dnumdiv_bottom += 1
                        bottom_label_col = div_colors[y]

                    if dist not in distances:
                        distances.add(dist)

                        if not cfg.showlines:
                            new_line = None
                        else:
                            x1 = idx[t - dist] if (t - dist) >= 0 else idx[0]
                            x2 = idx[t - startpoint] if (t - startpoint) >= 0 else idx[t]

                            if cfg.source == "Close":
                                y1 = float(close[t - dist]) if (t - dist) >= 0 else float(close[0])
                                y2 = float(close[t - startpoint]) if (t - startpoint) >= 0 else float(close[t])
                            else:
                                if (y % 2) == 0:
                                    y1 = float(low[t - dist]) if (t - dist) >= 0 else float(low[0])
                                    y2 = float(low[t - startpoint]) if (t - startpoint) >= 0 else float(low[t])
                                else:
                                    y1 = float(high[t - dist]) if (t - dist) >= 0 else float(high[0])
                                    y2 = float(high[t - startpoint]) if (t - startpoint) >= 0 else float(high[t])

                            dash = _line_dash(cfg.reg_div_l_style if y < 2 else cfg.hid_div_l_style)
                            width = cfg.reg_div_l_width if y < 2 else cfg.hid_div_l_width
                            new_line = dict(
                                x=[x1, x2], y=[y1, y2],
                                color=div_colors[y], width=width, dash=dash,
                                is_pos=((y % 2) == 0)
                            )

                        if new_line is not None:
                            if (y % 2) == 0:
                                pos_div_lines.append(new_line)
                            else:
                                neg_div_lines.append(new_line)

            if div_type >= 0 and names[x] != "":
                if (div_type % 2) == 1:
                    divergence_text_top += names[x] + "\n"
                else:
                    divergence_text_bottom += names[x] + "\n"

        if dnumdiv_top > 0:
            div_events.append((t, "top"))
        if dnumdiv_bottom > 0:
            div_events.append((t, "bottom"))

        last_feat = compute_div_features_from_events(
            div_events, t_now=t, n_last=feat_n, rate_M=feat_M, age_cap=age_cap
        )

        if (cfg.showindis != "Don't Show") or cfg.shownum:
            if cfg.shownum and dnumdiv_top > 0:
                divergence_text_top += str(dnumdiv_top)
            if cfg.shownum and dnumdiv_bottom > 0:
                divergence_text_bottom += str(dnumdiv_bottom)

            if divergence_text_top.strip() != "":
                y_top = float(max(high[t], high[t-1])) if t-1 >= 0 else float(high[t])
                neg_div_labels.append((idx[t], y_top, divergence_text_top.strip(), top_label_col, cfg.neg_div_text_col))

            if divergence_text_bottom.strip() != "":
                y_bot = float(min(low[t], low[t-1])) if t-1 >= 0 else float(low[t])
                pos_div_labels.append((idx[t], y_bot, divergence_text_bottom.strip(), bottom_label_col, cfg.pos_div_text_col))

    return pos_div_lines, neg_div_lines, pos_div_labels, neg_div_labels, last_feat


# =========================
# Plotly render
# =========================

def build_plot(df: pd.DataFrame, cfg: DivConfig, title: str, out_html: str, out_png: str = "") -> None:
    pos_lines, neg_lines, pos_labels, neg_labels, last_feat = run_divergence_engine(df, cfg)

    if isinstance(df.index, pd.DatetimeIndex):
        xcat = df.index.strftime("%Y-%m-%d %H:%M").tolist()
    else:
        xcat = [str(x) for x in df.index.tolist()]
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.78, 0.22],
        specs=[[{"type": "xy"}], [{"type": "xy"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=xcat,
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
            showlegend=False
        ),
        row=1, col=1
    )

    if cfg.showmas:
        ma50 = df["close"].rolling(50, min_periods=50).mean()
        ma200 = df["close"].rolling(200, min_periods=200).mean()
        fig.add_trace(go.Scatter(x=xcat, y=ma50, mode="lines", line=dict(width=1), name="MA50",
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=xcat, y=ma200, mode="lines", line=dict(width=1), name="MA200",
                                 hoverinfo="skip", showlegend=False), row=1, col=1)

    vol_colors = np.where(df["close"] >= df["open"],
                          "rgba(38,166,154,0.6)",
                          "rgba(239,83,80,0.6)")
    fig.add_trace(
        go.Bar(x=xcat, y=df["volume"], marker_color=vol_colors, showlegend=False),
        row=2, col=1
    )

    div_panel_lines = [
        "背离特征（最后一根）",
        f"div_bias (bottom-top)/n: {last_feat.get('div_bias', 0.0):.3f}",
        f"div_last_type: {str(last_feat.get('div_last_type', 'none'))}",
        f"div_last_age (bars): {int(last_feat.get('div_last_age', 999))}",
        f"div_last_age_norm: {last_feat.get('div_last_age_norm', 1.0):.3f}",
        f"div_rate (events/M): {last_feat.get('div_rate', 0.0):.4f}",
    ]
    fig.add_annotation(
        x=0.015, y=0.985,
        xref="paper", yref="paper",
        text="<br>".join(div_panel_lines),
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0.55)",
        bordercolor="rgba(255,255,255,0.25)",
        borderwidth=1,
        font=dict(color="white", size=12, family="Consolas, monospace"),
    )

    print("\n[Last-bar Divergence Features]")
    for line in div_panel_lines[1:]:
        print(line)

    def add_lines(lines: list[dict]):
        for ln in lines:
            fig.add_trace(
                go.Scatter(
                    x=ln["x"], y=ln["y"],
                    mode="lines",
                    line=dict(width=ln["width"], dash=ln["dash"]),
                    marker=dict(color=ln["color"]),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=1, col=1
            )

    if cfg.showlines:
        add_lines(pos_lines)
        add_lines(neg_lines)

    if cfg.showpivot:
        pivot_src_h = df["close"] if cfg.source == "Close" else df["high"]
        pivot_src_l = df["close"] if cfg.source == "Close" else df["low"]
        ph_conf, _ = pivots_confirmed(pivot_src_h.to_numpy(float), cfg.prd)
        _, pl_conf = pivots_confirmed(pivot_src_l.to_numpy(float), cfg.prd)

        xs_h, ys_h = [], []
        xs_l, ys_l = [], []
        for t in range(len(df)):
            if np.isfinite(ph_conf[t]) and (t - cfg.prd) >= 0:
                xs_h.append(xcat[t - cfg.prd])
                ys_h.append(float(df["high"].iloc[t - cfg.prd]))
            if np.isfinite(pl_conf[t]) and (t - cfg.prd) >= 0:
                xs_l.append(xcat[t - cfg.prd])
                ys_l.append(float(df["low"].iloc[t - cfg.prd]))

        if xs_h:
            fig.add_trace(go.Scatter(x=xs_h, y=ys_h, mode="markers+text",
                                     text=["H"]*len(xs_h), textposition="top center",
                                     marker=dict(size=6),
                                     hoverinfo="skip", showlegend=False), row=1, col=1)
        if xs_l:
            fig.add_trace(go.Scatter(x=xs_l, y=ys_l, mode="markers+text",
                                     text=["L"]*len(xs_l), textposition="bottom center",
                                     marker=dict(size=6),
                                     hoverinfo="skip", showlegend=False), row=1, col=1)

    for (x, y, text_, bg, tc) in neg_labels:
        fig.add_annotation(
            x=x, y=y, xref="x", yref="y",
            text=text_.replace("\n", "<br>"),
            showarrow=True, arrowhead=2, ax=0, ay=-25,
            bgcolor=bg, font=dict(color=tc, size=12),
            bordercolor=bg, borderwidth=1,
            row=1, col=1
        )

    for (x, y, text_, bg, tc) in pos_labels:
        fig.add_annotation(
            x=x, y=y, xref="x", yref="y",
            text=text_.replace("\n", "<br>"),
            showarrow=True, arrowhead=2, ax=0, ay=25,
            bgcolor=bg, font=dict(color=tc, size=12),
            bordercolor=bg, borderwidth=1,
            row=1, col=1
        )

    fig.update_layout(
        title=title,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=950,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False, type="category", categoryorder="array", categoryarray=xcat)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=2, col=1, rangemode="tozero")

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_html}")

    if out_png:
        fig.write_image(out_png, scale=2)
        print(f"[OK] PNG saved: {out_png}")


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)

    ap.add_argument("--symbol", type=str, default="000426", help="A 股代码，如 000426 / 002099")
    ap.add_argument("--freq", type=str, default="d", help="周期：5m/15m/30m/60m/d/w/month")
    ap.add_argument("--start", type=str, default="", help="开始日期时间 (含)，如 2025-01-01 或 2025-01-01 09:30")
    ap.add_argument("--end", type=str, default="", help="结束日期时间 (含)，默认=现在")
    ap.add_argument("--bars", type=int, default=2500, help="当未指定 start 时：向前回溯的 bar 数量粗估")

    ap.add_argument("--out", type=str, default="divergence.html", help="输出 HTML 文件名")
    ap.add_argument("--png", type=str, default="", help="可选：导出 PNG")

    ap.add_argument("--debug_ts", type=str, default="", help="调试单根 K: 形如 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--debug_indicator", type=str, default="hist", help="调试哪个指标")

    ap.add_argument("--prd", type=int, default=5, help="Pivot Period")
    ap.add_argument("--source", type=str, default="Close", choices=["Close", "High/Low"])
    ap.add_argument("--searchdiv", type=str, default="Regular", choices=["Regular", "Hidden", "Regular/Hidden"])
    ap.add_argument("--showindis", type=str, default="Full", choices=["Full", "First Letter", "Don't Show"])
    ap.add_argument("--showlimit", type=int, default=1, help="Minimum Number of Divergence")
    ap.add_argument("--maxpp", type=int, default=10, help="Maximum Pivot Points to Check")
    ap.add_argument("--maxbars", type=int, default=100, help="Maximum Bars to Check")
    ap.add_argument("--shownum", action="store_true", help="Show Divergence Number")
    ap.add_argument("--no-shownum", action="store_true", help="Disable shownum")
    ap.add_argument("--showlast", action="store_true", help="Show Only Last Divergence")
    ap.add_argument("--dontconfirm", action="store_true", help="Don't Wait for Confirmation")
    ap.add_argument("--showlines", action="store_true", help="Show Divergence Lines")
    ap.add_argument("--no-showlines", action="store_true", help="Disable lines")
    ap.add_argument("--showpivot", action="store_true", help="Show Pivot Points")

    def add_bool(name: str):
        ap.add_argument(f"--{name}", action="store_true", help=f"Enable {name}")
        ap.add_argument(f"--no-{name}", action="store_true", help=f"Disable {name}")

    for k in ["calcmacd","calcmacda","calcrsi","calcstoc","calccci","calcmom","calcobv","calcvwmacd","calccmf","calcmfi","calcext"]:
        add_bool(k)

    args = ap.parse_args()

    end_dt = pd.to_datetime(args.end) if args.end else pd.Timestamp.now()
    if args.start:
        start_dt = pd.to_datetime(args.start)
    else:
        f = str(args.freq).lower().strip()
        if f in ("5","5m","m5"):
            start_dt = end_dt - pd.Timedelta(minutes=5*args.bars)
        elif f in ("15","15m","m15"):
            start_dt = end_dt - pd.Timedelta(minutes=15*args.bars)
        elif f in ("30","30m","m30"):
            start_dt = end_dt - pd.Timedelta(minutes=30*args.bars)
        elif f in ("60","60m","1h","h1","hour"):
            start_dt = end_dt - pd.Timedelta(hours=1*args.bars)
        elif f in ("w","1w","week","weekly"):
            start_dt = end_dt - pd.Timedelta(days=7*args.bars)
        elif f in ("mon","m","1m","month","monthly"):
            start_dt = end_dt - pd.Timedelta(days=30*args.bars)
        else:
            start_dt = end_dt - pd.Timedelta(days=1*args.bars)

    df = fetch_bars_pytdx(args.symbol, args.freq, start=str(start_dt), end=str(end_dt))
    if df.empty:
        raise RuntimeError(f"无数据：symbol={args.symbol} freq={args.freq} start={start_dt} end={end_dt}")

    cfg = DivConfig(
        prd=args.prd,
        source=args.source,
        searchdiv=args.searchdiv,
        showindis=args.showindis,
        showlimit=args.showlimit,
        maxpp=args.maxpp,
        maxbars=args.maxbars,
        shownum=(False if args.no_shownum else (True if args.shownum else True)),
        showlast=args.showlast,
        dontconfirm=args.dontconfirm,
        showlines=(False if args.no_showlines else (True if args.showlines else True)),
        showpivot=args.showpivot,
    )

    if args.debug_ts:
        cfg.debug_ts = args.debug_ts.strip()
        cfg.debug_indicator = (args.debug_indicator or "hist").strip().lower()

    for k in ["calcmacd","calcmacda","calcrsi","calcstoc","calccci","calcmom","calcobv","calcvwmacd","calccmf","calcmfi","calcext"]:
        if getattr(args, f"no_{k}".replace("-", "_"), False):
            setattr(cfg, k, False)
        elif getattr(args, k, False):
            setattr(cfg, k, True)

    title = f"{args.symbol} {args.freq} Divergence (prd={cfg.prd}, {cfg.searchdiv})"
    build_plot(df, cfg, title=title, out_html=args.out, out_png=args.png)


if __name__ == "__main__":
    main()
