# -*- coding: utf-8 -*-
"""
Dynamic Swing Anchored VWAP (Zeiierman) -> Python (Plotly)

What this script does
- Fetches A-share daily OHLCV using qstock.get_data (same style as your attached data-source script).
- Ports the Pine v6 logic of:
  1) Detecting swing highs/lows by `ta.highestbars(high, prd) == 0` and `ta.lowestbars(low, prd) == 0`
  2) Determining direction `dir = phL > plL ? 1 : -1`
  3) When `dir` flips, re-anchor the VWAP at the last pivot bar and rebuild the polyline points
  4) Otherwise, update p/vol with an EWMA alpha derived from APT (Adaptive Price Tracking)

Experiment setting
- Example symbol: 山东黄金 = 600547
- “220日线” experiment object: this script overlays SMA(220) as the baseline line on the same panel.
- All Zeiierman indicator parameters keep Pine defaults:
    prd=50, baseAPT=20, useAdapt=False, volBias=10

Usage
    python dynamic_swing_anchored_vwap.py --symbol 600547 --bars 220 --out sdhj_vwap.html

Dependencies
    pip install pandas numpy plotly qstock

Notes
- Pine uses `hlc3 = (high+low+close)/3` and volume-weighted EWMA of (hlc3*vol) and (vol).
- If your qstock fork differs, adjust fetch_daily_qstock() signature mapping.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目根目录到路径（支持直接运行脚本）
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from datasource.pytdx_client import connect_pytdx, PERIOD_MAP


# =========================
# Data source (pytdx)
# =========================

def _category_from_freq(freq: str) -> int:
    f = str(freq).strip().lower()
    if f in ("d", "day", "daily", "101"):
        return 4
    if f in ("60", "60m", "1h"):
        return 3
    if f in ("30", "30m"):
        return 2
    if f in ("15", "15m"):
        return 1
    if f in ("5", "5m"):
        return 0
    if f in ("1", "1m"):
        return 7
    raise ValueError(f"不支持的频率: {freq}")


def _connect_pytdx() -> TdxHq_API:
    servers = [
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
    last_errors = []
    for host, port in servers:
        try:
            api = TdxHq_API(raise_exception=True, auto_retry=True)
            if api.connect(host, port):
                return api
        except TdxConnectionError as exc:
            last_errors.append(f"{host}:{port} {exc}")
        except Exception as exc:
            last_errors.append(f"{host}:{port} {exc}")
    err = "; ".join(last_errors[-5:])
    raise RuntimeError(f"pytdx连接失败; {err}")


def fetch_kline_pytdx(symbol: str, start: str, end: str, freq, api=None) -> pd.DataFrame:
    """
    从 pytdx 获取 K 线数据
    
    Args:
        symbol: 股票代码
        start: 开始日期
        end: 结束日期
        freq: 频率
        api: pytdx API 连接（可选，用于复用连接）
    """
    # 使用传入的 api 或创建新连接
    if api is None:
        api = connect_pytdx()
        should_close = True
    else:
        should_close = False
    
    try:
        cat = _category_from_freq(freq)
        mkt = 1 if symbol.startswith("6") else 0
        page = 0
        size = 700
        frames = []
        while True:
            recs = api.get_security_bars(cat, mkt, symbol, page * size, size)
            if not recs:
                break
            d = pd.DataFrame(recs)
            if "datetime" in d.columns:
                d["date"] = pd.to_datetime(d["datetime"])
            elif {"year", "month", "day", "hour", "minute"}.issubset(d.columns):
                d["date"] = pd.to_datetime(d[["year", "month", "day", "hour", "minute"]].astype(int))
            else:
                raise RuntimeError("pytdx 返回数据缺少时间列")
            if "vol" in d.columns:
                d = d.rename(columns={"vol": "volume"})
            d = d[["date", "open", "high", "low", "close"] + (["volume"] if "volume" in d.columns else [])]
            frames.append(d.sort_values("date"))
            if len(recs) < size:
                break
            page += 1
        if not frames:
            raise RuntimeError("pytdx 无数据")
        all_df = pd.concat(frames).sort_values("date")
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        all_df = all_df[(all_df["date"] >= start_dt) & (all_df["date"] <= end_dt)]
        all_df = all_df.drop_duplicates(subset=["date"], keep="last").set_index("date")
        return all_df
    finally:
        # 只在函数内部创建的连接才需要关闭
        if should_close:
            api.disconnect()



# =========================
# Zeiierman Pine v6 defaults
# =========================

@dataclass
class DSAConfig:
    # Swing detection
    prd: int = 50

    # Adaptive Price Tracking
    baseAPT: float = 20.0
    useAdapt: bool = False
    volBias: float = 10.0

    # ATR settings (fixed in code)
    atrLen: int = 50

    # Plot
    line_width: int = 2


# =========================
# Helpers (Pine-aligned)
# =========================


def hlc3(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    """Pine ta.atr(n): Wilder's ATR."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Wilder's RMA ~= EWM(alpha=1/n)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def alpha_from_apt(apt: float) -> float:
    """Pine: alpha from APT (half-life -> EWMA alpha)

    decay = exp(-ln(2)/apt)
    alpha = 1 - decay
    """
    apt = max(1.0, float(apt))
    decay = np.exp(-np.log(2.0) / apt)
    return float(1.0 - decay)


# =========================
# Core port: Dynamic Swing Anchored VWAP
# =========================



# =========================
# Core port: Dynamic Swing Anchored VWAP (Pine-exact)
# =========================


def dynamic_swing_anchored_vwap(df: pd.DataFrame, cfg: DSAConfig):
    """
    Pine v6 逐行对齐版（并返回 polyline segments 以匹配 TradingView 的绘制方式）
    Returns:
      vwap: pd.Series
      dir_series: pd.Series (1/-1)
      pivot_labels: list[dict]
      segments: list[dict]  # each: {"dir": int, "x": np.ndarray(datetime64), "y": np.ndarray(float)}
    """
    d = df.copy()
    d["hlc3"] = hlc3(d)

    # --- APT series (Pine-consistent) ---
    atr = atr_wilder(d, cfg.atrLen)
    atr_avg = atr.ewm(alpha=1 / cfg.atrLen, adjust=False).mean()  # ta.rma(atr, atrLen)
    ratio = np.where(atr_avg.values > 0, atr.values / atr_avg.values, 1.0)

    if cfg.useAdapt:
        apt_raw = cfg.baseAPT / np.power(ratio, cfg.volBias)
    else:
        apt_raw = np.full(len(d), cfg.baseAPT, dtype=float)

    apt_clamped = np.clip(apt_raw, 5.0, 300.0)
    apt_series = np.rint(apt_clamped).astype(int)

    high = d["high"].to_numpy(float)
    low  = d["low"].to_numpy(float)
    volu = d["volume"].to_numpy(float)
    h3   = d["hlc3"].to_numpy(float)

    n = len(d)
    if n < 2:
        raise ValueError("数据长度太短")

    # --- Pine swing tracking vars ---
    ph = np.nan
    pl = np.nan
    phL = 0
    plL = 0

    # label helper var
    prev = np.nan
    ph_prev_store = np.nan
    pl_prev_store = np.nan

    # EWMA state (Pine var p/vol)
    p = h3[0] * volu[0]
    v = volu[0]

    vwap_out = np.full(n, np.nan, dtype=float)
    dir_out  = np.full(n, np.nan, dtype=float)
    pivot_labels = []

    # --- segment drawing (match Pine polyline behavior) ---
    segments = []
    cur_points_x = []
    cur_points_y = []
    cur_dir = None

    last_dir = None

    for t in range(n):
        # pivot update: st=max(0, t-prd+1)
        st = 0 if (t - cfg.prd + 1) < 0 else (t - cfg.prd + 1)
        win_h = high[st : t + 1]
        win_l = low[st : t + 1]

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

        # init segment dir
        if cur_dir is None:
            cur_dir = dir_

        if dir_ != last_dir:
            # --- freeze previous polyline segment (as Pine does before clearing points) ---
            if len(cur_points_x) >= 2:
                segments.append({
                    "dir": int(last_dir),
                    "x": np.array(cur_points_x),
                    "y": np.array(cur_points_y, dtype=float),
                })

            # anchor
            x_anchor = plL if dir_ > 0 else phL
            y_anchor = pl  if dir_ > 0 else ph

            # label text (same as Pine)
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

            pivot_labels.append({
                "t": int(x_anchor),
                "x": d.index[x_anchor],
                "y": float(y_anchor) if np.isfinite(y_anchor) else np.nan,
                "text": txt,
                "dir": int(dir_)
            })

            # prev := dir>0 ? ph[1] : pl[1]
            prev = ph_prev_store if dir_ > 0 else pl_prev_store

            # reset EWMA at anchor
            p = y_anchor * volu[x_anchor]
            v = volu[x_anchor]

            # clear points & rebuild from anchor..t (this is the big visual difference vs one series)
            cur_points_x = []
            cur_points_y = []
            cur_dir = dir_

            for k in range(x_anchor, t + 1):
                alpha = alpha_from_apt(float(apt_series[k]))
                pxv = h3[k] * volu[k]
                v_i = volu[k]
                p = (1.0 - alpha) * p + alpha * pxv
                v = (1.0 - alpha) * v + alpha * v_i
                vv = (p / v) if v > 0 else np.nan
                vwap_out[k] = vv
                cur_points_x.append(d.index[k])
                cur_points_y.append(vv)

            last_dir = dir_

        else:
            alpha = alpha_from_apt(float(apt_series[t]))
            pxv = h3[t] * volu[t]
            v0  = volu[t]
            p = (1.0 - alpha) * p + alpha * pxv
            v = (1.0 - alpha) * v + alpha * v0
            vv = (p / v) if v > 0 else np.nan
            vwap_out[t] = vv

            # extend current segment points (like Pine points.push)
            cur_points_x.append(d.index[t])
            cur_points_y.append(vv)

        ph_prev_store = ph
        pl_prev_store = pl

    # finalize last segment
    if len(cur_points_x) >= 2:
        segments.append({
            "dir": int(last_dir if last_dir is not None else cur_dir),
            "x": np.array(cur_points_x),
            "y": np.array(cur_points_y, dtype=float),
        })

    vwap_series = pd.Series(vwap_out, index=d.index, name="DSA_VWAP")
    dir_series  = pd.Series(dir_out,  index=d.index, name="DSA_DIR")
    return vwap_series, dir_series, pivot_labels, segments




def compute_extra_factors(
    df: pd.DataFrame,
    vwap: pd.Series,
    dir_series: pd.Series,
    pivot_labels: list[dict],
) -> pd.DataFrame:
    """
    计算两个附加因子（尽量保持原脚本框架不变）：

    1) lh_hh_low_pos:
       最近一次 LH/HH 拐点 到 当前区间最低点 之间，当前收盘价所处的位置，范围 [0, 1]
       公式：
         pos = (close - min_low_since_last_lh_hh) / (last_lh_hh_price - min_low_since_last_lh_hh)

    2) bull_vwap_dev_pct:
       仅当当前处于多头状态(dir > 0)时，计算 (close - vwap) / vwap * 100
       若为空头，则直接记为 0
    """
    out = pd.DataFrame(index=df.index)

    label_map = {}
    for lab in pivot_labels:
        x = lab.get("x")
        txt = str(lab.get("text", "")).strip().upper()
        if x in df.index and txt in {"LH", "HH"}:
            label_map[x] = {
                "text": txt,
                "price": float(lab.get("y")) if pd.notna(lab.get("y")) else np.nan,
            }

    latest_lh_hh_idx = None
    latest_lh_hh_price = np.nan
    pos_values = []

    lows = df["low"].astype(float)
    closes = df["close"].astype(float)

    for ts in df.index:
        if ts in label_map:
            latest_lh_hh_idx = ts
            latest_lh_hh_price = float(label_map[ts]["price"])

        if latest_lh_hh_idx is None or not np.isfinite(latest_lh_hh_price):
            pos_values.append(np.nan)
            continue

        window_low = float(lows.loc[latest_lh_hh_idx:ts].min())
        denom = latest_lh_hh_price - window_low

        if not np.isfinite(window_low) or denom <= 0:
            pos = np.nan
        else:
            pos = (float(closes.loc[ts]) - window_low) / denom
            pos = float(np.clip(pos, 0.0, 1.0))
        pos_values.append(pos)

    out["lh_hh_low_pos"] = pos_values

    vwap_safe = vwap.astype(float)
    bull_dev = np.where(
        (dir_series.astype(float) > 0) & np.isfinite(vwap_safe.to_numpy()) & (vwap_safe.to_numpy() != 0),
        (closes.to_numpy() - vwap_safe.to_numpy()) / vwap_safe.to_numpy() * 100.0,
        0.0,
    )
    out["bull_vwap_dev_pct"] = bull_dev

    return out


# =========================
# Plot
# =========================
# Plot
# =========================


def compute_daily_sma_on_intraday(df: pd.DataFrame, days: int = 220) -> pd.Series:
    """Compute a *daily* SMA(days) from intraday data, then forward-fill back to intraday index.

    This keeps the meaning of "220日线" even when you plot 15-minute bars.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be DatetimeIndex")

    tmp = df.copy()
    tmp["_date"] = tmp.index.date
    daily_close = tmp.groupby("_date")["close"].last()
    daily_sma = daily_close.rolling(days, min_periods=days).mean()

    # Map each intraday bar to its session date SMA
    mapper = {pd.to_datetime(k): v for k, v in daily_sma.items()}
    intraday_dates = pd.to_datetime(tmp["_date"].astype(str))
    sma_intraday = intraday_dates.map(mapper).astype(float)
    sma_intraday.index = df.index
    return sma_intraday


def add_feature_table(fig: go.Figure, df: pd.DataFrame, vwap: pd.Series, dir_series: pd.Series, extra_factors: pd.DataFrame | None = None) -> None:
    """Overlay a small feature table near top-left in paper coordinates."""
    if len(df) == 0:
        return
    last_ts = df.index[-1]
    last_close = float(df["close"].iloc[-1])
    last_vwap = float(vwap.iloc[-1]) if pd.notna(vwap.iloc[-1]) else np.nan
    last_dir = int(dir_series.iloc[-1]) if pd.notna(dir_series.iloc[-1]) else 0

    if np.isfinite(last_vwap) and last_vwap != 0:
        dev_pct = (last_close - last_vwap) / last_vwap * 100.0
        dev_abs = (last_close - last_vwap)
    else:
        dev_pct = np.nan
        dev_abs = np.nan

    dir_txt = "UP (+1)" if last_dir > 0 else ("DOWN (-1)" if last_dir < 0 else "N/A")

    lh_hh_low_pos = np.nan
    bull_vwap_dev_pct = np.nan
    if extra_factors is not None and len(extra_factors) > 0:
        if "lh_hh_low_pos" in extra_factors.columns:
            lh_hh_low_pos = float(extra_factors["lh_hh_low_pos"].iloc[-1]) if pd.notna(extra_factors["lh_hh_low_pos"].iloc[-1]) else np.nan
        if "bull_vwap_dev_pct" in extra_factors.columns:
            bull_vwap_dev_pct = float(extra_factors["bull_vwap_dev_pct"].iloc[-1]) if pd.notna(extra_factors["bull_vwap_dev_pct"].iloc[-1]) else np.nan

    rows = [
        ("time", last_ts.strftime("%Y-%m-%d %H:%M") if hasattr(last_ts, "strftime") else str(last_ts)),
        ("dir", dir_txt),
        ("close", f"{last_close:.3f}"),
        ("vwap", f"{last_vwap:.3f}" if np.isfinite(last_vwap) else "n/a"),
        ("close - vwap", f"{dev_abs:.3f}" if np.isfinite(dev_abs) else "n/a"),
        ("dev(vwap)", f"{dev_pct:.3f}%" if np.isfinite(dev_pct) else "n/a"),
        ("lh_hh_low_pos", f"{lh_hh_low_pos:.4f}" if np.isfinite(lh_hh_low_pos) else "n/a"),
        ("bull_vwap_dev_pct", f"{bull_vwap_dev_pct:.4f}%" if np.isfinite(bull_vwap_dev_pct) else "n/a"),
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Feature", "Value"],
                fill_color="rgba(0,0,0,0.65)",
                font=dict(color="white", size=12, family="Consolas, monospace"),
                line_color="rgba(255,255,255,0.25)",
                align=["left", "left"],
            ),
            cells=dict(
                values=[[r[0] for r in rows], [r[1] for r in rows]],
                fill_color="rgba(0,0,0,0.45)",
                font=dict(color="white", size=12, family="Consolas, monospace"),
                line_color="rgba(255,255,255,0.18)",
                align=["left", "left"],
                height=22,
            ),
            columnwidth=[0.42, 0.58],
            # domain uses paper coords: put table on the left-top
            domain=dict(x=[0.02, 0.30], y=[0.73, 0.98]),
        )
    )


def build_plot(df, vwap, dir_series, pivot_labels, segments, out_html, title, sma_days=220):
    df = df.copy()

    is_intraday = (df.index.to_series().diff().median() is not pd.NaT) and (df.index.to_series().diff().median() < pd.Timedelta("20H"))
    if is_intraday:
        df["SMA220D"] = compute_daily_sma_on_intraday(df, days=sma_days)
    else:
        df["SMA220D"] = df["close"].rolling(sma_days, min_periods=sma_days).mean()

    extra_factors = compute_extra_factors(df, vwap, dir_series, pivot_labels)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=[0.78, 0.22],
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
            showlegend=False,
        ),
        row=1, col=1
    )

    # --- Plot VWAP as TradingView-style polylines (segments) ---
    # Pine 使用 polyline，每次 dir 翻转会“冻结上一段”，并从 anchor 重新生成新段。
    # 所以这里按 segments 逐段画线，不会出现你看到的“被截断/硬连”的问题。
    for seg in segments:
        seg_dir = seg["dir"]
        color = "#ff1744" if seg_dir > 0 else "#00e676"  # dir>0 red, dir<0 green (match Pine)
        fig.add_trace(
            go.Scatter(x=seg["x"], y=seg["y"], mode="lines",
                       line=dict(width=2, color=color),
                       hoverinfo="skip", showlegend=False),
            row=1, col=1
        )

    # SMA220D


    fig.add_trace(
        go.Scatter(x=df.index, y=df["SMA220D"], mode="lines",
                   line=dict(width=1, dash="dot"),
                   hoverinfo="skip", showlegend=False),
        row=1, col=1
    )

    # --- Add HL/LH/HH/LL labels as annotations ---
    for lab in pivot_labels:
        if lab["x"] not in df.index:
            continue  # 只画当前窗口内的
        txt = lab["text"]
        if not txt:
            continue

        # up pivot labels near lows -> arrow down; down pivot labels near highs -> arrow up
        is_up = (lab["dir"] > 0)
        bgcolor = "rgba(0,230,118,0.85)" if is_up else "rgba(255,23,68,0.85)"
        ay = 25 if is_up else -25

        fig.add_annotation(
            x=lab["x"], y=lab["y"], xref="x", yref="y",
            text=txt,
            showarrow=True, arrowhead=2, ax=0, ay=ay,
            bgcolor=bgcolor,
            font=dict(color="white", size=12),
            bordercolor=bgcolor, borderwidth=1,
            row=1, col=1
        )

    # Volume
    vol_colors = np.where(df["close"] >= df["open"], "rgba(38,166,154,0.6)", "rgba(239,83,80,0.6)")
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=vol_colors, showlegend=False), row=2, col=1)

    fig.update_layout(
        title=title,
        plot_bgcolor="#0b0f14", paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=950,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=2, col=1, rangemode="tozero")

    # --- Feature table (top-left) ---
    add_feature_table(fig, df, vwap, dir_series, extra_factors=extra_factors)

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_html}")


# =========================
# CLI
# =========================


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)

    ap.add_argument("--symbol", type=str, default="600547", help="A股代码，如 600547(山东黄金)")
    ap.add_argument("--freq", type=str, default="d", help="K线周期：d(日线) 或 1/5/15/30/60(分钟)；qstock freq 使用数字分钟")
    ap.add_argument("--bars", type=int, default=220, help="用于展示/实验的最近N根K线（注意：15分钟下，220=220根15m，不是220天）")

    ap.add_argument("--out", type=str, default="dsa_vwap.html", help="输出HTML文件")

    # Keep Pine defaults unless user overrides
    ap.add_argument("--prd", type=int, default=50, help="Swing Period (Pine default=50)")
    ap.add_argument("--baseAPT", type=float, default=20.0, help="Adaptive Price Tracking (Pine default=20)")
    ap.add_argument("--useAdapt", action="store_true", help="Adapt APT by ATR ratio (default False)")
    ap.add_argument("--volBias", type=float, default=10.0, help="Volatility Bias (Pine default=10.0)")

    args = ap.parse_args()
    from datetime import datetime, timedelta

    end = datetime.now().date()
        # Parse freq
    freq_arg = args.freq.strip().lower()
    if freq_arg in ["d", "day", "daily", "101"]:
        freq = "d"  # qstock supports 'd' / 'D' / 101
        # for daily, 220-day SMA needs ~ 1.2y+ data
        start = end - timedelta(days=max(900, int(args.bars * 4)))
    else:
        # intraday minutes: qstock expects int minutes (1/5/15/30/60)
        try:
            freq = int(freq_arg)
        except ValueError as e:
            raise ValueError("--freq 只能是 d 或 1/5/15/30/60（分钟，数字）") from e

        # intraday needs more calendar days to cover daily SMA(220D) warmup
        # rough: 220 trading days ~ 330-380 calendar days, add buffer
        start = end - timedelta(days=900)

    df = fetch_kline_pytdx(args.symbol, start=str(start), end=str(end), freq=freq)

    # Slice last bars for plot, but compute vwap on full df to keep state consistent.
    cfg = DSAConfig(prd=args.prd, baseAPT=args.baseAPT, useAdapt=bool(args.useAdapt), volBias=args.volBias)
    vwap_full, dir_full, labels_full, segments_full = dynamic_swing_anchored_vwap(df, cfg)

    df_show = df.tail(args.bars)
    vwap_show = vwap_full.reindex(df_show.index)
    dir_show  = dir_full.reindex(df_show.index)

    # labels 也过滤到展示窗口
    labels_show = [x for x in labels_full if x["x"] in df_show.index]
    title = f"{args.symbol} ({args.freq}) Dynamic Swing Anchored VWAP (prd={cfg.prd}, APT={cfg.baseAPT}, useAdapt={cfg.useAdapt}) + SMA220D"
    # 可选：只画展示窗口内的 segments（避免画出窗口外的线段）
    min_x = df_show.index.min()
    max_x = df_show.index.max()

    segments_show = []
    for seg in segments_full:
        seg_x = pd.to_datetime(seg["x"])  # 统一成 Timestamp
        mask = (seg_x >= min_x) & (seg_x <= max_x)
        if mask.sum() >= 2:
            segments_show.append({
                "dir": seg["dir"],
                "x": seg_x[mask],
                "y": np.asarray(seg["y"])[mask],
            })

    build_plot(df_show, vwap_show, dir_show, labels_show, segments_show,
            out_html=args.out, title=title, sma_days=220)


if __name__ == "__main__":
    main()
