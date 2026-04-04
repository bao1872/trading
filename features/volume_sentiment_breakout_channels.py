
# -*- coding: utf-8 -*-
"""
Volume Sentiment Breakout Channels [AlgoAlpha] -> Python (Plotly)

What this script does
- Fetches A-share OHLCV using pytdx (same overall framework style as the uploaded
  dynamic_swing_anchored_vwap.py).
- Ports the core logic of:
  1) Channel formation when `upper` (stdev of normalized price) crosses above
     `lower` (WMA of that stdev), with duration > 10
  2) Active channel boxes with upper/lower sentiment zones
  3) Channel breakouts above/below
  4) Right-side sentiment profile (volume-by-price signed by candle direction)
  5) Post-break bullish/bearish trailing trend lines
- Exports an interactive Plotly HTML.

Notes
- This is a faithful Python port of the indicator logic, but TradingView box/line
  objects are approximated with Plotly shapes and traces.
- Profile rendering is updated on each active channel using the latest channel span,
  matching the Pine logic conceptually.

Usage
    python volume_sentiment_breakout_channels.py \
      --symbol 300136 --freq d --bars 260 \
      --length-norm 100 --length-box 14 \
      --strong --out 300136_sentiment_breakout.html
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 添加项目根目录到路径（支持直接运行脚本）
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from datasource.pytdx_client import connect_pytdx


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


def fetch_kline_pytdx(symbol: str, start: str, end: str, freq, api=None) -> pd.DataFrame:
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
                d["date"] = pd.to_datetime(d["datetime"]).tz_localize(None)
            else:
                raise RuntimeError("pytdx 返回数据缺少 datetime 列")
            if "vol" in d.columns:
                d = d.rename(columns={"vol": "volume"})
            d = d[["date", "open", "high", "low", "close", "volume"]]
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
        if should_close:
            api.disconnect()


# =========================
# Helpers
# =========================

def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / max(length, 1), adjust=False).mean()


def atr_wilder(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rma(tr, length)


def rolling_wma(series: pd.Series, length: int) -> pd.Series:
    full_weights = np.arange(1, length + 1, dtype=float)

    def _wma(x: np.ndarray) -> float:
        if len(x) == 0 or np.isnan(x).all():
            return np.nan
        w = full_weights[-len(x):]
        return float(np.dot(x, w) / w.sum())

    return series.rolling(length, min_periods=1).apply(_wma, raw=True)


def barssince(cond: pd.Series) -> pd.Series:
    out = np.full(len(cond), np.nan, dtype=float)
    last_true = None
    for i, flag in enumerate(cond.fillna(False).to_numpy(bool)):
        if flag:
            last_true = i
            out[i] = 0
        else:
            out[i] = np.nan if last_true is None else i - last_true
    return pd.Series(out, index=cond.index)


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def alpha_color(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def blend_with_bg(hex_color: str, bg: str = "#0b0f14", ratio: float = 0.5) -> str:
    h1 = hex_color.lstrip("#")
    h2 = bg.lstrip("#")
    r1, g1, b1 = int(h1[0:2], 16), int(h1[2:4], 16), int(h1[4:6], 16)
    r2, g2, b2 = int(h2[0:2], 16), int(h2[2:4], 16), int(h2[4:6], 16)
    r = int(r1 * (1 - ratio) + r2 * ratio)
    g = int(g1 * (1 - ratio) + g2 * ratio)
    b = int(b1 * (1 - ratio) + b2 * ratio)
    return f"rgb({r},{g},{b})"


# =========================
# Core structures
# =========================

@dataclass
class Channel:
    left_idx: int
    right_idx: int
    top: float
    bottom: float
    upper_zone_bottom: float
    lower_zone_top: float
    center_y: float
    breakout: Optional[str] = None
    breakout_idx: Optional[int] = None
    current_sentiment_text: str = ""
    overall_sentiment_text: str = ""
    current_sentiment_color: str = ""
    profile_bins: list = field(default_factory=list)


# =========================
# Sentiment profile
# =========================

def profile_calculation(df: pd.DataFrame,
                        start_idx: int,
                        end_idx: int,
                        res: int,
                        scale: int,
                        top_truncation: Optional[float],
                        bottom_truncation: Optional[float]):

    sub = df.iloc[start_idx:end_idx + 1]
    if sub.empty:
        return [], [], [], [], np.nan

    highs = sub["high"].to_numpy(float)
    lows = sub["low"].to_numpy(float)
    volumes = sub["volume"].to_numpy(float)
    dirs = np.where(sub["close"].to_numpy(float) > sub["open"].to_numpy(float), 1.0, -1.0)

    maxx = float(np.nanmax(highs))
    minn = float(np.nanmin(lows))

    if top_truncation is not None:
        maxx = min(maxx, float(top_truncation))
    if bottom_truncation is not None:
        minn = max(minn, float(bottom_truncation))

    if (not np.isfinite(maxx)) or (not np.isfinite(minn)) or maxx <= minn:
        return [], [], [], [], np.nan

    step = (maxx - minn) / res
    top_boundaries = []
    bottom_boundaries = []
    binlen = []
    bintype = []

    for i in range(res):
        bottom = minn + i * step
        top = minn + (i + 1) * step
        bottom_boundaries.append(bottom)
        top_boundaries.append(top)

        mask = ~((lows > top) | (highs < bottom))
        v = volumes[mask]
        d = dirs[mask]

        bin_size = float(v.sum()) if len(v) else 0.0
        bin_signed = float((v * d).sum()) if len(v) else 0.0
        binlen.append(bin_size)
        bintype.append(bin_signed)

    if len(binlen) == 0 or max(binlen) <= 0:
        poc = np.nan
    else:
        boci = int(np.argmax(binlen))
        poc = (top_boundaries[boci] + bottom_boundaries[boci]) / 2.0

    return top_boundaries, bottom_boundaries, binlen, bintype, poc


# =========================
# Core indicator
# =========================

def compute_indicator(
    df: pd.DataFrame,
    length_norm: int = 100,
    length_box: int = 14,
    strong: bool = True,
    overlap: bool = False,
    res: int = 20,
    scale: int = 30,
    green: str = "#00ffbb",
    red: str = "#ff1100",
):
    d = df.copy()

    lowest_low = d["low"].rolling(length_norm, min_periods=1).min()
    highest_high = d["high"].rolling(length_norm, min_periods=1).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    normalized_price = (d["close"] - lowest_low) / denom
    vol = normalized_price.rolling(14, min_periods=1).std()
    upper = vol
    lower = rolling_wma(vol, length_box)

    cross_lo_up = crossover(lower, upper)
    cross_up_lo = crossover(upper, lower)

    bs = barssince(cross_lo_up)
    duration = np.maximum(bs.fillna(0).to_numpy(int), 1)
    duration_s = pd.Series(duration, index=d.index)

    # ta.highest(duration) / ta.lowest(duration) => price default sources high/low
    h = pd.Series(index=d.index, dtype=float)
    l = pd.Series(index=d.index, dtype=float)
    for i in range(len(d)):
        dur = int(max(duration[i], 1))
        start = max(0, i - dur + 1)
        h.iloc[i] = d["high"].iloc[start:i + 1].max()
        l.iloc[i] = d["low"].iloc[start:i + 1].min()

    vola = atr_wilder(d, length_box) / 2.0

    channels: List[Channel] = []
    bullish_break = pd.Series(False, index=d.index)
    bearish_break = pd.Series(False, index=d.index)
    new_channel = pd.Series(False, index=d.index)

    def can_create(t_new: float, b_new: float) -> bool:
        if overlap:
            return True
        for ch in channels:
            if ch.breakout is None:
                if (t_new > ch.bottom) and (b_new < ch.top):
                    return False
        return True

    for i in range(len(d)):
        if bool(cross_up_lo.iloc[i]) and duration[i] > 10:
            top_i = float(h.iloc[i])
            bot_i = float(l.iloc[i])
            if np.isfinite(top_i) and np.isfinite(bot_i) and can_create(top_i, bot_i):
                left_idx = max(0, i - int(duration[i]))
                center_y = (top_i + bot_i) / 2.0
                vola_i = float(vola.iloc[i]) if np.isfinite(vola.iloc[i]) else 0.0
                ch = Channel(
                    left_idx=left_idx,
                    right_idx=i,
                    top=top_i,
                    bottom=bot_i,
                    upper_zone_bottom=top_i - vola_i,
                    lower_zone_top=bot_i + vola_i,
                    center_y=center_y,
                )
                top_b, bot_b, binlen, bintype, poc = profile_calculation(
                    d, left_idx, i, res, scale, ch.top, ch.bottom
                )
                ch.profile_bins = list(zip(top_b, bot_b, binlen, bintype))
                if len(top_b) and len(bintype):
                    cur_idx = None
                    c = float(d["close"].iloc[i])
                    for kk in range(len(top_b)):
                        if c >= bot_b[kk] and c <= top_b[kk]:
                            cur_idx = kk
                            break
                    if cur_idx is not None:
                        max_abs_bt = max(max(abs(x) for x in bintype), 1e-12)
                        cur_bt = bintype[cur_idx]
                        ch.current_sentiment_text = f"Current Sentiment: {cur_bt / max_abs_bt * 100:.1f}%"
                        ch.current_sentiment_color = green if cur_bt >= 0 else red
                        channel_sum = float(sum(bintype))
                        ch.overall_sentiment_text = f"Overall Sentiment: {'Bullish' if channel_sum > 0 else 'Bearish'}"
                channels.insert(0, ch)
                new_channel.iloc[i] = True

        # update active channels
        to_remove = []
        for idx, ch in enumerate(channels):
            if ch.breakout is not None:
                continue

            price_to_compare = float((d["close"].iloc[i] + d["open"].iloc[i]) / 2.0) if strong else float(d["close"].iloc[i])

            if price_to_compare > ch.top:
                ch.breakout = "bull"
                ch.breakout_idx = i
                ch.right_idx = i
                bullish_break.iloc[i] = True
                to_remove.append(idx)
            elif price_to_compare < ch.bottom:
                ch.breakout = "bear"
                ch.breakout_idx = i
                ch.right_idx = i
                bearish_break.iloc[i] = True
                to_remove.append(idx)
            else:
                ch.right_idx = i
                top_b, bot_b, binlen, bintype, poc = profile_calculation(
                    d, ch.left_idx, i, res, scale, ch.top, ch.bottom
                )
                ch.profile_bins = list(zip(top_b, bot_b, binlen, bintype))
                if len(top_b) and len(bintype):
                    cur_idx = None
                    c = float(d["close"].iloc[i])
                    for kk in range(len(top_b)):
                        if c >= bot_b[kk] and c <= top_b[kk]:
                            cur_idx = kk
                            break
                    if cur_idx is not None:
                        max_abs_bt = max(max(abs(x) for x in bintype), 1e-12)
                        cur_bt = bintype[cur_idx]
                        ch.current_sentiment_text = f"Current Sentiment: {cur_bt / max_abs_bt * 100:.1f}%"
                        ch.current_sentiment_color = green if cur_bt >= 0 else red
                        channel_sum = float(sum(bintype))
                        ch.overall_sentiment_text = f"Overall Sentiment: {'Bullish' if channel_sum > 0 else 'Bearish'}"

    # post-break trend lines
    bs_bull = barssince(bullish_break)
    bull_len = np.maximum(bs_bull.fillna(0).to_numpy(int), 1)
    trnd_up = pd.Series(index=d.index, dtype=float)
    active_up = False
    bull_line = pd.Series(np.nan, index=d.index, dtype=float)

    for i in range(len(d)):
        if bullish_break.iloc[i]:
            active_up = True
        trnd_up.iloc[i] = d["low"].iloc[max(0, i - bull_len[i] + 1): i + 1].mean()
        if active_up:
            bull_line.iloc[i] = trnd_up.iloc[i]
            if i > 0 and d["close"].iloc[i] < trnd_up.iloc[i] and d["close"].iloc[i - 1] >= trnd_up.iloc[i - 1]:
                active_up = False
                bull_line.iloc[i] = np.nan

    bs_bear = barssince(bearish_break)
    bear_len = np.maximum(bs_bear.fillna(0).to_numpy(int), 1)
    trnd_dn = pd.Series(index=d.index, dtype=float)
    active_dn = False
    bear_line = pd.Series(np.nan, index=d.index, dtype=float)

    for i in range(len(d)):
        if bearish_break.iloc[i]:
            active_dn = True
        trnd_dn.iloc[i] = d["high"].iloc[max(0, i - bear_len[i] + 1): i + 1].mean()
        if active_dn:
            bear_line.iloc[i] = trnd_dn.iloc[i]
            if i > 0 and d["close"].iloc[i] > trnd_dn.iloc[i] and d["close"].iloc[i - 1] <= trnd_dn.iloc[i - 1]:
                active_dn = False
                bear_line.iloc[i] = np.nan

    return {
        "upper": upper,
        "lower": lower,
        "duration": duration_s,
        "high_track": h,
        "low_track": l,
        "vola": vola,
        "channels": channels,
        "new_channel": new_channel,
        "bullish_break": bullish_break,
        "bearish_break": bearish_break,
        "bull_line": bull_line,
        "bear_line": bear_line,
    }


# =========================
# Plot
# =========================

def build_plot(
    df: pd.DataFrame,
    result: dict,
    out_html: str,
    title: str,
    green: str = "#00ffbb",
    red: str = "#ff1100",
    scale: int = 30,
):
    bg = "#0b0f14"
    fg = "#c9d1d9"
    grid = "rgba(255,255,255,0.06)"

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            name="K线",
        )
    )

    # active / historical channels
    channels = result["channels"]
    last_x = df.index[-1]
    x_step = df.index[-1] - df.index[-2] if len(df.index) > 1 else pd.Timedelta(days=1)

    for ch in channels:
        x0 = df.index[ch.left_idx]
        x1 = df.index[ch.right_idx]

        # main channel
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=ch.bottom, y1=ch.top,
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(201,209,217,0.10)",
            layer="below",
        )
        # upper zone
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=ch.upper_zone_bottom, y1=ch.top,
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor=alpha_color(red, 0.28),
            layer="below",
        )
        # lower zone
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=ch.bottom, y1=ch.lower_zone_top,
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor=alpha_color(green, 0.28),
            layer="below",
        )
        # center line + zone boundaries
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[ch.center_y, ch.center_y],
            mode="lines", line=dict(color="rgba(201,209,217,0.55)", width=1, dash="dash"),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[ch.top, ch.top],
            mode="lines", line=dict(color=red, width=2),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[ch.upper_zone_bottom, ch.upper_zone_bottom],
            mode="lines", line=dict(color=alpha_color(red, 0.65), width=1, dash="dot"),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[ch.lower_zone_top, ch.lower_zone_top],
            mode="lines", line=dict(color=alpha_color(green, 0.65), width=1, dash="dot"),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[ch.bottom, ch.bottom],
            mode="lines", line=dict(color=green, width=2),
            hoverinfo="skip", showlegend=False
        ))

        # sentiment profile to the right
        if ch.profile_bins:
            max_bin = max([b[2] for b in ch.profile_bins] + [1.0])
            profile_right = x1 + x_step * (7 + scale)
            for top_b, bot_b, binlen, bintype in ch.profile_bins:
                ratio = 0.0 if max_bin <= 0 else binlen / max_bin
                width_steps = max(1, int(round(ratio * scale)))
                left = profile_right - x_step * width_steps
                color_base = green if bintype > 0 else red
                mag = 0.0
                if ch.profile_bins:
                    max_abs_bt = max(max(abs(b[3]) for b in ch.profile_bins), 1e-12)
                    mag = abs(bintype) / max_abs_bt
                col = blend_with_bg(color_base, bg=bg, ratio=0.75 * (1 - mag))
                fig.add_shape(
                    type="rect",
                    x0=left, x1=profile_right, y0=bot_b, y1=top_b,
                    line=dict(color=col, width=1),
                    fillcolor=col,
                    layer="above",
                )

        # text annotations
        text_y1 = ch.lower_zone_top if df["close"].iloc[ch.right_idx] > ch.center_y else ch.top
        text_y2 = ch.top if df["close"].iloc[ch.right_idx] > ch.center_y else ch.bottom
        if ch.current_sentiment_text:
            fig.add_annotation(
                x=x1, y=text_y1,
                text=ch.current_sentiment_text,
                font=dict(color=ch.current_sentiment_color or fg, size=11),
                bgcolor="rgba(0,0,0,0.35)",
                bordercolor="rgba(255,255,255,0.10)",
                xanchor="right",
                showarrow=False,
            )
        if ch.overall_sentiment_text:
            fig.add_annotation(
                x=x1, y=text_y2,
                text=ch.overall_sentiment_text,
                font=dict(color=fg, size=11),
                bgcolor="rgba(0,0,0,0.35)",
                bordercolor="rgba(255,255,255,0.10)",
                xanchor="right",
                showarrow=False,
            )

    # post-break trend lines
    bull_line = result["bull_line"]
    bear_line = result["bear_line"]

    fig.add_trace(go.Scatter(
        x=df.index, y=bull_line, mode="lines",
        line=dict(color=green, width=2), name="Bullish Trend",
        connectgaps=False
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=bear_line, mode="lines",
        line=dict(color=red, width=2), name="Bearish Trend",
        connectgaps=False
    ))

    # fills versus mid price
    midp = (df["close"] + df["open"]) / 2.0

    fig.add_trace(go.Scatter(
        x=df.index, y=midp, mode="lines", line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=bull_line, mode="lines", line=dict(color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor=alpha_color(green, 0.15),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=midp, mode="lines", line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=bear_line, mode="lines", line=dict(color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor=alpha_color(red, 0.15),
        hoverinfo="skip", showlegend=False
    ))

    # breakout markers
    bull_idx = np.where(result["bullish_break"].to_numpy(bool))[0]
    bear_idx = np.where(result["bearish_break"].to_numpy(bool))[0]

    if len(bull_idx):
        y = [float(np.min(df["low"].iloc[max(0, i - 2): i + 1])) for i in bull_idx]
        fig.add_trace(go.Scatter(
            x=df.index[bull_idx], y=y, mode="markers+text",
            marker=dict(symbol="triangle-up", size=10, color=green),
            text=["▲"] * len(bull_idx), textposition="top center",
            name="Bullish Breakout"
        ))
    if len(bear_idx):
        y = [float(np.max(df["high"].iloc[max(0, i - 2): i + 1])) for i in bear_idx]
        fig.add_trace(go.Scatter(
            x=df.index[bear_idx], y=y, mode="markers+text",
            marker=dict(symbol="triangle-down", size=10, color=red),
            text=["▼"] * len(bear_idx), textposition="bottom center",
            name="Bearish Breakout"
        ))

    fig.update_layout(
        title=title,
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=fg),
        height=900,
        margin=dict(l=50, r=50, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor=grid, rangeslider_visible=False),
        yaxis=dict(showgrid=True, gridcolor=grid, title="Price"),
        legend=dict(orientation="h", x=0.01, y=1.03),
    )

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_html}")


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--symbol", type=str, default="300136", help="A股代码")
    ap.add_argument("--freq", type=str, default="d", help="K线周期：d 或 1/5/15/30/60")
    ap.add_argument("--bars", type=int, default=260, help="展示最近N根K线")
    ap.add_argument("--out", type=str, default="volume_sentiment_breakout_channels.html", help="输出HTML文件")

    ap.add_argument("--length-norm", type=int, default=100, help="Normalization Length")
    ap.add_argument("--length-box", type=int, default=14, help="Box Detection Length")
    ap.add_argument("--strong", action="store_true", help="Strong closes only")
    ap.add_argument("--green", type=str, default="#00ffbb", help="Bullish colour")
    ap.add_argument("--red", type=str, default="#ff1100", help="Bearish colour")
    ap.add_argument("--res", type=int, default=20, help="Profile resolution")
    ap.add_argument("--scale", type=int, default=30, help="Profile horizontal scale")
    args = ap.parse_args()

    end = datetime.now().date()
    freq_arg = args.freq.strip().lower()
    if freq_arg in ["d", "day", "daily", "101"]:
        freq = "d"
        start = end - timedelta(days=max(900, int(args.bars * 4)))
    else:
        try:
            freq = int(freq_arg)
        except ValueError as e:
            raise ValueError("--freq 只能是 d 或 1/5/15/30/60") from e
        start = end - timedelta(days=900)

    df = fetch_kline_pytdx(args.symbol, str(start), str(end), freq)
    df_show = df.tail(args.bars).copy()

    result = compute_indicator(
        df_show,
        length_norm=args.length_norm,
        length_box=args.length_box,
        strong=bool(args.strong),
        overlap=False,
        res=args.res,
        scale=args.scale,
        green=args.green,
        red=args.red,
    )

    title = (
        f"Volume Sentiment Breakout Channels [AlgoAlpha] - {args.symbol} "
        f"(norm={args.length_norm}, box={args.length_box}, strong={bool(args.strong)})"
    )

    build_plot(
        df_show,
        result,
        out_html=args.out,
        title=title,
        green=args.green,
        red=args.red,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
