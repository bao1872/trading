# -*- coding: utf-8 -*-
"""
Trendlines with Breaks [LuxAlgo] -> Python (Plotly)

Features
- pytdx data source (same project style as user's uploaded framework)
- Plotly candlestick + automatic upper/lower trendlines
- Supports Atr / Stdev / Linreg slope methods
- Supports backpaint on/off
- Marks upward/downward breakouts with B labels
- Exports interactive HTML

Example
python trendlines_with_breaks_luxalgo.py \
    --symbol 300136 \
    --freq d \
    --bars 240 \
    --length 14 \
    --mult 1.0 \
    --calc-method Atr \
    --backpaint \
    --show-ext \
    --out 300136_trendlines_breaks.html
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Match user's framework import style
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

try:
    from datasource.pytdx_client import connect_pytdx
except Exception:
    connect_pytdx = None


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
    if connect_pytdx is None and api is None:
        raise RuntimeError("无法导入 datasource.pytdx_client.connect_pytdx，请在你的项目环境中运行。")

    if api is None:
        api = connect_pytdx()
        should_close = True
    else:
        should_close = False

    try:
        cat = _category_from_freq(freq)
        mkt = 1 if symbol.startswith(("5", "6", "9")) else 0
        page = 0
        size = 700
        frames: List[pd.DataFrame] = []

        while True:
            recs = api.get_security_bars(cat, mkt, symbol, page * size, size)
            if not recs:
                break
            d = pd.DataFrame(recs)
            if "datetime" in d.columns:
                d["date"] = pd.to_datetime(d["datetime"]).tz_localize(None)
            elif {"year", "month", "day", "hour", "minute"}.issubset(d.columns):
                d["date"] = pd.to_datetime(d[["year", "month", "day", "hour", "minute"]].astype(int))
            else:
                raise RuntimeError("pytdx 返回数据缺少时间列")

            if "vol" in d.columns:
                d = d.rename(columns={"vol": "volume"})

            required = ["date", "open", "high", "low", "close"]
            if "volume" not in d.columns:
                raise RuntimeError("pytdx 返回数据缺少 volume 列")
            d = d[required + ["volume"]]
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
            try:
                api.disconnect()
            except Exception:
                pass


@dataclass
class TLBConfig:
    length: int = 14
    mult: float = 1.0
    calc_method: str = "Atr"  # Atr | Stdev | Linreg
    backpaint: bool = True
    show_ext: bool = True
    up_color: str = "#00897b"
    down_color: str = "#e53935"


def rma_pine_like(series: pd.Series, length: int) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    if len(vals) < length:
        return pd.Series(out, index=series.index)
    first = np.nanmean(vals[:length])
    out[length - 1] = first
    alpha = 1.0 / float(length)
    prev = first
    for i in range(length, len(vals)):
        v = vals[i]
        if np.isnan(v):
            out[i] = prev
        else:
            prev = alpha * v + (1.0 - alpha) * prev
            out[i] = prev
    return pd.Series(out, index=series.index)


def atr_pine_like(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rma_pine_like(tr, length)


def stdev_pine_like(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).std(ddof=0)


def sma_pine_like(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def variance_pine_like(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).var(ddof=0)


def pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    """Mimic ta.pivothigh(left, right): value appears on the confirmation bar (i+right)."""
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(left, len(vals) - right):
        center = vals[i]
        if not np.isfinite(center):
            continue
        win = vals[i - left : i + right + 1]
        if not np.isfinite(win).all():
            continue
        if center >= np.max(win):
            # prefer the pivot to be the rightmost occurrence of the max inside the window,
            # so confirmation happens only after "right" bars have passed.
            rightmost_max = np.where(win == np.max(win))[0][-1]
            if rightmost_max == left:
                out[i + right] = center
    return pd.Series(out, index=series.index)


def pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    """Mimic ta.pivotlow(left, right): value appears on the confirmation bar (i+right)."""
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(left, len(vals) - right):
        center = vals[i]
        if not np.isfinite(center):
            continue
        win = vals[i - left : i + right + 1]
        if not np.isfinite(win).all():
            continue
        if center <= np.min(win):
            rightmost_min = np.where(win == np.min(win))[0][-1]
            if rightmost_min == left:
                out[i + right] = center
    return pd.Series(out, index=series.index)


def compute_slope(df: pd.DataFrame, cfg: TLBConfig) -> pd.Series:
    length = cfg.length
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    n = pd.Series(np.arange(len(df), dtype=float), index=df.index)

    if cfg.calc_method == "Atr":
        return atr_pine_like(high, low, close, length) / length * cfg.mult
    if cfg.calc_method == "Stdev":
        return stdev_pine_like(close, length) / length * cfg.mult
    if cfg.calc_method == "Linreg":
        sma_src_n = sma_pine_like(close * n, length)
        sma_src = sma_pine_like(close, length)
        sma_n = sma_pine_like(n, length)
        var_n = variance_pine_like(n, length)
        slope = (sma_src_n - sma_src * sma_n).abs() / var_n / 2.0 * cfg.mult
        return slope
    raise ValueError(f"不支持的 calc_method: {cfg.calc_method}")


def trendlines_with_breaks(df: pd.DataFrame, cfg: TLBConfig) -> pd.DataFrame:
    d = df.copy()
    close = d["close"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    slope = compute_slope(d, cfg)

    ph = pivot_high(high, cfg.length, cfg.length)
    pl = pivot_low(low, cfg.length, cfg.length)

    upper = np.full(len(d), np.nan)
    lower = np.full(len(d), np.nan)
    slope_ph = np.full(len(d), np.nan)
    slope_pl = np.full(len(d), np.nan)
    upos = np.zeros(len(d), dtype=int)
    dnos = np.zeros(len(d), dtype=int)

    cur_upper = np.nan
    cur_lower = np.nan
    cur_slope_ph = np.nan
    cur_slope_pl = np.nan
    cur_upos = 0
    cur_dnos = 0

    for i in range(len(d)):
        if np.isfinite(ph.iloc[i]):
            cur_slope_ph = slope.iloc[i]
            cur_upper = ph.iloc[i]
        else:
            if np.isfinite(cur_upper) and np.isfinite(cur_slope_ph):
                cur_upper = cur_upper - cur_slope_ph

        if np.isfinite(pl.iloc[i]):
            cur_slope_pl = slope.iloc[i]
            cur_lower = pl.iloc[i]
        else:
            if np.isfinite(cur_lower) and np.isfinite(cur_slope_pl):
                cur_lower = cur_lower + cur_slope_pl

        if np.isfinite(ph.iloc[i]):
            cur_upos = 0
        elif np.isfinite(cur_upper) and np.isfinite(cur_slope_ph):
            cur_upos = 1 if close.iloc[i] > (cur_upper - cur_slope_ph * cfg.length) else cur_upos

        if np.isfinite(pl.iloc[i]):
            cur_dnos = 0
        elif np.isfinite(cur_lower) and np.isfinite(cur_slope_pl):
            cur_dnos = 1 if close.iloc[i] < (cur_lower + cur_slope_pl * cfg.length) else cur_dnos

        upper[i] = cur_upper
        lower[i] = cur_lower
        slope_ph[i] = cur_slope_ph
        slope_pl[i] = cur_slope_pl
        upos[i] = cur_upos
        dnos[i] = cur_dnos

    d["ph"] = ph
    d["pl"] = pl
    d["slope"] = slope
    d["upper"] = upper
    d["lower"] = lower
    d["slope_ph"] = slope_ph
    d["slope_pl"] = slope_pl
    upper_line = d["upper"] if cfg.backpaint else d["upper"] - d["slope_ph"] * cfg.length
    lower_line = d["lower"] if cfg.backpaint else d["lower"] + d["slope_pl"] * cfg.length

    # Pine plot() hides the line on pivot-confirmation bars: color = ph ? na : upCss / pl ? na : dnCss
    upper_line = upper_line.mask(d["ph"].notna())
    lower_line = lower_line.mask(d["pl"].notna())

    # mimic Pine offset = -length when backpainting
    if cfg.backpaint:
        upper_line = upper_line.shift(-cfg.length)
        lower_line = lower_line.shift(-cfg.length)

    d["upper_plot"] = upper_line
    d["lower_plot"] = lower_line

    d["upos"] = upos
    d["dnos"] = dnos
    d["upper_break"] = (pd.Series(upos, index=d.index) > pd.Series(upos, index=d.index).shift(1).fillna(0)).astype(int)
    d["lower_break"] = (pd.Series(dnos, index=d.index) > pd.Series(dnos, index=d.index).shift(1).fillna(0)).astype(int)
    return d


def _fmt_x(index: pd.DatetimeIndex, freq: str) -> List[str]:
    f = str(freq).lower()
    if f in ("d", "day", "daily", "101"):
        return [ts.strftime("%Y-%m-%d") for ts in index]
    return [ts.strftime("%Y-%m-%d %H:%M") for ts in index]


def build_extended_segments(df: pd.DataFrame, cfg: TLBConfig, freq: str) -> list[tuple[list[str], list[float], str]]:
    """Draw only the latest active extended upper/lower lines, like Pine's single line objects."""
    if not cfg.show_ext:
        return []
    xs = _fmt_x(df.index, freq)
    segments: list[tuple[list[str], list[float], str]] = []
    n = len(df)

    def _last_valid_pos(s: pd.Series) -> Optional[int]:
        idx = np.where(np.isfinite(s.to_numpy(dtype=float)))[0]
        return int(idx[-1]) if len(idx) else None

    i = _last_valid_pos(df["ph"])
    if i is not None and np.isfinite(df["slope_ph"].iloc[i]):
        x0_i = max(0, i - (cfg.length if cfg.backpaint else 0))
        y0 = float(df["ph"].iloc[i] if cfg.backpaint else (df["upper"].iloc[i] - df["slope_ph"].iloc[i] * cfg.length))
        s = float(df["slope_ph"].iloc[i])
        x_seg = xs[x0_i:]
        y_seg = [y0 - s * j for j in range(len(x_seg))]
        segments.append((x_seg, y_seg, cfg.up_color))

    i = _last_valid_pos(df["pl"])
    if i is not None and np.isfinite(df["slope_pl"].iloc[i]):
        x0_i = max(0, i - (cfg.length if cfg.backpaint else 0))
        y0 = float(df["pl"].iloc[i] if cfg.backpaint else (df["lower"].iloc[i] + df["slope_pl"].iloc[i] * cfg.length))
        s = float(df["slope_pl"].iloc[i])
        x_seg = xs[x0_i:]
        y_seg = [y0 + s * j for j in range(len(x_seg))]
        segments.append((x_seg, y_seg, cfg.down_color))

    return segments


def plot_indicator(df: pd.DataFrame, symbol: str, freq: str, cfg: TLBConfig, out_html: str) -> None:
    x = _fmt_x(df.index, freq)
    fig = go.Figure()

    bg = "#0b0f14"
    grid = "rgba(255,255,255,0.06)"
    font_col = "#c9d1d9"

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["upper_plot"],
            mode="lines",
            name="Upper",
            line=dict(color=cfg.up_color, width=2),
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["lower_plot"],
            mode="lines",
            name="Lower",
            line=dict(color=cfg.down_color, width=2),
            connectgaps=False,
        )
    )

    # extended dashed lines (approximation of TradingView line.extend.right)
    for idx, (xs, ys, col) in enumerate(build_extended_segments(df, cfg, freq)):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="Ext" if idx == 0 else None,
                showlegend=(idx == 0),
                line=dict(color=col, width=1, dash="dash"),
                opacity=0.45,
                hoverinfo="skip",
            )
        )

    up_break = df[df["upper_break"] == 1]
    dn_break = df[df["lower_break"] == 1]

    if not up_break.empty:
        fig.add_trace(
            go.Scatter(
                x=_fmt_x(up_break.index, freq),
                y=up_break["low"],
                mode="markers+text",
                text=["B"] * len(up_break),
                textposition="top center",
                marker=dict(symbol="triangle-up", size=10, color=cfg.up_color),
                textfont=dict(color="white"),
                name="Upper Break",
            )
        )
    if not dn_break.empty:
        fig.add_trace(
            go.Scatter(
                x=_fmt_x(dn_break.index, freq),
                y=dn_break["high"],
                mode="markers+text",
                text=["B"] * len(dn_break),
                textposition="bottom center",
                marker=dict(symbol="triangle-down", size=10, color=cfg.down_color),
                textfont=dict(color="white"),
                name="Lower Break",
            )
        )

    fig.update_layout(
        title=f"Trendlines with Breaks [LuxAlgo] - {symbol}",
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=font_col),
        xaxis=dict(type="category", rangeslider=dict(visible=False), showgrid=True, gridcolor=grid),
        yaxis=dict(title="Price", showgrid=True, gridcolor=grid),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=70, b=40),
    )

    fig.write_html(out_html, include_plotlyjs="cdn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trendlines with Breaks [LuxAlgo] Python复刻")
    parser.add_argument("--symbol", type=str, required=True, help="股票代码，例如 300136")
    parser.add_argument("--freq", type=str, default="d", help="周期: d/60/30/15/5/1")
    parser.add_argument("--bars", type=int, default=240, help="显示最近多少根K线")
    parser.add_argument("--start-date", type=str, default=None, help="开始日期 YYYY-MM-DD，可选")
    parser.add_argument("--end-date", type=str, default=None, help="结束日期 YYYY-MM-DD，可选")
    parser.add_argument("--length", type=int, default=14, help="Swing Detection Lookback")
    parser.add_argument("--mult", type=float, default=1.0, help="Slope multiplier")
    parser.add_argument("--calc-method", type=str, default="Atr", choices=["Atr", "Stdev", "Linreg"], help="Slope Calculation Method")
    parser.add_argument("--backpaint", action="store_true", help="开启 backpaint（更接近 TV 默认显示）")
    parser.add_argument("--show-ext", action="store_true", help="显示延长虚线")
    parser.add_argument("--out", type=str, default="trendlines_with_breaks.html", help="输出HTML文件名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_dt = pd.to_datetime(args.end_date) if args.end_date else pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    if args.start_date:
        start_dt = pd.to_datetime(args.start_date)
    else:
        # fetch enough warmup bars for pivots/rolling stats
        if str(args.freq).lower() in ("d", "day", "daily", "101"):
            start_dt = end_dt - pd.Timedelta(days=max(900, args.bars * 5))
        else:
            start_dt = end_dt - pd.Timedelta(days=max(120, args.bars * 2))

    raw = fetch_kline_pytdx(args.symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), args.freq)
    if raw.empty:
        raise RuntimeError("未获取到行情数据")

    cfg = TLBConfig(
        length=args.length,
        mult=args.mult,
        calc_method=args.calc_method,
        backpaint=bool(args.backpaint),
        show_ext=bool(args.show_ext),
    )
    ind = trendlines_with_breaks(raw, cfg)
    vis = ind.tail(args.bars).copy()
    plot_indicator(vis, args.symbol, args.freq, cfg, args.out)
    print(f"已输出: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
