# -*- coding: utf-8 -*-
"""
Support Resistance Channels (Pine v6 by LonesomeTheBlue) -> Python / Plotly

Features
- Uses the same pytdx-style datasource pattern and categorical-x HTML style as the reference script.
- Preserves the Pine indicator's main logic:
  * pivot high / pivot low confirmation with symmetric left/right period
  * rolling pivot storage with loopback pruning
  * channel width derived from highest/lowest of last 300 bars
  * channel strength = pivot-count strength + touch-count strength
  * strongest non-overlapping S/R channels selection and sorting
  * price-in-channel coloring (resistance / support / in-channel)
  * optional pivot markers, broken support/resistance markers
  * optional MA1 / MA2 overlays

Example
python support_resistance_channels.py \
  --symbol 300136 \
  --freq d \
  --bars 260 \
  --out 300136_srchannel.html \
  --show-ma1 --ma1-len 50 --ma1-type SMA \
  --show-ma2 --ma2-len 200 --ma2-type SMA
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 支持直接运行脚本时导入项目内 datasource
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

try:
    from datasource.pytdx_client import connect_pytdx
except Exception:
    connect_pytdx = None


# --------------------------- datasource ---------------------------
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


# --------------------------- config ---------------------------
@dataclass
class SRChannelConfig:
    prd: int = 10
    ppsrc: str = "High/Low"  # High/Low or Close/Open
    channel_w: int = 5
    minstrength: int = 1
    maxnumsr: int = 6
    loopback: int = 290
    showpp: bool = False
    showsrbroken: bool = False
    show_ma1: bool = False
    ma1_len: int = 50
    ma1_type: str = "SMA"
    show_ma2: bool = False
    ma2_len: int = 200
    ma2_type: str = "SMA"


# --------------------------- helpers ---------------------------
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.astype(float).rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.astype(float).ewm(span=length, adjust=False, min_periods=1).mean()


def calc_ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    return sma(series, length) if ma_type.upper() == "SMA" else ema(series, length)


def _fmt_x_labels(index: pd.Index) -> List[str]:
    idx = pd.to_datetime(index)
    has_time = any((ts.hour != 0 or ts.minute != 0) for ts in idx)
    if has_time:
        return [ts.strftime("%Y-%m-%d %H:%M") for ts in idx]
    return [ts.strftime("%Y-%m-%d") for ts in idx]


def pivothigh(series: pd.Series, left: int, right: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        window = arr[i - left:i + right + 1]
        center = arr[i]
        if np.isnan(center) or np.isnan(window).any():
            continue
        if center == np.max(window) and np.sum(window == center) == 1:
            out[i + right] = center  # Pine outputs on confirmation bar; plot offset -right later.
    return pd.Series(out, index=series.index)


def pivotlow(series: pd.Series, left: int, right: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        window = arr[i - left:i + right + 1]
        center = arr[i]
        if np.isnan(center) or np.isnan(window).any():
            continue
        if center == np.min(window) and np.sum(window == center) == 1:
            out[i + right] = center
    return pd.Series(out, index=series.index)


def _rgba(rgb: Tuple[int, int, int], alpha: float) -> str:
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.3f})"


# --------------------------- core pine port ---------------------------
def compute_sr_channels(df: pd.DataFrame, cfg: SRChannelConfig) -> Tuple[pd.DataFrame, List[dict]]:
    d = df.copy()
    n = len(d)
    close = d["close"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    open_ = d["open"].astype(float)

    if cfg.ppsrc == "High/Low":
        src1 = high.copy()
        src2 = low.copy()
    else:
        src1 = pd.concat([close, open_], axis=1).max(axis=1)
        src2 = pd.concat([close, open_], axis=1).min(axis=1)

    ph = pivothigh(src1, cfg.prd, cfg.prd)
    pl = pivotlow(src2, cfg.prd, cfg.prd)

    d["ph"] = ph
    d["pl"] = pl
    d["ma1"] = calc_ma(close, cfg.ma1_len, cfg.ma1_type) if cfg.show_ma1 else np.nan
    d["ma2"] = calc_ma(close, cfg.ma2_len, cfg.ma2_type) if cfg.show_ma2 else np.nan

    pivotvals: List[float] = []
    pivotlocs: List[int] = []

    # final channels snapshot to display on the chart's last bar
    suportresistance = [0.0] * 20
    resistancebroken = np.full(n, False)
    supportbroken = np.full(n, False)
    channels_by_bar: List[List[dict]] = [[] for _ in range(n)]

    def get_sr_vals(ind: int, cwidth: float) -> Tuple[float, float, int]:
        lo = pivotvals[ind]
        hi = lo
        numpp = 0
        for cpp in pivotvals:
            wdth = (hi - cpp) if cpp <= hi else (cpp - lo)
            if wdth <= cwidth:
                if cpp <= hi:
                    lo = min(lo, cpp)
                else:
                    hi = max(hi, cpp)
                numpp += 20
        return hi, lo, numpp

    def changeit(arr: List[float], x: int, y: int) -> None:
        arr[y * 2], arr[x * 2] = arr[x * 2], arr[y * 2]
        arr[y * 2 + 1], arr[x * 2 + 1] = arr[x * 2 + 1], arr[y * 2 + 1]

    def level_color(hi_: float, lo_: float, c: float) -> str:
        # Pine default colors: res red 75, sup lime 75, inch gray 75
        if hi_ > c and lo_ > c:
            return _rgba((255, 0, 0), 0.25)
        if hi_ < c and lo_ < c:
            return _rgba((0, 255, 0), 0.25)
        return _rgba((128, 128, 128), 0.25)

    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)

    for i in range(n):
        # calculate channel width from highest/lowest(300) up to current bar
        start_300 = max(0, i - 299)
        prdhighest = float(np.nanmax(highs[start_300:i + 1]))
        prdlowest = float(np.nanmin(lows[start_300:i + 1]))
        cwidth = (prdhighest - prdlowest) * cfg.channel_w / 100.0

        new_pivot = False
        if pd.notna(ph.iloc[i]) or pd.notna(pl.iloc[i]):
            new_pivot = True
            pivotvals.insert(0, float(ph.iloc[i]) if pd.notna(ph.iloc[i]) else float(pl.iloc[i]))
            pivotlocs.insert(0, i)
            # prune old pivots; Pine removes from array tail while old
            while pivotlocs and (i - pivotlocs[-1] > cfg.loopback):
                pivotlocs.pop()
                pivotvals.pop()

        if new_pivot and pivotvals:
            supres: List[float] = []  # triplets: strength, hi, lo
            strengths = [0.0] * 10

            for x in range(len(pivotvals)):
                hi_, lo_, strength = get_sr_vals(x, cwidth)
                supres.extend([float(strength), float(hi_), float(lo_)])

            # add each HL touch to strength
            for x in range(len(pivotvals)):
                h = supres[x * 3 + 1]
                l = supres[x * 3 + 2]
                s = 0
                y_max = min(cfg.loopback, i)
                for y in range(y_max + 1):
                    if (highs[i - y] <= h and highs[i - y] >= l) or (lows[i - y] <= h and lows[i - y] >= l):
                        s += 1
                supres[x * 3] = supres[x * 3] + s

            suportresistance = [0.0] * 20
            src = 0
            for _ in range(len(pivotvals)):
                stv = -1.0
                stl = -1
                for y in range(len(pivotvals)):
                    if supres[y * 3] > stv and supres[y * 3] >= cfg.minstrength * 20:
                        stv = supres[y * 3]
                        stl = y
                if stl >= 0:
                    hh = supres[stl * 3 + 1]
                    ll = supres[stl * 3 + 2]
                    suportresistance[src * 2] = hh
                    suportresistance[src * 2 + 1] = ll
                    strengths[src] = supres[stl * 3]

                    for y in range(len(pivotvals)):
                        cond1 = supres[y * 3 + 1] <= hh and supres[y * 3 + 1] >= ll
                        cond2 = supres[y * 3 + 2] <= hh and supres[y * 3 + 2] >= ll
                        if cond1 or cond2:
                            supres[y * 3] = -1
                    src += 1
                    if src >= 10:
                        break

            for x in range(9):
                for y in range(x + 1, 10):
                    if strengths[y] > strengths[x]:
                        strengths[y], strengths[x] = strengths[x], strengths[y]
                        changeit(suportresistance, x, y)

        # per-bar channels snapshot for optional inspection / broken logic
        current_channels: List[dict] = []
        limit = min(9, cfg.maxnumsr - 1)
        for x in range(limit + 1):
            hi_ = suportresistance[x * 2]
            lo_ = suportresistance[x * 2 + 1]
            if hi_ != 0:
                current_channels.append({
                    "rank": x,
                    "hi": hi_,
                    "lo": lo_,
                    "color": level_color(hi_, lo_, closes[i]),
                })
        channels_by_bar[i] = current_channels

        # broken support / resistance logic
        not_in_a_channel = True
        for x in range(limit + 1):
            hi_ = suportresistance[x * 2]
            lo_ = suportresistance[x * 2 + 1]
            if hi_ != 0 and closes[i] <= hi_ and closes[i] >= lo_:
                not_in_a_channel = False
                break

        if i > 0 and not_in_a_channel:
            for x in range(limit + 1):
                hi_ = suportresistance[x * 2]
                lo_ = suportresistance[x * 2 + 1]
                if hi_ == 0:
                    continue
                if closes[i - 1] <= hi_ and closes[i] > hi_:
                    resistancebroken[i] = True
                if closes[i - 1] >= lo_ and closes[i] < lo_:
                    supportbroken[i] = True

    d["resistancebroken"] = resistancebroken
    d["supportbroken"] = supportbroken
    return d, channels_by_bar[-1] if channels_by_bar else []


# --------------------------- plotting ---------------------------
def add_feature_table(fig: go.Figure, df: pd.DataFrame, final_channels: List[dict], cfg: SRChannelConfig) -> None:
    if len(df) == 0:
        return
    last = df.iloc[-1]
    rows = [
        ("time", df.index[-1].strftime("%Y-%m-%d %H:%M") if hasattr(df.index[-1], "strftime") else str(df.index[-1])),
        ("close", f"{float(last['close']):.3f}"),
        ("pivot_period", str(cfg.prd)),
        ("pp_source", cfg.ppsrc),
        ("channel_width_%", str(cfg.channel_w)),
        ("loopback", str(cfg.loopback)),
        ("channels", str(len(final_channels))),
    ]
    for i, ch in enumerate(final_channels[:4], 1):
        rows.append((f"SR{i}", f"{ch['lo']:.3f} ~ {ch['hi']:.3f}"))

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
            domain=dict(x=[0.02, 0.28], y=[0.74, 0.98]),
        )
    )


def build_plot(df: pd.DataFrame, final_channels: List[dict], out_html: str, title: str, cfg: SRChannelConfig) -> None:
    bg = "#0b0f14"
    grid = "rgba(255,255,255,0.06)"
    bull = "#26a69a"
    bear = "#ef5350"

    x = _fmt_x_labels(df.index)
    plot_df = df.copy()
    plot_df["x"] = x

    fig = go.Figure()

    # SR channel rectangles behind candles
    for ch in final_channels[: cfg.maxnumsr]:
        fig.add_hrect(
            y0=ch["lo"],
            y1=ch["hi"],
            line_width=1,
            line_color=ch["color"],
            fillcolor=ch["color"],
            layer="below",
        )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=plot_df["open"], high=plot_df["high"], low=plot_df["low"], close=plot_df["close"],
            increasing_line_color=bull,
            decreasing_line_color=bear,
            increasing_fillcolor=bull,
            decreasing_fillcolor=bear,
            whiskerwidth=0.3,
            name="Price",
            showlegend=False,
        )
    )

    # Volume as bottom overlay using secondary y range feel with low opacity
    vol_colors = np.where(plot_df["close"] >= plot_df["open"], "rgba(38,166,154,0.55)", "rgba(239,83,80,0.55)")
    max_p = float(np.nanmax(plot_df["high"]))
    min_p = float(np.nanmin(plot_df["low"]))
    yr = max_p - min_p
    vol_height = yr * 0.18
    vol_scaled = (plot_df["volume"] / float(np.nanmax(plot_df["volume"]))).fillna(0) * vol_height
    vol_base = min_p - vol_height * 0.05
    fig.add_trace(
        go.Bar(
            x=x,
            y=vol_scaled,
            base=vol_base,
            marker_color=vol_colors,
            marker_line_width=0,
            width=0.78,
            hovertemplate="%{x}<br>Volume: %{customdata:,.0f}<extra></extra>",
            customdata=plot_df["volume"],
            showlegend=False,
            name="Volume",
            opacity=1.0,
        )
    )

    if cfg.show_ma1:
        fig.add_trace(go.Scatter(x=x, y=plot_df["ma1"], mode="lines", line=dict(color="#2962ff", width=2), name=f"MA1 {cfg.ma1_type}({cfg.ma1_len})"))
    if cfg.show_ma2:
        fig.add_trace(go.Scatter(x=x, y=plot_df["ma2"], mode="lines", line=dict(color="#ff1744", width=2), name=f"MA2 {cfg.ma2_type}({cfg.ma2_len})"))

    if cfg.showpp:
        ph_mask = plot_df["ph"].notna()
        pl_mask = plot_df["pl"].notna()
        if ph_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=plot_df.loc[ph_mask, "x"],
                    y=plot_df.loc[ph_mask, "ph"],
                    mode="text",
                    text=["H"] * int(ph_mask.sum()),
                    textposition="top center",
                    textfont=dict(color="#ff5252", size=11),
                    hovertemplate="%{x}<br>Pivot High: %{y:.2f}<extra></extra>",
                    showlegend=False,
                )
            )
        if pl_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=plot_df.loc[pl_mask, "x"],
                    y=plot_df.loc[pl_mask, "pl"],
                    mode="text",
                    text=["L"] * int(pl_mask.sum()),
                    textposition="bottom center",
                    textfont=dict(color="#69f0ae", size=11),
                    hovertemplate="%{x}<br>Pivot Low: %{y:.2f}<extra></extra>",
                    showlegend=False,
                )
            )

    if cfg.showsrbroken:
        rb = plot_df["resistancebroken"].fillna(False)
        sb = plot_df["supportbroken"].fillna(False)
        if rb.any():
            fig.add_trace(
                go.Scatter(
                    x=plot_df.loc[rb, "x"],
                    y=plot_df.loc[rb, "low"] * 0.995,
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=9, color="#69f0ae"),
                    hovertemplate="%{x}<br>Resistance Broken<extra></extra>",
                    showlegend=False,
                )
            )
        if sb.any():
            fig.add_trace(
                go.Scatter(
                    x=plot_df.loc[sb, "x"],
                    y=plot_df.loc[sb, "high"] * 1.005,
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=9, color="#ff5252"),
                    hovertemplate="%{x}<br>Support Broken<extra></extra>",
                    showlegend=False,
                )
            )

    n = len(x)
    tick_step = max(1, n // 12)
    tickvals = x[::tick_step]

    fig.update_layout(
        title=title,
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color="#c9d1d9"),
        height=880,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        bargap=0.08,
        bargroupgap=0.0,
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(0,0,0,0.0)", orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=x,
        showgrid=True,
        gridcolor=grid,
        tickmode="array",
        tickvals=tickvals,
        ticktext=tickvals,
    )
    fig.update_yaxes(showgrid=True, gridcolor=grid)

    add_feature_table(fig, plot_df, final_channels, cfg)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_html}")


# --------------------------- cli ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--symbol", type=str, default="300136", help="A股代码，如 300136")
    ap.add_argument("--freq", type=str, default="d", help="K线周期：d(日线) 或 1/5/15/30/60(分钟)")
    ap.add_argument("--bars", type=int, default=260, help="展示最近 N 根K线")
    ap.add_argument("--out", type=str, default="support_resistance_channels.html", help="输出HTML文件")

    ap.add_argument("--prd", type=int, default=10)
    ap.add_argument("--ppsrc", type=str, default="High/Low", choices=["High/Low", "Close/Open"])
    ap.add_argument("--channel-w", type=int, default=5)
    ap.add_argument("--minstrength", type=int, default=1)
    ap.add_argument("--maxnumsr", type=int, default=6)
    ap.add_argument("--loopback", type=int, default=290)
    ap.add_argument("--showpp", action="store_true", default=False)
    ap.add_argument("--showsrbroken", action="store_true", default=False)
    ap.add_argument("--show-ma1", action="store_true", default=False)
    ap.add_argument("--ma1-len", type=int, default=50)
    ap.add_argument("--ma1-type", type=str, default="SMA", choices=["SMA", "EMA"])
    ap.add_argument("--show-ma2", action="store_true", default=False)
    ap.add_argument("--ma2-len", type=int, default=200)
    ap.add_argument("--ma2-type", type=str, default="SMA", choices=["SMA", "EMA"])

    args = ap.parse_args()

    end = datetime.now().date()
    freq_arg = args.freq.strip().lower()
    if freq_arg in ["d", "day", "daily", "101"]:
        freq = "d"
        start = end - timedelta(days=max(1200, int(args.bars * 5)))
    else:
        try:
            freq = int(freq_arg)
        except ValueError as e:
            raise ValueError("--freq 只能是 d 或 1/5/15/30/60") from e
        start = end - timedelta(days=900)

    df = fetch_kline_pytdx(args.symbol, start=str(start), end=str(end), freq=freq)
    cfg = SRChannelConfig(
        prd=args.prd,
        ppsrc=args.ppsrc,
        channel_w=args.channel_w,
        minstrength=args.minstrength,
        maxnumsr=args.maxnumsr,
        loopback=args.loopback,
        showpp=args.showpp,
        showsrbroken=args.showsrbroken,
        show_ma1=args.show_ma1,
        ma1_len=args.ma1_len,
        ma1_type=args.ma1_type,
        show_ma2=args.show_ma2,
        ma2_len=args.ma2_len,
        ma2_type=args.ma2_type,
    )

    full, final_channels = compute_sr_channels(df, cfg)
    show = full.tail(args.bars).copy()

    # only keep channels relevant to visible price range? keep Pine-like final selection.
    title = (
        f"{args.symbol} ({args.freq}) Support Resistance Channels "
        f"[prd={cfg.prd}, src={cfg.ppsrc}, width={cfg.channel_w}%, loopback={cfg.loopback}, maxsr={cfg.maxnumsr}]"
    )
    build_plot(show, final_channels, out_html=args.out, title=title, cfg=cfg)


if __name__ == "__main__":
    main()
