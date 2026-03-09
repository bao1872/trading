# -*- coding: utf-8 -*-
"""
Liquidity Zones [BigBeluga] -> Python (Plotly)

Ports the Pine v5 indicator logic you pasted:
- Detect pivots using ta.pivothigh/ta.pivotlow (confirmed after rightBars)
- Filter pivots by "Volume Strength Filter" using normalized volume (avg_vol / stdev(avg_vol, 500))
- Build "liquidity zones" as rectangles + horizontal lines
- Extend lines forward until price crosses them, then mark as "Liquidity Grabbed" (dashed line, box text updated, red circle)

Data source:
- Reuses the same pytdx kline fetch approach as your attached script (dynamic_swing_anchored_vwap.py).

Output:
- One self-contained HTML file (Plotly).

Usage examples
    python liquidity_zones_plotly.py --symbol 600547 --freq d --bars 500 --out lz.html
    python liquidity_zones_plotly.py --symbol 600547 --freq 15 --bars 800 --dynamic --flt High

Dependencies
    pip install pandas numpy plotly pytdx
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datasource.pytdx_client import connect_pytdx, PERIOD_MAP

# =========================
# pytdx data fetch (same style as your attached script)
# =========================

def _category_from_freq(freq: str | int) -> int:
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

def fetch_kline_pytdx(symbol: str, start: str, end: str, freq: str | int) -> pd.DataFrame:
    api = connect_pytdx()
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
                raise RuntimeError("pytdx返回数据缺少时间列")

            if "vol" in d.columns and "volume" not in d.columns:
                d = d.rename(columns={"vol": "volume"})
            if "volume" not in d.columns:
                d["volume"] = np.nan

            d = d[["date", "open", "high", "low", "close", "volume"]]
            frames.append(d.sort_values("date"))
            if len(recs) < size:
                break
            page += 1

        if not frames:
            raise RuntimeError("pytdx无数据")

        all_df = pd.concat(frames).sort_values("date")
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        all_df = all_df[(all_df["date"] >= start_dt) & (all_df["date"] <= end_dt)]
        all_df = all_df.drop_duplicates(subset=["date"], keep="last").set_index("date")
        # Ensure numeric
        for c in ["open", "high", "low", "close", "volume"]:
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
        all_df = all_df.dropna(subset=["open", "high", "low", "close"]).copy()
        all_df["volume"] = all_df["volume"].fillna(0.0)
        return all_df
    finally:
        api.disconnect()


# =========================
# Pine-aligned calculations
# =========================

def atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    """TradingView ta.atr(n): Wilder's ATR (RMA of TR)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def stdev(series: pd.Series, n: int) -> pd.Series:
    # Pine ta.stdev uses population stdev? In TradingView it's sample stdev by default (stdev).
    # Pandas rolling std defaults to sample (ddof=1) -> closer to Pine.
    return series.rolling(n, min_periods=n).std()

def pivothigh(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    Pine ta.pivothigh(left, right):
      - returns pivot value at pivot bar (i), but only becomes known at i+right (confirmation).
    We'll return an array ph_confirmed[t] where t is current bar:
      ph_confirmed[t] = high[pivot_i] if pivot confirmed at t, else nan
      pivot_i = t - right
      Condition: high[pivot_i] is the maximum in [pivot_i-left ... pivot_i+right]
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        pivot_i = t - right
        if pivot_i < left or pivot_i < 0:
            continue
        if pivot_i + right >= n:
            continue
        w0 = pivot_i - left
        w1 = pivot_i + right
        window = high[w0 : w1 + 1]
        pv = high[pivot_i]
        if np.isfinite(pv) and pv == np.nanmax(window):
            out[t] = pv
    return out

def pivotlow(low: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(low)
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        pivot_i = t - right
        if pivot_i < left or pivot_i < 0:
            continue
        if pivot_i + right >= n:
            continue
        w0 = pivot_i - left
        w1 = pivot_i + right
        window = low[w0 : w1 + 1]
        pv = low[pivot_i]
        if np.isfinite(pv) and pv == np.nanmin(window):
            out[t] = pv
    return out


# =========================
# Liquidity Zones logic
# =========================

@dataclass
class LZConfig:
    leftBars: int = 10
    qty_pivots: int = 10
    flt: str = "Mid"          # Low/Mid/High
    dynamic: bool = False
    hidePivot: bool = True

    upper_col: str = "#2370a3"
    lower_col: str = "#23a372"

    # Pine constants
    box_width_bars: int = 8
    max_extend_bars: int = 1500
    atr_len: int = 200
    vol_sma_len_from_right: bool = True  # keep pine semantics


def _filter_value(flt: str) -> int:
    flt = (flt or "Mid").strip().title()
    return 1 if flt == "Low" else 2 if flt == "Mid" else 3 if flt == "High" else 0

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def _gradient_color(intense_0_100: float, col_hex: str, alpha_lo: float, alpha_hi: float) -> str:
    """
    Approximate Pine color.from_gradient(intense, 0,100, color.new(col, a1), color.new(col, a2))
    We'll keep RGB constant and interpolate alpha between (lo->hi).
    alpha expressed as 0..1 in Plotly rgba.
    """
    intense = _clamp(float(intense_0_100), 0.0, 100.0) / 100.0
    a = alpha_lo + (alpha_hi - alpha_lo) * intense

    col_hex = col_hex.lstrip("#")
    r = int(col_hex[0:2], 16)
    g = int(col_hex[2:4], 16)
    b = int(col_hex[4:6], 16)
    return f"rgba({r},{g},{b},{a:.3f})"


def build_liquidity_zones(df: pd.DataFrame, cfg: LZConfig) -> Dict[str, Any]:
    """
    Returns dict containing:
      zones: list of dicts with box + line + states
      pivot_points: list of dicts for plotting filtered pivots
    """
    d = df.copy()
    n = len(d)
    if n == 0:
        raise ValueError("空数据")

    leftBars = int(cfg.leftBars)
    rightBars = leftBars - 2
    if rightBars < 1:
        raise ValueError("leftBars太小，导致 rightBars<1；请调大 leftBars")

    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    vol = d["volume"].to_numpy(float)

    # Pine: avg_vol = sma(volume, rightBars)
    avg_vol = sma(pd.Series(vol, index=d.index), rightBars)
    # Pine: normalized_vol = avg_vol / stdev(avg_vol, 500)
    denom = stdev(avg_vol, 500)
    normalized_vol = (avg_vol / denom).replace([np.inf, -np.inf], np.nan)

    # Pine: color_intense = round(normalized_vol)*15, clipped at 100
    color_intense = (normalized_vol.round() * 15.0).clip(lower=0.0, upper=100.0)

    # Pine: aTR = ta.atr(200)
    aTR = atr_wilder(d, cfg.atr_len)

    # Pine pivots (confirmed at t)
    ph_conf = pivothigh(high, leftBars, rightBars)  # array aligned with t (confirmation bar)
    pl_conf = pivotlow(low, leftBars, rightBars)

    flt_val = _filter_value(cfg.flt)

    zones: List[Dict[str, Any]] = []
    pivot_pts: List[Dict[str, Any]] = []

    # active lines: index in zones list; we will update as we go
    for t in range(n):
        # confirmation uses normalized_vol[rightBars] at time t -> that's value at pivot bar (t-rightBars)
        pivot_i = t - rightBars
        if pivot_i >= 0:
            nv_pivot = float(normalized_vol.iloc[pivot_i]) if pd.notna(normalized_vol.iloc[pivot_i]) else np.nan
            ci_pivot = float(color_intense.iloc[pivot_i]) if pd.notna(color_intense.iloc[pivot_i]) else 0.0
            atr_pivot = float(aTR.iloc[pivot_i]) if pd.notna(aTR.iloc[pivot_i]) else np.nan
            av_pivot = float(avg_vol.iloc[pivot_i]) if pd.notna(avg_vol.iloc[pivot_i]) else np.nan
        else:
            nv_pivot = np.nan
            ci_pivot = 0.0
            atr_pivot = np.nan
            av_pivot = np.nan

        # Create new zone on pivot confirmation bar (t) using pivot bar index (pivot_i)
        if pivot_i >= 0 and np.isfinite(nv_pivot) and (nv_pivot > flt_val):
            # pivot high
            if np.isfinite(ph_conf[t]):
                distance = (nv_pivot * atr_pivot) / 2.0 if cfg.dynamic else atr_pivot
                if not np.isfinite(distance):
                    distance = 0.0

                # Pine gradient: upper uses color.new(upper_col,60) -> alpha ~0.40; to alpha 1.0
                col = _gradient_color(ci_pivot, cfg.upper_col, alpha_lo=0.40, alpha_hi=1.00)

                zone = dict(
                    kind="upper",
                    pivot_i=pivot_i,
                    pivot_time=d.index[pivot_i],
                    base=float(high[pivot_i]),
                    y=float(high[pivot_i] + distance),
                    x1=pivot_i,
                    x2=t,  # extend to current, will be updated
                    grabbed=False,
                    grabbed_i=None,
                    line_style="solid",
                    line_width=2,
                    color=col,
                    text=f"Volume:\n{(round(av_pivot, 2) if np.isfinite(av_pivot) else 'n/a')}",
                )
                zones.append(zone)

                if cfg.hidePivot:
                    pivot_pts.append(dict(i=pivot_i, time=d.index[pivot_i], y=float(high[pivot_i]), kind="ph"))

            # pivot low
            if np.isfinite(pl_conf[t]):
                distance = (nv_pivot * atr_pivot) / 2.0 if cfg.dynamic else atr_pivot
                if not np.isfinite(distance):
                    distance = 0.0

                # lower uses color.new(lower,80) -> alpha ~0.20; to alpha 1.0
                col = _gradient_color(ci_pivot, cfg.lower_col, alpha_lo=0.20, alpha_hi=1.00)

                zone = dict(
                    kind="lower",
                    pivot_i=pivot_i,
                    pivot_time=d.index[pivot_i],
                    base=float(low[pivot_i]),
                    y=float(low[pivot_i] - distance),
                    x1=pivot_i,
                    x2=t,
                    grabbed=False,
                    grabbed_i=None,
                    line_style="solid",
                    line_width=2,
                    color=col,
                    text=f"Volume:\n{(round(av_pivot, 2) if np.isfinite(av_pivot) else 'n/a')}",
                )
                zones.append(zone)

                if cfg.hidePivot:
                    pivot_pts.append(dict(i=pivot_i, time=d.index[pivot_i], y=float(low[pivot_i]), kind="pl"))

            # Keep only the most recent qty_pivots zones (Pine deletes oldest boxes/lines globally)
            if len(zones) > cfg.qty_pivots:
                # drop oldest zones; also drop pivot_pts that reference dropped zones is ok (pivots are just markers)
                zones = zones[-cfg.qty_pivots:]

        # Update existing zones: extend or mark grabbed
        # Only update zones that are still active and within max_extend_bars from last bar (Pine condition)
        # Pine: if lineArray.size()>0 and last_bar_index-bar_index < 1500
        for z in zones:
            if z["grabbed"]:
                continue

            # respect max extend window relative to last bar
            if (n - 1 - t) >= cfg.max_extend_bars:
                continue

            yv = z["y"]
            if not np.isfinite(yv):
                continue

            # If price crosses the line at bar t
            if (high[t] > yv) and (low[t] < yv):
                z["grabbed"] = True
                z["grabbed_i"] = t
                z["x2"] = t
                z["line_style"] = "dash"
                z["line_width"] = 1
                z["text"] = "Liquidity\nGrabbed"
            else:
                # extend line forward by 1 bar (we'll just set x2=t for final draw)
                z["x2"] = t

    return dict(zones=zones, pivot_points=pivot_pts, rightBars=rightBars)


# =========================
# Plot
# =========================

def _add_dashboard(fig: go.Figure, cfg: LZConfig, zones: List[Dict], df: pd.DataFrame):
    mode_txt = "Dynamic Mode" if cfg.dynamic else "Simple Mode"
    qty = int(cfg.qty_pivots)
    
    last_close = float(df["close"].iloc[-1])
    
    upper_zones = [z for z in zones if z["kind"] == "upper" and not z["grabbed"]]
    lower_zones = [z for z in zones if z["kind"] == "lower" and not z["grabbed"]]
    
    upper_prices = sorted([z["y"] for z in upper_zones if z["y"] > last_close])
    lower_prices = sorted([z["y"] for z in lower_zones if z["y"] < last_close], reverse=True)
    
    nearest_upper = upper_prices[0] if upper_prices else None
    nearest_lower = lower_prices[0] if lower_prices else None
    
    upper_dist = f"{((nearest_upper - last_close) / last_close * 100):.2f}%" if nearest_upper else "N/A"
    lower_dist = f"{((last_close - nearest_lower) / last_close * 100):.2f}%" if nearest_lower else "N/A"
    
    upper_price_str = f"{nearest_upper:.2f}" if nearest_upper else "N/A"
    lower_price_str = f"{nearest_lower:.2f}" if nearest_lower else "N/A"
    
    if nearest_upper and nearest_lower:
        upper_dist_val = (nearest_upper - last_close) / last_close * 100
        lower_dist_val = (last_close - nearest_lower) / last_close * 100
        risk_reward = upper_dist_val / lower_dist_val if lower_dist_val > 0 else None
        risk_reward_str = f"{risk_reward:.2f}" if risk_reward else "N/A"
    else:
        risk_reward_str = "N/A"
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Liquidity Zones", ""],
                fill_color="rgba(0,0,0,0.65)",
                line_color="rgba(255,255,255,0.25)",
                font=dict(color="white", size=12, family="Consolas, monospace"),
                align=["left", "left"],
                height=22,
            ),
            cells=dict(
                values=[
                    ["Qty", "Mode", "Close", "卖方流动性", "距卖方", "买方流动性", "距买方", "盈亏比"],
                    [str(qty), mode_txt, f"{last_close:.2f}", upper_price_str, upper_dist, lower_price_str, lower_dist, risk_reward_str],
                ],
                fill_color=[["rgba(0,0,0,0.45)"] * 8, ["rgba(0,0,0,0.35)"] * 8],
                line_color="rgba(255,255,255,0.12)",
                font=dict(color=["white", "#2370a3", "white", "#2370a3", "#2370a3", "#23a372", "#23a372", "#f0b90b"], size=11),
                align=["left", "left"],
                height=20,
            ),
            columnwidth=[0.45, 0.55],
            domain=dict(x=[0.02, 0.28], y=[0.80, 0.98]),
        )
    )

def build_plot(df: pd.DataFrame, zones_payload: Dict[str, Any], cfg: LZConfig, out_html: str, title: str):
    d = df.copy()
    zones = zones_payload["zones"]
    pivot_pts = zones_payload["pivot_points"]
    offset = int(zones_payload.get("offset", 0))  # global->local index shift when plotting tail window
    rightBars = zones_payload["rightBars"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=[0.78, 0.22],
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
            showlegend=False,
        ),
        row=1, col=1
    )

    # Zones: add boxes (as shapes) + line traces + grabbed markers
    shapes = []
    annotations = []

    grabbed_x = []
    grabbed_y = []

    for z in zones:
        pivot_i = int(z["pivot_i"]) - offset
        if pivot_i < 0 or pivot_i >= len(d):
            continue
        # box x-range: pivot_i .. pivot_i+box_width_bars
        x_left = d.index[pivot_i] if pivot_i < len(d) else d.index[0]
        x_right_i = min(pivot_i + cfg.box_width_bars, len(d)-1)
        x_right = d.index[x_right_i]

        if z["kind"] == "upper":
            y0 = z["base"]
            y1 = z["y"]
        else:
            y0 = z["y"]
            y1 = z["base"]

        # Rectangle
        shapes.append(dict(
            type="rect",
            xref="x", yref="y",
            x0=x_left, x1=x_right,
            y0=y0, y1=y1,
            line=dict(color=z["color"], width=2 if not z["grabbed"] else 1),
            fillcolor=z["color"] if not z["grabbed"] else "rgba(0,0,0,0)",
            layer="below",
        ))

        # Box text annotation near center of box
        x_mid = d.index[min(pivot_i + cfg.box_width_bars//2, len(d)-1)]
        y_mid = (y0 + y1) / 2.0
        annotations.append(dict(
            x=x_mid, y=y_mid, xref="x", yref="y",
            text=z["text"].replace("\n", "<br>"),
            showarrow=False,
            font=dict(color="#c9d1d9", size=10),
            bgcolor="rgba(0,0,0,0.0)",
            align="left",
        ))

        # Horizontal line from pivot_i to x2
        x2_i = int(z["x2"]) - offset
        x2_i = max(pivot_i, min(x2_i, len(d)-1))
        xs = [d.index[pivot_i], d.index[x2_i]]
        ys = [z["y"], z["y"]]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=z["color"], width=int(z["line_width"]), dash=z["line_style"]),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=1
        )

        if z["grabbed"] and (z["grabbed_i"] is not None):
            gi = int(z["grabbed_i"]) - offset
            if 0 <= gi < len(d):
                grabbed_x.append(d.index[gi])
                grabbed_y.append(z["y"])

    if len(grabbed_x) > 0:
        fig.add_trace(
            go.Scatter(
                x=grabbed_x, y=grabbed_y,
                mode="text",
                text=["〇"] * len(grabbed_x),
                textfont=dict(color="#df1c1c", size=16),
                hovertemplate="Claim Liquidity Point<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1
        )

    # Filtered pivots (plotshape circles)
    if cfg.hidePivot and len(pivot_pts) > 0:
        ph_x = [p["time"] for p in pivot_pts if p["kind"] == "ph"]
        ph_y = [p["y"] for p in pivot_pts if p["kind"] == "ph"]
        pl_x = [p["time"] for p in pivot_pts if p["kind"] == "pl"]
        pl_y = [p["y"] for p in pivot_pts if p["kind"] == "pl"]

        # Big faint circle
        if ph_x:
            fig.add_trace(go.Scatter(
                x=ph_x, y=ph_y, mode="markers",
                marker=dict(size=10, color="rgba(35,112,163,0.4)"),
                hoverinfo="skip", showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=ph_x, y=ph_y, mode="markers",
                marker=dict(size=4, color="rgba(35,112,163,1.0)"),
                hoverinfo="skip", showlegend=False
            ), row=1, col=1)
        if pl_x:
            fig.add_trace(go.Scatter(
                x=pl_x, y=pl_y, mode="markers",
                marker=dict(size=10, color="rgba(35,163,114,0.4)"),
                hoverinfo="skip", showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=pl_x, y=pl_y, mode="markers",
                marker=dict(size=4, color="rgba(35,163,114,1.0)"),
                hoverinfo="skip", showlegend=False
            ), row=1, col=1)

    # Volume bars
    vol_colors = np.where(d["close"] >= d["open"], "rgba(38,166,154,0.6)", "rgba(239,83,80,0.6)")
    fig.add_trace(go.Bar(x=d.index, y=d["volume"], marker_color=vol_colors, showlegend=False), row=2, col=1)

    # Layout styling (TradingView-like dark)
    fig.update_layout(
        title=title,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=950,
        margin=dict(l=40, r=40, t=60, b=40),
        shapes=shapes,
        annotations=annotations,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=2, col=1, rangemode="tozero")

    # Dashboard
    _add_dashboard(fig, cfg, zones, d)

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_html}")


# =========================
# CLI
# =========================

def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)

    ap.add_argument("--symbol", type=str, default="600547", help="A股代码，如 600547(山东黄金)")
    ap.add_argument("--freq", type=str, default="d", help="K线周期：d(日线) 或 1/5/15/30/60(分钟)")
    ap.add_argument("--bars", type=int, default=500, help="展示最近N根K线")
    ap.add_argument("--out", type=str, default="liquidity_zones.html", help="输出HTML文件")

    ap.add_argument("--leftBars", type=int, default=10, help="Length (Pine input)")
    ap.add_argument("--qty_pivots", type=int, default=10, help="Zones Amount")
    ap.add_argument("--flt", type=str, default="Mid", choices=["Low", "Mid", "High"], help="Volume Strength Filter")
    ap.add_argument("--dynamic", action="store_true", help="Dynamic Distance")
    ap.add_argument("--hidePivot", action="store_true", help="Show filtered pivots (same meaning as Pine hidePivot=true)")

    args = ap.parse_args()

    # Date range: enough history for ATR(200) and stdev(500)
    end = datetime.now().date()
    freq_arg = args.freq.strip().lower()
    if freq_arg in ["d", "day", "daily", "101"]:
        freq = "d"
        # Need >= 700 bars for stdev warmup; add buffer
        start = end - timedelta(days=1600)
    else:
        # intraday, fetch more days to cover 500-bar rolling stdev & ATR
        try:
            _ = int(freq_arg)
        except ValueError as e:
            raise ValueError("--freq 只能是 d 或 1/5/15/30/60（分钟，数字）") from e
        freq = freq_arg
        start = end - timedelta(days=1200)

    df = fetch_kline_pytdx(args.symbol, start=str(start), end=str(end), freq=freq)
    if len(df) == 0:
        raise RuntimeError("未获取到数据")

    # Keep full df for indicator state, but only plot last N bars
    cfg = LZConfig(
        leftBars=int(args.leftBars),
        qty_pivots=int(args.qty_pivots),
        flt=str(args.flt),
        dynamic=bool(args.dynamic),
        hidePivot=bool(args.hidePivot),
    )

    payload_full = build_liquidity_zones(df, cfg)
    df_show = df.tail(int(args.bars)).copy()

    # Rebuild payload on df_show only? Pine logic uses full history for rolling stats; but TV also only has chart history.
    # We'll compute using full df but then keep zones that intersect display window.
    idx0 = df_show.index[0]
    idx1 = df_show.index[-1]

    zones_kept = []
    for z in payload_full["zones"]:
        # keep if pivot time within display window or line extends into it
        pt = z["pivot_time"]
        if (pt >= idx0) and (pt <= idx1):
            zones_kept.append(z)
        else:
            # if line exists within window (rare if pivot earlier and still active)
            x2_i = int(z["x2"]) - offset
            if x2_i >= (len(df) - len(df_show)):
                zones_kept.append(z)

    pivot_kept = [p for p in payload_full["pivot_points"] if (p["time"] >= idx0 and p["time"] <= idx1)]
    payload_show = dict(zones=zones_kept, pivot_points=pivot_kept, rightBars=payload_full["rightBars"], offset=(len(df) - len(df_show)))

    title = f"{args.symbol} Liquidity Zones (leftBars={cfg.leftBars}, flt={cfg.flt}, dynamic={cfg.dynamic})"
    build_plot(df_show, payload_show, cfg, out_html=args.out, title=title)


if __name__ == "__main__":
    main()
