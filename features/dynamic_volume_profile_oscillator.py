# -*- coding: utf-8 -*-
"""
Dynamic Volume Profile Oscillator | AlphaAlgos -> Python (Plotly)

Updates in this version
- X axis uses string/categorical labels so non-trading time gaps are removed.
- Plotting is made closer to the Pine script:
  * exact 10-layer upper/lower gradient bands
  * dynamic main/fast/slow line colors driven by is_bullish
  * exact upper/lower gradient transparency ladder from the Pine script
  * categorical x-axis labels like TradingView session bars (no non-trading gaps)
- Core Pine logic is preserved, including segmented profile refresh.

python dynamic_volume_profile_oscillator.py \
      --symbol 300136 \
      --freq d \
      --bars 220 \
      --out 300136_dvpo.html

      
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 支持直接运行脚本时导入项目内 datasource
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
                d["date"] = pd.to_datetime(d["datetime"])
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
class DVPOConfig:
    lookback: int = 50
    profile_periods: int = 10
    smoothing: int = 5
    sensitivity: float = 1.0
    mean_reversion: bool = True
    use_adaptive_midline: bool = True
    midline_period: int = 50
    zone_width: float = 1.5
    color_bars: bool = True


def ema_pine_like(series: pd.Series, length: int) -> pd.Series:
    if length <= 1:
        return series.astype(float)
    return series.astype(float).ewm(span=length, adjust=False, min_periods=1).mean()


def sma_pine_like(series: pd.Series, length: int) -> pd.Series:
    return series.astype(float).rolling(length, min_periods=length).mean()


def stdev_pine_like(series: pd.Series, length: int) -> pd.Series:
    return series.astype(float).rolling(length, min_periods=length).std(ddof=0)


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    range1 = max_val - min_val
    if not np.isfinite(range1) or range1 <= 0:
        return 50.0
    out = ((value - min_val) / range1) * 100.0
    return float(np.clip(out, 0.0, 100.0))


def rolling_normalized_volume(volume: pd.Series, smoothing: int, lookback: int) -> pd.Series:
    v_sma = sma_pine_like(volume, smoothing)
    out = np.full(len(volume), np.nan, dtype=float)
    vals = v_sma.to_numpy(dtype=float)
    for i in range(len(volume)):
        start = max(0, i - lookback + 1)
        window = vals[start:i + 1]
        finite = window[np.isfinite(window)]
        if finite.size == 0 or not np.isfinite(vals[i]):
            continue
        out[i] = normalize_value(vals[i], float(np.min(finite)), float(np.max(finite)))
    return pd.Series(out, index=volume.index, name="oscillator_raw")


def calculate_volume_profile_metrics(price: pd.Series, volume: pd.Series, lookback: int, profile_periods: int) -> Tuple[pd.Series, pd.Series]:
    n = len(price)
    px = price.to_numpy(dtype=float)
    vol = volume.to_numpy(dtype=float)

    vwap_level = np.full(n, np.nan, dtype=float)
    price_deviation = np.full(n, np.nan, dtype=float)

    cached_prices: List[float] = []
    cached_profile: List[float] = []

    for i in range(n):
        if len(cached_profile) == 0 or i % profile_periods == 0:
            cached_prices = []
            cached_profile = []
            start = max(0, i - lookback + 1)
            for j in range(i, start - 1, -1):
                cached_prices.append(px[j])
                cached_profile.append(vol[j])

        prof = np.asarray(cached_profile, dtype=float)
        pri = np.asarray(cached_prices, dtype=float)
        finite_mask = np.isfinite(prof) & np.isfinite(pri)
        prof = prof[finite_mask]
        pri = pri[finite_mask]

        sum_vol = float(np.sum(prof)) if prof.size else 0.0
        if sum_vol > 0:
            vwap = float(np.sum(pri * prof) / sum_vol)
            vol_weight = prof / sum_vol
            sum_dev = float(np.sum(np.abs(pri - vwap) * vol_weight))
            vwap_level[i] = vwap
            price_deviation[i] = sum_dev
        else:
            vwap_level[i] = px[i]
            price_deviation[i] = 0.0

    return (
        pd.Series(vwap_level, index=price.index, name="vwap_level"),
        pd.Series(price_deviation, index=price.index, name="price_deviation"),
    )


def dynamic_volume_profile_oscillator(df: pd.DataFrame, cfg: DVPOConfig) -> pd.DataFrame:
    d = df.copy()
    price = d["close"].astype(float)
    volume = d["volume"].astype(float)

    vwap_level, price_deviation = calculate_volume_profile_metrics(
        price=price,
        volume=volume,
        lookback=cfg.lookback,
        profile_periods=cfg.profile_periods,
    )

    if cfg.mean_reversion:
        denom = price_deviation * float(cfg.sensitivity)
        denom_safe = denom.replace(0.0, np.nan)
        oscillator_raw = 50.0 + ((price - vwap_level) / denom_safe) * 25.0
        oscillator_raw = oscillator_raw.replace([np.inf, -np.inf], np.nan)
    else:
        oscillator_raw = rolling_normalized_volume(volume, cfg.smoothing, cfg.lookback)

    oscillator = ema_pine_like(oscillator_raw, cfg.smoothing)

    if cfg.use_adaptive_midline:
        adaptive_midline = sma_pine_like(oscillator, cfg.midline_period)
    else:
        adaptive_midline = pd.Series(50.0, index=d.index, name="adaptive_midline")

    oscillator_stdev = stdev_pine_like(oscillator, cfg.midline_period) * float(cfg.zone_width)
    upper_zone = adaptive_midline + oscillator_stdev
    lower_zone = adaptive_midline - oscillator_stdev

    fast_signal = ema_pine_like(oscillator, 5)
    slow_signal = ema_pine_like(oscillator, 15)

    is_bullish = (oscillator > adaptive_midline) | (fast_signal > slow_signal)
    is_bearish = ~is_bullish

    crossover_up = (fast_signal > slow_signal) & (fast_signal.shift(1) <= slow_signal.shift(1))
    crossover_down = (fast_signal < slow_signal) & (fast_signal.shift(1) >= slow_signal.shift(1))

    out = d.copy()
    out["vwap_level"] = vwap_level
    out["price_deviation"] = price_deviation
    out["oscillator_raw"] = oscillator_raw
    out["oscillator"] = oscillator
    out["adaptive_midline"] = adaptive_midline
    out["oscillator_stdev"] = oscillator_stdev
    out["upper_zone"] = upper_zone
    out["lower_zone"] = lower_zone
    out["fast_signal"] = fast_signal
    out["slow_signal"] = slow_signal
    out["is_bullish"] = is_bullish.astype(bool)
    out["is_bearish"] = is_bearish.astype(bool)
    out["crossover_up"] = crossover_up.fillna(False)
    out["crossover_down"] = crossover_down.fillna(False)
    return out


def add_feature_table(fig: go.Figure, df: pd.DataFrame) -> None:
    if len(df) == 0:
        return
    last_ts = df.index[-1]
    rows = [
        ("time", last_ts.strftime("%Y-%m-%d %H:%M") if hasattr(last_ts, "strftime") else str(last_ts)),
        ("close", f"{float(df['close'].iloc[-1]):.3f}"),
        ("oscillator", f"{float(df['oscillator'].iloc[-1]):.3f}" if pd.notna(df['oscillator'].iloc[-1]) else "n/a"),
        ("fast_signal", f"{float(df['fast_signal'].iloc[-1]):.3f}" if pd.notna(df['fast_signal'].iloc[-1]) else "n/a"),
        ("slow_signal", f"{float(df['slow_signal'].iloc[-1]):.3f}" if pd.notna(df['slow_signal'].iloc[-1]) else "n/a"),
        ("midline", f"{float(df['adaptive_midline'].iloc[-1]):.3f}" if pd.notna(df['adaptive_midline'].iloc[-1]) else "n/a"),
        ("upper_zone", f"{float(df['upper_zone'].iloc[-1]):.3f}" if pd.notna(df['upper_zone'].iloc[-1]) else "n/a"),
        ("lower_zone", f"{float(df['lower_zone'].iloc[-1]):.3f}" if pd.notna(df['lower_zone'].iloc[-1]) else "n/a"),
        ("trend", "BULL" if bool(df['is_bullish'].iloc[-1]) else "BEAR"),
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
            domain=dict(x=[0.02, 0.28], y=[0.73, 0.98]),
        )
    )


def _fmt_x_labels(index: pd.Index) -> List[str]:
    idx = pd.to_datetime(index)
    has_time = any((ts.hour != 0 or ts.minute != 0) for ts in idx)
    if has_time:
        return [ts.strftime("%Y-%m-%d %H:%M") for ts in idx]
    return [ts.strftime("%Y-%m-%d") for ts in idx]


def _mask_series(series: pd.Series, mask: pd.Series) -> pd.Series:
    s = series.astype(float).copy()
    return s.where(mask, np.nan)


def build_gradient_boundaries(plot_df: pd.DataFrame):
    upper = plot_df["upper_zone"]
    mid = plot_df["adaptive_midline"]
    lower = plot_df["lower_zone"]

    upper_step = (upper - mid) / 10.0
    lower_step = (mid - lower) / 10.0

    upper_bounds = [upper]
    for i in range(1, 10):
        upper_bounds.append(upper - upper_step * i)
    upper_bounds.append(mid)

    lower_bounds = [lower]
    for i in range(1, 10):
        lower_bounds.append(lower + lower_step * i)
    lower_bounds.append(mid)
    return upper_bounds, lower_bounds


def build_plot(df: pd.DataFrame, out_html: str, title: str, color_bars: bool = True) -> None:
    bull = "#00f1ff"
    bear = "#ff019a"
    mid_gray = "rgba(128,128,128,1.0)"
    grid = "rgba(255,255,255,0.06)"
    bg = "#0b0f14"

    plot_df = df.copy()
    x = _fmt_x_labels(plot_df.index)
    plot_df["x"] = x

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.50, 0.16, 0.34],
    )

    # Main candles: use categorical x to remove non-trading gaps.
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=plot_df["open"], high=plot_df["high"], low=plot_df["low"], close=plot_df["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            whiskerwidth=0.3,
            name="Price",
            showlegend=False,
        ),
        row=1, col=1
    )

    # Pine barcolor approximation: overlay per-bar tinted body rectangles.
    if color_bars:
        state_colors = np.where(plot_df["is_bullish"], "rgba(0,241,255,0.20)", "rgba(255,1,154,0.20)")
        fig.add_trace(
            go.Bar(
                x=x,
                y=(plot_df["close"] - plot_df["open"]).abs(),
                base=np.minimum(plot_df["open"], plot_df["close"]),
                marker_color=state_colors,
                marker_line_width=0,
                opacity=1.0,
                width=0.78,
                hoverinfo="skip",
                showlegend=False,
                name="barcolor",
            ),
            row=1, col=1
        )

    vol_colors = np.where(plot_df["close"] >= plot_df["open"], "rgba(38,166,154,0.65)", "rgba(239,83,80,0.65)")
    fig.add_trace(
        go.Bar(x=x, y=plot_df["volume"], marker_color=vol_colors, marker_line_width=0, showlegend=False, name="Volume"),
        row=2, col=1
    )

    # --- Oscillator panel ---
    # Exact Pine-like gradient transparency ladder.
    upper_fill_alphas = [0.30, 0.25, 0.20, 0.17, 0.14, 0.11, 0.08, 0.05, 0.02, 0.01]
    lower_fill_alphas = [0.30, 0.25, 0.20, 0.17, 0.14, 0.11, 0.08, 0.05, 0.02, 0.01]
    upper_line_alphas = [1.00, 0.30, 0.25, 0.20, 0.17, 0.14, 0.11, 0.08, 0.05, 0.02]
    lower_line_alphas = [1.00, 0.30, 0.25, 0.20, 0.17, 0.14, 0.11, 0.08, 0.05, 0.02]

    upper_bounds, lower_bounds = build_gradient_boundaries(plot_df)

    # Upper zone lines + fills.
    for i in range(10):
        y_top = upper_bounds[i]
        y_bot = upper_bounds[i + 1]
        fig.add_trace(
            go.Scatter(
                x=x, y=y_top, mode="lines",
                line=dict(color=f"rgba(255,1,154,{upper_line_alphas[i]:.3f})", width=1),
                hoverinfo="skip", showlegend=False,
                name=f"UG{i+1}",
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=y_bot, mode="lines",
                line=dict(color=f"rgba(255,1,154,{upper_line_alphas[min(i+1, 9)]:.3f})", width=1),
                fill="tonexty",
                fillcolor=f"rgba(255,1,154,{upper_fill_alphas[i]:.3f})",
                hoverinfo="skip", showlegend=False,
                name=f"UGF{i+1}",
            ),
            row=3, col=1,
        )

    # Lower zone lines + fills.
    for i in range(10):
        y_bot = lower_bounds[i]
        y_top = lower_bounds[i + 1]
        fig.add_trace(
            go.Scatter(
                x=x, y=y_bot, mode="lines",
                line=dict(color=f"rgba(0,241,255,{lower_line_alphas[i]:.3f})", width=1),
                hoverinfo="skip", showlegend=False,
                name=f"LG{i+1}",
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=y_top, mode="lines",
                line=dict(color=f"rgba(0,241,255,{lower_line_alphas[min(i+1, 9)]:.3f})", width=1),
                fill="tonexty",
                fillcolor=f"rgba(0,241,255,{lower_fill_alphas[i]:.3f})",
                hoverinfo="skip", showlegend=False,
                name=f"LGF{i+1}",
            ),
            row=3, col=1,
        )

    # Main oscillator: segment by bullish/bearish state to mimic Pine dynamic color.
    bull_mask = plot_df["is_bullish"].fillna(False)
    bear_mask = ~bull_mask
    osc_bull = _mask_series(plot_df["oscillator"], bull_mask)
    osc_bear = _mask_series(plot_df["oscillator"], bear_mask)
    fast_bull = _mask_series(plot_df["fast_signal"], bull_mask)
    fast_bear = _mask_series(plot_df["fast_signal"], bear_mask)
    slow_bull = _mask_series(plot_df["slow_signal"], bull_mask)
    slow_bear = _mask_series(plot_df["slow_signal"], bear_mask)

    fig.add_trace(
        go.Scatter(x=x, y=plot_df["adaptive_midline"], mode="lines", line=dict(color=mid_gray, width=1), showlegend=False, name="Adaptive Midline"),
        row=3, col=1,
    )

    fig.add_trace(go.Scatter(x=x, y=osc_bull, mode="lines", line=dict(color=bull, width=2), showlegend=False, name="Volume Profile Oscillator"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=osc_bear, mode="lines", line=dict(color=bear, width=2), showlegend=False, name="Volume Profile Oscillator"), row=3, col=1)

    fig.add_trace(go.Scatter(x=x, y=fast_bull, mode="lines", line=dict(color="rgba(0,241,255,0.60)", width=1), showlegend=False, name="Fast Signal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=fast_bear, mode="lines", line=dict(color="rgba(255,1,154,0.60)", width=1), showlegend=False, name="Fast Signal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=slow_bull, mode="lines", line=dict(color="rgba(0,241,255,0.30)", width=1), showlegend=False, name="Slow Signal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=slow_bear, mode="lines", line=dict(color="rgba(255,1,154,0.30)", width=1), showlegend=False, name="Slow Signal"), row=3, col=1)

    # Crossovers.
    up_mask = plot_df["crossover_up"].fillna(False)
    dn_mask = plot_df["crossover_down"].fillna(False)
    if up_mask.any():
        fig.add_trace(
            go.Scatter(
                x=plot_df.loc[up_mask, "x"], y=plot_df.loc[up_mask, "oscillator"], mode="markers",
                marker=dict(symbol="triangle-up", size=8, color=bull, line=dict(width=0)),
                hovertemplate="%{x}<br>CrossUp: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=3, col=1,
        )
    if dn_mask.any():
        fig.add_trace(
            go.Scatter(
                x=plot_df.loc[dn_mask, "x"], y=plot_df.loc[dn_mask, "oscillator"], mode="markers",
                marker=dict(symbol="triangle-down", size=8, color=bear, line=dict(width=0)),
                hovertemplate="%{x}<br>CrossDown: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=3, col=1,
        )

    # X-axis tick reduction for readability while preserving category gaps removal.
    n = len(x)
    tick_step = max(1, n // 12)
    tickvals = x[::tick_step]

    fig.update_layout(
        title=title,
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color="#c9d1d9"),
        height=980,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        bargap=0.08,
        bargroupgap=0.0,
        xaxis_rangeslider_visible=False,
    )

    for r in (1, 2, 3):
        fig.update_xaxes(
            row=r, col=1,
            type="category",
            categoryorder="array",
            categoryarray=x,
            showgrid=True,
            gridcolor=grid,
            tickmode="array",
            tickvals=tickvals,
            ticktext=tickvals,
        )
    fig.update_yaxes(showgrid=True, gridcolor=grid, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=grid, row=2, col=1, rangemode="tozero")
    fig.update_yaxes(showgrid=True, gridcolor=grid, row=3, col=1)

    add_feature_table(fig, plot_df)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_html}")


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--symbol", type=str, default="300136", help="A股代码，如 300136")
    ap.add_argument("--freq", type=str, default="d", help="K线周期：d(日线) 或 1/5/15/30/60(分钟)")
    ap.add_argument("--bars", type=int, default=220, help="展示最近 N 根K线")
    ap.add_argument("--out", type=str, default="dynamic_volume_profile_oscillator.html", help="输出HTML文件")

    ap.add_argument("--lookback", type=int, default=50)
    ap.add_argument("--profile-periods", type=int, default=10)
    ap.add_argument("--smoothing", type=int, default=5)
    ap.add_argument("--sensitivity", type=float, default=1.0)
    ap.add_argument("--mean-reversion", action="store_true", default=True)
    ap.add_argument("--no-mean-reversion", action="store_false", dest="mean_reversion")
    ap.add_argument("--use-adaptive-midline", action="store_true", default=True)
    ap.add_argument("--fixed-midline", action="store_false", dest="use_adaptive_midline")
    ap.add_argument("--midline-period", type=int, default=50)
    ap.add_argument("--zone-width", type=float, default=1.5)
    ap.add_argument("--color-bars", action="store_true", default=True)
    ap.add_argument("--no-color-bars", action="store_false", dest="color_bars")

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

    df = fetch_kline_pytdx(args.symbol, start=str(start), end=str(end), freq=freq)
    cfg = DVPOConfig(
        lookback=args.lookback,
        profile_periods=args.profile_periods,
        smoothing=args.smoothing,
        sensitivity=args.sensitivity,
        mean_reversion=bool(args.mean_reversion),
        use_adaptive_midline=bool(args.use_adaptive_midline),
        midline_period=args.midline_period,
        zone_width=args.zone_width,
        color_bars=bool(args.color_bars),
    )

    full = dynamic_volume_profile_oscillator(df, cfg)
    show = full.tail(args.bars).copy()

    title = (
        f"{args.symbol} ({args.freq}) Dynamic Volume Profile Oscillator "
        f"[lookback={cfg.lookback}, pp={cfg.profile_periods}, smooth={cfg.smoothing}, "
        f"sens={cfg.sensitivity}, adaptive={cfg.use_adaptive_midline}]"
    )
    build_plot(show, out_html=args.out, title=title, color_bars=cfg.color_bars)


if __name__ == "__main__":
    main()
