#!/usr/bin/env python3
"""
LuxAlgo-style Volume Profile with Node Detection, ported from Pine Script to Python.

Original indicator: "Volume Profile with Node Detection [LuxAlgo]"
License noted in source: CC BY-NC-SA 4.0, © LuxAlgo.

What this port preserves:
- Lookback-limited fixed-range volume profile
- Volume distribution into price rows by overlap proportion
- Bullish/down volume split using close > open polarity
- POC, VAH, VAL calculation using the same expansion logic from POC
- Peak/trough node detection by N preceding and N succeeding rows
- Highest/lowest N volume node highlighting
- TradingView-like dark HTML visualization with candlesticks + right/left profile

Data sources:
- CSV OHLCV input, same as the earlier version
- pytdx A-share K-line input via --tdx-code / --tdx-period / --tdx-count

Main limitation versus TradingView:
- Pine's request.security_lower_tf can pull lower-timeframe bars automatically. Offline Python cannot
  do that unless you fetch/provide the same lower-timeframe bars. With pytdx, use --tdx-period 1m/5m/etc.
  when you want a closer match to TradingView lower-timeframe volume allocation.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional
import socket

import numpy as np
import pandas as pd
import plotly.graph_objects as go


Placement = Literal["right", "left"]
NodeMode = Literal["peaks", "clusters", "none", "troughs"]
PocMode = Literal["developing", "regular", "none"]
GradientMode = Literal["gradient", "classic"]


@dataclass
class VolumeProfileConfig:
    profile_lookback_length: int = 360
    value_area_threshold: float = 0.70
    profile_number_of_rows: int = 100
    profile_width: float = 0.31
    profile_placement: Placement = "right"
    profile_horizontal_offset: int = 13
    profile_show: bool = True
    profile_gradient_colors: GradientMode = "gradient"

    poc_show: PocMode = "none"
    poc_width: int = 2
    vah_show: bool = False
    val_show: bool = False
    profile_price_labels: Literal["tiny", "small", "normal", "none"] = "small"
    value_area_background: bool = False
    profile_background: bool = False

    peaks_show: Literal["peaks", "clusters", "none"] = "peaks"
    peaks_detection_percent: float = 0.09
    troughs_show: Literal["troughs", "clusters", "none"] = "none"
    troughs_detection_percent: float = 0.07
    volume_node_threshold: float = 0.01
    highest_n_volume_nodes: int = 0
    lowest_n_volume_nodes: int = 0

    # TradingView-like colors. Values are RGBA strings or hex strings.
    value_area_up_color: str = "rgba(41,98,255,0.70)"
    value_area_down_color: str = "rgba(251,192,45,0.70)"
    profile_up_volume_color: str = "rgba(93,96,107,0.50)"
    profile_down_volume_color: str = "rgba(209,212,220,0.50)"
    poc_color: str = "#fbc02d"
    vah_color: str = "#2962ff"
    val_color: str = "#2962ff"
    peak_volume_color: str = "rgba(33,150,243,0.50)"
    trough_volume_color: str = "rgba(128,128,128,0.50)"
    highest_volume_color: str = "rgba(255,152,0,0.75)"
    lowest_volume_color: str = "rgba(0,0,128,0.75)"
    value_area_background_color: str = "rgba(41,98,255,0.11)"
    profile_background_color: str = "rgba(41,98,255,0.05)"


@dataclass
class VolumeProfileResult:
    profile_df: pd.DataFrame
    lowest_price: float
    highest_price: float
    price_step: float
    poc_level: int
    vah_level: int
    val_level: int
    poc_price: float
    vah_price: float
    val_price: float
    start_index: int
    last_index: int
    developing_poc: Optional[pd.DataFrame]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rename).copy()
    aliases = {
        "datetime": ["datetime", "date", "time", "timestamp"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "adj close"],
        "volume": ["volume", "vol", "v"],
    }
    out = pd.DataFrame(index=df.index)
    for canonical, names in aliases.items():
        for name in names:
            if name in df.columns:
                out[canonical] = df[name]
                break
    missing = [c for c in ["open", "high", "low", "close", "volume"] if c not in out]
    if missing:
        raise ValueError(f"CSV must contain OHLCV columns. Missing: {missing}")
    if "datetime" in out:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    else:
        out["datetime"] = pd.RangeIndex(len(out)).astype(str)
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("Need at least 2 valid OHLCV rows.")
    return out


def _overlap_proportion(level_low: float, level_high: float, price_level: float, price_step: float) -> float:
    """Pine switch block replicated for volume proportion in each price bin."""
    denom = level_high - level_low
    if denom <= 0 or price_step <= 0:
        return 0.0
    if level_low >= price_level and level_high > price_level + price_step:
        return (price_level + price_step - level_low) / denom
    if level_high <= price_level + price_step and level_low < price_level:
        return (level_high - price_level) / denom
    if level_low >= price_level and level_high <= price_level + price_step:
        return 1.0
    return price_step / denom


def _rgba_with_alpha(color: str, alpha: float) -> str:
    """Best-effort alpha override for rgba/hex colors."""
    alpha = max(0.0, min(1.0, alpha))
    s = color.strip()
    if s.startswith("rgba"):
        nums = s[s.find("(") + 1 : s.rfind(")")].split(",")
        return f"rgba({int(float(nums[0]))},{int(float(nums[1]))},{int(float(nums[2]))},{alpha:.4f})"
    if s.startswith("rgb"):
        nums = s[s.find("(") + 1 : s.rfind(")")].split(",")
        return f"rgba({int(float(nums[0]))},{int(float(nums[1]))},{int(float(nums[2]))},{alpha:.4f})"
    if s.startswith("#") and len(s) == 7:
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        return f"rgba({r},{g},{b},{alpha:.4f})"
    return color


def _tv_gradient(color: str, ratio: float, min_alpha: float = 0.05, max_alpha: float = 1.0) -> str:
    ratio = 0.0 if not np.isfinite(ratio) else max(0.0, min(1.0, ratio))
    alpha = min_alpha + (max_alpha - min_alpha) * ratio
    return _rgba_with_alpha(color, alpha)


def compute_volume_profile(df: pd.DataFrame, cfg: VolumeProfileConfig = VolumeProfileConfig()) -> VolumeProfileResult:
    data = _normalize_columns(df)

    lookback = min(max(int(cfg.profile_lookback_length), 10), 5000)
    # Pine uses vp_profileLength := input - 1, and processes from last_bar_index - vp_profileLength.
    lookback = min(lookback, len(data) - 1)
    window = data.iloc[-lookback:].copy().reset_index(drop=True)
    start_index = len(data) - len(window)
    last_index = len(data) - 1

    lowest_price = float(window["low"].min())
    highest_price = float(window["high"].max())
    rows = int(np.clip(cfg.profile_number_of_rows, 30, 130))
    price_range = highest_price - lowest_price
    if price_range <= 0:
        raise ValueError("Highest and lowest prices are equal; cannot build a volume profile.")
    price_step = price_range / rows

    total_volume = np.zeros(rows, dtype=float)
    bullish_volume = np.zeros(rows, dtype=float)

    # Developing POC path, equivalent in purpose to the Pine polyline.
    dev_points = []

    for i, row in window.iterrows():
        level_high = float(row["high"])
        level_low = float(row["low"])
        vol = float(row["volume"])
        polarity = bool(row["close"] > row["open"])

        if level_high <= level_low:
            # Degenerate bar: put full volume in the containing slot.
            slot = int(np.clip(math.floor((level_low - lowest_price) / price_step), 0, rows - 1))
            total_volume[slot] += vol
            if polarity:
                bullish_volume[slot] += vol
        else:
            start_slot = max(math.floor((level_low - lowest_price) / price_step), 0)
            end_slot = min(math.floor((level_high - lowest_price) / price_step), rows - 1)
            for price_level_index in range(start_slot, end_slot + 1):
                price_level = lowest_price + price_level_index * price_step
                prop = _overlap_proportion(level_low, level_high, price_level, price_step)
                add = vol * prop
                total_volume[price_level_index] += add
                if polarity:
                    bullish_volume[price_level_index] += add

        if cfg.poc_show == "developing":
            poc_i = int(np.argmax(total_volume)) if total_volume.max() > 0 else -1
            if poc_i >= 0:
                dev_points.append(
                    {
                        "x": start_index + i,
                        "poc": lowest_price + (poc_i + 0.5) * price_step,
                    }
                )

    bearish_volume = np.abs(2 * bullish_volume - total_volume)
    max_vol = float(total_volume.max())
    poc_level = int(np.argmax(total_volume)) if max_vol > 0 else -1

    target_value_area_volume = float(total_volume.sum()) * float(cfg.value_area_threshold)
    value_area_volume = float(total_volume[poc_level]) if poc_level >= 0 else 0.0
    vah_level = poc_level
    val_level = poc_level

    while value_area_volume < target_value_area_volume and poc_level >= 0:
        if val_level == 0 and vah_level == rows - 1:
            break
        volume_above = float(total_volume[vah_level + 1]) if vah_level < rows - 1 else 0.0
        volume_below = float(total_volume[val_level - 1]) if val_level > 0 else 0.0
        if volume_below == 0 and volume_above == 0:
            break
        if volume_above >= volume_below:
            value_area_volume += volume_above
            vah_level += 1
        else:
            value_area_volume += volume_below
            val_level -= 1

    price_low = lowest_price + np.arange(rows) * price_step
    price_high = lowest_price + (np.arange(rows) + 1) * price_step
    price_mid = lowest_price + (np.arange(rows) + 0.5) * price_step
    profile_df = pd.DataFrame(
        {
            "row": np.arange(rows),
            "price_low": price_low,
            "price_mid": price_mid,
            "price_high": price_high,
            "total_volume": total_volume,
            "bullish_volume": bullish_volume,
            "bearish_volume": np.maximum(total_volume - bullish_volume, 0.0),
            "pine_bearish_volume_formula": bearish_volume,
            "is_value_area": (np.arange(rows) >= val_level) & (np.arange(rows) <= vah_level),
            "is_poc": np.arange(rows) == poc_level,
            "is_peak": False,
            "is_trough": False,
            "is_highest_node": False,
            "is_lowest_node": False,
        }
    )

    _detect_nodes(profile_df, cfg)

    return VolumeProfileResult(
        profile_df=profile_df,
        lowest_price=lowest_price,
        highest_price=highest_price,
        price_step=price_step,
        poc_level=poc_level,
        vah_level=int(vah_level),
        val_level=int(val_level),
        poc_price=lowest_price + (poc_level + 0.5) * price_step if poc_level >= 0 else float("nan"),
        vah_price=lowest_price + (vah_level + 1.0) * price_step if vah_level >= 0 else float("nan"),
        val_price=lowest_price + (val_level + 0.0) * price_step if val_level >= 0 else float("nan"),
        start_index=start_index,
        last_index=last_index,
        developing_poc=pd.DataFrame(dev_points) if dev_points else None,
    )


def _detect_nodes(profile_df: pd.DataFrame, cfg: VolumeProfileConfig) -> None:
    vols = profile_df["total_volume"].to_numpy(dtype=float)
    rows = len(vols)
    max_vol = float(np.max(vols)) if len(vols) else 0.0
    if max_vol <= 0:
        return

    # Peaks: center must be strictly greater than N rows before and after.
    if cfg.peaks_show != "none" and cfg.peaks_detection_percent > 0:
        n = int(rows * cfg.peaks_detection_percent)
        padded = np.concatenate([np.zeros(n), vols, np.zeros(n)])
        for k in range(rows):
            center = padded[k + n]
            left = padded[k : k + n]
            right = padded[k + n + 1 : k + 2 * n + 1]
            is_peak = (n == 0 or (np.all(center > left) and np.all(center > right))) and center / max_vol > cfg.volume_node_threshold
            if is_peak:
                profile_df.loc[k, "is_peak"] = True
                if cfg.peaks_show == "clusters":
                    lo = max(0, k - n)
                    hi = min(rows - 1, k + n)
                    profile_df.loc[lo:hi, "is_peak"] = True

    # Troughs: center must be strictly lower than N rows before and after, and above threshold per Pine code.
    if cfg.troughs_show != "none" and cfg.troughs_detection_percent > 0:
        n = int(rows * cfg.troughs_detection_percent)
        padded = np.concatenate([np.full(n, max_vol), vols, np.full(n, max_vol)])
        for k in range(rows):
            center = padded[k + n]
            left = padded[k : k + n]
            right = padded[k + n + 1 : k + 2 * n + 1]
            is_trough = (n == 0 or (np.all(center < left) and np.all(center < right))) and center / max_vol > cfg.volume_node_threshold
            if is_trough:
                profile_df.loc[k, "is_trough"] = True
                if cfg.troughs_show == "clusters":
                    lo = max(0, k - n)
                    hi = min(rows - 1, k + n)
                    profile_df.loc[lo:hi, "is_trough"] = True

    if cfg.highest_n_volume_nodes > 0:
        n = min(int(cfg.highest_n_volume_nodes), rows)
        idx = np.argsort(-vols)[:n]
        profile_df.loc[idx, "is_highest_node"] = True

    if cfg.lowest_n_volume_nodes > 0:
        non_duplicate_order = []
        seen = set()
        for idx in np.argsort(vols):
            val = float(vols[idx])
            if val not in seen:
                seen.add(val)
                non_duplicate_order.append(idx)
            if len(non_duplicate_order) >= int(cfg.lowest_n_volume_nodes):
                break
        profile_df.loc[non_duplicate_order, "is_lowest_node"] = True


def make_volume_profile_figure(
    df: pd.DataFrame,
    result: VolumeProfileResult,
    cfg: VolumeProfileConfig = VolumeProfileConfig(),
    title: str = "Volume Profile with Node Detection",
) -> go.Figure:
    data = _normalize_columns(df)
    lookback = result.last_index - result.start_index + 1
    window = data.iloc[-lookback:].copy().reset_index(drop=True)

    # Use numeric x so TradingView-like future bars can host the profile.
    x = np.arange(len(window))
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=window["open"],
            high=window["high"],
            low=window["low"],
            close=window["close"],
            increasing_line_color="#26a69a",
            increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350",
            decreasing_fillcolor="#ef5350",
            name="OHLC",
        )
    )

    rows = len(result.profile_df)
    max_vol = float(result.profile_df["total_volume"].max()) or 1.0
    profile_plotting_length = min(360, lookback)
    profile_width_bars = max(1.0, profile_plotting_length * cfg.profile_width)
    offset = int(profile_width_bars + cfg.profile_horizontal_offset)
    last_x = len(window) - 1
    profile_anchor = last_x + offset if cfg.profile_placement == "right" else 0

    shapes = []
    annotations = []

    if cfg.profile_background:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=0,
                x1=last_x,
                y0=result.lowest_price,
                y1=result.highest_price,
                line=dict(color=cfg.profile_background_color, dash="dot"),
                fillcolor=cfg.profile_background_color,
                layer="below",
            )
        )
    if cfg.value_area_background:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=0,
                x1=last_x,
                y0=result.val_price,
                y1=result.vah_price,
                line=dict(color=cfg.value_area_background_color, dash="dot"),
                fillcolor=cfg.value_area_background_color,
                layer="below",
            )
        )

    # Profile bars: bullish part and remaining/down part, stacked horizontally like TV boxes.
    if cfg.profile_show:
        for _, r in result.profile_df.iterrows():
            total = float(r.total_volume)
            bull = float(r.bullish_volume)
            down = max(total - bull, 0.0)
            y0 = float(r.price_low + 0.1 * result.price_step)
            y1 = float(r.price_low + 0.9 * result.price_step)
            ratio = total / max_vol
            in_va = bool(r.is_value_area)

            up_base = cfg.value_area_up_color if in_va else cfg.profile_up_volume_color
            down_base = cfg.value_area_down_color if in_va else cfg.profile_down_volume_color
            up_color = _tv_gradient(up_base, ratio) if cfg.profile_gradient_colors == "gradient" else up_base
            down_color = _tv_gradient(down_base, ratio) if cfg.profile_gradient_colors == "gradient" else down_base

            bull_w = bull / max_vol * profile_width_bars
            down_w = down / max_vol * profile_width_bars

            if cfg.profile_placement == "right":
                x1 = profile_anchor
                x0 = x1 - bull_w
                shapes.append(dict(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(width=0), fillcolor=up_color, layer="above"))
                x1d = x0
                x0d = x1d - down_w
                shapes.append(dict(type="rect", xref="x", yref="y", x0=x0d, x1=x1d, y0=y0, y1=y1, line=dict(width=0), fillcolor=down_color, layer="above"))
                end_x = x0d
            else:
                x0 = profile_anchor
                x1 = x0 + bull_w
                shapes.append(dict(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(width=0), fillcolor=up_color, layer="above"))
                x0d = x1
                x1d = x0d + down_w
                shapes.append(dict(type="rect", xref="x", yref="y", x0=x0d, x1=x1d, y0=y0, y1=y1, line=dict(width=0), fillcolor=down_color, layer="above"))
                end_x = x1d

            # Node overlays follow Pine's full-span overlay behavior.
            overlay_color = None
            if bool(r.is_peak):
                overlay_color = cfg.peak_volume_color
            if bool(r.is_trough):
                overlay_color = cfg.trough_volume_color
            if bool(r.is_highest_node):
                overlay_color = cfg.highest_volume_color
            if bool(r.is_lowest_node):
                overlay_color = cfg.lowest_volume_color
            if overlay_color:
                if cfg.profile_placement == "right":
                    shapes.append(dict(type="rect", xref="x", yref="y", x0=0, x1=end_x, y0=y0, y1=y1, line=dict(width=0), fillcolor=overlay_color, layer="above"))
                else:
                    shapes.append(dict(type="rect", xref="x", yref="y", x0=end_x, x1=last_x, y0=y0, y1=y1, line=dict(width=0), fillcolor=overlay_color, layer="above"))

    # POC/VAH/VAL lines.
    def add_hline(y: float, color: str, width: int = 1, name: str = "") -> None:
        shapes.append(dict(type="line", xref="x", yref="y", x0=0, x1=profile_anchor, y0=y, y1=y, line=dict(color=color, width=width), layer="above"))
        annotations.append(dict(x=profile_anchor, y=y, text=name, showarrow=False, xanchor="left", font=dict(color=color, size=11), bgcolor="rgba(19,23,34,0.75)"))

    if cfg.vah_show:
        add_hline(result.vah_price, cfg.vah_color, 1, f"VAH {result.vah_price:.4g}")
    if cfg.poc_show == "regular":
        add_hline(result.poc_price, cfg.poc_color, cfg.poc_width, f"POC {result.poc_price:.4g}")
    if cfg.val_show:
        add_hline(result.val_price, cfg.val_color, 1, f"VAL {result.val_price:.4g}")

    if cfg.poc_show == "developing" and result.developing_poc is not None:
        fig.add_trace(
            go.Scatter(
                x=result.developing_poc["x"] - result.start_index,
                y=result.developing_poc["poc"],
                mode="lines",
                line=dict(color=cfg.poc_color, width=cfg.poc_width),
                name="Developing POC",
            )
        )

    # Price labels similar to TV labels.
    if cfg.profile_price_labels != "none":
        label_x = profile_anchor + 1 if cfg.profile_placement == "right" else -1
        for y, label, color in [
            (result.highest_price, f"High {result.highest_price:.4g}", "#d1d4dc"),
            (result.vah_price, f"VAH {result.vah_price:.4g}", cfg.vah_color),
            (result.poc_price, f"POC {result.poc_price:.4g}", cfg.poc_color),
            (result.val_price, f"VAL {result.val_price:.4g}", cfg.val_color),
            (result.lowest_price, f"Low {result.lowest_price:.4g}", "#d1d4dc"),
        ]:
            annotations.append(dict(x=label_x, y=y, text=label, showarrow=False, xanchor="left", font=dict(color=color, size=11), bgcolor="rgba(19,23,34,0.85)"))

    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(color="#d1d4dc", size=18)),
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(color="#d1d4dc", family="Arial, sans-serif"),
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor="rgba(120,123,134,0.18)",
            zeroline=False,
            tickmode="array",
            tickvals=list(np.linspace(0, max(0, len(window) - 1), min(8, len(window))).astype(int)),
            ticktext=[str(window.loc[i, "datetime"])[:16] for i in list(np.linspace(0, max(0, len(window) - 1), min(8, len(window))).astype(int))],
            range=[-2, profile_anchor + max(4, profile_width_bars * 0.08)],
        ),
        yaxis=dict(
            side="right",
            showgrid=True,
            gridcolor="rgba(120,123,134,0.18)",
            zeroline=False,
            fixedrange=False,
        ),
        margin=dict(l=24, r=84, t=52, b=34),
        height=820,
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
        hovermode="x unified",
    )
    return fig



TDX_PERIOD_MAP: dict[str, int] = {
    "5m": 0,
    "15m": 1,
    "30m": 2,
    "60m": 3,
    "1h": 3,
    "day": 4,
    "d": 4,
    "week": 5,
    "w": 5,
    "month": 6,
    "m": 6,
    "1m": 7,
}

TDX_DEFAULT_SERVERS: list[tuple[str, int]] = [
    ("119.147.212.81", 7709),
    ("14.215.128.18", 7709),
    ("180.153.18.170", 7709),
    ("180.153.18.171", 7709),
    ("202.108.253.130", 7709),
    ("47.103.48.45", 7709),
    ("59.173.18.69", 7709),
    ("218.75.126.9", 7709),
]


def infer_tdx_market(code: str) -> int:
    """pytdx market: 0=深圳, 1=上海. Covers common A-share / index prefixes."""
    s = str(code).strip().upper()
    if s.startswith(("SH", "SSE")):
        return 1
    if s.startswith(("SZ", "SZE")):
        return 0
    s = s.replace("SH", "").replace("SZ", "").replace(".", "")[-6:]
    if s.startswith(("5", "6", "9")):
        return 1
    # 0/1/2/3 are generally Shenzhen securities and indexes in pytdx.
    return 0


def normalize_tdx_code(code: str) -> str:
    s = str(code).strip().upper()
    s = s.replace("SH", "").replace("SZ", "").replace("SSE", "").replace("SZE", "")
    s = s.replace(".", "")
    if len(s) < 6:
        s = s.zfill(6)
    return s[-6:]


def _tdx_datetime_from_row(row: pd.Series) -> pd.Timestamp:
    if "datetime" in row and pd.notna(row["datetime"]):
        return pd.to_datetime(row["datetime"])
    if {"year", "month", "day"}.issubset(row.index):
        hour = int(row.get("hour", 0) or 0)
        minute = int(row.get("minute", 0) or 0)
        return pd.Timestamp(int(row["year"]), int(row["month"]), int(row["day"]), hour, minute)
    if "date" in row and pd.notna(row["date"]):
        return pd.to_datetime(row["date"])
    raise ValueError("pytdx returned rows without recognizable datetime fields.")


def fetch_pytdx_bars(
    code: str,
    period: str = "day",
    count: int = 800,
    market: Optional[int] = None,
    server: Optional[str] = None,
    port: int = 7709,
    timeout: float = 3.0,
) -> pd.DataFrame:
    """
    Fetch OHLCV from pytdx and return canonical columns:
    datetime, open, high, low, close, volume.

    Parameters
    ----------
    code: stock/index code, e.g. 000001, 300750, 600519, SH000001, SZ399001
    period: one of 1m, 5m, 15m, 30m, 60m/1h, day/d, week/w, month/m
    count: number of bars to fetch. pytdx normally caps each request around 800 bars, so this paginates.
    market: pytdx market id. Leave None to infer: 0=SZ, 1=SH.
    server/port: optional fixed TDX server; otherwise multiple servers are tried automatically.
    """
    try:
        from pytdx.hq import TdxHq_API
    except ImportError as exc:
        raise ImportError("未安装 pytdx。请先运行：pip install pytdx") from exc

    code6 = normalize_tdx_code(code)
    market_id = infer_tdx_market(code) if market is None else int(market)
    period_key = str(period).lower()
    if period_key not in TDX_PERIOD_MAP:
        raise ValueError(f"Unsupported --tdx-period: {period}. Use one of: {', '.join(TDX_PERIOD_MAP)}")
    category = TDX_PERIOD_MAP[period_key]
    count = int(max(1, count))
    servers = [(server, port)] if server else TDX_DEFAULT_SERVERS

    errors: list[str] = []
    for host, host_port in servers:
        api = TdxHq_API(heartbeat=True, auto_retry=True, raise_exception=False)
        try:
            socket.setdefaulttimeout(timeout)
            if not api.connect(host, int(host_port), time_out=timeout):
                errors.append(f"{host}:{host_port} connect failed")
                continue

            parts = []
            # pytdx offset 0 is the latest page; larger offsets walk backward.
            for start in range(0, count, 800):
                batch_count = min(800, count - start)
                raw = api.get_security_bars(category, market_id, code6, start, batch_count)
                if not raw:
                    break
                part = api.to_df(raw)
                if part is not None and not part.empty:
                    parts.append(part)

            if not parts:
                errors.append(f"{host}:{host_port} returned no data for {code6}")
                continue

            out = pd.concat(parts, ignore_index=True)
            # Normalize columns from pytdx. Common fields: open/high/low/close/vol/amount/datetime.
            if "vol" in out.columns and "volume" not in out.columns:
                out = out.rename(columns={"vol": "volume"})
            out["datetime"] = out.apply(_tdx_datetime_from_row, axis=1)
            out = out[["datetime", "open", "high", "low", "close", "volume"]].copy()
            out = out.dropna().drop_duplicates(subset=["datetime"]).sort_values("datetime").tail(count).reset_index(drop=True)
            if out.empty:
                errors.append(f"{host}:{host_port} normalized data empty for {code6}")
                continue
            return out
        except Exception as exc:  # keep trying other servers
            errors.append(f"{host}:{host_port} {type(exc).__name__}: {exc}")
        finally:
            try:
                api.disconnect()
            except Exception:
                pass

    detail = "\n".join(errors[-8:])
    raise RuntimeError(f"pytdx 拉取失败：{code6}, market={market_id}, period={period_key}\n{detail}")


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_html(
    df: pd.DataFrame,
    output_html: str | Path,
    cfg: VolumeProfileConfig = VolumeProfileConfig(),
    title: str = "LuxAlgo-style Volume Profile with Node Detection",
) -> VolumeProfileResult:
    result = compute_volume_profile(df, cfg)
    fig = make_volume_profile_figure(df, result, cfg, title=title)
    output_html = Path(output_html)
    fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)
    return result


def _demo_data(n: int = 520, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.012, n)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.018, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.018, n))
    volume = rng.lognormal(mean=13, sigma=0.35, size=n).astype(int)
    dt = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({"datetime": dt, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a TradingView-style LuxAlgo Volume Profile HTML from CSV or pytdx A-share data.")
    p.add_argument("--csv", type=str, help="Input OHLCV CSV. Required columns: datetime/date/time, open, high, low, close, volume.")
    p.add_argument("--tdx-code", type=str, help="Use pytdx as data source. Example: 600519, 300750, SH000001, SZ399001.")
    p.add_argument("--tdx-period", type=str, default="day", choices=list(TDX_PERIOD_MAP.keys()), help="pytdx period: 1m, 5m, 15m, 30m, 60m/1h, day, week, month.")
    p.add_argument("--tdx-count", type=int, default=800, help="Number of pytdx bars to fetch before applying lookback.")
    p.add_argument("--tdx-market", type=int, choices=[0, 1], help="Optional pytdx market override: 0=SZ, 1=SH.")
    p.add_argument("--tdx-server", type=str, help="Optional fixed TDX server IP.")
    p.add_argument("--tdx-port", type=int, default=7709, help="TDX server port, default 7709.")
    p.add_argument("--output", type=str, default="volume_profile.html", help="Output HTML path.")
    p.add_argument("--lookback", type=int, default=360, help="Profile lookback length.")
    p.add_argument("--rows", type=int, default=100, help="Profile number of rows, 30-130.")
    p.add_argument("--value-area", type=float, default=70.0, help="Value area percentage, e.g. 70.")
    p.add_argument("--placement", choices=["right", "left"], default="right")
    p.add_argument("--poc", choices=["developing", "regular", "none"], default="regular")
    p.add_argument("--vah", action="store_true", help="Show VAH line.")
    p.add_argument("--val", action="store_true", help="Show VAL line.")
    p.add_argument("--peaks", choices=["peaks", "clusters", "none"], default="peaks")
    p.add_argument("--peak-percent", type=float, default=9.0, help="Peak node detection percent.")
    p.add_argument("--troughs", choices=["troughs", "clusters", "none"], default="none")
    p.add_argument("--trough-percent", type=float, default=7.0, help="Trough node detection percent.")
    p.add_argument("--threshold", type=float, default=1.0, help="Volume node threshold percent.")
    p.add_argument("--highest", type=int, default=0, help="Highlight highest N volume nodes.")
    p.add_argument("--lowest", type=int, default=0, help="Highlight lowest N unique volume nodes.")
    p.add_argument("--demo", action="store_true", help="Generate HTML from synthetic demo data.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    source_count = sum(bool(x) for x in [args.csv, args.demo, args.tdx_code])
    if source_count != 1:
        raise SystemExit("Choose exactly one data source: --tdx-code, --csv, or --demo.")

    if args.demo:
        df = _demo_data()
        source_title = "Synthetic demo"
    elif args.tdx_code:
        df = fetch_pytdx_bars(
            code=args.tdx_code,
            period=args.tdx_period,
            count=args.tdx_count,
            market=args.tdx_market,
            server=args.tdx_server,
            port=args.tdx_port,
        )
        source_title = f"pytdx {normalize_tdx_code(args.tdx_code)} {args.tdx_period}"
    else:
        df = read_csv(args.csv)
        source_title = Path(args.csv).name
    cfg = VolumeProfileConfig(
        profile_lookback_length=args.lookback,
        profile_number_of_rows=args.rows,
        value_area_threshold=args.value_area / 100.0,
        profile_placement=args.placement,
        poc_show=args.poc,
        vah_show=args.vah,
        val_show=args.val,
        peaks_show=args.peaks,
        peaks_detection_percent=args.peak_percent / 100.0,
        troughs_show=args.troughs,
        troughs_detection_percent=args.trough_percent / 100.0,
        volume_node_threshold=args.threshold / 100.0,
        highest_n_volume_nodes=args.highest,
        lowest_n_volume_nodes=args.lowest,
    )
    result = write_html(df, args.output, cfg, title=f"LuxAlgo-style Volume Profile - {source_title}")
    print(f"HTML written: {args.output}")
    print(f"Rows loaded: {len(df)}")
    print(f"POC={result.poc_price:.6g}, VAH={result.vah_price:.6g}, VAL={result.val_price:.6g}")


if __name__ == "__main__":
    main()
