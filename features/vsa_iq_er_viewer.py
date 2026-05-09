# -*- coding: utf-8 -*-
"""
Volume Spread Analysis IQ [TradingIQ] -> Python viewer

Purpose
- 参考用户提供的 Pine 脚本，复刻核心 VSA / Effort-vs-Result 逻辑
- 通过 pytdx 拉取 A 股 K 线
- 输出交互式 HTML：主图 K 线，副图为 Effort / Result / ER 因子与 ER MA
- 导出包含最终解释（effort_vs_result）在内的研究数据 CSV

Notes
- 该脚本聚焦用户要求的 ER 因子、ER 的 MA、最终解释。
- 由于 Pine 中 array.percentrank 为动态数组百分位排序，这里用滚动窗口内
  "当前值的百分位等级" 做等价近似，窗口默认 100，与 Pine pushShift(100) 一致。
- 最终解释对应 Pine 中 effortVsResult 分类。

Example
    python vsa_iq_er_viewer.py --symbol 600547 --freq d --bars 300 \
        --out vsa_iq_600547.html --csv-out vsa_iq_600547.csv
"""
from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


# -----------------------------
# Data helpers
# -----------------------------
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
    try:
        from pytdx.params import TDXParams
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("请先安装 pytdx: pip install pytdx") from exc
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


def connect_pytdx():
    try:
        from pytdx.hq import TdxHq_API
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("请先安装 pytdx: pip install pytdx") from exc
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


# -----------------------------
# Indicator helpers
# -----------------------------
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


def calc_ma(src: pd.Series, length: int, ma_type: str) -> pd.Series:
    typ = str(ma_type).upper()
    if typ == "EMA":
        return pine_ema(src, length)
    if typ == "WMA":
        return src.rolling(length, min_periods=length).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
            raw=True,
        )
    if typ == "RMA":
        return src.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return src.rolling(length, min_periods=length).mean()


def rolling_percent_rank_last(series: pd.Series, window: int = 100) -> pd.Series:
    values = series.astype(float).to_numpy()
    out = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        win = values[start : i + 1]
        win = win[~np.isnan(win)]
        if len(win) == 0 or np.isnan(values[i]):
            continue
        # Pine array.percentrank gives 0..100; use weak rank for current value.
        out[i] = 100.0 * np.sum(win <= values[i]) / len(win)
    return pd.Series(out, index=series.index)


def zscore_strong_move(close: pd.Series, lag: int = 3, ma_len: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(lag))
    mean = log_ret.rolling(ma_len, min_periods=ma_len).mean()
    std = log_ret.rolling(ma_len, min_periods=ma_len).std(ddof=1)
    return (log_ret - mean) / std.replace(0.0, np.nan)


def compute_simple_direction(close: pd.Series, lookback: int = 3) -> pd.Series:
    pivot_ma = close.rolling(lookback, min_periods=lookback).mean()
    slope = pivot_ma.diff()
    direction = np.where(slope > 0, 1, np.where(slope < 0, -1, 0))
    return pd.Series(direction, index=close.index)


def classify_easy_rank(rank: pd.Series) -> pd.Series:
    arr = rank.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for i, x in enumerate(arr):
        if np.isnan(x):
            continue
        if x == 100:
            out[i] = 10
        elif x > 90:
            out[i] = 9
        elif x > 80:
            out[i] = 8
        elif x > 70:
            out[i] = 7
        elif x > 60:
            out[i] = 6
        elif x > 50:
            out[i] = 5
        elif x > 40:
            out[i] = 4
        elif x > 30:
            out[i] = 3
        elif x > 20:
            out[i] = 2
        else:
            out[i] = 1
    return pd.Series(out, index=rank.index)


def compute_vsa_iq(
    df: pd.DataFrame,
    ma_length: int = 14,
    ma_type: str = "SMA",
    bg_lookback: int = 10,
    trend_rule: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    # Thresholds: default Standard
    small_body_thresh = 0.10
    rej_body_max = 0.45
    rej_wick_thresh = 0.45
    close_thresh = 0.60
    spread_normal_min = 30
    spread_wide_min = 70
    spread_ultra_wide_min = 90
    spread_ultra_narrow_max = 10
    low_vol_max = 30
    ultra_low_vol_max = 10
    high_vol_min = 70
    ultra_high_vol_min = 90
    bg_thresh = 2.0

    out["bar_range"] = out["high"] - out["low"]
    out["vol_rank"] = rolling_percent_rank_last(out["volume"], 100)
    out["spread_rank"] = rolling_percent_rank_last(out["bar_range"], 100)

    out["R"] = np.where(
        out["bar_range"].replace(0.0, np.nan).notna(),
        ((out["close"] - out["low"]) - (out["high"] - out["close"]))
        / out["bar_range"].replace(0.0, np.nan),
        0.0,
    )
    out["R"] = out["R"].fillna(0.0)

    out["bar_direction"] = np.sign(out["close"].diff())
    out["up_bar"] = out["bar_direction"] > 0
    out["down_bar"] = out["bar_direction"] < 0

    # Volume class / effort
    out["vol_class"] = np.select(
        [
            out["vol_rank"] >= ultra_high_vol_min,
            out["vol_rank"] >= high_vol_min,
            out["vol_rank"] >= low_vol_max,
            out["vol_rank"] >= ultra_low_vol_max,
        ],
        ["Ultra High", "High", "Normal", "Low"],
        default="Ultra Low",
    )
    out["effort"] = np.select(
        [
            out["vol_rank"] >= high_vol_min,
            out["vol_rank"] >= low_vol_max,
        ],
        ["High Effort", "Normal"],
        default="Low Effort",
    )

    # Spread class
    out["spread_class"] = np.select(
        [
            out["spread_rank"] >= spread_ultra_wide_min,
            out["spread_rank"] >= spread_wide_min,
            out["spread_rank"] >= spread_normal_min,
            out["spread_rank"] >= spread_ultra_narrow_max,
        ],
        ["Ultra Wide", "Wide", "Normal", "Narrow"],
        default="Ultra Narrow",
    )

    # Close position
    out["close_pos"] = np.select(
        [out["R"] >= close_thresh, out["R"] <= -close_thresh],
        ["High", "Low"],
        default="Middle",
    )

    out["body"] = (out["close"] - out["open"]).abs()
    out["upper_wick"] = out["high"] - np.maximum(out["open"], out["close"])
    out["lower_wick"] = np.minimum(out["open"], out["close"]) - out["low"]

    out["body_ratio"] = (out["body"] / out["bar_range"].replace(0.0, np.nan)).fillna(0.0)
    out["upper_wick_ratio"] = (out["upper_wick"] / out["bar_range"].replace(0.0, np.nan)).fillna(0.0)
    out["lower_wick_ratio"] = (out["lower_wick"] / out["bar_range"].replace(0.0, np.nan)).fillna(0.0)

    out["small_body"] = out["body_ratio"] <= small_body_thresh
    out["up_rej"] = (
        (out["upper_wick_ratio"] >= rej_wick_thresh)
        & (out["upper_wick_ratio"] > out["lower_wick_ratio"])
        & (out["body_ratio"] <= rej_body_max)
    )
    out["low_rej"] = (
        (out["lower_wick_ratio"] >= rej_wick_thresh)
        & (out["lower_wick_ratio"] > out["upper_wick_ratio"])
        & (out["body_ratio"] <= rej_body_max)
    )
    out["wide_body"] = out["body_ratio"] >= 0.60

    out["shape"] = np.select(
        [out["up_rej"], out["low_rej"], out["small_body"], out["wide_body"]],
        ["Upper Rejection", "Lower Rejection", "Small-Body", "Wide-Body"],
        default="Balanced",
    )

    # Result classification (Pine order preserved)
    result = np.full(len(out), "Neutral", dtype=object)
    conds_vals: List[Tuple[np.ndarray, str]] = [
        ((out["up_bar"] & (out["close_pos"] == "High") & out["spread_class"].isin(["Ultra Wide", "Wide"])).to_numpy(), "Strong Upward"),
        ((out["up_bar"] & (out["close_pos"] == "High") & (out["spread_class"] == "Normal")).to_numpy(), "Moderate Upward"),
        ((out["up_bar"] & (out["close_pos"] == "Middle") & out["spread_class"].isin(["Ultra Wide", "Wide"])).to_numpy(), "Questionable Upward"),
        ((out["up_bar"] & out["close_pos"].isin(["Middle", "High"]) & out["spread_class"].isin(["Narrow", "Ultra Narrow"])).to_numpy(), "Weak Upward"),
        ((out["up_bar"] & (out["close_pos"] == "Low")).to_numpy(), "Poor Upward"),
        ((out["down_bar"] & (out["close_pos"] == "Low") & out["spread_class"].isin(["Ultra Wide", "Wide"])).to_numpy(), "Strong Downward"),
        ((out["down_bar"] & (out["close_pos"] == "Low") & (out["spread_class"] == "Normal")).to_numpy(), "Moderate Downward"),
        ((out["down_bar"] & (out["close_pos"] == "Middle") & out["spread_class"].isin(["Ultra Wide", "Wide"])).to_numpy(), "Questionable Downward"),
        ((out["down_bar"] & out["close_pos"].isin(["Middle", "Low"]) & out["spread_class"].isin(["Narrow", "Ultra Narrow"])).to_numpy(), "Weak Downward"),
        ((out["down_bar"] & (out["close_pos"] == "High")).to_numpy(), "Poor Downward"),
        (((out["bar_direction"] == 0) & (out["close_pos"] == "High") & out["spread_class"].isin(["Ultra Wide", "Wide"])).to_numpy(), "Upward Bias"),
        (((out["bar_direction"] == 0) & (out["close_pos"] == "Low") & out["spread_class"].isin(["Ultra Wide", "Wide"])).to_numpy(), "Downward Bias"),
        (((out["bar_direction"] == 0) & (out["close_pos"] == "Middle") & out["spread_class"].isin(["Ultra Wide", "Wide"])).to_numpy(), "Neutral Wide"),
    ]
    assigned = np.zeros(len(out), dtype=bool)
    for cond, label in conds_vals:
        mask = cond & (~assigned)
        result[mask] = label
        assigned |= mask
    out["result"] = result

    out["up_result"] = out["result"].isin(
        ["Strong Upward", "Moderate Upward", "Questionable Upward", "Weak Upward", "Poor Upward", "Upward Bias"]
    )
    out["down_result"] = out["result"].isin(
        ["Strong Downward", "Moderate Downward", "Questionable Downward", "Weak Downward", "Poor Downward", "Downward Bias"]
    )

    # effort vs result
    evr = np.full(len(out), "Unclassified", dtype=object)
    rules = [
        (((out["effort"] == "High Effort") & (out["result"] == "Strong Upward")).to_numpy(), "Bullish Efficiency"),
        (((out["effort"] == "High Effort") & (out["result"] == "Strong Downward")).to_numpy(), "Bearish Efficiency"),
        (((out["effort"] == "High Effort") & out["up_result"] & (out["result"] != "Strong Upward")).to_numpy(), "Possible Hidden Weakness"),
        (((out["effort"] == "High Effort") & out["down_result"] & (out["result"] != "Strong Downward")).to_numpy(), "Possible Hidden Strength"),
        (((out["effort"] == "High Effort") & out["result"].isin(["Neutral", "Neutral Wide"])).to_numpy(), "Compression / Absorption"),
        (((out["effort"] == "Low Effort") & (out["result"] == "Strong Upward")).to_numpy(), "Easy Upward Movement"),
        (((out["effort"] == "Low Effort") & (out["result"] == "Strong Downward")).to_numpy(), "Easy Downward Movement"),
        (((out["effort"] == "Low Effort")).to_numpy(), "Low-Effort / Low-Result"),
        (((out["effort"] == "Normal") & out["result"].isin(["Strong Upward", "Moderate Upward"])).to_numpy(), "Constructive Upward"),
        (((out["effort"] == "Normal") & out["result"].isin(["Strong Downward", "Moderate Downward"])).to_numpy(), "Constructive Downward"),
        (((out["effort"] == "Normal")).to_numpy(), "Neutral / Mixed"),
    ]
    assigned = np.zeros(len(out), dtype=bool)
    for cond, label in rules:
        mask = cond & (~assigned)
        evr[mask] = label
        assigned |= mask
    out["effort_vs_result"] = evr

    out["bull_score"] = np.select(
        [
            out["effort_vs_result"] == "Bullish Efficiency",
            out["effort_vs_result"] == "Easy Upward Movement",
            out["effort_vs_result"] == "Constructive Upward",
        ],
        [3.0, 1.0, 2.0],
        default=0.0,
    )
    out["bear_score"] = np.select(
        [
            out["effort_vs_result"] == "Bearish Efficiency",
            out["effort_vs_result"] == "Easy Downward Movement",
            out["effort_vs_result"] == "Constructive Downward",
        ],
        [3.0, 1.0, 2.0],
        default=0.0,
    )
    out["strength_score"] = out["bull_score"].rolling(bg_lookback, min_periods=1).sum()
    out["weakness_score"] = out["bear_score"].rolling(bg_lookback, min_periods=1).sum()

    out["direction"] = compute_simple_direction(out["close"], 3)
    if trend_rule:
        out.loc[out["direction"] == 1, "strength_score"] = out.loc[out["direction"] == 1, "strength_score"] + 5
        out.loc[out["direction"] != 1, "weakness_score"] = out.loc[out["direction"] != 1, "weakness_score"] + 5

    out["net_score"] = out["strength_score"] - out["weakness_score"]
    out["background"] = np.select(
        [out["net_score"] > bg_thresh, out["net_score"] < -bg_thresh],
        ["Strong", "Weak"],
        default="Neutral",
    )

    out["effort_rank_easy"] = classify_easy_rank(out["vol_rank"])
    out["result_rank_easy"] = classify_easy_rank(out["spread_rank"])

    # User-requested ER factor: numeric Effort - Result mismatch
    out["er_factor"] = out["effort_rank_easy"] - out["result_rank_easy"]
    out["er_factor_ma"] = calc_ma(out["er_factor"], ma_length, ma_type)
    out["effort_ma"] = calc_ma(out["effort_rank_easy"], ma_length, ma_type)
    out["result_ma"] = calc_ma(out["result_rank_easy"], ma_length, ma_type)
    out["diff_abs"] = (out["er_factor"]).abs()

    # Optional extras from Pine special signals (not central but useful in export)
    out["vol_less_prev2"] = (out["volume"] < out["volume"].shift(1)) & (out["volume"] < out["volume"].shift(2))
    out["low_vol_vsa"] = (out["effort"] == "Low Effort") | out["vol_less_prev2"]
    out["narrow_spread"] = out["spread_class"].isin(["Narrow", "Ultra Narrow"])
    out["no_demand"] = out["up_bar"] & out["low_vol_vsa"] & out["narrow_spread"] & (out["close_pos"] == "Middle")
    out["no_supply"] = out["down_bar"] & out["low_vol_vsa"] & out["narrow_spread"] & (out["close_pos"] == "Middle")
    out["no_demand_confirmed"] = out["no_demand"].shift(1).fillna(False) & (out["close"] < out["close"].shift(1))
    out["no_supply_confirmed"] = out["no_supply"].shift(1).fillna(False) & (out["close"] > out["close"].shift(1))

    z = zscore_strong_move(out["close"])
    out["strong_move_z"] = z

    return out


# -----------------------------
# Visualization
# -----------------------------
def build_figure(df: pd.DataFrame, out_html: str, title: str) -> None:
    x_num = np.arange(len(df), dtype=float)
    intraday = len(df.index) > 1 and (df.index[1] - df.index[0]) < pd.Timedelta("20H")
    tick_text = [ts.strftime("%Y-%m-%d %H:%M") if intraday else ts.strftime("%Y-%m-%d") for ts in df.index]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.42, 0.33, 0.25],
        subplot_titles=(title, "Effort / Result", "ER Factor & MA"),
    )

    # Row 1: price
    fig.add_trace(
        go.Candlestick(
            x=x_num,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#00c2a0",
            decreasing_line_color="#ff4d5a",
            increasing_fillcolor="#00c2a0",
            decreasing_fillcolor="#ff4d5a",
            name="K线",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Signals on price
    def add_price_marker(mask_col: str, text: str, color: str, symbol: str, y: pd.Series, pos: str) -> None:
        idxs = np.where(df[mask_col].fillna(False).to_numpy())[0]
        if len(idxs) == 0:
            return
        fig.add_trace(
            go.Scatter(
                x=idxs,
                y=y.iloc[idxs],
                mode="markers+text",
                text=[text] * len(idxs),
                textposition=pos,
                marker=dict(symbol=symbol, size=10, color=color),
                name=text,
            ),
            row=1,
            col=1,
        )

    add_price_marker("no_supply_confirmed", "NS", "#00e676", "triangle-up", df["low"] * 0.985, "bottom center")
    add_price_marker("no_demand_confirmed", "ND", "#ff6d6d", "triangle-down", df["high"] * 1.015, "top center")

    # Row 2: effort/result
    fig.add_trace(
        go.Scatter(
            x=x_num,
            y=df["effort_rank_easy"],
            mode="lines+markers",
            line=dict(color="#ff9ef9", width=2),
            marker=dict(size=5, color="#ff9ef9"),
            name="Effort",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_num,
            y=df["result_rank_easy"],
            mode="lines+markers",
            line=dict(color="#e9e2ff", width=2),
            marker=dict(size=5, color="#e9e2ff"),
            name="Result",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_num,
            y=df["effort_ma"],
            mode="lines",
            line=dict(color="#d56fe8", width=1.6, dash="dot"),
            name="Effort MA",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_num,
            y=df["result_ma"],
            mode="lines",
            line=dict(color="#bfc7ff", width=1.6, dash="dot"),
            name="Result MA",
        ),
        row=2,
        col=1,
    )

    # Row 3: ER factor / MA
    bar_colors = np.where(df["er_factor"].fillna(0).to_numpy() >= 0, "#5b9cf6", "#faa1a4")
    fig.add_trace(
        go.Bar(
            x=x_num,
            y=df["er_factor"],
            marker_color=bar_colors,
            name="ER Factor",
            opacity=0.7,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_num,
            y=df["er_factor_ma"],
            mode="lines",
            line=dict(color="#ffd54f", width=2),
            name="ER MA",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#90a4ae", row=3, col=1)

    # Latest summary annotation
    last = df.iloc[-1]
    summary = (
        f"Latest | Effort={int(last['effort_rank_easy']) if pd.notna(last['effort_rank_easy']) else 'NA'} | "
        f"Result={int(last['result_rank_easy']) if pd.notna(last['result_rank_easy']) else 'NA'} | "
        f"ER={last['er_factor']:.2f} | BG={last['background']} | {last['effort_vs_result']}"
    )
    fig.add_annotation(
        x=0.995,
        y=0.995,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        text=summary,
        showarrow=False,
        font=dict(size=12, color="#ffffff"),
        bgcolor="rgba(28,32,44,0.92)",
        bordercolor="#4c566a",
        borderwidth=1,
        align="right",
    )

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
        height=1100,
    )

    for r in [1, 2, 3]:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True, zeroline=False, row=r, col=1)
        fig.update_yaxes(showgrid=True, zeroline=False, row=r, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Rank 1-10", row=2, col=1, range=[0, 10.5])
    fig.update_yaxes(title_text="ER", row=3, col=1)

    fig.write_html(out_html, include_plotlyjs="cdn")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VSA IQ ER Viewer")
    p.add_argument("--symbol", required=True, help="A股代码，如 600547")
    p.add_argument("--freq", default="d", help="d/w/mo/1m/5m/15m/30m/60m")
    p.add_argument("--bars", type=int, default=300, help="展示最近 N 根K线")
    p.add_argument("--fetch-bars", type=int, default=1200, help="实际抓取并用于计算的K线数量，建议 > bars")
    p.add_argument("--out", default="vsa_iq_er_viewer.html", help="输出HTML文件")
    p.add_argument("--csv-input", default="", help="可选：本地OHLCV CSV输入，列需包含 datetime/open/high/low/close/volume")
    p.add_argument("--csv-out", default="", help="可选：输出CSV文件")
    p.add_argument("--ma-length", type=int, default=14, help="ER / Effort / Result 的 MA 长度")
    p.add_argument("--ma-type", default="SMA", choices=["SMA", "EMA", "WMA", "RMA"], help="MA 类型")
    p.add_argument("--bg-lookback", type=int, default=10, help="背景评分回看窗口")
    p.add_argument("--no-trend-rule", action="store_true", help="关闭趋势对背景的加权")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.freq = normalize_freq(args.freq)
    fetch_bars = max(args.fetch_bars, args.bars, 300)

    if args.csv_input:
        raw = pd.read_csv(args.csv_input)
        required = {"datetime", "open", "high", "low", "close", "volume"}
        miss = required - set(raw.columns)
        if miss:
            raise ValueError(f"CSV 缺少列: {sorted(miss)}")
        raw["datetime"] = pd.to_datetime(raw["datetime"]).dt.tz_localize(None)
        raw = raw.sort_values("datetime").set_index("datetime")
        for c in ["open", "high", "low", "close", "volume"]:
            raw[c] = raw[c].astype(float)
        if "amount" not in raw.columns:
            raw["amount"] = np.nan
    else:
        raw = fetch_kline_pytdx(args.symbol, args.freq, fetch_bars)
    merged = compute_vsa_iq(
        raw,
        ma_length=args.ma_length,
        ma_type=args.ma_type,
        bg_lookback=args.bg_lookback,
        trend_rule=not args.no_trend_rule,
    ).tail(args.bars).copy()

    title = f"{args.symbol} [{args.freq}] VSA IQ ER Viewer"
    build_figure(merged, args.out, title)

    if args.csv_out:
        merged.to_csv(args.csv_out, encoding="utf-8-sig")
        print(f"CSV 已生成: {args.csv_out}")
    print(f"HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()
