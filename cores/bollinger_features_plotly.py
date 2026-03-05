# -*- coding: utf-8 -*-
"""
Bollinger Bands 特征可视化（Plotly HTML）

Purpose:
    可视化布林带宽度的 z-score 和股价在布林带中的位置

Inputs:
    - symbol: A 股代码 (如 600547)
    - years: 拉取近 N 年数据
    - bb_win: 布林带窗口
    - bb_k: 布林带倍数
    - z_win: bb_width zscore 窗口

Outputs:
    - HTML 图表文件 (包含 K 线、布林带、bb_width_z_255、bb_pos)
    - 左上角特征表格

How to Run:
    python cores/bollinger_features_plotly.py --symbol 600547 --years 3 --bb_win 20 --bb_k 2 --z_win 255 --out bb_feat.html
    python cores/bollinger_features_plotly.py --symbol 002099 --years 5 --out bb_feat.html

Side Effects:
    - 写文件：输出 HTML 文件到指定路径
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datasource.pytdx_client import connect_pytdx, PERIOD_MAP


def fetch_daily_pytdx(symbol: str, start: str, end: str, *, max_bars: int = 800) -> pd.DataFrame:
    """Fetch daily bars via pytdx."""
    api = connect_pytdx()
    try:
        cat = PERIOD_MAP['d']
        mkt = 1 if symbol.startswith("6") else 0
        page = 0
        size = 700
        frames = []
        while True:
            recs = api.get_security_bars(cat, mkt, symbol, page * size, size)
            if not recs:
                break
            df = pd.DataFrame(recs)
            if df.empty:
                break
            if "datetime" in df.columns:
                df["date"] = pd.to_datetime(df["datetime"])
            elif {"year", "month", "day", "hour", "minute"}.issubset(df.columns):
                df["date"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]].astype(int))
            else:
                raise RuntimeError("pytdx 返回数据缺少时间列")
            if "vol" in df.columns:
                df = df.rename(columns={"vol": "volume"})
            df = df[["date", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])]
            df = df.sort_values("date").set_index("date")
            frames.append(df.astype(float))
            if len(recs) < size:
                break
            page += 1
        if not frames:
            raise RuntimeError("pytdx 无数据")
        df_all = pd.concat(frames).sort_index()
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        return df_all[(df_all.index >= start_dt) & (df_all.index <= end_dt)]
    finally:
        api.disconnect()


# =========================
# Bollinger features
# =========================

@dataclass
class BBcfg:
    bb_win: int = 20
    bb_k: float = 2.0
    z_win: int = 255
    show_panel_table: bool = True
    x_as_category: bool = True


def rolling_zscore(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


def bollinger(df: pd.DataFrame, win: int, k: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = df["close"].rolling(win, min_periods=win).mean()
    sd = df["close"].rolling(win, min_periods=win).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    return mid, upper, lower


def compute_features(df: pd.DataFrame, cfg: BBcfg) -> pd.DataFrame:
    d = df.copy()
    mid, upper, lower = bollinger(d, cfg.bb_win, cfg.bb_k)
    d["bb_mid"] = mid
    d["bb_upper"] = upper
    d["bb_lower"] = lower
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"]
    d["bb_width_z_255"] = rolling_zscore(d["bb_width"], cfg.z_win)
    denom = (d["bb_upper"] - d["bb_lower"]).replace(0.0, np.nan)
    d["bb_pos"] = (d["close"] - d["bb_lower"]) / denom
    d["dir"] = np.where(d["close"] >= d["open"], "UP", "DOWN")
    return d


# =========================
# Plot
# =========================

def add_feature_table(fig: go.Figure, d: pd.DataFrame, cfg: BBcfg) -> None:
    last = d.iloc[-1]
    ts = d.index[-1]

    def fnum(x, fmt="{:,.4f}"):
        return "n/a" if (pd.isna(x) or not np.isfinite(float(x))) else fmt.format(float(x))

    rows = [
        ("time", ts.strftime("%Y-%m-%d %H:%M") if isinstance(ts, pd.Timestamp) else str(ts)),
        ("dir", str(last["dir"])),
        ("close", fnum(last["close"], "{:,.3f}")),
        ("bb_mid", fnum(last["bb_mid"], "{:,.3f}")),
        ("bb_upper", fnum(last["bb_upper"], "{:,.3f}")),
        ("bb_lower", fnum(last["bb_lower"], "{:,.3f}")),
        ("bb_width", fnum(last["bb_width"], "{:,.4f}")),
        ("bb_width_z_255", fnum(last["bb_width_z_255"], "{:,.3f}")),
        ("bb_pos (0=lower,1=upper)", fnum(last["bb_pos"], "{:,.3f}")),
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Feature", "Value"],
                fill_color="rgba(0,0,0,0.75)",
                font=dict(color="white", size=12, family="Consolas, monospace"),
                align=["left", "left"],
            ),
            cells=dict(
                values=[[r[0] for r in rows], [r[1] for r in rows]],
                fill_color="rgba(0,0,0,0.45)",
                font=dict(color="white", size=11, family="Consolas, monospace"),
                align=["left", "left"],
                height=22,
            ),
            domain=dict(x=[0.02, 0.32], y=[0.68, 0.98]),
        ),
        row=None,
        col=None
    )


def build_plot(d: pd.DataFrame, cfg: BBcfg, title: str, out_html: str) -> None:
    if cfg.x_as_category:
        x = d.index.strftime("%Y-%m-%d").tolist()
    else:
        x = d.index

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.62, 0.20, 0.18],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=d["open"], high=d["high"], low=d["low"], close=d["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
            showlegend=False,
        ),
        row=1, col=1
    )

    fig.add_trace(go.Scatter(x=x, y=d["bb_mid"], mode="lines", line=dict(width=1.2), name="bb_mid"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["bb_upper"], mode="lines", line=dict(width=1.0), name="bb_upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["bb_lower"], mode="lines", line=dict(width=1.0), name="bb_lower"), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=x, y=d["bb_width_z_255"], mode="lines", line=dict(width=1.6), name="bb_width_z_255"),
        row=2, col=1
    )
    for y0 in [0, 1, -1, 2, -2]:
        fig.add_hline(y=y0, row=2, col=1, line_width=1, opacity=0.30)

    fig.add_trace(
        go.Scatter(x=x, y=d["bb_width"], mode="lines", line=dict(width=1.0, dash="dot"), name="bb_width"),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=x, y=d["bb_pos"], mode="lines", line=dict(width=1.6), name="bb_pos (0=lower,1=upper)"),
        row=3, col=1
    )
    fig.add_hline(y=0, row=3, col=1, line_width=1, opacity=0.25)
    fig.add_hline(y=1, row=3, col=1, line_width=1, opacity=0.25)

    fig.update_layout(
        title=title,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=980,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )

    if cfg.x_as_category:
        fig.update_xaxes(type="category", showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False)
    else:
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False)

    for r in [1, 2, 3]:
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=r, col=1)

    if cfg.show_panel_table and len(d) > 0:
        add_feature_table(fig, d, cfg)

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_html}")


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)

    ap.add_argument("--symbol", type=str, default="600547", help="A 股代码，如 600547 / 002099")
    ap.add_argument("--years", type=int, default=3, help="拉取近 N 年（按自然日粗估）")
    ap.add_argument("--freq", type=str, default="d", help="频率：默认 d（日线）")
    ap.add_argument("--fqt", type=int, default=1, help="复权：1 常用")

    ap.add_argument("--bars", type=int, default=800, help="最多保留最近 bars 根（防止 HTML 太大）")

    ap.add_argument("--bb_win", type=int, default=20, help="布林带窗口")
    ap.add_argument("--bb_k", type=float, default=2.0, help="布林带倍数 k")
    ap.add_argument("--z_win", type=int, default=255, help="bb_width zscore 窗口（255 交易日）")

    ap.add_argument("--no-table", action="store_true", help="关闭左上角特征表格")
    ap.add_argument("--x-time", action="store_true", help="横轴用时间 (会有非交易日空档)。默认用 category 字符串轴")

    ap.add_argument("--out", type=str, default="bollinger_features.html", help="输出 HTML 文件名")

    args = ap.parse_args()

    end = datetime.now().date()
    start = end - timedelta(days=365 * args.years + 30)

    df = fetch_daily_pytdx(args.symbol, start=str(start), end=str(end), max_bars=args.bars)

    if args.bars and len(df) > args.bars:
        df = df.iloc[-args.bars:]

    cfg = BBcfg(
        bb_win=int(args.bb_win),
        bb_k=float(args.bb_k),
        z_win=int(args.z_win),
        show_panel_table=(not args.no_table),
        x_as_category=(not args.x_time),
    )

    d = compute_features(df, cfg)

    title = f"{args.symbol} Bollinger Features | win={cfg.bb_win}, k={cfg.bb_k}, width_z={cfg.z_win} | freq={args.freq}"
    build_plot(d, cfg, title=title, out_html=args.out)


if __name__ == "__main__":
    main()
