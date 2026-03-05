# -*- coding: utf-8 -*-
"""
Volume Z-Score (成交量 ZScore) -> Plotly HTML

Purpose:
    计算成交量 ZScore 并生成 Plotly HTML 图表

Inputs:
    - symbol: A 股代码 (如 600547)
    - years: 拉取近 N 年数据
    - win: ZScore rolling window

Outputs:
    - HTML 图表文件 (包含 K 线、成交量柱、成交量 ZScore)
    - 可选 PNG 图片

How to Run:
    python cores/volume_zscore_plotly.py --symbol 600547 --years 3 --win 20 --out volz.html
    python cores/volume_zscore_plotly.py --symbol 002099 --years 5 --win 255 --out volz.html

Side Effects:
    - 写文件：输出 HTML/PNG 文件到指定路径
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


def fetch_daily_pytdx(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily bars via pytdx and filter by datetime range."""
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
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        return all_df[(all_df.index >= start_dt) & (all_df.index <= end_dt)]
    finally:
        api.disconnect()


# =========================
# Indicator: Volume ZScore
# =========================

@dataclass
class VolZCfg:
    win: int = 20
    clip: float = 6.0
    show_bands: bool = True
    show_panel: bool = True


def volume_zscore(vol: pd.Series, win: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    mu = vol.rolling(win, min_periods=win).mean()
    sd = vol.rolling(win, min_periods=win).std(ddof=0)
    z = (vol - mu) / sd.replace(0.0, np.nan)
    return z, mu, sd


# =========================
# Plotly render
# =========================

def build_plot(df: pd.DataFrame, cfg: VolZCfg, title: str, out_html: str, out_png: str = "") -> None:
    vol = df["volume"].astype(float)
    z, mu, sd = volume_zscore(vol, cfg.win)

    if cfg.clip and cfg.clip > 0:
        z_plot = z.clip(lower=-cfg.clip, upper=cfg.clip)
    else:
        z_plot = z

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.64, 0.18, 0.18],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
            showlegend=False
        ),
        row=1, col=1
    )

    vol_colors = np.where(df["close"] >= df["open"],
                          "rgba(38,166,154,0.6)",
                          "rgba(239,83,80,0.6)")
    fig.add_trace(
        go.Bar(x=df.index, y=vol, marker_color=vol_colors, showlegend=False),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=z_plot, mode="lines", line=dict(width=1.6), showlegend=False),
        row=3, col=1
    )

    if cfg.show_bands:
        for y0 in [0, 1, -1, 2, -2]:
            fig.add_hline(y=y0, row=3, col=1, line_width=1, opacity=0.35)

    if cfg.show_panel:
        last_i = df.index[-1]
        last_z = float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else np.nan
        last_mu = float(mu.iloc[-1]) if pd.notna(mu.iloc[-1]) else np.nan
        last_sd = float(sd.iloc[-1]) if pd.notna(sd.iloc[-1]) else np.nan
        last_vol = float(vol.iloc[-1])

        panel_lines = [
            "成交量 ZScore（最后一根）",
            f"win: {cfg.win}",
            f"z: {last_z:.3f}" if np.isfinite(last_z) else "z: n/a",
        ]
        fig.add_annotation(
            x=0.015, y=0.985,
            xref="paper", yref="paper",
            text="<br>".join(panel_lines),
            showarrow=False,
            align="left",
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
            font=dict(color="white", size=12, family="Consolas, monospace"),
        )

    fig.update_layout(
        title=title,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=980,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=2, col=1, rangemode="tozero")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=3, col=1)

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

    ap.add_argument("--symbol", type=str, default="600547", help="A 股代码，如 600547 / 002099")
    ap.add_argument("--years", type=int, default=3, help="拉取近 N 年（按 250 交易日/年粗估）")
    ap.add_argument("--freq", type=str, default="d", help="频率：d（日线）")
    ap.add_argument("--fqt", type=int, default=1, help="复权：1 常用")
    ap.add_argument("--win", type=int, default=255, help="ZScore rolling window")
    ap.add_argument("--clip", type=float, default=6.0, help="zscore 画图裁剪范围（0 表示不裁剪）")
    ap.add_argument("--no-bands", action="store_true", help="关闭 ±1/±2 参考线")
    ap.add_argument("--no-panel", action="store_true", help="关闭右上角信息面板")

    ap.add_argument("--out", type=str, default="volume_zscore.html", help="输出 HTML 文件名")
    ap.add_argument("--png", type=str, default="", help="可选：导出 PNG")

    args = ap.parse_args()

    end = datetime.now().date()
    start = end - timedelta(days=250 * args.years)

    df = fetch_daily_pytdx(args.symbol, start=str(start), end=str(end))

    cfg = VolZCfg(
        win=int(args.win),
        clip=float(args.clip),
        show_bands=(not args.no_bands),
        show_panel=(not args.no_panel),
    )

    title = f"{args.symbol} Volume ZScore (win={cfg.win}, freq={args.freq})"
    build_plot(df, cfg, title=title, out_html=args.out, out_png=args.png)


if __name__ == "__main__":
    main()
