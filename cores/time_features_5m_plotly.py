# -*- coding: utf-8 -*-
"""
5m 时间特征可视化（Plotly HTML）- 修正版

画：
- T1 tod_sin / tod_cos（48根/日）
- T3 bars_to_close
- T4 intraday_vol_z_5m / intraday_range_z_5m（按 bar_idx 跨天分组 z-score）

关键修正：
1) qstock 的 freq 在不同版本写法不一致：自动尝试多个候选，避免“时间频率输入有误”
2) 横轴用字符串类别轴，避免非交易时段的巨大空档
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datasource.pytdx_client import connect_pytdx, PERIOD_MAP


def _category_from_freq(freq: str) -> int:
    f = str(freq).strip().lower()
    if f in ("5", "5m"):
        return 0
    if f in ("1", "1m"):
        return 7
    if f in ("15", "15m"):
        return 1
    if f in ("30", "30m"):
        return 2
    if f in ("60", "60m", "1h"):
        return 3
    raise ValueError(f"不支持的分钟频率: {freq}")


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


def fetch_pytdx_5m(symbol: str, start: str, end: str, freq_hint: str) -> tuple[pd.DataFrame, str]:
    api = connect_pytdx()
    try:
        cat = _category_from_freq(freq_hint)
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
                d["dt"] = pd.to_datetime(d["datetime"])
            elif {"year", "month", "day", "hour", "minute"}.issubset(d.columns):
                d["dt"] = pd.to_datetime(d[["year", "month", "day", "hour", "minute"]].astype(int))
            else:
                raise RuntimeError("pytdx返回数据缺少时间列")
            if "vol" in d.columns:
                d = d.rename(columns={"vol": "volume"})
            d = d.rename(columns={"amount": "amount"})
            d = d[["dt", "open", "high", "low", "close"] + (["volume"] if "volume" in d.columns else [])]
            frames.append(d.sort_values("dt"))
            if len(recs) < size:
                break
            page += 1
        if not frames:
            raise RuntimeError("pytdx无数据")
        all_df = pd.concat(frames).sort_values("dt")
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        all_df = all_df[(all_df["dt"] >= start_dt) & (all_df["dt"] <= end_dt)]
        all_df = all_df.drop_duplicates(subset=["dt"], keep="last").set_index("dt")
        return all_df, freq_hint
    finally:
        api.disconnect()


# =========================
# Feature calc
# =========================

@dataclass
class Cfg:
    show_panel: bool = True
    clip_z: float = 6.0


def _bar_idx_5m(ts: pd.Timestamp) -> int | None:
    t = ts.time()
    # Morning [09:30, 11:30)
    if dtime(9, 30) <= t < dtime(11, 30):
        mins = (t.hour * 60 + t.minute) - (9 * 60 + 30)
        if mins % 5 != 0:
            return None
        idx = mins // 5
        return int(idx) if 0 <= idx <= 23 else None

    # Afternoon [13:00, 15:00)
    if dtime(13, 0) <= t < dtime(15, 0):
        mins = (t.hour * 60 + t.minute) - (13 * 60 + 0)
        if mins % 5 != 0:
            return None
        idx = 24 + mins // 5
        return int(idx) if 24 <= idx <= 47 else None

    return None


def compute_time_features_5m(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["bar_idx"] = [_bar_idx_5m(ts) for ts in d.index]
    d = d.dropna(subset=["bar_idx"])
    d["bar_idx"] = d["bar_idx"].astype(int)

    # T1
    d["tod_sin"] = np.sin(2 * np.pi * d["bar_idx"] / 48.0)
    d["tod_cos"] = np.cos(2 * np.pi * d["bar_idx"] / 48.0)

    # T3
    d["bars_to_close"] = 47 - d["bar_idx"]

    # T4
    d["range"] = (d["high"] - d["low"]).astype(float)
    g = d.groupby("bar_idx", sort=True)

    vol_mu = g["volume"].transform("mean")
    vol_sd = g["volume"].transform(lambda s: s.std(ddof=0))
    d["intraday_vol_z_5m"] = (d["volume"] - vol_mu) / vol_sd.replace(0.0, np.nan)

    rng_mu = g["range"].transform("mean")
    rng_sd = g["range"].transform(lambda s: s.std(ddof=0))
    d["intraday_range_z_5m"] = (d["range"] - rng_mu) / rng_sd.replace(0.0, np.nan)

    return d


# =========================
# Plot (x as string category)
# =========================

def build_plot(d: pd.DataFrame, cfg: Cfg, title: str, out_html: str) -> None:
    # x as string => no gaps
    x = d.index.strftime("%Y-%m-%d %H:%M").tolist()

    z1 = d["intraday_vol_z_5m"].clip(-cfg.clip_z, cfg.clip_z)
    z2 = d["intraday_range_z_5m"].clip(-cfg.clip_z, cfg.clip_z)

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.48, 0.16, 0.16, 0.10, 0.10],
        specs=[[{"type": "xy"}]] * 5,
    )

    # Candles (x is string, open/high/low/close same length)
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=d["open"], high=d["high"], low=d["low"], close=d["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
            showlegend=False
        ),
        row=1, col=1
    )

    # intraday_vol_z_5m
    fig.add_trace(
        go.Scatter(x=x, y=z1, mode="lines", line=dict(width=1.5), name="intraday_vol_z_5m"),
        row=2, col=1
    )
    for y0 in [0, 1, -1, 2, -2]:
        fig.add_hline(y=y0, row=2, col=1, line_width=1, opacity=0.30)

    # intraday_range_z_5m
    fig.add_trace(
        go.Scatter(x=x, y=z2, mode="lines", line=dict(width=1.5), name="intraday_range_z_5m"),
        row=3, col=1
    )
    for y0 in [0, 1, -1, 2, -2]:
        fig.add_hline(y=y0, row=3, col=1, line_width=1, opacity=0.30)

    # bars_to_close
    fig.add_trace(
        go.Scatter(x=x, y=d["bars_to_close"], mode="lines", line=dict(width=1.3), name="bars_to_close"),
        row=4, col=1
    )
    fig.update_yaxes(autorange="reversed", row=4, col=1)

    # tod_sin / tod_cos
    fig.add_trace(
        go.Scatter(x=x, y=d["tod_sin"], mode="lines", line=dict(width=1.2), name="tod_sin"),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=d["tod_cos"], mode="lines", line=dict(width=1.2), name="tod_cos"),
        row=5, col=1
    )
    fig.add_hline(y=0, row=5, col=1, line_width=1, opacity=0.25)

    # Info panel
    if cfg.show_panel and len(d) > 0:
        last = d.iloc[-1]
        panel_lines = [
            "5m 时间特征（最后一根）",
            f"time: {d.index[-1].strftime('%Y-%m-%d %H:%M')}",
            f"bar_idx: {int(last['bar_idx'])}/47",
            f"bars_to_close: {int(last['bars_to_close'])}",
            f"tod_sin: {float(last['tod_sin']): .4f}",
            f"tod_cos: {float(last['tod_cos']): .4f}",
            f"intraday_vol_z_5m: {float(last['intraday_vol_z_5m']): .3f}" if pd.notna(last["intraday_vol_z_5m"]) else "intraday_vol_z_5m: n/a",
            f"intraday_range_z_5m: {float(last['intraday_range_z_5m']): .3f}" if pd.notna(last["intraday_range_z_5m"]) else "intraday_range_z_5m: n/a",
        ]
        fig.add_annotation(
            x=0.012, y=0.985,
            xref="paper", yref="paper",
            text="<br>".join(panel_lines),
            showarrow=False,
            align="left",
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
            font=dict(color="white", size=12, family="Consolas, monospace"),
        )

    # Layout
    fig.update_layout(
        title=title,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=1100,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )

    # X axis: category => no gaps
    fig.update_xaxes(
        type="category",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        rangeslider_visible=False,
        tickangle=0,
    )
    for r in range(1, 6):
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=r, col=1)

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_html}")


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--symbol", type=str, default="600547")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--freq", type=str, default="5", help="优先写 5；脚本也会自动尝试 5min/min5/5m 等")
    ap.add_argument("--fqt", type=int, default=1)
    ap.add_argument("--clipz", type=float, default=6.0)
    ap.add_argument("--no-panel", action="store_true")
    ap.add_argument("--out", type=str, default="time_features_5m.html")
    args = ap.parse_args()

    end = datetime.now().date()
    start = end - timedelta(days=int(args.days))

    df, fq_used = fetch_pytdx_5m(args.symbol, start=str(start), end=str(end), freq_hint=args.freq)

    d = compute_time_features_5m(df)
    if len(d) == 0:
        raise RuntimeError(
            "拿到了分钟数据，但过滤到A股 5m 交易时段后为空。\n"
            "可能原因：时间戳不是本地时区/不是 5 分钟对齐/或数据不含日内分钟。"
        )

    cfg = Cfg(show_panel=(not args.no_panel), clip_z=float(args.clipz))
    title = f"{args.symbol} 5m 时间特征（T1+T3+T4） days={args.days} freq={fq_used}"
    build_plot(d, cfg, title=title, out_html=args.out)


if __name__ == "__main__":
    main()
