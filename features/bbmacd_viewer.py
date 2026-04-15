# -*- coding: utf-8 -*-
"""
Bollinguer sobre Macd（MACD 上的布林带）可视化脚本（独立可运行）

Purpose
- 直接通过 pytdx 拉取 A 股 K 线（不依赖项目内 datasource 模块）
- 按用户提供的 Pine 脚本复刻：
  * m_rapida = ema(close, rapida)
  * m_lenta = ema(close, lenta)
  * BBMacd = m_rapida - m_lenta
  * Avg = ema(BBMacd, 9)
  * SDev = stdev(BBMacd, 9)
  * banda_supe = Avg + stdv * SDev
  * banda_inf = Avg - stdv * SDev
  * Compra = crossover(BBMacd, banda_supe)
  * Venta = crossunder(BBMacd, banda_inf)
- 输出交互式 HTML：主图 K 线，副图为 BSM 指标，并标注买卖信号
- 可选输出 CSV，便于后续回测/研究

How to Run
    python bbmacd_viewer.py --symbol 600547 --freq d --bars 300 \
        --out bbmacd_600547.html --csv-out bbmacd_600547.csv

    python bbmacd_viewer.py --symbol 300750 --freq 60m --bars 500 \
        --rapida 8 --lenta 26 --stdv 0.8 --out bbmacd_300750_60m.html

Dependencies
    pip install pandas numpy plotly pytdx
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from pytdx.hq import TdxHq_API
    from pytdx.params import TDXParams
except Exception as exc:  # pragma: no cover
    raise RuntimeError("请先安装 pytdx: pip install pytdx") from exc


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


def connect_pytdx() -> TdxHq_API:
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


def cross_over(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_under(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def compute_bbmacd(
    df: pd.DataFrame,
    rapida: int = 8,
    lenta: int = 26,
    stdv: float = 0.8,
    signal_len: int = 9,
    width_z_window: int = 60,
) -> pd.DataFrame:
    out = df.copy()
    out["m_rapida"] = pine_ema(out["close"], rapida)
    out["m_lenta"] = pine_ema(out["close"], lenta)
    out["bbmacd"] = out["m_rapida"] - out["m_lenta"]
    out["avg"] = pine_ema(out["bbmacd"], signal_len)
    out["sdev"] = out["bbmacd"].rolling(signal_len, min_periods=signal_len).std(ddof=1)
    out["banda_supe"] = out["avg"] + stdv * out["sdev"]
    out["banda_inf"] = out["avg"] - stdv * out["sdev"]
    out["bb_width"] = out["banda_supe"] - out["banda_inf"]
    out["bb_width_mean"] = out["bb_width"].rolling(width_z_window, min_periods=width_z_window).mean()
    out["bb_width_std"] = out["bb_width"].rolling(width_z_window, min_periods=width_z_window).std(ddof=1)
    out["bb_width_zscore"] = (out["bb_width"] - out["bb_width_mean"]) / out["bb_width_std"].replace(0.0, np.nan)

    out["compra"] = cross_over(out["bbmacd"], out["banda_supe"])
    out["venta"] = cross_under(out["bbmacd"], out["banda_inf"])
    out["cross_up_lower"] = cross_over(out["bbmacd"], out["banda_inf"])
    out["cross_down_upper"] = cross_under(out["bbmacd"], out["banda_supe"])

    # Pine 颜色逻辑
    out["state"] = np.select(
        [out["bbmacd"] < out["banda_inf"], out["bbmacd"] > out["banda_supe"]],
        [-1, 1],
        default=0,
    )
    return out


def build_figure(df: pd.DataFrame, out_html: str, title: str) -> None:
    x_num = np.arange(len(df), dtype=float)
    intraday = len(df.index) > 1 and (df.index[1] - df.index[0]) < pd.Timedelta("20H")
    tick_text = [ts.strftime("%Y-%m-%d %H:%M") if intraday else ts.strftime("%Y-%m-%d") for ts in df.index]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.42, 0.38, 0.20],
        subplot_titles=(title, "BSM: Bollinguer sobre Macd", "Bollinger 带宽 Z-Score"),
    )

    # 主图 K 线
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

    buy_idx = df.index[df["compra"]].tolist()
    sell_idx = df.index[df["venta"]].tolist()
    if buy_idx:
        buy_pos = [df.index.get_loc(i) for i in buy_idx]
        buy_y = df.loc[buy_idx, "low"] * 0.985
        fig.add_trace(
            go.Scatter(
                x=buy_pos,
                y=buy_y,
                mode="markers+text",
                text=["买"] * len(buy_pos),
                textposition="bottom center",
                marker=dict(symbol="triangle-up", size=10, color="#00e676"),
                name="Compra",
            ),
            row=1,
            col=1,
        )
    if sell_idx:
        sell_pos = [df.index.get_loc(i) for i in sell_idx]
        sell_y = df.loc[sell_idx, "high"] * 1.015
        fig.add_trace(
            go.Scatter(
                x=sell_pos,
                y=sell_y,
                mode="markers+text",
                text=["卖"] * len(sell_pos),
                textposition="top center",
                marker=dict(symbol="triangle-down", size=10, color="#ff4d5a"),
                name="Venta",
            ),
            row=1,
            col=1,
        )

    # 副图：先画带区
    fig.add_trace(
        go.Scatter(
            x=x_num,
            y=df["banda_supe"],
            mode="lines",
            line=dict(color="rgba(0,191,255,0.9)", width=1),
            name="上轨",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_num,
            y=df["banda_inf"],
            mode="lines",
            line=dict(color="rgba(0,191,255,0.9)", width=1),
            fill="tonexty",
            fillcolor="rgba(0,191,255,0.16)",
            name="下轨",
        ),
        row=2,
        col=1,
    )

    # 分段画 BBMacd，模拟 Pine 颜色切换
    state = df["state"].fillna(0).astype(int)
    runs = []
    start = 0
    for i in range(1, len(df)):
        if state.iloc[i] != state.iloc[i - 1]:
            runs.append((start, i - 1, int(state.iloc[i - 1])))
            start = i - 1
    if len(df) > 0:
        runs.append((start, len(df) - 1, int(state.iloc[-1])))

    color_map = {1: "#008000", -1: "#FF0000", 0: "#1e88e5"}
    name_map = {1: "BBMacd>上轨", -1: "BBMacd<下轨", 0: "BBMacd带内"}
    shown = set()
    for s, e, st in runs:
        xs = x_num[s:e + 1]
        ys = df["bbmacd"].iloc[s:e + 1]
        fill = None
        fillcolor = None
        if st == 1:
            base = df["banda_supe"].iloc[s:e + 1]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=base,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fill = "tonexty"
            fillcolor = "rgba(0,128,0,0.25)"
        elif st == -1:
            base = df["banda_inf"].iloc[s:e + 1]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=base,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fill = "tonexty"
            fillcolor = "rgba(255,0,0,0.22)"

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color_map[st], width=2),
                fill=fill,
                fillcolor=fillcolor,
                name=name_map[st],
                showlegend=name_map[st] not in shown,
            ),
            row=2,
            col=1,
        )
        shown.add(name_map[st])

    # 副图信号：用箭头+文字直接标记
    def add_signal_markers(mask_col: str, text_label: str, color: str, arrow_symbol: str, y_series: pd.Series, text_position: str) -> None:
        idxs = np.where(df[mask_col].fillna(False).to_numpy())[0]
        if len(idxs) == 0:
            return
        fig.add_trace(
            go.Scatter(
                x=idxs,
                y=y_series.iloc[idxs],
                mode="markers+text",
                text=[text_label] * len(idxs),
                textposition=text_position,
                marker=dict(symbol=arrow_symbol, size=11, color=color, line=dict(width=0.5, color=color)),
                name=text_label,
            ),
            row=2,
            col=1,
        )

    add_signal_markers(
        mask_col="compra",
        text_label="上穿上轨",
        color="#00e676",
        arrow_symbol="triangle-up",
        y_series=df["bbmacd"],
        text_position="top center",
    )
    add_signal_markers(
        mask_col="venta",
        text_label="下穿下轨",
        color="#ff4d5a",
        arrow_symbol="triangle-down",
        y_series=df["bbmacd"],
        text_position="bottom center",
    )
    add_signal_markers(
        mask_col="cross_up_lower",
        text_label="上穿下轨",
        color="#ffd54f",
        arrow_symbol="arrow-up",
        y_series=df["banda_inf"],
        text_position="bottom center",
    )
    add_signal_markers(
        mask_col="cross_down_upper",
        text_label="下穿上轨",
        color="#ab47bc",
        arrow_symbol="arrow-down",
        y_series=df["banda_supe"],
        text_position="top center",
    )

    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#90a4ae", row=2, col=1)

    # 第三副图：布林带宽度 z-score
    fig.add_trace(
        go.Bar(
            x=x_num,
            y=df["bb_width_zscore"],
            name="带宽Z分数",
            marker_color=np.where(df["bb_width_zscore"].fillna(0).to_numpy() >= 0, "#26a69a", "#ef5350"),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_num,
            y=df["bb_width_zscore"],
            mode="lines",
            line=dict(color="#90caf9", width=1.5),
            name="带宽Z分数",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#90a4ae", row=3, col=1)
    fig.add_hline(y=1.0, line_width=1, line_dash="dash", line_color="#26a69a", row=3, col=1)
    fig.add_hline(y=2.0, line_width=1, line_dash="dash", line_color="#2e7d32", row=3, col=1)
    fig.add_hline(y=-1.0, line_width=1, line_dash="dash", line_color="#ef5350", row=3, col=1)
    fig.add_hline(y=-2.0, line_width=1, line_dash="dash", line_color="#b71c1c", row=3, col=1)

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
        height=1200,
    )
    for r in [1, 2, 3]:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True, zeroline=False, row=r, col=1)
        fig.update_yaxes(showgrid=True, zeroline=False, row=r, col=1)

    fig.write_html(out_html, include_plotlyjs="cdn")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bollinguer sobre Macd 可视化脚本")
    p.add_argument("--symbol", required=True, help="A股代码，如 600547")
    p.add_argument("--freq", default="d", help="d/w/mo/1m/5m/15m/30m/60m")
    p.add_argument("--bars", type=int, default=300, help="展示最近 N 根K线")
    p.add_argument("--fetch-bars", type=int, default=1200, help="实际抓取并用于计算的K线数量，建议 > bars")
    p.add_argument("--out", default="bbmacd_viewer.html", help="输出HTML文件")
    p.add_argument("--csv-out", default="", help="可选：输出CSV文件")
    p.add_argument("--rapida", type=int, default=8, help="Pine: Media Rapida")
    p.add_argument("--lenta", type=int, default=26, help="Pine: Media Lenta")
    p.add_argument("--stdv", type=float, default=0.8, help="Pine: Stdv")
    p.add_argument("--signal-len", type=int, default=9, help="Avg / SDev 窗口，Pine 固定为 9")
    p.add_argument("--width-z-window", type=int, default=60, help="布林带宽度 zscore 回看窗口")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.freq = normalize_freq(args.freq)
    fetch_bars = max(args.fetch_bars, args.bars, 300)
    raw = fetch_kline_pytdx(args.symbol, args.freq, fetch_bars)
    merged = compute_bbmacd(
        raw,
        rapida=args.rapida,
        lenta=args.lenta,
        stdv=args.stdv,
        signal_len=args.signal_len,
        width_z_window=args.width_z_window,
    ).tail(args.bars).copy()

    title = f"{args.symbol} [{args.freq}] Bollinguer sobre Macd"
    build_figure(merged, args.out, title)

    if args.csv_out:
        merged.to_csv(args.csv_out, encoding="utf-8-sig")
        print(f"CSV 已生成: {args.csv_out}")
    print(f"HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()
