# -*- coding: utf-8 -*-
"""
Volume Z-Score (成交量 ZScore) -> Plotly HTML (TradingView 风格：柱状 + 分段着色 + 🐋)

Pine 参考（等价逻辑）：
    z = (volume - sma(volume, length)) / stdev(volume, length)
    whale = z > 3
    color = z>2 ? green : z>1.5 ? lime : z<1 ? gray : silver
    plot(z, style=columns)
    plotshape(whale, location=top, labeldown, "🐋")

How to run:
    python volume_zscore_plotly.py --symbol 600547 --years 3 --win 14 --out volz_tv.html
    python volume_zscore_plotly.py --symbol 002099 --years 5 --win 255 --clip 6 --out volz_tv.html

Notes on exact match vs TradingView:
    - Pandas rolling std uses ddof; here we use ddof=0 (population std) which is commonly needed to对齐TV的σ类指标。
    - 数据源/复权口径不同也会导致轻微偏差。
"""

from __future__ import annotations

import argparse
import sys
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 依赖你本地工程里的 pytdx 封装（保持与你原脚本一致）
from datasource.pytdx_client import connect_pytdx, PERIOD_MAP


def fetch_daily_pytdx(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily bars via pytdx and filter by datetime range."""
    api = connect_pytdx()
    try:
        cat = PERIOD_MAP["d"]
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
                df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
                df = df.set_index("datetime")
            elif {"year", "month", "day", "hour", "minute"}.issubset(df.columns):
                df["datetime"] = pd.to_datetime(
                    df[["year", "month", "day", "hour", "minute"]].astype(int)
                )
                df = df.set_index("datetime")

            if "vol" in df.columns and "volume" not in df.columns:
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
        out = all_df[(all_df.index >= start_dt) & (all_df.index <= end_dt)].copy()

        need_cols = ["open", "high", "low", "close", "volume"]
        miss = [c for c in need_cols if c not in out.columns]
        if miss:
            raise RuntimeError(f"pytdx 数据缺字段：{miss}")

        return out
    finally:
        api.disconnect()


# =========================
# Indicator: Volume ZScore
# =========================

@dataclass
class VolZCfg:
    win: int = 14
    clip: float = 0.0          # 0 表示不裁剪
    show_bands: bool = True    # 参考线（0/±1/±2），TV脚本没有，但保留为可选
    show_panel: bool = True    # 右上角信息面板
    whale_th: float = 3.0      # 🐋 阈值（Pine: z>3）


def volume_zscore(vol: pd.Series, win: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    z = (vol - SMA(vol, win)) / STDEV(vol, win)
    - SMA: rolling mean
    - STDEV: rolling std, ddof=0 (population std) for better TV alignment
    """
    mu = vol.rolling(win, min_periods=win).mean()
    sd = vol.rolling(win, min_periods=win).std(ddof=0)
    z = (vol - mu) / sd.replace(0.0, np.nan)
    return z, mu, sd


def tv_color_bar(zv: float) -> str:
    """TradingView 分段颜色（等价 Pine color_bar）"""
    if not np.isfinite(zv):
        return "rgba(160,160,160,0.35)"
    if zv > 2:
        return "green"
    if zv > 1.5:
        return "lime"
    if zv < 1:
        return "gray"
    return "silver"


# =========================
# Plotly render
# =========================

def build_plot(df: pd.DataFrame, cfg: VolZCfg, title: str, out_html: str, out_png: str = "") -> None:
    df = df.sort_index().copy()
    vol = df["volume"].astype(float)
    
    # 将日期索引转换为字符串列表，避免显示非交易时间段
    x_labels = df.index.strftime("%Y-%m-%d").tolist()

    z_raw, mu, sd = volume_zscore(vol, cfg.win)

    # 画图裁剪只影响视觉，不影响🐋/着色判断（判断用 z_raw）
    if cfg.clip and cfg.clip > 0:
        z_plot = z_raw.clip(lower=-cfg.clip, upper=cfg.clip)
    else:
        z_plot = z_raw

    whale = (z_raw > cfg.whale_th) & pd.notna(z_raw)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.64, 0.18, 0.18],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
    )

    # --- 1) K线 ---
    fig.add_trace(
        go.Candlestick(
            x=x_labels,
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
            showlegend=False,
            xaxis="x1",
        ),
        row=1, col=1
    )

    # --- 2) 成交量柱（保持你原脚本：按涨跌着色） ---
    vol_colors = np.where(
        df["close"] >= df["open"],
        "rgba(38,166,154,0.6)",
        "rgba(239,83,80,0.6)",
    )
    fig.add_trace(
        go.Bar(x=x_labels, y=vol, marker_color=vol_colors, showlegend=False,
               hovertemplate="Date=%{x}<br>Vol=%{y:.0f}<extra></extra>"),
        row=2, col=1
    )

    # --- 3) ZScore：TV 风格 columns + 分段着色 ---
    z_colors = [tv_color_bar(v) for v in z_raw.values]
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=z_plot,
            marker_color=z_colors,
            showlegend=False,
            hovertemplate="Date=%{x}<br>Z=%{y:.3f}<extra></extra>",
        ),
        row=3, col=1
    )

    # 🐋 标记：模仿 location.top + labeldown
    if whale.any():
        # 标记放在 panel 的顶部附近
        if cfg.clip and cfg.clip > 0:
            y_mark = float(cfg.clip)
        else:
            y_max = float(np.nanmax(z_plot.values)) if np.any(np.isfinite(z_plot.values)) else 0.0
            y_mark = y_max

        whale_labels = [x_labels[i] for i in range(len(whale)) if whale.iloc[i]]
        fig.add_trace(
            go.Scatter(
                x=whale_labels,
                y=np.full(int(whale.sum()), y_mark),
                mode="text",
                text=["🐋"] * int(whale.sum()),
                textposition="top center",
                showlegend=False,
                hovertemplate="Small Whale 🐋<br>Date=%{x}<br>Z>3<extra></extra>",
            ),
            row=3, col=1
        )

    # 参考线（可选）
    if cfg.show_bands:
        for y0 in [0, 1, -1, 2, -2, 3]:
            fig.add_hline(y=y0, row=3, col=1, line_width=1, opacity=0.35)

    # 信息面板（可选）
    if cfg.show_panel:
        last_z = float(z_raw.iloc[-1]) if pd.notna(z_raw.iloc[-1]) else np.nan
        last_mu = float(mu.iloc[-1]) if pd.notna(mu.iloc[-1]) else np.nan
        last_sd = float(sd.iloc[-1]) if pd.notna(sd.iloc[-1]) else np.nan
        last_vol = float(vol.iloc[-1])

        panel_lines = [
            "Volume Z-Score（最后一根）",
            f"win: {cfg.win}",
            f"vol: {last_vol:.0f}",
            f"sma: {last_mu:.1f}" if np.isfinite(last_mu) else "sma: n/a",
            f"std: {last_sd:.1f}" if np.isfinite(last_sd) else "std: n/a",
            f"z: {last_z:.3f}" if np.isfinite(last_z) else "z: n/a",
            f"🐋: {'YES' if (np.isfinite(last_z) and last_z > cfg.whale_th) else 'NO'}",
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

    # 统一外观（深色 TV 风）
    fig.update_layout(
        title=title,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=980,
        margin=dict(l=40, r=40, t=60, b=40),
        bargap=0.0,
    )
    
    # 强制所有 x 轴为字符串类型（category）
    fig.update_xaxes(
        showgrid=True, 
        gridcolor="rgba(255,255,255,0.06)", 
        rangeslider_visible=False,
        type="category",  # 强制为字符串类型
    )
    
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

def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)

    ap.add_argument("--symbol", type=str, default="600547", help="A 股代码，如 600547 / 002099")
    ap.add_argument("--years", type=int, default=3, help="拉取近 N 年（按 250 交易日/年粗估）")
    ap.add_argument("--freq", type=str, default="d", help="频率：d（日线）")
    ap.add_argument("--fqt", type=int, default=1, help="复权：1 常用（此脚本内未展开，保留参数位）")

    # Pine 默认 length=14；这里默认也设为 14，确保你不传参数就更接近 TV
    ap.add_argument("--win", type=int, default=14, help="ZScore rolling window（Pine 默认 14）")

    ap.add_argument("--clip", type=float, default=0.0, help="zscore 画图裁剪范围（0 表示不裁剪）")
    ap.add_argument("--no-bands", action="store_true", help="关闭 0/±1/±2/3 参考线")
    ap.add_argument("--no-panel", action="store_true", help="关闭右上角信息面板")
    ap.add_argument("--whale", type=float, default=3.0, help="🐋 阈值（Pine: 3）")

    ap.add_argument("--out", type=str, default="volume_zscore_tv.html", help="输出 HTML 文件名")
    ap.add_argument("--png", type=str, default="", help="可选：导出 PNG（需要 kaleido）")

    args = ap.parse_args()

    end = datetime.now().date()
    start = end - timedelta(days=250 * args.years)

    df = fetch_daily_pytdx(args.symbol, start=str(start), end=str(end))

    cfg = VolZCfg(
        win=int(args.win),
        clip=float(args.clip),
        show_bands=(not args.no_bands),
        show_panel=(not args.no_panel),
        whale_th=float(args.whale),
    )

    title = f"{args.symbol} Volume Z-Score (win={cfg.win}, freq={args.freq})"
    build_plot(df, cfg, title=title, out_html=args.out, out_png=args.png)


if __name__ == "__main__":
    main()
