# -*- coding: utf-8 -*-
"""
文件名：
    bbawe_pytdx_plotly.py

用途：
    按 TradingView Pine v4 指标
    "Bollinger Awesome Alert R1.1 by JustUncleL"
    用 pytdx 数据源 + Plotly 输出 HTML。

实现内容：
1. Bollinger Bands（可选 SMA / EMA 中轨）
2. Fast EMA（默认 3）
3. Awesome Oscillator 状态判定
4. Buy / Sell 信号箭头
5. 可选 BB 过滤（信号K线收盘需留在带内）
6. 可选 Relative BB Squeeze 过滤
7. 在主图显示：K线、布林带、快线、买卖箭头、Squeeze 着色带
8. 在副图显示：AO 柱状图 + 零轴

依赖环境：
    pip install pandas numpy plotly pytdx

示例：
    python bbawe_pytdx_plotly.py --symbol 600489 --freq 15m --bars 500 --out bbawe.html
    python bbawe_pytdx_plotly.py --symbol 300308 --freq 5m --bars 800 --bb-filter --sqz-filter --out bbawe_5m.html
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot

if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _base_dir not in sys.path:
        sys.path.insert(0, _base_dir)

from datasource.pytdx_client import connect_pytdx, get_kline_data, get_stock_name

GREEN = "#00c853"
RED = "#ff5252"
BLUE = "#3d7eff"
GRAY = "rgba(180,180,180,0.14)"
WHITE = "rgba(255,255,255,0.10)"
BLACK = "#111111"
AO_GREEN = "#66bb6a"
AO_DARK_GREEN = "#1b5e20"
AO_RED = "#ef5350"
AO_DARK_RED = "#8e0000"


@dataclass
class SignalSummary:
    buy_count: int = 0
    sell_count: int = 0


def normalize_freq(freq: str) -> str:
    f = freq.lower().strip()
    if f in {"d", "1d", "day", "daily"}:
        return "d"
    if f in {"w", "1w", "week", "weekly"}:
        return "w"
    if f in {"m", "mo", "month", "monthly"}:
        return "mo"
    if f in {"5", "5m", "5min"}:
        return "5m"
    if f in {"15", "15m", "15min"}:
        return "15m"
    if f in {"30", "30m", "30min"}:
        return "30m"
    if f in {"60", "60m", "60min", "1h"}:
        return "60m"
    raise ValueError(f"Unsupported freq: {freq}")


def fetch_data(symbol: str, freq: str, bars: int) -> Tuple[pd.DataFrame, object]:
    client = connect_pytdx()
    freq = normalize_freq(freq)

    df = get_kline_data(client, symbol, freq, count=bars)
    if df is None or len(df) == 0:
        client.disconnect()
        raise RuntimeError(f"获取数据失败: {symbol} {freq}")

    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"datetime", "date", "time"}:
            rename_map[c] = "datetime"
        elif cl in {"open", "o"}:
            rename_map[c] = "open"
        elif cl in {"high", "h"}:
            rename_map[c] = "high"
        elif cl in {"low", "l"}:
            rename_map[c] = "low"
        elif cl in {"close", "c"}:
            rename_map[c] = "close"
        elif cl in {"vol", "volume"}:
            rename_map[c] = "vol"
        elif cl in {"amount", "amt", "turnover"}:
            rename_map[c] = "amount"

    df = df.rename(columns=rename_map).copy()
    for c in ["vol", "amount"]:
        if c not in df.columns:
            df[c] = 0.0

    need = ["datetime", "open", "high", "low", "close"]
    miss = [x for x in need if x not in df.columns]
    if miss:
        client.disconnect()
        raise RuntimeError(f"缺少字段: {miss}, 当前={list(df.columns)}")

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).set_index("datetime")
    return df[["open", "high", "low", "close", "vol", "amount"]].astype(float), client


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def stdev(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).std(ddof=0)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()


class BBAWEIndicator:
    def __init__(self, df: pd.DataFrame, args: argparse.Namespace):
        self.df = df.copy()
        self.args = args
        self.summary = SignalSummary()

    def compute(self) -> pd.DataFrame:
        df = self.df
        src = df["close"]

        bb_basis = ema(src, self.args.bb_length) if self.args.bb_use_ema else sma(src, self.args.bb_length)
        fast_ma = ema(src, self.args.fast_ma_len)
        dev = stdev(src, self.args.bb_length)
        bb_dev = self.args.bb_mult * dev
        bb_upper = bb_basis + bb_dev
        bb_lower = bb_basis - bb_dev

        hl2 = (df["high"] + df["low"]) / 2.0
        ao_fast = sma(hl2, self.args.ao_fast)
        ao_slow = sma(hl2, self.args.ao_slow)
        ao_raw = ao_fast - ao_slow

        # Pine 原脚本 AO 状态：
        #  1  = AO>=0 且上升
        #  2  = AO>=0 且下降
        # -1  = AO<0 且上升
        # -2  = AO<0 且下降
        ao_prev = ao_raw.shift(1)
        ao_state = np.where(
            ao_raw >= 0,
            np.where(ao_raw > ao_prev, 1, 2),
            np.where(ao_raw > ao_prev, -1, -2),
        )
        ao_state = pd.Series(ao_state, index=df.index, dtype="float64")

        spread = bb_upper - bb_lower
        avgspread = sma(spread, self.args.sqz_length)
        bb_squeeze = spread / avgspread * 100.0
        bb_offset = atr(df, 14) * 0.5
        bb_sqz_upper = bb_upper + bb_offset
        bb_sqz_lower = bb_lower - bb_offset

        cross_up = (fast_ma > bb_basis) & (fast_ma.shift(1) <= bb_basis.shift(1))
        cross_down = (fast_ma < bb_basis) & (fast_ma.shift(1) >= bb_basis.shift(1))

        break_up = (
            cross_up
            & (df["close"] > bb_basis)
            & (ao_state.abs() == 1)
            & ((~self.args.bb_filter) | (df["close"] < bb_upper))
            & ((~self.args.sqz_filter) | (bb_squeeze > self.args.sqz_threshold))
        )
        break_down = (
            cross_down
            & (df["close"] < bb_basis)
            & (ao_state.abs() == 2)
            & ((~self.args.bb_filter) | (df["close"] > bb_lower))
            & ((~self.args.sqz_filter) | (bb_squeeze > self.args.sqz_threshold))
        )

        ao_color = np.where(
            ao_raw >= 0,
            np.where(ao_raw > ao_prev, AO_GREEN, AO_DARK_GREEN),
            np.where(ao_raw > ao_prev, AO_RED, AO_DARK_RED),
        )

        sqz_is_wide = bb_squeeze > self.args.sqz_threshold
        sqz_fill_color = np.where(sqz_is_wide, WHITE, "rgba(61,126,255,0.28)")

        out = df.copy()
        out["bb_basis"] = bb_basis
        out["bb_upper"] = bb_upper
        out["bb_lower"] = bb_lower
        out["fast_ma"] = fast_ma
        out["ao"] = ao_raw
        out["ao_state"] = ao_state
        out["ao_color"] = ao_color
        out["spread"] = spread
        out["avgspread"] = avgspread
        out["bb_squeeze"] = bb_squeeze
        out["bb_sqz_upper"] = bb_sqz_upper
        out["bb_sqz_lower"] = bb_sqz_lower
        out["sqz_fill_color"] = sqz_fill_color
        out["break_up"] = break_up.fillna(False)
        out["break_down"] = break_down.fillna(False)

        self.summary.buy_count = int(out["break_up"].sum())
        self.summary.sell_count = int(out["break_down"].sum())
        self.df = out
        return out

    def figure(self, title: str) -> go.Figure:
        df = self.df
        is_intraday = self.args.freq in {"5m", "15m", "30m", "60m"}
        x = df.index.strftime("%Y-%m-%d %H:%M") if is_intraday else df.index.strftime("%Y-%m-%d")

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.72, 0.28],
            subplot_titles=("价格 / BBAWE", "Awesome Oscillator"),
        )

        fig.add_trace(
            go.Candlestick(
                x=x,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color=GREEN,
                decreasing_line_color=RED,
                increasing_fillcolor=GREEN,
                decreasing_fillcolor=RED,
                name="K线",
            ),
            row=1,
            col=1,
        )

        # 布林带外侧 squeeze 提示区域：上侧
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["bb_upper"],
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="bb_upper_fill_base",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["bb_sqz_upper"],
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(61,126,255,0.25)",
                hoverinfo="skip",
                showlegend=False,
                name="上侧Squeeze区",
            ),
            row=1,
            col=1,
        )

        # 下侧 squeeze 提示区域
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["bb_lower"],
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="bb_lower_fill_base",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["bb_sqz_lower"],
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(61,126,255,0.25)",
                hoverinfo="skip",
                showlegend=False,
                name="下侧Squeeze区",
            ),
            row=1,
            col=1,
        )

        # 中间布林带本体
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["bb_upper"],
                mode="lines",
                line=dict(color=BLUE, width=1),
                name="BB Upper",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["bb_lower"],
                mode="lines",
                line=dict(color=BLUE, width=1),
                fill="tonexty",
                fillcolor=GRAY,
                name="BB Channel",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["bb_basis"],
                mode="lines",
                line=dict(color="red", width=2),
                name="BB Basis",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["fast_ma"],
                mode="lines",
                line=dict(color=BLACK, width=2),
                name=f"EMA{self.args.fast_ma_len}",
            ),
            row=1,
            col=1,
        )

        buy_df = df[df["break_up"]]
        sell_df = df[df["break_down"]]

        if not buy_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_df.index.strftime("%Y-%m-%d %H:%M") if is_intraday else buy_df.index.strftime("%Y-%m-%d"),
                    y=buy_df["low"] * 0.995,
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=12, color=GREEN),
                    text=["Buy"] * len(buy_df),
                    textposition="bottom center",
                    name="Buy",
                ),
                row=1,
                col=1,
            )

        if not sell_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_df.index.strftime("%Y-%m-%d %H:%M") if is_intraday else sell_df.index.strftime("%Y-%m-%d"),
                    y=sell_df["high"] * 1.005,
                    mode="markers+text",
                    marker=dict(symbol="triangle-down", size=12, color=RED),
                    text=["Sell"] * len(sell_df),
                    textposition="top center",
                    name="Sell",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Bar(
                x=x,
                y=df["ao"],
                marker_color=df["ao_color"],
                name="AO",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_width=1, line_color="#aaaaaa", row=2, col=1)

        fig.update_layout(
            title=(
                f"{title}<br>"
                f"Buy={self.summary.buy_count} | Sell={self.summary.sell_count} | "
                f"bb_filter={self.args.bb_filter} | sqz_filter={self.args.sqz_filter}"
            ),
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=90, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.update_xaxes(type="category", showgrid=True)
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="AO", row=2, col=1)
        return fig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bollinger Awesome Alert (BBAWE) - pytdx + plotly")
    p.add_argument("--symbol", required=True, help="股票代码，如 600489 / 300308")
    p.add_argument("--freq", default="15m", help="d/w/mo/5m/15m/30m/60m")
    p.add_argument("--bars", type=int, default=600, help="获取K线数量")
    p.add_argument("--out", default="bbawe.html", help="输出 HTML 文件")

    p.add_argument("--bb-use-ema", action="store_true", help="布林中轨使用 EMA，默认使用 SMA")
    p.add_argument("--bb-filter", action="store_true", help="买卖信号需收盘留在布林带内")
    p.add_argument("--sqz-filter", action="store_true", help="启用 BB relative squeeze 过滤")
    p.add_argument("--bb-length", type=int, default=20)
    p.add_argument("--bb-mult", type=float, default=2.0)
    p.add_argument("--fast-ma-len", type=int, default=3)
    p.add_argument("--ao-fast", type=int, default=5)
    p.add_argument("--ao-slow", type=int, default=34)
    p.add_argument("--sqz-length", type=int, default=100)
    p.add_argument("--sqz-threshold", type=float, default=50.0)
    return p


def main():
    args = build_parser().parse_args()
    args.freq = normalize_freq(args.freq)

    df, client = fetch_data(args.symbol, args.freq, args.bars)

    name = args.symbol
    try:
        stock_name = get_stock_name(client, args.symbol)
        if stock_name:
            name = stock_name
    except Exception:
        pass
    finally:
        try:
            client.disconnect()
        except Exception:
            pass

    engine = BBAWEIndicator(df, args)
    engine.compute()
    fig = engine.figure(f"BBAWE - {name}({args.symbol}) [{args.freq}]")
    plot(fig, filename=args.out, auto_open=False, include_plotlyjs=True)
    print(f"✅ HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()
