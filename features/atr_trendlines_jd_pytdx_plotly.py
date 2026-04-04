# -*- coding: utf-8 -*-
"""
文件名：
    atr_trendlines_jd_with_factors.py

用途：
    在 ATR Trendlines - JD Python 版框架中，加入逐 bar 计算的因子，
    并在副图中显示（方案1（前两个距离因子改为最近3条中最匹配）精简版）：
        1) res_dist_atr
        2) sup_dist_atr
        3) res_slope_pct252
        4) sup_slope_pct252
        5) hold_above_res_ratio
        6) hold_below_sup_ratio

说明：
    - 主图趋势线逻辑保持 Pine 复刻思路不变。
    - 因子严格逐 bar 计算：每根 bar 仅使用当时已经确认并存在的最新 high/low trendline。
    - 副图为单一 panel，绘制 6 个因子序列。

依赖：
    pip install pandas numpy plotly pytdx
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _base_dir not in sys.path:
        sys.path.insert(0, _base_dir)

from datasource.pytdx_client import connect_pytdx, get_kline_data, get_stock_name

GREEN = "#00FF00"
FUCHSIA = "#FF00FF"
TEAL = "#008080"


@dataclass
class TrendlineRecord:
    kind: str                 # "high" or "low"
    pivot_bar_index: int      # bar_index - rightbars
    confirm_bar_index: int    # current bar_index at line.new()
    pivot_price: float        # high_point / low_point
    slope_per_bar: float      # absolute slope per bar
    atr_at_creation: float    # current atr(len/2) at line creation

    def y_at(self, x_bar: float) -> float:
        dist = x_bar - self.pivot_bar_index
        if self.kind == "high":
            return self.pivot_price - self.slope_per_bar * dist
        return self.pivot_price + self.slope_per_bar * dist


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


def time_to_str(ts: pd.Timestamp, is_intraday: bool) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if is_intraday else ts.strftime("%Y-%m-%d")


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def pine_rma(src: pd.Series, length: int) -> pd.Series:
    arr = src.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if length <= 0 or len(arr) == 0:
        return pd.Series(out, index=src.index)
    if len(arr) < length:
        return pd.Series(out, index=src.index)

    init = np.nanmean(arr[:length])
    out[length - 1] = init
    alpha = 1.0 / length
    prev = init
    for i in range(length, len(arr)):
        prev = alpha * arr[i] + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=src.index)


def atr_pine(df: pd.DataFrame, length: int) -> pd.Series:
    return pine_rma(true_range(df), length)


def fixnan_pine(values: List[float]) -> List[float]:
    out: List[float] = []
    last = math.nan
    for v in values:
        if pd.isna(v):
            out.append(last)
        else:
            last = float(v)
            out.append(last)
    return out


def pivothigh_like_pine(src: pd.Series, leftbars: int, rightbars: int) -> pd.Series:
    n = len(src)
    arr = src.to_numpy(dtype=float)
    out = np.full(n, np.nan, dtype=float)
    for i in range(leftbars, n - rightbars):
        p = arr[i]
        if np.isnan(p):
            continue
        left = arr[i - leftbars:i]
        right = arr[i + 1:i + rightbars + 1]
        if len(left) != leftbars or len(right) != rightbars:
            continue
        if np.all(p >= left) and np.all(p > right):
            out[i + rightbars] = p
    return pd.Series(out, index=src.index)


def pivotlow_like_pine(src: pd.Series, leftbars: int, rightbars: int) -> pd.Series:
    n = len(src)
    arr = src.to_numpy(dtype=float)
    out = np.full(n, np.nan, dtype=float)
    for i in range(leftbars, n - rightbars):
        p = arr[i]
        if np.isnan(p):
            continue
        left = arr[i - leftbars:i]
        right = arr[i + 1:i + rightbars + 1]
        if len(left) != leftbars or len(right) != rightbars:
            continue
        if np.all(p <= left) and np.all(p < right):
            out[i + rightbars] = p
    return pd.Series(out, index=src.index)


def percentile_rank_last(window_vals: List[float], current_val: float) -> float:
    """返回当前值在过去窗口中的经验分位数 [0,1]，逐 bar 可用。"""
    valid = [float(v) for v in window_vals if pd.notna(v)]
    if pd.isna(current_val) or len(valid) == 0:
        return math.nan
    arr = np.asarray(valid, dtype=float)
    return float(np.mean(arr <= float(current_val)))


class ATRTrendlinesJDWithFactors:
    def __init__(self, df: pd.DataFrame, args: argparse.Namespace):
        self.df = df.copy()
        self.args = args
        self.len = int(args.len)
        self.leftbars = self.len
        self.rightbars = max(1, self.len // 2)
        self.atr_len = max(1, self.len // 2)
        self.atr_percent = float(args.atr_percent) / 100.0
        self.is_intraday = args.freq in {"5m", "15m", "30m", "60m"}
        self.times = list(self.df.index)
        self.x_numeric = np.arange(len(self.df), dtype=float)
        self.tick_text = [time_to_str(t, self.is_intraday) for t in self.times]
        self.right_pad_bars = max(60, self.rightbars * 4)
        self.right_x = float(len(self.df) - 1 + self.right_pad_bars)

        self.df["atr_half"] = atr_pine(self.df, self.atr_len)

        high_src = self.df["high"] if args.wicks else self.df[["open", "close"]].max(axis=1)
        low_src = self.df["low"] if args.wicks else self.df[["open", "close"]].min(axis=1)

        raw_ph = pivothigh_like_pine(high_src, self.leftbars, self.rightbars)
        raw_pl = pivotlow_like_pine(low_src, self.leftbars, self.rightbars)

        self.df["high_point"] = fixnan_pine(raw_ph.tolist())
        self.df["low_point"] = fixnan_pine(raw_pl.tolist())

        self.line_color_high = TEAL if args.do_mono else GREEN
        self.line_color_low = TEAL if args.do_mono else FUCHSIA
        self.trendlines: List[TrendlineRecord] = []

    def angle_to_slope_percent(self, atr_value: float) -> float:
        if pd.isna(atr_value):
            return math.nan
        return (self.rightbars / 100.0) * float(atr_value)

    def run(self):
        hp = self.df["high_point"].tolist()
        lp = self.df["low_point"].tolist()
        atr_half = self.df["atr_half"].tolist()
        closes = self.df["close"].tolist()
        highs = self.df["high"].tolist()
        lows = self.df["low"].tolist()

        active_res: Optional[TrendlineRecord] = None
        active_sup: Optional[TrendlineRecord] = None

        res_slope_raw_hist: List[float] = []
        sup_slope_raw_hist: List[float] = []

        res_above_count = 0
        sup_below_count = 0

        # preallocate factor columns
        factor_cols = [
            "res_dist_atr", "sup_dist_atr",
            "res_close_break_strength", "sup_close_break_strength",
            "res_slope_pct252", "sup_slope_pct252",
            "hold_above_res_ratio", "hold_below_sup_ratio",
            "res_line_value", "sup_line_value",
        ]
        for c in factor_cols:
            self.df[c] = np.nan

        for i in range(len(self.df)):
            atr_i = atr_half[i]
            slope_high = self.atr_percent * self.angle_to_slope_percent(atr_i)
            slope_low = self.atr_percent * self.angle_to_slope_percent(atr_i)

            # create new lines first: in Pine, line is created on current confirmation bar and is available from this bar onward
            if i > 0 and pd.notna(hp[i]) and pd.notna(hp[i - 1]) and float(hp[i]) != float(hp[i - 1]) and pd.notna(slope_high):
                pivot_bar_index = i - self.rightbars
                if pivot_bar_index >= 0:
                    rec = TrendlineRecord(
                        kind="high",
                        pivot_bar_index=pivot_bar_index,
                        confirm_bar_index=i,
                        pivot_price=float(hp[i]),
                        slope_per_bar=float(slope_high),
                        atr_at_creation=float(atr_i),
                    )
                    self.trendlines.append(rec)
                    active_res = rec
                    res_slope_raw_hist.append(abs(rec.slope_per_bar))

            if i > 0 and pd.notna(lp[i]) and pd.notna(lp[i - 1]) and float(lp[i]) != float(lp[i - 1]) and pd.notna(slope_low):
                pivot_bar_index = i - self.rightbars
                if pivot_bar_index >= 0:
                    rec = TrendlineRecord(
                        kind="low",
                        pivot_bar_index=pivot_bar_index,
                        confirm_bar_index=i,
                        pivot_price=float(lp[i]),
                        slope_per_bar=float(slope_low),
                        atr_at_creation=float(atr_i),
                    )
                    self.trendlines.append(rec)
                    active_sup = rec
                    sup_slope_raw_hist.append(abs(rec.slope_per_bar))

            close_i = closes[i]
            high_i = highs[i]
            low_i = lows[i]

            recent_res = [tl for tl in self.trendlines if tl.kind == "high" and tl.confirm_bar_index <= i][-3:]
            recent_sup = [tl for tl in self.trendlines if tl.kind == "low" and tl.confirm_bar_index <= i][-3:]

            best_res = None
            if recent_res:
                res_candidates_above = [(tl.y_at(i) - close_i, tl) for tl in recent_res if tl.y_at(i) >= close_i]
                if res_candidates_above:
                    _, best_res = min(res_candidates_above, key=lambda x: x[0])
                else:
                    best_res = min(recent_res, key=lambda tl: abs(tl.y_at(i) - close_i))

            best_sup = None
            if recent_sup:
                sup_candidates_below = [(close_i - tl.y_at(i), tl) for tl in recent_sup if tl.y_at(i) <= close_i]
                if sup_candidates_below:
                    _, best_sup = min(sup_candidates_below, key=lambda x: x[0])
                else:
                    best_sup = min(recent_sup, key=lambda tl: abs(tl.y_at(i) - close_i))

            # resistance line factors
            if active_res is not None and pd.notna(atr_i) and atr_i != 0:
                active_res_line = active_res.y_at(i)
                self.df.iat[i, self.df.columns.get_loc("res_line_value")] = active_res_line
                if best_res is not None:
                    res_dist_atr = (close_i - best_res.y_at(i)) / atr_i
                    self.df.iat[i, self.df.columns.get_loc("res_dist_atr")] = res_dist_atr
                window = res_slope_raw_hist[-252:]
                res_slope_pct252 = percentile_rank_last(window, abs(active_res.slope_per_bar))
                if close_i > active_res_line:
                    res_above_count += 1
                else:
                    res_above_count = 0
                res_age = i - active_res.confirm_bar_index + 1
                hold_above_res_ratio = res_above_count / max(1, res_age)

                self.df.iat[i, self.df.columns.get_loc("res_slope_pct252")] = res_slope_pct252
                self.df.iat[i, self.df.columns.get_loc("hold_above_res_ratio")] = hold_above_res_ratio
            else:
                res_above_count = 0

            # support line factors
            if active_sup is not None and pd.notna(atr_i) and atr_i != 0:
                active_sup_line = active_sup.y_at(i)
                self.df.iat[i, self.df.columns.get_loc("sup_line_value")] = active_sup_line
                if best_sup is not None:
                    sup_dist_atr = (close_i - best_sup.y_at(i)) / atr_i
                    self.df.iat[i, self.df.columns.get_loc("sup_dist_atr")] = sup_dist_atr
                window = sup_slope_raw_hist[-252:]
                sup_slope_pct252 = percentile_rank_last(window, abs(active_sup.slope_per_bar))
                if close_i < active_sup_line:
                    sup_below_count += 1
                else:
                    sup_below_count = 0
                sup_age = i - active_sup.confirm_bar_index + 1
                hold_below_sup_ratio = sup_below_count / max(1, sup_age)

                self.df.iat[i, self.df.columns.get_loc("sup_slope_pct252")] = sup_slope_pct252
                self.df.iat[i, self.df.columns.get_loc("hold_below_sup_ratio")] = hold_below_sup_ratio
            else:
                sup_below_count = 0

    def build_figure(self, title: str) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.68, 0.32],
            subplot_titles=(title, "Factors"),
        )

        customdata = np.array(self.tick_text, dtype=object).reshape(-1, 1)
        fig.add_trace(
            go.Candlestick(
                x=self.x_numeric,
                open=self.df["open"],
                high=self.df["high"],
                low=self.df["low"],
                close=self.df["close"],
                increasing_line_color="#00c2a0",
                decreasing_line_color="#ff4d5a",
                increasing_fillcolor="#00c2a0",
                decreasing_fillcolor="#ff4d5a",
                customdata=customdata,
                hovertext=[
                    f"时间={t}<br>开={o}<br>高={h}<br>低={l}<br>收={c}"
                    for t, o, h, l, c in zip(
                        self.tick_text,
                        self.df["open"],
                        self.df["high"],
                        self.df["low"],
                        self.df["close"],
                    )
                ],
                name="K线",
            ),
            row=1,
            col=1,
        )

        for rec in self.trendlines:
            x0 = float(rec.pivot_bar_index)
            x1 = self.right_x
            y0 = rec.y_at(x0)
            y1 = rec.y_at(x1)
            color = self.line_color_high if rec.kind == "high" else self.line_color_low
            fig.add_shape(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                xref="x1",
                yref="y1",
                line=dict(color=color, width=1, dash="dot"),
                layer="above",
            )

        factor_specs = [
            ("res_dist_atr", "res_dist_atr"),
            ("sup_dist_atr", "sup_dist_atr"),
            ("res_slope_pct252", "res_slope_pct252"),
            ("sup_slope_pct252", "sup_slope_pct252"),
            ("hold_above_res_ratio", "hold_above_res_ratio"),
            ("hold_below_sup_ratio", "hold_below_sup_ratio"),
        ]
        for col_name, display_name in factor_specs:
            fig.add_trace(
                go.Scatter(
                    x=self.x_numeric,
                    y=self.df[col_name],
                    mode="lines",
                    name=display_name,
                    line=dict(width=1.4),
                    hovertemplate=f"{display_name}=%{{y:.4f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # zero line on factor panel for dist/break visual aid
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#888", row=2, col=1)

        n = len(self.df)
        tick_step = max(1, n // 12)
        tickvals = list(range(0, n, tick_step))
        if tickvals[-1] != n - 1:
            tickvals.append(n - 1)
        ticktext = [self.tick_text[i] for i in tickvals]

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=40, r=80, t=90, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        )
        fig.update_xaxes(
            range=[-1, self.right_x + 1],
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True,
            zeroline=False,
            row=2,
            col=1,
        )
        fig.update_xaxes(showgrid=True, zeroline=False, row=1, col=1)
        fig.update_yaxes(showgrid=True, zeroline=False, row=1, col=1)
        fig.update_yaxes(showgrid=True, zeroline=False, row=2, col=1)
        return fig



def fetch_data(symbol: str, freq: str, bars: int):
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


def build_parser():
    p = argparse.ArgumentParser(description="ATR Based Trendlines - JD with factors (scheme1)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--freq", default="d")
    p.add_argument("--bars", type=int, default=1000)
    p.add_argument("--out", default="atr_trendlines_jd_with_factors_scheme1.html")
    p.add_argument("--len", type=int, default=30, dest="len")
    p.add_argument("--atr-percent", type=float, default=100.0)
    p.add_argument("--wicks", action="store_true", default=True)
    p.add_argument("--no-wicks", action="store_false", dest="wicks")
    p.add_argument("--do-mono", action="store_true", default=False)
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
    client.disconnect()

    engine = ATRTrendlinesJDWithFactors(df, args)
    engine.run()
    fig = engine.build_figure(f"ATR Based Trendlines - JD + Factors - {name}({args.symbol}) [{args.freq}]")
    plot(fig, filename=args.out, auto_open=False, include_plotlyjs=True)
    print(f"✅ HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()
