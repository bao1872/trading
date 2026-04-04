# -*- coding: utf-8 -*-
"""
ATR Rope 趋势跟踪策略 + 因子计算 (pytdx/plotly)

Purpose: 基于 ATR 的动态趋势线指标，识别趋势方向、盘整区间、突破信号，并输出量化因子供后续策略使用。
        核心算法faithful复刻自 TradingView Pine Script 的 ATR Rope 实现。

Inputs:
    --symbol  股票代码，如 "000001"（平安银行）、"600519"（贵州茅台）
    --freq    K线周期：d=日线，w=周线，m=月线，5m/15m/30m/60m=分钟线
    --bars    获取K线数量，默认1000
    --len     ATR周期，默认14
    --multi   ATR倍数（rope移动阈值），默认1.5
    --source  触发源价格，默认close，可选open/high/low/close

Outputs:
    --out         输出HTML文件名（交互式K线图），默认 atr_rope_tv_clone.html
    --csv-out     输出CSV文件名（含所有计算因子），可选

How to Run:
    # 日线：获取平安银行最近1000条日K，输出交互图
    python atr_rope_with_factors_pytdx_plotly.py --symbol 000001 --freq d --bars 1000

    # 周线：贵州茅台周线，ATR倍数2.0，输出CSV供后续分析
    python atr_rope_with_factors_pytdx_plotly.py --symbol 600519 --freq w --multi 2.0 --csv-out atr_rope_600519_w.csv

    # 分钟线：日线日内15分钟，隐藏区间和因子面板
    python atr_rope_with_factors_pytdx_plotly.py --symbol 000001 --freq 15m --bars 500 --hide-ranges --hide-factor-panels

    # 只看ATR通道和突破信号
    python atr_rope_with_factors_pytdx_plotly.py --symbol 000001 --freq d --show-break-markers --show-atr-channel

Side Effects: 写入 --out 指定的HTML文件和 --csv-out 指定的CSV文件。

Strategy Logic（策略逻辑）:
    1. ATR Rope 核心算法:
       - rope: 动态跟踪线，只有当价格变动超过 ATR*multi 时才移动
       - rope向上移动=多头趋势，向下=空头趋势，不动=盘整
       - 相比传统均线，ATR Rope 不会在震荡行情中频繁来回穿梭

    2. 趋势方向判断:
       - dir=1: 上涨趋势（rope向上）
       - dir=-1: 下跌趋势（rope向下）
       - dir=0: 盘整/区间震荡

    3. 区间(Consolidation)识别:
       - 当dir=0时，记录横向支撑区间(c_hi/c_lo)
       - ff(flip flag)交替标记，将区间分为两组，用于识别交替出现的支撑阻力
       - 常用于识别吸筹区、派发区

    4. 突破信号:
       - range_break_up: 价格向上突破区间高点
       - range_break_down: 价格向下突破区间低点
       - 突破强度 = 突破幅度 / ATR

    5. 输出因子（用于后续量化策略）:
       状态因子: rope_dir, bars_since_dir_change, is_consolidating, consolidation_bars
       距离因子: dist_to_rope_atr, dist_to_upper_atr, dist_to_lower_atr（价格到各线的距离，以ATR为单位）
       斜率因子: rope_slope_atr_5（5日前rope到当前的slope/ATR），rope_slope_pct_5（slope百分比）
       区间因子: range_width_atr（区间宽度/ATR），dist_to_range_high_atr, dist_to_range_low_atr
       突破因子: range_break_up/down, range_break_up/down_strength

Examples:
    python atr_rope_with_factors_pytdx_plotly.py --symbol 000001 --freq d --bars 1000 --out test.html
    python atr_rope_with_factors_pytdx_plotly.py --symbol 600519 --freq w --csv-out 600519_weekly.csv
    python atr_rope_with_factors_pytdx_plotly.py --symbol 000001 --freq 60m --bars 500 --show-break-markers

A-Share Market Notes（A股注意事项）:
    - T+1结算：当日买入不能在当日卖出，rope方向信号在日内可能失效
    - 涨跌停：涨跌停时价格无法继续向该方向移动，rope可能停驻
    - 10%日涨跌幅限制：极端行情下ATR会被放大，影响rope灵敏度
    - 建议在日线及以上周期使用，分钟线需考虑T+1和流动性
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

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

UP_COL = "#3daa45"
DOWN_COL = "#ff033e"
FLAT_COL = "#004d92"
RANGE_LINE_COL = "rgba(0, 114, 230, 0.95)"
RANGE_FILL_COL = "rgba(0, 77, 146, 0.20)"
CHANNEL_LINE_COL = "rgba(38, 139, 255, 0.98)"
CHANNEL_FILL_COL = "rgba(0, 114, 230, 0.12)"
ZERO_COL = "#888888"


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
    if length <= 0 or len(arr) == 0 or len(arr) < length:
        return pd.Series(out, index=src.index)
    init = np.nanmean(arr[:length])
    out[length - 1] = init
    prev = init
    alpha = 1.0 / length
    for i in range(length, len(arr)):
        prev = alpha * arr[i] + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=src.index)


def atr_pine(df: pd.DataFrame, length: int) -> pd.Series:
    return pine_rma(true_range(df), length)


def nz(v: float, default: float = 0.0) -> float:
    return default if pd.isna(v) else float(v)


def pine_cross(curr_a: float, curr_b: float, prev_a: float, prev_b: float) -> bool:
    """Approximate Pine ta.cross(a, b) using series semantics."""
    if any(pd.isna(v) for v in (curr_a, curr_b, prev_a, prev_b)):
        return False
    curr_rel = curr_a - curr_b
    prev_rel = prev_a - prev_b
    crossover = curr_rel > 0 and prev_rel <= 0
    crossunder = curr_rel < 0 and prev_rel >= 0
    return bool(crossover or crossunder)


def segment_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


@dataclass
class Segments:
    rope_up: np.ndarray
    rope_down: np.ndarray
    rope_flat: np.ndarray


class ATRRopeEngine:
    def __init__(self, df: pd.DataFrame, args: argparse.Namespace):
        self.df = df.copy()
        self.args = args
        self.len = int(args.len)
        self.multi = float(args.multi)
        self.src_col = args.source
        self.is_intraday = args.freq in {"5m", "15m", "30m", "60m"}
        self.times = list(self.df.index)
        self.x_numeric = np.arange(len(self.df), dtype=float)
        self.tick_text = [time_to_str(t, self.is_intraday) for t in self.times]

        self.df["src"] = self.df[self.src_col].astype(float)
        self.df["atr_raw"] = atr_pine(self.df, self.len)
        self.df["atr"] = self.df["atr_raw"] * self.multi

    def run(self) -> None:
        src = self.df["src"].to_numpy(dtype=float)
        atr = self.df["atr"].to_numpy(dtype=float)
        close = self.df["close"].to_numpy(dtype=float)
        n = len(self.df)

        rope_arr = np.full(n, np.nan)
        upper_arr = np.full(n, np.nan)
        lower_arr = np.full(n, np.nan)
        dir_arr = np.full(n, np.nan)
        ff_arr = np.full(n, np.nan)
        chi_arr = np.full(n, np.nan)
        clo_arr = np.full(n, np.nan)
        bars_since_dir_change = np.full(n, np.nan)
        cons_bars_arr = np.full(n, np.nan)
        is_cons_arr = np.full(n, np.nan)

        dist_rope_arr = np.full(n, np.nan)
        dist_upper_arr = np.full(n, np.nan)
        dist_lower_arr = np.full(n, np.nan)
        slope_atr_arr = np.full(n, np.nan)
        slope_pct_arr = np.full(n, np.nan)
        width_atr_arr = np.full(n, np.nan)
        dist_rh_arr = np.full(n, np.nan)
        dist_rl_arr = np.full(n, np.nan)
        break_up_arr = np.zeros(n, dtype=float)
        break_dn_arr = np.zeros(n, dtype=float)
        break_up_strength_arr = np.full(n, np.nan)
        break_dn_strength_arr = np.full(n, np.nan)

        rope = math.nan
        dir_val = 0
        last_dir_change_idx = None
        c_hi = math.nan
        c_lo = math.nan
        h_sum = 0.0
        l_sum = 0.0
        c_count = 0
        ff = True

        for i in range(n):
            src_i = src[i]
            atr_i = atr[i]
            if pd.isna(src_i):
                continue
            if pd.isna(rope):
                rope = float(src_i)

            move = float(src_i) - rope
            rope = rope + max(abs(move) - nz(atr_i, 0.0), 0.0) * float(np.sign(move))
            upper = rope + atr_i if pd.notna(atr_i) else math.nan
            lower = rope - atr_i if pd.notna(atr_i) else math.nan

            prev_dir = dir_val
            prev_rope = rope_arr[i - 1] if i > 0 else math.nan
            if i > 0 and pd.notna(prev_rope):
                if rope > prev_rope:
                    dir_val = 1
                elif rope < prev_rope:
                    dir_val = -1

            if i > 0 and pine_cross(src_i, rope, src[i - 1], prev_rope):
                dir_val = 0

            if i == 0 or dir_val != prev_dir:
                last_dir_change_idx = i

            if dir_val == 0:
                if prev_dir != 0:
                    h_sum = 0.0
                    l_sum = 0.0
                    c_count = 0
                    ff = not ff
                if pd.notna(upper) and pd.notna(lower):
                    h_sum += upper
                    l_sum += lower
                    c_count += 1
                    c_hi = h_sum / c_count
                    c_lo = l_sum / c_count

            rope_arr[i] = rope
            upper_arr[i] = upper
            lower_arr[i] = lower
            dir_arr[i] = dir_val
            ff_arr[i] = 1.0 if ff else 0.0
            chi_arr[i] = c_hi
            clo_arr[i] = c_lo
            is_cons_arr[i] = 1.0 if dir_val == 0 else 0.0
            cons_bars_arr[i] = float(c_count) if dir_val == 0 else 0.0
            bars_since_dir_change[i] = float(i - last_dir_change_idx) if last_dir_change_idx is not None else np.nan

            if pd.notna(atr_i) and atr_i != 0:
                dist_rope_arr[i] = (close[i] - rope) / atr_i
                dist_upper_arr[i] = (close[i] - upper) / atr_i if pd.notna(upper) else np.nan
                dist_lower_arr[i] = (close[i] - lower) / atr_i if pd.notna(lower) else np.nan
                if i >= 5 and pd.notna(rope_arr[i - 5]):
                    slope_atr_arr[i] = (rope - rope_arr[i - 5]) / atr_i
                    if rope_arr[i - 5] != 0:
                        slope_pct_arr[i] = rope / rope_arr[i - 5] - 1.0
                if pd.notna(c_hi) and pd.notna(c_lo):
                    width_atr_arr[i] = (c_hi - c_lo) / atr_i
                    dist_rh_arr[i] = (c_hi - close[i]) / atr_i
                    dist_rl_arr[i] = (close[i] - c_lo) / atr_i
                    if i > 0 and pd.notna(chi_arr[i - 1]) and pd.notna(clo_arr[i - 1]):
                        up_break = close[i] > c_hi and close[i - 1] <= chi_arr[i - 1]
                        dn_break = close[i] < c_lo and close[i - 1] >= clo_arr[i - 1]
                        if up_break:
                            break_up_arr[i] = 1.0
                            break_up_strength_arr[i] = (close[i] - c_hi) / atr_i
                        if dn_break:
                            break_dn_arr[i] = 1.0
                            break_dn_strength_arr[i] = (c_lo - close[i]) / atr_i

        self.df["rope"] = rope_arr
        self.df["upper"] = upper_arr
        self.df["lower"] = lower_arr
        self.df["rope_dir"] = dir_arr
        self.df["ff"] = ff_arr
        self.df["c_hi"] = chi_arr
        self.df["c_lo"] = clo_arr
        self.df["is_consolidating"] = is_cons_arr
        self.df["bars_since_dir_change"] = bars_since_dir_change
        self.df["consolidation_bars"] = cons_bars_arr
        self.df["dist_to_rope_atr"] = dist_rope_arr
        self.df["dist_to_upper_atr"] = dist_upper_arr
        self.df["dist_to_lower_atr"] = dist_lower_arr
        self.df["rope_slope_atr_5"] = slope_atr_arr
        self.df["rope_slope_pct_5"] = slope_pct_arr
        self.df["range_width_atr"] = width_atr_arr
        self.df["dist_to_range_high_atr"] = dist_rh_arr
        self.df["dist_to_range_low_atr"] = dist_rl_arr
        self.df["range_break_up"] = break_up_arr
        self.df["range_break_down"] = break_dn_arr
        self.df["range_break_up_strength"] = break_up_strength_arr
        self.df["range_break_down_strength"] = break_dn_strength_arr

        self.df["rope_up"] = self.df["rope"].where(self.df["rope_dir"] == 1)
        self.df["rope_down"] = self.df["rope"].where(self.df["rope_dir"] == -1)
        self.df["rope_flat"] = self.df["rope"].where(self.df["rope_dir"] == 0)

        # Two alternating groups exactly like Pine's ff/h1-l1/h2-l2.
        ff_bool = self.df["ff"] == 1.0
        self.df["range_high_1"] = self.df["c_hi"].where(~ff_bool)
        self.df["range_low_1"] = self.df["c_lo"].where(~ff_bool)
        self.df["range_high_2"] = self.df["c_hi"].where(ff_bool)
        self.df["range_low_2"] = self.df["c_lo"].where(ff_bool)

        # Continuous analysis variants for CSV/modeling only.
        self.df["range_width_atr_ffill"] = self.df["range_width_atr"].ffill()
        self.df["dist_to_range_high_atr_ffill"] = self.df["dist_to_range_high_atr"].ffill()
        self.df["dist_to_range_low_atr_ffill"] = self.df["dist_to_range_low_atr"].ffill()

    def _add_range_segments(self, fig: go.Figure, hi_col: str, lo_col: str, row: int, col: int, showlegend: bool) -> None:
        hi = self.df[hi_col].to_numpy(dtype=float)
        lo = self.df[lo_col].to_numpy(dtype=float)
        mask = np.isfinite(hi) & np.isfinite(lo)
        runs = segment_runs(mask)
        first = showlegend
        for s, e in runs:
            x = self.x_numeric[s:e+1]
            hi_y = hi[s:e+1]
            lo_y = lo[s:e+1]
            fig.add_trace(
                go.Scatter(
                    x=x, y=hi_y, mode="lines",
                    line=dict(width=1.25, color=RANGE_LINE_COL),
                    name="range",
                    legendgroup="range",
                    showlegend=first,
                    connectgaps=False,
                    hovertemplate="range_high=%{y:.4f}<extra></extra>",
                ), row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=lo_y, mode="lines",
                    line=dict(width=1.25, color=RANGE_LINE_COL),
                    fill="tonexty",
                    fillcolor=RANGE_FILL_COL,
                    name="range_fill",
                    legendgroup="range",
                    showlegend=False,
                    connectgaps=False,
                    hovertemplate="range_low=%{y:.4f}<extra></extra>",
                ), row=row, col=col
            )
            first = False

    def build_figure(self, title: str) -> go.Figure:
        if self.args.show_factor_panels:
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                row_heights=[0.58, 0.12, 0.15, 0.15],
                subplot_titles=(title, "State Factors", "Distance / Slope Factors", "Range / Breakout Factors"),
            )
        else:
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=(title,))

        main_row = 1
        fig.add_trace(
            go.Candlestick(
                x=self.x_numeric,
                open=self.df["open"], high=self.df["high"], low=self.df["low"], close=self.df["close"],
                increasing_line_color="#00c2a0", decreasing_line_color="#ff4d5a",
                increasing_fillcolor="#00c2a0", decreasing_fillcolor="#ff4d5a",
                name="K线", showlegend=False,
            ), row=main_row, col=1
        )

        for col_name, color, name, lg in [
            ("rope_up", UP_COL, "rope", True),
            ("rope_down", DOWN_COL, "rope", False),
            ("rope_flat", FLAT_COL, "rope", False),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=self.x_numeric, y=self.df[col_name], mode="lines",
                    line=dict(width=3.0, color=color),
                    connectgaps=False, name=name, legendgroup="rope", showlegend=lg,
                    hovertemplate="rope=%{y:.4f}<extra></extra>",
                ), row=main_row, col=1
            )

        if self.args.show_atr_channel:
            fig.add_trace(
                go.Scatter(
                    x=self.x_numeric, y=self.df["upper"], mode="lines",
                    line=dict(width=1.8, color=CHANNEL_LINE_COL),
                    name="upper", showlegend=False, connectgaps=False,
                    hovertemplate="upper=%{y:.4f}<extra></extra>",
                ), row=main_row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.x_numeric, y=self.df["lower"], mode="lines",
                    line=dict(width=1.8, color=CHANNEL_LINE_COL),
                    fill="tonexty", fillcolor=CHANNEL_FILL_COL,
                    name="lower", showlegend=False, connectgaps=False,
                    hovertemplate="lower=%{y:.4f}<extra></extra>",
                ), row=main_row, col=1
            )

        if self.args.show_ranges:
            self._add_range_segments(fig, "range_high_1", "range_low_1", main_row, 1, True)
            self._add_range_segments(fig, "range_high_2", "range_low_2", main_row, 1, False)

        if self.args.show_break_markers:
            up_mask = self.df["range_break_up"] == 1
            dn_mask = self.df["range_break_down"] == 1
            fig.add_trace(go.Scatter(x=self.x_numeric[up_mask.to_numpy()], y=self.df.loc[up_mask, "close"], mode="markers", marker=dict(symbol="circle-open", size=9, color="#2f7cff", line=dict(width=1.5)), name="range_break_up", showlegend=False), row=main_row, col=1)
            fig.add_trace(go.Scatter(x=self.x_numeric[dn_mask.to_numpy()], y=self.df.loc[dn_mask, "close"], mode="markers", marker=dict(symbol="circle-open", size=9, color="#2f7cff", line=dict(width=1.5)), name="range_break_down", showlegend=False), row=main_row, col=1)

        if self.args.show_factor_panels:
            state_factors = [
                ("rope_dir", "lines"),
                ("bars_since_dir_change", "lines"),
                ("is_consolidating", "lines"),
                ("consolidation_bars", "lines"),
            ]
            for nm, mode in state_factors:
                fig.add_trace(
                    go.Scatter(
                        x=self.x_numeric, y=self.df[nm], mode=mode, name=nm,
                        line=dict(width=1.35), connectgaps=False, showlegend=False
                    ), row=2, col=1
                )

            geom_factors = [
                "dist_to_rope_atr",
                "dist_to_upper_atr",
                "dist_to_lower_atr",
                "rope_slope_atr_5",
            ]
            for nm in geom_factors:
                fig.add_trace(
                    go.Scatter(
                        x=self.x_numeric, y=self.df[nm], mode="lines", name=nm,
                        line=dict(width=1.35), connectgaps=False, showlegend=False
                    ), row=3, col=1
                )

            range_factors = [
                "range_width_atr",
                "range_break_up_strength",
            ]
            for nm in range_factors:
                fig.add_trace(
                    go.Scatter(
                        x=self.x_numeric, y=self.df[nm], mode="lines", name=nm,
                        line=dict(width=1.35), connectgaps=False, showlegend=False
                    ), row=4, col=1
                )

            fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=2, col=1)
            fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=3, col=1)
            fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=4, col=1)

        n = len(self.df)
        tick_step = max(1, n // 10)
        tickvals = list(range(0, n, tick_step))
        if tickvals[-1] != n - 1:
            tickvals.append(n - 1)
        ticktext = [self.tick_text[i] for i in tickvals]

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=40, r=20, t=70, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.01),
        )
        rows = (1, 2, 3, 4) if self.args.show_factor_panels else (1,)
        for r in rows:
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True, zeroline=False, row=r, col=1)
            fig.update_yaxes(showgrid=True, zeroline=False, row=r, col=1)
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
    p = argparse.ArgumentParser(description="ATR Rope faithful clone + factors (pytdx/plotly)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--freq", default="d")
    p.add_argument("--bars", type=int, default=1000)
    p.add_argument("--out", default="atr_rope_tv_clone.html")
    p.add_argument("--csv-out", default="")
    p.add_argument("--len", type=int, default=14, dest="len")
    p.add_argument("--multi", type=float, default=1.5)
    p.add_argument("--source", default="close", choices=["open", "high", "low", "close"])
    p.add_argument("--show-ranges", action="store_true", default=True)
    p.add_argument("--hide-ranges", action="store_false", dest="show_ranges")
    p.add_argument("--show-atr-channel", action="store_true", default=False)
    p.add_argument("--show-break-markers", action="store_true", default=False)
    p.add_argument("--hide-atr-channel", action="store_false", dest="show_atr_channel")
    p.add_argument("--show-factor-panels", action="store_true", default=True)
    p.add_argument("--hide-factor-panels", action="store_false", dest="show_factor_panels")
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

    engine = ATRRopeEngine(df, args)
    engine.run()
    fig = engine.build_figure(f"ATR Rope - {name}({args.symbol}) [{args.freq}]")
    plot(fig, filename=args.out, auto_open=False, include_plotlyjs=True)

    if args.csv_out:
        engine.df.to_csv(args.csv_out, encoding="utf-8-sig")
        print(f"CSV 已生成: {args.csv_out}")
    print(f"HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()
