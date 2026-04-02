# -*- coding: utf-8 -*-
"""
文件名：
    dynamic_money_flow_with_factors.py

用途：
    参考 ATR Trendlines - JD Python 脚本结构，实现 TradingView 指标
    Dynamic Money Flow (DMF) 的 Python 版本，并加入逐 bar 计算的因子。

功能：
    1) 支持 pytdx 拉取行情数据
    2) 复刻 Pine 中 DMF 逻辑：ctr / ctc / alpha / vol / dmf / fast_ma / slow_ma
    3) 计算建议提取的关键因子
    4) 输出 HTML 图表（主图K线 + DMF面板 + 因子面板）
    5) 可选输出 CSV 因子文件

依赖：
    pip install pandas numpy plotly pytdx
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional

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

# ---- Colors ----
FAST_UP = "#00FF8B"
SLOW_UP = "#28DF8B"
SLOW_DN = "#DF287C"
FAST_DN = "#FF0074"
NEUTRAL = "#CAA335"
FAST_MA_COLOR = "rgba(0,255,139,0.18)"
SLOW_MA_COLOR = "rgba(255,0,116,0.18)"
ZERO_LINE_COLOR = "#888888"


# ---- Utils ----
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
        v = arr[i]
        if np.isnan(v):
            out[i] = prev
            continue
        prev = alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=src.index)


def ema_pine(src: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return pd.Series(np.nan, index=src.index)
    return src.ewm(span=length, adjust=False, min_periods=length).mean()


def wma_pine(src: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return pd.Series(np.nan, index=src.index)
    weights = np.arange(1, length + 1, dtype=float)
    denom = weights.sum()
    return src.rolling(length).apply(lambda x: float(np.dot(x, weights) / denom), raw=True)


def t3_pine(src: pd.Series, length: int, vfactor: float = 0.7) -> pd.Series:
    # Tillson T3 approximation with chained EMAs.
    e1 = ema_pine(src, length)
    e2 = ema_pine(e1, length)
    e3 = ema_pine(e2, length)
    e4 = ema_pine(e3, length)
    e5 = ema_pine(e4, length)
    e6 = ema_pine(e5, length)
    c1 = -vfactor**3
    c2 = 3 * vfactor**2 + 3 * vfactor**3
    c3 = -6 * vfactor**2 - 3 * vfactor - 3 * vfactor**3
    c4 = 1 + 3 * vfactor + vfactor**3 + 3 * vfactor**2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def frama_pine(src: pd.Series, length: int) -> pd.Series:
    # Practical FRAMA approximation for Python replication.
    arr = src.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if length <= 1 or len(arr) < length:
        return pd.Series(out, index=src.index)
    half = max(1, length // 2)
    for i in range(length - 1, len(arr)):
        window = arr[i - length + 1 : i + 1]
        if np.isnan(window).any():
            continue
        first = window[:half]
        second = window[-half:]
        n1 = (np.max(first) - np.min(first)) / half if half > 0 else 0.0
        n2 = (np.max(second) - np.min(second)) / half if half > 0 else 0.0
        n3 = (np.max(window) - np.min(window)) / length
        if n1 <= 0 or n2 <= 0 or n3 <= 0:
            alpha = 1.0
        else:
            dim = (math.log(n1 + n2) - math.log(n3)) / math.log(2)
            alpha = math.exp(-4.6 * (dim - 1))
            alpha = min(max(alpha, 0.01), 1.0)
        if i == length - 1 or np.isnan(out[i - 1]):
            out[i] = arr[i]
        else:
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=src.index)


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    for i in range(len(vals)):
        if i < window - 1:
            continue
        w = vals[i - window + 1 : i + 1]
        w = w[~np.isnan(w)]
        if len(w) == 0 or np.isnan(vals[i]):
            continue
        out[i] = np.mean(w <= vals[i])
    return pd.Series(out, index=series.index)


def bars_since_event(event: pd.Series) -> pd.Series:
    out = np.full(len(event), np.nan, dtype=float)
    last_idx: Optional[int] = None
    ev = event.fillna(False).astype(bool).to_numpy()
    for i, flag in enumerate(ev):
        if flag:
            last_idx = i
            out[i] = 0.0
        elif last_idx is not None:
            out[i] = float(i - last_idx)
    return pd.Series(out, index=event.index)


def consecutive_sign_bars(series: pd.Series) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    count = 0
    last_sign = 0
    for i, v in enumerate(vals):
        if np.isnan(v) or v == 0:
            count = 0
            last_sign = 0
            out[i] = 0.0 if not np.isnan(v) else np.nan
            continue
        s = 1 if v > 0 else -1
        if s == last_sign:
            count += 1
        else:
            count = 1
            last_sign = s
        out[i] = float(count)
    return pd.Series(out, index=series.index)


def pick_ma(src: pd.Series, ma_type: str, length: int) -> pd.Series:
    if length == 0 or ma_type == "OFF":
        return pd.Series(np.nan, index=src.index)
    if ma_type == "EMA":
        return ema_pine(src, length)
    if ma_type == "WMA":
        return wma_pine(src, length)
    if ma_type == "T3":
        return t3_pine(src, length)
    if ma_type == "FRAMA":
        return frama_pine(src, length)
    raise ValueError(f"Unsupported ma_type: {ma_type}")


# ---- Main Engine ----
class DynamicMoneyFlowWithFactors:
    def __init__(self, df: pd.DataFrame, args: argparse.Namespace):
        self.df = df.copy()
        self.args = args
        self.is_intraday = args.freq in {"5m", "15m", "30m", "60m"}
        self.times = list(self.df.index)
        self.x_numeric = np.arange(len(self.df), dtype=float)
        self.tick_text = [time_to_str(t, self.is_intraday) for t in self.times]

    def run(self):
        df = self.df
        prev_close = df["close"].shift(1)
        tr = true_range(df)
        trh = pd.concat([df["high"], prev_close], axis=1).max(axis=1)
        trl = pd.concat([df["low"], prev_close], axis=1).min(axis=1)

        if self.args.simulative_vol:
            sim_vol = (
                (df["close"] - prev_close).abs()
                + (df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)).abs() * 2
                + (pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]).abs() * 2
            )
            vol = sim_vol.pow(self.args.vol_power)
        else:
            vol = df["vol"].fillna(0.0).pow(self.args.vol_power)

        if self.args.weight_distribution == "Dynamic":
            alpha = pd.Series(np.where(tr == 0, 0.0, (df["close"] - prev_close).abs() / tr), index=df.index)
            alpha = alpha.fillna(0.0)
        else:
            alpha = pd.Series(float(self.args.static_dist_bias) / 100.0, index=df.index)

        ctr = pd.Series(
            np.where(
                tr == 0,
                0.0,
                ((df["close"] - trl + df["close"] - trh) / tr) * (1 - alpha) * vol,
            ),
            index=df.index,
        )

        ctc = pd.Series(
            np.where(
                (df["close"] - prev_close).fillna(0.0) == 0,
                0.0,
                np.where(df["close"] > prev_close, alpha * vol, -alpha * vol),
            ),
            index=df.index,
        )

        flow_raw = ctr + ctc

        if self.args.mode == "Index":
            vol_rma = pine_rma(vol, self.args.period)
            dmf = pine_rma(flow_raw, self.args.period) / vol_rma.replace(0.0, np.nan)
        else:
            dmf = flow_raw.cumsum()

        fast_ma = pick_ma(dmf, self.args.ma_switch, self.args.fast_len)
        slow_ma = pick_ma(dmf, self.args.ma_switch, self.args.slow_len)

        # --- Suggested factors ---
        ctr_norm = ctr / vol.replace(0.0, np.nan)
        ctc_norm = ctc / vol.replace(0.0, np.nan)
        flow_norm = flow_raw / vol.replace(0.0, np.nan)
        dmf_slope_1 = dmf.diff(1)
        dmf_slope_5 = (dmf - dmf.shift(5)) / 5.0
        dmf_accel = dmf_slope_1.diff(1)
        dmf_zscore_60 = (dmf - dmf.rolling(60).mean()) / dmf.rolling(60).std(ddof=0)
        dmf_percentile_252 = rolling_percentile_rank(dmf, 252)
        dmf_cross_zero = pd.Series(
            np.where((dmf > 0) & (dmf.shift(1) <= 0), 1,
                     np.where((dmf < 0) & (dmf.shift(1) >= 0), -1, 0)),
            index=df.index,
        )
        dmf_cross_up_01 = ((dmf > 0.1) & (dmf.shift(1) <= 0.1)).astype(float)
        dmf_cross_dn_01 = ((dmf < -0.1) & (dmf.shift(1) >= -0.1)).astype(float)
        dmf_bars_since_zero_cross = bars_since_event(dmf_cross_zero != 0)
        dmf_sign_persist = consecutive_sign_bars(dmf)
        dmf_ma_gap = fast_ma - slow_ma
        dmf_vs_fastma = dmf - fast_ma
        ma_cross_state = (fast_ma > slow_ma).astype(float)
        flow_impulse_20 = flow_norm - pine_rma(flow_norm.fillna(0.0), 20)

        # color state to mimic Pine visuals
        main_color = []
        for i in range(len(df)):
            dv = dmf.iat[i]
            fv = fast_ma.iat[i] if i < len(fast_ma) else np.nan
            sv = slow_ma.iat[i] if i < len(slow_ma) else np.nan
            if np.isnan(dv):
                main_color.append(NEUTRAL)
                continue
            if self.args.mode == "Index":
                if dv >= 0 and dv >= 0.1:
                    main_color.append(FAST_UP)
                elif dv >= 0 and dv < 0.1:
                    main_color.append(SLOW_UP)
                elif dv < 0 and dv >= -0.1:
                    main_color.append(SLOW_DN)
                else:
                    main_color.append(FAST_DN)
            elif not np.isnan(fv):
                if dv >= fv and (np.isnan(sv) or fv >= sv):
                    main_color.append(FAST_UP)
                elif dv >= fv and (not np.isnan(sv) and fv < sv):
                    main_color.append(SLOW_UP)
                elif dv < fv and (np.isnan(sv) or fv >= sv):
                    main_color.append(SLOW_DN)
                else:
                    main_color.append(FAST_DN)
            else:
                main_color.append(NEUTRAL)

        df["tr"] = tr
        df["trh"] = trh
        df["trl"] = trl
        df["dmf_vol"] = vol
        df["dmf_alpha"] = alpha
        df["dmf_ctr"] = ctr
        df["dmf_ctc"] = ctc
        df["dmf_flow_raw"] = flow_raw
        df["dmf_ctr_norm"] = ctr_norm
        df["dmf_ctc_norm"] = ctc_norm
        df["dmf_flow_norm"] = flow_norm
        df["dmf"] = dmf
        df["dmf_fast_ma"] = fast_ma
        df["dmf_slow_ma"] = slow_ma
        df["dmf_slope_1"] = dmf_slope_1
        df["dmf_slope_5"] = dmf_slope_5
        df["dmf_accel"] = dmf_accel
        df["dmf_zscore_60"] = dmf_zscore_60
        df["dmf_percentile_252"] = dmf_percentile_252
        df["dmf_cross_zero"] = dmf_cross_zero
        df["dmf_cross_up_01"] = dmf_cross_up_01
        df["dmf_cross_dn_01"] = dmf_cross_dn_01
        df["dmf_bars_since_zero_cross"] = dmf_bars_since_zero_cross
        df["dmf_sign_persist"] = dmf_sign_persist
        df["dmf_ma_gap"] = dmf_ma_gap
        df["dmf_vs_fastma"] = dmf_vs_fastma
        df["dmf_ma_cross_state"] = ma_cross_state
        df["dmf_flow_impulse_20"] = flow_impulse_20
        df["dmf_color"] = main_color

        self.df = df

    @staticmethod
    def _contiguous_segments(mask: pd.Series) -> list[tuple[int, int]]:
        vals = mask.fillna(False).astype(bool).to_numpy()
        segs = []
        start = None
        for i, flag in enumerate(vals):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                segs.append((start, i - 1))
                start = None
        if start is not None:
            segs.append((start, len(vals) - 1))
        return segs

    def _add_segmented_line(self, fig: go.Figure, row: int, col: int, y: pd.Series, colors: list[str], name: str, width: float = 2.0):
        n = len(y)
        if n == 0:
            return
        start = 0
        curr = colors[0]
        for i in range(1, n + 1):
            changed = i == n or colors[i] != curr
            if changed:
                lo = max(0, start - 1)
                hi = i - 1
                fig.add_trace(
                    go.Scatter(
                        x=self.x_numeric[lo:hi + 1],
                        y=y.iloc[lo:hi + 1],
                        mode="lines",
                        name=name if start == 0 else None,
                        showlegend=(start == 0),
                        line=dict(color=curr, width=width),
                        hovertemplate=f"{name}=%{{y:.4f}}<extra></extra>",
                        connectgaps=False,
                    ),
                    row=row,
                    col=col,
                )
                start = i
                if i < n:
                    curr = colors[i]

    def _add_ma_fill_segments(self, fig: go.Figure, row: int, col: int, fast: pd.Series, slow: pd.Series):
        valid = fast.notna() & slow.notna()
        bull = valid & (fast > slow)
        bear = valid & (fast <= slow)
        for mask, fill_color in [(bull, FAST_MA_COLOR), (bear, SLOW_MA_COLOR)]:
            for s, e in self._contiguous_segments(mask):
                if e - s < 1:
                    continue
                xseg = self.x_numeric[s:e + 1]
                fseg = fast.iloc[s:e + 1]
                sseg = slow.iloc[s:e + 1]
                fig.add_trace(
                    go.Scatter(
                        x=xseg, y=fseg, mode="lines", line=dict(color='rgba(0,0,0,0)', width=0.1),
                        showlegend=False, hoverinfo='skip'
                    ),
                    row=row, col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=xseg, y=sseg, mode="lines", line=dict(color='rgba(0,0,0,0)', width=0.1),
                        fill='tonexty', fillcolor=fill_color,
                        showlegend=False, hoverinfo='skip'
                    ),
                    row=row, col=col,
                )

    def _add_zero_fill_segments(self, fig: go.Figure, row: int, col: int, y: pd.Series):
        valid = y.notna()
        pos = valid & (y >= 0)
        neg = valid & (y < 0)
        for mask, fill_color in [(pos, 'rgba(0,255,139,0.05)'), (neg, 'rgba(255,0,116,0.05)')]:
            for s, e in self._contiguous_segments(mask):
                if e - s < 1:
                    continue
                xseg = self.x_numeric[s:e + 1]
                yseg = y.iloc[s:e + 1]
                zseg = np.zeros(len(xseg))
                fig.add_trace(
                    go.Scatter(
                        x=xseg, y=yseg, mode="lines", line=dict(color='rgba(0,0,0,0)', width=0.1),
                        showlegend=False, hoverinfo='skip'
                    ),
                    row=row, col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=xseg, y=zseg, mode="lines", line=dict(color='rgba(0,0,0,0)', width=0.1),
                        fill='tonexty', fillcolor=fill_color,
                        showlegend=False, hoverinfo='skip'
                    ),
                    row=row, col=col,
                )

    def build_figure(self, title: str) -> go.Figure:
        df = self.df
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.46, 0.27, 0.27],
            subplot_titles=(title, "DMF", "Factors"),
        )

        fig.add_trace(
            go.Candlestick(
                x=self.x_numeric,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="#00c2a0",
                decreasing_line_color="#ff4d5a",
                increasing_fillcolor="#00c2a0",
                decreasing_fillcolor="#ff4d5a",
                name="K线",
                hovertext=[
                    f"时间={t}<br>开={o:.3f}<br>高={h:.3f}<br>低={l:.3f}<br>收={c:.3f}"
                    for t, o, h, l, c in zip(self.tick_text, df["open"], df["high"], df["low"], df["close"])
                ],
            ),
            row=1,
            col=1,
        )

        # TV-like DMF panel: zero fill + MA fill + colored main line
        if self.args.mode == "Index":
            self._add_zero_fill_segments(fig, row=2, col=1, y=df["dmf"])

        self._add_ma_fill_segments(fig, row=2, col=1, fast=df["dmf_fast_ma"], slow=df["dmf_slow_ma"])

        fig.add_trace(
            go.Scatter(
                x=self.x_numeric,
                y=df["dmf_fast_ma"],
                mode="lines",
                name="Fast MA",
                line=dict(color=FAST_MA_COLOR, width=1),
                hovertemplate="Fast MA=%{y:.4f}<extra></extra>",
                connectgaps=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.x_numeric,
                y=df["dmf_slow_ma"],
                mode="lines",
                name="Slow MA",
                line=dict(color=SLOW_MA_COLOR, width=1),
                hovertemplate="Slow MA=%{y:.4f}<extra></extra>",
                connectgaps=False,
            ),
            row=2,
            col=1,
        )
        self._add_segmented_line(fig, row=2, col=1, y=df["dmf"], colors=df["dmf_color"].tolist(), name="DMF", width=2)

        if self.args.mode == "Index":
            fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color=ZERO_LINE_COLOR, row=2, col=1)
            fig.add_hline(y=0.1, line_width=1, line_dash="dash", line_color='rgba(120,123,134,0.5)', row=2, col=1)
            fig.add_hline(y=-0.1, line_width=1, line_dash="dash", line_color='rgba(120,123,134,0.5)', row=2, col=1)

        # factors panel
        factor_specs = [
            ("dmf_flow_norm", "dmf_flow_norm"),
            ("dmf_alpha", "dmf_alpha"),
            ("dmf_slope_1", "dmf_slope_1"),
            ("dmf_accel", "dmf_accel"),
            ("dmf_ma_gap", "dmf_ma_gap"),
            ("dmf_bars_since_zero_cross", "dmf_bars_since_zero_cross"),
        ]
        for col_name, display_name in factor_specs:
            fig.add_trace(
                go.Scatter(
                    x=self.x_numeric,
                    y=df[col_name],
                    mode="lines",
                    name=display_name,
                    line=dict(width=1.2),
                    hovertemplate=f"{display_name}=%{{y:.4f}}<extra></extra>",
                    connectgaps=False,
                ),
                row=3,
                col=1,
            )
        fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color=ZERO_LINE_COLOR, row=3, col=1)

        n = len(df)
        tick_step = max(1, n // 12)
        tickvals = list(range(0, n, tick_step))
        if tickvals[-1] != n - 1:
            tickvals.append(n - 1)
        ticktext = [self.tick_text[i] for i in tickvals]

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=40, r=70, t=90, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
            bargap=0.0,
            plot_bgcolor='#131722',
            paper_bgcolor='#131722',
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.06)',
            zeroline=False,
            row=3,
            col=1,
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False, row=1, col=1)
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False, row=2, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False, row=2, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False, row=3, col=1)
        return fig


# ---- Data I/O ----
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

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).set_index("datetime")
    return df[["open", "high", "low", "close", "vol", "amount"]].astype(float), client


def build_parser():
    p = argparse.ArgumentParser(description="Dynamic Money Flow + factors (pytdx / plotly)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--freq", default="d")
    p.add_argument("--bars", type=int, default=1000)
    p.add_argument("--out", default="dynamic_money_flow_with_factors.html")
    p.add_argument("--csv-out", default="dynamic_money_flow_with_factors.csv")

    p.add_argument("--mode", default="Index", choices=["Index", "Cumulative"])
    p.add_argument("--period", type=int, default=26)
    p.add_argument("--ma-switch", default="EMA", choices=["OFF", "EMA", "WMA", "T3", "FRAMA"])
    p.add_argument("--fast-len", type=int, default=8)
    p.add_argument("--slow-len", type=int, default=20)
    p.add_argument("--simulative-vol", action="store_true", default=False)
    p.add_argument("--vol-power", type=float, default=1.0)
    p.add_argument("--weight-distribution", default="Dynamic", choices=["Dynamic", "Static"])
    p.add_argument("--static-dist-bias", type=int, default=50)
    p.add_argument("--export-csv", action="store_true", default=False)
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

    engine = DynamicMoneyFlowWithFactors(df, args)
    engine.run()
    fig = engine.build_figure(f"Dynamic Money Flow + Factors - {name}({args.symbol}) [{args.freq}]")
    plot(fig, filename=args.out, auto_open=False, include_plotlyjs=True)
    print(f"✅ HTML 已生成: {args.out}")

    if args.export_csv:
        engine.df.to_csv(args.csv_out, encoding="utf-8-sig")
        print(f"✅ CSV 已生成: {args.csv_out}")


if __name__ == "__main__":
    main()
