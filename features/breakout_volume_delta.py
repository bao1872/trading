# -*- coding: utf-8 -*-
"""
Breakout Volume Delta | Flux Charts 的 Python/Plotly 主周期自洽版。

Purpose:
    参考用户提供的 Pine Script 逻辑，提取 swing high/low 水平位，
    检测突破，并仅使用当前主周期 bar 自身的方向来估算 bull / bear volume，
    最终输出交互式 HTML 图与可选 CSV。

Inputs:
    --symbol      股票代码，例如 600988
    --bars        获取主周期 K 线数量，默认 255
    --freq        主周期，可选 d/60m/30m/15m/5m
    --swing-left  swing 左侧长度，默认 10
    --swing-right swing 右侧长度，默认 10
    --breakout-by 突破判定方式：Close/Wick
    --max-levels  图上最多保留的最近水平位数量，默认 5
    --br-vol-filter             是否启用突破量能过滤
    --br-vol-filter-pct         过滤阈值，默认 60
    --csv-out     可选，导出因子表 CSV
    --out         输出 HTML 文件

Outputs:
    1) HTML：K 线 + swing 水平位 + 突破标记 + bull/bear volume delta 标签
    2) CSV：逐 bar 的结构与突破因子

How to Run:
    python breakout_volume_delta_pine_aligned_multi_tf.py --symbol 600988 --freq d --bars 255 --out 600988_d.html
    python breakout_volume_delta_pine_aligned_multi_tf.py --symbol 600988 --freq 15m --bars 500 --out 600988_15m.html

Side Effects:
    写出 HTML / CSV 文件，并从 pytdx 拉取指定主周期数据。

Notes:
    - 不再抓取低周期数据，也不做 1m 子级 volume 聚合。
    - bull/bear volume 直接按当前主周期 bar 的 open/close 方向近似。
    - 适合你当前需求：查看不同主周期下的结构/突破 HTML。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _base_dir not in sys.path:
        sys.path.insert(0, _base_dir)

from datasource.pytdx_client import connect_pytdx, get_kline_data


BULL_COL = "#00FF6A"
BEAR_COL = "#FF2D55"
UNMIT_HI_COL = "#32CD32"
UNMIT_LO_COL = "#FF4D4F"
BRK_HI_COL = "#9CFF9C"
BRK_LO_COL = "#FF9C9C"
ZERO_COL = "#888888"


@dataclass
class Level:
    price: float
    source_index: int
    source_time: pd.Timestamp
    mitigated: bool = False
    broken_index: Optional[int] = None


@dataclass
class BreakEvent:
    index: int
    time: pd.Timestamp
    direction: str  # "up" / "down"
    level_price: float
    bull_vol: float
    bear_vol: float
    bull_pct: float
    bear_pct: float
    n_levels_broken: int


@dataclass
class PxT:
    p: float
    t: pd.Timestamp


@dataclass
class SwBar:
    h: PxT
    l: PxT


class EastmoneyFetcher:
    """通过 Eastmoney 公共接口抓取日线与分钟线。"""

    BASE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

    @staticmethod
    def secid(symbol: str) -> str:
        s = str(symbol)
        if s.startswith(("6", "9")):
            return f"1.{s}"
        return f"0.{s}"

    @classmethod
    def fetch_kline(cls, symbol: str, klt: int, lmt: int = 255, beg: str = "0", end: str = "20500101") -> pd.DataFrame:
        params = {
            "secid": cls.secid(symbol),
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": str(klt),
            "fqt": "1",
            "beg": beg,
            "end": end,
            "lmt": str(lmt),
        }
        r = requests.get(cls.BASE, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        data = js.get("data") or {}
        klines = data.get("klines") or []
        if not klines:
            raise RuntimeError(f"Eastmoney 未返回K线数据: symbol={symbol}, klt={klt}")

        rows = []
        for item in klines:
            parts = item.split(",")
            # f51-f61: 日期,开,收,高,低,量,额,振幅,涨跌幅,涨跌额,换手率
            if len(parts) < 11:
                continue
            rows.append(
                {
                    "datetime": pd.to_datetime(parts[0]),
                    "open": float(parts[1]),
                    "close": float(parts[2]),
                    "high": float(parts[3]),
                    "low": float(parts[4]),
                    "vol": float(parts[5]),
                    "amount": float(parts[6]),
                    "amp_pct": float(parts[7]),
                    "pct_chg": float(parts[8]),
                    "chg": float(parts[9]),
                    "turnover": float(parts[10]),
                }
            )
        df = pd.DataFrame(rows).sort_values("datetime").drop_duplicates("datetime")
        return df.set_index("datetime")

    @classmethod
    def fetch_daily(cls, symbol: str, bars: int) -> pd.DataFrame:
        return cls.fetch_kline(symbol=symbol, klt=101, lmt=bars)

    @classmethod
    def fetch_1m_recent(cls, symbol: str, lmt: int = 50000) -> pd.DataFrame:
        # 1分钟线接口：klt=1
        return cls.fetch_kline(symbol=symbol, klt=1, lmt=lmt)


class BreakoutVolumeDeltaEngine:
    def __init__(self, df: pd.DataFrame, args: argparse.Namespace):
        self.df = df.copy()
        self.args = args
        self.times = list(self.df.index)
        self.x = np.arange(len(self.df), dtype=float)
        self.tick_text = [self._format_tick_text(t) for t in self.times]

        self.high_levels: List[Level] = []
        self.low_levels: List[Level] = []
        self.break_events: List[BreakEvent] = []

    @staticmethod
    def _format_tick_text(ts: pd.Timestamp) -> str:
        if pd.isna(ts):
            return ""
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
            return ts.strftime("%Y-%m-%d")
        return ts.strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _fmt(v: float) -> str:
        av = abs(v)
        if av >= 1e9:
            return f"{v / 1e9:.2f}B"
        if av >= 1e6:
            return f"{v / 1e6:.2f}M"
        if av >= 1e3:
            return f"{v / 1e3:.2f}K"
        return f"{v:.0f}"

    @staticmethod
    def _pct_str(pct01: float) -> str:
        p = max(0.0, min(1.0, pct01)) * 100.0
        return f"{p:.1f}%"

    @staticmethod
    def _trim_levels(levels: List[Level], max_lvls: int) -> List[Level]:
        if len(levels) <= max_lvls:
            return levels
        return levels[-max_lvls:]

    def _calc_main_tf_volume_delta(self) -> pd.DataFrame:
        """
        按用户最新要求：不做低周期聚合。
        bull / bear volume 直接基于当前主周期 bar 自身方向估算。
        """
        out = pd.DataFrame(index=self.df.index)
        up_mask = self.df["close"] > self.df["open"]
        dn_mask = self.df["close"] < self.df["open"]
        out["bull_vol"] = np.where(up_mask, self.df["vol"], np.where(dn_mask, 0.0, self.df["vol"] * 0.5))
        out["bear_vol"] = np.where(dn_mask, self.df["vol"], np.where(up_mask, 0.0, self.df["vol"] * 0.5))
        return out

    @staticmethod
    def _px_new(p: float, t: pd.Timestamp) -> PxT:
        return PxT(p=float(p), t=t)

    @staticmethod
    def _swbar_empty() -> SwBar:
        na_ts = pd.NaT
        return SwBar(h=PxT(p=np.nan, t=na_ts), l=PxT(p=np.nan, t=na_ts))

    def _detect_pivots(self) -> Tuple[pd.Series, pd.Series]:
        """
        Pine 对应逻辑：
        - 用 swCur/swHist 聚合 swing bars；
        - 在每次新 swing bar 开始时，把上一个 swCur 推入 swHist；
        - 再对 swHist[r] 做左右确认。

        当前 Python 脚本只支持日线主图，等价于 swTf == 主图周期，
        因而每根 bar 自成一个 swing bar，但依然保留 Pine 的状态机路径，
        避免继续使用普通窗口分形实现。
        """
        idx = self.df.index
        ph = pd.Series(np.nan, index=idx, dtype=float)
        pl = pd.Series(np.nan, index=idx, dtype=float)

        left = int(self.args.swing_left)
        right = int(self.args.swing_right)
        keep_n = max(50, left + right + 10)

        sw_hist: List[SwBar] = []
        sw_cur: Optional[SwBar] = None
        sw_inited = False
        last_ph_t = pd.NaT
        last_pl_t = pd.NaT

        for i, ts in enumerate(idx):
            high = float(self.df.iat[i, self.df.columns.get_loc("high")])
            low = float(self.df.iat[i, self.df.columns.get_loc("low")])

            sw_new_tf = i > 0
            if sw_new_tf and sw_inited and sw_cur is not None:
                sw_hist.insert(0, sw_cur)
                if len(sw_hist) > keep_n:
                    sw_hist = sw_hist[:keep_n]
                sw_inited = False
                sw_cur = self._swbar_empty()

            if not sw_inited or sw_cur is None:
                sw_cur = SwBar(h=self._px_new(high, ts), l=self._px_new(low, ts))
                sw_inited = True
            else:
                if np.isnan(sw_cur.h.p) or high > sw_cur.h.p:
                    sw_cur.h = self._px_new(high, ts)
                if np.isnan(sw_cur.l.p) or low < sw_cur.l.p:
                    sw_cur.l = self._px_new(low, ts)

            got_pivot_event = sw_new_tf and len(sw_hist) >= (left + right + 1)
            if not got_pivot_event:
                continue

            cand = sw_hist[right]
            cand_h = cand.h.p
            cand_l = cand.l.p
            is_ph = not np.isnan(cand_h)
            is_pl = not np.isnan(cand_l)

            for j in range(0, right):
                b = sw_hist[j]
                if is_ph and not np.isnan(b.h.p) and cand_h <= b.h.p:
                    is_ph = False
                if is_pl and not np.isnan(b.l.p) and cand_l >= b.l.p:
                    is_pl = False

            for j in range(right + 1, right + left + 1):
                b = sw_hist[j]
                if is_ph and not np.isnan(b.h.p) and cand_h <= b.h.p:
                    is_ph = False
                if is_pl and not np.isnan(b.l.p) and cand_l >= b.l.p:
                    is_pl = False

            if is_ph and (pd.isna(last_ph_t) or cand.h.t != last_ph_t):
                ph.loc[cand.h.t] = cand_h
                last_ph_t = cand.h.t

            if is_pl and (pd.isna(last_pl_t) or cand.l.t != last_pl_t):
                pl.loc[cand.l.t] = cand_l
                last_pl_t = cand.l.t

        return ph, pl

    def run(self) -> None:
        df = self.df
        vol_delta = self._calc_main_tf_volume_delta()
        df = df.join(vol_delta)
        df["total_vol_delta"] = df["bull_vol"].fillna(0.0) + df["bear_vol"].fillna(0.0)
        df["bull_pct"] = np.where(df["total_vol_delta"] > 0, df["bull_vol"] / df["total_vol_delta"], 0.5)
        df["bull_pct"] = df["bull_pct"].clip(0.0, 1.0)
        df["bear_pct"] = np.where(df["total_vol_delta"] > 0, 1.0 - df["bull_pct"], 0.5)
        df["bear_pct"] = df["bear_pct"].clip(0.0, 1.0)
        df["delta_pct"] = df["bull_pct"] - df["bear_pct"]

        ph, pl = self._detect_pivots()
        df["pivot_high"] = ph
        df["pivot_low"] = pl

        # 输出字段初始化
        fields = [
            "nearest_high",
            "nearest_low",
            "dist_to_nearest_high",
            "dist_to_nearest_low",
            "position_in_structure",
            "high_break_flag",
            "low_break_flag",
            "n_high_levels_broken",
            "n_low_levels_broken",
            "last_break_dir",
            "broken_high_price",
            "broken_low_price",
            "break_label_text",
        ]
        for c in fields:
            df[c] = np.nan if c != "break_label_text" else ""

        br_vol_thr = max(0.0, min(1.0, float(self.args.br_vol_filter_pct) / 100.0))
        use_wick = self.args.breakout_by.lower() == "wick"
        last_break_dir = 0

        for i, ts in enumerate(df.index):
            if pd.notna(df.at[ts, "pivot_high"]):
                self.high_levels.append(Level(price=float(df.at[ts, "pivot_high"]), source_index=i, source_time=ts))
                self.high_levels = self._trim_levels(self.high_levels, self.args.max_levels)
            if pd.notna(df.at[ts, "pivot_low"]):
                self.low_levels.append(Level(price=float(df.at[ts, "pivot_low"]), source_index=i, source_time=ts))
                self.low_levels = self._trim_levels(self.low_levels, self.args.max_levels)

            active_highs = [lv for lv in self.high_levels if not lv.mitigated]
            active_lows = [lv for lv in self.low_levels if not lv.mitigated]

            nearest_high = min([lv.price for lv in active_highs], default=np.nan)
            nearest_low = max([lv.price for lv in active_lows], default=np.nan)
            df.at[ts, "nearest_high"] = nearest_high
            df.at[ts, "nearest_low"] = nearest_low
            df.at[ts, "dist_to_nearest_high"] = (nearest_high - df.at[ts, "close"]) / df.at[ts, "close"] if pd.notna(nearest_high) else np.nan
            df.at[ts, "dist_to_nearest_low"] = (df.at[ts, "close"] - nearest_low) / df.at[ts, "close"] if pd.notna(nearest_low) else np.nan
            if pd.notna(nearest_high) and pd.notna(nearest_low) and nearest_high > nearest_low:
                df.at[ts, "position_in_structure"] = (df.at[ts, "close"] - nearest_low) / (nearest_high - nearest_low)

            high_break_count = 0
            low_break_count = 0
            broken_high_price = np.nan
            broken_low_price = np.nan

            # High break：和 Pine 一样，量能过滤失败时直接删除该 level
            remove_high_ids: set[int] = set()
            for lv in active_highs:
                br = df.at[ts, "high"] > lv.price if use_wick else df.at[ts, "close"] > lv.price
                if br:
                    vol_ok = (not self.args.br_vol_filter) or (float(df.at[ts, "bull_pct"]) > br_vol_thr)
                    if self.args.br_vol_filter and not vol_ok:
                        remove_high_ids.add(id(lv))
                    else:
                        lv.mitigated = True
                        lv.broken_index = i
                        high_break_count += 1
                        broken_high_price = lv.price if np.isnan(broken_high_price) else min(broken_high_price, lv.price)
            if remove_high_ids:
                self.high_levels = [lv for lv in self.high_levels if id(lv) not in remove_high_ids]

            # Low break：和 Pine 一样，量能过滤失败时直接删除该 level
            remove_low_ids: set[int] = set()
            for lv in active_lows:
                br = df.at[ts, "low"] < lv.price if use_wick else df.at[ts, "close"] < lv.price
                if br:
                    vol_ok = (not self.args.br_vol_filter) or (float(df.at[ts, "bear_pct"]) > br_vol_thr)
                    if self.args.br_vol_filter and not vol_ok:
                        remove_low_ids.add(id(lv))
                    else:
                        lv.mitigated = True
                        lv.broken_index = i
                        low_break_count += 1
                        broken_low_price = lv.price if np.isnan(broken_low_price) else max(broken_low_price, lv.price)
            if remove_low_ids:
                self.low_levels = [lv for lv in self.low_levels if id(lv) not in remove_low_ids]

            did_high_break = high_break_count > 0
            did_low_break = low_break_count > 0
            if did_high_break:
                last_break_dir = 1
            elif did_low_break:
                last_break_dir = -1

            df.at[ts, "high_break_flag"] = float(did_high_break)
            df.at[ts, "low_break_flag"] = float(did_low_break)
            df.at[ts, "n_high_levels_broken"] = float(high_break_count)
            df.at[ts, "n_low_levels_broken"] = float(low_break_count)
            df.at[ts, "last_break_dir"] = float(last_break_dir)
            df.at[ts, "broken_high_price"] = broken_high_price
            df.at[ts, "broken_low_price"] = broken_low_price

            break_bar = did_high_break or did_low_break
            if break_bar:
                is_bull_break = did_high_break
                txt = f"{self._fmt(df.at[ts, 'bull_vol'])} ({self._pct_str(df.at[ts, 'bull_pct'])}) | {self._fmt(df.at[ts, 'bear_vol'])} ({self._pct_str(df.at[ts, 'bear_pct'])})"
                df.at[ts, "break_label_text"] = txt
                self.break_events.append(
                    BreakEvent(
                        index=i,
                        time=ts,
                        direction="up" if is_bull_break else "down",
                        level_price=float(broken_high_price if is_bull_break else broken_low_price),
                        bull_vol=float(df.at[ts, "bull_vol"]),
                        bear_vol=float(df.at[ts, "bear_vol"]),
                        bull_pct=float(df.at[ts, "bull_pct"]),
                        bear_pct=float(df.at[ts, "bear_pct"]),
                        n_levels_broken=int(high_break_count if is_bull_break else low_break_count),
                    )
                )

        self.df = df

    def build_figure(self, title: str) -> go.Figure:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.68, 0.16, 0.16],
            subplot_titles=(title, "Breakout / Volume Delta", "Position Structure"),
        )

        fig.add_trace(
            go.Candlestick(
                x=self.x,
                open=self.df["open"], high=self.df["high"], low=self.df["low"], close=self.df["close"],
                increasing_line_color="#00c2a0", decreasing_line_color="#ff4d5a",
                increasing_fillcolor="#00c2a0", decreasing_fillcolor="#ff4d5a",
                name="K线", showlegend=False,
            ), row=1, col=1
        )

        # 水平位：未突破与已突破分别画
        for lv in self.high_levels:
            end_idx = lv.broken_index if lv.broken_index is not None else len(self.df) - 1
            fig.add_trace(
                go.Scatter(
                    x=self.x[lv.source_index:end_idx + 1],
                    y=[lv.price] * (end_idx - lv.source_index + 1),
                    mode="lines",
                    line=dict(color=BRK_HI_COL if lv.mitigated else UNMIT_HI_COL, width=1.4, dash="solid" if lv.mitigated else "dot"),
                    name="high_level",
                    legendgroup="levels",
                    showlegend=False,
                    hovertemplate=f"high_level={lv.price:.3f}<extra></extra>",
                ), row=1, col=1
            )

        for lv in self.low_levels:
            end_idx = lv.broken_index if lv.broken_index is not None else len(self.df) - 1
            fig.add_trace(
                go.Scatter(
                    x=self.x[lv.source_index:end_idx + 1],
                    y=[lv.price] * (end_idx - lv.source_index + 1),
                    mode="lines",
                    line=dict(color=BRK_LO_COL if lv.mitigated else UNMIT_LO_COL, width=1.4, dash="solid" if lv.mitigated else "dot"),
                    name="low_level",
                    legendgroup="levels",
                    showlegend=False,
                    hovertemplate=f"low_level={lv.price:.3f}<extra></extra>",
                ), row=1, col=1
            )

        # 突破标记
        up_mask = self.df["high_break_flag"] == 1
        dn_mask = self.df["low_break_flag"] == 1
        if up_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=self.x[up_mask.to_numpy()],
                    y=self.df.loc[up_mask, "close"],
                    mode="markers+text",
                    text=[f"↑{int(v)}" for v in self.df.loc[up_mask, "n_high_levels_broken"]],
                    textposition="top center",
                    marker=dict(symbol="triangle-up", size=11, color=BULL_COL),
                    name="high_break",
                    showlegend=False,
                ), row=1, col=1
            )
        if dn_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=self.x[dn_mask.to_numpy()],
                    y=self.df.loc[dn_mask, "close"],
                    mode="markers+text",
                    text=[f"↓{int(v)}" for v in self.df.loc[dn_mask, "n_low_levels_broken"]],
                    textposition="bottom center",
                    marker=dict(symbol="triangle-down", size=11, color=BEAR_COL),
                    name="low_break",
                    showlegend=False,
                ), row=1, col=1
            )

        # debug label / volume delta text
        for ev in self.break_events:
            y = self.df.iloc[ev.index]["high"] * 1.01 if ev.direction == "up" else self.df.iloc[ev.index]["low"] * 0.99
            fig.add_annotation(
                x=self.x[ev.index], y=y,
                text=f"{self._fmt(ev.bull_vol)} ({self._pct_str(ev.bull_pct)}) | {self._fmt(ev.bear_vol)} ({self._pct_str(ev.bear_pct)})",
                showarrow=False,
                font=dict(size=9, color=BULL_COL if ev.direction == "up" else BEAR_COL),
                bgcolor="rgba(0,0,0,0.0)",
                row=1, col=1,
            )

        # panel 2: delta / breakout info
        for col_name in ["bull_pct", "bear_pct", "delta_pct", "n_high_levels_broken", "n_low_levels_broken"]:
            fig.add_trace(
                go.Scatter(x=self.x, y=self.df[col_name], mode="lines", name=col_name, line=dict(width=1.4), showlegend=False),
                row=2, col=1
            )
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=2, col=1)

        # panel 3: structure info
        for col_name in ["dist_to_nearest_high", "dist_to_nearest_low", "position_in_structure"]:
            fig.add_trace(
                go.Scatter(x=self.x, y=self.df[col_name], mode="lines", name=col_name, line=dict(width=1.4), showlegend=False),
                row=3, col=1
            )
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=3, col=1)
        fig.add_hline(y=1, line_width=1, line_dash="dot", line_color=ZERO_COL, row=3, col=1)

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
        for r in (1, 2, 3):
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True, zeroline=False, row=r, col=1)
            fig.update_yaxes(showgrid=True, zeroline=False, row=r, col=1)
        return fig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Breakout Volume Delta faithful clone (multi timeframe main-bar version)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--freq", default="d", choices=["d", "60m", "30m", "15m", "5m"])
    p.add_argument("--bars", type=int, default=255)
    p.add_argument("--out", default="breakout_volume_delta_600988.html")
    p.add_argument("--csv-out", default="")
    p.add_argument("--swing-left", type=int, default=10)
    p.add_argument("--swing-right", type=int, default=10)
    p.add_argument("--breakout-by", default="Close", choices=["Close", "Wick"])
    p.add_argument("--max-levels", type=int, default=5)
    p.add_argument("--br-vol-filter", action="store_true", default=False)
    p.add_argument("--br-vol-filter-pct", type=float, default=60.0)
    return p


def main() -> None:
    args = build_parser().parse_args()

    client = connect_pytdx()
    df = get_kline_data(client, args.symbol, args.freq, count=args.bars)
    client.disconnect()
    if df is None or len(df) == 0:
        raise RuntimeError(f"获取主周期数据失败: {args.symbol}, freq={args.freq}")

    df = df.rename(columns={"volume": "vol"})
    df = df.set_index("datetime")
    df = df.sort_index()

    engine = BreakoutVolumeDeltaEngine(df, args)
    engine.run()
    fig = engine.build_figure(f"Breakout Volume Delta - {args.symbol} [{args.freq}]")
    plot(fig, filename=args.out, auto_open=False, include_plotlyjs=True)

    if args.csv_out:
        engine.df.to_csv(args.csv_out, encoding="utf-8-sig")
        print(f"CSV 已生成: {args.csv_out}")
    print(f"HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()
