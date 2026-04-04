# -*- coding: utf-8 -*-
"""
ATR Rope + Breakout Volume Delta 合并版 (pytdx/plotly)

Purpose:
    将 ATR Rope 趋势/区间/突破因子 与 Breakout Volume Delta 的结构/突破/量能因子
    合并到一个脚本中，基于同一份 pytdx K 线数据输出统一 HTML 与统一 CSV。

Features:
    1) 主图叠加：K线 + ATR Rope + ATR Channel + Consolidation Range + Swing Levels + Breakout Markers
    2) 因子面板：统一展示 rope 状态、量能 delta、距离/斜率、结构位置、区间/突破强度
    3) 单一数据源：只抓取一次主周期 K 线，避免两个脚本各自取数导致对不齐

How to Run:
    python merged_atr_rope_breakout_volume_delta.py --symbol 600988 --freq d --bars 255 --out merged.html
    python merged_atr_rope_breakout_volume_delta.py --symbol 600988 --freq 15m --bars 500 --csv-out merged.csv
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
BULL_COL = "#00FF6A"
BEAR_COL = "#FF2D55"
UNMIT_HI_COL = "#32CD32"
UNMIT_LO_COL = "#FF4D4F"
BRK_HI_COL = "#9CFF9C"
BRK_LO_COL = "#FF9C9C"


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
    direction: str
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


class MergedEngine:
    def __init__(self, df: pd.DataFrame, args: argparse.Namespace):
        self.df = df.copy()
        self.args = args
        self.len = int(args.len)
        self.multi = float(args.multi)
        self.src_col = args.source
        self.is_intraday = args.freq in {"5m", "15m", "30m", "60m"}
        self.times = list(self.df.index)
        self.x = np.arange(len(self.df), dtype=float)
        self.tick_text = [time_to_str(t, self.is_intraday) for t in self.times]
        self.high_levels: List[Level] = []
        self.low_levels: List[Level] = []
        self.break_events: List[BreakEvent] = []

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
        return levels if len(levels) <= max_lvls else levels[-max_lvls:]

    @staticmethod
    def _px_new(p: float, t: pd.Timestamp) -> PxT:
        return PxT(p=float(p), t=t)

    @staticmethod
    def _swbar_empty() -> SwBar:
        na_ts = pd.NaT
        return SwBar(h=PxT(p=np.nan, t=na_ts), l=PxT(p=np.nan, t=na_ts))

    def _run_atr_rope(self) -> None:
        self.df["src"] = self.df[self.src_col].astype(float)
        self.df["atr_raw"] = atr_pine(self.df, self.len)
        self.df["atr"] = self.df["atr_raw"] * self.multi

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

        d = self.df
        d["rope"] = rope_arr
        d["upper"] = upper_arr
        d["lower"] = lower_arr
        d["rope_dir"] = dir_arr
        d["ff"] = ff_arr
        d["c_hi"] = chi_arr
        d["c_lo"] = clo_arr
        d["is_consolidating"] = is_cons_arr
        d["bars_since_dir_change"] = bars_since_dir_change
        d["consolidation_bars"] = cons_bars_arr
        d["dist_to_rope_atr"] = dist_rope_arr
        d["dist_to_upper_atr"] = dist_upper_arr
        d["dist_to_lower_atr"] = dist_lower_arr
        d["rope_slope_atr_5"] = slope_atr_arr
        d["rope_slope_pct_5"] = slope_pct_arr
        d["range_width_atr"] = width_atr_arr
        d["dist_to_range_high_atr"] = dist_rh_arr
        d["dist_to_range_low_atr"] = dist_rl_arr
        d["range_break_up"] = break_up_arr
        d["range_break_down"] = break_dn_arr
        d["range_break_up_strength"] = break_up_strength_arr
        d["range_break_down_strength"] = break_dn_strength_arr
        d["rope_up"] = d["rope"].where(d["rope_dir"] == 1)
        d["rope_down"] = d["rope"].where(d["rope_dir"] == -1)
        d["rope_flat"] = d["rope"].where(d["rope_dir"] == 0)
        ff_bool = d["ff"] == 1.0
        d["range_high_1"] = d["c_hi"].where(~ff_bool)
        d["range_low_1"] = d["c_lo"].where(~ff_bool)
        d["range_high_2"] = d["c_hi"].where(ff_bool)
        d["range_low_2"] = d["c_lo"].where(ff_bool)
        d["range_width_atr_ffill"] = d["range_width_atr"].ffill()
        d["dist_to_range_high_atr_ffill"] = d["dist_to_range_high_atr"].ffill()
        d["dist_to_range_low_atr_ffill"] = d["dist_to_range_low_atr"].ffill()

    def _detect_pivots(self) -> Tuple[pd.Series, pd.Series]:
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

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        hist = series.shift(1)
        mean = hist.rolling(window, min_periods=window).mean()
        std = hist.rolling(window, min_periods=window).std(ddof=0)
        z = (series - mean) / std.replace(0.0, np.nan)
        return z

    @staticmethod
    def _record_high_days(series: pd.Series, max_lookback: int = 250) -> pd.Series:
        arr = series.to_numpy(dtype=float)
        out = np.full(len(arr), np.nan, dtype=float)
        for i in range(len(arr)):
            cur = arr[i]
            if not np.isfinite(cur) or i == 0:
                continue
            days = 0
            max_back = min(max_lookback, i)
            for n in range(1, max_back + 1):
                prev_max = np.nanmax(arr[i-n:i])
                if np.isfinite(prev_max) and cur > prev_max:
                    days = n
                else:
                    break
            out[i] = float(days)
        return pd.Series(out, index=series.index)

    def _run_breakout(self) -> None:
        d = self.df
        up_mask = d["close"] > d["open"]
        dn_mask = d["close"] < d["open"]
        d["bull_vol"] = np.where(up_mask, d["vol"], np.where(dn_mask, 0.0, d["vol"] * 0.5))
        d["bear_vol"] = np.where(dn_mask, d["vol"], np.where(up_mask, 0.0, d["vol"] * 0.5))
        d["total_vol_delta"] = d["bull_vol"].fillna(0.0) + d["bear_vol"].fillna(0.0)
        d["bull_pct"] = np.where(d["total_vol_delta"] > 0, d["bull_vol"] / d["total_vol_delta"], 0.5)
        d["bull_pct"] = d["bull_pct"].clip(0.0, 1.0)
        d["bear_pct"] = np.where(d["total_vol_delta"] > 0, 1.0 - d["bull_pct"], 0.5)
        d["bear_pct"] = d["bear_pct"].clip(0.0, 1.0)
        d["delta_pct"] = d["bull_pct"] - d["bear_pct"]
        lookback = int(getattr(self.args, "liq_lookback", 20))
        d["vol_zscore"] = self._rolling_zscore(d["vol"], lookback)
        d["amt_zscore"] = self._rolling_zscore(d["amount"], lookback)
        d["vol_record_days"] = self._record_high_days(d["vol"], max_lookback=max(lookback * 12, 250))
        d["amt_record_days"] = self._record_high_days(d["amount"], max_lookback=max(lookback * 12, 250))

        ph, pl = self._detect_pivots()
        d["pivot_high"] = ph
        d["pivot_low"] = pl

        fields = [
            "nearest_high", "nearest_low", "dist_to_nearest_high", "dist_to_nearest_low",
            "position_in_structure", "high_break_flag", "low_break_flag",
            "n_high_levels_broken", "n_low_levels_broken", "last_break_dir",
            "broken_high_price", "broken_low_price", "break_label_text",
        ]
        for c in fields:
            d[c] = np.nan if c != "break_label_text" else ""

        br_vol_thr = max(0.0, min(1.0, float(self.args.br_vol_filter_pct) / 100.0))
        use_wick = self.args.breakout_by.lower() == "wick"
        last_break_dir = 0

        for i, ts in enumerate(d.index):
            if pd.notna(d.at[ts, "pivot_high"]):
                self.high_levels.append(Level(price=float(d.at[ts, "pivot_high"]), source_index=i, source_time=ts))
                self.high_levels = self._trim_levels(self.high_levels, self.args.max_levels)
            if pd.notna(d.at[ts, "pivot_low"]):
                self.low_levels.append(Level(price=float(d.at[ts, "pivot_low"]), source_index=i, source_time=ts))
                self.low_levels = self._trim_levels(self.low_levels, self.args.max_levels)

            active_highs = [lv for lv in self.high_levels if not lv.mitigated]
            active_lows = [lv for lv in self.low_levels if not lv.mitigated]

            nearest_high = min([lv.price for lv in active_highs], default=np.nan)
            nearest_low = max([lv.price for lv in active_lows], default=np.nan)
            d.at[ts, "nearest_high"] = nearest_high
            d.at[ts, "nearest_low"] = nearest_low
            d.at[ts, "dist_to_nearest_high"] = (nearest_high - d.at[ts, "close"]) / d.at[ts, "close"] if pd.notna(nearest_high) else np.nan
            d.at[ts, "dist_to_nearest_low"] = (d.at[ts, "close"] - nearest_low) / d.at[ts, "close"] if pd.notna(nearest_low) else np.nan
            if pd.notna(nearest_high) and pd.notna(nearest_low) and nearest_high > nearest_low:
                d.at[ts, "position_in_structure"] = (d.at[ts, "close"] - nearest_low) / (nearest_high - nearest_low)

            high_break_count = 0
            low_break_count = 0
            broken_high_price = np.nan
            broken_low_price = np.nan

            remove_high_ids: set[int] = set()
            for lv in active_highs:
                br = d.at[ts, "high"] > lv.price if use_wick else d.at[ts, "close"] > lv.price
                if br:
                    vol_ok = (not self.args.br_vol_filter) or (float(d.at[ts, "bull_pct"]) > br_vol_thr)
                    if self.args.br_vol_filter and not vol_ok:
                        remove_high_ids.add(id(lv))
                    else:
                        lv.mitigated = True
                        lv.broken_index = i
                        high_break_count += 1
                        broken_high_price = lv.price if np.isnan(broken_high_price) else min(broken_high_price, lv.price)
            if remove_high_ids:
                self.high_levels = [lv for lv in self.high_levels if id(lv) not in remove_high_ids]

            remove_low_ids: set[int] = set()
            for lv in active_lows:
                br = d.at[ts, "low"] < lv.price if use_wick else d.at[ts, "close"] < lv.price
                if br:
                    vol_ok = (not self.args.br_vol_filter) or (float(d.at[ts, "bear_pct"]) > br_vol_thr)
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

            d.at[ts, "high_break_flag"] = float(did_high_break)
            d.at[ts, "low_break_flag"] = float(did_low_break)
            d.at[ts, "n_high_levels_broken"] = float(high_break_count)
            d.at[ts, "n_low_levels_broken"] = float(low_break_count)
            d.at[ts, "last_break_dir"] = float(last_break_dir)
            d.at[ts, "broken_high_price"] = broken_high_price
            d.at[ts, "broken_low_price"] = broken_low_price

            if did_high_break or did_low_break:
                is_bull_break = did_high_break
                txt = f"{self._fmt(d.at[ts, 'bull_vol'])} ({self._pct_str(d.at[ts, 'bull_pct'])}) | {self._fmt(d.at[ts, 'bear_vol'])} ({self._pct_str(d.at[ts, 'bear_pct'])})"
                d.at[ts, "break_label_text"] = txt
                self.break_events.append(
                    BreakEvent(
                        index=i,
                        time=ts,
                        direction="up" if is_bull_break else "down",
                        level_price=float(broken_high_price if is_bull_break else broken_low_price),
                        bull_vol=float(d.at[ts, "bull_vol"]),
                        bear_vol=float(d.at[ts, "bear_vol"]),
                        bull_pct=float(d.at[ts, "bull_pct"]),
                        bear_pct=float(d.at[ts, "bear_pct"]),
                        n_levels_broken=int(high_break_count if is_bull_break else low_break_count),
                    )
                )

    def run(self) -> None:
        self._run_atr_rope()
        self._run_breakout()
        self._score_breakouts()
        self._generate_trade_signals()
        self.df = self.df.copy()

    @staticmethod
    def _score_bucket(v: float, rules: list[tuple[float, float, float]]) -> float:
        if pd.isna(v):
            return 0.0
        for lo, hi, score in rules:
            if lo <= float(v) < hi:
                return float(score)
        return 0.0

    def _forward_max_return(self, start_idx: int, horizon: int) -> float:
        if start_idx >= len(self.df) - 1:
            return np.nan
        end = min(len(self.df), start_idx + horizon + 1)
        entry = float(self.df.iloc[start_idx]["close"])
        if entry <= 0:
            return np.nan
        future_high = float(self.df.iloc[start_idx + 1:end]["high"].max())
        if not np.isfinite(future_high):
            return np.nan
        return future_high / entry - 1.0

    def _forward_max_drawdown(self, start_idx: int, horizon: int) -> float:
        if start_idx >= len(self.df) - 1:
            return np.nan
        end = min(len(self.df), start_idx + horizon + 1)
        entry = float(self.df.iloc[start_idx]["close"])
        if entry <= 0:
            return np.nan
        future_low = float(self.df.iloc[start_idx + 1:end]["low"].min())
        if not np.isfinite(future_low):
            return np.nan
        return future_low / entry - 1.0

    def _score_breakouts(self) -> None:
        d = self.df
        score_cols = [
            "close_pos", "body_to_range", "upper_wick_ratio",
            "score_bg_rope_slope", "score_bg_dist_to_rope", "score_bg_consolidation",
            "score_candle_close_pos", "score_candle_body_to_range", "score_candle_upper_wick",
            "score_volume_vol_z", "score_volume_vol_record",
            "score_freshness_count", "score_freshness_cum_gain",
            "score_trend_total", "score_candle_total", "score_volume_total", "score_freshness_total",
            "dir_turn_long_flag", "breakout_freshness_count", "breakout_freshness_cum_gain",
            "breakout_quality_score", "breakout_quality_grade",
            "breakout_score_detail", "breakout_watch_flag",
            "fwd_max_ret_5", "fwd_max_ret_10", "fwd_max_ret_20", "breakout_hover_text",
        ]
        for c in score_cols:
            if c in {"breakout_quality_grade", "breakout_score_detail", "breakout_hover_text"}:
                d[c] = ""
            else:
                d[c] = np.nan

        def _clip01(v: float) -> float:
            if pd.isna(v):
                return 0.0
            return float(min(1.0, max(0.0, float(v))))

        def _gauss_peak(x: float, mu: float, sigma: float) -> float:
            if pd.isna(x) or sigma <= 0:
                return 0.0
            z = (float(x) - mu) / sigma
            return _clip01(math.exp(-(z ** 2)))

        def _sigmoid(x: float, mu: float, beta: float) -> float:
            if pd.isna(x) or beta <= 0:
                return 0.0
            z = (float(x) - mu) / beta
            if z >= 60:
                return 1.0
            if z <= -60:
                return 0.0
            return 1.0 / (1.0 + math.exp(-z))

        def _sigmoid_down(x: float, mu: float, beta: float) -> float:
            return 1.0 - _sigmoid(x, mu, beta)

        prev_trigger_idx: Optional[int] = None
        chain_count = 0
        chain_start_close = np.nan

        for i, ts in enumerate(d.index):
            close = float(d.at[ts, "close"])
            high = float(d.at[ts, "high"])
            low = float(d.at[ts, "low"])
            open_ = float(d.at[ts, "open"])
            tr = max(high - low, 1e-9)
            close_pos = (close - low) / tr
            body_to_range = abs(close - open_) / tr
            upper_wick_ratio = (high - max(open_, close)) / tr

            d.at[ts, "close_pos"] = close_pos
            d.at[ts, "body_to_range"] = body_to_range
            d.at[ts, "upper_wick_ratio"] = upper_wick_ratio

            prev_dir = d.iloc[i - 1]["rope_dir"] if i > 0 else np.nan
            curr_dir = d.iloc[i]["rope_dir"]
            dir_turn_long = pd.notna(curr_dir) and float(curr_dir) == 1.0 and pd.notna(prev_dir) and float(prev_dir) in (-1.0, 0.0)
            d.at[ts, "dir_turn_long_flag"] = 1.0 if dir_turn_long else 0.0
            if not dir_turn_long:
                continue

            if prev_trigger_idx is None:
                reset_chain = True
            else:
                gap = i - prev_trigger_idx
                mid = d.iloc[prev_trigger_idx + 1:i]
                too_far = gap > 30
                rope_break = False
                if len(mid) > 0:
                    rope_break = ((mid["rope_dir"] == -1) | (mid["close"] < mid["rope"]).fillna(False)).any()
                reset_chain = too_far or rope_break

            if reset_chain:
                chain_count = 1
                chain_start_close = close
            else:
                chain_count += 1

            prev_trigger_idx = i
            cum_gain = 0.0
            if pd.notna(chain_start_close) and float(chain_start_close) > 0:
                cum_gain = close / float(chain_start_close) - 1.0
            d.at[ts, "breakout_freshness_count"] = float(chain_count)
            d.at[ts, "breakout_freshness_cum_gain"] = float(cum_gain)

            rope_slope = d.at[ts, "rope_slope_atr_5"]
            dist_to_rope = d.at[ts, "dist_to_rope_atr"]
            consolidation_bars = d.at[ts, "consolidation_bars"]
            vol_zscore = d.at[ts, "vol_zscore"]
            vol_record_days = d.at[ts, "vol_record_days"]

            score_rope_slope_n = _sigmoid(rope_slope, 0.15, 0.18)
            score_dist_n = _gauss_peak(dist_to_rope, 0.55, 0.45)
            score_cons_n = _gauss_peak(consolidation_bars, 6.0, 4.0)
            trend_score_n = 0.45 * score_rope_slope_n + 0.35 * score_dist_n + 0.20 * score_cons_n

            score_close_pos_n = _clip01(close_pos) ** 1.2
            score_body_n = _clip01(body_to_range) ** 1.1
            score_upper_wick_n = _sigmoid_down(upper_wick_ratio, 0.28, 0.10)
            candle_score_n = 0.40 * score_close_pos_n + 0.35 * score_body_n + 0.25 * score_upper_wick_n

            score_vol_z_n = _sigmoid(vol_zscore, 1.2, 0.8)
            log_record_days = np.log1p(max(float(vol_record_days), 0.0)) if pd.notna(vol_record_days) else np.nan
            score_vol_record_n = _sigmoid(log_record_days, math.log(11.0), 0.8)
            volume_score_n = 0.65 * score_vol_z_n + 0.35 * score_vol_record_n

            score_fresh_count_n = _sigmoid_down(chain_count, 2.0, 0.8)
            score_fresh_gain_n = _sigmoid_down(cum_gain, 0.08, 0.05)
            fresh_score_n = 0.55 * score_fresh_count_n + 0.45 * score_fresh_gain_n

            total = 100.0 * (0.35 * trend_score_n + 0.25 * candle_score_n + 0.20 * volume_score_n + 0.20 * fresh_score_n)
            grade = "A" if total >= 82 else "B" if total >= 70 else "C" if total >= 60 else "D"

            s_bg_rope = 15.75 * score_rope_slope_n
            s_bg_dist = 12.25 * score_dist_n
            s_bg_cons = 7.00 * score_cons_n
            s_close_pos = 10.00 * score_close_pos_n
            s_body = 8.75 * score_body_n
            s_wick = 6.25 * score_upper_wick_n
            s_vol_z = 13.00 * score_vol_z_n
            s_vol_record = 7.00 * score_vol_record_n
            s_fresh_count = 11.00 * score_fresh_count_n
            s_fresh_gain = 9.00 * score_fresh_gain_n

            d.at[ts, "score_bg_rope_slope"] = s_bg_rope
            d.at[ts, "score_bg_dist_to_rope"] = s_bg_dist
            d.at[ts, "score_bg_consolidation"] = s_bg_cons
            d.at[ts, "score_candle_close_pos"] = s_close_pos
            d.at[ts, "score_candle_body_to_range"] = s_body
            d.at[ts, "score_candle_upper_wick"] = s_wick
            d.at[ts, "score_volume_vol_z"] = s_vol_z
            d.at[ts, "score_volume_vol_record"] = s_vol_record
            d.at[ts, "score_freshness_count"] = s_fresh_count
            d.at[ts, "score_freshness_cum_gain"] = s_fresh_gain
            d.at[ts, "score_trend_total"] = 35.0 * trend_score_n
            d.at[ts, "score_candle_total"] = 25.0 * candle_score_n
            d.at[ts, "score_volume_total"] = 20.0 * volume_score_n
            d.at[ts, "score_freshness_total"] = 20.0 * fresh_score_n
            d.at[ts, "breakout_quality_score"] = total
            d.at[ts, "breakout_quality_grade"] = grade
            d.at[ts, "breakout_watch_flag"] = 1.0 if total >= 70 else 0.0
            d.at[ts, "fwd_max_ret_5"] = self._forward_max_return(i, 5)
            d.at[ts, "fwd_max_ret_10"] = self._forward_max_return(i, 10)
            d.at[ts, "fwd_max_ret_20"] = self._forward_max_return(i, 20)

            detail = (
                f"总分={total:.1f} 等级={grade} | 趋势翻转={35.0 * trend_score_n:.1f} | "
                f"K线质量={25.0 * candle_score_n:.1f} | 量能确认={20.0 * volume_score_n:.1f} | "
                f"新鲜度={20.0 * fresh_score_n:.1f} | 第{int(chain_count)}次触发, 累计涨幅={cum_gain * 100:.1f}%"
            )
            d.at[ts, "breakout_score_detail"] = detail
            hover = (
                f"时间={time_to_str(ts, self.is_intraday)}<br>"
                f"触发=rope_dir 从 -1/0 -> 1<br>"
                f"总分={total:.1f} 等级={grade}<br>"
                f"趋势翻转: rope_slope_atr_5={self._fmt_num(rope_slope)}, dist_to_rope_atr={self._fmt_num(dist_to_rope)}, consolidation_bars={self._fmt_num(consolidation_bars)}<br>"
                f"K线质量: close_pos={close_pos:.3f}, body_to_range={body_to_range:.3f}, upper_wick_ratio={upper_wick_ratio:.3f}<br>"
                f"量能确认: vol_zscore={self._fmt_num(vol_zscore)}, vol_record_days={self._fmt_num(vol_record_days)}<br>"
                f"新鲜度: 第{int(chain_count)}次触发, 累计涨幅={cum_gain * 100:.1f}%<br>"
                f"趋势翻转分: slope={s_bg_rope:.1f}, dist={s_bg_dist:.1f}, cons={s_bg_cons:.1f}<br>"
                f"K线质量分: close={s_close_pos:.1f}, body={s_body:.1f}, wick={s_wick:.1f}<br>"
                f"量能确认分: vol_z={s_vol_z:.1f}, vol_record={s_vol_record:.1f}<br>"
                f"新鲜度分: count={s_fresh_count:.1f}, gain={s_fresh_gain:.1f}<br>"
                f"未来5/10/20最大收益={self._fmt_pct(d.at[ts, 'fwd_max_ret_5'])} / {self._fmt_pct(d.at[ts, 'fwd_max_ret_10'])} / {self._fmt_pct(d.at[ts, 'fwd_max_ret_20'])}<br>"
                f"执行路径={d.at[ts, 'breakout_action'] if 'breakout_action' in d.columns else ''}"
            )
            d.at[ts, "breakout_hover_text"] = hover

    def _generate_trade_signals(self) -> None:
        d = self.df
        signal_cols = [
            "breakout_action", "direct_buy_flag", "pullback_watch_flag",
            "pullback_touch_support_flag", "pullback_hhhl_seen_flag",
            "pullback_close_above_rope_flag", "pullback_dist_ok_flag",
            "pullback_invalid_break_lower_flag", "pullback_buy_flag", "signal_buy_flag", "buy_type",
            "signal_note", "source_breakout_index", "source_breakout_time",
            "dir_turn_upper_price", "dir_turn_atr_raw", "dir_turn_tol_price", "dir_turn_band_low", "dir_turn_band_high",
            "buy_fwd_max_ret_5", "buy_fwd_max_ret_10", "buy_fwd_max_ret_20",
            "buy_fwd_max_dd_5", "buy_fwd_max_dd_10", "buy_fwd_max_dd_20",
        ]
        for c in signal_cols:
            if c in {"breakout_action", "buy_type", "signal_note", "source_breakout_time"}:
                d[c] = ""
            else:
                d[c] = np.nan

        direct_buy_score = 80.0
        touch_tol_atr = 0.2

        pending_breakout_idx: Optional[int] = None

        for i in range(len(d)):
            ts = d.index[i]
            score = d.iloc[i]["breakout_quality_score"]
            is_trigger = d.iloc[i].get("dir_turn_long_flag", np.nan) == 1.0 and pd.notna(score)

            if is_trigger:
                prev_upper = d.iloc[i - 1]["upper"] if i > 0 else np.nan
                prev_atr_raw = d.iloc[i - 1]["atr_raw"] if i > 0 else np.nan
                if pd.notna(prev_upper):
                    d.at[ts, "dir_turn_upper_price"] = float(prev_upper)
                if pd.notna(prev_atr_raw):
                    prev_atr_raw_f = float(prev_atr_raw)
                    d.at[ts, "dir_turn_atr_raw"] = prev_atr_raw_f
                    tol_price = touch_tol_atr * prev_atr_raw_f
                    d.at[ts, "dir_turn_tol_price"] = tol_price
                    if pd.notna(prev_upper):
                        prev_upper_f = float(prev_upper)
                        d.at[ts, "dir_turn_band_low"] = prev_upper_f - tol_price
                        d.at[ts, "dir_turn_band_high"] = prev_upper_f + tol_price
                if float(score) >= direct_buy_score:
                    d.at[ts, "breakout_action"] = "direct_buy"
                    d.at[ts, "direct_buy_flag"] = 1.0
                    d.at[ts, "signal_buy_flag"] = 1.0
                    d.at[ts, "buy_type"] = "dir_turn_direct"
                    d.at[ts, "signal_note"] = f"dir从-1/0变到1，触发分数={float(score):.1f}>=80，当天尾盘直接买入"
                    d.at[ts, "buy_fwd_max_ret_5"] = self._forward_max_return(i, 5)
                    d.at[ts, "buy_fwd_max_ret_10"] = self._forward_max_return(i, 10)
                    d.at[ts, "buy_fwd_max_ret_20"] = self._forward_max_return(i, 20)
                    d.at[ts, "buy_fwd_max_dd_5"] = self._forward_max_drawdown(i, 5)
                    d.at[ts, "buy_fwd_max_dd_10"] = self._forward_max_drawdown(i, 10)
                    d.at[ts, "buy_fwd_max_dd_20"] = self._forward_max_drawdown(i, 20)
                    pending_breakout_idx = None
                    continue
                d.at[ts, "breakout_action"] = "watch_pullback"
                d.at[ts, "pullback_watch_flag"] = 1.0
                d.at[ts, "signal_note"] = f"dir从-1/0变到1，触发分数={float(score):.1f}<80，进入回踩观察"
                pending_breakout_idx = i

            if pending_breakout_idx is None or i <= pending_breakout_idx:
                continue

            lower_now = d.iloc[i]["lower"]
            low_now = d.iloc[i]["low"]
            if pd.notna(lower_now) and pd.notna(low_now) and float(low_now) < float(lower_now):
                d.at[ts, "breakout_action"] = "invalid_break_lower"
                d.at[ts, "pullback_invalid_break_lower_flag"] = 1.0
                d.at[ts, "signal_note"] = "回踩观察失效：当日最低价跌破lower"
                pending_breakout_idx = None
                continue

            rope_dir = d.iloc[i]["rope_dir"]
            if pd.isna(rope_dir) or float(rope_dir) == -1.0:
                continue

            lookback_start = max(pending_breakout_idx + 1, i - 2)
            touch_flag = False
            for j in range(lookback_start, i):
                row = d.iloc[j]
                atr_raw = row["atr_raw"]
                rope = row["rope"]
                lower = row["lower"]
                low = row["low"]
                if pd.isna(atr_raw) or float(atr_raw) <= 0 or pd.isna(low):
                    continue
                tol = touch_tol_atr * float(atr_raw)
                near_rope = pd.notna(rope) and abs(float(low) - float(rope)) <= tol
                near_lower = pd.notna(lower) and abs(float(low) - float(lower)) <= tol
                if near_rope or near_lower:
                    touch_flag = True
                    break
            d.at[ts, "pullback_touch_support_flag"] = 1.0 if touch_flag else 0.0
            if not touch_flag:
                continue

            prev_high = d.iloc[i - 1]["high"]
            prev_low = d.iloc[i - 1]["low"]
            high = d.iloc[i]["high"]
            low = d.iloc[i]["low"]
            hhhl_today = pd.notna(prev_high) and pd.notna(prev_low) and pd.notna(high) and pd.notna(low) and float(high) > float(prev_high) and float(low) > float(prev_low)
            d.at[ts, "pullback_hhhl_seen_flag"] = 1.0 if hhhl_today else 0.0
            if not hhhl_today:
                continue

            src_ts = d.index[pending_breakout_idx]
            src_row = d.iloc[pending_breakout_idx]
            src_score = src_row["breakout_quality_score"]
            d.at[ts, "pullback_buy_flag"] = 1.0
            d.at[ts, "signal_buy_flag"] = 1.0
            d.at[ts, "buy_type"] = "pullback_reentry"
            d.at[ts, "source_breakout_index"] = float(pending_breakout_idx)
            d.at[ts, "source_breakout_time"] = time_to_str(src_ts, self.is_intraday)
            d.at[ts, "breakout_quality_score"] = src_score
            d.at[ts, "score_trend_total"] = src_row.get("score_trend_total", None)
            d.at[ts, "score_candle_total"] = src_row.get("score_candle_total", None)
            d.at[ts, "score_volume_total"] = src_row.get("score_volume_total", None)
            d.at[ts, "score_freshness_total"] = src_row.get("score_freshness_total", None)
            d.at[ts, "signal_note"] = (
                f"源触发分数={float(src_score):.1f}，回踩买入满足: 前2个交易日最低价贴近rope/lower(<=0.1ATR) + 当日出现HH/HL"
            )
            d.at[ts, "buy_fwd_max_ret_5"] = self._forward_max_return(i, 5)
            d.at[ts, "buy_fwd_max_ret_10"] = self._forward_max_return(i, 10)
            d.at[ts, "buy_fwd_max_ret_20"] = self._forward_max_return(i, 20)
            d.at[ts, "buy_fwd_max_dd_5"] = self._forward_max_drawdown(i, 5)
            d.at[ts, "buy_fwd_max_dd_10"] = self._forward_max_drawdown(i, 10)
            d.at[ts, "buy_fwd_max_dd_20"] = self._forward_max_drawdown(i, 20)
            pending_breakout_idx = None

    @staticmethod
    def _fmt_pct(v: float) -> str:
        return "nan" if pd.isna(v) else f"{float(v) * 100:.2f}%"

    @staticmethod
    def _fmt_num(v: float) -> str:
        return "nan" if pd.isna(v) else f"{float(v):.2f}"

    def _add_range_segments(self, fig: go.Figure, hi_col: str, lo_col: str, row: int, col: int, showlegend: bool) -> None:
        hi = self.df[hi_col].to_numpy(dtype=float)
        lo = self.df[lo_col].to_numpy(dtype=float)
        mask = np.isfinite(hi) & np.isfinite(lo)
        runs = segment_runs(mask)
        first = showlegend
        for s, e in runs:
            x = self.x[s:e+1]
            hi_y = hi[s:e+1]
            lo_y = lo[s:e+1]
            fig.add_trace(go.Scatter(x=x, y=hi_y, mode="lines", line=dict(width=1.25, color=RANGE_LINE_COL), name="range", legendgroup="range", showlegend=first, connectgaps=False, hovertemplate="range_high=%{y:.4f}<extra></extra>"), row=row, col=col)
            fig.add_trace(go.Scatter(x=x, y=lo_y, mode="lines", line=dict(width=1.25, color=RANGE_LINE_COL), fill="tonexty", fillcolor=RANGE_FILL_COL, name="range_fill", legendgroup="range", showlegend=False, connectgaps=False, hovertemplate="range_low=%{y:.4f}<extra></extra>"), row=row, col=col)
            first = False

    def build_figure(self, title: str) -> go.Figure:
        if self.args.show_factor_panels:
            fig = make_subplots(
                rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                row_heights=[0.54, 0.10, 0.12, 0.12, 0.12],
                subplot_titles=(
                    title,
                    "ATR Rope State",
                    "Dir翻多连续评分 / 量能",
                    "Distance / Slope",
                    "Score / Forward Return",
                ),
            )
        else:
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=(title,))

        main_row = 1
        fig.add_trace(go.Candlestick(x=self.x, open=self.df["open"], high=self.df["high"], low=self.df["low"], close=self.df["close"], increasing_line_color="#00c2a0", decreasing_line_color="#ff4d5a", increasing_fillcolor="#00c2a0", decreasing_fillcolor="#ff4d5a", name="K线", showlegend=False), row=main_row, col=1)

        for col_name, color, name, lg in [("rope_up", UP_COL, "rope", True), ("rope_down", DOWN_COL, "rope", False), ("rope_flat", FLAT_COL, "rope", False)]:
            fig.add_trace(go.Scatter(x=self.x, y=self.df[col_name], mode="lines", line=dict(width=3.0, color=color), connectgaps=False, name=name, legendgroup="rope", showlegend=lg, hovertemplate="rope=%{y:.4f}<extra></extra>"), row=main_row, col=1)

        if self.args.show_atr_channel:
            fig.add_trace(go.Scatter(x=self.x, y=self.df["upper"], mode="lines", line=dict(width=1.8, color=CHANNEL_LINE_COL), name="upper", showlegend=False, connectgaps=False, hovertemplate="upper=%{y:.4f}<extra></extra>"), row=main_row, col=1)
            fig.add_trace(go.Scatter(x=self.x, y=self.df["lower"], mode="lines", line=dict(width=1.8, color=CHANNEL_LINE_COL), fill="tonexty", fillcolor=CHANNEL_FILL_COL, name="lower", showlegend=False, connectgaps=False, hovertemplate="lower=%{y:.4f}<extra></extra>"), row=main_row, col=1)

        if self.args.show_ranges:
            self._add_range_segments(fig, "range_high_1", "range_low_1", main_row, 1, True)
            self._add_range_segments(fig, "range_high_2", "range_low_2", main_row, 1, False)

        for lv in self.high_levels:
            end_idx = lv.broken_index if lv.broken_index is not None else len(self.df) - 1
            fig.add_trace(go.Scatter(x=self.x[lv.source_index:end_idx + 1], y=[lv.price] * (end_idx - lv.source_index + 1), mode="lines", line=dict(color=BRK_HI_COL if lv.mitigated else UNMIT_HI_COL, width=1.4, dash="solid" if lv.mitigated else "dot"), name="high_level", legendgroup="levels", showlegend=False, hovertemplate=f"high_level={lv.price:.3f}<extra></extra>"), row=main_row, col=1)
        for lv in self.low_levels:
            end_idx = lv.broken_index if lv.broken_index is not None else len(self.df) - 1
            fig.add_trace(go.Scatter(x=self.x[lv.source_index:end_idx + 1], y=[lv.price] * (end_idx - lv.source_index + 1), mode="lines", line=dict(color=BRK_LO_COL if lv.mitigated else UNMIT_LO_COL, width=1.4, dash="solid" if lv.mitigated else "dot"), name="low_level", legendgroup="levels", showlegend=False, hovertemplate=f"low_level={lv.price:.3f}<extra></extra>"), row=main_row, col=1)

        if self.args.show_break_markers:
            up_mask = self.df["range_break_up"] == 1
            dn_mask = self.df["range_break_down"] == 1
            if up_mask.any():
                fig.add_trace(go.Scatter(x=self.x[up_mask.to_numpy()], y=self.df.loc[up_mask, "close"], mode="markers", marker=dict(symbol="circle-open", size=9, color="#2f7cff", line=dict(width=1.5)), name="range_break_up", showlegend=False), row=main_row, col=1)
            if dn_mask.any():
                fig.add_trace(go.Scatter(x=self.x[dn_mask.to_numpy()], y=self.df.loc[dn_mask, "close"], mode="markers", marker=dict(symbol="circle-open", size=9, color="#2f7cff", line=dict(width=1.5)), name="range_break_down", showlegend=False), row=main_row, col=1)


        dir_trigger_mask = self.df.get("dir_turn_long_flag", pd.Series(index=self.df.index, dtype=float)) == 1
        if dir_trigger_mask.any():
            trigger_scores = []
            trigger_hover = []
            for _, row in self.df.loc[dir_trigger_mask].iterrows():
                sc = row.get("breakout_quality_score", np.nan)
                trigger_scores.append("" if pd.isna(sc) else f"{sc:.0f}")
                trigger_hover.append(
                    (row.get("breakout_hover_text", "dir翻多触发") or "dir翻多触发")
                    + f"<br>翻多前一根bar的upper={self._fmt_num(row.get('dir_turn_upper_price', np.nan))}"
                    + f"<br>翻多前一根bar的0.1*ATR={self._fmt_num(row.get('dir_turn_tol_price', np.nan))}"
                    + f"<br>翻多前一根bar价格区间=[{self._fmt_num(row.get('dir_turn_band_low', np.nan))}, {self._fmt_num(row.get('dir_turn_band_high', np.nan))}]"
                )
            fig.add_trace(go.Scatter(
                x=self.x[dir_trigger_mask.to_numpy()],
                y=self.df.loc[dir_trigger_mask, "high"],
                mode="markers+text",
                text=trigger_scores,
                textposition="top center",
                marker=dict(symbol="triangle-up", size=10, color="#00E5FF"),
                name="dir_turn_long",
                showlegend=False,
                customdata=np.array(trigger_hover, dtype=object),
                hovertemplate="%{customdata}<extra></extra>"
            ), row=main_row, col=1)

        buy_direct_mask = self.df.get("direct_buy_flag", pd.Series(index=self.df.index, dtype=float)) == 1
        if buy_direct_mask.any():
            hover_text = []
            for _, row in self.df.loc[buy_direct_mask].iterrows():
                extra = (
                    f"<br>触发分数={self._fmt_num(row.get('breakout_quality_score', np.nan))}"
                    f"<br>未来5/10/20天最大收益={self._fmt_pct(row.get('buy_fwd_max_ret_5', np.nan))} / {self._fmt_pct(row.get('buy_fwd_max_ret_10', np.nan))} / {self._fmt_pct(row.get('buy_fwd_max_ret_20', np.nan))}"
                    f"<br>未来5/10/20天最大回撤={self._fmt_pct(row.get('buy_fwd_max_dd_5', np.nan))} / {self._fmt_pct(row.get('buy_fwd_max_dd_10', np.nan))} / {self._fmt_pct(row.get('buy_fwd_max_dd_20', np.nan))}"
                )
                hover_text.append((row.get("signal_note", "") or "直接买入") + extra)
            fig.add_trace(go.Scatter(
                x=self.x[buy_direct_mask.to_numpy()],
                y=self.df.loc[buy_direct_mask, "low"],
                mode="markers",
                marker=dict(symbol="star", size=13, color="#FFD700"),
                name="direct_buy", showlegend=False,
                customdata=np.array(hover_text, dtype=object),
                hovertemplate="%{customdata}<extra></extra>"
            ), row=main_row, col=1)

        buy_pullback_mask = self.df.get("pullback_buy_flag", pd.Series(index=self.df.index, dtype=float)) == 1
        if buy_pullback_mask.any():
            hover_text = []
            for _, row in self.df.loc[buy_pullback_mask].iterrows():
                src_time = row.get("source_breakout_time", "")
                extra = f"<br>源突破时间={src_time}" if src_time else ""
                extra += (
                    f"<br>未来5/10/20天最大收益={self._fmt_pct(row.get('buy_fwd_max_ret_5', np.nan))} / {self._fmt_pct(row.get('buy_fwd_max_ret_10', np.nan))} / {self._fmt_pct(row.get('buy_fwd_max_ret_20', np.nan))}"
                    f"<br>未来5/10/20天最大回撤={self._fmt_pct(row.get('buy_fwd_max_dd_5', np.nan))} / {self._fmt_pct(row.get('buy_fwd_max_dd_10', np.nan))} / {self._fmt_pct(row.get('buy_fwd_max_dd_20', np.nan))}"
                )
                hover_text.append((row.get("signal_note", "") or "回调买入") + extra)
            fig.add_trace(go.Scatter(
                x=self.x[buy_pullback_mask.to_numpy()],
                y=self.df.loc[buy_pullback_mask, "low"],
                mode="markers",
                marker=dict(symbol="diamond", size=11, color="#FFA500"),
                name="pullback_buy", showlegend=False,
                customdata=np.array(hover_text, dtype=object),
                hovertemplate="%{customdata}<extra></extra>"
            ), row=main_row, col=1)

        if self.args.show_factor_panels:
            for nm in ["rope_dir", "bars_since_dir_change", "is_consolidating", "consolidation_bars"]:
                fig.add_trace(go.Scatter(x=self.x, y=self.df[nm], mode="lines", name=nm, line=dict(width=1.35), connectgaps=False, showlegend=False), row=2, col=1)

            for nm in ["vol_zscore", "vol_record_days", "breakout_quality_score", "score_volume_total", "score_freshness_total"]:
                if nm in self.df.columns:
                    fig.add_trace(go.Scatter(x=self.x, y=self.df[nm], mode="lines", name=nm, line=dict(width=1.35), connectgaps=False, showlegend=False), row=3, col=1)

            for nm in ["dist_to_rope_atr", "dist_to_upper_atr", "dist_to_lower_atr", "rope_slope_atr_5", "rope_slope_pct_5"]:
                fig.add_trace(go.Scatter(x=self.x, y=self.df[nm], mode="lines", name=nm, line=dict(width=1.35), connectgaps=False, showlegend=False), row=4, col=1)

            for nm in ["score_trend_total", "score_candle_total", "score_volume_total", "score_freshness_total", "breakout_quality_score", "signal_buy_flag", "pullback_watch_flag", "fwd_max_ret_5", "fwd_max_ret_10", "fwd_max_ret_20"]:
                if nm in self.df.columns:
                    fig.add_trace(go.Scatter(x=self.x, y=self.df[nm], mode="lines", name=nm, line=dict(width=1.35), connectgaps=False, showlegend=False), row=5, col=1)

            for r in (2, 3, 4, 5):
                fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=r, col=1)
            fig.add_hline(y=1, line_width=1, line_dash="dot", line_color=ZERO_COL, row=5, col=1)

        n = len(self.df)
        tick_step = max(1, n // 10)
        tickvals = list(range(0, n, tick_step))
        if tickvals[-1] != n - 1:
            tickvals.append(n - 1)
        ticktext = [self.tick_text[i] for i in tickvals]

        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=40, r=20, t=80, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.01))
        rows = (1, 2, 3, 4, 5) if self.args.show_factor_panels else (1,)
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merged ATR Rope + Breakout Volume Delta (pytdx/plotly)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--freq", default="d")
    p.add_argument("--bars", type=int, default=255)
    p.add_argument("--out", default="merged_atr_rope_breakout.html")
    p.add_argument("--csv-out", default="")
    p.add_argument("--len", type=int, default=14, dest="len")
    p.add_argument("--multi", type=float, default=1.5)
    p.add_argument("--source", default="close", choices=["open", "high", "low", "close"])
    p.add_argument("--show-ranges", action="store_true", default=True)
    p.add_argument("--hide-ranges", action="store_false", dest="show_ranges")
    p.add_argument("--show-atr-channel", action="store_true", default=True)
    p.add_argument("--hide-atr-channel", action="store_false", dest="show_atr_channel")
    p.add_argument("--show-break-markers", action="store_true", default=True)
    p.add_argument("--hide-break-markers", action="store_false", dest="show_break_markers")
    p.add_argument("--show-factor-panels", action="store_true", default=True)
    p.add_argument("--hide-factor-panels", action="store_false", dest="show_factor_panels")
    p.add_argument("--swing-left", type=int, default=10)
    p.add_argument("--swing-right", type=int, default=10)
    p.add_argument("--breakout-by", default="Close", choices=["Close", "Wick"])
    p.add_argument("--max-levels", type=int, default=5)
    p.add_argument("--br-vol-filter", action="store_true", default=False)
    p.add_argument("--br-vol-filter-pct", type=float, default=60.0)
    p.add_argument("--liq-lookback", type=int, default=20)
    return p


def main() -> None:
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

    engine = MergedEngine(df, args)
    engine.run()
    fig = engine.build_figure(f"Merged ATR Rope + Breakout Volume Delta - {name}({args.symbol}) [{args.freq}]")
    plot(fig, filename=args.out, auto_open=False, include_plotlyjs=True)
    if args.csv_out:
        engine.df.to_csv(args.csv_out, encoding="utf-8-sig")
        print(f"CSV 已生成: {args.csv_out}")
    print(f"HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()
