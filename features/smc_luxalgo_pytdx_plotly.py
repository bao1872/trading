# -*- coding: utf-8 -*-
"""
文件名：
    smc_luxalgo_pytdx_plotly.py

用途：
    尽量按 LuxAlgo 的 Pine v5《Smart Money Concepts》逻辑，
    用 pytdx 数据源 + Plotly 输出 HTML。

已修正的关键点：
1. 修复 leg 状态污染：不同 lane / size 不再共用状态。
2. 修复 trailing extremes：top / bottom 更新逻辑贴近 Pine。
3. 修复 Order Block 取样区间：按 Pine 的 end-exclusive 区间取值。
4. 修复 FVG 右侧延伸与可视化。
5. 修复右侧 category padding，使 Strong/Weak High/Low、OB、FVG 更像 TV。
6. 调整结构标签、EQH/EQL 标签和部分可视化细节。

注意：
1. Plotly 和 TradingView 不是同一个渲染内核，不能保证像素级 1:1。
2. 但这版已经尽量往 Pine 的逐 bar 状态机靠拢。
3. 如果你后面还想继续逼近 TV 截图，可以再针对：
   - 标签偏移
   - 透明度
   - 线宽
   - 右侧留白比例
   继续微调。

依赖环境：
    pip install pandas numpy plotly pytdx

示例用法：
    # 日线，输出到 html
    python smc_luxalgo_pytdx_plotly.py --symbol 600489 --freq d --bars 1000 --out smc.html

    # 周线，接近你截图里的设置
    python smc_luxalgo_pytdx_plotly.py --symbol 600489 --freq w --bars 1000 --out smc_weekly.html

    # 开启 FVG / 日周月前高低点 / 区域
    python smc_luxalgo_pytdx_plotly.py \
        --symbol 600489 \
        --freq d \
        --bars 1200 \
        --out smc_full.html \
        --show-fvg \
        --show-daily-levels \
        --show-weekly-levels \
        --show-monthly-levels \
        --show-zones

    # 只显示最近结构（Present）
    python smc_luxalgo_pytdx_plotly.py \
        --symbol 600489 \
        --freq d \
        --bars 1000 \
        --mode Present \
        --out smc_present.html
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _base_dir not in sys.path:
        sys.path.insert(0, _base_dir)

from datasource.pytdx_client import connect_pytdx, get_kline_data, get_stock_name


BULLISH_LEG = 1
BEARISH_LEG = 0

BULLISH = 1
BEARISH = -1

GREEN = "#089981"
RED = "#F23645"
BLUE = "#2157f3"
GRAY = "#878b94"
MONO_BULLISH = "#b2b5be"
MONO_BEARISH = "#5d606b"

HISTORICAL = "Historical"
PRESENT = "Present"

COLORED = "Colored"
MONOCHROME = "Monochrome"

ALL = "All"
BOS = "BOS"
CHOCH = "CHoCH"

ATR = "Atr"
RANGE = "Cumulative Mean Range"

CLOSE = "Close"
HIGHLOW = "High/Low"

SOLID = "solid"
DASHED = "dash"
DOTTED = "dot"


@dataclass
class Alerts:
    internalBullishBOS: bool = False
    internalBearishBOS: bool = False
    internalBullishCHoCH: bool = False
    internalBearishCHoCH: bool = False
    swingBullishBOS: bool = False
    swingBearishBOS: bool = False
    swingBullishCHoCH: bool = False
    swingBearishCHoCH: bool = False
    internalBullishOrderBlock: bool = False
    internalBearishOrderBlock: bool = False
    swingBullishOrderBlock: bool = False
    swingBearishOrderBlock: bool = False
    equalHighs: bool = False
    equalLows: bool = False
    bullishFairValueGap: bool = False
    bearishFairValueGap: bool = False


@dataclass
class Pivot:
    currentLevel: float = np.nan
    lastLevel: float = np.nan
    crossed: bool = False
    barTime: Optional[pd.Timestamp] = None
    barIndex: Optional[int] = None


@dataclass
class Trend:
    bias: int = 0


@dataclass
class TrailingExtremes:
    top: float = np.nan
    bottom: float = np.nan
    barTime: Optional[pd.Timestamp] = None
    barIndex: Optional[int] = None
    lastTopTime: Optional[pd.Timestamp] = None
    lastBottomTime: Optional[pd.Timestamp] = None


@dataclass
class OrderBlock:
    barHigh: float
    barLow: float
    barTime: pd.Timestamp
    bias: int


@dataclass
class FairValueGap:
    top: float
    bottom: float
    bias: int
    left_time: pd.Timestamp
    right_time: pd.Timestamp


@dataclass
class EqualDisplay:
    line_key: Optional[str] = None
    label_key: Optional[str] = None


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


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return true_range(df).rolling(n, min_periods=1).mean()


def cumulative_mean_range(df: pd.DataFrame) -> pd.Series:
    tr = true_range(df)
    idx = np.arange(1, len(df) + 1)
    return pd.Series(tr.cumsum().values / idx, index=df.index)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "vol": "sum",
                "amount": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def map_tf(tf: str) -> str:
    tf = tf.strip().lower()
    if tf == "":
        return ""
    mp = {
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "d": "1D",
        "w": "1W",
        "m": "1M",
        "mo": "1M",
    }
    if tf not in mp:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return mp[tf]


def time_to_str(ts: pd.Timestamp, is_intraday: bool) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if is_intraday else ts.strftime("%Y-%m-%d")


class DrawBuffer:
    def __init__(self):
        self.lines: Dict[str, dict] = {}
        self.labels: Dict[str, dict] = {}
        self.boxes: Dict[str, dict] = {}

    def set_line(self, key: str, **kwargs):
        self.lines[key] = kwargs

    def delete_line(self, key: Optional[str]):
        if key and key in self.lines:
            del self.lines[key]

    def delete_lines_by_prefix(self, prefix: str):
        for k in [k for k in self.lines if k.startswith(prefix)]:
            del self.lines[k]

    def set_label(self, key: str, **kwargs):
        self.labels[key] = kwargs

    def delete_label(self, key: Optional[str]):
        if key and key in self.labels:
            del self.labels[key]

    def delete_labels_by_prefix(self, prefix: str):
        for k in [k for k in self.labels if k.startswith(prefix)]:
            del self.labels[k]

    def set_box(self, key: str, **kwargs):
        self.boxes[key] = kwargs

    def delete_boxes_by_prefix(self, prefix: str):
        for k in [k for k in self.boxes if k.startswith(prefix)]:
            del self.boxes[k]

    def render(self, fig: go.Figure):
        for _, ln in self.lines.items():
            fig.add_shape(
                type="line",
                x0=ln["x0"],
                y0=ln["y0"],
                x1=ln["x1"],
                y1=ln["y1"],
                line=dict(
                    color=ln["color"],
                    width=ln.get("width", 1.5),
                    dash=ln.get("dash", "solid"),
                ),
                layer=ln.get("layer", "above"),
            )
        for _, bx in self.boxes.items():
            fig.add_shape(
                type="rect",
                x0=bx["x0"],
                x1=bx["x1"],
                y0=bx["y0"],
                y1=bx["y1"],
                line=dict(
                    color=bx.get("line_color", bx["fillcolor"]),
                    width=bx.get("line_width", 0),
                ),
                fillcolor=bx["fillcolor"],
                opacity=bx.get("opacity", 0.25),
                layer=bx.get("layer", "below"),
            )
        for _, lb in self.labels.items():
            fig.add_annotation(
                x=lb["x"],
                y=lb["y"],
                text=lb["text"],
                showarrow=False,
                font=dict(color=lb["color"], size=lb.get("size", 11)),
                xanchor=lb.get("xanchor", "center"),
                yanchor=lb.get("yanchor", "middle"),
                xshift=lb.get("xshift", 0),
                yshift=lb.get("yshift", 0),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
            )


class SMCIndicatorPineCloser:
    def __init__(self, df: pd.DataFrame, args: argparse.Namespace):
        self.df = df.copy()
        self.args = args

        self.df["tr"] = true_range(self.df)
        self.df["atr200"] = atr(self.df, 200)
        self.df["cmr"] = cumulative_mean_range(self.df)
        self.df["volatilityMeasure"] = np.where(
            args.order_block_filter == ATR, self.df["atr200"], self.df["cmr"]
        )
        self.df["highVolatilityBar"] = (
            (self.df["high"] - self.df["low"]) >= (2.0 * self.df["volatilityMeasure"])
        )
        self.df["parsedHigh"] = np.where(self.df["highVolatilityBar"], self.df["low"], self.df["high"])
        self.df["parsedLow"] = np.where(self.df["highVolatilityBar"], self.df["high"], self.df["low"])

        self.swingHigh = Pivot()
        self.swingLow = Pivot()
        self.internalHigh = Pivot()
        self.internalLow = Pivot()
        self.equalHigh = Pivot()
        self.equalLow = Pivot()

        self.swingTrend = Trend()
        self.internalTrend = Trend()

        self.equalHighDisplay = EqualDisplay()
        self.equalLowDisplay = EqualDisplay()

        self.trailing = TrailingExtremes()
        self.currentAlerts = Alerts()

        self.swingOrderBlocks: List[OrderBlock] = []
        self.internalOrderBlocks: List[OrderBlock] = []
        self.fairValueGaps: List[FairValueGap] = []

        self.swing_bullish_choch_times: List[pd.Timestamp] = []
        self.internal_bullish_choch_times: List[pd.Timestamp] = []

        self.times = list(self.df.index)
        self.highs = self.df["high"].tolist()
        self.lows = self.df["low"].tolist()
        self.opens = self.df["open"].tolist()
        self.closes = self.df["close"].tolist()
        self.parsedHighs = self.df["parsedHigh"].tolist()
        self.parsedLows = self.df["parsedLow"].tolist()

        self.is_intraday = args.freq in {"5m", "15m", "30m", "60m"}
        self.xmap = {t: time_to_str(t, self.is_intraday) for t in self.times}
        self.base_x_list = [self.xmap[t] for t in self.times]

        self.right_pad_count = max(8, args.fvg_extend + 6)
        if self.base_x_list:
            self.pad_x_list = [f"__pad_{i}__" for i in range(self.right_pad_count)]
            self.right_edge_x = self.pad_x_list[min(5, len(self.pad_x_list) - 1)]
        else:
            self.pad_x_list = []
            self.right_edge_x = None
        self.all_x_list = self.base_x_list + self.pad_x_list

        self.buffer = DrawBuffer()
        self.leg_states: Dict[Tuple[str, int], Dict[int, int]] = {}

        self.swingBullishColor = MONO_BULLISH if args.style == MONOCHROME else args.swing_bull_color
        self.swingBearishColor = MONO_BEARISH if args.style == MONOCHROME else args.swing_bear_color
        self.internalBullishColor = MONO_BULLISH if args.style == MONOCHROME else args.internal_bull_color
        self.internalBearishColor = MONO_BEARISH if args.style == MONOCHROME else args.internal_bear_color
        self.fvgBullColor = "rgba(178,181,190,0.30)" if args.style == MONOCHROME else args.fvg_bull_color
        self.fvgBearColor = "rgba(93,96,107,0.30)" if args.style == MONOCHROME else args.fvg_bear_color
        self.premiumZoneColor = MONO_BEARISH if args.style == MONOCHROME else args.premium_zone_color
        self.discountZoneColor = MONO_BULLISH if args.style == MONOCHROME else args.discount_zone_color

    def x(self, t: pd.Timestamp) -> str:
        return self.xmap[t]

    def _right_x(self, extra_slots: int = 0) -> str:
        if self.right_edge_x is None:
            return self.base_x_list[-1]
        idx = min(max(0, extra_slots), len(self.pad_x_list) - 1)
        return self.pad_x_list[idx]

    def _highest_after_ref_window(self, ref_i: int, size: int) -> float:
        """
        对应 Pine:
            high[size] > ta.highest(size)

        在当前 bar=i 时，候选 pivot 是 ref_i = i-size。
        ta.highest(size) 实际对应 ref_i 后面直到当前 bar 的这 size 根窗口，
        即 [ref_i+1, i]。
        """
        start = ref_i + 1
        end = ref_i + size + 1  # Python 右开，最终覆盖到 i
        start = max(0, start)
        end = min(len(self.highs), end)

        if start >= end:
            return self.highs[ref_i]
        return max(self.highs[start:end])

    def _lowest_after_ref_window(self, ref_i: int, size: int) -> float:
        """
        对应 Pine:
            low[size] < ta.lowest(size)

        窗口是 ref_i 后面直到当前 bar 的 size 根，
        即 [ref_i+1, i]。
        """
        start = ref_i + 1
        end = ref_i + size + 1
        start = max(0, start)
        end = min(len(self.lows), end)

        if start >= end:
            return self.lows[ref_i]
        return min(self.lows[start:end])

    def leg(self, i: int, size: int, lane: str) -> int:
        """
        尽量按 Pine 的逐 bar 逻辑复刻：

            newLegHigh  = high[size] > ta.highest(size)
            newLegLow   = low[size]  < ta.lowest(size)

        这里 high[size] / low[size] 指的是 ref_i = i-size 那根 bar，
        ta.highest(size) / ta.lowest(size) 指 ref_i 之后到当前 i 的这段窗口。
        """
        if i < size:
            return 0

        state_map = self.leg_states.setdefault((lane, size), {})
        if i in state_map:
            return state_map[i]

        prev = state_map.get(i - 1, 0)
        ref_i = i - size

        new_leg_high = self.highs[ref_i] > self._highest_after_ref_window(ref_i, size)
        new_leg_low = self.lows[ref_i] < self._lowest_after_ref_window(ref_i, size)

        out = prev
        if new_leg_high:
            out = BEARISH_LEG
        elif new_leg_low:
            out = BULLISH_LEG

        state_map[i] = out
        return out

    def start_of_new_leg(self, i: int, size: int, lane: str) -> bool:
        return i > size and self.leg(i, size, lane) != self.leg(i - 1, size, lane)

    def start_of_bearish_leg(self, i: int, size: int, lane: str) -> bool:
        return i > size and (self.leg(i, size, lane) - self.leg(i - 1, size, lane) == -1)

    def start_of_bullish_leg(self, i: int, size: int, lane: str) -> bool:
        return i > size and (self.leg(i, size, lane) - self.leg(i - 1, size, lane) == 1)

    def draw_label(self, key: str, label_time: pd.Timestamp, label_price: float, tag: str, color: str, bullish: bool):
        self.buffer.set_label(
            key,
            x=self.x(label_time),
            y=label_price,
            text=tag,
            color=color,
            size=11,
            yshift=-10 if bullish else 10,
        )

    def draw_equal_high_low(self, piv: Pivot, level: float, size: int, equal_high: bool, current_i: int):
        display = self.equalHighDisplay if equal_high else self.equalLowDisplay
        tag = "EQH" if equal_high else "EQL"
        color = self.swingBearishColor if equal_high else self.swingBullishColor
        line_key = f"eq-line-{'h' if equal_high else 'l'}"
        label_key = f"eq-label-{'h' if equal_high else 'l'}"

        if self.args.mode == PRESENT:
            self.buffer.delete_line(display.line_key)
            self.buffer.delete_label(display.label_key)

        end_time = self.times[current_i - size]
        self.buffer.set_line(
            line_key,
            x0=self.x(piv.barTime),
            y0=piv.currentLevel,
            x1=self.x(end_time),
            y1=level,
            color=color,
            dash=DOTTED,
            width=1.2,
        )

        mid_i = round(0.5 * ((piv.barIndex or 0) + (current_i - size)))
        mid_i = max(0, min(len(self.times) - 1, mid_i))
        self.buffer.set_label(
            label_key,
            x=self.x(self.times[mid_i]),
            y=level,
            text=tag,
            color=color,
            size=10,
            yshift=8 if equal_high else -8,
        )

        display.line_key = line_key
        display.label_key = label_key

    def get_current_structure(self, i: int, size: int, equal_high_low: bool = False, internal: bool = False):
        if i <= size:
            return

        lane = "equal" if equal_high_low else "internal" if internal else "swing"
        new_pivot = self.start_of_new_leg(i, size, lane)
        pivot_low = self.start_of_bullish_leg(i, size, lane)
        pivot_high = self.start_of_bearish_leg(i, size, lane)
        atr_measure = float(self.df.iloc[i]["atr200"])

        if not new_pivot:
            return

        ref_i = i - size

        if pivot_low:
            piv = self.equalLow if equal_high_low else self.internalLow if internal else self.swingLow
            level = self.lows[ref_i]

            if (
                equal_high_low
                and pd.notna(piv.currentLevel)
                and abs(piv.currentLevel - level) < self.args.equal_threshold * atr_measure
            ):
                self.draw_equal_high_low(piv, level, size, False, i)
                self.currentAlerts.equalLows = True

            piv.lastLevel = piv.currentLevel
            piv.currentLevel = level
            piv.crossed = False
            piv.barTime = self.times[ref_i]
            piv.barIndex = ref_i

            if not equal_high_low and not internal:
                self.trailing.bottom = piv.currentLevel
                self.trailing.barTime = piv.barTime
                self.trailing.barIndex = piv.barIndex
                self.trailing.lastBottomTime = piv.barTime

            if self.args.show_swings and not internal and not equal_high_low:
                txt = "LL" if pd.notna(piv.lastLevel) and piv.currentLevel < piv.lastLevel else "HL"
                self.draw_label(f"swing-lbl-{i}-low", piv.barTime, piv.currentLevel, txt, self.swingBullishColor, True)

        elif pivot_high:
            piv = self.equalHigh if equal_high_low else self.internalHigh if internal else self.swingHigh
            level = self.highs[ref_i]

            if (
                equal_high_low
                and pd.notna(piv.currentLevel)
                and abs(piv.currentLevel - level) < self.args.equal_threshold * atr_measure
            ):
                self.draw_equal_high_low(piv, level, size, True, i)
                self.currentAlerts.equalHighs = True

            piv.lastLevel = piv.currentLevel
            piv.currentLevel = level
            piv.crossed = False
            piv.barTime = self.times[ref_i]
            piv.barIndex = ref_i

            if not equal_high_low and not internal:
                self.trailing.top = piv.currentLevel
                self.trailing.barTime = piv.barTime
                self.trailing.barIndex = piv.barIndex
                self.trailing.lastTopTime = piv.barTime

            if self.args.show_swings and not internal and not equal_high_low:
                txt = "HH" if pd.notna(piv.lastLevel) and piv.currentLevel > piv.lastLevel else "LH"
                self.draw_label(f"swing-lbl-{i}-high", piv.barTime, piv.currentLevel, txt, self.swingBearishColor, False)

    def draw_structure(self, prefix: str, piv: Pivot, current_i: int, tag: str, color: str, dashed: bool, bullish: bool):
        if self.args.mode == PRESENT:
            self.buffer.delete_lines_by_prefix(prefix)
            self.buffer.delete_labels_by_prefix(prefix)

        self.buffer.set_line(
            f"{prefix}-line-{current_i}",
            x0=self.x(piv.barTime),
            y0=piv.currentLevel,
            x1=self.x(self.times[current_i]),
            y1=piv.currentLevel,
            color=color,
            dash=DASHED if dashed else SOLID,
            width=1.6,
        )

        mid_i = round(0.5 * ((piv.barIndex or 0) + current_i))
        mid_i = max(0, min(len(self.times) - 1, mid_i))
        self.buffer.set_label(
            f"{prefix}-label-{current_i}",
            x=self.x(self.times[mid_i]),
            y=piv.currentLevel,
            text=tag,
            color=color,
            size=11 if not dashed else 10,
            yshift=-10 if bullish else 10,
        )

    def display_structure(self, i: int, internal: bool = False):
        row = self.df.iloc[i]

        bullish_bar = True
        bearish_bar = True
        if self.args.internal_filter_confluence:
            bullish_bar = (row["high"] - max(row["close"], row["open"])) > min(row["close"], row["open"] - row["low"])
            bearish_bar = (row["high"] - max(row["close"], row["open"])) < min(row["close"], row["open"] - row["low"])

        piv = self.internalHigh if internal else self.swingHigh
        trd = self.internalTrend if internal else self.swingTrend
        extra_condition = (piv.currentLevel != self.swingHigh.currentLevel and bullish_bar) if internal else True
        bullish_color = self.internalBullishColor if internal else self.swingBullishColor

        def _crossover_close(level: float) -> bool:
            return (
                i > 0
                and pd.notna(level)
                and self.closes[i - 1] <= level
                and self.closes[i] > level
            )

        def _crossunder_close(level: float) -> bool:
            return (
                i > 0
                and pd.notna(level)
                and self.closes[i - 1] >= level
                and self.closes[i] < level
            )
        crossed_up = _crossover_close(piv.currentLevel)

        if crossed_up and (not piv.crossed) and extra_condition:
            tag = CHOCH if trd.bias == BEARISH else BOS

            if internal:
                self.currentAlerts.internalBullishCHoCH = tag == CHOCH
                self.currentAlerts.internalBullishBOS = tag == BOS
            else:
                self.currentAlerts.swingBullishCHoCH = tag == CHOCH
                self.currentAlerts.swingBullishBOS = tag == BOS

            piv.crossed = True
            trd.bias = BULLISH

            if tag == CHOCH:
                if internal:
                    self.internal_bullish_choch_times.append(self.times[i])
                else:
                    self.swing_bullish_choch_times.append(self.times[i])

            display_condition = (
                self.args.show_internals
                and (
                    self.args.show_internal_bull == ALL
                    or (self.args.show_internal_bull == BOS and tag != CHOCH)
                    or (self.args.show_internal_bull == CHOCH and tag == CHOCH)
                )
            ) if internal else (
                self.args.show_structure
                and (
                    self.args.show_swing_bull == ALL
                    or (self.args.show_swing_bull == BOS and tag != CHOCH)
                    or (self.args.show_swing_bull == CHOCH and tag == CHOCH)
                )
            )

            if display_condition:
                self.draw_structure(
                    f"{'internal' if internal else 'swing'}-bull",
                    piv,
                    i,
                    tag,
                    bullish_color,
                    internal,
                    True,
                )

            self.store_order_block(piv, i, internal, BULLISH)

        piv = self.internalLow if internal else self.swingLow
        extra_condition = (piv.currentLevel != self.swingLow.currentLevel and bearish_bar) if internal else True
        bearish_color = self.internalBearishColor if internal else self.swingBearishColor

        crossed_down = _crossunder_close(piv.currentLevel)

        if crossed_down and (not piv.crossed) and extra_condition:
            tag = CHOCH if trd.bias == BULLISH else BOS

            if internal:
                self.currentAlerts.internalBearishCHoCH = tag == CHOCH
                self.currentAlerts.internalBearishBOS = tag == BOS
            else:
                self.currentAlerts.swingBearishCHoCH = tag == CHOCH
                self.currentAlerts.swingBearishBOS = tag == BOS

            piv.crossed = True
            trd.bias = BEARISH

            display_condition = (
                self.args.show_internals
                and (
                    self.args.show_internal_bear == ALL
                    or (self.args.show_internal_bear == BOS and tag != CHOCH)
                    or (self.args.show_internal_bear == CHOCH and tag == CHOCH)
                )
            ) if internal else (
                self.args.show_structure
                and (
                    self.args.show_swing_bear == ALL
                    or (self.args.show_swing_bear == BOS and tag != CHOCH)
                    or (self.args.show_swing_bear == CHOCH and tag == CHOCH)
                )
            )

            if display_condition:
                self.draw_structure(
                    f"{'internal' if internal else 'swing'}-bear",
                    piv,
                    i,
                    tag,
                    bearish_color,
                    internal,
                    False,
                )

            self.store_order_block(piv, i, internal, BEARISH)

    def delete_order_blocks(self, i: int, internal: bool = False):
        obs = self.internalOrderBlocks if internal else self.swingOrderBlocks
        row = self.df.iloc[i]
        bearish_src = row["close"] if self.args.order_block_mitigation == CLOSE else row["high"]
        bullish_src = row["close"] if self.args.order_block_mitigation == CLOSE else row["low"]

        kept = []
        for ob in obs:
            crossed = False
            if bearish_src > ob.barHigh and ob.bias == BEARISH:
                crossed = True
                if internal:
                    self.currentAlerts.internalBearishOrderBlock = True
                else:
                    self.currentAlerts.swingBearishOrderBlock = True
            elif bullish_src < ob.barLow and ob.bias == BULLISH:
                crossed = True
                if internal:
                    self.currentAlerts.internalBullishOrderBlock = True
                else:
                    self.currentAlerts.swingBullishOrderBlock = True

            if not crossed:
                kept.append(ob)

        if internal:
            self.internalOrderBlocks = kept
        else:
            self.swingOrderBlocks = kept

    def store_order_block(self, piv: Pivot, current_i: int, internal: bool, bias: int):
        if piv.barIndex is None:
            return
        if internal and not self.args.show_internal_order_blocks:
            return
        if (not internal) and not self.args.show_swing_order_blocks:
            return

        start = piv.barIndex
        end = current_i  # end-exclusive，贴近 Pine 的 slice(pivot.barIndex, bar_index)
        if end <= start:
            return

        if bias == BEARISH:
            arr = self.parsedHighs[start:end]
            if not arr:
                return
            local_idx = int(np.argmax(arr))
        else:
            arr = self.parsedLows[start:end]
            if not arr:
                return
            local_idx = int(np.argmin(arr))

        parsed_index = start + local_idx
        ob = OrderBlock(
            barHigh=float(self.parsedHighs[parsed_index]),
            barLow=float(self.parsedLows[parsed_index]),
            barTime=self.times[parsed_index],
            bias=bias,
        )

        target = self.internalOrderBlocks if internal else self.swingOrderBlocks
        if len(target) >= 100:
            target.pop()
        target.insert(0, ob)

    def draw_order_blocks(self):
        right_x = self._right_x(5)

        def _draw(arr: List[OrderBlock], internal: bool):
            max_count = self.args.internal_ob_size if internal else self.args.swing_ob_size
            use = arr[:max_count]

            if self.args.mode == PRESENT:
                self.buffer.delete_boxes_by_prefix("iob" if internal else "sob")

            for idx, ob in enumerate(use):
                if self.args.style == MONOCHROME:
                    color = "rgba(93,96,107,0.30)" if ob.bias == BEARISH else "rgba(178,181,190,0.30)"
                else:
                    if internal:
                        color = self.args.internal_bear_ob_color if ob.bias == BEARISH else self.args.internal_bull_ob_color
                    else:
                        color = self.args.swing_bear_ob_color if ob.bias == BEARISH else self.args.swing_bull_ob_color

                self.buffer.set_box(
                    f"{'iob' if internal else 'sob'}-{idx}",
                    x0=self.x(ob.barTime),
                    x1=right_x,
                    y0=ob.barLow,
                    y1=ob.barHigh,
                    fillcolor=color,
                    opacity=0.32 if internal else 0.28,
                    line_color=color if not internal else color,
                    line_width=0 if internal else 1,
                )

        if self.args.show_internal_order_blocks:
            _draw(self.internalOrderBlocks, True)
        if self.args.show_swing_order_blocks:
            _draw(self.swingOrderBlocks, False)

    def update_trailing_extremes(self, i: int):
        if self.trailing.barIndex is None:
            return

        row = self.df.iloc[i]

        if pd.isna(self.trailing.top) or row["high"] >= self.trailing.top:
            self.trailing.top = row["high"]
            self.trailing.lastTopTime = self.times[i]

        if pd.isna(self.trailing.bottom) or row["low"] <= self.trailing.bottom:
            self.trailing.bottom = row["low"]
            self.trailing.lastBottomTime = self.times[i]

    def draw_high_low_swings(self):
        right_x = self._right_x(5)

        if pd.notna(self.trailing.top) and self.trailing.lastTopTime is not None:
            self.buffer.set_line(
                "trailing-top",
                x0=self.x(self.trailing.lastTopTime),
                y0=self.trailing.top,
                x1=right_x,
                y1=self.trailing.top,
                color=self.swingBearishColor,
                width=1.2,
            )
            self.buffer.set_label(
                "trailing-top-label",
                x=right_x,
                y=self.trailing.top,
                text="Strong High" if self.swingTrend.bias == BEARISH else "Weak High",
                color=self.swingBearishColor,
                size=10,
                xshift=0,
                xanchor="left",
            )

        if pd.notna(self.trailing.bottom) and self.trailing.lastBottomTime is not None:
            self.buffer.set_line(
                "trailing-bottom",
                x0=self.x(self.trailing.lastBottomTime),
                y0=self.trailing.bottom,
                x1=right_x,
                y1=self.trailing.bottom,
                color=self.swingBullishColor,
                width=1.2,
            )
            self.buffer.set_label(
                "trailing-bottom-label",
                x=right_x,
                y=self.trailing.bottom,
                text="Strong Low" if self.swingTrend.bias == BULLISH else "Weak Low",
                color=self.swingBullishColor,
                size=10,
                xshift=0,
                xanchor="left",
            )

    def draw_zone(self, key: str, label_x: str, label_y: float, top: float, bottom: float, text: str, color: str):
        if self.trailing.barTime is None:
            return

        self.buffer.set_box(
            key + "-box",
            x0=self.x(self.trailing.barTime),
            x1=self.x(self.times[-1]),
            y0=bottom,
            y1=top,
            fillcolor=color,
            opacity=0.18,
            line_color=color,
            line_width=0,
        )
        self.buffer.set_label(
            key + "-label",
            x=label_x,
            y=label_y,
            text=text,
            color=color,
            size=11,
        )

    def draw_premium_discount_zones(self):
        if pd.isna(self.trailing.top) or pd.isna(self.trailing.bottom) or self.trailing.barTime is None:
            return

        mid_i = round(0.5 * ((self.trailing.barIndex or 0) + (len(self.times) - 1)))
        mid_i = max(0, min(len(self.times) - 1, mid_i))
        mid_x = self.x(self.times[mid_i])
        right_x = self._right_x(5)

        self.draw_zone(
            "premium",
            mid_x,
            self.trailing.top,
            self.trailing.top,
            0.95 * self.trailing.top + 0.05 * self.trailing.bottom,
            "Premium",
            self.premiumZoneColor,
        )

        eq = 0.5 * (self.trailing.top + self.trailing.bottom)
        self.draw_zone(
            "equilibrium",
            right_x,
            eq,
            0.525 * self.trailing.top + 0.475 * self.trailing.bottom,
            0.525 * self.trailing.bottom + 0.475 * self.trailing.top,
            "Equilibrium",
            self.args.equilibrium_zone_color,
        )

        self.draw_zone(
            "discount",
            mid_x,
            self.trailing.bottom,
            0.95 * self.trailing.bottom + 0.05 * self.trailing.top,
            self.trailing.bottom,
            "Discount",
            self.discountZoneColor,
        )

    def compute_fvg(self):
        if not self.args.show_fvg:
            return

        tf_rule = map_tf(self.args.fvg_timeframe) if self.args.fvg_timeframe else ""
        fvg_df = self.df if tf_rule == "" else resample_ohlc(self.df, tf_rule)
        if len(fvg_df) < 3:
            return

        bar_delta_percent = ((fvg_df["close"].shift(1) - fvg_df["open"].shift(1)) / (fvg_df["open"].shift(1) * 100.0))
        threshold = (
            bar_delta_percent.abs().expanding().mean() * 2.0
            if self.args.fvg_auto_threshold
            else pd.Series(0.0, index=fvg_df.index)
        )

        new_gaps: List[FairValueGap] = []
        for i in range(2, len(fvg_df)):
            last_close = fvg_df["close"].iloc[i - 1]
            current_high = fvg_df["high"].iloc[i]
            current_low = fvg_df["low"].iloc[i]
            last2_high = fvg_df["high"].iloc[i - 2]
            last2_low = fvg_df["low"].iloc[i - 2]
            delta = bar_delta_percent.iloc[i]
            th = threshold.iloc[i]

            bullish_gap = current_low > last2_high and last_close > last2_high and delta > th
            bearish_gap = current_high < last2_low and last_close < last2_low and (-delta) > th

            if bullish_gap:
                self.currentAlerts.bullishFairValueGap = True
                new_gaps.insert(
                    0,
                    FairValueGap(
                        top=float(current_low),
                        bottom=float(last2_high),
                        bias=BULLISH,
                        left_time=fvg_df.index[i - 1],
                        right_time=fvg_df.index[i],
                    ),
                )

            if bearish_gap:
                self.currentAlerts.bearishFairValueGap = True
                new_gaps.insert(
                    0,
                    FairValueGap(
                        top=float(current_high),
                        bottom=float(last2_low),
                        bias=BEARISH,
                        left_time=fvg_df.index[i - 1],
                        right_time=fvg_df.index[i],
                    ),
                )

        kept: List[FairValueGap] = []
        for gap in new_gaps:
            sub = self.df[self.df.index >= gap.right_time]
            invalid = (sub["low"] < gap.bottom).any() if gap.bias == BULLISH else (sub["high"] > gap.top).any()
            if not invalid:
                kept.append(gap)

        self.fairValueGaps = kept

    def draw_fvg(self):
        if not self.args.show_fvg:
            return

        if self.args.mode == PRESENT:
            self.buffer.delete_boxes_by_prefix("fvg")

        right_x = self._right_x(self.args.fvg_extend)
        for idx, gap in enumerate(self.fairValueGaps):
            color = self.fvgBullColor if gap.bias == BULLISH else self.fvgBearColor
            mid = 0.5 * (gap.top + gap.bottom)
            end_x = right_x if self.args.fvg_extend > 0 else self.x(gap.right_time)

            self.buffer.set_box(
                f"fvg-{idx}-top",
                x0=self.x(gap.left_time),
                x1=end_x,
                y0=mid,
                y1=gap.top,
                fillcolor=color,
                opacity=0.25,
                line_color=color,
                line_width=1,
            )
            self.buffer.set_box(
                f"fvg-{idx}-bot",
                x0=self.x(gap.left_time),
                x1=end_x,
                y0=gap.bottom,
                y1=mid,
                fillcolor=color,
                opacity=0.25,
                line_color=color,
                line_width=1,
            )

    def draw_levels(self, timeframe: str, style: str, color: str):
        if timeframe == "D":
            grp = self.df.resample("1D").agg({"high": "max", "low": "min"}).dropna()
            prefix = "PD"
            search_back = pd.Timedelta(days=5)
        elif timeframe == "W":
            grp = self.df.resample("1W").agg({"high": "max", "low": "min"}).dropna()
            prefix = "PW"
            search_back = pd.Timedelta(days=14)
        elif timeframe == "M":
            grp = self.df.resample("1M").agg({"high": "max", "low": "min"}).dropna()
            prefix = "PM"
            search_back = pd.Timedelta(days=45)
        else:
            return

        if len(grp) < 2:
            return

        prev_period = grp.iloc[-2]
        prev_high = float(prev_period["high"])
        prev_low = float(prev_period["low"])
        prev_period_end = grp.index[-2]
        prev_period_start = prev_period_end - search_back
        sub = self.df.loc[(self.df.index >= prev_period_start) & (self.df.index <= prev_period_end)]
        if len(sub) == 0:
            return

        high_idx = sub["high"].idxmax()
        low_idx = sub["low"].idxmin()
        right_x = self._right_x(5)

        self.buffer.set_line(
            f"{prefix}-H",
            x0=self.x(high_idx),
            y0=prev_high,
            x1=right_x,
            y1=prev_high,
            color=color,
            dash=style,
            width=1.1,
        )
        self.buffer.set_label(
            f"{prefix}-H-L",
            x=right_x,
            y=prev_high,
            text=f"{prefix}H",
            color=color,
            size=10,
            xanchor="left",
        )

        self.buffer.set_line(
            f"{prefix}-L",
            x0=self.x(low_idx),
            y0=prev_low,
            x1=right_x,
            y1=prev_low,
            color=color,
            dash=style,
            width=1.1,
        )
        self.buffer.set_label(
            f"{prefix}-L-L",
            x=right_x,
            y=prev_low,
            text=f"{prefix}L",
            color=color,
            size=10,
            xanchor="left",
        )

    def has_bullish_choch_since(self, start_time: pd.Timestamp) -> bool:
        """检查 start_time 之后是否有任何 bullish CHoCH (swing 或 internal)"""
        return any(t > start_time for t in self.swing_bullish_choch_times) or \
               any(t > start_time for t in self.internal_bullish_choch_times)

    def get_bullish_choch_times(self, internal: bool = False) -> List[pd.Timestamp]:
        """返回所有 bullish CHoCH 发生的时间点"""
        if internal:
            return self.internal_bullish_choch_times
        return self.swing_bullish_choch_times

    def run(self):
        for i in range(len(self.df)):
            self.currentAlerts = Alerts()

            self.get_current_structure(i, self.args.swings_length, False, False)
            self.get_current_structure(i, 5, False, True)

            if self.args.show_equal_hl:
                self.get_current_structure(i, self.args.equal_length, True, False)

            if self.args.show_internals or self.args.show_internal_order_blocks or self.args.show_trend:
                self.display_structure(i, True)

            if self.args.show_structure or self.args.show_swing_order_blocks or self.args.show_high_low_swings:
                self.display_structure(i, False)

            if self.trailing.barIndex is not None:
                self.update_trailing_extremes(i)

            if self.args.show_internal_order_blocks:
                self.delete_order_blocks(i, True)
            if self.args.show_swing_order_blocks:
                self.delete_order_blocks(i, False)

        if self.args.show_fvg:
            self.compute_fvg()
            self.draw_fvg()

        if self.args.show_internal_order_blocks or self.args.show_swing_order_blocks:
            self.draw_order_blocks()

        if self.args.show_high_low_swings:
            self.draw_high_low_swings()

        if self.args.show_zones:
            self.draw_premium_discount_zones()

        if self.args.show_daily_levels:
            self.draw_levels("D", self.args.daily_levels_style, self.args.daily_levels_color)
        if self.args.show_weekly_levels:
            self.draw_levels("W", self.args.weekly_levels_style, self.args.weekly_levels_color)
        if self.args.show_monthly_levels:
            self.draw_levels("M", self.args.monthly_levels_style, self.args.monthly_levels_color)

    def figure(self, title: str) -> go.Figure:
        inc = self.swingBullishColor if self.args.show_trend else GREEN
        dec = self.swingBearishColor if self.args.show_trend else RED

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=self.base_x_list,
                open=self.df["open"],
                high=self.df["high"],
                low=self.df["low"],
                close=self.df["close"],
                increasing_line_color=inc,
                decreasing_line_color=dec,
                increasing_fillcolor=inc,
                decreasing_fillcolor=dec,
                name="K线",
            )
        )

        self.buffer.render(fig)

        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=40, r=110, t=80, b=40),
            xaxis_title="时间",
            yaxis_title="价格",
            showlegend=False,
        )
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=self.all_x_list,
            showgrid=True,
        )
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
    p = argparse.ArgumentParser(description="LuxAlgo Smart Money Concepts - pine closer Python version")

    p.add_argument("--symbol", required=True)
    p.add_argument("--freq", default="d")
    p.add_argument("--bars", type=int, default=1000)
    p.add_argument("--out", default="smc.html")

    p.add_argument("--mode", default=HISTORICAL, choices=[HISTORICAL, PRESENT])
    p.add_argument("--style", default=COLORED, choices=[COLORED, MONOCHROME])
    p.add_argument("--show-trend", action="store_true")

    p.add_argument("--show-internals", action="store_true", default=True)
    p.add_argument("--show-internal-bull", default=ALL, choices=[ALL, BOS, CHOCH])
    p.add_argument("--show-internal-bear", default=ALL, choices=[ALL, BOS, CHOCH])
    p.add_argument("--internal-bull-color", default=GREEN)
    p.add_argument("--internal-bear-color", default=RED)
    p.add_argument("--internal-filter-confluence", action="store_true")

    p.add_argument("--show-structure", action="store_true", default=True)
    p.add_argument("--show-swing-bull", default=ALL, choices=[ALL, BOS, CHOCH])
    p.add_argument("--show-swing-bear", default=ALL, choices=[ALL, BOS, CHOCH])
    p.add_argument("--swing-bull-color", default=GREEN)
    p.add_argument("--swing-bear-color", default=RED)
    p.add_argument("--show-swings", action="store_true")
    p.add_argument("--swings-length", type=int, default=50)
    p.add_argument("--show-high-low-swings", action="store_true", default=True)

    p.add_argument("--show-internal-order-blocks", action="store_true", default=True)
    p.add_argument("--internal-ob-size", type=int, default=5)
    p.add_argument("--show-swing-ob", dest="show_swing_order_blocks", action="store_true", default=False)
    p.add_argument("--swing-ob-size", type=int, default=5)
    p.add_argument("--order-block-filter", default=ATR, choices=[ATR, RANGE])
    p.add_argument("--order-block-mitigation", default=HIGHLOW, choices=[CLOSE, HIGHLOW])
    p.add_argument("--internal-bull-ob-color", default="rgba(49,121,245,0.30)")
    p.add_argument("--internal-bear-ob-color", default="rgba(247,124,128,0.30)")
    p.add_argument("--swing-bull-ob-color", default="rgba(24,72,204,0.30)")
    p.add_argument("--swing-bear-ob-color", default="rgba(178,40,51,0.30)")

    p.add_argument("--show-equal-hl", action="store_true", default=True)
    p.add_argument("--equal-length", type=int, default=3)
    p.add_argument("--equal-threshold", type=float, default=0.1)

    p.add_argument("--show-fvg", action="store_true", default=False)
    p.add_argument("--fvg-auto-threshold", action="store_true", default=True)
    p.add_argument("--fvg-timeframe", default="")
    p.add_argument("--fvg-bull-color", default="rgba(0,255,104,0.30)")
    p.add_argument("--fvg-bear-color", default="rgba(255,0,8,0.30)")
    p.add_argument("--fvg-extend", type=int, default=1)

    p.add_argument("--show-daily-levels", action="store_true", default=False)
    p.add_argument("--daily-levels-style", default=SOLID, choices=[SOLID, DASHED, DOTTED])
    p.add_argument("--daily-levels-color", default=BLUE)

    p.add_argument("--show-weekly-levels", action="store_true", default=False)
    p.add_argument("--weekly-levels-style", default=SOLID, choices=[SOLID, DASHED, DOTTED])
    p.add_argument("--weekly-levels-color", default=BLUE)

    p.add_argument("--show-monthly-levels", action="store_true", default=False)
    p.add_argument("--monthly-levels-style", default=SOLID, choices=[SOLID, DASHED, DOTTED])
    p.add_argument("--monthly-levels-color", default=BLUE)

    p.add_argument("--show-zones", action="store_true", default=False)
    p.add_argument("--premium-zone-color", default=RED)
    p.add_argument("--equilibrium-zone-color", default=GRAY)
    p.add_argument("--discount-zone-color", default=GREEN)

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

    engine = SMCIndicatorPineCloser(df, args)
    engine.run()

    fig = engine.figure(f"LuxAlgo Smart Money Concepts - pine closer - {name}({args.symbol}) [{args.freq}]")
    plot(fig, filename=args.out, auto_open=False, include_plotlyjs=True)
    print(f"✅ HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()