# -*- coding: utf-8 -*-
"""
Zigzag + MACD 趋势/背离检测器（独立可运行）

Purpose
- 直接通过 pytdx 拉取 A 股 K 线（不依赖项目内 datasource 模块）
- 尽量按附件 Pine 脚本逻辑复刻 Zigzag + MACD + 基于枢轴历史的 Supertrend Bias
- 输出交互式 HTML：主图 K 线 + Zigzag + Supertrend + 背离标签，副图显示核心因子
- 可选输出 CSV，便于后续样本研究/回测

How to Run
    python zigzag_macd_divergence_viewer.py --symbol 600547 --freq d --bars 300 \
        --out zigzag_macd_600547.html --csv-out zigzag_macd_600547.csv

    python zigzag_macd_divergence_viewer.py --symbol 300750 --freq 60m --bars 500 \
        --out zigzag_macd_300750_60m.html

Dependencies
    pip install pandas numpy plotly pytdx

说明
- 指标固定使用 MACD（不是 RSI）
- 其余参数默认保持与附件 Pine 代码一致：
  useClosePrices=True, waitForConfirmation=False, pZigzagLength=5,
  oscillatorLength=22, oscillatorShortLength=12, oscillatorLongLength=26,
  supertrendHistory=4, atrPeriods=22, atrMult=1, max_array_size=10
- 该脚本以“逐 bar 顺序计算”为主，尽量贴近 Pine 的数组 / 状态机逻辑。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from pytdx.hq import TdxHq_API
    from pytdx.params import TDXParams
except Exception as exc:  # pragma: no cover
    raise RuntimeError("请先安装 pytdx: pip install pytdx") from exc


# =========================
# pytdx data fetch (standalone)
# =========================
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


# =========================
# Common helpers
# =========================
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
    first = min(length, len(arr))
    init = np.nanmean(arr[:first])
    out[first - 1] = init
    prev = init
    alpha = 1.0 / length
    for i in range(first, len(arr)):
        prev = alpha * arr[i] + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=src.index)


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


def atr_pine(df: pd.DataFrame, length: int) -> pd.Series:
    return pine_rma(true_range(df), length)


def adx_di_pine(df: pd.DataFrame, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    high = df['high']
    low = df['low']
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = true_range(df)
    dm_plus = pd.Series(
        np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0.0), 0.0),
        index=df.index,
        dtype=float,
    )
    dm_minus = pd.Series(
        np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0.0), 0.0),
        index=df.index,
        dtype=float,
    )

    smoothed_tr = pine_rma(tr, length)
    smoothed_dm_plus = pine_rma(dm_plus, length)
    smoothed_dm_minus = pine_rma(dm_minus, length)

    di_plus = 100.0 * smoothed_dm_plus / smoothed_tr.replace(0.0, np.nan)
    di_minus = 100.0 * smoothed_dm_minus / smoothed_tr.replace(0.0, np.nan)
    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0.0, np.nan)
    adx = dx.rolling(length, min_periods=length).mean()
    return di_plus, di_minus, dx, adx


def macd_pine(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = pine_ema(close, fast)
    ema_slow = pine_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = pine_ema(macd_line, signal)
    hist_line = macd_line - signal_line
    return macd_line, signal_line, hist_line


def add_to_array(arr: List, val, max_items: int) -> None:
    arr.insert(0, val)
    if len(arr) > max_items:
        arr.pop()


@dataclass
class ZigzagMacdConfig:
    use_close_prices: bool = True
    wait_for_confirmation: bool = False
    zigzag_length: int = 5
    oscillator_length: int = 22
    oscillator_short_length: int = 12
    oscillator_long_length: int = 26
    supertrend_history: int = 4
    atr_periods: int = 22
    atr_mult: float = 1.0
    max_array_size: int = 10


class ZigzagState:
    def __init__(self) -> None:
        self.zigzagsupertrenddirs: List[int] = [1]
        self.zigzagsupertrend: List[float] = [np.nan, np.nan]  # buyStop, sellStop

        self.pZigzagPivots: List[float] = []
        self.pZigzagPivotBars: List[int] = []
        self.pZigzagPivotDirs: List[int] = []
        self.opZigzagPivots: List[float] = []
        self.opZigzagPivotDirs: List[int] = []

        self.labels: List[Dict] = []
        self.lines: List[Dict] = []


def get_sentiment_by_divergence_and_bias(divergence: int, bias: int) -> Tuple[str, str]:
    if divergence == 1:
        sentiment_color = "#00e676" if bias == 1 else "#ff4d5a"
    elif divergence in (-1, -2):
        sentiment_color = "#a3ff12" if bias == 1 else "#ffaa00"
    else:
        sentiment_color = "#b0bec5"

    if divergence == 0:
        sentiment = "▣"
    elif bias > 0:
        sentiment = "⬆" if divergence == 1 else "⤴" if divergence == -1 else "↗" if divergence == -2 else "⤮"
    else:
        sentiment = "⬇" if divergence == 1 else "⤵" if divergence == -1 else "↘" if divergence == -2 else "⤭"
    return sentiment, sentiment_color


def f_get_divergence(p_dir: int, o_dir: int, s_dir: int) -> Tuple[int, str]:
    divergence = 0
    if p_dir == o_dir:
        if (p_dir % 2 == 0 and int(p_dir / 2) == s_dir) or (p_dir % 2 != 0 and p_dir != s_dir):
            divergence = 1
        else:
            divergence = 0
    elif p_dir == 2 * o_dir and o_dir == s_dir:
        divergence = -1
    elif 2 * p_dir == o_dir and p_dir == -s_dir:
        divergence = -2
    else:
        divergence = 0
    ident = "C" if divergence == 1 else "D" if divergence == -1 else "H" if divergence == -2 else "I"
    return divergence, ident


def f_get_bias(p_dir: int, o_dir: int) -> int:
    if p_dir == 2 and o_dir == 2:
        return 1
    if p_dir == -2 and o_dir == -2:
        return -1
    if p_dir > 0:
        return -1
    if p_dir < 0:
        return 1
    return 0


def compute_zigzag_macd(df: pd.DataFrame, cfg: ZigzagMacdConfig) -> Tuple[pd.DataFrame, List[Dict], List[Dict], List[Dict]]:
    d = df.copy()
    n = len(d)
    start_index = 1 if cfg.wait_for_confirmation else 0
    max_items = cfg.max_array_size + start_index

    macd_line, signal_line, hist_line = macd_pine(
        d["close"],
        cfg.oscillator_short_length,
        cfg.oscillator_long_length,
        cfg.oscillator_short_length,
    )
    oscillator = macd_line.fillna(d["close"]).to_numpy(dtype=float)  # 固定用 MACD
    atr = atr_pine(d, cfg.atr_periods).to_numpy(dtype=float)

    state = ZigzagState()

    supertrend_dir_arr = np.full(n, np.nan)
    buy_stop_arr = np.full(n, np.nan)
    sell_stop_arr = np.full(n, np.nan)

    pzg_event_arr = np.zeros(n, dtype=float)
    pzg_double_arr = np.zeros(n, dtype=float)
    pivot_price_arr = np.full(n, np.nan)
    pivot_bar_arr = np.full(n, np.nan)
    price_dir_arr = np.full(n, np.nan)
    osc_dir_arr = np.full(n, np.nan)
    divergence_arr = np.full(n, np.nan)
    bias_arr = np.full(n, np.nan)
    sentiment_code_arr = np.full(n, np.nan)
    divergence_id_arr = np.array(["" for _ in range(n)], dtype=object)
    sentiment_arr = np.array(["" for _ in range(n)], dtype=object)
    latest_label_arr = np.array(["" for _ in range(n)], dtype=object)

    stats_rows: List[Dict] = []

    high_src = d["close"].to_numpy(dtype=float) if cfg.use_close_prices else d["high"].to_numpy(dtype=float)
    low_src = d["close"].to_numpy(dtype=float) if cfg.use_close_prices else d["low"].to_numpy(dtype=float)

    for i in range(n):
        # ---- supertrend init, exactly as current-bar state update before zigzag() call ----
        dir_prev = state.zigzagsupertrenddirs[0]
        buy_stop_prev = state.zigzagsupertrend[0]
        sell_stop_prev = state.zigzagsupertrend[1]
        tail = state.pZigzagPivots[1 : 1 + cfg.supertrend_history] if len(state.pZigzagPivots) > 1 + cfg.supertrend_history else []
        highest = float(np.max(tail)) if len(tail) > 0 else np.nan
        lowest = float(np.min(tail)) if len(tail) > 0 else np.nan
        atr_diff = atr[i] * cfg.atr_mult if np.isfinite(atr[i]) else np.nan
        new_buy_stop = lowest - atr_diff if np.isfinite(lowest) and np.isfinite(atr_diff) else np.nan
        new_sell_stop = highest + atr_diff if np.isfinite(highest) and np.isfinite(atr_diff) else np.nan

        prev_close = d["close"].iloc[i - 1] if i > 0 else np.nan
        new_dir = dir_prev
        if dir_prev > 0 and np.isfinite(prev_close) and np.isfinite(buy_stop_prev) and prev_close < buy_stop_prev:
            new_dir = -1
        elif dir_prev < 0 and np.isfinite(prev_close) and np.isfinite(sell_stop_prev) and prev_close > sell_stop_prev:
            new_dir = 1

        if new_dir > 0 and np.isfinite(new_buy_stop):
            new_buy_stop = max(buy_stop_prev, new_buy_stop) if np.isfinite(buy_stop_prev) else new_buy_stop
        if new_dir < 0 and np.isfinite(new_sell_stop):
            new_sell_stop = min(sell_stop_prev, new_sell_stop) if np.isfinite(sell_stop_prev) else new_sell_stop

        state.zigzagsupertrenddirs[0] = int(new_dir)
        state.zigzagsupertrend[0] = new_buy_stop
        state.zigzagsupertrend[1] = new_sell_stop
        supertrend_dir_arr[i] = new_dir
        buy_stop_arr[i] = new_buy_stop
        sell_stop_arr[i] = new_sell_stop

        # ---- pivots(length, useAlternativeSource, source) ----
        st = max(0, i - cfg.zigzag_length + 1)
        win_high = high_src[st : i + 1]
        win_low = low_src[st : i + 1]
        phigh = high_src[i] if (len(win_high) > 0 and np.isfinite(high_src[i]) and high_src[i] == np.nanmax(win_high)) else np.nan
        plow = low_src[i] if (len(win_low) > 0 and np.isfinite(low_src[i]) and low_src[i] == np.nanmin(win_low)) else np.nan
        phigh_exists = np.isfinite(phigh)
        plow_exists = np.isfinite(plow)
        phigh_bar = i
        plow_bar = i

        # ---- zigzagcore ----
        p_dir = 1
        new_zg = False
        double_zg = bool(phigh_exists and plow_exists)
        if len(state.pZigzagPivots) >= 1:
            p_dir = int(state.pZigzagPivotDirs[0])
            p_dir = int(p_dir / 2) if p_dir % 2 == 0 else p_dir

        def addnewpivot(value: float, bar: int, dir_in: int) -> None:
            new_dir_local = dir_in
            oth_dir_local = dir_in
            other_val_now = oscillator[bar]
            if len(state.pZigzagPivots) >= 2:
                last_point = state.pZigzagPivots[1]
                new_dir_local = dir_in * 2 if dir_in * value > dir_in * last_point else dir_in

                last_other = state.opZigzagPivots[1]
                oth_dir_local = dir_in * 2 if dir_in * other_val_now > dir_in * last_other else dir_in

            add_to_array(state.pZigzagPivots, value, max_items)
            add_to_array(state.pZigzagPivotBars, bar, max_items)
            add_to_array(state.pZigzagPivotDirs, new_dir_local, max_items)
            add_to_array(state.opZigzagPivots, other_val_now, max_items)
            add_to_array(state.opZigzagPivotDirs, oth_dir_local, max_items)

        if ((p_dir == 1 and phigh_exists) or (p_dir == -1 and plow_exists)) and len(state.pZigzagPivots) >= 1:
            pivot = state.pZigzagPivots.pop(0)
            pivot_bar = state.pZigzagPivotBars.pop(0)
            pivot_dir = state.pZigzagPivotDirs.pop(0)
            _ = state.opZigzagPivots.pop(0)
            _ = state.opZigzagPivotDirs.pop(0)

            value = phigh if p_dir == 1 else plow
            bar = phigh_bar if p_dir == 1 else plow_bar
            use_new_values = (value * pivot_dir) > (pivot * pivot_dir)
            value = value if use_new_values else pivot
            bar = bar if use_new_values else pivot_bar
            new_zg = new_zg or bool(use_new_values)
            addnewpivot(float(value), int(bar), int(p_dir))

        if (p_dir == 1 and plow_exists) or (p_dir == -1 and phigh_exists):
            value = plow if p_dir == 1 else phigh
            bar = plow_bar if p_dir == 1 else phigh_bar
            dir2 = -1 if p_dir == 1 else 1
            new_zg = True
            addnewpivot(float(value), int(bar), int(dir2))

        pzg_event_arr[i] = 1.0 if new_zg else 0.0
        pzg_double_arr[i] = 1.0 if double_zg else 0.0

        # ---- draw_zigzag / labels ----
        def draw_zg_line(idx1: int, idx2: int) -> None:
            if len(state.pZigzagPivots) <= idx2:
                return
            y1 = float(state.pZigzagPivots[idx1])
            y2 = float(state.pZigzagPivots[idx2])
            x1 = int(state.pZigzagPivotBars[idx1])
            x2 = int(state.pZigzagPivotBars[idx2])
            pdir1 = int(state.pZigzagPivotDirs[idx1])
            odir1 = int(state.opZigzagPivotDirs[idx1])
            sdir1 = int(supertrend_dir_arr[x1]) if np.isfinite(supertrend_dir_arr[x1]) else 0
            divergence, divergence_identifier = f_get_divergence(pdir1, odir1, sdir1)
            bias = f_get_bias(pdir1, odir1)
            sentiment, sentiment_color = get_sentiment_by_divergence_and_bias(divergence, bias)

            state.lines.insert(0, {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "p_dir": pdir1, "o_dir": odir1, "s_dir": sdir1,
                "divergence": divergence, "divergence_id": divergence_identifier,
                "bias": bias, "sentiment": sentiment, "sentiment_color": sentiment_color,
            })
            state.labels.insert(0, {
                "x": x1, "y": y1, "text": divergence_identifier,
                "p_dir": pdir1, "o_dir": odir1, "s_dir": sdir1,
                "divergence": divergence, "bias": bias, "sentiment": sentiment,
                "sentiment_color": sentiment_color,
            })
            if len(state.lines) > 500:
                state.lines.pop()
            if len(state.labels) > 500:
                state.labels.pop()

            divergence_arr[x1] = divergence
            bias_arr[x1] = bias
            divergence_id_arr[x1] = divergence_identifier
            sentiment_arr[x1] = sentiment
            sentiment_code_arr[x1] = (
                1.0 if divergence == 1 else -1.0 if divergence == -1 else -2.0 if divergence == -2 else 0.0
            )
            pivot_price_arr[x1] = y1
            pivot_bar_arr[x1] = x1
            price_dir_arr[x1] = pdir1
            osc_dir_arr[x1] = odir1
            latest_label_arr[x1] = divergence_identifier

        if new_zg:
            if double_zg:
                draw_zg_line(start_index + 1, start_index + 2)
            if len(state.pZigzagPivots) >= start_index + 2:
                draw_zg_line(start_index + 0, start_index + 1)

        # snapshot stats like Pine table
        if len(state.pZigzagPivotBars) > start_index:
            last_pivot_bar = int(state.pZigzagPivotBars[start_index])
            l_pivot_dir = int(state.pZigzagPivotDirs[start_index])
            l_osc_dir = int(state.opZigzagPivotDirs[start_index])
            s_dir = int(supertrend_dir_arr[last_pivot_bar]) if np.isfinite(supertrend_dir_arr[last_pivot_bar]) else 0
            divergence, divergence_identifier = f_get_divergence(l_pivot_dir, l_osc_dir, s_dir)
            bias = f_get_bias(l_pivot_dir, l_osc_dir)
            sentiment, _ = get_sentiment_by_divergence_and_bias(divergence, bias)
            stats_rows.append({
                "pivot_bar": last_pivot_bar,
                "pivot_time": d.index[last_pivot_bar],
                "price": state.pZigzagPivots[start_index],
                "p_dir": l_pivot_dir,
                "o_dir": l_osc_dir,
                "s_dir": s_dir,
                "divergence": divergence,
                "divergence_id": divergence_identifier,
                "bias": bias,
                "sentiment": sentiment,
            })

    out = pd.DataFrame(index=d.index)
    out["macd_line"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist_line
    out["supertrend_dir"] = supertrend_dir_arr
    out["buy_stop"] = buy_stop_arr
    out["sell_stop"] = sell_stop_arr
    out["pzg_event"] = pzg_event_arr
    out["pzg_double"] = pzg_double_arr
    out["pivot_price"] = pivot_price_arr
    out["pivot_bar"] = pivot_bar_arr
    out["price_pivot_dir"] = price_dir_arr
    out["osc_pivot_dir"] = osc_dir_arr
    out["divergence"] = divergence_arr
    out["bias"] = bias_arr
    out["sentiment_code"] = sentiment_code_arr
    out["divergence_id"] = divergence_id_arr
    out["sentiment"] = sentiment_arr
    out["latest_label"] = latest_label_arr

    # 最近一次标签前向填充，便于观察阶段
    out["latest_label_ffill"] = out["latest_label"].replace("", np.nan).ffill().fillna("")
    out["divergence_id_ffill"] = out["divergence_id"].replace("", np.nan).ffill().fillna("")

    di_plus, di_minus, dx, adx = adx_di_pine(d, length=14)
    out["di_plus"] = di_plus
    out["di_minus"] = di_minus
    out["dx"] = dx
    out["adx"] = adx
    out["adx_threshold"] = 20.0

    # 建 stats 表（取最近 max_array_size+start_index 条唯一 pivot）
    stats_unique = []
    seen = set()
    for row in reversed(stats_rows):
        key = (row["pivot_bar"], row["divergence_id"])
        if key in seen:
            continue
        seen.add(key)
        stats_unique.append(row)
    stats_unique = list(reversed(stats_unique))[: max_items]

    return out, state.lines, state.labels, stats_unique


def build_figure(df: pd.DataFrame, zigzag_lines: List[Dict], zigzag_labels: List[Dict], stats_rows: List[Dict], out_html: str, title: str) -> None:
    x_num = np.arange(len(df), dtype=float)
    intraday = len(df.index) > 1 and (df.index[1] - df.index[0]) < pd.Timedelta("20H")
    tick_text = [ts.strftime("%Y-%m-%d %H:%M") if intraday else ts.strftime("%Y-%m-%d") for ts in df.index]

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.42, 0.15, 0.15, 0.14, 0.14],
        subplot_titles=(
            title,
            "MACD",
            "ADX / DI",
            "Pivot / Divergence / Bias",
            "Supertrend / Trigger",
        ),
    )

    # 主图
    fig.add_trace(
        go.Candlestick(
            x=x_num,
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color="#00c2a0", decreasing_line_color="#ff4d5a",
            increasing_fillcolor="#00c2a0", decreasing_fillcolor="#ff4d5a",
            name="K线", showlegend=False,
        ),
        row=1, col=1,
    )

    up_mask = df["supertrend_dir"] > 0
    dn_mask = df["supertrend_dir"] < 0
    fig.add_trace(go.Scatter(x=x_num, y=df["buy_stop"].where(up_mask), mode="lines", line=dict(width=2, color="#00e676"), name="Supertrend BuyStop", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["sell_stop"].where(dn_mask), mode="lines", line=dict(width=2, color="#ffaa00"), name="Supertrend SellStop", showlegend=True), row=1, col=1)

    for ln in reversed(zigzag_lines):
        fig.add_trace(
            go.Scatter(
                x=[ln["x1"], ln["x2"]], y=[ln["y1"], ln["y2"]], mode="lines",
                line=dict(width=1.8, color="rgba(251, 192, 45, 1)"),
                name="Zigzag", showlegend=False,
                hovertemplate=(
                    f"{ln['divergence_id']} | div={ln['divergence']} | bias={ln['bias']} | "
                    f"pDir={ln['p_dir']} | oDir={ln['o_dir']} | sDir={ln['s_dir']}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    for lb in reversed(zigzag_labels):
        ay = -25 if lb["p_dir"] > 0 else 25
        fig.add_annotation(
            x=lb["x"], y=lb["y"], xref="x", yref="y1",
            text=lb["text"], showarrow=True, arrowhead=2, ax=0, ay=ay,
            bgcolor=lb["sentiment_color"],
            font=dict(color="black", size=11),
        )

    # MACD
    fig.add_trace(go.Bar(x=x_num, y=df["macd_hist"], name="MACD Hist", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["macd_line"], mode="lines", line=dict(width=1.6), name="MACD", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["macd_signal"], mode="lines", line=dict(width=1.2), name="Signal", showlegend=False), row=2, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#90a4ae", row=2, col=1)

    # ADX / DI
    fig.add_trace(go.Scatter(x=x_num, y=df["di_plus"], mode="lines", line=dict(width=1.4, color="#00e676"), name="DI+", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["di_minus"], mode="lines", line=dict(width=1.4, color="#ff4d5a"), name="DI-", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["adx"], mode="lines", line=dict(width=1.6, color="#5c6bc0"), name="ADX", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["adx_threshold"], mode="lines", line=dict(width=1.0, dash="dot", color="#ffffff"), name="ADX阈值20", showlegend=False), row=3, col=1)

    # pivot/divergence/bias
    fig.add_trace(go.Scatter(x=x_num, y=df["price_pivot_dir"], mode="lines+markers", line=dict(width=1.2), marker=dict(size=4), name="price_pivot_dir", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["osc_pivot_dir"], mode="lines+markers", line=dict(width=1.2), marker=dict(size=4), name="osc_pivot_dir", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["divergence"], mode="lines+markers", line=dict(width=1.2), marker=dict(size=4), name="divergence", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["bias"], mode="lines+markers", line=dict(width=1.2), marker=dict(size=4), name="bias", showlegend=False), row=4, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#90a4ae", row=4, col=1)
    fig.update_yaxes(range=[-2.4, 2.4], tickmode="array", tickvals=[-2, -1, 0, 1, 2], ticktext=["-2", "-1", "0", "1", "2"], row=4, col=1)

    # supertrend/trigger
    fig.add_trace(go.Scatter(x=x_num, y=df["supertrend_dir"], mode="lines", line=dict(width=1.4), name="supertrend_dir", showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["pzg_event"], mode="lines", line=dict(width=1.2), name="pzg_event", showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["pzg_double"], mode="lines", line=dict(width=1.2, dash="dot"), name="pzg_double", showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df["sentiment_code"], mode="lines+markers", line=dict(width=1.2), marker=dict(size=4), name="sentiment_code", showlegend=False), row=5, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="#90a4ae", row=5, col=1)

    # 右上角模拟表格
    if stats_rows:
        table_df = pd.DataFrame(stats_rows)
        table_df = table_df.copy()
        table_df["pivot_time"] = pd.to_datetime(table_df["pivot_time"]).dt.strftime("%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d")
        table = go.Table(
            domain=dict(x=[0.73, 0.995], y=[0.74, 0.995]),
            header=dict(values=["Bar Time", "Price", "Osc", "Trend", "Sentiment"], fill_color="black", font=dict(color="white", size=10), align="left"),
            cells=dict(
                values=[
                    table_df["pivot_time"],
                    table_df["p_dir"],
                    table_df["o_dir"],
                    table_df["s_dir"],
                    table_df["divergence_id"].astype(str) + " " + table_df["sentiment"].astype(str),
                ],
                align="left",
                font=dict(size=10),
            ),
        )
        fig.add_trace(table)

    tick_step = max(1, len(df) // 10)
    tickvals = list(range(0, len(df), tick_step))
    if tickvals[-1] != len(df) - 1:
        tickvals.append(len(df) - 1)
    ticktext = [tick_text[i] for i in tickvals]

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        height=1750,
    )
    for r in [1, 2, 3, 4, 5]:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True, zeroline=False, row=r, col=1)
        fig.update_yaxes(showgrid=True, zeroline=False, row=r, col=1)
    fig.write_html(out_html, include_plotlyjs="cdn")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zigzag + MACD 趋势/背离检测器")
    p.add_argument("--symbol", required=True, help="A股代码，如 600547")
    p.add_argument("--freq", default="d", help="d/w/mo/1m/5m/15m/30m/60m")
    p.add_argument("--bars", type=int, default=300, help="展示最近 N 根K线")
    p.add_argument("--fetch-bars", type=int, default=1200, help="实际抓取并用于计算的K线数量，建议 > bars")
    p.add_argument("--out", default="zigzag_macd_divergence.html", help="输出HTML文件")
    p.add_argument("--csv-out", default="", help="可选：输出CSV文件")
    p.add_argument("--zigzag-length", type=int, default=5)
    p.add_argument("--wait-for-confirmation", action="store_true")
    p.add_argument("--supertrend-history", type=int, default=4)
    p.add_argument("--atr-periods", type=int, default=22)
    p.add_argument("--atr-mult", type=float, default=1.0)
    p.add_argument("--max-array-size", type=int, default=10)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.freq = normalize_freq(args.freq)
    fetch_bars = max(args.fetch_bars, args.bars, 300)
    df = fetch_kline_pytdx(args.symbol, args.freq, fetch_bars)

    zz_df, zigzag_lines, zigzag_labels, stats_rows = compute_zigzag_macd(
        df,
        ZigzagMacdConfig(
            use_close_prices=True,
            wait_for_confirmation=bool(args.wait_for_confirmation),
            zigzag_length=args.zigzag_length,
            oscillator_length=22,
            oscillator_short_length=12,
            oscillator_long_length=26,
            supertrend_history=args.supertrend_history,
            atr_periods=args.atr_periods,
            atr_mult=args.atr_mult,
            max_array_size=args.max_array_size,
        ),
    )

    merged = pd.concat([df, zz_df], axis=1).tail(args.bars).copy()

    # 只保留展示窗口内的线和标签
    offset = len(df) - len(merged)
    filtered_lines: List[Dict] = []
    for ln in zigzag_lines:
        if ln["x1"] >= offset and ln["x2"] >= offset:
            ln2 = dict(ln)
            ln2["x1"] -= offset
            ln2["x2"] -= offset
            filtered_lines.append(ln2)
    filtered_labels: List[Dict] = []
    for lb in zigzag_labels:
        if lb["x"] >= offset:
            lb2 = dict(lb)
            lb2["x"] -= offset
            filtered_labels.append(lb2)

    title = f"{args.symbol} [{args.freq}] Zigzag + MACD Trend/Divergence Detector"
    build_figure(merged, filtered_lines, filtered_labels, stats_rows, args.out, title)

    if args.csv_out:
        merged.to_csv(args.csv_out, encoding="utf-8-sig")
        print(f"CSV 已生成: {args.csv_out}")
    print(f"HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()
