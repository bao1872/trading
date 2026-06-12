# -*- coding: utf-8 -*-
"""
atr_rope_event_factor_lab.py

TradingView Pine 指标 ATR Rope（© SamRecio, MPL-2.0）的 Python / Plotly 复刻版 + 事件/因子实验版。

核心复刻内容：
1) ATR = ta.atr(len) * multi，默认 len=14, multi=1.5。
2) rope_smoother：价格偏离 rope 超过 ATR 阈值后，rope 才移动。
3) dir 状态：
   -  1：rope 上升，绿色；
   - -1：rope 下降，红色；
   -  0：source 与 rope 发生 cross，蓝色震荡。
4) Consolidation Ranges：
   在 dir == 0 时累计 upper/lower 的平均值，绘制震荡箱体。
5) 支持 pytdx 拉取 A 股 K 线，也支持本地 CSV。
6) 输出指标 CSV + Plotly HTML。
7) 按功能归类提取事件与因子：方向转换、Rope触碰/穿越、蓝色震荡区上下轨触碰/突破、方向/偏离/区间位置/区间宽度因子。
8) 新增 20bar 默认大背景 regime 因子：多头/空头/震荡背景。

依赖：
    pip install pandas numpy plotly pytdx

示例：
    python atr_rope_samrecio.py --symbol 300133 --freq d --bars 300 --fetch-bars 1200
    python atr_rope_samrecio.py --symbol 300133 --freq 60m --bars 300
    python atr_rope_samrecio.py --csv your_kline.csv --out-html atr_rope.html --out-csv atr_rope.csv

CSV 需要包含：datetime/open/high/low/close/volume；amount 可选。
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.*", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*", category=FutureWarning)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    go = None
    make_subplots = None

try:
    from pytdx.hq import TdxHq_API
    from pytdx.params import TDXParams
except Exception:  # pragma: no cover
    TdxHq_API = None
    TDXParams = None


SERVERS = [
    ("119.147.212.81", 7709), ("119.147.164.60", 7709), ("14.215.128.18", 7709),
    ("14.215.128.116", 7709), ("101.133.156.38", 7709), ("114.80.149.19", 7709),
    ("115.238.90.165", 7709), ("123.125.108.23", 7709), ("180.153.18.170", 7709),
    ("202.108.253.131", 7709),
]


# =============================================================================
# 数据读取：沿用上传框架中的 pytdx / CSV 风格
# =============================================================================
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
    if TDXParams is None:
        raise RuntimeError("请先安装 pytdx: pip install pytdx")
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


def connect_pytdx() -> "TdxHq_API":
    if TdxHq_API is None:
        raise RuntimeError("请先安装 pytdx: pip install pytdx")
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
                d["datetime"] = pd.to_datetime(d[["year", "month", "day", "hour", "minute"]].astype(int))
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


def read_kline_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.lower().strip() for c in df.columns})
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        df = df.drop(columns=["datetime"])
        df.index = dt
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"])
        df.index = dt
    else:
        first = df.columns[0]
        maybe_dt = pd.to_datetime(df[first], errors="coerce")
        if maybe_dt.notna().mean() > 0.8:
            df = df.drop(columns=[first])
            df.index = maybe_dt
        else:
            raise ValueError("CSV 需要 datetime/date 列，或第一列为日期时间")
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    if "amount" not in df.columns:
        df["amount"] = np.nan
    return df[["open", "high", "low", "close", "volume", "amount"]].sort_index().astype(float)


# =============================================================================
# ATR Rope 计算：按 Pine 逻辑逐 bar 复刻
# =============================================================================
@dataclass
class ATRRopeConfig:
    length: int = 14
    multi: float = 1.5
    source: str = "close"
    show_ranges: bool = True
    show_atr_channel: bool = False
    up_color: str = "#3daa45"
    down_color: str = "#ff033e"
    flat_color: str = "#004d92"
    range_color: str = "rgba(0,77,146,0.20)"
    regime_lookback: int = 20
    regime_threshold: float = 0.55


def _pine_true_range(df: pd.DataFrame) -> np.ndarray:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    pc = np.roll(c, 1)
    pc[0] = np.nan
    tr = np.nanmax(np.vstack([(h - l), np.abs(h - pc), np.abs(l - pc)]), axis=0)
    # TradingView ta.tr(true) 在首根没有 close[1] 时使用 high-low。
    if len(tr) > 0 and not np.isfinite(tr[0]):
        tr[0] = h[0] - l[0]
    return tr


def _pine_rma(values: np.ndarray, length: int) -> np.ndarray:
    """TradingView ta.rma 近似等价实现：首个有效值为前 length 个有效值的 SMA，然后递推。"""
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0:
        return out
    vals = np.asarray(values, dtype=float)
    valid_idx = np.where(np.isfinite(vals))[0]
    if len(valid_idx) < length:
        return out
    start = valid_idx[length - 1]
    window_idx = valid_idx[:length]
    out[start] = float(np.nanmean(vals[window_idx]))
    alpha = 1.0 / length
    for i in range(start + 1, n):
        if np.isfinite(vals[i]):
            out[i] = alpha * vals[i] + (1.0 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def _ta_cross(a: np.ndarray, b: np.ndarray, i: int) -> bool:
    """Pine ta.cross(a,b): crossover 或 crossunder。"""
    if i <= 0:
        return False
    if not (np.isfinite(a[i]) and np.isfinite(b[i]) and np.isfinite(a[i - 1]) and np.isfinite(b[i - 1])):
        return False
    return (a[i] > b[i] and a[i - 1] <= b[i - 1]) or (a[i] < b[i] and a[i - 1] >= b[i - 1])


def compute_atr_rope(df: pd.DataFrame, cfg: ATRRopeConfig) -> pd.DataFrame:
    """计算 ATR Rope，并按功能分组输出事件与因子。

    字段命名规则：
    - 原始指标：atr_rope_*。
    - 方向事件：evt_atr_rope_dir_*。
    - Rope 趋势线事件：evt_atr_rope_line_*。
    - 蓝色震荡区事件：evt_atr_rope_range_*。
    - 状态/趋势线因子：factor_atr_rope_state_* / factor_atr_rope_line_*。
    - 大背景因子/事件：factor_atr_rope_regime_* / evt_atr_rope_regime_*。
    - 蓝色震荡区因子：factor_atr_rope_range_*。
    """
    out = df.copy().sort_index()
    for col in ["open", "high", "low", "close"]:
        if col not in out.columns:
            raise ValueError(f"df 缺少字段: {col}")
    if cfg.source not in out.columns:
        raise ValueError(f"source={cfg.source!r} 不在数据列中，可用列: {list(out.columns)}")

    src = out[cfg.source].to_numpy(float)
    src = np.nan_to_num(src, nan=0.0)  # Pine: nz(input.source(close))
    o = out["open"].to_numpy(float)
    h = out["high"].to_numpy(float)
    l = out["low"].to_numpy(float)
    c = out["close"].to_numpy(float)
    n = len(out)

    tr = _pine_true_range(out)
    atr_raw = _pine_rma(tr, int(cfg.length))
    atr = atr_raw * float(cfg.multi)

    rope = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    dir_arr = np.zeros(n, dtype=int)

    c_hi = np.full(n, np.nan)
    c_lo = np.full(n, np.nan)
    h_sum = 0.0
    l_sum = 0.0
    c_count = 0
    ff = True
    ff_arr = np.full(n, False)

    # 方向事件：三色线转换
    evt_dir_change = np.zeros(n, dtype=bool)
    evt_red_to_blue = np.zeros(n, dtype=bool)
    evt_blue_to_green = np.zeros(n, dtype=bool)
    evt_green_to_blue = np.zeros(n, dtype=bool)
    evt_blue_to_red = np.zeros(n, dtype=bool)
    evt_red_to_green = np.zeros(n, dtype=bool)
    evt_green_to_red = np.zeros(n, dtype=bool)
    evt_turn_up = np.zeros(n, dtype=bool)
    evt_turn_down = np.zeros(n, dtype=bool)
    evt_turn_flat = np.zeros(n, dtype=bool)

    # Rope 趋势线事件
    evt_cross_rope = np.zeros(n, dtype=bool)  # Pine ta.cross(src, rope)，保留原始事件
    evt_line_touch_rope = np.zeros(n, dtype=bool)
    evt_line_cross_up = np.zeros(n, dtype=bool)
    evt_line_cross_down = np.zeros(n, dtype=bool)
    evt_line_retest_green = np.zeros(n, dtype=bool)
    evt_line_retest_red = np.zeros(n, dtype=bool)

    # 蓝色区间事件
    evt_range_start = np.zeros(n, dtype=bool)
    evt_range_touch_high = np.zeros(n, dtype=bool)
    evt_range_touch_low = np.zeros(n, dtype=bool)
    evt_range_break_up = np.zeros(n, dtype=bool)
    evt_range_break_down = np.zeros(n, dtype=bool)
    evt_range_reenter_from_above = np.zeros(n, dtype=bool)
    evt_range_reenter_from_below = np.zeros(n, dtype=bool)

    for i in range(n):
        # rope_smoother(float _src, float _threshold)
        if i == 0 or not np.isfinite(rope[i - 1]):
            prev_rope = src[i]  # var float _rope = _src
        else:
            prev_rope = rope[i - 1]

        threshold_nz = atr[i] if np.isfinite(atr[i]) else 0.0
        move = src[i] - prev_rope
        rope_i = prev_rope + max(abs(move) - threshold_nz, 0.0) * np.sign(move)

        rope[i] = rope_i
        upper[i] = rope_i + atr[i] if np.isfinite(atr[i]) else np.nan
        lower[i] = rope_i - atr[i] if np.isfinite(atr[i]) else np.nan

        # Directional Detection：严格按 Pine 先用 rope 变化判 dir，再用 ta.cross(src,rope) 置 0。
        prev_dir = int(dir_arr[i - 1]) if i > 0 else 0
        d = prev_dir
        if i > 0 and np.isfinite(rope[i - 1]):
            if rope[i] > rope[i - 1]:
                d = 1
            elif rope[i] < rope[i - 1]:
                d = -1

        crossed = _ta_cross(src, rope, i)
        if crossed:
            d = 0
            evt_cross_rope[i] = True

        dir_arr[i] = d

        # 方向转换事件
        if i > 0 and d != prev_dir:
            evt_dir_change[i] = True
            if prev_dir == -1 and d == 0:
                evt_red_to_blue[i] = True
            elif prev_dir == 0 and d == 1:
                evt_blue_to_green[i] = True
            elif prev_dir == 1 and d == 0:
                evt_green_to_blue[i] = True
            elif prev_dir == 0 and d == -1:
                evt_blue_to_red[i] = True
            elif prev_dir == -1 and d == 1:
                evt_red_to_green[i] = True
            elif prev_dir == 1 and d == -1:
                evt_green_to_red[i] = True

        if i > 0:
            if d == 1 and prev_dir != 1:
                evt_turn_up[i] = True
            if d == -1 and prev_dir != -1:
                evt_turn_down[i] = True
            if d == 0 and prev_dir != 0:
                evt_turn_flat[i] = True

        # Consolidation Ranges
        if d == 0:
            if i > 0 and prev_dir != 0:
                h_sum = 0.0
                l_sum = 0.0
                c_count = 0
                ff = not ff
                evt_range_start[i] = True

            if np.isfinite(h_sum) and np.isfinite(upper[i]):
                h_sum += upper[i]
            else:
                h_sum = np.nan
            if np.isfinite(l_sum) and np.isfinite(lower[i]):
                l_sum += lower[i]
            else:
                l_sum = np.nan
            c_count += 1

            c_hi[i] = h_sum / c_count if c_count > 0 and np.isfinite(h_sum) else np.nan
            c_lo[i] = l_sum / c_count if c_count > 0 and np.isfinite(l_sum) else np.nan
        else:
            # Pine var float：非震荡状态保持上一根区间值；用于后续判断离最近蓝区的位置。
            if i > 0:
                c_hi[i] = c_hi[i - 1]
                c_lo[i] = c_lo[i - 1]

        ff_arr[i] = ff

        # Rope 趋势线触碰/穿越事件：触碰用 K 线区间，穿越用 close 与 rope 的相对位置变化。
        if np.isfinite(rope[i]):
            evt_line_touch_rope[i] = bool(l[i] <= rope[i] <= h[i])
            if i > 0 and np.isfinite(rope[i - 1]):
                prev_rel = c[i - 1] - rope[i - 1]
                cur_rel = c[i] - rope[i]
                evt_line_cross_up[i] = bool(cur_rel > 0 and prev_rel <= 0)
                evt_line_cross_down[i] = bool(cur_rel < 0 and prev_rel >= 0)
            evt_line_retest_green[i] = bool(d == 1 and evt_line_touch_rope[i])
            evt_line_retest_red[i] = bool(d == -1 and evt_line_touch_rope[i])

        # 蓝色震荡区上下轨事件：使用最近有效区间；允许离开蓝区后继续判断回踩/反抽。
        rh = c_hi[i]
        rl = c_lo[i]
        if np.isfinite(rh) and np.isfinite(rl) and rh > rl:
            evt_range_touch_high[i] = bool(l[i] <= rh <= h[i])
            evt_range_touch_low[i] = bool(l[i] <= rl <= h[i])
            if i > 0 and np.isfinite(c_hi[i - 1]) and np.isfinite(c_lo[i - 1]) and c_hi[i - 1] > c_lo[i - 1]:
                prev_rh = c_hi[i - 1]
                prev_rl = c_lo[i - 1]
                prev_close = c[i - 1]
                cur_close = c[i]
                evt_range_break_up[i] = bool(cur_close > rh and prev_close <= prev_rh)
                evt_range_break_down[i] = bool(cur_close < rl and prev_close >= prev_rl)
                evt_range_reenter_from_above[i] = bool(prev_close > prev_rh and rl <= cur_close <= rh)
                evt_range_reenter_from_below[i] = bool(prev_close < prev_rl and rl <= cur_close <= rh)

    # 因子：方向持续根数
    dir_prev = np.roll(dir_arr, 1)
    if n:
        dir_prev[0] = 0
    dir_bars = np.zeros(n, dtype=int)
    for i in range(n):
        if i == 0 or dir_arr[i] != dir_arr[i - 1]:
            dir_bars[i] = 1
        else:
            dir_bars[i] = dir_bars[i - 1] + 1

    # 因子：20bar 大背景 regime，多头=1，震荡=0，空头=-1。
    # 逻辑：最近 N 根颜色占比 + rope N 根斜率过滤，避免把短暂变色误判为大背景切换。
    regime_lookback = max(1, int(cfg.regime_lookback))
    regime_threshold = float(cfg.regime_threshold)
    dir_s = pd.Series(dir_arr, index=out.index)
    green_ratio = (dir_s == 1).rolling(regime_lookback, min_periods=1).mean().to_numpy(float)
    red_ratio = (dir_s == -1).rolling(regime_lookback, min_periods=1).mean().to_numpy(float)
    blue_ratio = (dir_s == 0).rolling(regime_lookback, min_periods=1).mean().to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rope_slope_pct = rope / np.roll(rope, regime_lookback) - 1.0
    if n > 0:
        rope_slope_pct[:regime_lookback] = np.nan

    regime = np.zeros(n, dtype=int)
    bull_mask = (green_ratio >= regime_threshold) & (rope_slope_pct > 0)
    bear_mask = (red_ratio >= regime_threshold) & (rope_slope_pct < 0)
    regime[bull_mask] = 1
    regime[bear_mask] = -1

    regime_prev = np.roll(regime, 1)
    if n:
        regime_prev[0] = 0
    regime_bars = np.zeros(n, dtype=int)
    for i in range(n):
        if i == 0 or regime[i] != regime[i - 1]:
            regime_bars[i] = 1
        else:
            regime_bars[i] = regime_bars[i - 1] + 1

    evt_regime_change = np.zeros(n, dtype=bool)
    evt_regime_to_bull = np.zeros(n, dtype=bool)
    evt_regime_to_bear = np.zeros(n, dtype=bool)
    evt_regime_to_range = np.zeros(n, dtype=bool)
    if n > 1:
        evt_regime_change[1:] = regime[1:] != regime[:-1]
        evt_regime_to_bull[1:] = (regime[1:] == 1) & (regime[:-1] != 1)
        evt_regime_to_bear[1:] = (regime[1:] == -1) & (regime[:-1] != -1)
        evt_regime_to_range[1:] = (regime[1:] == 0) & (regime[:-1] != 0)

    regime_strength = np.where(
        regime > 0,
        green_ratio - red_ratio,
        np.where(regime < 0, red_ratio - green_ratio, blue_ratio),
    )

    # 因子：Rope 偏离与蓝区位置/宽度
    with np.errstate(divide="ignore", invalid="ignore"):
        rope_dev_pct = c / rope - 1.0
        rope_dev_atr = (c - rope) / atr
        range_mid = (c_hi + c_lo) / 2.0
        range_width = c_hi - c_lo
        range_pos_01 = (c - c_lo) / range_width
        range_width_pct = c_hi / c_lo - 1.0
        range_width_atr = range_width / atr

    # 基础指标
    out["atr_rope_tr"] = tr
    out["atr_rope_atr_raw"] = atr_raw
    out["atr_rope_atr"] = atr
    out["atr_rope_rope"] = rope
    out["atr_rope_upper"] = upper
    out["atr_rope_lower"] = lower
    out["atr_rope_dir"] = dir_arr
    out["atr_rope_c_hi"] = c_hi
    out["atr_rope_c_lo"] = c_lo
    out["atr_rope_ff"] = ff_arr

    # 方向事件组
    out["evt_atr_rope_dir_change"] = evt_dir_change
    out["evt_atr_rope_dir_red_to_blue"] = evt_red_to_blue
    out["evt_atr_rope_dir_blue_to_green"] = evt_blue_to_green
    out["evt_atr_rope_dir_green_to_blue"] = evt_green_to_blue
    out["evt_atr_rope_dir_blue_to_red"] = evt_blue_to_red
    out["evt_atr_rope_dir_red_to_green"] = evt_red_to_green
    out["evt_atr_rope_dir_green_to_red"] = evt_green_to_red
    # 兼容旧字段
    out["evt_atr_rope_turn_up"] = evt_turn_up
    out["evt_atr_rope_turn_down"] = evt_turn_down
    out["evt_atr_rope_turn_flat"] = evt_turn_flat

    # Rope 趋势线事件组
    out["evt_atr_rope_cross_rope"] = evt_cross_rope
    out["evt_atr_rope_line_touch_rope"] = evt_line_touch_rope
    out["evt_atr_rope_line_cross_up"] = evt_line_cross_up
    out["evt_atr_rope_line_cross_down"] = evt_line_cross_down
    out["evt_atr_rope_line_retest_green"] = evt_line_retest_green
    out["evt_atr_rope_line_retest_red"] = evt_line_retest_red

    # 蓝色震荡区事件组
    out["evt_atr_rope_range_start"] = evt_range_start
    out["evt_atr_rope_range_touch_high"] = evt_range_touch_high
    out["evt_atr_rope_range_touch_low"] = evt_range_touch_low
    out["evt_atr_rope_range_break_up"] = evt_range_break_up
    out["evt_atr_rope_range_break_down"] = evt_range_break_down
    out["evt_atr_rope_range_reenter_from_above"] = evt_range_reenter_from_above
    out["evt_atr_rope_range_reenter_from_below"] = evt_range_reenter_from_below

    # 因子组：状态/趋势线/蓝区
    out["factor_atr_rope_state_dir"] = dir_arr
    out["factor_atr_rope_state_dir_prev"] = dir_prev
    out["factor_atr_rope_state_dir_bars"] = dir_bars

    # 大背景 regime 事件组
    out["evt_atr_rope_regime_change"] = evt_regime_change
    out["evt_atr_rope_regime_to_bull"] = evt_regime_to_bull
    out["evt_atr_rope_regime_to_bear"] = evt_regime_to_bear
    out["evt_atr_rope_regime_to_range"] = evt_regime_to_range

    # 大背景 regime 因子组
    out["factor_atr_rope_regime"] = regime
    out["factor_atr_rope_regime_prev"] = regime_prev
    out["factor_atr_rope_regime_bars"] = regime_bars
    out[f"factor_atr_rope_green_ratio_{regime_lookback}"] = green_ratio
    out[f"factor_atr_rope_red_ratio_{regime_lookback}"] = red_ratio
    out[f"factor_atr_rope_blue_ratio_{regime_lookback}"] = blue_ratio
    out[f"factor_atr_rope_slope_pct_{regime_lookback}"] = rope_slope_pct
    out["factor_atr_rope_regime_strength"] = regime_strength

    out["factor_atr_rope_line_dev_pct"] = rope_dev_pct
    out["factor_atr_rope_line_dev_atr"] = rope_dev_atr
    out["factor_atr_rope_range_high"] = c_hi
    out["factor_atr_rope_range_low"] = c_lo
    out["factor_atr_rope_range_mid"] = range_mid
    out["factor_atr_rope_range_pos_01"] = range_pos_01
    out["factor_atr_rope_range_width_pct"] = range_width_pct
    out["factor_atr_rope_range_width_atr"] = range_width_atr

    return out

# =============================================================================
# 可视化
# =============================================================================
def _segment_by_color(x: np.ndarray, y: np.ndarray, colors: np.ndarray):
    """把线按颜色切成连续 segment，避免 Plotly 一条线不能逐段变色。"""
    if len(x) == 0:
        return []
    segs = []
    start = 0
    for i in range(1, len(x)):
        if colors[i] != colors[i - 1] or not (np.isfinite(y[i]) and np.isfinite(y[i - 1])):
            if i - start >= 2:
                segs.append((start, i - 1, colors[i - 1]))
            start = i
    if len(x) - start >= 2:
        segs.append((start, len(x) - 1, colors[-1]))
    return segs


def _range_segments(mask: np.ndarray):
    """返回 mask=True 的连续区间 [start,end]。"""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    segs = []
    s = idx[0]
    prev = idx[0]
    for v in idx[1:]:
        if v == prev + 1:
            prev = v
        else:
            segs.append((s, prev))
            s = prev = v
    segs.append((s, prev))
    return segs


def build_html(df_full: pd.DataFrame, df_plot: pd.DataFrame, out_html: str, title: str, cfg: ATRRopeConfig) -> None:
    if go is None or make_subplots is None:
        raise RuntimeError("未安装 plotly: pip install plotly")

    plot_start = len(df_full) - len(df_plot)
    x = np.arange(len(df_plot), dtype=float)
    intraday = len(df_plot.index) > 1 and (df_plot.index[1] - df_plot.index[0]) < pd.Timedelta("20H")
    tick_text = [ts.strftime("%Y-%m-%d %H:%M") if intraday else ts.strftime("%Y-%m-%d") for ts in df_plot.index]

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.028,
        row_heights=[0.62, 0.12, 0.12, 0.07, 0.07],
        subplot_titles=(title, "Volume", "ATR Rope Event Rows", "Rope Deviation / Regime Factors", "Range Position / Width Factors"),
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df_plot["open"], high=df_plot["high"], low=df_plot["low"], close=df_plot["close"],
            name="K线",
            increasing_line_color="#00c853", decreasing_line_color="#ff5252",
            increasing_fillcolor="#00c853", decreasing_fillcolor="#ff5252",
        ),
        row=1, col=1,
    )

    rope = df_plot["atr_rope_rope"].to_numpy(float)
    upper = df_plot["atr_rope_upper"].to_numpy(float)
    lower = df_plot["atr_rope_lower"].to_numpy(float)
    dir_arr = df_plot["atr_rope_dir"].to_numpy(int)
    colors = np.where(dir_arr > 0, cfg.up_color, np.where(dir_arr < 0, cfg.down_color, cfg.flat_color))

    # Rope 主线逐段着色
    for s, e, col in _segment_by_color(x, rope, colors):
        fig.add_trace(
            go.Scatter(
                x=x[s:e + 1], y=rope[s:e + 1], mode="lines",
                line=dict(color=col, width=3),
                name="Rope" if s == 0 else "Rope",
                showlegend=s == 0,
                hovertemplate="Rope=%{y:.3f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ATR Channel，默认关闭；通过 --show-atr-channel 打开。
    if cfg.show_atr_channel:
        for s, e, col in _segment_by_color(x, upper, colors):
            fig.add_trace(go.Scatter(x=x[s:e + 1], y=upper[s:e + 1], mode="lines",
                                     line=dict(color=col, width=1), name="Upper", showlegend=False), row=1, col=1)
        for s, e, col in _segment_by_color(x, lower, colors):
            fig.add_trace(go.Scatter(x=x[s:e + 1], y=lower[s:e + 1], mode="lines",
                                     line=dict(color=col, width=1), name="Lower", showlegend=False), row=1, col=1)

    # Consolidation Ranges：Pine 用 ff 在两组 plot/fill 间切换，离线图用连续区间填充表达同等效果。
    if cfg.show_ranges:
        flat_mask = df_plot["atr_rope_dir"].to_numpy(int) == 0
        c_hi = df_plot["atr_rope_c_hi"].to_numpy(float)
        c_lo = df_plot["atr_rope_c_lo"].to_numpy(float)
        for s, e in _range_segments(flat_mask & np.isfinite(c_hi) & np.isfinite(c_lo)):
            xs = x[s:e + 1]
            hi = c_hi[s:e + 1]
            lo = c_lo[s:e + 1]
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([xs, xs[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill="toself",
                    fillcolor=cfg.range_color,
                    line=dict(color="rgba(0,77,146,0.45)", width=1),
                    mode="lines",
                    name="Consolidation Range",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )
            fig.add_trace(go.Scatter(x=xs, y=hi, mode="lines",
                                     line=dict(color="rgba(0,77,146,0.70)", width=1), showlegend=False,
                                     name="Range High"), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=lo, mode="lines",
                                     line=dict(color="rgba(0,77,146,0.70)", width=1), showlegend=False,
                                     name="Range Low"), row=1, col=1)

    # 事件标记
    price_pad = float((df_plot["high"].max() - df_plot["low"].min()) * 0.018) if len(df_plot) else 0.0

    def event_mark(mask_col: str, price_col: str, label: str, color: str, symbol: str, shift: float = 0.0):
        if mask_col not in df_plot.columns:
            return
        mask = df_plot[mask_col].fillna(False).to_numpy(bool)
        if not mask.any():
            return
        y = df_plot[price_col].to_numpy(float)[mask] + shift
        cd = np.column_stack([
            df_plot.index.astype(str).to_numpy()[mask],
            df_plot["close"].to_numpy(float)[mask],
            df_plot["atr_rope_rope"].to_numpy(float)[mask],
            df_plot["atr_rope_dir"].to_numpy(int)[mask],
            df_plot["atr_rope_atr"].to_numpy(float)[mask],
        ])
        fig.add_trace(
            go.Scatter(
                x=x[mask], y=y, mode="markers",
                marker=dict(symbol=symbol, size=11, color=color, line=dict(color="#ffffff", width=1)),
                name=label,
                customdata=cd,
                hovertemplate=(
                    f"{label}<br>%{{customdata[0]}}"
                    "<br>close=%{customdata[1]:.3f}"
                    "<br>rope=%{customdata[2]:.3f}"
                    "<br>dir=%{customdata[3]}"
                    "<br>atr*multi=%{customdata[4]:.3f}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # 方向转换事件
    event_mark("evt_atr_rope_dir_blue_to_green", "high", "Blue→Green", cfg.up_color, "triangle-up", price_pad)
    event_mark("evt_atr_rope_dir_green_to_blue", "close", "Green→Blue", cfg.flat_color, "circle", 0.0)
    event_mark("evt_atr_rope_dir_blue_to_red", "low", "Blue→Red", cfg.down_color, "triangle-down", -price_pad)
    event_mark("evt_atr_rope_dir_red_to_blue", "close", "Red→Blue", cfg.flat_color, "circle", 0.0)
    event_mark("evt_atr_rope_dir_red_to_green", "high", "Red→Green", "#64dd17", "star", price_pad * 1.5)
    event_mark("evt_atr_rope_dir_green_to_red", "low", "Green→Red", "#ff1744", "star", -price_pad * 1.5)
    event_mark("evt_atr_rope_regime_to_bull", "high", "Regime→Bull", "#76ff03", "star-triangle-up", price_pad * 2.0)
    event_mark("evt_atr_rope_regime_to_bear", "low", "Regime→Bear", "#ff1744", "star-triangle-down", -price_pad * 2.0)
    event_mark("evt_atr_rope_regime_to_range", "close", "Regime→Range", "#29b6f6", "hexagon", 0.0)

    # Rope 趋势线事件
    event_mark("evt_atr_rope_line_retest_green", "low", "Green Retest Rope", "#00e676", "circle-open", -price_pad * 1.2)
    event_mark("evt_atr_rope_line_retest_red", "high", "Red Retest Rope", "#ff1744", "circle-open", price_pad * 1.2)
    event_mark("evt_atr_rope_line_cross_up", "high", "Close Cross Rope Up", "#40c4ff", "cross", price_pad * 1.8)
    event_mark("evt_atr_rope_line_cross_down", "low", "Close Cross Rope Down", "#ffab40", "cross", -price_pad * 1.8)

    # 蓝色震荡区事件
    event_mark("evt_atr_rope_range_touch_high", "high", "Touch Range High", "#42a5f5", "diamond-open", price_pad * 0.8)
    event_mark("evt_atr_rope_range_touch_low", "low", "Touch Range Low", "#42a5f5", "diamond-open", -price_pad * 0.8)
    event_mark("evt_atr_rope_range_break_up", "high", "Range Break Up", "#00e676", "diamond", price_pad * 2.4)
    event_mark("evt_atr_rope_range_break_down", "low", "Range Break Down", "#ff1744", "diamond", -price_pad * 2.4)
    event_mark("evt_atr_rope_range_reenter_from_above", "close", "Reenter Range From Above", "#90caf9", "square", 0.0)
    event_mark("evt_atr_rope_range_reenter_from_below", "close", "Reenter Range From Below", "#90caf9", "square", 0.0)

    # Volume
    vol = df_plot["volume"].astype(float)
    bar_colors = np.where(df_plot["close"].to_numpy(float) >= df_plot["open"].to_numpy(float), "#00c853", "#ff5252")
    fig.add_trace(go.Bar(x=x, y=vol, marker_color=bar_colors, name="volume"), row=2, col=1)

    # Event rows：按功能归类，和 CSV 字段前缀一致。
    rows = [
        ("evt_atr_rope_regime_to_bull", 13, "背景: 转多头", "#76ff03"),
        ("evt_atr_rope_regime_to_range", 12, "背景: 转震荡", "#29b6f6"),
        ("evt_atr_rope_regime_to_bear", 11, "背景: 转空头", "#ff1744"),
        ("evt_atr_rope_dir_blue_to_green", 10, "方向: 蓝→绿", cfg.up_color),
        ("evt_atr_rope_dir_green_to_blue", 9, "方向: 绿→蓝", cfg.flat_color),
        ("evt_atr_rope_dir_blue_to_red", 8, "方向: 蓝→红", cfg.down_color),
        ("evt_atr_rope_dir_red_to_blue", 7, "方向: 红→蓝", cfg.flat_color),
        ("evt_atr_rope_line_retest_green", 6, "Rope: 多头回踩", "#00e676"),
        ("evt_atr_rope_line_retest_red", 5, "Rope: 空头反抽", "#ff1744"),
        ("evt_atr_rope_line_cross_up", 4, "Rope: 上穿", "#40c4ff"),
        ("evt_atr_rope_line_cross_down", 3, "Rope: 下穿", "#ffab40"),
        ("evt_atr_rope_range_touch_high", 2.4, "蓝区: 触上轨", "#42a5f5"),
        ("evt_atr_rope_range_touch_low", 2.0, "蓝区: 触下轨", "#42a5f5"),
        ("evt_atr_rope_range_break_up", 1.4, "蓝区: 上破", "#00e676"),
        ("evt_atr_rope_range_break_down", 1.0, "蓝区: 下破", "#ff1744"),
    ]
    for col, yv, nm, color in rows:
        if col in df_plot.columns:
            mask = df_plot[col].fillna(False).to_numpy(bool)
            if mask.any():
                fig.add_trace(
                    go.Scatter(x=x[mask], y=np.full(mask.sum(), yv), mode="markers",
                               marker=dict(size=9, color=color), name=nm, showlegend=False),
                    row=3, col=1,
                )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ticktext=["蓝区下破", "触区间", "下穿Rope", "上穿Rope", "空头反抽", "多头回踩", "红→蓝", "蓝→红", "绿→蓝", "蓝→绿", "背景空", "背景震", "背景多"],
        row=3, col=1,
    )

    # Factor panels：离趋势线偏离度、蓝区位置、蓝区宽度。
    if "factor_atr_rope_line_dev_pct" in df_plot.columns:
        fig.add_trace(
            go.Scatter(x=x, y=df_plot["factor_atr_rope_line_dev_pct"] * 100, mode="lines",
                       line=dict(width=1.4), name="Rope dev %",
                       hovertemplate="Rope dev=%{y:.2f}%<extra></extra>"),
            row=4, col=1,
        )
        fig.add_hline(y=0, line_width=1, line_dash="dot", row=4, col=1)
        fig.update_yaxes(title_text="dev% / regime", row=4, col=1)

    if "factor_atr_rope_regime" in df_plot.columns:
        fig.add_trace(
            go.Scatter(x=x, y=df_plot["factor_atr_rope_regime"], mode="lines",
                       line=dict(width=1.2, dash="dash"), name="Regime 1/0/-1",
                       hovertemplate="Regime=%{y:.0f}<extra></extra>"),
            row=4, col=1,
        )
        fig.add_hline(y=1, line_width=1, line_dash="dot", row=4, col=1)
        fig.add_hline(y=-1, line_width=1, line_dash="dot", row=4, col=1)

    if "factor_atr_rope_range_pos_01" in df_plot.columns:
        fig.add_trace(
            go.Scatter(x=x, y=df_plot["factor_atr_rope_range_pos_01"], mode="lines",
                       line=dict(width=1.4), name="Range pos 0-1",
                       hovertemplate="Range pos=%{y:.3f}<extra></extra>"),
            row=5, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=df_plot["factor_atr_rope_range_width_pct"] * 100, mode="lines",
                       line=dict(width=1.2, dash="dot"), name="Range width %",
                       hovertemplate="Range width=%{y:.2f}%<extra></extra>"),
            row=5, col=1,
        )
        fig.add_hline(y=0, line_width=1, line_dash="dot", row=5, col=1)
        fig.add_hline(y=1, line_width=1, line_dash="dot", row=5, col=1)
        fig.update_yaxes(title_text="pos/width%", row=5, col=1)

    step = max(1, len(df_plot) // 10)
    fig.update_xaxes(tickmode="array", tickvals=list(x[::step]), ticktext=tick_text[::step])
    fig.update_layout(
        template="plotly_dark",
        height=1120,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        title=dict(text=title, x=0.5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=30, t=70, b=40),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")


def _make_default_paths(symbol: Optional[str], freq: str, out_dir: str) -> Tuple[str, str]:
    tag = symbol or "csv"
    f = normalize_freq(freq)
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"{tag}_atr_rope_{f}.csv"), str(base / f"{tag}_atr_rope_{f}.html")


def run_one(args: argparse.Namespace, freq: str) -> Tuple[str, str]:
    freq = normalize_freq(freq)
    if args.csv:
        df = read_kline_csv(args.csv)
        symbol_tag = Path(args.csv).stem
    else:
        if not args.symbol:
            raise ValueError("必须提供 --symbol 或 --csv")
        df = fetch_kline_pytdx(args.symbol, freq, int(args.fetch_bars))
        symbol_tag = args.symbol

    cfg = ATRRopeConfig(
        length=int(args.length),
        multi=float(args.multi),
        source=str(args.source),
        show_ranges=not args.no_ranges,
        show_atr_channel=bool(args.show_atr_channel),
        regime_lookback=int(args.regime_lookback),
        regime_threshold=float(args.regime_threshold),
    )

    feat = compute_atr_rope(df, cfg)

    if args.out_csv and (not args.freqs or len(args.freqs.split(",")) == 1):
        out_csv = args.out_csv
    else:
        out_csv, _ = _make_default_paths(symbol_tag, freq, args.out_dir)
    if args.out_html and (not args.freqs or len(args.freqs.split(",")) == 1):
        out_html = args.out_html
    else:
        _, out_html = _make_default_paths(symbol_tag, freq, args.out_dir)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_csv, encoding="utf-8-sig")

    df_plot = feat.tail(int(args.bars)) if int(args.bars) > 0 else feat
    title = (
        f"ATR Rope [SamRecio] | {symbol_tag} {freq} | "
        f"ATR Len={cfg.length}, Multi={cfg.multi}, Source={cfg.source}, Regime={cfg.regime_lookback}"
    )
    build_html(feat, df_plot, out_html, title, cfg)
    return out_csv, out_html


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ATR Rope [SamRecio] Pine -> Python/Plotly")
    p.add_argument("--symbol", type=str, default=None, help="A股代码，例如 300133；不传则需使用 --csv")
    p.add_argument("--csv", type=str, default=None, help="本地 OHLCV CSV")
    p.add_argument("--freq", type=str, default="d", help="单周期：d/w/mo/60m/30m/15m/5m/1m")
    p.add_argument("--freqs", type=str, default=None, help="多周期，例如 d,w,60m。使用 --csv 时通常只用 --freq")
    p.add_argument("--bars", type=int, default=300, help="HTML 图中展示最近多少根")
    p.add_argument("--fetch-bars", type=int, default=1200, help="pytdx 拉取多少根历史 K 线")
    p.add_argument("--out-dir", type=str, default="./output_atr_rope", help="默认输出目录")
    p.add_argument("--out-csv", type=str, default=None, help="单周期 CSV 输出路径")
    p.add_argument("--out-html", type=str, default=None, help="单周期 HTML 输出路径")

    # Pine 默认参数
    p.add_argument("--source", type=str, default="close", help="Source，默认 close")
    p.add_argument("--length", type=int, default=14, help="ATR Len，默认 14")
    p.add_argument("--multi", type=float, default=1.5, help="Multi，默认 1.5")
    p.add_argument("--regime-lookback", type=int, default=20, help="大背景判断窗口，默认 20bar")
    p.add_argument("--regime-threshold", type=float, default=0.55, help="大背景颜色占比阈值，默认 0.55")

    # Pine 默认 Display：Consolidation Ranges=True, ATR Channel=False
    p.add_argument("--no-ranges", action="store_true", help="关闭 Consolidation Ranges")
    p.add_argument("--show-atr-channel", action="store_true", help="显示 ATR Channel；Pine 默认关闭")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    freqs = [normalize_freq(x.strip()) for x in args.freqs.split(",")] if args.freqs else [normalize_freq(args.freq)]
    results: List[Tuple[str, str]] = []
    for f in freqs:
        results.append(run_one(args, f))
    print("输出完成：")
    for csv_path, html_path in results:
        print(f"CSV : {csv_path}")
        print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
