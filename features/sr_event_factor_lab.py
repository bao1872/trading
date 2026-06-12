# -*- coding: utf-8 -*-
"""
sr_event_factor_lab.py

事件因子实验版：支撑/压力结构事件库 + 因子库 + 未来收益标签 + HTML 标记图

核心任务：
1) 使用 TradingView 风格 pivot high / pivot low 定义最近压力/支撑。
2) 输出“刺破最近支撑后收回”的完整事件和质量因子。
3) 输出“低位上穿最近压力位”的完整事件和质量因子。
4) 输出位置因子、支撑/压力质量、K线形态、量能、趋势/波动、未来表现标签。
5) 支持多周期：d / w / mo / 60m / 30m / 15m / 5m / 1m。
6) 输出完整 CSV + Plotly HTML 标记图。

依赖：
    pip install pandas numpy plotly pytdx

示例：
    python sr_event_factor_lab.py --symbol 300133 --freq w --pivot-len 10 --bars 300 --fetch-bars 1200

多周期：
    python sr_event_factor_lab.py --symbol 300133 --freqs d,w,60m --pivot-len 10 --bars 300 --fetch-bars 1200

本地CSV：
    python sr_event_factor_lab.py --csv your_kline.csv --freq w --pivot-len 10 --out-csv out.csv --out-html out.html

CSV 需要包含：datetime/open/high/low/close/volume；amount 可选。
"""
from __future__ import annotations

SR_EVENT_FACTOR_LAB_VERSION = "cluster_confluence_v2_2026_05_18"

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
# 数据获取：保留与你上传 PAVP 脚本一致的 pytdx 取数风格
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
# 基础计算工具
# =============================================================================
def tv_pivots_confirmed(high: np.ndarray, low: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TradingView ta.pivothigh/ta.pivotlow 风格，确认延迟 = length。"""
    n = len(high)
    pvt_high = np.full(n, np.nan)
    pvt_low = np.full(n, np.nan)
    pvt_high_anchor = np.full(n, np.nan)
    pvt_low_anchor = np.full(n, np.nan)
    win = 2 * length + 1
    for i in range(win - 1, n):
        c = i - length
        lo = c - length
        hi = c + length + 1
        hwin = high[lo:hi]
        lwin = low[lo:hi]
        if np.all(np.isfinite(hwin)) and np.isfinite(high[c]) and high[c] == np.max(hwin):
            pvt_high[i] = high[c]
            pvt_high_anchor[i] = c
        if np.all(np.isfinite(lwin)) and np.isfinite(low[c]) and low[c] == np.min(lwin):
            pvt_low[i] = low[c]
            pvt_low_anchor[i] = c
    return pvt_high, pvt_low, pvt_high_anchor, pvt_low_anchor


def _ffill(arr: np.ndarray) -> np.ndarray:
    return pd.Series(arr).ffill().to_numpy(dtype=float)


def _safe_div(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray:
    num_arr = np.asarray(num, dtype=float)
    den_arr = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num_arr / den_arr
    return np.where(np.isfinite(out), out, np.nan)


def _rolling_z(s: pd.Series, win: int) -> pd.Series:
    mean = s.rolling(win, min_periods=max(3, win // 3)).mean()
    std = s.rolling(win, min_periods=max(3, win // 3)).std(ddof=0)
    return (s - mean) / std.replace(0, np.nan)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _bars_since_level_change(level: np.ndarray) -> np.ndarray:
    out = np.full(len(level), np.nan)
    last_change: Optional[int] = None
    prev_val = np.nan
    for i, val in enumerate(level):
        if np.isfinite(val):
            changed = (not np.isfinite(prev_val)) or abs(val - prev_val) > 1e-12
            if changed:
                last_change = i
            if last_change is not None:
                out[i] = float(i - last_change)
            prev_val = val
    return out


def _rolling_touch_count(price_arr: np.ndarray, level_arr: np.ndarray, atr_arr: np.ndarray, win: int, tol_atr: float = 0.20) -> np.ndarray:
    """统计过去 win 根是否触碰某水平附近。tol = max(0.2*ATR, 0.3%)。"""
    n = len(price_arr)
    out = np.full(n, np.nan)
    for i in range(n):
        level = level_arr[i]
        if not np.isfinite(level):
            continue
        start = max(0, i - win + 1)
        tol = max((atr_arr[i] if np.isfinite(atr_arr[i]) else 0.0) * tol_atr, abs(level) * 0.003)
        vals = price_arr[start:i + 1]
        out[i] = float(np.sum(np.isfinite(vals) & (np.abs(vals - level) <= tol)))
    return out


def _rolling_last_touch_bars(price_arr: np.ndarray, level_arr: np.ndarray, atr_arr: np.ndarray, lookback: int = 120, tol_atr: float = 0.20) -> np.ndarray:
    n = len(price_arr)
    out = np.full(n, np.nan)
    for i in range(n):
        level = level_arr[i]
        if not np.isfinite(level):
            continue
        start = max(0, i - lookback + 1)
        tol = max((atr_arr[i] if np.isfinite(atr_arr[i]) else 0.0) * tol_atr, abs(level) * 0.003)
        bars = np.arange(start, i + 1)
        vals = price_arr[start:i + 1]
        hit = np.isfinite(vals) & (np.abs(vals - level) <= tol)
        if hit.any():
            out[i] = float(i - bars[hit][-1])
    return out


def _change_events_from_level(level_arr: np.ndarray) -> np.ndarray:
    """把连续持有的水平序列压缩成“发生变化才记录一次”的事件序列。"""
    out = np.full(len(level_arr), np.nan)
    prev = np.nan
    for i, val in enumerate(level_arr):
        if not np.isfinite(val):
            prev = val
            continue
        changed = (not np.isfinite(prev)) or abs(float(val) - float(prev)) > 1e-12
        if changed:
            out[i] = float(val)
        prev = val
    return out


def _level_cluster_features(
    level_ref: np.ndarray,
    primary_events: np.ndarray,
    secondary_events: np.ndarray,
    extra_events: Optional[np.ndarray],
    atr_arr: np.ndarray,
    lookback: int = 120,
    tol_pct: float = 0.015,
    tol_atr: float = 0.50,
    primary_weight: float = 1.0,
    secondary_weight: float = 1.0,
    extra_weight: float = 1.50,
) -> Dict[str, np.ndarray]:
    """
    围绕当前 level_ref 统计过去 lookback 根内的结构位聚集程度。

    说明：
    - primary_events / secondary_events 是“只在确认 bar 有值”的事件序列，例如 pvt_low / pvt_high。
    - extra_events 用于 R2S/其他结构位，必须也是“只在变化 bar 有值”的事件序列，避免连续持有导致重复计数。
    - 返回 count/score/zone_low/zone_high/density/has_*，用于区分单线支撑与结构共振区。
    """
    n = len(level_ref)
    count = np.full(n, np.nan)
    score = np.full(n, np.nan)
    zone_low = np.full(n, np.nan)
    zone_high = np.full(n, np.nan)
    width_pct = np.full(n, np.nan)
    density = np.full(n, np.nan)
    has_primary = np.zeros(n, dtype=bool)
    has_secondary = np.zeros(n, dtype=bool)
    has_extra = np.zeros(n, dtype=bool)

    for i in range(n):
        level = level_ref[i]
        if not np.isfinite(level) or level == 0:
            continue
        atr_i = atr_arr[i] if np.isfinite(atr_arr[i]) else 0.0
        tol = max(abs(level) * tol_pct, atr_i * tol_atr)
        start = max(0, i - int(lookback) + 1)

        vals: List[float] = [float(level)]
        weights: List[float] = [1.0]

        pvals = primary_events[start:i + 1]
        mask = np.isfinite(pvals) & (np.abs(pvals - level) <= tol)
        if mask.any():
            has_primary[i] = True
            vals.extend([float(x) for x in pvals[mask]])
            weights.extend([primary_weight] * int(mask.sum()))

        svals = secondary_events[start:i + 1]
        mask = np.isfinite(svals) & (np.abs(svals - level) <= tol)
        if mask.any():
            has_secondary[i] = True
            vals.extend([float(x) for x in svals[mask]])
            weights.extend([secondary_weight] * int(mask.sum()))

        if extra_events is not None:
            evals = extra_events[start:i + 1]
            mask = np.isfinite(evals) & (np.abs(evals - level) <= tol)
            if mask.any():
                has_extra[i] = True
                vals.extend([float(x) for x in evals[mask]])
                weights.extend([extra_weight] * int(mask.sum()))

        # 去掉几乎完全重复的价格点，避免同一个水平被重复记录导致 count 虚高。
        # 但 score 保留来源权重，反映多来源共振。
        if vals:
            arr = np.array(vals, dtype=float)
            w = np.array(weights, dtype=float)
            order = np.argsort(arr)
            arr = arr[order]
            w = w[order]
            unique_vals: List[float] = []
            unique_w: List[float] = []
            bucket_tol = max(abs(level) * 0.001, atr_i * 0.05)
            for v, ww in zip(arr, w):
                if not unique_vals or abs(v - unique_vals[-1]) > bucket_tol:
                    unique_vals.append(float(v))
                    unique_w.append(float(ww))
                else:
                    unique_w[-1] += float(ww)
            arr_u = np.array(unique_vals, dtype=float)
            w_u = np.array(unique_w, dtype=float)
            count[i] = float(len(arr_u))
            score[i] = float(w_u.sum())
            zone_low[i] = float(arr_u.min())
            zone_high[i] = float(arr_u.max())
            width_pct[i] = float((zone_high[i] - zone_low[i]) / abs(level)) if level != 0 else np.nan
            density[i] = float(score[i] / max(width_pct[i], 0.001))

    return {
        "count": count,
        "score": score,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "width_pct": width_pct,
        "density": density,
        "has_primary": has_primary,
        "has_secondary": has_secondary,
        "has_extra": has_extra,
    }


def _pivot_sequence_features(pvt_confirm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 active/prev/is_higher/slope2/slope3。"""
    n = len(pvt_confirm)
    active = np.full(n, np.nan)
    prev = np.full(n, np.nan)
    prev2 = np.full(n, np.nan)
    is_higher = np.full(n, np.nan)
    slope2 = np.full(n, np.nan)
    slope3 = np.full(n, np.nan)
    seq: List[Tuple[int, float]] = []
    for i, val in enumerate(pvt_confirm):
        if np.isfinite(val):
            if not seq or abs(seq[-1][1] - val) > 1e-12 or seq[-1][0] != i:
                seq.append((i, float(val)))
        if seq:
            active[i] = seq[-1][1]
        if len(seq) >= 2:
            prev[i] = seq[-2][1]
            is_higher[i] = float(seq[-1][1] > seq[-2][1])
            den = max(1, seq[-1][0] - seq[-2][0])
            slope2[i] = (seq[-1][1] - seq[-2][1]) / den / seq[-2][1] if seq[-2][1] != 0 else np.nan
        if len(seq) >= 3:
            prev2[i] = seq[-3][1]
            den3 = max(1, seq[-1][0] - seq[-3][0])
            slope3[i] = (seq[-1][1] - seq[-3][1]) / den3 / seq[-3][1] if seq[-3][1] != 0 else np.nan
    return active, prev, is_higher, slope2, slope3


def _future_labels(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    for h in horizons:
        fwd_close = close.shift(-h)
        out[f"fwd_ret_{h}"] = fwd_close / close - 1.0
        # 未来 1..h 根内最高收益 / 最大回撤，不包含当前 bar。
        max_high = pd.concat([high.shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1)
        min_low = pd.concat([low.shift(-k) for k in range(1, h + 1)], axis=1).min(axis=1)
        out[f"fwd_max_ret_{h}"] = max_high / close - 1.0
        out[f"fwd_mdd_{h}"] = min_low / close - 1.0
        out[f"fwd_reward_risk_{h}"] = out[f"fwd_max_ret_{h}"] / out[f"fwd_mdd_{h}"].abs().replace(0, np.nan)
        out[f"fwd_win_{h}_3pct"] = (out[f"fwd_max_ret_{h}"] >= 0.03).astype("Int64")
        out[f"fwd_win_{h}_5pct"] = (out[f"fwd_max_ret_{h}"] >= 0.05).astype("Int64")
        out[f"fwd_loss_{h}_3pct"] = (out[f"fwd_mdd_{h}"] <= -0.03).astype("Int64")
        out[f"fwd_loss_{h}_5pct"] = (out[f"fwd_mdd_{h}"] <= -0.05).astype("Int64")
    return out


@dataclass
class LabConfig:
    pivot_len: int = 10
    use_prev_confirmed_level: bool = True
    low_zone_thresholds: Tuple[float, ...] = (0.25, 0.35, 0.50)
    low_zone_windows: Tuple[int, ...] = (5, 10)
    horizons: Tuple[int, ...] = (1, 3, 5, 10, 20)
    # 结构簇 / 共振区参数：用于区分单一支撑/压力与多结构位重叠区
    cluster_lookback: int = 120
    cluster_tolerance_pct: float = 0.015
    cluster_tolerance_atr: float = 0.50
    strong_cluster_count: int = 3
    strong_cluster_score: float = 3.0


# =============================================================================
# 核心因子/事件计算
# =============================================================================
def compute_sr_factor_lab(df: pd.DataFrame, cfg: LabConfig) -> pd.DataFrame:
    out = df.copy()
    for col in ["open", "high", "low", "close"]:
        if col not in out.columns:
            raise ValueError(f"df 缺少字段: {col}")
    if "volume" not in out.columns:
        out["volume"] = np.nan
    if "amount" not in out.columns:
        out["amount"] = np.nan

    o = out["open"].to_numpy(dtype=float)
    h = out["high"].to_numpy(dtype=float)
    l = out["low"].to_numpy(dtype=float)
    c = out["close"].to_numpy(dtype=float)
    n = len(out)

    pvt_high, pvt_low, pvt_high_anchor, pvt_low_anchor = tv_pivots_confirmed(h, l, cfg.pivot_len)
    out["pvt_high_confirm"] = pvt_high
    out["pvt_low_confirm"] = pvt_low
    out["pvt_high_anchor_index"] = pvt_high_anchor
    out["pvt_low_anchor_index"] = pvt_low_anchor

    resistance_active, prev_resistance_active, resistance_is_higher_high, resistance_slope_2, resistance_slope_3 = _pivot_sequence_features(pvt_high)
    support_active, prev_support_active, support_is_higher_low, support_slope_2, support_slope_3 = _pivot_sequence_features(pvt_low)

    out["resistance_active"] = resistance_active
    out["support_active"] = support_active
    out["prev_resistance_active"] = prev_resistance_active
    out["prev_support_active"] = prev_support_active
    out["active_resistance_is_higher_high"] = resistance_is_higher_high
    out["active_support_is_higher_low"] = support_is_higher_low
    out["active_resistance_slope_2"] = resistance_slope_2
    out["active_resistance_slope_3"] = resistance_slope_3
    out["active_support_slope_2"] = support_slope_2
    out["active_support_slope_3"] = support_slope_3

    # 默认事件判断只使用上一根已经确认的水平，避免当前 bar 刚确认、当前 bar 就用。
    if cfg.use_prev_confirmed_level:
        support_ref = pd.Series(support_active).shift(1).to_numpy(dtype=float)
        resistance_ref = pd.Series(resistance_active).shift(1).to_numpy(dtype=float)
        prev_support_ref = pd.Series(prev_support_active).shift(1).to_numpy(dtype=float)
        prev_resistance_ref = pd.Series(prev_resistance_active).shift(1).to_numpy(dtype=float)
        support_is_higher_low_ref = pd.Series(support_is_higher_low).shift(1).to_numpy(dtype=float)
        resistance_is_higher_high_ref = pd.Series(resistance_is_higher_high).shift(1).to_numpy(dtype=float)
        support_slope_2_ref = pd.Series(support_slope_2).shift(1).to_numpy(dtype=float)
        support_slope_3_ref = pd.Series(support_slope_3).shift(1).to_numpy(dtype=float)
        resistance_slope_2_ref = pd.Series(resistance_slope_2).shift(1).to_numpy(dtype=float)
        resistance_slope_3_ref = pd.Series(resistance_slope_3).shift(1).to_numpy(dtype=float)
    else:
        support_ref = support_active.copy()
        resistance_ref = resistance_active.copy()
        prev_support_ref = prev_support_active.copy()
        prev_resistance_ref = prev_resistance_active.copy()
        support_is_higher_low_ref = support_is_higher_low.copy()
        resistance_is_higher_high_ref = resistance_is_higher_high.copy()
        support_slope_2_ref = support_slope_2.copy()
        support_slope_3_ref = support_slope_3.copy()
        resistance_slope_2_ref = resistance_slope_2.copy()
        resistance_slope_3_ref = resistance_slope_3.copy()

    # -------------------------------------------------------------------------
    # R2S：压力突破后转化为支撑。
    # 保留 pivot_support_ref 作为传统 pivot low 支撑；support_ref 改为 active_support_ref，
    # 即 max(pivot_support_ref, flipped_support_ref)，兼容旧字段但更适合趋势股。
    # -------------------------------------------------------------------------
    pivot_support_ref = support_ref.copy()
    pivot_resistance_ref = resistance_ref.copy()
    prev_close_r2s = pd.Series(c).shift(1).to_numpy(dtype=float)
    evt_cross_res_r2s = np.isfinite(pivot_resistance_ref) & (c > pivot_resistance_ref) & (prev_close_r2s <= pivot_resistance_ref)
    flipped_active = np.full(n, np.nan)
    flipped_age_active = np.full(n, np.nan)
    cur_flip = np.nan
    cur_age: Optional[int] = None
    for i in range(n):
        # 先判断旧 R2S 是否失效：收盘跌破则取消。
        if np.isfinite(cur_flip) and c[i] < cur_flip:
            cur_flip = np.nan
            cur_age = None
        # 再判断本 bar 是否有效突破压力，生成新的 R2S。
        if evt_cross_res_r2s[i] and np.isfinite(pivot_resistance_ref[i]):
            cur_flip = float(pivot_resistance_ref[i])
            cur_age = 0
        if np.isfinite(cur_flip):
            flipped_active[i] = cur_flip
            flipped_age_active[i] = float(cur_age or 0)
            cur_age = (cur_age or 0) + 1

    if cfg.use_prev_confirmed_level:
        flipped_support_ref = pd.Series(flipped_active).shift(1).to_numpy(dtype=float)
        flipped_support_age_bars = pd.Series(flipped_age_active).shift(1).to_numpy(dtype=float)
    else:
        flipped_support_ref = flipped_active
        flipped_support_age_bars = flipped_age_active

    active_support_ref = np.full(n, np.nan)
    for i in range(n):
        vals = [v for v in (pivot_support_ref[i], flipped_support_ref[i]) if np.isfinite(v)]
        if vals:
            active_support_ref[i] = max(vals)
    is_support_flipped = np.isfinite(flipped_support_ref) & (
        ~np.isfinite(pivot_support_ref) | (flipped_support_ref >= pivot_support_ref)
    )
    support_gap_pct = np.where(
        np.isfinite(pivot_support_ref) & np.isfinite(flipped_support_ref) & (pivot_support_ref != 0),
        (flipped_support_ref - pivot_support_ref) / pivot_support_ref,
        np.nan,
    )

    out["pivot_support_ref"] = pivot_support_ref
    out["pivot_resistance_ref"] = pivot_resistance_ref
    out["flipped_support_ref"] = flipped_support_ref
    out["active_support_ref"] = active_support_ref
    out["is_support_flipped"] = is_support_flipped.astype(bool)
    out["flipped_support_age_bars"] = flipped_support_age_bars
    out["support_gap_pct"] = support_gap_pct

    support_ref = active_support_ref

    out["support_ref"] = support_ref
    out["resistance_ref"] = resistance_ref
    out["prev_support_ref"] = prev_support_ref
    out["prev_resistance_ref"] = prev_resistance_ref
    out["support_is_higher_low"] = support_is_higher_low_ref
    out["resistance_is_higher_high"] = resistance_is_higher_high_ref
    out["resistance_is_lower_high"] = np.where(np.isfinite(resistance_is_higher_high_ref), 1.0 - resistance_is_higher_high_ref, np.nan)
    out["support_slope_2"] = support_slope_2_ref
    out["support_slope_3"] = support_slope_3_ref
    out["resistance_slope_2"] = resistance_slope_2_ref
    out["resistance_slope_3"] = resistance_slope_3_ref

    # anchor / confirm index/date 信息
    support_confirm_idx_active = _ffill(np.where(np.isfinite(pvt_low), np.arange(n, dtype=float), np.nan))
    resistance_confirm_idx_active = _ffill(np.where(np.isfinite(pvt_high), np.arange(n, dtype=float), np.nan))
    support_anchor_idx_active = _ffill(pvt_low_anchor)
    resistance_anchor_idx_active = _ffill(pvt_high_anchor)
    if cfg.use_prev_confirmed_level:
        support_confirm_idx_ref = pd.Series(support_confirm_idx_active).shift(1).to_numpy(dtype=float)
        resistance_confirm_idx_ref = pd.Series(resistance_confirm_idx_active).shift(1).to_numpy(dtype=float)
        support_anchor_idx_ref = pd.Series(support_anchor_idx_active).shift(1).to_numpy(dtype=float)
        resistance_anchor_idx_ref = pd.Series(resistance_anchor_idx_active).shift(1).to_numpy(dtype=float)
    else:
        support_confirm_idx_ref = support_confirm_idx_active
        resistance_confirm_idx_ref = resistance_confirm_idx_active
        support_anchor_idx_ref = support_anchor_idx_active
        resistance_anchor_idx_ref = resistance_anchor_idx_active

    dates = pd.Series(out.index.astype(str), index=np.arange(n))
    def _idx_to_date(idx_arr: np.ndarray) -> List[Optional[str]]:
        ret: List[Optional[str]] = []
        for v in idx_arr:
            if np.isfinite(v) and 0 <= int(v) < n:
                ret.append(str(dates.iloc[int(v)]))
            else:
                ret.append(None)
        return ret

    out["support_confirm_index"] = support_confirm_idx_ref
    out["resistance_confirm_index"] = resistance_confirm_idx_ref
    out["support_anchor_index"] = support_anchor_idx_ref
    out["resistance_anchor_index"] = resistance_anchor_idx_ref
    out["support_confirm_date"] = _idx_to_date(support_confirm_idx_ref)
    out["resistance_confirm_date"] = _idx_to_date(resistance_confirm_idx_ref)
    out["support_anchor_date"] = _idx_to_date(support_anchor_idx_ref)
    out["resistance_anchor_date"] = _idx_to_date(resistance_anchor_idx_ref)
    out["support_age_bars"] = np.where(np.isfinite(support_confirm_idx_ref), np.arange(n) - support_confirm_idx_ref, np.nan)
    out["resistance_age_bars"] = np.where(np.isfinite(resistance_confirm_idx_ref), np.arange(n) - resistance_confirm_idx_ref, np.nan)

    width = resistance_ref - support_ref
    valid_band = np.isfinite(width) & (width > 0) & np.isfinite(support_ref) & np.isfinite(resistance_ref)
    out["sr_mid_price"] = np.where(valid_band, (support_ref + resistance_ref) / 2.0, np.nan)
    out["sr_range_abs"] = np.where(valid_band, width, np.nan)
    out["sr_range_pct"] = np.where(valid_band & (support_ref != 0), width / support_ref, np.nan)

    def _pos(arr: np.ndarray) -> np.ndarray:
        ret = np.full(n, np.nan)
        ret[valid_band] = (arr[valid_band] - support_ref[valid_band]) / width[valid_band]
        return ret

    out["sr_pos_open_raw"] = _pos(o)
    out["sr_pos_high_raw"] = _pos(h)
    out["sr_pos_low_raw"] = _pos(l)
    out["sr_pos_raw"] = _pos(c)
    out["sr_pos_open"] = np.clip(out["sr_pos_open_raw"], 0.0, 1.0)
    out["sr_pos_high"] = np.clip(out["sr_pos_high_raw"], 0.0, 1.0)
    out["sr_pos_low"] = np.clip(out["sr_pos_low_raw"], 0.0, 1.0)
    out["sr_pos_01"] = np.clip(out["sr_pos_raw"], 0.0, 1.0)

    out["close_to_support_pct"] = np.where(np.isfinite(support_ref) & (support_ref != 0), (c - support_ref) / support_ref, np.nan)
    out["low_to_support_pct"] = np.where(np.isfinite(support_ref) & (support_ref != 0), (l - support_ref) / support_ref, np.nan)
    out["close_to_resistance_pct"] = np.where(np.isfinite(resistance_ref) & (resistance_ref != 0), (c - resistance_ref) / resistance_ref, np.nan)
    out["high_to_resistance_pct"] = np.where(np.isfinite(resistance_ref) & (resistance_ref != 0), (h - resistance_ref) / resistance_ref, np.nan)
    out["upside_to_resistance_pct"] = np.where(np.isfinite(resistance_ref) & (c != 0), (resistance_ref - c) / c, np.nan)
    out["downside_to_support_pct"] = np.where(np.isfinite(support_ref) & (c != 0), (c - support_ref) / c, np.nan)

    # K线形态 / 波动
    atr14 = _atr(out, 14)
    out["atr_14"] = atr14
    out["atr_pct_14"] = atr14 / out["close"].replace(0, np.nan)
    rng = out["high"] - out["low"]
    body = (out["close"] - out["open"]).abs()
    out["bar_range_pct"] = rng / out["close"].replace(0, np.nan)
    out["bar_range_atr"] = rng / atr14.replace(0, np.nan)
    out["body_pct"] = body / rng.replace(0, np.nan)
    out["body_pct_of_close"] = body / out["close"].replace(0, np.nan)
    out["close_pos_in_bar"] = (out["close"] - out["low"]) / rng.replace(0, np.nan)
    out["open_pos_in_bar"] = (out["open"] - out["low"]) / rng.replace(0, np.nan)
    upper_shadow = out["high"] - np.maximum(out["open"], out["close"])
    lower_shadow = np.minimum(out["open"], out["close"]) - out["low"]
    out["upper_shadow_pct"] = upper_shadow / rng.replace(0, np.nan)
    out["lower_shadow_pct"] = lower_shadow / rng.replace(0, np.nan)
    out["is_bull_bar"] = (out["close"] > out["open"]).astype(bool)
    out["is_bear_bar"] = (out["close"] < out["open"]).astype(bool)
    out["is_doji_like"] = (out["body_pct"] <= 0.20).astype(bool)
    out["is_long_lower_shadow"] = ((out["lower_shadow_pct"] >= 0.45) & (out["close_pos_in_bar"] >= 0.50)).astype(bool)
    out["is_long_upper_shadow"] = ((out["upper_shadow_pct"] >= 0.45) & (out["close_pos_in_bar"] <= 0.50)).astype(bool)
    out["is_close_upper_half"] = (out["close_pos_in_bar"] >= 0.50).astype(bool)
    out["is_close_upper_third"] = (out["close_pos_in_bar"] >= 2 / 3).astype(bool)
    out["is_wide_range_bar"] = (out["bar_range_atr"] >= 1.50).astype(bool)
    out["is_narrow_range_bar"] = (out["bar_range_atr"] <= 0.70).astype(bool)

    atr_arr = out["atr_14"].to_numpy(dtype=float)

    # 支撑/压力质量
    out["support_touch_count_20"] = _rolling_touch_count(l, support_ref, atr_arr, 20)
    out["support_touch_count_60"] = _rolling_touch_count(l, support_ref, atr_arr, 60)
    out["resistance_touch_count_20"] = _rolling_touch_count(h, resistance_ref, atr_arr, 20)
    out["resistance_touch_count_60"] = _rolling_touch_count(h, resistance_ref, atr_arr, 60)
    out["support_last_touch_bars"] = _rolling_last_touch_bars(l, support_ref, atr_arr, 120)
    out["resistance_last_touch_bars"] = _rolling_last_touch_bars(h, resistance_ref, atr_arr, 120)
    out["support_is_fresh"] = (out["support_age_bars"] <= 10).astype(bool)
    out["resistance_is_fresh"] = (out["resistance_age_bars"] <= 10).astype(bool)
    out["support_is_overused"] = (out["support_touch_count_60"] >= 4).astype(bool)
    out["resistance_is_overused"] = (out["resistance_touch_count_60"] >= 4).astype(bool)

    # -------------------------------------------------------------------------
    # 支撑/压力簇：把“线”升级成“结构区”。
    # 逻辑：围绕当前 support_ref / resistance_ref，统计 lookback 内接近的 pivot/R2S 水平。
    # 目的：区分普通刺破/突破 与 强结构簇刺破/突破。
    # -------------------------------------------------------------------------
    flipped_support_events = _change_events_from_level(flipped_support_ref) if "flipped_support_ref" in out.columns else None
    support_cluster = _level_cluster_features(
        support_ref,
        primary_events=pvt_low,
        secondary_events=pvt_high,
        extra_events=flipped_support_events,
        atr_arr=atr_arr,
        lookback=cfg.cluster_lookback,
        tol_pct=cfg.cluster_tolerance_pct,
        tol_atr=cfg.cluster_tolerance_atr,
        primary_weight=1.0,
        secondary_weight=1.2,
        extra_weight=1.5,
    )
    resistance_cluster = _level_cluster_features(
        resistance_ref,
        primary_events=pvt_high,
        secondary_events=pvt_low,
        extra_events=None,
        atr_arr=atr_arr,
        lookback=cfg.cluster_lookback,
        tol_pct=cfg.cluster_tolerance_pct,
        tol_atr=cfg.cluster_tolerance_atr,
        primary_weight=1.0,
        secondary_weight=1.1,
        extra_weight=1.0,
    )

    out["support_cluster_count"] = support_cluster["count"]
    out["support_cluster_score"] = support_cluster["score"]
    out["support_cluster_density"] = support_cluster["density"]
    out["support_cluster_width_pct"] = support_cluster["width_pct"]
    out["support_zone_low"] = support_cluster["zone_low"]
    out["support_zone_high"] = support_cluster["zone_high"]
    out["support_cluster_has_pivot_low"] = support_cluster["has_primary"].astype(bool)
    out["support_cluster_has_old_resistance"] = support_cluster["has_secondary"].astype(bool)
    out["support_cluster_has_r2s"] = support_cluster["has_extra"].astype(bool)
    out["support_cluster_is_strong"] = (
        (out["support_cluster_count"] >= cfg.strong_cluster_count) |
        (out["support_cluster_score"] >= cfg.strong_cluster_score)
    ).fillna(False).astype(bool)

    out["resistance_cluster_count"] = resistance_cluster["count"]
    out["resistance_cluster_score"] = resistance_cluster["score"]
    out["resistance_cluster_density"] = resistance_cluster["density"]
    out["resistance_cluster_width_pct"] = resistance_cluster["width_pct"]
    out["resistance_zone_low"] = resistance_cluster["zone_low"]
    out["resistance_zone_high"] = resistance_cluster["zone_high"]
    out["resistance_cluster_has_pivot_high"] = resistance_cluster["has_primary"].astype(bool)
    out["resistance_cluster_has_old_support"] = resistance_cluster["has_secondary"].astype(bool)
    out["resistance_cluster_is_strong"] = (
        (out["resistance_cluster_count"] >= cfg.strong_cluster_count) |
        (out["resistance_cluster_score"] >= cfg.strong_cluster_score)
    ).fillna(False).astype(bool)

    # 共振分：后续可扩展 PAVP/多周期；当前先用 pivot/R2S/cluster/验证次数量化。
    out["support_confluence_score"] = (
        out["support_cluster_score"].fillna(0)
        + out["is_support_flipped"].astype(float) * 1.0
        + (out["flipped_support_age_bars"].fillna(0) > 20).astype(float) * 0.8
        + out["support_is_higher_low"].fillna(0).astype(float) * 0.5
        - out["support_is_overused"].astype(float) * 0.6
    )
    out["resistance_confluence_score"] = (
        out["resistance_cluster_score"].fillna(0)
        + out["resistance_is_higher_high"].fillna(0).astype(float) * 0.5
        + out["resistance_is_overused"].astype(float) * 0.4
    )
    out["support_confluence_is_strong"] = (out["support_confluence_score"] >= 3.5).astype(bool)
    out["resistance_confluence_is_strong"] = (out["resistance_confluence_score"] >= 3.0).astype(bool)

    # 事件 A：刺破支撑收回
    prev_close = out["close"].shift(1).to_numpy(dtype=float)
    prev_high = out["high"].shift(1).to_numpy(dtype=float)
    prev_low = out["low"].shift(1).to_numpy(dtype=float)
    evt_pierce_support = np.isfinite(support_ref) & (l < support_ref)
    evt_close_below_support = np.isfinite(support_ref) & (c < support_ref)
    evt_close_break_support = np.isfinite(support_ref) & (c < support_ref) & (prev_close >= support_ref)
    evt_pierce_reclaim = evt_pierce_support & (c >= support_ref)
    evt_open_above_pierce_reclaim = evt_pierce_reclaim & (o >= support_ref)
    evt_gap_down_reclaim_support = evt_pierce_reclaim & (o < support_ref)
    evt_failed_reclaim_support = evt_pierce_support & (c < support_ref)
    evt_reclaim_support_and_bull_bar = evt_pierce_reclaim & (c > o)
    evt_reclaim_support_close_strong = evt_pierce_reclaim & (out["close_pos_in_bar"].to_numpy(dtype=float) >= 0.60)

    out["evt_pierce_recent_support"] = evt_pierce_support.astype(bool)
    out["evt_close_below_recent_support"] = evt_close_below_support.astype(bool)
    out["evt_close_break_recent_support"] = evt_close_break_support.astype(bool)
    out["evt_pierce_support_reclaim"] = evt_pierce_reclaim.astype(bool)
    out["evt_open_above_pierce_reclaim"] = evt_open_above_pierce_reclaim.astype(bool)
    out["evt_gap_down_reclaim_support"] = evt_gap_down_reclaim_support.astype(bool)
    out["evt_failed_reclaim_support"] = evt_failed_reclaim_support.astype(bool)
    out["evt_reclaim_support_and_bull_bar"] = evt_reclaim_support_and_bull_bar.astype(bool)
    out["evt_reclaim_support_close_strong"] = evt_reclaim_support_close_strong.astype(bool)

    pierce_depth = np.where(evt_pierce_support, support_ref - l, np.nan)
    reclaim_strength = np.where(evt_pierce_reclaim, c - support_ref, np.nan)
    out["support_pierce_depth_pct"] = np.where(evt_pierce_support & (support_ref != 0), pierce_depth / support_ref, np.nan)
    out["support_pierce_depth_atr"] = pierce_depth / out["atr_14"].replace(0, np.nan)
    out["support_pierce_depth_sr"] = np.where(valid_band, pierce_depth / width, np.nan)
    out["support_reclaim_strength_pct"] = np.where(evt_pierce_reclaim & (support_ref != 0), reclaim_strength / support_ref, np.nan)
    out["support_reclaim_strength_atr"] = reclaim_strength / out["atr_14"].replace(0, np.nan)
    out["support_reclaim_strength_sr"] = np.where(valid_band, reclaim_strength / width, np.nan)
    out["is_shallow_support_pierce"] = (out["support_pierce_depth_atr"] <= 0.30).fillna(False).astype(bool)
    out["is_mid_support_pierce"] = ((out["support_pierce_depth_atr"] > 0.30) & (out["support_pierce_depth_atr"] <= 0.80)).fillna(False).astype(bool)
    out["is_deep_support_pierce"] = (out["support_pierce_depth_atr"] > 0.80).fillna(False).astype(bool)

    # 细分：pivot 支撑 / R2S 支撑 / active 支撑
    evt_pierce_pivot_support = np.isfinite(pivot_support_ref) & (l < pivot_support_ref)
    evt_pierce_flipped_support = np.isfinite(flipped_support_ref) & (l < flipped_support_ref)
    evt_pierce_pivot_reclaim = evt_pierce_pivot_support & (c >= pivot_support_ref)
    evt_pierce_flipped_reclaim = evt_pierce_flipped_support & (c >= flipped_support_ref)
    evt_failed_pivot_reclaim = evt_pierce_pivot_support & (c < pivot_support_ref)
    evt_failed_flipped_reclaim = evt_pierce_flipped_support & (c < flipped_support_ref)
    evt_breakdown_flipped_support = np.isfinite(flipped_support_ref) & (c < flipped_support_ref) & (prev_close >= flipped_support_ref)
    evt_retest_flipped_support = (
        np.isfinite(flipped_support_ref) & np.isfinite(atr_arr) &
        (l >= flipped_support_ref) & ((l - flipped_support_ref) <= 0.30 * atr_arr)
    )
    evt_clean_hold_flipped_support = (
        np.isfinite(flipped_support_ref) & np.isfinite(atr_arr) &
        (l > flipped_support_ref + 0.30 * atr_arr)
    )

    out["evt_pierce_pivot_support_reclaim"] = evt_pierce_pivot_reclaim.astype(bool)
    out["evt_pierce_flipped_support_reclaim"] = evt_pierce_flipped_reclaim.astype(bool)
    out["evt_pierce_active_support_reclaim"] = out["evt_pierce_support_reclaim"].astype(bool)
    out["evt_failed_reclaim_pivot_support"] = evt_failed_pivot_reclaim.astype(bool)
    out["evt_failed_reclaim_flipped_support"] = evt_failed_flipped_reclaim.astype(bool)
    out["evt_failed_reclaim_active_support"] = out["evt_failed_reclaim_support"].astype(bool)
    out["evt_breakdown_flipped_support"] = evt_breakdown_flipped_support.astype(bool)
    out["evt_retest_flipped_support"] = evt_retest_flipped_support.astype(bool)
    out["evt_clean_hold_flipped_support"] = evt_clean_hold_flipped_support.astype(bool)
    out["pivot_support_pierce_depth_atr"] = np.where(evt_pierce_pivot_support, (pivot_support_ref - l) / np.where(atr_arr == 0, np.nan, atr_arr), np.nan)
    out["pivot_support_reclaim_strength_atr"] = np.where(evt_pierce_pivot_reclaim, (c - pivot_support_ref) / np.where(atr_arr == 0, np.nan, atr_arr), np.nan)
    out["flipped_support_pierce_depth_atr"] = np.where(evt_pierce_flipped_support, (flipped_support_ref - l) / np.where(atr_arr == 0, np.nan, atr_arr), np.nan)
    out["flipped_support_reclaim_strength_atr"] = np.where(evt_pierce_flipped_reclaim, (c - flipped_support_ref) / np.where(atr_arr == 0, np.nan, atr_arr), np.nan)

    # 支撑区事件：用 zone_low/zone_high 替代单点支撑，识别强支撑簇假破/破位。
    support_zone_low = out["support_zone_low"].to_numpy(dtype=float)
    support_zone_high = out["support_zone_high"].to_numpy(dtype=float)
    strong_support_cluster = out["support_cluster_is_strong"].to_numpy(dtype=bool)
    evt_pierce_support_zone = np.isfinite(support_zone_low) & (l < support_zone_low)
    evt_pierce_support_zone_reclaim_low = evt_pierce_support_zone & (c >= support_zone_low)
    evt_pierce_support_zone_reclaim_high = evt_pierce_support_zone & (c >= support_zone_high)
    evt_break_support_zone_low = np.isfinite(support_zone_low) & (c < support_zone_low) & (prev_close >= support_zone_low)
    out["evt_pierce_support_zone_reclaim"] = evt_pierce_support_zone_reclaim_low.astype(bool)
    out["evt_pierce_support_zone_reclaim_strong"] = evt_pierce_support_zone_reclaim_high.astype(bool)
    out["evt_pierce_strong_support_cluster_reclaim"] = (evt_pierce_support_zone_reclaim_low & strong_support_cluster).astype(bool)
    out["evt_break_strong_support_cluster"] = (evt_break_support_zone_low & strong_support_cluster).astype(bool)

    # 事件 B：压力突破 / 低位突破
    evt_high_break_res = np.isfinite(resistance_ref) & (h > resistance_ref)
    evt_close_above_res = np.isfinite(resistance_ref) & (c > resistance_ref)
    evt_cross_res = evt_close_above_res & (prev_close <= resistance_ref)
    evt_wick_break_res_fail = evt_high_break_res & (c <= resistance_ref)
    evt_gap_up_break_res = evt_close_above_res & (o > resistance_ref)
    evt_break_res_bull_bar = evt_cross_res & (c > o)
    evt_break_res_close_strong = evt_cross_res & (out["close_pos_in_bar"].to_numpy(dtype=float) >= 0.60)

    out["evt_high_break_recent_resistance"] = evt_high_break_res.astype(bool)
    out["evt_close_above_recent_resistance"] = evt_close_above_res.astype(bool)
    out["evt_cross_recent_resistance"] = evt_cross_res.astype(bool)
    out["evt_wick_break_resistance_fail"] = evt_wick_break_res_fail.astype(bool)
    out["evt_gap_up_break_resistance"] = evt_gap_up_break_res.astype(bool)
    out["evt_break_resistance_bull_bar"] = evt_break_res_bull_bar.astype(bool)
    out["evt_break_resistance_close_strong"] = evt_break_res_close_strong.astype(bool)

    break_strength = np.where(evt_cross_res, c - resistance_ref, np.nan)
    high_break_strength = np.where(evt_high_break_res, h - resistance_ref, np.nan)
    out["resistance_break_strength_pct"] = np.where(evt_cross_res & (resistance_ref != 0), break_strength / resistance_ref, np.nan)
    out["resistance_break_strength_atr"] = break_strength / out["atr_14"].replace(0, np.nan)
    out["resistance_break_strength_sr"] = np.where(valid_band, break_strength / width, np.nan)
    out["high_break_strength_pct"] = np.where(evt_high_break_res & (resistance_ref != 0), high_break_strength / resistance_ref, np.nan)
    out["high_break_strength_atr"] = high_break_strength / out["atr_14"].replace(0, np.nan)
    out["breakout_body_pct"] = np.where(evt_cross_res, out["body_pct"], np.nan)
    out["breakout_close_pos_in_bar"] = np.where(evt_cross_res, out["close_pos_in_bar"], np.nan)
    out["breakout_upper_shadow_pct"] = np.where(evt_cross_res, out["upper_shadow_pct"], np.nan)
    out["breakout_is_strong_close"] = evt_break_res_close_strong.astype(bool)
    out["breakout_is_long_upper_shadow"] = (evt_cross_res & out["is_long_upper_shadow"].to_numpy(dtype=bool)).astype(bool)

    # 压力区事件：突破/假突破强压力簇上沿，而不只是单点 resistance_ref。
    resistance_zone_high = out["resistance_zone_high"].to_numpy(dtype=float)
    resistance_zone_low = out["resistance_zone_low"].to_numpy(dtype=float)
    strong_resistance_cluster = out["resistance_cluster_is_strong"].to_numpy(dtype=bool)
    evt_high_break_res_cluster = np.isfinite(resistance_zone_high) & (h > resistance_zone_high)
    evt_close_above_res_cluster = np.isfinite(resistance_zone_high) & (c > resistance_zone_high)
    evt_break_res_cluster = evt_close_above_res_cluster & (prev_close <= resistance_zone_high)
    evt_wick_break_res_cluster_fail = evt_high_break_res_cluster & (c <= resistance_zone_high)
    out["evt_high_break_resistance_cluster"] = evt_high_break_res_cluster.astype(bool)
    out["evt_break_resistance_cluster"] = (evt_break_res_cluster & strong_resistance_cluster).astype(bool)
    out["evt_break_strong_resistance_cluster"] = (evt_break_res_cluster & strong_resistance_cluster).astype(bool)
    out["evt_wick_break_resistance_cluster_fail"] = (evt_wick_break_res_cluster_fail & strong_resistance_cluster).astype(bool)
    out["evt_close_above_resistance_cluster_upper"] = evt_close_above_res_cluster.astype(bool)
    out["resistance_cluster_break_strength_pct"] = np.where(evt_break_res_cluster & (resistance_zone_high != 0), (c - resistance_zone_high) / resistance_zone_high, np.nan)
    out["resistance_cluster_break_strength_atr"] = np.where(evt_break_res_cluster, (c - resistance_zone_high) / np.where(atr_arr == 0, np.nan, atr_arr), np.nan)

    # 低位区：不要用突破当天 close 位置，而使用突破前 N 根位置。
    sr_pos_01 = out["sr_pos_01"]
    for win in cfg.low_zone_windows:
        prev_pos = sr_pos_01.shift(1)
        out[f"sr_pos_01_prev{win}_min"] = prev_pos.rolling(win, min_periods=1).min()
        out[f"sr_pos_01_prev{win}_mean"] = prev_pos.rolling(win, min_periods=1).mean()
        for th in cfg.low_zone_thresholds:
            th_tag = f"{int(round(th * 1000)):03d}"
            was_low = out[f"sr_pos_01_prev{win}_min"] <= th
            out[f"was_low_zone_{win}_{th_tag}"] = was_low.fillna(False).astype(bool)
            out[f"evt_breakout_from_low{th_tag}_{win}"] = (evt_cross_res & was_low.to_numpy(dtype=bool)).astype(bool)
    # 常用别名，方便读取
    out["evt_break_resistance_from_low_zone"] = out["evt_breakout_from_low035_5"] if "evt_breakout_from_low035_5" in out.columns else False
    out["evt_break_resistance_from_mid_zone"] = out["evt_breakout_from_low050_5"] if "evt_breakout_from_low050_5" in out.columns else False

    # 量能/成交额
    vol = out["volume"].astype(float)
    amt = out["amount"].astype(float)
    out["volume_ma_5"] = vol.rolling(5, min_periods=1).mean()
    out["volume_ma_20"] = vol.rolling(20, min_periods=3).mean()
    out["volume_ratio_5"] = vol / out["volume_ma_5"].replace(0, np.nan)
    out["volume_ratio_20"] = vol / out["volume_ma_20"].replace(0, np.nan)
    out["volume_z_20"] = _rolling_z(vol, 20)
    out["volume_z_60"] = _rolling_z(vol, 60)
    out["amount_ma_20"] = amt.rolling(20, min_periods=3).mean()
    out["amount_ratio_20"] = amt / out["amount_ma_20"].replace(0, np.nan)
    out["amount_z_20"] = _rolling_z(amt, 20)
    out["is_volume_expansion"] = (out["volume_ratio_20"] >= 1.50).fillna(False).astype(bool)
    out["is_volume_shrink"] = (out["volume_ratio_20"] <= 0.70).fillna(False).astype(bool)
    out["is_amount_expansion"] = (out["amount_ratio_20"] >= 1.50).fillna(False).astype(bool)

    # 支撑簇事件的量能质量：全量实验中“缩量收回”明显优于“放量回踩/放量收回”。
    if "evt_pierce_support_zone_reclaim" in out.columns:
        out["evt_pierce_support_cluster_reclaim_low_volume"] = (
            out["evt_pierce_support_zone_reclaim"].fillna(False).astype(bool) &
            out["support_cluster_is_strong"].fillna(False).astype(bool) &
            out["is_volume_shrink"].fillna(False).astype(bool)
        ).astype(bool)
        out["evt_pierce_support_cluster_reclaim_high_volume"] = (
            out["evt_pierce_support_zone_reclaim"].fillna(False).astype(bool) &
            out["support_cluster_is_strong"].fillna(False).astype(bool) &
            out["is_volume_expansion"].fillna(False).astype(bool)
        ).astype(bool)
        out["evt_break_strong_support_cluster_high_volume"] = (
            out["evt_break_strong_support_cluster"].fillna(False).astype(bool) &
            out["is_volume_expansion"].fillna(False).astype(bool)
        ).astype(bool)

    # 趋势/波动/前期收益
    for ma in [5, 10, 20, 60]:
        out[f"ma{ma}"] = out["close"].rolling(ma, min_periods=max(2, ma // 3)).mean()
    out["close_above_ma20"] = (out["close"] > out["ma20"]).fillna(False).astype(bool)
    out["close_above_ma60"] = (out["close"] > out["ma60"]).fillna(False).astype(bool)
    out["ma20_slope_pct"] = out["ma20"] / out["ma20"].shift(5) - 1.0
    out["ma60_slope_pct"] = out["ma60"] / out["ma60"].shift(10) - 1.0
    out["ma20_slope_rank_60"] = out["ma20_slope_pct"].rolling(60, min_periods=20).rank(pct=True)
    out["trend_ma_bull"] = ((out["ma20"] > out["ma60"]) & (out["close"] > out["ma20"])).fillna(False).astype(bool)
    out["trend_ma_bear"] = ((out["ma20"] < out["ma60"]) & (out["close"] < out["ma20"])).fillna(False).astype(bool)
    out["hh_hl_structure"] = ((out["support_is_higher_low"] == 1) & (out["resistance_is_higher_high"] == 1)).fillna(False).astype(bool)
    out["lh_ll_structure"] = ((out["support_is_higher_low"] == 0) & (out["resistance_is_higher_high"] == 0)).fillna(False).astype(bool)
    for k in [5, 10, 20]:
        out[f"ret_{k}"] = out["close"] / out["close"].shift(k) - 1.0
    out["ret_rank_60"] = out["ret_20"].rolling(60, min_periods=20).rank(pct=True)
    out["realized_vol_20"] = out["close"].pct_change().rolling(20, min_periods=10).std(ddof=0)
    out["realized_vol_rank_60"] = out["realized_vol_20"].rolling(60, min_periods=20).rank(pct=True)
    out["atr_rank_60"] = out["atr_pct_14"].rolling(60, min_periods=20).rank(pct=True)
    roll_max = out["close"].rolling(20, min_periods=5).max()
    out["max_drawdown_20"] = out["close"] / roll_max - 1.0

    # 事件辅助字段
    out["event_support_price"] = np.where(evt_pierce_support | evt_close_below_support | evt_close_break_support, support_ref, np.nan)
    out["event_resistance_price"] = np.where(evt_high_break_res | evt_close_above_res | evt_cross_res, resistance_ref, np.nan)
    out["event_low"] = np.where(evt_pierce_reclaim | evt_cross_res | evt_wick_break_res_fail | evt_failed_reclaim_support, l, np.nan)
    out["event_high"] = np.where(evt_pierce_reclaim | evt_cross_res | evt_wick_break_res_fail | evt_failed_reclaim_support, h, np.nan)

    # bars since 事件
    for col in [
        "evt_pierce_support_reclaim",
        "evt_failed_reclaim_support",
        "evt_cross_recent_resistance",
        "evt_wick_break_resistance_fail",
        "evt_break_resistance_from_low_zone",
        "evt_close_break_recent_support",
    ]:
        mask = out[col].fillna(False).to_numpy(dtype=bool)
        last = np.full(n, np.nan)
        last_idx: Optional[int] = None
        for i, flag in enumerate(mask):
            if flag:
                last_idx = i
            if last_idx is not None:
                last[i] = i - last_idx
        out[f"bars_since_{col.replace('evt_', '')}"] = last

    # 未来标签最后拼接，避免中间大量复制
    labels = _future_labels(out, cfg.horizons)
    out = pd.concat([out, labels], axis=1)

    return out


# =============================================================================
# 可视化
# =============================================================================
def _volume_weighted_bar_colors(df: pd.DataFrame, sma_len: int = 89, up_thresh: float = 1.618, low_thresh: float = 0.618) -> List[str]:
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    v_sma = vol.rolling(sma_len, min_periods=1).mean()
    colors: List[str] = []
    for o, cl, vv, ma in zip(df["open"], df["close"], vol, v_sma):
        bull = cl >= o
        if np.isfinite(vv) and np.isfinite(ma) and vv > ma * up_thresh:
            colors.append("#006400" if bull else "#910000")
        elif np.isfinite(vv) and np.isfinite(ma) and vv < ma * low_thresh:
            colors.append("#7FFFD4" if bull else "#FF9800")
        else:
            colors.append("#00c853" if bull else "#ff5252")
    return colors


def build_html(df_full: pd.DataFrame, df_plot: pd.DataFrame, out_html: str, title: str, pivot_len: int) -> None:
    if go is None or make_subplots is None:
        raise RuntimeError("未安装 plotly: pip install plotly")

    plot_start = len(df_full) - len(df_plot)
    x_num = np.arange(len(df_plot), dtype=float)
    intraday = len(df_plot.index) > 1 and (df_plot.index[1] - df_plot.index[0]) < pd.Timedelta("20H")
    tick_text = [ts.strftime("%Y-%m-%d %H:%M") if intraday else ts.strftime("%Y-%m-%d") for ts in df_plot.index]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.56, 0.16, 0.14, 0.14],
        subplot_titles=(title, "SR Position 0-1", "Volume / Volume Z", "Event Marker Rows"),
    )

    colors = _volume_weighted_bar_colors(df_plot)
    for i, (op, hi, lo, cl, clr) in enumerate(zip(df_plot["open"], df_plot["high"], df_plot["low"], df_plot["close"], colors)):
        fig.add_trace(go.Scatter(x=[i, i], y=[lo, hi], mode="lines", line=dict(color=clr, width=1), hoverinfo="skip", showlegend=False), row=1, col=1)
        body_low = min(op, cl)
        body_high = max(op, cl)
        fig.add_shape(type="rect", x0=i - 0.32, x1=i + 0.32, y0=body_low, y1=(body_high if body_high > body_low else body_low + 1e-9),
                      line=dict(color=clr, width=1), fillcolor=clr, xref="x1", yref="y1")

    # 支撑/压力阶梯线
    fig.add_trace(go.Scatter(x=x_num, y=df_plot["resistance_ref"], mode="lines", line=dict(width=1.4, dash="dot", color="#00e676"), name="resistance_ref"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df_plot["support_ref"], mode="lines", line=dict(width=1.4, dash="dot", color="#ff5252"), name="support_ref"), row=1, col=1)
    if "flipped_support_ref" in df_plot.columns:
        fig.add_trace(go.Scatter(x=x_num, y=df_plot["flipped_support_ref"], mode="lines", line=dict(width=1.2, dash="dash", color="#2196f3"), name="flipped_support_ref"), row=1, col=1)
    for col, nm, color in [("support_zone_low", "support_zone_low", "#ff8a80"), ("support_zone_high", "support_zone_high", "#ff8a80"), ("resistance_zone_low", "resistance_zone_low", "#69f0ae"), ("resistance_zone_high", "resistance_zone_high", "#69f0ae")]:
        if col in df_plot.columns:
            fig.add_trace(go.Scatter(x=x_num, y=df_plot[col], mode="lines", line=dict(width=0.8, dash="dash", color=color), name=nm, opacity=0.65), row=1, col=1)
    if "ma20" in df_plot.columns:
        fig.add_trace(go.Scatter(x=x_num, y=df_plot["ma20"], mode="lines", line=dict(width=1.0, color="#ffd54f"), name="ma20"), row=1, col=1)
    if "ma60" in df_plot.columns:
        fig.add_trace(go.Scatter(x=x_num, y=df_plot["ma60"], mode="lines", line=dict(width=1.0, color="#ab47bc"), name="ma60"), row=1, col=1)

    def _event_scatter(mask_col: str, y_col: str, name: str, symbol: str, size: int, color: str, y_shift: float = 0.0):
        if mask_col not in df_plot.columns:
            return
        mask = df_plot[mask_col].fillna(False).astype(bool).to_numpy()
        if not mask.any():
            return
        y = df_plot[y_col].to_numpy(dtype=float) + y_shift
        cd_cols = ["sr_pos_raw", "support_ref", "resistance_ref", "volume_ratio_20", "fwd_ret_5", "fwd_mdd_5"]
        custom = np.column_stack([
            df_plot.index.astype(str).to_numpy()[mask],
            *[df_plot[col].to_numpy(dtype=float)[mask] if col in df_plot.columns else np.full(mask.sum(), np.nan) for col in cd_cols],
        ])
        fig.add_trace(
            go.Scatter(
                x=x_num[mask], y=y[mask], mode="markers",
                marker=dict(symbol=symbol, size=size, color=color, line=dict(width=1, color="#ffffff")),
                name=name,
                customdata=custom,
                hovertemplate=(
                    f"{name}<br>%{{customdata[0]}}<br>price=%{{y:.3f}}"
                    "<br>sr_pos_raw=%{customdata[1]:.3f}"
                    "<br>support=%{customdata[2]:.3f}"
                    "<br>resistance=%{customdata[3]:.3f}"
                    "<br>vol_ratio20=%{customdata[4]:.2f}"
                    "<br>fwd_ret_5=%{customdata[5]:.2%}"
                    "<br>fwd_mdd_5=%{customdata[6]:.2%}<extra></extra>"
                ),
            ), row=1, col=1,
        )

    price_pad = float((df_plot["high"].max() - df_plot["low"].min()) * 0.015) if len(df_plot) else 0.0
    _event_scatter("evt_pierce_support_reclaim", "low", "刺破支撑后收回", "star", 16, "#40c4ff", -price_pad * 2)
    _event_scatter("evt_failed_reclaim_support", "low", "刺破支撑失败", "x", 13, "#ff1744", -price_pad * 3)
    _event_scatter("evt_cross_recent_resistance", "high", "上穿最近压力", "triangle-up", 13, "#00e676", price_pad)
    _event_scatter("evt_break_resistance_from_low_zone", "high", "低位上穿压力", "star-triangle-up", 16, "#ffd740", price_pad * 2)
    _event_scatter("evt_wick_break_resistance_fail", "high", "影线突破失败", "triangle-down", 12, "#ffab40", price_pad * 3)
    _event_scatter("evt_close_break_recent_support", "low", "收盘跌破支撑", "triangle-down", 13, "#ff5252", -price_pad)
    _event_scatter("evt_pierce_strong_support_cluster_reclaim", "low", "强支撑簇刺破收回", "diamond", 15, "#18ffff", -price_pad * 4)
    _event_scatter("evt_break_strong_support_cluster", "low", "跌破强支撑簇", "x", 15, "#d50000", -price_pad * 5)
    _event_scatter("evt_break_strong_resistance_cluster", "high", "突破强压力簇", "diamond", 15, "#76ff03", price_pad * 4)
    _event_scatter("evt_wick_break_resistance_cluster_fail", "high", "强压力簇假突破", "x", 14, "#ff6d00", price_pad * 5)

    # pivot 标签画回 anchor 位置
    for full_i in range(plot_start, len(df_full)):
        plot_i = full_i - plot_start
        if not (0 <= plot_i < len(df_plot)):
            continue
        ph = df_full["pvt_high_confirm"].iloc[full_i]
        if np.isfinite(ph):
            anchor = int(df_full["pvt_high_anchor_index"].iloc[full_i]) - plot_start
            if 0 <= anchor < len(df_plot):
                fig.add_annotation(x=anchor, y=float(ph), text=f"R {float(ph):.2f}", showarrow=True, arrowhead=2, arrowsize=1,
                                   arrowwidth=1, arrowcolor="#00e676", ax=0, ay=-24, bgcolor="rgba(0,230,118,0.35)",
                                   font=dict(color="white", size=10), xref="x1", yref="y1")
        pl = df_full["pvt_low_confirm"].iloc[full_i]
        if np.isfinite(pl):
            anchor = int(df_full["pvt_low_anchor_index"].iloc[full_i]) - plot_start
            if 0 <= anchor < len(df_plot):
                fig.add_annotation(x=anchor, y=float(pl), text=f"S {float(pl):.2f}", showarrow=True, arrowhead=2, arrowsize=1,
                                   arrowwidth=1, arrowcolor="#ff5252", ax=0, ay=24, bgcolor="rgba(255,82,82,0.35)",
                                   font=dict(color="white", size=10), xref="x1", yref="y1")

    # 位置因子
    fig.add_trace(go.Scatter(x=x_num, y=df_plot["sr_pos_01"], mode="lines", name="sr_pos_01", line=dict(width=2, color="#40c4ff")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df_plot["sr_pos_raw"], mode="lines", name="sr_pos_raw", line=dict(width=1, color="#b0bec5", dash="dot")), row=2, col=1)
    for yv, col in [(0.0, "#ff5252"), (0.25, "#78909c"), (0.35, "#78909c"), (0.5, "#78909c"), (1.0, "#00e676")]:
        fig.add_hline(y=yv, line_width=1, line_dash="dot", line_color=col, row=2, col=1)

    # 量能
    fig.add_trace(go.Bar(x=x_num, y=df_plot["volume_ratio_20"], name="volume_ratio_20", opacity=0.55), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_num, y=df_plot["volume_z_20"], mode="lines", name="volume_z_20", line=dict(width=1.4, color="#ffd740")), row=3, col=1)
    fig.add_hline(y=1.5, line_width=1, line_dash="dot", line_color="#ffd740", row=3, col=1)
    fig.add_hline(y=0.7, line_width=1, line_dash="dot", line_color="#78909c", row=3, col=1)

    # 事件布尔行
    event_rows = [
        ("evt_close_break_recent_support", 0.0, "收盘跌破"),
        ("evt_failed_reclaim_support", 1.0, "刺破失败"),
        ("evt_pierce_support_reclaim", 2.0, "刺破收回"),
        ("evt_wick_break_resistance_fail", 3.0, "压力假破"),
        ("evt_cross_recent_resistance", 4.0, "上穿压力"),
        ("evt_break_resistance_from_low_zone", 5.0, "低位突破"),
        ("evt_pierce_strong_support_cluster_reclaim", 6.0, "强支撑簇收回"),
        ("evt_break_strong_support_cluster", 7.0, "跌破强支撑簇"),
        ("evt_break_strong_resistance_cluster", 8.0, "突破强压力簇"),
        ("evt_wick_break_resistance_cluster_fail", 9.0, "强压力假破"),
    ]
    for col, base, name in event_rows:
        if col not in df_plot.columns:
            continue
        mask = df_plot[col].fillna(False).astype(bool).to_numpy()
        if mask.any():
            fig.add_trace(go.Scatter(x=x_num[mask], y=np.full(mask.sum(), base), mode="markers", marker=dict(size=10), name=name), row=4, col=1)
    fig.update_yaxes(tickmode="array", tickvals=[x[1] for x in event_rows], ticktext=[x[2] for x in event_rows], row=4, col=1)

    tick_step = max(1, len(df_plot) // 10)
    tickvals = list(range(0, len(df_plot), tick_step))
    if tickvals and tickvals[-1] != len(df_plot) - 1:
        tickvals.append(len(df_plot) - 1)
    ticktext = [tick_text[i] for i in tickvals] if tickvals else []

    fig.update_layout(
        template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified",
        margin=dict(l=50, r=30, t=90, b=40), height=1350,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    for r in [1, 2, 3, 4]:
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True, zeroline=False, row=r, col=1)
        fig.update_yaxes(showgrid=True, zeroline=False, row=r, col=1)
    fig.update_yaxes(range=[-0.20, 1.20], row=2, col=1)
    fig.write_html(out_html, include_plotlyjs="cdn")


# =============================================================================
# CLI
# =============================================================================
def _default_output_paths(symbol: str, freq: str, pivot_len: int, out_dir: str) -> Tuple[str, str, str]:
    stem = f"{symbol}_{freq}_sr_factor_lab_pv{pivot_len}"
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{stem}.csv"), str(p / f"{stem}.html"), str(p / f"{stem}_events_only.csv")


def run_one(args, freq: str) -> Dict[str, str]:
    freq = normalize_freq(freq)
    if args.csv:
        raw = read_kline_csv(args.csv)
        if args.fetch_bars:
            raw = raw.tail(args.fetch_bars)
        symbol = args.symbol or Path(args.csv).stem
    else:
        symbol = args.symbol
        if not symbol:
            raise ValueError("使用 pytdx 拉数据时必须提供 --symbol")
        fetch_bars = max(int(args.fetch_bars), int(args.bars), 2 * int(args.pivot_len) + 200)
        raw = fetch_kline_pytdx(symbol, freq, fetch_bars)

    cfg = LabConfig(
        pivot_len=int(args.pivot_len),
        use_prev_confirmed_level=not args.use_current_confirmed_level,
        horizons=tuple(int(x) for x in args.horizons.split(",") if x.strip()),
        cluster_lookback=int(args.cluster_lookback),
        cluster_tolerance_pct=float(args.cluster_tolerance_pct),
        cluster_tolerance_atr=float(args.cluster_tolerance_atr),
        strong_cluster_count=int(args.strong_cluster_count),
        strong_cluster_score=float(args.strong_cluster_score),
    )
    out_full = compute_sr_factor_lab(raw, cfg)

    if args.out_csv and len(args.freqs_list) == 1:
        out_csv = args.out_csv
        out_html = args.out_html or str(Path(out_csv).with_suffix(".html"))
        events_csv = str(Path(out_csv).with_name(Path(out_csv).stem + "_events_only.csv"))
    else:
        out_csv, out_html, events_csv = _default_output_paths(symbol, freq, cfg.pivot_len, args.out_dir)
        if args.out_html and len(args.freqs_list) == 1:
            out_html = args.out_html

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    out_full.to_csv(out_csv, encoding="utf-8-sig")

    event_mask_cols = [
        "evt_pierce_support_reclaim", "evt_failed_reclaim_support", "evt_cross_recent_resistance",
        "evt_break_resistance_from_low_zone", "evt_wick_break_resistance_fail", "evt_close_break_recent_support",
        "evt_pierce_strong_support_cluster_reclaim", "evt_pierce_support_cluster_reclaim_low_volume",
        "evt_break_strong_support_cluster", "evt_break_strong_resistance_cluster",
        "evt_wick_break_resistance_cluster_fail", "evt_pierce_flipped_support_reclaim",
        "evt_breakdown_flipped_support",
    ]
    mask_any_event = np.zeros(len(out_full), dtype=bool)
    for col in event_mask_cols:
        if col in out_full.columns:
            mask_any_event |= out_full[col].fillna(False).to_numpy(dtype=bool)
    out_full.loc[mask_any_event].to_csv(events_csv, encoding="utf-8-sig")

    out_plot = out_full.tail(int(args.bars)).copy()
    title = f"{symbol} [{freq}] SR Event Factor Lab | pivot_len={cfg.pivot_len}"
    build_html(out_full, out_plot, out_html, title, cfg.pivot_len)

    stats = {
        "symbol": symbol,
        "freq": freq,
        "csv": out_csv,
        "html": out_html,
        "events_csv": events_csv,
        "rows": str(len(out_full)),
        "pierce_support_reclaim": str(int(out_full["evt_pierce_support_reclaim"].sum())),
        "failed_reclaim_support": str(int(out_full["evt_failed_reclaim_support"].sum())),
        "cross_resistance": str(int(out_full["evt_cross_recent_resistance"].sum())),
        "breakout_from_low_zone": str(int(out_full["evt_break_resistance_from_low_zone"].sum())),
        "wick_break_resistance_fail": str(int(out_full["evt_wick_break_resistance_fail"].sum())),
        "close_break_support": str(int(out_full["evt_close_break_recent_support"].sum())),
        "strong_support_cluster_reclaim": str(int(out_full.get("evt_pierce_strong_support_cluster_reclaim", pd.Series(False, index=out_full.index)).sum())),
        "support_cluster_reclaim_low_volume": str(int(out_full.get("evt_pierce_support_cluster_reclaim_low_volume", pd.Series(False, index=out_full.index)).sum())),
        "break_strong_resistance_cluster": str(int(out_full.get("evt_break_strong_resistance_cluster", pd.Series(False, index=out_full.index)).sum())),
    }
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="支撑/压力事件因子实验版：完整 CSV + HTML 标记图")
    p.add_argument("--symbol", default="300133", help="股票代码，默认 300133")
    p.add_argument("--freq", default="w", help="周期：d/w/mo/60m/30m/15m/5m/1m，默认 w")
    p.add_argument("--freqs", default="", help="批量周期，例如 d,w,60m。设置后优先于 --freq")
    p.add_argument("--csv", default="", help="可选：读取本地K线CSV，不走 pytdx")
    p.add_argument("--pivot-len", type=int, default=10, help="pivot 左右确认长度，周线实验建议 10 或 20")
    p.add_argument("--bars", type=int, default=300, help="HTML 展示最近多少根K线")
    p.add_argument("--fetch-bars", type=int, default=1200, help="拉取/读取最近多少根K线参与计算")
    p.add_argument("--out-dir", default="sr_factor_lab_output", help="默认输出目录")
    p.add_argument("--out-csv", default="", help="单周期时可指定 CSV 输出路径")
    p.add_argument("--out-html", default="", help="单周期时可指定 HTML 输出路径")
    p.add_argument("--horizons", default="1,3,5,10,20", help="未来标签周期，例如 1,3,5,10,20")
    p.add_argument("--use-current-confirmed-level", action="store_true", help="事件判断使用当前bar刚确认的水平；默认使用上一bar已确认水平，避免当前确认当前使用")
    p.add_argument("--cluster-lookback", type=int, default=120, help="支撑/压力簇统计回看窗口")
    p.add_argument("--cluster-tolerance-pct", type=float, default=0.015, help="结构位聚类价格容忍度，例如 0.015=1.5%")
    p.add_argument("--cluster-tolerance-atr", type=float, default=0.50, help="结构位聚类 ATR 容忍度")
    p.add_argument("--strong-cluster-count", type=int, default=3, help="强支撑/压力簇的最小结构位数量")
    p.add_argument("--strong-cluster-score", type=float, default=3.0, help="强支撑/压力簇的最小加权分数")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.freqs:
        args.freqs_list = [normalize_freq(x) for x in args.freqs.split(",") if x.strip()]
    else:
        args.freqs_list = [normalize_freq(args.freq)]

    all_stats: List[Dict[str, str]] = []
    for f in args.freqs_list:
        stats = run_one(args, f)
        all_stats.append(stats)
        print("-" * 100)
        print(f"完成: {stats['symbol']} [{stats['freq']}]")
        print(f"完整CSV : {stats['csv']}")
        print(f"事件CSV : {stats['events_csv']}")
        print(f"HTML图  : {stats['html']}")
        print(
            "事件统计: "
            f"刺破收回={stats['pierce_support_reclaim']} | "
            f"刺破失败={stats['failed_reclaim_support']} | "
            f"上穿压力={stats['cross_resistance']} | "
            f"低位突破={stats['breakout_from_low_zone']} | "
            f"压力假破={stats['wick_break_resistance_fail']} | "
            f"收盘跌破={stats['close_break_support']}"
        )

    if len(all_stats) > 1:
        stat_path = Path(args.out_dir) / "sr_factor_lab_stats.csv"
        pd.DataFrame(all_stats).to_csv(stat_path, index=False, encoding="utf-8-sig")
        print("-" * 100)
        print(f"多周期统计汇总: {stat_path}")


if __name__ == "__main__":
    main()
