# -*- coding: utf-8 -*-
"""
Adaptive Market Profile (AMP) - Pine strict replica + factor extraction

目标：
1) 主图尽量按用户提供的 Pine 指标最后一根 bar 语义复刻：
   - 自适应周期检测
   - 通道 / 中轴 / Profile / Most Active Lines
   - 只显示“当前最后一根 bar”对象，不再构造 Pine 中不存在的历史副图
2) 额外提供研究用因子提取（滚动回填）：
   - amp_strength        : pR（Pine cD 里的 Pearson）
   - amp_pos_01          : close 在 [lower, upper] 中的位置，限定到 [0,1]
   - amp_dir             : 方向，按中轴斜率 sign 映射为 {-1,0,1}
   - amp_period          : 当期使用的窗口长度

说明：
- Pine 本体没有历史时序副图；本脚本把“严格复刻绘图”和“研究因子回填”分开。
- 图上只画最后一根 bar 对应对象；CSV 则可导出历史因子序列。

How to run:
python amp_pine_full_replica.py --symbol 000426 --freq d --bars 300 --out amp_000426.html --csv-out amp_000426.csv
python amp_pine_full_replica.py --symbol 605580 --freq d --bars 350 --touches --show-mid
"""
from __future__ import annotations

import argparse
import math
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

CANDIDATE_PERIODS = [50, 60, 70, 80, 90, 100, 115, 130, 145, 160, 180, 200, 220, 250, 280, 310, 340, 370, 400]


@dataclass
class AMPConfig:
    useAdaptive: bool = True
    pI: int = 200
    devMultiplier: float = 2.0
    uL: bool = True

    showRegLine: bool = False
    showMostActiveLines: bool = True
    numActivityLines: int = 2

    showProfile: bool = True
    activityMethod: str = "Volume"  # Touches / Volume
    nFills: int = 23

    regColor: str = "rgba(160,160,160,0.85)"
    channelFill: str = "rgba(144,148,151,0.10)"
    loAct: Tuple[int, int, int, float] = (0, 187, 255, 0.05)
    hiAct: Tuple[int, int, int, float] = (0, 187, 255, 0.75)
    useCustomColor: bool = True
    customActLine: str = "rgba(0,187,255,0.55)"
    minActivityThresholdFrac: float = 0.10


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
        except Exception as exc:
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
        target = max(int(count), 450)
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


def clamp01(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(max(0.0, min(1.0, x)))


def safe_ratio(num: float, den: float, default: float = np.nan) -> float:
    if den == 0 or (not np.isfinite(den)):
        return default
    return float(num / den)


def f_adjust(p: float, use_log: bool) -> float:
    return math.log(max(p, 1e-12)) if use_log else p


def f_unadjust(p: float, use_log: bool) -> float:
    return math.exp(p) if use_log else p


def calc_line_value(startY: float, endY: float, currentBar: float, totalBars: int, use_log: bool) -> float:
    if totalBars <= 0:
        return startY
    v = f_adjust(startY, use_log) + (f_adjust(endY, use_log) - f_adjust(startY, use_log)) * currentBar / totalBars
    return f_unadjust(v, use_log)


def gradient_rgba(percent: float, c1: Tuple[int, int, int, float], c2: Tuple[int, int, int, float]) -> str:
    p = float(np.clip(percent, 0.0, 1.0))
    r = c1[0] + (c2[0] - c1[0]) * p
    g = c1[1] + (c2[1] - c1[1]) * p
    b = c1[2] + (c2[2] - c1[2]) * p
    a = c1[3] + (c2[3] - c1[3]) * p
    return f"rgba({int(round(r))},{int(round(g))},{int(round(b))},{a:.4f})"


def calcDevATF(close_r: np.ndarray, length: int, use_log: bool) -> Tuple[float, float, float, float]:
    if length <= 1:
        return (np.nan, np.nan, np.nan, np.nan)
    n1 = length - 1
    sumX = sumXX = sumYX = sumY = 0.0
    for i in range(1, length + 1):
        val = f_adjust(float(close_r[i - 1]), use_log)
        sumX += i
        sumXX += i * i
        sumYX += i * val
        sumY += val
    denom = length * sumXX - sumX * sumX
    slope = 0.0 if denom == 0 else (length * sumYX - sumX * sumY) / denom
    avg = sumY / length
    intercept = avg - slope * sumX / length + slope

    sumDev = sumDxx = sumDyy = sumDyx = 0.0
    reg = intercept + slope * n1 * 0.5
    sumSlope = intercept
    for i in range(0, n1 + 1):
        v = f_adjust(float(close_r[i]), use_log)
        dxt = v - avg
        dyt = sumSlope - reg
        resid = v - sumSlope
        sumSlope += slope
        sumDxx += dxt * dxt
        sumDyy += dyt * dyt
        sumDyx += dxt * dyt
        sumDev += resid * resid
    unStdDev = math.sqrt(sumDev / n1) if n1 != 0 else 0.0
    divisor = sumDxx * sumDyy
    r = 0.0 if divisor <= 0 else (sumDyx / math.sqrt(divisor))
    if not np.isfinite(r):
        r = 0.0
    return (float(unStdDev), float(r), float(slope), float(intercept))


def cS(close_r: np.ndarray, length: int, use_log: bool) -> Tuple[float, float, float]:
    if length <= 1:
        return (np.nan, np.nan, np.nan)
    sX = sY = sXS = sXY = 0.0
    for i in range(length):
        v = f_adjust(float(close_r[i]), use_log)
        p = i + 1.0
        sX += p
        sY += v
        sXS += p * p
        sXY += v * p
    denom = length * sXS - sX * sX
    sl = 0.0 if denom == 0 else (length * sXY - sX * sY) / denom
    av = sY / length
    ic = av - sl * sX / length + sl
    return (float(sl), float(av), float(ic))


def cD(high_r: np.ndarray, low_r: np.ndarray, close_r: np.ndarray, length: int, sl: float, av: float, ic: float, use_log: bool) -> Tuple[float, float, float, float]:
    uD = dD = sDA = dxx = dyy = dxy = 0.0
    per = length - 1
    dY = ic + sl * per / 2.0
    v = ic
    for j in range(0, per + 1):
        pr = f_adjust(float(high_r[j]), use_log) - v
        if pr > uD:
            uD = pr
        pr = v - f_adjust(float(low_r[j]), use_log)
        if pr > dD:
            dD = pr
        pr = f_adjust(float(close_r[j]), use_log)
        dx = pr - av
        dy = v - dY
        resid = pr - v
        sDA += resid * resid
        dxx += dx * dx
        dyy += dy * dy
        dxy += dx * dy
        v += sl
    sD = math.sqrt(sDA / (per if per != 0 else 1))
    pR = 0.0 if (dxx == 0.0 or dyy == 0.0) else (dxy / math.sqrt(dxx * dyy))
    if not np.isfinite(pR):
        pR = 0.0
    return (float(sD), float(pR), float(uD), float(dD))


def apply_deviation(base_value: float, deviation: float, use_log: bool) -> float:
    return f_unadjust(f_adjust(base_value, use_log) + deviation, use_log)


def detect_period(close_r: np.ndarray, cfg: AMPConfig) -> Tuple[int, Optional[float]]:
    n = len(close_r)
    detected_period = min(cfg.pI, n)
    detected_r: Optional[float] = None
    if cfg.useAdaptive:
        best_r = -1e18
        for p in CANDIDATE_PERIODS:
            if p <= n:
                _, pr, _, _ = calcDevATF(close_r, p, cfg.uL)
                if pr > best_r:  # 严格复刻 Pine 的“首次最大值”优先
                    best_r = pr
                    detected_period = p
                    detected_r = pr
    return int(min(detected_period, n)), detected_r


def compute_amp_core(df_win: pd.DataFrame, cfg: AMPConfig) -> Dict:
    lI = len(df_win)
    if lI < 2:
        raise ValueError("窗口过短")

    close = df_win["close"].to_numpy(float)
    high = df_win["high"].to_numpy(float)
    low = df_win["low"].to_numpy(float)
    volume = df_win["volume"].to_numpy(float)

    close_r = close[::-1]
    high_r = high[::-1]
    low_r = low[::-1]
    vol_r = volume[::-1]

    s, a, ic = cS(close_r, lI, cfg.uL)
    sP = f_unadjust(ic + s * (lI - 1), cfg.uL)
    eP = f_unadjust(ic, cfg.uL)

    sD, pR, _, _ = cD(high_r, low_r, close_r, lI, s, a, ic, cfg.uL)
    dev = cfg.devMultiplier * sD

    upper_start = apply_deviation(sP, +dev, cfg.uL)
    upper_end = apply_deviation(eP, +dev, cfg.uL)
    lower_start = apply_deviation(sP, -dev, cfg.uL)
    lower_end = apply_deviation(eP, -dev, cfg.uL)

    bars = max(1, lI - 1)
    mid_slope = (f_adjust(eP, cfg.uL) - f_adjust(sP, cfg.uL)) / bars

    close_last = float(df_win["close"].iloc[-1])
    pos_01 = clamp01(safe_ratio(close_last - lower_end, upper_end - lower_end))
    amp_dir = 1 if mid_slope > 0 else (-1 if mid_slope < 0 else 0)

    counts = np.zeros(int(cfg.nFills), dtype=float)
    for idx1 in range(int(cfg.nFills)):
        y1_top = calc_line_value(lower_start, upper_start, idx1, cfg.nFills, cfg.uL)
        y1_bottom = calc_line_value(lower_start, upper_start, idx1 + 1, cfg.nFills, cfg.uL)
        y2_top = calc_line_value(lower_end, upper_end, idx1, cfg.nFills, cfg.uL)
        y2_bottom = calc_line_value(lower_end, upper_end, idx1 + 1, cfg.nFills, cfg.uL)
        y1_mid = (y1_top + y1_bottom) / 2.0
        y2_mid = (y2_top + y2_bottom) / 2.0
        c = 0.0
        for j in range(lI):
            line_value = calc_line_value(y1_mid, y2_mid, j, lI - 1, cfg.uL)
            lo_ = float(low_r[lI - 1 - j])
            hi_ = float(high_r[lI - 1 - j])
            if lo_ <= line_value <= hi_:
                c += 1.0 if cfg.activityMethod == "Touches" else float(vol_r[lI - 1 - j])
        counts[idx1] = c

    max_count = float(np.max(counts)) if len(counts) else 1.0
    if max_count <= 0:
        max_count = 1.0
    sorted_idx = list(np.argsort(-counts))

    idx0 = int(sorted_idx[0])
    actY1_0 = calc_line_value(lower_start, upper_start, idx0 + 0.5, cfg.nFills, cfg.uL)
    actY2_0 = calc_line_value(lower_end, upper_end, idx0 + 0.5, cfg.nFills, cfg.uL)
    activity_slope = (f_adjust(actY2_0, cfg.uL) - f_adjust(actY1_0, cfg.uL)) / bars

    x = df_win.index.to_list()
    activity_lines: List[Dict] = []
    profile_polys: List[Dict] = []
    min_threshold = max_count * cfg.minActivityThresholdFrac
    displayed = 0

    if cfg.showMostActiveLines:
        profile_length = int(round(lI / 5))
        for k in range(int(cfg.nFills)):
            if displayed >= cfg.numActivityLines:
                break
            idx = int(sorted_idx[k])
            count = float(counts[idx])
            if count < min_threshold:
                continue
            actY1 = calc_line_value(lower_start, upper_start, idx + 0.5, cfg.nFills, cfg.uL)
            actY2 = calc_line_value(lower_end, upper_end, idx + 0.5, cfg.nFills, cfg.uL)
            start_off = int(round((count / max_count) * profile_length)) if cfg.showProfile else 0
            start_off = min(lI - 1, start_off)
            startY = f_unadjust(f_adjust(actY1, cfg.uL) + activity_slope * start_off, cfg.uL) if cfg.showProfile else actY1
            color = cfg.customActLine if cfg.useCustomColor else gradient_rgba(count / max_count, cfg.loAct, cfg.hiAct)
            activity_lines.append({
                "x": [x[start_off], x[-1]],
                "y": [float(startY), float(actY2)],
                "color": color,
            })
            displayed += 1

    if cfg.showProfile:
        maxProfileBars = 25
        effectiveProfileBars = max(cfg.numActivityLines, min(cfg.nFills, max(maxProfileBars - (cfg.numActivityLines - 2), 2)))
        profile_length = int(round(lI / 5))
        for k in range(int(effectiveProfileBars)):
            idx = int(sorted_idx[k])
            count = float(counts[idx])
            percent = count / max_count
            fill = gradient_rgba(percent, cfg.loAct, cfg.hiAct)
            line_length = int(round(percent * profile_length))
            x2_off = min(lI - 1, line_length)
            y1_top = calc_line_value(lower_start, upper_start, idx, cfg.nFills, cfg.uL)
            y1_bottom = calc_line_value(lower_start, upper_start, idx + 1, cfg.nFills, cfg.uL)
            y2_top = f_unadjust(f_adjust(y1_top, cfg.uL) + activity_slope * line_length, cfg.uL)
            y2_bottom = f_unadjust(f_adjust(y1_bottom, cfg.uL) + activity_slope * line_length, cfg.uL)
            profile_polys.append({
                "fill": fill,
                "x0": x[0],
                "x1": x[x2_off],
                "y0_top": float(y1_top),
                "y0_bot": float(y1_bottom),
                "y1_top": float(y2_top),
                "y1_bot": float(y2_bottom),
            })

    return {
        "window_len": int(lI),
        "strength": float(pR),
        "mid_start": float(sP),
        "mid_end": float(eP),
        "upper_start": float(upper_start),
        "upper_end": float(upper_end),
        "lower_start": float(lower_start),
        "lower_end": float(lower_end),
        "close_last": close_last,
        "pos_01": float(pos_01) if np.isfinite(pos_01) else np.nan,
        "dir": int(amp_dir),
        "mid_slope": float(mid_slope),
        "activity_lines": activity_lines,
        "profile_polys": profile_polys,
    }


def compute_amp_last(df_all: pd.DataFrame, cfg: AMPConfig) -> Dict:
    close_r = df_all["close"].to_numpy(float)[::-1]
    final_period, detected_r = detect_period(close_r, cfg)
    lI = min(len(df_all), int(final_period))
    df_win = df_all.iloc[-lI:].copy()
    core = compute_amp_core(df_win, cfg)
    return {
        "metrics": {
            "window_len": lI,
            "finalPeriod": int(final_period),
            "pearson_best_from_devATF": float(detected_r) if detected_r is not None else np.nan,
            "strength": core["strength"],
            "close_last": core["close_last"],
            "mid_start": core["mid_start"],
            "mid_end": core["mid_end"],
            "upper_start": core["upper_start"],
            "upper_end": core["upper_end"],
            "lower_start": core["lower_start"],
            "lower_end": core["lower_end"],
            "pos_01": core["pos_01"],
            "dir": core["dir"],
            "mid_slope": core["mid_slope"],
        },
        "data_all": df_all,
        "window": df_win,
        "activityLines": core["activity_lines"],
        "profilePolys": core["profile_polys"],
        "cfg": cfg,
    }


def compute_amp_factor_timeseries(df_all: pd.DataFrame, cfg: AMPConfig) -> pd.DataFrame:
    n = len(df_all)
    out = pd.DataFrame(index=df_all.index)
    cols = [
        "amp_strength",
        "amp_pos_01",
        "amp_dir",
        "amp_period",
        "amp_upper",
        "amp_lower",
        "amp_mid",
    ]
    for c in cols:
        out[c] = np.nan

    start_end_idx = max(49, 0)  # 最小候选窗口 50
    for end_idx in range(start_end_idx, n):
        hist = df_all.iloc[: end_idx + 1]
        close_r = hist["close"].to_numpy(float)[::-1]
        try:
            period_i, _ = detect_period(close_r, cfg)
        except Exception:
            continue
        lI = min(len(hist), int(period_i))
        if lI < 2:
            continue
        df_win = hist.iloc[-lI:]
        try:
            core = compute_amp_core(df_win, cfg)
        except Exception:
            continue
        idx = hist.index[-1]
        out.at[idx, "amp_strength"] = core["strength"]
        out.at[idx, "amp_pos_01"] = core["pos_01"]
        out.at[idx, "amp_dir"] = core["dir"]
        out.at[idx, "amp_period"] = lI
        out.at[idx, "amp_upper"] = core["upper_end"]
        out.at[idx, "amp_lower"] = core["lower_end"]
        out.at[idx, "amp_mid"] = core["mid_end"]
    return out


def build_plot(payload: Dict, symbol: str) -> go.Figure:
    df_all = payload["data_all"]
    df_win = payload["window"]
    cfg: AMPConfig = payload["cfg"]
    m = payload["metrics"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=(f"{symbol} AMP Pine Replica", "Volume"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df_all.index,
            open=df_all["open"], high=df_all["high"], low=df_all["low"], close=df_all["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
            showlegend=False,
            name="Price",
        ),
        row=1, col=1,
    )

    vol_colors = np.where(df_all["close"] >= df_all["open"], "rgba(38,166,154,0.6)", "rgba(239,83,80,0.6)")
    fig.add_trace(go.Bar(x=df_all.index, y=df_all["volume"], marker_color=vol_colors, showlegend=False, name="Volume"), row=2, col=1)

    ch_x = [df_win.index[0], df_win.index[-1]]
    fig.add_trace(go.Scatter(x=ch_x, y=[m["upper_start"], m["upper_end"]], mode="lines", line=dict(color=cfg.regColor, width=1), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ch_x, y=[m["lower_start"], m["lower_end"]], mode="lines", line=dict(color=cfg.regColor, width=1), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ch_x, y=[m["upper_start"], m["upper_end"]], mode="lines", line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ch_x, y=[m["lower_start"], m["lower_end"]], mode="lines", fill="tonexty", fillcolor=cfg.channelFill, line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False), row=1, col=1)

    if cfg.showRegLine:
        fig.add_trace(go.Scatter(x=ch_x, y=[m["mid_start"], m["mid_end"]], mode="lines", line=dict(color="rgba(180,180,180,0.7)", width=1, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=1)

    for al in payload["activityLines"]:
        fig.add_trace(go.Scatter(x=al["x"], y=al["y"], mode="lines", line=dict(color=al["color"], width=1), hoverinfo="skip", showlegend=False), row=1, col=1)

    shapes = []
    for poly in payload["profilePolys"]:
        shapes.append(dict(
            type="path",
            xref="x", yref="y",
            path=f"M {poly['x0']} {poly['y0_top']} L {poly['x0']} {poly['y0_bot']} L {poly['x1']} {poly['y1_bot']} L {poly['x1']} {poly['y1_top']} Z",
            fillcolor=poly["fill"], line=dict(width=0), layer="below", opacity=1.0,
        ))

    info_html = (
        f"<b>AMP (strict Pine replica on last bar)</b><br>"
        f"window_len: {m['window_len']}<br>"
        f"finalPeriod: {m['finalPeriod']}<br>"
        f"strength(pR): {m['strength']:.3f}<br>"
        f"best Pearson(devATF): {m['pearson_best_from_devATF']:.3f}<br>"
        f"close: {m['close_last']:.2f}<br>"
        f"upper: {m['upper_start']:.2f} → {m['upper_end']:.2f}<br>"
        f"lower: {m['lower_start']:.2f} → {m['lower_end']:.2f}<br>"
        f"mid: {m['mid_start']:.2f} → {m['mid_end']:.2f}<br>"
        f"pos_01: {m['pos_01']:.3f}<br>"
        f"dir: {m['dir']}<br>"
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.012, y=0.988,
        xanchor="left", yanchor="top", text=info_html, align="left",
        showarrow=False, font=dict(size=11, color="#c9d1d9"),
        bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.15)", borderwidth=1, borderpad=7,
    )

    fig.add_annotation(
        x=df_win.index[0], y=m["lower_start"], xref="x", yref="y",
        text=f"{m['strength']:.3f}", showarrow=False,
        font=dict(size=12, color="rgba(200,200,200,0.9)"),
        bgcolor="rgba(0,0,0,0.25)", bordercolor="rgba(255,255,255,0.15)", borderwidth=1, borderpad=4, yshift=18,
    )

    fig.update_layout(
        shapes=shapes,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=80, b=40),
        height=950,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
    )
    for r in [1, 2]:
        fig.update_xaxes(showgrid=True, zeroline=False, row=r, col=1)
        fig.update_yaxes(showgrid=True, zeroline=False, row=r, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AMP Pine 完整复刻版 + 因子提取")
    p.add_argument("--symbol", required=True, help="A股代码，如 000426")
    p.add_argument("--freq", default="d", help="d/w/mo/1m/5m/15m/30m/60m")
    p.add_argument("--bars", type=int, default=300, help="展示最近 N 根K线")
    p.add_argument("--fetch-bars", type=int, default=1200, help="实际抓取并用于计算的K线数量，建议 > bars")
    p.add_argument("--out", default="amp_pine_full_replica.html", help="输出HTML文件")
    p.add_argument("--csv-out", default="", help="可选：输出CSV文件")
    p.add_argument("--png", default="", help="可选：输出PNG文件")

    p.add_argument("--use-log", action="store_true", dest="use_log", default=True)
    p.add_argument("--no-log", action="store_false", dest="use_log")
    p.add_argument("--no-adaptive", action="store_true")
    p.add_argument("--period", type=int, default=200)
    p.add_argument("--dev", type=float, default=2.0)
    p.add_argument("--fills", type=int, default=23)
    p.add_argument("--active-lines", type=int, default=2)
    p.add_argument("--touches", action="store_true")
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--show-mid", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.freq = normalize_freq(args.freq)
    fetch_bars = max(args.fetch_bars, args.bars, 450)
    raw = fetch_kline_pytdx(args.symbol, args.freq, fetch_bars)

    cfg = AMPConfig(
        useAdaptive=(not args.no_adaptive),
        pI=args.period,
        devMultiplier=args.dev,
        uL=args.use_log,
        activityMethod=("Touches" if args.touches else "Volume"),
        nFills=args.fills,
        numActivityLines=args.active_lines,
        showProfile=(not args.no_profile),
        showRegLine=args.show_mid,
    )

    raw = raw.tail(fetch_bars).copy()
    factors = compute_amp_factor_timeseries(raw, cfg)
    merged = raw.join(factors)
    payload = compute_amp_last(merged.tail(args.bars).copy(), cfg)
    fig = build_plot(payload, args.symbol)

    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"HTML 已生成: {args.out}")
    if args.png:
        fig.write_image(args.png, scale=2)
        print(f"PNG 已生成: {args.png}")

    if args.csv_out:
        export_cols = [
            "open", "high", "low", "close", "volume", "amount",
            "amp_strength", "amp_pos_01", "amp_dir", "amp_period", "amp_upper", "amp_lower", "amp_mid",
        ]
        merged.tail(args.bars)[export_cols].to_csv(args.csv_out, encoding="utf-8-sig")
        print(f"CSV 已生成: {args.csv_out}")


if __name__ == "__main__":
    main()
