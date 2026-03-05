# amp_plotly.py
# -*- coding: utf-8 -*-
"""
AMP 自适应移动通道算法及可视化

Purpose: AMP 指标计算与 Plotly 可视化
Inputs: symbol, freq, bars, 各种 AMP 参数
Outputs: HTML 可视化文件
How to Run: python cores/amp_plotly.py --symbol 600547 --freq d --bars 255 --out amp.html
Examples:
    # 日线周期，255 个 bar
    python cores/amp_plotly.py --symbol 600547 --freq d --bars 255 --out amp.html
    
    # 15 分钟周期，255 个 bar
    python cores/amp_plotly.py --symbol 600547 --freq 15m --bars 255 --out amp_15m.html
    
    # 60 分钟周期，500 个 bar，使用对数坐标
    python cores/amp_plotly.py --symbol 600547 --freq 60m --bars 500 --use-log --out amp_60m.html
    
    # 自定义参数：自适应周期 200，显示 2 条活跃线
    python cores/amp_plotly.py --symbol 600547 --freq d --bars 255 --period 200 --active-lines 2 --out amp.html
Side Effects: 无 (只写 HTML 文件，不写数据库)
"""
from __future__ import annotations

import argparse
import math
import sys
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.pytdx_client import connect_pytdx, get_kline_data, PERIOD_MAP


# python amp_plotly.py --symbol 600547 --years 3 --out sdj_amp.html --use-log --active-lines 2 --fills 23

# =========================
# Config (对应 Pine inputs)
# =========================

@dataclass
class AMPConfig:
    useAdaptive: bool = True
    pI: int = 200
    devMultiplier: float = 2.0
    uL: bool = True  # log calc

    showRegLine: bool = False
    showMostActiveLines: bool = True
    numActivityLines: int = 2

    showProfile: bool = True
    activityMethod: str = "Volume"  # 'Touches' or 'Volume'
    nFills: int = 23

    # colors (RGBA)
    regColor: str = "rgba(160,160,160,0.85)"
    channelFill: str = "rgba(144,148,151,0.10)"

    # profile gradient
    loAct: Tuple[int, int, int, float] = (0, 187, 255, 0.05)   # low activity
    hiAct: Tuple[int, int, int, float] = (0, 187, 255, 0.75)   # high activity

    # activity line color
    useCustomColor: bool = True
    customActLine: str = "rgba(0,187,255,0.55)"

    minActivityThresholdFrac: float = 0.10


CANDIDATE_PERIODS = [50, 60, 70, 80, 90, 100, 115, 130, 145, 160, 180, 200, 220, 250, 280, 310, 340, 370, 400]


# =========================
# Pine math helpers
# =========================

def f_adjust(p: float, use_log: bool) -> float:
    return math.log(p) if use_log else p

def f_unadjust(p: float, use_log: bool) -> float:
    return math.exp(p) if use_log else p

def calc_line_value(startY: float, endY: float, currentBar: int, totalBars: int, use_log: bool) -> float:
    if totalBars <= 0:
        return startY
    s = f_adjust(startY, use_log)
    e = f_adjust(endY, use_log)
    v = s + (e - s) * (currentBar / totalBars)
    return f_unadjust(v, use_log)

def gradient_rgba(percent: float, c1: Tuple[int,int,int,float], c2: Tuple[int,int,int,float]) -> str:
    p = float(np.clip(percent, 0.0, 1.0))
    r = c1[0] + (c2[0] - c1[0]) * p
    g = c1[1] + (c2[1] - c1[1]) * p
    b = c1[2] + (c2[2] - c1[2]) * p
    a = c1[3] + (c2[3] - c1[3]) * p
    return f"rgba({int(round(r))},{int(round(g))},{int(round(b))},{a:.4f})"

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def safe_ratio(num: float, den: float) -> float:
    if den == 0 or (not np.isfinite(den)):
        return 0.0
    return float(num / den)


# =========================
# 1) calcDevATF (严格按 Pine)
# =========================

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

    denom = (length * sumXX - sumX * sumX)
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


# =========================
# 2) cS / cD (严格按 Pine)
# =========================

def cS(close_r: np.ndarray, length: int, use_log: bool) -> Tuple[float, float, float]:
    if length <= 1:
        return (np.nan, np.nan, np.nan)
    sX = sY = sXS = sXY = 0.0
    for i in range(0, length):
        v = f_adjust(float(close_r[i]), use_log)
        p = i + 1.0
        sX += p
        sY += v
        sXS += p * p
        sXY += v * p
    denom = (length * sXS - sX * sX)
    sl = 0.0 if denom == 0 else (length * sXY - sX * sY) / denom
    av = sY / length
    ic = av - sl * sX / length + sl
    return (float(sl), float(av), float(ic))

def cD(high_r: np.ndarray, low_r: np.ndarray, close_r: np.ndarray,
       length: int, sl: float, av: float, ic: float, use_log: bool) -> Tuple[float, float, float, float]:
    uD = dD = sDA = dxx = dyy = dxy = 0.0
    per = length - 1
    dY = ic + sl * per / 2.0
    v = ic
    for j in range(0, per + 1):
        pr1 = f_adjust(float(high_r[j]), use_log) - v
        if pr1 > uD:
            uD = pr1
        pr2 = v - f_adjust(float(low_r[j]), use_log)
        if pr2 > dD:
            dD = pr2
        pr = f_adjust(float(close_r[j]), use_log)
        dx = pr - av
        dy = v - dY
        resid = pr - v
        sDA += resid * resid
        dxx += dx * dx
        dyy += dy * dy
        dxy += dx * dy
        v = v + sl
    sD = math.sqrt(sDA / (per if per != 0 else 1))
    pR = 0.0 if (dxx == 0.0 or dyy == 0.0) else (dxy / math.sqrt(dxx * dyy))
    if not np.isfinite(pR):
        pR = 0.0
    return (float(sD), float(pR), float(uD), float(dD))

def apply_deviation(base_value: float, deviation: float, use_log: bool) -> float:
    return f_unadjust(f_adjust(base_value, use_log) + deviation, use_log)


def fetch_daily_pytdx(symbol: str, start: str, end: str) -> pd.DataFrame:
    """获取日线数据 (使用统一数据源模块)"""
    api = connect_pytdx()
    try:
        cat = PERIOD_MAP['d']
        mkt = 1 if symbol.startswith("6") else 0
        
        all_bars = []
        fetch_count = 800
        start_pos = 0
        
        while True:
            data = api.get_security_bars(cat, mkt, symbol, start_pos, fetch_count)
            if not data:
                break
            all_bars.extend(data)
            if len(data) < fetch_count:
                break
            start_pos += fetch_count
        
        if not all_bars:
            raise RuntimeError("pytdx 无数据")
        
        df = pd.DataFrame(all_bars)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif {'year', 'month', 'day', 'hour', 'minute'}.issubset(df.columns):
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']].astype(int))
        
        df = df[['datetime', 'open', 'high', 'low', 'close', 'vol']]
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df = df.sort_values('datetime').set_index('datetime')
        
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        return df[(df.index >= start_dt) & (df.index <= end_dt)]
    finally:
        api.disconnect()


# =========================
# AMP compute (只按 last bar 计算，但返回全量用于画 K 线)
# =========================

def compute_amp_last(df_all: pd.DataFrame, cfg: AMPConfig) -> Dict:
    n = len(df_all)
    if n < 60:
        raise ValueError("数据太少，至少 60 根日 K")

    close = df_all["close"].to_numpy(float)
    high = df_all["high"].to_numpy(float)
    low = df_all["low"].to_numpy(float)
    vol = df_all["volume"].to_numpy(float)

    # Pine: close[0] 是当前 bar，所以我们倒序
    close_r = close[::-1]
    high_r = high[::-1]
    low_r = low[::-1]
    vol_r = vol[::-1]

    # 1) detect period by highest pearson r (calcDevATF)
    detected_period = cfg.pI
    detected_r = None
    if cfg.useAdaptive:
        best_r = -1e9
        for p in CANDIDATE_PERIODS:
            if p <= n:
                _, pr, _, _ = calcDevATF(close_r, p, cfg.uL)
                if pr > best_r:
                    best_r = pr
                    detected_period = p
                    detected_r = pr

    finalPeriod = detected_period if cfg.useAdaptive else cfg.pI
    lI = min(n, int(finalPeriod))  # window length

    # window slice (仅用于 AMP 绘制段)
    df_win = df_all.iloc[-lI:]
    x = df_win.index.to_list()  # 通道/profile 只覆盖窗口段

    # 2) regression (cS)
    s, a, ic = cS(close_r, lI, cfg.uL)
    sP = f_unadjust(ic + s * (lI - 1), cfg.uL)  # oldest mid
    eP = f_unadjust(ic, cfg.uL)                # current mid

    # 3) deviation stats (cD) -> pR 是你红框 0.979
    sD, pR, _, _ = cD(high_r, low_r, close_r, lI, s, a, ic, cfg.uL)

    # 4) channel boundaries
    dev = cfg.devMultiplier * sD
    uSP = apply_deviation(sP, +dev, cfg.uL)
    uEP = apply_deviation(eP, +dev, cfg.uL)
    lSP = apply_deviation(sP, -dev, cfg.uL)
    lEP = apply_deviation(eP, -dev, cfg.uL)

    # 5) profile counts
    nFills = int(cfg.nFills)
    counts = np.zeros(nFills, float)

    for idx1 in range(nFills):
        y1_top = calc_line_value(lSP, uSP, idx1, nFills, cfg.uL)
        y1_bottom = calc_line_value(lSP, uSP, idx1 + 1, nFills, cfg.uL)
        y2_top = calc_line_value(lEP, uEP, idx1, nFills, cfg.uL)
        y2_bottom = calc_line_value(lEP, uEP, idx1 + 1, nFills, cfg.uL)
        y1_mid = (y1_top + y1_bottom) / 2
        y2_mid = (y2_top + y2_bottom) / 2

        c = 0.0
        for j in range(lI):
            line_val = calc_line_value(y1_mid, y2_mid, j, lI - 1, cfg.uL)
            lo_ = float(low_r[lI - 1 - j])
            hi_ = float(high_r[lI - 1 - j])
            if lo_ <= line_val <= hi_:
                c += 1.0 if cfg.activityMethod == "Touches" else float(vol_r[lI - 1 - j])
        counts[idx1] = c

    maxCount = float(np.max(counts)) if len(counts) else 1.0
    if maxCount <= 0:
        maxCount = 1.0

    sorted_idx = list(np.argsort(-counts))

    # activitySlope from most active band midpoints
    idx0 = int(sorted_idx[0])
    actY1_0 = calc_line_value(lSP, uSP, idx0 + 0.5, nFills, cfg.uL)
    actY2_0 = calc_line_value(lEP, uEP, idx0 + 0.5, nFills, cfg.uL)
    activitySlope = (f_adjust(actY2_0, cfg.uL) - f_adjust(actY1_0, cfg.uL)) / max(1, (lI - 1))

    # Most Active Lines (并记录它们的"通道内高度占比/位置占比")
    activity_lines = []
    activity_line_stats = []
    minThreshold = maxCount * cfg.minActivityThresholdFrac
    displayed = 0

    upper_start = float(uSP)
    lower_start = float(lSP)
    upper_end = float(uEP)
    lower_end = float(lEP)
    channel_h_start = (upper_start - lower_start)
    channel_h_end = (upper_end - lower_end)

    if cfg.showMostActiveLines:
        profileLength = int(round(lI / 5))
        for k in range(nFills):
            if displayed >= cfg.numActivityLines:
                break
            idx = int(sorted_idx[k])
            c = float(counts[idx])
            if c < minThreshold:
                continue

            actY1 = calc_line_value(lSP, uSP, idx + 0.5, nFills, cfg.uL)  # at oldest
            actY2 = calc_line_value(lEP, uEP, idx + 0.5, nFills, cfg.uL)  # at current

            startX_off = int(round((c / maxCount) * profileLength)) if cfg.showProfile else 0
            startX_off = min(lI - 1, startX_off)

            if cfg.showProfile:
                startY = f_unadjust(f_adjust(actY1, cfg.uL) + activitySlope * (startX_off - 0), cfg.uL)
            else:
                startY = actY1

            color = cfg.customActLine if cfg.useCustomColor else gradient_rgba(c/maxCount, cfg.loAct, cfg.hiAct)

            activity_lines.append({
                "x": [x[startX_off], x[-1]],
                "y": [float(startY), float(actY2)],
                "color": color
            })

            # stats: height ratios (0~1) at start/end; position ratio for startX
            height_end = clamp01(safe_ratio(float(actY2) - lower_end, channel_h_end))
            height_start = clamp01(safe_ratio(float(startY) - lower_start, channel_h_start))
            x_start_pos = clamp01(safe_ratio(startX_off, (lI - 1)))

            activity_line_stats.append({
                "rank": displayed + 1,
                "count": c,
                "count_frac": c / maxCount,
                "x_start_pos_0_1": x_start_pos,
                "height_start_0_1": height_start,
                "height_end_0_1": height_end,
                "color": color,
            })

            displayed += 1

    # Profile polygons (parallelograms)
    profile_polys = []
    if cfg.showProfile:
        maxProfileBars = 25
        effectiveProfileBars = max(cfg.numActivityLines, min(nFills, max(maxProfileBars - (cfg.numActivityLines - 2), 2)))
        profileLength = int(round(lI / 5))

        for k in range(int(effectiveProfileBars)):
            idx = int(sorted_idx[k])
            c = float(counts[idx])
            percent = c / maxCount
            fill = gradient_rgba(percent, cfg.loAct, cfg.hiAct)

            lineLength = int(round(percent * profileLength))
            x2_off = min(lI - 1, lineLength)

            y1_top = calc_line_value(lSP, uSP, idx, nFills, cfg.uL)
            y1_bottom = calc_line_value(lSP, uSP, idx + 1, nFills, cfg.uL)

            y2_top = f_unadjust(f_adjust(y1_top, cfg.uL) + activitySlope * lineLength, cfg.uL)
            y2_bottom = f_unadjust(f_adjust(y1_bottom, cfg.uL) + activitySlope * lineLength, cfg.uL)

            profile_polys.append({
                "fill": fill,
                "x0": x[0],
                "x1": x[x2_off],
                "y0_top": float(y1_top),
                "y0_bot": float(y1_bottom),
                "y1_top": float(y2_top),
                "y1_bot": float(y2_bottom),
            })

    # =========================
    # 你要的"通道位置/斜率/强度"输出
    # =========================
    close_last = float(df_all["close"].iloc[-1])

    close_pos_0_1 = clamp01(safe_ratio(close_last - lower_end, (upper_end - lower_end)))

    # slope: 用 adjusted 空间的斜率 (对数) 更贴近 TV 的"收益斜率"
    bars = max(1, (lI - 1))

    def slope_metrics(start_price: float, end_price: float) -> Dict:
        a0 = f_adjust(start_price, cfg.uL)
        a1 = f_adjust(end_price, cfg.uL)
        slope_adj_per_bar = (a1 - a0) / bars  # per bar in adjusted space
        if cfg.uL:
            ret_per_bar = math.exp(slope_adj_per_bar) - 1.0
        else:
            # 线性时：用起点价近似换算成"每 bar 收益率"
            ret_per_bar = safe_ratio((end_price - start_price) / bars, start_price)
        total_ret = safe_ratio(end_price - start_price, start_price) if start_price != 0 else 0.0
        return {
            "slope_adj_per_bar": float(slope_adj_per_bar),
            "ret_per_bar": float(ret_per_bar),
            "total_ret": float(total_ret),
        }

    upper_slope = slope_metrics(upper_start, upper_end)
    lower_slope = slope_metrics(lower_start, lower_end)

    metrics = {
        "window_len": int(lI),
        "finalPeriod": int(finalPeriod),
        "pearson_best_from_devATF": float(detected_r) if detected_r is not None else None,
        "strength_pR_cD": float(pR),

        "close_last": close_last,
        "upper_start": upper_start,
        "lower_start": lower_start,
        "upper_end": upper_end,
        "lower_end": lower_end,
        "close_pos_0_1": float(close_pos_0_1),

        "upper_slope": upper_slope,
        "lower_slope": lower_slope,

        "active_lines_stats": activity_line_stats,  # 中间两条线的位置/高度占比
    }

    return {
        "metrics": metrics,
        "data_all": df_all,      # 全量两年 (画 K 线/量能)
        "window": df_win,        # 窗口段 (通道/profile 覆盖)
        "channel": {
            "x": [x[0], x[-1]],
            "upper": [upper_start, upper_end],
            "lower": [lower_start, lower_end],
            "mid": [float(sP), float(eP)],
            "showMid": cfg.showRegLine,
        },
        "activityLines": activity_lines,
        "profilePolys": profile_polys,
        "cfg": cfg,
    }


# =========================
# Plotly render & save (子图分屏 + HTML 信息框 + 强度标签)
# =========================

def build_plot(payload: Dict, symbol: str = "", stock_name: str = "") -> go.Figure:
    df_all = payload["data_all"]
    cfg: AMPConfig = payload["cfg"]
    m = payload["metrics"]
    ch = payload["channel"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.75, 0.25],
    )

    # Row1: 全量 K 线
    fig.add_trace(
        go.Candlestick(
            x=df_all.index, open=df_all["open"], high=df_all["high"], low=df_all["low"], close=df_all["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
            showlegend=False
        ),
        row=1, col=1
    )

    # Row2: 全量成交量
    vol_colors = np.where(df_all["close"] >= df_all["open"], "rgba(38,166,154,0.6)", "rgba(239,83,80,0.6)")
    fig.add_trace(
        go.Bar(x=df_all.index, y=df_all["volume"], marker_color=vol_colors, showlegend=False),
        row=2, col=1
    )

    # Row1: channel lines
    fig.add_trace(
        go.Scatter(x=ch["x"], y=ch["upper"], mode="lines",
                   line=dict(color=cfg.regColor, width=1),
                   hoverinfo="skip", showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ch["x"], y=ch["lower"], mode="lines",
                   line=dict(color=cfg.regColor, width=1),
                   hoverinfo="skip", showlegend=False),
        row=1, col=1
    )

    # Channel fill
    fig.add_trace(
        go.Scatter(x=ch["x"], y=ch["upper"], mode="lines",
                   line=dict(color="rgba(0,0,0,0)"),
                   hoverinfo="skip", showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ch["x"], y=ch["lower"], mode="lines",
                   fill="tonexty", fillcolor=cfg.channelFill,
                   line=dict(color="rgba(0,0,0,0)"),
                   hoverinfo="skip", showlegend=False),
        row=1, col=1
    )

    # Mid line
    if ch["showMid"]:
        fig.add_trace(
            go.Scatter(x=ch["x"], y=ch["mid"], mode="lines",
                       line=dict(color="rgba(180,180,180,0.7)", width=1, dash="dash"),
                       hoverinfo="skip", showlegend=False),
            row=1, col=1
        )

    # Activity lines
    for al in payload["activityLines"]:
        fig.add_trace(
            go.Scatter(x=al["x"], y=al["y"], mode="lines",
                       line=dict(color=al["color"], width=1),
                       hoverinfo="skip", showlegend=False),
            row=1, col=1
        )

    # Profile polygons as shapes (Row1 axes)
    shapes = []
    for poly in payload["profilePolys"]:
        x0 = poly["x0"]; x1 = poly["x1"]
        y0t = poly["y0_top"]; y0b = poly["y0_bot"]
        y1t = poly["y1_top"]; y1b = poly["y1_bot"]
        shapes.append(dict(
            type="path",
            xref="x", yref="y",
            path=f"M {x0} {y0t} L {x0} {y0b} L {x1} {y1b} L {x1} {y1t} Z",
            fillcolor=poly["fill"],
            line=dict(width=0),
            layer="below",
            opacity=1.0
        ))

    # Title
    stock_display = stock_name if stock_name else symbol
    title = f"{stock_display} AMP (window={m['window_len']} finalPeriod={m['finalPeriod']})"

    fig.update_layout(
        title=title,
        shapes=shapes,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=900,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=2, col=1, rangemode="tozero")

    # =========================
    # HTML 左上角信息框 (你要的所有数值)
    # =========================
    ul = m["upper_slope"]
    ll = m["lower_slope"]

    # DevATF best(可能是 None)
    pearson_best = m.get("pearson_best_from_devATF", None)
    pearson_best_str = "-" if pearson_best is None else f"{pearson_best:.3f}"

    # Most Active Lines 中文输出
    lines_info = ""
    stats = m.get("active_lines_stats", [])
    for st in stats:
        lines_info += (
            f"线{st['rank']}: "
            f"起点横向位置={st['x_start_pos_0_1']:.3f}, "
            f"起点高度占比={st['height_start_0_1']:.3f}, "
            f"终点高度占比={st['height_end_0_1']:.3f}"
            "<br>"
        )
    if not lines_info:
        lines_info = "(无活跃线)<br>"

    info_html = (
        f"<b>AMP 指标信息</b><br>"
        f"趋势窗口长度（bar）：{m['window_len']}<br>"
        f"强度（pR）：{m['strength_pR_cD']:.3f}<br>"
        f"自适应最优（DevATF Pearson）：{pearson_best_str}<br>"
        f"最新收盘价：{m['close_last']:.2f}<br>"
        f"上轨：起点 {m['upper_start']:.2f} → 终点 {m['upper_end']:.2f}<br>"
        f"下轨：起点 {m['lower_start']:.2f} → 终点 {m['lower_end']:.2f}<br>"
        f"收盘在通道内位置（0~1）：{m['close_pos_0_1']:.3f}<br>"
        f"<br><b>斜率/收益</b><br>"
        f"上轨 每 bar 收益：{ul['ret_per_bar']:.5f}，窗口总收益：{ul['total_ret']:.3f}<br>"
        f"下轨 每 bar 收益：{ll['ret_per_bar']:.5f}，窗口总收益：{ll['total_ret']:.3f}<br>"
        f"<br><b>最活跃价位线（Most Active Lines）</b><br>"
        f"{lines_info}"
    )



    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        xanchor="left", yanchor="top",
        text=info_html,
        align="left",
        showarrow=False,
        font=dict(size=12, color="#c9d1d9"),
        bgcolor="rgba(0,0,0,0.35)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        borderpad=8,
    )

    # =========================
    # 强度标签（像你截图红框 0.979）
    # 放在窗口起点附近：x=窗口起点，y=lower_start 附近
    # =========================
    fig.add_annotation(
        x=ch["x"][0],
        y=m["lower_start"],
        xref="x", yref="y",
        text=f"{m['strength_pR_cD']:.3f}",
        showarrow=False,
        font=dict(size=12, color="rgba(200,200,200,0.9)"),
        bgcolor="rgba(0,0,0,0.25)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        borderpad=4,
        yshift=20,
    )

    return fig


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--symbol", type=str, default="000426")
    ap.add_argument("--freq", type=str, default="d", help="周期：1m, 5m, 15m, 30m, 60m, d, w, m")
    ap.add_argument("--bars", type=int, default=255, help="获取的 bar 数量（默认 255）")
    ap.add_argument("--out", type=str, default="amp.html")
    ap.add_argument("--png", type=str, default="")

    ap.add_argument("--use-log", action="store_true", dest="use_log")
    ap.add_argument("--no-adaptive", action="store_true")
    ap.add_argument("--period", type=int, default=200)
    ap.add_argument("--dev", type=float, default=2.0)
    ap.add_argument("--fills", type=int, default=23)
    ap.add_argument("--active-lines", type=int, default=2)
    ap.add_argument("--touches", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--show-mid", action="store_true")

    args = ap.parse_args()

    # 根据 freq 和 bars 获取数据
    api = connect_pytdx()
    try:
        df = get_kline_data(api, args.symbol, args.freq, args.bars)
        
        # 获取股票名称
        from datasource.pytdx_client import get_stock_name
        stock_name = get_stock_name(api, args.symbol)
    finally:
        api.disconnect()

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

    payload = compute_amp_last(df, cfg)
    fig = build_plot(payload, symbol=args.symbol, stock_name=stock_name)

    # 完全离线把 include_plotlyjs=True；用 cdn 文件更小
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {args.out}")

    if args.png:
        # 需要 pip install kaleido
        fig.write_image(args.png, scale=2)
        print(f"[OK] PNG saved: {args.png}")


if __name__ == "__main__":
    main()
