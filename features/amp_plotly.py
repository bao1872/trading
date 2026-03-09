# amp_plotly.py
# -*- coding: utf-8 -*-
"""
AMP 自适应移动通道算法及可视化（含逐 bar 时序特征副图）

功能说明：
- 自适应移动通道（Adaptive Moving Channel）算法，自动识别最优趋势周期
- 逐 bar 计算 AMP 时序特征，支持历史回测和动态分析
- 可视化展示：
  主图：K 线 + AMP 通道（上轨/下轨/中轴）+ 活跃价位线 + 成交量分布
  副图 1：AMP Strength（趋势强度）/ Close Position（收盘位置）/ Channel Width（通道宽度）
  副图 2：Upper Slope / Lower Slope / Mid Slope（三条斜率曲线）
- slope 统一使用 adjusted-space slope（对数空间），避免线性模式下因起点价格过小导致数值爆炸

核心指标：
- amp_strength: 趋势强度（Pearson R 相关系数，范围 -1 到 1，越接近 1 趋势越强）
- mid_slope: 中轴斜率（每 bar 的对数收益率，表示趋势的陡峭程度）
- close_pos: 收盘价在通道内的相对位置（0-1，0 表示在下轨，1 表示在上轨）
- channel_width: 通道宽度（相对中轴的价格波动范围）

How to Run:
  # 基础用法（默认参数）
  python features/amp_plotly.py --symbol 000426 --freq d --bars 255
  
  # 指定输出文件
  python features/amp_plotly.py --symbol 605580 --freq d --bars 300 --out amp_analysis.html
  
  # 使用对数变换（推荐）
  python features/amp_plotly.py --symbol 000426 --use-log
  
  # 固定周期（不使用自适应）
  python features/amp_plotly.py --symbol 000426 --no-adaptive --period 200
  
  # 调整通道宽度（标准差倍数）
  python features/amp_plotly.py --symbol 000426 --dev 2.5
  
  # 使用成交量计算活跃度（默认）vs 使用接触次数
  python features/amp_plotly.py --symbol 000426 --touches
  
  # 显示中轴线
  python features/amp_plotly.py --symbol 000426 --show-mid
  
  # 导出 PNG 图片（需要安装 kaleido）
  python features/amp_plotly.py --symbol 000426 --png amp_chart.png

参数说明：
  --symbol: 股票代码（如 000426）
  --freq: K 线周期，支持 1m/5m/15m/30m/60m/d/w/m，默认 d（日线）
  --bars: 获取的 K 线数量，默认 255
  --out: 输出的 HTML 文件路径，默认 amp.html
  --png: 输出的 PNG 图片路径（可选）
  --use-log: 使用对数变换计算（推荐，默认开启）
  --no-adaptive: 不使用自适应周期，使用固定周期
  --period: 固定周期值，默认 200（仅在 no-adaptive 时有效）
  --dev: 标准差倍数（通道宽度），默认 2.0
  --fills: 通道内部分层数量，默认 23
  --active-lines: 显示的最活跃价位线数量，默认 2
  --touches: 使用接触次数而非成交量计算活跃度
  --no-profile: 不显示成交量分布剖面
  --show-mid: 显示中轴线（虚线）

输出文件：
  - HTML 交互式图表（默认 amp.html）：可在浏览器中查看，支持缩放、悬停查看数值
  - PNG 静态图片（可选）：适合插入报告或文档

依赖：
  - pandas, numpy, plotly
  - 可选：kaleido（用于导出 PNG）

示例输出说明：
  HTML 图表包含 4 个子图：
  1. 主图：K 线 + AMP 通道 + 活跃价位线
  2. 成交量柱状图
  3. AMP 指标副图（Strength/ClosePos/ChannelWidth）
  4. 斜率副图（Upper/Lower/Mid Slope）
  
  左上角信息框显示详细指标数值，包括：
  - 趋势窗口长度、强度值、自适应最优周期
  - 上下轨/中轴的起点和终点价格
  - 收盘位置、通道宽度
  - 三条斜率曲线的具体数值
  - 最活跃价位线的详细信息
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasource.pytdx_client import PERIOD_MAP, connect_pytdx, get_kline_data


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
    activityMethod: str = "Volume"  # 'Touches' or 'Volume'
    nFills: int = 23

    regColor: str = "rgba(160,160,160,0.85)"
    channelFill: str = "rgba(144,148,151,0.10)"
    loAct: Tuple[int, int, int, float] = (0, 187, 255, 0.05)
    hiAct: Tuple[int, int, int, float] = (0, 187, 255, 0.75)
    useCustomColor: bool = True
    customActLine: str = "rgba(0,187,255,0.55)"
    minActivityThresholdFrac: float = 0.10


CANDIDATE_PERIODS = [50, 60, 70, 80, 90, 100, 115, 130, 145, 160, 180, 200, 220, 250, 280, 310, 340, 370, 400]


def f_adjust(p: float, use_log: bool) -> float:
    if use_log:
        return math.log(max(p, 1e-12))
    return p


def f_unadjust(p: float, use_log: bool) -> float:
    return math.exp(p) if use_log else p


def calc_line_value(startY: float, endY: float, currentBar: float, totalBars: int, use_log: bool) -> float:
    if totalBars <= 0:
        return startY
    s = f_adjust(startY, use_log)
    e = f_adjust(endY, use_log)
    v = s + (e - s) * (currentBar / totalBars)
    return f_unadjust(v, use_log)


def gradient_rgba(percent: float, c1: Tuple[int, int, int, float], c2: Tuple[int, int, int, float]) -> str:
    p = float(np.clip(percent, 0.0, 1.0))
    r = c1[0] + (c2[0] - c1[0]) * p
    g = c1[1] + (c2[1] - c1[1]) * p
    b = c1[2] + (c2[2] - c1[2]) * p
    a = c1[3] + (c2[3] - c1[3]) * p
    return f"rgba({int(round(r))},{int(round(g))},{int(round(b))},{a:.4f})"


def clamp01(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(max(0.0, min(1.0, x)))


def safe_ratio(num: float, den: float, default: float = np.nan) -> float:
    if den == 0 or (not np.isfinite(den)):
        return default
    return float(num / den)


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


def detect_period(close_r: np.ndarray, cfg: AMPConfig) -> Tuple[int, Optional[float]]:
    n = len(close_r)
    detected_period = cfg.pI
    detected_r: Optional[float] = None
    if cfg.useAdaptive:
        best_r = -1e9
        for p in CANDIDATE_PERIODS:
            if p <= n:
                _, pr, _, _ = calcDevATF(close_r, p, cfg.uL)
                if pr > best_r:
                    best_r = pr
                    detected_period = p
                    detected_r = pr
    return int(min(detected_period, n)), detected_r


def _compute_core_from_window(df_win: pd.DataFrame, cfg: AMPConfig) -> Dict:
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

    def slope_adj(start_price: float, end_price: float) -> float:
        a0 = f_adjust(start_price, cfg.uL)
        a1 = f_adjust(end_price, cfg.uL)
        return float((a1 - a0) / bars)

    mid_slope = slope_adj(float(sP), float(eP))
    upper_slope = slope_adj(float(upper_start), float(upper_end))
    lower_slope = slope_adj(float(lower_start), float(lower_end))

    close_last = float(df_win["close"].iloc[-1])
    channel_span = upper_end - lower_end
    close_pos = clamp01(safe_ratio(close_last - lower_end, channel_span))

    mid_end = float(eP)
    channel_width = safe_ratio(upper_end - lower_end, abs(mid_end), default=np.nan)

    nFills = int(cfg.nFills)
    counts = np.zeros(nFills, float)
    for idx1 in range(nFills):
        y1_top = calc_line_value(lower_start, upper_start, idx1, nFills, cfg.uL)
        y1_bottom = calc_line_value(lower_start, upper_start, idx1 + 1, nFills, cfg.uL)
        y2_top = calc_line_value(lower_end, upper_end, idx1, nFills, cfg.uL)
        y2_bottom = calc_line_value(lower_end, upper_end, idx1 + 1, nFills, cfg.uL)
        y1_mid = (y1_top + y1_bottom) / 2.0
        y2_mid = (y2_top + y2_bottom) / 2.0

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

    idx0 = int(sorted_idx[0])
    actY1_0 = calc_line_value(lower_start, upper_start, idx0 + 0.5, nFills, cfg.uL)
    actY2_0 = calc_line_value(lower_end, upper_end, idx0 + 0.5, nFills, cfg.uL)
    activitySlope = (f_adjust(actY2_0, cfg.uL) - f_adjust(actY1_0, cfg.uL)) / bars

    x = df_win.index.to_list()
    activity_lines: List[Dict] = []
    activity_line_stats: List[Dict] = []
    minThreshold = maxCount * cfg.minActivityThresholdFrac
    displayed = 0
    channel_h_start = upper_start - lower_start
    channel_h_end = upper_end - lower_end

    if cfg.showMostActiveLines:
        profileLength = int(round(lI / 5))
        for k in range(nFills):
            if displayed >= cfg.numActivityLines:
                break
            idx = int(sorted_idx[k])
            c = float(counts[idx])
            if c < minThreshold:
                continue

            actY1 = calc_line_value(lower_start, upper_start, idx + 0.5, nFills, cfg.uL)
            actY2 = calc_line_value(lower_end, upper_end, idx + 0.5, nFills, cfg.uL)
            startX_off = int(round((c / maxCount) * profileLength)) if cfg.showProfile else 0
            startX_off = min(lI - 1, startX_off)

            if cfg.showProfile:
                startY = f_unadjust(f_adjust(actY1, cfg.uL) + activitySlope * startX_off, cfg.uL)
            else:
                startY = actY1

            color = cfg.customActLine if cfg.useCustomColor else gradient_rgba(c / maxCount, cfg.loAct, cfg.hiAct)
            activity_lines.append({
                "x": [x[startX_off], x[-1]],
                "y": [float(startY), float(actY2)],
                "color": color,
            })
            activity_line_stats.append({
                "rank": displayed + 1,
                "count": c,
                "count_frac": c / maxCount,
                "x_start_pos_0_1": clamp01(safe_ratio(startX_off, lI - 1, default=np.nan)),
                "height_start_0_1": clamp01(safe_ratio(float(startY) - lower_start, channel_h_start, default=np.nan)),
                "height_end_0_1": clamp01(safe_ratio(float(actY2) - lower_end, channel_h_end, default=np.nan)),
                "color": color,
            })
            displayed += 1

    profile_polys: List[Dict] = []
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

            y1_top = calc_line_value(lower_start, upper_start, idx, nFills, cfg.uL)
            y1_bottom = calc_line_value(lower_start, upper_start, idx + 1, nFills, cfg.uL)
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

    return {
        "window_len": int(lI),
        "mid_start": float(sP),
        "mid_end": float(eP),
        "upper_start": float(upper_start),
        "upper_end": float(upper_end),
        "lower_start": float(lower_start),
        "lower_end": float(lower_end),
        "strength_pR_cD": float(pR),
        "close_last": close_last,
        "close_pos_0_1": float(close_pos) if np.isfinite(close_pos) else np.nan,
        "channel_width_t": float(channel_width) if np.isfinite(channel_width) else np.nan,
        "mid_slope": float(mid_slope),
        "upper_slope": float(upper_slope),
        "lower_slope": float(lower_slope),
        "active_lines": activity_lines,
        "active_lines_stats": activity_line_stats,
        "profile_polys": profile_polys,
    }


def compute_amp_last(df_all: pd.DataFrame, cfg: AMPConfig) -> Dict:
    n = len(df_all)
    if n < 60:
        raise ValueError("数据太少，至少 60 根 K 线")

    close = df_all["close"].to_numpy(float)
    close_r = close[::-1]
    finalPeriod, detected_r = detect_period(close_r, cfg)
    lI = min(n, int(finalPeriod))

    df_win = df_all.iloc[-lI:].copy()
    core = _compute_core_from_window(df_win, cfg)

    metrics = {
        "window_len": int(lI),
        "finalPeriod": int(finalPeriod),
        "pearson_best_from_devATF": float(detected_r) if detected_r is not None else None,
        "strength_pR_cD": core["strength_pR_cD"],
        "close_last": core["close_last"],
        "mid_start": core["mid_start"],
        "mid_end": core["mid_end"],
        "upper_start": core["upper_start"],
        "upper_end": core["upper_end"],
        "lower_start": core["lower_start"],
        "lower_end": core["lower_end"],
        "close_pos_0_1": core["close_pos_0_1"],
        "channel_width_t": core["channel_width_t"],
        "mid_slope": core["mid_slope"],
        "upper_slope": core["upper_slope"],
        "lower_slope": core["lower_slope"],
        "active_lines_stats": core["active_lines_stats"],
    }

    return {
        "metrics": metrics,
        "data_all": df_all,
        "window": df_win,
        "channel": {
            "x": [df_win.index[0], df_win.index[-1]],
            "upper": [core["upper_start"], core["upper_end"]],
            "lower": [core["lower_start"], core["lower_end"]],
            "mid": [core["mid_start"], core["mid_end"]],
            "showMid": cfg.showRegLine,
        },
        "activityLines": core["active_lines"],
        "profilePolys": core["profile_polys"],
        "cfg": cfg,
    }


def compute_amp_timeseries(
    df_all: pd.DataFrame,
    cfg: AMPConfig,
    final_period: Optional[int] = None,
    adaptive_period: bool = True,
) -> pd.DataFrame:
    n = len(df_all)
    if n < 5:
        return pd.DataFrame(index=df_all.index)

    out = pd.DataFrame(index=df_all.index)
    cols = [
        "amp_strength_t", "upper_slope_t", "lower_slope_t", "mid_slope_t",
        "channel_width_t", "close_pos_t", "period_t",
    ]
    for c in cols:
        out[c] = np.nan

    if not adaptive_period:
        if final_period is None:
            close_r = df_all["close"].to_numpy(float)[::-1]
            final_period, _ = detect_period(close_r, cfg)
        fixed_lI = min(n, int(final_period))
        start_end_idx = fixed_lI - 1
    else:
        fixed_lI = None
        start_end_idx = max(59, 4)

    for end_idx in range(start_end_idx, n):
        if adaptive_period:
            close_hist = df_all.iloc[:end_idx + 1]["close"].to_numpy(float)
            close_r = close_hist[::-1]
            try:
                period_i, _ = detect_period(close_r, cfg)
            except Exception:
                continue
            lI = min(end_idx + 1, int(period_i))
        else:
            period_i = fixed_lI
            lI = fixed_lI

        if lI is None or lI < 5:
            continue

        df_win = df_all.iloc[end_idx - lI + 1:end_idx + 1]
        try:
            core = _compute_core_from_window(df_win, cfg)
        except Exception:
            continue
        out.iloc[end_idx, out.columns.get_loc("amp_strength_t")] = core["strength_pR_cD"]
        out.iloc[end_idx, out.columns.get_loc("upper_slope_t")] = core["upper_slope"]
        out.iloc[end_idx, out.columns.get_loc("lower_slope_t")] = core["lower_slope"]
        out.iloc[end_idx, out.columns.get_loc("mid_slope_t")] = core["mid_slope"]
        out.iloc[end_idx, out.columns.get_loc("channel_width_t")] = core["channel_width_t"]
        out.iloc[end_idx, out.columns.get_loc("close_pos_t")] = core["close_pos_0_1"]
        out.iloc[end_idx, out.columns.get_loc("period_t")] = lI

    return out


def build_plot(payload: Dict, ts: pd.DataFrame, symbol: str = "", stock_name: str = "") -> go.Figure:
    df_all = payload["data_all"]
    cfg: AMPConfig = payload["cfg"]
    m = payload["metrics"]
    ch = payload["channel"]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=[0.42, 0.15, 0.21, 0.22],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]],
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
    fig.add_trace(
        go.Bar(x=df_all.index, y=df_all["volume"], marker_color=vol_colors, showlegend=False, name="Volume"),
        row=2, col=1,
    )

    fig.add_trace(go.Scatter(x=ch["x"], y=ch["upper"], mode="lines", line=dict(color=cfg.regColor, width=1), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ch["x"], y=ch["lower"], mode="lines", line=dict(color=cfg.regColor, width=1), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ch["x"], y=ch["upper"], mode="lines", line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ch["x"], y=ch["lower"], mode="lines", fill="tonexty", fillcolor=cfg.channelFill, line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False), row=1, col=1)

    if ch["showMid"]:
        fig.add_trace(go.Scatter(x=ch["x"], y=ch["mid"], mode="lines", line=dict(color="rgba(180,180,180,0.7)", width=1, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=1)

    for al in payload["activityLines"]:
        fig.add_trace(go.Scatter(x=al["x"], y=al["y"], mode="lines", line=dict(color=al["color"], width=1), hoverinfo="skip", showlegend=False), row=1, col=1)

    shapes = []
    for poly in payload["profilePolys"]:
        shapes.append(dict(
            type="path",
            xref="x", yref="y",
            path=f"M {poly['x0']} {poly['y0_top']} L {poly['x0']} {poly['y0_bot']} L {poly['x1']} {poly['y1_bot']} L {poly['x1']} {poly['y1_top']} Z",
            fillcolor=poly["fill"],
            line=dict(width=0),
            layer="below",
            opacity=1.0,
        ))

    fig.add_trace(go.Scatter(x=ts.index, y=ts["amp_strength_t"], mode="lines", name="AMP Strength", line=dict(width=1.8, color="#00d1b2")), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["close_pos_t"], mode="lines", name="Close Pos", line=dict(width=1.5, color="#ffd400")), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["channel_width_t"], mode="lines", name="Channel Width", line=dict(width=1.5, color="#ff9800")), row=3, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=ts.index, y=ts["upper_slope_t"], mode="lines", name="Upper Slope", line=dict(width=1.4, color="#40c4ff")), row=4, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["lower_slope_t"], mode="lines", name="Lower Slope", line=dict(width=1.4, color="#ff5252")), row=4, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["mid_slope_t"], mode="lines", name="Mid Slope", line=dict(width=1.6, color="#b388ff")), row=4, col=1)
    fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.35)", row=4, col=1)

    stock_display = stock_name if stock_name else symbol
    title = f"{stock_display} AMP (window={m['window_len']} finalPeriod={m['finalPeriod']})"

    fig.update_layout(
        title=title,
        shapes=shapes,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=900,
        margin=dict(l=40, r=55, t=60, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False)
    for row in [1, 2, 3, 4]:
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", row=row, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="Strength / ClosePos", row=3, col=1, secondary_y=False, range=[-0.02, 1.05])
    fig.update_yaxes(title_text="Channel Width", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Slopes", row=4, col=1)

    pearson_best = m.get("pearson_best_from_devATF", None)
    pearson_best_str = "-" if pearson_best is None else f"{pearson_best:.3f}"

    lines_info = ""
    stats = m.get("active_lines_stats", [])
    for st in stats:
        lines_info += (
            f"线{st['rank']}: 起点横向={st['x_start_pos_0_1']:.3f}, "
            f"起点高度={st['height_start_0_1']:.3f}, 终点高度={st['height_end_0_1']:.3f}<br>"
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
        f"中轴：起点 {m['mid_start']:.2f} → 终点 {m['mid_end']:.2f}<br>"
        f"收盘在通道内位置（0~1）：{m['close_pos_0_1']:.3f}<br>"
        f"通道宽度（相对中轴）：{m['channel_width_t']:.3f}<br>"
        f"<br><b>Adjusted-space Slope</b><br>"
        f"Upper Slope：{m['upper_slope']:.6f}/bar<br>"
        f"Lower Slope：{m['lower_slope']:.6f}/bar<br>"
        f"Mid Slope：{m['mid_slope']:.6f}/bar<br>"
        f"<br><b>最活跃价位线</b><br>"
        f"{lines_info}"
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.012, y=0.988,
        xanchor="left", yanchor="top", text=info_html, align="left",
        showarrow=False, font=dict(size=11, color="#c9d1d9"),
        bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.15)", borderwidth=1, borderpad=7,
    )

    fig.add_annotation(
        x=ch["x"][0], y=m["lower_start"], xref="x", yref="y",
        text=f"{m['strength_pR_cD']:.3f}", showarrow=False,
        font=dict(size=12, color="rgba(200,200,200,0.9)"),
        bgcolor="rgba(0,0,0,0.25)", bordercolor="rgba(255,255,255,0.15)", borderwidth=1, borderpad=4, yshift=18,
    )

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--symbol", type=str, default="000426")
    ap.add_argument("--freq", type=str, default="d", help="周期：1m, 5m, 15m, 30m, 60m, d, w, m")
    ap.add_argument("--bars", type=int, default=255, help="获取的 bar 数量")
    ap.add_argument("--out", type=str, default="amp.html")
    ap.add_argument("--png", type=str, default="")

    ap.add_argument("--use-log", action="store_true", dest="use_log", default=True, help="使用对数变换计算 slope（默认开启）")
    ap.add_argument("--no-log", action="store_false", dest="use_log", help="不使用对数变换（线性空间）")
    ap.add_argument("--no-adaptive", action="store_true")
    ap.add_argument("--period", type=int, default=200)
    ap.add_argument("--dev", type=float, default=2.0)
    ap.add_argument("--fills", type=int, default=23)
    ap.add_argument("--active-lines", type=int, default=2)
    ap.add_argument("--touches", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--show-mid", action="store_true")
    args = ap.parse_args()

    api = connect_pytdx()
    try:
        df = get_kline_data(api, args.symbol, args.freq, args.bars)
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
    ts = compute_amp_timeseries(df, cfg, final_period=payload["metrics"]["finalPeriod"], adaptive_period=True)
    fig = build_plot(payload, ts, symbol=args.symbol, stock_name=stock_name)

    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {args.out}")
    if args.png:
        fig.write_image(args.png, scale=2)
        print(f"[OK] PNG saved: {args.png}")


if __name__ == "__main__":
    main()
