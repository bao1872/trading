# -*- coding: utf-8 -*-
"""
Divergence for Many Indicators v4 (TradingView Pine by LonesomeTheBlue) -> Plotly HTML

目标：尽量贴近 TradingView 原版的可视化与逐 bar 语义。
重点：
1. 支持 MACD / Hist / RSI / Stoch / CCI / MOM / OBV / VWMACD / CMF / MFI / 外部指标
2. 支持 4 类背离：
   - Positive Regular   (底背离)
   - Negative Regular   (顶背离)
   - Positive Hidden    (隐藏底背离)
   - Negative Hidden    (隐藏顶背离)
3. 支持 Plotly HTML 输出，主图风格尽量贴近 TradingView 暗色主题
4. showlast / 删除旧线旧标签语义尽量贴近 Pine 原脚本

How to Run:
    python features/divergence_many_plotly.py --symbol 000426 --freq d --bars 600 --searchdiv Regular/Hidden --out divergence_tv.html
    python features/divergence_many_plotly.py --symbol 000426 --freq d --bars 600 --searchdiv Regular/Hidden --only-pos-divs --out divergence_bottom_only.html
    python features/divergence_many_plotly.py --symbol 600519 --freq d --bars 250 --searchdiv Regular --showpivot --showlines --out茅台背离.html
    python features/divergence_many_plotly.py --symbol 000001 --freq 60m --bars 500 --calcrsi --showmas --out 平安银行_60 分钟_RSI 背离.html

Examples:
    # 基本用法：分析日线，检测所有背离类型
    python features/divergence_many_plotly.py --symbol 000426 --freq d --bars 600 --out output/divergence.html

    # 只检测常规背离（不包含隐藏背离）
    python features/divergence_many_plotly.py --symbol 000426 --freq d --bars 600 --searchdiv Regular --out output/regular_only.html

    # 只画底背离（常规底背离 + 隐藏底背离）
    python features/divergence_many_plotly.py --symbol 000426 --freq d --bars 600 --searchdiv Regular/Hidden --only-pos-divs --out output/bottom_divs.html

    # 使用 RSI 指标检测背离，显示枢轴点和背离线
    python features/divergence_many_plotly.py --symbol 600519 --freq d --bars 250 --calcrsi --no-calcmacd --no-calcobv --showpivot --showlines --out output/rsi_divs.html

    # 60 分钟级别，检测隐藏背离，显示均线
    python features/divergence_many_plotly.py --symbol 000001 --freq 60m --bars 500 --searchdiv Hidden --showmas --out output/hidden_60m.html

    # 指定时间范围
    python features/divergence_many_plotly.py --symbol 300059 --freq d --start 2024-01-01 --end 2024-12-31 --out output/2024_divs.html

Inputs:
    --symbol: A 股代码（6 位数字），如 000426, 600519
    --freq: 周期，支持 5m/15m/30m/60m/d/w/month
    --bars: 未指定 start 时向前回溯的 bar 数量（默认 2500）
    --start: 开始日期（YYYY-MM-DD），与 bars 互斥
    --end: 结束日期（YYYY-MM-DD），默认今天
    --out: 输出 HTML 文件路径（默认 divergence.html）
    --png: 可选输出 PNG 文件路径

    --prd: Pivot 确认周期（默认 5）
    --source: 枢轴源价格，Close 或 High/Low（默认 Close）
    --searchdiv: 背离类型，Regular/Hidden/Regular/Hidden（默认 Regular/Hidden）
    --showindis: 指标显示方式，Full/First Letter/Don't Show（默认 Full）
    --showlimit: 最少背离数量阈值（默认 1）
    --only-pos-divs: 只画底背离（常规底背离 + 隐藏底背离）
    --showpivot: 显示枢轴点标记（H/L）
    --showlines: 显示背离连线
    --showmas: 显示 50/200 日均线
    --shownum: 显示背离数量标签

    --calcmacd/--no-calcmacd: 启用/禁用 MACD 背离检测（默认启用）
    --calcmacda/--no-calcmacda: 启用/禁用 MACD 柱状图背离检测（默认启用）
    --calcrsi/--no-calcrsi: 启用/禁用 RSI 背离检测（默认禁用）
    --calcstoc/--no-calcstoc: 启用/禁用 Stochastic 背离检测（默认禁用）
    --calccci/--no-calccci: 启用/禁用 CCI 背离检测（默认禁用）
    --calcmom/--no-calcmom: 启用/禁用 Momentum 背离检测（默认禁用）
    --calcobv/--no-calcobv: 启用/禁用 OBV 背离检测（默认启用）
    --calcvwmacd/--no-calcvwmacd: 启用/禁用 VWAP MACD 背离检测（默认禁用）
    --calccmf/--no-calccmf: 启用/禁用 CMF 背离检测（默认禁用）
    --calcmfi/--no-calcmfi: 启用/禁用 MFI 背离检测（默认禁用）
    --calcext/--no-calcext: 启用/禁用外部指标背离检测（默认禁用）

Outputs:
    - HTML 文件：交互式 Plotly 图表，包含 K 线、成交量、背离连线、标签等
    - PNG 文件（可选）：静态图片输出

Side Effects:
    - 读取 pytdx 行情数据（通过 datasource/pytdx_client.py）
    - 写入 HTML/PNG 文件到指定路径
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.pytdx_client import connect_pytdx, PERIOD_MAP


@dataclass
class DivConfig:
    prd: int = 5
    source: str = "Close"
    searchdiv: str = "Regular/Hidden"
    showindis: str = "Full"
    showlimit: int = 1
    maxpp: int = 10
    maxbars: int = 100
    shownum: bool = True
    showlast: bool = False
    dontconfirm: bool = False
    showlines: bool = True
    showpivot: bool = False

    calcmacd: bool = True
    calcmacda: bool = True
    calcrsi: bool = False
    calcstoc: bool = False
    calccci: bool = False
    calcmom: bool = False
    calcobv: bool = True
    calcvwmacd: bool = False
    calccmf: bool = False
    calcmfi: bool = False
    calcext: bool = False

    pos_reg_div_col: str = "rgba(245, 222, 66, 1.0)"   # 黄
    neg_reg_div_col: str = "rgba(54, 40, 160, 1.0)"    # 深蓝/紫
    pos_hid_div_col: str = "rgba(0, 214, 106, 1.0)"    # 绿
    neg_hid_div_col: str = "rgba(255, 87, 87, 1.0)"    # 红
    pos_div_text_col: str = "rgba(0,0,0,1.0)"
    neg_div_text_col: str = "rgba(255,255,255,1.0)"

    reg_div_l_style: str = "Solid"
    hid_div_l_style: str = "Dashed"
    reg_div_l_width: int = 2
    hid_div_l_width: int = 1

    showmas: bool = False
    ma50_col: str = "rgba(111, 207, 151, 0.95)"
    ma200_col: str = "rgba(155, 89, 182, 0.95)"

    # 自定义：只画底背离（常规+隐藏底背离）
    only_pos_divs: bool = False


def fetch_bars_pytdx(symbol: str, freq: str, start: str, end: str, *, max_pages: int = 200) -> pd.DataFrame:
    api = connect_pytdx()
    try:
        cat = PERIOD_MAP.get(freq.lower())
        if cat is None:
            raise ValueError(f"不支持的 freq: {freq}. 可选：{list(PERIOD_MAP.keys())}")
        mkt = 1 if symbol.startswith("6") else 0

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        page = 0
        size = 800
        frames: List[pd.DataFrame] = []

        while page < max_pages:
            recs = api.get_security_bars(cat, mkt, symbol, page * size, size)
            if not recs:
                break
            df = pd.DataFrame(recs)
            if df.empty:
                break

            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
                df = df.set_index("datetime")
            elif {"year", "month", "day", "hour", "minute"}.issubset(df.columns):
                df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]].astype(int))
                df = df.set_index("datetime")

            if "vol" in df.columns:
                df = df.rename(columns={"vol": "volume"})

            frames.append(df)
            if len(recs) < size:
                break
            page += 1

        if not frames:
            raise RuntimeError("pytdx 无数据")

        all_df = pd.concat(frames).sort_index()
        out = all_df[(all_df.index >= start_dt) & (all_df.index <= end_dt)].copy()
        if out.empty:
            return out
        for c in ["open", "high", "low", "close", "volume"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["open", "high", "low", "close", "volume"])
        return out
    finally:
        api.disconnect()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def vwma(price: pd.Series, vol: pd.Series, n: int) -> pd.Series:
    pv = (price * vol).rolling(n, min_periods=n).sum()
    vv = vol.rolling(n, min_periods=n).sum()
    return pv / vv


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 10) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = sma(tp, n)
    md = (tp - ma).abs().rolling(n, min_periods=n).mean()
    return (tp - ma) / (0.015 * md)


def stoch_k(close: pd.Series, high: pd.Series, low: pd.Series, n: int = 14) -> pd.Series:
    ll = low.rolling(n, min_periods=n).min()
    hh = high.rolling(n, min_periods=n).max()
    denom = (hh - ll).replace(0.0, np.nan)
    return 100.0 * (close - ll) / denom


def obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    d = close.diff().fillna(0.0)
    direction = np.sign(d)
    return (direction * vol).cumsum()


def cmf(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, n: int = 21) -> pd.Series:
    denom = (high - low).replace(0.0, np.nan)
    cmfm = ((close - low) - (high - close)) / denom
    cmfv = cmfm * vol
    return sma(cmfv, n) / sma(vol, n)


def mfi_pine(src: pd.Series, vol: pd.Series, n: int = 14) -> pd.Series:
    tp = src
    rmf = tp * vol
    delta = tp.diff()
    pos = rmf.where(delta > 0, 0.0)
    neg = rmf.where(delta < 0, 0.0).abs()
    pos_sum = pos.rolling(n, min_periods=n).sum()
    neg_sum = neg.rolling(n, min_periods=n).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    return 100 - (100 / (1 + ratio))


def compute_indicators(df: pd.DataFrame) -> dict[str, pd.Series]:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    macd_line = ema(close, 12) - ema(close, 26)
    signal = ema(macd_line, 9)
    hist = macd_line - signal

    moment = close.diff(10)
    cci10 = cci(high, low, close, 10)
    obv_ = obv(close, vol)
    stk = sma(stoch_k(close, high, low, 14), 3)

    vw_fast = vwma(close, vol, 12)
    vw_slow = vwma(close, vol, 26)
    vwmacd_ = vw_fast - vw_slow

    cmf_ = cmf(high, low, close, vol, 21)
    mfi_ = mfi_pine(close, vol, 14)
    rsi_ = rsi_wilder(close, 14)

    return {
        "macd": macd_line,
        "hist": hist,
        "rsi": rsi_,
        "stoch": stk,
        "cci": cci10,
        "mom": moment,
        "obv": obv_,
        "vwmacd": vwmacd_,
        "cmf": cmf_,
        "mfi": mfi_,
        "external": close,
    }


# 尽量贴近 Pine pivothigh/pivotlow 的确认方式：在确认 bar t 上记录 pivot 值，pivot 实际发生于 t-prd
# 为避免平台高点/低点多点重复，这里要求中心点为窗口中“首次出现的极值”
def pivots_confirmed(src: np.ndarray, prd: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(src)
    ph = np.full(n, np.nan, dtype=float)
    pl = np.full(n, np.nan, dtype=float)
    w = prd

    for t in range(2 * w, n):
        i = t - w
        left = i - w
        right = i + w
        if left < 0 or right >= n:
            continue
        window = src[left:right + 1]
        center = src[i]
        if not np.isfinite(center):
            continue

        maxv = np.nanmax(window)
        minv = np.nanmin(window)
        if center == maxv:
            first_pos = left + int(np.nanargmax(window))
            if first_pos == i:
                ph[t] = center
        if center == minv:
            first_pos = left + int(np.nanargmin(window))
            if first_pos == i:
                pl[t] = center
    return ph, pl


def _line_dash(style: str) -> str:
    if style == "Solid":
        return "solid"
    if style == "Dashed":
        return "dash"
    return "dot"


def pos_reg_or_neg_hid(
    src: np.ndarray,
    close: np.ndarray,
    low: np.ndarray,
    pl_positions: list[int],
    pl_vals: list[float],
    t: int,
    cfg: DivConfig,
    cond: int,
) -> int:
    divlen = 0
    prsc = close if cfg.source == "Close" else low

    if cfg.dontconfirm or (t - 1 >= 0 and (src[t] > src[t - 1] or close[t] > close[t - 1])):
        startpoint = 0 if cfg.dontconfirm else 1
        for x in range(min(cfg.maxpp, len(pl_positions))):
            conf_idx = pl_positions[x]
            if conf_idx == 0:
                break
            length = t - conf_idx + cfg.prd
            if length > cfg.maxbars:
                break
            if length <= 5 or (t - length) < 0:
                continue

            src_sp = src[t - startpoint]
            src_len = src[t - length]
            pr_sp = prsc[t - startpoint]
            plv = pl_vals[x]
            if not (np.isfinite(src_sp) and np.isfinite(src_len) and np.isfinite(pr_sp) and np.isfinite(plv)):
                continue

            if cond == 1:
                ok = (src_sp > src_len) and (pr_sp < plv)
            else:
                ok = (src_sp < src_len) and (pr_sp > plv)
            if not ok:
                continue

            slope1 = (src_sp - src_len) / (length - startpoint)
            virtual1 = src_sp - slope1
            c_sp = close[t - startpoint]
            c_len = close[t - length]
            slope2 = (c_sp - c_len) / (length - startpoint)
            virtual2 = c_sp - slope2

            arrived = True
            for y in range(1 + startpoint, length):
                idx = t - y
                if idx < 0:
                    arrived = False
                    break
                if src[idx] < virtual1 or close[idx] < virtual2:
                    arrived = False
                    break
                virtual1 -= slope1
                virtual2 -= slope2
            if arrived:
                divlen = length
                break
    return divlen


def neg_reg_or_pos_hid(
    src: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    ph_positions: list[int],
    ph_vals: list[float],
    t: int,
    cfg: DivConfig,
    cond: int,
) -> int:
    divlen = 0
    prsc = close if cfg.source == "Close" else high

    if cfg.dontconfirm or (t - 1 >= 0 and (src[t] < src[t - 1] or close[t] < close[t - 1])):
        startpoint = 0 if cfg.dontconfirm else 1
        for x in range(min(cfg.maxpp, len(ph_positions))):
            conf_idx = ph_positions[x]
            if conf_idx == 0:
                break
            length = t - conf_idx + cfg.prd
            if length > cfg.maxbars:
                break
            if length <= 5 or (t - length) < 0:
                continue

            src_sp = src[t - startpoint]
            src_len = src[t - length]
            pr_sp = prsc[t - startpoint]
            phv = ph_vals[x]
            if not (np.isfinite(src_sp) and np.isfinite(src_len) and np.isfinite(pr_sp) and np.isfinite(phv)):
                continue

            if cond == 1:
                ok = (src_sp < src_len) and (pr_sp > phv)
            else:
                ok = (src_sp > src_len) and (pr_sp < phv)
            if not ok:
                continue

            slope1 = (src_sp - src_len) / (length - startpoint)
            virtual1 = src_sp - slope1
            c_sp = close[t - startpoint]
            c_len = close[t - length]
            slope2 = (c_sp - c_len) / (length - startpoint)
            virtual2 = c_sp - slope2

            arrived = True
            for y in range(1 + startpoint, length):
                idx = t - y
                if idx < 0:
                    arrived = False
                    break
                if src[idx] > virtual1 or close[idx] > virtual2:
                    arrived = False
                    break
                virtual1 -= slope1
                virtual2 -= slope2
            if arrived:
                divlen = length
                break
    return divlen


def calculate_divs(
    enabled: bool,
    src: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ph_positions: list[int],
    ph_vals: list[float],
    pl_positions: list[int],
    pl_vals: list[float],
    t: int,
    cfg: DivConfig,
) -> list[int]:
    out = [0, 0, 0, 0]
    if not enabled:
        return out

    if cfg.searchdiv in ("Regular", "Regular/Hidden"):
        out[0] = pos_reg_or_neg_hid(src, close, low, pl_positions, pl_vals, t, cfg, cond=1)  # 正常底背离
        out[1] = neg_reg_or_pos_hid(src, close, high, ph_positions, ph_vals, t, cfg, cond=1) # 正常顶背离

    if cfg.searchdiv in ("Hidden", "Regular/Hidden"):
        out[2] = pos_reg_or_neg_hid(src, close, low, pl_positions, pl_vals, t, cfg, cond=2)   # positive hidden
        out[3] = neg_reg_or_pos_hid(src, close, high, ph_positions, ph_vals, t, cfg, cond=2)  # negative hidden
    return out


def build_indicator_names(cfg: DivConfig) -> list[str]:
    if cfg.showindis == "Don't Show":
        return [""] * 11
    if cfg.showindis == "Full":
        return ["MACD", "Hist", "RSI", "Stoch", "CCI", "MOM", "OBV", "VWMACD", "CMF", "MFI", "Extrn"]
    return ["M", "H", "E", "S", "C", "M", "O", "V", "C", "M", "X"]


def build_div_colors(cfg: DivConfig) -> list[str]:
    return [cfg.pos_reg_div_col, cfg.neg_reg_div_col, cfg.pos_hid_div_col, cfg.neg_hid_div_col]


def make_line_dict(x1: str, y1: float, x2: str, y2: float, color: str, width: int, dash: str) -> dict:
    return {"x": [x1, x2], "y": [y1, y2], "color": color, "width": width, "dash": dash}


# 尽量模拟 Pine 删除旧线/旧标签语义
# pos: bottom divergences (y 0/2)
# neg: top divergences    (y 1/3)
def run_divergence_engine(df: pd.DataFrame, cfg: DivConfig):
    ind = compute_indicators(df)
    names = build_indicator_names(cfg)
    div_colors = build_div_colors(cfg)

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    pivot_src_h = (df["close"] if cfg.source == "Close" else df["high"]).to_numpy(dtype=float)
    pivot_src_l = (df["close"] if cfg.source == "Close" else df["low"]).to_numpy(dtype=float)
    ph_conf, _ = pivots_confirmed(pivot_src_h, cfg.prd)
    _, pl_conf = pivots_confirmed(pivot_src_l, cfg.prd)

    maxarraysize = 20
    ph_positions: list[int] = []
    pl_positions: list[int] = []
    ph_vals: list[float] = []
    pl_vals: list[float] = []

    pos_div_lines: list[dict] = []
    neg_div_lines: list[dict] = []
    pos_div_labels: list[tuple[str, float, str, str, str]] = []
    neg_div_labels: list[tuple[str, float, str, str, str]] = []

    # 记录每次新绘制数量，用于删除最后一批
    last_pos_div_lines = 0
    last_neg_div_lines = 0
    remove_last_pos_divs = False
    remove_last_neg_divs = False

    def delete_old_pos_div_lines():
        pos_div_lines.clear()

    def delete_old_neg_div_lines():
        neg_div_lines.clear()

    def delete_old_pos_div_labels():
        pos_div_labels.clear()

    def delete_old_neg_div_labels():
        neg_div_labels.clear()

    def delete_last_pos_div_lines_label(n: int):
        nonlocal pos_div_lines, pos_div_labels
        if n > 0 and len(pos_div_lines) >= n:
            for _ in range(n):
                pos_div_lines.pop()
            if pos_div_labels:
                pos_div_labels.pop()

    def delete_last_neg_div_lines_label(n: int):
        nonlocal neg_div_lines, neg_div_labels
        if n > 0 and len(neg_div_lines) >= n:
            for _ in range(n):
                neg_div_lines.pop()
            if neg_div_labels:
                neg_div_labels.pop()

    startpoint = 0 if cfg.dontconfirm else 1
    xcat = [pd.Timestamp(x).strftime("%Y-%m-%d %H:%M") for x in df.index]
    n = len(df)

    series_list = [
        ("macd", cfg.calcmacd),
        ("hist", cfg.calcmacda),
        ("rsi", cfg.calcrsi),
        ("stoch", cfg.calcstoc),
        ("cci", cfg.calccci),
        ("mom", cfg.calcmom),
        ("obv", cfg.calcobv),
        ("vwmacd", cfg.calcvwmacd),
        ("cmf", cfg.calccmf),
        ("mfi", cfg.calcmfi),
        ("external", cfg.calcext),
    ]

    for t in range(n):
        # Pine: 确认时把当前 bar_index（确认 bar）塞进数组，后续用 +prd 折回 pivot 实际位置
        if np.isfinite(ph_conf[t]):
            ph_positions.insert(0, t)
            ph_vals.insert(0, float(ph_conf[t]))
            if len(ph_positions) > maxarraysize:
                ph_positions.pop()
                ph_vals.pop()

        if np.isfinite(pl_conf[t]):
            pl_positions.insert(0, t)
            pl_vals.insert(0, float(pl_conf[t]))
            if len(pl_positions) > maxarraysize:
                pl_positions.pop()
                pl_vals.pop()

        # 遇到新 pivot，重置“可删除最后一批”计数器
        if np.isfinite(pl_conf[t]):
            remove_last_pos_divs = False
            last_pos_div_lines = 0
        if np.isfinite(ph_conf[t]):
            remove_last_neg_divs = False
            last_neg_div_lines = 0

        all_divs = [0] * 44
        for i, (key, enabled) in enumerate(series_list):
            if enabled:
                src_arr = ind[key].to_numpy(dtype=float)
                divs4 = calculate_divs(
                    enabled=True,
                    src=src_arr,
                    close=close,
                    high=high,
                    low=low,
                    ph_positions=ph_positions,
                    ph_vals=ph_vals,
                    pl_positions=pl_positions,
                    pl_vals=pl_vals,
                    t=t,
                    cfg=cfg,
                )
            else:
                divs4 = [0, 0, 0, 0]
            for k in range(4):
                all_divs[i * 4 + k] = int(divs4[k])

        total_div = sum(1 for v in all_divs if v > 0)
        if total_div < cfg.showlimit:
            all_divs = [0] * 44

        divergence_text_top = ""
        divergence_text_bottom = ""
        distances: set[int] = set()
        dnumdiv_top = 0
        dnumdiv_bottom = 0
        top_label_col = "rgba(255,255,255,1.0)"
        bottom_label_col = "rgba(255,255,255,1.0)"
        old_pos_divs_can_be_removed = True
        old_neg_divs_can_be_removed = True

        for x in range(11):
            div_type = -1
            for y in range(4):
                if cfg.only_pos_divs and y not in (0, 2):
                    continue

                dist = all_divs[x * 4 + y]
                if dist <= 0:
                    continue

                div_type = y
                if (y % 2) == 1:
                    dnumdiv_top += 1
                    top_label_col = div_colors[y]
                else:
                    dnumdiv_bottom += 1
                    bottom_label_col = div_colors[y]

                if dist not in distances:
                    distances.add(dist)
                    if cfg.showlines:
                        x1_idx = max(t - dist, 0)
                        x2_idx = max(t - startpoint, 0)
                        x1 = xcat[x1_idx]
                        x2 = xcat[x2_idx]

                        if cfg.source == "Close":
                            y1 = float(close[x1_idx])
                            y2 = float(close[x2_idx])
                        else:
                            if (y % 2) == 0:
                                y1 = float(low[x1_idx])
                                y2 = float(low[x2_idx])
                            else:
                                y1 = float(high[x1_idx])
                                y2 = float(high[x2_idx])

                        new_line = make_line_dict(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            color=div_colors[y],
                            width=(cfg.reg_div_l_width if y < 2 else cfg.hid_div_l_width),
                            dash=_line_dash(cfg.reg_div_l_style if y < 2 else cfg.hid_div_l_style),
                        )
                    else:
                        new_line = None

                    if (y % 2) == 0:
                        if old_pos_divs_can_be_removed:
                            old_pos_divs_can_be_removed = False
                            if not cfg.showlast and remove_last_pos_divs:
                                delete_last_pos_div_lines_label(last_pos_div_lines)
                                last_pos_div_lines = 0
                            if cfg.showlast:
                                delete_old_pos_div_lines()
                        if new_line is not None:
                            pos_div_lines.append(new_line)
                            last_pos_div_lines += 1
                        remove_last_pos_divs = True
                    else:
                        if old_neg_divs_can_be_removed:
                            old_neg_divs_can_be_removed = False
                            if not cfg.showlast and remove_last_neg_divs:
                                delete_last_neg_div_lines_label(last_neg_div_lines)
                                last_neg_div_lines = 0
                            if cfg.showlast:
                                delete_old_neg_div_lines()
                        if new_line is not None:
                            neg_div_lines.append(new_line)
                            last_neg_div_lines += 1
                        remove_last_neg_divs = True

            if div_type >= 0 and names[x] != "":
                if (div_type % 2) == 1 and not cfg.only_pos_divs:
                    divergence_text_top += names[x] + "\n"
                if (div_type % 2) == 0:
                    divergence_text_bottom += names[x] + "\n"

        if (cfg.showindis != "Don't Show") or cfg.shownum:
            if cfg.shownum and dnumdiv_top > 0 and not cfg.only_pos_divs:
                divergence_text_top += str(dnumdiv_top)
            if cfg.shownum and dnumdiv_bottom > 0:
                divergence_text_bottom += str(dnumdiv_bottom)

            if divergence_text_top.strip() != "" and not cfg.only_pos_divs:
                if cfg.showlast:
                    delete_old_neg_div_labels()
                y_top = float(max(high[t], high[t - 1])) if t - 1 >= 0 else float(high[t])
                neg_div_labels.append((xcat[t], y_top, divergence_text_top.strip(), top_label_col, cfg.neg_div_text_col))

            if divergence_text_bottom.strip() != "":
                if cfg.showlast:
                    delete_old_pos_div_labels()
                y_bot = float(min(low[t], low[t - 1])) if t - 1 >= 0 else float(low[t])
                pos_div_labels.append((xcat[t], y_bot, divergence_text_bottom.strip(), bottom_label_col, cfg.pos_div_text_col))

    return pos_div_lines, neg_div_lines, pos_div_labels, neg_div_labels, ph_conf, pl_conf


def build_plot(df: pd.DataFrame, cfg: DivConfig, title: str, out_html: str, out_png: str = "") -> None:
    pos_lines, neg_lines, pos_labels, neg_labels, ph_conf, pl_conf = run_divergence_engine(df, cfg)

    xcat = df.index.strftime("%Y-%m-%d %H:%M").tolist() if isinstance(df.index, pd.DatetimeIndex) else [str(x) for x in df.index]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.80, 0.20],
        specs=[[{"type": "xy"}], [{"type": "xy"}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=xcat,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            showlegend=False,
            name="Price",
        ),
        row=1,
        col=1,
    )

    if cfg.showmas:
        ma50 = df["close"].rolling(50, min_periods=50).mean()
        ma200 = df["close"].rolling(200, min_periods=200).mean()
        fig.add_trace(
            go.Scatter(x=xcat, y=ma50, mode="lines", line=dict(color=cfg.ma50_col, width=1.2), hoverinfo="skip", showlegend=False),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=xcat, y=ma200, mode="lines", line=dict(color=cfg.ma200_col, width=1.2), hoverinfo="skip", showlegend=False),
            row=1,
            col=1,
        )

    vol_colors = np.where(df["close"] >= df["open"], "rgba(38,166,154,0.60)", "rgba(239,83,80,0.60)")
    fig.add_trace(
        go.Bar(x=xcat, y=df["volume"], marker_color=vol_colors, showlegend=False, name="Volume"),
        row=2,
        col=1,
    )

    def add_lines(lines: list[dict]):
        for ln in lines:
            fig.add_trace(
                go.Scatter(
                    x=ln["x"],
                    y=ln["y"],
                    mode="lines",
                    line=dict(color=ln["color"], width=ln["width"], dash=ln["dash"]),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    if cfg.showlines:
        add_lines(pos_lines)
        if not cfg.only_pos_divs:
            add_lines(neg_lines)

    if cfg.showpivot:
        xs_h, ys_h, xs_l, ys_l = [], [], [], []
        for t in range(len(df)):
            if np.isfinite(ph_conf[t]) and (t - cfg.prd) >= 0:
                xs_h.append(xcat[t - cfg.prd])
                ys_h.append(float(df["high"].iloc[t - cfg.prd]))
            if np.isfinite(pl_conf[t]) and (t - cfg.prd) >= 0:
                xs_l.append(xcat[t - cfg.prd])
                ys_l.append(float(df["low"].iloc[t - cfg.prd]))

        if xs_h:
            fig.add_trace(
                go.Scatter(x=xs_h, y=ys_h, mode="markers+text", text=["H"] * len(xs_h), textposition="top center",
                           marker=dict(size=5, color="rgba(255,90,90,0.85)"), hoverinfo="skip", showlegend=False),
                row=1, col=1,
            )
        if xs_l:
            fig.add_trace(
                go.Scatter(x=xs_l, y=ys_l, mode="markers+text", text=["L"] * len(xs_l), textposition="bottom center",
                           marker=dict(size=5, color="rgba(0,220,120,0.85)"), hoverinfo="skip", showlegend=False),
                row=1, col=1,
            )

    # TradingView 风格的气泡标签：上方标签向下指，下方标签向上指
    for (x, y, text_, bg, tc) in neg_labels:
        fig.add_annotation(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text=text_.replace("\n", "<br>"),
            showarrow=True,
            arrowhead=0,
            ax=0,
            ay=-28,
            bgcolor=bg,
            bordercolor=bg,
            borderwidth=1,
            borderpad=6,
            font=dict(color=tc, size=11),
            row=1,
            col=1,
        )

    for (x, y, text_, bg, tc) in pos_labels:
        fig.add_annotation(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text=text_.replace("\n", "<br>"),
            showarrow=True,
            arrowhead=0,
            ax=0,
            ay=28,
            bgcolor=bg,
            bordercolor=bg,
            borderwidth=1,
            borderpad=6,
            font=dict(color=tc, size=11),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=title,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#d6dde5"),
        height=940,
        margin=dict(l=35, r=28, t=48, b=32),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        rangeslider_visible=False,
        type="category",
        categoryorder="array",
        categoryarray=xcat,
        showline=False,
        zeroline=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, rangemode="tozero", row=2, col=1)

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_html}")

    if out_png:
        fig.write_image(out_png, scale=2)
        print(f"[OK] PNG saved: {out_png}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--symbol", type=str, default="000426", help="A 股代码")
    ap.add_argument("--freq", type=str, default="d", help="周期：5m/15m/30m/60m/d/w/month")
    ap.add_argument("--start", type=str, default="", help="开始日期时间")
    ap.add_argument("--end", type=str, default="", help="结束日期时间")
    ap.add_argument("--bars", type=int, default=2500, help="未指定 start 时向前粗估回溯的 bar 数")
    ap.add_argument("--out", type=str, default="divergence.html", help="输出 HTML")
    ap.add_argument("--png", type=str, default="", help="可选输出 PNG")

    ap.add_argument("--prd", type=int, default=5)
    ap.add_argument("--source", type=str, default="Close", choices=["Close", "High/Low"])
    ap.add_argument("--searchdiv", type=str, default="Regular/Hidden", choices=["Regular", "Hidden", "Regular/Hidden"])
    ap.add_argument("--showindis", type=str, default="Full", choices=["Full", "First Letter", "Don't Show"])
    ap.add_argument("--showlimit", type=int, default=1)
    ap.add_argument("--maxpp", type=int, default=10)
    ap.add_argument("--maxbars", type=int, default=100)
    ap.add_argument("--shownum", action="store_true")
    ap.add_argument("--no-shownum", action="store_true")
    ap.add_argument("--showlast", action="store_true")
    ap.add_argument("--dontconfirm", action="store_true")
    ap.add_argument("--showlines", action="store_true")
    ap.add_argument("--no-showlines", action="store_true")
    ap.add_argument("--showpivot", action="store_true")
    ap.add_argument("--showmas", action="store_true")
    ap.add_argument("--only-pos-divs", action="store_true", help="只画底背离：常规底背离 + 隐藏底背离")

    def add_bool(name: str):
        ap.add_argument(f"--{name}", action="store_true", help=f"Enable {name}")
        ap.add_argument(f"--no-{name}", action="store_true", help=f"Disable {name}")

    for k in ["calcmacd", "calcmacda", "calcrsi", "calcstoc", "calccci", "calcmom", "calcobv", "calcvwmacd", "calccmf", "calcmfi", "calcext"]:
        add_bool(k)

    args = ap.parse_args()

    end_dt = pd.to_datetime(args.end) if args.end else pd.Timestamp.now()
    if args.start:
        start_dt = pd.to_datetime(args.start)
    else:
        f = str(args.freq).lower().strip()
        if f in ("5", "5m", "m5"):
            start_dt = end_dt - pd.Timedelta(minutes=5 * args.bars)
        elif f in ("15", "15m", "m15"):
            start_dt = end_dt - pd.Timedelta(minutes=15 * args.bars)
        elif f in ("30", "30m", "m30"):
            start_dt = end_dt - pd.Timedelta(minutes=30 * args.bars)
        elif f in ("60", "60m", "1h", "h1", "hour"):
            start_dt = end_dt - pd.Timedelta(hours=1 * args.bars)
        elif f in ("w", "1w", "week", "weekly"):
            start_dt = end_dt - pd.Timedelta(days=7 * args.bars)
        elif f in ("mon", "m", "1m", "month", "monthly"):
            start_dt = end_dt - pd.Timedelta(days=30 * args.bars)
        else:
            start_dt = end_dt - pd.Timedelta(days=1 * args.bars)

    df = fetch_bars_pytdx(args.symbol, args.freq, start=str(start_dt), end=str(end_dt))
    if df.empty:
        raise RuntimeError(f"无数据：symbol={args.symbol} freq={args.freq} start={start_dt} end={end_dt}")

    cfg = DivConfig(
        prd=args.prd,
        source=args.source,
        searchdiv=args.searchdiv,
        showindis=args.showindis,
        showlimit=args.showlimit,
        maxpp=args.maxpp,
        maxbars=args.maxbars,
        shownum=(False if args.no_shownum else (True if args.shownum else True)),
        showlast=args.showlast,
        dontconfirm=args.dontconfirm,
        showlines=(False if args.no_showlines else (True if args.showlines else True)),
        showpivot=args.showpivot,
        showmas=args.showmas,
        only_pos_divs=args.only_pos_divs,
    )

    for k in ["calcmacd", "calcmacda", "calcrsi", "calcstoc", "calccci", "calcmom", "calcobv", "calcvwmacd", "calccmf", "calcmfi", "calcext"]:
        if getattr(args, f"no_{k}".replace("-", "_"), False):
            setattr(cfg, k, False)
        elif getattr(args, k, False):
            setattr(cfg, k, True)

    title = f"{args.symbol} {args.freq} Divergence (prd={cfg.prd}, {cfg.searchdiv})"
    if cfg.only_pos_divs:
        title += " [Positive Regular + Hidden]"

    build_plot(df, cfg, title=title, out_html=args.out, out_png=args.png)


if __name__ == "__main__":
    main()
