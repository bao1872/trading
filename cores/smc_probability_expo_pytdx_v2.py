# smc_probability_expo_pytdx.py
# -*- coding: utf-8 -*-
"""
Smart Money Concepts Probability (Expo) — Python复刻版（pytdx + Plotly）

目标：
- 复刻 TradingView 脚本：Zeiierman / "Smart Money Concepts Probability (Expo)" 的核心逻辑与可视化
- 数据源：pytdx（日线/分钟线均可，默认日线）
- 输出：Plotly HTML（K线 + 成交量 + 结构线/标签(CHoCH/SMS/BMS) + Premium/Discount 区域 + 概率标签 + Win/Loss 面板）

用法示例：
    python smc_probability_expo_pytdx_v2.py --symbol 002099 --years 3 --prd 20 --resp 7 --show-pd --out expo.html

说明：
- 本脚本按 Pine 逻辑逐 bar 回放，尽量一一对应（包括 Up/Dn 的更新、pos 状态机、概率矩阵更新等）
- TradingView pivothigh/pivotlow 对“等值峰谷”的处理更复杂；这里使用“等于窗口最大/最小”的近似（与常见复刻一致）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datasource.pytdx_client import connect_pytdx, PERIOD_MAP
from zoneinfo import ZoneInfo

from pytdx.hq import TdxHq_API

SH_TZ = ZoneInfo("Asia/Shanghai")


def fmt_percent(x: float, digits: int = 2) -> str:
    if np.isnan(x):
        return "na"
    return f"{x:.{digits}f}%"


def to_market_and_code(symbol_6: str) -> Tuple[int, str]:
    s = symbol_6.strip()
    if len(s) != 6 or not s.isdigit():
        raise ValueError("symbol 必须是 6 位数字，例如 601398 / 000001")
    market = 1 if s.startswith("6") else 0
    return market, s


def default_stock_servers() -> List[Tuple[str, int]]:
    base = [
        ("119.147.212.81", 7709),
        ("119.147.212.82", 7709),
        ("180.153.18.170", 7709),
        ("218.80.248.229", 7709),
        ("61.152.107.141", 7709),
        ("shtdx.gtjas.com", 7709),
        ("sztdx.gtjas.com", 7709),
    ]
    # best_ip（若当前 pytdx 版本支持）
    try:
        with TdxHq_API() as api:
            if hasattr(api, "best_ip"):
                best = api.best_ip()
                if best and "ip" in best and "port" in best:
                    pair = (best["ip"], int(best["port"]))
                    if pair not in base:
                        base.insert(0, pair)
    except Exception:
        pass
    return base


def connect_any_stock_server(timeout: int = 2) -> Tuple[TdxHq_API, Tuple[str, int]]:
    servers = default_stock_servers()
    api = TdxHq_API()
    last_err = None
    for host, port in servers:
        try:
            ok = api.connect(host, port, time_out=timeout)
            if ok:
                return api, (host, port)
        except Exception as e:
            last_err = e
            continue
    try:
        api.disconnect()
    except Exception:
        pass
    raise RuntimeError(f"pytdx 连接失败：{last_err}")


def records_to_df(records: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif {"year", "month", "day", "hour", "minute"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]].astype(int))
    else:
        raise RuntimeError(f"无法解析 datetime，列={list(df.columns)}")

    df = df.sort_values("datetime").set_index("datetime")

    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc == "open":
            colmap[c] = "open"
        elif lc == "high":
            colmap[c] = "high"
        elif lc == "low":
            colmap[c] = "low"
        elif lc == "close":
            colmap[c] = "close"
        elif lc in ["vol", "volume"]:
            colmap[c] = "volume"
    df = df.rename(columns=colmap)

    need = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"缺少必要列 {missing}，现有列={list(df.columns)}")

    out = df[need].copy()
    out.index = out.index.tz_localize(SH_TZ, nonexistent="shift_forward", ambiguous="NaT")
    out = out[~out.index.isna()]
    return out


def fetch_security_bars(
    api: TdxHq_API,
    category: int,
    market: int,
    code: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    page_size: int = 800,
    max_pages: int = 2000,
) -> pd.DataFrame:
    frames = []
    start = 0
    pages = 0
    while pages < max_pages:
        recs = api.get_security_bars(category, market, code, start, page_size)
        if not recs:
            break
        df = records_to_df(recs)
        frames.append(df)

        pages += 1
        start += page_size

        if not df.empty and df.index.min() <= start_dt.tz_convert(SH_TZ):
            break
        if len(recs) < page_size:
            break

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    all_df = pd.concat(frames).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep="last")]
    all_df = all_df[(all_df.index >= start_dt.tz_convert(SH_TZ)) & (all_df.index <= end_dt.tz_convert(SH_TZ))]
    return all_df


def pivots_tv_style(high: np.ndarray, low: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    近似 TradingView ta.pivothigh/ta.pivotlow 的“确认型”枢轴点：
    - 在 bar i（当前）确认 piv = i-length 为枢轴中心，需要左右各 length 根K线
    - 为了更贴近 TV：要求枢轴值在窗口内“唯一”（等值峰谷往往不算枢轴）
    注意：TV 对等值峰谷的细节仍可能不同，但该处理通常能显著靠近。
    """
    n = len(high)
    pivotH = np.full(n, np.nan, dtype=float)
    pivotL = np.full(n, np.nan, dtype=float)

    if length <= 0 or n < 2 * length + 1:
        return pivotH, pivotL

    for i in range(2 * length, n):
        piv = i - length
        w0 = piv - length
        w1 = piv + length
        hh = high[w0:w1 + 1]
        ll = low[w0:w1 + 1]
        if len(hh) != 2 * length + 1:
            continue

        pv_h = high[piv]
        pv_l = low[piv]

        # 唯一最高/最低
        if pv_h == np.max(hh) and np.sum(hh == pv_h) == 1:
            pivotH[i] = pv_h
        if pv_l == np.min(ll) and np.sum(ll == pv_l) == 1:
            pivotL[i] = pv_l

    return pivotH, pivotL


@dataclass
class StructEvent:
    kind: str        # "CHoCH" / "SMS" / "BMS"
    is_bull: bool
    y: float
    x0: int
    x1: int
    x_label: int


def run_probability_expo(
    h: np.ndarray,
    l: np.ndarray,
    prd: int = 20,
    resp_on: bool = True,
    resp: int = 7,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, List[StructEvent]]:
    """
    逐 bar 回放，复刻 Pine 的结构状态机 + 概率矩阵更新。

    关键差异修正（更贴近 TV）：
    1) Up/Dn 初始为 na：在首次 pivot 确认前，Up/Dn 保持 na（与 Pine 的 math.max(na, x)=na 一致）
    2) pivothigh/pivotlow：使用“唯一最高/最低”的近似，避免等值峰谷造成结构偏移
    """
    n = len(h)
    if len(l) != n:
        raise ValueError("high/low length mismatch")

    # pivots（在 bar i 确认 piv=i-prd 的枢轴值）
    pvtHi, pvtLo = pivots_tv_style(h, l, prd)

    # Pine vars
    Up = float("nan")
    Dn = float("nan")
    iUp = -1
    iDn = -1

    Up_series = np.full(n, np.nan, dtype=float)
    Dn_series = np.full(n, np.nan, dtype=float)

    # vals: 9x4（与 Pine 一致的矩阵形状）
    vals = np.zeros((9, 4), dtype=float)
    txt = ["", ""]
    events: List[StructEvent] = []

    def resp_len() -> int:
        return resp if resp_on else prd

    def fmt_percent(v: float) -> str:
        if np.isnan(v):
            return "na"
        return f"{v:.2f}%"

    def current(v: int) -> Tuple[str, float, float]:
        if v >= 0:
            if v == 1:
                return "SMS: ", vals[0, 1], vals[0, 3]
            if v == 2:
                return "BMS: ", vals[1, 1], vals[1, 3]
            if v > 2:
                return "BMS: ", vals[2, 1], vals[2, 3]
        else:
            if v == -1:
                return "SMS: ", vals[3, 1], vals[3, 3]
            if v == -2:
                return "BMS: ", vals[4, 1], vals[4, 3]
            if v < -2:
                return "BMS: ", vals[5, 1], vals[5, 3]
        return "", float("nan"), float("nan")

    pos = 0

    for i in range(n):
        # Pivots + Up/Dn（注意 na 语义）
        if not np.isnan(Up):
            Up = max(Up, float(h[i]))
        if not np.isnan(Dn):
            Dn = min(Dn, float(l[i]))

        if not np.isnan(pvtHi[i]):
            Up = float(pvtHi[i])
        if not np.isnan(pvtLo[i]):
            Dn = float(pvtLo[i])

        Up_series[i] = Up
        Dn_series[i] = Dn

        if i == 0:
            continue

        Up_prev = Up_series[i - 1]
        Dn_prev = Dn_series[i - 1]

        prev_pos = pos
        rl = resp_len()

        # Structure (Bull)
        if (not np.isnan(Up)) and (not np.isnan(Up_prev)) and Up > Up_prev:
            iUp_prev = iUp
            iUp = i
            center = int(round((iUp_prev + i) / 2)) if iUp_prev >= 0 else i

            if pos <= 0:
                if iUp_prev >= 0:
                    events.append(StructEvent("CHoCH", True, Up_prev, iUp_prev, i, center))
                pos = 1
                vals[6, 0] += 1
            elif pos == 1 and i - rl >= 0 and Up_prev == Up_series[i - rl]:
                if iUp_prev >= 0:
                    events.append(StructEvent("SMS", True, Up_prev, iUp_prev, i, center))
                pos = 2
                vals[6, 1] += 1
            elif pos > 1 and i - rl >= 0 and Up_prev == Up_series[i - rl]:
                if iUp_prev >= 0:
                    events.append(StructEvent("BMS", True, Up_prev, iUp_prev, i, center))
                pos = pos + 1
                vals[6, 2] += 1

        elif (not np.isnan(Up)) and (not np.isnan(Up_prev)) and Up < Up_prev:
            iUp = i - prd

        # Structure (Bear)
        if (not np.isnan(Dn)) and (not np.isnan(Dn_prev)) and Dn < Dn_prev:
            iDn_prev = iDn
            iDn = i
            center = int(round((iDn_prev + i) / 2)) if iDn_prev >= 0 else i

            if pos >= 0:
                if iDn_prev >= 0:
                    events.append(StructEvent("CHoCH", False, Dn_prev, iDn_prev, i, center))
                pos = -1
                vals[7, 0] += 1
            elif pos == -1 and i - rl >= 0 and Dn_prev == Dn_series[i - rl]:
                if iDn_prev >= 0:
                    events.append(StructEvent("SMS", False, Dn_prev, iDn_prev, i, center))
                pos = -2
                vals[7, 1] += 1
            elif pos < -1 and i - rl >= 0 and Dn_prev == Dn_series[i - rl]:
                if iDn_prev >= 0:
                    events.append(StructEvent("BMS", False, Dn_prev, iDn_prev, i, center))
                pos = pos - 1
                vals[7, 2] += 1

        elif (not np.isnan(Dn)) and (not np.isnan(Dn_prev)) and Dn > Dn_prev:
            iDn = i - prd

        # Probability Calculation（等价 ta.change(pos)）
        if pos != prev_pos:
            # Results
            if (pos > 0 and prev_pos > 0) or (pos < 0 and prev_pos < 0):
                if vals[8, 0] < vals[8, 1]:
                    vals[8, 2] += 1
                else:
                    vals[8, 3] += 1
            else:
                if vals[8, 0] > vals[8, 1]:
                    vals[8, 2] += 1
                else:
                    vals[8, 3] += 1

            # Totals（避免 0 除）
            tbuC, tbuS, tbuB = vals[6, 0], vals[6, 1], vals[6, 2]
            tbeC, tbeS, tbeB = vals[7, 0], vals[7, 1], vals[7, 2]

            # Bull
            if (prev_pos == 1 or prev_pos == 0) and pos < 0 and tbuC > 0:
                vals[0, 0] += 1
                vals[0, 1] = round((vals[0, 0] / tbuC) * 100, 2)
            if (prev_pos == 1 or prev_pos == 0) and pos == 2 and tbuC > 0:
                vals[0, 2] += 1
                vals[0, 3] = round((vals[0, 2] / tbuC) * 100, 2)

            if prev_pos == 2 and pos < 0 and tbuS > 0:
                vals[1, 0] += 1
                vals[1, 1] = round((vals[1, 0] / tbuS) * 100, 2)
            if prev_pos == 2 and pos > 2 and tbuS > 0:
                vals[1, 2] += 1
                vals[1, 3] = round((vals[1, 2] / tbuS) * 100, 2)

            if prev_pos > 2 and pos < 0 and tbuB > 0:
                vals[2, 0] += 1
                vals[2, 1] = round((vals[2, 0] / tbuB) * 100, 2)
            if prev_pos > 2 and pos > prev_pos and tbuB > 0:
                vals[2, 2] += 1
                vals[2, 3] = round((vals[2, 2] / tbuB) * 100, 2)

            # Bear
            if (prev_pos == -1 or prev_pos == 0) and pos > 0 and tbeC > 0:
                vals[3, 0] += 1
                vals[3, 1] = round((vals[3, 0] / tbeC) * 100, 2)
            if (prev_pos == -1 or prev_pos == 0) and pos == -2 and tbeC > 0:
                vals[3, 2] += 1
                vals[3, 3] = round((vals[3, 2] / tbeC) * 100, 2)

            if prev_pos == -2 and pos > 0 and tbeS > 0:
                vals[4, 0] += 1
                vals[4, 1] = round((vals[4, 0] / tbeS) * 100, 2)
            if prev_pos == -2 and pos < -2 and tbeS > 0:
                vals[4, 2] += 1
                vals[4, 3] = round((vals[4, 2] / tbeS) * 100, 2)

            if prev_pos < -2 and pos > 0 and tbeB > 0:
                vals[5, 0] += 1
                vals[5, 1] = round((vals[5, 0] / tbeB) * 100, 2)
            if prev_pos < -2 and pos < prev_pos and tbeB > 0:
                vals[5, 2] += 1
                vals[5, 3] = round((vals[5, 2] / tbeB) * 100, 2)

            # Current -> txt + cache
            s, val1, val2 = current(pos)
            txt[0] = "CHoCH: " + s + fmt_percent(val1)
            txt[1] = s + fmt_percent(val2)
            vals[8, 0] = val1
            vals[8, 1] = val2

    return Up_series, Dn_series, pos, vals, events


def plot_expo(
    df: pd.DataFrame,
    Up: np.ndarray,
    Dn: np.ndarray,
    pos: int,
    vals: np.ndarray,
    events: List[StructEvent],
    show_pd: bool,
    hlloc: str,
    out_html: str,
) -> None:
    n = len(df)
    x = np.arange(n)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.78, 0.22]
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            increasing_line_color="rgba(38,166,154,1)",
            decreasing_line_color="rgba(239,83,80,1)",
            increasing_fillcolor="rgba(38,166,154,1)",
            decreasing_fillcolor="rgba(239,83,80,1)",
            showlegend=False,
        ),
        row=1, col=1
    )

    up = (df["close"] >= df["open"]).to_numpy()
    vol_colors = np.where(up, "rgba(38,166,154,0.55)", "rgba(239,83,80,0.55)")
    fig.add_trace(
        go.Bar(x=x, y=df["volume"], marker_color=vol_colors, showlegend=False),
        row=2, col=1
    )

    bull_col = "rgba(8,236,126,1)"
    bear_col = "rgba(255,34,34,1)"
    prem_col = "rgba(255,34,34,0.20)"
    disc_col = "rgba(8,236,126,0.20)"

    shapes = []
    annotations = []

    for ev in events:
        col = bull_col if ev.is_bull else bear_col
        shapes.append(dict(
            type="line", xref="x", yref="y",
            x0=ev.x0, x1=ev.x1, y0=ev.y, y1=ev.y,
            line=dict(color=col, width=2),
            layer="above",
        ))
        annotations.append(dict(
            x=ev.x_label, y=ev.y, xref="x", yref="y",
            text=ev.kind,
            showarrow=False,
            font=dict(size=12, color=col),
            bgcolor="rgba(0,0,0,0)",
            xanchor="center",
            yanchor="bottom" if ev.is_bull else "top",
        ))

    if show_pd and n >= 2:
        Up_last = float(Up[-1])
        Dn_last = float(Dn[-1])
        if np.isfinite(Up_last) and np.isfinite(Dn_last) and Up_last > Dn_last:
            PremiumTop = Up_last - (Up_last - Dn_last) * 0.1
            PremiumBot = Up_last - (Up_last - Dn_last) * 0.25
            DiscountTop = Dn_last + (Up_last - Dn_last) * 0.25
            DiscountBot = Dn_last + (Up_last - Dn_last) * 0.1
            MidTop = Up_last - (Up_last - Dn_last) * 0.45
            MidBot = Dn_last + (Up_last - Dn_last) * 0.45

            if events:
                loc_guess = min(ev.x0 for ev in events) if hlloc == "Left" else max(ev.x0 for ev in events)
            else:
                loc_guess = max(0, n - 200)

            shapes.append(dict(
                type="line", xref="x", yref="y",
                x0=loc_guess, x1=n - 1, y0=Up_last, y1=Up_last,
                line=dict(color=bear_col, width=1),
                layer="above",
            ))
            shapes.append(dict(
                type="line", xref="x", yref="y",
                x0=loc_guess, x1=n - 1, y0=Dn_last, y1=Dn_last,
                line=dict(color=bull_col, width=1),
                layer="above",
            ))

            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=loc_guess, x1=n - 1, y0=PremiumBot, y1=PremiumTop,
                fillcolor=prem_col, line=dict(color="rgba(0,0,0,0)", width=0),
                layer="below",
            ))
            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=loc_guess, x1=n - 1, y0=MidBot, y1=MidTop,
                fillcolor="rgba(128,128,128,0.25)", line=dict(color="rgba(0,0,0,0)", width=0),
                layer="below",
            ))
            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=loc_guess, x1=n - 1, y0=DiscountBot, y1=DiscountTop,
                fillcolor=disc_col, line=dict(color="rgba(0,0,0,0)", width=0),
                layer="below",
            ))

            # Pine 的 str1/str2
            # str1 = pos<0?txt[0]:txt[1] ; str2 = pos>0?txt[0]:txt[1]
            # 这里用“当前矩阵里对应 pos 的概率”重建两行文本
            choch_line = "CHoCH: " + fmt_percent(vals[8, 0])
            if pos >= 0:
                prefix = "SMS: " if pos == 1 else "BMS: "
                val2 = vals[0, 3] if pos == 1 else (vals[1, 3] if pos == 2 else vals[2, 3])
            else:
                prefix = "SMS: " if pos == -1 else "BMS: "
                val2 = vals[3, 3] if pos == -1 else (vals[4, 3] if pos == -2 else vals[5, 3])
            bms_line = prefix + fmt_percent(val2)

            str1 = choch_line if pos < 0 else bms_line
            str2 = choch_line if pos > 0 else bms_line

            annotations.append(dict(
                x=n - 1, y=Up_last, xref="x", yref="y",
                text=str1, showarrow=False,
                font=dict(size=12, color="#e6edf3"),
                bgcolor="rgba(0,0,0,0.55)",
                bordercolor="rgba(255,255,255,0.18)",
                borderwidth=1,
                xanchor="left", yanchor="middle",
            ))
            annotations.append(dict(
                x=n - 1, y=Dn_last, xref="x", yref="y",
                text=str2, showarrow=False,
                font=dict(size=12, color="#e6edf3"),
                bgcolor="rgba(0,0,0,0.55)",
                bordercolor="rgba(255,255,255,0.18)",
                borderwidth=1,
                xanchor="left", yanchor="middle",
            ))

    # Win/Loss 面板
    W = float(vals[8, 2])
    Ls = float(vals[8, 3])
    WR = (W / (W + Ls) * 100) if (W + Ls) > 0 else 0.0
    panel_txt = f"WIN: {int(W)}<br>LOSS: {int(Ls)}<br>Profitability: {WR:.2f}%"
    annotations.append(dict(
        x=0.99, y=0.99, xref="paper", yref="paper",
        text=panel_txt, showarrow=False, align="left",
        font=dict(size=12, color="#e6edf3"),
        bgcolor="rgba(0,0,0,0.55)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        xanchor="right", yanchor="top",
    ))

    dates = df.index.tz_convert(SH_TZ).to_pydatetime()
    tick_step = max(1, len(dates) // 10)
    tickvals = list(range(0, len(dates), tick_step))
    ticktext = [dates[i].strftime("%Y-%m-%d") for i in tickvals]

    fig.update_layout(
        title="Smart Money Concepts Probability (Expo) — pytdx + Plotly",
        shapes=shapes,
        annotations=annotations,
        plot_bgcolor="#0b0f14",
        paper_bgcolor="#0b0f14",
        font=dict(color="#c9d1d9"),
        height=900,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        xaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    )
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, row=2, col=1)

    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] HTML saved: {out_html}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default="002099")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--prd", type=int, default=20)
    ap.add_argument("--resp", type=int, default=7)
    ap.add_argument("--no-response", action="store_true")
    ap.add_argument("--show-pd", action="store_true")
    ap.add_argument("--hlloc", type=str, default="Right", choices=["Left", "Right"])
    ap.add_argument("--out", type=str, default="smc_probability_expo.html")
    ap.add_argument("--timeout", type=int, default=2)
    ap.add_argument("--category", type=int, default=4, help="pytdx category: 4=日线, 3=1h, 2=30m, 1=15m, 0=5m")
    args = ap.parse_args()

    end_dt = pd.Timestamp.now(tz=SH_TZ)
    start_dt = end_dt - pd.DateOffset(years=args.years)

    market, code = to_market_and_code(args.symbol)

    api, server = connect_any_stock_server(timeout=args.timeout)
    try:
        print(f"[pytdx] connected: {server[0]}:{server[1]} market={market} code={code}")
        df = fetch_security_bars(
            api=api, category=args.category, market=market, code=code,
            start_dt=start_dt, end_dt=end_dt,
            page_size=800, max_pages=200
        )
        if df.empty:
            raise RuntimeError("K线数据为空：检查 symbol / category / 服务器")
    finally:
        try:
            api.disconnect()
        except Exception:
            pass

    Up, Dn, pos, vals, events = run_probability_expo(
        df["high"].to_numpy(float),
        df["low"].to_numpy(float),
        args.prd,
        (not args.no_response),
        args.resp,
    )

    plot_expo(
        df=df,
        Up=Up, Dn=Dn,
        pos=pos,
        vals=vals,
        events=events,
        show_pd=args.show_pd,
        hlloc=args.hlloc,
        out_html=args.out,
    )


if __name__ == "__main__":
    main()
