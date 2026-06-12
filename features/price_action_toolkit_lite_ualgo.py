# -*- coding: utf-8 -*-
"""
price_action_toolkit_lite_ualgo.py

TradingView Pine 指标 Price Action Toolkit Lite [UAlgo] 的 Python/Plotly 复刻版。

核心输出：
1) 市场结构线：ZigZag swing line + CHoCH / BoS 水平突破线。
2) 流动性线与 sweep 标记：默认 liquidity_len=20。
3) Order Block：突破结构后生成 bullish/bearish OB，默认只显示最近 2 个未失效区间。
4) Trend Line：默认 trend_line_len=20，按最近两个 pivot high/low 画下降/上升趋势线。
5) 支持 pytdx 拉取 A 股 K 线，也支持本地 CSV。

依赖：
    pip install pandas numpy plotly pytdx

示例：
    python price_action_toolkit_lite_ualgo.py --symbol 300133 --freq w --bars 300 --fetch-bars 1200
    python price_action_toolkit_lite_ualgo.py --symbol 300133 --freq d --zigzag-len 9 --liquidity-len 20 --trend-line-len 20
    python price_action_toolkit_lite_ualgo.py --csv your_kline.csv --out-html out.html --out-csv out.csv

CSV 需要包含：datetime/open/high/low/close/volume；amount 可选。

说明：
- 本脚本尽量按 Pine 原逻辑复刻。由于 TradingView 的对象生命周期和 Python 离线渲染方式不同，
  HTML 会展示历史结构对象，而不是只保留图上最后一个对象。
- Pine 原码许可证为 CC BY-NC-SA 4.0，作者 © UAlgo。本 Python 转换稿应遵循同样的非商业共享协议。
"""
from __future__ import annotations

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
# 数据读取：保留你上传框架里的 pytdx / CSV 风格
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
# 指标计算
# =============================================================================
def _atr_wilder(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _tv_pivots_confirmed(high: np.ndarray, low: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TradingView ta.pivothigh/ta.pivotlow 风格，确认 bar 记录值，anchor 记录真实 pivot 位置。"""
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    ph_anchor = np.full(n, np.nan)
    pl_anchor = np.full(n, np.nan)
    win = 2 * length + 1
    for i in range(win - 1, n):
        c = i - length
        lo = c - length
        hi = c + length + 1
        hwin = high[lo:hi]
        lwin = low[lo:hi]
        if np.all(np.isfinite(hwin)) and np.isfinite(high[c]) and high[c] == np.max(hwin):
            ph[i] = high[c]
            ph_anchor[i] = c
        if np.all(np.isfinite(lwin)) and np.isfinite(low[c]) and low[c] == np.min(lwin):
            pl[i] = low[c]
            pl_anchor[i] = c
    return ph, pl, ph_anchor, pl_anchor


@dataclass
class PATConfig:
    zigzag_len: int = 9
    liquidity_len: int = 20
    trend_line_len: int = 20
    number_ob_show: int = 2
    show_market_structure: bool = True
    show_liquidity: bool = True
    show_order_blocks: bool = True
    show_trend_lines: bool = True


def compute_price_action_toolkit(df: pd.DataFrame, cfg: PATConfig) -> Tuple[pd.DataFrame, Dict[str, list]]:
    out = df.copy().sort_index()
    for col in ["open", "high", "low", "close"]:
        if col not in out.columns:
            raise ValueError(f"df 缺少字段: {col}")
    if "volume" not in out.columns:
        out["volume"] = np.nan
    if "amount" not in out.columns:
        out["amount"] = np.nan

    o = out["open"].to_numpy(float)
    h = out["high"].to_numpy(float)
    l = out["low"].to_numpy(float)
    c = out["close"].to_numpy(float)
    n = len(out)
    atr = _atr_wilder(out, 14).to_numpy(float)
    zlen = int(cfg.zigzag_len)

    # Pine: to_up := high[zigzagLen] >= ta.highest(high, zigzagLen)
    #       to_down := low[zigzagLen] <= ta.lowest(low, zigzagLen)
    to_up = np.zeros(n, dtype=bool)
    to_down = np.zeros(n, dtype=bool)
    for i in range(zlen, n):
        cand = i - zlen
        # ta.highest(high, zlen) 当前窗口包含 i-zlen+1..i，不包含 high[i-zlen]。
        # zlen=1 时退化为比较当前 bar。
        lo_idx = max(0, i - zlen + 1)
        if zlen <= 1:
            hi_window = h[i:i + 1]
            lo_window = l[i:i + 1]
        else:
            hi_window = h[lo_idx:i + 1]
            lo_window = l[lo_idx:i + 1]
        to_up[i] = np.isfinite(h[cand]) and h[cand] >= np.nanmax(hi_window)
        to_down[i] = np.isfinite(l[cand]) and l[cand] <= np.nanmin(lo_window)

    high_vals: List[Tuple[int, float]] = []
    low_vals: List[Tuple[int, float]] = []
    zigzag_lines: List[Dict] = []
    structure_lines: List[Dict] = []
    bullish_ob: List[Dict] = []
    bearish_ob: List[Dict] = []

    trend = 1
    last_state: Optional[str] = None
    draw_up = False
    draw_down = False

    trend_state = np.full(n, np.nan)
    active_last_high = np.full(n, np.nan)
    active_last_low = np.full(n, np.nan)
    evt_bos_up = np.zeros(n, dtype=bool)
    evt_choch_up = np.zeros(n, dtype=bool)
    evt_bos_down = np.zeros(n, dtype=bool)
    evt_choch_down = np.zeros(n, dtype=bool)
    evt_break_up = np.zeros(n, dtype=bool)
    evt_break_down = np.zeros(n, dtype=bool)

    for i in range(n):
        prev_trend = trend
        if i >= zlen:
            if trend == 1 and to_down[i]:
                trend = -1
            elif trend == -1 and to_up[i]:
                trend = 1

        if trend != prev_trend and i >= zlen and trend == 1:
            anchor = i - zlen
            high_vals.append((anchor, float(h[anchor])))
            if len(low_vals) > 1:
                x1, y1 = low_vals[-1]
                x2, y2 = high_vals[-1]
                zigzag_lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "direction": "up"})
                draw_up = False
        elif trend != prev_trend and i >= zlen and trend == -1:
            anchor = i - zlen
            low_vals.append((anchor, float(l[anchor])))
            if len(high_vals) > 1:
                x1, y1 = high_vals[-1]
                x2, y2 = low_vals[-1]
                zigzag_lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "direction": "down"})
                draw_down = False

        # 结构跌破：close < 最近 swing low，只触发一次，直到下一次反向 swing 重置 draw_down。
        if len(low_vals) > 1 and not draw_down:
            low_idx, low_val = low_vals[-1]
            if c[i] < low_val:
                label = "CHoCH" if (last_state is None or last_state == "up") else "BoS"
                structure_lines.append({"x1": low_idx, "x2": i, "y": low_val, "side": "down", "label": label})
                if label == "CHoCH":
                    evt_choch_down[i] = True
                else:
                    evt_bos_down[i] = True
                evt_break_down[i] = True
                draw_down = True
                last_state = "down"
                if cfg.show_order_blocks:
                    start = min(max(low_idx + 1, 0), i)
                    end = i + 1
                    if start < end:
                        rel = int(np.nanargmax(h[start:end]))
                        ob_idx = start + rel
                        val = float(h[ob_idx])
                        a = float(atr[i]) if np.isfinite(atr[i]) else float(np.nanmean(h - l))
                        bearish_ob.append({
                            "start": ob_idx, "end": i, "value": val, "top": val, "bottom": val - a,
                            "created": i, "broken": False,
                        })
                        if len(bearish_ob) > 20:
                            bearish_ob.pop(0)

        # 结构突破：close > 最近 swing high，只触发一次，直到下一次反向 swing 重置 draw_up。
        if len(high_vals) > 1 and not draw_up:
            high_idx, high_val = high_vals[-1]
            if c[i] > high_val:
                label = "CHoCH" if (last_state is None or last_state == "down") else "BoS"
                structure_lines.append({"x1": high_idx, "x2": i, "y": high_val, "side": "up", "label": label})
                if label == "CHoCH":
                    evt_choch_up[i] = True
                else:
                    evt_bos_up[i] = True
                evt_break_up[i] = True
                draw_up = True
                last_state = "up"
                if cfg.show_order_blocks:
                    start = min(max(high_idx + 1, 0), i)
                    end = i + 1
                    if start < end:
                        rel = int(np.nanargmin(l[start:end]))
                        ob_idx = start + rel
                        val = float(l[ob_idx])
                        a = float(atr[i]) if np.isfinite(atr[i]) else float(np.nanmean(h - l))
                        bullish_ob.append({
                            "start": ob_idx, "end": i, "value": val, "top": val + a, "bottom": val,
                            "created": i, "broken": False,
                        })
                        if len(bullish_ob) > 20:
                            bullish_ob.pop(0)

        # OB 失效更新：Pine 是每根 bar 对最近 numberObShow 个 active OB 延伸，收盘破 value 删掉。
        if bullish_ob:
            active = [ob for ob in bullish_ob if not ob.get("broken", False)]
            for ob in active[-int(cfg.number_ob_show):]:
                ob["end"] = i
                if c[i] < ob["value"]:
                    ob["broken"] = True
                    ob["broken_at"] = i
        if bearish_ob:
            active = [ob for ob in bearish_ob if not ob.get("broken", False)]
            for ob in active[-int(cfg.number_ob_show):]:
                ob["end"] = i
                if c[i] > ob["value"]:
                    ob["broken"] = True
                    ob["broken_at"] = i

        trend_state[i] = trend
        if high_vals:
            active_last_high[i] = high_vals[-1][1]
        if low_vals:
            active_last_low[i] = low_vals[-1][1]

    # Liquidity pivots and sweep lines.
    llen = int(cfg.liquidity_len)
    ph_liq, pl_liq, ph_liq_anchor, pl_liq_anchor = _tv_pivots_confirmed(h, l, llen)
    bearish_liq_active: List[Dict] = []  # line above price, breaks by high > value
    bullish_liq_active: List[Dict] = []  # line below price, breaks by low < value
    liquidity_lines: List[Dict] = []
    evt_bearish_liquidity_sweep = np.zeros(n, dtype=bool)  # sweep upper, close back below: bearish
    evt_bullish_liquidity_sweep = np.zeros(n, dtype=bool)  # sweep lower, close back above: bullish
    evt_upper_sweep_fail_up = np.zeros(n, dtype=bool)  # 上扫收回后反而向上突破/转强
    evt_lower_sweep_fail_down = np.zeros(n, dtype=bool)  # 下扫收回后反而向下破位/转弱
    active_bear_liq = np.full(n, np.nan)
    active_bull_liq = np.full(n, np.nan)
    pending_upper_sweep_fails = []  # [(sweep_bar_idx, line_value, remaining_bars)]
    pending_lower_sweep_fails = []  # [(sweep_bar_idx, line_value, remaining_bars)]
    sweep_fail_window = 3  # sweep 后多少根 bar 内检测 fail

    for i in range(n):
        if cfg.show_liquidity and np.isfinite(ph_liq[i]):
            a = int(ph_liq_anchor[i])
            line = {"start": a, "end": i, "value": float(h[a]), "side": "upper", "broken": False, "sweep": False, "sweep_at": None}
            bearish_liq_active.append(line)
            liquidity_lines.append(line)
            if len(bearish_liq_active) > 7:
                old = bearish_liq_active.pop(0)
                # TradingView 会删除最旧对象；离线图保留历史，但限制 end 至当时。
                old["hidden_by_limit"] = True

        if cfg.show_liquidity and np.isfinite(pl_liq[i]):
            a = int(pl_liq_anchor[i])
            line = {"start": a, "end": i, "value": float(l[a]), "side": "lower", "broken": False, "sweep": False, "sweep_at": None}
            bullish_liq_active.append(line)
            liquidity_lines.append(line)
            if len(bullish_liq_active) > 7:
                old = bullish_liq_active.pop(0)
                old["hidden_by_limit"] = True

        for line in list(reversed(bearish_liq_active)):
            if line.get("broken"):
                continue
            if h[i] > line["value"]:
                line["end"] = i
                line["broken"] = True
                if c[i] < line["value"]:
                    line["sweep"] = True
                    line["sweep_at"] = i
                    evt_bearish_liquidity_sweep[i] = True
                    pending_upper_sweep_fails.append((i, line["value"], sweep_fail_window))
                bearish_liq_active.remove(line)
            else:
                line["end"] = i

        for line in list(reversed(bullish_liq_active)):
            if line.get("broken"):
                continue
            if l[i] < line["value"]:
                line["end"] = i
                line["broken"] = True
                if c[i] > line["value"]:
                    line["sweep"] = True
                    line["sweep_at"] = i
                    evt_bullish_liquidity_sweep[i] = True
                    pending_lower_sweep_fails.append((i, line["value"], sweep_fail_window))
                bullish_liq_active.remove(line)
            else:
                line["end"] = i

        if bearish_liq_active:
            active_bear_liq[i] = bearish_liq_active[-1]["value"]
        if bullish_liq_active:
            active_bull_liq[i] = bullish_liq_active[-1]["value"]

        # Sweep fail 检测：扫收回后 N 根 bar 内价格再次突破流动性线
        new_upper = []
        for (sweep_idx, line_val, remaining) in pending_upper_sweep_fails:
            if remaining <= 0:
                continue
            if c[i] > line_val:
                evt_upper_sweep_fail_up[i] = True
                continue  # 已触发，不再跟踪
            new_upper.append((sweep_idx, line_val, remaining - 1))
        pending_upper_sweep_fails = new_upper

        new_lower = []
        for (sweep_idx, line_val, remaining) in pending_lower_sweep_fails:
            if remaining <= 0:
                continue
            if c[i] < line_val:
                evt_lower_sweep_fail_down[i] = True
                continue
            new_lower.append((sweep_idx, line_val, remaining - 1))
        pending_lower_sweep_fails = new_lower

    # Trend lines: 最后两个 confirmed pivot high/low，下降高点线、上升低点线。
    tlen = int(cfg.trend_line_len)
    ph_tr, pl_tr, ph_tr_anchor, pl_tr_anchor = _tv_pivots_confirmed(h, l, tlen)
    trend_lines: List[Dict] = []
    ph_seq: List[Tuple[int, float, int]] = []  # anchor, value, confirm
    pl_seq: List[Tuple[int, float, int]] = []
    cur_bear_line_id: Optional[int] = None
    cur_bull_line_id: Optional[int] = None

    for i in range(n):
        if np.isfinite(ph_tr[i]):
            ph_seq.append((int(ph_tr_anchor[i]), float(ph_tr[i]), i))
        if np.isfinite(pl_tr[i]):
            pl_seq.append((int(pl_tr_anchor[i]), float(pl_tr[i]), i))

        if cfg.show_trend_lines and len(ph_seq) >= 2:
            a1, v1, _ = ph_seq[-2]
            a2, v2, _ = ph_seq[-1]
            if a2 != a1:
                slope = (v2 - v1) / (a2 - a1)
                if slope < 0:
                    if cur_bear_line_id is None or trend_lines[cur_bear_line_id]["x1"] != a1 or trend_lines[cur_bear_line_id]["x2_pivot"] != a2:
                        trend_lines.append({"x1": a1, "y1": v1, "x2_pivot": a2, "y2_pivot": v2, "x2": i, "side": "bear", "slope": slope})
                        cur_bear_line_id = len(trend_lines) - 1
                    else:
                        trend_lines[cur_bear_line_id]["x2"] = i

        if cfg.show_trend_lines and len(pl_seq) >= 2:
            a1, v1, _ = pl_seq[-2]
            a2, v2, _ = pl_seq[-1]
            if a2 != a1:
                slope = (v2 - v1) / (a2 - a1)
                if slope > 0:
                    if cur_bull_line_id is None or trend_lines[cur_bull_line_id]["x1"] != a1 or trend_lines[cur_bull_line_id]["x2_pivot"] != a2:
                        trend_lines.append({"x1": a1, "y1": v1, "x2_pivot": a2, "y2_pivot": v2, "x2": i, "side": "bull", "slope": slope})
                        cur_bull_line_id = len(trend_lines) - 1
                    else:
                        trend_lines[cur_bull_line_id]["x2"] = i

    out["pat_trend_state"] = trend_state
    out["pat_last_swing_high"] = active_last_high
    out["pat_last_swing_low"] = active_last_low
    out["evt_pat_choch_up"] = evt_choch_up
    out["evt_pat_bos_up"] = evt_bos_up
    out["evt_pat_choch_down"] = evt_choch_down
    out["evt_pat_bos_down"] = evt_bos_down
    out["evt_pat_structure_break_up"] = evt_break_up
    out["evt_pat_structure_break_down"] = evt_break_down
    out["pat_liq_upper_active"] = active_bear_liq
    out["pat_liq_lower_active"] = active_bull_liq
    out["evt_pat_upper_liquidity_sweep"] = evt_bearish_liquidity_sweep
    out["evt_pat_lower_liquidity_sweep"] = evt_bullish_liquidity_sweep
    out["evt_pat_upper_sweep_fail_up"] = evt_upper_sweep_fail_up
    out["evt_pat_lower_sweep_fail_down"] = evt_lower_sweep_fail_down
    out["pat_atr14"] = atr

    objects = {
        "zigzag_lines": zigzag_lines,
        "structure_lines": structure_lines,
        "liquidity_lines": liquidity_lines,
        "bullish_ob": bullish_ob,
        "bearish_ob": bearish_ob,
        "trend_lines": trend_lines,
    }
    return out, objects


# =============================================================================
# 可视化
# =============================================================================
def _clamp_line(line: Dict, plot_start: int, plot_end_exclusive: int) -> Optional[Tuple[float, float, float, float]]:
    x1 = int(line["x1"])
    x2 = int(line["x2"])
    if x2 < plot_start or x1 >= plot_end_exclusive:
        return None
    cx1 = max(x1, plot_start)
    cx2 = min(x2, plot_end_exclusive - 1)
    if cx2 < cx1:
        return None
    # horizontal line
    if "y" in line:
        return cx1 - plot_start, float(line["y"]), cx2 - plot_start, float(line["y"])
    y1 = float(line["y1"])
    if "y2" in line:
        y2 = float(line["y2"])
        denom = max(1, x2 - x1)
        yy1 = y1 + (y2 - y1) * (cx1 - x1) / denom
        yy2 = y1 + (y2 - y1) * (cx2 - x1) / denom
        return cx1 - plot_start, yy1, cx2 - plot_start, yy2
    return None


def build_html(df_full: pd.DataFrame, objects: Dict[str, list], df_plot: pd.DataFrame, out_html: str, title: str, cfg: PATConfig) -> None:
    if go is None or make_subplots is None:
        raise RuntimeError("未安装 plotly: pip install plotly")

    plot_start = len(df_full) - len(df_plot)
    plot_end = len(df_full)
    x = np.arange(len(df_plot), dtype=float)
    intraday = len(df_plot.index) > 1 and (df_plot.index[1] - df_plot.index[0]) < pd.Timedelta("20H")
    tick_text = [ts.strftime("%Y-%m-%d %H:%M") if intraday else ts.strftime("%Y-%m-%d") for ts in df_plot.index]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.70, 0.16, 0.14],
        subplot_titles=(title, "Volume", "PAT Event Rows"),
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df_plot["open"], high=df_plot["high"], low=df_plot["low"], close=df_plot["close"],
            name="K线", increasing_line_color="#00c853", decreasing_line_color="#ff5252",
            increasing_fillcolor="#00c853", decreasing_fillcolor="#ff5252",
        ),
        row=1, col=1,
    )

    # Market structure swing lines.
    if cfg.show_market_structure:
        for line in objects.get("zigzag_lines", []):
            clipped = _clamp_line({"x1": line["x1"], "y1": line["y1"], "x2": line["x2"], "y2": line["y2"]}, plot_start, plot_end)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode="lines", line=dict(color="#9e9e9e", width=1.2), name="Market Structure", showlegend=False), row=1, col=1)

        for line in objects.get("structure_lines", []):
            clipped = _clamp_line(line, plot_start, plot_end)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            color = "#00bfa5" if line["side"] == "up" else "#ff5252"
            fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode="lines", line=dict(color=color, width=1.4), name=line["label"], showlegend=False), row=1, col=1)
            midx = (x1 + x2) / 2
            fig.add_annotation(x=midx, y=y1, text=line["label"], showarrow=False, font=dict(size=10, color=color), xref="x1", yref="y1")

    # Liquidity lines and sweeps.
    for line in objects.get("liquidity_lines", []):
        clipped = _clamp_line({"x1": line["start"], "x2": line["end"], "y": line["value"]}, plot_start, plot_end)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        color = "#ff5252" if line["side"] == "upper" else "#00bfa5"
        dash = "dash" if line.get("broken") else "solid"
        opacity = 0.28 if line.get("hidden_by_limit") else 0.75
        fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode="lines", line=dict(color=color, width=1, dash=dash), opacity=opacity, name="Liquidity", showlegend=False), row=1, col=1)
        if line.get("sweep") and line.get("sweep_at") is not None:
            si = int(line["sweep_at"])
            if plot_start <= si < plot_end:
                lx = si - plot_start
                ly = df_full["high"].iloc[si] if line["side"] == "upper" else df_full["low"].iloc[si]
                sym = "x"
                fig.add_trace(go.Scatter(x=[lx], y=[ly], mode="markers+text", text=["x"], textposition="middle center", marker=dict(symbol=sym, size=11, color="#7e57c2" if line["side"] == "upper" else "#00bfa5"), name="Sweep", showlegend=False), row=1, col=1)

    # Order blocks: show active/unbroken last N by default, and recently broken if inside plotting window.
    def add_ob(ob: Dict, color: str, name: str) -> None:
        start = int(ob["start"])
        end = int(ob.get("end", start))
        if end < plot_start or start >= plot_end:
            return
        x0 = max(start, plot_start) - plot_start
        x1 = min(end, plot_end - 1) - plot_start
        opacity = 0.18 if not ob.get("broken") else 0.07
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=float(ob["bottom"]), y1=float(ob["top"]),
            xref="x1", yref="y1", line=dict(color=color, width=1), fillcolor=color, opacity=opacity,
        )
        fig.add_trace(go.Scatter(x=[x0, x1], y=[float(ob["value"]), float(ob["value"])], mode="lines", line=dict(color=color, width=0.8, dash="dot"), name=name, showlegend=False), row=1, col=1)

    if cfg.show_order_blocks:
        for ob in [x for x in objects.get("bullish_ob", []) if not x.get("broken")][-int(cfg.number_ob_show):]:
            add_ob(ob, "#00bfa5", "Bullish OB")
        for ob in [x for x in objects.get("bearish_ob", []) if not x.get("broken")][-int(cfg.number_ob_show):]:
            add_ob(ob, "#ff5252", "Bearish OB")

    # Trend lines.
    if cfg.show_trend_lines:
        for tl in objects.get("trend_lines", []):
            x1 = int(tl["x1"])
            x2 = int(tl["x2"])
            if x2 < plot_start or x1 >= plot_end:
                continue
            cx1 = max(x1, plot_start)
            cx2 = min(x2, plot_end - 1)
            y1 = float(tl["y1"] + tl["slope"] * (cx1 - x1))
            y2 = float(tl["y1"] + tl["slope"] * (cx2 - x1))
            color = "#26a69a" if tl["side"] == "bull" else "#ef5350"
            fig.add_trace(go.Scatter(x=[cx1 - plot_start, cx2 - plot_start], y=[y1, y2], mode="lines", line=dict(color=color, width=2), name=f"{tl['side']} trendline", showlegend=False), row=1, col=1)

    # Event scatter helpers.
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
            df_plot["pat_last_swing_high"].to_numpy(float)[mask],
            df_plot["pat_last_swing_low"].to_numpy(float)[mask],
        ])
        fig.add_trace(
            go.Scatter(
                x=x[mask], y=y, mode="markers", marker=dict(symbol=symbol, size=12, color=color, line=dict(color="#ffffff", width=1)),
                name=label, customdata=cd,
                hovertemplate=(
                    f"{label}<br>%{{customdata[0]}}<br>close=%{{customdata[1]:.3f}}"
                    "<br>last_high=%{customdata[2]:.3f}<br>last_low=%{customdata[3]:.3f}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    event_mark("evt_pat_choch_up", "high", "CHoCH Up", "#00e676", "triangle-up", price_pad)
    event_mark("evt_pat_bos_up", "high", "BoS Up", "#00bfa5", "triangle-up", price_pad * 1.7)
    event_mark("evt_pat_choch_down", "low", "CHoCH Down", "#ff5252", "triangle-down", -price_pad)
    event_mark("evt_pat_bos_down", "low", "BoS Down", "#d50000", "triangle-down", -price_pad * 1.7)
    event_mark("evt_pat_upper_liquidity_sweep", "high", "Upper Sweep", "#7e57c2", "x", price_pad)
    event_mark("evt_pat_lower_liquidity_sweep", "low", "Lower Sweep", "#40c4ff", "x", -price_pad)
    event_mark("evt_pat_upper_sweep_fail_up", "high", "Upper Sweep Fail Up", "#ff9800", "triangle-up", price_pad * 2.4)
    event_mark("evt_pat_lower_sweep_fail_down", "low", "Lower Sweep Fail Down", "#e040fb", "triangle-down", -price_pad * 2.4)

    # Volume.
    vol = df_plot["volume"].astype(float)
    bar_colors = np.where(df_plot["close"].to_numpy(float) >= df_plot["open"].to_numpy(float), "#00c853", "#ff5252")
    fig.add_trace(go.Bar(x=x, y=vol, marker_color=bar_colors, name="volume"), row=2, col=1)

    # Event rows.
    rows = [
        ("evt_pat_choch_up", 8, "CHoCH↑", "#00e676"),
        ("evt_pat_bos_up", 7, "BoS↑", "#00bfa5"),
        ("evt_pat_upper_sweep_fail_up", 6, "扫高失败↑", "#ff9800"),
        ("evt_pat_lower_liquidity_sweep", 5, "扫低收回", "#40c4ff"),
        ("evt_pat_upper_liquidity_sweep", 4, "扫高收回", "#7e57c2"),
        ("evt_pat_lower_sweep_fail_down", 3, "扫低失败↓", "#e040fb"),
        ("evt_pat_choch_down", 2, "CHoCH↓", "#ff5252"),
        ("evt_pat_bos_down", 1, "BoS↓", "#d50000"),
    ]
    for col, yv, nm, color in rows:
        if col in df_plot.columns:
            mask = df_plot[col].fillna(False).to_numpy(bool)
            if mask.any():
                fig.add_trace(go.Scatter(x=x[mask], y=np.full(mask.sum(), yv), mode="markers", marker=dict(size=10, color=color), name=nm), row=3, col=1)
    fig.update_yaxes(tickmode="array", tickvals=[1, 2, 3, 4, 5, 6, 7, 8], ticktext=["BoS↓", "CHoCH↓", "扫低失败↓", "扫高", "扫低", "扫高失败↑", "BoS↑", "CHoCH↑"], row=3, col=1)

    # Axis/layout.
    step = max(1, len(df_plot) // 10)
    fig.update_xaxes(tickmode="array", tickvals=list(x[::step]), ticktext=tick_text[::step])
    fig.update_layout(
        template="plotly_dark",
        height=950,
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
    return str(base / f"{tag}_pat_lite_{f}.csv"), str(base / f"{tag}_pat_lite_{f}.html")


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

    cfg = PATConfig(
        zigzag_len=int(args.zigzag_len),
        liquidity_len=int(args.liquidity_len),
        trend_line_len=int(args.trend_line_len),
        number_ob_show=int(args.number_ob_show),
        show_market_structure=not args.no_market_structure,
        show_liquidity=not args.no_liquidity,
        show_order_blocks=not args.no_order_blocks,
        show_trend_lines=not args.no_trend_lines,
    )
    feat, objects = compute_price_action_toolkit(df, cfg)
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
        f"Price Action Toolkit Lite [UAlgo] | {symbol_tag} {freq} | "
        f"zigzag={cfg.zigzag_len}, liquidity={cfg.liquidity_len}, trendline={cfg.trend_line_len}"
    )
    build_html(feat, objects, df_plot, out_html, title, cfg)
    return out_csv, out_html


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Price Action Toolkit Lite [UAlgo] Pine -> Python/Plotly")
    p.add_argument("--symbol", type=str, default=None, help="A股代码，例如 300133；不传则需使用 --csv")
    p.add_argument("--csv", type=str, default=None, help="本地 OHLCV CSV")
    p.add_argument("--freq", type=str, default="d", help="单周期：d/w/mo/60m/30m/15m/5m/1m")
    p.add_argument("--freqs", type=str, default=None, help="多周期，例如 d,w,60m。使用 --csv 时通常只用 --freq")
    p.add_argument("--bars", type=int, default=300, help="HTML 图中展示最近多少根")
    p.add_argument("--fetch-bars", type=int, default=1200, help="pytdx 拉取多少根历史 K 线")
    p.add_argument("--out-dir", type=str, default="./output_pat_lite", help="默认输出目录")
    p.add_argument("--out-csv", type=str, default=None, help="单周期 CSV 输出路径")
    p.add_argument("--out-html", type=str, default=None, help="单周期 HTML 输出路径")

    p.add_argument("--zigzag-len", type=int, default=9, help="Pine ZigZag Length，默认 9")
    p.add_argument("--liquidity-len", type=int, default=20, help="Liquidity Length，按你的要求默认 20")
    p.add_argument("--trend-line-len", type=int, default=20, help="Trend Line Detection Sensitivity，默认 20")
    p.add_argument("--number-ob-show", type=int, default=2, help="显示最近几个未失效 Order Block，默认 2")

    p.add_argument("--no-market-structure", action="store_true", help="不画市场结构线/CHoCH/BoS")
    p.add_argument("--no-liquidity", action="store_true", help="不画流动性线/sweep")
    p.add_argument("--no-order-blocks", action="store_true", help="不画 Order Block")
    p.add_argument("--no-trend-lines", action="store_true", help="不画趋势线")
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
