# -*- coding: utf-8 -*-
"""
Stop Loss Clustering (Breakouts) + Native Factors (pytdx/plotly)

Purpose:
    参考 TradingView Pine 指标《Stop Loss Clustering (Breakouts) [Kioseff Trading]》与
    用户上传的 bbmacd_viewer.py 这种“独立可运行、内置 pytdx 拉数”的脚本架构，将核心逻辑转成 Python，
    并输出可直接用于回测/选股/因子研究的原生因子表。

Important note:
    原 Pine 脚本大量依赖 request.security_lower_tf、动态绘图对象(box/line/polyline/table)、
    以及 bar 内部 lower-TF 成交量分配。Python 版这里优先做“可研究、可落地、可导出 CSV 的因子化重构”，
    保留两条主线：

    1) Absorbtion Extremes（基于 ATR-ZigZag 摆点 + 同向成交量累积）
    2) Volatility-At-Entry（基于波动投影的止损价位密度层）

    因为离线回测环境通常没有 Pine 那样的 bar 内部 tick/1秒/1分钟数组，本脚本默认使用当前周期 OHLCV
    构建“native factors”。如果未来你有更细级别数据，可在 _get_signed_volume_series() 与
    _build_time_scaled_layers() 两处继续替换为 lower-TF 数据实现更高保真版本。

Outputs:
    - HTML: 主图 + 活跃/已触发 clusters + 因子面板
    - CSV : 每根K线的原生因子

Examples:
    python stop_loss_clustering_with_factors_pytdx_plotly.py --symbol 000001 --freq d --bars 1000
    python stop_loss_clustering_with_factors_pytdx_plotly.py --symbol 600519 --freq w --model volatility_at_entry --csv-out slc_600519_w.csv
    python stop_loss_clustering_with_factors_pytdx_plotly.py --symbol 000001 --freq 60m --bars 500 --show-historical-triggers
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
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

BUY_COL = "#55ffda"
SELL_COL = "#ff65fb"
BUY_WEAK = "#6929F2"
SELL_WEAK = "#ffb0fc"
ZERO_COL = "#888888"
REMOVED_ALPHA = 0.35


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


def get_stock_name(symbol: str) -> str:
    api = connect_pytdx()
    try:
        market = _market_from_symbol(symbol)
        df = pd.DataFrame(api.get_security_list(market, 0))
        if not df.empty and "code" in df.columns and "name" in df.columns:
            row = df[df["code"].astype(str) == str(symbol)]
            if not row.empty:
                return str(row.iloc[0]["name"])
        # fallback: page through lists
        start = 0
        while True:
            recs = api.get_security_list(market, start)
            if not recs:
                break
            d = pd.DataFrame(recs)
            if "code" in d.columns and "name" in d.columns:
                row = d[d["code"].astype(str) == str(symbol)]
                if not row.empty:
                    return str(row.iloc[0]["name"])
            if len(recs) < 1000:
                break
            start += 1000
    except Exception:
        pass
    finally:
        try:
            api.disconnect()
        except Exception:
            pass
    return str(symbol)


def freq_minutes(freq: str) -> int:
    mapping = {"5m": 5, "15m": 15, "30m": 30, "60m": 60, "d": 240, "w": 1200, "mo": 4800}
    return mapping.get(freq, 240)


def time_to_str(ts: pd.Timestamp, is_intraday: bool) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if is_intraday else ts.strftime("%Y-%m-%d")


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
    if length <= 0 or len(arr) == 0 or len(arr) < length:
        return pd.Series(out, index=src.index)
    init = np.nanmean(arr[:length])
    out[length - 1] = init
    prev = init
    alpha = 1.0 / length
    for i in range(length, len(arr)):
        prev = alpha * arr[i] + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=src.index)


def atr_pine(df: pd.DataFrame, length: int) -> pd.Series:
    return pine_rma(true_range(df), length)


def rolling_percentile(arr: np.ndarray, q: float, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        sample = arr[lo : i + 1]
        sample = sample[np.isfinite(sample)]
        if len(sample) > 0:
            out[i] = float(np.percentile(sample, q))
    return out


@dataclass
class SwingCluster:
    volume: float
    start_idx: int
    price: float
    barrier: float
    pivot_idx: int
    triggered_idx: Optional[int] = None
    intra_bar_move: float = np.nan


@dataclass
class TimeLayer:
    price: float
    volume: float
    first_idx: int
    last_idx: int
    side: str  # "buy" or "sell"
    triggered_idx: Optional[int] = None


class IQZZ:
    """ATR ZigZag approximation of the Pine IQZZ() helper."""

    def __init__(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray, atr_mult: float = 2.0):
        self.high = high
        self.low = low
        self.close = close
        self.atr = atr * atr_mult

    def run(self) -> Tuple[np.ndarray, List[float], List[int], List[str]]:
        n = len(self.close)
        direction = 0
        pivot_price = float(self.close[0]) if n else np.nan
        pivot_idx = 0
        prev_confirmed_price = np.nan
        prev_confirmed_idx = -1

        market_state = np.zeros(n, dtype=float)
        point_arr: List[float] = []
        time_arr: List[int] = []
        point_kind: List[str] = []

        for i in range(n):
            atr_i = self.atr[i]
            if not np.isfinite(atr_i):
                market_state[i] = direction
                continue

            if direction == 1:
                price = max(pivot_price, self.high[i])
                if price == self.high[i]:
                    pivot_price = self.high[i]
                    pivot_idx = i
                if self.low[i] <= pivot_price - atr_i and self.high[i] != pivot_price:
                    if np.isfinite(prev_confirmed_price) and prev_confirmed_price != pivot_price:
                        point_arr.append(pivot_price)
                        time_arr.append(pivot_idx)
                        point_kind.append("high")
                    prev_confirmed_price = pivot_price
                    prev_confirmed_idx = pivot_idx
                    direction = -1
                    pivot_price = self.low[i]
                    pivot_idx = i

            elif direction == -1:
                price = min(self.low[i], pivot_price)
                if price == self.low[i]:
                    pivot_price = self.low[i]
                    pivot_idx = i
                if self.high[i] >= pivot_price + atr_i and self.low[i] != pivot_price:
                    if np.isfinite(prev_confirmed_price) and prev_confirmed_price != pivot_price:
                        point_arr.append(pivot_price)
                        time_arr.append(pivot_idx)
                        point_kind.append("low")
                    prev_confirmed_price = pivot_price
                    prev_confirmed_idx = pivot_idx
                    direction = 1
                    pivot_price = self.high[i]
                    pivot_idx = i

            else:
                if self.high[i] >= pivot_price + atr_i:
                    direction = 1
                    pivot_price = self.high[i]
                    pivot_idx = i
                elif self.low[i] <= pivot_price - atr_i:
                    direction = -1
                    pivot_price = self.low[i]
                    pivot_idx = i

            market_state[i] = direction

        return market_state, point_arr, time_arr, point_kind


class StopLossClusteringEngine:
    def __init__(self, df: pd.DataFrame, args: argparse.Namespace):
        self.df = df.copy()
        self.args = args
        self.freq = args.freq
        self.is_intraday = self.freq in {"5m", "15m", "30m", "60m"}
        self.x_numeric = np.arange(len(self.df), dtype=float)
        self.tick_text = [time_to_str(t, self.is_intraday) for t in self.df.index]
        self.mintick = self._infer_mintick(self.df["close"].to_numpy(dtype=float))
        self.df["atr_14"] = atr_pine(self.df, 14)
        self.df["hlc3"] = (self.df["high"] + self.df["low"] + self.df["close"]) / 3.0
        self.df["signed_volume"] = self._get_signed_volume_series()

        self.buy_clusters_active: List[SwingCluster] = []
        self.sell_clusters_active: List[SwingCluster] = []
        self.buy_clusters_removed: List[SwingCluster] = []
        self.sell_clusters_removed: List[SwingCluster] = []
        self.buy_layers_active: List[TimeLayer] = []
        self.sell_layers_active: List[TimeLayer] = []
        self.buy_layers_removed: List[TimeLayer] = []
        self.sell_layers_removed: List[TimeLayer] = []

    @staticmethod
    def _infer_mintick(close: np.ndarray) -> float:
        diffs = np.abs(np.diff(np.unique(np.round(close[np.isfinite(close)], 6))))
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 0.01
        return float(np.min(diffs))

    def _get_signed_volume_series(self) -> pd.Series:
        close = self.df["close"].to_numpy(dtype=float)
        vol = self.df["vol"].to_numpy(dtype=float)
        prev = np.roll(close, 1)
        prev[0] = close[0]
        sign = np.sign(close - prev)
        return pd.Series(vol * sign, index=self.df.index)

    def run(self) -> None:
        if self.args.model == "absorbtion_extremes":
            self._run_absorbtion_extremes()
        else:
            self._run_volatility_at_entry()

    # ------------------------ Absorbtion Extremes ------------------------
    def _run_absorbtion_extremes(self) -> None:
        high = self.df["high"].to_numpy(dtype=float)
        low = self.df["low"].to_numpy(dtype=float)
        close = self.df["close"].to_numpy(dtype=float)
        atr = self.df["atr_14"].to_numpy(dtype=float)
        signed_vol = self.df["signed_volume"].to_numpy(dtype=float)
        n = len(self.df)

        market_state, point_arr, point_idx_arr, point_kind_arr = IQZZ(high, low, close, atr, atr_mult=2.0).run()
        self.df["market_state"] = market_state

        buy_side_fill = np.where(signed_vol > 0, signed_vol, 0.0)
        sell_side_fill = np.where(signed_vol < 0, -signed_vol, 0.0)

        # Output arrays / native factors
        buy_created = np.zeros(n, dtype=float)
        sell_created = np.zeros(n, dtype=float)
        buy_trigger = np.zeros(n, dtype=float)
        sell_trigger = np.zeros(n, dtype=float)
        buy_trigger_vol = np.zeros(n, dtype=float)
        sell_trigger_vol = np.zeros(n, dtype=float)
        active_buy_count = np.zeros(n, dtype=float)
        active_sell_count = np.zeros(n, dtype=float)
        removed_buy_count = np.zeros(n, dtype=float)
        removed_sell_count = np.zeros(n, dtype=float)
        sum_buys_active = np.zeros(n, dtype=float)
        sum_sells_active = np.zeros(n, dtype=float)
        sum_buys_removed = np.zeros(n, dtype=float)
        sum_sells_removed = np.zeros(n, dtype=float)
        nearest_buy_price = np.full(n, np.nan, dtype=float)
        nearest_sell_price = np.full(n, np.nan, dtype=float)
        nearest_buy_vol = np.full(n, np.nan, dtype=float)
        nearest_sell_vol = np.full(n, np.nan, dtype=float)
        nearest_buy_typical = np.full(n, np.nan, dtype=float)
        nearest_sell_typical = np.full(n, np.nan, dtype=float)
        buy_stop_pressure = np.full(n, np.nan, dtype=float)
        sell_stop_pressure = np.full(n, np.nan, dtype=float)
        stop_cluster_ratio = np.full(n, np.nan, dtype=float)
        dist_to_nearest_buy_atr = np.full(n, np.nan, dtype=float)
        dist_to_nearest_sell_atr = np.full(n, np.nan, dtype=float)
        cluster_buy_dominance = np.full(n, np.nan, dtype=float)
        cluster_sell_dominance = np.full(n, np.nan, dtype=float)
        # 新增：记录当天被突破的sell cluster中volume最大的那个的price
        sell_trigger_max_vol_price = np.full(n, np.nan, dtype=float)

        # Map confirmed pivot index -> meta for creation events
        point_meta = [(idx, px, kind) for idx, px, kind in zip(point_idx_arr, point_arr, point_kind_arr)]
        point_meta.sort(key=lambda x: x[0])
        next_point_ptr = 1  # mimic Pine getPointSize > 1 requirement

        typical_buy_moves: List[float] = []
        typical_sell_moves: List[float] = []
        point_indices_only = [p[0] for p in point_meta]

        for i in range(n):
            atr_i = atr[i]
            # create new cluster if confirmed pivot appears on this bar
            while next_point_ptr < len(point_meta) and point_meta[next_point_ptr][0] == i:
                idx, px, kind = point_meta[next_point_ptr]
                prev_idx, prev_px, prev_kind = point_meta[next_point_ptr - 1]
                bars_diff = max(1, idx - prev_idx)
                if kind == "high" and px > prev_px:
                    vol_now = float(np.nansum(sell_side_fill[max(0, idx - bars_diff + 1): idx + 1]))
                    price = high[idx] + self.mintick
                    barrier = price + (atr_i / 4.0 if np.isfinite(atr_i) else 0.0)
                    self.sell_clusters_active.insert(0, SwingCluster(vol_now, idx, price, barrier, idx))
                    sell_created[i] = 1.0
                elif kind == "low" and px < prev_px:
                    vol_now = float(np.nansum(buy_side_fill[max(0, idx - bars_diff + 1): idx + 1]))
                    price = low[idx] - self.mintick
                    barrier = price - (atr_i / 4.0 if np.isfinite(atr_i) else 0.0)
                    self.buy_clusters_active.insert(0, SwingCluster(vol_now, idx, price, barrier, idx))
                    buy_created[i] = 1.0
                next_point_ptr += 1

            # trigger detection
            triggered_sells: List[SwingCluster] = []
            max_vol_sell_price = np.nan
            max_vol_sell_volume = -1.0
            for cl in list(self.sell_clusters_active):
                if high[i] >= cl.barrier:
                    cl.triggered_idx = i
                    base = min(cl.price, cl.barrier)
                    cl.intra_bar_move = abs(high[i] / base - 1.0) if base > 0 else np.nan
                    triggered_sells.append(cl)
                    self.sell_clusters_active.remove(cl)
                    self.sell_clusters_removed.insert(0, cl)
                    sell_trigger[i] = 1.0
                    sell_trigger_vol[i] += cl.volume
                    # 记录volume最大的sell cluster的price
                    if cl.volume > max_vol_sell_volume:
                        max_vol_sell_volume = cl.volume
                        max_vol_sell_price = cl.price
                    if np.isfinite(cl.intra_bar_move):
                        typical_sell_moves.append(cl.intra_bar_move)
            if sell_trigger[i] == 1.0 and np.isfinite(max_vol_sell_price):
                sell_trigger_max_vol_price[i] = max_vol_sell_price

            triggered_buys: List[SwingCluster] = []
            for cl in list(self.buy_clusters_active):
                if low[i] <= cl.barrier:
                    cl.triggered_idx = i
                    base = max(cl.price, cl.barrier)
                    cl.intra_bar_move = abs(low[i] / base - 1.0) if base > 0 else np.nan
                    triggered_buys.append(cl)
                    self.buy_clusters_active.remove(cl)
                    self.buy_clusters_removed.insert(0, cl)
                    buy_trigger[i] = 1.0
                    buy_trigger_vol[i] += cl.volume
                    if np.isfinite(cl.intra_bar_move):
                        typical_buy_moves.append(cl.intra_bar_move)

            active_buy_count[i] = len(self.buy_clusters_active)
            active_sell_count[i] = len(self.sell_clusters_active)
            removed_buy_count[i] = len(self.buy_clusters_removed)
            removed_sell_count[i] = len(self.sell_clusters_removed)
            sum_buys_active[i] = float(sum(cl.volume for cl in self.buy_clusters_active))
            sum_sells_active[i] = float(sum(cl.volume for cl in self.sell_clusters_active))
            sum_buys_removed[i] = float(sum(cl.volume for cl in self.buy_clusters_removed))
            sum_sells_removed[i] = float(sum(cl.volume for cl in self.sell_clusters_removed))

            if self.buy_clusters_active:
                # nearest buy stop cluster is the closest support-like layer below/near price
                candidates = sorted(self.buy_clusters_active, key=lambda c: abs(close[i] - c.price))
                nb = candidates[0]
                nearest_buy_price[i] = nb.price
                nearest_buy_vol[i] = nb.volume
                if np.isfinite(atr_i) and atr_i != 0:
                    dist_to_nearest_buy_atr[i] = (close[i] - nb.price) / atr_i
                if typical_buy_moves:
                    nearest_buy_typical[i] = float(np.median(typical_buy_moves[-50:]))

            if self.sell_clusters_active:
                candidates = sorted(self.sell_clusters_active, key=lambda c: abs(c.price - close[i]))
                ns = candidates[0]
                nearest_sell_price[i] = ns.price
                nearest_sell_vol[i] = ns.volume
                if np.isfinite(atr_i) and atr_i != 0:
                    dist_to_nearest_sell_atr[i] = (ns.price - close[i]) / atr_i
                if typical_sell_moves:
                    nearest_sell_typical[i] = float(np.median(typical_sell_moves[-50:]))

            total_active = sum_buys_active[i] + sum_sells_active[i]
            total_removed = sum_buys_removed[i] + sum_sells_removed[i]
            if total_active > 0:
                buy_stop_pressure[i] = sum_buys_active[i] / total_active
                sell_stop_pressure[i] = sum_sells_active[i] / total_active
                stop_cluster_ratio[i] = sum_sells_active[i] / max(sum_buys_active[i], 1e-12)
                cluster_buy_dominance[i] = (sum_buys_active[i] - sum_sells_active[i]) / total_active
                cluster_sell_dominance[i] = (sum_sells_active[i] - sum_buys_active[i]) / total_active

        self.df["buy_cluster_created"] = buy_created
        self.df["sell_cluster_created"] = sell_created
        self.df["buy_stop_triggered"] = buy_trigger
        self.df["sell_stop_triggered"] = sell_trigger
        self.df["buy_stop_triggered_volume"] = buy_trigger_vol
        self.df["sell_stop_triggered_volume"] = sell_trigger_vol
        self.df["active_buy_cluster_count"] = active_buy_count
        self.df["active_sell_cluster_count"] = active_sell_count
        self.df["removed_buy_cluster_count"] = removed_buy_count
        self.df["removed_sell_cluster_count"] = removed_sell_count
        self.df["sum_buys_active"] = sum_buys_active
        self.df["sum_sells_active"] = sum_sells_active
        self.df["sum_buys_removed"] = sum_buys_removed
        self.df["sum_sells_removed"] = sum_sells_removed
        self.df["nearest_buy_stop_price"] = nearest_buy_price
        self.df["nearest_sell_stop_price"] = nearest_sell_price
        self.df["nearest_buy_stop_volume"] = nearest_buy_vol
        self.df["nearest_sell_stop_volume"] = nearest_sell_vol
        self.df["nearest_buy_typical_move"] = nearest_buy_typical
        self.df["nearest_sell_typical_move"] = nearest_sell_typical
        self.df["buy_stop_pressure"] = buy_stop_pressure
        self.df["sell_stop_pressure"] = sell_stop_pressure
        self.df["stop_cluster_ratio"] = stop_cluster_ratio
        self.df["dist_to_nearest_buy_stop_atr"] = dist_to_nearest_buy_atr
        self.df["dist_to_nearest_sell_stop_atr"] = dist_to_nearest_sell_atr
        self.df["cluster_buy_dominance"] = cluster_buy_dominance
        self.df["cluster_sell_dominance"] = cluster_sell_dominance
        self.df["sell_trigger_max_vol_price"] = sell_trigger_max_vol_price

        # Off-chart pulse equivalents from Pine
        self.df["buy_stops_pulse"] = -self.df["buy_stop_triggered_volume"]
        self.df["sell_stops_pulse"] = self.df["sell_stop_triggered_volume"]
        buy_thres = rolling_percentile(np.abs(self.df["buy_stops_pulse"].to_numpy(dtype=float)), 25, 50)
        sell_thres = rolling_percentile(np.abs(self.df["sell_stops_pulse"].to_numpy(dtype=float)), 75, 50)
        self.df["buy_stop_pulse_threshold"] = buy_thres
        self.df["sell_stop_pulse_threshold"] = sell_thres
        self.df["buy_stop_pulse_radiate"] = (np.abs(self.df["buy_stops_pulse"]) >= self.df["buy_stop_pulse_threshold"]).astype(float)
        self.df["sell_stop_pulse_radiate"] = (np.abs(self.df["sell_stops_pulse"]) >= self.df["sell_stop_pulse_threshold"]).astype(float)
        # Keep the same avg-line columns that the figure layer expects, even in absorbtion_extremes mode.
        # In Pine this mode uses rolling percentile thresholds; here we additionally expose 50-bar mean pulse lines
        # so the pane renderer has a stable set of columns across both models.
        self.df["buy_stops_avg_50"] = pd.Series(np.abs(self.df["buy_stops_pulse"].to_numpy(dtype=float)), index=self.df.index).rolling(50, min_periods=1).mean().to_numpy(dtype=float)
        self.df["sell_stops_avg_50"] = pd.Series(np.abs(self.df["sell_stops_pulse"].to_numpy(dtype=float)), index=self.df.index).rolling(50, min_periods=1).mean().to_numpy(dtype=float)

    # ------------------------ Volatility-At-Entry ------------------------
    def _run_volatility_at_entry(self) -> None:
        high = self.df["high"].to_numpy(dtype=float)
        low = self.df["low"].to_numpy(dtype=float)
        close = self.df["close"].to_numpy(dtype=float)
        atr = self.df["atr_14"].to_numpy(dtype=float)
        signed_vol = self.df["signed_volume"].to_numpy(dtype=float)
        hlc3 = self.df["hlc3"].to_numpy(dtype=float)
        n = len(self.df)

        tfm = freq_minutes(self.freq)
        base_minutes = 1.0
        factors = []
        for root_tf in [1, 5, 15, 30, 60, 240]:
            root = math.sqrt(root_tf / base_minutes)
            for m in [1.0, 1.5, 2.0]:
                factors.append(root * m)

        buy_trigger = np.zeros(n, dtype=float)
        sell_trigger = np.zeros(n, dtype=float)
        buy_trigger_vol = np.zeros(n, dtype=float)
        sell_trigger_vol = np.zeros(n, dtype=float)
        active_buy_count = np.zeros(n, dtype=float)
        active_sell_count = np.zeros(n, dtype=float)
        removed_buy_count = np.zeros(n, dtype=float)
        removed_sell_count = np.zeros(n, dtype=float)
        sum_buys_active = np.zeros(n, dtype=float)
        sum_sells_active = np.zeros(n, dtype=float)
        sum_buys_removed = np.zeros(n, dtype=float)
        sum_sells_removed = np.zeros(n, dtype=float)
        nearest_buy_price = np.full(n, np.nan, dtype=float)
        nearest_sell_price = np.full(n, np.nan, dtype=float)
        nearest_buy_vol = np.full(n, np.nan, dtype=float)
        nearest_sell_vol = np.full(n, np.nan, dtype=float)
        nearest_buy_share = np.full(n, np.nan, dtype=float)
        nearest_sell_share = np.full(n, np.nan, dtype=float)
        dist_to_nearest_buy_atr = np.full(n, np.nan, dtype=float)
        dist_to_nearest_sell_atr = np.full(n, np.nan, dtype=float)
        buy_pressure = np.full(n, np.nan, dtype=float)
        sell_pressure = np.full(n, np.nan, dtype=float)
        ratio = np.full(n, np.nan, dtype=float)

        for i in range(n):
            atr_i = atr[i]
            if not np.isfinite(atr_i) or atr_i <= 0:
                continue

            direction = np.sign(signed_vol[i])
            if direction != 0:
                unit_vol = abs(signed_vol[i]) / len(factors)
                for f in factors:
                    level = hlc3[i] + atr_i * f * direction
                    layer = TimeLayer(
                        price=float(level),
                        volume=float(unit_vol),
                        first_idx=i,
                        last_idx=i,
                        side="sell" if direction > 0 else "buy",
                    )
                    if direction > 0:
                        self.sell_layers_active.append(layer)
                    else:
                        self.buy_layers_active.append(layer)

            # trigger existing levels crossed by current bar range
            for layer in list(self.sell_layers_active):
                if high[i] >= layer.price:
                    layer.triggered_idx = i
                    self.sell_layers_active.remove(layer)
                    self.sell_layers_removed.append(layer)
                    sell_trigger[i] = 1.0
                    sell_trigger_vol[i] += layer.volume
                else:
                    layer.last_idx = i

            for layer in list(self.buy_layers_active):
                if low[i] <= layer.price:
                    layer.triggered_idx = i
                    self.buy_layers_active.remove(layer)
                    self.buy_layers_removed.append(layer)
                    buy_trigger[i] = 1.0
                    buy_trigger_vol[i] += layer.volume
                else:
                    layer.last_idx = i

            active_buy_count[i] = len(self.buy_layers_active)
            active_sell_count[i] = len(self.sell_layers_active)
            removed_buy_count[i] = len(self.buy_layers_removed)
            removed_sell_count[i] = len(self.sell_layers_removed)
            sum_buys_active[i] = float(sum(x.volume for x in self.buy_layers_active))
            sum_sells_active[i] = float(sum(x.volume for x in self.sell_layers_active))
            sum_buys_removed[i] = float(sum(x.volume for x in self.buy_layers_removed))
            sum_sells_removed[i] = float(sum(x.volume for x in self.sell_layers_removed))

            if self.buy_layers_active:
                nb = min(self.buy_layers_active, key=lambda x: abs(close[i] - x.price))
                nearest_buy_price[i] = nb.price
                nearest_buy_vol[i] = nb.volume
                dist_to_nearest_buy_atr[i] = (close[i] - nb.price) / atr_i
            if self.sell_layers_active:
                ns = min(self.sell_layers_active, key=lambda x: abs(x.price - close[i]))
                nearest_sell_price[i] = ns.price
                nearest_sell_vol[i] = ns.volume
                dist_to_nearest_sell_atr[i] = (ns.price - close[i]) / atr_i

            total_active = sum_buys_active[i] + sum_sells_active[i]
            if total_active > 0:
                nearest_buy_share[i] = nearest_buy_vol[i] / total_active if np.isfinite(nearest_buy_vol[i]) else np.nan
                nearest_sell_share[i] = nearest_sell_vol[i] / total_active if np.isfinite(nearest_sell_vol[i]) else np.nan
                buy_pressure[i] = sum_buys_active[i] / total_active
                sell_pressure[i] = sum_sells_active[i] / total_active
                ratio[i] = sum_sells_active[i] / max(sum_buys_active[i], 1e-12)

        self.df["buy_stop_triggered"] = buy_trigger
        self.df["sell_stop_triggered"] = sell_trigger
        self.df["buy_stop_triggered_volume"] = buy_trigger_vol
        self.df["sell_stop_triggered_volume"] = sell_trigger_vol
        self.df["active_buy_cluster_count"] = active_buy_count
        self.df["active_sell_cluster_count"] = active_sell_count
        self.df["removed_buy_cluster_count"] = removed_buy_count
        self.df["removed_sell_cluster_count"] = removed_sell_count
        self.df["sum_buys_active"] = sum_buys_active
        self.df["sum_sells_active"] = sum_sells_active
        self.df["sum_buys_removed"] = sum_buys_removed
        self.df["sum_sells_removed"] = sum_sells_removed
        self.df["nearest_buy_stop_price"] = nearest_buy_price
        self.df["nearest_sell_stop_price"] = nearest_sell_price
        self.df["nearest_buy_stop_volume"] = nearest_buy_vol
        self.df["nearest_sell_stop_volume"] = nearest_sell_vol
        self.df["nearest_buy_stop_share"] = nearest_buy_share
        self.df["nearest_sell_stop_share"] = nearest_sell_share
        self.df["dist_to_nearest_buy_stop_atr"] = dist_to_nearest_buy_atr
        self.df["dist_to_nearest_sell_stop_atr"] = dist_to_nearest_sell_atr
        self.df["buy_stop_pressure"] = buy_pressure
        self.df["sell_stop_pressure"] = sell_pressure
        self.df["stop_cluster_ratio"] = ratio
        self.df["buy_stops_pulse"] = -buy_trigger_vol
        self.df["sell_stops_pulse"] = sell_trigger_vol
        self.df["buy_stops_avg_50"] = pd.Series(self.df["buy_stops_pulse"].abs()).rolling(50, min_periods=1).mean().to_numpy(dtype=float)
        self.df["sell_stops_avg_50"] = pd.Series(self.df["sell_stops_pulse"].abs()).rolling(50, min_periods=1).mean().to_numpy(dtype=float)
        self.df["buy_stop_pulse_radiate"] = (self.df["buy_stops_pulse"].abs() >= self.df["buy_stops_avg_50"]).astype(float)
        self.df["sell_stop_pulse_radiate"] = (self.df["sell_stops_pulse"].abs() >= self.df["sell_stops_avg_50"]).astype(float)

    # ------------------------ Plotting ------------------------
    def build_figure(self, title: str) -> go.Figure:
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            row_heights=[0.58, 0.14, 0.14, 0.14],
            subplot_titles=(title, "Cluster Pressure", "Stop Trigger Pulses", "Nearest Cluster Distance / Ratio"),
        )

        fig.add_trace(
            go.Candlestick(
                x=self.x_numeric,
                open=self.df["open"],
                high=self.df["high"],
                low=self.df["low"],
                close=self.df["close"],
                increasing_line_color="#00c2a0",
                decreasing_line_color="#ff4d5a",
                increasing_fillcolor="#00c2a0",
                decreasing_fillcolor="#ff4d5a",
                showlegend=False,
                name="K线",
            ),
            row=1,
            col=1,
        )

        self._add_cluster_overlays(fig)

        # Pressure panel
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["buy_stop_pressure"], mode="lines", line=dict(color=BUY_COL, width=1.6), name="buy_stop_pressure", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["sell_stop_pressure"], mode="lines", line=dict(color=SELL_COL, width=1.6), name="sell_stop_pressure", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["sum_buys_active"], mode="lines", line=dict(color=BUY_WEAK, width=1.1), name="sum_buys_active", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["sum_sells_active"], mode="lines", line=dict(color=SELL_WEAK, width=1.1), name="sum_sells_active", showlegend=False), row=2, col=1)

        # Pulse panel - closer to Pine layered glow
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["buy_stops_avg_50"], mode="lines", line=dict(color="rgba(85,255,218,0.20)", width=10), showlegend=False, hoverinfo="skip"), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["buy_stops_avg_50"], mode="lines", line=dict(color="rgba(85,255,218,0.30)", width=6), showlegend=False, hoverinfo="skip"), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["buy_stops_avg_50"], mode="lines", line=dict(color=BUY_COL, width=2), name="buyStopsAvg", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["sell_stops_avg_50"], mode="lines", line=dict(color="rgba(255,101,251,0.20)", width=10), showlegend=False, hoverinfo="skip"), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["sell_stops_avg_50"], mode="lines", line=dict(color="rgba(255,101,251,0.30)", width=6), showlegend=False, hoverinfo="skip"), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["sell_stops_avg_50"], mode="lines", line=dict(color=SELL_COL, width=2), name="sellStopsAvg", showlegend=False), row=3, col=1)

        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["buy_stops_pulse"], mode="markers", marker=dict(color=BUY_COL, size=6), name="buy_stops_pulse", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["sell_stops_pulse"], mode="markers", marker=dict(color=SELL_COL, size=6), name="sell_stops_pulse", showlegend=False), row=3, col=1)

        buy_rad = self.df["buy_stop_pulse_radiate"] > 0
        sell_rad = self.df["sell_stop_pulse_radiate"] > 0
        for size, alpha in [(10, 0.06), (7, 0.10), (5, 0.15), (3, 0.22)]:
            fig.add_trace(go.Scatter(x=self.x_numeric[buy_rad.to_numpy()], y=self.df.loc[buy_rad, "buy_stops_pulse"], mode="markers", marker=dict(color=f"rgba(85,255,218,{alpha})", size=size), showlegend=False, hoverinfo="skip"), row=3, col=1)
            fig.add_trace(go.Scatter(x=self.x_numeric[sell_rad.to_numpy()], y=self.df.loc[sell_rad, "sell_stops_pulse"], mode="markers", marker=dict(color=f"rgba(255,101,251,{alpha})", size=size), showlegend=False, hoverinfo="skip"), row=3, col=1)

        # Distance / ratio panel
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["dist_to_nearest_buy_stop_atr"], mode="lines", line=dict(color=BUY_COL, width=1.3), name="dist_to_nearest_buy_stop_atr", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["dist_to_nearest_sell_stop_atr"], mode="lines", line=dict(color=SELL_COL, width=1.3), name="dist_to_nearest_sell_stop_atr", showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=self.x_numeric, y=self.df["stop_cluster_ratio"], mode="lines", line=dict(color="#f5d742", width=1.2), name="stop_cluster_ratio", showlegend=False), row=4, col=1)

        for row in (2, 3, 4):
            fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=ZERO_COL, row=row, col=1)

        n = len(self.df)
        tick_step = max(1, n // 10)
        tickvals = list(range(0, n, tick_step))
        if tickvals[-1] != n - 1:
            tickvals.append(n - 1)
        ticktext = [self.tick_text[i] for i in tickvals]

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=40, r=30, t=90, b=70),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.01),
            annotations=list(fig.layout.annotations),
        )
        self._add_dashboard_annotations(fig)
        for row in (1, 2, 3, 4):
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=True, zeroline=False, row=row, col=1)
            fig.update_yaxes(showgrid=True, zeroline=False, row=row, col=1)
        return fig

    def _add_cluster_overlays(self, fig: go.Figure) -> None:
        active_buy = self.buy_clusters_active if self.args.model == "absorbtion_extremes" else self.buy_layers_active
        active_sell = self.sell_clusters_active if self.args.model == "absorbtion_extremes" else self.sell_layers_active
        removed_buy = self.buy_clusters_removed if self.args.model == "absorbtion_extremes" else self.buy_layers_removed
        removed_sell = self.sell_clusters_removed if self.args.model == "absorbtion_extremes" else self.sell_layers_removed

        def add_lines(objs: Iterable, color: str, name: str, historical: bool = False) -> None:
            shown = False
            limit = self.args.max_lines
            count = 0
            for obj in objs:
                if count >= limit:
                    break
                start = obj.start_idx if hasattr(obj, "start_idx") else obj.first_idx
                end = obj.triggered_idx if historical and getattr(obj, "triggered_idx", None) is not None else len(self.df) - 1
                if end is None:
                    end = len(self.df) - 1
                widths = [25, 20, 10, 4, 2] if not historical else [5, 1]
                alphas = [0.02, 0.03, 0.05, 0.12, 0.95] if not historical else [0.10, 0.45]
                dashes = ["solid"] * len(widths) if not historical else ["solid", "dot"]
                for w, a, d in zip(widths, alphas, dashes):
                    fig.add_trace(
                        go.Scatter(
                            x=[start, end],
                            y=[obj.price, obj.price],
                            mode="lines",
                            line=dict(color=color if a >= 0.9 else f"rgba({85 if color==BUY_COL else 255},{255 if color==BUY_COL else 101},{218 if color==BUY_COL else 251},{a})", width=w, dash=d),
                            opacity=1.0,
                            name=name,
                            showlegend=(not shown and a >= 0.9),
                            hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                    shown = shown or a >= 0.9
                count += 1

        add_lines(active_buy, BUY_COL, "active_buy_cluster", historical=False)
        add_lines(active_sell, SELL_COL, "active_sell_cluster", historical=False)
        if self.args.show_historical_triggers:
            add_lines(removed_buy, BUY_COL, "removed_buy_cluster", historical=True)
            add_lines(removed_sell, SELL_COL, "removed_sell_cluster", historical=True)

    def _add_dashboard_annotations(self, fig: go.Figure) -> None:
        last = self.df.iloc[-1]
        buy_price = last.get("nearest_buy_stop_price", np.nan)
        sell_price = last.get("nearest_sell_stop_price", np.nan)
        buy_vol = last.get("nearest_buy_stop_volume", np.nan)
        sell_vol = last.get("nearest_sell_stop_volume", np.nan)
        if self.args.model == "volatility_at_entry":
            buy_head = "% Of All Buy-Stop Clusters"
            sell_head = "% Of All Sell-Stop Clusters"
            buy_extra = last.get("nearest_buy_stop_share", np.nan)
            sell_extra = last.get("nearest_sell_stop_share", np.nan)
        else:
            buy_head = "Typical Move"
            sell_head = "Typical Move"
            buy_extra = last.get("nearest_buy_typical_move", np.nan)
            sell_extra = last.get("nearest_sell_typical_move", np.nan)

        def fmt(v, pct=False):
            if pd.isna(v):
                return "None"
            return f"{v:.2%}" if pct else (f"{v:.4f}" if abs(v) < 10000 else f"{v:,.0f}")

        pct = self.args.model == "volatility_at_entry"
        text = (
            "<b>Stop-Loss Clustering</b><br>"
            f"<span style='color:{BUY_COL}'>Nearest Buy-Stop Cluster</span><br>"
            f"Price: {fmt(buy_price)}<br>Cluster: {fmt(buy_vol)}<br>{buy_head}: {fmt(buy_extra, pct=pct)}<br><br>"
            f"<span style='color:{SELL_COL}'>Nearest Sell-Stop Cluster</span><br>"
            f"Price: {fmt(sell_price)}<br>Cluster: {fmt(sell_vol)}<br>{sell_head}: {fmt(sell_extra, pct=pct)}"
        )
        fig.add_annotation(xref="paper", yref="paper", x=0.995, y=0.995, xanchor="right", yanchor="top",
                           text=text, showarrow=False, align="left",
                           bordercolor="#363843", borderwidth=1, borderpad=8,
                           bgcolor="#20222C", font=dict(color="white", size=11))

        # ratio meter
        buy_active = abs(float(last.get("sum_buys_active", 0.0) or 0.0))
        sell_active = abs(float(last.get("sum_sells_active", 0.0) or 0.0))
        total = buy_active + sell_active
        if total > 0:
            buy_ratio = buy_active / total
            sell_ratio = sell_active / total
            blocks = 20
            meter = "".join([f"<span style='color:{SELL_COL if i < int(round(sell_ratio*blocks)) else BUY_COL}'>█</span>" for i in range(blocks)])
            fig.add_annotation(xref="paper", yref="paper", x=0.5, y=-0.12, xanchor="center", yanchor="top",
                               text=f"<b>Cluster Ratio Meter</b><br>{meter}<br>Buy {buy_ratio:.1%} | Sell {sell_ratio:.1%}",
                               showarrow=False, align="center", bgcolor="#181b27",
                               bordercolor="#363843", borderwidth=1, borderpad=6, font=dict(color="white", size=11))


def fetch_data(symbol: str, freq: str, bars: int) -> pd.DataFrame:
    freq = normalize_freq(freq)
    df = fetch_kline_pytdx(symbol, freq, bars).copy()
    if "volume" in df.columns and "vol" not in df.columns:
        df = df.rename(columns={"volume": "vol"})
    for c in ["vol", "amount"]:
        if c not in df.columns:
            df[c] = 0.0
    return df[["open", "high", "low", "close", "vol", "amount"]].astype(float)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stop Loss Clustering closer Pine-style Python port + factors (pytdx/plotly)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--freq", default="d")
    p.add_argument("--bars", type=int, default=1000)
    p.add_argument("--model", default="absorbtion_extremes", choices=["absorbtion_extremes", "volatility_at_entry"])
    p.add_argument("--out", default="stop_loss_clustering.html")
    p.add_argument("--csv-out", default="")
    p.add_argument("--show-historical-triggers", action="store_true", default=True)
    p.add_argument("--hide-historical-triggers", action="store_false", dest="show_historical_triggers")
    p.add_argument("--max-lines", type=int, default=20)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.freq = normalize_freq(args.freq)

    df = fetch_data(args.symbol, args.freq, args.bars)
    name = get_stock_name(args.symbol)

    engine = StopLossClusteringEngine(df, args)
    engine.run()
    fig = engine.build_figure(f"Stop Loss Clustering - {name}({args.symbol}) [{args.freq}] / {args.model}")
    plot(fig, filename=args.out, auto_open=False, include_plotlyjs=True)

    if args.csv_out:
        engine.df.to_csv(args.csv_out, encoding="utf-8-sig")
        print(f"CSV 已生成: {args.csv_out}")
    print(f"HTML 已生成: {args.out}")


if __name__ == "__main__":
    main()
