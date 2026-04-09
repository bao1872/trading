# -*- coding: utf-8 -*-
"""
最终执行版策略脚本：Mixed 主买点家族 + S2/S3 卖点模板

Purpose
- 基于最终收敛后的研究结论，提供两类可直接使用的功能：
  1) 指定日期选股（扫描股票池，输出当日满足最终买点规则的个股清单）
  2) 单股买卖回测（按最终买点规则逐次开仓，按卖点模板回测）

Strategy Summary
================
一、主买点家族：Mixed
核心思想：不是单纯低位，也不是单纯追启动，而是筛选“宽度/压缩有利 + 趋势不过热 + 位置不高”的结构型买点。

二、最终买点规则（执行定义）
Rule A：宽度 + 趋势热度
    - bb_width_norm_high
    - trend_gap_10_20_low40

Rule B：DSA 宽度 + 趋势热度
    - dsa_trade_range_width_pct_high
    - trend_gap_10_20_low40

Rule C：增强版（家族规则）
    - trend_gap_10_20_low40
    - score_trend_total_low40
    - 且满足以下二选一：
        C1: bb_width_norm_high
        C2: dsa_trade_range_width_pct_high

三、事件池约束（先有事件，再看规则）
本脚本不会在所有 bar 上乱扫，而是先进入 C 邻域候选事件池：
    - dsa_trade_pos_01 <= 0.45
    - prev_confirmed_down_bars >= 8
    - rope_dir == 1
    - current_run_bars <= 20
    - bars_since_dir_change <= 12
    - 同一股票 20 bars 冷却去重

四、真实买入定义（仅为时序正确，不是研究对象）
    - signal_date = 事件识别日 t
    - buy_date = t+1
    - buy_price = t+1 开盘价
说明：
    收益、回撤、路径质量等一律从真实买入点开始统计；
    固定窗口收益仅作观察列，不作为买卖依据。

五、卖点模板
默认主模板：S2_early_protect
    - 共同硬止损：跌破 signal_low（事件日最低价）
    - 动态保护：自买入以来最高价回撤 4% 触发卖出

备选模板：S3_profit_extend
    - 共同硬止损：跌破 signal_low
    - 先观察是否达到 +8% 浮盈
    - 达到后，若收盘跌破 rope（或 rope_dir 翻空），触发卖出

六、输出要求
所有选股 / 回测输出都会明确给出：
    - 满足哪条买点规则（rule_name / rule_family）
    - 买点时间、买入价格
    - 卖点模板（exit_template）
    - 卖出原因（exit_reason）
    - 卖出时间、卖出价格

How to Run
==========
1) 指定单日选股：
python c0_strategy.py \
    --mode pick \
    --pick-date 2026-04-08 \
    --n-stocks 1000 \
    --bars 800 \
    --freq d \
    --cache-dir data_cache/strategy_exec \
    --out-excel strategy_pick_2026-04-08.xlsx

2) 指定日期区间选股：
python c0_strategy.py \
    --mode pick \
    --start-date 2026-03-10 \
    --end-date 2026-04-08 \
    --bars 800 \
    --freq d \
    --cache-dir data_cache/strategy_exec \
    --out-excel strategy_pick_2026-03-10_to_04-08.xlsx

3) 单股买卖回测：
python c0_strategy.py \
    --mode backtest \
    --symbol 600547.SH \
    --bars 1200 \
    --freq d \
    --exit-template S2_early_protect \
    --out-excel strategy_backtest_600547.xlsx

Outputs
=======
- pick 单日模式：
    Sheet[signals]  当日选股结果
    Sheet[summary]  汇总统计

- pick 区间模式：
    Sheet[pick_YYYY-MM-DD]  每日选股结果（每个日期一个sheet）
    Sheet[all_signals]      所有日期选股结果汇总
    Sheet[summary]  规则统计汇总

- backtest 模式：
    Sheet[trades]   每笔交易明细
    Sheet[equity]   资金曲线与统计
    Sheet[signals]  所有历史买点事件
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from datasource.database import get_engine
try:
    from features.merged_dsa_atr_rope_bb_factors import (
        DSAConfig,
        RopeConfig,
        compute_atr_rope,
        compute_bollinger,
        compute_dsa,
    )
except Exception:
    from merged_dsa_atr_rope_bb_factors import (
        DSAConfig,
        RopeConfig,
        compute_atr_rope,
        compute_bollinger,
        compute_dsa,
    )

warnings.filterwarnings("ignore", category=FutureWarning)

EVENT_DEDUP_BARS = 20
EXECUTION_LAG = 1
ENTRY_PRICE_MODE = "next_open"
RULE_LOOKBACK = 252
OUT_DIR = "strategy_exec_output"
os.makedirs(OUT_DIR, exist_ok=True)


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalize_01(s: pd.Series, reverse: bool = False, q: Tuple[float, float] = (0.05, 0.95)) -> pd.Series:
    x = safe_numeric(s).copy()
    lo = x.quantile(q[0])
    hi = x.quantile(q[1])
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        out = pd.Series(0.5, index=x.index, dtype=float)
    else:
        x = x.clip(lower=lo, upper=hi)
        out = (x - lo) / (hi - lo)
    if reverse:
        out = 1.0 - out
    return out.clip(0.0, 1.0)


def rolling_quantile(s: pd.Series, q: float, lookback: int = RULE_LOOKBACK) -> pd.Series:
    return s.shift(1).rolling(lookback, min_periods=max(60, lookback // 3)).quantile(q)


def load_kline(ts_code: str, freq: str = "d", bars: int = 1000) -> Optional[pd.DataFrame]:
    from sqlalchemy import text

    engine = get_engine()
    sql = """
        SELECT bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = :freq
        ORDER BY bar_time DESC LIMIT :limit
    """
    try:
        df = pd.read_sql_query(text(sql), engine, params={"ts_code": ts_code, "freq": freq, "limit": bars})
    except Exception as exc:
        print(f"[WARN] {ts_code} 查询失败: {exc}")
        return None
    finally:
        engine.dispose()
    if df.empty:
        return None
    df = df.sort_values("bar_time").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["bar_time"]).dt.tz_localize(None)
    df = df.set_index("datetime")
    df["vol"] = df["volume"].astype(float).replace(0, np.nan).ffill().fillna(1.0)
    return df[["open", "high", "low", "close", "vol"]].astype(float)


def get_stock_pool(n: Optional[int] = None, seed: int = 42) -> List[str]:
    from sqlalchemy import text

    engine = get_engine()
    sql = "SELECT ts_code FROM stock_pools ORDER BY ts_code"
    try:
        df = pd.read_sql_query(text(sql), engine)
    finally:
        engine.dispose()
    if df.empty:
        raise ValueError("stock_pools表为空")
    codes = df["ts_code"].tolist()
    if n is None:
        return codes
    rng = np.random.default_rng(seed)
    return rng.choice(codes, size=min(n, len(codes)), replace=False).tolist()


def get_cache_path(cache_dir: str, mode_tag: str, n_stocks: int, bars: int, freq: str, seed: int) -> str:
    param_str = f"{mode_tag}_{n_stocks}_{bars}_{freq}_{seed}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"strategy_{mode_tag}_{freq}_{bars}_{n_stocks}_{param_hash}.parquet")


def aggregate_weekly_strict(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    if "volume" not in d.columns and "vol" in d.columns:
        d["volume"] = d["vol"]
    wk = pd.DataFrame(index=d.resample("W-FRI").last().index)
    wk["open"] = d["open"].resample("W-FRI").first()
    wk["high"] = d["high"].resample("W-FRI").max()
    wk["low"] = d["low"].resample("W-FRI").min()
    wk["close"] = d["close"].resample("W-FRI").last()
    wk["volume"] = d["volume"].resample("W-FRI").sum()
    return wk.dropna()


def build_trade_safe_dsa_features(df: pd.DataFrame, lookback: int = 50, prefix: str = "") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    hi = df["high"].rolling(lookback, min_periods=max(10, lookback // 3)).max()
    lo = df["low"].rolling(lookback, min_periods=max(10, lookback // 3)).min()
    rng = (hi - lo).replace(0, np.nan)
    out[f"{prefix}dsa_trade_pos_01"] = ((df["close"] - lo) / rng).clip(0.0, 1.0)
    out[f"{prefix}dsa_trade_range_width_pct"] = rng / df["close"].replace(0, np.nan)
    out[f"{prefix}dsa_trade_dist_to_low_01"] = ((df["close"] - lo) / rng).clip(0.0, 1.0)
    out[f"{prefix}dsa_trade_dist_to_high_01"] = ((hi - df["close"]) / rng).clip(0.0, 1.0)
    out[f"{prefix}dsa_trade_breakout_20"] = (df["close"] >= df["high"].shift(1).rolling(20, min_periods=10).max()).astype(float)
    out[f"{prefix}dsa_trade_breakdown_20"] = (df["close"] <= df["low"].shift(1).rolling(20, min_periods=10).min()).astype(float)
    return out


def map_weekly_dsa_confirmed_to_daily(daily: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=daily.index)
    wk = aggregate_weekly_strict(daily)
    if len(wk) < 10:
        return out
    w_dsa, _, _ = compute_dsa(wk, DSAConfig(prd=50, base_apt=20.0))
    prev_friday = (daily.index.to_period("W-FRI").start_time + pd.offsets.Week(weekday=4) - pd.offsets.Week(1)).normalize()
    mapped = w_dsa.copy()
    mapped.index = mapped.index.normalize()
    out["w_DSA_DIR"] = mapped["DSA_DIR"].reindex(prev_friday).to_numpy()
    out["w_dsa_confirmed_pivot_pos_01"] = mapped["dsa_pivot_pos_01"].reindex(prev_friday).to_numpy()
    out["w_dsa_signed_vwap_dev_pct"] = mapped["signed_vwap_dev_pct"].reindex(prev_friday).to_numpy()
    out["w_prev_confirmed_up_bars"] = mapped["prev_confirmed_up_bars"].reindex(prev_friday).to_numpy() if "prev_confirmed_up_bars" in mapped.columns else np.nan
    out["w_prev_confirmed_down_bars"] = mapped["prev_confirmed_down_bars"].reindex(prev_friday).to_numpy() if "prev_confirmed_down_bars" in mapped.columns else np.nan
    return out


def map_weekly_dsa_trade_to_daily(daily: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=daily.index)
    wk = aggregate_weekly_strict(daily)
    if len(wk) < 10:
        return out
    trade_wk = build_trade_safe_dsa_features(wk, lookback=20)
    prev_friday = (daily.index.to_period("W-FRI").start_time + pd.offsets.Week(weekday=4) - pd.offsets.Week(1)).normalize()
    mapped = trade_wk.copy()
    mapped.index = mapped.index.normalize()
    for col in mapped.columns:
        out[f"w_{col}"] = mapped[col].reindex(prev_friday).to_numpy()
    out["w_factor_available"] = mapped["dsa_trade_pos_01"].reindex(prev_friday).notna().astype(float).to_numpy()
    return out


def add_micro_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    vol = d["vol"].astype(float)
    close = d["close"].astype(float)
    d["ret_1"] = close.pct_change(1)
    d["ret_3"] = close.pct_change(3)
    d["ret_5"] = close.pct_change(5)
    d["vol_ma_20"] = vol.rolling(20, min_periods=5).mean()
    d["vol_ratio_20"] = vol / d["vol_ma_20"].replace(0, np.nan)
    vol_std = vol.rolling(20, min_periods=10).std().replace(0, np.nan)
    d["vol_z_20"] = (vol - d["vol_ma_20"]) / vol_std
    d["close_ma_10"] = close.rolling(10, min_periods=5).mean()
    d["close_ma_20"] = close.rolling(20, min_periods=10).mean()
    d["trend_gap_10_20"] = (d["close_ma_10"] - d["close_ma_20"]) / close.replace(0, np.nan)
    return d


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    low_pos_parts = [c for c in [
        "dsa_trade_pos_01", "bb_pos_01", "channel_pos_01", "range_pos_01", "rope_pivot_pos_01", "w_dsa_trade_pos_01"
    ] if c in out.columns]
    if low_pos_parts:
        out["position_in_structure"] = pd.concat([safe_numeric(out[c]) for c in low_pos_parts], axis=1).mean(axis=1)
    else:
        out["position_in_structure"] = np.nan

    dir_flags = pd.DataFrame(index=out.index)
    dir_flags["rope_up"] = (safe_numeric(out.get("rope_dir", pd.Series(index=out.index))) == 1).astype(float)
    dir_flags["rope_slope_pos"] = (safe_numeric(out.get("rope_slope_atr_5", pd.Series(index=out.index))) > 0).astype(float)
    dir_flags["recent_turn"] = (safe_numeric(out.get("bars_since_dir_change", pd.Series(index=out.index))).fillna(99) <= 8).astype(float)
    dir_flags["breakout_or_expand"] = ((safe_numeric(out.get("range_break_up", pd.Series(index=out.index))).fillna(0) > 0) |
                                        (safe_numeric(out.get("bb_expanding", pd.Series(index=out.index))).fillna(0) > 0) |
                                        (safe_numeric(out.get("dsa_trade_breakout_20", pd.Series(index=out.index))).fillna(0) > 0)).astype(float)
    dir_flags["weekly_support"] = ((safe_numeric(out.get("w_DSA_DIR", pd.Series(index=out.index))).fillna(0) >= 0) |
                                    (safe_numeric(out.get("w_dsa_trade_pos_01", pd.Series(index=out.index))).fillna(1) <= 0.5)).astype(float)
    out["dir_consistent"] = dir_flags.sum(axis=1)

    trend_parts = pd.DataFrame(index=out.index)
    trend_parts["rope_slope"] = normalize_01(safe_numeric(out.get("rope_slope_atr_5", pd.Series(index=out.index))), reverse=False)
    trend_parts["trend_gap"] = normalize_01(safe_numeric(out.get("trend_gap_10_20", pd.Series(index=out.index))), reverse=False)
    trend_parts["signed_dev"] = normalize_01(safe_numeric(out.get("dsa_signed_vwap_dev_pct", pd.Series(index=out.index))), reverse=False)
    trend_parts["current_run_early"] = normalize_01(safe_numeric(out.get("current_run_bars", pd.Series(index=out.index))), reverse=True)
    trend_parts["bars_since_early"] = normalize_01(safe_numeric(out.get("bars_since_dir_change", pd.Series(index=out.index))), reverse=True)
    out["score_trend_total"] = trend_parts.mean(axis=1)

    width_parts = pd.DataFrame(index=out.index)
    width_parts["bbw_pct_low"] = normalize_01(safe_numeric(out.get("bb_width_percentile", pd.Series(index=out.index))), reverse=True)
    width_parts["range_atr_low"] = normalize_01(safe_numeric(out.get("range_width_atr", pd.Series(index=out.index))), reverse=True)
    width_parts["dsa_width_low"] = normalize_01(safe_numeric(out.get("dsa_trade_range_width_pct", pd.Series(index=out.index))), reverse=True)
    width_parts["contract"] = normalize_01(safe_numeric(out.get("bb_contract_streak", pd.Series(index=out.index))), reverse=False)
    out["score_width_total"] = width_parts.mean(axis=1)

    out["score_setup_total"] = pd.concat([
        safe_numeric(out["position_in_structure"]).pipe(lambda s: 1 - s.clip(0, 1)),
        normalize_01(out["dir_consistent"], reverse=False),
        safe_numeric(out["score_trend_total"]),
        safe_numeric(out["score_width_total"]),
    ], axis=1).mean(axis=1)
    return out


def add_rule_threshold_features(df: pd.DataFrame, lookback: int = RULE_LOOKBACK) -> pd.DataFrame:
    out = df.copy()
    # high = 高于历史60分位；low40 = 低于历史40分位
    for col in ["bb_width_norm", "dsa_trade_range_width_pct", "trend_gap_10_20", "score_trend_total", "bb_pos_01", "dsa_trade_pos_01"]:
        if col not in out.columns:
            continue
        q40 = rolling_quantile(out[col], 0.4, lookback)
        q60 = rolling_quantile(out[col], 0.6, lookback)
        out[f"{col}_low40"] = (out[col] <= q40).astype(int)
        out[f"{col}_high"] = (out[col] >= q60).astype(int)
    return out


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "vol" in d.columns and "volume" not in d.columns:
        d["volume"] = d["vol"]
    d = add_micro_features(d)
    dsa_df, _, _ = compute_dsa(d, DSAConfig(prd=50, base_apt=20.0))
    dsa_confirmed = dsa_df.rename(columns={
        "dsa_pivot_high": "dsa_confirmed_pivot_high",
        "dsa_pivot_low": "dsa_confirmed_pivot_low",
        "dsa_pivot_pos_01": "dsa_confirmed_pivot_pos_01",
        "last_pivot_type": "last_confirmed_pivot_type",
        "signed_vwap_dev_pct": "dsa_signed_vwap_dev_pct",
        "bull_vwap_dev_pct": "dsa_bull_vwap_dev_pct",
        "bear_vwap_dev_pct": "dsa_bear_vwap_dev_pct",
        "trend_aligned_vwap_dev_pct": "dsa_trend_aligned_vwap_dev_pct",
    })
    dsa_trade = build_trade_safe_dsa_features(d, lookback=50)
    rope_df = compute_atr_rope(d, RopeConfig(length=14, multi=1.5))
    bb_df = compute_bollinger(d, length=20, mult=2.0, pct_lookback=120)
    weekly_confirmed = map_weekly_dsa_confirmed_to_daily(d)
    weekly_trade = map_weekly_dsa_trade_to_daily(d)
    merged = pd.concat([
        d, dsa_confirmed, dsa_trade,
        rope_df.drop(columns=d.columns, errors="ignore"),
        bb_df.drop(columns=d.columns, errors="ignore"),
        weekly_confirmed, weekly_trade,
    ], axis=1)
    merged = add_composite_features(merged)
    merged = add_rule_threshold_features(merged)
    return merged


def add_observation_windows(df: pd.DataFrame, windows: Sequence[int] = (5, 10, 20, 40), entry_lag: int = EXECUTION_LAG) -> pd.DataFrame:
    out = df.copy()
    open_ = out["open"].to_numpy(float)
    close = out["close"].to_numpy(float)
    n = len(out)
    for w in windows:
        vals = np.full(n, np.nan)
        for i in range(n):
            entry_idx = i + entry_lag
            exit_idx = entry_idx + w
            if exit_idx >= n or entry_idx >= n:
                continue
            entry = open_[entry_idx]
            if not np.isfinite(entry) or entry <= 0:
                continue
            vals[i] = (close[exit_idx] - entry) / entry
        out[f"obs_ret_{w}"] = vals
    return out


def add_path_quality_labels(df: pd.DataFrame, horizon: int = 10, entry_lag: int = EXECUTION_LAG) -> pd.DataFrame:
    out = df.copy()
    open_ = out["open"].to_numpy(float)
    close = out["close"].to_numpy(float)
    high = out["high"].to_numpy(float)
    low = out["low"].to_numpy(float)
    n = len(out)

    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    bars_to_stop = np.full(n, np.nan)
    bars_to_3pct = np.full(n, np.nan)
    hit_3 = np.full(n, np.nan)
    hit_5 = np.full(n, np.nan)

    out["signal_idx"] = np.arange(n, dtype=float)
    out["entry_idx"] = np.arange(n, dtype=float) + float(entry_lag)
    out["entry_datetime"] = pd.NaT
    out["entry_price"] = np.nan

    for i in range(n):
        entry_idx = i + entry_lag
        end = min(n - 1, entry_idx + horizon)
        if entry_idx >= n or entry_idx >= end:
            continue
        entry = open_[entry_idx]
        if not np.isfinite(entry) or entry <= 0:
            continue
        fut_high = high[entry_idx + 1:end + 1]
        fut_low = low[entry_idx + 1:end + 1]
        if len(fut_high) == 0:
            continue
        out.at[out.index[i], "entry_datetime"] = out.index[entry_idx]
        out.at[out.index[i], "entry_price"] = entry
        mfe[i] = (np.nanmax(fut_high) - entry) / entry
        mae_raw = (np.nanmin(fut_low) - entry) / entry
        mae[i] = min(0.0, mae_raw)
        stop_price = low[i]
        stop_hits = np.where(fut_low <= stop_price)[0]
        stop_idx = int(stop_hits[0] + 1) if len(stop_hits) > 0 else None
        if stop_idx is not None:
            bars_to_stop[i] = stop_idx
        tp3_hits = np.where(fut_high >= entry * 1.03)[0]
        tp5_hits = np.where(fut_high >= entry * 1.05)[0]
        tp3_idx = int(tp3_hits[0] + 1) if len(tp3_hits) > 0 else None
        tp5_idx = int(tp5_hits[0] + 1) if len(tp5_hits) > 0 else None
        if tp3_idx is not None:
            bars_to_3pct[i] = tp3_idx
        hit_3[i] = float(tp3_idx is not None and (stop_idx is None or tp3_idx < stop_idx))
        hit_5[i] = float(tp5_idx is not None and (stop_idx is None or tp5_idx < stop_idx))

    out["path_mfe_10"] = mfe
    out["path_mae_10"] = mae
    out["bars_to_stop_10"] = bars_to_stop
    out["bars_to_3pct_10"] = bars_to_3pct
    out["hit_3pct_before_stop_10"] = hit_3
    out["hit_5pct_before_stop_10"] = hit_5
    out["path_efficiency_10"] = safe_numeric(out["path_mfe_10"]) / safe_numeric(out["path_mae_10"]).abs().replace(0, np.nan)
    out["asymmetry_score"] = out["path_efficiency_10"]
    out["confirm_before_invalid"] = ((out["hit_3pct_before_stop_10"] == 1.0) & (safe_numeric(out["bars_to_stop_10"]).isna())).astype(int)
    quality = pd.concat([
        normalize_01(out["hit_3pct_before_stop_10"]),
        normalize_01(out["hit_5pct_before_stop_10"]),
        normalize_01(out["path_mfe_10"]),
        normalize_01(out["path_mae_10"], reverse=True),
        normalize_01(out["path_efficiency_10"]),
        normalize_01(out["asymmetry_score"]),
        normalize_01(out["bars_to_3pct_10"], reverse=True),
        normalize_01(out["bars_to_stop_10"].fillna(11), reverse=False),
    ], axis=1).mean(axis=1)
    out["quality_trade_score"] = quality.clip(0.0, 1.0)
    out["good_path_main"] = ((out["hit_5pct_before_stop_10"] == 1.0) & (out["path_mae_10"] >= -0.08)).astype(int)
    return out


@dataclass
class NeighborSpec:
    dsa_max: float = 0.45
    prev_down_min: float = 8.0
    need_rope_up: bool = True
    current_run_max: float = 20.0
    bars_since_max: float = 12.0


def dedup_events(df: pd.DataFrame, cooldown_bars: int = EVENT_DEDUP_BARS) -> pd.DataFrame:
    if df.empty or "symbol" not in df.columns:
        return df.copy()
    work = df.sort_values(["symbol", "datetime"]).copy()
    keep_idx: List[int] = []
    for _, g in work.groupby("symbol", sort=False):
        accepted_positions: List[int] = []
        idx_list = g.index.tolist()
        for pos, idx in enumerate(idx_list):
            if not accepted_positions or pos - accepted_positions[-1] >= cooldown_bars:
                accepted_positions.append(pos)
                keep_idx.append(idx)
    return work.loc[keep_idx].sort_values(["datetime", "symbol"]).reset_index(drop=True)


def build_neighbor_events(df: pd.DataFrame, spec: NeighborSpec, apply_cooldown: bool = True) -> pd.DataFrame:
    cols = [c for c in [
        "symbol", "datetime", "open", "high", "low", "close", "entry_datetime", "entry_price",
        "dsa_trade_pos_01", "bb_pos_01", "dsa_trade_range_width_pct", "bb_width_norm",
        "trend_gap_10_20", "score_trend_total", "bars_since_dir_change", "bb_expand_streak",
        "prev_confirmed_down_bars", "current_run_bars", "rope_dir", "quality_trade_score",
        "good_path_main", "asymmetry_score", "confirm_before_invalid",
        "bb_width_norm_high", "dsa_trade_range_width_pct_high", "trend_gap_10_20_low40",
        "score_trend_total_low40", "bb_pos_01_low40", "dsa_trade_pos_01_low40",
        "obs_ret_5", "obs_ret_10", "obs_ret_20", "obs_ret_40"
    ] if c in df.columns]
    sub = df[cols].copy()
    mask = pd.Series(True, index=sub.index)
    if "dsa_trade_pos_01" in sub.columns:
        mask &= safe_numeric(sub["dsa_trade_pos_01"]) <= spec.dsa_max
    if "prev_confirmed_down_bars" in sub.columns:
        mask &= safe_numeric(sub["prev_confirmed_down_bars"]) >= spec.prev_down_min
    if spec.need_rope_up and "rope_dir" in sub.columns:
        mask &= safe_numeric(sub["rope_dir"]) == 1
    if "current_run_bars" in sub.columns:
        mask &= safe_numeric(sub["current_run_bars"]).fillna(999) <= spec.current_run_max
    if "bars_since_dir_change" in sub.columns:
        mask &= safe_numeric(sub["bars_since_dir_change"]).fillna(999) <= spec.bars_since_max
    out = sub.loc[mask].copy()
    if apply_cooldown:
        out = dedup_events(out)
    return out


def apply_final_rules(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    # 规则 A/B/C 的精确定义写在这里，便于选股和回测时明确标注满足哪条规则。
    out["rule_A_width_trend"] = ((out.get("bb_width_norm_high", 0) == 1) & (out.get("trend_gap_10_20_low40", 0) == 1)).astype(int)
    out["rule_B_dsa_width_trend"] = ((out.get("dsa_trade_range_width_pct_high", 0) == 1) & (out.get("trend_gap_10_20_low40", 0) == 1)).astype(int)
    out["rule_C_enhanced_bb"] = ((out.get("bb_width_norm_high", 0) == 1) & (out.get("trend_gap_10_20_low40", 0) == 1) & (out.get("score_trend_total_low40", 0) == 1)).astype(int)
    out["rule_C_enhanced_dsa"] = ((out.get("dsa_trade_range_width_pct_high", 0) == 1) & (out.get("trend_gap_10_20_low40", 0) == 1) & (out.get("score_trend_total_low40", 0) == 1)).astype(int)

    def pick_rule(row: pd.Series) -> Tuple[str, str]:
        if row.get("rule_C_enhanced_bb", 0) == 1:
            return "C_enhanced_bb", "C"
        if row.get("rule_C_enhanced_dsa", 0) == 1:
            return "C_enhanced_dsa", "C"
        if row.get("rule_A_width_trend", 0) == 1:
            return "A_width_trend", "A"
        if row.get("rule_B_dsa_width_trend", 0) == 1:
            return "B_dsa_width_trend", "B"
        return "", ""

    picked = out.apply(pick_rule, axis=1, result_type="expand")
    out["rule_name"] = picked[0]
    out["rule_family"] = picked[1]
    out = out[out["rule_name"] != ""].copy()
    return out


def simulate_trade(df: pd.DataFrame, signal_idx: int, rule_name: str, exit_template: str = "S2_early_protect") -> Dict[str, object]:
    # 所有回测都明确写清楚：
    # - 买点来自哪条规则
    # - 买入发生在 signal_idx + 1 开盘
    # - 卖点模板是哪套
    # - 卖出原因是什么
    n = len(df)
    entry_idx = signal_idx + EXECUTION_LAG
    if entry_idx >= n:
        return {}
    entry_dt = df.index[entry_idx]
    entry_price = float(df.iloc[entry_idx]["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {}
    signal_low = float(df.iloc[signal_idx]["low"])
    highest_high = entry_price
    reached_tp8 = False
    rope_col_exists = "rope" in df.columns or "atr_rope" in df.columns
    rope_price_col = "rope" if "rope" in df.columns else ("atr_rope" if "atr_rope" in df.columns else None)
    exit_idx = None
    exit_price = np.nan
    exit_reason = "hold_to_end"

    for i in range(entry_idx + 1, n):
        row = df.iloc[i]
        bar_open = float(row["open"])
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])
        highest_high = max(highest_high, bar_high)

        # 共同硬止损：跌破事件日最低点
        if bar_low <= signal_low:
            exit_idx = i
            exit_price = min(bar_open, signal_low) if bar_open <= signal_low else signal_low
            exit_reason = "event_low_stop"
            break

        if exit_template == "S2_early_protect":
            # S2：利润保护优先。
            # 若自买入后的最高价回撤达到 4%，卖出。
            trail_line = highest_high * (1.0 - 0.04)
            if bar_low <= trail_line:
                exit_idx = i
                exit_price = min(bar_open, trail_line) if bar_open <= trail_line else trail_line
                exit_reason = "peak_pullback_4"
                break

        elif exit_template == "S3_profit_extend":
            # S3：先要求达到 +8% 浮盈，再用 rope / trend 破坏退出。
            if highest_high >= entry_price * 1.08:
                reached_tp8 = True
            if reached_tp8:
                rope_break = False
                if rope_price_col is not None and pd.notna(row.get(rope_price_col, np.nan)):
                    rope_break = bar_close < float(row[rope_price_col])
                dir_flip = (row.get("rope_dir", np.nan) == -1)
                if rope_break or dir_flip:
                    exit_idx = i
                    exit_price = bar_close
                    exit_reason = "tp8_then_rope"
                    break
        else:
            raise ValueError(f"未知卖点模板: {exit_template}")

    if exit_idx is None:
        exit_idx = n - 1
        exit_price = float(df.iloc[exit_idx]["close"])
        exit_reason = "end_of_data"

    ret = (exit_price - entry_price) / entry_price if entry_price > 0 else np.nan
    return {
        "signal_date": df.index[signal_idx],
        "buy_date": entry_dt,
        "buy_price": entry_price,
        "rule_name": rule_name,
        "exit_template": exit_template,
        "sell_date": df.index[exit_idx],
        "sell_price": exit_price,
        "exit_reason": exit_reason,
        "holding_bars": int(exit_idx - entry_idx),
        "trade_return": ret,
    }


def prepare_symbol_factors(symbol: str, bars: int, freq: str) -> Optional[pd.DataFrame]:
    kline = load_kline(symbol, freq=freq, bars=bars)
    if kline is None or len(kline) < 150:
        return None
    fac = compute_factors(kline)
    fac = add_observation_windows(fac)
    fac = add_path_quality_labels(fac)
    fac["symbol"] = symbol
    fac["datetime"] = fac.index
    return fac


def build_pool_for_symbols(symbols: List[str], bars: int, freq: str, cache_dir: str, seed: int = 42) -> pd.DataFrame:
    cache_path = get_cache_path(cache_dir, "pick", len(symbols), bars, freq, seed)
    if os.path.exists(cache_path):
        print(f"发现缓存文件，正在加载: {cache_path}")
        return pd.read_parquet(cache_path)

    all_df: List[pd.DataFrame] = []
    for idx, sym in enumerate(symbols):
        fac = prepare_symbol_factors(sym, bars=bars, freq=freq)
        if fac is None:
            continue
        all_df.append(fac.reset_index(drop=True))
        if (idx + 1) % 50 == 0:
            print(f"已处理 {idx + 1}/{len(symbols)} 只股票")
    if not all_df:
        return pd.DataFrame()
    combined = pd.concat(all_df, ignore_index=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    return combined


def run_pick_mode(
    pick_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    n_stocks: Optional[int] = None,
    bars: int = 800,
    freq: str = "d",
    cache_dir: str = "data_cache/strategy_exec",
    out_excel: Optional[str] = None
) -> None:
    symbols = get_stock_pool(n=n_stocks)
    raw = build_pool_for_symbols(symbols, bars=bars, freq=freq, cache_dir=cache_dir)
    if raw.empty:
        print("无可用样本")
        return
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    events = build_neighbor_events(raw, NeighborSpec(), apply_cooldown=False)
    events = apply_final_rules(events)

    # 日期筛选逻辑：支持单日或日期区间
    if pick_date is not None:
        # 单日选股模式
        target_date = pd.to_datetime(pick_date)
        signals = events[events["datetime"].dt.normalize() == target_date.normalize()].copy()
        is_range_mode = False
        summary = pd.DataFrame({
            "metric": ["pick_date", "n_symbols", "event_pool_n", "signal_n"],
            "value": [pick_date, len(symbols), int(len(events)), int(len(signals))],
        })
    else:
        # 日期区间选股模式
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        signals = events[(events["datetime"].dt.normalize() >= start_dt.normalize()) &
                         (events["datetime"].dt.normalize() <= end_dt.normalize())].copy()
        is_range_mode = True
        summary = pd.DataFrame({
            "metric": ["start_date", "end_date", "n_symbols", "event_pool_n", "signal_n"],
            "value": [start_date, end_date, len(symbols), int(len(events)), int(len(signals))],
        })

    if signals.empty:
        print("指定日期/区间无满足最终规则的股票")
    else:
        signals["planned_exit_template"] = "S2_early_protect"
        keep = [c for c in [
            "datetime", "symbol", "rule_name", "rule_family", "entry_datetime", "entry_price",
            "planned_exit_template", "quality_trade_score", "good_path_main", "asymmetry_score", "confirm_before_invalid",
            "dsa_trade_pos_01", "bb_pos_01", "dsa_trade_range_width_pct", "bb_width_norm", "trend_gap_10_20", "score_trend_total",
            "obs_ret_5", "obs_ret_10", "obs_ret_20", "obs_ret_40"
        ] if c in signals.columns]
        signals = signals[keep].sort_values(["datetime", "rule_family", "quality_trade_score", "asymmetry_score"],
                                            ascending=[True, True, False, False])

        if is_range_mode:
            # 区间模式：按日期分组打印
            print("\n=== 日期区间选股结果 ===")
            for date, group in signals.groupby(signals["datetime"].dt.date):
                print(f"\n【{date}】选出 {len(group)} 只股票:")
                for _, row in group.iterrows():
                    print(f"  - {row['symbol']}: 规则={row['rule_name']}, 家族={row['rule_family']}, quality={row['quality_trade_score']:.3f}")
        else:
            # 单日模式：原有输出
            print(signals.head(30).to_string(index=False))

    if out_excel:
        out_path = out_excel if os.path.isabs(out_excel) else os.path.join(OUT_DIR, out_excel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            if is_range_mode and not signals.empty:
                # 区间模式：每个日期一个sheet，包含当日选股明细
                for date, group in signals.groupby(signals["datetime"].dt.date):
                    sheet_name = f"pick_{date}"
                    group.to_excel(writer, sheet_name=sheet_name, index=False)

                # 增加一个汇总sheet，列出每天选出的股票列表
                summary_rows = []
                for date, group in signals.groupby(signals["datetime"].dt.date):
                    for _, row in group.iterrows():
                        summary_rows.append({
                            "date": date,
                            "symbol": row["symbol"],
                            "rule_name": row["rule_name"],
                            "rule_family": row["rule_family"],
                            "quality_trade_score": row["quality_trade_score"]
                        })
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_excel(writer, sheet_name="all_signals", index=False)
            else:
                # 单日模式：原有输出
                signals.to_excel(writer, sheet_name="signals", index=False)
                summary.to_excel(writer, sheet_name="summary", index=False)
        print(f"已输出: {out_path}")


def run_backtest_mode(symbol: str, bars: int, freq: str, exit_template: str, out_excel: Optional[str]) -> None:
    fac = prepare_symbol_factors(symbol, bars=bars, freq=freq)
    if fac is None or fac.empty:
        print("无数据")
        return
    events = build_neighbor_events(fac.reset_index(drop=True), NeighborSpec())
    events = apply_final_rules(events)
    if events.empty:
        print("该股票无满足最终规则的事件")
        return

    fac_work = fac.copy()
    trades: List[Dict[str, object]] = []
    last_exit_dt = None
    for _, ev in events.sort_values("datetime").iterrows():
        signal_dt = pd.to_datetime(ev["datetime"])
        if last_exit_dt is not None and signal_dt <= last_exit_dt:
            continue
        signal_idx = fac_work.index.get_loc(signal_dt)
        trade = simulate_trade(fac_work, signal_idx=signal_idx, rule_name=ev["rule_name"], exit_template=exit_template)
        if trade:
            trades.append(trade)
            last_exit_dt = trade["sell_date"]

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("无交易")
        return
    trades_df["cum_return_factor"] = (1.0 + trades_df["trade_return"].fillna(0.0)).cumprod()
    equity = trades_df[["sell_date", "cum_return_factor"]].copy()
    summary = pd.DataFrame({
        "metric": [
            "symbol", "exit_template", "n_trades", "win_rate", "avg_trade_return", "cum_return_factor_final"
        ],
        "value": [
            symbol,
            exit_template,
            int(len(trades_df)),
            float((trades_df["trade_return"] > 0).mean()),
            float(trades_df["trade_return"].mean()),
            float(trades_df["cum_return_factor"].iloc[-1]),
        ]
    })
    print(trades_df.tail(20).to_string(index=False))

    if out_excel:
        out_path = out_excel if os.path.isabs(out_excel) else os.path.join(OUT_DIR, out_excel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            trades_df.to_excel(writer, sheet_name="trades", index=False)
            equity.to_excel(writer, sheet_name="equity", index=False)
            events.to_excel(writer, sheet_name="signals", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
        print(f"已输出: {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="最终执行版策略：指定日期选股 + 单股买卖回测")
    p.add_argument("--mode", choices=["pick", "backtest"], required=True)
    p.add_argument("--pick-date", type=str, default=None, help="指定单日选股，格式YYYY-MM-DD，与--start-date/--end-date二选一")
    p.add_argument("--start-date", type=str, default=None, help="选股开始日期，格式YYYY-MM-DD，与--pick-date二选一")
    p.add_argument("--end-date", type=str, default=None, help="选股结束日期，格式YYYY-MM-DD，与--pick-date二选一")
    p.add_argument("--symbol", type=str, default=None)
    p.add_argument("--n-stocks", type=int, default=None, help="股票数量，默认None表示全量")
    p.add_argument("--bars", type=int, default=1000)
    p.add_argument("--freq", type=str, default="d")
    p.add_argument("--cache-dir", type=str, default="data_cache/strategy_exec")
    p.add_argument("--exit-template", choices=["S2_early_protect", "S3_profit_extend"], default="S2_early_protect")
    p.add_argument("--out-excel", type=str, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.mode == "pick":
        # 参数校验：二选一
        has_single = args.pick_date is not None
        has_range = args.start_date is not None or args.end_date is not None

        if has_single and has_range:
            raise ValueError("--pick-date 与 --start-date/--end-date 不能同时使用，请二选一")

        if not has_single and not has_range:
            raise ValueError("pick 模式需要 --pick-date 或 --start-date/--end-date")

        if has_range and (args.start_date is None or args.end_date is None):
            raise ValueError("使用日期区间模式时，--start-date 和 --end-date 都必须指定")

        run_pick_mode(
            pick_date=args.pick_date,
            start_date=args.start_date,
            end_date=args.end_date,
            n_stocks=args.n_stocks,
            bars=args.bars,
            freq=args.freq,
            cache_dir=args.cache_dir,
            out_excel=args.out_excel
        )
    elif args.mode == "backtest":
        if not args.symbol:
            raise ValueError("backtest 模式需要 --symbol")
        run_backtest_mode(args.symbol, args.bars, args.freq, args.exit_template, args.out_excel)


if __name__ == "__main__":
    main()
