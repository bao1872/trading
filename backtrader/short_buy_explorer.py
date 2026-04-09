# -*- coding: utf-8 -*-
"""
超短关系探索器：基于 DSA / ATR Rope / Bollinger / 量价因子，研究未来 1/2/3 天收益与因子的关系

Purpose
- 复用现有 DSA VWAP + ATR Rope + Bollinger 因子主干，面向超短窗口（1/2/3 天）做关系探索
- 尽量少预设买点条件：默认使用全 bar 样本，而非先构造候选事件池
- 输出单因子分桶、双因子热力摘要、单因子相关性、规则分层、稳定性检验结果，帮助发现短线有效因子关系
- 新增 volume zscore 多周期家族、K 线短线因子、未来 1/2/3 天收益/冲高/回撤标签

Inputs
- 数据库表：stock_k_data, stock_pools（与原探索脚本一致）
- 外部因子模块：merged_dsa_atr_rope_bb_factors.py 或项目内 features.merged_dsa_atr_rope_bb_factors
- 运行参数：股票数、K线数量、频率、缓存目录、输出目录等

Outputs
- short_buy_explorer_results.xlsx       单工作簿输出，包含多个 sheet
  - 00_dataset_summary                  数据集摘要
  - 01_feature_inventory                因子清单
  - 02_research_panel_sample            研究主表样例
  - 03_single_factor_bucket_ret         单因子分桶：ret_1/2/3
  - 04_single_factor_bucket_trade       单因子分桶：max_up/max_dd/hit 比率
  - 05_pair_heatmap_summary             双因子热力摘要
  - 06_single_factor_correlation        单因子相关性摘要
  - 07_rule_layering_summary            规则分层摘要
  - 08_stability_by_time                时间切片稳定性
  - 09_stability_by_universe            股票分层稳定性
  - 10_stability_by_dedup               去重稳定性
  - 11_short_horizon_hypotheses         最终假设清单

How to Run
    python short_buy_explorer.py --n-stocks 200 --bars 1000 --freq d

    python short_buy_explorer.py --n-stocks 1000 --bars 1200 --freq d \
        --pair-top-k 15 --bucket-q 10 --analysis-mode full

Examples
    python short_buy_explorer.py --n-stocks 100 --bars 800 --freq d --seed 42
    python short_buy_explorer.py --n-stocks 500 --bars 1000 --freq d --sample-export-rows 3000

Side Effects
- 从数据库读取行情数据与股票池
- 在 cache_dir 下读写 parquet/json 缓存
- 在 output_dir 下生成单个 Excel 工作簿（多 sheet），不再输出多个 CSV 文件
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

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

from sqlalchemy import text

OUT_DIR = "short_buy_explorer_output"
RET_WINDOWS = [1, 2, 3]
RANDOM_STATE = 42
PAIR_BIN_Q = 5
DEFAULT_BUCKET_Q = 10
MIN_VALID_ROWS_PER_STOCK = 80
MIN_ROWS_FOR_BUCKET = 80


# =========================
# helpers
# =========================
def rr_from_ret_dd(ret: float, dd: float) -> float:
    if pd.isna(ret) or pd.isna(dd) or dd == 0:
        return np.nan
    return float(ret / abs(dd))


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


def rolling_rank_pct(series: pd.Series, window: int) -> pd.Series:
    def _rank(arr: np.ndarray) -> float:
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.nan
        last = arr[-1]
        return float(np.mean(arr <= last))
    return series.rolling(window, min_periods=max(3, window // 2)).apply(_rank, raw=True)


def bucket_series(s: pd.Series, q: int, labels: Optional[List[str]] = None) -> pd.Series:
    x = safe_numeric(s)
    if labels is None:
        labels = [f"Q{i}" for i in range(1, q + 1)]
    try:
        return pd.qcut(x, q=q, labels=labels, duplicates="drop")
    except Exception:
        return pd.Series(index=s.index, dtype="object")


# =========================
# data io / cache
# =========================
def load_kline(ts_code: str, freq: str = "d", bars: int = 1000) -> Optional[pd.DataFrame]:
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
        print(f"  [WARN] {ts_code} 查询失败: {exc}")
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


def get_stock_pool(n: int = 100, seed: int = 42) -> List[str]:
    engine = get_engine()
    sql = "SELECT ts_code FROM stock_pools ORDER BY ts_code"
    try:
        df = pd.read_sql_query(text(sql), engine)
    finally:
        engine.dispose()
    if df.empty:
        raise ValueError("stock_pools表为空")
    codes = df["ts_code"].tolist()
    rng = np.random.default_rng(seed)
    return rng.choice(codes, size=min(n, len(codes)), replace=False).tolist()


def get_cache_path(cache_dir: str, n_stocks: int, bars: int, freq: str, seed: int) -> str:
    param_str = f"{n_stocks}_{bars}_{freq}_{seed}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    cache_file = f"klines_{freq}_{bars}_{n_stocks}stocks_{param_hash}.pkl"
    return os.path.join(cache_dir, cache_file)


def cache_exists(cache_path: str) -> bool:
    parquet_path = cache_path.replace('.pkl', '.parquet')
    return os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0


def save_cache(data: Dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    all_dfs = data.get("all_dfs", [])
    metadata = data.get("metadata", {})
    parquet_path = cache_path.replace('.pkl', '.parquet')
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_parquet(parquet_path, index=False, compression='zstd')
    meta_path = cache_path.replace('.pkl', '_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_cache(cache_path: str) -> Optional[Dict]:
    try:
        parquet_path = cache_path.replace('.pkl', '.parquet')
        meta_path = cache_path.replace('.pkl', '_meta.json')
        if not os.path.exists(parquet_path):
            return None
        combined_df = pd.read_parquet(parquet_path)
        all_dfs = []
        for symbol in combined_df['symbol'].unique():
            df = combined_df[combined_df['symbol'] == symbol].copy()
            df['datetime'] = pd.to_datetime(df['datetime'])
            all_dfs.append(df)
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        return {"all_dfs": all_dfs, "metadata": metadata}
    except Exception as exc:
        print(f"  [WARN] 缓存加载失败: {exc}")
        return None


# =========================
# factor engineering
# =========================
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
    out["w_sample_count"] = pd.Series(np.arange(len(mapped), dtype=float) + 1.0, index=mapped.index).reindex(prev_friday).to_numpy()
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


def add_volume_zscore_family(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vol = out["vol"].astype(float)
    for n in [5, 10, 20]:
        ma = vol.rolling(n, min_periods=max(3, n // 2)).mean()
        std = vol.rolling(n, min_periods=max(3, n // 2)).std().replace(0, np.nan)
        z = (vol - ma) / std
        out[f"vol_ma_{n}"] = ma
        out[f"vol_ratio_{n}"] = vol / ma.replace(0, np.nan)
        out[f"vol_z_{n}"] = z
        for lb in [5, 10, 20]:
            out[f"vol_z{n}_mean_{lb}"] = z.rolling(lb, min_periods=max(3, lb // 2)).mean()
            out[f"vol_z{n}_max_{lb}"] = z.rolling(lb, min_periods=max(3, lb // 2)).max()
            out[f"vol_z{n}_min_{lb}"] = z.rolling(lb, min_periods=max(3, lb // 2)).min()
        out[f"vol_z_{n}_delta1"] = z.diff(1)
        out[f"vol_z_{n}_delta3"] = z.diff(3)
        out[f"vol_z{n}_last_rank_5"] = rolling_rank_pct(z, 5)
        out[f"vol_z{n}_last_rank_10"] = rolling_rank_pct(z, 10)
        out[f"vol_z{n}_last_rank_20"] = rolling_rank_pct(z, 20)
    out["vol_z_5_minus_20"] = out["vol_z_5"] - out["vol_z_20"]
    out["vol_z_10_minus_20"] = out["vol_z_10"] - out["vol_z_20"]
    out["volume_accel"] = out["vol_ratio_20"] - safe_numeric(out.get("vol_ratio_20", pd.Series(index=out.index))).shift(1)
    return out


def add_short_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    o = out["open"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    c = out["close"].astype(float)
    rng = (h - l).replace(0, np.nan)
    body_high = pd.concat([o, c], axis=1).max(axis=1)
    body_low = pd.concat([o, c], axis=1).min(axis=1)
    out["body_pct"] = (c - o) / o.replace(0, np.nan)
    out["upper_shadow_pct"] = (h - body_high) / c.replace(0, np.nan)
    out["lower_shadow_pct"] = (body_low - l) / c.replace(0, np.nan)
    out["close_in_bar"] = ((c - l) / rng).clip(0.0, 1.0)
    out["range_pct"] = rng / c.replace(0, np.nan)
    out["gap_pct"] = o / c.shift(1).replace(0, np.nan) - 1.0
    out["ret_2_back"] = c.pct_change(2)
    out["close_vs_prev_high"] = c / h.shift(1).replace(0, np.nan) - 1.0
    out["close_vs_prev_low"] = c / l.shift(1).replace(0, np.nan) - 1.0
    out["close_vs_3d_high"] = c / h.shift(1).rolling(3, min_periods=2).max().replace(0, np.nan) - 1.0
    out["close_vs_3d_low"] = c / l.shift(1).rolling(3, min_periods=2).min().replace(0, np.nan) - 1.0
    return out


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = {
        "x_bbwidthpct_volz20": ("bb_width_percentile", "vol_z_20"),
        "x_bbwidthpct_volz5": ("bb_width_percentile", "vol_z_5"),
        "x_barssince_volz20": ("bars_since_dir_change", "vol_z_20"),
        "x_distrope_sloperope": ("dist_to_rope_atr", "rope_slope_atr_5"),
        "x_posstruct_breakstrength": ("position_in_structure", "range_break_up_strength"),
        "x_dsapos_volz20": ("dsa_trade_pos_01", "vol_z_20"),
        "x_runbars_volz20": ("current_run_bars", "vol_z_20"),
        "x_volz5_volz20": ("vol_z_5", "vol_z_20"),
        "x_volz10_bbwchg5": ("vol_z_10", "bb_width_change_5"),
        "x_gap_volz20": ("gap_pct", "vol_z_20"),
    }
    for out_col, (a, b) in pairs.items():
        if a in out.columns and b in out.columns:
            out[out_col] = safe_numeric(out[a]) * safe_numeric(out[b])
    return out


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

    volume_parts = pd.DataFrame(index=out.index)
    volume_parts["vol_ratio"] = normalize_01(safe_numeric(out.get("vol_ratio_20", pd.Series(index=out.index))), reverse=False)
    volume_parts["vol_z"] = normalize_01(safe_numeric(out.get("vol_z_20", pd.Series(index=out.index))), reverse=False)
    volume_parts["break_strength"] = normalize_01(safe_numeric(out.get("range_break_up_strength", pd.Series(index=out.index))), reverse=False)
    volume_parts["vol_z_5"] = normalize_01(safe_numeric(out.get("vol_z_5", pd.Series(index=out.index))), reverse=False)
    volume_parts["vol_z_5_minus_20"] = normalize_01(safe_numeric(out.get("vol_z_5_minus_20", pd.Series(index=out.index))), reverse=False)
    out["score_volume_total"] = volume_parts.mean(axis=1)

    width_parts = pd.DataFrame(index=out.index)
    width_parts["bbw_pct_low"] = normalize_01(safe_numeric(out.get("bb_width_percentile", pd.Series(index=out.index))), reverse=True)
    width_parts["range_atr_low"] = normalize_01(safe_numeric(out.get("range_width_atr", pd.Series(index=out.index))), reverse=True)
    width_parts["dsa_width_low"] = normalize_01(safe_numeric(out.get("dsa_trade_range_width_pct", pd.Series(index=out.index))), reverse=True)
    width_parts["contract"] = normalize_01(safe_numeric(out.get("bb_contract_streak", pd.Series(index=out.index))), reverse=False)
    out["score_width_total"] = width_parts.mean(axis=1)

    rope_parts = pd.DataFrame(index=out.index)
    rope_parts["near_rope"] = normalize_01(safe_numeric(out.get("dist_to_rope_atr", pd.Series(index=out.index))).abs(), reverse=True)
    rope_parts["rope_up"] = dir_flags["rope_up"]
    rope_parts["rope_slope"] = normalize_01(safe_numeric(out.get("rope_slope_atr_5", pd.Series(index=out.index))), reverse=False)
    out["score_rope_total"] = rope_parts.mean(axis=1)

    out["score_setup_total"] = pd.concat([
        safe_numeric(out["position_in_structure"]).pipe(lambda s: 1 - s.clip(0, 1)),
        normalize_01(out["dir_consistent"], reverse=False),
        safe_numeric(out["score_trend_total"]),
        safe_numeric(out["score_volume_total"]),
        safe_numeric(out["score_width_total"]),
        safe_numeric(out["score_rope_total"]),
    ], axis=1).mean(axis=1)
    return out


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "vol" in d.columns and "volume" not in d.columns:
        d["volume"] = d["vol"]
    d = add_micro_features(d)
    d = add_volume_zscore_family(d)
    d = add_short_candle_features(d)
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
    merged = pd.concat(
        [d, dsa_confirmed, dsa_trade, rope_df.drop(columns=d.columns, errors="ignore"), bb_df.drop(columns=d.columns, errors="ignore"), weekly_confirmed, weekly_trade],
        axis=1,
    )
    merged = add_composite_features(merged)
    merged = add_interaction_features(merged)
    return merged


def add_short_horizon_targets(df: pd.DataFrame, windows: Sequence[int] = RET_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].to_numpy(float)
    high = out["high"].to_numpy(float)
    low = out["low"].to_numpy(float)
    open_ = out["open"].to_numpy(float)
    n = len(out)
    for w in windows:
        ret = np.full(n, np.nan)
        max_up = np.full(n, np.nan)
        max_dd = np.full(n, np.nan)
        open_ret = np.full(n, np.nan)
        open_to_high = np.full(n, np.nan)
        for i in range(n - w):
            entry = close[i]
            ret[i] = (close[i + w] - entry) / entry
            future_high = np.nanmax(high[i + 1:i + w + 1])
            future_low = np.nanmin(low[i + 1:i + w + 1])
            max_up[i] = (future_high - entry) / entry
            dd_raw = (future_low - entry) / entry
            max_dd[i] = min(0.0, dd_raw)
            open_ret[i] = (open_[i + 1] - entry) / entry
            open_to_high[i] = (high[i + 1] - open_[i + 1]) / open_[i + 1] if open_[i + 1] != 0 else np.nan
        out[f"ret_{w}d"] = ret
        out[f"max_up_{w}d"] = max_up
        out[f"max_dd_{w}d"] = max_dd
        out[f"win_{w}d"] = (pd.Series(ret, index=out.index) > 0).astype(float)
        out[f"open_ret_{w}d"] = open_ret
        out[f"open_to_high_{w}d"] = open_to_high
        out[f"rr_{w}d"] = [rr_from_ret_dd(r, d) for r, d in zip(max_up, max_dd)]
    out["good1_2pct"] = (safe_numeric(out["ret_1d"]) >= 0.02).astype(float)
    out["good2_3pct"] = (safe_numeric(out["ret_2d"]) >= 0.03).astype(float)
    out["hit3_3pct"] = (safe_numeric(out["max_up_3d"]) >= 0.03).astype(float)
    out["good3_5pct"] = (safe_numeric(out["max_up_3d"]) >= 0.05).astype(float)
    out["safe3"] = (safe_numeric(out["max_dd_3d"]) > -0.03).astype(float)
    return out


# =========================
# feature groups
# =========================
TRADE_FEATURES = [
    "dsa_trade_pos_01", "dsa_trade_dist_to_low_01", "dsa_trade_dist_to_high_01", "dsa_trade_range_width_pct",
    "bb_pos_01", "bb_width_norm", "bb_width_percentile", "bb_width_change_5", "bb_expanding", "bb_contracting",
    "bb_expand_streak", "bb_contract_streak",
    "rope_dir", "dist_to_rope_atr", "rope_slope_atr_5", "bars_since_dir_change", "is_consolidating",
    "range_break_up", "range_break_up_strength", "range_width_atr", "channel_pos_01", "range_pos_01", "rope_pivot_pos_01",
    "dsa_trade_breakout_20", "dsa_trade_breakdown_20",
    "w_dsa_trade_pos_01", "w_dsa_trade_dist_to_low_01", "w_dsa_trade_dist_to_high_01", "w_dsa_trade_range_width_pct",
    "w_factor_available", "w_sample_count",
    "ret_1", "ret_3", "ret_5", "vol_ratio_20", "vol_z_20", "trend_gap_10_20",
    "position_in_structure", "dir_consistent", "score_trend_total", "score_volume_total", "score_width_total", "score_rope_total", "score_setup_total",
]

RESEARCH_FEATURES = [
    "dsa_confirmed_pivot_pos_01", "dsa_signed_vwap_dev_pct", "dsa_trend_aligned_vwap_dev_pct",
    "lh_hh_low_pos", "dsa_bull_vwap_dev_pct", "dsa_bear_vwap_dev_pct",
    "prev_confirmed_up_bars", "prev_confirmed_down_bars", "last_confirmed_run_bars", "current_run_bars",
    "w_DSA_DIR", "w_dsa_confirmed_pivot_pos_01", "w_dsa_signed_vwap_dev_pct", "w_prev_confirmed_up_bars", "w_prev_confirmed_down_bars",
]

VOLZ_FEATURES = [
    "vol_z_5", "vol_z_10", "vol_z_20",
    "vol_z5_mean_5", "vol_z10_mean_10", "vol_z20_mean_20",
    "vol_z5_max_5", "vol_z10_max_10", "vol_z20_max_20",
    "vol_z_5_delta1", "vol_z_10_delta1", "vol_z_20_delta1",
    "vol_z_5_delta3", "vol_z_10_delta3", "vol_z_20_delta3",
    "vol_z_5_minus_20", "vol_z_10_minus_20",
    "vol_z20_last_rank_20",
]

CANDLE_FEATURES = [
    "body_pct", "upper_shadow_pct", "lower_shadow_pct", "close_in_bar", "range_pct", "gap_pct",
    "ret_2_back", "close_vs_prev_high", "close_vs_prev_low", "close_vs_3d_high", "close_vs_3d_low",
]

INTERACTION_FEATURES = [
    "x_bbwidthpct_volz20", "x_bbwidthpct_volz5", "x_barssince_volz20", "x_distrope_sloperope",
    "x_posstruct_breakstrength", "x_dsapos_volz20", "x_runbars_volz20", "x_volz5_volz20",
    "x_volz10_bbwchg5", "x_gap_volz20",
]

PRIMARY_HINT_FEATURES = [
    "position_in_structure", "dir_consistent", "range_width_atr", "score_trend_total", "score_volume_total",
    "score_width_total", "score_rope_total", "score_setup_total", "dsa_trade_pos_01", "bb_width_percentile",
    "bars_since_dir_change", "prev_confirmed_down_bars", "current_run_bars", "dist_to_rope_atr",
    "vol_z_5", "vol_z_10", "vol_z_20", "vol_z_5_minus_20", "close_in_bar", "gap_pct",
]

ALL_FEATURES = list(dict.fromkeys(TRADE_FEATURES + RESEARCH_FEATURES + VOLZ_FEATURES + CANDLE_FEATURES + INTERACTION_FEATURES))


# =========================
# dataset build
# =========================
def fetch_raw_klines_from_db(n_stocks: int, bars: int, freq: str, seed: int) -> List[pd.DataFrame]:
    """从数据库获取原始行情数据（不含因子计算）"""
    stocks = get_stock_pool(n_stocks, seed)
    print(f"股票池抽取: {len(stocks)}只 (seed={seed})")
    all_dfs: List[pd.DataFrame] = []
    for idx, code in enumerate(stocks):
        kline = load_kline(code, freq, bars)
        if kline is None or len(kline) < 100:
            continue
        kline["symbol"] = code
        kline["datetime"] = kline.index
        all_dfs.append(kline.reset_index(drop=True))
        if (idx + 1) % 100 == 0 or idx == len(stocks) - 1:
            print(f"  [{idx + 1}/{len(stocks)}] {code}: {len(kline)}bars")
    return all_dfs


def process_kline_to_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """将原始行情数据转换为因子面板，兼容已计算因子的缓存格式"""
    try:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "unknown"
        
        # 检查是否已经是因子计算后的缓存（gbdt_buy_point_explorer格式）
        has_existing_factors = "dsa_trade_pos_01" in df.columns and "ret_20" in df.columns
        
        if has_existing_factors:
            # 已有因子，只需添加 short_buy_explorer 特有的目标标签
            fac = add_short_horizon_targets(df, RET_WINDOWS)
        else:
            # 原始行情数据，需要完整计算因子
            fac = compute_factors(df)
            fac = add_short_horizon_targets(fac, RET_WINDOWS)
        
        fac["symbol"] = symbol
        fac["datetime"] = fac.index
        valid = fac.dropna(subset=["ret_3d"])
        if len(valid) >= MIN_VALID_ROWS_PER_STOCK:
            return valid.reset_index(drop=True)
        return None
    except Exception as exc:
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "unknown"
        print(f"  [WARN] {symbol} 因子计算失败: {exc}")
        return None


def build_research_panel(args: argparse.Namespace) -> pd.DataFrame:
    cache_path = get_cache_path(args.cache_dir, args.n_stocks, args.bars, args.freq, args.seed)
    used_cache = False
    raw_dfs: List[pd.DataFrame] = []
    
    if args.use_cache and cache_exists(cache_path):
        print(f"[CACHE] 检测到缓存: {cache_path.replace('.pkl', '.parquet')}")
        cached = load_cache(cache_path)
        if cached and cached.get("all_dfs"):
            raw_dfs = cached["all_dfs"]
            used_cache = True
            print(f"[CACHE] 缓存命中，加载原始行情数据 {len(raw_dfs)} 只")
        else:
            print(f"[CACHE] 缓存存在但加载失败，回退到数据库拉取")
            raw_dfs = fetch_raw_klines_from_db(args.n_stocks, args.bars, args.freq, args.seed)
    else:
        if args.use_cache:
            print(f"[CACHE] 未找到缓存，开始从数据库拉取行情: {cache_path.replace('.pkl', '.parquet')}")
        raw_dfs = fetch_raw_klines_from_db(args.n_stocks, args.bars, args.freq, args.seed)
    
    if args.use_cache and not used_cache and raw_dfs:
        save_cache({
            "all_dfs": raw_dfs,
            "metadata": {
                "n_stocks": args.n_stocks,
                "bars": args.bars,
                "freq": args.freq,
                "seed": args.seed,
                "rows": int(sum(len(x) for x in raw_dfs)),
            },
        }, cache_path)
        print(f"[CACHE] 已写入原始行情缓存: {cache_path.replace('.pkl', '.parquet')}")
    
    if not raw_dfs:
        return pd.DataFrame()
    
    # 从原始行情计算因子
    print(f"[FACTOR] 开始计算因子，共 {len(raw_dfs)} 只股票...")
    all_dfs: List[pd.DataFrame] = []
    for idx, raw_df in enumerate(raw_dfs):
        result = process_kline_to_features(raw_df)
        if result is not None:
            all_dfs.append(result)
        if (idx + 1) % 100 == 0 or idx == len(raw_dfs) - 1:
            print(f"  因子计算进度: {idx + 1}/{len(raw_dfs)}")
    
    if not all_dfs:
        return pd.DataFrame()
    
    panel = pd.concat(all_dfs, ignore_index=True)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel["year"] = panel["datetime"].dt.year
    panel["year_month"] = panel["datetime"].dt.to_period("M").astype(str)
    panel = panel.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    return panel


# =========================
# analysis
# =========================
def analyze_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = [{
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()) if "symbol" in df.columns else 0,
        "start_date": pd.to_datetime(df["datetime"]).min(),
        "end_date": pd.to_datetime(df["datetime"]).max(),
        "ret_1d_na": round(float(df["ret_1d"].isna().mean()), 6),
        "ret_2d_na": round(float(df["ret_2d"].isna().mean()), 6),
        "ret_3d_na": round(float(df["ret_3d"].isna().mean()), 6),
        "good3_5pct_rate": round(float(df["good3_5pct"].mean()), 6),
        "hit3_3pct_rate": round(float(df["hit3_3pct"].mean()), 6),
    }]
    return pd.DataFrame(rows)


def analyze_feature_inventory(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feat in features:
        if feat not in df.columns:
            continue
        s = safe_numeric(df[feat])
        rows.append({
            "feature": feat,
            "dtype": str(df[feat].dtype),
            "na_ratio": round(float(s.isna().mean()), 6),
            "nunique": int(s.nunique(dropna=True)),
            "mean": round(float(s.mean()), 6) if s.notna().any() else np.nan,
            "std": round(float(s.std()), 6) if s.notna().any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["na_ratio", "feature"]).reset_index(drop=True)


def build_single_factor_buckets(df: pd.DataFrame, features: Sequence[str], q: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_ret: List[Dict[str, object]] = []
    rows_trade: List[Dict[str, object]] = []
    labels = [f"Q{i}" for i in range(1, q + 1)]
    for feat in features:
        if feat not in df.columns:
            continue
        sub = df[[feat, "ret_1d", "ret_2d", "ret_3d", "max_up_1d", "max_up_2d", "max_up_3d", "max_dd_1d", "max_dd_2d", "max_dd_3d", "good3_5pct", "hit3_3pct"]].copy()
        sub[feat] = safe_numeric(sub[feat])
        sub = sub.dropna(subset=[feat, "ret_3d", "max_up_3d", "max_dd_3d"])
        if len(sub) < MIN_ROWS_FOR_BUCKET or sub[feat].nunique() < min(5, q):
            continue
        sub["bucket"] = bucket_series(sub[feat], q=q, labels=labels)
        sub = sub.dropna(subset=["bucket"])
        if sub.empty:
            continue
        grp = sub.groupby("bucket", observed=True).agg(
            n=(feat, "count"),
            mean_value=(feat, "mean"),
            ret_1d=("ret_1d", "mean"), ret_2d=("ret_2d", "mean"), ret_3d=("ret_3d", "mean"),
            med_ret_1d=("ret_1d", "median"), med_ret_2d=("ret_2d", "median"), med_ret_3d=("ret_3d", "median"),
            max_up_1d=("max_up_1d", "mean"), max_up_2d=("max_up_2d", "mean"), max_up_3d=("max_up_3d", "mean"),
            max_dd_1d=("max_dd_1d", "mean"), max_dd_2d=("max_dd_2d", "mean"), max_dd_3d=("max_dd_3d", "mean"),
            hit3_3pct=("hit3_3pct", "mean"), good3_5pct=("good3_5pct", "mean"),
        ).reset_index()
        best_ret3_bucket = grp.sort_values(["ret_3d", "good3_5pct"], ascending=[False, False]).iloc[0]["bucket"]
        best_up3_bucket = grp.sort_values(["max_up_3d", "good3_5pct"], ascending=[False, False]).iloc[0]["bucket"]
        for _, row in grp.iterrows():
            rows_ret.append({
                "feature": feat, "bucket": row["bucket"], "n": int(row["n"]),
                "mean_value": round(float(row["mean_value"]), 6),
                "ret_1d": round(float(row["ret_1d"]), 6), "ret_2d": round(float(row["ret_2d"]), 6), "ret_3d": round(float(row["ret_3d"]), 6),
                "med_ret_1d": round(float(row["med_ret_1d"]), 6), "med_ret_2d": round(float(row["med_ret_2d"]), 6), "med_ret_3d": round(float(row["med_ret_3d"]), 6),
                "is_best_ret3_bucket": int(row["bucket"] == best_ret3_bucket),
            })
            rows_trade.append({
                "feature": feat, "bucket": row["bucket"], "n": int(row["n"]),
                "mean_value": round(float(row["mean_value"]), 6),
                "max_up_1d": round(float(row["max_up_1d"]), 6), "max_up_2d": round(float(row["max_up_2d"]), 6), "max_up_3d": round(float(row["max_up_3d"]), 6),
                "max_dd_1d": round(float(row["max_dd_1d"]), 6), "max_dd_2d": round(float(row["max_dd_2d"]), 6), "max_dd_3d": round(float(row["max_dd_3d"]), 6),
                "hit3_3pct": round(float(row["hit3_3pct"]), 6), "good3_5pct": round(float(row["good3_5pct"]), 6),
                "rr_3d": round(rr_from_ret_dd(float(row["max_up_3d"]), float(row["max_dd_3d"])), 6) if pd.notna(row["max_up_3d"]) and pd.notna(row["max_dd_3d"]) and float(row["max_dd_3d"]) != 0 else np.nan,
                "is_best_up3_bucket": int(row["bucket"] == best_up3_bucket),
            })
    return pd.DataFrame(rows_ret), pd.DataFrame(rows_trade)



def build_single_factor_correlations(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    targets = ["ret_1d", "ret_2d", "ret_3d", "max_up_1d", "max_up_2d", "max_up_3d", "max_dd_1d", "max_dd_2d", "max_dd_3d", "good3_5pct", "hit3_3pct"]
    for feat in features:
        if feat not in df.columns:
            continue
        s = safe_numeric(df[feat])
        if s.notna().sum() < MIN_ROWS_FOR_BUCKET or s.nunique(dropna=True) < 5:
            continue
        for target in targets:
            if target not in df.columns:
                continue
            sub = pd.DataFrame({"x": s, "y": safe_numeric(df[target])}).dropna()
            if len(sub) < MIN_ROWS_FOR_BUCKET:
                continue
            pearson = sub["x"].corr(sub["y"], method="pearson")
            spearman = sub["x"].corr(sub["y"], method="spearman")
            rows.append({
                "feature": feat,
                "target": target,
                "n": int(len(sub)),
                "pearson_corr": round(float(pearson), 6) if pd.notna(pearson) else np.nan,
                "spearman_corr": round(float(spearman), 6) if pd.notna(spearman) else np.nan,
                "abs_pearson": round(abs(float(pearson)), 6) if pd.notna(pearson) else np.nan,
                "abs_spearman": round(abs(float(spearman)), 6) if pd.notna(spearman) else np.nan,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["target", "abs_spearman", "abs_pearson", "n"], ascending=[True, False, False, False]).reset_index(drop=True)


def select_top_features(bucket_trade: pd.DataFrame, corr_df: pd.DataFrame, top_k: int) -> List[str]:
    feats: List[str] = []
    if not bucket_trade.empty:
        tmp = bucket_trade.groupby("feature", as_index=False).agg(
            max_up_3d=("max_up_3d", "max"),
            good3_5pct=("good3_5pct", "max"),
            rr_3d=("rr_3d", "max"),
            n=("n", "sum"),
        )
        tmp["score"] = tmp[["max_up_3d", "good3_5pct", "rr_3d"]].fillna(0).mean(axis=1)
        feats.extend(tmp.sort_values(["score", "n"], ascending=[False, False])["feature"].head(top_k).tolist())
    if not corr_df.empty:
        corr_tmp = corr_df[corr_df["target"].isin(["ret_3d", "max_up_3d", "good3_5pct"])].copy()
        if not corr_tmp.empty:
            corr_tmp["score"] = corr_tmp[["abs_spearman", "abs_pearson"]].fillna(0).mean(axis=1)
            feats.extend(
                corr_tmp.groupby("feature", as_index=False)["score"].mean()
                .sort_values("score", ascending=False)["feature"].head(top_k).tolist()
            )
    feats.extend([f for f in PRIMARY_HINT_FEATURES if f in ALL_FEATURES])
    return list(dict.fromkeys([f for f in feats if f]))[:max(top_k, len(PRIMARY_HINT_FEATURES))]


def build_pair_heatmap_summary(df: pd.DataFrame, top_features: Sequence[str], out_target: str = "max_up_3d", q: int = PAIR_BIN_Q) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    feats = [f for f in top_features if f in df.columns]
    for a, b in combinations(feats, 2):
        sub = df[[a, b, out_target, "good3_5pct"]].copy()
        sub[a] = safe_numeric(sub[a])
        sub[b] = safe_numeric(sub[b])
        sub = sub.dropna(subset=[a, b, out_target])
        if len(sub) < 150 or sub[a].nunique() < q or sub[b].nunique() < q:
            continue
        sub["a_bin"] = bucket_series(sub[a], q=q)
        sub["b_bin"] = bucket_series(sub[b], q=q)
        sub = sub.dropna(subset=["a_bin", "b_bin"])
        if sub.empty:
            continue
        grp = sub.groupby(["a_bin", "b_bin"], observed=True).agg(
            n=(out_target, "count"),
            value=(out_target, "mean"),
            hit=("good3_5pct", "mean"),
        ).reset_index()
        grp = grp[grp["n"] >= max(5, len(sub) * 0.005)]
        if grp.empty:
            continue
        best = grp.sort_values(["value", "hit", "n"], ascending=[False, False, False]).iloc[0]
        rows.append({
            "feature_a": a,
            "feature_b": b,
            "target": out_target,
            "best_a_bin": best["a_bin"],
            "best_b_bin": best["b_bin"],
            "best_value": round(float(best["value"]), 6),
            "best_hit": round(float(best["hit"]), 6),
            "best_n": int(best["n"]),
            "global_n": int(len(sub)),
        })
    return pd.DataFrame(rows).sort_values(["best_value", "best_hit", "best_n"], ascending=[False, False, False]).reset_index(drop=True) if rows else pd.DataFrame()


def build_rule_layering_summary(df: pd.DataFrame, pair_df: pd.DataFrame, target: str = "max_up_3d") -> pd.DataFrame:
    if df.empty or pair_df.empty:
        return pd.DataFrame()
    base_target = safe_numeric(df[target]).mean() if target in df.columns else np.nan
    base_hit = safe_numeric(df["good3_5pct"]).mean() if "good3_5pct" in df.columns else np.nan
    rows: List[Dict[str, object]] = []
    for _, row in pair_df.iterrows():
        rows.append({
            "rule_name": f"{row['feature_a']} in {row['best_a_bin']} AND {row['feature_b']} in {row['best_b_bin']}",
            "feature_a": row["feature_a"],
            "feature_b": row["feature_b"],
            "best_a_bin": row["best_a_bin"],
            "best_b_bin": row["best_b_bin"],
            "target": row["target"],
            "global_value": round(float(base_target), 6) if pd.notna(base_target) else np.nan,
            "global_hit": round(float(base_hit), 6) if pd.notna(base_hit) else np.nan,
            "layer_value": row.get("best_value", np.nan),
            "layer_hit": row.get("best_hit", np.nan),
            "value_uplift": round(float(row.get("best_value", np.nan) - base_target), 6) if pd.notna(base_target) and pd.notna(row.get("best_value", np.nan)) else np.nan,
            "hit_uplift": round(float(row.get("best_hit", np.nan) - base_hit), 6) if pd.notna(base_hit) and pd.notna(row.get("best_hit", np.nan)) else np.nan,
            "layer_n": int(row.get("best_n", 0)),
            "global_n": int(row.get("global_n", 0)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["value_uplift", "hit_uplift", "layer_n"], ascending=[False, False, False]).reset_index(drop=True)


def dedup_by_symbol_distance(df: pd.DataFrame, min_gap: int) -> pd.DataFrame:
    if df.empty or "symbol" not in df.columns:
        return df.copy()
    work = df.sort_values(["symbol", "datetime"]).copy()
    work["keep"] = False
    for _, g in work.groupby("symbol", sort=False):
        last_pos: Optional[int] = None
        for pos, idx in enumerate(g.index.tolist()):
            if last_pos is None or pos - last_pos >= min_gap:
                work.loc[idx, "keep"] = True
                last_pos = pos
    return work[work["keep"]].drop(columns=["keep"]).reset_index(drop=True)


def summarize_single_feature_effect(df: pd.DataFrame, feature: str, target: str, q: int = 5) -> Optional[Dict[str, object]]:
    if feature not in df.columns or target not in df.columns:
        return None
    sub = df[[feature, target, "good3_5pct"]].copy()
    sub[feature] = safe_numeric(sub[feature])
    sub = sub.dropna(subset=[feature, target])
    if len(sub) < 80 or sub[feature].nunique() < q:
        return None
    sub["bucket"] = bucket_series(sub[feature], q=q)
    sub = sub.dropna(subset=["bucket"])
    grp = sub.groupby("bucket", observed=True).agg(v=(target, "mean"), hit=("good3_5pct", "mean"), n=(feature, "count")).reset_index()
    if grp.empty:
        return None
    top = grp.sort_values(["v", "hit", "n"], ascending=[False, False, False]).iloc[0]
    return {
        "feature": feature,
        "target": target,
        "best_bucket": top["bucket"],
        "best_value": round(float(top["v"]), 6),
        "best_hit": round(float(top["hit"]), 6),
        "sample_n": int(len(sub)),
    }


def build_stability_by_time(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for year, sub in df.groupby("year"):
        for feat in features:
            item = summarize_single_feature_effect(sub, feat, "max_up_3d")
            if item is not None:
                item["slice_type"] = "year"
                item["slice_value"] = year
                rows.append(item)
    return pd.DataFrame(rows)


def build_stability_by_universe(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["size_proxy"] = safe_numeric(tmp["close"]) * safe_numeric(tmp["vol"])
    tmp["vol_proxy"] = safe_numeric(tmp["range_pct"])
    rows: List[Dict[str, object]] = []
    tmp["size_group"] = pd.qcut(tmp["size_proxy"], q=2, labels=["small_liq", "large_liq"], duplicates="drop")
    tmp["vol_group"] = pd.qcut(tmp["vol_proxy"], q=2, labels=["low_vol", "high_vol"], duplicates="drop")
    for col in ["size_group", "vol_group"]:
        for grp, sub in tmp.groupby(col, observed=True):
            for feat in features:
                item = summarize_single_feature_effect(sub, feat, "max_up_3d")
                if item is not None:
                    item["slice_type"] = col
                    item["slice_value"] = str(grp)
                    rows.append(item)
    return pd.DataFrame(rows)


def build_stability_by_dedup(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for gap in [1, 3, 5]:
        sub = dedup_by_symbol_distance(df, gap) if gap > 1 else df.copy()
        for feat in features:
            item = summarize_single_feature_effect(sub, feat, "max_up_3d")
            if item is not None:
                item["slice_type"] = "dedup_gap"
                item["slice_value"] = gap
                rows.append(item)
    return pd.DataFrame(rows)


def build_hypothesis_shortlist(bucket_ret: pd.DataFrame, bucket_trade: pd.DataFrame, pair_df: pd.DataFrame, corr_df: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not bucket_trade.empty:
        best_trade = bucket_trade[bucket_trade["is_best_up3_bucket"] == 1].copy()
        for _, row in best_trade.iterrows():
            rows.append({
                "type": "single_factor_bucket",
                "name": row["feature"],
                "best_zone": row["bucket"],
                "score": round(float(np.nanmean([row.get("max_up_3d", np.nan), row.get("good3_5pct", np.nan), row.get("rr_3d", np.nan)])), 6),
                "value_1": row.get("max_up_3d", np.nan),
                "value_2": row.get("good3_5pct", np.nan),
                "sample_n": int(row.get("n", 0)),
            })
    if not pair_df.empty:
        for _, row in pair_df.head(top_n).iterrows():
            rows.append({
                "type": "pair_layer",
                "name": f"{row['feature_a']} × {row['feature_b']}",
                "best_zone": f"{row['best_a_bin']} × {row['best_b_bin']}",
                "score": round(float(np.nanmean([row.get("best_value", np.nan), row.get("best_hit", np.nan)])), 6),
                "value_1": row.get("best_value", np.nan),
                "value_2": row.get("best_hit", np.nan),
                "sample_n": int(row.get("best_n", 0)),
            })
    if not corr_df.empty:
        top_corr = corr_df[corr_df["target"].isin(["ret_3d", "max_up_3d", "good3_5pct"])].copy()
        if not top_corr.empty:
            top_corr["score"] = top_corr[["abs_spearman", "abs_pearson"]].fillna(0).mean(axis=1)
            top_corr = top_corr.sort_values(["score", "n"], ascending=[False, False]).head(top_n)
            for _, row in top_corr.iterrows():
                rows.append({
                    "type": "single_factor_corr",
                    "name": f"{row['feature']} -> {row['target']}",
                    "best_zone": "corr_top",
                    "score": round(float(row["score"]), 6),
                    "value_1": row.get("spearman_corr", np.nan),
                    "value_2": row.get("pearson_corr", np.nan),
                    "sample_n": int(row.get("n", 0)),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("score", ascending=False).drop_duplicates(subset=["type", "name"]).head(top_n).reset_index(drop=True)


# =========================
# main
# =========================
def get_default_cache_dir() -> str:
    """获取默认缓存目录，使用独立子目录避免与其他脚本冲突"""
    return os.path.join("data_cache", "short_buy_explorer")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="超短因子关系探索器 short_buy_explorer")
    p.add_argument("--n-stocks", type=int, default=200, help="抽取股票数量")
    p.add_argument("--bars", type=int, default=1000, help="每只股票读取最近多少根K线")
    p.add_argument("--freq", type=str, default="d", help="频率，如 d")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--bucket-q", type=int, default=DEFAULT_BUCKET_Q, help="单因子分桶数量")
    p.add_argument("--pair-top-k", type=int, default=15, help="双因子热图使用的候选因子数")
    p.add_argument("--sample-export-rows", type=int, default=2000, help="研究主表样例导出行数")
    p.add_argument("--cache-dir", type=str, default=get_default_cache_dir(), help="缓存目录（默认：data_cache/short_buy_explorer）")
    p.add_argument("--output-dir", type=str, default=OUT_DIR, help="输出目录")
    p.add_argument("--use-cache", action="store_true", default=True, help="启用缓存（默认开启）")
    p.add_argument("--no-cache", dest="use_cache", action="store_false", help="禁用缓存")
    return p


def export_results_to_excel(output_path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    """将全部输出写入单个 Excel，多 sheet，禁止输出多个 csv。"""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = str(sheet_name)[:31] if sheet_name else "Sheet1"
            if df is None or df.empty:
                pd.DataFrame({"message": ["empty"]}).to_excel(writer, sheet_name=safe_name, index=False)
                continue
            out_df = df.copy()
            # 处理重复列名
            if len(out_df.columns) != len(set(out_df.columns)):
                cols = pd.Series(out_df.columns)
                for dup in cols[cols.duplicated()].unique():
                    dup_mask = cols == dup
                    cols.loc[dup_mask] = [f"{dup}_{i}" if i > 0 else dup for i in range(dup_mask.sum())]
                out_df.columns = cols
            if "datetime" in out_df.columns:
                try:
                    out_df["datetime"] = pd.to_datetime(out_df["datetime"]).dt.tz_localize(None)
                except Exception:
                    pass
            out_df.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.sheets[safe_name]
            ws.freeze_panes = "A2"
            for idx, col in enumerate(out_df.columns, 1):
                try:
                    col_data = out_df.iloc[:200, idx-1]
                    col_values = col_data.tolist()
                    max_len = max(len(str(col)), max([len(str(v)) for v in col_values], default=0))
                    col_letter = chr(64 + idx) if idx <= 26 else ws.cell(row=1, column=idx).column_letter
                    ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 28)
                except Exception:
                    pass


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    panel = build_research_panel(args)
    if panel.empty:
        print("[WARN] 未构建出有效研究样本")
        return

    dataset_summary = analyze_dataset_summary(panel)
    feature_inventory = analyze_feature_inventory(panel, ALL_FEATURES)

    sample_cols = [c for c in ["symbol", "datetime", "open", "high", "low", "close", *PRIMARY_HINT_FEATURES, *VOLZ_FEATURES[:10], "ret_1d", "ret_2d", "ret_3d", "max_up_3d", "max_dd_3d", "good3_5pct"] if c in panel.columns]
    panel_sample = panel[sample_cols].head(args.sample_export_rows).copy()

    bucket_ret, bucket_trade = build_single_factor_buckets(panel, ALL_FEATURES, q=args.bucket_q)
    corr_df = build_single_factor_correlations(panel, ALL_FEATURES)

    top_features = select_top_features(bucket_trade, corr_df, top_k=args.pair_top_k)
    pair_df = build_pair_heatmap_summary(panel, top_features, out_target="max_up_3d", q=PAIR_BIN_Q)
    rule_layering = build_rule_layering_summary(panel, pair_df, target="max_up_3d")

    stability_feats = list(dict.fromkeys(top_features + [f for f in VOLZ_FEATURES if f in panel.columns][:6]))
    stability_time = build_stability_by_time(panel, stability_feats)
    stability_universe = build_stability_by_universe(panel, stability_feats)
    stability_dedup = build_stability_by_dedup(panel, stability_feats)
    hypotheses = build_hypothesis_shortlist(bucket_ret, bucket_trade, pair_df, corr_df, top_n=50)

    excel_path = os.path.join(out_dir, "short_buy_explorer_results.xlsx")
    export_results_to_excel(excel_path, {
        "00_dataset_summary": dataset_summary,
        "01_feature_inventory": feature_inventory,
        "02_research_panel_sample": panel_sample,
        "03_single_factor_bucket_ret": bucket_ret,
        "04_single_factor_bucket_trade": bucket_trade,
        "05_pair_heatmap_summary": pair_df,
        "06_single_factor_correlation": corr_df,
        "07_rule_layering_summary": rule_layering,
        "08_stability_by_time": stability_time,
        "09_stability_by_universe": stability_universe,
        "10_stability_by_dedup": stability_dedup,
        "11_short_horizon_hypotheses": hypotheses,
    })

    print(f"研究面板已生成: {len(panel)} 行, {panel['symbol'].nunique()} 只股票")
    print(f"Excel输出: {excel_path}")


if __name__ == "__main__":
    main()
