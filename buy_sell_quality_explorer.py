# -*- coding: utf-8 -*-
"""
榜首规则精修版：基于候选事件做小范围规则精修的实验脚本

Purpose
- 保留原有框架：数据加载 → 因子计算 → forward returns → 候选事件池 → 报表输出
- 轻量化默认配置：去掉 permutation importance，默认只跑单模型，缩小规则扫描组合
- 聚焦 20 日目标，优先验证 周线环境过滤 / 低位压缩 / 启动别太晚 / expand vs break 分层

How to Run
    python compression_launch_rule_explorer.py --n-stocks 200 --bars 1000 --freq d

Outputs
- 00_dataset_summary.csv
- 01_feature_inventory.csv
- 02_candidate_events.csv
- 03_trade_reg_importance.csv
- 03b_trade_clf_importance.csv
- 04_research_reg_importance.csv
- 04b_research_clf_importance.csv
- 05_fold_metrics.csv
- 06_trade_bucket_profiles.csv
- 06b_research_bucket_profiles.csv
- 07_parameter_hints.csv
- 08_composite_feature_profiles.csv
- 09_rule_reverse_scan.csv
- 09b_rule_reverse_top_events.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

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

OUT_DIR = "compression_launch_rule_explorer_output"
os.makedirs(OUT_DIR, exist_ok=True)

RET_WINDOWS = [5, 10, 20, 40, 60]  # 仅作离线评分/观察，不作为买卖触发依据
PATH_QUALITY_HORIZON = 10
EVENT_EXIT_MAX_BARS = 40  # 仅作事件实验的观察上限，不作为持有期卖出规则
EVENT_DEDUP_BARS = 20
RANDOM_STATE = 42
EXECUTION_LAG = 1  # 事件识别后，默认下一根bar开盘作为真实买入点（用于统计起点，不作为研究对象）
ENTRY_PRICE_MODE = "next_open"

ACTIVE_TRIGGER_TYPES = ["none", "expand_only", "expand_and_break"]
ACTIVE_SCAN_TRIGGER_TYPES = ["expand_only", "expand_and_break"]
DROPPED_DIRECTIONS = [
    "break_only：多轮实验已反复表现偏弱，不再进入主扫描/主比较。",
    "expand_or_break：信息混杂，解释性差，停止作为主线触发类型。",
    "反复验证 break_only 是否不适合主线：视为已验证结论，不再重复消耗算力。",
]


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


def summarize_events(events: pd.DataFrame, windows: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {"n": int(len(events))}
    if len(events) == 0:
        for w in windows:
            out[f"ret_{w}"] = np.nan
            out[f"mae_{w}"] = np.nan
            out[f"wr_{w}"] = np.nan
            out[f"rr_{w}"] = np.nan
        return out
    for w in windows:
        ret_col = f"ret_{w}"
        dd_col = f"max_dd_{w}"
        win_col = f"win_{w}"
        ret = events[ret_col].mean() if ret_col in events.columns else np.nan
        dd = events[dd_col].mean() if dd_col in events.columns else np.nan
        wr = events[win_col].mean() if win_col in events.columns else np.nan
        out[f"ret_{w}"] = round(float(ret), 5) if pd.notna(ret) else np.nan
        out[f"mae_{w}"] = round(float(dd), 5) if pd.notna(dd) else np.nan
        out[f"wr_{w}"] = round(float(wr), 4) if pd.notna(wr) else np.nan
        out[f"rr_{w}"] = round(rr_from_ret_dd(ret, dd), 3) if pd.notna(ret) and pd.notna(dd) else np.nan
    return out


def dedup_events(df: pd.DataFrame, cooldown_bars: int = EVENT_DEDUP_BARS) -> pd.DataFrame:
    if df.empty or "symbol" not in df.columns:
        return df.copy()
    work = df.sort_values(["symbol", "datetime"]).copy()
    keep_idx: List[int] = []
    for _, g in work.groupby("symbol", sort=False):
        accepted_positions: List[int] = []
        for pos, idx in enumerate(g.index.tolist()):
            if not accepted_positions or pos - accepted_positions[-1] >= cooldown_bars:
                accepted_positions.append(pos)
                keep_idx.append(idx)
    return work.loc[keep_idx].sort_values(["datetime", "symbol"]).reset_index(drop=True)


# =========================
# Data loaders
# =========================
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
    rng = np.random.default_rng(seed)
    return rng.choice(codes, size=min(n, len(codes)), replace=False).tolist()


# =========================
# Data caching
# =========================
def get_cache_path(cache_dir: str, n_stocks: int, bars: int, freq: str, seed: int) -> str:
    """生成缓存文件路径，基于参数生成唯一文件名"""
    param_str = f"{n_stocks}_{bars}_{freq}_{seed}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    cache_file = f"klines_{freq}_{bars}_{n_stocks}stocks_{param_hash}.pkl"
    return os.path.join(cache_dir, cache_file)


def cache_exists(cache_path: str) -> bool:
    """检查缓存文件是否存在且有效（检查parquet文件）"""
    parquet_path = cache_path.replace('.pkl', '.parquet')
    return os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0


def save_cache(data: Dict, cache_path: str) -> None:
    """保存数据到缓存文件（使用parquet格式，更稳定）"""
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
    """从缓存文件加载数据（使用parquet格式）"""
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


def fetch_data_from_db(n_stocks: int, bars: int, freq: str, seed: int) -> List[pd.DataFrame]:
    """从数据库获取所有股票数据"""
    stocks = get_stock_pool(n_stocks, seed)
    print(f"股票池抽取: {len(stocks)}只 (seed={seed})")

    all_dfs: List[pd.DataFrame] = []
    for idx, code in enumerate(stocks):
        kline = load_kline(code, freq, bars)
        if kline is None or len(kline) < 100:
            continue
        try:
            fac = compute_factors(kline)
            fac = add_forward_returns(fac, RET_WINDOWS)
            fac = add_path_quality_labels(fac, horizon=PATH_QUALITY_HORIZON)
            fac["symbol"] = code
            fac["datetime"] = fac.index
            valid = fac.dropna(subset=["quality_trade_score"])
            if len(valid) > 50:
                all_dfs.append(valid)
                print(f"  [{idx + 1}/{len(stocks)}] {code}: {len(kline)}bars → {len(valid)}有效样本")
        except Exception as exc:
            print(f"  [{idx + 1}/{len(stocks)}] {code} 因子计算失败: {exc}")

    return all_dfs


# =========================
# Factor computation
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
    return merged


def add_forward_returns(
    df: pd.DataFrame,
    windows: Sequence[int] = RET_WINDOWS,
    entry_lag: int = EXECUTION_LAG,
    entry_price_mode: str = ENTRY_PRICE_MODE,
) -> pd.DataFrame:
    """
    固定窗口观察列：从真实买入点 t+k 开始统计。
    注意：这些列仅用于观察/评分，不作为入池条件和买卖依据。
    """
    out = df.copy()
    open_ = out["open"].to_numpy(float)
    close = out["close"].to_numpy(float)
    low = out["low"].to_numpy(float)
    n = len(out)
    for w in windows:
        fut = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        for i in range(n):
            entry_idx = i + entry_lag
            exit_idx = entry_idx + w
            if exit_idx >= n or entry_idx >= n:
                continue
            entry = open_[entry_idx] if entry_price_mode == "next_open" else close[entry_idx]
            if not np.isfinite(entry) or entry <= 0:
                continue
            fut[i] = (close[exit_idx] - entry) / entry
            future_low = np.nanmin(low[entry_idx + 1: exit_idx + 1]) if exit_idx > entry_idx else np.nan
            mae_raw = (future_low - entry) / entry if np.isfinite(future_low) else np.nan
            mae[i] = min(0.0, mae_raw) if pd.notna(mae_raw) else np.nan
        out[f"ret_{w}"] = fut
        out[f"max_dd_{w}"] = mae
        out[f"win_{w}"] = (out[f"ret_{w}"] > 0).astype(float)
    return out


def add_path_quality_labels(
    df: pd.DataFrame,
    horizon: int = PATH_QUALITY_HORIZON,
    stop_mode: str = "event_low",
    tp_levels: Sequence[float] = (0.03, 0.05, 0.08),
    entry_lag: int = EXECUTION_LAG,
    entry_price_mode: str = ENTRY_PRICE_MODE,
) -> pd.DataFrame:
    """
    路径质量标签从真实买入点 t+k 开始统计。
    k 只是执行时序定义，不是研究对象。
    """
    out = df.copy()
    open_ = out["open"].to_numpy(float)
    close = out["close"].to_numpy(float)
    high = out["high"].to_numpy(float)
    low = out["low"].to_numpy(float)
    n = len(out)

    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    bars_to_stop = np.full(n, np.nan)
    stop_first = np.zeros(n, dtype=float)
    no_stop = np.zeros(n, dtype=float)
    eff = np.full(n, np.nan)
    net = np.full(n, np.nan)

    out["signal_idx"] = np.arange(n, dtype=float)
    out["entry_idx"] = np.arange(n, dtype=float) + float(entry_lag)
    out["entry_datetime"] = pd.NaT
    out["entry_price"] = np.nan

    for tp in tp_levels:
        key = int(round(tp * 100))
        out[f"hit_{key}pct_before_stop_{horizon}"] = np.nan
        out[f"bars_to_{key}pct_{horizon}"] = np.nan
        out[f"first_{key}pct_before_stop_{horizon}"] = np.nan

    for i in range(n):
        entry_idx = i + entry_lag
        end = min(n - 1, entry_idx + horizon)
        if entry_idx >= n or entry_idx >= end:
            continue
        entry = open_[entry_idx] if entry_price_mode == "next_open" else close[entry_idx]
        future_high = high[entry_idx + 1:end + 1]
        future_low = low[entry_idx + 1:end + 1]
        future_close = close[entry_idx + 1:end + 1]
        if len(future_close) == 0 or not np.isfinite(entry) or entry <= 0:
            continue

        out.at[out.index[i], "entry_datetime"] = out.index[entry_idx] if hasattr(out.index, '__getitem__') else pd.NaT
        out.at[out.index[i], "entry_price"] = entry

        mfe[i] = (np.nanmax(future_high) - entry) / entry
        mae_raw = (np.nanmin(future_low) - entry) / entry
        mae[i] = min(0.0, mae_raw)
        net[i] = (future_close[-1] - entry) / entry

        stop_price = low[i] if stop_mode == "event_low" else entry * 0.95
        stop_hits = np.where(future_low <= stop_price)[0]
        stop_idx = int(stop_hits[0] + 1) if len(stop_hits) > 0 else None
        if stop_idx is None:
            no_stop[i] = 1.0
        else:
            bars_to_stop[i] = stop_idx
            stop_first[i] = 1.0

        denom = max(abs(mae[i]), 0.01) if np.isfinite(mae[i]) else 0.01
        eff[i] = mfe[i] / denom if np.isfinite(mfe[i]) else np.nan

        for tp in tp_levels:
            key = int(round(tp * 100))
            tp_hits = np.where(future_high >= entry * (1.0 + tp))[0]
            tp_idx = int(tp_hits[0] + 1) if len(tp_hits) > 0 else None
            if tp_idx is not None:
                out.at[out.index[i], f"bars_to_{key}pct_{horizon}"] = tp_idx
            hit_before_stop = tp_idx is not None and (stop_idx is None or tp_idx < stop_idx)
            first_before_stop = tp_idx is not None and (stop_idx is None or tp_idx <= stop_idx)
            out.at[out.index[i], f"hit_{key}pct_before_stop_{horizon}"] = float(hit_before_stop)
            out.at[out.index[i], f"first_{key}pct_before_stop_{horizon}"] = float(first_before_stop)

    out[f"path_mfe_{horizon}"] = mfe
    out[f"path_mae_{horizon}"] = mae
    out[f"bars_to_stop_{horizon}"] = bars_to_stop
    out[f"stop_first_{horizon}"] = stop_first
    out[f"no_stop_{horizon}"] = no_stop
    out[f"path_efficiency_{horizon}"] = eff
    out[f"path_net_{horizon}"] = net
    out["asymmetry_score"] = safe_numeric(out[f"path_mfe_{horizon}"]) / safe_numeric(out[f"path_mae_{horizon}"]).abs().replace(0, np.nan)
    out["confirm_before_invalid"] = ((out[f"hit_3pct_before_stop_{horizon}"] == 1.0) & (out[f"stop_first_{horizon}"] == 0.0)).astype(int)

    quality = pd.concat([
        normalize_01(out[f"hit_3pct_before_stop_{horizon}"], reverse=False),
        normalize_01(out[f"hit_5pct_before_stop_{horizon}"], reverse=False),
        normalize_01(out[f"path_mfe_{horizon}"], reverse=False),
        normalize_01(out[f"path_mae_{horizon}"], reverse=True),
        normalize_01(out[f"path_efficiency_{horizon}"], reverse=False),
        normalize_01(out["asymmetry_score"], reverse=False),
        normalize_01(out[f"bars_to_3pct_{horizon}"], reverse=True),
        normalize_01(out[f"bars_to_stop_{horizon}"].fillna(horizon + 1), reverse=False),
    ], axis=1).mean(axis=1)
    out["quality_trade_score"] = quality.clip(0.0, 1.0)
    out["good_path_main"] = ((out[f"hit_5pct_before_stop_{horizon}"] == 1.0) & (out[f"path_mae_{horizon}"] >= -0.08)).astype(int)
    out["good_path_soft"] = ((out[f"hit_3pct_before_stop_{horizon}"] == 1.0) & (out[f"path_mae_{horizon}"] >= -0.06)).astype(int)
    return out


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

MOMENTUM_FEATURES = ["ret_1", "ret_3", "ret_5"]
TRADE_FEATURES_NO_MOMO = [f for f in TRADE_FEATURES if f not in MOMENTUM_FEATURES]
ENTRY_PROFILE_FEATURES_NO_MOMO = [
    "dsa_trade_pos_01", "dsa_confirmed_pivot_pos_01", "bb_pos_01", "bb_width_norm", "bb_width_percentile",
    "bb_width_change_5", "bb_expand_streak", "rope_pivot_pos_01", "bars_since_dir_change",
    "trend_gap_10_20", "w_dsa_trade_pos_01", "w_dsa_signed_vwap_dev_pct", "score_trend_total",
    "score_width_total", "score_setup_total",
]



STRUCTURE_CORE_FEATURES = [
    "dsa_trade_pos_01",
    "bb_pos_01",
    "dsa_trade_range_width_pct",
    "bb_width_norm",
    "bars_since_dir_change",
    "bb_expand_streak",
    "trend_gap_10_20",
    "score_trend_total",
]

STRUCTURE_PAIR_FEATURES = list(STRUCTURE_CORE_FEATURES)

RESEARCH_FEATURES = [
    "dsa_confirmed_pivot_pos_01", "dsa_signed_vwap_dev_pct", "dsa_trend_aligned_vwap_dev_pct",
    "lh_hh_low_pos", "dsa_bull_vwap_dev_pct", "dsa_bear_vwap_dev_pct",
    "prev_confirmed_up_bars", "prev_confirmed_down_bars", "last_confirmed_run_bars", "current_run_bars",
    "w_DSA_DIR", "w_dsa_confirmed_pivot_pos_01", "w_dsa_signed_vwap_dev_pct", "w_prev_confirmed_up_bars", "w_prev_confirmed_down_bars",
]

PRIMARY_HINT_FEATURES = [
    "position_in_structure", "dir_consistent", "range_width_atr", "score_trend_total", "score_volume_total",
    "score_width_total", "score_rope_total", "score_setup_total", "dsa_trade_pos_01", "bb_width_percentile",
    "bars_since_dir_change", "prev_confirmed_down_bars", "current_run_bars", "dist_to_rope_atr",
]

COMPOSITE_FEATURES = [
    "position_in_structure", "dir_consistent", "score_trend_total", "score_volume_total",
    "score_width_total", "score_rope_total", "score_setup_total",
]


@dataclass
class NeighborSpec:
    dsa_max: float = 0.45
    prev_down_min: float = 8.0
    need_rope_up: bool = True
    current_run_max: float = 20.0
    bars_since_max: float = 12.0


def build_neighbor_events(df: pd.DataFrame, spec: NeighborSpec, cooldown: int = EVENT_DEDUP_BARS) -> pd.DataFrame:
    need = list(dict.fromkeys([
        "symbol", "datetime", "open", "high", "low", "close",
        *TRADE_FEATURES, *RESEARCH_FEATURES,
        *[f"ret_{w}" for w in RET_WINDOWS],
        *[f"max_dd_{w}" for w in RET_WINDOWS],
        *[f"win_{w}" for w in RET_WINDOWS],
        "quality_trade_score", "good_path_main", "good_path_soft",
        f"path_mfe_{PATH_QUALITY_HORIZON}", f"path_mae_{PATH_QUALITY_HORIZON}", f"path_efficiency_{PATH_QUALITY_HORIZON}",
        f"bars_to_stop_{PATH_QUALITY_HORIZON}", f"bars_to_3pct_{PATH_QUALITY_HORIZON}", f"bars_to_5pct_{PATH_QUALITY_HORIZON}",
        f"hit_3pct_before_stop_{PATH_QUALITY_HORIZON}", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}",
    ]))
    need = [c for c in need if c in df.columns]
    sub = df[need].dropna(subset=[c for c in ["quality_trade_score", "dsa_trade_pos_01", "rope_dir"] if c in need]).copy()
    mask = sub["dsa_trade_pos_01"] <= float(spec.dsa_max)
    if "prev_confirmed_down_bars" in sub.columns:
        mask &= sub["prev_confirmed_down_bars"] >= float(spec.prev_down_min)
    if spec.need_rope_up and "rope_dir" in sub.columns:
        mask &= sub["rope_dir"] == 1
    if "current_run_bars" in sub.columns:
        mask &= sub["current_run_bars"] <= float(spec.current_run_max)
    if "bars_since_dir_change" in sub.columns:
        mask &= sub["bars_since_dir_change"] <= float(spec.bars_since_max)
    events = dedup_events(sub[mask].copy(), cooldown).reset_index(drop=True)
    if events.empty:
        return events
    if "ret_20" in events.columns and "max_dd_20" in events.columns:
        events["rr_20"] = events.apply(lambda r: rr_from_ret_dd(r["ret_20"], r["max_dd_20"]), axis=1)
    events["year"] = pd.to_datetime(events["datetime"]).dt.year
    events["year_month"] = pd.to_datetime(events["datetime"]).dt.to_period("M").astype(str)
    return events


def prepare_feature_matrix(
    events: pd.DataFrame,
    features: Sequence[str],
    fill_source: Optional[pd.DataFrame] = None,
    drop_constant: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    cols = [c for c in features if c in events.columns]
    if not cols:
        return pd.DataFrame(index=events.index), []
    x = events[cols].copy()
    for col in cols:
        x[col] = safe_numeric(x[col])
        med = fill_source[col].median() if fill_source is not None and col in fill_source.columns else x[col].median()
        if pd.isna(med):
            med = 0.0
        x[col] = x[col].fillna(med)
    if drop_constant:
        nunique = x.nunique(dropna=False)
        cols = [c for c in cols if nunique.get(c, 0) > 1]
        x = x[cols]
    return x, cols


def make_time_folds(events: pd.DataFrame, min_train: int = 80, min_test: int = 30) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    ordered_years = sorted(pd.Series(events["year"].dropna().unique()).astype(int).tolist())
    folds: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for i in range(1, len(ordered_years)):
        train_years = ordered_years[:i]
        test_year = ordered_years[i]
        train_idx = events.index[events["year"].isin(train_years)].to_numpy()
        test_idx = events.index[events["year"] == test_year].to_numpy()
        if len(train_idx) >= min_train and len(test_idx) >= min_test:
            folds.append((f"train_{train_years[0]}_{train_years[-1]}__test_{test_year}", train_idx, test_idx))
    if not folds and len(events) >= (min_train + min_test):
        split = int(len(events) * 0.7)
        folds.append(("fallback_70_30", events.index[:split].to_numpy(), events.index[split:].to_numpy()))
    return folds


def regression_score(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = safe_numeric(y_true)
    pred = pd.Series(y_pred, index=y_true.index)
    corr = y_true.corr(pred) if len(y_true) > 1 else np.nan
    mse = float(np.nanmean((y_true - pred) ** 2))
    return {"pred_corr": round(float(corr), 4) if pd.notna(corr) else np.nan, "mse": round(mse, 6)}


def classification_score(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    out: Dict[str, float] = {}
    if y_true.nunique() < 2:
        out["auc"] = np.nan
    else:
        try:
            out["auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            out["auc"] = np.nan
    pred = (y_prob >= 0.5).astype(int)
    out["acc"] = round(float((pred == y_true.to_numpy()).mean()), 4)
    return out



def collect_importance_rows(model_name: str, mode: str, fold_name: str, features: List[str], model, x_test: pd.DataFrame, y_test: pd.Series) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    base_importance = getattr(model, "feature_importances_", None)
    if base_importance is None:
        base_importance = np.zeros(len(features), dtype=float)
    for i, feat in enumerate(features):
        rows.append({
            "model_name": model_name,
            "mode": mode,
            "fold": fold_name,
            "feature": feat,
            "gain_importance": round(float(base_importance[i]), 6),
            "perm_importance": np.nan,
        })
    return rows


def run_single_model(events: pd.DataFrame, features: Sequence[str], target_col: str, model_type: str, feature_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folds = make_time_folds(events)
    metrics_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []
    for fold_name, train_idx, test_idx in folds:
        train_df = events.loc[train_idx].copy()
        test_df = events.loc[test_idx].copy()
        x_train, used_features = prepare_feature_matrix(train_df, features, fill_source=train_df, drop_constant=True)
        x_test, _ = prepare_feature_matrix(test_df, used_features, fill_source=train_df, drop_constant=False)
        x_test = x_test.reindex(columns=used_features, fill_value=0.0)
        if len(used_features) < 3:
            continue
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        if model_type == "reg":
            model = GradientBoostingRegressor(random_state=RANDOM_STATE, learning_rate=0.05, n_estimators=250, max_depth=3, subsample=0.8, min_samples_leaf=20)
            model.fit(x_train, safe_numeric(y_train))
            y_pred = model.predict(x_test)
            score = regression_score(y_test, y_pred)
        else:
            if pd.Series(y_train).nunique() < 2 or pd.Series(y_test).nunique() < 1:
                continue
            model = GradientBoostingClassifier(random_state=RANDOM_STATE, learning_rate=0.05, n_estimators=250, max_depth=3, subsample=0.8, min_samples_leaf=20)
            model.fit(x_train, pd.Series(y_train).astype(int))
            y_pred = model.predict_proba(x_test)[:, 1]
            score = classification_score(y_test, y_pred)
        metrics_rows.append({"feature_mode": feature_mode, "target": target_col, "model_type": model_type, "fold": fold_name, "train_n": len(train_df), "test_n": len(test_df), **score})
        importance_rows.extend(collect_importance_rows(f"{feature_mode}_{target_col}_{model_type}", feature_mode, fold_name, used_features, model, x_test, pd.Series(y_test).astype(int) if model_type == "clf" else safe_numeric(y_test)))
    return pd.DataFrame(metrics_rows), pd.DataFrame(importance_rows)


def aggregate_importance(imp_df: pd.DataFrame) -> pd.DataFrame:
    if imp_df.empty:
        return imp_df
    grp = imp_df.groupby(["model_name", "mode", "feature"], as_index=False).agg(
        gain_importance=("gain_importance", "mean"),
        perm_importance=("perm_importance", "mean"),
        fold_count=("fold", "nunique"),
    )
    grp["rank_score"] = grp[["gain_importance", "perm_importance"]].fillna(0).mean(axis=1)
    return grp.sort_values(["model_name", "rank_score"], ascending=[True, False]).reset_index(drop=True)


def build_bucket_profiles(events: pd.DataFrame, features: Sequence[str], out_path: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feat in features:
        if feat not in events.columns:
            continue
        keep = [c for c in [feat, "quality_trade_score", "good_path_main", "asymmetry_score", "confirm_before_invalid", f"path_mfe_{PATH_QUALITY_HORIZON}", f"path_mae_{PATH_QUALITY_HORIZON}", f"path_efficiency_{PATH_QUALITY_HORIZON}", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", f"bars_to_3pct_{PATH_QUALITY_HORIZON}", "ret_20", "max_dd_20"] if c in events.columns]
        sub = events[keep].copy()
        sub[feat] = safe_numeric(sub[feat])
        sub = sub.dropna(subset=[feat, "quality_trade_score"])
        if len(sub) < 40 or sub[feat].nunique() < 4:
            continue
        try:
            sub["bucket"] = pd.qcut(sub[feat], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        except Exception:
            continue
        agg_map = {
            "n": (feat, "count"),
            "mean_value": (feat, "mean"),
            "quality_trade_score": ("quality_trade_score", "mean"),
        }
        if "good_path_main" in sub.columns:
            agg_map["good_path_main_rate"] = ("good_path_main", "mean")
        if f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}" in sub.columns:
            agg_map[f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}"] = (f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", "mean")
        if f"path_efficiency_{PATH_QUALITY_HORIZON}" in sub.columns:
            agg_map[f"path_efficiency_{PATH_QUALITY_HORIZON}"] = (f"path_efficiency_{PATH_QUALITY_HORIZON}", "mean")
        if f"path_mae_{PATH_QUALITY_HORIZON}" in sub.columns:
            agg_map[f"path_mae_{PATH_QUALITY_HORIZON}"] = (f"path_mae_{PATH_QUALITY_HORIZON}", "mean")
        if f"bars_to_3pct_{PATH_QUALITY_HORIZON}" in sub.columns:
            agg_map[f"bars_to_3pct_{PATH_QUALITY_HORIZON}"] = (f"bars_to_3pct_{PATH_QUALITY_HORIZON}", "mean")
        grp = sub.groupby("bucket", observed=True).agg(**agg_map).reset_index()
        if grp.empty:
            continue
        grp["event_rank_score"] = grp.apply(lambda r: event_quality_rank({
            "quality_trade_score": r.get("quality_trade_score", np.nan),
            "good_path_main": r.get("good_path_main_rate", np.nan),
            f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}": r.get(f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", np.nan),
            f"path_efficiency_{PATH_QUALITY_HORIZON}": r.get(f"path_efficiency_{PATH_QUALITY_HORIZON}", np.nan),
            f"path_mae_{PATH_QUALITY_HORIZON}": r.get(f"path_mae_{PATH_QUALITY_HORIZON}", np.nan),
            f"bars_to_3pct_{PATH_QUALITY_HORIZON}": r.get(f"bars_to_3pct_{PATH_QUALITY_HORIZON}", np.nan),
            f"path_mfe_{PATH_QUALITY_HORIZON}": np.nan,
            f"bars_to_stop_{PATH_QUALITY_HORIZON}": np.nan,
            f"hit_3pct_before_stop_{PATH_QUALITY_HORIZON}": np.nan,
        }, int(r.get("n", 0))), axis=1)
        best_row = grp.sort_values(["event_rank_score", "quality_trade_score"], ascending=[False, False]).iloc[0]
        for _, row in grp.iterrows():
            rows.append({
                "feature": feat,
                "bucket": row["bucket"],
                "n": int(row["n"]),
                "mean_value": round(float(row["mean_value"]), 6),
                "quality_trade_score": round(float(row.get("quality_trade_score", np.nan)), 5),
                "good_path_main_rate": round(float(row.get("good_path_main_rate", np.nan)), 5) if pd.notna(row.get("good_path_main_rate", np.nan)) else np.nan,
                f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}": round(float(row.get(f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", np.nan)), 5) if pd.notna(row.get(f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", np.nan)) else np.nan,
                f"path_efficiency_{PATH_QUALITY_HORIZON}": round(float(row.get(f"path_efficiency_{PATH_QUALITY_HORIZON}", np.nan)), 5) if pd.notna(row.get(f"path_efficiency_{PATH_QUALITY_HORIZON}", np.nan)) else np.nan,
                f"path_mae_{PATH_QUALITY_HORIZON}": round(float(row.get(f"path_mae_{PATH_QUALITY_HORIZON}", np.nan)), 5) if pd.notna(row.get(f"path_mae_{PATH_QUALITY_HORIZON}", np.nan)) else np.nan,
                f"bars_to_3pct_{PATH_QUALITY_HORIZON}": round(float(row.get(f"bars_to_3pct_{PATH_QUALITY_HORIZON}", np.nan)), 3) if pd.notna(row.get(f"bars_to_3pct_{PATH_QUALITY_HORIZON}", np.nan)) else np.nan,
                "event_rank_score": round(float(row.get("event_rank_score", np.nan)), 4),
                "is_best_bucket": int(row["bucket"] == best_row["bucket"]),
            })
    rdf = pd.DataFrame(rows)
    return rdf


def build_parameter_hints(events: pd.DataFrame, trade_imp: pd.DataFrame, research_imp: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if events.empty:
        return pd.DataFrame()
    merged_top = pd.concat([
        trade_imp[["feature", "rank_score"]] if not trade_imp.empty else pd.DataFrame(columns=["feature", "rank_score"]),
        research_imp[["feature", "rank_score"]] if not research_imp.empty else pd.DataFrame(columns=["feature", "rank_score"]),
    ], ignore_index=True)
    if merged_top.empty:
        feature_list = [c for c in PRIMARY_HINT_FEATURES if c in events.columns]
    else:
        feature_list = (merged_top.groupby("feature", as_index=False)["rank_score"].mean().sort_values("rank_score", ascending=False)["feature"].tolist())
        feature_list = [f for f in feature_list if f in events.columns]
    feature_list = list(dict.fromkeys(feature_list + [f for f in PRIMARY_HINT_FEATURES if f in events.columns]))[:max(top_n, len(PRIMARY_HINT_FEATURES))]
    for feat in feature_list:
        s = safe_numeric(events[feat]).dropna()
        if len(s) < 40 or s.nunique() < 4:
            continue
        q20 = float(s.quantile(0.2)); q35 = float(s.quantile(0.35)); q50 = float(s.quantile(0.5)); q65 = float(s.quantile(0.65)); q80 = float(s.quantile(0.8))
        try:
            tmp = events[[c for c in [feat, "quality_trade_score", "good_path_main", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}"] if c in events.columns]].copy()
            tmp[feat] = safe_numeric(tmp[feat])
            tmp = tmp.dropna(subset=[feat, "quality_trade_score"])
            tmp["bucket"] = pd.qcut(tmp[feat], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
            grp = tmp.groupby("bucket", observed=True).agg(quality=("quality_trade_score", "mean"), good=("good_path_main", "mean") if "good_path_main" in tmp.columns else ("quality_trade_score", "mean"), hit5=(f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", "mean") if f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}" in tmp.columns else ("quality_trade_score", "mean"), n=(feat, "count"))
            grp["rank"] = grp.apply(lambda r: event_quality_rank({"quality_trade_score": r.get("quality", np.nan), "good_path_main": r.get("good", np.nan), f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}": r.get("hit5", np.nan)}, int(r.get("n", 0))), axis=1)
            best_bucket = grp.sort_values(["rank", "quality"], ascending=[False, False]).index[0]
        except Exception:
            best_bucket = "Q?"
        if best_bucket in {"Q1", "Q2"}:
            hint_rule = f"{feat} <= {q35:.4f}"
            direction = "lower_better"
        elif best_bucket in {"Q4", "Q5"}:
            hint_rule = f"{feat} >= {q65:.4f}"
            direction = "higher_better"
        else:
            hint_rule = f"{q35:.4f} <= {feat} <= {q65:.4f}"
            direction = "middle_better"
        rows.append({"feature": feat, "best_bucket": best_bucket, "direction": direction, "q20": round(q20, 6), "q35": round(q35, 6), "q50": round(q50, 6), "q65": round(q65, 6), "q80": round(q80, 6), "hint_rule": hint_rule, "sample_n": int(len(s))})
    rdf = pd.DataFrame(rows)
    return rdf



def _apply_weekly_filter(sub: pd.DataFrame, weekly_mode: str, weekly_dev_min: float, weekly_prev_down_min: Optional[float] = None, weekly_prev_down_max: Optional[float] = None) -> pd.DataFrame:
    cond = pd.Series(True, index=sub.index)
    if "w_dsa_signed_vwap_dev_pct" in sub.columns:
        cond &= safe_numeric(sub["w_dsa_signed_vwap_dev_pct"]).fillna(-99) >= weekly_dev_min
    if "w_factor_available" in sub.columns:
        cond &= safe_numeric(sub["w_factor_available"]).fillna(0) > 0
    if weekly_mode == "strict":
        if "w_dsa_trade_pos_01" in sub.columns:
            cond &= safe_numeric(sub["w_dsa_trade_pos_01"]).fillna(1) <= 0.70
        if "w_DSA_DIR" in sub.columns:
            cond &= safe_numeric(sub["w_DSA_DIR"]).fillna(-1) >= 0
    if weekly_prev_down_min is not None and "w_prev_confirmed_down_bars" in sub.columns:
        cond &= safe_numeric(sub["w_prev_confirmed_down_bars"]).fillna(-1) >= float(weekly_prev_down_min)
    if weekly_prev_down_max is not None and "w_prev_confirmed_down_bars" in sub.columns:
        cond &= safe_numeric(sub["w_prev_confirmed_down_bars"]).fillna(999) <= float(weekly_prev_down_max)
    return sub[cond]


def _apply_trigger_filter(sub: pd.DataFrame, trigger_type: str) -> pd.DataFrame:
    expand = safe_numeric(sub["bb_expanding"]).fillna(0) > 0 if "bb_expanding" in sub.columns else pd.Series(False, index=sub.index)
    break_sig = pd.Series(False, index=sub.index)
    for col in ["range_break_up", "dsa_trade_breakout_20"]:
        if col in sub.columns:
            break_sig |= safe_numeric(sub[col]).fillna(0) > 0
    if trigger_type == "expand_only":
        cond = expand & (~break_sig)
    elif trigger_type == "break_only":
        cond = break_sig & (~expand)
    elif trigger_type == "expand_and_break":
        cond = expand & break_sig
    elif trigger_type == "expand_or_break":
        cond = expand | break_sig
    else:
        cond = pd.Series(True, index=sub.index)
    return sub[cond]




def add_trigger_labels(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    expand = safe_numeric(out["bb_expanding"]).fillna(0) > 0 if "bb_expanding" in out.columns else pd.Series(False, index=out.index)
    break_sig = pd.Series(False, index=out.index)
    for col in ["range_break_up", "dsa_trade_breakout_20"]:
        if col in out.columns:
            break_sig |= safe_numeric(out[col]).fillna(0) > 0
    trigger = np.where(expand & break_sig, "expand_and_break",
               np.where(expand & (~break_sig), "expand_only",
               np.where((~expand) & break_sig, "break_only", "none")))
    out["trigger_type"] = trigger
    out["trigger_expand_flag"] = expand.astype(int)
    out["trigger_break_flag"] = break_sig.astype(int)
    return out


def _subset_by_rule(sub: pd.DataFrame, rule: Dict[str, object]) -> pd.DataFrame:
    x = sub.copy()
    if "dsa_confirmed_pivot_pos_01" in x.columns and pd.notna(rule.get("dsa_confirmed_pos_max")):
        x = x[safe_numeric(x["dsa_confirmed_pivot_pos_01"]).fillna(1) <= float(rule["dsa_confirmed_pos_max"])]
    if "dsa_trade_pos_01" in x.columns and pd.notna(rule.get("dsa_trade_pos_max")):
        x = x[safe_numeric(x["dsa_trade_pos_01"]).fillna(1) <= float(rule["dsa_trade_pos_max"])]
    if "bb_width_norm" in x.columns and pd.notna(rule.get("bb_width_norm_max")):
        x = x[safe_numeric(x["bb_width_norm"]).fillna(999) <= float(rule["bb_width_norm_max"])]
    if "bb_pos_01" in x.columns:
        if pd.notna(rule.get("bb_pos_min")):
            x = x[safe_numeric(x["bb_pos_01"]).fillna(-999) >= float(rule["bb_pos_min"])]
        if pd.notna(rule.get("bb_pos_max")):
            x = x[safe_numeric(x["bb_pos_01"]).fillna(999) <= float(rule["bb_pos_max"])]
    if "rope_pivot_pos_01" in x.columns and pd.notna(rule.get("rope_pivot_pos_max")):
        x = x[safe_numeric(x["rope_pivot_pos_01"]).fillna(999) <= float(rule["rope_pivot_pos_max"])]
    if "bars_since_dir_change" in x.columns and pd.notna(rule.get("bars_since_dir_change_max")):
        x = x[safe_numeric(x["bars_since_dir_change"]).fillna(99) <= float(rule["bars_since_dir_change_max"])]
    weekly_prev_down_range = rule.get("weekly_prev_down_range", "none")
    wk_rng = None if weekly_prev_down_range == "none" else tuple(int(v) for v in str(weekly_prev_down_range).split("_"))
    x = _apply_weekly_filter(
        x,
        weekly_mode=str(rule.get("weekly_mode", "soft")),
        weekly_dev_min=float(rule.get("w_dsa_signed_vwap_dev_min", -99)),
        weekly_prev_down_min=None if wk_rng is None else wk_rng[0],
        weekly_prev_down_max=None if wk_rng is None else wk_rng[1],
    )
    trigger_type = str(rule.get("trigger_type", "none"))
    if trigger_type != "none":
        x = x[x["trigger_type"] == trigger_type]
    else:
        x = x[x["trigger_type"] == "none"]
    if "ret_5_min" in rule and pd.notna(rule.get("ret_5_min")) and "ret_5" in x.columns:
        x = x[safe_numeric(x["ret_5"]).fillna(-999) >= float(rule["ret_5_min"])]
    return dedup_events(x)


def _score_rule_stats(stat: Dict[str, float], n: int, year_penalty: float = 0.0) -> float:
    return event_quality_rank(stat, n, stability_penalty=year_penalty)


def build_refined_rule_scan(events: pd.DataFrame, top_k_export: int = 300) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    events = add_trigger_labels(events)

    base_rule = {
        "dsa_confirmed_pos_max": 0.38,
        "dsa_trade_pos_max": 0.55,
        "bb_width_norm_max": 0.20,
        "bb_pos_min": 0.40,
        "bb_pos_max": 0.95,
        "rope_pivot_pos_max": 0.90,
        "bars_since_dir_change_max": 15,
        "w_dsa_signed_vwap_dev_min": -99.0,
        "weekly_mode": "soft",
        "weekly_prev_down_range": "none",
        "trigger_type": "none",
    }

    # staged scan: trigger -> structure -> weekly
    rules: List[Dict[str, object]] = []
    for trigger_type in ACTIVE_TRIGGER_TYPES:
        r = dict(base_rule)
        r["stage"] = "trigger"
        r["trigger_type"] = trigger_type
        rules.append(r)

    for dsa_c in [0.28, 0.33, 0.38]:
        for dsa_t in [0.35, 0.45, 0.55]:
            for bbw in [0.12, 0.18, 0.24]:
                r = dict(base_rule)
                r["stage"] = "structure"
                r["dsa_confirmed_pos_max"] = dsa_c
                r["dsa_trade_pos_max"] = dsa_t
                r["bb_width_norm_max"] = bbw
                rules.append(r)

    for bb_min in [0.40, 0.50, 0.60]:
        for bb_max in [0.80, 0.90, 1.00]:
            if bb_min >= bb_max:
                continue
            r = dict(base_rule)
            r["stage"] = "structure"
            r["bb_pos_min"] = bb_min
            r["bb_pos_max"] = bb_max
            rules.append(r)

    for rope_max in [0.70, 0.90]:
        for weekly_mode in ["off", "soft", "strict"]:
            for trigger_type in ACTIVE_TRIGGER_TYPES:
                r = dict(base_rule)
                r["stage"] = "weekly_trigger"
                r["rope_pivot_pos_max"] = rope_max
                r["weekly_mode"] = weekly_mode
                r["trigger_type"] = trigger_type
                rules.append(r)

    for ret5_min in [None, 0.0]:
        for weekly_prev in ["none", "11_28"]:
            r = dict(base_rule)
            r["stage"] = "weekly_filter"
            r["ret_5_min"] = ret5_min
            r["weekly_prev_down_range"] = weekly_prev
            rules.append(r)

    rows = []
    top_event_rows = []
    summary_rows = []

    years = sorted(pd.to_datetime(events["datetime"]).dt.year.dropna().astype(int).unique().tolist()) if "datetime" in events.columns else []

    for idx, rule in enumerate(rules, start=1):
        sub = _subset_by_rule(events, rule)
        if len(sub) < 12:
            continue
        qstat = summarize_quality(sub)
        score_stat = summarize_score_windows(sub, [10, 20, 40])
        year_scores = []
        if years:
            tmp = sub.copy()
            tmp["year"] = pd.to_datetime(tmp["datetime"]).dt.year
            for y, g in tmp.groupby("year"):
                if len(g) >= 6:
                    year_scores.append(event_quality_rank(summarize_quality(g), len(g), stability_penalty=0.0))
        year_penalty = float(np.std(year_scores)) / 10.0 if len(year_scores) >= 2 else 0.0
        score = _score_rule_stats(qstat, len(sub), year_penalty=year_penalty)

        trigger_mix = sub["trigger_type"].value_counts(normalize=True).to_dict() if "trigger_type" in sub.columns else {}
        row = dict(rule)
        row.update(qstat)
        row.update(score_stat)
        row["year_rr_std"] = round(year_penalty, 4)
        row["rank_score"] = score
        row["trigger_none_pct"] = round(float(trigger_mix.get("none", 0.0)), 4)
        row["trigger_expand_only_pct"] = round(float(trigger_mix.get("expand_only", 0.0)), 4)
        row["trigger_expand_and_break_pct"] = round(float(trigger_mix.get("expand_and_break", 0.0)), 4)
        rows.append(row)

    scan_df = pd.DataFrame(rows)
    if scan_df.empty:
        return scan_df, pd.DataFrame(), pd.DataFrame()

    scan_df = scan_df.sort_values(
        ["rank_score", "quality_trade_score", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", "n"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    for rank, row in scan_df.head(20).iterrows():
        sub = _subset_by_rule(events, row.to_dict())
        if sub.empty:
            continue
        keep_cols = [c for c in [
            "symbol", "datetime", "close", "ret_5", "ret_10", "ret_20", "ret_60", "max_dd_20", "rr_20",
            "quality_trade_score", "good_path_main", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", f"path_efficiency_{PATH_QUALITY_HORIZON}",
            "dsa_confirmed_pivot_pos_01", "dsa_trade_pos_01", "bb_width_norm", "bb_pos_01",
            "rope_pivot_pos_01", "bars_since_dir_change", "w_dsa_signed_vwap_dev_pct", "w_prev_confirmed_down_bars",
            "trigger_type", "bb_expanding", "range_break_up", "dsa_trade_breakout_20"
        ] if c in sub.columns]
        tmp = sub[keep_cols].copy()
        tmp["rule_rank"] = rank + 1
        top_event_rows.append(tmp)

        summary_rows.append({
            "rule_rank": rank + 1,
            "stage": row["stage"],
            "style_tag": "埋伏观察型" if row["trigger_type"] == "none" else ("确认增强型" if row["trigger_type"] == "expand_and_break" else "扩张主线型"),
            "trigger_type": row["trigger_type"],
            "sample_n": int(row["n"]),
            "quality_trade_score": row.get("quality_trade_score"),
            f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}": row.get(f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}"),
            f"path_efficiency_{PATH_QUALITY_HORIZON}": row.get(f"path_efficiency_{PATH_QUALITY_HORIZON}"),
            f"path_mae_{PATH_QUALITY_HORIZON}": row.get(f"path_mae_{PATH_QUALITY_HORIZON}"),
            "score_ret_20": row.get("score_ret_20"),
            "year_event_score_std": row["year_rr_std"],
            "rule_text": (
                f"dsa_confirmed<={row['dsa_confirmed_pos_max']}, dsa_trade<={row['dsa_trade_pos_max']}, "
                f"bb_width<={row['bb_width_norm_max']}, bb_pos in [{row['bb_pos_min']}, {row['bb_pos_max']}], "
                f"rope<={row['rope_pivot_pos_max']}, weekly={row['weekly_mode']}, trigger={row['trigger_type']}"
            ),
        })

    top_df = pd.concat(top_event_rows, ignore_index=True) if top_event_rows else pd.DataFrame()

    summary_df = pd.DataFrame(summary_rows)
    return scan_df, top_df, summary_df
def build_compression_launch_scan(events: pd.DataFrame, top_k_export: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    top_event_rows: List[pd.DataFrame] = []
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    dsa_grid = [0.30, 0.34, 0.38]
    width_grid = [0.22, 0.28, 0.35]
    bars_since_grid = [6, 9, 12]
    weekly_dev_min_grid = [-2.3, -1.6, -1.0]
    weekly_modes = ["soft", "strict"]
    weekly_prev_down_grid = [(None, None), (11, 28)]
    trigger_types = ACTIVE_SCAN_TRIGGER_TYPES

    for dsa_max, width_max, bars_since_max, weekly_dev_min, weekly_mode, weekly_prev_down_rng, trigger_type in product(
        dsa_grid, width_grid, bars_since_grid, weekly_dev_min_grid, weekly_modes, weekly_prev_down_grid, trigger_types
    ):
        sub = events.copy()

        if "dsa_confirmed_pivot_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_confirmed_pivot_pos_01"]).fillna(1) <= dsa_max]
        elif "dsa_trade_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_trade_pos_01"]).fillna(1) <= dsa_max]

        if "dsa_trade_range_width_pct" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_trade_range_width_pct"]).fillna(999) <= width_max]

        if "bars_since_dir_change" in sub.columns:
            sub = sub[safe_numeric(sub["bars_since_dir_change"]).fillna(99) <= bars_since_max]

        sub = _apply_weekly_filter(
            sub,
            weekly_mode=weekly_mode,
            weekly_dev_min=weekly_dev_min,
            weekly_prev_down_min=weekly_prev_down_rng[0],
            weekly_prev_down_max=weekly_prev_down_rng[1],
        )
        sub = _apply_trigger_filter(sub, trigger_type)
        sub = dedup_events(sub)

        if len(sub) < 12:
            continue

        qstat = summarize_quality(sub)
        score_stat = summarize_score_windows(sub, [20])
        quality_parts = []
        for c in ["score_width_total", "score_trend_total", "score_setup_total"]:
            if c in sub.columns:
                quality_parts.append(float(safe_numeric(sub[c]).mean()))
        structural_quality = float(np.nanmean(quality_parts)) if quality_parts else 0.0
        rank_score = event_quality_rank(qstat, len(sub), stability_penalty=0.0) + 2.0 * structural_quality
        rows.append({
            "dsa_confirmed_pos_max": dsa_max,
            "dsa_trade_width_pct_max": width_max,
            "bars_since_dir_change_max": bars_since_max,
            "w_dsa_signed_vwap_dev_min": weekly_dev_min,
            "weekly_mode": weekly_mode,
            "weekly_prev_down_range": "none" if weekly_prev_down_rng[0] is None else f"{weekly_prev_down_rng[0]}_{weekly_prev_down_rng[1]}",
            "trigger_type": trigger_type,
            **qstat,
            **score_stat,
            "structural_quality": round(float(structural_quality), 4),
            "rank_score": round(float(rank_score), 3),
        })

    scan_df = pd.DataFrame(rows).sort_values(["rank_score", "quality_trade_score", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", "n"], ascending=[False, False, False, False]).reset_index(drop=True) if rows else pd.DataFrame()

    for rank, row in scan_df.head(20).iterrows() if not scan_df.empty else []:
        sub = events.copy()
        if "dsa_confirmed_pivot_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_confirmed_pivot_pos_01"]).fillna(1) <= row["dsa_confirmed_pos_max"]]
        elif "dsa_trade_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_trade_pos_01"]).fillna(1) <= row["dsa_confirmed_pos_max"]]
        if "dsa_trade_range_width_pct" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_trade_range_width_pct"]).fillna(999) <= row["dsa_trade_width_pct_max"]]
        if "bars_since_dir_change" in sub.columns:
            sub = sub[safe_numeric(sub["bars_since_dir_change"]).fillna(99) <= row["bars_since_dir_change_max"]]
        wk_rng = None if row["weekly_prev_down_range"] == "none" else tuple(int(x) for x in str(row["weekly_prev_down_range"]).split("_"))
        sub = _apply_weekly_filter(
            sub,
            weekly_mode=str(row["weekly_mode"]),
            weekly_dev_min=float(row["w_dsa_signed_vwap_dev_min"]),
            weekly_prev_down_min=None if wk_rng is None else wk_rng[0],
            weekly_prev_down_max=None if wk_rng is None else wk_rng[1],
        )
        sub = _apply_trigger_filter(sub, str(row["trigger_type"]))
        sub = dedup_events(sub)
        if sub.empty:
            continue
        keep_cols = [c for c in [
            "symbol", "datetime", "close", "ret_20", "max_dd_20", "rr_20",
            "dsa_confirmed_pivot_pos_01", "dsa_trade_pos_01", "dsa_trade_range_width_pct",
            "bars_since_dir_change", "w_dsa_signed_vwap_dev_pct", "w_prev_confirmed_down_bars",
            "bb_expanding", "range_break_up", "dsa_trade_breakout_20",
            "score_width_total", "score_trend_total", "score_volume_total", "score_setup_total",
        ] if c in sub.columns]
        tmp = sub[keep_cols].copy()
        tmp["rule_rank"] = rank + 1
        tmp["trigger_type"] = row["trigger_type"]
        top_event_rows.append(tmp)

    top_df = pd.concat(top_event_rows, ignore_index=True).head(top_k_export) if top_event_rows else pd.DataFrame()
    return scan_df, top_df



def build_stability_slices(events: pd.DataFrame, scan_df: pd.DataFrame) -> pd.DataFrame:
    if events.empty or scan_df.empty:
        return pd.DataFrame()
    top = scan_df.head(6)
    rows: List[Dict[str, object]] = []
    work = events.copy()
    work["turnover_proxy"] = safe_numeric(work.get("close", pd.Series(index=work.index))) * safe_numeric(work.get("vol", pd.Series(index=work.index)))
    if "datetime" in work.columns:
        work["year"] = pd.to_datetime(work["datetime"]).dt.year
    else:
        work["year"] = np.nan
    if work["turnover_proxy"].notna().sum() >= 30:
        try:
            work["liquidity_bucket"] = pd.qcut(work["turnover_proxy"], q=3, labels=["small", "mid", "large"], duplicates="drop")
        except Exception:
            work["liquidity_bucket"] = "all"
    else:
        work["liquidity_bucket"] = "all"
    if "quality_trade_score" in work.columns and work["quality_trade_score"].notna().sum() >= 30:
        med = float(work["quality_trade_score"].median())
        work["env_bucket"] = np.where(work["quality_trade_score"] >= med, "better", "worse")
    else:
        work["env_bucket"] = "all"

    for rank, row in top.iterrows():
        sub = work.copy()
        if "dsa_confirmed_pivot_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_confirmed_pivot_pos_01"]).fillna(1) <= row["dsa_confirmed_pos_max"]]
        elif "dsa_trade_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_trade_pos_01"]).fillna(1) <= row["dsa_confirmed_pos_max"]]
        if "dsa_trade_range_width_pct" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_trade_range_width_pct"]).fillna(999) <= row["dsa_trade_width_pct_max"]]
        if "bars_since_dir_change" in sub.columns:
            sub = sub[safe_numeric(sub["bars_since_dir_change"]).fillna(99) <= row["bars_since_dir_change_max"]]
        wk_rng = None if row["weekly_prev_down_range"] == "none" else tuple(int(x) for x in str(row["weekly_prev_down_range"]).split("_"))
        sub = _apply_weekly_filter(
            sub,
            weekly_mode=str(row["weekly_mode"]),
            weekly_dev_min=float(row["w_dsa_signed_vwap_dev_min"]),
            weekly_prev_down_min=None if wk_rng is None else wk_rng[0],
            weekly_prev_down_max=None if wk_rng is None else wk_rng[1],
        )
        sub = _apply_trigger_filter(sub, str(row["trigger_type"]))
        sub = dedup_events(sub)
        if len(sub) < 12:
            continue
        for dim in ["year", "liquidity_bucket", "env_bucket", "trigger_type"]:
            if dim == "trigger_type":
                g = sub.copy()
                g["trigger_type"] = str(row["trigger_type"])
                groups = g.groupby("trigger_type", dropna=False)
            else:
                groups = sub.groupby(dim, dropna=False)
            for key, g in groups:
                if len(g) < 8:
                    continue
                stat = summarize_events(g, [20])
                rows.append({
                    "rule_rank": rank + 1,
                    "slice_dim": dim,
                    "slice_key": str(key),
                    **stat,
                })
    rdf = pd.DataFrame(rows)
    return rdf

def analyze_00_dataset_summary(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = [{"section": "global", "sample_n": int(len(df)), "stock_n": int(df["symbol"].nunique()), **summarize_quality(df), **summarize_score_windows(df, [20])}]
    if not events.empty:
        rows.append({"section": "candidate_pool", "sample_n": int(len(events)), "stock_n": int(events["symbol"].nunique()), **summarize_quality(events), **summarize_score_windows(events, [20])})
    rdf = pd.DataFrame(rows)
    return rdf


def analyze_01_feature_inventory(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_features = list(dict.fromkeys(TRADE_FEATURES + RESEARCH_FEATURES + COMPOSITE_FEATURES))
    for feat in all_features:
        present = feat in events.columns
        s = events[feat] if present else pd.Series(dtype=float)
        mode = "trade" if feat in TRADE_FEATURES else ("research" if feat in RESEARCH_FEATURES else "composite")
        rows.append({"feature": feat, "feature_mode": mode, "present": int(present), "coverage": round(float(s.notna().mean()), 4) if present else 0.0, "nunique": int(s.nunique(dropna=True)) if present else 0, "mean": round(float(safe_numeric(s).mean()), 6) if present else np.nan, "std": round(float(safe_numeric(s).std()), 6) if present else np.nan})
    rdf = pd.DataFrame(rows).sort_values(["feature_mode", "coverage", "feature"], ascending=[True, False, True])
    return rdf



# =========================
# 20日波段实验扩展
# =========================
SWING_PROFILE_FEATURES = [
    "dsa_trade_pos_01", "dsa_confirmed_pivot_pos_01", "bb_pos_01", "bb_width_norm", "bb_width_percentile",
    "bb_width_change_5", "bb_expand_streak", "rope_pivot_pos_01", "bars_since_dir_change", "ret_5",
    "trend_gap_10_20", "w_dsa_trade_pos_01", "w_dsa_signed_vwap_dev_pct", "score_trend_total",
    "score_width_total", "score_setup_total",
]


def apply_fixed_main_strategy(events: pd.DataFrame) -> pd.DataFrame:
    """固定主策略事件池。买点研究只在该事件池内做因子探索，不再额外预设买点规则。"""
    if events.empty:
        return events.copy()
    sub = add_trigger_labels(events.copy())
    if "trigger_type" in sub.columns:
        sub = sub[sub["trigger_type"].isin(ACTIVE_TRIGGER_TYPES)]
    return dedup_events(sub)


def summarize_quality(events: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {"n": int(len(events))}
    if len(events) == 0:
        return out
    cols = [
        "quality_trade_score",
        f"path_mfe_{PATH_QUALITY_HORIZON}",
        f"path_mae_{PATH_QUALITY_HORIZON}",
        f"path_efficiency_{PATH_QUALITY_HORIZON}",
        f"bars_to_3pct_{PATH_QUALITY_HORIZON}",
        f"bars_to_5pct_{PATH_QUALITY_HORIZON}",
        f"bars_to_stop_{PATH_QUALITY_HORIZON}",
        f"hit_3pct_before_stop_{PATH_QUALITY_HORIZON}",
        f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}",
        "good_path_main",
        "good_path_soft",
    ]
    for col in cols:
        if col in events.columns:
            out[col] = round(float(safe_numeric(events[col]).mean()), 5)
    return out


def event_quality_rank(qstat: Dict[str, float], n: int, stability_penalty: float = 0.0) -> float:
    q = 0.0 if pd.isna(qstat.get("quality_trade_score")) else float(qstat.get("quality_trade_score", 0.0))
    hit5 = 0.0 if pd.isna(qstat.get(f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}")) else float(qstat.get(f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", 0.0))
    hit3 = 0.0 if pd.isna(qstat.get(f"hit_3pct_before_stop_{PATH_QUALITY_HORIZON}")) else float(qstat.get(f"hit_3pct_before_stop_{PATH_QUALITY_HORIZON}", 0.0))
    eff = 0.0 if pd.isna(qstat.get(f"path_efficiency_{PATH_QUALITY_HORIZON}")) else float(qstat.get(f"path_efficiency_{PATH_QUALITY_HORIZON}", 0.0))
    mfe = 0.0 if pd.isna(qstat.get(f"path_mfe_{PATH_QUALITY_HORIZON}")) else float(qstat.get(f"path_mfe_{PATH_QUALITY_HORIZON}", 0.0))
    mae = 0.0 if pd.isna(qstat.get(f"path_mae_{PATH_QUALITY_HORIZON}")) else float(qstat.get(f"path_mae_{PATH_QUALITY_HORIZON}", 0.0))
    b3 = qstat.get(f"bars_to_3pct_{PATH_QUALITY_HORIZON}", np.nan)
    bst = qstat.get(f"bars_to_stop_{PATH_QUALITY_HORIZON}", np.nan)
    speed_bonus = 0.0 if pd.isna(b3) else max(0.0, (PATH_QUALITY_HORIZON + 1 - float(b3)) / (PATH_QUALITY_HORIZON + 1))
    survival_bonus = 0.0 if pd.isna(bst) else min(float(bst), PATH_QUALITY_HORIZON) / PATH_QUALITY_HORIZON
    sample_bonus = min(np.log1p(max(n, 1)), 4.0) / 4.0
    return round(
        35.0 * q
        + 20.0 * hit5
        + 8.0 * hit3
        + 10.0 * min(max(eff, 0.0), 4.0) / 4.0
        + 10.0 * min(max(mfe, 0.0), 0.20) / 0.20
        + 8.0 * speed_bonus
        + 6.0 * survival_bonus
        + 4.0 * sample_bonus
        - 14.0 * abs(min(mae, 0.0)) / 0.12
        - stability_penalty,
        4,
    )


def summarize_score_windows(events: pd.DataFrame, windows: Sequence[int] = RET_WINDOWS) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if len(events) == 0:
        return out
    for w in windows:
        ret_col = f"ret_{w}"
        dd_col = f"max_dd_{w}"
        if ret_col in events.columns:
            out[f"score_ret_{w}"] = round(float(safe_numeric(events[ret_col]).mean()), 5)
        if dd_col in events.columns:
            out[f"score_mae_{w}"] = round(float(safe_numeric(events[dd_col]).mean()), 5)
    return out


def build_entry_profile_reports(events: pd.DataFrame, features: Optional[Sequence[str]] = None, pair_defs: Optional[Sequence[Tuple[str, str]]] = None, profile_name: str = "with_momo", apply_main_strategy: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    base = apply_fixed_main_strategy(events.copy()) if apply_main_strategy else events.copy()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    use_features = list(features) if features is not None else list(SWING_PROFILE_FEATURES)
    use_pairs = list(pair_defs) if pair_defs is not None else [
        ("dsa_trade_pos_01", "bb_width_norm"),
        ("dsa_trade_pos_01", "bb_width_change_5"),
        ("ret_5", "bb_width_norm"),
        ("bars_since_dir_change", "bb_expand_streak"),
        ("w_dsa_trade_pos_01", "dsa_trade_pos_01"),
        ("trend_gap_10_20", "bb_pos_01"),
    ]

    single_rows: List[Dict[str, object]] = []
    for feat in use_features:
        if feat not in base.columns:
            continue
        s = safe_numeric(base[feat])
        valid = base[s.notna()].copy()
        if len(valid) < 30:
            continue
        try:
            valid["bucket"] = pd.qcut(s[s.notna()], q=5, duplicates="drop")
        except Exception:
            continue
        for bucket, g in valid.groupby("bucket", observed=True):
            stat = summarize_events(g, [10, 20, 40])
            qstat = summarize_quality(g)
            row = {
                "profile_name": profile_name,
                "feature": feat,
                "bucket": str(bucket),
                "bucket_n": int(len(g)),
                "feature_mean": round(float(safe_numeric(g[feat]).mean()), 6),
                "feature_median": round(float(safe_numeric(g[feat]).median()), 6),
            }
            row.update(stat)
            row.update(qstat)
            single_rows.append(row)
    single_df = pd.DataFrame(single_rows).sort_values(["profile_name", "feature", "quality_trade_score", "good_path_main_rate", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}"], ascending=[True, True, False, False, False]) if single_rows else pd.DataFrame()

    pair_rows: List[Dict[str, object]] = []
    for fx, fy in use_pairs:
        if fx not in base.columns or fy not in base.columns:
            continue
        keep = [c for c in [fx, fy, "ret_10", "ret_20", "ret_40", "max_dd_20", "win_20", "symbol", "datetime", "quality_trade_score", "good_path_main", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", f"path_efficiency_{PATH_QUALITY_HORIZON}"] if c in base.columns]
        sub = base[keep].copy()
        sub[fx] = safe_numeric(sub[fx])
        sub[fy] = safe_numeric(sub[fy])
        sub = sub.dropna(subset=[fx, fy])
        if len(sub) < 40:
            continue
        try:
            sub["bucket_x"] = pd.qcut(sub[fx], q=4, duplicates="drop")
            sub["bucket_y"] = pd.qcut(sub[fy], q=4, duplicates="drop")
        except Exception:
            continue
        for (bx, by), g in sub.groupby(["bucket_x", "bucket_y"], observed=True):
            stat = summarize_events(g, [10, 20, 40])
            qstat = summarize_quality(g)
            pair_rows.append({
                "profile_name": profile_name,
                "feature_x": fx,
                "feature_y": fy,
                "bucket_x": str(bx),
                "bucket_y": str(by),
                "n": int(len(g)),
                "mean_x": round(float(g[fx].mean()), 6),
                "mean_y": round(float(g[fy].mean()), 6),
                **stat,
                **qstat,
            })
    pair_df = pd.DataFrame(pair_rows).sort_values(["profile_name", "feature_x", "feature_y", "quality_trade_score", "good_path_main_rate", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}"], ascending=[True, True, True, False, False, False]) if pair_rows else pd.DataFrame()

    trigger_summary = base.groupby("trigger_type", observed=True).apply(lambda g: pd.Series({**summarize_events(g, [10, 20, 40]), **summarize_quality(g)})).reset_index() if "trigger_type" in base.columns else pd.DataFrame()
    if not trigger_summary.empty:
        trigger_summary.insert(0, "profile_name", profile_name)
    return single_df, pair_df, trigger_summary


def build_entry_momentum_comparison(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    base = apply_fixed_main_strategy(events.copy())
    if base.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    compare_sets = {
        "with_momo": TRADE_FEATURES,
        "without_momo": TRADE_FEATURES_NO_MOMO,
    }
    for profile_name, feats in compare_sets.items():
        available = [f for f in feats if f in base.columns]
        for feat in available:
            s = safe_numeric(base[feat])
            valid = base[s.notna()].copy()
            if len(valid) < 50 or s.nunique(dropna=True) < 5:
                continue
            try:
                valid["bucket"] = pd.qcut(s[s.notna()], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
            except Exception:
                continue
            grp = valid.groupby("bucket", observed=True).apply(lambda g: pd.Series({**summarize_events(g, [10, 20, 40]), **summarize_quality(g)})).reset_index()
            if grp.empty or "quality_trade_score" not in grp.columns:
                continue
            best = grp.sort_values(["quality_trade_score", "good_path_main_rate", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}"], ascending=[False, False, False]).iloc[0]
            worst = grp.sort_values(["quality_trade_score", "good_path_main_rate", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}"], ascending=[True, True, True]).iloc[0]
            rows.append({
                "profile_name": profile_name,
                "feature": feat,
                "available_n": int(len(valid)),
                "best_bucket": str(best["bucket"]),
                "best_quality": round(float(best.get("quality_trade_score", np.nan)), 5),
                "best_hit5": round(float(best.get(f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", np.nan)), 5),
                "best_efficiency": round(float(best.get(f"path_efficiency_{PATH_QUALITY_HORIZON}", np.nan)), 5),
                "best_hit5": round(float(best.get(f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", np.nan)), 5),
                "best_rr20": round(float(best.get("rr_20", np.nan)), 5),
                "worst_bucket": str(worst["bucket"]),
                "worst_quality": round(float(worst.get("quality_trade_score", np.nan)), 5),
                "quality_spread": round(float(best.get("quality_trade_score", np.nan) - worst.get("quality_trade_score", np.nan)), 5),
                "hit5_spread": round(float(best.get(f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", np.nan) - worst.get(f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", np.nan)), 5),
                "rr20_spread": round(float(best.get("rr_20", np.nan) - worst.get("rr_20", np.nan)), 5),
            })
    return pd.DataFrame(rows).sort_values(["profile_name", "quality_spread", "hit5_spread"], ascending=[True, False, False]) if rows else pd.DataFrame()


def _build_event_lookup(df: pd.DataFrame) -> Dict[Tuple[str, pd.Timestamp], int]:
    work = df[["symbol", "datetime"]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    return {(str(r.symbol), pd.Timestamp(r.datetime)): int(i) for i, r in work.reset_index(drop=True).iterrows()}


def _evaluate_event_exit_on_path(path: pd.DataFrame, entry_idx: int, strategy: str, max_bars: int = EVENT_EXIT_MAX_BARS) -> Dict[str, float]:
    if entry_idx >= len(path) - 1:
        return {"ret": np.nan, "bars_to_exit": np.nan, "exit_reason": "insufficient_future"}
    entry_close = float(path.iloc[entry_idx]["close"])
    entry_low = float(path.iloc[entry_idx]["low"])
    future = path.iloc[entry_idx + 1:min(len(path), entry_idx + max_bars + 1)].copy().reset_index(drop=True)
    if future.empty:
        return {"ret": np.nan, "bars_to_exit": np.nan, "exit_reason": "insufficient_future"}

    exit_bar = None
    exit_price = np.nan
    exit_reason = "no_event_within_window"

    if strategy == "event_low_stop":
        hit = future[future["low"] <= entry_low]
        if not hit.empty:
            exit_bar = int(hit.index[0]) + 1
            exit_price = entry_low
            exit_reason = "event_low_stop"
    elif strategy == "rope_break":
        if "rope" in future.columns:
            hit = future[future["close"] < future["rope"]]
            if not hit.empty:
                exit_bar = int(hit.index[0]) + 1
                exit_price = float(hit.iloc[0]["close"])
                exit_reason = "rope_break"
    elif strategy == "rope_flip_break":
        need_cols = [c for c in ["rope", "rope_slope_atr_5"] if c in future.columns]
        if len(need_cols) == 2:
            hit = future[(future["close"] < future["rope"]) & (future["rope_slope_atr_5"] < 0)]
            if not hit.empty:
                exit_bar = int(hit.index[0]) + 1
                exit_price = float(hit.iloc[0]["close"])
                exit_reason = "rope_flip_break"
    elif strategy == "bbmid_break":
        if "bb_mid" in future.columns:
            hit = future[future["close"] < future["bb_mid"]]
            if not hit.empty:
                exit_bar = int(hit.index[0]) + 1
                exit_price = float(hit.iloc[0]["close"])
                exit_reason = "bbmid_break"
    elif strategy == "peak_pullback_4":
        if "close" in future.columns:
            running_peak = future["close"].cummax()
            dd = future["close"] / running_peak - 1.0
            armed = running_peak >= entry_close * 1.08
            hit = future[armed & (dd <= -0.04)]
            if not hit.empty:
                exit_bar = int(hit.index[0]) + 1
                exit_price = float(hit.iloc[0]["close"])
                exit_reason = "peak_pullback_4"
    elif strategy == "tp8_then_rope":
        tp_hit = future[future["high"] >= entry_close * 1.08] if "high" in future.columns else pd.DataFrame()
        if not tp_hit.empty and "rope" in future.columns:
            tp_pos = int(tp_hit.index[0])
            after_tp = future.iloc[tp_pos:].copy()
            rb = after_tp[after_tp["close"] < after_tp["rope"]]
            if not rb.empty:
                exit_bar = tp_pos + int(rb.index[0]) + 1
                exit_price = float(rb.iloc[0]["close"])
                exit_reason = "tp8_then_rope"
    elif strategy == "layered_combo":
        profit_armed_soft = False
        profit_armed_hard = False
        grace_bars = 4
        fallback_bars = 8
        hard_stop_price = min(entry_low, entry_close * 0.96)
        for j, row in future.iterrows():
            bar_no = int(j) + 1
            bar_high = float(row.get("high", np.nan)) if pd.notna(row.get("high", np.nan)) else np.nan
            bar_low = float(row.get("low", np.nan)) if pd.notna(row.get("low", np.nan)) else np.nan
            bar_close = float(row.get("close", np.nan)) if pd.notna(row.get("close", np.nan)) else np.nan
            bar_rope = float(row.get("rope", np.nan)) if pd.notna(row.get("rope", np.nan)) else np.nan
            bar_bbmid = float(row.get("bb_mid", np.nan)) if pd.notna(row.get("bb_mid", np.nan)) else np.nan

            sub = future.iloc[:j + 1].copy()
            run_peak = float(sub["high"].cummax().iloc[-1]) if "high" in sub.columns else np.nan
            current_mfe = (run_peak - entry_close) / entry_close if np.isfinite(run_peak) else np.nan
            recent_high = float(sub["high"].iloc[-3:].max()) if "high" in sub.columns else np.nan
            stale_high = np.isfinite(run_peak) and np.isfinite(recent_high) and recent_high < run_peak * 0.995

            if np.isfinite(bar_high) and bar_high >= entry_close * 1.05:
                profit_armed_soft = True
            if np.isfinite(bar_high) and bar_high >= entry_close * 1.08:
                profit_armed_hard = True

            # 层1：失败保护。宽限期内只保留更宽的灾难性止损，宽限期后才恢复事件低点止损。
            if bar_no <= grace_bars:
                if np.isfinite(bar_low) and bar_low <= hard_stop_price:
                    exit_bar = bar_no
                    exit_price = hard_stop_price
                    exit_reason = "layered_hard_stop_in_grace"
                    break
            else:
                if np.isfinite(bar_low) and bar_low <= entry_low:
                    exit_bar = bar_no
                    exit_price = entry_low
                    exit_reason = "layered_event_low_stop"
                    break

            # 层2：利润保护。先有利润再启用；5%先软保护，8%再强保护。
            if profit_armed_soft:
                running_peak = sub["high"].cummax() if "high" in sub.columns else sub["close"].cummax()
                pullback = bar_close / float(running_peak.iloc[-1]) - 1.0 if np.isfinite(bar_close) and np.isfinite(float(running_peak.iloc[-1])) else np.nan
                soft_limit = -0.06
                hard_limit = -0.04 if profit_armed_hard else -0.06
                if np.isfinite(pullback) and pullback <= hard_limit and np.isfinite(current_mfe) and current_mfe >= 0.05:
                    exit_bar = bar_no
                    exit_price = bar_close
                    exit_reason = "layered_peak_pullback"
                    break
                if profit_armed_hard and np.isfinite(bar_rope) and bar_close < bar_rope:
                    exit_bar = bar_no
                    exit_price = bar_close
                    exit_reason = "layered_tp8_then_rope"
                    break
                if (not profit_armed_hard) and np.isfinite(bar_rope) and bar_close < bar_rope and np.isfinite(current_mfe) and current_mfe >= 0.05 and np.isfinite(pullback) and pullback <= soft_limit:
                    exit_bar = bar_no
                    exit_price = bar_close
                    exit_reason = "layered_soft_rope_guard"
                    break
                continue

            # 层3：趋势兜底。仅在观察期后、且已有一定浮盈或明显走弱时才接管。
            if bar_no >= fallback_bars and ((np.isfinite(current_mfe) and current_mfe >= 0.03) or stale_high):
                if np.isfinite(bar_rope) and bar_close < bar_rope and bar_close < entry_close * 1.01:
                    exit_bar = bar_no
                    exit_price = bar_close
                    exit_reason = "layered_rope_break"
                    break
                if np.isfinite(bar_bbmid) and bar_close < bar_bbmid and bar_close < entry_close * 1.00:
                    exit_bar = bar_no
                    exit_price = bar_close
                    exit_reason = "layered_bbmid_break"
                    break

    peak = float(future["high"].max()) if "high" in future.columns else np.nan
    trough = float(future["low"].min()) if "low" in future.columns else np.nan
    mfe = (peak - entry_close) / entry_close if np.isfinite(peak) else np.nan
    mae = min(0.0, (trough - entry_close) / entry_close) if np.isfinite(trough) else np.nan
    censor_ret = (float(future.iloc[-1]["close"]) - entry_close) / entry_close

    if exit_bar is None:
        return {
            "ret": np.nan,
            "bars_to_exit": np.nan,
            "exit_reason": exit_reason,
            "mfe": mfe,
            "mae": mae,
            "censor_ret": censor_ret,
            "exited": 0.0,
        }

    ret = (exit_price - entry_close) / entry_close
    capture = ret / mfe if np.isfinite(mfe) and mfe > 0 else np.nan
    giveback = (mfe - ret) / mfe if np.isfinite(mfe) and mfe > 0 else np.nan
    return {
        "ret": ret,
        "bars_to_exit": exit_bar,
        "exit_reason": exit_reason,
        "mfe": mfe,
        "mae": mae,
        "censor_ret": censor_ret,
        "capture_ratio": capture,
        "giveback_ratio": giveback,
        "exited": 1.0,
    }


def build_scored_entry_subsets(events: pd.DataFrame, top_pct: float = 0.2, min_group: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.empty or "datetime" not in events.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    outputs: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, object]] = []
    for variant_name, include_momentum in [("with_momo", True), ("without_momo", False)]:
        scored = compute_swing_launch_score(events.copy(), include_momentum=include_momentum)
        scored = apply_fixed_main_strategy(scored)
        if scored.empty or "swing_launch_score" not in scored.columns:
            outputs[variant_name] = pd.DataFrame()
            continue
        scored["year_month"] = pd.to_datetime(scored["datetime"]).dt.to_period("M").astype(str)
        selected_parts: List[pd.DataFrame] = []
        for ym, g in scored.groupby("year_month", sort=True):
            if len(g) < min_group:
                continue
            q = max(3, int(np.ceil(len(g) * top_pct)))
            top = g.sort_values(["swing_launch_score", "quality_trade_score"], ascending=[False, False]).head(q).copy()
            top["selection_pct"] = q / len(g)
            top["score_variant"] = variant_name
            selected_parts.append(top)
        subset = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
        outputs[variant_name] = subset
        if not subset.empty:
            summary_rows.append({
                "score_variant": variant_name,
                "months": int(subset["year_month"].nunique()),
                "n": int(len(subset)),
                "score_mean": round(float(subset["swing_launch_score"].mean()), 5),
                "quality_mean": round(float(safe_numeric(subset["quality_trade_score"]).mean()), 5),
                f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}": round(float(safe_numeric(subset[f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}"]).mean()), 5) if f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}" in subset.columns else np.nan,
                f"path_efficiency_{PATH_QUALITY_HORIZON}": round(float(safe_numeric(subset[f"path_efficiency_{PATH_QUALITY_HORIZON}"]).mean()), 5) if f"path_efficiency_{PATH_QUALITY_HORIZON}" in subset.columns else np.nan,
                "good_path_main_rate": round(float(safe_numeric(subset["good_path_main"]).mean()), 5) if "good_path_main" in subset.columns else np.nan,
                "score_ret_20_mean": round(float(safe_numeric(subset["ret_20"]).mean()), 5) if "ret_20" in subset.columns else np.nan,
            })
    summary_df = pd.DataFrame(summary_rows).sort_values("score_variant") if summary_rows else pd.DataFrame()
    return outputs.get("with_momo", pd.DataFrame()), outputs.get("without_momo", pd.DataFrame()), summary_df


def build_entry_momo_compare(with_momo: pd.DataFrame, without_momo: pd.DataFrame) -> pd.DataFrame:
    if with_momo.empty and without_momo.empty:
        return pd.DataFrame()
    wm = with_momo.copy() if not with_momo.empty else pd.DataFrame(columns=["feature", "bucket"])
    wom = without_momo.copy() if not without_momo.empty else pd.DataFrame(columns=["feature", "bucket"])
    keys = [c for c in ["feature", "bucket"] if c in wm.columns or c in wom.columns]
    if not keys:
        return pd.DataFrame()
    left_cols = [c for c in ["feature", "bucket", "n", "good_path_main_rate", "quality_trade_score", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", f"path_efficiency_{PATH_QUALITY_HORIZON}", f"path_mae_{PATH_QUALITY_HORIZON}", "ret_20"] if c in wm.columns]
    right_cols = [c for c in ["feature", "bucket", "n", "good_path_main_rate", "quality_trade_score", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", f"path_efficiency_{PATH_QUALITY_HORIZON}", f"path_mae_{PATH_QUALITY_HORIZON}", "ret_20"] if c in wom.columns]
    merged = pd.merge(
        wm[left_cols].rename(columns={c: f"{c}_with_momo" for c in left_cols if c not in keys}),
        wom[right_cols].rename(columns={c: f"{c}_without_momo" for c in right_cols if c not in keys}),
        on=keys, how="outer"
    )
    for metric in ["good_path_main_rate", "quality_trade_score", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", f"path_efficiency_{PATH_QUALITY_HORIZON}", f"path_mae_{PATH_QUALITY_HORIZON}", "ret_20"]:
        a = f"{metric}_with_momo"
        b = f"{metric}_without_momo"
        if a in merged.columns and b in merged.columns:
            merged[f"delta_{metric}"] = safe_numeric(merged[a]) - safe_numeric(merged[b])
    sort_col = f"delta_hit_5pct_before_stop_{PATH_QUALITY_HORIZON}" if f"delta_hit_5pct_before_stop_{PATH_QUALITY_HORIZON}" in merged.columns else ("delta_quality_trade_score" if "delta_quality_trade_score" in merged.columns else keys[0])
    return merged.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)


def build_exit_matrix_report(full_df: pd.DataFrame, events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if full_df.empty or events.empty:
        return pd.DataFrame(), pd.DataFrame()
    base = apply_fixed_main_strategy(events.copy())
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()
    keep_cols = [c for c in ["symbol", "datetime", "open", "high", "low", "close", "rope", "rope_dir", "rope_slope_atr_5", "bb_mid"] if c in full_df.columns]
    work = full_df[keep_cols].copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    work = work.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    grouped = {sym: g.reset_index(drop=True) for sym, g in work.groupby("symbol", sort=False)}

    strategies = ["event_low_stop", "rope_break", "rope_flip_break", "bbmid_break", "peak_pullback_4", "tp8_then_rope", "layered_combo"]
    rows: List[Dict[str, object]] = []
    detail_rows: List[pd.DataFrame] = []
    for strategy in strategies:
        per_event: List[Dict[str, object]] = []
        for _, ev in base.iterrows():
            sym = str(ev["symbol"])
            dt = pd.Timestamp(ev["datetime"])
            path = grouped.get(sym)
            if path is None:
                continue
            hit = path.index[path["datetime"] == dt]
            if len(hit) == 0:
                continue
            entry_idx = int(hit[0])
            res = _evaluate_event_exit_on_path(path, entry_idx, strategy=strategy, max_bars=EVENT_EXIT_MAX_BARS)
            row = {
                "strategy": strategy,
                "symbol": sym,
                "datetime": dt,
                "entry_close": float(path.iloc[entry_idx]["close"]),
                "ret": res.get("ret"),
                "bars_to_exit": res.get("bars_to_exit"),
                "exit_reason": res.get("exit_reason"),
                "mfe": res.get("mfe"),
                "mae": res.get("mae"),
                "censor_ret": res.get("censor_ret"),
                "capture_ratio": res.get("capture_ratio"),
                "giveback_ratio": res.get("giveback_ratio"),
                "exited": res.get("exited"),
            }
            per_event.append(row)
        if not per_event:
            continue
        det = pd.DataFrame(per_event)
        detail_rows.append(det)
        exited = det[det["exited"] == 1].copy()
        rows.append({
            "strategy": strategy,
            "n": int(len(det)),
            "exit_rate": round(float(det["exited"].mean()), 4),
            "no_exit_rate": round(float((det["exited"] == 0).mean()), 4),
            "ret_mean_exited": round(float(exited["ret"].mean()), 5) if not exited.empty else np.nan,
            "ret_median_exited": round(float(exited["ret"].median()), 5) if not exited.empty else np.nan,
            "win_rate_exited": round(float((exited["ret"] > 0).mean()), 4) if not exited.empty else np.nan,
            "capture_ratio_mean": round(float(exited["capture_ratio"].mean()), 4) if not exited.empty else np.nan,
            "giveback_ratio_mean": round(float(exited["giveback_ratio"].mean()), 4) if not exited.empty else np.nan,
            "bars_to_exit_mean": round(float(exited["bars_to_exit"].mean()), 2) if not exited.empty else np.nan,
            "mae_mean_all": round(float(det["mae"].mean()), 5),
            "mfe_mean_all": round(float(det["mfe"].mean()), 5),
            "censor_ret_mean_no_exit": round(float(det.loc[det["exited"] == 0, "censor_ret"].mean()), 5) if (det["exited"] == 0).any() else np.nan,
        })
    summary_df = pd.DataFrame(rows).sort_values(["capture_ratio_mean", "ret_mean_exited"], ascending=[False, False]) if rows else pd.DataFrame()
    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return summary_df, detail_df


def compute_swing_launch_score(events: pd.DataFrame, include_momentum: bool = True) -> pd.DataFrame:
    out = add_trigger_labels(events.copy())
    idx = out.index
    def get(name: str) -> pd.Series:
        return safe_numeric(out[name]) if name in out.columns else pd.Series(np.nan, index=idx, dtype=float)
    low_pos = pd.concat([
        normalize_01(get("dsa_trade_pos_01"), reverse=True),
        normalize_01(get("dsa_confirmed_pivot_pos_01"), reverse=True),
        normalize_01(get("w_dsa_trade_pos_01"), reverse=True),
    ], axis=1).mean(axis=1)
    expand = pd.concat([
        1.0 - (normalize_01((get("bb_width_norm") - 0.18).abs(), reverse=False)),
        normalize_01(get("bb_width_change_5"), reverse=False),
        normalize_01(get("bb_expand_streak"), reverse=False),
    ], axis=1).mean(axis=1)
    early = pd.concat([
        normalize_01(get("bars_since_dir_change"), reverse=True),
        normalize_01(get("current_run_bars"), reverse=True),
    ], axis=1).mean(axis=1)
    gentle_parts = [normalize_01(get("trend_gap_10_20"), reverse=False)]
    if include_momentum:
        gentle_parts.insert(0, 1.0 - normalize_01((get("ret_5") - 0.04).abs(), reverse=False))
    gentle = pd.concat(gentle_parts, axis=1).mean(axis=1)
    weekly = pd.concat([
        normalize_01(get("w_dsa_trade_pos_01"), reverse=True),
        normalize_01(get("w_dsa_signed_vwap_dev_pct"), reverse=False),
        normalize_01(get("w_DSA_DIR"), reverse=False),
    ], axis=1).mean(axis=1)
    out["score_low_pos"] = low_pos
    out["score_expand"] = expand
    out["score_early"] = early
    out["score_gentle"] = gentle
    out["score_weekly"] = weekly
    out["swing_launch_score"] = (0.28 * low_pos + 0.24 * expand + 0.18 * early + 0.18 * gentle + 0.12 * weekly).clip(0.0, 1.0)
    return out


def build_model_comparison_report(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    model_defs = [
        ("trade_with_momo", TRADE_FEATURES),
        ("trade_without_momo", TRADE_FEATURES_NO_MOMO),
        ("research", RESEARCH_FEATURES),
    ]
    metric_frames: List[pd.DataFrame] = []
    importance_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    for feature_mode, feats in model_defs:
        reg_metrics, reg_imp = run_single_model(events, feats, "quality_trade_score", "reg", feature_mode)
        clf_metrics, clf_imp = run_single_model(events, feats, "good_path_main", "clf", feature_mode)
        if not reg_metrics.empty:
            metric_frames.append(reg_metrics)
        if not clf_metrics.empty:
            metric_frames.append(clf_metrics)
        if not reg_imp.empty:
            importance_frames.append(reg_imp)
        if not clf_imp.empty:
            importance_frames.append(clf_imp)
        reg_corr = reg_metrics["pred_corr"].mean() if (not reg_metrics.empty and "pred_corr" in reg_metrics.columns) else np.nan
        clf_auc = clf_metrics["auc"].mean() if (not clf_metrics.empty and "auc" in clf_metrics.columns) else np.nan
        summary_rows.append({
            "feature_mode": feature_mode,
            "feature_count": len([f for f in feats if f in events.columns]),
            "reg_fold_count": int(len(reg_metrics)),
            "clf_fold_count": int(len(clf_metrics)),
            "mean_pred_corr": round(float(reg_corr), 4) if pd.notna(reg_corr) else np.nan,
            "mean_auc": round(float(clf_auc), 4) if pd.notna(clf_auc) else np.nan,
        })
    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    importances_df = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    imp_agg = aggregate_importance(importances_df) if not importances_df.empty else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_pred_corr", "mean_auc"], ascending=[False, False]) if summary_rows else pd.DataFrame()
    spread_rows: List[Dict[str, object]] = []
    if not imp_agg.empty:
        wm = imp_agg[imp_agg["mode"] == "trade_with_momo"].groupby("feature", as_index=False)["rank_score"].mean().rename(columns={"rank_score": "rank_with_momo"})
        wom = imp_agg[imp_agg["mode"] == "trade_without_momo"].groupby("feature", as_index=False)["rank_score"].mean().rename(columns={"rank_score": "rank_without_momo"})
        merged = wm.merge(wom, on="feature", how="outer")
        merged["delta_without_minus_with"] = merged["rank_without_momo"].fillna(0) - merged["rank_with_momo"].fillna(0)
        spread_rows = merged.sort_values("delta_without_minus_with", ascending=False).to_dict("records")
    spread_df = pd.DataFrame(spread_rows)
    return summary_df, metrics_df, imp_agg, spread_df


def build_score_oos_report(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty or "datetime" not in events.columns:
        return pd.DataFrame(), pd.DataFrame()
    variants: List[Tuple[str, pd.DataFrame]] = [
        ("with_momo", compute_swing_launch_score(events.copy(), include_momentum=True)),
        ("without_momo", compute_swing_launch_score(events.copy(), include_momentum=False)),
    ]

    detail_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    for variant_name, scored in variants:
        scored = apply_fixed_main_strategy(scored)
        if scored.empty or "swing_launch_score" not in scored.columns:
            continue
        scored["year"] = pd.to_datetime(scored["datetime"]).dt.year
        scored["year_month"] = pd.to_datetime(scored["datetime"]).dt.to_period("M").astype(str)
        variant_detail_rows: List[pd.DataFrame] = []
        for ym, g in scored.groupby("year_month", sort=True):
            if len(g) < 8:
                continue
            q = max(3, int(np.ceil(len(g) * 0.2)))
            top = g.sort_values(["swing_launch_score", "quality_trade_score"], ascending=[False, False]).head(q).copy()
            top["selection_pct"] = q / len(g)
            top["score_variant"] = variant_name
            variant_detail_rows.append(top)
        variant_df = pd.concat(variant_detail_rows, ignore_index=True) if variant_detail_rows else pd.DataFrame()
        if variant_df.empty:
            continue
        detail_frames.append(variant_df)
        for y, g in variant_df.groupby("year", sort=True):
            qstat = summarize_quality(g)
            sstat = summarize_score_windows(g, [10, 20, 40])
            summary_rows.append({
                "score_variant": variant_name,
                "year": int(y),
                "months": int(g["year_month"].nunique()),
                "n": int(len(g)),
                "score_mean": round(float(g["swing_launch_score"].mean()), 5),
                **qstat,
                **sstat,
            })
    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values(["score_variant", "year"]) if summary_rows else pd.DataFrame()
    if not detail_df.empty:
        keep_cols = [c for c in ["score_variant", "symbol", "datetime", "year_month", "swing_launch_score", "score_low_pos", "score_expand", "score_early", "score_gentle", "score_weekly", "quality_trade_score", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", f"path_efficiency_{PATH_QUALITY_HORIZON}", f"path_mae_{PATH_QUALITY_HORIZON}", f"bars_to_3pct_{PATH_QUALITY_HORIZON}", "ret_10", "ret_20", "ret_40", "max_dd_20", "trigger_type"] if c in detail_df.columns]
        detail_df = detail_df[keep_cols].copy()
    return summary_df, detail_df



def _build_excel_summary_sheet() -> pd.DataFrame:
    rows = [
        {"section": "研究定位", "item": "主策略", "value": "主策略事件池固定；本脚本不再讨论主策略本身，只研究事件池内的买点因子质量与卖点事件退出。"},
        {"section": "研究定位", "item": "买点研究", "value": "买点不是出现事件就买，而是在固定主策略事件池内继续做因子探索与分层；固定持有期只作离线评分，不作买卖依据。"},
        {"section": "研究定位", "item": "卖点研究", "value": "卖点采用状态/事件实验，严格禁止用固定持有期作为卖出依据；固定窗口收益仅保留为评分观察列。"},
        {"section": "导出说明", "item": "输出形式", "value": "不再输出任何CSV；所有结果仅写入一个Excel工作簿。"},
        {"section": "导出说明", "item": "工作簿结构", "value": "每个核心结果一张sheet；sheet名会自动裁剪并清洗非法字符。"},
    ]
    return pd.DataFrame(rows)


def _sanitize_sheet_name(name: str, used: set[str]) -> str:
    clean = str(name)
    for ch in [":", "\\", "/", "?", "*", "[", "]"]:
        clean = clean.replace(ch, "_")
    clean = clean[:31] if clean else "Sheet"
    base = clean
    i = 2
    while clean in used:
        suffix = f"_{i}"
        clean = f"{base[:31-len(suffix)]}{suffix}"
        i += 1
    used.add(clean)
    return clean


def export_results_workbook(result_tables: Dict[str, pd.DataFrame], out_dir: str, workbook_name: str = "swing20_experiment_report.xlsx") -> str:
    workbook_path = os.path.join(out_dir, workbook_name)
    used: set[str] = set()
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        _build_excel_summary_sheet().to_excel(writer, sheet_name=_sanitize_sheet_name("README", used), index=False)
        for sheet, df in result_tables.items():
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=_sanitize_sheet_name(sheet, used), index=False)
    return workbook_path

def main() -> None:
    parser = argparse.ArgumentParser(description="买卖点质量研究脚本（缓存增强 + 单Excel导出）")
    parser.add_argument("--n-stocks", type=int, default=100)
    parser.add_argument("--bars", type=int, default=800)
    parser.add_argument("--freq", default="d")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis-mode", type=str, default="quality_full", choices=["light", "full", "prepare_only", "model_only", "rule_only", "entry_profile", "exit_matrix", "score_oos", "swing_full", "quality_full"])
    parser.add_argument("--neighbor-dsa-max", type=float, default=0.45)
    parser.add_argument("--neighbor-prev-down-min", type=float, default=8.0)
    parser.add_argument("--neighbor-current-run-max", type=float, default=20.0)
    parser.add_argument("--neighbor-bars-since-max", type=float, default=12.0)
    parser.add_argument("--cache-dir", type=str, default="data_cache", help="数据缓存目录")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存，每次都从数据库重新计算")
    parser.add_argument("--refresh-cache", action="store_true", help="强制刷新缓存")
    parser.add_argument("--excel-name", type=str, default="buy_sell_quality_report.xlsx", help="唯一导出的Excel文件名")
    args = parser.parse_args()

    print("研究目标: 三层事件池池内质量归因 + 纯结构主线/动量对照辅线 + 固定窗口仅作观察 + 仅输出一个Excel")
    cache_path = get_cache_path(args.cache_dir, args.n_stocks, args.bars, args.freq, args.seed)

    all_dfs: List[pd.DataFrame] = []
    if not args.no_cache and not args.refresh_cache and cache_exists(cache_path):
        print(f"\n发现缓存文件，正在加载: {cache_path.replace('.pkl', '.parquet')}")
        cached_data = load_cache(cache_path)
        if cached_data is not None:
            all_dfs = cached_data.get("all_dfs", [])
            metadata = cached_data.get("metadata", {})
            print(f"缓存加载成功: {len(all_dfs)}只股票, 创建时间: {metadata.get('created_at', 'unknown')}")
        else:
            print("缓存加载失败，将重新从数据库获取")

    if not all_dfs:
        print("\n从数据库获取数据并计算因子...")
        all_dfs = fetch_data_from_db(args.n_stocks, args.bars, args.freq, args.seed)
        if not args.no_cache and all_dfs:
            metadata = {
                "n_stocks": args.n_stocks,
                "bars": args.bars,
                "freq": args.freq,
                "seed": args.seed,
                "created_at": str(pd.Timestamp.now()),
                "stock_count": len(all_dfs),
                "sample_count": int(sum(len(df_part) for df_part in all_dfs)),
            }
            save_cache({"all_dfs": all_dfs, "metadata": metadata}, cache_path)
            print(f"缓存已保存: {cache_path.replace('.pkl', '.parquet')}")

    if not all_dfs:
        print("无有效数据，退出")
        return

    for i, df_part in enumerate(all_dfs):
        if "quality_trade_score" not in df_part.columns:
            all_dfs[i] = add_path_quality_labels(df_part, horizon=PATH_QUALITY_HORIZON)

    df = pd.concat(all_dfs, ignore_index=True)
    neighbor_spec = NeighborSpec(
        dsa_max=args.neighbor_dsa_max,
        prev_down_min=args.neighbor_prev_down_min,
        current_run_max=args.neighbor_current_run_max,
        bars_since_max=args.neighbor_bars_since_max,
    )
    events = build_neighbor_events(df, neighbor_spec)

    result_tables: Dict[str, pd.DataFrame] = {}
    result_tables["00_dataset_summary"] = analyze_00_dataset_summary(df, events)
    result_tables["01_feature_inventory"] = analyze_01_feature_inventory(events)
    result_tables["02_candidate_events"] = events.copy()

    print(f"\n总样本: {len(df)}, 股票数: {df['symbol'].nunique()}")
    print(f"固定主策略事件池: {len(events)}, 股票数: {events['symbol'].nunique() if not events.empty else 0}")
    if events.empty:
        print("候选事件为空，退出")
        return

    if args.analysis_mode == "prepare_only":
        workbook_path = export_results_workbook(result_tables, OUT_DIR, args.excel_name)
        print(f"\n已导出Excel工作簿: {workbook_path}")
        return

    if args.analysis_mode not in ["rule_only", "entry_profile", "exit_matrix", "score_oos"]:
        run_light = args.analysis_mode == "light"
        if run_light:
            trade_reg_metrics, trade_reg_imp = run_single_model(events, TRADE_FEATURES, "quality_trade_score", "reg", "trade")
            trade_clf_metrics = pd.DataFrame()
            trade_clf_imp = pd.DataFrame()
            research_reg_metrics = pd.DataFrame()
            research_reg_imp = pd.DataFrame()
            research_clf_metrics = pd.DataFrame()
            research_clf_imp = pd.DataFrame()
            fold_metrics_df = trade_reg_metrics.copy()
            trade_reg_imp_agg = aggregate_importance(trade_reg_imp)
            trade_clf_imp_agg = pd.DataFrame()
            research_reg_imp_agg = pd.DataFrame()
            research_clf_imp_agg = pd.DataFrame()
            result_tables["03_trade_reg_importance"] = trade_reg_imp_agg
            result_tables["03b_trade_clf_importance"] = trade_clf_imp_agg
            result_tables["04_research_reg_importance"] = research_reg_imp_agg
            result_tables["04b_research_clf_importance"] = research_clf_imp_agg
            result_tables["05_fold_metrics"] = fold_metrics_df
        else:
            model_summary_df, model_metrics_df, model_importance_df, importance_spread_df = build_model_comparison_report(events)
            result_tables["03_model_summary"] = model_summary_df
            result_tables["05_fold_metrics"] = model_metrics_df
            result_tables["05b_importance_spread"] = importance_spread_df

            result_tables["03_trade_reg_importance"] = model_importance_df[
                (model_importance_df.get("mode") == "trade_with_momo") & (model_importance_df.get("model_type") == "reg")
            ].copy() if not model_importance_df.empty else pd.DataFrame()
            result_tables["03b_trade_clf_importance"] = model_importance_df[
                (model_importance_df.get("mode") == "trade_with_momo") & (model_importance_df.get("model_type") == "clf")
            ].copy() if not model_importance_df.empty else pd.DataFrame()
            result_tables["04_research_reg_importance"] = model_importance_df[
                (model_importance_df.get("mode") == "research") & (model_importance_df.get("model_type") == "reg")
            ].copy() if not model_importance_df.empty else pd.DataFrame()
            result_tables["04b_research_clf_importance"] = model_importance_df[
                (model_importance_df.get("mode") == "research") & (model_importance_df.get("model_type") == "clf")
            ].copy() if not model_importance_df.empty else pd.DataFrame()
            result_tables["04c_trade_no_momo_reg_importance"] = model_importance_df[
                (model_importance_df.get("mode") == "trade_without_momo") & (model_importance_df.get("model_type") == "reg")
            ].copy() if not model_importance_df.empty else pd.DataFrame()
            result_tables["04d_trade_no_momo_clf_importance"] = model_importance_df[
                (model_importance_df.get("mode") == "trade_without_momo") & (model_importance_df.get("model_type") == "clf")
            ].copy() if not model_importance_df.empty else pd.DataFrame()

            trade_top = result_tables["03_trade_reg_importance"].groupby("feature", as_index=False)["rank_score"].mean().sort_values("rank_score", ascending=False) if not result_tables["03_trade_reg_importance"].empty else pd.DataFrame()
            research_top = result_tables["04_research_reg_importance"].groupby("feature", as_index=False)["rank_score"].mean().sort_values("rank_score", ascending=False) if not result_tables["04_research_reg_importance"].empty else pd.DataFrame()
            trade_feats_for_bucket = trade_top["feature"].head(8).tolist() if not trade_top.empty else [f for f in PRIMARY_HINT_FEATURES if f in TRADE_FEATURES]
            research_feats_for_bucket = research_top["feature"].head(8).tolist() if not research_top.empty else [f for f in PRIMARY_HINT_FEATURES if f in RESEARCH_FEATURES]
            result_tables["06_trade_bucket_profiles"] = build_bucket_profiles(events, trade_feats_for_bucket, "")
            result_tables["06b_research_bucket_profiles"] = build_bucket_profiles(events, research_feats_for_bucket, "")
            result_tables["07_parameter_hints"] = build_parameter_hints(
                events,
                pd.concat([result_tables["03_trade_reg_importance"], result_tables["03b_trade_clf_importance"]], ignore_index=True) if not result_tables["03_trade_reg_importance"].empty or not result_tables["03b_trade_clf_importance"].empty else pd.DataFrame(columns=["feature", "rank_score"]),
                pd.concat([result_tables["04_research_reg_importance"], result_tables["04b_research_clf_importance"]], ignore_index=True) if not result_tables["04_research_reg_importance"].empty or not result_tables["04b_research_clf_importance"].empty else pd.DataFrame(columns=["feature", "rank_score"]),
            )
            result_tables["08_composite_feature_profiles"] = build_bucket_profiles(events, COMPOSITE_FEATURES, "")

            if not result_tables["03_trade_reg_importance"].empty:
                print("\n=== 关键结果预览 ===")
                print("\n[trade-safe(with_momo) 回归重要性 Top10]")
                cols = [c for c in ["feature", "rank_score", "gain_importance", "perm_importance", "fold_count"] if c in result_tables["03_trade_reg_importance"].columns]
                print(result_tables["03_trade_reg_importance"][cols].head(10).to_string(index=False))
    if args.analysis_mode in ["entry_profile", "swing_full", "quality_full"]:
        selected_wm, selected_wom, selected_summary = build_scored_entry_subsets(events, top_pct=0.2, min_group=8)
        entry_single_wm, entry_pair_wm, trigger_df = build_entry_profile_reports(
            selected_wm,
            features=SWING_PROFILE_FEATURES,
            profile_name="with_momo",
            apply_main_strategy=False,
        )
        entry_single_wom, entry_pair_wom, _ = build_entry_profile_reports(
            selected_wom,
            features=ENTRY_PROFILE_FEATURES_NO_MOMO,
            pair_defs=[
                ("dsa_trade_pos_01", "bb_width_norm"),
                ("dsa_trade_pos_01", "bb_width_change_5"),
                ("bars_since_dir_change", "bb_expand_streak"),
                ("w_dsa_trade_pos_01", "dsa_trade_pos_01"),
                ("trend_gap_10_20", "bb_pos_01"),
                ("score_width_total", "score_trend_total"),
            ],
            profile_name="without_momo",
            apply_main_strategy=False,
        )
        result_tables["20b_scored_subset_summary"] = selected_summary
        result_tables["20c_scored_subset_with_momo"] = selected_wm
        result_tables["20d_scored_subset_no_momo"] = selected_wom
        result_tables["21_entry_single_with_momo"] = entry_single_wm
        result_tables["21b_entry_single_no_momo"] = entry_single_wom
        result_tables["22_entry_pair_with_momo"] = entry_pair_wm
        result_tables["22b_entry_pair_no_momo"] = entry_pair_wom
        result_tables["23_trigger_summary"] = trigger_df
        result_tables["23c_entry_momo_compare"] = build_entry_momo_compare(entry_single_wm, entry_single_wom)
        if not entry_single_wm.empty:
            print("\n[买点画像 Top10 | with_momo]")
            print(entry_single_wm.head(10).to_string(index=False))
        if not entry_single_wom.empty:
            print("\n[买点画像 Top10 | without_momo]")
            print(entry_single_wom.head(10).to_string(index=False))

    if args.analysis_mode in ["exit_matrix", "swing_full", "quality_full"]:
        exit_summary_df, exit_detail_df = build_exit_matrix_report(df, events)
        result_tables["24_event_exit_summary"] = exit_summary_df
        result_tables["24b_event_exit_details"] = exit_detail_df
        if not exit_summary_df.empty:
            print("\n[事件卖点比较]")
            print(exit_summary_df.to_string(index=False))

    if args.analysis_mode in ["score_oos", "swing_full", "quality_full"]:
        score_oos_df, score_oos_picks_df = build_score_oos_report(events)
        result_tables["25_score_oos_summary"] = score_oos_df
        result_tables["25b_score_oos_picks"] = score_oos_picks_df
        if not score_oos_df.empty:
            print("\n[月度Top20%打分OOS]")
            print(score_oos_df.to_string(index=False))

    launch_scan_df, launch_top_df = build_compression_launch_scan(events)
    stability_df = build_stability_slices(events, launch_scan_df) if not launch_scan_df.empty else pd.DataFrame()
    refined_df, refined_top_df, summary_df = build_refined_rule_scan(events)
    result_tables["09_launch_scan"] = launch_scan_df
    result_tables["09b_launch_top_events"] = launch_top_df
    result_tables["10_rule_stability"] = stability_df
    result_tables["11_refined_rule_scan"] = refined_df
    result_tables["11b_refined_top_events"] = refined_top_df
    result_tables["12_rule_summary"] = summary_df

    if not refined_df.empty:
        print("\n[榜首规则 Top10]")
        cols = [c for c in [
            "stage", "dsa_confirmed_pos_max", "dsa_trade_pos_max", "bb_width_norm_max",
            "bb_pos_min", "bb_pos_max", "rope_pivot_pos_max", "weekly_mode", "trigger_type",
            "n", "quality_trade_score", f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}", f"path_efficiency_{PATH_QUALITY_HORIZON}", f"path_mae_{PATH_QUALITY_HORIZON}", "score_ret_20", "year_rr_std", "rank_score"
        ] if c in refined_df.columns]
        print(refined_df[cols].head(10).to_string(index=False))
    if not summary_df.empty:
        print("\n[主规则候选摘要 Top5]")
        print(summary_df.head(5).to_string(index=False))

    workbook_path = export_results_workbook(result_tables, OUT_DIR, args.excel_name)
    print(f"\n已导出Excel工作簿: {workbook_path}")




# =========================
# V7: 池内重要性归因 + 放宽门槛组合探索
# =========================

ATTRIBUTION_TARGET_FEATURES = {
    "trade_without_momo": TRADE_FEATURES_NO_MOMO,
    "trade_with_momo": TRADE_FEATURES,
    "research": RESEARCH_FEATURES,
}


def build_pool_variants(df: pd.DataFrame, base_spec: NeighborSpec) -> Dict[str, pd.DataFrame]:
    variants: Dict[str, NeighborSpec] = {
        "pool_base": NeighborSpec(
            dsa_max=base_spec.dsa_max,
            prev_down_min=base_spec.prev_down_min,
            need_rope_up=base_spec.need_rope_up,
            current_run_max=base_spec.current_run_max,
            bars_since_max=base_spec.bars_since_max,
        ),
        "pool_relaxed_soft": NeighborSpec(
            dsa_max=min(0.75, float(base_spec.dsa_max) + 0.05),
            prev_down_min=max(4.0, float(base_spec.prev_down_min) - 2.0),
            need_rope_up=base_spec.need_rope_up,
            current_run_max=float(base_spec.current_run_max) + 4.0,
            bars_since_max=float(base_spec.bars_since_max) + 3.0,
        ),
        "pool_relaxed_wide": NeighborSpec(
            dsa_max=min(0.85, float(base_spec.dsa_max) + 0.10),
            prev_down_min=max(2.0, float(base_spec.prev_down_min) - 4.0),
            need_rope_up=False,
            current_run_max=float(base_spec.current_run_max) + 8.0,
            bars_since_max=float(base_spec.bars_since_max) + 6.0,
        ),
    }
    pools: Dict[str, pd.DataFrame] = {}
    for pool_name, spec in variants.items():
        pool = build_neighbor_events(df, spec).copy()
        if not pool.empty:
            pool["pool_name"] = pool_name
            pool = add_trigger_labels(pool)
        pools[pool_name] = pool
    return pools


def summarize_pool_variants(df: pd.DataFrame, pools: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for pool_name, pool in pools.items():
        row: Dict[str, object] = {
            "pool_name": pool_name,
            "sample_n": int(len(pool)),
            "stock_n": int(pool["symbol"].nunique()) if not pool.empty and "symbol" in pool.columns else 0,
            "pool_ratio_vs_global": round(float(len(pool) / len(df)), 6) if len(df) > 0 else np.nan,
        }
        row.update(summarize_quality(pool))
        score_cols = summarize_score_windows(pool, [10, 20, 40])
        for k, v in score_cols.items():
            row[f"obs_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sample_n", "quality_trade_score"], ascending=[False, False])


def summarize_pool_trigger_mix(pools: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for pool_name, pool in pools.items():
        if pool.empty:
            continue
        pool2 = add_trigger_labels(pool)
        for trig, g in pool2.groupby("trigger_type", sort=True):
            row = {
                "pool_name": pool_name,
                "trigger_type": trig,
                "sample_n": int(len(g)),
                "sample_ratio": round(float(len(g) / len(pool2)), 5),
            }
            row.update(summarize_quality(g))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["pool_name", "quality_trade_score", "sample_n"], ascending=[True, False, False])


def _build_binary_target_subset(events: pd.DataFrame, target_name: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    sub = events.copy()
    q_col = "quality_trade_score"
    hit5_col = f"hit_5pct_before_stop_{PATH_QUALITY_HORIZON}"
    bars5_col = f"bars_to_5pct_{PATH_QUALITY_HORIZON}"
    eff_col = f"path_efficiency_{PATH_QUALITY_HORIZON}"
    mfe_col = f"path_mfe_{PATH_QUALITY_HORIZON}"
    mae_col = f"path_mae_{PATH_QUALITY_HORIZON}"

    asym = safe_numeric(sub[mfe_col]) / safe_numeric(sub[mae_col]).abs().clip(lower=0.01)
    sub["asymmetry_score"] = asym.replace([np.inf, -np.inf], np.nan)

    if target_name == "quality_top_vs_bottom":
        lo = safe_numeric(sub[q_col]).quantile(0.35)
        hi = safe_numeric(sub[q_col]).quantile(0.65)
        mask = (safe_numeric(sub[q_col]) <= lo) | (safe_numeric(sub[q_col]) >= hi)
        out = sub[mask].copy()
        out[target_name] = (safe_numeric(out[q_col]) >= hi).astype(int)
        return out

    if target_name == "asymmetry_top_vs_bottom":
        lo = safe_numeric(sub["asymmetry_score"]).quantile(0.35)
        hi = safe_numeric(sub["asymmetry_score"]).quantile(0.65)
        mask = (safe_numeric(sub["asymmetry_score"]) <= lo) | (safe_numeric(sub["asymmetry_score"]) >= hi)
        out = sub[mask].copy()
        out[target_name] = (safe_numeric(out["asymmetry_score"]) >= hi).astype(int)
        return out

    if target_name == "fast_confirm_vs_fail":
        pos_base = sub[safe_numeric(sub[hit5_col]) == 1.0].copy() if hit5_col in sub.columns else pd.DataFrame()
        if pos_base.empty:
            return pd.DataFrame()
        fast_cut = safe_numeric(pos_base[bars5_col]).dropna().quantile(0.45) if bars5_col in pos_base.columns and pos_base[bars5_col].notna().any() else np.nan
        pos_mask = (safe_numeric(sub[hit5_col]) == 1.0)
        if pd.notna(fast_cut) and bars5_col in sub.columns:
            pos_mask &= safe_numeric(sub[bars5_col]) <= fast_cut
        neg_mask = (safe_numeric(sub[hit5_col]).fillna(0.0) == 0.0)
        if eff_col in sub.columns:
            eff_cut = safe_numeric(sub[eff_col]).quantile(0.35)
            neg_mask |= safe_numeric(sub[eff_col]) <= eff_cut
        mask = pos_mask | neg_mask
        out = sub[mask].copy()
        out[target_name] = pos_mask.loc[out.index].astype(int)
        return out

    return pd.DataFrame()


def aggregate_importance_with_context(imp_df: pd.DataFrame) -> pd.DataFrame:
    if imp_df.empty:
        return imp_df
    grp = imp_df.groupby(["pool_name", "target_name", "mode", "model_type", "feature"], as_index=False).agg(
        gain_importance=("gain_importance", "mean"),
        perm_importance=("perm_importance", "mean"),
        fold_count=("fold", "nunique"),
    )
    grp["gain_rank"] = grp.groupby(["pool_name", "target_name", "mode", "model_type"])["gain_importance"].rank(ascending=False, method="average")
    grp["perm_rank"] = grp.groupby(["pool_name", "target_name", "mode", "model_type"])["perm_importance"].rank(ascending=False, method="average", na_option="bottom")
    grp["rank_score"] = (
        normalize_01(-grp["gain_rank"], reverse=False) * 0.8 +
        normalize_01(-grp["perm_rank"].fillna(grp["perm_rank"].max()), reverse=False) * 0.2
    )
    grp = grp.sort_values(["pool_name", "target_name", "mode", "model_type", "rank_score"], ascending=[True, True, True, True, False])
    return grp


def build_pool_attribution_reports(pools: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames: List[pd.DataFrame] = []
    imp_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    targets = [
        ("quality_reg", "reg", None),
        ("quality_top_vs_bottom", "clf", "quality_top_vs_bottom"),
        ("asymmetry_top_vs_bottom", "clf", "asymmetry_top_vs_bottom"),
        ("fast_confirm_vs_fail", "clf", "fast_confirm_vs_fail"),
    ]

    for pool_name, pool in pools.items():
        if pool.empty or len(pool) < 40:
            continue
        for mode_name, feats in ATTRIBUTION_TARGET_FEATURES.items():
            for target_alias, model_type, subset_target in targets:
                if model_type == "reg":
                    run_df = pool.copy()
                    target_col = "quality_trade_score"
                else:
                    run_df = _build_binary_target_subset(pool, subset_target)
                    target_col = subset_target
                if run_df.empty or len(run_df) < 36:
                    continue
                metrics_df, imp_df = run_single_model(run_df, feats, target_col, model_type, mode_name)
                if not metrics_df.empty:
                    metrics_df = metrics_df.copy()
                    metrics_df["pool_name"] = pool_name
                    metrics_df["target_name"] = target_alias
                    metric_frames.append(metrics_df)
                    summary_rows.append({
                        "pool_name": pool_name,
                        "target_name": target_alias,
                        "feature_mode": mode_name,
                        "model_type": model_type,
                        "sample_n": int(len(run_df)),
                        "feature_count": int(len([f for f in feats if f in run_df.columns])),
                        "mean_pred_corr": round(float(metrics_df["pred_corr"].mean()), 4) if "pred_corr" in metrics_df.columns and metrics_df["pred_corr"].notna().any() else np.nan,
                        "mean_auc": round(float(metrics_df["auc"].mean()), 4) if "auc" in metrics_df.columns and metrics_df["auc"].notna().any() else np.nan,
                    })
                if not imp_df.empty:
                    imp_df = imp_df.copy()
                    imp_df["pool_name"] = pool_name
                    imp_df["target_name"] = target_alias
                    imp_df["model_type"] = model_type
                    imp_frames.append(imp_df)
    metrics_all = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    imp_all = pd.concat(imp_frames, ignore_index=True) if imp_frames else pd.DataFrame()
    imp_agg = aggregate_importance_with_context(imp_all) if not imp_all.empty else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values(["pool_name", "target_name", "feature_mode", "model_type"]) if summary_rows else pd.DataFrame()
    return summary_df, metrics_all, imp_agg


def _infer_quality_masks(events: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    score = safe_numeric(events["quality_trade_score"])
    hi = score.quantile(0.65)
    lo = score.quantile(0.35)
    return score >= hi, score <= lo


def _infer_feature_direction(events: pd.DataFrame, feature: str) -> str:
    hi_mask, lo_mask = _infer_quality_masks(events)
    x = safe_numeric(events[feature])
    hi_mean = safe_numeric(events.loc[hi_mask, feature]).mean()
    lo_mean = safe_numeric(events.loc[lo_mask, feature]).mean()
    if pd.isna(hi_mean) or pd.isna(lo_mean):
        return "low"
    return "high" if hi_mean > lo_mean else "low"


def _apply_feature_directional_filter(events: pd.DataFrame, feature: str, direction: str, q: float = 0.4) -> pd.DataFrame:
    if feature not in events.columns:
        return events.iloc[0:0].copy()
    x = safe_numeric(events[feature])
    if direction == "high":
        thr = x.quantile(1.0 - q)
        return events[x >= thr].copy()
    thr = x.quantile(q)
    return events[x <= thr].copy()


def _select_combo_feature_pool(pool: pd.DataFrame, sub_imp: pd.DataFrame, mode_name: str, top_n: int = 8) -> List[str]:
    ranked = [f for f in sub_imp.sort_values("rank_score", ascending=False)["feature"].tolist() if f in pool.columns]
    if mode_name == "trade_without_momo":
        ranked = [f for f in ranked if f in STRUCTURE_CORE_FEATURES]
        ordered = ranked[:top_n]
        fallback = [f for f in STRUCTURE_CORE_FEATURES if f in pool.columns and f not in ordered]
        ordered.extend(fallback[:max(0, top_n - len(ordered))])
        return ordered[:top_n]
    return ranked[:top_n]


def _build_combo_candidates(top_feats: List[str], mode_name: str) -> List[Tuple[str, ...]]:
    if len(top_feats) < 2:
        return []
    combos: List[Tuple[str, ...]] = []
    if mode_name == "trade_without_momo":
        filtered = [f for f in top_feats if f in STRUCTURE_PAIR_FEATURES][:8]
        for r in [1, 2, 3]:
            combos.extend(list(combinations(filtered, r)))
    else:
        filtered = top_feats[:max(4, min(len(top_feats), 6))]
        for r in [1, 2, 3]:
            combos.extend(list(combinations(filtered, r)))
    return combos


def build_combo_scan_from_importance(pools: Dict[str, pd.DataFrame], imp_agg: pd.DataFrame, top_n: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    top_event_rows: List[pd.DataFrame] = []
    if imp_agg.empty:
        return pd.DataFrame(), pd.DataFrame()

    for pool_name, pool in pools.items():
        if pool.empty or len(pool) < 30:
            continue
        for mode_name in ["trade_without_momo", "trade_with_momo"]:
            sub_imp = imp_agg[
                (imp_agg["pool_name"] == pool_name) &
                (imp_agg["mode"] == mode_name) &
                (imp_agg["target_name"] == "quality_top_vs_bottom") &
                (imp_agg["model_type"] == "clf")
            ].copy()
            if sub_imp.empty:
                continue
            top_feats = _select_combo_feature_pool(pool, sub_imp, mode_name, top_n=top_n)
            if len(top_feats) < 2:
                continue
            directions = {f: _infer_feature_direction(pool, f) for f in top_feats}
            combos = _build_combo_candidates(top_feats, mode_name)
            for combo in combos:
                sub = pool.copy()
                rules = []
                for feat in combo:
                    direction = directions.get(feat, "low")
                    sub = _apply_feature_directional_filter(sub, feat, direction, q=0.4)
                    rules.append(f"{feat}_{'high' if direction == 'high' else 'low40'}")
                    if len(sub) < max(10, int(len(pool) * 0.04)):
                        break
                if len(sub) < max(10, int(len(pool) * 0.04)):
                    continue
                qstat = summarize_quality(sub)
                hi_mask, _ = _infer_quality_masks(pool)
                high_quality_cut = safe_numeric(pool["quality_trade_score"]).quantile(0.65)
                sub["is_high_quality_pool"] = (safe_numeric(sub["quality_trade_score"]) >= high_quality_cut).astype(int)
                row = {
                    "pool_name": pool_name,
                    "feature_mode": mode_name,
                    "combo_family": "structure" if mode_name == "trade_without_momo" else "all_features",
                    "combo_size": len(combo),
                    "combo_rule": " & ".join(rules),
                    "sample_n": int(len(sub)),
                    "sample_ratio": round(float(len(sub) / len(pool)), 5),
                    "high_quality_rate": round(float(sub["is_high_quality_pool"].mean()), 5),
                }
                row.update(qstat)
                row["rank_score"] = round(float(event_quality_rank(qstat, len(sub))), 6)
                rows.append(row)
                if len(rows) % 50 == 0:
                    pass
            if rows:
                pass
        # collect top events later
    combo_df = pd.DataFrame(rows)
    if combo_df.empty:
        return combo_df, pd.DataFrame()
    combo_df = combo_df.sort_values([
        "pool_name", "feature_mode", "rank_score", "high_quality_rate", "sample_n"
    ], ascending=[True, True, False, False, False]).reset_index(drop=True)

    for (pool_name, mode_name), g in combo_df.groupby(["pool_name", "feature_mode"], sort=False):
        top_rules = g.head(5)
        pool = pools.get(pool_name)
        if pool is None or pool.empty:
            continue
        for _, rule_row in top_rules.iterrows():
            sub = pool.copy()
            for token in str(rule_row["combo_rule"]).split(" & "):
                if token.endswith("_high"):
                    feat = token[:-5]
                    sub = _apply_feature_directional_filter(sub, feat, "high", q=0.4)
                elif token.endswith("_low40"):
                    feat = token[:-6]
                    sub = _apply_feature_directional_filter(sub, feat, "low", q=0.4)
            if sub.empty:
                continue
            top_events = sub.sort_values(["quality_trade_score", f"path_efficiency_{PATH_QUALITY_HORIZON}"], ascending=[False, False]).head(15).copy()
            top_events["pool_name"] = pool_name
            top_events["feature_mode"] = mode_name
            top_events["combo_rule"] = rule_row["combo_rule"]
            top_event_rows.append(top_events)
    top_events_df = pd.concat(top_event_rows, ignore_index=True) if top_event_rows else pd.DataFrame()
    return combo_df, top_events_df


def build_combo_stability_summary(combo_df: pd.DataFrame) -> pd.DataFrame:
    if combo_df.empty:
        return pd.DataFrame()
    work = combo_df.copy()
    keys = ["feature_mode", "combo_size", "combo_rule"]
    focus = work[work["feature_mode"].astype(str).str.contains("without_momo", na=False)].copy()
    if focus.empty:
        focus = work.copy()
    pivot = focus.pivot_table(
        index=keys,
        columns="pool_name",
        values="rank_score",
        aggfunc="mean",
    ).reset_index()
    for col in ["pool_base", "pool_relaxed_soft", "pool_relaxed_wide"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["base_soft_gap"] = (safe_numeric(pivot["pool_base"]) - safe_numeric(pivot["pool_relaxed_soft"])).abs()
    pivot["base_wide_decay"] = safe_numeric(pivot["pool_base"]) - safe_numeric(pivot["pool_relaxed_wide"])
    pivot["stability_score"] = (
        safe_numeric(pivot["pool_base"]).fillna(0) * 0.55
        + safe_numeric(pivot["pool_relaxed_soft"]).fillna(0) * 0.35
        + safe_numeric(pivot["pool_relaxed_wide"]).fillna(0) * 0.10
        - safe_numeric(pivot["base_soft_gap"]).fillna(0) * 0.25
        - safe_numeric(pivot["base_wide_decay"]).clip(lower=0).fillna(0) * 0.10
    )
    pivot["direction_consistency"] = np.where(
        safe_numeric(pivot["pool_base"]).notna() & safe_numeric(pivot["pool_relaxed_soft"]).notna(),
        1.0,
        0.0,
    )
    return pivot.sort_values(
        ["stability_score", "pool_base", "pool_relaxed_soft"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_pool_bucket_profiles(pools: Dict[str, pd.DataFrame], features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for pool_name, pool in pools.items():
        if pool.empty:
            continue
        for feat in features:
            if feat not in pool.columns:
                continue
            x = safe_numeric(pool[feat])
            valid = pool[x.notna()].copy()
            if len(valid) < 30:
                continue
            try:
                valid["bucket"] = pd.qcut(x[x.notna()], q=5, duplicates="drop")
            except Exception:
                continue
            for bucket, g in valid.groupby("bucket", observed=True):
                row = {
                    "pool_name": pool_name,
                    "feature": feat,
                    "bucket": str(bucket),
                    "sample_n": int(len(g)),
                    "feature_mean": round(float(safe_numeric(g[feat]).mean()), 6),
                }
                row.update(summarize_quality(g))
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["pool_name", "feature", "quality_trade_score"], ascending=[True, True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="严格结构主线收敛实验脚本（池内归因 + 纯结构组合 + 单Excel导出）")
    parser.add_argument("--n-stocks", type=int, default=100)
    parser.add_argument("--bars", type=int, default=800)
    parser.add_argument("--freq", default="d")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis-mode", type=str, default="pool_attribution", choices=["prepare_only", "pool_attribution"])
    parser.add_argument("--neighbor-dsa-max", type=float, default=0.45)
    parser.add_argument("--neighbor-prev-down-min", type=float, default=8.0)
    parser.add_argument("--neighbor-current-run-max", type=float, default=20.0)
    parser.add_argument("--neighbor-bars-since-max", type=float, default=12.0)
    parser.add_argument("--cache-dir", type=str, default="data_cache", help="数据缓存目录")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存，每次都从数据库重新计算")
    parser.add_argument("--refresh-cache", action="store_true", help="强制刷新缓存")
    parser.add_argument("--excel-name", type=str, default="event_pool_attribution_report.xlsx", help="唯一导出的Excel文件名")
    args = parser.parse_args()

    print("研究目标: 以三层事件池为框架，主归因切到无动量纯结构组合；pool_base负责发现，pool_relaxed_soft负责验证，pool_relaxed_wide负责压力测试；固定持有期只作观察，不作买卖依据。")
    cache_path = get_cache_path(args.cache_dir, args.n_stocks, args.bars, args.freq, args.seed)

    all_dfs: List[pd.DataFrame] = []
    if not args.no_cache and not args.refresh_cache and cache_exists(cache_path):
        print(f"\n发现缓存文件，正在加载: {cache_path.replace('.pkl', '.parquet')}")
        cached_data = load_cache(cache_path)
        if cached_data is not None:
            all_dfs = cached_data.get("all_dfs", [])
            metadata = cached_data.get("metadata", {})
            print(f"缓存加载成功: {len(all_dfs)}只股票, 创建时间: {metadata.get('created_at', 'unknown')}")
        else:
            print("缓存加载失败，将重新从数据库获取")

    if not all_dfs:
        print("\n从数据库获取数据并计算因子...")
        all_dfs = fetch_data_from_db(args.n_stocks, args.bars, args.freq, args.seed)
        if not args.no_cache and all_dfs:
            metadata = {
                "n_stocks": args.n_stocks,
                "bars": args.bars,
                "freq": args.freq,
                "seed": args.seed,
                "created_at": str(pd.Timestamp.now()),
                "stock_count": len(all_dfs),
                "sample_count": int(sum(len(df_part) for df_part in all_dfs)),
            }
            save_cache({"all_dfs": all_dfs, "metadata": metadata}, cache_path)
            print(f"缓存已保存: {cache_path.replace('.pkl', '.parquet')}")

    if not all_dfs:
        print("无有效数据，退出")
        return

    for i, df_part in enumerate(all_dfs):
        if "quality_trade_score" not in df_part.columns:
            all_dfs[i] = add_path_quality_labels(df_part, horizon=PATH_QUALITY_HORIZON)

    df = pd.concat(all_dfs, ignore_index=True)
    base_spec = NeighborSpec(
        dsa_max=args.neighbor_dsa_max,
        prev_down_min=args.neighbor_prev_down_min,
        current_run_max=args.neighbor_current_run_max,
        bars_since_max=args.neighbor_bars_since_max,
    )
    pools = build_pool_variants(df, base_spec)
    base_events = pools.get("pool_base", pd.DataFrame())

    result_tables: Dict[str, pd.DataFrame] = {}
    result_tables["00_pool_summary"] = summarize_pool_variants(df, pools)
    result_tables["01_feature_inventory"] = analyze_01_feature_inventory(base_events if not base_events.empty else df)
    result_tables["02_candidate_events_base"] = base_events.copy()
    result_tables["02b_pool_relaxed_soft"] = pools.get("pool_relaxed_soft", pd.DataFrame()).copy()
    result_tables["02c_pool_relaxed_wide"] = pools.get("pool_relaxed_wide", pd.DataFrame()).copy()
    result_tables["03_pool_trigger_mix"] = summarize_pool_trigger_mix(pools)

    print(f"\n总样本: {len(df)}, 股票数: {df['symbol'].nunique()}")
    for name, pool in pools.items():
        print(f"{name}: {len(pool)} 事件, 股票数={pool['symbol'].nunique() if not pool.empty else 0}")

    if args.analysis_mode == "prepare_only":
        workbook_path = export_results_workbook(result_tables, OUT_DIR, args.excel_name)
        print(f"\n已导出Excel工作簿: {workbook_path}")
        return

    attribution_summary, attribution_metrics, attribution_importance = build_pool_attribution_reports(pools)
    result_tables["10_attr_model_summary"] = attribution_summary
    result_tables["11_attr_fold_metrics"] = attribution_metrics
    result_tables["12_attr_importance"] = attribution_importance

    bucket_features = list(STRUCTURE_CORE_FEATURES)
    result_tables["13_pool_bucket_profiles"] = build_pool_bucket_profiles(pools, bucket_features)

    combo_df, combo_top_df = build_combo_scan_from_importance(pools, attribution_importance)
    combo_no_momo = combo_df[combo_df.get("feature_mode", pd.Series(dtype=object)).astype(str).str.contains("without_momo", na=False)].copy() if not combo_df.empty else pd.DataFrame()
    combo_no_momo_top = combo_top_df[combo_top_df.get("feature_mode", pd.Series(dtype=object)).astype(str).str.contains("without_momo", na=False)].copy() if not combo_top_df.empty else pd.DataFrame()
    combo_with_momo = combo_df[combo_df.get("feature_mode", pd.Series(dtype=object)).astype(str).str.contains("with_momo", na=False) & ~combo_df.get("feature_mode", pd.Series(dtype=object)).astype(str).str.contains("without_momo", na=False)].copy() if not combo_df.empty else pd.DataFrame()
    combo_with_momo_top = combo_top_df[combo_top_df.get("feature_mode", pd.Series(dtype=object)).astype(str).str.contains("with_momo", na=False) & ~combo_top_df.get("feature_mode", pd.Series(dtype=object)).astype(str).str.contains("without_momo", na=False)].copy() if not combo_top_df.empty else pd.DataFrame()
    combo_stability = build_combo_stability_summary(combo_df)
    result_tables["20_no_momo_struct_combo_scan"] = combo_no_momo
    result_tables["20b_no_momo_struct_top_events"] = combo_no_momo_top
    result_tables["20c_combo_stability_summary"] = combo_stability
    result_tables["21_with_momo_combo_scan"] = combo_with_momo
    result_tables["21b_with_momo_combo_top_events"] = combo_with_momo_top
    result_tables["22_combo_scan_all"] = combo_df

    if not attribution_summary.empty:
        print("\n[池内重要性归因摘要]")
        print(attribution_summary.head(20).to_string(index=False))
    if not combo_no_momo.empty:
        print("\n[纯结构组合探索 Top20]")
        print(combo_no_momo.head(20).to_string(index=False))
    elif not combo_df.empty:
        print("\n[组合探索 Top20]")
        print(combo_df.head(20).to_string(index=False))

    workbook_path = export_results_workbook(result_tables, OUT_DIR, args.excel_name)
    print(f"\n已导出Excel工作簿: {workbook_path}")


if __name__ == "__main__":
    main()
