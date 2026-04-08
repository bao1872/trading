# -*- coding: utf-8 -*-
"""
基于 GBDT 结果反推“压缩低位启动”规则的实验脚本（在 gbdt_buy_point_explorer.py 基础上扩展）

Purpose
- 保留原有框架：数据加载 → 因子计算 → forward returns → 候选事件池 → 报表输出
- 新增复合特征：结构位置 / 方向一致性 / 趋势分 / 量能分 / 宽度分
- 新增规则反推实验：围绕“压缩中的低位启动 + 周线不过弱”做参数扫描与切片验证
- 让更多可能影响因素先进入实验，再根据结果删除

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
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None  # type: ignore
    XGBRegressor = None  # type: ignore
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None  # type: ignore
    LGBMRegressor = None  # type: ignore
    HAS_LIGHTGBM = False

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

RET_WINDOWS = [5, 10, 20, 40, 60]
EVENT_DEDUP_BARS = 20
RANDOM_STATE = 42


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


def add_forward_returns(df: pd.DataFrame, windows: Sequence[int] = RET_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].to_numpy(float)
    low = out["low"].to_numpy(float)
    n = len(out)
    for w in windows:
        fut = np.full(n, np.nan)
        mae = np.full(n, np.nan)
        for i in range(n - w):
            entry = close[i]
            fut[i] = (close[i + w] - entry) / entry
            future_low = np.nanmin(low[i + 1: i + w + 1])
            mae_raw = (future_low - entry) / entry
            mae[i] = min(0.0, mae_raw)
        out[f"ret_{w}"] = fut
        out[f"max_dd_{w}"] = mae
        out[f"win_{w}"] = (out[f"ret_{w}"] > 0).astype(float)
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
    ]))
    need = [c for c in need if c in df.columns]
    sub = df[need].dropna(subset=[c for c in ["ret_20", "max_dd_20", "dsa_trade_pos_01", "rope_dir"] if c in need]).copy()
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
    events["rr_20"] = events.apply(lambda r: rr_from_ret_dd(r["ret_20"], r["max_dd_20"]), axis=1)
    med_ret20 = float(events["ret_20"].median()) if events["ret_20"].notna().any() else 0.0
    med_rr20 = float(events["rr_20"].dropna().median()) if events["rr_20"].notna().any() else 0.0
    events["good20"] = ((events["ret_20"] > med_ret20) & (events["rr_20"] > med_rr20)).astype(int)
    events["good20_hard"] = ((events["ret_20"] > 0.03) & (events["max_dd_20"] > -0.08)).astype(int)
    events["year"] = pd.to_datetime(events["datetime"]).dt.year
    events["year_month"] = pd.to_datetime(events["datetime"]).dt.to_period("M").astype(str)
    return events


def prepare_feature_matrix(events: pd.DataFrame, features: Sequence[str], fill_source: Optional[pd.DataFrame] = None, keep_feature_set: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    requested = list(dict.fromkeys(features))
    if not requested:
        return pd.DataFrame(index=events.index), []

    # 关键修复：测试集必须严格对齐训练集列集合，缺失列也要补出来，不能因为当前子集里没有该列就直接丢掉。
    x = events.reindex(columns=requested).copy()
    for col in requested:
        x[col] = safe_numeric(x[col])
        if fill_source is not None:
            base = fill_source[col] if col in fill_source.columns else pd.Series(dtype=float)
            med = safe_numeric(base).median() if len(base) > 0 else np.nan
        else:
            med = x[col].median()
        if pd.isna(med):
            med = 0.0
        x[col] = x[col].fillna(med)

    if keep_feature_set:
        return x[requested], requested

    nunique = x.nunique(dropna=False)
    cols = [c for c in requested if nunique.get(c, 0) > 1]
    return x[cols], cols


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


def build_highperf_model(model_type: str, random_state: int = RANDOM_STATE):
    """
    高性能 GBDT 后端：优先 xgboost，其次 lightgbm，最后 sklearn HistGB。
    返回 (backend_name, model)
    """
    if model_type == "reg":
        if HAS_XGBOOST:
            return "xgboost", XGBRegressor(
                random_state=random_state,
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                min_child_weight=8,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
            )
        if HAS_LIGHTGBM:
            return "lightgbm", LGBMRegressor(
                random_state=random_state,
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="regression",
                n_jobs=-1,
                verbosity=-1,
            )
        return "sklearn_histgb", HistGradientBoostingRegressor(
            random_state=random_state,
            learning_rate=0.03,
            max_iter=400,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
    else:
        if HAS_XGBOOST:
            return "xgboost", XGBClassifier(
                random_state=random_state,
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                min_child_weight=8,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_jobs=-1,
            )
        if HAS_LIGHTGBM:
            return "lightgbm", LGBMClassifier(
                random_state=random_state,
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="binary",
                n_jobs=-1,
                verbosity=-1,
            )
        return "sklearn_histgb", HistGradientBoostingClassifier(
            random_state=random_state,
            learning_rate=0.03,
            max_iter=400,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )


def collect_importance_rows(model_name: str, mode: str, fold_name: str, features: List[str], model, x_test: pd.DataFrame, y_test: pd.Series) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    base_importance = getattr(model, "feature_importances_", None)
    if base_importance is None:
        base_importance = np.zeros(len(features), dtype=float)
    perm_values = np.full(len(features), np.nan, dtype=float)
    if len(x_test) >= 25 and y_test.nunique(dropna=True) > 1:
        try:
            perm = permutation_importance(model, x_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
            perm_values = perm.importances_mean
        except Exception:
            pass
    for i, feat in enumerate(features):
        rows.append({
            "model_name": model_name,
            "mode": mode,
            "fold": fold_name,
            "feature": feat,
            "gain_importance": round(float(base_importance[i]), 6),
            "perm_importance": round(float(perm_values[i]), 6) if pd.notna(perm_values[i]) else np.nan,
        })
    return rows


def run_single_model(events: pd.DataFrame, features: Sequence[str], target_col: str, model_type: str, feature_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folds = make_time_folds(events)
    metrics_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []
    for fold_name, train_idx, test_idx in folds:
        train_df = events.loc[train_idx].copy()
        test_df = events.loc[test_idx].copy()
        x_train, used_features = prepare_feature_matrix(train_df, features, fill_source=train_df, keep_feature_set=False)
        x_test, _ = prepare_feature_matrix(test_df, used_features, fill_source=train_df, keep_feature_set=True)
        if len(used_features) < 3:
            continue
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        backend_name, model = build_highperf_model(model_type, random_state=RANDOM_STATE)
        if model_type == "reg":
            model.fit(x_train, safe_numeric(y_train))
            y_pred = model.predict(x_test)
            score = regression_score(y_test, y_pred)
        else:
            if pd.Series(y_train).nunique() < 2 or pd.Series(y_test).nunique() < 1:
                continue
            model.fit(x_train, pd.Series(y_train).astype(int))
            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(x_test)[:, 1]
            else:
                raw_pred = model.predict(x_test)
                y_pred = np.asarray(raw_pred, dtype=float)
            score = classification_score(y_test, y_pred)
        metrics_rows.append({"feature_mode": feature_mode, "target": target_col, "model_type": model_type, "model_backend": backend_name, "fold": fold_name, "train_n": len(train_df), "test_n": len(test_df), **score})
        importance_rows.extend(collect_importance_rows(f"{feature_mode}_{target_col}_{model_type}_{backend_name}", feature_mode, fold_name, used_features, model, x_test, pd.Series(y_test).astype(int) if model_type == "clf" else safe_numeric(y_test)))
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
        sub = events[[feat, "ret_20", "max_dd_20", "win_20", "rr_20"]].copy()
        sub[feat] = safe_numeric(sub[feat])
        sub = sub.dropna(subset=[feat, "ret_20", "max_dd_20"])
        if len(sub) < 40 or sub[feat].nunique() < 4:
            continue
        try:
            sub["bucket"] = pd.qcut(sub[feat], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        except Exception:
            continue
        grp = sub.groupby("bucket", observed=True).agg(n=(feat, "count"), mean_value=(feat, "mean"), ret20=("ret_20", "mean"), dd20=("max_dd_20", "mean"), wr20=("win_20", "mean"), rr20=("rr_20", "mean")).reset_index()
        best_row = grp.sort_values(["rr20", "ret20"], ascending=[False, False]).iloc[0]
        for _, row in grp.iterrows():
            rows.append({
                "feature": feat,
                "bucket": row["bucket"],
                "n": int(row["n"]),
                "mean_value": round(float(row["mean_value"]), 6),
                "ret20": round(float(row["ret20"]), 5),
                "dd20": round(float(row["dd20"]), 5),
                "wr20": round(float(row["wr20"]), 4),
                "rr20": round(float(row["rr20"]), 3),
                "is_best_bucket": int(row["bucket"] == best_row["bucket"]),
            })
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        rdf.to_csv(out_path, index=False)
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
            tmp = events[[feat, "ret_20", "max_dd_20", "win_20", "rr_20"]].copy()
            tmp[feat] = safe_numeric(tmp[feat])
            tmp = tmp.dropna(subset=[feat, "ret_20", "max_dd_20"])
            tmp["bucket"] = pd.qcut(tmp[feat], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
            grp = tmp.groupby("bucket", observed=True).agg(ret20=("ret_20", "mean"), rr20=("rr_20", "mean"), n=(feat, "count"))
            best_bucket = grp.sort_values(["rr20", "ret20"], ascending=[False, False]).index[0]
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
    if not rdf.empty:
        rdf.to_csv(f"{OUT_DIR}/07_parameter_hints.csv", index=False)
    return rdf


def build_compression_launch_scan(events: pd.DataFrame, top_k_export: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    围绕本轮 GBDT 提示的主线做规则扫描：
    - 低位，但不是绝对底部
    - BB 压缩
    - Rope/结构位置仍处在早中段
    - 周线不能太弱
    - 可选用 breakout / expand / range 宽度做二级过滤
    """
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: List[Dict[str, object]] = []
    top_event_rows: List[pd.DataFrame] = []

    # 主轴参数：优先围绕 GBDT 的 parameter hints 附近做扫描
    dsa_grid = [0.30, 0.35, 0.40, 0.45]
    bb_width_norm_grid = [0.10, 0.12, 0.14, 0.16, 0.18]
    bb_pos_low_grid = [0.45, 0.50, 0.55, 0.58]
    bb_pos_high_grid = [0.72, 0.78, 0.84, 0.90]
    rope_pivot_grid = [0.60, 0.70, 0.80, 0.90]
    weekly_dev_min_grid = [-1.0, -0.75, -0.50, -0.25]
    range_width_min_grid = [None, 1.5, 2.0, 2.5]
    breakout_modes = ["none", "expand_or_break", "break_only"]
    weekly_modes = ["soft", "strict"]
    ranking_modes = ["plain", "with_quality"]

    for dsa_max, bbw_max, bb_low, bb_high, rope_pivot_max, weekly_dev_min, range_width_min, breakout_mode, weekly_mode, ranking_mode in product(
        dsa_grid, bb_width_norm_grid, bb_pos_low_grid, bb_pos_high_grid, rope_pivot_grid, weekly_dev_min_grid, range_width_min_grid, breakout_modes, weekly_modes, ranking_modes
    ):
        if bb_high <= bb_low:
            continue
        sub = events.copy()
        # 低位结构（trade-safe）
        if "dsa_trade_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_trade_pos_01"]) <= dsa_max]
        # BB 局部压缩
        if "bb_width_norm" in sub.columns:
            sub = sub[safe_numeric(sub["bb_width_norm"]) <= bbw_max]
        # BB 中段启动，而不是绝对底/顶
        if "bb_pos_01" in sub.columns:
            bbpos = safe_numeric(sub["bb_pos_01"])
            sub = sub[bbpos.between(bb_low, bb_high, inclusive="both")]
        # Rope 结构位置仍早中段
        if "rope_pivot_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["rope_pivot_pos_01"]) <= rope_pivot_max]

        # 周线过滤：软版允许 trade-safe 代理，严版要求 confirmed 周偏离不过弱
        if weekly_mode == "soft":
            cond = pd.Series(True, index=sub.index)
            if "w_dsa_signed_vwap_dev_pct" in sub.columns:
                cond &= safe_numeric(sub["w_dsa_signed_vwap_dev_pct"]).fillna(-99) >= weekly_dev_min
            if "w_factor_available" in sub.columns:
                cond &= safe_numeric(sub["w_factor_available"]).fillna(0) > 0
            sub = sub[cond]
        else:
            cond = pd.Series(True, index=sub.index)
            if "w_dsa_signed_vwap_dev_pct" in sub.columns:
                cond &= safe_numeric(sub["w_dsa_signed_vwap_dev_pct"]).fillna(-99) >= weekly_dev_min
            if "w_dsa_trade_pos_01" in sub.columns:
                cond &= safe_numeric(sub["w_dsa_trade_pos_01"]).fillna(1) <= 0.70
            if "w_DSA_DIR" in sub.columns:
                cond &= safe_numeric(sub["w_DSA_DIR"]).fillna(-1) >= 0
            sub = sub[cond]

        # 整体结构仍有展开空间，不要太死窄
        if range_width_min is not None and "range_width_atr" in sub.columns:
            sub = sub[safe_numeric(sub["range_width_atr"]) >= range_width_min]

        # 启动确认
        if breakout_mode == "expand_or_break":
            cond = pd.Series(False, index=sub.index)
            for col in ["bb_expanding", "range_break_up", "dsa_trade_breakout_20"]:
                if col in sub.columns:
                    cond |= safe_numeric(sub[col]).fillna(0) > 0
            sub = sub[cond]
        elif breakout_mode == "break_only":
            cond = pd.Series(False, index=sub.index)
            for col in ["range_break_up", "dsa_trade_breakout_20"]:
                if col in sub.columns:
                    cond |= safe_numeric(sub[col]).fillna(0) > 0
            sub = sub[cond]

        # 仍保留“不要太晚”约束，但不再把 rope 当主角
        if "bars_since_dir_change" in sub.columns:
            sub = sub[safe_numeric(sub["bars_since_dir_change"]).fillna(99) <= 12]

        sub = dedup_events(sub)
        if len(sub) < 20:
            continue

        stat = summarize_events(sub, [20, 60])
        quality = 0.0
        if ranking_mode == "with_quality":
            pieces = []
            for c, reverse in [("score_width_total", False), ("score_trend_total", False), ("score_volume_total", False), ("score_setup_total", False)]:
                if c in sub.columns:
                    pieces.append(float(safe_numeric(sub[c]).mean()))
            quality = float(np.nanmean(pieces)) if pieces else 0.0
        rank_score = (
            100 * (stat.get("rr_20") or 0)
            + 12 * (stat.get("ret_20") or 0)
            + 4 * (stat.get("wr_20") or 0)
            + 10 * (stat.get("rr_60") or 0)
            + 3 * quality
            + min(len(sub), 120) / 12
        )
        rows.append({
            "dsa_trade_pos_max": dsa_max,
            "bb_width_norm_max": bbw_max,
            "bb_pos_low": bb_low,
            "bb_pos_high": bb_high,
            "rope_pivot_pos_max": rope_pivot_max,
            "w_dsa_signed_vwap_dev_min": weekly_dev_min,
            "range_width_atr_min": range_width_min,
            "breakout_mode": breakout_mode,
            "weekly_mode": weekly_mode,
            "ranking_mode": ranking_mode,
            **stat,
            "quality_score": round(float(quality), 4),
            "rank_score": round(float(rank_score), 3),
        })

    scan_df = pd.DataFrame(rows).sort_values(["rank_score", "rr_20", "ret_20", "n"], ascending=[False, False, False, False]).reset_index(drop=True) if rows else pd.DataFrame()
    if not scan_df.empty:
        scan_df.to_csv(f"{OUT_DIR}/09_compression_launch_scan.csv", index=False)

    for rank, row in scan_df.head(20).iterrows() if not scan_df.empty else []:
        sub = events.copy()
        if "dsa_trade_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_trade_pos_01"]) <= row["dsa_trade_pos_max"]]
        if "bb_width_norm" in sub.columns:
            sub = sub[safe_numeric(sub["bb_width_norm"]) <= row["bb_width_norm_max"]]
        if "bb_pos_01" in sub.columns:
            bbpos = safe_numeric(sub["bb_pos_01"])
            sub = sub[bbpos.between(row["bb_pos_low"], row["bb_pos_high"], inclusive="both")]
        if "rope_pivot_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["rope_pivot_pos_01"]) <= row["rope_pivot_pos_max"]]
        if row["weekly_mode"] == "soft":
            cond = pd.Series(True, index=sub.index)
            if "w_dsa_signed_vwap_dev_pct" in sub.columns:
                cond &= safe_numeric(sub["w_dsa_signed_vwap_dev_pct"]).fillna(-99) >= row["w_dsa_signed_vwap_dev_min"]
            if "w_factor_available" in sub.columns:
                cond &= safe_numeric(sub["w_factor_available"]).fillna(0) > 0
            sub = sub[cond]
        else:
            cond = pd.Series(True, index=sub.index)
            if "w_dsa_signed_vwap_dev_pct" in sub.columns:
                cond &= safe_numeric(sub["w_dsa_signed_vwap_dev_pct"]).fillna(-99) >= row["w_dsa_signed_vwap_dev_min"]
            if "w_dsa_trade_pos_01" in sub.columns:
                cond &= safe_numeric(sub["w_dsa_trade_pos_01"]).fillna(1) <= 0.70
            if "w_DSA_DIR" in sub.columns:
                cond &= safe_numeric(sub["w_DSA_DIR"]).fillna(-1) >= 0
            sub = sub[cond]
        if pd.notna(row["range_width_atr_min"]) and "range_width_atr" in sub.columns:
            sub = sub[safe_numeric(sub["range_width_atr"]) >= row["range_width_atr_min"]]
        if row["breakout_mode"] == "expand_or_break":
            cond = pd.Series(False, index=sub.index)
            for col in ["bb_expanding", "range_break_up", "dsa_trade_breakout_20"]:
                if col in sub.columns:
                    cond |= safe_numeric(sub[col]).fillna(0) > 0
            sub = sub[cond]
        elif row["breakout_mode"] == "break_only":
            cond = pd.Series(False, index=sub.index)
            for col in ["range_break_up", "dsa_trade_breakout_20"]:
                if col in sub.columns:
                    cond |= safe_numeric(sub[col]).fillna(0) > 0
            sub = sub[cond]
        if "bars_since_dir_change" in sub.columns:
            sub = sub[safe_numeric(sub["bars_since_dir_change"]).fillna(99) <= 12]
        sub = dedup_events(sub)
        if sub.empty:
            continue
        keep_cols = [c for c in [
            "symbol", "datetime", "close", "ret_20", "max_dd_20", "rr_20", "ret_60", "max_dd_60", "rr_60",
            "dsa_trade_pos_01", "bb_width_norm", "bb_pos_01", "rope_pivot_pos_01", "w_dsa_signed_vwap_dev_pct",
            "range_width_atr", "bb_expanding", "range_break_up", "dsa_trade_breakout_20",
            "score_width_total", "score_trend_total", "score_volume_total", "score_setup_total",
        ] if c in sub.columns]
        tmp = sub[keep_cols].copy()
        tmp["rule_rank"] = rank + 1
        top_event_rows.append(tmp)

    top_df = pd.concat(top_event_rows, ignore_index=True).head(top_k_export) if top_event_rows else pd.DataFrame()
    if not top_df.empty:
        top_df.to_csv(f"{OUT_DIR}/09b_compression_launch_top_events.csv", index=False)
    return scan_df, top_df


def build_stability_slices(events: pd.DataFrame, scan_df: pd.DataFrame) -> pd.DataFrame:
    if events.empty or scan_df.empty:
        return pd.DataFrame()
    top = scan_df.head(6)
    rows: List[Dict[str, object]] = []
    work = events.copy()
    # 粗略市值代理：用 close*vol 做成交额分层，环境代理：按全样本 ret_20 中位数分组
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
    if "ret_20" in work.columns and work["ret_20"].notna().sum() >= 30:
        med = float(work["ret_20"].median())
        work["env_bucket"] = np.where(work["ret_20"] >= med, "better", "worse")
    else:
        work["env_bucket"] = "all"

    for rank, row in top.iterrows():
        sub = work.copy()
        if "dsa_trade_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["dsa_trade_pos_01"]) <= row["dsa_trade_pos_max"]]
        if "bb_width_norm" in sub.columns:
            sub = sub[safe_numeric(sub["bb_width_norm"]) <= row["bb_width_norm_max"]]
        if "bb_pos_01" in sub.columns:
            bbpos = safe_numeric(sub["bb_pos_01"])
            sub = sub[bbpos.between(row["bb_pos_low"], row["bb_pos_high"], inclusive="both")]
        if "rope_pivot_pos_01" in sub.columns:
            sub = sub[safe_numeric(sub["rope_pivot_pos_01"]) <= row["rope_pivot_pos_max"]]
        if "w_dsa_signed_vwap_dev_pct" in sub.columns:
            sub = sub[safe_numeric(sub["w_dsa_signed_vwap_dev_pct"]).fillna(-99) >= row["w_dsa_signed_vwap_dev_min"]]
        if pd.notna(row["range_width_atr_min"]) and "range_width_atr" in sub.columns:
            sub = sub[safe_numeric(sub["range_width_atr"]) >= row["range_width_atr_min"]]
        if row["breakout_mode"] == "expand_or_break":
            cond = pd.Series(False, index=sub.index)
            for col in ["bb_expanding", "range_break_up", "dsa_trade_breakout_20"]:
                if col in sub.columns:
                    cond |= safe_numeric(sub[col]).fillna(0) > 0
            sub = sub[cond]
        elif row["breakout_mode"] == "break_only":
            cond = pd.Series(False, index=sub.index)
            for col in ["range_break_up", "dsa_trade_breakout_20"]:
                if col in sub.columns:
                    cond |= safe_numeric(sub[col]).fillna(0) > 0
            sub = sub[cond]
        sub = dedup_events(sub)
        if len(sub) < 12:
            continue
        for dim in ["year", "liquidity_bucket", "env_bucket"]:
            for key, g in sub.groupby(dim, dropna=False):
                if len(g) < 8:
                    continue
                stat = summarize_events(g, [20, 60])
                rows.append({
                    "rule_rank": rank + 1,
                    "slice_dim": dim,
                    "slice_key": str(key),
                    **stat,
                })
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        rdf.to_csv(f"{OUT_DIR}/10_rule_stability_slices.csv", index=False)
    return rdf


@dataclass
class StageSpec:
    name: str
    ret5_min: Optional[float] = None
    ret5_max: Optional[float] = None


@dataclass
class EnvGateSpec:
    name: str
    trend_min: Optional[float] = None
    weekly_dev_min: Optional[float] = None
    trend_gap_min: Optional[float] = None


@dataclass
class FixedRuleSpec:
    name: str = "fixed_rule_v1"
    dsa_trade_pos_max: float = 0.40
    dsa_confirmed_pos_max: float = 0.34
    prev_down_min: float = 46.0
    prev_down_max: float = 76.0
    rope_dir_required: int = 1
    bars_since_max: float = 8.0
    dsa_width_max: float = 0.28
    weekly_dev_min: float = -2.3
    trend_score_min: float = 0.68
    top_n_per_month: int = 8


def filter_stage_events(events: pd.DataFrame, spec: StageSpec) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    sub = events.copy()
    if "ret_5" in sub.columns:
        s = safe_numeric(sub["ret_5"])
        if spec.ret5_min is not None:
            sub = sub[s >= float(spec.ret5_min)]
            s = safe_numeric(sub["ret_5"])
        if spec.ret5_max is not None:
            sub = sub[s < float(spec.ret5_max)]
    return dedup_events(sub)


def filter_env_gate(events: pd.DataFrame, spec: EnvGateSpec) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    sub = events.copy()
    cond = pd.Series(True, index=sub.index)
    if spec.trend_min is not None and "score_trend_total" in sub.columns:
        cond &= safe_numeric(sub["score_trend_total"]).fillna(-99) >= float(spec.trend_min)
    if spec.weekly_dev_min is not None and "w_dsa_signed_vwap_dev_pct" in sub.columns:
        cond &= safe_numeric(sub["w_dsa_signed_vwap_dev_pct"]).fillna(-99) >= float(spec.weekly_dev_min)
    if spec.trend_gap_min is not None and "trend_gap_10_20" in sub.columns:
        cond &= safe_numeric(sub["trend_gap_10_20"]).fillna(-99) >= float(spec.trend_gap_min)
    return dedup_events(sub[cond].copy())


def build_stage_split_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    stage_specs = [
        StageSpec(name="all_candidate", ret5_min=None, ret5_max=None),
        StageSpec(name="pure_compression", ret5_min=None, ret5_max=0.02),
        StageSpec(name="launch_confirmation", ret5_min=0.02, ret5_max=None),
    ]
    env_specs = [
        EnvGateSpec(name="no_env_gate", trend_min=None, weekly_dev_min=None, trend_gap_min=None),
        EnvGateSpec(name="with_env_gate", trend_min=0.68, weekly_dev_min=-2.3, trend_gap_min=0.0),
    ]
    for stage in stage_specs:
        stage_df = filter_stage_events(events, stage)
        for env in env_specs:
            gated = filter_env_gate(stage_df, env)
            stat = summarize_events(gated, [20, 40, 60])
            rows.append({
                "stage_name": stage.name,
                "env_gate": env.name,
                "stock_n": int(gated["symbol"].nunique()) if not gated.empty and "symbol" in gated.columns else 0,
                **stat,
            })
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        rdf.to_csv(f"{OUT_DIR}/11_stage_split_summary.csv", index=False)
    return rdf


def build_env_gate_comparison(events: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    stage_specs = [
        StageSpec(name="pure_compression", ret5_min=None, ret5_max=0.02),
        StageSpec(name="launch_confirmation", ret5_min=0.02, ret5_max=None),
    ]
    env_specs = [
        EnvGateSpec(name="base", trend_min=None, weekly_dev_min=None, trend_gap_min=None),
        EnvGateSpec(name="trend_only", trend_min=0.68, weekly_dev_min=None, trend_gap_min=None),
        EnvGateSpec(name="weekly_only", trend_min=None, weekly_dev_min=-2.3, trend_gap_min=None),
        EnvGateSpec(name="trend_gap_only", trend_min=None, weekly_dev_min=None, trend_gap_min=0.0),
        EnvGateSpec(name="full_gate", trend_min=0.68, weekly_dev_min=-2.3, trend_gap_min=0.0),
    ]
    for stage in stage_specs:
        stage_df = filter_stage_events(events, stage)
        for env in env_specs:
            sub = filter_env_gate(stage_df, env)
            stat = summarize_events(sub, [20, 40, 60])
            rows.append({
                "stage_name": stage.name,
                "gate_name": env.name,
                "stock_n": int(sub["symbol"].nunique()) if not sub.empty and "symbol" in sub.columns else 0,
                **stat,
            })
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        rdf["coverage_vs_candidates"] = rdf["n"] / max(len(events), 1)
        rdf.to_csv(f"{OUT_DIR}/12_env_gate_comparison.csv", index=False)
    return rdf


def apply_fixed_rule(events: pd.DataFrame, rule: FixedRuleSpec, stage_name: str) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    sub = events.copy()
    if stage_name == "pure_compression" and "ret_5" in sub.columns:
        sub = sub[safe_numeric(sub["ret_5"]) < 0.02]
    elif stage_name == "launch_confirmation" and "ret_5" in sub.columns:
        sub = sub[safe_numeric(sub["ret_5"]) >= 0.02]
    if "dsa_trade_pos_01" in sub.columns:
        sub = sub[safe_numeric(sub["dsa_trade_pos_01"]) <= rule.dsa_trade_pos_max]
    if "dsa_confirmed_pivot_pos_01" in sub.columns:
        sub = sub[safe_numeric(sub["dsa_confirmed_pivot_pos_01"]).fillna(1) <= rule.dsa_confirmed_pos_max]
    if "prev_confirmed_down_bars" in sub.columns:
        prev_down = safe_numeric(sub["prev_confirmed_down_bars"])
        sub = sub[(prev_down >= rule.prev_down_min) & (prev_down <= rule.prev_down_max)]
    if "rope_dir" in sub.columns:
        sub = sub[safe_numeric(sub["rope_dir"]).fillna(0) == rule.rope_dir_required]
    if "bars_since_dir_change" in sub.columns:
        sub = sub[safe_numeric(sub["bars_since_dir_change"]).fillna(99) <= rule.bars_since_max]
    if "dsa_trade_range_width_pct" in sub.columns:
        sub = sub[safe_numeric(sub["dsa_trade_range_width_pct"]).fillna(99) <= rule.dsa_width_max]
    if "w_dsa_signed_vwap_dev_pct" in sub.columns:
        sub = sub[safe_numeric(sub["w_dsa_signed_vwap_dev_pct"]).fillna(-99) >= rule.weekly_dev_min]
    if "score_trend_total" in sub.columns:
        sub = sub[safe_numeric(sub["score_trend_total"]).fillna(-99) >= rule.trend_score_min]
    if sub.empty:
        return sub.copy()
    sub = dedup_events(sub)
    return sub


def add_soft_ranking_score(sub: pd.DataFrame) -> pd.DataFrame:
    out = sub.copy()
    score_parts = pd.DataFrame(index=out.index)
    score_parts["ret5"] = normalize_01(safe_numeric(out.get("ret_5", pd.Series(index=out.index))), reverse=False)
    score_parts["confirmed_low"] = normalize_01(safe_numeric(out.get("dsa_confirmed_pivot_pos_01", pd.Series(index=out.index))), reverse=True)
    score_parts["trend"] = normalize_01(safe_numeric(out.get("score_trend_total", pd.Series(index=out.index))), reverse=False)
    prev_down = safe_numeric(out.get("prev_confirmed_down_bars", pd.Series(index=out.index)))
    center_dist = (prev_down - 61.0).abs()
    score_parts["prev_down_mid"] = normalize_01(center_dist, reverse=True)
    score_parts["weekly_dev"] = normalize_01(safe_numeric(out.get("w_dsa_signed_vwap_dev_pct", pd.Series(index=out.index))), reverse=False)
    out["soft_rank_score"] = pd.concat([
        0.32 * score_parts["ret5"],
        0.22 * score_parts["confirmed_low"],
        0.22 * score_parts["trend"],
        0.12 * score_parts["prev_down_mid"],
        0.12 * score_parts["weekly_dev"],
    ], axis=1).sum(axis=1)
    return out


def build_fixed_rule_ranked_results(events: pd.DataFrame, top_k_export: int = 300) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: List[Dict[str, object]] = []
    export_frames: List[pd.DataFrame] = []
    rules = [
        FixedRuleSpec(name="fixed_rule_v1", top_n_per_month=8),
        FixedRuleSpec(name="fixed_rule_v1_top5", top_n_per_month=5),
        FixedRuleSpec(name="fixed_rule_v1_top10", top_n_per_month=10),
    ]
    for stage_name in ["pure_compression", "launch_confirmation"]:
        for rule in rules:
            sub = apply_fixed_rule(events, rule, stage_name)
            if sub.empty:
                continue
            sub = add_soft_ranking_score(sub)
            if "year_month" in sub.columns:
                ranked = sub.sort_values(["year_month", "soft_rank_score", "ret_5"], ascending=[True, False, False]).copy()
                ranked["month_pick_rank"] = ranked.groupby("year_month")["soft_rank_score"].rank(method="first", ascending=False)
                picked = ranked[ranked["month_pick_rank"] <= rule.top_n_per_month].copy()
            else:
                ranked = sub.sort_values(["soft_rank_score", "ret_5"], ascending=[False, False]).copy()
                ranked["month_pick_rank"] = np.arange(len(ranked)) + 1
                picked = ranked.head(rule.top_n_per_month).copy()
            picked = dedup_events(picked)
            if len(picked) < 12:
                continue
            stat = summarize_events(picked, [20, 40, 60])
            rows.append({
                "stage_name": stage_name,
                "rule_name": rule.name,
                "top_n_per_month": rule.top_n_per_month,
                "stock_n": int(picked["symbol"].nunique()) if "symbol" in picked.columns else 0,
                "avg_soft_rank_score": round(float(safe_numeric(picked["soft_rank_score"]).mean()), 6),
                **stat,
            })
            keep_cols = [c for c in [
                "symbol", "datetime", "year_month", "close", "ret_5", "ret_20", "ret_40", "ret_60", "max_dd_20", "max_dd_40", "max_dd_60",
                "rr_20", "dsa_trade_pos_01", "dsa_confirmed_pivot_pos_01", "prev_confirmed_down_bars", "bars_since_dir_change",
                "dsa_trade_range_width_pct", "w_dsa_signed_vwap_dev_pct", "score_trend_total", "trend_gap_10_20", "soft_rank_score", "month_pick_rank"
            ] if c in picked.columns]
            tmp = picked[keep_cols].copy()
            tmp["stage_name"] = stage_name
            tmp["rule_name"] = rule.name
            export_frames.append(tmp)
    scan_df = pd.DataFrame(rows).sort_values(["rr_20", "ret_20", "wr_20", "n"], ascending=[False, False, False, False]).reset_index(drop=True) if rows else pd.DataFrame()
    top_df = pd.concat(export_frames, ignore_index=True).sort_values(["stage_name", "rule_name", "datetime", "month_pick_rank"]).head(top_k_export) if export_frames else pd.DataFrame()
    if not scan_df.empty:
        scan_df.to_csv(f"{OUT_DIR}/13_fixed_rule_ranked_scan.csv", index=False)
    if not top_df.empty:
        top_df.to_csv(f"{OUT_DIR}/13b_fixed_rule_top_events.csv", index=False)
    return scan_df, top_df


def build_fixed_rule_stability(events: pd.DataFrame, fixed_scan_df: pd.DataFrame) -> pd.DataFrame:
    if events.empty or fixed_scan_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    work = events.copy()
    work["turnover_proxy"] = safe_numeric(work.get("close", pd.Series(index=work.index))) * safe_numeric(work.get("vol", pd.Series(index=work.index)))
    work["year"] = pd.to_datetime(work["datetime"]).dt.year if "datetime" in work.columns else np.nan
    if work["turnover_proxy"].notna().sum() >= 30:
        try:
            work["liquidity_bucket"] = pd.qcut(work["turnover_proxy"], q=3, labels=["small", "mid", "large"], duplicates="drop")
        except Exception:
            work["liquidity_bucket"] = "all"
    else:
        work["liquidity_bucket"] = "all"
    med = float(work["ret_20"].median()) if "ret_20" in work.columns and work["ret_20"].notna().sum() >= 30 else 0.0
    work["env_bucket"] = np.where(safe_numeric(work.get("ret_20", pd.Series(index=work.index))).fillna(-99) >= med, "better", "worse")
    for _, row in fixed_scan_df.head(4).iterrows():
        stage_name = str(row["stage_name"])
        rule = FixedRuleSpec(name=str(row["rule_name"]), top_n_per_month=int(row.get("top_n_per_month", 8)))
        sub = apply_fixed_rule(work, rule, stage_name)
        if sub.empty:
            continue
        sub = add_soft_ranking_score(sub)
        if "year_month" in sub.columns:
            sub = sub.sort_values(["year_month", "soft_rank_score"], ascending=[True, False]).copy()
            sub["month_pick_rank"] = sub.groupby("year_month")["soft_rank_score"].rank(method="first", ascending=False)
            sub = sub[sub["month_pick_rank"] <= rule.top_n_per_month].copy()
        sub = dedup_events(sub)
        if len(sub) < 12:
            continue
        for dim in ["year", "liquidity_bucket", "env_bucket"]:
            for key, g in sub.groupby(dim, dropna=False):
                if len(g) < 8:
                    continue
                stat = summarize_events(g, [20, 40, 60])
                rows.append({"stage_name": stage_name, "rule_name": rule.name, "top_n_per_month": rule.top_n_per_month, "slice_dim": dim, "slice_key": str(key), **stat})
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        rdf.to_csv(f"{OUT_DIR}/14_fixed_rule_stability_slices.csv", index=False)
    return rdf

def analyze_00_dataset_summary(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = [{"section": "global", "sample_n": int(len(df)), "stock_n": int(df["symbol"].nunique()), "ret20_mean": round(float(df["ret_20"].mean()), 5), "dd20_mean": round(float(df["max_dd_20"].mean()), 5), "wr20_mean": round(float(df["win_20"].mean()), 4), "rr20_mean": round(rr_from_ret_dd(df["ret_20"].mean(), df["max_dd_20"].mean()), 3)}]
    if not events.empty:
        rows.append({"section": "candidate_pool", "sample_n": int(len(events)), "stock_n": int(events["symbol"].nunique()), "ret20_mean": round(float(events["ret_20"].mean()), 5), "dd20_mean": round(float(events["max_dd_20"].mean()), 5), "wr20_mean": round(float(events["win_20"].mean()), 4), "rr20_mean": round(rr_from_ret_dd(events["ret_20"].mean(), events["max_dd_20"].mean()), 3)})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(f"{OUT_DIR}/00_dataset_summary.csv", index=False)
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
    rdf.to_csv(f"{OUT_DIR}/01_feature_inventory.csv", index=False)
    return rdf


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 GBDT 结果反推规则的买点实验脚本")
    parser.add_argument("--n-stocks", type=int, default=100)
    parser.add_argument("--bars", type=int, default=800)
    parser.add_argument("--freq", default="d")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis-mode", type=str, default="full", choices=["full", "prepare_only", "model_only", "rule_only"])
    parser.add_argument("--neighbor-dsa-max", type=float, default=0.45)
    parser.add_argument("--neighbor-prev-down-min", type=float, default=8.0)
    parser.add_argument("--neighbor-current-run-max", type=float, default=20.0)
    parser.add_argument("--neighbor-bars-since-max", type=float, default=12.0)
    args = parser.parse_args()

    stocks = get_stock_pool(args.n_stocks, args.seed)
    print(f"股票池抽取: {len(stocks)}只 (seed={args.seed})")
    print("研究目标: 保留 GBDT 归因，同时新增规则反推实验")

    all_dfs: List[pd.DataFrame] = []
    for idx, code in enumerate(stocks):
        kline = load_kline(code, args.freq, args.bars)
        if kline is None or len(kline) < 100:
            continue
        try:
            fac = compute_factors(kline)
            fac = add_forward_returns(fac, RET_WINDOWS)
            fac["symbol"] = code
            fac["datetime"] = fac.index
            valid = fac.dropna(subset=["ret_20"])
            if len(valid) > 50:
                all_dfs.append(valid)
                print(f"  [{idx + 1}/{len(stocks)}] {code}: {len(kline)}bars → {len(valid)}有效样本")
        except Exception as exc:
            print(f"  [{idx + 1}/{len(stocks)}] {code} 因子计算失败: {exc}")

    if not all_dfs:
        print("无有效数据，退出")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    neighbor_spec = NeighborSpec(dsa_max=args.neighbor_dsa_max, prev_down_min=args.neighbor_prev_down_min, current_run_max=args.neighbor_current_run_max, bars_since_max=args.neighbor_bars_since_max)
    events = build_neighbor_events(df, neighbor_spec)

    analyze_00_dataset_summary(df, events)
    analyze_01_feature_inventory(events)
    if not events.empty:
        events.to_csv(f"{OUT_DIR}/02_candidate_events.csv", index=False)

    print(f"\n总样本: {len(df)}, 股票数: {df['symbol'].nunique()}")
    print(f"C 邻域候选事件: {len(events)}, 股票数: {events['symbol'].nunique() if not events.empty else 0}")
    if events.empty:
        print("候选事件为空，退出")
        return

    if args.analysis_mode == "prepare_only":
        print(f"\n全部完成，结果保存在 {OUT_DIR}/")
        return

    if args.analysis_mode != "rule_only":
        fold_frames: List[pd.DataFrame] = []
        trade_reg_metrics, trade_reg_imp = run_single_model(events, TRADE_FEATURES, "ret_20", "reg", "trade")
        trade_clf_metrics, trade_clf_imp = run_single_model(events, TRADE_FEATURES, "good20", "clf", "trade")
        research_reg_metrics, research_reg_imp = run_single_model(events, RESEARCH_FEATURES, "ret_20", "reg", "research")
        research_clf_metrics, research_clf_imp = run_single_model(events, RESEARCH_FEATURES, "good20", "clf", "research")
        for frame in [trade_reg_metrics, trade_clf_metrics, research_reg_metrics, research_clf_metrics]:
            if not frame.empty:
                fold_frames.append(frame)
        if fold_frames:
            pd.concat(fold_frames, ignore_index=True).to_csv(f"{OUT_DIR}/05_fold_metrics.csv", index=False)

        trade_reg_imp_agg = aggregate_importance(trade_reg_imp)
        trade_clf_imp_agg = aggregate_importance(trade_clf_imp)
        research_reg_imp_agg = aggregate_importance(research_reg_imp)
        research_clf_imp_agg = aggregate_importance(research_clf_imp)

        if not trade_reg_imp_agg.empty:
            trade_reg_imp_agg.to_csv(f"{OUT_DIR}/03_trade_reg_importance.csv", index=False)
        if not trade_clf_imp_agg.empty:
            trade_clf_imp_agg.to_csv(f"{OUT_DIR}/03b_trade_clf_importance.csv", index=False)
        if not research_reg_imp_agg.empty:
            research_reg_imp_agg.to_csv(f"{OUT_DIR}/04_research_reg_importance.csv", index=False)
        if not research_clf_imp_agg.empty:
            research_clf_imp_agg.to_csv(f"{OUT_DIR}/04b_research_clf_importance.csv", index=False)

        trade_top = trade_reg_imp_agg.groupby("feature", as_index=False)["rank_score"].mean().sort_values("rank_score", ascending=False) if not trade_reg_imp_agg.empty else pd.DataFrame()
        research_top = research_reg_imp_agg.groupby("feature", as_index=False)["rank_score"].mean().sort_values("rank_score", ascending=False) if not research_reg_imp_agg.empty else pd.DataFrame()
        trade_feats_for_bucket = trade_top["feature"].head(8).tolist() if not trade_top.empty else [f for f in PRIMARY_HINT_FEATURES if f in TRADE_FEATURES]
        research_feats_for_bucket = research_top["feature"].head(8).tolist() if not research_top.empty else [f for f in PRIMARY_HINT_FEATURES if f in RESEARCH_FEATURES]
        build_bucket_profiles(events, trade_feats_for_bucket, f"{OUT_DIR}/06_trade_bucket_profiles.csv")
        build_bucket_profiles(events, research_feats_for_bucket, f"{OUT_DIR}/06b_research_bucket_profiles.csv")
        build_bucket_profiles(events, COMPOSITE_FEATURES, f"{OUT_DIR}/08_composite_feature_profiles.csv")
        build_parameter_hints(events, pd.concat([trade_reg_imp_agg, trade_clf_imp_agg], ignore_index=True), pd.concat([research_reg_imp_agg, research_clf_imp_agg], ignore_index=True))

        print("\n=== 关键结果预览 ===")
        if not trade_reg_imp_agg.empty:
            print("\n[trade-safe 回归重要性 Top10]")
            print(trade_reg_imp_agg[["feature", "rank_score", "gain_importance", "perm_importance", "fold_count"]].head(10).to_string(index=False))

    scan_df, _ = build_compression_launch_scan(events)
    build_stability_slices(events, scan_df)
    if not scan_df.empty:
        print("\n[压缩低位启动规则 Top10]")
        print(scan_df[[
            "dsa_trade_pos_max", "bb_width_norm_max", "bb_pos_low", "bb_pos_high", "rope_pivot_pos_max",
            "w_dsa_signed_vwap_dev_min", "range_width_atr_min", "breakout_mode", "weekly_mode", "n", "ret_20", "wr_20", "rr_20", "rank_score"
        ]].head(10).to_string(index=False))

    print(f"\n{'=' * 70}")
    print(f"# 全部完成! 结果保存在 {OUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
