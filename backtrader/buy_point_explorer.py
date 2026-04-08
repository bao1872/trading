# -*- coding: utf-8 -*-
"""
买入点收益风险比探索（全因子优先 + DSA双轨版）

Purpose
  - 全因子探索优先：先做因子普查、单因子、离散因子、双因子扫描，再进入策略验证
  - DSA 双轨：
      * confirmed/research：来自 merged_dsa_atr_rope_bb_factors_with_confirmed_run_bars.py 的回看确认特征
      * trade-safe：在本脚本内基于 rolling 区间构造的交易代理特征，默认用于策略入口
  - 纳入新增结构长度因子：
      * prev_confirmed_up_bars
      * prev_confirmed_down_bars
      * last_confirmed_run_bars
      * current_run_bars

Notes
  - rr = ret / abs(MAE)
  - MAE 使用未来窗口内最低价相对入场价的回撤，long 为负值
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
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
    from merged_dsa_atr_rope_bb_factors_with_confirmed_run_bars import (
        DSAConfig,
        RopeConfig,
        compute_atr_rope,
        compute_bollinger,
        compute_dsa,
    )

warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = "buy_point_explorer_redesigned"
os.makedirs(OUT_DIR, exist_ok=True)

RET_WINDOWS = [5, 10, 20, 40, 60]
EVENT_DEDUP_BARS = 20


def rr_from_ret_dd(ret: float, dd: float) -> float:
    if pd.isna(ret) or pd.isna(dd) or dd == 0:
        return np.nan
    return float(ret / abs(dd))


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


def slice_equal_time_buckets(df: pd.DataFrame, n_slices: int = 3, time_col: str = "datetime") -> pd.DataFrame:
    out = df.sort_values(time_col).copy()
    if len(out) == 0:
        out["time_slice"] = []
        return out
    labels = [f"T{i+1}" for i in range(n_slices)]
    ranks = np.linspace(0, n_slices, len(out), endpoint=False)
    out["time_slice"] = [labels[min(int(x), n_slices - 1)] for x in ranks]
    return out


def normalize_score(series: pd.Series, reverse: bool = False, clip_q: Tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    s = series.astype(float).copy()
    lo = s.quantile(clip_q[0])
    hi = s.quantile(clip_q[1])
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        z = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    else:
        s = s.clip(lower=lo, upper=hi)
        z = (s - lo) / (hi - lo)
    return 1.0 - z if reverse else z


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


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "vol" in d.columns and "volume" not in d.columns:
        d["volume"] = d["vol"]
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
            mae[i] = (future_low - entry) / entry
        out[f"ret_{w}"] = fut
        out[f"max_dd_{w}"] = mae
        out[f"win_{w}"] = (out[f"ret_{w}"] > 0).astype(float)
    return out


FACTOR_COLS = [
    # DSA trade-safe
    "dsa_trade_pos_01", "dsa_trade_range_width_pct", "dsa_trade_dist_to_low_01", "dsa_trade_dist_to_high_01",
    "dsa_trade_breakout_20", "dsa_trade_breakdown_20",
    # DSA confirmed / research
    "dsa_confirmed_pivot_pos_01", "dsa_signed_vwap_dev_pct", "dsa_trend_aligned_vwap_dev_pct",
    "lh_hh_low_pos", "dsa_bull_vwap_dev_pct", "dsa_bear_vwap_dev_pct",
    "prev_confirmed_up_bars", "prev_confirmed_down_bars", "last_confirmed_run_bars", "current_run_bars",
    # Rope
    "rope_dir", "dist_to_rope_atr", "rope_slope_atr_5", "is_consolidating", "bars_since_dir_change",
    "range_break_up", "range_break_up_strength", "range_width_atr", "channel_pos_01", "range_pos_01", "rope_pivot_pos_01",
    # BB
    "bb_pos_01", "bb_width_norm", "bb_width_percentile", "bb_width_change_5", "bb_expanding", "bb_contracting",
    "bb_expand_streak", "bb_contract_streak",
    # Weekly
    "w_DSA_DIR", "w_dsa_confirmed_pivot_pos_01", "w_dsa_signed_vwap_dev_pct", "w_prev_confirmed_up_bars", "w_prev_confirmed_down_bars",
    "w_dsa_trade_pos_01", "w_dsa_trade_range_width_pct", "w_dsa_trade_dist_to_low_01", "w_dsa_trade_dist_to_high_01",
    "w_factor_available", "w_sample_count",
]
FACTOR_GROUPS = {
    "DSA_TRADE": [
        "dsa_trade_pos_01", "dsa_trade_range_width_pct", "dsa_trade_dist_to_low_01", "dsa_trade_dist_to_high_01",
        "dsa_trade_breakout_20", "dsa_trade_breakdown_20", "w_dsa_trade_pos_01", "w_dsa_trade_range_width_pct",
        "w_dsa_trade_dist_to_low_01", "w_dsa_trade_dist_to_high_01",
    ],
    "DSA_CONFIRMED": [
        "dsa_confirmed_pivot_pos_01", "dsa_signed_vwap_dev_pct", "dsa_trend_aligned_vwap_dev_pct",
        "lh_hh_low_pos", "dsa_bull_vwap_dev_pct", "dsa_bear_vwap_dev_pct",
        "prev_confirmed_up_bars", "prev_confirmed_down_bars", "last_confirmed_run_bars", "current_run_bars",
        "w_DSA_DIR", "w_dsa_confirmed_pivot_pos_01", "w_dsa_signed_vwap_dev_pct", "w_prev_confirmed_up_bars", "w_prev_confirmed_down_bars",
    ],
    "ROPE": [
        "rope_dir", "dist_to_rope_atr", "rope_slope_atr_5", "is_consolidating", "bars_since_dir_change",
        "range_break_up", "range_break_up_strength", "range_width_atr", "channel_pos_01", "range_pos_01", "rope_pivot_pos_01",
    ],
    "BB": [
        "bb_pos_01", "bb_width_norm", "bb_width_percentile", "bb_width_change_5", "bb_expanding", "bb_contracting", "bb_expand_streak", "bb_contract_streak",
    ],
    "WEEKLY_META": ["w_factor_available", "w_sample_count"],
}
FACTOR_USAGE = {c: "trade" for c in FACTOR_COLS}
for c in [
    "dsa_confirmed_pivot_pos_01", "dsa_signed_vwap_dev_pct", "dsa_trend_aligned_vwap_dev_pct", "lh_hh_low_pos", "dsa_bull_vwap_dev_pct",
    "dsa_bear_vwap_dev_pct", "prev_confirmed_up_bars", "prev_confirmed_down_bars", "last_confirmed_run_bars", "current_run_bars",
    "w_DSA_DIR", "w_dsa_confirmed_pivot_pos_01", "w_dsa_signed_vwap_dev_pct", "w_prev_confirmed_up_bars", "w_prev_confirmed_down_bars",
]:
    FACTOR_USAGE[c] = "research"
DISCRETE_COLS = [
    "rope_dir", "is_consolidating", "range_break_up", "bb_expanding", "bb_contracting",
    "w_DSA_DIR", "dsa_trade_breakout_20", "dsa_trade_breakdown_20", "w_factor_available",
]
CONTINUOUS_COLS = [c for c in FACTOR_COLS if c not in DISCRETE_COLS]
TRADE_FACTOR_COLS = [c for c in FACTOR_COLS if FACTOR_USAGE.get(c) == "trade"]
RESEARCH_FACTOR_COLS = [c for c in FACTOR_COLS if FACTOR_USAGE.get(c) == "research"]


# =========================
# Exploration-first analyses
# =========================
def analyze_00_factor_inventory(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⓪: 因子全景盘点")
    print("=" * 70)
    rows = []
    for factor in FACTOR_COLS:
        present = factor in df.columns
        s = df[factor] if present else pd.Series(dtype=float)
        coverage = float(s.notna().mean()) if present else 0.0
        dtype_name = str(s.dtype) if present else "missing"
        nunique = int(s.nunique(dropna=True)) if present else 0
        rows.append({
            "factor": factor,
            "group": next((g for g, cols in FACTOR_GROUPS.items() if factor in cols), "OTHER"),
            "usage": FACTOR_USAGE.get(factor, "shared"),
            "present": int(present),
            "coverage": round(coverage, 4),
            "dtype": dtype_name,
            "nunique": nunique,
            "mean": round(float(s.mean()), 6) if present and pd.api.types.is_numeric_dtype(s) else np.nan,
            "std": round(float(s.std()), 6) if present and pd.api.types.is_numeric_dtype(s) else np.nan,
        })
    rdf = pd.DataFrame(rows).sort_values(["present", "coverage", "group", "factor"], ascending=[False, False, True, True])
    if not rdf.empty:
        print(rdf[["factor", "group", "usage", "coverage", "nunique", "mean", "std"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/00_factor_inventory.csv", index=False)
    return rdf


def analyze_01_single_factor(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次①: 单因子截面分析 (主口径 ret_20)")
    print("=" * 70)
    ret_cols = [f"ret_{w}" for w in RET_WINDOWS] + [f"max_dd_{w}" for w in RET_WINDOWS] + [f"win_{w}" for w in RET_WINDOWS]
    results = []
    for col in CONTINUOUS_COLS:
        if col not in df.columns:
            continue
        sub = df[[col] + ret_cols].dropna()
        if len(sub) < 200:
            continue
        try:
            qcol = f"{col}_q"
            sub[qcol] = pd.qcut(sub[col], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
            grp = sub.groupby(qcol, observed=True).agg(n=(col, "count"), mean_ret20=("ret_20", "mean"), mean_max_dd=("max_dd_20", "mean"), win_rate=("win_20", "mean"))
            grp["risk_reward"] = grp.apply(lambda r: rr_from_ret_dd(r["mean_ret20"], r["mean_max_dd"]), axis=1)
            ic = sub[col].corr(sub["ret_20"])
            rank_ic = sub[col].rank().corr(sub["ret_20"].rank())
            best_q = grp["risk_reward"].idxmax() if not grp["risk_reward"].isna().all() else "N/A"
            results.append({
                "factor": col,
                "usage": FACTOR_USAGE.get(col, "shared"),
                "IC": round(ic, 4),
                "Rank_IC": round(rank_ic, 4),
                "best_Q_rr": best_q,
                "Q1_rr": round(grp.loc["Q1", "risk_reward"], 3) if "Q1" in grp.index else np.nan,
                "Q5_rr": round(grp.loc["Q5", "risk_reward"], 3) if "Q5" in grp.index else np.nan,
                "Q1_ret20": round(grp.loc["Q1", "mean_ret20"], 5) if "Q1" in grp.index else np.nan,
                "Q5_ret20": round(grp.loc["Q5", "mean_ret20"], 5) if "Q5" in grp.index else np.nan,
            })
        except Exception as e:
            results.append({"factor": col, "usage": FACTOR_USAGE.get(col, "shared"), "error": str(e)[:80]})
    rdf = pd.DataFrame(results).sort_values("Rank_IC", key=lambda s: s.abs(), ascending=False)
    if not rdf.empty:
        print(rdf.head(25)[["factor", "usage", "IC", "Rank_IC", "best_Q_rr", "Q1_rr", "Q5_rr", "Q1_ret20", "Q5_ret20"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/01_single_factor.csv", index=False)
    return rdf


def analyze_01b_discrete_factor(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次①B: 离散因子分组分析")
    print("=" * 70)
    rows = []
    for col in DISCRETE_COLS:
        if col not in df.columns:
            continue
        sub = df[[col, "ret_20", "max_dd_20", "win_20"]].dropna()
        if len(sub) < 100:
            continue
        for key, g in sub.groupby(col, observed=True):
            if len(g) < 20:
                continue
            ret = g["ret_20"].mean()
            dd = g["max_dd_20"].mean()
            wr = g["win_20"].mean()
            rows.append({
                "factor": col,
                "usage": FACTOR_USAGE.get(col, "shared"),
                "bucket": key,
                "n": len(g),
                "ret20": round(ret, 5),
                "dd20": round(dd, 5),
                "wr20": round(wr, 4),
                "rr20": round(rr_from_ret_dd(ret, dd), 3),
            })
    rdf = pd.DataFrame(rows).sort_values(["factor", "rr20"], ascending=[True, False])
    if not rdf.empty:
        print(rdf.head(40).to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/01b_discrete_factor.csv", index=False)
    return rdf


SCENARIOS = {
    "S1_Rope触底反弹": lambda d: (d["rope_dir"] == 1) & (d["rope_dir"].shift(1) == -1) & (d["dist_to_rope_atr"] < 0.5),
    "S2_盘整突破": lambda d: (d["range_break_up"] == 1) & (d["range_width_atr"] < d["range_width_atr"].median()),
    "S3_BB下轨回弹": lambda d: (d["bb_pos_01"] < 0.15) & (d["close"] > d["open"]),
    "S4_BB_Squeeze突破": lambda d: (d["bb_width_percentile"] < 0.2) & (d["bb_expanding"] == 1),
    "S5_Trade低位+Rope向上": lambda d: (d["dsa_trade_pos_01"] < 0.4) & (d["rope_dir"] == 1),
    "S6_长下跌段后转强": lambda d: (d.get("prev_confirmed_down_bars", 0) >= 10) & (d["rope_dir"] == 1),
}


def eval_scenario(mask: pd.Series, name: str, df: pd.DataFrame, baseline_n: int) -> Optional[Dict[str, float]]:
    sm = df[mask]
    if len(sm) < 30:
        return None
    row: Dict[str, float] = {"scenario": name, "n": len(sm), "coverage_pct": round(len(sm) / baseline_n * 100, 1)}
    for w in [5, 10, 20, 60]:
        ret = sm[f"ret_{w}"].mean()
        dd = sm[f"max_dd_{w}"].mean()
        wr = sm[f"win_{w}"].mean()
        row[f"ret{w}"] = round(ret, 5)
        row[f"dd{w}"] = round(dd, 5)
        row[f"wr{w}"] = round(wr, 4)
        row[f"rr{w}"] = round(rr_from_ret_dd(ret, dd), 3)
    return row


def analyze_02_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次②: 经典买入场景回测")
    print("=" * 70)
    need = list(dict.fromkeys(FACTOR_COLS + ["open", "close"] + [f"ret_{w}" for w in [5, 10, 20, 60]] + [f"max_dd_{w}" for w in [5, 10, 20, 60]] + [f"win_{w}" for w in [5, 10, 20, 60]]))
    sub = df[[c for c in need if c in df.columns]].dropna()
    baseline_n = len(sub)
    rows = []
    for name, fn in SCENARIOS.items():
        try:
            r = eval_scenario(fn(sub), name, sub, baseline_n)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  [WARN] {name} 失败: {e}")
    rdf = pd.DataFrame(rows).sort_values("rr20", ascending=False)
    if not rdf.empty:
        print(rdf[[c for c in ["scenario", "n", "coverage_pct", "ret20", "dd20", "wr20", "rr20", "ret60", "rr60"] if c in rdf.columns]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/02_scenarios.csv", index=False)
    return rdf


def analyze_03_interaction(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("# 层次③: 因子交互效应分析 (四象限)")
    print("=" * 70)
    need = ["rope_dir", "bb_pos_01", "dsa_trade_pos_01", "rope_slope_atr_5", "bb_width_percentile", "ret_20", "max_dd_20", "win_20"]
    sub = df[[c for c in need if c in df.columns]].dropna()
    if len(sub) == 0:
        return
    inters = [
        ("rope_dir × bb_pos_01", "rope_dir", pd.cut(sub["bb_pos_01"], bins=[0, 0.2, 0.5, 0.8, 1.01], labels=["超卖", "偏弱", "偏强", "超买"], include_lowest=True)),
        ("trade_pos × rope_slope", pd.cut(sub["dsa_trade_pos_01"], bins=[0, 0.3, 0.7, 1.01], labels=["低位", "中位", "高位"], include_lowest=True), pd.cut(sub["rope_slope_atr_5"], bins=[-np.inf, -0.01, 0.01, 0.05, np.inf], labels=["强跌", "微跌", "微涨", "强涨"])),
    ]
    for label, a, b in inters:
        tmp = sub.copy()
        tmp["a"] = a if isinstance(a, pd.Series) else tmp[a]
        tmp["b"] = b if isinstance(b, pd.Series) else tmp[b]
        g = tmp.groupby(["a", "b"], observed=True).agg(n=("ret_20", "count"), ret20=("ret_20", "mean"), dd20=("max_dd_20", "mean"), wr20=("win_20", "mean"))
        g["rr20"] = g.apply(lambda r: rr_from_ret_dd(r["ret20"], r["dd20"]), axis=1).round(3)
        print(f"\n--- {label} ---")
        print(g.to_string())


def analyze_03b_factor_pairs(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次③B: 第二轮 trade-only 双因子组合扫描")
    print("=" * 70)
    candidate_cols = [c for c in ROUND2_TRADE_CANDIDATES if c in df.columns and c not in DISCRETE_COLS and df[c].notna().sum() >= 300]
    rows = []
    for i in range(len(candidate_cols)):
        for j in range(i + 1, len(candidate_cols)):
            a, b = candidate_cols[i], candidate_cols[j]
            sub = df[[a, b, "ret_20", "max_dd_20", "win_20"]].dropna()
            if len(sub) < 200:
                continue
            try:
                qa = pd.qcut(sub[a], q=3, labels=["L", "M", "H"], duplicates="drop")
                qb = pd.qcut(sub[b], q=3, labels=["L", "M", "H"], duplicates="drop")
                grp = sub.groupby([qa, qb], observed=True).agg(n=("ret_20", "count"), ret20=("ret_20", "mean"), dd20=("max_dd_20", "mean"), wr20=("win_20", "mean"))
                grp["rr20"] = grp.apply(lambda r: rr_from_ret_dd(r["ret20"], r["dd20"]), axis=1)
                best_idx = grp["rr20"].idxmax() if not grp["rr20"].isna().all() else None
                if best_idx is None:
                    continue
                best = grp.loc[best_idx]
                rows.append({
                    "f1": a, "f2": b, "best_bucket": f"{best_idx[0]}×{best_idx[1]}", "n": int(best["n"]),
                    "ret20": round(float(best["ret20"]), 5), "dd20": round(float(best["dd20"]), 5), "wr20": round(float(best["wr20"]), 4), "rr20": round(float(best["rr20"]), 3),
                })
            except Exception:
                continue
    rdf = pd.DataFrame(rows).sort_values("rr20", ascending=False)
    if not rdf.empty:
        print(rdf.head(25).to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/03b_factor_pairs.csv", index=False)
    return rdf


def analyze_04_rule_search(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次④: 最优规则搜索")
    print("=" * 70)
    need = [c for c in list(dict.fromkeys(FACTOR_COLS + ["ret_20", "max_dd_20", "win_20", "open", "close", "vol"])) if c in df.columns]
    sub = df[need].dropna(subset=["ret_20", "max_dd_20"]).copy()
    if len(sub) == 0:
        return pd.DataFrame()
    sub["vol_zscore"] = (sub["vol"] - sub["vol"].rolling(60, min_periods=20).mean()) / sub["vol"].rolling(60, min_periods=20).std()
    baseline_n = len(sub)

    def eval_rule(mask: pd.Series, name: str) -> Optional[Dict[str, float]]:
        g = sub[mask]
        if len(g) < 50 or len(g) / baseline_n < 0.005:
            return None
        ret = g["ret_20"].mean()
        dd = g["max_dd_20"].mean()
        wr = g["win_20"].mean()
        return {"rule": name, "n": len(g), "coverage_pct": round(len(g) / baseline_n * 100, 1), "ret20": round(ret, 5), "dd20": round(dd, 5), "wr20": round(wr, 4), "rr20": round(rr_from_ret_dd(ret, dd), 3)}

    results: List[Dict[str, float]] = []
    candidates = [
        ("dsa_trade_pos_01", [0.2, 0.3, 0.4]),
        ("dist_to_rope_atr", [-1.0, -0.5, 0.0]),
        ("bb_pos_01", [0.15, 0.2, 0.25]),
        ("bars_since_dir_change", [3, 5]),
        ("bb_width_percentile", [0.2, 0.25, 0.3]),
        ("prev_confirmed_down_bars", [5, 10, 20]),
    ]
    for col, vals in candidates:
        if col not in sub.columns:
            continue
        for hi in vals:
            mask = sub[col] <= hi if col != "prev_confirmed_down_bars" else sub[col] >= hi
            r = eval_rule(mask, f"{col}{'<=' if col != 'prev_confirmed_down_bars' else '>='}{hi}")
            if r:
                results.append(r)
    combos = {
        "tradepos<0.4&窄幅突破": (sub.get("dsa_trade_pos_01", np.nan) < 0.4) & (sub.get("range_break_up", 0) == 1) & (sub.get("range_width_atr", np.inf) < sub["range_width_atr"].median()),
        "tradepos<0.4&rope_up": (sub.get("dsa_trade_pos_01", np.nan) < 0.4) & (sub.get("rope_dir", 0) == 1) & (sub.get("bars_since_dir_change", np.inf) <= 3),
        "downbars_long&rope_up": (sub.get("prev_confirmed_down_bars", -np.inf) >= 10) & (sub.get("rope_dir", 0) == 1) & (sub.get("bb_pos_01", np.nan) < 0.3),
    }
    for name, mask in combos.items():
        r = eval_rule(mask, name)
        if r:
            results.append(r)
    rdf = pd.DataFrame(results).sort_values("rr20", ascending=False)
    if not rdf.empty:
        print(rdf.head(20).to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/04_rules.csv", index=False)
    return rdf




# =========================
# Round2 strategy specs
# =========================
ROUND2_TRADE_CANDIDATES = [
    "dsa_trade_pos_01", "dsa_trade_dist_to_low_01", "dist_to_rope_atr", "bb_pos_01",
    "rope_dir", "bars_since_dir_change", "rope_slope_atr_5", "range_break_up",
    "prev_confirmed_down_bars", "current_run_bars",
]

ROUND2_STRATEGY_SPECS: Dict[str, Dict[str, object]] = {
    "A_low_repair": {
        "desc": "低位修复主线",
        "dsa_thr": 0.35,
        "dist_thr": -0.8,
        "bars_thr": 8,
        "prev_down_min": 10,
        "need_rope_up": True,
    },
    "B_low_repair_strong_trigger": {
        "desc": "低位+更强触发",
        "dsa_thr": 0.35,
        "dist_thr": -0.5,
        "bars_thr": 5,
        "prev_down_min": None,
        "need_rope_up": True,
        "slope_min": 0.0,
    },
    "C_long_down_repair": {
        "desc": "长下跌段后的修复",
        "dsa_thr": 0.40,
        "dist_thr": None,
        "bars_thr": 8,
        "prev_down_min": 20,
        "need_rope_up": True,
        "current_run_max": 15,
    },
    "D_low_breakout": {
        "desc": "低位+breakout",
        "dsa_thr": 0.40,
        "dist_thr": None,
        "bars_thr": None,
        "prev_down_min": 10,
        "need_rope_up": False,
        "bb_pos_max": 0.35,
        "range_break_up": 1,
    },
    "E_rope_coil_launch": {
        "desc": "Rope贴线蓄势后启动",
        "dsa_thr": None,
        "dist_min": -1.0,
        "dist_max": 0.2,
        "bars_thr": 8,
        "prev_down_min": None,
        "need_rope_up": True,
        "slope_min": 0.0,
        "bb_pos_max": 0.5,
        "range_break_up_or_bb_expand": True,
    },
}


def build_round2_events(df: pd.DataFrame, strategy_name: str, cooldown: int = EVENT_DEDUP_BARS, override: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    if strategy_name not in ROUND2_STRATEGY_SPECS:
        raise ValueError(f"未知策略: {strategy_name}")
    spec = dict(ROUND2_STRATEGY_SPECS[strategy_name])
    if override:
        spec.update(override)
    need = [
        "symbol", "datetime", "open", "high", "low", "close",
        "dsa_trade_pos_01", "dsa_trade_dist_to_low_01", "dist_to_rope_atr", "bb_pos_01",
        "rope_dir", "bars_since_dir_change", "rope_slope_atr_5", "range_break_up", "bb_expanding",
        "prev_confirmed_down_bars", "current_run_bars",
        "w_DSA_DIR", "w_dsa_confirmed_pivot_pos_01", "w_dsa_trade_pos_01", "w_dsa_signed_vwap_dev_pct", "w_factor_available", "w_sample_count",
    ] + [f"ret_{w}" for w in [20, 40, 60]] + [f"max_dd_{w}" for w in [20, 40, 60]] + [f"win_{w}" for w in [20, 40, 60]]
    need = [c for c in need if c in df.columns]
    sub = df[need].dropna(subset=[c for c in ["ret_20", "max_dd_20", "rope_dir"] if c in need]).copy()
    mask = pd.Series(True, index=sub.index)
    dsa_thr = spec.get("dsa_thr")
    if dsa_thr is not None and "dsa_trade_pos_01" in sub.columns:
        mask &= sub["dsa_trade_pos_01"] <= float(dsa_thr)
    dist_thr = spec.get("dist_thr")
    if dist_thr is not None and "dist_to_rope_atr" in sub.columns:
        mask &= sub["dist_to_rope_atr"] <= float(dist_thr)
    dist_min = spec.get("dist_min")
    if dist_min is not None and "dist_to_rope_atr" in sub.columns:
        mask &= sub["dist_to_rope_atr"] >= float(dist_min)
    dist_max = spec.get("dist_max")
    if dist_max is not None and "dist_to_rope_atr" in sub.columns:
        mask &= sub["dist_to_rope_atr"] <= float(dist_max)
    if spec.get("need_rope_up") and "rope_dir" in sub.columns:
        mask &= sub["rope_dir"] == 1
    bars_thr = spec.get("bars_thr")
    if bars_thr is not None and "bars_since_dir_change" in sub.columns:
        mask &= sub["bars_since_dir_change"] <= int(bars_thr)
    slope_min = spec.get("slope_min")
    if slope_min is not None and "rope_slope_atr_5" in sub.columns:
        mask &= sub["rope_slope_atr_5"] > float(slope_min)
    prev_down_min = spec.get("prev_down_min")
    if prev_down_min is not None and "prev_confirmed_down_bars" in sub.columns:
        mask &= sub["prev_confirmed_down_bars"] >= float(prev_down_min)
    current_run_max = spec.get("current_run_max")
    if current_run_max is not None and "current_run_bars" in sub.columns:
        mask &= sub["current_run_bars"] <= float(current_run_max)
    bb_pos_max = spec.get("bb_pos_max")
    if bb_pos_max is not None and "bb_pos_01" in sub.columns:
        mask &= sub["bb_pos_01"] <= float(bb_pos_max)
    if spec.get("range_break_up") is not None and "range_break_up" in sub.columns:
        mask &= sub["range_break_up"] == float(spec["range_break_up"])
    if spec.get("range_break_up_or_bb_expand"):
        cond = pd.Series(False, index=sub.index)
        if "range_break_up" in sub.columns:
            cond |= sub["range_break_up"] == 1
        if "bb_expanding" in sub.columns:
            cond |= sub["bb_expanding"] == 1
        mask &= cond
    events = dedup_events(sub[mask].copy(), cooldown)
    for w in [20, 40, 60]:
        if f"ret_{w}" in events.columns:
            events[f"rr_{w}"] = events.apply(lambda r: rr_from_ret_dd(r[f"ret_{w}"], r[f"max_dd_{w}"]), axis=1)
    return events.reset_index(drop=True)

# =========================
# Strategy analyses
# =========================
def build_c2_events(df: pd.DataFrame, dsa_thr: float = 0.30, bars_thr: int = 3, vwap_thr: Optional[float] = -1.0, cooldown: int = EVENT_DEDUP_BARS) -> pd.DataFrame:
    need = [
        "symbol", "datetime", "open", "high", "low", "close",
        "dsa_trade_pos_01", "dsa_confirmed_pivot_pos_01", "dsa_signed_vwap_dev_pct",
        "rope_dir", "bars_since_dir_change", "rope_slope_atr_5", "bb_pos_01", "bb_width_percentile",
        "prev_confirmed_up_bars", "prev_confirmed_down_bars",
        "w_DSA_DIR", "w_dsa_confirmed_pivot_pos_01", "w_dsa_trade_pos_01", "w_dsa_signed_vwap_dev_pct", "w_factor_available", "w_sample_count",
    ] + [f"ret_{w}" for w in [20, 40, 60]] + [f"max_dd_{w}" for w in [20, 40, 60]] + [f"win_{w}" for w in [20, 40, 60]]
    need = [c for c in need if c in df.columns]
    sub = df[need].dropna(subset=[c for c in ["ret_20", "max_dd_20", "dsa_trade_pos_01", "rope_dir", "bars_since_dir_change"] if c in need]).copy()
    mask = (sub["dsa_trade_pos_01"] <= dsa_thr) & (sub["rope_dir"] == 1) & (sub["bars_since_dir_change"] <= bars_thr)
    if vwap_thr is not None and "dsa_signed_vwap_dev_pct" in sub.columns:
        mask &= sub["dsa_signed_vwap_dev_pct"] <= vwap_thr
    if "bb_pos_01" in sub.columns:
        mask &= sub["bb_pos_01"] <= 0.35
    events = dedup_events(sub[mask].copy(), cooldown)
    for w in [20, 40, 60]:
        if f"ret_{w}" in events.columns:
            events[f"rr_{w}"] = events.apply(lambda r: rr_from_ret_dd(r[f"ret_{w}"], r[f"max_dd_{w}"]), axis=1)
    return events.reset_index(drop=True)


def analyze_05_candidate_events(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑤: Round2 候选策略事件表")
    print("=" * 70)
    rows, all_events = [], []
    for name in ROUND2_STRATEGY_SPECS:
        events = build_round2_events(df, name)
        if len(events) == 0:
            continue
        row = {"strategy": name}
        row.update(summarize_events(events, [20, 60]))
        row = {("event_n" if k == "n" else k): v for k, v in row.items()}
        rows.append(row)
        tmp = events.copy(); tmp["strategy"] = name; all_events.append(tmp)
    if not rows:
        print("  无足够事件，跳过候选策略")
        return pd.DataFrame()
    rdf = pd.DataFrame(rows).sort_values("rr_20", ascending=False)
    print(rdf[["strategy", "event_n", "ret_20", "mae_20", "wr_20", "rr_20", "ret_60", "rr_60"]].to_string(index=False))
    rdf.to_csv(f"{OUT_DIR}/05_candidate_summary.csv", index=False)
    if all_events:
        pd.concat(all_events, ignore_index=True).to_csv(f"{OUT_DIR}/05_candidate_events.csv", index=False)
    return rdf


def analyze_06_c2_param_scan(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑥: Round2 基准策略邻域参数扫描")
    print("=" * 70)
    rows = []
    for dsa_thr in [0.30, 0.35, 0.40]:
        for dist_thr in [-1.0, -0.8, -0.5]:
            for bars_thr in [5, 8, 10]:
                for prev_down_min in [10, 20, 30]:
                    events = build_round2_events(df, "A_low_repair", override={"dsa_thr": dsa_thr, "dist_thr": dist_thr, "bars_thr": bars_thr, "prev_down_min": prev_down_min})
                    if len(events) < 20:
                        continue
                    row = {"dsa_thr": dsa_thr, "dist_thr": dist_thr, "bars_thr": bars_thr, "prev_down_min": prev_down_min, "event_n": len(events)}
                    row.update(summarize_events(events, [20, 40, 60]))
                    rows.append(row)
    if not rows:
        print("  无足够事件，跳过参数扫描")
        return pd.DataFrame()
    rdf = pd.DataFrame(rows)
    rdf["rr_20"] = rdf.get("rr_20", pd.Series(dtype=float)).fillna(-np.inf)
    rdf = rdf.sort_values("rr_20", ascending=False)
    print(rdf.head(12)[["dsa_thr", "dist_thr", "bars_thr", "prev_down_min", "event_n", "ret_20", "mae_20", "wr_20", "rr_20"]].to_string(index=False))
    rdf.to_csv(f"{OUT_DIR}/06_c2_param_scan.csv", index=False)
    return rdf


def analyze_07_c2_time_slices(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑦: Round2 时间切片验证")
    print("=" * 70)
    events = slice_equal_time_buckets(build_round2_events(df, strategy_name), 3)
    rows = []
    for name, g in events.groupby("time_slice", observed=True):
        row = {"time_slice": str(name), "start": g["datetime"].min().strftime("%Y-%m"), "end": g["datetime"].max().strftime("%Y-%m")}
        row.update(summarize_events(g, [20, 60]))
        rows.append(row)
    rdf = pd.DataFrame(rows).sort_values("time_slice") if rows else pd.DataFrame()
    if not rdf.empty:
        print(rdf[["time_slice", "start", "end", "n", "ret_20", "mae_20", "wr_20", "rr_20", "ret_60", "rr_60"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/07_c2_time_slices.csv", index=False)
    return rdf


def analyze_08_c2_weekly_compare(events: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑧: 周线过滤与周线覆盖率对比")
    print("=" * 70)
    if events.empty:
        print("  无事件数据，跳过周线对比")
        return pd.DataFrame()
    variants: List[Tuple[str, pd.Series]] = [("Base", pd.Series(True, index=events.index))]
    if "w_factor_available" in events.columns:
        variants += [("W0_has_weekly", events["w_factor_available"] == 1)]
    if "w_dsa_trade_pos_01" in events.columns:
        variants += [
            ("W1_wtrade_lt_0.7", events["w_dsa_trade_pos_01"] < 0.7),
            ("W2_wtrade_le_0.4", events["w_dsa_trade_pos_01"] <= 0.4),
        ]
    if "w_DSA_DIR" in events.columns:
        variants += [("W3_wconfirmed_dir_up", events["w_DSA_DIR"] == 1)]
    if {"w_dsa_trade_pos_01", "w_DSA_DIR"}.issubset(events.columns):
        variants += [("W4_wtrade_le_0.4_and_wdir_up", (events["w_dsa_trade_pos_01"] <= 0.4) & (events["w_DSA_DIR"] == 1))]
    rows = []
    for name, mask in variants:
        g = events[mask.fillna(False)] if hasattr(mask, "fillna") else events[mask]
        if len(g) < 10:
            continue
        row = {"variant": name, "event_n": len(g)}
        row.update(summarize_events(g, [20, 40]))
        rows.append(row)
    if not rows:
        print("  无足够事件，跳过周线对比")
        return pd.DataFrame()
    rdf = pd.DataFrame(rows)
    # 确保 rr_40 列存在，缺失值填充为 -inf 以便排序
    if "rr_40" not in rdf.columns:
        rdf["rr_40"] = -np.inf
    rdf["rr_40"] = rdf["rr_40"].fillna(-np.inf)
    rdf = rdf.sort_values("rr_40", ascending=False)
    if not rdf.empty:
        print(rdf.to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/08_c2_weekly_compare.csv", index=False)
    return rdf


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="买入点收益风险比探索（全因子优先 + DSA双轨版）")
    parser.add_argument("--n-stocks", type=int, default=100, help="随机抽取股票数")
    parser.add_argument("--bars", type=int, default=800, help="每只股票K线数")
    parser.add_argument("--freq", default="d", help="频率 d/w/mo/60m等")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--analysis-mode", type=str, default="exploration_first", choices=["exploration_only", "exploration_first", "strategy_only"], help="exploration_only=只做全因子探索；exploration_first=先探索后策略；strategy_only=只做策略验证")
    parser.add_argument("--base-strategy", type=str, default="A_low_repair", choices=list(ROUND2_STRATEGY_SPECS.keys()), help="时间切片与周线对比默认使用的基准策略")
    args = parser.parse_args()

    stocks = get_stock_pool(args.n_stocks, args.seed)
    print(f"股票池抽取: {len(stocks)}只 (seed={args.seed})")
    print(f"分析模式: {args.analysis_mode}")
    print("DSA口径说明: confirmed字段仅用于研究解释；策略入口默认使用trade-safe代理字段")

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
                print(f"  [{idx+1}/{len(stocks)}] {code}: {len(kline)}bars → {len(valid)}有效样本")
        except Exception as e:
            print(f"  [{idx+1}/{len(stocks)}] {code} 因子计算失败: {e}")

    if not all_dfs:
        print("无有效数据，退出")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n总样本: {len(df)}, 股票数: {df['symbol'].nunique()}")
    for w in [5, 10, 20, 40, 60]:
        ret = df[f"ret_{w}"].mean()
        dd = df[f"max_dd_{w}"].mean()
        wr = df[f"win_{w}"].mean()
        print(f"  ret_{w}: 均值={ret:+.5f}, max_dd均值={dd:.5f}, 胜率={wr:.4f}, rr={rr_from_ret_dd(ret, dd):.3f}")

    run_exploration = args.analysis_mode in {"exploration_only", "exploration_first"}
    run_strategy = args.analysis_mode in {"strategy_only", "exploration_first"}

    if run_exploration:
        analyze_00_factor_inventory(df)
        analyze_01_single_factor(df)
        analyze_01b_discrete_factor(df)
        analyze_02_scenarios(df)
        analyze_03_interaction(df)
        analyze_03b_factor_pairs(df)
        analyze_04_rule_search(df)

    if run_strategy:
        analyze_05_candidate_events(df)
        analyze_06_c2_param_scan(df)
        base_events = build_round2_events(df, args.base_strategy)
        analyze_07_c2_time_slices(df, args.base_strategy)
        analyze_08_c2_weekly_compare(base_events)

    print(f"\n{'=' * 70}")
    print(f"# 全部完成! 结果保存在 {OUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
