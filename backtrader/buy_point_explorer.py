# -*- coding: utf-8 -*-
"""
基于 DSA VWAP + ATR Rope + Bollinger 因子的买入点收益风险比探索（收敛验证版）

Purpose
  - 从股票池随机抽样，计算三套技术因子
  - 研究低位+转强确认（C2）策略的稳健性、退出、时间切片、周线过滤、执行层排序、市场环境过滤

Dependencies
  - 项目内: datasource.database, features.merged_dsa_atr_rope_bb_factors
  - 可选: tushare（仅实验16需要）

Notes
  - rr = ret / abs(MAE)
  - MAE 使用未来窗口内最低价相对入场价的回撤，long 为负值
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from datasource.database import get_engine
from features.merged_dsa_atr_rope_bb_factors import (
    DSAConfig,
    RopeConfig,
    compute_atr_rope,
    compute_bollinger,
    compute_dsa,
)

warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = "buy_point_explorer"
os.makedirs(OUT_DIR, exist_ok=True)

RET_WINDOWS = [5, 10, 20, 40, 60]
EVENT_DEDUP_BARS = 20


# =========================
# Generic helpers
# =========================
def rr_from_ret_dd(ret: float, dd: float) -> float:
    if pd.isna(ret) or pd.isna(dd) or dd == 0:
        return np.nan
    return float(ret / abs(dd))


def rr_from_means(ret: float, dd: float) -> float:
    return rr_from_ret_dd(ret, dd)


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
    if df.empty:
        return df.copy()
    if "symbol" not in df.columns:
        return df.copy()
    work = df.sort_values(["symbol", "datetime"]).copy()
    keep_idx: List[int] = []
    for sym, g in work.groupby("symbol", sort=False):
        g = g.copy()
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


def map_weekly_dsa_strict_to_daily(daily: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=daily.index)
    wk = aggregate_weekly_strict(daily)
    if len(wk) < 10:
        return out
    w_dsa, _, _ = compute_dsa(wk, DSAConfig(prd=50, base_apt=20.0))
    # strict weekly means only previously completed week is known for each day
    prev_friday = (daily.index.to_period("W-FRI").start_time + pd.offsets.Week(weekday=4) - pd.offsets.Week(1)).normalize()
    mapped = w_dsa.copy()
    mapped.index = mapped.index.normalize()
    out["w_DSA_DIR"] = mapped["DSA_DIR"].reindex(prev_friday).to_numpy()
    out["w_dsa_pivot_pos_01"] = mapped["dsa_pivot_pos_01"].reindex(prev_friday).to_numpy()
    out["w_signed_vwap_dev_pct"] = mapped["signed_vwap_dev_pct"].reindex(prev_friday).to_numpy()
    return out


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "vol" in d.columns and "volume" not in d.columns:
        d["volume"] = d["vol"]
    dsa_df, _, _ = compute_dsa(d, DSAConfig(prd=50, base_apt=20.0))
    rope_df = compute_atr_rope(d, RopeConfig(length=14, multi=1.5))
    bb_df = compute_bollinger(d, length=20, mult=2.0, pct_lookback=120)
    weekly_df = map_weekly_dsa_strict_to_daily(d)
    merged = pd.concat(
        [
            d,
            dsa_df,
            rope_df.drop(columns=d.columns, errors="ignore"),
            bb_df.drop(columns=d.columns, errors="ignore"),
            weekly_df,
        ],
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
            future_low = np.nanmin(low[i + 1 : i + w + 1])
            mae[i] = (future_low - entry) / entry
        out[f"ret_{w}"] = fut
        out[f"max_dd_{w}"] = mae
        out[f"win_{w}"] = (out[f"ret_{w}"] > 0).astype(float)
    return out


FACTOR_COLS = [
    "dsa_pivot_pos_01", "signed_vwap_dev_pct", "trend_aligned_vwap_dev_pct",
    "lh_hh_low_pos", "bull_vwap_dev_pct", "bear_vwap_dev_pct",
    "rope_dir", "dist_to_rope_atr", "rope_slope_atr_5",
    "is_consolidating", "bars_since_dir_change",
    "range_break_up", "range_break_up_strength", "range_width_atr",
    "channel_pos_01", "range_pos_01", "rope_pivot_pos_01",
    "bb_pos_01", "bb_width_norm", "bb_width_percentile",
    "bb_width_change_5", "bb_expanding", "bb_contracting",
    "bb_expand_streak", "bb_contract_streak",
    "w_DSA_DIR", "w_dsa_pivot_pos_01", "w_signed_vwap_dev_pct",
]
DISCRETE_COLS = ["rope_dir", "is_consolidating", "range_break_up", "bb_expanding", "bb_contracting", "w_DSA_DIR"]
CONTINUOUS_COLS = [c for c in FACTOR_COLS if c not in DISCRETE_COLS]


# =========================
# Existing analyses 01-04
# =========================
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
                "IC": round(ic, 4),
                "Rank_IC": round(rank_ic, 4),
                "best_Q_rr": best_q,
                "Q1_rr": round(grp.loc["Q1", "risk_reward"], 3) if "Q1" in grp.index else np.nan,
                "Q5_rr": round(grp.loc["Q5", "risk_reward"], 3) if "Q5" in grp.index else np.nan,
                "Q1_ret20": round(grp.loc["Q1", "mean_ret20"], 5) if "Q1" in grp.index else np.nan,
                "Q5_ret20": round(grp.loc["Q5", "mean_ret20"], 5) if "Q5" in grp.index else np.nan,
            })
        except Exception as e:
            results.append({"factor": col, "error": str(e)[:80]})
    rdf = pd.DataFrame(results).sort_values("Rank_IC", key=lambda s: s.abs(), ascending=False)
    if not rdf.empty:
        print(rdf.head(20)[["factor", "IC", "Rank_IC", "best_Q_rr", "Q1_rr", "Q5_rr", "Q1_ret20", "Q5_ret20"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/01_single_factor.csv", index=False)
    return rdf


SCENARIOS = {
    "S1_Rope触底反弹": lambda d: (d["rope_dir"] == 1) & (d["rope_dir"].shift(1) == -1) & (d["dist_to_rope_atr"] < 0.5),
    "S2_盘整突破": lambda d: (d["range_break_up"] == 1) & (d["range_width_atr"] < d["range_width_atr"].median()),
    "S3_BB下轨回弹": lambda d: (d["bb_pos_01"] < 0.15) & (d["close"] > d["open"]),
    "S4_BB_Squeeze突破": lambda d: (d["bb_width_percentile"] < 0.2) & (d["bb_expanding"] == 1),
    "S5_DSA_VWAP回归": lambda d: (d["trend_aligned_vwap_dev_pct"] < -2) & (d["DSA_DIR"] == 1),
    "S6_通道下轨支撑": lambda d: (d["channel_pos_01"] < 0.2) & (d["rope_dir"] == 1),
    "S7_Rope动量启动": lambda d: (d["rope_slope_atr_5"] > 0) & (d["rope_dir"] == 1) & (d["bars_since_dir_change"] < 5),
    "S9_DSA低位+Rope向上": lambda d: (d["dsa_pivot_pos_01"] < 0.4) & (d["rope_dir"] == 1),
    "S10_BB下轨+Rope支撑": lambda d: (d["bb_pos_01"] < 0.2) & (d["dist_to_rope_atr"] > -0.5),
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
    need = list(dict.fromkeys(FACTOR_COLS + ["DSA_DIR", "open", "close"] + [f"ret_{w}" for w in [5,10,20,60]] + [f"max_dd_{w}" for w in [5,10,20,60]] + [f"win_{w}" for w in [5,10,20,60]]))
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
        print(rdf[[c for c in ["scenario","n","coverage_pct","ret20","dd20","wr20","rr20","ret60","rr60"] if c in rdf.columns]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/02_scenarios.csv", index=False)
    return rdf


def analyze_03_interaction(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("# 层次③: 因子交互效应分析 (四象限)")
    print("=" * 70)
    need = ["rope_dir", "bb_pos_01", "is_consolidating", "range_break_up", "dsa_pivot_pos_01", "rope_slope_atr_5", "bb_width_percentile", "ret_20", "max_dd_20", "win_20"]
    sub = df[[c for c in need if c in df.columns]].dropna()
    if len(sub) == 0:
        return
    inters = [
        ("rope_dir × bb_pos_01", "rope_dir", pd.cut(sub["bb_pos_01"], bins=[0, 0.2, 0.5, 0.8, 1.01], labels=["超卖", "偏弱", "偏强", "超买"], include_lowest=True)),
        ("dsa_pivot_pos × rope_slope", pd.cut(sub["dsa_pivot_pos_01"], bins=[0, 0.3, 0.7, 1.01], labels=["低位", "中位", "高位"], include_lowest=True), pd.cut(sub["rope_slope_atr_5"], bins=[-np.inf, -0.01, 0.01, 0.05, np.inf], labels=["强跌", "微跌", "微涨", "强涨"]))
    ]
    for label, a, b in inters:
        tmp = sub.copy()
        tmp["a"] = a if isinstance(a, pd.Series) else tmp[a]
        tmp["b"] = b if isinstance(b, pd.Series) else tmp[b]
        g = tmp.groupby(["a", "b"], observed=True).agg(n=("ret_20", "count"), ret20=("ret_20", "mean"), dd20=("max_dd_20", "mean"), wr20=("win_20", "mean"))
        g["rr20"] = g.apply(lambda r: rr_from_ret_dd(r["ret20"], r["dd20"]), axis=1).round(3)
        print(f"\n--- {label} ---")
        print(g.to_string())


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
        return {"rule": name, "n": len(g), "coverage_pct": round(len(g) / baseline_n * 100, 1), "ret20": round(ret,5), "dd20": round(dd,5), "wr20": round(wr,4), "rr20": round(rr_from_ret_dd(ret, dd),3)}

    results: List[Dict[str, float]] = []
    candidates = [
        ("dsa_pivot_pos_01", [0.3, 0.4, 0.5]),
        ("signed_vwap_dev_pct", [-3, -2]),
        ("bb_pos_01", [0.15, 0.2, 0.25]),
        ("bars_since_dir_change", [3, 5]),
        ("bb_width_percentile", [0.2, 0.25, 0.3]),
    ]
    for col, vals in candidates:
        if col not in sub.columns:
            continue
        for hi in vals:
            mask = sub[col] <= hi
            r = eval_rule(mask, f"{col}<={hi}")
            if r:
                results.append(r)
    combos = {
        "dsa<0.4&窄幅突破": (sub.get("dsa_pivot_pos_01", np.nan) < 0.4) & (sub.get("range_break_up", 0) == 1) & (sub.get("range_width_atr", np.inf) < sub["range_width_atr"].median()),
        "dsa<0.4&vwap_dev<-2": (sub.get("dsa_pivot_pos_01", np.nan) < 0.4) & (sub.get("signed_vwap_dev_pct", np.inf) <= -2),
        "slope>0&vwap_dev<-2": (sub.get("rope_slope_atr_5", -np.inf) > 0) & (sub.get("signed_vwap_dev_pct", np.inf) <= -2),
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
# C2 event builders and analyses 05-13
# =========================
def build_c2_events(df: pd.DataFrame, dsa_thr: float = 0.30, bars_thr: int = 3, vwap_thr: Optional[float] = -1.0, cooldown: int = EVENT_DEDUP_BARS) -> pd.DataFrame:
    need = [
        "symbol", "datetime", "open", "high", "low", "close",
        "dsa_pivot_pos_01", "signed_vwap_dev_pct", "rope_dir", "bars_since_dir_change",
        "rope_slope_atr_5", "bb_pos_01", "bb_width_percentile",
        "w_DSA_DIR", "w_dsa_pivot_pos_01", "w_signed_vwap_dev_pct",
    ] + [f"ret_{w}" for w in [20, 40, 60]] + [f"max_dd_{w}" for w in [20, 40, 60]] + [f"win_{w}" for w in [20, 40, 60]]
    need = [c for c in need if c in df.columns]
    sub = df[need].dropna(subset=[c for c in ["ret_20", "max_dd_20", "dsa_pivot_pos_01", "rope_dir", "bars_since_dir_change"] if c in need]).copy()
    mask = (sub["dsa_pivot_pos_01"] <= dsa_thr) & (sub["rope_dir"] == 1) & (sub["bars_since_dir_change"] <= bars_thr)
    if vwap_thr is not None and "signed_vwap_dev_pct" in sub.columns:
        mask &= sub["signed_vwap_dev_pct"] <= vwap_thr
    events = dedup_events(sub[mask].copy(), cooldown)
    for w in [20, 40, 60]:
        if f"ret_{w}" in events.columns:
            events[f"rr_{w}"] = events.apply(lambda r: rr_from_ret_dd(r[f"ret_{w}"], r[f"max_dd_{w}"]), axis=1)
    return events.reset_index(drop=True)


def analyze_05_candidate_events(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑤: 候选策略事件表")
    print("=" * 70)
    variants = {
        "C1_纯低位反转": dict(dsa_thr=0.30, bars_thr=999, vwap_thr=-1.0),
        "C2_低位+转强确认": dict(dsa_thr=0.35, bars_thr=3, vwap_thr=-1.0),
        "C3_低位+窄幅突破": dict(dsa_thr=0.40, bars_thr=3, vwap_thr=-2.0),
    }
    rows = []
    all_events = []
    for name, cfg in variants.items():
        events = build_c2_events(df, **cfg)
        if len(events) == 0:
            continue
        row = {"strategy": name}
        row.update(summarize_events(events, [20, 60]))
        row = {("event_n" if k == "n" else k): v for k, v in row.items()}
        rows.append(row)
        tmp = events.copy()
        tmp["strategy"] = name
        all_events.append(tmp)
    rdf = pd.DataFrame(rows).sort_values("rr_20", ascending=False)
    if not rdf.empty:
        print(rdf[["strategy", "event_n", "ret_20", "mae_20", "wr_20", "rr_20", "ret_60", "rr_60"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/05_candidate_summary.csv", index=False)
        if all_events:
            pd.concat(all_events, ignore_index=True).to_csv(f"{OUT_DIR}/05_candidate_events.csv", index=False)
    return rdf


def analyze_06_c2_param_scan(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑥: C2 参数扫描")
    print("=" * 70)
    rows = []
    for dsa_thr in [0.25, 0.30, 0.35, 0.40]:
        for bars_thr in [3, 5, 8]:
            for vwap_thr in [None, -1.0, -2.0, -3.0]:
                events = build_c2_events(df, dsa_thr=dsa_thr, bars_thr=bars_thr, vwap_thr=vwap_thr)
                if len(events) < 30:
                    continue
                row = {"dsa_thr": dsa_thr, "bars_thr": bars_thr, "vwap_thr": "none" if vwap_thr is None else vwap_thr, "event_n": len(events)}
                row.update(summarize_events(events, [20, 40, 60]))
                rows.append(row)
    rdf = pd.DataFrame(rows).sort_values("rr_20", ascending=False)
    if not rdf.empty:
        print(rdf.head(10)[["dsa_thr", "bars_thr", "vwap_thr", "event_n", "ret_20", "mae_20", "wr_20", "rr_20"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/06_c2_param_scan.csv", index=False)
    return rdf


def analyze_07_c2_exit_compare(df: pd.DataFrame, dsa_thr: float, bars_thr: int, vwap_thr: Optional[float]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑦: C2 退出规则对比")
    print("=" * 70)
    events = build_c2_events(df, dsa_thr, bars_thr, vwap_thr)
    rows = []
    for hold in [10, 20, 40]:
        rows.append({"exit_rule": f"固定持有{hold}", **summarize_events(events, [hold]), "avg_hold_bars": hold})
    # repair exit: proxy based on 20 bars summary
    rows.append({"exit_rule": "修复完成", **summarize_events(events, [20]), "avg_hold_bars": 13})
    weaken = events[events.get("rope_slope_atr_5", pd.Series(index=events.index, dtype=float)) < 0]
    rows.append({"exit_rule": "Rope转弱", **summarize_events(weaken if len(weaken) else events.iloc[:0], [20]), "avg_hold_bars": 5})
    for stop in [0.05, 0.07, 0.08]:
        hit = events[events["max_dd_40"] <= -stop]
        rows.append({"exit_rule": f"止损{int(stop*100)}%", **summarize_events(hit, [40]), "avg_hold_bars": 40})
    rdf = pd.DataFrame(rows)
    rr_cols = [c for c in rdf.columns if c.startswith("rr_")]
    if not rdf.empty:
        cols = ["exit_rule", "n"] + [c for c in rdf.columns if c in ["avg_hold_bars"] or c.startswith(("ret_", "mae_", "wr_", "rr_"))]
        print(rdf[cols].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/07_c2_exit_compare.csv", index=False)
    return rdf


def analyze_08_c2_time_slices(df: pd.DataFrame, dsa_thr: float, bars_thr: int, vwap_thr: Optional[float]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑧: C2 时间切片验证")
    print("=" * 70)
    events = slice_equal_time_buckets(build_c2_events(df, dsa_thr, bars_thr, vwap_thr), 3)
    rows = []
    for name, g in events.groupby("time_slice", observed=True):
        row = {"time_slice": str(name), "start": g["datetime"].min().strftime("%Y-%m"), "end": g["datetime"].max().strftime("%Y-%m")}
        row.update(summarize_events(g, [20, 60]))
        rows.append(row)
    rdf = pd.DataFrame(rows).sort_values("time_slice")
    if not rdf.empty:
        print(rdf[["time_slice", "start", "end", "n", "ret_20", "mae_20", "wr_20", "rr_20", "ret_60", "rr_60"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/08_c2_time_slices.csv", index=False)
    return rdf


def analyze_09_c2_fixed_validation(df: pd.DataFrame, base_events: pd.DataFrame, stock_counts: Sequence[int]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑨: C2 固定参数大样本验证")
    print("=" * 70)
    rows = []
    syms = sorted(df["symbol"].dropna().unique().tolist())
    for n_stocks in stock_counts:
        keep = set(syms[: min(n_stocks, len(syms))])
        g = base_events[base_events["symbol"].isin(keep)]
        if len(g) < 20:
            continue
        row = {"n_stocks": min(n_stocks, len(syms)), "event_n": len(g), "coverage_pct": round(len(g) / len(base_events) * 100, 1) if len(base_events) else np.nan}
        row.update(summarize_events(g, [20, 40]))
        rows.append(row)
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        print(rdf[["n_stocks", "event_n", "coverage_pct", "ret_20", "mae_20", "wr_20", "rr_20", "ret_40", "rr_40"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/09_c2_fixed_validation.csv", index=False)
    return rdf


def analyze_10_c2_local_grid(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑩: C2 邻域稳健性测试")
    print("=" * 70)
    rows = []
    for dsa_thr in [0.30, 0.35, 0.40]:
        for bars_thr in [2, 3, 5]:
            for vwap_thr in [None, -1.0, -2.0]:
                events = build_c2_events(df, dsa_thr, bars_thr, vwap_thr)
                if len(events) < 20:
                    continue
                row = {"dsa_thr": dsa_thr, "bars_thr": bars_thr, "vwap_thr": "none" if vwap_thr is None else vwap_thr, "event_n": len(events)}
                row.update(summarize_events(events, [20, 40]))
                rows.append(row)
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        rdf["ret_rank"] = rdf["ret_20"].rank(ascending=False, method="dense")
        rdf["rr_rank"] = rdf["rr_20"].rank(ascending=False, method="dense")
        rdf = rdf.sort_values(["rr_20", "ret_20"], ascending=False)
        print(rdf.head(15)[["dsa_thr", "bars_thr", "vwap_thr", "event_n", "ret_20", "mae_20", "wr_20", "rr_20", "ret_rank", "rr_rank"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/10_c2_local_grid.csv", index=False)
    return rdf


def bucketize(series: pd.Series, mode: str) -> pd.Series:
    s = series.astype(float)
    if mode == "vol":
        return pd.cut(s, bins=[-np.inf, 0.2, 0.8, np.inf], labels=["low_vol", "mid_vol", "high_vol"], include_lowest=True)
    if mode == "slope":
        q1, q2 = s.quantile([0.33, 0.67])
        return pd.cut(s, bins=[-np.inf, q1, q2, np.inf], labels=["weak", "mid", "strong"], include_lowest=True)
    if mode == "bbpos":
        return pd.cut(s, bins=[-np.inf, 0.33, 0.66, np.inf], labels=["low", "mid", "high"], include_lowest=True)
    if mode == "vwap":
        return pd.cut(s, bins=[-np.inf, -2, -1, np.inf], labels=["deep", "medium", "mild"], include_lowest=True)
    raise ValueError(mode)


def analyze_11_c2_t3_regime_breakdown(events: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑪: C2 T3 走弱拆解")
    print("=" * 70)
    tmp = slice_equal_time_buckets(events, 3)
    t3 = tmp[tmp["time_slice"] == "T3"].copy()
    rows = []
    if len(t3) == 0:
        return pd.DataFrame()
    regimes = {
        "bb_width_percentile": bucketize(t3["bb_width_percentile"], "vol"),
        "rope_slope_atr_5": bucketize(t3["rope_slope_atr_5"], "slope"),
        "bb_pos_01": bucketize(t3["bb_pos_01"], "bbpos"),
        "signed_vwap_dev_pct": bucketize(t3["signed_vwap_dev_pct"], "vwap"),
    }
    for regime_type, buckets in regimes.items():
        work = t3.copy(); work["bucket"] = buckets
        for bucket, g in work.groupby("bucket", observed=True):
            if len(g) < 10:
                continue
            row = {"regime_type": regime_type, "bucket": str(bucket), "event_n": len(g)}
            row.update(summarize_events(g, [20, 40]))
            rows.append(row)
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        print(rdf[["regime_type", "bucket", "event_n", "ret_20", "mae_20", "wr_20", "rr_20", "ret_40", "rr_40"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/11_c2_t3_regime_breakdown.csv", index=False)
    return rdf


def apply_max_signals_per_day(events: pd.DataFrame, max_signals_per_day: int, rank_by: str) -> pd.DataFrame:
    if max_signals_per_day <= 0 or events.empty:
        return events.copy()
    e = events.copy()
    e["trade_date"] = pd.to_datetime(e["datetime"]).dt.normalize()
    ascending = True
    if rank_by in {"signed_vwap_dev_pct", "dsa_pivot_pos_01", "bars_since_dir_change"}:
        ascending = True
    elif rank_by in {"score_2f", "score_3f"}:
        ascending = False
    else:
        ascending = True
    out = []
    for _, g in e.groupby("trade_date", sort=True):
        if rank_by not in g.columns:
            out.append(g.head(max_signals_per_day))
        else:
            out.append(g.sort_values(rank_by, ascending=ascending).head(max_signals_per_day))
    return pd.concat(out, ignore_index=True) if out else e.iloc[:0].copy()


def add_rank_scores(events: pd.DataFrame) -> pd.DataFrame:
    e = events.copy()
    e["early_turn_score"] = 1.0 / (1.0 + e["bars_since_dir_change"].astype(float).clip(lower=0.0))
    e["dsa_low_score"] = 1.0 - normalize_score(e["dsa_pivot_pos_01"], reverse=False)
    # use discount only for negative deviations; cap at 3%
    disc = (-e["signed_vwap_dev_pct"]).clip(lower=0, upper=3.0) / 3.0
    e["vwap_discount_score"] = disc
    e["score_2f"] = 0.6 * e["dsa_low_score"] + 0.4 * e["early_turn_score"]
    e["score_3f"] = 0.5 * e["dsa_low_score"] + 0.3 * e["early_turn_score"] + 0.2 * e["vwap_discount_score"]
    return e


def analyze_12_c2_execution_filter(events: pd.DataFrame, max_signals_per_day: int, signal_rank_by: str) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑫: C2 执行层信号限制验证")
    print("=" * 70)
    e = add_rank_scores(events)
    after = apply_max_signals_per_day(e, max_signals_per_day, signal_rank_by)
    rows = []
    for stage, g in [("before", e), ("after", after)]:
        row = {"stage": stage, "event_n": len(g), "max_signals_per_day": max_signals_per_day, "signal_rank_by": signal_rank_by}
        row.update(summarize_events(g, [20, 40]))
        rows.append(row)
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        print(rdf[["stage", "event_n", "ret_20", "mae_20", "wr_20", "rr_20", "ret_40", "rr_40"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/12_c2_execution_filter.csv", index=False)
    return rdf


def analyze_13_c2_weekly_dsa_filter(events: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑬: C2 + 周线DSA背景过滤")
    print("=" * 70)
    rows = []
    tests: List[Tuple[str, pd.Series]] = []
    if "w_DSA_DIR" in events.columns:
        tests.append(("strict_dir", events["w_DSA_DIR"].map(lambda x: "up" if x == 1 else "down" if x == -1 else "other")))
    if "w_dsa_pivot_pos_01" in events.columns:
        tests.append(("strict_pos", pd.cut(events["w_dsa_pivot_pos_01"], bins=[-np.inf, 0.4, 0.7, np.inf], labels=["low", "mid", "high"], include_lowest=True)))
    for regime_type, buckets in tests:
        work = events.copy(); work["bucket"] = buckets
        for bucket, g in work.groupby("bucket", observed=True):
            if len(g) < 10:
                continue
            row = {"regime_type": regime_type, "bucket": str(bucket), "event_n": len(g)}
            row.update(summarize_events(g, [20, 40]))
            rows.append(row)
    # explicit combo groups
    if "w_dsa_pivot_pos_01" in events.columns:
        for label, mask in {
            "strict_non_high": events["w_dsa_pivot_pos_01"] < 0.7,
            "strict_low": events["w_dsa_pivot_pos_01"] <= 0.4,
            "strict_low_and_daily_low": (events["w_dsa_pivot_pos_01"] <= 0.5) & (events["dsa_pivot_pos_01"] <= 0.30),
        }.items():
            g = events[mask]
            if len(g) >= 10:
                row = {"regime_type": "weekly_combo", "bucket": label, "event_n": len(g)}
                row.update(summarize_events(g, [20, 40]))
                rows.append(row)
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        print(rdf[["regime_type", "bucket", "event_n", "ret_20", "mae_20", "wr_20", "rr_20", "ret_40", "rr_40"]].to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/13_c2_weekly_dsa_filter.csv", index=False)
    return rdf


# =========================
# New layers 14-17
# =========================
def build_c2_events_base(df: pd.DataFrame, dsa_thr: float = 0.30, bars_thr: int = 3, vwap_thr: Optional[float] = -1.0, cooldown: int = EVENT_DEDUP_BARS, hold_bars: int = 40) -> pd.DataFrame:
    _ = hold_bars
    return add_rank_scores(build_c2_events(df, dsa_thr=dsa_thr, bars_thr=bars_thr, vwap_thr=vwap_thr, cooldown=cooldown))


def analyze_14_c2_weekly_filter_compare(events: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑭: 严格周线 DSA 过滤对比")
    print("=" * 70)
    variants: List[Tuple[str, pd.Series]] = [("Base", pd.Series(True, index=events.index))]
    if "w_dsa_pivot_pos_01" in events.columns:
        variants += [
            ("W1_wpos_lt_0.7", events["w_dsa_pivot_pos_01"] < 0.7),
            ("W2_wpos_le_0.4", events["w_dsa_pivot_pos_01"] <= 0.4),
        ]
    if "w_DSA_DIR" in events.columns:
        variants += [("W3_wdir_up", events["w_DSA_DIR"] == 1)]
    if {"w_dsa_pivot_pos_01", "w_DSA_DIR"}.issubset(events.columns):
        variants += [("W4_wpos_le_0.4_and_wdir_up", (events["w_dsa_pivot_pos_01"] <= 0.4) & (events["w_DSA_DIR"] == 1))]
    if "w_signed_vwap_dev_pct" in events.columns:
        variants += [("W5_wdev_le_0", events["w_signed_vwap_dev_pct"] <= 0)]
    rows = []
    for name, mask in variants:
        g = events[mask.fillna(False)] if hasattr(mask, "fillna") else events[mask]
        if len(g) < 10:
            continue
        row = {"variant": name, "event_n": len(g)}
        row.update(summarize_events(g, [20, 40]))
        rows.append(row)
    rdf = pd.DataFrame(rows).sort_values("rr_40", ascending=False)
    if not rdf.empty:
        print(rdf.to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/14_c2_weekly_filter_compare.csv", index=False)
    return rdf


def rank_events_with_method(events_df: pd.DataFrame, method: str, max_signals_per_day: int) -> pd.DataFrame:
    e = add_rank_scores(events_df)
    rank_col = {
        "signed_vwap_dev_pct": "signed_vwap_dev_pct",
        "dsa_pivot_pos_01": "dsa_pivot_pos_01",
        "bars_since_dir_change": "bars_since_dir_change",
        "score_2f": "score_2f",
        "score_3f": "score_3f",
    }.get(method, method)
    return apply_max_signals_per_day(e, max_signals_per_day, rank_col)


def analyze_15_c2_execution_rank_compare(events: pd.DataFrame, max_signals_per_day: int) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑮: 执行层排序对比")
    print("=" * 70)
    methods = ["signed_vwap_dev_pct", "dsa_pivot_pos_01", "bars_since_dir_change", "score_2f", "score_3f"]
    rows = []
    base = add_rank_scores(events)
    base_row = {"rank_method": "no_limit", "max_signals_per_day": max_signals_per_day, "event_n_before": len(base), "event_n_after": len(base)}
    s = summarize_events(base, [20, 40])
    for k, v in s.items():
        if k == "n":
            continue
        base_row[f"{k}_after"] = v
    rows.append(base_row)
    for method in methods:
        g = rank_events_with_method(events, method, max_signals_per_day)
        row = {"rank_method": method, "max_signals_per_day": max_signals_per_day, "event_n_before": len(base), "event_n_after": len(g)}
        s = summarize_events(g, [20, 40])
        for k, v in s.items():
            if k == "n":
                continue
            row[f"{k}_after"] = v
        rows.append(row)
    rdf = pd.DataFrame(rows).sort_values("rr_40_after", ascending=False)
    if not rdf.empty:
        print(rdf.to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/15_c2_execution_rank_compare.csv", index=False)
    return rdf


def load_market_index_from_tushare(index_code: str, start_date: str, end_date: str, token: Optional[str] = None) -> pd.DataFrame:
    cache_dir = os.path.join(OUT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"market_{index_code.replace('.', '_')}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date")
    else:
        try:
            import tushare as ts
        except Exception as exc:
            raise RuntimeError("实验16需要安装 tushare") from exc
        token = token or os.getenv("TUSHARE_TOKEN", "")
        if not token:
            raise RuntimeError("实验16需要 tushare token，请传 --tushare-token 或设置 TUSHARE_TOKEN")
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.index_daily(ts_code=index_code, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
        if df.empty:
            raise RuntimeError(f"Tushare 指数无数据: {index_code}")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date")
        df.to_csv(cache_path, index=False)
    df = df[(df["trade_date"] >= pd.to_datetime(start_date)) & (df["trade_date"] <= pd.to_datetime(end_date))].copy()
    out = pd.DataFrame(index=pd.to_datetime(df["trade_date"]))
    out["open"] = df["open"].astype(float).to_numpy()
    out["high"] = df["high"].astype(float).to_numpy()
    out["low"] = df["low"].astype(float).to_numpy()
    out["close"] = df["close"].astype(float).to_numpy()
    out["volume"] = df.get("vol", pd.Series(np.ones(len(df)))).astype(float).replace(0, np.nan).ffill().fillna(1.0).to_numpy()
    return out.sort_index()


def compute_market_regime_features(index_df: pd.DataFrame) -> pd.DataFrame:
    dsa_df, _, _ = compute_dsa(index_df.copy(), DSAConfig(prd=50, base_apt=20.0))
    bb_df = compute_bollinger(index_df.copy(), length=20, mult=2.0, pct_lookback=120)
    out = pd.DataFrame(index=index_df.index)
    out["mkt_DSA_DIR"] = dsa_df["DSA_DIR"]
    out["mkt_dsa_pivot_pos_01"] = dsa_df["dsa_pivot_pos_01"]
    out["mkt_signed_vwap_dev_pct"] = dsa_df["signed_vwap_dev_pct"]
    out["mkt_bb_width_percentile"] = bb_df["bb_width_percentile"]
    return out


def merge_market_features_to_events(events_df: pd.DataFrame, market_features_df: pd.DataFrame) -> pd.DataFrame:
    e = events_df.copy()
    e["trade_date"] = pd.to_datetime(e["datetime"]).dt.normalize()
    m = market_features_df.copy()
    m.index = pd.to_datetime(m.index).normalize()
    return e.merge(m, left_on="trade_date", right_index=True, how="left")


def analyze_16_c2_market_regime_compare(events: pd.DataFrame, market_events: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑯: 市场环境过滤对比")
    print("=" * 70)
    variants: List[Tuple[str, pd.Series]] = [("M0_no_filter", pd.Series(True, index=market_events.index))]
    if "mkt_dsa_pivot_pos_01" in market_events.columns:
        variants += [
            ("M1_mkt_pos_lt_0.7", market_events["mkt_dsa_pivot_pos_01"] < 0.7),
            ("M2_mkt_pos_le_0.4", market_events["mkt_dsa_pivot_pos_01"] <= 0.4),
        ]
    if "mkt_DSA_DIR" in market_events.columns:
        variants += [("M3_mkt_dir_up", market_events["mkt_DSA_DIR"] == 1)]
    if "mkt_bb_width_percentile" in market_events.columns:
        variants += [("M4_mkt_bbwidth_lt_0.8", market_events["mkt_bb_width_percentile"] < 0.8)]
    if {"mkt_dsa_pivot_pos_01", "mkt_bb_width_percentile"}.issubset(market_events.columns):
        variants += [("M5_mkt_pos_lt_0.7_and_bbwidth_lt_0.8", (market_events["mkt_dsa_pivot_pos_01"] < 0.7) & (market_events["mkt_bb_width_percentile"] < 0.8))]
    if {"mkt_dsa_pivot_pos_01", "mkt_DSA_DIR"}.issubset(market_events.columns):
        variants += [("M6_mkt_pos_le_0.4_and_dir_up", (market_events["mkt_dsa_pivot_pos_01"] <= 0.4) & (market_events["mkt_DSA_DIR"] == 1))]
    rows = []
    for name, mask in variants:
        g = market_events[mask.fillna(False)] if hasattr(mask, "fillna") else market_events[mask]
        if len(g) < 10:
            continue
        row = {"variant": name, "event_n": len(g)}
        row.update(summarize_events(g, [20, 40]))
        rows.append(row)
    rdf = pd.DataFrame(rows).sort_values("rr_40", ascending=False)
    if not rdf.empty:
        print(rdf.to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/16_c2_market_regime_compare.csv", index=False)
    return rdf


def analyze_17_c2_local_state_compare(events: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("# 层次⑰: 个股局部状态过滤对比")
    print("=" * 70)
    variants: List[Tuple[str, pd.Series]] = [("S0_no_filter", pd.Series(True, index=events.index))]
    if "bb_width_percentile" in events.columns:
        variants += [("S1_bbwidth_lt_0.8", events["bb_width_percentile"] < 0.8), ("S2_bbwidth_le_0.5", events["bb_width_percentile"] <= 0.5)]
    if "bb_pos_01" in events.columns:
        variants += [("S3_bbpos_lt_0.8", events["bb_pos_01"] < 0.8)]
    if "rope_slope_atr_5" in events.columns:
        variants += [("S4_rope_slope_gt_0", events["rope_slope_atr_5"] > 0)]
    if {"bb_width_percentile", "rope_slope_atr_5"}.issubset(events.columns):
        variants += [("S5_bbwidth_lt_0.8_and_slope_gt_0", (events["bb_width_percentile"] < 0.8) & (events["rope_slope_atr_5"] > 0))]
    if {"bb_width_percentile", "bb_pos_01"}.issubset(events.columns):
        variants += [("S6_bbwidth_lt_0.8_and_bbpos_lt_0.8", (events["bb_width_percentile"] < 0.8) & (events["bb_pos_01"] < 0.8))]
    rows = []
    for name, mask in variants:
        g = events[mask.fillna(False)] if hasattr(mask, "fillna") else events[mask]
        if len(g) < 10:
            continue
        row = {"variant": name, "event_n": len(g)}
        row.update(summarize_events(g, [20, 40]))
        rows.append(row)
    rdf = pd.DataFrame(rows).sort_values("rr_40", ascending=False)
    if not rdf.empty:
        print(rdf.to_string(index=False))
        rdf.to_csv(f"{OUT_DIR}/17_c2_local_state_compare.csv", index=False)
    return rdf


# =========================
# Rope coil explosion analyses 18
# =========================
def _true_streak(mask: pd.Series) -> pd.Series:
    m = mask.fillna(False).astype(bool)
    grp = (~m).cumsum()
    return m.groupby(grp).cumsum().astype(float)


def add_rope_coil_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dist = out.get("dist_to_rope_atr", pd.Series(np.nan, index=out.index)).astype(float)
    out["near_rope_loose"] = (dist.abs() <= 0.5).astype(int)
    out["near_rope_mid"] = (dist.abs() <= 0.4).astype(int)
    out["near_rope_strict"] = (dist.abs() <= 0.3).astype(int)
    if "symbol" in out.columns:
        out["near_rope_streak_loose"] = out.groupby("symbol", sort=False)["near_rope_loose"].transform(lambda s: _true_streak(s.astype(bool)))
        out["near_rope_streak_mid"] = out.groupby("symbol", sort=False)["near_rope_mid"].transform(lambda s: _true_streak(s.astype(bool)))
        out["near_rope_streak_strict"] = out.groupby("symbol", sort=False)["near_rope_strict"].transform(lambda s: _true_streak(s.astype(bool)))
        out["near_rope_ratio_5_loose"] = out.groupby("symbol", sort=False)["near_rope_loose"].transform(lambda s: s.rolling(5, min_periods=5).mean())
        out["near_rope_ratio_5_mid"] = out.groupby("symbol", sort=False)["near_rope_mid"].transform(lambda s: s.rolling(5, min_periods=5).mean())
        out["near_rope_ratio_5_strict"] = out.groupby("symbol", sort=False)["near_rope_strict"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    else:
        out["near_rope_streak_loose"] = _true_streak(out["near_rope_loose"].astype(bool))
        out["near_rope_streak_mid"] = _true_streak(out["near_rope_mid"].astype(bool))
        out["near_rope_streak_strict"] = _true_streak(out["near_rope_strict"].astype(bool))
        out["near_rope_ratio_5_loose"] = out["near_rope_loose"].rolling(5, min_periods=5).mean()
        out["near_rope_ratio_5_mid"] = out["near_rope_mid"].rolling(5, min_periods=5).mean()
        out["near_rope_ratio_5_strict"] = out["near_rope_strict"].rolling(5, min_periods=5).mean()

    bbwp = out.get("bb_width_percentile", pd.Series(np.nan, index=out.index)).astype(float)
    out["coil_loose"] = (((out.get("is_consolidating", 0) == 1) | (bbwp <= 0.40) | (out.get("bb_contracting", 0) == 1))).astype(int)
    out["coil_mid"] = (((out.get("is_consolidating", 0) == 1) & (bbwp <= 0.35)) | (bbwp <= 0.25)).astype(int)
    out["coil_strict"] = (((out.get("is_consolidating", 0) == 1) & (bbwp <= 0.25)) | ((out.get("bb_contract_streak", 0).fillna(0) >= 2) & (bbwp <= 0.30))).astype(int)

    out["trigger_slope_up"] = (out.get("rope_slope_atr_5", 0).astype(float) > 0).astype(int)
    out["trigger_above_rope"] = (dist > 0).astype(int)
    out["trigger_depart_from_rope"] = (dist > 0.5).astype(int)
    out["trigger_break_up"] = (out.get("range_break_up", 0) == 1).astype(int)
    rbs = out.get("range_break_up_strength", pd.Series(np.nan, index=out.index)).astype(float)
    out["trigger_break_up_strong"] = ((out["trigger_break_up"] == 1) & (rbs >= rbs.median(skipna=True))).astype(int)
    out["trigger_dual_confirm"] = ((out["trigger_break_up"] == 1) & (out["trigger_above_rope"] == 1) & (out["trigger_depart_from_rope"] == 1)).astype(int)

    out["not_too_high"] = (((out.get("bb_pos_01", pd.Series(0.5, index=out.index)).astype(float) <= 0.90) &
                             (out.get("dsa_pivot_pos_01", pd.Series(0.5, index=out.index)).astype(float) <= 0.90))).astype(int)
    for tag, thr in [("10", 0.10), ("15", 0.15)]:
        out[f"hit20_{tag}"] = (out.get("ret_20", pd.Series(np.nan, index=out.index)).astype(float) >= thr).astype(float)
    out["hit40_20"] = (out.get("ret_40", pd.Series(np.nan, index=out.index)).astype(float) >= 0.20).astype(float)
    return out


def summarize_events_plus(events: pd.DataFrame, windows: Sequence[int] = (20, 40)) -> Dict[str, float]:
    out = summarize_events(events, windows)
    for c in ["hit20_10", "hit20_15", "hit40_20"]:
        out[c] = round(float(events[c].mean()), 4) if c in events.columns and len(events) else np.nan
    return out


def build_rope_coil_events(df: pd.DataFrame, variant: str = "A", cooldown: int = EVENT_DEDUP_BARS) -> pd.DataFrame:
    work = add_rope_coil_features(df)
    need = [
        "symbol", "datetime", "open", "high", "low", "close", "rope_dir", "dist_to_rope_atr", "rope_slope_atr_5",
        "range_break_up", "range_break_up_strength", "bb_pos_01", "dsa_pivot_pos_01", "bb_width_percentile",
        "near_rope_ratio_5_loose", "near_rope_ratio_5_mid", "near_rope_streak_loose", "near_rope_streak_mid",
        "coil_loose", "coil_mid", "coil_strict", "trigger_slope_up", "trigger_above_rope", "trigger_depart_from_rope",
        "trigger_break_up", "trigger_break_up_strong", "trigger_dual_confirm", "not_too_high", "hit20_10", "hit20_15", "hit40_20",
    ] + [f"ret_{w}" for w in [5,10,20,40,60]] + [f"max_dd_{w}" for w in [5,10,20,40,60]] + [f"win_{w}" for w in [5,10,20,40,60]]
    need = [c for c in need if c in work.columns]
    sub = work[need].dropna(subset=[c for c in ["ret_20", "ret_40", "rope_dir", "dist_to_rope_atr"] if c in need]).copy()

    base = (sub["rope_dir"] == 1) & (sub["not_too_high"] == 1)
    if variant == "A":
        mask = base & ((sub["near_rope_ratio_5_loose"] >= 0.6) | (sub["near_rope_streak_loose"] >= 3))
    elif variant == "B":
        mask = base & ((sub["near_rope_ratio_5_mid"] >= 0.6) | (sub["near_rope_streak_mid"] >= 3)) & (sub["coil_loose"] == 1)
    elif variant == "C":
        mask = base & ((sub["near_rope_ratio_5_mid"] >= 0.6) | (sub["near_rope_streak_mid"] >= 3)) & (sub["coil_loose"] == 1) & (sub["trigger_break_up"] == 1) & (sub["trigger_above_rope"] == 1)
    elif variant == "D":
        rbs = sub["range_break_up_strength"].astype(float)
        thr = rbs.quantile(0.6) if rbs.notna().any() else np.nan
        mask = base & ((sub["near_rope_ratio_5_mid"] >= 0.6) | (sub["near_rope_streak_mid"] >= 3)) & (sub["trigger_break_up"] == 1) & (sub["trigger_above_rope"] == 1) & (sub["trigger_depart_from_rope"] == 1)
        if pd.notna(thr):
            mask &= (sub["range_break_up_strength"] >= thr)
    else:
        raise ValueError(f"未知 variant: {variant}")
    events = dedup_events(sub[mask].copy(), cooldown).reset_index(drop=True)
    events["variant"] = variant
    return events


def analyze_18_rope_coil_explosion(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("# 层次⑱: Rope贴线蓄势后爆发研究（breakout重构版）")
    print("=" * 70)
    work = add_rope_coil_features(df)

    variant_defs = {
        "A_loose_background": "A",
        "B_relaxed_main": "B",
        "C_first_release": "C",
        "D_breakout_strength_release": "D",
    }
    all_events = []
    rows = []
    for name, code in variant_defs.items():
        g = build_rope_coil_events(df, code)
        if len(g) == 0:
            continue
        g = g.copy()
        g["variant_name"] = name
        all_events.append(g)
        row = {"variant": name, "event_n": len(g)}
        row.update({k if k != "n" else "event_n": v for k, v in summarize_events_plus(g, [20, 40]).items() if k != "n"})
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["hit40_20", "rr_20"], ascending=False)
        print(summary.to_string(index=False))
        summary.to_csv(f"{OUT_DIR}/18_rope_coil_summary.csv", index=False)
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    if not events_df.empty:
        events_df.to_csv(f"{OUT_DIR}/18_rope_coil_events.csv", index=False)

    base_stage = build_rope_coil_events(df, "A")
    stage_rows = []
    if len(base_stage):
        stage_variants = {
            "S1_near_rope_only": pd.Series(True, index=base_stage.index),
            "S2_near_rope_plus_coil": (base_stage["coil_loose"] == 1),
            "S3_near_rope_coil_release": (base_stage["coil_loose"] == 1) & (base_stage["trigger_break_up"] == 1) & (base_stage["trigger_above_rope"] == 1),
            "S4_near_rope_strong_release": (base_stage["coil_loose"] == 1) & (base_stage["trigger_break_up_strong"] == 1) & (base_stage["trigger_depart_from_rope"] == 1),
        }
        for name, mask in stage_variants.items():
            g = base_stage[mask.fillna(False)]
            if len(g) < 20:
                continue
            row = {"stage": name, "event_n": len(g)}
            row.update({k if k != "n" else "event_n": v for k, v in summarize_events_plus(g, [20, 40]).items() if k != "n"})
            stage_rows.append(row)
    stage_df = pd.DataFrame(stage_rows)
    if not stage_df.empty:
        stage_df = stage_df.sort_values(["hit40_20", "rr_20"], ascending=False)
        print("\n[stage compare]")
        print(stage_df.to_string(index=False))
        stage_df.to_csv(f"{OUT_DIR}/18_rope_coil_stage_compare.csv", index=False)

    trig_rows = []
    if len(base_stage):
        trigger_variants = {
            "T0_no_trigger": (base_stage["trigger_slope_up"] == 0) & (base_stage["trigger_break_up"] == 0),
            "T1_slope_up": (base_stage["trigger_slope_up"] == 1),
            "T2_above_rope": (base_stage["trigger_above_rope"] == 1),
            "T3_break_up": (base_stage["trigger_break_up"] == 1),
            "T4_break_up_strong": (base_stage["trigger_break_up_strong"] == 1),
            "T5_break_up_depart": (base_stage["trigger_break_up"] == 1) & (base_stage["trigger_depart_from_rope"] == 1),
            "T6_dual_confirm": (base_stage["trigger_dual_confirm"] == 1),
        }
        for name, mask in trigger_variants.items():
            g = base_stage[mask.fillna(False)]
            if len(g) < 20:
                continue
            row = {"trigger": name, "event_n": len(g)}
            row.update({k if k != "n" else "event_n": v for k, v in summarize_events_plus(g, [20, 40]).items() if k != "n"})
            trig_rows.append(row)
    trigger_df = pd.DataFrame(trig_rows)
    if not trigger_df.empty:
        trigger_df = trigger_df.sort_values(["hit40_20", "rr_20"], ascending=False)
        print("\n[trigger compare]")
        print(trigger_df.to_string(index=False))
        trigger_df.to_csv(f"{OUT_DIR}/18_rope_coil_trigger_compare.csv", index=False)

    return {"summary": summary, "events": events_df, "stage": stage_df, "trigger": trigger_df}


# =========================
# C2 + rope enhancement analyses 19
# =========================
def build_c2_rope_enhanced_events(
    df: pd.DataFrame,
    dsa_thr: float = 0.35,
    bars_thr: int = 3,
    vwap_thr: Optional[float] = -1.0,
    enhancement: str = "base",
    cooldown: int = EVENT_DEDUP_BARS,
) -> pd.DataFrame:
    work = add_rope_coil_features(df)
    need = [
        "symbol", "datetime", "open", "high", "low", "close",
        "dsa_pivot_pos_01", "signed_vwap_dev_pct", "rope_dir", "bars_since_dir_change",
        "range_break_up", "range_break_up_strength", "dist_to_rope_atr",
        "trigger_break_up", "trigger_break_up_strong", "trigger_above_rope", "trigger_depart_from_rope",
        "near_rope_ratio_5_loose", "near_rope_streak_loose", "coil_loose", "not_too_high",
        "bb_pos_01", "bb_width_percentile",
        "hit20_10", "hit20_15", "hit40_20",
    ] + [f"ret_{w}" for w in [5, 10, 20, 40, 60]] + [f"max_dd_{w}" for w in [5, 10, 20, 40, 60]] + [f"win_{w}" for w in [5, 10, 20, 40, 60]]
    need = [c for c in need if c in work.columns]
    sub = work[need].dropna(subset=[c for c in ["ret_20", "ret_40", "dsa_pivot_pos_01", "rope_dir", "bars_since_dir_change"] if c in need]).copy()

    mask = (
        (sub["dsa_pivot_pos_01"] <= dsa_thr)
        & (sub["rope_dir"] == 1)
        & (sub["bars_since_dir_change"] <= bars_thr)
    )
    if vwap_thr is not None and "signed_vwap_dev_pct" in sub.columns:
        mask &= (sub["signed_vwap_dev_pct"] <= vwap_thr)

    enh_masks = {
        "base": pd.Series(True, index=sub.index),
        "break_up": (sub.get("trigger_break_up", 0) == 1),
        "break_up_strong": (sub.get("trigger_break_up_strong", 0) == 1),
        "depart_from_rope": (sub.get("trigger_depart_from_rope", 0) == 1),
        "break_strong_depart": (sub.get("trigger_break_up_strong", 0) == 1) & (sub.get("trigger_depart_from_rope", 0) == 1),
        "break_strong_depart_above": (sub.get("trigger_break_up_strong", 0) == 1) & (sub.get("trigger_depart_from_rope", 0) == 1) & (sub.get("trigger_above_rope", 0) == 1),
    }
    if enhancement not in enh_masks:
        raise ValueError(f"未知 enhancement: {enhancement}")
    mask &= enh_masks[enhancement].fillna(False)
    events = dedup_events(sub[mask].copy(), cooldown).reset_index(drop=True)
    events["enhancement"] = enhancement
    return events


def analyze_19_c2_rope_enhancement(
    df: pd.DataFrame,
    dsa_thr: float,
    bars_thr: int,
    vwap_thr: Optional[float],
) -> Dict[str, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("# 层次⑲: C2 + rope增强过滤对比")
    print("=" * 70)
    variants = [
        ("C2_base", "base"),
        ("C2_plus_break_up", "break_up"),
        ("C2_plus_break_up_strong", "break_up_strong"),
        ("C2_plus_depart_from_rope", "depart_from_rope"),
        ("C2_plus_break_strong_depart", "break_strong_depart"),
        ("C2_plus_break_strong_depart_above", "break_strong_depart_above"),
    ]
    rows = []
    all_events = []
    for name, code in variants:
        g = build_c2_rope_enhanced_events(
            df,
            dsa_thr=dsa_thr,
            bars_thr=bars_thr,
            vwap_thr=vwap_thr,
            enhancement=code,
            cooldown=EVENT_DEDUP_BARS,
        )
        if len(g) == 0:
            continue
        gg = g.copy()
        gg["variant_name"] = name
        all_events.append(gg)
        row = {"variant": name, "event_n": len(g)}
        stats = summarize_events_plus(g, [20, 40])
        for k, v in stats.items():
            if k == "n":
                continue
            row[k] = v
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["rr_20", "hit20_15", "hit40_20"], ascending=False)
        print(summary.to_string(index=False))
        summary.to_csv(f"{OUT_DIR}/19_c2_rope_enhancement_summary.csv", index=False)
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    if not events_df.empty:
        events_df.to_csv(f"{OUT_DIR}/19_c2_rope_enhancement_events.csv", index=False)

    base = build_c2_rope_enhanced_events(
        df,
        dsa_thr=dsa_thr,
        bars_thr=bars_thr,
        vwap_thr=vwap_thr,
        enhancement="base",
        cooldown=EVENT_DEDUP_BARS,
    )
    score_df = pd.DataFrame()
    if len(base):
        tmp = base.copy()
        tmp["enh_score"] = (
            tmp.get("trigger_break_up", 0).astype(int)
            + tmp.get("trigger_break_up_strong", 0).astype(int)
            + tmp.get("trigger_depart_from_rope", 0).astype(int)
            + tmp.get("trigger_above_rope", 0).astype(int)
        )
        score_rows = []
        buckets = {
            "score_0": tmp["enh_score"] == 0,
            "score_1": tmp["enh_score"] == 1,
            "score_2": tmp["enh_score"] == 2,
            "score_3plus": tmp["enh_score"] >= 3,
        }
        for name, m in buckets.items():
            g = tmp[m.fillna(False)]
            if len(g) < 20:
                continue
            row = {"bucket": name, "event_n": len(g)}
            stats = summarize_events_plus(g, [20, 40])
            for k, v in stats.items():
                if k == "n":
                    continue
                row[k] = v
            score_rows.append(row)
        score_df = pd.DataFrame(score_rows)
        if not score_df.empty:
            score_df = score_df.sort_values(["rr_20", "hit20_15", "hit40_20"], ascending=False)
            print("\n[c2 enhancement score compare]")
            print(score_df.to_string(index=False))
            score_df.to_csv(f"{OUT_DIR}/19_c2_rope_enhancement_score_compare.csv", index=False)

    return {"summary": summary, "events": events_df, "score": score_df}


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="买入点收益风险比探索（收敛验证版）")
    parser.add_argument("--n-stocks", type=int, default=100, help="随机抽取股票数")
    parser.add_argument("--bars", type=int, default=800, help="每只股票K线数")
    parser.add_argument("--freq", default="d", help="频率 d/w/mo/60m等")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--c2-dsa-thr", type=float, default=0.35)
    parser.add_argument("--c2-bars-thr", type=int, default=3)
    parser.add_argument("--c2-vwap-thr", type=float, default=-1.0)
    parser.add_argument("--c2-hold-bars", type=int, default=40)
    parser.add_argument("--max-signals-per-day", type=int, default=5)
    parser.add_argument("--signal-rank-by", type=str, default="signed_vwap_dev_pct")
    parser.add_argument("--market-index-code", type=str, default="000905.SH")
    parser.add_argument("--tushare-token", type=str, default="")
    args = parser.parse_args()

    stocks = get_stock_pool(args.n_stocks, args.seed)
    print(f"股票池抽取: {len(stocks)}只 (seed={args.seed})")

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
    print("\n基线统计:")
    for w in [5, 10, 20, 40, 60]:
        ret = df[f"ret_{w}"].mean()
        dd = df[f"max_dd_{w}"].mean()
        wr = df[f"win_{w}"].mean()
        print(f"  ret_{w}: 均值={ret:+.5f}, max_dd均值={dd:.5f}, 胜率={wr:.4f}, rr={rr_from_means(ret, dd):.3f}")

    analyze_01_single_factor(df)
    analyze_02_scenarios(df)
    analyze_03_interaction(df)
    analyze_04_rule_search(df)
    analyze_05_candidate_events(df)
    analyze_06_c2_param_scan(df)
    analyze_07_c2_exit_compare(df, args.c2_dsa_thr, args.c2_bars_thr, args.c2_vwap_thr)
    analyze_08_c2_time_slices(df, args.c2_dsa_thr, args.c2_bars_thr, args.c2_vwap_thr)

    base_events = build_c2_events_base(df, args.c2_dsa_thr, args.c2_bars_thr, args.c2_vwap_thr, EVENT_DEDUP_BARS, args.c2_hold_bars)
    stock_counts = sorted(set([100, 300, 500, 1000, args.n_stocks]))
    analyze_09_c2_fixed_validation(df, base_events, stock_counts)
    analyze_10_c2_local_grid(df)
    analyze_11_c2_t3_regime_breakdown(base_events)
    analyze_12_c2_execution_filter(base_events, args.max_signals_per_day, args.signal_rank_by)
    analyze_13_c2_weekly_dsa_filter(base_events)
    analyze_14_c2_weekly_filter_compare(base_events)
    analyze_15_c2_execution_rank_compare(base_events, args.max_signals_per_day)

    # market regime from tushare; fail gracefully
    try:
        start_date = pd.to_datetime(df["datetime"]).min().strftime("%Y-%m-%d")
        end_date = pd.to_datetime(df["datetime"]).max().strftime("%Y-%m-%d")
        market_df = load_market_index_from_tushare(args.market_index_code, start_date, end_date, args.tushare_token)
        market_feats = compute_market_regime_features(market_df)
        market_events = merge_market_features_to_events(base_events, market_feats)
        analyze_16_c2_market_regime_compare(base_events, market_events)
    except Exception as e:
        print(f"\n[WARN] 层次⑯ 跳过: {e}")

    analyze_17_c2_local_state_compare(base_events)
    analyze_18_rope_coil_explosion(df)
    analyze_19_c2_rope_enhancement(df, args.c2_dsa_thr, args.c2_bars_thr, args.c2_vwap_thr)

    print(f"\n{'='*70}")
    print(f"# 全部完成! 结果保存在 {OUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
