# -*- coding: utf-8 -*-
"""
基于 GBDT 的买点归因研究脚本（沿用 buy_point_explorer.py 框架）

Purpose
- 沿用 buy_point_explorer.py 的数据加载、因子计算、forward return 标签计算框架
- 聚焦 C_long_down_repair 邻域样本，做 GBDT 归因与参数区间建议
- 分离 trade-safe 与 research 两套特征，避免 hindsight 因子污染交易结论
- 输出时间滚动验证结果、特征重要性、分桶画像、参数提示表

Research Scope
- 主样本池：C_long_down_repair 的“宽口径邻域样本”
- 模型：
    * GradientBoostingRegressor 预测 ret_20
    * GradientBoostingClassifier 预测 good20
- 结果用途：
    * 归因（哪些因子真正有边际解释力）
    * 提示参数区间（不是直接生成最终交易规则）

How to Run
    python gbdt_buy_point_explorer.py --n-stocks 100 --bars 800 --freq d

    python gbdt_buy_point_explorer.py --n-stocks 500 --bars 1000 --freq d \
        --sample-mode c_neighbor --neighbor-dsa-max 0.45 --neighbor-prev-down-min 8

Examples
    python gbdt_buy_point_explorer.py --analysis-mode full
    python gbdt_buy_point_explorer.py --analysis-mode model_only --n-stocks 300 --bars 1000

Outputs
- 00_dataset_summary.csv           数据集摘要
- 01_feature_inventory.csv         因子覆盖率与特征分层
- 02_candidate_events.csv          C 邻域事件样本表
- 03_trade_reg_importance.csv      trade-safe 回归重要性
- 03b_trade_clf_importance.csv     trade-safe 分类重要性
- 04_research_reg_importance.csv   research 回归重要性
- 04b_research_clf_importance.csv  research 分类重要性
- 05_fold_metrics.csv              时间滚动验证结果
- 06_trade_bucket_profiles.csv     trade-safe 关键变量分桶画像
- 06b_research_bucket_profiles.csv research 关键变量分桶画像
- 07_parameter_hints.csv           参数提示表（候选阈值区间）

Side Effects
- 读取数据库 stock_k_data / stock_pools
- 在 gbdt_buy_point_explorer_output/ 目录写入 CSV 文件
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
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
    from merged_dsa_atr_rope_bb_factors_with_confirmed_run_bars import (
        DSAConfig,
        RopeConfig,
        compute_atr_rope,
        compute_bollinger,
        compute_dsa,
    )

warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = "gbdt_buy_point_explorer_output"
os.makedirs(OUT_DIR, exist_ok=True)

RET_WINDOWS = [5, 10, 20, 40, 60]
EVENT_DEDUP_BARS = 20
RANDOM_STATE = 42


# =========================
# Common helpers
# =========================
def rr_from_ret_dd(ret: float, dd: float) -> float:
    if pd.isna(ret) or pd.isna(dd) or dd == 0:
        return np.nan
    return float(ret / abs(dd))


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


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
]

RESEARCH_FEATURES = [
    "dsa_confirmed_pivot_pos_01", "dsa_signed_vwap_dev_pct", "dsa_trend_aligned_vwap_dev_pct",
    "lh_hh_low_pos", "dsa_bull_vwap_dev_pct", "dsa_bear_vwap_dev_pct",
    "prev_confirmed_up_bars", "prev_confirmed_down_bars", "last_confirmed_run_bars", "current_run_bars",
    "w_DSA_DIR", "w_dsa_confirmed_pivot_pos_01", "w_dsa_signed_vwap_dev_pct", "w_prev_confirmed_up_bars", "w_prev_confirmed_down_bars",
]

PRIMARY_HINT_FEATURES = [
    "dsa_trade_pos_01", "dist_to_rope_atr", "bb_width_percentile", "bars_since_dir_change",
    "prev_confirmed_down_bars", "current_run_bars", "bb_pos_01", "rope_slope_atr_5",
]


# =========================
# Candidate pool / labels
# =========================
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


# =========================
# Model helpers
# =========================
def prepare_feature_matrix(events: pd.DataFrame, features: Sequence[str], fill_source: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
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
    nunique = x.nunique(dropna=False)
    cols = [c for c in cols if nunique.get(c, 0) > 1]
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


def collect_importance_rows(
    model_name: str,
    mode: str,
    fold_name: str,
    features: List[str],
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> List[Dict[str, object]]:
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


def run_single_model(
    events: pd.DataFrame,
    features: Sequence[str],
    target_col: str,
    model_type: str,
    feature_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folds = make_time_folds(events)
    metrics_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []
    for fold_name, train_idx, test_idx in folds:
        train_df = events.loc[train_idx].copy()
        test_df = events.loc[test_idx].copy()
        x_train, used_features = prepare_feature_matrix(train_df, features, fill_source=train_df)
        x_test, _ = prepare_feature_matrix(test_df, used_features, fill_source=train_df)
        if len(used_features) < 3:
            continue
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        if model_type == "reg":
            model = GradientBoostingRegressor(
                random_state=RANDOM_STATE,
                learning_rate=0.05,
                n_estimators=250,
                max_depth=3,
                subsample=0.8,
                min_samples_leaf=20,
            )
            model.fit(x_train, safe_numeric(y_train))
            y_pred = model.predict(x_test)
            score = regression_score(y_test, y_pred)
        else:
            if pd.Series(y_train).nunique() < 2 or pd.Series(y_test).nunique() < 1:
                continue
            model = GradientBoostingClassifier(
                random_state=RANDOM_STATE,
                learning_rate=0.05,
                n_estimators=250,
                max_depth=3,
                subsample=0.8,
                min_samples_leaf=20,
            )
            model.fit(x_train, pd.Series(y_train).astype(int))
            y_pred = model.predict_proba(x_test)[:, 1]
            score = classification_score(y_test, y_pred)
        metrics_rows.append({
            "feature_mode": feature_mode,
            "target": target_col,
            "model_type": model_type,
            "fold": fold_name,
            "train_n": len(train_df),
            "test_n": len(test_df),
            **score,
        })
        importance_rows.extend(
            collect_importance_rows(
                model_name=f"{feature_mode}_{target_col}_{model_type}",
                mode=feature_mode,
                fold_name=fold_name,
                features=used_features,
                model=model,
                x_test=x_test,
                y_test=pd.Series(y_test).astype(int) if model_type == "clf" else safe_numeric(y_test),
            )
        )
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


# =========================
# Profiles / hints
# =========================
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
        grp = sub.groupby("bucket", observed=True).agg(
            n=(feat, "count"),
            mean_value=(feat, "mean"),
            ret20=("ret_20", "mean"),
            dd20=("max_dd_20", "mean"),
            wr20=("win_20", "mean"),
            rr20=("rr_20", "mean"),
        ).reset_index()
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


def build_parameter_hints(
    events: pd.DataFrame,
    trade_imp: pd.DataFrame,
    research_imp: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
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
        feature_list = (
            merged_top.groupby("feature", as_index=False)["rank_score"].mean()
            .sort_values("rank_score", ascending=False)["feature"]
            .tolist()
        )
        feature_list = [f for f in feature_list if f in events.columns]
    feature_list = list(dict.fromkeys(feature_list + [f for f in PRIMARY_HINT_FEATURES if f in events.columns]))[:max(top_n, len(PRIMARY_HINT_FEATURES))]

    for feat in feature_list:
        s = safe_numeric(events[feat]).dropna()
        if len(s) < 40 or s.nunique() < 4:
            continue
        q20 = float(s.quantile(0.2))
        q35 = float(s.quantile(0.35))
        q50 = float(s.quantile(0.5))
        q65 = float(s.quantile(0.65))
        q80 = float(s.quantile(0.8))
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
        rows.append({
            "feature": feat,
            "best_bucket": best_bucket,
            "direction": direction,
            "q20": round(q20, 6),
            "q35": round(q35, 6),
            "q50": round(q50, 6),
            "q65": round(q65, 6),
            "q80": round(q80, 6),
            "hint_rule": hint_rule,
            "sample_n": int(len(s)),
        })
    rdf = pd.DataFrame(rows)
    if not rdf.empty:
        rdf.to_csv(f"{OUT_DIR}/07_parameter_hints.csv", index=False)
    return rdf


# =========================
# Reports
# =========================
def analyze_00_dataset_summary(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({
        "section": "global",
        "sample_n": int(len(df)),
        "stock_n": int(df["symbol"].nunique()),
        "ret20_mean": round(float(df["ret_20"].mean()), 5),
        "dd20_mean": round(float(df["max_dd_20"].mean()), 5),
        "wr20_mean": round(float(df["win_20"].mean()), 4),
        "rr20_mean": round(rr_from_ret_dd(df["ret_20"].mean(), df["max_dd_20"].mean()), 3),
    })
    if not events.empty:
        rows.append({
            "section": "candidate_pool",
            "sample_n": int(len(events)),
            "stock_n": int(events["symbol"].nunique()),
            "ret20_mean": round(float(events["ret_20"].mean()), 5),
            "dd20_mean": round(float(events["max_dd_20"].mean()), 5),
            "wr20_mean": round(float(events["win_20"].mean()), 4),
            "rr20_mean": round(rr_from_ret_dd(events["ret_20"].mean(), events["max_dd_20"].mean()), 3),
        })
    rdf = pd.DataFrame(rows)
    rdf.to_csv(f"{OUT_DIR}/00_dataset_summary.csv", index=False)
    return rdf


def analyze_01_feature_inventory(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in list(dict.fromkeys(TRADE_FEATURES + RESEARCH_FEATURES)):
        present = feat in events.columns
        s = events[feat] if present else pd.Series(dtype=float)
        rows.append({
            "feature": feat,
            "feature_mode": "trade" if feat in TRADE_FEATURES else "research",
            "present": int(present),
            "coverage": round(float(s.notna().mean()), 4) if present else 0.0,
            "nunique": int(s.nunique(dropna=True)) if present else 0,
            "mean": round(float(safe_numeric(s).mean()), 6) if present else np.nan,
            "std": round(float(safe_numeric(s).std()), 6) if present else np.nan,
        })
    rdf = pd.DataFrame(rows).sort_values(["feature_mode", "coverage", "feature"], ascending=[True, False, True])
    rdf.to_csv(f"{OUT_DIR}/01_feature_inventory.csv", index=False)
    return rdf


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="基于 GBDT 的买点归因研究脚本")
    parser.add_argument("--n-stocks", type=int, default=100, help="随机抽取股票数")
    parser.add_argument("--bars", type=int, default=800, help="每只股票K线数")
    parser.add_argument("--freq", default="d", help="频率 d/w/mo/60m等")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--analysis-mode", type=str, default="full", choices=["full", "prepare_only", "model_only"], help="full=准备+建模；prepare_only=只准备数据；model_only=准备后直接建模")
    parser.add_argument("--sample-mode", type=str, default="c_neighbor", choices=["c_neighbor"], help="当前仅支持 C_long_down_repair 邻域样本")
    parser.add_argument("--neighbor-dsa-max", type=float, default=0.45, help="C 邻域样本 dsa_trade_pos_01 上限")
    parser.add_argument("--neighbor-prev-down-min", type=float, default=8.0, help="C 邻域样本 prev_confirmed_down_bars 下限")
    parser.add_argument("--neighbor-current-run-max", type=float, default=20.0, help="C 邻域样本 current_run_bars 上限")
    parser.add_argument("--neighbor-bars-since-max", type=float, default=12.0, help="C 邻域样本 bars_since_dir_change 上限")
    args = parser.parse_args()

    stocks = get_stock_pool(args.n_stocks, args.seed)
    print(f"股票池抽取: {len(stocks)}只 (seed={args.seed})")
    print("研究目标: 用 GBDT 做归因与参数提示，不直接替代最终规则裁决")

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
    neighbor_spec = NeighborSpec(
        dsa_max=args.neighbor_dsa_max,
        prev_down_min=args.neighbor_prev_down_min,
        current_run_max=args.neighbor_current_run_max,
        bars_since_max=args.neighbor_bars_since_max,
    )
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

    # bucket profiles use the strongest features from each side
    trade_top = trade_reg_imp_agg.groupby("feature", as_index=False)["rank_score"].mean().sort_values("rank_score", ascending=False) if not trade_reg_imp_agg.empty else pd.DataFrame()
    research_top = research_reg_imp_agg.groupby("feature", as_index=False)["rank_score"].mean().sort_values("rank_score", ascending=False) if not research_reg_imp_agg.empty else pd.DataFrame()
    trade_feats_for_bucket = trade_top["feature"].head(8).tolist() if not trade_top.empty else [f for f in PRIMARY_HINT_FEATURES if f in TRADE_FEATURES]
    research_feats_for_bucket = research_top["feature"].head(8).tolist() if not research_top.empty else [f for f in PRIMARY_HINT_FEATURES if f in RESEARCH_FEATURES]
    build_bucket_profiles(events, trade_feats_for_bucket, f"{OUT_DIR}/06_trade_bucket_profiles.csv")
    build_bucket_profiles(events, research_feats_for_bucket, f"{OUT_DIR}/06b_research_bucket_profiles.csv")

    trade_hint_source = pd.concat([trade_reg_imp_agg, trade_clf_imp_agg], ignore_index=True)
    research_hint_source = pd.concat([research_reg_imp_agg, research_clf_imp_agg], ignore_index=True)
    build_parameter_hints(events, trade_hint_source, research_hint_source)

    print("\n=== 关键结果预览 ===")
    if not trade_reg_imp_agg.empty:
        print("\n[trade-safe 回归重要性 Top10]")
        print(trade_reg_imp_agg[["feature", "rank_score", "gain_importance", "perm_importance", "fold_count"]].head(10).to_string(index=False))
    if not trade_clf_imp_agg.empty:
        print("\n[trade-safe 分类重要性 Top10]")
        print(trade_clf_imp_agg[["feature", "rank_score", "gain_importance", "perm_importance", "fold_count"]].head(10).to_string(index=False))
    if not research_reg_imp_agg.empty:
        print("\n[research 回归重要性 Top10]")
        print(research_reg_imp_agg[["feature", "rank_score", "gain_importance", "perm_importance", "fold_count"]].head(10).to_string(index=False))

    print(f"\n{'=' * 70}")
    print(f"# 全部完成! 结果保存在 {OUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
