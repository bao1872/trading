# -*- coding: utf-8 -*-
"""
Purpose:
    股东投资质量画像模型 - 核心计算模块（SSOT）
    从 stock_top10_holders_tushare 出发，独立构建6维度画像评分

Inputs:
    - 数据库表: stock_top10_holders_tushare
    - 数据库表: stock_k_data (行情)
    - 数据库表: stock_pools (行业信息)
    - Tushare API: 指数行情

Outputs:
    - 数据库表: stock_holder_quality_portrait

How to Run:
    python financial_factors/holder_quality_portrait.py --mode single --ts_code 600519.SH --dry_run
    python financial_factors/holder_quality_portrait.py --mode batch --limit 50 --dry_run
    python financial_factors/holder_quality_portrait.py --mode batch --limit 100

Examples:
    python financial_factors/holder_quality_portrait.py --mode single --ts_code 000001.SZ --dry_run
    python financial_factors/holder_quality_portrait.py --mode batch --limit 20

Side Effects:
    - 写入 stock_holder_quality_portrait 表（非 dry_run 时）
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import bulk_upsert, get_engine, get_session, query_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

INPUT_TABLE = "stock_top10_holders_tushare"
PORTRAIT_TABLE = "stock_holder_quality_portrait"
PORTRAIT_UNIQUE_KEYS = ["holder_name_std"]
EPS = 1e-12
DEFAULT_BENCH = "000300.SH"
DELTA_THRESHOLD_ABS = 0.10
DELTA_THRESHOLD_REL = 0.05
PROFILE_SCOPE_FULL = "full_sample"
PROFILE_SCOPE_ASOF = "asof"


def normalize_holder_name(name: object) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    s = str(name).strip()
    for old, new in {
        "（": "(", "）": ")", "股份有限公司": "股份", "有限合伙企业": "有限合伙",
        "－": "-", "—": "-", "\u3000": "", " ": "", "\u201c": "", "\u201d": "",
        "\u2018": "", "\u2019": "",
    }.items():
        s = s.replace(old, new)
    return s.upper()


def normalize_ts_code(ts_code: str) -> str:
    ts_code = str(ts_code).strip().upper()
    if ts_code.endswith((".SZ", ".SH", ".BJ")):
        return ts_code
    if ts_code.startswith("6"):
        return f"{ts_code}.SH"
    if ts_code.startswith(("8", "4")):
        return f"{ts_code}.BJ"
    return f"{ts_code}.SZ"


def map_holder_prior_score(holder_type: object) -> float:
    v = "" if holder_type is None else str(holder_type).upper()
    if any(k in v for k in ["社保", "全国社会保障", "社保基金"]):
        return 80.0
    if any(k in v for k in ["国有", "国资", "汇金", "证金"]):
        return 80.0
    if any(k in v for k in ["保险", "险资"]):
        return 78.0
    if "QFII" in v or "境外" in v:
        return 76.0
    if any(k in v for k in ["基金", "公募"]):
        return 68.0
    if any(k in v for k in ["券商", "证券", "资管"]):
        return 62.0
    if any(k in v for k in ["信托"]):
        return 55.0
    if any(k in v for k in ["个人", "自然人"]):
        return 45.0
    return 50.0


def safe_float(v: object) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def robust_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std(ddof=0)
    if pd.isna(std) or std < EPS:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mean) / std


def rank01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return s.rank(pct=True)


def clip_score(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def parse_date_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str[:8], format="%Y%m%d", errors="coerce")


def ensure_portrait_table() -> None:
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {PORTRAIT_TABLE} (
        holder_name_std TEXT PRIMARY KEY,
        holder_type TEXT,
        sample_total INTEGER,
        sample_stocks INTEGER,
        sample_periods INTEGER,
        sample_entry INTEGER,
        sample_add INTEGER,
        sample_reduce INTEGER,
        sample_exit INTEGER,
        entry_excess_ret_60 REAL,
        add_excess_ret_60 REAL,
        entry_win_rate_60 REAL,
        add_win_rate_60 REAL,
        information_ratio REAL,
        long_horizon_alpha REAL,
        picking_score REAL,
        avg_tenure REAL,
        stock_diversity REAL,
        turnover_rate REAL,
        contrarian_ratio REAL,
        momentum_ratio REAL,
        style_score REAL,
        industry_coverage INTEGER,
        best_industry TEXT,
        best_industry_excess REAL,
        industry_concentration REAL,
        industry_alpha_spread REAL,
        expertise_score REAL,
        lowpos_fit REAL,
        smallcap_fit REAL,
        growth_fit REAL,
        bear_fit REAL,
        adapt_score REAL,
        avg_mdd_60 REAL,
        avg_vol_20 REAL,
        tail_loss REAL,
        stop_loss_ratio REAL,
        risk_score REAL,
        avg_hold_ratio REAL,
        total_stock_count INTEGER,
        hold_ratio_trend REAL,
        scale_score REAL,
        composite_raw REAL,
        prior_score REAL,
        shrinkage_weight REAL,
        composite_score REAL,
        quality_grade TEXT,
        style_label TEXT,
        period_label TEXT,
        industry_label TEXT,
        ability_label TEXT,
        profile_scope TEXT,
        profile_asof_date TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """
    try:
        with get_engine().begin() as conn:
            conn.execute(text(create_sql))
        logger.info("Portrait table ensured")
    except Exception as e:
        logger.error("Failed to create portrait table: %s", e)
        raise


def load_top10_data(ts_codes: Optional[Sequence[str]] = None, lookback_years: int = 5) -> pd.DataFrame:
    with get_session() as session:
        df = query_df(session, INPUT_TABLE)
    if df.empty:
        return df
    df = df.copy()
    df["ts_code"] = df["ts_code"].astype(str).map(normalize_ts_code)
    if ts_codes:
        ts_code_set = {normalize_ts_code(x) for x in ts_codes}
        df = df[df["ts_code"].isin(ts_code_set)].copy()
    df["report_date"] = parse_date_column(df["report_date"])
    df["ann_date"] = parse_date_column(df["ann_date"])
    df["holder_name_std"] = df["holder_name"].map(normalize_holder_name)
    for col in ["hold_amount", "hold_ratio", "hold_float_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "holder_rank" in df.columns:
        df["holder_rank"] = pd.to_numeric(df["holder_rank"], errors="coerce").astype("Int64")
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=lookback_years)
    df = df[df["report_date"] >= cutoff].copy()
    df = df.sort_values(["ts_code", "report_date", "holder_rank", "holder_name_std"]).reset_index(drop=True)
    return df


def load_stock_pool(limit: Optional[int] = None) -> List[Tuple[str, Optional[str]]]:
    with get_session() as session:
        df = query_df(session, "stock_pools", columns=["ts_code", "name"])
    if df.empty:
        return []
    df = df.drop_duplicates(subset=["ts_code"])
    result = list(zip(df["ts_code"].astype(str).map(normalize_ts_code), df["name"].tolist()))
    if limit:
        result = result[:limit]
    return result


def load_industry_info() -> pd.DataFrame:
    cols = ["ts_code", "name", "market_cap", "industry_l2", "industry_l3"]
    try:
        with get_session() as session:
            df = query_df(session, "stock_pools", columns=cols)
    except Exception:
        return pd.DataFrame(columns=cols)
    df = df[[c for c in cols if c in df.columns]].copy()
    if df.empty:
        return df
    df["ts_code"] = df["ts_code"].astype(str).map(normalize_ts_code)
    for c in ["market_cap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["industry_l2", "industry_l3"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": None})
    df = df.drop_duplicates(subset=["ts_code"], keep="last")
    return df


def load_market_data_from_db(ts_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    sql = """
        SELECT bar_time as datetime, open, high, low, close, volume
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = 'd'
          AND bar_time >= :start_date AND bar_time <= :end_date
        ORDER BY bar_time
    """
    with get_session() as session:
        result = session.execute(text(sql), {
            "ts_code": ts_code,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        })
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    if df.empty:
        return pd.DataFrame()
    df["datetime"] = pd.to_datetime(df["datetime"])
    if hasattr(df["datetime"].dtype, "tz") and df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    df = df.set_index("datetime").sort_index()
    return df


def load_bench_data(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    from tushare_data.fetcher import fetch_market_data
    start = (start_date - pd.Timedelta(days=10)).strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")
    df = fetch_market_data(DEFAULT_BENCH, start, end, fqt=0)
    if df.empty:
        return pd.DataFrame()
    if "date" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"date": "datetime"})
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
    return df


def infer_effective_date(ann_date: pd.Timestamp, trading_index: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    if pd.isna(ann_date):
        return None
    ann_date = pd.Timestamp(ann_date)
    if ann_date.tzinfo is not None:
        ann_date = ann_date.tz_localize(None)
    idx = trading_index.tz_localize(None) if hasattr(trading_index, "tz") and trading_index.tz else trading_index
    pos = idx.searchsorted(ann_date)
    if pos >= len(idx):
        return None
    if idx[pos] == ann_date:
        pos += 1
    if pos >= len(idx):
        return None
    return pd.Timestamp(idx[pos])


def calc_future_return(prices: pd.Series, start_idx: int, horizon: int) -> Optional[float]:
    end_idx = start_idx + horizon
    if start_idx < 0 or end_idx >= len(prices):
        return None
    p0, p1 = prices.iloc[start_idx], prices.iloc[end_idx]
    if pd.isna(p0) or pd.isna(p1) or abs(p0) < EPS:
        return None
    return float(p1 / p0 - 1.0)


def calc_max_drawdown_forward(prices: pd.Series, start_idx: int, horizon: int) -> Optional[float]:
    end_idx = start_idx + horizon
    if start_idx < 0 or end_idx >= len(prices):
        return None
    base = prices.iloc[start_idx]
    window = prices.iloc[start_idx:(end_idx + 1)]
    if pd.isna(base) or abs(base) < EPS or window.empty:
        return None
    return float((window / base - 1.0).min())


def calc_backward_return(prices: pd.Series, idx: int, lookback: int) -> Optional[float]:
    prev_idx = idx - lookback
    if prev_idx < 0 or idx >= len(prices):
        return None
    p0, p1 = prices.iloc[prev_idx], prices.iloc[idx]
    if pd.isna(p0) or pd.isna(p1) or abs(p0) < EPS:
        return None
    return float(p1 / p0 - 1.0)


def build_holder_tenure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ts_code", "holder_name_std", "report_date"]).copy()
    if df.empty:
        df["tenure"] = []
        return df
    df["tenure"] = 1
    parts = []
    for _, g in df.groupby(["ts_code", "holder_name_std"], sort=False):
        g = g.sort_values("report_date").copy()
        month_diff = g["report_date"].diff().dt.days.div(30.0).round().fillna(3)
        reset_flag = ~month_diff.isin([3, 6])
        g["tenure"] = reset_flag.cumsum()
        g["tenure"] = g.groupby("tenure").cumcount() + 1
        parts.append(g)
    return pd.concat(parts, ignore_index=False).sort_values(
        ["ts_code", "report_date", "holder_rank", "holder_name_std"]
    ).reset_index(drop=True)


def _parse_hold_change_signal(v: object) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, str):
        s = v.strip().upper()
        if not s:
            return None
        if any(k in s for k in ["新进", "新增", "增持"]):
            return 1.0
        if any(k in s for k in ["退出", "减持", "减少"]):
            return -1.0
        if any(k in s for k in ["不变", "持平"]):
            return 0.0
        try:
            return float(s)
        except Exception:
            return None
    try:
        return float(v)
    except Exception:
        return None


def _classify_overlap_action(cur_row: pd.Series, prev_row: pd.Series) -> Tuple[str, Optional[float], Optional[float]]:
    cur_ratio = safe_float(cur_row.get("hold_float_ratio"))
    prev_ratio = safe_float(prev_row.get("hold_float_ratio"))
    cur_amt = safe_float(cur_row.get("hold_amount"))
    prev_amt = safe_float(prev_row.get("hold_amount"))
    delta_ratio = None if cur_ratio is None or prev_ratio is None else cur_ratio - prev_ratio
    delta_amt = None if cur_amt is None or prev_amt is None else cur_amt - prev_amt
    rel_change = None
    if delta_ratio is not None and prev_ratio is not None and abs(prev_ratio) > EPS:
        rel_change = abs(delta_ratio) / abs(prev_ratio)

    hold_change_signal = _parse_hold_change_signal(cur_row.get("hold_change"))
    if hold_change_signal is not None:
        if hold_change_signal > 0:
            return "add", delta_ratio, delta_amt
        if hold_change_signal < 0:
            return "reduce", delta_ratio, delta_amt
        return "hold", delta_ratio, delta_amt

    if delta_ratio is None:
        return "hold", delta_ratio, delta_amt
    if (delta_ratio > DELTA_THRESHOLD_ABS) or (rel_change is not None and rel_change > DELTA_THRESHOLD_REL and delta_ratio > 0):
        return "add", delta_ratio, delta_amt
    if (delta_ratio < -DELTA_THRESHOLD_ABS) or (rel_change is not None and rel_change > DELTA_THRESHOLD_REL and delta_ratio < 0):
        return "reduce", delta_ratio, delta_amt
    return "hold", delta_ratio, delta_amt


def build_event_metrics_cache(df: pd.DataFrame, stock_df_map: Dict[str, pd.DataFrame],
                              bench_df: pd.DataFrame) -> Dict[Tuple[str, pd.Timestamp], Dict]:
    cache: Dict[Tuple[str, pd.Timestamp], Dict] = {}
    groups = list(df.groupby(["ts_code", "report_date"]))
    total = len(groups)
    logger.info("构建事件市场指标缓存: %s 个事件...", total)

    for idx, ((ts_code, report_date), snap) in enumerate(groups, 1):
        if idx % 200 == 0 or idx == total:
            logger.info("  缓存进度: %s/%s", idx, total)
        stock_df = stock_df_map.get(ts_code)
        if stock_df is None or stock_df.empty:
            continue
        ann_date = snap["ann_date"].dropna().min()
        if pd.isna(ann_date):
            continue
        effective_date = infer_effective_date(pd.Timestamp(ann_date), stock_df.index)
        if effective_date is None:
            continue

        close = stock_df["close"].dropna()
        if close.empty:
            continue
        trading_index = close.index
        if hasattr(trading_index, "tz") and trading_index.tz:
            trading_index = trading_index.tz_localize(None)
        eff = pd.Timestamp(effective_date)
        if eff.tzinfo:
            eff = eff.tz_localize(None)
        pos = trading_index.searchsorted(eff)
        if pos >= len(trading_index):
            pos = len(trading_index) - 1
        if pos < 0:
            pos = 0

        future_ret_20 = calc_future_return(close, pos, 20)
        future_ret_60 = calc_future_return(close, pos, 60)
        future_ret_120 = calc_future_return(close, pos, 120)
        future_mdd_60 = calc_max_drawdown_forward(close, pos, 60)
        ret_120 = calc_backward_return(close, pos, 120)
        mom_20 = calc_backward_return(close, pos, 20)
        mom_60 = calc_backward_return(close, pos, 60)

        bench_close = bench_df["close"].dropna() if not bench_df.empty and "close" in bench_df.columns else pd.Series(dtype=float)
        bench_ret_20 = bench_ret_60 = bench_ret_120 = None
        if len(bench_close) > 0:
            if hasattr(bench_close.index, "tz") and bench_close.index.tz:
                bench_close.index = bench_close.index.tz_localize(None)
            bench_pos = bench_close.index.searchsorted(eff)
            if bench_pos >= len(bench_close):
                bench_pos = len(bench_close) - 1
            bench_ret_20 = calc_future_return(bench_close, bench_pos, 20)
            bench_ret_60 = calc_future_return(bench_close, bench_pos, 60)
            bench_ret_120 = calc_future_return(bench_close, bench_pos, 120)

        future_excess_ret_20 = None if future_ret_20 is None or bench_ret_20 is None else future_ret_20 - bench_ret_20
        future_excess_ret_60 = None if future_ret_60 is None or bench_ret_60 is None else future_ret_60 - bench_ret_60
        future_excess_ret_120 = None if future_ret_120 is None or bench_ret_120 is None else future_ret_120 - bench_ret_120

        vol_20 = None
        if pos >= 19:
            rets20 = close.pct_change().iloc[pos - 19:pos + 1]
            vol_20 = float(rets20.std(ddof=0)) if rets20.notna().sum() >= 5 else None

        drawdown_from_high_120 = None
        start_idx = max(0, pos - 119)
        window = close.iloc[start_idx:pos + 1]
        if not window.empty:
            high, cur = window.max(), close.iloc[pos]
            if pd.notna(high) and pd.notna(cur) and abs(high) > EPS:
                drawdown_from_high_120 = float(cur / high - 1.0)

        turnover_value_mean_20 = None
        if "volume" in stock_df.columns and "close" in stock_df.columns and pos >= 19:
            vol_s = pd.to_numeric(stock_df["volume"].iloc[pos - 19:pos + 1], errors="coerce").dropna()
            close_s = pd.to_numeric(stock_df["close"].iloc[pos - 19:pos + 1], errors="coerce").dropna()
            if len(vol_s) >= 10 and len(close_s) >= 10:
                turnover_value_mean_20 = float((vol_s * close_s).mean())

        cache[(ts_code, pd.Timestamp(report_date))] = {
            "effective_date": effective_date,
            "future_ret_20": future_ret_20, "future_ret_60": future_ret_60, "future_ret_120": future_ret_120,
            "future_excess_ret_20": future_excess_ret_20, "future_excess_ret_60": future_excess_ret_60,
            "future_excess_ret_120": future_excess_ret_120,
            "future_mdd_60": future_mdd_60, "ret_120": ret_120,
            "mom_20": mom_20, "mom_60": mom_60,
            "turnover_value_mean_20": turnover_value_mean_20, "vol_20": vol_20,
            "drawdown_from_high_120": drawdown_from_high_120,
        }
    return cache


def build_change_events(df: pd.DataFrame, event_cache: Dict) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    by_stock = df.groupby("ts_code")
    total_stocks = len(by_stock)
    logger.info("构建变动事件: %s 只股票...", total_stocks)

    for idx, (ts_code, g) in enumerate(by_stock, 1):
        if idx % 100 == 0 or idx == total_stocks:
            logger.info("  进度: %s/%s", idx, total_stocks)
        reports = sorted(g["report_date"].dropna().unique())
        prev_snap: Optional[pd.DataFrame] = None

        for i, report_date in enumerate(reports):
            snap = g[g["report_date"] == report_date].copy().sort_values(["holder_rank", "holder_name_std"])
            ann_date = snap["ann_date"].dropna().min()
            metrics = event_cache.get((ts_code, pd.Timestamp(report_date)))
            if pd.isna(ann_date) or metrics is None:
                continue
            if i == 0 or prev_snap is None:
                prev_snap = snap
                continue

            cur_map = {row["holder_name_std"]: row for _, row in snap.iterrows()}
            prev_map = {row["holder_name_std"]: row for _, row in prev_snap.iterrows()}
            names = sorted(set(cur_map.keys()) | set(prev_map.keys()))

            industry_l2 = snap["industry_l2"].dropna().iloc[0] if "industry_l2" in snap.columns and snap["industry_l2"].notna().any() else None

            for name in names:
                cur = cur_map.get(name)
                prev = prev_map.get(name)
                if cur is not None and prev is None:
                    action_type = "entry"
                    cur_ratio = safe_float(cur.get("hold_float_ratio"))
                    delta_ratio = cur_ratio
                    delta_amt = safe_float(cur.get("hold_amount"))
                elif cur is None and prev is not None:
                    action_type = "exit"
                    cur_ratio = None
                    prev_ratio = safe_float(prev.get("hold_float_ratio"))
                    delta_ratio = None if prev_ratio is None else -prev_ratio
                    delta_amt = None
                else:
                    action_type, delta_ratio, delta_amt = _classify_overlap_action(cur, prev)
                    cur_ratio = safe_float(cur.get("hold_float_ratio"))

                base_row = cur if cur is not None else prev
                records.append({
                    "ts_code": ts_code,
                    "report_date": report_date,
                    "ann_date": ann_date,
                    "effective_date": metrics["effective_date"],
                    "holder_name_std": name,
                    "holder_type": base_row.get("holder_type"),
                    "holder_rank_curr": int(cur["holder_rank"]) if cur is not None and pd.notna(cur.get("holder_rank")) else None,
                    "hold_float_ratio_curr": cur_ratio if cur is not None else None,
                    "delta_hold_float_ratio": delta_ratio,
                    "action_type": action_type,
                    "tenure_curr": int(cur["tenure"]) if cur is not None and pd.notna(cur.get("tenure")) else 0,
                    "industry_l2": industry_l2,
                    "future_ret_20": metrics.get("future_ret_20"),
                    "future_ret_60": metrics.get("future_ret_60"),
                    "future_ret_120": metrics.get("future_ret_120"),
                    "future_excess_ret_20": metrics.get("future_excess_ret_20"),
                    "future_excess_ret_60": metrics.get("future_excess_ret_60"),
                    "future_excess_ret_120": metrics.get("future_excess_ret_120"),
                    "future_mdd_60": metrics.get("future_mdd_60"),
                    "ret_120": metrics.get("ret_120"),
                    "mom_20": metrics.get("mom_20"),
                    "mom_60": metrics.get("mom_60"),
                    "turnover_value_mean_20": metrics.get("turnover_value_mean_20"),
                    "vol_20": metrics.get("vol_20"),
                    "drawdown_from_high_120": metrics.get("drawdown_from_high_120"),
                })
            prev_snap = snap

    return pd.DataFrame(records)


def _safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return np.nan
    return float(s.mean())


def _safe_win_rate(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float((s > 0).mean())


def compute_picking_score(events: pd.DataFrame) -> pd.DataFrame:
    entry = events[events["action_type"] == "entry"]
    add = events[events["action_type"] == "add"]

    entry_agg = entry.groupby("holder_name_std").agg(
        entry_excess_ret_60=("future_excess_ret_60", lambda x: _safe_mean(x)),
        entry_win_rate_60=("future_excess_ret_60", lambda x: _safe_win_rate(x)),
    )
    add_agg = add.groupby("holder_name_std").agg(
        add_excess_ret_60=("future_excess_ret_60", lambda x: _safe_mean(x)),
        add_win_rate_60=("future_excess_ret_60", lambda x: _safe_win_rate(x)),
    )

    all_excess = pd.to_numeric(events["future_excess_ret_60"], errors="coerce")
    events_temp = events.copy()
    events_temp["excess_60"] = all_excess
    ir = events_temp.groupby("holder_name_std")["excess_60"].agg(
        lambda x: float(x.dropna().mean() / x.dropna().std()) if x.dropna().std() > EPS and x.dropna().notna().sum() >= 3 else np.nan
    ).rename("information_ratio")

    entry_120 = entry.groupby("holder_name_std")["future_excess_ret_120"].apply(_safe_mean).rename("entry_excess_ret_120")
    add_120 = add.groupby("holder_name_std")["future_excess_ret_120"].apply(_safe_mean).rename("add_excess_ret_120")

    result = entry_agg.join(add_agg, how="outer").join(ir, how="outer").join(entry_120, how="outer").join(add_120, how="outer")

    result["long_horizon_alpha"] = np.where(
        result["entry_excess_ret_120"].notna() & result["entry_excess_ret_60"].notna(),
        result["entry_excess_ret_120"] - result["entry_excess_ret_60"],
        np.nan
    )

    picking_raw = (
        0.30 * robust_zscore(result["entry_excess_ret_60"]) +
        0.25 * robust_zscore(result["add_excess_ret_60"]) +
        0.15 * robust_zscore(result["entry_win_rate_60"]) +
        0.10 * robust_zscore(result["add_win_rate_60"]) +
        0.12 * robust_zscore(result["information_ratio"]) +
        0.08 * robust_zscore(result["long_horizon_alpha"])
    )
    result["picking_score"] = clip_score(50 + 15 * picking_raw, 0, 100)
    return result


def compute_style_score(events: pd.DataFrame, top10_df: pd.DataFrame) -> pd.DataFrame:
    avg_tenure = events.groupby("holder_name_std")["tenure_curr"].apply(
        lambda x: _safe_mean(pd.to_numeric(x, errors="coerce"))
    ).rename("avg_tenure")

    stock_diversity = events.groupby("holder_name_std")["ts_code"].nunique().rename("sample_stocks_for_diversity")
    stock_diversity = np.log1p(stock_diversity).rename("stock_diversity")

    action_counts = events.groupby(["holder_name_std", "action_type"]).size().unstack(fill_value=0)
    for col in ["entry", "exit"]:
        if col not in action_counts.columns:
            action_counts[col] = 0
    total_by_holder = events.groupby("holder_name_std").size().rename("total_events")
    turnover_rate = ((action_counts["entry"] + action_counts["exit"]) / total_by_holder).rename("turnover_rate")

    ret_120_vals = pd.to_numeric(events["ret_120"], errors="coerce")
    ret_120_rank = ret_120_vals.rank(pct=True)
    entry_mask = events["action_type"] == "entry"
    entry_with_rank = pd.DataFrame({"holder_name_std": events.loc[entry_mask, "holder_name_std"], "ret_120_rank": ret_120_rank[entry_mask]})
    contrarian = entry_with_rank.groupby("holder_name_std")["ret_120_rank"].apply(
        lambda x: (x <= 0.3).mean() if len(x) > 0 else np.nan
    ).rename("contrarian_ratio")
    momentum = entry_with_rank.groupby("holder_name_std")["ret_120_rank"].apply(
        lambda x: (x >= 0.7).mean() if len(x) > 0 else np.nan
    ).rename("momentum_ratio")

    result = pd.DataFrame({
        "avg_tenure": avg_tenure,
        "stock_diversity": stock_diversity,
        "turnover_rate": turnover_rate,
        "contrarian_ratio": contrarian,
        "momentum_ratio": momentum,
    })

    style_raw = (
        0.30 * robust_zscore(result["avg_tenure"]) -
        0.25 * robust_zscore(result["turnover_rate"]) +
        0.25 * robust_zscore(result["contrarian_ratio"]) +
        0.20 * robust_zscore(result["stock_diversity"])
    )
    result["style_score"] = clip_score(50 + 12 * style_raw, 0, 100)
    return result


def compute_expertise_score(events: pd.DataFrame) -> pd.DataFrame:
    if "industry_l2" not in events.columns:
        return pd.DataFrame()

    events_with_ind = events.dropna(subset=["industry_l2"])
    if events_with_ind.empty:
        return pd.DataFrame()

    industry_coverage = events_with_ind.groupby("holder_name_std")["industry_l2"].nunique().rename("industry_coverage")

    entry_with_ind = events_with_ind[events_with_ind["action_type"].isin(["entry", "add"])]
    industry_excess = entry_with_ind.groupby(["holder_name_std", "industry_l2"])["future_excess_ret_60"].apply(
        lambda x: _safe_mean(pd.to_numeric(x, errors="coerce"))
    ).dropna()

    best_industry = industry_excess.groupby("holder_name_std").idxmax().map(lambda x: x[1] if isinstance(x, tuple) else None).rename("best_industry")
    best_industry_excess = industry_excess.groupby("holder_name_std").max().rename("best_industry_excess")
    worst_industry_excess = industry_excess.groupby("holder_name_std").min().rename("worst_industry_excess")
    industry_alpha_spread = (best_industry_excess - worst_industry_excess).rename("industry_alpha_spread")

    industry_counts = events_with_ind.groupby(["holder_name_std", "industry_l2"]).size()
    industry_shares = industry_counts.groupby("holder_name_std").apply(lambda x: (x ** 2).sum() / (x.sum() ** 2))
    industry_concentration = industry_shares.rename("industry_concentration")

    result = pd.DataFrame({
        "industry_coverage": industry_coverage,
        "best_industry": best_industry,
        "best_industry_excess": best_industry_excess,
        "industry_concentration": industry_concentration,
        "industry_alpha_spread": industry_alpha_spread,
    })

    expertise_raw = (
        0.30 * robust_zscore(result["best_industry_excess"]) +
        0.25 * robust_zscore(result["industry_alpha_spread"]) +
        0.25 * robust_zscore(result["industry_coverage"]) -
        0.20 * robust_zscore(result["industry_concentration"])
    )
    result["expertise_score"] = clip_score(50 + 12 * expertise_raw, 0, 100)
    return result


def compute_adapt_score(events: pd.DataFrame) -> pd.DataFrame:
    entry = events[events["action_type"] == "entry"].copy()
    if entry.empty:
        return pd.DataFrame()

    ret_120 = pd.to_numeric(entry["ret_120"], errors="coerce")
    ret_120_rank = ret_120.rank(pct=True)
    entry["ret_120_rank"] = ret_120_rank

    lowpos_mask = entry["ret_120_rank"] <= 0.3
    lowpos_fit = entry[lowpos_mask].groupby("holder_name_std")["future_excess_ret_60"].apply(
        lambda x: 50 + (_safe_mean(pd.to_numeric(x, errors="coerce")) or 0.0) * 100
    ).rename("lowpos_fit")

    tv_median = pd.to_numeric(entry["turnover_value_mean_20"], errors="coerce").median()
    smallcap_mask = pd.to_numeric(entry["turnover_value_mean_20"], errors="coerce") <= tv_median
    smallcap_fit = entry[smallcap_mask].groupby("holder_name_std")["future_excess_ret_60"].apply(
        lambda x: 50 + (_safe_mean(pd.to_numeric(x, errors="coerce")) or 0.0) * 100
    ).rename("smallcap_fit")

    growth_mask = pd.to_numeric(entry["mom_60"], errors="coerce") > 0
    growth_fit = entry[growth_mask].groupby("holder_name_std")["future_excess_ret_60"].apply(
        lambda x: 50 + (_safe_mean(pd.to_numeric(x, errors="coerce")) or 0.0) * 100
    ).rename("growth_fit")

    bench_ret_60 = pd.to_numeric(entry.get("future_ret_60", pd.Series(dtype=float)), errors="coerce") - pd.to_numeric(entry.get("future_excess_ret_60", pd.Series(dtype=float)), errors="coerce")
    bear_mask = bench_ret_60 < 0
    entry_bear = entry[bear_mask]
    if not entry_bear.empty:
        bear_fit = entry_bear.groupby("holder_name_std")["future_excess_ret_60"].apply(
            lambda x: 50 + (_safe_mean(pd.to_numeric(x, errors="coerce")) or 0.0) * 100
        ).rename("bear_fit")
    else:
        bear_fit = pd.Series(dtype=float, name="bear_fit")

    result = pd.DataFrame({
        "lowpos_fit": lowpos_fit,
        "smallcap_fit": smallcap_fit,
        "growth_fit": growth_fit,
        "bear_fit": bear_fit,
    })

    adapt_raw = (
        0.30 * robust_zscore(result["lowpos_fit"]) +
        0.25 * robust_zscore(result["smallcap_fit"]) +
        0.25 * robust_zscore(result["growth_fit"]) +
        0.20 * robust_zscore(result["bear_fit"])
    )
    result["adapt_score"] = clip_score(50 + 12 * adapt_raw, 0, 100)
    return result


def compute_risk_score(events: pd.DataFrame) -> pd.DataFrame:
    active = events[events["action_type"].isin(["entry", "add"])]

    avg_mdd_60 = active.groupby("holder_name_std")["future_mdd_60"].apply(
        lambda x: _safe_mean(pd.to_numeric(x, errors="coerce"))
    ).rename("avg_mdd_60")

    avg_vol_20 = active.groupby("holder_name_std")["vol_20"].apply(
        lambda x: _safe_mean(pd.to_numeric(x, errors="coerce"))
    ).rename("avg_vol_20")

    excess_rets = pd.to_numeric(active["future_excess_ret_60"], errors="coerce")
    tail_loss = active.groupby("holder_name_std")["future_excess_ret_60"].apply(
        lambda x: pd.to_numeric(x, errors="coerce").quantile(0.1) if pd.to_numeric(x, errors="coerce").notna().sum() >= 3 else np.nan
    ).rename("tail_loss")

    reduce_exit = events[events["action_type"].isin(["reduce", "exit"])].copy()
    if not reduce_exit.empty and "future_excess_ret_60" in reduce_exit.columns:
        reduce_exit_excess = pd.to_numeric(reduce_exit["future_excess_ret_60"], errors="coerce")
        stop_loss = reduce_exit[reduce_exit_excess < 0].groupby("holder_name_std").size() / reduce_exit.groupby("holder_name_std").size()
        stop_loss_ratio = stop_loss.rename("stop_loss_ratio")
    else:
        stop_loss_ratio = pd.Series(dtype=float, name="stop_loss_ratio")

    result = pd.DataFrame({
        "avg_mdd_60": avg_mdd_60,
        "avg_vol_20": avg_vol_20,
        "tail_loss": tail_loss,
        "stop_loss_ratio": stop_loss_ratio,
    })

    risk_raw = (
        -0.35 * robust_zscore(result["avg_mdd_60"]) -
        0.25 * robust_zscore(result["avg_vol_20"]) -
        0.25 * robust_zscore(result["tail_loss"]) +
        0.15 * robust_zscore(result["stop_loss_ratio"])
    )
    result["risk_score"] = clip_score(50 + 12 * risk_raw, 0, 100)
    return result


def compute_scale_score(top10_df: pd.DataFrame) -> pd.DataFrame:
    avg_hold_ratio = top10_df.groupby("holder_name_std")["hold_float_ratio"].apply(
        lambda x: _safe_mean(pd.to_numeric(x, errors="coerce"))
    ).rename("avg_hold_ratio")

    total_stock_count = top10_df.groupby("holder_name_std")["ts_code"].nunique().rename("total_stock_count")

    sorted_df = top10_df.sort_values(["holder_name_std", "report_date"])
    first_half = sorted_df.groupby("holder_name_std").head(len(sorted_df) // 2 + 1)
    second_half = sorted_df.groupby("holder_name_std").tail(len(sorted_df) // 2 + 1)
    early_ratio = first_half.groupby("holder_name_std")["hold_float_ratio"].apply(
        lambda x: _safe_mean(pd.to_numeric(x, errors="coerce"))
    )
    late_ratio = second_half.groupby("holder_name_std")["hold_float_ratio"].apply(
        lambda x: _safe_mean(pd.to_numeric(x, errors="coerce"))
    )
    hold_ratio_trend = (late_ratio - early_ratio).rename("hold_ratio_trend")

    result = pd.DataFrame({
        "avg_hold_ratio": avg_hold_ratio,
        "total_stock_count": total_stock_count,
        "hold_ratio_trend": hold_ratio_trend,
    })

    scale_raw = (
        0.40 * robust_zscore(result["avg_hold_ratio"]) +
        0.35 * robust_zscore(result["total_stock_count"]) +
        0.25 * robust_zscore(result["hold_ratio_trend"])
    )
    result["scale_score"] = clip_score(50 + 10 * scale_raw, 0, 100)
    return result


def compute_holder_portrait(events: pd.DataFrame, top10_df: pd.DataFrame,
                            industry_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    logger.info("计算选股能力...")
    picking = compute_picking_score(events)

    logger.info("计算持仓风格...")
    style = compute_style_score(events, top10_df)

    logger.info("计算行业专长...")
    expertise = compute_expertise_score(events)

    logger.info("计算场景适配...")
    adapt = compute_adapt_score(events)

    logger.info("计算风控能力...")
    risk = compute_risk_score(events)

    logger.info("计算规模影响力...")
    scale = compute_scale_score(top10_df)

    holder_meta = events.groupby("holder_name_std").agg(
        holder_type=("holder_type", lambda x: x.mode().iloc[0] if len(x.dropna()) > 0 else None),
        sample_total=("action_type", "count"),
    )

    action_counts = events.groupby(["holder_name_std", "action_type"]).size().unstack(fill_value=0)
    for col in ["entry", "add", "reduce", "exit"]:
        if col not in action_counts.columns:
            action_counts[col] = 0
    action_counts = action_counts.rename(columns={
        "entry": "sample_entry", "add": "sample_add",
        "reduce": "sample_reduce", "exit": "sample_exit",
    })
    keep_cols = ["sample_entry", "sample_add", "sample_reduce", "sample_exit"]
    action_counts = action_counts[[c for c in keep_cols if c in action_counts.columns]]

    sample_stocks = events.groupby("holder_name_std")["ts_code"].nunique().rename("sample_stocks")
    sample_periods = events.groupby("holder_name_std")["report_date"].nunique().rename("sample_periods")

    portrait = holder_meta.join(action_counts, how="left").join(
        pd.DataFrame({"sample_stocks": sample_stocks, "sample_periods": sample_periods}), how="left"
    ).join(picking, how="left").join(style, how="left").join(expertise, how="left").join(
        adapt, how="left"
    ).join(risk, how="left").join(scale, how="left")

    portrait = portrait.reset_index()

    for col in ["sample_total", "sample_entry", "sample_add", "sample_reduce", "sample_exit", "sample_stocks", "sample_periods"]:
        if col in portrait.columns:
            portrait[col] = portrait[col].fillna(0).astype(int)

    drop_cols = ["entry_excess_ret_120", "add_excess_ret_120", "sample_stocks_for_diversity"]
    portrait = portrait.drop(columns=[c for c in drop_cols if c in portrait.columns], errors="ignore")

    return portrait


def compute_composite_score(portrait: pd.DataFrame) -> pd.DataFrame:
    if portrait.empty:
        return portrait

    for col in ["picking_score", "style_score", "expertise_score", "adapt_score", "risk_score", "scale_score"]:
        if col not in portrait.columns:
            portrait[col] = 50.0
        portrait[col] = portrait[col].fillna(50.0)

    portrait["composite_raw"] = (
        0.40 * portrait["picking_score"] +
        0.05 * portrait["style_score"] +
        0.10 * portrait["expertise_score"] +
        0.25 * portrait["adapt_score"] +
        0.05 * portrait["risk_score"] +
        0.15 * portrait["scale_score"]
    )

    portrait["prior_score"] = portrait["holder_type"].map(map_holder_prior_score).fillna(50.0)

    n_i = portrait["sample_entry"].fillna(0) + portrait["sample_add"].fillna(0)
    post_w = np.where(n_i < 3, 0.10, np.where(n_i < 5, 0.25, np.where(n_i < 10, 0.50, np.where(n_i < 20, 0.75, 0.90))))
    portrait["shrinkage_weight"] = post_w
    portrait["composite_score"] = post_w * portrait["composite_raw"] + (1 - post_w) * portrait["prior_score"]

    portrait["quality_grade"] = np.where(
        portrait["composite_score"] >= 75, "A",
        np.where(portrait["composite_score"] >= 60, "B",
                 np.where(portrait["composite_score"] >= 45, "C", "D"))
    )

    contrarian = portrait["contrarian_ratio"].fillna(0)
    tenure = portrait["avg_tenure"].fillna(0)
    momentum_r = portrait["momentum_ratio"].fillna(0)
    growth_f = portrait["growth_fit"].fillna(50)
    portrait["style_label"] = np.where(
        (contrarian >= 0.6) & (tenure >= 4), "价值型",
        np.where((momentum_r >= 0.4) & (growth_f >= 55), "成长型", "均衡型")
    )

    portrait["period_label"] = np.where(
        tenure >= 5, "长线",
        np.where(tenure >= 2, "中线", "短线")
    )

    best_ind_excess = portrait["best_industry_excess"].fillna(0)
    ind_conc = portrait["industry_concentration"].fillna(0)
    ind_cov = portrait["industry_coverage"].fillna(0)
    portrait["industry_label"] = np.where(
        (best_ind_excess >= 60) & (ind_conc >= 0.4), "行业专家",
        np.where(ind_cov >= 5, "行业分散", "无明确行业偏好")
    )

    picking_s = portrait["picking_score"]
    risk_s = portrait["risk_score"]
    lowpos_f = portrait["lowpos_fit"].fillna(50)
    abilities = []
    for _, row in portrait.iterrows():
        tags = []
        if row["picking_score"] >= 70:
            tags.append("选股强")
        if row["risk_score"] >= 70:
            tags.append("风控强")
        if row["contrarian_ratio"] >= 0.5 and row["lowpos_fit"] >= 60:
            tags.append("逆向高手")
        abilities.append(",".join(tags) if tags else "无突出能力")
    portrait["ability_label"] = abilities

    portrait["profile_scope"] = PROFILE_SCOPE_FULL
    portrait["profile_asof_date"] = None

    return portrait


def filter_events_asof(events: pd.DataFrame, effective_date: pd.Timestamp,
                       min_realized_bdays: int = 60) -> pd.DataFrame:
    if events.empty or pd.isna(effective_date):
        return pd.DataFrame(columns=events.columns)
    if "effective_date" not in events.columns:
        return events.iloc[0:0].copy()
    eff = pd.to_datetime(events["effective_date"], errors="coerce")
    asof = pd.Timestamp(effective_date)
    cutoff = asof - pd.offsets.BDay(min_realized_bdays)
    mask = eff.notna() & (eff <= cutoff)
    if "future_excess_ret_60" in events.columns:
        mask &= pd.to_numeric(events["future_excess_ret_60"], errors="coerce").notna()
    return events.loc[mask].copy()


def build_portrait_asof(events: pd.DataFrame, effective_date: pd.Timestamp) -> pd.DataFrame:
    hist_events = filter_events_asof(events, effective_date)
    if hist_events.empty:
        return pd.DataFrame()
    portrait = compute_holder_portrait(hist_events, hist_events, None)
    if portrait.empty:
        return portrait
    portrait = compute_composite_score(portrait)
    portrait["profile_scope"] = PROFILE_SCOPE_ASOF
    portrait["profile_asof_date"] = pd.Timestamp(effective_date).strftime("%Y%m%d")
    return portrait


def save_portrait(df: pd.DataFrame, dry_run: bool = False) -> int:
    if df.empty:
        return 0
    if dry_run:
        logger.info("[DRY RUN] portrait rows=%s", len(df))
        return len(df)

    records = df.to_dict(orient="records")
    cleaned = []
    for rec in records:
        clean_rec = {}
        for k, v in rec.items():
            if isinstance(v, float) and np.isnan(v):
                clean_rec[k] = None
            elif v is pd.NaT:
                clean_rec[k] = None
            else:
                clean_rec[k] = v
        cleaned.append(clean_rec)

    if not cleaned:
        return 0

    columns = list(cleaned[0].keys())
    non_key_columns = [c for c in columns if c not in PORTRAIT_UNIQUE_KEYS]
    col_clause = ", ".join(columns)
    placeholders = ", ".join([f":{c}" for c in columns])
    update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in non_key_columns])
    upsert_sql = f"""
        INSERT INTO {PORTRAIT_TABLE} ({col_clause})
        VALUES ({placeholders})
        ON CONFLICT ({", ".join(PORTRAIT_UNIQUE_KEYS)}) DO UPDATE SET {update_clause}
    """

    with get_engine().begin() as conn:
        for i in range(0, len(cleaned), 1000):
            batch = cleaned[i:i + 1000]
            conn.execute(text(upsert_sql), batch)
    return len(cleaned)


def process_single(ts_code: str, lookback_years: int = 5, dry_run: bool = False) -> None:
    ts_code = normalize_ts_code(ts_code)
    logger.info("处理单只股票: %s", ts_code)

    top10_df = load_top10_data(ts_codes=[ts_code], lookback_years=lookback_years)
    if top10_df.empty:
        logger.warning("%s 无 top10 数据", ts_code)
        return

    industry_df = load_industry_info()
    if not industry_df.empty:
        top10_df = top10_df.merge(industry_df.drop(columns=["name"], errors="ignore"), on="ts_code", how="left")

    top10_df = build_holder_tenure(top10_df)

    start_date = top10_df["ann_date"].dropna().min()
    if pd.isna(start_date):
        start_date = top10_df["report_date"].dropna().min()
    end_date = pd.Timestamp.today().normalize()

    stock_df = load_market_data_from_db(ts_code, start_date - pd.Timedelta(days=250), end_date)
    if stock_df.empty:
        logger.warning("%s 无行情数据", ts_code)
        return

    bench_df = load_bench_data(start_date - pd.Timedelta(days=250), end_date)

    event_cache = build_event_metrics_cache(top10_df, {ts_code: stock_df}, bench_df)
    events_df = build_change_events(top10_df, event_cache)

    if events_df.empty:
        logger.warning("%s 无变动事件", ts_code)
        return

    logger.info("事件数: %s", len(events_df))
    portrait = compute_holder_portrait(events_df, top10_df, industry_df)
    portrait = compute_composite_score(portrait)

    if portrait.empty:
        logger.warning("%s 画像为空", ts_code)
        return

    n = save_portrait(portrait, dry_run=dry_run)
    logger.info("%s 画像: %s 个股东, 写入 %s 行", ts_code, len(portrait), n)

    if dry_run and not portrait.empty:
        preview_cols = ["holder_name_std", "holder_type", "sample_stocks", "composite_score", "quality_grade", "style_label", "period_label"]
        available = [c for c in preview_cols if c in portrait.columns]
        print(portrait[available].head(10).to_string(index=False))


def process_batch(limit: Optional[int] = None, lookback_years: int = 5, dry_run: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("批量处理: limit=%s, lookback_years=%s", limit, lookback_years)
    logger.info("=" * 60)

    pool = load_stock_pool(limit)
    if not pool:
        logger.warning("股票池为空")
        return
    ts_codes = [x[0] for x in pool]
    logger.info("股票池: %s 只", len(ts_codes))

    top10_df = load_top10_data(ts_codes=ts_codes, lookback_years=lookback_years)
    if top10_df.empty:
        logger.warning("top10 数据为空")
        return
    logger.info("top10 数据: %s 行", len(top10_df))

    industry_df = load_industry_info()
    if not industry_df.empty:
        top10_df = top10_df.merge(industry_df.drop(columns=["name"], errors="ignore"), on="ts_code", how="left")

    top10_df = build_holder_tenure(top10_df)

    stock_df_map: Dict[str, pd.DataFrame] = {}
    unique_codes = top10_df["ts_code"].drop_duplicates().tolist()
    start_date_global = top10_df["ann_date"].dropna().min()
    if pd.isna(start_date_global):
        start_date_global = top10_df["report_date"].dropna().min()
    end_date = pd.Timestamp.today().normalize()

    from tqdm import tqdm
    for ts_code in tqdm(unique_codes, desc="加载行情", unit="股票"):
        g = top10_df[top10_df["ts_code"] == ts_code]
        start_date = g["ann_date"].dropna().min()
        if pd.isna(start_date):
            start_date = g["report_date"].dropna().min()
        if pd.isna(start_date):
            continue
        stock_df = load_market_data_from_db(ts_code, start_date - pd.Timedelta(days=250), end_date)
        if not stock_df.empty:
            stock_df_map[ts_code] = stock_df

    valid_codes = [c for c in unique_codes if c in stock_df_map]
    top10_df = top10_df[top10_df["ts_code"].isin(valid_codes)].copy()
    logger.info("有效股票: %s 只", len(valid_codes))

    bench_df = load_bench_data(start_date_global - pd.Timedelta(days=250), end_date)

    event_cache = build_event_metrics_cache(top10_df, stock_df_map, bench_df)
    events_df = build_change_events(top10_df, event_cache)
    logger.info("变动事件: %s 条", len(events_df))

    if events_df.empty:
        logger.warning("无变动事件")
        return

    portrait = compute_holder_portrait(events_df, top10_df, industry_df)
    portrait = compute_composite_score(portrait)
    logger.info("画像: %s 个股东", len(portrait))

    n = save_portrait(portrait, dry_run=dry_run)
    logger.info("=" * 60)
    logger.info("处理完成: %s 个股东, 写入 %s 行", len(portrait), n)
    logger.info("=" * 60)

    if not portrait.empty:
        grade_dist = portrait["quality_grade"].value_counts()
        logger.info("质量等级分布:")
        for grade, cnt in grade_dist.items():
            logger.info("  %s: %s (%.1f%%)", grade, cnt, cnt / len(portrait) * 100)

        style_dist = portrait["style_label"].value_counts()
        logger.info("风格标签分布:")
        for label, cnt in style_dist.items():
            logger.info("  %s: %s (%.1f%%)", label, cnt, cnt / len(portrait) * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="股东投资质量画像模型")
    parser.add_argument("--mode", choices=["single", "batch"], default="batch")
    parser.add_argument("--ts_code", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lookback_years", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    ensure_portrait_table()

    if args.mode == "single":
        if not args.ts_code:
            parser.error("single 模式需要 --ts_code")
        process_single(args.ts_code, args.lookback_years, args.dry_run)
    else:
        process_batch(args.limit, args.lookback_years, args.dry_run)


if __name__ == "__main__":
    main()
