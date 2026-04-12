# -*- coding: utf-8 -*-
"""
Purpose:
    股东投资质量画像模型 - 数据探索脚本
    验证 top10 持股数据的分布、指标计算可行性、边界情况

Inputs:
    - 数据库表: stock_top10_holders_tushare
    - 数据库表: stock_k_data (行情)
    - 数据库表: stock_pools (行业信息)

Outputs:
    - 控制台: 探索报告
    - CSV: holder_portrait_explore_result.csv (可选)

How to Run:
    python financial_factors/holder_portrait_explore.py --limit 100
    python financial_factors/holder_portrait_explore.py --limit 50 --csv

Examples:
    python financial_factors/holder_portrait_explore.py --limit 100
    python financial_factors/holder_portrait_explore.py --limit 20 --csv

Side Effects:
    - 无写库操作
    - 可选写 CSV 文件
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_session, query_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

INPUT_TABLE = "stock_top10_holders_tushare"
EPS = 1e-12


def normalize_holder_name(name: object) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    s = str(name).strip()
    replacements = {
        "（": "(", "）": ")", "股份有限公司": "股份", "有限合伙企业": "有限合伙",
        "－": "-", "—": "-", "\u3000": "", " ": "", "\u201c": "", "\u201d": "",
        "\u2018": "", "\u2019": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s.upper()


def map_holder_prior_score(holder_type: object) -> float:
    v = "" if holder_type is None else str(holder_type).upper()
    if any(k in v for k in ["社保", "全国社会保障", "社保基金"]):
        return 80.0
    if "QFII" in v or "境外" in v:
        return 76.0
    if any(k in v for k in ["保险", "险资"]):
        return 78.0
    if any(k in v for k in ["基金", "公募"]):
        return 68.0
    if any(k in v for k in ["券商", "证券", "资管"]):
        return 62.0
    if any(k in v for k in ["信托"]):
        return 55.0
    if any(k in v for k in ["个人", "自然人"]):
        return 45.0
    if any(k in v for k in ["国有", "国资", "汇金", "证金"]):
        return 80.0
    return 50.0


def safe_float(v: object) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def normalize_ts_code(ts_code: str) -> str:
    ts_code = str(ts_code).strip().upper()
    if ts_code.endswith((".SZ", ".SH", ".BJ")):
        return ts_code
    if ts_code.startswith("6"):
        return f"{ts_code}.SH"
    if ts_code.startswith(("8", "4")):
        return f"{ts_code}.BJ"
    return f"{ts_code}.SZ"


def load_top10_data(ts_codes: Optional[List[str]] = None, lookback_years: int = 5) -> pd.DataFrame:
    with get_session() as session:
        df = query_df(session, INPUT_TABLE)
    if df.empty:
        return df
    df = df.copy()
    df["ts_code"] = df["ts_code"].astype(str).map(normalize_ts_code)
    if ts_codes:
        ts_code_set = {normalize_ts_code(x) for x in ts_codes}
        df = df[df["ts_code"].isin(ts_code_set)].copy()
    df["report_date"] = pd.to_datetime(df["report_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    df["ann_date"] = pd.to_datetime(df["ann_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
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
    df = fetch_market_data("000300.SH", start, end, fqt=0)
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
    trading_index_naive = trading_index.tz_localize(None) if hasattr(trading_index, "tz") and trading_index.tz else trading_index
    pos = trading_index_naive.searchsorted(ann_date)
    if pos >= len(trading_index_naive):
        return None
    if trading_index_naive[pos] == ann_date:
        pos += 1
    if pos >= len(trading_index_naive):
        return None
    return pd.Timestamp(trading_index_naive[pos])


def calc_future_return(prices: pd.Series, start_idx: int, horizon: int) -> Optional[float]:
    end_idx = start_idx + horizon
    if start_idx < 0 or end_idx >= len(prices):
        return None
    p0 = prices.iloc[start_idx]
    p1 = prices.iloc[end_idx]
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
    dd = (window / base - 1.0).min()
    return float(dd)


def calc_backward_return(prices: pd.Series, idx: int, lookback: int) -> Optional[float]:
    prev_idx = idx - lookback
    if prev_idx < 0 or idx >= len(prices):
        return None
    p0 = prices.iloc[prev_idx]
    p1 = prices.iloc[idx]
    if pd.isna(p0) or pd.isna(p1) or abs(p0) < EPS:
        return None
    return float(p1 / p0 - 1.0)


def build_holder_tenure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ts_code", "holder_name_std", "report_date"]).copy()
    if df.empty:
        df["tenure"] = []
        return df
    df["tenure"] = 1
    tenure_parts = []
    for _, g in df.groupby(["ts_code", "holder_name_std"], sort=False):
        g = g.sort_values("report_date").copy()
        month_diff = g["report_date"].diff().dt.days.div(30.0).round().fillna(3)
        reset_flag = ~month_diff.isin([3, 6])
        g["tenure"] = reset_flag.cumsum()
        g["tenure"] = g.groupby("tenure").cumcount() + 1
        tenure_parts.append(g)
    return pd.concat(tenure_parts, ignore_index=False).sort_values(
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


def _classify_overlap_action(cur_row: pd.Series, prev_row: pd.Series,
                             abs_threshold: float = 0.10, rel_threshold: float = 0.05) -> Tuple[str, Optional[float], Optional[float]]:
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
    if (delta_ratio > abs_threshold) or (rel_change is not None and rel_change > rel_threshold and delta_ratio > 0):
        return "add", delta_ratio, delta_amt
    if (delta_ratio < -abs_threshold) or (rel_change is not None and rel_change > rel_threshold and delta_ratio < 0):
        return "reduce", delta_ratio, delta_amt
    return "hold", delta_ratio, delta_amt


def build_change_events(df: pd.DataFrame, event_cache: Dict) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    by_stock = df.groupby("ts_code")
    total_stocks = len(by_stock)
    logger.info("处理 %s 只股票的变动事件...", total_stocks)

    for idx, (ts_code, g) in enumerate(by_stock, 1):
        if idx % 50 == 0 or idx == total_stocks:
            logger.info("  处理进度: %s/%s 只股票", idx, total_stocks)
        reports = sorted(g["report_date"].dropna().unique())
        prev_snap: Optional[pd.DataFrame] = None
        prev_report_date: Optional[pd.Timestamp] = None

        for i, report_date in enumerate(reports):
            snap = g[g["report_date"] == report_date].copy().sort_values(["holder_rank", "holder_name_std"])
            ann_date = snap["ann_date"].dropna().min()
            metrics = event_cache.get((ts_code, pd.Timestamp(report_date)))
            if pd.isna(ann_date) or metrics is None:
                continue
            if i == 0 or prev_snap is None:
                prev_snap = snap
                prev_report_date = report_date
                continue

            cur_map = {row["holder_name_std"]: row for _, row in snap.iterrows()}
            prev_map = {row["holder_name_std"]: row for _, row in prev_snap.iterrows()}
            names = sorted(set(cur_map.keys()) | set(prev_map.keys()))

            for name in names:
                cur = cur_map.get(name)
                prev = prev_map.get(name)
                if cur is not None and prev is None:
                    action_type = "entry"
                    cur_ratio = safe_float(cur.get("hold_float_ratio"))
                    prev_ratio = None
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
                    prev_ratio = safe_float(prev.get("hold_float_ratio"))

                base_row = cur if cur is not None else prev
                records.append({
                    "ts_code": ts_code,
                    "report_date": report_date,
                    "ann_date": ann_date,
                    "effective_date": metrics.get("effective_date"),
                    "holder_name_std": name,
                    "holder_type": base_row.get("holder_type"),
                    "holder_rank_curr": int(cur["holder_rank"]) if cur is not None and pd.notna(cur.get("holder_rank")) else None,
                    "hold_float_ratio_curr": cur_ratio,
                    "hold_float_ratio_prev": prev_ratio,
                    "delta_hold_float_ratio": delta_ratio,
                    "action_type": action_type,
                    "tenure_curr": int(cur["tenure"]) if cur is not None and pd.notna(cur.get("tenure")) else 0,
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
            prev_report_date = report_date

    return pd.DataFrame(records)


def build_event_metrics_cache(df: pd.DataFrame, stock_df_map: Dict[str, pd.DataFrame],
                              bench_df: pd.DataFrame) -> Dict[Tuple[str, pd.Timestamp], Dict]:
    cache: Dict[Tuple[str, pd.Timestamp], Dict] = {}
    groups = list(df.groupby(["ts_code", "report_date"]))
    total_groups = len(groups)
    logger.info("构建事件市场指标缓存: %s 个事件...", total_groups)

    for idx, ((ts_code, report_date), snap) in enumerate(groups, 1):
        if idx % 100 == 0 or idx == total_groups:
            logger.info("  缓存进度: %s/%s", idx, total_groups)
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
            high = window.max()
            cur = close.iloc[pos]
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
            "future_ret_20": future_ret_20,
            "future_ret_60": future_ret_60,
            "future_ret_120": future_ret_120,
            "future_excess_ret_20": future_excess_ret_20,
            "future_excess_ret_60": future_excess_ret_60,
            "future_excess_ret_120": future_excess_ret_120,
            "future_mdd_60": future_mdd_60,
            "ret_120": ret_120,
            "mom_20": mom_20,
            "mom_60": mom_60,
            "turnover_value_mean_20": turnover_value_mean_20,
            "vol_20": vol_20,
            "drawdown_from_high_120": drawdown_from_high_120,
        }
    return cache


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


def explore_data(limit: int = 100, csv: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("股东投资质量画像模型 - 数据探索")
    logger.info("=" * 60)

    pool = load_stock_pool(limit=limit)
    if not pool:
        logger.warning("股票池为空")
        return
    ts_codes = [x[0] for x in pool]
    logger.info("股票池: %s 只", len(ts_codes))

    logger.info("加载 top10 数据...")
    top10_df = load_top10_data(ts_codes=ts_codes, lookback_years=5)
    if top10_df.empty:
        logger.warning("top10 数据为空")
        return
    logger.info("top10 数据: %s 行, %s 只股票", len(top10_df), top10_df["ts_code"].nunique())

    logger.info("加载行业信息...")
    industry_df = load_industry_info()
    if not industry_df.empty:
        top10_df = top10_df.merge(industry_df.drop(columns=["name"], errors="ignore"), on="ts_code", how="left")

    logger.info("构建持仓任期...")
    top10_df = build_holder_tenure(top10_df)

    logger.info("加载行情数据...")
    stock_df_map: Dict[str, pd.DataFrame] = {}
    unique_codes = top10_df["ts_code"].drop_duplicates().tolist()
    start_date_global = top10_df["ann_date"].dropna().min()
    if pd.isna(start_date_global):
        start_date_global = top10_df["report_date"].dropna().min()
    end_date_global = pd.Timestamp.today().normalize()

    for ts_code in unique_codes:
        g = top10_df[top10_df["ts_code"] == ts_code]
        start_date = g["ann_date"].dropna().min()
        if pd.isna(start_date):
            start_date = g["report_date"].dropna().min()
        if pd.isna(start_date):
            continue
        stock_df = load_market_data_from_db(ts_code, start_date - pd.Timedelta(days=250), end_date_global)
        if not stock_df.empty:
            stock_df_map[ts_code] = stock_df

    logger.info("行情数据: %s 只股票", len(stock_df_map))

    valid_codes = [c for c in unique_codes if c in stock_df_map]
    top10_df = top10_df[top10_df["ts_code"].isin(valid_codes)].copy()
    logger.info("有效股票: %s 只", len(valid_codes))

    logger.info("加载基准指数行情...")
    bench_df = load_bench_data(start_date_global - pd.Timedelta(days=250), end_date_global)

    logger.info("构建事件市场指标缓存...")
    event_cache = build_event_metrics_cache(top10_df, stock_df_map, bench_df)
    logger.info("事件缓存: %s 条", len(event_cache))

    logger.info("构建变动事件...")
    events_df = build_change_events(top10_df, event_cache)
    if events_df.empty:
        logger.warning("无变动事件")
        return
    logger.info("变动事件: %s 条", len(events_df))

    logger.info("")
    logger.info("=" * 60)
    logger.info("探索报告")
    logger.info("=" * 60)

    logger.info("")
    logger.info("--- 1. 股东出现频次分布 ---")
    holder_stock_cnt = events_df.groupby("holder_name_std")["ts_code"].nunique()
    logger.info("去重股东数: %s", len(holder_stock_cnt))
    for threshold in [1, 2, 3, 5, 10, 20]:
        cnt = (holder_stock_cnt >= threshold).sum()
        logger.info("  出现在 >= %s 只股票: %s 个股东 (%.1f%%)", threshold, cnt, cnt / len(holder_stock_cnt) * 100)

    logger.info("")
    logger.info("--- 2. 事件类型分布 ---")
    action_counts = events_df["action_type"].value_counts()
    for action, cnt in action_counts.items():
        logger.info("  %s: %s (%.1f%%)", action, cnt, cnt / len(events_df) * 100)

    logger.info("")
    logger.info("--- 3. 选股能力指标分布 ---")
    for action in ["entry", "add"]:
        sub = events_df[events_df["action_type"] == action]
        for col in ["future_excess_ret_60", "future_excess_ret_120", "future_mdd_60"]:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) > 0:
                logger.info("  %s %s: n=%s, mean=%.4f, median=%.4f, std=%.4f, p25=%.4f, p75=%.4f",
                            action, col, len(vals), vals.mean(), vals.median(), vals.std(),
                            vals.quantile(0.25), vals.quantile(0.75))
            else:
                logger.info("  %s %s: 无有效数据", action, col)

    logger.info("")
    logger.info("--- 4. 持仓风格指标分布 ---")
    tenure_vals = pd.to_numeric(events_df["tenure_curr"], errors="coerce").dropna()
    if len(tenure_vals) > 0:
        logger.info("  tenure: n=%s, mean=%.2f, median=%.1f, p75=%.1f", len(tenure_vals), tenure_vals.mean(), tenure_vals.median(), tenure_vals.quantile(0.75))

    ret_120_vals = pd.to_numeric(events_df["ret_120"], errors="coerce").dropna()
    if len(ret_120_vals) > 0:
        ret_120_rank = ret_120_vals.rank(pct=True)
        entry_mask = events_df.loc[ret_120_vals.index, "action_type"] == "entry"
        contrarian = (ret_120_rank[entry_mask] <= 0.3).mean() if entry_mask.sum() > 0 else np.nan
        momentum = (ret_120_rank[entry_mask] >= 0.7).mean() if entry_mask.sum() > 0 else np.nan
        logger.info("  逆向比例 (entry at ret_120_rank<=0.3): %.3f", contrarian)
        logger.info("  顺向比例 (entry at ret_120_rank>=0.7): %.3f", momentum)

    logger.info("")
    logger.info("--- 5. 行业专长指标分布 ---")
    if "industry_l2" in events_df.columns:
        industry_counts = events_df.dropna(subset=["industry_l2"]).groupby("holder_name_std")["industry_l2"].nunique()
        if len(industry_counts) > 0:
            logger.info("  行业覆盖数: mean=%.2f, median=%.1f, p75=%.1f",
                        industry_counts.mean(), industry_counts.median(), industry_counts.quantile(0.75))

        entry_events = events_df[(events_df["action_type"] == "entry") & events_df["industry_l2"].notna()]
        if not entry_events.empty:
            industry_excess = entry_events.groupby(["holder_name_std", "industry_l2"])["future_excess_ret_60"].apply(
                lambda x: pd.to_numeric(x, errors="coerce").mean()
            ).dropna()
            if len(industry_excess) > 0:
                best_by_holder = industry_excess.groupby("holder_name_std").max()
                logger.info("  最强行业超额: mean=%.4f, median=%.4f, p75=%.4f",
                            best_by_holder.mean(), best_by_holder.median(), best_by_holder.quantile(0.75))

    logger.info("")
    logger.info("--- 6. 场景适配指标分布 ---")
    entry_events = events_df[events_df["action_type"] == "entry"]
    if not entry_events.empty and "ret_120" in entry_events.columns:
        ret_120_e = pd.to_numeric(entry_events["ret_120"], errors="coerce")
        ret_120_rank_e = ret_120_e.rank(pct=True)
        lowpos_mask = ret_120_rank_e <= 0.3
        lowpos_excess = pd.to_numeric(entry_events.loc[lowpos_mask, "future_excess_ret_60"], errors="coerce").dropna()
        if len(lowpos_excess) > 0:
            logger.info("  低位收筹超额: n=%s, mean=%.4f", len(lowpos_excess), lowpos_excess.mean())

    if not entry_events.empty and "mom_60" in entry_events.columns:
        growth_mask = pd.to_numeric(entry_events["mom_60"], errors="coerce") > 0
        growth_excess = pd.to_numeric(entry_events.loc[growth_mask, "future_excess_ret_60"], errors="coerce").dropna()
        if len(growth_excess) > 0:
            logger.info("  成长股超额: n=%s, mean=%.4f", len(growth_excess), growth_excess.mean())

    logger.info("")
    logger.info("--- 7. 风控能力指标分布 ---")
    for action in ["entry", "add"]:
        sub = events_df[events_df["action_type"] == action]
        mdd_vals = pd.to_numeric(sub["future_mdd_60"], errors="coerce").dropna()
        if len(mdd_vals) > 0:
            logger.info("  %s MDD_60: n=%s, mean=%.4f, median=%.4f", action, len(mdd_vals), mdd_vals.mean(), mdd_vals.median())
        vol_vals = pd.to_numeric(sub["vol_20"], errors="coerce").dropna()
        if len(vol_vals) > 0:
            logger.info("  %s vol_20: n=%s, mean=%.4f, median=%.4f", action, len(vol_vals), vol_vals.mean(), vol_vals.median())

    logger.info("")
    logger.info("--- 8. 规模影响力指标分布 ---")
    hold_ratio_vals = pd.to_numeric(events_df["hold_float_ratio_curr"], errors="coerce").dropna()
    if len(hold_ratio_vals) > 0:
        logger.info("  持股比例: n=%s, mean=%.4f, median=%.4f", len(hold_ratio_vals), hold_ratio_vals.mean(), hold_ratio_vals.median())

    logger.info("")
    logger.info("--- 9. 名称标准化歧义检查 ---")
    raw_names = top10_df["holder_name"].dropna().unique()
    std_names = top10_df["holder_name_std"].dropna().unique()
    logger.info("  原始名称数: %s, 标准化后: %s, 压缩比: %.2f%%",
                len(raw_names), len(std_names), (1 - len(std_names) / max(1, len(raw_names))) * 100)

    name_map = top10_df.groupby("holder_name_std")["holder_name"].nunique()
    ambiguous = name_map[name_map > 1].sort_values(ascending=False)
    if len(ambiguous) > 0:
        logger.info("  歧义名称 (标准化后对应多个原始名称): %s 个", len(ambiguous))
        for name, cnt in ambiguous.head(10).items():
            originals = top10_df[top10_df["holder_name_std"] == name]["holder_name"].unique()[:5]
            logger.info("    %s -> %s 个原始名称: %s", name, cnt, list(originals))

    logger.info("")
    logger.info("--- 10. 类型先验分分布 ---")
    holder_types = events_df.groupby("holder_name_std")["holder_type"].first()
    prior_scores = holder_types.map(map_holder_prior_score)
    prior_dist = prior_scores.value_counts().sort_index(ascending=False)
    for score, cnt in prior_dist.items():
        types_with_score = holder_types[prior_scores == score].unique()[:3]
        logger.info("  先验分 %.0f: %s 个股东 (类型: %s)", score, cnt, list(types_with_score))

    logger.info("")
    logger.info("--- 11. 抽样画像计算验证 (>=5只股票的股东) ---")
    qualified = holder_stock_cnt[holder_stock_cnt >= 5].index
    qualified_events = events_df[events_df["holder_name_std"].isin(qualified)]
    logger.info("  符合条件的股东: %s 个, 事件: %s 条", len(qualified), len(qualified_events))

    if len(qualified_events) > 0:
        sample_holders = list(qualified)[:20]
        sample_events = qualified_events[qualified_events["holder_name_std"].isin(sample_holders)]

        portrait_rows = []
        for holder in sample_holders:
            h_events = sample_events[sample_events["holder_name_std"] == holder]
            entry_events_h = h_events[h_events["action_type"] == "entry"]
            add_events_h = h_events[h_events["action_type"] == "add"]

            entry_excess_60 = pd.to_numeric(entry_events_h["future_excess_ret_60"], errors="coerce").mean()
            add_excess_60 = pd.to_numeric(add_events_h["future_excess_ret_60"], errors="coerce").mean()
            entry_wr_60 = (pd.to_numeric(entry_events_h["future_excess_ret_60"], errors="coerce") > 0).mean() if len(entry_events_h) > 0 else np.nan
            add_wr_60 = (pd.to_numeric(add_events_h["future_excess_ret_60"], errors="coerce") > 0).mean() if len(add_events_h) > 0 else np.nan
            avg_mdd = pd.to_numeric(h_events["future_mdd_60"], errors="coerce").mean()
            avg_tenure = pd.to_numeric(h_events["tenure_curr"], errors="coerce").mean()
            stock_cnt = h_events["ts_code"].nunique()
            h_type = h_events["holder_type"].iloc[0] if len(h_events) > 0 else None
            prior = map_holder_prior_score(h_type)

            portrait_rows.append({
                "holder_name_std": holder,
                "holder_type": h_type,
                "sample_stocks": stock_cnt,
                "sample_entry": len(entry_events_h),
                "sample_add": len(add_events_h),
                "entry_excess_ret_60": entry_excess_60,
                "add_excess_ret_60": add_excess_60,
                "entry_win_rate_60": entry_wr_60,
                "add_win_rate_60": add_wr_60,
                "avg_mdd_60": avg_mdd,
                "avg_tenure": avg_tenure,
                "prior_score": prior,
            })

        portrait_df = pd.DataFrame(portrait_rows)
        logger.info("  抽样画像 (%s 个股东):", len(portrait_df))
        for col in ["entry_excess_ret_60", "add_excess_ret_60", "avg_mdd_60", "avg_tenure"]:
            vals = portrait_df[col].dropna()
            if len(vals) > 0:
                logger.info("    %s: mean=%.4f, median=%.4f, min=%.4f, max=%.4f",
                            col, vals.mean(), vals.median(), vals.min(), vals.max())

        logger.info("")
        logger.info("  贝叶斯收缩效果验证:")
        n_i = portrait_df["sample_entry"].fillna(0) + portrait_df["sample_add"].fillna(0)
        post_w = np.where(n_i < 3, 0.10, np.where(n_i < 5, 0.25, np.where(n_i < 10, 0.50, np.where(n_i < 20, 0.75, 0.90))))
        portrait_df["shrinkage_weight"] = post_w
        portrait_df["composite_raw"] = 50 + portrait_df["entry_excess_ret_60"].fillna(0) * 100
        portrait_df["composite_score"] = post_w * portrait_df["composite_raw"] + (1 - post_w) * portrait_df["prior_score"]
        for _, row in portrait_df.head(10).iterrows():
            logger.info("    %s (n=%d, w=%.2f): raw=%.1f, prior=%.1f, shrunk=%.1f",
                        row["holder_name_std"][:20], int(n_i.iloc[0] if _ < len(n_i) else 0),
                        row["shrinkage_weight"], row["composite_raw"], row["prior_score"], row["composite_score"])

    if csv:
        output_path = os.path.join(os.path.dirname(__file__), "holder_portrait_explore_result.csv")
        if len(qualified_events) > 0:
            summary = qualified_events.groupby("holder_name_std").agg({
                "ts_code": "nunique",
                "action_type": "count",
                "future_excess_ret_60": lambda x: pd.to_numeric(x, errors="coerce").mean(),
                "future_mdd_60": lambda x: pd.to_numeric(x, errors="coerce").mean(),
                "tenure_curr": lambda x: pd.to_numeric(x, errors="coerce").mean(),
            }).rename(columns={
                "ts_code": "stock_count",
                "action_type": "event_count",
                "future_excess_ret_60": "avg_excess_ret_60",
                "future_mdd_60": "avg_mdd_60",
                "tenure_curr": "avg_tenure",
            })
            summary.to_csv(output_path)
            logger.info("探索结果已保存: %s (%s 行)", output_path, len(summary))

    logger.info("")
    logger.info("=" * 60)
    logger.info("探索完成")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="股东投资质量画像模型 - 数据探索")
    parser.add_argument("--limit", type=int, default=100, help="限制股票数量")
    parser.add_argument("--csv", action="store_true", help="保存探索结果到 CSV")
    args = parser.parse_args()
    explore_data(limit=args.limit, csv=args.csv)


if __name__ == "__main__":
    main()
