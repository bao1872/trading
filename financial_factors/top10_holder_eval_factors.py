# -*- coding: utf-8 -*-
"""
前十大流通股东评价体系指标计算脚本（Tushare）

Purpose:
    1. 从数据库读取 stock_top10_holders_tushare 表中的历史前十大流通股东数据
    2. 使用 Tushare Pro 获取个股与基准指数历史行情（含 turnover_rate）
    3. 计算股东画像后验质量、个股结构分 / 稳定性分 / 质量分 / 耦合分 / 风险扣分
    4. 写入数据库表:
       - stock_top10_holder_profiles_tushare
       - stock_top10_holder_eval_scores_tushare

Inputs:
    - 数据库表: stock_top10_holders_tushare
      关键字段来自当前 tushare fetcher:
        ts_code, stock_name, report_date, ann_date, holder_rank,
        holder_name, holder_type, hold_amount, hold_ratio,
        hold_float_ratio, hold_change
    - 行情数据源: Tushare Pro (tushare_data.fetcher.fetch_market_data)
      默认使用字段: open/high/low/close/vol/turnover/turnover_rate

Outputs:
    - 数据库表: stock_top10_holder_profiles_tushare
    - 数据库表: stock_top10_holder_eval_scores_tushare

How to Run:
    # 批量
    python financial_factors/top10_holder_eval_factors.py --mode batch

    # 批量测试
    python financial_factors/top10_holder_eval_factors.py --mode batch --limit 50 --lookback_years 5

    # 单只股票
    python financial_factors/top10_holder_eval_factors.py --mode single --ts_code 600519.SH

    # 只预览不写库
    python financial_factors/top10_holder_eval_factors.py --mode single --ts_code 000001.SZ --dry_run

Examples:
    python financial_factors/top10_holder_eval_factors.py --mode batch --limit 20
    python financial_factors/top10_holder_eval_factors.py --mode single --ts_code 300750.SZ --lookback_years 6

Side Effects:
    - 写入 stock_top10_holder_profiles_tushare
    - 写入 stock_top10_holder_eval_scores_tushare
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import bulk_upsert, get_session, query_df
from tushare_data.fetcher import fetch_market_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

INPUT_TABLE = "stock_top10_holders_tushare"
PROFILE_TABLE = "stock_top10_holder_profiles_tushare"
SCORE_TABLE = "stock_top10_holder_eval_scores_tushare"
PROFILE_UNIQUE_KEYS = ["holder_name_std"]
SCORE_UNIQUE_KEYS = ["ts_code", "report_date"]
EPS = 1e-12
DEFAULT_BENCH = "000300.SH"
DEFAULT_INDEX_MAP = {
    "主板": "000300.SH",
    "创业板": "399006.SZ",
    "科创板": "000688.SH",
    "北交所": "399673.SZ",
}
DEFAULT_INDUSTRY_FILE = "stock_concepts_cache.xlsx"


@dataclass
class EventMarketMetrics:
    effective_date: pd.Timestamp
    future_ret_20: Optional[float]
    future_ret_60: Optional[float]
    future_ret_120: Optional[float]
    future_excess_ret_20: Optional[float]
    future_excess_ret_60: Optional[float]
    future_excess_ret_120: Optional[float]
    future_mdd_60: Optional[float]
    ret_120: Optional[float]
    mom_20: Optional[float]
    mom_60: Optional[float]
    turnover_rate_mean_20: Optional[float]
    turnover_value_mean_20: Optional[float]
    vol_20: Optional[float]
    drawdown_from_high_120: Optional[float]


def normalize_ts_code(ts_code: str) -> str:
    ts_code = str(ts_code).strip().upper()
    if ts_code.endswith((".SZ", ".SH", ".BJ")):
        return ts_code
    if ts_code.startswith("6"):
        return f"{ts_code}.SH"
    if ts_code.startswith(("8", "4")):
        return f"{ts_code}.BJ"
    return f"{ts_code}.SZ"


def code_only(ts_code: str) -> str:
    return normalize_ts_code(ts_code).split(".")[0]


def normalize_holder_name(name: object) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    s = str(name).strip()
    replacements = {
        "（": "(",
        "）": ")",
        "股份有限公司": "股份",
        "有限合伙企业": "有限合伙",
        "有限合伙": "有限合伙",
        "－": "-",
        "—": "-",
        "\u3000": "",
        " ": "",
        "“": "",
        "”": "",
        "‘": "",
        "’": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s.upper()


def safe_float(v: object) -> Optional[float]:
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


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


def robust_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std(ddof=0)
    if pd.isna(std) or std < EPS:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mean) / std



def group_robust_zscore(values: pd.Series, groups: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    groups = groups.astype(str).fillna("未知行业")
    global_z = robust_zscore(values)
    out = pd.Series(index=values.index, dtype=float)
    for g, idx in groups.groupby(groups).groups.items():
        sub = values.loc[idx]
        if sub.notna().sum() < 3:
            out.loc[idx] = global_z.loc[idx]
        else:
            out.loc[idx] = robust_zscore(sub)
    return out.fillna(global_z).fillna(0.0)


def rank01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return s.rank(pct=True)


def clip_score(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def parse_date_column(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series.astype(str).str[:8], format="%Y%m%d", errors="coerce")
    return s


def ensure_output_tables() -> None:
    create_profile_sql = f"""
    CREATE TABLE IF NOT EXISTS {PROFILE_TABLE} (
        holder_name_std TEXT PRIMARY KEY,
        holder_type TEXT,
        sample_total INTEGER,
        sample_entry INTEGER,
        sample_add INTEGER,
        sample_reduce INTEGER,
        sample_hold INTEGER,
        avg_excess_ret_20_entry REAL,
        avg_excess_ret_60_entry REAL,
        avg_excess_ret_120_entry REAL,
        avg_excess_ret_20_add REAL,
        avg_excess_ret_60_add REAL,
        avg_excess_ret_120_add REAL,
        win_rate_60_entry REAL,
        win_rate_60_add REAL,
        avg_mdd_60_entry REAL,
        avg_mdd_60_add REAL,
        posterior_score_raw REAL,
        posterior_score_shrink REAL,
        scenefit_lowpos REAL,
        scenefit_smallcap REAL,
        scenefit_growth REAL,
        scenefit_industry_l2 REAL,
        final_holder_quality REAL,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """

    create_score_sql = f"""
    CREATE TABLE IF NOT EXISTS {SCORE_TABLE} (
        ts_code TEXT NOT NULL,
        stock_name TEXT,
        industry_l2 TEXT,
        industry_l3 TEXT,
        market_cap REAL,
        industry_pe REAL,
        report_date TEXT NOT NULL,
        ann_date TEXT,
        effective_date TEXT,
        cr3 REAL,
        cr5 REAL,
        cr10 REAL,
        delta_cr3 REAL,
        delta_cr5 REAL,
        delta_cr10 REAL,
        top1_ratio REAL,
        top3_inner_ratio REAL,
        hhi REAL,
        delta_hhi REAL,
        overlap_ratio REAL,
        entry_count INTEGER,
        exit_count INTEGER,
        add_count INTEGER,
        reduce_count INTEGER,
        add_ratio REAL,
        core_add_ratio REAL,
        entry_weight REAL,
        exit_weight REAL,
        net_entry_weight REAL,
        turnover_struct REAL,
        avg_tenure REAL,
        top3_stable_flag INTEGER,
        head_tail_divergence_flag INTEGER,
        head_tail_support_flag INTEGER,
        top10_weighted_quality REAL,
        add_weighted_quality REAL,
        entry_weighted_quality REAL,
        quality_net_add REAL,
        future_ret_20 REAL,
        future_ret_60 REAL,
        future_ret_120 REAL,
        future_excess_ret_20 REAL,
        future_excess_ret_60 REAL,
        future_excess_ret_120 REAL,
        future_mdd_60 REAL,
        ret_120 REAL,
        ret_120_rank REAL,
        mom_20 REAL,
        mom_60 REAL,
        turnover_rate_mean_20 REAL,
        turnover_rate_rank_20 REAL,
        turnover_value_mean_20 REAL,
        vol_20 REAL,
        drawdown_from_high_120 REAL,
        lowpos_accu REAL,
        lock_in REAL,
        trend_follow REAL,
        highpos_crowded REAL,
        score_change REAL,
        score_stability REAL,
        score_quality REAL,
        score_interaction REAL,
        score_risk REAL,
        score_change_neutral REAL,
        score_stability_neutral REAL,
        score_quality_neutral REAL,
        score_interaction_neutral REAL,
        total_score REAL,
        total_score_industry_neutral REAL,
        label_primary TEXT,
        label_secondary TEXT,
        label_score_consistent INTEGER,
        label_score_consistency_note TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (ts_code, report_date)
    )
    """

    with get_session() as session:
        session.execute(text(create_profile_sql))
        session.execute(text(create_score_sql))
        session.commit()


def get_stock_pool(limit: Optional[int] = None) -> List[Tuple[str, Optional[str]]]:
    with get_session() as session:
        df = query_df(session, "stock_pools", columns=["ts_code", "name"])
    if df.empty:
        return []
    df = df.drop_duplicates(subset=["ts_code"])
    result = list(zip(df["ts_code"].astype(str).map(normalize_ts_code), df["name"].tolist()))
    if limit:
        result = result[:limit]
    return result


def load_top10_data(ts_codes: Optional[Sequence[str]] = None, lookback_years: int = 6) -> pd.DataFrame:
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



def load_industry_info(industry_file: Optional[str] = None) -> pd.DataFrame:
    cols = ["ts_code", "name", "market_cap", "total_market_cap", "industry_l2", "industry_l3", "industry_pe"]
    df = pd.DataFrame()
    file_path = industry_file or DEFAULT_INDUSTRY_FILE
    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            logger.warning("读取行业文件失败 %s: %s", file_path, e)
    if df.empty:
        try:
            with get_session() as session:
                df = query_df(session, "stock_pools", columns=cols)
        except Exception as e:
            logger.warning("读取 stock_pools 失败: %s", e)
            return pd.DataFrame(columns=cols)
    df = df[[c for c in cols if c in df.columns]].copy()
    if df.empty:
        return df
    df["ts_code"] = df["ts_code"].astype(str).map(normalize_ts_code)
    for c in ["market_cap", "total_market_cap", "industry_pe"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["industry_l2", "industry_l3"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": None})
    df = df.drop_duplicates(subset=["ts_code"], keep="last")
    if "industry_l2" in df.columns and len(df) > 0:
        coverage = df["industry_l2"].notna().mean()
        if coverage < 0.80:
            logger.warning("industry_l2 覆盖率较低: %.1f%%，行业超额收益与行业中性化可能退化", coverage * 100)
        else:
            logger.info("industry_l2 覆盖率: %.1f%%", coverage * 100)
    return df


def infer_effective_date(ann_date: pd.Timestamp, trading_index: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    if pd.isna(ann_date):
        return None
    pos = trading_index.searchsorted(ann_date)
    if pos >= len(trading_index):
        return None
    # 若公告日在交易日内，当天收盘后披露无法稳定判断，这里保守取下一交易日
    if trading_index[pos] == ann_date:
        pos += 1
    if pos >= len(trading_index):
        return None
    return pd.Timestamp(trading_index[pos])


def load_market_data(ts_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
                     bench_code: str = DEFAULT_BENCH, fqt: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start = (start_date - pd.Timedelta(days=250)).strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")
    stock_df = fetch_market_data(ts_code, start, end, fqt=fqt)
    bench_df = fetch_market_data(bench_code, start, end, fqt=fqt)
    return stock_df, bench_df



def build_industry_benchmarks(stock_df_map: Dict[str, pd.DataFrame], industry_df: pd.DataFrame,
                              industry_col: str = "industry_l2") -> Dict[str, pd.DataFrame]:
    if industry_df is None or industry_df.empty or industry_col not in industry_df.columns:
        return {}
    code_to_ind = industry_df.set_index("ts_code")[industry_col].to_dict()
    buckets: Dict[str, List[pd.Series]] = {}
    for ts_code, df in stock_df_map.items():
        ind = code_to_ind.get(ts_code)
        if not ind or df is None or df.empty or "close" not in df.columns:
            continue
        ret = pd.to_numeric(df["close"], errors="coerce").pct_change()
        if ret.dropna().empty:
            continue
        buckets.setdefault(str(ind), []).append(ret.rename(ts_code))
    result: Dict[str, pd.DataFrame] = {}
    for ind, series_list in buckets.items():
        panel = pd.concat(series_list, axis=1)
        if panel.empty:
            continue
        ew_ret = panel.mean(axis=1, skipna=True).fillna(0.0)
        close = (1.0 + ew_ret).cumprod()
        result[ind] = pd.DataFrame({"close": close})
    return result


def build_event_metrics_cache(df: pd.DataFrame, stock_df_map: Dict[str, pd.DataFrame],
                              bench_df_map: Dict[str, pd.DataFrame],
                              industry_bench_map: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[Tuple[str, pd.Timestamp], EventMarketMetrics]:
    cache: Dict[Tuple[str, pd.Timestamp], EventMarketMetrics] = {}
    for (ts_code, report_date), snap in df.groupby(["ts_code", "report_date"]):
        stock_df = stock_df_map.get(ts_code)
        if stock_df is None or stock_df.empty:
            continue
        ann_date = snap["ann_date"].dropna().min()
        if pd.isna(ann_date):
            continue
        effective_date = infer_effective_date(pd.Timestamp(ann_date), stock_df.index)
        if effective_date is None:
            continue
        industry_l2 = snap["industry_l2"].dropna().iloc[0] if "industry_l2" in snap.columns and snap["industry_l2"].notna().any() else None
        bench_df = None
        if industry_bench_map and industry_l2 in industry_bench_map:
            bench_df = industry_bench_map.get(industry_l2)
        if bench_df is None or bench_df.empty:
            bench_df = bench_df_map.get(ts_code, pd.DataFrame())
        cache[(ts_code, pd.Timestamp(report_date))] = calc_event_market_metrics(stock_df, bench_df, effective_date)
    return cache


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


def calc_drawdown_from_high(prices: pd.Series, idx: int, lookback: int) -> Optional[float]:
    start_idx = idx - lookback + 1
    if start_idx < 0 or idx >= len(prices):
        return None
    window = prices.iloc[start_idx:(idx + 1)]
    if window.empty:
        return None
    high = window.max()
    cur = prices.iloc[idx]
    if pd.isna(high) or pd.isna(cur) or abs(high) < EPS:
        return None
    return float(cur / high - 1.0)


def calc_event_market_metrics(stock_df: pd.DataFrame, bench_df: pd.DataFrame,
                              effective_date: pd.Timestamp) -> EventMarketMetrics:
    close = stock_df["close"].dropna()
    bench_close = bench_df["close"].dropna() if not bench_df.empty and "close" in bench_df.columns else pd.Series(dtype=float)
    trading_index = close.index
    pos = trading_index.searchsorted(effective_date)
    if pos >= len(trading_index):
        pos = len(trading_index) - 1
    if pos < 0:
        pos = 0

    future_ret_20 = calc_future_return(close, pos, 20)
    future_ret_60 = calc_future_return(close, pos, 60)
    future_ret_120 = calc_future_return(close, pos, 120)

    bench_pos = bench_close.index.searchsorted(effective_date) if len(bench_close) > 0 else -1
    if len(bench_close) > 0 and bench_pos >= len(bench_close):
        bench_pos = len(bench_close) - 1
    bench_ret_20 = calc_future_return(bench_close, bench_pos, 20) if bench_pos >= 0 else None
    bench_ret_60 = calc_future_return(bench_close, bench_pos, 60) if bench_pos >= 0 else None
    bench_ret_120 = calc_future_return(bench_close, bench_pos, 120) if bench_pos >= 0 else None

    future_excess_ret_20 = None if future_ret_20 is None or bench_ret_20 is None else future_ret_20 - bench_ret_20
    future_excess_ret_60 = None if future_ret_60 is None or bench_ret_60 is None else future_ret_60 - bench_ret_60
    future_excess_ret_120 = None if future_ret_120 is None or bench_ret_120 is None else future_ret_120 - bench_ret_120

    future_mdd_60 = calc_max_drawdown_forward(close, pos, 60)
    ret_120 = calc_backward_return(close, pos, 120)
    mom_20 = calc_backward_return(close, pos, 20)
    mom_60 = calc_backward_return(close, pos, 60)

    if "turnover_rate" in stock_df.columns and pos >= 19:
        turnover_rate_mean_20 = float(pd.to_numeric(stock_df["turnover_rate"].iloc[pos - 19: pos + 1], errors="coerce").mean())
    else:
        turnover_rate_mean_20 = None
    if "turnover" in stock_df.columns and pos >= 19:
        turnover_value_mean_20 = float(pd.to_numeric(stock_df["turnover"].iloc[pos - 19: pos + 1], errors="coerce").mean())
    else:
        turnover_value_mean_20 = None

    if pos >= 19:
        rets20 = close.pct_change().iloc[pos - 19: pos + 1]
        vol_20 = float(rets20.std(ddof=0)) if rets20.notna().sum() >= 5 else None
    else:
        vol_20 = None

    drawdown_from_high_120 = calc_drawdown_from_high(close, pos, 120)

    return EventMarketMetrics(
        effective_date=effective_date,
        future_ret_20=future_ret_20,
        future_ret_60=future_ret_60,
        future_ret_120=future_ret_120,
        future_excess_ret_20=future_excess_ret_20,
        future_excess_ret_60=future_excess_ret_60,
        future_excess_ret_120=future_excess_ret_120,
        future_mdd_60=future_mdd_60,
        ret_120=ret_120,
        mom_20=mom_20,
        mom_60=mom_60,
        turnover_rate_mean_20=turnover_rate_mean_20,
        turnover_value_mean_20=turnover_value_mean_20,
        vol_20=vol_20,
        drawdown_from_high_120=drawdown_from_high_120,
    )


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
    return pd.concat(tenure_parts, ignore_index=False).sort_values(["ts_code", "report_date", "holder_rank", "holder_name_std"]).reset_index(drop=True)



def build_change_events(df: pd.DataFrame, event_cache: Dict[Tuple[str, pd.Timestamp], EventMarketMetrics],
                        delta_threshold: float = 0.01) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    by_stock = df.groupby("ts_code")

    for ts_code, g in by_stock:
        reports = sorted(g["report_date"].dropna().unique())
        prev_snap: Optional[pd.DataFrame] = None
        prev_report_date: Optional[pd.Timestamp] = None

        for report_date in reports:
            snap = g[g["report_date"] == report_date].copy().sort_values(["holder_rank", "holder_name_std"])
            ann_date = snap["ann_date"].dropna().min()
            metrics = event_cache.get((ts_code, pd.Timestamp(report_date)))
            if pd.isna(ann_date) or metrics is None:
                continue

            cur_map = {row["holder_name_std"]: row for _, row in snap.iterrows()}
            prev_map = {} if prev_snap is None else {row["holder_name_std"]: row for _, row in prev_snap.iterrows()}
            names = sorted(set(cur_map.keys()) | set(prev_map.keys()))

            for name in names:
                cur = cur_map.get(name)
                prev = prev_map.get(name)
                cur_ratio = safe_float(cur["hold_float_ratio"]) if cur is not None else None
                prev_ratio = safe_float(prev["hold_float_ratio"]) if prev is not None else None
                cur_amt = safe_float(cur["hold_amount"]) if cur is not None else None
                prev_amt = safe_float(prev["hold_amount"]) if prev is not None else None
                delta_ratio = None if cur_ratio is None or prev_ratio is None else cur_ratio - prev_ratio
                delta_amt = None if cur_amt is None or prev_amt is None else cur_amt - prev_amt

                if cur is not None and prev is None:
                    action_type = "entry"
                elif cur is None and prev is not None:
                    action_type = "exit"
                else:
                    if delta_ratio is None:
                        action_type = "hold"
                    elif delta_ratio > delta_threshold:
                        action_type = "add"
                    elif delta_ratio < -delta_threshold:
                        action_type = "reduce"
                    else:
                        action_type = "hold"

                base_row = cur if cur is not None else prev
                records.append({
                    "ts_code": ts_code,
                    "stock_name": base_row.get("stock_name"),
                    "industry_l2": base_row.get("industry_l2"),
                    "industry_l3": base_row.get("industry_l3"),
                    "market_cap": safe_float(base_row.get("market_cap")),
                    "industry_pe": safe_float(base_row.get("industry_pe")),
                    "report_date": report_date,
                    "prev_report_date": prev_report_date,
                    "ann_date": ann_date,
                    "effective_date": metrics.effective_date,
                    "holder_name_std": name,
                    "holder_type": base_row.get("holder_type"),
                    "holder_rank_curr": int(cur["holder_rank"]) if cur is not None and pd.notna(cur["holder_rank"]) else None,
                    "holder_rank_prev": int(prev["holder_rank"]) if prev is not None and pd.notna(prev["holder_rank"]) else None,
                    "hold_float_ratio_curr": cur_ratio,
                    "hold_float_ratio_prev": prev_ratio,
                    "delta_hold_float_ratio": delta_ratio,
                    "hold_amount_curr": cur_amt,
                    "hold_amount_prev": prev_amt,
                    "delta_hold_amount": delta_amt,
                    "action_type": action_type,
                    "is_core_curr": int(cur is not None and pd.notna(cur["holder_rank"]) and int(cur["holder_rank"]) <= 5),
                    "is_core_prev": int(prev is not None and pd.notna(prev["holder_rank"]) and int(prev["holder_rank"]) <= 5),
                    "tenure_curr": int(cur["tenure"]) if cur is not None and pd.notna(cur["tenure"]) else 0,
                    "future_ret_20": metrics.future_ret_20,
                    "future_ret_60": metrics.future_ret_60,
                    "future_ret_120": metrics.future_ret_120,
                    "future_excess_ret_20": metrics.future_excess_ret_20,
                    "future_excess_ret_60": metrics.future_excess_ret_60,
                    "future_excess_ret_120": metrics.future_excess_ret_120,
                    "future_mdd_60": metrics.future_mdd_60,
                    "ret_120": metrics.ret_120,
                    "mom_20": metrics.mom_20,
                    "mom_60": metrics.mom_60,
                    "turnover_rate_mean_20": metrics.turnover_rate_mean_20,
                    "turnover_value_mean_20": metrics.turnover_value_mean_20,
                    "vol_20": metrics.vol_20,
                    "drawdown_from_high_120": metrics.drawdown_from_high_120,
                })

            prev_snap = snap
            prev_report_date = report_date

    return pd.DataFrame(records)


def load_existing_data(stock_codes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with get_session() as session:
        scores_df = query_df(session, SCORE_TABLE)
    with get_session() as session:
        profiles_df = query_df(session, PROFILE_TABLE)
    if stock_codes:
        existing_scores = scores_df[scores_df["ts_code"].isin(stock_codes)].copy()
    else:
        existing_scores = scores_df.copy()
    return existing_scores, profiles_df


def merge_profiles(new_profiles: pd.DataFrame, existing_profiles: pd.DataFrame) -> pd.DataFrame:
    if existing_profiles.empty or new_profiles.empty:
        return new_profiles

    count_cols = ["sample_entry", "sample_add", "sample_reduce", "sample_hold", "sample_total"]
    mean_cols = [
        "avg_excess_ret_20_entry", "avg_excess_ret_60_entry", "avg_excess_ret_120_entry",
        "avg_excess_ret_20_add", "avg_excess_ret_60_add", "avg_excess_ret_120_add",
        "win_rate_60_entry", "win_rate_60_add",
        "avg_mdd_60_entry", "avg_mdd_60_add",
    ]

    merged = new_profiles.copy()
    for _, row in existing_profiles.iterrows():
        name = row["holder_name_std"]
        match = merged[merged["holder_name_std"] == name]
        if match.empty:
            merged = pd.concat([merged, row.to_frame().T], ignore_index=True)
            continue
        idx = match.index[0]
        n_new = int(row["sample_total"]) if "sample_total" in row and pd.notna(row["sample_total"]) else 0
        n_cur = int(merged.at[idx, "sample_total"])
        total = n_new + n_cur
        if total == 0:
            continue
        for col in mean_cols:
            if col in row.index and pd.notna(row[col]) and col in merged.columns:
                cur_val = merged.at[idx, col] if pd.notna(merged.at[idx, col]) else 0.0
                merged.at[idx, col] = (n_cur * cur_val + n_new * float(row[col])) / total
        for col in count_cols:
            if col in row.index and col in merged.columns:
                merged.at[idx, col] = int(merged.at[idx, col]) + int(row[col]) if pd.notna(row[col]) else int(merged.at[idx, col])
    return merged


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


def _grouped_agg(events: pd.DataFrame, action_type: str, cols: List[str]) -> pd.DataFrame:
    df = events[events["action_type"] == action_type]
    if df.empty:
        return pd.DataFrame()
    agg_dict = {c: _safe_mean for c in cols}
    agg_dict["future_excess_ret_60"] = _safe_mean
    result = df.groupby("holder_name_std")[cols].agg(_safe_mean)
    result["win_rate_60"] = df.groupby("holder_name_std")["future_excess_ret_60"].apply(_safe_win_rate)
    return result


def build_holder_profiles(events: pd.DataFrame, existing_profiles: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """基于传入事件全量重算股东画像。

    注意：这里显式不再把 existing_profiles 伪造成 dummy events 回灌，
    避免增量模式下旧样本被重复累计并污染画像统计。
    """
    if events.empty:
        return pd.DataFrame()

    events = events.copy()
    for col in ["future_excess_ret_20", "future_excess_ret_60", "future_excess_ret_120",
                "future_mdd_60", "ret_120", "turnover_value_mean_20", "mom_60"]:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce")

    holder_meta = events.groupby("holder_name_std").agg({
        "holder_type": lambda x: x.mode().iloc[0] if len(x.dropna()) > 0 else None,
        "action_type": "count",
    }).rename(columns={"action_type": "sample_total"})

    agg_cols = ["future_excess_ret_20", "future_excess_ret_60", "future_excess_ret_120", "future_mdd_60"]
    entry_agg = events[events["action_type"] == "entry"].groupby("holder_name_std").agg({
        c: _safe_mean for c in agg_cols
    })
    entry_agg.columns = [f"{c}_entry" for c in entry_agg.columns]
    entry_agg["win_rate_60_entry"] = events[events["action_type"] == "entry"].groupby("holder_name_std")["future_excess_ret_60"].apply(_safe_win_rate)

    add_agg = events[events["action_type"] == "add"].groupby("holder_name_std").agg({
        c: _safe_mean for c in agg_cols
    })
    add_agg.columns = [f"{c}_add" for c in add_agg.columns]
    add_agg["win_rate_60_add"] = events[events["action_type"] == "add"].groupby("holder_name_std")["future_excess_ret_60"].apply(_safe_win_rate)

    sample_counts = events.groupby(["holder_name_std", "action_type"]).size().unstack(fill_value=0)
    for action in ["entry", "add", "reduce", "hold"]:
        if action not in sample_counts.columns:
            sample_counts[action] = 0
    sample_counts = sample_counts[["entry", "add", "reduce", "hold"]].rename(columns={
        "entry": "sample_entry", "add": "sample_add", "reduce": "sample_reduce", "hold": "sample_hold"
    })

    type_mu = {}
    for holder_type, gt in events.groupby("holder_type"):
        add_s = gt[gt["action_type"] == "add"]["future_excess_ret_60"]
        entry_s = gt[gt["action_type"] == "entry"]["future_excess_ret_60"]
        add_wr = _safe_win_rate(gt[gt["action_type"] == "add"]["future_excess_ret_60"])
        entry_wr = _safe_win_rate(gt[gt["action_type"] == "entry"]["future_excess_ret_60"])
        mu = (
            45 * (_safe_mean(add_s) or 0.0)
            + 25 * (_safe_mean(entry_s) or 0.0)
            + 20 * ((add_wr or 0.5) - 0.5)
            + 10 * ((entry_wr or 0.5) - 0.5)
            - 20 * abs(_safe_mean(gt[gt["action_type"] == "add"]["future_mdd_60"]) or 0.0)
        )
        type_mu[str(holder_type)] = 50 + mu * 100

    profiles_df = holder_meta.join(entry_agg, how="left").join(add_agg, how="left").join(sample_counts, how="left")
    profiles_df = profiles_df.reset_index()

    for col in ["sample_total", "sample_entry", "sample_add", "sample_reduce", "sample_hold"]:
        profiles_df[col] = profiles_df[col].fillna(0).astype(int)

    prior_scores = profiles_df["holder_type"].map(map_holder_prior_score)
    profiles_df["prior_score"] = prior_scores

    posterior_raw = (
        45 * profiles_df["future_excess_ret_60_add"].fillna(0)
        + 25 * profiles_df["future_excess_ret_60_entry"].fillna(0)
        + 20 * ((profiles_df["win_rate_60_add"].fillna(0.5) - 0.5))
        + 10 * ((profiles_df["win_rate_60_entry"].fillna(0.5) - 0.5))
        - 20 * abs(profiles_df["future_mdd_60_add"].fillna(0))
    )
    profiles_df["posterior_score_raw"] = 50 + posterior_raw * 100

    profiles_df["type_mu"] = profiles_df["holder_type"].astype(str).map(type_mu).fillna(50.0)
    n_i = profiles_df["sample_entry"].fillna(0) + profiles_df["sample_add"].fillna(0)
    k = 10.0
    profiles_df["posterior_score_shrink"] = (n_i / (n_i + k)) * profiles_df["posterior_score_raw"] + (k / (n_i + k)) * profiles_df["type_mu"]

    ret_120_median = events.groupby("holder_name_std")["ret_120"].median()
    turnover_median = events.groupby("holder_name_std")["turnover_value_mean_20"].median()

    def _scenefit_for_holder(g: pd.DataFrame) -> Dict[str, float]:
        holder = g.name
        ret_med = ret_120_median.get(holder, np.nan)
        turnover_med = turnover_median.get(holder, np.nan)

        lowpos_mask = g["ret_120"].notna() & (g["ret_120"] <= ret_med)
        smallcap_mask = g["turnover_value_mean_20"].notna() & (g["turnover_value_mean_20"] <= turnover_med)
        growth_mask = g["mom_60"].notna() & (g["mom_60"] > 0)

        scenefit_lowpos = 50 + (_safe_mean(g.loc[lowpos_mask, "future_excess_ret_60"]) or 0.0) * 100
        scenefit_smallcap = 50 + (_safe_mean(g.loc[smallcap_mask, "future_excess_ret_60"]) or 0.0) * 100
        scenefit_growth = 50 + (_safe_mean(g.loc[growth_mask, "future_excess_ret_60"]) or 0.0) * 100

        industry_means = g.dropna(subset=["industry_l2"]).groupby("industry_l2")["future_excess_ret_60"].mean()
        scenefit_industry = 50 + (float(industry_means.mean()) * 100 if len(industry_means) > 0 else 0.0)

        return {
            "holder_name_std": holder,
            "scenefit_lowpos": scenefit_lowpos,
            "scenefit_smallcap": scenefit_smallcap,
            "scenefit_growth": scenefit_growth,
            "scenefit_industry_l2": scenefit_industry,
        }

    scenefit_list = events.groupby("holder_name_std").apply(_scenefit_for_holder).tolist()
    scenefit_df = pd.DataFrame(scenefit_list).set_index("holder_name_std")
    profiles_df = profiles_df.join(scenefit_df, on="holder_name_std", how="left")

    profiles_df["scenefit_score"] = (
        0.3 * profiles_df["scenefit_lowpos"].fillna(50) +
        0.2 * profiles_df["scenefit_smallcap"].fillna(50) +
        0.2 * profiles_df["scenefit_growth"].fillna(50) +
        0.3 * profiles_df["scenefit_industry_l2"].fillna(50)
    )

    final_holder_quality = np.where(
        n_i < 5,
        0.60 * profiles_df["prior_score"] + 0.30 * profiles_df["posterior_score_shrink"] + 0.10 * profiles_df["scenefit_score"],
        0.35 * profiles_df["prior_score"] + 0.45 * profiles_df["posterior_score_shrink"] + 0.20 * profiles_df["scenefit_score"],
    )
    profiles_df["final_holder_quality"] = final_holder_quality

    result_cols = [
        "holder_name_std", "holder_type", "sample_total", "sample_entry", "sample_add",
        "sample_reduce", "sample_hold", "future_excess_ret_20_entry", "future_excess_ret_60_entry",
        "future_excess_ret_120_entry", "future_excess_ret_20_add", "future_excess_ret_60_add",
        "future_excess_ret_120_add", "win_rate_60_entry", "win_rate_60_add",
        "avg_mdd_60_entry", "avg_mdd_60_add", "posterior_score_raw", "posterior_score_shrink",
        "scenefit_lowpos", "scenefit_smallcap", "scenefit_growth", "scenefit_industry_l2",
        "final_holder_quality",
    ]
    rename_cols = {
        "future_excess_ret_20_entry": "avg_excess_ret_20_entry",
        "future_excess_ret_60_entry": "avg_excess_ret_60_entry",
        "future_excess_ret_120_entry": "avg_excess_ret_120_entry",
        "future_excess_ret_20_add": "avg_excess_ret_20_add",
        "future_excess_ret_60_add": "avg_excess_ret_60_add",
        "future_excess_ret_120_add": "avg_excess_ret_120_add",
        "future_mdd_60_entry": "avg_mdd_60_entry",
        "future_mdd_60_add": "avg_mdd_60_add",
    }
    profiles_df = profiles_df.rename(columns=rename_cols)
    result_cols = [rename_cols.get(c, c) for c in result_cols]
    result_cols = [c for c in result_cols if c in profiles_df.columns]
    return profiles_df[result_cols].copy()



def build_holder_industry_scores(events: pd.DataFrame, profiles: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    if events.empty or profiles.empty or "industry_l2" not in events.columns:
        return {}
    overall = profiles.set_index("holder_name_std")["final_holder_quality"].to_dict()
    result: Dict[Tuple[str, str], float] = {}
    for (holder_name_std, industry_l2), g in events.dropna(subset=["industry_l2"]).groupby(["holder_name_std", "industry_l2"]):
        n = len(g)
        mean_excess = pd.to_numeric(g["future_excess_ret_60"], errors="coerce").mean()
        if pd.isna(mean_excess):
            continue
        raw = 50 + float(mean_excess) * 100
        base = overall.get(holder_name_std, 50.0)
        w = min(max((n - 2) / 8.0, 0.0), 0.35)
        result[(holder_name_std, str(industry_l2))] = (1 - w) * base + w * raw
    return result


def format_report_key(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.notna(ts):
        return pd.Timestamp(ts).strftime("%Y%m%d")
    return str(value)


def filter_events_for_profile_asof(events: pd.DataFrame, effective_date: pd.Timestamp, min_realized_days: int = 90) -> pd.DataFrame:
    """仅保留在当前时点之前、且后验观察窗口基本已走完的历史事件。

    这里使用保守的 90 个自然日近似 60 个交易日，避免把尚未在当时真正可见的
    future_excess_ret_60 / future_mdd_60 提前泄露到更早的评分时点。
    """
    if events.empty or pd.isna(effective_date):
        return pd.DataFrame(columns=events.columns)
    if "effective_date" not in events.columns:
        return events.iloc[0:0].copy()
    cutoff = pd.Timestamp(effective_date) - pd.Timedelta(days=min_realized_days)
    eff = pd.to_datetime(events["effective_date"], errors="coerce")
    mask = eff.notna() & (eff <= cutoff)
    return events.loc[mask].copy()


def build_quality_maps_asof(events: pd.DataFrame, effective_date: pd.Timestamp) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    hist_events = filter_events_for_profile_asof(events, effective_date)
    if hist_events.empty:
        return {}, {}
    hist_profiles = build_holder_profiles(hist_events)
    profile_map = hist_profiles.set_index("holder_name_std")["final_holder_quality"].to_dict() if not hist_profiles.empty else {}
    industry_map = build_holder_industry_scores(hist_events, hist_profiles) if not hist_profiles.empty else {}
    return profile_map, industry_map


def compute_stock_scores(df: pd.DataFrame, events: pd.DataFrame, stock_df_map: Dict[str, pd.DataFrame],
                         event_cache: Dict[Tuple[str, pd.Timestamp], EventMarketMetrics],
                         existing_scores: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    existing_keys: Set[Tuple[str, str]] = set()
    if existing_scores is not None and not existing_scores.empty:
        existing_keys = set(zip(
            existing_scores["ts_code"].astype(str),
            existing_scores["report_date"].map(format_report_key),
        ))
    quality_cache: Dict[str, Tuple[Dict[str, float], Dict[Tuple[str, str], float]]] = {}

    for ts_code, g in df.groupby("ts_code"):
        stock_df = stock_df_map.get(ts_code)
        if stock_df is None or stock_df.empty:
            continue
        reports = sorted(g["report_date"].dropna().unique())
        prev_snap: Optional[pd.DataFrame] = None

        for report_date in reports:
            snap = g[g["report_date"] == report_date].copy().sort_values(["holder_rank", "holder_name_std"])
            ann_date = snap["ann_date"].dropna().min()
            metrics = event_cache.get((ts_code, pd.Timestamp(report_date)))
            report_key = format_report_key(report_date)
            if (ts_code, report_key) in existing_keys:
                prev_snap = snap
                continue
            if pd.isna(ann_date) or metrics is None:
                continue

            current_industry = snap["industry_l2"].dropna().iloc[0] if "industry_l2" in snap.columns and snap["industry_l2"].notna().any() else None
            effective_key = metrics.effective_date.strftime("%Y%m%d")
            if effective_key not in quality_cache:
                quality_cache[effective_key] = build_quality_maps_asof(events, metrics.effective_date)
            profile_map, holder_industry_quality_map = quality_cache[effective_key]
            base_quality = snap["holder_name_std"].map(profile_map).fillna(snap["holder_type"].map(map_holder_prior_score))
            if current_industry:
                ind_quality = snap["holder_name_std"].map(lambda x: holder_industry_quality_map.get((x, str(current_industry)), np.nan))
                snap["holder_quality"] = np.where(pd.notna(ind_quality), 0.7 * base_quality + 0.3 * pd.to_numeric(ind_quality, errors="coerce"), base_quality)
            else:
                snap["holder_quality"] = base_quality

            cr3 = float(pd.to_numeric(snap.loc[snap["holder_rank"] <= 3, "hold_float_ratio"], errors="coerce").sum())
            cr5 = float(pd.to_numeric(snap.loc[snap["holder_rank"] <= 5, "hold_float_ratio"], errors="coerce").sum())
            cr10 = float(pd.to_numeric(snap["hold_float_ratio"], errors="coerce").sum())
            top1_ratio = safe_float(snap.loc[snap["holder_rank"] == 1, "hold_float_ratio"].iloc[0]) if (snap["holder_rank"] == 1).any() else None
            hhi = float(np.nansum(np.square(pd.to_numeric(snap["hold_float_ratio"], errors="coerce"))))
            top3_inner_ratio = None if cr10 is None or abs(cr10) < EPS else cr3 / cr10
            top10_weighted_quality = None if abs(cr10) < EPS else float(np.nansum(pd.to_numeric(snap["hold_float_ratio"], errors="coerce") * pd.to_numeric(snap["holder_quality"], errors="coerce")) / cr10)

            delta_cr3 = delta_cr5 = delta_cr10 = delta_hhi = overlap_ratio = add_ratio = None
            entry_count = exit_count = add_count = reduce_count = 0
            core_add_ratio = entry_weight = exit_weight = net_entry_weight = turnover_struct = avg_tenure = None
            top3_stable_flag = 0
            head_tail_divergence_flag = 0
            head_tail_support_flag = 0
            add_weighted_quality = entry_weighted_quality = quality_net_add = None

            if prev_snap is not None:
                prev_snap = prev_snap.copy()
                prev_base_quality = prev_snap["holder_name_std"].map(profile_map).fillna(prev_snap["holder_type"].map(map_holder_prior_score))
                if current_industry:
                    prev_ind_quality = prev_snap["holder_name_std"].map(lambda x: holder_industry_quality_map.get((x, str(current_industry)), np.nan))
                    prev_snap["holder_quality"] = np.where(pd.notna(prev_ind_quality), 0.7 * prev_base_quality + 0.3 * pd.to_numeric(prev_ind_quality, errors="coerce"), prev_base_quality)
                else:
                    prev_snap["holder_quality"] = prev_base_quality
                prev_cr3 = float(pd.to_numeric(prev_snap.loc[prev_snap["holder_rank"] <= 3, "hold_float_ratio"], errors="coerce").sum())
                prev_cr5 = float(pd.to_numeric(prev_snap.loc[prev_snap["holder_rank"] <= 5, "hold_float_ratio"], errors="coerce").sum())
                prev_cr10 = float(pd.to_numeric(prev_snap["hold_float_ratio"], errors="coerce").sum())
                prev_hhi = float(np.nansum(np.square(pd.to_numeric(prev_snap["hold_float_ratio"], errors="coerce"))))
                delta_cr3 = cr3 - prev_cr3
                delta_cr5 = cr5 - prev_cr5
                delta_cr10 = cr10 - prev_cr10
                delta_hhi = hhi - prev_hhi

                cur_map = {r["holder_name_std"]: r for _, r in snap.iterrows()}
                prev_map = {r["holder_name_std"]: r for _, r in prev_snap.iterrows()}
                overlap_names = set(cur_map.keys()) & set(prev_map.keys())
                overlap_ratio = len(overlap_names) / max(1, min(len(cur_map), len(prev_map)))

                entry_names = set(cur_map.keys()) - set(prev_map.keys())
                exit_names = set(prev_map.keys()) - set(cur_map.keys())
                entry_count = len(entry_names)
                exit_count = len(exit_names)
                entry_weight = float(sum((safe_float(cur_map[n]["hold_float_ratio"]) or 0.0) for n in entry_names))
                exit_weight = float(sum((safe_float(prev_map[n]["hold_float_ratio"]) or 0.0) for n in exit_names))
                net_entry_weight = entry_weight - exit_weight

                add_quality_numer = 0.0
                add_quality_denom = 0.0
                quality_net_add_val = 0.0
                turnover_struct_val = entry_weight + exit_weight
                core_add_ratio_val = 0.0
                head_net = 0.0
                tail_net = 0.0

                for name in overlap_names:
                    cur_row = cur_map[name]
                    prev_row = prev_map[name]
                    cur_ratio = safe_float(cur_row["hold_float_ratio"]) or 0.0
                    prev_ratio = safe_float(prev_row["hold_float_ratio"]) or 0.0
                    delta = cur_ratio - prev_ratio
                    turnover_struct_val += abs(delta)
                    if delta > 0.01:
                        add_count += 1
                        q = safe_float(cur_row["holder_quality"]) or 50.0
                        add_quality_numer += delta * q
                        add_quality_denom += delta
                    elif delta < -0.01:
                        reduce_count += 1

                    rank_cur = int(cur_row["holder_rank"])
                    rank_prev = int(prev_row["holder_rank"])
                    if rank_cur <= 5 or rank_prev <= 5:
                        core_add_ratio_val += delta
                    if rank_prev <= 5:
                        head_net += delta
                    else:
                        tail_net += delta
                    q = safe_float(cur_row["holder_quality"]) or 50.0
                    if q >= 70:
                        quality_net_add_val += delta

                add_ratio = add_count / max(1, len(overlap_names))
                turnover_struct = turnover_struct_val
                core_add_ratio = core_add_ratio_val
                add_weighted_quality = None if add_quality_denom < EPS else add_quality_numer / add_quality_denom
                if entry_weight and entry_weight > EPS:
                    entry_weighted_quality = float(sum((safe_float(cur_map[n]["hold_float_ratio"]) or 0.0) * (safe_float(cur_map[n]["holder_quality"]) or 50.0) for n in entry_names) / entry_weight)
                quality_net_add = quality_net_add_val
                avg_tenure = float(pd.to_numeric(snap["tenure"], errors="coerce").mean()) if "tenure" in snap.columns else None
                cur_top3 = set(snap.loc[snap["holder_rank"] <= 3, "holder_name_std"])
                prev_top3 = set(prev_snap.loc[prev_snap["holder_rank"] <= 3, "holder_name_std"])
                top3_stable_flag = int(cur_top3 == prev_top3 and len(cur_top3) == 3)
                head_tail_divergence_flag = int(head_net < 0 and tail_net > 0)
                head_tail_support_flag = int(head_net > 0 and tail_net < 0)
            else:
                avg_tenure = float(pd.to_numeric(snap["tenure"], errors="coerce").mean()) if "tenure" in snap.columns else None

            records.append({
                "ts_code": ts_code,
                "stock_name": snap["stock_name"].iloc[0] if "stock_name" in snap.columns else None,
                "industry_l2": snap["industry_l2"].iloc[0] if "industry_l2" in snap.columns else None,
                "industry_l3": snap["industry_l3"].iloc[0] if "industry_l3" in snap.columns else None,
                "market_cap": safe_float(snap["market_cap"].iloc[0]) if "market_cap" in snap.columns else None,
                "industry_pe": safe_float(snap["industry_pe"].iloc[0]) if "industry_pe" in snap.columns else None,
                "report_date": report_date.strftime("%Y%m%d") if isinstance(report_date, pd.Timestamp) else str(report_date),
                "ann_date": ann_date.strftime("%Y%m%d") if isinstance(ann_date, pd.Timestamp) else str(ann_date),
                "effective_date": metrics.effective_date.strftime("%Y%m%d"),
                "cr3": cr3,
                "cr5": cr5,
                "cr10": cr10,
                "delta_cr3": delta_cr3,
                "delta_cr5": delta_cr5,
                "delta_cr10": delta_cr10,
                "top1_ratio": top1_ratio,
                "top3_inner_ratio": top3_inner_ratio,
                "hhi": hhi,
                "delta_hhi": delta_hhi,
                "overlap_ratio": overlap_ratio,
                "entry_count": entry_count,
                "exit_count": exit_count,
                "add_count": add_count,
                "reduce_count": reduce_count,
                "add_ratio": add_ratio,
                "core_add_ratio": core_add_ratio,
                "entry_weight": entry_weight,
                "exit_weight": exit_weight,
                "net_entry_weight": net_entry_weight,
                "turnover_struct": turnover_struct,
                "avg_tenure": avg_tenure,
                "top3_stable_flag": top3_stable_flag,
                "head_tail_divergence_flag": head_tail_divergence_flag,
                "head_tail_support_flag": head_tail_support_flag,
                "top10_weighted_quality": top10_weighted_quality,
                "add_weighted_quality": add_weighted_quality,
                "entry_weighted_quality": entry_weighted_quality,
                "quality_net_add": quality_net_add,
                "future_ret_20": metrics.future_ret_20,
                "future_ret_60": metrics.future_ret_60,
                "future_ret_120": metrics.future_ret_120,
                "future_excess_ret_20": metrics.future_excess_ret_20,
                "future_excess_ret_60": metrics.future_excess_ret_60,
                "future_excess_ret_120": metrics.future_excess_ret_120,
                "future_mdd_60": metrics.future_mdd_60,
                "ret_120": metrics.ret_120,
                "mom_20": metrics.mom_20,
                "mom_60": metrics.mom_60,
                "turnover_rate_mean_20": metrics.turnover_rate_mean_20,
                "turnover_value_mean_20": metrics.turnover_value_mean_20,
                "vol_20": metrics.vol_20,
                "drawdown_from_high_120": metrics.drawdown_from_high_120,
            })
            prev_snap = snap

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result["ret_120_rank"] = rank01(result["ret_120"])
    result["turnover_rate_rank_20"] = rank01(result["turnover_rate_mean_20"])
    result["lowpos_accu"] = pd.to_numeric(result["delta_cr10"], errors="coerce") * (1 - pd.to_numeric(result["ret_120_rank"], errors="coerce"))
    result["lock_in"] = pd.to_numeric(result["delta_cr10"], errors="coerce") * (1 - pd.to_numeric(result["turnover_rate_rank_20"], errors="coerce"))
    result["trend_follow"] = pd.to_numeric(result["core_add_ratio"], errors="coerce") * pd.to_numeric(result["mom_60"], errors="coerce")
    result["highpos_crowded"] = pd.to_numeric(result["hhi"], errors="coerce") * pd.to_numeric(result["ret_120_rank"], errors="coerce")

    score_change_raw = (
        0.33 * robust_zscore(result["delta_cr10"]) +
        0.23 * robust_zscore(result["delta_cr5"]) +
        0.18 * robust_zscore(result["core_add_ratio"]) +
        0.10 * robust_zscore(result["net_entry_weight"]) +
        0.08 * robust_zscore(result["add_ratio"]) +
        0.08 * pd.to_numeric(result["head_tail_support_flag"], errors="coerce").fillna(0) -
        0.08 * pd.to_numeric(result["head_tail_divergence_flag"], errors="coerce").fillna(0)
    )
    score_stability_raw = (
        0.35 * robust_zscore(result["overlap_ratio"]) +
        0.25 * robust_zscore(result["avg_tenure"]) -
        0.25 * robust_zscore(result["turnover_struct"]) +
        0.15 * pd.to_numeric(result["top3_stable_flag"], errors="coerce").fillna(0)
    )
    score_quality_raw = (
        0.33 * robust_zscore(result["top10_weighted_quality"]) +
        0.28 * robust_zscore(result["add_weighted_quality"]) +
        0.19 * robust_zscore(result["entry_weighted_quality"]) +
        0.12 * robust_zscore(result["quality_net_add"]) +
        0.08 * pd.to_numeric(result["head_tail_support_flag"], errors="coerce").fillna(0)
    )
    score_interaction_raw = (
        0.40 * robust_zscore(result["lowpos_accu"]) +
        0.35 * robust_zscore(result["lock_in"]) +
        0.25 * robust_zscore(result["trend_follow"])
    )

    industry_key = result["industry_l2"].fillna("未知行业") if "industry_l2" in result.columns else pd.Series(["全市场"] * len(result), index=result.index)
    result["score_change_neutral"] = clip_score(12.5 + 5 * group_robust_zscore(score_change_raw, industry_key), 0, 25)
    result["score_stability_neutral"] = clip_score(10 + 4 * group_robust_zscore(score_stability_raw, industry_key), 0, 20)
    result["score_quality_neutral"] = clip_score(15 + 6 * group_robust_zscore(score_quality_raw, industry_key), 0, 30)
    result["score_interaction_neutral"] = clip_score(7.5 + 3 * group_robust_zscore(score_interaction_raw, industry_key), 0, 15)

    result["score_change"] = clip_score(12.5 + 5 * score_change_raw, 0, 25)
    result["score_stability"] = clip_score(10 + 4 * score_stability_raw, 0, 20)
    result["score_quality"] = clip_score(15 + 6 * score_quality_raw, 0, 30)
    result["score_interaction"] = clip_score(7.5 + 3 * score_interaction_raw, 0, 15)

    risk = pd.Series(np.zeros(len(result)), index=result.index, dtype=float)
    risk -= ((rank01(result["highpos_crowded"]) >= 0.85).astype(float) * 2.0)
    risk -= (((pd.to_numeric(result["quality_net_add"], errors="coerce") < 0) & (rank01(result["quality_net_add"].fillna(0).abs()) >= 0.70)).astype(float) * 3.0)
    risk -= (pd.to_numeric(result["head_tail_divergence_flag"], errors="coerce").fillna(0) * 2.0)
    risk += (pd.to_numeric(result["head_tail_support_flag"], errors="coerce").fillna(0) * 1.0)
    risk -= (((pd.to_numeric(result["overlap_ratio"], errors="coerce") <= 0.4).astype(float)) * 2.0)
    risk -= (((pd.to_numeric(result["entry_weighted_quality"], errors="coerce") < 45).astype(float)) * 1.0)
    result["score_risk"] = risk.clip(lower=-10, upper=1)

    result["total_score"] = (
        pd.to_numeric(result["score_change"], errors="coerce").fillna(0)
        + pd.to_numeric(result["score_stability"], errors="coerce").fillna(0)
        + pd.to_numeric(result["score_quality"], errors="coerce").fillna(0)
        + pd.to_numeric(result["score_interaction"], errors="coerce").fillna(0)
        + pd.to_numeric(result["score_risk"], errors="coerce").fillna(0)
    ).clip(lower=0, upper=100)
    result["total_score_industry_neutral"] = (
        pd.to_numeric(result["score_change_neutral"], errors="coerce").fillna(0)
        + pd.to_numeric(result["score_stability_neutral"], errors="coerce").fillna(0)
        + pd.to_numeric(result["score_quality_neutral"], errors="coerce").fillna(0)
        + pd.to_numeric(result["score_interaction_neutral"], errors="coerce").fillna(0)
        + pd.to_numeric(result["score_risk"], errors="coerce").fillna(0)
    ).clip(lower=0, upper=100)

    cond_core_exit = (
        (pd.to_numeric(result["core_add_ratio"], errors="coerce") < 0)
        & (pd.to_numeric(result["delta_cr5"], errors="coerce") < 0)
        & (pd.to_numeric(result["quality_net_add"], errors="coerce") < 0)
        & (pd.to_numeric(result["score_change_neutral"], errors="coerce") <= 8)
    )
    cond_high_crowded = (
        (pd.to_numeric(result["ret_120_rank"], errors="coerce") >= 0.8)
        & (rank01(result["highpos_crowded"]) >= 0.8)
        & (pd.to_numeric(result["score_risk"], errors="coerce") <= -4)
    )
    cond_core_add = (
        (pd.to_numeric(result["score_change_neutral"], errors="coerce") >= 18)
        & (pd.to_numeric(result["core_add_ratio"], errors="coerce") > 0)
        & (pd.to_numeric(result["delta_cr5"], errors="coerce") > 0)
        & (pd.to_numeric(result["add_ratio"], errors="coerce") >= 0.6)
        & (pd.to_numeric(result["quality_net_add"], errors="coerce") > 0)
    )
    cond_lock = (
        (pd.to_numeric(result["score_stability_neutral"], errors="coerce") >= 15)
        & (pd.to_numeric(result["score_quality_neutral"], errors="coerce") >= 20)
        & (pd.to_numeric(result["overlap_ratio"], errors="coerce") >= 0.7)
        & (pd.to_numeric(result["quality_net_add"], errors="coerce") >= 0)
    )
    cond_lowpos = (
        (pd.to_numeric(result["score_interaction_neutral"], errors="coerce") >= 10)
        & (pd.to_numeric(result["ret_120_rank"], errors="coerce") <= 0.4)
        & (rank01(result["lowpos_accu"]) >= 0.7)
        & (pd.to_numeric(result["score_quality_neutral"], errors="coerce") >= 15)
    )
    cond_trading = (
        (pd.to_numeric(result["overlap_ratio"], errors="coerce") <= 0.5)
        & ((pd.to_numeric(result["entry_count"], errors="coerce").fillna(0) + pd.to_numeric(result["exit_count"], errors="coerce").fillna(0)) >= 4)
        & (rank01(result["turnover_struct"]) >= 0.7)
        & (pd.to_numeric(result["score_stability_neutral"], errors="coerce") <= 8)
    )

    result["label_primary"] = "中性观察"
    result.loc[cond_core_exit, "label_primary"] = "核心资金撤退型"
    result.loc[~cond_core_exit & cond_high_crowded, "label_primary"] = "高位抱团拥挤型"
    result.loc[~cond_core_exit & ~cond_high_crowded & cond_core_add, "label_primary"] = "核心资金持续加仓型"
    result.loc[~cond_core_exit & ~cond_high_crowded & ~cond_core_add & cond_lock, "label_primary"] = "长线资金锁仓型"
    result.loc[~cond_core_exit & ~cond_high_crowded & ~cond_core_add & ~cond_lock & cond_lowpos, "label_primary"] = "低位收筹待启动型"
    result.loc[~cond_core_exit & ~cond_high_crowded & ~cond_core_add & ~cond_lock & ~cond_lowpos & cond_trading, "label_primary"] = "短线资金换手博弈型"

    result["label_secondary"] = np.where(
        pd.to_numeric(result["score_quality_neutral"], errors="coerce") >= 22,
        "高质量资金主导",
        np.where(pd.to_numeric(result["score_risk"], errors="coerce") <= -4, "风险偏高", "一般")
    )

    total_for_check = pd.to_numeric(result["total_score_industry_neutral"], errors="coerce").fillna(pd.to_numeric(result["total_score"], errors="coerce").fillna(0))
    negative_labels = {"核心资金撤退型", "高位抱团拥挤型"}
    positive_labels = {"核心资金持续加仓型", "长线资金锁仓型", "低位收筹待启动型"}
    result["label_score_consistent"] = 1
    result["label_score_consistency_note"] = ""
    result.loc[(total_for_check >= 70) & result["label_primary"].isin(negative_labels), "label_score_consistent"] = 0
    result.loc[(total_for_check >= 70) & result["label_primary"].isin(negative_labels), "label_score_consistency_note"] = "高分但标签偏空"
    result.loc[(total_for_check <= 35) & result["label_primary"].isin(positive_labels), "label_score_consistent"] = 0
    result.loc[(total_for_check <= 35) & result["label_primary"].isin(positive_labels), "label_score_consistency_note"] = "低分但标签偏多"

    return result


def save_profiles(df: pd.DataFrame, dry_run: bool = False) -> int:
    if df.empty:
        return 0
    if dry_run:
        logger.info("[DRY RUN] profiles rows=%s", len(df))
        return len(df)
    with get_session() as session:
        return bulk_upsert(session, PROFILE_TABLE, df, PROFILE_UNIQUE_KEYS)


def save_scores(df: pd.DataFrame, dry_run: bool = False) -> int:
    if df.empty:
        return 0
    if dry_run:
        logger.info("[DRY RUN] scores rows=%s", len(df))
        return len(df)
    with get_session() as session:
        return bulk_upsert(session, SCORE_TABLE, df, SCORE_UNIQUE_KEYS)


def process_batch(limit: Optional[int], lookback_years: int, dry_run: bool,
                  bench_code: str, fqt: int, industry_file: Optional[str],
                  incremental: bool = True) -> None:
    pool = get_stock_pool(limit)
    if not pool:
        logger.warning("未找到股票池")
        return

    ts_codes = [x[0] for x in pool]
    top10_df = load_top10_data(ts_codes=ts_codes, lookback_years=lookback_years)
    if top10_df.empty:
        logger.warning("%s 表为空或无有效样本", INPUT_TABLE)
        return
    industry_df = load_industry_info(industry_file)
    if not industry_df.empty:
        top10_df = top10_df.merge(industry_df.drop(columns=["name"], errors="ignore"), on="ts_code", how="left")
    top10_df = build_holder_tenure(top10_df)

    stock_df_map: Dict[str, pd.DataFrame] = {}
    bench_df_map: Dict[str, pd.DataFrame] = {}

    unique_codes = top10_df["ts_code"].drop_duplicates().tolist()
    for ts_code in tqdm(unique_codes, desc="加载行情数据", unit="股票"):
        g = top10_df[top10_df["ts_code"] == ts_code]
        start_date = min(g["ann_date"].dropna().min(), g["report_date"].dropna().min())
        end_date = pd.Timestamp.today().normalize()
        try:
            stock_df, bench_df = load_market_data(ts_code, start_date, end_date, bench_code=bench_code, fqt=fqt)
            if stock_df.empty:
                logger.warning("%s Tushare 行情为空，跳过", ts_code)
                continue
            stock_df_map[ts_code] = stock_df
            bench_df_map[ts_code] = bench_df
        except Exception as e:
            logger.exception("加载 %s 行情失败: %s", ts_code, e)

    valid_codes = [c for c in top10_df["ts_code"].drop_duplicates() if c in stock_df_map]
    top10_df = top10_df[top10_df["ts_code"].isin(valid_codes)].copy()
    if top10_df.empty:
        logger.warning("无可计算样本")
        return

    industry_bench_map = build_industry_benchmarks(stock_df_map, industry_df, "industry_l2") if not industry_df.empty else {}
    event_cache = build_event_metrics_cache(top10_df, stock_df_map, bench_df_map, industry_bench_map)

    existing_scores: Optional[pd.DataFrame] = None
    if incremental and not dry_run:
        stock_codes = top10_df["ts_code"].unique().tolist()
        existing_scores, _ = load_existing_data(stock_codes)
        logger.info("增量模式：已有 scores %d 行", len(existing_scores))

    events_df = build_change_events(top10_df.copy(), event_cache)
    profiles_df = build_holder_profiles(events_df) if not events_df.empty else pd.DataFrame()
    scores_df = compute_stock_scores(top10_df, events_df, stock_df_map, event_cache, existing_scores=existing_scores)

    n1 = save_profiles(profiles_df, dry_run=dry_run)
    n2 = save_scores(scores_df, dry_run=dry_run)
    logger.info("完成：profiles=%s, scores=%s", n1, n2)


def process_single(ts_code: str, lookback_years: int, dry_run: bool,
                   bench_code: str, fqt: int, industry_file: Optional[str],
                   incremental: bool = True) -> None:
    ts_code = normalize_ts_code(ts_code)
    top10_df = load_top10_data(ts_codes=[ts_code], lookback_years=lookback_years)
    if top10_df.empty:
        logger.warning("%s 无前十大流通股东数据", ts_code)
        return
    industry_df = load_industry_info(industry_file)
    if not industry_df.empty:
        top10_df = top10_df.merge(industry_df.drop(columns=["name"], errors="ignore"), on="ts_code", how="left")
    top10_df = build_holder_tenure(top10_df)
    start_date = min(top10_df["ann_date"].dropna().min(), top10_df["report_date"].dropna().min())
    end_date = pd.Timestamp.today().normalize()
    stock_df, bench_df = load_market_data(ts_code, start_date, end_date, bench_code=bench_code, fqt=fqt)
    if stock_df.empty:
        logger.warning("%s 无可用行情数据", ts_code)
        return
    event_cache = build_event_metrics_cache(top10_df, {ts_code: stock_df}, {ts_code: bench_df}, {})

    existing_scores: Optional[pd.DataFrame] = None
    if incremental and not dry_run:
        existing_scores, _ = load_existing_data([ts_code])
        logger.info("增量模式：已有 scores %d 行", len(existing_scores))

    events_df = build_change_events(top10_df.copy(), event_cache)
    profiles_df = build_holder_profiles(events_df) if not events_df.empty else pd.DataFrame()
    scores_df = compute_stock_scores(top10_df, events_df, {ts_code: stock_df}, event_cache, existing_scores=existing_scores)
    n1 = save_profiles(profiles_df, dry_run=dry_run)
    n2 = save_scores(scores_df, dry_run=dry_run)
    logger.info("%s 完成：profiles=%s, scores=%s", ts_code, n1, n2)
    if dry_run and not scores_df.empty:
        preview_cols = [
            "ts_code", "industry_l2", "report_date", "ann_date", "effective_date", "total_score",
            "total_score_industry_neutral", "score_change_neutral", "score_quality_neutral",
            "score_risk", "label_primary", "label_secondary", "label_score_consistent"
        ]
        print(scores_df[preview_cols].tail(8).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="前十大流通股东评价体系指标计算（Tushare）")
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    parser.add_argument("--ts_code", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lookback_years", type=int, default=6)
    parser.add_argument("--bench_code", type=str, default=DEFAULT_BENCH, help="Tushare 指数代码，如 000300.SH / 000905.SH / 399001.SZ")
    parser.add_argument("--fqt", type=int, default=0, help="复权类型：0 不复权")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--industry_file", type=str, default=DEFAULT_INDUSTRY_FILE, help="行业分类文件路径，默认 stock_concepts_cache.xlsx")
    parser.add_argument("--incremental", action="store_true", help="增量模式：已有数据跳过，新数据写入")
    parser.add_argument("--full", action="store_true", help="全量模式：清除历史后重新计算")
    args = parser.parse_args()

    incremental = args.incremental and not args.full

    ensure_output_tables()

    if args.mode == "single":
        if not args.ts_code:
            parser.error("single 模式下必须提供 --ts_code")
        process_single(
            ts_code=args.ts_code,
            lookback_years=args.lookback_years,
            dry_run=args.dry_run,
            bench_code=args.bench_code,
            fqt=args.fqt,
            industry_file=args.industry_file,
            incremental=incremental,
        )
    else:
        process_batch(
            limit=args.limit,
            lookback_years=args.lookback_years,
            dry_run=args.dry_run,
            bench_code=args.bench_code,
            fqt=args.fqt,
            industry_file=args.industry_file,
            incremental=incremental,
        )


if __name__ == "__main__":
    main()
