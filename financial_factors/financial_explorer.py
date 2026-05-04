#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    基于季度财报原始字段，构建“公告后 60 个交易日收益”的财务因子预测力探索脚本。
    脚本直接从 financial_quarterly_data 和 stock_k_data 读取数据，围绕季度报告事件做
    横截面 IC、分组收益、稳定性拆解，并输出为单个 Excel 多 sheet 文件。

核心设计：
    1. 只研究季度报告事件，不使用 total_score / 维度分。
    2. 只看公告后 60 个交易日绝对收益。
    3. 以 fetcher.py 获取到的原始字段为起点，构建基础字段体检层 + 经济含义派生层。
    4. 强制输出单个 Excel，多 sheet，便于后续人工复盘与探索调整。

Inputs:
    - 数据库表: financial_quarterly_data
    - 数据库表: stock_k_data (freq='d')
    - 可选数据库表: stock_pools (若存在，将尝试读取 name / stock_name)

Outputs:
    - Excel: out/financial_explorer.xlsx
      包含样本概览、因子汇总、报告期 IC 明细、分组收益、财报类型拆解、年份拆解、相关性等多个 sheet

How to Run:
    python financial_explorer.py
    python financial_explorer.py --start 20160101 --out out/financial_explorer.xlsx
    python financial_explorer.py --limit 500 --min-period-n 30
    python financial_explorer.py --test-stock-limit 50 --out out/financial_explorer_test50.xlsx
    python financial_explorer.py --test-stock-limit 50 --out out/financial_explorer_test50.xlsx

Examples:
    python financial_explorer.py --start 20160101 --out out/financial_explorer.xlsx
    python financial_explorer.py --limit 200 --out /tmp/financial_explorer_test.xlsx
    python financial_explorer.py --test-stock-limit 50 --out /tmp/financial_explorer_test50.xlsx

Side Effects:
    - 不写数据库
    - 写出一个 Excel 文件
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

try:
    from sqlalchemy import create_engine, text
except Exception:  # pragma: no cover
    create_engine = None
    text = None

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

try:
    from config import DATABASE_URL  # type: ignore
except Exception:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCE_TABLE = "financial_quarterly_data"
DAILY_TABLE = "stock_k_data"
STOCK_POOL_TABLE = "stock_pools"
RETURN_HORIZON = 60
EPSILON = 1e-9
DEFAULT_OUT = os.path.join("out", "financial_explorer.xlsx")

CORE_RAW_FIELDS = [
    "revenue", "oper_cost", "operate_profit", "total_profit", "n_income",
    "n_income_attr_p", "net_after_nr_lp_correct", "ebit", "ebitda", "rd_exp",
    "n_cashflow_act", "free_cashflow", "c_fr_sale_sg", "c_pay_acq_const_fiolta",
    "total_assets", "accounts_receiv", "inventories", "accounts_pay",
    "contract_liab", "total_hldr_eqy_exc_min_int", "basic_eps", "diluted_eps",
]

SUPPLEMENT_RAW_FIELDS = [
    "sell_exp", "admin_exp", "fin_exp", "oth_income", "asset_disp_income",
    "credit_impa_loss", "assets_impair_loss", "non_oper_income", "non_oper_exp",
]

FINANCIAL_FIELDS = [
    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", "end_type",
] + CORE_RAW_FIELDS + SUPPLEMENT_RAW_FIELDS

FACTOR_GROUPS: Dict[str, str] = {}
FACTOR_LABELS: Dict[str, str] = {}


def register_factor(name: str, label: str, group: str) -> None:
    FACTOR_LABELS[name] = label
    FACTOR_GROUPS[name] = group


# 原始字段体检层：原值 / YoY / QoQ / 标准化版本
FIELD_LABELS = {
    "revenue": "营收",
    "operate_profit": "营业利润",
    "n_income_attr_p": "归母净利润",
    "net_after_nr_lp_correct": "扣非归母净利润",
    "ebit": "EBIT",
    "n_cashflow_act": "经营现金流",
    "free_cashflow": "自由现金流",
    "total_assets": "总资产",
    "accounts_receiv": "应收账款",
    "inventories": "存货",
    "accounts_pay": "应付账款",
    "contract_liab": "合同负债",
    "rd_exp": "研发费用",
}

RAW_FIELD_EXPLORER_FIELDS = [
    "revenue", "operate_profit", "n_income_attr_p", "net_after_nr_lp_correct", "ebit",
    "n_cashflow_act", "free_cashflow", "total_assets", "accounts_receiv",
    "inventories", "accounts_pay", "contract_liab", "rd_exp",
]

for base in RAW_FIELD_EXPLORER_FIELDS:
    base_label = FIELD_LABELS[base]
    register_factor(base, f"原值_{base_label}", "原始字段体检")
    register_factor(f"{base}_yoy", f"同比_{base_label}", "原始字段体检")
    register_factor(f"{base}_qoq", f"环比_{base_label}", "原始字段体检")

register_factor("revenue_to_assets", "营收/总资产", "原始字段体检")
register_factor("operate_profit_to_assets", "营业利润/总资产", "原始字段体检")
register_factor("np_parent_to_assets", "归母净利/总资产", "原始字段体检")
register_factor("nr_np_to_assets", "扣非归母净利/总资产", "原始字段体检")
register_factor("cfo_to_assets", "经营现金流/总资产", "原始字段体检")
register_factor("fcf_to_assets", "自由现金流/总资产", "原始字段体检")
register_factor("ar_to_rev", "应收/营收", "原始字段体检")
register_factor("inv_to_rev", "存货/营收", "原始字段体检")
register_factor("ap_to_rev", "应付/营收", "原始字段体检")
register_factor("contract_liab_to_rev", "合同负债/营收", "原始字段体检")
register_factor("rd_to_rev", "研发/营收", "原始字段体检")

# 经济含义派生层
for name, label, group in [
    ("revenue_yoy", "营收同比", "增长"),
    ("operate_profit_yoy", "营业利润同比", "增长"),
    ("n_income_attr_p_yoy", "归母净利同比", "增长"),
    ("net_after_nr_lp_correct_yoy", "扣非归母净利同比", "增长"),
    ("ebit_yoy", "EBIT同比", "增长"),
    ("n_cashflow_act_yoy", "经营现金流同比", "增长"),
    ("free_cashflow_yoy", "自由现金流同比", "增长"),
    ("gross_margin", "毛利率", "利润率水平"),
    ("op_margin", "营业利润率", "利润率水平"),
    ("np_parent_margin", "归母净利率", "利润率水平"),
    ("nr_np_margin", "扣非净利率", "利润率水平"),
    ("ebit_margin", "EBIT利润率", "利润率水平"),
    ("roa_parent", "归母ROA", "利润率水平"),
    ("nr_roa", "扣非ROA", "利润率水平"),
    ("rd_ratio", "研发强度", "利润率水平"),
    ("asset_turnover", "总资产周转率", "利润率水平"),
    ("cfo_to_np_parent", "经营现金流/归母净利", "现金质量"),
    ("cfo_to_nr_np", "经营现金流/扣非净利", "现金质量"),
    ("cfo_to_ebit", "经营现金流/EBIT", "现金质量"),
    ("fcf_to_np_parent", "自由现金流/归母净利", "现金质量"),
    ("fcf_to_nr_np", "自由现金流/扣非净利", "现金质量"),
    ("cash_sales_ratio", "销售收现比", "现金质量"),
    ("capex_to_cfo", "资本开支/经营现金流", "现金质量"),
    ("nr_np_qoq", "扣非净利环比", "扣非真实性"),
    ("nr_np_to_np_parent", "扣非净利/归母净利", "扣非真实性"),
    ("nr_gap_ratio", "非经常性损益依赖度", "扣非真实性"),
    ("ar_to_rev", "应收/营收", "营运资本"),
    ("inv_to_rev", "存货/营收", "营运资本"),
    ("ap_to_rev", "应付/营收", "营运资本"),
    ("contract_liab_to_rev", "合同负债/营收", "营运资本"),
    ("ccc_proxy", "简化CCC代理", "营运资本"),
    ("gross_margin_yoy_chg", "毛利率同比变化", "边际改善"),
    ("op_margin_yoy_chg", "营业利润率同比变化", "边际改善"),
    ("nr_np_margin_yoy_chg", "扣非净利率同比变化", "边际改善"),
    ("cfo_to_np_parent_yoy_chg", "经营现金流/归母净利同比变化", "边际改善"),
    ("cfo_to_nr_np_yoy_chg", "经营现金流/扣非净利同比变化", "边际改善"),
    ("contract_liab_to_rev_yoy_chg", "合同负债/营收同比变化", "边际改善"),
    ("rd_ratio_yoy_chg", "研发强度同比变化", "边际改善"),
]:
    register_factor(name, label, group)

ALL_FACTOR_COLS = list(FACTOR_LABELS.keys())


@dataclass
class SummaryThresholds:
    min_period_n: int = 30
    n_groups: int = 5
    winsor_quantile: float = 0.01


def create_engine_or_raise():
    if create_engine is None:
        raise RuntimeError("未安装 SQLAlchemy，无法连接数据库。")
    if not DATABASE_URL:
        raise RuntimeError("未找到 DATABASE_URL，请在 config.py 或环境变量中配置。")
    return create_engine(DATABASE_URL, pool_size=5, max_overflow=10, pool_recycle=3600)


def safe_div(numerator, denominator):
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = num / den
    out = out.where(den.abs() > EPSILON)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def yoy(series: pd.Series) -> pd.Series:
    return safe_div(series - series.shift(4), series.shift(4))


def qoq(series: pd.Series) -> pd.Series:
    return safe_div(series - series.shift(1), series.shift(1))


def winsorize_series(s: pd.Series, q: float) -> pd.Series:
    valid = s.dropna()
    if valid.empty:
        return s
    lower = valid.quantile(q)
    upper = valid.quantile(1 - q)
    return s.clip(lower=lower, upper=upper)


def robust_zscore(s: pd.Series) -> pd.Series:
    median = s.median(skipna=True)
    mad = (s - median).abs().median(skipna=True)
    if pd.isna(mad) or mad < EPSILON:
        std = s.std(skipna=True)
        if pd.isna(std) or std < EPSILON:
            return pd.Series(np.nan, index=s.index)
        return (s - s.mean(skipna=True)) / std
    return 0.6745 * (s - median) / mad


def pct_change_direction(values: Sequence[float]) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return "unknown"
    diffs = np.diff(arr)
    if np.all(diffs >= -1e-12):
        return "ascending"
    if np.all(diffs <= 1e-12):
        return "descending"
    return "non_monotonic"


def resolve_worker_count(requested: Optional[int]) -> int:
    if requested is None:
        return max(1, min(4, (os.cpu_count() or 1)))
    return max(1, int(requested))


def _preprocess_single_factor(raw_series: pd.Series, period_groups: Sequence[np.ndarray], winsor_q: float) -> Tuple[pd.Series, pd.Series]:
    raw_series = pd.to_numeric(raw_series, errors="coerce")
    xsec_series = pd.Series(np.nan, index=raw_series.index, dtype=float)
    for idx in period_groups:
        s = raw_series.iloc[idx]
        if s.notna().sum() < 5:
            continue
        s = winsorize_series(s, winsor_q)
        s = robust_zscore(s)
        xsec_series.iloc[idx] = s.to_numpy()
    return raw_series, xsec_series



def load_test_stock_codes(engine, start: str, stock_limit: int) -> List[str]:
    """测试模式：先确定前N只股票代码，后续只加载这些股票的数据，避免全量取数。"""
    sql = f'''
        SELECT DISTINCT ts_code
        FROM "{SOURCE_TABLE}"
        WHERE ann_date IS NOT NULL
          AND end_date >= :start
        ORDER BY ts_code
        LIMIT :limit
    '''
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"start": start, "limit": stock_limit})
    codes = sorted(df["ts_code"].dropna().astype(str).unique().tolist())
    logger.info("测试模式预选股票: %s 只", len(codes))
    return codes

def load_financial_data(engine, start: str, limit: Optional[int], ts_codes: Optional[List[str]] = None) -> pd.DataFrame:
    cols = ", ".join([f'"{c}"' for c in FINANCIAL_FIELDS])
    sql = f'''
        SELECT {cols}
        FROM "{SOURCE_TABLE}"
        WHERE ann_date IS NOT NULL
          AND end_date >= :start
    '''
    params = {"start": start}
    if ts_codes:
        sql += ' AND ts_code = ANY(:ts_codes)'
        params["ts_codes"] = list(ts_codes)
    if limit:
        sql += ' ORDER BY ts_code, end_date LIMIT :limit'
        params["limit"] = limit
    else:
        sql += ' ORDER BY ts_code, end_date'
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    if df.empty:
        return df
    for col in ["ann_date", "f_ann_date", "end_date"]:
        df[col] = pd.to_datetime(df[col].astype(str), format="%Y%m%d", errors="coerce")
    logger.info("加载财务数据: %s 条, %s 只股票", len(df), df["ts_code"].nunique())
    return df


def load_stock_names(engine) -> pd.DataFrame:
    if text is None:
        return pd.DataFrame(columns=["ts_code", "stock_name"])
    candidates = [
        f'SELECT ts_code, name AS stock_name FROM "{STOCK_POOL_TABLE}"',
        f'SELECT ts_code, stock_name FROM "{STOCK_POOL_TABLE}"',
    ]
    for sql in candidates:
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)
            if not df.empty:
                df = df.drop_duplicates("ts_code")
                return df
        except Exception:
            continue
    logger.warning("未能从 %s 读取股票名称，将仅使用 ts_code。", STOCK_POOL_TABLE)
    return pd.DataFrame(columns=["ts_code", "stock_name"])


def load_daily_prices(engine, ts_codes: Optional[List[str]] = None) -> pd.DataFrame:
    sql = f'''
        SELECT ts_code, bar_time, close
        FROM "{DAILY_TABLE}"
        WHERE freq = 'd'
    '''
    params = {}
    if ts_codes:
        sql += ' AND ts_code = ANY(:ts_codes)'
        params["ts_codes"] = list(ts_codes)
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params or None)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["bar_time"]).dt.normalize()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    logger.info("加载日线数据: %s 条, %s 只股票", len(df), df["ts_code"].nunique())
    return df[["ts_code", "trade_date", "close"]]



def prepare_base_financial_df(fin_df: pd.DataFrame, names_df: pd.DataFrame) -> pd.DataFrame:
    df = fin_df.copy()
    for col in CORE_RAW_FIELDS + SUPPLEMENT_RAW_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ts_code", "end_date", "ann_date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["ts_code", "end_date"], keep="last").copy()
    if not names_df.empty:
        df = df.merge(names_df, on="ts_code", how="left")
    else:
        df["stock_name"] = df["ts_code"]
    df["stock_name"] = df["stock_name"].fillna(df["ts_code"])
    df["report_date"] = df["end_date"]
    df["report_year"] = df["report_date"].dt.year
    df["report_quarter"] = df["report_date"].dt.quarter
    df["report_type_name"] = df["report_quarter"].map({1: "一季报", 2: "中报", 3: "三季报", 4: "年报"})
    return df


def add_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby("ts_code", group_keys=False)
    new_cols: Dict[str, pd.Series] = {}

    logger.info("构建因子：开始计算基础体检层与派生层...")

    # 基础体检层
    for col in tqdm(RAW_FIELD_EXPLORER_FIELDS, desc="构建基础体检因子", leave=False):
        new_cols[f"{col}_yoy"] = grp[col].transform(yoy)
        new_cols[f"{col}_qoq"] = grp[col].transform(qoq)

    # 原始字段标准化版本
    new_cols["revenue_to_assets"] = safe_div(df["revenue"], df["total_assets"])
    new_cols["operate_profit_to_assets"] = safe_div(df["operate_profit"], df["total_assets"])
    new_cols["np_parent_to_assets"] = safe_div(df["n_income_attr_p"], df["total_assets"])
    new_cols["nr_np_to_assets"] = safe_div(df["net_after_nr_lp_correct"], df["total_assets"])
    new_cols["cfo_to_assets"] = safe_div(df["n_cashflow_act"], df["total_assets"])
    new_cols["fcf_to_assets"] = safe_div(df["free_cashflow"], df["total_assets"])

    # 增长层
    growth_cols = [
        "operate_profit", "n_income_attr_p", "net_after_nr_lp_correct",
        "ebit", "n_cashflow_act", "free_cashflow",
    ]
    for col in tqdm(growth_cols, desc="构建增长因子", leave=False):
        new_cols[f"{col}_yoy"] = grp[col].transform(yoy)

    # 利润率 / 水平
    new_cols["gross_margin"] = safe_div(df["revenue"] - df["oper_cost"], df["revenue"])
    new_cols["op_margin"] = safe_div(df["operate_profit"], df["revenue"])
    new_cols["np_parent_margin"] = safe_div(df["n_income_attr_p"], df["revenue"])
    new_cols["nr_np_margin"] = safe_div(df["net_after_nr_lp_correct"], df["revenue"])
    new_cols["ebit_margin"] = safe_div(df["ebit"], df["revenue"])
    new_cols["roa_parent"] = safe_div(df["n_income_attr_p"], df["total_assets"])
    new_cols["nr_roa"] = safe_div(df["net_after_nr_lp_correct"], df["total_assets"])
    new_cols["rd_ratio"] = safe_div(df["rd_exp"], df["revenue"])
    new_cols["asset_turnover"] = safe_div(df["revenue"], df["total_assets"])

    # 现金质量
    new_cols["cfo_to_np_parent"] = safe_div(df["n_cashflow_act"], df["n_income_attr_p"])
    new_cols["cfo_to_nr_np"] = safe_div(df["n_cashflow_act"], df["net_after_nr_lp_correct"])
    new_cols["cfo_to_ebit"] = safe_div(df["n_cashflow_act"], df["ebit"])
    new_cols["fcf_to_np_parent"] = safe_div(df["free_cashflow"], df["n_income_attr_p"])
    new_cols["fcf_to_nr_np"] = safe_div(df["free_cashflow"], df["net_after_nr_lp_correct"])
    new_cols["cash_sales_ratio"] = safe_div(df["c_fr_sale_sg"], df["revenue"])
    new_cols["capex_to_cfo"] = safe_div(df["c_pay_acq_const_fiolta"], df["n_cashflow_act"])

    # 扣非真实性
    new_cols["nr_np_qoq"] = grp["net_after_nr_lp_correct"].transform(qoq)
    new_cols["nr_np_to_np_parent"] = safe_div(df["net_after_nr_lp_correct"], df["n_income_attr_p"])
    new_cols["nr_gap_ratio"] = safe_div(df["n_income_attr_p"] - df["net_after_nr_lp_correct"], df["n_income_attr_p"].abs())

    # 营运资本
    new_cols["ar_to_rev"] = safe_div(df["accounts_receiv"], df["revenue"])
    new_cols["inv_to_rev"] = safe_div(df["inventories"], df["revenue"])
    new_cols["ap_to_rev"] = safe_div(df["accounts_pay"], df["revenue"])
    new_cols["contract_liab_to_rev"] = safe_div(df["contract_liab"], df["revenue"])
    new_cols["ccc_proxy"] = new_cols["ar_to_rev"] + new_cols["inv_to_rev"] - new_cols["ap_to_rev"]

    # 先一次性合并，避免 DataFrame 碎片化
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # 边际改善：同比变化（需要重新创建groupby对象，因为df已更新）
    grp = df.groupby("ts_code", group_keys=False)
    margin_change_cols = {
        "gross_margin_yoy_chg": "gross_margin",
        "op_margin_yoy_chg": "op_margin",
        "nr_np_margin_yoy_chg": "nr_np_margin",
        "cfo_to_np_parent_yoy_chg": "cfo_to_np_parent",
        "cfo_to_nr_np_yoy_chg": "cfo_to_nr_np",
        "contract_liab_to_rev_yoy_chg": "contract_liab_to_rev",
        "rd_ratio_yoy_chg": "rd_ratio",
    }
    delta_cols: Dict[str, pd.Series] = {}
    for new_name, src_col in tqdm(margin_change_cols.items(), desc="构建边际改善因子", leave=False):
        delta_cols[new_name] = grp[src_col].transform(lambda s: s - s.shift(4))
    if delta_cols:
        df = pd.concat([df, pd.DataFrame(delta_cols, index=df.index)], axis=1)

    # 清理 inf，增加进度条
    for col in tqdm(ALL_FACTOR_COLS, desc="清理因子数值", leave=False):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    df = df.copy()  # 主动去碎片化
    logger.info("构建因子完成：生成 %s 个因子列", len([c for c in ALL_FACTOR_COLS if c in df.columns]))
    return df


def build_price_maps(daily_df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    date_map: Dict[str, np.ndarray] = {}
    price_map: Dict[str, np.ndarray] = {}
    for ts_code, group in daily_df.groupby("ts_code"):
        sorted_group = group.sort_values("trade_date")
        date_map[ts_code] = sorted_group["trade_date"].values.astype("datetime64[ns]")
        price_map[ts_code] = pd.to_numeric(sorted_group["close"], errors="coerce").values.astype(float)
    return date_map, price_map



def next_trade_index(dates: np.ndarray, ann_date: pd.Timestamp) -> Optional[int]:
    if len(dates) == 0 or pd.isna(ann_date):
        return None
    target = np.datetime64(ann_date.to_datetime64())
    idx = int(np.searchsorted(dates, target, side="right"))
    if idx >= len(dates):
        return None
    return idx


def compute_forward_returns(df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    date_map, price_map = build_price_maps(daily_df)
    records: List[Dict[str, object]] = []

    for row in df.itertuples(index=False):
        ts_code = getattr(row, "ts_code")
        ann_date = getattr(row, "ann_date")
        if ts_code not in date_map or pd.isna(ann_date):
            continue
        stock_dates = date_map[ts_code]
        stock_prices = price_map[ts_code]
        entry_idx = next_trade_index(stock_dates, pd.Timestamp(ann_date))
        if entry_idx is None:
            continue
        target_idx = entry_idx + RETURN_HORIZON
        if target_idx >= len(stock_dates):
            continue

        entry_date = pd.Timestamp(stock_dates[entry_idx])
        exit_date = pd.Timestamp(stock_dates[target_idx])
        entry_price = stock_prices[entry_idx]
        exit_price = stock_prices[target_idx]
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        ret_60d = (exit_price - entry_price) / entry_price

        record = row._asdict()
        record["entry_date"] = entry_date
        record["exit_date"] = exit_date
        record["entry_price"] = entry_price
        record["exit_price"] = exit_price
        record["ret_60d"] = ret_60d
        records.append(record)

    out = pd.DataFrame(records)
    logger.info("构建完成: %s 条事件样本, %s 只股票", len(out), out["ts_code"].nunique() if not out.empty else 0)
    return out


def preprocess_factors_by_report_date(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    winsor_q: float,
    workers: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    added_cols: Dict[str, pd.Series] = {}
    valid_factors = [factor for factor in factor_cols if factor in out.columns]
    if not valid_factors:
        return out

    period_groups = [np.asarray(idx, dtype=int) for _, idx in out.groupby("report_date").groups.items()]
    workers = resolve_worker_count(workers)

    if workers <= 1 or len(valid_factors) <= 1:
        for factor in tqdm(valid_factors, desc="报告期横截面预处理", leave=False):
            raw_series, xsec_series = _preprocess_single_factor(out[factor], period_groups, winsor_q)
            added_cols[f"{factor}__raw"] = raw_series
            added_cols[f"{factor}__xsec"] = xsec_series
    else:
        logger.info("报告期横截面预处理启用多线程: workers=%s", workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_preprocess_single_factor, out[factor], period_groups, winsor_q): factor
                for factor in valid_factors
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="报告期横截面预处理", leave=False):
                factor = futures[fut]
                raw_series, xsec_series = fut.result()
                added_cols[f"{factor}__raw"] = raw_series
                added_cols[f"{factor}__xsec"] = xsec_series

    if added_cols:
        ordered_cols = {}
        for factor in valid_factors:
            ordered_cols[f"{factor}__raw"] = added_cols[f"{factor}__raw"]
            ordered_cols[f"{factor}__xsec"] = added_cols[f"{factor}__xsec"]
        out = pd.concat([out, pd.DataFrame(ordered_cols, index=out.index)], axis=1)
    out = out.copy()  # 去碎片化，避免后续再次触发 PerformanceWarning
    return out


def compute_ic_by_period(df: pd.DataFrame, factor_col: str, return_col: str, min_n: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    xcol = f"{factor_col}__xsec" if f"{factor_col}__xsec" in df.columns else factor_col
    for report_date, group in df.groupby("report_date"):
        valid = group[[xcol, return_col]].dropna()
        if len(valid) < min_n:
            continue
        spearman = stats.spearmanr(valid[xcol], valid[return_col]).statistic
        pearson = valid[xcol].corr(valid[return_col], method="pearson")
        rows.append({
            "report_date": report_date,
            "factor": factor_col,
            "label": FACTOR_LABELS.get(factor_col, factor_col),
            "group_name": FACTOR_GROUPS.get(factor_col, "未分组"),
            "IC": spearman,
            "Pearson": pearson,
            "n": len(valid),
        })
    return pd.DataFrame(rows)


def compute_group_returns_by_period(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str,
    min_n: int,
    n_groups: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xcol = f"{factor_col}__xsec" if f"{factor_col}__xsec" in df.columns else factor_col
    detail_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for report_date, group in df.groupby("report_date"):
        valid = group[["ts_code", xcol, return_col]].dropna().copy()
        if len(valid) < max(min_n, n_groups * 8):
            continue
        try:
            valid["group"] = pd.qcut(valid[xcol], q=n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue
        if valid["group"].nunique() < 2:
            continue

        grp_df = valid.groupby("group").agg(
            mean_ret=(return_col, "mean"),
            median_ret=(return_col, "median"),
            win_rate=(return_col, lambda s: float((s > 0).mean())),
            n=(return_col, "size"),
        ).reset_index()
        grp_df["group"] = grp_df["group"].astype(int) + 1
        rets = grp_df.sort_values("group")["mean_ret"].values.astype(float)
        direction = pct_change_direction(rets)
        long_short = np.nan
        effective_direction = "unknown"
        if len(grp_df) >= 2:
            g_first = grp_df.loc[grp_df["group"] == grp_df["group"].min(), "mean_ret"].iloc[0]
            g_last = grp_df.loc[grp_df["group"] == grp_df["group"].max(), "mean_ret"].iloc[0]
            if pd.notna(g_first) and pd.notna(g_last):
                if g_last >= g_first:
                    long_short = g_last - g_first
                    effective_direction = "positive"
                else:
                    long_short = g_first - g_last
                    effective_direction = "negative"
        monotone = direction in {"ascending", "descending"}

        for _, row in grp_df.iterrows():
            detail_rows.append({
                "report_date": report_date,
                "factor": factor_col,
                "label": FACTOR_LABELS.get(factor_col, factor_col),
                "group_name": FACTOR_GROUPS.get(factor_col, "未分组"),
                "group": int(row["group"]),
                "mean_ret": row["mean_ret"],
                "median_ret": row["median_ret"],
                "win_rate": row["win_rate"],
                "n": int(row["n"]),
                "long_short": long_short,
                "direction": effective_direction,
                "monotone": monotone,
            })

        summary_rows.append({
            "report_date": report_date,
            "factor": factor_col,
            "label": FACTOR_LABELS.get(factor_col, factor_col),
            "group_name": FACTOR_GROUPS.get(factor_col, "未分组"),
            "long_short": long_short,
            "direction": effective_direction,
            "monotone": monotone,
            "n_groups": int(grp_df["group"].nunique()),
            "avg_n_per_group": float(grp_df["n"].mean()),
        })
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)




def evaluate_single_factor(
    dataset: pd.DataFrame,
    factor: str,
    return_col: str,
    min_period_n: int,
    n_groups: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ic_df = compute_ic_by_period(dataset, factor, return_col, min_period_n)
    g_detail, g_period = compute_group_returns_by_period(dataset, factor, return_col, min_period_n, n_groups)
    return ic_df, g_detail, g_period


def summarize_factors(
    dataset: pd.DataFrame,
    ic_detail: pd.DataFrame,
    group_period_summary: pd.DataFrame,
    factor_cols: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for factor in factor_cols:
        if factor not in dataset.columns:
            continue
        raw = pd.to_numeric(dataset[factor], errors="coerce")
        coverage = float(raw.notna().mean()) if len(raw) else np.nan
        missing = float(raw.isna().mean()) if len(raw) else np.nan

        ic_df = ic_detail[ic_detail["factor"] == factor].copy()
        gp_df = group_period_summary[group_period_summary["factor"] == factor].copy()
        if ic_df.empty and gp_df.empty:
            continue

        ic_mean = ic_df["IC"].mean() if not ic_df.empty else np.nan
        ic_median = ic_df["IC"].median() if not ic_df.empty else np.nan
        ic_std = ic_df["IC"].std() if not ic_df.empty else np.nan
        icir = ic_mean / ic_std if pd.notna(ic_mean) and pd.notna(ic_std) and abs(ic_std) > EPSILON else np.nan
        abs_ic_mean = ic_df["IC"].abs().mean() if not ic_df.empty else np.nan
        positive_ic_ratio = (ic_df["IC"] > 0).mean() if not ic_df.empty else np.nan
        negative_ic_ratio = (ic_df["IC"] < 0).mean() if not ic_df.empty else np.nan
        empirical_direction = "unstable"
        if pd.notna(positive_ic_ratio) and pd.notna(negative_ic_ratio):
            if positive_ic_ratio >= 0.6:
                empirical_direction = "positive"
            elif negative_ic_ratio >= 0.6:
                empirical_direction = "negative"

        avg_ls = gp_df["long_short"].mean() if not gp_df.empty else np.nan
        positive_ls_ratio = (gp_df["long_short"] > 0).mean() if not gp_df.empty else np.nan
        monotonic_ratio = gp_df["monotone"].mean() if not gp_df.empty else np.nan

        rows.append({
            "factor": factor,
            "label": FACTOR_LABELS.get(factor, factor),
            "group_name": FACTOR_GROUPS.get(factor, "未分组"),
            "coverage_ratio": round(coverage, 4) if pd.notna(coverage) else np.nan,
            "missing_ratio": round(missing, 4) if pd.notna(missing) else np.nan,
            "n_periods_ic": int(len(ic_df)),
            "avg_period_n": round(float(ic_df["n"].mean()), 1) if not ic_df.empty else np.nan,
            "IC_mean": round(float(ic_mean), 6) if pd.notna(ic_mean) else np.nan,
            "IC_median": round(float(ic_median), 6) if pd.notna(ic_median) else np.nan,
            "IC_std": round(float(ic_std), 6) if pd.notna(ic_std) else np.nan,
            "ICIR": round(float(icir), 6) if pd.notna(icir) else np.nan,
            "IC_abs_mean": round(float(abs_ic_mean), 6) if pd.notna(abs_ic_mean) else np.nan,
            "IC_positive_ratio": round(float(positive_ic_ratio), 4) if pd.notna(positive_ic_ratio) else np.nan,
            "IC_negative_ratio": round(float(negative_ic_ratio), 4) if pd.notna(negative_ic_ratio) else np.nan,
            "empirical_direction": empirical_direction,
            "avg_long_short": round(float(avg_ls), 6) if pd.notna(avg_ls) else np.nan,
            "positive_long_short_ratio": round(float(positive_ls_ratio), 4) if pd.notna(positive_ls_ratio) else np.nan,
            "monotonic_ratio": round(float(monotonic_ratio), 4) if pd.notna(monotonic_ratio) else np.nan,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(["IC_abs_mean", "avg_long_short"], ascending=[False, False]).reset_index(drop=True)
    return result


def breakdown_report_type(ic_detail: pd.DataFrame, group_period_summary: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    if ic_detail.empty and group_period_summary.empty:
        return pd.DataFrame()
    report_map = dataset[["report_date", "report_type_name"]].drop_duplicates()
    ic = ic_detail.merge(report_map, on="report_date", how="left")
    gp = group_period_summary.merge(report_map, on="report_date", how="left")

    rows: List[Dict[str, object]] = []
    for factor in sorted(set(ic.get("factor", pd.Series(dtype=str)).tolist()) | set(gp.get("factor", pd.Series(dtype=str)).tolist())):
        ic_f = ic[ic["factor"] == factor]
        gp_f = gp[gp["factor"] == factor]
        for rpt in ["一季报", "中报", "三季报", "年报"]:
            ic_r = ic_f[ic_f["report_type_name"] == rpt]
            gp_r = gp_f[gp_f["report_type_name"] == rpt]
            if ic_r.empty and gp_r.empty:
                continue
            rows.append({
                "factor": factor,
                "label": FACTOR_LABELS.get(factor, factor),
                "group_name": FACTOR_GROUPS.get(factor, "未分组"),
                "report_type_name": rpt,
                "n_periods": int(len(ic_r)),
                "IC_mean": round(float(ic_r["IC"].mean()), 6) if not ic_r.empty else np.nan,
                "IC_positive_ratio": round(float((ic_r["IC"] > 0).mean()), 4) if not ic_r.empty else np.nan,
                "avg_long_short": round(float(gp_r["long_short"].mean()), 6) if not gp_r.empty else np.nan,
                "monotonic_ratio": round(float(gp_r["monotone"].mean()), 4) if not gp_r.empty else np.nan,
            })
    return pd.DataFrame(rows)


def breakdown_year(ic_detail: pd.DataFrame, group_period_summary: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    if ic_detail.empty and group_period_summary.empty:
        return pd.DataFrame()
    year_map = dataset[["report_date", "report_year"]].drop_duplicates()
    ic = ic_detail.merge(year_map, on="report_date", how="left")
    gp = group_period_summary.merge(year_map, on="report_date", how="left")

    rows: List[Dict[str, object]] = []
    for factor in sorted(set(ic.get("factor", pd.Series(dtype=str)).tolist()) | set(gp.get("factor", pd.Series(dtype=str)).tolist())):
        ic_f = ic[ic["factor"] == factor]
        gp_f = gp[gp["factor"] == factor]
        years = sorted(set(ic_f.get("report_year", pd.Series(dtype=int)).dropna().astype(int).tolist()) |
                       set(gp_f.get("report_year", pd.Series(dtype=int)).dropna().astype(int).tolist()))
        for year in years:
            ic_y = ic_f[ic_f["report_year"] == year]
            gp_y = gp_f[gp_f["report_year"] == year]
            if ic_y.empty and gp_y.empty:
                continue
            rows.append({
                "factor": factor,
                "label": FACTOR_LABELS.get(factor, factor),
                "group_name": FACTOR_GROUPS.get(factor, "未分组"),
                "report_year": int(year),
                "n_periods": int(len(ic_y)),
                "IC_mean": round(float(ic_y["IC"].mean()), 6) if not ic_y.empty else np.nan,
                "IC_positive_ratio": round(float((ic_y["IC"] > 0).mean()), 4) if not ic_y.empty else np.nan,
                "avg_long_short": round(float(gp_y["long_short"].mean()), 6) if not gp_y.empty else np.nan,
                "monotonic_ratio": round(float(gp_y["monotone"].mean()), 4) if not gp_y.empty else np.nan,
            })
    return pd.DataFrame(rows)


def compute_correlation_matrix(df: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    cols = [f for f in factor_cols if f in df.columns and df[f].notna().sum() >= 100]
    if not cols:
        return pd.DataFrame()
    mat = df[cols].corr(method="spearman")
    mat.index = [FACTOR_LABELS.get(c, c) for c in mat.index]
    mat.columns = [FACTOR_LABELS.get(c, c) for c in mat.columns]
    return mat


def build_factor_dictionary() -> pd.DataFrame:
    rows = []
    for factor in ALL_FACTOR_COLS:
        rows.append({
            "factor": factor,
            "label": FACTOR_LABELS.get(factor, factor),
            "group_name": FACTOR_GROUPS.get(factor, "未分组"),
        })
    return pd.DataFrame(rows)


def build_overview(dataset: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    rows = [
        {"metric": "样本数", "value": len(dataset)},
        {"metric": "股票数", "value": dataset["ts_code"].nunique()},
        {"metric": "报告期数", "value": dataset["report_date"].nunique()},
        {"metric": "起始报告期", "value": dataset["report_date"].min()},
        {"metric": "结束报告期", "value": dataset["report_date"].max()},
        {"metric": "ret_60d有效样本", "value": int(dataset["ret_60d"].notna().sum())},
        {"metric": "因子总数", "value": len([c for c in factor_cols if c in dataset.columns])},
    ]
    return pd.DataFrame(rows)


def build_factor_coverage(dataset: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for factor in factor_cols:
        if factor not in dataset.columns:
            continue
        s = pd.to_numeric(dataset[factor], errors="coerce")
        rows.append({
            "factor": factor,
            "label": FACTOR_LABELS.get(factor, factor),
            "group_name": FACTOR_GROUPS.get(factor, "未分组"),
            "coverage_ratio": round(float(s.notna().mean()), 4),
            "missing_ratio": round(float(s.isna().mean()), 4),
            "median": float(s.median()) if s.notna().any() else np.nan,
            "p01": float(s.quantile(0.01)) if s.notna().any() else np.nan,
            "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["coverage_ratio", "factor"], ascending=[False, True])


def pick_candidate_factors(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    cond = (
        summary_df["coverage_ratio"].fillna(0) >= 0.40
    ) & (
        summary_df["monotonic_ratio"].fillna(0) >= 0.50
    ) & (
        summary_df["positive_long_short_ratio"].fillna(0) >= 0.50
    ) & (
        summary_df["IC_abs_mean"].fillna(0) >= 0.01
    )
    out = summary_df[cond].copy()
    return out.sort_values(["IC_abs_mean", "avg_long_short"], ascending=[False, False]).reset_index(drop=True)




def build_group_detail_compact(group_detail: pd.DataFrame, group_period_summary: pd.DataFrame, n_groups: int) -> pd.DataFrame:
    """
    将长表分组收益明细压缩为宽表：每个 factor + report_date 仅保留一行。
    这样在不减少信息量的情况下，显著降低 Excel 行数和重复字符串。
    """
    if group_detail.empty and group_period_summary.empty:
        return pd.DataFrame()
    if group_detail.empty:
        return group_period_summary.copy()

    base = group_detail[["report_date", "factor", "label", "group_name", "group", "mean_ret", "median_ret", "win_rate", "n"]].copy()
    metric_frames = []
    for metric in ["mean_ret", "median_ret", "win_rate", "n"]:
        piv = base.pivot_table(index=["report_date", "factor", "label", "group_name"], columns="group", values=metric, aggfunc="first")
        if not piv.empty:
            piv.columns = [f"{metric}_g{int(c)}" for c in piv.columns]
            metric_frames.append(piv)
    compact = pd.concat(metric_frames, axis=1).reset_index() if metric_frames else pd.DataFrame()

    if not group_period_summary.empty:
        extra_cols = [c for c in ["report_date", "factor", "long_short", "direction", "monotone", "n_groups", "avg_n_per_group"] if c in group_period_summary.columns]
        gp = group_period_summary[extra_cols].drop_duplicates(["report_date", "factor"])
        compact = compact.merge(gp, on=["report_date", "factor"], how="left")

    # 统一列顺序
    ordered = ["report_date", "factor", "label", "group_name"]
    for metric in ["mean_ret", "median_ret", "win_rate", "n"]:
        for g in range(1, n_groups + 1):
            col = f"{metric}_g{g}"
            if col in compact.columns:
                ordered.append(col)
    for col in ["long_short", "direction", "monotone", "n_groups", "avg_n_per_group"]:
        if col in compact.columns:
            ordered.append(col)
    remaining = [c for c in compact.columns if c not in ordered]
    compact = compact[ordered + remaining]
    return compact


def optimize_sheet_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """尽量减小 xlsx 体积，不改变核心信息。"""
    if df is None or df.empty:
        return df
    out = df.copy()

    # 日期列转为字符串，避免 Excel 序列值和样式膨胀；同时更利于后续人工复盘。
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime('%Y-%m-%d')

    # 浮点列降到合理展示精度：实验复盘通常不需要 15 位小数。
    float_cols = out.select_dtypes(include=['float64', 'float32']).columns
    if len(float_cols) > 0:
        out[float_cols] = out[float_cols].round(6)

    # 整数列下压类型。
    int_cols = out.select_dtypes(include=['int64', 'int32', 'int16']).columns
    for col in int_cols:
        out[col] = pd.to_numeric(out[col], downcast='integer')

    return out

def autosize_worksheet_columns(writer, sheet_name: str, df: pd.DataFrame, max_width: int = 40) -> None:
    worksheet = writer.sheets[sheet_name]
    for idx, col in enumerate(df.columns):
        values = [str(col)]
        if not df.empty:
            values.extend(df[col].astype(str).head(500).tolist())
        width = min(max(len(str(v)) for v in values) + 2, max_width)
        worksheet.set_column(idx, idx, width)


def write_excel(output_path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(
        output_path,
        engine="xlsxwriter",
        datetime_format="yyyy-mm-dd",
        date_format="yyyy-mm-dd",
        engine_kwargs={"options": {"strings_to_urls": False}},
    ) as writer:
        for sheet_name, df in sheets.items():
            clean_sheet = sheet_name[:31]
            out_df = optimize_sheet_for_excel(df)
            out_df.to_excel(writer, sheet_name=clean_sheet, index=False)
            autosize_worksheet_columns(writer, clean_sheet, out_df)
    logger.info("Excel 已输出: %s", output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="季度财务原始因子 60 日预测力探索脚本")
    parser.add_argument("--start", type=str, default="20160101", help="起始报告期 YYYYMMDD，默认 20160101")
    parser.add_argument("--limit", type=int, default=None, help="限制财务数据行数（测试用）")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="输出 Excel 路径")
    parser.add_argument("--return-col", type=str, default="ret_60d", choices=["ret_60d"], help="主评估收益列")
    parser.add_argument("--test-stock-limit", type=int, default=None, help="仅保留前N只股票完整跑通流程，例如 50")
    parser.add_argument("--min-period-n", type=int, default=30, help="单个报告期最少有效样本数")
    parser.add_argument("--n-groups", type=int, default=5, help="分组回测组数")
    parser.add_argument("--winsor-q", type=float, default=0.01, help="横截面 winsorize 分位数，默认 0.01")
    parser.add_argument("--factor-workers", type=int, default=None, help="因子预处理与评估线程数，默认自动(最多4)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = SummaryThresholds(
        min_period_n=args.min_period_n,
        n_groups=args.n_groups,
        winsor_quantile=args.winsor_q,
    )

    engine = create_engine_or_raise()

    test_codes = None
    if args.test_stock_limit:
        test_codes = load_test_stock_codes(engine, start=args.start, stock_limit=args.test_stock_limit)
        if not test_codes:
            logger.error("测试模式未能找到可用股票代码")
            return 1

    logger.info("加载原始财务数据...")
    fin_df = load_financial_data(engine, start=args.start, limit=args.limit, ts_codes=test_codes)
    if fin_df.empty:
        logger.error("financial_quarterly_data 无可用数据")
        return 1

    names_df = load_stock_names(engine)
    if test_codes and not names_df.empty:
        names_df = names_df[names_df["ts_code"].isin(test_codes)].copy()
    base_df = prepare_base_financial_df(fin_df, names_df)

    logger.info("构建原始字段与派生因子...")
    factor_df = add_factor_columns(base_df)

    logger.info("加载日线行情...")
    daily_df = load_daily_prices(engine, ts_codes=test_codes)
    if daily_df.empty:
        logger.error("stock_k_data 无可用日线数据")
        return 1

    if args.test_stock_limit:
        keep_codes = sorted(factor_df["ts_code"].dropna().unique().tolist())[: args.test_stock_limit]
        factor_df = factor_df[factor_df["ts_code"].isin(keep_codes)].copy()
        daily_df = daily_df[daily_df["ts_code"].isin(keep_codes)].copy()
        logger.info("测试模式：仅保留前 %s 只股票，实际纳入 %s 只，财务样本 %s 条，日线 %s 条", args.test_stock_limit, factor_df["ts_code"].nunique(), len(factor_df), len(daily_df))

    logger.info("构建公告后 60 日收益标签...")
    dataset = compute_forward_returns(factor_df, daily_df)
    if dataset.empty:
        logger.error("构建后的事件样本为空，请检查财务数据与日线数据覆盖")
        return 1

    factor_cols = [c for c in ALL_FACTOR_COLS if c in dataset.columns]
    logger.info("可用因子数量: %s", len(factor_cols))

    logger.info("做报告期横截面预处理（winsorize + zscore）...")
    logger.info("因子线程数配置: %s", resolve_worker_count(args.factor_workers))
    dataset = preprocess_factors_by_report_date(
        dataset, factor_cols, winsor_q=thresholds.winsor_quantile, workers=resolve_worker_count(args.factor_workers)
    )

    logger.info("计算 IC 与分组收益...")
    ic_detail_frames: List[pd.DataFrame] = []
    group_detail_frames: List[pd.DataFrame] = []
    group_period_frames: List[pd.DataFrame] = []
    factor_workers = resolve_worker_count(args.factor_workers)
    if factor_workers <= 1 or len(factor_cols) <= 1:
        for factor in tqdm(factor_cols, desc="评估因子", leave=False):
            ic_df, g_detail, g_period = evaluate_single_factor(
                dataset, factor, args.return_col, thresholds.min_period_n, thresholds.n_groups
            )
            if not ic_df.empty:
                ic_detail_frames.append(ic_df)
            if not g_detail.empty:
                group_detail_frames.append(g_detail)
            if not g_period.empty:
                group_period_frames.append(g_period)
    else:
        logger.info("因子评估启用多线程: workers=%s", factor_workers)
        with ThreadPoolExecutor(max_workers=factor_workers) as ex:
            futures = {
                ex.submit(
                    evaluate_single_factor, dataset, factor, args.return_col, thresholds.min_period_n, thresholds.n_groups
                ): factor
                for factor in factor_cols
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="评估因子", leave=False):
                ic_df, g_detail, g_period = fut.result()
                if not ic_df.empty:
                    ic_detail_frames.append(ic_df)
                if not g_detail.empty:
                    group_detail_frames.append(g_detail)
                if not g_period.empty:
                    group_period_frames.append(g_period)

    ic_detail = pd.concat(ic_detail_frames, ignore_index=True) if ic_detail_frames else pd.DataFrame()
    group_detail = pd.concat(group_detail_frames, ignore_index=True) if group_detail_frames else pd.DataFrame()
    group_period_summary = pd.concat(group_period_frames, ignore_index=True) if group_period_frames else pd.DataFrame()

    factor_summary = summarize_factors(dataset, ic_detail, group_period_summary, factor_cols)
    factor_candidates = pick_candidate_factors(factor_summary)
    report_type_breakdown = breakdown_report_type(ic_detail, group_period_summary, dataset)
    year_breakdown = breakdown_year(ic_detail, group_period_summary, dataset)
    corr_df = compute_correlation_matrix(dataset, factor_cols)

    logger.info("整理 Excel 输出...")
    sheets = {
        "样本概览": build_overview(dataset, factor_cols),
        "因子字典": build_factor_dictionary(),
        "因子覆盖率": build_factor_coverage(dataset, factor_cols),
        "因子汇总": factor_summary,
        "候选因子": factor_candidates,
        "报告期IC明细": ic_detail.sort_values(["factor", "report_date"]) if not ic_detail.empty else pd.DataFrame(),
        "分组收益紧凑": build_group_detail_compact(group_detail, group_period_summary, args.n_groups).sort_values(["factor", "report_date"]) if (not group_detail.empty or not group_period_summary.empty) else pd.DataFrame(),
        "分组收益明细": group_detail.sort_values(["factor", "report_date", "group"]) if not group_detail.empty else pd.DataFrame(),
        "分组收益汇总": group_period_summary.sort_values(["factor", "report_date"]) if not group_period_summary.empty else pd.DataFrame(),
        "财报类型拆解": report_type_breakdown.sort_values(["factor", "report_type_name"]) if not report_type_breakdown.empty else pd.DataFrame(),
        "年份拆解": year_breakdown.sort_values(["factor", "report_year"]) if not year_breakdown.empty else pd.DataFrame(),
        "因子相关性": corr_df.reset_index().rename(columns={"index": "factor_label"}) if not corr_df.empty else pd.DataFrame(),
        "事件样本预览": dataset[[
            "ts_code", "stock_name", "report_date", "ann_date", "entry_date", "exit_date",
            "ret_60d"
        ] + [c for c in ["revenue_yoy", "net_after_nr_lp_correct_yoy", "cfo_to_nr_np", "nr_gap_ratio"] if c in dataset.columns]].head(1000),
    }
    logger.info("Excel 压缩优化：分组收益增加紧凑宽表，事件样本预览缩至 1000 行，浮点展示精度压缩到 6 位小数")
    write_excel(args.out, sheets)

    logger.info("主收益列: %s", args.return_col)
    if args.test_stock_limit:
        logger.info("测试模式股票数上限: %s", args.test_stock_limit)
    logger.info("因子线程数: %s", resolve_worker_count(args.factor_workers))
    logger.info("事件样本: %s 条, 报告期: %s 个, 候选因子: %s 个", len(dataset), dataset["report_date"].nunique(), len(factor_candidates))
    if not factor_candidates.empty:
        logger.info("Top 10 候选因子:\n%s", factor_candidates.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
