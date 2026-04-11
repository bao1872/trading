#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
季度财务评分脚本（季频单季度版，单文件）

核心变化：
1. 数据源改为 financial_quarterly_data（季频单季度数据），不再从累计值(YTD)差分还原单季度；
2. 合并原 db_score.py + batch_score.py 为一个脚本；
3. 既支持单股导出 CSV，也支持全股票池批量写入 stock_financial_score_pool；
4. 支持可选 Excel 输入（用于离线调试），默认仍从数据库读取。

推荐用法：
1) 单股分析：
   python financial_quarterly_score.py --mode single --ts-code 000426.SZ --name 兴业银锡 --start 20200101 --outdir ./out

2) 批量评分入库：
   python financial_quarterly_score.py --mode batch --start 20120101 --lookback 12 --recent-n 16

3) 用导出的 Excel 调试：
   python financial_quarterly_score.py --mode single --ts-code 001332.SZ --excel-path query_result.xlsx

说明：
- 该版本假设 financial_quarterly_data 中的收入、利润、现金流、资本开支等字段均为“单季度值”；
- 资产负债表类字段（总资产、应收、存货、应付、合同负债、股东权益）为季度末时点值；
- 若数据库真实字段名与 Excel 导出字段存在大小写/符号差异，本脚本会做标准化匹配。
"""

import os
import sys
import time
import argparse
import logging
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
try:
    from sqlalchemy import create_engine, text
except Exception:
    create_engine = None
    text = None
from tqdm import tqdm

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

TARGET_TS_CODE = ""
SOURCE_TABLE = "financial_quarterly_data"
OUTPUT_TABLE = "stock_financial_score_pool"

PRE_SCORED_FACTORS = {"profit_cash_sync", "margin_profit_sync"}

FACTOR_CONFIG: List[Dict] = [
    {"dimension": "规模与增长", "factor_name": "q_rev_yoy", "label": "单季营业收入同比", "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "规模与增长", "factor_name": "q_op_yoy", "label": "单季营业利润同比", "direction": "higher_better", "weight": 0.05, "is_core": True},
    {"dimension": "规模与增长", "factor_name": "q_np_parent_yoy", "label": "单季归母净利润同比", "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "规模与增长", "factor_name": "ytd_rev_yoy", "label": "累计营业收入同比", "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "规模与增长", "factor_name": "ytd_np_parent_yoy", "label": "累计归母净利润同比", "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "规模与增长", "factor_name": "q_ebit_yoy", "label": "单季EBIT同比", "direction": "higher_better", "weight": 0.02, "is_core": False},
    {"dimension": "规模与增长", "factor_name": "q_rev_qoq", "label": "单季营业收入环比", "direction": "higher_better", "weight": 0.01, "is_core": False},
    {"dimension": "规模与增长", "factor_name": "q_op_qoq", "label": "单季营业利润环比", "direction": "higher_better", "weight": 0.01, "is_core": False},

    {"dimension": "盈利能力", "factor_name": "q_gross_margin", "label": "单季毛利率", "direction": "higher_better", "weight": 0.05, "is_core": True},
    {"dimension": "盈利能力", "factor_name": "q_gm_yoy_change", "label": "单季毛利率同比变化", "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "盈利能力", "factor_name": "q_op_margin", "label": "单季营业利润率", "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "盈利能力", "factor_name": "q_np_parent_margin", "label": "单季归母净利率", "direction": "higher_better", "weight": 0.02, "is_core": True},
    {"dimension": "盈利能力", "factor_name": "q_gm_qoq_change", "label": "单季毛利率环比变化", "direction": "higher_better", "weight": 0.01, "is_core": False},
    {"dimension": "盈利能力", "factor_name": "op_margin_change", "label": "单季营业利润率同比变化", "direction": "higher_better", "weight": 0.01, "is_core": False},
    {"dimension": "盈利能力", "factor_name": "q_ebit_margin", "label": "单季EBIT利润率", "direction": "higher_better", "weight": 0.01, "is_core": False},

    {"dimension": "利润质量", "factor_name": "q_cfo_to_np_parent", "label": "单季经营现金流/归母净利润", "direction": "higher_better", "weight": 0.06, "is_core": True},
    {"dimension": "利润质量", "factor_name": "ttm_cfo_to_np_parent", "label": "TTM经营现金流/TTM归母净利润", "direction": "higher_better", "weight": 0.05, "is_core": True},
    {"dimension": "利润质量", "factor_name": "q_accruals_to_assets", "label": "单季应计项/平均总资产", "direction": "lower_better", "weight": 0.05, "is_core": True},
    {"dimension": "利润质量", "factor_name": "ttm_cfo_to_ebit", "label": "TTM经营现金流/TTM EBIT", "direction": "higher_better", "weight": 0.02, "is_core": False},
    {"dimension": "利润质量", "factor_name": "q_np_parent_to_np", "label": "单季归母净利润/净利润", "direction": "higher_better", "weight": 0.01, "is_core": False},

    {"dimension": "现金创造能力", "factor_name": "q_cfo_to_rev", "label": "单季经营现金流/收入", "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "现金创造能力", "factor_name": "q_cfo_yoy", "label": "单季经营现金流同比", "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "现金创造能力", "factor_name": "ytd_cfo_yoy", "label": "累计经营现金流同比", "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "现金创造能力", "factor_name": "ttm_fcf_to_np_parent", "label": "TTM自由现金流/TTM归母净利润", "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "现金创造能力", "factor_name": "capex_to_cfo", "label": "TTM资本开支/TTM经营现金流", "direction": "lower_better", "weight": 0.02, "is_core": False},
    {"dimension": "现金创造能力", "factor_name": "cash_sales_ratio", "label": "销售收现比", "direction": "higher_better", "weight": 0.02, "is_core": False},
    {"dimension": "现金创造能力", "factor_name": "cash_sales_yoy", "label": "销售收现同比", "direction": "higher_better", "weight": 0.01, "is_core": False},

    {"dimension": "资产效率与资金占用", "factor_name": "roa_parent", "label": "归母ROA", "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "资产效率与资金占用", "factor_name": "cfo_to_assets", "label": "经营现金流/总资产", "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "资产效率与资金占用", "factor_name": "asset_turnover", "label": "总资产周转率", "direction": "higher_better", "weight": 0.02, "is_core": False},
    {"dimension": "资产效率与资金占用", "factor_name": "ccc", "label": "现金转换周期", "direction": "lower_better", "weight": 0.02, "is_core": False},
    {"dimension": "资产效率与资金占用", "factor_name": "contract_liab_to_rev", "label": "合同负债/收入", "direction": "higher_better", "weight": 0.01, "is_core": False},

    {"dimension": "边际变化与持续性", "factor_name": "q_rev_yoy_delta", "label": "单季收入同比变化", "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "边际变化与持续性", "factor_name": "q_np_parent_yoy_delta", "label": "单季归母净利润同比变化", "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "边际变化与持续性", "factor_name": "trend_consistency", "label": "趋势连续性", "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "边际变化与持续性", "factor_name": "profit_cash_sync", "label": "利润与现金流同步改善度", "direction": "higher_better", "weight": 0.01, "is_core": False},
    {"dimension": "边际变化与持续性", "factor_name": "margin_profit_sync", "label": "毛利率与利润率同步改善度", "direction": "higher_better", "weight": 0.01, "is_core": False},
    {"dimension": "边际变化与持续性", "factor_name": "cfo_to_np_change", "label": "经营现金流/归母净利润变化", "direction": "higher_better", "weight": 0.01, "is_core": False},
]

DIMENSION_WEIGHTS: Dict[str, float] = {
    "规模与增长": 0.24,
    "盈利能力": 0.18,
    "利润质量": 0.18,
    "现金创造能力": 0.18,
    "资产效率与资金占用": 0.10,
    "边际变化与持续性": 0.12,
}

SCORE_COLS = [
    "total_score",
    "规模与增长_score",
    "盈利能力_score",
    "利润质量_score",
    "现金创造能力_score",
    "资产效率与资金占用_score",
    "边际变化与持续性_score",
]
FACTOR_COLS = [cfg["factor_name"] for cfg in FACTOR_CONFIG]

RAW_TO_STD = {
    "tscode": "ts_code",
    "enddate": "end_date",
    "reporttype": "report_type",
    "anndate": "ann_date",
    "fanndate": "f_ann_date",
    "totalrevenue": "total_revenue",
    "revenue": "revenue",
    "opercost": "oper_cost",
    "operateprofit": "operate_profit",
    "ebit": "ebit",
    "ebitda": "ebitda",
    "nincome": "n_income",
    "nincomeattrp": "n_income_attr_p",
    "rdexp": "rd_exp",
    "ncashflowact": "n_cashflow_act",
    "freecashflow": "free_cashflow",
    "cfrsalesg": "c_fr_sale_sg",
    "cpayacqconstfiolta": "c_pay_acq_const_fiolta",
    "totalassets": "total_assets",
    "accountsreceiv": "accounts_receiv",
    "inventories": "inventories",
    "accountspay": "accounts_pay",
    "contractliab": "contract_liab",
    "totalhldreqyexcminint": "total_hldr_eqy_exc_min_int",
    "stockname": "stock_name",
    "name": "stock_name",
    "createdat": "created_at",
    "updatedat": "updated_at",
}


def normalize_colname(col: str) -> str:
    return "".join(ch.lower() for ch in str(col) if ch.isalnum())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = normalize_colname(col)
        if key in RAW_TO_STD:
            rename_map[col] = RAW_TO_STD[key]
    return df.rename(columns=rename_map).copy()


def safe_div(a, b, eps=1e-8):
    if isinstance(b, pd.Series):
        bb = b.copy()
        bb = bb.mask(bb.abs() < eps)
        return a / bb
    if b is None or pd.isna(b) or abs(b) < eps:
        return np.nan
    return a / b


def yoy(series: pd.Series, lag: int = 4) -> pd.Series:
    return safe_div(series - series.shift(lag), series.shift(lag).abs())


def qoq(series: pd.Series, lag: int = 1) -> pd.Series:
    return safe_div(series - series.shift(lag), series.shift(lag).abs())


def winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    ql = s.quantile(lower)
    qu = s.quantile(upper)
    return s.clip(lower=ql, upper=qu)


def time_series_score(series: pd.Series, direction: str, lookback: int = 12) -> pd.Series:
    s = winsorize_series(series)
    scores = []
    for i in range(len(s)):
        start = max(0, i - lookback + 1)
        window = s.iloc[start:i + 1].dropna()
        val = s.iloc[i]
        if pd.isna(val) or len(window) < 4:
            scores.append(np.nan)
            continue
        pct = (window <= val).mean() * 100
        if direction == "lower_better":
            pct = 100 - pct
        scores.append(pct)
    return pd.Series(scores, index=s.index)


def create_engine_or_raise():
    """创建PostgreSQL数据库引擎"""
    if create_engine is None or text is None:
        raise ImportError("当前环境未安装 sqlalchemy，无法使用数据库模式。可先用 --excel-path 调试，或在项目环境中安装 sqlalchemy。")
    if not DATABASE_URL:
        raise ValueError("未找到 DATABASE_URL。请在项目 config.py 中配置，或通过环境变量传入。")
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def read_source_table(
    ts_code: Optional[str] = None,
    start_date: Optional[str] = None,
    limit_n_quarters: Optional[int] = None,
    excel_path: Optional[str] = None,
    preloaded_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if preloaded_data is not None:
        raw = preloaded_data.copy()
    elif excel_path:
        logger.info(f"从 Excel 读取季频财务数据: {excel_path}")
        raw = pd.read_excel(excel_path)
    else:
        engine = create_engine_or_raise()
        with engine.connect() as conn:
            raw = pd.read_sql(text(f'SELECT * FROM "{SOURCE_TABLE}"'), conn)

    raw = normalize_columns(raw)
    if "ts_code" not in raw.columns or "end_date" not in raw.columns:
        raise ValueError("源数据缺少 ts_code 或 end_date 字段，请检查 financial_quarterly_data / Excel 列名。")

    if ts_code:
        raw = raw[raw["ts_code"].astype(str) == str(ts_code)].copy()
    if start_date:
        raw = raw[pd.to_numeric(raw["end_date"], errors="coerce") >= int(start_date)].copy()

    if not pd.api.types.is_datetime64_any_dtype(raw["end_date"]):
        raw["end_date"] = pd.to_datetime(raw["end_date"].astype(str), format="%Y%m%d", errors="coerce")
    raw = raw.dropna(subset=["end_date"]).copy()

    if limit_n_quarters:
        raw = (
            raw.sort_values(["ts_code", "end_date"])
               .groupby("ts_code", group_keys=False)
               .tail(limit_n_quarters)
               .reset_index(drop=True)
        )
    return raw.sort_values(["ts_code", "end_date"]).reset_index(drop=True)


def dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    for c in ["ann_date", "f_ann_date", "end_date"]:
        if c in tmp.columns:
            if c == "end_date":
                tmp[c] = pd.to_datetime(tmp[c], errors="coerce")
            else:
                tmp[c] = pd.to_datetime(tmp[c].astype(str), format="%Y%m%d", errors="coerce")
    sort_cols = [c for c in ["end_date", "f_ann_date", "ann_date"] if c in tmp.columns]
    if sort_cols:
        tmp = tmp.sort_values(sort_cols)
    if "ts_code" in tmp.columns:
        tmp = tmp.groupby(["ts_code", "end_date"], as_index=False).tail(1)
    else:
        tmp = tmp.groupby("end_date", as_index=False).tail(1)
    return tmp.sort_values(["ts_code", "end_date"]).reset_index(drop=True)


def check_quarter_integrity(df: pd.DataFrame) -> None:
    if df.empty:
        return
    tmp = df[["ts_code", "end_date", "fiscal_year", "quarter"]].copy()
    tmp = tmp.sort_values(["ts_code", "end_date"]).reset_index(drop=True)

    dup = tmp.groupby(["ts_code", "fiscal_year", "quarter"]).size()
    dup = dup[dup > 1]
    if not dup.empty:
        warnings.warn(f"发现同一股票同一财年重复季度记录，可能影响 TTM / YTD 计算：{dup.to_dict()}")

    tmp["q_num"] = tmp["fiscal_year"] * 4 + tmp["quarter"]
    tmp["q_num_prev"] = tmp.groupby("ts_code")["q_num"].shift(1)
    gap = tmp[(tmp["q_num_prev"].notna()) & ((tmp["q_num"] - tmp["q_num_prev"]) > 1)]
    for _, row in gap.iterrows():
        missing_qnum = int(row["q_num_prev"] + 1)
        missing_year = missing_qnum // 4
        missing_quarter = missing_qnum % 4
        if missing_quarter == 0:
            missing_year -= 1
            missing_quarter = 4
        warnings.warn(
            f"[{row['ts_code']}] 季度序列不连续：缺失 {missing_year}Q{missing_quarter}，"
            f"当前报告期 {row['end_date'].strftime('%Y-%m-%d')}"
        )


def prepare_base_dataframe(
    ts_code: str,
    start_date: str,
    limit_n_quarters: Optional[int] = None,
    excel_path: Optional[str] = None,
    stock_name: Optional[str] = None,
    preloaded_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    raw = read_source_table(
        ts_code=ts_code,
        start_date=start_date,
        limit_n_quarters=limit_n_quarters,
        excel_path=excel_path,
        preloaded_data=preloaded_data,
    )
    raw = dedup_latest(raw)

    df = raw.copy()
    df["name"] = stock_name
    df["fiscal_year"] = df["end_date"].dt.year
    df["quarter"] = df["end_date"].dt.quarter

    numeric_cols = [c for c in df.columns if c not in ["ts_code", "name", "stock_name", "end_date"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rev_q"] = df["total_revenue"] if "total_revenue" in df.columns else np.nan
    if "revenue" in df.columns:
        df["rev_q"] = df["rev_q"].where(df["rev_q"].notna(), df["revenue"])

    df["cost_q"] = df["oper_cost"] if "oper_cost" in df.columns else np.nan
    df["op_q"] = df["operate_profit"] if "operate_profit" in df.columns else np.nan
    df["ebit_q"] = df["ebit"] if "ebit" in df.columns else np.nan
    df["ebitda_q"] = df["ebitda"] if "ebitda" in df.columns else np.nan
    df["np_q"] = df["n_income"] if "n_income" in df.columns else np.nan
    df["np_parent_q"] = df["n_income_attr_p"] if "n_income_attr_p" in df.columns else np.nan
    df["cfo_q"] = df["n_cashflow_act"] if "n_cashflow_act" in df.columns else np.nan
    df["capex_q"] = df["c_pay_acq_const_fiolta"] if "c_pay_acq_const_fiolta" in df.columns else np.nan
    df["fcf_q"] = df["free_cashflow"] if "free_cashflow" in df.columns else np.nan
    if df["fcf_q"].isna().all():
        df["fcf_q"] = df["cfo_q"] - df["capex_q"]
    df["cash_sales_q"] = df["c_fr_sale_sg"] if "c_fr_sale_sg" in df.columns else np.nan

    for c in ["total_assets", "accounts_receiv", "inventories", "accounts_pay", "contract_liab", "total_hldr_eqy_exc_min_int"]:
        if c not in df.columns:
            df[c] = np.nan

    df = df.dropna(subset=["rev_q", "np_parent_q", "cfo_q"], how="all")
    df = df.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    check_quarter_integrity(df)
    return df


def add_ytd_and_ttm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for base_col in ["rev_q", "np_parent_q", "cfo_q"]:
        ytd_col = base_col.replace("_q", "_ytd")
        out[ytd_col] = out.groupby(["ts_code", "fiscal_year"])[base_col].cumsum()

    lag = out[["ts_code", "fiscal_year", "quarter", "rev_ytd", "np_parent_ytd", "cfo_ytd"]].copy()
    lag["fiscal_year"] = lag["fiscal_year"] + 1
    lag = lag.rename(columns={
        "rev_ytd": "rev_ytd_lag4",
        "np_parent_ytd": "np_parent_ytd_lag4",
        "cfo_ytd": "cfo_ytd_lag4",
    })
    out = out.merge(lag, on=["ts_code", "fiscal_year", "quarter"], how="left")

    for base_col in ["rev_q", "op_q", "ebit_q", "np_parent_q", "cfo_q", "fcf_q", "capex_q"]:
        out[base_col.replace("_q", "_ttm")] = (
            out.groupby("ts_code")[base_col]
               .rolling(4, min_periods=4)
               .sum()
               .reset_index(level=0, drop=True)
        )

    out["cost_ttm"] = (
        out.groupby("ts_code")["cost_q"]
           .rolling(4, min_periods=4)
           .sum()
           .reset_index(level=0, drop=True)
    )
    out["avg_assets"] = (out["total_assets"] + out.groupby("ts_code")["total_assets"].shift(1)) / 2
    return out


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["q_rev_yoy"] = yoy(out["rev_q"])
    out["q_op_yoy"] = yoy(out["op_q"])
    out["q_np_parent_yoy"] = yoy(out["np_parent_q"])
    out["q_ebit_yoy"] = yoy(out["ebit_q"])

    out["q_rev_qoq"] = qoq(out["rev_q"])
    out["q_op_qoq"] = qoq(out["op_q"])

    out["ytd_rev_yoy"] = safe_div(out["rev_ytd"] - out["rev_ytd_lag4"], out["rev_ytd_lag4"].abs())
    out["ytd_np_parent_yoy"] = safe_div(out["np_parent_ytd"] - out["np_parent_ytd_lag4"], out["np_parent_ytd_lag4"].abs())

    out["q_gross_margin"] = 1 - safe_div(out["cost_q"], out["rev_q"])
    out["q_gm_yoy_change"] = out["q_gross_margin"] - out["q_gross_margin"].shift(4)
    out["q_gm_qoq_change"] = out["q_gross_margin"] - out["q_gross_margin"].shift(1)

    out["q_op_margin"] = safe_div(out["op_q"], out["rev_q"])
    out["op_margin_change"] = out["q_op_margin"] - out["q_op_margin"].shift(4)
    out["q_np_parent_margin"] = safe_div(out["np_parent_q"], out["rev_q"])
    out["q_ebit_margin"] = safe_div(out["ebit_q"], out["rev_q"])

    out["q_cfo_to_np_parent"] = safe_div(out["cfo_q"], out["np_parent_q"])
    out["ttm_cfo_to_np_parent"] = safe_div(out["cfo_ttm"], out["np_parent_ttm"])
    out["q_accruals_to_assets"] = safe_div(out["np_parent_q"] - out["cfo_q"], out["avg_assets"])
    out["ttm_cfo_to_ebit"] = safe_div(out["cfo_ttm"], out["ebit_ttm"])
    out["q_np_parent_to_np"] = safe_div(out["np_parent_q"], out["np_q"])

    out["q_cfo_to_rev"] = safe_div(out["cfo_q"], out["rev_q"])
    out["q_cfo_yoy"] = yoy(out["cfo_q"])
    out["ytd_cfo_yoy"] = safe_div(out["cfo_ytd"] - out["cfo_ytd_lag4"], out["cfo_ytd_lag4"].abs())
    out["ttm_fcf_to_np_parent"] = safe_div(out["fcf_ttm"], out["np_parent_ttm"])
    out["capex_to_cfo"] = safe_div(out["capex_ttm"], out["cfo_ttm"])
    out["cash_sales_ratio"] = safe_div(out["cash_sales_q"], out["rev_q"])
    out["cash_sales_yoy"] = yoy(out["cash_sales_q"])

    out["roa_parent"] = safe_div(out["np_parent_ttm"], out["avg_assets"])
    out["cfo_to_assets"] = safe_div(out["cfo_ttm"], out["avg_assets"])
    out["asset_turnover"] = safe_div(out["rev_ttm"], out["avg_assets"])

    out["ar_days"] = safe_div(out["accounts_receiv"], out["rev_ttm"]) * 365
    out["inv_days"] = safe_div(out["inventories"], out["cost_ttm"]) * 365
    out["ap_days"] = safe_div(out["accounts_pay"], out["cost_ttm"]) * 365
    out["ccc"] = out["ar_days"] + out["inv_days"] - out["ap_days"]
    out["contract_liab_to_rev"] = safe_div(out["contract_liab"], out["rev_ttm"])

    out["q_rev_yoy_delta"] = out["q_rev_yoy"] - out["q_rev_yoy"].shift(1)
    out["q_np_parent_yoy_delta"] = out["q_np_parent_yoy"] - out["q_np_parent_yoy"].shift(1)

    out["cfo_to_rev_change"] = out["q_cfo_to_rev"] - out["q_cfo_to_rev"].shift(4)
    out["cfo_to_np_change"] = out["q_cfo_to_np_parent"] - out["q_cfo_to_np_parent"].shift(4)

    out["rev_improve"] = (out["q_rev_yoy"] > out["q_rev_yoy"].shift(1)).astype(float)
    out["np_improve"] = (out["q_np_parent_yoy"] > out["q_np_parent_yoy"].shift(1)).astype(float)
    out["gm_improve"] = (out["q_gm_yoy_change"] > out["q_gm_yoy_change"].shift(1)).astype(float)
    raw_consistency = (
        out["rev_improve"].rolling(4).sum()
        + out["np_improve"].rolling(4).sum()
        + out["gm_improve"].rolling(4).sum()
    )
    out["trend_consistency"] = raw_consistency / 12 * 100

    out["profit_cash_sync"] = (
        time_series_score(out["q_np_parent_yoy_delta"], "higher_better", 12).fillna(0)
        + time_series_score(out["cfo_to_rev_change"], "higher_better", 12).fillna(0)
    ) / 2

    out["margin_profit_sync"] = (
        time_series_score(out["q_gm_yoy_change"], "higher_better", 12).fillna(0)
        + time_series_score(out["op_margin_change"], "higher_better", 12).fillna(0)
    ) / 2

    return out


def score_dataframe(df: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    out = df.copy()

    for cfg in FACTOR_CONFIG:
        fname = cfg["factor_name"]
        score_col = f"{fname}_score"
        if fname not in out.columns:
            out[score_col] = np.nan
            continue
        if fname in PRE_SCORED_FACTORS:
            out[score_col] = out[fname]
        else:
            out[score_col] = time_series_score(out[fname], cfg["direction"], lookback=lookback)

    dimensions = sorted(set(c["dimension"] for c in FACTOR_CONFIG))
    for dim in dimensions:
        sub = [c for c in FACTOR_CONFIG if c["dimension"] == dim]
        dim_scores = np.full(len(out), np.nan)
        for idx in range(len(out)):
            score_num = 0.0
            score_den = 0.0
            for cfg in sub:
                score_col = f"{cfg['factor_name']}_score"
                if score_col in out.columns:
                    s = out[score_col].iloc[idx]
                    if pd.notna(s):
                        score_num += s * cfg["weight"]
                        score_den += cfg["weight"]
            dim_scores[idx] = score_num / score_den if score_den > 0 else np.nan
        out[f"{dim}_score"] = dim_scores

    out["total_score"] = np.nan
    for idx in range(len(out)):
        total_num = 0.0
        total_den = 0.0
        for dim, w in DIMENSION_WEIGHTS.items():
            s = out[f"{dim}_score"].iloc[idx]
            if pd.notna(s):
                total_num += s * w
                total_den += w
        out.loc[out.index[idx], "total_score"] = total_num / total_den if total_den > 0 else np.nan
    return out


def fetch_stock_pool(engine, limit: Optional[int] = None) -> pd.DataFrame:
    sql = 'SELECT ts_code, name FROM stock_pools ORDER BY ts_code'
    if limit:
        sql += f" LIMIT {int(limit)}"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    logger.info(f"股票池共 {len(df)} 只")
    return df


def fetch_already_processed(engine) -> set:
    with engine.connect() as conn:
        df = pd.read_sql(text(f'SELECT ts_code FROM "{OUTPUT_TABLE}"'), conn)
    return set(df["ts_code"].tolist())


def ensure_output_table_exists(engine):
    """确保输出表存在（PostgreSQL）"""
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = :name"),
            {"name": OUTPUT_TABLE},
        ).fetchone()[0] > 0
        pk_def = "id SERIAL PRIMARY KEY"
        float_type = "DOUBLE PRECISION"
        ts_default = "CURRENT_TIMESTAMP"

        if not exists:
            logger.info(f"创建表 {OUTPUT_TABLE} ...")
            columns = [
                pk_def,
                'ts_code VARCHAR(20) NOT NULL',
                'stock_name VARCHAR(50)',
                'report_date VARCHAR(8) NOT NULL',
            ]
            for col in SCORE_COLS + FACTOR_COLS:
                columns.append(f'"{col}" {float_type}')
            columns.append(f"created_at TIMESTAMP DEFAULT {ts_default}")
            create_sql = f'''CREATE TABLE IF NOT EXISTS "{OUTPUT_TABLE}" (
                {", ".join(columns)},
                UNIQUE(ts_code, report_date)
            )'''
            conn.execute(text(create_sql))
            conn.commit()
            logger.info(f"表 {OUTPUT_TABLE} 创建完成")
        else:
            logger.info(f"表 {OUTPUT_TABLE} 已存在，检查并补充缺失列...")
            all_cols = SCORE_COLS + FACTOR_COLS
            if engine.dialect.name == "postgresql":
                result = conn.execute(
                    text("SELECT column_name FROM information_schema.columns WHERE table_name = :name"),
                    {"name": OUTPUT_TABLE},
                )
            else:
                result = conn.execute(
                    text(f"PRAGMA table_info(\"{OUTPUT_TABLE}\")")
                )
            existing = {row[0] for row in result.fetchall()}
            missing = [c for c in all_cols if c not in existing]
            if missing:
                for col in missing:
                    conn.execute(text(f'ALTER TABLE "{OUTPUT_TABLE}" ADD COLUMN "{col}" {float_type}'))
                conn.commit()
                logger.info(f"已补充 {len(missing)} 个缺失列: {missing}")
            else:
                logger.info("无缺失列")


def clean_report_date(engine, report_date: str) -> int:
    with engine.connect() as conn:
        result = conn.execute(
            text(f'DELETE FROM "{OUTPUT_TABLE}" WHERE report_date = :rd'),
            {"rd": report_date},
        )
        conn.commit()
        return result.rowcount


def compute_single_stock(
    ts_code: str,
    name: str,
    start_date: str,
    lookback: int,
    limit_n_quarters: int = 16,
    excel_path: Optional[str] = None,
    preloaded_data: Optional[pd.DataFrame] = None,
):
    try:
        df = prepare_base_dataframe(
            ts_code=ts_code,
            start_date=start_date,
            limit_n_quarters=limit_n_quarters,
            excel_path=excel_path,
            stock_name=name,
            preloaded_data=preloaded_data,
        )
        if df.empty:
            return None
        df = add_ytd_and_ttm(df)
        df = add_factors(df)
        return score_dataframe(df, lookback=lookback)
    except Exception as e:
        logger.warning(f"[{ts_code}] 计算异常: {e}")
        return None


def build_output_row(latest: pd.DataFrame, ts_code: str, name: str) -> dict:
    row = {
        "ts_code": ts_code,
        "stock_name": name,
        "report_date": latest["end_date"].iloc[0].strftime("%Y%m%d"),
    }
    for col in SCORE_COLS + FACTOR_COLS:
        if col in latest.columns:
            v = latest[col].iloc[0]
            row[col] = None if pd.isna(v) else float(v)
    return row


def upsert_rows(engine, rows: List[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    cols = list(df.columns)
    uniq = ["ts_code", "report_date"]
    non_key = [c for c in cols if c not in uniq]

    qcols = ", ".join([f'"{c}"' for c in cols])
    placeholders = ", ".join([f":{c}" for c in cols])
    update_clause = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in non_key])

    sql = f'''INSERT INTO "{OUTPUT_TABLE}" ({qcols})
        VALUES ({placeholders})
        ON CONFLICT ({", ".join(uniq)}) DO UPDATE SET {update_clause}
    '''
    with engine.connect() as conn:
        for _, r in df.iterrows():
            conn.execute(text(sql), r.to_dict())
        conn.commit()
    return len(rows)


def run_single_mode(args):
    engine = create_engine_or_raise()
    ensure_output_table_exists(engine)

    scored = compute_single_stock(
        ts_code=args.ts_code,
        name=args.name or args.ts_code,
        start_date=args.start,
        lookback=args.lookback,
        limit_n_quarters=args.recent_n,
        excel_path=args.excel_path,
    )
    if scored is None or scored.empty:
        raise ValueError(f"未取到 {args.ts_code} 的季频财务数据，请检查 ts_code、起始日期与数据源。")

    latest = scored.iloc[[-1]].copy()
    out_row = build_output_row(latest, args.ts_code, args.name or args.ts_code)
    n = upsert_rows(engine, [out_row])
    logger.info(f"已写入 {n} 条评分记录到 {OUTPUT_TABLE}")


def run_batch_mode(args):
    if args.excel_path:
        raise ValueError("batch 模式不建议使用单个 Excel 文件作为全股票池来源。请改为数据库模式。")

    engine = create_engine_or_raise()
    ensure_output_table_exists(engine)

    if args.clean:
        n = clean_report_date(engine, args.clean)
        logger.info(f"已清空 report_date={args.clean} 的 {n} 条旧数据")

    logger.info("一次性加载 financial_quarterly_data ...")
    preloaded_data = read_source_table(
        start_date=args.start,
        limit_n_quarters=args.recent_n,
    )
    logger.info(f"已加载 {len(preloaded_data)} 行财务数据")

    stock_pool = fetch_stock_pool(engine, limit=args.limit)
    if args.resume:
        already = fetch_already_processed(engine)
        stock_pool = stock_pool[~stock_pool["ts_code"].isin(already)]
        logger.info(f"断点续算模式，跳过 {len(already)} 只已有数据，待处理 {len(stock_pool)} 只")

    success, skipped = 0, 0
    for _, row in tqdm(stock_pool.iterrows(), total=len(stock_pool), desc="财务评分"):
        ts_code = row["ts_code"]
        name = row["name"] if pd.notna(row["name"]) else ts_code

        scored = compute_single_stock(
            ts_code=ts_code,
            name=name,
            start_date=args.start,
            lookback=args.lookback,
            limit_n_quarters=args.recent_n,
            excel_path=None,
            preloaded_data=preloaded_data,
        )
        if scored is None or scored.empty:
            skipped += 1
            tqdm.write(f"[{ts_code}] 无数据，已跳过")
            time.sleep(0.1)
            continue

        latest = scored.iloc[[-1]].copy()
        out_row = build_output_row(latest, ts_code, name)
        upsert_rows(engine, [out_row])
        success += 1
        time.sleep(0.1)

    logger.info(f"完成！成功 {success} 只，跳过 {skipped} 只")
    with engine.connect() as conn:
        total = conn.execute(text(f'SELECT COUNT(*) FROM "{OUTPUT_TABLE}"')).fetchone()[0]
        latest_date = conn.execute(text(f'SELECT MAX(report_date) FROM "{OUTPUT_TABLE}"')).fetchone()[0]
    logger.info(f"当前表共 {total} 条，最新报告期: {latest_date}")


def parse_args():
    parser = argparse.ArgumentParser(description="季度财务评分脚本（季频单季度版，单文件）")
    parser.add_argument("--mode", choices=["single", "batch"], default="single", help="运行模式：single 单股入库；batch 股票池批量入库")
    parser.add_argument("--ts-code", type=str, default="", help="股票代码，single 模式下使用")
    parser.add_argument("--name", type=str, default="", help="股票简称（仅展示用）")
    parser.add_argument("--start", type=str, default="20120101", help="开始日期 YYYYMMDD")
    parser.add_argument("--lookback", type=int, default=12, help="时序打分回看季度数，默认 12")
    parser.add_argument("--recent-n", type=int, default=16, help="每只股票最多读取最近 N 个季度，默认 16")
    parser.add_argument("--excel-path", type=str, default=None, help="可选：从 Excel 读取季频数据做单股调试")
    parser.add_argument("--clean", type=str, default=None, help="batch 模式下清空指定报告期后重算，如 20251231")
    parser.add_argument("--limit", type=int, default=None, help="batch 模式限制股票数量（测试用）")
    parser.add_argument("--resume", action="store_true", help="batch 模式跳过已入库股票")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "single":
        run_single_mode(args)
    else:
        run_batch_mode(args)


if __name__ == "__main__":
    main()
