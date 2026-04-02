
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中钨高新（000657.SZ）基于 Tushare 三大报表的季度财务评分脚本

目的：
- 从 financial_quarterly_data 表获取利润表、资产负债表、现金流量表
- 构造季度 / 累计(YTD) / TTM 财务因子
- 按 6 个维度对中钨高新进行单股时序评分
- 评分结果写入 stock_financial_score_pool 表

说明：
- 数据从 financial_quarterly_data 表读取（由 tushare/fetcher.py 从 Tushare API 获取并保存）
- 单股评分无法做横截面分位数，这里采用"相对自身过去 lookback 个季度"的时序分位数打分
- 利润表、现金流量表使用 Tushare 单季度合并报表（report_type=2）
- 资产负债表为时点值，使用普通合并报表口径（report_type=1）
- 当前脚本适合做"当前/历史财务状态分析"；如果用于严格历史回测，需要按当时可见 ann_date/f_ann_date 截断，避免修订版前视偏差
- 部分增强因子依赖字段（如合同负债、销售收现、自由现金流等），如果数据缺失会自动跳过对应打分

使用方式：
    # Step 1: 获取 Tushare 数据并保存到 DB
    python -m tushare_data.fetcher --ts_code 000657.SZ --start_date 20120101

    # Step 2: 计算评分（从 DB 读取数据）
    python financial_factors/sample_score.py
"""

import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


TARGET_TS_CODE = "000657.SZ"
TARGET_NAME = "中钨高新"

# 这些因子在 add_factors 中已经生成了 0~100 分数，score_dataframe 不再重复打分
PRE_SCORED_FACTORS = {"profit_cash_sync", "margin_profit_sync"}

FACTOR_CONFIG: List[Dict] = [
    # 规模与增长
    {"dimension": "规模与增长", "factor_name": "q_rev_yoy", "label": "单季营业收入同比",
     "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "规模与增长", "factor_name": "q_op_yoy", "label": "单季营业利润同比",
     "direction": "higher_better", "weight": 0.05, "is_core": True},
    {"dimension": "规模与增长", "factor_name": "q_np_parent_yoy", "label": "单季归母净利润同比",
     "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "规模与增长", "factor_name": "ytd_rev_yoy", "label": "累计营业收入同比",
     "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "规模与增长", "factor_name": "ytd_np_parent_yoy", "label": "累计归母净利润同比",
     "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "规模与增长", "factor_name": "q_ebit_yoy", "label": "单季EBIT同比",
     "direction": "higher_better", "weight": 0.02, "is_core": False},
    {"dimension": "规模与增长", "factor_name": "q_rev_qoq", "label": "单季营业收入环比",
     "direction": "higher_better", "weight": 0.01, "is_core": False},
    {"dimension": "规模与增长", "factor_name": "q_op_qoq", "label": "单季营业利润环比",
     "direction": "higher_better", "weight": 0.01, "is_core": False},

    # 盈利能力
    {"dimension": "盈利能力", "factor_name": "q_gross_margin", "label": "单季毛利率",
     "direction": "higher_better", "weight": 0.05, "is_core": True},
    {"dimension": "盈利能力", "factor_name": "q_gm_yoy_change", "label": "单季毛利率同比变化",
     "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "盈利能力", "factor_name": "q_op_margin", "label": "单季营业利润率",
     "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "盈利能力", "factor_name": "q_np_parent_margin", "label": "单季归母净利率",
     "direction": "higher_better", "weight": 0.02, "is_core": True},
    {"dimension": "盈利能力", "factor_name": "q_gm_qoq_change", "label": "单季毛利率环比变化",
     "direction": "higher_better", "weight": 0.01, "is_core": False},
    {"dimension": "盈利能力", "factor_name": "op_margin_change", "label": "单季营业利润率同比变化",
     "direction": "higher_better", "weight": 0.01, "is_core": False},
    {"dimension": "盈利能力", "factor_name": "q_ebit_margin", "label": "单季EBIT利润率",
     "direction": "higher_better", "weight": 0.01, "is_core": False},

    # 利润质量
    {"dimension": "利润质量", "factor_name": "q_cfo_to_np_parent", "label": "单季经营现金流/归母净利润",
     "direction": "higher_better", "weight": 0.06, "is_core": True},
    {"dimension": "利润质量", "factor_name": "ttm_cfo_to_np_parent", "label": "TTM经营现金流/TTM归母净利润",
     "direction": "higher_better", "weight": 0.05, "is_core": True},
    {"dimension": "利润质量", "factor_name": "q_accruals_to_assets", "label": "单季应计项/平均总资产",
     "direction": "lower_better", "weight": 0.05, "is_core": True},
    {"dimension": "利润质量", "factor_name": "ttm_cfo_to_ebit", "label": "TTM经营现金流/TTM EBIT",
     "direction": "higher_better", "weight": 0.02, "is_core": False},
    {"dimension": "利润质量", "factor_name": "q_np_parent_to_np", "label": "单季归母净利润/净利润",
     "direction": "higher_better", "weight": 0.01, "is_core": False},

    # 现金创造能力
    {"dimension": "现金创造能力", "factor_name": "q_cfo_to_rev", "label": "单季经营现金流/收入",
     "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "现金创造能力", "factor_name": "q_cfo_yoy", "label": "单季经营现金流同比",
     "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "现金创造能力", "factor_name": "ytd_cfo_yoy", "label": "累计经营现金流同比",
     "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "现金创造能力", "factor_name": "ttm_fcf_to_np_parent", "label": "TTM自由现金流/TTM归母净利润",
     "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "现金创造能力", "factor_name": "capex_to_cfo", "label": "TTM资本开支/TTM经营现金流",
     "direction": "lower_better", "weight": 0.02, "is_core": False},
    {"dimension": "现金创造能力", "factor_name": "cash_sales_ratio", "label": "销售收现比",
     "direction": "higher_better", "weight": 0.02, "is_core": False},
    {"dimension": "现金创造能力", "factor_name": "cash_sales_yoy", "label": "销售收现同比",
     "direction": "higher_better", "weight": 0.01, "is_core": False},

    # 资产效率与资金占用
    {"dimension": "资产效率与资金占用", "factor_name": "roa_parent", "label": "归母ROA",
     "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "资产效率与资金占用", "factor_name": "cfo_to_assets", "label": "经营现金流/总资产",
     "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "资产效率与资金占用", "factor_name": "asset_turnover", "label": "总资产周转率",
     "direction": "higher_better", "weight": 0.02, "is_core": False},
    {"dimension": "资产效率与资金占用", "factor_name": "ccc", "label": "现金转换周期",
     "direction": "lower_better", "weight": 0.02, "is_core": False},
    {"dimension": "资产效率与资金占用", "factor_name": "contract_liab_to_rev", "label": "合同负债/收入",
     "direction": "higher_better", "weight": 0.01, "is_core": False},

    # 边际变化与持续性
    {"dimension": "边际变化与持续性", "factor_name": "q_rev_yoy_delta", "label": "单季收入同比变化",
     "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "边际变化与持续性", "factor_name": "q_np_parent_yoy_delta", "label": "单季归母净利润同比变化",
     "direction": "higher_better", "weight": 0.03, "is_core": True},
    {"dimension": "边际变化与持续性", "factor_name": "trend_consistency", "label": "趋势连续性",
     "direction": "higher_better", "weight": 0.04, "is_core": True},
    {"dimension": "边际变化与持续性", "factor_name": "profit_cash_sync", "label": "利润与现金流同步改善度",
     "direction": "higher_better", "weight": 0.01, "is_core": False},
    {"dimension": "边际变化与持续性", "factor_name": "margin_profit_sync", "label": "毛利率与利润率同步改善度",
     "direction": "higher_better", "weight": 0.01, "is_core": False},
    {"dimension": "边际变化与持续性", "factor_name": "cfo_to_np_change", "label": "经营现金流/归母净利润变化",
     "direction": "higher_better", "weight": 0.01, "is_core": False},
]

DIMENSION_WEIGHTS: Dict[str, float] = {
    "规模与增长": 0.24,
    "盈利能力": 0.18,
    "利润质量": 0.18,
    "现金创造能力": 0.18,
    "资产效率与资金占用": 0.10,
    "边际变化与持续性": 0.12,
}


def safe_div(a, b, eps=1e-8):
    if isinstance(b, pd.Series):
        bb = b.copy()
        bb = bb.mask(bb.abs() < eps)
        return a / bb
    if b is None or pd.isna(b) or abs(b) < eps:
        return np.nan
    return a / b


def yoy(series: pd.Series, lag: int = 4) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex) and "ts_code" in series.index.names:
        return series.groupby("ts_code").transform(
            lambda x: safe_div(x - x.shift(lag), x.shift(lag).abs())
        )
    return safe_div(series - series.shift(lag), series.shift(lag).abs())


def qoq(series: pd.Series, lag: int = 1) -> pd.Series:
    if isinstance(series.index, pd.MultiIndex) and "ts_code" in series.index.names:
        return series.groupby("ts_code").transform(
            lambda x: safe_div(x - x.shift(lag), x.shift(lag).abs())
        )
    return safe_div(series - series.shift(lag), series.shift(lag).abs())


def winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    ql = s.quantile(lower)
    qu = s.quantile(upper)
    return s.clip(lower=ql, upper=qu)


def time_series_score(series: pd.Series, direction: str, lookback: int = 12) -> pd.Series:
    """
    单股时序打分：当前值相对自身过去 lookback 个季度的历史分位数
    """
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


FINANCIAL_QUARTERLY_COLUMNS = [
    "ts_code", "end_date", "report_type", "ann_date", "f_ann_date",
    "total_revenue", "revenue", "oper_cost", "operate_profit",
    "ebit", "ebitda", "n_income", "n_income_attr_p", "rd_exp",
    "n_cashflow_act", "free_cashflow", "c_fr_sale_sg", "c_pay_acq_const_fiolta",
    "total_assets", "accounts_receiv", "inventories", "accounts_pay",
    "contract_liab", "total_hldr_eqy_exc_min_int",
]


def fetch_financial_from_db(ts_code: str, start_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    从 financial_quarterly_data 表读取数据

    Args:
        ts_code: 股票代码（如 '000657.SZ'）
        start_date: 起始日期（如 '20120101'）

    Returns:
        (income, cash, bs) 三个 DataFrame，字段名与 Tushare 格式一致
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from sqlalchemy import create_engine, text
    from config import DATABASE_URL

    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

    sql = f'''
        SELECT {", ".join(FINANCIAL_QUARTERLY_COLUMNS)}
        FROM financial_quarterly_data
        WHERE ts_code = :ts_code
    '''
    params = {"ts_code": ts_code}
    if start_date:
        sql += " AND end_date >= :start_date"
        params["start_date"] = start_date
    sql += " ORDER BY end_date"

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["end_date"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
    df = df.sort_values("end_date").reset_index(drop=True)

    income_cols = ["ts_code", "end_date", "report_type", "ann_date", "f_ann_date",
                   "total_revenue", "revenue", "oper_cost", "operate_profit",
                   "ebit", "ebitda", "n_income", "n_income_attr_p", "rd_exp"]
    cash_cols = ["ts_code", "end_date", "report_type", "ann_date", "f_ann_date",
                 "n_cashflow_act", "free_cashflow", "c_fr_sale_sg", "c_pay_acq_const_fiolta"]
    bs_cols = ["ts_code", "end_date", "report_type", "ann_date", "f_ann_date",
               "total_assets", "accounts_receiv", "inventories", "accounts_pay",
               "contract_liab", "total_hldr_eqy_exc_min_int"]

    income = df[[c for c in income_cols if c in df.columns]].copy()
    cash = df[[c for c in cash_cols if c in df.columns]].copy()
    bs = df[[c for c in bs_cols if c in df.columns]].copy()

    return income, cash, bs


def dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
    """
    同一 end_date 可能有多条记录，保留最新披露版本。
    适用于当前时点分析；若做严格历史回测，需要按可见 ann_date 截断。
    """
    if df.empty:
        return df
    tmp = df.copy()
    for c in ["ann_date", "f_ann_date", "end_date"]:
        if c in tmp.columns:
            tmp[c] = pd.to_datetime(tmp[c], format="%Y%m%d", errors="coerce")
    sort_cols = [c for c in ["end_date", "f_ann_date", "ann_date"] if c in tmp.columns]
    tmp = tmp.sort_values(sort_cols)
    tmp = tmp.groupby(["ts_code", "end_date"], as_index=False).tail(1) if "ts_code" in tmp.columns else tmp.groupby("end_date", as_index=False).tail(1)
    return tmp.sort_values("end_date").reset_index(drop=True)


def check_quarter_integrity(df: pd.DataFrame) -> None:
    """
    检查季度序列是否完整。当前只发 warning，不阻断执行。
    """
    if df.empty:
        return
    tmp = df[["ts_code", "end_date", "fiscal_year", "quarter"]].copy()
    tmp = tmp.sort_values(["ts_code", "end_date"]).reset_index(drop=True)

    dup = tmp.groupby(["ts_code", "fiscal_year", "quarter"]).size()
    dup = dup[dup > 1]
    if not dup.empty:
        warnings.warn(f"发现同一股票同一财年重复季度记录，可能影响 YTD/TTM 计算：{dup.to_dict()}")

    qnum = tmp["fiscal_year"] * 4 + tmp["quarter"]
    tmp["qnum"] = qnum
    diffs = tmp.groupby("ts_code")["qnum"].diff().dropna()
    bad = diffs[diffs != 1]
    if not bad.empty:
        for idx in bad.index:
            prev_qnum = tmp.loc[idx - 1, "qnum"] if idx > 0 else None
            curr_qnum = tmp.loc[idx, "qnum"]
            if prev_qnum is not None:
                missing_qnum = int(prev_qnum + 1)
                missing_year = missing_qnum // 4
                missing_quarter = missing_qnum % 4
                if missing_quarter == 0:
                    missing_year -= 1
                    missing_quarter = 4
                ts = tmp.loc[idx, "ts_code"]
                warnings.warn(
                    f"[{ts}] 季度序列不连续："
                    f"缺失 {missing_year}Q{missing_quarter} (qnum={missing_qnum})，"
                    f"当前 qnum={int(curr_qnum)}，diff={int(bad.loc[idx])}"
                )


def prepare_base_dataframe(ts_code: str, start_date: str) -> pd.DataFrame:
    income, cash, bs = fetch_financial_from_db(ts_code=ts_code, start_date=start_date)
    income = dedup_latest(income)
    cash = dedup_latest(cash)
    bs = dedup_latest(bs)

    df = income.merge(
        cash.drop(columns=["report_type", "ann_date", "f_ann_date"], errors="ignore"),
        on=["ts_code", "end_date"], how="outer"
    ).merge(
        bs.drop(columns=["report_type", "ann_date", "f_ann_date"], errors="ignore"),
        on=["ts_code", "end_date"], how="outer"
    )

    df["name"] = TARGET_NAME
    df["end_date"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
    df = df.sort_values(["ts_code", "end_date"]).reset_index(drop=True)

    for col in df.columns:
        if col not in ["ts_code", "name", "end_date"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rev_q"] = df["total_revenue"].where(df["total_revenue"].notna(), df["revenue"])
    df["cost_q"] = df["oper_cost"]
    df["op_q"] = df["operate_profit"]
    df["ebit_q"] = df["ebit"]
    df["ebitda_q"] = df["ebitda"]
    df["np_q"] = df["n_income"]
    df["np_parent_q"] = df["n_income_attr_p"]
    df["cfo_q"] = df["n_cashflow_act"]
    df["capex_q"] = df["c_pay_acq_const_fiolta"]
    df["fcf_q"] = df["cfo_q"] - df["capex_q"]
    df["cash_sales_q"] = df["c_fr_sale_sg"]

    df["fiscal_year"] = df["end_date"].dt.year
    df["quarter"] = df["end_date"].dt.quarter

    df = df.dropna(subset=["rev_q", "np_parent_q", "cfo_q"], how="all")

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
            out.groupby("ts_code")[base_col].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        )

    out["cost_ttm"] = out.groupby("ts_code")["cost_q"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
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
    # 统一口径：这里改用归母净利润与其他核心因子保持一致
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

    # 复合边际改善项：这里直接生成“分数”，后续不重复 time_series_score
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

    dimensions = sorted(set([c["dimension"] for c in FACTOR_CONFIG]))
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


def build_latest_outputs(scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    latest = scored.iloc[[-1]].copy()

    summary_cols = [
        "end_date", "ts_code", "name", "total_score",
        "规模与增长_score", "盈利能力_score", "利润质量_score",
        "现金创造能力_score", "资产效率与资金占用_score", "边际变化与持续性_score"
    ]
    summary = latest[summary_cols].copy()

    factor_rows = []
    for cfg in FACTOR_CONFIG:
        fname = cfg["factor_name"]
        raw_val = latest[fname].iloc[0] if fname in latest.columns else np.nan
        score_val = latest[f"{fname}_score"].iloc[0] if f"{fname}_score" in latest.columns else np.nan
        factor_rows.append({
            "所属维度": cfg["dimension"],
            "因子名": fname,
            "中文名": cfg["label"],
            "原始值": raw_val,
            "分数": score_val,
            "方向": cfg["direction"],
            "权重": cfg["weight"],
            "是否主模型": "是" if cfg["is_core"] else "否",
        })
    factor_df = pd.DataFrame(factor_rows).sort_values(["所属维度", "权重"], ascending=[True, False])
    return summary, factor_df


def parse_args():
    parser = argparse.ArgumentParser(description="中钨高新 财务评分")
    return parser.parse_args()


def upsert_score_to_db(engine, latest: pd.DataFrame, ts_code: str, name: str):
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

    row = {
        "ts_code": ts_code,
        "stock_name": name,
        "report_date": latest["end_date"].iloc[0].strftime("%Y%m%d"),
    }
    for col in SCORE_COLS:
        if col in latest.columns:
            v = latest[col].iloc[0]
            row[col] = None if pd.isna(v) else float(v)
    for col in FACTOR_COLS:
        if col in latest.columns:
            v = latest[col].iloc[0]
            row[col] = None if pd.isna(v) else float(v)

    cols = list(row.keys())
    uniq = ["ts_code", "report_date"]
    non_key = [c for c in cols if c not in uniq]
    update_clause = ", ".join([f"{c} = EXCLUDED.{c}" for c in non_key])
    col_clause = ", ".join(cols)
    placeholders = ", ".join([f":{c}" for c in cols])
    sql = f"""
        INSERT INTO stock_financial_score_pool ({col_clause})
        VALUES ({placeholders})
        ON CONFLICT ({', '.join(uniq)}) DO UPDATE SET {update_clause}
    """

    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text(sql), row)
        conn.commit()
    return row


def main():
    args = parse_args()

    df = prepare_base_dataframe(ts_code=TARGET_TS_CODE, start_date="20120101")
    if df.empty:
        raise ValueError(f"未取到 {TARGET_TS_CODE} 的报表数据")

    df = add_ytd_and_ttm(df)
    df = add_factors(df)
    scored = score_dataframe(df, lookback=12)

    latest = scored.iloc[[-1]].copy()

    from config import DATABASE_URL
    from sqlalchemy import create_engine
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

    row = upsert_score_to_db(engine, latest, TARGET_TS_CODE, TARGET_NAME)

    print(f"已写入: {row['ts_code']} {row['report_date']} total_score={row['total_score']:.2f}")


if __name__ == "__main__":
    main()
