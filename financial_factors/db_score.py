
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
季度财务评分脚本（权威实现）

数据源：SQLite 数据库（stock_financial_summary 表，YTD 累计格式）

核心计算流程：
  fetch_financial_statements_from_db()  →  prepare_base_dataframe()
  →  add_ytd_and_ttm()  →  add_factors()  →  score_dataframe()

Quarter还原逻辑：
  数据库中存储的是 YTD 累计值，本脚本通过 convert_flows_to_single_quarter()
  将其还原为单季度值：Q1 使用原始 YTD，Q2+ 使用 YTD 差分

输出：
  - 终端打印最新报告期总分、维度分、主要因子分
  - 同目录生成：
    - {ts_code}_scores.csv：全时间序列打分结果
    - {ts_code}_latest_summary.csv：最新报告期总分 + 维度分
    - {ts_code}_latest_factor_scores.csv：最新报告期各因子原始值 + 得分明细

Usage:
  python db_score.py --ts-code 000426.SZ --name "兴业银锡" --start 20200101 --lookback 12

Author: 财务因子评分项目组
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


TARGET_TS_CODE = "000657.SZ"
TARGET_NAME = "中钨高新"

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

FLOW_COLUMNS = [
    "营业总收入", "营业收入", "营业成本", "营业利润",
    "归母净利润", "少数股东损益", "EBIT",
    "经营活动现金流净额", "资本开支", "销售商品提供劳务收到的现金",
    "EBITDA", "研发费用",
]
POINT_IN_TIME_COLUMNS = [
    "总资产", "应收账款", "存货", "应付账款", "合同负债", "股东权益"
]
REQUIRED_COLUMNS = [
    "ts_code", "stock_name", "report_date",
    "营业总收入", "营业收入", "营业成本", "营业利润",
    "归母净利润", "少数股东损益", "EBIT", "经营活动现金流净额",
    "资本开支", "销售商品提供劳务收到的现金",
    "总资产", "应收账款", "存货", "应付账款", "合同负债", "EBITDA", "研发费用"
]


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


def _convert_group_mixed_flow_to_single(g: pd.DataFrame, col: str) -> pd.Series:
    """
    对单个股票、单个财年内某个流量字段做“single/ytd 混合口径”解析，输出单季度值。

    规则：
    - single：直接使用
    - ytd：
        * Q1：直接视为单季度
        * 若上一季度存在且也是 ytd：当前单季度 = 当前YTD - 上一季度YTD
        * 若上一季度存在且是 single：当前单季度 = 当前YTD - 之前已累计的 single
        * 若缺少上一季度，则无法精确还原，记为 NaN
    """
    g = g.sort_values("quarter").copy()
    result = []
    quarter_to_raw = {}
    quarter_to_mode = {}
    cum_single = 0.0
    warned = False

    for _, row in g.iterrows():
        q = int(row["quarter"])
        mode = str(row["flow_mode"]).lower()
        val = row[col]

        if pd.isna(val):
            result.append(np.nan)
            continue

        if mode == "single":
            qval = val
        elif mode == "ytd":
            if q == 1:
                qval = val
            elif (q - 1) in quarter_to_raw:
                prev_mode = quarter_to_mode[q - 1]
                prev_raw = quarter_to_raw[q - 1]
                if prev_mode == "ytd" and pd.notna(prev_raw):
                    qval = val - prev_raw
                else:
                    # 上一季度是 single，则需要已有累计 single
                    qval = val - cum_single if pd.notna(cum_single) else np.nan
            else:
                qval = np.nan
                if not warned:
                    warnings.warn(
                        f"[{row['ts_code']}] {row['fiscal_year']}Q{q} 的字段 {col} 标记为 YTD，"
                        f"但缺少上一季度数据，无法准确还原，已记为 NaN。"
                    )
                    warned = True
        else:
            raise ValueError(f"未知 flow_mode: {mode}")

        result.append(qval)

        quarter_to_raw[q] = val
        quarter_to_mode[q] = mode
        if pd.notna(qval):
            if q == 1:
                cum_single = qval
            elif q == 2:
                # 只有在Q1存在或Q2本身 single 的情况下，累计才可信
                if 1 in quarter_to_raw or mode == "single":
                    cum_single = (0.0 if pd.isna(cum_single) else cum_single) + qval
                else:
                    cum_single = np.nan
            else:
                # 对于 Q3/Q4，只要当前单季度能解析，累计沿用
                cum_single = (0.0 if pd.isna(cum_single) else cum_single) + qval

    return pd.Series(result, index=g.index)


def convert_flows_to_single_quarter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["ts_code", "fiscal_year", "quarter", "report_date"]).reset_index(drop=True)

    for col in FLOW_COLUMNS:
        if col not in out.columns:
            continue
        single_col = f"{col}__single"
        out[single_col] = np.nan
        for (ts_code, fiscal_year), idx in out.groupby(["ts_code", "fiscal_year"]).groups.items():
            g = out.loc[idx, ["ts_code", "fiscal_year", "quarter", "flow_mode", col]].copy()
            converted = _convert_group_mixed_flow_to_single(g, col)
            out.loc[converted.index, single_col] = converted

    return out


def fetch_financial_statements_from_db(
    ts_code: str,
    start_date: str = None,
    limit_n_quarters: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    从 SQLite 数据库 stock_financial_summary 表获取财务数据（YTD格式），
    复用水/ytd 混合口径转换逻辑 convert_flows_to_single_quarter，
    输出与 Tushare 格式一致的三张表。

    Args:
        ts_code: 股票代码
        start_date: 起始日期 YYYYMMDD
        limit_n_quarters: 最多读取最近 N 个季度（避免全量拉取，节省内存）
    """
    from sqlalchemy import create_engine, text
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    from config import DATABASE_URL

    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL)

    base_sql = f'''
        SELECT ts_code, stock_name, report_date,
               营业总收入, 营业收入, 营业成本, 营业利润,
               归母净利润, 少数股东损益,
               EBIT, 经营活动现金流净额,
               资本开支, 销售商品提供劳务收到的现金,
               总资产, 应收账款, 存货, 应付账款,
               合同负债, EBITDA, 研发费用
        FROM stock_financial_summary
        WHERE ts_code = :ts_code
        {'AND report_date >= :start_date' if start_date else ''}
    '''

    if limit_n_quarters:
        sql = base_sql + ' ORDER BY report_date DESC LIMIT :limit_n'
    else:
        sql = base_sql + ' ORDER BY report_date ASC'

    params = {"ts_code": ts_code}
    if start_date:
        params["start_date"] = start_date
    if limit_n_quarters:
        params["limit_n"] = limit_n_quarters

    with engine.connect() as conn:
        raw = pd.read_sql(text(sql), conn, params=params)

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw = raw.copy()
    if limit_n_quarters:
        raw = raw.sort_values("report_date").reset_index(drop=True)
    raw["source_file"] = "DB"
    raw["report_date"] = pd.to_datetime(raw["report_date"].astype(str), format="%Y%m%d", errors="coerce")
    raw["flow_mode"] = "ytd"
    raw["fiscal_year"] = raw["report_date"].dt.year
    raw["quarter"] = raw["report_date"].dt.quarter

    for col in REQUIRED_COLUMNS:
        if col in {"ts_code", "stock_name", "report_date"}:
            continue
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = convert_flows_to_single_quarter(raw)
    raw = raw.sort_values(["ts_code", "report_date"]).reset_index(drop=True)

    df_quarterly = pd.DataFrame({
        "ts_code": raw["ts_code"],
        "stock_name": raw["stock_name"],
        "report_date": raw["report_date"].dt.strftime("%Y%m%d"),
        "营业总收入": raw["营业总收入__single"],
        "营业收入": raw["营业收入__single"],
        "营业成本": raw["营业成本__single"],
        "营业利润": raw["营业利润__single"],
        "归母净利润": raw["归母净利润__single"],
        "少数股东损益": raw["少数股东损益__single"],
        "EBIT": raw["EBIT__single"],
        "经营活动现金流净额": raw["经营活动现金流净额__single"],
        "资本开支": raw["资本开支__single"],
        "销售商品提供劳务收到的现金": raw["销售商品提供劳务收到的现金__single"],
        "总资产": raw["总资产"],
        "应收账款": raw["应收账款"],
        "存货": raw["存货"],
        "应付账款": raw["应付账款"],
        "合同负债": raw["合同负债"],
        "EBITDA": raw["EBITDA__single"],
        "研发费用": raw["研发费用__single"],
        "股东权益": raw["股东权益"] if "股东权益" in raw.columns else np.nan,
    })

    df_quarterly["end_date"] = pd.to_datetime(df_quarterly["report_date"], format="%Y%m%d", errors="coerce")

    income = pd.DataFrame({
        "ts_code": df_quarterly["ts_code"],
        "end_date": df_quarterly["end_date"],
        "report_type": 2,
        "ann_date": df_quarterly["report_date"],
        "f_ann_date": df_quarterly["report_date"],
        "total_revenue": df_quarterly["营业总收入"],
        "revenue": df_quarterly["营业收入"],
        "oper_cost": df_quarterly["营业成本"],
        "operate_profit": df_quarterly["营业利润"],
        "ebit": df_quarterly["EBIT"],
        "ebitda": df_quarterly["EBITDA"],
        "n_income": df_quarterly["归母净利润"] + df_quarterly["少数股东损益"],
        "n_income_attr_p": df_quarterly["归母净利润"],
        "rd_exp": df_quarterly["研发费用"],
    })

    cash = pd.DataFrame({
        "ts_code": df_quarterly["ts_code"],
        "end_date": df_quarterly["end_date"],
        "report_type": 2,
        "ann_date": df_quarterly["report_date"],
        "f_ann_date": df_quarterly["report_date"],
        "n_cashflow_act": df_quarterly["经营活动现金流净额"],
        "free_cashflow": np.nan,
        "c_fr_sale_sg": df_quarterly["销售商品提供劳务收到的现金"],
        "c_pay_acq_const_fiolta": df_quarterly["资本开支"],
    })

    bs = pd.DataFrame({
        "ts_code": df_quarterly["ts_code"],
        "end_date": df_quarterly["end_date"],
        "report_type": 1,
        "ann_date": df_quarterly["report_date"],
        "f_ann_date": df_quarterly["report_date"],
        "total_assets": df_quarterly["总资产"],
        "accounts_receiv": df_quarterly["应收账款"],
        "inventories": df_quarterly["存货"],
        "accounts_pay": df_quarterly["应付账款"],
        "contract_liab": df_quarterly["合同负债"],
        "total_hldr_eqy_exc_min_int": df_quarterly.get("股东权益", np.nan),
    })

    return income, cash, bs


def dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    for c in ["ann_date", "f_ann_date", "end_date"]:
        if c in tmp.columns:
            if c == "end_date":
                tmp[c] = pd.to_datetime(tmp[c], errors="coerce")
            else:
                tmp[c] = pd.to_datetime(tmp[c], format="%Y%m%d", errors="coerce")
    sort_cols = [c for c in ["end_date", "f_ann_date", "ann_date"] if c in tmp.columns]
    tmp = tmp.sort_values(sort_cols)
    if "ts_code" in tmp.columns:
        tmp = tmp.groupby(["ts_code", "end_date"], as_index=False).tail(1)
    else:
        tmp = tmp.groupby("end_date", as_index=False).tail(1)
    return tmp.sort_values("end_date").reset_index(drop=True)


def check_quarter_integrity(df: pd.DataFrame) -> None:
    if df.empty:
        return
    tmp = df[["ts_code", "end_date", "fiscal_year", "quarter"]].copy()
    tmp = tmp.sort_values(["ts_code", "end_date"]).reset_index(drop=True)

    dup = tmp.groupby(["ts_code", "fiscal_year", "quarter"]).size()
    dup = dup[dup > 1]
    if not dup.empty:
        warnings.warn(f"发现同一股票同一财年重复季度记录，可能影响 YTD/TTM 计算：{dup.to_dict()}")

    tmp["qnum"] = tmp["fiscal_year"] * 4 + tmp["quarter"]
    tmp["qdiff"] = tmp.groupby("ts_code")["qnum"].diff()
    bad = tmp.loc[tmp["qdiff"].notna() & (tmp["qdiff"] != 1)]
    if not bad.empty:
        for _, row in bad.iterrows():
            qnum = int(row["qnum"])
            prev_qnum = qnum - int(row["qdiff"])
            missing_qnum = prev_qnum + 1
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
    limit_n_quarters: int = None,
) -> pd.DataFrame:
    income, cash, bs = fetch_financial_statements_from_db(
        ts_code=ts_code,
        start_date=start_date,
        limit_n_quarters=limit_n_quarters,
    )
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
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
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
    parser = argparse.ArgumentParser(description="中钨高新季度财务评分脚本")
    parser.add_argument("--ts-code", type=str, default=TARGET_TS_CODE,
                        help="股票代码，默认 000657.SZ")
    parser.add_argument("--name", type=str, default=TARGET_NAME,
                        help="股票简称（仅输出展示用）")
    parser.add_argument("--start", type=str, default="20120101",
                        help="开始日期 YYYYMMDD，默认 20120101")
    parser.add_argument("--lookback", type=int, default=12,
                        help="单股时序打分回看季度数，默认 12")
    parser.add_argument("--outdir", type=str, default=".",
                        help="输出目录，默认当前目录")
    return parser.parse_args()


def main():
    global TARGET_NAME
    args = parse_args()
    TARGET_NAME = args.name

    df = prepare_base_dataframe(ts_code=args.ts_code, start_date=args.start)
    if df.empty:
        raise ValueError(f"未取到 {args.ts_code} 的财务数据，请检查 ts_code、起始日期与数据源。")

    df = add_ytd_and_ttm(df)
    df = add_factors(df)
    scored = score_dataframe(df, lookback=args.lookback)

    summary, factor_df = build_latest_outputs(scored)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    safe_name = args.ts_code.replace(".", "_")
    scored_path = os.path.join(outdir, f"{safe_name}_scores.csv")
    summary_path = os.path.join(outdir, f"{safe_name}_latest_summary.csv")
    factor_path = os.path.join(outdir, f"{safe_name}_latest_factor_scores.csv")

    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    factor_df.to_csv(factor_path, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)

    print(f"=== {TARGET_NAME}（{args.ts_code}）最新报告期总览 ===")
    print(summary.to_string(index=False))

    print(f"\n=== {TARGET_NAME}（{args.ts_code}）最新报告期因子明细 ===")
    print(factor_df.to_string(index=False))

    print("\n输出文件：")
    print(scored_path)
    print(summary_path)
    print(factor_path)


if __name__ == "__main__":
    main()
