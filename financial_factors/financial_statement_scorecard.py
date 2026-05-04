#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_stock_financial_scorecard.py

单股财报边际改善评分卡 - 数据库版本

定位：
- 只做单只股票或每只股票独立的连续季报分析。
- 不做行业排名、不做全市场横截面排名。
- 不使用股价、估值、题材、新闻、技术指标。
- 数据从数据库读取，评分结果保存到数据库。
- 支持批量计算所有股票的历史评分。

核心逻辑：
1. 从数据库读取连续单季财报数据。
2. 计算15项原始财报改善指标。
3. 每个指标按固定连续映射函数转换成该指标权重内得分。
4. 维度分 = 该维度内指标得分求和。
5. 总分 = 五大维度分求和，满分100。
6. 评分结果保存到数据库。

运行示例：
# 单只股票评分
python financial_factors/financial_statement_scorecard.py \
  --ts-code 688012.SH \
  --end-date 20251231

# 批量计算所有股票过去8个报告期
python financial_factors/financial_statement_scorecard.py \
  --batch \
  --quarters 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目根目录到 Python 路径（用于数据库导入）
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)


# ============================================================
# 1. 权重：按用户指定框架
# ============================================================

METRIC_WEIGHTS: Dict[str, float] = {
    # 主业盈利改善：45
    # 说明：无正式扣非归母净利润字段时，保留主业经营利润 proxy，
    # 但所有改善强度必须经过“低基数 + 盈利质量 + 持续性”修正。
    "F1_core_profit_improve_strength": 22.0,
    "F2_core_profit_acceleration": 8.0,
    "F3_parent_np_improve_strength": 8.0,
    "F15_core_profit_persistence": 7.0,

    # 收入与费用效率：18
    "F4_revenue_momentum": 6.0,
    "F5_gross_margin_expansion": 7.0,
    "F6_expense_ratio_improvement": 5.0,

    # 现金流验证：17
    "F7_sales_cash_collection_improvement": 5.0,
    "F8_ocf_improve_strength": 7.0,
    "F9_profit_cash_gap_improvement": 5.0,

    # 营运资本质量：15
    "F10_ar_pressure_improvement": 5.0,
    "F11_inventory_pressure_improvement": 5.0,
    "F12_contract_liab_improvement": 5.0,

    # 投入与资产效率：5
    "F13_capex_efficiency_improvement": 2.0,
    "F14_asset_turnover_improvement": 3.0,
}

DIMENSIONS: Dict[str, List[str]] = {
    "main_profit_score": [
        "F1_core_profit_improve_strength",
        "F2_core_profit_acceleration",
        "F3_parent_np_improve_strength",
        "F15_core_profit_persistence",
    ],
    "revenue_expense_efficiency_score": [
        "F4_revenue_momentum",
        "F5_gross_margin_expansion",
        "F6_expense_ratio_improvement",
    ],
    "cashflow_validation_score": [
        "F7_sales_cash_collection_improvement",
        "F8_ocf_improve_strength",
        "F9_profit_cash_gap_improvement",
    ],
    "working_capital_quality_score": [
        "F10_ar_pressure_improvement",
        "F11_inventory_pressure_improvement",
        "F12_contract_liab_improvement",
    ],
    "investment_asset_efficiency_score": [
        "F13_capex_efficiency_improvement",
        "F14_asset_turnover_improvement",
    ],
}

DIMENSION_WEIGHTS: Dict[str, float] = {
    "main_profit_score": 45.0,
    "revenue_expense_efficiency_score": 18.0,
    "cashflow_validation_score": 17.0,
    "working_capital_quality_score": 15.0,
    "investment_asset_efficiency_score": 5.0,
}


# ============================================================
# 2. 单股固定评分映射参数
# ============================================================
# 解释：
# 每个指标都已经设计为“越大越好”。
# 单股评分不做排名，而是做连续映射：
#
# score_ratio = clip(0.5 + raw_value / (2 * scale), 0, 1)
# score = score_ratio * weight
#
# raw_value = 0 时，得该指标一半分；
# raw_value >= +scale 时，得满分；
# raw_value <= -scale 时，得0分。
#
# scale 表示“足以从中性推到满分/零分的改善幅度”。
# 这些scale不是行业排名，而是单股财报改善的固定经济阈值。

METRIC_SCALES: Dict[str, float] = {
    # 主业盈利改善
    # 利润改善强度除以平均总资产，±2%已经是很强的资产标准化改善。
    "F1_core_profit_improve_strength": 0.020,
    # 加速度通常比改善强度更小，±1.5%作为强信号。
    "F2_core_profit_acceleration": 0.015,
    # 归母净利润改善强度，±2%作为强信号。
    "F3_parent_np_improve_strength": 0.020,
    # 主业利润连续性，用最近2季原始改善强度均值衡量，±2%作为强信号。
    "F15_core_profit_persistence": 0.020,

    # 收入与费用效率
    # 收入动量单位是百分点，±30pct为强加速/强减速。
    "F4_revenue_momentum": 0.300,
    # 毛利率扩张，±5pct为强信号。
    "F5_gross_margin_expansion": 0.050,
    # 期间费用率改善，±3pct为强信号。
    "F6_expense_ratio_improvement": 0.030,

    # 现金流验证
    # 销售收现率改善，±30pct为强信号。
    "F7_sales_cash_collection_improvement": 0.300,
    # OCF改善强度除以平均总资产，±5%为强信号。
    "F8_ocf_improve_strength": 0.050,
    # 利润现金背离压力改善，±3%平均资产为强信号。
    "F9_profit_cash_gap_improvement": 0.030,

    # 营运资本质量
    # 应收/TTM收入压力变化，±10pct为强信号。
    "F10_ar_pressure_improvement": 0.100,
    # 存货/TTM成本压力变化弹性更大，±30pct为强信号。
    "F11_inventory_pressure_improvement": 0.300,
    # 合同负债/TTM收入变化，±10pct为强信号。
    "F12_contract_liab_improvement": 0.100,

    # 投入与资产效率
    # F13 改为“TTM毛利增量 / 过去8季资本开支”的投入产出改善，±20%作为强信号。
    "F13_capex_efficiency_improvement": 0.200,
    # 资产周转率改善，±10pct作为强信号。
    "F14_asset_turnover_improvement": 0.100,
}

NUMERIC_COLS = [
    "total_revenue",
    "revenue",
    "oper_cost",
    "operate_profit",
    "n_income",
    "n_income_attr_p",
    "rd_exp",
    "n_cashflow_act",
    "c_fr_sale_sg",
    "c_pay_acq_const_fiolta",
    "total_assets",
    "accounts_receiv",
    "inventories",
    "accounts_pay",
    "contract_liab",
    "total_hldr_eqy_exc_min_int",
    "fv_value_chg_gain",
    "invest_income",
    "oth_income",
    "asset_disp_income",
    "sell_exp",
    "admin_exp",
    "fin_exp",
    "total_profit",
    "income_tax",
    "minority_gain",
]

CORE_REQUIRED_COLS = [
    "ts_code",
    "end_date",
    "revenue",
    "oper_cost",
    "operate_profit",
    "n_income_attr_p",
    "n_cashflow_act",
    "c_fr_sale_sg",
    "c_pay_acq_const_fiolta",
    "total_assets",
    "accounts_receiv",
    "inventories",
    "contract_liab",
    "sell_exp",
    "admin_exp",
    "fin_exp",
    # Note: 以下字段缺失率较高，作为可选字段处理
    # fv_value_chg_gain 缺失率约40%
    # invest_income 缺失率约8%
    # oth_income 缺失率约1%
    # asset_disp_income 缺失率约22%
    # rd_exp 缺失率约8%
]


# ============================================================
# 3. 工具函数
# ============================================================

def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = b.replace(0, np.nan)
    return a / denom


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan}),
        errors="coerce",
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    aliases = {
        "operating_profit": "operate_profit",
        "parent_np": "n_income_attr_p",
        "ocf": "n_cashflow_act",
        "cash_received_from_sales": "c_fr_sale_sg",
        "capex_paid": "c_pay_acq_const_fiolta",
        "ar": "accounts_receiv",
        "inventory": "inventories",
        "contract_liabilities": "contract_liab",
        "selling_expense": "sell_exp",
        "admin_expense": "admin_exp",
        "financial_expense": "fin_exp",
        "r_and_d_expense": "rd_exp",
        "fv_gain": "fv_value_chg_gain",
        "investment_income": "invest_income",
        "other_income": "oth_income",
        "asset_disposal_income": "asset_disp_income",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "revenue" not in df.columns and "total_revenue" in df.columns:
        df["revenue"] = df["total_revenue"]

    return df


def read_from_database(ts_code: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """从数据库读取单只股票的连续季度数据"""
    from datasource.database import get_session
    from sqlalchemy import text

    with get_session() as session:
        # 查询该股票所有历史数据（用于TTM计算）
        sql = """
            SELECT
                ts_code, end_date, ann_date, report_type,
                revenue, oper_cost, operate_profit, n_income, n_income_attr_p,
                rd_exp, n_cashflow_act, c_fr_sale_sg, c_pay_acq_const_fiolta,
                total_assets, accounts_receiv, inventories, accounts_pay, contract_liab,
                total_hldr_eqy_exc_min_int, fv_value_chg_gain, invest_income, oth_income,
                asset_disp_income, sell_exp, admin_exp, fin_exp, total_profit,
                income_tax, minority_gain
            FROM financial_quarterly_data
            WHERE ts_code = :ts_code
              AND (:end_date IS NULL OR end_date <= :end_date)
            ORDER BY end_date ASC
        """
        result = session.execute(text(sql), {"ts_code": ts_code, "end_date": end_date})

        # 转换为DataFrame
        columns = result.keys()
        rows = result.fetchall()

        if not rows:
            raise ValueError(f"未找到股票 {ts_code} 的数据")

        df = pd.DataFrame(rows, columns=columns)

        # 数值字段转换
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = to_numeric_series(df[col])

        # 日期处理
        df["end_date"] = df["end_date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
        df["end_date_dt"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")

        # 设置名称
        if "name" not in df.columns:
            df["name"] = ts_code

        return df


def warn_missing_core_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in CORE_REQUIRED_COLS if c not in df.columns or df[c].isna().all()]


def continuous_score(raw: pd.Series, weight: float, scale: float) -> pd.Series:
    """
    单股连续评分函数。
    raw = 0 -> 0.5 * weight
    raw >= scale -> weight
    raw <= -scale -> 0
    """
    if scale <= 0:
        raise ValueError("scale必须为正数")
    ratio = 0.5 + raw / (2.0 * scale)
    ratio = ratio.clip(lower=0.0, upper=1.0)
    return ratio * weight


# ============================================================
# 4. 基础派生字段
# ============================================================

def add_base_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["fv_value_chg_gain", "invest_income", "oth_income", "asset_disp_income"]:
        df[col] = df[col].fillna(0)

    # 主业经营利润proxy：不是正式扣非归母净利润
    df["core_operating_profit_proxy"] = (
        df["operate_profit"]
        - df["fv_value_chg_gain"]
        - df["invest_income"]
        - df["oth_income"]
        - df["asset_disp_income"]
    )

    df["gross_profit"] = df["revenue"] - df["oper_cost"]
    df["gross_margin"] = safe_div(df["gross_profit"], df["revenue"])
    df["core_margin"] = safe_div(df["core_operating_profit_proxy"], df["revenue"])

    for col in ["sell_exp", "admin_exp", "fin_exp", "rd_exp"]:
        df[col] = df[col].fillna(0)

    # 费用率评分只评价销售+管理费用率。
    # 财务费用可能包含利息收入/汇兑收益，现金充裕公司 fin_exp 可能为负，
    # 若纳入主评分容易导致费用率被异常拉低，因此 fin_exp 仅作为观察项。
    # 研发费用率也仅作为观察项，不直接惩罚研发投入。
    df["period_expense"] = df["sell_exp"] + df["admin_exp"] + df["fin_exp"] + df["rd_exp"]
    df["controllable_expense"] = df["sell_exp"] + df["admin_exp"]
    df["expense_ratio"] = safe_div(df["controllable_expense"], df["revenue"])
    df["fin_expense_ratio"] = safe_div(df["fin_exp"], df["revenue"])
    df["rd_expense_ratio"] = safe_div(df["rd_exp"], df["revenue"])

    df["sales_cash_collection_rate_single"] = safe_div(df["c_fr_sale_sg"], df["revenue"])
    df["capex_intensity_single"] = safe_div(df["c_pay_acq_const_fiolta"], df["revenue"])

    # 仅辅助观察，不参与评分
    tax_rate = safe_div(df["income_tax"], df["total_profit"]).clip(lower=0, upper=0.5)
    parent_share = safe_div(df["n_income_attr_p"], df["n_income"]).clip(lower=0, upper=1.5)
    df["estimated_core_parent_profit"] = (
        df["core_operating_profit_proxy"] * (1 - tax_rate) * parent_share
    )

    return df


def add_ttm_and_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ts_code", group_keys=False)

    rolling_sum_cols = {
        "revenue": "ttm_revenue",
        "oper_cost": "ttm_oper_cost",
        "gross_profit": "ttm_gross_profit",
        "core_operating_profit_proxy": "ttm_core_operating_profit_proxy",
        "n_income_attr_p": "ttm_parent_np",
        "n_cashflow_act": "ttm_ocf",
        "c_fr_sale_sg": "ttm_cash_received_from_sales",
        "c_pay_acq_const_fiolta": "ttm_capex",
    }

    for src, dst in rolling_sum_cols.items():
        df[dst] = (
            g[src]
            .rolling(window=4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )

    lag_cols = [
        "revenue", "oper_cost", "gross_profit", "core_operating_profit_proxy", "n_income_attr_p",
        "n_cashflow_act", "gross_margin", "core_margin", "expense_ratio", "fin_expense_ratio", "rd_expense_ratio", "total_assets",
        "accounts_receiv", "inventories", "contract_liab",
        "sales_cash_collection_rate_single", "capex_intensity_single",
        "ttm_revenue", "ttm_oper_cost", "ttm_gross_profit", "ttm_core_operating_profit_proxy",
        "ttm_parent_np", "ttm_ocf", "ttm_cash_received_from_sales", "ttm_capex",
    ]

    # 研发费用观察项滞后（不参与评分，仅观察）
    df["rd_expense_ratio_lag4"] = g["rd_expense_ratio"].shift(4)

    for col in lag_cols:
        df[f"{col}_lag1"] = g[col].shift(1)
        df[f"{col}_lag4"] = g[col].shift(4)

    df["avg_assets_yoy"] = (df["total_assets"] + df["total_assets_lag4"]) / 2

    df["sales_cash_collection_rate_ttm"] = safe_div(
        df["ttm_cash_received_from_sales"], df["ttm_revenue"]
    )
    df["capex_intensity_ttm"] = safe_div(df["ttm_capex"], df["ttm_revenue"])
    df["asset_turnover_ttm"] = safe_div(df["ttm_revenue"], df["avg_assets_yoy"])

    df["ar_pressure"] = safe_div(df["accounts_receiv"], df["ttm_revenue"])
    df["inventory_pressure"] = safe_div(df["inventories"], df["ttm_oper_cost"])
    df["contract_liab_ratio"] = safe_div(df["contract_liab"], df["ttm_revenue"])

    df["profit_cash_gap_pressure"] = safe_div(
        np.maximum(df["ttm_core_operating_profit_proxy"] - df["ttm_ocf"], 0),
        df["avg_assets_yoy"],
    )

    derived_lag_cols = [
        "sales_cash_collection_rate_ttm",
        "capex_intensity_ttm",
        "asset_turnover_ttm",
        "ar_pressure",
        "inventory_pressure",
        "contract_liab_ratio",
        "profit_cash_gap_pressure",
    ]

    for col in derived_lag_cols:
        df[f"{col}_lag1"] = g[col].shift(1)
        df[f"{col}_lag4"] = g[col].shift(4)

    # 低基数保护：用去年同期收入相对自身历史中枢的位置，压低“低基数同比改善”。
    df["revenue_lag4_median_12"] = (
        g["revenue_lag4"]
        .rolling(window=12, min_periods=6)
        .median()
        .reset_index(level=0, drop=True)
    )
    df["revenue_base_coef"] = safe_div(
        df["revenue_lag4"], df["revenue_lag4_median_12"]
    ).clip(lower=0.3, upper=1.0)

    df["ttm_revenue_yoy_growth"] = safe_div(df["ttm_revenue"], df["ttm_revenue_lag4"]) - 1
    df["ttm_revenue_trend_coef"] = (
        (safe_div(df["ttm_revenue"], df["ttm_revenue_lag4"]) - 0.90) / 0.20
    ).clip(lower=0.0, upper=1.0)

    # 过去8季资本开支与收入，用于 F13 投入产出改善。资本开支低于0时按0处理，避免退款/冲回扰动。
    df["capex_paid_positive"] = df["c_pay_acq_const_fiolta"].clip(lower=0)
    df["rolling_8q_capex"] = (
        g["capex_paid_positive"]
        .rolling(window=8, min_periods=6)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["rolling_8q_revenue"] = (
        g["revenue"]
        .rolling(window=8, min_periods=6)
        .sum()
        .reset_index(level=0, drop=True)
    )
    df["capex_base_8q"] = safe_div(df["rolling_8q_capex"], df["rolling_8q_revenue"])
    df["capex_valid_coef"] = ((df["capex_base_8q"] - 0.01) / 0.02).clip(lower=0.0, upper=1.0)

    return df


# ============================================================
# 5. 原始指标
# ============================================================

def add_raw_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ts_code", group_keys=False)

    # F1：主业经营利润 proxy 的同比改善强度，但加入低基数与盈利质量修正。
    df["F1_core_profit_improve_strength_raw"] = safe_div(
        df["core_operating_profit_proxy"] - df["core_operating_profit_proxy_lag4"],
        df["avg_assets_yoy"],
    )
    df["profit_quality_coef"] = ((df["core_margin"] + 0.02) / 0.07).clip(lower=0.0, upper=1.0)
    df["F1_core_profit_improve_strength"] = (
        df["F1_core_profit_improve_strength_raw"]
        * df["profit_quality_coef"]
        * df["revenue_base_coef"]
    )

    # F2：加速度仍保留，但基于修正后的 F1，且权重已下调。
    df["F2_core_profit_acceleration"] = (
        df["F1_core_profit_improve_strength"]
        - g["F1_core_profit_improve_strength"].shift(1)
    )

    # F3：归母净利润改善强度，加入与主业 proxy 的一致性修正，避免非经常性扰动拉高分数。
    df["F3_parent_np_improve_strength_raw"] = safe_div(
        df["n_income_attr_p"] - df["n_income_attr_p_lag4"],
        df["avg_assets_yoy"],
    )
    parent_np_abs = df["n_income_attr_p"].abs().replace(0, np.nan)
    df["parent_np_quality_gap"] = safe_div(
        (df["n_income_attr_p"] - df["estimated_core_parent_profit"]).abs(),
        parent_np_abs,
    )
    # 弱化后的 parent_np_quality_coef：gap 0->1.0, gap 0.3->0.85, gap 1.0->0.7
    df["parent_np_quality_coef"] = (1.0 - df["parent_np_quality_gap"] * 0.3).clip(lower=0.7, upper=1.0).fillna(0.85)
    df["F3_parent_np_improve_strength"] = (
        df["F3_parent_np_improve_strength_raw"]
        * df["parent_np_quality_coef"]
        * df["revenue_base_coef"]
    )

    # F15：新增主业利润连续性，用最近2季“原始改善强度”均值衡量，抑制一季脉冲。
    df["F15_core_profit_persistence"] = (
        g["F1_core_profit_improve_strength_raw"]
        .rolling(window=2, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # F4：收入动量加入低基数与 TTM 收入趋势修正。
    df["revenue_yoy_growth"] = safe_div(df["revenue"], df["revenue_lag4"]) - 1
    df["F4_revenue_momentum_raw"] = df["revenue_yoy_growth"] - g["revenue_yoy_growth"].shift(1)
    df["F4_revenue_momentum"] = (
        df["F4_revenue_momentum_raw"]
        * df["revenue_base_coef"]
        * df["ttm_revenue_trend_coef"]
    )

    df["F5_gross_margin_expansion"] = df["gross_margin"] - df["gross_margin_lag4"]

    # F6：只评价销售+管理费用率改善；财务费用率与研发费用率作为观察项，不直接参与评分。
    df["F6_expense_ratio_improvement"] = df["expense_ratio_lag4"] - df["expense_ratio"]

    df["F7_sales_cash_collection_improvement"] = (
        df["sales_cash_collection_rate_ttm"] - df["sales_cash_collection_rate_ttm_lag4"]
    )

    df["F8_ocf_improve_strength"] = safe_div(
        df["ttm_ocf"] - df["ttm_ocf_lag4"],
        df["avg_assets_yoy"],
    )

    df["F9_profit_cash_gap_improvement"] = (
        df["profit_cash_gap_pressure_lag4"] - df["profit_cash_gap_pressure"]
    )

    df["F10_ar_pressure_improvement"] = df["ar_pressure_lag4"] - df["ar_pressure"]

    df["F11_inventory_pressure_improvement"] = (
        df["inventory_pressure_lag4"] - df["inventory_pressure"]
    )

    # F12：合同负债改善改为“加权质量折扣”，避免收入/毛利/现金流三重乘法惩罚过重。
    # 收入趋势、毛利变化、现金流改善都用于确认合同负债的质量，但最低只折扣到0.5。
    df["F12_contract_liab_improvement_raw"] = (
        df["contract_liab_ratio"] - df["contract_liab_ratio_lag4"]
    )
    df["contract_liab_revenue_coef"] = df["ttm_revenue_trend_coef"].clip(lower=0.0, upper=1.0)
    # 毛利率确认系数改成 0.7–1.0 的轻折扣：毛利率扩张-5%->0.7, 0%->0.85, +5%->1.0
    df["contract_liab_gm_coef"] = (
        (df["F5_gross_margin_expansion"] + 0.05) / 0.05 * 0.3 + 0.7
    ).clip(lower=0.7, upper=1.0)
    df["contract_liab_cashflow_coef"] = (
        (df["F8_ocf_improve_strength"] + 0.02) / 0.02
    ).clip(lower=0.0, upper=1.0)
    df["contract_liab_quality_coef"] = (
        0.4 * df["contract_liab_revenue_coef"]
        + 0.3 * df["contract_liab_gm_coef"]
        + 0.3 * df["contract_liab_cashflow_coef"]
    ).clip(lower=0.5, upper=1.0)
    df["F12_contract_liab_improvement"] = (
        df["F12_contract_liab_improvement_raw"] * df["contract_liab_quality_coef"]
    )

    # F13：投入产出改善 = TTM毛利增量 / 过去8季累计资本开支。
    # 低 capex 公司通过 capex_valid_coef 压低，避免小分母爆分。
    df["F13_capex_efficiency_improvement_raw"] = safe_div(
        df["ttm_gross_profit"] - df["ttm_gross_profit_lag4"],
        df["rolling_8q_capex"],
    ).clip(lower=-0.3, upper=0.3)
    df["F13_capex_efficiency_improvement"] = (
        df["F13_capex_efficiency_improvement_raw"] * df["capex_valid_coef"]
    )

    df["F14_asset_turnover_improvement"] = (
        df["asset_turnover_ttm"] - df["asset_turnover_ttm_lag4"]
    )

    # 研发费用观察项（不参与评分）
    df["RD_obs_expense_ratio_change"] = df["rd_expense_ratio"] - df["rd_expense_ratio_lag4"]

    # 轻资产公司标记（8季资本开支/收入 < 1%）
    df["is_light_asset"] = df["capex_base_8q"] < 0.01

    return df


# ============================================================
# 6. 单股指标得分
# ============================================================

def add_single_stock_scores(
    df: pd.DataFrame,
    min_valid_weight_ratio: float = 0.70,
) -> pd.DataFrame:
    df = df.copy()

    def get_f_number(metric_name: str) -> int:
        parts = metric_name.split('_')
        if parts and parts[0].startswith('F') and parts[0][1:].isdigit():
            return int(parts[0][1:])
        raise ValueError(f"无法从指标名提取F编号: {metric_name}")

    # 按指标名中的 F 编号生成得分列，避免新增 F15 后因字典顺序导致 F4_score 等列错位。
    for metric, weight in METRIC_WEIGHTS.items():
        scale = METRIC_SCALES[metric]
        f_num = get_f_number(metric)
        score_col = f"F{f_num}_score"
        ratio_col = f"{metric}_score_ratio"

        # F13 轻资产公司特殊处理：给中性分（0.5 * weight）
        if metric == "F13_capex_efficiency_improvement":
            df[score_col] = np.where(
                df["is_light_asset"] & (df[metric].isna() | (df["capex_valid_coef"] == 0)),
                weight * 0.5,  # 轻资产公司给中性分
                continuous_score(df[metric], weight=weight, scale=scale)
            )
        else:
            df[score_col] = continuous_score(df[metric], weight=weight, scale=scale)
        df[ratio_col] = df[score_col] / weight

    # 维度分：按该维度可计算指标权重归一化到维度满分
    for dim, metrics in DIMENSIONS.items():
        score_cols = [f"F{get_f_number(m)}_score" for m in metrics]
        dim_full_weight = DIMENSION_WEIGHTS[dim]

        raw_sum = df[score_cols].sum(axis=1, min_count=1)

        valid_weight = pd.Series(0.0, index=df.index)
        for m in metrics:
            f_num = get_f_number(m)
            valid_weight += df[f"F{f_num}_score"].notna().astype(float) * METRIC_WEIGHTS[m]

        df[f"{dim}_raw"] = raw_sum
        df[f"{dim}_valid_weight"] = valid_weight
        df[dim] = np.where(
            valid_weight > 0,
            raw_sum / valid_weight * dim_full_weight,
            np.nan,
        )

    all_score_cols = [f"F{get_f_number(m)}_score" for m in METRIC_WEIGHTS]
    df["raw_score_sum"] = df[all_score_cols].sum(axis=1, min_count=1)

    df["valid_weight"] = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        f_num = get_f_number(metric)
        df["valid_weight"] += df[f"F{f_num}_score"].notna().astype(float) * weight

    total_weight = sum(METRIC_WEIGHTS.values())
    df["valid_weight_ratio"] = df["valid_weight"] / total_weight

    df["total_score"] = np.where(
        df["valid_weight"] > 0,
        df["raw_score_sum"] / df["valid_weight"] * 100,
        np.nan,
    )

    df["score_status"] = np.where(
        df["valid_weight_ratio"] >= min_valid_weight_ratio,
        "valid",
        "insufficient_data",
    )

    # 核心指标缺失时，不输出总分；避免缺少盈利/现金流核心指标后被归一化抬高。
    core_required_metrics = [
        "F1_core_profit_improve_strength",
        "F3_parent_np_improve_strength",
        "F8_ocf_improve_strength",
    ]
    missing_core_metric = df[core_required_metrics].isna().any(axis=1)
    df.loc[(df["score_status"] == "valid") & missing_core_metric, "score_status"] = "insufficient_core_data"

    df.loc[df["score_status"] != "valid", "total_score"] = np.nan

    # 缺失值惩罚：可计算权重不足但仍达标时，不再完全归一化到100分。
    valid_mask = df["score_status"] == "valid"
    df.loc[valid_mask & (df["valid_weight_ratio"] < 0.85), "total_score"] *= 0.95
    df.loc[valid_mask & (df["valid_weight_ratio"] >= 0.85) & (df["valid_weight_ratio"] < 0.95), "total_score"] *= 0.98

    # 总分调整：从“普遍硬封顶”改为“折扣为主、严重红旗才硬封顶”。
    # 这样可以防止低基数/现金流背离虚高，同时避免误伤真实反转公司。
    df["score_cap_reason"] = ""
    df["score_penalty_coef"] = 1.0

    core_loss_penalty = (df["core_operating_profit_proxy"] < 0) | (
        (df["ttm_core_operating_profit_proxy"] < 0) & (df["core_operating_profit_proxy"] <= 0)
    )
    single_quarter_turnaround_penalty = (
        (df["core_operating_profit_proxy"] > 0)
        & (df["ttm_core_operating_profit_proxy"] < 0)
    )
    low_base_revenue_penalty = df["revenue_base_coef"] < 0.5
    cashflow_divergence_penalty = (
        (df["ttm_core_operating_profit_proxy"] > 0)
        & (df["ttm_ocf"] < 0)
    )

    df.loc[valid_mask & core_loss_penalty, "score_penalty_coef"] *= 0.92
    df.loc[valid_mask & core_loss_penalty, "score_cap_reason"] += "core_loss_penalty;"

    df.loc[valid_mask & single_quarter_turnaround_penalty, "score_penalty_coef"] *= 0.92
    df.loc[valid_mask & single_quarter_turnaround_penalty, "score_cap_reason"] += "single_quarter_turnaround_penalty;"

    df.loc[valid_mask & low_base_revenue_penalty, "score_penalty_coef"] *= 0.90
    df.loc[valid_mask & low_base_revenue_penalty, "score_cap_reason"] += "low_base_revenue_penalty;"

    df.loc[valid_mask & cashflow_divergence_penalty, "score_penalty_coef"] *= 0.90
    df.loc[valid_mask & cashflow_divergence_penalty, "score_cap_reason"] += "cashflow_divergence_penalty;"

    # 避免多个轻度折扣连乘后过度压分。
    df["score_penalty_raw"] = df["score_penalty_coef"].copy()  # 原始折扣（未截断）
    df.loc[valid_mask, "score_penalty_coef"] = df.loc[valid_mask, "score_penalty_coef"].clip(lower=0.75, upper=1.0)
    df["penalty_truncated_flag"] = df["score_penalty_raw"] != df["score_penalty_coef"]  # 是否被截断
    df.loc[valid_mask, "total_score"] = (
        df.loc[valid_mask, "total_score"] * df.loc[valid_mask, "score_penalty_coef"]
    )

    # 严重红旗才硬封顶：主业TTM亏损、OCF为负，且营运资本恶化。
    severe_core_red_flag_cap = (
        (df["ttm_core_operating_profit_proxy"] < 0)
        & (df["ttm_ocf"] < 0)
        & ((df["F10_ar_pressure_improvement"] < 0) | (df["F11_inventory_pressure_improvement"] < 0))
    )

    # 主业TTM盈利但OCF为负，同时利润现金背离压力恶化，才作为严重现金流背离封顶。
    severe_cashflow_divergence_cap = (
        (df["ttm_core_operating_profit_proxy"] > 0)
        & (df["ttm_ocf"] < 0)
        & (df["F9_profit_cash_gap_improvement"] < 0)
    )

    df.loc[valid_mask & severe_core_red_flag_cap, "total_score"] = np.minimum(
        df.loc[valid_mask & severe_core_red_flag_cap, "total_score"], 65
    )
    df.loc[valid_mask & severe_core_red_flag_cap, "score_cap_reason"] += "severe_core_red_flag_cap;"

    df.loc[valid_mask & severe_cashflow_divergence_cap, "total_score"] = np.minimum(
        df.loc[valid_mask & severe_cashflow_divergence_cap, "total_score"], 70
    )
    df.loc[valid_mask & severe_cashflow_divergence_cap, "score_cap_reason"] += "severe_cashflow_divergence_cap;"

    return df


# ============================================================
# 7. 标记、字典与输出
# ============================================================

def add_flags_and_notes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 报表类型标记（2=母公司报表，1=合并报表）
    df["is_parent_report"] = df["report_type"].astype(str) == "2"

    df["cashflow_red_flag"] = (
        (df["ttm_core_operating_profit_proxy"] > 0)
        & (df["ttm_ocf"] < 0)
        & (df["profit_cash_gap_pressure"] > 0)
    )

    df["ar_pressure_worsen_flag"] = df["F10_ar_pressure_improvement"] < 0
    df["inventory_pressure_worsen_flag"] = df["F11_inventory_pressure_improvement"] < 0
    df["core_profit_worsen_flag"] = (
        (df["F1_core_profit_improve_strength"] < 0)
        & (df["F2_core_profit_acceleration"] < 0)
    )

    metrics = list(METRIC_WEIGHTS.keys())
    df["missing_metric_count"] = df[metrics].isna().sum(axis=1)
    df["available_metric_count"] = len(metrics) - df["missing_metric_count"]

    return df


def make_metric_dictionary() -> pd.DataFrame:
    rows = [
        ("F1_core_profit_improve_strength", 22, 0.020, "修正后主业经营利润改善强度", "[(本季主业经营利润proxy - 去年同期主业经营利润proxy) / 平均总资产] × 盈利质量系数 × 收入基数系数"),
        ("F2_core_profit_acceleration", 8, 0.015, "主业经营利润加速度", "本期修正后F1 - 上一季修正后F1；权重下调以降低二阶噪声"),
        ("F3_parent_np_improve_strength", 8, 0.020, "归母净利润改善强度", "[(本季归母净利润 - 去年同期归母净利润) / 平均总资产] × 主业一致性系数 × 盈利质量系数 × 收入基数系数"),
        ("F15_core_profit_persistence", 7, 0.020, "主业经营利润连续性", "最近2季F1_raw均值，用于抑制一季脉冲"),
        ("F4_revenue_momentum", 6, 0.300, "修正后收入动量", "(本季收入同比增速 - 上季收入同比增速) × 收入基数系数 × TTM收入趋势系数"),
        ("F5_gross_margin_expansion", 7, 0.050, "毛利率扩张", "本季毛利率 - 去年同期毛利率"),
        ("F6_expense_ratio_improvement", 5, 0.030, "费用率改善", "去年同期销售+管理费用率 - 本季销售+管理费用率；财务费用率和研发费用率仅观察"),
        ("F7_sales_cash_collection_improvement", 5, 0.300, "销售收现率改善", "本期TTM销售收现率 - 去年同期TTM销售收现率"),
        ("F8_ocf_improve_strength", 7, 0.050, "经营现金流改善强度", "(TTM OCF - 去年同期TTM OCF) / 平均总资产"),
        ("F9_profit_cash_gap_improvement", 5, 0.030, "利润现金背离改善", "去年同期利润现金背离压力 - 本期利润现金背离压力"),
        ("F10_ar_pressure_improvement", 5, 0.100, "应收压力改善", "去年同期应收/TTM收入 - 本期应收/TTM收入"),
        ("F11_inventory_pressure_improvement", 5, 0.300, "存货压力改善", "去年同期存货/TTM营业成本 - 本期存货/TTM营业成本"),
        ("F12_contract_liab_improvement", 5, 0.100, "合同负债改善", "(本期合同负债/TTM收入 - 去年同期合同负债/TTM收入) × 加权质量折扣系数；收入40%+毛利30%+现金流30%，最低0.5"),
        ("F13_capex_efficiency_improvement", 2, 0.200, "投入产出改善", "[(TTM毛利 - 去年同期TTM毛利) / 过去8季累计资本开支] × capex有效性系数"),
        ("F14_asset_turnover_improvement", 3, 0.100, "资产周转率改善", "本期TTM收入/平均总资产 - 去年同期同口径"),
    ]
    df = pd.DataFrame(rows, columns=["metric", "weight", "scale", "name_cn", "formula"])
    df["score_rule"] = "score = weight * clip(0.5 + raw_value / (2 * scale), 0, 1)"
    return df


def select_output_columns(df: pd.DataFrame) -> List[str]:
    base_cols = [
        "ts_code", "name", "end_date", "ann_date",
        "score_status", "valid_weight_ratio", "total_score",
        "main_profit_score", "revenue_expense_efficiency_score",
        "cashflow_validation_score", "working_capital_quality_score",
        "investment_asset_efficiency_score",
    ]

    raw_cols = list(METRIC_WEIGHTS.keys())
    score_cols = []
    for m in METRIC_WEIGHTS:
        f_num = int(m.split('_')[0][1:])
        score_cols.extend([f"F{f_num}_score", f"{m}_score_ratio"])

    derived_cols = [
        "core_operating_profit_proxy", "estimated_core_parent_profit",
        "gross_profit", "gross_margin", "core_margin", "profit_quality_coef",
        "revenue_base_coef", "ttm_revenue_trend_coef", "parent_np_quality_coef",
        "expense_ratio", "fin_expense_ratio", "rd_expense_ratio",
        "sales_cash_collection_rate_ttm", "capex_intensity_ttm",
        "asset_turnover_ttm",
        "ar_pressure", "inventory_pressure", "contract_liab_ratio",
        "profit_cash_gap_pressure",
        "ttm_revenue", "ttm_gross_profit", "ttm_core_operating_profit_proxy",
        "ttm_parent_np", "ttm_ocf", "rolling_8q_capex", "capex_base_8q", "capex_valid_coef", "contract_liab_quality_coef", "score_penalty_raw", "score_penalty_coef", "penalty_truncated_flag", "score_cap_reason",
        "cashflow_red_flag", "ar_pressure_worsen_flag",
        "inventory_pressure_worsen_flag", "core_profit_worsen_flag",
        "available_metric_count", "missing_metric_count",
        # 新增观察项和标记
        "RD_obs_expense_ratio_change", "is_light_asset", "is_parent_report",
    ]

    cols = []
    for c in base_cols + raw_cols + score_cols + derived_cols:
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def latest_per_stock(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.sort_values(["ts_code", "end_date_dt"])
    latest_idx = valid.groupby("ts_code")["end_date_dt"].idxmax()
    return valid.loc[latest_idx].sort_values("total_score", ascending=False, na_position="last")


# ============================================================
# 8. 主程序
# ============================================================

def init_financial_scores_table():
    """初始化评分结果表"""
    from datasource.database import get_session
    from sqlalchemy import text

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS financial_scores (
        id SERIAL PRIMARY KEY,
        ts_code VARCHAR(20) NOT NULL,
        end_date VARCHAR(8) NOT NULL,
        ann_date VARCHAR(8),
        total_score DECIMAL(10,4),
        score_status VARCHAR(20),
        valid_weight_ratio DECIMAL(5,4),
        main_profit_score DECIMAL(10,4),
        revenue_expense_efficiency_score DECIMAL(10,4),
        cashflow_validation_score DECIMAL(10,4),
        working_capital_quality_score DECIMAL(10,4),
        investment_asset_efficiency_score DECIMAL(10,4),
        F1_core_profit_improve_strength DECIMAL(15,8),
        F2_core_profit_acceleration DECIMAL(15,8),
        F3_parent_np_improve_strength DECIMAL(15,8),
        F4_revenue_momentum DECIMAL(15,8),
        F5_gross_margin_expansion DECIMAL(15,8),
        F6_expense_ratio_improvement DECIMAL(15,8),
        F7_sales_cash_collection_improvement DECIMAL(15,8),
        F8_ocf_improve_strength DECIMAL(15,8),
        F9_profit_cash_gap_improvement DECIMAL(15,8),
        F10_ar_pressure_improvement DECIMAL(15,8),
        F11_inventory_pressure_improvement DECIMAL(15,8),
        F12_contract_liab_improvement DECIMAL(15,8),
        F13_capex_efficiency_improvement DECIMAL(15,8),
        F14_asset_turnover_improvement DECIMAL(15,8),
        F15_core_profit_persistence DECIMAL(15,8),
        F1_score DECIMAL(10,4),
        F2_score DECIMAL(10,4),
        F3_score DECIMAL(10,4),
        F4_score DECIMAL(10,4),
        F5_score DECIMAL(10,4),
        F6_score DECIMAL(10,4),
        F7_score DECIMAL(10,4),
        F8_score DECIMAL(10,4),
        F9_score DECIMAL(10,4),
        F10_score DECIMAL(10,4),
        F11_score DECIMAL(10,4),
        F12_score DECIMAL(10,4),
        F13_score DECIMAL(10,4),
        F14_score DECIMAL(10,4),
        F15_score DECIMAL(10,4),
        cashflow_red_flag BOOLEAN,
        ar_pressure_worsen_flag BOOLEAN,
        inventory_pressure_worsen_flag BOOLEAN,
        core_profit_worsen_flag BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ts_code, end_date)
    )
    """

    create_index_sql = """
    CREATE INDEX IF NOT EXISTS idx_financial_scores_ts_code ON financial_scores(ts_code);
    CREATE INDEX IF NOT EXISTS idx_financial_scores_end_date ON financial_scores(end_date);
    """

    alter_table_sql = """
    ALTER TABLE financial_scores ADD COLUMN IF NOT EXISTS F15_core_profit_persistence DECIMAL(15,8);
    ALTER TABLE financial_scores ADD COLUMN IF NOT EXISTS F15_score DECIMAL(10,4);
    ALTER TABLE financial_scores ADD COLUMN IF NOT EXISTS RD_obs_expense_ratio_change DECIMAL(15,8);
    ALTER TABLE financial_scores ADD COLUMN IF NOT EXISTS is_light_asset BOOLEAN;
    ALTER TABLE financial_scores ADD COLUMN IF NOT EXISTS is_parent_report BOOLEAN;
    ALTER TABLE financial_scores ADD COLUMN IF NOT EXISTS score_penalty_raw DECIMAL(5,4);
    ALTER TABLE financial_scores ADD COLUMN IF NOT EXISTS score_penalty_coef DECIMAL(5,4);
    ALTER TABLE financial_scores ADD COLUMN IF NOT EXISTS penalty_truncated_flag BOOLEAN;
    """

    with get_session() as session:
        session.execute(text(create_table_sql))
        session.execute(text(create_index_sql))
        session.execute(text(alter_table_sql))
        session.commit()
        print("✅ financial_scores 表初始化完成")


def save_scores_to_database(df: pd.DataFrame) -> int:
    """将评分结果保存到数据库"""
    from datasource.database import get_session, bulk_upsert

    # 准备保存的数据
    save_cols = [
        "ts_code", "end_date", "ann_date",
        "total_score", "score_status", "valid_weight_ratio",
        "main_profit_score", "revenue_expense_efficiency_score",
        "cashflow_validation_score", "working_capital_quality_score",
        "investment_asset_efficiency_score",
    ]

    # 添加指标原始值与得分。按 METRIC_WEIGHTS 动态保存，避免新增 F15 后遗漏。
    save_cols.extend(list(METRIC_WEIGHTS.keys()))

    for m in METRIC_WEIGHTS:
        f_num = int(m.split('_')[0][1:])
        save_cols.append(f"F{f_num}_score")

    # 添加风险标记和新增观察项
    save_cols.extend([
        "cashflow_red_flag", "ar_pressure_worsen_flag",
        "inventory_pressure_worsen_flag", "core_profit_worsen_flag",
        "RD_obs_expense_ratio_change", "is_light_asset", "is_parent_report",
        "score_penalty_raw", "score_penalty_coef", "penalty_truncated_flag",
    ])

    # 筛选存在的列
    available_cols = [c for c in save_cols if c in df.columns]
    save_df = df[available_cols].copy()

    # 处理布尔值
    for col in ["cashflow_red_flag", "ar_pressure_worsen_flag",
                "inventory_pressure_worsen_flag", "core_profit_worsen_flag",
                "is_light_asset", "is_parent_report", "penalty_truncated_flag"]:
        if col in save_df.columns:
            save_df[col] = save_df[col].astype(bool)

    # 处理NaN值
    save_df = save_df.where(pd.notnull(save_df), None)

    # 保存到数据库
    with get_session() as session:
        bulk_upsert(session, "financial_scores", save_df, unique_keys=["ts_code", "end_date"])

    return len(save_df)


def calculate_stock_scores(ts_code: str, min_valid_weight_ratio: float = 0.70) -> pd.DataFrame:
    """计算单只股票的所有历史评分"""
    print(f"\n处理股票: {ts_code}")

    # 从数据库读取数据
    df = read_from_database(ts_code)
    print(f"  读取完成，共 {len(df)} 条季度记录")

    if len(df) < 5:
        print(f"  ⚠️ 数据不足（少于5个季度），跳过")
        return pd.DataFrame()

    missing_core = warn_missing_core_cols(df)
    if missing_core:
        print(f"  ⚠️ 核心字段缺失: {missing_core}")

    # 计算评分
    df = add_base_fields(df)
    df = add_ttm_and_lags(df)
    df = add_raw_metrics(df)
    df = add_single_stock_scores(df, min_valid_weight_ratio=min_valid_weight_ratio)
    df = add_flags_and_notes(df)

    return df


def batch_calculate_scores(quarters: int = 8, min_valid_weight_ratio: float = 0.70, limit: Optional[int] = None):
    """批量计算所有股票过去N个报告期的评分（跳过金融行业）"""
    from datasource.database import get_session
    from sqlalchemy import text
    from tqdm import tqdm

    print(f"\n{'='*70}")
    print(f"批量计算财报评分")
    print(f"{'='*70}")
    print(f"计算过去 {quarters} 个报告期的评分")
    print(f"{'='*70}\n")

    # 初始化表
    init_financial_scores_table()

    # 获取所有股票列表（排除金融行业：银行、保险、证券等）
    # 金融行业代码特征：
    # - 银行：000001, 600000, 601xxx 等
    # - 保险：601318, 601628 等
    # - 证券：600030, 600837, 000776 等
    # 通过排除这些特定代码来跳过金融行业
    financial_stocks = {
        # 银行
        '000001', '000002', '600000', '600015', '600016', '600036', '601009', 
        '601166', '601169', '601288', '601328', '601398', '601818', '601939',
        '601988', '601998', '603323',
        # 保险
        '601318', '601628', '601336', '601601',
        # 证券
        '600030', '600837', '600999', '601066', '601099', '601108', '601162',
        '601211', '601377', '601555', '601688', '601788', '601881', '601901',
        '000166', '000617', '000686', '000728', '000750', '000776', '000987',
        '002500', '002673', '002736', '002797', '002926', '002939', '002945',
        # 信托/金融控股
        '600643', '600705', '600816', '000563',
    }
    
    with get_session() as session:
        sql = "SELECT DISTINCT ts_code FROM financial_quarterly_data ORDER BY ts_code"
        result = session.execute(text(sql))
        all_stocks = [row[0] for row in result.fetchall()]
    
    # 过滤掉金融行业股票
    all_stocks = [s for s in all_stocks if s.split('.')[0] not in financial_stocks]

    if limit:
        all_stocks = all_stocks[:limit]

    print(f"共 {len(all_stocks)} 只非金融股票需要处理（已跳过金融行业）\n")

    total_saved = 0
    success_count = 0
    fail_count = 0

    for ts_code in tqdm(all_stocks, desc="处理股票"):
        try:
            # 计算评分
            df = calculate_stock_scores(ts_code, min_valid_weight_ratio)

            if df.empty:
                fail_count += 1
                continue

            # 只保留最近N个报告期
            df_sorted = df.sort_values("end_date_dt", ascending=False)
            recent_df = df_sorted.head(quarters)

            # 保存到数据库
            saved_count = save_scores_to_database(recent_df)
            total_saved += saved_count
            success_count += 1

        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            fail_count += 1
            continue

    print(f"\n{'='*70}")
    print(f"批量计算完成")
    print(f"{'='*70}")
    print(f"成功: {success_count} 只")
    print(f"失败: {fail_count} 只")
    print(f"共保存: {total_saved} 条评分记录")
    print(f"{'='*70}\n")


def run_single_stock(ts_code: str, end_date: str, min_valid_weight_ratio: float = 0.70):
    """运行单只股票评分"""
    print(f"\n{'='*70}")
    print(f"单只股票评分")
    print(f"{'='*70}")
    print(f"股票: {ts_code}")
    print(f"目标报告期: {end_date}")
    print(f"{'='*70}\n")

    # 初始化表
    init_financial_scores_table()

    # 计算评分
    df = calculate_stock_scores(ts_code, min_valid_weight_ratio)

    if df.empty:
        print("❌ 评分计算失败")
        return

    # 保存到数据库
    saved_count = save_scores_to_database(df)
    print(f"✅ 已保存 {saved_count} 条评分记录到数据库")

    # 输出目标报告期的结果
    target_df = df[df["end_date"] == end_date]
    if target_df.empty:
        print(f"⚠️ 未找到报告期 {end_date} 的评分结果")
        return

    row = target_df.iloc[0]
    print(f"\n{'='*70}")
    print(f"评分结果 - {ts_code} ({end_date})")
    print(f"{'='*70}")
    print(f"总分: {row.get('total_score', 'N/A'):.2f}" if pd.notna(row.get('total_score')) else "总分: N/A")
    print(f"评分状态: {row.get('score_status', 'N/A')}")
    print(f"\n维度得分：")
    print(f"  主业盈利改善: {row.get('main_profit_score', 'N/A'):.2f}" if pd.notna(row.get('main_profit_score')) else "  主业盈利改善: N/A")
    print(f"  收入费用效率: {row.get('revenue_expense_efficiency_score', 'N/A'):.2f}" if pd.notna(row.get('revenue_expense_efficiency_score')) else "  收入费用效率: N/A")
    print(f"  现金流验证: {row.get('cashflow_validation_score', 'N/A'):.2f}" if pd.notna(row.get('cashflow_validation_score')) else "  现金流验证: N/A")
    print(f"  营运资本质量: {row.get('working_capital_quality_score', 'N/A'):.2f}" if pd.notna(row.get('working_capital_quality_score')) else "  营运资本质量: N/A")
    print(f"  投入资产效率: {row.get('investment_asset_efficiency_score', 'N/A'):.2f}" if pd.notna(row.get('investment_asset_efficiency_score')) else "  投入资产效率: N/A")
    print(f"{'='*70}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单股财报边际改善评分卡 - 数据库版本")
    parser.add_argument("--ts-code", default=None, help="股票代码，例如 688012.SH")
    parser.add_argument("--end-date", default=None, help="目标报告期，例如 20251231")
    parser.add_argument("--batch", action="store_true", help="批量计算所有股票")
    parser.add_argument("--quarters", type=int, default=8, help="批量计算时，计算过去N个报告期（默认8）")
    parser.add_argument("--limit", type=int, default=None, help="批量计算时，限制处理的股票数量（用于测试）")
    parser.add_argument("--min-valid-weight-ratio", type=float, default=0.70, help="总分有效所需最低可计算权重覆盖率")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.batch:
        # 批量计算模式
        batch_calculate_scores(
            quarters=args.quarters,
            min_valid_weight_ratio=args.min_valid_weight_ratio,
            limit=args.limit
        )
    elif args.ts_code and args.end_date:
        # 单只股票模式
        run_single_stock(
            ts_code=args.ts_code,
            end_date=args.end_date,
            min_valid_weight_ratio=args.min_valid_weight_ratio
        )
    else:
        print("\n❌ 错误：必须提供以下参数之一：")
        print("  1. --batch （批量计算所有股票）")
        print("  2. --ts-code 和 --end-date （单只股票评分）")
        print("\n示例：")
        print("  python financial_factors/financial_statement_scorecard.py --batch --quarters 8")
        print("  python financial_factors/financial_statement_scorecard.py --ts-code 688012.SH --end-date 20251231")
        exit(1)
