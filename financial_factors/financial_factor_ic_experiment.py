#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财务核心因子IC实验脚本

Purpose: 评估 financial_quarterly_score.py 中的核心财务因子与公告后60日收益率的相关性和预测力

Inputs:
    - financial_quarterly_data: 季度财务数据
    - stock_k_data: 日频股价数据

Outputs:
    - Excel报告: 包含IC分析、分组收益、稳定性分析等多sheet

How to Run:
    python financial_factor_ic_experiment.py --start 20210101 --output out/factor_ic_report.xlsx
    python financial_factor_ic_experiment.py --help

Examples:
    # 全量测试（过去5年）
    python financial_factor_ic_experiment.py --start 20210101 --output out/factor_ic_5y.xlsx
    
    # 快速测试（限制股票数量）
    python financial_factor_ic_experiment.py --start 20230101 --stock-limit 100 --output out/factor_ic_test.xlsx

Side Effects:
    - 读取数据库表 financial_quarterly_data, stock_k_data
    - 写入Excel文件到指定路径
    - 不修改数据库
"""

import os
import sys
import argparse
import logging
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

try:
    from sqlalchemy import create_engine, text
except Exception:
    create_engine = None
    text = None

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

try:
    from config import DATABASE_URL
except Exception:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

# 核心因子配置（来自 financial_quarterly_score.py）
CORE_FACTORS = [
    {"name": "q_rev_yoy", "label": "单季营业收入同比", "direction": "higher_better"},
    {"name": "q_op_yoy", "label": "单季营业利润同比", "direction": "higher_better"},
    {"name": "q_np_parent_yoy", "label": "单季归母净利润同比", "direction": "higher_better"},
    {"name": "ytd_rev_yoy", "label": "累计营业收入同比", "direction": "higher_better"},
    {"name": "ytd_np_parent_yoy", "label": "累计归母净利润同比", "direction": "higher_better"},
    {"name": "q_gross_margin", "label": "单季毛利率", "direction": "higher_better"},
    {"name": "q_gm_yoy_change", "label": "单季毛利率同比变化", "direction": "higher_better"},
    {"name": "q_op_margin", "label": "单季营业利润率", "direction": "higher_better"},
    {"name": "q_np_parent_margin", "label": "单季归母净利率", "direction": "higher_better"},
    {"name": "q_cfo_to_np_parent", "label": "单季经营现金流/归母净利润", "direction": "higher_better"},
    {"name": "ttm_cfo_to_np_parent", "label": "TTM经营现金流/TTM归母净利润", "direction": "higher_better"},
    {"name": "q_accruals_to_assets", "label": "单季应计项/平均总资产", "direction": "lower_better"},
    {"name": "q_cfo_to_rev", "label": "单季经营现金流/收入", "direction": "higher_better"},
    {"name": "q_cfo_yoy", "label": "单季经营现金流同比", "direction": "higher_better"},
    {"name": "ytd_cfo_yoy", "label": "累计经营现金流同比", "direction": "higher_better"},
    {"name": "ttm_fcf_to_np_parent", "label": "TTM自由现金流/TTM归母净利润", "direction": "higher_better"},
    {"name": "roa_parent", "label": "归母ROA", "direction": "higher_better"},
    {"name": "cfo_to_assets", "label": "经营现金流/总资产", "direction": "higher_better"},
    {"name": "q_rev_yoy_delta", "label": "单季收入同比变化", "direction": "higher_better"},
    {"name": "q_np_parent_yoy_delta", "label": "单季归母净利润同比变化", "direction": "higher_better"},
    {"name": "trend_consistency", "label": "趋势连续性", "direction": "higher_better"},
]


def create_engine_or_raise():
    """创建数据库引擎"""
    if create_engine is None or text is None:
        raise ImportError("sqlalchemy未安装")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL未配置")
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def safe_div(a, b, eps=1e-8):
    """安全除法"""
    if isinstance(b, pd.Series):
        bb = b.copy()
        bb = bb.mask(bb.abs() < eps)
        result = a / bb
        result = result.replace([np.inf, -np.inf], np.nan)
        return result
    if b is None or pd.isna(b) or abs(b) < eps:
        return np.nan
    result = a / b
    if np.isinf(result) or np.isnan(result):
        return np.nan
    return result


def yoy(series: pd.Series, lag: int = 4) -> pd.Series:
    """计算同比增长率"""
    prev = series.shift(lag)
    return safe_div(series - prev, prev.abs())


def qoq(series: pd.Series, lag: int = 1) -> pd.Series:
    """计算环比增长率"""
    prev = series.shift(lag)
    return safe_div(series - prev, prev.abs())


def winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    """去极值处理"""
    if s.dropna().empty:
        return s
    ql = s.quantile(lower)
    qu = s.quantile(upper)
    return s.clip(lower=ql, upper=qu)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """标准化列名"""
    RAW_TO_STD = {
        "tscode": "ts_code",
        "enddate": "end_date",
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
    }
    
    def normalize_colname(col: str) -> str:
        return "".join(ch.lower() for ch in str(col) if ch.isalnum())
    
    rename_map = {}
    for col in df.columns:
        key = normalize_colname(col)
        if key in RAW_TO_STD:
            rename_map[col] = RAW_TO_STD[key]
    return df.rename(columns=rename_map).copy()


def load_financial_data(start_date: str, stock_limit: Optional[int] = None) -> pd.DataFrame:
    """加载财务数据"""
    logger.info("加载财务数据...")
    engine = create_engine_or_raise()
    
    with engine.connect() as conn:
        sql = f'SELECT * FROM "financial_quarterly_data" WHERE end_date >= \'{start_date}\''
        if stock_limit:
            sql += f' LIMIT {stock_limit * 20}'
        df = pd.read_sql(text(sql), conn)
    
    df = normalize_columns(df)
    
    # 日期转换
    df["end_date"] = pd.to_datetime(df["end_date"].astype(str), format="%Y%m%d", errors="coerce")
    df["ann_date"] = pd.to_datetime(df["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    df["f_ann_date"] = pd.to_datetime(df["f_ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    
    # 去重：取每个ts_code+end_date的最新记录
    df = df.sort_values(["ts_code", "end_date", "f_ann_date", "ann_date"])
    df = df.groupby(["ts_code", "end_date"], as_index=False).last()
    
    logger.info(f"加载财务数据: {len(df)} 条, {df['ts_code'].nunique()} 只股票")
    return df


def load_stock_prices(ts_codes: List[str]) -> pd.DataFrame:
    """加载股价数据"""
    logger.info("加载股价数据...")
    engine = create_engine_or_raise()
    
    # 分批查询避免SQL过长
    batch_size = 500
    all_prices = []
    
    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i:i+batch_size]
        codes_str = ",".join([f"'{c}'" for c in batch])
        
        with engine.connect() as conn:
            sql = f"""
            SELECT ts_code, bar_time::date as trade_date, close
            FROM stock_k_data
            WHERE ts_code IN ({codes_str}) AND freq = 'd'
            ORDER BY ts_code, bar_time
            """
            batch_df = pd.read_sql(text(sql), conn)
            all_prices.append(batch_df)
    
    df = pd.concat(all_prices, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    
    logger.info(f"加载股价数据: {len(df)} 条, {df['ts_code'].nunique()} 只股票")
    return df


def calculate_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算核心因子"""
    logger.info("计算核心因子...")
    
    out = df.copy()
    
    # 基础字段
    out["rev_q"] = out["total_revenue"].fillna(out["revenue"])
    out["cost_q"] = out["oper_cost"]
    out["op_q"] = out["operate_profit"]
    out["ebit_q"] = out["ebit"]
    out["np_q"] = out["n_income"]
    out["np_parent_q"] = out["n_income_attr_p"]
    out["cfo_q"] = out["n_cashflow_act"]
    out["capex_q"] = out["c_pay_acq_const_fiolta"]
    out["fcf_q"] = out["free_cashflow"].fillna(out["cfo_q"] - out["capex_q"])
    out["cash_sales_q"] = out["c_fr_sale_sg"]
    
    # 资产负债表字段
    for c in ["total_assets", "accounts_receiv", "inventories", "accounts_pay", "contract_liab"]:
        if c not in out.columns:
            out[c] = np.nan
    
    # 计算YTD
    out["fiscal_year"] = out["end_date"].dt.year
    out["quarter"] = out["end_date"].dt.quarter
    
    for base_col in ["rev_q", "np_parent_q", "cfo_q"]:
        ytd_col = base_col.replace("_q", "_ytd")
        out[ytd_col] = out.groupby(["ts_code", "fiscal_year"])[base_col].cumsum()
    
    # 去年同期YTD（用于计算ytd_yoy）
    lag = out[["ts_code", "fiscal_year", "quarter", "rev_ytd", "np_parent_ytd", "cfo_ytd"]].copy()
    lag["fiscal_year"] = lag["fiscal_year"] + 1
    lag = lag.rename(columns={
        "rev_ytd": "rev_ytd_lag4",
        "np_parent_ytd": "np_parent_ytd_lag4",
        "cfo_ytd": "cfo_ytd_lag4",
    })
    out = out.merge(lag, on=["ts_code", "fiscal_year", "quarter"], how="left")
    
    # 计算TTM
    for base_col in ["rev_q", "op_q", "ebit_q", "np_parent_q", "cfo_q", "fcf_q", "capex_q"]:
        ttm_col = base_col.replace("_q", "_ttm")
        out[ttm_col] = (
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
    
    # 平均总资产
    out["avg_assets"] = (out["total_assets"] + out.groupby("ts_code")["total_assets"].shift(1)) / 2
    
    # 计算因子
    out["q_rev_yoy"] = yoy(out["rev_q"])
    out["q_op_yoy"] = yoy(out["op_q"])
    out["q_np_parent_yoy"] = yoy(out["np_parent_q"])
    
    out["ytd_rev_yoy"] = safe_div(out["rev_ytd"] - out["rev_ytd_lag4"], out["rev_ytd_lag4"])
    out["ytd_np_parent_yoy"] = safe_div(out["np_parent_ytd"] - out["np_parent_ytd_lag4"], out["np_parent_ytd_lag4"])
    
    out["q_gross_margin"] = 1 - safe_div(out["cost_q"], out["rev_q"])
    out["q_gm_yoy_change"] = out["q_gross_margin"] - out["q_gross_margin"].shift(4)
    
    out["q_op_margin"] = safe_div(out["op_q"], out["rev_q"])
    out["q_np_parent_margin"] = safe_div(out["np_parent_q"], out["rev_q"])
    
    out["q_cfo_to_np_parent"] = safe_div(out["cfo_q"], out["np_parent_q"])
    out["ttm_cfo_to_np_parent"] = safe_div(out["cfo_ttm"], out["np_parent_ttm"])
    out["q_accruals_to_assets"] = safe_div(out["np_parent_q"] - out["cfo_q"], out["avg_assets"])
    
    out["q_cfo_to_rev"] = safe_div(out["cfo_q"], out["rev_q"])
    out["q_cfo_yoy"] = yoy(out["cfo_q"])
    out["ytd_cfo_yoy"] = safe_div(out["cfo_ytd"] - out["cfo_ytd_lag4"], out["cfo_ytd_lag4"])
    out["ttm_fcf_to_np_parent"] = safe_div(out["fcf_ttm"], out["np_parent_ttm"])
    
    out["roa_parent"] = safe_div(out["np_parent_ttm"], out["avg_assets"])
    out["cfo_to_assets"] = safe_div(out["cfo_ttm"], out["avg_assets"])
    
    out["q_rev_yoy_delta"] = out["q_rev_yoy"] - out["q_rev_yoy"].shift(1)
    out["q_np_parent_yoy_delta"] = out["q_np_parent_yoy"] - out["q_np_parent_yoy"].shift(1)
    
    # 趋势连续性
    out["rev_improve"] = (out["q_rev_yoy"] > out["q_rev_yoy"].shift(1)).astype(float)
    out["np_improve"] = (out["q_np_parent_yoy"] > out["q_np_parent_yoy"].shift(1)).astype(float)
    out["gm_improve"] = (out["q_gm_yoy_change"] > out["q_gm_yoy_change"].shift(1)).astype(float)
    raw_consistency = (
        out["rev_improve"].rolling(4).sum()
        + out["np_improve"].rolling(4).sum()
        + out["gm_improve"].rolling(4).sum()
    )
    out["trend_consistency"] = raw_consistency / 12 * 100
    
    logger.info(f"因子计算完成: {len(out)} 条记录")
    return out


def calculate_forward_returns(fin_df: pd.DataFrame, price_df: pd.DataFrame, days: int = 60) -> pd.DataFrame:
    """计算公告后60日收益"""
    logger.info(f"计算公告后{days}日收益...")
    
    df = fin_df.copy()
    df["entry_date"] = pd.NaT
    df["exit_date"] = pd.NaT
    df["entry_price"] = np.nan
    df["exit_price"] = np.nan
    df[f"ret_{days}d"] = np.nan
    
    # 按股票处理
    for ts_code in tqdm(df["ts_code"].unique(), desc="计算收益", leave=False):
        stock_fin = df[df["ts_code"] == ts_code].copy()
        stock_price = price_df[price_df["ts_code"] == ts_code].sort_values("trade_date")
        
        if stock_price.empty:
            continue
        
        price_dates = stock_price["trade_date"].values
        
        for idx, row in stock_fin.iterrows():
            ann_date = row["ann_date"]
            if pd.isna(ann_date):
                continue
            
            # 找到公告日后的第一个交易日
            future_dates = price_dates[price_dates > np.datetime64(ann_date)]
            if len(future_dates) < days + 1:
                continue
            
            entry_date = future_dates[0]
            exit_date = future_dates[days]
            
            entry_price = stock_price[stock_price["trade_date"] == entry_date]["close"].values
            exit_price = stock_price[stock_price["trade_date"] == exit_date]["close"].values
            
            if len(entry_price) > 0 and len(exit_price) > 0:
                df.loc[idx, "entry_date"] = entry_date
                df.loc[idx, "exit_date"] = exit_date
                df.loc[idx, "entry_price"] = entry_price[0]
                df.loc[idx, "exit_price"] = exit_price[0]
                df.loc[idx, f"ret_{days}d"] = (exit_price[0] - entry_price[0]) / entry_price[0]
    
    valid_count = df[f"ret_{days}d"].notna().sum()
    logger.info(f"收益计算完成: {valid_count}/{len(df)} 条有效记录")
    return df


def calculate_ic(df: pd.DataFrame, factor_name: str, return_col: str = "ret_60d") -> Dict:
    """计算单个因子的IC指标"""
    factor_df = df[["end_date", factor_name, return_col]].dropna()
    
    if len(factor_df) < 30:
        return None
    
    ic_list = []
    pearson_ic_list = []
    
    for date, group in factor_df.groupby("end_date"):
        if len(group) < 10:
            continue
        
        # Spearman IC
        spearman_ic, _ = stats.spearmanr(group[factor_name], group[return_col])
        # Pearson IC
        pearson_ic, _ = stats.pearsonr(group[factor_name], group[return_col])
        
        if not np.isnan(spearman_ic):
            ic_list.append({"end_date": date, "ic": spearman_ic, "pearson_ic": pearson_ic, "n": len(group)})
    
    if not ic_list:
        return None
    
    ic_df = pd.DataFrame(ic_list)
    
    return {
        "factor": factor_name,
        "n_periods": len(ic_df),
        "avg_n": ic_df["n"].mean(),
        "ic_mean": ic_df["ic"].mean(),
        "ic_median": ic_df["ic"].median(),
        "ic_std": ic_df["ic"].std(),
        "icir": ic_df["ic"].mean() / ic_df["ic"].std() if ic_df["ic"].std() > 0 else np.nan,
        "ic_positive_ratio": (ic_df["ic"] > 0).mean(),
        "ic_negative_ratio": (ic_df["ic"] < 0).mean(),
        "ic_abs_mean": ic_df["ic"].abs().mean(),
        "pearson_ic_mean": ic_df["pearson_ic"].mean(),
        "t_stat": ic_df["ic"].mean() / (ic_df["ic"].std() / np.sqrt(len(ic_df))) if ic_df["ic"].std() > 0 else np.nan,
        "ic_detail": ic_df,
    }


def calculate_group_returns(df: pd.DataFrame, factor_name: str, n_groups: int = 5, return_col: str = "ret_60d") -> pd.DataFrame:
    """计算分组收益"""
    results = []
    
    for date, group in df.groupby("end_date"):
        if len(group) < n_groups * 10:
            continue
        
        group = group.dropna(subset=[factor_name, return_col])
        if len(group) < n_groups * 5:
            continue
        
        try:
            group["group"] = pd.qcut(group[factor_name], q=n_groups, labels=False, duplicates="drop")
        except:
            continue
        
        for g in range(n_groups):
            g_data = group[group["group"] == g]
            if len(g_data) > 0:
                results.append({
                    "end_date": date,
                    "factor": factor_name,
                    "group": g,
                    "mean_ret": g_data[return_col].mean(),
                    "median_ret": g_data[return_col].median(),
                    "n": len(g_data),
                })
    
    return pd.DataFrame(results)


def generate_report(df: pd.DataFrame, output_path: str, n_groups: int = 5):
    """生成Excel报告"""
    logger.info("生成Excel报告...")
    
    # 1. IC汇总
    ic_summaries = []
    for cfg in tqdm(CORE_FACTORS, desc="计算IC"):
        result = calculate_ic(df, cfg["name"])
        if result:
            ic_summaries.append({
                "factor": cfg["name"],
                "label": cfg["label"],
                "direction": cfg["direction"],
                "n_periods": result["n_periods"],
                "avg_n": result["avg_n"],
                "ic_mean": result["ic_mean"],
                "ic_median": result["ic_median"],
                "ic_std": result["ic_std"],
                "icir": result["icir"],
                "ic_abs_mean": result["ic_abs_mean"],
                "ic_positive_ratio": result["ic_positive_ratio"],
                "pearson_ic_mean": result["pearson_ic_mean"],
                "t_stat": result["t_stat"],
            })
    
    ic_summary_df = pd.DataFrame(ic_summaries)
    
    # 2. IC时序
    ic_detail_list = []
    for cfg in CORE_FACTORS:
        result = calculate_ic(df, cfg["name"])
        if result and result["ic_detail"] is not None:
            detail = result["ic_detail"].copy()
            detail["factor"] = cfg["name"]
            ic_detail_list.append(detail[["end_date", "factor", "ic", "pearson_ic", "n"]])
    
    ic_detail_df = pd.concat(ic_detail_list, ignore_index=True) if ic_detail_list else pd.DataFrame()
    
    # 3. 分组收益
    group_returns_list = []
    for cfg in tqdm(CORE_FACTORS[:5], desc="计算分组收益"):  # 只计算前5个因子的分组收益（性能考虑）
        gr = calculate_group_returns(df, cfg["name"], n_groups)
        if not gr.empty:
            group_returns_list.append(gr)
    
    group_returns_df = pd.concat(group_returns_list, ignore_index=True) if group_returns_list else pd.DataFrame()
    
    # 4. 样本概览
    overview = pd.DataFrame({
        "指标": ["总样本数", "有效样本数", "股票数", "报告期数", "因子数"],
        "数值": [
            len(df),
            df["ret_60d"].notna().sum(),
            df["ts_code"].nunique(),
            df["end_date"].nunique(),
            len(CORE_FACTORS),
        ],
    })
    
    # 写入Excel
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        overview.to_excel(writer, sheet_name="样本概览", index=False)
        ic_summary_df.to_excel(writer, sheet_name="IC汇总", index=False)
        if not ic_detail_df.empty:
            ic_detail_df.to_excel(writer, sheet_name="IC时序", index=False)
        if not group_returns_df.empty:
            group_returns_df.to_excel(writer, sheet_name="分组收益", index=False)
        
        # 因子配置
        factor_config_df = pd.DataFrame(CORE_FACTORS)
        factor_config_df.to_excel(writer, sheet_name="因子配置", index=False)
    
    logger.info(f"Excel报告已保存: {output_path}")
    return ic_summary_df


def parse_args():
    parser = argparse.ArgumentParser(description="财务核心因子IC实验")
    parser.add_argument("--start", type=str, default="20210101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--output", type=str, default="out/factor_ic_report.xlsx", help="输出Excel路径")
    parser.add_argument("--stock-limit", type=int, default=None, help="限制股票数量（测试用）")
    parser.add_argument("--n-groups", type=int, default=5, help="分组数量")
    parser.add_argument("--return-days", type=int, default=60, help="收益持有天数")
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("财务核心因子IC实验")
    logger.info("=" * 60)
    logger.info(f"时间范围: {args.start} - {args.end}")
    logger.info(f"输出路径: {args.output}")
    logger.info(f"核心因子数: {len(CORE_FACTORS)}")
    
    # 1. 加载财务数据
    fin_df = load_financial_data(args.start, args.stock_limit)
    
    if args.stock_limit:
        fin_df = fin_df[fin_df["ts_code"].isin(fin_df["ts_code"].unique()[:args.stock_limit])]
    
    # 2. 计算因子
    fin_df = calculate_factors(fin_df)
    
    # 3. 加载股价数据
    ts_codes = fin_df["ts_code"].unique().tolist()
    price_df = load_stock_prices(ts_codes)
    
    # 4. 计算收益
    fin_df = calculate_forward_returns(fin_df, price_df, days=args.return_days)
    
    # 5. 生成报告
    ic_summary = generate_report(fin_df, args.output, n_groups=args.n_groups)
    
    # 6. 输出Top因子
    logger.info("\n" + "=" * 60)
    logger.info("Top 10 因子（按|IC|排序）:")
    logger.info("=" * 60)
    top_factors = ic_summary.nlargest(10, "ic_abs_mean")[["factor", "label", "ic_mean", "ic_abs_mean", "icir"]]
    for _, row in top_factors.iterrows():
        logger.info(f"  {row['factor']}: IC={row['ic_mean']:.4f}, |IC|={row['ic_abs_mean']:.4f}, ICIR={row['icir']:.4f}")
    
    logger.info("\n实验完成!")


if __name__ == "__main__":
    main()
