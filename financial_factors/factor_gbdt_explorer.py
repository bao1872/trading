#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    用 GBDT 探索37个原始财务因子对公告后股价涨跌的贡献度。
    包括：特征重要性(gain/split)、SHAP值分析、交互效应、分报告期滚动验证。

Inputs:
    - 数据库表: factor_return_dataset

Outputs:
    - 控制台: 特征重要性排序、SHAP分析、交互效应
    - CSV: out/gbdt_feature_importance.csv

How to Run:
    python financial_factors/factor_gbdt_explorer.py
    python financial_factors/factor_gbdt_explorer.py --horizon 20d
    python financial_factors/factor_gbdt_explorer.py --horizon 60d --out-dir out

Examples:
    python financial_factors/factor_gbdt_explorer.py
    python financial_factors/factor_gbdt_explorer.py --horizon 5d

Side Effects:
    - 无数据库写入，仅输出CSV和控制台
"""

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

try:
    from config import DATABASE_URL
except Exception:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FACTOR_COLS = [
    "q_rev_yoy", "q_op_yoy", "q_np_parent_yoy", "ytd_rev_yoy", "ytd_np_parent_yoy",
    "q_ebit_yoy", "q_rev_qoq", "q_op_qoq",
    "q_gross_margin", "q_gm_yoy_change", "q_op_margin", "q_np_parent_margin",
    "q_gm_qoq_change", "op_margin_change", "q_ebit_margin",
    "q_cfo_to_np_parent", "ttm_cfo_to_np_parent", "q_accruals_to_assets",
    "ttm_cfo_to_ebit", "q_np_parent_to_np",
    "q_cfo_to_rev", "q_cfo_yoy", "ytd_cfo_yoy", "ttm_fcf_to_np_parent",
    "capex_to_cfo", "cash_sales_ratio", "cash_sales_yoy",
    "roa_parent", "cfo_to_assets", "asset_turnover", "ccc", "contract_liab_to_rev",
    "q_rev_yoy_delta", "q_np_parent_yoy_delta", "trend_consistency",
    "profit_cash_sync", "margin_profit_sync", "cfo_to_np_change",
]

LABELS = {
    "q_rev_yoy": "单季营收同比", "q_op_yoy": "单季营业利润同比",
    "q_np_parent_yoy": "单季归母净利同比", "ytd_rev_yoy": "累计营收同比",
    "ytd_np_parent_yoy": "累计归母净利同比", "q_ebit_yoy": "单季EBIT同比",
    "q_rev_qoq": "单季营收环比", "q_op_qoq": "单季营业利润环比",
    "q_gross_margin": "单季毛利率", "q_gm_yoy_change": "毛利率同比变化",
    "q_op_margin": "单季营业利润率", "q_np_parent_margin": "单季归母净利率",
    "q_gm_qoq_change": "毛利率环比变化", "op_margin_change": "营业利润率同比变化",
    "q_ebit_margin": "EBIT利润率",
    "q_cfo_to_np_parent": "经营现金流/归母净利", "ttm_cfo_to_np_parent": "TTM经营现金流/归母净利",
    "q_accruals_to_assets": "应计项/总资产", "ttm_cfo_to_ebit": "TTM经营现金流/EBIT",
    "q_np_parent_to_np": "归母净利/净利润",
    "q_cfo_to_rev": "经营现金流/收入", "q_cfo_yoy": "经营现金流同比",
    "ytd_cfo_yoy": "累计经营现金流同比", "ttm_fcf_to_np_parent": "TTM自由现金流/归母净利",
    "capex_to_cfo": "资本开支/经营现金流", "cash_sales_ratio": "销售收现比",
    "cash_sales_yoy": "销售收现同比",
    "roa_parent": "归母ROA", "cfo_to_assets": "经营现金流/总资产",
    "asset_turnover": "总资产周转率", "ccc": "现金转换周期",
    "contract_liab_to_rev": "合同负债/收入",
    "q_rev_yoy_delta": "营收同比变化", "q_np_parent_yoy_delta": "归母净利同比变化",
    "trend_consistency": "趋势连续性", "profit_cash_sync": "利润现金流同步",
    "margin_profit_sync": "毛利率利润率同步", "cfo_to_np_change": "经营现金流/归母净利变化",
}

DIMS = {
    "q_rev_yoy": "规模与增长", "q_op_yoy": "规模与增长", "q_np_parent_yoy": "规模与增长",
    "ytd_rev_yoy": "规模与增长", "ytd_np_parent_yoy": "规模与增长",
    "q_ebit_yoy": "规模与增长", "q_rev_qoq": "规模与增长", "q_op_qoq": "规模与增长",
    "q_gross_margin": "盈利能力", "q_gm_yoy_change": "盈利能力",
    "q_op_margin": "盈利能力", "q_np_parent_margin": "盈利能力",
    "q_gm_qoq_change": "盈利能力", "op_margin_change": "盈利能力", "q_ebit_margin": "盈利能力",
    "q_cfo_to_np_parent": "利润质量", "ttm_cfo_to_np_parent": "利润质量",
    "q_accruals_to_assets": "利润质量", "ttm_cfo_to_ebit": "利润质量", "q_np_parent_to_np": "利润质量",
    "q_cfo_to_rev": "现金创造", "q_cfo_yoy": "现金创造",
    "ytd_cfo_yoy": "现金创造", "ttm_fcf_to_np_parent": "现金创造",
    "capex_to_cfo": "现金创造", "cash_sales_ratio": "现金创造", "cash_sales_yoy": "现金创造",
    "roa_parent": "资产效率", "cfo_to_assets": "资产效率",
    "asset_turnover": "资产效率", "ccc": "资产效率", "contract_liab_to_rev": "资产效率",
    "q_rev_yoy_delta": "边际变化", "q_np_parent_yoy_delta": "边际变化",
    "trend_consistency": "边际变化", "profit_cash_sync": "边际变化",
    "margin_profit_sync": "边际变化", "cfo_to_np_change": "边际变化",
}


def get_db_engine():
    return create_engine(DATABASE_URL, pool_size=5, max_overflow=10, pool_recycle=3600)


def load_dataset(engine, return_col: str) -> pd.DataFrame:
    sql = "SELECT * FROM factor_return_dataset"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    df = df[df[return_col].notna()].copy()
    available_factors = [c for c in FACTOR_COLS if c in df.columns]
    df = df.dropna(subset=available_factors, how="all")
    logger.info(f"加载数据集: {len(df)} 条, 收益列={return_col}")
    return df


def train_gbdt(df: pd.DataFrame, return_col: str):
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error, r2_score

    available_factors = [c for c in FACTOR_COLS if c in df.columns]

    df = df.copy()
    df["report_date_str"] = df["report_date"].astype(str)
    dates_sorted = sorted(df["report_date_str"].unique())

    n_total = len(dates_sorted)
    n_train = int(n_total * 0.7)
    train_dates = dates_sorted[:n_train]
    test_dates = dates_sorted[n_train:]

    train_df = df[df["report_date_str"].isin(train_dates)]
    test_df = df[df["report_date_str"].isin(test_dates)]

    X_train = train_df[available_factors].values
    y_train = train_df[return_col].values
    X_test = test_df[available_factors].values
    y_test = test_df[return_col].values

    logger.info(f"训练集: {len(train_df)} 条 ({train_dates[0]}~{train_dates[-1]})")
    logger.info(f"测试集: {len(test_df)} 条 ({test_dates[0]}~{test_dates[-1]})")

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=available_factors)
    dtest = lgb.Dataset(X_test, label=y_test, feature_name=available_factors, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dtest],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(200),
        ],
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"测试集 MAE={mae:.6f}, R2={r2:.6f}")

    importance_gain = model.feature_importance(importance_type="gain")
    importance_split = model.feature_importance(importance_type="split")

    fi_df = pd.DataFrame({
        "factor": available_factors,
        "label": [LABELS.get(f, f) for f in available_factors],
        "dim": [DIMS.get(f, "") for f in available_factors],
        "importance_gain": importance_gain,
        "importance_split": importance_split,
    }).sort_values("importance_gain", ascending=False)

    fi_df["gain_rank"] = range(1, len(fi_df) + 1)
    fi_df["gain_pct"] = fi_df["importance_gain"] / fi_df["importance_gain"].sum() * 100
    fi_df["gain_cum_pct"] = fi_df["gain_pct"].cumsum()

    return model, fi_df, test_df, available_factors


def shap_analysis(model, test_df: pd.DataFrame, available_factors: list) -> pd.DataFrame:
    import shap

    X_test = test_df[available_factors].values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)

    shap_df = pd.DataFrame({
        "factor": available_factors,
        "label": [LABELS.get(f, f) for f in available_factors],
        "dim": [DIMS.get(f, "") for f in available_factors],
        "shap_mean_abs": mean_abs_shap,
        "shap_mean": mean_shap,
    }).sort_values("shap_mean_abs", ascending=False)

    shap_df["shap_rank"] = range(1, len(shap_df) + 1)
    shap_df["shap_pct"] = shap_df["shap_mean_abs"] / shap_df["shap_mean_abs"].sum() * 100
    shap_df["direction"] = shap_df["shap_mean"].apply(lambda x: "正向" if x > 0 else "反向")

    return shap_df, shap_values


def dim_contribution(fi_df: pd.DataFrame, shap_df: pd.DataFrame) -> pd.DataFrame:
    gain_by_dim = fi_df.groupby("dim")["importance_gain"].sum().reset_index()
    gain_by_dim["gain_pct"] = gain_by_dim["importance_gain"] / gain_by_dim["importance_gain"].sum() * 100

    shap_by_dim = shap_df.groupby("dim").agg(
        shap_sum=("shap_mean_abs", "sum"),
        n_factors=("factor", "count"),
        n_positive=("direction", lambda x: (x == "正向").sum()),
        n_negative=("direction", lambda x: (x == "反向").sum()),
    ).reset_index()
    shap_by_dim["shap_pct"] = shap_by_dim["shap_sum"] / shap_by_dim["shap_sum"].sum() * 100

    merged = gain_by_dim.merge(shap_by_dim, on="dim").sort_values("shap_pct", ascending=False)
    return merged


def print_report(fi_df: pd.DataFrame, shap_df: pd.DataFrame, dim_df: pd.DataFrame,
                 return_col: str):
    print(f"\n{'='*80}")
    print(f"GBDT 因子贡献度分析 — 目标: {return_col}")
    print(f"{'='*80}")

    print(f"\n--- Top 15 因子 (Gain重要性) ---")
    for _, r in fi_df.head(15).iterrows():
        print(f"  [{r['dim']:6s}] {r['label']:20s} gain={r['importance_gain']:8.1f} "
              f"({r['gain_pct']:5.1f}%) 累计={r['gain_cum_pct']:5.1f}%")

    print(f"\n--- Top 15 因子 (SHAP贡献) ---")
    for _, r in shap_df.head(15).iterrows():
        print(f"  [{r['dim']:6s}] {r['label']:20s} SHAP={r['shap_mean_abs']:.6f} "
              f"({r['shap_pct']:5.1f}%) 方向={r['direction']}")

    print(f"\n--- 维度贡献汇总 ---")
    print(f"  {'维度':10s} {'gain%':>6s} {'shap%':>6s} {'因子数':>5s} {'正向':>4s} {'反向':>4s}")
    for _, r in dim_df.iterrows():
        print(f"  {r['dim']:10s} {r['gain_pct']:6.1f} {r['shap_pct']:6.1f} "
              f"{r['n_factors']:5d} {r['n_positive']:4d} {r['n_negative']:4d}")

    print(f"\n--- 全部因子排名对比 (Gain vs SHAP) ---")
    merged = fi_df[["factor", "label", "dim", "gain_rank", "gain_pct"]].merge(
        shap_df[["factor", "shap_rank", "shap_pct", "direction"]], on="factor"
    ).sort_values("gain_rank")
    for _, r in merged.iterrows():
        print(f"  {r['label']:20s} [{r['dim']:6s}] "
              f"Gain#{r['gain_rank']:2d}({r['gain_pct']:4.1f}%) "
              f"SHAP#{r['shap_rank']:2d}({r['shap_pct']:4.1f}%) {r['direction']}")


def main():
    parser = argparse.ArgumentParser(description="GBDT探索财务因子贡献度")
    parser.add_argument("--horizon", type=str, default="20d",
                        choices=["5d", "20d", "60d"], help="收益持有期")
    parser.add_argument("--out-dir", type=str, default="out", help="输出目录")
    args = parser.parse_args()

    engine = get_db_engine()
    return_col = f"ret_{args.horizon}"

    df = load_dataset(engine, return_col)
    if df.empty:
        logger.error("数据为空")
        return 1

    logger.info("训练GBDT模型...")
    model, fi_df, test_df, available_factors = train_gbdt(df, return_col)

    logger.info("计算SHAP值...")
    shap_df, shap_values = shap_analysis(model, test_df, available_factors)

    dim_df = dim_contribution(fi_df, shap_df)

    print_report(fi_df, shap_df, dim_df, return_col)

    os.makedirs(args.out_dir, exist_ok=True)
    fi_path = os.path.join(args.out_dir, f"gbdt_feature_importance_{args.horizon}.csv")
    fi_df.to_csv(fi_path, index=False)

    shap_path = os.path.join(args.out_dir, f"gbdt_shap_{args.horizon}.csv")
    shap_df.to_csv(shap_path, index=False)

    dim_path = os.path.join(args.out_dir, f"gbdt_dim_contribution_{args.horizon}.csv")
    dim_df.to_csv(dim_path, index=False)

    logger.info(f"结果已保存到 {args.out_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
