#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
db_score.py vs sample_score.py 数据源一致性对比测试

目的：以 Tushare 单季度数据为基准，验证同花顺 DB 数据（经过 db_score YTD 差分还原后）是否一致。

数据源：
- db_score.py: 同花顺 DB（YTD累计 → convert_flows_to_single_quarter() 还原）
- sample_score.py: Tushare API（直接单季度，report_type=2）

对比层次：
A. 季度基础字段 (rev_q, np_parent_q, cfo_q 等)
B. YTD/TTM 累计值
C. 因子原始值（35个因子）
D. 因子得分 + 维度分 + 总分

前置条件：需要 TUSHARE_TOKEN 环境变量
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TS_CODE = "000426.SZ"
STOCK_NAME = "兴业银锡"
START_DATE = "20200101"
TARGET_QUARTER = (2025, 3)
LOOKBACK = 12

COMPARE_COLS_A = ["rev_q", "cost_q", "op_q", "ebit_q", "np_parent_q", "cfo_q", "capex_q"]
COMPARE_COLS_B = [
    "rev_ytd", "np_parent_ytd", "cfo_ytd",
    "rev_ttm", "np_parent_ttm", "cfo_ttm", "fcf_ttm", "capex_ttm",
    "cost_ttm", "avg_assets",
]
COMPARE_COLS_C = [
    "q_rev_yoy", "q_op_yoy", "q_np_parent_yoy", "q_ebit_yoy",
    "q_rev_qoq", "q_op_qoq",
    "ytd_rev_yoy", "ytd_np_parent_yoy",
    "q_gross_margin", "q_gm_yoy_change", "q_gm_qoq_change",
    "q_op_margin", "op_margin_change", "q_np_parent_margin", "q_ebit_margin",
    "q_cfo_to_np_parent", "ttm_cfo_to_np_parent", "q_accruals_to_assets",
    "ttm_cfo_to_ebit", "q_np_parent_to_np",
    "q_cfo_to_rev", "q_cfo_yoy", "ytd_cfo_yoy",
    "ttm_fcf_to_np_parent", "capex_to_cfo",
    "cash_sales_ratio", "cash_sales_yoy",
    "roa_parent", "cfo_to_assets", "asset_turnover", "ccc", "contract_liab_to_rev",
    "q_rev_yoy_delta", "q_np_parent_yoy_delta",
    "trend_consistency", "profit_cash_sync", "margin_profit_sync", "cfo_to_np_change",
]
DIM_SCORE_COLS = [
    "规模与增长_score", "盈利能力_score", "利润质量_score",
    "现金创造能力_score", "资产效率与资金占用_score", "边际变化与持续性_score",
]


def rel_diff(v1, v2):
    if pd.isna(v1) and pd.isna(v2):
        return 0.0
    denom = (abs(v1) + abs(v2)) / 2
    if denom < 1e-10:
        return abs(v1 - v2)
    return abs(v1 - v2) / denom


def align_to_quarter(df: pd.DataFrame, year: int, quarter: int) -> pd.Series:
    mask = (df["fiscal_year"] == year) & (df["quarter"] == quarter)
    if mask.sum() == 0:
        available = df[["fiscal_year", "quarter"]].drop_duplicates().tail(6)
        raise ValueError(
            f"在 DataFrame 中未找到 {year}Q{quarter}，可用季度：\n{available.to_string()}"
        )
    return df[mask].iloc[0]


def compare_row(row_db: pd.Series, row_ts: pd.Series, cols: List[str], tol: float = 0.01) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in row_db.index or col not in row_ts.index:
            continue
        v_db = row_db[col]
        v_ts = row_ts[col]
        if pd.isna(v_db) and pd.isna(v_ts):
            ok = True
            diff_str = "both NaN"
        elif pd.isna(v_db) or pd.isna(v_ts):
            ok = False
            diff_str = f"one is NaN (db={v_db}, ts={v_ts})"
        else:
            rd = rel_diff(v_db, v_ts)
            ok = rd < tol
            diff_str = f"abs={abs(v_db-v_ts):.4g}, rel={rd:.2%}"
        rows.append({
            "字段": col,
            "同花顺(db)": round(v_db, 6) if not pd.isna(v_db) else np.nan,
            "Tushare": round(v_ts, 6) if not pd.isna(v_ts) else np.nan,
            "差异说明": diff_str,
            "一致(1%)": "✓" if ok else "✗",
        })
    return pd.DataFrame(rows)


def main():
    print(f"\n{'='*70}")
    print(f"db_score vs sample_score 一致性对比测试")
    print(f"股票: {STOCK_NAME}（{TS_CODE}）目标季度: {TARGET_QUARTER[0]}Q{TARGET_QUARTER[1]}")
    print(f"{'='*70}\n")

    sys.path.insert(0, ".")
    from financial_factors import db_score, sample_score

    db_score.TARGET_TS_CODE = TS_CODE
    db_score.TARGET_NAME = STOCK_NAME
    sample_score.TARGET_TS_CODE = TS_CODE
    sample_score.TARGET_NAME = STOCK_NAME

    print("[1/5] 获取 db_score（同花顺DB）中间结果 ...")
    df_db = db_score.prepare_base_dataframe(ts_code=TS_CODE, start_date=START_DATE)
    df_db = db_score.add_ytd_and_ttm(df_db)
    df_db = db_score.add_factors(df_db)
    scored_db = db_score.score_dataframe(df_db, lookback=LOOKBACK)
    print(f"    同花顺 DB: {len(df_db)} 条季度记录")
    print(f"    最新季度: {df_db['fiscal_year'].iloc[-1]}Q{df_db['quarter'].iloc[-1]}")

    print("\n[2/5] 获取 sample_score（Tushare API）中间结果 ...")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=os.getenv("TUSHARE_TOKEN", ""),
                        help="Tushare token（优先取命令行参数，其次环境变量 TUSHARE_TOKEN）")
    parser.add_argument("--ts-code", type=str, default=TS_CODE)
    parser.add_argument("--name", type=str, default=STOCK_NAME)
    parser.add_argument("--quarter", type=str, default=f"{TARGET_QUARTER[0]}Q{TARGET_QUARTER[1]}")
    args = parser.parse_args()
    token = args.token
    if not token:
        raise ValueError("需要 --token 参数或设置 TUSHARE_TOKEN 环境变量")
    pro = sample_score.get_pro(token)
    df_ts = sample_score.prepare_base_dataframe(
        pro=pro, ts_code=TS_CODE, start_date=START_DATE, source="tushare"
    )
    df_ts = sample_score.add_ytd_and_ttm(df_ts)
    df_ts = sample_score.add_factors(df_ts)
    scored_ts = sample_score.score_dataframe(df_ts, lookback=LOOKBACK)
    print(f"    Tushare: {len(df_ts)} 条季度记录")
    print(f"    最新季度: {df_ts['fiscal_year'].iloc[-1]}Q{df_ts['quarter'].iloc[-1]}")

    print("\n[3/5] 按报告期季度对齐数据 ...")
    year, quarter = TARGET_QUARTER
    row_db = align_to_quarter(df_db, year, quarter)
    row_ts = align_to_quarter(df_ts, year, quarter)
    s_db = align_to_quarter(scored_db, year, quarter)
    s_ts = align_to_quarter(scored_ts, year, quarter)
    print(f"    已对齐: {year}Q{quarter}")
    print(f"    同花顺 end_date: {row_db['end_date']}")
    print(f"    Tushare  end_date: {row_ts['end_date']}")

    print(f"\n[4/5] 对比层次 A-D ...")

    print(f"\n{'='*70}")
    print(f"层次 A: 单季度基础字段 (相对误差容忍度 1%)")
    print(f"{'='*70}")
    df_a = compare_row(row_db, row_ts, COMPARE_COLS_A, tol=0.01)
    print(df_a.to_string(index=False))
    n_ok_a = (df_a["一致(1%)"] == "✓").sum()
    print(f"  -> 一致 {n_ok_a}/{len(df_a)} 项")

    print(f"\n{'='*70}")
    print(f"层次 B: YTD/TTM 累计值 (相对误差容忍度 1%)")
    print(f"{'='*70}")
    df_b = compare_row(row_db, row_ts, COMPARE_COLS_B, tol=0.01)
    print(df_b.to_string(index=False))
    n_ok_b = (df_b["一致(1%)"] == "✓").sum()
    print(f"  -> 一致 {n_ok_b}/{len(df_b)} 项")

    print(f"\n{'='*70}")
    print(f"层次 C: 因子原始值 (相对误差容忍度 1%)")
    print(f"{'='*70}")
    df_c = compare_row(row_db, row_ts, COMPARE_COLS_C, tol=0.01)
    print(df_c.to_string(index=False))
    n_ok_c = (df_c["一致(1%)"] == "✓").sum()
    print(f"  -> 一致 {n_ok_c}/{len(df_c)} 项")

    print(f"\n{'='*70}")
    print(f"层次 D: 因子得分 (绝对误差容忍度 0.1分)")
    print(f"{'='*70}")
    score_cols = [f"{c}_score" for c in COMPARE_COLS_C]
    score_rows = []
    for col in score_cols:
        c_base = col.replace("_score", "")
        v_db = s_db[col] if col in s_db.index else np.nan
        v_ts = s_ts[col] if col in s_ts.index else np.nan
        if pd.isna(v_db) and pd.isna(v_ts):
            ok = True
            diff_str = "both NaN"
        elif pd.isna(v_db) or pd.isna(v_ts):
            ok = False
            diff_str = f"one is NaN"
        else:
            diff = abs(v_db - v_ts)
            ok = diff < 0.1
            diff_str = f"abs_diff={diff:.4f}"
        score_rows.append({
            "因子": c_base,
            "同花顺得分": round(v_db, 2) if not pd.isna(v_db) else np.nan,
            "Tushare得分": round(v_ts, 2) if not pd.isna(v_ts) else np.nan,
            "差异说明": diff_str,
            "一致(0.1)": "✓" if ok else "✗",
        })
    df_d = pd.DataFrame(score_rows)
    print(df_d.to_string(index=False))
    n_ok_d = (df_d["一致(0.1)"] == "✓").sum()
    print(f"  -> 一致 {n_ok_d}/{len(df_d)} 项")

    print(f"\n{'='*70}")
    print(f"层次 D: 维度分 + 总分 (绝对误差容忍度 0.1分)")
    print(f"{'='*70}")
    dim_rows = []
    for dim_col in DIM_SCORE_COLS:
        v_db = s_db[dim_col] if dim_col in s_db.index else np.nan
        v_ts = s_ts[dim_col] if dim_col in s_ts.index else np.nan
        if pd.isna(v_db) and pd.isna(v_ts):
            ok = True
        elif pd.isna(v_db) or pd.isna(v_ts):
            ok = False
        else:
            diff = abs(v_db - v_ts)
            ok = diff < 0.1
        dim_rows.append({
            "维度": dim_col.replace("_score", ""),
            "同花顺": round(v_db, 2) if not pd.isna(v_db) else np.nan,
            "Tushare": round(v_ts, 2) if not pd.isna(v_ts) else np.nan,
            "差异": round(abs(v_db - v_ts), 4) if not (pd.isna(v_db) or pd.isna(v_ts)) else np.nan,
            "一致": "✓" if ok else "✗",
        })

    v_db_total = s_db["total_score"] if "total_score" in s_db.index else np.nan
    v_ts_total = s_ts["total_score"] if "total_score" in s_ts.index else np.nan
    total_diff = abs(v_db_total - v_ts_total) if not (pd.isna(v_db_total) or pd.isna(v_ts_total)) else np.nan
    dim_rows.append({
        "维度": "总分",
        "同花顺": round(v_db_total, 2) if not pd.isna(v_db_total) else np.nan,
        "Tushare": round(v_ts_total, 2) if not pd.isna(v_ts_total) else np.nan,
        "差异": round(total_diff, 4) if not pd.isna(total_diff) else np.nan,
        "一致": "✓" if (not pd.isna(total_diff) and total_diff < 0.1) else "✗",
    })
    df_dim = pd.DataFrame(dim_rows)
    print(df_dim.to_string(index=False))
    n_ok_dim = (df_dim["一致"] == "✓").sum()
    print(f"  -> 一致 {n_ok_dim}/{len(df_dim)} 项")

    print(f"\n{'='*70}")
    print(f"最终结论")
    print(f"{'='*70}")
    total_items = len(df_a) + len(df_b) + len(df_c)
    total_ok = n_ok_a + n_ok_b + n_ok_c
    overall_pct = total_ok / total_items * 100 if total_items > 0 else 0
    print(f"层次 A(季度基础字段): {n_ok_a}/{len(df_a)} 一致")
    print(f"层次 B(YTD/TTM累计值): {n_ok_b}/{len(df_b)} 一致")
    print(f"层次 C(因子原始值): {n_ok_c}/{len(df_c)} 一致")
    print(f"层次 D(得分): {n_ok_d}/{len(df_d)} 因子分, {n_ok_dim-1}/{len(df_dim)-1} 维度分, "
          f"总分差={total_diff:.2f}分" if not pd.isna(total_diff) else "  总分: NaN")
    print(f"\n综合一致性: {overall_pct:.1f}% ({total_ok}/{total_items})")

    if n_ok_a == len(df_a) and n_ok_b == len(df_b) and n_ok_c == len(df_c):
        if total_diff is not None and total_diff < 0.1:
            print(f"\n【结论】✓ 两套数据源高度一致，同花顺 DB 质量良好，db_score 转换逻辑正确。")
        else:
            print(f"\n【结论】△ 因子原始值一致但评分略有差异（总分差={total_diff:.2f}分），"
                  f"可能原因：lookback窗口数据量不同。")
    else:
        inconsistent = df_a[df_a["一致(1%)"] == "✗"]["字段"].tolist() + \
                       df_b[df_b["一致(1%)"] == "✗"]["字段"].tolist() + \
                       df_c[df_c["一致(1%)"] == "✗"]["字段"].tolist()
        print(f"\n【结论】✗ 发现 {len(inconsistent)} 个不一致字段：{inconsistent}")
        print("可能原因：")
        print("  1. 同花顺原始 YTD 数据与 Tushare 单季度数据本身存在差异（正常，不同数据源）")
        print("  2. db_score 的 YTD 差分还原逻辑问题（若差异较大）")
        print("  3. 报告期日期不一致（如 09-30 vs 09-28）导致的季度归属差异")


if __name__ == "__main__":
    main()
