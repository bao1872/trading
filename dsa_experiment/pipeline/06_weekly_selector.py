#!/usr/bin/env python3
"""
DSA V型反转精选股票方案（增量价值验证版）

Purpose: 基于 return_model + risk_model 的精选股票方案，BBMACD召回+GBDT排序提纯+风险veto
Inputs: stock_dsa_vreversal_results (DB), return_model/model.txt, risk_model/model.txt
Outputs: 选股排名 CSV
How to Run:
    python dsa_experiment/pipeline/06_weekly_selector.py --date 2026-04-30
    python dsa_experiment/pipeline/06_weekly_selector.py --date 2026-04-30 --top-n 20
    python dsa_experiment/pipeline/06_weekly_selector.py --date 2026-04-30 --veto 0.20
Examples:
    python dsa_experiment/pipeline/06_weekly_selector.py --date 2026-04-30
    python dsa_experiment/pipeline/06_weekly_selector.py --date 2026-04-30 --top-n 10 --output /root/trading/dsa_selection.csv
Side Effects: 只读操作，不写数据库

管线位置: Step 6/7 — 周线精选选股（GBDT排序+档位+veto）

================================================================================
【选股方案设计（对齐 advicement.txt 建议）】

一、触发条件：BBMacd V型反转（高召回触发器）
  bbmacd[t] > bbmacd[t-1] 且 bbmacd[t-1] < bbmacd[t-2]

二、硬过滤：不使用
  实验验证：5项硬过滤保留率仅3%，胜率从69%降到50%，严重错杀。
  GBDT 排序已能从高噪音池中提纯，无需硬过滤。

三、评分模型（实验验证版）：
  return_model: LightGBM 回归模型（IC=0.19, ICIR=9.14）→ opportunity_score
  risk_model:   LightGBM 二分类模型（AUC=0.67）→ risk_score
  定位: BBMACD 召回 + GBDT 排序提纯 + 风险轻 veto

四、档位系统（方案B，软过滤）：
  A档: opportunity_score 前20% 且 risk_score 后50%（高机会低风险）
  B档: opportunity_score 前20% 且 risk_score 前50%（高机会高风险）
  C档: opportunity_score 后80% 且 risk_score 后50%（低机会低风险）
  D档: 其余（低机会高风险）

五、风险 veto（方案C，轻 veto）：
  剔除 risk_score 最高的 veto_pct 比例，默认 20%

六、输出：按 opportunity_score 排序，标注档位和风控标记
================================================================================
"""

import sys
import os
import argparse
from datetime import datetime, date

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "output")

RAW_FEATURES = [
    "dsa_dir", "prev_pivot_code", "last_confirmed_high", "last_confirmed_low",
    "dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct", "current_stage_bars", "prev_stage_bars",
    "bars_since_last_high", "bars_since_last_low", "prev_stage_amp_pct",
    "current_stage_ret_pct", "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct", "bbmacd", "bbmacd_minus_avg",
    "bbmacd_state", "bbmacd_band_pos_01", "bbmacd_bandwidth_zscore",
    "bbmacd_cross_upper", "bbmacd_cross_lower", "trend_align_momo",
    "vol_zscore_5", "vol_zscore_10", "vol_zscore_20", "vol_ratio_10",
    "vol_stage_cv", "vol_prev_stage_cv", "vol_cv_ratio",
    "price_vol_coord", "momo_vol_coord", "low_pos_break_coord", "coord_consistency",
    "coord_stage_current", "coord_stage_prev", "coord_stage_ratio",
]

CATEGORICAL_FEATURES = []

DERIVED_FEATURES = [
    "high_low_range", "high_low_range_pct",
    "pivot_pos_x_trend", "stage_bars_ratio", "amp_x_pullback", "bbmacd_band_width",
]

ALL_FEATURES = RAW_FEATURES + DERIVED_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high = df["last_confirmed_high"].fillna(0)
    low = df["last_confirmed_low"].fillna(0)
    df["high_low_range"] = high - low
    df["high_low_range_pct"] = np.where(low > 0, (high - low) / low, 0)
    df["pivot_pos_x_trend"] = df["dsa_pivot_pos_01"].fillna(0) * df["trend_align_momo"].fillna(0)
    prev_bars = df["prev_stage_bars"].fillna(0).replace(0, np.nan)
    df["stage_bars_ratio"] = df["current_stage_bars"].fillna(0) / prev_bars
    df["stage_bars_ratio"] = df["stage_bars_ratio"].fillna(0)
    df["amp_x_pullback"] = df["current_stage_amp_pct"].fillna(0) * df["current_pullback_from_stage_extreme_pct"].fillna(0)
    df["bbmacd_band_width"] = df["bbmacd_band_pos_01"].fillna(0) * df["bbmacd_bandwidth_zscore"].fillna(0)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    valid = []
    for col in ALL_FEATURES:
        if col not in df.columns:
            continue
        if df[col].isna().all():
            continue
        valid.append(col)
    return valid


def load_experiment_model(model_name: str) -> lgb.Booster:
    model_path = os.path.join(OUTPUT_DIR, model_name, "model.txt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    return lgb.Booster(model_file=model_path)


def assign_grade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["grade"] = "D"
        return df

    opp_p80 = df["opportunity_score"].quantile(0.80)
    risk_p50 = df["risk_score"].quantile(0.50)

    df["grade"] = "D"
    df.loc[
        (df["opportunity_score"] >= opp_p80) & (df["risk_score"] <= risk_p50),
        "grade",
    ] = "A"
    df.loc[
        (df["opportunity_score"] >= opp_p80) & (df["risk_score"] > risk_p50),
        "grade",
    ] = "B"
    df.loc[
        (df["opportunity_score"] < opp_p80) & (df["risk_score"] <= risk_p50),
        "grade",
    ] = "C"

    return df


def apply_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_flag"] = ""
    df.loc[df["ret_to_last_low_pct"] > 0.3, "risk_flag"] += "离低点>30% "
    df.loc[df["current_stage_ret_pct"] > 0.5, "risk_flag"] += "阶段涨幅>50% "
    df.loc[df["current_stage_amp_pct"] > 1.0, "risk_flag"] += "振幅>100% "
    if df["bbmacd"].notna().sum() > 10:
        bbmacd_p90 = df["bbmacd"].quantile(0.9)
        df.loc[df["bbmacd"] > bbmacd_p90, "risk_flag"] += "bbmacd过大 "
    return df


def fetch_stock_names(ts_codes: list) -> dict:
    if not ts_codes:
        return {}
    all_codes = []
    for code in ts_codes:
        if code.startswith(("6", "5")):
            all_codes.append(f"{code}.SH")
        else:
            all_codes.append(f"{code}.SZ")
    placeholders = ", ".join([f"'{c}'" for c in all_codes])
    try:
        sql = text(f"SELECT ts_code, name FROM stock_pools WHERE ts_code IN ({placeholders})")
        with engine.connect() as conn:
            result = conn.execute(sql)
            return {row[0].split(".")[0]: row[1] for row in result}
    except Exception:
        return {}


def run_selection(
    selection_date: date,
    top_n: int = 20,
    veto_pct: float = 0.20,
    output: str = "",
) -> pd.DataFrame:
    print(f"\n{'=' * 80}")
    print(f"DSA V型反转精选股票方案 — {selection_date}")
    print(f"{'=' * 80}")

    print("\n  [1/7] 读取触发点...")
    sql = text("""
        SELECT * FROM stock_dsa_vreversal_results
        WHERE selection_date = :sel_date
    """)
    df = pd.read_sql(sql, engine, params={"sel_date": selection_date.strftime("%Y-%m-%d")})
    if df.empty:
        print(f"  {selection_date} 无触发点记录")
        return pd.DataFrame()
    print(f"  触发点: {len(df)}")

    print("  [2/7] 特征工程...")
    df = build_features(df)
    feature_cols = get_feature_cols(df)
    print(f"  特征数: {len(feature_cols)}")

    print("  [3/7] GBDT 评分...")
    try:
        return_model = load_experiment_model("return_model")
        risk_model = load_experiment_model("risk_model")
        X = df[feature_cols].fillna(0)
        df["opportunity_score"] = return_model.predict(X, num_iteration=return_model.best_iteration)
        df["risk_score"] = risk_model.predict(X, num_iteration=risk_model.best_iteration)
        print(f"  return_model: ✅ (best_iter={return_model.best_iteration})")
        print(f"  risk_model:   ✅ (best_iter={risk_model.best_iteration})")
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return pd.DataFrame()

    print("  [4/7] 档位分配...")
    df = assign_grade(df)
    grade_counts = df["grade"].value_counts().to_dict()
    for g in ["A", "B", "C", "D"]:
        print(f"    {g}档: {grade_counts.get(g, 0)} 只")

    print("  [5/7] 风险 veto...")
    veto_cutoff = df["risk_score"].quantile(1 - veto_pct)
    df["vetoed"] = df["risk_score"] > veto_cutoff
    n_vetoed = df["vetoed"].sum()
    print(f"  veto 阈值: risk_score > {veto_cutoff:.4f} (前{veto_pct:.0%})")
    print(f"  被 veto: {n_vetoed}/{len(df)}")

    print("  [6/7] 风控标记...")
    df = apply_risk_flags(df)

    print("  [7/7] 排序输出...")
    candidates = df[~df["vetoed"]].copy()
    candidates = candidates.sort_values("opportunity_score", ascending=False)

    names = fetch_stock_names(candidates["ts_code"].unique().tolist())
    candidates["stock_name"] = candidates["ts_code"].map(names)

    print(f"\n  精选 Top {min(top_n, len(candidates))}（veto后 {len(candidates)} 只可选）:")
    print(f"  {'排名':>4}  {'代码':>6}  {'名称':>8}  {'收盘':>8}  {'机会分':>8}  {'风险分':>8}  {'档位':>4}  {'VWAP偏离':>8}  {'距高点':>8}  {'距低点':>8}  {'风控':>20}")
    print("  " + "-" * 120)

    top = candidates.head(top_n)
    for i, (_, row) in enumerate(top.iterrows()):
        code = row["ts_code"]
        name = str(row.get("stock_name", ""))[:8]
        close = row.get("trigger_close", np.nan)
        opp = row.get("opportunity_score", np.nan)
        risk = row.get("risk_score", np.nan)
        grade = row.get("grade", "")
        vwap = row.get("price_vs_dsa_vwap_pct", np.nan)
        rth = row.get("ret_to_last_high_pct", np.nan)
        rtl = row.get("ret_to_last_low_pct", np.nan)
        flag = str(row.get("risk_flag", "")).strip()

        s_close = f"{close:.2f}" if not np.isnan(close) else "N/A"
        s_opp = f"{opp:.4f}" if not np.isnan(opp) else "N/A"
        s_risk = f"{risk:.4f}" if not np.isnan(risk) else "N/A"
        s_vwap = f"{vwap:+.4f}" if not np.isnan(vwap) else "N/A"
        s_rth = f"{rth:+.2%}" if not np.isnan(rth) else "N/A"
        s_rtl = f"{rtl:+.2%}" if not np.isnan(rtl) else "N/A"

        print(f"  {i+1:>4}  {code:>6}  {name:>8}  {s_close:>8}  {s_opp:>8}  {s_risk:>8}  {grade:>4}  {s_vwap:>8}  {s_rth:>8}  {s_rtl:>8}  {flag:>20}")

    print(f"\n  --- 统计 ---")
    print(f"  触发点: {len(df)}, veto后可选: {len(candidates)}")
    print(f"  A档: {grade_counts.get('A', 0)}, B档: {grade_counts.get('B', 0)}, C档: {grade_counts.get('C', 0)}, D档: {grade_counts.get('D', 0)}")
    print(f"  有风控标记: {(candidates['risk_flag'] != '').sum()}/{len(candidates)}")

    if output:
        save_cols = [
            "ts_code", "stock_name", "opportunity_score", "risk_score",
            "grade", "vetoed", "risk_flag",
            "trigger_close", "price_vs_dsa_vwap_pct",
            "ret_to_last_high_pct", "ret_to_last_low_pct",
            "current_stage_amp_pct", "current_stage_ret_pct",
            "bbmacd", "trend_align_momo", "bbmacd_state",
        ]
        avail = [c for c in save_cols if c in candidates.columns]
        candidates[avail].to_csv(output, index=False, encoding="utf-8-sig")
        print(f"\n  已保存: {output} ({len(candidates)} 只)")

    return candidates


def main():
    parser = argparse.ArgumentParser(description="DSA V型反转精选股票方案（增量价值验证版）")
    parser.add_argument("--date", type=str, required=True, help="选股日期 (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=20, help="输出Top N，默认20")
    parser.add_argument("--veto", type=float, default=0.20, help="风险veto比例，默认0.20")
    parser.add_argument("--output", type=str, default="", help="输出CSV路径")
    args = parser.parse_args()

    selection_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    run_selection(selection_date, args.top_n, args.veto, args.output)


if __name__ == "__main__":
    main()
