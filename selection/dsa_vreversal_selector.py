#!/usr/bin/env python3
"""
DSA V型反转完整选股方案

Purpose: 基于 IC 分析 + GBDT 探索 + SHAP 分析的综合选股模型
Inputs: stock_dsa_vreversal_results (选股结果表) + GBDT 模型文件
Outputs: 选股排名 CSV
How to Run:
    python selection/dsa_vreversal_selector.py --date 2026-04-30
    python selection/dsa_vreversal_selector.py --date 2026-04-30 --model-dir /tmp/dsa_vreversal_gbdt_full
    python selection/dsa_vreversal_selector.py --date 2026-04-30 --top-n 20
Examples:
    python selection/dsa_vreversal_selector.py --date 2026-04-30
    python selection/dsa_vreversal_selector.py --date 2026-04-30 --top-n 10 --output /root/trading/dsa_selection.csv
Side Effects: 只读操作，不写数据库

================================================================================
【选股方案设计】

一、触发条件：BBMacd V型反转
  bbmacd[t] > bbmacd[t-1] 且 bbmacd[t-1] < bbmacd[t-2]

二、硬过滤（必要条件，全部用选股时已知因子）：
  1. trend_align_momo == 1    — 趋势-动量同向（资金效率 1.76x）
  2. ret_to_last_high_pct < -0.05 — 距确认高点至少5%空间（避免追高）
  3. bbmacd_state >= 0        — BBMacd不在带下（避免弱势）
  4. ret_to_last_low_pct < 0.5   — 离确认低点未涨超50%（控制回撤风险）
  5. current_stage_amp_pct < 2.0 — 当前振幅未超200%（控制波动风险）

三、评分模型（双模型组合）：
  模型A: is_profitable GBDT (AUC=0.88) — 预测盈利>10%的概率
  模型B: is_low_drawdown GBDT (AUC=0.82) — 预测回撤<10%的概率
  综合评分 = 0.6 * P(盈利) + 0.4 * P(低回撤)

四、风控标记（软约束，标注但不禁入）：
  - 离低点>30%：中等回撤风险
  - 阶段涨幅>50%：可能透支
  - 振幅>100%：波动较大
  - bbmacd > 训练集90分位数：动量过大（SHAP反向效应）

五、输出：按综合评分降序排列，取 Top N
================================================================================
"""

import sys
import os
import argparse
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

RAW_FEATURES = [
    "dsa_dir", "prev_pivot_code", "last_confirmed_high", "last_confirmed_low",
    "dsa_pivot_pos_01", "ret_to_last_high_pct", "ret_to_last_low_pct",
    "price_vs_dsa_vwap_pct", "current_stage_bars", "prev_stage_bars",
    "bars_since_last_high", "bars_since_last_low", "prev_stage_amp_pct",
    "current_stage_ret_pct", "current_stage_amp_pct",
    "current_pullback_from_stage_extreme_pct", "bbmacd", "bbmacd_minus_avg",
    "bbmacd_band_pos_01", "bbmacd_bandwidth_zscore",
    "bbmacd_cross_upper", "bbmacd_cross_lower",
]

CATEGORICAL_FEATURES = ["bbmacd_state", "trend_align_momo"]

DERIVED_FEATURES = [
    "high_low_range", "high_low_range_pct", "vwap_dev_x_bbmacd",
    "pivot_pos_x_trend", "stage_bars_ratio", "amp_x_pullback", "bbmacd_band_width",
]

ALL_FEATURES = RAW_FEATURES + CATEGORICAL_FEATURES + DERIVED_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high = df["last_confirmed_high"].fillna(0)
    low = df["last_confirmed_low"].fillna(0)
    df["high_low_range"] = high - low
    df["high_low_range_pct"] = np.where(low > 0, (high - low) / low, 0)
    df["vwap_dev_x_bbmacd"] = df["price_vs_dsa_vwap_pct"].fillna(0) * df["bbmacd"].fillna(0)
    df["pivot_pos_x_trend"] = df["dsa_pivot_pos_01"].fillna(0) * df["trend_align_momo"].fillna(0)
    prev_bars = df["prev_stage_bars"].fillna(0).replace(0, np.nan)
    df["stage_bars_ratio"] = df["current_stage_bars"].fillna(0) / prev_bars
    df["stage_bars_ratio"] = df["stage_bars_ratio"].fillna(0)
    df["amp_x_pullback"] = df["current_stage_amp_pct"].fillna(0) * df["current_pullback_from_stage_extreme_pct"].fillna(0)
    df["bbmacd_band_width"] = df["bbmacd_band_pos_01"].fillna(0) * df["bbmacd_bandwidth_zscore"].fillna(0)
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    valid = []
    for col in ALL_FEATURES:
        if col not in df.columns:
            continue
        if df[col].isna().all():
            continue
        valid.append(col)
    return valid


def load_model(model_dir: str, target: str) -> Optional[lgb.Booster]:
    model_path = os.path.join(model_dir, target, "dsa_gbdt_model.txt")
    if os.path.exists(model_path):
        return lgb.Booster(model_file=model_path)
    return None


def hard_filter(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    mask &= (df["trend_align_momo"] == 1)
    mask &= (df["ret_to_last_high_pct"] < -0.05)
    mask &= (df["bbmacd_state"] >= 0)
    mask &= (df["ret_to_last_low_pct"] < 0.5)
    mask &= (df["current_stage_amp_pct"] < 2.0)
    return df[mask].copy()


def apply_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_flag"] = ""
    df.loc[df["ret_to_last_low_pct"] > 0.3, "risk_flag"] += "离低点>30% "
    df.loc[df["current_stage_ret_pct"] > 0.5, "risk_flag"] += "阶段涨幅>50% "
    df.loc[df["current_stage_amp_pct"] > 1.0, "risk_flag"] += "振幅>100% "
    bbmacd_p90 = df["bbmacd"].quantile(0.9) if df["bbmacd"].notna().sum() > 10 else 999
    df.loc[df["bbmacd"] > bbmacd_p90, "risk_flag"] += "bbmacd过大 "
    return df


def score_with_gbdt(
    df: pd.DataFrame, feature_cols: List[str],
    profit_model: Optional[lgb.Booster], drawdown_model: Optional[lgb.Booster],
) -> pd.DataFrame:
    df = df.copy()
    X = df[feature_cols].fillna(0)

    if profit_model is not None:
        df["p_profit"] = profit_model.predict(X, num_iteration=profit_model.best_iteration)
    else:
        df["p_profit"] = np.nan

    if drawdown_model is not None:
        df["p_low_drawdown"] = drawdown_model.predict(X, num_iteration=drawdown_model.best_iteration)
    else:
        df["p_low_drawdown"] = np.nan

    has_both = df["p_profit"].notna() & df["p_low_drawdown"].notna()
    df["composite_score"] = np.nan
    df.loc[has_both, "composite_score"] = (
        0.6 * df.loc[has_both, "p_profit"] + 0.4 * df.loc[has_both, "p_low_drawdown"]
    )

    if profit_model is not None and drawdown_model is None:
        df["composite_score"] = df["p_profit"]
    elif drawdown_model is not None and profit_model is None:
        df["composite_score"] = df["p_low_drawdown"]

    return df


def score_with_ic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score_factors = {
        "current_stage_amp_pct": +0.17,
        "ret_to_last_high_pct": -0.17,
        "bars_since_last_low": -0.10,
        "ret_to_last_low_pct": -0.07,
        "current_stage_ret_pct": -0.06,
        "bbmacd": +0.06,
        "bbmacd_minus_avg": +0.04,
        "bbmacd_bandwidth_zscore": +0.04,
    }
    for col in score_factors:
        if col in df.columns:
            df[f"rank_{col}"] = df[col].rank(pct=True)
    df["ic_score"] = 0.0
    for col, weight in score_factors.items():
        if f"rank_{col}" in df.columns:
            df["ic_score"] += weight * df[f"rank_{col}"]
    return df


def fetch_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    if not ts_codes:
        return {}
    codes_sh = [f"{c}.SH" for c in ts_codes]
    codes_sz = [f"{c}.SZ" for c in ts_codes]
    all_codes = codes_sh + codes_sz
    placeholders = ", ".join([f"'{c}'" for c in all_codes])
    try:
        sql = text(f"SELECT ts_code, name FROM stock_pools WHERE ts_code IN ({placeholders})")
        with engine.connect() as conn:
            result = conn.execute(sql)
            name_map = {}
            for row in result:
                code = row[0].split(".")[0]
                name_map[code] = row[1]
            return name_map
    except Exception:
        return {}


def run_selection(
    selection_date: date,
    model_dir: str,
    top_n: int = 30,
    output: str = "",
) -> pd.DataFrame:
    print(f"\n{'=' * 80}")
    print(f"DSA V型反转选股 — {selection_date}")
    print(f"{'=' * 80}")

    # 1. 读取触发点
    sql = text("""
        SELECT * FROM stock_dsa_vreversal_results
        WHERE selection_date = :sel_date
    """)
    df = pd.read_sql(sql, engine, params={"sel_date": selection_date.strftime("%Y-%m-%d")})
    if df.empty:
        print(f"  {selection_date} 无触发点记录")
        return pd.DataFrame()
    print(f"  触发点: {len(df)}")

    # 2. 硬过滤
    filtered = hard_filter(df)
    print(f"  硬过滤后: {len(filtered)}")
    if filtered.empty:
        print("  无符合条件的股票")
        return pd.DataFrame()

    # 3. 特征工程
    filtered = build_features(filtered)
    feature_cols = get_feature_cols(filtered)

    # 4. GBDT 评分
    profit_model = load_model(model_dir, "is_profitable")
    drawdown_model = load_model(model_dir, "is_low_drawdown")

    if profit_model is not None or drawdown_model is not None:
        filtered = score_with_gbdt(filtered, feature_cols, profit_model, drawdown_model)
        score_col = "composite_score"
        score_label = "GBDT综合评分"
        print(f"  GBDT模型: is_profitable={'✅' if profit_model else '❌'}, is_low_drawdown={'✅' if drawdown_model else '❌'}")
    else:
        print("  未找到GBDT模型，使用IC加权评分")
        score_col = "ic_score"
        score_label = "IC加权评分"

    # 5. IC加权评分（作为备选/对比）
    filtered = score_with_ic(filtered)

    # 6. 风控标记
    filtered = apply_risk_flags(filtered)

    # 7. 排序
    filtered = filtered.sort_values(score_col, ascending=False)

    # 8. 股票名称
    names = fetch_stock_names(filtered["ts_code"].unique().tolist())
    filtered["stock_name"] = filtered["ts_code"].map(names)

    # 9. 输出
    display_cols = [
        "ts_code", "stock_name", "trigger_close", score_col,
        "p_profit", "p_low_drawdown", "ic_score",
        "trend_align_momo", "bbmacd_state",
        "current_stage_amp_pct", "ret_to_last_high_pct",
        "ret_to_last_low_pct", "current_stage_ret_pct",
        "price_vs_dsa_vwap_pct", "bbmacd",
        "risk_flag",
    ]
    avail = [c for c in display_cols if c in filtered.columns]

    print(f"\n  {score_label} Top {min(top_n, len(filtered))}:")
    print(f"  {'排名':>4}  {'代码':>6}  {'名称':>8}  {'收盘':>8}  {'评分':>8}  {'P(盈利)':>8}  {'P(低回撤)':>8}  {'VWAP偏离':>8}  {'距高点':>8}  {'距低点%':>8}  {'风控':>20}")
    print("  " + "-" * 110)

    top = filtered.head(top_n)
    for i, (_, row) in enumerate(top.iterrows()):
        code = row["ts_code"]
        name = str(row.get("stock_name", ""))[:8]
        close = row["trigger_close"]
        score = row.get(score_col, np.nan)
        p_profit = row.get("p_profit", np.nan)
        p_low_dd = row.get("p_low_drawdown", np.nan)
        vwap = row.get("price_vs_dsa_vwap_pct", np.nan)
        rth = row.get("ret_to_last_high_pct", np.nan)
        rtl = row.get("ret_to_last_low_pct", np.nan)
        flag = str(row.get("risk_flag", "")).strip()

        s_score = f"{score:.4f}" if not np.isnan(score) else "N/A"
        s_pp = f"{p_profit:.3f}" if not np.isnan(p_profit) else "N/A"
        s_pld = f"{p_low_dd:.3f}" if not np.isnan(p_low_dd) else "N/A"
        s_vwap = f"{vwap:+.4f}" if not np.isnan(vwap) else "N/A"
        s_rth = f"{rth:+.2%}" if not np.isnan(rth) else "N/A"
        s_rtl = f"{rtl:+.2%}" if not np.isnan(rtl) else "N/A"

        print(f"  {i+1:>4}  {code:>6}  {name:>8}  {close:>8.2f}  {s_score:>8}  {s_pp:>8}  {s_pld:>8}  {s_vwap:>8}  {s_rth:>8}  {s_rtl:>8}  {flag:>20}")

    # 统计
    print(f"\n  --- 统计 ---")
    print(f"  硬过滤后: {len(filtered)} 只")
    print(f"  有风控标记: {(filtered['risk_flag'] != '').sum()}/{len(filtered)}")
    if score_col in filtered.columns and filtered[score_col].notna().any():
        print(f"  {score_label}: mean={filtered[score_col].mean():.4f}, median={filtered[score_col].median():.4f}")

    # 保存
    if output:
        out_cols = avail + [score_col]
        out_cols = [c for c in out_cols if c in filtered.columns and c not in avail or c == score_col]
        final_cols = avail + [c for c in [score_col] if c not in avail]
        filtered[final_cols].to_csv(output, index=False, encoding="utf-8-sig")
        print(f"\n  已保存: {output} ({len(filtered)} 只)")

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="DSA V型反转完整选股方案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/dsa_vreversal_selector.py --date 2026-04-30
  python selection/dsa_vreversal_selector.py --date 2026-04-30 --top-n 20
  python selection/dsa_vreversal_selector.py --date 2026-04-30 --output /root/trading/dsa_selection.csv
        """,
    )
    parser.add_argument("--date", type=str, required=True, help="选股日期 (YYYY-MM-DD)")
    parser.add_argument("--model-dir", type=str, default="/tmp/dsa_vreversal_gbdt_full", help="GBDT模型目录")
    parser.add_argument("--top-n", type=int, default=30, help="输出Top N，默认30")
    parser.add_argument("--output", type=str, default="", help="输出CSV路径")

    args = parser.parse_args()
    selection_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    run_selection(selection_date, args.model_dir, args.top_n, args.output)


if __name__ == "__main__":
    main()
