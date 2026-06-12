#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATR Rope 选股结果二次过滤脚本

Purpose: 基于 ATR 因子分析结论，对 atr_rope_selection 表的选股结果做二次过滤
Inputs:  atr_rope_selection 表 + 指定日期
Outputs: 终端打印过滤统计和候选股列表，可选输出 CSV

How to Run:
    # 基本用法（必须规则）
    python atr_experiment/atr_filter.py --date 2026-05-21

    # 启用推荐规则
    python atr_experiment/atr_filter.py --date 2026-05-21 --level recommended

    # 启用所有规则
    python atr_experiment/atr_filter.py --date 2026-05-21 --level strict

    # 自定义阈值
    python atr_experiment/atr_filter.py --date 2026-05-21 --dsa-vwap-min 2.0 --regime-strength-max 0.004

    # 输出到CSV
    python atr_experiment/atr_filter.py --date 2026-05-21 --level strict --output output/filtered_20260521.csv

Examples:
    python atr_experiment/atr_filter.py --date 2026-05-21
    python atr_experiment/atr_filter.py --date 2026-05-21 --level strict --output output/filtered.csv

Side Effects: 只读数据库，可选写 CSV 文件
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "atr_experiment" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 默认阈值 ====================

DEFAULT_PARAMS = {
    "dsa_vwap_dev_pct_min": 1.0,
    "regime_strength_max": 0.006,
    "rope_dir1_pct_min": 30.0,
    "rope_dir_neg1_pct_max": 25.0,
    "low_rope_dev_std_pct_min": 3.4,
    "avg_amount_20d_min": 1.0,
}

# ==================== 数据加载 ====================

def load_selection_data(selection_date: str) -> pd.DataFrame:
    """从 atr_rope_selection 表加载指定日期的选股结果"""
    from sqlalchemy import text
    from datasource.database import get_engine

    engine = get_engine()
    sql = text("""
        SELECT * FROM atr_rope_selection
        WHERE selection_date = :selection_date
        ORDER BY ts_code
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"selection_date": selection_date})
    engine.dispose()
    return df


def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """计算衍生百分比字段"""
    df = df.copy()

    # low_rope_dev_std_pct = low_rope_dev_std / rope_value * 100
    if "low_rope_dev_std" in df.columns and "rope_value" in df.columns:
        df["low_rope_dev_std_pct"] = np.where(
            df["rope_value"].notna() & (df["rope_value"] != 0),
            df["low_rope_dev_std"] / df["rope_value"] * 100,
            np.nan,
        )
    else:
        df["low_rope_dev_std_pct"] = np.nan

    return df


# ==================== 过滤逻辑 ====================

def apply_filters(df: pd.DataFrame, level: str, params: dict) -> tuple:
    """逐条应用过滤规则，记录每条规则的过滤效果

    Args:
        df: 原始选股数据
        level: 过滤级别 basic/recommended/strict
        params: 各规则的阈值参数

    Returns:
        (filtered_df, filter_stats)
    """
    stats = []
    total = len(df)

    # Rule 1: signal_type = '多头蓝色区间'（所有级别必须）
    before = len(df)
    df = df[df["signal_type"] == "多头蓝色区间"]
    stats.append(("signal_type = 多头蓝色区间", before, len(df)))

    # Rule 2: dsa_vwap_dev_pct > 阈值（所有级别必须）
    threshold = params["dsa_vwap_dev_pct_min"]
    before = len(df)
    df = df[df["dsa_vwap_dev_pct"] > threshold]
    stats.append((f"dsa_vwap_dev_pct > {threshold:.1f}%", before, len(df)))

    # Rule 3: regime_strength < 阈值（所有级别必须）
    threshold = params["regime_strength_max"]
    before = len(df)
    df = df[df["regime_strength"] < threshold]
    stats.append((f"regime_strength < {threshold:.4f}", before, len(df)))

    if level in ("recommended", "strict"):
        # Rule 4: rope_dir1_pct > 阈值
        threshold = params["rope_dir1_pct_min"]
        before = len(df)
        df = df[df["rope_dir1_pct"] > threshold]
        stats.append((f"rope_dir1_pct > {threshold:.0f}%", before, len(df)))

        # Rule 5: rope_dir_neg1_pct < 阈值
        threshold = params["rope_dir_neg1_pct_max"]
        before = len(df)
        df = df[df["rope_dir_neg1_pct"] < threshold]
        stats.append((f"rope_dir_neg1_pct < {threshold:.0f}%", before, len(df)))

    if level == "strict":
        # Rule 6: low_rope_dev_std_pct > 阈值
        threshold = params["low_rope_dev_std_pct_min"]
        before = len(df)
        if "low_rope_dev_std_pct" in df.columns:
            df = df[df["low_rope_dev_std_pct"] > threshold]
        stats.append((f"low_rope_dev_std_pct > {threshold:.1f}%", before, len(df)))

        # Rule 7: avg_amount_20d >= 阈值
        threshold = params["avg_amount_20d_min"]
        before = len(df)
        df = df[df["avg_amount_20d"] >= threshold]
        stats.append((f"avg_amount_20d >= {threshold:.1f}亿", before, len(df)))

    return df, stats, total


# ==================== 输出格式化 ====================

DISPLAY_COLUMNS = [
    "ts_code", "stock_name", "change_pct",
    "regime_strength", "rope_dir1_pct", "rope_dir_neg1_pct",
    "range_pos_01", "dsa_vwap_dev_pct", "dsa_dir_bars",
    "range_width_pct", "avg_amount_20d", "low_rope_signal",
    "low_rope_dev_std_pct",
]

DISPLAY_NAMES = {
    "ts_code": "股票代码", "stock_name": "股票名称", "change_pct": "涨跌幅%",
    "regime_strength": "趋势强度", "rope_dir1_pct": "Rope+1占比",
    "rope_dir_neg1_pct": "Rope-1占比", "range_pos_01": "箱体位置",
    "dsa_vwap_dev_pct": "VWAP偏离%", "dsa_dir_bars": "DSA持续bar",
    "range_width_pct": "带宽%", "avg_amount_20d": "20日均额(亿)",
    "low_rope_signal": "低Rope信号", "low_rope_dev_std_pct": "低Rope偏差std%",
}


def print_results(df: pd.DataFrame, stats: list, total: int, level: str):
    """打印过滤统计和候选股列表"""
    print(f"\n{'='*60}")
    print(f"ATR 选股过滤结果 (级别: {level})")
    print(f"{'='*60}")

    # 过滤统计
    print(f"\n过滤统计 (原始总数: {total}):")
    print(f"{'规则':<35s} {'过滤前':>8s} {'过滤后':>8s} {'通过率':>8s}")
    print("-" * 62)
    for rule_name, before, after in stats:
        rate = after / before * 100 if before > 0 else 0
        print(f"{rule_name:<35s} {before:>8d} {after:>8d} {rate:>7.1f}%")

    print(f"\n最终结果: {len(df)} 只 / {total} 只 ({len(df)/total*100:.1f}%)")

    if df.empty:
        print("\n无候选股通过过滤")
        return

    # 候选股列表
    available_cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    display_df = df[available_cols].copy()
    display_df.rename(columns=DISPLAY_NAMES, inplace=True)

    # 按 dsa_vwap_dev_pct 降序
    if "dsa_vwap_dev_pct" in df.columns:
        display_df["_sort"] = df["dsa_vwap_dev_pct"].values
        display_df.sort_values("_sort", ascending=False, inplace=True)
        display_df.drop(columns=["_sort"], inplace=True)

    print(f"\n候选股列表 (按 VWAP偏离% 降序):")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(display_df.to_string(index=False))


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="ATR Rope 选股结果二次过滤")
    parser.add_argument("--date", required=True, help="选股日期 (YYYY-MM-DD)")
    parser.add_argument("--level", choices=["basic", "recommended", "strict"],
                        default="basic", help="过滤级别 (默认: basic)")
    parser.add_argument("--dsa-vwap-min", type=float, default=None,
                        help=f"dsa_vwap_dev_pct 最小值 (默认: {DEFAULT_PARAMS['dsa_vwap_dev_pct_min']})")
    parser.add_argument("--regime-strength-max", type=float, default=None,
                        help=f"regime_strength 最大值 (默认: {DEFAULT_PARAMS['regime_strength_max']})")
    parser.add_argument("--rope-dir1-min", type=float, default=None,
                        help=f"rope_dir1_pct 最小值 (默认: {DEFAULT_PARAMS['rope_dir1_pct_min']})")
    parser.add_argument("--rope-dir-neg1-max", type=float, default=None,
                        help=f"rope_dir_neg1_pct 最大值 (默认: {DEFAULT_PARAMS['rope_dir_neg1_pct_max']})")
    parser.add_argument("--low-rope-std-min", type=float, default=None,
                        help=f"low_rope_dev_std_pct 最小值 (默认: {DEFAULT_PARAMS['low_rope_dev_std_pct_min']})")
    parser.add_argument("--avg-amount-min", type=float, default=None,
                        help=f"avg_amount_20d 最小值/亿 (默认: {DEFAULT_PARAMS['avg_amount_20d_min']})")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 文件路径 (相对于 atr_experiment/)")
    args = parser.parse_args()

    # 合并参数
    params = DEFAULT_PARAMS.copy()
    if args.dsa_vwap_min is not None:
        params["dsa_vwap_dev_pct_min"] = args.dsa_vwap_min
    if args.regime_strength_max is not None:
        params["regime_strength_max"] = args.regime_strength_max
    if args.rope_dir1_min is not None:
        params["rope_dir1_pct_min"] = args.rope_dir1_min
    if args.rope_dir_neg1_max is not None:
        params["rope_dir_neg1_pct_max"] = args.rope_dir_neg1_max
    if args.low_rope_std_min is not None:
        params["low_rope_dev_std_pct_min"] = args.low_rope_std_min
    if args.avg_amount_min is not None:
        params["avg_amount_20d_min"] = args.avg_amount_min

    # 加载数据
    print(f"加载 {args.date} 的选股数据...")
    df = load_selection_data(args.date)
    if df.empty:
        print(f"{args.date} 无选股数据")
        return

    print(f"原始数据: {len(df)} 条")
    if "signal_type" in df.columns:
        print(f"信号类型: {df['signal_type'].value_counts().to_dict()}")

    # 计算衍生字段
    df = compute_derived_fields(df)

    # 应用过滤
    filtered_df, stats, total = apply_filters(df, args.level, params)

    # 打印结果
    print_results(filtered_df, stats, total, args.level)

    # 输出 CSV
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = OUTPUT_DIR / output_path
        filtered_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\nCSV 已保存: {output_path}")


if __name__ == "__main__":
    main()
