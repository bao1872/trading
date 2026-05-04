#!/usr/bin/env python3
"""
检测并删除受除权除息影响的样本（向量化版本）

Purpose: 通过检测持仓期内隔夜跳空超过涨跌停限制，识别除权除息样本并删除
Inputs: candidate_with_scores.parquet, stock_k_data (DB)
Outputs: candidate_with_scores.parquet (覆盖写入，删除分红样本后)
How to Run:
    python dsa_experiment/remove_ex_dividend_samples.py
    python dsa_experiment/remove_ex_dividend_samples.py --dry-run
Examples:
    python dsa_experiment/remove_ex_dividend_samples.py
    python dsa_experiment/remove_ex_dividend_samples.py --dry-run
Side Effects: 覆盖写入 candidate_with_scores.parquet（--dry-run 时不写入）
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def main():
    parser = argparse.ArgumentParser(description="检测并删除受除权除息影响的样本")
    parser.add_argument("--dry-run", action="store_true", help="仅检测不删除")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 80)
    print("检测并删除受除权除息影响的样本（向量化版本）")
    print("=" * 80)

    input_path = os.path.join(args.output_dir, "candidate_with_scores.parquet")
    print(f"\n  加载: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  记录数: {len(df)}")

    print("\n  [1/3] 加载日线行情...")
    ts_code_col = "ts_code_raw" if "ts_code_raw" in df.columns else "ts_code"
    ts_codes_raw = df[ts_code_col].unique()
    suffix_map = {}
    for code in ts_codes_raw:
        if code.startswith(("6", "5")):
            suffix_map[code] = f"{code}.SH"
        else:
            suffix_map[code] = f"{code}.SZ"

    all_db_codes = list(set(suffix_map.values()))
    print(f"  股票数: {len(all_db_codes)}")

    all_daily = []
    batch_size = 500
    for i in range(0, len(all_db_codes), batch_size):
        batch = all_db_codes[i:i + batch_size]
        placeholders = ", ".join([f"'{c}'" for c in batch])
        sql = text(f"""
            SELECT ts_code, bar_time, open, close
            FROM stock_k_data
            WHERE ts_code IN ({placeholders}) AND freq = 'd'
        """)
        with engine.connect() as conn:
            batch_df = pd.read_sql(sql, conn)
        all_daily.append(batch_df)

    daily = pd.concat(all_daily, ignore_index=True)
    daily["bar_time"] = pd.to_datetime(daily["bar_time"])
    daily = daily.sort_values(["ts_code", "bar_time"]).reset_index(drop=True)
    daily["prev_close"] = daily.groupby("ts_code")["close"].shift(1)
    daily["overnight_ret"] = np.where(
        daily["prev_close"].notna() & (daily["prev_close"] > 0),
        (daily["open"] - daily["prev_close"]) / daily["prev_close"],
        np.nan,
    )
    daily["raw_code"] = daily["ts_code"].str.replace(r"\.(SH|SZ)", "", regex=True)
    print(f"  日线数据: {len(daily)} 条")

    print("\n  [2/3] 检测除权除息（向量化）...")
    ex_div_dates = daily[daily["overnight_ret"] < -0.12].copy()
    ex_div_dates["year"] = ex_div_dates["bar_time"].dt.year
    print(f"  检测到 {len(ex_div_dates)} 个除权除息日（隔夜跌幅<-12%）")

    ex_div_set = set(zip(ex_div_dates["raw_code"], ex_div_dates["bar_time"].dt.date))

    df["sel_date"] = pd.to_datetime(df["selection_date"]).dt.date
    df["raw_code"] = df[ts_code_col]

    affected = np.zeros(len(df), dtype=bool)
    for offset in range(1, 8):
        check_dates = pd.to_datetime(df["selection_date"]) + pd.Timedelta(days=offset)
        check_date_vals = check_dates.dt.date.values
        codes = df["raw_code"].values
        matches = np.array([
            (c, d) in ex_div_set for c, d in zip(codes, check_date_vals)
        ])
        affected |= matches

    n_affected = affected.sum()
    print(f"  受影响样本: {n_affected}/{len(df)} ({n_affected/len(df):.1%})")

    if n_affected > 0:
        aff_df = df[affected].copy()
        aff_df["year"] = pd.to_datetime(aff_df["selection_date"]).dt.year
        print("  分年统计:")
        for year, grp in aff_df.groupby("year"):
            print(f"    {year}: {len(grp)} 条")

        affected_ret = df[affected]["ret_5_open_to_open"].mean()
        unaffected_ret = df[~affected]["ret_5_open_to_open"].mean()
        print(f"  受影响样本平均ret5: {affected_ret:.2%}")
        print(f"  未受影响样本平均ret5: {unaffected_ret:.2%}")

    print("\n  [3/3] 处理...")
    if n_affected == 0:
        print("  ✅ 无需删除任何样本")
        return

    if args.dry_run:
        print(f"  [dry-run] 将删除 {n_affected} 条受除权除息影响的样本")
        return

    clean = df[~affected].copy()
    clean = clean.drop(columns=["sel_date", "raw_code"], errors="ignore")
    print(f"  删除 {n_affected} 条，剩余 {len(clean)} 条")

    backup_path = os.path.join(args.output_dir, "candidate_with_scores_backup.parquet")
    df.drop(columns=["sel_date", "raw_code"], errors="ignore").to_parquet(backup_path, index=False)
    print(f"  备份原文件: {backup_path}")

    clean.to_parquet(input_path, index=False)
    print(f"  已覆盖写入: {input_path}")

    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
