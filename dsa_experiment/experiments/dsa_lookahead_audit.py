#!/usr/bin/env python3
"""
DSA 选股特征前视偏差审计

Purpose: 验证 selection_dsa.py 中的特征是否存在前视偏差
Inputs: dsa_selection 表, stock_k_data 表
Outputs: 控制台报告

How to Run:
    python dsa_experiment/experiments/dsa_lookahead_audit.py
    python dsa_experiment/experiments/dsa_lookahead_audit.py --n-samples 50

Examples:
    python dsa_experiment/experiments/dsa_lookahead_audit.py
    python dsa_experiment/experiments/dsa_lookahead_audit.py --n-samples 20

Side Effects: 只读操作
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date

from selection.selection_dsa import process_stock

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

COMPARE_COLS = [
    "regime_strength", "dsa_dir_bars",
    "offset_rate", "offset_mean", "offset_std", "offset_percentile",
    "vwap_ret_total", "vwap_ret_5", "vwap_ret_10", "vwap_ret_20",
    "cross_up_count", "cross_down_count",
    "dsa_vwap", "dsa_vwap_dev_pct",
    "rope_dir1_pct", "rope_dir0_pct", "rope_dir_neg1_pct",
    "touch_rope", "touch_vwap",
]


def load_samples(n_samples: int = 40) -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM dsa_selection ORDER BY selection_date", conn)

    df["selection_date"] = pd.to_datetime(df["selection_date"]).dt.date

    # 采样：大涨 + 大跌 + 随机
    # 先计算未来收益来分层
    ts_codes = df["ts_code"].unique().tolist()
    kline_chunks = []
    for i in range(0, len(ts_codes), 500):
        batch = ts_codes[i:i + 500]
        codes_str = ",".join(f"'{c}'" for c in batch)
        sql = f"SELECT ts_code, bar_time, close FROM stock_k_data WHERE ts_code IN ({codes_str}) AND freq = 'd' ORDER BY ts_code, bar_time"
        with engine.connect() as conn:
            chunk = pd.read_sql(sql, conn)
        kline_chunks.append(chunk)
    kline_df = pd.concat(kline_chunks, ignore_index=True)
    kline_df["bar_time"] = pd.to_datetime(kline_df["bar_time"]).dt.date

    close_pivot = kline_df.pivot(index="bar_time", columns="ts_code", values="close").sort_index()

    # 计算 ret_20
    ret_20 = close_pivot.shift(-20) / close_pivot - 1
    ret_20_stacked = ret_20.stack().reset_index()
    ret_20_stacked.columns = ["selection_date", "ts_code", "ret_20"]

    df = df.merge(ret_20_stacked, on=["selection_date", "ts_code"], how="left")

    n_up = min(n_samples // 2, len(df[df["ret_20"] > 0.10].dropna()))
    n_down = min(n_samples // 4, len(df[df["ret_20"] < -0.10].dropna()))
    n_rand = n_samples - n_up - n_down

    up_samples = df[df["ret_20"] > 0.10].dropna(subset=["ret_20"]).sample(n_up, random_state=42)
    down_samples = df[df["ret_20"] < -0.10].dropna(subset=["ret_20"]).sample(n_down, random_state=42)
    rand_samples = df.dropna(subset=["ret_20"]).sample(n_rand, random_state=42)

    samples = pd.concat([up_samples, down_samples, rand_samples]).drop_duplicates(subset=["ts_code", "selection_date"])
    print(f"  采样: {len(up_samples)} 大涨 + {len(down_samples)} 大跌 + {len(rand_samples)} 随机 = {len(samples)} 总计")
    return samples


def recompute_no_lookahead(ts_code: str, selection_date: date) -> dict:
    result = process_stock(ts_code, selection_date)
    if result is None:
        return {}
    return result


def compare_features(samples: pd.DataFrame) -> pd.DataFrame:
    results = []
    for idx, row in samples.iterrows():
        ts_code = row["ts_code"]
        sel_date = row["selection_date"] if isinstance(row["selection_date"], date) else row["selection_date"]
        if isinstance(sel_date, pd.Timestamp):
            sel_date = sel_date.date()

        print(f"  对比 {ts_code} @ {sel_date} ...", end=" ")

        # 方式 A: 从 dsa_selection 表读取（全量计算）
        features_a = {col: row.get(col) for col in COMPARE_COLS}

        # 方式 B: 用截断数据重算（无前视）
        features_b = recompute_no_lookahead(ts_code, sel_date)
        if not features_b:
            print("无结果（不满足条件）")
            continue

        # 对比
        diffs = {}
        has_diff = False
        for col in COMPARE_COLS:
            val_a = features_a.get(col)
            val_b = features_b.get(col)

            if val_a is None and val_b is None:
                diffs[col] = {"A": None, "B": None, "diff": None, "pct_diff": None}
                continue
            if val_a is None or val_b is None:
                diffs[col] = {"A": val_a, "B": val_b, "diff": "一方为None", "pct_diff": "N/A"}
                has_diff = True
                continue

            # 布尔值特殊处理
            if isinstance(val_a, bool) or isinstance(val_b, bool):
                val_a_bool = bool(val_a)
                val_b_bool = bool(val_b)
                diffs[col] = {"A": val_a_bool, "B": val_b_bool, "diff": val_a_bool != val_b_bool, "pct_diff": "N/A"}
                if val_a_bool != val_b_bool:
                    has_diff = True
                continue

            diff = float(val_b) - float(val_a)
            if abs(float(val_a)) > 1e-10:
                pct_diff = abs(diff / float(val_a)) * 100
            else:
                pct_diff = 0.0 if abs(diff) < 1e-10 else float("inf")

            diffs[col] = {
                "A": round(float(val_a), 6),
                "B": round(float(val_b), 6),
                "diff": round(diff, 6),
                "pct_diff": round(pct_diff, 4),
            }
            if pct_diff > 1.0:
                has_diff = True

        status = "有差异!" if has_diff else "一致"
        print(status)

        result_row = {
            "ts_code": ts_code,
            "selection_date": str(sel_date),
            "ret_20": round(float(row.get("ret_20", np.nan)), 4) if pd.notna(row.get("ret_20")) else None,
            "has_diff": has_diff,
        }
        for col in COMPARE_COLS:
            d = diffs[col]
            result_row[f"{col}_A"] = d["A"]
            result_row[f"{col}_B"] = d["B"]
            result_row[f"{col}_diff"] = d["diff"]
            result_row[f"{col}_pct"] = d["pct_diff"]

        results.append(result_row)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="DSA 选股特征前视偏差审计")
    parser.add_argument("--n-samples", type=int, default=40, help="采样数量")
    args = parser.parse_args()

    print("=" * 80)
    print("DSA 选股特征前视偏差审计")
    print("=" * 80)

    print("\n[1/3] 采样 ...")
    samples = load_samples(n_samples=args.n_samples)

    print("\n[2/3] 对比特征 ...")
    result_df = compare_features(samples)

    if result_df.empty:
        print("  无有效对比结果")
        return

    print("\n[3/3] 差异报告 ...")

    n_total = len(result_df)
    n_diff = result_df["has_diff"].sum()
    print(f"\n  总样本: {n_total}, 存在差异: {n_diff} ({n_diff/n_total*100:.1f}%)")

    # 逐特征统计差异率
    print(f"\n  逐特征差异统计（差异 > 1% 的样本占比）:")
    print(f"  {'特征':<25} {'差异数':>6} {'差异率':>8} {'最大差异%':>10}")
    print(f"  {'-'*55}")

    significant_cols = []
    for col in COMPARE_COLS:
        pct_col = f"{col}_pct"
        if pct_col not in result_df.columns:
            continue
        pcts = result_df[pct_col]
        # 过滤 N/A 和 inf
        valid_pcts = pcts[pcts.apply(lambda x: isinstance(x, (int, float)) and np.isfinite(x))]
        n_valid = len(valid_pcts)
        if n_valid == 0:
            continue
        n_sig = (valid_pcts > 1.0).sum()
        max_pct = valid_pcts.max() if len(valid_pcts) > 0 else 0
        sig_rate = n_sig / n_valid * 100
        print(f"  {col:<25} {n_sig:>6} {sig_rate:>7.1f}% {max_pct:>9.2f}%")
        if sig_rate > 10:
            significant_cols.append(col)

    if significant_cols:
        print(f"\n  *** 确认存在前视偏差的特征（差异率 > 10%）***:")
        for col in significant_cols:
            print(f"    - {col}")
    else:
        print(f"\n  未发现显著前视偏差（所有特征差异率 <= 10%）")

    # 打印具体差异样本
    diff_samples = result_df[result_df["has_diff"] == True]
    if not diff_samples.empty:
        print(f"\n  存在差异的样本详情:")
        for _, row in diff_samples.iterrows():
            print(f"\n  {row['ts_code']} @ {row['selection_date']} (ret_20={row.get('ret_20', 'N/A')}):")
            for col in COMPARE_COLS:
                a_val = row.get(f"{col}_A")
                b_val = row.get(f"{col}_B")
                pct = row.get(f"{col}_pct")
                if a_val != b_val:
                    if isinstance(pct, (int, float)) and np.isfinite(pct) and pct > 0.01:
                        print(f"    {col}: A={a_val} B={b_val} diff%={pct}")

    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dsa_lookahead_audit.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\n  详细结果已保存: {output_path}")

    print("\n" + "=" * 80)
    print("审计完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
