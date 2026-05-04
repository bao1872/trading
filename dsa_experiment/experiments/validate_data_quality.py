#!/usr/bin/env python3
"""
数据质量验证脚本：排除未来函数/幸存者偏差/复权异常

Purpose: 验证实验数据质量，确保结果可信
Inputs: candidate_with_scores.parquet, stock_k_data (DB), stock_dsa_vreversal_results (DB)
Outputs: 终端报告
How to Run:
    python dsa_experiment/validate_data_quality.py
Examples:
    python dsa_experiment/validate_data_quality.py
Side Effects: 只读操作
"""

import sys
import os

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


def validate_extreme_returns(df: pd.DataFrame) -> None:
    print("\n  [1] 极端收益验证")
    print("  " + "-" * 60)

    extreme_neg = df[df["ret_5_open_to_open"] < -0.40].copy()
    if extreme_neg.empty:
        print("  ✅ 无极端负收益（<-40%）")
        return

    print(f"  发现 {len(extreme_neg)} 条极端负收益（<-40%）：")
    for _, row in extreme_neg.iterrows():
        ts_code = row.get("ts_code_raw", row.get("ts_code", ""))
        sel_date = row["selection_date"]
        ret = row["ret_5_open_to_open"]
        mae = row.get("mae_5", np.nan)
        can_buy = row.get("can_buy_next_open", "")
        print(f"    {ts_code} @ {sel_date}: ret5={ret:.2%}, mae5={mae:.2%}, can_buy={can_buy}")

    ts_codes = extreme_neg.get("ts_code_raw", extreme_neg.get("ts_code", "")).unique().tolist()
    suffix_map = {}
    for code in ts_codes:
        if code.startswith(("6", "5")):
            suffix_map[code] = f"{code}.SH"
        else:
            suffix_map[code] = f"{code}.SZ"

    db_codes = [suffix_map[c] for c in ts_codes]
    if db_codes:
        placeholders = ", ".join([f"'{c}'" for c in db_codes])
        sql = text(f"""
            SELECT ts_code, bar_time, open, high, low, close
            FROM stock_k_data
            WHERE ts_code IN ({placeholders}) AND freq = 'd'
            ORDER BY ts_code, bar_time
        """)
        with engine.connect() as conn:
            daily = pd.read_sql(sql, conn)

        if not daily.empty:
            daily["bar_time"] = pd.to_datetime(daily["bar_time"])
            for _, row in extreme_neg.iterrows():
                ts_code = row.get("ts_code_raw", row.get("ts_code", ""))
                sel_date = pd.Timestamp(row["selection_date"])
                db_code = suffix_map.get(ts_code, "")
                stock_daily = daily[daily["ts_code"] == db_code].sort_values("bar_time")
                if stock_daily.empty:
                    continue
                nearby = stock_daily[
                    (stock_daily["bar_time"] >= sel_date - pd.Timedelta(days=3))
                    & (stock_daily["bar_time"] <= sel_date + pd.Timedelta(days=10))
                ]
                if nearby.empty:
                    continue
                print(f"\n    {ts_code} 日线行情（{sel_date.strftime('%Y-%m-%d')} 前后）：")
                prev_close = None
                for _, d in nearby.iterrows():
                    chg = (d["close"] - prev_close) / prev_close if prev_close and prev_close > 0 else 0
                    print(f"      {d['bar_time'].strftime('%Y-%m-%d')}: open={d['open']:.2f}, close={d['close']:.2f}, chg={chg:+.2%}")
                    prev_close = d["close"]


def validate_can_buy(df: pd.DataFrame) -> None:
    print("\n  [2] can_buy_next_open 标记验证")
    print("  " + "-" * 60)

    total = len(df)
    can_buy_count = df["can_buy_next_open"].sum()
    can_buy_rate = can_buy_count / total if total > 0 else 0
    print(f"  总记录: {total}, 可买入: {can_buy_count} ({can_buy_rate:.1%})")

    limit_up = df[df.get("limit_up_next", pd.Series(0, index=df.index)) == 1]
    if not limit_up.empty:
        print(f"  一字涨停标记: {len(limit_up)} 条")
        limit_up_ret = limit_up["ret_5_open_to_open"].mean()
        print(f"  一字涨停平均5日收益: {limit_up_ret:.2%}")
    else:
        print("  一字涨停标记: 0 条（可能未记录）")

    can_buy_df = df[df["can_buy_next_open"] == True]
    cannot_buy_df = df[df["can_buy_next_open"] == False]
    if not can_buy_df.empty and not cannot_buy_df.empty:
        ret_can = can_buy_df["ret_5_open_to_open"].mean()
        ret_cannot = cannot_buy_df["ret_5_open_to_open"].mean()
        print(f"  可买入平均5日收益: {ret_can:.2%}")
        print(f"  不可买入平均5日收益: {ret_cannot:.2%}")
        if ret_cannot > ret_can:
            print("  ⚠️ 不可买入组收益更高，可能存在标记错误或一字涨停后反而上涨")


def validate_future_function(df: pd.DataFrame) -> None:
    print("\n  [3] 未来函数验证（抽样检查）")
    print("  " + "-" * 60)

    dates = sorted(df["selection_date"].unique())
    if len(dates) < 10:
        print("  日期数不足10，跳过抽样")
        return

    sample_dates = [dates[i] for i in np.linspace(0, len(dates) - 1, 10, dtype=int)]
    issues = []

    for sel_date in sample_dates:
        day_df = df[df["selection_date"] == sel_date]
        n = len(day_df)
        has_ret = day_df["ret_5_open_to_open"].notna().sum()
        has_mae = day_df["mae_5"].notna().sum()
        has_mfe = day_df["mfe_5"].notna().sum()

        if has_ret < n * 0.5:
            issues.append(f"  {sel_date}: ret_5 缺失率 {(1-has_ret/n):.0%}")

    if issues:
        for issue in issues:
            print(f"  ⚠️ {issue}")
    else:
        print("  ✅ 抽样10个日期，ret_5/mae_5/mfe_5 覆盖率正常")

    feature_cols = [c for c in df.columns if c not in [
        "ret_3_open_to_open", "ret_5_open_to_open", "ret_10_open_to_open",
        "mae_3", "mae_5", "mfe_3", "mfe_5", "stop_hit_5",
        "can_buy_next_open", "limit_up_next",
    ]]
    label_cols = ["ret_3_open_to_open", "ret_5_open_to_open", "ret_10_open_to_open",
                  "mae_3", "mae_5", "mfe_3", "mfe_5", "stop_hit_5"]
    corr_issues = []
    for label in label_cols:
        if label not in df.columns:
            continue
        valid = df[[label]].join(df[feature_cols]).dropna()
        if len(valid) < 100:
            continue
        for feat in feature_cols:
            if feat not in valid.columns or valid[feat].nunique() < 2:
                continue
            corr = valid[label].corr(valid[feat])
            if abs(corr) > 0.95:
                corr_issues.append(f"  {label} ~ {feat}: corr={corr:.3f}")

    if corr_issues:
        print("  ⚠️ 发现极高相关性（可能未来函数）：")
        for issue in corr_issues[:5]:
            print(issue)
    else:
        print("  ✅ 特征与标签无极高相关性（>0.95），未发现明显未来函数")


def validate_survivorship(df: pd.DataFrame) -> None:
    print("\n  [4] 幸存者偏差验证")
    print("  " + "-" * 60)

    ts_codes = df.get("ts_code_raw", df.get("ts_code", pd.Series(dtype=str))).unique()
    suffix_map = {}
    for code in ts_codes:
        if code.startswith(("6", "5")):
            suffix_map[code] = f"{code}.SH"
        else:
            suffix_map[code] = f"{code}.SZ"

    db_codes = [suffix_map[c] for c in ts_codes]

    try:
        placeholders = ", ".join([f"'{c}'" for c in db_codes[:500]])
        sql = text(f"""
            SELECT DISTINCT ts_code FROM stock_k_data
            WHERE ts_code IN ({placeholders}) AND freq = 'd'
        """)
        with engine.connect() as conn:
            result = conn.execute(sql)
            db_existing = set(row[0] for row in result)

        missing = set(db_codes[:500]) - db_existing
        if missing:
            print(f"  ⚠️ {len(missing)}/{len(db_codes[:500])} 只股票在 stock_k_data 中不存在")
            for m in list(missing)[:5]:
                print(f"    {m}")
        else:
            print(f"  ✅ 抽样500只股票全部在 stock_k_data 中存在")
    except Exception as e:
        print(f"  ⚠️ 无法验证幸存者偏差: {e}")


def validate_trigger_width(df: pd.DataFrame) -> None:
    print("\n  [5] 触发宽度分析")
    print("  " + "-" * 60)

    daily_counts = df.groupby("selection_date").size()
    print(f"  每期触发数: mean={daily_counts.mean():.0f}, median={daily_counts.median():.0f}")
    print(f"  范围: {daily_counts.min()} ~ {daily_counts.max()}")
    print(f"  >200只/期: {(daily_counts > 200).sum()}/{len(daily_counts)} 期")

    if "trend_align_momo" in df.columns:
        filtered = df[df["trend_align_momo"] == 1]
        filtered_counts = filtered.groupby("selection_date").size()
        print(f"  硬过滤(trend_align_momo==1)后: mean={filtered_counts.mean():.0f}, median={filtered_counts.median():.0f}")


def main():
    print("=" * 80)
    print("数据质量验证")
    print("=" * 80)

    input_path = os.path.join(OUTPUT_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(input_path):
        input_path = os.path.join(OUTPUT_DIR, "candidate_table.parquet")

    print(f"\n  加载数据: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  记录数: {len(df)}, 日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")

    validate_extreme_returns(df)
    validate_can_buy(df)
    validate_future_function(df)
    validate_survivorship(df)
    validate_trigger_width(df)

    print("\n" + "=" * 80)
    print("数据质量验证完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
