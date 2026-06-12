#!/usr/bin/env python3
"""
DSA 选股结果探索：特征与未来收益关系分析

Purpose: 探索 dsa_selection 中特征与未来涨幅的关系，评估特征预测能力
Inputs: dsa_selection 表, stock_k_data 表
Outputs: dsa_experiment/output/dsa_selection_explorer_*.csv

How to Run:
    python dsa_experiment/experiments/dsa_selection_explorer.py
    python dsa_experiment/experiments/dsa_selection_explorer.py --start 2024-01-01
    python dsa_experiment/experiments/dsa_selection_explorer.py --start 2025-01-01 --sample-limit 5000

Examples:
    python dsa_experiment/experiments/dsa_selection_explorer.py --start 2025-01-01
    python dsa_experiment/experiments/dsa_selection_explorer.py

Side Effects: 只读操作，输出 CSV 文件
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

HORIZONS = [5, 10, 20, 40, 60, 80]

FEATURE_COLS = [
    "regime_strength", "dsa_dir_bars", "offset_rate", "offset_mean",
    "offset_std", "offset_percentile", "vwap_ret_total",
    "vwap_ret_5", "vwap_ret_10", "vwap_ret_20",
    "cross_up_count", "cross_down_count",
    "change_pct", "vol_zscore", "avg_amount_20d", "dsa_vwap_dev_pct",
    "rope_dir1_pct", "rope_dir0_pct", "rope_dir_neg1_pct",
]

BOOL_FEATURE_COLS = ["touch_rope", "touch_vwap"]

ALL_FEATURE_COLS = FEATURE_COLS + BOOL_FEATURE_COLS


def load_data(start_date: str = None, sample_limit: int = None) -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        sql = "SELECT * FROM dsa_selection"
        conditions = []
        if start_date:
            conditions.append(f"selection_date >= '{start_date}'")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY selection_date, ts_code"
        df = pd.read_sql(sql, conn)
    print(f"  dsa_selection 记录数: {len(df)}")
    if sample_limit and len(df) > sample_limit:
        df = df.sample(sample_limit, random_state=42).sort_values(["selection_date", "ts_code"])
        print(f"  采样后记录数: {len(df)}")
    return df


def compute_future_returns(sel_df: pd.DataFrame) -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)
    sel_df = sel_df.copy()
    sel_df["selection_date"] = pd.to_datetime(sel_df["selection_date"]).dt.normalize()

    label_cols = []
    for n in HORIZONS:
        label_cols.extend([f"ret_{n}", f"mae_{n}", f"mfe_{n}"])
    for col in label_cols:
        sel_df[col] = np.nan

    ts_codes = sel_df["ts_code"].unique().tolist()
    print(f"  需处理 {len(ts_codes)} 只股票 ...")

    processed = 0
    for ts_code in ts_codes:
        sql = f"SELECT bar_time, open, high, low, close FROM stock_k_data WHERE ts_code = '{ts_code}' AND freq = 'd' ORDER BY bar_time"
        with engine.connect() as conn:
            kline = pd.read_sql(sql, conn)
        if len(kline) < 20:
            continue

        kline["bar_time"] = pd.to_datetime(kline["bar_time"]).dt.normalize()
        kline = kline.set_index("bar_time")

        mask = sel_df["ts_code"] == ts_code
        stock_rows = sel_df.loc[mask]
        if stock_rows.empty:
            continue

        close_s = kline["close"]
        high_s = kline["high"]
        low_s = kline["low"]

        for n in HORIZONS:
            ret_n = close_s.shift(-n) / close_s - 1
            mae_n = low_s.shift(-1).rolling(n, min_periods=1).min().shift(-(n - 1)) / close_s - 1
            mfe_n = high_s.shift(-1).rolling(n, min_periods=1).max().shift(-(n - 1)) / close_s - 1

            dates = stock_rows["selection_date"]
            valid_dates = dates[dates.isin(ret_n.index)]
            if valid_dates.empty:
                continue
            sel_df.loc[valid_dates.index, f"ret_{n}"] = ret_n.loc[valid_dates].values
            sel_df.loc[valid_dates.index, f"mae_{n}"] = mae_n.loc[valid_dates].values
            sel_df.loc[valid_dates.index, f"mfe_{n}"] = mfe_n.loc[valid_dates].values

        processed += 1
        if processed % 500 == 0:
            print(f"    已处理 {processed}/{len(ts_codes)} 只股票")

    print(f"  已处理 {processed}/{len(ts_codes)} 只股票")

    valid_count = sel_df[label_cols].notna().sum()
    print(f"  未来收益标签非空统计:")
    for col in label_cols:
        print(f"    {col}: {valid_count[col]}/{len(sel_df)}")

    return sel_df


def compute_single_factor_ic(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = [f"ret_{n}" for n in HORIZONS]
    results = []
    for factor in ALL_FEATURE_COLS:
        if factor not in df.columns:
            continue
        row = {"factor": factor}
        for label in label_cols:
            valid = df[[factor, label, "selection_date"]].dropna()
            if len(valid) < 30:
                row[f"ic_{label}"] = np.nan
                row[f"icir_{label}"] = np.nan
                row[f"ic_pos_pct_{label}"] = np.nan
                row[f"n_days_{label}"] = 0
                continue

            daily_ics = []
            for _, day_df in valid.groupby("selection_date"):
                if len(day_df) < 5:
                    continue
                ic_val, _ = stats.spearmanr(day_df[factor], day_df[label])
                if np.isfinite(ic_val):
                    daily_ics.append(ic_val)

            if not daily_ics:
                row[f"ic_{label}"] = np.nan
                row[f"icir_{label}"] = np.nan
                row[f"ic_pos_pct_{label}"] = np.nan
                row[f"n_days_{label}"] = 0
                continue

            mean_ic = np.mean(daily_ics)
            std_ic = np.std(daily_ics)
            icir = mean_ic / std_ic if std_ic > 0 else 0
            ic_pos_pct = np.mean([1 for x in daily_ics if x > 0]) * 100

            row[f"ic_{label}"] = round(mean_ic, 4)
            row[f"icir_{label}"] = round(icir, 4)
            row[f"ic_pos_pct_{label}"] = round(ic_pos_pct, 1)
            row[f"n_days_{label}"] = len(daily_ics)

        results.append(row)

    return pd.DataFrame(results)


def compute_quintile_backtest(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = [f"ret_{n}" for n in HORIZONS]
    results = []

    for factor in ALL_FEATURE_COLS:
        if factor not in df.columns:
            continue
        for label in label_cols:
            valid = df[[factor, label, "selection_date"]].dropna()
            if len(valid) < 50:
                continue

            def qcut_daily(g):
                if len(g) < 5:
                    return g
                g = g.copy()
                try:
                    g["quintile"] = pd.qcut(g[factor], 5, labels=False, duplicates="drop") + 1
                except ValueError:
                    return g
                return g

            valid = valid.groupby("selection_date", group_keys=False).apply(qcut_daily)
            if "quintile" not in valid.columns:
                continue

            q_means = valid.groupby("quintile")[label].mean()
            if len(q_means) < 2:
                continue

            q1 = q_means.get(1, np.nan)
            q5 = q_means.get(5, np.nan)
            spread = q5 - q1 if pd.notna(q5) and pd.notna(q1) else np.nan

            means_list = [q_means.get(i, np.nan) for i in range(1, 6)]
            valid_means = [m for m in means_list if pd.notna(m)]
            if len(valid_means) >= 3:
                diffs = np.diff(valid_means)
                monotonic_pct = np.mean([1 if d > 0 else 0 for d in diffs]) * 100
            else:
                monotonic_pct = np.nan

            results.append({
                "factor": factor,
                "label": label,
                "Q1": round(q1, 4) if pd.notna(q1) else np.nan,
                "Q2": round(q_means.get(2, np.nan), 4),
                "Q3": round(q_means.get(3, np.nan), 4),
                "Q4": round(q_means.get(4, np.nan), 4),
                "Q5": round(q5, 4) if pd.notna(q5) else np.nan,
                "spread": round(spread, 4) if pd.notna(spread) else np.nan,
                "monotonic_pct": round(monotonic_pct, 1) if pd.notna(monotonic_pct) else np.nan,
            })

    return pd.DataFrame(results)


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in ALL_FEATURE_COLS if c in df.columns]
    corr_data = df[available].dropna()
    if len(corr_data) < 30:
        return pd.DataFrame()
    corr_matrix = corr_data.corr(method="spearman")
    return corr_matrix


def compute_lgb_validation(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except ImportError:
        print("  lightgbm 未安装，跳过 LightGBM 验证")
        return pd.DataFrame()

    label_cols = [f"ret_{n}" for n in HORIZONS]
    available_features = [c for c in ALL_FEATURE_COLS if c in df.columns]
    results = []

    dates_sorted = sorted(df["selection_date"].unique())
    if len(dates_sorted) < 60:
        print("  日期数不足60天，跳过 LightGBM 验证")
        return pd.DataFrame()

    n_train = int(len(dates_sorted) * 0.7)
    train_dates = set(dates_sorted[:n_train])
    test_dates = set(dates_sorted[n_train:])

    train_df = df[df["selection_date"].isin(train_dates)]
    test_df = df[df["selection_date"].isin(test_dates)]

    for label in label_cols:
        train_valid = train_df[available_features + [label]].dropna()
        test_valid = test_df[available_features + [label]].dropna()

        if len(train_valid) < 100 or len(test_valid) < 30:
            results.append({
                "label": label,
                "train_n": len(train_valid),
                "test_n": len(test_valid),
                "ic": np.nan,
                "icir": np.nan,
                "mae": np.nan,
            })
            continue

        X_train = train_valid[available_features]
        y_train = train_valid[label]
        X_test = test_valid[available_features]
        y_test = test_valid[label]

        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        ic_val, _ = stats.spearmanr(y_pred, y_test)
        daily_ics = []
        for dt, day_df in test_df[[label]].join(pd.DataFrame({"pred": y_pred}, index=test_valid.index)).groupby(test_df.loc[test_valid.index, "selection_date"]):
            if len(day_df) < 5:
                continue
            day_ic, _ = stats.spearmanr(day_df["pred"], day_df[label])
            if np.isfinite(day_ic):
                daily_ics.append(day_ic)

        mean_ic = np.mean(daily_ics) if daily_ics else np.nan
        std_ic = np.std(daily_ics) if daily_ics else np.nan
        icir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan
        mae_val = np.mean(np.abs(y_pred - y_test))

        importance = dict(zip(available_features, model.feature_importances_))
        top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_str = "; ".join(f"{k}={v}" for k, v in top5)

        results.append({
            "label": label,
            "train_n": len(train_valid),
            "test_n": len(test_valid),
            "ic": round(ic_val, 4) if np.isfinite(ic_val) else np.nan,
            "daily_ic": round(mean_ic, 4) if np.isfinite(mean_ic) else np.nan,
            "icir": round(icir, 4) if np.isfinite(icir) else np.nan,
            "mae": round(mae_val, 4),
            "top5_features": top5_str,
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        all_train_valid = train_df[available_features + label_cols].dropna()
        if len(all_train_valid) >= 100:
            model_all = lgb.LGBMRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                num_leaves=31, min_child_samples=50, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                verbose=-1, random_state=42,
            )
            model_all.fit(all_train_valid[available_features], all_train_valid["ret_5"])
            importance_all = dict(zip(available_features, model_all.feature_importances_))
            sorted_imp = sorted(importance_all.items(), key=lambda x: x[1], reverse=True)
            print("\n  LightGBM 特征重要性排序 (ret_5 模型):")
            for fname, fval in sorted_imp:
                print(f"    {fname:<25} {fval}")

    return result_df


def print_basic_stats(df: pd.DataFrame):
    label_cols = []
    for n in HORIZONS:
        label_cols.extend([f"ret_{n}", f"mae_{n}", f"mfe_{n}"])

    print(f"\n  样本量: {len(df)}")
    print(f"  日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")
    print(f"  股票数: {df['ts_code'].nunique()}")

    print(f"\n  未来收益分布:")
    print(f"  {'标签':<12} {'均值':>8} {'中位数':>8} {'25%':>8} {'75%':>8} {'胜率%':>8} {'非空数':>8}")
    print(f"  {'-'*72}")
    for col in label_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        win_rate = (s > 0).mean() * 100
        print(f"  {col:<12} {s.mean():>8.4f} {s.median():>8.4f} {s.quantile(0.25):>8.4f} {s.quantile(0.75):>8.4f} {win_rate:>7.1f}% {len(s):>8}")

    df["year"] = df["selection_date"].dt.year
    print(f"\n  按年度 ret_5 统计:")
    print(f"  {'年份':<8} {'样本数':>8} {'均值':>8} {'中位数':>8} {'胜率%':>8}")
    print(f"  {'-'*48}")
    for year, year_df in df.groupby("year"):
        s = year_df["ret_5"].dropna()
        if len(s) == 0:
            continue
        win_rate = (s > 0).mean() * 100
        print(f"  {year:<8} {len(s):>8} {s.mean():>8.4f} {s.median():>8.4f} {win_rate:>7.1f}%")


def main():
    parser = argparse.ArgumentParser(description="DSA 选股结果探索")
    parser.add_argument("--start", type=str, default=None, help="起始日期，如 2024-01-01")
    parser.add_argument("--sample-limit", type=int, default=None, help="采样限制（调试用）")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("DSA 选股结果探索：特征与未来收益关系分析")
    print("=" * 80)

    print("\n[1/6] 加载数据 ...")
    sel_df = load_data(start_date=args.start, sample_limit=args.sample_limit)
    if sel_df.empty:
        print("  无数据，退出")
        return

    print("\n[2/6] 计算未来收益 ...")
    df = compute_future_returns(sel_df)

    print("\n[3/6] 基础统计 ...")
    print_basic_stats(df)

    print("\n[4/6] 单因子 IC 分析 ...")
    ic_df = compute_single_factor_ic(df)

    label_cols_ret = [f"ret_{n}" for n in HORIZONS]
    print(f"\n  {'因子':<25}", end="")
    for label in label_cols_ret:
        print(f" {'IC_'+label[-2:]:>8}", end="")
    print()
    print(f"  {'-'*73}")
    for _, row in ic_df.iterrows():
        print(f"  {row['factor']:<25}", end="")
        for label in label_cols_ret:
            v = row.get(f"ic_{label}", np.nan)
            print(f" {v:>8.4f}" if pd.notna(v) else f" {'N/A':>8}", end="")
        print()

    ic_path = os.path.join(OUTPUT_DIR, "dsa_selection_explorer_ic.csv")
    ic_df.to_csv(ic_path, index=False)
    print(f"\n  IC 结果已保存: {ic_path}")

    print("\n[5/6] 单因子分组回测 ...")
    quintile_df = compute_quintile_backtest(df)
    if not quintile_df.empty:
        quintile_path = os.path.join(OUTPUT_DIR, "dsa_selection_explorer_quintile.csv")
        quintile_df.to_csv(quintile_path, index=False)
        print(f"  分组回测结果已保存: {quintile_path}")

        print(f"\n  显著因子 (|spread| > 1% 且 monotonic_pct > 60%):")
        sig = quintile_df[(quintile_df["spread"].abs() > 0.01) & (quintile_df["monotonic_pct"] > 60)]
        if sig.empty:
            print("    无")
        else:
            for _, row in sig.iterrows():
                print(f"    {row['factor']:<20} {row['label']:<8} spread={row['spread']:.4f} mono={row['monotonic_pct']:.0f}%")

    print("\n[6/6] 特征相关性 + LightGBM 验证 ...")
    corr_matrix = compute_correlation_matrix(df)
    if not corr_matrix.empty:
        corr_path = os.path.join(OUTPUT_DIR, "dsa_selection_explorer_corr.csv")
        corr_matrix.to_csv(corr_path)
        print(f"  相关性矩阵已保存: {corr_path}")

        high_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                v = corr_matrix.iloc[i, j]
                if pd.notna(v) and abs(v) > 0.7:
                    high_corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], round(v, 3)))
        if high_corr_pairs:
            print(f"\n  高相关特征对 (|corr| > 0.7):")
            for f1, f2, v in high_corr_pairs:
                print(f"    {f1} <-> {f2}: {v}")

    lgb_df = compute_lgb_validation(df)
    if not lgb_df.empty:
        lgb_path = os.path.join(OUTPUT_DIR, "dsa_selection_explorer_lgb.csv")
        lgb_df.to_csv(lgb_path, index=False)
        print(f"\n  LightGBM 结果已保存: {lgb_path}")

        print(f"\n  {'标签':<10} {'IC':>8} {'ICIR':>8} {'MAE':>8} {'train_n':>8} {'test_n':>8}")
        print(f"  {'-'*56}")
        for _, row in lgb_df.iterrows():
            print(f"  {row['label']:<10} {row['ic']:>8.4f} {row.get('icir', np.nan):>8.4f} {row['mae']:>8.4f} {row['train_n']:>8} {row['test_n']:>8}")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
