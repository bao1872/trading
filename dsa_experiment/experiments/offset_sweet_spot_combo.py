#!/usr/bin/env python3
"""
Offset 甜点区间组合特征探索

Purpose: 在 offset_mean 甜点区间 [0.11, 0.15] 内，探索 offset 组合特征对未来涨幅的预测力
Inputs: dsa_selection 表, stock_k_data 表
Outputs: dsa_experiment/output/sweet_spot_combo_*.csv, dsa_experiment/output/sweet_spot_combo_*.png

How to Run:
    python dsa_experiment/experiments/offset_sweet_spot_combo.py
    python dsa_experiment/experiments/offset_sweet_spot_combo.py --start 2024-01-01
    python dsa_experiment/experiments/offset_sweet_spot_combo.py --start 2025-01-01 --sample-limit 5000

Examples:
    python dsa_experiment/experiments/offset_sweet_spot_combo.py --start 2025-01-01
    python dsa_experiment/experiments/offset_sweet_spot_combo.py

Side Effects: 只读操作，输出 CSV 和 PNG 文件
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

HORIZONS = [5, 10, 20]
SWEET_SPOT_LOW = 0.11
SWEET_SPOT_HIGH = 0.15

# 原始 offset 特征
RAW_OFFSET_FEATURES = ["offset_rate", "offset_mean", "offset_std", "offset_percentile"]

# 组合特征名
COMBO_FEATURES = [
    "om_x_ostd",       # offset_mean * offset_std
    "om_div_ostd",     # offset_mean / offset_std (类 Sharpe)
    "or_minus_om",     # offset_rate - offset_mean (加速/回归信号)
    "om_x_bars",       # offset_mean * dsa_dir_bars
    "om_x_vret",       # offset_mean * vwap_ret_total
    "ostd_x_bars",     # offset_std * dsa_dir_bars
]


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────

def load_data(start_date: str = None, sample_limit: int = None) -> pd.DataFrame:
    """从 dsa_selection 表加载选股结果。"""
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
    """为选股结果计算未来收益标签。"""
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
        sql = (
            f"SELECT bar_time, open, high, low, close "
            f"FROM stock_k_data WHERE ts_code = '{ts_code}' AND freq = 'd' ORDER BY bar_time"
        )
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
    return sel_df


# ──────────────────────────────────────────────
# 组合特征构造
# ──────────────────────────────────────────────

def build_combo_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造 offset 组合特征。"""
    df = df.copy()

    # offset_mean * offset_std
    df["om_x_ostd"] = df["offset_mean"] * df["offset_std"]

    # offset_mean / offset_std (类 Sharpe，避免除零)
    ostd_safe = df["offset_std"].replace(0, np.nan)
    df["om_div_ostd"] = df["offset_mean"] / ostd_safe

    # offset_rate - offset_mean (加速/回归信号)
    df["or_minus_om"] = df["offset_rate"] - df["offset_mean"]

    # offset_mean * dsa_dir_bars
    df["om_x_bars"] = df["offset_mean"] * df["dsa_dir_bars"]

    # offset_mean * vwap_ret_total
    df["om_x_vret"] = df["offset_mean"] * df["vwap_ret_total"]

    # offset_std * dsa_dir_bars
    df["ostd_x_bars"] = df["offset_std"] * df["dsa_dir_bars"]

    return df


# ──────────────────────────────────────────────
# 单因子 IC 分析
# ──────────────────────────────────────────────

def compute_single_factor_ic(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """计算单因子日频 IC / ICIR。"""
    label_cols = [f"ret_{n}" for n in HORIZONS]
    results = []

    for factor in feature_cols:
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


# ──────────────────────────────────────────────
# 双因子分组热力图
# ──────────────────────────────────────────────

def compute_dual_factor_heatmap(df: pd.DataFrame, factor1: str, factor2: str,
                                 label: str = "ret_5", n_bins: int = 5) -> pd.DataFrame:
    """对两个因子做 n_bins × n_bins 分组，计算每组平均收益。"""
    valid = df[[factor1, factor2, label]].dropna()
    if len(valid) < n_bins * n_bins * 5:
        return pd.DataFrame()

    valid = valid.copy()
    try:
        valid["g1"] = pd.qcut(valid[factor1], n_bins, labels=False, duplicates="drop")
        valid["g2"] = pd.qcut(valid[factor2], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    pivot = valid.groupby(["g1", "g2"], observed=True)[label].mean().unstack()
    return pivot


def plot_dual_factor_heatmaps(df: pd.DataFrame, best_features: list, output_dir: str):
    """绘制双因子热力图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_features = len(best_features)
    if n_features == 0:
        return

    fig, axes = plt.subplots(1, n_features, figsize=(7 * n_features, 6))
    if n_features == 1:
        axes = [axes]

    for idx, feat in enumerate(best_features):
        ax = axes[idx]
        pivot = compute_dual_factor_heatmap(df, "offset_mean", feat, label="ret_5")
        if pivot.empty:
            ax.text(0.5, 0.5, f"No data for {feat}", transform=ax.transAxes, ha="center")
            continue

        im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"Q{int(c)+1}" for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"Q{int(i)+1}" for i in pivot.index], fontsize=9)
        ax.set_xlabel(f"{feat} Quintile")
        ax.set_ylabel("offset_mean Quintile")
        ax.set_title(f"offset_mean x {feat} -> ret_5 (%)")
        plt.colorbar(im, ax=ax, label="ret_5 (%)")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val*100:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(val * 100) > 0.5 else "black")

    plt.tight_layout()
    path = os.path.join(output_dir, "sweet_spot_combo_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  双因子热力图已保存: {path}")


# ──────────────────────────────────────────────
# LightGBM 交叉验证
# ──────────────────────────────────────────────

def compute_lgb_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """在甜点区间内对比：仅原始特征 vs 原始+组合特征。"""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  lightgbm 未安装，跳过 LightGBM 对比")
        return pd.DataFrame()

    label_cols = [f"ret_{n}" for n in HORIZONS]
    results = []

    dates_sorted = sorted(df["selection_date"].unique())
    if len(dates_sorted) < 30:
        print("  日期数不足30天，跳过 LightGBM 对比")
        return pd.DataFrame()

    n_train = int(len(dates_sorted) * 0.7)
    train_dates = set(dates_sorted[:n_train])
    test_dates = set(dates_sorted[n_train:])

    train_df = df[df["selection_date"].isin(train_dates)]
    test_df = df[df["selection_date"].isin(test_dates)]

    raw_features = [c for c in RAW_OFFSET_FEATURES if c in df.columns]
    all_features = raw_features + [c for c in COMBO_FEATURES if c in df.columns]

    for label in label_cols:
        for feat_set_name, feat_set in [("raw_only", raw_features), ("raw+combo", all_features)]:
            train_valid = train_df[feat_set + [label]].dropna()
            test_valid = test_df[feat_set + [label]].dropna()

            if len(train_valid) < 50 or len(test_valid) < 20:
                results.append({
                    "label": label,
                    "feature_set": feat_set_name,
                    "train_n": len(train_valid),
                    "test_n": len(test_valid),
                    "ic": np.nan,
                    "icir": np.nan,
                })
                continue

            X_train = train_valid[feat_set]
            y_train = train_valid[label]
            X_test = test_valid[feat_set]
            y_test = test_valid[label]

            model = lgb.LGBMRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                num_leaves=31, min_child_samples=30, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                verbose=-1, random_state=42,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            ic_val, _ = stats.spearmanr(y_pred, y_test)

            # 日频 IC
            daily_ics = []
            test_with_pred = test_df[[label, "selection_date"]].copy()
            test_with_pred["pred"] = np.nan
            test_with_pred.loc[test_valid.index, "pred"] = y_pred
            test_with_pred = test_with_pred.dropna(subset=["pred"])
            for dt, day_df in test_with_pred.groupby("selection_date"):
                if len(day_df) < 3:
                    continue
                day_ic, _ = stats.spearmanr(day_df["pred"], day_df[label])
                if np.isfinite(day_ic):
                    daily_ics.append(day_ic)

            mean_ic = np.mean(daily_ics) if daily_ics else np.nan
            std_ic = np.std(daily_ics) if daily_ics else np.nan
            icir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan

            results.append({
                "label": label,
                "feature_set": feat_set_name,
                "train_n": len(train_valid),
                "test_n": len(test_valid),
                "ic": round(ic_val, 4) if np.isfinite(ic_val) else np.nan,
                "daily_ic": round(mean_ic, 4) if np.isfinite(mean_ic) else np.nan,
                "icir": round(icir, 4) if np.isfinite(icir) else np.nan,
            })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Offset 甜点区间组合特征探索")
    parser.add_argument("--start", type=str, default=None, help="起始日期")
    parser.add_argument("--sample-limit", type=int, default=None, help="采样限制")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("Offset 甜点区间组合特征探索")
    print(f"甜点区间: offset_mean ∈ [{SWEET_SPOT_LOW}, {SWEET_SPOT_HIGH}]")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/6] 加载数据 ...")
    sel_df = load_data(start_date=args.start, sample_limit=args.sample_limit)
    if sel_df.empty:
        print("  无数据，退出")
        return

    # 2. 计算未来收益
    print("\n[2/6] 计算未来收益 ...")
    df = compute_future_returns(sel_df)

    # 3. 构造组合特征
    print("\n[3/6] 构造组合特征 ...")
    df = build_combo_features(df)

    # 甜点区间 vs 全量
    sweet_mask = (df["offset_mean"] >= SWEET_SPOT_LOW) & (df["offset_mean"] <= SWEET_SPOT_HIGH)
    df_sweet = df[sweet_mask].copy()
    print(f"  全量样本: {len(df)}")
    print(f"  甜点区间样本: {len(df_sweet)} ({len(df_sweet)/len(df)*100:.1f}%)")

    # 4. 单因子 IC 分析
    print("\n[4/6] 单因子 IC 分析 ...")
    all_features = RAW_OFFSET_FEATURES + COMBO_FEATURES

    ic_sweet = compute_single_factor_ic(df_sweet, all_features)
    ic_full = compute_single_factor_ic(df, all_features)

    # 合并对比
    ic_compare = ic_sweet.merge(ic_full, on="factor", suffixes=("_sweet", "_full"))
    ic_path = os.path.join(OUTPUT_DIR, "sweet_spot_combo_ic.csv")
    ic_compare.to_csv(ic_path, index=False)
    print(f"  IC 对比已保存: {ic_path}")

    print(f"\n  {'因子':<20} {'IC_5d_甜点':>10} {'IC_5d_全量':>10} {'ICIR_5d_甜点':>12} {'ICIR_5d_全量':>12} {'IC_10d_甜点':>10} {'IC_10d_全量':>10}")
    print(f"  {'-'*90}")
    for _, row in ic_compare.iterrows():
        ic5s = row.get("ic_ret_5_sweet", np.nan)
        ic5f = row.get("ic_ret_5_full", np.nan)
        icir5s = row.get("icir_ret_5_sweet", np.nan)
        icir5f = row.get("icir_ret_5_full", np.nan)
        ic10s = row.get("ic_ret_10_sweet", np.nan)
        ic10f = row.get("ic_ret_10_full", np.nan)
        print(f"  {row['factor']:<20} {ic5s:>10.4f} {ic5f:>10.4f} {icir5s:>12.4f} {icir5f:>12.4f} {ic10s:>10.4f} {ic10f:>10.4f}")

    # 5. 双因子分组热力图
    print("\n[5/6] 双因子分组热力图 ...")
    # 选 IC 绝对值最高的 3 个组合特征
    combo_ic = ic_sweet[ic_sweet["factor"].isin(COMBO_FEATURES)].copy()
    combo_ic["abs_ic"] = combo_ic["ic_ret_5"].abs()
    combo_ic = combo_ic.sort_values("abs_ic", ascending=False)
    best_combo = combo_ic["factor"].head(3).tolist()
    print(f"  IC 最高的组合特征: {best_combo}")

    if best_combo:
        plot_dual_factor_heatmaps(df_sweet, best_combo, OUTPUT_DIR)

        # 输出双因子分组统计
        for feat in best_combo:
            pivot = compute_dual_factor_heatmap(df_sweet, "offset_mean", feat, label="ret_5")
            if not pivot.empty:
                print(f"\n  offset_mean x {feat} -> ret_5 (%):")
                print(pivot.apply(lambda x: round(x * 100, 2)).to_string())

    # 6. LightGBM 对比
    print("\n[6/6] LightGBM 对比 ...")
    lgb_df = compute_lgb_comparison(df_sweet)
    if not lgb_df.empty:
        lgb_path = os.path.join(OUTPUT_DIR, "sweet_spot_combo_lgb.csv")
        lgb_df.to_csv(lgb_path, index=False)
        print(f"  LightGBM 对比已保存: {lgb_path}")

        print(f"\n  {'标签':<8} {'特征集':<12} {'IC':>8} {'日频IC':>8} {'ICIR':>8} {'train_n':>8} {'test_n':>8}")
        print(f"  {'-'*70}")
        for _, row in lgb_df.iterrows():
            ic = row.get("ic", np.nan)
            daily_ic = row.get("daily_ic", np.nan)
            icir = row.get("icir", np.nan)
            ic_str = f"{ic:>8.4f}" if pd.notna(ic) else f"{'N/A':>8}"
            dic_str = f"{daily_ic:>8.4f}" if pd.notna(daily_ic) else f"{'N/A':>8}"
            icir_str = f"{icir:>8.4f}" if pd.notna(icir) else f"{'N/A':>8}"
            print(f"  {row['label']:<8} {row['feature_set']:<12} {ic_str} "
                  f"{dic_str} {icir_str} {row['train_n']:>8} {row['test_n']:>8}")

    # 规则过滤统计
    print("\n[7/6] 规则过滤统计: 甜点 + change_pct>1 + offset_percentile<0.7 ...")
    rule_mask = (
        sweet_mask
        & (df["change_pct"] > 1)
        & (df["offset_percentile"] < 0.7)
    )
    df_rule = df[rule_mask].copy()
    print(f"  规则过滤后样本: {len(df_rule)} (甜点区间 {len(df_sweet)} 的 {len(df_rule)/max(len(df_sweet),1)*100:.1f}%)")

    # 三组对比统计
    groups = {
        "全量样本": df,
        "甜点区间": df_sweet,
        "甜点+规则": df_rule,
    }
    compare_rows = []
    for group_name, group_df in groups.items():
        row = {"group": group_name, "n": len(group_df)}
        for n in HORIZONS:
            ret_col = f"ret_{n}"
            mae_col = f"mae_{n}"
            mfe_col = f"mfe_{n}"
            rets = group_df[ret_col].dropna() if ret_col in group_df.columns else pd.Series(dtype=float)
            maes = group_df[mae_col].dropna() if mae_col in group_df.columns else pd.Series(dtype=float)
            mfes = group_df[mfe_col].dropna() if mfe_col in group_df.columns else pd.Series(dtype=float)
            if len(rets) > 0:
                row[f"ret_{n}_mean"] = rets.mean()
                row[f"ret_{n}_median"] = rets.median()
                row[f"ret_{n}_winrate"] = (rets > 0).mean() * 100
            else:
                row[f"ret_{n}_mean"] = np.nan
                row[f"ret_{n}_median"] = np.nan
                row[f"ret_{n}_winrate"] = np.nan
            if len(maes) > 0:
                row[f"mae_{n}_mean"] = maes.mean()
            else:
                row[f"mae_{n}_mean"] = np.nan
            if len(mfes) > 0:
                row[f"mfe_{n}_mean"] = mfes.mean()
            else:
                row[f"mfe_{n}_mean"] = np.nan
        compare_rows.append(row)

    compare_df = pd.DataFrame(compare_rows)
    rule_path = os.path.join(OUTPUT_DIR, "sweet_spot_rule_filter.csv")
    compare_df.to_csv(rule_path, index=False)
    print(f"  规则过滤统计已保存: {rule_path}")

    print(f"\n  {'组':<12} {'N':>7} ", end="")
    for n in HORIZONS:
        print(f" {'ret'+str(n)+'%':>7} {'胜率%':>6} {'mae%':>6} {'mfe%':>6}", end="")
    print()
    print(f"  {'-'*90}")
    for _, row in compare_df.iterrows():
        print(f"  {row['group']:<12} {row['n']:>7} ", end="")
        for n in HORIZONS:
            rm = row.get(f"ret_{n}_mean", np.nan)
            wr = row.get(f"ret_{n}_winrate", np.nan)
            mm = row.get(f"mae_{n}_mean", np.nan)
            fm = row.get(f"mfe_{n}_mean", np.nan)
            print(f" {rm*100:>7.2f} {wr:>6.1f} {mm*100:>6.2f} {fm*100:>6.2f}", end="")
        print()

    # 总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)

    if not combo_ic.empty:
        print(f"\n  甜点区间内组合特征 IC 排名 (ret_5):")
        for _, row in combo_ic.head(5).iterrows():
            ic5 = row.get("ic_ret_5", np.nan)
            icir5 = row.get("icir_ret_5", np.nan)
            print(f"    {row['factor']:<20} IC={ic5:.4f}, ICIR={icir5:.4f}")

    if not lgb_df.empty:
        for label in [f"ret_{n}" for n in HORIZONS]:
            sub = lgb_df[lgb_df["label"] == label]
            if len(sub) == 2:
                raw_icir = sub[sub["feature_set"] == "raw_only"]["icir"].values[0]
                combo_icir = sub[sub["feature_set"] == "raw+combo"]["icir"].values[0]
                delta = combo_icir - raw_icir if pd.notna(raw_icir) and pd.notna(combo_icir) else np.nan
                print(f"\n  {label}: raw ICIR={raw_icir:.4f}, raw+combo ICIR={combo_icir:.4f}, delta={delta:+.4f}")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
