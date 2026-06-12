# -*- coding: utf-8 -*-
"""
特征-标签相关性分析实验

Purpose:
    不用GBDT，用纯统计方法分析哪些特征和未来20天最高涨跌幅(mfe_20)有显著关系。
    5层递进分析：Spearman秩相关 → 分桶均值 → 偏相关(控制波动率) → 分市场状态 → IC时序稳定性

Inputs:
    - stop_experiment/output/dataset_batches/batch_*.parquet (训练数据集)
    - 沪深300指数K线 (用于划分市场状态)

Outputs:
    - stop_experiment/output/filter_quality/correlation_analysis.csv
    - stop_experiment/output/filter_quality/ic_monthly.csv
    - stop_experiment/output/filter_quality/partial_corr_by_market_state.csv

How to Run:
    python -m stop_experiment.eval.feature_correlation_analysis
    python -m stop_experiment.eval.feature_correlation_analysis --sample-rate 0.1
    python -m stop_experiment.eval.feature_correlation_analysis --target mae_20

Examples:
    python -m stop_experiment.eval.feature_correlation_analysis --sample-rate 0.1
    python -m stop_experiment.eval.feature_correlation_analysis --target mfe_20

Side Effects:
    - 读取数据集parquet文件（只读）
    - 写入3个CSV到 filter_quality/ 目录
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from stop_experiment.pipeline.factor_columns import ALL_FEATURE_COLS, FACTOR_CATEGORIES
from stop_experiment.pipeline.stop_config import FILTER_QUALITY_DIR

TARGET_COL = "mfe_20"
VOL_CONTROL_COLS = ["volatility_20d", "atr_pct", "beta"]
MARKET_INDEX_CODE = "399300.SZ"


def _load_dataset(sample_rate: float = 1.0) -> pd.DataFrame:
    batches_dir = os.path.join(os.path.dirname(FILTER_QUALITY_DIR), "dataset_batches")
    if not os.path.exists(batches_dir):
        print(f"数据集目录不存在: {batches_dir}")
        sys.exit(1)

    files = sorted([f for f in os.listdir(batches_dir) if f.endswith(".parquet")])
    if not files:
        print(f"无parquet文件: {batches_dir}")
        sys.exit(1)

    dfs = []
    for f in tqdm(files, desc="加载数据集"):
        path = os.path.join(batches_dir, f)
        df = pd.read_parquet(path)
        if sample_rate < 1.0:
            df = df.sample(frac=sample_rate, random_state=42)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    if "obs_date" in data.columns:
        data["obs_date"] = pd.to_datetime(data["obs_date"])
    return data


def _load_market_index() -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    from config import DATABASE_URL
    engine = create_engine(DATABASE_URL)
    sql = (
        f"SELECT trade_date, close FROM index_daily "
        f"WHERE ts_code = '{MARKET_INDEX_CODE}' "
        f"ORDER BY trade_date"
    )
    idx = pd.read_sql(sql, engine)
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    idx["ret_20d"] = idx["close"].pct_change(20)
    return idx


def _spearman_fast(x: np.ndarray, y: np.ndarray) -> tuple:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return np.nan, np.nan
    rx = stats.rankdata(x[mask])
    ry = stats.rankdata(y[mask])
    r, p = stats.pearsonr(rx, ry)
    return r, p


def _pearson_fast(x: np.ndarray, y: np.ndarray) -> tuple:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p


def _partial_corr_residual(X: np.ndarray, Y: np.ndarray, C: np.ndarray) -> tuple:
    mask = np.isfinite(X) & np.isfinite(Y) & np.all(np.isfinite(C), axis=1)
    if mask.sum() < 100:
        return np.nan, np.nan
    Xc, Yc, Cc = X[mask], Y[mask], C[mask]

    Cc_int = np.column_stack([Cc, np.ones(len(Cc))])
    beta_x = np.linalg.lstsq(Cc_int, Xc, rcond=None)[0]
    beta_y = np.linalg.lstsq(Cc_int, Yc, rcond=None)[0]
    res_x = Xc - Cc_int @ beta_x
    res_y = Yc - Cc_int @ beta_y
    r, p = stats.pearsonr(res_x, res_y)
    return r, p


def _quantile_bucket_means(feature: np.ndarray, target: np.ndarray, n_buckets: int = 5) -> list:
    mask = np.isfinite(feature) & np.isfinite(target)
    if mask.sum() < n_buckets * 10:
        return [np.nan] * n_buckets
    f, t = feature[mask], target[mask]
    quantiles = np.percentile(f, np.linspace(0, 100, n_buckets + 1))
    means = []
    for i in range(n_buckets):
        lo, hi = quantiles[i], quantiles[i + 1]
        if i == n_buckets - 1:
            b_mask = (f >= lo) & (f <= hi)
        else:
            b_mask = (f >= lo) & (f < hi)
        means.append(t[b_mask].mean() if b_mask.sum() > 0 else np.nan)
    return means


def layer1_spearman(data: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    y = data[target].to_numpy(float)
    rows = []
    for feat in tqdm(features, desc="Layer1: Spearman"):
        x = data[feat].to_numpy(float)
        rho, pval = _spearman_fast(x, y)
        pr, pp = _pearson_fast(x, y)
        rows.append({
            "feature": feat,
            "spearman_rho": rho,
            "spearman_pval": pval,
            "pearson_r": pr,
            "pearson_pval": pp,
        })
    return pd.DataFrame(rows)


def layer2_quantile_buckets(data: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    y = data[target].to_numpy(float)
    rows = []
    for feat in tqdm(features, desc="Layer2: 分桶均值"):
        x = data[feat].to_numpy(float)
        bucket_means = _quantile_bucket_means(x, y, n_buckets=5)
        rows.append({
            "feature": feat,
            "q1_mfe": bucket_means[0],
            "q2_mfe": bucket_means[1],
            "q3_mfe": bucket_means[2],
            "q4_mfe": bucket_means[3],
            "q5_mfe": bucket_means[4],
            "q5_minus_q1": bucket_means[4] - bucket_means[0] if all(np.isfinite(bucket_means)) else np.nan,
        })
    return pd.DataFrame(rows)


def layer3_partial_corr(data: pd.DataFrame, features: list, target: str, controls: list) -> pd.DataFrame:
    available_controls = [c for c in controls if c in data.columns]
    if not available_controls:
        print("  ⚠️ 无可用控制变量，跳过偏相关分析")
        return pd.DataFrame(columns=["feature", "partial_rho_vol_controlled", "partial_pval"])

    C = data[available_controls].to_numpy(float)
    Y = data[target].to_numpy(float)
    rows = []
    for feat in tqdm(features, desc="Layer3: 偏相关(控制波动率)"):
        if feat in available_controls:
            rows.append({"feature": feat, "partial_rho_vol_controlled": np.nan, "partial_pval": np.nan})
            continue
        X = data[feat].to_numpy(float)
        rho, pval = _partial_corr_residual(X, Y, C)
        rows.append({"feature": feat, "partial_rho_vol_controlled": rho, "partial_pval": pval})
    return pd.DataFrame(rows)


def layer4_market_state(data: pd.DataFrame, features: list, target: str, index_df: pd.DataFrame) -> pd.DataFrame:
    data_with_index = data.merge(
        index_df[["trade_date", "ret_20d"]].rename(columns={"trade_date": "obs_date"}),
        on="obs_date", how="left",
    )
    data_with_index["market_state"] = pd.cut(
        data_with_index["ret_20d"],
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=["下跌", "震荡", "上涨"],
    )

    rows = []
    for state in ["上涨", "震荡", "下跌"]:
        subset = data_with_index[data_with_index["market_state"] == state]
        if len(subset) < 100:
            continue
        y = subset[target].to_numpy(float)
        for feat in features:
            x = subset[feat].to_numpy(float)
            rho, _ = _spearman_fast(x, y)
            rows.append({
                "market_state": state,
                "feature": feat,
                "spearman_rho": rho,
                "n_samples": len(subset),
            })
    return pd.DataFrame(rows)


def layer5_ic_stability(data: pd.DataFrame, features: list, target: str) -> tuple:
    data["month"] = data["obs_date"].dt.to_period("M")
    months = sorted(data["month"].unique())

    ic_rows = []
    for month in tqdm(months, desc="Layer5: IC时序"):
        subset = data[data["month"] == month]
        if len(subset) < 100:
            continue
        y = subset[target].to_numpy(float)
        for feat in features:
            x = subset[feat].to_numpy(float)
            rho, _ = _spearman_fast(x, y)
            ic_rows.append({
                "month": str(month),
                "feature": feat,
                "ic": rho,
                "n_samples": len(subset),
            })

    ic_df = pd.DataFrame(ic_rows)

    summary_rows = []
    for feat in features:
        feat_ic = ic_df[ic_df["feature"] == feat]["ic"].dropna()
        if len(feat_ic) < 3:
            summary_rows.append({
                "feature": feat,
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "ic_ir": np.nan,
                "ic_positive_rate": np.nan,
            })
            continue
        ic_mean = feat_ic.mean()
        ic_std = feat_ic.std()
        summary_rows.append({
            "feature": feat,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_mean / ic_std if ic_std > 0 else np.nan,
            "ic_positive_rate": (feat_ic > 0).mean(),
        })

    return pd.DataFrame(summary_rows), ic_df


def run_analysis(target: str = TARGET_COL, sample_rate: float = 1.0) -> None:
    print(f"加载训练数据集 (sample_rate={sample_rate})...")
    data = _load_dataset(sample_rate)
    print(f"  数据: {len(data)} 行, {len(data.columns)} 列")

    features = [f for f in ALL_FEATURE_COLS if f in data.columns]
    print(f"  可用特征: {len(features)}/{len(ALL_FEATURE_COLS)}")

    valid = data[features + [target]].dropna(subset=[target])
    print(f"  有效行: {len(valid)}")

    valid_with_dates = data[features + [target, "obs_date"]].dropna(subset=[target])

    cat_map = {}
    for cat, cols in FACTOR_CATEGORIES.items():
        for c in cols:
            cat_map[c] = cat

    print("\n===== Layer 1: Spearman 秩相关 =====")
    l1 = layer1_spearman(valid, features, target)

    print("\n===== Layer 2: 分桶均值 =====")
    l2 = layer2_quantile_buckets(valid, features, target)

    print("\n===== Layer 3: 偏相关(控制波动率) =====")
    l3 = layer3_partial_corr(valid, features, target, VOL_CONTROL_COLS)

    print("\n===== Layer 4: 分市场状态 =====")
    try:
        index_df = _load_market_index()
        l4 = layer4_market_state(valid_with_dates, features, target, index_df)
    except Exception as e:
        print(f"  ⚠️ 加载市场指数失败: {e}，跳过Layer4")
        l4 = pd.DataFrame(columns=["market_state", "feature", "spearman_rho", "n_samples"])

    print("\n===== Layer 5: IC时序稳定性 =====")
    l5_summary, ic_monthly = layer5_ic_stability(valid_with_dates, features, target)

    print("\n===== 合并结果 =====")
    result = l1.merge(l2, on="feature", how="outer")
    result = result.merge(l3, on="feature", how="outer")
    result = result.merge(l5_summary, on="feature", how="outer")
    result["category"] = result["feature"].map(cat_map).fillna("other")
    result = result.sort_values("spearman_rho", key=abs, ascending=False).reset_index(drop=True)

    os.makedirs(FILTER_QUALITY_DIR, exist_ok=True)

    corr_path = os.path.join(FILTER_QUALITY_DIR, "correlation_analysis.csv")
    result.to_csv(corr_path, index=False)
    print(f"  → {corr_path} ({len(result)} 行)")

    ic_path = os.path.join(FILTER_QUALITY_DIR, "ic_monthly.csv")
    ic_monthly.to_csv(ic_path, index=False)
    print(f"  → {ic_path} ({len(ic_monthly)} 行)")

    market_path = os.path.join(FILTER_QUALITY_DIR, "partial_corr_by_market_state.csv")
    l4.to_csv(market_path, index=False)
    print(f"  → {market_path} ({len(l4)} 行)")

    print("\n===== Top 15 特征 (Spearman |ρ| 排序) =====")
    top = result.head(15)
    for _, row in top.iterrows():
        rho = row["spearman_rho"]
        partial = row.get("partial_rho_vol_controlled", np.nan)
        ic_ir = row.get("ic_ir", np.nan)
        partial_str = f"{partial:+.3f}" if pd.notna(partial) else "N/A"
        ic_ir_str = f"{ic_ir:.2f}" if pd.notna(ic_ir) else "N/A"
        print(f"  {row['feature']:40s} ρ={rho:+.3f}  partial={partial_str}  IC_IR={ic_ir_str}  cat={row['category']}")


def main():
    parser = argparse.ArgumentParser(description="特征-标签相关性分析")
    parser.add_argument("--target", default=TARGET_COL, choices=["mfe_20", "mae_20"],
                        help="目标变量")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="采样率(0~1), 1.0=全量, 0.1=10%%采样")
    args = parser.parse_args()

    run_analysis(target=args.target, sample_rate=args.sample_rate)


if __name__ == "__main__":
    main()
