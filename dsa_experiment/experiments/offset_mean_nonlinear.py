#!/usr/bin/env python3
"""
Offset Mean 非线性规律探索实验

Purpose: 验证 offset_mean 与未来收益之间是否存在倒 U 型非线性关系
Inputs: dsa_selection 表, stock_k_data 表
Outputs: dsa_experiment/output/offset_mean_*.csv, dsa_experiment/output/offset_mean_*.png

How to Run:
    python dsa_experiment/experiments/offset_mean_nonlinear.py
    python dsa_experiment/experiments/offset_mean_nonlinear.py --start 2024-01-01
    python dsa_experiment/experiments/offset_mean_nonlinear.py --start 2025-01-01 --sample-limit 5000

Examples:
    python dsa_experiment/experiments/offset_mean_nonlinear.py --start 2025-01-01
    python dsa_experiment/experiments/offset_mean_nonlinear.py

Side Effects: 只读操作，输出 CSV 和 PNG 文件
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

# 主要分析的未来收益周期
PRIMARY_HORIZONS = [5, 10, 20]


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
    """为选股结果计算未来收益标签（ret/mae/mfe）。"""
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

    valid_count = sel_df[[c for c in label_cols if c in sel_df.columns]].notna().sum()
    print(f"  未来收益标签非空统计:")
    for col in label_cols:
        if col in valid_count.index:
            print(f"    {col}: {valid_count[col]}/{len(sel_df)}")

    return sel_df


# ──────────────────────────────────────────────
# Decile 分组统计
# ──────────────────────────────────────────────

def compute_decile_stats(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """将 offset_mean 分成 n_bins 个分位组，计算各组的收益统计。"""
    valid = df[["offset_mean", "selection_date"] + [f"ret_{n}" for n in HORIZONS]].dropna(subset=["offset_mean"])
    if len(valid) < n_bins * 5:
        print(f"  样本量不足 ({len(valid)})，减少分位数")
        n_bins = max(3, len(valid) // 5)

    valid = valid.copy()
    try:
        valid["decile"] = pd.qcut(valid["offset_mean"], n_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        print("  qcut 失败，使用等宽分箱")
        valid["decile"] = pd.cut(valid["offset_mean"], n_bins, labels=False, duplicates="drop") + 1

    results = []
    for decile, group in valid.groupby("decile", observed=True):
        row = {
            "decile": int(decile),
            "n": len(group),
            "offset_mean_min": group["offset_mean"].min(),
            "offset_mean_max": group["offset_mean"].max(),
            "offset_mean_mean": group["offset_mean"].mean(),
        }
        for n in HORIZONS:
            ret_col = f"ret_{n}"
            if ret_col not in group.columns:
                continue
            rets = group[ret_col].dropna()
            if len(rets) == 0:
                continue
            row[f"ret_{n}_mean"] = rets.mean()
            row[f"ret_{n}_median"] = rets.median()
            row[f"ret_{n}_winrate"] = (rets > 0).mean() * 100
        results.append(row)

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# D10 细分统计
# ──────────────────────────────────────────────

def compute_d10_subdivision(df: pd.DataFrame, n_sub_bins: int = 10) -> pd.DataFrame:
    """对 offset_mean 的 D10（最高分位）再细分为 n_sub_bins 组，观察极端高值区域收益分布。"""
    # 先计算 D10 的下界
    valid = df[["offset_mean"] + [f"ret_{n}" for n in HORIZONS]].dropna(subset=["offset_mean"])
    if len(valid) < 100:
        return pd.DataFrame()

    try:
        deciles = pd.qcut(valid["offset_mean"], 10, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    d10_mask = deciles == 9  # 第 10 组（0-indexed）
    d10_data = valid[d10_mask].copy()

    if len(d10_data) < n_sub_bins * 5:
        print(f"  D10 样本量不足 ({len(d10_data)})，减少细分组数")
        n_sub_bins = max(3, len(d10_data) // 5)

    try:
        d10_data["sub_decile"] = pd.qcut(d10_data["offset_mean"], n_sub_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        d10_data["sub_decile"] = pd.cut(d10_data["offset_mean"], n_sub_bins, labels=False, duplicates="drop") + 1

    results = []
    for sub_decile, group in d10_data.groupby("sub_decile", observed=True):
        row = {
            "sub_decile": int(sub_decile),
            "n": len(group),
            "offset_mean_min": group["offset_mean"].min(),
            "offset_mean_max": group["offset_mean"].max(),
            "offset_mean_mean": group["offset_mean"].mean(),
        }
        for n in HORIZONS:
            ret_col = f"ret_{n}"
            if ret_col not in group.columns:
                continue
            rets = group[ret_col].dropna()
            if len(rets) == 0:
                continue
            row[f"ret_{n}_mean"] = rets.mean()
            row[f"ret_{n}_median"] = rets.median()
            row[f"ret_{n}_winrate"] = (rets > 0).mean() * 100
        results.append(row)

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# 高值区分组统计
# ──────────────────────────────────────────────

def compute_high_value_subdivision(df: pd.DataFrame, threshold: float = 0.2, n_sub_bins: int = 10) -> pd.DataFrame:
    """对 offset_mean >= threshold 的高值区再细分，计算收益/胜率/夏普率。"""
    valid = df[["offset_mean"] + [f"ret_{n}" for n in HORIZONS]].dropna(subset=["offset_mean"])
    hv_data = valid[valid["offset_mean"] >= threshold].copy()

    if len(hv_data) < n_sub_bins * 5:
        print(f"  高值区样本量不足 ({len(hv_data)})，减少细分组数")
        n_sub_bins = max(3, len(hv_data) // 5)

    print(f"  高值区 (offset_mean >= {threshold}) 样本数: {len(hv_data)}")

    try:
        hv_data["sub_bin"] = pd.qcut(hv_data["offset_mean"], n_sub_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        hv_data["sub_bin"] = pd.cut(hv_data["offset_mean"], n_sub_bins, labels=False, duplicates="drop") + 1

    results = []
    for sub_bin, group in hv_data.groupby("sub_bin", observed=True):
        row = {
            "sub_bin": int(sub_bin),
            "n": len(group),
            "offset_mean_min": group["offset_mean"].min(),
            "offset_mean_max": group["offset_mean"].max(),
            "offset_mean_mean": group["offset_mean"].mean(),
        }
        for n in HORIZONS:
            ret_col = f"ret_{n}"
            if ret_col not in group.columns:
                continue
            rets = group[ret_col].dropna()
            if len(rets) == 0:
                continue
            row[f"ret_{n}_mean"] = rets.mean()
            row[f"ret_{n}_median"] = rets.median()
            row[f"ret_{n}_winrate"] = (rets > 0).mean() * 100
            # 夏普率：mean / std（年化因子忽略，仅做组间对比）
            if rets.std() > 0:
                row[f"ret_{n}_sharpe"] = rets.mean() / rets.std()
            else:
                row[f"ret_{n}_sharpe"] = np.nan
        results.append(row)

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# 最优区间 offset_percentile 与未来收益关系
# ──────────────────────────────────────────────

def compute_percentile_in_optimal_zone(df: pd.DataFrame) -> pd.DataFrame:
    """在最优区间 offset_mean ∈ [0.239, 0.268] 内，探索 offset_percentile 与未来收益的关系。"""
    valid = df[[c for c in ["offset_mean", "offset_percentile"] + [f"ret_{n}" for n in HORIZONS] if c in df.columns]].dropna()
    opt_data = valid[(valid["offset_mean"] >= 0.239) & (valid["offset_mean"] <= 0.268)].copy()

    if len(opt_data) < 50:
        print(f"  最优区间样本量不足 ({len(opt_data)})")
        return pd.DataFrame()

    print(f"  最优区间 (offset_mean ∈ [0.239, 0.268]) 样本数: {len(opt_data)}")

    try:
        opt_data["p_bin"] = pd.qcut(opt_data["offset_percentile"], 10, labels=False, duplicates="drop") + 1
    except ValueError:
        opt_data["p_bin"] = pd.cut(opt_data["offset_percentile"], 10, labels=False, duplicates="drop") + 1

    results = []
    for p_bin, group in opt_data.groupby("p_bin", observed=True):
        row = {
            "p_bin": int(p_bin),
            "n": len(group),
            "p_min": group["offset_percentile"].min(),
            "p_max": group["offset_percentile"].max(),
            "p_mean": group["offset_percentile"].mean(),
        }
        for n in HORIZONS:
            ret_col = f"ret_{n}"
            if ret_col not in group.columns:
                continue
            rets = group[ret_col].dropna()
            if len(rets) == 0:
                continue
            row[f"ret_{n}_mean"] = rets.mean()
            row[f"ret_{n}_median"] = rets.median()
            row[f"ret_{n}_winrate"] = (rets > 0).mean() * 100
            if rets.std() > 0:
                row[f"ret_{n}_sharpe"] = rets.mean() / rets.std()
            else:
                row[f"ret_{n}_sharpe"] = np.nan
        results.append(row)

    df_groups = pd.DataFrame(results)

    # 计算 Spearman 相关系数
    if not df_groups.empty:
        print(f"\n  offset_percentile 分组统计:")
        for n in HORIZONS:
            ret_col = f"ret_{n}"
            if ret_col not in df_groups.columns:
                continue
            # 逐组计算
            opt_data_valid = opt_data.dropna(subset=["offset_percentile", ret_col])
            if len(opt_data_valid) > 10:
                corr, p_val = sp_stats.spearmanr(opt_data_valid["offset_percentile"], opt_data_valid[ret_col])
                print(f"    {ret_col} vs offset_percentile: Spearman r={corr:.4f}, p={p_val:.4f}")

    return df_groups


# ──────────────────────────────────────────────
# 非线性回归检验
# ──────────────────────────────────────────────

def compute_nonlinear_regression(df: pd.DataFrame) -> pd.DataFrame:
    """对 offset_mean vs ret_N 做线性和二次回归，检验二次项是否显著。"""
    results = []
    for n in HORIZONS:
        ret_col = f"ret_{n}"
        valid = df[["offset_mean", ret_col]].dropna()
        if len(valid) < 100:
            continue

        x = valid["offset_mean"].values
        y = valid[ret_col].values

        # 线性回归: y = b1*x + b0
        slope_lin, intercept_lin, r_lin, p_lin, se_lin = stats.linregress(x, y)
        ss_res_lin = np.sum((y - (intercept_lin + slope_lin * x)) ** 2)

        # 二次回归: y = a*x^2 + b*x + c
        X_quad = np.column_stack([x ** 2, x, np.ones_like(x)])
        beta_quad, residuals_quad, _, _ = np.linalg.lstsq(X_quad, y, rcond=None)
        a_quad, b_quad, c_quad = beta_quad
        y_pred_quad = X_quad @ beta_quad
        ss_res_quad = np.sum((y - y_pred_quad) ** 2)

        # F 检验: 二次 vs 线性
        n_obs = len(y)
        p_lin_model = 2  # 线性参数数
        p_quad_model = 3  # 二次参数数
        f_stat = ((ss_res_lin - ss_res_quad) / (p_quad_model - p_lin_model)) / (ss_res_quad / (n_obs - p_quad_model))
        f_pvalue = 1 - stats.f.cdf(f_stat, p_quad_model - p_lin_model, n_obs - p_quad_model)

        # 二次项 t 检验
        # 用最小二乘的方差估计
        mse_quad = ss_res_quad / (n_obs - p_quad_model)
        cov_matrix = mse_quad * np.linalg.inv(X_quad.T @ X_quad)
        se_a = np.sqrt(cov_matrix[0, 0])
        t_stat_a = a_quad / se_a if se_a > 0 else 0
        p_value_a = 2 * (1 - stats.t.cdf(abs(t_stat_a), n_obs - p_quad_model))

        # 二次回归 R²
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0
        r2_lin = r_lin ** 2

        # 顶点位置（如果 a < 0，倒 U 型）
        vertex_x = -b_quad / (2 * a_quad) if a_quad != 0 else np.nan
        vertex_y = a_quad * vertex_x ** 2 + b_quad * vertex_x + c_quad if a_quad != 0 else np.nan

        results.append({
            "horizon": n,
            "n_obs": n_obs,
            # 线性
            "lin_slope": slope_lin,
            "lin_r2": r2_lin,
            "lin_pvalue": p_lin,
            # 二次
            "quad_a": a_quad,
            "quad_b": b_quad,
            "quad_c": c_quad,
            "quad_r2": r2_quad,
            "quad_a_tstat": t_stat_a,
            "quad_a_pvalue": p_value_a,
            # F 检验
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            # 顶点
            "vertex_offset_mean": vertex_x,
            "vertex_ret": vertex_y,
            # 判断
            "is_inverted_u": a_quad < 0 and p_value_a < 0.05,
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# LOWESS 平滑
# ──────────────────────────────────────────────

def lowess_smooth(x, y, frac=0.3, max_samples=20000):
    """简单 LOWESS 平滑实现（使用 statsmodels 如可用，否则降级）。"""
    # 大样本时子采样加速
    if len(x) > max_samples:
        idx = np.random.RandomState(42).choice(len(x), max_samples, replace=False)
        x = x[idx]
        y = y[idx]

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
        result = sm_lowess(y, x, frac=frac, return_sorted=True)
        return result[:, 0], result[:, 1]
    except ImportError:
        # 降级：滚动均值
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]
        window = max(int(len(x) * frac), 10)
        x_smooth = np.convolve(x_sorted, np.ones(window) / window, mode='valid')
        y_smooth = np.convolve(y_sorted, np.ones(window) / window, mode='valid')
        return x_smooth, y_smooth


# ──────────────────────────────────────────────
# 样条回归
# ──────────────────────────────────────────────

def compute_spline_fit(df: pd.DataFrame, n_knots: int = 5) -> pd.DataFrame:
    """使用三次样条拟合 offset_mean → ret_N 关系。"""
    try:
        from scipy.interpolate import UnivariateSpline
    except ImportError:
        print("  scipy.interpolate 不可用，跳过样条回归")
        return pd.DataFrame()

    results = []
    for n in PRIMARY_HORIZONS:
        ret_col = f"ret_{n}"
        valid = df[["offset_mean", ret_col]].dropna()
        if len(valid) < 100:
            continue

        # 大样本时子采样加速（UnivariateSpline 在 50 万+样本上极慢）
        if len(valid) > 20000:
            sample = valid.sample(20000, random_state=42)
        else:
            sample = valid
        x = sample["offset_mean"].values
        y = sample[ret_col].values

        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        # 使用 UnivariateSpline，加大平滑防止过拟合
        try:
            # s 参数越大越平滑：使用 len * var * 0.5 确保充分平滑
            spline = UnivariateSpline(x_sorted, y_sorted, k=3, s=len(x_sorted) * np.var(y_sorted) * 0.5)
        except Exception:
            continue

        # 在均匀网格上预测
        x_grid = np.linspace(x_sorted.min(), x_sorted.max(), 200)
        y_grid = spline(x_grid)

        # 找峰值
        peak_idx = np.argmax(y_grid)
        peak_x = x_grid[peak_idx]
        peak_y = y_grid[peak_idx]

        results.append({
            "horizon": n,
            "n_obs": len(valid),
            "spline_peak_offset_mean": peak_x,
            "spline_peak_ret": peak_y,
            "spline_offset_mean_min": x_sorted.min(),
            "spline_offset_mean_max": x_sorted.max(),
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# 甜点区间识别
# ──────────────────────────────────────────────

def identify_sweet_spot(df: pd.DataFrame, reg_df: pd.DataFrame) -> pd.DataFrame:
    """综合多种方法识别 offset_mean 的甜点区间。"""
    results = []

    for n in PRIMARY_HORIZONS:
        ret_col = f"ret_{n}"
        valid = df[["offset_mean", ret_col, "selection_date"]].dropna()
        if len(valid) < 100:
            continue

        # 方法1：二次回归顶点 ± 0.5*std（仅倒U型成立时可用）
        reg_row = reg_df[reg_df["horizon"] == n]
        vertex = np.nan
        sweet_low_quad = np.nan
        sweet_high_quad = np.nan
        if not reg_row.empty and reg_row.iloc[0]["is_inverted_u"]:
            vertex = reg_row.iloc[0]["vertex_offset_mean"]
            om_std = valid["offset_mean"].std()
            sweet_low_quad = vertex - 0.5 * om_std
            sweet_high_quad = vertex + 0.5 * om_std

        # 方法2：decile 分析中收益最高的区间（始终可用）
        try:
            deciles = pd.qcut(valid["offset_mean"], 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        valid_copy = valid.copy()
        valid_copy["decile"] = deciles
        decile_means = valid_copy.groupby("decile", observed=True)[ret_col].mean()
        best_decile = decile_means.idxmax()
        best_decile_mask = valid_copy["decile"] == best_decile
        decile_low = valid_copy.loc[best_decile_mask, "offset_mean"].min()
        decile_high = valid_copy.loc[best_decile_mask, "offset_mean"].max()

        # 综合甜点区间：优先用二次回归，否则用 decile
        if pd.notna(sweet_low_quad) and pd.notna(sweet_high_quad):
            sweet_low = sweet_low_quad
            sweet_high = sweet_high_quad
            method = "quadratic"
        else:
            sweet_low = decile_low
            sweet_high = decile_high
            method = "decile_best"

        # 甜点区间内 vs 区间外收益对比
        in_spot = valid[(valid["offset_mean"] >= sweet_low) & (valid["offset_mean"] <= sweet_high)]
        out_spot = valid[(valid["offset_mean"] < sweet_low) | (valid["offset_mean"] > sweet_high)]
        if len(in_spot) > 10 and len(out_spot) > 10:
            t_stat, t_pvalue = stats.ttest_ind(in_spot[ret_col], out_spot[ret_col], equal_var=False)
        else:
            t_stat, t_pvalue = np.nan, np.nan

        results.append({
            "horizon": n,
            "method": method,
            "vertex_offset_mean": vertex,
            "sweet_spot_low": sweet_low,
            "sweet_spot_high": sweet_high,
            "best_decile_low": decile_low,
            "best_decile_high": decile_high,
            "in_spot_n": len(in_spot),
            "in_spot_ret_mean": in_spot[ret_col].mean() if len(in_spot) > 0 else np.nan,
            "in_spot_ret_median": in_spot[ret_col].median() if len(in_spot) > 0 else np.nan,
            "in_spot_winrate": (in_spot[ret_col] > 0).mean() * 100 if len(in_spot) > 0 else np.nan,
            "out_spot_n": len(out_spot),
            "out_spot_ret_mean": out_spot[ret_col].mean() if len(out_spot) > 0 else np.nan,
            "t_stat": t_stat,
            "t_pvalue": t_pvalue,
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# 分层分析
# ──────────────────────────────────────────────

def stratified_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """分层分析 offset_mean 非线性规律是否在控制变量后仍然成立。"""
    results = []

    stratify_configs = [
        ("dsa_dir_bars", [0, 100, 200, 500], "趋势长度"),
        ("vwap_ret_total", None, "VWAP收益率"),
    ]

    for col, bins, label in stratify_configs:
        if col not in df.columns:
            continue

        valid = df[["offset_mean", col, "ret_5"]].dropna()
        if len(valid) < 200:
            continue

        if bins is not None:
            valid = valid.copy()
            try:
                valid["stratum"] = pd.cut(valid[col], bins=bins, labels=False, duplicates="drop")
            except ValueError:
                continue
        else:
            valid = valid.copy()
            try:
                valid["stratum"] = pd.qcut(valid[col], 3, labels=False, duplicates="drop")
            except ValueError:
                continue

        for stratum, group in valid.groupby("stratum", observed=True):
            if len(group) < 50:
                continue

            x = group["offset_mean"].values
            y = group["ret_5"].values

            # 二次回归
            X_quad = np.column_stack([x ** 2, x, np.ones_like(x)])
            try:
                beta_quad, residuals, _, _ = np.linalg.lstsq(X_quad, y, rcond=None)
            except Exception:
                continue
            a_quad = beta_quad[0]

            # t 检验
            n_obs = len(y)
            ss_res = np.sum((y - X_quad @ beta_quad) ** 2)
            mse = ss_res / (n_obs - 3) if n_obs > 3 else np.nan
            if mse > 0:
                try:
                    cov_matrix = mse * np.linalg.inv(X_quad.T @ X_quad)
                    se_a = np.sqrt(cov_matrix[0, 0])
                    t_stat = a_quad / se_a if se_a > 0 else 0
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - 3))
                except Exception:
                    p_value = np.nan
            else:
                p_value = np.nan

            results.append({
                "stratify_col": col,
                "stratify_label": label,
                "stratum": int(stratum) if pd.notna(stratum) else -1,
                "stratum_min": group[col].min(),
                "stratum_max": group[col].max(),
                "n_obs": n_obs,
                "quad_a": a_quad,
                "is_inverted_u": a_quad < 0,
                "quad_a_pvalue": p_value,
                "mean_ret5": y.mean(),
            })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# 分年度稳健性检验
# ──────────────────────────────────────────────

def yearly_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """按年度拆分，检验倒 U 型是否跨年度稳定。"""
    results = []
    df = df.copy()
    df["year"] = df["selection_date"].dt.year

    for year, year_df in df.groupby("year"):
        valid = year_df[["offset_mean", "ret_5"]].dropna()
        if len(valid) < 100:
            continue

        x = valid["offset_mean"].values
        y = valid["ret_5"].values

        # 二次回归
        X_quad = np.column_stack([x ** 2, x, np.ones_like(x)])
        try:
            beta_quad, _, _, _ = np.linalg.lstsq(X_quad, y, rcond=None)
        except Exception:
            continue
        a_quad = beta_quad[0]

        n_obs = len(y)
        ss_res = np.sum((y - X_quad @ beta_quad) ** 2)
        mse = ss_res / (n_obs - 3) if n_obs > 3 else np.nan
        if mse > 0:
            try:
                cov_matrix = mse * np.linalg.inv(X_quad.T @ X_quad)
                se_a = np.sqrt(cov_matrix[0, 0])
                t_stat = a_quad / se_a if se_a > 0 else 0
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - 3))
            except Exception:
                p_value = np.nan
        else:
            p_value = np.nan

        results.append({
            "year": year,
            "n_obs": n_obs,
            "quad_a": a_quad,
            "is_inverted_u": a_quad < 0,
            "quad_a_pvalue": p_value,
            "mean_ret5": y.mean(),
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────

def _setup_chinese_font():
    """配置 matplotlib 中文字体。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 尝试常见中文字体
    chinese_fonts = [
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "Noto Sans SC",
        "SimHei", "Microsoft YaHei", "STHeiti",
    ]
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    # 无中文字体时用英文标签
    plt.rcParams["axes.unicode_minus"] = False
    return None


def plot_nonlinear_analysis(df: pd.DataFrame, reg_df: pd.DataFrame, output_dir: str):
    """绘制核心可视化图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Offset Mean 非线性规律探索", fontsize=16)

    # 1. Decile 柱状图 (ret_5)
    ax = axes[0, 0]
    valid = df[["offset_mean", "ret_5"]].dropna()
    if len(valid) > 50:
        valid = valid.copy()
        try:
            valid["decile"] = pd.qcut(valid["offset_mean"], 10, labels=False, duplicates="drop") + 1
        except ValueError:
            valid["decile"] = pd.cut(valid["offset_mean"], 10, labels=False, duplicates="drop") + 1
        decile_means = valid.groupby("decile", observed=True)["ret_5"].agg(["mean", "count", "sem"])
        decile_means["ci95"] = 1.96 * decile_means["sem"]
        bars = ax.bar(decile_means.index, decile_means["mean"] * 100,
                      yerr=decile_means["ci95"] * 100, capsize=3, color="steelblue", alpha=0.8)
        ax.set_xlabel("Offset Mean Decile")
        ax.set_ylabel("平均 ret_5 (%)")
        ax.set_title("Decile 分组: offset_mean → ret_5")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        for i, (idx, row) in enumerate(decile_means.iterrows()):
            ax.text(idx, row["mean"] * 100 + row["ci95"] * 100 + 0.1,
                    f'n={int(row["count"])}', ha='center', fontsize=7)

    # 2. LOWESS 散点图
    ax = axes[0, 1]
    for n in PRIMARY_HORIZONS:
        ret_col = f"ret_{n}"
        valid = df[["offset_mean", ret_col]].dropna()
        if len(valid) < 50:
            continue
        x = valid["offset_mean"].values
        y = valid[ret_col].values
        x_smooth, y_smooth = lowess_smooth(x, y, frac=0.3)
        ax.plot(x_smooth, y_smooth * 100, label=f"ret_{n}", linewidth=2)
    ax.set_xlabel("Offset Mean")
    ax.set_ylabel("未来收益 (%)")
    ax.set_title("LOWESS 平滑曲线")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 3. 二次拟合曲线
    ax = axes[1, 0]
    for n in PRIMARY_HORIZONS:
        ret_col = f"ret_{n}"
        reg_row = reg_df[reg_df["horizon"] == n]
        if reg_row.empty:
            continue
        row = reg_row.iloc[0]
        valid = df[["offset_mean", ret_col]].dropna()
        if len(valid) < 50:
            continue
        x_grid = np.linspace(valid["offset_mean"].min(), valid["offset_mean"].max(), 200)
        y_quad = row["quad_a"] * x_grid ** 2 + row["quad_b"] * x_grid + row["quad_c"]
        ax.plot(x_grid, y_quad * 100, label=f"ret_{n} (a={row['quad_a']:.2f}, p={row['quad_a_pvalue']:.4f})", linewidth=2)
    ax.set_xlabel("Offset Mean")
    ax.set_ylabel("未来收益 (%)")
    ax.set_title("二次回归拟合")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 4. 多周期 Decile 热力图
    ax = axes[1, 1]
    valid = df[["offset_mean"] + [f"ret_{n}" for n in HORIZONS]].dropna(subset=["offset_mean"])
    if len(valid) > 50:
        valid = valid.copy()
        try:
            valid["decile"] = pd.qcut(valid["offset_mean"], 10, labels=False, duplicates="drop") + 1
        except ValueError:
            valid["decile"] = pd.cut(valid["offset_mean"], 10, labels=False, duplicates="drop") + 1
        heatmap_data = []
        for n in HORIZONS:
            ret_col = f"ret_{n}"
            if ret_col not in valid.columns:
                continue
            decile_means = valid.groupby("decile", observed=True)[ret_col].mean() * 100
            heatmap_data.append(decile_means)
        if heatmap_data:
            heat_df = pd.DataFrame(heatmap_data, index=[f"ret_{n}" for n in HORIZONS if f"ret_{n}" in valid.columns])
            im = ax.imshow(heat_df.values, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(heat_df.columns)))
            ax.set_xticklabels([f"D{int(c)}" for c in heat_df.columns])
            ax.set_yticks(range(len(heat_df.index)))
            ax.set_yticklabels(heat_df.index)
            ax.set_title("Decile × Horizon 收益热力图 (%)")
            plt.colorbar(im, ax=ax, label="平均收益 (%)")
            # 标注数值
            for i in range(len(heat_df.index)):
                for j in range(len(heat_df.columns)):
                    val = heat_df.iloc[i, j]
                    if pd.notna(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if abs(val) > 1 else "black")

    plt.tight_layout()
    path = os.path.join(output_dir, "offset_mean_nonlinear.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  核心可视化已保存: {path}")


def plot_by_year(df: pd.DataFrame, output_dir: str):
    """按年度绘制 LOWESS 图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.copy()
    df["year"] = df["selection_date"].dt.year
    years = sorted(df["year"].dropna().unique())
    if len(years) == 0:
        return

    n_cols = min(3, len(years))
    n_rows = (len(years) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, year in enumerate(years):
        ax = axes[idx // n_cols, idx % n_cols]
        year_df = df[df["year"] == year]
        valid = year_df[["offset_mean", "ret_5"]].dropna()
        if len(valid) < 30:
            ax.text(0.5, 0.5, f"{year}: 样本不足", transform=ax.transAxes, ha="center")
            continue
        x = valid["offset_mean"].values
        y = valid["ret_5"].values
        x_smooth, y_smooth = lowess_smooth(x, y, frac=0.4)
        ax.plot(x_smooth, y_smooth * 100, linewidth=2, color="steelblue")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{year} (n={len(valid)})")
        ax.set_xlabel("Offset Mean")
        ax.set_ylabel("ret_5 (%)")

    # 隐藏多余子图
    for idx in range(len(years), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Offset Mean → ret_5 分年度 LOWESS", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "offset_mean_by_year.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  分年度图已保存: {path}")


def plot_stratified(strat_df: pd.DataFrame, output_dir: str):
    """绘制分层分析图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if strat_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, col in enumerate(strat_df["stratify_col"].unique()):
        ax = axes[idx]
        sub = strat_df[strat_df["stratify_col"] == col]
        colors = ["green" if row["is_inverted_u"] else "red" for _, row in sub.iterrows()]
        bars = ax.bar(range(len(sub)), sub["quad_a"], color=colors, alpha=0.7)
        ax.set_xticks(range(len(sub)))
        labels = []
        for _, row in sub.iterrows():
            sig = "***" if row.get("quad_a_pvalue", 1) < 0.01 else ("**" if row.get("quad_a_pvalue", 1) < 0.05 else ("*" if row.get("quad_a_pvalue", 1) < 0.1 else ""))
            labels.append(f"[{row['stratum_min']:.2f},{row['stratum_max']:.2f}]\nn={row['n_obs']}{sig}")
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("二次项系数 (a)")
        ax.set_title(f"分层: {sub.iloc[0]['stratify_label'] if len(sub) > 0 else col}")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "offset_mean_stratified.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  分层分析图已保存: {path}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Offset Mean 非线性规律探索实验")
    parser.add_argument("--start", type=str, default=None, help="起始日期，如 2024-01-01")
    parser.add_argument("--sample-limit", type=int, default=None, help="采样限制（调试用）")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("Offset Mean 非线性规律探索实验")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/8] 加载数据 ...")
    sel_df = load_data(start_date=args.start, sample_limit=args.sample_limit)
    if sel_df.empty:
        print("  无数据，退出")
        return

    # 2. 计算未来收益
    print("\n[2/8] 计算未来收益 ...")
    df = compute_future_returns(sel_df)

    # offset_mean 基本分布
    print(f"\n  offset_mean 分布:")
    print(f"    均值: {df['offset_mean'].mean():.4f}")
    print(f"    中位数: {df['offset_mean'].median():.4f}")
    print(f"    标准差: {df['offset_mean'].std():.4f}")
    print(f"    范围: [{df['offset_mean'].min():.4f}, {df['offset_mean'].max():.4f}]")
    print(f"    非空数: {df['offset_mean'].notna().sum()}/{len(df)}")

    # 3. Decile 分组统计
    print("\n[3/8] Decile 分组统计 ...")
    decile_df = compute_decile_stats(df, n_bins=10)
    decile_path = os.path.join(OUTPUT_DIR, "offset_mean_decile_stats.csv")
    decile_df.to_csv(decile_path, index=False)
    print(f"  Decile 统计已保存: {decile_path}")

    print(f"\n  Decile × ret_5:")
    print(f"  {'Decile':>7} {'N':>6} {'OM范围':>20} {'均值%':>8} {'中位%':>8} {'胜率%':>7}")
    print(f"  {'-'*70}")
    for _, row in decile_df.iterrows():
        om_range = f"[{row['offset_mean_min']:.3f}, {row['offset_mean_max']:.3f}]"
        ret_mean = row.get("ret_5_mean", np.nan)
        ret_med = row.get("ret_5_median", np.nan)
        wr = row.get("ret_5_winrate", np.nan)
        print(f"  {int(row['decile']):>7} {int(row['n']):>6} {om_range:>20} "
              f"{ret_mean*100:>8.2f} {ret_med*100:>8.2f} {wr:>7.1f}")

    # 3.5 D10 细分统计
    print("\n[3.5/8] D10 细分统计 ...")
    d10_df = compute_d10_subdivision(df, n_sub_bins=10)
    if not d10_df.empty:
        d10_path = os.path.join(OUTPUT_DIR, "offset_mean_d10_subdivision.csv")
        d10_df.to_csv(d10_path, index=False)
        print(f"  D10 细分已保存: {d10_path}")

        print(f"\n  D10 Sub-Decile × ret_5/10/20:")
        print(f"  {'SubD':>5} {'N':>6} {'OM范围':>22} {'ret5%':>7} {'ret10%':>7} {'ret20%':>7} {'ret40%':>7} {'ret60%':>7} {'ret80%':>7} {'胜率5d%':>7}")
        print(f"  {'-'*90}")
        for _, row in d10_df.iterrows():
            om_range = f"[{row['offset_mean_min']:.3f}, {row['offset_mean_max']:.3f}]"
            r5 = row.get("ret_5_mean", np.nan)
            r10 = row.get("ret_10_mean", np.nan)
            r20 = row.get("ret_20_mean", np.nan)
            r40 = row.get("ret_40_mean", np.nan)
            r60 = row.get("ret_60_mean", np.nan)
            r80 = row.get("ret_80_mean", np.nan)
            wr = row.get("ret_5_winrate", np.nan)
            print(f"  {int(row['sub_decile']):>5} {int(row['n']):>6} {om_range:>22} "
                  f"{r5*100:>7.2f} {r10*100:>7.2f} {r20*100:>7.2f} {r40*100:>7.2f} {r60*100:>7.2f} {r80*100:>7.2f} {wr:>7.1f}")

    # 3.6 高值区分组统计
    print("\n[3.6/8] 高值区 (offset_mean >= 0.2) 细分统计 ...")
    hv_df = compute_high_value_subdivision(df, threshold=0.2, n_sub_bins=10)
    if not hv_df.empty:
        hv_path = os.path.join(OUTPUT_DIR, "offset_mean_high_value_subdivision.csv")
        hv_df.to_csv(hv_path, index=False)
        print(f"  高值区细分已保存: {hv_path}")

        print(f"\n  高值区 Sub-Decile × ret/胜率/夏普:")
        print(f"  {'SubD':>5} {'N':>6} {'OM范围':>22} {'ret5%':>7} {'胜率5d':>7} {'夏普5d':>7} {'ret10%':>7} {'胜率10d':>7} {'夏普10d':>7} {'ret20%':>7} {'胜率20d':>7} {'夏普20d':>7}")
        print(f"  {'-'*120}")
        for _, row in hv_df.iterrows():
            om_range = f"[{row['offset_mean_min']:.3f}, {row['offset_mean_max']:.3f}]"
            r5 = row.get("ret_5_mean", np.nan)
            wr5 = row.get("ret_5_winrate", np.nan)
            sh5 = row.get("ret_5_sharpe", np.nan)
            r10 = row.get("ret_10_mean", np.nan)
            wr10 = row.get("ret_10_winrate", np.nan)
            sh10 = row.get("ret_10_sharpe", np.nan)
            r20 = row.get("ret_20_mean", np.nan)
            wr20 = row.get("ret_20_winrate", np.nan)
            sh20 = row.get("ret_20_sharpe", np.nan)
            print(f"  {int(row['sub_bin']):>5} {int(row['n']):>6} {om_range:>22} "
                  f"{r5*100:>7.2f} {wr5:>7.1f} {sh5:>7.3f} "
                  f"{r10*100:>7.2f} {wr10:>7.1f} {sh10:>7.3f} "
                  f"{r20*100:>7.2f} {wr20:>7.1f} {sh20:>7.3f}")

    # 3.7 最优区间 offset_percentile 与未来收益关系
    print("\n[3.7/8] 最优区间 offset_percentile 与未来收益关系 ...")
    pct_df = compute_percentile_in_optimal_zone(df)
    if not pct_df.empty:
        pct_path = os.path.join(OUTPUT_DIR, "optimal_zone_percentile_relation.csv")
        pct_df.to_csv(pct_path, index=False)
        print(f"  最优区间 percentile 分组已保存: {pct_path}")

        print(f"\n  offset_percentile 分组 × ret/胜率/夏普:")
        print(f"  {'P':>4} {'N':>6} {'P范围':>22} {'ret5%':>7} {'胜率5d':>7} {'夏普5d':>7} {'ret10%':>7} {'胜率10d':>7} {'夏普10d':>7} {'ret20%':>7} {'胜率20d':>7} {'夏普20d':>7}")
        print(f"  {'-'*120}")
        for _, row in pct_df.iterrows():
            p_range = f"[{row['p_min']:.3f}, {row['p_max']:.3f}]"
            r5 = row.get("ret_5_mean", np.nan)
            wr5 = row.get("ret_5_winrate", np.nan)
            sh5 = row.get("ret_5_sharpe", np.nan)
            r10 = row.get("ret_10_mean", np.nan)
            wr10 = row.get("ret_10_winrate", np.nan)
            sh10 = row.get("ret_10_sharpe", np.nan)
            r20 = row.get("ret_20_mean", np.nan)
            wr20 = row.get("ret_20_winrate", np.nan)
            sh20 = row.get("ret_20_sharpe", np.nan)
            print(f"  {int(row['p_bin']):>4} {int(row['n']):>6} {p_range:>22} "
                  f"{r5*100:>7.2f} {wr5:>7.1f} {sh5:>7.3f} "
                  f"{r10*100:>7.2f} {wr10:>7.1f} {sh10:>7.3f} "
                  f"{r20*100:>7.2f} {wr20:>7.1f} {sh20:>7.3f}")

    # 4. 非线性回归检验
    print("\n[4/8] 非线性回归检验 ...")
    reg_df = compute_nonlinear_regression(df)
    reg_path = os.path.join(OUTPUT_DIR, "offset_mean_regression.csv")
    reg_df.to_csv(reg_path, index=False)
    print(f"  回归结果已保存: {reg_path}")

    print(f"\n  {'周期':>4} {'线性R²':>8} {'二次a':>8} {'二次p值':>10} {'F统计':>8} {'F p值':>10} {'倒U型':>6} {'顶点OM':>8}")
    print(f"  {'-'*80}")
    for _, row in reg_df.iterrows():
        inv_u = "✓" if row["is_inverted_u"] else "✗"
        print(f"  {int(row['horizon']):>4} {row['lin_r2']:>8.4f} {row['quad_a']:>8.4f} "
              f"{row['quad_a_pvalue']:>10.4f} {row['f_stat']:>8.2f} {row['f_pvalue']:>10.4f} "
              f"{inv_u:>6} {row['vertex_offset_mean']:>8.4f}")

    # 5. 样条回归
    print("\n[5/8] 样条回归 ...")
    spline_df = compute_spline_fit(df)
    if not spline_df.empty:
        spline_path = os.path.join(OUTPUT_DIR, "offset_mean_spline.csv")
        spline_df.to_csv(spline_path, index=False)
        print(f"  样条回归结果已保存: {spline_path}")
        for _, row in spline_df.iterrows():
            print(f"    ret_{int(row['horizon'])}: 峰值 offset_mean={row['spline_peak_offset_mean']:.4f}, "
                  f"峰值 ret={row['spline_peak_ret']*100:.2f}%")

    # 6. 甜点区间识别
    print("\n[6/8] 甜点区间识别 ...")
    sweet_df = identify_sweet_spot(df, reg_df)
    sweet_path = os.path.join(OUTPUT_DIR, "offset_mean_sweet_spot.csv")
    sweet_df.to_csv(sweet_path, index=False)
    print(f"  甜点区间已保存: {sweet_path}")

    for _, row in sweet_df.iterrows():
        print(f"    ret_{int(row['horizon'])}: 甜点 [{row['sweet_spot_low']:.4f}, {row['sweet_spot_high']:.4f}]")
        print(f"      区间内: n={int(row['in_spot_n'])}, 均值={row['in_spot_ret_mean']*100:.2f}%, "
              f"胜率={row['in_spot_winrate']:.1f}%")
        print(f"      区间外: n={int(row['out_spot_n'])}, 均值={row['out_spot_ret_mean']*100:.2f}%")
        print(f"      t检验: t={row['t_stat']:.3f}, p={row['t_pvalue']:.4f}")

    # 7. 分层分析 + 稳健性检验
    print("\n[7/8] 分层分析 + 稳健性检验 ...")
    strat_df = stratified_analysis(df)
    if not strat_df.empty:
        strat_path = os.path.join(OUTPUT_DIR, "offset_mean_stratified.csv")
        strat_df.to_csv(strat_path, index=False)
        print(f"  分层分析已保存: {strat_path}")
        for _, row in strat_df.iterrows():
            inv_u = "✓" if row["is_inverted_u"] else "✗"
            sig = "***" if row.get("quad_a_pvalue", 1) < 0.01 else ("**" if row.get("quad_a_pvalue", 1) < 0.05 else "")
            print(f"    {row['stratify_label']} [{row['stratum_min']:.3f},{row['stratum_max']:.3f}]: "
                  f"a={row['quad_a']:.4f}, p={row.get('quad_a_pvalue', np.nan):.4f} {sig} 倒U={inv_u}")

    yearly_df = yearly_robustness(df)
    if not yearly_df.empty:
        yearly_path = os.path.join(OUTPUT_DIR, "offset_mean_yearly_robustness.csv")
        yearly_df.to_csv(yearly_path, index=False)
        print(f"  分年度稳健性已保存: {yearly_path}")
        print(f"\n  {'年份':>6} {'N':>6} {'二次a':>8} {'p值':>10} {'倒U型':>6}")
        print(f"  {'-'*40}")
        for _, row in yearly_df.iterrows():
            inv_u = "✓" if row["is_inverted_u"] else "✗"
            print(f"  {int(row['year']):>6} {int(row['n_obs']):>6} {row['quad_a']:>8.4f} "
                  f"{row['quad_a_pvalue']:>10.4f} {inv_u:>6}")

    # 8. 可视化
    print("\n[8/8] 可视化 ...")
    font_name = _setup_chinese_font()
    if font_name:
        print(f"  使用中文字体: {font_name}")
    else:
        print("  未找到中文字体，图表标签将使用英文")
    plot_nonlinear_analysis(df, reg_df, OUTPUT_DIR)
    plot_by_year(df, OUTPUT_DIR)
    if not strat_df.empty:
        plot_stratified(strat_df, OUTPUT_DIR)

    # 总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)

    inverted_u_count = reg_df["is_inverted_u"].sum() if not reg_df.empty else 0
    total_horizons = len(reg_df) if not reg_df.empty else 0
    print(f"  倒 U 型成立周期数: {inverted_u_count}/{total_horizons}")

    if inverted_u_count > 0:
        print(f"\n  倒 U 型成立的周期:")
        for _, row in reg_df[reg_df["is_inverted_u"]].iterrows():
            print(f"    ret_{int(row['horizon'])}: 顶点 offset_mean={row['vertex_offset_mean']:.4f}, "
                  f"a={row['quad_a']:.4f}, p={row['quad_a_pvalue']:.4f}")

    if not yearly_df.empty:
        yearly_inv_u = yearly_df["is_inverted_u"].sum()
        yearly_total = len(yearly_df)
        print(f"\n  分年度倒 U 型: {yearly_inv_u}/{yearly_total} 个年度成立")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
