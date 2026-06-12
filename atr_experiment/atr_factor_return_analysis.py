#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ATR Rope 选股因子与未来收益率关系分析

Purpose: 探索 atr_rope_selection 表中各计算字段与未来 3/5/10/20 天收益率的关系
Inputs:  atr_rope_selection 表 + stock_k_data 表
Outputs: atr_experiment/output/ 下 Excel报告 + PNG图表 + CSV数据集

How to Run:
    # 小批量验证（1个月）
    python atr_experiment/atr_factor_return_analysis.py --start 2026-04-01 --end 2026-04-30

    # 全量运行
    python atr_experiment/atr_factor_return_analysis.py --start 2025-01-01 --end 2026-05-21

    # 仅数据准备（不跑分析）
    python atr_experiment/atr_factor_return_analysis.py --start 2026-04-01 --end 2026-04-30 --prepare-only

Examples:
    python atr_experiment/atr_factor_return_analysis.py --start 2026-04-01 --end 2026-04-30
    python atr_experiment/atr_factor_return_analysis.py --start 2025-01-01 --end 2026-05-21 --horizons 3 5 10 20

Side Effects: 只读数据库，写入 output/ 目录下的文件
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "atr_experiment" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 因子配置 ====================

# 天然可跨股比较的数值型因子
NUMERIC_FACTORS = [
    "regime_strength",
    "rope_dev_pct",
    "rope_dev_atr",
    "range_width_pct",
    "change_pct",
    "vol_zscore",
    "dsa_dir_bars",
    "rope_dir1_pct",
    "rope_dir_neg1_pct",
    "range_pos_01",
    "dsa_vwap_dev_pct",
    "avg_amount_20d",
]

# 衍生百分比因子（绝对价格差 / 基准 * 100）
DERIVED_PCT_FACTORS = {
    # (原字段, 基准字段) → 衍生字段名
    "low_rope_dev_mean": ("rope_value", "low_rope_dev_mean_pct"),
    "low_rope_dev_std": ("rope_value", "low_rope_dev_std_pct"),
    "low_rope_dev_today": ("rope_value", "low_rope_dev_today_pct"),
    "low_vwap_dev_mean": ("close", "low_vwap_dev_mean_pct"),
    "low_vwap_dev_std": ("close", "low_vwap_dev_std_pct"),
    "low_vwap_dev_today": ("close", "low_vwap_dev_today_pct"),
}

# 分类因子
CATEGORICAL_FACTORS = ["regime", "bbmacd_event", "low_rope_signal", "low_vwap_signal"]

# 未来收益期限
DEFAULT_HORIZONS = [3, 5, 10, 20]


# ==================== 数据准备 ====================

def load_selection_data(start_date: str, end_date: str) -> pd.DataFrame:
    """从 atr_rope_selection 表加载选股记录"""
    from sqlalchemy import text
    from datasource.database import get_engine

    engine = get_engine()
    sql = text("""
        SELECT * FROM atr_rope_selection
        WHERE selection_date >= :start AND selection_date <= :end
        ORDER BY selection_date, ts_code
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"start": start_date, "end": end_date})
    engine.dispose()
    return df


def compute_derived_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算衍生百分比因子"""
    df = df.copy()

    # low_rope_dev 系列 → 除以 rope_value
    for src_col, (base_col, dst_col) in DERIVED_PCT_FACTORS.items():
        if src_col in df.columns and base_col in df.columns:
            df[dst_col] = np.where(
                df[base_col].notna() & (df[base_col] != 0),
                df[src_col] / df[base_col] * 100,
                np.nan,
            )
        elif src_col in df.columns and base_col not in df.columns:
            # close 不在表中，用 rope_value * (1 + rope_dev_pct/100) 近似
            # 或从 rope_value + rope_dev_atr * atr 推导，但最简单的方式：
            # low_vwap_dev 的基准是 close，close ≈ rope_value * (1 + rope_dev_pct/100)
            if base_col == "close" and "rope_value" in df.columns and "rope_dev_pct" in df.columns:
                approx_close = df["rope_value"] * (1 + df["rope_dev_pct"] / 100)
                df[dst_col] = np.where(
                    approx_close.notna() & (approx_close != 0),
                    df[src_col] / approx_close * 100,
                    np.nan,
                )

    # avg_amount_20d → log 变换
    if "avg_amount_20d" in df.columns:
        df["log_avg_amount_20d"] = np.log1p(df["avg_amount_20d"])

    # rope_dev_pct 统一转为百分比形式（核心模块输出小数，如0.02=2%）
    if "rope_dev_pct" in df.columns:
        sample_max = df["rope_dev_pct"].dropna().abs().max()
        if sample_max < 1.0:
            df["rope_dev_pct"] = df["rope_dev_pct"] * 100

    return df


def compute_future_metrics_corrected(price_pivot: dict, horizons: list) -> dict:
    """以 open[T+1] 为入场价计算未来收益，消除前视偏差。

    与 filter_quality_evaluator.compute_future_metrics() 的区别：
    - 入场价：open[T+1]（而非 close[T]），因为信号在 T 日收盘后确认，实际只能次日开盘买入
    - MFE/MAE 窗口：从 T+1 开始（而非 T），因为 T 日的 high/low 在信号确认时已知
    - 返回格式与 compute_future_metrics() 一致：DataFrame(index=日期, columns=6位股票代码)
    """
    open_df = price_pivot["open"]
    close_df = price_pivot["close"]
    high_df = price_pivot["high"]
    low_df = price_pivot["low"]

    metrics = {}
    for N in horizons:
        # entry_price = open[T+1]，即信号日次日的开盘价
        entry = open_df.shift(-1)

        # return_N = close[T+N] / open[T+1] - 1
        ret = close_df.shift(-N) / entry - 1

        # MFE_N = max(high[T+1:T+N+1]) / open[T+1] - 1
        # MAE_N = min(low[T+1:T+N+1]) / open[T+1] - 1
        # 窗口从 T+1 开始，不含 T 日
        # 使用 rolling 方式计算，保持 DataFrame 格式
        mfe = (high_df.shift(-1).rolling(N, min_periods=1).max().shift(-(N - 1))) / entry - 1
        mae = (low_df.shift(-1).rolling(N, min_periods=1).min().shift(-(N - 1))) / entry - 1

        metrics[N] = {"return": ret, "mfe": mfe, "mae": mae}

    return metrics


def load_price_data_and_compute_returns(start_date: str, end_date: str, horizons: list) -> dict:
    """加载价格数据并计算未来收益率（修正版：以 open[T+1] 为入场价）"""
    from stop_experiment.backtest.simple_backtest import load_daily_prices, build_price_pivot

    # 价格数据需要多加载 max(horizons) 天，确保有足够的未来数据
    price_start = start_date
    price_end = pd.Timestamp(end_date) + pd.Timedelta(days=max(horizons) + 10)

    print(f"加载价格数据: {price_start} ~ {price_end.strftime('%Y-%m-%d')}")
    daily_prices = load_daily_prices(price_start, price_end.strftime("%Y-%m-%d"))
    price_pivot, _, _ = build_price_pivot(daily_prices)

    print(f"计算未来收益率 (horizons={horizons}, 入场价=open[T+1])...")
    metrics = compute_future_metrics_corrected(price_pivot, horizons=horizons)
    return metrics


def merge_selection_with_returns(df: pd.DataFrame, future_metrics: dict, horizons: list) -> pd.DataFrame:
    """将选股记录与未来收益率合并"""
    from stop_experiment.eval.filter_quality_evaluator import _lookup_values

    df = df.copy()
    df["raw_code"] = df["ts_code"].str[:6]
    df["lookup_date"] = pd.to_datetime(df["selection_date"]).dt.normalize()

    for N in horizons:
        for metric_name in ["return", "mfe", "mae"]:
            col = f"{metric_name}_{N}"
            df[col] = _lookup_values(
                future_metrics[N][metric_name],
                df["lookup_date"].values,
                df["raw_code"].values,
            )

    df.drop(columns=["lookup_date"], inplace=True)
    return df


# ==================== Layer 1: Spearman 秩相关 ====================

def layer1_spearman(df: pd.DataFrame, horizons: list) -> pd.DataFrame:
    """计算每个数值型因子与未来收益率的 Spearman 相关系数"""
    all_numeric_factors = NUMERIC_FACTORS + [dst for _, dst in DERIVED_PCT_FACTORS.values()] + ["log_avg_amount_20d"]
    available_factors = [f for f in all_numeric_factors if f in df.columns]
    return_cols = [f"return_{h}" for h in horizons]
    available_return_cols = [c for c in return_cols if c in df.columns]

    rows = []
    for factor in available_factors:
        for ret_col in available_return_cols:
            valid = df[[factor, ret_col]].dropna()
            if len(valid) < 30:
                continue
            corr, pval = stats.spearmanr(valid[factor], valid[ret_col])
            rows.append({
                "factor": factor,
                "return_col": ret_col,
                "spearman_corr": corr,
                "p_value": pval,
                "n_samples": len(valid),
                "significant_005": pval < 0.05,
            })

    result_df = pd.DataFrame(rows)

    # 按信号类型分组
    group_rows = []
    if "signal_type" in df.columns:
        for st in df["signal_type"].unique():
            sub = df[df["signal_type"] == st]
            for factor in available_factors:
                for ret_col in available_return_cols:
                    valid = sub[[factor, ret_col]].dropna()
                    if len(valid) < 30:
                        continue
                    corr, pval = stats.spearmanr(valid[factor], valid[ret_col])
                    group_rows.append({
                        "signal_type": st,
                        "factor": factor,
                        "return_col": ret_col,
                        "spearman_corr": corr,
                        "p_value": pval,
                        "n_samples": len(valid),
                    })

    group_df = pd.DataFrame(group_rows)
    return result_df, group_df


# ==================== Layer 2: 分桶均值分析 ====================

def layer2_bucket_means(df: pd.DataFrame, horizons: list, n_buckets: int = 5) -> pd.DataFrame:
    """将每个因子按分位分桶，计算每桶的平均未来收益率"""
    all_numeric_factors = NUMERIC_FACTORS + [dst for _, dst in DERIVED_PCT_FACTORS.values()] + ["log_avg_amount_20d"]
    available_factors = [f for f in all_numeric_factors if f in df.columns]
    return_cols = [f"return_{h}" for h in horizons]
    available_return_cols = [c for c in return_cols if c in df.columns]

    rows = []
    for factor in available_factors:
        for ret_col in available_return_cols:
            sub = df[[factor, ret_col]].dropna().copy()
            if len(sub) < n_buckets * 10:
                continue
            try:
                sub["bucket"] = pd.qcut(sub[factor], n_buckets, labels=False, duplicates="drop")
            except ValueError:
                continue
            for b in sorted(sub["bucket"].unique()):
                bucket_data = sub[sub["bucket"] == b]
                rows.append({
                    "factor": factor,
                    "return_col": ret_col,
                    "bucket": int(b),
                    "bucket_min": bucket_data[factor].min(),
                    "bucket_max": bucket_data[factor].max(),
                    "mean_return": bucket_data[ret_col].mean(),
                    "median_return": bucket_data[ret_col].median(),
                    "win_rate": (bucket_data[ret_col] > 0).mean(),
                    "n_samples": len(bucket_data),
                })

    return pd.DataFrame(rows)


# ==================== Layer 3: 偏相关分析 ====================

def layer3_partial_corr(df: pd.DataFrame, horizons: list, top_n: int = 10) -> pd.DataFrame:
    """对显著因子计算偏相关（控制 change_pct 和 avg_amount_20d）"""
    try:
        import pingouin as pg
    except ImportError:
        print("警告: pingouin 未安装，跳过偏相关分析。安装: pip install pingouin")
        return pd.DataFrame()

    # 从 Layer 1 结果中选取 top_n 最显著因子
    all_numeric_factors = NUMERIC_FACTORS + [dst for _, dst in DERIVED_PCT_FACTORS.values()] + ["log_avg_amount_20d"]
    available_factors = [f for f in all_numeric_factors if f in df.columns]
    return_cols = [f"return_{h}" for h in horizons]
    available_return_cols = [c for c in return_cols if c in df.columns]

    # 先快速计算全样本 Spearman 排序
    factor_corr = {}
    for factor in available_factors:
        for ret_col in available_return_cols:
            valid = df[[factor, ret_col]].dropna()
            if len(valid) < 30:
                continue
            corr, _ = stats.spearmanr(valid[factor], valid[ret_col])
            factor_corr[(factor, ret_col)] = abs(corr)

    # 取 top_n
    sorted_pairs = sorted(factor_corr.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_factors = list(set([p[0][0] for p in sorted_pairs]))

    control_vars = ["change_pct", "log_avg_amount_20d"]
    available_controls = [c for c in control_vars if c in df.columns]

    rows = []
    for factor in top_factors:
        for ret_col in available_return_cols:
            cols_needed = [factor, ret_col] + available_controls
            sub = df[cols_needed].dropna()
            if len(sub) < 50:
                continue
            try:
                result = pg.partial_corr(
                    data=sub, x=factor, y=ret_col,
                    covar=available_controls, method="spearman"
                )
                rows.append({
                    "factor": factor,
                    "return_col": ret_col,
                    "partial_corr": result["r"].values[0],
                    "p_value": result["p-val"].values[0],
                    "n_samples": len(sub),
                    "controls": ", ".join(available_controls),
                })
            except Exception:
                continue

    return pd.DataFrame(rows)


# ==================== Layer 4: 分市场状态分析 ====================

def layer4_by_regime(df: pd.DataFrame, horizons: list) -> pd.DataFrame:
    """按 regime 分组计算 Spearman 相关"""
    if "regime" not in df.columns:
        return pd.DataFrame()

    all_numeric_factors = NUMERIC_FACTORS + [dst for _, dst in DERIVED_PCT_FACTORS.values()] + ["log_avg_amount_20d"]
    available_factors = [f for f in all_numeric_factors if f in df.columns]
    return_cols = [f"return_{h}" for h in horizons]
    available_return_cols = [c for c in return_cols if c in df.columns]

    rows = []
    for regime_val in df["regime"].unique():
        sub = df[df["regime"] == regime_val]
        for factor in available_factors:
            for ret_col in available_return_cols:
                valid = sub[[factor, ret_col]].dropna()
                if len(valid) < 30:
                    continue
                corr, pval = stats.spearmanr(valid[factor], valid[ret_col])
                rows.append({
                    "regime": regime_val,
                    "factor": factor,
                    "return_col": ret_col,
                    "spearman_corr": corr,
                    "p_value": pval,
                    "n_samples": len(valid),
                })

    return pd.DataFrame(rows)


# ==================== Layer 5: IC 时序稳定性 ====================

def layer5_ic_stability(df: pd.DataFrame, horizons: list) -> pd.DataFrame:
    """按月计算 IC，评估时序稳定性"""
    all_numeric_factors = NUMERIC_FACTORS + [dst for _, dst in DERIVED_PCT_FACTORS.values()] + ["log_avg_amount_20d"]
    available_factors = [f for f in all_numeric_factors if f in df.columns]
    return_cols = [f"return_{h}" for h in horizons]
    available_return_cols = [c for c in return_cols if c in df.columns]

    df = df.copy()
    df["month"] = pd.to_datetime(df["selection_date"]).dt.to_period("M")

    rows = []
    for factor in available_factors:
        for ret_col in available_return_cols:
            ic_list = []
            for month, sub in df.groupby("month"):
                valid = sub[[factor, ret_col]].dropna()
                if len(valid) < 20:
                    continue
                corr, _ = stats.spearmanr(valid[factor], valid[ret_col])
                ic_list.append({"month": str(month), "ic": corr, "n": len(valid)})

            if len(ic_list) < 3:
                continue

            ic_series = pd.Series([x["ic"] for x in ic_list])
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            n_months = len(ic_series)
            t_stat = ic_mean / ic_std * np.sqrt(n_months) if ic_std > 0 else 0

            rows.append({
                "factor": factor,
                "return_col": ret_col,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "t_stat": t_stat,
                "n_months": n_months,
                "is_stable": abs(icir) > 0.5 and abs(t_stat) > 2,
                "monthly_ics": ic_list,
            })

    return pd.DataFrame(rows)


# ==================== 图表生成 ====================

def plot_spearman_heatmap(corr_df: pd.DataFrame, output_path: Path):
    """绘制因子×收益期限 Spearman 相关系数热力图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if corr_df.empty:
        return

    pivot = corr_df.pivot_table(index="factor", columns="return_col", values="spearman_corr")
    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # 标注数值
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > 0.15 else "black")

    plt.colorbar(im, ax=ax, label="Spearman Correlation")
    ax.set_title("ATR因子 × 未来收益率 Spearman相关")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bucket_returns(bucket_df: pd.DataFrame, output_dir: Path):
    """绘制各因子分桶收益柱状图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if bucket_df.empty:
        return

    for ret_col in bucket_df["return_col"].unique():
        sub = bucket_df[bucket_df["return_col"] == ret_col]
        factors = sub["factor"].unique()

        n_factors = len(factors)
        n_cols = 3
        n_rows = (n_factors + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        if n_rows == 1:
            axes = np.array([axes]) if n_cols == 1 else axes
        axes = axes.flatten()

        for idx, factor in enumerate(factors):
            ax = axes[idx]
            fdata = sub[sub["factor"] == factor].sort_values("bucket")
            if fdata.empty:
                continue
            bars = ax.bar(fdata["bucket"], fdata["mean_return"] * 100, color="steelblue", alpha=0.8)
            ax.set_title(factor, fontsize=9)
            ax.set_xlabel("Bucket")
            ax.set_ylabel("Mean Return (%)")
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        for idx in range(len(factors), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"分桶均值 - {ret_col}", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f"02_bucket_returns_{ret_col}.png", dpi=150)
        plt.close()


def plot_ic_timeseries(ic_df: pd.DataFrame, output_dir: Path):
    """绘制 IC 时序图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if ic_df.empty:
        return

    # 只画稳定因子和接近稳定的因子
    plot_df = ic_df[ic_df["icir"].abs() > 0.2].copy()
    if plot_df.empty:
        plot_df = ic_df.head(10)

    for _, row in plot_df.iterrows():
        monthly = row["monthly_ics"]
        months = [x["month"] for x in monthly]
        ics = [x["ic"] for x in monthly]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(months, ics, color=["steelblue" if v > 0 else "salmon" for v in ics], alpha=0.8)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(f"IC时序 - {row['factor']} vs {row['return_col']} (ICIR={row['icir']:.2f})")
        ax.set_ylabel("Spearman IC")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.tight_layout()

        safe_name = f"{row['factor']}_{row['return_col']}".replace(" ", "_")
        plt.savefig(output_dir / f"05_ic_timeseries_{safe_name}.png", dpi=150)
        plt.close()


# ==================== Excel 报告 ====================

def generate_excel_report(
    corr_df, group_corr_df, bucket_df, partial_df, regime_df, ic_df,
    output_path: Path
):
    """生成 Excel 报告"""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: 因子概览
        all_factors = NUMERIC_FACTORS + [dst for _, dst in DERIVED_PCT_FACTORS.values()] + ["log_avg_amount_20d"]
        overview = pd.DataFrame({
            "factor": all_factors,
            "type": ["原字段"] * len(NUMERIC_FACTORS) +
                    ["衍生百分比"] * len(DERIVED_PCT_FACTORS) +
                    ["衍生log"],
            "cross_stock_comparable": [True] * len(NUMERIC_FACTORS) +
                                      [True] * len(DERIVED_PCT_FACTORS) +
                                      [True],
        })
        overview.to_excel(writer, sheet_name="因子概览", index=False)

        # Sheet 2: Spearman 相关
        if not corr_df.empty:
            corr_df.to_excel(writer, sheet_name="Spearman相关", index=False)

        # Sheet 2b: 分信号类型相关
        if not group_corr_df.empty:
            group_corr_df.to_excel(writer, sheet_name="分信号类型相关", index=False)

        # Sheet 3: 分桶收益
        if not bucket_df.empty:
            bucket_df.to_excel(writer, sheet_name="分桶收益", index=False)

        # Sheet 4: 偏相关
        if not partial_df.empty:
            partial_df.to_excel(writer, sheet_name="偏相关", index=False)

        # Sheet 5: 分市场状态
        if not regime_df.empty:
            regime_df.to_excel(writer, sheet_name="分市场状态", index=False)

        # Sheet 6: IC 稳定性
        if not ic_df.empty:
            ic_summary = ic_df.drop(columns=["monthly_ics"]).copy()
            ic_summary.to_excel(writer, sheet_name="IC稳定性", index=False)


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="ATR Rope 选股因子与未来收益率关系分析")
    parser.add_argument("--start", required=True, help="数据起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="数据结束日期 (YYYY-MM-DD)")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
                        help="未来收益期限 (默认: 3 5 10 20)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="仅准备数据集，不跑分析")
    args = parser.parse_args()

    print("=" * 60)
    print("ATR Rope 选股因子与未来收益率关系分析")
    print(f"数据范围: {args.start} ~ {args.end}")
    print(f"收益期限: {args.horizons}")
    print("=" * 60)

    # Step 1: 加载选股记录
    print("\n[Step 1] 加载选股记录...")
    df = load_selection_data(args.start, args.end)
    print(f"  选股记录: {len(df)} 条, 日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")
    if "signal_type" in df.columns:
        print(f"  信号类型分布: {df['signal_type'].value_counts().to_dict()}")

    # Step 2: 计算衍生因子
    print("\n[Step 2] 计算衍生百分比因子...")
    df = compute_derived_factors(df)
    derived_cols = [dst for _, dst in DERIVED_PCT_FACTORS.values()] + ["log_avg_amount_20d"]
    for col in derived_cols:
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, non-null={df[col].notna().sum()}")

    # Step 3: 计算未来收益率
    print("\n[Step 3] 计算未来收益率...")
    future_metrics = load_price_data_and_compute_returns(args.start, args.end, args.horizons)

    # Step 4: 合并
    print("\n[Step 4] 合并选股记录与未来收益...")
    df = merge_selection_with_returns(df, future_metrics, args.horizons)
    for h in args.horizons:
        ret_col = f"return_{h}"
        if ret_col in df.columns:
            valid = df[ret_col].notna().sum()
            print(f"  {ret_col}: {valid} 条有效 ({valid/len(df)*100:.1f}%)")

    # 保存数据集
    csv_path = OUTPUT_DIR / "atr_factor_return_dataset.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  数据集已保存: {csv_path}")

    if args.prepare_only:
        print("\n--prepare-only 模式，跳过分析。")
        return

    # Layer 1: Spearman 相关
    print("\n[Layer 1] Spearman 秩相关分析...")
    corr_df, group_corr_df = layer1_spearman(df, args.horizons)
    print(f"  全样本: {len(corr_df)} 条相关记录")
    if not corr_df.empty:
        top_corr = corr_df.reindex(corr_df["spearman_corr"].abs().sort_values(ascending=False).index).head(10)
        print("  Top 10 相关:")
        for _, row in top_corr.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            print(f"    {row['factor']} vs {row['return_col']}: r={row['spearman_corr']:.4f} p={row['p_value']:.4f} {sig}")

    # Layer 2: 分桶均值
    print("\n[Layer 2] 分桶均值分析...")
    bucket_df = layer2_bucket_means(df, args.horizons)
    print(f"  {len(bucket_df)} 条分桶记录")

    # Layer 3: 偏相关
    print("\n[Layer 3] 偏相关分析...")
    partial_df = layer3_partial_corr(df, args.horizons)
    print(f"  {len(partial_df)} 条偏相关记录")

    # Layer 4: 分市场状态
    print("\n[Layer 4] 分市场状态分析...")
    regime_df = layer4_by_regime(df, args.horizons)
    print(f"  {len(regime_df)} 条分组记录")

    # Layer 5: IC 稳定性
    print("\n[Layer 5] IC 时序稳定性分析...")
    ic_df = layer5_ic_stability(df, args.horizons)
    print(f"  {len(ic_df)} 条IC记录")
    if not ic_df.empty:
        stable = ic_df[ic_df["is_stable"]]
        print(f"  稳定因子: {len(stable)} 个")
        for _, row in stable.iterrows():
            print(f"    {row['factor']} vs {row['return_col']}: ICIR={row['icir']:.3f}, t={row['t_stat']:.2f}")

    # 生成图表
    print("\n[图表生成]")
    plot_spearman_heatmap(corr_df, OUTPUT_DIR / "01_spearman_heatmap.png")
    print("  01_spearman_heatmap.png")
    plot_bucket_returns(bucket_df, OUTPUT_DIR)
    print("  02_bucket_returns_*.png")
    plot_ic_timeseries(ic_df, OUTPUT_DIR)
    print("  05_ic_timeseries_*.png")

    # 生成 Excel 报告
    print("\n[Excel 报告]")
    xlsx_path = OUTPUT_DIR / "atr_factor_return_report.xlsx"
    generate_excel_report(corr_df, group_corr_df, bucket_df, partial_df, regime_df, ic_df, xlsx_path)
    print(f"  {xlsx_path}")

    # 汇总
    print("\n" + "=" * 60)
    print("分析完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    if not ic_df.empty:
        stable_factors = ic_df[ic_df["is_stable"]]["factor"].unique().tolist()
        print(f"稳定因子: {stable_factors if stable_factors else '无'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
