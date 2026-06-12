"""
Tick因子与次日实体涨跌幅相关性分析

Purpose: 基于tick_selection表的PVDI因子和tick汇总数据，系统性分析时序特征与次日实体涨跌幅的相关性
Inputs: tick_selection表, stock_k_data表
Outputs: 控制台报告 + CSV结果 + PNG可视化图表
How to Run:
    python tick_experiment/01_feature_correlation.py
    python tick_experiment/01_feature_correlation.py --output-dir tick_experiment/results
Examples:
    python tick_experiment/01_feature_correlation.py
    python tick_experiment/01_feature_correlation.py --output-dir /tmp/tick_results
Side Effects: 仅读取数据库，不写入任何表；输出CSV和PNG到指定目录
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_URL
from sqlalchemy import create_engine, text


# ── 中文字体配置 ──
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════════════
# 1. 数据提取
# ══════════════════════════════════════════════════════════════════

def load_tick_selection(engine) -> pd.DataFrame:
    """从tick_selection表加载全部数据"""
    sql = text("""
        SELECT selection_date, ts_code, stock_name,
               f_center, f_spread, f_skew, skew_b, skew_s, pvdi_weighted,
               pattern, label, signal, strength,
               buy_volume, sell_volume, buy_trades, sell_trades,
               buy_sell_volume_ratio, buy_sell_trades_ratio,
               dsa_vwap, high_dev_pct, low_dev_pct, close_dev_pct,
               dsa_dir, dsa_bars, regime,
               close_price, change_pct, avg_amount_20d
        FROM tick_selection
        ORDER BY ts_code, selection_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    return df


def load_next_day_ohlcv(engine, ts_codes: list, min_date: str, max_date: str) -> pd.DataFrame:
    """从stock_k_data表加载次日OHLCV数据

    需要加载从min_date到max_date+若干天的数据，以便获取最后一日的次日数据
    """
    # 多取5天确保覆盖节假日
    # 在Python中计算结束日期，避免SQL日期加法兼容问题
    from datetime import timedelta
    end_date_extended = (pd.Timestamp(max_date) + timedelta(days=10)).strftime("%Y-%m-%d")
    sql = text("""
        SELECT ts_code, bar_time::date AS trade_date, open, high, low, close, volume
        FROM stock_k_data
        WHERE freq = 'd'
          AND ts_code = ANY(:codes)
          AND bar_time::date >= :start_date
          AND bar_time::date <= :end_date
        ORDER BY ts_code, bar_time
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={
            "codes": ts_codes,
            "start_date": min_date,
            "end_date": end_date_extended,
        })
    return df


def merge_next_day_body_change(tick_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """计算次日实体涨跌幅并合并到tick数据

    实体涨跌幅 = (next_close - next_open) / next_open * 100
    """
    # 为ohlcv数据创建次日列
    ohlcv_df = ohlcv_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    ohlcv_df["next_trade_date"] = ohlcv_df.groupby("ts_code")["trade_date"].shift(-1)
    ohlcv_df["next_open"] = ohlcv_df.groupby("ts_code")["open"].shift(-1)
    ohlcv_df["next_close"] = ohlcv_df.groupby("ts_code")["close"].shift(-1)
    ohlcv_df["next_body_change_pct"] = (ohlcv_df["next_close"] - ohlcv_df["next_open"]) / ohlcv_df["next_open"] * 100

    # 只保留需要的列
    next_day = ohlcv_df[["ts_code", "trade_date", "next_trade_date", "next_open", "next_close", "next_body_change_pct"]].copy()
    next_day = next_day.rename(columns={"trade_date": "selection_date"})

    # 合并
    merged = tick_df.merge(next_day, on=["ts_code", "selection_date"], how="left")
    return merged


# ══════════════════════════════════════════════════════════════════
# 2. 特征工程
# ══════════════════════════════════════════════════════════════════

# 需要计算时序衍生特征的因子列
FACTOR_COLS = ["f_center", "f_spread", "f_skew", "pvdi_weighted",
               "buy_sell_volume_ratio", "buy_sell_trades_ratio",
               "high_dev_pct", "low_dev_pct", "close_dev_pct"]


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """按股票分组计算时序衍生特征"""
    df = df.sort_values(["ts_code", "selection_date"]).reset_index(drop=True)

    for col in FACTOR_COLS:
        g = df.groupby("ts_code")[col]
        # 1日变化量
        df[f"delta_{col}"] = g.diff(1)
        # 3日滚动均值
        df[f"ma3_{col}"] = g.transform(lambda x: x.rolling(3, min_periods=2).mean())
        # 3日滚动标准差
        df[f"std3_{col}"] = g.transform(lambda x: x.rolling(3, min_periods=2).std())

    # 买卖量比5日趋势
    g_ratio = df.groupby("ts_code")["buy_sell_volume_ratio"]
    df["ma5_buy_sell_ratio"] = g_ratio.transform(lambda x: x.rolling(5, min_periods=3).mean())

    return df


# ══════════════════════════════════════════════════════════════════
# 3. 相关性分析
# ══════════════════════════════════════════════════════════════════

def get_numeric_feature_cols(df: pd.DataFrame) -> list:
    """获取所有数值型特征列名（排除非特征列）"""
    exclude = {"selection_date", "ts_code", "stock_name", "next_trade_date",
               "next_open", "next_close", "next_body_change_pct",
               "pattern", "dsa_dir", "dsa_bars",
               "buy_volume", "sell_volume", "buy_trades", "sell_trades",
               "dsa_vwap", "close_price", "avg_amount_20d"}
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude]


def compute_correlations(df: pd.DataFrame, target: str = "next_body_change_pct") -> pd.DataFrame:
    """计算所有特征与目标变量的Pearson和Spearman相关系数"""
    feature_cols = get_numeric_feature_cols(df)
    results = []

    for col in feature_cols:
        valid = df[[col, target]].dropna()
        if len(valid) < 10:
            continue
        pearson_r, pearson_p = stats.pearsonr(valid[col], valid[target])
        spearman_r, spearman_p = stats.spearmanr(valid[col], valid[target])
        results.append({
            "feature": col,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "n_samples": len(valid),
        })

    result_df = pd.DataFrame(results)
    result_df["pearson_sig"] = result_df["pearson_p"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
    result_df["spearman_sig"] = result_df["spearman_p"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
    return result_df.sort_values("spearman_r", key=abs, ascending=False).reset_index(drop=True)


def compute_ic_by_date(df: pd.DataFrame, target: str = "next_body_change_pct") -> pd.DataFrame:
    """按日期计算各因子的IC（Spearman秩相关）"""
    feature_cols = [c for c in FACTOR_COLS if c in df.columns]
    dates = sorted(df["selection_date"].unique())
    results = []

    for date in dates:
        day_df = df[df["selection_date"] == date]
        for col in feature_cols:
            valid = day_df[[col, target]].dropna()
            if len(valid) < 5:
                continue
            ic, _ = stats.spearmanr(valid[col], valid[target])
            results.append({
                "date": date,
                "feature": col,
                "ic": ic,
                "n_stocks": len(valid),
            })

    return pd.DataFrame(results)


def compute_group_stats(df: pd.DataFrame, target: str = "next_body_change_pct") -> pd.DataFrame:
    """按signal/strength/pattern分组统计次日涨跌幅"""
    target_col = df[target]

    # 按signal分组
    signal_stats = df.groupby("signal").agg(
        count=(target, "count"),
        mean=(target, "mean"),
        median=(target, "median"),
        std=(target, "std"),
        win_rate=(target, lambda x: (x > 0).mean()),
    ).reset_index()
    signal_stats["group_type"] = "signal"

    # 按strength分组
    strength_stats = df.groupby("strength").agg(
        count=(target, "count"),
        mean=(target, "mean"),
        median=(target, "median"),
        std=(target, "std"),
        win_rate=(target, lambda x: (x > 0).mean()),
    ).reset_index().rename(columns={"strength": "group"})
    strength_stats["group_type"] = "strength"

    # 按pattern分组
    pattern_stats = df.groupby("pattern").agg(
        count=(target, "count"),
        mean=(target, "mean"),
        median=(target, "median"),
        std=(target, "std"),
        win_rate=(target, lambda x: (x > 0).mean()),
    ).reset_index().rename(columns={"pattern": "group"})
    pattern_stats["group_type"] = "pattern"

    # 统一列名
    signal_stats = signal_stats.rename(columns={"signal": "group"})

    return pd.concat([signal_stats, strength_stats, pattern_stats], ignore_index=True)


def compute_quantile_stats(df: pd.DataFrame, factor: str, target: str = "next_body_change_pct", n_bins: int = 5) -> pd.DataFrame:
    """按因子分位数分组统计次日涨跌幅"""
    valid = df[[factor, target]].dropna().copy()
    if len(valid) < n_bins * 5:
        return pd.DataFrame()

    valid["quantile"] = pd.qcut(valid[factor], n_bins, labels=False, duplicates="drop") + 1
    stats_df = valid.groupby("quantile").agg(
        count=(target, "count"),
        factor_mean=(factor, "mean"),
        target_mean=(target, "mean"),
        target_median=(target, "median"),
        win_rate=(target, lambda x: (x > 0).mean()),
    ).reset_index()
    return stats_df


# ══════════════════════════════════════════════════════════════════
# 4. 可视化
# ══════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(corr_df: pd.DataFrame, output_path: str):
    """绘制因子相关性热力图（Pearson + Spearman）"""
    top_n = min(20, len(corr_df))
    top_features = corr_df.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, top_n * 0.4)))

    for ax, col, title in [(axes[0], "pearson_r", "Pearson相关系数"),
                            (axes[1], "spearman_r", "Spearman秩相关")]:
        data = top_features.set_index("feature")[col].values.reshape(-1, 1)
        im = ax.imshow(data, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features["feature"].values, fontsize=9)
        ax.set_xticks([0])
        ax.set_xticklabels([title], fontsize=10)
        ax.set_title(title, fontsize=12)

        # 标注数值
        for i in range(top_n):
            val = data[i, 0]
            sig = top_features.iloc[i][f"{col.split('_')[0]}_sig"]
            ax.text(0, i, f"{val:.3f}{sig}", ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > 0.15 else "black")

        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Tick因子与次日实体涨跌幅相关性", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_signal_boxplot(df: pd.DataFrame, output_path: str):
    """绘制各signal分组次日涨跌幅分布箱线图"""
    order = ["extreme_bull", "bullish", "neutral", "bearish", "extreme_bear"]
    labels_zh = {"extreme_bull": "追涨狂热", "bullish": "看涨", "neutral": "中性",
                 "bearish": "看跌", "extreme_bear": "恐慌"}
    valid = df[df["signal"].isin(order)].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [valid[valid["signal"] == s]["next_body_change_pct"].dropna().values for s in order]
    data = [d for d in data if len(d) > 0]
    labels = [f"{labels_zh.get(s, s)}\n(n={len(d)})" for s, d in zip(order, data) if len(d) > 0]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.3))
    colors = ["#FFC107", "#4CAF50", "#9E9E9E", "#f44336", "#D32F2F"]
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("次日实体涨跌幅(%)", fontsize=11)
    ax.set_title("PVDI信号分组 vs 次日实体涨跌幅分布", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rolling_ic(ic_df: pd.DataFrame, output_path: str):
    """绘制滚动IC时序图"""
    if ic_df.empty:
        return
    features = ic_df["feature"].unique()
    n_features = len(features)

    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features), squeeze=False)
    for i, feat in enumerate(features):
        ax = axes[i, 0]
        feat_ic = ic_df[ic_df["feature"] == feat].sort_values("date")
        ax.bar(feat_ic["date"].astype(str), feat_ic["ic"], color=["#4CAF50" if v > 0 else "#f44336" for v in feat_ic["ic"]], alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.8)
        mean_ic = feat_ic["ic"].mean()
        ax.axhline(y=mean_ic, color="blue", linestyle="--", linewidth=1, label=f"均值IC={mean_ic:.3f}")
        ax.set_title(f"{feat} 滚动IC", fontsize=11)
        ax.set_ylabel("IC")
        ax.legend(fontsize=9)
        # 减少x轴标签密度
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    plt.suptitle("各因子滚动IC（按日期Spearman秩相关）", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_quantile_analysis(df: pd.DataFrame, factor: str, output_path: str, n_bins: int = 5):
    """绘制因子分位数组合收益对比"""
    q_stats = compute_quantile_stats(df, factor, n_bins=n_bins)
    if q_stats.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = q_stats["quantile"].values
    width = 0.35

    bars = ax1.bar(x - width / 2, q_stats["target_mean"], width, label="次日实体涨跌幅均值(%)", color="#2196F3", alpha=0.8)
    ax1.set_xlabel(f"{factor} 分位数组", fontsize=11)
    ax1.set_ylabel("次日实体涨跌幅均值(%)", fontsize=11, color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, q_stats["win_rate"] * 100, width, label="胜率(%)", color="#FF9800", alpha=0.8)
    ax2.set_ylabel("胜率(%)", fontsize=11, color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    # 标注数值
    for bar, val in zip(bars, q_stats["target_mean"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Q{int(q)}" for q in x])
    ax1.set_title(f"{factor} 分位数组 vs 次日实体涨跌幅", fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# 5. 报告输出
# ══════════════════════════════════════════════════════════════════

def print_report(corr_df: pd.DataFrame, group_df: pd.DataFrame, ic_df: pd.DataFrame, n_total: int):
    """打印分析报告到控制台"""
    print("=" * 80)
    print("  Tick因子与次日实体涨跌幅相关性分析报告")
    print("=" * 80)
    print(f"\n总样本数: {n_total}")

    # 1. 相关系数Top10
    print("\n─── 1. Spearman秩相关Top10（绝对值排序） ───")
    top10 = corr_df.head(10)
    for _, row in top10.iterrows():
        print(f"  {row['feature']:35s}  ρ={row['spearman_r']:+.4f} (p={row['spearman_p']:.4f}){row['spearman_sig']:3s}  "
              f"r={row['pearson_r']:+.4f} (p={row['pearson_p']:.4f}){row['pearson_sig']:3s}")

    # 2. 显著因子
    sig_factors = corr_df[corr_df["spearman_p"] < 0.05]
    print(f"\n─── 2. Spearman显著因子（p<0.05）: {len(sig_factors)}个 ───")
    for _, row in sig_factors.iterrows():
        print(f"  {row['feature']:35s}  ρ={row['spearman_r']:+.4f}  p={row['spearman_p']:.4f}")

    # 3. Signal分组统计
    print("\n─── 3. Signal分组统计 ───")
    signal_stats = group_df[group_df["group_type"] == "signal"]
    for _, row in signal_stats.iterrows():
        print(f"  {str(row['group']):16s}  n={int(row['count']):5d}  "
              f"均值={row['mean']:+.3f}%  中位数={row['median']:+.3f}%  "
              f"胜率={row['win_rate']:.1%}")

    # 4. Pattern分组统计
    print("\n─── 4. Pattern分组统计 ───")
    pattern_stats = group_df[group_df["group_type"] == "pattern"].sort_values("group")
    pattern_labels = {0: "中性", 1: "吸筹式上涨", 2: "谨慎推升", 3: "追涨狂热",
                      4: "虚涨派发", 5: "恐慌下跌", 6: "承接式下跌", 7: "多杀多", 8: "震荡出货"}
    for _, row in pattern_stats.iterrows():
        label = pattern_labels.get(int(row["group"]), str(int(row["group"])))
        print(f"  模式{int(row['group'])}({label:6s})  n={int(row['count']):5d}  "
              f"均值={row['mean']:+.3f}%  中位数={row['median']:+.3f}%  "
              f"胜率={row['win_rate']:.1%}")

    # 5. IC均值
    if not ic_df.empty:
        print("\n─── 5. 因子IC均值（Spearman秩相关） ───")
        ic_mean = ic_df.groupby("feature")["ic"].agg(["mean", "std", "count"]).reset_index()
        ic_mean["icir"] = ic_mean["mean"] / ic_mean["std"].replace(0, np.nan)
        ic_mean = ic_mean.sort_values("mean", key=abs, ascending=False)
        for _, row in ic_mean.iterrows():
            print(f"  {row['feature']:30s}  IC均值={row['mean']:+.4f}  IC_STD={row['std']:.4f}  "
                  f"ICIR={row['icir']:+.3f}  n_days={int(row['count'])}")

    print("\n" + "=" * 80)


# ══════════════════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Tick因子与次日实体涨跌幅相关性分析")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "tick_experiment" / "results"),
                        help="输出目录（默认 tick_experiment/results）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(DATABASE_URL)

    # 1. 加载数据
    print("加载tick_selection数据...")
    tick_df = load_tick_selection(engine)
    print(f"  tick_selection: {len(tick_df)} 条, {tick_df['ts_code'].nunique()} 只股票, "
          f"{tick_df['selection_date'].nunique()} 个交易日")

    if tick_df.empty:
        print("tick_selection表无数据，退出")
        return

    # 2. 加载次日OHLCV
    print("加载次日OHLCV数据...")
    ts_codes = tick_df["ts_code"].unique().tolist()
    min_date = tick_df["selection_date"].min()
    max_date = tick_df["selection_date"].max()
    ohlcv_df = load_next_day_ohlcv(engine, ts_codes, str(min_date), str(max_date))
    print(f"  stock_k_data: {len(ohlcv_df)} 条")

    # 3. 合并次日实体涨跌幅
    print("计算次日实体涨跌幅...")
    merged_df = merge_next_day_body_change(tick_df, ohlcv_df)
    valid_count = merged_df["next_body_change_pct"].notna().sum()
    print(f"  有效次日数据: {valid_count}/{len(merged_df)}")

    # 4. 特征工程
    print("计算时序衍生特征...")
    merged_df = add_time_series_features(merged_df)

    # 5. 相关性分析
    print("计算相关性...")
    corr_df = compute_correlations(merged_df)
    corr_df.to_csv(output_dir / "correlation_results.csv", index=False, encoding="utf-8-sig")

    # 6. IC分析
    print("计算滚动IC...")
    ic_df = compute_ic_by_date(merged_df)
    ic_df.to_csv(output_dir / "rolling_ic.csv", index=False, encoding="utf-8-sig")

    # 7. 分组统计
    print("计算分组统计...")
    group_df = compute_group_stats(merged_df)
    group_df.to_csv(output_dir / "group_stats.csv", index=False, encoding="utf-8-sig")

    # 8. 可视化
    print("生成可视化图表...")
    plot_correlation_heatmap(corr_df, output_dir / "01_correlation_heatmap.png")
    plot_signal_boxplot(merged_df, output_dir / "02_signal_boxplot.png")
    plot_rolling_ic(ic_df, output_dir / "03_rolling_ic.png")

    # 关键因子的分位数分析
    for factor in ["pvdi_weighted", "f_center", "f_skew", "buy_sell_volume_ratio"]:
        if factor in merged_df.columns:
            plot_quantile_analysis(merged_df, factor,
                                   output_dir / f"04_quantile_{factor}.png")

    # 9. 输出报告
    print_report(corr_df, group_df, ic_df, len(merged_df))

    # 10. 保存完整数据
    merged_df.to_csv(output_dir / "merged_data.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
