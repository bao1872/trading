"""
F_spread截面分位数验证实验

Purpose: 验证假设——当F_spread处于截面20%以下分位时，次日价格涨幅是否有显著差异
Inputs: tick_cache表(全市场), stock_k_data表(日线OHLCV)
Outputs: 控制台报告 + CSV统计结果 + PNG可视化图表
How to Run:
    python tick_experiment/04_f_spread_quantile.py
    python tick_experiment/04_f_spread_quantile.py --output-dir tick_experiment/results
Examples:
    python tick_experiment/04_f_spread_quantile.py
    python tick_experiment/04_f_spread_quantile.py --output-dir /tmp/f_spread_results
Side Effects: 仅读取数据库，不写入任何表；输出CSV和PNG到指定目录

F_spread公式（SSOT: datasource/pytdx_client.py L333）:
    f_spread = (sigma_s - sigma_b) / (sigma_s + sigma_b)
    sigma_b/sigma_s: 买入/卖出侧成交量加权价格标准差
    范围[-1,1]，正=卖方更分散，负=买方更分散

截面分位数定义:
    每个交易日，对所有股票的F_spread排序，等频分为Q1-Q5五组
    Q1(0-20%)即"20%以下水分位"组
"""

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_URL

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════════════════════

def load_tick_cache(engine) -> pd.DataFrame:
    """从tick_cache加载全市场数据（含f_spread）"""
    sql = text("""
        SELECT trade_date, ts_code,
               f_spread, f_center, f_skew, pvdi_weighted,
               buy_volume, sell_volume, buy_trades, sell_trades,
               pattern, label, signal, strength
        FROM tick_cache
        WHERE f_spread IS NOT NULL
        ORDER BY ts_code, trade_date
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def load_price_data(engine, ts_codes: list, min_date: str, max_date: str) -> pd.DataFrame:
    """加载日线OHLCV数据"""
    end_extended = (pd.Timestamp(max_date) + timedelta(days=15)).strftime("%Y-%m-%d")
    sql = text("""
        SELECT ts_code, bar_time::date AS trade_date,
               open, high, low, close, volume
        FROM stock_k_data
        WHERE freq = 'd'
          AND ts_code = ANY(:codes)
          AND bar_time::date >= :start_date
          AND bar_time::date <= :end_date
        ORDER BY ts_code, bar_time
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={
            "codes": ts_codes, "start_date": min_date, "end_date": end_extended,
        })


def merge_next_day_returns(tick_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """合并次日涨跌幅数据

    计算两个口径:
    1. next_change_pct: 次日涨跌幅 = (next_close - close) / close * 100
    2. next_body_change_pct: 次日实体涨跌幅 = (next_close - next_open) / next_open * 100
    """
    price_df = price_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    price_df["next_open"] = price_df.groupby("ts_code")["open"].shift(-1)
    price_df["next_close"] = price_df.groupby("ts_code")["close"].shift(-1)
    price_df["next_change_pct"] = (price_df["next_close"] - price_df["close"]) / price_df["close"] * 100
    price_df["next_body_change_pct"] = (price_df["next_close"] - price_df["next_open"]) / price_df["next_open"] * 100

    # 当日涨跌幅（用于排除涨跌停）
    price_df["change_pct"] = (price_df["close"] - price_df.groupby("ts_code")["close"].shift(1)) / price_df.groupby("ts_code")["close"].shift(1) * 100

    next_day = price_df[["ts_code", "trade_date", "close", "volume",
                          "change_pct", "next_change_pct", "next_body_change_pct"]].copy()
    next_day = next_day.rename(columns={"trade_date": "tick_date"})

    merged = tick_df.merge(next_day, left_on=["ts_code", "trade_date"],
                           right_on=["ts_code", "tick_date"], how="left")
    return merged


# ══════════════════════════════════════════════════════════════════
# 2. 截面分位数计算
# ══════════════════════════════════════════════════════════════════

def compute_cross_sectional_quantile(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """计算截面分位数分组

    每个交易日对所有股票的f_spread排序，等频分为n_bins组
    Q1=最低20%, Q5=最高20%
    """
    df = df.copy()
    quantile_labels = []

    for date, day_df in df.groupby("trade_date"):
        try:
            q = pd.qcut(day_df["f_spread"], n_bins, labels=False, duplicates="drop") + 1
        except ValueError:
            q = pd.cut(day_df["f_spread"].rank(pct=True) * n_bins,
                        bins=n_bins, labels=False, include_lowest=True) + 1
        quantile_labels.extend(q.tolist())

    df["quantile"] = quantile_labels
    return df


def filter_extreme_records(df: pd.DataFrame) -> pd.DataFrame:
    """排除涨跌停和停牌记录"""
    # 排除当日涨跌停（|change_pct| > 9.5%）
    mask = df["change_pct"].isna() | (df["change_pct"].abs() <= 9.5)
    # 排除停牌（volume=0 或 NaN）
    mask = mask & (df["volume"].fillna(0) > 0)
    return df[mask].copy()


# ══════════════════════════════════════════════════════════════════
# 3. 统计验证
# ══════════════════════════════════════════════════════════════════

def compute_quantile_stats(df: pd.DataFrame, target: str = "next_change_pct") -> pd.DataFrame:
    """按分位组统计次日涨幅"""
    results = []
    for q in sorted(df["quantile"].unique()):
        q_df = df[df["quantile"] == q]
        valid = q_df[target].dropna()
        if len(valid) < 5:
            continue
        results.append({
            "quantile": f"Q{int(q)}",
            "count": len(valid),
            "mean": valid.mean(),
            "median": valid.median(),
            "std": valid.std(),
            "min": valid.min(),
            "max": valid.max(),
            "win_rate": (valid > 0).mean(),
            "f_spread_mean": q_df["f_spread"].mean(),
        })
    return pd.DataFrame(results)


def compute_significance_tests(df: pd.DataFrame, target: str = "next_change_pct") -> dict:
    """统计显著性检验"""
    q1 = df[df["quantile"] == 1][target].dropna()
    q5 = df[df["quantile"] == 5][target].dropna()
    all_valid = df[target].dropna()

    results = {}

    # 1. t检验: Q1 vs Q5
    if len(q1) > 10 and len(q5) > 10:
        t_stat, t_pval = stats.ttest_ind(q1, q5, equal_var=False)
        results["ttest_Q1_vs_Q5"] = {"statistic": t_stat, "p_value": t_pval,
                                      "q1_mean": q1.mean(), "q5_mean": q5.mean(),
                                      "diff": q1.mean() - q5.mean()}

    # 2. Mann-Whitney U检验: Q1 vs Q5
    if len(q1) > 10 and len(q5) > 10:
        u_stat, u_pval = stats.mannwhitneyu(q1, q5, alternative="two-sided")
        results["mannwhitney_Q1_vs_Q5"] = {"statistic": u_stat, "p_value": u_pval}

    # 3. KS检验: Q1 vs 全体
    if len(q1) > 10 and len(all_valid) > 10:
        ks_stat, ks_pval = stats.ks_2samp(q1, all_valid)
        results["ks_Q1_vs_all"] = {"statistic": ks_stat, "p_value": ks_pval}

    # 4. t检验: Q1 vs Q2-Q5合并
    q2to5 = df[df["quantile"].isin([2, 3, 4, 5])][target].dropna()
    if len(q1) > 10 and len(q2to5) > 10:
        t_stat2, t_pval2 = stats.ttest_ind(q1, q2to5, equal_var=False)
        results["ttest_Q1_vs_Q2to5"] = {"statistic": t_stat2, "p_value": t_pval2,
                                         "q1_mean": q1.mean(), "q2to5_mean": q2to5.mean()}

    # 5. Q1内部: 正收益概率的二项检验
    if len(q1) > 10:
        n_pos = (q1 > 0).sum()
        n_total = len(q1)
        binom_pval = stats.binomtest(n_pos, n_total, 0.5).pvalue
        results["binomtest_Q1_winrate"] = {"statistic": n_pos,
                                            "win_rate": n_pos / n_total,
                                            "p_value": binom_pval,
                                            "n_positive": n_pos, "n_total": n_total}

    return results


def compute_daily_excess_returns(df: pd.DataFrame, target: str = "next_change_pct") -> pd.DataFrame:
    """计算每日Q1组相对全市场的超额收益"""
    daily_stats = []
    for date, day_df in df.groupby("trade_date"):
        q1_return = day_df[day_df["quantile"] == 1][target].mean()
        all_return = day_df[target].mean()
        n_q1 = (day_df["quantile"] == 1).sum()
        n_all = len(day_df)
        daily_stats.append({
            "date": date,
            "q1_return": q1_return,
            "all_return": all_return,
            "excess_return": q1_return - all_return if not np.isnan(q1_return) else np.nan,
            "n_q1": n_q1,
            "n_all": n_all,
        })
    return pd.DataFrame(daily_stats)


# ══════════════════════════════════════════════════════════════════
# 4. 可视化
# ══════════════════════════════════════════════════════════════════

def plot_f_spread_distribution(df: pd.DataFrame, output_path: str):
    """F_spread分布直方图 + 分位线标注"""
    fig, ax = plt.subplots(figsize=(12, 6))

    data = df["f_spread"].dropna()
    ax.hist(data, bins=80, color="#2196F3", alpha=0.7, edgecolor="white")

    # 分位线
    for q, color, label in [(0.2, "#f44336", "P20"), (0.5, "#FF9800", "P50"), (0.8, "#4CAF50", "P80")]:
        val = data.quantile(q)
        ax.axvline(x=val, color=color, linewidth=2, linestyle="--", label=f"{label}={val:.3f}")

    ax.set_xlabel("F_spread", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("F_spread Distribution with Quantile Lines", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_quantile_boxplot(df: pd.DataFrame, target: str, output_path: str):
    """5个分位组次日涨幅箱线图"""
    fig, ax = plt.subplots(figsize=(12, 7))

    data = []
    labels = []
    for q in sorted(df["quantile"].unique()):
        q_data = df[df["quantile"] == q][target].dropna()
        if len(q_data) > 5:
            data.append(q_data.values)
            labels.append(f"Q{int(q)}\n(n={len(q_data)})")

    if not data:
        return

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2, alpha=0.3))

    colors = ["#f44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")

    target_label = "Next Day Return (%)" if "body" not in target else "Next Day Body Return (%)"
    ax.set_ylabel(target_label, fontsize=11)
    ax.set_xlabel("F_spread Quantile Group", fontsize=11)
    ax.set_title(f"F_spread Quantile Groups vs {target_label}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_daily_excess_returns(daily_df: pd.DataFrame, output_path: str):
    """Q1组 vs 全市场每日超额收益时序图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    dates = daily_df["date"].astype(str)

    # 上图: Q1收益 vs 全市场收益
    ax1.plot(dates, daily_df["q1_return"], color="#f44336", linewidth=1.5, label="Q1 (Low F_spread)", marker="o", markersize=4)
    ax1.plot(dates, daily_df["all_return"], color="#2196F3", linewidth=1.5, label="All Stocks", marker="s", markersize=4)
    ax1.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Next Day Return (%)", fontsize=11)
    ax1.set_title("Q1 (Low F_spread) vs All Stocks - Daily Next Day Return", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.tick_params(axis="x", rotation=45, labelsize=8)

    # 下图: 超额收益
    colors = ["#4CAF50" if v > 0 else "#f44336" for v in daily_df["excess_return"].fillna(0)]
    ax2.bar(dates, daily_df["excess_return"], color=colors, alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    mean_excess = daily_df["excess_return"].mean()
    ax2.axhline(y=mean_excess, color="blue", linewidth=1, linestyle="--",
                label=f"Mean Excess={mean_excess:+.3f}%")
    ax2.set_ylabel("Excess Return (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_quantile_return_bar(stats_df: pd.DataFrame, output_path: str):
    """分位组-次日涨幅关系柱状图（均值+胜率双轴）"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = range(len(stats_df))
    width = 0.35

    bars = ax1.bar([i - width / 2 for i in x], stats_df["mean"], width,
                   color="#2196F3", alpha=0.8, label="Mean Return (%)")
    ax1.set_xlabel("F_spread Quantile Group", fontsize=11)
    ax1.set_ylabel("Mean Next Day Return (%)", fontsize=11, color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    # 标注数值
    for bar, val in zip(bars, stats_df["mean"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:+.3f}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.bar([i + width / 2 for i in x], stats_df["win_rate"] * 100, width,
            color="#FF9800", alpha=0.8, label="Win Rate (%)")
    ax2.set_ylabel("Win Rate (%)", fontsize=11, color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")
    ax2.axhline(y=50, color="gray", linewidth=0.8, linestyle="--")

    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df["quantile"])
    ax1.set_title("F_spread Quantile: Mean Return & Win Rate", fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# 5. 报告输出
# ══════════════════════════════════════════════════════════════════

def print_report(stats_df: pd.DataFrame, sig_tests: dict, daily_df: pd.DataFrame,
                 n_total: int, n_after_filter: int):
    """打印验证报告"""
    print("=" * 80)
    print("  F_spread Cross-Sectional Quantile Verification Report")
    print("=" * 80)

    print(f"\nTotal tick_cache records: {n_total}")
    print(f"After filtering (limit/stop): {n_after_filter}")

    # 分位组统计
    print("\n─── Quantile Group Statistics (Next Day Return) ───")
    print(f"{'Group':<8} {'Count':>7} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'WinRate':>8} {'F_spread':>10}")
    for _, row in stats_df.iterrows():
        print(f"{row['quantile']:<8} {int(row['count']):>7} {row['mean']:>+10.4f}% {row['median']:>+10.4f}% "
              f"{row['std']:>10.4f}% {row['min']:>+10.4f}% {row['max']:>+10.4f}% "
              f"{row['win_rate']:>7.1%} {row['f_spread_mean']:>+10.4f}")

    # 显著性检验
    print("\n─── Statistical Significance Tests ───")
    for test_name, result in sig_tests.items():
        sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else ""
        print(f"  {test_name}:")
        print(f"    statistic={result['statistic']:.4f}, p_value={result['p_value']:.4f} {sig}")
        if "diff" in result:
            print(f"    diff={result['diff']:+.4f}%")
        if "q1_mean" in result:
            print(f"    Q1_mean={result['q1_mean']:+.4f}%, other_mean={result.get('q5_mean', result.get('q2to5_mean', 'N/A'))}")

    # 每日超额收益
    if not daily_df.empty:
        mean_excess = daily_df["excess_return"].mean()
        std_excess = daily_df["excess_return"].std()
        win_days = (daily_df["excess_return"] > 0).sum()
        total_days = len(daily_df)
        print(f"\n─── Daily Excess Return (Q1 vs All) ───")
        print(f"  Mean excess: {mean_excess:+.4f}%")
        print(f"  Std excess:  {std_excess:.4f}%")
        print(f"  Days Q1 > All: {win_days}/{total_days} ({win_days/total_days:.1%})")

    # 结论
    print("\n─── Conclusion ───")
    q1_stats = stats_df[stats_df["quantile"] == "Q1"]
    q5_stats = stats_df[stats_df["quantile"] == "Q5"]

    if not q1_stats.empty and not q5_stats.empty:
        q1_mean = q1_stats["mean"].values[0]
        q5_mean = q5_stats["mean"].values[0]
        q1_wr = q1_stats["win_rate"].values[0]
        q5_wr = q5_stats["win_rate"].values[0]

        ttest = sig_tests.get("ttest_Q1_vs_Q5", {})
        p_val = ttest.get("p_value", 1.0)

        if p_val < 0.05:
            direction = "higher" if q1_mean > q5_mean else "lower"
            print(f"  F_spread Q1 (low) group has SIGNIFICANTLY {direction} next-day returns than Q5 (p={p_val:.4f})")
        else:
            print(f"  No significant difference between Q1 and Q5 next-day returns (p={p_val:.4f})")

        print(f"  Q1: mean={q1_mean:+.4f}%, win_rate={q1_wr:.1%}")
        print(f"  Q5: mean={q5_mean:+.4f}%, win_rate={q5_wr:.1%}")

    print("\n" + "=" * 80)


# ══════════════════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="F_spread Cross-Sectional Quantile Verification")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "tick_experiment" / "results"),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(DATABASE_URL)

    # 1. 加载数据
    print("Loading tick_cache data...")
    tick_df = load_tick_cache(engine)
    n_total = len(tick_df)
    print(f"  {n_total} records, {tick_df['ts_code'].nunique()} stocks, "
          f"{tick_df['trade_date'].nunique()} trading days")

    if tick_df.empty:
        print("No data, exiting")
        return

    print("Loading price data...")
    ts_codes = tick_df["ts_code"].unique().tolist()
    min_date = str(tick_df["trade_date"].min())
    max_date = str(tick_df["trade_date"].max())
    price_df = load_price_data(engine, ts_codes, min_date, max_date)
    print(f"  {len(price_df)} price records")

    # 2. 合并次日涨跌幅
    print("Merging next-day returns...")
    merged_df = merge_next_day_returns(tick_df, price_df)
    valid_count = merged_df["next_change_pct"].notna().sum()
    print(f"  Valid next-day data: {valid_count}/{len(merged_df)}")

    # 3. 排除涨跌停/停牌
    print("Filtering extreme records...")
    filtered_df = filter_extreme_records(merged_df)
    n_after_filter = len(filtered_df)
    print(f"  After filter: {n_after_filter} (removed {len(merged_df) - n_after_filter})")

    # 4. 截面分位数
    print("Computing cross-sectional quantiles...")
    filtered_df = compute_cross_sectional_quantile(filtered_df, n_bins=5)
    print(f"  Quantile distribution:")
    for q in sorted(filtered_df["quantile"].unique()):
        n = (filtered_df["quantile"] == q).sum()
        print(f"    Q{int(q)}: {n}")

    # 5. 统计验证
    print("Computing quantile statistics...")
    stats_change = compute_quantile_stats(filtered_df, "next_change_pct")
    stats_body = compute_quantile_stats(filtered_df, "next_body_change_pct")

    print("Running significance tests...")
    sig_tests = compute_significance_tests(filtered_df, "next_change_pct")

    # 6. 每日超额收益
    print("Computing daily excess returns...")
    daily_df = compute_daily_excess_returns(filtered_df, "next_change_pct")

    # 7. 输出报告
    print_report(stats_change, sig_tests, daily_df, n_total, n_after_filter)

    # 8. 实体涨跌幅统计
    print("\n─── Body Change (next_open -> next_close) ───")
    for _, row in stats_body.iterrows():
        print(f"  {row['quantile']:<8} n={int(row['count']):>5} mean={row['mean']:+.4f}% win_rate={row['win_rate']:.1%}")

    # 9. 可视化
    print("Generating charts...")
    plot_f_spread_distribution(filtered_df, output_dir / "09_f_spread_distribution.png")
    plot_quantile_boxplot(filtered_df, "next_change_pct", output_dir / "10_quantile_boxplot.png")
    plot_daily_excess_returns(daily_df, output_dir / "11_daily_excess_returns.png")
    plot_quantile_return_bar(stats_change, output_dir / "12_quantile_return_bar.png")

    # 10. 保存CSV
    stats_change.to_csv(output_dir / "f_spread_quantile_stats.csv", index=False, encoding="utf-8-sig")
    stats_body.to_csv(output_dir / "f_spread_quantile_body_stats.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(output_dir / "f_spread_daily_excess.csv", index=False, encoding="utf-8-sig")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
