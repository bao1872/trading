"""
PVDI信号多空回测

Purpose: 基于PVDI信号(bullish/bearish/neutral)构建简单多空组合，验证信号对次日实体涨跌幅的预测能力
Inputs: tick_selection表, stock_k_data表
Outputs: 控制台回测报告 + CSV结果 + PNG净值曲线
How to Run:
    python tick_experiment/02_signal_backtest.py
    python tick_experiment/02_signal_backtest.py --output-dir tick_experiment/results
Examples:
    python tick_experiment/02_signal_backtest.py
    python tick_experiment/02_signal_backtest.py --output-dir /tmp/tick_results
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_URL
from sqlalchemy import create_engine, text

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_backtest_data(engine) -> pd.DataFrame:
    """加载tick_selection + 次日OHLCV数据"""
    # tick_selection
    tick_sql = text("""
        SELECT selection_date, ts_code, stock_name,
               f_center, f_spread, f_skew, pvdi_weighted,
               pattern, label, signal, strength,
               buy_sell_volume_ratio, close_dev_pct, change_pct
        FROM tick_selection
        ORDER BY ts_code, selection_date
    """)
    with engine.connect() as conn:
        tick_df = pd.read_sql(tick_sql, conn)

    if tick_df.empty:
        return tick_df

    # 次日OHLCV
    ts_codes = tick_df["ts_code"].unique().tolist()
    min_date = str(tick_df["selection_date"].min())
    max_date = str(tick_df["selection_date"].max())

    from datetime import timedelta
    end_date_extended = (pd.Timestamp(max_date) + timedelta(days=10)).strftime("%Y-%m-%d")
    ohlcv_sql = text("""
        SELECT ts_code, bar_time::date AS trade_date, open, close
        FROM stock_k_data
        WHERE freq = 'd'
          AND ts_code = ANY(:codes)
          AND bar_time::date >= :start_date
          AND bar_time::date <= :end_date
        ORDER BY ts_code, bar_time
    """)
    with engine.connect() as conn:
        ohlcv_df = pd.read_sql(ohlcv_sql, conn, params={
            "codes": ts_codes, "start_date": min_date, "end_date": end_date_extended,
        })

    # 计算次日实体涨跌幅
    ohlcv_df = ohlcv_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    ohlcv_df["next_open"] = ohlcv_df.groupby("ts_code")["open"].shift(-1)
    ohlcv_df["next_close"] = ohlcv_df.groupby("ts_code")["close"].shift(-1)
    ohlcv_df["next_body_change_pct"] = (ohlcv_df["next_close"] - ohlcv_df["next_open"]) / ohlcv_df["next_open"] * 100

    next_day = ohlcv_df[["ts_code", "trade_date", "next_body_change_pct"]].copy()
    next_day = next_day.rename(columns={"trade_date": "selection_date"})

    merged = tick_df.merge(next_day, on=["ts_code", "selection_date"], how="left")
    return merged


def run_signal_backtest(df: pd.DataFrame) -> dict:
    """基于PVDI信号的多空回测

    策略：
    - bullish做多：次日实体涨跌幅
    - bearish做空：-次日实体涨跌幅
    - extreme_bull做空（反转）：-次日实体涨跌幅
    - extreme_bear做多（反转）：次日实体涨跌幅
    - neutral空仓：0
    """
    valid = df.dropna(subset=["next_body_change_pct"]).copy()

    # 策略1：简单多空
    def simple_position(signal):
        if signal == "bullish":
            return 1
        elif signal == "bearish":
            return -1
        else:
            return 0

    # 策略2：含极端反转
    def reversal_position(signal):
        if signal == "bullish":
            return 1
        elif signal == "bearish":
            return -1
        elif signal == "extreme_bull":
            return -1  # 过热反转
        elif signal == "extreme_bear":
            return 1   # 恐慌反转
        else:
            return 0

    # 策略3：仅bullish
    def long_only_position(signal):
        return 1 if signal == "bullish" else 0

    # 策略4：仅bearish做空
    def short_only_position(signal):
        return -1 if signal == "bearish" else 0

    strategies = {
        "简单多空(bullish多/bearish空)": simple_position,
        "含极端反转": reversal_position,
        "仅bullish做多": long_only_position,
        "仅bearish做空": short_only_position,
    }

    results = {}
    for name, pos_func in strategies.items():
        valid["position"] = valid["signal"].apply(pos_func)
        valid["strategy_return"] = valid["position"] * valid["next_body_change_pct"]

        # 按日期汇总
        daily = valid.groupby("selection_date").agg(
            n_stocks=("strategy_return", "count"),
            n_active=("position", lambda x: (x != 0).sum()),
            strategy_return=("strategy_return", "mean"),
        ).reset_index()
        daily = daily.sort_values("selection_date")
        daily["cum_return"] = daily["strategy_return"].cumsum()

        # 统计指标
        total_return = daily["strategy_return"].sum()
        mean_daily = daily["strategy_return"].mean()
        std_daily = daily["strategy_return"].std()
        sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0
        win_rate = (daily["strategy_return"] > 0).mean()

        # 最大回撤
        cum = daily["cum_return"].values
        running_max = np.maximum.accumulate(cum)
        drawdown = cum - running_max
        max_drawdown = drawdown.min()

        results[name] = {
            "daily_df": daily,
            "total_return": total_return,
            "mean_daily": mean_daily,
            "std_daily": std_daily,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "n_days": len(daily),
        }

    return results


def run_strength_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """按signal+strength交叉分组统计"""
    valid = df.dropna(subset=["next_body_change_pct"]).copy()
    cross = valid.groupby(["signal", "strength"]).agg(
        count=("next_body_change_pct", "count"),
        mean_return=("next_body_change_pct", "mean"),
        median_return=("next_body_change_pct", "median"),
        win_rate=("next_body_change_pct", lambda x: (x > 0).mean()),
    ).reset_index()
    return cross


def run_pattern_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """按pattern分组统计"""
    valid = df.dropna(subset=["next_body_change_pct"]).copy()
    pattern = valid.groupby("pattern").agg(
        count=("next_body_change_pct", "count"),
        mean_return=("next_body_change_pct", "mean"),
        median_return=("next_body_change_pct", "median"),
        win_rate=("next_body_change_pct", lambda x: (x > 0).mean()),
    ).reset_index()
    return pattern


def plot_equity_curves(results: dict, output_path: str):
    """绘制各策略累计收益曲线"""
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#f44336"]

    for (name, data), color in zip(results.items(), colors):
        daily = data["daily_df"]
        ax.plot(daily["selection_date"], daily["cum_return"],
                label=f"{name} (总收益={data['total_return']:+.2f}%, Sharpe={data['sharpe']:.2f})",
                color=color, linewidth=1.5)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("累计收益(%)", fontsize=11)
    ax.set_title("PVDI信号多空策略累计收益曲线（基于次日实体涨跌幅）", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_strength_heatmap(cross_df: pd.DataFrame, output_path: str):
    """绘制signal×strength分组收益热力图"""
    pivot = cross_df.pivot(index="signal", columns="strength", values="mean_return")
    # 排列顺序
    signal_order = ["extreme_bull", "bullish", "neutral", "bearish", "extreme_bear"]
    strength_order = ["weak", "medium", "strong"]
    pivot = pivot.reindex(index=[s for s in signal_order if s in pivot.index],
                          columns=[s for s in strength_order if s in pivot.columns])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # 标注数值
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                n = cross_df[(cross_df["signal"] == pivot.index[i]) &
                             (cross_df["strength"] == pivot.columns[j])]["count"].values[0]
                ax.text(j, i, f"{val:+.2f}%\n(n={n})", ha="center", va="center", fontsize=9)

    ax.set_title("Signal × Strength 分组次日实体涨跌幅均值(%)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_backtest_report(results: dict, cross_df: pd.DataFrame, pattern_df: pd.DataFrame):
    """打印回测报告"""
    print("=" * 80)
    print("  PVDI信号多空回测报告")
    print("=" * 80)

    # 策略汇总
    print("\n─── 策略汇总 ───")
    for name, data in results.items():
        print(f"\n  {name}:")
        print(f"    总收益: {data['total_return']:+.2f}%")
        print(f"    日均收益: {data['mean_daily']:+.4f}%")
        print(f"    日收益标准差: {data['std_daily']:.4f}%")
        print(f"    年化Sharpe: {data['sharpe']:.3f}")
        print(f"    日胜率: {data['win_rate']:.1%}")
        print(f"    最大回撤: {data['max_drawdown']:+.2f}%")
        print(f"    交易天数: {data['n_days']}")

    # Signal × Strength
    print("\n─── Signal × Strength 分组统计 ───")
    for _, row in cross_df.iterrows():
        print(f"  {row['signal']:16s} × {row['strength']:8s}  n={int(row['count']):5d}  "
              f"均值={row['mean_return']:+.3f}%  中位数={row['median_return']:+.3f}%  "
              f"胜率={row['win_rate']:.1%}")

    # Pattern
    print("\n─── Pattern 分组统计 ───")
    pattern_labels = {0: "中性", 1: "吸筹式上涨", 2: "谨慎推升", 3: "追涨狂热",
                      4: "虚涨派发", 5: "恐慌下跌", 6: "承接式下跌", 7: "多杀多", 8: "震荡出货"}
    for _, row in pattern_df.iterrows():
        label = pattern_labels.get(int(row["pattern"]), str(int(row["pattern"])))
        print(f"  模式{int(row['pattern'])}({label:6s})  n={int(row['count']):5d}  "
              f"均值={row['mean_return']:+.3f}%  中位数={row['median_return']:+.3f}%  "
              f"胜率={row['win_rate']:.1%}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="PVDI信号多空回测")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "tick_experiment" / "results"),
                        help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(DATABASE_URL)

    print("加载数据...")
    df = load_backtest_data(engine)
    print(f"  总记录: {len(df)}, 有效次日数据: {df['next_body_change_pct'].notna().sum()}")

    if df.empty or df["next_body_change_pct"].notna().sum() < 10:
        print("数据不足，退出")
        return

    # 策略回测
    print("执行策略回测...")
    results = run_signal_backtest(df)

    # 分组统计
    print("计算分组统计...")
    cross_df = run_strength_backtest(df)
    pattern_df = run_pattern_backtest(df)

    # 可视化
    print("生成图表...")
    plot_equity_curves(results, output_dir / "05_equity_curves.png")
    plot_strength_heatmap(cross_df, output_dir / "06_signal_strength_heatmap.png")

    # 报告
    print_backtest_report(results, cross_df, pattern_df)

    # 保存
    for name, data in results.items():
        safe_name = name.replace("(", "").replace(")", "").replace("/", "_")[:30]
        data["daily_df"].to_csv(output_dir / f"backtest_{safe_name}.csv", index=False, encoding="utf-8-sig")
    cross_df.to_csv(output_dir / "signal_strength_stats.csv", index=False, encoding="utf-8-sig")
    pattern_df.to_csv(output_dir / "pattern_stats.csv", index=False, encoding="utf-8-sig")

    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
