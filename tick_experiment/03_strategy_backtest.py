"""
Tick谨慎推升策略回测

Purpose: 基于tick实验结果设计的多条件过滤策略回测，核心思路是少做少错、多重过滤
Inputs: tick_selection表, stock_k_data表
Outputs: 控制台回测报告 + CSV交易明细 + PNG可视化图表
How to Run:
    python tick_experiment/03_strategy_backtest.py
    python tick_experiment/03_strategy_backtest.py --output-dir tick_experiment/results
    python tick_experiment/03_strategy_backtest.py --hold-days 5
Examples:
    python tick_experiment/03_strategy_backtest.py
    python tick_experiment/03_strategy_backtest.py --hold-days 5 --max-holdings 3
Side Effects: 仅读取数据库，不写入任何表；输出CSV和PNG到指定目录
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
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_URL

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 交易成本 ──
BUY_COST = 0.0005   # 万五佣金
SELL_COST = 0.001    # 千一印花税+佣金

# ── 策略参数默认值 ──
DEFAULT_HOLD_DAYS = 3
DEFAULT_MAX_HOLDINGS = 5
DEFAULT_STOP_LOSS = -0.05   # -5%
DEFAULT_TAKE_PROFIT = 0.08  # +8%


# ══════════════════════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════════════════════

def load_tick_selection(engine) -> pd.DataFrame:
    """加载tick_selection全部数据"""
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
        return pd.read_sql(sql, conn)


def load_price_data(engine, ts_codes: list, min_date: str, max_date: str) -> pd.DataFrame:
    """加载日线价格数据（用于持有期收益计算和止损止盈）"""
    end_extended = (pd.Timestamp(max_date) + timedelta(days=15)).strftime("%Y-%m-%d")
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
        return pd.read_sql(sql, conn, params={
            "codes": ts_codes, "start_date": min_date, "end_date": end_extended,
        })


# ══════════════════════════════════════════════════════════════════
# 2. 信号生成
# ══════════════════════════════════════════════════════════════════

def generate_signals(tick_df: pd.DataFrame) -> pd.DataFrame:
    """应用5层入场过滤 + 3层排除条件，生成买入候选

    入场条件（全部满足）：
      L1: dsa_dir=1 AND dsa_bars>50（已在tick_selection中过滤）
      L2: pattern IN (1, 2)
      L3: f_center > 0 AND pvdi_weighted > 0
      L4: close_dev_pct > -2.0
      L5: avg_amount_20d > 1.0

    排除条件（任一满足则排除）：
      - signal = 'extreme_bull'
      - strength = 'medium'
      - skew_b < -0.5
    """
    df = tick_df.copy()

    # 入场过滤
    mask = (
        (df["pattern"].isin([1, 2])) &           # L2: 谨慎推升或吸筹式上涨
        (df["f_center"] > 0) &                    # L3: 买方均价更高
        (df["pvdi_weighted"] > 0) &               # L3: 综合因子为正
        (df["close_dev_pct"] > -2.0) &            # L4: 未严重跌破VWAP
        (df["avg_amount_20d"] > 1.0)              # L5: 流动性充足
    )

    # 排除条件
    exclude = (
        (df["signal"] == "extreme_bull") |         # 追涨狂热
        (df["strength"] == "medium") |             # medium强度最差
        (df["skew_b"] < -0.5)                      # 买方严重左偏
    )

    df["is_candidate"] = mask & ~exclude
    return df


# ══════════════════════════════════════════════════════════════════
# 3. 回测引擎
# ══════════════════════════════════════════════════════════════════

def build_price_lookup(price_df: pd.DataFrame) -> dict:
    """构建价格查找字典: (ts_code, date_str) -> {open, close, high, low}"""
    lookup = {}
    for _, row in price_df.iterrows():
        key = (row["ts_code"], str(row["trade_date"]))
        lookup[key] = {
            "open": row["open"],
            "close": row["close"],
            "high": row["high"],
            "low": row["low"],
        }
    return lookup


def get_next_trade_date(trade_dates: list, current_date: str) -> str | None:
    """获取下一个交易日"""
    for d in trade_dates:
        if d > current_date:
            return d
    return None


def run_backtest(
    tick_df: pd.DataFrame,
    price_df: pd.DataFrame,
    hold_days: int = DEFAULT_HOLD_DAYS,
    max_holdings: int = DEFAULT_MAX_HOLDINGS,
    stop_loss: float = DEFAULT_STOP_LOSS,
    take_profit: float = DEFAULT_TAKE_PROFIT,
) -> dict:
    """执行策略回测

    返回:
        {
            "daily_returns": DataFrame,  # 日度收益
            "trades": DataFrame,          # 交易明细
            "stats": dict,                # 统计指标
        }
    """
    price_lookup = build_price_lookup(price_df)
    all_dates = sorted(tick_df["selection_date"].unique())
    trade_dates = sorted(price_df["trade_date"].unique())
    trade_date_strs = [str(d) for d in trade_dates]

    # 生成信号
    signal_df = generate_signals(tick_df)

    # 持仓列表: [{ts_code, buy_date, buy_price, hold_days_left}]
    holdings = []
    trades = []
    daily_returns = []

    for date in all_dates:
        date_str = str(date)
        next_date_str = get_next_trade_date(trade_date_strs, date_str)

        # ── 1. 检查现有持仓的止损/止盈 ──
        holdings_to_sell = []
        holdings_to_keep = []

        for h in holdings:
            price_key = (h["ts_code"], date_str)
            if price_key not in price_lookup:
                holdings_to_keep.append(h)
                continue

            current_close = price_lookup[price_key]["close"]
            pnl_pct = (current_close - h["buy_price"]) / h["buy_price"]

            # 止损/止盈/到期
            should_sell = False
            sell_reason = ""
            if pnl_pct <= stop_loss:
                should_sell = True
                sell_reason = "stop_loss"
            elif pnl_pct >= take_profit:
                should_sell = True
                sell_reason = "take_profit"
            elif h["hold_days_left"] <= 1:
                should_sell = True
                sell_reason = "expire"

            if should_sell:
                # 次日开盘卖出
                sell_price_key = (h["ts_code"], next_date_str) if next_date_str else None
                if sell_price_key and sell_price_key in price_lookup:
                    sell_price = price_lookup[sell_price_key]["open"]
                    net_return = (sell_price - h["buy_price"]) / h["buy_price"] - BUY_COST - SELL_COST
                    holdings_to_sell.append({
                        "ts_code": h["ts_code"],
                        "buy_date": h["buy_date"],
                        "buy_price": h["buy_price"],
                        "sell_date": next_date_str,
                        "sell_price": sell_price,
                        "net_return": net_return,
                        "sell_reason": sell_reason,
                        "hold_days": hold_days - h["hold_days_left"] + 1,
                    })
                else:
                    # 无法卖出，继续持有
                    h["hold_days_left"] -= 1
                    holdings_to_keep.append(h)
            else:
                h["hold_days_left"] -= 1
                holdings_to_keep.append(h)

        # 记录交易
        trades.extend(holdings_to_sell)
        holdings = holdings_to_keep

        # ── 2. 计算当日持仓收益（基于收盘价） ──
        day_pnl = 0.0
        n_holdings = len(holdings)
        if n_holdings > 0:
            for h in holdings:
                price_key = (h["ts_code"], date_str)
                if price_key in price_lookup:
                    current_close = price_lookup[price_key]["close"]
                    pnl = (current_close - h["buy_price"]) / h["buy_price"] - BUY_COST
                    day_pnl += pnl / max_holdings
                # 如果停牌则不计收益

        daily_returns.append({
            "date": date_str,
            "n_holdings": n_holdings,
            "daily_return": day_pnl,
        })

        # ── 3. 生成新买入信号 ──
        if next_date_str is None:
            continue

        current_holdings_codes = {h["ts_code"] for h in holdings}
        n_available_slots = max_holdings - len(holdings)

        if n_available_slots <= 0:
            continue

        # 当日信号
        day_signals = signal_df[
            (signal_df["selection_date"] == date) & (signal_df["is_candidate"])
        ].copy()

        # 排除已持仓
        day_signals = day_signals[~day_signals["ts_code"].isin(current_holdings_codes)]

        # 按pvdi_weighted降序排列
        day_signals = day_signals.sort_values("pvdi_weighted", ascending=False)

        # 取Top N
        new_buys = day_signals.head(n_available_slots)

        for _, row in new_buys.iterrows():
            buy_price_key = (row["ts_code"], next_date_str)
            if buy_price_key in price_lookup:
                buy_price = price_lookup[buy_price_key]["open"]
                holdings.append({
                    "ts_code": row["ts_code"],
                    "buy_date": next_date_str,
                    "buy_price": buy_price,
                    "hold_days_left": hold_days,
                })

    # ── 处理剩余持仓（回测结束时按最后日收盘价清仓） ──
    last_date = trade_date_strs[-1] if trade_date_strs else None
    for h in holdings:
        price_key = (h["ts_code"], last_date)
        if price_key and price_key in price_lookup:
            sell_price = price_lookup[price_key]["close"]
            net_return = (sell_price - h["buy_price"]) / h["buy_price"] - BUY_COST - SELL_COST
            trades.append({
                "ts_code": h["ts_code"],
                "buy_date": h["buy_date"],
                "buy_price": h["buy_price"],
                "sell_date": last_date,
                "sell_price": sell_price,
                "net_return": net_return,
                "sell_reason": "end_of_backtest",
                "hold_days": hold_days - h["hold_days_left"],
            })

    # ── 统计 ──
    daily_df = pd.DataFrame(daily_returns)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["ts_code", "buy_date", "buy_price", "sell_date", "sell_price", "net_return", "sell_reason", "hold_days"]
    )

    stats = compute_stats(daily_df, trades_df)

    return {
        "daily_returns": daily_df,
        "trades": trades_df,
        "stats": stats,
    }


def compute_stats(daily_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    """计算回测统计指标"""
    if daily_df.empty:
        return {}

    total_return = daily_df["daily_return"].sum()
    mean_daily = daily_df["daily_return"].mean()
    std_daily = daily_df["daily_return"].std()
    sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0

    # 累计收益曲线
    cum = daily_df["daily_return"].cumsum()
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    max_drawdown = drawdown.min()

    # 交易统计
    n_trades = len(trades_df)
    if n_trades > 0:
        win_trades = (trades_df["net_return"] > 0).sum()
        win_rate = win_trades / n_trades
        avg_return = trades_df["net_return"].mean()
        avg_win = trades_df[trades_df["net_return"] > 0]["net_return"].mean() if win_trades > 0 else 0
        avg_loss = trades_df[trades_df["net_return"] <= 0]["net_return"].mean() if n_trades - win_trades > 0 else 0
        profit_factor = abs(avg_win * win_trades / (avg_loss * (n_trades - win_trades))) if n_trades - win_trades > 0 and avg_loss != 0 else float("inf")

        # 按出场原因统计
        reason_stats = trades_df.groupby("sell_reason")["net_return"].agg(["count", "mean"]).to_dict("index")
    else:
        win_rate = avg_return = avg_win = avg_loss = profit_factor = 0
        reason_stats = {}

    return {
        "total_return": total_return,
        "mean_daily_return": mean_daily,
        "std_daily_return": std_daily,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "n_trading_days": len(daily_df),
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_return_per_trade": avg_return,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "reason_stats": reason_stats,
        "avg_holdings_per_day": daily_df["n_holdings"].mean(),
    }


# ══════════════════════════════════════════════════════════════════
# 4. 可视化
# ══════════════════════════════════════════════════════════════════

def plot_equity_curve(daily_df: pd.DataFrame, output_path: str):
    """绘制累计收益曲线"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    cum_return = daily_df["daily_return"].cumsum()

    ax1.plot(daily_df["date"], cum_return, color="#2196F3", linewidth=1.5, label="Strategy")
    ax1.fill_between(daily_df["date"], 0, cum_return, alpha=0.15, color="#2196F3")
    ax1.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax1.set_title("Tick Cautious Advance Strategy - Equity Curve", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.tick_params(axis="x", rotation=45, labelsize=8)

    # 持仓数量
    ax2.bar(daily_df["date"], daily_df["n_holdings"], color="#4CAF50", alpha=0.7)
    ax2.set_ylabel("Holdings", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_trade_distribution(trades_df: pd.DataFrame, output_path: str):
    """绘制交易收益分布"""
    if trades_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. 收益分布直方图
    ax = axes[0]
    ax.hist(trades_df["net_return"] * 100, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Return (%)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Trade Return Distribution", fontsize=11)

    # 2. 按出场原因分组
    ax = axes[1]
    reasons = trades_df.groupby("sell_reason")["net_return"].agg(["count", "mean"])
    colors = {"expire": "#2196F3", "stop_loss": "#f44336", "take_profit": "#4CAF50", "end_of_backtest": "#9E9E9E"}
    bar_colors = [colors.get(r, "#9E9E9E") for r in reasons.index]
    ax.barh(reasons.index, reasons["mean"] * 100, color=bar_colors, alpha=0.8)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Avg Return (%)", fontsize=10)
    ax.set_title("Return by Exit Reason", fontsize=11)
    for i, (idx, row) in enumerate(reasons.iterrows()):
        ax.text(row["mean"] * 100, i, f" n={int(row['count'])}", va="center", fontsize=9)

    # 3. 持仓天数分布
    ax = axes[2]
    ax.hist(trades_df["hold_days"], bins=range(1, trades_df["hold_days"].max() + 2),
            color="#FF9800", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Hold Days", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Hold Days Distribution", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# 5. 报告输出
# ══════════════════════════════════════════════════════════════════

def print_report(stats: dict, tick_df: pd.DataFrame, signal_df: pd.DataFrame):
    """打印回测报告"""
    print("=" * 80)
    print("  Tick Cautious Advance Strategy - Backtest Report")
    print("=" * 80)

    # 信号统计
    total = len(tick_df)
    candidates = signal_df["is_candidate"].sum()
    print(f"\n─── Signal Filtering ───")
    print(f"  Total tick_selection records: {total}")
    print(f"  After 5-layer filter + 3-layer exclusion: {candidates} ({candidates/total*100:.1f}%)")

    # 每层过滤效果
    df = tick_df
    l2 = df["pattern"].isin([1, 2]).sum()
    l3 = (df["pattern"].isin([1, 2]) & (df["f_center"] > 0) & (df["pvdi_weighted"] > 0)).sum()
    l4 = (df["pattern"].isin([1, 2]) & (df["f_center"] > 0) & (df["pvdi_weighted"] > 0) & (df["close_dev_pct"] > -2.0)).sum()
    l5 = (df["pattern"].isin([1, 2]) & (df["f_center"] > 0) & (df["pvdi_weighted"] > 0) & (df["close_dev_pct"] > -2.0) & (df["avg_amount_20d"] > 1.0)).sum()
    exc = l5 - candidates
    print(f"  L2 (pattern=1/2):           {l2:>5d} ({l2/total*100:.1f}%)")
    print(f"  L3 (+f_center & +pvdi_w):   {l3:>5d} ({l3/total*100:.1f}%)")
    print(f"  L4 (close_dev > -2%):       {l4:>5d} ({l4/total*100:.1f}%)")
    print(f"  L5 (avg_amount > 1Y):       {l5:>5d} ({l5/total*100:.1f}%)")
    print(f"  Exclusions:                 -{exc:>4d}")

    # 策略统计
    print(f"\n─── Strategy Performance ───")
    print(f"  Total Return:       {stats['total_return']:+.2f}%")
    print(f"  Mean Daily Return:  {stats['mean_daily_return']:+.4f}%")
    print(f"  Std Daily Return:   {stats['std_daily_return']:.4f}%")
    print(f"  Annualized Sharpe:  {stats['sharpe']:.3f}")
    print(f"  Max Drawdown:       {stats['max_drawdown']:+.2f}%")
    print(f"  Trading Days:       {stats['n_trading_days']}")

    # 交易统计
    print(f"\n─── Trade Statistics ───")
    print(f"  Total Trades:       {stats['n_trades']}")
    print(f"  Win Rate:           {stats['win_rate']:.1%}")
    print(f"  Avg Return/Trade:   {stats['avg_return_per_trade']:+.3f}%")
    print(f"  Avg Win:            {stats['avg_win']:+.3f}%")
    print(f"  Avg Loss:           {stats['avg_loss']:+.3f}%")
    print(f"  Profit Factor:      {stats['profit_factor']:.2f}")
    print(f"  Avg Holdings/Day:   {stats['avg_holdings_per_day']:.1f}")

    # 出场原因
    if stats["reason_stats"]:
        print(f"\n─── Exit Reason Breakdown ───")
        for reason, data in stats["reason_stats"].items():
            print(f"  {reason:20s}  n={int(data['count']):3d}  avg_return={data['mean']*100:+.3f}%")

    print("\n" + "=" * 80)


# ══════════════════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Tick Cautious Advance Strategy Backtest")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "tick_experiment" / "results"),
                        help="Output directory")
    parser.add_argument("--hold-days", type=int, default=DEFAULT_HOLD_DAYS,
                        help=f"Holding period in trading days (default: {DEFAULT_HOLD_DAYS})")
    parser.add_argument("--max-holdings", type=int, default=DEFAULT_MAX_HOLDINGS,
                        help=f"Max concurrent holdings (default: {DEFAULT_MAX_HOLDINGS})")
    parser.add_argument("--stop-loss", type=float, default=DEFAULT_STOP_LOSS,
                        help=f"Stop loss threshold (default: {DEFAULT_STOP_LOSS})")
    parser.add_argument("--take-profit", type=float, default=DEFAULT_TAKE_PROFIT,
                        help=f"Take profit threshold (default: {DEFAULT_TAKE_PROFIT})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(DATABASE_URL)

    # 1. 加载数据
    print("Loading tick_selection...")
    tick_df = load_tick_selection(engine)
    print(f"  {len(tick_df)} records, {tick_df['ts_code'].nunique()} stocks, "
          f"{tick_df['selection_date'].nunique()} trading days")

    if tick_df.empty:
        print("No data, exiting")
        return

    print("Loading price data...")
    ts_codes = tick_df["ts_code"].unique().tolist()
    min_date = str(tick_df["selection_date"].min())
    max_date = str(tick_df["selection_date"].max())
    price_df = load_price_data(engine, ts_codes, min_date, max_date)
    print(f"  {len(price_df)} price records")

    # 2. 生成信号
    signal_df = generate_signals(tick_df)

    # 3. 回测
    print(f"Running backtest (hold={args.hold_days}d, max_holdings={args.max_holdings}, "
          f"stop_loss={args.stop_loss*100:.0f}%, take_profit={args.take_profit*100:.0f}%)...")
    result = run_backtest(
        tick_df, price_df,
        hold_days=args.hold_days,
        max_holdings=args.max_holdings,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
    )

    # 4. 输出报告
    print_report(result["stats"], tick_df, signal_df)

    # 5. 保存结果
    result["daily_returns"].to_csv(output_dir / "strategy_daily_returns.csv", index=False, encoding="utf-8-sig")
    result["trades"].to_csv(output_dir / "strategy_trades.csv", index=False, encoding="utf-8-sig")

    # 6. 可视化
    print("Generating charts...")
    plot_equity_curve(result["daily_returns"], output_dir / "07_equity_curve.png")
    if not result["trades"].empty:
        plot_trade_distribution(result["trades"], output_dir / "08_trade_distribution.png")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
