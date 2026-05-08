#!/usr/bin/env python3
"""
账户级组合回测引擎

Purpose: 验证周线池+日线V1信号在账户级能否跑出稳定净值
Inputs: output/daily_factor_with_scores.parquet or daily_factor_with_scores_strict.parquet, DB: stock_k_data
Outputs: output/account_backtest/ 目录下5张表
How to Run:
    python dsa_experiment/backtest/account_backtest.py
    python dsa_experiment/backtest/account_backtest.py --all-groups
    python dsa_experiment/backtest/account_backtest.py --strict
Examples:
    python dsa_experiment/backtest/account_backtest.py
    python dsa_experiment/backtest/account_backtest.py --all-groups
    python dsa_experiment/backtest/account_backtest.py --strict
Side Effects: 只读操作，输出CSV到 dsa_experiment/output/account_backtest/

交易口径: T日收盘信号 → T+1开盘买 → T+N+1开盘卖
交易成本: 0.15%双边（买0.05%+卖0.05%+印花税0.05%）
交易约束（--strict模式）:
  涨停无法买入: 当日开盘价=最高价 且 涨幅>=9.5%
  跌停无法卖出: 当日开盘价=最低价 且 跌幅>=9.5%
  停牌无法交易: 当日成交量为0
  阈值来源: 训练集OOF分位数（无泄漏）
"""

import sys
import os
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
BACKTEST_DIR = os.path.join(OUTPUT_DIR, "account_backtest")

BUY_COST = 0.0005
SELL_COST = 0.0010
STOP_THRESHOLD = -0.05


def load_daily_prices(start_date: str, end_date: str) -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    from config import DATABASE_URL
    engine = create_engine(DATABASE_URL)
    sql = text("""
        SELECT ts_code, bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE freq = 'd' AND bar_time >= :start AND bar_time <= :end
        ORDER BY ts_code, bar_time
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"start": start_date, "end": end_date})
    df["raw_code"] = df["ts_code"].str[:6]
    df["bar_time"] = pd.to_datetime(df["bar_time"])
    return df


LIMIT_UP_THRESHOLD = 0.095
LIMIT_DOWN_THRESHOLD = -0.095


def is_limit_up(row_open, row_high, row_close, prev_close) -> bool:
    if pd.isna(row_open) or pd.isna(row_high) or pd.isna(prev_close) or prev_close <= 0:
        return False
    if abs(row_open - row_high) < 1e-6 and (row_open - prev_close) / prev_close >= LIMIT_UP_THRESHOLD:
        return True
    return False


def is_limit_down(row_open, row_low, prev_close) -> bool:
    if pd.isna(row_open) or pd.isna(row_low) or pd.isna(prev_close) or prev_close <= 0:
        return False
    if abs(row_open - row_low) < 1e-6 and (row_open - prev_close) / prev_close <= LIMIT_DOWN_THRESHOLD:
        return True
    return False


def is_suspended(row_volume) -> bool:
    if pd.isna(row_volume) or row_volume <= 0:
        return True
    return False


def run_backtest(
    signals_df: pd.DataFrame,
    daily_prices: pd.DataFrame,
    max_stocks: int = 8,
    hold_days: int = 5,
    use_stopsell: bool = False,
    use_layered_weight: bool = False,
    group_name: str = "",
    strict: bool = False,
) -> dict:
    price_pivot = daily_prices.pivot_table(index="bar_time", columns="raw_code", values=["open", "high", "low", "close", "volume"], aggfunc="first")
    trading_days = sorted(daily_prices["bar_time"].unique())
    all_codes = daily_prices["raw_code"].unique()

    prev_close_map = {}
    for code in all_codes:
        if "close" in price_pivot and code in price_pivot["close"].columns:
            prev_close_map[code] = price_pivot["close"][code].shift(1)

    signals_sorted = signals_df.sort_values(["bar_time", "daily_opportunity_score"], ascending=[True, False])

    holdings = {}
    nav = 1.0
    trade_details = []
    nav_records = []
    skipped_limit_up = 0
    skipped_limit_down = 0
    skipped_suspended = 0

    for t_idx, current_date in enumerate(trading_days):
        if current_date not in price_pivot.index:
            continue

        day_open = price_pivot.loc[current_date, "open"] if "open" in price_pivot else pd.Series(dtype=float)
        day_high = price_pivot.loc[current_date, "high"] if "high" in price_pivot else pd.Series(dtype=float)
        day_low = price_pivot.loc[current_date, "low"] if "low" in price_pivot else pd.Series(dtype=float)
        day_close = price_pivot.loc[current_date, "close"] if "close" in price_pivot else pd.Series(dtype=float)

        to_sell = []
        for code, h in list(holdings.items()):
            h["days_held"] += 1

            if use_stopsell and not h.get("stopped", False):
                if code in day_low.index and not np.isnan(day_low[code]):
                    mae = (day_low[code] - h["buy_price"]) / h["buy_price"]
                    if mae < STOP_THRESHOLD:
                        h["stopped"] = True

            should_sell = False
            sell_reason = ""
            if h.get("stopped", False) and h["days_held"] >= 2:
                should_sell = True
                sell_reason = "stop"
            elif h["days_held"] > hold_days:
                should_sell = True
                sell_reason = "fixed"

            if should_sell:
                if code in day_open.index and not np.isnan(day_open[code]):
                    sell_price = day_open[code]
                elif code in day_close.index and not np.isnan(day_close[code]):
                    sell_price = day_close[code]
                else:
                    continue

                if strict:
                    if "volume" in price_pivot and code in price_pivot["volume"].columns:
                        vol_series = price_pivot["volume"][code]
                        if current_date in vol_series.index:
                            if is_suspended(vol_series.get(current_date, np.nan)):
                                skipped_suspended += 1
                                continue
                    if code in prev_close_map and current_date in prev_close_map[code].index:
                        prev_c = prev_close_map[code].get(current_date, np.nan)
                        if not np.isnan(prev_c) and prev_c > 0:
                            if "low" in price_pivot and code in price_pivot["low"].columns:
                                day_low_series = price_pivot["low"][code]
                                if current_date in day_low_series.index:
                                    if is_limit_down(sell_price, day_low_series.get(current_date, np.nan), prev_c):
                                        skipped_limit_down += 1
                                        continue

                gross_ret = (sell_price - h["buy_price"]) / h["buy_price"]
                net_ret = gross_ret - BUY_COST - SELL_COST

                mae_val = np.nan
                mfe_val = np.nan
                for d_back in range(1, h["days_held"] + 1):
                    check_idx = t_idx - d_back
                    if check_idx < 0 or check_idx >= len(trading_days):
                        continue
                    check_date = trading_days[check_idx]
                    if check_date not in price_pivot.index:
                        continue
                    check_low = price_pivot.loc[check_date, "low"]
                    check_high = price_pivot.loc[check_date, "high"]
                    if code in check_low.index and not np.isnan(check_low[code]):
                        c_mae = (check_low[code] - h["buy_price"]) / h["buy_price"]
                        if np.isnan(mae_val) or c_mae < mae_val:
                            mae_val = c_mae
                    if code in check_high.index and not np.isnan(check_high[code]):
                        c_mfe = (check_high[code] - h["buy_price"]) / h["buy_price"]
                        if np.isnan(mfe_val) or c_mfe > mfe_val:
                            mfe_val = c_mfe

                trade_details.append({
                    "ts_code": code, "buy_date": h["buy_date"], "sell_date": current_date,
                    "buy_price": h["buy_price"], "sell_price": sell_price,
                    "hold_days": h["days_held"], "gross_ret": gross_ret, "net_ret": net_ret,
                    "mae": mae_val, "mfe": mfe_val, "sell_reason": sell_reason,
                    "opp_score": h.get("opp_score", 0), "risk_score": h.get("risk_score", 0),
                    "weight": h["weight"],
                })
                to_sell.append(code)

        for code in to_sell:
            del holdings[code]

        day_signals = signals_sorted[signals_sorted["bar_time"] == current_date]
        day_signals = day_signals.drop_duplicates(subset=["ts_code"], keep="first")
        n_available = max_stocks - len(holdings)

        if n_available > 0 and len(day_signals) > 0:
            candidates = day_signals.head(n_available * 2)
            new_codes = []

            next_idx = t_idx + 1
            if next_idx >= len(trading_days):
                continue
            buy_date = trading_days[next_idx]

            if buy_date not in price_pivot.index:
                continue
            buy_day_open = price_pivot.loc[buy_date, "open"]

            for _, sig in candidates.iterrows():
                if len(holdings) + len(new_codes) >= max_stocks:
                    break
                code = sig["ts_code"]
                if code in holdings:
                    continue
                if code not in buy_day_open.index or np.isnan(buy_day_open[code]):
                    continue
                buy_price = buy_day_open[code]
                if buy_price <= 0:
                    continue

                if strict:
                    if "volume" in price_pivot and code in price_pivot["volume"].columns:
                        vol_series = price_pivot["volume"][code]
                        if buy_date in vol_series.index:
                            if is_suspended(vol_series.get(buy_date, np.nan)):
                                skipped_suspended += 1
                                continue
                    if code in prev_close_map and buy_date in prev_close_map[code].index:
                        prev_c = prev_close_map[code].get(buy_date, np.nan)
                        if not np.isnan(prev_c) and prev_c > 0:
                            if "high" in price_pivot and code in price_pivot["high"].columns:
                                day_high_series = price_pivot["high"][code]
                                if buy_date in day_high_series.index:
                                    if is_limit_up(buy_price, day_high_series.get(buy_date, np.nan), buy_price, prev_c):
                                        skipped_limit_up += 1
                                        continue

                new_codes.append((code, buy_price, sig))

            if new_codes:
                n_total = len(holdings) + len(new_codes)
                if use_layered_weight:
                    scores = [sig["daily_opportunity_score"] - sig["daily_risk_score"] for _, _, sig in new_codes]
                    sorted_idx = np.argsort(scores)[::-1]
                    weights = np.ones(len(new_codes))
                    top_k = max(1, len(new_codes) // 5)
                    for rank, idx in enumerate(sorted_idx):
                        if rank < top_k:
                            weights[idx] = 1.5
                        elif rank < top_k + max(1, len(new_codes) * 3 // 10):
                            weights[idx] = 1.0
                        else:
                            weights[idx] = 0.5
                    for code in holdings:
                        holdings[code]["weight"] = 1.0 / n_total
                    w_sum = weights.sum() + len(holdings) * 1.0
                    weights = weights / w_sum
                    for i, (code, bp, sig) in enumerate(new_codes):
                        holdings[code] = {
                            "buy_date": buy_date, "buy_price": bp,
                            "weight": float(weights[i]), "days_held": 0,
                            "opp_score": sig["daily_opportunity_score"],
                            "risk_score": sig["daily_risk_score"],
                        }
                else:
                    w = 1.0 / n_total
                    for code in holdings:
                        holdings[code]["weight"] = w
                    for code, bp, sig in new_codes:
                        holdings[code] = {
                            "buy_date": buy_date, "buy_price": bp,
                            "weight": w, "days_held": 0,
                            "opp_score": sig["daily_opportunity_score"],
                            "risk_score": sig["daily_risk_score"],
                        }

        daily_ret = 0.0
        for code, h in holdings.items():
            if code in day_close.index and not np.isnan(day_close[code]):
                prev_close = day_close[code]
                if t_idx > 0:
                    prev_date = trading_days[t_idx - 1]
                    if prev_date in price_pivot.index:
                        prev_day_close = price_pivot.loc[prev_date, "close"]
                        if code in prev_day_close.index and not np.isnan(prev_day_close[code]):
                            prev_close = prev_day_close[code]
                if h["days_held"] == 0:
                    if code in day_open.index and not np.isnan(day_open[code]):
                        stock_ret = (day_close[code] - day_open[code]) / day_open[code]
                    else:
                        stock_ret = 0
                else:
                    if t_idx > 0:
                        prev_date = trading_days[t_idx - 1]
                        if prev_date in price_pivot.index:
                            prev_day_close = price_pivot.loc[prev_date, "close"]
                            if code in prev_day_close.index and not np.isnan(prev_day_close[code]):
                                stock_ret = (day_close[code] - prev_day_close[code]) / prev_day_close[code]
                            else:
                                stock_ret = 0
                        else:
                            stock_ret = 0
                    else:
                        stock_ret = 0
                daily_ret += h["weight"] * stock_ret

        nav *= (1 + daily_ret)
        cash_ratio = 1.0 - sum(h["weight"] for h in holdings.values())

        nav_records.append({
            "date": current_date, "nav": nav, "daily_ret": daily_ret,
            "n_positions": len(holdings), "cash_ratio": max(0, cash_ratio),
        })

    nav_df = pd.DataFrame(nav_records)
    trades_df = pd.DataFrame(trade_details)
    return {"group_name": group_name, "nav_df": nav_df, "trades_df": trades_df,
            "max_stocks": max_stocks, "hold_days": hold_days,
            "use_stopsell": use_stopsell, "use_layered_weight": use_layered_weight,
            "skipped_limit_up": skipped_limit_up, "skipped_limit_down": skipped_limit_down,
            "skipped_suspended": skipped_suspended}


def compute_summary(result: dict) -> dict:
    nav_df = result["nav_df"]
    trades_df = result["trades_df"]
    if nav_df.empty:
        return {}

    nav_df = nav_df.copy()
    nav_df["cummax"] = nav_df["nav"].cummax()
    nav_df["drawdown"] = (nav_df["nav"] - nav_df["cummax"]) / nav_df["cummax"]

    total_days = len(nav_df)
    total_years = total_days / 252
    final_nav = nav_df["nav"].iloc[-1]
    annual_ret = (final_nav ** (1 / total_years) - 1) if total_years > 0 and final_nav > 0 else 0
    max_dd = nav_df["drawdown"].min()
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 1e-6 else 0
    daily_rets = nav_df["daily_ret"]
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0

    n_trades = len(trades_df)
    win_rate = (trades_df["net_ret"] > 0).mean() if n_trades > 0 else 0
    avg_ret = trades_df["net_ret"].mean() if n_trades > 0 else 0
    avg_mae = trades_df["mae"].mean() if n_trades > 0 else 0
    avg_mfe = trades_df["mfe"].mean() if n_trades > 0 else 0
    rr = avg_ret / abs(avg_mae) if abs(avg_mae) > 1e-6 else 0
    stop_rate = (trades_df["sell_reason"] == "stop").mean() if n_trades > 0 else 0

    nav_df["year"] = pd.to_datetime(nav_df["date"]).dt.year
    yearly = {}
    for year, grp in nav_df.groupby("year"):
        year_start = grp["nav"].iloc[0]
        year_end = grp["nav"].iloc[-1]
        year_ret = (year_end / year_start - 1) if year_start > 0 else 0
        year_dd = grp["drawdown"].min()
        year_trades = trades_df[pd.to_datetime(trades_df["sell_date"]).dt.year == year] if n_trades > 0 else pd.DataFrame()
        yearly[year] = {
            "year": year, "annual_ret": year_ret, "max_dd": year_dd,
            "n_trades": len(year_trades),
            "win_rate": (year_trades["net_ret"] > 0).mean() if len(year_trades) > 0 else 0,
            "avg_ret": year_trades["net_ret"].mean() if len(year_trades) > 0 else 0,
        }

    return {
        "group": result["group_name"], "annual_ret": annual_ret, "max_dd": max_dd,
        "calmar": calmar, "sharpe": sharpe, "n_trades": n_trades,
        "win_rate": win_rate, "avg_ret": avg_ret, "avg_mae": avg_mae,
        "avg_mfe": avg_mfe, "rr": rr, "stop_rate": stop_rate,
        "final_nav": final_nav, "yearly": yearly,
    }


def main():
    parser = argparse.ArgumentParser(description="账户级组合回测")
    parser.add_argument("--all-groups", action="store_true")
    parser.add_argument("--strict", action="store_true", help="严格模式：使用OOS数据+交易约束+训练集阈值")
    args = parser.parse_args()

    os.makedirs(BACKTEST_DIR, exist_ok=True)

    print("=" * 80)
    print("账户级组合回测" + (" [严格模式]" if args.strict else ""))
    print("=" * 80)

    print("\n[1/4] 加载日线信号...")
    if args.strict:
        strict_path = os.path.join(OUTPUT_DIR, "daily_factor_with_scores_strict.parquet")
        if os.path.exists(strict_path):
            signals = pd.read_parquet(strict_path)
            print(f"  使用严格OOS数据: {strict_path}")
        else:
            print(f"  错误: 未找到严格OOS数据 {strict_path}，请先运行 train_daily_models.py --strict-oos")
            return
    else:
        signals = pd.read_parquet(os.path.join(OUTPUT_DIR, "daily_factor_with_scores.parquet"))

    signals["bar_time"] = pd.to_datetime(signals["bar_time"])

    if args.strict and "opp_train_q70" in signals.columns:
        opp_q = signals["opp_train_q70"].iloc[0]
        risk_q = signals["risk_train_q30"].iloc[0]
        print(f"  使用训练集OOF阈值（无泄漏）: opp_q70={opp_q:.4f}, risk_q30={risk_q:.4f}")
    else:
        opp_q = signals["daily_opportunity_score"].quantile(0.7)
        risk_q = signals["daily_risk_score"].quantile(0.3)
        print(f"  使用全样本分位数: opp_q70={opp_q:.4f}, risk_q30={risk_q:.4f}")

    signals = signals[(signals["daily_opportunity_score"] >= opp_q) & (signals["daily_risk_score"] >= risk_q)].copy()
    print(f"  入场信号数: {len(signals)}")

    print("\n[2/4] 加载日线价格...")
    start_date = str(signals["bar_time"].min().date())
    end_date = str(signals["bar_time"].max().date())
    daily_prices = load_daily_prices(start_date, end_date)
    print(f"  日线记录: {len(daily_prices)}")

    print("\n[3/4] 构建价格表...")

    groups = [
        {"max_stocks": 8, "hold_days": 5, "use_stopsell": False, "use_layered_weight": False, "name": "P1_8st_5d_equal"},
        {"max_stocks": 8, "hold_days": 10, "use_stopsell": False, "use_layered_weight": False, "name": "P2_8st_10d_equal"},
        {"max_stocks": 8, "hold_days": 5, "use_stopsell": False, "use_layered_weight": True, "name": "P3_8st_5d_layered"},
        {"max_stocks": 8, "hold_days": 5, "use_stopsell": True, "use_layered_weight": False, "name": "P4_8st_5d_stop"},
    ] if args.all_groups else [
        {"max_stocks": 8, "hold_days": 5, "use_stopsell": False, "use_layered_weight": False, "name": "P1_8st_5d_equal"},
    ]

    print(f"\n[4/4] 运行回测 ({len(groups)} 组)...")
    all_summaries = []
    for g in groups:
        print(f"\n  运行: {g['name']}...")
        result = run_backtest(
            signals, daily_prices,
            max_stocks=g["max_stocks"], hold_days=g["hold_days"],
            use_stopsell=g["use_stopsell"], use_layered_weight=g["use_layered_weight"],
            group_name=g["name"], strict=args.strict,
        )
        summary = compute_summary(result)
        all_summaries.append(summary)
        result["nav_df"].to_csv(os.path.join(BACKTEST_DIR, f"nav_{g['name']}.csv"), index=False)
        result["trades_df"].to_csv(os.path.join(BACKTEST_DIR, f"trades_{g['name']}.csv"), index=False)
        print(f"    年化={summary['annual_ret']:.2%}, 最大回撤={summary['max_dd']:.2%}, 夏普={summary['sharpe']:.2f}, 卡玛={summary['calmar']:.2f}")
        print(f"    交易数={summary['n_trades']}, 胜率={summary['win_rate']:.0%}, avg_ret={summary['avg_ret']:.2%}, RR={summary['rr']:.2f}")
        if args.strict:
            print(f"    交易约束跳过: 涨停={result['skipped_limit_up']}, 跌停={result['skipped_limit_down']}, 停牌={result['skipped_suspended']}")

    print(f"\n{'=' * 80}")
    print("组合汇总")
    print(f"{'=' * 80}")
    print(f"{'组':<25} {'年化':>8} {'最大回撤':>8} {'夏普':>6} {'卡玛':>6} {'胜率':>5} {'avg_ret':>8} {'RR':>5} {'交易数':>5}")
    print("-" * 80)
    for s in all_summaries:
        print(f"{s['group']:<25} {s['annual_ret']:>7.2%} {s['max_dd']:>7.2%} {s['sharpe']:>5.2f} {s['calmar']:>5.2f} {s['win_rate']:>4.0%} {s['avg_ret']:>7.2%} {s['rr']:>5.2f} {s['n_trades']:>5}")

    print(f"\n分年统计:")
    for s in all_summaries:
        print(f"\n  {s['group']}:")
        for year, yd in sorted(s["yearly"].items()):
            print(f"    {year}: ret={yd['annual_ret']:.2%}, dd={yd['max_dd']:.2%}, n={yd['n_trades']}, win={yd['win_rate']:.0%}, avg={yd['avg_ret']:.2%}")

    pd.DataFrame([{k: v for k, v in s.items() if k != "yearly"} for s in all_summaries]).to_csv(
        os.path.join(BACKTEST_DIR, "portfolio_summary.csv"), index=False
    )
    print(f"\n已保存到: {BACKTEST_DIR}")


if __name__ == "__main__":
    main()
