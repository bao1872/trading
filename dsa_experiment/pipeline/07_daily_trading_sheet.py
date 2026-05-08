#!/usr/bin/env python3
"""
模拟盘每日决策脚本

Purpose: 每天运行一次，输出完整的4层决策表+1张交易清单
Inputs: stock_dsa_vreversal_results (DB), stock_k_data (DB), stock_pools (DB),
        return_model/model.txt, risk_model/model.txt,
        daily_return_model/model.txt, daily_risk_model/model.txt,
        output/portfolio_state.json
Outputs: output/trading_sheet_{date}.csv (交易清单)
         output/watch_pool_{date}.csv (周线观察池)
         output/daily_candidates_{date}.csv (日线候选)
         output/trading_orders_{date}.csv (交易指令)
         output/portfolio_monitor_{date}.csv (持仓监控)
         output/portfolio_state.json (持仓状态持久化)
How to Run:
    python dsa_experiment/pipeline/07_daily_trading_sheet.py --date 2026-04-30
    python dsa_experiment/pipeline/07_daily_trading_sheet.py --date 2026-04-30 --dry-run
Examples:
    python dsa_experiment/pipeline/07_daily_trading_sheet.py --date 2026-04-30
    python dsa_experiment/pipeline/07_daily_trading_sheet.py --date today
    python dsa_experiment/pipeline/07_daily_trading_sheet.py --date 2026-04-30 --dry-run
Side Effects: 更新 portfolio_state.json 持仓状态文件

管线位置: Step 7/7 — 每日决策输出（4层决策表+交易清单）

固定参数（基于实验验证）:
    周线档位: A+B档
    日线机会阈值: opp_train_q70
    日线风险阈值: risk_train_q30
    最大持仓数: 8
    默认持有天数: 5
    止损: -5%
    仓位: 分层（A类1.5份/B类1份/C类0.5份）
"""

import sys
import os
import json
import argparse
import warnings
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dsa_experiment.pipeline.dsa_config import DSAConfig

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

warnings.filterwarnings("ignore")

engine = create_engine(DATABASE_URL)

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "output")
PORTFOLIO_STATE_PATH = os.path.join(OUTPUT_DIR, "portfolio_state.json")

from dsa_experiment.pipeline.factor_columns import WEEKLY_FEATURE_COLS, DAILY_SELECT_FACTORS
from dsa_experiment.pipeline.derived_features import WEEKLY_DERIVED, DAILY_DERIVED, build_weekly_features, build_daily_features

DSA_CFG = DSAConfig(prd=50, base_apt=20.0, use_adapt=False, vol_bias=10.0, atr_len=50)

MAX_HOLDINGS = 8
HOLD_DAYS = 5
STOP_LOSS_PCT = -0.05
POSITION_WEIGHTS = {"A": 1.5, "B": 1.0, "C": 0.5}

DAILY_EXTRA = ["weekly_return_score", "weekly_risk_score", "day_offset"]


def load_thresholds():
    strict_path = os.path.join(OUTPUT_DIR, "daily_factor_with_scores_strict.parquet")
    if os.path.exists(strict_path):
        df = pd.read_parquet(strict_path)
        if "opp_train_q70" in df.columns:
            return {
                "opp_q70": float(df["opp_train_q70"].iloc[0]),
                "risk_q30": float(df["risk_train_q30"].iloc[0]),
            }
    return {"opp_q70": 0.0123, "risk_q30": -0.0541}


def load_portfolio_state():
    if os.path.exists(PORTFOLIO_STATE_PATH):
        with open(PORTFOLIO_STATE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_portfolio_state(state):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PORTFOLIO_STATE_PATH, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_weekly_feature_cols(df):
    all_features = WEEKLY_FEATURE_COLS + list(WEEKLY_DERIVED.keys())
    return [c for c in all_features if c in df.columns and not df[c].isna().all()]


def assign_weekly_grade(df):
    df = df.copy()
    if df.empty:
        df["watch_tier"] = "D"
        return df
    opp_p80 = df["weekly_opportunity_score"].quantile(0.80)
    risk_p50 = df["weekly_risk_score"].quantile(0.50)
    df["watch_tier"] = "D"
    df.loc[(df["weekly_opportunity_score"] >= opp_p80) & (df["weekly_risk_score"] <= risk_p50), "watch_tier"] = "A"
    df.loc[(df["weekly_opportunity_score"] >= opp_p80) & (df["weekly_risk_score"] > risk_p50), "watch_tier"] = "B"
    df.loc[(df["weekly_opportunity_score"] < opp_p80) & (df["weekly_risk_score"] <= risk_p50), "watch_tier"] = "C"
    return df


def apply_risk_veto(df, veto_pct=0.20):
    df = df.copy()
    if df.empty:
        df["vetoed"] = False
        return df
    veto_cutoff = df["weekly_risk_score"].quantile(1 - veto_pct)
    df["vetoed"] = df["weekly_risk_score"] > veto_cutoff
    return df


def fetch_stock_names(ts_codes):
    if not ts_codes:
        return {}
    all_codes = []
    for code in ts_codes:
        if code.startswith(("6", "5")):
            all_codes.append(f"{code}.SH")
        else:
            all_codes.append(f"{code}.SZ")
    try:
        sql = text("SELECT ts_code, name, industry_l2, industry_l3 FROM stock_pools WHERE ts_code = ANY(:codes)")
        with engine.connect() as conn:
            result = conn.execute(sql, {"codes": all_codes})
            return {row[0].split(".")[0]: {"name": row[1], "industry_l2": row[2], "industry_l3": row[3]} for row in result}
    except Exception:
        return {}


def build_daily_features_batch(watch_pool_ab, target_date):
    """从 factor_value 读取日线因子 + K线最后一行为每日特征"""
    codes = watch_pool_ab["ts_code"].unique().tolist()
    if not codes:
        return pd.DataFrame()

    all_codes_db = []
    for code in codes:
        if code.startswith(("6", "5")):
            all_codes_db.append(f"{code}.SH")
        else:
            all_codes_db.append(f"{code}.SZ")

    end_date_str = target_date.strftime("%Y-%m-%d")

    factor_sql = text("""
        SELECT ts_code, factor_name, factor_value
        FROM factor_value
        WHERE ts_code = ANY(:codes) AND freq = '1d' AND as_of_date = :as_of
    """)
    with engine.connect() as conn:
        factor_long = pd.read_sql(factor_sql, conn,
                                  params={"codes": all_codes_db, "as_of": end_date_str})

    if factor_long.empty:
        return pd.DataFrame()

    factors_wide = factor_long.pivot(index="ts_code", columns="factor_name",
                                     values="factor_value").reset_index()
    factors_wide["raw_code"] = factors_wide["ts_code"].str[:6]
    factors_wide = factors_wide.set_index("raw_code", drop=False)

    k_sql = text("""
        SELECT ts_code, bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE ts_code = ANY(:codes) AND freq = 'd' AND bar_time = :bar_time
    """)
    with engine.connect() as conn:
        last_klines = pd.read_sql(k_sql, conn,
                                  params={"codes": all_codes_db, "bar_time": end_date_str})
    if not last_klines.empty:
        last_klines["raw_code"] = last_klines["ts_code"].str[:6]
        last_klines = last_klines.set_index("raw_code")

    results = []
    code_map = dict(zip(
        watch_pool_ab["ts_code"],
        zip(watch_pool_ab["weekly_opportunity_score"], watch_pool_ab["weekly_risk_score"]),
    ))

    for raw_code in codes:
        if raw_code not in code_map or raw_code not in factors_wide.index:
            continue
        w_opp, w_risk = code_map[raw_code]
        fac_row = factors_wide.loc[raw_code].to_dict()
        fac_row["ts_code"] = raw_code
        fac_row["weekly_return_score"] = w_opp
        fac_row["weekly_risk_score"] = w_risk
        fac_row["day_offset"] = 0
        if raw_code in last_klines.index:
            krow = last_klines.loc[raw_code]
            fac_row["bar_time"] = krow["bar_time"]
            fac_row["close"] = krow["close"]
            fac_row["open"] = krow["open"]
            fac_row["high"] = krow["high"]
            fac_row["low"] = krow["low"]
            fac_row["volume"] = krow["volume"]
        else:
            fac_row["bar_time"] = np.nan
            fac_row["close"] = np.nan
            fac_row["open"] = np.nan
            fac_row["high"] = np.nan
            fac_row["low"] = np.nan
            fac_row["volume"] = np.nan
        results.append(fac_row)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def get_daily_feature_cols(df):
    all_features = DAILY_SELECT_FACTORS + DAILY_EXTRA + list(DAILY_DERIVED.keys())
    return [c for c in all_features if c in df.columns and not df[c].isna().all()]


# ============================================================
# 模块1：周线观察池总表
# ============================================================
WATCH_POOL_LOOKBACK_DAYS = 14  # 无当日触发点时，回溯最近N天内的触发点


def generate_watch_pool(target_date):
    # 优先查当天触发点
    sql = text("SELECT * FROM stock_dsa_vreversal_results WHERE selection_date = :sel_date")
    with engine.connect() as conn:
        triggers = pd.read_sql(sql, conn, params={"sel_date": target_date.strftime("%Y-%m-%d")})

    if triggers.empty:
        # 当天无触发点，回溯最近 N 天内的触发点
        lookback = (target_date - timedelta(days=WATCH_POOL_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        today_str = target_date.strftime("%Y-%m-%d")
        sql_lb = text("""
            SELECT * FROM stock_dsa_vreversal_results
            WHERE selection_date >= :lookback AND selection_date < :today
            ORDER BY selection_date DESC
        """)
        with engine.connect() as conn:
            triggers = pd.read_sql(sql_lb, conn, params={"lookback": lookback, "today": today_str})

        if triggers.empty:
            return pd.DataFrame()

        # 同一股票保留最近日期的触发记录（去重）
        triggers = triggers.sort_values("selection_date", ascending=False)
        triggers = triggers.drop_duplicates(subset=["ts_code"], keep="first")
        print(f"  [回溯] 使用近{WATCH_POOL_LOOKBACK_DAYS}天触发点: {len(triggers)} 只")

    triggers = build_weekly_features(triggers)
    feature_cols = get_weekly_feature_cols(triggers)

    return_model = lgb.Booster(model_file=os.path.join(OUTPUT_DIR, "return_model", "model.txt"))
    risk_model = lgb.Booster(model_file=os.path.join(OUTPUT_DIR, "risk_model", "model.txt"))

    X = triggers[feature_cols].fillna(0)
    triggers["weekly_opportunity_score"] = return_model.predict(X, num_iteration=return_model.best_iteration)
    triggers["weekly_risk_score"] = risk_model.predict(X, num_iteration=risk_model.best_iteration)

    triggers = assign_weekly_grade(triggers)
    triggers = apply_risk_veto(triggers)

    names_map = fetch_stock_names(triggers["ts_code"].unique().tolist())
    triggers["stock_name"] = triggers["ts_code"].map(lambda c: names_map.get(c, {}).get("name", ""))
    triggers["industry_l2"] = triggers["ts_code"].map(lambda c: names_map.get(c, {}).get("industry_l2", ""))
    triggers["industry_l3"] = triggers["ts_code"].map(lambda c: names_map.get(c, {}).get("industry_l3", ""))

    # trigger_weeks_ago: 距离选股日的周数（0=当天触发，>0为历史触发点）
    if "selection_date" in triggers.columns:
        triggers["trigger_weeks_ago"] = triggers["selection_date"].apply(
            lambda d: (target_date - datetime.strptime(str(d)[:10], "%Y-%m-%d").date()).days // 7
            if pd.notna(d) else 0
        )
    else:
        triggers["trigger_weeks_ago"] = 0

    output_cols = [
        "ts_code", "stock_name", "industry_l2", "industry_l3",
        "weekly_opportunity_score", "weekly_risk_score",
        "watch_tier", "vetoed",
        "trigger_bar_time", "trigger_weeks_ago",
        "price_vs_dsa_vwap_pct", "bars_since_last_low", "ret_to_last_low_pct",
        "bbmacd", "vol_zscore_5", "vol_zscore_10", "vol_zscore_20",
    ]
    avail = [c for c in output_cols if c in triggers.columns]
    return triggers[avail]


# ============================================================
# 模块2：日线信号候选表
# ============================================================
def generate_daily_candidates(watch_pool, target_date, thresholds):
    ab_pool = watch_pool[(watch_pool["watch_tier"].isin(["A", "B"])) & (~watch_pool["vetoed"])].copy()
    if ab_pool.empty:
        return pd.DataFrame()

    daily_factors = build_daily_features_batch(ab_pool, target_date)
    if daily_factors.empty:
        return pd.DataFrame()

    daily_factors = build_daily_features(daily_factors)
    feature_cols = get_daily_feature_cols(daily_factors)

    daily_return_model = lgb.Booster(model_file=os.path.join(OUTPUT_DIR, "daily_return_model", "model.txt"))
    daily_risk_model = lgb.Booster(model_file=os.path.join(OUTPUT_DIR, "daily_risk_model", "model.txt"))

    X = daily_factors[feature_cols].fillna(0)
    daily_factors["daily_opportunity_score"] = daily_return_model.predict(X, num_iteration=daily_return_model.best_iteration)
    daily_factors["daily_risk_score"] = daily_risk_model.predict(X, num_iteration=daily_risk_model.best_iteration)

    opp_q70 = thresholds["opp_q70"]
    risk_q30 = thresholds["risk_q30"]
    daily_factors["meets_entry"] = (
        (daily_factors["daily_opportunity_score"] >= opp_q70)
        & (daily_factors["daily_risk_score"] >= risk_q30)
    )
    n_total = len(daily_factors)
    daily_factors["opp_percentile"] = daily_factors["daily_opportunity_score"].rank(pct=True)
    daily_factors["risk_percentile"] = daily_factors["daily_risk_score"].rank(pct=True)

    portfolio = load_portfolio_state()
    daily_factors["has_position"] = daily_factors["ts_code"].isin(portfolio.keys())
    daily_factors["is_new_signal"] = daily_factors["meets_entry"] & ~daily_factors["has_position"]

    watch_map = watch_pool.set_index("ts_code")[["watch_tier", "weekly_opportunity_score", "weekly_risk_score"]].to_dict("index")
    daily_factors["watch_tier"] = daily_factors["ts_code"].map(lambda c: watch_map.get(c, {}).get("watch_tier", "D"))
    daily_factors["weekly_opportunity_score"] = daily_factors["ts_code"].map(lambda c: watch_map.get(c, {}).get("weekly_opportunity_score", np.nan))
    daily_factors["weekly_risk_score"] = daily_factors["ts_code"].map(lambda c: watch_map.get(c, {}).get("weekly_risk_score", np.nan))

    snapshot_cols = [
        "ts_code", "watch_tier", "weekly_opportunity_score", "weekly_risk_score",
        "daily_opportunity_score", "daily_risk_score",
        "meets_entry", "opp_percentile", "risk_percentile",
        "price_vs_dsa_vwap_pct", "dsa_pivot_pos_01",
        "bars_since_last_high", "bars_since_last_low",
        "current_pullback_from_stage_extreme_pct",
        "bbmacd_minus_avg", "bbmacd_state",
        "vol_zscore_5", "vol_zscore_10",
        "coord_stage_current", "price_vol_coord",
        "is_new_signal", "has_position", "close",
    ]
    avail = [c for c in snapshot_cols if c in daily_factors.columns]
    return daily_factors[avail]


# ============================================================
# 模块3：当日交易指令表
# ============================================================
def generate_trading_orders(daily_candidates, target_date, portfolio):
    orders = []

    # 即使 daily_candidates 为空，有持仓也需检查止损/到期
    if not portfolio:
        return pd.DataFrame()

    # 获取持仓股票的当前价格（优先从 daily_candidates 取，否则从K线取）
    price_map = {}
    for ts_code in portfolio:
        if not daily_candidates.empty and ts_code in daily_candidates["ts_code"].values:
            row = daily_candidates[daily_candidates["ts_code"] == ts_code].iloc[0]
            price_map[ts_code] = row.get("close", np.nan)

    missing_codes = [c for c in portfolio if c not in price_map or np.isnan(price_map.get(c, np.nan))]
    if missing_codes:
        all_codes_db = []
        for code in missing_codes:
            if code.startswith(("6", "5")):
                all_codes_db.append(f"{code}.SH")
            else:
                all_codes_db.append(f"{code}.SZ")
        try:
            k_sql = text("""
                SELECT ts_code, close FROM stock_k_data
                WHERE freq = 'd' AND ts_code = ANY(:codes) AND bar_time = :bar_time
            """)
            with engine.connect() as conn:
                k_df = pd.read_sql(k_sql, conn, params={
                    "codes": all_codes_db,
                    "bar_time": target_date.strftime("%Y-%m-%d"),
                })
            if not k_df.empty:
                for _, row in k_df.iterrows():
                    raw_code = row["ts_code"][:6]
                    price_map[raw_code] = row["close"]
        except Exception:
            pass

    for ts_code, pos in portfolio.items():
        buy_date = datetime.strptime(pos["buy_date"], "%Y-%m-%d").date()
        buy_price = pos["buy_price"]
        stop_price = pos["stop_price"]
        exit_date = datetime.strptime(pos["exit_date"], "%Y-%m-%d").date()
        hold_days = (target_date - buy_date).days

        current_price = price_map.get(ts_code, np.nan)

        pnl_pct = (current_price / buy_price - 1) if not np.isnan(current_price) and buy_price > 0 else np.nan
        hit_stop = (pnl_pct is not np.nan and pnl_pct <= STOP_LOSS_PCT) or (not np.isnan(current_price) and current_price <= stop_price)
        expired = target_date >= exit_date

        action = "HOLD"
        reason = ""
        if hit_stop:
            action = "SELL"
            reason = "触发止损"
        elif expired:
            action = "SELL"
            reason = "持有到期"

        orders.append({
            "ts_code": ts_code,
            "action": action,
            "priority_rank": 0,
            "position_weight": 0,
            "stop_price": stop_price,
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "hold_days": hold_days,
            "buy_price": buy_price,
            "current_price": current_price,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "watch_tier": pos.get("tier", ""),
            "weekly_opp": pos.get("weekly_opp", np.nan),
            "weekly_risk": pos.get("weekly_risk", np.nan),
        })

    new_signals = daily_candidates[daily_candidates["is_new_signal"]].copy() if ("is_new_signal" in daily_candidates.columns) else pd.DataFrame()
    if not new_signals.empty:
        tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
        new_signals["tier_sort"] = new_signals["watch_tier"].map(tier_order).fillna(4)
        new_signals = new_signals.sort_values(
            ["tier_sort", "daily_opportunity_score", "daily_risk_score"],
            ascending=[True, False, False],
        )

        current_holdings = len(portfolio)
        slots_available = MAX_HOLDINGS - current_holdings

        for i, (_, row) in enumerate(new_signals.iterrows()):
            if len([o for o in orders if o["action"] == "BUY"]) >= slots_available:
                break
            tier = row.get("watch_tier", "B")
            weight = POSITION_WEIGHTS.get(tier, 1.0)
            close = row.get("close", np.nan)
            stop_price = close * (1 + STOP_LOSS_PCT) if not np.isnan(close) else np.nan
            exit_date = (target_date + timedelta(days=HOLD_DAYS)).strftime("%Y-%m-%d")

            orders.append({
                "ts_code": row["ts_code"],
                "action": "BUY",
                "priority_rank": i + 1,
                "position_weight": weight,
                "stop_price": stop_price,
                "exit_date": exit_date,
                "hold_days": 0,
                "buy_price": np.nan,
                "current_price": close,
                "pnl_pct": 0,
                "reason": f"新信号({tier}档,opp={row.get('daily_opportunity_score', 0):.4f})",
                "watch_tier": tier,
                "weekly_opp": row.get("weekly_opportunity_score", np.nan),
                "weekly_risk": row.get("weekly_risk_score", np.nan),
            })

    return pd.DataFrame(orders)


# ============================================================
# 模块4：持仓监控表
# ============================================================
def generate_portfolio_monitor(portfolio, target_date, watch_pool):
    if not portfolio:
        return pd.DataFrame()

    codes = list(portfolio.keys())
    all_codes_db = []
    for code in codes:
        if code.startswith(("6", "5")):
            all_codes_db.append(f"{code}.SH")
        else:
            all_codes_db.append(f"{code}.SZ")

    sql = text("""
        SELECT ts_code, bar_time, open, high, low, close
        FROM stock_k_data
        WHERE freq = 'd' AND ts_code = ANY(:codes)
          AND bar_time <= :end_date
        ORDER BY ts_code, bar_time
    """)
    with engine.connect() as conn:
        prices = pd.read_sql(sql, conn, params={
            "codes": all_codes_db,
            "end_date": target_date.strftime("%Y-%m-%d"),
        })

    if prices.empty:
        return pd.DataFrame()

    prices["raw_code"] = prices["ts_code"].str[:6]
    price_map = {}
    for code, grp in prices.groupby("raw_code"):
        grp = grp.sort_values("bar_time")
        price_map[code] = {
            "current": grp.iloc[-1]["close"],
            "high_since": grp["high"].max(),
            "low_since": grp["low"].min(),
        }

    monitor = []
    for ts_code, pos in portfolio.items():
        buy_date = datetime.strptime(pos["buy_date"], "%Y-%m-%d").date()
        buy_price = pos["buy_price"]
        exit_date = datetime.strptime(pos["exit_date"], "%Y-%m-%d").date()
        hold_days = (target_date - buy_date).days

        pm = price_map.get(ts_code, {})
        current = pm.get("current", np.nan)
        high_since = pm.get("high_since", np.nan)
        low_since = pm.get("low_since", np.nan)

        pnl_pct = (current / buy_price - 1) if not np.isnan(current) and buy_price > 0 else np.nan
        mfe = (high_since / buy_price - 1) if not np.isnan(high_since) and buy_price > 0 else np.nan
        mae = (low_since / buy_price - 1) if not np.isnan(low_since) and buy_price > 0 else np.nan

        hit_stop = not np.isnan(current) and current <= pos["stop_price"]
        expired = target_date >= exit_date
        in_pool = ts_code in watch_pool["ts_code"].values if not watch_pool.empty else False

        monitor.append({
            "ts_code": ts_code,
            "buy_date": pos["buy_date"],
            "hold_days": hold_days,
            "buy_price": buy_price,
            "current_price": current,
            "pnl_pct": pnl_pct,
            "mfe": mfe,
            "mae": mae,
            "hit_stop": hit_stop,
            "expired": expired,
            "in_pool": in_pool,
            "stop_price": pos["stop_price"],
            "exit_date": pos["exit_date"],
        })

    return pd.DataFrame(monitor)


# ============================================================
# 模块5：交易清单（合并输出）
# ============================================================
def generate_trading_sheet(watch_pool, daily_candidates, trading_orders, portfolio, target_date):
    rows = []

    if trading_orders.empty:
        return pd.DataFrame(rows)

    sell_orders = trading_orders[trading_orders["action"] == "SELL"]
    for _, order in sell_orders.iterrows():
        rows.append({
            "ts_code": order["ts_code"],
            "action": "SELL",
            "priority": 0,
            "watch_tier": order.get("watch_tier", ""),
            "weekly_opp": order.get("weekly_opp", np.nan),
            "weekly_risk": order.get("weekly_risk", np.nan),
            "daily_opp": np.nan,
            "daily_risk": np.nan,
            "position_weight": 0,
            "exit_date": order.get("exit_date", ""),
            "reason": order.get("reason", ""),
        })

    buy_orders = trading_orders[trading_orders["action"] == "BUY"]
    for _, order in buy_orders.iterrows():
        daily_opp = np.nan
        daily_risk = np.nan
        if order["ts_code"] in daily_candidates["ts_code"].values:
            dc_row = daily_candidates[daily_candidates["ts_code"] == order["ts_code"]].iloc[0]
            daily_opp = dc_row.get("daily_opportunity_score", np.nan)
            daily_risk = dc_row.get("daily_risk_score", np.nan)

        rows.append({
            "ts_code": order["ts_code"],
            "action": "BUY",
            "priority": order.get("priority_rank", 0),
            "watch_tier": order.get("watch_tier", ""),
            "weekly_opp": order.get("weekly_opp", np.nan),
            "weekly_risk": order.get("weekly_risk", np.nan),
            "daily_opp": daily_opp,
            "daily_risk": daily_risk,
            "position_weight": order.get("position_weight", 0),
            "exit_date": order.get("exit_date", ""),
            "reason": order.get("reason", ""),
        })

    for ts_code in portfolio:
        if ts_code not in trading_orders["ts_code"].values:
            pos = portfolio[ts_code]
            rows.append({
                "ts_code": ts_code,
                "action": "HOLD",
                "priority": 0,
                "watch_tier": pos.get("tier", ""),
                "weekly_opp": pos.get("weekly_opp", np.nan),
                "weekly_risk": pos.get("weekly_risk", np.nan),
                "daily_opp": np.nan,
                "daily_risk": np.nan,
                "position_weight": POSITION_WEIGHTS.get(pos.get("tier", "B"), 1.0),
                "exit_date": pos.get("exit_date", ""),
                "reason": "持有中",
            })

    if not rows:
        return pd.DataFrame()

    sheet = pd.DataFrame(rows)
    names_map = fetch_stock_names(sheet["ts_code"].unique().tolist())
    sheet["stock_name"] = sheet["ts_code"].map(lambda c: names_map.get(c, {}).get("name", ""))

    action_order = {"SELL": 0, "BUY": 1, "HOLD": 2}
    sheet["action_sort"] = sheet["action"].map(action_order)
    sheet = sheet.sort_values(["action_sort", "priority"]).reset_index(drop=True)
    sheet = sheet.drop(columns=["action_sort"])

    output_cols = [
        "ts_code", "stock_name", "action", "priority",
        "watch_tier", "weekly_opp", "weekly_risk",
        "daily_opp", "daily_risk",
        "position_weight", "exit_date", "reason",
    ]
    avail = [c for c in output_cols if c in sheet.columns]
    return sheet[avail]


def update_portfolio_state(portfolio, trading_orders, target_date, daily_candidates, dry_run=False):
    new_portfolio = dict(portfolio)

    if trading_orders.empty:
        if not dry_run:
            save_portfolio_state(new_portfolio)
        return new_portfolio

    sell_orders = trading_orders[trading_orders["action"] == "SELL"]
    for _, order in sell_orders.iterrows():
        code = order["ts_code"]
        if code in new_portfolio:
            del new_portfolio[code]

    buy_orders = trading_orders[trading_orders["action"] == "BUY"]
    for _, order in buy_orders.iterrows():
        code = order["ts_code"]
        if code not in new_portfolio:
            close = order.get("current_price", np.nan)
            if np.isnan(close):
                if code in daily_candidates["ts_code"].values:
                    close = daily_candidates[daily_candidates["ts_code"] == code].iloc[0].get("close", np.nan)

            new_portfolio[code] = {
                "buy_date": target_date.strftime("%Y-%m-%d"),
                "buy_price": float(close) if not np.isnan(close) else 0,
                "stop_price": float(order["stop_price"]) if not np.isnan(order.get("stop_price", np.nan)) else 0,
                "exit_date": order.get("exit_date", ""),
                "tier": order.get("watch_tier", "B"),
                "weekly_opp": float(order.get("weekly_opp", 0)) if not np.isnan(order.get("weekly_opp", np.nan)) else 0,
                "weekly_risk": float(order.get("weekly_risk", 0)) if not np.isnan(order.get("weekly_risk", np.nan)) else 0,
            }

    if not dry_run:
        save_portfolio_state(new_portfolio)
    return new_portfolio


def generate_reason_tag(action, tier, daily_opp, daily_risk, opp_threshold, risk_threshold,
                        hit_stop=False, expired=False, pnl_pct=np.nan):
    if action == "SELL":
        if hit_stop:
            return "触发止损"
        if expired:
            return "到期退出"
        return "卖出"
    if action == "BUY":
        tier_label = f"周{tier}" if tier else ""
        if not np.isnan(daily_opp) and not np.isnan(daily_risk):
            high_opp = daily_opp >= opp_threshold
            low_risk = daily_risk >= risk_threshold
            if high_opp and low_risk:
                risk_label = "高机低风"
            elif high_opp:
                risk_label = "机会优先"
            else:
                risk_label = "信号触发"
        else:
            risk_label = "信号触发"
        return f"{tier_label}+{risk_label}" if tier_label else risk_label
    if action == "HOLD":
        if hit_stop:
            return "触发止损"
        if expired:
            return "到期卖出"
        if not np.isnan(pnl_pct) and pnl_pct < -0.03:
            return "接近止损"
        if not np.isnan(daily_risk) and daily_risk < risk_threshold:
            return "风险抬升"
        return "风险平稳"
    return ""


def format_feishu_message(target_date, watch_pool, daily_candidates,
                          trading_orders, trading_sheet, portfolio_monitor):
    thresholds = load_thresholds()
    opp_threshold = thresholds["opp_q70"]
    risk_threshold = thresholds["risk_q30"]
    elements = []

    portfolio = load_portfolio_state()
    n_buy = len(trading_orders[trading_orders["action"] == "BUY"]) if not trading_orders.empty else 0
    n_sell = len(trading_orders[trading_orders["action"] == "SELL"]) if not trading_orders.empty else 0

    if n_buy > 0:
        header_template = "green"
    elif n_sell > 0:
        has_stop = any("止损" in str(r) for r in trading_orders[trading_orders["action"] == "SELL"]["reason"]) if not trading_orders.empty else False
        header_template = "orange" if has_stop else "red"
    else:
        header_template = "blue"
    header_title = f"DSA交易决策 | {target_date}"

    # 概览区：多列布局
    overview_cols = []
    if not watch_pool.empty:
        tier_counts = watch_pool["watch_tier"].value_counts().to_dict()
        n_a = tier_counts.get("A", 0)
        n_b = tier_counts.get("B", 0)
        overview_cols.append({
            "tag": "column",
            "width": "weighted",
            "weight": 1,
            "elements": [{"tag": "markdown", "content": f"**触发** {len(watch_pool)}\nA档 {n_a} B档 {n_b}"}],
        })
    else:
        overview_cols.append({
            "tag": "column",
            "width": "weighted",
            "weight": 1,
            "elements": [{"tag": "markdown", "content": "**触发** 0"}],
        })

    if not daily_candidates.empty:
        n_meets = int(daily_candidates["meets_entry"].sum()) if "meets_entry" in daily_candidates.columns else 0
        n_new = int(daily_candidates["is_new_signal"].sum()) if "is_new_signal" in daily_candidates.columns else 0
        overview_cols.append({
            "tag": "column",
            "width": "weighted",
            "weight": 1,
            "elements": [{"tag": "markdown", "content": f"**入场** {n_meets}\n新信号 {n_new}"}],
        })
    else:
        overview_cols.append({
            "tag": "column",
            "width": "weighted",
            "weight": 1,
            "elements": [{"tag": "markdown", "content": "**入场** 0"}],
        })

    n_hold = len(portfolio) - n_sell
    overview_cols.append({
        "tag": "column",
        "width": "weighted",
        "weight": 1,
        "elements": [{"tag": "markdown", "content": f"**持仓** {len(portfolio)}/{MAX_HOLDINGS}\n买入{n_buy} 卖出{n_sell} 持有{n_hold}"}],
    })

    elements.append({"tag": "column_set", "flex_mode": "bisect", "background_style": "grey", "columns": overview_cols})
    elements.append({"tag": "hr"})

    # 构建周线排名映射
    weekly_rank_map = {}
    if not watch_pool.empty and "weekly_opportunity_score" in watch_pool.columns:
        ranked = watch_pool.sort_values("weekly_opportunity_score", ascending=False).reset_index(drop=True)
        for i, row in ranked.iterrows():
            weekly_rank_map[row["ts_code"]] = i + 1

    # 构建名称映射
    all_codes = set()
    if not trading_orders.empty:
        all_codes.update(trading_orders["ts_code"].tolist())
    if portfolio_monitor is not None and not portfolio_monitor.empty:
        all_codes.update(portfolio_monitor["ts_code"].tolist())
    names_map = fetch_stock_names(list(all_codes)) if all_codes else {}

    # 构建日线分数映射
    daily_score_map = {}
    if not daily_candidates.empty:
        for _, row in daily_candidates.iterrows():
            daily_score_map[row["ts_code"]] = {
                "opp": row.get("daily_opportunity_score", np.nan),
                "risk": row.get("daily_risk_score", np.nan),
            }

    # 买入区
    buys = trading_orders[trading_orders["action"] == "BUY"] if not trading_orders.empty else pd.DataFrame()
    if not buys.empty:
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "🟢 **今日新机会**"}})
        for _, order in buys.iterrows():
            code = order["ts_code"]
            name = names_map.get(code, {}).get("name", "")
            tier = order.get("watch_tier", "")
            w_rank = weekly_rank_map.get(code, 0)
            w_opp = order.get("weekly_opp", np.nan)
            w_risk = order.get("weekly_risk", np.nan)
            d_scores = daily_score_map.get(code, {})
            d_opp = d_scores.get("opp", np.nan)
            d_risk = d_scores.get("risk", np.nan)
            priority = order.get("priority_rank", 0)
            weight = order.get("position_weight", 0)
            stop_price = order.get("stop_price", np.nan)
            exit_date = order.get("exit_date", "")

            tier_label = f"`{tier}档`" if tier else ""
            rank_label = f"周#{w_rank}" if w_rank > 0 else ""
            tier_rank = f"{tier_label} {rank_label}".strip()
            w_opp_s = f"{w_opp:.1%}" if not np.isnan(w_opp) else "N/A"
            w_risk_s = f"{w_risk:.0%}" if not np.isnan(w_risk) else "N/A"
            d_opp_s = f"{d_opp:.1%}" if not np.isnan(d_opp) else "N/A"
            d_risk_s = f"{d_risk:.1%}" if not np.isnan(d_risk) else "N/A"
            stop_s = f"{stop_price:.2f}" if not np.isnan(stop_price) else "N/A"
            exit_s = exit_date[5:] if isinstance(exit_date, str) and len(exit_date) >= 10 else str(exit_date)
            tag = generate_reason_tag("BUY", tier, d_opp, d_risk, opp_threshold, risk_threshold)

            line1 = f"**#{priority}** {code} {name} | {tier_rank} | {weight:.1f}份 | {tag}"
            line2 = f"周收益{w_opp_s} 止损率{w_risk_s} | 日收益{d_opp_s} 回撤{d_risk_s} | 止损{stop_s} 退出{exit_s}"
            elements.append({"tag": "markdown", "content": f"{line1}\n{line2}"})
            elements.append({"tag": "hr"})

    # 持仓处理区
    sells = trading_orders[trading_orders["action"] == "SELL"] if not trading_orders.empty else pd.DataFrame()
    has_hold_items = (portfolio_monitor is not None and not portfolio_monitor.empty)

    if not sells.empty or has_hold_items:
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "📈 **今日持仓处理**"}})

    # 卖出
    if not sells.empty:
        for _, order in sells.iterrows():
            code = order["ts_code"]
            name = names_map.get(code, {}).get("name", "")
            tier = order.get("watch_tier", "") or portfolio.get(code, {}).get("tier", "")
            w_rank = weekly_rank_map.get(code, 0)
            hold_days = order.get("hold_days", 0)
            pnl = order.get("pnl_pct", np.nan)
            buy_price = order.get("buy_price", np.nan)
            current_price = order.get("current_price", np.nan)
            hit_stop = order.get("reason", "").startswith("触发止损")
            expired = order.get("reason", "").startswith("持有到期")

            tier_label = f"`{tier}档`" if tier else ""
            rank_label = f"周#{w_rank}" if w_rank > 0 else ""
            tier_rank = f"{tier_label} {rank_label}".strip()
            pnl_s = f"{pnl:+.1%}" if not np.isnan(pnl) else "N/A"
            buy_s = f"{buy_price:.2f}" if not np.isnan(buy_price) else "N/A"
            cur_s = f"{current_price:.2f}" if not np.isnan(current_price) else "N/A"
            tag = generate_reason_tag("SELL", tier, np.nan, np.nan, opp_threshold, risk_threshold,
                                     hit_stop=hit_stop, expired=expired)

            line1 = f"🔴 {code} {name} | {tier_rank} | 持{hold_days}天 | {pnl_s} | {tag}"
            line2 = f"买入{buy_s} → 当前{cur_s}"
            elements.append({"tag": "markdown", "content": f"{line1}\n{line2}"})
            elements.append({"tag": "hr"})

    # 持仓（HOLD）
    if portfolio_monitor is not None and not portfolio_monitor.empty:
        for _, row in portfolio_monitor.iterrows():
            code = row["ts_code"]
            if code in sells["ts_code"].values if not sells.empty else False:
                continue
            name = names_map.get(code, {}).get("name", "")
            tier = portfolio.get(code, {}).get("tier", "")
            w_rank = weekly_rank_map.get(code, 0)
            hold_days = row["hold_days"]
            pnl = row["pnl_pct"]
            hit_stop = row["hit_stop"]
            expired = row["expired"]
            exit_date = row.get("exit_date", "")
            stop_price = portfolio.get(code, {}).get("stop_price", np.nan)

            d_scores = daily_score_map.get(code, {})
            d_opp = d_scores.get("opp", np.nan)
            d_risk = d_scores.get("risk", np.nan)

            tier_label = f"`{tier}档`" if tier else ""
            rank_label = f"周#{w_rank}" if w_rank > 0 else ""
            tier_rank = f"{tier_label} {rank_label}".strip()
            pnl_s = f"{pnl:+.1%}" if not np.isnan(pnl) else "N/A"
            exit_s = exit_date[5:] if isinstance(exit_date, str) and len(exit_date) >= 10 else str(exit_date)
            stop_s = f"{stop_price:.2f}" if not np.isnan(stop_price) else "N/A"

            tag = generate_reason_tag("HOLD", tier, d_opp, d_risk, opp_threshold, risk_threshold,
                                     hit_stop=hit_stop, expired=expired, pnl_pct=pnl)

            d_opp_s = f"{d_opp:.1%}" if not np.isnan(d_opp) else "N/A"
            d_risk_s = f"{d_risk:.1%}" if not np.isnan(d_risk) else "N/A"

            alert_prefix = "⚠️" if (hit_stop or (not np.isnan(pnl) and pnl < -0.03)) else "→"
            line1 = f"{alert_prefix} {code} {name} | {tier_rank} | 持{hold_days}天 | {pnl_s} | {tag}"
            line2 = f"日收益{d_opp_s} 回撤{d_risk_s} | 止损{stop_s} 退出{exit_s}"
            elements.append({"tag": "markdown", "content": f"{line1}\n{line2}"})
            elements.append({"tag": "hr"})

    if buys.empty and sells.empty and (portfolio_monitor is None or portfolio_monitor.empty):
        elements.append({"tag": "markdown", "content": "💤 今日无交易动作"})

    # 去掉末尾多余的 hr
    while elements and elements[-1].get("tag") == "hr":
        elements.pop()

    return header_title, header_template, elements


def main():
    parser = argparse.ArgumentParser(description="模拟盘每日决策脚本")
    parser.add_argument("--date", type=str, required=True, help="决策日期 (YYYY-MM-DD 或 'today')")
    parser.add_argument("--dry-run", action="store_true", help="试运行，不更新持仓状态")
    parser.add_argument("--notify", action="store_true", help="通过飞书推送交易决策")
    args = parser.parse_args()

    if args.date == "today":
        target_date = date.today()
    else:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print(f"模拟盘每日决策 — {target_date}")
    print("=" * 80)

    thresholds = load_thresholds()
    portfolio = load_portfolio_state()

    print(f"\n  当前持仓: {len(portfolio)} 只")
    print(f"  日线阈值: opp>={thresholds['opp_q70']:.4f}, risk>={thresholds['risk_q30']:.4f}")

    # 模块1：周线观察池
    print(f"\n[1/5] 生成周线观察池...")
    watch_pool = generate_watch_pool(target_date)
    if watch_pool.empty:
        print(f"  {target_date} 无触发点")
        if not portfolio:
            print(f"  无持仓，退出")
            return
        print(f"  有持仓 {len(portfolio)} 只，继续执行持仓监控")
    else:
        tier_counts = watch_pool["watch_tier"].value_counts().to_dict()
        n_ab = tier_counts.get("A", 0) + tier_counts.get("B", 0)
        n_vetoed = watch_pool["vetoed"].sum()
        print(f"  触发点: {len(watch_pool)}, A档: {tier_counts.get('A', 0)}, B档: {tier_counts.get('B', 0)}, C档: {tier_counts.get('C', 0)}, D档: {tier_counts.get('D', 0)}")
        print(f"  A+B档: {n_ab}, vetoed: {n_vetoed}")

    # 模块2：日线候选
    print(f"\n[2/5] 生成日线信号候选（A+B档）...")
    daily_candidates = pd.DataFrame()
    if not watch_pool.empty:
        daily_candidates = generate_daily_candidates(watch_pool, target_date, thresholds)
    if daily_candidates.empty:
        print(f"  无日线候选")
    else:
        n_meets = daily_candidates["meets_entry"].sum()
        n_new = daily_candidates["is_new_signal"].sum()
        print(f"  A+B档股票: {len(daily_candidates)}, 满足入场: {n_meets}, 新信号: {n_new}")

    # 模块3：交易指令（即使无新触发点，仍需检查持仓止损/到期）
    print(f"\n[3/5] 生成交易指令...")
    trading_orders = generate_trading_orders(daily_candidates, target_date, portfolio)
    if trading_orders.empty:
        n_buy = 0
        n_sell = 0
    else:
        n_buy = len(trading_orders[trading_orders["action"] == "BUY"])
        n_sell = len(trading_orders[trading_orders["action"] == "SELL"])
    print(f"  买入: {n_buy}, 卖出: {n_sell}")

    # 模块4：持仓监控
    print(f"\n[4/5] 生成持仓监控...")
    portfolio_monitor = generate_portfolio_monitor(portfolio, target_date, watch_pool)
    if not portfolio_monitor.empty:
        for _, row in portfolio_monitor.iterrows():
            status = ""
            if row["hit_stop"]:
                status += "⚠️止损 "
            if row["expired"]:
                status += "⏰到期 "
            print(f"  {row['ts_code']}: 持{row['hold_days']}天, 浮盈={row['pnl_pct']:.2%}, MFE={row['mfe']:.2%}, MAE={row['mae']:.2%} {status}")

    # 模块5：交易清单
    print(f"\n[5/5] 生成交易清单...")
    trading_sheet = generate_trading_sheet(watch_pool, daily_candidates, trading_orders, portfolio, target_date)

    # 更新持仓
    new_portfolio = update_portfolio_state(portfolio, trading_orders, target_date, daily_candidates, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"  持仓更新: {len(portfolio)} → {len(new_portfolio)} 只")
    else:
        print(f"  [试运行] 持仓未更新")

    # 保存输出
    date_str = target_date.strftime("%Y%m%d")
    if not watch_pool.empty:
        watch_pool.to_csv(os.path.join(OUTPUT_DIR, f"watch_pool_{date_str}.csv"), index=False, encoding="utf-8-sig")
    if not daily_candidates.empty:
        daily_candidates.to_csv(os.path.join(OUTPUT_DIR, f"daily_candidates_{date_str}.csv"), index=False, encoding="utf-8-sig")
    if not trading_orders.empty:
        trading_orders.to_csv(os.path.join(OUTPUT_DIR, f"trading_orders_{date_str}.csv"), index=False, encoding="utf-8-sig")
    if not portfolio_monitor.empty:
        portfolio_monitor.to_csv(os.path.join(OUTPUT_DIR, f"portfolio_monitor_{date_str}.csv"), index=False, encoding="utf-8-sig")
    if not trading_sheet.empty:
        trading_sheet.to_csv(os.path.join(OUTPUT_DIR, f"trading_sheet_{date_str}.csv"), index=False, encoding="utf-8-sig")

    # 打印交易清单
    if not trading_sheet.empty:
        print(f"\n{'=' * 80}")
        print(f"交易清单 — {target_date}")
        print(f"{'=' * 80}")
        print(f"  {'代码':>6}  {'名称':>8}  {'动作':>4}  {'优先':>4}  {'档位':>4}  {'周opp':>8}  {'周risk':>8}  {'日opp':>8}  {'日risk':>8}  {'仓位':>4}  {'退出':>10}  {'备注':<20}")
        print(f"  {'-' * 110}")
        for _, row in trading_sheet.iterrows():
            code = row["ts_code"]
            name = str(row.get("stock_name", ""))[:8]
            action = row["action"]
            priority = int(row.get("priority", 0))
            tier = row.get("watch_tier", "")
            w_opp = row.get("weekly_opp", np.nan)
            w_risk = row.get("weekly_risk", np.nan)
            d_opp = row.get("daily_opp", np.nan)
            d_risk = row.get("daily_risk", np.nan)
            weight = row.get("position_weight", 0)
            exit_d = row.get("exit_date", "")
            reason = str(row.get("reason", ""))[:20]

            s_w_opp = f"{w_opp:.4f}" if not np.isnan(w_opp) else "N/A"
            s_w_risk = f"{w_risk:.4f}" if not np.isnan(w_risk) else "N/A"
            s_d_opp = f"{d_opp:.4f}" if not np.isnan(d_opp) else "N/A"
            s_d_risk = f"{d_risk:.4f}" if not np.isnan(d_risk) else "N/A"

            print(f"  {code:>6}  {name:>8}  {action:>4}  {priority:>4}  {tier:>4}  {s_w_opp:>8}  {s_w_risk:>8}  {s_d_opp:>8}  {s_d_risk:>8}  {weight:>4.1f}  {exit_d:>10}  {reason:<20}")

    print(f"\n{'=' * 80}")
    print(f"模拟盘每日决策完成 — {target_date}")
    print(f"{'=' * 80}")

    if args.notify:
        print(f"\n[飞书推送] 格式化消息...")
        header_title, header_template, elements = format_feishu_message(
            target_date, watch_pool, daily_candidates,
            trading_orders, trading_sheet, portfolio_monitor,
        )
        try:
            from app.feishu_notifier import FeishuNotifier
            notifier = FeishuNotifier()
            notifier.send_card(header_title, elements, header_template=header_template)
            print(f"  ✅ 飞书推送成功")
        except Exception as e:
            print(f"  ⚠️ 飞书推送失败: {e}")


if __name__ == "__main__":
    main()
