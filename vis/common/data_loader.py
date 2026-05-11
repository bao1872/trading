# -*- coding: utf-8 -*-
"""
Purpose:
    交易工作台数据加载层。只读底层账本和后处理产物，不重复计算逻辑。
    所有数据加载函数引用 stop_experiment.pipeline 的 SSOT 接口。

Inputs:
    - output/holdings/YYYY-MM-DD.parquet
    - output/live/decisions/YYYY-MM-DD.parquet
    - output/live/executions/YYYY-MM-DD.parquet
    - output/predictions/YYYY-MM-DD.parquet
    - output/live/live_equity_curve.csv
    - output/live/live_trade_report.csv
    - output/live/action_plans/YYYY-MM-DD.json
    - output/live/manual_plans/YYYY-MM-DD.parquet

Outputs:
    - 各 load 函数返回 DataFrame 或 dict
    - save_manual_plan() 写入 manual_plans/

How to Run:
    python -m vis.common.data_loader

Examples:
    from vis.common.data_loader import load_latest_date, load_holdings_df
    date = load_latest_date()
    holdings = load_holdings_df(date)

Side Effects:
    - 读取 parquet/CSV 文件
    - save_manual_plan() 写入 output/live/manual_plans/
    - get_stock_name_map() 读取数据库 stock_pools 表
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import glob
import pandas as pd
from datetime import datetime

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, HOLDINGS_DIR, DECISIONS_DIR, EXECUTIONS_DIR,
    PREDICTIONS_DIR, LIVE_DIR,
)

MANUAL_PLANS_DIR = os.path.join(LIVE_DIR, "manual_plans")
ACTION_PLANS_DIR = os.path.join(LIVE_DIR, "action_plans")
EQUITY_CURVE_PATH = os.path.join(LIVE_DIR, "live_equity_curve.csv")
TRADE_REPORT_PATH = os.path.join(LIVE_DIR, "live_trade_report.csv")


def load_available_dates() -> list:
    """获取所有有 holdings 数据的交易日列表（降序）"""
    pattern = os.path.join(HOLDINGS_DIR, "*.parquet")
    files = glob.glob(pattern)
    dates = sorted(
        [os.path.splitext(os.path.basename(f))[0] for f in files],
        reverse=True,
    )
    return dates


def load_latest_date() -> str:
    """获取最新有数据的交易日"""
    dates = load_available_dates()
    return dates[0] if dates else ""


def load_holdings_df(date: str) -> pd.DataFrame:
    """读取持仓账本，返回 DataFrame"""
    path = os.path.join(HOLDINGS_DIR, f"{date}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_decisions_df(date: str) -> pd.DataFrame:
    """读取决策账本，返回 DataFrame"""
    path = os.path.join(DECISIONS_DIR, f"{date}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_executions_df(date: str) -> pd.DataFrame:
    """读取执行账本，返回 DataFrame"""
    path = os.path.join(EXECUTIONS_DIR, f"{date}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_predictions_df(date: str) -> pd.DataFrame:
    """读取预测账本，返回 DataFrame"""
    path = os.path.join(PREDICTIONS_DIR, f"{date}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_equity_curve() -> pd.DataFrame:
    """读取净值曲线 CSV"""
    if not os.path.exists(EQUITY_CURVE_PATH):
        return pd.DataFrame()
    df = pd.read_csv(EQUITY_CURVE_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_trade_report() -> pd.DataFrame:
    """读取交易报告 CSV"""
    if not os.path.exists(TRADE_REPORT_PATH):
        return pd.DataFrame()
    return pd.read_csv(TRADE_REPORT_PATH)


def load_action_plan_json(date: str) -> dict:
    """读取行动计划 JSON"""
    path = os.path.join(ACTION_PLANS_DIR, f"{date}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manual_plan_df(date: str) -> pd.DataFrame:
    """读取人工计划 parquet"""
    path = os.path.join(MANUAL_PLANS_DIR, f"{date}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


def save_manual_plan(date: str, plan_df: pd.DataFrame) -> str:
    """保存人工计划到 manual_plans/YYYY-MM-DD.parquet"""
    os.makedirs(MANUAL_PLANS_DIR, exist_ok=True)
    path = os.path.join(MANUAL_PLANS_DIR, f"{date}.parquet")
    plan_df.to_parquet(path, index=False)
    return path


def get_stock_name_map() -> dict:
    """获取 ts_code -> 股票名称映射（带缓存）"""
    from vis.common.db import get_stock_name_map as _db_get
    return _db_get()


def compute_risk_tags(holdings_df: pd.DataFrame, decisions_df: pd.DataFrame) -> pd.DataFrame:
    """为持仓计算风险标签

    标签逻辑：
    - sell: decisions 中 action=sell
    - risk: buy_cls > 0.70（模型认为继续持有风险高）
    - near_max_hold: days_held >= 17
    - near_stop_loss: cur_ret 在 -6% ~ -7% 之间
    - strong_hold: sell_reg 高 且 buy_cls 低
    """
    if holdings_df.empty:
        return holdings_df

    sell_codes = set()
    if not decisions_df.empty and "action" in decisions_df.columns:
        sell_dec = decisions_df[decisions_df["action"] == "sell"]
        if "ts_code" in sell_dec.columns:
            sell_codes = set(sell_dec["ts_code"].tolist())

    tags = []
    for _, row in holdings_df.iterrows():
        ts_code = row.get("ts_code", row.get("code", ""))
        code_tags = []

        if ts_code in sell_codes:
            code_tags.append("sell")
        else:
            buy_cls = row.get("buy_cls", None)
            if buy_cls is not None and pd.notna(buy_cls) and buy_cls > 0.70:
                code_tags.append("risk")

        days_held = row.get("days_held", 0)
        if pd.notna(days_held) and days_held >= 17:
            code_tags.append("near_max_hold")

        cur_ret = row.get("cur_ret", None)
        if cur_ret is not None and pd.notna(cur_ret) and -0.07 < cur_ret < -0.06:
            code_tags.append("near_stop_loss")

        if not code_tags:
            code_tags.append("strong_hold")

        tags.append(",".join(code_tags))

    result = holdings_df.copy()
    result["risk_tags"] = tags
    return result


def get_prev_date(date: str) -> str:
    """获取前一个交易日"""
    dates = load_available_dates()
    if date in dates:
        idx = dates.index(date)
        if idx + 1 < len(dates):
            return dates[idx + 1]
    sorted_dates = sorted(dates)
    for i, d in enumerate(sorted_dates):
        if d >= date and i > 0:
            return sorted_dates[i - 1]
    return ""


if __name__ == "__main__":
    print("=" * 60)
    print("  data_loader 自测")
    print("=" * 60)

    latest = load_latest_date()
    print(f"  最新交易日: {latest}")

    dates = load_available_dates()
    print(f"  可用交易日数: {len(dates)}")

    if latest:
        holdings = load_holdings_df(latest)
        print(f"  持仓数据: {len(holdings)} 只")
        if not holdings.empty:
            print(f"  持仓列: {list(holdings.columns)}")

        decisions = load_decisions_df(latest)
        print(f"  决策数据: {len(decisions)} 条")
        if not decisions.empty and "action" in decisions.columns:
            print(f"  决策 action 分布: {decisions['action'].value_counts().to_dict()}")

        executions = load_executions_df(latest)
        print(f"  执行数据: {len(executions)} 条")

        predictions = load_predictions_df(latest)
        print(f"  预测数据: {len(predictions)} 条")

    eq = load_equity_curve()
    print(f"  净值曲线: {len(eq)} 行")
    if not eq.empty:
        print(f"  净值曲线列: {list(eq.columns)}")

    tr = load_trade_report()
    print(f"  交易报告: {len(tr)} 行")

    print(f"\n  ✅ data_loader 自测完成")
