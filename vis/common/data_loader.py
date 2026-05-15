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


# ==================== QMT 实盘数据加载 ====================

from qmt_trader.client import QmtClient
from qmt_trader.session import SessionManager
from qmt_trader.query import QueryAPI
from qmt_trader.order import OrderAPI
from qmt_trader.executor import TradeExecutor
from qmt_trader.config import QMT_MODE, QMT_PROXY_URL

_qmt_client: QmtClient | None = None
_qmt_session_id: str | None = None


def get_qmt_client() -> QmtClient:
    """懒加载 QMT 客户端（全局单例）

    根据 QMT_MODE 返回真实客户端或 Mock 客户端。
    供 Streamlit app 和页面使用。

    Returns:
        QmtClient 或 MockQmtClient
    """
    global _qmt_client
    if _qmt_client is not None:
        return _qmt_client
    if QMT_MODE == "mock":
        from qmt_trader.mock import MockQmtClient
        _qmt_client = MockQmtClient()
    else:
        _qmt_client = QmtClient()
    return _qmt_client


def get_qmt_session_id() -> str:
    """懒加载 QMT 交易会话 ID（全局单例）

    Returns:
        str session_id，空字符串表示连接失败
    """
    global _qmt_session_id
    if _qmt_session_id is not None:
        return _qmt_session_id
    try:
        client = get_qmt_client()
        if not client.health():
            _qmt_session_id = ""
            return ""
        mgr = SessionManager(client)
        sess = mgr.open()
        _qmt_session_id = sess.session_id
        return _qmt_session_id
    except Exception:
        _qmt_session_id = ""
        return ""


def refresh_qmt_connection() -> dict:
    """强制刷新 QMT 连接，重新创建客户端和会话

    Returns:
        dict: {"connected": bool, "session_id": str, "error": str}
    """
    global _qmt_client, _qmt_session_id
    _qmt_client = None
    _qmt_session_id = None
    try:
        client = get_qmt_client()
        if not client.health():
            return {"connected": False, "session_id": "", "error": "服务不可达"}
        mgr = SessionManager(client)
        sess = mgr.open()
        _qmt_session_id = sess.session_id
        return {
            "connected": True,
            "session_id": sess.session_id,
            "error": "",
        }
    except Exception as e:
        return {"connected": False, "session_id": "", "error": str(e)}


def load_qmt_asset() -> dict:
    """从 QMT 加载资产信息

    Returns:
        dict: {"total_asset": float, "cash": float, "market_value": float, ...}
              连接失败返回空 dict
    """
    sid = get_qmt_session_id()
    if not sid:
        return {}
    try:
        api = QueryAPI(get_qmt_client())
        asset = api.get_asset(sid)
        return {
            "account_id": asset.account_id,
            "total_asset": asset.total_asset,
            "cash": asset.cash,
            "frozen_cash": asset.frozen_cash,
            "market_value": asset.market_value,
            "fetch_balance": asset.fetch_balance,
        }
    except Exception:
        return {}


def load_qmt_positions_df() -> pd.DataFrame:
    """从 QMT 加载持仓，转换为 DataFrame

    Returns:
        DataFrame 含列: ts_code, instrument_name, volume, can_use_volume,
                      avg_price, last_price, market_value, profit_rate
    """
    sid = get_qmt_session_id()
    if not sid:
        return pd.DataFrame()
    try:
        api = QueryAPI(get_qmt_client())
        positions = api.get_positions(sid)
        if not positions:
            return pd.DataFrame()
        rows = []
        for p in positions:
            rows.append({
                "ts_code": p.stock_code,
                "instrument_name": p.instrument_name,
                "volume": p.volume,
                "can_use_volume": p.can_use_volume,
                "avg_price": p.avg_price,
                "last_price": p.last_price,
                "market_value": p.market_value,
                "profit_rate": p.profit_rate,
                "direction": p.direction,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def load_qmt_orders_df() -> pd.DataFrame:
    """从 QMT 加载委托，转换为 DataFrame

    Returns:
        DataFrame 含列: stock_code, order_id, order_sysid, order_type,
                      order_volume, price, traded_volume, status_msg, ...
    """
    sid = get_qmt_session_id()
    if not sid:
        return pd.DataFrame()
    try:
        api = QueryAPI(get_qmt_client())
        orders = api.get_orders(sid)
        if not orders:
            return pd.DataFrame()
        rows = []
        for o in orders:
            rows.append({
                "stock_code": o.stock_code,
                "order_id": o.order_id,
                "order_sysid": o.order_sysid,
                "order_type": o.order_type,
                "order_volume": o.order_volume,
                "price": o.price,
                "traded_volume": o.traded_volume,
                "traded_price": o.traded_price,
                "order_status_code": o.order_status_code,
                "status_msg": o.status_msg,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def load_qmt_trades_df() -> pd.DataFrame:
    """从 QMT 加载成交，转换为 DataFrame

    Returns:
        DataFrame 含列: stock_code, traded_volume, traded_price, side, ...
    """
    sid = get_qmt_session_id()
    if not sid:
        return pd.DataFrame()
    try:
        api = QueryAPI(get_qmt_client())
        trades = api.get_trades(sid)
        if not trades:
            return pd.DataFrame()
        rows = []
        for t in trades:
            rows.append({
                "stock_code": t.stock_code,
                "traded_volume": t.traded_volume,
                "traded_price": t.traded_price,
                "side": t.side,
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def get_qmt_executor() -> TradeExecutor:
    """获取 TradeExecutor 实例（供页面调用）"""
    return TradeExecutor(get_qmt_client())


def load_real_holdings_with_risk_tags() -> pd.DataFrame:
    """从 QMT 加载真实持仓，并融合 AI 风险标签

    融合逻辑:
    1. 从 QMT 加载真实持仓
    2. 从最新 decisions 获取 AI 建议
    3. 计算风险标签

    Returns:
        DataFrame 含 ts_code, instrument_name, volume, ..., risk_tags
    """
    positions_df = load_qmt_positions_df()
    if positions_df.empty:
        return positions_df

    try:
        latest = load_latest_date()
        if latest:
            decisions_df = load_decisions_df(latest)
            positions_df = compute_risk_tags(positions_df, decisions_df)
    except Exception:
        positions_df["risk_tags"] = ""

    return positions_df


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

    print("\n--- QMT 实盘数据 ---")
    conn = refresh_qmt_connection()
    print(f"  QMT 连接: {'🟢 已连接' if conn['connected'] else '🔴 不可达'}")
    if conn["connected"]:
        asset = load_qmt_asset()
        if asset:
            print(f"  总资产: ¥{asset['total_asset']:,.2f}")
            print(f"  现金: ¥{asset['cash']:,.2f}")
        positions = load_qmt_positions_df()
        print(f"  持仓: {len(positions)} 只")
        orders = load_qmt_orders_df()
        print(f"  委托: {len(orders)} 条")
        trades = load_qmt_trades_df()
        print(f"  成交: {len(trades)} 条")

    print(f"\n  ✅ data_loader 自测完成")
