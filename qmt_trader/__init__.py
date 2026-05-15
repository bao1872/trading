# -*- coding: utf-8 -*-
"""
qmt_trader — QMT 实盘交易基座

Purpose:
    通过远端 quant-qmt-proxy 的 REST API 对接真实 QMT/MiniQMT，
    提供会话管理、资产/持仓/订单/成交查询、下单、撤单、计划执行等能力。

Usage:
    from qmt_trader import QmtClient, QueryAPI, OrderAPI, TradeExecutor
    from qmt_trader.client import QmtClient
    from qmt_trader.query import QueryAPI
    from qmt_trader.order import OrderAPI
    from qmt_trader.executor import TradeExecutor
"""

from qmt_trader.config import (
    QMT_PROXY_URL,
    QMT_PROXY_API_KEY,
    QMT_ACCOUNT_ID,
    QMT_ACCOUNT_TYPE,
    QMT_TIMEOUT,
    QMT_MAX_RETRIES,
    QMT_MODE,
    MAX_ORDERS_PER_BATCH,
)

from qmt_trader.models import (
    Asset,
    Position,
    Order,
    Trade,
    SessionInfo,
    ExecutionResult,
    PlanValidation,
)

from qmt_trader.client import QmtClient
from qmt_trader.session import SessionManager
from qmt_trader.query import QueryAPI
from qmt_trader.order import OrderAPI
from qmt_trader.executor import TradeExecutor
from qmt_trader.sync import PositionSync

__all__ = [
    "QmtClient",
    "SessionManager",
    "QueryAPI",
    "OrderAPI",
    "TradeExecutor",
    "PositionSync",
    "Asset",
    "Position",
    "Order",
    "Trade",
    "SessionInfo",
    "ExecutionResult",
    "PlanValidation",
    "QMT_PROXY_URL",
    "QMT_PROXY_API_KEY",
    "QMT_ACCOUNT_ID",
    "QMT_ACCOUNT_TYPE",
    "QMT_TIMEOUT",
    "QMT_MAX_RETRIES",
    "QMT_MODE",
    "MAX_ORDERS_PER_BATCH",
]

__version__ = "0.1.0"