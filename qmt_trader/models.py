# -*- coding: utf-8 -*-
"""
QMT 交易核心数据模型

Purpose:
    定义 QMT 交易中使用的所有核心数据结构，统一 vis/ 前端和 qmt_trader 后端的数据格式。

Inputs:
    - QMT proxy API 返回的原始 JSON

Outputs:
    - dataclass 实例

How to Run:
    from qmt_trader.models import Asset, Position, Order, Trade, SessionInfo

Side Effects:
    - 无（纯数据结构）
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Asset:
    account_id: str
    cash: float
    frozen_cash: float
    market_value: float
    total_asset: float
    fetch_balance: float


@dataclass
class Position:
    account_id: str
    stock_code: str
    instrument_name: str
    volume: int
    can_use_volume: int
    frozen_volume: int
    on_road_volume: int
    yesterday_volume: int
    open_price: float
    avg_price: float
    last_price: float
    market_value: float
    profit_rate: float
    direction: str


@dataclass
class Order:
    stock_code: str
    order_id: str
    order_sysid: str
    order_time_ms: int
    order_type: int       # 23=限价 42=市价
    order_volume: int
    price_type: int       # 50=限价
    price: float
    traded_volume: int
    traded_price: float
    order_status_code: int
    status_msg: str


@dataclass
class Trade:
    stock_code: str
    traded_volume: int
    traded_price: float
    traded_time_ms: int
    side: str


@dataclass
class SessionInfo:
    session_id: str
    account_id: str
    account_type: str
    is_real: bool
    orders_enabled: bool


@dataclass
class ExecutionResult:
    ts_code: str
    plan_action: str
    result: str            # success / failed / skipped / dry_run
    order_id: str = ""
    order_sysid: str = ""
    error: str = ""
    price: float = 0.0
    volume: int = 0


@dataclass
class PlanValidation:
    is_valid: bool
    message: str
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)