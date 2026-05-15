# -*- coding: utf-8 -*-
"""
qmt_trader Mock 模式

Purpose:
    在本地开发/测试环境中模拟 QMT 代理数据，无需连接远程 QMT 服务器。
    当 QMT_MODE=mock 时自动使用，或显式调用。

Inputs:
    - 无（数据全部内置模拟）

Outputs:
    - 模拟的 Asset / Position / Order / Trade 数据

How to Run:
    from qmt_trader.mock import mock_client, mock_positions, mock_asset, mock_orders, mock_trades
    client = mock_client()

Examples:
    from qmt_trader import QMT_MODE
    from qmt_trader.client import QmtClient
    from qmt_trader.mock import MockQmtClient

    if QMT_MODE == "mock":
        client = MockQmtClient()
    else:
        client = QmtClient()

Side Effects:
    - 无（纯内存数据）
"""

import time
from dataclasses import asdict

from qmt_trader.models import Asset, Position, Order, Trade, SessionInfo


def mock_asset():
    return Asset(
        account_id="8884903985",
        cash=500000.00,
        frozen_cash=0.0,
        market_value=667038.67,
        total_asset=1167038.67,
        fetch_balance=500000.00,
    )


def mock_positions():
    return [
        Position(
            account_id="8884903985",
            stock_code="301381.SZ",
            instrument_name="浙江荣泰",
            volume=5200,
            can_use_volume=5200,
            frozen_volume=0,
            on_road_volume=0,
            yesterday_volume=5200,
            open_price=35.20,
            avg_price=35.20,
            last_price=37.50,
            market_value=195000.00,
            profit_rate=0.065,
            direction="1",
        ),
        Position(
            account_id="8884903985",
            stock_code="603060.SH",
            instrument_name="麦加芯彩",
            volume=3000,
            can_use_volume=3000,
            frozen_volume=0,
            on_road_volume=0,
            yesterday_volume=3000,
            open_price=52.00,
            avg_price=52.00,
            last_price=55.80,
            market_value=167400.00,
            profit_rate=0.073,
            direction="1",
        ),
        Position(
            account_id="8884903985",
            stock_code="688365.SH",
            instrument_name="光云科技",
            volume=8000,
            can_use_volume=8000,
            frozen_volume=0,
            on_road_volume=0,
            yesterday_volume=8000,
            open_price=18.50,
            avg_price=18.50,
            last_price=20.10,
            market_value=160800.00,
            profit_rate=0.086,
            direction="1",
        ),
    ]


def mock_orders():
    now_ms = int(time.time() * 1000)
    return [
        Order(
            stock_code="000001.SZ",
            order_id="mock_order_001",
            order_sysid="sys_mock_001",
            order_time_ms=now_ms - 3600000,
            order_type=23,
            order_volume=100,
            price_type=50,
            price=11.00,
            traded_volume=0,
            traded_price=0.0,
            order_status_code=50,
            status_msg="submitted",
        ),
    ]


def mock_trades():
    now_ms = int(time.time() * 1000)
    return [
        Trade(
            stock_code="000001.SZ",
            traded_volume=100,
            traded_price=11.25,
            traded_time_ms=now_ms - 86400000,
            side="BUY",
        ),
    ]


def mock_session():
    return SessionInfo(
        session_id="session_mock_8884903985_0000000000",
        account_id="8884903985",
        account_type="STOCK",
        is_real=False,
        orders_enabled=False,
    )


class MockQmtClient:
    """Mock QMT 客户端，接口与 QmtClient 一致，返回模拟数据"""

    def __init__(self):
        self.base_url = "mock://localhost"
        self.api_key = "mock-key"
        self._connected = True
        self._session_id = None

    def get(self, path):
        if "/asset" in path:
            return asdict(mock_asset())
        if "/positions" in path:
            return {"items": [asdict(p) for p in mock_positions()]}
        if "/orders" in path:
            return {"items": [asdict(o) for o in mock_orders()]}
        if "/trades" in path:
            return [asdict(t) for t in mock_trades()]
        if "/sessions/" in path and "/" not in path.split("/sessions/")[1]:
            return asdict(mock_session())
        return {"message": "mock ok"}

    def post(self, path, body=None):
        if "/sessions" in path and "cancel" not in path:
            self._session_id = "session_mock_8884903985_0000000000"
            return {
                "session_id": self._session_id,
                "account_id": "8884903985",
                "account_type": "STOCK",
                "is_real": False,
                "orders_enabled": False,
            }
        if "/orders" in path:
            return {
                "order_id": "mock_order_new",
                "order_sysid": "sys_mock_new",
                "stock_code": body.get("stock_code", "000001.SZ"),
                "order_status_code": 50,
                "status_msg": "submitted",
            }
        if "/cancel" in path:
            return {"success": True, "message": "撤单成功 (mock)"}
        return {"success": True, "message": "mock ok"}

    def delete(self, path):
        return {"message": "session closed (mock)"}

    def health(self):
        return self._connected