# -*- coding: utf-8 -*-
"""
QMT 查询接口（资产/持仓/订单/成交）

Purpose:
    封装 quant-qmt-proxy 的查询类 REST 接口，返回结构化数据。

Inputs:
    - QmtClient 实例
    - session_id

Outputs:
    - Asset / list[Position] / list[Order] / list[Trade] / SessionInfo

How to Run:
    from qmt_trader.client import QmtClient
    from qmt_trader.query import QueryAPI
    client = QmtClient()
    api = QueryAPI(client)
    asset = api.get_asset(session_id)

Examples:
    api = QueryAPI(QmtClient())
    positions = api.get_positions(session_id)
    for p in positions:
        print(p.stock_code, p.volume, p.profit_rate)

Side Effects:
    - 每次调用产生 HTTP 请求
"""

import logging

from qmt_trader.models import Asset, Position, Order, Trade, SessionInfo

logger = logging.getLogger(__name__)


class QueryAPI:
    """QMT 查询接口"""

    def __init__(self, client):
        self._client = client

    def get_asset(self, session_id):
        """GET /api/v1/trading/sessions/{id}/asset

        Returns:
            Asset
        """
        data = self._client.get(f"/api/v1/trading/sessions/{session_id}/asset")
        return Asset(
            account_id=data.get("account_id", ""),
            cash=float(data.get("cash", 0)),
            frozen_cash=float(data.get("frozen_cash", 0)),
            market_value=float(data.get("market_value", 0)),
            total_asset=float(data.get("total_asset", 0)),
            fetch_balance=float(data.get("fetch_balance", 0)),
        )

    def get_positions(self, session_id):
        """GET /api/v1/trading/sessions/{id}/positions

        Returns:
            list[Position]
        """
        data = self._client.get(f"/api/v1/trading/sessions/{session_id}/positions")
        items = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        return [
            Position(
                account_id=item.get("account_id", ""),
                stock_code=item.get("stock_code", ""),
                instrument_name=item.get("instrument_name", ""),
                volume=int(item.get("volume", 0)),
                can_use_volume=int(item.get("can_use_volume", 0)),
                frozen_volume=int(item.get("frozen_volume", 0)),
                on_road_volume=int(item.get("on_road_volume", 0)),
                yesterday_volume=int(item.get("yesterday_volume", 0)),
                open_price=float(item.get("open_price", 0)),
                avg_price=float(item.get("avg_price", 0)),
                last_price=float(item.get("last_price", 0)),
                market_value=float(item.get("market_value", 0)),
                profit_rate=float(item.get("profit_rate", 0)),
                direction=str(item.get("direction", "")),
            )
            for item in items
        ]

    def get_orders(self, session_id):
        """GET /api/v1/trading/sessions/{id}/orders

        Returns:
            list[Order]
        """
        data = self._client.get(f"/api/v1/trading/sessions/{session_id}/orders")
        items = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        return [
            Order(
                stock_code=item.get("stock_code", ""),
                order_id=str(item.get("order_id", "")),
                order_sysid=str(item.get("order_sysid", "")),
                order_time_ms=int(item.get("order_time_ms", 0)),
                order_type=int(item.get("order_type", 0)),
                order_volume=int(item.get("order_volume", 0)),
                price_type=int(item.get("price_type", 0)),
                price=float(item.get("price", 0)),
                traded_volume=int(item.get("traded_volume", 0)),
                traded_price=float(item.get("traded_price", 0)),
                order_status_code=int(item.get("order_status_code", 0)),
                status_msg=item.get("status_msg", ""),
            )
            for item in items
        ]

    def get_trades(self, session_id):
        """GET /api/v1/trading/sessions/{id}/trades

        Returns:
            list[Trade]
        """
        data = self._client.get(f"/api/v1/trading/sessions/{session_id}/trades")
        items = data.get("items", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        return [
            Trade(
                stock_code=item.get("stock_code", ""),
                traded_volume=int(item.get("traded_volume", 0)),
                traded_price=float(item.get("traded_price", 0)),
                traded_time_ms=int(item.get("traded_time_ms", 0)),
                side=item.get("side", item.get("direction", "")),
            )
            for item in items
        ]

    def get_session(self, session_id):
        """GET /api/v1/trading/sessions/{id}

        Returns:
            SessionInfo
        """
        data = self._client.get(f"/api/v1/trading/sessions/{session_id}")
        return SessionInfo(
            session_id=data.get("session_id", session_id),
            account_id=data.get("account_id", ""),
            account_type=data.get("account_type", ""),
            is_real=data.get("is_real", False),
            orders_enabled=data.get("orders_enabled", False),
        )


if __name__ == "__main__":
    from qmt_trader.client import QmtClient
    from qmt_trader.session import SessionManager

    client = QmtClient()
    if not client.health():
        print("QMT 服务不可达，跳过测试")
        exit(0)

    mgr = SessionManager(client)
    sess = mgr.open()
    sid = sess.session_id
    print(f"会话: {sid}")

    api = QueryAPI(client)

    asset = api.get_asset(sid)
    print(f"资产: 总={asset.total_asset:.2f}, 现金={asset.cash:.2f}")

    positions = api.get_positions(sid)
    print(f"持仓: {len(positions)} 只")
    for p in positions[:3]:
        print(f"  {p.stock_code} {p.instrument_name} x{p.volume} 盈亏率={p.profit_rate:.2%}")

    orders = api.get_orders(sid)
    print(f"委托: {len(orders)} 条")

    trades = api.get_trades(sid)
    print(f"成交: {len(trades)} 条")

    mgr.close(sid)
    print("完成")