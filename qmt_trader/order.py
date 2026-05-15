# -*- coding: utf-8 -*-
"""
QMT 下单与撤单接口

Purpose:
    封装 quant-qmt-proxy 的下单和撤单 REST 接口。

Inputs:
    - QmtClient 实例
    - session_id, stock_code, side, volume, price 等参数

Outputs:
    - 下单返回 Order 对象
    - 撤单返回 bool

How to Run:
    from qmt_trader.client import QmtClient
    from qmt_trader.order import OrderAPI
    client = QmtClient()
    api = OrderAPI(client)
    result = api.submit(session_id, "000001.SZ", "BUY", 100, 11.00)

Examples:
    api = OrderAPI(QmtClient())
    # 限价买入
    api.submit(sid, "000001.SZ", "BUY", 100, 11.00, price_type=50)
    # 市价卖出
    api.submit(sid, "000001.SZ", "SELL", 100, 0, price_type=42)
    # 撤单
    api.cancel(sid, "order_id", market="SZ")

Side Effects:
    - 下单/撤单会实际影响 QMT 账户
    - 请确保在 dry_run=False 时再调用
"""

import logging

from qmt_trader.models import Order

logger = logging.getLogger(__name__)

PRICE_TYPE_LIMIT = 50
PRICE_TYPE_MARKET = 42


class OrderAPI:
    """QMT 下单与撤单"""

    def __init__(self, client):
        self._client = client

    def submit(self, session_id, stock_code, side, volume,
               price=0.0, price_type=PRICE_TYPE_LIMIT,
               strategy_name="", order_remark=""):
        """POST /api/v1/trading/sessions/{id}/orders

        Args:
            session_id: 交易会话 ID
            stock_code: 股票代码，如 "000001.SZ"
            side: "BUY" 或 "SELL"
            volume: 委托数量（股）
            price: 委托价格（市价时传 0）
            price_type: 50=限价 42=市价
            strategy_name: 策略备注
            order_remark: 订单备注

        Returns:
            Order
        """
        body = {
            "stock_code": stock_code,
            "price_type": price_type,
            "side": side.upper(),
            "volume": volume,
            "price": price,
        }
        if strategy_name:
            body["strategy_name"] = strategy_name
        if order_remark:
            body["order_remark"] = order_remark

        data = self._client.post(
            f"/api/v1/trading/sessions/{session_id}/orders",
            body,
        )
        return Order(
            stock_code=stock_code,
            order_id=str(data.get("order_id", "-1")),
            order_sysid=str(data.get("order_sysid", "")),
            order_time_ms=int(data.get("order_time_ms", 0)),
            order_type=price_type,
            order_volume=volume,
            price_type=price_type,
            price=price,
            traded_volume=int(data.get("traded_volume", 0)),
            traded_price=float(data.get("traded_price", 0)),
            order_status_code=int(data.get("order_status_code", 0)),
            status_msg=data.get("status_msg", ""),
        )

    def cancel(self, session_id, order_id, market="", order_sysid=""):
        """POST /api/v1/trading/sessions/{id}/cancel

        Args:
            session_id: 交易会话 ID
            order_id: 订单 ID（QMT 内部分配）
            market: 市场 "SZ" / "SH"
            order_sysid: 系统订单 ID（可选）

        Returns:
            bool 撤单是否成功
        """
        body = {
            "order_id": str(order_id),
            "market": market,
            "order_sysid": order_sysid or "",
        }
        data = self._client.post(
            f"/api/v1/trading/sessions/{session_id}/cancel",
            body,
        )
        success = data.get("success", False) if isinstance(data, dict) else False
        if not success:
            logger.warning(f"撤单返回失败: order_id={order_id} resp={data}")
        return success

    @staticmethod
    def infer_market(stock_code):
        """从股票代码推断市场

        000001.SZ → "SZ"
        600000.SH → "SH"
        """
        if not stock_code or "." not in stock_code:
            return ""
        suffix = stock_code.split(".")[-1].upper()
        if suffix in ("SZ", "SH", "BJ"):
            return suffix
        return ""


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

    if not sess.orders_enabled:
        print(f"orders_enabled=False，跳过下单测试")
        mgr.close(sid)
        exit(0)

    api = OrderAPI(client)

    print("测试下单（限价买入 000001.SZ x100 @11.00）...")
    order = api.submit(sid, "000001.SZ", "BUY", 100, 11.00)
    print(f"  结果: order_id={order.order_id} status={order.status_msg}")

    if order.status_msg == "submitted":
        market = api.infer_market("000001.SZ")
        print(f"测试撤单: order_id={order.order_id} market={market}")
        ok = api.cancel(sid, order.order_id, market=market)
        print(f"  撤单结果: {'成功' if ok else '失败'}")

    mgr.close(sid)
    print("完成")