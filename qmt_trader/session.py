# -*- coding: utf-8 -*-
"""
交易会话管理

Purpose:
    管理 quant-qmt-proxy 交易会话的生命周期：创建、查询、关闭。

Inputs:
    - QMT_ACCOUNT_ID, QMT_ACCOUNT_TYPE（配置）
    - QmtClient 实例

Outputs:
    - SessionInfo 对象

How to Run:
    from qmt_trader.client import QmtClient
    from qmt_trader.session import SessionManager
    client = QmtClient()
    mgr = SessionManager(client)
    session_info = mgr.open()
    mgr.close(session_info.session_id)

Examples:
    mgr = SessionManager(QmtClient(), account_id="8884903985")
    sess = mgr.open()
    print(sess.session_id)

Side Effects:
    - 创建会话会调用 QMT proxy（需要网络连接）
    - 关闭会话会释放 QMT 侧资源
"""

import logging

from qmt_trader.config import QMT_ACCOUNT_ID, QMT_ACCOUNT_TYPE
from qmt_trader.models import SessionInfo

logger = logging.getLogger(__name__)


class SessionManager:
    """管理 quant-qmt-proxy 交易会话生命周期"""

    def __init__(self, client, account_id=None, account_type=None):
        self._client = client
        self._account_id = account_id or QMT_ACCOUNT_ID
        self._account_type = account_type or QMT_ACCOUNT_TYPE

    def open(self, account_id=None, account_type=None):
        """POST /api/v1/trading/sessions 创建交易会话

        Returns:
            SessionInfo
        """
        aid = account_id or self._account_id
        atype = account_type or self._account_type
        body = {"account_id": aid, "account_type": atype}
        data = self._client.post("/api/v1/trading/sessions", body)
        return SessionInfo(
            session_id=data.get("session_id", ""),
            account_id=data.get("account_id", aid),
            account_type=data.get("account_type", atype),
            is_real=data.get("is_real", False),
            orders_enabled=data.get("orders_enabled", False),
        )

    def close(self, session_id):
        """DELETE /api/v1/trading/sessions/{id} 关闭会话"""
        return self._client.delete(f"/api/v1/trading/sessions/{session_id}")

    def get(self, session_id):
        """GET /api/v1/trading/sessions/{id} 查询会话"""
        data = self._client.get(f"/api/v1/trading/sessions/{session_id}")
        return SessionInfo(
            session_id=data.get("session_id", session_id),
            account_id=data.get("account_id", ""),
            account_type=data.get("account_type", ""),
            is_real=data.get("is_real", False),
            orders_enabled=data.get("orders_enabled", False),
        )

    @property
    def account_id(self):
        return self._account_id


if __name__ == "__main__":
    from qmt_trader.client import QmtClient
    client = QmtClient()
    mgr = SessionManager(client)
    if client.health():
        sess = mgr.open()
        print(f"会话已创建: {sess.session_id}")
        print(f"  orders_enabled: {sess.orders_enabled}")
        info = mgr.get(sess.session_id)
        print(f"会话状态确认: {info.is_real=}, {info.orders_enabled=}")
        mgr.close(sess.session_id)
        print("会话已关闭")
    else:
        print("QMT 服务不可达，跳过测试")