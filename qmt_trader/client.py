# -*- coding: utf-8 -*-
"""
qmt_trader HTTP 客户端

Purpose:
    统管所有与远端 quant-qmt-proxy 的 HTTP 通信。
    处理认证 Bearer Token、超时、重试、错误映射。

Inputs:
    - QMT_PROXY_URL, QMT_PROXY_API_KEY, QMT_TIMEOUT, QMT_MAX_RETRIES（配置）

Outputs:
    - get/post 返回 dict（已提取 data 字段）
    - health() 返回 bool

How to Run:
    from qmt_trader.client import QmtClient
    client = QmtClient()
    data = client.get("/api/v1/trading/sessions/xxx/asset")

Examples:
    client = QmtClient()
    print(client.health())
    asset = client.get(f"/api/v1/trading/sessions/{sid}/asset")
    orders = client.post(f"/api/v1/trading/sessions/{sid}/orders", body)

Side Effects:
    - 每次调用产生 HTTP 请求
    - 不修改本地数据
"""

import time
import logging

import requests

from qmt_trader.config import (
    QMT_PROXY_URL,
    QMT_PROXY_API_KEY,
    QMT_TIMEOUT,
    QMT_MAX_RETRIES,
)

logger = logging.getLogger(__name__)


class QmtClientError(Exception):
    """QMT 客户端基础异常"""


class QmtConnectionError(QmtClientError):
    """连接错误"""


class QmtAuthError(QmtClientError):
    """认证错误 (401/403)"""


class QmtApiError(QmtClientError):
    """API 业务错误"""


class QmtTimeoutError(QmtClientError):
    """请求超时"""


class QmtClient:
    """qmt-proxy REST API 客户端"""

    def __init__(self, base_url=None, api_key=None, timeout=None, max_retries=None):
        self.base_url = (base_url or QMT_PROXY_URL).rstrip("/")
        self.api_key = api_key or QMT_PROXY_API_KEY
        self.timeout = timeout or QMT_TIMEOUT
        self.max_retries = max_retries or QMT_MAX_RETRIES
        self._session = self._build_session()

    def _build_session(self):
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        return session

    def _url(self, path):
        return f"{self.base_url}{path}"

    def _request(self, method, path, body=None):
        url = self._url(path)
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if method == "GET":
                    resp = self._session.get(url, timeout=self.timeout)
                elif method == "POST":
                    resp = self._session.post(url, json=body, timeout=self.timeout)
                elif method == "DELETE":
                    resp = self._session.delete(url, timeout=self.timeout)
                else:
                    raise ValueError(f"不支持的 HTTP 方法: {method}")

                if resp.status_code == 401:
                    raise QmtAuthError(f"认证失败: API Key 无效或过期 | url={url}")
                if resp.status_code == 403:
                    detail = resp.json().get("detail", "Forbidden") if resp.text else "Forbidden"
                    raise QmtAuthError(f"权限不足: {detail} | url={url}")

                payload = resp.json()
                success = payload.get("success", resp.status_code == 200)
                if not success:
                    msg = payload.get("message", payload.get("detail", str(payload)))
                    raise QmtApiError(f"API 业务错误: {msg} | status={resp.status_code} | url={url}")

                return payload.get("data", payload)

            except requests.exceptions.Timeout as e:
                last_error = QmtTimeoutError(f"请求超时: {url} | attempt={attempt + 1}")
                logger.warning(str(last_error))
            except requests.exceptions.ConnectionError as e:
                last_error = QmtConnectionError(f"连接失败: {url} | {e}")
                logger.warning(str(last_error))
            except QmtClientError:
                raise

            if attempt < self.max_retries:
                wait = 2 ** attempt
                time.sleep(wait)

        raise last_error

    def get(self, path):
        return self._request("GET", path)

    def post(self, path, body=None):
        return self._request("POST", path, body or {})

    def delete(self, path):
        return self._request("DELETE", path)

    def health(self):
        try:
            resp = self._session.get(
                self._url("/health/ready"),
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False


if __name__ == "__main__":
    client = QmtClient()
    connected = client.health()
    print(f"QMT 服务连接状态: {'🟢 正常' if connected else '🔴 不可达'}")
    print(f"目标: {client.base_url}")