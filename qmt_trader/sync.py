# -*- coding: utf-8 -*-
"""
QMT 持仓同步到本地

Purpose:
    从 QMT 拉取真实持仓，写入本地 parquet 文件。
    用于前端展示和历史回溯。

Inputs:
    - QueryAPI 实例
    - session_id
    - output_dir

Outputs:
    - holdings/YYYY-MM-DD.parquet

How to Run:
    from qmt_trader.sync import PositionSync
    syncer = PositionSync(query_api)
    syncer.sync_positions(session_id, "/root/trading/stop_experiment/output/holdings")

Examples:
    syncer = PositionSync(QueryAPI(client))
    path = syncer.sync_positions(sid, OUTPUT_DIR)
    print(f"持仓已同步到: {path}")

Side Effects:
    - 写入 parquet 文件到本地磁盘
    - 不自带定时任务，需外部调度
"""

import os
import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class PositionSync:
    """从 QMT 拉取真实持仓写入本地 parquet"""

    def __init__(self, query_api):
        self._query = query_api

    def sync_positions(self, session_id, output_dir):
        """拉取 QMT 持仓 → holdings/YYYY-MM-DD.parquet

        Args:
            session_id: 交易会话 ID
            output_dir: 输出目录（如 .../stop_experiment/output/holdings）

        Returns:
            str 写入的文件路径
        """
        positions = self._query.get_positions(session_id)
        if not positions:
            logger.warning("QMT 持仓为空，跳过同步")
            return ""

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
                "account_id": p.account_id,
                "sync_time": datetime.now(),
            })

        df = pd.DataFrame(rows)
        today = datetime.now().strftime("%Y-%m-%d")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{today}.parquet")
        df.to_parquet(filepath, index=False)
        logger.info(f"持仓同步完成: {len(df)} 条 → {filepath}")
        return filepath

    def compare(self, session_id, local_df):
        """对比 QMT 真实持仓 vs 本地 DataFrame

        Args:
            session_id: 交易会话 ID
            local_df: 本地持仓 DataFrame（需含 ts_code 列）

        Returns:
            pd.DataFrame 差异对比结果
        """
        positions = self._query.get_positions(session_id)
        qmt_map = {p.stock_code: p for p in positions}
        local_map = {}
        if not local_df.empty and "ts_code" in local_df.columns:
            local_map = local_df.set_index("ts_code").to_dict("index")

        all_codes = set(list(qmt_map.keys()) + list(local_map.keys()))
        diff_rows = []
        for code in sorted(all_codes):
            in_qmt = code in qmt_map
            in_local = code in local_map
            qmt_vol = qmt_map[code].volume if in_qmt else 0
            local_vol = local_map[code].get("volume", 0) if in_local else 0
            diff_rows.append({
                "ts_code": code,
                "in_qmt": "yes" if in_qmt else "no",
                "in_local": "yes" if in_local else "no",
                "qmt_volume": qmt_vol,
                "local_volume": local_vol,
                "diff": qmt_vol - local_vol,
            })
        return pd.DataFrame(diff_rows)


if __name__ == "__main__":
    from qmt_trader.client import QmtClient
    from qmt_trader.session import SessionManager
    from qmt_trader.query import QueryAPI

    client = QmtClient()
    if not client.health():
        print("QMT 服务不可达，跳过测试")
        exit(0)

    mgr = SessionManager(client)
    sess = mgr.open()
    api = QueryAPI(client)
    syncer = PositionSync(api)

    test_dir = "/tmp/qmt_sync_test"
    path = syncer.sync_positions(sess.session_id, test_dir)
    if path:
        df = pd.read_parquet(path)
        print(f"同步成功: {len(df)} 只持仓")
        print(df[["ts_code", "volume", "market_value"]].to_string())
        syncer.compare(sess.session_id, df)

    mgr.close(sess.session_id)
    print("完成")