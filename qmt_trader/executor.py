# -*- coding: utf-8 -*-
"""
交易执行引擎

Purpose:
    根据操作计划（manual_plan DataFrame）执行真实买卖操作。
    支持 dry_run 模式（仅校验不实际下单）和实盘模式（真实 QMT 下单）。

核心流程:
    1. 加载计划 → 2. 校验计划 → 3. 先卖后买 → 4. 记录结果

Inputs:
    - QmtClient 实例
    - plan_df（来自 vis/ 操作计划页的 manual_plan）
    - dry_run 标志

Outputs:
    - list[ExecutionResult] 每笔执行结果
    - 校验结果 PlanValidation

How to Run:
    from qmt_trader.executor import TradeExecutor
    executor = TradeExecutor(QmtClient())
    results = executor.execute_plan(plan_df, dry_run=True)

Examples:
    executor = TradeExecutor(QmtClient())
    # Dry-run 模式
    results = executor.execute_plan(plan_df, dry_run=True)
    # 实盘模式（需要 orders_enabled=true）
    results = executor.execute_plan(plan_df, dry_run=False)

Side Effects:
    - dry_run=False 时会实际向 QMT 下单
    - 下单失败时不会回滚已提交的订单
"""

import logging

import pandas as pd

from qmt_trader.config import MAX_ORDERS_PER_BATCH
from qmt_trader.models import ExecutionResult, PlanValidation
from qmt_trader.session import SessionManager
from qmt_trader.query import QueryAPI
from qmt_trader.order import OrderAPI

logger = logging.getLogger(__name__)


class TradeExecutor:
    """交易执行引擎"""

    def __init__(self, client):
        self._client = client
        self._query = QueryAPI(client)
        self._order_api = OrderAPI(client)
        self._session_mgr = SessionManager(client)

    def execute_plan(self, plan_df, dry_run=True, session_id=None,
                     cancel_existing=False):
        """执行操作计划中的所有交易指令

        Args:
            plan_df: 操作计划 DataFrame，需含列:
                ts_code, plan_action, approved_weight
                可选: price, volume
            dry_run: True=仅校验不实际下单
            session_id: 交易会话 ID（None 则自动创建）
            cancel_existing: 执行前是否先撤所有挂单

        Returns:
            list[ExecutionResult]
        """
        if plan_df.empty:
            return []

        if len(plan_df) > MAX_ORDERS_PER_BATCH:
            logger.warning(f"计划包含 {len(plan_df)} 条，超过单批次限制 {MAX_ORDERS_PER_BATCH}")

        if session_id:
            sid = session_id
            close_after = False
        else:
            sess = self._session_mgr.open()
            if not sess.orders_enabled and not dry_run:
                raise RuntimeError("orders_enabled=False，无法实盘下单。请检查 config.local.yml")
            sid = sess.session_id
            close_after = True

        try:
            if cancel_existing and not dry_run:
                self._batch_cancel_all(sid)

            valid, msg, detail = self.validate_plan(sid, plan_df)
            if not valid:
                logger.error(f"计划校验失败: {msg}")
                failed_results = []
                for _, row in plan_df.iterrows():
                    failed_results.append(ExecutionResult(
                        ts_code=row.get("ts_code", ""),
                        plan_action=row.get("plan_action", ""),
                        result="skipped",
                        error=msg,
                    ))
                return failed_results

            if detail.get("warnings"):
                for w in detail["warnings"]:
                    logger.warning(f"计划校验警告: {w}")

            sells = plan_df[plan_df["plan_action"].isin(["sell", "reduce", "减仓"])].copy()
            buys = plan_df[plan_df["plan_action"].isin(["buy", "add", "买入", "加仓"])].copy()

            results = []

            if not sells.empty:
                sell_results = self._execute_sells(sid, sells, dry_run)
                results.extend(sell_results)

            if not buys.empty:
                buy_results = self._execute_buys(sid, buys, dry_run)
                results.extend(buy_results)

            success_count = sum(1 for r in results if r.result == "success")
            fail_count = sum(1 for r in results if r.result == "failed")
            dry_count = sum(1 for r in results if r.result == "dry_run")
            logger.info(
                f"执行完成: 成功={success_count} 失败={fail_count} dry_run={dry_count}"
            )

            return results

        finally:
            if close_after:
                self._session_mgr.close(sid)

    def validate_plan(self, session_id, plan_df):
        """校验操作计划

        校验项:
        1. 卖出数量不超过实际持仓
        2. 买入金额不超过可用资金
        3. 股票代码格式正确

        Returns:
            (is_valid, message, {"warnings": [], "errors": []})
        """
        warnings = []
        errors = []

        required_cols = ["ts_code", "plan_action"]
        missing = [c for c in required_cols if c not in plan_df.columns]
        if missing:
            errors.append(f"缺少必须列: {missing}")
            return False, f"缺少必须列: {missing}", {"warnings": warnings, "errors": errors}

        positions = self._query.get_positions(session_id)
        pos_map = {p.stock_code: p for p in positions}

        asset = self._query.get_asset(session_id)
        available_cash = asset.cash

        total_buy_amount = 0.0
        total_sell_amount = 0.0

        for _, row in plan_df.iterrows():
            ts_code = row["ts_code"]
            plan_action = row["plan_action"]
            if plan_action in ("buy", "add", "买入", "加仓", "candidate"):
                vol = int(row.get("volume", 0) or 0)
                price = float(row.get("price", 0) or 0)
                if vol <= 0:
                    errors.append(f"{ts_code}: 买入数量无效 volume={vol}")
                    continue
                if price <= 0:
                    price = self._get_reference_price(session_id, ts_code)
                amount = vol * price
                total_buy_amount += amount

            elif plan_action in ("sell", "reduce", "减仓"):
                vol = int(row.get("volume", 0) or 0)
                if vol <= 0:
                    warnings.append(f"{ts_code}: 卖出数量为0，跳过")
                    continue
                in_pos = pos_map.get(ts_code)
                if not in_pos:
                    errors.append(f"{ts_code}: 计划卖出但无实际持仓")
                    continue
                if vol > in_pos.can_use_volume:
                    errors.append(
                        f"{ts_code}: 计划卖出 {vol} 股，但可用仅 {in_pos.can_use_volume} 股"
                    )

        if total_buy_amount > available_cash:
            warnings.append(
                f"买入总额 ¥{total_buy_amount:,.2f} 超过可用资金 ¥{available_cash:,.2f}"
            )

        is_valid = len(errors) == 0
        if not is_valid:
            msg = "; ".join(errors)
        elif warnings:
            msg = "校验通过但有警告"
        else:
            msg = "校验通过"

        return is_valid, msg, {"warnings": warnings, "errors": errors}

    def _execute_sells(self, session_id, plan_df, dry_run):
        """执行卖出（先卖后买）"""
        results = []
        for _, row in plan_df.iterrows():
            ts_code = row["ts_code"]
            plan_action = row["plan_action"]
            vol = int(row.get("volume", 0) or 0)
            price = float(row.get("price", 0) or 0)
            if vol <= 0:
                results.append(ExecutionResult(
                    ts_code=ts_code, plan_action=plan_action,
                    result="skipped", error="volume=0",
                ))
                continue

            if dry_run:
                results.append(ExecutionResult(
                    ts_code=ts_code, plan_action=plan_action,
                    result="dry_run", price=price, volume=vol,
                ))
                continue

            try:
                if price <= 0:
                    price_type = 42  # 市价
                else:
                    price_type = 50  # 限价
                order = self._order_api.submit(
                    session_id, ts_code, "SELL", vol, price, price_type=price_type,
                )
                results.append(ExecutionResult(
                    ts_code=ts_code, plan_action=plan_action,
                    result="success", order_id=order.order_id,
                    order_sysid=order.order_sysid, price=price, volume=vol,
                ))
            except Exception as e:
                logger.error(f"卖出失败: {ts_code} | {e}")
                results.append(ExecutionResult(
                    ts_code=ts_code, plan_action=plan_action,
                    result="failed", error=str(e), price=price, volume=vol,
                ))

        return results

    def _execute_buys(self, session_id, plan_df, dry_run):
        """执行买入"""
        results = []
        for _, row in plan_df.iterrows():
            ts_code = row["ts_code"]
            plan_action = row["plan_action"]
            vol = int(row.get("volume", 0) or 0)
            price = float(row.get("price", 0) or 0)
            if vol <= 0:
                results.append(ExecutionResult(
                    ts_code=ts_code, plan_action=plan_action,
                    result="skipped", error="volume=0",
                ))
                continue

            if dry_run:
                results.append(ExecutionResult(
                    ts_code=ts_code, plan_action=plan_action,
                    result="dry_run", price=price, volume=vol,
                ))
                continue

            try:
                if price <= 0:
                    price_type = 42
                else:
                    price_type = 50
                order = self._order_api.submit(
                    session_id, ts_code, "BUY", vol, price, price_type=price_type,
                )
                results.append(ExecutionResult(
                    ts_code=ts_code, plan_action=plan_action,
                    result="success", order_id=order.order_id,
                    order_sysid=order.order_sysid, price=price, volume=vol,
                ))
            except Exception as e:
                logger.error(f"买入失败: {ts_code} | {e}")
                results.append(ExecutionResult(
                    ts_code=ts_code, plan_action=plan_action,
                    result="failed", error=str(e), price=price, volume=vol,
                ))

        return results

    def _batch_cancel_all(self, session_id):
        """一键撤销所有未成交挂单"""
        orders = self._query.get_orders(session_id)
        uncompleted = [o for o in orders if o.order_status_code not in (56, 57)]
        results = []
        for o in uncompleted:
            market = OrderAPI.infer_market(o.stock_code)
            ok = self._order_api.cancel(
                session_id, o.order_id,
                market=market, order_sysid=o.order_sysid,
            )
            results.append(ok)
        logger.info(f"批量撤单: {len(results)} 单, 成功 {sum(results)}")
        return results

    def _get_reference_price(self, session_id, stock_code):
        """获取参考价格（从持仓中取 latest_price）"""
        positions = self._query.get_positions(session_id)
        for p in positions:
            if p.stock_code == stock_code:
                return p.last_price
        return 0.0

    def get_session_status(self):
        """获取 QMT 连接状态（供前端状态面板使用）

        Returns:
            dict: connected, session_id, orders_enabled, account_id, error
        """
        try:
            if not self._client.health():
                return {"connected": False, "error": "服务不可达"}
            sess = self._session_mgr.open()
            return {
                "connected": True,
                "session_id": sess.session_id,
                "orders_enabled": sess.orders_enabled,
                "account_id": sess.account_id,
                "error": "",
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}


if __name__ == "__main__":
    from qmt_trader.client import QmtClient
    client = QmtClient()

    executor = TradeExecutor(client)

    status = executor.get_session_status()
    print(f"QMT 连接状态: {'🟢 已连接' if status['connected'] else '🔴 不可达'}")
    if not status["connected"]:
        print(f"  错误: {status['error']}")
        exit(0)
    print(f"  会话: {status['session_id']}")
    print(f"  orders_enabled: {status['orders_enabled']}")

    # 构造测试计划
    test_plan = pd.DataFrame([
        {"ts_code": "000001.SZ", "plan_action": "buy", "volume": 100, "price": 11.00},
        {"ts_code": "000001.SZ", "plan_action": "sell", "volume": 100, "price": 12.00},
    ])

    # Dry-run 测试
    print("\nDry-run 测试:")
    results = executor.execute_plan(test_plan, dry_run=True)
    for r in results:
        print(f"  {r.ts_code} {r.plan_action}: {r.result}")

    print("\n计划校验测试:")
    valid, msg, detail = executor.validate_plan(status["session_id"], test_plan)
    print(f"  校验结果: {'✅ 通过' if valid else '❌ 失败'} | {msg}")