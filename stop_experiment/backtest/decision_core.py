#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策内核（SSOT）：退出预测查找 + 模型退出评估 + 买卖决策

Purpose:
    回测、逐日推理、日报三条链路共用的决策逻辑，保证口径完全一致。
    按 advicement.txt 建议，退出预测使用严格精确匹配（无回退），
    避免回测和 replay 的退出行为不一致。

Pipeline Position:
    被 dynamic_exit_backtest_v2.py / 06_daily_inference_replay.py / 08_daily_inference_report.py 引用

Inputs:
    - pred_lookup: {(signal_id, obs_date): pred_dict}
    - holdings: dict
    - candidates: DataFrame

Outputs:
    - find_exit_pred() → pred_dict or None
    - evaluate_model_exit() → (should_sell: bool, reason: str)

How to Run:
    python stop_experiment/backtest/decision_core.py  (自测)

Side Effects:
    - 纯函数，无副作用
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd


def find_exit_pred(signal_id, prev_date, pred_lookup):
    """
    严格精确匹配：查找 (signal_id, prev_date) 的预测。

    不再使用回退逻辑（不再查找 <= prev_date 的最新日期），
    因为使用过期预测做决策是危险的（回测和 replay 口径不一致）。

    Input:
        signal_id:    信号 ID
        prev_date:    当前决策日的前一个交易日
        pred_lookup:  {(signal_id, obs_date): pred_dict}

    Output:
        pred_dict 或 None
    """
    sid_int = int(signal_id)
    return pred_lookup.get((sid_int, prev_date))


def evaluate_model_exit(holding, code, prev_date, pred_lookup, threshold=None):
    """
    模型退出判断（buy-only exit）：
    - 持仓中取前一交易日的 pred_buy_cls
    - 若 pred_buy_cls > threshold → 模型认为风险较高，建议退出

    仅对 days_held > 1 的持仓生效（次日才允许模型退出）。
    使用 find_exit_pred 严格精确匹配。

    Input:
        holding:     持仓 dict {ts_code, buy_date, buy_price, weight, days_held, signal_id, score, ...}
        code:        股票代码
        prev_date:   前一交易日
        pred_lookup: 预测查找表
        threshold:   pred_buy_cls 退出阈值，None 时使用 BUY_CLS_EXIT_THRESHOLD

    Output:
        (should_sell: bool, reason: str)
    """
    from stop_experiment.pipeline.stop_config import V1_PARAMS
    days_held = holding["days_held"]
    signal_id = holding.get("signal_id")
    _threshold = threshold if threshold is not None else V1_PARAMS.get("buy_cls_exit_threshold", 0.70)

    if signal_id is not None and days_held > 1 and prev_date is not None:
        pred = find_exit_pred(signal_id, prev_date, pred_lookup)
        if pred is not None:
            bc = pred.get("pred_buy_cls", np.nan)
            if not np.isnan(bc) and bc > _threshold:
                return True, "model_risk"
    return False, ""


def evaluate_max_hold(holding, max_hold_days=None):
    """
    最大持仓天数退出检查。

    Input:
        holding:       持仓 dict
        max_hold_days: 最大持仓天数，None 时使用 V1_PARAMS["max_hold_days"]

    Output:
        (should_sell: bool, reason: str)
    """
    from stop_experiment.pipeline.stop_config import V1_PARAMS
    _max_days = max_hold_days if max_hold_days is not None else V1_PARAMS.get("max_hold_days", 20)
    if holding["days_held"] > _max_days:
        return True, "max_hold"
    return False, ""


def evaluate_stop_loss(holding, current_price, stop_loss_threshold=None):
    """
    止损退出检查。

    Input:
        holding:             持仓 dict
        current_price:       当前价格
        stop_loss_threshold: 止损阈值（如 -0.07），None 时使用 V1_PARAMS["stop_loss"]

    Output:
        (should_sell: bool, reason: str)
    """
    from stop_experiment.pipeline.stop_config import V1_PARAMS
    _threshold = stop_loss_threshold if stop_loss_threshold is not None else V1_PARAMS.get("stop_loss", -0.07)
    if current_price is None or np.isnan(current_price) or current_price <= 0:
        return False, ""
    ret = (current_price - holding["buy_price"]) / holding["buy_price"]
    if ret < _threshold:
        return True, "stop_loss"
    return False, ""


def evaluate_eod_exits(holdings, prev_date, pred_lookup,
                       max_hold_days=None, stop_loss=None,
                       exit_threshold=None, current_price_map=None):
    """
    评估所有持仓的退出信号（max_hold → stop_loss → model_exit）。

    Input:
        holdings:         dict {code: holding_dict}
        prev_date:        前一交易日
        pred_lookup:      {(signal_id, obs_date): pred_dict}
        max_hold_days:    最大持仓天数
        stop_loss:        止损阈值
        exit_threshold:   model_exit 阈值
        current_price_map: {code: price} 当日收盘价映射，用于止损计算

    Output:
        sell_decisions: [(code, holding, reason), ...]
    """
    from stop_experiment.pipeline.stop_config import V1_PARAMS
    _max_hold = max_hold_days if max_hold_days is not None else V1_PARAMS.get("max_hold_days", 20)
    _stop = stop_loss if stop_loss is not None else V1_PARAMS.get("stop_loss", -0.07)
    _threshold = exit_threshold if exit_threshold is not None else V1_PARAMS.get("buy_cls_exit_threshold", 0.70)

    sells = []
    for code, h in list(holdings.items()):
        # a. max_hold
        if h["days_held"] > _max_hold:
            sells.append((code, dict(h), "max_hold"))
            continue

        # b. stop_loss
        cp_map = current_price_map or {}
        cp = cp_map.get(code)
        if cp is not None and not np.isnan(cp) and cp > 0:
            ret = (cp - h["buy_price"]) / h["buy_price"]
            if ret < _stop:
                sells.append((code, dict(h), "stop_loss"))
                continue

        # c. model_exit
        sid = h.get("signal_id")
        if sid is not None and prev_date is not None and h["days_held"] > 1:
            pred = find_exit_pred(sid, prev_date, pred_lookup)
            if pred is not None:
                bc = pred.get("pred_buy_cls", np.nan)
                if not np.isnan(bc) and bc > _threshold:
                    sells.append((code, dict(h), "model_risk"))

    return sells


# ==================== 自测入口 ====================
if __name__ == "__main__":
    print("decision_core 自测")
    print("=" * 60)

    # 构造测试数据
    test_date = pd.Timestamp("2026-03-01")

    test_lookup = {
        (1001, test_date): {"pred_buy_cls": 0.85, "pred_sell_reg": 0.4},
        (1001, test_date + pd.Timedelta(days=1)): {"pred_buy_cls": 0.3, "pred_sell_reg": 0.7},
        (1002, test_date): {"pred_buy_cls": 0.1, "pred_sell_reg": 0.2},
    }

    # 测试1: 精确匹配成功
    pred = find_exit_pred(1001, test_date, test_lookup)
    assert pred is not None and pred["pred_buy_cls"] == 0.85, f"Exact match failed: {pred}"
    print("✅ 测试1: 精确匹配成功")

    # 测试2: 无匹配（不存在的日期）
    pred = find_exit_pred(1001, test_date + pd.Timedelta(days=3), test_lookup)
    assert pred is None, f"Should return None: {pred}"
    print("✅ 测试2: 无匹配返回 None")

    # 测试3: 无匹配（不存在的 signal_id）
    pred = find_exit_pred(9999, test_date, test_lookup)
    assert pred is None, f"Should return None: {pred}"
    print("✅ 测试3: 不存在 signal_id 返回 None")

    # 测试4: model_exit 触发
    holding = {"days_held": 2, "signal_id": 1001, "buy_price": 10.0, "weight": 1.0}
    should_sell, reason = evaluate_model_exit(holding, "000001", test_date, test_lookup, threshold=0.7)
    assert should_sell and reason == "model_risk", f"Expected model_risk: {should_sell}, {reason}"
    print("✅ 测试4: model_exit 触发 (pred_buy_cls=0.85 > 0.7)")

    # 测试5: model_exit 不触发（低概率）
    holding2 = {"days_held": 2, "signal_id": 1002, "buy_price": 10.0, "weight": 1.0}
    should_sell, reason = evaluate_model_exit(holding2, "000002", test_date, test_lookup, threshold=0.7)
    assert not should_sell, f"Expected no exit: {should_sell}, {reason}"
    print("✅ 测试5: model_exit 不触发 (pred_buy_cls=0.1 <= 0.7)")

    # 测试6: days_held=1 不触发 model_exit
    holding3 = {"days_held": 1, "signal_id": 1001, "buy_price": 10.0, "weight": 1.0}
    should_sell, reason = evaluate_model_exit(holding3, "000003", test_date, test_lookup, threshold=0.7)
    assert not should_sell, f"days_held=1 should not trigger: {should_sell}, {reason}"
    print("✅ 测试6: days_held=1 不触发")

    # 测试7: evaluate_eod_exits
    holdings = {
        "000001": {"days_held": 3, "signal_id": 1001, "buy_price": 10.0, "weight": 1.0},
        "000002": {"days_held": 2, "signal_id": 1002, "buy_price": 10.0, "weight": 1.0},
        "000003": {"days_held": 25, "signal_id": None, "buy_price": 10.0, "weight": 1.0},
    }
    price_map = {"000001": 9.5, "000002": 10.5, "000003": 8.0}
    sells = evaluate_eod_exits(holdings, test_date, test_lookup,
                               max_hold_days=20, stop_loss=-0.07,
                               exit_threshold=0.7, current_price_map=price_map)
    reasons = {code: reason for code, h, reason in sells}
    assert "000003" in reasons and reasons["000003"] == "max_hold", f"max_hold: {reasons}"
    assert "000001" in reasons and reasons["000001"] == "model_risk", f"model_risk: {reasons}"
    assert "000002" not in reasons, f"Should not sell 000002: {reasons}"
    print(f"✅ 测试7: evaluate_eod_exits → {reasons}")

    # 测试8: stop_loss
    holdings4 = {"000004": {"days_held": 2, "signal_id": None, "buy_price": 10.0, "weight": 1.0}}
    price_map4 = {"000004": 9.2}
    sells4 = evaluate_eod_exits(holdings4, test_date, test_lookup,
                                max_hold_days=20, stop_loss=-0.07,
                                exit_threshold=0.7, current_price_map=price_map4)
    reasons4 = {code: reason for code, h, reason in sells4}
    assert "000004" in reasons4 and reasons4["000004"] == "stop_loss", f"stop_loss: {reasons4}"
    print("✅ 测试8: stop_loss 触发")

    print("\n✅ 全部自测通过")