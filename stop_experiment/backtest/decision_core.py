#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策内核（SSOT）：退出预测查找 + 模型退出评估 + 完整日终决策

Purpose:
    回测、逐日推理、日报三条链路共用的决策逻辑，保证口径完全一致。
    decide_eod() 是唯一的完整日终决策函数，使用严格精确匹配（无回退）。

Pipeline Position:
    被 dynamic_exit_backtest_v2.py / 06_daily_inference_replay.py / 08_daily_inference_report.py 引用

Inputs:
    - pred_lookup: {(signal_id, obs_date): pred_dict}
    - holdings: dict
    - candidates: DataFrame

Outputs:
    - find_exit_pred() → pred_dict or None
    - decide_eod() → (new_holdings, pending_buys, pending_sells, sell_reasons)

How to Run:
    python stop_experiment/backtest/decision_core.py  (自测)

Side Effects:
    - decide_eod() 会修改 holdings 的 days_held 值
    - 不删除 holdings 元素（sells 由调用方在下日开盘执行）
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
                       exit_threshold=None, current_price_map=None,
                       exit_sub_mode=None, buy_reg_exit_threshold=None,
                       prev_decision_date=None):
    """
    评估所有持仓的退出信号（max_hold → stop_loss → model_exit）。

    exit_sub_mode (仅 model_exit 有效):
        None / "buy_cls_only" (X1): 仅 buy_cls > exit_threshold 退出
        "sell_decay" (X3): buy_cls > exit_threshold 且 sell_reg(t) < sell_reg(t-1)
        "or_buy_reg" (X4): buy_cls > exit_threshold 或 buy_reg < -buy_reg_exit_threshold

    Input:
        holdings:         dict {code: holding_dict}
        prev_date:        前一交易日 (T-1)
        pred_lookup:      {(signal_id, obs_date): pred_dict}
        max_hold_days:    最大持仓天数
        stop_loss:        止损阈值
        exit_threshold:   model_exit 阈值
        current_price_map: {code: price} 当日收盘价映射，用于止损计算
        exit_sub_mode:    model_exit 子模式
        buy_reg_exit_threshold: X4 的 buy_reg 退出阈值 (正数，如 0.05)
        prev_decision_date: X3 用 T-2 日期，用于比较 sell_reg 衰减

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

        # c. model_exit (supporting sub-modes: X1/X3/X4)
        sid = h.get("signal_id")
        if sid is not None and prev_date is not None and h["days_held"] > 1:
            pred = find_exit_pred(sid, prev_date, pred_lookup)
            if pred is not None:
                bc = pred.get("pred_buy_cls", np.nan)

                if exit_sub_mode == "sell_decay":
                    # X3: buy_cls > threshold 且 sell_reg 较前日下降
                    sr = pred.get("pred_sell_reg", np.nan)
                    prev_pred = find_exit_pred(sid, prev_decision_date, pred_lookup) if prev_decision_date else None
                    sr_prev = prev_pred.get("pred_sell_reg", np.nan) if prev_pred else np.nan
                    if (not np.isnan(bc) and bc > _threshold and
                            not np.isnan(sr) and not np.isnan(sr_prev) and sr < sr_prev):
                        sells.append((code, dict(h), "model_risk"))
                        continue

                elif exit_sub_mode == "or_buy_reg":
                    # X4: buy_cls > threshold 或 buy_reg < -k
                    br = pred.get("pred_buy_reg", np.nan)
                    _br_th = buy_reg_exit_threshold if buy_reg_exit_threshold is not None else 0.05
                    if ((not np.isnan(bc) and bc > _threshold) or
                            (not np.isnan(br) and br < -_br_th)):
                        sells.append((code, dict(h), "model_risk"))
                        continue

                else:
                    # X1: 仅 buy_cls > threshold
                    if not np.isnan(bc) and bc > _threshold:
                        sells.append((code, dict(h), "model_risk"))

    return sells


def decide_eod(decision_date, holdings, candidates, pred_lookup, prev_date,
               day_close, day_open_next=None, max_stocks=10, max_hold_days=20,
               stop_loss=-0.07, exit_threshold=0.70, strategy="sell_score",
               exit_sub_mode=None, buy_reg_exit_threshold=None,
               prev_decision_date=None):
    """
    完整日终决策（SSOT），供 backtest/replay/report 共用。

    决策顺序（与回测 run_backtest Step 2~4 完全一致）：
    1. 所有持仓 days_held += 1
    2. max_hold → stop_loss → model_exit 依次检查，命中后 continue
    3. n_avail = max_stocks - (len(holdings) - len(pending_sells))
       即预留即将卖出的仓位，与回测语义一致
    4. 从 candidates 中按 score 降序选取 top-k 新买入

    注意：不删除 holdings 中的 pending_sells，由调用方在下日开盘执行（与 backtest Step 1.5 一致）。
    不计算 NAV（调用方在调用前自行计算）。

    Input:
        decision_date: 决策日（T日）
        holdings:      开盘执行后持仓 {code: {buy_date, buy_price, weight, days_held, ts_code, signal_id, score}}
        candidates:    T日候选池 DataFrame（columns: ts_code, signal_id, score/composite_score）
        pred_lookup:   {(signal_id, obs_date): pred_dict}
        prev_date:     前一交易日
        day_close:     T日收盘价 Series（index=code），用于 stop_loss 计算
        day_open_next: T+1日开盘价 Series（index=code），用于确定买入价，None 时不买入
        max_stocks:    最大持仓数
        max_hold_days: 最大持仓天数
        stop_loss:     止损阈值（如 -0.07）
        exit_threshold: model_exit pred_buy_cls 阈值
        strategy:      策略名
        exit_sub_mode: model_exit 子模式 (None="buy_cls_only" / "sell_decay" / "or_buy_reg")
        buy_reg_exit_threshold: X4 的 buy_reg 退出阈值
        prev_decision_date: X3 用 T-2 日期，比较 sell_reg 衰减

    Output:
        holdings:       原 holdings dict（days_held 已递增，sells 仍保留在 dict 中）
        pending_buys:   [(code, buy_price, ts_code, score, signal_id), ...]
        pending_sells:  [{"code": ..., "holding": {...}, "reason": ...}, ...]
        sell_reasons:   {ts_code: reason}
        extra:          {"buy_details": [...], "sell_details": [...], "hold_details": [...]}
                        供 10_tomorrow_action_plan.py 生成决策链条，下游可忽略
    """
    pending_sells = []
    sell_reasons = {}

    # ---- 1. days_held += 1 ----
    for h in holdings.values():
        h["days_held"] += 1

    # ---- 2. 退出评估: max_hold → stop_loss → model_exit ----
    for code, h in list(holdings.items()):
        # a. max_hold
        if h["days_held"] > max_hold_days:
            pending_sells.append({"code": code, "holding": dict(h), "reason": "max_hold"})
            sell_reasons[h.get("ts_code", code)] = "max_hold"
            continue

        # b. stop_loss（使用 day_close，与回测完全一致）
        if code in day_close.index:
            cp = day_close[code]
            if not np.isnan(cp) and cp > 0 and h.get("buy_price") and h["buy_price"] > 0:
                ret = (cp - h["buy_price"]) / h["buy_price"]
                if ret < stop_loss:
                    pending_sells.append({"code": code, "holding": dict(h), "reason": "stop_loss"})
                    sell_reasons[h.get("ts_code", code)] = "stop_loss"
                    continue

        # c. model_exit（仅对 days_held > 1 且 signal_id 已知的持仓, 支持 X1/X3/X4 子模式）
        if h["days_held"] > 1 and prev_date is not None:
            sid = h.get("signal_id")
            if sid is not None:
                pred = find_exit_pred(sid, prev_date, pred_lookup)
                if pred is not None:
                    bc = pred.get("pred_buy_cls", np.nan)

                    def _do_sell():
                        pending_sells.append({"code": code, "holding": dict(h), "reason": "model_risk"})
                        sell_reasons[h.get("ts_code", code)] = "model_risk"

                    if exit_sub_mode == "sell_decay":
                        sr = pred.get("pred_sell_reg", np.nan)
                        prev_pred = find_exit_pred(sid, prev_decision_date, pred_lookup) if prev_decision_date else None
                        sr_prev = prev_pred.get("pred_sell_reg", np.nan) if prev_pred else np.nan
                        if (not np.isnan(bc) and bc > exit_threshold and
                                not np.isnan(sr) and not np.isnan(sr_prev) and sr < sr_prev):
                            _do_sell()

                    elif exit_sub_mode == "or_buy_reg":
                        br = pred.get("pred_buy_reg", np.nan)
                        _br_th = buy_reg_exit_threshold if buy_reg_exit_threshold is not None else 0.05
                        if ((not np.isnan(bc) and bc > exit_threshold) or
                                (not np.isnan(br) and br < -_br_th)):
                            _do_sell()

                    else:
                        if not np.isnan(bc) and bc > exit_threshold:
                            _do_sell()

    # ---- 3. 新买入 ----
    # n_avail: 预留 pending_sells 的仓位（sell 在下日开盘执行后释放额度）
    # holdings 此时仍包含 pending_sells（尚未执行），所以 n_avail = max - (|H| - |S|) = max - |H| + |S|
    n_avail = max_stocks - len(holdings) + len(pending_sells)
    pending_buys = []
    if n_avail > 0 and not candidates.empty and day_open_next is not None:
        score_col = "score" if "score" in candidates.columns else "composite_score"
        if score_col in candidates.columns:
            cand_sorted = candidates.sort_values([score_col, "ts_code"], ascending=[False, True])
        else:
            cand_sorted = candidates

        bought_codes = set(holdings.keys())
        used_codes = set()

        for _, row in cand_sorted.iterrows():
            if len(used_codes) >= n_avail:
                break

            ts_code = row.get("ts_code") or row.get("code")
            if ts_code is None:
                continue
            code_clean = ts_code[:6] if "." in ts_code else ts_code

            if code_clean in bought_codes or code_clean in used_codes:
                continue

            bp = None
            if code_clean in day_open_next.index:
                bp_val = day_open_next[code_clean]
                if not np.isnan(bp_val) and bp_val > 0:
                    bp = bp_val
            if bp is None:
                bp = row.get("buy_price") or row.get("close")
                if bp is None or np.isnan(bp) or bp <= 0:
                    continue

            sc = row.get(score_col, 0)
            sid = row.get("signal_id")
            pending_buys.append((code_clean, bp, ts_code, sc, sid))
            used_codes.add(code_clean)

    # ---- extra: rich reason for action plan ----
    buy_details = []
    for idx, (code, bp, ts_code, sc, sid) in enumerate(pending_buys):
        buy_details.append({
            "code": code, "ts_code": ts_code, "signal_id": sid,
            "score": sc if isinstance(sc, (int, float)) else (float(sc) if sc is not None else 0),
            "rank": idx + 1, "obs_day": None,
            "why": f"候选池排序第{idx+1} → 组合空位允许 → 不在持仓中 → 生成T+1买入单",
        })
    sell_details = []
    for item in pending_sells:
        code = item["code"]
        h = item["holding"]
        cur_ret = None
        if code in day_close.index and h.get("buy_price") and h["buy_price"] > 0:
            cp = day_close[code]
            if not np.isnan(cp) and cp > 0:
                cur_ret = round(float((cp - h["buy_price"]) / h["buy_price"]), 4)
        reason = item["reason"]
        threshold_str = None
        if reason == "max_hold":
            threshold_str = f"max_hold_days={max_hold_days}"
        elif reason == "stop_loss":
            threshold_str = f"stop_loss={stop_loss:.0%}"
        elif reason == "model_risk":
            threshold_str = f"pred_buy_cls>{exit_threshold:.2f}"
        sell_details.append({
            "code": code, "ts_code": h.get("ts_code", code),
            "signal_id": h.get("signal_id"), "reason": reason,
            "days_held": h["days_held"], "cur_ret": cur_ret,
            "threshold": threshold_str,
            "why": f"持仓{h['days_held']}天 → 触发{reason}({threshold_str}) → decide_eod判定卖出 → 生成T+1卖出单",
        })
    hold_details = []
    sell_codes = {item["code"] for item in pending_sells}
    for code, h in holdings.items():
        if code in sell_codes:
            continue
        cur_ret = None
        if code in day_close.index and h.get("buy_price") and h["buy_price"] > 0:
            cp = day_close[code]
            if not np.isnan(cp) and cp > 0:
                cur_ret = round(float((cp - h["buy_price"]) / h["buy_price"]), 4)
        risk_flags = []
        if h["days_held"] >= max_hold_days - 3:
            risk_flags.append("即将触发max_hold")
        if cur_ret is not None and cur_ret < stop_loss * 0.7:
            risk_flags.append("接近止损线")
        why_parts = [f"持仓{h['days_held']}天"]
        if not risk_flags:
            why_parts.append("未触发max_hold/stop_loss/model_exit")
            why_parts.append("继续持有")
        else:
            why_parts.append("继续持有(" + ";".join(risk_flags) + ")")
        hold_details.append({
            "code": code, "ts_code": h.get("ts_code", code),
            "days_held": h["days_held"], "cur_ret": cur_ret,
            "risk_flags": risk_flags, "why": " → ".join(why_parts),
        })
    extra = {"buy_details": buy_details, "sell_details": sell_details, "hold_details": hold_details}

    return holdings, pending_buys, pending_sells, sell_reasons, extra


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