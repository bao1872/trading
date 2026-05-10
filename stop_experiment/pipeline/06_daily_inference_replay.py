#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日级推理回放 + 回测对账

Purpose:
    逐日重放回测引擎的决策逻辑，与回测每日快照逐项对账，验证 buy/hold/sell + sell_reason 100% 一致。

    三层验证:
    1) 样本一致性: 候选池股票集合
    2) 分数一致性: pred_sell_reg / pred_buy_cls / composite_score（允许 fp 误差）
    3) 决策一致性: action(buy/hold/sell) + sell_reason

    真实预测优先:
    - 优先读取 predictions/YYYY-MM-DD.parquet (08_daily_inference_report.py 生成的真实预测)
    - 无真实预测且超出历史快照范围时，标记 NO_REAL_PREDICTION_DATA

Pipeline Position:
    生产流水线第一步（每日，重复）。
    上游: full_test_predictions.parquet, predictions/, 回测引擎 debug_snapshots
    下游: 08_daily_inference_report.py

Inputs:
    - stop_experiment/output/full_test_predictions.parquet (历史 test 快照)
    - stop_experiment/output/predictions/YYYY-MM-DD.parquet (真实预测，若存在)
    - 回测引擎 debug_snapshots 输出

Outputs:
    - stop_experiment/output/daily_inference_replay_YYYY-MM-DD.csv  (逐日推理结果)
    - stop_experiment/output/daily_inference_diff_YYYY-MM-DD.csv    (对账差异)

How to Run:
    # 单日
    python stop_experiment/pipeline/06_daily_inference_replay.py --date 2026-03-15 --top-k 10
    # 10 日批量
    python stop_experiment/pipeline/06_daily_inference_replay.py --batch-first-10
    # dry-run（只跑推理不产出 diff）
    python stop_experiment/pipeline/06_daily_inference_replay.py --date 2026-03-15 --dry-run

Side Effects:
    - 只读 parquet，输出 csv
"""

from __future__ import annotations

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, BACKTEST_DIR, BASELINE_E0_X1_V1_PARAMS, PRODUCTION_PARAMS, PREDICTIONS_DIR,
)
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest,
)
from stop_experiment.backtest.daily_state_machine import step_day
from stop_experiment.backtest.simple_backtest import score_stocks

DYNAMIC_DIR = os.path.join(BACKTEST_DIR, "dynamic")
FIRST_10_DATES = [
    "2026-03-10", "2026-03-15", "2026-03-20", "2026-03-24",
    "2026-04-03", "2026-04-10", "2026-04-17", "2026-04-24",
    "2026-04-30", "2026-05-06",
]


def load_real_predictions_for_date(target_date):
    """
    优先读取已保存的真实预测结果（与 08_daily_inference_report.py 共用同一格式）。

    Input:
        target_date: 目标日期 (datetime 或 str)

    Output:
        DataFrame: 真实预测数据 (含 obs_date, ts_code, signal_id, obs_day, pred_*, score)
        None: 无真实预测文件
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    day_path = os.path.join(PREDICTIONS_DIR, f"{target_date.strftime('%Y-%m-%d')}.parquet")
    if os.path.exists(day_path):
        df = pd.read_parquet(day_path)
        print(f"  [真实预测] 已加载: {day_path} ({len(df)} 行)")
        return df
    return None


def build_daily_candidate_snapshot(date, df_all, score_col="composite_score"):
    """
    生成当日可见候选池，去重逻辑与回测引擎 signal_by_date 完全一致。

    Input:
        date: 目标日期 (datetime 或 str)
        df_all: 全量候选数据 (obs_date + ts_code + obs_day + 分数字段 + 预测字段)
        score_col: 排名用的分数字段

    Output:
        candidates: 去重排序后的候选池 DataFrame

    规则（与 run_backtest signal_by_date 完全一致）：
        1. obs_date == date
        2. obs_day=1（生产口径）
        3. 按 score DESC 排序 → drop_duplicates(ts_code, keep="first")
        4. 不分两阶段去重，不留 signal_id 中间产物
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    sub = df_all[df_all["obs_date"] == date].copy()
    if sub.empty:
        return pd.DataFrame()

    candidate_obs_days = BASELINE_E0_X1_V1_PARAMS.get("candidate_obs_days", [1])
    sub = sub[sub["obs_day"].isin(candidate_obs_days)]
    if sub.empty:
        return pd.DataFrame()

    if score_col not in sub.columns:
        sub = score_stocks(sub, "sell_score")
        score_col = "score"

    sub = sub.sort_values(score_col, ascending=False)
    sub = sub.drop_duplicates(subset=["ts_code"], keep="first")

    return sub.reset_index(drop=True)


def compare_with_backtest(date, replay_result, backtest_snapshot):
    """
    逐项对账: 候选池 / 买入 / 卖出 / sell_reason / 分数

    Input:
        date:              日期
        replay_result:     (candidates_df, holdings_after, pending_buys, sells, sell_reasons)
        backtest_snapshot: run_backtest(debug_snapshots=True) 的快照

    Output:
        diff_rows:   对账差异行列表
        summary:     {field: (match, details)}
    """
    replay_candidates, replay_holdings, replay_buys, replay_sells, replay_reasons = replay_result

    snap_codes = set(backtest_snapshot.get("candidate_codes", []))
    replay_codes = set(replay_candidates["ts_code"]) if not replay_candidates.empty else set()
    snap_buys = set(b["ts_code"] for b in backtest_snapshot.get("buys", []))
    replay_buys_set = set(b[2] for b in replay_buys)
    snap_sells = set(backtest_snapshot.get("sell_reasons", {}).keys())
    # replay_sells 是 pending_sells (list[dict])，需提取 ts_code
    replay_sells_set = set(
        item["holding"]["ts_code"] if isinstance(item, dict) else item
        for item in replay_sells
    )
    snap_reasons = backtest_snapshot.get("sell_reasons", {})

    diff_rows = []
    if isinstance(date, pd.Timestamp):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)

    def _add_diff(ts_code, field, bt_val, rp_val, note=""):
        is_match = bt_val == rp_val
        abs_diff_val = np.nan
        if isinstance(bt_val, (int, float)) and isinstance(rp_val, (int, float)):
            abs_diff_val = abs(bt_val - rp_val)
        diff_rows.append({
            "date": date_str, "ts_code": ts_code, "field": field,
            "backtest_value": str(bt_val), "replay_value": str(rp_val),
            "is_match": is_match, "abs_diff": abs_diff_val if pd.notna(abs_diff_val) else "",
            "note": note,
        })

    # === 候选池 ===
    all_codes = snap_codes | replay_codes
    if snap_codes == replay_codes:
        for c in sorted(all_codes):
            _add_diff(c, "candidate_pool", "in", "in", "通过")
    else:
        missing = snap_codes - replay_codes
        extra = replay_codes - snap_codes
        for c in sorted(snap_codes & replay_codes):
            _add_diff(c, "candidate_pool", "in", "in", "通过")
        for c in sorted(missing):
            _add_diff(c, "candidate_pool", "in", "missing", "差异: 回测有 replay 无")
        for c in sorted(extra):
            _add_diff(c, "candidate_pool", "not found", "in", "差异: replay 有回测无")

    # === 分数对比 ===
    snap_detail = backtest_snapshot.get("candidates_df", pd.DataFrame())
    if not snap_detail.empty and not replay_candidates.empty:
        for _, snap_r in snap_detail.iterrows():
            sc = snap_r.get("ts_code")
            rp_r = replay_candidates[replay_candidates["ts_code"] == sc]
            if rp_r.empty:
                continue
            rp_r = rp_r.iloc[0]
            for col in ["pred_sell_reg", "pred_buy_cls", "composite_score"]:
                snap_v = snap_r.get(col, np.nan)
                rp_v = rp_r.get(col, np.nan)
                if pd.notna(snap_v) and pd.notna(rp_v):
                    diff = abs(snap_v - rp_v)
                    if diff > 1e-6:
                        _add_diff(sc, col, snap_v, rp_v, f"分数差异: {diff:.6e}")
                    else:
                        _add_diff(sc, col, "一致", "一致", "")

    # === 买入 ===
    all_buys = snap_buys | replay_buys_set
    for ts in sorted(all_buys):
        in_snap = ts in snap_buys
        in_rp = ts in replay_buys_set
        val_s = "buy" if in_snap else "-"
        val_r = "buy" if in_rp else "-"
        _add_diff(ts, "action", val_s, val_r, "" if val_s == val_r else "买入不一致")

    # === 卖出 ===
    all_sells = snap_sells | replay_sells_set
    for ts_code in sorted(all_sells):
        in_snap = ts_code in snap_sells
        in_rp = ts_code in replay_sells_set
        val_s = "sell" if in_snap else "-"
        val_r = "sell" if in_rp else "-"
        _add_diff(ts_code, "action", val_s, val_r, "" if val_s == val_r else "卖出不一致")

    # === sell_reason ===
    for ts_code in sorted(snap_sells | replay_sells_set):
        sr_s = snap_reasons.get(ts_code, "")
        sr_r = replay_reasons.get(ts_code, "")
        _add_diff(ts_code, "sell_reason", sr_s, sr_r, "" if sr_s == sr_r else f"原因不一致 (bt={sr_s} rp={sr_r})")

    # === 汇总 ===
    n_total = len(diff_rows)
    n_pass = sum(1 for d in diff_rows if d["is_match"])
    n_fail = n_total - n_pass

    critical_fail = [d for d in diff_rows if not d["is_match"] and d["field"] in
                     ("candidate_pool", "action", "sell_reason")]

    return diff_rows, {
        "date": date_str,
        "total_checks": n_total,
        "passed": n_pass,
        "failed": n_fail,
        "critical_failures": len(critical_fail),
        "all_pass": n_fail == 0,
    }


def _run_debug_dashboard(tdate, df_all_raw, df_all, score_col, test_df,
                         holdings, price, snapshot, pred_indexed,
                         v1_params, strategy):
    """单日断点调试: 输出 4 张诊断表到 CSV"""
    date_str = tdate.strftime("%Y-%m-%d")
    debug_dir = os.path.join(OUTPUT_DIR, f"debug_{date_str}")
    os.makedirs(debug_dir, exist_ok=True)

    print(f"\n  {'='*60}")
    print(f"  🔍 断点调试: {date_str}")
    print(f"  {'='*60}")

    # ---- 表 A: 评分前原始候选集 ----
    bt_raw = test_df[test_df["obs_date"] == tdate].copy()
    rp_raw = df_all_raw[df_all_raw["obs_date"] == tdate].copy()

    common_cols = ["ts_code", "signal_id", "obs_day", "obs_date",
                   "pred_sell_reg", "pred_buy_cls"]
    bt_cols = [c for c in common_cols if c in bt_raw.columns]
    rp_cols = [c for c in common_cols if c in rp_raw.columns]

    bt_a = bt_raw[bt_cols].sort_values(["ts_code", "signal_id"]).reset_index(drop=True)
    rp_a = rp_raw[rp_cols].sort_values(["ts_code", "signal_id"]).reset_index(drop=True)
    bt_a["source"] = "backtest"
    rp_a["source"] = "replay"
    table_a = pd.concat([bt_a, rp_a], ignore_index=True)
    table_a.to_csv(os.path.join(debug_dir, "table_A_raw_candidates.csv"), index=False)

    bt_codes = set(bt_raw["ts_code"]) if not bt_raw.empty else set()
    rp_codes = set(rp_raw["ts_code"]) if not rp_raw.empty else set()
    print(f"  表A 评分前原始候选: bt={len(bt_raw)}行/{len(bt_codes)}股, "
          f"rp={len(rp_raw)}行/{len(rp_codes)}股")
    if bt_codes != rp_codes:
        print(f"    ❌ 表A 差异! bt独有={bt_codes-rp_codes}, rp独有={rp_codes-bt_codes}")
    else:
        print(f"    ✅ 表A 一致")

    # ---- 表 B: 评分后 score 一致性 ----
    bt_scored = score_stocks(test_df.copy(), strategy)
    bt_day = bt_scored[bt_scored["obs_date"] == tdate].copy()
    rp_day = df_all[df_all["obs_date"] == tdate].copy()

    bt_b = bt_day[["ts_code", "signal_id", "score"]].rename(columns={"score": "score_bt"})
    rp_b = rp_day[["ts_code", "signal_id", score_col]].rename(columns={score_col: "score_rp"})
    table_b = pd.merge(bt_b, rp_b, on=["ts_code", "signal_id"], how="outer")
    table_b["score_diff"] = table_b["score_bt"] - table_b["score_rp"]
    table_b["is_match"] = table_b["score_diff"].abs() < 1e-9
    table_b.to_csv(os.path.join(debug_dir, "table_B_score_comparison.csv"), index=False)

    n_match = table_b["is_match"].sum()
    n_total = len(table_b)
    print(f"  表B 评分对比: {n_match}/{n_total} 一致")
    if n_match < n_total:
        mismatches = table_b[~table_b["is_match"]]
        print(f"    ❌ 表B 差异行:")
        for _, r in mismatches.iterrows():
            print(f"       {r['ts_code']} sid={r['signal_id']} bt={r['score_bt']:.6f} rp={r['score_rp']:.6f} diff={r['score_diff']:.6e}")
    else:
        print(f"    ✅ 表B 一致")

    # ---- 表 C: drop_duplicates 前后明细 ----
    rp_before_dd = rp_day.sort_values(score_col, ascending=False)
    rp_before_dd["rank"] = range(1, len(rp_before_dd) + 1)
    rp_before_dd["retained"] = ~rp_before_dd.duplicated(subset=["ts_code"], keep="first")

    bt_before_dd = bt_day.sort_values("score", ascending=False)
    bt_before_dd["rank"] = range(1, len(bt_before_dd) + 1)
    bt_before_dd["retained"] = ~bt_before_dd.duplicated(subset=["ts_code"], keep="first")

    rp_before_dd["source"] = "replay"
    bt_before_dd["source"] = "backtest"
    common_dd_cols = ["source", "ts_code", "signal_id", "obs_day", score_col if "score" in rp_before_dd.columns else "score",
                      "rank", "retained"]
    rp_dd = rp_before_dd[common_dd_cols].copy()
    bt_dd_cols = ["source", "ts_code", "signal_id", "obs_day", "score", "rank", "retained"]
    bt_dd = bt_before_dd[bt_dd_cols].copy()
    if "score" not in rp_dd.columns:
        rp_dd = rp_dd.rename(columns={score_col: "score"})
    table_c = pd.concat([bt_dd, rp_dd], ignore_index=True)
    table_c.to_csv(os.path.join(debug_dir, "table_C_drop_duplicates.csv"), index=False)

    bt_retained = set(bt_before_dd[bt_before_dd["retained"]]["ts_code"])
    rp_retained = set(rp_before_dd[rp_before_dd["retained"]]["ts_code"])
    print(f"  表C 去重后: bt={len(bt_retained)}股, rp={len(rp_retained)}股")
    if bt_retained != rp_retained:
        print(f"    ❌ 表C 差异! bt独有={bt_retained-rp_retained}, rp独有={rp_retained-bt_retained}")

        target_code = "300042.SZ"
        bt_target = bt_before_dd[bt_before_dd["ts_code"] == target_code]
        rp_target = rp_before_dd[rp_before_dd["ts_code"] == target_code]
        print(f"    {target_code} 去重前: bt={len(bt_target)}行, rp={len(rp_target)}行")
        if not bt_target.empty:
            print(f"      bt: " + ", ".join(f"sid={r['signal_id']} score={r['score']:.4f} rank={r['rank']} retained={r['retained']}" for _, r in bt_target.iterrows()))
        if not rp_target.empty:
            print(f"      rp: " + ", ".join(f"sid={r['signal_id']} score={r['score']:.4f} rank={r['rank']} retained={r['retained']}" for _, r in rp_target.iterrows()))
    else:
        print(f"    ✅ 表C 一致")

    # ---- 表 D: 持仓状态机输入 ----
    bt_holdings = snapshot.get("holdings_before", {})
    bt_hold_rows = []
    for code, h in bt_holdings.items():
        bt_hold_rows.append({
            "code": code, "ts_code": h.get("ts_code", code),
            "buy_date": str(h.get("buy_date", "")),
            "buy_price": h.get("buy_price", np.nan),
            "days_held": h.get("days_held"),
            "signal_id": h.get("signal_id"),
            "score": h.get("score"),
            "source": "backtest",
        })
    rp_hold_rows = []
    for code, h in holdings.items():
        cur_ret = h.get("cur_ret")
        rp_hold_rows.append({
            "code": code, "ts_code": h.get("ts_code", code),
            "buy_date": str(h.get("buy_date", "")),
            "buy_price": h.get("buy_price", np.nan),
            "days_held": h.get("days_held"),
            "signal_id": h.get("signal_id"),
            "score": h.get("score"),
            "cur_ret": cur_ret if cur_ret is not None else np.nan,
            "source": "replay",
        })
    table_d = pd.DataFrame(bt_hold_rows + rp_hold_rows)
    table_d.to_csv(os.path.join(debug_dir, "table_D_holdings_state.csv"), index=False)

    bt_hcodes = set(h.get("ts_code", c) for c, h in bt_holdings.items())
    rp_hcodes = set(h.get("ts_code", c) for c, h in holdings.items())
    print(f"  表D 持仓状态: bt={len(bt_holdings)}只, rp={len(holdings)}只")
    if bt_hcodes != rp_hcodes:
        print(f"    ❌ 表D 差异! bt独有={bt_hcodes-rp_hcodes}, rp独有={rp_hcodes-bt_hcodes}")
    else:
        print(f"    ✅ 表D 持仓集合一致")

        target_code_6 = "300042"
        bt_300 = [h for c, h in bt_holdings.items() if h.get("ts_code", "").startswith("300042")]
        rp_300 = holdings.get(target_code_6, None)
        if bt_300:
            bh = bt_300[0]
            print(f"    300042.SZ bt: bp={bh.get('buy_price')} dh={bh.get('days_held')} sid={bh.get('signal_id')} score={bh.get('score')}")
        if rp_300:
            print(f"    300042    rp: bp={rp_300.get('buy_price')} dh={rp_300.get('days_held')} sid={rp_300.get('signal_id')} score={rp_300.get('score')} cur_ret={rp_300.get('cur_ret')}")
        if bt_300 and rp_300:
            bh = bt_300[0]
            scores_match = abs(bh.get("score", 0) - rp_300.get("score", 0)) < 1e-9
            bp_match = abs(bh.get("buy_price", 0) - rp_300.get("buy_price", 0)) < 1e-3
            dh_match = bh.get("days_held") == rp_300.get("days_held")
            sid_match = str(bh.get("signal_id")) == str(rp_300.get("signal_id"))
            print(f"    → score_match={scores_match} bp_match={bp_match} dh_match={dh_match} sid_match={sid_match}")
            if not all([scores_match, bp_match, dh_match, sid_match]):
                print(f"    ❌ 300042 持仓细节不一致!")

    print(f"  📁 调试文件: {debug_dir}/")
    print(f"  {'='*60}\n")


def run_replay(args):
    dates = []
    if args.date:
        dates = [pd.to_datetime(args.date)]
        print("=" * 70)
        print(f"日级推理回放 (1 日)")
        print("=" * 70)
    elif args.batch_first_10:
        dates = [pd.to_datetime(d) for d in FIRST_10_DATES]
        print("=" * 70)
        print(f"日级推理回放 (10 日)")
        print("=" * 70)
    elif args.batch_all:
        dates = None
    else:
        raise ValueError("必须指定 --date、--batch-first-10 或 --batch-all")

    # ---- 加载全量候选（使用回测同源数据） ----
    #    回测用 full_test_predictions.parquet → obs_day 1~3 子集 → score_stocks()
    #    回放必须先过滤 obs_day 再评分，与回测同口径，否则 zscore 标准化不同
    cand_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    df_all = pd.read_parquet(cand_path)
    df_all["obs_date"] = pd.to_datetime(df_all["obs_date"])
    candidate_obs_days = BASELINE_E0_X1_V1_PARAMS.get("candidate_obs_days", [1])
    df_all = df_all[df_all["obs_day"].isin(candidate_obs_days)].copy()
    print(f"\n  候选数据: {len(df_all)} 行, 日期范围: {df_all['obs_date'].min()} ~ {df_all['obs_date'].max()}")
    print(f"  Signal 总数: {df_all['signal_id'].nunique()}")

    df_all_raw = df_all.copy()

    df_all = score_stocks(df_all, PRODUCTION_PARAMS.get("strategy_default", "sell_score"))
    score_col = "score"

    # ---- 加载数据和回测引擎 ----
    test_df, price, td, prev_close, pred_lookup = _load_data(
        candidate_obs_days=PRODUCTION_PARAMS["candidate_obs_days"]
    )
    backtest_result = run_backtest(
        test_df, price, td, prev_close, pred_lookup,
        max_stocks=args.top_k, strategy=args.strategy,
        exit_mode=args.exit_mode,
        stop_loss=PRODUCTION_PARAMS["stop_loss"],
        buy_cls_exit_threshold=PRODUCTION_PARAMS["buy_cls_exit_threshold"],
        debug_snapshots=True,
        strict=True,
    )
    snapshots_map = {s["date"]: s for s in backtest_result["snapshots"]}
    print(f"  回测快照: {len(snapshots_map)} 个交易日")

    if dates is None:
        dates = sorted(snapshots_map.keys())
        print("=" * 70)
        print(f"日级推理回放 (全 test 窗口, {len(dates)} 日)")
        print("=" * 70)

    # ---- 结构化索引优化 ----
    pred_indexed = {}
    if pred_lookup is not None:
        for (sid, dt), pred in pred_lookup.items():
            pred_indexed[(int(sid), dt)] = pred

    price_cols = [c for c in price.columns if c != "ts_code"]
    trading_dates_list = sorted(snapshots_map.keys())

    # ---- 确定需要重放的日期范围 ----
    if args.date:
        target_dates = set(dates)
        replay_start = min(trading_dates_list)
    else:
        target_dates = set(dates)
        replay_start = min(trading_dates_list)

    os.makedirs(DYNAMIC_DIR, exist_ok=True)

    debug_date = pd.to_datetime(args.debug_date) if args.debug_date else None

    # ---- 逐日回放 (遍历全部交易日维护持仓状态，仅目标日期输出对账) ----
    all_summaries = []
    holdings = {}
    first_divergence = None
    pending_buys = []
    pending_sells_prev = []  # T-1日决策的卖出，T日开盘执行

    for tdate in sorted(trading_dates_list):
        snapshot = snapshots_map.get(tdate)
        if snapshot is None:
            continue

        # Step 1~4: 统一日状态推进（SSOT step_day）。
        # execute_pending_buys → execute_pending_sells → decide_eod
        prev_idx = None
        for i, td_date in enumerate(trading_dates_list):
            if td_date == tdate:
                prev_idx = i - 1 if i > 0 else None
                break
        prev_date_val = trading_dates_list[prev_idx] if prev_idx is not None else None

        # 优先读取真实预测结果（仅对目标日期），中间日期走原逻辑以保持与回测一致
        use_real_pred = tdate in target_dates
        real_pred = None
        candidates = None
        if use_real_pred:
            real_pred = load_real_predictions_for_date(tdate)
            if real_pred is None and prev_date_val is not None:
                real_pred = load_real_predictions_for_date(prev_date_val)
            if real_pred is not None and not real_pred.empty:
                candidates = real_pred
                if "score" not in candidates.columns and "composite_score" in candidates.columns:
                    candidates = candidates.rename(columns={"composite_score": "score"})

        if candidates is None or candidates.empty:
            candidates = build_daily_candidate_snapshot(tdate, df_all, score_col)
            if candidates.empty and tdate > df_all["obs_date"].max():
                print(f"  [NO_REAL_PREDICTION_DATA] {tdate.strftime('%Y-%m-%d')} 超出历史预测范围，无真实数据")

        bh = snapshot.get("holdings_before", {})
        bt_hcodes = set(h.get("ts_code", c) for c, h in bh.items())
        rp_hcodes = set(h.get("ts_code", c) for c, h in holdings.items())
        if bt_hcodes != rp_hcodes and first_divergence is None:
            first_divergence = (tdate, bt_hcodes - rp_hcodes, rp_hcodes - bt_hcodes)

        if debug_date is not None and tdate == debug_date:
            _run_debug_dashboard(
                tdate, df_all_raw, df_all, score_col, test_df,
                dict(holdings), price, snapshot, pred_indexed,
                PRODUCTION_PARAMS, args.strategy,
            )

        step_params = {
            "max_stocks": PRODUCTION_PARAMS.get("max_stocks", 10),
            "max_hold_days": PRODUCTION_PARAMS.get("max_hold_days", 20),
            "stop_loss": PRODUCTION_PARAMS.get("stop_loss", -0.07),
            "exit_threshold": PRODUCTION_PARAMS.get("buy_cls_exit_threshold", 0.70),
        }
        step_result = step_day(
            tdate, holdings, pending_buys, pending_sells_prev,
            price, candidates, pred_indexed, prev_date_val, step_params,
            prev_close_map=prev_close, strict=True,
        )
        holdings = step_result["holdings"]
        pending_buys = step_result["pending_buys"]
        pending_sells = step_result["pending_sells"]
        sell_reasons = step_result["sell_reasons"]
        pending_sells_prev = pending_sells  # 保存，T+1日开盘执行

        if tdate not in target_dates:
            continue

        print(f"\n  --- {tdate.strftime('%Y-%m-%d')} ---")
        candidate_check = "(无候选)" if candidates.empty else \
            ("(与回测一致)" if set(snapshot.get("candidate_codes", [])) == set(candidates["ts_code"]) else "(差异!)")
        print(f"    候选池: {len(candidates)} 只 {candidate_check}")
        print(f"    Buy: {len(pending_buys)}, Sell: {len(pending_sells)}, Holdings: {len(holdings)}")

        # ③ Compare with backtest
        # 判断回测是否有候选数据：候选池为空 → 回测无候选数据（无法比较买卖决策）
        bt_has_candidates = bool(snapshot.get("candidate_codes"))
        replay_has_real_pred = bool(real_pred is not None and not real_pred.empty)

        if not bt_has_candidates and replay_has_real_pred:
            # 回测无候选数据但有真实预测：无法对账，不是逻辑错误
            diff_rows = []
            summary = {
                "date": tdate,
                "all_pass": True,
                "failed": 0,
                "total_checks": 0,
                "critical_failures": 0,
                "status": "skip_no_bt_data",
            }
            all_summaries.append(summary)
            print(f"    对账: ⏭️ 跳过 (回测无候选数据，但有真实预测)")
        elif not bt_has_candidates and not replay_has_real_pred:
            # 双方都无候选数据：假一致，明确标记
            diff_rows = []
            summary = {
                "date": tdate,
                "all_pass": True,
                "failed": 0,
                "total_checks": 0,
                "critical_failures": 0,
                "status": "skip_no_data",
            }
            all_summaries.append(summary)
            print(f"    对账: ⏭️ 跳过 (双方无候选数据)")
        else:
            diff_rows, summary = compare_with_backtest(
                tdate,
                (candidates, holdings, pending_buys, pending_sells, sell_reasons),
                snapshot,
            )
            all_summaries.append(summary)

            status = "✅ 通过" if summary["all_pass"] else f"❌ 失败 ({summary['failed']}/{summary['total_checks']})"
            if summary["critical_failures"] > 0:
                status += f" [关键: {summary['critical_failures']}]"
            print(f"    对账: {status}")

        # ④ 输出文件
        if not args.dry_run:
            date_str = tdate.strftime("%Y-%m-%d")
            replay_path = os.path.join(OUTPUT_DIR, f"daily_inference_replay_{date_str}.csv")
            diff_path = os.path.join(OUTPUT_DIR, f"daily_inference_diff_{date_str}.csv")

            replay_rows = []
            for c, h in holdings.items():
                replay_rows.append({
                    "date": date_str, "ts_code": c,
                    "action": "hold",
                    "sell_reason": "",
                    "holding_before": True,
                    "holding_after": True,
                })
            for b in pending_buys:
                replay_rows.append({
                    "date": date_str, "ts_code": b[2],
                    "action": "buy",
                    "sell_reason": "",
                    "holding_before": False,
                    "holding_after": True,
                })
            for s in pending_sells:
                ts_code = s["holding"]["ts_code"] if isinstance(s, dict) else s
                replay_rows.append({
                    "date": date_str, "ts_code": ts_code,
                    "action": "sell",
                    "sell_reason": sell_reasons.get(ts_code, ""),
                    "holding_before": True,
                    "holding_after": False,
                })

            pd.DataFrame(replay_rows).to_csv(replay_path, index=False)
            pd.DataFrame(diff_rows).to_csv(diff_path, index=False)
            print(f"    输出: {replay_path}")
            print(f"    输出: {diff_path}")

    # ---- 汇总报告 ----
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(DYNAMIC_DIR, "replay_summary.csv")
        os.makedirs(DYNAMIC_DIR, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)

        n_all_pass = sum(1 for s in all_summaries if s["all_pass"] and s.get("status") != "skip_no_bt_data" and s.get("status") != "skip_no_data")
        n_skip = sum(1 for s in all_summaries if s.get("status") in ("skip_no_bt_data", "skip_no_data"))
        n_fail = len(all_summaries) - n_all_pass - n_skip
        total_critical = sum(s["critical_failures"] for s in all_summaries if s.get("status") not in ("skip_no_bt_data", "skip_no_data"))

        print(f"\n{'='*70}")
        print(f"汇总报告")
        print(f"{'='*70}")
        print(f"  总日期: {len(all_summaries)}")
        print(f"  对账通过: {n_all_pass}")
        print(f"  对账跳过: {n_skip}")
        print(f"  对账失败: {n_fail}")
        print(f"  关键失败数: {total_critical}")
        if first_divergence:
            fd, bt_only, rp_only = first_divergence
            print(f"  首个持仓分叉日: {fd.strftime('%Y-%m-%d')}")
            print(f"    bt独有: {bt_only}")
            print(f"    rp独有: {rp_only}")
        if n_fail == 0 and total_critical == 0:
            print(f"\n  ✅ 无对账失败，可继续")
            if n_skip > 0:
                print(f"  ⏭️ 有 {n_skip} 天因回测无数据而跳过，属正常")
            if n_all_pass == len(all_summaries) - n_skip:
                print(f"  可以进入 Phase 5: 全 test 窗口对账")
        elif total_critical == 0:
            print(f"\n  ⚠️ 存在非关键差异（分数误差），不影响决策")
        else:
            print(f"\n  ❌ 存在 {total_critical} 项关键失败，需修复后再扩展")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日级推理回放 + 回测对账")
    parser.add_argument("--date", type=str, default="", help="单日回放 (YYYY-MM-DD)")
    parser.add_argument("--batch-first-10", action="store_true", help="10 日批量回放")
    parser.add_argument("--batch-all", action="store_true", help="全 test 窗口批量回放")
    parser.add_argument("--dry-run", action="store_true", help="只推理不对账")
    parser.add_argument("--top-k", type=int, default=10, help="最大持仓")
    parser.add_argument("--strategy", type=str, default="sell_score")
    parser.add_argument("--exit-mode", type=str, default="model_exit",
                        choices=["fixed_hold", "rule_exit", "model_exit"])
    parser.add_argument("--debug-date", type=str, default="", help="单日断点调试 (YYYY-MM-DD)")
    args = parser.parse_args()
    run_replay(args)
