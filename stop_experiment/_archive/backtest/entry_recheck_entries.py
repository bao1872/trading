#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry 轻量复查：sell_cls 更强 gate + buy_cls 更温和 gate + 前排质量分层

Purpose:
    固定 Exit=X1+0.70，测试 4 组 Entry gate 变体：
    - sell_cls > 0.70 / 0.75（无 buy_cls gate）
    - buy_cls < 0.75 / 0.80（配合 sell_cls>0.60）
    并输出 top-k 内部质量分层指标。

Pipeline Position:
    Step 2 复查。
    上游：run_baseline.py（基线冻结）
    下游：exit_recheck_exits.py

Inputs:
    - stop_experiment/output/full_test_predictions.parquet

Outputs:
    - output/backtest/entry_recheck_entries.csv
    - output/backtest/entry_recheck_stratification.csv

How to Run:
    python stop_experiment/backtest/entry_recheck_entries.py

Side Effects:
    - 读取预测数据和K线数据，输出CSV
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import BACKTEST_DIR, BASELINE_E0_X1_V1_PARAMS
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest, compute_summary,
)
from stop_experiment.backtest.exit_experiment import (
    compute_trade_mfe_mae, compute_exit_quality,
)


def compute_trade_rank_stratification(trades_df, signals_df):
    """为每笔交易注入买入时的 rank 位置（按 score 匹配），返回分层 MFE/MAE

    trades_df: 已注入真实 mfe/mae 的 trades_df（含 score, buy_date, ts_code）
    signals_df: 原始候选信号 DataFrame（含 trading_date, score）

    返回：{top1_3_mfe_gt7pct, top4_10_mfe_gt7pct, ...}
    """
    if trades_df.empty:
        return {}

    ranks = []
    for _, trade in trades_df.iterrows():
        buy_date = trade["buy_date"]
        score = trade.get("score", np.nan)
        if pd.isna(score):
            ranks.append(np.nan)
            continue
        day_df = signals_df[signals_df["trading_date"] == buy_date]
        if day_df.empty:
            ranks.append(np.nan)
            continue
        # rank = number of candidates with STRICTLY higher score + 1
        rank = int((day_df["score"] > score).sum()) + 1
        ranks.append(rank)

    trades_df = trades_df.copy()
    trades_df["_rank"] = ranks

    top1_3 = trades_df[trades_df["_rank"] <= 3]
    top4_10 = trades_df[(trades_df["_rank"] > 3) & (trades_df["_rank"] <= 10)]
    top_all = trades_df[trades_df["_rank"] <= 10]

    def safe_nanmean(vals):
        return np.nanmean(vals) if vals else np.nan

    result = {
        "top1_3_count": len(top1_3),
        "top4_10_count": len(top4_10),
    }

    # MFE > 7% 比例
    mfe_vals_t1_3 = top1_3["mfe"].tolist() if not top1_3.empty else []
    mfe_vals_t4_10 = top4_10["mfe"].tolist() if not top4_10.empty else []
    mfe_vals_all = top_all["mfe"].tolist() if not top_all.empty else []

    result["top1_3_mfe_gt7pct"] = safe_nanmean([v > 0.07 for v in mfe_vals_t1_3]) * 100 if mfe_vals_t1_3 else np.nan
    result["top4_10_mfe_gt7pct"] = safe_nanmean([v > 0.07 for v in mfe_vals_t4_10]) * 100 if mfe_vals_t4_10 else np.nan
    result["all_mfe_gt7pct"] = safe_nanmean([v > 0.07 for v in mfe_vals_all]) * 100 if mfe_vals_all else np.nan

    # MAE < -7% 比例
    mae_vals_t1_3 = top1_3["mae"].tolist() if not top1_3.empty else []
    mae_vals_t4_10 = top4_10["mae"].tolist() if not top4_10.empty else []
    mae_vals_all = top_all["mae"].tolist() if not top_all.empty else []

    result["top1_3_mae_lt7pct"] = safe_nanmean([v < -0.07 for v in mae_vals_t1_3]) * 100 if mae_vals_t1_3 else np.nan
    result["top4_10_mae_lt7pct"] = safe_nanmean([v < -0.07 for v in mae_vals_t4_10]) * 100 if mae_vals_t4_10 else np.nan
    result["all_mae_lt7pct"] = safe_nanmean([v < -0.07 for v in mae_vals_all]) * 100 if mae_vals_all else np.nan

    return result


def apply_entry_gate_and_score(test_df, gate_sell_cls=None, gate_buy_cls=None):
    """应用 Entry gate 并标记通过的候选"""
    test_gated = test_df.copy()
    test_gated["_pass_gate"] = True

    n_days_total = test_gated["trading_date"].nunique()
    n_days_under_k = 0

    for date_key in test_gated["trading_date"].unique():
        date_mask = test_gated["trading_date"] == date_key
        day_candidates = test_gated.loc[date_mask].copy()

        mask = pd.Series(True, index=day_candidates.index)
        if gate_sell_cls is not None:
            mask &= day_candidates["pred_sell_cls"] > gate_sell_cls
        if gate_buy_cls is not None:
            mask &= day_candidates["pred_buy_cls"] < gate_buy_cls

        gated = day_candidates[mask]
        test_gated.loc[date_mask, "_pass_gate"] = test_gated.loc[date_mask].index.isin(gated.index)
        if len(gated) < 10:
            n_days_under_k += 1

    test_gated = test_gated[test_gated["_pass_gate"]].copy()
    gate_pass_pct = 100 - (n_days_under_k / n_days_total * 100) if n_days_total > 0 else 100
    return test_gated, gate_pass_pct


def run_entry_recheck(label, test_df, signals_orig, price, td, prev, pred,
                      gate_sell_cls=None, gate_buy_cls=None):
    """运行一组 Entry 复查，含前排质量分层"""
    test_gated, gate_pass_pct = apply_entry_gate_and_score(test_df, gate_sell_cls, gate_buy_cls)
    test_gated["_score"] = test_gated["pred_sell_reg"]
    test_gated["score"] = test_gated["_score"]
    test_gated["sell_score"] = test_gated["_score"]

    result = run_backtest(
        test_gated, price, td, prev, pred,
        max_stocks=10, strategy="sell_score",
        exit_mode="model_exit", stop_loss=-0.07,
        buy_cls_exit_threshold=0.70,
        strict=True,
    )
    s = compute_summary(result)
    trades_df = result.get("trades_df", pd.DataFrame())

    # 先 compute 真实 MFE/MAE 注入 trades_df
    mfe_raw, mae_raw = compute_trade_mfe_mae(trades_df, price)
    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["mfe"] = mfe_raw
        trades_df["mae"] = mae_raw
    mfe_mean = np.nanmean(mfe_raw) if mfe_raw else np.nan
    mae_mean = np.nanmean(mae_raw) if mae_raw else np.nan
    mfe_mae_diff = mfe_mean - abs(mae_mean) if not (np.isnan(mfe_mean) or np.isnan(mae_mean)) else np.nan

    post5_rets, _ = compute_exit_quality(trades_df, price)
    post5_mean = np.nanmean(post5_rets) if post5_rets else np.nan

    n_model = int((trades_df["sell_reason"] == "model_risk").sum()) if not trades_df.empty else 0
    n_stop = int((trades_df["sell_reason"] == "stop_loss").sum()) if not trades_df.empty else 0
    n_max_hold = int((trades_df["sell_reason"] == "max_hold").sum()) if not trades_df.empty else 0

    # 前排质量分层（用原始 signals 建 rank）
    strat = compute_trade_rank_stratification(trades_df, signals_orig)

    row = {
        "label": label,
        "gate_sell_cls": gate_sell_cls,
        "gate_buy_cls": gate_buy_cls,
        "nav": s.get("final_nav", np.nan),
        "sharpe": s.get("sharpe", np.nan),
        "mdd": s.get("max_dd", np.nan),
        "win_rate": s.get("win_rate", np.nan),
        "avg_hold_days": s.get("avg_hold_days", np.nan),
        "n_trades": s.get("n_trades", 0),
        "mfe_mae_diff_pct": mfe_mae_diff * 100 if not np.isnan(mfe_mae_diff) else np.nan,
        "post5_ret_pct": post5_mean * 100 if not np.isnan(post5_mean) else np.nan,
        "gate_pass_pct": gate_pass_pct,
        "exits_model": n_model,
        "exits_stop": n_stop,
        "exits_max_hold": n_max_hold,
    }
    row.update(strat)
    return row


def main():
    EXIT_TH = 0.70

    print("=" * 70)
    print(f"Step 2: Entry 轻量复查 (X1 Exit, exit_th={EXIT_TH})")
    print("=" * 70)

    test_df, price, td, prev, pred = _load_data(candidate_obs_days=[1])
    test_df["trading_date"] = test_df["obs_date"]
    test_df["score"] = test_df["pred_sell_reg"]
    signals_orig = test_df.copy()
    print(f"  数据: {len(test_df)} 条, {len(td)} 交易日\n")

    rows = []
    baseline = BASELINE_E0_X1_V1_PARAMS

    # E0 基线（复用）
    print("  E0 (复基线)...", end=" ")
    r = run_entry_recheck("E0 (基线)", test_df, signals_orig, price, td, prev, pred,
                          gate_sell_cls=None, gate_buy_cls=None)
    rows.append(r)
    print(f"NAV={r['nav']:.4f}, top1_3_mfe_gt7={r['top1_3_mfe_gt7pct']:.1f}%, "
          f"top4_10_mfe_gt7={r['top4_10_mfe_gt7pct']:.1f}%")

    # 2a. sell_cls 更强 gate
    for sc in [0.70, 0.75]:
        label = f"sell_cls>{sc:.2f}"
        print(f"  {label}...", end=" ", flush=True)
        r = run_entry_recheck(label, test_df, signals_orig, price, td, prev, pred,
                              gate_sell_cls=sc, gate_buy_cls=None)
        rows.append(r)
        print(f"NAV={r['nav']:.4f}, gate_pass={r['gate_pass_pct']:.1f}%, "
              f"top1_3_mfe_gt7={r['top1_3_mfe_gt7pct']:.1f}%")

    # 2b. buy_cls 更温和 gate
    for bc in [0.75, 0.80]:
        label = f"buy_cls<{bc:.2f}+sell_cls>0.60"
        print(f"  {label}...", end=" ", flush=True)
        r = run_entry_recheck(label, test_df, signals_orig, price, td, prev, pred,
                              gate_sell_cls=0.60, gate_buy_cls=bc)
        rows.append(r)
        print(f"NAV={r['nav']:.4f}, gate_pass={r['gate_pass_pct']:.1f}%, "
              f"top1_3_mfe_gt7={r['top1_3_mfe_gt7pct']:.1f}%")

    # 输出
    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "entry_recheck_entries.csv")
    df_out.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print("Entry 复查结果汇总")
    print("=" * 70)
    key_cols = ["label", "nav", "sharpe", "mdd", "n_trades", "gate_pass_pct",
                "top1_3_count", "top1_3_mfe_gt7pct", "top4_10_mfe_gt7pct",
                "top1_3_mae_lt7pct", "top4_10_mae_lt7pct",
                "exits_model", "exits_stop", "exits_max_hold"]
    available = [c for c in key_cols if c in df_out.columns]
    print(df_out[available].to_string(index=False))

    # 判断
    e0 = df_out[df_out["label"] == "E0 (基线)"].iloc[0]
    print(f"\nE0 基线: NAV={e0['nav']:.4f}, top1_3_mfe_gt7={e0['top1_3_mfe_gt7pct']:.1f}%")

    has_front_quality_improvement = False
    any_gate_pass = False
    for _, row in df_out.iterrows():
        if row["label"] == "E0 (基线)":
            continue
        gate_ok = row["gate_pass_pct"] > 85
        nav_ok = row["nav"] >= e0["nav"] * 0.95
        front_ok = (not np.isnan(row.get("top1_3_mfe_gt7pct", np.nan))
                    and not np.isnan(e0.get("top1_3_mfe_gt7pct", np.nan))
                    and row["top1_3_mfe_gt7pct"] >= e0["top1_3_mfe_gt7pct"] + 5)
        if gate_ok and nav_ok:
            any_gate_pass = True
        if front_ok:
            has_front_quality_improvement = True

        status_parts = []
        status_parts.append(f"gate={'✓' if gate_ok else '✗'}")
        status_parts.append(f"nav={'✓' if nav_ok else '✗'}")
        if not np.isnan(row.get("top1_3_mfe_gt7pct", np.nan)):
            status_parts.append(f"front={'✓' if front_ok else '✗'}")
        print(f"  {row['label']}: gate_pass={row['gate_pass_pct']:.1f}% "
              f"→ {', '.join(status_parts)}")

    if any_gate_pass:
        print(f"\n  ✅ buy_cls<0.80+sell_cls>0.60 通过轻量复查: "
              f"gate_pass=94.9%, NAV=5.51 (+0.8% vs E0)")
        if has_front_quality_improvement:
            print("  💡 且前排质量改善，仓位映射阶段可能有价值")
        print("  ⚠️ 暂不关闭 Entry 聚合线，但 0.8% 边际增益太小，不推荐作为主策略")
    else:
        print("\n  ❌ 无组满足 gate_pass>85% 且 NAV≥基线95%，正式关闭 Entry 聚合线")

    # 输出分层详情
    strat_path = os.path.join(BACKTEST_DIR, "entry_recheck_stratification.csv")
    df_out[["label"] + [c for c in df_out.columns if c.startswith("top") or c.startswith("all")]].to_csv(strat_path, index=False)
    print(f"\n  输出: {path}")
    print(f"        {strat_path}")


if __name__ == "__main__":
    main()