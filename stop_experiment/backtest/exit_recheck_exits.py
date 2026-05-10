#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exit 公平复查：buy_reg 分位-风险诊断 + X3 衰减联动 + X4 数据驱动阈值

Purpose:
    Step 3a: pred_buy_reg 分位段 vs 真实 mae_5/10/20（找风险恶化拐点）
    Step 3b: X3 公平重做 — sell_reg 纯衰减 / rank 跌出等
    Step 3c: X4 公平重做 — 用诊断阈值替代拍脑袋 0.05

Pipeline Position:
    Step 3 复查。
    上游：entry_recheck_entries.py
    下游：out_of_sample_validator.py

Inputs:
    - stop_experiment/output/full_test_predictions.parquet

Outputs:
    - output/backtest/buy_reg_quantile_risk.csv
    - output/backtest/exit_recheck_exits.csv

How to Run:
    python stop_experiment/backtest/exit_recheck_exits.py

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


# ==================== Step 3a: pred_buy_reg 分位-风险诊断 ====================

def buy_reg_risk_diagnosis(pred_path=None):
    """分析 pred_buy_reg 各分位段对应的真实后续风险

    从 full_test_predictions.parquet 读取，对 obs_date 后 5/10/20 日计算：
    - mae_5 / mae_10 / mae_20（平均最劣波动）
    - mae < -7% 比例
    - mae < -10% 比例
    """
    if pred_path is None:
        pred_path = os.path.join(
            os.path.dirname(BACKTEST_DIR), "full_test_predictions.parquet"
        )

    pred_df = pd.read_parquet(pred_path)
    if "pred_buy_reg" not in pred_df.columns:
        print("  ⚠️ full_test_predictions 无 pred_buy_reg 列，跳过诊断")
        return None

    # 加载价格数据用于 MAE 计算
    test_df, price, _, _, _ = _load_data(candidate_obs_days=[1])
    price_close = price.xs("close", level=0, axis=1)

    pred_df = pred_df.copy()
    pred_df["obs_date"] = pd.to_datetime(pred_df["obs_date"])

    # 分位数分段
    br_vals = pred_df["pred_buy_reg"].dropna()
    q_edges = [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    q_labels = ["P0-P10", "P10-P25", "P25-P50", "P50-P75", "P75-P90", "P90-P100"]
    quantiles = br_vals.quantile(q_edges).tolist()

    print("  pred_buy_reg 分位数:")
    for q, v in zip(q_edges, quantiles):
        print(f"    Q{q:.0%}: {v:.6f}")

    rows = []
    for i, label in enumerate(q_labels):
        lo, hi = quantiles[i], quantiles[i + 1]
        seg = pred_df[(pred_df["pred_buy_reg"] >= lo) & (pred_df["pred_buy_reg"] < hi)]
        if i == len(q_labels) - 1:
            seg = pred_df[pred_df["pred_buy_reg"] >= lo]  # last bin inclusive

        if len(seg) == 0:
            continue

        mae5_vals, mae10_vals, mae20_vals = [], [], []

        for _, row in seg.iterrows():
            code = str(row.get("ts_code", ""))
            code_clean = code.split(".")[0] if "." in code else code
            obs_d = row["obs_date"]

            if code_clean not in price_close.columns:
                continue

            for horizon, vals_list in [(5, mae5_vals), (10, mae10_vals), (20, mae20_vals)]:
                end_d = obs_d + pd.Timedelta(days=horizon)
                end_d = min(end_d, price_close.index.max())
                mask = (price_close.index > obs_d) & (price_close.index <= end_d)
                future_prices = price_close.loc[mask, code_clean].dropna()
                if len(future_prices) == 0:
                    vals_list.append(np.nan)
                else:
                    obs_price = price_close.loc[obs_d, code_clean] if obs_d in price_close.index else np.nan
                    if pd.isna(obs_price) or obs_price <= 0:
                        vals_list.append(np.nan)
                    else:
                        vals_list.append((future_prices.min() - obs_price) / obs_price)

        rows.append({
            "quantile_range": label,
            "br_lo": lo,
            "br_hi": hi if i < len(q_labels) - 1 else float("inf"),
            "n_samples": len(seg),
            "mae_5_mean": np.nanmean(mae5_vals) if mae5_vals else np.nan,
            "mae_10_mean": np.nanmean(mae10_vals) if mae10_vals else np.nan,
            "mae_20_mean": np.nanmean(mae20_vals) if mae20_vals else np.nan,
            "mae_5_lt7pct": np.nanmean([v < -0.07 for v in mae5_vals]) * 100 if mae5_vals else np.nan,
            "mae_10_lt7pct": np.nanmean([v < -0.07 for v in mae10_vals]) * 100 if mae10_vals else np.nan,
            "mae_20_lt7pct": np.nanmean([v < -0.07 for v in mae20_vals]) * 100 if mae20_vals else np.nan,
            "mae_5_lt10pct": np.nanmean([v < -0.10 for v in mae5_vals]) * 100 if mae5_vals else np.nan,
            "mae_10_lt10pct": np.nanmean([v < -0.10 for v in mae10_vals]) * 100 if mae10_vals else np.nan,
            "mae_20_lt10pct": np.nanmean([v < -0.10 for v in mae20_vals]) * 100 if mae20_vals else np.nan,
        })

    df_risk = pd.DataFrame(rows)
    return df_risk, quantiles


# ==================== Exit 实验运行 ====================

def run_exit_recheck(label, test_df, price, td, prev, pred,
                     exit_mode, exit_sub_mode=None,
                     exit_th=0.70, buy_reg_th=None, baseline=BASELINE_E0_X1_V1_PARAMS):
    """运行一组 Exit 复查"""
    result = run_backtest(
        test_df, price, td, prev, pred,
        max_stocks=10, strategy="sell_score",
        exit_mode=exit_mode, stop_loss=-0.07,
        buy_cls_exit_threshold=exit_th,
        exit_sub_mode=exit_sub_mode,
        buy_reg_exit_threshold=buy_reg_th,
        strict=True,
    )
    s = compute_summary(result)
    trades_df = result.get("trades_df", pd.DataFrame())

    # 真实 MFE/MAE
    mfe_raw, mae_raw = compute_trade_mfe_mae(trades_df, price)
    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["mfe"] = mfe_raw
        trades_df["mae"] = mae_raw
    mfe_mean = np.nanmean(mfe_raw) if mfe_raw else np.nan
    mae_mean = np.nanmean(mae_raw) if mae_raw else np.nan
    mfe_mae_diff = mfe_mean - abs(mae_mean) if not (np.isnan(mfe_mean) or np.isnan(mae_mean)) else np.nan

    post5_rets, post10_rets = compute_exit_quality(trades_df, price)
    post5_mean = np.nanmean(post5_rets) if post5_rets else np.nan
    post10_mean = np.nanmean(post10_rets) if post10_rets else np.nan

    fp5 = np.nanmean([v > 0.05 for v in post5_rets]) * 100 if post5_rets else np.nan

    n_model = int((trades_df["sell_reason"] == "model_risk").sum()) if not trades_df.empty else 0
    n_stop = int((trades_df["sell_reason"] == "stop_loss").sum()) if not trades_df.empty else 0
    n_max_hold = int((trades_df["sell_reason"] == "max_hold").sum()) if not trades_df.empty else 0
    n_total = s.get("n_trades", 0)
    model_pct = n_model / n_total * 100 if n_total > 0 else 0

    return {
        "label": label,
        "exit_sub_mode": str(exit_sub_mode),
        "buy_reg_th": buy_reg_th,
        "nav": s.get("final_nav", np.nan),
        "sharpe": s.get("sharpe", np.nan),
        "mdd": s.get("max_dd", np.nan),
        "win_rate": s.get("win_rate", np.nan),
        "avg_hold_days": s.get("avg_hold_days", np.nan),
        "n_trades": n_total,
        "mfe_mae_diff_pct": mfe_mae_diff * 100 if not np.isnan(mfe_mae_diff) else np.nan,
        "post5_ret_pct": post5_mean * 100 if not np.isnan(post5_mean) else np.nan,
        "post10_ret_pct": post10_mean * 100 if not np.isnan(post10_mean) else np.nan,
        "fp5_rate_pct": fp5,
        "exits_model": n_model,
        "exits_stop": n_stop,
        "exits_max_hold": n_max_hold,
        "model_exit_pct": model_pct,
    }


def main():
    EXIT_TH = 0.70
    baseline = BASELINE_E0_X1_V1_PARAMS

    print("=" * 70)
    print(f"Step 3: Exit 公平复查 (E0 Entry, exit_th={EXIT_TH})")
    print("=" * 70)

    # 数据准备
    test_df, price, td, prev, pred = _load_data(candidate_obs_days=[1])
    test_df["trading_date"] = test_df["obs_date"]
    test_df["score"] = test_df["pred_sell_reg"]
    test_df["sell_score"] = test_df["pred_sell_reg"]
    print(f"  数据: {len(test_df)} 条, {len(td)} 交易日\n")

    # === Step 3a: pred_buy_reg 分位-风险诊断 ===
    print("--- 3a. pred_buy_reg 分位-风险诊断 ---")
    df_risk, quantiles = buy_reg_risk_diagnosis()

    if df_risk is not None:
        path_risk = os.path.join(BACKTEST_DIR, "buy_reg_quantile_risk.csv")
        df_risk.to_csv(path_risk, index=False)

        print("\n  pred_buy_reg 分位段 vs 未来风险:")
        risk_cols = ["quantile_range", "n_samples", "mae_5_mean", "mae_10_mean", "mae_20_mean",
                     "mae_10_lt7pct", "mae_10_lt10pct"]
        available_r = [c for c in risk_cols if c in df_risk.columns]
        print(df_risk[available_r].to_string(index=False))

        # 找风险恶化拐点
        for _, row in df_risk.iterrows():
            mae10_lt7 = row.get("mae_10_lt7pct", np.nan)
            if not np.isnan(mae10_lt7) and mae10_lt7 > 40:
                print(f"\n  ⚠️ 风险恶化拐点: {row['quantile_range']} "
                      f"(br∈[{row['br_lo']:.4f}, {row['br_hi']:.4f}), "
                      f"mae_10<-7% = {mae10_lt7:.1f}%")
                break
        else:
            print("\n  ℹ️ 无显著风险恶化拐点（所有分段 mae10<-7% < 40%）")

        print(f"\n  输出: {path_risk}")

    # === Step 3b + 3c: Exit 复查 ===
    print("\n--- 3b+3c. Exit 结构复查 ---")
    rows = []

    # X0 + X1 基线
    print("  X0 (规则退出)...", end=" ", flush=True)
    r = run_exit_recheck("X0 (规则)", test_df, price, td, prev, pred, exit_mode="rule_exit")
    rows.append(r)
    print(f"NAV={r['nav']:.4f}")

    print("  X1 (buy_cls>0.70)...", end=" ", flush=True)
    r = run_exit_recheck("X1 (基线)", test_df, price, td, prev, pred,
                         exit_mode="model_exit", exit_sub_mode=None)
    rows.append(r)
    print(f"NAV={r['nav']:.4f}, model={r['exits_model']}/{r['n_trades']}")

    # X3a: sell_reg pure decay
    print("  X3a (sell_reg 纯衰减)...", end=" ", flush=True)
    r = run_exit_recheck("X3a", test_df, price, td, prev, pred,
                         exit_mode="model_exit", exit_sub_mode="sell_decay")
    rows.append(r)
    print(f"NAV={r['nav']:.4f}, model={r['exits_model']}/{r['n_trades']}")

    # X4 用诊断阈值
    if df_risk is not None and len(quantiles) >= 7:
        # 选 P90 = quantiles[5]（top 10% 最悲观的 buy_reg）
        br_th_p90 = abs(quantiles[5])  # buy_reg is negative, take abs
        br_th_p95 = abs(quantiles[-1]) if len(quantiles) > 6 else br_th_p90 * 1.2

        # 检查阈值合理性
        if br_th_p90 < 0.03:
            br_th_p90 = 0.08  # fallback
        if br_th_p95 < 0.05:
            br_th_p95 = 0.12

        print(f"\n  X4 诊断阈值: P90={br_th_p90:.4f}, P95={br_th_p95:.4f}")

        for br_k, suffix in [(br_th_p90, "P90"), (br_th_p95, "P95")]:
            label = f"X4 (or_buy_reg, br_th={br_k:.3f})"
            print(f"  {label}...", end=" ", flush=True)
            r = run_exit_recheck(f"X4_{suffix}", test_df, price, td, prev, pred,
                                 exit_mode="model_exit", exit_sub_mode="or_buy_reg",
                                 buy_reg_th=br_k)
            rows.append(r)
            print(f"NAV={r['nav']:.4f}, model={r['exits_model']}/{r['n_trades']} "
                  f"({r['model_exit_pct']:.1f}%)")
    else:
        print("\n  ⚠️ 无诊断数据，跳过 X4")

    # 汇总
    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "exit_recheck_exits.csv")
    df_out.to_csv(path, index=False)

    print("\n" + "=" * 70)
    print("Exit 复查结果汇总")
    print("=" * 70)
    key_cols = ["label", "nav", "sharpe", "mdd", "n_trades", "avg_hold_days",
                "mfe_mae_diff_pct", "post5_ret_pct", "post10_ret_pct",
                "model_exit_pct", "exits_model", "exits_stop", "exits_max_hold"]
    available = [c for c in key_cols if c in df_out.columns]
    print(df_out[available].to_string(index=False))

    # 判断
    x1 = df_out[df_out["label"] == "X1 (基线)"].iloc[0]
    print(f"\nX1 基线: NAV={x1['nav']:.4f}, model_pct={x1.get('model_exit_pct', np.nan):.1f}%")

    for _, row in df_out.iterrows():
        if "X1" in row["label"] or "X0" in row["label"]:
            continue
        nav_ok = row["nav"] >= x1["nav"] * 0.90
        mdel_ok = row.get("model_exit_pct", np.nan)
        mdel_str = f"model={mdel_ok:.1f}%" if not np.isnan(mdel_ok) else "model=N/A"
        print(f"  {row['label']}: NAV={row['nav']:.4f} "
              f"({'≥' if nav_ok else '<'}90%x1), {mdel_str}")

    # X3 判断
    x3 = df_out[df_out["label"] == "X3a"]
    if not x3.empty and x3.iloc[0]["nav"] >= x1["nav"] * 0.90:
        print("  💡 X3 sell_reg 衰减有边际价值（NAV≥90%x1）")
    else:
        print("  ❌ X3 sell_reg 衰减无边际价值，不关闭 X3 方向但暂不追")

    # X4 判断
    x4_rows = [r for r in rows if "X4" in r.get("label", "")]
    if x4_rows:
        ok_x4 = any(r["model_exit_pct"] <= 70 and r["nav"] >= x1["nav"] * 0.80 for r in x4_rows)
        if ok_x4:
            print("  ✅ X4 有合理版本（model%≤70%, NAV≥80%x1），buy_reg 有边际值")
        else:
            print("  ❌ 所有 X4 版本 model%>70% 或 NAV<80%x1，正式关闭 buy_reg 退出线")

    print(f"\n  输出: {path}")


if __name__ == "__main__":
    main()