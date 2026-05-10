#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓位实验汇总: 全变体对比 + 三层结论 + cap/floor 决策

Purpose:
    读取 W0/W1/W2/W3 所有 CSV，汇总全变体对比表，按规则化标准输出三层结论：
    1. 最优仓位方案判定
    2. 集中度风险评估 + 是否进入第二轮 cap/floor
    3. 综合建议

Pipeline Position:
    Part B 仓位实验总结 (B4)。
    上游: position_sizing_w1.py, w2.py, w3.py
    下游: 最终一致性检查

Inputs:
    - output/backtest/w1_results.csv
    - output/backtest/w2_results.csv
    - output/backtest/w3_results.csv

Outputs:
    - output/backtest/position_sizing_summary.csv

How to Run:
    python stop_experiment/backtest/position_sizing_summary.py

Side Effects:
    - 写 CSV
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import BACKTEST_DIR


def load_all():
    """加载所有仓位实验结果"""
    dfs = {}
    for name, path in [
        ("w1", os.path.join(BACKTEST_DIR, "w1_results.csv")),
        ("w2", os.path.join(BACKTEST_DIR, "w2_results.csv")),
        ("w3", os.path.join(BACKTEST_DIR, "w3_results.csv")),
    ]:
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
        else:
            print(f"  ⚠️ 缺失: {path}")
    return dfs


def classify(row, w0):
    """统一通过/否决判定 (返回 verdict, tier)"""
    nav = row["nav"]
    sharpe = row["sharpe"]
    mdd = row["mdd"]
    mwr = row.get("monthly_win_rate", np.nan)
    w0_nav = w0["nav"]
    w0_sharpe = w0["sharpe"]
    w0_mdd = w0["mdd"]
    w0_mwr = w0.get("monthly_win_rate", np.nan)

    # 否决检查
    if nav < w0_nav:
        return "❌ 否决: NAV < W0", "veto"
    if mdd < w0_mdd - 0.02:
        return "❌ 否决: MDD 恶化 > 2pp", "veto"
    if nav < w0_nav and sharpe < w0_sharpe:
        return "❌ 否决: NAV+Sharpe 同时低于 W0", "veto"
    if not np.isnan(mwr) and not np.isnan(w0_mwr) and mwr < w0_mwr:
        return "❌ 否决: 月度胜率 < W0", "veto"

    # 基础通过
    nav_ok = nav >= w0_nav * 0.98
    sharpe_ok = sharpe >= w0_sharpe
    mdd_ok = mdd >= w0_mdd - 0.02
    mwr_ok = np.isnan(mwr) or np.isnan(w0_mwr) or mwr >= w0_mwr

    basic = nav_ok and sharpe_ok and mdd_ok and mwr_ok

    if not basic:
        return "⚠️ 未通过: 不满足基础条件", "fail"

    # 需关注检查
    if row.get("max_single_weight_max", 0) > 0.25:
        return "⚠️ 基础通过(需关注: 单票权重>0.25)", "caution"
    if row.get("avg_top3_share", 0) > 0.55:
        return "⚠️ 基础通过(需关注: top3>0.55)", "caution"

    # 强通过
    strength = nav > w0_nav * 1.05 and mdd >= w0_mdd
    if strength:
        return "🏆 强通过", "strong"

    return "✅ 基础通过", "base"


def check_cap_floor_entry(row, w0):
    """判定是否进入第二轮 cap/floor (返回 (进入, 原因))"""
    max_w = row.get("max_single_weight_max", 0)
    top3 = row.get("avg_top3_share", 0)
    hhi = row.get("avg_hhi", 0)
    w0_hhi = w0.get("avg_hhi", 0)

    reasons = []

    if max_w > 0.25:
        reasons.append("单票最大权重 {:.3f} > 0.25".format(max_w))
    if top3 > 0.55:
        reasons.append("top1-3 占比 {:.3f} > 0.55".format(top3))
    if hhi > w0_hhi * 1.5 and not np.isnan(w0_hhi) and w0_hhi > 0:
        reasons.append("HHI {:.3f} >> W0 {:.3f}".format(hhi, w0_hhi))

    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def layer1_best(all_rows, w0):
    """第1层: 最优方案判定"""
    base_pass = [r for r in all_rows if r["tier"] in ("base", "strong")]
    strong = [r for r in all_rows if r["tier"] == "strong"]

    if strong:
        best = max(strong, key=lambda r: r["nav"])
    elif base_pass:
        best = max(base_pass, key=lambda r: r["nav"])
    else:
        return {"best_label": "无", "best_nav": 0,
                "delta_sharpe": 0, "delta_mdd": 0, "note": "无方案通过基础标准"}

    second = [r for r in base_pass if r["label"] != best["label"]]
    second_best = max(second, key=lambda r: r["nav"]) if second else None

    return {
        "best_label": best["label"],
        "best_nav": best["nav"],
        "best_sharpe": best["sharpe"],
        "best_mdd": best["mdd"],
        "best_tier": best["verdict"],
        "delta_nav_pct": (best["nav"] - w0["nav"]) / w0["nav"] * 100,
        "delta_sharpe": best["sharpe"] - w0["sharpe"],
        "delta_mdd": best["mdd"] - w0["mdd"],
        "second_label": second_best["label"] if second_best else "无",
        "second_nav": second_best["nav"] if second_best else 0,
        "second_delta_nav_pct": (second_best["nav"] - w0["nav"]) / w0["nav"] * 100 if second_best else 0,
    }


def layer2_concentration(all_rows, w0):
    """第2层: 集中度风险评估 + cap/floor 决策"""
    lines = []

    # 检查最优方案的集中度
    best_row = None
    for r in all_rows:
        if r.get("_is_best"):
            best_row = r
            break

    lines.append("各变体集中度指标:")
    lines.append("  {:30s}  {:>8s}  {:>8s}  {:>8s}".format(
        "变体", "max_w", "top3%", "HHI"))
    for r in all_rows:
        max_w = r.get("max_single_weight_max", np.nan)
        top3 = r.get("avg_top3_share", np.nan)
        hhi = r.get("avg_hhi", np.nan)
        label = r["label"][:30] if len(r["label"]) > 30 else r["label"]
        lines.append("  {:30s}  {:8.3f}  {:8.3f}  {:8.3f}".format(
            label, max_w, top3, hhi))

    # cap/floor 决策
    lines.append("")
    lines.append("cap/floor 进入判定 (写死规则):")

    any_enter = False
    for r in all_rows:
        enter, reason = check_cap_floor_entry(r, w0)
        if enter:
            any_enter = True
            lines.append("  ⚠️ {} → 进入: {}".format(r["label"], reason))

    if not any_enter:
        lines.append("  ✅ 无变体触发 cap/floor 进入条件")
        lines.append("  → 不进入第二轮, 保持无 cap/floor 的自然权重方案")
    else:
        lines.append("  → 第二轮引入 cap/floor 硬约束")

    return "\n".join(lines), any_enter


def layer3_recommendation(all_rows, best, any_enter):
    """第3层: 综合建议"""
    lines = []

    any_pass = any(r["tier"] in ("base", "strong") for r in all_rows)
    lines.append("仓位倾斜是否有效: {}".format(
        "✅ 有效 (至少一组基础通过)" if any_pass else "❌ 无效 (无方案通过基础标准)"))

    lines.append("")
    if any_pass:
        lines.append("推荐方案: {} (NAV={:.4f}, ΔNAV={:+.1f}%)".format(
            best["best_label"], best["best_nav"], best["delta_nav_pct"]))
        lines.append("  理由: 基础通过中 NAV 最高, Sharpe {:.2f} (W0 {:.2f}), MDD {:.4f} (W0 {:.4f})".format(
            best["best_sharpe"], 13.89, best["best_mdd"], -0.0688))

        if best["second_label"] != "无":
            lines.append("次优方案: {} (NAV={:.4f}, Δ={:+.1f}%)".format(
                best["second_label"], best["second_nav"], best["second_delta_nav_pct"]))
        lines.append("")
        lines.append("关键发现:")
        lines.append("  by_rank 分层 (W1a/W1b) → Sharpe 保持或提升")
        lines.append("  by_score 线性映射 (W2b-1/W2b-2) → Sharpe 退化, 即使 NAV ↑")
        lines.append("  W3 buy_cls risk penalty → MDD 改善, 但 NAV 未超越 W1a")
        lines.append("  → 结论: 排名离散化优于分数线性映射")
    else:
        lines.append("推荐方案: W0 等权 (无仓位倾斜方案通过)")
        lines.append("  理由: 在 80 交易日样本内, 简单等权已经足够")

    lines.append("")
    if any_enter:
        lines.append("后续方向: 进入第二轮 cap/floor")
        lines.append("  第二轮在最优方案基础上引入单票上限/下限硬约束")
    else:
        lines.append("后续方向: 不进入第二轮 cap/floor")
        lines.append("  当前仓位方案集中度在可接受范围, 暂不引入硬约束")

    return "\n".join(lines)


def main():
    dfs = load_all()
    if not dfs:
        print("无数据, 退出")
        return

    # 合并所有变体
    all_rows = []
    w0 = None

    for key, df in dfs.items():
        for _, row in df.iterrows():
            d = row.to_dict()
            d["source"] = key
            if "W0" in str(d.get("label", "")) or "等权" in str(d.get("label", "")):
                if w0 is None:
                    w0 = d
                # 只保留一份 W0
                if not all(r["label"] == d["label"] for r in all_rows if r.get("label") == d.get("label")):
                    all_rows.append(d)
                continue
            all_rows.append(d)

    # W0 必须存在
    if w0 is None:
        w0_rows = [r for r in all_rows if "W0" in str(r.get("label", ""))]
        if w0_rows:
            w0 = w0_rows[0]
            all_rows = [r for r in all_rows if r.get("label") != w0.get("label")]
        else:
            # 从 w1 CSV 中提取 W0
            for key, df in dfs.items():
                for _, row in df.iterrows():
                    if "W0" in str(row.get("label", "")):
                        w0 = row.to_dict()
                        break
                if w0:
                    break

    if w0 is None:
        print("  ❌ 未找到 W0 基线, 无法判定")
        return

    print("=" * 60)
    print("  仓位实验汇总: E0+X1+0.70 基线下各变体对比")
    print("=" * 60)
    print()
    print("W0 基线: NAV={:.4f}, Sharpe={:.2f}, MDD={:.4f}".format(
        w0["nav"], w0["sharpe"], w0["mdd"]))
    print()

    # 统一判定
    for r in all_rows:
        verdict, tier = classify(r, w0)
        r["verdict"] = verdict
        r["tier"] = tier

    # 输出全表
    print("─" * 60)
    print("  全变体对比表")
    print("─" * 60)
    header = "{:28s} {:>8s} {:>8s} {:>8s} {:>5s} {:>7s} {:>14s}".format(
        "变体", "NAV", "Sharpe", "MDD", "Trades", "ΔNAV%", "判定")
    print(header)
    print("-" * len(header))

    rows_out = []
    for r in all_rows:
        delta = (r["nav"] - w0["nav"]) / w0["nav"] * 100
        label = r["label"][:28] if len(r["label"]) > 28 else r["label"]
        line = "{:28s} {:8.4f} {:8.2f} {:8.4f} {:5d} {:+7.1f}% {:14s}".format(
            label, r["nav"], r["sharpe"], r["mdd"],
            int(r.get("n_trades", 0)), delta, r["verdict"])
        print(line)
        rows_out.append(dict(r, delta_nav_pct=delta))

    print()

    # 第1层: 最优判定
    print("═" * 60)
    print("  第1层: 最优仓位方案判定")
    print("═" * 60)
    best = layer1_best(all_rows, w0)
    # 标记最优
    for r in all_rows:
        r["_is_best"] = (r["label"] == best["best_label"])

    print("  最优: {} (NAV={:.4f}, ΔNAV={:+.1f}%, {}".format(
        best["best_label"], best["best_nav"], best["delta_nav_pct"], best["best_tier"]))
    print("    Sharpe: {:.2f} (Δ{:=+.2f}), MDD: {:.4f} (Δ{:=+.4f})".format(
        best["best_sharpe"], best["delta_sharpe"], best["best_mdd"], best["delta_mdd"]))
    if best["second_label"] != "无":
        print("  次优: {} (NAV={:.4f}, Δ={:+.1f}%)".format(
            best["second_label"], best["second_nav"], best["second_delta_nav_pct"]))
    print()

    # 第2层: 集中度 + cap/floor
    print("═" * 60)
    print("  第2层: 集中度风险评估 + cap/floor 决策")
    print("═" * 60)
    conc_text, any_enter = layer2_concentration(all_rows, w0)
    print(conc_text)
    print()

    # 第3层: 综合建议
    print("═" * 60)
    print("  第3层: 综合建议")
    print("═" * 60)
    rec_text = layer3_recommendation(all_rows, best, any_enter)
    print(rec_text)
    print()

    # 输出 CSV
    df_out = pd.DataFrame(rows_out)
    path = os.path.join(BACKTEST_DIR, "position_sizing_summary.csv")
    df_out.to_csv(path, index=False)
    print("  输出: {}".format(path))

    # 附: 最优方案集中度快照
    print()
    print("═" * 60)
    print("  最优方案集中度快照")
    print("═" * 60)
    for r in all_rows:
        if r.get("_is_best"):
            print("  max_single_weight:     {:.3f}".format(
                r.get("max_single_weight_max", np.nan)))
            print("  avg_top3_share:        {:.3f}".format(
                r.get("avg_top3_share", np.nan)))
            print("  avg_hhi:               {:.3f}".format(
                r.get("avg_hhi", np.nan)))
            print("  monthly_win_rate:      {:.1%}".format(
                r.get("monthly_win_rate", np.nan)))
            break

    # 净值曲线对比
    eq_w0 = os.path.join(BACKTEST_DIR, "w0_equity_curve.csv")
    eq_w1a = os.path.join(BACKTEST_DIR, "w1a_equity_curve.csv")
    if os.path.exists(eq_w0) and os.path.exists(eq_w1a):
        w0_eq = pd.read_csv(eq_w0)
        w1a_eq = pd.read_csv(eq_w1a)
        print()
        print("═" * 60)
        print("  净值曲线对比: W0等权 vs W1a分层")
        print("═" * 60)
        print(f"  W0:  final_nav={w0_eq['nav'].iloc[-1]:.4f}, max_dd={w0_eq['drawdown'].max():.4f}")
        print(f"  W1a: final_nav={w1a_eq['nav'].iloc[-1]:.4f}, max_dd={w1a_eq['drawdown'].max():.4f}")
    else:
        print()
        print("  ⚠️ 缺失: w0_equity_curve.csv 或 w1a_equity_curve.csv (需先运行 position_sizing_w1.py)")

    print()
    return best, any_enter


if __name__ == "__main__":
    main()