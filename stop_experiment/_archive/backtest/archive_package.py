#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
归档包生成器：conclusions_and_boundaries + known_limitations + README

Purpose:
    读取 final_summary.csv，生成正式归档三件套：
    1. conclusions_and_boundaries.csv — 三层结论分区（A/B/C）
    2. known_limitations.csv — 6条已知限制 + 实验证据引用
    3. README.txt — 基线参数说明 + 产物清单 + 结论摘要

Pipeline Position:
    Part A 收尾。
    上游：final_summary.py
    下游：归档目录 archive_e0_x1_v1/

Inputs:
    - output/backtest/final_summary.csv
    - stop_experiment/pipeline/stop_config.py (BASELINE_E0_X1_V1_PARAMS)

Outputs:
    - output/backtest/archive_e0_x1_v1/conclusions_and_boundaries.csv
    - output/backtest/archive_e0_x1_v1/known_limitations.csv
    - output/backtest/archive_e0_x1_v1/README.txt

How to Run:
    python stop_experiment/backtest/archive_package.py

Side Effects:
    - 写 CSV/TXT 到归档目录
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from stop_experiment.pipeline.stop_config import BASELINE_E0_X1_V1_PARAMS, BACKTEST_DIR

ARCHIVE_DIR = os.path.join(BACKTEST_DIR, "archive_e0_x1_v1")


def generate_conclusions_and_boundaries():
    """读取 final_summary.csv，按 advicement 三层重组"""

    src = os.path.join(ARCHIVE_DIR, "final_summary.csv")
    if not os.path.exists(src):
        src = os.path.join(BACKTEST_DIR, "final_summary.csv")

    df = pd.read_csv(src)

    # 映射为三层
    cat_map = {
        "当前版本下成立": "A. 当前版本下成立的结论",
        "正式关闭": "B. 暂时关闭的方向",
        "尚未充分验证": "C. 保留但不优先的方向",
        "保留不优先": "C. 保留但不优先的方向",
    }
    df["layer"] = df["category"].map(cat_map)
    df_out = df[["layer", "id", "conclusion", "evidence", "confidence", "caveat", "action"]]
    df_out = df_out.sort_values(["layer", "id"])

    path = os.path.join(ARCHIVE_DIR, "conclusions_and_boundaries.csv")
    df_out.to_csv(path, index=False)

    n_a = (df_out["layer"] == "A. 当前版本下成立的结论").sum()
    n_b = (df_out["layer"] == "B. 暂时关闭的方向").sum()
    n_c = (df_out["layer"] == "C. 保留但不优先的方向").sum()
    print(f"  conclusions_and_boundaries: A={n_a}, B={n_b}, C={n_c} 条 → {path}")
    return n_a, n_b, n_c


def generate_known_limitations():
    """生成6条已知限制"""

    p = BASELINE_E0_X1_V1_PARAMS
    nav_bl = p["expected_nav"]
    sharpe_bl = p["expected_sharpe"]
    mdd_bl = p["expected_mdd"]

    limitations = [
        {
            "id": "L1",
            "limitation": "样本量限制",
            "description": (
                f"当前回测仅覆盖 80 个交易日 (2026-01-05 → 2026-05-08)，"
                f"共 {p['expected_n_trades']} 笔交易。样本量不足以支撑长期稳定性结论。"
            ),
            "evidence": f"全量回测: NAV={nav_bl:.4f}, Sharpe={sharpe_bl:.2f}, MDD={mdd_bl:.4f}",
            "mitigation": "样本扩展后重新验证；当前实盘跟踪时标注样本边界",
        },
        {
            "id": "L2",
            "limitation": "参数局部最优",
            "description": (
                f"buy_cls_exit_threshold=0.70 是当前样本和当前实现下的局部最优控制值，"
                f"不等于长期最优。前期扫描中 0.65 曾被识别为更优。"
            ),
            "evidence": "exit_scan_2b.csv 和 robustness_thresholds.csv 中的阈值扫描结果",
            "mitigation": "参数冻结但标注为'当前局部最优'；遇重大 regime 变化重新扫描",
        },
        {
            "id": "L3",
            "limitation": "市场环境覆盖不足",
            "description": (
                "80 交易日覆盖区间 (2026-01~05) 为相对正常的震荡上行市场，"
                "未包含熊市、暴跌、流动性危机等极端场景。"
            ),
            "evidence": "4个月度均 NAV>1.0，最差月 NAV=1.20，无重大市场下跌",
            "mitigation": "需历史回测扩展或等待实盘验证极端市场表现",
        },
        {
            "id": "L4",
            "limitation": "仓位分配仅验证 by_rank (W1a), by_score 已关闭",
            "description": (
                "W1a (排名分层 1.25/1.0/0.75) 在当前样本中优于等权 (+3.4%)，"
                "但尚未经过月度/滚动窗/成本敏感性三验证。by_score 线性映射已在 "
                "W2 实验中证实逊于 by_rank 排名分层，正式关闭。"
            ),
            "evidence": "W0 NAV=5.46, W1a NAV=5.65 (+3.4%); W2b-1/b-2 Sharpe 均 < W0",
            "mitigation": "W1a 三验证 (月度/滚动/成本) 通过后方可进入 shadow evaluation 阶段",
        },
        {
            "id": "L5",
            "limitation": "交易成本鲁棒性弱",
            "description": (
                "成本敏感性仅在千分之一量级测试。差异不大不说明策略对更大冲击成本稳健，"
                "尤其在小票流动性不足时滑点可能远超当前假设。"
            ),
            "evidence": "robustness_cost.csv: 成本 sensitivity 差异在 narrow range 内不显著",
            "mitigation": "实盘监控实际滑点；后续在更大成本范围做压力测试",
        },
        {
            "id": "L6",
            "limitation": "样本量限制 obs_day 结论",
            "description": (
                "obs_day=[1] 已被冻结为生产基线，验证了优于 [3] 和 [1,2,3]。"
                "但此结论基于 80 交易日样本，扩展后可能需要重新确认。"
            ),
            "evidence": "前期实验: obs_day=[3] 和 [1,2,3] 均弱于 [1]",
            "mitigation": "扩展样本后确认 obs_day=[1] 仍最优",
        },
    ]

    df = pd.DataFrame(limitations)
    path = os.path.join(ARCHIVE_DIR, "known_limitations.csv")
    df.to_csv(path, index=False)
    print(f"  known_limitations: {len(limitations)} 条 → {path}")
    return len(limitations)


def generate_readme(n_a, n_b, n_c):
    """生成归档 README.txt"""

    p = BASELINE_E0_X1_V1_PARAMS

    lines = [
        "=" * 60,
        f"  实验归档: {p['profile']}",
        "=" * 60,
        "",
        f"冻结日期: {p['frozen_at']}",
        f"描述: {p['description']}",
        "",
        "─" * 40,
        "  基线参数",
        "─" * 40,
        f"  obs_day:           {p['candidate_obs_days']}",
        f"  max_stocks:        {p['max_stocks']} (等权)",
        f"  score_col (排序):  {p['score_col']}",
        f"  exit_mode:         {p['exit_mode']}",
        f"  exit_threshold:    {p['buy_cls_exit_threshold']} (buy_cls > 0.70)",
        f"  stop_loss:         {p['stop_loss']}",
        f"  max_hold_days:     {p['max_hold_days']}",
        f"  buy_cost / sell_cost: {p['buy_cost']} / {p['sell_cost']}",
        "",
        "─" * 40,
        "  基线回测指标",
        "─" * 40,
        f"  NAV:     {p['expected_nav']:.4f}",
        f"  Sharpe:  {p['expected_sharpe']:.2f}",
        f"  MDD:     {p['expected_mdd']:.4f}",
        f"  n_trades: {p['expected_n_trades']}",
        "",
        "─" * 40,
        "  归档产物清单",
        "─" * 40,
        "  [基线回测]",
        "    baseline_e0_x1_v1_summary.csv  — 摘要指标",
        "    baseline_e0_x1_v1_nav.csv      — 逐日NAV",
        "    baseline_e0_x1_v1_trades.csv   — 每笔交易明细",
        "  [Entry复查]",
        "    entry_recheck_entries.csv       — Entry gate 变体对比",
        "    entry_recheck_stratification.csv — 前排质量分层",
        "  [Exit复查]",
        "    buy_reg_quantile_risk.csv       — buy_reg 分位-风险诊断",
        "    exit_recheck_exits.csv          — Exit 变体对比",
        "  [样本外验证]",
        "    rolling_20d.csv                 — 20日滑窗",
        "    rolling_40d.csv                 — 40日滑窗",
        "    monthly_validation.csv          — 月度切分",
        "  [最终汇总]",
        "    final_summary.csv               — 四类结论分区",
        "    conclusions_and_boundaries.csv  — 三层结论(A/B/C)",
        "    known_limitations.csv           — 6条已知限制",
        "",
        "─" * 40,
        "  结论摘要",
        "─" * 40,
        f"  A. 当前版本下成立的结论: {n_a} 条",
        f"     (含 E0 最优、X1 最优、0.70 局部最优、样本外全通过等)",
        f"  B. 暂时关闭的方向: {n_b} 条",
        f"     (buy_reg 硬阈值退出、Entry 复杂聚合、sell_cls 强 gate)",
        f"  C. 保留但不优先: {n_c} 条",
        f"     (X3a 衰减联动、buy_cls 温和 gate、仓位映射等)",
        "",
        "⚠️  注意: 所有结论基于 80 交易日样本",
        "    详见 known_limitations.csv",
        "",
        "=" * 60,
    ]

    path = os.path.join(ARCHIVE_DIR, "README.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  README.txt → {path}")


def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    print("=" * 60)
    print("  归档包生成: archive_e0_x1_v1/")
    print("=" * 60)

    n_a, n_b, n_c = generate_conclusions_and_boundaries()
    generate_known_limitations()
    generate_readme(n_a, n_b, n_c)

    print("\n  归档完成 →", ARCHIVE_DIR)


if __name__ == "__main__":
    main()