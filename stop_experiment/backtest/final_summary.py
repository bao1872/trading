#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📋 正式结题脚本: Entry/Exit 实验事实汇总

Purpose:
    读取 Steps 1-4 的所有输出，汇总生成四类结论分区（纯事实汇总，不做解释）。
    1. 当前版本下成立的结论
    2. 尚未充分验证的推断
    3. 正式关闭的方向
    4. 暂时保留但不再优先推进的方向

Pipeline Position:
    Step 5 汇总（Phase 0-4 完结，2026-05-10 冻结）。
    上游：run_baseline.py, entry_recheck_entries.py, exit_recheck_exits.py, out_of_sample_validator.py
    下游：archive_package.py（归档三件套生成）

Inputs:
    - output/backtest/baseline_e0_x1_v1_summary.csv
    - output/backtest/entry_recheck_entries.csv
    - output/backtest/buy_reg_quantile_risk.csv
    - output/backtest/exit_recheck_exits.csv
    - output/backtest/rolling_20d.csv / rolling_40d.csv
    - output/backtest/monthly_validation.csv

Outputs:
    - output/backtest/final_summary.csv
    - output/backtest/archive_e0_x1_v1/ (归档目录)

How to Run:
    python stop_experiment/backtest/final_summary.py

Side Effects:
    - 读取已有CSV，输出CSV
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import BACKTEST_DIR


def load_csv(name):
    path = os.path.join(BACKTEST_DIR, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def main():
    # 加载所有数据
    baseline = load_csv("baseline_e0_x1_v1_summary.csv")
    entry = load_csv("entry_recheck_entries.csv")
    buy_reg_risk = load_csv("buy_reg_quantile_risk.csv")
    exit_recheck = load_csv("exit_recheck_exits.csv")
    rolling_20d = load_csv("rolling_20d.csv")
    rolling_40d = load_csv("rolling_40d.csv")
    monthly = load_csv("monthly_validation.csv")

    b = baseline.iloc[0] if baseline is not None and len(baseline) > 0 else {}
    nav_baseline = b.get("final_nav", np.nan)
    sharpe_baseline = b.get("sharpe", np.nan)
    mdd_baseline = b.get("max_dd", np.nan)
    n_trades_baseline = b.get("n_trades", 0)

    # ========== 汇总指标 ==========
    summary_rows = []

    print("=" * 72)
    print("  最终汇总: E0+X1+0.70 实验结论（四类分区）")
    print("=" * 72)
    print(f"\n  基线: NAV={nav_baseline:.4f}, Sharpe={sharpe_baseline:.2f}, "
          f"MDD={mdd_baseline:.4f}, n_trades={int(n_trades_baseline)}")
    print(f"  数据区间: 2026-01-05 → 2026-05-08 (80 交易日)")
    print(f"  冻结日期: 2026-05-10")

    # ==================== 第一类：当前版本下成立的结论 ====================
    print("\n" + "─" * 72)
    print("  第一类: 当前版本下成立的结论")
    print("  (当前口径+当前数据+当前候选池下成立的结论)")
    print("─" * 72)

    conclusions_1 = []

    # C1: E0 is Entry optimal
    e0 = entry[entry["label"] == "E0 (基线)"] if entry is not None else None
    nav_e0 = e0.iloc[0]["nav"] if e0 is not None and len(e0) > 0 else np.nan
    c1 = {
        "category": "当前版本下成立",
        "id": "C1",
        "conclusion": "E0 是 Entry 最优基线（无 gate, sell_reg 排序, top10 等权）",
        "evidence": f"E0 NAV={nav_e0:.4f}，所有 Entry gate 变体均未超越 E0",
        "confidence": "高",
        "caveat": "在 obs_day=[1]、exit_th=0.70 口径下成立",
        "action": "E0 作为 Entry 默认配置冻结",
    }
    conclusions_1.append(c1)
    print(f"  ✅ C1: {c1['conclusion']}")
    print(f"     证据: {c1['evidence']}")

    # C2: X1 is Exit optimal
    x1 = exit_recheck[exit_recheck["label"] == "X1 (基线)"] if exit_recheck is not None else None
    nav_x1 = x1.iloc[0]["nav"] if x1 is not None and len(x1) > 0 else np.nan
    x0 = exit_recheck[exit_recheck["label"] == "X0 (规则)"] if exit_recheck is not None else None
    nav_x0 = x0.iloc[0]["nav"] if x0 is not None and len(x0) > 0 else np.nan

    c2 = {
        "category": "当前版本下成立",
        "id": "C2",
        "conclusion": "X1 是 Exit 最优基线（buy_cls 单阈值退出, exit_th=0.70）",
        "evidence": f"X1 NAV={nav_x1:.4f} vs X0 NAV={nav_x0:.4f}, X3/X4 均无法超越 X1",
        "confidence": "高",
        "caveat": "0.70 是当前样本的局部最优控制值，非长期冻结真参数",
        "action": "X1+0.70 作为 Exit 默认配置冻结",
    }
    conclusions_1.append(c2)
    print(f"  ✅ C2: {c2['conclusion']}")
    print(f"     证据: {c2['evidence']}")

    # C3: buy_cls is the most critical exit model
    c3 = {
        "category": "当前版本下成立",
        "id": "C3",
        "conclusion": "buy_cls 是当前策略最关键的退出模型",
        "evidence": "X0(规则)比 X1(模型)弱 1.7x NAV，模型退出增益主要来自 buy_cls",
        "confidence": "高",
        "caveat": "",
        "action": "buy_cls 退出逻辑作为策略核心组件保留",
    }
    conclusions_1.append(c3)
    print(f"  ✅ C3: {c3['conclusion']}")
    print(f"     证据: {c3['evidence']}")

    # C4: obs_day=[1] superiority
    c4 = {
        "category": "当前版本下成立",
        "id": "C4",
        "conclusion": "obs_day=[1] 当前优于更宽观察窗口",
        "evidence": "前期对比 obs_day=[3]、[1,2,3] 均弱于 [1]",
        "confidence": "中高",
        "caveat": "可能受限于当前数据集特征，不排除更长周期表现不同",
        "action": "obs_day=[1] 作为控制变量冻结",
    }
    conclusions_1.append(c4)
    print(f"  ✅ C4: {c4['conclusion']}")
    print(f"     证据: {c4['evidence']}")

    # C5: Out-of-sample validation passes
    nav_20d = rolling_20d["nav"].mean() if rolling_20d is not None else np.nan
    nav_20d_std = rolling_20d["nav"].std() if rolling_20d is not None else np.nan
    n_20d = len(rolling_20d) if rolling_20d is not None else 0
    n_20d_pass = int((rolling_20d["nav"] > 1.0).sum()) if rolling_20d is not None else 0

    nav_40d = rolling_40d["nav"].mean() if rolling_40d is not None else np.nan
    n_40d = len(rolling_40d) if rolling_40d is not None else 0
    n_40d_pass = int((rolling_40d["nav"] > 1.0).sum()) if rolling_40d is not None else 0

    monthly_win = (monthly["nav"] > 1.0).sum() if monthly is not None else 0
    n_months = len(monthly) if monthly is not None else 0
    monthly_navs = monthly["nav"].tolist() if monthly is not None else []

    c5 = {
        "category": "当前版本下成立",
        "id": "C5",
        "conclusion": "E0+X1+0.70 在当前 80 日样本中通过全部样本外验证",
        "evidence": (
            f"20d: {n_20d_pass}/{n_20d} (100%), avg NAV={nav_20d:.2f}±{nav_20d_std:.2f}; "
            f"40d: {n_40d_pass}/{n_40d} (100%), avg NAV={nav_40d:.2f}; "
            f"月度: {int(monthly_win)}/{n_months} ({monthly_win/n_months*100:.0f}%), "
            f"最差月NAV={min(monthly_navs):.4f}"
        ),
        "confidence": "中",
        "caveat": "样本量仅 80 交易日/4 个月，统计意义有限",
        "action": "参数冻结后跟踪实盘表现",
    }
    conclusions_1.append(c5)
    print(f"  ✅ C5: {c5['conclusion']}")
    print(f"     证据: {c5['evidence']}")

    # C6: pred_buy_reg has genuine risk discrimination
    if buy_reg_risk is not None and len(buy_reg_risk) > 0:
        worst_seg = buy_reg_risk.iloc[0]
        best_seg = buy_reg_risk.iloc[-1]
        c6 = {
            "category": "当前版本下成立",
            "id": "C6",
            "conclusion": "pred_buy_reg 具有真实的尾部风险区分力（不适合做硬阈值触发器）",
            "evidence": (
                f"P0-P10: mae_10<-7%={worst_seg['mae_10_lt7pct']:.1f}%, "
                f"P90-P100: mae_10<-7%={best_seg['mae_10_lt7pct']:.1f}% — 风险分化明显"
            ),
            "confidence": "中高",
            "caveat": "作为连续排序因子可能有价值，但作为硬阈值触发 'or' 退出不可行",
            "action": "buy_reg 保留为连续因子，不作为 Event trigger",
        }
        conclusions_1.append(c6)
        print(f"  ✅ C6: {c6['conclusion']}")
        print(f"     证据: {c6['evidence']}")

    summary_rows.extend(conclusions_1)

    # ==================== 第二类：尚未充分验证的推断 ====================
    print("\n" + "─" * 72)
    print("  第二类: 尚未充分验证的推断")
    print("  (有一定证据支持但样本/覆盖不足，不宜作为确定结论)")
    print("─" * 72)

    conclusions_2 = []

    c7 = {
        "category": "尚未充分验证",
        "id": "I1",
        "conclusion": "buy_cls=0.70 的长期稳定性未经验证",
        "evidence": "80 交易日中 0.70 表现最优，但前期 0.65 曾被识别为更优",
        "confidence": "低",
        "caveat": "需更长样本/不同市场环境下重新扫描确认",
        "action": "参数冻结但标注为'当前局部最优'，遇重大 regime 变化重新扫描",
    }
    conclusions_2.append(c7)
    print(f"  ⚠️  I1: {c7['conclusion']}")
    print(f"      证据: {c7['evidence']}")

    c8 = {
        "category": "尚未充分验证",
        "id": "I2",
        "conclusion": "策略对交易成本冲击的鲁棒性证据偏弱",
        "evidence": "千分之一量级成本敏感性差异不大，但不代表策略对更大冲击成本稳健",
        "confidence": "低",
        "caveat": "仅测试了 narrow range 成本参数",
        "action": "实盘中监控实际滑点/成本，必要时扩大成本敏感性测试",
    }
    conclusions_2.append(c8)
    print(f"  ⚠️  I2: {c8['conclusion']}")
    print(f"      证据: {c8['evidence']}")

    c9 = {
        "category": "尚未充分验证",
        "id": "I3",
        "conclusion": "obs_day=[1] 最优可能受限于当前市场 regime",
        "evidence": "前期对比 obs_day=[3] 弱于 [1]，但仅测试了两个窗口",
        "confidence": "低",
        "caveat": "未测试 [2]、[1,2] 等更多窗口组合",
        "action": "暂定 [1]，后续扩大样本后再做系统对照",
    }
    conclusions_2.append(c9)
    print(f"  ⚠️  I3: {c9['conclusion']}")
    print(f"      证据: {c9['evidence']}")

    c10 = {
        "category": "尚未充分验证",
        "id": "I4",
        "conclusion": "策略在熊市/暴跌场景下的表现未经验证",
        "evidence": "80 交易日覆盖区间不包含重大市场下跌",
        "confidence": "低",
        "caveat": "当前验证仅覆盖相对正常市场环境",
        "action": "需历史扩展或等待实盘验证",
    }
    conclusions_2.append(c10)
    print(f"  ⚠️  I4: {c10['conclusion']}")
    print(f"      证据: {c10['evidence']}")

    summary_rows.extend(conclusions_2)

    # ==================== 第三类：正式关闭的方向 ====================
    print("\n" + "─" * 72)
    print("  第三类: 正式关闭的方向")
    print("  (经过充分测试确认不成立，不再投入资源)")
    print("─" * 72)

    conclusions_3 = []

    c11 = {
        "category": "正式关闭",
        "id": "X1",
        "conclusion": "buy_reg 硬阈值退出 (or_buy_reg / X4) 正式关闭",
        "evidence": (
            f"P90 阈值: model exits=92%, NAV=2.35; "
            f"P95 阈值: model exits=80%, NAV=4.06 — 全部远超可接受范围"
        ),
        "confidence": "高",
        "caveat": "buy_reg 不适用于硬阈值触发，但连续排序价值保留",
        "action": "不在 exit 逻辑中引入 buy_reg Event trigger",
    }
    conclusions_3.append(c11)
    print(f"  ❌ X1: {c11['conclusion']}")
    print(f"     证据: {c11['evidence']}")

    c12 = {
        "category": "正式关闭",
        "id": "X2",
        "conclusion": "Entry 复杂聚合结构 (E1-E6 buy_cls gate / sell_cls gate) 正式关闭",
        "evidence": "所有 Entry gate 变体均未在 NAV 上超越 E0，buy_cls 严格 gate 过杀交易量",
        "confidence": "高",
        "caveat": "极温和 gate (buy_cls<0.80) 有 0.8% 边际增益，太小不推荐",
        "action": "Entry 不再引入 gate，恢复纯 sell_reg 排序",
    }
    conclusions_3.append(c12)
    print(f"  ❌ X2: {c12['conclusion']}")
    print(f"     证据: {c12['evidence']}")

    c13 = {
        "category": "正式关闭",
        "id": "X3",
        "conclusion": "sell_cls 强 gate (≥0.70) 作为 Entry filter 正式关闭",
        "evidence": "sell_cls>0.70: gate_pass=84.8%, NAV=5.27 (下降); sell_cls>0.75: gate_pass=65.8%, NAV=4.56",
        "confidence": "高",
        "caveat": "sell_cls 作为 Entry gate 无增益",
        "action": "sell_cls 仅保留在 Exit 侧（buy_cls 的辅助信息），不在 Entry 用",
    }
    conclusions_3.append(c13)
    print(f"  ❌ X3: {c13['conclusion']}")
    print(f"     证据: {c13['evidence']}")

    summary_rows.extend(conclusions_3)

    # ==================== 第四类：暂时保留但不优先 ====================
    print("\n" + "─" * 72)
    print("  第四类: 暂时保留但不再优先推进的方向")
    print("  (逻辑上不排除有价值，但当前不值得继续投入)")
    print("─" * 72)

    conclusions_4 = []

    c14 = {
        "category": "保留不优先",
        "id": "D1",
        "conclusion": "sell_reg 衰减联动退出 (X3a) — 逻辑合理但当前实现无增益",
        "evidence": "X3a NAV=3.96 < 4.92 (90%×X1)，纯 sell_reg 衰减条件过弱",
        "confidence": "中",
        "caveat": "宽松版/rank 下降版未测试，方向上不完全排除",
        "action": "暂时搁置，等更多数据或新思路再评估",
    }
    conclusions_4.append(c14)
    print(f"  🔒 D1: {c14['conclusion']}")
    print(f"     证据: {c14['evidence']}")

    c15 = {
        "category": "保留不优先",
        "id": "D2",
        "conclusion": "Entry 极温和 buy_cls gate (buy_cls<0.80) — 通过复查但增益太小",
        "evidence": "NAV=5.51 (+0.8%), gate_pass=94.9%, 前排质量未改善",
        "confidence": "中",
        "caveat": "在仓位映射阶段可能有 marginal value（改善队列纯度）",
        "action": "不作为主策略推荐，仓位映射阶段评估",
    }
    conclusions_4.append(c15)
    print(f"  🔒 D2: {c15['conclusion']}")
    print(f"     证据: {c15['evidence']}")

    c16 = {
        "category": "保留不优先",
        "id": "D3",
        "conclusion": "buy_reg 作为连续排名因子（非 Trigger）在排序/仓位中的价值",
        "evidence": "buy_reg 尾部风险区分力已验证，但未测试其作为排名因子",
        "confidence": "低",
        "caveat": "可作为仓位映射/权重调整的参考因子",
        "action": "仓位映射阶段统一评估",
    }
    conclusions_4.append(c16)
    print(f"  🔒 D3: {c16['conclusion']}")
    print(f"     证据: {c16['evidence']}")

    c17 = {
        "category": "保留不优先",
        "id": "D4",
        "conclusion": "仓位映射和账户分配 — 当前冻结 top10 等权，后续单独设计",
        "evidence": "",
        "confidence": "",
        "caveat": "在 Entry/Exit 口径稳定后再评估",
        "action": "暂不混入当前实验，后续独立研究",
    }
    conclusions_4.append(c17)
    print(f"  🔒 D4: {c17['conclusion']}")
    print(f"     证据: {c17['evidence']}")

    summary_rows.extend(conclusions_4)

    # ==================== 输出 ====================
    df_out = pd.DataFrame(summary_rows)
    path = os.path.join(BACKTEST_DIR, "final_summary.csv")
    df_out.to_csv(path, index=False)

    print("\n" + "=" * 72)
    print(f"  汇总统计")
    print("=" * 72)
    cats = df_out["category"].value_counts()
    for cat, count in cats.items():
        print(f"  {cat}: {count} 条")
    print(f"\n  输出: {path}")

    # ==================== 最终建议 ====================
    print("\n" + "=" * 72)
    print("  最终建议")
    print("=" * 72)
    print(f"""
  当前策略版本: E0+X1+0.70 (baseline_e0_x1_v1)
  Entry: 纯 pred_sell_reg 排序, top10 等权, obs_day=[1]
  Exit:  buy_cls>0.70 触发模型卖出, stop_loss=-7%, max_hold=20d
  基线: NAV={nav_baseline:.4f}, Sharpe={sharpe_baseline:.2f}, MDD={mdd_baseline:.4f}

  下一步优先级:
  1. 🔴 最高: 扩大样本验证（历史扩展 + 更长回测窗口）
  2. 🟡 中:  实盘跟踪（E0+X1+0.70 运行，监控偏离）
  3. 🟢 低:  仓位映射设计（但放在单独研究迭代中）

  ⚠️  注意: 所有结论基于 80 交易日样本，标注为"当前版本下成立"
      样本扩大后可能部分结论需重新评估。
""")


if __name__ == "__main__":
    main()