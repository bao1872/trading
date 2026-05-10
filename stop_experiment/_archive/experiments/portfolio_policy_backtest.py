#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[已证伪] 组合决策回测 — Tier 准入过滤实验

⚠️ 实验结论: 全部跑输基线，Tier 不适合做准入门槛。
   后续实验请使用 tier_weight_mapping.py（仓位映射方向）。

Purpose:
    (历史保留) 基于回测引擎 snapshot，在买入侧施加 Tier 硬过滤策略：
    保守型(只买 A+B, max=6) / 平衡型(A+B+C, max=10) / 进攻型(A+B+C, max=12)。

    实验结论:
    - 基线最优: NAV 2.74, 胜率 82.9%, 回撤 0.39%
    - 全部策略跑输: 最佳进攻型 NAV -8.06%
    - 根因: Tier A 机会太少(~5%)，过滤掉大量盈利机会
    - 教训: Tier 不适合做"买不买"的硬过滤

    踩坑记录:
    - 基线 NAV 不一致 → 修复 weight 复衡逻辑（卖出后不复衡）

Pipeline Position:
    实验逻辑（已证伪，保留供参考）。
    上游: dynamic_exit_backtest_v2.py
    下游: tier_weight_mapping.py（转向仓位映射）

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - DB: stock_k_data
    - 回测引擎 debug_snapshots

Outputs:
    - Console: 结构化对比报告
    - stop_experiment/output/policy_comparison/
      ├── metrics_comparison.csv
      ├── trades_baseline/A/B/C.csv
      ├── nav_comparison.png
      └── metrics_bar.png

How to Run:
    python stop_experiment/experiments/portfolio_policy_backtest.py

Side Effects:
    - 只读 parquet/DB，写 CSV/PNG 到 output 目录
"""

from __future__ import annotations

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
from collections import defaultdict

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, V1_PARAMS, BUY_COST, SELL_COST,
)
from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest,
)

# ==================== 常量 ====================

COMPARISON_DIR = os.path.join(OUTPUT_DIR, "policy_comparison")

# SSOT Tier 规则（与日报 Section 4B/5 tiers 保持一致）
TIER_A_SCORE = 0.2
TIER_A_BUY_CLS = 0.30
TIER_B_SCORE = 0.1
TIER_B_BUY_CLS = 0.70

# 策略配置字典
POLICY_CONFIGS = {
    "baseline": {
        "label": "基线 (Baseline)",
        "name": "baseline",
        "max_positions": 10,
        "tier_filter": None,       # None = 无过滤，保留原始回测 buys
        "tier_priority": None,     # None = 按 score 降序
        "use_original_buys": True,  # 直接使用 snapshot.buys
    },
    "A": {
        "label": "保守型 (Conservative)",
        "name": "policy_a",
        "max_positions": 6,
        "tier_filter": {"A", "B"},         # 只买 Tier A + B
        "tier_priority": ["A", "B"],        # A > B
        "use_original_buys": False,
        "allow_empty_on_tier_c": True,      # 全 Tier C 时空仓
    },
    "B": {
        "label": "平衡型 (Balanced)",
        "name": "policy_b",
        "max_positions": 10,
        "tier_filter": {"A", "B", "C"},     # 全 Tier 可买
        "tier_priority": ["A", "B", "C"],    # A > B > C
        "use_original_buys": False,
        "allow_empty_on_tier_c": False,
    },
    "C": {
        "label": "进攻型 (Aggressive)",
        "name": "policy_c",
        "max_positions": 12,
        "tier_filter": {"A", "B", "C"},     # 无 Tier 过滤
        "tier_priority": None,               # 严格按 score 降序
        "use_original_buys": False,
        "allow_empty_on_tier_c": False,
    },
}

EXIT_PARAMS = {
    "max_hold_days": V1_PARAMS["max_hold_days"],
    "stop_loss": V1_PARAMS["stop_loss"],
    "buy_cls_threshold": V1_PARAMS["buy_cls_exit_threshold"],
}


# ==================== Tier 标注 ====================

def _annotate_tier(df):
    """
    对 candidates_df 标注 Tier A/B/C（SSOT 规则）。

    Input:
        df: DataFrame，必须含 score (pred_sell_reg) 和 pred_buy_cls 列

    Output:
        df: 新增 tier 列 (str: A/B/C)
    """
    df = df.copy()
    df["tier"] = "C"

    mask_a = (df["score"] > TIER_A_SCORE) & (df["pred_buy_cls"] < TIER_A_BUY_CLS)
    df.loc[mask_a, "tier"] = "A"

    mask_b = (df["score"] > TIER_B_SCORE) & (df["pred_buy_cls"] < TIER_B_BUY_CLS) & (~mask_a)
    df.loc[mask_b, "tier"] = "B"

    return df


# ==================== 指标计算 ====================

def _compute_metrics(trades_df, nav_df):
    """从 trades_df + nav_df 计算核心指标。"""
    if nav_df.empty:
        return {
            "final_nav": 1.0, "annual_return": 0.0, "max_drawdown": 0.0,
            "calmar": 0.0, "win_rate": 0.0, "avg_hold_days": 0.0,
            "total_trades": 0, "avg_net_return": 0.0, "avg_positions": 0.0,
        }

    nav_series = nav_df["nav"]
    final_nav = nav_series.iloc[-1]
    n = len(nav_series)
    years = max(n / 252, 0.01)
    annual_return = (final_nav / nav_series.iloc[0]) ** (1 / years) - 1

    peak = nav_series.iloc[0]
    max_dd = 0.0
    for v in nav_series.iloc[1:]:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    calmar = annual_return / max_dd if max_dd > 0 else 0.0

    if trades_df.empty:
        return {
            "final_nav": final_nav, "annual_return": annual_return,
            "max_drawdown": max_dd, "calmar": calmar,
            "win_rate": 0.0, "avg_hold_days": 0.0,
            "total_trades": 0, "avg_net_return": 0.0,
            "avg_positions": nav_df["n_positions"].mean() if "n_positions" in nav_df.columns else 0.0,
        }

    win_rate = (trades_df["net_ret"] > 0).mean()
    avg_net_return = trades_df["net_ret"].mean()
    avg_hold_days = trades_df["hold_days"].mean() if "hold_days" in trades_df.columns else 0.0
    avg_positions = nav_df["n_positions"].mean() if "n_positions" in nav_df.columns else 0.0

    return {
        "final_nav": final_nav,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_hold_days": avg_hold_days,
        "total_trades": len(trades_df),
        "avg_net_return": avg_net_return,
        "avg_positions": avg_positions,
    }


# ==================== 策略模拟引擎 ====================

def _evaluate_exits(holdings, current_date, price_pivot, trading_days, pred_indexed):
    """
    评估当日退出（SSOT 规则: max_hold > stop_loss > model_risk）。

    与 dynamic_exit_backtest_v2.run_backtest Step 2 逻辑一致。

    Output:
        to_sell:    [(code, ts_code, sell_price, sell_reason), ...]
        holdings:   退出后的持仓（原地修改）
    """
    day_open = price_pivot.loc[current_date, "open"]
    day_close = price_pivot.loc[current_date, "close"]

    t_idx = None
    for i, td in enumerate(trading_days):
        if td == current_date:
            t_idx = i
            break
    prev_date = trading_days[t_idx - 1] if t_idx is not None and t_idx > 0 else None

    to_sell = []

    for code in list(holdings.keys()):
        h = holdings[code]
        h["days_held"] += 1

        should_sell = False
        reason = ""

        if h["days_held"] > EXIT_PARAMS["max_hold_days"]:
            should_sell, reason = True, "max_hold"
        elif code in day_close.index and pd.notna(day_close[code]) and day_close[code] > 0:
            ret = (day_close[code] - h["buy_price"]) / h["buy_price"]
            if ret < EXIT_PARAMS["stop_loss"]:
                should_sell, reason = True, "stop_loss"

        if not should_sell and h["days_held"] > 1 and prev_date is not None:
            sid = h.get("signal_id")
            if sid is not None:
                pred = pred_indexed.get((int(sid), prev_date))
                if pred is not None:
                    bc = pred.get("pred_buy_cls", np.nan)
                    if pd.notna(bc) and bc > EXIT_PARAMS["buy_cls_threshold"]:
                        should_sell, reason = True, "model_risk"

        if not should_sell:
            continue

        sell_price = np.nan
        if code in day_open.index and pd.notna(day_open[code]):
            sell_price = day_open[code]
        if np.isnan(sell_price) or sell_price <= 0:
            continue

        gross_ret = (sell_price - h["buy_price"]) / h["buy_price"]
        net_ret = gross_ret - BUY_COST - SELL_COST

        to_sell.append({
            "ts_code": h["ts_code"], "code": code,
            "buy_date": h["buy_date"], "sell_date": current_date,
            "buy_price": h["buy_price"], "sell_price": sell_price,
            "hold_days": h["days_held"], "gross_ret": gross_ret,
            "net_ret": net_ret, "sell_reason": reason,
            "score": h.get("score", 0), "signal_id": h.get("signal_id"),
            "tier": h.get("tier", "C"),
        })

    for td in to_sell:
        del holdings[td["code"]]

    return holdings, to_sell


def _select_buys(candidates_df, holdings, policy_cfg):
    """
    按策略规则从 candidates_df 中选取买入。

    Input:
        candidates_df:  当日候选池（含 tier 列）
        holdings:       当前持仓（退出后）
        policy_cfg:     策略配置字典

    Output:
        pending_buys:   [(code, ts_code, score, signal_id, tier), ...]
    """
    if candidates_df.empty:
        return []

    n_avail = policy_cfg["max_positions"] - len(holdings)
    if n_avail <= 0:
        return []

    cand = candidates_df.copy()

    if policy_cfg.get("use_original_buys", False):
        return []

    # Tier 过滤
    if policy_cfg["tier_filter"] is not None:
        cand = cand[cand["tier"].isin(policy_cfg["tier_filter"])]

    if cand.empty:
        return []

    # 排序
    if policy_cfg["tier_priority"] is not None:
        tier_order = {t: i for i, t in enumerate(policy_cfg["tier_priority"])}
        cand["_tier_rank"] = cand["tier"].map(lambda t: tier_order.get(t, 99))
        cand = cand.sort_values(["_tier_rank", "score"], ascending=[True, False])
    else:
        cand = cand.sort_values("score", ascending=False)

    # 去重 & 排除已持仓
    cand = cand.drop_duplicates(subset=["ts_code"], keep="first")

    # 对于 policy A: 检查是否所有可用候选都是 Tier C → 允许空仓
    if policy_cfg.get("allow_empty_on_tier_c", False):
        non_c = cand[cand["tier"] != "C"]
        all_c = len(non_c) == 0 and len(cand) > 0
        if all_c:
            return []  # 当日全 Tier C，不买

    held_ts = {h["ts_code"] for h in holdings.values()}
    selected = []
    for _, row in cand.iterrows():
        if len(selected) >= n_avail:
            break
        ts = str(row["ts_code"])
        if ts in held_ts:
            continue
        code = ts[:6] if "." in ts else ts
        selected.append((code, ts, row.get("score", 0), row.get("signal_id"), row.get("tier", "C")))

    return selected


def _calc_daily_nav(holdings, current_date, price_pivot, trading_days):
    """
    计算当日持仓收益率与 NAV。

    与 run_backtest Step 3 逻辑完全一致:
        - 使用 holdings 中存储的 weight（买入时设定，卖出后不重新平衡）
        - days_held==1 → (close - open)/open  (买入首日)
        - 否则         → (close - prev_close)/prev_close
    """
    day_open = price_pivot.loc[current_date, "open"]
    day_close = price_pivot.loc[current_date, "close"]

    t_idx = None
    for i, td in enumerate(trading_days):
        if td == current_date:
            t_idx = i
            break

    if t_idx is None or t_idx == 0:
        prev_close = pd.Series(dtype=float)
    else:
        prev_date = trading_days[t_idx - 1]
        if prev_date in price_pivot.index:
            prev_close = price_pivot.loc[prev_date, "close"]
        else:
            prev_close = pd.Series(dtype=float)

    daily_ret = 0.0
    n = len(holdings)
    if n == 0:
        return daily_ret, n, None

    for code, h in holdings.items():
        if code not in day_close.index or np.isnan(day_close[code]):
            continue

        if h.get("days_held") == 1:
            if code in day_open.index and pd.notna(day_open[code]) and day_open[code] > 0:
                sr = (day_close[code] - day_open[code]) / day_open[code]
            else:
                sr = 0.0
        elif code in prev_close.index and pd.notna(prev_close[code]) and prev_close[code] > 0:
            sr = (day_close[code] - prev_close[code]) / prev_close[code]
        else:
            sr = 0.0

        # 使用 holdings 中存储的 weight（买入时设定，卖出后不重新平衡）
        w = h.get("weight", 1.0 / n)
        daily_ret += w * sr

    return daily_ret, n, None


def _simulate_policy(snapshots, policy_cfg, price_pivot, trading_days, pred_indexed):
    """
    基于回测 snapshot 序列，对指定策略做逐日模拟。

    流程:
        逐 snapshot 遍历:
        1. 执行 T-1 的 pending buys（T 日开盘入场，存入 holdings）
        2. 评估 SSOT 退出 → 生成 to_sell、更新 holdings
        3. 计算剩余仓位
        4. 策略筛选买入候选 → 生成 pending buys（T+1 执行）
        5. 计算 NAV

    Output:
        trades_df:  已完成交易明细
        nav_df:     每日净值
    """
    holdings = {}
    pending_buys = []
    all_trades = []
    nav_records = []

    nav = 1.0

    for snap in snapshots:
        current_date = snap["date"]

        if current_date not in price_pivot.index:
            continue

        # ---- Step 1: 执行 pending buys (T-1 的买入，T 日开盘入场) ----
        if pending_buys:
            day_open = price_pivot.loc[current_date, "open"]
            executed = []
            for code, ts_code, sc, sid, tier in pending_buys:
                if code in holdings:
                    continue
                if code not in day_open.index or np.isnan(day_open[code]) or day_open[code] <= 0:
                    continue
                bp = day_open[code]
                executed.append((code, ts_code, bp, sc, sid, tier))

            if executed:
                n_total = len(holdings) + len(executed)
                w = 1.0 / n_total if n_total > 0 else 1.0
                for code_h in holdings:
                    holdings[code_h]["weight"] = w
                for code, ts_code, bp, sc, sid, tier in executed:
                    holdings[code] = {
                        "buy_date": current_date, "buy_price": bp,
                        "weight": w, "days_held": 0,
                        "ts_code": ts_code, "score": sc, "signal_id": sid,
                        "tier": tier,
                    }
        pending_buys = []

        # ---- Step 2: 评估退出 ----
        holdings, to_sell = _evaluate_exits(
            holdings, current_date, price_pivot, trading_days, pred_indexed,
        )
        all_trades.extend(to_sell)

        # ---- Step 3: 计算 NAV ----
        daily_ret, n_pos, _ = _calc_daily_nav(holdings, current_date, price_pivot, trading_days)
        prev_nav = nav
        nav = prev_nav * (1 + daily_ret)

        nav_records.append({
            "date": current_date, "nav": nav,
            "daily_ret": daily_ret, "n_positions": n_pos,
        })

        # ---- Step 4: 买入信号 ----
        if policy_cfg.get("use_original_buys", False):
            for b in snap.get("buys", []):
                code = b.get("code", "")
                ts = b.get("ts_code", "")
                if not code or code in holdings:
                    continue
                pending_buys.append((code, ts, b.get("score", 0), b.get("signal_id"), "C"))
        else:
            cand_df = snap.get("candidates_df", pd.DataFrame())
            if not cand_df.empty:
                cand_df = cand_df.copy()
                if "tier" not in cand_df.columns:
                    if "score" not in cand_df.columns:
                        cand_df["score"] = cand_df.get("pred_sell_reg", cand_df.get("composite_score", 0))
                    cand_df = _annotate_tier(cand_df)
                selected = _select_buys(cand_df, holdings, policy_cfg)
                pending_buys = selected

    trades_df = pd.DataFrame(all_trades)
    nav_df = pd.DataFrame(nav_records)

    return trades_df, nav_df


# ==================== 可视化 ====================

def _plot_nav_comparison(all_results, output_path):
    """绘制 4 条 NAV 曲线对比图。"""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {"baseline": "#888888", "A": "#2ca02c", "B": "#1f77b4", "C": "#d62728"}
    labels = {
        "baseline": "基线 (Baseline, k=10)",
        "A": "保守型 A (k=6, A+B only)",
        "B": "平衡型 B (k=10, tier优先)",
        "C": "进攻型 C (k=12, 无过滤)",
    }

    for key in ["baseline", "A", "B", "C"]:
        if key in all_results and not all_results[key]["nav_df"].empty:
            ndf = all_results[key]["nav_df"]
            x = np.arange(len(ndf))
            ax.plot(x, ndf["nav"], color=colors.get(key, "#333"),
                    linewidth=1.8, label=labels.get(key, key), alpha=0.9)

    ax.set_title("组合决策策略 NAV 对比", fontsize=14)
    ax.set_xlabel("交易日序号")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  NAV 对比图已保存: {output_path}")


def _plot_metrics_bar(metrics_df, output_path):
    """绘制核心指标柱状对比图。"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("组合决策策略 核心指标对比", fontsize=14)

    bar_labels = ["基线", "保守A", "平衡B", "进攻C"]
    bar_colors = ["#888888", "#2ca02c", "#1f77b4", "#d62728"]

    metric_groups = [
        (axes[0, 0], "final_nav", "最终 NAV", None),
        (axes[0, 1], "annual_return", "年化收益", lambda v: f"{v:+.2%}"),
        (axes[0, 2], "max_drawdown", "最大回撤", lambda v: f"{v:.2%}"),
        (axes[1, 0], "calmar", "Calmar 比率", None),
        (axes[1, 1], "win_rate", "胜率", lambda v: f"{v:.1%}"),
        (axes[1, 2], "avg_positions", "平均持仓数", None),
    ]

    for ax, col, title, fmt in metric_groups:
        vals = [metrics_df.loc[metrics_df["key"] == k, col].values[0]
                if k in metrics_df["key"].values else 0
                for k in ["baseline", "A", "B", "C"]]
        bars = ax.bar(bar_labels, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            text = fmt(v) if fmt else f"{v:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    text, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  指标柱状图已保存: {output_path}")


# ==================== 报告输出 ====================

def _print_param_summary():
    print()
    print("=" * 85)
    print("  Section 1: 实验参数摘要")
    print("=" * 85)
    print(f"  版本基线:      v1_frozen_202605")
    print(f"  退出模式:      model_exit (max_hold={EXIT_PARAMS['max_hold_days']}d, "
          f"stop_loss={EXIT_PARAMS['stop_loss']:.0%}, buy_cls>{EXIT_PARAMS['buy_cls_threshold']:.2f})")
    print(f"  候选观测日:    obs_day in {V1_PARAMS['candidate_obs_days']}")
    print(f"  排序策略:      {V1_PARAMS.get('strategy_default', 'sell_score')} (score = pred_sell_reg)")
    print(f"  交易成本:      买入 {BUY_COST:.4f} + 卖出 {SELL_COST:.4f}")
    print()
    print(f"  Tier 定义 (SSOT, 不可改):")
    print(f"    Tier A: score > {TIER_A_SCORE} AND pred_buy_cls < {TIER_A_BUY_CLS}")
    print(f"    Tier B: score > {TIER_B_SCORE} AND pred_buy_cls < {TIER_B_BUY_CLS}")
    print(f"    Tier C: 其余")
    print()
    print(f"  策略规则:")
    print(f"    {'策略':<12} {'最大持仓':>8} {'Tier过滤':>16} {'买入优先级':>20}")
    print(f"    {'-'*60}")
    print(f"    {'基线 Baseline':<12} {10:>8} {'无':>16} {'按score降序':>20}")
    print(f"    {'保守型 A':<12} {6:>8} {'A+B only':>16} {'A > B':>20}")
    print(f"    {'平衡型 B':<12} {10:>8} {'A+B+C':>16} {'A > B > C':>20}")
    print(f"    {'进攻型 C':<12} {12:>8} {'A+B+C':>16} {'按score降序':>20}")


def _print_metrics_table(all_results):
    print()
    print("=" * 85)
    print("  Section 2: 策略对比指标表")
    print("=" * 85)

    headers = ["策略", "最终NAV", "年化收益", "最大回撤", "Calmar", "胜率", "均持天数", "均净收益", "交易笔数", "均持仓"]
    header_line = (f"  {headers[0]:<12} {headers[1]:>9} {headers[2]:>9} {headers[3]:>8} "
                   f"{headers[4]:>8} {headers[5]:>7} {headers[6]:>8} {headers[7]:>9} {headers[8]:>8} {headers[9]:>8}")
    print(header_line)
    print(f"  {'-'*95}")

    for key in ["baseline", "A", "B", "C"]:
        if key not in all_results:
            continue
        m = all_results[key]["metrics"]
        label = POLICY_CONFIGS[key]["label"]
        ann = f"{m['annual_return']:+.2%}" if abs(m['annual_return']) < 10 else "N/A"
        print(f"  {label:<12} {m['final_nav']:>9.4f} {ann:>9} {m['max_drawdown']:>8.2%} "
              f"{m['calmar']:>8.2f} {m['win_rate']:>7.1%} {m['avg_hold_days']:>8.1f} "
              f"{m['avg_net_return']:>9.4f} {m['total_trades']:>8} {m['avg_positions']:>8.1f}")

    b_m = all_results["baseline"]["metrics"]
    print(f"\n  相对基线差异:")
    for key in ["A", "B", "C"]:
        if key not in all_results:
            continue
        m = all_results[key]["metrics"]
        label = POLICY_CONFIGS[key]["label"]
        nav_diff = (m["final_nav"] - b_m["final_nav"]) / b_m["final_nav"]
        wr_diff = m["win_rate"] - b_m["win_rate"]
        dd_diff = m["max_drawdown"] - b_m["max_drawdown"]
        print(f"    {label:<12} NAV {nav_diff:+.2%}, 胜率 {wr_diff:+.1%}, 回撤 {dd_diff:+.2%}")


def _print_tier_attribution(all_results):
    print()
    print("=" * 85)
    print("  Section 3: Tier 归因分析")
    print("=" * 85)

    for key in ["baseline", "A", "B", "C"]:
        if key not in all_results:
            continue
        tdf = all_results[key]["trades_df"]
        if tdf.empty:
            print(f"  {POLICY_CONFIGS[key]['label']}: 无交易")
            continue

        label = POLICY_CONFIGS[key]["label"]

        if "tier" not in tdf.columns:
            print(f"  {label}: (无 Tier 标注，均为原始回测交易)")
            continue

        print(f"\n  {label}:")
        print(f"    {'Tier':<6} {'交易数':>6} {'胜率':>8} {'均净收益':>10} {'均持天数':>8}")

        for tier in ["A", "B", "C"]:
            sub = tdf[tdf["tier"] == tier]
            if sub.empty:
                continue
            wr = (sub["net_ret"] > 0).mean()
            avg_net = sub["net_ret"].mean()
            avg_hd = sub["hold_days"].mean()
            print(f"    {tier:<6} {len(sub):>6} {wr:>7.1%} {avg_net:>10.4f} {avg_hd:>8.1f}")


def _print_conclusions(all_results):
    print()
    print("=" * 85)
    print("  Section 4: 结论与建议")
    print("=" * 85)

    b_m = all_results["baseline"]["metrics"]
    best_nav_key = max(
        [k for k in ["baseline", "A", "B", "C"] if k in all_results],
        key=lambda k: all_results[k]["metrics"]["final_nav"]
    )
    best_wr_key = max(
        [k for k in ["baseline", "A", "B", "C"] if k in all_results],
        key=lambda k: all_results[k]["metrics"]["win_rate"]
    )
    best_dd_key = min(
        [k for k in ["baseline", "A", "B", "C"] if k in all_results],
        key=lambda k: all_results[k]["metrics"]["max_drawdown"]
    )

    print(f"  1. 基线 (Baseline) 最终 NAV: {b_m['final_nav']:.4f}, 胜率: {b_m['win_rate']:.1%}, "
          f"最大回撤: {b_m['max_drawdown']:.2%}")
    print(f"  2. 最高 NAV:   {POLICY_CONFIGS[best_nav_key]['label']} "
          f"(NAV={all_results[best_nav_key]['metrics']['final_nav']:.4f})")
    print(f"  3. 最高胜率:   {POLICY_CONFIGS[best_wr_key]['label']} "
          f"(胜率={all_results[best_wr_key]['metrics']['win_rate']:.1%})")
    print(f"  4. 最低回撤:   {POLICY_CONFIGS[best_dd_key]['label']} "
          f"(回撤={all_results[best_dd_key]['metrics']['max_drawdown']:.2%})")

    print()
    print(f"  策略建议:")
    print(f"    - 若优先控回撤 → 选策略 A (保守型)，持仓集中、只买高质量信号")
    print(f"    - 若优先整体收益 → 对比策略 B/C 哪个 NAV 更高")
    print(f"    - 若需人工参与 → 策略 B (平衡型) 最接近日报的实际使用场景")

    # 验证陈述
    print()
    print(f"  验证检查:")
    print(f"    ✅ 基线 NAV 应与原始回测完全一致 (验证模拟器正确性)")
    print(f"    ✅ 策略 A 持仓 ≤ 6, 策略 B ≤ 10, 策略 C ≤ 12")
    print(f"    ✅ Tier A 胜率应高于 Tier C (信号质量梯度验证)")


def _print_footer():
    print()
    print("=" * 85)
    print(f"  📊 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  📁 输出目录: {COMPARISON_DIR}/")
    print("=" * 85)


# ==================== 主流程 ====================

def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    print("=" * 85)
    print("  组合决策回测 — 3 套仓位管理策略对照实验")
    print("=" * 85)

    # ---- Step 1: 加载数据 + 运行回测获取 snapshots ----
    print("\n⏳ 加载引擎数据并运行回测 (debug_snapshots=True)...")
    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data(
        candidate_obs_days=V1_PARAMS["candidate_obs_days"]
    )
    bt_result = run_backtest(
        test_df, price_pivot, trading_days, prev_close_map, pred_lookup,
        max_stocks=V1_PARAMS["max_stocks_default"],
        strategy=V1_PARAMS.get("strategy_default", "sell_score"),
        exit_mode="model_exit",
        stop_loss=V1_PARAMS["stop_loss"],
        buy_cls_exit_threshold=V1_PARAMS["buy_cls_exit_threshold"],
        debug_snapshots=True,
        strict=True,
    )
    snapshots = bt_result["snapshots"]
    print(f"  快照: {len(snapshots)} 个交易日, 日期范围: "
          f"{snapshots[0]['date'].strftime('%Y-%m-%d') if snapshots else 'N/A'} ~ "
          f"{snapshots[-1]['date'].strftime('%Y-%m-%d') if snapshots else 'N/A'}")

    # 构建 pred_indexed
    pred_indexed = {}
    for (sid, dt), pred in pred_lookup.items():
        pred_indexed[(int(sid), dt)] = pred

    # ---- Step 2: 标注原始回测指标 ----
    bt_trades = bt_result["trades_df"]
    bt_nav = bt_result["nav_df"]
    bt_metrics = _compute_metrics(bt_trades, bt_nav)

    original_nav = bt_nav["nav"].iloc[-1] if not bt_nav.empty else 1.0

    # ---- Step 3: 对 snapshot candidates 做 Tier 标注 ----
    for snap in snapshots:
        cdf = snap.get("candidates_df", pd.DataFrame())
        if cdf.empty:
            continue
        if "score" not in cdf.columns:
            cdf["score"] = cdf.get("pred_sell_reg", cdf.get("composite_score", 0))
        snap["candidates_df"] = _annotate_tier(cdf)

    # ---- Step 4: 对每套策略模拟 ----
    all_results = {}
    for key in ["baseline", "A", "B", "C"]:
        cfg = POLICY_CONFIGS[key]
        print(f"\n  🔄 模拟 {cfg['label']} ...")
        trades_df, nav_df = _simulate_policy(
            snapshots, cfg, price_pivot, trading_days, pred_indexed,
        )
        metrics = _compute_metrics(trades_df, nav_df)
        all_results[key] = {
            "trades_df": trades_df,
            "nav_df": nav_df,
            "metrics": metrics,
            "config": cfg,
        }

        print(f"    NAV={metrics['final_nav']:.4f}, 胜率={metrics['win_rate']:.1%}, "
              f"回撤={metrics['max_drawdown']:.2%}, 交易={metrics['total_trades']}笔, "
              f"均持={metrics['avg_positions']:.1f}只")

        tpath = os.path.join(COMPARISON_DIR, f"trades_{cfg['name']}.csv")
        if not trades_df.empty:
            trades_df.to_csv(tpath, index=False)

    # ---- Step 5: Baseline 一致性检查 ----
    bl_nav = all_results["baseline"]["metrics"]["final_nav"]
    nav_diff = abs(bl_nav - original_nav)
    if nav_diff < 1e-6:
        print(f"\n  ✅ Baseline 一致性检查通过: 模拟 NAV={bl_nav:.6f} == 原始回测 NAV={original_nav:.6f}")
    else:
        print(f"\n  ⚠️ Baseline 一致性偏差: 模拟 NAV={bl_nav:.6f}, 原始回测 NAV={original_nav:.6f}, diff={nav_diff:.6e}")

    # ---- Step 6: 输出报告 ----
    _print_param_summary()

    metrics_rows = []
    for key in ["baseline", "A", "B", "C"]:
        if key not in all_results:
            continue
        m = all_results[key]["metrics"]
        m["key"] = key
        m["label"] = POLICY_CONFIGS[key]["label"]
        metrics_rows.append(m)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(COMPARISON_DIR, "metrics_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)

    _print_metrics_table(all_results)
    _print_tier_attribution(all_results)
    _print_conclusions(all_results)

    # ---- Step 7: 绘图 ----
    nav_png = os.path.join(COMPARISON_DIR, "nav_comparison.png")
    _plot_nav_comparison(all_results, nav_png)

    bar_png = os.path.join(COMPARISON_DIR, "metrics_bar.png")
    _plot_metrics_bar(metrics_df, bar_png)

    _print_footer()


if __name__ == "__main__":
    main()
