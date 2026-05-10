#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型性能完整报告：计算胜率、盈亏比、夏普率等所有关键指标

Purpose:
    生成模型驱动的动态退出策略的完整性能报告。
    指标包括: 胜率、盈亏比、夏普率、Calmar、最大回撤、年化收益、交易笔数、平均持仓天数等。

Pipeline Position:
    诊断工具（回测完成后运行）。
    上游: dynamic_exit_backtest_v2.py
    下游: —

Inputs:
    - 回测结果 (trades_df, nav_df) — 由 dynamic_exit_backtest_v2.py 生成

Outputs:
    - Console: 性能指标报告
    - CSV: 性能指标汇总

How to Run:
    python stop_experiment/backtest/performance_report.py

Side Effects:
    - 输出性能报告到控制台和CSV
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.backtest.dynamic_exit_backtest_v2 import (
    _load_data, run_backtest, compute_summary as base_compute_summary
)
from stop_experiment.pipeline.stop_config import OUTPUT_DIR, BACKTEST_DIR


def compute_full_metrics(result: dict) -> dict:
    """
    计算完整的性能指标，包括：
    - 基础指标：NAV、年化收益、最大回撤、夏普率、卡玛比率
    - 交易指标：胜率、盈亏比、平均盈亏、最大单笔盈亏
    - 风险指标：VaR、波动率
    """
    nav_df = result["nav_df"]
    trades_df = result["trades_df"]
    params = result.get("params", {})
    
    if nav_df.empty or trades_df.empty:
        return {"error": "Empty data"}
    
    # ===== 基础指标 =====
    nav_df = nav_df.copy()
    nav_df["cummax"] = nav_df["nav"].cummax()
    nav_df["drawdown"] = (nav_df["nav"] - nav_df["cummax"]) / nav_df["cummax"]
    
    total_days = len(nav_df)
    total_years = total_days / 252
    final_nav = nav_df["nav"].iloc[-1]
    annual_ret = (final_nav ** (1 / total_years) - 1) if total_years > 0 and final_nav > 0 else 0
    max_dd = nav_df["drawdown"].min()
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 1e-6 else 0
    
    daily_rets = nav_df["daily_ret"]
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 1e-6 else 0
    volatility = daily_rets.std() * np.sqrt(252)  # 年化波动率
    
    # ===== 交易指标 =====
    n_trades = len(trades_df)
    net_rets = trades_df["net_ret"]
    gross_rets = trades_df["gross_ret"]
    
    # 胜率
    win_rate = (net_rets > 0).mean()
    
    # 盈亏比 (Profit Factor)
    total_wins = net_rets[net_rets > 0].sum()
    total_losses = abs(net_rets[net_rets < 0].sum())
    profit_factor = total_wins / total_losses if total_losses > 1e-6 else np.inf
    
    # 平均盈亏
    avg_win = net_rets[net_rets > 0].mean() if (net_rets > 0).any() else 0
    avg_loss = net_rets[net_rets < 0].mean() if (net_rets < 0).any() else 0
    win_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-6 else np.inf
    
    # 最大单笔盈亏
    max_win = net_rets.max()
    max_loss = net_rets.min()
    
    # 盈亏金额统计（假设本金100万）
    capital = 1_000_000
    avg_trade_value = capital / params.get("max_stocks", 10)
    total_pnl = net_rets.sum() * avg_trade_value
    avg_pnl = net_rets.mean() * avg_trade_value
    
    # ===== 风险指标 =====
    # VaR (95%)
    var_95 = np.percentile(daily_rets, 5) if len(daily_rets) > 0 else 0
    
    # 连续盈亏统计
    win_streak = 0
    loss_streak = 0
    current_streak = 0
    for ret in net_rets:
        if ret > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            win_streak = max(win_streak, current_streak)
        elif ret < 0:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            loss_streak = max(loss_streak, abs(current_streak))
        else:
            current_streak = 0
    
    return {
        # 基础指标
        "final_nav": final_nav,
        "annual_return": annual_ret,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "volatility": volatility,
        "total_days": total_days,
        
        # 交易统计
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "win_loss_ratio": win_loss_ratio,
        
        # 盈亏统计
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_net_return": net_rets.mean(),
        "avg_gross_return": gross_rets.mean(),
        "max_win": max_win,
        "max_loss": max_loss,
        
        # 金额统计（假设100万本金）
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl,
        
        #  streaks
        "max_win_streak": win_streak,
        "max_loss_streak": loss_streak,
        
        # 风险
        "var_95": var_95,
        
        # 参数
        **params
    }


def print_performance_report(metrics: dict, title: str = "性能报告"):
    """打印格式化的性能报告"""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    # 基础指标
    print("\n【基础收益指标】")
    print(f"  期末净值:           {metrics['final_nav']:.4f}")
    print(f"  年化收益率:         {metrics['annual_return']:.2%}")
    print(f"  最大回撤:           {metrics['max_drawdown']:.2%}")
    print(f"  夏普比率:           {metrics['sharpe_ratio']:.2f}")
    print(f"  卡玛比率:           {metrics['calmar_ratio']:.2f}")
    print(f"  年化波动率:         {metrics['volatility']:.2%}")
    print(f"  交易日数:           {metrics['total_days']}")
    
    # 交易统计
    print("\n【交易统计】")
    print(f"  总交易次数:         {metrics['n_trades']}")
    print(f"  胜率:               {metrics['win_rate']:.2%}")
    print(f"  盈亏比 (PF):        {metrics['profit_factor']:.2f}")
    print(f"  平均盈亏比:         {metrics['win_loss_ratio']:.2f}")
    
    # 盈亏详情
    print("\n【盈亏详情】")
    print(f"  平均盈利:           {metrics['avg_win']:.2%}")
    print(f"  平均亏损:           {metrics['avg_loss']:.2%}")
    print(f"  平均净收益:         {metrics['avg_net_return']:.2%}")
    print(f"  最大单笔盈利:       {metrics['max_win']:.2%}")
    print(f"  最大单笔亏损:       {metrics['max_loss']:.2%}")
    
    # 金额统计
    print("\n【金额统计 (假设本金100万)】")
    print(f"  总盈亏:             ¥{metrics['total_pnl']:,.0f}")
    print(f"  平均每笔盈亏:       ¥{metrics['avg_pnl_per_trade']:,.0f}")
    
    # 连续交易
    print("\n【连续交易统计】")
    print(f"  最长连续盈利:       {metrics['max_win_streak']} 笔")
    print(f"  最长连续亏损:       {metrics['max_loss_streak']} 笔")
    
    # 风险指标
    print("\n【风险指标】")
    print(f"  VaR (95%):          {metrics['var_95']:.2%}")
    
    print("=" * 80)


def main():
    print("=" * 80)
    print("模型性能完整报告")
    print("=" * 80)
    
    # 加载数据
    print("\n[1/2] 加载数据...")
    test_df, price_pivot, trading_days, prev_close_map, pred_lookup = _load_data([1, 2, 3])
    print(f"  候选信号: {len(test_df)}, 交易日: {len(trading_days)}")
    
    # 运行回测 - 模型退出
    print("\n[2/2] 运行回测 (模型退出策略)...")
    result = run_backtest(
        test_df, price_pivot, trading_days, prev_close_map, pred_lookup,
        max_stocks=10, strategy="sell_score",
        exit_mode="model_exit", strict=True,
    )
    
    # 计算完整指标
    metrics = compute_full_metrics(result)
    
    # 打印报告
    print_performance_report(metrics, "模型驱动动态退出 - 性能报告")
    
    # 对比：固定持有
    print("\n" + "=" * 80)
    print("对比: 固定持有策略 (hold_days=5)")
    print("=" * 80)
    
    result_fixed = run_backtest(
        test_df, price_pivot, trading_days, prev_close_map, pred_lookup,
        max_stocks=10, strategy="sell_score",
        exit_mode="fixed_hold", hold_days=5, strict=True,
    )
    metrics_fixed = compute_full_metrics(result_fixed)
    print_performance_report(metrics_fixed, "固定持有 - 性能报告")
    
    # 对比表
    print("\n" + "=" * 80)
    print("策略对比")
    print("=" * 80)
    print(f"{'指标':<20} {'固定持有':>15} {'模型退出':>15} {'差异':>15}")
    print("-" * 80)
    
    compare_items = [
        ("期末净值", "final_nav", ".4f"),
        ("年化收益率", "annual_return", ".2%"),
        ("最大回撤", "max_drawdown", ".2%"),
        ("夏普比率", "sharpe_ratio", ".2f"),
        ("卡玛比率", "calmar_ratio", ".2f"),
        ("胜率", "win_rate", ".2%"),
        ("盈亏比 (PF)", "profit_factor", ".2f"),
        ("平均盈亏比", "win_loss_ratio", ".2f"),
        ("平均盈利", "avg_win", ".2%"),
        ("平均亏损", "avg_loss", ".2%"),
        ("总交易次数", "n_trades", ".0f"),
    ]
    
    for name, key, fmt in compare_items:
        f_val = metrics_fixed.get(key, 0)
        m_val = metrics.get(key, 0)
        diff = m_val - f_val
        
        if "rate" in key or "return" in key or "drawdown" in key or "win" in key:
            # 百分比或比率类
            print(f"{name:<20} {f_val:>14.2%} {m_val:>14.2%} {diff:>+14.2%}")
        elif "ratio" in key or "factor" in key:
            # 比率类
            print(f"{name:<20} {f_val:>15.2f} {m_val:>15.2f} {diff:>+15.2f}")
        else:
            # 数字类
            print(f"{name:<20} {f_val:>15{fmt.replace('%', '').replace('f', '.0f')[1:]}} "
                  f"{m_val:>15{fmt.replace('%', '').replace('f', '.0f')[1:]}} "
                  f"{diff:>+15{fmt.replace('%', '').replace('f', '.0f')[1:]}}")
    
    print("=" * 80)
    
    # 保存到CSV
    output_dir = os.path.join(BACKTEST_DIR, "dynamic")
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_df = pd.DataFrame([
        {"strategy": "fixed_hold", **{k: metrics_fixed.get(k, np.nan) for k, _, _ in compare_items}},
        {"strategy": "model_exit", **{k: metrics.get(k, np.nan) for k, _, _ in compare_items}},
    ])
    
    output_path = os.path.join(output_dir, "performance_comparison.csv")
    comparison_df.to_csv(output_path, index=False)
    print(f"\n对比结果已保存: {output_path}")


if __name__ == "__main__":
    main()
