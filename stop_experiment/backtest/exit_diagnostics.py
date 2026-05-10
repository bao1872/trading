#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
退出诊断专项：回答4个具体问题

Purpose:
    基于 full_test_predictions.parquet 中的 pred_buy_cls 和标签 (mfe_20, mae_20, buy_signal)，
    回答 buy_cls 退出阈值的 4 个核心问题，不依赖完整回测。

    问题1: pred_buy_cls 分桶后，后续收益 (mfe_20/mae_20) 是否单调变差
    问题2: pred_buy_cls 分桶后，MDD (mae_20) 是否单调变差
    问题3: 阈值上移时，误杀率 vs 回撤保护的 trade-off
    问题4: model_exit 相对 stop_loss/max_hold 的贡献分解

Pipeline Position:
    诊断工具（Step 5）。
    上游: generate_full_predictions.py
    下游: —

Inputs:
    - stop_experiment/output/full_test_predictions.parquet

Outputs:
    - stop_experiment/output/backtest/exit_diagnostics_bucket.csv
    - stop_experiment/output/backtest/exit_diagnostics_tradeoff.csv
    - stop_experiment/output/backtest/exit_diagnostics_contribution.csv

How to Run:
    python stop_experiment/backtest/exit_diagnostics.py

Side Effects:
    - 只读parquet，输出csv
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import OUTPUT_DIR, BACKTEST_DIR, V1_PARAMS, BASELINE_3Y_PARAMS


def load_data():
    """加载 test 集预测数据"""
    path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    df = pd.read_parquet(path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    print(f"加载: {path}")
    print(f"  行数: {len(df)}, 信号数: {df['signal_id'].nunique()}")

    required_cols = ["pred_buy_cls", "mfe_20", "mae_20", "buy_signal", "sell_signal"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")
    return df


def question1_bucket_analysis(df):
    """
    问题1+2: pred_buy_cls 分桶后，后续5/10/20日收益 & MDD 是否单调变差
    """
    print("\n" + "=" * 60)
    print("问题1+2: pred_buy_cls 分桶 vs 后续收益 & MDD")
    print("=" * 60)

    buckets = [
        (0.0, 0.3, "0.0-0.3"),
        (0.3, 0.5, "0.3-0.5"),
        (0.5, 0.7, "0.5-0.7"),
        (0.7, 0.8, "0.7-0.8"),
        (0.8, 1.0, "0.8-1.0"),
    ]

    rows = []
    for lo, hi, label in buckets:
        mask = (df["pred_buy_cls"] >= lo) & (df["pred_buy_cls"] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue

        row = {
            "bucket": label,
            "count": len(sub),
            "pct_total": len(sub) / len(df) * 100,
            "pred_buy_cls_mean": sub["pred_buy_cls"].mean(),
            "mfe_20_mean": sub["mfe_20"].mean() * 100,        # 后续20日最佳收益(%)
            "mfe_20_median": sub["mfe_20"].median() * 100,
            "mae_20_mean": sub["mae_20"].mean() * 100,        # 后续20日最差回撤(%)
            "mae_20_median": sub["mae_20"].median() * 100,
            "buy_signal_rate": sub["buy_signal"].mean() * 100,  # 实际买入信号率(%)
            "sell_signal_rate": sub["sell_signal"].mean() * 100,
            "mfe_gt_7pct": (sub["mfe_20"] > 0.07).mean() * 100,  # 有机会涨7%以上的比例
            "mae_lt_neg7pct": (sub["mae_20"] < -0.07).mean() * 100,  # 跌7%以上的比例
        }
        rows.append(row)
        print(f"\n  桶 {label} (n={len(sub)}):")
        print(f"    mfe_20 mean={sub['mfe_20'].mean()*100:.2f}%, median={sub['mfe_20'].median()*100:.2f}%")
        print(f"    mae_20 mean={sub['mae_20'].mean()*100:.2f}%, median={sub['mae_20'].median()*100:.2f}%")
        print(f"    买入信号率={sub['buy_signal'].mean()*100:.1f}%, 卖出信号率={sub['sell_signal'].mean()*100:.1f}%")
        print(f"    mfe>7%: {(sub['mfe_20']>0.07).mean()*100:.1f}%, mae<-7%: {(sub['mae_20']<-0.07).mean()*100:.1f}%")

    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "exit_diagnostics_bucket.csv")
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    df_out.to_csv(path, index=False)
    print(f"\n  输出: {path}")

    # 单调性判断
    mfe_means = df_out["mfe_20_mean"].values
    mae_means = df_out["mae_20_mean"].values
    mfe_monotonic = all(mfe_means[i] >= mfe_means[i+1] for i in range(len(mfe_means)-1))
    mae_monotonic = all(mae_means[i] <= mae_means[i+1] for i in range(len(mae_means)-1))
    print(f"\n  ✅ mfe_20 单调递减: {mfe_monotonic}")
    print(f"  ✅ mae_20 单调递减(越负): {mae_monotonic}")

    return df_out


def question3_threshold_tradeoff(df):
    """
    问题3: 阈值上移时，误杀率 vs 回撤保护的 trade-off
    """
    print("\n" + "=" * 60)
    print("问题3: 阈值 trade-off (误杀 vs 回撤保护)")
    print("=" * 60)

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    rows = []

    for th in thresholds:
        exit_mask = df["pred_buy_cls"] > th
        hold_mask = ~exit_mask

        n_exit = exit_mask.sum()
        n_hold = hold_mask.sum()
        n_total = len(df)

        # "本不该卖" (误杀): 卖出后实际 mfe_20 > 0 (还有上涨空间)
        false_positives = (exit_mask & (df["mfe_20"] > 0)).sum()
        fp_rate = false_positives / n_exit * 100 if n_exit > 0 else 0

        # "避免了后续大跌": 卖出后实际 mae_20 < -0.07 (确实该卖)
        true_exits_bad = (exit_mask & (df["mae_20"] < -0.07)).sum()
        te_rate = true_exits_bad / n_exit * 100 if n_exit > 0 else 0

        # "本不该留" (漏杀): 没卖但实际 mae_20 < -0.07
        false_negatives = (hold_mask & (df["mae_20"] < -0.07)).sum()
        fn_rate = false_negatives / n_hold * 100 if n_hold > 0 else 0

        # 卖出组的平均 mfe_20 和 mae_20
        exit_mfe = df.loc[exit_mask, "mfe_20"].mean() * 100 if n_exit > 0 else np.nan
        exit_mae = df.loc[exit_mask, "mae_20"].mean() * 100 if n_exit > 0 else np.nan
        hold_mfe = df.loc[hold_mask, "mfe_20"].mean() * 100 if n_hold > 0 else np.nan
        hold_mae = df.loc[hold_mask, "mae_20"].mean() * 100 if n_hold > 0 else np.nan

        print(f"\n  threshold={th:.2f}:")
        print(f"    卖出 {n_exit} 笔 ({n_exit/n_total*100:.1f}%)")
        print(f"    误杀(卖出后mfe>0): {false_positives}笔 ({fp_rate:.1f}%)")
        print(f"    正确卖出(卖出后mae<-7%): {true_exits_bad}笔 ({te_rate:.1f}%)")
        print(f"    漏杀(未卖但mae<-7%): {false_negatives}笔 ({fn_rate:.1f}%)")
        print(f"    卖出组 mfe_20均值={exit_mfe:.2f}%, mae_20均值={exit_mae:.2f}%")
        print(f"    持有组 mfe_20均值={hold_mfe:.2f}%, mae_20均值={hold_mae:.2f}%")

        rows.append({
            "threshold": th,
            "n_exit": n_exit,
            "exit_pct": n_exit / n_total * 100,
            "false_positives": false_positives,
            "fp_rate_pct": fp_rate,
            "true_exits_bad": true_exits_bad,
            "te_rate_pct": te_rate,
            "false_negatives": false_negatives,
            "fn_rate_pct": fn_rate,
            "exit_mfe_mean_pct": exit_mfe,
            "exit_mae_mean_pct": exit_mae,
            "hold_mfe_mean_pct": hold_mfe,
            "hold_mae_mean_pct": hold_mae,
        })

    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "exit_diagnostics_tradeoff.csv")
    df_out.to_csv(path, index=False)
    print(f"\n  输出: {path}")

    print(f"\n  trade-off 总结:")
    for _, row in df_out.iterrows():
        print(f"    th={row['threshold']:.2f}: 误杀={row['fp_rate_pct']:.1f}%, "
              f"漏杀={row['fn_rate_pct']:.1f}%, "
              f"正确卖出={row['te_rate_pct']:.1f}%")

    return df_out


def question4_exit_contribution(df):
    """
    问题4: model_exit vs stop_loss vs max_hold 的净值改善贡献
    基于标签数据进行反事实分析（不运行完整回测）

    思路:
    - 假设在每个 obs_date 买入，观察后续 20 天表现
    - 模拟三种退出策略对最终收益的影响:
      a. 无退出 (hold 20d): 收益 = mfe_20 或实际收益（取决于路径）
      b. model_exit: 若 pred_buy_cls > threshold 则提前退出
      c. stop_loss: 若达到 -7% 则退出
      d. max_hold: 20d 退出
    """
    print("\n" + "=" * 60)
    print("问题4: 退出原因贡献分解")
    print("=" * 60)

    threshold = V1_PARAMS.get("buy_cls_exit_threshold", 0.70)

    rows = []

    # 1. 仅 model_exit 触发 (pred_buy_cls > threshold 但 mae_20 > -0.07 即没触发止损)
    model_only = df[(df["pred_buy_cls"] > threshold) & (df["mae_20"] > -0.07)]
    rows.append({
        "exit_type": "model_risk",
        "count": len(model_only),
        "mfe_20_mean_pct": model_only["mfe_20"].mean() * 100 if len(model_only) > 0 else np.nan,
        "mae_20_mean_pct": model_only["mae_20"].mean() * 100 if len(model_only) > 0 else np.nan,
        "buy_signal_rate_pct": model_only["buy_signal"].mean() * 100 if len(model_only) > 0 else np.nan,
    })

    # 2. 仅 stop_loss 触发 (mae_20 < -0.07 但pred_buy_cls <= threshold)
    stop_only = df[(df["mae_20"] < -0.07) & (df["pred_buy_cls"] <= threshold)]
    rows.append({
        "exit_type": "stop_loss",
        "count": len(stop_only),
        "mfe_20_mean_pct": stop_only["mfe_20"].mean() * 100 if len(stop_only) > 0 else np.nan,
        "mae_20_mean_pct": stop_only["mae_20"].mean() * 100 if len(stop_only) > 0 else np.nan,
        "buy_signal_rate_pct": stop_only["buy_signal"].mean() * 100 if len(stop_only) > 0 else np.nan,
    })

    # 3. 两者同时触发 (pred_buy_cls > threshold AND mae_20 < -0.07)
    both = df[(df["pred_buy_cls"] > threshold) & (df["mae_20"] < -0.07)]
    rows.append({
        "exit_type": "both",
        "count": len(both),
        "mfe_20_mean_pct": both["mfe_20"].mean() * 100 if len(both) > 0 else np.nan,
        "mae_20_mean_pct": both["mae_20"].mean() * 100 if len(both) > 0 else np.nan,
        "buy_signal_rate_pct": both["buy_signal"].mean() * 100 if len(both) > 0 else np.nan,
    })

    # 4. 两者均不触发 (正常持有)
    neither = df[(df["pred_buy_cls"] <= threshold) & (df["mae_20"] >= -0.07)]
    rows.append({
        "exit_type": "neither",
        "count": len(neither),
        "mfe_20_mean_pct": neither["mfe_20"].mean() * 100 if len(neither) > 0 else np.nan,
        "mae_20_mean_pct": neither["mae_20"].mean() * 100 if len(neither) > 0 else np.nan,
        "buy_signal_rate_pct": neither["buy_signal"].mean() * 100 if len(neither) > 0 else np.nan,
    })

    for row in rows:
        print(f"\n  {row['exit_type']}: n={row['count']}")
        print(f"    mfe_20均值={row['mfe_20_mean_pct']:.2f}%, mae_20均值={row['mae_20_mean_pct']:.2f}%")
        print(f"    买入信号率={row['buy_signal_rate_pct']:.1f}%")

    # 反事实分析: 如果触发 model_exit 时我们就卖出
    # 对比: 实际持有20天的收益 vs 提前退出的收益
    # 简化: 用 mfe_20 作为"持有20天最优收益"，mae_20 作为风险
    # model_exit 的"改善" = 避免的损失 (mae_20 改善)
    model_risk_samples = df[df["pred_buy_cls"] > threshold]
    if len(model_risk_samples) > 0:
        avg_mae_model = model_risk_samples["mae_20"].mean()
        avg_mae_all = df["mae_20"].mean()
        improvement = (avg_mae_model - avg_mae_all) * 100  # 正值=更差，负值=更好
        print(f"\n  model_exit组 mae_20均值={avg_mae_model*100:.2f}% vs 全样本={avg_mae_all*100:.2f}%")
        print(f"  model_exit 筛选出的样本风险差: {improvement:+.2f}%")

    df_out = pd.DataFrame(rows)
    path = os.path.join(BACKTEST_DIR, "exit_diagnostics_contribution.csv")
    df_out.to_csv(path, index=False)
    print(f"\n  输出: {path}")

    return df_out


def main():
    print("=" * 60)
    print("退出诊断专项 (基于 full_test_predictions.parquet)")
    print("=" * 60)

    df = load_data()
    print(f"\n  pred_buy_cls 分布: mean={df['pred_buy_cls'].mean():.4f}, "
          f"median={df['pred_buy_cls'].median():.4f}, "
          f"std={df['pred_buy_cls'].std():.4f}")

    q1 = question1_bucket_analysis(df)
    q3 = question3_threshold_tradeoff(df)
    q4 = question4_exit_contribution(df)

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    main()