#!/usr/bin/env python3
"""
增量价值验证分析报告：按 advicement.txt 建议的渐进角色测试分析 GBDT 增量价值

Purpose: 对7组策略做8项指标对比，分析 GBDT 分数与收益的相关性，判断 GBDT 增量价值
Inputs: portfolio_nav.csv, portfolio_trade_log.csv, candidate_with_scores.parquet,
        return_model/fold_metrics.csv, risk_model/fold_metrics.csv
Outputs: 终端报告, portfolio/incremental_value_report.csv, portfolio/score_correlation.csv,
         portfolio/decile_report.csv, portfolio/factor_stability_by_year.csv
How to Run:
    python dsa_experiment/analyze_results.py
Examples:
    python dsa_experiment/analyze_results.py
Side Effects: 只读操作，输出文件到 dsa_experiment/output/portfolio/
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

STRATEGY_LABELS = {
    "baseline_all": "S1:全量等权(baseline)",
    "simple_bbmacd_top20": "S2:bbmacd排序top20",
    "gbdt_record_only": "S3:全量等权+记录分数",
    "gbdt_weighted_all": "S4:全量GBDT加权",
    "gbdt_return_top20": "S5:GBDT排序top20",
    "veto_only_20": "S6:全量veto20%",
    "gbdt_return_top20_veto20": "S7:GBDT top20+veto20%",
}

STRATEGY_ROLES = {
    "baseline_all": "基线组",
    "simple_bbmacd_top20": "基线组(简单规则)",
    "gbdt_record_only": "角色1:被动观察",
    "gbdt_weighted_all": "角色2:排序微调",
    "gbdt_return_top20": "角色2:强筛选",
    "veto_only_20": "角色3:独立veto",
    "gbdt_return_top20_veto20": "角色3:组合版",
}


def compute_portfolio_stats(nav: pd.DataFrame) -> dict:
    if nav.empty or len(nav) < 2:
        return {}
    rets = nav["daily_ret_net"].values
    n_periods = len(rets)
    total_ret = nav["nav_net"].iloc[-1] / nav["nav_net"].iloc[0] - 1
    ann_ret = (1 + total_ret) ** (52 / n_periods) - 1 if n_periods > 0 else 0
    ann_vol = np.std(rets) * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    dd = nav["nav_net"] / nav["nav_net"].cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (rets > 0).mean()
    avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss = abs(rets[rets < 0].mean()) if (rets < 0).any() else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    avg_n = nav["n_stocks"].mean()
    avg_stop_rate = nav["stop_hit_rate"].mean()
    avg_mae = nav["avg_mae_5"].mean()
    avg_mfe = nav["avg_mfe_5"].mean()

    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "n_periods": n_periods,
        "avg_n_stocks": avg_n,
        "avg_stop_rate": avg_stop_rate,
        "avg_mae_5": avg_mae,
        "avg_mfe_5": avg_mfe,
    }


def compute_trade_level_stats(trade_log: pd.DataFrame) -> dict:
    if trade_log.empty:
        return {}
    strategy = trade_log["strategy"].iloc[0] if "strategy" in trade_log.columns else ""
    use_weight = strategy == "gbdt_weighted_all"
    w = trade_log["weight"] if use_weight and "weight" in trade_log.columns else None

    rets = trade_log["ret_5_gross"]
    if w is not None and w.sum() > 0:
        w_norm = w / w.sum()
        win_rate = (rets > 0).mean()
        opportunity_rate = (rets > 0.05).mean()
        avg_ret = np.average(rets, weights=w_norm)
        avg_mae = np.average(trade_log["mae_5"], weights=w_norm)
        avg_mfe = np.average(trade_log["mfe_5"], weights=w_norm)
        risk_event_rate = np.average(trade_log["stop_hit_5"], weights=w_norm)
        win_mask = rets > 0
        loss_mask = rets < 0
        avg_win = np.average(rets[win_mask], weights=w_norm[win_mask.values]) if win_mask.any() else 0
        avg_loss = abs(np.average(rets[loss_mask], weights=w_norm[loss_mask.values])) if loss_mask.any() else 0
    else:
        win_rate = (rets > 0).mean()
        opportunity_rate = (rets > 0.05).mean()
        avg_ret = rets.mean()
        avg_mae = trade_log["mae_5"].mean()
        avg_mfe = trade_log["mfe_5"].mean()
        risk_event_rate = trade_log["stop_hit_5"].mean()
        avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
        avg_loss = abs(rets[rets < 0].mean()) if (rets < 0).any() else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 0
    coverage = trade_log.groupby("selection_date")["ts_code"].count().mean()

    return {
        "win_rate": win_rate,
        "opportunity_rate": opportunity_rate,
        "avg_return": avg_ret,
        "avg_mae_5": avg_mae,
        "avg_mfe_5": avg_mfe,
        "risk_event_rate": risk_event_rate,
        "rr": rr,
        "coverage": coverage,
    }


def compute_yearly_stats(nav: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame()
    nav = nav.copy()
    nav["year"] = pd.to_datetime(nav["selection_date"]).dt.year
    rows = []
    for year, grp in nav.groupby("year"):
        if len(grp) < 2:
            continue
        rets = grp["daily_ret_net"].values
        total_ret = grp["nav_net"].iloc[-1] / grp["nav_net"].iloc[0] - 1
        max_dd = (grp["nav_net"] / grp["nav_net"].cummax() - 1).min()
        win_rate = (rets > 0).mean()
        rows.append({
            "strategy": strategy,
            "year": year,
            "n_periods": len(grp),
            "total_return": total_ret,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "avg_n_stocks": grp["n_stocks"].mean(),
        })
    return pd.DataFrame(rows)


def analyze_score_correlation(trade_log: pd.DataFrame) -> pd.DataFrame:
    record_only = trade_log[trade_log["strategy"] == "gbdt_record_only"].copy()
    if record_only.empty:
        return pd.DataFrame()

    valid = record_only[record_only["return_score"].notna() & record_only["ret_5_gross"].notna()].copy()
    if len(valid) < 50:
        return pd.DataFrame()

    rows = []
    for sel_date, grp in valid.groupby("selection_date"):
        if len(grp) < 10:
            continue
        ic_ret, _ = stats.spearmanr(grp["return_score"], grp["ret_5_gross"])
        ic_risk, _ = stats.spearmanr(grp["risk_score"], grp["mae_5"])
        rows.append({
            "selection_date": sel_date,
            "n": len(grp),
            "ic_return_score_vs_ret5": ic_ret,
            "ic_risk_score_vs_mae5": ic_risk,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    mean_ic_ret = result["ic_return_score_vs_ret5"].mean()
    std_ic_ret = result["ic_return_score_vs_ret5"].std()
    mean_ic_risk = result["ic_risk_score_vs_mae5"].mean()
    std_ic_risk = result["ic_risk_score_vs_mae5"].std()

    print(f"    收益分 IC: {mean_ic_ret:.4f} ± {std_ic_ret:.4f}, ICIR={mean_ic_ret/std_ic_ret:.2f}" if std_ic_ret > 0 else f"    收益分 IC: {mean_ic_ret:.4f}")
    print(f"    风险分 IC: {mean_ic_risk:.4f} ± {std_ic_risk:.4f}, ICIR={mean_ic_risk/std_ic_risk:.2f}" if std_ic_risk > 0 else f"    风险分 IC: {mean_ic_risk:.4f}")

    valid_risk = record_only[record_only["risk_score"].notna() & record_only["stop_hit_5"].notna()].copy()
    if len(valid_risk) > 50 and valid_risk["stop_hit_5"].nunique() > 1:
        auc = roc_auc_score(valid_risk["stop_hit_5"], valid_risk["risk_score"])
        print(f"    风险分 AUC (预测 stop_hit_5): {auc:.4f}")

    return result


def analyze_decile_report(trade_log: pd.DataFrame) -> pd.DataFrame:
    record_only = trade_log[trade_log["strategy"] == "gbdt_record_only"].copy()
    if record_only.empty:
        return pd.DataFrame()

    valid = record_only[record_only["return_score"].notna() & record_only["ret_5_gross"].notna()].copy()
    if len(valid) < 50:
        return pd.DataFrame()

    valid["decile"] = pd.qcut(valid["return_score"], 10, labels=False, duplicates="drop") + 1

    decile_stats = valid.groupby("decile").agg(
        n=("ret_5_gross", "count"),
        avg_ret=("ret_5_gross", "mean"),
        avg_mae=("mae_5", "mean"),
        avg_mfe=("mfe_5", "mean"),
        win_rate=("ret_5_gross", lambda x: (x > 0).mean()),
        stop_rate=("stop_hit_5", "mean"),
        avg_return_score=("return_score", "mean"),
        avg_risk_score=("risk_score", "mean"),
    ).reset_index()

    print(f"\n    {'Decile':>6} {'N':>6} {'avg_ret':>8} {'avg_MAE':>8} {'avg_MFE':>8} {'胜率':>6} {'止损率':>6}")
    print(f"    {'-'*56}")
    for _, row in decile_stats.iterrows():
        print(f"    {int(row['decile']):>6} {int(row['n']):>6} {row['avg_ret']:>7.2%} {row['avg_mae']:>7.2%} {row['avg_mfe']:>7.2%} {row['win_rate']:>5.0%} {row['stop_rate']:>5.0%}")

    d1 = decile_stats[decile_stats["decile"] == 1]
    d10 = decile_stats[decile_stats["decile"] == 10]
    if not d1.empty and not d10.empty:
        spread = d10["avg_ret"].values[0] - d1["avg_ret"].values[0]
        print(f"\n    D10-D1 收益差: {spread:.2%}")

    return decile_stats


def analyze_incremental_value(trade_log: pd.DataFrame) -> pd.DataFrame:
    strategies = trade_log["strategy"].unique()
    rows = []

    for strategy in strategies:
        strat_log = trade_log[trade_log["strategy"] == strategy]
        ts = compute_trade_level_stats(strat_log)
        ts["strategy"] = strategy
        ts["role"] = STRATEGY_ROLES.get(strategy, "")
        rows.append(ts)

    result = pd.DataFrame(rows)

    baseline = result[result["strategy"] == "baseline_all"]
    if baseline.empty:
        return result

    b = baseline.iloc[0]
    print(f"\n    {'策略':<30} {'角色':<16} {'胜率':>6} {'收益':>7} {'MFE':>7} {'MAE':>7} {'rr':>5} {'风险率':>6} {'机会率':>6} {'覆盖':>5}")
    print(f"    {'-'*105}")
    for _, row in result.iterrows():
        print(f"    {STRATEGY_LABELS.get(row['strategy'], row['strategy']):<30} {row['role']:<16} {row['win_rate']:>5.0%} {row['avg_return']:>6.2%} {row['avg_mfe_5']:>6.2%} {row['avg_mae_5']:>6.2%} {row['rr']:>5.2f} {row['risk_event_rate']:>5.0%} {row['opportunity_rate']:>5.0%} {row['coverage']:>5.0f}")

    print(f"\n    --- 相对 baseline_all 的增量 ---")
    print(f"    {'策略':<30} {'Δ胜率':>7} {'Δ收益':>7} {'ΔMFE':>7} {'ΔMAE':>7} {'Δrr':>6} {'Δ风险率':>7} {'Δ机会率':>7} {'Δ覆盖':>6}")
    print(f"    {'-'*105}")
    for _, row in result.iterrows():
        if row["strategy"] == "baseline_all":
            continue
        d_win = row["win_rate"] - b["win_rate"]
        d_ret = row["avg_return"] - b["avg_return"]
        d_mfe = row["avg_mfe_5"] - b["avg_mfe_5"]
        d_mae = row["avg_mae_5"] - b["avg_mae_5"]
        d_rr = row["rr"] - b["rr"]
        d_risk = row["risk_event_rate"] - b["risk_event_rate"]
        d_opp = row["opportunity_rate"] - b["opportunity_rate"]
        d_cov = row["coverage"] - b["coverage"]
        print(f"    {STRATEGY_LABELS.get(row['strategy'], row['strategy']):<30} {d_win:>+6.1%} {d_ret:>+6.2%} {d_mfe:>+6.2%} {d_mae:>+6.2%} {d_rr:>+5.2f} {d_risk:>+6.1%} {d_opp:>+6.1%} {d_cov:>+5.0f}")

    return result


def judge_gbdt_value(incremental_df: pd.DataFrame) -> None:
    baseline = incremental_df[incremental_df["strategy"] == "baseline_all"]
    simple = incremental_df[incremental_df["strategy"] == "simple_bbmacd_top20"]
    if baseline.empty:
        print("    无法判断：baseline_all 缺失")
        return

    b = baseline.iloc[0]

    criterion_1_pass = False
    criterion_2_pass = False
    criterion_3_pass = False

    gbdt_top20 = incremental_df[incremental_df["strategy"] == "gbdt_return_top20"]
    if not gbdt_top20.empty and not simple.empty:
        r = gbdt_top20.iloc[0]
        s = simple.iloc[0]
        if r["win_rate"] > s["win_rate"] and r["avg_return"] > s["avg_return"]:
            criterion_1_pass = True

    gbdt_weighted = incremental_df[incremental_df["strategy"] == "gbdt_weighted_all"]
    if not gbdt_weighted.empty:
        r = gbdt_weighted.iloc[0]
        coverage_drop = (b["coverage"] - r["coverage"]) / b["coverage"] if b["coverage"] > 0 else 0
        if coverage_drop < 0.1:
            if r["avg_mfe_5"] > b["avg_mfe_5"] or r["avg_mae_5"] < b["avg_mae_5"] or r["rr"] > b["rr"] + 0.05:
                criterion_2_pass = True

    veto = incremental_df[incremental_df["strategy"] == "veto_only_20"]
    if not veto.empty:
        r = veto.iloc[0]
        coverage_drop = (b["coverage"] - r["coverage"]) / b["coverage"] if b["coverage"] > 0 else 0
        if coverage_drop < 0.3:
            if r["risk_event_rate"] < b["risk_event_rate"] - 0.02:
                criterion_3_pass = True

    print(f"    标准1 - GBDT top20 vs 简单bbmacd top20（同覆盖率）提高胜率: {'✅ 通过' if criterion_1_pass else '❌ 未通过'}")
    print(f"    标准2 - GBDT加权 vs baseline（同覆盖率）改善MFE/MAE/rr: {'✅ 通过' if criterion_2_pass else '❌ 未通过'}")
    print(f"    标准3 - 独立veto vs baseline 降低风险事件率: {'✅ 通过' if criterion_3_pass else '❌ 未通过'}")

    n_pass = sum([criterion_1_pass, criterion_2_pass, criterion_3_pass])
    if n_pass >= 1:
        print(f"\n    结论: GBDT 通过 {n_pass}/3 条标准，有增量价值")
        if criterion_1_pass:
            print("    定位建议: GBDT 排序能力显著优于简单规则，适合做 top20 筛选（角色2）")
        if criterion_2_pass:
            print("    定位建议: GBDT 加权可改善全量池表现，适合做排序微调（角色2）")
        if criterion_3_pass:
            print("    定位建议: GBDT 风险模型可降低风险事件率，适合做 veto（角色3）")
    else:
        print(f"\n    结论: GBDT 未通过任何标准，大概率没必要作为主筛选器")
        print("    定位建议: 如需使用，仅做辅助 veto 或放弃")


def main():
    parser = argparse.ArgumentParser(description="增量价值验证分析报告")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 80)
    print("增量价值验证分析报告")
    print("（对齐 advicement.txt 第七节渐进角色测试）")
    print("=" * 80)

    portfolio_dir = os.path.join(args.output_dir, "portfolio")
    nav_path = os.path.join(portfolio_dir, "portfolio_nav.csv")
    log_path = os.path.join(portfolio_dir, "portfolio_trade_log.csv")

    nav_all = pd.read_csv(nav_path)
    trade_log = pd.read_csv(log_path)

    strategies = nav_all["strategy"].unique()

    # ── 第1部分：实验设计 ──
    print("\n" + "=" * 80)
    print("1. 实验设计（对齐 advicement.txt）")
    print("=" * 80)
    print("""
  核心问题: GBDT 相对于"只用 BBMACD 周线选股"到底带来了什么净变化？

  7组策略:
    S1 baseline_all:          全量 BBMACD 触发股，等权（第一步 baseline）
    S2 simple_bbmacd_top20:   按 bbmacd 原始值排序取 top20，等权（组4：简单规则）
    S3 gbdt_record_only:      全量持仓+记录GBDT分数，等权（角色1：被动观察）
    S4 gbdt_weighted_all:     全量持仓，按 return_score 加权（角色2：排序微调）
    S5 gbdt_return_top20:     按 return_score 排序取 top20，等权（角色2：强筛选版）
    S6 veto_only_20:          全量持仓，剔除 risk_score 前20%，剩余等权（角色3：独立veto）
    S7 gbdt_return_top20_veto20: top20 + veto 20%（角色3：组合版）

  8项比较指标: 胜率、平均收益、MFE、MAE、盈亏比、风险事件率、机会事件率、覆盖率
  3条判断标准:
    1. 不明显降低覆盖率前提下提高胜率
    2. 胜率差不多前提下改善 MFE/MAE/rr
    3. 收益差不多前提下降低风险事件率
""")

    # ── 第2部分：样本定义 ──
    print("\n" + "=" * 80)
    print("2. 样本定义")
    print("=" * 80)
    scored_path = os.path.join(args.output_dir, "candidate_with_scores.parquet")
    if os.path.exists(scored_path):
        df = pd.read_parquet(scored_path)
        print(f"  候选池记录: {len(df)}")
        print(f"  日期范围: {df['selection_date'].min()} ~ {df['selection_date'].max()}")
        print(f"  可交易比例: {df['can_buy_next_open'].mean():.1%}")
        print(f"  ret_5_open_to_open 均值: {df['ret_5_open_to_open'].mean():.4%}")
        print(f"  stop_hit_5 命中率: {df['stop_hit_5'].mean():.2%}")
        print(f"  成本假设: 买入0.2% + 卖出0.2% = 0.4%/轮")

    # ── 第3部分：模型结果 ──
    print("\n" + "=" * 80)
    print("3. 收益模型结果")
    print("=" * 80)
    ret_metrics_path = os.path.join(args.output_dir, "return_model", "fold_metrics.csv")
    if os.path.exists(ret_metrics_path):
        ret_m = pd.read_csv(ret_metrics_path)
        print(f"\n  {'折名':<10} {'IC':>8} {'MAE':>8} {'训练量':>8} {'测试量':>8}")
        print(f"  {'-'*50}")
        for _, row in ret_m.iterrows():
            print(f"  {row['fold']:<10} {row['test_ic_spearman']:>8.4f} {row['test_mae']:>8.4f} {row['n_train']:>8} {row['n_test']:>8}")
        ic_mean = ret_m["test_ic_spearman"].mean()
        ic_std = ret_m["test_ic_spearman"].std()
        print(f"\n  跨折: IC={ic_mean:.4f} ± {ic_std:.4f}, ICIR={ic_mean/ic_std:.2f}" if ic_std > 0 else f"  IC={ic_mean:.4f}")

    print("\n" + "=" * 80)
    print("4. 风险模型结果")
    print("=" * 80)
    risk_metrics_path = os.path.join(args.output_dir, "risk_model", "fold_metrics.csv")
    if os.path.exists(risk_metrics_path):
        risk_m = pd.read_csv(risk_metrics_path)
        print(f"\n  {'折名':<10} {'AUC':>8} {'AP':>8} {'训练量':>8} {'测试量':>8}")
        print(f"  {'-'*50}")
        for _, row in risk_m.iterrows():
            print(f"  {row['fold']:<10} {row.get('test_auc', 0):>8.4f} {row.get('test_ap', 0):>8.4f} {row['n_train']:>8} {row['n_test']:>8}")
        auc_mean = risk_m["test_auc"].mean()
        print(f"\n  跨折: AUC={auc_mean:.4f}")

    # ── 第5部分：组合净值结果 ──
    print("\n" + "=" * 80)
    print("5. 组合净值结果")
    print("=" * 80)
    all_stats = []
    for strategy in strategies:
        nav = nav_all[nav_all["strategy"] == strategy].sort_values("selection_date").reset_index(drop=True)
        s = compute_portfolio_stats(nav)
        if s:
            s["strategy"] = strategy
            all_stats.append(s)

    if all_stats:
        print(f"\n  {'策略':<30} {'年化':>8} {'最大回撤':>8} {'卡玛':>8} {'夏普':>8} {'胜率':>6} {'盈亏比':>6} {'止损率':>6} {'持仓':>5}")
        print(f"  {'-'*95}")
        for s in all_stats:
            print(f"  {STRATEGY_LABELS.get(s['strategy'], s['strategy']):<30} {s['ann_return']:>7.1%} {s['max_drawdown']:>7.1%} {s['calmar']:>7.2f} {s['sharpe']:>7.2f} {s['win_rate']:>5.0%} {s['profit_loss_ratio']:>6.2f} {s['avg_stop_rate']:>5.0%} {s['avg_n_stocks']:>5.0f}")

    # ── 第6部分：增量价值对比（8项指标） ──
    print("\n" + "=" * 80)
    print("6. 增量价值对比（8项指标，对齐 advicement.txt 第三节）")
    print("=" * 80)
    incremental_df = analyze_incremental_value(trade_log)
    if not incremental_df.empty:
        incremental_df.to_csv(os.path.join(portfolio_dir, "incremental_value_report.csv"), index=False)

    # ── 第7部分：角色1 分数相关性分析 ──
    print("\n" + "=" * 80)
    print("7. 角色1 分数相关性分析（GBDT 分数是否与实际收益相关？）")
    print("=" * 80)
    score_corr = analyze_score_correlation(trade_log)
    if not score_corr.empty:
        score_corr.to_csv(os.path.join(portfolio_dir, "score_correlation.csv"), index=False)

    print("\n  Decile 报告（按 return_score 分10组，看实际收益/风险单调性）:")
    decile_df = analyze_decile_report(trade_log)
    if not decile_df.empty:
        decile_df.to_csv(os.path.join(portfolio_dir, "decile_report.csv"), index=False)

    # ── 第8部分：稳定性分析（分年） ──
    print("\n" + "=" * 80)
    print("8. 稳定性分析（分年）")
    print("=" * 80)
    yearly_dfs = []
    for strategy in strategies:
        nav = nav_all[nav_all["strategy"] == strategy].sort_values("selection_date").reset_index(drop=True)
        yearly = compute_yearly_stats(nav, strategy)
        yearly_dfs.append(yearly)

    if yearly_dfs:
        yearly_all = pd.concat(yearly_dfs, ignore_index=True)
        for strategy in strategies:
            strat_yearly = yearly_all[yearly_all["strategy"] == strategy]
            if strat_yearly.empty:
                continue
            label = STRATEGY_LABELS.get(strategy, strategy)
            print(f"\n  {label}:")
            print(f"    {'年份':>6} {'收益':>8} {'回撤':>8} {'胜率':>6} {'期数':>6}")
            print(f"    {'-'*40}")
            for _, row in strat_yearly.iterrows():
                print(f"    {int(row['year']):>6} {row['total_return']:>7.1%} {row['max_drawdown']:>7.1%} {row['win_rate']:>5.0%} {row['n_periods']:>6}")

        yearly_all.to_csv(os.path.join(portfolio_dir, "factor_stability_by_year.csv"), index=False)

    # ── 第9部分：GBDT 增量价值判断 ──
    print("\n" + "=" * 80)
    print("9. GBDT 增量价值判断（对齐 advicement.txt 第八节）")
    print("=" * 80)
    print("\n  三条判断标准:")
    judge_gbdt_value(incremental_df)

    # ── 第10部分：失效点与问题 ──
    print("\n" + "=" * 80)
    print("10. 失效点与问题")
    print("=" * 80)
    if all_stats:
        issues = []
        top20_stats = next((s for s in all_stats if s["strategy"] == "gbdt_return_top20"), {})
        baseline_stats = next((s for s in all_stats if s["strategy"] == "baseline_all"), {})

        if top20_stats:
            if top20_stats["ann_return"] < 0:
                issues.append("❌ GBDT top20 成本后年化为负")
            if abs(top20_stats["max_drawdown"]) > 0.20:
                issues.append("❌ GBDT top20 最大回撤 > 20%")
            if top20_stats["calmar"] < 1.0:
                issues.append("⚠️ GBDT top20 卡玛比率 < 1.0")

        if baseline_stats:
            if baseline_stats["ann_return"] < 0:
                issues.append("⚠️ Baseline 年化为负，BBMACD 本身非强信号")

        simple_stats = next((s for s in all_stats if s["strategy"] == "simple_bbmacd_top20"), {})
        if simple_stats and top20_stats:
            if simple_stats["ann_return"] > 0 and top20_stats["ann_return"] <= simple_stats["ann_return"]:
                issues.append("⚠️ GBDT top20 不优于简单 bbmacd top20，GBDT 无增量价值")

        veto_stats = next((s for s in all_stats if s["strategy"] == "veto_only_20"), {})
        if veto_stats and baseline_stats:
            dd_improve = abs(baseline_stats["max_drawdown"]) - abs(veto_stats["max_drawdown"])
            if dd_improve < 0.03:
                issues.append("⚠️ 独立 veto 回撤改善不足3个百分点")
            if veto_stats["ann_return"] < baseline_stats["ann_return"] * 0.8:
                issues.append("⚠️ 独立 veto 牺牲收益过多")

        if not issues:
            print("  ✅ 未发现严重问题")
        else:
            for issue in issues:
                print(f"  {issue}")

    # ── 第11部分：最终结论 ──
    print("\n" + "=" * 80)
    print("11. 最终结论")
    print("=" * 80)
    if all_stats:
        baseline_s = next((s for s in all_stats if s["strategy"] == "baseline_all"), {})
        simple_s = next((s for s in all_stats if s["strategy"] == "simple_bbmacd_top20"), {})
        top20_s = next((s for s in all_stats if s["strategy"] == "gbdt_return_top20"), {})
        veto_s = next((s for s in all_stats if s["strategy"] == "veto_only_20"), {})

        if baseline_s:
            print(f"\n  Baseline (全量等权): 年化={baseline_s.get('ann_return', 0):.1%}, 回撤={baseline_s.get('max_drawdown', 0):.1%}")
        if simple_s:
            print(f"  简单规则 (bbmacd top20): 年化={simple_s.get('ann_return', 0):.1%}, 回撤={simple_s.get('max_drawdown', 0):.1%}")
        if top20_s:
            print(f"  GBDT top20: 年化={top20_s.get('ann_return', 0):.1%}, 回撤={top20_s.get('max_drawdown', 0):.1%}")
        if veto_s:
            print(f"  独立 veto: 年化={veto_s.get('ann_return', 0):.1%}, 回撤={veto_s.get('max_drawdown', 0):.1%}")

        if simple_s and top20_s:
            if top20_s["ann_return"] > simple_s["ann_return"] * 1.2:
                print(f"\n  🟢 GBDT top20 显著优于简单规则，GBDT 有增量价值")
            elif top20_s["ann_return"] > simple_s["ann_return"]:
                print(f"\n  🟡 GBDT top20 略优于简单规则，增量价值有限")
            else:
                print(f"\n  🔴 GBDT top20 不优于简单规则，GBDT 无增量价值")

        if baseline_s and baseline_s["ann_return"] < 0:
            print(f"\n  ⚠️ Baseline 年化为负，BBMACD 本身并非'强信号'，GBDT 排序价值可能来自信号提取而非增强")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
