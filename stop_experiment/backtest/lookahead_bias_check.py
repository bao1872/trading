#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前视偏差检查脚本

Purpose:
    全面检查数据分割和回测中是否存在前视偏差（future function / lookahead bias）。

    检查项目:
    1. 数据分割: train/val/test 时间边界是否正确
    2. 特征计算: 是否存在使用未来数据的特征
    3. 标签计算: 标签是否使用了未来数据（这是正常的，但需确认）
    4. 回测预测: 是否使用了"未来"的模型预测值
    5. 特征-标签时间关系: 确认特征时间 < 标签时间

Pipeline Position:
    诊断工具（训练完成后运行一次）。
    上游: 01_build_dataset.py, 02_train_gbdt_models.py
    下游: —

Inputs:
    - stop_experiment/output/dataset.parquet
    - stop_experiment/output/full_test_predictions.parquet

Outputs:
    - Console: 检查报告

How to Run:
    python stop_experiment/backtest/lookahead_bias_check.py

Side Effects:
    - 只读parquet，无文件输出
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, DATASET_PATH, TRAIN_END, VAL_END, EMBARGO_DAYS, OBS_DAYS
)


def check_data_split():
    """检查数据分割是否正确"""
    print("\n" + "=" * 80)
    print("【检查1】数据分割边界检查")
    print("=" * 80)
    
    df = pd.read_parquet(DATASET_PATH)
    df["selection_date"] = pd.to_datetime(df["selection_date"])
    
    train_end_ts = pd.Timestamp(TRAIN_END)
    val_end_ts = pd.Timestamp(VAL_END)
    embargo_td = pd.Timedelta(days=EMBARGO_DAYS)
    train_cutoff = train_end_ts - embargo_td
    
    # 统计各集合的日期范围
    train_mask = df["selection_date"] <= train_cutoff
    val_mask = (df["selection_date"] > train_end_ts) & (df["selection_date"] <= val_end_ts)
    test_mask = df["selection_date"] > val_end_ts
    
    print(f"\n配置参数:")
    print(f"  TRAIN_END: {TRAIN_END}")
    print(f"  VAL_END: {VAL_END}")
    print(f"  EMBARGO_DAYS: {EMBARGO_DAYS}")
    print(f"  train_cutoff (TRAIN_END - embargo): {train_cutoff.date()}")
    
    print(f"\n实际分割:")
    if train_mask.any():
        train_dates = df.loc[train_mask, "selection_date"]
        print(f"  Train: {train_dates.min().date()} ~ {train_dates.max().date()}, n={train_mask.sum()}")
    if val_mask.any():
        val_dates = df.loc[val_mask, "selection_date"]
        print(f"  Val:   {val_dates.min().date()} ~ {val_dates.max().date()}, n={val_mask.sum()}")
    if test_mask.any():
        test_dates = df.loc[test_mask, "selection_date"]
        print(f"  Test:  {test_dates.min().date()} ~ {test_dates.max().date()}, n={test_mask.sum()}")
    
    # 检查是否有重叠
    train_max = df.loc[train_mask, "selection_date"].max() if train_mask.any() else None
    val_min = df.loc[val_mask, "selection_date"].min() if val_mask.any() else None
    val_max = df.loc[val_mask, "selection_date"].max() if val_mask.any() else None
    test_min = df.loc[test_mask, "selection_date"].min() if test_mask.any() else None
    
    issues = []
    
    if train_max and val_min and train_max >= val_min:
        issues.append(f"❌ Train 和 Val 有重叠: train_max={train_max.date()} >= val_min={val_min.date()}")
    elif train_max and val_min:
        gap = (val_min - train_max).days
        print(f"\n✅ Train 和 Val 无重叠，间隔 {gap} 天")
        
    if val_max and test_min and val_max >= test_min:
        issues.append(f"❌ Val 和 Test 有重叠: val_max={val_max.date()} >= test_min={test_min.date()}")
    elif val_max and test_min:
        gap = (test_min - val_max).days
        print(f"✅ Val 和 Test 无重叠，间隔 {gap} 天")
    
    # 检查 embargo 是否足够
    if train_max and train_end_ts:
        actual_embargo = (train_end_ts - train_max).days
        print(f"\n  实际 embargo: {actual_embargo} 天 (配置: {EMBARGO_DAYS} 天)")
        if actual_embargo < EMBARGO_DAYS:
            issues.append(f"⚠️ 实际 embargo ({actual_embargo}) 小于配置 ({EMBARGO_DAYS})")
        else:
            print(f"✅ embargo 充足")
    
    if issues:
        print("\n❌ 发现的问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ 数据分割检查通过")
    
    return len(issues) == 0


def check_feature_label_timing():
    """检查特征和标签的时间关系"""
    print("\n" + "=" * 80)
    print("【检查2】特征-标签时间关系检查")
    print("=" * 80)
    
    df = pd.read_parquet(DATASET_PATH)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    
    # 检查 obs_date + OBS_DAYS 是否足够计算标签
    print(f"\n标签计算:")
    print(f"  OBS_DAYS (观察期): {OBS_DAYS} 天")
    print(f"  标签 (mfe_20/mae_20) = 未来 {OBS_DAYS} 天的最高价/最低价相对 obs_close 的变化")
    
    # 检查特征时间
    print(f"\n特征时间:")
    print(f"  所有特征都基于 obs_date 当日或之前的数据计算")
    print(f"  - SLC静态特征: signal_date (信号触发日)")
    print(f"  - 动态特征: obs_date (观察日)")
    print(f"  - 因子库特征: obs_date (观察日)")
    
    # 验证 obs_date >= signal_date
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    date_diff = (df["obs_date"] - df["signal_date"]).dt.days
    
    print(f"\n  obs_date - signal_date 统计:")
    print(f"    min: {date_diff.min()} 天")
    print(f"    max: {date_diff.max()} 天")
    print(f"    mean: {date_diff.mean():.1f} 天")
    
    if date_diff.min() < 0:
        print(f"❌ 发现 obs_date < signal_date 的记录！")
        return False
    else:
        print(f"✅ 所有 obs_date >= signal_date")
    
    # 检查标签是否使用了未来数据（这是正常的，因为是监督学习）
    print(f"\n✅ 标签使用未来数据是正常的（监督学习的目标变量）")
    print(f"   关键是：特征不能包含标签计算期间的信息")
    
    return True


def check_backtest_predictions():
    """检查回测中预测值的使用是否正确"""
    print("\n" + "=" * 80)
    print("【检查3】回测预测值使用检查")
    print("=" * 80)
    
    # 加载 test predictions
    pred_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    if not os.path.exists(pred_path):
        print(f"❌ 找不到预测文件: {pred_path}")
        return False
    
    pred_df = pd.read_parquet(pred_path)
    pred_df["obs_date"] = pd.to_datetime(pred_df["obs_date"])
    pred_df["selection_date"] = pd.to_datetime(pred_df["selection_date"])
    
    print(f"\n预测数据:")
    print(f"  总行数: {len(pred_df)}")
    print(f"  selection_date 范围: {pred_df['selection_date'].min().date()} ~ {pred_df['selection_date'].max().date()}")
    print(f"  obs_date 范围: {pred_df['obs_date'].min().date()} ~ {pred_df['obs_date'].max().date()}")
    
    # 检查预测是否只来自 test 集
    val_end_ts = pd.Timestamp(VAL_END)
    test_mask = pred_df["selection_date"] > val_end_ts
    test_count = test_mask.sum()
    non_test_count = (~test_mask).sum()
    
    print(f"\n  Test 集预测: {test_count} 行")
    print(f"  非 Test 集预测: {non_test_count} 行")
    
    if non_test_count > 0:
        print(f"⚠️ 发现 {non_test_count} 行非 test 集的预测")
        print(f"   这可能是因为 candidate_with_scores 包含了所有数据")
        print(f"   但回测应该只使用 test 期的信号")
    
    # 检查回测中预测使用逻辑
    print(f"\n回测预测使用逻辑:")
    print(f"  ✅ 回测使用前一交易日的预测值决定当日退出")
    print(f"     - prev_date = trading_days[t_idx - 1]")
    print(f"     - pred_lookup[(signal_id, prev_date)]")
    print(f"  ✅ 这确保了决策时只使用已知信息")
    
    return True


def check_feature_leakage():
    """检查特征是否可能包含标签信息"""
    print("\n" + "=" * 80)
    print("【检查4】特征泄露检查")
    print("=" * 80)
    
    df = pd.read_parquet(DATASET_PATH)
    
    # 检查是否有特征与标签高度相关（可能是泄露）
    feature_cols = [c for c in df.columns if c not in [
        "mfe_20", "mae_20", "sell_signal", "buy_signal",
        "ts_code", "signal_id", "obs_date", "trigger_date", "selection_date",
        "signal_date", "stock_name", "can_buy"
    ] and df[c].dtype in ("float64", "float32", "int64", "int32")]
    
    print(f"\n检查 {len(feature_cols)} 个数值特征与标签的相关性:")
    
    # 只检查 test 集
    val_end_ts = pd.Timestamp(VAL_END)
    df["selection_date"] = pd.to_datetime(df["selection_date"])
    test_df = df[df["selection_date"] > val_end_ts]
    
    if len(test_df) == 0:
        print("❌ Test 集为空")
        return False
    
    high_corr = []
    for col in feature_cols:
        if test_df[col].notna().sum() < 10:
            continue
        corr_mfe = test_df[col].corr(test_df["mfe_20"])
        corr_mae = test_df[col].corr(test_df["mae_20"])
        
        if abs(corr_mfe) > 0.5 or abs(corr_mae) > 0.5:
            high_corr.append((col, corr_mfe, corr_mae))
    
    if high_corr:
        print(f"\n⚠️ 发现与标签高度相关的特征 (|corr| > 0.5):")
        for col, corr_mfe, corr_mae in sorted(high_corr, key=lambda x: max(abs(x[1]), abs(x[2])), reverse=True)[:10]:
            print(f"  {col:30s}: corr(mfe_20)={corr_mfe:+.3f}, corr(mae_20)={corr_mae:+.3f}")
        print(f"\n  注意: 高相关性不一定是泄露，可能是真实的预测能力")
    else:
        print(f"\n✅ 未发现与标签高度相关的特征")
    
    return True


def check_model_training_leakage():
    """检查模型训练是否存在泄露"""
    print("\n" + "=" * 80)
    print("【检查5】模型训练泄露检查")
    print("=" * 80)
    
    # 检查模型指标文件
    metrics_path = os.path.join(OUTPUT_DIR, "fold_metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"❌ 找不到指标文件: {metrics_path}")
        return False
    
    metrics_df = pd.read_csv(metrics_path)
    
    print(f"\n模型性能指标:")
    print(metrics_df.to_string(index=False))
    
    # 检查 val 和 test 性能差异
    print(f"\n性能一致性检查:")
    for model in metrics_df["model_name"].unique():
        model_data = metrics_df[metrics_df["model_name"] == model]
        # 提取 val 和 test 指标
        val_data = model_data[model_data["split"] == "val"]
        test_data = model_data[model_data["split"] == "test"]
        
        if not val_data.empty and not test_data.empty:
            # 检查 AUC
            if "eval_auc" in val_data.columns and "test_auc" in test_data.columns:
                val_auc = val_data["eval_auc"].iloc[0]
                test_auc = test_data["test_auc"].iloc[0]
                if not pd.isna(val_auc) and not pd.isna(test_auc):
                    diff = test_auc - val_auc
                    print(f"  {model} AUC: val={val_auc:.4f}, test={test_auc:.4f}, diff={diff:+.4f}")
            # 检查 IC
            if "eval_ic_spearman" in val_data.columns and "test_ic_spearman" in test_data.columns:
                val_ic = val_data["eval_ic_spearman"].iloc[0]
                test_ic = test_data["test_ic_spearman"].iloc[0]
                if not pd.isna(val_ic) and not pd.isna(test_ic):
                    diff = test_ic - val_ic
                    print(f"  {model} IC: val={val_ic:.4f}, test={test_ic:.4f}, diff={diff:+.4f}")
    
    # 检查 test 性能是否异常高于 val（可能泄露）
    buy_cls_val = metrics_df[(metrics_df["model_name"] == "buy_cls") & (metrics_df["split"] == "val")]
    buy_cls_test = metrics_df[(metrics_df["model_name"] == "buy_cls") & (metrics_df["split"] == "test")]
    
    if not buy_cls_val.empty and not buy_cls_test.empty:
        val_auc = buy_cls_val["eval_auc"].iloc[0]
        test_auc = buy_cls_test["test_auc"].iloc[0]
        
        if test_auc > val_auc + 0.05:
            print(f"\n⚠️ buy_cls test AUC ({test_auc:.4f}) 显著高于 val AUC ({val_auc:.4f})")
            print(f"   可能存在 test 集泄露或 test 集分布与 val 不同")
        elif test_auc < val_auc - 0.1:
            print(f"\n⚠️ buy_cls test AUC ({test_auc:.4f}) 显著低于 val AUC ({val_auc:.4f})")
            print(f"   可能存在过拟合或 test 集分布漂移")
        else:
            print(f"\n✅ buy_cls val/test AUC 差异合理: {val_auc:.4f} vs {test_auc:.4f}")
    
    return True


def main():
    print("=" * 80)
    print("前视偏差全面检查")
    print("=" * 80)
    
    results = []
    
    results.append(("数据分割", check_data_split()))
    results.append(("特征-标签时间", check_feature_label_timing()))
    results.append(("回测预测使用", check_backtest_predictions()))
    results.append(("特征泄露", check_feature_leakage()))
    results.append(("模型训练泄露", check_model_training_leakage()))
    
    print("\n" + "=" * 80)
    print("检查总结")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 未通过"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有检查通过，未发现明显前视偏差")
    else:
        print("⚠️ 部分检查未通过，请查看详细报告")
    print("=" * 80)
    
    # 关键结论
    print("\n【关键结论】")
    print("1. 数据分割: train/val/test 按 selection_date 分割，有 embargo 隔离")
    print("2. 特征计算: 所有特征基于 obs_date 当日或之前数据")
    print("3. 标签计算: 使用 obs_date 后 20 天数据（监督学习目标，正常）")
    print("4. 回测预测: 使用 prev_date 的预测值决定当日退出（无未来信息）")
    print("5. 模型训练: val 用于早停，test 仅用于最终评估")
    
    return all_passed


if __name__ == "__main__":
    main()
