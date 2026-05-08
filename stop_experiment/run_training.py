#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练入口：GBDT 模型训练流水线编排

Purpose:
    训练流水线的统一编排入口，依次执行：
    1. 数据集构建 — 01_build_dataset.py
    2. GBDT 模型训练 — 02_train_gbdt_models.py
    3. 因子重要性分析 — 03_factor_importance.py
    4. 信号精选排序 — 04_signal_selector.py
    5. 全量 test 预测生成 — backtest/generate_full_predictions.py

Pipeline Position:
    训练入口（离线，一次性）。
    上游: —
    下游: 01_build_dataset.py → 02_train_gbdt_models.py → 03_factor_importance.py → 04_signal_selector.py → generate_full_predictions.py

Inputs:
    - DB: stop_loss_selection, stock_k_data
    - DB: factor_value_1d (验证用)

Outputs:
    - stop_experiment/output/dataset.parquet
    - stop_experiment/output/models/ (4个模型txt + 4个final模型txt)
    - stop_experiment/output/fold_metrics.csv
    - stop_experiment/output/feature_importance.csv
    - stop_experiment/output/candidate_with_scores.parquet
    - stop_experiment/output/test_predictions.parquet
    - stop_experiment/output/full_test_predictions.parquet

How to Run:
    # 完整训练流水线
    python stop_experiment/run_training.py

    # 跳过数据集构建（已有 dataset.parquet）
    python stop_experiment/run_training.py --skip-build

    # 跳过模型训练（已有模型）
    python stop_experiment/run_training.py --skip-train

    # 跳过因子重要性（耗时）
    python stop_experiment/run_training.py --skip-importance

    # 跳过 SHAP 分析（更耗时）
    python stop_experiment/run_training.py --no-shap

    # 小批量调试
    python stop_experiment/run_training.py --sample-limit 10000

Side Effects:
    - 读 DB，写 parquet/CSV/模型文件到 output 目录
    - 训练过程可能耗时较长（~30-60分钟）
"""

import argparse
import subprocess
import sys
import os


def run_step(name, module_path, extra_args=None):
    """运行单个训练步骤。"""
    cmd = [sys.executable, "-m", module_path]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[训练] {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if result.returncode != 0:
        print(f"[ERROR] {name} 失败，停止后续步骤")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="训练入口：GBDT 模型训练流水线编排",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stop_experiment/run_training.py
    python stop_experiment/run_training.py --skip-build
    python stop_experiment/run_training.py --skip-train --skip-importance
    python stop_experiment/run_training.py --sample-limit 10000
    python stop_experiment/run_training.py --no-shap
        """,
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="跳过数据集构建（假设已有 dataset.parquet）",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过模型训练（假设已有模型文件）",
    )
    parser.add_argument(
        "--skip-importance",
        action="store_true",
        help="跳过因子重要性分析",
    )
    parser.add_argument(
        "--skip-selector",
        action="store_true",
        help="跳过信号精选排序",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="跳过全量 test 预测生成",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="跳过 SHAP 分析（因子重要性中）",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="限制样本数（调试用）",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  训练入口: GBDT 模型训练流水线")
    print("=" * 70)

    extra = []
    if args.sample_limit:
        extra = ["--sample-limit", str(args.sample_limit)]

    steps = []

    # Step 1: 数据集构建
    if not args.skip_build:
        steps.append(("数据集构建", "stop_experiment.pipeline.01_build_dataset", extra))
    else:
        print("[跳过] 数据集构建 (--skip-build)")

    # Step 2: 模型训练
    if not args.skip_train:
        steps.append(("模型训练", "stop_experiment.pipeline.02_train_gbdt_models", extra))
    else:
        print("[跳过] 模型训练 (--skip-train)")

    # Step 3: 因子重要性
    if not args.skip_importance:
        imp_extra = extra.copy()
        if args.no_shap:
            imp_extra.append("--no-shap")
        steps.append(("因子重要性", "stop_experiment.pipeline.03_factor_importance", imp_extra))
    else:
        print("[跳过] 因子重要性 (--skip-importance)")

    # Step 4: 信号精选
    if not args.skip_selector:
        steps.append(("信号精选", "stop_experiment.pipeline.04_signal_selector", extra))
    else:
        print("[跳过] 信号精选 (--skip-selector)")

    # Step 5: 全量 test 预测生成
    if not args.skip_predictions:
        steps.append(("全量预测生成", "stop_experiment.backtest.generate_full_predictions", []))
    else:
        print("[跳过] 全量预测生成 (--skip-predictions)")

    success = True
    for i, (name, module, step_extra) in enumerate(steps, 1):
        print()
        print(f"[Step {i}/{len(steps)}] {name}")
        if not run_step(name, module, step_extra):
            success = False
            break

    print()
    if success:
        print("[完成] 训练流水线执行完毕")
        print("  输出文件:")
        print("    - output/dataset.parquet")
        print("    - output/models/")
        print("    - output/full_test_predictions.parquet")
        print("  下一步: 运行生产入口 python stop_experiment/run_daily.py")
    else:
        print("[失败] 部分步骤执行失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
