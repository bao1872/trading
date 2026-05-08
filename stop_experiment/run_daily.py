#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产入口：每日推理流水线编排

Purpose:
    每日生产流水线的统一编排入口，依次执行：
    1. 日级推理回放（回测对账）— 06_daily_inference_replay.py
    2. 日级推理报告（日报生成，含双轨仓位映射 W1/W3）— 08_daily_inference_report.py

Pipeline Position:
    生产入口（每日，重复）。
    上游: —
    下游: 06_daily_inference_replay.py, 08_daily_inference_report.py

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - DB: stock_k_data

Outputs:
    - Console: 对账结果 + 日报报告
    - stop_experiment/output/daily_inference_replay_YYYY-MM-DD.csv
    - stop_experiment/output/daily_inference_diff_YYYY-MM-DD.csv
    - stop_experiment/output/daily_report/YYYY-MM-DD.md

How to Run:
    # 默认跑最新日期
    python stop_experiment/run_daily.py

    # 指定日期
    python stop_experiment/run_daily.py --date 2026-05-08

    # 跳过回测对账（只生成日报）
    python stop_experiment/run_daily.py --date 2026-05-08 --skip-replay

    # dry-run（只输出 console，不写文件）
    python stop_experiment/run_daily.py --date 2026-05-08 --dry-run

Side Effects:
    - 只读 parquet 和 DB
    - 写 CSV（对账结果）和 Markdown（日报）到 output 目录
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime


def run_replay(date_str, dry_run=False):
    """运行日级推理回放（回测对账）。"""
    cmd = [
        sys.executable,
        "-m", "stop_experiment.pipeline.06_daily_inference_replay",
        "--date", date_str,
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"[Step 1/2] 运行回测对账: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return result.returncode == 0


def run_report(date_str, dry_run=False, send_feishu=False):
    """运行日级推理报告（日报生成）。"""
    cmd = [
        sys.executable,
        "-m", "stop_experiment.pipeline.08_daily_inference_report",
        "--date", date_str,
    ]
    if dry_run:
        cmd.append("--dry-run")
    if send_feishu:
        cmd.append("--send-feishu")

    print(f"[Step 2/2] 生成日报报告: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="生产入口：每日推理流水线编排",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stop_experiment/run_daily.py
    python stop_experiment/run_daily.py --date 2026-05-08
    python stop_experiment/run_daily.py --date 2026-05-08 --skip-replay
    python stop_experiment/run_daily.py --date 2026-05-08 --dry-run
        """,
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="指定日期 (YYYY-MM-DD)，默认自动推断最新交易日",
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="跳过回测对账，只生成日报",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出 console，不写文件",
    )
    parser.add_argument(
        "--send-feishu",
        action="store_true",
        help="生成报告后发送飞书卡片消息到手机",
    )

    args = parser.parse_args()

    date_str = args.date
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        print(f"未指定日期，使用今天: {date_str}")

    print("=" * 70)
    print(f"  生产入口: 每日推理流水线 ({date_str})")
    print("=" * 70)

    success = True

    if not args.skip_replay:
        if not run_replay(date_str, args.dry_run):
            print("[ERROR] 回测对账失败，停止后续步骤")
            success = False
    else:
        print("[跳过] 回测对账 (--skip-replay)")

    if success:
        if not run_report(date_str, args.dry_run, args.send_feishu):
            print("[ERROR] 日报生成失败")
            success = False

    print()
    if success:
        print("[完成] 每日推理流水线执行完毕")
    else:
        print("[失败] 部分步骤执行失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
