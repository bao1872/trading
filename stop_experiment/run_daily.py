#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产入口：每日盘后流水线编排（4+1 步）

Purpose:
    每日盘后一键运行完整流水线，依次执行：
    1. 日级推理回放（回测对账）— 06_daily_inference_replay.py
    2. 四本账落地（模拟盘运行）— 09_paper_trading_runner.py
    3. T+1 行动计划生成          — 10_tomorrow_action_plan.py
    4. 日报报告生成              — 08_daily_inference_report.py

Pipeline Position:
    生产入口（每日盘后）。
    上游: —
    下游: 06 → 09 → 10 → 08

Inputs:
    - stop_experiment/output/full_test_predictions.parquet
    - DB: stock_k_data

Outputs:
    - stop_experiment/output/daily_inference_replay_YYYY-MM-DD.csv (对账结果)
    - stop_experiment/output/live/decisions/YYYY-MM-DD.parquet    (决策账本)
    - stop_experiment/output/live/executions/YYYY-MM-DD.parquet   (执行账本)
    - stop_experiment/output/holdings/YYYY-MM-DD.parquet           (持仓账本)
    - stop_experiment/output/live/action_plans/YYYY-MM-DD.md       (T+1 行动计划)
    - stop_experiment/output/live/action_plans/YYYY-MM-DD.json     (T+1 行动计划 JSON)
    - stop_experiment/output/daily_report/YYYY-MM-DD.md            (研究日报)

How to Run:
    # 盘后一键运行（自动推断最新交易日）
    python stop_experiment/run_daily.py

    # 指定日期
    python stop_experiment/run_daily.py --date 2026-05-08

    # 跳过回测对账（只跑落账+行动计划+日报）
    python stop_experiment/run_daily.py --date 2026-05-08 --skip-replay

    # 跳过四本账落账
    python stop_experiment/run_daily.py --date 2026-05-08 --skip-ledger

    # 跳过行动计划
    python stop_experiment/run_daily.py --date 2026-05-08 --skip-action-plan

    # dry-run
    python stop_experiment/run_daily.py --date 2026-05-08 --dry-run

Side Effects:
    - 写 CSV / parquet / MD / JSON 到 output 目录
    - 不直接写数据库
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime


def run_replay(date_str, dry_run=False):
    """Step 1: 日级推理回放（回测对账）"""
    cmd = [
        sys.executable,
        "-m", "stop_experiment.pipeline.06_daily_inference_replay",
        "--date", date_str,
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n{'─'*70}")
    print(f"[Step 1/4] 回测对账")
    print(f"{'─'*70}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return result.returncode == 0


def run_ledger(date_str, dry_run=False):
    """Step 2: 四本账落地（模拟盘运行）"""
    cmd = [
        sys.executable,
        "-m", "stop_experiment.pipeline.09_paper_trading_runner",
        "--date", date_str,
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n{'─'*70}")
    print(f"[Step 2/4] 四本账落地")
    print(f"{'─'*70}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return result.returncode == 0


def run_action_plan(date_str):
    """Step 3: T+1 行动计划生成"""
    cmd = [
        sys.executable,
        "-m", "stop_experiment.pipeline.10_tomorrow_action_plan",
        "--date", date_str,
    ]

    print(f"\n{'─'*70}")
    print(f"[Step 3/4] T+1 行动计划")
    print(f"{'─'*70}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return result.returncode == 0


def run_report(date_str, dry_run=False, send_feishu=False):
    """Step 4: 日报报告生成"""
    cmd = [
        sys.executable,
        "-m", "stop_experiment.pipeline.08_daily_inference_report",
        "--date", date_str,
    ]
    if dry_run:
        cmd.append("--dry-run")
    if send_feishu:
        cmd.append("--send-feishu")

    print(f"\n{'─'*70}")
    print(f"[Step 4/4] 研究日报")
    print(f"{'─'*70}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="生产入口：每日盘后流水线编排（4+1 步）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 盘后一键运行
    python stop_experiment/run_daily.py
    python stop_experiment/run_daily.py --date 2026-05-08
    python stop_experiment/run_daily.py --date 2026-05-08 --skip-replay
    python stop_experiment/run_daily.py --date 2026-05-08 --skip-ledger
    python stop_experiment/run_daily.py --date 2026-05-08 --dry-run
        """,
    )
    parser.add_argument("--date", type=str, default=None,
                        help="指定日期 (YYYY-MM-DD)，默认自动推断最新交易日")
    parser.add_argument("--skip-replay", action="store_true",
                        help="跳过回测对账（Step 1）")
    parser.add_argument("--skip-ledger", action="store_true",
                        help="跳过四本账落地（Step 2）")
    parser.add_argument("--skip-action-plan", action="store_true",
                        help="跳过行动计划生成（Step 3）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只输出 console，不写文件")
    parser.add_argument("--send-feishu", action="store_true",
                        help="生成报告后发送飞书卡片消息到手机")

    args = parser.parse_args()

    date_str = args.date
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        print(f"未指定日期，使用今天: {date_str}")

    print("=" * 70)
    print(f"  生产入口: 每日盘后流水线 ({date_str})")
    print(f"  流程: 回测对账 → 四本账落地 → 行动计划 → 研究日报")
    print("=" * 70)

    ok = True
    step_results = {}

    # Step 1: 回测对账
    if not args.skip_replay:
        ok = run_replay(date_str, args.dry_run)
        step_results["replay"] = "✅" if ok else "❌"
        if not ok:
            print("\n  [WARNING] 回测对账失败，继续执行后续步骤...")
            # 不阻止后续步骤，因为 09 可独立运行
    else:
        step_results["replay"] = "⏭️"
        print(f"\n[跳过 Step 1] 回测对账")

    # Step 2: 四本账落地
    if not args.skip_ledger:
        if not run_ledger(date_str, args.dry_run):
            step_results["ledger"] = "❌"
            print("\n  [ERROR] 四本账落地失败，停止后续步骤")
            print(f"\n{'='*70}")
            print(f"  结果: ❌ 失败 (四本账落地异常)")
            print(f"  回测: {step_results.get('replay','?')}")
            print(f"  落账: ❌")
            sys.exit(1)
        step_results["ledger"] = "✅"
    else:
        step_results["ledger"] = "⏭️"
        print(f"\n[跳过 Step 2] 四本账落地")

    # Step 3: 行动计划
    if not args.skip_action_plan:
        if not run_action_plan(date_str):
            step_results["action_plan"] = "❌"
            print("\n  [WARNING] 行动计划生成失败，继续执行日报...")
        else:
            step_results["action_plan"] = "✅"
    else:
        step_results["action_plan"] = "⏭️"
        print(f"\n[跳过 Step 3] 行动计划")

    # Step 4: 研究日报
    if not run_report(date_str, args.dry_run, args.send_feishu):
        step_results["report"] = "❌"
    else:
        step_results["report"] = "✅"

    print(f"\n{'='*70}")
    print(f"  流水线完成")
    print(f"{'='*70}")
    print(f"  回测:     {step_results.get('replay','?')}")
    print(f"  四本账:   {step_results.get('ledger','?')}")
    print(f"  行动计划: {step_results.get('action_plan','?')}")
    print(f"  研究日报: {step_results.get('report','?')}")
    print()

    # 输出文件路径
    if not args.dry_run:
        from stop_experiment.pipeline.stop_config import DECISIONS_DIR, EXECUTIONS_DIR, HOLDINGS_DIR
        print("  产物位置:")
        print(f"    决策: {os.path.join(DECISIONS_DIR, f'{date_str}.parquet')}")
        print(f"    执行: {os.path.join(EXECUTIONS_DIR, f'{date_str}.parquet')}")
        print(f"    持仓: {os.path.join(HOLDINGS_DIR, f'{date_str}.parquet')}")
        print(f"    行动: {os.path.join(os.path.dirname(DECISIONS_DIR), 'action_plans', f'{date_str}.md')}")


if __name__ == "__main__":
    main()