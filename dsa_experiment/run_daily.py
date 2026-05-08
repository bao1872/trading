#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSA策略每日生产主入口

Purpose: 编排 DSA 策略每日生产流程（因子计算入库 → 触发点扫描 → 周线选股 → 每日决策+飞书推送）
Inputs: 数据库行情数据 + 已训练模型
Outputs: 交易清单CSV + 飞书推送
How to Run:
    python dsa_experiment/run_daily.py --date 2026-04-30
    python dsa_experiment/run_daily.py --date today --notify
    python dsa_experiment/run_daily.py --date 2026-04-30 --dry-run
    python dsa_experiment/run_daily.py --date 2026-04-30 --step 3 --notify
    python dsa_experiment/run_daily.py --date 2026-04-30 --skip-factors  # 因子已计算时跳过
Examples:
    python dsa_experiment/run_daily.py --date 2026-04-30
    python dsa_experiment/run_daily.py --date today --notify
Side Effects:
    - Step 0: 写入 factor_value / event_trigger 表
    - Step 1: 写入 stock_dsa_vreversal_results 表
    - Step 3: 更新 portfolio_state.json, 发送飞书消息

流程:
    Step 0: 因子计算入库 (pipeline/daily_factor_update.py --freq 1d --skip-events)
    Step 0b: [周五] 周线因子入库 (pipeline/daily_factor_update.py --freq 1w --skip-events)
    Step 1: 增量扫描当日BBMACD V型反转触发点 (01_selection_dsa.py)
    Step 2: 周线精选选股 (06_weekly_selector.py)
    Step 3: 每日决策输出+飞书推送 (07_daily_trading_sheet.py)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline")
FACTOR_UPDATE_DIR = os.path.join(PROJECT_ROOT, "pipeline")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

REQUIRED_MODELS = [
    ("return_model", os.path.join(OUTPUT_DIR, "return_model", "model.txt")),
    ("risk_model", os.path.join(OUTPUT_DIR, "risk_model", "model.txt")),
    ("daily_return_model", os.path.join(OUTPUT_DIR, "daily_return_model", "model.txt")),
    ("daily_risk_model", os.path.join(OUTPUT_DIR, "daily_risk_model", "model.txt")),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="DSA策略每日生产主入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", type=str, required=True,
                        help="决策日期 (YYYY-MM-DD 或 'today')")
    parser.add_argument("--step", type=str, default="all",
                        choices=["0", "1", "2", "3", "all"],
                        help="执行步骤: 0=因子计算, 1=扫描触发点, 2=周线选股, 3=每日决策, all=全流程")
    parser.add_argument("--dry-run", action="store_true",
                        help="试运行，不更新持仓状态")
    parser.add_argument("--notify", action="store_true",
                        help="推送飞书消息")
    parser.add_argument("--skip-factors", action="store_true",
                        help="跳过 Step 0（因子已在 factor_value 中）")
    parser.add_argument("--skip-scan", action="store_true",
                        help="跳过 Step 1（已有触发点数据时）")
    return parser.parse_args()


def check_prerequisites(require_notify: bool = False):
    errors = []
    for name, path in REQUIRED_MODELS:
        if not os.path.exists(path):
            errors.append(f"模型文件缺失: {name} ({path})")

    sys.path.insert(0, PROJECT_ROOT)
    try:
        from config import DATABASE_URL
        from sqlalchemy import create_engine
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as e:
        errors.append(f"数据库连接失败: {e}")

    if require_notify:
        try:
            from config import FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_USER_ID
            if not all([FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_USER_ID]):
                errors.append("飞书配置不完整")
        except ImportError:
            errors.append("飞书配置缺失")

    if errors:
        for e in errors:
            print(f"  ❌ {e}")
        return False

    print("  ✅ 前置条件检查通过")
    print(f"     模型文件: {len(REQUIRED_MODELS)} 个")
    print(f"     数据库: 可连接")
    if require_notify:
        print(f"     飞书: 配置完整")
    return True


def run_step_script(script_path: str, extra_args: list[str], step_label: str,
                    cwd: str = None) -> bool:
    if not os.path.exists(script_path):
        print(f"  ❌ 脚本不存在: {script_path}")
        return False

    cmd = [sys.executable, script_path] + extra_args
    print(f"  执行: {' '.join(cmd)}")
    print()

    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=False)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"\n  ❌ {step_label} 失败 (exit={result.returncode}, 耗时{elapsed:.1f}s)")
            return False
        print(f"\n  ✅ {step_label} 完成 (耗时{elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ {step_label} 异常: {e} (耗时{elapsed:.1f}s)")
        return False


def step0_compute_factors(target_date: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Step 0: 因子计算入库 — {target_date}")
    print(f"{'=' * 60}")

    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    is_friday = target_dt.weekday() == 4

    script_path = os.path.join(FACTOR_UPDATE_DIR, "daily_factor_update.py")
    ok = run_step_script(
        script_path,
        ["--date", target_date, "--freq", "1d", "--skip-events"],
        "Step 0: 日线因子入库",
        cwd=PROJECT_ROOT,
    )
    if not ok:
        return False

    if is_friday:
        print(f"\n  [周五] 额外执行周线因子更新...")
        ok = run_step_script(
            script_path,
            ["--date", target_date, "--freq", "1w", "--skip-events"],
            "Step 0b: 周线因子入库",
            cwd=PROJECT_ROOT,
        )
        if not ok:
            return False

    return True


def step1_scan_triggers(target_date: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Step 1: 扫描 {target_date} BBMACD V型反转触发点")
    print(f"{'=' * 60}")
    return run_step_script(
        os.path.join(PIPELINE_DIR, "01_selection_dsa.py"),
        ["--date", target_date],
        "Step 1: 触发点扫描",
    )


def step2_weekly_select(target_date: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Step 2: 周线精选选股 — {target_date}")
    print(f"{'=' * 60}")
    return run_step_script(
        os.path.join(PIPELINE_DIR, "06_weekly_selector.py"),
        ["--date", target_date],
        "Step 2: 周线选股",
    )


def step3_daily_decision(target_date: str, dry_run: bool, notify: bool) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Step 3: 每日决策 — {target_date}")
    print(f"{'=' * 60}")
    args = ["--date", target_date]
    if dry_run:
        args.append("--dry-run")
    if notify:
        args.append("--notify")
    return run_step_script(
        os.path.join(PIPELINE_DIR, "07_daily_trading_sheet.py"),
        args,
        "Step 3: 每日决策",
    )


def main():
    args = parse_args()

    if args.date == "today":
        target_date = datetime.now().strftime("%Y-%m-%d")
    else:
        target_date = args.date

    print("=" * 60)
    print(f"DSA策略每日生产 — {target_date}")
    print("=" * 60)
    print(f"  步骤: {args.step}")
    print(f"  试运行: {args.dry_run}")
    print(f"  飞书推送: {args.notify}")
    print(f"  跳过因子: {args.skip_factors}")
    print(f"  跳过扫描: {args.skip_scan}")

    if not check_prerequisites(require_notify=args.notify):
        print("\n❌ 前置条件检查失败，退出")
        sys.exit(1)

    steps_to_run = []
    if args.step == "all":
        if not args.skip_factors:
            steps_to_run.append("0")
        if not args.skip_scan:
            steps_to_run.append("1")
        steps_to_run.extend(["2", "3"])
    else:
        steps_to_run.append(args.step)

    t_start = time.time()
    failed = False

    for step in steps_to_run:
        if step == "0":
            ok = step0_compute_factors(target_date)
        elif step == "1":
            ok = step1_scan_triggers(target_date)
        elif step == "2":
            ok = step2_weekly_select(target_date)
        elif step == "3":
            ok = step3_daily_decision(target_date, args.dry_run, args.notify)
        else:
            ok = False

        if not ok:
            failed = True
            break

    elapsed_total = time.time() - t_start

    print(f"\n{'=' * 60}")
    if failed:
        print(f"❌ DSA每日生产失败 — {target_date} (总耗时{elapsed_total:.1f}s)")
        sys.exit(1)
    else:
        print(f"✅ DSA每日生产完成 — {target_date} (总耗时{elapsed_total:.1f}s)")
        print(f"   输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
