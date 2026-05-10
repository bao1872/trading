#!/usr/bin/env python3
"""
每日盘后统一入口 — 预测账本 → 模拟盘 → 行动计划 → 飞书通知

生产口径: 统一使用主线冻结基线 (PRODUCTION_PARAMS = BASELINE_E0_X1_V1_PARAMS)
  E0 Entry (pred_sell_reg 排序, 无 gate)
  X1 Exit (buy_cls > 0.70)
  obs_day = [1]
  top10 等权

用法：
  python -m stop_experiment.run_daily                        # production: 最新交易日
  python -m stop_experiment.run_daily --date 2026-05-08      # 指定日期(production需最新)
  python -m stop_experiment.run_daily --date 2026-05-08 --mode debug  # 调试模式
  python -m stop_experiment.run_daily --date 2026-05-08 --with-report --with-replay  # 含日报+回放

流程（production）：
  Step 0: 解析目标交易日
  Step 1: 上游覆盖检查（DB 原始数据是否覆盖 T）
  Step 2: 自动生成预测账本 (07, 幂等, 使用主线基线参数)
  Step 3: 09 模拟盘 (live 模式, 落四本账 + 净值曲线 + 交易报告)
  Step 3b: 产物验收 (净值曲线/交易报告存在性检查)
  Step 4: 10 行动计划 (正式模式)
  Step 5: 11 飞书通知 (含净值/日收益/回撤)
  [可选]: 08 日报 / 06 回放

输出：
  predictions/         → 预测账本 (由 Step 2 保证)
  live/executions/     → 执行账本
  live/decisions/      → 决策账本
  live/holdings/       → 持仓账本 (实际在 output/ 根目录)
  live/live_equity_curve.csv  → 净值曲线 (Step 3 后处理)
  live/live_trade_report.csv  → 交易盈亏报告 (Step 3 后处理)
  live/action_plans/   → 行动计划 MD+JSON
  reports/             → 日报 MD (--with-report)

副作用：通过子脚本写 parquet/MD，不直接写库表
"""

import sys
import os
import subprocess
import argparse
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, HOLDINGS_DIR, DECISIONS_DIR, EXECUTIONS_DIR, PREDICTIONS_DIR,
    PRODUCTION_PARAMS,
)

PD = pd


class Status:
    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


EXIT_CODES = {Status.SUCCESS: 0, Status.DEGRADED: 1, Status.FAILED: 2}


# ==================== 交易日推断 ====================

def resolve_target_trade_date(args):
    if args.date:
        return args.date
    from stop_experiment.backtest.simple_backtest import load_daily_prices, build_price_pivot
    daily = load_daily_prices("2024-01-01", "2027-01-01")
    _, trading_days, _ = build_price_pivot(daily)
    if len(trading_days) == 0:
        raise RuntimeError("无法获取交易日列表，价格数据为空")
    latest = max(trading_days)
    ds = latest.strftime("%Y-%m-%d")
    print(f"  [信息] 未指定 --date，自动使用最新交易日: {ds}")
    return ds


# ==================== 上游覆盖检查 ====================

def check_source_coverage(target_date):
    """
    检查上游 DB 原始数据是否覆盖目标交易日。
    检查项: stock_k_data 最新日线日期、stop_loss_selection 最新信号日期。
    返回 (ok, message)。
    """
    from sqlalchemy import text
    from datasource.database import get_engine

    target_dt = PD.to_datetime(target_date)
    engine = get_engine()

    try:
        with engine.connect() as conn:
            sql_k = "SELECT MAX(bar_time) FROM stock_k_data WHERE freq = 'd'"
            latest_k = conn.execute(text(sql_k)).scalar()
            sql_s = "SELECT MAX(signal_date) FROM stop_loss_selection"
            latest_s = conn.execute(text(sql_s)).scalar()
    finally:
        engine.dispose()

    messages = []
    messages.append(f"  目标交易日: {target_date}")
    messages.append(f"  K线最新日期: {str(latest_k)[:10] if latest_k else 'None'}")
    messages.append(f"  信号最新日期: {str(latest_s)[:10] if latest_s else 'None'}")

    if latest_k is None or PD.to_datetime(latest_k) < target_dt:
        messages.append(f"  ❌ K线数据未覆盖 {target_date}")
        messages.append(f"  请检查数据源 (stock_k_data) 更新状态")
        return False, "\n".join(messages)

    if latest_s is None or PD.to_datetime(latest_s) < target_dt:
        messages.append(f"  ❌ 信号数据未覆盖 {target_date}")
        messages.append(f"  请检查数据源 (stop_loss_selection) 更新状态")
        return False, "\n".join(messages)

    pred_path = os.path.join(PREDICTIONS_DIR, f"{target_date}.parquet")
    if os.path.exists(pred_path):
        messages.append(f"  ✅ predictions/{target_date}.parquet 已存在")
    else:
        messages.append(f"  ⚠️ predictions/{target_date}.parquet 不存在 (将在 Step2 自动生成)")

    messages.append(f"  ✅ 原始数据覆盖检查通过")
    return True, "\n".join(messages)


# ==================== 预测账本保证 ====================

def ensure_prediction_ledger(target_date):
    """
    若 predictions/T.parquet 不存在，自动调用 07 生成。
    返回 (success, message)。
    """
    pred_path = os.path.join(PREDICTIONS_DIR, f"{target_date}.parquet")
    if os.path.exists(pred_path):
        return True, f"预测账本已存在: {pred_path}"

    print(f"  [自动] 预测账本不存在，调用 07 生成...")
    cmd = [sys.executable, "-m",
           "stop_experiment.pipeline.07_generate_daily_predictions",
           "--date", target_date]
    try:
        result = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=300)
        output = result.stdout + "\n" + result.stderr
        print(output[-2000:] if len(output) > 2000 else output)
        if result.returncode == 0:
            return True, f"预测账本已生成: {pred_path}"
        else:
            return False, f"07_generate_daily_predictions 失败:\n{output[-1000:]}"
    except subprocess.TimeoutExpired:
        return False, "07_generate_daily_predictions 超时 (300s)"
    except Exception as e:
        return False, f"07_generate_daily_predictions 异常: {e}"


# ==================== 产物验收 ====================

def validate_daily_products(date_str):
    """
    每日产物存在性校验。
    核心账本 + 净值曲线 + 交易报告 + 行动计划。
    返回 (status, message)。
    """
    messages = []
    missing_required = False
    live_dir = os.path.join(OUTPUT_DIR, "live")

    required = [
        (os.path.join(DECISIONS_DIR, f"{date_str}.parquet"), "决策账本", True),
        (os.path.join(EXECUTIONS_DIR, f"{date_str}.parquet"), "执行账本", True),
        (os.path.join(live_dir, "live_equity_curve.csv"), "净值曲线", True),
        (os.path.join(live_dir, "live_trade_report.csv"), "交易报告", True),
        (os.path.join(live_dir, "action_plans", f"{date_str}.md"), "行动计划 MD", True),
    ]

    for path, label, required_flag in required:
        if os.path.exists(path):
            messages.append(f"    ✅ {label}: {path}")
        else:
            messages.append(f"    ❌ {label}: 未生成 — {path}")
            if required_flag:
                missing_required = True

    if missing_required:
        return Status.DEGRADED, "\n".join(messages)
    return Status.SUCCESS, "\n".join(messages)


# ==================== 子步骤执行 ====================

def run_step(step_name, cmd_parts):
    """运行一个子步骤，返回 (status, output)。"""
    full_cmd = [sys.executable, "-m"] + cmd_parts
    print(f"\n{'='*60}")
    print(f"  [{step_name}] 开始执行...")
    print(f"  命令: {' '.join(full_cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(full_cmd, cwd=_PROJECT_ROOT,
                                capture_output=True, text=True, timeout=600)
        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
        if result.returncode == 0:
            print(output[-2000:] if len(output) > 2000 else output)
            print(f"  [{step_name}] ✅ 完成")
            return Status.SUCCESS, output
        else:
            print(output[-3000:] if len(output) > 3000 else output)
            print(f"  [{step_name}] ❌ 失败 (exit={result.returncode})")
            return Status.FAILED, output
    except subprocess.TimeoutExpired:
        print(f"  [{step_name}] ❌ 超时 (600s)")
        return Status.FAILED, "超时"
    except Exception as e:
        print(f"  [{step_name}] ❌ 异常: {e}")
        return Status.FAILED, str(e)


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="每日盘后统一入口: 预测账本→模拟盘→行动计划→[可选日报/回放]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m stop_experiment.run_daily                                    # production 最新交易日
  python -m stop_experiment.run_daily --date 2026-05-08 --mode debug     # debug 旧日期
  python -m stop_experiment.run_daily --date 2026-05-08 --with-report --with-replay  # 含日报+回放
""",
    )
    parser.add_argument("--date", help="目标日期 YYYY-MM-DD")
    parser.add_argument("--mode", choices=["production", "debug"], default="production",
                        help="运行模式: production(严格)/debug(宽松)")
    parser.add_argument("--with-report", action="store_true", help="追加 08 日报")
    parser.add_argument("--with-replay", action="store_true", help="追加 06 回放(历史一致性校验)")
    parser.add_argument("--feishu", action="store_true", default=None,
                        help="启用飞书通知 (production 默认开启)")
    parser.add_argument("--no-feishu", action="store_true",
                        help="强制关闭飞书通知 (含 production)")
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"  🚀 run_daily 启动")
    print(f"  模式: {args.mode}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  生产基线: {PRODUCTION_PARAMS['profile']}")
    print(f"    Entry=E0({PRODUCTION_PARAMS['score_col']}), Exit=X1(buy_cls>{PRODUCTION_PARAMS['buy_cls_exit_threshold']})")
    print(f"    obs_day={PRODUCTION_PARAMS['candidate_obs_days']}, top{PRODUCTION_PARAMS['max_stocks']}等权")
    print(f"{'='*70}")

    # Step 0: 解析目标交易日
    date_str = resolve_target_trade_date(args)
    target_dt = PD.to_datetime(date_str)

    if args.mode == "production" and args.date:
        from stop_experiment.backtest.simple_backtest import load_daily_prices, build_price_pivot
        daily = load_daily_prices("2024-01-01", "2027-01-01")
        _, trading_days, _ = build_price_pivot(daily)
        latest_price = max(trading_days) if len(trading_days) > 0 else None
        if latest_price is not None and target_dt < latest_price:
            raise RuntimeError(
                f"production 模式不允许回退旧日期 {date_str}"
                f" (最新交易日: {latest_price.strftime('%Y-%m-%d')})。"
                f"请使用 --mode debug 允许旧日期"
            )

    print(f"\n  目标交易日: {date_str}")

    # Step 1: 上游覆盖检查
    print(f"\n--- Step 1: 上游覆盖检查 ---")
    ok, msg = check_source_coverage(date_str)
    print(msg)
    if not ok and args.mode == "production":
        print(f"\n❌ 上游覆盖不足，production 模式终止运行")
        sys.exit(EXIT_CODES[Status.FAILED])
    elif not ok:
        print(f"  [降级] debug 模式下继续运行（上游覆盖不足）")

    step_statuses = {}

    # Step 2: 自动生成预测账本
    print(f"\n--- Step 2: 预测账本 ---")
    ok2, msg2 = ensure_prediction_ledger(date_str)
    print(f"  {msg2}")
    if not ok2 and args.mode == "production":
        print(f"\n❌ 预测账本生成失败，production 模式终止运行")
        sys.exit(EXIT_CODES[Status.FAILED])
    elif not ok2:
        print(f"  [降级] debug 模式下继续运行（预测账本生成失败）")
        step_statuses["07_predictions"] = Status.FAILED
    else:
        step_statuses["07_predictions"] = Status.SUCCESS

    # Step 3: 09 模拟盘 — 优先 replay 模式（回测一致），日期不在 test 集时降级 live
    full_test_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    use_replay = False
    try:
        import pandas as pd
        full_dates = pd.read_parquet(
            full_test_path, columns=["obs_date"]
        )["obs_date"].unique()
        full_dates_set = {str(d)[:10] for d in full_dates}
        use_replay = date_str in full_dates_set
    except Exception:
        pass

    mode = "replay" if use_replay else "live"
    print(f"  使用 mode={mode} (date_str={date_str} in full_test={'Y' if use_replay else 'N'})")
    status, _ = run_step(
        "Step3-09模拟盘",
        ["stop_experiment.pipeline.09_paper_trading_runner", "--date", date_str, "--mode", mode],
    )
    step_statuses["09_ledger"] = status

    # Step 3b: 产物验收（仅校验，不阻断后续 Steps）
    if status == Status.SUCCESS:
        v_status, v_msg = validate_daily_products(date_str)
        print(f"\n--- Step 3b: 产物验收 ---")
        print(v_msg)
        if v_status != Status.SUCCESS:
            print(f"  ⚠️ 部分产物缺失，标记 DEGRADED")
        step_statuses["products_check"] = v_status

    # Step 4: 10 行动计划（正式模式，缺预测直接失败）
    status, _ = run_step(
        "Step4-10行动计划",
        ["stop_experiment.pipeline.10_tomorrow_action_plan", "--date", date_str],
    )
    step_statuses["10_action_plan"] = status

    # Step 5: 11 飞书通知
    send_feishu = args.feishu if args.feishu is not None else (args.mode == "production")
    if args.no_feishu:
        send_feishu = False
    if send_feishu:
        status, _ = run_step(
            "Step5-11飞书通知",
            ["stop_experiment.pipeline.11_feishu_notification", "--date", date_str],
        )
        step_statuses["11_feishu"] = status
    else:
        print(f"\n  ⏭️ 跳过 11 飞书通知 (--feishu 可启用)")

    # 可选: 06 回放
    if args.with_replay or args.mode == "debug":
        status, _ = run_step(
            "Step5-06回放",
            ["stop_experiment.pipeline.06_daily_inference_replay", "--date", date_str],
        )
        step_statuses["06_replay"] = status
    else:
        print(f"\n  ⏭️ 跳过 06 回放 (--with-replay 可启用)")

    # 可选: 08 日报
    if args.with_report or args.mode == "debug":
        status, _ = run_step(
            "Step6-08日报",
            ["stop_experiment.pipeline.08_daily_inference_report", "--date", date_str],
        )
        step_statuses["08_report"] = status
    else:
        print(f"\n  ⏭️ 跳过 08 日报 (--with-report 可启用)")

    # Step 7: 操作包生成
    status, _ = run_step(
        "Step7-操作包",
        ["stop_experiment.pipeline.12_build_operator_packet", "--date", date_str],
    )
    step_statuses["12_operator_packet"] = status

    # 汇总
    print(f"\n{'='*70}")
    print(f"  📊 运行汇总")
    print(f"{'='*70}")
    for step_name, s in step_statuses.items():
        icon = "✅" if s == Status.SUCCESS else "⚠️" if s == Status.DEGRADED else "❌"
        print(f"  {icon} {step_name}: {s}")

    if all(s == Status.SUCCESS for s in step_statuses.values()):
        overall = Status.SUCCESS
    elif any(s == Status.FAILED for s in step_statuses.values()):
        overall = Status.FAILED
    else:
        overall = Status.DEGRADED

    print(f"\n  总状态: {overall}")
    print(f"{'='*70}")

    # 最终: 只列真实存在的产物文件
    print(f"\n  产出文件:")
    candidates = [
        (os.path.join(PREDICTIONS_DIR, f"{date_str}.parquet"), "预测账本"),
        (os.path.join(HOLDINGS_DIR, f"{date_str}.parquet"), "持仓账本"),
        (os.path.join(DECISIONS_DIR, f"{date_str}.parquet"), "决策账本"),
        (os.path.join(EXECUTIONS_DIR, f"{date_str}.parquet"), "执行账本"),
        (os.path.join(OUTPUT_DIR, "live", "live_equity_curve.csv"), "净值曲线"),
        (os.path.join(OUTPUT_DIR, "live", "live_trade_report.csv"), "交易报告"),
        (os.path.join(OUTPUT_DIR, "live", "action_plans",
                      f"{date_str}.md"), "行动计划 MD"),
        (os.path.join(OUTPUT_DIR, "live", "action_plans",
                      f"{date_str}.json"), "行动计划 JSON"),
        (os.path.join(OUTPUT_DIR, "reports",
                      f"daily_report_{date_str}.md"), "日报"),
        (os.path.join(OUTPUT_DIR, f"daily_inference_diff_{date_str}.csv"), "06对账结果"),
    ]
    REQUIRED_LABELS = {"预测账本", "持仓账本", "决策账本", "执行账本", "净值曲线", "交易报告", "行动计划 MD", "行动计划 JSON"}
    for path, label in candidates:
        if os.path.exists(path):
            print(f"    ✅ {label}: {path}")
        elif label in REQUIRED_LABELS:
            print(f"    ❌ {label}: 未生成 — {path}")

    print(f"\n  ✅ run_daily 完成: {overall}")

    print(f"\n今日操作包：stop_experiment/output/live/operator_packets/{date_str}/")
    print(f"重点查看：")
    print(f"  - 02_account_snapshot.md")
    print(f"  - 03_t1_action_plan.md")

    sys.exit(EXIT_CODES[overall])


if __name__ == "__main__":
    main()