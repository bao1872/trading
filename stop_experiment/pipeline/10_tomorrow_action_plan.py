#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盘后行动计划生成器 — 读四本账，产出 T+1 操作指令

Purpose:
    每天收盘后，读取当天四本账（decisions/executions/holdings/predictions），
    生成一份可执行、可解释的 T+1 行动清单。
    不做任何交易逻辑，纯读账本 + 纯生成报告。

Pipeline Position:
    下游，被 09_paper_trading_runner.py 或 run_daily.py 编排调用。
    上游: 09_paper_trading_runner.py（四本账已落地）
    下游: 飞书/手机通知（未来）

Inputs:
    - stop_experiment/output/live/decisions/YYYY-MM-DD.parquet
    - stop_experiment/output/live/executions/YYYY-MM-DD.parquet
    - stop_experiment/output/holdings/YYYY-MM-DD.parquet
    - stop_experiment/output/predictions/YYYY-MM-DD.parquet (可选)

Outputs:
    - stop_experiment/output/live/action_plans/YYYY-MM-DD.md
    - stop_experiment/output/live/action_plans/YYYY-MM-DD.json

How to Run:
    # 指定日期
    python -m stop_experiment.pipeline.10_tomorrow_action_plan --date 2026-05-08

    # 自动找最新有 decisions 的日期
    python -m stop_experiment.pipeline.10_tomorrow_action_plan --latest

    # 单日运行
    python stop_experiment/pipeline/10_tomorrow_action_plan.py --date 2026-05-08

Side Effects:
    - 创建 action_plans/ 目录
    - 写 MD + JSON 两份文件
    - 只读 parquet，不写数据库
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, DECISIONS_DIR, EXECUTIONS_DIR, HOLDINGS_DIR, PREDICTIONS_DIR,
)

ACTION_PLANS_DIR = os.path.join(OUTPUT_DIR, "live", "action_plans")

DEFAULT_MAX_STOCKS = 10


# ==================== 数据加载 ====================

def _safe_read_parquet(dir_path, date_str, label):
    path = os.path.join(dir_path, f"{date_str}.parquet")
    if not os.path.exists(path):
        print(f"  [警告] 缺少{label}: {path}")
        return None
    return pd.read_parquet(path)


def _find_latest_decisions_date():
    if not os.path.exists(DECISIONS_DIR):
        return None
    files = [f for f in os.listdir(DECISIONS_DIR) if f.endswith(".parquet")]
    if not files:
        return None
    dates = [f.replace(".parquet", "") for f in files]
    dates.sort(reverse=True)
    return dates[0]


# ==================== 股票名称映射 ====================

_STOCK_NAME_CACHE = {}

def _get_stock_name(ts_code):
    code_clean = ts_code[:6] if "." in ts_code else ts_code
    if code_clean in _STOCK_NAME_CACHE:
        return _STOCK_NAME_CACHE.get(code_clean, code_clean)
    try:
        from stop_experiment.data.fetch_ak_data import ak
        import akshare as ak_mod
        info = ak_mod.stock_individual_info_em(symbol=code_clean)
        if info is not None and not info.empty:
            name_row = info[info["item"] == "股票简称"]
            if not name_row.empty:
                name = str(name_row["value"].values[0])
                _STOCK_NAME_CACHE[code_clean] = name
                return name
    except Exception:
        pass
    _STOCK_NAME_CACHE[code_clean] = code_clean
    return code_clean


# ==================== 报告生成 ====================

def _fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v*100:.2f}%"


def _fmt_float(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def _generate_empty_plan(date_str):
    """当日无决策数据时生成精简报告"""
    lines = [
        f"# 📋 T+1 行动计划 — {date_str}",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 执行日期: T+1={date_str} (下一个交易日)",
        "",
        "## ⚠️ 今日无交易信号",
        "",
        "当日候选池为空或无符合条件的交易信号，系统无买卖决策。",
        "",
        "---",
        "> ℹ️ 系统运行正常，等待下一个交易日信号",
    ]
    md_content = "\n".join(lines)
    json_data = {"decision_date": date_str, "generated_at": datetime.now().isoformat(),
                 "status": "no_signal", "overview": {},
                 "buy_list": [], "sell_list": [], "hold_list": [], "skipped_list": []}
    summary = f"[{date_str}] 无信号 ⏭️"
    return md_content, json_data, summary


def generate_action_plan(date_str):
    """
    读四本账 → 生成 T+1 行动计划的 MD + JSON 两份文件。
    返回 (md_content, json_data, summary_text)
    """

    # 加载数据
    dec_df = _safe_read_parquet(DECISIONS_DIR, date_str, "决策账本")
    exec_df = _safe_read_parquet(EXECUTIONS_DIR, date_str, "执行账本")
    hold_df = _safe_read_parquet(HOLDINGS_DIR, date_str, "持仓账本")
    pred_df = _safe_read_parquet(PREDICTIONS_DIR, date_str, "预测账本")

    if dec_df is None or len(dec_df) == 0:
        if dec_df is None:
            raise RuntimeError(f"缺少决策账本: {date_str}，无法生成行动计划。请先运行 09_paper_trading_runner.py")
        # 空 decisions: 生成精简报告
        print(f"  [信息] 决策账本为空 ({date_str})，可能当日无候选或无交易")
        md_content, json_data, summary = _generate_empty_plan(date_str)
        return md_content, json_data, summary

    # 分类
    buy_df = dec_df[dec_df["action"] == "buy"].copy() if dec_df is not None else pd.DataFrame()
    sell_df = dec_df[dec_df["action"] == "sell"].copy() if dec_df is not None else pd.DataFrame()
    hold_df_dec = dec_df[dec_df["action"] == "hold"].copy() if dec_df is not None else pd.DataFrame()

    skipped_df = pd.DataFrame()
    if exec_df is not None and "status" in exec_df.columns:
        skipped_df = exec_df[exec_df["status"] == "skipped"].copy()

    system_anomalies = []
    if dec_df is None:
        system_anomalies.append("missing_decisions")
    if exec_df is None:
        system_anomalies.append("missing_executions")
    if hold_df_dec is None or (isinstance(hold_df_dec, pd.DataFrame) and hold_df_dec.empty):
        system_anomalies.append("holdings_empty")

    n_buy = len(buy_df)
    n_sell = len(sell_df)
    n_hold = len(hold_df_dec)
    n_skip = len(skipped_df)
    n_anomaly = len(system_anomalies)

    # 获取股票名称
    all_codes = set()
    for _, r in buy_df.iterrows():
        all_codes.add(str(r.get("ts_code", "")))
    for _, r in sell_df.iterrows():
        all_codes.add(str(r.get("ts_code", "")))
    for _, r in hold_df_dec.iterrows():
        all_codes.add(str(r.get("ts_code", "")))
    name_map = {}
    for c in all_codes:
        if c:
            name_map[c] = _get_stock_name(c)

    # --- 构建 MD ---
    lines = []
    lines.append(f"# 📋 T+1 行动计划 — {date_str}")
    lines.append("")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 执行日期: T+1={date_str} (下一个交易日)")
    lines.append("")

    # 模块 1: 总览
    lines.append("## 📊 一、明日待执行总览")
    lines.append("")
    lines.append("| 类别 | 数量 | 说明 |")
    lines.append("|------|------|------|")
    lines.append(f"| 🟢 待买入 | **{n_buy}** 只 | 明日开盘买入 |")
    lines.append(f"| 🔴 待卖出 | **{n_sell}** 只 | 明日开盘卖出 |")
    lines.append(f"| 🔵 继续持有 | **{n_hold}** 只 | 不动 |")
    skip_note = f"({'; '.join(skipped_df['skip_reason'].unique())})" if n_skip > 0 else ""
    lines.append(f"| ⚠️ 执行跳过 | {n_skip} 只 {skip_note} | 因规则未执行 |")
    anomaly_note = ", ".join(system_anomalies) if system_anomalies else "无"
    lines.append(f"| ❌ 系统异常 | {n_anomaly} 条 ({anomaly_note}) | 需人工检查 |")
    lines.append("")

    # 模块 2: 买入清单
    lines.append("## 🟢 二、T+1 买入清单")
    lines.append("")
    if n_buy > 0:
        lines.append(f"共 **{n_buy}** 只，按优先级排列：")
        lines.append("")
        lines.append("| # | 股票 | 名称 | 分数 | 计划权重 | 决策链条 |")
        lines.append("|---|---|---|---|---|---|")
        for idx, (_, r) in enumerate(buy_df.iterrows()):
            ts = str(r.get("ts_code", "?"))
            name = name_map.get(ts, ts)
            score = _fmt_float(r.get("score"))
            weight = f"{r.get('planned_weight', 0.1)*100:.0f}%"
            chain = f"候选池排序第{idx+1} → 组合空位允许 → 不在持仓 → 生成买入单"
            lines.append(f"| {idx+1} | {ts} | {name} | {score} | {weight} | {chain} |")
        lines.append("")
    else:
        lines.append("无买入计划。")
        lines.append("")

    # 模块 3: 卖出清单
    lines.append("## 🔴 三、T+1 卖出清单")
    lines.append("")
    if n_sell > 0:
        lines.append(f"共 **{n_sell}** 只：")
        lines.append("")
        lines.append("| 股票 | 名称 | 原因 | 持有天数 | 收益率 | 触发条件 |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sell_df.iterrows():
            ts = str(r.get("ts_code", "?"))
            name = name_map.get(ts, ts)
            reason = r.get("reason", "unknown")
            # 从 holdings 获取收益率
            cur_ret = None
            days_held = "?"
            if hold_df is not None and "entry_price" in hold_df.columns:
                hr = hold_df[hold_df["code"] == ts[:6]]
                if not hr.empty:
                    hr1 = hr.iloc[0]
                    bp = hr1.get("entry_price")
                    cp = hr1.get("close") if "close" in hr1 else None
                    days_held = str(hr1.get("days_held", "?"))
                    if bp and cp and bp > 0 and not pd.isna(cp):
                        cur_ret = (cp - bp) / bp

            ret_str = _fmt_pct(cur_ret) if cur_ret is not None else "N/A"
            trigger = {
                "max_hold": "持仓天数>最大持有期",
                "stop_loss": "跌破止损线",
                "model_risk": "pred_buy_cls>退出阈值",
            }.get(reason, reason)
            lines.append(f"| {ts} | {name} | {reason} | {days_held} | {ret_str} | {trigger} |")
        lines.append("")
    else:
        lines.append("无卖出计划。")
        lines.append("")

    # 模块 4: 继续持有
    lines.append("## 🔵 四、继续持有清单")
    lines.append("")
    if n_hold > 0:
        lines.append(f"共 **{n_hold}** 只，继续观察：")
        lines.append("")
        lines.append("| 股票 | 名称 | 持有天数 | 浮盈 | 风险提示 |")
        lines.append("|---|---|---|---|---|")
        for _, r in hold_df_dec.iterrows():
            ts = str(r.get("ts_code", "?"))
            name = name_map.get(ts, ts)
            days = "?"
            cur_ret = None
            if hold_df is not None and "code" in hold_df.columns:
                hr = hold_df[hold_df["code"] == ts[:6]]
                if not hr.empty:
                    hr1 = hr.iloc[0]
                    days = str(hr1.get("days_held", "?"))
                    bp = hr1.get("entry_price")
                    if bp and bp > 0:
                        cur_ret = None  # 需要 T+1 开盘价，此处不计算
            ret_str = _fmt_pct(cur_ret) if cur_ret is not None else "—"
            risk = "正常"
            if days != "?" and int(days) >= 17:
                risk = "⚠️ 接近最大持有期"
            lines.append(f"| {ts} | {name} | {days}天 | {ret_str} | {risk} |")
        lines.append("")
    else:
        lines.append("无继续持有的股票。")
        lines.append("")

    # 模块 5: 跳过清单
    lines.append("## ⚠️ 五、被跳过/未执行清单")
    lines.append("")
    if n_skip > 0:
        lines.append(f"共 **{n_skip}** 笔，需人工确认：")
        lines.append("")
        lines.append("| 股票 | 动作 | 跳过原因 |")
        lines.append("|---|---|---|")
        for _, r in skipped_df.iterrows():
            ts = str(r.get("ts_code", "?"))
            action = r.get("action", "?")
            reason = r.get("skip_reason", "?")
            reason_cn = {
                "limit_up_skip": "涨停未成交",
                "limit_down_skip": "跌停未成交",
                "suspended_skip": "停牌跳过",
                "missing_price": "缺少开盘价",
                "already_held": "已在持仓中",
            }.get(reason, reason)
            lines.append(f"| {ts} | {action} | {reason_cn} |")
        lines.append("")
    else:
        lines.append("今日无执行跳过。")
        lines.append("")

    # 模块 6: 决策链条
    lines.append("## 🔗 六、决策链条摘要")
    lines.append("")
    lines.append("### 买入链条")
    if n_buy > 0:
        for _, r in buy_df.iterrows():
            ts = str(r.get("ts_code", "?"))
            name = name_map.get(ts, ts)
            score = _fmt_float(r.get("score"))
            lines.append(f"- **{ts} {name}**: 候选池排序 → 分数{score} → 组合空位允许 → 不在持仓 → 生成T+1买入单")
    else:
        lines.append("- 今日无买入信号")
    lines.append("")
    lines.append("### 卖出链条")
    if n_sell > 0:
        for _, r in sell_df.iterrows():
            ts = str(r.get("ts_code", "?"))
            name = name_map.get(ts, ts)
            reason = r.get("reason", "?")
            lines.append(f"- **{ts} {name}**: 触发{reason} → decide_eod判定卖出 → 生成T+1卖出单")
    else:
        lines.append("- 今日无卖出信号")
    lines.append("")
    lines.append("### 继续持有")
    if n_hold > 0:
        for _, r in hold_df_dec.iterrows():
            ts = str(r.get("ts_code", "?"))
            name = name_map.get(ts, ts)
            lines.append(f"- **{ts} {name}**: 未触发退出条件 → 继续持有")
    lines.append("")

    # 系统状态
    if system_anomalies:
        lines.append("---")
        lines.append(f"## ⚠️ 系统异常: {', '.join(system_anomalies)}")
        lines.append("")
    else:
        lines.append("---")
        lines.append("> ✅ 系统健康，今日无异常")
        lines.append("")

    md_content = "\n".join(lines)

    # --- 构建 JSON ---
    json_data = {
        "decision_date": date_str,
        "generated_at": datetime.now().isoformat(),
        "overview": {
            "pending_buys": n_buy,
            "pending_sells": n_sell,
            "continuing_holds": n_hold,
            "skipped": n_skip,
            "system_anomalies": n_anomaly,
        },
        "buy_list": [
            {
                "ts_code": str(r.get("ts_code", "")),
                "name": name_map.get(str(r.get("ts_code", "")), ""),
                "score": float(r.get("score", 0)) if r.get("score") is not None else None,
                "planned_weight": float(r.get("planned_weight", 0)) if r.get("planned_weight") is not None else None,
                "rank": i + 1,
            }
            for i, (_, r) in enumerate(buy_df.iterrows())
        ],
        "sell_list": [
            {
                "ts_code": str(r.get("ts_code", "")),
                "name": name_map.get(str(r.get("ts_code", "")), ""),
                "reason": str(r.get("reason", "")),
            }
            for _, r in sell_df.iterrows()
        ],
        "hold_list": [
            {
                "ts_code": str(r.get("ts_code", "")),
                "name": name_map.get(str(r.get("ts_code", "")), ""),
            }
            for _, r in hold_df_dec.iterrows()
        ],
        "skipped_list": [
            {
                "ts_code": str(r.get("ts_code", "")),
                "action": str(r.get("action", "")),
                "skip_reason": str(r.get("skip_reason", "")),
            }
            for _, r in skipped_df.iterrows()
        ] if n_skip > 0 else [],
        "system_anomalies": system_anomalies,
    }

    summary = f"[{date_str}] 买{n_buy} 卖{n_sell} 持{n_hold} 跳{n_skip}" + \
              (f" 异常:{','.join(system_anomalies)}" if system_anomalies else " ✅正常")

    return md_content, json_data, summary


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(description="盘后行动计划生成器")
    parser.add_argument("--date", type=str, default=None, help="指定日期 YYYY-MM-DD")
    parser.add_argument("--latest", action="store_true", help="自动找最新有 decisions 的日期")
    parser.add_argument("--json-only", action="store_true", help="只输出 JSON")
    args = parser.parse_args()

    date_str = args.date
    if args.latest or date_str is None:
        date_str = _find_latest_decisions_date()
        if date_str is None:
            print("错误: 找不到任何 decisions 文件，请先运行 09_paper_trading_runner.py")
            sys.exit(1)
        print(f"自动选择最新日期: {date_str}")

    print(f"\n{'='*70}")
    print(f"  📋 T+1 行动计划生成 — {date_str}")
    print(f"{'='*70}")

    md_content, json_data, summary = generate_action_plan(date_str)

    os.makedirs(ACTION_PLANS_DIR, exist_ok=True)

    if not args.json_only:
        md_path = os.path.join(ACTION_PLANS_DIR, f"{date_str}.md")
        with open(md_path, "w") as f:
            f.write(md_content)
        print(f"\n  ✅ Markdown: {md_path}")

    json_path = os.path.join(ACTION_PLANS_DIR, f"{date_str}.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ JSON:     {json_path}")

    print(f"\n  📊 {summary}")
    print(f"\n{'='*70}")

    if not args.json_only:
        print(f"\n{md_content}")


if __name__ == "__main__":
    main()