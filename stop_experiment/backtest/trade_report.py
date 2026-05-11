#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易盈亏报告生成器（后处理模块）

Purpose:
    从回测产物 trades_df 生成可读的交易盈亏明细报告，含中文股票名称、
    逐笔盈亏百分比、盈亏标签、汇总统计。

Pipeline Position:
    后处理模块，位于 run_backtest() 之后。
    上游：dynamic_exit_backtest_v2.run_backtest()
    下游：run_baseline.py, position_sizing_w1.py

Inputs:
    - trades_df: 含 ts_code, buy_date, sell_date, buy_price, sell_price,
                 hold_days, gross_ret, net_ret, sell_reason, score,
                 exit_mode, weight (可选)

Outputs:
    - trade_report.csv: 逐笔交易明细（含中文名称）
    - trade_summary.csv: 汇总统计

How to Run:
    python stop_experiment/backtest/trade_report.py

Side Effects:
    - 写 CSV
    - 读数据库 (stock_pools) 查询股票名称
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd


def _fetch_stock_names_batch(ts_codes):
    """批量查询 ts_code -> 股票名称，从 stock_pools 表"""
    if not ts_codes:
        return {}
    try:
        from datasource.database import get_engine
        from sqlalchemy import text
        engine = get_engine()
        sql = text("SELECT ts_code, name FROM stock_pools WHERE ts_code = ANY(:codes)")
        with engine.connect() as conn:
            result = conn.execute(sql, {"codes": list(ts_codes)})
            return {row[0]: row[1] for row in result}
    except Exception:
        return {}


def build_trade_report(trades_df):
    """
    从 trades_df 生成可读的交易盈亏报告

    Args:
        trades_df: 含 ts_code, buy_date, sell_date, buy_price, sell_price,
                   hold_days, gross_ret, net_ret, sell_reason, score, exit_mode

    Returns:
        (report_df, summary_dict)
    """
    if trades_df.empty:
        return pd.DataFrame(), {"总交易数": 0}

    df = trades_df.copy()

    has_weight = "weight" in df.columns

    unique_codes = df["ts_code"].unique()
    name_map = _fetch_stock_names_batch(unique_codes)

    records = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        ts = row["ts_code"]
        gross_pct = row["gross_ret"] * 100
        net_pct = row["net_ret"] * 100
        if net_pct > 0:
            label = "盈利"
        elif net_pct < 0:
            label = "亏损"
        else:
            label = "持平"

        reason = row.get("sell_reason", "")
        reason_display = _translate_reason(reason)

        rec = {
            "序号": idx,
            "股票代码": ts,
            "股票名称": name_map.get(ts, ts),
            "买入日期": _fmt_date(row["buy_date"]),
            "卖出日期": _fmt_date(row["sell_date"]),
            "买入价": round(row["buy_price"], 2),
            "卖出价": round(row["sell_price"], 2),
            "持仓天数": int(row["hold_days"]),
            "毛盈亏%": f"{gross_pct:+.2f}%",
            "净盈亏%": f"{net_pct:+.2f}%",
            "盈亏标签": label,
            "退出原因": reason_display,
            "入场评分": round(row.get("score", 0), 4),
        }
        if has_weight:
            rec["权重"] = f"{row['weight']*100:.1f}%"
        records.append(rec)

    report_df = pd.DataFrame(records)

    n = len(df)
    n_win = (df["net_ret"] > 0).sum()
    n_loss = (df["net_ret"] < 0).sum()
    win_rate = n_win / n if n > 0 else 0.0
    avg_gross_pct = df["gross_ret"].mean() * 100
    avg_net_pct = df["net_ret"].mean() * 100

    win_df = df[df["net_ret"] > 0]
    loss_df = df[df["net_ret"] < 0]
    avg_win_pct = win_df["net_ret"].mean() * 100 if len(win_df) > 0 else 0.0
    avg_loss_pct = loss_df["net_ret"].mean() * 100 if len(loss_df) > 0 else 0.0
    wl_ratio = abs(avg_win_pct / avg_loss_pct) if abs(avg_loss_pct) > 1e-6 else 0.0

    max_win_pct = df["net_ret"].max() * 100
    max_loss_pct = df["net_ret"].min() * 100
    avg_hold = df["hold_days"].mean()

    reason_counts = df["sell_reason"].value_counts().to_dict() if "sell_reason" in df.columns else {}

    summary = {
        "总交易数": n,
        "盈利次数": int(n_win),
        "亏损次数": int(n_loss),
        "胜率": f"{win_rate:.1%}",
        "平均毛盈亏%": f"{avg_gross_pct:+.2f}%",
        "平均净盈亏%": f"{avg_net_pct:+.2f}%",
        "盈利平均%": f"{avg_win_pct:+.2f}%",
        "亏损平均%": f"{avg_loss_pct:+.2f}%",
        "盈亏比": f"{wl_ratio:.2f}",
        "最大单笔盈利%": f"{max_win_pct:+.2f}%",
        "最大单笔亏损%": f"{max_loss_pct:+.2f}%",
        "平均持仓天数": f"{avg_hold:.1f}",
        "model_risk退出": reason_counts.get("model_risk", 0),
        "stop_loss退出": reason_counts.get("stop_loss", 0),
        "max_hold退出": reason_counts.get("max_hold", 0),
    }

    return report_df, summary


def save_trade_report(trades_df, output_csv, output_summary_csv=None, label="基线"):
    """
    保存交易报告 CSV

    Args:
        trades_df: 交易明细
        output_csv: 明细输出路径
        output_summary_csv: 汇总输出路径 (可选)
        label: 标签（用于打印）
    """
    report_df, summary = build_trade_report(trades_df)
    report_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if output_summary_csv:
        pd.DataFrame([summary]).to_csv(output_summary_csv, index=False, encoding="utf-8-sig")
    print(f"  [{label}] 交易报告: {len(report_df)}笔, 胜率={summary.get('胜率', 'N/A')}")
    return report_df, summary


def _translate_reason(reason):
    """退出原因翻译"""
    mapping = {
        "model_risk": "模型风险退出",
        "stop_loss": "止损退出",
        "max_hold": "最大持有到期",
        "take_profit": "止盈退出",
        "fixed": "固定持有到期",
    }
    return mapping.get(reason, reason)


def _fmt_date(val):
    """日期格式化"""
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    elif isinstance(val, str) and " " in val:
        return val.split(" ")[0]
    return str(val)[:10]


if __name__ == "__main__":
    from stop_experiment.pipeline.stop_config import BACKTEST_DIR, PRODUCTION_PARAMS

    p = PRODUCTION_PARAMS
    print(f"冒烟测试: trade_report ({p['profile']})")

    trades_path = os.path.join(BACKTEST_DIR, f"{p['profile']}_trades.csv")
    if not os.path.exists(trades_path):
        print(f"  跳过: 无 trades 文件 {trades_path} (需先运行 run_baseline.py)")
        sys.exit(0)

    trades_df = pd.read_csv(trades_path)
    report_df, summary = build_trade_report(trades_df)

    print(f"  交易笔数: {len(report_df)}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = os.path.join(BACKTEST_DIR, f"{p['profile']}_trade_report.csv")
    report_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  输出: {out}")