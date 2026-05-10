#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实盘交易盈亏报告生成器 — 从模拟盘执行账本重建交易明细

Purpose:
    从 live 的 executions + decisions + holdings 账本提取每笔买入→卖出配对，
    计算 P&L、关联中文股票名称，输出格式与回测 trade_report.csv 一致。

Inputs:
    - stop_experiment/output/live/executions/*.parquet (执行账本: buy/sell)
    - stop_experiment/output/live/decisions/*.parquet (决策账本: sell_reason)
    - stop_experiment/output/holdings/*.parquet (持仓账本: entry_score)
    - price_pivot: MultiIndex DataFrame (close, 用于验证)

Outputs:
    - stop_experiment/output/live/live_trade_report.csv
      columns: 序号, 股票代码, 股票名称, 买入日期, 卖出日期,
               买入价, 卖出价, 持仓天数, 毛盈亏%, 净盈亏%,
               盈亏标签, 退出原因, 入场评分

How to Run:
    # 单独运行（需要先有 executions/holdings/decisions 数据）
    python stop_experiment/pipeline/build_live_trade_report.py

    # 从其他脚本导入调用
    from stop_experiment.pipeline.build_live_trade_report import build_live_trade_report

Side Effects:
    - 只读 executions/decisions/holdings/price_pivot，不写数据库
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from glob import glob

from stop_experiment.pipeline.stop_config import EXECUTIONS_DIR, DECISIONS_DIR, HOLDINGS_DIR, OUTPUT_DIR

LIVE_DIR = os.path.join(OUTPUT_DIR, "live")
BUY_COST = 0.0005
SELL_COST = 0.0010


def _extract_code(ts_code):
    """从 ts_code (如 002361.SZ) 提取纯代码 (002361)"""
    if isinstance(ts_code, str) and "." in ts_code:
        return ts_code.split(".")[0]
    return ts_code


def _fetch_stock_names_batch(ts_codes):
    """批量查询 ts_code -> 股票中文名称，从 stock_pools 表"""
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


def _fmt_date(val):
    """日期格式化"""
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    elif isinstance(val, str) and " " in val:
        return val.split(" ")[0]
    return str(val)[:10]


_TRANSLATE = {
    "model_risk": "模型风险退出",
    "stop_loss": "止损退出",
    "max_hold": "最大持有到期",
}


def _translate_reason(reason):
    return _TRANSLATE.get(reason, reason)


def _load_all_live_data():
    """加载所有 executions + decisions + holdings 数据"""
    # Executions (可能有空文件)
    exec_files = sorted(glob(os.path.join(EXECUTIONS_DIR, "*.parquet")))
    exec_dfs = []
    for f in exec_files:
        df = pd.read_parquet(f)
        if df.empty or "action" not in df.columns:
            continue
        exec_dfs.append(df)
    exec_df = pd.concat(exec_dfs, ignore_index=True) if exec_dfs else pd.DataFrame()
    if not exec_df.empty:
        exec_df["execution_date"] = pd.to_datetime(exec_df["execution_date"])
        exec_df["code"] = exec_df["ts_code"].apply(_extract_code)

    # Decisions (卖出原因)
    dec_files = sorted(glob(os.path.join(DECISIONS_DIR, "*.parquet")))
    dec_dfs = []
    for f in dec_files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        dec_dfs.append(df)
    dec_df = pd.concat(dec_dfs, ignore_index=True) if dec_dfs else pd.DataFrame()
    if not dec_df.empty:
        dec_df["decision_date"] = pd.to_datetime(dec_df["decision_date"])

    # Holdings (入场评分)
    hold_files = sorted(glob(os.path.join(HOLDINGS_DIR, "*.parquet")))
    hold_dfs = []
    for f in hold_files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        hold_dfs.append(df)
    hold_df = pd.concat(hold_dfs, ignore_index=True) if hold_dfs else pd.DataFrame()

    return exec_df, dec_df, hold_df


def build_live_trade_report(buy_cost=BUY_COST, sell_cost=SELL_COST):
    """
    从 live 账本重建交易盈亏报告。

    Returns:
        (report_df, summary_dict): 交易明细和汇总
    """
    exec_df, dec_df, hold_df = _load_all_live_data()

    if exec_df.empty:
        print("  [交易报告] 无执行记录，返回空")
        return pd.DataFrame(), {"总交易数": 0}

    # 分离买卖，按日期排序
    buys = exec_df[exec_df["action"] == "buy"].sort_values(["execution_date", "ts_code"]).copy()
    sells = exec_df[exec_df["action"] == "sell"].sort_values(["execution_date", "ts_code"]).copy()

    if sells.empty:
        print("  [交易报告] 无卖出记录，无已完成交易")
        return pd.DataFrame(), {"总交易数": 0}

    # 构建决策查找表: {ts_code: {decision_date: reason}}
    sell_reason_map = {}
    all_decision_dates = set()
    if not dec_df.empty:
        sells_in_dec = dec_df[dec_df["action"] == "sell"]
        for _, r in sells_in_dec.iterrows():
            ts = r["ts_code"]
            dd = r["decision_date"]
            sell_reason_map.setdefault(ts, {})[dd] = r.get("reason", "unknown")
            all_decision_dates.add(dd)

    sorted_decision_dates = sorted(all_decision_dates)

    # 构建持仓入场评分查找表: {ts_code: {entry_date: score}}
    score_map = {}
    if not hold_df.empty:
        for _, r in hold_df.iterrows():
            ts = r.get("ts_code", "")
            ed = r.get("entry_date")
            if ed is not None:
                score_map.setdefault(ts, {})[pd.Timestamp(ed)] = r.get("score", 0)

    # 匹配买入→卖出: 对每笔卖出找到最早的未匹配买入
    matched_buy_indices = set()
    trades = []

    for _, sell in sells.iterrows():
        code = sell["code"]
        ts = sell["ts_code"]
        sell_date = sell["execution_date"]
        sell_price = sell["executed_price"]
        if pd.isna(sell_price) or sell_price <= 0:
            continue

        # 找到该代码最早的未匹配买入
        code_buys = buys[(buys["code"] == code) & (~buys.index.isin(matched_buy_indices))]
        if code_buys.empty:
            continue

        matched_buy = code_buys.iloc[0]
        matched_buy_indices.add(matched_buy.name)

        buy_date = matched_buy["execution_date"]
        buy_price = matched_buy["executed_price"]
        if pd.isna(buy_price) or buy_price <= 0:
            continue

        hold_days = (sell_date - buy_date).days
        gross_ret = (sell_price - buy_price) / buy_price
        net_ret = (sell_price * (1.0 - sell_cost) - buy_price * (1.0 + buy_cost)) / (buy_price * (1.0 + buy_cost))

        # 卖出原因（execution_date 在 T+1，decision_date 在 T，用交易日历回溯）
        reason = sell_reason_map.get(ts, {}).get(sell_date, None)
        if reason is None and sorted_decision_dates:
            for prev_dd in reversed(sorted_decision_dates):
                if prev_dd < sell_date:
                    reason = sell_reason_map.get(ts, {}).get(prev_dd, None)
                    if reason is not None:
                        break
        if reason is None:
            reason = "unknown"

        # 入场评分
        score = score_map.get(ts, {}).get(buy_date, 0)

        trades.append({
            "ts_code": ts,
            "buy_date": buy_date,
            "buy_price": float(buy_price),
            "sell_date": sell_date,
            "sell_price": float(sell_price),
            "hold_days": int(hold_days),
            "gross_ret": float(gross_ret),
            "net_ret": float(net_ret),
            "sell_reason": reason,
            "score": float(score) if score else 0.0,
        })

    if not trades:
        print("  [交易报告] 无匹配的买入卖出对")
        return pd.DataFrame(), {"总交易数": 0}

    df = pd.DataFrame(trades)

    # 查中文名称
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

        records.append({
            "序号": idx,
            "股票代码": ts,
            "股票名称": name_map.get(ts, ts),
            "买入日期": _fmt_date(row["buy_date"]),
            "卖出日期": _fmt_date(row["sell_date"]),
            "买入价": round(row["buy_price"], 2),
            "卖出价": round(row["sell_price"], 2),
            "持仓天数": row["hold_days"],
            "毛盈亏%": f"{gross_pct:+.2f}%",
            "净盈亏%": f"{net_pct:+.2f}%",
            "盈亏标签": label,
            "退出原因": _translate_reason(row["sell_reason"]),
            "入场评分": round(row["score"], 4),
        })

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

    reason_counts = df["sell_reason"].value_counts().to_dict()

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


def save_live_trade_report(report_df, summary, output_path=None):
    """保存交易报告到 live/live_trade_report.csv"""
    if output_path is None:
        os.makedirs(LIVE_DIR, exist_ok=True)
        output_path = os.path.join(LIVE_DIR, "live_trade_report.csv")
    if report_df.empty:
        print("  [交易报告] 无数据，跳过保存")
        return report_df, summary
    report_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  [交易报告] 已保存 {output_path} ({len(report_df)} 笔)")
    print(f"    胜率={summary.get('胜率', 'N/A')}, "
          f"平均净盈亏={summary.get('平均净盈亏%', 'N/A')}")
    return report_df, summary


# ==================== 自测入口 ====================
if __name__ == "__main__":
    print("冒烟测试: build_live_trade_report")
    report_df, summary = build_live_trade_report()
    if report_df.empty:
        print("  无已完成交易")
    else:
        print(f"  交易笔数: {len(report_df)}")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        save_live_trade_report(report_df, summary)