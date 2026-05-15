#!/usr/bin/env python3
"""
每日持仓+四模型打分+买卖决策审计报告生成器

用途：
    从回测账本（holdings/decisions/predictions）合并生成两个审计 CSV：
    1. daily_holdings_report.csv — 每日持仓+四模型打分
    2. daily_trade_audit.csv — 买卖决策审计

用法：
    python -m stop_experiment.pipeline.build_daily_holdings_report
    python -m stop_experiment.pipeline.build_daily_holdings_report --start-date 2026-01-06 --end-date 2026-03-01

参数：
    --start-date  起始日期 YYYY-MM-DD（默认取全部）
    --end-date    结束日期 YYYY-MM-DD（默认取全部）

输入：
    - output/backtest_ledger/holdings/*.parquet
    - output/backtest_ledger/decisions/*.parquet
    - output/predictions/*.parquet

输出：
    - output/backtest_ledger/daily_holdings_report.csv
    - output/backtest_ledger/daily_trade_audit.csv

副作用：只读 parquet，写 CSV（幂等）
"""

import sys
import os
import argparse
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np

from stop_experiment.pipeline.stop_config import OUTPUT_DIR, PREDICTIONS_DIR

LIVE_DIR = os.path.join(OUTPUT_DIR, "backtest_ledger")
HOLDINGS_DIR = os.path.join(LIVE_DIR, "holdings")
DECISIONS_DIR = os.path.join(LIVE_DIR, "decisions")

PRED_COLS = ["pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls"]


def _load_all_parquets(directory, date_col="date"):
    """加载目录下所有 parquet 并合并为一个 DataFrame。"""
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    if date_col in result.columns:
        result[date_col] = pd.to_datetime(result[date_col])
    return result


def _build_pred_lookup(start_date=None, end_date=None):
    """从 full_test_predictions.parquet 构建预测查找表。

    优先使用 full_test_predictions（包含所有股票所有日期的预测），
    fallback 到 predictions/ 目录（仅包含当天有新信号的股票）。
    同一股票同一天多条记录时，取 obs_day 最小的（最新信号）。
    """
    ftp_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    if os.path.exists(ftp_path):
        pred_all = pd.read_parquet(ftp_path)
    else:
        files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*.parquet")))
        if not files:
            return pd.DataFrame()
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            if not df.empty:
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        pred_all = pd.concat(dfs, ignore_index=True)

    if pred_all.empty:
        return pd.DataFrame()

    if "obs_date" in pred_all.columns:
        pred_all["obs_date"] = pd.to_datetime(pred_all["obs_date"])

    if start_date:
        pred_all = pred_all[pred_all["obs_date"] >= pd.to_datetime(start_date)]
    if end_date:
        pred_all = pred_all[pred_all["obs_date"] <= pd.to_datetime(end_date)]

    # 同一 (ts_code, obs_date) 多条记录时，取 obs_day 最小的
    if "obs_day" in pred_all.columns:
        pred_all = pred_all.sort_values("obs_day").drop_duplicates(
            subset=["ts_code", "obs_date"], keep="first"
        )

    lookup_cols = ["obs_date", "ts_code", "signal_id"] + PRED_COLS
    available = [c for c in lookup_cols if c in pred_all.columns]
    return pred_all[available]


def _merge_predictions(df, pred_lookup, date_col, ts_code_col="ts_code"):
    """将 pred_lookup 中的四模型打分合并到 df。

    优先按 (signal_id, date) 精确匹配，fallback 到 (ts_code, date)。
    """
    if pred_lookup.empty:
        for c in PRED_COLS:
            df[c] = np.nan
        return df

    df = df.copy()
    df["_merge_date"] = pd.to_datetime(df[date_col])

    # 构建 lookup 索引
    pred_lookup = pred_lookup.copy()
    if "obs_date" in pred_lookup.columns:
        pred_lookup["_merge_date"] = pred_lookup["obs_date"]
    else:
        pred_lookup["_merge_date"] = pd.to_datetime(pred_lookup.get("obs_date", pd.NaT))

    # 方式1: (signal_id, date) 精确匹配
    if "signal_id" in df.columns and "signal_id" in pred_lookup.columns:
        pred_idx1 = pred_lookup.set_index(["signal_id", "_merge_date"])[PRED_COLS]
        valid_mask = df["signal_id"].notna()
        if valid_mask.any():
            merge_keys = list(zip(df.loc[valid_mask, "signal_id"], df.loc[valid_mask, "_merge_date"]))
            matched = pred_idx1.reindex(merge_keys)
            for c in PRED_COLS:
                df.loc[valid_mask, c] = matched[c].values

    # 方式2: (ts_code, date) fallback — 仅对仍未匹配的行
    still_nan = df[PRED_COLS[0]].isna()
    if still_nan.any() and ts_code_col in df.columns and ts_code_col in pred_lookup.columns:
        pred_idx2 = pred_lookup.set_index([ts_code_col, "_merge_date"])[PRED_COLS]
        merge_keys = list(zip(df.loc[still_nan, ts_code_col], df.loc[still_nan, "_merge_date"]))
        matched = pred_idx2.reindex(merge_keys)
        for c in PRED_COLS:
            df.loc[still_nan, c] = matched[c].values

    df = df.drop(columns=["_merge_date"])
    return df


def _load_stock_names():
    """从 DB 加载 ts_code → stock_name 映射。"""
    try:
        from datasource.database import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT ts_code, name FROM stock_pools "
                "WHERE name IS NOT NULL "
                "UNION "
                "SELECT ts_code, stock_name AS name FROM stop_loss_selection "
                "WHERE stock_name IS NOT NULL"
            )).fetchall()
        engine.dispose()
        return {r[0]: r[1] for r in result if r[0] and r[1]}
    except Exception:
        return {}


def build_daily_holdings_report(start_date=None, end_date=None):
    """生成 daily_holdings_report.csv：每日持仓+四模型打分。"""
    print("[Step 1] 加载持仓快照...")
    holdings_all = _load_all_parquets(HOLDINGS_DIR, date_col="date")
    if holdings_all.empty:
        print("  无持仓数据")
        return

    if start_date:
        holdings_all = holdings_all[holdings_all["date"] >= pd.to_datetime(start_date)]
    if end_date:
        holdings_all = holdings_all[holdings_all["date"] <= pd.to_datetime(end_date)]

    print(f"  持仓记录: {len(holdings_all)} 行, 日期范围: {holdings_all['date'].min().strftime('%Y-%m-%d')} ~ {holdings_all['date'].max().strftime('%Y-%m-%d')}")

    print("[Step 2] 加载预测数据...")
    pred_lookup = _build_pred_lookup(start_date, end_date)
    print(f"  预测记录: {len(pred_lookup)} 行")

    print("[Step 3] 关联持仓+预测...")
    holdings_all = _merge_predictions(holdings_all, pred_lookup, date_col="date")

    print("[Step 4] 加载股票名称...")
    name_map = _load_stock_names()
    holdings_all["stock_name"] = holdings_all["ts_code"].map(name_map).fillna("")

    # 计算 cur_ret（当日收益率）
    if "entry_price" in holdings_all.columns:
        # cur_ret 需要从 decisions 中获取，这里先留空
        pass

    # 从 decisions 中获取 cur_ret
    print("[Step 5] 补充 cur_ret...")
    decisions_all = _load_all_parquets(DECISIONS_DIR, date_col="decision_date")
    if not decisions_all.empty and "cur_ret" in decisions_all.columns:
        hold_decisions = decisions_all[decisions_all["action"].isin(["hold", "sell"])].copy()
        if not hold_decisions.empty:
            ret_map = {}
            for _, row in hold_decisions.iterrows():
                key = (row.get("ts_code"), pd.Timestamp(row.get("decision_date")))
                if pd.notna(row.get("cur_ret")):
                    ret_map[key] = row["cur_ret"]
            holdings_all["cur_ret"] = holdings_all.apply(
                lambda r: ret_map.get((r.get("ts_code"), pd.Timestamp(r.get("date")))), axis=1
            )

    # 排列列顺序
    output_cols = [
        "date", "ts_code", "stock_name", "entry_date", "entry_price",
        "days_held", "score", "cur_ret",
        "pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls",
    ]
    available_cols = [c for c in output_cols if c in holdings_all.columns]
    result = holdings_all[available_cols].sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)

    out_path = os.path.join(LIVE_DIR, "daily_holdings_report.csv")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ {out_path} ({len(result)} 行)")

    return result


def build_daily_trade_audit(start_date=None, end_date=None):
    """生成 daily_trade_audit.csv：买卖决策审计。"""
    print("\n[Step 6] 加载决策账本...")
    decisions_all = _load_all_parquets(DECISIONS_DIR, date_col="decision_date")
    if decisions_all.empty:
        print("  无决策数据")
        return

    if start_date:
        decisions_all = decisions_all[decisions_all["decision_date"] >= pd.to_datetime(start_date)]
    if end_date:
        decisions_all = decisions_all[decisions_all["decision_date"] <= pd.to_datetime(end_date)]

    # 只保留 hold/sell/buy 行（candidate 行已有四模型打分）
    main_actions = decisions_all[decisions_all["action"].isin(["hold", "sell", "buy"])].copy()
    candidate_actions = decisions_all[decisions_all["action"] == "candidate"].copy()

    print(f"  决策记录: {len(main_actions)} 行 (hold/sell/buy), {len(candidate_actions)} 行 (candidate)")

    print("[Step 7] 关联 hold/sell/buy + 预测...")
    pred_lookup = _build_pred_lookup(start_date, end_date)

    # 对 hold/sell/buy 行补充四模型打分
    main_actions = _merge_predictions(main_actions, pred_lookup, date_col="decision_date")

    # 合并回 candidate 行（candidate 已有四模型打分）
    audit = pd.concat([main_actions, candidate_actions], ignore_index=True)

    print("[Step 8] 加载股票名称...")
    name_map = _load_stock_names()
    audit["stock_name"] = audit["ts_code"].map(name_map).fillna("")

    # 排列列顺序
    output_cols = [
        "decision_date", "ts_code", "stock_name", "action", "reason",
        "score", "days_held", "cur_ret", "threshold_value", "why",
        "pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls",
    ]
    extra_cols = ["rank", "signal_id", "is_held", "is_pending_buy", "obs_day", "planned_price", "planned_weight"]
    for c in extra_cols:
        if c in audit.columns:
            output_cols.append(c)

    available_cols = [c for c in output_cols if c in audit.columns]
    result = audit[available_cols].sort_values(["decision_date", "action", "score"],
                                                ascending=[True, True, False]).reset_index(drop=True)

    out_path = os.path.join(LIVE_DIR, "daily_trade_audit.csv")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ {out_path} ({len(result)} 行)")

    return result


def main():
    parser = argparse.ArgumentParser(description="每日持仓+四模型打分+买卖决策审计报告生成器")
    parser.add_argument("--start-date", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", help="结束日期 YYYY-MM-DD")
    args = parser.parse_args()

    build_daily_holdings_report(args.start_date, args.end_date)
    build_daily_trade_audit(args.start_date, args.end_date)

    print("\n✅ 全部完成")


if __name__ == "__main__":
    main()
