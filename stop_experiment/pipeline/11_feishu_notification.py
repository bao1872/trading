#!/usr/bin/env python3
"""
飞书消息通知 — SLC 策略证据面板

用途：读取 4 本账 → 计算 Tier/Top30 追踪 → 读取净值曲线 → 生成飞书 Markdown → 发送
消息包含：净值/日收益/回撤 | 持仓分组(风险/安全/待更新) | 候选高优先级 | 新买入清单

用法：
  python -m stop_experiment.pipeline.11_feishu_notification --date 2026-05-08 --dry-run   # 本地预览
  python -m stop_experiment.pipeline.11_feishu_notification --date 2026-05-08              # 正式发送
  python -m stop_experiment.pipeline.11_feishu_notification --latest --dry-run              # 最新日期预览

输入：
  - live/decisions/T.parquet (决策账本)
  - live/holdings/T.parquet (持仓账本)
  - predictions/T.parquet (预测账本)
  - live/live_equity_curve.csv (净值曲线，读取最新 NAV)

输出：飞书卡片消息 (Markdown 格式)
副作用：发送飞书消息 (非 dry-run 模式)
"""

import sys
import os
import argparse
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np

from stop_experiment.pipeline.stop_config import (
    DECISIONS_DIR, EXECUTIONS_DIR, HOLDINGS_DIR, PREDICTIONS_DIR, OUTPUT_DIR,
    BASELINE_E0_X1_V1_PARAMS,
)

TIER_A_SCORE = 0.2
TIER_A_BUY_CLS = 0.30
TIER_B_SCORE = 0.1
TIER_B_BUY_CLS = 0.70

EXIT_THRESHOLD = BASELINE_E0_X1_V1_PARAMS.get("buy_cls_exit_threshold", 0.70)
STOP_LOSS = BASELINE_E0_X1_V1_PARAMS.get("stop_loss", -0.07)
MAX_HOLD_DAYS = BASELINE_E0_X1_V1_PARAMS.get("max_hold_days", 20)
LIVE_EQUITY_PATH = os.path.join(OUTPUT_DIR, "live", "live_equity_curve.csv")


def _get_latest_nav():
    """从 live_equity_curve.csv 读取最新净值/日收益/回撤，无文件返回 (None, None, None)"""
    if not os.path.exists(LIVE_EQUITY_PATH):
        return None, None, None
    try:
        eq = pd.read_csv(LIVE_EQUITY_PATH)
        if eq.empty:
            return None, None, None
        latest = eq.iloc[-1]
        return latest.get("nav"), latest.get("daily_return"), latest.get("drawdown")
    except Exception:
        return None, None, None


def _safe_read_parquet(dir_path, date_str, label):
    path = os.path.join(dir_path, f"{date_str}.parquet")
    if not os.path.exists(path):
        print(f"  [信息] {label} 不存在: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if len(df.columns) == 0:
        return pd.DataFrame()
    return df


def _fmt_float(v, default="N/A"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return f"{v:.4f}"


def _fetch_stock_names_batch(ts_codes):
    """批量查询 ts_code → 股票中文名称，从 stock_pools 表"""
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


def _fmt_pct(v, default="N/A"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return f"{v*100:+.2f}%"


def _fmt_date(d):
    if d is None:
        return "?"
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


def _compute_tier(score, buy_cls):
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "?"
    if buy_cls is None or (isinstance(buy_cls, float) and np.isnan(buy_cls)):
        return "?"
    if score >= TIER_A_SCORE and buy_cls < TIER_A_BUY_CLS:
        return "A"
    if buy_cls >= TIER_B_BUY_CLS:
        return "C"
    if score >= TIER_B_SCORE and buy_cls < TIER_B_BUY_CLS:
        return "B"
    return "B"


def _compute_top30_tracking(pred_df):
    """
    对 predictions 按 obs_day 分组求 Top30/Top10，
    统计每只股票出现次数和最优排名。
    返回 (top30_map, top10_map): {ts_code: {"appearances": N, "best_rank": R}}
    """
    if pred_df.empty or "obs_day" not in pred_df.columns or "score" not in pred_df.columns:
        return {}, {}

    top30_map = {}
    top10_map = {}

    for obs_day, grp in pred_df.groupby("obs_day"):
        grp_sorted = grp.sort_values("score", ascending=False)
        top30_codes = grp_sorted["ts_code"].head(30).tolist()
        top10_codes = grp_sorted["ts_code"].head(10).tolist()

        for rank, code in enumerate(top30_codes):
            if code not in top30_map:
                top30_map[code] = {"appearances": 0, "best_rank": 999}
            top30_map[code]["appearances"] += 1
            top30_map[code]["best_rank"] = min(top30_map[code]["best_rank"], rank + 1)

        for rank, code in enumerate(top10_codes):
            if code not in top10_map:
                top10_map[code] = {"appearances": 0, "best_rank": 999}
            top10_map[code]["appearances"] += 1
            top10_map[code]["best_rank"] = min(top10_map[code]["best_rank"], rank + 1)

    return top30_map, top10_map


def _build_predictions_lookup(pred_df):
    """构建 {signal_id: {pred_buy_cls, pred_sell_reg, score}} 查找表 (取最新 obs_day 的预测)。"""
    if pred_df.empty:
        return {}
    lookup = {}
    for _, row in pred_df.sort_values("obs_day", ascending=False).iterrows():
        sid = row["signal_id"]
        if sid not in lookup:
            lookup[sid] = {
                "pred_buy_cls": row.get("pred_buy_cls"),
                "pred_sell_reg": row.get("pred_sell_reg"),
                "score": row.get("score"),
            }
    return lookup


def generate_feishu_markdown(date_str):
    """
    读取 4 本账，生成飞书 Markdown 消息。
    返回 (markdown_text, summary_dict)。
    """
    dec_df = _safe_read_parquet(DECISIONS_DIR, date_str, "决策账本")
    hold_df = _safe_read_parquet(HOLDINGS_DIR, date_str, "持仓账本")
    pred_df = _safe_read_parquet(PREDICTIONS_DIR, date_str, "预测账本")

    pred_lookup = _build_predictions_lookup(pred_df)
    top30_map, top10_map = _compute_top30_tracking(pred_df)

    # 构建候选池 Top10 (从 predictions 去重取 top 10)
    candidates_top10 = pd.DataFrame()
    if not pred_df.empty and "score" in pred_df.columns:
        cand = pred_df.sort_values("score", ascending=False)
        cand = cand.drop_duplicates(subset=["ts_code"], keep="first")
        candidates_top10 = cand.head(10).copy()

    # 持股数据：优先从 decisions (action=hold)，备选从 holdings
    holdings_data = []
    risk_triggered = []
    safe_holdings = []
    pending_update = []

    if not dec_df.empty:
        holds_in_dec = dec_df[dec_df["action"] == "hold"]
        sells_in_dec = dec_df[dec_df["action"] == "sell"]
        sold_codes = set(sells_in_dec["ts_code"].tolist()) if not sells_in_dec.empty else set()

        for _, r in holds_in_dec.iterrows():
            ts = str(r.get("ts_code", ""))
            sid = r.get("signal_id")
            entry_score = r.get("score", 0)
            cur_ret = r.get("cur_ret")
            days_held = r.get("days_held", 0)
            pred = pred_lookup.get(sid, {})
            buy_cls = pred.get("pred_buy_cls")
            tier = _compute_tier(entry_score, buy_cls)
            t30 = top30_map.get(ts, {"appearances": 0, "best_rank": "-"})
            t10 = top10_map.get(ts, {"appearances": 0, "best_rank": "-"})

            entry = {
                "ts_code": ts,
                "signal_id": sid,
                "entry_score": entry_score,
                "cur_ret": cur_ret,
                "days_held": days_held,
                "buy_cls": buy_cls,
                "tier": tier,
                "top30_appearances": t30["appearances"],
                "top30_best_rank": t30["best_rank"],
                "top10_appearances": t10["appearances"],
                "top10_best_rank": t10["best_rank"],
            }
            holdings_data.append(entry)

            if buy_cls is not None and not (isinstance(buy_cls, float) and np.isnan(buy_cls)):
                if buy_cls >= EXIT_THRESHOLD:
                    risk_triggered.append(entry)
                else:
                    safe_holdings.append(entry)
            else:
                pending_update.append(entry)

        # 已触发卖出的(含 stop_loss/max_hold) 放入风险触发
        for _, r in sells_in_dec.iterrows():
            ts = str(r.get("ts_code", ""))
            sid = r.get("signal_id")
            entry_score = r.get("score", 0)
            cur_ret = r.get("cur_ret")
            days_held = r.get("days_held", 0)
            reason = r.get("reason", "unknown")
            why = r.get("why", "")
            pred = pred_lookup.get(sid, {})
            buy_cls = pred.get("pred_buy_cls")
            tier = _compute_tier(entry_score, buy_cls)
            t30 = top30_map.get(ts, {"appearances": 0, "best_rank": "-"})
            t10 = top10_map.get(ts, {"appearances": 0, "best_rank": "-"})

            entry = {
                "ts_code": ts,
                "signal_id": sid,
                "entry_score": entry_score,
                "cur_ret": cur_ret,
                "days_held": days_held,
                "buy_cls": buy_cls,
                "tier": tier,
                "top30_appearances": t30["appearances"],
                "top30_best_rank": t30["best_rank"],
                "top10_appearances": t10["appearances"],
                "top10_best_rank": t10["best_rank"],
                "reason": reason,
                "why": why,
            }
            risk_triggered.append(entry)

    # 买入决策 (新买入清单)
    buy_list = []
    if not dec_df.empty:
        buys = dec_df[dec_df["action"] == "buy"]
        for _, r in buys.iterrows():
            ts = str(r.get("ts_code", ""))
            sc = r.get("score", 0)
            sid = r.get("signal_id")
            pred = pred_lookup.get(sid, {})
            buy_cls = pred.get("pred_buy_cls")
            tier = _compute_tier(sc, buy_cls)
            t30 = top30_map.get(ts, {"appearances": 0, "best_rank": "-"})
            t10 = top10_map.get(ts, {"appearances": 0, "best_rank": "-"})

            buy_list.append({
                "ts_code": ts,
                "score_val": sc,
                "buy_cls": buy_cls,
                "tier": tier,
                "top30_appearances": t30["appearances"],
                "top30_best_rank": t30["best_rank"],
                "top10_appearances": t10["appearances"],
                "top10_best_rank": t10["best_rank"],
            })

    n_hold = len(holdings_data)
    n_risk = len(risk_triggered)
    n_safe = len(safe_holdings)
    n_pending = len(pending_update)
    n_buy = len(buy_list)
    n_sell = len(dec_df[dec_df["action"] == "sell"]) if not dec_df.empty else 0

    # Top30 池计数 (有多少只不同的股票至少出现在一次 Top30 中)
    n_top30_pool = len(top30_map)

    # 构建股票名称映射
    all_ts = set()
    for d in risk_triggered + safe_holdings + pending_update + buy_list:
        all_ts.add(d["ts_code"])
    if not candidates_top10.empty:
        for ts in candidates_top10["ts_code"].tolist():
            all_ts.add(str(ts))
    name_map = _fetch_stock_names_batch(all_ts)

    def stock_display(ts):
        return name_map.get(ts, ts.split(".")[0] if "." in ts else ts)

    # ============ 构建 Markdown ============
    lines = []

    # Header
    lines.append(f" SLC 策略证据面板 | {date_str} ")
    lines.append("")

    # 净值行（从 live_equity_curve.csv 读取）
    latest_nav, daily_ret, dd = _get_latest_nav()
    if latest_nav is not None:
        nav_str = f"净值 {latest_nav:.4f}"
        if daily_ret is not None and not np.isnan(daily_ret):
            nav_str += f"  |  日收益 {daily_ret:+.2%}"
        if dd is not None and not np.isnan(dd):
            nav_str += f"  |  回撤 {dd:.2%}"
        lines.append(f"  {nav_str}")
        lines.append("")

    # Summary line
    summary_parts = [f"持仓{n_hold}"]
    if n_sell > 0:
        summary_parts.append(f"卖出{n_sell}")
    summary_parts.append(f"风险触发{n_risk}")
    summary_parts.append(f"Top30池{n_top30_pool}")
    summary_parts.append(f"Top10候选{len(candidates_top10)}")
    lines.append("  " + "  |  ".join(summary_parts))
    lines.append("")

    # Config line
    lines.append(f" 止损{STOP_LOSS*100:.0f}%  |  最大持有{MAX_HOLD_DAYS}天  |  退出阈值 pred_buy_cls > {EXIT_THRESHOLD:.2f}")
    lines.append("")

    # ============ 一、持仓-风险触发 ============
    if risk_triggered or n_sell > 0:
        lines.append(f"🔴  一、持仓-风险触发  ({len(risk_triggered)}只) ")
        lines.append("")
        for i, e in enumerate(risk_triggered):
            ts = e["ts_code"]
            lines.append(f" {i+1}) {stock_display(ts)}  `{ts}` ")
            lines.append(f" 买入 {_fmt_date(e.get('entry_date'))}  |  买入价 {e.get('entry_price', '?')}  |  持仓 {e['days_held']}天  |  entry_score {_fmt_float(e['entry_score'])}")
            t30_str = f"是({e['top30_appearances']}次)" if e['top30_appearances'] > 0 else f"否(0次)"
            t10_str = f"是(排{e['top10_best_rank']})" if e['top10_appearances'] > 0 else f"否(排{e['top10_best_rank']})"
            lines.append(f" Top30: {t30_str}  |  Top10: {t10_str}")
            lines.append(f" score {_fmt_float(e['entry_score'])}  |  buy_cls {_fmt_float(e['buy_cls'])}  |  cur_ret {_fmt_pct(e['cur_ret'])}")
            if e.get("why"):
                lines.append(f" > 依据: {e['why']}")
            else:
                lines.append(f" > 依据: 模型风险")
        lines.append("")

    # ============ 二、持仓-安全 ============
    if safe_holdings:
        lines.append(f"🟢  二、持仓-安全  ({len(safe_holdings)}只) ")
        lines.append("")
        for i, e in enumerate(safe_holdings):
            ts = e["ts_code"]
            lines.append(f" {i+1}) {stock_display(ts)}  `{ts}` ")
            lines.append(f" 买入 {_fmt_date(e.get('entry_date'))}  |  买入价 {e.get('entry_price', '?')}  |  持仓 {e['days_held']}天  |  entry_score {_fmt_float(e['entry_score'])}")
            t30_str = f"是({e['top30_appearances']}次)" if e['top30_appearances'] > 0 else f"否(0次)"
            t10_str = f"是(排{e['top10_best_rank']})" if e['top10_appearances'] > 0 else f"否(排0)"
            lines.append(f" Top30: {t30_str}  |  Top10: {t10_str}")
            lines.append(f" score {_fmt_float(e['entry_score'])}  |  buy_cls {_fmt_float(e['buy_cls'])}  |  cur_ret {_fmt_pct(e['cur_ret'])}")
        lines.append("")

    # ============ 三、持仓-待数据更新 ============
    if pending_update:
        lines.append(f"⚠️  三、持仓-待数据更新  ({len(pending_update)}只) ")
        lines.append("")
        for i, e in enumerate(pending_update):
            ts = e["ts_code"]
            lines.append(f" {i+1}) {stock_display(ts)}  `{ts}` ")
            lines.append(f" 买入 {_fmt_date(e.get('entry_date'))}  |  买入价 {e.get('entry_price', '?')}  |  持仓 {e['days_held']}天  |  entry_score {_fmt_float(e['entry_score'])}")
            t30_str = f"是({e['top30_appearances']}次)" if e['top30_appearances'] > 0 else f"否(0次)"
            t10_str = f"是(排{e['top10_best_rank']})" if e['top10_appearances'] > 0 else f"否(排0)"
            lines.append(f" Top30: {t30_str}  |  Top10: {t10_str}")
            lines.append(f" score {_fmt_float(e['entry_score'])}  |  buy_cls N/A  |  cur_ret {_fmt_pct(e['cur_ret'])}")
            lines.append(f" > ⚠️ pred_buy_cls 缺失，需关注")
        lines.append("")

    # ============ 四、候选-高优先级 ============
    if not candidates_top10.empty:
        lines.append(f"📥  四、候选-高优先级  ({len(candidates_top10)}只) ")
        lines.append("")
        for i, (_, r) in enumerate(candidates_top10.iterrows()):
            ts = r.get("ts_code", "")
            sc = r.get("score", 0)
            bcls = r.get("pred_buy_cls")
            od = int(r.get("obs_day", 0))
            sid = r.get("signal_id")
            tier = _compute_tier(sc, bcls)
            t30 = top30_map.get(ts, {"appearances": 0, "best_rank": "-"})
            t30_str = f"出现 {t30['appearances']} 次, 最优排{t30['best_rank']}" if t30['appearances'] > 0 else ""
            lines.append(f" {i+1}) {stock_display(ts)}  `{ts}`  `{tier}` ")
            lines.append(f" score {_fmt_float(sc)}  |  buy_cls {_fmt_float(bcls)}  |  obs_day {od}")
            if t30_str:
                lines.append(f" Top30: {t30_str}")
        lines.append("")

    # ============ 五、新买入清单 ============
    if buy_list:
        max_stocks = BASELINE_E0_X1_V1_PARAMS.get("max_stocks", 10)
        avail = max_stocks - n_hold
        lines.append(f"📤 新买入清单  ({len(buy_list)}只) ")
        lines.append(f" 可用仓位: {avail}/{max_stocks}")
        for i, e in enumerate(buy_list):
            ts = e["ts_code"]
            lines.append(f" {i+1}) {stock_display(ts)} `{ts}` `{e['tier']}`  score {_fmt_float(e['score_val'])}")
        lines.append("")

    if not any([risk_triggered, safe_holdings, pending_update, not candidates_top10.empty, buy_list]):
        lines.append("> ℹ️ 当日无持仓无候选，系统初始化中")
        lines.append("")

    md_text = "\n".join(lines)

    summary = {
        "date": date_str,
        "holdings": n_hold,
        "risk_triggered": n_risk,
        "safe": n_safe,
        "pending_update": n_pending,
        "sell_decisions": n_sell,
        "buy_decisions": n_buy,
        "top30_pool": n_top30_pool,
        "top10_candidates": len(candidates_top10),
    }

    return md_text, summary


def send_feishu_notification(date_str, dry_run=False):
    """生成 Markdown → 调用 FeishuNotifier.send_card() 发送。"""
    print(f"\n{'='*60}")
    print(f"  📨 飞书通知 — {date_str}")
    print(f"{'='*60}")

    md_text, summary = generate_feishu_markdown(date_str)

    print(f"")
    print(md_text)
    print(f"")
    print(f"  摘要: 持仓{summary['holdings']} 卖出{summary['sell_decisions']} "
          f"风险{summary['risk_triggered']} 候选{summary['top10_candidates']}")

    if dry_run:
        print(f"\n  [dry-run] 不实际发送飞书消息")
        return True, "dry-run 完成"

    try:
        from app.feishu_notifier import FeishuNotifier
        notifier = FeishuNotifier()
        notifier.send_markdown(md_text)
        print(f"\n  ✅ 飞书消息已发送")
        return True, "发送成功"
    except Exception as e:
        print(f"\n  ❌ 飞书发送失败: {e}")
        return False, str(e)


def _find_latest_decisions_date():
    if not os.path.isdir(DECISIONS_DIR):
        return None
    dates = []
    for f in os.listdir(DECISIONS_DIR):
        if f.endswith(".parquet") and len(f) >= 10:
            ds = f.replace(".parquet", "")
            if len(ds) == 10 and ds[4] == "-":
                dates.append(ds)
    if not dates:
        return None
    return sorted(dates)[-1]


def main():
    parser = argparse.ArgumentParser(description="飞书消息通知 — SLC 策略证据面板")
    parser.add_argument("--date", type=str, default=None, help="指定日期 YYYY-MM-DD")
    parser.add_argument("--latest", action="store_true", help="自动找最新有 decisions 的日期")
    parser.add_argument("--dry-run", action="store_true", help="只打印 Markdown 不发送")
    args = parser.parse_args()

    date_str = args.date
    if args.latest or date_str is None:
        date_str = _find_latest_decisions_date()
        if date_str is None:
            print("错误: 找不到任何 decisions 文件，请先运行 09_paper_trading_runner.py")
            sys.exit(1)
        print(f"自动选择最新日期: {date_str}")

    ok, msg = send_feishu_notification(date_str, dry_run=args.dry_run)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()