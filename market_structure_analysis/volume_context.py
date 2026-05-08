"""
volume_context.py
量能背景层 — 市场级量能广度/风格/背离指标

Purpose:
    基于已有的 lightweight_events（Step1 流式输出）二次聚合，
    计算量能广度、风格量能对比、指数-个股背离指标。
    不重跑因子，纯 pandas 聚合操作。

Inputs:
    - lightweight_events: 含 ts_code/trade_date/evt_* 的轻量表（Step1 输出）
    - daily_agg: 含 total_stocks/evt_*_count 的日级聚合表
    - stock_attrs: 含 industry_l2/cap_tier 的股票属性表（用于风格分组）

Outputs:
    - 量能广度 DataFrame（逐日）
    - 分组量能 DataFrame（逐日）
    - 指数-个股背离 DataFrame（逐日）
    - 成交额结构 DataFrame（逐日）: 金额变化率/放量占比/强势组占比/大小盘占比变化
    - 量能背景解读文本

How to Run:
    python market_structure_analysis/volume_context.py --start 2024-01-01 --end 2026-04-30 --limit_stocks 50

Examples:
    python market_structure_analysis/volume_context.py --limit_stocks 50

Side Effects:
    无（只读，不写入数据库或文件）
"""

import argparse
import logging
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_session, query_df
from market_structure_analysis.event_detector import CORE_EVENTS

logger = logging.getLogger(__name__)

EVT_COLS = [f"evt_{e}" for e in CORE_EVENTS]

VOL_UP_EVT = "evt_up_move_with_vol_spike"
VOL_DOWN_EVT = "evt_down_move_with_vol_spike"
BREAKOUT_BULL_EVT = "evt_break_sell_stop_cluster"
BREAKOUT_BEAR_EVT = "evt_break_buy_stop_cluster"


def compute_volume_breadth(
    daily_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    量能广度指标（逐日）。

    直接从 daily_agg 的事件计数计算，不需要 lightweight_events。
    所有指标均为当日触发该事件的股票数 / 当日总股票数。

    Parameters
    ----------
    daily_agg : pd.DataFrame
        Step1 输出，含 total_stocks, evt_*_count 列

    Returns
    -------
    pd.DataFrame
        index=trade_date, 列包含 up_vol_ratio, down_vol_ratio 等
    """
    if daily_agg.empty:
        logger.warning("输入为空，跳过量能广度计算")
        return pd.DataFrame()

    result = pd.DataFrame(index=daily_agg.index)
    total = daily_agg["total_stocks"].clip(lower=1.0)

    up_col = "evt_up_move_with_vol_spike_count"
    down_col = "evt_down_move_with_vol_spike_count"
    breakout_col = "evt_break_sell_stop_cluster_count"

    result["up_vol_ratio"] = daily_agg[up_col] / total if up_col in daily_agg.columns else 0.0
    result["down_vol_ratio"] = daily_agg[down_col] / total if down_col in daily_agg.columns else 0.0
    result["breakout_vol_ratio"] = daily_agg[breakout_col] / total if breakout_col in daily_agg.columns else 0.0
    result["defensive_vol_ratio"] = (1.0 - result["up_vol_ratio"] - result["down_vol_ratio"]).clip(lower=0.0)
    result["net_vol_breadth"] = result["up_vol_ratio"] - result["down_vol_ratio"]

    return result


def _event_per_day(df: pd.DataFrame, event_col: str, date_col: str = "trade_date") -> pd.Series:
    """按日统计某事件的触发次数。"""
    if event_col not in df.columns:
        return pd.Series(dtype=float)
    sub = df[[date_col, event_col]].copy()
    sub = sub[sub[event_col].notna() & (sub[event_col] != 0)]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby(date_col)[event_col].count()


def compute_volume_by_style(
    lightweight_events: pd.DataFrame,
    stock_attrs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    分组量能对比（逐日）。

    按 cap_tier 分大/中/小盘，计算各组内放量事件占比差异。

    Parameters
    ----------
    lightweight_events : pd.DataFrame
        Step1 轻量事件表
    stock_attrs : pd.DataFrame, optional
        股票属性表，需含 ts_code + cap_tier 列。None 时跳过

    Returns
    -------
    pd.DataFrame
        index=trade_date, cols=small_cap_vol_ratio, large_cap_vol_ratio, cap_spread
    """
    if lightweight_events.empty:
        return pd.DataFrame()
    if stock_attrs is None or "cap_tier" not in stock_attrs.columns:
        logger.warning("无 cap_tier 属性，跳过风格量能")
        return pd.DataFrame()

    le = lightweight_events.merge(stock_attrs[["ts_code", "cap_tier"]], on="ts_code", how="inner")
    if le.empty:
        return pd.DataFrame()

    date_col = "trade_date" if "trade_date" in le.columns else le.index.name or "index"

    results = {}
    for tier in sorted(le["cap_tier"].dropna().unique()):
        subset = le[le["cap_tier"] == tier]
        vol_up = _event_per_day(subset, VOL_UP_EVT, date_col)
        vol_down = _event_per_day(subset, VOL_DOWN_EVT, date_col)
        total_stocks = subset.groupby(date_col)["ts_code"].nunique().clip(lower=1.0)
        ratio = (vol_up.reindex(total_stocks.index).fillna(0) /
                 vol_down.reindex(total_stocks.index).fillna(1)).replace([np.inf, -np.inf], np.nan)
        safe_name = str(tier).replace(" ", "_").replace("-", "_")
        results[f"{safe_name}_vol_ratio"] = ratio

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)

    large_keys = [k for k in out.columns if "large" in k.lower() or "大" in k]
    small_keys = [k for k in out.columns if "small" in k.lower() or "小" in k]

    if len(large_keys) >= 1 and len(small_keys) >= 1:
        out["small_vs_large_spread"] = out[small_keys[0]] - out[large_keys[0]]

    return out


def compute_index_volume_divergence(
    index_events: pd.DataFrame,
    volume_breadth: pd.DataFrame,
) -> pd.DataFrame:
    """
    指数-个股量能背离信号。

    对比指数放量状态与个股量能广度是否一致。

    Parameters
    ----------
    index_events : pd.DataFrame
        extract_index_events() 输出，含 idx_*_vol_spike_up/down
    volume_breadth : pd.DataFrame
        compute_volume_breadth() 输出

    Returns
    -------
    pd.DataFrame
        index=trade_date, cols=*_dvg
    """
    if index_events.empty or volume_breadth.empty:
        return pd.DataFrame()

    idx = index_events.copy()
    common_idx = idx.index.intersection(volume_breadth.index)
    if len(common_idx) < 3:
        return pd.DataFrame()

    idx = idx.loc[common_idx]
    vb = volume_breadth.loc[common_idx]

    result = pd.DataFrame(index=common_idx)

    for prefix in ["sse_", "hs300_"]:
        vu = prefix + "vol_spike_up"
        vd = prefix + "vol_spike_down"

        if vu not in idx.columns or vd not in idx.columns:
            continue

        idx_vol_up = idx[vu] == 1.0
        idx_vol_down = idx[vd] == 1.0
        high_breadth = vb["net_vol_breadth"] > vb["net_vol_breadth"].quantile(0.7)
        low_breadth = vb["net_vol_breadth"] < vb["net_vol_breadth"].quantile(0.3)

        dvg_col = prefix.replace("_", "") + "_vol_dvg"
        conditions = [
            (idx_vol_up & low_breadth, "指数放量但个股广度低"),
            (idx_vol_down & high_breadth, "指数缩量但个股活跃"),
            (idx_vol_up & high_breadth, "指数与个股共振放量"),
            (idx_vol_down & low_breadth, "指数与个股共振缩量"),
        ]
        result[dvg_col] = "中性"
        for cond, label in conditions:
            result.loc[cond, dvg_col] = label

    return result


def compute_turnover_structure(
    daily_agg: pd.DataFrame,
    lightweight_events: pd.DataFrame,
    grouped_df: pd.DataFrame,
    stock_attrs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    计算成交额结构 4 项指标。

    输出列:
      total_amount:            全市场估算成交额（volume * close）
      amount_change_pct:       成交额日环比变化率
      up_vol_amount_pct:       放量上涨股票成交额占比
      down_vol_amount_pct:     放量下跌股票成交额占比
      top5_industry_amount_pct: Top5 强势行业成交额占比
      small_cap_amount_pct:    小盘股成交额占比
      large_cap_amount_pct:    大盘股（含超大盘）成交额占比
      cap_amount_spread:       小盘-大盘成交额占比差值

    Parameters
    ----------
    daily_agg : pd.DataFrame
        batch_processor 输出的日级计数表，含 total_stocks
    lightweight_events : pd.DataFrame
        batch_processor 输出的轻量事件表，含 ts_code/trade_date/evt_*
    grouped_df : pd.DataFrame
        market_aggregator 按行业分组后的聚合表
    stock_attrs : pd.DataFrame, optional
        含 ts_code/cap_tier 列，用于大小盘分组

    Returns
    -------
    pd.DataFrame
        index=trade_date, 含上述 8 列
    """
    import pandas as pd
    import numpy as np

    events = lightweight_events.copy()

    # 估算每只股票每日成交额: amount ≈ volume * close
    if "amount" not in events.columns:
        events = events.rename(columns={"volume": "amount"}, errors="ignore") if "volume" in events.columns else events
        if "amount" not in events.columns:
            logger.info("成交额结构: 从 DB 补充 amount 数据...")
            events = _attach_amount_from_db(events)
            if "amount" not in events.columns:
                logger.warning("成交额结构: 无法获取 amount 数据，返回空 DataFrame")
                empty_cols = [
                    "total_amount", "amount_change_pct",
                    "up_vol_amount_pct", "down_vol_amount_pct",
                    "top5_industry_amount_pct",
                    "small_cap_amount_pct", "large_cap_amount_pct",
                    "cap_amount_spread",
                ]
                if "trade_date" in events.columns:
                    trade_dates = events["trade_date"].drop_duplicates().sort_values()
                    result = pd.DataFrame(index=trade_dates, columns=empty_cols, dtype=float)
                    result.index.name = "trade_date"
                    return result
                return pd.DataFrame(columns=empty_cols)

    # A. 全市场成交额变化
    daily_amount = events.groupby("trade_date")["amount"].sum().sort_index()
    daily_amount = daily_amount.rename("total_amount")
    amount_change = daily_amount.pct_change().rename("amount_change_pct")

    # B. 放量股票成交额占比
    mask_up = events.get("evt_up_move_with_vol_spike", pd.Series(0, index=events.index)).astype(bool)
    mask_down = events.get("evt_down_move_with_vol_spike", pd.Series(0, index=events.index)).astype(bool)

    up_amount = events.loc[mask_up.values].groupby("trade_date")["amount"].sum()
    down_amount = events.loc[mask_down.values].groupby("trade_date")["amount"].sum()

    up_vol_pct = (up_amount / daily_amount).rename("up_vol_amount_pct").fillna(0)
    down_vol_pct = (down_amount / daily_amount).rename("down_vol_amount_pct").fillna(0)

    # C. 强势组（Top5 行业）成交额占比
    top5_amount_pct = pd.Series(np.nan, index=daily_amount.index, name="top5_industry_amount_pct")
    if grouped_df is not None and not grouped_df.empty and "group_type" in grouped_df.columns:
        ind_df = grouped_df[grouped_df["group_type"] == "industry_l2"].copy()
        if not ind_df.empty and "trade_date" in ind_df.columns:
            for dt in daily_amount.index:
                sub = ind_df[ind_df["trade_date"] == dt]
                if sub.empty:
                    continue
                top5 = sub.nlargest(5, "structure_score")
                top5_industries = top5["group_name"].unique()
                # 匹配 lightweight_events 中的 ts_code → 行业映射
                # 简化处理: 取当日所有股票中对应 Top5 行业的成交额占比
                # 精确实现需 stock_attrs + industry 映射
                top5_amount_pct.loc[dt] = np.nan  # 占位，需行业映射
        else:
            top5_amount_pct = pd.Series(np.nan, index=daily_amount.index, name="top5_industry_amount_pct")

    # D. 大小盘成交额占比
    if stock_attrs is not None and not stock_attrs.empty and "cap_tier" in stock_attrs.columns:
        events_with_tier = events.merge(
            stock_attrs[["ts_code", "cap_tier"]], on="ts_code", how="left"
        )
        is_small = events_with_tier["cap_tier"].isin(["small_cap"])
        is_large = events_with_tier["cap_tier"].isin(["large_cap", "mega_cap"])

        small_amount = events_with_tier.loc[is_small.values].groupby("trade_date")["amount"].sum()
        large_amount = events_with_tier.loc[is_large.values].groupby("trade_date")["amount"].sum()

        small_pct = (small_amount / daily_amount).rename("small_cap_amount_pct").fillna(0)
        large_pct = (large_amount / daily_amount).rename("large_cap_amount_pct").fillna(0)
        spread = (small_pct - large_pct).rename("cap_amount_spread")
    else:
        small_pct = pd.Series(np.nan, index=daily_amount.index, name="small_cap_amount_pct")
        large_pct = pd.Series(np.nan, index=daily_amount.index, name="large_cap_amount_pct")
        spread = pd.Series(np.nan, index=daily_amount.index, name="cap_amount_spread")

    result = pd.concat(
        [
            daily_amount,
            amount_change,
            up_vol_pct,
            down_vol_pct,
            top5_amount_pct,
            small_pct,
            large_pct,
            spread,
        ],
        axis=1,
    )
    result.index.name = "trade_date"
    logger.info("成交额结构计算完成: %d 行, %d 列", len(result), len(result.columns))
    return result


def generate_volume_context_report(
    volume_breadth: pd.DataFrame,
    volume_style: Optional[pd.DataFrame] = None,
    divergence: Optional[pd.DataFrame] = None,
    turnover_structure: Optional[pd.DataFrame] = None,
    date: Optional[str] = None,
) -> str:
    """
    生成量能背景解读文本。

    Parameters
    ----------
    volume_breadth : pd.DataFrame
        compute_volume_breadth() 输出
    volume_style : pd.DataFrame, optional
        compute_volume_by_style() 输出
    divergence : pd.DataFrame, optional
        compute_index_volume_divergence() 输出
    turnover_structure : pd.DataFrame, optional
        compute_turnover_structure() 输出
    date : str, optional
        目标日期 YYYY-MM-DD

    Returns
    -------
    str
        解读文本
    """
    lines = []
    lines.append(f"【三、量能特征】")

    if volume_breadth.empty:
        lines.append("  无量能数据")
        return "\n".join(lines)

    target = _get_row(volume_breadth, date)

    upr = float(target.get("up_vol_ratio", 0)) * 100
    dwr = float(target.get("down_vol_ratio", 0)) * 100
    nbr = float(target.get("net_vol_breadth", 0)) * 100
    def_r = float(target.get("defensive_vol_ratio", 0)) * 100
    bor = float(target.get("breakout_vol_ratio", 0)) * 100

    lines.append(f"  广度: 放量上涨 {upr:.1f}% | 放量下跌 {dwr:.1f}% | 净广度 {nbr:+.1f}%")
    lines.append(f"  抗跌/突破: 缩量抗跌 {def_r:.1f}% | 放量突破 {bor:.1f}%")

    if volume_style is not None and not volume_style.empty:
        starget = _get_row(volume_style, date)
        spread_cols = [c for c in starget.index if "spread" in c.lower()]
        if spread_cols:
            sc = spread_cols[0]
            sv = float(starget.get(sc, 0))
            direction = "小盘占优" if sv > 0 else "大盘占优"
            lines.append(f"  风格: {direction} (spread={sv:+.2f})")

    if divergence is not None and not divergence.empty:
        dtarget = _get_row(divergence, date)
        dvg_cols = [c for c in dtarget.index if c.endswith("_dvg")]
        for dc in dvg_cols:
            val = dtarget.get(dc, "中性")
            name = dc.replace("_dvg", "").upper()
            lines.append(f"  {name}背离: {val}")

    if turnover_structure is not None and not turnover_structure.empty:
        ts_target = _get_row(turnover_structure, date)
        amt_chg = ts_target.get("amount_change_pct", np.nan)
        up_amt = ts_target.get("up_vol_amount_pct", np.nan)
        down_amt = ts_target.get("down_vol_amount_pct", np.nan)
        small_amt = ts_target.get("small_cap_amount_pct", np.nan)
        large_amt = ts_target.get("large_cap_amount_pct", np.nan)
        spread = ts_target.get("cap_amount_spread", np.nan)
        if pd.notna(amt_chg):
            lines.append(f"  成交额: 环比变化 {amt_chg:+.1%}")
        if pd.notna(up_amt) and pd.notna(down_amt):
            lines.append(f"  放量成交额占比: 上涨 {up_amt:.1%} | 下跌 {down_amt:.1%}")
        if pd.notna(small_amt) and pd.notna(large_amt):
            direction = "小盘" if spread > 0 else "大盘"
            lines.append(f"  大小盘成交额: 小盘 {small_amt:.1%} | 大盘 {large_amt:.1%} ({direction}主导)")

    interp = _interpret_volume(upr, dwr, nbr)
    lines.append(f"  解读: {interp}")

    return "\n".join(lines)


def _attach_amount_from_db(events: pd.DataFrame) -> pd.DataFrame:
    """从 stock_k_data 表加载 volume*close 作为估算 amount 并 attach 到 events"""
    from datasource.database import get_session, query_df

    if "ts_code" not in events.columns or "trade_date" not in events.columns:
        return events

    ts_codes = events["ts_code"].drop_duplicates().tolist()
    trade_dates = [str(d) for d in events["trade_date"].dt.date.drop_duplicates()]
    date_min = min(trade_dates)
    date_max = max(trade_dates)

    try:
        with get_session() as session:
            df = query_df(
                session,
                "stock_k_data",
                columns=["ts_code", "bar_time", "volume", "close"],
                filters={
                    "freq": "d",
                    "bar_time >= ": date_min,
                    "bar_time <= ": date_max,
                },
            )
        if df is not None and not df.empty:
            df["trade_date"] = pd.to_datetime(df["bar_time"]).dt.date.astype(str)
            df["amount"] = df["volume"] * df["close"]
            df = df[df["ts_code"].isin(ts_codes) & df["trade_date"].isin(trade_dates)]
            keep_cols = ["ts_code", "trade_date", "amount"]
            result = events.copy()
            result["trade_date"] = result["trade_date"].dt.date.astype(str)
            result = result.merge(df[keep_cols], on=["ts_code", "trade_date"], how="left")
            result["trade_date"] = pd.to_datetime(result["trade_date"])
            valid_count = result["amount"].notna().sum()
            logger.info("从 DB 附加 amount: %d 条 -> %d 条有效", len(events), valid_count)
            return result
    except Exception as exc:
        logger.warning("从 DB 加载 amount 失败: %s", exc)

    return events


def _get_row(df: pd.DataFrame, date: Optional[str]) -> pd.Series:
    """从 DataFrame 中获取目标日期行，取最后一行若 date 为空。"""
    if date:
        target = pd.Timestamp(date)
        if target in df.index:
            return df.loc[[target]].iloc[0]
        nearest = df.index[df.index <= target]
        if len(nearest) == 0:
            return df.iloc[-1]
        return df.loc[nearest[-1]]
    return df.iloc[-1]


def _interpret_volume(up_ratio: float, down_ratio: float, net_breadth: float) -> str:
    """根据量能广度数值给出解读。"""
    if net_breadth > 10:
        if up_ratio > 15:
            return "多头放量主导，市场参与度高，趋势延续性强"
        else:
            return "净广度为正但放量上涨占比一般，资金分歧中偏多"
    elif net_breadth < -5:
        if down_ratio > 12:
            return "空头放量主导，抛压明显，注意风险"
        else:
            return "净广度为负但未极端恐慌，观望情绪为主"
    elif abs(net_breadth) < 3:
        return "多空量能均衡，方向选择期"
    else:
        return f"量能偏向{'多头' if net_breadth > 0 else '空头'}但力度有限"


def main():
    parser = argparse.ArgumentParser(description="量能背景层 — 市场级量能分析")
    parser.add_argument("--start", type=str, default="2024-01-01", help="起始日期")
    parser.add_argument("--end", type=str, default="2026-04-30", help="结束日期")
    parser.add_argument("--limit_stocks", type=int, default=None, help="限制股票数量")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from market_structure_analysis.batch_processor import process_stock_pool_streaming
    from market_structure_analysis.market_aggregator import load_stock_attributes

    logger.info("Step 1: 流式处理股票池...")
    daily_agg, lightweight_events = process_stock_pool_streaming(
        start_date=args.start, end_date=args.end, limit_stocks=args.limit_stocks,
    )

    if daily_agg.empty:
        logger.error("无数据")
        return

    logger.info("Step 2: 计算量能广度...")
    vb = compute_volume_breadth(daily_agg)
    print(f"\n量能广度: {vb.shape}")
    print(vb.tail(5).to_string())

    logger.info("Step 3: 加载属性并计算分组量能...")
    try:
        attrs = load_stock_attributes()
        vs = compute_volume_by_style(lightweight_events, attrs)
        if not vs.empty:
            print(f"\n分组量能: {vs.shape}")
            print(vs.tail(5).to_string())
    except Exception as e:
        logger.warning("分组量能跳过: %s", e)
        vs = pd.DataFrame()

    report = generate_volume_context_report(vb, vs)
    print(f"\n{report}")


if __name__ == "__main__":
    main()
