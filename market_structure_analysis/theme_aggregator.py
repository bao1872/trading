"""
theme_aggregator.py — 题材主线识别层

Purpose:  从 stock_pools.concepts 构建 股票↔概念映射，按日聚合事件/score，输出每日主线题材排序。
Inputs:   stock_pools 表的 concepts 列；market_aggregator 输出的 lightweight_events / grouped 数据
Outputs:  每日 Top N 主线题材排名、强度趋势、题材报告文本
How to Run:
    python market_structure_analysis/theme_aggregator.py --date 2024-12-31
Examples:
    from market_structure_analysis.theme_aggregator import rank_daily_themes, load_concept_map
    concept_map = load_concept_map()
    top = rank_daily_themes(theme_agg, date="2024-12-31", top_n=10)
Side Effects: 无（只读数据库 + 只读 DataFrame）
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_session, query_df
from market_structure_analysis.market_aggregator import (
    aggregate_by_group, load_stock_attributes, GENERIC_CONCEPTS, EVT_COLS, SCORE_NAMES,
)

logger = logging.getLogger(__name__)

from market_structure_analysis._config import MIN_THEME_SAMPLE


def load_concept_map(min_stocks: int = MIN_THEME_SAMPLE) -> pd.DataFrame:
    """
    从 stock_pools.concepts 构建 股票↔概念 映射表，过滤泛概念。

    Parameters
    ----------
    min_stocks : int
        最小股票数，低于此的概念不保留

    Returns
    -------
    pd.DataFrame
        columns=[ts_code, concept]
    """
    stock_attrs = load_stock_attributes()
    if "concepts" not in stock_attrs.columns:
        raise KeyError("stock_pools 表缺少 concepts 列")

    concept_map = (
        stock_attrs[["ts_code", "concepts"]]
        .dropna(subset=["concepts"])
        .assign(concepts=lambda x: x["concepts"].str.split(";"))
        .explode("concepts")
    )
    concept_map["concepts"] = concept_map["concepts"].str.strip()
    concept_map = concept_map[~concept_map["concepts"].isin(GENERIC_CONCEPTS)]
    concept_map = concept_map[concept_map["concepts"] != ""]
    concept_map = concept_map.rename(columns={"concepts": "concept"})

    concept_sizes = concept_map.groupby("concept")["ts_code"].nunique()
    valid_concepts = concept_sizes[concept_sizes >= min_stocks].index
    concept_map = concept_map[concept_map["concept"].isin(valid_concepts)]

    logger.info("概念映射: %d 个有效概念, %d 行", concept_map["concept"].nunique(), len(concept_map))
    return concept_map


def aggregate_themes(
    lightweight_events: pd.DataFrame,
    concept_map: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    按概念聚合事件和 score（thin wrapper around aggregate_by_group with group_type='concept'）。

    Parameters
    ----------
    lightweight_events : pd.DataFrame
        process_stock_pool_streaming() 输出的轻量事件表
    concept_map : pd.DataFrame, optional
        load_concept_map() 输出

    Returns
    -------
    pd.DataFrame
        GroupedMarketSummary (group_type='concept')
        额外列: composite_strength = structure_score × sqrt(stock_count)
    """
    if concept_map is None:
        concept_map = load_concept_map()

    stock_attrs = load_stock_attributes()

    events_with_concept = lightweight_events.merge(
        concept_map, left_on="ts_code", right_on="ts_code", how="inner"
    )
    events_with_concept = events_with_concept.rename(columns={"concept": "concepts"})

    import copy
    adjusted_attrs = stock_attrs.copy()
    adjusted_attrs["concepts"] = None

    theme_agg = aggregate_by_group(events_with_concept, "concept", adjusted_attrs)
    if theme_agg.empty:
        return theme_agg

    if "stock_count" in theme_agg.columns and "structure_score" in theme_agg.columns:
        theme_agg["composite_strength"] = (
            theme_agg["structure_score"] * np.sqrt(theme_agg["stock_count"].clip(lower=1))
        )

    return theme_agg


def rank_daily_themes(
    theme_agg: pd.DataFrame,
    date=None,
    top_n: int = 10,
    sort_by: str = "composite_strength",
    min_event_ratio: float = 0.0,
    lightweight_events: Optional[pd.DataFrame] = None,
    concept_map: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    获取指定日期的 Top N 主线题材，含可信度过滤。

    Parameters
    ----------
    theme_agg : pd.DataFrame
        aggregate_themes() 输出
    date : str or pd.Timestamp, optional
        默认取最新日期
    top_n : int
    sort_by : str
        排序字段
    min_event_ratio : float
        最小事件覆盖率阈值。0.0 = 不过滤。建议 0.05~0.10
    lightweight_events : pd.DataFrame, optional
        用于计算事件覆盖率，含 ts_code/trade_date/evt_*
    concept_map : pd.DataFrame, optional
        用于计算事件覆盖率

    Returns
    -------
    pd.DataFrame
        Top N 题材，含 credibility 标注
    """
    if theme_agg.empty:
        return pd.DataFrame()

    if date is None:
        date = theme_agg["trade_date"].max()
    else:
        date = pd.Timestamp(date)

    day_data = theme_agg[theme_agg["trade_date"] == date]
    if day_data.empty:
        return pd.DataFrame()

    # 事件覆盖率计算
    if min_event_ratio > 0 and lightweight_events is not None and not lightweight_events.empty \
            and concept_map is not None and not concept_map.empty:
        day_events = lightweight_events[lightweight_events["trade_date"] == date]
        if not day_events.empty:
            evt_cols_list = [c for c in day_events.columns if c.startswith("evt_")]
            # 每只股票是否触发了任意事件
            day_events["_has_event"] = day_events[evt_cols_list].max(axis=1).fillna(0) > 0
            events_with_cp = day_events.merge(
                concept_map, on="ts_code", how="inner"
            )
            cp_stats = events_with_cp.groupby("concept").agg(
                total_stocks=("ts_code", "nunique"),
                event_stocks=("_has_event", "sum"),
            )
            cp_stats["event_ratio"] = cp_stats["event_stocks"] / cp_stats["total_stocks"].clip(lower=1)
            day_data = day_data.merge(
                cp_stats[["event_ratio"]],
                left_on="group_name", right_index=True, how="left"
            )
            # 过滤低覆盖率题材
            day_data = day_data[
                day_data["event_ratio"].isna() | (day_data["event_ratio"] >= min_event_ratio)
            ]
            day_data["credibility"] = day_data["event_ratio"].apply(
                lambda x: "可信" if pd.notna(x) and x >= 0.10 else "一般" if pd.notna(x) else "未评估"
            )
        else:
            day_data["credibility"] = "未评估"
    else:
        day_data["credibility"] = "未评估"

    if sort_by not in day_data.columns:
        sort_by = "structure_score" if "structure_score" in day_data.columns else day_data.columns[0]

    ranked = day_data.nlargest(top_n, sort_by)
    display_cols = ["group_name", "stock_count", "structure_score", sort_by, "credibility"]
    avail_display = [c for c in display_cols if c in ranked.columns]
    return ranked[avail_display].reset_index(drop=True)


def identify_theme_trends(
    theme_agg: pd.DataFrame,
    n_days: int = 5,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    识别最近 N 日强度上升/下降的概念。

    Parameters
    ----------
    theme_agg : pd.DataFrame
    n_days : int
        回看天数
    top_n : int
        分析 Top N 概念

    Returns
    -------
    pd.DataFrame
        columns=[concept, current_strength, prev_strength, change, trend]
    """
    if theme_agg.empty or "trade_date" not in theme_agg.columns:
        return pd.DataFrame()

    dates = sorted(theme_agg["trade_date"].unique())
    if len(dates) < n_days:
        return pd.DataFrame()

    recent = dates[-1]
    earlier = dates[-n_days] if len(dates) >= n_days else dates[0]

    recent_data = theme_agg[theme_agg["trade_date"] == recent]
    earlier_data = theme_agg[theme_agg["trade_date"] == earlier]

    strength_col = "composite_strength" if "composite_strength" in theme_agg.columns else "structure_score"

    top_recent = set(recent_data.nlargest(top_n, strength_col)["group_name"])
    top_earlier = set(earlier_data.nlargest(top_n, strength_col)["group_name"])

    all_concepts = top_recent | top_earlier

    rows = []
    for concept in all_concepts:
        rc = recent_data[recent_data["group_name"] == concept]
        ec = earlier_data[earlier_data["group_name"] == concept]
        curr = rc[strength_col].values[0] if not rc.empty else 0.0
        prev = ec[strength_col].values[0] if not ec.empty else 0.0

        if prev > 0:
            change_pct = (curr - prev) / prev
        else:
            change_pct = 1.0 if curr > 0 else 0.0

        if change_pct > 0.1:
            trend = "上升"
        elif change_pct < -0.1:
            trend = "下降"
        else:
            trend = "持平"

        rows.append({
            "concept": concept,
            "current_strength": round(curr, 4),
            "prev_strength": round(prev, 4),
            "change_pct": round(change_pct, 4),
            "trend": trend,
            "is_new_entry": concept in top_recent and concept not in top_earlier,
            "is_dropout": concept not in top_recent and concept in top_earlier,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("current_strength", ascending=False)
    return result


def generate_theme_report(
    theme_agg: pd.DataFrame,
    date=None,
    top_n: int = 10,
    show_trends: bool = True,
) -> str:
    """生成单日题材报告文本。"""
    if theme_agg.empty:
        return "无题材数据"

    top = rank_daily_themes(theme_agg, date=date, top_n=top_n)
    if top.empty:
        return "当日无题材数据"

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"【主线题材】 {pd.Timestamp(date).date() if date else '最新'}")
    lines.append(f"")

    for i, row in top.iterrows():
        name = row["group_name"]
        stock_cnt = int(row.get("stock_count", 0))
        strength = row.get("composite_strength", row.get("structure_score", 0))
        cred = row.get("credibility", "未评估")
        cred_flag = "" if cred == "可信" else f" [{cred}]" if cred != "未评估" else ""
        lines.append(f"  {i+1:2d}. {name:20s}  股票数:{stock_cnt:4d}  强度:{strength:+.4f}{cred_flag}")

    if show_trends:
        trends = identify_theme_trends(theme_agg, n_days=5)
        if not trends.empty:
            rising = trends[trends["trend"] == "上升"]
            falling = trends[trends["trend"] == "下降"]
            new_entries = trends[trends["is_new_entry"]]

            lines.append(f"")
            if not rising.empty:
                lines.append(f"  强度上升({len(rising)}): {', '.join(rising['concept'].head(5))}")
            if not falling.empty:
                lines.append(f"  强度下降({len(falling)}): {', '.join(falling['concept'].head(5))}")
            if not new_entries.empty:
                lines.append(f"  新进入 Top20({len(new_entries)}): {', '.join(new_entries['concept'].head(5))}")

    lines.append(f"{'='*60}")
    return "\n".join(lines)


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="题材主线识别")
    parser.add_argument("--date", type=str, default=None, help="日期 (YYYY-MM-DD)，默认最新")
    parser.add_argument("--top_n", type=int, default=10, help="Top N 主线")
    parser.add_argument("--include_concept", action="store_true", help="在聚合时包含概念分组")
    parser.add_argument("--limit_stocks", type=int, default=None, help="限制股票数（调试）")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--freq", type=str, default="d")
    args = parser.parse_args()

    from market_structure_analysis.market_aggregator import run_aggregation
    from market_structure_analysis.batch_processor import process_stock_pool_streaming

    logger.info("加载概念映射...")
    concept_map = load_concept_map()

    logger.info("流式处理...")
    daily_agg, lightweight_events = process_stock_pool_streaming(
        freq=args.freq, start_date=args.start, end_date=args.end, limit_stocks=args.limit_stocks,
    )

    if lightweight_events.empty:
        logger.error("无事件数据")
        return

    logger.info("按概念聚合...")
    theme_agg = aggregate_themes(lightweight_events, concept_map)
    logger.info("题材聚合: %d 行", len(theme_agg))

    report = generate_theme_report(theme_agg, date=args.date, top_n=args.top_n)
    print(report)


if __name__ == "__main__":
    main()
