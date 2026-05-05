"""
market_state_classifier.py
市场状态分类器 — 基于市场聚合指标给每个交易日贴盘面标签

Purpose:
    基于每日市场聚合表的 5 个状态指标（结构/动能/量价/筹码/止损流向），
    结合分组聚合的大小盘对比和行业扩散度，给每个交易日分类盘面标签。
    标签体系：4 主标签（互斥）+ 2 副标签（风格/扩散）。

Inputs:
    - daily_df: pd.DataFrame, market_aggregator.aggregate_daily() 输出
    - grouped_df: pd.DataFrame, market_aggregator.aggregate_by_group() 输出（可选）

Outputs:
    - pd.DataFrame, 含 5 个状态指标 + main_label + size_style_label + breadth_label + confidence

How to Run:
    python market_structure_analysis/market_state_classifier.py --start 2024-06-01 --end 2024-12-31 --limit_stocks 10

Examples:
    python market_structure_analysis/market_state_classifier.py --limit_stocks 10 --start 2024-06-01
    python market_structure_analysis/market_state_classifier.py --start 2024-01-01 --end 2024-12-31

Side Effects:
    无（只读，不写入数据库或文件）
"""

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_structure_analysis.market_aggregator import SCORE_NAMES, CAP_TIER_LABELS, MIN_INDUSTRY_SAMPLE

logger = logging.getLogger(__name__)

MAIN_LABELS = ["扩张", "修复", "退潮", "防守"]
SIZE_STYLE_LABELS = ["小票占优", "大票占优"]
BREADTH_LABELS = ["板块扩散", "核心集中"]


def _get_cap_tier_scores(grouped_df: pd.DataFrame, trade_date, tier: str) -> Optional[pd.Series]:
    """从分组聚合表中获取指定市值层级在某日的指标"""
    if grouped_df is None or grouped_df.empty:
        return None
    mask = (
        (grouped_df["group_type"] == "cap_tier")
        & (grouped_df["group_name"] == tier)
    )
    if "trade_date" in grouped_df.columns:
        mask = mask & (grouped_df["trade_date"] == trade_date)
    sub = grouped_df[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


def _classify_main_label(row: pd.Series) -> str:
    """根据 structure_score 和 momentum_score 判定主标签 (v1, 4档)"""
    s = row["structure_score"]
    m = row["momentum_score"]

    if s > 0 and m > 0:
        return "扩张"
    if s > 0 and m <= 0:
        return "修复"
    if s <= 0 and m < 0:
        return "退潮"
    return "防守"


V2_CONFIDENCE_THRESHOLD = 0.4


def _classify_main_label_v2(row: pd.Series, v1_label: str) -> str:
    """
    v2 五档标签: 强扩张/弱扩张/中性/弱退潮/强退潮。

    逻辑:
      - 扩张 + confidence >= 0.4 → 强扩张
      - 扩张 + confidence < 0.4 → 弱扩张
      - 退潮 + confidence >= 0.4 → 强退潮
      - 退潮 + confidence < 0.4 → 弱退潮
      - 修复/防守 → 中性（过渡态）
    """
    confidence = row.get("confidence", 0.0)
    if pd.isna(confidence):
        confidence = 0.0

    if v1_label == "扩张":
        return "强扩张" if confidence >= V2_CONFIDENCE_THRESHOLD else "弱扩张"
    if v1_label == "退潮":
        return "强退潮" if confidence >= V2_CONFIDENCE_THRESHOLD else "弱退潮"
    return "中性"


def _classify_size_style(
    row: pd.Series,
    large_cap_scores: Optional[pd.Series],
    small_cap_scores: Optional[pd.Series],
) -> str:
    """基于大小盘 structure_score 对比判定风格副标签"""
    if large_cap_scores is None or small_cap_scores is None:
        return "大票占优"

    large_struct = large_cap_scores.get("structure_score", 0)
    small_struct = small_cap_scores.get("structure_score", 0)

    if pd.isna(large_struct) or pd.isna(small_struct):
        return "大票占优"

    if small_struct > large_struct:
        return "小票占优"
    return "大票占优"


def _classify_breadth(
    row: pd.Series,
    grouped_df: pd.DataFrame,
    trade_date,
) -> str:
    """基于 breadth_diff 判定扩散副标签（优先），回退到行业截面标准差"""
    if "breadth_diff" in row.index and pd.notna(row.get("breadth_diff")):
        if row["breadth_diff"] > 0:
            return "板块扩散"
        return "核心集中"

    if grouped_df is None or grouped_df.empty:
        return "核心集中"

    ind_sub = grouped_df[grouped_df["group_type"] == "industry_l2"]
    if "trade_date" in ind_sub.columns:
        ind_sub = ind_sub[ind_sub["trade_date"] == trade_date]
    if ind_sub.empty:
        return "核心集中"

    current_std = ind_sub["structure_score"].std()

    all_dates = sorted(ind_sub["trade_date"].unique()) if "trade_date" in ind_sub.columns else []
    if len(all_dates) < 5:
        return "核心集中"

    all_stds = []
    for d in all_dates[-20:]:
        day_sub = ind_sub[ind_sub["trade_date"] == d]
        if len(day_sub) >= 3:
            all_stds.append(day_sub["structure_score"].std())

    if not all_stds:
        return "核心集中"

    avg_std = np.mean(all_stds)

    if current_std > avg_std:
        return "板块扩散"
    return "核心集中"


def _compute_confidence(row: pd.Series, main_label: str) -> float:
    """基于 z20 列计算主标签置信度 (0-1)"""
    z_cols = [f"{s}_z20" for s in SCORE_NAMES]
    z_vals = [row[c] for c in z_cols if c in row.index and pd.notna(row.get(c))]

    if not z_vals:
        abs_scores = [abs(row[c]) for c in SCORE_NAMES if c in row.index and pd.notna(row.get(c))]
        if not abs_scores:
            return 0.0
        return min(np.mean(abs_scores) * 10, 1.0)

    avg_z = np.mean([abs(z) for z in z_vals])
    return min(avg_z / 2.0, 1.0)


def classify_market_state(
    daily_df: pd.DataFrame,
    grouped_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    基于每日市场指标分类盘面状态。

    Parameters
    ----------
    daily_df : pd.DataFrame
        market_aggregator.aggregate_daily() 输出
    grouped_df : pd.DataFrame, optional
        market_aggregator.aggregate_by_group() 输出，用于大小盘对比和行业扩散度

    Returns
    -------
    pd.DataFrame
        含 5 个状态指标 + main_label + size_style_label + breadth_label + confidence
    """
    if daily_df.empty:
        return pd.DataFrame()

    result = daily_df.copy()

    main_labels = []
    main_labels_v2 = []
    size_labels = []
    breadth_labels = []
    confidences = []

    for trade_date, row in result.iterrows():
        td_val = trade_date
        if hasattr(trade_date, "date"):
            td_val = trade_date.date()
        elif isinstance(trade_date, pd.Timestamp):
            td_val = trade_date.date()

        main_label = _classify_main_label(row)
        confidence = _compute_confidence(row, main_label)

        row_with_conf = row.copy()
        row_with_conf["confidence"] = confidence
        main_label_v2 = _classify_main_label_v2(row_with_conf, main_label)

        large_scores = _get_cap_tier_scores(grouped_df, td_val, "large_cap")
        small_scores = _get_cap_tier_scores(grouped_df, td_val, "small_cap")
        size_label = _classify_size_style(row, large_scores, small_scores)

        breadth_label = _classify_breadth(row, grouped_df, td_val)

        main_labels.append(main_label)
        main_labels_v2.append(main_label_v2)
        size_labels.append(size_label)
        breadth_labels.append(breadth_label)
        confidences.append(confidence)

    result["main_label"] = main_labels
    result["main_label_v2"] = main_labels_v2
    result["size_style_label"] = size_labels
    result["breadth_label"] = breadth_labels
    result["confidence"] = confidences

    return result


def generate_market_report(
    labeled_df: pd.DataFrame,
    grouped_df: Optional[pd.DataFrame] = None,
    trade_date: Optional[str] = None,
) -> str:
    """
    生成指定日期的盘面解读报告。

    Parameters
    ----------
    labeled_df : pd.DataFrame
        classify_market_state() 输出
    grouped_df : pd.DataFrame, optional
        分组聚合表
    trade_date : str, optional
        指定日期，None 则取最后一日

    Returns
    -------
    str
        盘面解读文本
    """
    if labeled_df.empty:
        return "无数据"

    if trade_date is not None:
        td = pd.Timestamp(trade_date)
        if td not in labeled_df.index:
            td = labeled_df.index[labeled_df.index.get_indexer([td], method="nearest")[0]]
    else:
        td = labeled_df.index[-1]

    row = labeled_df.loc[td]

    lines = []
    lines.append(f"{'='*50}")
    lines.append(f"盘面解读: {td.strftime('%Y-%m-%d')}")
    lines.append(f"{'='*50}")
    lines.append(f"")
    lines.append(f"【主标签】{row['main_label']} (置信度: {row['confidence']:.2f})")
    lines.append(f"【风格】{row['size_style_label']}")
    lines.append(f"【扩散】{row['breadth_label']}")

    dominant = row.get("dominant_driver", "unknown")
    secondary = row.get("secondary_driver", "unknown")
    if dominant and dominant != "unknown":
        lines.append(f"【主导因子】{dominant} (次: {secondary})")

    lines.append(f"")
    lines.append(f"【状态指标】")
    for s in SCORE_NAMES:
        val = row.get(s, np.nan)
        z_val = row.get(f"{s}_z20", np.nan)
        z_str = f" (z20: {z_val:+.2f})" if pd.notna(z_val) else ""
        lines.append(f"  {s}: {val:+.4f}{z_str}")
    lines.append(f"  多空比: {row.get('bull_bear_ratio', np.nan):.2f}")

    if grouped_df is not None and not grouped_df.empty:
        td_val = td.date() if hasattr(td, "date") else td
        lines.append(f"")
        lines.append(f"【市值分层对比】")
        for tier in ["mega_cap", "large_cap", "mid_cap", "small_cap"]:
            sub = grouped_df[
                (grouped_df["group_type"] == "cap_tier")
                & (grouped_df["group_name"] == tier)
            ]
            if "trade_date" in sub.columns:
                sub = sub[sub["trade_date"] == td_val]
            if not sub.empty:
                s = sub.iloc[0]
                label = CAP_TIER_LABELS.get(tier, tier)
                lines.append(
                    f"  {label}: 结构{s['structure_score']:+.4f} "
                    f"动能{s['momentum_score']:+.4f} "
                    f"量价{s['volume_confirm_score']:+.4f}"
                )

        lines.append(f"")
        lines.append(f"【行业结构 Top5】")
        ind_sub = grouped_df[grouped_df["group_type"] == "industry_l2"]
        if "trade_date" in ind_sub.columns:
            ind_sub = ind_sub[ind_sub["trade_date"] == td_val]
        if not ind_sub.empty:
            ind_sub = ind_sub[ind_sub["stock_count"] >= MIN_INDUSTRY_SAMPLE].copy()
            ind_sub["_composite"] = ind_sub["structure_score"] * np.sqrt(ind_sub["stock_count"])
            top5 = ind_sub.nlargest(5, "_composite")
            for _, r in top5.iterrows():
                lines.append(
                    f"  {r['group_name']}: 结构{r['structure_score']:+.4f} "
                    f"动能{r['momentum_score']:+.4f} ({int(r['stock_count'])}只)"
                )

    lines.append(f"{'='*50}")
    return "\n".join(lines)


BREADTH_NAMES = [
    "bull_event_rate",
    "bear_event_rate",
    "breadth_diff",
    "stock_coverage",
    "event_concentration",
]


def export_daily_summary(
    labeled_df: pd.DataFrame,
    grouped_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    fmt: str = "csv",
) -> pd.DataFrame:
    """
    导出每日盘面摘要为结构化 CSV/JSON。

    Parameters
    ----------
    labeled_df : pd.DataFrame
        classify_market_state() 输出
    grouped_df : pd.DataFrame, optional
        分组聚合表，用于提取 Top 行业
    output_path : str, optional
        输出文件路径，None 则不写文件
    fmt : str
        输出格式: 'csv' 或 'json'

    Returns
    -------
    pd.DataFrame
        结构化摘要表
    """
    if labeled_df.empty:
        return pd.DataFrame()

    export_cols = [
        "main_label", "main_label_v2", "confidence", "size_style_label", "breadth_label",
        "dominant_driver", "secondary_driver",
    ]
    score_cols = [c for c in SCORE_NAMES if c in labeled_df.columns]
    z_cols = [f"{s}_z20" for s in SCORE_NAMES if f"{s}_z20" in labeled_df.columns]
    breadth_cols = [c for c in BREADTH_NAMES if c in labeled_df.columns]

    all_cols = export_cols + score_cols + z_cols + breadth_cols
    present_cols = [c for c in all_cols if c in labeled_df.columns]

    result = labeled_df[present_cols].copy()
    result.index.name = "trade_date"

    if grouped_df is not None and not grouped_df.empty:
        ind_sub = grouped_df[grouped_df["group_type"] == "industry_l2"]
        if "stock_count" in ind_sub.columns:
            ind_sub = ind_sub[ind_sub["stock_count"] >= 20]

        top1_list = []
        top2_list = []
        for trade_date in result.index:
            td_val = trade_date.date() if hasattr(trade_date, "date") else trade_date
            day_ind = ind_sub[ind_sub["trade_date"] == td_val] if "trade_date" in ind_sub.columns else pd.DataFrame()
            if not day_ind.empty:
                top = day_ind.nlargest(2, "structure_score")["group_name"].tolist()
                top1_list.append(top[0] if len(top) >= 1 else "")
                top2_list.append(top[1] if len(top) >= 2 else "")
            else:
                top1_list.append("")
                top2_list.append("")
        result["top_industry_1"] = top1_list
        result["top_industry_2"] = top2_list

    risk_notes = []
    for _, row in result.iterrows():
        notes = []
        if "confidence" in row.index and pd.notna(row.get("confidence")) and row["confidence"] < 0.3:
            notes.append("低置信度")
        if "breadth_diff" in row.index and pd.notna(row.get("breadth_diff")) and row["breadth_diff"] < 0:
            notes.append("广度偏弱")
        if "bull_bear_ratio" in row.index and pd.notna(row.get("bull_bear_ratio")) and row["bull_bear_ratio"] < 0.5:
            notes.append("空头主导")
        risk_notes.append(";".join(notes) if notes else "")
    result["risk_note"] = risk_notes

    if output_path:
        if fmt == "json":
            result.to_json(output_path, orient="index", date_format="iso", force_ascii=False)
        else:
            result.to_csv(output_path)
        logger.info("已输出到: %s", output_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="市场状态分类器")
    parser.add_argument("--start", type=str, default="2024-06-01", help="起始日期")
    parser.add_argument("--end", type=str, default="2024-12-31", help="结束日期")
    parser.add_argument("--freq", type=str, default="d", help="K线周期")
    parser.add_argument("--limit_stocks", type=int, default=None, help="限制股票数量")
    parser.add_argument("--report_date", type=str, default=None, help="生成报告的日期")
    parser.add_argument("--include_concept", action="store_true", help="包含概念分组")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（支持 .csv/.json）")
    parser.add_argument("--all", action="store_true", help="一站式输出：市场状态+风格+breadth+主导因子+主线题材+风险")
    parser.add_argument("--top_themes", type=int, default=10, help="--all 模式下输出 Top N 主线题材")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from market_structure_analysis.market_aggregator import run_aggregation

    daily, grouped, _, _ = run_aggregation(
        start_date=args.start,
        end_date=args.end,
        freq=args.freq,
        limit_stocks=args.limit_stocks,
        include_concept=args.include_concept,
    )

    if daily.empty:
        logger.error("聚合无结果")
        return

    labeled = classify_market_state(daily, grouped)

    print("\n=== 主标签分布 ===")
    print(labeled["main_label"].value_counts().to_string())

    print("\n=== 风格副标签分布 ===")
    print(labeled["size_style_label"].value_counts().to_string())

    print("\n=== 扩散副标签分布 ===")
    print(labeled["breadth_label"].value_counts().to_string())

    print("\n=== 最近 10 日标签 ===")
    label_cols = SCORE_NAMES + ["main_label", "size_style_label", "breadth_label", "confidence"]
    label_cols = [c for c in label_cols if c in labeled.columns]
    print(labeled[label_cols].tail(10).to_string())

    report = generate_market_report(labeled, grouped, trade_date=args.report_date)
    print(f"\n{report}")

    if args.output:
        fmt = "json" if args.output.endswith(".json") else "csv"
        export_daily_summary(labeled, grouped, output_path=args.output, fmt=fmt)

    if args.all:
        from market_structure_analysis.theme_aggregator import (
            load_concept_map, aggregate_themes, generate_theme_report, identify_theme_trends,
        )
        from market_structure_analysis.regime_filter import generate_regime_summary

        report_date = args.report_date or labeled.index[-1]

        print(f"\n{'='*60}")
        print(f"一站日报 — {pd.Timestamp(report_date).date() if report_date else '最新'}")
        print(f"{'='*60}")

        print(generate_regime_summary(labeled["main_label_v2"]))

        latest = labeled.loc[report_date] if report_date in labeled.index else labeled.iloc[-1]
        print(f"\n=== 当日市场状态 ===")
        print(f"  主标签(v2): {latest.get('main_label_v2', '未知')}")
        print(f"  置信度: {latest.get('confidence', 0):.2f}")
        print(f"  风格: {latest.get('size_style_label', '未知')}")
        print(f"  广度: {latest.get('breadth_label', '未知')}")
        print(f"  主导因子: {latest.get('dominant_driver', '未知')}")
        print(f"  次导因子: {latest.get('secondary_driver', '未知')}")

        try:
            concept_map = load_concept_map()
            theme_agg = aggregate_themes(grouped.groupby("group_name") if not grouped.empty else pd.DataFrame(), concept_map)
        except Exception as e:
            logger.warning("题材聚合失败: %s", e)
            theme_agg = pd.DataFrame()

        if not theme_agg.empty:
            theme_report = generate_theme_report(theme_agg, date=report_date, top_n=args.top_themes)
            print(f"\n{theme_report}")


if __name__ == "__main__":
    main()
