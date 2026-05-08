"""
enhanced_daily_report.py
增强日报模板 — 整合市场状态/指数背景/量能/题材/风险 五段式输出

Purpose:
    整合所有分析层输出，生成结构化 A 股盘面日报。
    替代之前散落的 print 输出，提供统一格式的每日报告。

Inputs:
    - labeled_df: classify_market_state() 输出（主标签+风格+广度+分数）
    - index_states: index_context.classify_index_state() 输出
    - volume_breadth: volume_context.compute_volume_breadth() 输出
    - theme_report_text: theme_aggregator.generate_theme_report() 输出文本
    - regime_rules: regime_filter.REGIME_RULES 字典

Outputs:
    - 结构化文本报告（六段式）
    - 增强版 CSV（含所有新增列）

How to Run:
    python market_structure_analysis/enhanced_daily_report.py --date 2026-04-30

Examples:
    python market_structure_analysis/enhanced_daily_report.py
    python market_structure_analysis/enhanced_daily_report.py --date 2026-04-30 --output /tmp/report.md

Side Effects:
    可选写入文件（--output 参数），不写入数据库
"""

import argparse
import logging
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def build_enhanced_daily_report(
    labeled_df: pd.DataFrame,
    index_states: Optional[pd.DataFrame] = None,
    volume_breadth: Optional[pd.DataFrame] = None,
    turnover_structure: Optional[pd.DataFrame] = None,
    theme_report_text: str = "",
    regime_rules: Optional[Dict] = None,
    date: Optional[str] = None,
) -> str:
    """
    构建单日增强日报文本（六段式定版）。

    Parameters
    ----------
    labeled_df : pd.DataFrame
        classify_market_state() 输出
    index_states : pd.DataFrame, optional
        classify_index_state() 输出
    volume_breadth : pd.DataFrame, optional
        compute_volume_breadth() 输出
    turnover_structure : pd.DataFrame, optional
        compute_turnover_structure() 输出
    theme_report_text : str
        题材报告文本
    regime_rules : dict, optional
        REGIME_RULES 字典
    date : str, optional
        目标日期 YYYY-MM-DD

    Returns
    -------
    str
        完整日报文本（六段）
    """
    lines = []
    target_date = _resolve_date(labeled_df, date)

    lines.append("=" * 60)
    if target_date and pd.notna(target_date):
        lines.append(f"A股盘面日报 — {target_date.strftime('%Y-%m-%d')}")
    else:
        lines.append("A股盘面日报 — (无数据)")
    lines.append("=" * 60)

    lines.append(_build_section_one(labeled_df, target_date))
    lines.append(_build_section_two(index_states, target_date))
    lines.append(_build_section_three(volume_breadth, turnover_structure, target_date))
    lines.append(_build_section_four(theme_report_text))
    lines.append(_build_section_five(labeled_df, target_date))
    lines.append(_build_section_six(labeled_df, regime_rules, target_date))

    lines.append("=" * 60)
    return "\n".join(lines)


def _resolve_date(df: pd.DataFrame, date: Optional[str]) -> Optional[pd.Timestamp]:
    """解析目标日期，空/无效数据返回 None"""
    if date:
        return pd.Timestamp(date)
    if df is not None and not df.empty and hasattr(df.index, 'max'):
        result = df.index.max()
        if pd.notna(result):
            return result
    return None


def _build_section_one(labeled_df: pd.DataFrame, date: pd.Timestamp) -> str:
    """【一、市场状态】主标签 + 风格 + 广度 + 主导因子"""
    lines = ["", "【一、市场状态】"]

    if len(labeled_df) == 0 or date not in labeled_df.index:
        nearest = labeled_df.index[labeled_df.index <= date]
        if len(nearest) == 0:
            lines.append("  无数据")
            return "\n".join(lines)
        row = labeled_df.loc[nearest[-1]]
    else:
        row = labeled_df.loc[date]

    main_label = row.get("main_label_v2", row.get("main_label", "未知"))
    confidence = row.get("confidence", 0)
    size_style = row.get("size_style_label", "未知")
    breadth = row.get("breadth_label", "未知")

    dominant = row.get("dominant_driver", "未知")
    sub_dominant = row.get("sub_dominant_driver", "未知")
    dom_z = row.get("dominant_z20", 0)
    sub_z = row.get("sub_dominant_z20", 0)

    lines.append(f"  主标签: {main_label} (置信度 {confidence:.0%})")
    lines.append(f"  风格: {size_style} | 扩散: {breadth}")
    lines.append(f"  主导因子: {dominant} ({dom_z:+.1f}σ) | 次主导: {sub_dominant} ({sub_z:+.1f}σ)")

    score_names = ["structure_score", "momentum_score", "volume_confirm_score",
                    "breakout_score", "stop_flow_score"]
    score_line_parts = []
    for sn in score_names:
        z_col = f"{sn}_z20"
        if z_col in row.index and not pd.isna(row[z_col]):
            score_line_parts.append(f"{sn[:4]}={row[z_col]:+.1f}")
    if score_line_parts:
        lines.append(f"  分数: {' | '.join(score_line_parts)}")

    return "\n".join(lines)


def _build_section_two(index_states: Optional[pd.DataFrame], date: pd.Timestamp) -> str:
    """【二、指数背景】"""
    from market_structure_analysis.index_context import generate_index_context_report
    return generate_index_context_report(index_states if index_states is not None else pd.DataFrame(), date=date)


def _build_section_three(volume_breadth: Optional[pd.DataFrame],
                          turnover_structure: Optional[pd.DataFrame] = None,
                          date: pd.Timestamp = None) -> str:
    """【三、量能特征】含成交额结构 4 项"""
    from market_structure_analysis.volume_context import generate_volume_context_report, compute_volume_breadth
    return generate_volume_context_report(
        volume_breadth if volume_breadth is not None else pd.DataFrame(),
        turnover_structure=turnover_structure,
        date=date,
    )


def _build_section_four(theme_report_text: str) -> str:
    """【四、主线题材】"""
    lines = ["", "【四、主线题材】"]
    if theme_report_text and len(theme_report_text.strip()) > 5:
        for line in theme_report_text.strip().split("\n"):
            if line.strip():
                lines.append(f"  {line}")
    else:
        lines.append("  暂无题材数据")
    return "\n".join(lines)


def _build_section_five(labeled_df: pd.DataFrame, date: pd.Timestamp) -> str:
    """【五、风险提示】stop_flow / breadth 偏弱 / 指数背离 / 低置信度"""
    from market_structure_analysis._config import CONFIDENCE_LOW_THRESHOLD

    lines = ["", "【五、风险提示】"]

    if len(labeled_df) == 0 or date not in labeled_df.index:
        nearest = labeled_df.index[labeled_df.index <= date]
        row = labeled_df.loc[nearest[-1]] if len(nearest) > 0 else pd.Series()
    else:
        row = labeled_df.loc[date]

    notes = []

    stop_z = row.get("stop_flow_score_z20", 0)
    if pd.notna(stop_z) and float(stop_z) < -1.0:
        notes.append(f"止损流出加剧 (stop_flow z20={stop_z:+.1f}σ)")

    if "breadth_diff" in row.index and pd.notna(row.get("breadth_diff")) and row["breadth_diff"] < 0:
        notes.append("上涨事件集中在少数股票（核心集中），警惕结构性脆弱")

    if "confidence" in row.index and pd.notna(row.get("confidence")) and row["confidence"] < CONFIDENCE_LOW_THRESHOLD:
        notes.append(f"低置信度 ({row['confidence']:.0%}，低于 {CONFIDENCE_LOW_THRESHOLD})")

    if "risk_note" in row.index and pd.notna(row.get("risk_note")) and str(row["risk_note"]).strip():
        notes.append(str(row["risk_note"]).strip())

    if notes:
        for n in notes:
            lines.append(f"  ⚠ {n}")
    else:
        lines.append("  无明显风险信号")

    return "\n".join(lines)


def _build_section_six(labeled_df: pd.DataFrame, regime_rules: Optional[Dict], date: pd.Timestamp) -> str:
    """【六、交易环境建议】Regime + 仓位 + 允许/禁止的交易类型"""
    lines = ["", "【六、交易环境建议】"]

    if len(labeled_df) == 0 or date not in labeled_df.index:
        nearest = labeled_df.index[labeled_df.index <= date]
        row = labeled_df.loc[nearest[-1]] if len(nearest) > 0 else pd.Series()
    else:
        row = labeled_df.loc[date]

    label = row.get("main_label_v2", row.get("main_label", "中性"))

    if regime_rules and label in regime_rules:
        rule = regime_rules[label]
        pos = rule.get("position", "unknown")
        pos_pct = rule.get("position_pct", 0)
        allow_agg = rule.get("allow_aggressive", False)
        lines.append(f"  Regime: 当前「{label}」环境, 建议 {pos} ({pos_pct:.0%})")
        lines.append(f"  策略: {'允许激进' if allow_agg else '禁止激进'} | "
                     f"突破{'允许' if rule.get('allow_breakout', False) else '限制'} | "
                     f"反转{'允许' if rule.get('allow_reversal', False) else '限制'}")
    else:
        lines.append(f"  Regime: 「{label}」 (无映射规则)")

    lines.append("")
    lines.append("  注: 本报告基于事件聚合与统计模型，不构成投资建议。")

    return "\n".join(lines)


def export_enhanced_report_csv(
    labeled_df: pd.DataFrame,
    index_states: Optional[pd.DataFrame] = None,
    volume_breadth: Optional[pd.DataFrame] = None,
    turnover_structure: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    导出增强版 CSV（合并所有新增列到 labeled_df）。

    Parameters
    ----------
    labeled_df : pd.DataFrame
        原始标签表
    index_states : pd.DataFrame, optional
        指数状态表
    volume_breadth : pd.DataFrame, optional
        量能广度表
    turnover_structure : pd.DataFrame, optional
        成交额结构表
    output_path : str, optional
        输出路径，默认 /tmp/market_enhanced_daily.csv

    Returns
    -------
    pd.DataFrame
        合并后的 DataFrame
    """
    out = labeled_df.copy()

    if index_states is not None and not index_states.empty:
        idx_cols = [c for c in index_states.columns if c not in out.columns]
        if idx_cols:
            out = out.join(index_states[idx_cols], how="left")

    if volume_breadth is not None and not volume_breadth.empty:
        vb_cols = [c for c in volume_breadth.columns if c not in out.columns]
        if vb_cols:
            out = out.join(volume_breadth[vb_cols], how="left")

    if turnover_structure is not None and not turnover_structure.empty:
        ts_cols = [c for c in turnover_structure.columns if c not in out.columns]
        if ts_cols:
            out = out.join(turnover_structure[ts_cols], how="left")

    out.to_csv(output_path, encoding="utf-8-sig")
    logger.info("增强 CSV 已保存: %s (%d 行)", output_path, len(out))
    return out


def main():
    parser = argparse.ArgumentParser(description="增强日报模板 — 整合六段式输出")
    parser.add_argument("--date", type=str, default=None, help="目标日期 YYYY-MM-DD")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--start", type=str, default="2026-01-01", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2026-05-08", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--limit_stocks", type=int, default=None, help="限制处理股票数")
    parser.add_argument("--cache-dir", type=str, default="/tmp/market_cache", help="中间结果缓存目录")
    parser.add_argument("--skip-pavp", action="store_true", help="跳过 PAVP 计算")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from market_structure_analysis.market_aggregator import run_aggregation
    from market_structure_analysis.market_state_classifier import classify_market_state
    from market_structure_analysis.index_context import (
        compute_index_factors, extract_index_events, classify_index_state,
    )
    from market_structure_analysis.volume_context import compute_volume_breadth, compute_turnover_structure

    logger.info("加载数据...")
    daily, grouped, _, lightweight_events = run_aggregation(
        start_date=args.start, end_date=args.end,
        limit_stocks=args.limit_stocks, include_concept=True,
        cache_dir=args.cache_dir, skip_pavp=args.skip_pavp,
    )

    if daily.empty:
        logger.error("无数据")
        return

    labeled = classify_market_state(daily, grouped)

    logger.info("计算指数锚点...")
    idx_factors = compute_index_factors(start_date=args.start, end_date=args.end)
    idx_events = extract_index_events(idx_factors)
    idx_states = classify_index_state(idx_events)

    logger.info("计算量能背景...")
    vb = compute_volume_breadth(daily)

    logger.info("计算成交额结构...")
    ts = compute_turnover_structure(daily, lightweight_events, grouped)

    theme_text = ""
    try:
        from market_structure_analysis.theme_aggregator import generate_theme_report
        concept_grouped = grouped[grouped["group_type"] == "concept"] if "group_type" in grouped.columns else pd.DataFrame()
        if len(concept_grouped) > 0:
            theme_text = generate_theme_report(concept_grouped, top_n=5)
    except Exception:
        pass

    from market_structure_analysis.regime_filter import REGIME_RULES

    report = build_enhanced_daily_report(labeled, idx_states, vb, ts, theme_text, REGIME_RULES, date=args.date)
    print(report)

    if args.output:
        export_enhanced_report_csv(labeled, idx_states, vb, ts, output_path=args.output)


if __name__ == "__main__":
    main()
