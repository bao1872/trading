"""
market_validator.py
市场验证器 — 验证标签和分数的历史解释力

Purpose:
    验证盘面标签和状态指标是否具有解释力、稳定性和交易价值。
    核心验证：标签后指数/全A收益、事件延续性、风格表现。

Inputs:
    - labeled_df: pd.DataFrame, classify_market_state() 输出
    - events_df: pd.DataFrame, batch_processor 输出的个股事件表
    - stock_k_data 表: 用于计算全A等权收益

Outputs:
    - 标签后收益统计表
    - 事件延续性统计表
    - 验证报告文本

How to Run:
    python market_structure_analysis/market_validator.py --start 2024-01-01 --end 2026-04-30

Examples:
    python market_structure_analysis/market_validator.py --start 2024-01-01 --end 2026-04-30
    python market_structure_analysis/market_validator.py --limit_stocks 100 --start 2024-06-01

Side Effects:
    无（只读，不写入数据库或文件）
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_session, query_df
from market_structure_analysis.event_detector import CORE_EVENTS
from market_structure_analysis._config import CONFIDENCE_LOW_THRESHOLD

logger = logging.getLogger(__name__)

EVT_COLS = [f"evt_{e}" for e in CORE_EVENTS]

BULL_EVENTS = [
    "evt_dsa_dir_flip_up",
    "evt_cross_above_dsa_vwap",
    "evt_bbmacd_cross_upper",
    "evt_up_move_with_vol_spike",
    "evt_cross_above_value_area_high",
    "evt_break_sell_stop_cluster",
]

BEAR_EVENTS = [
    "evt_dsa_dir_flip_down",
    "evt_cross_below_dsa_vwap",
    "evt_bbmacd_cross_lower",
    "evt_down_move_with_vol_spike",
    "evt_cross_below_value_area_low",
    "evt_break_buy_stop_cluster",
]


def compute_all_a_equal_weight_return(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    从 stock_k_data 计算全A等权日收益率（SQL 端聚合，避免全量加载到 Python）。

    Parameters
    ----------
    start_date : str, optional
        起始日期
    end_date : str, optional
        结束日期

    Returns
    -------
    pd.Series
        index=日期, values=全A等权日收益率
    """
    from sqlalchemy import text

    sql = """
        WITH daily_ret AS (
            SELECT
                bar_time,
                ts_code,
                close::float / LAG(close::float) OVER (
                    PARTITION BY ts_code ORDER BY bar_time
                ) - 1 AS pct_change
            FROM stock_k_data
            WHERE freq = 'd'
    """
    params = {}
    if start_date:
        sql += " AND bar_time >= :start_date"
        params["start_date"] = start_date
    if end_date:
        sql += " AND bar_time <= :end_date"
        params["end_date"] = end_date

    sql += """
        )
        SELECT
            bar_time AS trade_date,
            AVG(pct_change) AS equal_weight_return
        FROM daily_ret
        WHERE pct_change IS NOT NULL
        GROUP BY bar_time
        ORDER BY bar_time
    """

    with get_session() as session:
        result_proxy = session.execute(text(sql), params)
        rows = result_proxy.fetchall()

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows, columns=["trade_date", "equal_weight_return"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")
    return df["equal_weight_return"]


def _load_index_return(ts_code: str, start_date: str, end_date: str) -> pd.Series:
    """在线拉取指数日线，返回日收益率序列"""
    try:
        from tushare_data.fetcher import fetch_market_data
        df = fetch_market_data(ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            logger.warning("未获取到 %s 数据", ts_code)
            return pd.Series(dtype=float)
        df["pct_chg"] = df["pct_chg"].astype(float) / 100.0
        s = df.set_index("trade_date")["pct_chg"]
        s.index = pd.to_datetime(s.index)
        s.index.name = "trade_date"
        return s
    except Exception as exc:
        logger.error("拉取 %s 失败: %s", ts_code, exc)
        return pd.Series(dtype=float)


def validate_label_returns(
    labeled_df: pd.DataFrame,
    forward_days: List[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    按主标签分组，统计未来 N 日收益。

    Parameters
    ----------
    labeled_df : pd.DataFrame
        classify_market_state() 输出
    forward_days : list
        前向天数，默认 [1, 3, 5]
    start_date : str, optional
        起始日期
    end_date : str, optional
        结束日期

    Returns
    -------
    pd.DataFrame
        每行: (label, forward_days, metric, value)
    """
    if forward_days is None:
        forward_days = [1, 3, 5]

    if labeled_df.empty:
        return pd.DataFrame()

    logger.info("计算全A等权收益...")
    all_a_ret = compute_all_a_equal_weight_return(start_date, end_date)

    logger.info("拉取沪深300收益...")
    hs300_ret = _load_index_return("000300.SH", start_date or "20200101", end_date or "20261231")

    logger.info("拉取中证1000收益...")
    zz1000_ret = _load_index_return("000852.SH", start_date or "20200101", end_date or "20261231")

    ret_dict = {
        "全A等权": all_a_ret,
        "沪深300": hs300_ret,
        "中证1000": zz1000_ret,
    }

    if not hs300_ret.empty and not zz1000_ret.empty:
        common_idx = hs300_ret.index.intersection(zz1000_ret.index)
        size_spread = (zz1000_ret.loc[common_idx] - hs300_ret.loc[common_idx])
        size_spread.index.name = "trade_date"
        ret_dict["小盘超额(1000-300)"] = size_spread

    results = []

    for label_name, label_col in [("main_label", "main_label"), ("main_label_v2", "main_label_v2"), ("size_style_label", "size_style_label")]:
        if label_col not in labeled_df.columns:
            continue

        for label_val in labeled_df[label_col].unique():
            if pd.isna(label_val):
                continue

            label_dates = labeled_df[labeled_df[label_col] == label_val].index

            for ret_name, ret_series in ret_dict.items():
                if ret_series.empty:
                    continue

                for fd in forward_days:
                    fwd_ret = ret_series.rolling(fd).sum().shift(-fd + 1)

                    matched_fwd = fwd_ret.reindex(label_dates).dropna()

                    if len(matched_fwd) < 3:
                        continue

                    mean_ret = matched_fwd.mean()
                    std_ret = matched_fwd.std()
                    win_rate = (matched_fwd > 0).mean()
                    profit_loss = matched_fwd[matched_fwd > 0].mean() / abs(matched_fwd[matched_fwd <= 0].mean()) if (matched_fwd <= 0).any() else np.nan

                    results.append({
                        "label_type": label_col,
                        "label": label_val,
                        "return_type": ret_name,
                        "forward_days": fd,
                        "count": len(matched_fwd),
                        "mean": mean_ret,
                        "std": std_ret,
                        "win_rate": win_rate,
                        "profit_loss_ratio": profit_loss,
                    })

    return pd.DataFrame(results)


def validate_event_continuation(
    labeled_df: pd.DataFrame,
    events_df: pd.DataFrame = None,
    daily_rates: pd.DataFrame = None,
    forward_days: List[int] = None,
) -> pd.DataFrame:
    """
    按主标签分组，统计未来 N 日各事件的平均触发率 vs 全样本均值。

    Parameters
    ----------
    labeled_df : pd.DataFrame
        classify_market_state() 输出
    events_df : pd.DataFrame, optional
        batch_processor 输出的个股事件表（全量模式用）
    daily_rates : pd.DataFrame, optional
        每日各事件触发率表（流式模式用，优先于 events_df）
        index=trade_date, columns=evt_* 列
    forward_days : list
        前向天数，默认 [1, 2]

    Returns
    -------
    pd.DataFrame
        每行: (label, forward_days, event, label_rate, overall_rate, ratio)
    """
    if forward_days is None:
        forward_days = [1, 2]

    if labeled_df.empty:
        return pd.DataFrame()

    if "main_label" not in labeled_df.columns:
        return pd.DataFrame()

    if daily_rates is None and (events_df is None or events_df.empty):
        return pd.DataFrame()

    if daily_rates is None:
        df = events_df
        df["trade_date"] = df.index.date if hasattr(df.index, "date") else pd.to_datetime(df.index).date

        evt_cols_present = [c for c in EVT_COLS if c in df.columns]
        if not evt_cols_present:
            return pd.DataFrame()

        daily_rates = df.groupby("trade_date")[evt_cols_present].mean()

    if daily_rates.empty:
        return pd.DataFrame()

    evt_cols_present = [c for c in EVT_COLS if c in daily_rates.columns]
    if not evt_cols_present:
        return pd.DataFrame()

    overall_mean = daily_rates.mean()

    results = []

    label_cols_to_check = ["main_label"]
    if "main_label_v2" in labeled_df.columns:
        label_cols_to_check.append("main_label_v2")

    for label_col in label_cols_to_check:
        for label_val in labeled_df[label_col].unique():
            if pd.isna(label_val):
                continue

            label_dates = labeled_df[labeled_df[label_col] == label_val].index

        for fd in forward_days:
            fwd_dates_list = []
            for d in label_dates:
                d_val = d.date() if hasattr(d, "date") else d
                for offset in range(1, fd + 1):
                    fwd_d = d_val + pd.Timedelta(days=offset)
                    fwd_dates_list.append(fwd_d)

            if not fwd_dates_list:
                continue

            fwd_dates = pd.to_datetime(fwd_dates_list)
            available = daily_rates.index[daily_rates.index.isin(fwd_dates)]

            if len(available) < 3:
                continue

            label_mean = daily_rates.loc[available].mean()

            for evt in evt_cols_present:
                results.append({
                    "label_type": label_col,
                    "label": label_val,
                    "forward_days": fd,
                    "event": evt,
                    "label_rate": label_mean[evt],
                    "overall_rate": overall_mean[evt],
                    "ratio": label_mean[evt] / overall_mean[evt] if overall_mean[evt] > 0 else np.nan,
                })

    return pd.DataFrame(results)


def generate_validation_report(
    label_returns: pd.DataFrame,
    event_continuation: pd.DataFrame,
) -> str:
    """
    生成验证报告。

    Parameters
    ----------
    label_returns : pd.DataFrame
        validate_label_returns() 输出
    event_continuation : pd.DataFrame
        validate_event_continuation() 输出

    Returns
    -------
    str
        验证报告文本
    """
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"市场结构性分析 — 标签验证报告")
    lines.append(f"{'='*60}")

    if not label_returns.empty:
        lines.append(f"")
        lines.append(f"【一、主标签后 N 日收益统计】")
        main_ret = label_returns[label_returns["label_type"] == "main_label"]
        if not main_ret.empty:
            for fd in sorted(main_ret["forward_days"].unique()):
                lines.append(f"")
                lines.append(f"--- 前向 {fd} 日 ---")
                sub = main_ret[main_ret["forward_days"] == fd]
                for ret_type in sub["return_type"].unique():
                    lines.append(f"  [{ret_type}]")
                    rt_sub = sub[sub["return_type"] == ret_type]
                    for _, r in rt_sub.iterrows():
                        lines.append(
                            f"    {r['label']}: 均值{r['mean']:+.4f} "
                            f"标准差{r['std']:.4f} "
                            f"胜率{r['win_rate']:.1%} "
                            f"盈亏比{r['profit_loss_ratio']:.2f} "
                            f"(n={int(r['count'])})"
                        )

        lines.append(f"")
        lines.append(f"【二、v2 五档标签后 N 日收益统计】")
        v2_ret = label_returns[label_returns["label_type"] == "main_label_v2"]
        if not v2_ret.empty:
            for fd in sorted(v2_ret["forward_days"].unique()):
                lines.append(f"")
                lines.append(f"--- 前向 {fd} 日 ---")
                sub = v2_ret[v2_ret["forward_days"] == fd]
                for ret_type in ["全A等权"]:
                    rt_sub = sub[sub["return_type"] == ret_type]
                    if rt_sub.empty:
                        continue
                    lines.append(f"  [{ret_type}]")
                    label_order = ["强扩张", "弱扩张", "中性", "弱退潮", "强退潮"]
                    for lbl in label_order:
                        row = rt_sub[rt_sub["label"] == lbl]
                        if row.empty:
                            continue
                        r = row.iloc[0]
                        lines.append(
                            f"    {r['label']}: 均值{r['mean']:+.4f} "
                            f"标准差{r['std']:.4f} "
                            f"胜率{r['win_rate']:.1%} "
                            f"盈亏比{r['profit_loss_ratio']:.2f} "
                            f"(n={int(r['count'])})"
                        )

        lines.append(f"")
        lines.append(f"【三、风格副标签后 N 日收益统计】")
        style_ret = label_returns[label_returns["label_type"] == "size_style_label"]
        if not style_ret.empty:
            for fd in sorted(style_ret["forward_days"].unique()):
                lines.append(f"")
                lines.append(f"--- 前向 {fd} 日 ---")
                sub = style_ret[style_ret["forward_days"] == fd]
                for ret_type in ["小盘超额(1000-300)"]:
                    rt_sub = sub[sub["return_type"] == ret_type]
                    if rt_sub.empty:
                        continue
                    lines.append(f"  [{ret_type}]")
                    for _, r in rt_sub.iterrows():
                        lines.append(
                            f"    {r['label']}: 均值{r['mean']:+.4f} "
                            f"胜率{r['win_rate']:.1%} "
                            f"(n={int(r['count'])})"
                        )

    if not event_continuation.empty:
        lines.append(f"")
        lines.append(f"【四、主标签后事件延续性】")
        for fd in sorted(event_continuation["forward_days"].unique()):
            lines.append(f"")
            lines.append(f"--- 前向 {fd} 日 ---")
            sub = event_continuation[event_continuation["forward_days"] == fd]
            for label in sub["label"].unique():
                lines.append(f"  [{label}]")
                l_sub = sub[sub["label"] == label]
                significant = l_sub[abs(l_sub["ratio"] - 1.0) > 0.1]
                for _, r in significant.sort_values("ratio", ascending=False).iterrows():
                    direction = "↑" if r["ratio"] > 1 else "↓"
                    lines.append(
                        f"    {r['event']}: {r['label_rate']:.4f} vs {r['overall_rate']:.4f} "
                        f"(x{r['ratio']:.2f} {direction})"
                    )

    lines.append(f"")
    lines.append(f"{'='*60}")
    lines.append(f"【结论】")
    if not label_returns.empty:
        main_ret = label_returns[label_returns["label_type"] == "main_label"]
        if not main_ret.empty:
            best = main_ret.loc[main_ret["mean"].idxmax()]
            worst = main_ret.loc[main_ret["mean"].idxmin()]
            lines.append(f"  v1 最强标签: {best['label']} → {best['return_type']} 前{int(best['forward_days'])}日 均值{best['mean']:+.4f}")
            lines.append(f"  v1 最弱标签: {worst['label']} → {worst['return_type']} 前{int(worst['forward_days'])}日 均值{worst['mean']:+.4f}")

            expansion_ret = main_ret[(main_ret["label"] == "扩张") & (main_ret["forward_days"] == 1) & (main_ret["return_type"] == "全A等权")]
            retreat_ret = main_ret[(main_ret["label"] == "退潮") & (main_ret["forward_days"] == 1) & (main_ret["return_type"] == "全A等权")]
            if not expansion_ret.empty and not retreat_ret.empty:
                diff = expansion_ret["mean"].values[0] - retreat_ret["mean"].values[0]
                lines.append(f"  v1 扩张 vs 退潮 次日全A收益差: {diff:+.4f}")

        v2_ret = label_returns[label_returns["label_type"] == "main_label_v2"]
        if not v2_ret.empty:
            v2_fd1 = v2_ret[(v2_ret["forward_days"] == 1) & (v2_ret["return_type"] == "全A等权")]
            if not v2_fd1.empty:
                strong_exp = v2_fd1[v2_fd1["label"] == "强扩张"]
                weak_exp = v2_fd1[v2_fd1["label"] == "弱扩张"]
                neutral = v2_fd1[v2_fd1["label"] == "中性"]
                weak_ret = v2_fd1[v2_fd1["label"] == "弱退潮"]
                strong_ret = v2_fd1[v2_fd1["label"] == "强退潮"]

                if not strong_exp.empty and not strong_ret.empty:
                    v2_diff = strong_exp["mean"].values[0] - strong_ret["mean"].values[0]
                    lines.append(f"  v2 强扩张 vs 强退潮 次日全A收益差: {v2_diff:+.4f}")

                if not neutral.empty:
                    lines.append(f"  v2 中性标签占比: {neutral['count'].values[0]}/{v2_fd1['count'].sum():.0f}")

                is_monotonic = True
                label_order = ["强扩张", "弱扩张", "中性", "弱退潮", "强退潮"]
                means = []
                for lbl in label_order:
                    sub = v2_fd1[v2_fd1["label"] == lbl]
                    if not sub.empty:
                        means.append(sub["mean"].values[0])
                    else:
                        means.append(None)
                valid_means = [m for m in means if m is not None]
                if len(valid_means) >= 3:
                    for i in range(1, len(valid_means)):
                        if valid_means[i] > valid_means[i-1]:
                            is_monotonic = False
                            break
                    lines.append(f"  v2 收益单调性: {'✓ 单调递减' if is_monotonic else '✗ 非单调'}")

    lines.append(f"{'='*60}")

    return "\n".join(lines)


def validate_score_distribution(
    labeled_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, str]:
    """
    统计 score_z20 的历史分布，校准阈值，判断标签是否过松。

    Parameters
    ----------
    labeled_df : pd.DataFrame
        classify_market_state() 输出，需含 {score}_z20 列和 main_label 列

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (distribution_df, calibration_report)
        distribution_df: 每行 = (score_name, stat_name, value)
        calibration_report: 阈值校准结论文本
    """
    from market_structure_analysis.market_aggregator import SCORE_NAMES

    if labeled_df.empty:
        return pd.DataFrame(), "无数据"

    z_cols = [f"{s}_z20" for s in SCORE_NAMES]
    z_present = [c for c in z_cols if c in labeled_df.columns]

    if not z_present:
        return pd.DataFrame(), "无 z20 列"

    rows = []
    for col in z_present:
        s = labeled_df[col].dropna()
        if s.empty:
            continue
        base = col.replace("_z20", "")
        for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
            rows.append({"score_name": base, "stat_name": f"p{int(q*100)}", "value": s.quantile(q)})
        rows.append({"score_name": base, "stat_name": "mean", "value": s.mean()})
        rows.append({"score_name": base, "stat_name": "std", "value": s.std()})

        for label in labeled_df["main_label"].unique():
            if pd.isna(label):
                continue
            sub = labeled_df[labeled_df["main_label"] == label][col].dropna()
            if len(sub) < 3:
                continue
            rows.append({"score_name": base, "stat_name": f"mean_{label}", "value": sub.mean()})

    dist_df = pd.DataFrame(rows)

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"【四、分数阈值校准】")
    lines.append(f"")

    for col in z_present:
        base = col.replace("_z20", "")
        s = labeled_df[col].dropna()
        if s.empty:
            continue
        p10, p25, p50, p75, p90 = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        lines.append(f"  {base}_z20:")
        lines.append(f"    分布: p10={p10:+.2f} p25={p25:+.2f} p50={p50:+.2f} p75={p75:+.2f} p90={p90:+.2f}")
        lines.append(f"    均值={s.mean():+.2f} 标准差={s.std():.2f}")

        strong_thresh = p90
        weak_thresh = p10
        strong_pct = (s > strong_thresh).mean() * 100
        weak_pct = (s < weak_thresh).mean() * 100
        lines.append(f"    建议阈值: 强(>{strong_thresh:+.2f}, 占{strong_pct:.0f}%) 弱(<{weak_thresh:+.2f}, 占{weak_pct:.0f}%)")

    if "main_label" in labeled_df.columns and "confidence" in labeled_df.columns:
        low_conf = labeled_df[labeled_df["confidence"] < CONFIDENCE_LOW_THRESHOLD]
        low_conf_pct = len(low_conf) / len(labeled_df) * 100
        lines.append(f"")
        lines.append(f"  置信度 < {CONFIDENCE_LOW_THRESHOLD} 的占比: {low_conf_pct:.1f}% ({len(low_conf)}/{len(labeled_df)})")
        if low_conf_pct > 40:
            lines.append(f"  → 标签阈值偏松，建议引入中性层（强/弱/中性）")
        elif low_conf_pct > 20:
            lines.append(f"  → 标签阈值略松，部分边界样本可考虑归入中性")
        else:
            lines.append(f"  → 标签阈值合理，低置信度样本占比较小")

    lines.append(f"{'='*60}")

    return dist_df, "\n".join(lines)


def validate_group_stability(
    labeled_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    min_concept_sample: int = 10,
) -> Tuple[pd.DataFrame, str]:
    """
    验证行业/市值分组排名的稳定性。

    Parameters
    ----------
    labeled_df : pd.DataFrame
        classify_market_state() 输出
    grouped_df : pd.DataFrame
        aggregate_by_group() 输出
    min_concept_sample : int
        概念组最小样本数

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (stability_df, stability_report)
    """
    if grouped_df.empty or labeled_df.empty:
        return pd.DataFrame(), "无分组数据"

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"【五、分组稳定性检验】")
    lines.append(f"")

    ind_sub = grouped_df[grouped_df["group_type"] == "industry_l2"]
    if not ind_sub.empty and "trade_date" in ind_sub.columns:
        from market_structure_analysis._config import MIN_INDUSTRY_SAMPLE

        ind_filtered = ind_sub[ind_sub["stock_count"] >= MIN_INDUSTRY_SAMPLE]

        dates = sorted(ind_filtered["trade_date"].unique())
        overlaps = []
        for i in range(1, len(dates)):
            prev = ind_filtered[ind_filtered["trade_date"] == dates[i - 1]].nlargest(5, "structure_score")["group_name"]
            curr = ind_filtered[ind_filtered["trade_date"] == dates[i]].nlargest(5, "structure_score")["group_name"]
            overlap = len(set(prev) & set(curr))
            overlaps.append(overlap / 5.0)

        if overlaps:
            avg_overlap = np.mean(overlaps)
            lines.append(f"  行业 Top5 日间重叠率: {avg_overlap:.1%} (均值)")
            if avg_overlap < 0.3:
                lines.append(f"  → 行业排名极不稳定，Top5 跳变严重")
            elif avg_overlap < 0.5:
                lines.append(f"  → 行业排名较不稳定，建议加长回看窗口")
            else:
                lines.append(f"  → 行业排名较稳定，Top5 延续性可接受")
        else:
            avg_overlap = np.nan

        small_sample = ind_sub[ind_sub["stock_count"] < MIN_INDUSTRY_SAMPLE]
        if not small_sample.empty:
            lines.append(f"  样本不足({MIN_INDUSTRY_SAMPLE}只)的行业组: {small_sample['group_name'].nunique()} 个")

    concept_sub = grouped_df[grouped_df["group_type"] == "concept"]
    if not concept_sub.empty and "stock_count" in concept_sub.columns:
        small_concept = concept_sub[concept_sub["stock_count"] < min_concept_sample]
        if not small_concept.empty:
            lines.append(f"  样本不足({min_concept_sample}只)的概念组: {small_concept['group_name'].nunique()} 个")

    lines.append(f"{'='*60}")

    stability_df = pd.DataFrame([{
        "avg_top5_overlap": avg_overlap if 'avg_overlap' in dir() else np.nan,
    }])

    return stability_df, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="市场验证器")
    parser.add_argument("--start", type=str, default="2024-01-01", help="起始日期")
    parser.add_argument("--end", type=str, default="2026-04-30", help="结束日期")
    parser.add_argument("--freq", type=str, default="d", help="K线周期")
    parser.add_argument("--limit_stocks", type=int, default=None, help="限制股票数量")
    parser.add_argument("--include_concept", action="store_true", help="包含概念分组")
    parser.add_argument("--top_themes", type=int, default=10, help="输出 Top N 主线题材")
    parser.add_argument("--cache-dir", type=str, default="/tmp/market_cache", help="中间结果缓存目录（默认 /tmp/market_cache）")
    parser.add_argument("--skip-pavp", action="store_true", help="跳过 PAVP 计算（速度提升 4x，丢失 2 事件+1 状态）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from market_structure_analysis.market_aggregator import run_aggregation
    from market_structure_analysis.market_state_classifier import classify_market_state

    logger.info("Step 1: 运行聚合+分类...")
    daily, grouped, events_df, lightweight_events = run_aggregation(
        start_date=args.start,
        end_date=args.end,
        freq=args.freq,
        limit_stocks=args.limit_stocks,
        include_concept=True,
        cache_dir=args.cache_dir,
        skip_pavp=args.skip_pavp,
    )

    if daily.empty:
        logger.error("聚合无结果")
        return

    labeled = classify_market_state(daily, grouped)

    logger.info("Step 2: 验证标签后收益...")
    label_returns = validate_label_returns(
        labeled, start_date=args.start, end_date=args.end,
    )

    logger.info("Step 3: 验证事件延续性...")
    rate_cols = [c for c in daily.columns if c.endswith("_rate") and c.startswith("evt_")]
    if rate_cols and not events_df.empty:
        event_cont = validate_event_continuation(labeled, events_df=events_df)
    elif rate_cols:
        daily_rates = daily[rate_cols]
        event_cont = validate_event_continuation(labeled, daily_rates=daily_rates)
    else:
        event_cont = pd.DataFrame()

    logger.info("Step 4: 分数阈值校准...")
    _, score_report = validate_score_distribution(labeled)

    logger.info("Step 5: 分组稳定性检验...")
    _, stability_report = validate_group_stability(labeled, grouped)

    report = generate_validation_report(label_returns, event_cont)
    print(report)
    print(score_report)
    print(stability_report)

    from market_structure_analysis.market_state_classifier import export_daily_summary
    from market_structure_analysis.regime_filter import generate_regime_summary

    csv_path = "/tmp/market_daily_summary.csv"
    labeled_out = export_daily_summary(labeled, grouped, output_path=csv_path, fmt="csv")
    logger.info("标签 CSV 已保存: %s (%d 行)", csv_path, len(labeled_out))

    print(generate_regime_summary(labeled["main_label_v2"]))

    concept_grouped = pd.DataFrame()
    try:
        from market_structure_analysis.theme_aggregator import (
            generate_theme_report, rank_daily_themes, identify_theme_trends,
        )
        concept_grouped = grouped[grouped["group_type"] == "concept"]
        if len(concept_grouped) > 0 and "structure_score" in concept_grouped.columns:
            import numpy as np
            concept_grouped = concept_grouped.copy()
            concept_grouped["composite_strength"] = (
                concept_grouped["structure_score"] * np.sqrt(concept_grouped["stock_count"].clip(lower=1))
            )
            theme_report = generate_theme_report(concept_grouped, top_n=args.top_themes or 10)
            print(theme_report)
        else:
            logger.warning("无概念分组数据")
    except Exception as e:
        logger.warning("题材聚合失败: %s", e)

    logger.info("Step 6: 计算指数锚点...")
    try:
        from market_structure_analysis.index_context import (
            compute_index_factors, extract_index_events, classify_index_state,
        )
        index_factors = compute_index_factors(start_date=args.start, end_date=args.end)
        index_events = extract_index_events(index_factors)
        index_states = classify_index_state(index_events)
        if not index_states.empty:
            from market_structure_analysis.index_context import generate_index_context_report
            print(generate_index_context_report(index_states, index_events))
    except Exception as e:
        logger.warning("指数锚点计算失败: %s", e)
        index_states = pd.DataFrame()

    logger.info("Step 7: 计算量能背景...")
    try:
        from market_structure_analysis.volume_context import (
            compute_volume_breadth, compute_volume_by_style,
            generate_volume_context_report,
        )
        from market_structure_analysis.market_aggregator import load_stock_attributes
        vol_breadth = compute_volume_breadth(daily)
        stock_attrs = load_stock_attributes()
        vol_style = compute_volume_by_style(lightweight_events, stock_attrs)
        print(generate_volume_context_report(vol_breadth, vol_style))
    except Exception as e:
        logger.warning("量能背景计算失败: %s", e)
        vol_breadth = pd.DataFrame()

    logger.info("Step 8: 生成增强日报...")
    try:
        from market_structure_analysis.enhanced_daily_report import (
            build_enhanced_daily_report, export_enhanced_report_csv,
        )
        theme_text = ""
        try:
            if len(concept_grouped) > 0 and "structure_score" in concept_grouped.columns:
                theme_text = generate_theme_report(concept_grouped, top_n=args.top_themes or 10)
        except Exception:
            pass
        from market_structure_analysis.regime_filter import REGIME_RULES
        enhanced_report = build_enhanced_daily_report(
            labeled, index_states, vol_breadth, theme_text, REGIME_RULES,
        )
        print(enhanced_report)
        export_enhanced_report_csv(labeled, index_states, vol_breadth,
                                   output_path="/tmp/market_enhanced_daily.csv")
    except Exception as e:
        logger.warning("增强日报生成失败: %s", e)


if __name__ == "__main__":
    main()
