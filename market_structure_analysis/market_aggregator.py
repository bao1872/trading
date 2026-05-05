"""
market_aggregator.py
全市场聚合器 — 将个股事件表聚合为市场级统计表

Purpose:
    将 batch_processor 输出的个股事件表，按交易日聚合为市场级统计表，
    支持全市场聚合和按行业/市值分组聚合，计算 5 个市场状态指标及滚动基线。

Inputs:
    - events_df: pd.DataFrame, batch_processor 输出的个股因子+事件表
    - stock_pools 表: 行业/市值/概念属性

Outputs:
    - DailyMarketSummary: 每日全市场事件统计 + 5 个状态指标 + 滚动 zscore
    - GroupedMarketSummary: 按行业/市值分组的事件统计

How to Run:
    python market_structure_analysis/market_aggregator.py --start 2024-06-01 --end 2024-12-31 --limit_stocks 10

Examples:
    python market_structure_analysis/market_aggregator.py --limit_stocks 10 --start 2024-06-01
    python market_structure_analysis/market_aggregator.py --start 2024-01-01 --end 2024-12-31 --include_concept

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
from market_structure_analysis.event_detector import CORE_EVENTS, AUX_STATES

logger = logging.getLogger(__name__)

EVT_COLS = [f"evt_{e}" for e in CORE_EVENTS]

BULL_EVT_COLS = [
    "evt_dsa_dir_flip_up",
    "evt_cross_above_dsa_vwap",
    "evt_bbmacd_cross_upper",
    "evt_up_move_with_vol_spike",
    "evt_cross_above_value_area_high",
    "evt_break_sell_stop_cluster",
]

BEAR_EVT_COLS = [
    "evt_dsa_dir_flip_down",
    "evt_cross_below_dsa_vwap",
    "evt_bbmacd_cross_lower",
    "evt_down_move_with_vol_spike",
    "evt_cross_below_value_area_low",
    "evt_break_buy_stop_cluster",
]

SCORE_NAMES = [
    "structure_score",
    "momentum_score",
    "volume_confirm_score",
    "breakout_score",
    "stop_flow_score",
]

ZSCORE_WINDOW = 20

CAP_TIERS: Dict[str, Tuple[float, float]] = {
    "mega_cap": (200e9, float("inf")),
    "large_cap": (50e9, 200e9),
    "mid_cap": (10e9, 50e9),
    "small_cap": (0, 10e9),
}

CAP_TIER_LABELS = {
    "mega_cap": "超大盘(>2000亿)",
    "large_cap": "大盘(500-2000亿)",
    "mid_cap": "中盘(100-500亿)",
    "small_cap": "小盘(<100亿)",
}

GENERIC_CONCEPTS = {
    "融资融券", "沪股通", "深股通", "标普道琼斯A股", "MSCI概念",
    "富时罗素概念", "新股与次新股", "注册制次新股", "转融券标的",
    "央企国企改革", "地方国企改革", "预盈预增", "预亏预减",
}

MIN_INDUSTRY_SAMPLE = 20


def _assign_cap_tier(total_market_cap: pd.Series) -> pd.Series:
    """根据总市值分桶"""
    conditions = []
    choices = []
    for tier, (lo, hi) in CAP_TIERS.items():
        conditions.append((total_market_cap >= lo) & (total_market_cap < hi))
        choices.append(tier)
    return np.select(conditions, choices, default="small_cap").astype(object)


def load_stock_attributes() -> pd.DataFrame:
    """
    从 stock_pools 加载股票属性表（行业/市值/概念），并计算市值分桶。

    Returns
    -------
    pd.DataFrame
        列: ts_code, name, industry_l2, market_cap, total_market_cap, cap_tier, concepts
    """
    with get_session() as session:
        df = query_df(
            session,
            table_name="stock_pools",
            columns=["ts_code", "name", "industry_l2", "industry_l3",
                      "market_cap", "total_market_cap", "concepts"],
        )

    if df.empty:
        return df

    df["cap_tier"] = _assign_cap_tier(df["total_market_cap"])
    return df


def _compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """计算 5 个市场状态指标 + 多空比（PAVP 缺失时 breakout_score 置 0）"""
    df = df.copy()

    df["structure_score"] = (
        df["evt_dsa_dir_flip_up_rate"] + df["evt_cross_above_dsa_vwap_rate"]
    ) - (
        df["evt_dsa_dir_flip_down_rate"] + df["evt_cross_below_dsa_vwap_rate"]
    )

    df["momentum_score"] = (
        df["evt_bbmacd_cross_upper_rate"] - df["evt_bbmacd_cross_lower_rate"]
    )

    df["volume_confirm_score"] = (
        df["evt_up_move_with_vol_spike_rate"] - df["evt_down_move_with_vol_spike_rate"]
    )

    if "evt_cross_above_value_area_high_rate" in df.columns:
        df["breakout_score"] = (
            df["evt_cross_above_value_area_high_rate"] - df["evt_cross_below_value_area_low_rate"]
        )
    else:
        df["breakout_score"] = 0.0

    df["stop_flow_score"] = (
        df["evt_break_sell_stop_cluster_rate"] - df["evt_break_buy_stop_cluster_rate"]
    )

    bull = (df["evt_dsa_dir_flip_up_rate"] + df["evt_cross_above_dsa_vwap_rate"]
            + df["evt_bbmacd_cross_upper_rate"])
    bear = (df["evt_dsa_dir_flip_down_rate"] + df["evt_cross_below_dsa_vwap_rate"]
            + df["evt_bbmacd_cross_lower_rate"])
    df["bull_bear_ratio"] = bull / bear.clip(lower=1e-6)

    return df


def compute_breadth_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算事件广度面板：5 个 breadth 指标。

    Parameters
    ----------
    df : pd.DataFrame
        含 evt_*_rate 列和 total_stocks 列的日级聚合表

    Returns
    -------
    pd.DataFrame
        追加 bull_event_rate / bear_event_rate / breadth_diff / stock_coverage / event_concentration 列
    """
    bull_rate_cols = [f"{c}_rate" for c in BULL_EVT_COLS if f"{c}_rate" in df.columns]
    bear_rate_cols = [f"{c}_rate" for c in BEAR_EVT_COLS if f"{c}_rate" in df.columns]

    if bull_rate_cols:
        df["bull_event_rate"] = df[bull_rate_cols].sum(axis=1)
    else:
        df["bull_event_rate"] = 0.0

    if bear_rate_cols:
        df["bear_event_rate"] = df[bear_rate_cols].sum(axis=1)
    else:
        df["bear_event_rate"] = 0.0

    df["breadth_diff"] = df["bull_event_rate"] - df["bear_event_rate"]

    all_rate_cols = [c for c in df.columns if c.endswith("_rate") and c.startswith("evt_")]
    if all_rate_cols and "total_stocks" in df.columns:
        any_evt = (df[all_rate_cols] > 0).any(axis=1)
        df["stock_coverage"] = np.where(any_evt, 1.0, 0.0)
    else:
        df["stock_coverage"] = np.nan

    if all_rate_cols:
        total = df[all_rate_cols].sum(axis=1).replace(0, np.nan)
        shares = df[all_rate_cols].div(total, axis=0)
        df["event_concentration"] = (shares ** 2).sum(axis=1)
    else:
        df["event_concentration"] = np.nan

    return df


def _add_rolling_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """为 score 列增加滚动 zscore，为 rate 列增加滚动均值"""
    df = df.copy()

    for score in SCORE_NAMES:
        if score not in df.columns:
            continue
        rolling = df[score].rolling(window=ZSCORE_WINDOW, min_periods=5)
        ma = rolling.mean()
        std = rolling.std()
        df[f"{score}_z20"] = (df[score] - ma) / std.replace(0, np.nan)

    rate_cols = [c for c in df.columns if c.endswith("_rate") and c not in SCORE_NAMES]
    for rc in rate_cols:
        df[f"{rc}_ma20"] = df[rc].rolling(window=ZSCORE_WINDOW, min_periods=5).mean()

    return df


def compute_dominant_driver(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 z20 列判定每日主导因子和次主导因子。

    Parameters
    ----------
    daily_df : pd.DataFrame
        aggregate_daily() 输出，需含 {score}_z20 列

    Returns
    -------
    pd.DataFrame
        追加 dominant_driver, secondary_driver, driver_strength_rank 列
    """
    df = daily_df.copy()

    z_cols = {s: f"{s}_z20" for s in SCORE_NAMES}
    z_present = {k: v for k, v in z_cols.items() if v in df.columns}

    if not z_present:
        df["dominant_driver"] = "unknown"
        df["secondary_driver"] = "unknown"
        df["driver_strength_rank"] = ""
        return df

    z_df = df[list(z_present.values())].copy()
    z_df.columns = list(z_present.keys())

    abs_z = z_df.abs()

    all_nan_mask = abs_z.isna().all(axis=1)

    rank_df = abs_z.rank(axis=1, ascending=False, na_option="keep")

    df["dominant_driver"] = "unknown"
    df["secondary_driver"] = "unknown"
    df["driver_strength_rank"] = ""

    valid_mask = ~all_nan_mask
    if valid_mask.any():
        df.loc[valid_mask, "dominant_driver"] = abs_z.loc[valid_mask].idxmax(axis=1)
        df.loc[valid_mask, "secondary_driver"] = abs_z.loc[valid_mask].apply(
            lambda row: row.drop(row.idxmax()).idxmax() if row.notna().sum() >= 2 else "unknown",
            axis=1,
        )
        df.loc[valid_mask, "driver_strength_rank"] = rank_df.loc[valid_mask].apply(
            lambda row: ",".join(row.dropna().sort_values().index.astype(str)),
            axis=1,
        )

    return df


def aggregate_daily(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    按交易日聚合全市场事件统计。

    Parameters
    ----------
    events_df : pd.DataFrame
        batch_processor 输出的个股事件表，需含 ts_code 列和 evt_* 列

    Returns
    -------
    pd.DataFrame
        DailyMarketSummary: 每行一个交易日，含事件触发数/率 + 5 个状态指标 + 滚动基线
    """
    if events_df.empty:
        return pd.DataFrame()

    trade_dates = events_df.index.date if hasattr(events_df.index, "date") else pd.to_datetime(events_df.index).date

    evt_cols_present = [c for c in EVT_COLS if c in events_df.columns]
    if not evt_cols_present:
        logger.warning("事件表中无 evt_* 列")
        return pd.DataFrame()

    grouped = events_df.groupby(trade_dates)
    counts = grouped[evt_cols_present].sum()
    counts.columns = [c + "_count" for c in counts.columns]

    total_stocks = grouped["ts_code"].nunique().rename("total_stocks")

    result = pd.concat([total_stocks, counts], axis=1)

    rate_cols = {}
    for col in counts.columns:
        base = col.replace("_count", "")
        rate_cols[base + "_rate"] = result[col] / result["total_stocks"]
    rates = pd.DataFrame(rate_cols, index=result.index)
    result = pd.concat([result, rates], axis=1)

    result = _compute_scores(result)
    result = compute_breadth_panel(result)
    result = _add_rolling_baseline(result)
    result = compute_dominant_driver(result)

    result.index = pd.to_datetime(result.index)
    result.index.name = "trade_date"

    return result


def aggregate_daily_from_counts(daily_agg: pd.DataFrame) -> pd.DataFrame:
    """
    从流式累加的日级计数表构建 DailyMarketSummary（无需完整 events_df）。

    Parameters
    ----------
    daily_agg : pd.DataFrame
        process_stock_pool_streaming() 输出的日级聚合表，
        需含 total_stocks 列和 evt_*_count 列，index 为 trade_date

    Returns
    -------
    pd.DataFrame
        DailyMarketSummary: 含事件触发数/率 + 5 个状态指标 + 滚动基线
    """
    if daily_agg.empty:
        return pd.DataFrame()

    result = daily_agg.copy()

    count_cols = [c for c in result.columns if c.endswith("_count")]
    rate_cols = {}
    for col in count_cols:
        base = col.replace("_count", "")
        rate_cols[base + "_rate"] = result[col] / result["total_stocks"]
    rates = pd.DataFrame(rate_cols, index=result.index)
    result = pd.concat([result, rates], axis=1)

    result = _compute_scores(result)
    result = compute_breadth_panel(result)
    result = _add_rolling_baseline(result)
    result = compute_dominant_driver(result)

    result.index.name = "trade_date"

    return result


def aggregate_by_group(
    events_df: pd.DataFrame,
    group_type: str,
    stock_attrs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    按指定维度分组聚合事件统计。

    Parameters
    ----------
    events_df : pd.DataFrame
        个股事件表（完整或轻量版均可，需含 ts_code/trade_date/evt_* 列）
    group_type : str
        'industry_l2' | 'cap_tier' | 'concept'
    stock_attrs : pd.DataFrame, optional
        股票属性表，None 则自动加载

    Returns
    -------
    pd.DataFrame
        GroupedMarketSummary
    """
    if events_df.empty:
        return pd.DataFrame()

    if stock_attrs is None:
        stock_attrs = load_stock_attributes()

    if "trade_date" not in events_df.columns:
        trade_dates = events_df.index.date if hasattr(events_df.index, "date") else pd.to_datetime(events_df.index).date
        df = events_df.assign(trade_date=trade_dates)
    else:
        df = events_df

    if "ts_code" not in df.columns:
        logger.error("事件表缺少 ts_code 列")
        return pd.DataFrame()

    if group_type in ("industry_l2", "cap_tier"):
        attr_col = group_type
        if attr_col not in stock_attrs.columns:
            logger.error("stock_pools 表缺少 %s 列", attr_col)
            return pd.DataFrame()

        mapping = stock_attrs.set_index("ts_code")[attr_col].to_dict()
        df["_group"] = df["ts_code"].map(mapping)

    elif group_type == "concept":
        if "concepts" not in stock_attrs.columns:
            logger.error("stock_pools 表缺少 concepts 列")
            return pd.DataFrame()

        concept_map = (
            stock_attrs[["ts_code", "concepts"]]
            .dropna(subset=["concepts"])
            .assign(concepts=lambda x: x["concepts"].str.split(";"))
            .explode("concepts")
        )
        concept_map["concepts"] = concept_map["concepts"].str.strip()
        concept_map = concept_map[~concept_map["concepts"].isin(GENERIC_CONCEPTS)]
        concept_map = concept_map[concept_map["concepts"] != ""]

        df = df.merge(
            concept_map.rename(columns={"concepts": "_group"}),
            on="ts_code",
            how="inner",
        )
    else:
        raise ValueError(f"不支持的 group_type: {group_type}")

    if "_group" not in df.columns:
        logger.warning("分组列不存在，跳过 %s 聚合", group_type)
        return pd.DataFrame()
    df = df.dropna(subset=["_group"])

    evt_cols_present = [c for c in EVT_COLS if c in df.columns]
    if not evt_cols_present:
        return pd.DataFrame()

    grouped = df.groupby(["trade_date", "_group"])
    counts = grouped[evt_cols_present].sum()
    counts.columns = [c + "_count" for c in counts.columns]

    total_stocks = grouped["ts_code"].nunique().rename("stock_count")

    result = pd.concat([total_stocks, counts], axis=1)

    rate_cols = {}
    for col in counts.columns:
        base = col.replace("_count", "")
        rate_cols[base + "_rate"] = result[col] / result["stock_count"]
    rates = pd.DataFrame(rate_cols, index=result.index)
    result = pd.concat([result, rates], axis=1)

    result = _compute_scores(result)

    result = result.reset_index()
    result = result.rename(columns={"_group": "group_name"})
    result["group_type"] = group_type

    if group_type == "cap_tier":
        tier_order = list(CAP_TIERS.keys())
        result["group_name"] = pd.Categorical(
            result["group_name"], categories=tier_order, ordered=True
        )

    return result


def run_aggregation(
    start_date: str,
    end_date: str,
    freq: str = "d",
    limit_stocks: Optional[int] = None,
    include_concept: bool = False,
    streaming: bool = True,
    cache_dir: Optional[str] = None,
    skip_pavp: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    完整流程: 加载数据 → 计算因子 → 检测事件 → 聚合统计。

    Parameters
    ----------
    start_date : str
        起始日期
    end_date : str
        结束日期
    freq : str
        K线周期
    limit_stocks : int, optional
        限制股票数量
    include_concept : bool
        是否包含概念分组（默认不包含，第二阶段功能）
    streaming : bool
        是否使用流式模式（默认 True，内存友好）
    cache_dir : str, optional
        中间结果缓存目录。非空时：Step1 完成后保存 daily_agg + lightweight_events 到该目录，
        下次运行时若缓存文件存在则直接加载跳过重算。

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (daily_summary, grouped_summary, events_df_or_empty)
        streaming=True 时 events_df 为空 DataFrame（节省内存）
    """
    import pickle

    _cache_key = f"stream_{start_date}_{end_date}_{freq}_{limit_stocks or 'all'}"
    _agg_cache = None
    _evt_cache = None
    if cache_dir:
        import os as _os
        _os.makedirs(cache_dir, exist_ok=True)
        _agg_cache = _os.path.join(cache_dir, f"daily_agg_{_cache_key}.pkl")
        _evt_cache = _os.path.join(cache_dir, f"lightweight_events_{_cache_key}.pkl")

    if streaming:
        from market_structure_analysis.batch_processor import process_stock_pool_streaming

        daily_agg = None
        lightweight_events = None

        if cache_dir and _agg_cache is not None:
            try:
                with open(_agg_cache, "rb") as _f:
                    daily_agg = pickle.load(_f)
                with open(_evt_cache, "rb") as _f:
                    lightweight_events = pickle.load(_f)
                logger.info("从缓存加载中间结果: %s (%d 行), %s (%d 行)",
                            _agg_cache, len(daily_agg),
                            _evt_cache, len(lightweight_events))
            except Exception:
                logger.info("无可用缓存，重新计算...")
                daily_agg = None
                lightweight_events = None

        if daily_agg is None:
            logger.info("Step 1: 流式批量处理股票池...")
            daily_agg, lightweight_events = process_stock_pool_streaming(
                freq=freq,
                start_date=start_date,
                end_date=end_date,
                limit_stocks=limit_stocks,
                skip_pavp=skip_pavp,
            )

            if not daily_agg.empty and cache_dir and _agg_cache is not None:
                daily_agg.to_pickle(_agg_cache)
                lightweight_events.to_pickle(_evt_cache)
                logger.info("中间结果已缓存: %s, %s", _agg_cache, _evt_cache)

        if daily_agg.empty:
            logger.error("流式处理无结果")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        logger.info("Step 2: 从累加计数构建每日聚合...")
        daily_summary = aggregate_daily_from_counts(daily_agg)
        logger.info("每日聚合: %d 个交易日", len(daily_summary))

        logger.info("Step 3: 加载股票属性...")
        stock_attrs = load_stock_attributes()

        group_types = ["industry_l2", "cap_tier"]
        if include_concept:
            group_types.append("concept")

        grouped_list = []
        for gt in group_types:
            logger.info("Step 4: 按 %s 分组聚合...", gt)
            g = aggregate_by_group(lightweight_events, gt, stock_attrs)
            if not g.empty:
                grouped_list.append(g)
                logger.info("  %s: %d 行", gt, len(g))

        grouped_summary = pd.concat(grouped_list, ignore_index=True) if grouped_list else pd.DataFrame()

        return daily_summary, grouped_summary, pd.DataFrame(), lightweight_events
    else:
        from market_structure_analysis.batch_processor import process_stock_pool

        logger.info("Step 1: 批量处理股票池（全量模式）...")
        events_df = process_stock_pool(
            freq=freq,
            start_date=start_date,
            end_date=end_date,
            limit_stocks=limit_stocks,
        )

        if events_df.empty:
            logger.error("批量处理无结果")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        logger.info("Step 2: 每日全市场聚合...")
        daily_summary = aggregate_daily(events_df)
        logger.info("每日聚合: %d 个交易日", len(daily_summary))

        logger.info("Step 3: 加载股票属性...")
        stock_attrs = load_stock_attributes()

        group_types = ["industry_l2", "cap_tier"]
        if include_concept:
            group_types.append("concept")

        grouped_list = []
        for gt in group_types:
            logger.info("Step 4: 按 %s 分组聚合...", gt)
            g = aggregate_by_group(events_df, gt, stock_attrs)
            if not g.empty:
                grouped_list.append(g)
                logger.info("  %s: %d 行", gt, len(g))

        grouped_summary = pd.concat(grouped_list, ignore_index=True) if grouped_list else pd.DataFrame()

        return daily_summary, grouped_summary, events_df, lightweight_events


def main():
    parser = argparse.ArgumentParser(description="全市场聚合器")
    parser.add_argument("--start", type=str, default="2024-06-01", help="起始日期")
    parser.add_argument("--end", type=str, default="2024-12-31", help="结束日期")
    parser.add_argument("--freq", type=str, default="d", help="K线周期")
    parser.add_argument("--limit_stocks", type=int, default=None, help="限制股票数量")
    parser.add_argument("--include_concept", action="store_true", help="包含概念分组（第二阶段）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    daily, grouped, _, _ = run_aggregation(
        start_date=args.start,
        end_date=args.end,
        freq=args.freq,
        limit_stocks=args.limit_stocks,
        include_concept=args.include_concept,
    )

    if not daily.empty:
        print("\n=== 每日市场聚合 (最近 10 日) ===")
        display_cols = SCORE_NAMES + [f"{s}_z20" for s in SCORE_NAMES] + ["bull_bear_ratio"]
        display_cols = [c for c in display_cols if c in daily.columns]
        print(daily[display_cols].tail(10).to_string())

    if not grouped.empty:
        print("\n=== 分组聚合统计 ===")
        for gt in grouped["group_type"].unique():
            sub = grouped[grouped["group_type"] == gt]
            print(f"\n--- {gt}: {sub['group_name'].nunique()} 个组 ---")
            latest_date = sub["trade_date"].max()
            latest = sub[sub["trade_date"] == latest_date]
            if gt == "industry_l2":
                latest = latest[latest["stock_count"] >= MIN_INDUSTRY_SAMPLE]
                latest = latest.copy()
                latest["_composite"] = latest["structure_score"] * np.sqrt(latest["stock_count"])
                top3 = latest.nlargest(3, "_composite")[["group_name", "stock_count", "structure_score"]]
            else:
                top3 = latest.nlargest(3, "structure_score")[["group_name", "stock_count", "structure_score"]]
            print(f"结构最强 Top3 ({latest_date}):")
            print(top3.to_string(index=False))


if __name__ == "__main__":
    main()
