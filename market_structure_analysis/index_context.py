"""
index_context.py
指数锚点层 — 上证指数/沪深300 的因子+事件+状态标签

Purpose:
    对上证指数(000001.SH)和沪深300(000300.SH)计算轻量因子与事件，
    生成指数背景标签，作为市场状态的"参照系"。

Inputs:
    - tushare_data.fetcher.fetch_market_data() 获取指数日线（OHLCV + turnover）
    - 复用 factor_engine.compute_all_factors() + event_detector.detect_events()

Outputs:
    - 指数因子 DataFrame: {ts_code: factors_df}
    - 指数事件宽表: index=trade_date, cols=idx_*
    - 指数状态标签: {sse,hs300}_trend/state/volume_state

How to Run:
    python market_structure_analysis/index_context.py --start 2024-01-01 --end 2026-04-30

Examples:
    python market_structure_analysis/index_context.py
    python market_structure_analysis/index_context.py --start 2025-01-01

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

from market_structure_analysis.factor_engine import compute_all_factors
from market_structure_analysis.event_detector import detect_events, CORE_EVENTS

logger = logging.getLogger(__name__)

DEFAULT_INDEX_CODES = ["000001.SH", "000300.SH"]
INDEX_NAME_MAP = {"000001.SH": "上证指数", "000300.SH": "沪深300"}

INDEX_EVENT_COLS = [
    "idx_dsa_dir_flip_up",
    "idx_dsa_dir_flip_down",
    "idx_cross_above_dsa_vwap",
    "idx_cross_below_dsa_vwap",
    "idx_bbmacd_cross_upper",
    "idx_bbmacd_cross_lower",
    "idx_vol_spike_up",
    "idx_vol_spike_down",
]

VOL_SPIKE_THRESH = 2.0


def _fetch_index_kline(ts_code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """获取单只指数的 K 线数据。优先从 DB 加载，失败则走 tushare 在线拉取。"""
    from datasource.k_data_loader import load_k_data

    df = load_k_data(ts_code, freq="d", start_date=start_date, end_date=end_date)
    if df is not None and not df.empty and len(df) >= 60:
        logger.info("%s (%s): 从 DB 加载 %d 行", ts_code, INDEX_NAME_MAP.get(ts_code, ts_code), len(df))
        return df

    logger.info("%s: DB 无数据，尝试在线拉取...", ts_code)
    try:
        from tushare_data.fetcher import fetch_market_data
        start = (start_date or "20200101").replace("-", "")
        end = (end_date or "20301231").replace("-", "")
        df = fetch_market_data(ts_code, start_date=start, end_date=end)
        if df is not None and not df.empty:
            if "vol" in df.columns and "volume" not in df.columns:
                df = df.rename(columns={"vol": "volume"})
            logger.info("%s: 在线获取 %d 行", ts_code, len(df))
            return df
    except Exception as exc:
        logger.warning("%s 在线拉取失败: %s", ts_code, exc)

    return pd.DataFrame()


def compute_index_factors(
    index_codes: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    对每只指数计算全套因子+事件。

    Parameters
    ----------
    index_codes : list, optional
        指数代码列表，默认 ["000001.SH", "000300.SH"]
    start_date : str, optional
        起始日期
    end_date : str, optional
        结束日期

    Returns
    -------
    dict
        {ts_code: factors_with_events_df}
    """
    if index_codes is None:
        index_codes = DEFAULT_INDEX_CODES

    result = {}
    for code in index_codes:
        name = INDEX_NAME_MAP.get(code, code)
        logger.info("计算指数因子: %s (%s)", code, name)

        df = _fetch_index_kline(code, start_date=start_date, end_date=end_date)
        if df.empty or len(df) < 60:
            logger.warning("%s (%s): 数据不足(%d bars)，跳过", code, name, len(df))
            continue

        try:
            factors = compute_all_factors(df)
            events = detect_events(factors)
            events["ts_code"] = code
            result[code] = events
            evt_cols = [c for c in events.columns if c.startswith("evt_")]
            logger.info("%s (%s): 因子计算完成, %d 事件列, %d 行",
                        code, name, len(evt_cols), len(events))
        except Exception as exc:
            logger.error("%s (%s): 因子计算失败: %s", code, name, exc)

    return result


def extract_index_events(index_factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    从指数因子结果中提取 8 个轻量事件，拼成宽表。

    Parameters
    ----------
    index_factors : dict
        compute_index_factors() 输出

    Returns
    -------
    pd.DataFrame
        index=trade_date, cols=idx_* (每只指数 8 列)
    """
    rows = []
    for code, df in index_factors.items():
        prefix = "sse_" if code == "000001.SH" else "hs300_"
        dates = df.index if hasattr(df.index, 'to_datetime') else pd.to_datetime(df.index)

        event_map = {
            "evt_dsa_dir_flip_up": prefix + "dsa_dir_flip_up",
            "evt_dsa_dir_flip_down": prefix + "dsa_dir_flip_down",
            "evt_cross_above_dsa_vwap": prefix + "cross_above_dsa_vwap",
            "evt_cross_below_dsa_vwap": prefix + "cross_below_dsa_vwap",
            "evt_bbmacd_cross_upper": prefix + "bbmacd_cross_upper",
            "evt_bbmacd_cross_lower": prefix + "bbmacd_cross_lower",
        }

        vol_z_col = "vol_zscore"
        if vol_z_col not in df.columns:
            vol_z_col = None

        for i, dt in enumerate(dates):
            row = {"trade_date": dt}
            for src, dst in event_map.items():
                if src in df.columns:
                    row[dst] = float(df.iloc[i].get(src, 0.0))
                else:
                    row[dst] = 0.0

            if vol_z_col is not None and vol_z_col in df.columns:
                vz = df.iloc[i].get(vol_z_col, 0.0)
                row[prefix + "vol_spike_up"] = 1.0 if float(vz) > VOL_SPIKE_THRESH else 0.0
                row[prefix + "vol_spike_down"] = 1.0 if float(vz) < -VOL_SPIKE_THRESH else 0.0
            else:
                row[prefix + "vol_spike_up"] = 0.0
                row[prefix + "vol_spike_down"] = 0.0

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    return out.set_index("trade_date")


def classify_index_state(index_events_df: pd.DataFrame) -> pd.DataFrame:
    """
    为每只指数生成趋势/状态/量能标签。

    规则：
      *_trend:   基于 DSA 方向翻转 + DSA 相对 VWAP 位置
                 强多(dsa_dir_flip_up 且 dsa_above_vwap) / 多(dsa_above_vwap) /
                 中性 / 空(dsa_below_vwap) / 强空(dsa_dir_flip_down 且 dsa_below_vwap)
      *_state:   复用 v2 五档逻辑但放宽阈值（指数波动小于个股）
      *_volume_state: 放量(vol_zscore > 0.5) / 平量 / 缩量(vol_zscore < -0.5)

    Parameters
    ----------
    index_events_df : pd.DataFrame
        extract_index_events() 输出

    Returns
    -------
    pd.DataFrame
        index=trade_date, cols={sse,hs300}_{trend,state,volume_state}
    """
    if index_events_df.empty:
        return pd.DataFrame()

    result = index_events_df.copy()

    for prefix in ["sse_", "hs300_"]:
        flip_up = prefix + "dsa_dir_flip_up"
        flip_down = prefix + "dsa_dir_flip_down"
        above = prefix + "cross_above_dsa_vwap"
        below = prefix + "cross_below_dsa_vwap"

        def _trend_rule(row):
            if row.get(flip_up, 0) == 1.0 and row.get(above, 0) == 1.0:
                return "强多"
            elif row.get(flip_up, 0) == 1.0 or row.get(above, 0) == 1.0:
                return "多"
            elif row.get(flip_down, 0) == 1.0 and row.get(below, 0) == 1.0:
                return "强空"
            elif row.get(flip_down, 0) == 1.0 or row.get(below, 0) == 1.0:
                return "空"
            return "中性"

        result[prefix + "trend"] = result.apply(_trend_rule, axis=1)

        bb_up = prefix + "bbmacd_cross_upper"
        bb_down = prefix + "bbmacd_cross_lower"

        def _state_rule(row):
            bull_signals = sum([
                row.get(flip_up, 0),
                row.get(above, 0),
                row.get(bb_up, 0),
            ])
            bear_signals = sum([
                row.get(flip_down, 0),
                row.get(below, 0),
                row.get(bb_down, 0),
            ])
            net = bull_signals - bear_signals
            if net >= 2:
                return "扩张"
            elif net >= 1:
                return "修复"
            elif net <= -2:
                return "退潮"
            elif net <= -1:
                return "防守"
            return "中性"

        result[prefix + "state"] = result.apply(_state_rule, axis=1)

        vol_up = prefix + "vol_spike_up"
        vol_down = prefix + "vol_spike_down"

        def _vol_rule(row):
            vu = row.get(vol_up, 0)
            vd = row.get(vol_down, 0)
            if vu == 1.0:
                return "放量"
            elif vd == 1.0:
                return "缩量"
            return "平量"

        result[prefix + "volume_state"] = result.apply(_vol_rule, axis=1)

    keep_cols = ["trade_date"] if "trade_date" in result.columns else []
    for pfx in ["sse_", "hs300_"]:
        for suffix in ["trend", "state", "volume_state"]:
            col = pfx + suffix
            if col in result.columns:
                keep_cols.append(col)

    return result[keep_cols].set_index("trade_date") if "trade_date" in result.columns else result


def generate_index_context_report(
    index_states: pd.DataFrame,
    index_events: Optional[pd.DataFrame] = None,
    date: Optional[str] = None,
) -> str:
    """
    生成指数背景解读文本。

    Parameters
    ----------
    index_states : pd.DataFrame
        classify_index_state() 输出
    index_events : pd.DataFrame, optional
        原始事件数据（用于补充细节）
    date : str, optional
        目标日期 YYYY-MM-DD，None 取最后一行

    Returns
    -------
    str
        解读文本
    """
    if index_states.empty:
        return "[无指数数据]"

    if date:
        target = pd.Timestamp(date)
        if target in index_states.index:
            row = index_states.loc[[target]].iloc[0]
        else:
            nearest = index_states.index[index_states.index <= target]
            if len(nearest) == 0:
                row = index_states.iloc[-1]
            else:
                row = index_states.loc[nearest[-1]]
    else:
        row = index_states.iloc[-1]

    lines = []
    lines.append(f"【二、指数背景】")

    sse_trend = row.get("sse_trend", "未知")
    sse_state = row.get("sse_state", "未知")
    sse_vol = row.get("sse_volume_state", "未知")
    hs300_trend = row.get("hs300_trend", "未知")
    hs300_state = row.get("hs300_state", "未知")
    hs300_vol = row.get("hs300_volume_state", "未知")

    lines.append(f"  上证指数: {sse_trend}{sse_vol} ({sse_state})")
    lines.append(f"  沪深300:  {hs300_trend}{hs300_vol} ({hs300_state})")

    interpretation = _interpret_index_combo(sse_trend, sse_state, hs300_trend, hs300_state)
    lines.append(f"  解读: {interpretation}")

    return "\n".join(lines)


def _interpret_index_combo(
    sse_trend: str, sse_state: str,
    hs300_trend: str, hs300_state: str,
) -> str:
    """根据两只指数的组合给出盘面解读"""
    strong_bull = {"强多", "多"}
    strong_bear = {"强空", "空"}
    expansion = {"扩张", "修复"}
    retreat = {"退潮", "防守"}

    sse_bull = sse_trend in strong_bull
    hs300_bull = hs300_trend in strong_bull
    both_expansion = sse_state in expansion and hs300_state in expansion
    any_retreat = sse_state in retreat or hs300_state in retreat

    if sse_bull and hs300_bull and both_expansion:
        return "权重指数共振确认扩张 → 权重共振型行情"
    if sse_bull and not hs300_bull:
        return "上证偏强但沪深300未跟进 → 可能是小票/题材主导行情"
    if not sse_bull and hs300_bull:
        return "沪深300偏强但上证未确认 → 核心资产有支撑但市场情绪一般"
    if any_retreat:
        return "指数层面出现退潮/防守信号 → 需警惕系统性风险"
    if sse_state == "中性" and hs300_state == "中性":
        return "指数方向不明确 → 等待方向选择"
    return "指数组合需综合判断"


def main():
    parser = argparse.ArgumentParser(description="指数锚点层 — 计算指数因子与状态标签")
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Step 1: 计算指数因子...")
    factors = compute_index_factors(start_date=args.start, end_date=args.end)
    logger.info("完成: %d 只指数", len(factors))

    if not factors:
        logger.error("无指数数据")
        return

    logger.info("Step 2: 提取指数事件...")
    events = extract_index_events(factors)
    print(f"\n指数事件表: {events.shape}")
    if not events.empty:
        print(events.tail(10).to_string())

    logger.info("Step 3: 分类指数状态...")
    states = classify_index_state(events)
    print(f"\n指数状态表: {states.shape}")
    if not states.empty:
        print(states.tail(10).to_string())

    report = generate_index_context_report(states, events)
    print(f"\n{report}")

    if not states.empty:
        dist_cols = [c for c in states.columns if c.endswith("_trend") or c.endswith("_state")]
        for col in dist_cols:
            if col in states.columns:
                print(f"\n  {col} 分布:")
                vc = states[col].value_counts()
                total = vc.sum()
                for val, cnt in vc.items():
                    pct = cnt / total * 100
                    bar = "█" * int(pct / 5)
                    print(f"    {val:>6}: {cnt:>4} ({pct:.1f}%) {bar}")


if __name__ == "__main__":
    main()
