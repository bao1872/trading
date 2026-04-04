# -*- coding: utf-8 -*-
"""
Breakout Pullback 回测脚本（增强版 - 含质量分数相关性分析）

Purpose:
    回测突破后回踩买入策略的收益率，并分析质量分数与收益的相关性
    
    策略逻辑：
    1. 从 breakout_dir_turn_events 读取翻多信号（含买入区间和质量分数）
    2. 按股票遍历，使用 MergedEngine 计算 rope_dir
    3. 当价格进入 [dir_turn_band_low, dir_turn_band_high] 时买入
    4. 当 rope_dir 变为 0 或 -1 时卖出
    
Inputs:
    - breakout_dir_turn_events: 翻多事件表（含买入区间 + 质量分数）
    - stock_k_data: 日线行情数据

Outputs:
    - backtest_trades.csv: 详细交易记录（含原始质量分数）
    - backtest_quality_analysis.csv: 质量分数相关性分析报告

How to Run:
    # 全量回测（输出到CSV）
    python backtrader/backtest_breakout_pullback.py --output-dir ./backtest_output

    # 测试模式（少量信号）
    python backtrader/backtest_breakout_pullback.py --limit-stocks 5 --output-dir ./backtest_output

Side Effects:
    - 只读取数据库，不写入数据
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasource.database import get_session, query_df
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backtest_pullback")


# 质量分数列名定义
QUALITY_COLS = {
    "total": "breakout_quality_score",
    "grade": "breakout_quality_grade",
    # 一级维度
    "trend": "score_trend_total",
    "candle": "score_candle_total",
    "volume": "score_volume_total",
    "freshness": "score_freshness_total",
    # 二级子项
    "bg_rope_slope": "score_bg_rope_slope",
    "bg_dist_to_rope": "score_bg_dist_to_rope",
    "bg_consolidation": "score_bg_consolidation",
    "close_pos": "score_candle_close_pos",
    "body_to_range": "score_candle_body_to_range",
    "upper_wick": "score_candle_upper_wick",
    "vol_z": "score_volume_vol_z",
    "vol_record": "score_volume_vol_record",
    "fresh_count": "score_freshness_count",
    "fresh_cum_gain": "score_freshness_cum_gain",
}


@dataclass
class Signal:
    """翻多信号（含质量分数）"""
    ts_code: str
    name: str
    event_time: str
    dir_turn_upper_price: float
    dir_turn_atr_raw: float
    dir_turn_tol_price: float
    dir_turn_band_low: float
    dir_turn_band_high: float
    quality_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class TradeRecord:
    """交易记录"""
    ts_code: str = ""
    name: str = ""
    signal_event_time: str = ""
    entry_date: Optional[str] = None
    entry_price: Optional[float] = None
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None
    holding_days: Optional[int] = None
    # 质量分数（从信号继承）
    quality_score: Optional[float] = None
    quality_grade: Optional[str] = None
    score_trend: Optional[float] = None
    score_candle: Optional[float] = None
    score_volume: Optional[float] = None
    score_freshness: Optional[float] = None


@dataclass
class WatchItem:
    """观察列表项"""
    signal: Signal
    status: str = "waiting"  # waiting / entered / exited / missed (未回踩直接上涨)
    trade: TradeRecord = field(default_factory=TradeRecord)


@dataclass
class MissedRecord:
    """未回踩直接上涨记录"""
    ts_code: str = ""
    name: str = ""
    signal_event_time: str = ""
    upper_price: float = 0.0
    band_low: float = 0.0
    band_high: float = 0.0
    # 信号后的价格走势
    first_high_after_signal: float = 0.0  # 信号后第一个高点
    first_high_date: str = ""
    max_price_after_signal: float = 0.0   # 信号后最高价
    max_price_date: str = ""
    price_change_pct: float = 0.0         # 从信号到最高价的涨幅
    days_to_max: int = 0                  # 信号到最高价的交易日数
    # 质量分数
    quality_score: Optional[float] = None
    quality_grade: Optional[str] = None
    score_trend: Optional[float] = None
    score_candle: Optional[float] = None
    score_volume: Optional[float] = None
    score_freshness: Optional[float] = None


def load_signals(start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Signal]:
    """从 breakout_dir_turn_events 加载日线翻多信号"""
    with get_session() as session:
        filters = {"freq": "d"}
        if start_date:
            filters["event_time >= "] = start_date
        if end_date:
            filters["event_time <= "] = end_date
        
        df = query_df(
            session,
            table_name="breakout_dir_turn_events",
            filters=filters,
            order_by="+event_time"
        )
    
    if df.empty:
        return []
    
    signals = []
    for _, row in df.iterrows():
        if pd.isna(row.get("dir_turn_band_low")) or pd.isna(row.get("dir_turn_band_high")):
            continue
        
        # 提取质量分数
        q_scores = {}
        for key, col in QUALITY_COLS.items():
            val = row.get(col)
            if pd.notna(val):
                if key == "grade":
                    q_scores[key] = str(val)
                else:
                    try:
                        q_scores[key] = float(val)
                    except (ValueError, TypeError):
                        pass
        
        signals.append(Signal(
            ts_code=row["ts_code"],
            name=row["name"] or "",
            event_time=row["event_time"],
            dir_turn_upper_price=row["dir_turn_upper_price"],
            dir_turn_atr_raw=row["dir_turn_atr_raw"],
            dir_turn_tol_price=row["dir_turn_tol_price"],
            dir_turn_band_low=row["dir_turn_band_low"],
            dir_turn_band_high=row["dir_turn_band_high"],
            quality_scores=q_scores,
        ))
    
    return signals


def load_kline_data(ts_code: str, start_date: str, bars: int = 500) -> Optional[pd.DataFrame]:
    """加载单只股票的K线数据"""
    code = ts_code.split(".")[0] if "." in ts_code else ts_code
    
    with get_session() as session:
        filters = {"freq": "d", "ts_code": ts_code}
        
        df = query_df(
            session,
            table_name="stock_k_data",
            columns=["bar_time", "open", "high", "low", "close", "volume"],
            filters=filters,
            order_by="+bar_time"
        )
    
    if df.empty or len(df) < 50:
        return None
    
    # MergedEngine 期望的列名: open, high, low, close, vol, amount
    if "volume" in df.columns and "vol" not in df.columns:
        df = df.rename(columns={"volume": "vol"})
    
    # 计算成交额 (amount = close * vol)
    if "amount" not in df.columns and "vol" in df.columns and "close" in df.columns:
        df["amount"] = df["close"] * df["vol"]
    
    df["bar_time"] = pd.to_datetime(df["bar_time"]).dt.tz_localize(None)
    df = df.set_index("bar_time")
    
    # 只保留 start_date 之前60天开始的数据（确保因子计算有足够历史）
    start_ts = pd.Timestamp(start_date) - timedelta(days=80)
    df = df[df.index >= start_ts]
    
    if len(df) < min(bars, 100):
        return None
    
    return df


def normalize_freq(freq: str) -> str:
    freq_map = {"d": "daily", "w": "weekly", "60m": "min60"}
    return freq_map.get(freq, freq)


def build_engine_args(symbol: str, freq: str, bars: int) -> argparse.Namespace:
    """构建 MergedEngine 参数（复用 scan_breakout_events.py 的逻辑）"""
    args = argparse.Namespace()
    args.symbol = symbol
    args.freq = normalize_freq(freq)
    args.bars = bars
    args.len = 14
    args.multi = 1.5
    args.source = "close"
    args.show_ranges = False
    args.show_atr_channel = False
    args.show_break_markers = False
    args.show_factor_panels = False
    args.swing_left = 10
    args.swing_right = 10
    args.breakout_by = "Close"
    args.max_levels = 5
    args.br_vol_filter = False
    args.br_vol_filter_pct = 60.0
    args.liq_lookback = 20
    return args


def process_stock(signals_for_stock: List[Signal], kdf: pd.DataFrame, bars: int = 500) -> Tuple[List[TradeRecord], List[MissedRecord]]:
    """
    处理单只股票的所有信号
    
    使用 MergedEngine 计算全量因子，逐 bar 判断买卖
    返回: (交易记录列表, 未回踩直接上涨记录列表)
    """
    if not signals_for_stock or kdf is None or len(kdf) < 100:
        return [], []
    
    ts_code = signals_for_stock[0].ts_code
    symbol = ts_code.split(".")[0]
    
    try:
        from features.merged_atr_rope_breakout_volume_delta import MergedEngine
        
        args = build_engine_args(symbol, "d", bars)
        engine = MergedEngine(kdf, args)
        engine.run()
        result_df = engine.df.copy()
    except Exception as e:
        logger.warning("MergedEngine 计算失败 %s: %s", ts_code, e)
        return [], []
    
    if result_df.empty or "rope_dir" not in result_df.columns:
        logger.warning("%s rope_dir 未生成", ts_code)
        return [], []
    
    trades: List[TradeRecord] = []
    missed_records: List[MissedRecord] = []
    
    # 初始化观察列表
    watchlist: List[WatchItem] = []
    for sig in signals_for_stock:
        item = WatchItem(signal=sig)
        item.trade.ts_code = sig.ts_code
        item.trade.name = sig.name
        item.trade.signal_event_time = sig.event_time
        item.trade.quality_score = sig.quality_scores.get("total")
        item.trade.quality_grade = sig.quality_scores.get("grade")
        item.trade.score_trend = sig.quality_scores.get("trend")
        item.trade.score_candle = sig.quality_scores.get("candle")
        item.trade.score_volume = sig.quality_scores.get("volume")
        item.trade.score_freshness = sig.quality_scores.get("freshness")
        
        # 解析信号日期（用于后续只检测信号之后的K线）
        try:
            signal_date = pd.Timestamp(sig.event_time[:10])
            item.signal._signal_date = signal_date
        except Exception:
            item.signal._signal_date = None
        
        watchlist.append(item)
    
    # 逐 bar 遍历
    for idx, row in result_df.iterrows():
        bar_date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx).split(" ")[0]
        current_dir = row.get("rope_dir")
        
        if pd.isna(current_dir):
            continue
        current_dir_val = float(current_dir)
        
        for item in watchlist:
            if item.status == "exited":
                continue
            
            # 关键修复：只在信号日期之后才检测（信号是收盘后计算的，当天不能买入）
            sig_date = getattr(item.signal, '_signal_date', None)
            if sig_date is not None:
                try:
                    bar_date_ts = pd.Timestamp(bar_date_str)
                    if bar_date_ts <= sig_date:  # 信号当天及之前都跳过
                        continue
                except Exception:
                    pass
            
            band_low = item.signal.dir_turn_band_low
            band_high = item.signal.dir_turn_band_high
            upper_price = item.signal.dir_turn_upper_price
            
            if item.status == "waiting":
                # 买入条件：价格进入 [band_low, band_high]
                if row["low"] <= band_high and row["high"] >= band_low:
                    item.status = "entered"
                    item.trade.entry_date = bar_date_str
                    item.trade.entry_price = row["close"]
                # 未回踩直接上涨检测：价格突破 upper_price 且从未进入买入区间
                elif row["high"] > upper_price * 1.01:  # 突破upper_price 1%以上视为missed
                    item.status = "missed"
                    item.status = "exited"
                    # 记录未回踩信息
                    missed = MissedRecord()
                    missed.ts_code = item.signal.ts_code
                    missed.name = item.signal.name
                    missed.signal_event_time = item.signal.event_time
                    missed.upper_price = upper_price
                    missed.band_low = band_low
                    missed.band_high = band_high
                    missed.first_high_after_signal = row["high"]
                    missed.first_high_date = bar_date_str
                    missed.max_price_after_signal = row["high"]
                    missed.max_price_date = bar_date_str
                    if item.signal.dir_turn_upper_price > 0:
                        missed.price_change_pct = round((row["high"] - item.signal.dir_turn_upper_price) / item.signal.dir_turn_upper_price * 100, 2)
                    missed.quality_score = item.trade.quality_score
                    missed.quality_grade = item.trade.quality_grade
                    missed.score_trend = item.trade.score_trend
                    missed.score_candle = item.trade.score_candle
                    missed.score_volume = item.trade.score_volume
                    missed.score_freshness = item.trade.score_freshness
                    missed_records.append(missed)
            
            elif item.status == "entered":
                # 卖出条件：rope_dir 变为 0 或 -1
                if current_dir_val in [0, -1]:
                    item.status = "exited"
                    item.trade.exit_date = bar_date_str
                    item.trade.exit_price = row["close"]
                    
                    if item.trade.entry_price and item.trade.entry_price > 0:
                        pnl = (item.trade.exit_price - item.trade.entry_price) / item.trade.entry_price * 100
                        item.trade.pnl_pct = round(pnl, 2)
                        
                        entry_dt = pd.Timestamp(item.trade.entry_date)
                        exit_dt = pd.Timestamp(bar_date_str)
                        item.trade.holding_days = (exit_dt - entry_dt).days
                        
                        reason = f"dir_turned_{int(current_dir_val)}"
                        if current_dir_val == 0:
                            item.trade.exit_reason = "dir_flat"
                        else:
                            item.trade.exit_reason = "dir_down"
                    
                    trades.append(item.trade)
    
    # 处理未退出的（超时或数据结束）
    last_idx = result_df.index[-1]
    last_date_str = last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else str(last_idx).split(" ")[0]
    
    for item in watchlist:
        if item.status == "entered":
            # 强制退出（超时或数据结束）
            item.status = "exited"
            item.trade.exit_date = last_date_str
            item.trade.exit_price = result_df.iloc[-1]["close"]
            
            if item.trade.entry_price and item.trade.entry_price > 0:
                pnl = (item.trade.exit_price - item.trade.entry_price) / item.trade.entry_price * 100
                item.trade.pnl_pct = round(pnl, 2)
                
                entry_dt = pd.Timestamp(item.trade.entry_date)
                exit_dt = pd.Timestamp(last_date_str)
                item.trade.holding_days = (exit_dt - entry_dt).days
            
            item.trade.exit_reason = "timeout_end_of_data"
            trades.append(item.trade)
        elif item.status == "waiting":
            # 如果信号后没有触发任何条件，检查是否超过一定天数后标记为missed
            # 这部分逻辑已在上面处理
            pass
    
    return trades, missed_records


def run_backtest(signals: List[Signal], limit_stocks: Optional[int] = None) -> Tuple[List[TradeRecord], List[MissedRecord]]:
    """
    运行回测：按股票遍历，使用 MergedEngine 计算 rope_dir
    返回: (交易记录列表, 未回踩直接上涨记录列表)
    """
    # 按股票分组
    stocks_map: Dict[str, List[Signal]] = {}
    for sig in signals:
        key = sig.ts_code
        if key not in stocks_map:
            stocks_map[key] = []
        stocks_map[key].append(sig)
    
    logger.info(f"涉及 {len(stocks_map)} 只股票，{len(signals)} 个信号")
    
    if limit_stocks:
        stocks_map = dict(list(stocks_map.items())[:limit_stocks])
        total_sigs = sum(len(v) for v in stocks_map.values())
        logger.info(f"限制为前 {limit_stocks} 只股票，{total_sigs} 个信号")
    
    all_trades: List[TradeRecord] = []
    all_missed: List[MissedRecord] = []
    
    # 找出最早和最晚的信号日期
    signal_dates = [s.event_time[:10] for s in signals]
    earliest_signal = min(signal_dates) if signal_dates else "2025-01-01"
    
    processed = 0
    with tqdm(stocks_map.items(), desc="回测进度") as pbar:
        for ts_code, stock_signals in pbar:
            # 加载K线数据
            kdf = load_kline_data(ts_code, start_date=earliest_signal, bars=500)
            
            if kdf is None:
                continue
            
            # 处理这只股票的所有信号
            stock_trades, stock_missed = process_stock(stock_signals, kdf)
            all_trades.extend(stock_trades)
            all_missed.extend(stock_missed)
            
            processed += 1
            if processed % 50 == 0:
                logger.info(f"已处理 {processed}/{len(stocks_map)} 只股票, {len(all_trades)} 笔交易, {len(all_missed)} 个未回踩")
    
    return all_trades, all_missed


def generate_base_report(trades: List[TradeRecord]) -> Dict:
    """基础统计报告"""
    completed = [t for t in trades if t.pnl_pct is not None]
    winning = [t for t in completed if t.pnl_pct > 0]
    losing = [t for t in completed if t.pnl_pct <= 0]
    
    win_sum = sum(t.pnl_pct for t in winning) if winning else 0
    loss_sum = sum(abs(t.pnl_pct) for t in losing) if losing else 1
    
    return {
        "total_trades": len(trades),
        "completed": len(completed),
        "winning": len(winning),
        "losing": len(losing),
        "win_rate": round(len(winning) / len(completed) * 100, 2) if completed else 0,
        "avg_pnl": round(sum(t.pnl_pct for t in completed) / len(completed), 2) if completed else 0,
        "avg_win": round(win_sum / len(winning), 2) if winning else 0,
        "avg_loss": round(-loss_sum / len(losing), 2) if losing else 0,
        "max_win": round(max((t.pnl_pct or 0) for t in completed), 2) if completed else 0,
        "max_loss": round(min((t.pnl_pct or 0) for t in completed), 2) if completed else 0,
        "median_pnl": round(np.median([t.pnl_pct for t in completed]), 2) if completed else 0,
        "total_pnl": round(sum(t.pnl_pct or 0 for t in completed), 2),
        "profit_factor": round(win_sum / loss_sum, 2),
        "avg_holding_days": round(np.mean([t.holding_days for t in completed if t.holding_days]), 1) if completed else 0,
    }


def generate_quality_report(trades: List[TradeRecord]) -> Dict:
    """质量分数相关性分析报告"""
    completed = [t for t in trades if t.pnl_pct is not None]
    if len(completed) < 10:
        return {"error": "样本不足，无法进行相关性分析"}
    
    pnls = np.array([t.pnl_pct for t in completed])
    
    report = {}
    
    # === 1. 按 grade 分组统计 ===
    grade_stats = {}
    for grade in ["A", "B", "C", "D"]:
        subset = [t for t in completed if t.quality_grade == grade]
        if subset:
            sub_pnls = [t.pnl_pct for t in subset]
            wins = sum(1 for p in sub_pnls if p > 0)
            grade_stats[grade] = {
                "count": len(subset),
                "win_rate": round(wins / len(subset) * 100, 2),
                "avg_pnl": round(np.mean(sub_pnls), 2),
                "median_pnl": round(np.median(sub_pnls), 2),
                "total_pnl": round(sum(sub_pnls), 2),
            }
        else:
            grade_stats[grade] = {"count": 0, "win_rate": 0, "avg_pnl": 0, "median_pnl": 0, "total_pnl": 0}
    
    report["by_grade"] = grade_stats
    
    # === 2. 按总分四分位分组 ===
    scores = [t.quality_score for t in completed if t.quality_score is not None]
    if len(scores) >= 20:
        q25, q50, q75 = np.percentile(scores, [25, 50, 75])
        
        quartile_stats = {}
        for label, low, high in [("Q1(低分)", None, q25), ("Q2(中低)", q25, q50), ("Q3(中高)", q50, q75), ("Q4(高分)", q75, None)]:
            if low is None and high is None:
                subset = completed
            elif low is None:
                subset = [t for t in completed if t.quality_score is not None and t.quality_score <= high]
            elif high is None:
                subset = [t for t in completed if t.quality_score is not None and t.quality_score > low]
            else:
                subset = [t for t in completed if t.quality_score is not None and low < t.quality_score <= high]
            
            if subset:
                sub_pnls = [t.pnl_pct for t in subset]
                wins = sum(1 for p in sub_pnls if p > 0)
                quartile_stats[label] = {
                    "count": len(subset),
                    "win_rate": round(wins / len(subset) * 100, 2),
                    "avg_pnl": round(np.mean(sub_pnls), 2),
                    "total_pnl": round(sum(sub_pnls), 2),
                }
        
        report["by_quartile"] = quartile_stats
    
    # === 3. 各维度与收益率的相关系数 ===
    corr_results = {}
    dim_cols = [
        ("quality_score", "总分"),
        ("score_trend", "趋势分"),
        ("score_candle", "K线分"),
        ("score_volume", "成交量分"),
        ("score_freshness", "新鲜度分"),
        ("score_bg_rope_slope", "ROPE斜率"),
        ("score_bg_dist_to_rope", "距离ROPE"),
        ("score_bg_consolidation", "整理度"),
        ("score_candle_close_pos", "收盘位置"),
        ("score_candle_body_to_range", "实体占比"),
        ("score_candle_upper_wick", "上影线"),
        ("score_volume_vol_z", "量Z-score"),
        ("score_freshness_count", "突破次数"),
        ("score_freshness_cum_gain", "累计涨幅"),
    ]
    
    for attr_key, label in dim_cols:
        vals = [getattr(t, attr_key, None) for t in completed]
        valid_mask = [(v is not None and not np.isnan(v)) for v in vals]
        if sum(valid_mask) >= 10:
            x = np.array([vals[i] for i in range(len(vals)) if valid_mask[i]])
            y = pnls[valid_mask]
            r, p = sp_stats.pearsonr(x, y)
            corr_results[attr_key] = {
                "label": label,
                "r": round(r, 4),
                "p_value": round(p, 6),
                "n": int(sum(valid_mask)),
                "interpretation": "显著正相关" if r > 0.05 and p < 0.05 
                                 else "显著负相关" if r < -0.05 and p < 0.05
                                 else "无显著相关",
            }
    
    report["correlation_with_pnl"] = corr_results
    
    # === 4. 高分 vs 低分 对比 ===
    high_score_trades = [t for t in completed if t.quality_score is not None and t.quality_score >= np.median(scores)]
    low_score_trades = [t for t in completed if t.quality_score is not None and t.quality_score < np.median(scores)]
    
    if high_score_trades and low_score_trades:
        hs_pnls = [t.pnl_pct for t in high_score_trades]
        ls_pnls = [t.pnl_pct for t in low_score_trades]
        hs_wr = sum(1 for p in hs_pnls if p > 0) / len(hs_pnls) * 100
        ls_wr = sum(1 for p in ls_pnls if p > 0) / len(ls_pnls) * 100
        
        report["high_vs_low_score"] = {
            "high_score_median": round(np.median(scores), 2),
            "high_n": len(high_score_trades),
            "high_win_rate": round(hs_wr, 2),
            "high_avg_pnl": round(np.mean(hs_pnls), 2),
            "low_n": len(low_score_trades),
            "low_win_rate": round(ls_wr, 2),
            "low_avg_pnl": round(np.mean(ls_pnls), 2),
            "diff_win_rate": round(hs_wr - ls_wr, 2),
            "diff_avg_pnl": round(np.mean(hs_pnls) - np.mean(ls_pnls), 2),
        }
    
    # === 5. 各维度组合表现 ===
    trend_vals = [t.score_trend for t in completed if t.score_trend is not None]
    vol_vals = [t.score_volume for t in completed if t.score_volume is not None]
    
    if len(trend_vals) >= 20 and len(vol_vals) >= 20:
        t_med = np.median(trend_vals)
        v_med = np.median(vol_vals)
        
        combo_stats = {}
        for t_label, t_cond in [
            ("高趋势+高量", lambda t: (t.score_trend or 0) >= t_med and (t.score_volume or 0) >= v_med),
            ("高趋势+低量", lambda t: (t.score_trend or 0) >= t_med and (t.score_volume or 0) < v_med),
            ("低趋势+高量", lambda t: (t.score_trend or 0) < t_med and (t.score_volume or 0) >= v_med),
            ("低趋势+低量", lambda t: (t.score_trend or 0) < t_med and (t.score_volume or 0) < v_med),
        ]:
            subset = [t for t in completed if t_cond(t)]
            if len(subset) >= 5:
                sub_pnls = [t.pnl_pct for t in subset]
                wins = sum(1 for p in sub_pnls if p > 0)
                combo_stats[t_label] = {
                    "n": len(subset),
                    "win_rate": round(wins / len(subset) * 100, 2),
                    "avg_pnl": round(np.mean(sub_pnls), 2),
                }
        
        report["dimension_combo"] = combo_stats
    
    return report


def generate_missed_report(missed_records: List[MissedRecord]) -> Dict:
    """未回踩直接上涨的特征分析报告"""
    if not missed_records:
        return {"error": "没有未回踩直接上涨的数据"}
    
    report = {}
    
    # === 1. 基础统计 ===
    total_missed = len(missed_records)
    avg_price_change = np.mean([m.price_change_pct for m in missed_records if m.price_change_pct])
    
    report["summary"] = {
        "total_missed": total_missed,
        "avg_price_change_pct": round(avg_price_change, 2) if avg_price_change else 0,
        "median_price_change_pct": round(np.median([m.price_change_pct for m in missed_records if m.price_change_pct]), 2),
    }
    
    # === 2. 按Grade分组统计 ===
    grade_stats = {}
    for grade in ["A", "B", "C", "D"]:
        subset = [m for m in missed_records if m.quality_grade == grade]
        if subset:
            changes = [m.price_change_pct for m in subset if m.price_change_pct]
            grade_stats[grade] = {
                "count": len(subset),
                "avg_price_change": round(np.mean(changes), 2) if changes else 0,
                "median_price_change": round(np.median(changes), 2) if changes else 0,
            }
        else:
            grade_stats[grade] = {"count": 0, "avg_price_change": 0, "median_price_change": 0}
    report["by_grade"] = grade_stats
    
    # === 3. 质量分数对比（missed vs 正常交易）===
    # 这里只统计missed的质量分数分布
    scores = [m.quality_score for m in missed_records if m.quality_score is not None]
    if scores:
        report["quality_score_distribution"] = {
            "mean": round(np.mean(scores), 2),
            "median": round(np.median(scores), 2),
            "std": round(np.std(scores), 2),
            "min": round(np.min(scores), 2),
            "max": round(np.max(scores), 2),
        }
    
    # === 4. 各维度分数对比 ===
    dim_analysis = {}
    dims = [
        ("score_trend", "趋势分"),
        ("score_candle", "K线分"),
        ("score_volume", "成交量分"),
        ("score_freshness", "新鲜度分"),
    ]
    for attr, label in dims:
        vals = [getattr(m, attr) for m in missed_records if getattr(m, attr) is not None]
        if vals:
            dim_analysis[label] = {
                "mean": round(np.mean(vals), 2),
                "median": round(np.median(vals), 2),
            }
    report["dimension_analysis"] = dim_analysis
    
    # === 5. 高分特征分析 ===
    # 分析哪些特征更容易出现未回踩直接上涨
    high_score_missed = [m for m in missed_records if m.quality_score is not None and m.quality_score >= np.percentile(scores, 75)] if scores else []
    low_score_missed = [m for m in missed_records if m.quality_score is not None and m.quality_score < np.percentile(scores, 25)] if scores else []
    
    if high_score_missed and low_score_missed:
        report["high_vs_low_score_missed"] = {
            "high_score_count": len(high_score_missed),
            "low_score_count": len(low_score_missed),
            "high_avg_price_change": round(np.mean([m.price_change_pct for m in high_score_missed if m.price_change_pct]), 2),
            "low_avg_price_change": round(np.mean([m.price_change_pct for m in low_score_missed if m.price_change_pct]), 2),
        }
    
    return report


def save_outputs(trades: List[TradeRecord], missed_records: List[MissedRecord], 
                base_report: Dict, quality_report: Dict, missed_report: Dict, output_dir: str):
    """保存所有输出到 CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 详细交易记录
    rows = []
    for t in trades:
        rows.append({
            "ts_code": t.ts_code,
            "name": t.name,
            "signal_time": t.signal_event_time,
            "entry_date": t.entry_date,
            "entry_price": t.entry_price,
            "exit_date": t.exit_date,
            "exit_price": t.exit_price,
            "pnl_pct": t.pnl_pct,
            "holding_days": t.holding_days,
            "exit_reason": t.exit_reason,
            "quality_score": t.quality_score,
            "quality_grade": t.quality_grade,
            "score_trend": t.score_trend,
            "score_candle": t.score_candle,
            "score_volume": t.score_volume,
            "score_freshness": t.score_freshness,
        })
    
    trades_df = pd.DataFrame(rows)
    trades_path = os.path.join(output_dir, "backtest_trades.csv")
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    logger.info(f"交易记录已保存到: {trades_path}")
    
    # 2. 基础报告
    base_path = os.path.join(output_dir, "backtest_summary.csv")
    pd.DataFrame([base_report]).to_csv(base_path, index=False, encoding="utf-8-sig")
    logger.info(f"基础报告已保存到: {base_path}")
    
    # 3. Grade 分组统计
    if "by_grade" in quality_report:
        grade_df = pd.DataFrame.from_dict(quality_report["by_grade"], orient="index").reset_index()
        grade_df.columns = ["grade", "count", "win_rate", "avg_pnl", "median_pnl", "total_pnl"]
        grade_path = os.path.join(output_dir, "backtest_by_grade.csv")
        grade_df.to_csv(grade_path, index=False, encoding="utf-8-sig")
        logger.info(f"Grade统计已保存到: {grade_path}")
    
    # 4. 四分位统计
    if "by_quartile" in quality_report:
        qtile_df = pd.DataFrame.from_dict(quality_report["by_quartile"], orient="index").reset_index()
        qtile_df.columns = ["quartile", "count", "win_rate", "avg_pnl", "total_pnl"]
        qtile_path = os.path.join(output_dir, "backtest_by_quartile.csv")
        qtile_df.to_csv(qtile_path, index=False, encoding="utf-8-sig")
        logger.info(f"四分位统计已保存到: {qtile_path}")
    
    # 5. 相关性分析
    if "correlation_with_pnl" in quality_report:
        corr_rows = []
        for key, info in quality_report["correlation_with_pnl"].items():
            corr_rows.append({
                "field": key,
                "label": info["label"],
                "r": info["r"],
                "p_value": info["p_value"],
                "n": info["n"],
                "interpretation": info["interpretation"],
            })
        corr_df = pd.DataFrame(corr_rows)
        corr_path = os.path.join(output_dir, "backtest_correlation.csv")
        corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
        logger.info(f"相关性分析已保存到: {corr_path}")
    
    # 6. 维度组合
    if "dimension_combo" in quality_report:
        combo_df = pd.DataFrame.from_dict(quality_report["dimension_combo"], orient="index").reset_index()
        combo_df.columns = ["combo", "n", "win_rate", "avg_pnl"]
        combo_path = os.path.join(output_dir, "backtest_dimension_combo.csv")
        combo_df.to_csv(combo_path, index=False, encoding="utf-8-sig")
        logger.info(f"维度组合分析已保存到: {combo_path}")
    
    # 7. 未回踩直接上涨记录
    if missed_records:
        missed_rows = []
        for m in missed_records:
            missed_rows.append({
                "ts_code": m.ts_code,
                "name": m.name,
                "signal_time": m.signal_event_time,
                "upper_price": m.upper_price,
                "band_low": m.band_low,
                "band_high": m.band_high,
                "first_high_date": m.first_high_date,
                "first_high_price": m.first_high_after_signal,
                "price_change_pct": m.price_change_pct,
                "quality_score": m.quality_score,
                "quality_grade": m.quality_grade,
                "score_trend": m.score_trend,
                "score_candle": m.score_candle,
                "score_volume": m.score_volume,
                "score_freshness": m.score_freshness,
            })
        missed_df = pd.DataFrame(missed_rows)
        missed_path = os.path.join(output_dir, "backtest_missed_records.csv")
        missed_df.to_csv(missed_path, index=False, encoding="utf-8-sig")
        logger.info(f"未回踩记录已保存到: {missed_path}")
    
    # 8. 未回踩分析报告
    if missed_report and "error" not in missed_report:
        # 保存为JSON格式以便查看完整结构
        import json
        missed_report_path = os.path.join(output_dir, "backtest_missed_report.json")
        with open(missed_report_path, 'w', encoding='utf-8') as f:
            json.dump(missed_report, f, ensure_ascii=False, indent=2)
        logger.info(f"未回踩分析报告已保存到: {missed_report_path}")


def print_report(base_report: Dict, quality_report: Dict, missed_report: Dict):
    """打印控制台报告"""
    br = base_report
    print("\n" + "=" * 70)
    print("Breakout Pullback 回测报告（增强版）")
    print("=" * 70)
    print()
    print(f"总交易数:     {br['total_trades']}")
    print(f"已完成交易:   {br['completed']}")
    print(f"胜率:         {br['win_rate']}% ({br['winning']}胜/{br['losing']}负)")
    print()
    print("--- 收益 ---")
    print(f"平均收益:     {br['avg_pnl']}%")
    print(f"中位数收益:   {br['median_pnl']}%")
    print(f"平均盈利:     +{br['avg_win']}%")
    print(f"平均亏损:     {br['avg_loss']}%")
    print(f"最大盈利:     +{br['max_win']}%")
    print(f"最大亏损:     {br['max_loss']}%")
    print(f"盈亏比:       {br['profit_factor']}")
    print(f"总收益率:     {br['total_pnl']}%")
    print(f"平均持仓天数: {br['avg_holding_days']}天")
    print()
    
    if "by_grade" in quality_report:
        print("--- 按Grade分组 ---")
        for grade, stats in quality_report["by_grade"].items():
            print(f"  {grade}: {stats['count']}笔 | 胜率={stats['win_rate']}% | 平均={stats['avg_pnl']}% | 总计={stats['total_pnl']}%")
        print()
    
    if "correlation_with_pnl" in quality_report:
        print("--- 与收益率的相关性 ---")
        sorted_corr = sorted(quality_report["correlation_with_pnl"].items(), 
                             key=lambda x: abs(x[1]["r"]), reverse=True)
        for key, info in sorted_corr[:10]:
            marker = "***" if abs(info["r"]) > 0.05 and info["p_value"] < 0.05 else ""
            print(f"  {info['label']:12s}: r={info['r']:+.4f} (p={info['p_value']:.4f}) {info['interpretation']} {marker}")
        print()
    
    if "high_vs_low_score" in quality_report:
        hv = quality_report["high_vs_low_score"]
        print("--- 高分 vs 低分对比 ---")
        print(f"  高分组({hv['high_n']}笔): 胜率={hv['high_win_rate']}%, 平均={hv['high_avg_pnl']}%")
        print(f"  低分组({hv['low_n']}笔):  胜率={hv['low_win_rate']}%, 平均={hv['low_avg_pnl']}%")
        print(f"  差异: 胜率差={hv['diff_win_rate']}%, 收益差={hv['diff_avg_pnl']}%")
        print()
    
    if "dimension_combo" in quality_report:
        print("--- 维度组合表现 ---")
        for combo, stats in quality_report["dimension_combo"].items():
            print(f"  {combo}: {stats['n']}笔 | 胜率={stats['win_rate']}% | 平均={stats['avg_pnl']}%")
        print()
    
    # 未回踩直接上涨统计
    if missed_report and "error" not in missed_report:
        print("--- 未回踩直接上涨统计 ---")
        summary = missed_report.get("summary", {})
        print(f"  未回踩数量: {summary.get('total_missed', 0)}")
        print(f"  平均涨幅:   {summary.get('avg_price_change_pct', 0)}%")
        print(f"  中位数涨幅: {summary.get('median_price_change_pct', 0)}%")
        
        if "by_grade" in missed_report:
            print("\n  按Grade分布:")
            for grade, stats in missed_report["by_grade"].items():
                if stats['count'] > 0:
                    print(f"    {grade}: {stats['count']}个 | 平均涨幅={stats['avg_price_change']}%")
        
        if "dimension_analysis" in missed_report:
            print("\n  各维度平均分:")
            for dim, stats in missed_report["dimension_analysis"].items():
                print(f"    {dim}: {stats['mean']}")
        print()
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Breakout Pullback 回测（增强版）")
    parser.add_argument("--start-date", type=str, default=None, help="回测开始日期")
    parser.add_argument("--end-date", type=str, default=None, help="回测结束日期")
    parser.add_argument("--limit-stocks", type=int, default=None, help="限制股票数量（用于测试）")
    parser.add_argument("--output-dir", type=str, default="./backtest_output", help="输出目录")
    args = parser.parse_args()
    
    start_date = args.start_date or "2025-02-01"
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n{'='*70}")
    print(f"Breakout Pullback 回测（增强版 - 含质量分数分析）")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"{'='*70}\n")
    
    # Step 1: 加载信号
    print("Step 1/4: 加载翻多信号...")
    signals = load_signals(start_date=start_date, end_date=end_date)
    print(f"  加载了 {len(signals)} 个信号\n")
    
    if not signals:
        print("没有找到信号，退出")
        return
    
    # Step 2: 运行回测（按股票遍历 + MergedEngine）
    print("Step 2/4: 运行回测...")
    trades, missed_records = run_backtest(signals, limit_stocks=args.limit_stocks)
    print(f"  完成，共产生 {len(trades)} 笔交易, {len(missed_records)} 个未回踩\n")
    
    # Step 3: 生成报告
    print("Step 3/4: 生成分析报告...")
    base_report = generate_base_report(trades)
    quality_report = generate_quality_report(trades)
    missed_report = generate_missed_report(missed_records)
    print_report(base_report, quality_report, missed_report)
    
    # Step 4: 保存输出
    print("Step 4/4: 保存结果...")
    save_outputs(trades, missed_records, base_report, quality_report, missed_report, args.output_dir)
    print(f"\n所有结果已保存到: {args.output_dir}/")


if __name__ == "__main__":
    main()
