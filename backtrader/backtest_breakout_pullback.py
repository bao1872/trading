# -*- coding: utf-8 -*-
"""
Breakout Pullback 回测脚本

Purpose:
    回测突破后回踩买入策略的收益率
    
    策略逻辑：
    1. 从 breakout_dir_turn_events 读取翻多信号（已包含买入区间）
    2. 当价格进入 [dir_turn_band_low, dir_turn_band_high] 时买入
    3. 当 rope_dir 变为 0 或 -1 时卖出
    
Inputs:
    - breakout_dir_turn_events: 翻多事件表（含买入区间）
    - stock_k_data: 日线行情数据（含 rope_dir）
    
Outputs:
    - 控制台报告：胜率、盈亏比、总收益率等
    - 可选：详细交易记录 CSV

How to Run:
    # 回测所有日线信号
    python backtrader/backtest_breakout_pullback.py
    
    # 指定日期范围
    python backtrader/backtest_breakout_pullback.py --start-date 2025-02-01 --end-date 2026-04-04
    
    # 输出交易记录到CSV
    python backtrader/backtest_breakout_pullback.py --output trades.csv

Side Effects:
    - 只读取数据库，不写入数据
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATABASE_URL
from datasource.database import get_session, query_df
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backtest_pullback")


@dataclass
class Signal:
    """翻多信号"""
    ts_code: str
    name: str
    event_time: str
    dir_turn_upper_price: float
    dir_turn_atr_raw: float
    dir_turn_tol_price: float
    dir_turn_band_low: float
    dir_turn_band_high: float


@dataclass
class Trade:
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


@dataclass
class WatchlistItem:
    """观察列表项"""
    signal: Signal
    status: str = "waiting"  # waiting / entered / exited
    trade: Trade = field(default_factory=Trade)


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
        
        signals.append(Signal(
            ts_code=row["ts_code"],
            name=row["name"] or "",
            event_time=row["event_time"],
            dir_turn_upper_price=row["dir_turn_upper_price"],
            dir_turn_atr_raw=row["dir_turn_atr_raw"],
            dir_turn_tol_price=row["dir_turn_tol_price"],
            dir_turn_band_low=row["dir_turn_band_low"],
            dir_turn_band_high=row["dir_turn_band_high"],
        ))
    
    return signals


def load_kline_data(ts_codes: List[str], start_date: str) -> Dict[str, pd.DataFrame]:
    """加载股票日线数据"""
    result = {}
    
    with get_session() as session:
        for ts_code in ts_codes:
            code = ts_code.split(".")[0] if "." in ts_code else ts_code
            
            filters = {"freq": "d", "ts_code": ts_code}
            
            df = query_df(
                session,
                table_name="stock_k_data",
                columns=["bar_time", "open", "high", "low", "close", "volume"],
                filters=filters,
                order_by="+bar_time"
            )
            
            if not df.empty:
                df["bar_time"] = pd.to_datetime(df["bar_time"])
                df = df.set_index("bar_time")
                
                # 统一时区后再过滤
                start_ts = pd.Timestamp(start_date)
                if df.index.tz is not None:
                    start_ts = start_ts.tz_localize(df.index.tz)
                
                # 只保留信号日期之后的数据
                df = df[df.index >= start_ts]
                
                if not df.empty:
                    result[ts_code] = df
    
    return result


def run_backtest(signals: List[Signal], kline_data: Dict[str, pd.DataFrame]) -> List[Trade]:
    """
    运行回测
    
    核心逻辑：
    1. 对每个信号加入观察列表
    2. 逐日遍历：
       - 检查观察中的信号是否满足买入条件（价格进入 buy zone）
       - 检查持仓是否满足卖出条件（rope_dir 变为 0 或 -1）
    """
    trades: List[Trade] = []
    watchlist: List[WatchlistItem] = []
    
    # 初始化观察列表
    for sig in signals:
        item = WatchlistItem(signal=sig)
        item.trade.ts_code = sig.ts_code
        item.trade.name = sig.name
        item.trade.signal_event_time = sig.event_time
        watchlist.append(item)
    
    logger.info(f"观察列表: {len(watchlist)} 个信号")
    
    # 收集所有需要检查的日期和股票代码
    all_dates = set()
    for ts_code, df in kline_data.items():
        all_dates.update(df.index.strftime("%Y-%m-%d"))
    
    sorted_dates = sorted(all_dates)
    logger.info(f"交易日期范围: {sorted_dates[0]} ~ {sorted_dates[-1]}, 共 {len(sorted_dates)} 天")
    
    # 逐日遍历
    for date_str in tqdm(sorted_dates, desc="回测进度"):
        date_ts = pd.Timestamp(date_str)
        
        # 处理每个观察中的信号
        for item in watchlist:
            if item.status == "exited":
                continue
            
            ts_code = item.signal.ts_code
            kdf = kline_data.get(ts_code)
            
            if kdf is None:
                continue
            
            # 获取当天的K线数据（可能有多条，取第一条）
            mask = kdf.index.strftime("%Y-%m-%d") == date_str
            if not mask.any():
                continue
            
            bar = kdf[mask].iloc[0]
            
            if item.status == "waiting":
                # 检查买入条件：价格进入 [band_low, band_high]
                band_low = item.signal.dir_turn_band_low
                band_high = item.signal.dir_turn_band_high
                
                if bar["low"] <= band_high and bar["high"] >= band_low:
                    # 价格触及买入区间，以收盘价买入
                    item.status = "entered"
                    item.trade.entry_date = date_str
                    item.trade.entry_price = bar["close"]
                    logger.debug(f"买入: {ts_code} @ {bar['close']:.2f}, 区间=[{band_low:.2f}, {band_high:.2f}]")
                    
            elif item.status == "entered":
                # 检查卖出条件：rope_dir 变为 0 或 -1
                # 注意：rope_dir 在因子引擎输出中，这里用 close 和 upper 的关系近似判断
                # 如果有 rope_dir 字段则直接使用
                
                # 尝试从 kdf 中获取 rope_dir
                if "rope_dir" in kdf.columns:
                    current_dir = bar.get("rope_dir", None)
                    if current_dir is not None and float(current_dir) in [0, -1]:
                        item.status = "exited"
                        item.trade.exit_date = date_str
                        item.trade.exit_price = bar["close"]
                        
                        if item.trade.entry_price and item.trade.entry_price > 0:
                            pnl_pct = (item.trade.exit_price - item.trade.entry_price) / item.trade.entry_price * 100
                            item.trade.pnl_pct = round(pnl_pct, 2)
                            item.trade.exit_reason = f"dir_turned_{int(current_dir)}"
                        
                        trades.append(item.trade)
                        logger.debug(f"卖出: {ts_code} @ {bar['close']:.2f}, 原因={item.trade.exit_reason}")
                
                # 备选卖出条件：如果持仓超过20天强制退出
                elif item.trade.entry_date:
                    entry_date = pd.Timestamp(item.trade.entry_date)
                    holding_days = (date_ts - entry_date).days
                    if holding_days > 20:
                        item.status = "exited"
                        item.trade.exit_date = date_str
                        item.trade.exit_price = bar["close"]
                        
                        if item.trade.entry_price and item.trade.entry_price > 0:
                            pnl_pct = (item.trade.exit_price - item.trade.entry_price) / item.trade.entry_price * 100
                            item.trade.pnl_pct = round(pnl_pct, 2)
                            item.trade.exit_reason = f"timeout_{holding_days}days"
                        
                        trades.append(item.trade)
                        logger.debug(f"超时卖出: {ts_code} @ {bar['close']:.2f}")
    
    return trades


def generate_report(trades: List[Trade]) -> Dict:
    """生成回测报告"""
    if not trades:
        return {
            "total_trades": 0,
            "completed_trades": 0,
            "win_rate": 0,
            "avg_pnl_pct": 0,
            "total_return_pct": 0,
            "max_win": 0,
            "max_loss": 0,
        }
    
    completed = [t for t in trades if t.pnl_pct is not None]
    winning = [t for t in completed if t.pnl_pct > 0]
    losing = [t for t in completed if t.pnl_pct <= 0]
    
    total_pnl = sum(t.pnl_pct or 0 for t in completed)
    
    report = {
        "total_trades": len(trades),
        "completed_trades": len(completed),
        "still_holding": len(trades) - len(completed),
        "win_count": len(winning),
        "loss_count": len(losing),
        "win_rate": round(len(winning) / len(completed) * 100, 2) if completed else 0,
        "avg_pnl_pct": round(total_pnl / len(completed), 2) if completed else 0,
        "avg_win_pct": round(sum(t.pnl_pct for t in winning) / len(winning), 2) if winning else 0,
        "avg_loss_pct": round(sum(t.pnl_pct for t in losing) / len(losing), 2) if losing else 0,
        "max_win_pct": round(max((t.pnl_pct or 0) for t in completed), 2) if completed else 0,
        "max_loss_pct": round(min((t.pnl_pct or 0) for t in completed), 2) if completed else 0,
        "total_return_pct": round(total_pnl, 2),
        "profit_factor": round(abs(sum(t.pnl_pct for t in winning)) / abs(sum(t.pnl_pct for t in losing)), 2) if losing else float('inf'),
    }
    
    return report


def print_report(report: Dict):
    """打印回测报告"""
    print("\n" + "=" * 60)
    print("Breakout Pullback 回测报告")
    print("=" * 60)
    print()
    print(f"总交易数:     {report['total_trades']}")
    print(f"已完成交易:   {report['completed_trades']}")
    print(f"仍持有:       {report['still_holding']}")
    print()
    print("--- 盈亏统计 ---")
    print(f"胜率:         {report['win_rate']}%")
    print(f"盈利次数:     {report['win_count']}")
    print(f"亏损次数:     {report['loss_count']}")
    print()
    print("--- 收益统计 ---")
    print(f"平均收益:     {report['avg_pnl_pct']}%")
    print(f"平均盈利:     {report['avg_win_pct']}%")
    print(f"平均亏损:     {report['avg_loss_pct']}%")
    print(f"最大盈利:     {report['max_win_pct']}%")
    print(f"最大亏损:     {report['max_loss_pct']}%")
    print(f"总收益率:     {report['total_return_pct']}%")
    print(f"盈亏比:       {report['profit_factor']}")
    print()
    print("=" * 60)


def save_trades_to_csv(trades: List[Trade], filepath: str):
    """保存交易记录到 CSV"""
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
            "exit_reason": t.exit_reason,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info(f"交易记录已保存到: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Breakout Pullback 回测")
    parser.add_argument("--start-date", type=str, default=None, help="回测开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="回测结束日期 (YYYY-MM-DD)")
    parser.add_argument("--limit-signals", type=int, default=None, help="限制信号数量（用于测试）")
    parser.add_argument("--output", type=str, default=None, help="输出交易记录 CSV 文件路径")
    args = parser.parse_args()
    
    # 设置默认日期范围
    start_date = args.start_date or "2025-02-01"
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"Breakout Pullback 回测")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"{'='*60}\n")
    
    # Step 1: 加载信号
    print("Step 1/3: 加载翻多信号...")
    signals = load_signals(start_date=start_date, end_date=end_date)
    
    if args.limit_signals:
        signals = signals[:args.limit_signals]
        print(f"  限制为前 {args.limit_signals} 个信号")
    
    print(f"  加载了 {len(signals)} 个信号\n")
    
    if not signals:
        print("没有找到信号，退出")
        return
    
    # Step 2: 加载K线数据
    print("Step 2/3: 加载K线数据...")
    ts_codes = list(set(s.ts_code for s in signals))
    print(f"  需要加载 {len(ts_codes)} 只股票的日线数据")
    
    kline_data = load_kline_data(ts_codes, start_date=start_date)
    print(f"  成功加载 {len(kline_data)} 只股票数据\n")
    
    # Step 3: 运行回测
    print("Step 3/3: 运行回测...")
    trades = run_backtest(signals, kline_data)
    print(f"  完成，共产生 {len(trades)} 笔交易\n")
    
    # 生成报告
    report = generate_report(trades)
    print_report(report)
    
    # 保存交易记录
    if args.output:
        save_trades_to_csv(trades, args.output)


if __name__ == "__main__":
    main()
