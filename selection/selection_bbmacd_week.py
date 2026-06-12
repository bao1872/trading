#!/usr/bin/env python3
"""
BBMACD 周线拐点选股脚本

Purpose:
    基于周线 BBMACD 指标，检测"V 型拐点"信号：
    bbmacd[t-1] < bbmacd[t-2] 且 bbmacd[t] > bbmacd[t-1]，
    即 bbmacd 先下降再拐头向上的拐点时刻。

Inputs:
    stock_k_data (周线K线数据, freq='w', open/high/low/close/volume)
    stock_adj_factor (复权因子，用于前复权转换)

Outputs:
    bbmacd_week_selection (选股结果表)

How to Run:
    python selection/selection_bbmacd_week.py              # 当天
    python selection/selection_bbmacd_week.py 2026-05-22   # 指定日期
    python selection/selection_bbmacd_week.py --test 002585 # 测试单只
    python selection/selection_bbmacd_week.py --no-save     # 不写入数据库
    python selection/selection_bbmacd_week.py --backfill 2025-07-01 2026-04-30  # 回补历史

Examples:
    python selection/selection_bbmacd_week.py
    python selection/selection_bbmacd_week.py --test 002585
    python selection/selection_bbmacd_week.py 2026-05-22 --no-save

Side Effects:
    写入 bbmacd_week_selection 表（幂等：同一日期先删后插）

================================================================================
【选股逻辑】

条件1 — BBMACD V型拐点：
    bbmacd[t-1] < bbmacd[t-2]  （前一周比再前一周低，下降趋势）
    bbmacd[t] > bbmacd[t-1]    （当周比前一周高，拐头向上）

条件2 — DSA VWAP 多头确认：
    当周 DSA VWAP direction = 1（多头方向）

【核心计算】
    bbmacd 计算引用 features.dsa_bbmacd_24factors_viewer.compute_bbmacd（SSOT）
    DSA VWAP 引用 features.dynamic_swing_anchored_vwap.dynamic_swing_anchored_vwap
    通用函数引用 selection.selection_atr_week（SSOT，不重复实现）

【除权处理】
    复用 selection_atr_week.get_weekly_kline_db（已内含前复权处理）

【选股日期】
    选股日期只是标记，实际数据到"选股日期当天或之前最后一个交易周"
    周线数据通过 python app/build_dataset.py --update --period w 每周更新

【保存逻辑】
    按选股日期统一保存，先删旧数据再插新数据（幂等性）
================================================================================
"""

import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, Optional, List
from tqdm import tqdm

from features.dsa_bbmacd_24factors_viewer import compute_bbmacd
from features.dynamic_swing_anchored_vwap import dynamic_swing_anchored_vwap, DSAConfig
from selection.selection_atr_week import (
    get_weekly_kline_db,
    normalize_ts_code,
    batch_get_stock_names,
    parse_date,
    volume_zscore,
    compute_change_pct,
    check_volume_filter,
)

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "bbmacd_week_selection"

WEEKLY_BARS = 300


def process_stock(ts_code: str, selection_date: date) -> Optional[Dict]:
    """
    处理单只股票的周线 BBMACD 拐点选股逻辑

    选股逻辑：
        bbmacd[t-1] < bbmacd[t-2] 且 bbmacd[t] > bbmacd[t-1]

    Returns: 信号字典，如果满足条件则返回结果，否则返回None
    """
    weekly_df = get_weekly_kline_db(ts_code, bars=WEEKLY_BARS, end_date=selection_date)
    if weekly_df.empty or len(weekly_df) < 60:
        return None

    try:
        bbmacd_result = compute_bbmacd(weekly_df)
    except Exception:
        return None

    if bbmacd_result.empty or len(bbmacd_result) < 3:
        return None

    bbmacd_now = bbmacd_result['bbmacd'].iloc[-1]
    bbmacd_prev = bbmacd_result['bbmacd'].iloc[-2]
    bbmacd_prev2 = bbmacd_result['bbmacd'].iloc[-3]

    if pd.isna(bbmacd_now) or pd.isna(bbmacd_prev) or pd.isna(bbmacd_prev2):
        return None

    if not (bbmacd_prev < bbmacd_prev2 and bbmacd_now > bbmacd_prev):
        return None

    try:
        dsa_vwap, dsa_dir_series, _, _ = dynamic_swing_anchored_vwap(weekly_df, DSAConfig())
    except Exception:
        return None

    dsa_dir_now = int(dsa_dir_series.iloc[-1]) if len(dsa_dir_series) > 0 and pd.notna(dsa_dir_series.iloc[-1]) else 0
    if dsa_dir_now != 1:
        return None

    close_now = float(weekly_df['close'].iloc[-1])
    dsa_vwap_now = dsa_vwap.iloc[-1] if len(dsa_vwap) > 0 else None
    dsa_vwap_dev_pct = round(float(close_now / dsa_vwap_now - 1) * 100, 4) if pd.notna(dsa_vwap_now) and dsa_vwap_now != 0 else None

    bar_time = weekly_df.index[-1]

    def safe_float(val, default=None):
        return float(val) if pd.notna(val) else default

    return {
        'ts_code': ts_code,
        'bbmacd': safe_float(bbmacd_now),
        'bbmacd_prev': safe_float(bbmacd_prev),
        'bbmacd_prev2': safe_float(bbmacd_prev2),
        'bbmacd_avg': safe_float(bbmacd_result['bbmacd_avg'].iloc[-1]),
        'bbmacd_upper': safe_float(bbmacd_result['bbmacd_upper'].iloc[-1]),
        'bbmacd_lower': safe_float(bbmacd_result['bbmacd_lower'].iloc[-1]),
        'bbmacd_state': int(bbmacd_result['bbmacd_state'].iloc[-1]) if pd.notna(bbmacd_result['bbmacd_state'].iloc[-1]) else None,
        'bbmacd_band_pos_01': safe_float(bbmacd_result['bbmacd_band_pos_01'].iloc[-1]),
        'bbmacd_bandwidth_zscore': safe_float(bbmacd_result['bbmacd_bandwidth_zscore'].iloc[-1]),
        'dsa_dir': dsa_dir_now,
        'dsa_vwap': float(dsa_vwap_now) if pd.notna(dsa_vwap_now) else None,
        'dsa_vwap_dev_pct': dsa_vwap_dev_pct,
        'change_pct': compute_change_pct(weekly_df),
        'vol_zscore': volume_zscore(weekly_df['volume'], win=20),
        'avg_amount_5w': float(((weekly_df['open'] + weekly_df['close']) / 2 * weekly_df['volume']).tail(5).mean()) / 1e8 if len(weekly_df) >= 5 else None,
        'signal_date': bar_time,
    }


def ensure_table_exists():
    """确保 bbmacd_week_selection 表存在"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS bbmacd_week_selection (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        signal_date DATE,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),

        bbmacd FLOAT,
        bbmacd_prev FLOAT,
        bbmacd_prev2 FLOAT,
        bbmacd_avg FLOAT,
        bbmacd_upper FLOAT,
        bbmacd_lower FLOAT,
        bbmacd_state INT,
        bbmacd_band_pos_01 FLOAT,
        bbmacd_bandwidth_zscore FLOAT,

        dsa_dir INT,
        dsa_vwap FLOAT,
        dsa_vwap_dev_pct FLOAT,

        change_pct FLOAT,
        vol_zscore FLOAT,
        avg_amount_5w FLOAT,

        batch_no INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_bw_selection_date ON bbmacd_week_selection(selection_date);
    CREATE INDEX IF NOT EXISTS idx_bw_ts_code ON bbmacd_week_selection(ts_code);
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()


def save_to_database(df: pd.DataFrame, selection_date: date) -> int:
    """保存选股结果到数据库（幂等性：先删后插）"""
    if df.empty:
        print("数据为空，跳过数据库保存")
        return 0

    ensure_table_exists()

    with engine.connect() as conn:
        delete_sql = text(f"DELETE FROM {SELECTION_TABLE} WHERE selection_date = :selection_date")
        result = conn.execute(delete_sql, {'selection_date': selection_date})
        conn.commit()
        if result.rowcount > 0:
            print(f"  清除旧数据: {result.rowcount} 条")

    records = []
    for _, row in df.iterrows():
        record = {
            'selection_date': selection_date,
            'signal_date': row['signal_date'],
            'ts_code': row['ts_code'],
            'stock_name': row.get('stock_name', '') or '',
            'bbmacd': float(row['bbmacd']) if pd.notna(row.get('bbmacd')) else None,
            'bbmacd_prev': float(row['bbmacd_prev']) if pd.notna(row.get('bbmacd_prev')) else None,
            'bbmacd_prev2': float(row['bbmacd_prev2']) if pd.notna(row.get('bbmacd_prev2')) else None,
            'bbmacd_avg': float(row['bbmacd_avg']) if pd.notna(row.get('bbmacd_avg')) else None,
            'bbmacd_upper': float(row['bbmacd_upper']) if pd.notna(row.get('bbmacd_upper')) else None,
            'bbmacd_lower': float(row['bbmacd_lower']) if pd.notna(row.get('bbmacd_lower')) else None,
            'bbmacd_state': int(row['bbmacd_state']) if pd.notna(row.get('bbmacd_state')) else None,
            'bbmacd_band_pos_01': float(row['bbmacd_band_pos_01']) if pd.notna(row.get('bbmacd_band_pos_01')) else None,
            'bbmacd_bandwidth_zscore': float(row['bbmacd_bandwidth_zscore']) if pd.notna(row.get('bbmacd_bandwidth_zscore')) else None,
            'dsa_dir': int(row['dsa_dir']) if pd.notna(row.get('dsa_dir')) else None,
            'dsa_vwap': float(row['dsa_vwap']) if pd.notna(row.get('dsa_vwap')) else None,
            'dsa_vwap_dev_pct': float(row['dsa_vwap_dev_pct']) if pd.notna(row.get('dsa_vwap_dev_pct')) else None,
            'change_pct': float(row['change_pct']) if pd.notna(row.get('change_pct')) else None,
            'vol_zscore': float(row['vol_zscore']) if pd.notna(row.get('vol_zscore')) else None,
            'avg_amount_5w': float(row['avg_amount_5w']) if pd.notna(row.get('avg_amount_5w')) else None,
            'batch_no': int(row['batch_no']) if pd.notna(row.get('batch_no')) else None,
        }
        records.append(record)

    if records:
        insert_df = pd.DataFrame(records)
        insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)
        print(f"  保存新数据: {len(records)} 条")
        return len(records)

    return 0


def select_bbmacd_week_stocks(selection_date: Optional[date] = None, save_to_db: bool = True) -> pd.DataFrame:
    """
    根据周线 BBMACD 拐点逻辑选出满足条件的股票

    Args:
        selection_date: 选股日期，默认为当天
        save_to_db: 是否保存到数据库
    """
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("选股条件（BBMACD 周线拐点策略）：")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  条件: bbmacd[t-1] < bbmacd[t-2] 且 bbmacd[t] > bbmacd[t-1]")
    print("=" * 80)

    with engine.connect() as conn:
        print("\n查询所有股票（周线）...")
        sql = text("""
            SELECT DISTINCT ts_code
            FROM stock_k_data
            WHERE freq = 'w' AND DATE(bar_time) = :selection_date
        """)
        stock_list = pd.read_sql(sql, conn, params={'selection_date': selection_date.strftime('%Y-%m-%d')})
        print(f"  找到 {len(stock_list)} 只股票")

    if len(stock_list) == 0:
        print("\n未找到符合条件的股票")
        return pd.DataFrame()

    print("\n" + "=" * 80)
    print("开始 BBMACD 周线拐点筛选...")
    print(f"  原股票数: {len(stock_list)}")

    filtered_results = []
    skip_stats = {'no_data': 0, 'no_signal': 0, 'no_dsa_bull': 0}

    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="周线选股", unit="只"):
        ts_code = row['ts_code']

        weekly_df = get_weekly_kline_db(ts_code, bars=WEEKLY_BARS, end_date=selection_date)
        if weekly_df.empty or len(weekly_df) < 60:
            skip_stats['no_data'] += 1
            continue

        try:
            bbmacd_result = compute_bbmacd(weekly_df)
        except Exception:
            skip_stats['no_data'] += 1
            continue

        if bbmacd_result.empty or len(bbmacd_result) < 3:
            skip_stats['no_data'] += 1
            continue

        bbmacd_now = bbmacd_result['bbmacd'].iloc[-1]
        bbmacd_prev = bbmacd_result['bbmacd'].iloc[-2]
        bbmacd_prev2 = bbmacd_result['bbmacd'].iloc[-3]

        if pd.isna(bbmacd_now) or pd.isna(bbmacd_prev) or pd.isna(bbmacd_prev2):
            skip_stats['no_data'] += 1
            continue

        if not (bbmacd_prev < bbmacd_prev2 and bbmacd_now > bbmacd_prev):
            skip_stats['no_signal'] += 1
            continue

        try:
            dsa_vwap, dsa_dir_series, _, _ = dynamic_swing_anchored_vwap(weekly_df, DSAConfig())
        except Exception:
            skip_stats['no_data'] += 1
            continue

        close_now = float(weekly_df['close'].iloc[-1])
        dsa_dir_now = int(dsa_dir_series.iloc[-1]) if len(dsa_dir_series) > 0 and pd.notna(dsa_dir_series.iloc[-1]) else 0
        if dsa_dir_now != 1:
            skip_stats['no_dsa_bull'] += 1
            continue
        dsa_vwap_now = dsa_vwap.iloc[-1] if len(dsa_vwap) > 0 else None
        dsa_vwap_dev_pct = round(float(close_now / dsa_vwap_now - 1) * 100, 4) if pd.notna(dsa_vwap_now) and dsa_vwap_now != 0 else None

        bar_time = weekly_df.index[-1]

        def safe_float(val, default=None):
            return float(val) if pd.notna(val) else default

        filtered_results.append({
            'ts_code': ts_code,
            'bbmacd': safe_float(bbmacd_now),
            'bbmacd_prev': safe_float(bbmacd_prev),
            'bbmacd_prev2': safe_float(bbmacd_prev2),
            'bbmacd_avg': safe_float(bbmacd_result['bbmacd_avg'].iloc[-1]),
            'bbmacd_upper': safe_float(bbmacd_result['bbmacd_upper'].iloc[-1]),
            'bbmacd_lower': safe_float(bbmacd_result['bbmacd_lower'].iloc[-1]),
            'bbmacd_state': int(bbmacd_result['bbmacd_state'].iloc[-1]) if pd.notna(bbmacd_result['bbmacd_state'].iloc[-1]) else None,
            'bbmacd_band_pos_01': safe_float(bbmacd_result['bbmacd_band_pos_01'].iloc[-1]),
            'bbmacd_bandwidth_zscore': safe_float(bbmacd_result['bbmacd_bandwidth_zscore'].iloc[-1]),
            'dsa_dir': dsa_dir_now,
            'dsa_vwap': float(dsa_vwap_now) if pd.notna(dsa_vwap_now) else None,
            'dsa_vwap_dev_pct': dsa_vwap_dev_pct,
            'change_pct': compute_change_pct(weekly_df),
            'vol_zscore': volume_zscore(weekly_df['volume'], win=20),
            'avg_amount_5w': float(((weekly_df['open'] + weekly_df['close']) / 2 * weekly_df['volume']).tail(5).mean()) / 1e8 if len(weekly_df) >= 5 else None,
            'signal_date': bar_time,
        })

    result_df_out = pd.DataFrame(filtered_results)

    if not result_df_out.empty:
        stock_names = batch_get_stock_names(result_df_out['ts_code'].tolist())
        result_df_out['stock_name'] = result_df_out['ts_code'].map(stock_names)
        result_df_out['batch_no'] = (result_df_out.index // 10) + 1

    print("\n" + "=" * 80)
    print("筛选统计：")
    print("=" * 80)
    print(f"  数据不足: {skip_stats['no_data']} 只")
    print(f"  无拐点信号: {skip_stats['no_signal']} 只")
    print(f"  DSA非多头: {skip_stats['no_dsa_bull']} 只")

    print("\n" + "=" * 80)
    print("选股结果汇总：")
    print("=" * 80)
    print(f"BBMACD 周线拐点筛选后: {len(result_df_out)} 只")

    if not result_df_out.empty:
        batch_count = result_df_out['batch_no'].max()
        print(f"\n批次信息：共 {batch_count} 批，每批10只股票")

        print("\n" + "=" * 80)
        print("前20名股票：")
        print("=" * 80)
        display_cols = [
            'ts_code', 'stock_name', 'bbmacd', 'bbmacd_prev', 'bbmacd_prev2',
            'bbmacd_state', 'change_pct', 'avg_amount_5w'
        ]
        print_cols = [c for c in display_cols if c in result_df_out.columns]
        print(result_df_out[print_cols].head(20).to_string(index=False))

    if save_to_db:
        print("\n" + "-" * 80)
        print("保存到数据库...")
        saved_count = save_to_database(result_df_out, selection_date)
        print("-" * 80)

    return result_df_out


def test_single_stock(ts_code: str, selection_date: date):
    """测试单只股票的计算逻辑"""
    print("\n" + "=" * 80)
    print(f"测试单只股票: {ts_code}")
    print(f"选股日期: {selection_date}")
    print("=" * 80)

    weekly_df = get_weekly_kline_db(ts_code, bars=WEEKLY_BARS, end_date=selection_date)
    if weekly_df.empty:
        print("无周线数据")
        return None
    print(f"周线数据: {len(weekly_df)} 根, 日期范围 {weekly_df.index[0]} ~ {weekly_df.index[-1]}")

    try:
        bbmacd_result = compute_bbmacd(weekly_df)
    except Exception as e:
        print(f"compute_bbmacd 异常: {e}")
        return None

    if bbmacd_result.empty:
        print("BBMACD 计算结果为空")
        return None

    last = bbmacd_result.iloc[-1]
    prev = bbmacd_result.iloc[-2]
    prev2 = bbmacd_result.iloc[-3]

    print(f"\n最近3周 BBMACD 值:")
    print(f"  t-2 ({bbmacd_result.index[-3]}): bbmacd={prev2['bbmacd']:.4f}")
    print(f"  t-1 ({bbmacd_result.index[-2]}): bbmacd={prev['bbmacd']:.4f}")
    print(f"  t   ({bbmacd_result.index[-1]}): bbmacd={last['bbmacd']:.4f}")
    print(f"\n拐点条件检查:")
    print(f"  bbmacd[t-1] < bbmacd[t-2]: {prev['bbmacd']:.4f} < {prev2['bbmacd']:.4f} = {prev['bbmacd'] < prev2['bbmacd']}")
    print(f"  bbmacd[t] > bbmacd[t-1]:   {last['bbmacd']:.4f} > {prev['bbmacd']:.4f} = {last['bbmacd'] > prev['bbmacd']}")
    print(f"  bbmacd_state: {int(last['bbmacd_state']) if pd.notna(last['bbmacd_state']) else 'N/A'}")
    print(f"  bbmacd_band_pos_01: {last['bbmacd_band_pos_01']:.4f}" if pd.notna(last['bbmacd_band_pos_01']) else "  bbmacd_band_pos_01: N/A")

    try:
        dsa_vwap, dsa_dir_series, _, _ = dynamic_swing_anchored_vwap(weekly_df, DSAConfig())
        dsa_dir_now = int(dsa_dir_series.iloc[-1]) if len(dsa_dir_series) > 0 and pd.notna(dsa_dir_series.iloc[-1]) else 0
        print(f"\nDSA VWAP: dir={dsa_dir_now}, vwap={dsa_vwap.iloc[-1]:.2f}" if len(dsa_vwap) > 0 and pd.notna(dsa_vwap.iloc[-1]) else f"\nDSA VWAP: N/A")
    except Exception as e:
        print(f"\nDSA VWAP 计算异常: {e}")

    result = process_stock(ts_code, selection_date)
    if result:
        print("\n选股结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("\n该股票不满足选股条件")

    return result


def backfill_stock_events(ts_code: str, start_date: date, end_date: date) -> List[Dict]:
    """
    遍历单只股票在日期区间内的所有交易周，记录 BBMACD 拐点事件。

    优化：只调用一次 compute_bbmacd 和 dynamic_swing_anchored_vwap 计算全量数据，
    然后遍历结果行判断信号，避免对每个交易周重复计算。

    Args:
        ts_code: 股票代码
        start_date: 起始日期（含）
        end_date: 结束日期（含）

    Returns:
        该股票在区间内所有触发周的记录列表
    """
    weekly_df = get_weekly_kline_db(ts_code, bars=WEEKLY_BARS, end_date=end_date)
    if weekly_df.empty or len(weekly_df) < 60:
        return []

    if not isinstance(weekly_df.index, pd.DatetimeIndex):
        weekly_df.index = pd.to_datetime(weekly_df.index)

    try:
        bbmacd_result = compute_bbmacd(weekly_df)
    except Exception:
        return []

    if bbmacd_result.empty or len(bbmacd_result) < 3:
        return []

    try:
        dsa_vwap, dsa_dir_series, _, _ = dynamic_swing_anchored_vwap(weekly_df, DSAConfig())
    except Exception:
        dsa_vwap = pd.Series(dtype=float, index=weekly_df.index)
        dsa_dir_series = pd.Series(dtype=float, index=weekly_df.index)

    if not isinstance(bbmacd_result.index, pd.DatetimeIndex):
        bbmacd_result.index = pd.to_datetime(bbmacd_result.index)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    mask = (bbmacd_result.index >= start_ts) & (bbmacd_result.index <= end_ts)
    trade_dates = bbmacd_result.loc[mask].index.tolist()
    if not trade_dates:
        return []

    results = []
    for trade_dt in trade_dates:
        loc = bbmacd_result.index.get_loc(trade_dt)
        if loc < 2:
            continue

        bbmacd_now = bbmacd_result['bbmacd'].iloc[loc]
        bbmacd_prev = bbmacd_result['bbmacd'].iloc[loc - 1]
        bbmacd_prev2 = bbmacd_result['bbmacd'].iloc[loc - 2]

        if pd.isna(bbmacd_now) or pd.isna(bbmacd_prev) or pd.isna(bbmacd_prev2):
            continue

        if not (bbmacd_prev < bbmacd_prev2 and bbmacd_now > bbmacd_prev):
            continue

        cur = bbmacd_result.iloc[loc]
        close_now = float(weekly_df['close'].iloc[loc])

        def safe_float(val, default=None):
            return float(val) if pd.notna(val) else default

        dsa_dir_now = int(dsa_dir_series.iloc[loc]) if len(dsa_dir_series) > loc and pd.notna(dsa_dir_series.iloc[loc]) else 0
        if dsa_dir_now != 1:
            continue
        dsa_vwap_now = dsa_vwap.iloc[loc] if len(dsa_vwap) > loc else None
        dsa_vwap_dev_pct = round(float(close_now / dsa_vwap_now - 1) * 100, 4) if pd.notna(dsa_vwap_now) and dsa_vwap_now != 0 else None

        results.append({
            'ts_code': ts_code,
            'selection_date': trade_dt,
            'signal_date': trade_dt,
            'bbmacd': safe_float(bbmacd_now),
            'bbmacd_prev': safe_float(bbmacd_prev),
            'bbmacd_prev2': safe_float(bbmacd_prev2),
            'bbmacd_avg': safe_float(cur.get('bbmacd_avg')),
            'bbmacd_upper': safe_float(cur.get('bbmacd_upper')),
            'bbmacd_lower': safe_float(cur.get('bbmacd_lower')),
            'bbmacd_state': int(cur['bbmacd_state']) if pd.notna(cur.get('bbmacd_state')) else None,
            'bbmacd_band_pos_01': safe_float(cur.get('bbmacd_band_pos_01')),
            'bbmacd_bandwidth_zscore': safe_float(cur.get('bbmacd_bandwidth_zscore')),
            'dsa_dir': dsa_dir_now,
            'dsa_vwap': float(dsa_vwap_now) if pd.notna(dsa_vwap_now) else None,
            'dsa_vwap_dev_pct': dsa_vwap_dev_pct,
            'change_pct': compute_change_pct(weekly_df, loc=loc),
            'vol_zscore': volume_zscore(weekly_df['volume'], win=20, loc=loc),
            'avg_amount_5w': float(((weekly_df['open'] + weekly_df['close']) / 2 * weekly_df['volume']).iloc[max(0, loc-4):loc+1].mean()) / 1e8 if loc >= 0 and len(weekly_df) >= 5 else None,
        })

    return results


def _save_single_stock_records(records: List[Dict], stock_name_map: Dict[str, str]):
    """
    将单只股票的触发记录立即保存到数据库。
    按日期分组，每个日期先删后插（幂等性）。
    """
    if not records:
        return 0

    ensure_table_exists()

    date_groups = defaultdict(list)
    for rec in records:
        dt = rec['selection_date']
        date_groups[dt].append(rec)

    total_saved = 0
    for dt, day_records in date_groups.items():
        ts_codes_in_day = [r['ts_code'] for r in day_records]
        placeholders = ', '.join([f"'{c}'" for c in ts_codes_in_day])
        with engine.connect() as conn:
            delete_sql = text(
                f"DELETE FROM {SELECTION_TABLE} "
                f"WHERE selection_date = :selection_date AND ts_code IN ({placeholders})"
            )
            conn.execute(delete_sql, {'selection_date': dt})
            conn.commit()

        insert_records = []
        for rec in day_records:
            insert_records.append({
                'selection_date': dt,
                'signal_date': rec['signal_date'],
                'ts_code': rec['ts_code'],
                'stock_name': stock_name_map.get(rec['ts_code'], '') or '',
                'bbmacd': float(rec['bbmacd']) if rec.get('bbmacd') is not None else None,
                'bbmacd_prev': float(rec['bbmacd_prev']) if rec.get('bbmacd_prev') is not None else None,
                'bbmacd_prev2': float(rec['bbmacd_prev2']) if rec.get('bbmacd_prev2') is not None else None,
                'bbmacd_avg': float(rec['bbmacd_avg']) if rec.get('bbmacd_avg') is not None else None,
                'bbmacd_upper': float(rec['bbmacd_upper']) if rec.get('bbmacd_upper') is not None else None,
                'bbmacd_lower': float(rec['bbmacd_lower']) if rec.get('bbmacd_lower') is not None else None,
                'bbmacd_state': int(rec['bbmacd_state']) if rec.get('bbmacd_state') is not None else None,
                'bbmacd_band_pos_01': float(rec['bbmacd_band_pos_01']) if rec.get('bbmacd_band_pos_01') is not None else None,
                'bbmacd_bandwidth_zscore': float(rec['bbmacd_bandwidth_zscore']) if rec.get('bbmacd_bandwidth_zscore') is not None else None,
                'dsa_dir': int(rec['dsa_dir']) if rec.get('dsa_dir') is not None else None,
                'dsa_vwap': float(rec['dsa_vwap']) if rec.get('dsa_vwap') is not None else None,
                'dsa_vwap_dev_pct': float(rec['dsa_vwap_dev_pct']) if rec.get('dsa_vwap_dev_pct') is not None else None,
                'change_pct': float(rec['change_pct']) if rec.get('change_pct') is not None else None,
                'vol_zscore': float(rec['vol_zscore']) if rec.get('vol_zscore') is not None else None,
                'avg_amount_5w': float(rec['avg_amount_5w']) if rec.get('avg_amount_5w') is not None else None,
                'batch_no': None,
            })

        if insert_records:
            insert_df = pd.DataFrame(insert_records)
            insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)
            total_saved += len(insert_records)

    return total_saved


def backfill_range(start_date: date, end_date: date, stock_list: Optional[List[str]] = None):
    """
    回补指定日期区间内所有股票的 BBMACD 周线拐点事件。
    遍历个股，对每个交易周逐周计算，每处理完一只股票立即保存。

    Args:
        start_date: 起始日期（含）
        end_date: 结束日期（含）
        stock_list: 指定股票列表，None 则自动获取全市场
    """
    print("=" * 80)
    print("BBMACD 周线拐点事件回补")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  模式: 遍历个股，逐周计算事件，每只股票立即保存")
    print("=" * 80)

    if stock_list is None:
        with engine.connect() as conn:
            print("\n查询股票列表...")
            sql = text("""
                SELECT DISTINCT ts_code
                FROM stock_k_data
                WHERE freq = 'w'
                AND DATE(bar_time) BETWEEN :start_date AND :end_date
            """)
            df = pd.read_sql(sql, conn, params={
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })
            stock_list = df['ts_code'].tolist()
            print(f"  找到 {len(stock_list)} 只股票")

    if not stock_list:
        print("无股票可处理")
        return

    total_saved = 0
    total_stocks_with_events = 0

    for ts_code in tqdm(stock_list, desc="回补个股", unit="只"):
        records = backfill_stock_events(ts_code, start_date, end_date)
        if records:
            stock_name_map = batch_get_stock_names([ts_code])
            saved = _save_single_stock_records(records, stock_name_map)
            total_saved += saved
            total_stocks_with_events += 1

    print("-" * 80)
    print(f"\n回补完成")
    print(f"共处理 {len(stock_list)} 只股票，其中 {total_stocks_with_events} 只有触发事件")
    print(f"共保存 {total_saved} 条记录")
    print(f"日期范围: {start_date} ~ {end_date}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='BBMACD 周线拐点选股工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/selection_bbmacd_week.py                    # 使用当天日期选股
  python selection/selection_bbmacd_week.py 2026-05-22         # 指定日期选股
  python selection/selection_bbmacd_week.py --test 002585      # 测试单只股票
  python selection/selection_bbmacd_week.py --backfill 2025-07-01 2026-04-30  # 回补历史事件
        """
    )
    parser.add_argument(
        'date',
        nargs='?',
        help='选股日期 (格式: YYYY-MM-DD 或 YYYYMMDD)，默认为当天'
    )
    parser.add_argument(
        '--test',
        help='测试单只股票，例如: --test 002585'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='不保存到数据库（仅显示结果）'
    )
    parser.add_argument(
        '--backfill',
        nargs=2,
        metavar=('START_DATE', 'END_DATE'),
        help='回补历史事件，遍历个股逐周计算，例如: --backfill 2025-07-01 2026-04-30'
    )

    args = parser.parse_args()

    if args.backfill:
        start_date = parse_date(args.backfill[0])
        end_date = parse_date(args.backfill[1])
        backfill_range(start_date, end_date)
        sys.exit(0)

    if args.date:
        try:
            selection_date = parse_date(args.date)
        except ValueError as e:
            print(f"错误: {e}")
            print("日期格式应为: YYYY-MM-DD 或 YYYYMMDD")
            sys.exit(1)
    else:
        selection_date = date.today()

    print("\n" + "=" * 80)
    print("BBMACD 周线拐点选股工具")
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print("=" * 80)

    if args.test:
        test_single_stock(args.test, selection_date)
        sys.exit(0)

    df = select_bbmacd_week_stocks(selection_date=selection_date, save_to_db=not args.no_save)

    print("\n" + "=" * 80)
    print("选股完成")
    print(f"选股日期: {selection_date}")
    print(f"选中股票数: {len(df)}")
    print(f"查询SQL: SELECT * FROM {SELECTION_TABLE} WHERE selection_date = '{selection_date}'")
    print("=" * 80)

    return df


if __name__ == '__main__':
    main()
