#!/usr/bin/env python3
"""
ATR Rope 周线突破选股脚本

Purpose:
    基于周线 ATR Rope 指标，检测"突破蓝色箱体上轨"的买入信号，
    并用 DSA VWAP direction=1 做多头确认。

Inputs:
    stock_k_data (周线K线数据, freq='w', open/high/low/close/volume)
    stock_adj_factor (复权因子，用于前复权转换)

Outputs:
    atr_week_selection (选股结果表)

How to Run:
    python selection/selection_atr_week.py              # 当天
    python selection/selection_atr_week.py 2026-05-22   # 指定日期
    python selection/selection_atr_week.py --test 002585 # 测试单只
    python selection/selection_atr_week.py --no-save     # 不写入数据库
    python selection/selection_atr_week.py --backfill 2025-07-01 2026-04-30  # 回补历史

Examples:
    python selection/selection_atr_week.py
    python selection/selection_atr_week.py --test 002585
    python selection/selection_atr_week.py 2026-05-22 --no-save

Side Effects:
    写入 atr_week_selection 表（幂等：同一日期先删后插）

================================================================================
【选股逻辑】

条件1 — 突破c_hi：
    ATR Rope dir=0（蓝色箱体）时，close 上穿 c_hi（蓝色箱体上轨）。
    使用 compute_atr_rope 输出的 evt_atr_rope_range_break_up 事件列检测（SSOT）。

条件2 — DSA VWAP 多头确认：
    突破当周 DSA VWAP direction = 1（多头方向）。

【核心计算】
    全部引用 features.atr_rope_event_factor_lab_v4.compute_atr_rope
    和 features.dynamic_swing_anchored_vwap.dynamic_swing_anchored_vwap
    本脚本只做数据准备、条件判断、结果保存，不重复计算逻辑（SSOT）

【除权处理】
    stock_k_data 存储的是不复权原始数据，get_weekly_kline_db 读取后
    通过 datasource.adj_factor.apply_adj_factor(df, ts_code, freq='w')
    做前复权转换（OHLC × 历史adj_factor/最新adj_factor），volume 不变

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, Optional, List
from tqdm import tqdm

from datasource.adj_factor import apply_adj_factor

# 核心计算逻辑（SSOT原则）：只引用，不重复实现
from features.atr_rope_event_factor_lab_v4 import compute_atr_rope, ATRRopeConfig
from features.dynamic_swing_anchored_vwap import dynamic_swing_anchored_vwap, DSAConfig

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "atr_week_selection"

# 周线数据配置
WEEKLY_BARS = 300  # 周线获取300根（约6年，足够ATR Rope和DSA VWAP预热）


def normalize_ts_code(ts_code: str) -> str:
    """标准化股票代码"""
    return str(ts_code).strip().upper().split('.')[0]


def get_weekly_kline_db(ts_code: str, bars: int = WEEKLY_BARS, end_date: Optional[date] = None) -> pd.DataFrame:
    """从数据库获取周线K线数据（前复权）"""
    symbol = normalize_ts_code(ts_code)
    if end_date is not None:
        sql = """
            SELECT bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE (ts_code = :ts_code OR ts_code = :ts_code_sh OR ts_code = :ts_code_sz) AND freq = 'w'
            AND DATE(bar_time) <= :end_date
            ORDER BY bar_time DESC
            LIMIT :bars
        """
        params = {
            'ts_code': symbol,
            'ts_code_sh': f'{symbol}.SH',
            'ts_code_sz': f'{symbol}.SZ',
            'bars': bars,
            'end_date': end_date.strftime('%Y-%m-%d')
        }
    else:
        sql = """
            SELECT bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE (ts_code = :ts_code OR ts_code = :ts_code_sh OR ts_code = :ts_code_sz) AND freq = 'w'
            ORDER BY bar_time DESC
            LIMIT :bars
        """
        params = {
            'ts_code': symbol,
            'ts_code_sh': f'{symbol}.SH',
            'ts_code_sz': f'{symbol}.SZ',
            'bars': bars
        }

    df = pd.read_sql(text(sql), engine, params=params)
    if not df.empty:
        df = df.sort_values('bar_time').set_index('bar_time')
        suffix = '.SH' if symbol.startswith(('6', '9')) else '.SZ'
        actual_ts_code = f'{symbol}{suffix}'
        df = apply_adj_factor(df, actual_ts_code, freq='w')
    return df


def check_volume_filter(df: pd.DataFrame, weeks: int = 5, min_amount: float = 100_000_000) -> bool:
    """
    检查成交额过滤条件

    Args:
        df: DataFrame with 'volume' and 'close' columns
        weeks: 计算过去N周的平均成交额，默认5周
        min_amount: 最小成交额阈值（元），默认1亿

    Returns:
        True if 平均成交额 >= min_amount
    """
    if len(df) < weeks:
        return False

    recent_df = df.tail(weeks)
    daily_amount = recent_df['volume'] * recent_df['close']
    avg_amount = daily_amount.mean()

    return avg_amount >= min_amount


def volume_zscore(vol: pd.Series, win: int = 20, loc: int = -1) -> float:
    """计算成交量Z-Score

    Args:
        vol: 成交量序列
        win: 滚动窗口大小
        loc: 目标行索引，默认-1（最后一行）
    """
    if len(vol) < win:
        return None

    if loc == -1:
        window = vol.iloc[-win:]
    else:
        end = loc + 1
        start = max(0, end - win)
        window = vol.iloc[start:end]

    mu_val = window.mean()
    sd_val = window.std(ddof=0)

    if sd_val == 0 or pd.isna(sd_val):
        return None

    vol_val = vol.iloc[loc]
    z = (vol_val - mu_val) / sd_val
    return float(z)


def compute_change_pct(df: pd.DataFrame, loc: int = -1) -> float:
    """计算指定位置的周涨跌幅

    Args:
        df: 周线 DataFrame
        loc: 目标行索引，默认-1（最后一行）
    """
    if len(df) < 2:
        return None
    if loc == 0:
        return None
    close_this = df['close'].iloc[loc]
    close_prev = df['close'].iloc[loc - 1]
    if close_prev == 0:
        return None
    change = float(close_this - close_prev) / float(close_prev) * 100
    return round(change, 2)


def batch_get_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    """批量获取股票名称（一次查询）"""
    if not ts_codes:
        return {}
    placeholders = ', '.join([f"'{c}'" for c in ts_codes])
    sql = text(f"SELECT ts_code, name FROM stock_pools WHERE ts_code IN ({placeholders})")
    with engine.connect() as conn:
        result = conn.execute(sql)
        return {row[0]: row[1] for row in result}


def parse_date(date_str: str) -> date:
    """解析日期字符串"""
    for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")


def process_stock(ts_code: str, selection_date: date) -> Optional[Dict]:
    """
    处理单只股票的周线 ATR Rope 突破选股逻辑

    选股逻辑：
        1. 当周 close 突破 c_hi（蓝色箱体上轨）
        2. 当周 DSA VWAP direction=1（多头确认）

    Returns: 信号字典，如果满足条件则返回结果，否则返回None
    """
    # 获取周线数据
    weekly_df = get_weekly_kline_db(ts_code, bars=WEEKLY_BARS, end_date=selection_date)
    if weekly_df.empty or len(weekly_df) < 60:
        return None

    # 调用 SSOT 核心计算（只引用，不重复实现）
    cfg = ATRRopeConfig(regime_lookback=55)
    try:
        result_df = compute_atr_rope(weekly_df, cfg)
    except Exception:
        return None

    if result_df.empty or len(result_df) < 2:
        return None

    # 计算 DSA VWAP
    try:
        dsa_vwap, dsa_dir_series, _, _ = dynamic_swing_anchored_vwap(weekly_df, DSAConfig())
    except Exception:
        return None

    # --- 条件1：当周突破 c_hi ---
    last = result_df.iloc[-1]
    if not result_df['evt_atr_rope_range_break_up'].iloc[-1]:
        return None

    # --- 条件2：DSA VWAP direction = 1（多头确认） ---
    dsa_dir_now = int(dsa_dir_series.iloc[-1]) if pd.notna(dsa_dir_series.iloc[-1]) else 0
    if dsa_dir_now != 1:
        return None

    # --- 提取指标状态 ---
    close_now = float(last['close'])

    def safe_float(val, default=None):
        return float(val) if pd.notna(val) else default

    atr_rope_dir_val = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None

    # DSA VWAP 偏离率
    dsa_vwap_now = dsa_vwap.iloc[-1]
    dsa_vwap_dev_pct = round(float(close_now / dsa_vwap_now - 1) * 100, 4) if pd.notna(dsa_vwap_now) and dsa_vwap_now != 0 else None

    bar_time = weekly_df.index[-1]

    return {
        'ts_code': ts_code,
        # ATR Rope 状态
        'rope_dir': atr_rope_dir_val,
        'rope_value': safe_float(last.get('atr_rope_rope')),
        'c_hi': safe_float(last.get('atr_rope_c_hi')),
        'c_lo': safe_float(last.get('atr_rope_c_lo')),
        'atr_value': safe_float(last.get('atr_rope_atr')),
        'rope_dev_pct': safe_float(last.get('factor_atr_rope_line_dev_pct')),
        'rope_dev_atr': safe_float(last.get('factor_atr_rope_line_dev_atr')),
        'range_width_pct': safe_float(last.get('factor_atr_rope_range_width_pct')) * 100 if atr_rope_dir_val == 0 and safe_float(last.get('factor_atr_rope_range_width_pct')) is not None else None,
        'range_pos_01': safe_float(last.get('factor_atr_rope_range_pos_01')) if atr_rope_dir_val == 0 else None,
        # DSA VWAP
        'dsa_dir': dsa_dir_now,
        'dsa_vwap': float(dsa_vwap_now) if pd.notna(dsa_vwap_now) else None,
        'dsa_vwap_dev_pct': dsa_vwap_dev_pct,
        # 观察项
        'change_pct': compute_change_pct(weekly_df),
        'vol_zscore': volume_zscore(weekly_df['volume'], win=20),
        'avg_amount_5w': float(((weekly_df['open'] + weekly_df['close']) / 2 * weekly_df['volume']).tail(5).mean()) / 1e8 if len(weekly_df) >= 5 else None,
        'signal_date': bar_time,
    }


def ensure_table_exists():
    """确保 atr_week_selection 表存在"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS atr_week_selection (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        signal_date DATE,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),

        -- ATR Rope 状态
        rope_dir INT,
        rope_value FLOAT,
        c_hi FLOAT,
        c_lo FLOAT,
        atr_value FLOAT,
        rope_dev_pct FLOAT,
        rope_dev_atr FLOAT,
        range_width_pct FLOAT,
        range_pos_01 FLOAT,

        -- DSA VWAP
        dsa_dir INT,
        dsa_vwap FLOAT,
        dsa_vwap_dev_pct FLOAT,

        -- 观察项
        change_pct FLOAT,
        vol_zscore FLOAT,
        avg_amount_5w FLOAT,

        batch_no INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_aw_selection_date ON atr_week_selection(selection_date);
    CREATE INDEX IF NOT EXISTS idx_aw_ts_code ON atr_week_selection(ts_code);
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

    # 先删除该日期的旧数据
    with engine.connect() as conn:
        delete_sql = text(f"DELETE FROM {SELECTION_TABLE} WHERE selection_date = :selection_date")
        result = conn.execute(delete_sql, {'selection_date': selection_date})
        conn.commit()
        if result.rowcount > 0:
            print(f"  清除旧数据: {result.rowcount} 条")

    # 准备插入数据
    records = []
    for _, row in df.iterrows():
        record = {
            'selection_date': selection_date,
            'signal_date': row['signal_date'],
            'ts_code': row['ts_code'],
            'stock_name': row.get('stock_name', '') or '',
            'rope_dir': int(row['rope_dir']) if pd.notna(row.get('rope_dir')) else None,
            'rope_value': float(row['rope_value']) if pd.notna(row.get('rope_value')) else None,
            'c_hi': float(row['c_hi']) if pd.notna(row.get('c_hi')) else None,
            'c_lo': float(row['c_lo']) if pd.notna(row.get('c_lo')) else None,
            'atr_value': float(row['atr_value']) if pd.notna(row.get('atr_value')) else None,
            'rope_dev_pct': float(row['rope_dev_pct']) if pd.notna(row.get('rope_dev_pct')) else None,
            'rope_dev_atr': float(row['rope_dev_atr']) if pd.notna(row.get('rope_dev_atr')) else None,
            'range_width_pct': float(row['range_width_pct']) if pd.notna(row.get('range_width_pct')) else None,
            'range_pos_01': float(row['range_pos_01']) if pd.notna(row.get('range_pos_01')) else None,
            'dsa_dir': int(row['dsa_dir']) if pd.notna(row.get('dsa_dir')) else None,
            'dsa_vwap': float(row['dsa_vwap']) if pd.notna(row.get('dsa_vwap')) else None,
            'dsa_vwap_dev_pct': float(row['dsa_vwap_dev_pct']) if pd.notna(row.get('dsa_vwap_dev_pct')) else None,
            'change_pct': float(row['change_pct']) if pd.notna(row.get('change_pct')) else None,
            'vol_zscore': float(row['vol_zscore']) if pd.notna(row.get('vol_zscore')) else None,
            'avg_amount_5w': float(row['avg_amount_5w']) if pd.notna(row.get('avg_amount_5w')) else None,
            'batch_no': int(row['batch_no']) if pd.notna(row.get('batch_no')) else None,
        }
        records.append(record)

    # 批量插入新数据
    if records:
        insert_df = pd.DataFrame(records)
        insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)
        print(f"  保存新数据: {len(records)} 条")
        return len(records)

    return 0


def select_atr_week_stocks(selection_date: Optional[date] = None, save_to_db: bool = True) -> pd.DataFrame:
    """
    根据周线 ATR Rope 突破逻辑选出满足条件的股票

    Args:
        selection_date: 选股日期，默认为当天
        save_to_db: 是否保存到数据库
    """
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("选股条件（ATR Rope 周线突破策略）：")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  条件1: 当周 close 突破 c_hi（蓝色箱体上轨）")
    print(f"  条件2: 当周 DSA VWAP direction=1（多头确认）")
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
    print("开始 ATR Rope 周线突破筛选...")
    print(f"  原股票数: {len(stock_list)}")

    filtered_results = []
    skip_stats = {'no_data': 0, 'no_breakout': 0, 'no_dsa_bull': 0}

    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="周线选股", unit="只"):
        ts_code = row['ts_code']

        # 获取数据并计算指标
        weekly_df = get_weekly_kline_db(ts_code, bars=WEEKLY_BARS, end_date=selection_date)
        if weekly_df.empty or len(weekly_df) < 60:
            skip_stats['no_data'] += 1
            continue

        # 计算 ATR Rope
        cfg = ATRRopeConfig(regime_lookback=55)
        try:
            result_df = compute_atr_rope(weekly_df, cfg)
        except Exception:
            skip_stats['no_data'] += 1
            continue

        if result_df.empty or len(result_df) < 2:
            skip_stats['no_data'] += 1
            continue

        # 计算 DSA VWAP
        try:
            dsa_vwap, dsa_dir_series, _, _ = dynamic_swing_anchored_vwap(weekly_df, DSAConfig())
        except Exception:
            skip_stats['no_data'] += 1
            continue

        # 条件1：当周突破 c_hi
        if not result_df['evt_atr_rope_range_break_up'].iloc[-1]:
            skip_stats['no_breakout'] += 1
            continue

        # 条件2：DSA VWAP direction = 1
        dsa_dir_now = int(dsa_dir_series.iloc[-1]) if pd.notna(dsa_dir_series.iloc[-1]) else 0
        if dsa_dir_now != 1:
            skip_stats['no_dsa_bull'] += 1
            continue

        # 提取指标状态
        last = result_df.iloc[-1]
        close_now = float(last['close'])

        def safe_float(val, default=None):
            return float(val) if pd.notna(val) else default

        atr_rope_dir_val = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None

        # DSA VWAP 偏离率
        dsa_vwap_now = dsa_vwap.iloc[-1]
        dsa_vwap_dev_pct = round(float(close_now / dsa_vwap_now - 1) * 100, 4) if pd.notna(dsa_vwap_now) and dsa_vwap_now != 0 else None

        bar_time = weekly_df.index[-1]

        filtered_results.append({
            'ts_code': ts_code,
            # ATR Rope 状态
            'rope_dir': atr_rope_dir_val,
            'rope_value': safe_float(last.get('atr_rope_rope')),
            'c_hi': safe_float(last.get('atr_rope_c_hi')),
            'c_lo': safe_float(last.get('atr_rope_c_lo')),
            'atr_value': safe_float(last.get('atr_rope_atr')),
            'rope_dev_pct': safe_float(last.get('factor_atr_rope_line_dev_pct')),
            'rope_dev_atr': safe_float(last.get('factor_atr_rope_line_dev_atr')),
            'range_width_pct': safe_float(last.get('factor_atr_rope_range_width_pct')) * 100 if atr_rope_dir_val == 0 and safe_float(last.get('factor_atr_rope_range_width_pct')) is not None else None,
            'range_pos_01': safe_float(last.get('factor_atr_rope_range_pos_01')) if atr_rope_dir_val == 0 else None,
            # DSA VWAP
            'dsa_dir': dsa_dir_now,
            'dsa_vwap': float(dsa_vwap_now) if pd.notna(dsa_vwap_now) else None,
            'dsa_vwap_dev_pct': dsa_vwap_dev_pct,
            # 观察项
            'change_pct': compute_change_pct(weekly_df),
            'vol_zscore': volume_zscore(weekly_df['volume'], win=20),
            'avg_amount_5w': float(((weekly_df['open'] + weekly_df['close']) / 2 * weekly_df['volume']).tail(5).mean()) / 1e8 if len(weekly_df) >= 5 else None,
            'signal_date': bar_time,
        })

    result_df_out = pd.DataFrame(filtered_results)

    if not result_df_out.empty:
        # 获取股票名称
        stock_names = batch_get_stock_names(result_df_out['ts_code'].tolist())
        result_df_out['stock_name'] = result_df_out['ts_code'].map(stock_names)

        # 分配批次号
        result_df_out['batch_no'] = (result_df_out.index // 10) + 1

    # 打印筛选统计
    print("\n" + "=" * 80)
    print("筛选统计：")
    print("=" * 80)
    print(f"  数据不足: {skip_stats['no_data']} 只")
    print(f"  无突破事件: {skip_stats['no_breakout']} 只")
    print(f"  DSA非多头: {skip_stats['no_dsa_bull']} 只")

    print("\n" + "=" * 80)
    print("选股结果汇总：")
    print("=" * 80)
    print(f"ATR Rope 周线突破筛选后: {len(result_df_out)} 只")

    if not result_df_out.empty:
        batch_count = result_df_out['batch_no'].max()
        print(f"\n批次信息：共 {batch_count} 批，每批10只股票")

        print("\n" + "=" * 80)
        print("前20名股票：")
        print("=" * 80)
        display_cols = [
            'ts_code', 'stock_name', 'rope_value', 'c_hi', 'c_lo',
            'dsa_vwap', 'change_pct', 'avg_amount_5w'
        ]
        print_cols = [c for c in display_cols if c in result_df_out.columns]
        print(result_df_out[print_cols].head(20).to_string(index=False))

    # 保存到数据库
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

    # 先看原始数据
    weekly_df = get_weekly_kline_db(ts_code, bars=WEEKLY_BARS, end_date=selection_date)
    if weekly_df.empty:
        print("无周线数据")
        return None
    print(f"周线数据: {len(weekly_df)} 根, 日期范围 {weekly_df.index[0]} ~ {weekly_df.index[-1]}")

    # 计算 ATR Rope
    cfg = ATRRopeConfig(regime_lookback=55)
    try:
        result_df = compute_atr_rope(weekly_df, cfg)
    except Exception as e:
        print(f"compute_atr_rope 异常: {e}")
        return None

    if result_df.empty:
        print("ATR Rope 计算结果为空")
        return None

    # 查看 ATR Rope 状态
    last = result_df.iloc[-1]
    print(f"\n最新周K ATR Rope 状态:")
    print(f"  close: {last['close']:.2f}")
    print(f"  dir: {int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else 'N/A'}")
    print(f"  rope: {last['atr_rope_rope']:.2f}" if pd.notna(last.get('atr_rope_rope')) else "  rope: N/A")
    print(f"  c_hi: {last['atr_rope_c_hi']:.2f}" if pd.notna(last.get('atr_rope_c_hi')) else "  c_hi: N/A")
    print(f"  c_lo: {last['atr_rope_c_lo']:.2f}" if pd.notna(last.get('atr_rope_c_lo')) else "  c_lo: N/A")
    print(f"  atr: {last['atr_rope_atr']:.2f}" if pd.notna(last.get('atr_rope_atr')) else "  atr: N/A")
    print(f"  突破c_hi: {'是' if result_df['evt_atr_rope_range_break_up'].iloc[-1] else '否'}")

    # 计算 DSA VWAP
    try:
        dsa_vwap, dsa_dir_series, _, _ = dynamic_swing_anchored_vwap(weekly_df, DSAConfig())
        dsa_dir_now = int(dsa_dir_series.iloc[-1]) if pd.notna(dsa_dir_series.iloc[-1]) else 0
        print(f"\nDSA VWAP: dir={dsa_dir_now}, vwap={dsa_vwap.iloc[-1]:.2f}" if pd.notna(dsa_vwap.iloc[-1]) else f"\nDSA VWAP: dir={dsa_dir_now}, vwap=N/A")
    except Exception as e:
        print(f"\nDSA VWAP 计算异常: {e}")
        dsa_dir_now = 0

    # 完整选股结果
    result = process_stock(ts_code, selection_date)

    if result:
        print("\n选股结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("\n该股票不满足选股条件")
        if not result_df['evt_atr_rope_range_break_up'].iloc[-1]:
            print("  原因: 当周未突破c_hi")
        elif dsa_dir_now != 1:
            print("  原因: DSA VWAP非多头方向")

    return result


def backfill_stock_events(ts_code: str, start_date: date, end_date: date) -> List[Dict]:
    """
    遍历单只股票在日期区间内的所有交易周，记录突破c_hi+DSA确认事件。

    优化：只调用一次 compute_atr_rope 和 dynamic_swing_anchored_vwap 计算全量数据，
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

    # 一次计算全量 ATR Rope
    cfg = ATRRopeConfig(regime_lookback=55)
    try:
        result_df = compute_atr_rope(weekly_df, cfg)
    except Exception:
        return []

    if result_df.empty or len(result_df) < 2:
        return []

    # 一次计算全量 DSA VWAP
    try:
        dsa_vwap, dsa_dir_series, _, _ = dynamic_swing_anchored_vwap(weekly_df, DSAConfig())
    except Exception:
        return []

    if not isinstance(result_df.index, pd.DatetimeIndex):
        result_df.index = pd.to_datetime(result_df.index)

    # 过滤出区间内的交易周
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    mask = (result_df.index >= start_ts) & (result_df.index <= end_ts)
    trade_dates = result_df.loc[mask].index.tolist()
    if not trade_dates:
        return []

    results = []
    for trade_dt in trade_dates:
        loc = result_df.index.get_loc(trade_dt)

        # 条件1：当周突破 c_hi
        if not result_df['evt_atr_rope_range_break_up'].iloc[loc]:
            continue

        # 条件2：DSA VWAP direction = 1
        if trade_dt not in dsa_dir_series.index:
            continue
        dsa_dir_now = int(dsa_dir_series.loc[trade_dt]) if pd.notna(dsa_dir_series.loc[trade_dt]) else 0
        if dsa_dir_now != 1:
            continue

        # 提取指标状态
        cur = result_df.iloc[loc]
        close_now = float(cur['close'])

        def safe_float(val, default=None):
            return float(val) if pd.notna(val) else default

        atr_rope_dir_val = int(cur['atr_rope_dir']) if pd.notna(cur.get('atr_rope_dir')) else None

        # DSA VWAP 偏离率
        dsa_vwap_now = dsa_vwap.loc[trade_dt] if trade_dt in dsa_vwap.index else None
        dsa_vwap_dev_pct = round(float(close_now / dsa_vwap_now - 1) * 100, 4) if pd.notna(dsa_vwap_now) and dsa_vwap_now != 0 else None

        results.append({
            'ts_code': ts_code,
            'selection_date': trade_dt,
            'signal_date': trade_dt,
            'rope_dir': atr_rope_dir_val,
            'rope_value': safe_float(cur.get('atr_rope_rope')),
            'c_hi': safe_float(cur.get('atr_rope_c_hi')),
            'c_lo': safe_float(cur.get('atr_rope_c_lo')),
            'atr_value': safe_float(cur.get('atr_rope_atr')),
            'rope_dev_pct': safe_float(cur.get('factor_atr_rope_line_dev_pct')),
            'rope_dev_atr': safe_float(cur.get('factor_atr_rope_line_dev_atr')),
            'range_width_pct': safe_float(cur.get('factor_atr_rope_range_width_pct')) * 100 if atr_rope_dir_val == 0 and safe_float(cur.get('factor_atr_rope_range_width_pct')) is not None else None,
            'range_pos_01': safe_float(cur.get('factor_atr_rope_range_pos_01')) if atr_rope_dir_val == 0 else None,
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

    from collections import defaultdict
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
                'rope_dir': int(rec['rope_dir']) if rec.get('rope_dir') is not None else None,
                'rope_value': float(rec['rope_value']) if rec.get('rope_value') is not None else None,
                'c_hi': float(rec['c_hi']) if rec.get('c_hi') is not None else None,
                'c_lo': float(rec['c_lo']) if rec.get('c_lo') is not None else None,
                'atr_value': float(rec['atr_value']) if rec.get('atr_value') is not None else None,
                'rope_dev_pct': float(rec['rope_dev_pct']) if rec.get('rope_dev_pct') is not None else None,
                'rope_dev_atr': float(rec['rope_dev_atr']) if rec.get('rope_dev_atr') is not None else None,
                'range_width_pct': float(rec['range_width_pct']) if rec.get('range_width_pct') is not None else None,
                'range_pos_01': float(rec['range_pos_01']) if rec.get('range_pos_01') is not None else None,
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
    回补指定日期区间内所有股票的 ATR Rope 周线突破事件。
    遍历个股，对每个交易周逐周计算，每处理完一只股票立即保存。

    Args:
        start_date: 起始日期（含）
        end_date: 结束日期（含）
        stock_list: 指定股票列表，None 则自动获取全市场
    """
    print("=" * 80)
    print("ATR Rope 周线突破事件回补")
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
        description='ATR Rope 周线突破选股工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/selection_atr_week.py                    # 使用当天日期选股
  python selection/selection_atr_week.py 2026-05-22         # 指定日期选股
  python selection/selection_atr_week.py 20260522           # 指定日期选股（无分隔符）
  python selection/selection_atr_week.py --test 002585      # 测试单只股票
  python selection/selection_atr_week.py --backfill 2025-07-01 2026-04-30  # 回补历史事件
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

    # 回补模式
    if args.backfill:
        start_date = parse_date(args.backfill[0])
        end_date = parse_date(args.backfill[1])
        backfill_range(start_date, end_date)
        sys.exit(0)

    # 解析日期
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
    print("ATR Rope 周线突破选股工具")
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print("=" * 80)

    # 测试模式
    if args.test:
        test_single_stock(args.test, selection_date)
        sys.exit(0)

    # 正常选股模式
    df = select_atr_week_stocks(selection_date=selection_date, save_to_db=not args.no_save)

    print("\n" + "=" * 80)
    print("选股完成")
    print(f"选股日期: {selection_date}")
    print(f"选中股票数: {len(df)}")
    print(f"查询SQL: SELECT * FROM {SELECTION_TABLE} WHERE selection_date = '{selection_date}'")
    print("=" * 80)

    return df


if __name__ == '__main__':
    main()
