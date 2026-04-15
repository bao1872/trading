#!/usr/bin/env python3
"""
BSM指标选股脚本

Purpose: 基于BSM（布林带动量）指标买点信号选股
Inputs: stock_k_data (日线/周线K线数据)
Outputs: stock_selection_results (选股结果)
How to Run:
    python selection/selection_ana.py              # 当天
    python selection/selection_ana.py 2026-04-10  # 指定日期
Side Effects: 写入 stock_selection_results 表

================================================================================
【选股条件】

保存条件：任意BSM买点信号为True
  - 日线反转（日线一类）：BBMACD上穿下轨
  - 日线突破（日线二类）：BBMACD上穿上轨
  - 周线反转（周线一类）：BBMACD上穿下轨
  - 周线突破（周线二类）：BBMACD上穿上轨

必要条件：周线 bbmacd >= banda_inf
  - 周线买点天然满足（因为上穿时下穿下轨时bbmacd必>=banda_inf）
  - 日线买点也需满足此条件（通过weekly_bbmacd判断）
  - 无论日线还是周线买点，都必须满足此必要条件

保存字段：BSM买点信号 + 布林带宽度Z-Score + 成交量Z-Score（win=20）

【选股日期】

选股日期只是标记，实际数据到"选股日期当天或之前最后一个交易日"：
  - 选股日期=2026-04-13(周一) → 数据到2026-04-13（当天是交易日）

周线只在周五收盘后更新：
  - 周一~周四：无周线数据 → 只计算日线买点（仍需满足必要条件）
  - 周五：有周线数据 → 日线+周线买点都计算（都需满足必要条件）

【保存逻辑】

按日线日期(daily_bar_time)分组保存：
  - 有周线数据时：weekly_bar_time = daily_bar_time = 周五日期（重合）
  - 无周线数据时：weekly_bar_time = None，用daily_bar_time作为selection_date
  - 保存时先删旧数据再插新数据
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

# 从 bbmacd_viewer 导入核心计算逻辑（SSOT原则）
from features.bbmacd_viewer import compute_bbmacd
# 使用 merged_dsa_atr_rope_bb_factors 中的 DSA 计算逻辑
from features.merged_dsa_atr_rope_bb_factors import compute_dsa, DSAConfig

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "stock_selection_results"


def volume_zscore(vol: pd.Series, win: int = 20) -> float:
    """
    计算成交量Z-Score
    z = (vol - SMA(vol, win)) / STDEV(vol, win)
    使用 population std (ddof=0) 对齐 TradingView
    """
    mu = vol.rolling(win, min_periods=win).mean()
    sd = vol.rolling(win, min_periods=win).std(ddof=0)
    if sd.iloc[-1] == 0 or pd.isna(sd.iloc[-1]):
        return None
    z = (vol.iloc[-1] - mu.iloc[-1]) / sd.iloc[-1]
    return float(z)


def get_stock_name(ts_code: str) -> str:
    """从stock_pools表获取股票名称"""
    sql = text("SELECT name FROM stock_pools WHERE ts_code = :ts_code LIMIT 1")
    with engine.connect() as conn:
        result = conn.execute(sql, {'ts_code': ts_code})
        row = result.fetchone()
        return row[0] if row else ''


def get_kline_data(ts_code: str, freq: str, bars: int = 120, end_date: Optional[date] = None) -> pd.DataFrame:
    """从数据库获取K线数据（默认120根）

    Args:
        ts_code: 股票代码
        freq: 频率 ('d'日线, 'w'周线)
        bars: 获取的K线数量
        end_date: 截止日期，只获取该日期及之前的K线
    """
    if end_date is not None:
        sql = """
            SELECT bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE ts_code = :ts_code AND freq = :freq
            AND DATE(bar_time) <= :end_date
            ORDER BY bar_time DESC
            LIMIT :bars
        """
        params = {
            'ts_code': ts_code,
            'freq': freq,
            'bars': bars,
            'end_date': end_date.strftime('%Y-%m-%d')
        }
    else:
        sql = """
            SELECT bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE ts_code = :ts_code AND freq = :freq
            ORDER BY bar_time DESC
            LIMIT :bars
        """
        params = {
            'ts_code': ts_code,
            'freq': freq,
            'bars': bars
        }

    df = pd.read_sql(text(sql), engine, params=params)
    if not df.empty:
        df = df.sort_values('bar_time').set_index('bar_time')
    return df


def has_weekly_data_for_date(target_date: date) -> bool:
    """检查指定日期是否有周线数据（周线数据只在周五收盘后更新）

    Args:
        target_date: 目标日期

    Returns:
        True 如果存在 freq='w' AND DATE(bar_time)=target_date 的记录，否则 False
    """
    sql = """
        SELECT EXISTS (
            SELECT 1 FROM stock_k_data
            WHERE freq = 'w' AND DATE(bar_time) = :target_date
            LIMIT 1
        )
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), {'target_date': target_date.strftime('%Y-%m-%d')})
        return bool(result.scalar())


def check_weekly_not_below_lower(ts_code: str, selection_date: Optional[date] = None) -> bool:
    """
    检查周线BSM是否不在下轨以下（必要条件）
    Args:
        ts_code: 股票代码
        selection_date: 选股日期，用于获取该日期之前的数据
    Returns: True-满足条件, False-不满足
    """
    df = get_kline_data(ts_code, 'w', bars=120, end_date=selection_date)
    if df.empty or len(df) < 26:  # 至少需要26根计算BSM
        return False
    
    bbmacd_df = compute_bbmacd(df)
    latest = bbmacd_df.iloc[-1]
    
    # 检查bbmacd是否为有效值
    if pd.isna(latest['bbmacd']) or pd.isna(latest['banda_inf']):
        return False
    
    return latest['bbmacd'] >= latest['banda_inf']


def batch_get_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    """批量获取股票名称（一次查询）"""
    if not ts_codes:
        return {}
    placeholders = ', '.join([f"'{c}'" for c in ts_codes])
    sql = text(f"SELECT ts_code, name FROM stock_pools WHERE ts_code IN ({placeholders})")
    with engine.connect() as conn:
        result = conn.execute(sql)
        return {row[0]: row[1] for row in result}


def cross_over(a: pd.Series, b: pd.Series) -> pd.Series:
    """判断上穿：当日>基准且前一日<=基准"""
    return (a > b) & (a.shift(1) <= b.shift(1))


def process_stock_bsm(ts_code: str, selection_date: date, has_weekly: bool) -> Optional[Dict]:
    """
    处理单只股票的BSM指标计算（合并所有查询和计算）

    必要条件：周线bbmacd不在下轨以下（bbmacd >= banda_inf）
    周线突破额外条件：
        - 周线DSA的dir必须为-1（空头趋势）
        - 当周收盘价 >= DSA VWAP * 0.85（不能小于DSA VWAP的15%）
    周线反转额外条件：
        - 周线DSA的dir必须为1（多头趋势）
        - BBMACD形成局部低点形态(t-2>t-1 且 t>t-1)
        - 当周收盘价 <= DSA VWAP * 1.10（不能大于DSA VWAP的10%）
    只有满足必要条件，才会计算并返回买点信号

    Returns: 信号字典，如果有任意BSM买点信号则返回结果，否则返回None
    """
    daily_df = get_kline_data(ts_code, 'd', bars=120, end_date=selection_date)
    if len(daily_df) < 26:
        return None

    daily_bbmacd = compute_bbmacd(daily_df)
    daily_bar_time = daily_df.index[-1]

    daily_reversal = cross_over(daily_bbmacd['bbmacd'], daily_bbmacd['banda_inf']).iloc[-1]
    daily_breakout = cross_over(daily_bbmacd['bbmacd'], daily_bbmacd['banda_supe']).iloc[-1]

    if has_weekly:
        weekly_df = get_kline_data(ts_code, 'w', bars=120, end_date=selection_date)
        if len(weekly_df) >= 26:
            weekly_bbmacd = compute_bbmacd(weekly_df)
            weekly_bar_time = weekly_df.index[-1]
            weekly_close = weekly_df['close'].iloc[-1]

            if weekly_bbmacd['bbmacd'].iloc[-1] >= weekly_bbmacd['banda_inf'].iloc[-1]:
                weekly_reversal = cross_over(weekly_bbmacd['bbmacd'], weekly_bbmacd['banda_inf']).iloc[-1]
                weekly_breakout = cross_over(weekly_bbmacd['bbmacd'], weekly_bbmacd['banda_supe']).iloc[-1]

                # 计算周线DSA（两个信号类型都可能用到）
                # 使用 merged_dsa_atr_rope_bb_factors 中的 compute_dsa
                try:
                    dsa_df, pivot_labels, segments = compute_dsa(weekly_df, DSAConfig())
                    current_dir = dsa_df['DSA_DIR'].iloc[-1]
                    current_vwap = dsa_df['DSA_VWAP'].iloc[-1]
                except Exception:
                    current_dir = None
                    current_vwap = None

                # 周线突破额外条件：
                # 1. DSA dir = -1
                # 2. 收盘价 >= DSA VWAP * 0.85
                if weekly_breakout:
                    if current_dir != -1:
                        weekly_breakout = False
                    elif current_vwap is not None and weekly_close < current_vwap * 0.85:
                        weekly_breakout = False

                # 周线反转额外条件：
                # 1. DSA dir = 1
                # 2. BBMACD形态 (t-2 > t-1 且 t > t-1)
                # 3. 收盘价 <= DSA VWAP * 1.10
                if weekly_reversal:
                    # 条件1: DSA dir = 1
                    if current_dir != 1:
                        weekly_reversal = False
                    else:
                        # 条件2: BBMACD形态
                        bbmacd_vals = weekly_bbmacd['bbmacd'].values
                        if len(bbmacd_vals) >= 3:
                            t_2 = bbmacd_vals[-3]
                            t_1 = bbmacd_vals[-2]
                            t = bbmacd_vals[-1]
                            if not (t_2 > t_1 and t > t_1):
                                weekly_reversal = False
                        else:
                            weekly_reversal = False

                        # 条件3: 收盘价 <= DSA VWAP * 1.10
                        if weekly_reversal and current_vwap is not None:
                            if weekly_close > current_vwap * 1.10:
                                weekly_reversal = False

                # 计算周线VWAP偏移比率（仅当有周线信号时）
                weekly_vwap_deviation = None
                if (weekly_reversal or weekly_breakout) and current_vwap is not None:
                    weekly_vwap_deviation = (weekly_close - current_vwap) / current_vwap

                if daily_reversal or daily_breakout or weekly_reversal or weekly_breakout:
                    return {
                        'ts_code': ts_code,
                        'daily_reversal_buy': daily_reversal,
                        'daily_breakout_buy': daily_breakout,
                        'weekly_reversal_buy': weekly_reversal,
                        'weekly_breakout_buy': weekly_breakout,
                        'daily_bb_width_zscore': daily_bbmacd['bb_width_zscore'].iloc[-1],
                        'weekly_bb_width_zscore': weekly_bbmacd['bb_width_zscore'].iloc[-1],
                        'daily_vol_zscore': volume_zscore(daily_df['volume'], win=20),
                        'weekly_vol_zscore': volume_zscore(weekly_df['volume'], win=20),
                        'daily_bar_time': daily_bar_time,
                        'weekly_bar_time': weekly_bar_time,
                        'weekly_vwap_deviation': weekly_vwap_deviation,
                    }
    else:
        # 无周线数据时（非周五），仍需检查周线必要条件
        weekly_df = get_kline_data(ts_code, 'w', bars=120, end_date=selection_date)
        if len(weekly_df) >= 26:
            weekly_bbmacd = compute_bbmacd(weekly_df)

            # 检查必要条件：周线bbmacd不在下轨以下
            if weekly_bbmacd['bbmacd'].iloc[-1] >= weekly_bbmacd['banda_inf'].iloc[-1]:
                if daily_reversal or daily_breakout:
                    return {
                        'ts_code': ts_code,
                        'daily_reversal_buy': daily_reversal,
                        'daily_breakout_buy': daily_breakout,
                        'weekly_reversal_buy': False,
                        'weekly_breakout_buy': False,
                        'daily_bb_width_zscore': daily_bbmacd['bb_width_zscore'].iloc[-1],
                        'weekly_bb_width_zscore': None,
                        'daily_vol_zscore': volume_zscore(daily_df['volume'], win=20),
                        'weekly_vol_zscore': None,
                        'daily_bar_time': daily_bar_time,
                        'weekly_bar_time': None,
                    }

    return None


def detect_bsm_signals(ts_code: str, selection_date: Optional[date] = None) -> Dict:
    """
    检测BSM买点信号
    Args:
        ts_code: 股票代码
        selection_date: 选股日期，用于获取该日期之前的数据
    Returns: 信号字典（包含买点信号、Z-Score值和实际使用的行情日期）
    """
    signals = {
        'daily_reversal_buy': False,      # 日线反转买点（一类）
        'daily_breakout_buy': False,      # 日线突破买点（二类）
        'weekly_reversal_buy': False,     # 周线反转买点（周线一类）
        'weekly_breakout_buy': False,     # 周线突破买点（周线二类）
        'daily_bb_width_zscore': None,    # 日线布林带宽度Z-Score
        'weekly_bb_width_zscore': None,   # 周线布林带宽度Z-Score
        'daily_bar_time': None,           # 实际使用的日线日期
        'weekly_bar_time': None,          # 实际使用的周线日期
    }

    # 日线信号
    daily_df = get_kline_data(ts_code, 'd', bars=120, end_date=selection_date)
    if len(daily_df) >= 26:
        daily_bbmacd = compute_bbmacd(daily_df)

        # 记录实际使用的日线日期
        signals['daily_bar_time'] = daily_df.index[-1]

        # 日线布林带宽度Z-Score
        signals['daily_bb_width_zscore'] = daily_bbmacd['bb_width_zscore'].iloc[-1]

        # 日线反转买点：上穿下轨（前一日<=下轨，当日>下轨）
        reversal_signal = cross_over(daily_bbmacd['bbmacd'], daily_bbmacd['banda_inf'])
        if reversal_signal.iloc[-1]:
            signals['daily_reversal_buy'] = True

        # 日线突破买点：上穿上轨（前一日<=上轨，当日>上轨）
        breakout_signal = cross_over(daily_bbmacd['bbmacd'], daily_bbmacd['banda_supe'])
        if breakout_signal.iloc[-1]:
            signals['daily_breakout_buy'] = True

    # 周线信号（只有存在周线数据时才计算，周线只在周五收盘后更新）
    if has_weekly_data_for_date(selection_date):
        weekly_df = get_kline_data(ts_code, 'w', bars=120, end_date=selection_date)
        if len(weekly_df) >= 26:
            weekly_bbmacd = compute_bbmacd(weekly_df)

            # 记录实际使用的周线日期
            signals['weekly_bar_time'] = weekly_df.index[-1]

            # 周线布林带宽度Z-Score
            signals['weekly_bb_width_zscore'] = weekly_bbmacd['bb_width_zscore'].iloc[-1]

            # 周线反转买点：上穿下轨
            weekly_reversal_signal = cross_over(weekly_bbmacd['bbmacd'], weekly_bbmacd['banda_inf'])
            if weekly_reversal_signal.iloc[-1]:
                signals['weekly_reversal_buy'] = True

            # 周线突破买点：上穿上轨
            weekly_breakout_signal = cross_over(weekly_bbmacd['bbmacd'], weekly_bbmacd['banda_supe'])
            if weekly_breakout_signal.iloc[-1]:
                signals['weekly_breakout_buy'] = True

    return signals


def ensure_table_exists(engine):
    """确保选股结果表存在（添加BSM字段和Z-Score字段）"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS stock_selection_results (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),
        report_date VARCHAR(8) NOT NULL,
        total_score FLOAT,
        margin_score FLOAT,
        scale_growth_score FLOAT,
        profitability_score FLOAT,
        profit_quality_score FLOAT,
        cash_creation_score FLOAT,
        asset_efficiency_score FLOAT,
        q_rev_yoy_delta FLOAT,
        q_np_parent_yoy_delta FLOAT,
        trend_consistency FLOAT,
        ann_date VARCHAR(8),
        -- BSM指标字段（BOOLEAN类型，可同时满足多个买点）
        daily_reversal_buy BOOLEAN DEFAULT FALSE,
        daily_breakout_buy BOOLEAN DEFAULT FALSE,
        weekly_reversal_buy BOOLEAN DEFAULT FALSE,
        weekly_breakout_buy BOOLEAN DEFAULT FALSE,
        -- 布林带宽度Z-Score字段
        daily_bb_width_zscore FLOAT,
        weekly_bb_width_zscore FLOAT,
        -- 成交量Z-Score字段
        daily_vol_zscore FLOAT,
        weekly_vol_zscore FLOAT,
        -- VWAP偏移比率字段（周线买点）
        weekly_vwap_deviation FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_selection_date ON stock_selection_results(selection_date);
    CREATE INDEX IF NOT EXISTS idx_selection_ts_code ON stock_selection_results(ts_code);
    CREATE INDEX IF NOT EXISTS idx_selection_report_date ON stock_selection_results(report_date);
    CREATE INDEX IF NOT EXISTS idx_selection_margin_score ON stock_selection_results(margin_score);
    -- BSM字段索引
    CREATE INDEX IF NOT EXISTS idx_selection_daily_reversal ON stock_selection_results(daily_reversal_buy);
    CREATE INDEX IF NOT EXISTS idx_selection_daily_breakout ON stock_selection_results(daily_breakout_buy);
    CREATE INDEX IF NOT EXISTS idx_selection_weekly_reversal ON stock_selection_results(weekly_reversal_buy);
    CREATE INDEX IF NOT EXISTS idx_selection_weekly_breakout ON stock_selection_results(weekly_breakout_buy);
    """
    with engine.connect() as conn:
        conn.execute(text(create_sql))
        conn.commit()


def save_to_database(df, selection_date):
    """保存选股结果到数据库（包含BSM字段和Z-Score字段），按日期覆盖"""
    if df.empty:
        print("数据为空，跳过数据库保存")
        return 0

    ensure_table_exists(engine)

    # 准备插入数据
    records = []
    for _, row in df.iterrows():
        record = {
            'selection_date': selection_date,
            'ts_code': row['ts_code'],
            'stock_name': row.get('stock_name', ''),
            'report_date': row['report_date'],
            'total_score': float(row['total_score']) if pd.notna(row['total_score']) else None,
            'margin_score': float(row['边际变化与持续性_score']) if pd.notna(row.get('边际变化与持续性_score')) else None,
            'scale_growth_score': float(row['规模与增长_score']) if pd.notna(row.get('规模与增长_score')) else None,
            'profitability_score': float(row['盈利能力_score']) if pd.notna(row.get('盈利能力_score')) else None,
            'profit_quality_score': float(row['利润质量_score']) if pd.notna(row.get('利润质量_score')) else None,
            'cash_creation_score': float(row['现金创造能力_score']) if pd.notna(row.get('现金创造能力_score')) else None,
            'asset_efficiency_score': float(row['资产效率与资金占用_score']) if pd.notna(row.get('资产效率与资金占用_score')) else None,
            'q_rev_yoy_delta': float(row['q_rev_yoy_delta']) if pd.notna(row.get('q_rev_yoy_delta')) else None,
            'q_np_parent_yoy_delta': float(row['q_np_parent_yoy_delta']) if pd.notna(row.get('q_np_parent_yoy_delta')) else None,
            'trend_consistency': float(row['trend_consistency']) if pd.notna(row.get('trend_consistency')) else None,
            'ann_date': row.get('ann_date', ''),
            # BSM字段（BOOLEAN类型）
            'daily_reversal_buy': bool(row.get('daily_reversal_buy', False)),
            'daily_breakout_buy': bool(row.get('daily_breakout_buy', False)),
            'weekly_reversal_buy': bool(row.get('weekly_reversal_buy', False)),
            'weekly_breakout_buy': bool(row.get('weekly_breakout_buy', False)),
            # Z-Score字段
            'daily_bb_width_zscore': float(row['daily_bb_width_zscore']) if pd.notna(row.get('daily_bb_width_zscore')) else None,
            'weekly_bb_width_zscore': float(row['weekly_bb_width_zscore']) if pd.notna(row.get('weekly_bb_width_zscore')) else None,
            # 成交量Z-Score字段
            'daily_vol_zscore': float(row['daily_vol_zscore']) if pd.notna(row.get('daily_vol_zscore')) else None,
            'weekly_vol_zscore': float(row['weekly_vol_zscore']) if pd.notna(row.get('weekly_vol_zscore')) else None,
        }
        records.append(record)

    # 先删除该日期的旧数据（幂等性：直接覆盖）
    with engine.connect() as conn:
        delete_sql = text(f"DELETE FROM {SELECTION_TABLE} WHERE selection_date = :selection_date")
        result = conn.execute(delete_sql, {'selection_date': selection_date})
        conn.commit()
        if result.rowcount > 0:
            print(f"  清除旧数据: {result.rowcount} 条")

    # 批量插入新数据
    insert_df = pd.DataFrame(records)
    insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)

    return len(records)


def save_to_database_by_weekly_date(df, fallback_date=None):
    """
    保存选股结果到数据库
    所有股票统一使用传入的 selection_date（即 fallback_date）作为选股日期
    不再根据实际周线日期分组保存，避免数据跨日期覆盖问题

    Args:
        df: 选股结果DataFrame
        fallback_date: 选股日期（即调用者传入的 selection_date）
    """
    if df.empty:
        print("数据为空，跳过数据库保存")
        return 0

    ensure_table_exists(engine)

    if fallback_date is None:
        print("错误: fallback_date（选股日期）不能为空")
        return 0

    sel_date = fallback_date
    print(f"\n  处理选股日期: {sel_date}, 股票数: {len(df)}")

    records = []
    for _, row in df.iterrows():
        record = {
            'selection_date': sel_date,
            'ts_code': row['ts_code'],
            'stock_name': row.get('stock_name', '') or '',
            'report_date': row.get('report_date', '') or '',
            'daily_reversal_buy': bool(row.get('daily_reversal_buy', False)),
            'daily_breakout_buy': bool(row.get('daily_breakout_buy', False)),
            'weekly_reversal_buy': bool(row.get('weekly_reversal_buy', False)),
            'weekly_breakout_buy': bool(row.get('weekly_breakout_buy', False)),
            'daily_bb_width_zscore': float(row['daily_bb_width_zscore']) if pd.notna(row.get('daily_bb_width_zscore')) else None,
            'weekly_bb_width_zscore': float(row['weekly_bb_width_zscore']) if pd.notna(row.get('weekly_bb_width_zscore')) else None,
            'daily_vol_zscore': float(row['daily_vol_zscore']) if pd.notna(row.get('daily_vol_zscore')) else None,
            'weekly_vol_zscore': float(row['weekly_vol_zscore']) if pd.notna(row.get('weekly_vol_zscore')) else None,
            'weekly_vwap_deviation': float(row['weekly_vwap_deviation']) if pd.notna(row.get('weekly_vwap_deviation')) else None,
        }
        records.append(record)

    with engine.connect() as conn:
        delete_sql = text(f"DELETE FROM {SELECTION_TABLE} WHERE selection_date = :selection_date")
        result = conn.execute(delete_sql, {'selection_date': sel_date})
        conn.commit()
        if result.rowcount > 0:
            print(f"    清除旧数据: {result.rowcount} 条")

    # 批量插入新数据
    if records:
        insert_df = pd.DataFrame(records)
        insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)
        print(f"    保存新数据: {len(records)} 条")
        print(f"\n  总计保存: {len(records)} 条")
        return len(records)

    return 0


def select_high_margin_stocks(selection_date: Optional[date] = None, save_to_db: bool = True):
    """
    根据BSM指标选出满足条件的股票

    Args:
        selection_date: 选股日期，默认为当天
        save_to_db: 是否保存到数据库
    """
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("选股条件：")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  数据来源: stock_k_data 日线数据")
    print(f"  必要条件: 周线BSM不在下轨以下 (bbmacd >= banda_inf)")
    print("=" * 80)

    with engine.connect() as conn:
        print("\n查询所有股票...")
        sql = text("""
            SELECT DISTINCT ts_code
            FROM stock_k_data
            WHERE freq = 'd' AND DATE(bar_time) = :selection_date
        """)
        stock_list = pd.read_sql(sql, conn, params={'selection_date': selection_date.strftime('%Y-%m-%d')})
        print(f"  找到 {len(stock_list)} 只股票")

    if len(stock_list) > 0:

        print("\n" + "=" * 80)
        print("开始BSM指标筛选...")
        print(f"  原股票数: {len(stock_list)}")

        has_weekly = has_weekly_data_for_date(selection_date)
        print(f"  当天有周线数据: {has_weekly}")

        filtered_results = []

        for _, row in stock_list.iterrows():
            ts_code = row['ts_code']

            result = process_stock_bsm(ts_code, selection_date, has_weekly)
            if result:
                filtered_results.append(result)

            if len(filtered_results) % 10 == 0:
                print(f"  已筛选 {len(filtered_results)} 只股票...")

        result_df = pd.DataFrame(filtered_results)

        if not result_df.empty:
            stock_names = batch_get_stock_names(result_df['ts_code'].tolist())
            result_df['stock_name'] = result_df['ts_code'].map(stock_names)

        print("\n" + "=" * 80)
        print("选股结果汇总：")
        print("=" * 80)
        print(f"BSM筛选后: {len(result_df)} 只")

        print(f"\nBSM买点信号统计：")
        print(f"  日线反转买点（一类）: {result_df['daily_reversal_buy'].sum() if len(result_df) > 0 else 0}")
        print(f"  日线突破买点（二类）: {result_df['daily_breakout_buy'].sum() if len(result_df) > 0 else 0}")
        print(f"  周线反转买点（周线一类）: {result_df['weekly_reversal_buy'].sum() if len(result_df) > 0 else 0}")
        print(f"  周线突破买点（周线二类）: {result_df['weekly_breakout_buy'].sum() if len(result_df) > 0 else 0}")

        if not result_df.empty:
            print("\n" + "=" * 80)
            print("前20名股票：")
            print("=" * 80)
            display_cols = ['ts_code', 'daily_reversal_buy', 'daily_breakout_buy', 'weekly_reversal_buy', 'weekly_breakout_buy']
            print(result_df[display_cols].head(20).to_string(index=False))
        
        # 保存到数据库
        if save_to_db:
            print("\n" + "-" * 80)
            print("保存到数据库...")
            # 按周线日期分组保存，每只股票使用其实际使用的周线日期
            # 如果weekly_bar_time为None（无周线数据），则使用selection_date作为替代
            saved_count = save_to_database_by_weekly_date(result_df, fallback_date=selection_date)
            print("-" * 80)

        return result_df
    else:
        print("\n未找到符合条件的股票")
        return pd.DataFrame()


def parse_date(date_str: str) -> date:
    """解析日期字符串"""
    for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='财务评分选股工具（含BSM指标筛选）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/selection_ana.py                    # 使用当天日期选股
  python selection/selection_ana.py 2025-12-31         # 指定日期选股
  python selection/selection_ana.py 20251231           # 指定日期选股（无分隔符）
        """
    )
    parser.add_argument(
        'date',
        nargs='?',
        help='选股日期 (格式: YYYY-MM-DD 或 YYYYMMDD)，默认为当天'
    )
    
    args = parser.parse_args()
    
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
    print("财务评分选股工具（含BSM指标筛选）")
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print("=" * 80)
    
    df = select_high_margin_stocks(selection_date=selection_date, save_to_db=True)

    print("\n" + "=" * 80)
    print("选股完成")
    print(f"选股日期: {selection_date}")
    print(f"选中股票数: {len(df)}")
    print(f"查询SQL: SELECT * FROM stock_selection_results WHERE selection_date = '{selection_date}'")
    print("=" * 80)

    return df


if __name__ == '__main__':
    main()
