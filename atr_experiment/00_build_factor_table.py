#!/usr/bin/env python3
"""
ATR Rope 因子数据构建脚本

Purpose:
    遍历所有股票，调用 compute_atr_rope 计算全量因子和事件，
    保存到 atr_rope_factors 表，为 GBDT 训练和因子挖掘准备数据。

Inputs:
    stock_k_data (日线K线数据)

Outputs:
    atr_rope_factors (全量因子表，约 400 万行)

How to Run:
    python atr_experiment/00_build_factor_table.py
    python atr_experiment/00_build_factor_table.py --start-date 2023-01-01
    python atr_experiment/00_build_factor_table.py --test 600547
    python atr_experiment/00_build_factor_table.py --no-save

Examples:
    python atr_experiment/00_build_factor_table.py
    python atr_experiment/00_build_factor_table.py --test 000001

Side Effects:
    写入 atr_rope_factors 表（幂等：按 ts_code 先删后插）

================================================================================
【分批次读写策略】
- 写入：每处理完一只股票立即写入数据库（单股约 800 行），不累积在内存中
- 删除：按 ts_code 逐只删除旧数据，不用大批量 DELETE
- 单次内存中最多持有 1 只股票的因子数据（约 800 行 × 40 列）
================================================================================
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from tqdm import tqdm

# 核心计算逻辑（SSOT）
from features.atr_rope_event_factor_lab_v4 import compute_atr_rope, ATRRopeConfig

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

FACTORS_TABLE = "atr_rope_factors"
DAILY_BARS = 800

# compute_atr_rope 输出的所有字段（排除 open/high/low/close/volume 等原始K线字段）
FACTOR_COLUMNS = [
    # 基础指标
    "atr_rope_tr", "atr_rope_atr_raw", "atr_rope_atr",
    "atr_rope_rope", "atr_rope_upper", "atr_rope_lower",
    "atr_rope_dir", "atr_rope_c_hi", "atr_rope_c_lo", "atr_rope_ff",
    # 方向事件
    "evt_atr_rope_dir_change", "evt_atr_rope_dir_red_to_blue",
    "evt_atr_rope_dir_blue_to_green", "evt_atr_rope_dir_green_to_blue",
    "evt_atr_rope_dir_blue_to_red", "evt_atr_rope_dir_red_to_green",
    "evt_atr_rope_dir_green_to_red", "evt_atr_rope_turn_up",
    "evt_atr_rope_turn_down", "evt_atr_rope_turn_flat",
    # Rope 趋势线事件
    "evt_atr_rope_cross_rope", "evt_atr_rope_line_touch_rope",
    "evt_atr_rope_line_cross_up", "evt_atr_rope_line_cross_down",
    "evt_atr_rope_line_retest_green", "evt_atr_rope_line_retest_red",
    # 蓝色震荡区事件
    "evt_atr_rope_range_start", "evt_atr_rope_range_touch_high",
    "evt_atr_rope_range_touch_low", "evt_atr_rope_range_break_up",
    "evt_atr_rope_range_break_down", "evt_atr_rope_range_reenter_from_above",
    "evt_atr_rope_range_reenter_from_below",
    # Regime 事件
    "evt_atr_rope_regime_change", "evt_atr_rope_regime_to_bull",
    "evt_atr_rope_regime_to_bear", "evt_atr_rope_regime_to_range",
    # 状态因子
    "factor_atr_rope_state_dir", "factor_atr_rope_state_dir_prev",
    "factor_atr_rope_state_dir_bars",
    # Regime 因子
    "factor_atr_rope_regime", "factor_atr_rope_regime_prev",
    "factor_atr_rope_regime_bars",
    "factor_atr_rope_green_ratio_20", "factor_atr_rope_red_ratio_20",
    "factor_atr_rope_blue_ratio_20", "factor_atr_rope_slope_pct_20",
    "factor_atr_rope_regime_strength",
    # 趋势线/蓝区因子
    "factor_atr_rope_line_dev_pct", "factor_atr_rope_line_dev_atr",
    "factor_atr_rope_range_high", "factor_atr_rope_range_low",
    "factor_atr_rope_range_mid", "factor_atr_rope_range_pos_01",
    "factor_atr_rope_range_width_pct", "factor_atr_rope_range_width_atr",
]


def normalize_ts_code(ts_code: str) -> str:
    """标准化股票代码为6位纯数字"""
    return str(ts_code).strip().upper().split('.')[0]


def get_kline_data_db(ts_code: str, bars: int = DAILY_BARS, end_date: date = None) -> pd.DataFrame:
    """从数据库获取日线K线数据"""
    symbol = normalize_ts_code(ts_code)
    if end_date is not None:
        sql = """
            SELECT bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE (ts_code = :ts_code OR ts_code = :ts_code_sh OR ts_code = :ts_code_sz) AND freq = 'd'
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
            WHERE (ts_code = :ts_code OR ts_code = :ts_code_sh OR ts_code = :ts_code_sz) AND freq = 'd'
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
    return df


def ensure_table_exists():
    """确保 atr_rope_factors 表存在"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS atr_rope_factors (
        id BIGSERIAL PRIMARY KEY,
        bar_time TIMESTAMP NOT NULL,
        ts_code VARCHAR(20) NOT NULL,

        -- 基础指标
        atr_rope_tr FLOAT, atr_rope_atr_raw FLOAT, atr_rope_atr FLOAT,
        atr_rope_rope FLOAT, atr_rope_upper FLOAT, atr_rope_lower FLOAT,
        atr_rope_dir SMALLINT, atr_rope_c_hi FLOAT, atr_rope_c_lo FLOAT, atr_rope_ff BOOLEAN,

        -- 方向事件
        evt_atr_rope_dir_change BOOLEAN, evt_atr_rope_dir_red_to_blue BOOLEAN,
        evt_atr_rope_dir_blue_to_green BOOLEAN, evt_atr_rope_dir_green_to_blue BOOLEAN,
        evt_atr_rope_dir_blue_to_red BOOLEAN, evt_atr_rope_dir_red_to_green BOOLEAN,
        evt_atr_rope_dir_green_to_red BOOLEAN, evt_atr_rope_turn_up BOOLEAN,
        evt_atr_rope_turn_down BOOLEAN, evt_atr_rope_turn_flat BOOLEAN,

        -- Rope 趋势线事件
        evt_atr_rope_cross_rope BOOLEAN, evt_atr_rope_line_touch_rope BOOLEAN,
        evt_atr_rope_line_cross_up BOOLEAN, evt_atr_rope_line_cross_down BOOLEAN,
        evt_atr_rope_line_retest_green BOOLEAN, evt_atr_rope_line_retest_red BOOLEAN,

        -- 蓝色震荡区事件
        evt_atr_rope_range_start BOOLEAN, evt_atr_rope_range_touch_high BOOLEAN,
        evt_atr_rope_range_touch_low BOOLEAN, evt_atr_rope_range_break_up BOOLEAN,
        evt_atr_rope_range_break_down BOOLEAN, evt_atr_rope_range_reenter_from_above BOOLEAN,
        evt_atr_rope_range_reenter_from_below BOOLEAN,

        -- Regime 事件
        evt_atr_rope_regime_change BOOLEAN, evt_atr_rope_regime_to_bull BOOLEAN,
        evt_atr_rope_regime_to_bear BOOLEAN, evt_atr_rope_regime_to_range BOOLEAN,

        -- 状态因子
        factor_atr_rope_state_dir SMALLINT, factor_atr_rope_state_dir_prev SMALLINT,
        factor_atr_rope_state_dir_bars INT,

        -- Regime 因子
        factor_atr_rope_regime SMALLINT, factor_atr_rope_regime_prev SMALLINT,
        factor_atr_rope_regime_bars INT,
        factor_atr_rope_green_ratio_20 FLOAT, factor_atr_rope_red_ratio_20 FLOAT,
        factor_atr_rope_blue_ratio_20 FLOAT, factor_atr_rope_slope_pct_20 FLOAT,
        factor_atr_rope_regime_strength FLOAT,

        -- 趋势线/蓝区因子
        factor_atr_rope_line_dev_pct FLOAT, factor_atr_rope_line_dev_atr FLOAT,
        factor_atr_rope_range_high FLOAT, factor_atr_rope_range_low FLOAT,
        factor_atr_rope_range_mid FLOAT, factor_atr_rope_range_pos_01 FLOAT,
        factor_atr_rope_range_width_pct FLOAT, factor_atr_rope_range_width_atr FLOAT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(bar_time, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_arf_bar_time ON atr_rope_factors(bar_time);
    CREATE INDEX IF NOT EXISTS idx_arf_ts_code ON atr_rope_factors(ts_code);
    CREATE INDEX IF NOT EXISTS idx_arf_regime ON atr_rope_factors(factor_atr_rope_regime);
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()


def compute_and_save_stock(ts_code: str, end_date: date = None, save: bool = True) -> int:
    """
    计算单只股票的 ATR Rope 因子并保存到数据库

    Args:
        ts_code: 股票代码
        end_date: 数据截止日期
        save: 是否保存到数据库

    Returns:
        保存的行数
    """
    symbol = normalize_ts_code(ts_code)

    # 获取日线数据
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=end_date)
    if daily_df.empty or len(daily_df) < 60:
        return 0

    # 调用 SSOT 核心计算
    cfg = ATRRopeConfig()
    try:
        result_df = compute_atr_rope(daily_df, cfg)
    except Exception:
        return 0

    if result_df.empty:
        return 0

    # 只保留因子列（排除原始K线字段）
    available_cols = [c for c in FACTOR_COLUMNS if c in result_df.columns]
    factor_df = result_df[available_cols].copy()

    # 添加 bar_time 和 ts_code
    factor_df = factor_df.reset_index()
    factor_df = factor_df.rename(columns={'index': 'bar_time'})
    if 'bar_time' not in factor_df.columns:
        return 0

    factor_df['bar_time'] = pd.to_datetime(factor_df['bar_time'])
    factor_df['ts_code'] = symbol

    if not save:
        return len(factor_df)

    # 幂等写入：先删该股票的旧数据，再插新数据
    with engine.connect() as conn:
        delete_sql = text(f"DELETE FROM {FACTORS_TABLE} WHERE ts_code = :ts_code")
        conn.execute(delete_sql, {'ts_code': symbol})
        conn.commit()

    # 写入新数据
    factor_df.to_sql(FACTORS_TABLE, engine, if_exists='append', index=False)
    return len(factor_df)


def get_stock_list(start_date: str = None) -> pd.DataFrame:
    """获取股票列表"""
    if start_date:
        sql = text("""
            SELECT DISTINCT ts_code
            FROM stock_k_data
            WHERE freq = 'd' AND DATE(bar_time) >= :start_date
        """)
        df = pd.read_sql(sql, engine, params={'start_date': start_date})
    else:
        sql = text("""
            SELECT DISTINCT ts_code FROM stock_k_data WHERE freq = 'd'
        """)
        df = pd.read_sql(sql, engine)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='ATR Rope 因子数据构建',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python atr_experiment/00_build_factor_table.py                    # 全量构建
  python atr_experiment/00_build_factor_table.py --start-date 2023-01-01  # 2023年以来的股票
  python atr_experiment/00_build_factor_table.py --test 600547      # 测试单只
  python atr_experiment/00_build_factor_table.py --no-save          # 不写入数据库
        """
    )
    parser.add_argument('--start-date', help='只处理该日期后有数据的股票 (YYYY-MM-DD)')
    parser.add_argument('--test', help='测试单只股票，例如: --test 600547')
    parser.add_argument('--no-save', action='store_true', help='不保存到数据库（仅显示结果）')

    args = parser.parse_args()

    ensure_table_exists()

    # 测试模式
    if args.test:
        print(f"测试单只股票: {args.test}")
        count = compute_and_save_stock(args.test, save=not args.no_save)
        print(f"  生成 {count} 行因子数据")
        return

    # 全量模式
    print("=" * 80)
    print("ATR Rope 因子数据构建")
    print("=" * 80)

    stock_df = get_stock_list(args.start_date)
    stock_list = stock_df['ts_code'].tolist()
    print(f"共 {len(stock_list)} 只股票")

    total_saved = 0
    total_failed = 0

    for ts_code in tqdm(stock_list, desc="构建因子数据", unit="只"):
        try:
            count = compute_and_save_stock(ts_code, save=not args.no_save)
            total_saved += count
        except Exception as e:
            total_failed += 1
            if total_failed <= 5:
                print(f"  错误: {ts_code} - {e}")

    print("\n" + "=" * 80)
    print("因子数据构建完成")
    print(f"  成功: {len(stock_list) - total_failed} 只")
    print(f"  失败: {total_failed} 只")
    print(f"  总行数: {total_saved}")
    print(f"  表名: {FACTORS_TABLE}")
    print("=" * 80)


if __name__ == '__main__':
    main()
