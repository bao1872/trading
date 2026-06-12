#!/usr/bin/env python3
"""
Tick 因子选股脚本（PVDI因子 + DSA VWAP偏离率 + tick汇总缓存）

Purpose:
    为每只股票每日计算PVDI三因子并进行标签分类，
    记录每个交易bar的最高点和最低点相对于DSA VWAP趋势线的偏离率，
    实现30天tick数据DB缓存。

Inputs:
    stock_k_data (日线K线数据)
    pytdx (实时tick数据接口)

Outputs:
    tick_cache (tick汇总+PVDI因子缓存表)
    tick_selection (选股结果表)

How to Run:
    python selection/selection_tick.py              # 当天
    python selection/selection_tick.py 2026-05-19   # 指定日期
    python selection/selection_tick.py --test 002916 # 测试单只
    python selection/selection_tick.py --backfill 2025-07-01 2026-04-30  # 回补历史

Side Effects:
    写入 tick_cache 和 tick_selection 表（幂等：同一日期先删后插）
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List
from tqdm import tqdm

from datasource.pytdx_client import (
    connect_pytdx, get_tick_data_for_dates, compute_pvdi_for_dates,
    classify_pvdi_pattern,
)
from features.dynamic_swing_anchored_vwap import DSAConfig
from selection.selection_dsa import (
    normalize_ts_code, get_kline_data_db, batch_get_stock_names,
    compute_dsa_regime, remove_vwap_lookahead,
)

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

TICK_CACHE_TABLE = "tick_cache"
TICK_SELECTION_TABLE = "tick_selection"
DAILY_BARS = 800
CACHE_DAYS = 30

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ========== DB建表 ==========

def ensure_cache_table_exists():
    """创建tick缓存表（若不存在）"""
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {TICK_CACHE_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        trade_date DATE NOT NULL,
        ts_code VARCHAR(20) NOT NULL,
        buy_volume BIGINT,
        sell_volume BIGINT,
        buy_trades INT,
        sell_trades INT,
        f_center FLOAT,
        f_spread FLOAT,
        f_skew FLOAT,
        skew_b FLOAT,
        skew_s FLOAT,
        pvdi_weighted FLOAT,
        pattern INT,
        label VARCHAR(20),
        signal VARCHAR(20),
        strength VARCHAR(10),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(trade_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_tc_trade_date ON {TICK_CACHE_TABLE}(trade_date);
    CREATE INDEX IF NOT EXISTS idx_tc_ts_code ON {TICK_CACHE_TABLE}(ts_code);
    CREATE INDEX IF NOT EXISTS idx_tc_ts_date ON {TICK_CACHE_TABLE}(ts_code, trade_date);
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def ensure_selection_table_exists():
    """创建tick选股结果表（若不存在）"""
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {TICK_SELECTION_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),
        f_center FLOAT,
        f_spread FLOAT,
        f_skew FLOAT,
        skew_b FLOAT,
        skew_s FLOAT,
        pvdi_weighted FLOAT,
        pattern INT,
        label VARCHAR(20),
        signal VARCHAR(20),
        strength VARCHAR(10),
        buy_volume BIGINT,
        sell_volume BIGINT,
        buy_trades INT,
        sell_trades INT,
        buy_sell_volume_ratio FLOAT,
        buy_sell_trades_ratio FLOAT,
        dsa_vwap FLOAT,
        high_dev_pct FLOAT,
        low_dev_pct FLOAT,
        close_dev_pct FLOAT,
        dsa_dir INT,
        dsa_bars INT,
        regime VARCHAR(10),
        close_price FLOAT,
        change_pct FLOAT,
        avg_amount_20d FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_ts_sel_date ON {TICK_SELECTION_TABLE}(selection_date);
    CREATE INDEX IF NOT EXISTS idx_ts_ts_code ON {TICK_SELECTION_TABLE}(ts_code);
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# ========== 缓存逻辑 ==========

def get_cached_tick_data(ts_code: str, n_days: int = CACHE_DAYS) -> pd.DataFrame:
    """从DB读取最近N天的tick缓存数据

    Args:
        ts_code: 股票代码（如 000001.SZ）
        n_days: 回查天数

    Returns:
        DataFrame，列与 tick_cache 表一致
    """
    sql = text(f"""
        SELECT trade_date, ts_code, buy_volume, sell_volume, buy_trades, sell_trades,
               f_center, f_spread, f_skew, skew_b, skew_s, pvdi_weighted,
               pattern, label, signal, strength
        FROM {TICK_CACHE_TABLE}
        WHERE ts_code = :ts_code
        AND trade_date >= CURRENT_DATE - INTERVAL :n_days
        ORDER BY trade_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={'ts_code': ts_code, 'n_days': f'{n_days + 5} days'})
    return df


def cache_tick_data(ts_code: str, tick_df: pd.DataFrame, pvdi_df: pd.DataFrame) -> int:
    """将tick汇总和PVDI因子写入缓存表（幂等：先删后插）

    Args:
        ts_code: 股票代码
        tick_df: get_tick_data_for_dates 返回的DataFrame
        pvdi_df: compute_pvdi_for_dates 返回的DataFrame

    Returns:
        写入行数
    """
    if tick_df.empty and pvdi_df.empty:
        return 0

    ensure_cache_table_exists()

    # 合并tick和PVDI数据
    if not tick_df.empty and not pvdi_df.empty:
        merged = pd.merge(tick_df, pvdi_df, on='date_int', how='outer')
    elif not tick_df.empty:
        merged = tick_df.copy()
        for col in ['f_center', 'f_spread', 'f_skew', 'skew_b', 'skew_s', 'pvdi_weighted', 'pattern', 'label', 'signal', 'strength']:
            merged[col] = None
    else:
        merged = pvdi_df.copy()
        for col in ['buy_volume', 'sell_volume', 'buy_trades', 'sell_trades']:
            merged[col] = None

    if merged.empty:
        return 0

    # date_int -> trade_date
    merged['trade_date'] = pd.to_datetime(merged['date_int'].astype(str), format='%Y%m%d').dt.date
    merged['ts_code'] = ts_code

    # 删除旧缓存（该股票这些日期的数据）
    date_list = merged['trade_date'].tolist()
    if date_list:
        placeholders = ', '.join([f"'{d}'" for d in date_list])
        with engine.connect() as conn:
            conn.execute(text(
                f"DELETE FROM {TICK_CACHE_TABLE} WHERE ts_code = :ts_code AND trade_date IN ({placeholders})"
            ), {'ts_code': ts_code})
            conn.commit()

    # 写入新缓存
    insert_cols = ['trade_date', 'ts_code', 'buy_volume', 'sell_volume', 'buy_trades', 'sell_trades',
                   'f_center', 'f_spread', 'f_skew', 'skew_b', 'skew_s', 'pvdi_weighted',
                   'pattern', 'label', 'signal', 'strength']
    insert_df = merged[insert_cols].copy()
    insert_df.to_sql(TICK_CACHE_TABLE, engine, if_exists='append', index=False)

    # 清理超过35天的旧缓存
    with engine.connect() as conn:
        conn.execute(text(
            f"DELETE FROM {TICK_CACHE_TABLE} WHERE ts_code = :ts_code AND trade_date < CURRENT_DATE - INTERVAL '35 days'"
        ), {'ts_code': ts_code})
        conn.commit()

    return len(insert_df)


def refresh_tick_cache(api, ts_code: str, date_ints: List[int]) -> pd.DataFrame:
    """检查缓存缺失日期，仅拉取缺失部分，写入缓存后返回全量

    Args:
        api: TdxHq_API 实例
        ts_code: 股票代码（如 000001.SZ）
        date_ints: 需要的日期整数列表

    Returns:
        DataFrame，包含所有请求日期的缓存数据
    """
    if not date_ints:
        return pd.DataFrame()

    ensure_cache_table_exists()

    # 1. 从DB读已有缓存
    cached = get_cached_tick_data(ts_code)
    if not cached.empty:
        cached_date_ints = set(
            int(pd.Timestamp(d).strftime('%Y%m%d')) if isinstance(d, date) else int(d.strftime('%Y%m%d'))
            for d in cached['trade_date']
        )
    else:
        cached_date_ints = set()

    # 2. 找出缺失的日期
    missing_date_ints = sorted(set(date_ints) - cached_date_ints)

    if missing_date_ints:
        symbol = normalize_ts_code(ts_code)

        # 3. 拉取缺失日期的tick和PVDI数据
        tick_df = pd.DataFrame()
        pvdi_df = pd.DataFrame()
        try:
            tick_df = get_tick_data_for_dates(api, symbol, missing_date_ints)
        except Exception as e:
            logger.warning(f"获取tick数据失败 {ts_code}: {e}")
        try:
            pvdi_df = compute_pvdi_for_dates(api, symbol, missing_date_ints)
        except Exception as e:
            logger.warning(f"获取PVDI数据失败 {ts_code}: {e}")

        # 4. 写入缓存
        saved = cache_tick_data(ts_code, tick_df, pvdi_df)
        if saved > 0:
            logger.debug(f"缓存写入 {ts_code}: {saved} 条")

    # 5. 返回全量缓存
    result = get_cached_tick_data(ts_code)
    # 过滤只保留请求的日期
    if not result.empty:
        result_date_ints = set(
            int(pd.Timestamp(d).strftime('%Y%m%d')) if isinstance(d, date) else int(d.strftime('%Y%m%d'))
            for d in result['trade_date']
        )
        requested_set = set(date_ints)
        result = result[
            result['trade_date'].apply(
                lambda d: int(pd.Timestamp(d).strftime('%Y%m%d')) in requested_set
            )
        ]
    return result


# ========== 计算逻辑 ==========

def compute_dsa_vwap_deviation(daily_df: pd.DataFrame, trade_dt) -> Optional[Dict]:
    """计算单日K线高低点相对DSA VWAP的偏离率

    Args:
        daily_df: 日线DataFrame（含high/low/close列，bar_time为索引）
        trade_dt: 交易日期

    Returns:
        {'dsa_vwap': float, 'high_dev_pct': float, 'low_dev_pct': float,
         'close_dev_pct': float, 'dsa_dir': int, 'dsa_bars': int, 'regime': str}
        或 None（计算失败时）
    """
    try:
        regime_series, _, dsa_bars_series, vwap_series, dir_series = compute_dsa_regime(daily_df)
    except Exception as e:
        logger.debug(f"DSA计算异常 {trade_dt}: {e}")
        return None

    if trade_dt not in vwap_series.index:
        return None

    vwap_val = float(vwap_series.loc[trade_dt])
    if not np.isfinite(vwap_val) or vwap_val == 0:
        return None

    high_val = float(daily_df['high'].loc[trade_dt])
    low_val = float(daily_df['low'].loc[trade_dt])
    close_val = float(daily_df['close'].loc[trade_dt])

    dsa_dir_val = int(dir_series.loc[trade_dt]) if trade_dt in dir_series.index else 0
    dsa_bars_val = int(dsa_bars_series.loc[trade_dt]) if trade_dt in dsa_bars_series.index else 0
    regime_val = int(regime_series.loc[trade_dt]) if trade_dt in regime_series.index else 0
    regime_str = {1: '多头', -1: '空头'}.get(regime_val, '震荡')

    return {
        'dsa_vwap': round(vwap_val, 4),
        'high_dev_pct': round((high_val - vwap_val) / vwap_val * 100, 4),
        'low_dev_pct': round((low_val - vwap_val) / vwap_val * 100, 4),
        'close_dev_pct': round((close_val - vwap_val) / vwap_val * 100, 4),
        'dsa_dir': dsa_dir_val,
        'dsa_bars': dsa_bars_val,
        'regime': regime_str,
    }


def process_stock(ts_code: str, selection_date: date) -> Optional[Dict]:
    """处理单只股票的tick因子选股逻辑（供 --test 调用）

    计算内容：
        - PVDI三因子 + 分类标签
        - tick汇总（买卖量/笔数）
        - DSA VWAP偏离率（high/low/close）
        - 买卖量比、买卖笔数比

    Returns: 结果字典，失败返回None
    """
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
    if daily_df.empty or len(daily_df) < 60:
        return None

    # DSA VWAP偏离率
    vwap_dev = compute_dsa_vwap_deviation(daily_df, daily_df.index[-1])
    if vwap_dev is None:
        return None

    # 过滤：DSA dir=1 持续 > 50 bars
    if vwap_dev['dsa_dir'] != 1 or vwap_dev['dsa_bars'] <= 50:
        return None

    # 获取最近30个交易日的date_ints
    recent_bars = daily_df.tail(CACHE_DAYS)
    date_ints = tuple(int(pd.Timestamp(d).strftime('%Y%m%d')) for d in recent_bars.index)

    # 获取tick缓存数据
    symbol = normalize_ts_code(ts_code)
    api = connect_pytdx()
    try:
        cache_df = refresh_tick_cache(api, ts_code, list(date_ints))
    finally:
        api.disconnect()

    # 取当日数据
    selection_date_int = int(selection_date.strftime('%Y%m%d'))
    day_cache = cache_df[
        cache_df['trade_date'].apply(
            lambda d: int(pd.Timestamp(d).strftime('%Y%m%d')) == selection_date_int
        )
    ]

    if day_cache.empty:
        logger.debug(f"无tick缓存 {ts_code} {selection_date}")
        return None

    row = day_cache.iloc[0]

    # 买卖量比
    buy_vol = row.get('buy_volume')
    sell_vol = row.get('sell_volume')
    buy_trades = row.get('buy_trades')
    sell_trades = row.get('sell_trades')

    vol_ratio = float(buy_vol / sell_vol) if buy_vol and sell_vol and sell_vol > 0 else None
    trades_ratio = float(buy_trades / sell_trades) if buy_trades and sell_trades and sell_trades > 0 else None

    # 价格信息
    last_close = float(daily_df['close'].iloc[-1])
    change_pct = None
    if len(daily_df) >= 2:
        prev_close = float(daily_df['close'].iloc[-2])
        if prev_close > 0:
            change_pct = round((last_close / prev_close - 1) * 100, 4)

    avg_amount_20d = None
    if len(daily_df) >= 20:
        avg_amount_20d = round(
            float(((daily_df['open'] + daily_df['close']) / 2 * daily_df['volume']).tail(20).mean()) / 1e8, 4
        )

    def safe_float(val, default=None):
        return float(val) if pd.notna(val) else default

    return {
        'ts_code': ts_code,
        'f_center': safe_float(row.get('f_center')),
        'f_spread': safe_float(row.get('f_spread')),
        'f_skew': safe_float(row.get('f_skew')),
        'skew_b': safe_float(row.get('skew_b')),
        'skew_s': safe_float(row.get('skew_s')),
        'pvdi_weighted': safe_float(row.get('pvdi_weighted')),
        'pattern': int(row['pattern']) if pd.notna(row.get('pattern')) else None,
        'label': row.get('label'),
        'signal': row.get('signal'),
        'strength': row.get('strength'),
        'buy_volume': int(buy_vol) if pd.notna(buy_vol) else None,
        'sell_volume': int(sell_vol) if pd.notna(sell_vol) else None,
        'buy_trades': int(buy_trades) if pd.notna(buy_trades) else None,
        'sell_trades': int(sell_trades) if pd.notna(sell_trades) else None,
        'buy_sell_volume_ratio': vol_ratio,
        'buy_sell_trades_ratio': trades_ratio,
        'dsa_vwap': vwap_dev['dsa_vwap'],
        'high_dev_pct': vwap_dev['high_dev_pct'],
        'low_dev_pct': vwap_dev['low_dev_pct'],
        'close_dev_pct': vwap_dev['close_dev_pct'],
        'dsa_dir': vwap_dev['dsa_dir'],
        'dsa_bars': vwap_dev['dsa_bars'],
        'regime': vwap_dev['regime'],
        'close_price': last_close,
        'change_pct': change_pct,
        'avg_amount_20d': avg_amount_20d,
        'signal_date': daily_df.index[-1],
    }


# ========== 当日选股 ==========

def select_tick_stocks(selection_date: Optional[date] = None) -> pd.DataFrame:
    """当日tick因子选股主函数"""
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("Tick 因子选股")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  核心条件: DSA VWAP dir=1 持续 > 50 bars")
    print(f"  记录满足条件股票的PVDI因子和VWAP偏离率")
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

    if len(stock_list) == 0:
        print("\n未找到符合条件的股票")
        return pd.DataFrame()

    print("\n" + "=" * 80)
    print("开始 Tick 因子计算...")

    results = []
    api = connect_pytdx()
    try:
        for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="Tick因子选股", unit="只"):
            ts_code = row['ts_code']

            daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
            if daily_df.empty or len(daily_df) < 60:
                continue

            # DSA VWAP偏离率
            vwap_dev = compute_dsa_vwap_deviation(daily_df, daily_df.index[-1])
            if vwap_dev is None:
                continue

            # 过滤：DSA dir=1 持续 > 50 bars
            if vwap_dev['dsa_dir'] != 1 or vwap_dev['dsa_bars'] <= 50:
                continue

            # 获取最近30个交易日的date_ints
            recent_bars = daily_df.tail(CACHE_DAYS)
            date_ints = [int(pd.Timestamp(d).strftime('%Y%m%d')) for d in recent_bars.index]

            # 刷新tick缓存
            try:
                cache_df = refresh_tick_cache(api, ts_code, date_ints)
            except Exception as e:
                logger.warning(f"缓存刷新失败 {ts_code}: {e}")
                continue

            # 取当日数据
            selection_date_int = int(selection_date.strftime('%Y%m%d'))
            day_cache = cache_df[
                cache_df['trade_date'].apply(
                    lambda d: int(pd.Timestamp(d).strftime('%Y%m%d')) == selection_date_int
                )
            ]

            if day_cache.empty:
                continue

            cr = day_cache.iloc[0]

            # 买卖量比
            buy_vol = cr.get('buy_volume')
            sell_vol = cr.get('sell_volume')
            buy_trades = cr.get('buy_trades')
            sell_trades = cr.get('sell_trades')
            vol_ratio = float(buy_vol / sell_vol) if buy_vol and sell_vol and sell_vol > 0 else None
            trades_ratio = float(buy_trades / sell_trades) if buy_trades and sell_trades and sell_trades > 0 else None

            # 价格信息
            last_close = float(daily_df['close'].iloc[-1])
            change_pct = None
            if len(daily_df) >= 2:
                prev_close = float(daily_df['close'].iloc[-2])
                if prev_close > 0:
                    change_pct = round((last_close / prev_close - 1) * 100, 4)
            avg_amount_20d = None
            if len(daily_df) >= 20:
                avg_amount_20d = round(
                    float(((daily_df['open'] + daily_df['close']) / 2 * daily_df['volume']).tail(20).mean()) / 1e8, 4
                )

            def safe_float(val, default=None):
                return float(val) if pd.notna(val) else default

            results.append({
                'ts_code': ts_code,
                'f_center': safe_float(cr.get('f_center')),
                'f_spread': safe_float(cr.get('f_spread')),
                'f_skew': safe_float(cr.get('f_skew')),
                'skew_b': safe_float(cr.get('skew_b')),
                'skew_s': safe_float(cr.get('skew_s')),
                'pvdi_weighted': safe_float(cr.get('pvdi_weighted')),
                'pattern': int(cr['pattern']) if pd.notna(cr.get('pattern')) else None,
                'label': cr.get('label'),
                'signal': cr.get('signal'),
                'strength': cr.get('strength'),
                'buy_volume': int(buy_vol) if pd.notna(buy_vol) else None,
                'sell_volume': int(sell_vol) if pd.notna(sell_vol) else None,
                'buy_trades': int(buy_trades) if pd.notna(buy_trades) else None,
                'sell_trades': int(sell_trades) if pd.notna(sell_trades) else None,
                'buy_sell_volume_ratio': vol_ratio,
                'buy_sell_trades_ratio': trades_ratio,
                'dsa_vwap': vwap_dev['dsa_vwap'],
                'high_dev_pct': vwap_dev['high_dev_pct'],
                'low_dev_pct': vwap_dev['low_dev_pct'],
                'close_dev_pct': vwap_dev['close_dev_pct'],
                'dsa_dir': vwap_dev['dsa_dir'],
                'dsa_bars': vwap_dev['dsa_bars'],
                'regime': vwap_dev['regime'],
                'close_price': last_close,
                'change_pct': change_pct,
                'avg_amount_20d': avg_amount_20d,
                'signal_date': daily_df.index[-1],
            })
    finally:
        api.disconnect()

    if not results:
        print("\n无符合条件的结果")
        return pd.DataFrame()

    # 批量获取股票名称
    ts_codes = [r['ts_code'] for r in results]
    stock_name_map = batch_get_stock_names(ts_codes)
    for r in results:
        r['stock_name'] = stock_name_map.get(r['ts_code'], '')

    result_df = pd.DataFrame(results)
    saved = save_to_database(result_df, selection_date)
    print(f"\n选股完成: {len(results)} 只股票, 保存 {saved} 条记录")
    return result_df


def save_to_database(df: pd.DataFrame, selection_date: date) -> int:
    """保存选股结果到数据库（幂等：先删后插）"""
    if df.empty:
        return 0

    ensure_selection_table_exists()

    with engine.connect() as conn:
        delete_sql = text(f"DELETE FROM {TICK_SELECTION_TABLE} WHERE selection_date = :selection_date")
        result = conn.execute(delete_sql, {'selection_date': selection_date})
        conn.commit()
        if result.rowcount > 0:
            print(f"  清除旧数据: {result.rowcount} 条")

    insert_records = []
    for _, row in df.iterrows():
        insert_records.append({
            'selection_date': selection_date,
            'ts_code': row['ts_code'],
            'stock_name': row.get('stock_name', ''),
            'f_center': float(row['f_center']) if pd.notna(row.get('f_center')) else None,
            'f_spread': float(row['f_spread']) if pd.notna(row.get('f_spread')) else None,
            'f_skew': float(row['f_skew']) if pd.notna(row.get('f_skew')) else None,
            'skew_b': float(row['skew_b']) if pd.notna(row.get('skew_b')) else None,
            'skew_s': float(row['skew_s']) if pd.notna(row.get('skew_s')) else None,
            'pvdi_weighted': float(row['pvdi_weighted']) if pd.notna(row.get('pvdi_weighted')) else None,
            'pattern': int(row['pattern']) if pd.notna(row.get('pattern')) else None,
            'label': row.get('label'),
            'signal': row.get('signal'),
            'strength': row.get('strength'),
            'buy_volume': int(row['buy_volume']) if pd.notna(row.get('buy_volume')) else None,
            'sell_volume': int(row['sell_volume']) if pd.notna(row.get('sell_volume')) else None,
            'buy_trades': int(row['buy_trades']) if pd.notna(row.get('buy_trades')) else None,
            'sell_trades': int(row['sell_trades']) if pd.notna(row.get('sell_trades')) else None,
            'buy_sell_volume_ratio': float(row['buy_sell_volume_ratio']) if pd.notna(row.get('buy_sell_volume_ratio')) else None,
            'buy_sell_trades_ratio': float(row['buy_sell_trades_ratio']) if pd.notna(row.get('buy_sell_trades_ratio')) else None,
            'dsa_vwap': float(row['dsa_vwap']) if pd.notna(row.get('dsa_vwap')) else None,
            'high_dev_pct': float(row['high_dev_pct']) if pd.notna(row.get('high_dev_pct')) else None,
            'low_dev_pct': float(row['low_dev_pct']) if pd.notna(row.get('low_dev_pct')) else None,
            'close_dev_pct': float(row['close_dev_pct']) if pd.notna(row.get('close_dev_pct')) else None,
            'dsa_dir': int(row['dsa_dir']) if pd.notna(row.get('dsa_dir')) else None,
            'dsa_bars': int(row['dsa_bars']) if pd.notna(row.get('dsa_bars')) else None,
            'regime': row.get('regime'),
            'close_price': float(row['close_price']) if pd.notna(row.get('close_price')) else None,
            'change_pct': float(row['change_pct']) if pd.notna(row.get('change_pct')) else None,
            'avg_amount_20d': float(row['avg_amount_20d']) if pd.notna(row.get('avg_amount_20d')) else None,
        })

    if insert_records:
        insert_df = pd.DataFrame(insert_records)
        insert_df.to_sql(TICK_SELECTION_TABLE, engine, if_exists='append', index=False)
    return len(insert_records)


# ========== 回补模式 ==========

def backfill_stock_events(ts_code: str, start_date: date, end_date: date) -> List[Dict]:
    """遍历单只股票在日期区间内的所有交易日，计算tick因子和VWAP偏离率

    一次加载全量数据，然后遍历结果行判断信号。
    """
    daily_df = get_kline_data_db(ts_code, bars=1200, end_date=end_date)
    if daily_df.empty or len(daily_df) < 60:
        return []

    if not isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df.index = pd.to_datetime(daily_df.index)

    # 一次性计算DSA VWAP全量指标
    try:
        regime_series, _, dsa_bars_series, vwap_series, dir_series = compute_dsa_regime(daily_df)
    except Exception:
        return []

    # 获取日期区间内的交易日
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    mask = (daily_df.index >= start_ts) & (daily_df.index <= end_ts)
    trade_dates = daily_df.loc[mask].index.tolist()
    if not trade_dates:
        return []

    # 刷新tick缓存：取最近30个交易日的date_ints
    recent_bars = daily_df.tail(CACHE_DAYS)
    all_date_ints = [int(pd.Timestamp(d).strftime('%Y%m%d')) for d in recent_bars.index]

    api = connect_pytdx()
    try:
        cache_df = refresh_tick_cache(api, ts_code, all_date_ints)
    except Exception as e:
        logger.warning(f"缓存刷新失败 {ts_code}: {e}")
        api.disconnect()
        return []
    finally:
        api.disconnect()

    # 构建缓存查找表：date_int -> row
    cache_lookup = {}
    if not cache_df.empty:
        for _, cr in cache_df.iterrows():
            td = cr['trade_date']
            di = int(pd.Timestamp(td).strftime('%Y%m%d')) if not isinstance(td, int) else td
            cache_lookup[di] = cr

    results = []
    for trade_dt in trade_dates:
        loc = daily_df.index.get_loc(trade_dt)
        if loc < 1:
            continue

        # DSA VWAP偏离率
        vwap_val = vwap_series.loc[trade_dt] if trade_dt in vwap_series.index else None
        if vwap_val is None or not np.isfinite(vwap_val) or vwap_val == 0:
            continue

        high_val = float(daily_df['high'].loc[trade_dt])
        low_val = float(daily_df['low'].loc[trade_dt])
        close_val = float(daily_df['close'].loc[trade_dt])
        vwap_f = float(vwap_val)

        dsa_dir_val = int(dir_series.loc[trade_dt]) if trade_dt in dir_series.index else 0
        dsa_bars_val = int(dsa_bars_series.loc[trade_dt]) if trade_dt in dsa_bars_series.index else 0
        regime_val = int(regime_series.loc[trade_dt]) if trade_dt in regime_series.index else 0
        regime_str = {1: '多头', -1: '空头'}.get(regime_val, '震荡')

        # 过滤：DSA dir=1 持续 > 50 bars
        if dsa_dir_val != 1 or dsa_bars_val <= 50:
            continue

        # 从缓存取tick数据
        trade_date_int = int(pd.Timestamp(trade_dt).strftime('%Y%m%d'))
        cr = cache_lookup.get(trade_date_int)

        # 买卖量比
        buy_vol = cr.get('buy_volume') if cr is not None else None
        sell_vol = cr.get('sell_volume') if cr is not None else None
        buy_trades = cr.get('buy_trades') if cr is not None else None
        sell_trades = cr.get('sell_trades') if cr is not None else None
        vol_ratio = float(buy_vol / sell_vol) if buy_vol and sell_vol and sell_vol > 0 else None
        trades_ratio = float(buy_trades / sell_trades) if buy_trades and sell_trades and sell_trades > 0 else None

        # 价格信息
        change_pct = None
        if loc >= 1:
            prev_close = float(daily_df['close'].iloc[loc - 1])
            if prev_close > 0:
                change_pct = round((close_val / prev_close - 1) * 100, 4)

        sub_df = daily_df.loc[daily_df.index <= trade_dt]
        avg_amount_20d = None
        if len(sub_df) >= 20:
            avg_amount_20d = round(
                float(((sub_df['open'] + sub_df['close']) / 2 * sub_df['volume']).tail(20).mean()) / 1e8, 4
            )

        def safe_float(val, default=None):
            return float(val) if pd.notna(val) else default

        results.append({
            'ts_code': ts_code,
            'f_center': safe_float(cr.get('f_center')) if cr is not None else None,
            'f_spread': safe_float(cr.get('f_spread')) if cr is not None else None,
            'f_skew': safe_float(cr.get('f_skew')) if cr is not None else None,
            'skew_b': safe_float(cr.get('skew_b')) if cr is not None else None,
            'skew_s': safe_float(cr.get('skew_s')) if cr is not None else None,
            'pvdi_weighted': safe_float(cr.get('pvdi_weighted')) if cr is not None else None,
            'pattern': int(cr['pattern']) if cr is not None and pd.notna(cr.get('pattern')) else None,
            'label': cr.get('label') if cr is not None else None,
            'signal': cr.get('signal') if cr is not None else None,
            'strength': cr.get('strength') if cr is not None else None,
            'buy_volume': int(buy_vol) if pd.notna(buy_vol) else None,
            'sell_volume': int(sell_vol) if pd.notna(sell_vol) else None,
            'buy_trades': int(buy_trades) if pd.notna(buy_trades) else None,
            'sell_trades': int(sell_trades) if pd.notna(sell_trades) else None,
            'buy_sell_volume_ratio': vol_ratio,
            'buy_sell_trades_ratio': trades_ratio,
            'dsa_vwap': round(vwap_f, 4),
            'high_dev_pct': round((high_val - vwap_f) / vwap_f * 100, 4),
            'low_dev_pct': round((low_val - vwap_f) / vwap_f * 100, 4),
            'close_dev_pct': round((close_val - vwap_f) / vwap_f * 100, 4),
            'dsa_dir': dsa_dir_val,
            'dsa_bars': dsa_bars_val,
            'regime': regime_str,
            'close_price': close_val,
            'change_pct': change_pct,
            'avg_amount_20d': avg_amount_20d,
            'signal_date': trade_dt,
            'selection_date': trade_dt.date() if hasattr(trade_dt, 'date') else trade_dt,
        })

    return results


def _save_single_stock_records(records: List[Dict], stock_name_map: Dict[str, str]) -> int:
    """回补模式专用保存：按日期分组，按股票+日期删后插"""
    if not records:
        return 0

    ensure_selection_table_exists()

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
                f"DELETE FROM {TICK_SELECTION_TABLE} "
                f"WHERE selection_date = :selection_date AND ts_code IN ({placeholders})"
            )
            conn.execute(delete_sql, {'selection_date': dt})
            conn.commit()

        insert_records = []
        for rec in day_records:
            insert_records.append({
                'selection_date': dt,
                'ts_code': rec['ts_code'],
                'stock_name': stock_name_map.get(rec['ts_code'], ''),
                'f_center': float(rec['f_center']) if rec.get('f_center') is not None else None,
                'f_spread': float(rec['f_spread']) if rec.get('f_spread') is not None else None,
                'f_skew': float(rec['f_skew']) if rec.get('f_skew') is not None else None,
                'skew_b': float(rec['skew_b']) if rec.get('skew_b') is not None else None,
                'skew_s': float(rec['skew_s']) if rec.get('skew_s') is not None else None,
                'pvdi_weighted': float(rec['pvdi_weighted']) if rec.get('pvdi_weighted') is not None else None,
                'pattern': int(rec['pattern']) if rec.get('pattern') is not None else None,
                'label': rec.get('label'),
                'signal': rec.get('signal'),
                'strength': rec.get('strength'),
                'buy_volume': int(rec['buy_volume']) if rec.get('buy_volume') is not None else None,
                'sell_volume': int(rec['sell_volume']) if rec.get('sell_volume') is not None else None,
                'buy_trades': int(rec['buy_trades']) if rec.get('buy_trades') is not None else None,
                'sell_trades': int(rec['sell_trades']) if rec.get('sell_trades') is not None else None,
                'buy_sell_volume_ratio': float(rec['buy_sell_volume_ratio']) if rec.get('buy_sell_volume_ratio') is not None else None,
                'buy_sell_trades_ratio': float(rec['buy_sell_trades_ratio']) if rec.get('buy_sell_trades_ratio') is not None else None,
                'dsa_vwap': float(rec['dsa_vwap']) if rec.get('dsa_vwap') is not None else None,
                'high_dev_pct': float(rec['high_dev_pct']) if rec.get('high_dev_pct') is not None else None,
                'low_dev_pct': float(rec['low_dev_pct']) if rec.get('low_dev_pct') is not None else None,
                'close_dev_pct': float(rec['close_dev_pct']) if rec.get('close_dev_pct') is not None else None,
                'dsa_dir': int(rec['dsa_dir']) if rec.get('dsa_dir') is not None else None,
                'dsa_bars': int(rec['dsa_bars']) if rec.get('dsa_bars') is not None else None,
                'regime': rec.get('regime'),
                'close_price': float(rec['close_price']) if rec.get('close_price') is not None else None,
                'change_pct': float(rec['change_pct']) if rec.get('change_pct') is not None else None,
                'avg_amount_20d': float(rec['avg_amount_20d']) if rec.get('avg_amount_20d') is not None else None,
            })

        if insert_records:
            insert_df = pd.DataFrame(insert_records)
            insert_df.to_sql(TICK_SELECTION_TABLE, engine, if_exists='append', index=False)
            total_saved += len(insert_records)

    return total_saved


def backfill_range(start_date: date, end_date: date, stock_list: Optional[List[str]] = None):
    """回补入口：外层遍历股票，每只立即保存"""
    print("=" * 80)
    print("Tick 因子回补")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  模式: 遍历个股，逐日计算因子，每只股票立即保存")
    print("=" * 80)

    if stock_list is None:
        with engine.connect() as conn:
            print("\n查询股票列表...")
            sql = text("""
                SELECT DISTINCT ts_code
                FROM stock_k_data
                WHERE freq = 'd'
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
    print(f"共处理 {len(stock_list)} 只股票，其中 {total_stocks_with_events} 只有数据")
    print(f"共保存 {total_saved} 条记录")
    print(f"日期范围: {start_date} ~ {end_date}")


# ========== CLI ==========

def parse_date(date_str: str) -> date:
    for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")


def test_single_stock(ts_code: str, selection_date: date):
    print("\n" + "=" * 80)
    print(f"测试单只股票: {ts_code}")
    print(f"选股日期: {selection_date}")
    print("=" * 80)

    result = process_stock(ts_code, selection_date)

    if result:
        print("\n选股结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("\n该股票无tick数据或计算失败")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Tick 因子选股工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/selection_tick.py                    # 使用当天日期选股
  python selection/selection_tick.py 2026-05-19         # 指定日期选股
  python selection/selection_tick.py 20260519           # 指定日期选股（无分隔符）
  python selection/selection_tick.py --test 002916      # 测试单只股票
  python selection/selection_tick.py --backfill 2025-07-01 2026-04-30  # 回补历史
        """
    )
    parser.add_argument(
        'date',
        nargs='?',
        help='选股日期 (格式: YYYY-MM-DD 或 YYYYMMDD)，默认为当天'
    )
    parser.add_argument(
        '--test',
        help='测试单只股票，例如: --test 002916'
    )
    parser.add_argument(
        '--backfill',
        nargs=2,
        metavar=('START_DATE', 'END_DATE'),
        help='回补历史事件，遍历个股逐日计算，例如: --backfill 2025-07-01 2026-04-30'
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
    print("Tick 因子选股工具")
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print("=" * 80)

    if args.test:
        test_single_stock(args.test, selection_date)
        sys.exit(0)

    select_tick_stocks(selection_date)


if __name__ == '__main__':
    main()
