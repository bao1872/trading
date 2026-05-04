#!/usr/bin/env python3
"""
Stop-Loss Clustering 选股脚本（基于 absorbtion_extremes 模型）

Purpose:
    基于 stop_loss_clustering_with_factors.py 的 absorbtion_extremes 模型，
    识别日线级别突破 sell-stop cluster 或跌破 buy-stop cluster 的股票。

Inputs:
    stock_k_data (日线K线数据，open/high/low/close/volume)

Outputs:
    stop_loss_selection (选股结果表)

How to Run:
    python selection/selection_stop.py              # 当天
    python selection/selection_stop.py 2026-04-10  # 指定日期
    python selection/selection_stop.py --test 600547   # 测试单只
    python selection/selection_stop.py --no-save       # 不写入数据库
    python selection/selection_stop.py --backfill 2025-07-01 2026-04-30  # 回补历史

Side Effects:
    写入 stop_loss_selection 表（幂等：同一日期先删后插）

================================================================================
【选股逻辑】

两种场景均记录（满足任一即入选）：
  1) 突破 sell-stop cluster：当天 high >= 某个活跃 sell cluster 的 barrier
     → sell_stop_triggered = True
  2) 跌破 buy-stop cluster：当天 low <= 某个活跃 buy cluster 的 barrier
     → buy_stop_triggered = True

过滤条件：
  - 过去5天平均成交额 >= 1亿

记录字段（核心）：
  - sell_stop_triggered / buy_stop_triggered：布尔，是否触发
  - sell_trigger_volume / buy_trigger_volume：当天触发释放的成交量
  - active_sell_cluster_count / active_buy_cluster_count：活跃 cluster 数
  - sum_sells_active / sum_buys_active：活跃 cluster 累计成交量（规模）
  - stop_cluster_ratio：sell/buy 活跃量比
  - nearest_sell_stop_price / nearest_buy_stop_price：最近止损价位
  - dist_to_nearest_sell_stop_atr / dist_to_nearest_buy_stop_atr：ATR距离

观察项：
  - change_pct：选股日涨跌幅
  - vol_zscore：成交量 Z-Score
  - bbmacd_event：BBMACD 事件
  - daily_bb_width_zscore：日线布林带宽度 Z-Score

【核心计算】
  - 全部引用 features.stop_loss_clustering_with_factors.StopLossClusteringEngine
  - 本脚本只做数据准备、条件判断、结果保存，不重复计算逻辑（SSOT）

【选股日期】
  - 选股日期只是标记，实际数据到"选股日期当天或之前最后一个交易日"
  - 日线数据通过 python app/build_dataset.py --update --period d 每日更新

【保存逻辑】
  - 按选股日期统一保存，先删旧数据再插新数据（幂等性）
================================================================================
"""

import sys
import os
import argparse
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, Optional, List
from tqdm import tqdm

# 核心计算逻辑（SSOT原则）：只引用，不重复实现
from features.stop_loss_clustering_with_factors import StopLossClusteringEngine
from features.bbmacd_viewer import compute_bbmacd

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "stop_loss_selection"

# K线数据配置
DAILY_BARS = 800  # 日线数据获取800根（约3年）


def normalize_ts_code(ts_code: str) -> str:
    """标准化股票代码"""
    return str(ts_code).strip().upper().split('.')[0]


def get_kline_data_db(ts_code: str, bars: int = DAILY_BARS, end_date: Optional[date] = None) -> pd.DataFrame:
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


def check_volume_filter(df: pd.DataFrame, days: int = 5, min_amount: float = 100_000_000) -> bool:
    """
    检查成交额过滤条件

    Args:
        df: DataFrame with 'volume' and 'close' columns
        days: 计算过去N天的平均成交额，默认5天
        min_amount: 最小成交额阈值（元），默认1亿

    Returns:
        True if 平均成交额 >= min_amount
    """
    if len(df) < days:
        return False

    recent_df = df.tail(days)
    daily_amount = recent_df['volume'] * recent_df['close']
    avg_amount = daily_amount.mean()

    return avg_amount >= min_amount


def volume_zscore(vol: pd.Series, win: int = 20) -> float:
    """计算成交量Z-Score"""
    mu = vol.rolling(win, min_periods=win).mean()
    sd = vol.rolling(win, min_periods=win).std(ddof=0)

    if len(vol) < win:
        return None

    mu_val = mu.iloc[-1]
    sd_val = sd.iloc[-1]

    if sd_val == 0 or pd.isna(sd_val):
        return None
    z = (vol.iloc[-1] - mu_val) / sd_val
    return float(z)


def compute_change_pct(daily_df: pd.DataFrame) -> float:
    """计算选股日当天的涨跌幅"""
    if len(daily_df) < 2:
        return None
    close_today = daily_df['close'].iloc[-1]
    close_yesterday = daily_df['close'].iloc[-2]
    if close_yesterday == 0:
        return None
    change = float(close_today - close_yesterday) / float(close_yesterday) * 100
    return round(change, 2)


def detect_bbmacd_event(bbmacd_df: pd.DataFrame) -> str:
    """检测BBMACD事件类型"""
    if len(bbmacd_df) == 0:
        return "无"

    last_idx = -1
    if bbmacd_df['compra'].iloc[last_idx]:
        return "上穿上轨"
    elif bbmacd_df['cross_down_upper'].iloc[last_idx]:
        return "下穿上轨"
    elif bbmacd_df['cross_up_lower'].iloc[last_idx]:
        return "上穿下轨"
    elif bbmacd_df['venta'].iloc[last_idx]:
        return "下穿下轨"
    return "无"


def process_stock(ts_code: str, selection_date: date) -> Optional[Dict]:
    """
    处理单只股票的 Stop-Loss Clustering 选股逻辑

    选股逻辑：
        - 获取日线数据
        - 调用 StopLossClusteringEngine 计算 absorbtion_extremes
        - 条件：突破 sell-stop cluster（sell_stop_triggered == 1）
        - 过滤：过去5天平均成交额 >= 1亿
        - 新增字段：sell_stop_scale = sum_sells_active * sell_trigger_max_vol_price

    Returns: 信号字典，如果满足条件则返回结果，否则返回None
    """
    # 获取日线数据
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
    if daily_df.empty or len(daily_df) < 60:
        return None

    # 列名对齐：核心计算模块需要 'vol' 列
    if 'volume' in daily_df.columns and 'vol' not in daily_df.columns:
        daily_df = daily_df.rename(columns={'volume': 'vol'})

    # 保留原始 volume 列用于后续过滤计算
    if 'vol' in daily_df.columns and 'volume' not in daily_df.columns:
        daily_df['volume'] = daily_df['vol']

    # 构造兼容的 args 对象（StopLossClusteringEngine 需要 argparse.Namespace）
    args = SimpleNamespace(
        freq='d',
        model='absorbtion_extremes',
        show_historical_triggers=False,
        max_lines=20,
    )

    # 运行核心计算（SSOT：只引用，不重复实现）
    try:
        slc_engine = StopLossClusteringEngine(daily_df, args)
        slc_engine.run()
    except Exception:
        return None

    result_df = slc_engine.df
    if result_df.empty:
        return None

    # 取最后一天的数据做判断
    last = result_df.iloc[-1]

    sell_triggered = bool(last.get('sell_stop_triggered', 0) == 1.0)

    # 只保留突破 sell-stop cluster 的股票
    if not sell_triggered:
        return None

    # 计算 sell_stop_scale = sum_sells_active * sell_trigger_max_vol_price
    sum_sells_active = float(last.get('sum_sells_active', 0)) if pd.notna(last.get('sum_sells_active')) else 0.0
    sell_trigger_max_vol_price = float(last.get('sell_trigger_max_vol_price', 0)) if pd.notna(last.get('sell_trigger_max_vol_price')) else 0.0
    sell_stop_scale = sum_sells_active * sell_trigger_max_vol_price if sum_sells_active > 0 and sell_trigger_max_vol_price > 0 else 0.0

    # 计算上一次事件类型：从倒数第二根向前找最近一次触发（只找sell_stop）
    last_event_type = None
    last_event_volume = 0.0
    last_event_bars_ago = None
    for idx in range(len(result_df) - 2, -1, -1):
        row = result_df.iloc[idx]
        if row.get('sell_stop_triggered', 0) == 1.0:
            last_event_type = 'sell_stop'
            last_event_volume = float(row.get('sell_stop_triggered_volume', 0)) if pd.notna(row.get('sell_stop_triggered_volume')) else 0.0
            last_event_bars_ago = len(result_df) - 1 - idx
            break
    else:
        last_event_type = 'sell_stop'
        last_event_volume = float(last.get('sell_stop_triggered_volume', 0)) if pd.notna(last.get('sell_stop_triggered_volume')) else 0.0
        last_event_bars_ago = 0

    # 成交额过滤
    if not check_volume_filter(daily_df, days=5, min_amount=100_000_000):
        return None

    # 计算观察项
    bbmacd_df = compute_bbmacd(daily_df)
    bbmacd_event = detect_bbmacd_event(bbmacd_df)

    bar_time = daily_df.index[-1]

    return {
        'ts_code': ts_code,
        # 核心 Stop-Loss Clustering 字段（仅sell场景）
        'sell_stop_triggered': sell_triggered,
        'sell_trigger_volume': float(last.get('sell_stop_triggered_volume', 0)) if pd.notna(last.get('sell_stop_triggered_volume')) else 0.0,
        'active_sell_cluster_count': float(last.get('active_sell_cluster_count', 0)) if pd.notna(last.get('active_sell_cluster_count')) else 0.0,
        'sum_sells_active': sum_sells_active,
        'sell_trigger_max_vol_price': sell_trigger_max_vol_price,
        'sell_stop_scale': sell_stop_scale,
        'nearest_sell_stop_price': float(last.get('nearest_sell_stop_price', 0)) if pd.notna(last.get('nearest_sell_stop_price')) else None,
        'dist_to_nearest_sell_stop_atr': float(last.get('dist_to_nearest_sell_stop_atr', 0)) if pd.notna(last.get('dist_to_nearest_sell_stop_atr')) else None,
        # 上一次事件记录
        'last_event_type': last_event_type,
        'last_event_volume': last_event_volume if last_event_type is not None else None,
        'last_event_bars_ago': last_event_bars_ago if last_event_type is not None else None,
        # 观察项
        'change_pct': compute_change_pct(daily_df),
        'vol_zscore': volume_zscore(daily_df['volume'], win=20),
        'bbmacd_event': bbmacd_event,
        'daily_bb_width_zscore': float(bbmacd_df['bb_width_zscore'].iloc[-1]) if 'bb_width_zscore' in bbmacd_df.columns and len(bbmacd_df) > 0 else None,
        'signal_date': bar_time,
    }


def batch_get_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    """批量获取股票名称（一次查询）"""
    if not ts_codes:
        return {}
    placeholders = ', '.join([f"'{c}'" for c in ts_codes])
    sql = text(f"SELECT ts_code, name FROM stock_pools WHERE ts_code IN ({placeholders})")
    with engine.connect() as conn:
        result = conn.execute(sql)
        return {row[0]: row[1] for row in result}


def ensure_table_exists():
    """确保 stop_loss_selection 表存在"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS stop_loss_selection (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        signal_date DATE,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),

        -- 核心 Stop-Loss Clustering 字段（仅突破sell-stop场景）
        sell_stop_triggered BOOLEAN DEFAULT FALSE,
        sell_trigger_volume FLOAT,
        active_sell_cluster_count FLOAT,
        sum_sells_active FLOAT,
        sell_trigger_max_vol_price FLOAT,
        sell_stop_scale FLOAT,
        nearest_sell_stop_price FLOAT,
        dist_to_nearest_sell_stop_atr FLOAT,

        -- 上一次事件记录
        last_event_type VARCHAR(20),
        last_event_volume FLOAT,
        last_event_bars_ago INT,

        -- 观察项
        change_pct FLOAT,
        vol_zscore FLOAT,
        bbmacd_event VARCHAR(20),
        daily_bb_width_zscore FLOAT,

        batch_no INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_sl_selection_date ON stop_loss_selection(selection_date);
    CREATE INDEX IF NOT EXISTS idx_sl_ts_code ON stop_loss_selection(ts_code);
    CREATE INDEX IF NOT EXISTS idx_sl_sell_triggered ON stop_loss_selection(sell_stop_triggered);
    CREATE INDEX IF NOT EXISTS idx_sl_batch_no ON stop_loss_selection(batch_no);
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
            'sell_stop_triggered': bool(row.get('sell_stop_triggered', False)),
            'sell_trigger_volume': float(row['sell_trigger_volume']) if pd.notna(row.get('sell_trigger_volume')) else None,
            'active_sell_cluster_count': float(row['active_sell_cluster_count']) if pd.notna(row.get('active_sell_cluster_count')) else None,
            'sum_sells_active': float(row['sum_sells_active']) if pd.notna(row.get('sum_sells_active')) else None,
            'sell_trigger_max_vol_price': float(row['sell_trigger_max_vol_price']) if pd.notna(row.get('sell_trigger_max_vol_price')) else None,
            'sell_stop_scale': float(row['sell_stop_scale']) if pd.notna(row.get('sell_stop_scale')) else None,
            'nearest_sell_stop_price': float(row['nearest_sell_stop_price']) if pd.notna(row.get('nearest_sell_stop_price')) else None,
            'dist_to_nearest_sell_stop_atr': float(row['dist_to_nearest_sell_stop_atr']) if pd.notna(row.get('dist_to_nearest_sell_stop_atr')) else None,
            'last_event_type': row.get('last_event_type') if pd.notna(row.get('last_event_type')) else None,
            'last_event_volume': float(row['last_event_volume']) if pd.notna(row.get('last_event_volume')) else None,
            'last_event_bars_ago': int(row['last_event_bars_ago']) if pd.notna(row.get('last_event_bars_ago')) else None,
            'change_pct': float(row['change_pct']) if pd.notna(row.get('change_pct')) else None,
            'vol_zscore': float(row['vol_zscore']) if pd.notna(row.get('vol_zscore')) else None,
            'bbmacd_event': row.get('bbmacd_event', '无') or '无',
            'daily_bb_width_zscore': float(row['daily_bb_width_zscore']) if pd.notna(row.get('daily_bb_width_zscore')) else None,
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


def select_stop_loss_stocks(selection_date: Optional[date] = None, save_to_db: bool = True) -> pd.DataFrame:
    """
    根据 Stop-Loss Clustering 指标选出满足条件的股票

    Args:
        selection_date: 选股日期，默认为当天
        save_to_db: 是否保存到数据库
    """
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("选股条件（Stop-Loss Clustering 策略）：")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  场景: 突破 sell-stop cluster（sell_stop_triggered = True）")
    print(f"  新增字段: sell_stop_scale = sum_sells_active * sell_trigger_max_vol_price")
    print(f"  过滤条件: 过去5天平均成交额 >= 1亿")
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
    print("开始 Stop-Loss Clustering 指标筛选...")
    print(f"  原股票数: {len(stock_list)}")

    filtered_results = []

    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="SLC选股", unit="只"):
        ts_code = row['ts_code']
        result = process_stock(ts_code, selection_date)
        if result:
            filtered_results.append(result)

    result_df = pd.DataFrame(filtered_results)

    if not result_df.empty:
        # 获取股票名称
        stock_names = batch_get_stock_names(result_df['ts_code'].tolist())
        result_df['stock_name'] = result_df['ts_code'].map(stock_names)

        # 分配批次号
        result_df['batch_no'] = (result_df.index // 10) + 1

    print("\n" + "=" * 80)
    print("选股结果汇总：")
    print("=" * 80)
    print(f"SLC筛选后: {len(result_df)} 只")

    if not result_df.empty:
        print(f"\n场景统计：")
        print(f"  突破 sell-stop: {result_df['sell_stop_triggered'].sum()} 只")

        batch_count = result_df['batch_no'].max()
        print(f"\n批次信息：共 {batch_count} 批，每批10只股票")

        print("\n" + "=" * 80)
        print("前20名股票：")
        print("=" * 80)
        display_cols = [
            'ts_code', 'stock_name', 'sell_stop_triggered',
            'sell_trigger_volume', 'sum_sells_active', 'sell_stop_scale'
        ]
        print_cols = [c for c in display_cols if c in result_df.columns]
        print(result_df[print_cols].head(20).to_string(index=False))

    # 保存到数据库
    if save_to_db:
        print("\n" + "-" * 80)
        print("保存到数据库...")
        saved_count = save_to_database(result_df, selection_date)
        print("-" * 80)

    return result_df


def parse_date(date_str: str) -> date:
    """解析日期字符串"""
    for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")


def test_single_stock(ts_code: str, selection_date: date):
    """测试单只股票的计算逻辑"""
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
        print("\n该股票不满足选股条件（未触发任何 stop cluster 或成交额不足）")

    return result


def backfill_stock_events(ts_code: str, start_date: date, end_date: date) -> List[Dict]:
    """
    遍历单只股票在日期区间内的所有交易日，逐日计算并记录触发事件。
    与按日期选股不同，此处按个股遍历，对每个交易日运行一次完整计算。

    Args:
        ts_code: 股票代码
        start_date: 起始日期（含）
        end_date: 结束日期（含）

    Returns:
        该股票在区间内所有触发日的记录列表
    """
    # 获取区间内全部日线数据（一次查询，减少数据库往返）
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=end_date)
    if daily_df.empty or len(daily_df) < 60:
        return []

    # 列名对齐
    if 'volume' in daily_df.columns and 'vol' not in daily_df.columns:
        daily_df = daily_df.rename(columns={'volume': 'vol'})
    if 'vol' in daily_df.columns and 'volume' not in daily_df.columns:
        daily_df['volume'] = daily_df['vol']

    # 过滤出区间内的交易日
    mask = (daily_df.index >= start_date) & (daily_df.index <= end_date)
    trade_dates = daily_df.loc[mask].index.tolist()
    if not trade_dates:
        return []

    results = []
    for trade_dt in trade_dates:
        # 截取到当前交易日为止的数据
        sub_df = daily_df.loc[daily_df.index <= trade_dt].copy()
        if len(sub_df) < 60:
            continue

        # 构造参数并运行核心计算
        args = SimpleNamespace(
            freq='d',
            model='absorbtion_extremes',
            show_historical_triggers=False,
            max_lines=20,
        )
        try:
            slc_engine = StopLossClusteringEngine(sub_df, args)
            slc_engine.run()
        except Exception:
            continue

        result_df = slc_engine.df
        if result_df.empty:
            continue

        last = result_df.iloc[-1]
        sell_triggered = bool(last.get('sell_stop_triggered', 0) == 1.0)

        # 只保留突破 sell-stop cluster 的股票
        if not sell_triggered:
            continue

        # 成交额过滤（当前交易日及前4日）
        if not check_volume_filter(sub_df, days=5, min_amount=100_000_000):
            continue

        # 计算 sell_stop_scale = sum_sells_active * sell_trigger_max_vol_price
        sum_sells_active = float(last.get('sum_sells_active', 0)) if pd.notna(last.get('sum_sells_active')) else 0.0
        sell_trigger_max_vol_price = float(last.get('sell_trigger_max_vol_price', 0)) if pd.notna(last.get('sell_trigger_max_vol_price')) else 0.0
        sell_stop_scale = sum_sells_active * sell_trigger_max_vol_price if sum_sells_active > 0 and sell_trigger_max_vol_price > 0 else 0.0

        # 计算上一次事件类型：从倒数第二根向前找最近一次触发（只找sell_stop）
        last_event_type = None
        last_event_volume = 0.0
        last_event_bars_ago = None
        for idx in range(len(result_df) - 2, -1, -1):
            row = result_df.iloc[idx]
            if row.get('sell_stop_triggered', 0) == 1.0:
                last_event_type = 'sell_stop'
                last_event_volume = float(row.get('sell_stop_triggered_volume', 0)) if pd.notna(row.get('sell_stop_triggered_volume')) else 0.0
                last_event_bars_ago = len(result_df) - 1 - idx
                break
        else:
            last_event_type = 'sell_stop'
            last_event_volume = float(last.get('sell_stop_triggered_volume', 0)) if pd.notna(last.get('sell_stop_triggered_volume')) else 0.0
            last_event_bars_ago = 0

        # 观察项
        bbmacd_df = compute_bbmacd(sub_df)
        bbmacd_event = detect_bbmacd_event(bbmacd_df)

        results.append({
            'ts_code': ts_code,
            'selection_date': trade_dt,
            'signal_date': trade_dt,
            'sell_stop_triggered': sell_triggered,
            'sell_trigger_volume': float(last.get('sell_stop_triggered_volume', 0)) if pd.notna(last.get('sell_stop_triggered_volume')) else 0.0,
            'active_sell_cluster_count': float(last.get('active_sell_cluster_count', 0)) if pd.notna(last.get('active_sell_cluster_count')) else 0.0,
            'sum_sells_active': sum_sells_active,
            'sell_trigger_max_vol_price': sell_trigger_max_vol_price,
            'sell_stop_scale': sell_stop_scale,
            'nearest_sell_stop_price': float(last.get('nearest_sell_stop_price', 0)) if pd.notna(last.get('nearest_sell_stop_price')) else None,
            'dist_to_nearest_sell_stop_atr': float(last.get('dist_to_nearest_sell_stop_atr', 0)) if pd.notna(last.get('dist_to_nearest_sell_stop_atr')) else None,
            'last_event_type': last_event_type,
            'last_event_volume': last_event_volume if last_event_type is not None else None,
            'last_event_bars_ago': last_event_bars_ago if last_event_type is not None else None,
            'change_pct': compute_change_pct(sub_df),
            'vol_zscore': volume_zscore(sub_df['volume'], win=20),
            'bbmacd_event': bbmacd_event,
            'daily_bb_width_zscore': float(bbmacd_df['bb_width_zscore'].iloc[-1]) if 'bb_width_zscore' in bbmacd_df.columns and len(bbmacd_df) > 0 else None,
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

    # 按日期分组
    from collections import defaultdict
    date_groups = defaultdict(list)
    for rec in records:
        dt = rec['selection_date']
        date_groups[dt].append(rec)

    total_saved = 0
    for dt, day_records in date_groups.items():
        # 先删除该日期该股票的数据（避免重复）
        ts_codes_in_day = [r['ts_code'] for r in day_records]
        placeholders = ', '.join([f"'{c}'" for c in ts_codes_in_day])
        with engine.connect() as conn:
            delete_sql = text(
                f"DELETE FROM {SELECTION_TABLE} "
                f"WHERE selection_date = :selection_date AND ts_code IN ({placeholders})"
            )
            conn.execute(delete_sql, {'selection_date': dt})
            conn.commit()

        # 插入新数据
        insert_records = []
        for rec in day_records:
            insert_records.append({
                'selection_date': dt,
                'signal_date': rec['signal_date'],
                'ts_code': rec['ts_code'],
                'stock_name': stock_name_map.get(rec['ts_code'], '') or '',
                'sell_stop_triggered': bool(rec.get('sell_stop_triggered', False)),
                'buy_stop_triggered': bool(rec.get('buy_stop_triggered', False)),
                'sell_trigger_volume': float(rec['sell_trigger_volume']) if pd.notna(rec.get('sell_trigger_volume')) else None,
                'active_sell_cluster_count': float(rec['active_sell_cluster_count']) if pd.notna(rec.get('active_sell_cluster_count')) else None,
                'sum_sells_active': float(rec['sum_sells_active']) if pd.notna(rec.get('sum_sells_active')) else None,
                'sell_trigger_max_vol_price': float(rec['sell_trigger_max_vol_price']) if pd.notna(rec.get('sell_trigger_max_vol_price')) else None,
                'sell_stop_scale': float(rec['sell_stop_scale']) if pd.notna(rec.get('sell_stop_scale')) else None,
                'nearest_sell_stop_price': float(rec['nearest_sell_stop_price']) if pd.notna(rec.get('nearest_sell_stop_price')) else None,
                'dist_to_nearest_sell_stop_atr': float(rec['dist_to_nearest_sell_stop_atr']) if pd.notna(rec.get('dist_to_nearest_sell_stop_atr')) else None,
                'last_event_type': rec.get('last_event_type') if pd.notna(rec.get('last_event_type')) else None,
                'last_event_volume': float(rec['last_event_volume']) if pd.notna(rec.get('last_event_volume')) else None,
                'last_event_bars_ago': int(rec['last_event_bars_ago']) if pd.notna(rec.get('last_event_bars_ago')) else None,
                'change_pct': float(rec['change_pct']) if pd.notna(rec.get('change_pct')) else None,
                'vol_zscore': float(rec['vol_zscore']) if pd.notna(rec.get('vol_zscore')) else None,
                'bbmacd_event': rec.get('bbmacd_event', '无') or '无',
                'daily_bb_width_zscore': float(rec['daily_bb_width_zscore']) if pd.notna(rec.get('daily_bb_width_zscore')) else None,
                'batch_no': None,
            })

        if insert_records:
            insert_df = pd.DataFrame(insert_records)
            insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)
            total_saved += len(insert_records)

    return total_saved


def backfill_range(start_date: date, end_date: date, stock_list: Optional[List[str]] = None):
    """
    回补指定日期区间内所有股票的 Stop-Loss Clustering 事件。
    遍历个股，对每个交易日逐日计算，每处理完一只股票立即保存。

    Args:
        start_date: 起始日期（含）
        end_date: 结束日期（含）
        stock_list: 指定股票列表，None 则自动获取全市场
    """
    print("=" * 80)
    print("Stop-Loss Clustering 事件回补")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  模式: 遍历个股，逐日计算事件，每只股票立即保存")
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
            # 立即保存该股票的结果
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
        description='Stop-Loss Clustering 选股工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/selection_stop.py                    # 使用当天日期选股
  python selection/selection_stop.py 2025-12-31         # 指定日期选股
  python selection/selection_stop.py 20251231           # 指定日期选股（无分隔符）
  python selection/selection_stop.py --test 600547      # 测试单只股票
  python selection/selection_stop.py --backfill 2025-07-01 2026-04-30  # 回补历史事件
        """
    )
    parser.add_argument(
        'date',
        nargs='?',
        help='选股日期 (格式: YYYY-MM-DD 或 YYYYMMDD)，默认为当天'
    )
    parser.add_argument(
        '--test',
        help='测试单只股票，例如: --test 600547'
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
        help='回补历史事件，遍历个股逐日计算，例如: --backfill 2025-07-01 2026-04-30'
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
    print("Stop-Loss Clustering 选股工具")
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print("=" * 80)

    # 测试模式
    if args.test:
        test_single_stock(args.test, selection_date)
        sys.exit(0)

    # 正常选股模式
    df = select_stop_loss_stocks(selection_date=selection_date, save_to_db=not args.no_save)

    print("\n" + "=" * 80)
    print("选股完成")
    print(f"选股日期: {selection_date}")
    print(f"选中股票数: {len(df)}")
    print(f"查询SQL: SELECT * FROM {SELECTION_TABLE} WHERE selection_date = '{selection_date}'")
    print("=" * 80)

    return df


if __name__ == '__main__':
    main()
