#!/usr/bin/env python3
"""
Z-score观察池选股脚本（日线级别，滚动3-bar重采样）

Purpose: 基于滚动3-bar成交量Z-score构建观察池，检查当天或前100bar内是否出现过Z-score >= 2.56
         所有其他指标（AMP、PAVP、DSA）仅作为观察项记录，不做硬过滤
Inputs: stock_k_data (日线K线数据)
Outputs: stock_selection_results (选股结果，字段带d_前缀表示日线)
How to Run:
    python selection/selection_amp.py              # 当天
    python selection/selection_amp.py 2026-04-10  # 指定日期
Examples:
    python selection/selection_amp.py
    python selection/selection_amp.py 2026-04-10
    python selection/selection_amp.py --test-single 000001
Side Effects: 写入 stock_selection_results 表（字段d_zscore_*, d_amp_*, d_pavp_*等）

================================================================================
【Z-score计算逻辑】

滚动3-bar重采样：
  - 每个交易日T，以T为终点，向前取3个bar（T-2, T-1, T）进行重采样
  - 成交量取3根之和，计算滚动均值后再计算Z-score
  - 每个交易日都有自己的Z-score值（不再是每3个交易日一个值）

Z-score公式：
  - 计算3-bar滚动成交量均值（当前窗口）
  - 历史回看50个滚动均值，计算历史均值和标准差
  - Z-score = (当前均值 - 历史均值) / 历史标准差

【选股条件】

保存条件：
  - 当天Z-score >= 2.56，当天即为触发日（距今0天）
  - 或前100个日线bar内出现过Z-score >= 2.56，记录最近的触发日
  - Z-score基于滚动3-bar重采样计算

观察项（记录但不筛选）：
  - d_amp_dir: AMP方向
  - d_amp_strength: AMP强度
  - d_amp_pos_01: AMP位置
  - d_pavp_lower_tail_share: 下尾占比
  - d_pavp_upper_tail_share: 上尾占比
  - d_dsa_vwap_dev: VWAP偏离度
  - d_zscore_value: 当前bar的Z-score值
  - d_zscore_trigger_date: 触发bar日期（当天或历史最近触发日）
  - d_zscore_bars_ago: 与触发bar相差的日线bar数（0表示当天触发）
  - d_price_change_pct: 当前股价与触发日股价的涨跌幅

保存字段：
  - d_zscore_triggered: 是否触发Z-score信号
  - d_zscore_value: 当前Z-score值
  - d_zscore_trigger_date: 触发日期
  - d_zscore_bars_ago: 距今bar数
  - d_zscore_trigger_value: 触发时的Z-score值
  - d_price_change_pct: 股价涨跌幅（当前vs触发日）
  - d_amp_strength: 日线AMP强度（观察项）
  - d_amp_dir: 日线AMP方向（观察项）
  - d_amp_pos_01: 日线收盘价在通道中的位置（观察项）
  - d_pavp_lower_tail_share: PAVP下尾占比（观察项）
  - d_pavp_upper_tail_share: PAVP上尾占比（观察项）
  - d_bar_time: 日线bar时间

【数据源】

优先从数据库stock_k_data表获取日线数据（freq='d'），
支持pytdx实时获取作为备选（通过--market-data-source=pytdx）

【保存逻辑】

按selection_date分组保存：
  - 保存前先删除该日期的旧数据（幂等性）
  - 再插入新数据
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
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

# 复用amp_pine_full_replica的核心计算逻辑（SSOT原则）
from features.amp_pine_full_replica import (
    compute_amp_core,
    detect_period,
    normalize_freq,
    fetch_kline_pytdx,
    AMPConfig,
    clamp01,
    safe_ratio,
)

# 复用dynamic_swing_anchored_vwap的DSA计算逻辑（SSOT原则）
from features.dynamic_swing_anchored_vwap import (
    DSAConfig,
    dynamic_swing_anchored_vwap,
)

# 复用bbmacd_viewer的BBMACD计算逻辑（SSOT原则）
from features.bbmacd_viewer import compute_bbmacd

# 复用pavp_tv_fixed_params_factors的PAVP计算逻辑（SSOT原则）
from features.pavp_tv_fixed_params_factors import compute_pavp

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "stock_selection_results"

# 默认配置（日线级别）
AMP_HISTORY_BARS = 400  # AMP计算需要的历史数据量
AMP_CFG = AMPConfig(useAdaptive=True, pI=200, devMultiplier=2.0, uL=True)

# DSA配置（日线级别）
DSA_CFG = DSAConfig(prd=50, baseAPT=20.0, useAdapt=False, volBias=10.0)

# Z-score相关配置
ZSCORE_WINDOW = 3          # Z-score滚动窗口（3个重采样bar）
ZSCORE_LOOKBACK = 50       # Z-score历史回看（50个重采样bar）
ZSCORE_THRESHOLD = 2.56    # Z-score阈值
ZSCORE_HISTORY_BARS = 100  # 回溯检查bar数（100个日线bar）

# 数据需求计算
# 需要：AMP_HISTORY_BARS(400) + ZSCORE_LOOKBACK*3(150) + ZSCORE_HISTORY_BARS(100) + 缓冲(50) ≈ 700bar
AMP_HISTORY_BARS_TOTAL = 700  # 更新为总需求
DSA_HISTORY_BARS = 200  # DSA计算需要的历史数据量

# 数据源配置
MARKET_DATA_SOURCE = "db"


def normalize_ts_code(ts_code: str) -> str:
    """标准化股票代码"""
    return str(ts_code).strip().upper().split('.')[0]


def get_kline_data_db(ts_code: str, freq: str = 'd', bars: int = 400, end_date: Optional[date] = None) -> pd.DataFrame:
    """从数据库获取K线数据

    Args:
        ts_code: 股票代码
        freq: 频率，默认'd'日线
        bars: 获取的K线数量
        end_date: 截止日期，只获取该日期及之前的K线

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    symbol = normalize_ts_code(ts_code)
    if end_date is not None:
        sql = """
            SELECT bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE (ts_code = :ts_code OR ts_code = :ts_code_sh OR ts_code = :ts_code_sz) AND freq = :freq
            AND DATE(bar_time) <= :end_date
            ORDER BY bar_time DESC
            LIMIT :bars
        """
        params = {
            'ts_code': symbol,
            'ts_code_sh': f'{symbol}.SH',
            'ts_code_sz': f'{symbol}.SZ',
            'freq': freq,
            'bars': bars,
            'end_date': end_date.strftime('%Y-%m-%d')
        }
    else:
        sql = """
            SELECT bar_time, open, high, low, close, volume
            FROM stock_k_data
            WHERE (ts_code = :ts_code OR ts_code = :ts_code_sh OR ts_code = :ts_code_sz) AND freq = :freq
            ORDER BY bar_time DESC
            LIMIT :bars
        """
        params = {
            'ts_code': symbol,
            'ts_code_sh': f'{symbol}.SH',
            'ts_code_sz': f'{symbol}.SZ',
            'freq': freq,
            'bars': bars
        }

    df = pd.read_sql(text(sql), engine, params=params)
    if not df.empty:
        df = df.sort_values('bar_time').set_index('bar_time')
    return df


def get_kline_data_pytdx(ts_code: str, freq: str = 'd', bars: int = 400, end_date: Optional[date] = None) -> pd.DataFrame:
    """从pytdx获取K线数据

    Args:
        ts_code: 股票代码
        freq: 频率，默认'd'日线
        bars: 获取的K线数量
        end_date: 截止日期，只获取该日期及之前的K线

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    raw = fetch_kline_pytdx(ts_code, freq, max(bars, 800))
    if end_date is not None:
        raw = raw[raw.index.date <= end_date]
    raw = raw.tail(bars).copy()
    raw.index.name = "bar_time"
    if "amount" in raw.columns:
        raw = raw.drop(columns=["amount"])
    return raw


def get_kline_data(ts_code: str, freq: str = 'd', bars: int = 400, end_date: Optional[date] = None) -> pd.DataFrame:
    """获取K线数据（根据 MARKET_DATA_SOURCE 选择数据源）

    Args:
        ts_code: 股票代码
        freq: 频率，默认'd'日线
        bars: 获取的K线数量
        end_date: 截止日期，只获取该日期及之前的K线
    """
    if MARKET_DATA_SOURCE == "pytdx":
        return get_kline_data_pytdx(ts_code, freq, bars, end_date)
    else:
        return get_kline_data_db(ts_code, freq, bars, end_date)


def get_stock_name(ts_code: str) -> str:
    """从stock_pools表获取股票名称"""
    sql = text("SELECT name FROM stock_pools WHERE ts_code = :ts_code LIMIT 1")
    with engine.connect() as conn:
        result = conn.execute(sql, {'ts_code': ts_code})
        row = result.fetchone()
        return row[0] if row else ''


def batch_get_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    """批量获取股票名称（一次查询）"""
    if not ts_codes:
        return {}
    placeholders = ', '.join([f"'{c}'" for c in ts_codes])
    sql = text(f"SELECT ts_code, name FROM stock_pools WHERE ts_code IN ({placeholders})")
    with engine.connect() as conn:
        result = conn.execute(sql)
        return {row[0]: row[1] for row in result}


def resample_to_3bar(df: pd.DataFrame) -> pd.DataFrame:
    """将原始日线数据重采样为3-bar数据

    从volume_zscore_pavp_explorer.py复用，保持逻辑完全一致
    """
    if len(df) < 6:
        return pd.DataFrame()

    resampled_data = []

    # 从后向前处理，确保最新的bar被包含
    for i in range(len(df) - 1, -1, -3):
        if i < 2:
            continue

        chunk = df.iloc[max(0, i-2):i+1]

        if len(chunk) < 3:
            continue

        resampled_data.append({
            'bar_time': df.index[i],  # 使用第3个bar的时间作为索引
            'open': chunk['open'].iloc[0],
            'high': chunk['high'].max(),
            'low': chunk['low'].min(),
            'close': chunk['close'].iloc[-1],
            'volume': chunk['volume'].sum(),
        })

    if not resampled_data:
        return pd.DataFrame()

    resampled_data.reverse()
    result_df = pd.DataFrame(resampled_data)
    result_df.set_index('bar_time', inplace=True)

    return result_df


def calculate_volume_zscore(df: pd.DataFrame, window: int = 3, lookback: int = 50) -> pd.Series:
    """计算成交量Z-Score（基于滚动3-bar重采样数据）

    修正后的逻辑：每个交易日都以当前日为终点，向前取3个bar进行重采样
    这样每个交易日都有自己的Z-score值，而不是每3个交易日才有一个

    返回：每个原始日线bar对应的Z-score值
    """
    if len(df) < lookback + window + 10:
        return pd.Series(index=df.index, dtype=float)

    # 滚动3-bar重采样：每个位置都计算以当前日为终点的3-bar重采样成交量
    rolling_3bar_volume = df['volume'].rolling(window=3, min_periods=3).sum()

    # 计算滚动window的成交量均值（基于3-bar重采样数据）
    rolling_vol_mean = rolling_3bar_volume.rolling(window=window, min_periods=window).mean()

    # 计算每个交易日的Z-score
    zscore_list = []

    for i in range(len(df)):
        if i < lookback + window + 2:  # 需要足够的历史数据
            zscore_list.append(np.nan)
            continue

        current_mean = rolling_vol_mean.iloc[i]

        if pd.isna(current_mean):
            zscore_list.append(np.nan)
            continue

        # 历史数据：向前看lookback个滚动均值
        hist_start = max(0, i - lookback - window + 1)
        hist_end = i - window + 1
        hist_data = rolling_vol_mean.iloc[hist_start:hist_end]

        if len(hist_data) < 10:
            zscore_list.append(np.nan)
            continue

        hist_mean = hist_data.mean()
        hist_std = hist_data.std()

        if hist_std == 0 or np.isnan(hist_std):
            zscore_list.append(np.nan)
            continue

        zscore = (current_mean - hist_mean) / hist_std
        zscore_list.append(zscore)

    return pd.Series(zscore_list, index=df.index)


def check_zscore_triggered(df: pd.DataFrame, current_idx: int,
                           lookback_bars: int = 100,
                           threshold: float = 2.56) -> Tuple[bool, float, Optional[pd.Timestamp], Optional[int], Optional[float], Optional[float]]:
    """检查指定日线bar是否满足Z-score条件（当天或前lookback_bars个bar内）

    逻辑：
    1. 如果当天Z-score >= threshold，当天就是触发日（距今0天）
    2. 如果当天Z-score < threshold，回溯历史找最近的触发日

    Args:
        df: 包含zscore列的DataFrame（日线级别）
        current_idx: 当前日线bar的索引
        lookback_bars: 回溯的日线bar数
        threshold: Z-score阈值

    Returns:
        Tuple[bool, float, Optional[pd.Timestamp], Optional[int], Optional[float], Optional[float]]:
            (是否触发, 当前Z-score值, 触发bar日期, 距今bar数, 触发日收盘价, 触发时Z-score值)
    """
    # 获取当前bar的Z-score
    current_zscore = df['zscore'].iloc[current_idx]

    # 首先检查当天是否触发
    if pd.notna(current_zscore) and current_zscore >= threshold:
        # 当天触发，距今0天
        trigger_date = df.index[current_idx]
        trigger_close = float(df['close'].iloc[current_idx])
        trigger_zscore = float(current_zscore)
        return True, current_zscore, trigger_date, 0, trigger_close, trigger_zscore

    # 当天未触发，回溯历史
    start_idx = max(0, current_idx - lookback_bars)
    history_zscore = df['zscore'].iloc[start_idx:current_idx]

    # 检查历史是否有触发（Z-score >= threshold）
    triggered = (history_zscore >= threshold).any()

    if triggered:
        # 找到最近的触发bar（在history_zscore中）
        trigger_mask = history_zscore >= threshold
        trigger_times = trigger_mask[trigger_mask].index

        # 获取最后一个触发的时间（最近的）
        trigger_time = trigger_times[-1]
        trigger_date = trigger_time

        # 获取触发bar在原始df中的位置索引
        trigger_idx = df.index.get_loc(trigger_time)
        bars_ago = current_idx - trigger_idx
        trigger_close = float(df['close'].iloc[trigger_idx])
        trigger_zscore = float(df['zscore'].iloc[trigger_idx])

        return True, current_zscore, trigger_date, bars_ago, trigger_close, trigger_zscore
    else:
        return False, current_zscore, None, None, None, None


def check_bbmacd_v_shape(daily_df: pd.DataFrame) -> bool:
    """检查日线BBMACD是否形成V型反转
    
    V型反转定义：
        - t-2 > t-1（前两天的BBMACD大于前一天，即下降）
        - t > t-1（当天BBMACD大于前一天，即上升）
        - 形成局部低点（V型底部）
    
    Args:
        daily_df: 日线DataFrame，需要包含最近3天的数据
        
    Returns: True-形成V型反转，False-未形成
    """
    if len(daily_df) < 3:
        return False
    
    try:
        # 计算BBMACD
        bbmacd_df = compute_bbmacd(daily_df)
        
        # 获取最近3天的BBMACD值
        bbmacd_vals = bbmacd_df['bbmacd'].values
        
        if len(bbmacd_vals) < 3:
            return False
        
        t_2 = bbmacd_vals[-3]  # 前两天
        t_1 = bbmacd_vals[-2]  # 前一天
        t = bbmacd_vals[-1]    # 当天
        
        # 检查是否为有效值
        if not (np.isfinite(t_2) and np.isfinite(t_1) and np.isfinite(t)):
            return False
        
        # V型反转条件：t-2 > t-1（下降）且 t > t-1（上升）
        return (t_2 > t_1) and (t > t_1)
    except Exception:
        return False


def process_stock_amp(ts_code: str, selection_date: Optional[date] = None) -> Optional[Dict]:
    """处理单只股票的AMP指标计算（日线级别，改造后基于Z-score观察池）

    选股逻辑（改造后）：
        1. 获取足够历史数据（AMP计算 + Z-score计算 + 回溯100日线bar）
        2. 计算Z-score序列（滚动3-bar重采样，每个交易日都有值）
        3. 对当前日线bar，检查当天或前100个日线bar是否有Z-score >= 2.56（硬过滤）
        4. 计算AMP/PAVP/DSA等指标（观察项，记录但不筛选）

    Args:
        ts_code: 股票代码
        selection_date: 选股日期，用于获取该日期之前的数据

    Returns: 信号字典，如果满足Z-score观察池条件则返回结果，否则返回None
    """
    # 1. 获取日线数据（需要足够历史数据）
    daily_df = get_kline_data(ts_code, 'd', bars=AMP_HISTORY_BARS_TOTAL, end_date=selection_date)
    if len(daily_df) < AMP_HISTORY_BARS_TOTAL * 0.8:
        return None

    try:
        # 2. 计算Z-score序列（基于滚动3-bar重采样，每个交易日都有值）
        zscore_series = calculate_volume_zscore(daily_df, window=ZSCORE_WINDOW, lookback=ZSCORE_LOOKBACK)
        daily_df['zscore'] = zscore_series

        # 3. 对当前bar（最后一个日线bar）检查当天或前100个日线bar是否有Z-score触发（硬过滤）
        current_idx = len(daily_df) - 1
        has_zscore, current_zscore, trigger_date, bars_ago, trigger_close, trigger_zscore = check_zscore_triggered(
            daily_df, current_idx, ZSCORE_HISTORY_BARS, ZSCORE_THRESHOLD
        )

        if not has_zscore:
            return None  # 当前bar不进入观察池

        # 4. 计算当前股价与触发日股价的涨跌幅
        current_close = float(daily_df['close'].iloc[current_idx])
        price_change_pct = (current_close - trigger_close) / trigger_close if trigger_close > 0 else 0.0

        # 5. 计算AMP指标（观察项，不再用于筛选）
        try:
            close_r = daily_df["close"].to_numpy(float)[::-1]
            final_period, _ = detect_period(close_r, AMP_CFG)
            lI = min(len(daily_df), int(final_period))
            if lI < 2:
                core = {"strength": np.nan, "dir": np.nan, "pos_01": np.nan,
                        "upper_end": np.nan, "lower_end": np.nan, "mid_end": np.nan}
            else:
                df_win = daily_df.iloc[-lI:].copy()
                core = compute_amp_core(df_win, AMP_CFG)
        except Exception:
            core = {"strength": np.nan, "dir": np.nan, "pos_01": np.nan,
                    "upper_end": np.nan, "lower_end": np.nan, "mid_end": np.nan}

        # 6. 计算DSA指标（观察项）
        try:
            dsa_df = daily_df.copy()
            dsa_df['hlc3'] = (dsa_df['high'] + dsa_df['low'] + dsa_df['close']) / 3.0
            vwap_series, dir_series, pivot_labels, segments = dynamic_swing_anchored_vwap(dsa_df, DSA_CFG)
            last_vwap = float(vwap_series.iloc[-1])
            last_dsa_dir = int(dir_series.iloc[-1])
            signed_vwap_dev = (float(daily_df['close'].iloc[-1]) - last_vwap) / last_vwap if last_vwap > 0 else 0.0
        except Exception:
            last_vwap = np.nan
            last_dsa_dir = np.nan
            signed_vwap_dev = np.nan

        # 7. 计算PAVP指标（观察项，不再用于筛选）
        try:
            pavp_df, fixed_segments, last_dev = compute_pavp(daily_df)
            pavp_va_pos_01 = float(pavp_df['va_pos_01'].iloc[-1]) if 'va_pos_01' in pavp_df.columns else np.nan
            pavp_poc_va_pos_01 = float(pavp_df['poc_in_va_pos_01'].iloc[-1]) if 'poc_in_va_pos_01' in pavp_df.columns else np.nan
            pavp_close_to_poc_pct = float(pavp_df['close_to_poc_pct'].iloc[-1]) if 'close_to_poc_pct' in pavp_df.columns else np.nan
            pavp_lower_tail_share = float(pavp_df['profile_lower_tail_share'].iloc[-1]) if 'profile_lower_tail_share' in pavp_df.columns else np.nan
            pavp_upper_tail_share = float(pavp_df['profile_upper_tail_share'].iloc[-1]) if 'profile_upper_tail_share' in pavp_df.columns else np.nan
        except Exception:
            pavp_va_pos_01 = np.nan
            pavp_poc_va_pos_01 = np.nan
            pavp_close_to_poc_pct = np.nan
            pavp_lower_tail_share = np.nan
            pavp_upper_tail_share = np.nan

        # 8. 计算当日涨跌幅（相对于前一日收盘价）
        if len(daily_df) >= 2:
            last_close = float(daily_df['close'].iloc[-1])
            prev_close = float(daily_df['close'].iloc[-2])
            change_pct = (last_close - prev_close) / prev_close if prev_close > 0 else 0.0
        else:
            change_pct = 0.0

        # 返回结果（包含Z-score触发信息和所有观察项）
        return {
            'ts_code': ts_code,
            # Z-score触发信息
            'd_zscore_triggered': 1,
            'd_zscore_value': current_zscore,
            'd_zscore_trigger_date': trigger_date,
            'd_zscore_bars_ago': bars_ago,
            'd_zscore_trigger_value': trigger_zscore,
            'd_price_change_pct': price_change_pct,
            # AMP观察项
            'd_amp_strength': core["strength"],
            'd_amp_dir': core["dir"],
            'd_amp_pos_01': core["pos_01"],
            'd_amp_period': lI,
            'd_amp_upper': core["upper_end"],
            'd_amp_lower': core["lower_end"],
            'd_amp_mid': core["mid_end"],
            # DSA观察项
            'd_dsa_dir': last_dsa_dir,
            'd_dsa_vwap': last_vwap,
            'd_dsa_vwap_dev': signed_vwap_dev,
            # PAVP观察项
            'd_pavp_va_pos_01': pavp_va_pos_01,
            'd_pavp_poc_va_pos_01': pavp_poc_va_pos_01,
            'd_pavp_close_to_poc_pct': pavp_close_to_poc_pct,
            'd_pavp_lower_tail_share': pavp_lower_tail_share,
            'd_pavp_upper_tail_share': pavp_upper_tail_share,
            # 其他
            'd_change_pct': change_pct,
            'd_bar_time': daily_df.index[-1],
        }
    except Exception:
        return None


def ensure_table_exists(engine):
    """确保选股结果表存在（添加AMP字段，带d_前缀表示日线）"""
    # 先确保基础表存在（参考selection_ana.py）
    create_sql = """
    CREATE TABLE IF NOT EXISTS stock_selection_results (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        signal_date DATE,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),
        report_date VARCHAR(8),
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
        daily_reversal_buy BOOLEAN DEFAULT FALSE,
        daily_breakout_buy BOOLEAN DEFAULT FALSE,
        weekly_reversal_buy BOOLEAN DEFAULT FALSE,
        weekly_breakout_buy BOOLEAN DEFAULT FALSE,
        daily_bb_width_zscore FLOAT,
        weekly_bb_width_zscore FLOAT,
        daily_vol_zscore FLOAT,
        weekly_vol_zscore FLOAT,
        weekly_vwap_deviation FLOAT,
        change_pct FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_selection_date ON stock_selection_results(selection_date);
    CREATE INDEX IF NOT EXISTS idx_signal_date ON stock_selection_results(signal_date);
    CREATE INDEX IF NOT EXISTS idx_selection_ts_code ON stock_selection_results(ts_code);
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()

        # 添加AMP相关字段（带d_前缀表示日线）
        amp_columns = [
            ('d_amp_strength', 'FLOAT'),
            ('d_amp_dir', 'INTEGER'),
            ('d_amp_pos_01', 'FLOAT'),
            ('d_amp_period', 'INTEGER'),
            ('d_amp_upper', 'FLOAT'),
            ('d_amp_lower', 'FLOAT'),
            ('d_amp_mid', 'FLOAT'),
            ('d_dsa_dir', 'INTEGER'),
            ('d_dsa_vwap', 'FLOAT'),
            ('d_dsa_vwap_dev', 'FLOAT'),
            ('d_bar_time', 'TIMESTAMP'),
            # PAVP字段
            ('d_pavp_va_pos_01', 'FLOAT'),
            ('d_pavp_poc_va_pos_01', 'FLOAT'),
            ('d_pavp_close_to_poc_pct', 'FLOAT'),
            ('d_pavp_lower_tail_share', 'FLOAT'),
            ('d_pavp_upper_tail_share', 'FLOAT'),
            # 涨跌幅字段
            ('d_change_pct', 'FLOAT'),
            # Z-score字段
            ('d_zscore_triggered', 'INTEGER'),
            ('d_zscore_value', 'FLOAT'),
            ('d_zscore_trigger_date', 'DATE'),
            ('d_zscore_bars_ago', 'INTEGER'),
            ('d_zscore_trigger_value', 'FLOAT'),
            ('d_price_change_pct', 'FLOAT'),
        ]
        for col_name, col_type in amp_columns:
            try:
                conn.execute(text(f"ALTER TABLE {SELECTION_TABLE} ADD COLUMN IF NOT EXISTS {col_name} {col_type}"))
                conn.commit()
            except Exception:
                conn.rollback()


def save_to_database(df, selection_date):
    """保存选股结果到数据库（AMP字段带d_前缀），按日期覆盖"""
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
            'report_date': selection_date.strftime('%Y%m%d'),
            'd_amp_strength': float(row['d_amp_strength']) if pd.notna(row.get('d_amp_strength')) else None,
            'd_amp_dir': int(row['d_amp_dir']) if pd.notna(row.get('d_amp_dir')) else None,
            'd_amp_pos_01': float(row['d_amp_pos_01']) if pd.notna(row.get('d_amp_pos_01')) else None,
            'd_amp_period': int(row['d_amp_period']) if pd.notna(row.get('d_amp_period')) else None,
            'd_amp_upper': float(row['d_amp_upper']) if pd.notna(row.get('d_amp_upper')) else None,
            'd_amp_lower': float(row['d_amp_lower']) if pd.notna(row.get('d_amp_lower')) else None,
            'd_amp_mid': float(row['d_amp_mid']) if pd.notna(row.get('d_amp_mid')) else None,
            'd_dsa_dir': int(row['d_dsa_dir']) if pd.notna(row.get('d_dsa_dir')) else None,
            'd_dsa_vwap': float(row['d_dsa_vwap']) if pd.notna(row.get('d_dsa_vwap')) else None,
            'd_dsa_vwap_dev': float(row['d_dsa_vwap_dev']) if pd.notna(row.get('d_dsa_vwap_dev')) else None,
            'd_bar_time': row.get('d_bar_time', None),
            # PAVP字段
            'd_pavp_va_pos_01': float(row['d_pavp_va_pos_01']) if pd.notna(row.get('d_pavp_va_pos_01')) else None,
            'd_pavp_poc_va_pos_01': float(row['d_pavp_poc_va_pos_01']) if pd.notna(row.get('d_pavp_poc_va_pos_01')) else None,
            'd_pavp_close_to_poc_pct': float(row['d_pavp_close_to_poc_pct']) if pd.notna(row.get('d_pavp_close_to_poc_pct')) else None,
            'd_pavp_lower_tail_share': float(row['d_pavp_lower_tail_share']) if pd.notna(row.get('d_pavp_lower_tail_share')) else None,
            'd_pavp_upper_tail_share': float(row['d_pavp_upper_tail_share']) if pd.notna(row.get('d_pavp_upper_tail_share')) else None,
            # 涨跌幅字段
            'd_change_pct': float(row['d_change_pct']) if pd.notna(row.get('d_change_pct')) else None,
            # Z-score字段
            'd_zscore_triggered': int(row['d_zscore_triggered']) if pd.notna(row.get('d_zscore_triggered')) else None,
            'd_zscore_value': float(row['d_zscore_value']) if pd.notna(row.get('d_zscore_value')) else None,
            'd_zscore_trigger_date': row.get('d_zscore_trigger_date'),
            'd_zscore_bars_ago': int(row['d_zscore_bars_ago']) if pd.notna(row.get('d_zscore_bars_ago')) else None,
            'd_zscore_trigger_value': float(row['d_zscore_trigger_value']) if pd.notna(row.get('d_zscore_trigger_value')) else None,
            'd_price_change_pct': float(row['d_price_change_pct']) if pd.notna(row.get('d_price_change_pct')) else None,
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
    if records:
        insert_df = pd.DataFrame(records)
        insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)
        print(f"  保存新数据: {len(records)} 条")

    return len(records)


def select_amp_stocks(selection_date: Optional[date] = None, save_to_db: bool = True):
    """根据AMP指标选出满足条件的股票（日线级别）

    Args:
        selection_date: 选股日期，默认为当天
        save_to_db: 是否保存到数据库
    """
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("AMP指标选股（日线级别）")
    print("=" * 80)
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"行情数据源: {MARKET_DATA_SOURCE}")
    print(f"选股条件:")
    print(f"  - 日线AMP方向向上 (d_amp_dir=1)")
    print(f"  - 日线强度大于0.8 (d_amp_strength>0.8)")
    print(f"  - 股价通道位置<0.6 (d_amp_pos_01<0.6，排除高位股)")
    print(f"  - 下尾占比>上尾*1.2 (d_pavp_lower_tail_share>d_pavp_upper_tail_share*1.2，底部放量)")
    print("=" * 80)

    with engine.connect() as conn:
        print("\n查询所有股票...")
        # 从日线数据中获取当日有数据的股票列表
        sql = text("""
            SELECT DISTINCT ts_code
            FROM stock_k_data
            WHERE freq = 'd' AND DATE(bar_time) = :selection_date
        """)
        stock_list = pd.read_sql(sql, conn, params={'selection_date': selection_date.strftime('%Y-%m-%d')})
        print(f"  找到 {len(stock_list)} 只股票")

    if len(stock_list) > 0:
        print("\n" + "=" * 80)
        print("开始AMP指标筛选...")
        print("=" * 80)

        filtered_results = []
        success_count = 0
        fail_count = 0

        # 使用tqdm显示进度条
        for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="AMP选股处理", unit="只"):
            ts_code = row['ts_code']

            try:
                result = process_stock_amp(ts_code, selection_date)
                if result:
                    filtered_results.append(result)
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                # 可选：记录错误日志
                # print(f"\n  处理 {ts_code} 时出错: {e}")

        print(f"\n  处理完成: 成功 {success_count} 只, 不满足条件 {fail_count} 只")

        result_df = pd.DataFrame(filtered_results)

        if not result_df.empty:
            stock_names = batch_get_stock_names(result_df['ts_code'].tolist())
            result_df['stock_name'] = result_df['ts_code'].map(stock_names)

        print("\n" + "=" * 80)
        print("选股结果汇总：")
        print("=" * 80)
        print(f"AMP筛选后: {len(result_df)} 只")

        if not result_df.empty:
            print(f"\n平均强度: {result_df['d_amp_strength'].mean():.3f}")
            print(f"强度范围: {result_df['d_amp_strength'].min():.3f} ~ {result_df['d_amp_strength'].max():.3f}")

            print("\n" + "=" * 80)
            print("前20名股票：")
            print("=" * 80)
            display_cols = ['ts_code', 'stock_name', 'd_amp_strength', 'd_amp_pos_01', 'd_change_pct', 'd_pavp_lower_tail_share', 'd_pavp_upper_tail_share', 'd_amp_period']
            print(result_df[display_cols].head(20).to_string(index=False))

        # 保存到数据库
        if save_to_db:
            print("\n" + "-" * 80)
            print("保存到数据库...")
            saved_count = save_to_database(result_df, selection_date)
            print("-" * 80)

        return result_df
    else:
        print("\n未找到符合条件的股票")
        return pd.DataFrame()


def test_single_stock(ts_code: str, selection_date: Optional[date] = None):
    """测试单只股票的Z-score观察池计算"""
    print("\n" + "=" * 80)
    print("单只股票Z-score观察池测试（日线级别）")
    print("=" * 80)
    print(f"股票代码: {ts_code}")
    print(f"选股日期: {selection_date or date.today()}")
    print(f"数据源: {MARKET_DATA_SOURCE}")
    print(f"Z-score阈值: {ZSCORE_THRESHOLD}")
    print(f"回溯bar数: {ZSCORE_HISTORY_BARS}")
    print("=" * 80)

    result = process_stock_amp(ts_code, selection_date)

    if result:
        print("\n满足Z-score观察池条件！")
        print("\n【Z-score触发信息】")
        print(f"  d_zscore_triggered: {result['d_zscore_triggered']}")
        print(f"  d_zscore_value: {result['d_zscore_value']:.4f}")
        print(f"  d_zscore_trigger_date: {result['d_zscore_trigger_date']}")
        print(f"  d_zscore_bars_ago: {result['d_zscore_bars_ago']}")
        print(f"  d_zscore_trigger_value: {result['d_zscore_trigger_value']:.4f}")
        print(f"  d_price_change_pct: {result['d_price_change_pct']*100:.2f}%")
        print("\n【AMP观察项】")
        print(f"  d_amp_strength: {result['d_amp_strength']:.4f}")
        print(f"  d_amp_dir: {result['d_amp_dir']}")
        print(f"  d_amp_pos_01: {result['d_amp_pos_01']:.4f}")
        print(f"  d_amp_period: {result['d_amp_period']}")
        print(f"  d_amp_upper: {result['d_amp_upper']:.2f}")
        print(f"  d_amp_lower: {result['d_amp_lower']:.2f}")
        print(f"  d_amp_mid: {result['d_amp_mid']:.2f}")
        print("\n【PAVP观察项】")
        print(f"  d_pavp_va_pos_01: {result.get('d_pavp_va_pos_01', 'N/A'):.4f}")
        print(f"  d_pavp_lower_tail_share: {result.get('d_pavp_lower_tail_share', 'N/A'):.4f}")
        print(f"  d_pavp_upper_tail_share: {result.get('d_pavp_upper_tail_share', 'N/A'):.4f}")
        print("\n【其他】")
        print(f"  d_change_pct: {result.get('d_change_pct', 'N/A'):.4f}")
        print(f"  d_bar_time: {result['d_bar_time']}")
    else:
        print("\n不满足Z-score观察池条件（前100bar内未出现Z-score >= 2.56）")

        # 打印原始计算结果用于调试
        daily_df = get_kline_data(ts_code, 'd', bars=AMP_HISTORY_BARS_TOTAL, end_date=selection_date)
        if len(daily_df) >= AMP_HISTORY_BARS_TOTAL * 0.8:
            try:
                # Z-score计算
                zscore_series = calculate_volume_zscore(daily_df, window=ZSCORE_WINDOW, lookback=ZSCORE_LOOKBACK)
                daily_df['zscore'] = zscore_series

                # 检查最近100bar的Z-score
                recent_zscore = zscore_series.tail(ZSCORE_HISTORY_BARS).dropna()
                max_zscore = recent_zscore.max() if len(recent_zscore) > 0 else 0

                print(f"\n实际计算结果（用于调试）：")
                print(f"  最近100bar最大Z-score: {max_zscore:.4f}")
                print(f"  Z-score阈值: {ZSCORE_THRESHOLD}")
                if max_zscore >= ZSCORE_THRESHOLD:
                    trigger_idx = recent_zscore.idxmax()
                    print(f"  最近触发日期: {trigger_idx}")
                    print(f"  距今bar数: {len(daily_df) - daily_df.index.get_loc(trigger_idx) - 1}")

                # AMP计算（观察项）
                close_r = daily_df["close"].to_numpy(float)[::-1]
                final_period, _ = detect_period(close_r, AMP_CFG)
                lI = min(len(daily_df), int(final_period))
                if lI >= 2:
                    df_win = daily_df.iloc[-lI:].copy()
                    core = compute_amp_core(df_win, AMP_CFG)
                    print(f"\n【AMP观察项（调试）】")
                    print(f"  d_amp_strength: {core['strength']:.4f}")
                    print(f"  d_amp_dir: {core['dir']}")
                    print(f"  d_amp_pos_01: {core['pos_01']:.4f}")

            except Exception as e:
                print(f"计算出错: {e}")
        else:
            print(f"数据不足: 只有{len(daily_df)}根日线，需要至少{int(AMP_HISTORY_BARS_TOTAL * 0.8)}根")

    print("=" * 80)
    return result


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
        description='AMP指标选股工具（日线级别，方向向上且强度大于0.8）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/selection_amp.py                    # 使用当天日期选股
  python selection/selection_amp.py 2025-12-31         # 指定日期选股
  python selection/selection_amp.py 20251231           # 指定日期选股（无分隔符）
  python selection/selection_amp.py --test-single 000001  # 测试单只股票
        """
    )
    parser.add_argument(
        'date',
        nargs='?',
        help='选股日期 (格式: YYYY-MM-DD 或 YYYYMMDD)，默认为当天'
    )
    parser.add_argument(
        '--test-single',
        help='测试单只股票的AMP计算，如 000001'
    )
    parser.add_argument(
        '--market-data-source',
        default='db',
        choices=['db', 'pytdx'],
        help='K线行情数据源，默认 db；如需pytdx可传 pytdx'
    )

    args = parser.parse_args()

    # 设置全局数据源
    global MARKET_DATA_SOURCE
    MARKET_DATA_SOURCE = args.market_data_source

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
    print("AMP指标选股工具（日线级别）")
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"行情数据源: {MARKET_DATA_SOURCE}")
    print("=" * 80)

    if args.test_single:
        result = test_single_stock(args.test_single, selection_date)
        sys.exit(0 if result else 1)

    df = select_amp_stocks(selection_date=selection_date, save_to_db=True)

    print("\n" + "=" * 80)
    print("选股完成")
    print(f"选股日期: {selection_date}")
    print(f"选中股票数: {len(df)}")
    print(f"查询SQL: SELECT * FROM {SELECTION_TABLE} WHERE selection_date = '{selection_date}'")
    print("=" * 80)

    return df


if __name__ == '__main__':
    # 自测入口：测试单只股票的AMP计算
    # 运行: python selection/selection_amp.py --test-single 000001
    main()
