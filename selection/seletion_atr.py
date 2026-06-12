#!/usr/bin/env python3
"""
ATR Rope 选股脚本（基于 atr_rope_event_factor_lab_v4 模型）

Purpose:
    基于 ATR Rope 指标，先判断个股大背景（多头/空头/震荡），
    再按场景触发买入信号：多头场景上穿趋势线，震荡场景上穿下轨或中轨。

Inputs:
    stock_k_data (日线K线数据，open/high/low/close/volume)

Outputs:
    atr_rope_selection (选股结果表)

How to Run:
    python selection/seletion_atr.py              # 当天
    python selection/seletion_atr.py 2026-05-19   # 指定日期
    python selection/seletion_atr.py --test 600547 # 测试单只
    python selection/seletion_atr.py --no-save     # 不写入数据库
    python selection/seletion_atr.py --backfill 2025-07-01 2026-04-30  # 回补历史

Side Effects:
    写入 atr_rope_selection 表（幂等：同一日期先删后插）

================================================================================
【选股逻辑】

大背景判断（DSA VWAP dir 持续 bar 数）：
  - 1 = 多头：DSA VWAP dir=1 持续 > 50 bars
  - -1 = 空头：DSA VWAP dir=-1 持续 > 50 bars
  - 0 = 震荡：不满足上述条件

场景1 — 多头（regime == 1）：
  事件触发：ATR Rope dir=0（进入蓝色区间）→ 记录"多头蓝色区间"事件
  记录：蓝色箱体内位置 range_pos_01（0=下轨，1=上轨）
  注：dir=1 时不触发任何事件

场景2 — 震荡（regime == 0）：
  买入触发：ATR Rope dir=0（活跃蓝色箱体）时，close 上穿蓝色箱体下轨（atr_rope_c_lo）或 上穿箱体中轨（(c_hi+c_lo)/2）
    注：dir≠0 时 c_lo/c_hi 是旧蓝色区间残留值，不触发
  卖出参考：跌破蓝色箱体下轨（close < c_lo）
  趋势转换：如果 regime 从震荡转为多头，卖出参考切换为 rope 线

场景3 — 空头（regime == -1）：
  不触发买入

过滤条件：
  - 过去5天平均成交额 >= 1亿

记录字段（核心）：
  - regime / regime_value / regime_strength：大背景类型/数值/强度
  - signal_type：买入信号类型
  - rope_value / lower_value / upper_value：rope/下轨/上轨值
    注：lower/upper 仅在震荡 或 多头+ATR Rope dir=0 时记录(c_lo/c_hi)，
    多头+dir=1 时为 NULL（无有意义的上下边界）
  - rope_dir / rope_dev_pct / rope_dev_atr：方向/偏离百分比/偏离ATR倍数
  - range_width_pct：蓝色箱体宽度百分比（(c_hi/c_lo-1)*100），仅dir=0时有效
  - rope_dir1_pct / rope_dir_neg1_pct：DSA趋势区间内ATR Rope dir=1/dir=-1占比（%）
  - range_pos_01：蓝色箱体内位置（0=下轨，1=上轨），仅dir=0时有效
  - low_rope_dev_mean/std/upper/mid/lower/today/signal：多头蓝色箱体内low-rope偏离布林带统计
  - low_vwap_dev_mean/std/upper/mid/lower/today/signal：多头趋势内low-vwap偏离布林带统计
  - dsa_vwap_dev_pct：收盘价与DSA VWAP偏离率（%）

观察项：
  - change_pct：选股日涨跌幅
  - vol_zscore：成交量 Z-Score
  - bbmacd_event：BBMACD 事件

【核心计算】
  - 全部引用 features.atr_rope_event_factor_lab_v4.compute_atr_rope
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
from features.bbmacd_viewer import compute_bbmacd

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "atr_rope_selection"

# K线数据配置
DAILY_BARS = 800  # 日线数据获取800根（约3年）


def normalize_ts_code(ts_code: str) -> str:
    """标准化股票代码"""
    return str(ts_code).strip().upper().split('.')[0]


def get_kline_data_db(ts_code: str, bars: int = DAILY_BARS, end_date: Optional[date] = None) -> pd.DataFrame:
    """从数据库获取日线K线数据（前复权）"""
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
        # 确定实际 ts_code 用于前复权转换
        suffix = '.SH' if symbol.startswith(('6', '9')) else '.SZ'
        actual_ts_code = f'{symbol}{suffix}'
        df = apply_adj_factor(df, actual_ts_code, freq='d')
    return df


def compute_dsa_regime(df: pd.DataFrame, cfg: DSAConfig = None, min_bars: int = 50):
    """
    基于 DSA VWAP dir 计算大背景 regime 和趋势强度：
    - dir=1 持续 > min_bars → 1（多头）
    - dir=-1 持续 > min_bars → -1（空头）
    - 其他 → 0（震荡）

    趋势强度 = 当前 dir 持续期间内 VWAP 平均每 bar 收益率
    - 多头时为正值（VWAP 上升），空头时为负值

    Returns:
        (regime: pd.Series, trend_strength: pd.Series)
    """
    if cfg is None:
        cfg = DSAConfig()
    vwap_series, dir_series, _, _ = dynamic_swing_anchored_vwap(df, cfg)
    dir_vals = dir_series.fillna(0).astype(int)
    # 计算连续同向 bar 数（正数=dir=1持续, 负数=dir=-1持续）
    bars = pd.Series(0, index=df.index, dtype=int)
    for i in range(len(dir_vals)):
        if i == 0:
            bars.iloc[i] = dir_vals.iloc[i]
        elif dir_vals.iloc[i] == dir_vals.iloc[i - 1]:
            bars.iloc[i] = bars.iloc[i - 1] + dir_vals.iloc[i]
        else:
            bars.iloc[i] = dir_vals.iloc[i]

    # regime: dir=1 持续>50 → 1, dir=-1 持续>50 → -1, 其他 → 0
    regime = pd.Series(0, index=df.index, dtype=int)
    regime[bars > min_bars] = 1
    regime[bars < -min_bars] = -1

    # 趋势强度：当前 dir 持续期间内 VWAP 平均每 bar 收益率
    trend_strength = pd.Series(0.0, index=df.index)
    vwap_vals = vwap_series.astype(float)
    for i in range(len(dir_vals)):
        num_bars = abs(bars.iloc[i])
        if num_bars <= 1:
            trend_strength.iloc[i] = 0.0
            continue
        start_idx = i - num_bars + 1
        if start_idx < 0:
            start_idx = 0
        vwap_start = vwap_vals.iloc[start_idx]
        vwap_end = vwap_vals.iloc[i]
        if pd.notna(vwap_start) and pd.notna(vwap_end) and vwap_start != 0:
            trend_strength.iloc[i] = (vwap_end / vwap_start - 1) / num_bars

    return regime, trend_strength, bars, vwap_series


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


def compute_rope_dir_pct_in_dsa_trend(result_df: pd.DataFrame, dsa_bars_val: int):
    """
    在DSA最近一段趋势区间内，统计ATR Rope dir=1和dir=-1的占比。
    dsa_bars_val: 当前DSA趋势持续的bar数（正数=多头，负数=空头）
    Returns: (dir1_pct, dir_neg1_pct) 百分比，如 (60.0, 15.0)
    """
    n = abs(dsa_bars_val)
    if n <= 0 or len(result_df) < n:
        return None, None
    segment = result_df['atr_rope_dir'].iloc[-n:]
    total = len(segment)
    if total == 0:
        return None, None
    dir1_pct = round(float((segment == 1).sum()) / total * 100, 2)
    dir_neg1_pct = round(float((segment == -1).sum()) / total * 100, 2)
    return dir1_pct, dir_neg1_pct


def compute_low_rope_dev_bollinger(result_df: pd.DataFrame, dsa_bars_val: int):
    """
    在DSA多头趋势区间内，仅取ATR Rope dir=0（蓝色箱体）的bar，
    计算 low-rope 偏离差值的布林带统计。
    用于判断多头趋势中蓝色箱体内的低点偏离趋势线的正常波动范围。

    Returns: dict with mean/std/upper/mid/lower/today/signal
    """
    _empty = {k: None for k in ['low_rope_dev_mean', 'low_rope_dev_std',
                                  'low_rope_dev_upper', 'low_rope_dev_mid',
                                  'low_rope_dev_lower', 'low_rope_dev_today',
                                  'low_rope_signal']}
    n = abs(dsa_bars_val)
    if n <= 0 or len(result_df) < n:
        return _empty

    # 取DSA趋势区间内 dir=0 的bar
    segment = result_df.iloc[-n:]
    blue_mask = segment['atr_rope_dir'] == 0
    blue_bars = segment[blue_mask]

    if len(blue_bars) < 2:
        return _empty

    dev = blue_bars['low'] - blue_bars['atr_rope_rope']
    dev_mean = float(dev.mean())
    dev_std = float(dev.std())
    if dev_std == 0 or pd.isna(dev_std):
        return _empty

    upper = dev_mean + 2 * dev_std
    mid = dev_mean
    lower = dev_mean - 2 * dev_std

    # 当日偏离值
    last = result_df.iloc[-1]
    dev_today = float(last['low'] - last['atr_rope_rope'])

    # 信号判断
    if dev_today <= lower:
        signal = '触及下轨'
    elif dev_today <= mid:
        signal = '触及中轨'
    elif dev_today >= upper:
        signal = '触及上轨'
    else:
        signal = '无'

    return {
        'low_rope_dev_mean': round(dev_mean, 4),
        'low_rope_dev_std': round(dev_std, 4),
        'low_rope_dev_upper': round(upper, 4),
        'low_rope_dev_mid': round(mid, 4),
        'low_rope_dev_lower': round(lower, 4),
        'low_rope_dev_today': round(dev_today, 4),
        'low_rope_signal': signal,
    }


def compute_low_vwap_dev_bollinger(daily_df: pd.DataFrame, dsa_vwap: pd.Series, dsa_bars_val: int):
    """
    在DSA多头趋势区间内，取所有bar计算 low - dsa_vwap 偏离差值的布林带统计。
    用于判断多头趋势中低点偏离VWAP的正常波动范围。

    Returns: dict with mean/std/upper/mid/lower/today/signal
    """
    _empty = {k: None for k in ['low_vwap_dev_mean', 'low_vwap_dev_std',
                                  'low_vwap_dev_upper', 'low_vwap_dev_mid',
                                  'low_vwap_dev_lower', 'low_vwap_dev_today',
                                  'low_vwap_signal']}
    n = abs(dsa_bars_val)
    if n <= 0 or len(daily_df) < n:
        return _empty

    segment = daily_df.iloc[-n:]
    vwap_segment = dsa_vwap.iloc[-n:]

    # 对齐索引
    common_idx = segment.index.intersection(vwap_segment.index)
    if len(common_idx) < 2:
        return _empty

    low_vals = segment.loc[common_idx, 'low']
    vwap_vals = vwap_segment.loc[common_idx]

    dev = low_vals - vwap_vals
    dev = dev.dropna()
    if len(dev) < 2:
        return _empty

    dev_mean = float(dev.mean())
    dev_std = float(dev.std())
    if dev_std == 0 or pd.isna(dev_std):
        return _empty

    upper = dev_mean + 2 * dev_std
    mid = dev_mean
    lower = dev_mean - 2 * dev_std

    # 当日偏离值
    last = daily_df.iloc[-1]
    last_vwap = dsa_vwap.iloc[-1]
    if pd.isna(last_vwap) or last_vwap == 0:
        return _empty
    dev_today = float(last['low'] - last_vwap)

    # 信号判断
    if dev_today <= lower:
        signal = '触及下轨'
    elif dev_today <= mid:
        signal = '触及中轨'
    elif dev_today >= upper:
        signal = '触及上轨'
    else:
        signal = '无'

    return {
        'low_vwap_dev_mean': round(dev_mean, 4),
        'low_vwap_dev_std': round(dev_std, 4),
        'low_vwap_dev_upper': round(upper, 4),
        'low_vwap_dev_mid': round(mid, 4),
        'low_vwap_dev_lower': round(lower, 4),
        'low_vwap_dev_today': round(dev_today, 4),
        'low_vwap_signal': signal,
    }


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


def compute_stock_features(ts_code: str, selection_date: date) -> Optional[Dict]:
    """计算单只股票的ATR Rope特征（不做选股过滤，SSOT）

    与 process_stock 的区别：
    - 不做 regime 过滤（空头也计算特征）
    - 不做 signal_type 过滤（无信号也返回特征）
    - 不做成交额过滤
    - 返回GBDT模型所需的全部22个特征字段

    Returns: 特征字典，数据不足返回None
    """
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
    if daily_df.empty or len(daily_df) < 60:
        return None

    cfg = ATRRopeConfig(regime_lookback=55)
    try:
        result_df = compute_atr_rope(daily_df, cfg)
    except Exception:
        return None

    if result_df.empty or len(result_df) < 2:
        return None

    last = result_df.iloc[-1]

    # DSA regime
    try:
        dsa_regime, dsa_trend_strength, dsa_bars, dsa_vwap = compute_dsa_regime(daily_df)
    except Exception:
        return None

    regime = int(dsa_regime.iloc[-1])
    trend_strength_val = float(dsa_trend_strength.iloc[-1])
    dsa_dir_bars_val = int(dsa_bars.iloc[-1])

    # DSA趋势区间内 ATR Rope dir 占比
    rope_dir1_pct, rope_dir_neg1_pct = compute_rope_dir_pct_in_dsa_trend(result_df, dsa_dir_bars_val)

    # 布林带统计（所有regime都计算，空头时为None）
    bollinger = compute_low_rope_dev_bollinger(result_df, dsa_dir_bars_val) if regime == 1 else {k: None for k in ['low_rope_dev_mean', 'low_rope_dev_std', 'low_rope_dev_upper', 'low_rope_dev_mid', 'low_rope_dev_lower', 'low_rope_dev_today', 'low_rope_signal']}
    vwap_bollinger = compute_low_vwap_dev_bollinger(daily_df, dsa_vwap, dsa_dir_bars_val) if regime == 1 else {k: None for k in ['low_vwap_dev_mean', 'low_vwap_dev_std', 'low_vwap_dev_upper', 'low_vwap_dev_mid', 'low_vwap_dev_lower', 'low_vwap_dev_today', 'low_vwap_signal']}

    # BBMACD
    bbmacd_df = compute_bbmacd(daily_df)
    bbmacd_event = detect_bbmacd_event(bbmacd_df)

    def safe_float(val, default=None):
        return float(val) if pd.notna(val) else default

    atr_rope_dir_val = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None
    rope_value = safe_float(last.get('atr_rope_rope'))

    # 信号类型计算（与 process_stock 一致）
    signal_type = None
    close_today = float(last['close'])
    prev = result_df.iloc[-2]

    if regime == 1:  # 多头
        if atr_rope_dir_val == 0:
            signal_type = '多头蓝色区间'
    elif regime == 0:  # 震荡
        if atr_rope_dir_val == 0:
            c_lo_today = safe_float(last.get('atr_rope_c_lo'))
            c_lo_prev = safe_float(prev.get('atr_rope_c_lo'))
            c_hi_today = safe_float(last.get('atr_rope_c_hi'))
            if c_lo_today is not None and c_hi_today is not None and c_hi_today > c_lo_today:
                range_mid = (c_hi_today + c_lo_today) / 2.0
                close_prev = float(prev['close'])
                cross_above_clo = (c_lo_prev is not None and
                                   close_today > c_lo_today and close_prev <= c_lo_prev)
                range_mid_prev = (safe_float(prev.get('atr_rope_c_hi')) + c_lo_prev) / 2.0 if c_lo_prev is not None and safe_float(prev.get('atr_rope_c_hi')) is not None else None
                cross_above_mid = (range_mid_prev is not None and
                                   close_today > range_mid and close_prev <= range_mid_prev)
                if cross_above_clo:
                    signal_type = '震荡上穿下轨'
                elif cross_above_mid:
                    signal_type = '震荡上穿中轨'

    # 震荡场景 或 多头+ATR Rope蓝色区间(dir=0)：使用活跃的蓝色箱体 c_lo/c_hi
    # 多头+ATR Rope绿色(dir=1)：c_lo/c_hi 是旧残留值，无有意义的上下边界
    if regime == 0 or (regime == 1 and atr_rope_dir_val == 0):
        lower_val = safe_float(last.get('atr_rope_c_lo'))
        upper_val = safe_float(last.get('atr_rope_c_hi'))
    else:
        lower_val = None
        upper_val = None

    return {
        'ts_code': ts_code,
        'regime': '多头' if regime == 1 else ('空头' if regime == -1 else '震荡'),
        'regime_value': regime,
        'signal_type': signal_type,
        'regime_strength': trend_strength_val,
        'rope_dev_pct': safe_float(last.get('factor_atr_rope_line_dev_pct')),
        'rope_dev_atr': safe_float(last.get('factor_atr_rope_line_dev_atr')),
        'range_width_pct': safe_float(last.get('factor_atr_rope_range_width_pct')) * 100 if atr_rope_dir_val == 0 and safe_float(last.get('factor_atr_rope_range_width_pct')) is not None else None,
        'lower_value': lower_val,
        'upper_value': upper_val,
        'change_pct': compute_change_pct(daily_df),
        'vol_zscore': volume_zscore(daily_df['volume'], win=20),
        'bbmacd_event': bbmacd_event,
        'avg_amount_20d': float(((daily_df['open'] + daily_df['close']) / 2 * daily_df['volume']).tail(20).mean()) / 1e8 if len(daily_df) >= 20 else None,
        'dsa_dir_bars': dsa_dir_bars_val,
        'rope_dir1_pct': rope_dir1_pct,
        'rope_dir_neg1_pct': rope_dir_neg1_pct,
        'range_pos_01': safe_float(last.get('factor_atr_rope_range_pos_01')) if atr_rope_dir_val == 0 else None,
        'dsa_vwap_dev_pct': round(float(last['close'] / dsa_vwap.iloc[-1] - 1) * 100, 4) if pd.notna(dsa_vwap.iloc[-1]) and dsa_vwap.iloc[-1] != 0 else None,
        'low_rope_dev_mean': bollinger['low_rope_dev_mean'],
        'low_rope_dev_std': bollinger['low_rope_dev_std'],
        'low_rope_dev_today': bollinger['low_rope_dev_today'],
        'low_rope_signal': bollinger['low_rope_signal'],
        'low_vwap_dev_mean': vwap_bollinger['low_vwap_dev_mean'],
        'low_vwap_dev_std': vwap_bollinger['low_vwap_dev_std'],
        'low_vwap_dev_today': vwap_bollinger['low_vwap_dev_today'],
        'low_vwap_signal': vwap_bollinger['low_vwap_signal'],
        # 以下字段供 compute_derived_fields 使用
        'rope_value': rope_value,
        'rope_dir': atr_rope_dir_val,
        'signal_date': last.get('bar_time'),
    }


def process_stock(ts_code: str, selection_date: date) -> Optional[Dict]:
    """
    处理单只股票的 ATR Rope 选股逻辑

    选股逻辑：
        - 调用 compute_atr_rope 计算 ATR Rope 指标
        - 判断大背景（regime）：多头/空头/震荡
        - 多头场景：close 上穿 rope 趋势线 + close > rope → 买入信号
        - 震荡场景：close 上穿下轨 或 上穿中轨 → 买入信号
        - 空头场景：不触发买入
        - 过滤：过去5天平均成交额 >= 1亿

    Returns: 信号字典，如果满足条件则返回结果，否则返回None
    """
    # 获取日线数据
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
    if daily_df.empty or len(daily_df) < 60:
        return None

    # 调用 SSOT 核心计算（只引用，不重复实现）
    cfg = ATRRopeConfig(regime_lookback=55)  # 默认：length=14, multi=1.5, regime_lookback=20, regime_threshold=0.55
    try:
        result_df = compute_atr_rope(daily_df, cfg)
    except Exception:
        return None

    if result_df.empty or len(result_df) < 2:
        return None

    # 取最后一天数据
    last = result_df.iloc[-1]
    prev = result_df.iloc[-2]

    # 大背景判断：DSA VWAP dir 持续 bar 数
    try:
        dsa_regime, dsa_trend_strength, dsa_bars, dsa_vwap = compute_dsa_regime(daily_df)
    except Exception:
        return None
    regime = int(dsa_regime.iloc[-1])
    trend_strength_val = float(dsa_trend_strength.iloc[-1])
    dsa_dir_bars_val = int(dsa_bars.iloc[-1])

    # DSA趋势区间内 ATR Rope dir 占比
    rope_dir1_pct, rope_dir_neg1_pct = compute_rope_dir_pct_in_dsa_trend(result_df, dsa_dir_bars_val)

    # 多头趋势下：蓝色箱体内 low-rope 偏离布林带统计
    bollinger = compute_low_rope_dev_bollinger(result_df, dsa_dir_bars_val) if regime == 1 else {k: None for k in ['low_rope_dev_mean', 'low_rope_dev_std', 'low_rope_dev_upper', 'low_rope_dev_mid', 'low_rope_dev_lower', 'low_rope_dev_today', 'low_rope_signal']}

    # 多头趋势下：所有bar的 low-vwap 偏离布林带统计
    vwap_bollinger = compute_low_vwap_dev_bollinger(daily_df, dsa_vwap, dsa_dir_bars_val) if regime == 1 else {k: None for k in ['low_vwap_dev_mean', 'low_vwap_dev_std', 'low_vwap_dev_upper', 'low_vwap_dev_mid', 'low_vwap_dev_lower', 'low_vwap_dev_today', 'low_vwap_signal']}

    # 空头不选
    if regime == -1:
        return None

    # 判断买入触发
    signal_type = None
    close_today = float(last['close'])

    if regime == 1:  # 多头
        # 多头趋势下，ATR Rope dir=0 记录为蓝色区间事件
        atr_rope_dir_today = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None
        if atr_rope_dir_today == 0:
            signal_type = '多头蓝色区间'

    elif regime == 0:  # 震荡
        # 震荡信号仅在 ATR Rope dir=0（活跃蓝色箱体）时有效
        # dir≠0 时 c_lo/c_hi 是旧蓝色区间残留值，不触发
        atr_rope_dir_today = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None
        if atr_rope_dir_today == 0:
            # 震荡场景使用蓝色箱体的 c_lo/c_hi 作为下轨/上轨
            c_lo_today = float(last['atr_rope_c_lo']) if pd.notna(last.get('atr_rope_c_lo')) else None
            c_lo_prev = float(prev['atr_rope_c_lo']) if pd.notna(prev.get('atr_rope_c_lo')) else None
            c_hi_today = float(last['atr_rope_c_hi']) if pd.notna(last.get('atr_rope_c_hi')) else None

            # 蓝色箱体必须有效
            if c_lo_today is not None and c_hi_today is not None and c_hi_today > c_lo_today:
                range_mid = (c_hi_today + c_lo_today) / 2.0
                close_prev = float(prev['close'])

                # 上穿下轨：当日 close > c_lo 且 前一日 close <= 前一日 c_lo
                cross_above_clo = False
                if c_lo_prev is not None:
                    cross_above_clo = (close_today > c_lo_today) and (close_prev <= c_lo_prev)

                # 上穿中轨：当日 close > range_mid 且 前一日 close <= range_mid
                range_mid_prev = (float(prev['atr_rope_c_hi']) + c_lo_prev) / 2.0 if c_lo_prev is not None and pd.notna(prev.get('atr_rope_c_hi')) else None
                cross_above_mid = False
                if range_mid_prev is not None:
                    cross_above_mid = (close_today > range_mid) and (close_prev <= range_mid_prev)

                if cross_above_clo:
                    signal_type = '震荡上穿下轨'
                elif cross_above_mid:
                    signal_type = '震荡上穿中轨'

    if signal_type is None:
        return None

    # 成交额过滤
    if not check_volume_filter(daily_df, days=5, min_amount=100_000_000):
        return None

    # 计算观察项
    bbmacd_df = compute_bbmacd(daily_df)
    bbmacd_event = detect_bbmacd_event(bbmacd_df)

    bar_time = daily_df.index[-1]

    # 提取字段值（安全取值）
    def safe_float(val, default=None):
        return float(val) if pd.notna(val) else default

    # 震荡场景 或 多头+ATR Rope蓝色区间(dir=0)：使用活跃的蓝色箱体 c_lo/c_hi
    # 多头+ATR Rope绿色(dir=1)：c_lo/c_hi 是旧残留值，无有意义的上下边界
    atr_rope_dir_val = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None
    if regime == 0 or (regime == 1 and atr_rope_dir_val == 0):
        lower_val = safe_float(last.get('atr_rope_c_lo'))
        upper_val = safe_float(last.get('atr_rope_c_hi'))
    else:
        lower_val = None
        upper_val = None

    return {
        'ts_code': ts_code,
        # 大背景
        'regime': '多头' if regime == 1 else '震荡',
        'regime_value': regime,
        'regime_strength': trend_strength_val,
        # 买入信号
        'signal_type': signal_type,
        'rope_value': safe_float(last.get('atr_rope_rope')),
        'lower_value': lower_val,
        'upper_value': upper_val,
        # ATR Rope 状态
        'rope_dir': int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None,
        'rope_dev_pct': safe_float(last.get('factor_atr_rope_line_dev_pct')),
        'rope_dev_atr': safe_float(last.get('factor_atr_rope_line_dev_atr')),
        'range_width_pct': safe_float(last.get('factor_atr_rope_range_width_pct')) * 100 if atr_rope_dir_val == 0 and safe_float(last.get('factor_atr_rope_range_width_pct')) is not None else None,
        # 观察项
        'change_pct': compute_change_pct(daily_df),
        'vol_zscore': volume_zscore(daily_df['volume'], win=20),
        'bbmacd_event': bbmacd_event,
        'avg_amount_20d': float(((daily_df['open'] + daily_df['close']) / 2 * daily_df['volume']).tail(20).mean()) / 1e8 if len(daily_df) >= 20 else None,
        'dsa_dir_bars': dsa_dir_bars_val,
        'rope_dir1_pct': rope_dir1_pct,
        'rope_dir_neg1_pct': rope_dir_neg1_pct,
        'range_pos_01': safe_float(last.get('factor_atr_rope_range_pos_01')) if atr_rope_dir_val == 0 else None,
        # 多头布林带统计
        'low_rope_dev_mean': bollinger['low_rope_dev_mean'],
        'low_rope_dev_std': bollinger['low_rope_dev_std'],
        'low_rope_dev_upper': bollinger['low_rope_dev_upper'],
        'low_rope_dev_mid': bollinger['low_rope_dev_mid'],
        'low_rope_dev_lower': bollinger['low_rope_dev_lower'],
        'low_rope_dev_today': bollinger['low_rope_dev_today'],
        'low_rope_signal': bollinger['low_rope_signal'],
        # 多头 low-vwap 布林带统计
        'low_vwap_dev_mean': vwap_bollinger['low_vwap_dev_mean'],
        'low_vwap_dev_std': vwap_bollinger['low_vwap_dev_std'],
        'low_vwap_dev_upper': vwap_bollinger['low_vwap_dev_upper'],
        'low_vwap_dev_mid': vwap_bollinger['low_vwap_dev_mid'],
        'low_vwap_dev_lower': vwap_bollinger['low_vwap_dev_lower'],
        'low_vwap_dev_today': vwap_bollinger['low_vwap_dev_today'],
        'low_vwap_signal': vwap_bollinger['low_vwap_signal'],
        # 收盘价与DSA VWAP偏离率
        'dsa_vwap_dev_pct': round(float(last['close'] / dsa_vwap.iloc[-1] - 1) * 100, 4) if pd.notna(dsa_vwap.iloc[-1]) and dsa_vwap.iloc[-1] != 0 else None,
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
    """确保 atr_rope_selection 表存在"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS atr_rope_selection (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        signal_date DATE,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),

        -- 大背景
        regime VARCHAR(10) NOT NULL,
        regime_value INT,
        regime_strength FLOAT,

        -- 买入信号
        signal_type VARCHAR(20) NOT NULL,
        rope_value FLOAT,
        lower_value FLOAT,
        upper_value FLOAT,

        -- ATR Rope 状态
        rope_dir INT,
        rope_dev_pct FLOAT,
        rope_dev_atr FLOAT,
        range_width_pct FLOAT,

        -- 观察项
        change_pct FLOAT,
        vol_zscore FLOAT,
        bbmacd_event VARCHAR(20),
        avg_amount_20d FLOAT,
        dsa_dir_bars INT,
        rope_dir1_pct FLOAT,
        rope_dir_neg1_pct FLOAT,
        range_pos_01 FLOAT,
        low_rope_dev_mean FLOAT,
        low_rope_dev_std FLOAT,
        low_rope_dev_upper FLOAT,
        low_rope_dev_mid FLOAT,
        low_rope_dev_lower FLOAT,
        low_rope_dev_today FLOAT,
        low_rope_signal VARCHAR(20),
        low_vwap_dev_mean FLOAT,
        low_vwap_dev_std FLOAT,
        low_vwap_dev_upper FLOAT,
        low_vwap_dev_mid FLOAT,
        low_vwap_dev_lower FLOAT,
        low_vwap_dev_today FLOAT,
        low_vwap_signal VARCHAR(20),
        dsa_vwap_dev_pct FLOAT,

        batch_no INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_ar_selection_date ON atr_rope_selection(selection_date);
    CREATE INDEX IF NOT EXISTS idx_ar_ts_code ON atr_rope_selection(ts_code);
    CREATE INDEX IF NOT EXISTS idx_ar_regime ON atr_rope_selection(regime);
    CREATE INDEX IF NOT EXISTS idx_ar_signal_type ON atr_rope_selection(signal_type);
    CREATE INDEX IF NOT EXISTS idx_ar_batch_no ON atr_rope_selection(batch_no);

    -- 增量加列（兼容已有表）
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS avg_amount_20d FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS dsa_dir_bars INT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS range_width_pct FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS rope_dir1_pct FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS rope_dir_neg1_pct FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS range_pos_01 FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_rope_dev_mean FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_rope_dev_std FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_rope_dev_upper FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_rope_dev_mid FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_rope_dev_lower FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_rope_dev_today FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_rope_signal VARCHAR(20);
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_vwap_dev_mean FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_vwap_dev_std FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_vwap_dev_upper FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_vwap_dev_mid FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_vwap_dev_lower FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_vwap_dev_today FLOAT;
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS low_vwap_signal VARCHAR(20);
    ALTER TABLE atr_rope_selection ADD COLUMN IF NOT EXISTS dsa_vwap_dev_pct FLOAT;
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
            'regime': row.get('regime', ''),
            'regime_value': int(row['regime_value']) if pd.notna(row.get('regime_value')) else None,
            'regime_strength': float(row['regime_strength']) if pd.notna(row.get('regime_strength')) else None,
            'signal_type': row.get('signal_type', ''),
            'rope_value': float(row['rope_value']) if pd.notna(row.get('rope_value')) else None,
            'lower_value': float(row['lower_value']) if pd.notna(row.get('lower_value')) else None,
            'upper_value': float(row['upper_value']) if pd.notna(row.get('upper_value')) else None,
            'rope_dir': int(row['rope_dir']) if pd.notna(row.get('rope_dir')) else None,
            'rope_dev_pct': float(row['rope_dev_pct']) if pd.notna(row.get('rope_dev_pct')) else None,
            'rope_dev_atr': float(row['rope_dev_atr']) if pd.notna(row.get('rope_dev_atr')) else None,
            'change_pct': float(row['change_pct']) if pd.notna(row.get('change_pct')) else None,
            'vol_zscore': float(row['vol_zscore']) if pd.notna(row.get('vol_zscore')) else None,
            'bbmacd_event': row.get('bbmacd_event', '无') or '无',
            'avg_amount_20d': float(row['avg_amount_20d']) if pd.notna(row.get('avg_amount_20d')) else None,
            'dsa_dir_bars': int(row['dsa_dir_bars']) if pd.notna(row.get('dsa_dir_bars')) else None,
            'range_width_pct': float(row['range_width_pct']) if pd.notna(row.get('range_width_pct')) else None,
            'rope_dir1_pct': float(row['rope_dir1_pct']) if pd.notna(row.get('rope_dir1_pct')) else None,
            'rope_dir_neg1_pct': float(row['rope_dir_neg1_pct']) if pd.notna(row.get('rope_dir_neg1_pct')) else None,
            'range_pos_01': float(row['range_pos_01']) if pd.notna(row.get('range_pos_01')) else None,
            'low_rope_dev_mean': float(row['low_rope_dev_mean']) if pd.notna(row.get('low_rope_dev_mean')) else None,
            'low_rope_dev_std': float(row['low_rope_dev_std']) if pd.notna(row.get('low_rope_dev_std')) else None,
            'low_rope_dev_upper': float(row['low_rope_dev_upper']) if pd.notna(row.get('low_rope_dev_upper')) else None,
            'low_rope_dev_mid': float(row['low_rope_dev_mid']) if pd.notna(row.get('low_rope_dev_mid')) else None,
            'low_rope_dev_lower': float(row['low_rope_dev_lower']) if pd.notna(row.get('low_rope_dev_lower')) else None,
            'low_rope_dev_today': float(row['low_rope_dev_today']) if pd.notna(row.get('low_rope_dev_today')) else None,
            'low_rope_signal': row.get('low_rope_signal') or None,
            'low_vwap_dev_mean': float(row['low_vwap_dev_mean']) if pd.notna(row.get('low_vwap_dev_mean')) else None,
            'low_vwap_dev_std': float(row['low_vwap_dev_std']) if pd.notna(row.get('low_vwap_dev_std')) else None,
            'low_vwap_dev_upper': float(row['low_vwap_dev_upper']) if pd.notna(row.get('low_vwap_dev_upper')) else None,
            'low_vwap_dev_mid': float(row['low_vwap_dev_mid']) if pd.notna(row.get('low_vwap_dev_mid')) else None,
            'low_vwap_dev_lower': float(row['low_vwap_dev_lower']) if pd.notna(row.get('low_vwap_dev_lower')) else None,
            'low_vwap_dev_today': float(row['low_vwap_dev_today']) if pd.notna(row.get('low_vwap_dev_today')) else None,
            'low_vwap_signal': row.get('low_vwap_signal') or None,
            'dsa_vwap_dev_pct': float(row['dsa_vwap_dev_pct']) if pd.notna(row.get('dsa_vwap_dev_pct')) else None,
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


def select_atr_rope_stocks(selection_date: Optional[date] = None, save_to_db: bool = True) -> pd.DataFrame:
    """
    根据 ATR Rope 指标选出满足条件的股票

    Args:
        selection_date: 选股日期，默认为当天
        save_to_db: 是否保存到数据库
    """
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("选股条件（ATR Rope 策略）：")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  多头场景: close 上穿 rope 趋势线 + close > rope")
    print(f"  震荡场景: close 上穿下轨 或 上穿中轨")
    print(f"  空头场景: 不触发买入")
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
    print("开始 ATR Rope 指标筛选...")
    print(f"  原股票数: {len(stock_list)}")

    # 统计大背景分布
    regime_stats = {'多头': 0, '震荡': 0, '空头': 0, '无数据': 0}
    filtered_results = []

    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="ATR Rope选股", unit="只"):
        ts_code = row['ts_code']

        # 先快速获取数据判断 regime（避免对空头股票做完整计算）
        daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
        if daily_df.empty or len(daily_df) < 60:
            regime_stats['无数据'] += 1
            continue

        try:
            cfg = ATRRopeConfig(regime_lookback=55)
            result_df = compute_atr_rope(daily_df, cfg)
        except Exception:
            regime_stats['无数据'] += 1
            continue

        if result_df.empty or len(result_df) < 2:
            regime_stats['无数据'] += 1
            continue

        last = result_df.iloc[-1]

        # 大背景判断：DSA VWAP dir 持续 bar 数
        try:
            dsa_regime, dsa_trend_strength, dsa_bars, dsa_vwap = compute_dsa_regime(daily_df)
        except Exception:
            regime_stats['无数据'] += 1
            continue
        regime = int(dsa_regime.iloc[-1])
        trend_strength_val = float(dsa_trend_strength.iloc[-1])
        dsa_dir_bars_val = int(dsa_bars.iloc[-1])

        # DSA趋势区间内 ATR Rope dir 占比
        rope_dir1_pct, rope_dir_neg1_pct = compute_rope_dir_pct_in_dsa_trend(result_df, dsa_dir_bars_val)

        # 多头趋势下：蓝色箱体内 low-rope 偏离布林带统计
        bollinger = compute_low_rope_dev_bollinger(result_df, dsa_dir_bars_val) if regime == 1 else {k: None for k in ['low_rope_dev_mean', 'low_rope_dev_std', 'low_rope_dev_upper', 'low_rope_dev_mid', 'low_rope_dev_lower', 'low_rope_dev_today', 'low_rope_signal']}

        # 多头趋势下：所有bar的 low-vwap 偏离布林带统计
        vwap_bollinger = compute_low_vwap_dev_bollinger(daily_df, dsa_vwap, dsa_dir_bars_val) if regime == 1 else {k: None for k in ['low_vwap_dev_mean', 'low_vwap_dev_std', 'low_vwap_dev_upper', 'low_vwap_dev_mid', 'low_vwap_dev_lower', 'low_vwap_dev_today', 'low_vwap_signal']}

        if regime == 1:
            regime_stats['多头'] += 1
        elif regime == -1:
            regime_stats['空头'] += 1
        else:
            regime_stats['震荡'] += 1

        # 空头跳过
        if regime == -1:
            continue

        # 判断买入触发
        signal_type = None
        close_today = float(last['close'])

        if regime == 1:  # 多头
            # 多头趋势下，ATR Rope dir=0 记录为蓝色区间事件
            atr_rope_dir_today = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None
            if atr_rope_dir_today == 0:
                signal_type = '多头蓝色区间'

        elif regime == 0:  # 震荡
            # 震荡信号仅在 ATR Rope dir=0（活跃蓝色箱体）时有效
            # dir≠0 时 c_lo/c_hi 是旧蓝色区间残留值，不触发
            atr_rope_dir_today = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None
            if atr_rope_dir_today == 0:
                prev = result_df.iloc[-2]
                # 震荡场景使用蓝色箱体的 c_lo/c_hi 作为下轨/上轨
                c_lo_today = float(last['atr_rope_c_lo']) if pd.notna(last.get('atr_rope_c_lo')) else None
                c_lo_prev = float(prev['atr_rope_c_lo']) if pd.notna(prev.get('atr_rope_c_lo')) else None
                c_hi_today = float(last['atr_rope_c_hi']) if pd.notna(last.get('atr_rope_c_hi')) else None

                if c_lo_today is not None and c_hi_today is not None and c_hi_today > c_lo_today:
                    range_mid = (c_hi_today + c_lo_today) / 2.0
                    close_prev = float(prev['close'])

                    cross_above_clo = False
                    if c_lo_prev is not None:
                        cross_above_clo = (close_today > c_lo_today) and (close_prev <= c_lo_prev)

                    range_mid_prev = (float(prev['atr_rope_c_hi']) + c_lo_prev) / 2.0 if c_lo_prev is not None and pd.notna(prev.get('atr_rope_c_hi')) else None
                    cross_above_mid = False
                    if range_mid_prev is not None:
                        cross_above_mid = (close_today > range_mid) and (close_prev <= range_mid_prev)

                    if cross_above_clo:
                        signal_type = '震荡上穿下轨'
                    elif cross_above_mid:
                        signal_type = '震荡上穿中轨'

        if signal_type is None:
            continue

        # 成交额过滤
        if not check_volume_filter(daily_df, days=5, min_amount=100_000_000):
            continue

        # 计算观察项
        bbmacd_df = compute_bbmacd(daily_df)
        bbmacd_event = detect_bbmacd_event(bbmacd_df)

        bar_time = daily_df.index[-1]

        def safe_float(val, default=None):
            return float(val) if pd.notna(val) else default

        # 震荡场景 或 多头+ATR Rope蓝色区间(dir=0)：使用活跃的蓝色箱体 c_lo/c_hi
        # 多头+ATR Rope绿色(dir=1)：c_lo/c_hi 是旧残留值，无有意义的上下边界
        atr_rope_dir_val = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None
        if regime == 0 or (regime == 1 and atr_rope_dir_val == 0):
            lower_val = safe_float(last.get('atr_rope_c_lo'))
            upper_val = safe_float(last.get('atr_rope_c_hi'))
        else:
            lower_val = None
            upper_val = None

        filtered_results.append({
            'ts_code': ts_code,
            'regime': '多头' if regime == 1 else '震荡',
            'regime_value': regime,
            'regime_strength': trend_strength_val,
            'signal_type': signal_type,
            'rope_value': safe_float(last.get('atr_rope_rope')),
            'lower_value': lower_val,
            'upper_value': upper_val,
            'rope_dir': int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None,
            'rope_dev_pct': safe_float(last.get('factor_atr_rope_line_dev_pct')),
            'rope_dev_atr': safe_float(last.get('factor_atr_rope_line_dev_atr')),
            'range_width_pct': safe_float(last.get('factor_atr_rope_range_width_pct')) * 100 if atr_rope_dir_val == 0 and safe_float(last.get('factor_atr_rope_range_width_pct')) is not None else None,
            'change_pct': compute_change_pct(daily_df),
            'vol_zscore': volume_zscore(daily_df['volume'], win=20),
            'bbmacd_event': bbmacd_event,
            'avg_amount_20d': float(((daily_df['open'] + daily_df['close']) / 2 * daily_df['volume']).tail(20).mean()) / 1e8 if len(daily_df) >= 20 else None,
            'dsa_dir_bars': dsa_dir_bars_val,
            'rope_dir1_pct': rope_dir1_pct,
            'rope_dir_neg1_pct': rope_dir_neg1_pct,
            'range_pos_01': safe_float(last.get('factor_atr_rope_range_pos_01')) if atr_rope_dir_val == 0 else None,
            # 多头布林带统计
            'low_rope_dev_mean': bollinger['low_rope_dev_mean'],
            'low_rope_dev_std': bollinger['low_rope_dev_std'],
            'low_rope_dev_upper': bollinger['low_rope_dev_upper'],
            'low_rope_dev_mid': bollinger['low_rope_dev_mid'],
            'low_rope_dev_lower': bollinger['low_rope_dev_lower'],
            'low_rope_dev_today': bollinger['low_rope_dev_today'],
            'low_rope_signal': bollinger['low_rope_signal'],
            # 多头 low-vwap 布林带统计
            'low_vwap_dev_mean': vwap_bollinger['low_vwap_dev_mean'],
            'low_vwap_dev_std': vwap_bollinger['low_vwap_dev_std'],
            'low_vwap_dev_upper': vwap_bollinger['low_vwap_dev_upper'],
            'low_vwap_dev_mid': vwap_bollinger['low_vwap_dev_mid'],
            'low_vwap_dev_lower': vwap_bollinger['low_vwap_dev_lower'],
            'low_vwap_dev_today': vwap_bollinger['low_vwap_dev_today'],
            'low_vwap_signal': vwap_bollinger['low_vwap_signal'],
            # 收盘价与DSA VWAP偏离率
            'dsa_vwap_dev_pct': round(float(last['close'] / dsa_vwap.iloc[-1] - 1) * 100, 4) if pd.notna(dsa_vwap.iloc[-1]) and dsa_vwap.iloc[-1] != 0 else None,
            'signal_date': bar_time,
        })

    result_df_out = pd.DataFrame(filtered_results)

    if not result_df_out.empty:
        # 获取股票名称
        stock_names = batch_get_stock_names(result_df_out['ts_code'].tolist())
        result_df_out['stock_name'] = result_df_out['ts_code'].map(stock_names)

        # 分配批次号
        result_df_out['batch_no'] = (result_df_out.index // 10) + 1

    # 打印大背景分布
    print("\n" + "=" * 80)
    print("大背景分布：")
    print("=" * 80)
    total_with_data = sum(regime_stats.values()) - regime_stats['无数据']
    for k, v in regime_stats.items():
        pct = v / total_with_data * 100 if total_with_data > 0 and k != '无数据' else 0
        if k != '无数据':
            print(f"  {k}: {v} 只 ({pct:.1f}%)")
        else:
            print(f"  {k}: {v} 只")

    print("\n" + "=" * 80)
    print("选股结果汇总：")
    print("=" * 80)
    print(f"ATR Rope 筛选后: {len(result_df_out)} 只")

    if not result_df_out.empty:
        print(f"\n信号类型统计：")
        for st, cnt in result_df_out['signal_type'].value_counts().items():
            print(f"  {st}: {cnt} 只")

        print(f"\n大背景统计：")
        for rg, cnt in result_df_out['regime'].value_counts().items():
            print(f"  {rg}: {cnt} 只")

        batch_count = result_df_out['batch_no'].max()
        print(f"\n批次信息：共 {batch_count} 批，每批10只股票")

        print("\n" + "=" * 80)
        print("前20名股票：")
        print("=" * 80)
        display_cols = [
            'ts_code', 'stock_name', 'regime', 'signal_type',
            'rope_value', 'lower_value', 'regime_strength', 'change_pct'
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
        print("\n该股票不满足选股条件（大背景为空头 或 无买入信号 或 成交额不足）")

    return result


def backfill_stock_events(ts_code: str, start_date: date, end_date: date) -> List[Dict]:
    """
    遍历单只股票在日期区间内的所有交易日，记录触发事件。

    优化：只调用一次 compute_atr_rope 计算全量数据，然后遍历结果行判断信号，
    避免对每个交易日重复计算。

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

    # 确保 index 为 DatetimeIndex
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df.index = pd.to_datetime(daily_df.index)

    # 一次计算全量 ATR Rope（核心优化：不再逐日重复计算）
    cfg = ATRRopeConfig(regime_lookback=55)
    try:
        result_df = compute_atr_rope(daily_df, cfg)
    except Exception:
        return []

    if result_df.empty or len(result_df) < 2:
        return []

    # 一次计算全量 DSA regime（大背景判断 + 趋势强度 + dir持续bar数）
    try:
        dsa_regime, dsa_trend_strength, dsa_bars, dsa_vwap = compute_dsa_regime(daily_df)
    except Exception:
        return []

    # 确保 result_df index 也是 DatetimeIndex
    if not isinstance(result_df.index, pd.DatetimeIndex):
        result_df.index = pd.to_datetime(result_df.index)

    # 过滤出区间内的交易日
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    mask = (result_df.index >= start_ts) & (result_df.index <= end_ts)
    trade_dates = result_df.loc[mask].index.tolist()
    if not trade_dates:
        return []

    results = []
    for trade_dt in trade_dates:
        # 直接从全量结果中取当前行和前一行
        loc = result_df.index.get_loc(trade_dt)
        if loc < 1:
            continue

        last = result_df.iloc[loc]
        prev = result_df.iloc[loc - 1]

        # 大背景判断：DSA VWAP dir 持续 bar 数
        # 使用预计算的 dsa_regime（在函数开头一次计算）
        if trade_dt not in dsa_regime.index:
            continue
        regime = int(dsa_regime.loc[trade_dt])
        trend_strength_val = float(dsa_trend_strength.loc[trade_dt])
        dsa_dir_bars_val = int(dsa_bars.loc[trade_dt])

        # DSA趋势区间内 ATR Rope dir 占比
        rope_dir1_pct, rope_dir_neg1_pct = compute_rope_dir_pct_in_dsa_trend(result_df, dsa_dir_bars_val)

        # 多头趋势下：蓝色箱体内 low-rope 偏离布林带统计
        bollinger = compute_low_rope_dev_bollinger(result_df, dsa_dir_bars_val) if regime == 1 else {k: None for k in ['low_rope_dev_mean', 'low_rope_dev_std', 'low_rope_dev_upper', 'low_rope_dev_mid', 'low_rope_dev_lower', 'low_rope_dev_today', 'low_rope_signal']}

        # 空头跳过
        if regime == -1:
            continue

        # 判断买入触发
        signal_type = None
        close_today = float(last['close'])

        if regime == 1:  # 多头
            # 多头趋势下，ATR Rope dir=0 记录为蓝色区间事件
            atr_rope_dir_today = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None
            if atr_rope_dir_today == 0:
                signal_type = '多头蓝色区间'

        elif regime == 0:  # 震荡
            # 震荡信号仅在 ATR Rope dir=0（活跃蓝色箱体）时有效
            # dir≠0 时 c_lo/c_hi 是旧蓝色区间残留值，不触发
            atr_rope_dir_today = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None
            if atr_rope_dir_today == 0:
                # 震荡场景使用蓝色箱体的 c_lo/c_hi 作为下轨/上轨
                c_lo_today = float(last['atr_rope_c_lo']) if pd.notna(last.get('atr_rope_c_lo')) else None
                c_lo_prev = float(prev['atr_rope_c_lo']) if pd.notna(prev.get('atr_rope_c_lo')) else None
                c_hi_today = float(last['atr_rope_c_hi']) if pd.notna(last.get('atr_rope_c_hi')) else None

                if c_lo_today is not None and c_hi_today is not None and c_hi_today > c_lo_today:
                    range_mid = (c_hi_today + c_lo_today) / 2.0
                    close_prev = float(prev['close'])

                    cross_above_clo = False
                    if c_lo_prev is not None:
                        cross_above_clo = (close_today > c_lo_today) and (close_prev <= c_lo_prev)

                    range_mid_prev = (float(prev['atr_rope_c_hi']) + c_lo_prev) / 2.0 if c_lo_prev is not None and pd.notna(prev.get('atr_rope_c_hi')) else None
                    cross_above_mid = False
                    if range_mid_prev is not None:
                        cross_above_mid = (close_today > range_mid) and (close_prev <= range_mid_prev)

                    if cross_above_clo:
                        signal_type = '震荡上穿下轨'
                    elif cross_above_mid:
                        signal_type = '震荡上穿中轨'

        if signal_type is None:
            continue

        # 成交额过滤（用截至当日的数据）
        sub_df = daily_df.loc[daily_df.index <= trade_dt]
        if not check_volume_filter(sub_df, days=5, min_amount=100_000_000):
            continue

        # 多头趋势下：所有bar的 low-vwap 偏离布林带统计（需要 sub_df）
        vwap_bollinger = compute_low_vwap_dev_bollinger(sub_df, dsa_vwap, dsa_dir_bars_val) if regime == 1 else {k: None for k in ['low_vwap_dev_mean', 'low_vwap_dev_std', 'low_vwap_dev_upper', 'low_vwap_dev_mid', 'low_vwap_dev_lower', 'low_vwap_dev_today', 'low_vwap_signal']}

        # 观察项
        bbmacd_df = compute_bbmacd(sub_df)
        bbmacd_event = detect_bbmacd_event(bbmacd_df)

        def safe_float(val, default=None):
            return float(val) if pd.notna(val) else default

        # 震荡场景 或 多头+ATR Rope蓝色区间(dir=0)：使用活跃的蓝色箱体 c_lo/c_hi
        # 多头+ATR Rope绿色(dir=1)：c_lo/c_hi 是旧残留值，无有意义的上下边界
        atr_rope_dir_val = int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None
        if regime == 0 or (regime == 1 and atr_rope_dir_val == 0):
            lower_val = safe_float(last.get('atr_rope_c_lo'))
            upper_val = safe_float(last.get('atr_rope_c_hi'))
        else:
            lower_val = None
            upper_val = None

        results.append({
            'ts_code': ts_code,
            'selection_date': trade_dt,
            'signal_date': trade_dt,
            'regime': '多头' if regime == 1 else '震荡',
            'regime_value': regime,
            'regime_strength': trend_strength_val,
            'signal_type': signal_type,
            'rope_value': safe_float(last.get('atr_rope_rope')),
            'lower_value': lower_val,
            'upper_value': upper_val,
            'rope_dir': int(last['atr_rope_dir']) if pd.notna(last.get('atr_rope_dir')) else None,
            'rope_dev_pct': safe_float(last.get('factor_atr_rope_line_dev_pct')),
            'rope_dev_atr': safe_float(last.get('factor_atr_rope_line_dev_atr')),
            'range_width_pct': safe_float(last.get('factor_atr_rope_range_width_pct')) * 100 if atr_rope_dir_val == 0 and safe_float(last.get('factor_atr_rope_range_width_pct')) is not None else None,
            'change_pct': compute_change_pct(sub_df),
            'vol_zscore': volume_zscore(sub_df['volume'], win=20),
            'bbmacd_event': bbmacd_event,
            'avg_amount_20d': float(((sub_df['open'] + sub_df['close']) / 2 * sub_df['volume']).tail(20).mean()) / 1e8 if len(sub_df) >= 20 else None,
            'dsa_dir_bars': dsa_dir_bars_val,
            'rope_dir1_pct': rope_dir1_pct,
            'rope_dir_neg1_pct': rope_dir_neg1_pct,
            'range_pos_01': safe_float(last.get('factor_atr_rope_range_pos_01')) if atr_rope_dir_val == 0 else None,
            # 多头布林带统计
            'low_rope_dev_mean': bollinger['low_rope_dev_mean'],
            'low_rope_dev_std': bollinger['low_rope_dev_std'],
            'low_rope_dev_upper': bollinger['low_rope_dev_upper'],
            'low_rope_dev_mid': bollinger['low_rope_dev_mid'],
            'low_rope_dev_lower': bollinger['low_rope_dev_lower'],
            'low_rope_dev_today': bollinger['low_rope_dev_today'],
            'low_rope_signal': bollinger['low_rope_signal'],
            # 多头 low-vwap 布林带统计
            'low_vwap_dev_mean': vwap_bollinger['low_vwap_dev_mean'],
            'low_vwap_dev_std': vwap_bollinger['low_vwap_dev_std'],
            'low_vwap_dev_upper': vwap_bollinger['low_vwap_dev_upper'],
            'low_vwap_dev_mid': vwap_bollinger['low_vwap_dev_mid'],
            'low_vwap_dev_lower': vwap_bollinger['low_vwap_dev_lower'],
            'low_vwap_dev_today': vwap_bollinger['low_vwap_dev_today'],
            'low_vwap_signal': vwap_bollinger['low_vwap_signal'],
            # 收盘价与DSA VWAP偏离率
            'dsa_vwap_dev_pct': round(float(last['close'] / dsa_vwap.loc[trade_dt] - 1) * 100, 4) if pd.notna(dsa_vwap.loc[trade_dt]) and dsa_vwap.loc[trade_dt] != 0 else None,
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
        # 先删除该日期该股票的数据
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
                'regime': rec.get('regime', ''),
                'regime_value': int(rec['regime_value']) if rec.get('regime_value') is not None else None,
                'regime_strength': float(rec['regime_strength']) if rec.get('regime_strength') is not None else None,
                'signal_type': rec.get('signal_type', ''),
                'rope_value': float(rec['rope_value']) if rec.get('rope_value') is not None else None,
                'lower_value': float(rec['lower_value']) if rec.get('lower_value') is not None else None,
                'upper_value': float(rec['upper_value']) if rec.get('upper_value') is not None else None,
                'rope_dir': int(rec['rope_dir']) if rec.get('rope_dir') is not None else None,
                'rope_dev_pct': float(rec['rope_dev_pct']) if rec.get('rope_dev_pct') is not None else None,
                'rope_dev_atr': float(rec['rope_dev_atr']) if rec.get('rope_dev_atr') is not None else None,
                'range_width_pct': float(rec['range_width_pct']) if rec.get('range_width_pct') is not None else None,
                'change_pct': float(rec['change_pct']) if rec.get('change_pct') is not None else None,
                'vol_zscore': float(rec['vol_zscore']) if rec.get('vol_zscore') is not None else None,
                'bbmacd_event': rec.get('bbmacd_event', '无') or '无',
                'avg_amount_20d': float(rec['avg_amount_20d']) if rec.get('avg_amount_20d') is not None else None,
                'dsa_dir_bars': int(rec['dsa_dir_bars']) if rec.get('dsa_dir_bars') is not None else None,
                'rope_dir1_pct': float(rec['rope_dir1_pct']) if rec.get('rope_dir1_pct') is not None else None,
                'rope_dir_neg1_pct': float(rec['rope_dir_neg1_pct']) if rec.get('rope_dir_neg1_pct') is not None else None,
                'range_pos_01': float(rec['range_pos_01']) if rec.get('range_pos_01') is not None else None,
                'low_rope_dev_mean': float(rec['low_rope_dev_mean']) if rec.get('low_rope_dev_mean') is not None else None,
                'low_rope_dev_std': float(rec['low_rope_dev_std']) if rec.get('low_rope_dev_std') is not None else None,
                'low_rope_dev_upper': float(rec['low_rope_dev_upper']) if rec.get('low_rope_dev_upper') is not None else None,
                'low_rope_dev_mid': float(rec['low_rope_dev_mid']) if rec.get('low_rope_dev_mid') is not None else None,
                'low_rope_dev_lower': float(rec['low_rope_dev_lower']) if rec.get('low_rope_dev_lower') is not None else None,
                'low_rope_dev_today': float(rec['low_rope_dev_today']) if rec.get('low_rope_dev_today') is not None else None,
                'low_rope_signal': rec.get('low_rope_signal') or None,
                'low_vwap_dev_mean': float(rec['low_vwap_dev_mean']) if rec.get('low_vwap_dev_mean') is not None else None,
                'low_vwap_dev_std': float(rec['low_vwap_dev_std']) if rec.get('low_vwap_dev_std') is not None else None,
                'low_vwap_dev_upper': float(rec['low_vwap_dev_upper']) if rec.get('low_vwap_dev_upper') is not None else None,
                'low_vwap_dev_mid': float(rec['low_vwap_dev_mid']) if rec.get('low_vwap_dev_mid') is not None else None,
                'low_vwap_dev_lower': float(rec['low_vwap_dev_lower']) if rec.get('low_vwap_dev_lower') is not None else None,
                'low_vwap_dev_today': float(rec['low_vwap_dev_today']) if rec.get('low_vwap_dev_today') is not None else None,
                'low_vwap_signal': rec.get('low_vwap_signal') or None,
                'dsa_vwap_dev_pct': float(rec['dsa_vwap_dev_pct']) if rec.get('dsa_vwap_dev_pct') is not None else None,
                'batch_no': None,
            })

        if insert_records:
            insert_df = pd.DataFrame(insert_records)
            insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)
            total_saved += len(insert_records)

    return total_saved


def backfill_range(start_date: date, end_date: date, stock_list: Optional[List[str]] = None):
    """
    回补指定日期区间内所有股票的 ATR Rope 事件。
    遍历个股，对每个交易日逐日计算，每处理完一只股票立即保存。

    Args:
        start_date: 起始日期（含）
        end_date: 结束日期（含）
        stock_list: 指定股票列表，None 则自动获取全市场
    """
    print("=" * 80)
    print("ATR Rope 事件回补")
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
        description='ATR Rope 选股工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/seletion_atr.py                    # 使用当天日期选股
  python selection/seletion_atr.py 2026-05-19         # 指定日期选股
  python selection/seletion_atr.py 20260519           # 指定日期选股（无分隔符）
  python selection/seletion_atr.py --test 600547      # 测试单只股票
  python selection/seletion_atr.py --backfill 2025-07-01 2026-04-30  # 回补历史事件
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
    print("ATR Rope 选股工具")
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print("=" * 80)

    # 测试模式
    if args.test:
        test_single_stock(args.test, selection_date)
        sys.exit(0)

    # 正常选股模式
    df = select_atr_rope_stocks(selection_date=selection_date, save_to_db=not args.no_save)

    print("\n" + "=" * 80)
    print("选股完成")
    print(f"选股日期: {selection_date}")
    print(f"选中股票数: {len(df)}")
    print(f"查询SQL: SELECT * FROM {SELECTION_TABLE} WHERE selection_date = '{selection_date}'")
    print("=" * 80)

    return df


if __name__ == '__main__':
    main()
