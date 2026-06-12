#!/usr/bin/env python3
"""
DSA 多头选股脚本（基于 DSA VWAP 偏离率与交叉事件）

Purpose:
    筛选日线 DSA VWAP dir=1 持续超过 50 bars 的多头趋势股票，
    计算 close 与 VWAP 的偏离率统计、VWAP 收益指标、VWAP 交叉事件。

Inputs:
    stock_k_data (日线K线数据，open/high/low/close/volume)

Outputs:
    dsa_selection (选股结果表)

How to Run:
    python selection/selection_dsa.py              # 当天
    python selection/selection_dsa.py 2026-05-19   # 指定日期
    python selection/selection_dsa.py --test 600547 # 测试单只
    python selection/selection_dsa.py --backfill 2025-07-01 2026-04-30  # 回补历史

Side Effects:
    写入 dsa_selection 表（幂等：同一日期先删后插）

================================================================================
【选股逻辑】

核心条件：DSA VWAP dir=1 持续 > 50 bars（多头趋势确认）

偏离率统计：
  - offset_rate = (close - VWAP) / VWAP
  - expanding window：当前 dir=1 区间内从起始 bar 到当前 bar 的全部 offset_rate
  - 计算 mean / std / percentile（正态分布 CDF）

VWAP 收益指标：
  - 全程平均每 bar 收益率
  - 近 5 / 10 / 20 bars 平均每 bar 收益率

VWAP 交叉事件（仅在 dir=1 区间内检测）：
  - 全部引用 features.dynamic_swing_anchored_vwap 计算DSA（SSOT）
  - 本脚本只做数据准备、条件判断、结果保存，不重复计算逻辑
  - 补充记录交叉时的 DSA Dir 指标数值

ATR Rope 交叉事件（全数据范围检测）：
  - close 上穿/下穿 ATR Rope 趋势线
  - 同步记录交叉时的 ATR Rope Dir 指标数值
  - 引用 features.atr_rope_event_factor_lab_v4 的 atr_rope_rope / atr_rope_dir 列（SSOT）

ATR Rope 统计（在 dir=1 趋势区间内）：
  - rope_dir1_pct / rope_dir0_pct / rope_dir_neg1_pct：ATR Rope dir=1/0/-1 占比(%)
  - touch_rope：当天 low 是否碰触 ATR Rope 趋势线（low <= atr_rope_rope）
  - touch_vwap：当天 low 是否碰触 DSA VWAP 趋势线（low <= dsa_vwap）

【选股日期】
  - 选股日期只是标记，实际数据到"选股日期当天或之前最后一个交易日"

【保存逻辑】
  - 按选股日期统一保存，先删旧数据再插新数据（幂等性）
================================================================================
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from datetime import datetime, date
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

from features.dynamic_swing_anchored_vwap import dynamic_swing_anchored_vwap, DSAConfig
from features.atr_rope_event_factor_lab_v4 import compute_atr_rope, ATRRopeConfig
from datasource.adj_factor import apply_adj_factor

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "dsa_selection"
DAILY_BARS = 800
MIN_DIR_BARS = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_ts_code(ts_code: str) -> str:
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
        suffix = '.SH' if symbol.startswith(('6', '9')) else '.SZ'
        actual_ts_code = f'{symbol}{suffix}'
        df = apply_adj_factor(df, actual_ts_code, freq='d')
    return df


def batch_get_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    if not ts_codes:
        return {}
    placeholders = ', '.join([f"'{c}'" for c in ts_codes])
    sql = text(f"SELECT ts_code, name FROM stock_pools WHERE ts_code IN ({placeholders})")
    with engine.connect() as conn:
        result = conn.execute(sql)
        return {row[0]: row[1] for row in result}


def remove_vwap_lookahead(daily_df: pd.DataFrame, vwap_series: pd.Series,
                          dir_series: pd.Series, cfg: DSAConfig = None) -> pd.Series:
    """
    消除 DSA VWAP 的前视偏差。

    原理：全量计算时，方向翻转点 T 处，vwap_out[anchor..T] 被新方向的
    回填递推覆盖。但 anchor 到 T-1 的 bar 在实时中不可能知道方向会翻转，
    这些 bar 的 VWAP 应该是旧方向的递推结果。

    解决：找到所有方向翻转点，对每个翻转点用截断到 T-1 的数据重算 DSA VWAP，
    用截断结果替换被覆盖的值（anchor 到 T-1 之间的 bar）。
    """
    dir_vals = dir_series.fillna(0).astype(int)
    flip_mask = dir_vals != dir_vals.shift(1)
    # 排除第一个 bar（无前一个方向可比较）
    flip_mask.iloc[0] = False
    flip_indices = daily_df.index[flip_mask].tolist()

    if not flip_indices:
        return vwap_series

    vwap_corrected = vwap_series.copy()

    for flip_idx in flip_indices:
        loc = daily_df.index.get_loc(flip_idx)
        if loc < 2:
            continue

        # 截断到翻转点前一个 bar（T-1），此时方向还没翻转
        truncated_df = daily_df.iloc[:loc]
        try:
            vwap_trunc, _, _, _ = dynamic_swing_anchored_vwap(truncated_df, cfg)
        except Exception:
            continue

        # 截断结果中每个 bar 的 VWAP 是该 bar 时刻的"实时值"（无前视偏差）
        # 用截断结果替换全量结果中被回填覆盖的值
        common_idx = vwap_trunc.index.intersection(vwap_corrected.index)
        for idx in common_idx:
            if pd.notna(vwap_trunc[idx]) and pd.notna(vwap_corrected[idx]):
                if abs(float(vwap_trunc[idx]) - float(vwap_corrected[idx])) > 0.001:
                    vwap_corrected[idx] = vwap_trunc[idx]

    return vwap_corrected


def compute_dsa_regime(df: pd.DataFrame, cfg: DSAConfig = None, min_bars: int = MIN_DIR_BARS):
    """
    基于 DSA VWAP dir 计算大背景 regime 和趋势强度：
    - dir=1 持续 > min_bars → 1（多头）
    - dir=-1 持续 > min_bars → -1（空头）
    - 其他 → 0（震荡）

    趋势强度 = 当前 dir 持续期间内 VWAP 平均每 bar 收益率

    Returns:
        (regime: pd.Series, trend_strength: pd.Series, dsa_bars: pd.Series, vwap_series: pd.Series, dir_series: pd.Series)
    """
    if cfg is None:
        cfg = DSAConfig()
    vwap_series, dir_series, _, _ = dynamic_swing_anchored_vwap(df, cfg)
    # 消除前视偏差：方向翻转时回填覆盖了过去 bar 的 VWAP 值
    vwap_series = remove_vwap_lookahead(df, vwap_series, dir_series, cfg)
    dir_vals = dir_series.fillna(0).astype(int)

    bars = pd.Series(0, index=df.index, dtype=int)
    for i in range(len(dir_vals)):
        if i == 0:
            bars.iloc[i] = dir_vals.iloc[i]
        elif dir_vals.iloc[i] == dir_vals.iloc[i - 1]:
            bars.iloc[i] = bars.iloc[i - 1] + dir_vals.iloc[i]
        else:
            bars.iloc[i] = dir_vals.iloc[i]

    regime = pd.Series(0, index=df.index, dtype=int)
    regime[bars > min_bars] = 1
    regime[bars < -min_bars] = -1

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

    return regime, trend_strength, bars, vwap_series, dir_series


def compute_rope_dir_pct_in_dsa_trend(result_df: pd.DataFrame, dsa_bars_val: int):
    """
    在DSA趋势区间内，统计ATR Rope dir=1/0/-1的占比(%)。

    Args:
        result_df: compute_atr_rope 返回的 DataFrame（含 atr_rope_dir 列）
        dsa_bars_val: 当前DSA趋势持续的bar数（正数=多头）

    Returns:
        (dir1_pct, dir0_pct, dir_neg1_pct) 百分比，如 (60.0, 25.0, 15.0)
    """
    n = abs(dsa_bars_val)
    if n <= 0 or len(result_df) < n:
        return None, None, None
    segment = result_df['atr_rope_dir'].iloc[-n:]
    total = len(segment)
    if total == 0:
        return None, None, None
    dir1_pct = round(float((segment == 1).sum()) / total * 100, 2)
    dir0_pct = round(float((segment == 0).sum()) / total * 100, 2)
    dir_neg1_pct = round(float((segment == -1).sum()) / total * 100, 2)
    return dir1_pct, dir0_pct, dir_neg1_pct


def compute_offset_rate_stats(
    close: pd.Series,
    vwap: pd.Series,
    dir_series: pd.Series,
    dsa_bars: pd.Series,
) -> pd.DataFrame:
    """
    在 dir=1 区间内计算 close 与 VWAP 的偏离率统计（expanding window）。

    对每根 bar：
    - offset_rate = (close - VWAP) / VWAP
    - expanding window = 当前 dir=1 区间内从起始 bar 到当前 bar 的全部 offset_rate
    - 计算 mean / std / percentile（正态分布 CDF）

    Returns:
        DataFrame with columns: offset_rate, offset_mean, offset_std, offset_percentile
    """
    n = len(close)
    offset_rate = np.full(n, np.nan)
    offset_mean = np.full(n, np.nan)
    offset_std = np.full(n, np.nan)
    offset_percentile = np.full(n, np.nan)

    close_arr = close.to_numpy(float)
    vwap_arr = vwap.to_numpy(float)
    dir_arr = dir_series.fillna(0).to_numpy(float)
    bars_arr = dsa_bars.to_numpy(float)

    for i in range(n):
        if dir_arr[i] != 1 or not np.isfinite(vwap_arr[i]) or vwap_arr[i] == 0:
            continue

        offset_rate[i] = (close_arr[i] - vwap_arr[i]) / vwap_arr[i]

        num_bars = int(abs(bars_arr[i]))
        if num_bars < 2 or not np.isfinite(bars_arr[i]):
            continue

        start_idx = i - num_bars + 1
        if start_idx < 0:
            start_idx = 0

        window_rates = offset_rate[start_idx:i + 1]
        valid_mask = np.isfinite(window_rates)
        valid_rates = window_rates[valid_mask]

        if len(valid_rates) < 2:
            continue

        mu = float(np.mean(valid_rates))
        sd = float(np.std(valid_rates, ddof=0))
        offset_mean[i] = mu
        offset_std[i] = sd

        if sd > 0 and np.isfinite(offset_rate[i]):
            offset_percentile[i] = float(sp_stats.norm.cdf(offset_rate[i], mu, sd))

    return pd.DataFrame({
        'offset_rate': offset_rate,
        'offset_mean': offset_mean,
        'offset_std': offset_std,
        'offset_percentile': offset_percentile,
    }, index=close.index)


def compute_vwap_return_metrics(
    vwap: pd.Series,
    dsa_bars: pd.Series,
) -> Dict[str, Optional[float]]:
    """
    计算 VWAP 收益指标（仅取最后一根 bar 的结果）。

    - vwap_ret_total: 整个 dir=1 持续期间 VWAP 平均每 bar 收益率
    - vwap_ret_5/10/20: 最近 5/10/20 bars 的 VWAP 平均每 bar 收益率

    Returns:
        dict with vwap_ret_total, vwap_ret_5, vwap_ret_10, vwap_ret_20
    """
    vwap_arr = vwap.to_numpy(float)
    bars_val = int(abs(dsa_bars.iloc[-1]))
    result = {
        'vwap_ret_total': None,
        'vwap_ret_5': None,
        'vwap_ret_10': None,
        'vwap_ret_20': None,
    }

    if bars_val < 2:
        return result

    start_idx = len(vwap_arr) - bars_val
    if start_idx < 0:
        start_idx = 0

    vwap_start = vwap_arr[start_idx]
    vwap_end = vwap_arr[-1]
    if pd.notna(vwap_start) and pd.notna(vwap_end) and vwap_start != 0:
        result['vwap_ret_total'] = round((vwap_end / vwap_start - 1) / bars_val, 6)

    for period, key in [(5, 'vwap_ret_5'), (10, 'vwap_ret_10'), (20, 'vwap_ret_20')]:
        if len(vwap_arr) < period + 1:
            continue
        v_start = vwap_arr[-(period + 1)]
        v_end = vwap_arr[-1]
        if pd.notna(v_start) and pd.notna(v_end) and v_start != 0:
            result[key] = round((v_end / v_start - 1) / period, 6)

    return result


def detect_vwap_crossover_events(
    close: pd.Series,
    vwap: pd.Series,
    dir_series: pd.Series,
    dsa_bars: pd.Series,
) -> Dict[str, Optional[object]]:
    """
    在当前 dir=1 区间内检测 close 与 VWAP 的交叉事件。

    仅统计最后一根 bar 所在的 dir=1 持续期间内的交叉，排除历史区间。

    上穿：close[t] > VWAP[t] 且 close[t-1] <= VWAP[t-1]
    下穿：close[t] < VWAP[t] 且 close[t-1] >= VWAP[t-1]

    Returns:
        dict with:
        - last_cross_up_date / last_cross_up_price: 最近一次上穿
        - last_cross_down_date / last_cross_down_price: 最近一次下穿
        - cross_up_count / cross_down_count: 当前dir=1期间交叉次数
        - last_cross_up_dsa_dir / last_cross_down_dsa_dir: 交叉时DSA Dir值
    """
    result = {
        'last_cross_up_date': None,
        'last_cross_up_price': None,
        'last_cross_down_date': None,
        'last_cross_down_price': None,
        'cross_up_count': 0,
        'cross_down_count': 0,
        'last_cross_up_dsa_dir': None,
        'last_cross_down_dsa_dir': None,
    }

    close_arr = close.to_numpy(float)
    vwap_arr = vwap.to_numpy(float)
    dir_arr = dir_series.fillna(0).to_numpy(float)
    bars_arr = dsa_bars.to_numpy(float)

    last_bars = int(abs(bars_arr[-1]))
    if last_bars < 2:
        return result

    start_idx = len(close_arr) - last_bars
    if start_idx < 1:
        start_idx = 1

    for i in range(start_idx, len(close_arr)):
        if dir_arr[i] != 1:
            continue
        if not (np.isfinite(close_arr[i]) and np.isfinite(close_arr[i - 1])
                and np.isfinite(vwap_arr[i]) and np.isfinite(vwap_arr[i - 1])):
            continue

        is_cross_up = close_arr[i] > vwap_arr[i] and close_arr[i - 1] <= vwap_arr[i - 1]
        is_cross_down = close_arr[i] < vwap_arr[i] and close_arr[i - 1] >= vwap_arr[i - 1]

        if is_cross_up:
            result['last_cross_up_date'] = close.index[i]
            result['last_cross_up_price'] = round(float(close_arr[i]), 4)
            result['last_cross_up_dsa_dir'] = int(dir_arr[i]) if np.isfinite(dir_arr[i]) else None
            result['cross_up_count'] += 1

        if is_cross_down:
            result['last_cross_down_date'] = close.index[i]
            result['last_cross_down_price'] = round(float(close_arr[i]), 4)
            result['last_cross_down_dsa_dir'] = int(dir_arr[i]) if np.isfinite(dir_arr[i]) else None
            result['cross_down_count'] += 1

    return result


def detect_rope_crossover_events(
    close: pd.Series,
    atr_rope_df: pd.DataFrame,
) -> Dict[str, Optional[object]]:
    """
    检测 close 与 ATR Rope 趋势线的交叉事件（全数据范围）。

    上穿：close[t] > rope[t] 且 close[t-1] <= rope[t-1]
    下穿：close[t] < rope[t] 且 close[t-1] >= rope[t-1]

    使用 features.atr_rope_event_factor_lab_v4 的 atr_rope_rope / atr_rope_dir 列（SSOT）。

    Returns:
        dict with:
        - last_rope_cross_up_date / last_rope_cross_up_price: 最近一次上穿日期/价格
        - last_rope_cross_up_rope / last_rope_cross_up_dir: 上穿时Rope值/Dir值
        - last_rope_cross_down_date / last_rope_cross_down_price: 最近一次下穿日期/价格
        - last_rope_cross_down_rope / last_rope_cross_down_dir: 下穿时Rope值/Dir值
        - rope_cross_up_count / rope_cross_down_count: 全数据范围交叉次数
    """
    result = {
        'last_rope_cross_up_date': None,
        'last_rope_cross_up_price': None,
        'last_rope_cross_up_rope': None,
        'last_rope_cross_up_dir': None,
        'last_rope_cross_down_date': None,
        'last_rope_cross_down_price': None,
        'last_rope_cross_down_rope': None,
        'last_rope_cross_down_dir': None,
        'rope_cross_up_count': 0,
        'rope_cross_down_count': 0,
    }

    if atr_rope_df is None or atr_rope_df.empty:
        return result
    if 'atr_rope_rope' not in atr_rope_df.columns or 'atr_rope_dir' not in atr_rope_df.columns:
        return result

    close_arr = close.to_numpy(float)
    rope_arr = atr_rope_df['atr_rope_rope'].to_numpy(float)
    dir_arr = atr_rope_df['atr_rope_dir'].to_numpy(float)

    if len(close_arr) != len(rope_arr):
        return result

    for i in range(1, len(close_arr)):
        if not (np.isfinite(close_arr[i]) and np.isfinite(close_arr[i - 1])
                and np.isfinite(rope_arr[i]) and np.isfinite(rope_arr[i - 1])):
            continue

        is_cross_up = close_arr[i] > rope_arr[i] and close_arr[i - 1] <= rope_arr[i - 1]
        is_cross_down = close_arr[i] < rope_arr[i] and close_arr[i - 1] >= rope_arr[i - 1]

        if is_cross_up:
            result['last_rope_cross_up_date'] = close.index[i]
            result['last_rope_cross_up_price'] = round(float(close_arr[i]), 4)
            result['last_rope_cross_up_rope'] = round(float(rope_arr[i]), 4)
            result['last_rope_cross_up_dir'] = int(dir_arr[i]) if np.isfinite(dir_arr[i]) else None
            result['rope_cross_up_count'] += 1

        if is_cross_down:
            result['last_rope_cross_down_date'] = close.index[i]
            result['last_rope_cross_down_price'] = round(float(close_arr[i]), 4)
            result['last_rope_cross_down_rope'] = round(float(rope_arr[i]), 4)
            result['last_rope_cross_down_dir'] = int(dir_arr[i]) if np.isfinite(dir_arr[i]) else None
            result['rope_cross_down_count'] += 1

    return result


def check_volume_filter(df: pd.DataFrame, days: int = 5, min_amount: float = 100_000_000) -> bool:
    if len(df) < days:
        return False
    recent_df = df.tail(days)
    daily_amount = recent_df['volume'] * recent_df['close']
    avg_amount = daily_amount.mean()
    return avg_amount >= min_amount


def volume_zscore(vol: pd.Series, win: int = 20) -> Optional[float]:
    mu = vol.rolling(win, min_periods=win).mean()
    sd = vol.rolling(win, min_periods=win).std(ddof=0)
    if len(vol) < win:
        return None
    mu_val = mu.iloc[-1]
    sd_val = sd.iloc[-1]
    if sd_val == 0 or pd.isna(sd_val):
        return None
    return float((vol.iloc[-1] - mu_val) / sd_val)


def compute_change_pct(daily_df: pd.DataFrame) -> Optional[float]:
    if len(daily_df) < 2:
        return None
    close_today = daily_df['close'].iloc[-1]
    close_yesterday = daily_df['close'].iloc[-2]
    if close_yesterday == 0:
        return None
    return round(float(close_today - close_yesterday) / float(close_yesterday) * 100, 2)


def process_stock(ts_code: str, selection_date: date) -> Optional[Dict]:
    """
    处理单只股票的 DSA 多头选股逻辑

    选股逻辑：
        - DSA VWAP dir=1 持续 > 50 bars → 多头
        - 计算偏离率统计、VWAP 收益、VWAP 交叉事件

    Returns: 信号字典，满足条件返回结果，否则返回None
    """
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
    if daily_df.empty or len(daily_df) < 60:
        return None

    try:
        dsa_regime, dsa_trend_strength, dsa_bars, dsa_vwap, dsa_dir_series = compute_dsa_regime(daily_df)
    except Exception as e:
        logger.debug(f"DSA计算异常 {ts_code}: {e}")
        return None

    regime = int(dsa_regime.iloc[-1])
    if regime != 1:
        return None

    trend_strength_val = float(dsa_trend_strength.iloc[-1])
    dsa_dir_bars_val = int(dsa_bars.iloc[-1])

    atr_rope_df = None
    try:
        cfg = ATRRopeConfig(regime_lookback=55)
        atr_rope_df = compute_atr_rope(daily_df, cfg)
    except Exception as e:
        logger.debug(f"ATR Rope计算异常 {ts_code}: {e}")

    rope_dir1_pct, rope_dir0_pct, rope_dir_neg1_pct = None, None, None
    touch_rope = False
    touch_vwap = False
    if atr_rope_df is not None and not atr_rope_df.empty:
        rope_dir1_pct, rope_dir0_pct, rope_dir_neg1_pct = compute_rope_dir_pct_in_dsa_trend(atr_rope_df, dsa_dir_bars_val)
        last_rope_val = atr_rope_df['atr_rope_rope'].iloc[-1]
        last_low = daily_df['low'].iloc[-1]
        if pd.notna(last_rope_val) and pd.notna(last_low):
            touch_rope = bool(last_low <= last_rope_val)
        if pd.notna(dsa_vwap.iloc[-1]) and pd.notna(last_low):
            touch_vwap = bool(last_low <= dsa_vwap.iloc[-1])

    offset_stats = compute_offset_rate_stats(
        daily_df['close'], dsa_vwap, dsa_dir_series, dsa_bars
    )
    last_offset = offset_stats.iloc[-1]

    vwap_metrics = compute_vwap_return_metrics(dsa_vwap, dsa_bars)

    crossover = detect_vwap_crossover_events(daily_df['close'], dsa_vwap, dsa_dir_series, dsa_bars)

    rope_crossover = detect_rope_crossover_events(daily_df['close'], atr_rope_df)

    bar_time = daily_df.index[-1]
    last_close = float(daily_df['close'].iloc[-1])
    last_vwap = float(dsa_vwap.iloc[-1]) if pd.notna(dsa_vwap.iloc[-1]) else None

    change_pct = compute_change_pct(daily_df)
    # 除权防御：前复权后日跌幅仍超过-15%，视为除权未修正的虚假跌幅，跳过
    if change_pct is not None and change_pct < -15:
        logger.debug(f"疑似除权虚假跌幅 {ts_code}: {change_pct:.2f}%，跳过")
        return None

    def safe_float(val, default=None):
        return float(val) if pd.notna(val) else default

    return {
        'ts_code': ts_code,
        'regime': '多头',
        'regime_value': regime,
        'regime_strength': trend_strength_val,
        'dsa_dir_bars': dsa_dir_bars_val,
        'offset_rate': safe_float(last_offset['offset_rate']),
        'offset_mean': safe_float(last_offset['offset_mean']),
        'offset_std': safe_float(last_offset['offset_std']),
        'offset_percentile': safe_float(last_offset['offset_percentile']),
        'vwap_ret_total': vwap_metrics['vwap_ret_total'],
        'vwap_ret_5': vwap_metrics['vwap_ret_5'],
        'vwap_ret_10': vwap_metrics['vwap_ret_10'],
        'vwap_ret_20': vwap_metrics['vwap_ret_20'],
        'last_cross_up_date': crossover['last_cross_up_date'],
        'last_cross_up_price': crossover['last_cross_up_price'],
        'last_cross_down_date': crossover['last_cross_down_date'],
        'last_cross_down_price': crossover['last_cross_down_price'],
        'cross_up_count': crossover['cross_up_count'],
        'cross_down_count': crossover['cross_down_count'],
        'change_pct': change_pct,
        'vol_zscore': volume_zscore(daily_df['volume'], win=20),
        'avg_amount_20d': float(((daily_df['open'] + daily_df['close']) / 2 * daily_df['volume']).tail(20).mean()) / 1e8 if len(daily_df) >= 20 else None,
        'dsa_vwap': last_vwap,
        'dsa_vwap_dev_pct': round((last_close / last_vwap - 1) * 100, 4) if last_vwap and last_vwap != 0 else None,
        'rope_dir1_pct': rope_dir1_pct,
        'rope_dir0_pct': rope_dir0_pct,
        'rope_dir_neg1_pct': rope_dir_neg1_pct,
        'touch_rope': touch_rope,
        'touch_vwap': touch_vwap,
        'signal_date': bar_time,
        'rope_cross_up_date': rope_crossover['last_rope_cross_up_date'],
        'rope_cross_up_price': rope_crossover['last_rope_cross_up_price'],
        'rope_cross_up_rope': rope_crossover['last_rope_cross_up_rope'],
        'rope_cross_up_dir': rope_crossover['last_rope_cross_up_dir'],
        'rope_cross_down_date': rope_crossover['last_rope_cross_down_date'],
        'rope_cross_down_price': rope_crossover['last_rope_cross_down_price'],
        'rope_cross_down_rope': rope_crossover['last_rope_cross_down_rope'],
        'rope_cross_down_dir': rope_crossover['last_rope_cross_down_dir'],
        'rope_cross_up_count': rope_crossover['rope_cross_up_count'],
        'rope_cross_down_count': rope_crossover['rope_cross_down_count'],
        'vwap_cross_up_dsa_dir': crossover['last_cross_up_dsa_dir'],
        'vwap_cross_down_dsa_dir': crossover['last_cross_down_dsa_dir'],
    }


def ensure_table_exists():
    create_sql = """
    CREATE TABLE IF NOT EXISTS dsa_selection (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        signal_date DATE,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),

        regime VARCHAR(10) NOT NULL,
        regime_value INT,
        regime_strength FLOAT,
        dsa_dir_bars INT,

        offset_rate FLOAT,
        offset_mean FLOAT,
        offset_std FLOAT,
        offset_percentile FLOAT,

        vwap_ret_total FLOAT,
        vwap_ret_5 FLOAT,
        vwap_ret_10 FLOAT,
        vwap_ret_20 FLOAT,

        last_cross_up_date DATE,
        last_cross_up_price FLOAT,
        last_cross_down_date DATE,
        last_cross_down_price FLOAT,
        cross_up_count INT,
        cross_down_count INT,

        change_pct FLOAT,
        vol_zscore FLOAT,
        avg_amount_20d FLOAT,
        dsa_vwap FLOAT,
        dsa_vwap_dev_pct FLOAT,

        rope_dir1_pct FLOAT,
        rope_dir0_pct FLOAT,
        rope_dir_neg1_pct FLOAT,
        touch_rope BOOLEAN,
        touch_vwap BOOLEAN,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_ds_selection_date ON dsa_selection(selection_date);
    CREATE INDEX IF NOT EXISTS idx_ds_ts_code ON dsa_selection(ts_code);
    CREATE INDEX IF NOT EXISTS idx_ds_regime ON dsa_selection(regime);

    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_dir1_pct FLOAT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_dir0_pct FLOAT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_dir_neg1_pct FLOAT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS touch_rope BOOLEAN;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS touch_vwap BOOLEAN;

    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_cross_up_date DATE;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_cross_up_price FLOAT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_cross_up_rope FLOAT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_cross_up_dir INT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_cross_down_date DATE;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_cross_down_price FLOAT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_cross_down_rope FLOAT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_cross_down_dir INT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_cross_up_count INT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS rope_cross_down_count INT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS vwap_cross_up_dsa_dir INT;
    ALTER TABLE dsa_selection ADD COLUMN IF NOT EXISTS vwap_cross_down_dsa_dir INT;
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def save_to_database(df: pd.DataFrame, selection_date: date) -> int:
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
            'regime': row.get('regime', ''),
            'regime_value': int(row['regime_value']) if pd.notna(row.get('regime_value')) else None,
            'regime_strength': float(row['regime_strength']) if pd.notna(row.get('regime_strength')) else None,
            'dsa_dir_bars': int(row['dsa_dir_bars']) if pd.notna(row.get('dsa_dir_bars')) else None,
            'offset_rate': float(row['offset_rate']) if pd.notna(row.get('offset_rate')) else None,
            'offset_mean': float(row['offset_mean']) if pd.notna(row.get('offset_mean')) else None,
            'offset_std': float(row['offset_std']) if pd.notna(row.get('offset_std')) else None,
            'offset_percentile': float(row['offset_percentile']) if pd.notna(row.get('offset_percentile')) else None,
            'vwap_ret_total': float(row['vwap_ret_total']) if pd.notna(row.get('vwap_ret_total')) else None,
            'vwap_ret_5': float(row['vwap_ret_5']) if pd.notna(row.get('vwap_ret_5')) else None,
            'vwap_ret_10': float(row['vwap_ret_10']) if pd.notna(row.get('vwap_ret_10')) else None,
            'vwap_ret_20': float(row['vwap_ret_20']) if pd.notna(row.get('vwap_ret_20')) else None,
            'last_cross_up_date': row.get('last_cross_up_date'),
            'last_cross_up_price': float(row['last_cross_up_price']) if pd.notna(row.get('last_cross_up_price')) else None,
            'last_cross_down_date': row.get('last_cross_down_date'),
            'last_cross_down_price': float(row['last_cross_down_price']) if pd.notna(row.get('last_cross_down_price')) else None,
            'cross_up_count': int(row['cross_up_count']) if pd.notna(row.get('cross_up_count')) else None,
            'cross_down_count': int(row['cross_down_count']) if pd.notna(row.get('cross_down_count')) else None,
            'change_pct': float(row['change_pct']) if pd.notna(row.get('change_pct')) else None,
            'vol_zscore': float(row['vol_zscore']) if pd.notna(row.get('vol_zscore')) else None,
            'avg_amount_20d': float(row['avg_amount_20d']) if pd.notna(row.get('avg_amount_20d')) else None,
            'dsa_vwap': float(row['dsa_vwap']) if pd.notna(row.get('dsa_vwap')) else None,
            'dsa_vwap_dev_pct': float(row['dsa_vwap_dev_pct']) if pd.notna(row.get('dsa_vwap_dev_pct')) else None,
            'rope_dir1_pct': float(row['rope_dir1_pct']) if pd.notna(row.get('rope_dir1_pct')) else None,
            'rope_dir0_pct': float(row['rope_dir0_pct']) if pd.notna(row.get('rope_dir0_pct')) else None,
            'rope_dir_neg1_pct': float(row['rope_dir_neg1_pct']) if pd.notna(row.get('rope_dir_neg1_pct')) else None,
            'touch_rope': bool(row['touch_rope']) if pd.notna(row.get('touch_rope')) else None,
            'touch_vwap': bool(row['touch_vwap']) if pd.notna(row.get('touch_vwap')) else None,
            'rope_cross_up_date': row.get('rope_cross_up_date'),
            'rope_cross_up_price': float(row['rope_cross_up_price']) if pd.notna(row.get('rope_cross_up_price')) else None,
            'rope_cross_up_rope': float(row['rope_cross_up_rope']) if pd.notna(row.get('rope_cross_up_rope')) else None,
            'rope_cross_up_dir': int(row['rope_cross_up_dir']) if pd.notna(row.get('rope_cross_up_dir')) else None,
            'rope_cross_down_date': row.get('rope_cross_down_date'),
            'rope_cross_down_price': float(row['rope_cross_down_price']) if pd.notna(row.get('rope_cross_down_price')) else None,
            'rope_cross_down_rope': float(row['rope_cross_down_rope']) if pd.notna(row.get('rope_cross_down_rope')) else None,
            'rope_cross_down_dir': int(row['rope_cross_down_dir']) if pd.notna(row.get('rope_cross_down_dir')) else None,
            'rope_cross_up_count': int(row['rope_cross_up_count']) if pd.notna(row.get('rope_cross_up_count')) else None,
            'rope_cross_down_count': int(row['rope_cross_down_count']) if pd.notna(row.get('rope_cross_down_count')) else None,
            'vwap_cross_up_dsa_dir': int(row['vwap_cross_up_dsa_dir']) if pd.notna(row.get('vwap_cross_up_dsa_dir')) else None,
            'vwap_cross_down_dsa_dir': int(row['vwap_cross_down_dsa_dir']) if pd.notna(row.get('vwap_cross_down_dsa_dir')) else None,
        }
        records.append(record)

    if records:
        insert_df = pd.DataFrame(records)
        insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)
        print(f"  保存新数据: {len(records)} 条")
        return len(records)

    return 0


def select_dsa_stocks(selection_date: Optional[date] = None) -> pd.DataFrame:
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("选股条件（DSA 多头策略）：")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  核心条件: DSA VWAP dir=1 持续 > {MIN_DIR_BARS} bars")
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
    print("开始 DSA 多头指标筛选...")
    print(f"  原股票数: {len(stock_list)}")

    regime_stats = {'多头': 0, '震荡': 0, '空头': 0, '无数据': 0}
    filtered_results = []

    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="DSA多头选股", unit="只"):
        ts_code = row['ts_code']

        daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
        if daily_df.empty or len(daily_df) < 60:
            regime_stats['无数据'] += 1
            continue

        try:
            dsa_regime, dsa_trend_strength, dsa_bars, dsa_vwap, dsa_dir_series = compute_dsa_regime(daily_df)
        except Exception:
            regime_stats['无数据'] += 1
            continue

        regime = int(dsa_regime.iloc[-1])

        if regime == 1:
            regime_stats['多头'] += 1
        elif regime == -1:
            regime_stats['空头'] += 1
        else:
            regime_stats['震荡'] += 1

        if regime != 1:
            continue

        trend_strength_val = float(dsa_trend_strength.iloc[-1])
        dsa_dir_bars_val = int(dsa_bars.iloc[-1])

        atr_rope_df = None
        try:
            cfg = ATRRopeConfig(regime_lookback=55)
            atr_rope_df = compute_atr_rope(daily_df, cfg)
        except Exception:
            pass

        rope_dir1_pct, rope_dir0_pct, rope_dir_neg1_pct = None, None, None
        touch_rope = False
        touch_vwap = False
        if atr_rope_df is not None and not atr_rope_df.empty:
            rope_dir1_pct, rope_dir0_pct, rope_dir_neg1_pct = compute_rope_dir_pct_in_dsa_trend(atr_rope_df, dsa_dir_bars_val)
            last_rope_val = atr_rope_df['atr_rope_rope'].iloc[-1]
            last_low = daily_df['low'].iloc[-1]
            if pd.notna(last_rope_val) and pd.notna(last_low):
                touch_rope = bool(last_low <= last_rope_val)
            if pd.notna(dsa_vwap.iloc[-1]) and pd.notna(last_low):
                touch_vwap = bool(last_low <= dsa_vwap.iloc[-1])

        offset_stats = compute_offset_rate_stats(
            daily_df['close'], dsa_vwap, dsa_dir_series, dsa_bars
        )
        last_offset = offset_stats.iloc[-1]

        vwap_metrics = compute_vwap_return_metrics(dsa_vwap, dsa_bars)

        crossover = detect_vwap_crossover_events(daily_df['close'], dsa_vwap, dsa_dir_series, dsa_bars)

        rope_crossover = detect_rope_crossover_events(daily_df['close'], atr_rope_df)

        bar_time = daily_df.index[-1]
        last_close = float(daily_df['close'].iloc[-1])
        last_vwap = float(dsa_vwap.iloc[-1]) if pd.notna(dsa_vwap.iloc[-1]) else None

        def safe_float(val, default=None):
            return float(val) if pd.notna(val) else default

        filtered_results.append({
            'ts_code': ts_code,
            'regime': '多头',
            'regime_value': regime,
            'regime_strength': trend_strength_val,
            'dsa_dir_bars': dsa_dir_bars_val,
            'offset_rate': safe_float(last_offset['offset_rate']),
            'offset_mean': safe_float(last_offset['offset_mean']),
            'offset_std': safe_float(last_offset['offset_std']),
            'offset_percentile': safe_float(last_offset['offset_percentile']),
            'vwap_ret_total': vwap_metrics['vwap_ret_total'],
            'vwap_ret_5': vwap_metrics['vwap_ret_5'],
            'vwap_ret_10': vwap_metrics['vwap_ret_10'],
            'vwap_ret_20': vwap_metrics['vwap_ret_20'],
            'last_cross_up_date': crossover['last_cross_up_date'],
            'last_cross_up_price': crossover['last_cross_up_price'],
            'last_cross_down_date': crossover['last_cross_down_date'],
            'last_cross_down_price': crossover['last_cross_down_price'],
            'cross_up_count': crossover['cross_up_count'],
            'cross_down_count': crossover['cross_down_count'],
            'change_pct': compute_change_pct(daily_df),
            'vol_zscore': volume_zscore(daily_df['volume'], win=20),
            'avg_amount_20d': float(((daily_df['open'] + daily_df['close']) / 2 * daily_df['volume']).tail(20).mean()) / 1e8 if len(daily_df) >= 20 else None,
            'dsa_vwap': last_vwap,
            'dsa_vwap_dev_pct': round((last_close / last_vwap - 1) * 100, 4) if last_vwap and last_vwap != 0 else None,
            'rope_dir1_pct': rope_dir1_pct,
            'rope_dir0_pct': rope_dir0_pct,
            'rope_dir_neg1_pct': rope_dir_neg1_pct,
            'touch_rope': touch_rope,
            'touch_vwap': touch_vwap,
            'signal_date': bar_time,
            'rope_cross_up_date': rope_crossover['last_rope_cross_up_date'],
            'rope_cross_up_price': rope_crossover['last_rope_cross_up_price'],
            'rope_cross_up_rope': rope_crossover['last_rope_cross_up_rope'],
            'rope_cross_up_dir': rope_crossover['last_rope_cross_up_dir'],
            'rope_cross_down_date': rope_crossover['last_rope_cross_down_date'],
            'rope_cross_down_price': rope_crossover['last_rope_cross_down_price'],
            'rope_cross_down_rope': rope_crossover['last_rope_cross_down_rope'],
            'rope_cross_down_dir': rope_crossover['last_rope_cross_down_dir'],
            'rope_cross_up_count': rope_crossover['rope_cross_up_count'],
            'rope_cross_down_count': rope_crossover['rope_cross_down_count'],
            'vwap_cross_up_dsa_dir': crossover['last_cross_up_dsa_dir'],
            'vwap_cross_down_dsa_dir': crossover['last_cross_down_dsa_dir'],
        })

    result_df_out = pd.DataFrame(filtered_results)

    if not result_df_out.empty:
        stock_names = batch_get_stock_names(result_df_out['ts_code'].tolist())
        result_df_out['stock_name'] = result_df_out['ts_code'].map(stock_names)

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
    print(f"DSA 多头筛选后: {len(result_df_out)} 只")

    if not result_df_out.empty:
        print(f"\n偏离率百分位分布：")
        pct_col = result_df_out['offset_percentile'].dropna()
        if len(pct_col) > 0:
            for threshold in [0.05, 0.1, 0.2, 0.5, 0.8, 0.9]:
                count = (pct_col <= threshold).sum()
                print(f"  <= {threshold:.0%}: {count} 只 ({count / len(pct_col) * 100:.1f}%)")

        print(f"\nVWAP交叉事件统计：")
        print(f"  有上穿事件: {(result_df_out['cross_up_count'] > 0).sum()} 只")
        print(f"  有下穿事件: {(result_df_out['cross_down_count'] > 0).sum()} 只")

        print("\n" + "=" * 80)
        print("前20名股票：")
        print("=" * 80)
        display_cols = [
            'ts_code', 'stock_name', 'dsa_dir_bars',
            'offset_rate', 'offset_percentile',
            'vwap_ret_total', 'change_pct'
        ]
        print_cols = [c for c in display_cols if c in result_df_out.columns]
        print(result_df_out[print_cols].head(20).to_string(index=False))

    print("\n" + "-" * 80)
    print("保存到数据库...")
    saved_count = save_to_database(result_df_out, selection_date)
    print("-" * 80)

    return result_df_out


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
        print("\n该股票不满足选股条件（非多头 或 成交额不足）")

    return result


def backfill_stock_events(ts_code: str, start_date: date, end_date: date) -> List[Dict]:
    """
    遍历单只股票在日期区间内的所有交易日，记录触发事件。
    一次计算全量数据，然后遍历结果行判断信号。
    """
    daily_df = get_kline_data_db(ts_code, bars=1200, end_date=end_date)
    if daily_df.empty or len(daily_df) < 60:
        return []

    if not isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df.index = pd.to_datetime(daily_df.index)

    try:
        dsa_regime, dsa_trend_strength, dsa_bars, dsa_vwap, dsa_dir_series = compute_dsa_regime(daily_df)
    except Exception:
        return []

    atr_rope_df = None
    try:
        cfg = ATRRopeConfig(regime_lookback=55)
        atr_rope_df = compute_atr_rope(daily_df, cfg)
    except Exception:
        pass

    if atr_rope_df is not None and not isinstance(atr_rope_df.index, pd.DatetimeIndex):
        atr_rope_df.index = pd.to_datetime(atr_rope_df.index)

    offset_stats = compute_offset_rate_stats(
        daily_df['close'], dsa_vwap, dsa_dir_series, dsa_bars
    )

    crossover = detect_vwap_crossover_events(daily_df['close'], dsa_vwap, dsa_dir_series, dsa_bars)

    rope_crossover = detect_rope_crossover_events(daily_df['close'], atr_rope_df)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    mask = (daily_df.index >= start_ts) & (daily_df.index <= end_ts)
    trade_dates = daily_df.loc[mask].index.tolist()
    if not trade_dates:
        return []

    results = []
    for trade_dt in trade_dates:
        loc = daily_df.index.get_loc(trade_dt)
        if loc < 1:
            continue

        regime = int(dsa_regime.loc[trade_dt])
        if regime != 1:
            continue

        sub_df = daily_df.loc[daily_df.index <= trade_dt]

        trend_strength_val = float(dsa_trend_strength.loc[trade_dt])
        dsa_dir_bars_val = int(dsa_bars.loc[trade_dt])

        last_offset = offset_stats.loc[trade_dt]
        vwap_metrics = compute_vwap_return_metrics(
            dsa_vwap.loc[:trade_dt], dsa_bars.loc[:trade_dt]
        )

        sub_crossover = detect_vwap_crossover_events(
            sub_df['close'], dsa_vwap.loc[sub_df.index], dsa_dir_series.loc[sub_df.index],
            dsa_bars.loc[sub_df.index]
        )

        sub_rope_crossover = detect_rope_crossover_events(
            sub_df['close'], atr_rope_df.loc[:trade_dt] if atr_rope_df is not None and trade_dt in atr_rope_df.index else None
        )

        last_close = float(daily_df['close'].loc[trade_dt])
        last_vwap = float(dsa_vwap.loc[trade_dt]) if pd.notna(dsa_vwap.loc[trade_dt]) else None

        rope_dir1_pct, rope_dir0_pct, rope_dir_neg1_pct = None, None, None
        touch_rope = False
        touch_vwap = False
        if atr_rope_df is not None and trade_dt in atr_rope_df.index:
            sub_rope_df = atr_rope_df.loc[:trade_dt]
            rope_dir1_pct, rope_dir0_pct, rope_dir_neg1_pct = compute_rope_dir_pct_in_dsa_trend(sub_rope_df, dsa_dir_bars_val)
            last_rope_val = atr_rope_df.loc[trade_dt, 'atr_rope_rope']
            last_low = daily_df.loc[trade_dt, 'low']
            if pd.notna(last_rope_val) and pd.notna(last_low):
                touch_rope = bool(last_low <= last_rope_val)
            if pd.notna(dsa_vwap.loc[trade_dt]) and pd.notna(last_low):
                touch_vwap = bool(last_low <= dsa_vwap.loc[trade_dt])

        def safe_float(val, default=None):
            return float(val) if pd.notna(val) else default

        results.append({
            'ts_code': ts_code,
            'selection_date': trade_dt,
            'signal_date': trade_dt,
            'regime': '多头',
            'regime_value': regime,
            'regime_strength': trend_strength_val,
            'dsa_dir_bars': dsa_dir_bars_val,
            'offset_rate': safe_float(last_offset['offset_rate']),
            'offset_mean': safe_float(last_offset['offset_mean']),
            'offset_std': safe_float(last_offset['offset_std']),
            'offset_percentile': safe_float(last_offset['offset_percentile']),
            'vwap_ret_total': vwap_metrics['vwap_ret_total'],
            'vwap_ret_5': vwap_metrics['vwap_ret_5'],
            'vwap_ret_10': vwap_metrics['vwap_ret_10'],
            'vwap_ret_20': vwap_metrics['vwap_ret_20'],
            'last_cross_up_date': sub_crossover['last_cross_up_date'],
            'last_cross_up_price': sub_crossover['last_cross_up_price'],
            'last_cross_down_date': sub_crossover['last_cross_down_date'],
            'last_cross_down_price': sub_crossover['last_cross_down_price'],
            'cross_up_count': sub_crossover['cross_up_count'],
            'cross_down_count': sub_crossover['cross_down_count'],
            'change_pct': compute_change_pct(sub_df),
            'vol_zscore': volume_zscore(sub_df['volume'], win=20),
            'avg_amount_20d': float(((sub_df['open'] + sub_df['close']) / 2 * sub_df['volume']).tail(20).mean()) / 1e8 if len(sub_df) >= 20 else None,
            'dsa_vwap': last_vwap,
            'dsa_vwap_dev_pct': round((last_close / last_vwap - 1) * 100, 4) if last_vwap and last_vwap != 0 else None,
            'rope_dir1_pct': rope_dir1_pct,
            'rope_dir0_pct': rope_dir0_pct,
            'rope_dir_neg1_pct': rope_dir_neg1_pct,
            'touch_rope': touch_rope,
            'touch_vwap': touch_vwap,
            'rope_cross_up_date': sub_rope_crossover['last_rope_cross_up_date'],
            'rope_cross_up_price': sub_rope_crossover['last_rope_cross_up_price'],
            'rope_cross_up_rope': sub_rope_crossover['last_rope_cross_up_rope'],
            'rope_cross_up_dir': sub_rope_crossover['last_rope_cross_up_dir'],
            'rope_cross_down_date': sub_rope_crossover['last_rope_cross_down_date'],
            'rope_cross_down_price': sub_rope_crossover['last_rope_cross_down_price'],
            'rope_cross_down_rope': sub_rope_crossover['last_rope_cross_down_rope'],
            'rope_cross_down_dir': sub_rope_crossover['last_rope_cross_down_dir'],
            'rope_cross_up_count': sub_rope_crossover['rope_cross_up_count'],
            'rope_cross_down_count': sub_rope_crossover['rope_cross_down_count'],
            'vwap_cross_up_dsa_dir': sub_crossover['last_cross_up_dsa_dir'],
            'vwap_cross_down_dsa_dir': sub_crossover['last_cross_down_dsa_dir'],
        })

    return results


def _save_single_stock_records(records: List[Dict], stock_name_map: Dict[str, str]):
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
                'regime': rec.get('regime', ''),
                'regime_value': int(rec['regime_value']) if rec.get('regime_value') is not None else None,
                'regime_strength': float(rec['regime_strength']) if rec.get('regime_strength') is not None else None,
                'dsa_dir_bars': int(rec['dsa_dir_bars']) if rec.get('dsa_dir_bars') is not None else None,
                'offset_rate': float(rec['offset_rate']) if rec.get('offset_rate') is not None else None,
                'offset_mean': float(rec['offset_mean']) if rec.get('offset_mean') is not None else None,
                'offset_std': float(rec['offset_std']) if rec.get('offset_std') is not None else None,
                'offset_percentile': float(rec['offset_percentile']) if rec.get('offset_percentile') is not None else None,
                'vwap_ret_total': float(rec['vwap_ret_total']) if rec.get('vwap_ret_total') is not None else None,
                'vwap_ret_5': float(rec['vwap_ret_5']) if rec.get('vwap_ret_5') is not None else None,
                'vwap_ret_10': float(rec['vwap_ret_10']) if rec.get('vwap_ret_10') is not None else None,
                'vwap_ret_20': float(rec['vwap_ret_20']) if rec.get('vwap_ret_20') is not None else None,
                'last_cross_up_date': rec.get('last_cross_up_date'),
                'last_cross_up_price': float(rec['last_cross_up_price']) if rec.get('last_cross_up_price') is not None else None,
                'last_cross_down_date': rec.get('last_cross_down_date'),
                'last_cross_down_price': float(rec['last_cross_down_price']) if rec.get('last_cross_down_price') is not None else None,
                'cross_up_count': int(rec['cross_up_count']) if rec.get('cross_up_count') is not None else None,
                'cross_down_count': int(rec['cross_down_count']) if rec.get('cross_down_count') is not None else None,
                'change_pct': float(rec['change_pct']) if rec.get('change_pct') is not None else None,
                'vol_zscore': float(rec['vol_zscore']) if rec.get('vol_zscore') is not None else None,
                'avg_amount_20d': float(rec['avg_amount_20d']) if rec.get('avg_amount_20d') is not None else None,
                'dsa_vwap': float(rec['dsa_vwap']) if rec.get('dsa_vwap') is not None else None,
                'dsa_vwap_dev_pct': float(rec['dsa_vwap_dev_pct']) if rec.get('dsa_vwap_dev_pct') is not None else None,
                'rope_dir1_pct': float(rec['rope_dir1_pct']) if rec.get('rope_dir1_pct') is not None else None,
                'rope_dir0_pct': float(rec['rope_dir0_pct']) if rec.get('rope_dir0_pct') is not None else None,
                'rope_dir_neg1_pct': float(rec['rope_dir_neg1_pct']) if rec.get('rope_dir_neg1_pct') is not None else None,
                'touch_rope': bool(rec['touch_rope']) if rec.get('touch_rope') is not None else None,
                'touch_vwap': bool(rec['touch_vwap']) if rec.get('touch_vwap') is not None else None,
                'rope_cross_up_date': rec.get('rope_cross_up_date'),
                'rope_cross_up_price': float(rec['rope_cross_up_price']) if rec.get('rope_cross_up_price') is not None else None,
                'rope_cross_up_rope': float(rec['rope_cross_up_rope']) if rec.get('rope_cross_up_rope') is not None else None,
                'rope_cross_up_dir': int(rec['rope_cross_up_dir']) if rec.get('rope_cross_up_dir') is not None else None,
                'rope_cross_down_date': rec.get('rope_cross_down_date'),
                'rope_cross_down_price': float(rec['rope_cross_down_price']) if rec.get('rope_cross_down_price') is not None else None,
                'rope_cross_down_rope': float(rec['rope_cross_down_rope']) if rec.get('rope_cross_down_rope') is not None else None,
                'rope_cross_down_dir': int(rec['rope_cross_down_dir']) if rec.get('rope_cross_down_dir') is not None else None,
                'rope_cross_up_count': int(rec['rope_cross_up_count']) if rec.get('rope_cross_up_count') is not None else None,
                'rope_cross_down_count': int(rec['rope_cross_down_count']) if rec.get('rope_cross_down_count') is not None else None,
                'vwap_cross_up_dsa_dir': int(rec['vwap_cross_up_dsa_dir']) if rec.get('vwap_cross_up_dsa_dir') is not None else None,
                'vwap_cross_down_dsa_dir': int(rec['vwap_cross_down_dsa_dir']) if rec.get('vwap_cross_down_dsa_dir') is not None else None,
            })

        if insert_records:
            insert_df = pd.DataFrame(insert_records)
            insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)
            total_saved += len(insert_records)

    return total_saved


def backfill_range(start_date: date, end_date: date, stock_list: Optional[List[str]] = None):
    print("=" * 80)
    print("DSA 多头事件回补")
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
    parser = argparse.ArgumentParser(
        description='DSA 多头选股工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/selection_dsa.py                    # 使用当天日期选股
  python selection/selection_dsa.py 2026-05-19         # 指定日期选股
  python selection/selection_dsa.py 20260519           # 指定日期选股（无分隔符）
  python selection/selection_dsa.py --test 600547      # 测试单只股票
  python selection/selection_dsa.py --backfill 2025-07-01 2026-04-30  # 回补历史事件
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
    print("DSA 多头选股工具")
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print("=" * 80)

    if args.test:
        test_single_stock(args.test, selection_date)
        sys.exit(0)

    df = select_dsa_stocks(selection_date=selection_date)

    print("\n" + "=" * 80)
    print("选股完成")
    print(f"选股日期: {selection_date}")
    print(f"选中股票数: {len(df)}")
    print(f"查询SQL: SELECT * FROM {SELECTION_TABLE} WHERE selection_date = '{selection_date}'")
    print("=" * 80)

    return df


if __name__ == '__main__':
    main()
