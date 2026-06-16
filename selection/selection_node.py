#!/usr/bin/env python3
"""
Node 触碰选股脚本（Peak Node 触碰 + 最近10bar涨停）

Purpose:
    筛选当日价格区间触碰 Volume Profile Peak Node 且最近10个bar内出现过涨停的股票。

Inputs:
    stock_k_data (日线K线数据 + 15分钟K线数据)
    stock_pools (股票名称，用于排除ST)

Outputs:
    node_selection (选股结果表)

How to Run:
    python selection/selection_node.py              # 当天
    python selection/selection_node.py 2026-06-12   # 指定日期
    python selection/selection_node.py --test 000151 # 测试单只
    python selection/selection_node.py --backfill 2023-01-01 2026-06-12  # 回补

Examples:
    python selection/selection_node.py --test 000151.SZ 2026-06-12
    python selection/selection_node.py 2026-06-12
    python selection/selection_node.py --backfill 2023-01-01 2026-06-12

Side Effects:
    写入 node_selection 表（幂等：同一日期+同一股票先删后插）

================================================================================
【选股逻辑】

核心条件（全部满足）：
  1. Node 触碰：当日 [low, high] 区间与任意 Peak Node 价格区间有重叠
     day_low <= node_price_high AND day_high >= node_price_low
  2. 最近10bar涨停：最近10个交易日（含当日）内至少出现一次涨停
     涨停定义：close >= round(prev_close * (1 + limit_pct), 2) - 0.01

排除规则：
  - ST 股票（名称含 ST）
  - 北交所股票（ts_code 以 8 或 4 开头）

涨停幅度：
  - 主板：10%
  - 创业板（30开头）：20%
  - 科创板（688开头）：20%

Volume Profile 节点分析（日线价格范围 + 15m成交量分配）：
  标准参数（与 luxalgo_volume_profile_pytdx_15m_aligned.py CLI 一致）：
    - 数据源: 日线K线确定价格范围 + 15m K线分配成交量（均前复权）
    - 日线加载: 1500根（对应 CLI --tdx-count 1500）
    - 15m加载: 全量（对应 CLI 自动估算约6784根）
    - profile_lookback_length=360（对应 CLI --lookback 360）
    - profile_number_of_rows=100（对应 CLI --rows 100）
    - peaks_show="peaks"
    - main_period="day"
    对应 CLI: python luxalgo_volume_profile_pytdx_15m_aligned.py --tdx-code XXX --tdx-count 1500 --lookback 360 --rows 100

【回补模式】
  - 外层遍历股票，每只股票一次拉取全量数据
  - 日线K线 + 15分钟K线一次加载
  - 遍历日期区间内的每个交易日，判断 Node 触碰 + 涨停条件
  - 每只股票处理完立即保存

【保存逻辑】
  - 单日选股：按选股日期先删后插
  - 回补模式：按日期+股票先删后插（幂等性）
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
from datetime import datetime, date
from typing import Dict, Optional, List
from collections import defaultdict
from tqdm import tqdm

from selection.selection_dsa import (
    get_kline_data_db,
    batch_get_stock_names,
    compute_dsa_regime,
    compute_change_pct,
    volume_zscore,
)
from features.luxalgo_volume_profile_pytdx_15m_aligned import (
    compute_volume_profile,
    VolumeProfileConfig,
)
from datasource.k_data_loader import load_k_data

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "node_selection"
DAILY_BARS = 1500  # 与 CLI --tdx-count 1500 一致
RECENT_LIMIT_UP_BARS = 10

# Volume Profile 标准参数（与 luxalgo_volume_profile_pytdx_15m_aligned.py CLI 一致）
VP_LOOKBACK = 360
VP_ROWS = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 涨停价计算
# ---------------------------------------------------------------------------

def get_limit_pct(ts_code: str) -> float:
    """获取涨跌停幅度比例（按板块区分）"""
    symbol = ts_code.split('.')[0] if '.' in ts_code else ts_code
    if symbol.startswith("30"):
        return 0.20
    if symbol.startswith("688"):
        return 0.20
    return 0.10


def calc_limit_up_price(prev_close: float, limit_pct: float) -> float:
    """计算涨停价（四舍五入到2位小数）"""
    return round(prev_close * (1 + limit_pct), 2)


def is_limit_up_price(today_close: float, prev_close: float, limit_pct: float) -> bool:
    """判断是否涨停（0.01元容差，应对前复权精度偏差）"""
    limit_price = calc_limit_up_price(prev_close, limit_pct)
    return today_close >= limit_price - 0.01


# ---------------------------------------------------------------------------
# 涨停检测（最近10bar）
# ---------------------------------------------------------------------------

def check_recent_limit_up(daily_df: pd.DataFrame, loc: int, limit_pct: float) -> Dict:
    """
    检查最近 RECENT_LIMIT_UP_BARS(10) 个bar内是否出现涨停。

    涨停定义（与 selection_limit_up.py 一致）：
      is_limit_up_price(close, prev_close, limit_pct)
      即 close >= round(prev_close * (1 + limit_pct), 2) - 0.01

    Args:
        daily_df: 日线数据（前复权）
        loc: 当日bar在daily_df中的位置
        limit_pct: 涨停幅度（主板0.10, 创业板/科创板0.20）

    Returns:
        {recent_limit_up: bool, recent_limit_up_date: date, recent_limit_up_count: int}
    """
    start_loc = max(1, loc - RECENT_LIMIT_UP_BARS + 1)
    count = 0
    last_date = None
    for i in range(start_loc, loc + 1):
        close_i = float(daily_df['close'].iloc[i])
        prev_close_i = float(daily_df['close'].iloc[i - 1])
        if is_limit_up_price(close_i, prev_close_i, limit_pct):
            count += 1
            last_date = daily_df.index[i]
    return {
        'recent_limit_up': count > 0,
        'recent_limit_up_date': last_date,
        'recent_limit_up_count': count,
    }


# ---------------------------------------------------------------------------
# Node 触碰检测
# ---------------------------------------------------------------------------

def check_node_touch(vp_result: 'VolumeProfileResult', day_low: float, day_high: float) -> Dict:
    """
    检查当日 [day_low, day_high] 是否触碰任何 Peak Node。

    触碰定义：当日价格区间与 Node 价格区间有重叠
      day_low <= node_price_high AND day_high >= node_price_low

    Args:
        vp_result: VolumeProfileResult 对象（含 peak_df, poc_price, vah_price, val_price）
        day_low: 当日最低价
        day_high: 当日最高价

    Returns:
        dict with keys: touched, peak_node_count, touched_node_prices,
                        nearest_above_node_price, nearest_below_node_price,
                        poc_price, vah_price, val_price
    """
    from features.luxalgo_volume_profile_pytdx_15m_aligned import extract_nearest_nodes

    peak_df = vp_result.peak_df
    touched_nodes = []
    if peak_df is not None and not peak_df.empty:
        for _, row in peak_df.iterrows():
            if day_low <= row['price_high'] and day_high >= row['price_low']:
                touched_nodes.append(row)

    # 直接使用 VolumeProfileResult 已有字段，禁止从 profile_df 重复计算
    poc_price = vp_result.poc_price if not np.isnan(vp_result.poc_price) else None
    vah_price = vp_result.vah_price if not np.isnan(vp_result.vah_price) else None
    val_price = vp_result.val_price if not np.isnan(vp_result.val_price) else None

    # 上下方最近 Node：调用 SSOT，分别以 day_high/day_low 为参考
    above_info = extract_nearest_nodes(vp_result, reference_price=day_high)
    below_info = extract_nearest_nodes(vp_result, reference_price=day_low)
    nearest_above = above_info['nearest_above_node_price']
    nearest_below = below_info['nearest_below_node_price']

    return {
        'touched': len(touched_nodes) > 0,
        'peak_node_count': len(touched_nodes),
        'touched_node_prices': ','.join([f"{n['price_mid']:.2f}" for n in touched_nodes]) if touched_nodes else None,
        'nearest_above_node_price': nearest_above,
        'nearest_below_node_price': nearest_below,
        'poc_price': round(poc_price, 4) if poc_price is not None else None,
        'vah_price': round(vah_price, 4) if vah_price is not None else None,
        'val_price': round(val_price, 4) if val_price is not None else None,
    }


# ---------------------------------------------------------------------------
# Volume Profile 计算（日线价格范围 + 15m成交量分配）
# ---------------------------------------------------------------------------

def compute_volume_profile_15m(
    ts_code: str,
    end_date,
    day_low: float,
    day_high: float,
    daily_df: pd.DataFrame = None,
) -> Dict:
    """
    使用日线K线确定价格范围 + 15m K线分配成交量，计算 Volume Profile 并检测 Node 触碰。

    复现 Pine Script 的 request.security_lower_tf 行为：
    - 日线K线：确定价格范围（high/low），lookback=360根日线
    - 15m K线：分配成交量，更精细的成交量分布

    Args:
        ts_code: 股票代码
        end_date: 截止日期（包含当日数据）
        day_low: 当日最低价
        day_high: 当日最高价
        daily_df: 日线K线数据（可选，若提供则不再重复加载）

    Returns:
        check_node_touch() 的返回字典
    """
    empty_result = {
        'touched': False,
        'peak_node_count': 0,
        'touched_node_prices': None,
        'nearest_above_node_price': None,
        'nearest_below_node_price': None,
        'poc_price': None,
        'vah_price': None,
        'val_price': None,
    }

    try:
        # 1. 加载日线K线（确定价格范围）
        if daily_df is None:
            daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=end_date)
        if daily_df.empty or len(daily_df) < 10:
            return empty_result

        # 2. 加载15m K线（分配成交量，前复权以匹配日线价格）
        ltf_df = load_k_data(ts_code, freq='15m')
        if ltf_df.empty or len(ltf_df) < 10:
            return empty_result

        from datasource.adj_factor import apply_adj_factor_intraday
        ltf_df = apply_adj_factor_intraday(ltf_df, ts_code)

        # 3. 截取 end_date 之前的数据（包含当日）
        if not isinstance(ltf_df.index, pd.DatetimeIndex):
            ltf_df.index = pd.to_datetime(ltf_df.index)

        end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
        ltf_df = ltf_df[ltf_df.index <= end_ts]

        if ltf_df.empty or len(ltf_df) < 10:
            return empty_result

        # 4. 准备数据：只保留lookback范围内的OHLCV+datetime列
        daily_for_vp = daily_df.tail(VP_LOOKBACK)[['open', 'high', 'low', 'close', 'volume']].copy()
        daily_for_vp['datetime'] = daily_for_vp.index

        vp_start = daily_for_vp.index[0]
        ltf_for_vp = ltf_df[ltf_df.index >= vp_start][['open', 'high', 'low', 'close', 'volume']].copy()
        ltf_for_vp['datetime'] = ltf_for_vp.index

        # 5. 计算 Volume Profile
        cfg = VolumeProfileConfig(
            peaks_show="peaks",
            profile_lookback_length=VP_LOOKBACK,
            profile_number_of_rows=VP_ROWS,
            peaks_detection_percent=0.05,  # 与 UI/monitoring 保持一致
        )
        vp_result = compute_volume_profile(
            daily_for_vp, cfg,
            profile_df=ltf_for_vp,
            main_period="day",
        )
        return check_node_touch(vp_result, day_low, day_high)

    except Exception as e:
        logger.debug(f"Volume Profile计算异常 {ts_code}: {e}")
        return empty_result


# ---------------------------------------------------------------------------
# 单股处理（单日选股模式）
# ---------------------------------------------------------------------------

def process_stock(ts_code: str, selection_date: date, stock_name: str = "") -> Optional[Dict]:
    """
    处理单只股票的 Node 触碰选股逻辑（单日模式）。

    选股条件（全部满足）：
        1. 当日 [low, high] 触碰任意 Peak Node
        2. 最近10个bar内出现过涨停
        3. 排除ST和北交所

    Returns: 信号字典，满足条件返回结果，否则返回None
    """
    # 排除北交所
    symbol = ts_code.split('.')[0] if '.' in ts_code else ts_code
    if symbol.startswith("8") or symbol.startswith("4"):
        return None

    # 排除ST
    if stock_name and "ST" in stock_name.upper():
        return None

    # 获取K线数据
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
    if daily_df.empty or len(daily_df) < RECENT_LIMIT_UP_BARS + 1:
        return None

    loc = len(daily_df) - 1
    today_low = float(daily_df['low'].iloc[loc])
    today_high = float(daily_df['high'].iloc[loc])
    today_close = float(daily_df['close'].iloc[loc])
    today_open = float(daily_df['open'].iloc[loc])

    # 条件②：最近10bar内出现涨停（快速过滤，大部分股票在此被排除）
    limit_pct = get_limit_pct(symbol)
    limit_up_info = check_recent_limit_up(daily_df, loc, limit_pct)
    if not limit_up_info['recent_limit_up']:
        return None

    # 条件①：Node 触碰
    node_info = compute_volume_profile_15m(
        ts_code, selection_date, today_low, today_high, daily_df=daily_df
    )
    if not node_info['touched']:
        return None

    # 计算 DSA VWAP（仅记录方向，不过滤；放在筛选条件之后以减少计算量）
    try:
        dsa_regime, dsa_trend_strength, dsa_bars, dsa_vwap, dsa_dir_series = compute_dsa_regime(daily_df)
    except Exception as e:
        logger.debug(f"DSA计算异常 {ts_code}: {e}")
        dsa_regime = None
        dsa_bars = None
        dsa_vwap = None

    # DSA 方向信息（仅记录，不过滤）
    regime = int(dsa_regime.iloc[-1]) if dsa_regime is not None else None
    regime_name = {1: '多头', 0: '震荡', -1: '空头'}.get(regime, '未知') if regime is not None else None
    dsa_dir_bars_val = int(dsa_bars.iloc[-1]) if dsa_bars is not None else None
    last_vwap = float(dsa_vwap.iloc[-1]) if dsa_vwap is not None and pd.notna(dsa_vwap.iloc[-1]) else None

    # 组装结果
    bar_time = daily_df.index[loc]
    change_pct = compute_change_pct(daily_df)

    return {
        'ts_code': ts_code,
        'regime': regime_name or '',
        'regime_value': regime,
        'dsa_dir_bars': dsa_dir_bars_val,
        'dsa_vwap': last_vwap,
        'dsa_vwap_dev_pct': round((today_close / last_vwap - 1) * 100, 4) if last_vwap and last_vwap != 0 else None,
        'today_open': today_open,
        'today_high': today_high,
        'today_low': today_low,
        'today_close': today_close,
        'change_pct': change_pct,
        'vol_zscore': volume_zscore(daily_df['volume'], win=20),
        'avg_amount_20d': float(((daily_df['open'] + daily_df['close']) / 2 * daily_df['volume']).tail(20).mean()) / 1e8 if len(daily_df) >= 20 else None,
        'recent_limit_up': limit_up_info['recent_limit_up'],
        'recent_limit_up_date': limit_up_info['recent_limit_up_date'],
        'recent_limit_up_count': limit_up_info['recent_limit_up_count'],
        'peak_node_count': node_info['peak_node_count'],
        'nearest_above_node_price': node_info['nearest_above_node_price'],
        'nearest_below_node_price': node_info['nearest_below_node_price'],
        'touched_node_prices': node_info['touched_node_prices'],
        'poc_price': node_info['poc_price'],
        'vah_price': node_info['vah_price'],
        'val_price': node_info['val_price'],
        'signal_date': bar_time,
    }


# ---------------------------------------------------------------------------
# 单股回补（遍历日期区间内的所有符合条件的交易日）
# ---------------------------------------------------------------------------

def backfill_stock_events(ts_code: str, start_date: date, end_date: date) -> List[Dict]:
    """
    遍历单只股票在日期区间内的所有交易日，记录 Node 触碰 + 涨停事件。
    一次拉取全量K线数据，15m数据一次加载，每个交易日复用。

    优化：先用涨停条件预扫，无涨停日则跳过15m加载和DSA计算。
    """
    symbol = ts_code.split('.')[0] if '.' in ts_code else ts_code
    if symbol.startswith("8") or symbol.startswith("4"):
        return []

    # 获取全量日线K线
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=end_date)
    if daily_df.empty or len(daily_df) < RECENT_LIMIT_UP_BARS + 1:
        return []

    if not isinstance(daily_df.index, pd.DatetimeIndex):
        daily_df.index = pd.to_datetime(daily_df.index)

    limit_pct = get_limit_pct(symbol)

    # 遍历日期区间内的交易日
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    mask = (daily_df.index >= start_ts) & (daily_df.index <= end_ts)
    trade_dates = daily_df.loc[mask].index.tolist()
    if not trade_dates:
        return []

    # 预扫：先找出所有满足"最近10bar内出现涨停"的交易日
    # 大部分股票在此步骤被完全过滤，无需加载15m数据和计算DSA
    candidate_dates = []
    for trade_dt in trade_dates:
        loc = daily_df.index.get_loc(trade_dt)
        if loc < RECENT_LIMIT_UP_BARS:
            continue
        limit_up_info = check_recent_limit_up(daily_df, loc, limit_pct)
        if limit_up_info['recent_limit_up']:
            candidate_dates.append((trade_dt, loc, limit_up_info))

    if not candidate_dates:
        return []

    # 有候选日期才加载15m数据和计算DSA（节省大量时间）
    dsa_regime = None
    dsa_bars = None
    dsa_vwap = None
    try:
        dsa_regime, dsa_trend_strength, dsa_bars, dsa_vwap, dsa_dir_series = compute_dsa_regime(daily_df)
    except Exception:
        pass

    ltf_df = None
    try:
        ltf_df = load_k_data(ts_code, freq='15m')
        if ltf_df.empty or len(ltf_df) < 10:
            ltf_df = None
        else:
            from datasource.adj_factor import apply_adj_factor_intraday
            ltf_df = apply_adj_factor_intraday(ltf_df, ts_code)
            if not isinstance(ltf_df.index, pd.DatetimeIndex):
                ltf_df.index = pd.to_datetime(ltf_df.index)
    except Exception as e:
        logger.debug(f"15m数据加载异常 {ts_code}: {e}")
        ltf_df = None

    # 遍历候选日期，计算 Node 触碰
    results = []
    for trade_dt, loc, limit_up_info in candidate_dates:
        today_low = float(daily_df['low'].iloc[loc])
        today_high = float(daily_df['high'].iloc[loc])
        today_close = float(daily_df['close'].iloc[loc])
        today_open = float(daily_df['open'].iloc[loc])

        # 条件①：Node 触碰
        node_info = {
            'touched': False,
            'peak_node_count': 0,
            'touched_node_prices': None,
            'nearest_above_node_price': None,
            'nearest_below_node_price': None,
            'poc_price': None,
            'vah_price': None,
            'val_price': None,
        }
        if ltf_df is not None:
            sub_ltf = ltf_df[ltf_df.index <= trade_dt + pd.Timedelta(hours=23, minutes=59)]
            if len(sub_ltf) >= 10:
                try:
                    cfg = VolumeProfileConfig(
                        peaks_show="peaks",
                        profile_lookback_length=VP_LOOKBACK,
                        profile_number_of_rows=VP_ROWS,
                        peaks_detection_percent=0.05,  # 与 UI/monitoring 保持一致
                    )
                    sub_daily = daily_df.loc[:trade_dt]
                    # 只截取lookback范围内的日线和15m数据，避免传入全量数据导致VP内部处理慢
                    sub_daily_vp = sub_daily.tail(VP_LOOKBACK)[['open', 'high', 'low', 'close', 'volume']].copy()
                    sub_daily_vp['datetime'] = sub_daily_vp.index
                    # 15m数据只保留lookback日线覆盖的时间范围（约 VP_LOOKBACK*16 条）
                    vp_start = sub_daily_vp.index[0]
                    sub_ltf_vp = sub_ltf[sub_ltf.index >= vp_start][['open', 'high', 'low', 'close', 'volume']].copy()
                    sub_ltf_vp['datetime'] = sub_ltf_vp.index
                    if len(sub_ltf_vp) < 10:
                        continue
                    vp_result = compute_volume_profile(
                        sub_daily_vp, cfg,
                        profile_df=sub_ltf_vp,
                        main_period="day",
                    )
                    node_info = check_node_touch(vp_result, today_low, today_high)
                except Exception as e:
                    logger.debug(f"VP计算异常 {ts_code} {trade_dt}: {e}")

        if not node_info['touched']:
            continue

        sub_df = daily_df.loc[:trade_dt]

        # DSA 方向信息（仅记录，不过滤）
        regime = int(dsa_regime.loc[trade_dt]) if dsa_regime is not None and trade_dt in dsa_regime.index else None
        regime_name = {1: '多头', 0: '震荡', -1: '空头'}.get(regime, '未知') if regime is not None else None
        dsa_dir_bars_val = int(dsa_bars.loc[trade_dt]) if dsa_bars is not None and trade_dt in dsa_bars.index else None
        last_vwap = float(dsa_vwap.loc[trade_dt]) if dsa_vwap is not None and trade_dt in dsa_vwap.index and pd.notna(dsa_vwap.loc[trade_dt]) else None

        results.append({
            'ts_code': ts_code,
            'selection_date': trade_dt,
            'signal_date': trade_dt,
            'regime': regime_name or '',
            'regime_value': regime,
            'dsa_dir_bars': dsa_dir_bars_val,
            'dsa_vwap': last_vwap,
            'dsa_vwap_dev_pct': round((today_close / last_vwap - 1) * 100, 4) if last_vwap and last_vwap != 0 else None,
            'today_open': today_open,
            'today_high': today_high,
            'today_low': today_low,
            'today_close': today_close,
            'change_pct': compute_change_pct(sub_df),
            'vol_zscore': volume_zscore(sub_df['volume'], win=20),
            'avg_amount_20d': float(((sub_df['open'] + sub_df['close']) / 2 * sub_df['volume']).tail(20).mean()) / 1e8 if len(sub_df) >= 20 else None,
            'recent_limit_up': limit_up_info['recent_limit_up'],
            'recent_limit_up_date': limit_up_info['recent_limit_up_date'],
            'recent_limit_up_count': limit_up_info['recent_limit_up_count'],
            'peak_node_count': node_info['peak_node_count'],
            'nearest_above_node_price': node_info['nearest_above_node_price'],
            'nearest_below_node_price': node_info['nearest_below_node_price'],
            'touched_node_prices': node_info['touched_node_prices'],
            'poc_price': node_info['poc_price'],
            'vah_price': node_info['vah_price'],
            'val_price': node_info['val_price'],
        })

    return results


# ---------------------------------------------------------------------------
# 数据库
# ---------------------------------------------------------------------------

def ensure_table_exists():
    create_sql = """
    CREATE TABLE IF NOT EXISTS node_selection (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        signal_date DATE,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),

        regime VARCHAR(10),
        regime_value INT,
        dsa_dir_bars INT,
        dsa_vwap FLOAT,
        dsa_vwap_dev_pct FLOAT,

        today_open FLOAT,
        today_high FLOAT,
        today_low FLOAT,
        today_close FLOAT,
        change_pct FLOAT,
        vol_zscore FLOAT,
        avg_amount_20d FLOAT,

        recent_limit_up BOOLEAN,
        recent_limit_up_date DATE,
        recent_limit_up_count INT,

        peak_node_count INT,
        nearest_above_node_price FLOAT,
        nearest_below_node_price FLOAT,
        touched_node_prices TEXT,
        poc_price FLOAT,
        vah_price FLOAT,
        val_price FLOAT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_ns_selection_date ON node_selection(selection_date);
    CREATE INDEX IF NOT EXISTS idx_ns_ts_code ON node_selection(ts_code);
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def save_to_database(df: pd.DataFrame, selection_date: date) -> int:
    """单日选股模式保存（按日期先删后插）"""
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

    return _insert_records(df, selection_date=selection_date)


def _insert_records(df: pd.DataFrame, selection_date: date = None) -> int:
    """将 DataFrame 插入数据库"""
    records = []
    for _, row in df.iterrows():
        sel_date = selection_date or row.get('selection_date')
        record = {
            'selection_date': sel_date,
            'signal_date': row.get('signal_date'),
            'ts_code': row['ts_code'],
            'stock_name': row.get('stock_name', '') or '',
            'regime': row.get('regime', ''),
            'regime_value': int(row['regime_value']) if pd.notna(row.get('regime_value')) else None,
            'dsa_dir_bars': int(row['dsa_dir_bars']) if pd.notna(row.get('dsa_dir_bars')) else None,
            'dsa_vwap': float(row['dsa_vwap']) if pd.notna(row.get('dsa_vwap')) else None,
            'dsa_vwap_dev_pct': float(row['dsa_vwap_dev_pct']) if pd.notna(row.get('dsa_vwap_dev_pct')) else None,
            'today_open': float(row['today_open']) if pd.notna(row.get('today_open')) else None,
            'today_high': float(row['today_high']) if pd.notna(row.get('today_high')) else None,
            'today_low': float(row['today_low']) if pd.notna(row.get('today_low')) else None,
            'today_close': float(row['today_close']) if pd.notna(row.get('today_close')) else None,
            'change_pct': float(row['change_pct']) if pd.notna(row.get('change_pct')) else None,
            'vol_zscore': float(row['vol_zscore']) if pd.notna(row.get('vol_zscore')) else None,
            'avg_amount_20d': float(row['avg_amount_20d']) if pd.notna(row.get('avg_amount_20d')) else None,
            'recent_limit_up': bool(row['recent_limit_up']) if pd.notna(row.get('recent_limit_up')) else None,
            'recent_limit_up_date': row.get('recent_limit_up_date'),
            'recent_limit_up_count': int(row['recent_limit_up_count']) if pd.notna(row.get('recent_limit_up_count')) else None,
            'peak_node_count': int(row['peak_node_count']) if pd.notna(row.get('peak_node_count')) else None,
            'nearest_above_node_price': float(row['nearest_above_node_price']) if pd.notna(row.get('nearest_above_node_price')) else None,
            'nearest_below_node_price': float(row['nearest_below_node_price']) if pd.notna(row.get('nearest_below_node_price')) else None,
            'touched_node_prices': row.get('touched_node_prices'),
            'poc_price': float(row['poc_price']) if pd.notna(row.get('poc_price')) else None,
            'vah_price': float(row['vah_price']) if pd.notna(row.get('vah_price')) else None,
            'val_price': float(row['val_price']) if pd.notna(row.get('val_price')) else None,
        }
        records.append(record)

    if records:
        insert_df = pd.DataFrame(records)
        insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)
        return len(records)

    return 0


def _save_single_stock_records(records: List[Dict], stock_name_map: Dict[str, str]) -> int:
    """回补模式：单股记录保存（按日期+股票先删后插）"""
    if not records:
        return 0

    ensure_table_exists()

    for rec in records:
        rec['stock_name'] = stock_name_map.get(rec['ts_code'], '') or ''

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

        df = pd.DataFrame(day_records)
        saved = _insert_records(df)
        total_saved += saved

    return total_saved


# ---------------------------------------------------------------------------
# 批量选股（单日模式）
# ---------------------------------------------------------------------------

def select_node_stocks(selection_date: Optional[date] = None) -> pd.DataFrame:
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("选股条件（Node 触碰策略）：")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  条件①: 当日 [low, high] 触碰任意 Peak Node")
    print(f"  条件②: 最近{RECENT_LIMIT_UP_BARS}个bar内出现过涨停")
    print(f"  排除规则: ST股、北交所")
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

    # 批量获取股票名称
    ts_codes = stock_list['ts_code'].tolist()
    stock_names = batch_get_stock_names(ts_codes)

    # 排除北交所和ST（提前过滤，减少计算量）
    filtered_codes = []
    for code in ts_codes:
        symbol = code.split('.')[0] if '.' in code else code
        if symbol.startswith("8") or symbol.startswith("4"):
            continue
        name = stock_names.get(code, "")
        if name and "ST" in name.upper():
            continue
        filtered_codes.append(code)

    print(f"  排除北交所和ST后: {len(filtered_codes)} 只")

    print("\n" + "=" * 80)
    print("开始 Node 触碰筛选...")

    filtered_results = []

    for ts_code in tqdm(filtered_codes, desc="Node选股", unit="只"):
        stock_name = stock_names.get(ts_code, "")
        result = process_stock(ts_code, selection_date, stock_name)
        if result is None:
            continue
        filtered_results.append(result)

    result_df = pd.DataFrame(filtered_results)

    if not result_df.empty:
        result_df['stock_name'] = result_df['ts_code'].map(stock_names)

    print("\n" + "=" * 80)
    print("选股结果汇总：")
    print("=" * 80)
    print(f"  排除北交所/ST后扫描: {len(filtered_codes)} 只")
    print(f"  Node触碰+近期涨停: {len(result_df)} 只")

    if not result_df.empty:
        print(f"  近期涨停次数分布: {dict(result_df['recent_limit_up_count'].value_counts().sort_index())}")
        print(f"  触碰Node数量分布: {dict(result_df['peak_node_count'].value_counts().sort_index())}")
        if 'regime' in result_df.columns:
            print(f"  DSA方向分布: {dict(result_df['regime'].value_counts())}")

        print("\n" + "=" * 80)
        print("前20名股票：")
        print("=" * 80)
        display_cols = [
            'ts_code', 'stock_name', 'regime', 'dsa_dir_bars',
            'today_close', 'change_pct',
            'recent_limit_up_count', 'peak_node_count',
            'nearest_above_node_price', 'nearest_below_node_price',
            'poc_price',
        ]
        print_cols = [c for c in display_cols if c in result_df.columns]
        print(result_df[print_cols].head(20).to_string(index=False))

    print("\n" + "-" * 80)
    print("保存到数据库...")
    saved_count = save_to_database(result_df, selection_date)
    print(f"  保存 {saved_count} 条")
    print("-" * 80)

    return result_df


# ---------------------------------------------------------------------------
# 批量回补
# ---------------------------------------------------------------------------

def backfill_range(start_date: date, end_date: date, stock_list: Optional[List[str]] = None,
                   limit: int = None):
    """外层遍历股票，逐只回补 Node 触碰 + 涨停事件"""
    print("=" * 80)
    print("Node 触碰事件回补")
    print(f"  日期范围: {start_date} ~ {end_date}")
    print(f"  条件①: 当日 [low, high] 触碰任意 Peak Node")
    print(f"  条件②: 最近{RECENT_LIMIT_UP_BARS}个bar内出现过涨停")
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

    # 排除北交所
    stock_list = [c for c in stock_list if not (
        c.split('.')[0].startswith("8") or c.split('.')[0].startswith("4")
    )]

    if not stock_list:
        print("无股票可处理")
        return

    # 批量获取股票名称
    stock_names = batch_get_stock_names(stock_list)

    # 排除ST
    stock_list = [c for c in stock_list if "ST" not in (stock_names.get(c, "") or "").upper()]

    print(f"  排除北交所/ST后: {len(stock_list)} 只")

    if limit:
        stock_list = stock_list[:limit]
        print(f"  限制数量: {limit} 只")

    total_saved = 0
    total_stocks_with_events = 0

    for ts_code in tqdm(stock_list, desc="回补个股", unit="只"):
        records = backfill_stock_events(ts_code, start_date, end_date)
        if records:
            name_map = {ts_code: stock_names.get(ts_code, "")}
            saved = _save_single_stock_records(records, name_map)
            total_saved += saved
            total_stocks_with_events += 1

    print("-" * 80)
    print(f"\n回补完成")
    print(f"共处理 {len(stock_list)} 只股票，其中 {total_stocks_with_events} 只有 Node 触碰事件")
    print(f"共保存 {total_saved} 条记录")
    print(f"日期范围: {start_date} ~ {end_date}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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

    # 获取股票名称
    stock_names = batch_get_stock_names([ts_code])
    stock_name = stock_names.get(ts_code, "")

    # 加载K线数据
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
    if daily_df.empty or len(daily_df) < RECENT_LIMIT_UP_BARS + 1:
        print(f"K线数据不足（需要至少{RECENT_LIMIT_UP_BARS + 1}根）")
        return None

    symbol = ts_code.split('.')[0] if '.' in ts_code else ts_code
    limit_pct = get_limit_pct(symbol)
    loc = len(daily_df) - 1

    today_low = float(daily_df['low'].iloc[loc])
    today_high = float(daily_df['high'].iloc[loc])
    today_close = float(daily_df['close'].iloc[loc])
    today_open = float(daily_df['open'].iloc[loc])

    print(f"\n当日行情：")
    print(f"  开盘: {today_open:.2f}, 最高: {today_high:.2f}, 最低: {today_low:.2f}, 收盘: {today_close:.2f}")

    # 涨停检测
    limit_up_info = check_recent_limit_up(daily_df, loc, limit_pct)
    print(f"\n涨停检测（最近{RECENT_LIMIT_UP_BARS}个bar）：")
    print(f"  板块涨幅限制: {limit_pct * 100:.0f}%")
    print(f"  是否出现涨停: {limit_up_info['recent_limit_up']}")
    print(f"  最近涨停日期: {limit_up_info['recent_limit_up_date']}")
    print(f"  涨停次数: {limit_up_info['recent_limit_up_count']}")

    # Node 触碰检测
    print(f"\nVolume Profile 计算...")
    node_info = compute_volume_profile_15m(
        ts_code, selection_date, today_low, today_high, daily_df=daily_df
    )
    print(f"  Node 触碰: {node_info['touched']}")
    print(f"  触碰 Node 数量: {node_info['peak_node_count']}")
    print(f"  触碰 Node 价格: {node_info['touched_node_prices']}")
    print(f"  上方最近 Node: {node_info['nearest_above_node_price']}")
    print(f"  下方最近 Node: {node_info['nearest_below_node_price']}")
    print(f"  POC: {node_info['poc_price']}")
    print(f"  VAH: {node_info['vah_price']}")
    print(f"  VAL: {node_info['val_price']}")

    # 完整选股判断
    result = process_stock(ts_code, selection_date, stock_name)

    if result:
        print("\n选股结果: 入选")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("\n选股结果: 不满足条件")
        if not limit_up_info['recent_limit_up']:
            print("  原因: 最近10bar内未出现涨停")
        elif not node_info['touched']:
            print("  原因: 当日价格区间未触碰 Peak Node")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Node触碰选股（Peak Node触碰 + 最近10bar涨停）")
    parser.add_argument("date", nargs="?", help="选股日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--test", metavar="TSCODE", help="测试单只股票")
    parser.add_argument(
        '--backfill',
        nargs=2,
        metavar=('START_DATE', 'END_DATE'),
        help='回补历史事件，遍历个股逐日计算，例如: --backfill 2023-01-01 2026-06-12'
    )
    parser.add_argument('--limit', type=int, default=None, help='限制回补股票数量（用于测试）')
    args = parser.parse_args()

    if args.backfill:
        start_date = parse_date(args.backfill[0])
        end_date = parse_date(args.backfill[1])
        backfill_range(start_date, end_date, limit=args.limit)
    elif args.test:
        sel_date = parse_date(args.date) if args.date else date.today()
        test_single_stock(args.test, sel_date)
    else:
        sel_date = parse_date(args.date) if args.date else date.today()
        select_node_stocks(sel_date)
