#!/usr/bin/env python3
"""
BSM指标选股脚本（基于周线V型形态）

Purpose: 基于周线BBMACD V型形态选股
Inputs: stock_k_data (日线/周线K线数据)
Outputs: stock_selection_results (选股结果)
How to Run:
    python selection/selection_ana.py              # 当天
    python selection/selection_ana.py 2026-04-10  # 指定日期
Side Effects: 写入 stock_selection_results 表

================================================================================
【选股条件】

选股条件：周线V型形态 且 过去5天平均成交额 > 1亿
  - 核心条件：周线BBMACD V型形态（t_2 > t_1 and t > t_1）
  - 过滤条件：成交额 > 1亿（5日平均）

观察项（不参与选股，仅记录）：
  - 周线：DSA方向(dir)、VWAP、收盘价与VWAP偏差率
  - 日线：DSA方向(dir)、VWAP、收盘价与VWAP偏离度、PAVP/BBMACD指标

周线数据：
  - 使用周线数据计算BSM信号（bars=120）
  - 周线数据只在周五收盘后更新

保存字段：
  - 选股字段：weekly_reversal_buy（V型形态）
  - 周线观察项：weekly_dsa_dir, weekly_dsa_vwap, weekly_vwap_deviation
  - 日线观察项：daily_dsa_dir, daily_dsa_vwap, daily_vwap_deviation
  - PAVP字段（观察）：vah_1/2/3, val_1/2/3, poc_1/2/3, va_pos_01
  - BBMACD观察项：daily_bb_width_zscore, daily_vol_zscore, bbmacd_event
  - 批次号：batch_no（每10只一批）

批次号规则：
  - 每天从1开始编号，按处理顺序每10只股票为一批
  - 例：第1-10只 → batch_no=1，第11-20只 → batch_no=2

【选股日期】

选股日期只是标记，实际数据到"选股日期当天或之前最后一个交易日"：
  - 选股日期=2026-04-13(周一) → 数据到2026-04-13（当天是交易日）

周线数据每周更新：
  - 通过 python app/build_dataset.py --update --period w 每周更新周线数据

【保存逻辑】

按选股日期统一保存：
  - 所有股票使用传入的selection_date作为选股日期
  - 保存时先删旧数据再插新数据（幂等性）
================================================================================
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

try:
    from pytdx.hq import TdxHq_API
    from pytdx.params import TDXParams
except Exception:
    TdxHq_API = None
    TDXParams = None

# 从 bbmacd_viewer 导入核心计算逻辑（SSOT原则）
from features.bbmacd_viewer import compute_bbmacd
# 使用 merged_dsa_atr_rope_bb_factors 中的 DSA 计算逻辑
from features.merged_dsa_atr_rope_bb_factors import compute_dsa, DSAConfig
# 使用 PAVP 指标计算 VAH/VAL/POC（与vis页面一致）
from features.pavp_tv_fixed_params_factors import compute_pavp
from datasource.adj_factor import apply_adj_factor

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "stock_selection_results"

# 为了与 merged_dsa_atr_rope_bb_factors.py 的 HTML 输出口径保持一致，
# DSA 计算使用更长历史窗口，而不是仅取 120 根。
# HTML 默认 fetch-bars=1200，因此这里也对齐为 1200。
DSA_HISTORY_BARS = 1200
DSA_HTML_CFG = DSAConfig(prd=50, base_apt=20.0, use_adapt=False, vol_bias=10.0, atr_len=50)



MARKET_DATA_SOURCE = "db"
PYTDX_SERVERS: List[Tuple[str, int]] = [
    ("119.147.212.81", 7709),
    ("119.147.164.60", 7709),
    ("14.215.128.18", 7709),
    ("14.215.128.116", 7709),
    ("101.133.156.38", 7709),
    ("114.80.149.19", 7709),
    ("115.238.90.165", 7709),
    ("123.125.108.23", 7709),
    ("180.153.18.170", 7709),
    ("202.108.253.131", 7709),
]


def normalize_ts_code(ts_code: str) -> str:
    return str(ts_code).strip().upper().split('.')[0]


def normalize_freq(freq: str) -> str:
    """与 merged_dsa_atr_rope_bb_factors.py 保持一致"""
    f = str(freq).strip().lower()
    if f in {"d", "1d", "day", "daily", "101"}:
        return "d"
    if f in {"w", "1w", "week", "weekly"}:
        return "w"
    if f in {"m", "mo", "month", "monthly"}:
        return "mo"
    if f in {"60", "60m", "1h"}:
        return "60m"
    if f in {"30", "30m"}:
        return "30m"
    if f in {"15", "15m"}:
        return "15m"
    if f in {"5", "5m"}:
        return "5m"
    if f in {"1", "1m"}:
        return "1m"
    raise ValueError(f"不支持的频率: {freq}")


def _category_from_freq(freq: str) -> int:
    """与 merged_dsa_atr_rope_bb_factors.py 保持一致"""
    if TDXParams is None:
        raise RuntimeError("未安装 pytdx，无法使用 pytdx 行情源")
    f = normalize_freq(freq)
    return {
        "d": TDXParams.KLINE_TYPE_RI_K,
        "w": TDXParams.KLINE_TYPE_WEEKLY,
        "mo": TDXParams.KLINE_TYPE_MONTHLY,
        "60m": TDXParams.KLINE_TYPE_1HOUR,
        "30m": TDXParams.KLINE_TYPE_30MIN,
        "15m": TDXParams.KLINE_TYPE_15MIN,
        "5m": TDXParams.KLINE_TYPE_5MIN,
        "1m": TDXParams.KLINE_TYPE_1MIN,
    }[f]


def _market_from_symbol(symbol: str) -> int:
    """与 merged_dsa_atr_rope_bb_factors.py 保持一致"""
    symbol = normalize_ts_code(symbol)
    return 1 if str(symbol).startswith(("5", "6", "9")) else 0


def connect_pytdx() -> TdxHq_API:
    """与 merged_dsa_atr_rope_bb_factors.py 保持一致"""
    if TdxHq_API is None:
        raise RuntimeError("未安装 pytdx，无法使用 pytdx 行情源")
    errors: List[str] = []
    for host, port in PYTDX_SERVERS:
        try:
            api = TdxHq_API(raise_exception=True, auto_retry=True)
            if api.connect(host, port):
                return api
        except Exception as exc:
            errors.append(f"{host}:{port} {exc}")
    raise RuntimeError("pytdx 连接失败: " + "; ".join(errors[-5:]))


def _fetch_kline_pytdx_raw(symbol: str, freq: str, count: int) -> pd.DataFrame:
    """直接复用 merged_dsa_atr_rope_bb_factors.py 的抓数逻辑，不加任何额外口径。"""
    api = connect_pytdx()
    symbol = normalize_ts_code(symbol)
    try:
        cat = _category_from_freq(freq)
        mkt = _market_from_symbol(symbol)
        size = 800
        frames: List[pd.DataFrame] = []
        start = 0
        target = max(int(count), 300)
        while start < target + size:
            recs = api.get_security_bars(cat, mkt, symbol, start, size)
            if not recs:
                break
            d = pd.DataFrame(recs)
            if "datetime" in d.columns:
                d["datetime"] = pd.to_datetime(d["datetime"]).dt.tz_localize(None)
            else:
                d["datetime"] = pd.to_datetime(
                    d[["year", "month", "day", "hour", "minute"]].astype(int)
                )
            if "vol" in d.columns:
                d = d.rename(columns={"vol": "volume"})
            if "amount" not in d.columns:
                d["amount"] = np.nan
            keep = ["datetime", "open", "high", "low", "close", "volume", "amount"]
            frames.append(d[keep].sort_values("datetime"))
            if len(recs) < size:
                break
            start += size
        if not frames:
            raise RuntimeError("pytdx 无数据")
        out = (
            pd.concat(frames)
            .sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
            .tail(count)
            .set_index("datetime")
        )
        return out.astype(float)
    finally:
        try:
            api.disconnect()
        except Exception:
            pass


def fetch_kline_pytdx(ts_code: str, freq: str, count: int, end_date: Optional[date] = None) -> pd.DataFrame:
    """
    先按 merged_dsa_atr_rope_bb_factors.py 的原始逻辑抓取，再在外层按 end_date 截断。
    这样当 end_date 为空时，结果与指标脚本保持完全一致。
    """
    raw = _fetch_kline_pytdx_raw(ts_code, freq, count if end_date is None else max(int(count), 3000))
    if end_date is not None:
        raw = raw[raw.index.date <= end_date]
    raw = raw.tail(count).copy()
    raw.index.name = "bar_time"
    if "amount" in raw.columns:
        raw = raw.drop(columns=["amount"])
    return raw


def get_kline_data_db(ts_code: str, freq: str, bars: int = 120, end_date: Optional[date] = None) -> pd.DataFrame:
    """从数据库获取K线数据（默认120根）"""
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
        suffix = '.SH' if symbol.startswith(('6', '9')) else '.SZ'
        df = apply_adj_factor(df, f'{symbol}{suffix}', freq=freq)
    return df


def volume_zscore(vol: pd.Series, win: int = 20, position: int = -1) -> float:
    """
    计算成交量Z-Score
    z = (vol - SMA(vol, win)) / STDEV(vol, win)
    使用 population std (ddof=0) 对齐 TradingView
    
    Args:
        vol: 成交量序列
        win: 窗口期，默认20
        position: 目标位置，默认-1表示最后一个bar
    """
    mu = vol.rolling(win, min_periods=win).mean()
    sd = vol.rolling(win, min_periods=win).std(ddof=0)
    
    # 处理正向索引
    if position < 0:
        idx = position
    else:
        idx = position
    
    if idx >= len(vol) or abs(idx) > len(vol):
        return None
    
    # 处理正向索引
    if position >= 0:
        actual_idx = position
    else:
        actual_idx = len(vol) + position
    
    if actual_idx < win - 1:
        return None
        
    mu_val = mu.iloc[actual_idx]
    sd_val = sd.iloc[actual_idx]
    
    if sd_val == 0 or pd.isna(sd_val):
        return None
    z = (vol.iloc[actual_idx] - mu_val) / sd_val
    return float(z)


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

    # 计算每日成交额 = 成交量 × 收盘价
    recent_df = df.tail(days)
    daily_amount = recent_df['volume'] * recent_df['close']
    avg_amount = daily_amount.mean()

    return avg_amount >= min_amount


def compute_change_pct(daily_df: pd.DataFrame) -> float:
    """
    计算选股日当天的涨跌幅
    涨跌幅 = (当日收盘价 - 前一日收盘价) / 前一日收盘价 * 100
    """
    if len(daily_df) < 2:
        return None
    close_today = daily_df['close'].iloc[-1]
    close_yesterday = daily_df['close'].iloc[-2]
    if close_yesterday == 0:
        return None
    change = float(close_today - close_yesterday) / float(close_yesterday) * 100
    return round(change, 2)


def get_stock_name(ts_code: str) -> str:
    """从stock_pools表获取股票名称"""
    sql = text("SELECT name FROM stock_pools WHERE ts_code = :ts_code LIMIT 1")
    with engine.connect() as conn:
        result = conn.execute(sql, {'ts_code': ts_code})
        row = result.fetchone()
        return row[0] if row else ''


def get_kline_data(ts_code: str, freq: str, bars: int = 120, end_date: Optional[date] = None) -> pd.DataFrame:
    """获取K线数据（根据 MARKET_DATA_SOURCE 选择数据源）

    Args:
        ts_code: 股票代码
        freq: 频率 ('d'日线, 'w'周线)
        bars: 获取的K线数量
        end_date: 截止日期，只获取该日期及之前的K线
    """
    if MARKET_DATA_SOURCE == "pytdx":
        return fetch_kline_pytdx(ts_code, freq, bars, end_date)
    else:
        return get_kline_data_db(ts_code, freq, bars, end_date)


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


def batch_get_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    """批量获取股票名称（一次查询）"""
    if not ts_codes:
        return {}
    placeholders = ', '.join([f"'{c}'" for c in ts_codes])
    sql = text(f"SELECT ts_code, name FROM stock_pools WHERE ts_code IN ({placeholders})")
    with engine.connect() as conn:
        result = conn.execute(sql)
        return {row[0]: row[1] for row in result}


def analyze_pavp_vah(fixed_segments) -> Dict:
    """
    分析PAVP最近3次震荡段的VAH/VAL/POC

    Args:
        fixed_segments: compute_pavp返回的fixed_segments列表（含last_dev）

    Returns:
        {
            'cons_count': int,
            'vahs': List[float],
            'vals': List[float],
            'pocs': List[float],
            'poc_bars': int,  # 第1段到第3段的bar数量
            'is_ascending': bool,
            'trend_desc': str,
        }
    """
    if not fixed_segments or len(fixed_segments) < 3:
        return {
            'cons_count': len(fixed_segments) if fixed_segments else 0,
            'vahs': [],
            'vals': [],
            'pocs': [],
            'poc_bars': 0,
            'is_ascending': False,
            'trend_desc': '',
        }

    recent = fixed_segments[-3:]
    vahs = [seg.vah_price for seg in recent]
    vals = [seg.val_price for seg in recent]
    pocs = [seg.poc_price for seg in recent]
    poc_bars = recent[2].confirm_index - recent[0].confirm_index

    is_ascending = False
    trend_desc = ''

    if len(vahs) >= 2:
        trends = []
        valid = True
        for i in range(1, len(vahs)):
            if pd.isna(vahs[i]) or pd.isna(vahs[i - 1]):
                valid = False
                break
            if vahs[i] > vahs[i - 1]:
                trends.append('up')
            elif vahs[i] < vahs[i - 1]:
                trends.append('down')
            else:
                trends.append('flat')

        if valid and len(trends) >= 2:
            trend_desc = '_'.join(trends)
            is_ascending = all(t == 'up' for t in trends)

    return {
        'cons_count': len(recent),
        'vahs': vahs,
        'vals': vals,
        'pocs': pocs,
        'poc_bars': poc_bars,
        'is_ascending': is_ascending,
        'trend_desc': trend_desc,
    }


def compute_poc_avg_return(pocs: List[float], poc_bars: int) -> float:
    """
    计算三次震荡POC的平均日收益率

    公式：((poc_3 - poc_1) / poc_1) / bar数量

    Args:
        pocs: 三次震荡的POC价格列表 [poc_1, poc_2, poc_3]（时间顺序）
        poc_bars: 第1段到第3段的bar数量

    Returns:
        平均日收益率（float），无效时返回 nan
    """
    if len(pocs) < 3 or poc_bars <= 0:
        return np.nan

    poc_1, poc_3 = pocs[0], pocs[2]

    if pd.isna(poc_1) or pd.isna(poc_3) or poc_1 <= 0:
        return np.nan

    total_return = (poc_3 - poc_1) / poc_1
    return float(total_return / poc_bars)


def get_html_consistent_dsa_snapshot(
    ts_code: str,
    freq: str,
    selection_date: Optional[date],
    bars: int = DSA_HISTORY_BARS,
    cfg: DSAConfig = DSA_HTML_CFG,
) -> Dict[str, Optional[float]]:
    """
    按 HTML 指标脚本口径提取 DSA 最新快照。

    对齐点：
    1. 使用同一个 compute_dsa 核心函数；
    2. 使用与 HTML 相同的默认 DSA 参数；
    3. 使用更长历史窗口（默认 1200 根）以对齐 HTML 的 fetch-bars。

    Returns:
        {
            'dir': int|None,
            'vwap': float|None,
            'bar_time': Timestamp|None,
            'bars_used': int,
        }
    """
    df = get_kline_data(ts_code, freq, bars=bars, end_date=selection_date)
    if df.empty:
        return {'dir': None, 'vwap': None, 'bar_time': None, 'bars_used': 0}

    try:
        dsa_df, _, _ = compute_dsa(df, cfg)
    except Exception:
        return {'dir': None, 'vwap': None, 'bar_time': df.index[-1], 'bars_used': len(df)}

    if dsa_df.empty:
        return {'dir': None, 'vwap': None, 'bar_time': df.index[-1], 'bars_used': len(df)}

    latest = dsa_df.iloc[-1]
    raw_dir = latest.get('DSA_DIR')
    raw_vwap = latest.get('DSA_VWAP')

    dir_value = None if pd.isna(raw_dir) else int(raw_dir)
    vwap_value = None if pd.isna(raw_vwap) else float(raw_vwap)
    return {
        'dir': dir_value,
        'vwap': vwap_value,
        'bar_time': dsa_df.index[-1],
        'bars_used': len(df),
    }


def compute_dsa_state_stats(
    ts_code: str,
    selection_date: Optional[date],
    freq: str = 'd',
    bars: int = DSA_HISTORY_BARS,
    cfg: DSAConfig = DSA_HTML_CFG,
) -> Dict[str, Optional[float]]:
    """
    计算DSA状态持续时间和偏离率统计。

    计算内容：
    1. 当前DSA dir值
    2. 当前状态持续了多少个bar（从最新bar向前，直到dir变化）
    3. 持续状态内股价与VWAP的平均偏离率
    4. 持续状态内股价与VWAP偏离率的方差

    Returns:
        {
            'dir': int|None,                    # DSA方向
            'duration': int,                     # 状态持续bar数
            'avg_deviation': float|None,         # 平均偏离率
            'deviation_variance': float|None,    # 偏离率方差
        }
    """
    df = get_kline_data(ts_code, freq, bars=bars, end_date=selection_date)
    if df.empty:
        return {'dir': None, 'duration': 0, 'avg_deviation': None, 'deviation_variance': None}

    try:
        dsa_df, _, _ = compute_dsa(df, cfg)
    except Exception:
        return {'dir': None, 'duration': 0, 'avg_deviation': None, 'deviation_variance': None}

    if dsa_df.empty:
        return {'dir': None, 'duration': 0, 'avg_deviation': None, 'deviation_variance': None}

    # 获取最新dir值
    latest_dir = dsa_df['DSA_DIR'].iloc[-1]
    if pd.isna(latest_dir):
        return {'dir': None, 'duration': 0, 'avg_deviation': None, 'deviation_variance': None}

    latest_dir = int(latest_dir)

    # 从最新bar向前遍历，找到dir变化的点
    duration = 0
    deviations = []

    # 从原始数据获取close价格（dsa_df可能没有close列）
    df_aligned = df.loc[dsa_df.index]

    for i in range(len(dsa_df) - 1, -1, -1):
        current_dir = dsa_df['DSA_DIR'].iloc[i]
        if pd.isna(current_dir) or int(current_dir) != latest_dir:
            break

        # 计算当前bar的偏离率
        close = df_aligned['close'].iloc[i]
        vwap = dsa_df['DSA_VWAP'].iloc[i]

        if not pd.isna(close) and not pd.isna(vwap) and vwap != 0:
            deviation = (close - vwap) / vwap
            deviations.append(deviation)

        duration += 1

    # 计算平均偏离率和方差
    if deviations:
        avg_deviation = sum(deviations) / len(deviations)
        if len(deviations) > 1:
            # 计算方差
            variance = sum((d - avg_deviation) ** 2 for d in deviations) / len(deviations)
        else:
            variance = 0.0
    else:
        avg_deviation = None
        variance = None

    return {
        'dir': latest_dir,
        'duration': duration,
        'avg_deviation': avg_deviation,
        'deviation_variance': variance,
    }


def process_stock_pavp(ts_code: str, selection_date: date, has_daily: bool) -> Optional[Dict]:
    """
    处理单只股票的V型形态选股

    选股逻辑：
        - 条件1：周线BBMACD出现V型形态
        - 条件2：过去5天平均成交额 > 1亿
        - DSA dir和收盘价vs VWAP偏差作为观察项，不参与选股

    核心字段：
        - weekly_reversal_buy: 周线V型形态（选股条件）
        - weekly_bb_width_zscore: 周线布林带宽度Z-Score

    观察字段（周线）：
        - weekly_dsa_dir: 周线DSA方向
        - weekly_dsa_vwap: 周线DSA VWAP
        - weekly_vwap_deviation: 周线收盘价与VWAP偏差率

    观察字段（日线）：
        - vah_1/2/3, val_1/2/3, poc_1/2/3: PAVP成交量分布
        - va_pos_01: 当前收盘价在VA中的位置
        - daily_bb_width_zscore, daily_vol_zscore: 日线Z-Score
        - daily_dsa_dir: 日线DSA方向
        - daily_dsa_vwap: 日线DSA VWAP
        - daily_vwap_deviation: 日线VWAP偏离度
        - bbmacd_event: BBMACD事件

    Returns: 信号字典，如果满足选股条件则返回结果，否则返回None
    """
    if not has_daily:
        return None

    # 首先检测周线V型形态（核心选股条件）
    bsm_signals = detect_bsm_signals(ts_code, selection_date)

    # 核心选股条件：周线V型形态
    weekly_reversal_buy = bsm_signals.get('weekly_reversal_buy', False)

    if not weekly_reversal_buy:
        return None

    # 获取日线数据用于成交额过滤和观察项计算
    daily_df = get_kline_data_db(ts_code, 'd', bars=250)
    if daily_df.empty:
        return None
    daily_df = daily_df.tail(250).copy()
    if len(daily_df) < 60:
        return None

    # 检查成交额过滤条件：过去5天平均成交额 > 1亿
    if not check_volume_filter(daily_df, days=5, min_amount=100_000_000):
        return None

    # 计算PAVP指标（观察项，不参与选股）
    pavp_df, fixed_segments, last_dev = compute_pavp(daily_df)

    # 过滤fixed_segments中bar数过小的噪音段
    MIN_SEGMENT_BARS = 5
    fixed_segments = [s for s in fixed_segments if (s.end_index - s.start_index) >= MIN_SEGMENT_BARS]
    all_segments = list(fixed_segments) if fixed_segments else []
    if last_dev is not None:
        all_segments.append(last_dev)

    # 分析震荡段（仅用于观察）
    cons_analysis = analyze_pavp_vah(all_segments) if len(all_segments) >= 3 else {
        'vahs': [], 'vals': [], 'pocs': [], 'poc_bars': 0, 'trend_desc': ''
    }
    vahs = cons_analysis.get('vahs', [])
    vals = cons_analysis.get('vals', [])
    pocs = cons_analysis.get('pocs', [])

    # 计算POC平均日收益率（观察项）
    poc_avg_ret = compute_poc_avg_return(pocs, cons_analysis.get('poc_bars', 0))

    # 计算日线BBMACD和DSA（观察项）
    daily_bbmacd = compute_bbmacd(daily_df)
    dsa_snapshot = get_html_consistent_dsa_snapshot(
        ts_code=ts_code,
        freq='d',
        selection_date=selection_date,
    )

    # 计算DSA状态持续时间和偏离率统计
    dsa_stats = compute_dsa_state_stats(
        ts_code=ts_code,
        selection_date=selection_date,
        freq='d',
    )

    bar_time = daily_df.index[-1]
    bar_close = daily_df['close'].iloc[-1]
    current_vwap = dsa_snapshot['vwap']

    signal_vwap_deviation = None
    if current_vwap is not None and current_vwap != 0:
        signal_vwap_deviation = (bar_close - current_vwap) / current_vwap

    # 检测BBMACD事件（最新bar）- 4种类型
    bbmacd_event = "无"
    if len(daily_bbmacd) > 0:
        last_idx = -1
        if daily_bbmacd['compra'].iloc[last_idx]:
            bbmacd_event = "上穿上轨"
        elif daily_bbmacd['cross_down_upper'].iloc[last_idx]:
            bbmacd_event = "下穿上轨"
        elif daily_bbmacd['cross_up_lower'].iloc[last_idx]:
            bbmacd_event = "上穿下轨"
        elif daily_bbmacd['venta'].iloc[last_idx]:
            bbmacd_event = "下穿下轨"

    return {
        'ts_code': ts_code,
        # BSM核心选股字段（周线）
        'weekly_reversal_buy': weekly_reversal_buy,
        'weekly_breakout_buy': False,  # 已废弃，始终为False
        'weekly_bb_width_zscore': bsm_signals.get('weekly_bb_width_zscore'),
        'weekly_vol_zscore': bsm_signals.get('weekly_vol_zscore'),
        # 周线DSA观察项
        'weekly_dsa_dir': bsm_signals.get('weekly_dsa_dir'),
        'weekly_dsa_vwap': bsm_signals.get('weekly_dsa_vwap'),
        'weekly_vwap_deviation': bsm_signals.get('weekly_vwap_deviation'),
        # PAVP字段（观察项，日线）
        'vah_1': vahs[0] if len(vahs) > 0 else None,
        'vah_2': vahs[1] if len(vahs) > 1 else None,
        'vah_3': vahs[2] if len(vahs) > 2 else None,
        'val_1': vals[0] if len(vals) > 0 else None,
        'val_2': vals[1] if len(vals) > 1 else None,
        'val_3': vals[2] if len(vals) > 2 else None,
        'poc_1': pocs[0] if len(pocs) > 0 else None,
        'poc_2': pocs[1] if len(pocs) > 1 else None,
        'poc_3': pocs[2] if len(pocs) > 2 else None,
        'poc_bars': cons_analysis.get('poc_bars', 0),
        'poc_avg_ret': poc_avg_ret,
        'va_pos_01': float(pavp_df['va_pos_01'].iloc[-1]) if 'va_pos_01' in pavp_df.columns and len(pavp_df) > 0 else None,
        # BBMACD观察项（日线）
        'bbmacd_event': bbmacd_event,
        'daily_bb_width_zscore': daily_bbmacd['bb_width_zscore'].iloc[-1] if len(daily_bbmacd) > 0 else None,
        'daily_vol_zscore': volume_zscore(daily_df['volume'], win=20),
        # DSA观察项（日线）
        'daily_dsa_dir': dsa_snapshot['dir'],
        'daily_dsa_vwap': dsa_snapshot['vwap'],
        'daily_dsa_bars_used': dsa_snapshot['bars_used'],
        'daily_dsa_duration': dsa_stats['duration'],
        'daily_dsa_avg_deviation': dsa_stats['avg_deviation'],
        'daily_dsa_deviation_variance': dsa_stats['deviation_variance'],
        'daily_vwap_deviation': signal_vwap_deviation,
        'daily_bar_time': bar_time,
        'signal_date': bar_time,
        'change_pct': compute_change_pct(daily_df),
        # 保留pavp_selected作为观察标记（不再用于选股）
        'pavp_selected': False,
        'pavp_trend': cons_analysis.get('trend_desc', ''),
    }


def detect_bsm_signals(ts_code: str, selection_date: Optional[date] = None) -> Dict:
    """
    检测BSM信号（仅V型形态选股，DSA为观察项）

    选股条件：周线BBMACD出现V型形态（t_2 > t_1 and t > t_1）
    观察项：周线DSA dir、VWAP、收盘价与VWAP偏差率

    Args:
        ts_code: 股票代码
        selection_date: 选股日期，用于获取该日期之前的数据
    Returns: 信号字典（包含V型形态信号、Z-Score值、DSA观察项和实际使用的行情日期）
    """
    signals = {
        'weekly_reversal_buy': False,     # 周线V型形态（选股条件）
        'weekly_breakout_buy': False,     # 已废弃，始终为False
        'weekly_bb_width_zscore': None,   # 周线布林带宽度Z-Score
        'weekly_vol_zscore': None,        # 周线成交量Z-Score
        'daily_bar_time': None,           # 实际使用的日线日期（用于成交额过滤）
        'weekly_bar_time': None,          # 实际使用的周线日期
        # 观察项：周线DSA
        'weekly_dsa_dir': None,           # 周线DSA方向
        'weekly_dsa_vwap': None,          # 周线DSA VWAP值
        'weekly_vwap_deviation': None,    # 周线收盘价与VWAP偏差率
    }

    # 获取日线数据用于成交额过滤
    daily_df = get_kline_data(ts_code, 'd', bars=120, end_date=selection_date)
    if len(daily_df) >= 5:
        signals['daily_bar_time'] = daily_df.index[-1]

    # 周线信号（只有存在周线数据时才计算）
    if has_weekly_data_for_date(selection_date):
        weekly_df = get_kline_data(ts_code, 'w', bars=120, end_date=selection_date)
        if len(weekly_df) >= 26:
            weekly_bbmacd = compute_bbmacd(weekly_df)
            weekly_close = weekly_df['close'].iloc[-1]

            # 记录实际使用的周线日期
            signals['weekly_bar_time'] = weekly_df.index[-1]

            # 周线布林带宽度Z-Score
            signals['weekly_bb_width_zscore'] = weekly_bbmacd['bb_width_zscore'].iloc[-1]

            # 周线成交量Z-Score
            signals['weekly_vol_zscore'] = volume_zscore(weekly_df['volume'], win=20)

            # 获取周线DSA数据（观察项）
            dsa_snapshot = get_html_consistent_dsa_snapshot(
                ts_code=ts_code,
                freq='w',
                selection_date=selection_date,
            )
            current_dir = dsa_snapshot['dir']
            current_vwap = dsa_snapshot['vwap']

            # 记录周线DSA观察项
            signals['weekly_dsa_dir'] = current_dir
            signals['weekly_dsa_vwap'] = current_vwap
            if current_vwap is not None and current_vwap != 0:
                signals['weekly_vwap_deviation'] = (weekly_close - current_vwap) / current_vwap

            # V型形态检测（唯一选股条件）
            bbmacd_vals = weekly_bbmacd['bbmacd'].values
            has_v_shape = False
            if len(bbmacd_vals) >= 3:
                t_2 = bbmacd_vals[-3]
                t_1 = bbmacd_vals[-2]
                t = bbmacd_vals[-1]
                has_v_shape = (t_2 > t_1) and (t > t_1)

            if has_v_shape:
                signals['weekly_reversal_buy'] = True

    return signals


def ensure_table_exists(engine):
    """确保选股结果表存在（添加ATR Rope字段、BSM字段和Z-Score字段）"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS stock_selection_results (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        signal_date DATE,
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
        -- PAVP选股字段
        pavp_selected BOOLEAN DEFAULT FALSE,
        pavp_trend VARCHAR(20),
        vah_1 FLOAT,
        vah_2 FLOAT,
        vah_3 FLOAT,
        val_1 FLOAT,
        val_2 FLOAT,
        val_3 FLOAT,
        -- POC相关字段
        poc_1 FLOAT,
        poc_2 FLOAT,
        poc_3 FLOAT,
        poc_bars INT,
        poc_avg_ret FLOAT,
        -- BSM指标字段（BOOLEAN类型，改为观察项）
        daily_reversal_buy BOOLEAN DEFAULT FALSE,
        daily_breakout_buy BOOLEAN DEFAULT FALSE,
        weekly_reversal_buy BOOLEAN DEFAULT FALSE,
        weekly_breakout_buy BOOLEAN DEFAULT FALSE,
        va_pos_01 FLOAT,
        -- BBMACD事件字段
        bbmacd_event VARCHAR(20),
        -- 布林带宽度Z-Score字段（日线）
        daily_bb_width_zscore FLOAT,
        -- 成交量Z-Score字段（日线）
        daily_vol_zscore FLOAT,
        -- VWAP偏移比率字段（日线）
        daily_vwap_deviation FLOAT,
        -- 批次号字段
        batch_no INT,
        -- 选股日涨跌幅字段
        change_pct FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_selection_date ON stock_selection_results(selection_date);
    CREATE INDEX IF NOT EXISTS idx_signal_date ON stock_selection_results(signal_date);
    CREATE INDEX IF NOT EXISTS idx_selection_ts_code ON stock_selection_results(ts_code);
    CREATE INDEX IF NOT EXISTS idx_selection_report_date ON stock_selection_results(report_date);
    CREATE INDEX IF NOT EXISTS idx_selection_margin_score ON stock_selection_results(margin_score);
    -- PAVP字段索引
    CREATE INDEX IF NOT EXISTS idx_selection_pavp_selected ON stock_selection_results(pavp_selected);
    CREATE INDEX IF NOT EXISTS idx_selection_poc_avg_ret ON stock_selection_results(poc_avg_ret);
    -- BSM字段索引
    CREATE INDEX IF NOT EXISTS idx_selection_daily_reversal ON stock_selection_results(daily_reversal_buy);
    CREATE INDEX IF NOT EXISTS idx_selection_daily_breakout ON stock_selection_results(daily_breakout_buy);
    CREATE INDEX IF NOT EXISTS idx_selection_weekly_reversal ON stock_selection_results(weekly_reversal_buy);
    CREATE INDEX IF NOT EXISTS idx_selection_weekly_breakout ON stock_selection_results(weekly_breakout_buy);
    CREATE INDEX IF NOT EXISTS idx_selection_batch_no ON stock_selection_results(batch_no);
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()

        # 添加缺失的列（如果表已存在）
        # PAVP字段
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS pavp_selected BOOLEAN DEFAULT FALSE"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS pavp_trend VARCHAR(20)"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS vah_1 FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS vah_2 FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS vah_3 FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS val_1 FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS val_2 FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS val_3 FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS poc_1 FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS poc_2 FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS poc_3 FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS poc_bars INT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS poc_avg_ret FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS va_pos_01 FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS bbmacd_event VARCHAR(20)"))
        # 原有字段
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS signal_date DATE"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS change_pct FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS batch_no INT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS daily_bb_width_zscore FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS daily_vol_zscore FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS daily_vwap_deviation FLOAT"))
        # DSA状态统计字段
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS daily_dsa_duration INT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS daily_dsa_avg_deviation FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS daily_dsa_deviation_variance FLOAT"))
        # 周线DSA观察项
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS weekly_dsa_dir INT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS weekly_dsa_vwap FLOAT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS weekly_vwap_deviation FLOAT"))
        # 日线DSA观察项（已计算但此前未持久化）
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS daily_dsa_dir INT"))
        conn.execute(text("ALTER TABLE stock_selection_results ADD COLUMN IF NOT EXISTS daily_dsa_vwap FLOAT"))
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
        signal_date_raw = row.get('signal_date', None)
        if signal_date_raw is not None and not pd.isna(signal_date_raw):
            if hasattr(signal_date_raw, 'date'):
                signal_date_val = signal_date_raw.date()
            else:
                signal_date_val = signal_date_raw
        else:
            signal_date_val = None

        record = {
            'selection_date': sel_date,
            'signal_date': signal_date_val,
            'ts_code': row['ts_code'],
            'stock_name': row.get('stock_name', '') or '',
            'report_date': row.get('report_date', '') or '',
            'batch_no': int(row['batch_no']) if pd.notna(row.get('batch_no')) else None,
            # PAVP字段
            'pavp_selected': bool(row.get('pavp_selected', False)),
            'pavp_trend': row.get('pavp_trend', '') or '',
            'vah_1': float(row['vah_1']) if pd.notna(row.get('vah_1')) else None,
            'vah_2': float(row['vah_2']) if pd.notna(row.get('vah_2')) else None,
            'vah_3': float(row['vah_3']) if pd.notna(row.get('vah_3')) else None,
            'val_1': float(row['val_1']) if pd.notna(row.get('val_1')) else None,
            'val_2': float(row['val_2']) if pd.notna(row.get('val_2')) else None,
            'val_3': float(row['val_3']) if pd.notna(row.get('val_3')) else None,
            'va_pos_01': float(row['va_pos_01']) if pd.notna(row.get('va_pos_01')) else None,
            'bbmacd_event': row.get('bbmacd_event', '无') or '无',
            # POC字段
            'poc_1': float(row['poc_1']) if pd.notna(row.get('poc_1')) else None,
            'poc_2': float(row['poc_2']) if pd.notna(row.get('poc_2')) else None,
            'poc_3': float(row['poc_3']) if pd.notna(row.get('poc_3')) else None,
            'poc_bars': int(row['poc_bars']) if pd.notna(row.get('poc_bars')) else None,
            'poc_avg_ret': float(row['poc_avg_ret']) if pd.notna(row.get('poc_avg_ret')) else None,
            # BSM字段（观察项）
            'daily_reversal_buy': bool(row.get('daily_reversal_buy', False)),
            'daily_breakout_buy': bool(row.get('daily_breakout_buy', False)),
            'weekly_reversal_buy': bool(row.get('weekly_reversal_buy', False)),
            'weekly_breakout_buy': bool(row.get('weekly_breakout_buy', False)),
            # Z-Score字段（周线）
            'weekly_bb_width_zscore': float(row['weekly_bb_width_zscore']) if pd.notna(row.get('weekly_bb_width_zscore')) else None,
            'weekly_vol_zscore': float(row['weekly_vol_zscore']) if pd.notna(row.get('weekly_vol_zscore')) else None,
            # Z-Score字段（日线）
            'daily_bb_width_zscore': float(row['daily_bb_width_zscore']) if pd.notna(row.get('daily_bb_width_zscore')) else None,
            'daily_vol_zscore': float(row['daily_vol_zscore']) if pd.notna(row.get('daily_vol_zscore')) else None,
            'daily_vwap_deviation': float(row['daily_vwap_deviation']) if pd.notna(row.get('daily_vwap_deviation')) else None,
            'change_pct': float(row['change_pct']) if pd.notna(row.get('change_pct')) else None,
            # DSA状态统计字段（日线）
            'daily_dsa_duration': int(row['daily_dsa_duration']) if pd.notna(row.get('daily_dsa_duration')) else None,
            'daily_dsa_avg_deviation': float(row['daily_dsa_avg_deviation']) if pd.notna(row.get('daily_dsa_avg_deviation')) else None,
            'daily_dsa_deviation_variance': float(row['daily_dsa_deviation_variance']) if pd.notna(row.get('daily_dsa_deviation_variance')) else None,
            # 周线DSA观察项
            'weekly_dsa_dir': int(row['weekly_dsa_dir']) if pd.notna(row.get('weekly_dsa_dir')) else None,
            'weekly_dsa_vwap': float(row['weekly_dsa_vwap']) if pd.notna(row.get('weekly_dsa_vwap')) else None,
            'weekly_vwap_deviation': float(row['weekly_vwap_deviation']) if pd.notna(row.get('weekly_vwap_deviation')) else None,
            # 日线DSA观察项
            'daily_dsa_dir': int(row['daily_dsa_dir']) if pd.notna(row.get('daily_dsa_dir')) else None,
            'daily_dsa_vwap': float(row['daily_dsa_vwap']) if pd.notna(row.get('daily_dsa_vwap')) else None,
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
    根据PAVP指标选出满足条件的股票（基于VAH抬升趋势）

    Args:
        selection_date: 选股日期，默认为当天
        save_to_db: 是否保存到数据库
    """
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("选股条件（周线V型形态策略）：")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  行情数据源: {MARKET_DATA_SOURCE}")
    print(f"  核心条件: 周线BBMACD V型形态")
    print(f"  过滤条件: 过去5天平均成交额 > 1亿")
    print(f"  观察项: DSA dir/收盘价vs VWAP偏差（周线+日线）、PAVP/BBMACD日线指标")
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

        has_daily = True
        print(f"  使用周线数据检测BSM信号")

        filtered_results = []

        # 使用tqdm进度条，固定底部显示，动态信息另行输出
        for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="BSM选股", unit="只"):
            ts_code = row['ts_code']

            result = process_stock_pavp(ts_code, selection_date, has_daily)
            if result:
                filtered_results.append(result)

        result_df = pd.DataFrame(filtered_results)

        if not result_df.empty:
            stock_names = batch_get_stock_names(result_df['ts_code'].tolist())
            result_df['stock_name'] = result_df['ts_code'].map(stock_names)

            result_df['batch_no'] = (result_df.index // 10) + 1

        print("\n" + "=" * 80)
        print("选股结果汇总：")
        print("=" * 80)
        print(f"V型形态筛选后: {len(result_df)} 只")

        if not result_df.empty:
            print(f"\n选股统计：")
            print(f"  周线V型形态: {result_df['weekly_reversal_buy'].sum()} 只")

            # 显示周线DSA观察项统计
            valid_w_dir = result_df['weekly_dsa_dir'].dropna()
            if len(valid_w_dir) > 0:
                dir_counts = valid_w_dir.value_counts().sort_index()
                dir_desc = ', '.join([f"dir={k}:{v}只" for k, v in dir_counts.items()])
                print(f"\n周线DSA dir分布（观察项）：{dir_desc}")

            valid_w_dev = result_df['weekly_vwap_deviation'].dropna()
            if len(valid_w_dev) > 0:
                print(f"周线收盘价vs VWAP偏差（观察项）：")
                print(f"  平均值: {valid_w_dev.mean():.4%}")
                print(f"  中位数: {valid_w_dev.median():.4%}")

            # 显示日线DSA观察项统计
            valid_d_dir = result_df['daily_dsa_dir'].dropna()
            if len(valid_d_dir) > 0:
                dir_counts = valid_d_dir.value_counts().sort_index()
                dir_desc = ', '.join([f"dir={k}:{v}只" for k, v in dir_counts.items()])
                print(f"\n日线DSA dir分布（观察项）：{dir_desc}")

            valid_d_dev = result_df['daily_vwap_deviation'].dropna()
            if len(valid_d_dev) > 0:
                print(f"日线收盘价vs VWAP偏差（观察项）：")
                print(f"  平均值: {valid_d_dev.mean():.4%}")
                print(f"  中位数: {valid_d_dev.median():.4%}")

            # 显示POC平均日收益率统计（观察项）
            valid_ret = result_df['poc_avg_ret'].dropna()
            if len(valid_ret) > 0:
                print(f"\nPOC平均日收益率统计（观察项）：")
                print(f"  平均值: {valid_ret.mean():.6f}")
                print(f"  中位数: {valid_ret.median():.6f}")
                print(f"  最小值: {valid_ret.min():.6f}")
                print(f"  最大值: {valid_ret.max():.6f}")

            batch_count = result_df['batch_no'].max()
            print(f"\n批次信息：共 {batch_count} 批，每批10只股票")

            print("\n" + "=" * 80)
            print("前20名股票：")
            print("=" * 80)
            display_cols = ['ts_code', 'batch_no', 'weekly_dsa_dir', 'weekly_vwap_deviation', 'daily_dsa_dir', 'daily_vwap_deviation', 'weekly_bb_width_zscore']
            print(result_df[display_cols].head(20).to_string(index=False))
        
        # 保存到数据库
        if save_to_db:
            print("\n" + "-" * 80)
            print("保存到数据库...")
            saved_count = save_to_database_by_weekly_date(result_df, fallback_date=selection_date)
            print("-" * 80)

        return result_df
    else:
        print("\n未找到符合条件的股票")
        return pd.DataFrame()



def test_dsa_last_bar(ts_code: str, selection_date: date, freq: str = 'w', bars: int = DSA_HISTORY_BARS) -> int:
    """测试指定股票最后一个 bar 的 DSA 快照，便于与 HTML 对账。"""
    df = get_kline_data(ts_code, freq, bars=bars, end_date=selection_date)
    if df.empty:
        print(f"未获取到行情数据: ts_code={ts_code}, freq={freq}, selection_date={selection_date}")
        return 1

    snapshot = get_html_consistent_dsa_snapshot(
        ts_code=ts_code,
        freq=freq,
        selection_date=selection_date,
        bars=bars,
    )

    latest_close = float(df['close'].iloc[-1]) if pd.notna(df['close'].iloc[-1]) else None
    latest_bar_time = df.index[-1]

    print("\n" + "=" * 80)
    print("DSA 最后一个 bar 测试")
    print("=" * 80)
    print(f"ts_code           : {ts_code}")
    print(f"freq              : {freq}")
    print(f"market_data_source: {MARKET_DATA_SOURCE}")
    print(f"selection_date    : {selection_date}")
    print(f"latest_bar_time   : {latest_bar_time}")
    print(f"bars_used         : {snapshot['bars_used']}")
    print(f"latest_close      : {latest_close}")
    print(f"latest_dsa_dir    : {snapshot['dir']}")
    print(f"latest_dsa_vwap   : {snapshot['vwap']}")
    if snapshot['vwap'] is not None and latest_close is not None and snapshot['vwap'] != 0:
        deviation = (latest_close - snapshot['vwap']) / snapshot['vwap']
        print(f"close_vs_vwap_pct : {deviation:.6%}")
    print("=" * 80)
    return 0

def parse_date(date_str: str) -> date:
    """解析日期字符串"""
    for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")


def backfill_selection_results(
    start_date: date,
    end_date: date,
    save_to_db: bool = True,
) -> Dict[date, int]:
    """
    回补指定日期范围内的选股结果。

    只处理交易日（跳过周末），每天运行一次选股逻辑。

    Args:
        start_date: 回补开始日期
        end_date: 回补结束日期
        save_to_db: 是否保存到数据库

    Returns:
        字典，key为日期，value为当天选股结果数量
    """
    from pandas.tseries.offsets import BDay

    results = {}
    current_date = start_date

    print("\n" + "=" * 80)
    print(f"开始回补选股结果")
    print(f"  开始日期: {start_date.strftime('%Y-%m-%d')}")
    print(f"  结束日期: {end_date.strftime('%Y-%m-%d')}")
    print(f"  保存到数据库: {save_to_db}")
    print("=" * 80)

    # BSM选股基于周线数据，每周只需在周五运行一次
    # 生成周五列表（周线数据在周五收盘后更新）
    fridays = []
    while current_date <= end_date:
        # 周五 = 4
        if current_date.weekday() == 4:
            fridays.append(current_date)
        current_date += timedelta(days=1)

    print(f"\n共 {len(fridays)} 个周五需要处理（BSM基于周线，每周运行一次）\n")

    for friday in tqdm(fridays, desc="回补进度", unit="周"):
        print(f"\n{'=' * 80}")
        print(f"处理日期: {friday.strftime('%Y-%m-%d')} (周五)")
        print(f"{'=' * 80}")

        try:
            df = select_high_margin_stocks(selection_date=friday, save_to_db=save_to_db)
            results[friday] = len(df)
            print(f"  结果: {len(df)} 只股票")
        except Exception as e:
            print(f"  错误: {e}")
            results[friday] = 0

    print("\n" + "=" * 80)
    print("回补完成")
    print("=" * 80)
    print(f"\n汇总:")
    print(f"  总周数: {len(fridays)}")
    print(f"  有选股结果的周数: {sum(1 for v in results.values() if v > 0)}")
    print(f"  总选股数: {sum(results.values())}")

    return results


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
    parser.add_argument(
        '--test-dsa-last-bar',
        action='store_true',
        help='测试指定股票最后一个 bar 的 DSA dir/vwap，便于和 HTML 对账'
    )
    parser.add_argument(
        '--ts-code',
        help='测试 DSA 最后一个 bar 时使用的股票代码，例如 605016'
    )
    parser.add_argument(
        '--dsa-freq',
        default='w',
        choices=['d', 'w'],
        help='测试 DSA 最后一个 bar 时使用的频率，默认 w'
    )
    parser.add_argument(
        '--dsa-bars',
        type=int,
        default=DSA_HISTORY_BARS,
        help=f'TEST 模式下 DSA 计算使用的 bars 数，默认 {DSA_HISTORY_BARS}'
    )
    parser.add_argument(
        '--market-data-source',
        default='pytdx',
        choices=['pytdx', 'db'],
        help='K线行情数据源，默认 pytdx；如需回退数据库可传 db'
    )
    parser.add_argument(
        '--backfill',
        action='store_true',
        help='回补模式：回补从 start-date 到 end-date 的选股结果'
    )
    parser.add_argument(
        '--start-date',
        help='回补开始日期 (格式: YYYY-MM-DD)，与 --backfill 一起使用'
    )
    parser.add_argument(
        '--end-date',
        help='回补结束日期 (格式: YYYY-MM-DD)，与 --backfill 一起使用，默认为今天'
    )

    args = parser.parse_args()

    # 回补模式
    if args.backfill:
        if not args.start_date:
            print("错误: 使用 --backfill 时必须提供 --start-date")
            sys.exit(1)

        try:
            start_date = parse_date(args.start_date)
        except ValueError as e:
            print(f"错误: 开始日期格式错误: {e}")
            sys.exit(1)

        if args.end_date:
            try:
                end_date = parse_date(args.end_date)
            except ValueError as e:
                print(f"错误: 结束日期格式错误: {e}")
                sys.exit(1)
        else:
            end_date = date.today()

        results = backfill_selection_results(
            start_date=start_date,
            end_date=end_date,
            save_to_db=True,
        )
        return results

    # 解析日期（单日模式）
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
    print(f"行情数据源: {MARKET_DATA_SOURCE}")
    print("=" * 80)

    if args.test_dsa_last_bar:
        if not args.ts_code:
            print("错误: 使用 --test-dsa-last-bar 时必须同时提供 --ts-code")
            sys.exit(1)
        exit_code = test_dsa_last_bar(
            ts_code=args.ts_code,
            selection_date=selection_date,
            freq=args.dsa_freq,
            bars=args.dsa_bars,
        )
        sys.exit(exit_code)

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