# -*- coding: utf-8 -*-
"""
AMP 特征扫描脚本 - 扫描股票池中所有股票的多周期特征并写入数据库
引用 features/amp_plotly.py 的核心 AMP 算法
数据源：k_data_loader.py

典型用法:
  # 扫描整个股票池的日线周期
  python -m app.amp_scanner --freqs "d"
  
  # 扫描整个股票池的常用周期组合
  python -m app.amp_scanner --freqs "15m,60m,d,w"
  
  # 扫描指定股票（按名称过滤）
  python -m app.amp_scanner --filter "中金黄金" --freqs "5m,15m,60m,d"
  
  # 限制扫描前 N 只股票（用于测试）
  python -m app.amp_scanner --freqs "d" --limit 5
"""
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.amp_plotly import (
    calcDevATF, cS, cD, apply_deviation, clamp01, safe_ratio, 
    f_adjust, f_unadjust, calc_line_value, AMPConfig as AMPConfigBase
)
from app.config import STOCK_POOL_PATH, PROJECT_ROOT, PYTDX_SERVERS
from app.logger import get_logger
from app.db import get_session
from app.models import AMP_FEATURES_TABLE, AMP_FEATURES_UPSERT_SQL
from datasource.pytdx_client import TdxHq_API


def _normalize_freq(freq: str) -> str:
    """标准化频率格式，将 "15" 转为 "15m"，"60" 转为 "60m" 等"""
    f = str(freq).strip().lower()
    if f == "1":
        return "1m"
    if f == "5":
        return "5m"
    if f == "15":
        return "15m"
    if f == "30":
        return "30m"
    if f == "60" or f == "1h":
        return "60m"
    if f in ("d", "day", "daily"):
        return "d"
    if f in ("w", "week"):
        return "w"
    return f


def _category_from_freq(freq: str) -> int:
    f = str(freq).strip().lower()
    if f in ("d", "day", "daily"):
        return 4
    if f in ("w", "week"):
        return 5
    if f in ("60", "1h", "60m"):
        return 3
    if f in ("30", "30m"):
        return 2
    if f in ("15", "15m"):
        return 1
    if f in ("5", "5m"):
        return 0
    if f in ("1", "1m"):
        return 7
    raise ValueError(f"不支持的频率: {freq}")


def _market_from_code(code: str) -> int:
    c = str(code)
    return 1 if c.startswith("6") else 0


def _records_to_df(records: List[dict]) -> pd.DataFrame:
    d = pd.DataFrame(records)
    if d.empty:
        return d
    if "datetime" in d.columns:
        d["datetime"] = pd.to_datetime(d["datetime"])
        d = d.set_index("datetime")
    elif {"year", "month", "day", "hour", "minute"}.issubset(d.columns):
        d["datetime"] = pd.to_datetime(d[["year", "month", "day", "hour", "minute"]].astype(int))
        d = d.set_index("datetime")
    if "vol" in d.columns:
        d = d.rename(columns={"vol": "volume"})
    req = ["open", "high", "low", "close"]
    miss = [x for x in req if x not in d.columns]
    if miss:
        raise RuntimeError(f"缺少列: {miss}")
    cols = ["open", "high", "low", "close"] + (["volume"] if "volume" in d.columns else [])
    out = d[cols].astype(float).sort_index()
    return out


class PytdxConnection:
    """pytdx 连接管理器（单例模式），支持断线自动重连"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._api: Optional[TdxHq_API] = None
            self._current_server: Tuple[str, int] = None
            self._initialized = True
    
    def _try_connect(self, host: str, port: int) -> bool:
        try:
            api = TdxHq_API(raise_exception=True, auto_retry=True)
            if api.connect(host, port):
                self._api = api
                self._current_server = (host, port)
                return True
        except Exception:
            pass
        return False
    
    def connect(self) -> None:
        last_errors = []
        for host, port in PYTDX_SERVERS:
            if self._try_connect(host, port):
                return
            last_errors.append(f"{host}:{port}")
        err = "; ".join(last_errors)
        raise RuntimeError(f"pytdx连接失败，尝试服务器: {err}")
    
    def get_api(self) -> TdxHq_API:
        return self._api
    
    def disconnect(self) -> None:
        if self._api:
            try:
                self._api.disconnect()
            except Exception:
                pass
            self._api = None
            self._current_server = None


def _get_last_bar_time_from_db(ts_code: str, freq: str) -> Optional[datetime]:
    """从数据库获取某股票某频率的最后时间戳"""
    try:
        # 确保 ts_code 格式正确（带后缀）
        if '.' not in ts_code:
            # 根据股票代码补全后缀
            if ts_code.startswith('6'):
                ts_code = ts_code + '.SH'
            else:
                ts_code = ts_code + '.SZ'
        
        with get_session() as session:
            sql = text("""
                SELECT MAX(bar_time) as last_time
                FROM stock_amp_features
                WHERE ts_code = :ts_code AND freq = :freq
            """)
            result = session.execute(sql, {"ts_code": ts_code, "freq": freq})
            row = result.fetchone()
            if row and row.last_time:
                return row.last_time
            return None
    except Exception as e:
        logger.warning(f"查询数据库最后时间失败：{e}")
        return None


# 全局 pytdx 连接单例
_pytdx_conn: Optional[PytdxConnection] = None


def _get_pytdx_connection() -> PytdxConnection:
    """获取全局 pytdx 连接单例"""
    global _pytdx_conn
    if _pytdx_conn is None:
        _pytdx_conn = PytdxConnection()
        _pytdx_conn.connect()
    return _pytdx_conn


def load_k_data_from_pytdx(ts_code: str, freq: str = "5m", bars_needed: int = 2500) -> pd.DataFrame:
    """直接用 pytdx 读取 K 线数据"""
    conn = _get_pytdx_connection()
    
    try:
        cat = _category_from_freq(freq)
        mkt = _market_from_code(ts_code[:6])
        page = 0
        size = 700
        frames = []
        
        while True:
            api = conn.get_api()
            # 检查连接是否有效
            if api is None:
                conn.connect()
                api = conn.get_api()
            
            recs = api.get_security_bars(cat, mkt, ts_code[:6], page * size, size)
            
            if not recs:
                break
            df = _records_to_df(recs)
            frames.append(df)
            if len(recs) < size:
                break
            page += 1
            if sum(len(f) for f in frames) >= bars_needed:
                break
        
        if not frames:
            return pd.DataFrame()
        
        df = pd.concat(frames).sort_index()
        return df
    except Exception:
        # 发生异常时重置连接，下次会自动重连
        global _pytdx_conn
        if _pytdx_conn:
            _pytdx_conn.disconnect()
            _pytdx_conn = None
        raise

logger = get_logger(__name__)


AMP_CANDIDATE_PERIODS_LOCAL = [50, 60, 70, 80, 90, 100, 115, 130, 145, 160, 180, 200, 220, 250, 280, 310, 340, 370, 400]


def compute_activity_pos_vectorized(df: pd.DataFrame, bar_idx: int, lI: int, 
                                     uSP: float, uEP: float, lSP: float, lEP: float,
                                     cfg: AMPConfigBase) -> Optional[float]:
    """向量化计算 activity_pos"""
    nFills = 23
    
    window_start = max(0, bar_idx - lI + 1)
    window_len = bar_idx - window_start + 1
    
    if window_len < 10:
        return 0.5
    
    high_arr = df["high"].iloc[window_start:bar_idx+1].values.astype(float)
    low_arr = df["low"].iloc[window_start:bar_idx+1].values.astype(float)
    vol_arr = df["volume"].iloc[window_start:bar_idx+1].values.astype(float)
    
    fill_height = (uEP - lEP) / nFills
    fill_indices = np.arange(nFills)
    fill_lows = lEP + fill_indices * fill_height
    fill_highs = fill_lows + fill_height
    
    low_expanded = low_arr[:, np.newaxis]
    high_expanded = high_arr[:, np.newaxis]
    vol_expanded = vol_arr[:, np.newaxis]
    
    mask = (low_expanded <= fill_highs) & (high_expanded >= fill_lows)
    counts = np.sum(vol_expanded * mask, axis=0)
    
    total_vol = np.sum(counts)
    if total_vol <= 0:
        return 0.5
    
    weighted_idx = np.sum(fill_indices * counts) / total_vol
    return clamp01(weighted_idx / nFills)


def compute_amp_features_at_bar(df: pd.DataFrame, bar_idx: int, cfg: AMPConfigBase) -> Optional[Dict]:
    """计算指定 bar 位置的 AMP 特征"""
    n = bar_idx + 1
    if n < 60:
        return None
    
    close = df["close"].iloc[:n+1].to_numpy(float)
    high = df["high"].iloc[:n+1].to_numpy(float)
    low = df["low"].iloc[:n+1].to_numpy(float)
    
    has_volume = "volume" in df.columns
    
    close_r = close[::-1]
    high_r = high[::-1]
    low_r = low[::-1]
    
    detected_period = cfg.pI
    detected_r = None
    if cfg.useAdaptive:
        best_r = -1e9
        for p in AMP_CANDIDATE_PERIODS_LOCAL:
            if p <= n:
                _, pr, _, _ = calcDevATF(close_r, p, cfg.uL)
                if pr > best_r:
                    best_r = pr
                    detected_period = p
                    detected_r = pr
    
    finalPeriod = detected_period if cfg.useAdaptive else cfg.pI
    lI = min(n, int(finalPeriod))
    
    s, a, ic = cS(close_r, lI, cfg.uL)
    sP = f_unadjust(ic + s * (lI - 1), cfg.uL)
    eP = f_unadjust(ic, cfg.uL)
    
    sD, pR, _, _ = cD(high_r, low_r, close_r, lI, s, a, ic, cfg.uL)
    
    dev = cfg.devMultiplier * sD
    uSP = apply_deviation(sP, +dev, cfg.uL)
    uEP = apply_deviation(eP, +dev, cfg.uL)
    lSP = apply_deviation(sP, -dev, cfg.uL)
    lEP = apply_deviation(eP, -dev, cfg.uL)
    
    bars = max(1, (lI - 1))
    
    bar_close = float(df.iloc[bar_idx]["close"])
    bar_upper = uEP
    bar_lower = lEP
    
    close_pos = clamp01(safe_ratio(bar_close - bar_lower, (bar_upper - bar_lower)))
    
    activity_pos = None
    if has_volume:
        activity_pos = compute_activity_pos_vectorized(
            df, bar_idx, lI, uSP, uEP, lSP, lEP, cfg
        )
    
    def slope_metrics(start_price: float, end_price: float) -> Dict:
        a0 = f_adjust(start_price, cfg.uL)
        a1 = f_adjust(end_price, cfg.uL)
        slope_adj_per_bar = (a1 - a0) / bars
        if cfg.uL:
            ret_per_bar = math.exp(slope_adj_per_bar) - 1.0
        else:
            ret_per_bar = safe_ratio((end_price - start_price) / bars, start_price)
        total_ret = safe_ratio(end_price - start_price, start_price) if start_price != 0 else 0.0
        return {
            "slope_adj_per_bar": float(slope_adj_per_bar),
            "ret_per_bar": float(ret_per_bar),
            "total_ret": float(total_ret),
        }
    
    upper_slope = slope_metrics(uSP, uEP)
    lower_slope = slope_metrics(lSP, lEP)
    
    return {
        "bar_time": df.index[bar_idx],
        "window_len": int(lI),
        "final_period": int(finalPeriod),
        "pearson_r": float(detected_r) if detected_r is not None else None,
        "strength_pr": float(pR),
        "bar_close": bar_close,
        "bar_upper": float(bar_upper),
        "bar_lower": float(bar_lower),
        "close_pos_0_1": float(close_pos),
        "activity_pos_0_1": float(activity_pos) if activity_pos is not None else None,
        "upper_ret_per_bar": upper_slope["ret_per_bar"],
        "upper_total_ret": upper_slope["total_ret"],
        "lower_ret_per_bar": lower_slope["ret_per_bar"],
        "lower_total_ret": lower_slope["total_ret"],
    }


def compute_all_bars_features(df: pd.DataFrame, cfg: AMPConfigBase, 
                               recent_bars: int = 255) -> List[Dict]:
    """
    计算 bar 的特征
    
    Args:
        df: K 线数据
        cfg: AMP 配置
        recent_bars: 只计算最近 N 个 bar 的特征（默认 255，约 1 年交易日）
                     如果为 None 则计算所有 bar
    
    Returns:
        特征记录列表
    """
    n = len(df)
    if n < 60:
        return []
    
    # 确定计算范围
    if recent_bars is None:
        # 计算所有 bar
        start_idx = 59
    else:
        # 只计算最近 N 个 bar
        # 需要保证有足够的历史数据进行计算（至少 60 个 bar）
        effective_start = max(59, n - recent_bars)
        start_idx = effective_start
    
    records = []
    for i in range(start_idx, n):
        features = compute_amp_features_at_bar(df, i, cfg)
        if features is not None:
            records.append(features)
    
    return records


def scan_single_stock(ts_code: str, name: str, 
                       frequencies: List[str], cfg: AMPConfigBase,
                       recent_bars: int = 255) -> List[Dict]:
    """
    扫描单只股票的多周期特征（增量更新）
    
    Args:
        ts_code: 股票代码
        name: 股票名称
        frequencies: 周期列表
        cfg: AMP 配置
        recent_bars: 只计算最近 N 个 bar（默认 255；None 表示计算所有）
    """
    results = []
    
    for freq in frequencies:
        # 标准化频率格式
        normalized_freq = _normalize_freq(freq)
        
        try:
            # 1. 查询数据库中该股票该频率的最后时间戳
            last_bar_time = _get_last_bar_time_from_db(ts_code, normalized_freq)
            
            # 2. 计算需要从 pytdx 获取的 bars 数量
            # 需要：recent_bars(用于计算) + 缓冲 (用于重采样和计算起点)
            if last_bar_time:
                # 增量更新：只需要获取 last_bar_time 之后的数据 + 最近 300 个 bars 用于计算
                bars_needed = min(300, recent_bars or 300)
                logger.debug(f"{name}({ts_code}) {freq}: 数据库最后时间 {last_bar_time}, 需要获取 {bars_needed} bars")
            else:
                # 首次扫描：获取足够的数据
                bars_needed = 3500
                logger.debug(f"{name}({ts_code}) {freq}: 首次扫描，需要获取 {bars_needed} bars")
            
            # 3. 从 pytdx 获取数据
            df = load_k_data_from_pytdx(ts_code, freq=normalized_freq, bars_needed=bars_needed)
            
            if df.empty or len(df) < 60:
                if df.empty:
                    logger.debug(f"{name}({ts_code}) {freq}: 无数据")
                else:
                    logger.warning(f"{name}({ts_code}) {freq}: 数据不足 ({len(df)} bars)")
                continue
            
            # 4. 获取完整数据用于计算，然后过滤已存在的数据
            if last_bar_time:
                # 使用完整数据计算所有特征（不限制 recent_bars）
                all_records = compute_all_bars_features(df, cfg, recent_bars=None)
                
                # 只保留比数据库最新时间更新的记录
                new_records = [r for r in all_records if r['bar_time'] > last_bar_time]
                
                if not new_records:
                    logger.debug(f"{name}({ts_code}) {freq}: 无新数据")
                    continue
                
                records = new_records
            else:
                # 首次扫描，只计算最近 N 个 bars
                records = compute_all_bars_features(df, cfg, recent_bars=recent_bars)
            
            if records is None or len(records) == 0:
                logger.warning(f"{name}({ts_code}) {freq}: 特征计算失败")
                continue
            
            for rec in records:
                record = {
                    "ts_code": ts_code,
                    "name": name,
                    "freq": normalized_freq,
                    **rec
                }
                results.append(record)
            
            last_record = records[-1]
            activity_str = f"{last_record['activity_pos_0_1']:.3f}" if last_record['activity_pos_0_1'] is not None else "N/A"
            logger.info(f"{name}({ts_code}) {normalized_freq}: {len(records)} bars, strength={last_record['strength_pr']:.3f}, close_pos={last_record['close_pos_0_1']:.3f}, activity={activity_str}")
            
        except FileNotFoundError:
            logger.warning(f"{name}({ts_code}) {freq}: 本地数据不存在")
        except Exception as e:
            logger.error(f"{name}({ts_code}) {freq}: {e}")
    
    return results


def save_features_to_db(features: List[Dict], max_retries: int = 3) -> int:
    """
    保存特征到数据库（支持重试）
    
    Args:
        features: 特征记录列表
        max_retries: 最大重试次数（默认 3 次）
    
    Returns:
        成功保存的记录数
    """
    if not features:
        return 0
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            with get_session() as session:
                for record in features:
                    session.execute(text(AMP_FEATURES_UPSERT_SQL), record)
            
            logger.info(f"共保存 {len(features)} 条特征记录到数据库")
            return len(features)
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            
            # 判断是否是连接断开错误
            if "server closed the connection" in error_msg or "connection" in error_msg.lower():
                logger.warning(f"数据库连接断开（第 {retry_count}/{max_retries} 次）: {error_msg}")
                
                if retry_count < max_retries:
                    # 重置全局 pytdx 连接，触发重连
                    global _pytdx_conn
                    if _pytdx_conn:
                        _pytdx_conn.disconnect()
                        _pytdx_conn = None
                    logger.info("已重置 pytdx 连接，将尝试重连...")
                    continue
                else:
                    logger.error(f"重试 {max_retries} 次后仍然失败，跳过该股票")
                    return 0
            else:
                # 其他错误直接抛出
                logger.error(f"数据库操作异常：{error_msg}")
                return 0
    
    return 0


def scan_stock_pool(
    stock_pool_path: str,
    frequencies: List[str],
    limit: int = None,
    filter_name: str = None,
    recent_bars: int = 255
) -> int:
    """
    扫描股票池中所有股票
    
    Args:
        stock_pool_path: 股票池 Excel 路径（已废弃，现在从数据库读取）
        frequencies: 周期列表
        limit: 限制扫描股票数量
        filter_name: 按名称过滤股票
        recent_bars: 只计算最近 N 个 bar 的特征（默认 255；None 表示计算所有）
    """
    # 从数据库读取股票池
    from app.db import get_session
    from sqlalchemy import text
    
    with get_session() as session:
        sql = "SELECT ts_code, name FROM stock_concepts_cache"
        if filter_name:
            sql += " WHERE name LIKE :filter_name"
            result = session.execute(text(sql), {"filter_name": f"%{filter_name}%"})
        else:
            result = session.execute(text(sql))
        
        rows = result.fetchall()
        df_pool = pd.DataFrame(rows, columns=['ts_code', 'name'])
    
    # 如果数据库为空，尝试从 Excel 文件读取（向后兼容）
    if df_pool.empty and os.path.exists(stock_pool_path):
        logger.warning(f"⚠️  数据库中没有股票数据，从 {stock_pool_path} 读取")
        df_pool = pd.read_excel(stock_pool_path)
    
    logger.info(f"股票池共 {len(df_pool)} 只股票")
    
    if limit:
        df_pool = df_pool.head(limit)
        logger.info(f"限制扫描前 {limit} 只股票")
    
    # 创建表（如果不存在）
    with get_session() as session:
        session.execute(text(AMP_FEATURES_TABLE))
    logger.info("表 stock_amp_features 已创建/确认存在")
    
    cfg = AMPConfigBase(useAdaptive=True, uL=True)
    
    total_saved = 0
    failed_stocks = []
    for _, row in tqdm(df_pool.iterrows(), total=len(df_pool), desc="扫描股票"):
        ts_code = row['ts_code']
        name = row['name']
        
        try:
            features = scan_single_stock(ts_code, name, frequencies, cfg, 
                                         recent_bars=recent_bars)
            
            # 每只股票处理完立即保存
            if features:
                saved = save_features_to_db(features)
                total_saved += saved
                
                if saved == 0 and features:
                    # 保存失败，记录但不中断
                    failed_stocks.append(f"{name}({ts_code})")
                    logger.warning(f"跳过股票：{name}({ts_code})")
        except Exception as e:
            # 捕获任何异常，确保不会中断整个扫描
            logger.error(f"处理股票 {name}({ts_code}) 时出错：{e}")
            failed_stocks.append(f"{name}({ts_code})")
            continue
    
    # 输出统计信息
    if failed_stocks:
        logger.warning(f"共有 {len(failed_stocks)} 只股票处理失败：{', '.join(failed_stocks[:10])}{'...' if len(failed_stocks) > 10 else ''}")
    
    if total_saved > 0:
        logger.info(f"扫描完成，共保存 {total_saved} 条记录")
        return total_saved
    else:
        logger.warning("无有效特征数据")
        return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="扫描股票池 AMP 特征")
    ap.add_argument("--freqs", type=str, default="15m,60m,d,w", help="周期列表，逗号分隔")
    ap.add_argument("--limit", type=int, default=None, help="限制扫描股票数量")
    ap.add_argument("--filter", type=str, default=None, help="按名称过滤股票")
    ap.add_argument("--recent-bars", type=int, default=255, 
                    help="只计算最近 N 个 bar 的特征（默认 255，约 1 年交易日；设为 0 或 -1 表示计算所有 bar）")
    args = ap.parse_args()
    
    frequencies = [f.strip() for f in args.freqs.split(',')]
    recent_bars = None if args.recent_bars <= 0 else args.recent_bars
    logger.info(f"扫描周期：{frequencies}")
    logger.info(f"计算最近 {recent_bars if recent_bars else '所有'} 个 bar")
    
    saved = scan_stock_pool(
        STOCK_POOL_PATH, 
        frequencies, 
        limit=args.limit, 
        filter_name=args.filter,
        recent_bars=recent_bars
    )
    logger.info(f"扫描完成，共保存 {saved} 条记录")
