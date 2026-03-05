# -*- coding: utf-8 -*-
"""
K线数据保存脚本 - 获取股票池所有股票的5分钟K线数据
支持增量更新和历史回补，数据保存到PostgreSQL数据库

入口:
  python -m app.k_data_saver --full          # 全量回补（3 年数据）
  python -m app.k_data_saver --incremental   # 增量更新（只更新缺失数据）
  python -m app.k_data_saver --create-table  # 创建数据库表
"""
import os
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import STOCK_POOL_PATH, PYTDX_SERVERS, DATABASE_URL
from app.db import get_session, execute_sql, query_sql
from app.logger import get_logger
from app.models import K_DATA_TABLE
from datasource.pytdx_client import TdxHq_API

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_FREQ = "5m"
DEFAULT_DAYS_BACK = 365 * 3
BARS_PER_DAY = 48


class AsyncDbSaver:
    """异步数据库保存器，后台线程处理保存任务，不影响pytdx读取效率"""
    
    def __init__(self):
        self._task_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._total_submitted = 0
        self._total_completed = 0
    
    def start(self):
        self._stop_event.clear()
        self._total_submitted = 0
        self._total_completed = 0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._task_queue.put(None)
        if self._thread:
            self._thread.join()
        logger.info(f"异步保存完成: 提交={self._total_submitted}, 完成={self._total_completed}")
    
    def _worker(self):
        while True:
            try:
                task = self._task_queue.get(timeout=1)
                if task is None:
                    break
                self._process_task(task)
                with self._lock:
                    self._total_completed += 1
                self._task_queue.task_done()
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
    
    def _process_task(self, task: dict):
        task_type = task.get("type")
        if task_type == "save_bars":
            self._save_bars_to_db(task)
    
    def _save_bars_to_db(self, task: dict):
        ts_code = task["ts_code"]
        freq = task["freq"]
        df = task["data"]
        
        if df.empty:
            return
        
        df_to_save = df.copy()
        df_to_save["ts_code"] = ts_code
        df_to_save["freq"] = freq
        df_to_save["bar_time"] = df_to_save.index
        df_to_save = df_to_save.reset_index(drop=True)
        df_to_save = df_to_save[["ts_code", "freq", "bar_time", "open", "high", "low", "close", "volume"]]
        
        records = df_to_save.to_dict(orient="records")
        
        sql = """
            INSERT INTO stock_k_data (ts_code, freq, bar_time, open, high, low, close, volume)
            VALUES (:ts_code, :freq, :bar_time, :open, :high, :low, :close, :volume)
            ON CONFLICT (ts_code, freq, bar_time) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """
        
        try:
            with get_session() as session:
                batch_size = 1000
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    session.execute(text(sql), batch)
                session.commit()
            logger.debug(f"{ts_code} 保存 {len(records)} 条记录到数据库")
        except Exception as e:
            logger.error(f"{ts_code} 保存数据库失败: {e}")
            raise
    
    def submit_save_task(self, ts_code: str, freq: str, df: pd.DataFrame):
        if df.empty:
            return
        
        self._task_queue.put({
            "type": "save_bars",
            "ts_code": ts_code,
            "freq": freq,
            "data": df,
        })
        with self._lock:
            self._total_submitted += 1


class PytdxConnection:
    """pytdx连接管理器，支持断线自动重连"""
    
    def __init__(self):
        self._api: Optional[TdxHq_API] = None
        self._current_server: Tuple[str, int] = None
    
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
    
    def reconnect(self) -> None:
        servers = list(PYTDX_SERVERS)
        if self._current_server in servers:
            servers.remove(self._current_server)
        
        if self._api:
            try:
                self._api.disconnect()
            except Exception:
                pass
            self._api = None
        
        for host, port in servers:
            if self._try_connect(host, port):
                logger.info(f"切换到服务器 {host}:{port}")
                return
        
        raise RuntimeError("所有服务器均无法连接")
    
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


def get_stock_last_bar_time(ts_code: str, freq: str) -> Optional[datetime]:
    """从数据库获取股票最后一个K线时间"""
    sql = """
        SELECT MAX(bar_time) as last_time
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = :freq
    """
    with get_session() as session:
        result = query_sql(session, sql, {"ts_code": ts_code, "freq": freq})
        if result.empty or result["last_time"].isna().all():
            return None
        return pd.to_datetime(result["last_time"].iloc[0])


def get_stock_date_count(ts_code: str, freq: str) -> int:
    """获取股票已有数据的交易日数量"""
    sql = """
        SELECT COUNT(DISTINCT DATE(bar_time)) as day_count
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = :freq
    """
    with get_session() as session:
        result = query_sql(session, sql, {"ts_code": ts_code, "freq": freq})
        if result.empty:
            return 0
        return int(result["day_count"].iloc[0]) or 0


def fetch_bars_pytdx(conn: "PytdxConnection", symbol: str, freq: str, 
                     start_date: Optional[datetime] = None,
                     bars_needed: int = 25000) -> pd.DataFrame:
    cat = _category_from_freq(freq)
    mkt = _market_from_code(symbol)
    page = 0
    size = 700
    frames = []
    
    while True:
        api = conn.get_api()
        try:
            recs = api.get_security_bars(cat, mkt, symbol, page * size, size)
        except Exception as e:
            logger.warning(f"数据获取异常: {e}，尝试重连...")
            conn.reconnect()
            api = conn.get_api()
            recs = api.get_security_bars(cat, mkt, symbol, page * size, size)
        
        if not recs:
            break
        df = _records_to_df(recs)
        frames.append(df)
        if len(recs) < size:
            break
        page += 1
        if sum(len(f) for f in frames) >= bars_needed:
            break
        
        if start_date and len(frames) > 0:
            earliest = frames[0].index[0]
            if earliest <= start_date:
                break
    
    if not frames:
        return pd.DataFrame()
    
    all_df = pd.concat(frames).sort_index()
    
    if start_date:
        all_df = all_df[all_df.index >= start_date]
    
    return all_df


def fetch_recent_bars(conn: "PytdxConnection", symbol: str, freq: str,
                      days: int = 10) -> pd.DataFrame:
    """快速获取最近N天的数据，用于增量更新"""
    cat = _category_from_freq(freq)
    mkt = _market_from_code(symbol)
    bars_needed = days * BARS_PER_DAY
    size = 700
    
    recs = conn.get_api().get_security_bars(cat, mkt, symbol, 0, min(size, bars_needed + 500))
    
    if not recs:
        return pd.DataFrame()
    
    df = _records_to_df(recs)
    return df.sort_index()


def download_for_incremental(conn: "PytdxConnection", ts_code: str, name: str) -> Tuple[pd.DataFrame, int]:
    """增量更新：只获取数据库最后一条记录之后的数据"""
    symbol = ts_code.split('.')[0]
    
    last_bar_time = get_stock_last_bar_time(ts_code, DEFAULT_FREQ)
    
    if last_bar_time is None:
        return pd.DataFrame(), 0
    
    today = datetime.now()
    days_diff = (today.date() - last_bar_time.date()).days
    days_to_fetch = max(days_diff, 1) + 2
    
    try:
        df = fetch_recent_bars(conn, symbol, DEFAULT_FREQ, days=days_to_fetch)
        if df.empty:
            return pd.DataFrame(), 0
        
        df = df[df.index > last_bar_time]
        
        if df.empty:
            return pd.DataFrame(), 0
        
        total_days = len(df.index.normalize().unique())
        return df, total_days
        
    except Exception as e:
        logger.error(f"{name}({ts_code}): {e}")
        return pd.DataFrame(), 0


def download_for_full_backfill(conn: "PytdxConnection", ts_code: str, name: str,
                               days_back: int) -> Tuple[pd.DataFrame, int]:
    """全量回补：获取指定天数的历史数据"""
    symbol = ts_code.split('.')[0]
    
    start_date = datetime.now() - timedelta(days=days_back)
    
    try:
        df = fetch_bars_pytdx(conn, symbol, DEFAULT_FREQ, start_date=start_date)
        if df.empty:
            return pd.DataFrame(), 0
        
        total_days = len(df.index.normalize().unique())
        return df, total_days
        
    except Exception as e:
        logger.error(f"{name}({ts_code}): {e}")
        return pd.DataFrame(), 0


def create_k_data_table():
    """创建K线数据表"""
    execute_sql(K_DATA_TABLE)
    logger.info("K线数据表创建成功")


def incremental_update(limit: Optional[int] = None, 
                       filter_name: Optional[str] = None) -> Dict:
    """增量更新：只更新每只股票最后一条记录之后的数据"""
    df_pool = pd.read_excel(STOCK_POOL_PATH)
    logger.info(f"股票池共 {len(df_pool)} 只股票")
    
    if filter_name:
        df_pool = df_pool[df_pool['name'].str.contains(filter_name, na=False)]
        logger.info(f"过滤后共 {len(df_pool)} 只股票")
    
    if limit:
        df_pool = df_pool.head(limit)
        logger.info(f"限制处理前 {limit} 只股票")
    
    conn = PytdxConnection()
    conn.connect()
    
    async_saver = AsyncDbSaver()
    async_saver.start()
    
    stats = {
        "total_stocks": len(df_pool),
        "success": 0,
        "failed": 0,
        "updated_days": 0,
    }
    
    try:
        for _, row in tqdm(df_pool.iterrows(), total=len(df_pool), desc="增量更新"):
            ts_code = row['ts_code']
            name = row['name']
            
            df, total_days = download_for_incremental(conn, ts_code, name)
            
            if not df.empty and total_days > 0:
                async_saver.submit_save_task(ts_code, DEFAULT_FREQ, df)
                stats["success"] += 1
                stats["updated_days"] += total_days
                logger.info(f"{name}({ts_code}): 更新 {total_days} 天")
            else:
                stats["failed"] += 1
    
    finally:
        async_saver.stop()
        conn.disconnect()
    
    logger.info(f"增量更新完成: 总股票数={stats['total_stocks']}, 有更新={stats['success']}, 无更新={stats['failed']}, 更新天数={stats['updated_days']}")
    
    return stats


def full_backfill(days_back: int = DEFAULT_DAYS_BACK,
                  limit: Optional[int] = None, 
                  filter_name: Optional[str] = None) -> Dict:
    """全量回补：获取指定天数的历史数据"""
    df_pool = pd.read_excel(STOCK_POOL_PATH)
    logger.info(f"股票池共 {len(df_pool)} 只股票")
    
    if filter_name:
        df_pool = df_pool[df_pool['name'].str.contains(filter_name, na=False)]
        logger.info(f"过滤后共 {len(df_pool)} 只股票")
    
    if limit:
        df_pool = df_pool.head(limit)
        logger.info(f"限制处理前 {limit} 只股票")
    
    conn = PytdxConnection()
    conn.connect()
    
    async_saver = AsyncDbSaver()
    async_saver.start()
    
    stats = {
        "total_stocks": len(df_pool),
        "success": 0,
        "failed": 0,
        "total_days": 0,
    }
    
    try:
        for _, row in tqdm(df_pool.iterrows(), total=len(df_pool), desc="全量回补"):
            ts_code = row['ts_code']
            name = row['name']
            
            df, total_days = download_for_full_backfill(
                conn, ts_code, name, 
                days_back=days_back
            )
            
            if not df.empty and total_days > 0:
                async_saver.submit_save_task(ts_code, DEFAULT_FREQ, df)
                stats["success"] += 1
                stats["total_days"] += total_days
                logger.info(f"{name}({ts_code}): 下载 {total_days} 天")
            else:
                stats["failed"] += 1
    
    finally:
        async_saver.stop()
        conn.disconnect()
    
    logger.info(f"全量回补完成: 总股票数={stats['total_stocks']}, 成功={stats['success']}, 失败={stats['failed']}, 总交易日={stats['total_days']}")
    
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="K线数据保存脚本")
    ap.add_argument("--full", action="store_true", help="全量回补（默认3年数据）")
    ap.add_argument("--incremental", action="store_true", help="增量更新（只更新缺失数据）")
    ap.add_argument("--create-table", action="store_true", help="创建数据库表")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS_BACK, help="全量回补天数（默认1095天）")
    ap.add_argument("--limit", type=int, default=None, help="限制股票数量")
    ap.add_argument("--filter", type=str, default=None, help="按名称过滤股票")
    args = ap.parse_args()
    
    if args.create_table:
        create_k_data_table()
    elif args.full:
        logger.info(f"开始全量数据回补: 回补周期={args.days}天, 数据频率={DEFAULT_FREQ}")
        full_backfill(days_back=args.days, limit=args.limit, filter_name=args.filter)
    elif args.incremental:
        logger.info(f"开始增量更新: 数据频率={DEFAULT_FREQ}")
        incremental_update(limit=args.limit, filter_name=args.filter)
    else:
        logger.info("请指定运行模式: --full (全量回补) 或 --incremental (增量更新) 或 --create-table (创建表)")
