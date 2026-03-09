# -*- coding: utf-8 -*-
"""
背离特征扫描脚本 - 扫描股票池中所有股票的多周期背离特征并写入数据库
引用 features/divergence_many_plotly.py 的核心背离算法
数据源: k_data_loader.py

python -m src.divergence_scanner --filter "中金黄金" --freqs "5m,15m,60m,d"
"""
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.divergence_many_plotly import (
    DivConfig, compute_indicators, pivots_confirmed as pivots_confirmed_original,
    pos_reg_or_neg_hid, neg_reg_or_pos_hid, calculate_divs,
    _line_dash
)
from app.config import STOCK_POOL_PATH
from app.db import get_session
from app.models import DIV_FEATURES_TABLE, DIV_FEATURES_UPSERT_SQL
from app.k_data_loader import load_k_data
from app.logger import get_logger
from sqlalchemy import text

logger = get_logger(__name__)


def pivots_confirmed_vectorized(src: np.ndarray, prd: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化版本的 pivots_confirmed
    Return (ph_conf, pl_conf) arrays of shape (n,)
    ph_conf[t] = pivot high value (at pivot bar t-prd) if confirmed at bar t else nan
    pl_conf[t] = pivot low value (at pivot bar t-prd)  if confirmed at bar t else nan
    """
    n = len(src)
    w = prd
    
    ph = np.full(n, np.nan, dtype=float)
    pl = np.full(n, np.nan, dtype=float)
    
    if n < 2 * w + 1:
        return ph, pl
    
    # 使用滑动窗口向量化计算
    window_size = 2 * w + 1
    shape = (n - 2 * w, window_size)
    strides = (src.strides[0], src.strides[0])
    windows = np.lib.stride_tricks.as_strided(src, shape=shape, strides=strides)
    
    # 获取窗口中心值
    center_vals = windows[:, w]
    
    # 计算窗口最大最小值（忽略nan）
    window_max = np.nanmax(windows, axis=1)
    window_min = np.nanmin(windows, axis=1)
    
    # 找出满足条件的pivot点
    valid_mask = np.isfinite(center_vals)
    pivot_high_mask = valid_mask & (center_vals == window_max)
    pivot_low_mask = valid_mask & (center_vals == window_min)
    
    # 设置结果
    ph[2 * w:] = np.where(pivot_high_mask, center_vals, np.nan)
    pl[2 * w:] = np.where(pivot_low_mask, center_vals, np.nan)
    
    return ph, pl


def compute_divergence_features_at_bar(
    df: pd.DataFrame, 
    bar_idx: int, 
    ind: Dict[str, pd.Series],
    cfg: DivConfig,
    ph_conf: np.ndarray,
    pl_conf: np.ndarray,
    ph_positions: List[int],
    ph_vals: List[float],
    pl_positions: List[int],
    pl_vals: List[float]
) -> Optional[Dict]:
    """计算指定 bar 位置的背离特征"""
    n = len(df)
    if bar_idx < 60:
        return None
    
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    
    close_arr = close
    high_arr = high
    low_arr = low
    
    div_features = {}
    
    for indicator_key in ["macd", "hist", "obv"]:
        if indicator_key not in ind:
            continue
        
        src_arr = ind[indicator_key].to_numpy(dtype=float)
        
        divs4 = calculate_divs(
            enabled=True,
            src=src_arr,
            close=close_arr, high=high_arr, low=low_arr,
            ph_positions=ph_positions, ph_vals=ph_vals,
            pl_positions=pl_positions, pl_vals=pl_vals,
            t=bar_idx, cfg=cfg
        )
        
        prefix = indicator_key
        div_features[f"{prefix}_pos_reg_div"] = divs4[0]
        div_features[f"{prefix}_neg_reg_div"] = divs4[1]
        div_features[f"{prefix}_pos_hid_div"] = divs4[2]
        div_features[f"{prefix}_neg_hid_div"] = divs4[3]
        
        div_features[f"{prefix}_has_div"] = 1 if any(d > 0 for d in divs4) else 0
        # div_type: 0=底背离(正背离), 1=顶背离(负背离), 2=隐藏底背离, 3=隐藏顶背离, -1=无背离
        div_features[f"{prefix}_div_type"] = next((i for i, d in enumerate(divs4) if d > 0), -1)
        div_features[f"{prefix}_div_len"] = next((d for d in divs4 if d > 0), 0)
    
    return div_features


def compute_all_bars_divergence(df: pd.DataFrame, cfg: DivConfig) -> List[Dict]:
    """计算所有 bar 的背离特征"""
    n = len(df)
    if n < 60:
        return []
    
    ind = compute_indicators(df)
    
    pivot_src_high = (df["close"] if cfg.source == "Close" else df["high"]).to_numpy(dtype=float)
    pivot_src_low = (df["close"] if cfg.source == "Close" else df["low"]).to_numpy(dtype=float)
    
    ph_conf_high, pl_conf_high = pivots_confirmed_vectorized(pivot_src_high, cfg.prd)
    ph_conf_low, pl_conf_low = pivots_confirmed_vectorized(pivot_src_low, cfg.prd)
    ph_conf = ph_conf_high
    pl_conf = pl_conf_low
    
    maxarraysize = 20
    ph_positions: List[int] = []
    pl_positions: List[int] = []
    ph_vals: List[float] = []
    pl_vals: List[float] = []
    
    records = []
    start_idx = 59
    
    for t in range(start_idx, n):
        if np.isfinite(ph_conf[t]):
            ph_positions.insert(0, t)
            ph_vals.insert(0, float(ph_conf[t]))
            if len(ph_positions) > maxarraysize:
                ph_positions.pop()
                ph_vals.pop()
        
        if np.isfinite(pl_conf[t]):
            pl_positions.insert(0, t)
            pl_vals.insert(0, float(pl_conf[t]))
            if len(pl_positions) > maxarraysize:
                pl_positions.pop()
                pl_vals.pop()
        
        div_feats = compute_divergence_features_at_bar(
            df, t, ind, cfg,
            ph_conf, pl_conf,
            ph_positions, ph_vals,
            pl_positions, pl_vals
        )
        
        if div_feats is None:
            continue
        
        total_div = sum(div_feats.get(f"{k}_has_div", 0) for k in ["macd", "hist", "obv"])
        
        if total_div == 0:
            continue
        
        record = {
            "bar_time": df.index[t],
            "total_div_count": total_div,
        }
        
        for k in ["macd", "hist", "obv"]:
            record[f"{k}_has_div"] = div_feats.get(f"{k}_has_div", 0)
            record[f"{k}_div_type"] = div_feats.get(f"{k}_div_type", -1)
            record[f"{k}_div_len"] = div_feats.get(f"{k}_div_len", 0)
            record[f"{k}_pos_reg"] = div_feats.get(f"{k}_pos_reg_div", 0)
            record[f"{k}_neg_reg"] = div_feats.get(f"{k}_neg_reg_div", 0)
            record[f"{k}_pos_hid"] = div_feats.get(f"{k}_pos_hid_div", 0)
            record[f"{k}_neg_hid"] = div_feats.get(f"{k}_neg_hid_div", 0)
        
        records.append(record)
    
    return records


def create_table_if_not_exists():
    """创建背离特征表"""
    with get_session() as session:
        session.execute(text(DIV_FEATURES_TABLE))
    logger.info("表 stock_div_features 已创建/确认存在")


def scan_single_stock(ts_code: str, name: str, 
                       frequencies: List[str], cfg: DivConfig) -> List[Dict]:
    """扫描单只股票的多周期背离特征"""
    results = []
    
    for freq in frequencies:
        try:
            df = load_k_data(ts_code, freq=freq)
            if df.empty or len(df) < 60:
                logger.warning(f"{name}({ts_code}) {freq}: 数据不足({len(df)} bars)")
                continue
            
            records = compute_all_bars_divergence(df, cfg)
            if records is None or len(records) == 0:
                logger.warning(f"{name}({ts_code}) {freq}: 背离计算失败")
                continue
            
            for rec in records:
                record = {
                    "ts_code": ts_code,
                    "freq": freq,
                    **rec
                }
                results.append(record)
            
            last_record = records[-1]
            div_count = records[-1].get("total_div_count", 0)
            logger.info(f"{name}({ts_code}) {freq}: {len(records)} bars, total_div={div_count}")
            
        except FileNotFoundError:
            logger.warning(f"{name}({ts_code}) {freq}: 本地数据不存在")
        except Exception as e:
            import traceback
            logger.error(f"{name}({ts_code}) {freq}: {e}")
            traceback.print_exc()
    
    return results


def save_features_to_db(features: List[Dict]) -> int:
    """保存特征到数据库"""
    if not features:
        return 0
    
    df = pd.DataFrame(features)
    
    with get_session() as session:
        records = df.to_dict(orient="records")
        for record in records:
            session.execute(text(DIV_FEATURES_UPSERT_SQL), record)
    
    return len(df)


def scan_stock_pool(stock_pool_path: str, frequencies: List[str], 
                    limit: Optional[int] = None, 
                    filter_name: Optional[str] = None) -> int:
    """扫描股票池中所有股票"""
    df_pool = pd.read_excel(stock_pool_path)
    logger.info(f"股票池共 {len(df_pool)} 只股票")
    
    if filter_name:
        df_pool = df_pool[df_pool['name'].str.contains(filter_name, na=False)]
        logger.info(f"过滤后共 {len(df_pool)} 只股票")
    
    if limit:
        df_pool = df_pool.head(limit)
        logger.info(f"限制扫描前 {limit} 只股票")
    
    with get_session() as session:
        session.execute(text("TRUNCATE TABLE stock_div_features RESTART IDENTITY CASCADE"))
        logger.info("表 stock_div_features 已清空")
    
    create_table_if_not_exists()
    
    cfg = DivConfig(
        prd=5,
        source="Close",
        searchdiv="Regular",
        showlimit=1,
        maxpp=10,
        maxbars=100,
        dontconfirm=False,
        showlines=False,
        showpivot=False,
        calcmacd=True,
        calcmacda=True,
        calcrsi=False,
        calcstoc=False,
        calccci=False,
        calcmom=False,
        calcobv=True,
        calcvwmacd=False,
        calccmf=False,
        calcmfi=False,
        calcext=False,
    )
    
    all_features = []
    for _, row in tqdm(df_pool.iterrows(), total=len(df_pool), desc="扫描股票"):
        ts_code = row['ts_code']
        name = row['name']
        
        features = scan_single_stock(ts_code, name, frequencies, cfg)
        all_features.extend(features)
    
    if all_features:
        saved = save_features_to_db(all_features)
        logger.info(f"共保存 {saved} 条背离特征记录到数据库")
        return saved
    else:
        logger.warning("无有效特征数据")
        return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="扫描股票池背离特征")
    ap.add_argument("--freqs", type=str, default="15m,60m,d", help="周期列表，逗号分隔")
    ap.add_argument("--limit", type=int, default=None, help="限制扫描股票数量")
    ap.add_argument("--filter", type=str, default=None, help="按名称过滤股票")
    args = ap.parse_args()
    
    frequencies = [f.strip() for f in args.freqs.split(',')]
    logger.info(f"扫描周期: {frequencies}")
    
    saved = scan_stock_pool(
        STOCK_POOL_PATH, 
        frequencies, 
        limit=args.limit, 
        filter_name=args.filter
    )
    print(f"[DONE] 扫描完成，共保存 {saved} 条记录")
