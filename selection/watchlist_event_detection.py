# -*- coding: utf-8 -*-
"""
自选股BSM事件检测 + DSA指标 + PAVP价格记录模块

Purpose: 针对自选股批量检测BSM指标穿越上下轨事件、DSA指标(dir/vwap)及PAVP价格，结果写入stock_watchlist表
Inputs:  从stock_watchlist表读取自选股列表，从stock_k_data表读取K线数据
Outputs: 更新stock_watchlist表的bsm_event、bsm_event_date、pavp_prices、dsa_dir、dsa_vwap字段
How to Run:
    python selection/watchlist_event_detection.py                    # 全量检测
    python selection/watchlist_event_detection.py --dry-run          # 只输出不写库
    python selection/watchlist_event_detection.py --codes 000001,600000  # 指定股票
Examples:
    python selection/watchlist_event_detection.py
    python selection/watchlist_event_detection.py --dry-run --codes 000001,300750
Side Effects: 写入stock_watchlist表的bsm_event、bsm_event_date、pavp_prices、dsa_dir、dsa_vwap字段
"""

import os
import sys
import argparse
import logging

import pandas as pd
import numpy as np
from sqlalchemy import text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from features.bbmacd_viewer import compute_bbmacd
from features.pavp_tv_fixed_params_factors import compute_pavp
from features.dynamic_swing_anchored_vwap import dynamic_swing_anchored_vwap, DSAConfig

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL)

EVENT_MAP = {
    "compra": "上穿上轨",
    "cross_down_upper": "下穿上轨",
    "cross_up_lower": "上穿下轨",
    "venta": "下穿下轨",
}

DSA_CFG = DSAConfig(prd=50, baseAPT=20, useAdapt=False, volBias=10)


def normalize_ts_code(ts_code: str) -> str:
    return str(ts_code).strip().upper().split('.')[0]


def format_ts_code_for_kdata(ts_code: str) -> str:
    """将纯数字代码转为带交易所后缀的格式"""
    code = normalize_ts_code(ts_code)
    if code.startswith('6') or code.startswith('9'):
        return code + '.SH'
    elif code.startswith('8') or code.startswith('4'):
        return code + '.BJ'
    else:
        return code + '.SZ'


def get_kline_data(ts_code: str, freq: str = 'd', bars: int = 60) -> pd.DataFrame:
    """从数据库获取K线数据

    Args:
        ts_code: 股票代码（纯数字或带后缀）
        freq: 频率 'd'日线, 'w'周线
        bars: 获取最近多少根K线
    """
    symbol = normalize_ts_code(ts_code)
    symbol_sh = f'{symbol}.SH'
    symbol_sz = f'{symbol}.SZ'

    sql = """
        SELECT bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE (ts_code = :ts_code OR ts_code = :ts_code_sh OR ts_code = :ts_code_sz) AND freq = :freq
        ORDER BY bar_time DESC
        LIMIT :bars
    """
    params = {
        'ts_code': symbol,
        'ts_code_sh': symbol_sh,
        'ts_code_sz': symbol_sz,
        'freq': freq,
        'bars': bars,
    }

    df = pd.read_sql(text(sql), engine, params=params)
    if not df.empty:
        df = df.sort_values('bar_time').set_index('bar_time')
    return df


def detect_events_for_stock(ts_code: str, freq: str = 'd') -> dict:
    """检测单只股票的BSM事件和PAVP价格

    Args:
        ts_code: 股票代码（纯数字或带后缀）
        freq: K线频率

    Returns:
        dict: {ts_code, bsm_event, bsm_event_date, pavp_prices, dsa_dir, dsa_vwap}
    """
    result = {
        'ts_code': normalize_ts_code(ts_code),
        'bsm_event': '',
        'bsm_event_date': None,
        'pavp_prices': '',
        'dsa_dir': None,
        'dsa_vwap': None,
    }

    df = get_kline_data(ts_code, freq=freq, bars=250)
    if len(df) < 26:
        return result

    # BSM 事件检测
    try:
        bbmacd_df = compute_bbmacd(df)
        if not bbmacd_df.empty:
            last_bar = bbmacd_df.iloc[-1]
            events = []
            for field, event_name in EVENT_MAP.items():
                if field in bbmacd_df.columns and bool(last_bar.get(field, False)):
                    events.append(event_name)
            result['bsm_event'] = ','.join(events)
            result['bsm_event_date'] = bbmacd_df.index[-1].date() if hasattr(bbmacd_df.index[-1], 'date') else str(bbmacd_df.index[-1])[:10]
    except Exception as e:
        logger.warning(f"计算BSM指标失败 {ts_code}: {e}")

    # DSA 指标计算（与个股分析页面日线逻辑保持一致）
    try:
        vwap_series, dir_series, _pivot_labels, _segments = dynamic_swing_anchored_vwap(df, DSA_CFG)
        if not vwap_series.empty:
            raw_vwap = vwap_series.iloc[-1]
            if pd.notna(raw_vwap):
                result['dsa_vwap'] = float(raw_vwap)
        if not dir_series.empty:
            raw_dir = dir_series.iloc[-1]
            if pd.notna(raw_dir):
                result['dsa_dir'] = int(raw_dir)
    except Exception as e:
        logger.warning(f"计算DSA指标失败 {ts_code}: {e}")

    # PAVP 价格检测（至少需要40根K线）
    if len(df) >= 40:
        try:
            pavp_df, _fixed_segments, _last_dev = compute_pavp(df)
            if not pavp_df.empty:
                last_pavp = pavp_df.iloc[-1]
                vah = last_pavp.get('vah_price')
                poc = last_pavp.get('poc_price')
                val = last_pavp.get('val_price')
                parts = []
                if pd.notna(vah):
                    parts.append(f"VAH:{vah:.2f}")
                if pd.notna(poc):
                    parts.append(f"POC:{poc:.2f}")
                if pd.notna(val):
                    parts.append(f"VAL:{val:.2f}")
                result['pavp_prices'] = ','.join(parts)
        except Exception as e:
            logger.warning(f"计算PAVP指标失败 {ts_code}: {e}")

    return result


def load_watchlist_codes() -> list:
    """从数据库读取所有自选股代码"""
    sql = "SELECT ts_code FROM stock_watchlist ORDER BY sort_order, added_date DESC"
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return [row[0] for row in result]


def save_events(results: list):
    """批量更新stock_watchlist表的bsm_event、bsm_event_date、pavp_prices、dsa_dir、dsa_vwap字段"""
    if not results:
        return

    with engine.connect() as conn:
        for r in results:
            update_sql = text("""
                UPDATE stock_watchlist
                SET bsm_event = :bsm_event,
                    bsm_event_date = :bsm_event_date,
                    pavp_prices = :pavp_prices,
                    dsa_dir = :dsa_dir,
                    dsa_vwap = :dsa_vwap
                WHERE ts_code = :ts_code
            """)
            conn.execute(update_sql, {
                'ts_code': r['ts_code'],
                'bsm_event': r['bsm_event'],
                'bsm_event_date': r['bsm_event_date'],
                'pavp_prices': r['pavp_prices'],
                'dsa_dir': r['dsa_dir'],
                'dsa_vwap': r['dsa_vwap'],
            })
        conn.commit()


def detect_and_save_bsm_events(dry_run: bool = False, codes: list = None):
    """检测自选股BSM事件和PAVP价格并保存到数据库

    Args:
        dry_run: 只输出不写库
        codes: 指定股票代码列表，为None时从数据库读取全部自选股
    """
    if codes is None:
        ts_codes = load_watchlist_codes()
    else:
        ts_codes = codes

    if not ts_codes:
        logger.info("无自选股需要检测")
        return

    logger.info(f"开始检测 {len(ts_codes)} 只自选股的BSM事件和PAVP价格")

    results = []
    event_count = 0

    for i, ts_code in enumerate(ts_codes, 1):
        try:
            r = detect_events_for_stock(ts_code, freq='d')
            results.append(r)

            parts = []
            if r['bsm_event']:
                event_count += 1
                parts.append(f"BSM:{r['bsm_event']}")
            if r['dsa_dir'] is not None:
                dir_text = "多头" if r['dsa_dir'] > 0 else "空头"
                parts.append(f"DSA:{dir_text}")
            if r['dsa_vwap'] is not None:
                parts.append(f"VWAP:{r['dsa_vwap']:.2f}")
            if r['pavp_prices']:
                parts.append(r['pavp_prices'])

            if parts:
                logger.info(f"[{i}/{len(ts_codes)}] {ts_code}: {' | '.join(parts)}")
            else:
                logger.debug(f"[{i}/{len(ts_codes)}] {ts_code}: 无事件")

        except Exception as e:
            logger.error(f"[{i}/{len(ts_codes)}] {ts_code}: 检测失败 - {e}")
            results.append({
                'ts_code': normalize_ts_code(ts_code),
                'bsm_event': '',
                'bsm_event_date': None,
                'pavp_prices': '',
                'dsa_dir': None,
                'dsa_vwap': None,
            })

    logger.info(f"检测完成: {len(ts_codes)} 只股票中有 {event_count} 只触发BSM事件")

    if dry_run:
        logger.info("[dry-run] 不写入数据库，结果预览:")
        for r in results:
            parts = []
            if r['bsm_event']:
                parts.append(f"BSM:{r['bsm_event']}")
            if r['dsa_dir'] is not None:
                dir_text = "多头" if r['dsa_dir'] > 0 else "空头"
                parts.append(f"DSA:{dir_text}")
            if r['dsa_vwap'] is not None:
                parts.append(f"VWAP:{r['dsa_vwap']:.2f}")
            if r['pavp_prices']:
                parts.append(r['pavp_prices'])
            if parts:
                logger.info(f"  {r['ts_code']}: {' | '.join(parts)}")
    else:
        save_events(results)
        logger.info(f"已更新 {len(results)} 条记录到 stock_watchlist 表")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    parser = argparse.ArgumentParser(description='自选股BSM事件检测、DSA指标与PAVP价格记录')
    parser.add_argument('--dry-run', action='store_true', help='只输出不写库')
    parser.add_argument('--codes', type=str, help='指定股票代码，逗号分隔，如 000001,600000')
    args = parser.parse_args()

    codes = None
    if args.codes:
        codes = [c.strip() for c in args.codes.split(',') if c.strip()]

    detect_and_save_bsm_events(dry_run=args.dry_run, codes=codes)
