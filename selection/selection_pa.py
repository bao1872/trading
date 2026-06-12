#!/usr/bin/env python3
"""
PA (Price Action) 选股脚本（基于日线CHOCH/BOS/扫流动性事件）

Purpose: 选出日线出现CHoCH/BOS/扫流动性6类事件的股票
Inputs: stock_k_data (日线K线数据)
Outputs: pa_selection (选股结果表)
How to Run:
    python selection/selection_pa.py              # 当天
    python selection/selection_pa.py 2026-05-19  # 指定日期
    python selection/selection_pa.py --test 300133   # 测试单只
    python selection/selection_pa.py --no-save       # 不写入数据库
    python selection/selection_pa.py --backfill 2025-01-01 2026-05-19  # 回补历史
Side Effects: 写入 pa_selection 表（幂等：同一日期先删后插）

================================================================================
【选股逻辑】

选股条件（满足任一即入选）：
  1) CHoCH↑ (evt_pat_choch_up)   : 趋势由下转上，收盘突破前高（首次突破）
  2) BoS↑   (evt_pat_bos_up)     : 上升趋势延续，收盘突破前高（非首次）
  3) CHoCH↓ (evt_pat_choch_down) : 趋势由上转下，收盘跌破前低（首次跌破）
  4) BoS↓   (evt_pat_bos_down)   : 下降趋势延续，收盘跌破前低（非首次）
  5) 扫高收回 (evt_pat_upper_liquidity_sweep) : 高点突破流动性线后收盘收回线下
  6) 扫低收回 (evt_pat_lower_liquidity_sweep) : 低点跌破流动性线后收盘收回线上

过滤条件：
  - 过去5天平均成交额 >= 1亿

观察项：
  - change_pct：选股日涨跌幅
  - vol_zscore：成交量 Z-Score
  - bbmacd_event：BBMACD 事件
  - daily_bb_width_zscore：日线布林带宽度 Z-Score
  - pat_trend_state：PA趋势状态（1=上升, -1=下降）
  - pat_last_swing_high/low：最近swing高低价格
  - pat_atr14：ATR(14)
  - avg_amount_20d：过去20日平均成交额（(open+close)/2 * volume）
  - dsa_dir_bars：当前 DSA VWAP dir 趋势持续的 bar 数

【核心计算】
  - 全部引用 features.price_action_toolkit_lite_ualgo.compute_price_action_toolkit
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
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List
from tqdm import tqdm

from features.price_action_toolkit_lite_ualgo import compute_price_action_toolkit, PATConfig
from features.bbmacd_viewer import compute_bbmacd
from features.dynamic_swing_anchored_vwap import dynamic_swing_anchored_vwap, DSAConfig

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "pa_selection"

DAILY_BARS = 800

PA_EVENTS = [
    ("evt_pat_choch_up", "evt_choch_up", "CHoCH↑"),
    ("evt_pat_bos_up", "evt_bos_up", "BoS↑"),
    ("evt_pat_choch_down", "evt_choch_down", "CHoCH↓"),
    ("evt_pat_bos_down", "evt_bos_down", "BoS↓"),
    ("evt_pat_upper_liquidity_sweep", "evt_upper_liq_sweep", "扫高收回"),
    ("evt_pat_lower_liquidity_sweep", "evt_lower_liq_sweep", "扫低收回"),
    ("evt_pat_upper_sweep_fail_up", "evt_upper_sweep_fail_up", "扫高失败↑"),
    ("evt_pat_lower_sweep_fail_down", "evt_lower_sweep_fail_down", "扫低失败↓"),
]


def normalize_ts_code(ts_code: str) -> str:
    return str(ts_code).strip().upper().split('.')[0]


def compute_dsa_dir_bars(df: pd.DataFrame, cfg: DSAConfig = None) -> pd.Series:
    """
    计算 DSA VWAP dir 持续的 bar 数
    正数=dir=1 持续, 负数=dir=-1 持续, 0=无方向
    """
    if cfg is None:
        cfg = DSAConfig()
    _, dir_series, _, _ = dynamic_swing_anchored_vwap(df, cfg)
    dir_vals = dir_series.fillna(0).astype(int)
    bars = pd.Series(0, index=df.index, dtype=int)
    for i in range(len(dir_vals)):
        if i == 0:
            bars.iloc[i] = dir_vals.iloc[i]
        elif dir_vals.iloc[i] == dir_vals.iloc[i - 1]:
            bars.iloc[i] = bars.iloc[i - 1] + dir_vals.iloc[i]
        else:
            bars.iloc[i] = dir_vals.iloc[i]
    return bars


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
    """检查成交额过滤条件：过去N天平均成交额 >= min_amount"""
    if len(df) < days:
        return False
    recent_df = df.tail(days)
    daily_amount = recent_df['volume'] * recent_df['close']
    avg_amount = daily_amount.mean()
    return avg_amount >= min_amount


def volume_zscore(vol: pd.Series, win: int = 20) -> Optional[float]:
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


def compute_change_pct(daily_df: pd.DataFrame) -> Optional[float]:
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


def batch_get_stock_names(ts_codes: List[str]) -> Dict[str, str]:
    """批量获取股票名称（一次查询，同时匹配带/不带交易所后缀）"""
    if not ts_codes:
        return {}
    all_codes = set()
    for c in ts_codes:
        all_codes.add(c)
        all_codes.add(f"{c}.SH")
        all_codes.add(f"{c}.SZ")
    placeholders = ', '.join([f"'{c}'" for c in all_codes])
    sql = text(f"SELECT ts_code, name FROM stock_pools WHERE ts_code IN ({placeholders})")
    with engine.connect() as conn:
        result = conn.execute(sql)
        name_map = {}
        for row in result:
            code_with_suffix = row[0]
            name = row[1]
            name_map[code_with_suffix] = name
            code_bare = code_with_suffix.split('.')[0]
            name_map[code_bare] = name
        return name_map


def _check_top_signal(triggered_events: Dict[str, bool], vol_zscore_20: Optional[float], pat_trend_state: Optional[int]) -> bool:
    """判断是否属于最强组合

    最强组合规则（基于全量回测验证）：
      1) 事件 = 扫低失败↓(evt_lower_sweep_fail_down) 或 扫低收回(evt_lower_liq_sweep)
      2) vol_zscore_20 > 1（高量）
      3) pat_trend_state = 1（趋势↑）
    """
    if vol_zscore_20 is None or pat_trend_state != 1:
        return False
    if vol_zscore_20 <= 1:
        return False
    if triggered_events.get('evt_lower_sweep_fail_down') or triggered_events.get('evt_lower_liq_sweep'):
        return True
    return False


def process_stock_pa(ts_code: str, selection_date: date) -> Optional[Dict]:
    """
    处理单只股票的PA事件检测

    选股逻辑：
        - 获取日线数据，调用 compute_price_action_toolkit 计算 PA 指标
        - 检查最后一根 bar 是否出现 6 类事件中的任意一个
        - 过滤：过去5天平均成交额 >= 1亿

    Returns: 信号字典，如果满足条件则返回结果，否则返回None
    """
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
    if daily_df.empty or len(daily_df) < 60:
        return None

    if 'volume' not in daily_df.columns:
        daily_df['volume'] = np.nan

    cfg = PATConfig(
        zigzag_len=9,
        liquidity_len=20,
        trend_line_len=20,
        number_ob_show=2,
        show_market_structure=True,
        show_liquidity=True,
        show_order_blocks=True,
        show_trend_lines=True,
    )

    try:
        pat_df, _ = compute_price_action_toolkit(daily_df, cfg)
    except Exception:
        return None

    if pat_df.empty:
        return None

    last = pat_df.iloc[-1]

    triggered_events = {}
    has_event = False
    for src_col, dst_col, label in PA_EVENTS:
        if src_col in pat_df.columns:
            val = last.get(src_col)
            triggered = bool(val) if pd.notna(val) else False
            triggered_events[dst_col] = triggered
            if triggered:
                has_event = True
        else:
            triggered_events[dst_col] = False

    if not has_event:
        return None

    if not check_volume_filter(daily_df, days=5, min_amount=100_000_000):
        return None

    bbmacd_df = compute_bbmacd(daily_df)
    bbmacd_event = detect_bbmacd_event(bbmacd_df)

    bar_time = daily_df.index[-1]

    pat_trend_state = None
    if 'pat_trend_state' in pat_df.columns and pd.notna(last.get('pat_trend_state')):
        pat_trend_state = int(last['pat_trend_state'])

    pat_last_swing_high = None
    if 'pat_last_swing_high' in pat_df.columns and pd.notna(last.get('pat_last_swing_high')):
        pat_last_swing_high = float(last['pat_last_swing_high'])

    pat_last_swing_low = None
    if 'pat_last_swing_low' in pat_df.columns and pd.notna(last.get('pat_last_swing_low')):
        pat_last_swing_low = float(last['pat_last_swing_low'])

    pat_atr14 = None
    if 'pat_atr14' in pat_df.columns and pd.notna(last.get('pat_atr14')):
        pat_atr14 = float(last['pat_atr14'])

    result = {
        'ts_code': ts_code,
        'signal_date': bar_time,
        **triggered_events,
        'pat_trend_state': pat_trend_state,
        'pat_last_swing_high': pat_last_swing_high,
        'pat_last_swing_low': pat_last_swing_low,
        'pat_atr14': pat_atr14,
        'change_pct': compute_change_pct(daily_df),
        'vol_zscore': volume_zscore(daily_df['volume'], win=20),
        'vol_zscore_5': volume_zscore(daily_df['volume'], win=5),
        'vol_zscore_10': volume_zscore(daily_df['volume'], win=10),
        'bbmacd_event': bbmacd_event,
        'daily_bb_width_zscore': float(bbmacd_df['bb_width_zscore'].iloc[-1]) if 'bb_width_zscore' in bbmacd_df.columns and len(bbmacd_df) > 0 else None,
        'avg_amount_20d': float(((daily_df['open'] + daily_df['close']) / 2 * daily_df['volume']).tail(20).mean()) if len(daily_df) >= 20 else None,
        'dsa_dir_bars': None,  # 后面填充
    }

    # DSA VWAP dir 持续 bar 数
    try:
        dsa_bars = compute_dsa_dir_bars(daily_df)
        result['dsa_dir_bars'] = int(dsa_bars.iloc[-1])
    except Exception:
        pass

    # 标注最强组合
    result['is_top_signal'] = _check_top_signal(
        triggered_events, result['vol_zscore'], result['pat_trend_state']
    )

    return result


def ensure_table_exists():
    """确保 pa_selection 表存在"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS pa_selection (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        signal_date DATE,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),

        evt_choch_up BOOLEAN DEFAULT FALSE,
        evt_bos_up BOOLEAN DEFAULT FALSE,
        evt_choch_down BOOLEAN DEFAULT FALSE,
        evt_bos_down BOOLEAN DEFAULT FALSE,
        evt_upper_liq_sweep BOOLEAN DEFAULT FALSE,
        evt_lower_liq_sweep BOOLEAN DEFAULT FALSE,
        evt_upper_sweep_fail_up BOOLEAN DEFAULT FALSE,
        evt_lower_sweep_fail_down BOOLEAN DEFAULT FALSE,

        pat_trend_state INT,
        pat_last_swing_high FLOAT,
        pat_last_swing_low FLOAT,
        pat_atr14 FLOAT,

        change_pct FLOAT,
        vol_zscore FLOAT,
        vol_zscore_5 FLOAT,
        vol_zscore_10 FLOAT,
        bbmacd_event VARCHAR(20),
        daily_bb_width_zscore FLOAT,

        is_top_signal BOOLEAN DEFAULT FALSE,

        batch_no INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_pa_selection_date ON pa_selection(selection_date);
    CREATE INDEX IF NOT EXISTS idx_pa_ts_code ON pa_selection(ts_code);
    CREATE INDEX IF NOT EXISTS idx_pa_choch_up ON pa_selection(evt_choch_up);
    CREATE INDEX IF NOT EXISTS idx_pa_bos_up ON pa_selection(evt_bos_up);
    CREATE INDEX IF NOT EXISTS idx_pa_choch_down ON pa_selection(evt_choch_down);
    CREATE INDEX IF NOT EXISTS idx_pa_bos_down ON pa_selection(evt_bos_down);
    CREATE INDEX IF NOT EXISTS idx_pa_upper_sweep ON pa_selection(evt_upper_liq_sweep);
    CREATE INDEX IF NOT EXISTS idx_pa_lower_sweep ON pa_selection(evt_lower_liq_sweep);
    CREATE INDEX IF NOT EXISTS idx_pa_batch_no ON pa_selection(batch_no);
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()

    # 为已有表添加新列（幂等）
    alter_sqls = [
        "ALTER TABLE pa_selection ADD COLUMN IF NOT EXISTS evt_upper_sweep_fail_up BOOLEAN DEFAULT FALSE",
        "ALTER TABLE pa_selection ADD COLUMN IF NOT EXISTS evt_lower_sweep_fail_down BOOLEAN DEFAULT FALSE",
        "ALTER TABLE pa_selection ADD COLUMN IF NOT EXISTS vol_zscore_5 FLOAT",
        "ALTER TABLE pa_selection ADD COLUMN IF NOT EXISTS vol_zscore_10 FLOAT",
        "ALTER TABLE pa_selection ADD COLUMN IF NOT EXISTS is_top_signal BOOLEAN DEFAULT FALSE",
        "ALTER TABLE pa_selection ADD COLUMN IF NOT EXISTS avg_amount_20d FLOAT",
        "ALTER TABLE pa_selection ADD COLUMN IF NOT EXISTS dsa_dir_bars INT",
    ]
    with engine.connect() as conn:
        for sql in alter_sqls:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                conn.rollback()


def save_to_database(df: pd.DataFrame, selection_date: date) -> int:
    """保存选股结果到数据库（幂等性：先删后插）"""
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

    return _insert_records(df, selection_date)


def append_to_database(df: pd.DataFrame, selection_date: date) -> int:
    """追加选股结果到数据库（不删除旧数据，用于回补增量保存）"""
    if df.empty:
        return 0

    ensure_table_exists()
    return _insert_records(df, selection_date)


def _insert_records(df: pd.DataFrame, selection_date: date) -> int:
    records = []
    for _, row in df.iterrows():
        record = {
            'selection_date': selection_date,
            'signal_date': row['signal_date'],
            'ts_code': row['ts_code'],
            'stock_name': row.get('stock_name', '') or '',
            'evt_choch_up': bool(row.get('evt_choch_up', False)),
            'evt_bos_up': bool(row.get('evt_bos_up', False)),
            'evt_choch_down': bool(row.get('evt_choch_down', False)),
            'evt_bos_down': bool(row.get('evt_bos_down', False)),
            'evt_upper_liq_sweep': bool(row.get('evt_upper_liq_sweep', False)),
            'evt_lower_liq_sweep': bool(row.get('evt_lower_liq_sweep', False)),
            'evt_upper_sweep_fail_up': bool(row.get('evt_upper_sweep_fail_up', False)),
            'evt_lower_sweep_fail_down': bool(row.get('evt_lower_sweep_fail_down', False)),
            'pat_trend_state': int(row['pat_trend_state']) if pd.notna(row.get('pat_trend_state')) else None,
            'pat_last_swing_high': float(row['pat_last_swing_high']) if pd.notna(row.get('pat_last_swing_high')) else None,
            'pat_last_swing_low': float(row['pat_last_swing_low']) if pd.notna(row.get('pat_last_swing_low')) else None,
            'pat_atr14': float(row['pat_atr14']) if pd.notna(row.get('pat_atr14')) else None,
            'change_pct': float(row['change_pct']) if pd.notna(row.get('change_pct')) else None,
            'vol_zscore': float(row['vol_zscore']) if pd.notna(row.get('vol_zscore')) else None,
            'vol_zscore_5': float(row['vol_zscore_5']) if pd.notna(row.get('vol_zscore_5')) else None,
            'vol_zscore_10': float(row['vol_zscore_10']) if pd.notna(row.get('vol_zscore_10')) else None,
            'bbmacd_event': row.get('bbmacd_event', '无') or '无',
            'daily_bb_width_zscore': float(row['daily_bb_width_zscore']) if pd.notna(row.get('daily_bb_width_zscore')) else None,
            'is_top_signal': bool(row.get('is_top_signal', False)),
            'avg_amount_20d': float(row['avg_amount_20d']) if pd.notna(row.get('avg_amount_20d')) else None,
            'dsa_dir_bars': int(row['dsa_dir_bars']) if pd.notna(row.get('dsa_dir_bars')) else None,
            'batch_no': int(row['batch_no']) if pd.notna(row.get('batch_no')) else None,
        }
        records.append(record)

    if records:
        insert_df = pd.DataFrame(records)
        insert_df.to_sql(SELECTION_TABLE, engine, if_exists='append', index=False)
        print(f"  保存新数据: {len(records)} 条")
        return len(records)

    return 0


def select_pa_stocks(selection_date: Optional[date] = None, save_to_db: bool = True) -> pd.DataFrame:
    """
    根据 PA (Price Action) 指标选出满足条件的股票

    Args:
        selection_date: 选股日期，默认为当天
        save_to_db: 是否保存到数据库
    """
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("选股条件（PA Price Action 策略）：")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  事件: CHoCH↑/↓, BoS↑/↓, 扫高收回, 扫低收回")
    print(f"  过滤条件: 过去5天平均成交额 >= 1亿")
    print(f"  PATConfig: zigzag_len=9, liquidity_len=20, trend_line_len=20")
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
    print("开始 PA 事件筛选...")
    print(f"  原股票数: {len(stock_list)}")

    filtered_results = []

    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="PA选股", unit="只"):
        ts_code = row['ts_code']
        result = process_stock_pa(ts_code, selection_date)
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
    print(f"PA事件筛选后: {len(result_df)} 只")

    if not result_df.empty:
        print(f"\n事件统计：")
        for _, dst_col, label in PA_EVENTS:
            if dst_col in result_df.columns:
                cnt = result_df[dst_col].sum()
                if cnt > 0:
                    print(f"  {label}: {cnt} 只")

        batch_count = result_df['batch_no'].max()
        print(f"\n批次信息：共 {batch_count} 批，每批10只股票")

        print("\n" + "=" * 80)
        print("前20名股票：")
        print("=" * 80)
        display_cols = ['ts_code', 'stock_name'] + [dc for _, dc, _ in PA_EVENTS] + ['pat_trend_state', 'change_pct']
        print_cols = [c for c in display_cols if c in result_df.columns]
        print(result_df[print_cols].head(20).to_string(index=False))

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
    """测试单只股票的PA事件检测"""
    print("\n" + "=" * 80)
    print(f"测试单只股票: {ts_code}")
    print(f"选股日期: {selection_date}")
    print("=" * 80)

    result = process_stock_pa(ts_code, selection_date)

    if result:
        print("\n选股结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("\n该股票不满足选股条件（无PA事件或成交额不足）")

    return result


def backfill_selection_results(
    start_date: date,
    end_date: date,
    save_to_db: bool = True,
) -> Dict[date, int]:
    """
    回补指定日期范围内的选股结果（外层遍历股票，高效）

    逻辑：
        1. 一次性获取目标日期范围内有交易数据的股票列表
        2. 外层遍历股票，每只股票加载全部K线 + 800根预热数据
        3. 一次性 compute_price_action_toolkit 计算所有bar的PA事件
        4. 筛选 [start_date, end_date] 范围内有事件的bar
        5. 按日期分组，批量写入数据库

    Args:
        start_date: 回补开始日期
        end_date: 回补结束日期
        save_to_db: 是否保存到数据库

    Returns:
        字典，key为日期，value为当天选股结果数量
    """
    from collections import defaultdict

    results = defaultdict(list)

    print("\n" + "=" * 80)
    print(f"开始回补PA选股结果")
    print(f"  开始日期: {start_date.strftime('%Y-%m-%d')}")
    print(f"  结束日期: {end_date.strftime('%Y-%m-%d')}")
    print(f"  保存到数据库: {save_to_db}")
    print(f"  模式: 外层遍历股票（高效）")
    print("=" * 80)

    # 1. 获取目标日期范围内有交易的股票（用最新交易日）
    with engine.connect() as conn:
        sql = text("""
            SELECT DISTINCT ts_code
            FROM stock_k_data
            WHERE freq = 'd' AND DATE(bar_time) = :end_date
        """)
        stock_list_df = pd.read_sql(sql, conn, params={'end_date': end_date.strftime('%Y-%m-%d')})
    stock_list = stock_list_df['ts_code'].tolist()
    print(f"\n共 {len(stock_list)} 只股票需要处理\n")

    # 2. 获取所有股票名称
    stock_names = batch_get_stock_names(stock_list)

    # 3. 外层遍历股票
    batch_size = 100  # 每处理100只股票保存一次
    processed_count = 0

    for ts_code in tqdm(stock_list, desc="回补进度", unit="只"):
        try:
            df = get_kline_data_db(ts_code, bars=1500, end_date=end_date)
            if df.empty or len(df) < 60:
                continue

            if 'volume' not in df.columns:
                df['volume'] = np.nan

            cfg = PATConfig(
                zigzag_len=9,
                liquidity_len=20,
                trend_line_len=20,
                number_ob_show=2,
                show_market_structure=True,
                show_liquidity=True,
                show_order_blocks=True,
                show_trend_lines=True,
            )

            pat_df, _ = compute_price_action_toolkit(df, cfg)
            if pat_df.empty:
                continue

            # 一次计算全量 DSA dir bars（供回补循环 lookup）
            try:
                dsa_bars_series = compute_dsa_dir_bars(df)
            except Exception:
                dsa_bars_series = None

            # 筛选目标日期范围内的bar（索引可能为object/datetime.date类型）
            target_df = pat_df[[idx.date() >= start_date if hasattr(idx, 'date') else idx >= start_date for idx in pat_df.index]].copy()
            if target_df.empty:
                continue

            # 检查每根bar是否有事件，按 (selection_date, ts_code) 合并
            stock_events = {}  # key: sel_date, value: merged record dict

            for bar_time in target_df.index:
                last = target_df.loc[bar_time]

                triggered_events = {}
                has_event = False
                for src_col, dst_col, label in PA_EVENTS:
                    if src_col in pat_df.columns:
                        val = last.get(src_col)
                        triggered = bool(val) if pd.notna(val) else False
                        triggered_events[dst_col] = triggered
                        if triggered:
                            has_event = True
                    else:
                        triggered_events[dst_col] = False

                if not has_event:
                    continue

                # 成交额过滤（用该bar之前5天的数据）
                try:
                    bar_idx = df.index.get_loc(bar_time)
                except KeyError:
                    bar_idx = None
                    for i, idx in enumerate(df.index):
                        if hasattr(idx, 'date') and idx.date() == bar_time:
                            bar_idx = i
                            break
                        elif idx == bar_time:
                            bar_idx = i
                            break
                    if bar_idx is None:
                        continue

                pre_daily = df.iloc[max(0, bar_idx-5):bar_idx+1]
                if len(pre_daily) < 5:
                    continue
                daily_amount = pre_daily['volume'] * pre_daily['close']
                avg_amount = daily_amount.mean()
                if avg_amount < 100_000_000:
                    continue

                # 计算该bar的涨跌幅
                change_pct = None
                if bar_idx >= 1:
                    close_today = df.iloc[bar_idx]['close']
                    close_yesterday = df.iloc[bar_idx - 1]['close']
                    if close_yesterday != 0:
                        change_pct = round(float(close_today - close_yesterday) / float(close_yesterday) * 100, 2)

                # 成交量Z-Score
                vol_zscore = None
                vol_zscore_5 = None
                vol_zscore_10 = None
                if bar_idx >= 20:
                    vol_window = df['volume'].iloc[bar_idx-20:bar_idx+1]
                    mu = vol_window.mean()
                    sd = vol_window.std(ddof=0)
                    if sd > 0:
                        vol_zscore = float((df['volume'].iloc[bar_idx] - mu) / sd)
                if bar_idx >= 5:
                    vol_window5 = df['volume'].iloc[bar_idx-5:bar_idx+1]
                    mu5 = vol_window5.mean()
                    sd5 = vol_window5.std(ddof=0)
                    if sd5 > 0:
                        vol_zscore_5 = float((df['volume'].iloc[bar_idx] - mu5) / sd5)
                if bar_idx >= 10:
                    vol_window10 = df['volume'].iloc[bar_idx-10:bar_idx+1]
                    mu10 = vol_window10.mean()
                    sd10 = vol_window10.std(ddof=0)
                    if sd10 > 0:
                        vol_zscore_10 = float((df['volume'].iloc[bar_idx] - mu10) / sd10)

                sel_date = bar_time.date() if hasattr(bar_time, 'date') else bar_time

                record = {
                    'ts_code': ts_code,
                    'stock_name': stock_names.get(ts_code, ''),
                    'signal_date': bar_time,
                    **triggered_events,
                    'pat_trend_state': int(last['pat_trend_state']) if 'pat_trend_state' in pat_df.columns and pd.notna(last.get('pat_trend_state')) else None,
                    'pat_last_swing_high': float(last['pat_last_swing_high']) if 'pat_last_swing_high' in pat_df.columns and pd.notna(last.get('pat_last_swing_high')) else None,
                    'pat_last_swing_low': float(last['pat_last_swing_low']) if 'pat_last_swing_low' in pat_df.columns and pd.notna(last.get('pat_last_swing_low')) else None,
                    'pat_atr14': float(last['pat_atr14']) if 'pat_atr14' in pat_df.columns and pd.notna(last.get('pat_atr14')) else None,
                    'change_pct': change_pct,
                    'vol_zscore': vol_zscore,
                    'vol_zscore_5': vol_zscore_5,
                    'vol_zscore_10': vol_zscore_10,
                    'bbmacd_event': '无',
                    'daily_bb_width_zscore': None,
                    'avg_amount_20d': float(((df['open'].iloc[max(0,bar_idx-19):bar_idx+1] + df['close'].iloc[max(0,bar_idx-19):bar_idx+1]) / 2 * df['volume'].iloc[max(0,bar_idx-19):bar_idx+1]).mean()) if bar_idx >= 19 else None,
                    'dsa_dir_bars': int(dsa_bars_series.iloc[bar_idx]) if dsa_bars_series is not None and bar_idx < len(dsa_bars_series) else None,
                    'batch_no': None,
                }

                # 标注最强组合
                record['is_top_signal'] = _check_top_signal(
                    triggered_events, vol_zscore, record['pat_trend_state']
                )

                # 合并同一股票同一天的多个事件
                if sel_date in stock_events:
                    existing = stock_events[sel_date]
                    for dst_col, _ in [(dc, _) for _, dc, _ in PA_EVENTS]:
                        if triggered_events.get(dst_col):
                            existing[dst_col] = True
                    # 合并 is_top_signal（任一事件满足即为 True）
                    if record.get('is_top_signal'):
                        existing['is_top_signal'] = True
                else:
                    stock_events[sel_date] = record

            for sel_date, record in stock_events.items():
                results[sel_date].append(record)

            processed_count += 1
            if processed_count % batch_size == 0 and save_to_db and results:
                # 增量保存（append模式，不删除旧数据）
                for sel_date, records in list(results.items()):
                    if records:
                        save_df = pd.DataFrame(records)
                        append_to_database(save_df, sel_date)
                    results[sel_date] = []

        except Exception as e:
            tqdm.write(f"  股票 {ts_code} 错误: {e}")
            continue

    # 保存剩余结果（append模式）
    if save_to_db:
        for sel_date, records in results.items():
            if records:
                save_df = pd.DataFrame(records)
                append_to_database(save_df, sel_date)

    # 汇总
    date_counts = {}
    total = 0
    for sel_date in sorted(results.keys()):
        # 重新从数据库读取统计（因为可能已保存）
        pass

    # 从数据库读取最终统计
    with engine.connect() as conn:
        sql = text("""
            SELECT selection_date, COUNT(*) as cnt FROM pa_selection
            WHERE selection_date BETWEEN :start_date AND :end_date
            GROUP BY selection_date ORDER BY selection_date
        """)
        df_stats = pd.read_sql(sql, conn, params={'start_date': start_date, 'end_date': end_date})

    print("\n" + "=" * 80)
    print("回补完成")
    print("=" * 80)
    print(f"\n汇总:")
    print(f"  处理股票数: {processed_count}")
    if not df_stats.empty:
        total = df_stats['cnt'].sum()
        print(f"  有选股结果的天数: {len(df_stats)}")
        print(f"  总选股数: {total}")
    else:
        print(f"  无选股结果")

    return {row['selection_date']: row['cnt'] for _, row in df_stats.iterrows()}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='PA (Price Action) 选股工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/selection_pa.py                    # 使用当天日期选股
  python selection/selection_pa.py 2026-05-19         # 指定日期选股
  python selection/selection_pa.py 20260519           # 指定日期选股（无分隔符）
  python selection/selection_pa.py --test 300133      # 测试单只股票
  python selection/selection_pa.py --no-save          # 不写入数据库
  python selection/selection_pa.py --backfill 2025-01-01 2026-05-19  # 回补历史
        """
    )
    parser.add_argument(
        'date',
        nargs='?',
        help='选股日期 (格式: YYYY-MM-DD 或 YYYYMMDD)，默认为当天'
    )
    parser.add_argument(
        '--test',
        type=str,
        default=None,
        help='测试单只股票代码，例如 300133'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='不写入数据库（仅打印结果）'
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

        backfill_selection_results(
            start_date=start_date,
            end_date=end_date,
            save_to_db=True,
        )
        return

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
    print("PA (Price Action) 选股工具")
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print("=" * 80)

    if args.test:
        test_single_stock(args.test, selection_date)
        return

    df = select_pa_stocks(selection_date=selection_date, save_to_db=not args.no_save)

    print("\n" + "=" * 80)
    print("选股完成")
    print(f"选股日期: {selection_date}")
    print(f"选中股票数: {len(df)}")
    print(f"查询SQL: SELECT * FROM pa_selection WHERE selection_date = '{selection_date}'")
    print("=" * 80)

    return df


if __name__ == '__main__':
    main()
