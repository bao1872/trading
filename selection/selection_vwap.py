#!/usr/bin/env python3
"""
DSA VWAP指标选股脚本（基于DSA方向持续时间和VWAP偏离率）

Purpose: 基于DSA VWAP指标的dir=1持续时间和偏离率积分选股
Inputs: stock_k_data (日线K线数据)
Outputs: vwap_selection (选股结果表)
How to Run:
    python selection/selection_vwap.py              # 当天
    python selection/selection_vwap.py 2026-04-10  # 指定日期
Side Effects: 写入 vwap_selection 表

================================================================================
【选股条件】

保存条件：DSA dir=1（多头方向）
  - 核心条件：最新bar的DSA方向为1（多头）
  - 统计指标1：dir=1持续bar数量
  - 统计指标2：dir=1时间段内收盘价与VWAP偏离率的积分（累计偏离）
  - 统计指标3：平均偏离率 = 累计偏离率 / bar数量
  - 统计指标4：偏离率方差（衡量偏离率的波动程度）
  - 统计指标5：正值偏离率比例（收盘价>VWAP的bar占比）
  - 统计指标6：负值偏离率比例（收盘价<VWAP的bar占比）
  - 观察项：BBMACD事件、PAVP成交量分布、成交量Z-Score

指标计算：
  - DSA VWAP：使用dynamic_swing_anchored_vwap计算（与vis页面一致）
    * 参数: prd=50, baseAPT=20.0, useAdapt=False, volBias=10.0, atrLen=50
  - BBMACD：使用compute_bbmacd计算
    * 参数: rapida=8, lenta=26, stdv=0.8, signal_len=9
  - PAVP：使用compute_pavp计算（与vis页面一致）
    * 返回VAH、VAL、POC、成交量分布段

保存字段：
  - DSA字段：dsa_dir, dir1_bars, cumulative_deviation, avg_deviation, deviation_variance, positive_ratio, negative_ratio
  - 价格字段：current_vwap, current_close, current_deviation
  - BBMACD字段：bbmacd_event, bb_width_zscore
  - PAVP字段：pavp_vah, pavp_val, pavp_poc, va_pos_01
  - 其他：vol_zscore, change_pct, batch_no

批次号规则：
  - 每天从1开始编号，按处理顺序每10只股票为一批
  - 例：第1-10只 → batch_no=1，第11-20只 → batch_no=2

【选股日期】

选股日期只是标记，实际数据到"选股日期当天或之前最后一个交易日"：
  - 选股日期=2026-04-13(周一) → 数据到2026-04-13（当天是交易日）

日线数据每天更新：
  - 通过 python app/build_dataset.py --update --period d 每天更新日线数据
  - 使用日线数据计算DSA指标（bars=800，约3年数据）

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
from datetime import datetime, date
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

# 导入核心计算逻辑（SSOT原则）
from features.dynamic_swing_anchored_vwap import dynamic_swing_anchored_vwap, DSAConfig
from features.bbmacd_viewer import compute_bbmacd
from features.pavp_tv_fixed_params_factors import compute_pavp

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

SELECTION_TABLE = "vwap_selection"

# DSA配置参数（与vis页面保持一致）
DSA_CFG = DSAConfig(prd=50, baseAPT=20.0, useAdapt=False, volBias=10.0, atrLen=50)

# K线数据配置
DAILY_BARS = 800  # 日线数据获取800根（约3年）


def normalize_ts_code(ts_code: str) -> str:
    """标准化股票代码"""
    return str(ts_code).strip().upper().split('.')[0]


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


def compute_dir1_stats(dir_series: pd.Series, close_series: pd.Series, vwap_series: pd.Series) -> Dict:
    """
    计算DSA dir=1的统计指标

    Args:
        dir_series: DSA方向序列 (1/-1)
        close_series: 收盘价序列
        vwap_series: VWAP序列

    Returns:
        {
            'dir1_bars': int,  # dir=1持续bar数
            'cumulative_deviation': float,  # 累计偏离率(%)
            'avg_deviation': float,  # 平均偏离率(%)
            'deviation_variance': float,  # 偏离率方差
            'positive_ratio': float,  # 正值偏离率比例
            'negative_ratio': float,  # 负值偏离率比例
        }
    """
    n = len(dir_series)
    if n == 0:
        return {
            'dir1_bars': 0,
            'cumulative_deviation': None,
            'avg_deviation': None,
            'deviation_variance': None,
            'positive_ratio': None,
            'negative_ratio': None,
        }

    # 从最新bar向前查找dir=1的连续段
    dir1_bars = 0
    cumulative_dev = 0.0
    deviations = []  # 存储每个bar的偏离率用于计算方差

    for i in range(n - 1, -1, -1):
        if dir_series.iloc[i] == 1:
            dir1_bars += 1
            # 计算该bar的偏离率
            close = close_series.iloc[i]
            vwap = vwap_series.iloc[i]
            if pd.notna(close) and pd.notna(vwap) and vwap != 0:
                deviation = (close - vwap) / vwap * 100
                cumulative_dev += deviation
                deviations.append(deviation)
        else:
            # 遇到dir!=1，停止计数
            break

    avg_deviation = cumulative_dev / dir1_bars if dir1_bars > 0 else None

    # 计算偏离率方差（使用样本方差 ddof=1）
    deviation_variance = None
    if len(deviations) >= 2:
        deviation_variance = float(np.var(deviations, ddof=1))
    elif len(deviations) == 1:
        deviation_variance = 0.0  # 只有一个数据点时方差为0

    # 计算正值/负值偏离率比例
    positive_ratio = None
    negative_ratio = None
    if len(deviations) > 0:
        positive_count = sum(1 for d in deviations if d > 0)
        negative_count = sum(1 for d in deviations if d < 0)
        zero_count = sum(1 for d in deviations if d == 0)
        total = len(deviations)
        positive_ratio = positive_count / total if total > 0 else None
        negative_ratio = negative_count / total if total > 0 else None

    return {
        'dir1_bars': dir1_bars,
        'cumulative_deviation': cumulative_dev if dir1_bars > 0 else None,
        'avg_deviation': avg_deviation,
        'deviation_variance': deviation_variance,
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
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


def process_stock(ts_code: str, selection_date: date) -> Optional[Dict]:
    """
    处理单只股票的DSA VWAP选股逻辑
    
    选股逻辑：
        - 获取日线数据计算DSA VWAP指标
        - 条件：最新bar的DSA dir = 1（多头方向）
        - 统计：dir=1持续bar数、累计偏离率、平均偏离率
        - 观察项：BBMACD、PAVP、成交量Z-Score
    
    Returns: 信号字典，如果满足条件则返回结果，否则返回None
    """
    # 获取日线数据
    daily_df = get_kline_data_db(ts_code, bars=DAILY_BARS, end_date=selection_date)
    if daily_df.empty or len(daily_df) < 60:
        return None
    
    # 计算DSA VWAP指标
    try:
        vwap_series, dir_series, pivot_labels, segments = dynamic_swing_anchored_vwap(daily_df, DSA_CFG)
    except Exception as e:
        return None
    
    # 首要条件：最新bar的DSA dir = 1
    latest_dir = dir_series.iloc[-1]
    if latest_dir != 1:
        return None
    
    # 计算dir=1的统计指标
    dir1_stats = compute_dir1_stats(dir_series, daily_df['close'], vwap_series)
    
    # 如果dir=1持续bar数为0，不保存
    if dir1_stats['dir1_bars'] == 0:
        return None
    
    # 计算BBMACD指标
    bbmacd_df = compute_bbmacd(daily_df)
    bbmacd_event = detect_bbmacd_event(bbmacd_df)
    
    # 计算PAVP指标
    try:
        pavp_df, fixed_segments, last_dev = compute_pavp(daily_df)
        pavp_vah = pavp_df['vah_price'].iloc[-1] if 'vah_price' in pavp_df.columns else None
        pavp_val = pavp_df['val_price'].iloc[-1] if 'val_price' in pavp_df.columns else None
        pavp_poc = pavp_df['poc_price'].iloc[-1] if 'poc_price' in pavp_df.columns else None
        va_pos_01 = pavp_df['va_pos_01'].iloc[-1] if 'va_pos_01' in pavp_df.columns else None
    except Exception:
        pavp_vah = None
        pavp_val = None
        pavp_poc = None
        va_pos_01 = None
    
    # 当前价格和偏离率
    current_close = daily_df['close'].iloc[-1]
    current_vwap = vwap_series.iloc[-1]
    current_deviation = ((current_close - current_vwap) / current_vwap * 100) if current_vwap != 0 else None
    
    bar_time = daily_df.index[-1]
    
    return {
        'ts_code': ts_code,
        'dsa_dir': int(latest_dir),
        'dir1_bars': dir1_stats['dir1_bars'],
        'cumulative_deviation': dir1_stats['cumulative_deviation'],
        'avg_deviation': dir1_stats['avg_deviation'],
        'deviation_variance': dir1_stats['deviation_variance'],
        'positive_ratio': dir1_stats['positive_ratio'],
        'negative_ratio': dir1_stats['negative_ratio'],
        'current_vwap': float(current_vwap) if pd.notna(current_vwap) else None,
        'current_close': float(current_close) if pd.notna(current_close) else None,
        'current_deviation': current_deviation,
        'bbmacd_event': bbmacd_event,
        'bb_width_zscore': float(bbmacd_df['bb_width_zscore'].iloc[-1]) if 'bb_width_zscore' in bbmacd_df.columns else None,
        'pavp_vah': float(pavp_vah) if pd.notna(pavp_vah) else None,
        'pavp_val': float(pavp_val) if pd.notna(pavp_val) else None,
        'pavp_poc': float(pavp_poc) if pd.notna(pavp_poc) else None,
        'va_pos_01': float(va_pos_01) if pd.notna(va_pos_01) else None,
        'vol_zscore': volume_zscore(daily_df['volume'], win=20),
        'change_pct': compute_change_pct(daily_df),
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


def ensure_table_exists(engine):
    """确保vwap_selection表存在"""
    create_sql = """
    CREATE TABLE IF NOT EXISTS vwap_selection (
        id BIGSERIAL PRIMARY KEY,
        selection_date DATE NOT NULL,
        signal_date DATE,
        ts_code VARCHAR(20) NOT NULL,
        stock_name VARCHAR(50),
        dsa_dir INT,
        dir1_bars INT,
        cumulative_deviation FLOAT,
        avg_deviation FLOAT,
        deviation_variance FLOAT,
        positive_ratio FLOAT,
        negative_ratio FLOAT,
        current_vwap FLOAT,
        current_close FLOAT,
        current_deviation FLOAT,
        bbmacd_event VARCHAR(20),
        bb_width_zscore FLOAT,
        pavp_vah FLOAT,
        pavp_val FLOAT,
        pavp_poc FLOAT,
        va_pos_01 FLOAT,
        vol_zscore FLOAT,
        change_pct FLOAT,
        batch_no INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(selection_date, ts_code)
    );
    CREATE INDEX IF NOT EXISTS idx_vwap_selection_date ON vwap_selection(selection_date);
    CREATE INDEX IF NOT EXISTS idx_vwap_ts_code ON vwap_selection(ts_code);
    CREATE INDEX IF NOT EXISTS idx_vwap_dsa_dir ON vwap_selection(dsa_dir);
    CREATE INDEX IF NOT EXISTS idx_vwap_dir1_bars ON vwap_selection(dir1_bars);
    CREATE INDEX IF NOT EXISTS idx_vwap_avg_deviation ON vwap_selection(avg_deviation);
    CREATE INDEX IF NOT EXISTS idx_vwap_batch_no ON vwap_selection(batch_no);
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(create_sql))
            conn.commit()
        except Exception:
            conn.rollback()

    # 添加新列（如果表已存在）
    alter_sql = """
    ALTER TABLE vwap_selection ADD COLUMN IF NOT EXISTS deviation_variance FLOAT;
    ALTER TABLE vwap_selection ADD COLUMN IF NOT EXISTS positive_ratio FLOAT;
    ALTER TABLE vwap_selection ADD COLUMN IF NOT EXISTS negative_ratio FLOAT;
    """
    with engine.connect() as conn:
        try:
            conn.execute(text(alter_sql))
            conn.commit()
        except Exception:
            conn.rollback()


def save_to_database(df: pd.DataFrame, selection_date: date) -> int:
    """保存选股结果到数据库（幂等性：先删后插）"""
    if df.empty:
        print("数据为空，跳过数据库保存")
        return 0

    ensure_table_exists(engine)

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
            'stock_name': row.get('stock_name', ''),
            'dsa_dir': int(row['dsa_dir']) if pd.notna(row['dsa_dir']) else None,
            'dir1_bars': int(row['dir1_bars']) if pd.notna(row['dir1_bars']) else None,
            'cumulative_deviation': float(row['cumulative_deviation']) if pd.notna(row.get('cumulative_deviation')) else None,
            'avg_deviation': float(row['avg_deviation']) if pd.notna(row.get('avg_deviation')) else None,
            'deviation_variance': float(row['deviation_variance']) if pd.notna(row.get('deviation_variance')) else None,
            'positive_ratio': float(row['positive_ratio']) if pd.notna(row.get('positive_ratio')) else None,
            'negative_ratio': float(row['negative_ratio']) if pd.notna(row.get('negative_ratio')) else None,
            'current_vwap': float(row['current_vwap']) if pd.notna(row.get('current_vwap')) else None,
            'current_close': float(row['current_close']) if pd.notna(row.get('current_close')) else None,
            'current_deviation': float(row['current_deviation']) if pd.notna(row.get('current_deviation')) else None,
            'bbmacd_event': row.get('bbmacd_event', '无') or '无',
            'bb_width_zscore': float(row['bb_width_zscore']) if pd.notna(row.get('bb_width_zscore')) else None,
            'pavp_vah': float(row['pavp_vah']) if pd.notna(row.get('pavp_vah')) else None,
            'pavp_val': float(row['pavp_val']) if pd.notna(row.get('pavp_val')) else None,
            'pavp_poc': float(row['pavp_poc']) if pd.notna(row.get('pavp_poc')) else None,
            'va_pos_01': float(row['va_pos_01']) if pd.notna(row.get('va_pos_01')) else None,
            'vol_zscore': float(row['vol_zscore']) if pd.notna(row.get('vol_zscore')) else None,
            'change_pct': float(row['change_pct']) if pd.notna(row.get('change_pct')) else None,
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


def select_dsa_vwap_stocks(selection_date: Optional[date] = None, save_to_db: bool = True) -> pd.DataFrame:
    """
    根据DSA VWAP指标选出满足条件的股票

    Args:
        selection_date: 选股日期，默认为当天
        save_to_db: 是否保存到数据库
    """
    if selection_date is None:
        selection_date = date.today()

    print("=" * 80)
    print("选股条件（DSA VWAP策略）：")
    print(f"  选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print(f"  核心条件: DSA dir=1（多头方向）")
    print(f"  统计指标: dir=1持续bar数、累计偏离率、平均偏离率")
    print(f"  观察项: BBMACD、PAVP、成交量Z-Score")
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
    print("开始DSA VWAP指标筛选...")
    print(f"  原股票数: {len(stock_list)}")

    filtered_results = []

    # 使用tqdm进度条
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="DSA VWAP选股", unit="只"):
        ts_code = row['ts_code']

        result = process_stock(ts_code, selection_date)
        if result:
            filtered_results.append(result)

    result_df = pd.DataFrame(filtered_results)

    if not result_df.empty:
        # 获取股票名称
        stock_names = batch_get_stock_names(result_df['ts_code'].tolist())
        result_df['stock_name'] = result_df['ts_code'].map(stock_names)

        # 分配批次号
        result_df['batch_no'] = (result_df.index // 10) + 1

    print("\n" + "=" * 80)
    print("选股结果汇总：")
    print("=" * 80)
    print(f"DSA VWAP筛选后: {len(result_df)} 只")

    if not result_df.empty:
        print(f"\n统计指标汇总：")
        print(f"  dir=1平均持续bar数: {result_df['dir1_bars'].mean():.2f}")
        print(f"  平均累计偏离率: {result_df['cumulative_deviation'].mean():.4f}%")
        print(f"  平均偏离率: {result_df['avg_deviation'].mean():.4f}%")

        batch_count = result_df['batch_no'].max()
        print(f"\n批次信息：共 {batch_count} 批，每批10只股票")

        print("\n" + "=" * 80)
        print("前20名股票（按dir1_bars降序）：")
        print("=" * 80)
        display_df = result_df.sort_values('dir1_bars', ascending=False)
        display_cols = ['ts_code', 'stock_name', 'dir1_bars', 'avg_deviation', 'current_deviation', 'bbmacd_event']
        print(display_df[display_cols].head(20).to_string(index=False))
    
    # 保存到数据库
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
        print("\n该股票不满足选股条件（DSA dir != 1 或数据不足）")
    
    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='DSA VWAP指标选股工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python selection/selection_vwap.py                    # 使用当天日期选股
  python selection/selection_vwap.py 2025-12-31         # 指定日期选股
  python selection/selection_vwap.py 20251231           # 指定日期选股（无分隔符）
  python selection/selection_vwap.py --test 600547      # 测试单只股票
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
    
    args = parser.parse_args()
    
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
    print("DSA VWAP指标选股工具")
    print(f"选股日期: {selection_date.strftime('%Y-%m-%d')}")
    print("=" * 80)

    # 测试模式
    if args.test:
        test_single_stock(args.test, selection_date)
        sys.exit(0)

    # 正常选股模式
    df = select_dsa_vwap_stocks(selection_date=selection_date, save_to_db=not args.no_save)

    print("\n" + "=" * 80)
    print("选股完成")
    print(f"选股日期: {selection_date}")
    print(f"选中股票数: {len(df)}")
    print(f"查询SQL: SELECT * FROM {SELECTION_TABLE} WHERE selection_date = '{selection_date}'")
    print("=" * 80)

    return df


if __name__ == '__main__':
    # 自测入口
    main()
