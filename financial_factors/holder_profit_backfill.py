#!/usr/bin/env python3
"""
股东持股收益回补脚本（简化版，使用连接池）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

# 使用连接池限制连接数
engine = create_engine(
    DATABASE_URL,
    pool_size=4,
    max_overflow=2,
    pool_timeout=30,
    pool_recycle=3600
)

print_lock = Lock()


def get_stock_price(ts_code, date_str):
    """获取股价"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT close, bar_time::date as trade_date
                FROM stock_k_data
                WHERE ts_code = :ts_code
                AND bar_time < (:date || ' 23:59:59')::timestamp
                ORDER BY bar_time DESC
                LIMIT 1
            """), {'ts_code': ts_code, 'date': date_str}).fetchone()
            if result:
                return {'price': result[0], 'date': str(result[1])}
    except Exception as e:
        print(f"股价查询失败: {e}")
    return None


def calculate_and_save_holder_profit(holder_name, latest_dates_map):
    """计算并保存单个股东的收益"""
    try:
        with engine.connect() as conn:
            # 查询持股记录
            result = conn.execute(text("""
                SELECT ts_code, stock_name, report_date, hold_ratio, hold_change, holder_rank, hold_amount,
                       LEAD(report_date) OVER (PARTITION BY ts_code ORDER BY report_date) as next_report_date
                FROM stock_top10_holders_tushare
                WHERE holder_name = :holder_name
                ORDER BY ts_code, report_date ASC
            """), {'holder_name': holder_name})
            
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if df.empty:
                return {'status': 'empty', 'records': 0, 'stats': 0}
            
            # 简化的收益计算逻辑
            records = []
            for ts_code, group in df.groupby('ts_code'):
                for idx, row in group.iterrows():
                    price_info = get_stock_price(ts_code, row['report_date'])
                    if price_info:
                        records.append({
                            'holder_name': holder_name,
                            'ts_code': ts_code,
                            'stock_name': row['stock_name'],
                            'report_date': row['report_date'],
                            'hold_ratio': row['hold_ratio'],
                            'hold_amount': row['hold_amount'],
                            'current_price': price_info['price']
                        })
            
            # 批量插入
            if records:
                for record in records:
                    conn.execute(text("""
                        INSERT INTO holder_trade_records 
                        (holder_name, ts_code, stock_name, report_date, hold_ratio, hold_amount, current_price)
                        VALUES (:holder_name, :ts_code, :stock_name, :report_date, :hold_ratio, :hold_amount, :current_price)
                        ON CONFLICT DO NOTHING
                    """), record)
                conn.commit()
            
            return {'status': 'success', 'records': len(records), 'stats': df['ts_code'].nunique()}
            
    except Exception as e:
        return {'status': 'failed', 'error': str(e), 'records': 0, 'stats': 0}


def main():
    """主函数"""
    print("=" * 80)
    print("股东持股收益回补（连接池版）")
    print("=" * 80)
    
    # 获取股东列表
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT holder_name FROM stock_top10_holders_tushare
            WHERE holder_name IS NOT NULL AND holder_name != ''
            ORDER BY holder_name
        """))
        holders = [row[0] for row in result]
    
    print(f"共 {len(holders)} 个股东需要处理")
    print("=" * 80)
    
    # 预缓存最新报告期
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT ts_code, MAX(report_date) FROM stock_top10_holders_tushare GROUP BY ts_code
        """))
        latest_dates_map = dict(result.fetchall())
    
    print(f"已缓存 {len(latest_dates_map)} 只股票最新报告期")
    
    # 多线程处理
    processed = 0
    success = 0
    failed = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(calculate_and_save_holder_profit, h, latest_dates_map): h for h in holders}
        
        for future in as_completed(futures):
            holder = futures[future]
            result = future.result()
            processed += 1
            
            with print_lock:
                if result['status'] == 'success':
                    success += 1
                    print(f"[{processed}/{len(holders)}] ✓ {holder}: {result['records']}条记录")
                elif result['status'] == 'empty':
                    print(f"[{processed}/{len(holders)}] ○ {holder}: 无数据")
                else:
                    print(f"[{processed}/{len(holders)}] ✗ {holder}: {result.get('error', '未知错误')}")
                
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"\n>>> 进度: {processed}/{len(holders)} ({processed/len(holders)*100:.1f}%), 用时: {elapsed/60:.1f}分钟\n")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"完成! 共处理 {processed} 个股东, 成功 {success} 个")
    print(f"总用时: {elapsed/60:.1f} 分钟")
    print("=" * 80)


if __name__ == "__main__":
    main()
