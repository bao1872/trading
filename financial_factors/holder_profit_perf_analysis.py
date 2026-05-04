#!/usr/bin/env python3
"""
股东持股收益计算 - 详细性能分析报告

Purpose: 分析 calculate_holder_profit 函数的每一步性能瓶颈
Usage: python holder_profit_perf_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from typing import Dict, Optional
import pandas as pd
from datetime import datetime
import time

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)


class PerformanceTimer:
    """性能计时器"""
    def __init__(self):
        self.timings = {}
        self.current = None
        self.start_time = None
    
    def start(self, name: str):
        """开始计时"""
        self.current = name
        self.start_time = time.time()
    
    def stop(self):
        """停止计时"""
        if self.current and self.start_time:
            elapsed = time.time() - self.start_time
            if self.current not in self.timings:
                self.timings[self.current] = []
            self.timings[self.current].append(elapsed)
            self.current = None
            self.start_time = None
            return elapsed
        return 0
    
    def report(self):
        """生成报告"""
        print("\n" + "=" * 80)
        print("详细性能分析报告")
        print("=" * 80)
        
        total_time = sum(sum(v) for v in self.timings.values())
        
        for name, times in sorted(self.timings.items()):
            total = sum(times)
            avg = total / len(times)
            pct = (total / total_time * 100) if total_time > 0 else 0
            print(f"\n{name}:")
            print(f"  调用次数: {len(times)}")
            print(f"  总用时: {total:.3f}s")
            print(f"  平均用时: {avg:.3f}s")
            print(f"  占比: {pct:.1f}%")
        
        print(f"\n{'=' * 80}")
        print(f"总用时: {total_time:.3f}s")
        print("=" * 80)
        
        return self.timings


timer = PerformanceTimer()


def get_stock_price_timed(ts_code: str, date_str: str, thread_engine=None) -> Optional[Dict]:
    """带计时的股价查询"""
    timer.start("股价查询")
    try:
        eng = thread_engine if thread_engine else engine
        with eng.connect() as conn:
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
        print(f"查询股价失败 {ts_code} {date_str}: {e}")
    finally:
        timer.stop()
    return None


def calculate_holder_profit_timed(holder_name: str, ts_code: Optional[str] = None) -> Dict:
    """
    带详细性能计时的股东收益计算
    """
    eng = engine
    
    # 步骤1: 查询股东持股记录
    timer.start("1.查询股东持股记录")
    with eng.connect() as conn:
        if ts_code:
            sql = """
                SELECT
                    h.ts_code,
                    h.stock_name,
                    h.report_date,
                    h.ann_date,
                    h.hold_ratio,
                    h.hold_change,
                    h.holder_rank,
                    h.hold_amount,
                    LEAD(h.report_date) OVER (PARTITION BY h.ts_code ORDER BY h.report_date) as next_report_date
                FROM stock_top10_holders_tushare h
                WHERE h.holder_name = :holder_name
                AND h.ts_code = :ts_code
                ORDER BY h.ts_code, h.report_date ASC
            """
            params = {'holder_name': holder_name, 'ts_code': ts_code}
        else:
            sql = """
                SELECT
                    h.ts_code,
                    h.stock_name,
                    h.report_date,
                    h.ann_date,
                    h.hold_ratio,
                    h.hold_change,
                    h.holder_rank,
                    h.hold_amount,
                    LEAD(h.report_date) OVER (PARTITION BY h.ts_code ORDER BY h.report_date) as next_report_date
                FROM stock_top10_holders_tushare h
                WHERE h.holder_name = :holder_name
                ORDER BY h.ts_code, h.report_date ASC
            """
            params = {'holder_name': holder_name}

        result = conn.execute(text(sql), params)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    timer.stop()
    
    if df.empty:
        return {'records': [], 'stock_profits': {}, 'summary': {}}
    
    record_count = len(df)
    stock_count = df['ts_code'].nunique()
    print(f"  股东 {holder_name}: {record_count} 条持股记录, {stock_count} 只股票")
    
    # 步骤2: 获取最新报告期
    timer.start("2.获取最新报告期")
    latest_dates_sql = """
        SELECT ts_code, MAX(report_date) as latest_date
        FROM stock_top10_holders_tushare
        GROUP BY ts_code
    """
    latest_dates_df = pd.read_sql(latest_dates_sql, eng)
    latest_dates_map = dict(zip(latest_dates_df['ts_code'], latest_dates_df['latest_date']))
    timer.stop()
    
    # 步骤3: 计算操作类型（CPU计算）
    timer.start("3.计算操作类型(CPU)")
    
    def calc_operation_and_exit(row, stock_records, latest_dates_map):
        ts_code = row['ts_code']
        current_date = row['report_date']
        is_first_entry = current_date == stock_records['report_date'].min()
        
        is_re_entry = False
        if not is_first_entry:
            prev_records = stock_records[stock_records['report_date'] < current_date]
            if not prev_records.empty:
                prev_date = prev_records['report_date'].max()
                prev_dt = datetime.strptime(str(prev_date), '%Y%m%d')
                curr_dt = datetime.strptime(str(current_date), '%Y%m%d')
                months_diff = (curr_dt.year - prev_dt.year) * 12 + (curr_dt.month - prev_dt.month)
                if months_diff > 6:
                    is_re_entry = True
        
        is_exit = False
        if pd.isna(row['next_report_date']):
            stock_latest = latest_dates_map.get(ts_code)
            if stock_latest and row['report_date'] != stock_latest:
                is_exit = True
        
        if is_first_entry or is_re_entry:
            operation = '入场'
        elif pd.isna(row['hold_change']) or row['hold_change'] == 0:
            operation = '持仓'
        elif row['hold_change'] > 0:
            operation = '加仓'
        else:
            operation = '减仓'
        
        return operation, is_exit
    
    def generate_exit_record(last_record, exit_date):
        return {
            'ts_code': last_record['ts_code'],
            'stock_name': last_record['stock_name'],
            'report_date': exit_date,
            'ann_date': None,
            'hold_ratio': 0,
            'hold_change': None,
            'holder_rank': None,
            'hold_amount': last_record.get('hold_amount'),
            'operation': '出场'
        }
    
    def calc_exit_date(entry_date_str):
        dt = datetime.strptime(str(entry_date_str), '%Y%m%d')
        return f"{dt.year}1231"
    
    records = []
    exit_records_to_add = []
    
    for ts_code_group, group_df in df.groupby('ts_code'):
        stock_records_list = []
        last_record = None
        need_exit = False
        prev_period_last_record = None
        
        for idx, row in group_df.iterrows():
            operation, is_exit = calc_operation_and_exit(row, group_df, latest_dates_map)
            is_re_entry = (operation == '入场' and not (row['report_date'] == group_df['report_date'].min()))
            
            if is_re_entry and prev_period_last_record:
                exit_date = calc_exit_date(prev_period_last_record['report_date'])
                exit_record = generate_exit_record(prev_period_last_record, exit_date)
                exit_records_to_add.append(exit_record)
            
            record = {
                'ts_code': row['ts_code'],
                'stock_name': row['stock_name'],
                'report_date': row['report_date'],
                'ann_date': row['ann_date'],
                'hold_ratio': row['hold_ratio'],
                'hold_change': row['hold_change'],
                'holder_rank': row['holder_rank'],
                'hold_amount': row['hold_amount'],
                'operation': operation
            }
            stock_records_list.append(record)
            last_record = record
            
            if not is_re_entry:
                prev_period_last_record = record
            if is_exit:
                need_exit = True
        
        if need_exit and last_record:
            stock_latest = latest_dates_map.get(ts_code_group)
            if stock_latest:
                exit_record = generate_exit_record(last_record, stock_latest)
                exit_records_to_add.append(exit_record)
        
        records.extend(stock_records_list)
    
    records.extend(exit_records_to_add)
    timer.stop()
    
    # 步骤4: 排序（CPU计算）
    timer.start("4.DataFrame排序(CPU)")
    records_df = pd.DataFrame(records)
    records_df = records_df.sort_values(['ts_code', 'report_date'])
    records = records_df.to_dict('records')
    timer.stop()
    
    # 步骤5: 查询股价（数据库I/O）
    timer.start("5.查询所有股价(DB)")
    stock_profits = {}
    price_query_count = 0
    
    for ts_code_group, group_df in df.groupby('ts_code'):
        stock_records = [r for r in records if r['ts_code'] == ts_code_group]
        stock_name = stock_records[0]['stock_name'] if stock_records else ''
        
        # 查询每条记录的股价
        for i, record in enumerate(stock_records):
            price_query_count += 1
            curr_price_info = get_stock_price_timed(ts_code_group, record['report_date'])
            
            if curr_price_info:
                record['current_price'] = curr_price_info['price']
                record['price_date'] = curr_price_info['date']
            else:
                record['current_price'] = None
                record['price_date'] = None
    
    timer.stop()
    print(f"  共查询 {price_query_count} 次股价")
    
    # 步骤6: 计算收益（CPU计算）
    timer.start("6.计算收益(CPU)")
    
    for ts_code_group, group_df in df.groupby('ts_code'):
        stock_records = [r for r in records if r['ts_code'] == ts_code_group]
        stock_name = stock_records[0]['stock_name'] if stock_records else ''
        
        current_entry_date = None
        current_entry_price = None
        current_entry_shares = None
        
        for i, record in enumerate(stock_records):
            hold_amount = record.get('hold_amount')
            if hold_amount and not pd.isna(hold_amount):
                record['share_count'] = hold_amount
            else:
                record['share_count'] = None
            
            if record['operation'] == '入场':
                record['profit'] = None
                record['profit_pct'] = None
                record['profit_amount'] = None
                record['entry_price'] = record.get('current_price')
                record['entry_date'] = record['report_date']
                record['entry_amount'] = record.get('share_count', 0) * record.get('current_price', 0) if record.get('share_count') and record.get('current_price') else None
                
                current_entry_date = record['report_date']
                current_entry_price = record.get('current_price')
                current_entry_shares = record.get('share_count', 0)
            
            elif record['operation'] == '出场':
                if i > 0:
                    entry_date = current_entry_date
                    entry_price = current_entry_price
                    entry_shares = current_entry_shares
                    exit_price = record.get('current_price')
                    exit_date = record['report_date']
                    
                    record['entry_date'] = entry_date
                    record['entry_price'] = entry_price
                    record['entry_shares'] = entry_shares
                    record['entry_amount'] = entry_shares * entry_price if entry_shares and entry_price else None
                    record['exit_date'] = exit_date
                    record['exit_price'] = exit_price
                    record['exit_shares'] = entry_shares
                    record['exit_amount'] = entry_shares * exit_price if entry_shares and exit_price else None
                    
                    if exit_price and entry_price and entry_shares:
                        profit_pct = (exit_price - entry_price) / entry_price * 100
                        record['profit'] = exit_price - entry_price
                        record['profit_pct'] = profit_pct
                        record['profit_amount'] = (exit_price - entry_price) * entry_shares
                
                current_entry_date = None
                current_entry_price = None
                current_entry_shares = None
            
            elif record['operation'] in ['加仓', '减仓', '持仓']:
                if i > 0 and stock_records[i-1].get('current_price'):
                    last_price = stock_records[i-1]['current_price']
                    curr_price = record['current_price']
                    curr_shares = record.get('share_count', 0)
                    
                    record['entry_date'] = current_entry_date
                    record['entry_price'] = current_entry_price
                    record['entry_shares'] = current_entry_shares
                    
                    if curr_price and last_price:
                        profit_pct = (curr_price - last_price) / last_price * 100
                        record['profit'] = curr_price - last_price
                        record['profit_pct'] = profit_pct
                        if curr_shares:
                            record['profit_amount'] = (curr_price - last_price) * curr_shares
        
        # 计算最新股价
        if stock_records:
            last_record = stock_records[-1]
            has_exit = any(r.get('operation') == '出场' for r in stock_records)
            
            if not has_exit and last_record.get('operation') in ['入场', '加仓', '减仓', '持仓']:
                today = datetime.now().strftime('%Y%m%d')
                latest_price_info = get_stock_price_timed(ts_code_group, today)
                
                if latest_price_info and current_entry_price:
                    latest_price = latest_price_info['price']
                    entry_price = current_entry_price
                    curr_shares = last_record.get('share_count', 0)
                    
                    last_record['entry_date'] = current_entry_date
                    last_record['entry_price'] = entry_price
                    last_record['entry_shares'] = current_entry_shares
                    last_record['latest_price'] = latest_price
                    last_record['price_date'] = latest_price_info['date']
                    
                    profit_pct = (latest_price - entry_price) / entry_price * 100
                    last_record['profit'] = latest_price - entry_price
                    last_record['profit_pct'] = profit_pct
                    
                    if curr_shares:
                        last_record['profit_amount'] = (latest_price - entry_price) * curr_shares
        
        profit_records = [r for r in stock_records if r.get('profit_pct') is not None]
        if profit_records:
            total_profit = sum(r['profit_pct'] for r in profit_records)
            win_count = sum(1 for r in profit_records if r['profit_pct'] > 0)
            win_rate = win_count / len(profit_records) * 100
        else:
            total_profit = 0
            win_count = 0
            win_rate = 0
        
        stock_profits[ts_code_group] = {
            'stock_name': stock_name,
            'records': stock_records,
            'total_profit': total_profit,
            'win_rate': win_rate,
            'operation_count': len(profit_records),
            'win_count': win_count
        }
    timer.stop()
    
    # 步骤7: 汇总统计（CPU计算）
    timer.start("7.汇总统计(CPU)")
    all_profit_records = []
    for stock_data in stock_profits.values():
        all_profit_records.extend([r for r in stock_data['records'] if r.get('profit_pct') is not None])
    
    if all_profit_records:
        overall_total_profit = sum(r['profit_pct'] for r in all_profit_records)
        overall_win_count = sum(1 for r in all_profit_records if r['profit_pct'] > 0)
        overall_win_rate = overall_win_count / len(all_profit_records) * 100
    else:
        overall_total_profit = 0
        overall_win_count = 0
        overall_win_rate = 0
    
    summary = {
        'total_stocks': len(stock_profits),
        'total_operations': len(all_profit_records),
        'total_profit': overall_total_profit,
        'win_rate': overall_win_rate,
        'win_count': overall_win_count
    }
    timer.stop()
    
    return {
        'records': records,
        'stock_profits': stock_profits,
        'summary': summary
    }


def save_holder_profit_to_db_timed(holder_name: str, result: Dict) -> Dict:
    """带计时的数据库保存"""
    timer.start("8.保存到数据库")
    
    records_inserted = 0
    stats_inserted = 0
    
    with engine.connect() as conn:
        # 插入交易记录
        for record in result['records']:
            conn.execute(text("""
                INSERT INTO holder_trade_records (
                    holder_name, ts_code, stock_name, report_date, operation,
                    hold_ratio, hold_amount, current_price,
                    entry_date, entry_price, entry_shares, entry_amount,
                    exit_date, exit_price, exit_shares, exit_amount,
                    profit_pct, profit_amount
                ) VALUES (
                    :holder_name, :ts_code, :stock_name, :report_date, :operation,
                    :hold_ratio, :hold_amount, :current_price,
                    :entry_date, :entry_price, :entry_shares, :entry_amount,
                    :exit_date, :exit_price, :exit_shares, :exit_amount,
                    :profit_pct, :profit_amount
                )
            """), {
                'holder_name': holder_name,
                'ts_code': record.get('ts_code'),
                'stock_name': record.get('stock_name'),
                'report_date': str(record.get('report_date')) if record.get('report_date') else None,
                'operation': record.get('operation'),
                'hold_ratio': record.get('hold_ratio'),
                'hold_amount': int(record.get('hold_amount')) if record.get('hold_amount') is not None else None,
                'current_price': record.get('current_price'),
                'entry_date': str(record.get('entry_date')) if record.get('entry_date') else None,
                'entry_price': record.get('entry_price'),
                'entry_shares': int(record.get('entry_shares')) if record.get('entry_shares') is not None else None,
                'entry_amount': record.get('entry_amount'),
                'exit_date': str(record.get('exit_date')) if record.get('exit_date') else None,
                'exit_price': record.get('exit_price'),
                'exit_shares': int(record.get('exit_shares')) if record.get('exit_shares') is not None else None,
                'exit_amount': record.get('exit_amount'),
                'profit_pct': record.get('profit_pct'),
                'profit_amount': record.get('profit_amount')
            })
            records_inserted += 1
        
        # 插入/更新统计记录
        for ts_code, stock_data in result['stock_profits'].items():
            records = stock_data['records']
            profit_records = [r for r in records if r.get('profit_pct') is not None]
            
            total_entry_amount = sum(r.get('entry_amount', 0) for r in records if r.get('entry_amount'))
            total_exit_amount = sum(r.get('exit_amount', 0) for r in records if r.get('exit_amount'))
            total_profit_amount = sum(r.get('profit_amount', 0) for r in profit_records if r.get('profit_amount'))
            
            first_entry = next((r for r in records if r.get('operation') == '入场'), None)
            last_exit = next((r for r in reversed(records) if r.get('operation') == '出场'), None)
            
            conn.execute(text("""
                INSERT INTO holder_trade_stats (
                    holder_name, ts_code, stock_name, total_profit, win_rate,
                    operation_count, win_count, loss_count,
                    total_entry_amount, total_exit_amount, total_profit_amount,
                    first_entry_date, last_exit_date
                ) VALUES (
                    :holder_name, :ts_code, :stock_name, :total_profit, :win_rate,
                    :operation_count, :win_count, :loss_count,
                    :total_entry_amount, :total_exit_amount, :total_profit_amount,
                    :first_entry_date, :last_exit_date
                )
                ON CONFLICT (holder_name, ts_code) DO UPDATE SET
                    total_profit = EXCLUDED.total_profit,
                    win_rate = EXCLUDED.win_rate,
                    operation_count = EXCLUDED.operation_count,
                    win_count = EXCLUDED.win_count,
                    loss_count = EXCLUDED.loss_count,
                    total_entry_amount = EXCLUDED.total_entry_amount,
                    total_exit_amount = EXCLUDED.total_exit_amount,
                    total_profit_amount = EXCLUDED.total_profit_amount,
                    first_entry_date = EXCLUDED.first_entry_date,
                    last_exit_date = EXCLUDED.last_exit_date,
                    updated_at = NOW()
            """), {
                'holder_name': holder_name,
                'ts_code': ts_code,
                'stock_name': stock_data.get('stock_name'),
                'total_profit': stock_data.get('total_profit'),
                'win_rate': stock_data.get('win_rate'),
                'operation_count': stock_data.get('operation_count'),
                'win_count': stock_data.get('win_count'),
                'loss_count': stock_data.get('operation_count', 0) - stock_data.get('win_count', 0),
                'total_entry_amount': total_entry_amount if total_entry_amount > 0 else None,
                'total_exit_amount': total_exit_amount if total_exit_amount > 0 else None,
                'total_profit_amount': total_profit_amount if total_profit_amount != 0 else None,
                'first_entry_date': str(first_entry.get('entry_date')) if first_entry else None,
                'last_exit_date': str(last_exit.get('exit_date')) if last_exit else None
            })
            stats_inserted += 1
        
        conn.commit()
    
    timer.stop()
    
    return {
        'records_inserted': records_inserted,
        'stats_inserted': stats_inserted
    }


def run_performance_analysis():
    """运行完整性能分析"""
    print("=" * 80)
    print("股东持股收益计算 - 详细性能分析")
    print("=" * 80)
    
    # 选择几个不同类型的股东进行测试
    test_holders = [
        '626投资控股有限公司',  # 少量记录
        'ACADIAN资产管理有限责任公司-ACADIAN中国A股股票主基金(交易所)',  # 中等记录
        'ACEUNIONHOLDINGLIMITED',  # 较多记录
    ]
    
    print(f"\n测试股东: {test_holders}")
    print("=" * 80)
    
    for holder_name in test_holders:
        print(f"\n{'=' * 80}")
        print(f"分析股东: {holder_name}")
        print("=" * 80)
        
        # 计算收益（带计时）
        result = calculate_holder_profit_timed(holder_name)
        
        if result['records']:
            # 保存到数据库（带计时）
            stats = save_holder_profit_to_db_timed(holder_name, result)
            print(f"\n保存结果: {stats['records_inserted']} 条交易记录, {stats['stats_inserted']} 条统计记录")
    
    # 生成最终报告
    timer.report()
    
    # 瓶颈分析
    print("\n" + "=" * 80)
    print("性能瓶颈分析")
    print("=" * 80)
    
    timings = timer.timings
    total_db_time = sum(timings.get('1.查询股东持股记录', [])) + \
                    sum(timings.get('2.获取最新报告期', [])) + \
                    sum(timings.get('5.查询所有股价(DB)', [])) + \
                    sum(timings.get('8.保存到数据库', []))
    
    total_cpu_time = sum(timings.get('3.计算操作类型(CPU)', [])) + \
                     sum(timings.get('4.DataFrame排序(CPU)', [])) + \
                     sum(timings.get('6.计算收益(CPU)', [])) + \
                     sum(timings.get('7.汇总统计(CPU)', []))
    
    total_price_time = sum(timings.get('股价查询', []))
    
    total_time = total_db_time + total_cpu_time
    
    print(f"\n数据库I/O总用时: {total_db_time:.3f}s ({total_db_time/total_time*100:.1f}%)")
    print(f"  - 查询股东持股记录: {sum(timings.get('1.查询股东持股记录', [])):.3f}s")
    print(f"  - 获取最新报告期: {sum(timings.get('2.获取最新报告期', [])):.3f}s")
    print(f"  - 查询所有股价: {sum(timings.get('5.查询所有股价(DB)', [])):.3f}s")
    print(f"    └─ 其中单次股价查询: {total_price_time:.3f}s")
    print(f"  - 保存到数据库: {sum(timings.get('8.保存到数据库', [])):.3f}s")
    
    print(f"\nCPU计算总用时: {total_cpu_time:.3f}s ({total_cpu_time/total_time*100:.1f}%)")
    print(f"  - 计算操作类型: {sum(timings.get('3.计算操作类型(CPU)', [])):.3f}s")
    print(f"  - DataFrame排序: {sum(timings.get('4.DataFrame排序(CPU)', [])):.3f}s")
    print(f"  - 计算收益: {sum(timings.get('6.计算收益(CPU)', [])):.3f}s")
    print(f"  - 汇总统计: {sum(timings.get('7.汇总统计(CPU)', [])):.3f}s")
    
    print("\n" + "=" * 80)
    print("结论:")
    if total_db_time > total_cpu_time:
        print("主要瓶颈: 数据库I/O操作")
        if total_price_time > total_db_time * 0.5:
            print("  └─ 股价查询是最大瓶颈，考虑批量查询或缓存优化")
    else:
        print("主要瓶颈: CPU计算")
    print("=" * 80)


if __name__ == "__main__":
    run_performance_analysis()
