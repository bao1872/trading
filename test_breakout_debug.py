# -*- coding: utf-8 -*-
"""
调试 scan_breakout_events.py 4月2号无事件问题
"""

import sys
import os
sys.path.insert(0, '.')

import importlib.util
spec = importlib.util.spec_from_file_location('scan_breakout_events', 'backtrader/scan_breakout_events.py')
scan_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan_module)

load_kline_data_from_db = scan_module.load_kline_data_from_db
build_engine_args = scan_module.build_engine_args
extract_dir_turn_events = scan_module.extract_dir_turn_events
extract_pullback_buy_events = scan_module.extract_pullback_buy_events
from features.merged_atr_rope_breakout_volume_delta import MergedEngine

ts_code = '600519.SH'
name = '贵州茅台'
freq = 'd'
bars = 255
target_date = '2026-04-02'

print(f"=== 测试股票: {ts_code} ===")
print(f"目标日期: {target_date}")

# 1. 加载数据
df = load_kline_data_from_db(ts_code, freq, bars, target_date)
if df is None:
    print("错误: 无数据")
    sys.exit(1)

print(f"\n数据条数: {len(df)}")
print(f"日期范围: {df.index[0]} ~ {df.index[-1]}")
print(f"最后5条数据:")
print(df.tail())

# 2. 检查最后日期是否匹配目标日期
last_date = df.index[-1].strftime("%Y-%m-%d")
print(f"\n最后一条数据日期: {last_date}")
print(f"目标日期: {target_date}")
print(f"日期匹配: {last_date == target_date}")

# 3. 运行因子引擎
args = build_engine_args(ts_code, freq, bars)
engine = MergedEngine(df, args)
engine.run()
result_df = engine.df

print(f"\n=== 因子引擎结果 ===")
print(f"结果数据条数: {len(result_df)}")

# 4. 检查关键列
key_cols = ['close', 'rope', 'dir_turn_long_flag', 'pullback_buy_flag', 'breakout_quality_score']
available_cols = [c for c in key_cols if c in result_df.columns]
print(f"\n关键列数据 (最后10行):")
print(result_df[available_cols].tail(10))

# 5. 检查是否有事件标志
if 'dir_turn_long_flag' in result_df.columns:
    dir_turn_count = result_df['dir_turn_long_flag'].sum()
    print(f"\ndir_turn_long_flag=1 的行数: {dir_turn_count}")
    
    # 查看所有dir_turn_long_flag=1的行
    dir_turn_rows = result_df[result_df['dir_turn_long_flag'] == 1]
    if not dir_turn_rows.empty:
        print(f"\n所有翻多事件:")
        print(dir_turn_rows[['close', 'rope', 'dir_turn_long_flag', 'breakout_quality_score']])

if 'pullback_buy_flag' in result_df.columns:
    pullback_count = result_df['pullback_buy_flag'].sum()
    print(f"\npullback_buy_flag=1 的行数: {pullback_count}")

# 6. 测试事件提取函数
print(f"\n=== 测试事件提取 ===")
dir_events = extract_dir_turn_events(result_df, ts_code, name, freq, target_date)
pullback_events = extract_pullback_buy_events(result_df, ts_code, name, freq, target_date)

print(f"提取到的翻多事件数: {len(dir_events)}")
print(f"提取到的回踩买点事件数: {len(pullback_events)}")

# 7. 检查4月2号当天的数据
print(f"\n=== 4月2号当天数据详情 ===")
apr_02_data = result_df[result_df.index.strftime('%Y-%m-%d') == '2026-04-02']
if not apr_02_data.empty:
    print(apr_02_data[available_cols])
else:
    print("未找到4月2号的数据")
