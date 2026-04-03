# -*- coding: utf-8 -*-
"""
进一步调试 - 查看4月2号前后的详细因子数据
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

from features.merged_atr_rope_breakout_volume_delta import MergedEngine

ts_code = '600519.SH'
freq = 'd'
bars = 255
target_date = '2026-04-02'

df = load_kline_data_from_db(ts_code, freq, bars, target_date)
args = build_engine_args(ts_code, freq, bars)
engine = MergedEngine(df, args)
engine.run()
result_df = engine.df

# 查看3月17日到4月2日的详细数据
print("=== 3月17日到4月2日的详细数据 ===")
mask = (result_df.index >= '2026-03-17') & (result_df.index <= '2026-04-02')
recent_df = result_df[mask]

# 显示更多列
display_cols = ['close', 'high', 'low', 'rope', 'dir_turn_long_flag', 
                'pullback_buy_flag', 'breakout_quality_score', 'vol_zscore']
available_cols = [c for c in display_cols if c in result_df.columns]

print(recent_df[available_cols])

print("\n=== 翻多事件提取条件分析 ===")
print("翻多事件需要满足:")
print("1. dir_turn_long_flag == 1.0")
print("2. breakout_quality_score 不为 NaN")

# 检查3月17日翻多事件后的数据
mar_17_idx = result_df.index.get_loc('2026-03-17 15:00:00')
print(f"\n3月17日翻多事件后的数据（从3月18日开始）:")
post_breakout = result_df.iloc[mar_17_idx+1:mar_17_idx+15]
print(post_breakout[available_cols])
