# -*- coding: utf-8 -*-
"""
查看完整列名和4月2号的数据
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

# 4月2号的数据
print("=== 4月2号的完整数据 ===")
apr_02 = result_df.loc['2026-04-02 15:00:00']
print(apr_02)

print("\n=== 所有列名 ===")
for i, col in enumerate(result_df.columns):
    print(f"{i}: {col}")

print("\n=== 4月2号关键指标 ===")
key_indicators = ['close', 'rope', 'dir', 'dir_turn_long_flag', 'pullback_buy_flag',
                  'breakout_quality_score', 'upper_band', 'lower_band']
for ind in key_indicators:
    if ind in result_df.columns:
        val = result_df.loc['2026-04-02 15:00:00', ind]
        print(f"{ind}: {val}")
