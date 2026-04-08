# -*- coding: utf-8 -*-
"""验证华电辽能 2026-03-02 的 DSA 计算"""
import pandas as pd
import importlib.util
import sys

spec = importlib.util.spec_from_file_location('c2_module', '/Users/zhenbao/Nextcloud/coding/交易/backtrader/c2_main_strategy.py')
c2_module = importlib.util.module_from_spec(spec)
sys.modules['c2_module'] = c2_module
spec.loader.exec_module(c2_module)

load_daily_from_db = c2_module.load_daily_from_db
FactorEngine = c2_module.FactorEngine

symbol = '600396.SH'
end_date = '2026-03-04'
start_date = '2025-07-01'

raw = load_daily_from_db(symbol, start_date, end_date)
if raw.empty:
    print(f'无法获取 {symbol} 的数据')
    exit()

raw.index = pd.to_datetime(raw.index)
print(f'数据范围: {raw.index.min()} ~ {raw.index.max()}')
print(f'数据条数: {len(raw)}')

engine = FactorEngine()
daily = engine.compute_daily_factors(raw)
weekly = engine.compute_weekly_strict_dsa(raw, 'W-FRI')
df = pd.concat([daily, weekly], axis=1)

# 归一化日期
df.index = df.index.normalize()

# 检查 2026-03-02 的数据
target = pd.Timestamp('2026-03-02')
if target not in df.index:
    print(f'2026-03-02 不在数据中')
    print(f'可用日期: {df.index[-10:].tolist()}')
    exit()

row = df.loc[target]
print(f'\n2026-03-02 的因子值:')
print(f'  close: {row["close"]}')
print(f'  dsa_pivot_pos_01: {row["dsa_pivot_pos_01"]}')
print(f'  dsa_pivot_high: {row["dsa_pivot_high"]}')
print(f'  dsa_pivot_low: {row["dsa_pivot_low"]}')
print(f'  DSA_VWAP: {row["DSA_VWAP"]}')
print(f'  DSA_DIR: {row["DSA_DIR"]}')

# 手动计算验证
close = row['close']
high = row.get('dsa_pivot_high', float('nan'))
low = row.get('dsa_pivot_low', float('nan'))

if pd.notna(high) and pd.notna(low) and high > low:
    manual_calc = (close - low) / (high - low)
    print(f'\n手动计算:')
    print(f'  (close - low) / (high - low)')
    print(f'  = ({close} - {low}) / ({high} - {low})')
    print(f'  = {close - low} / {high - low}')
    print(f'  = {manual_calc}')
    print(f'  代码结果: {row["dsa_pivot_pos_01"]}')

# 查看最近的 pivot points
print(f'\n最近 30 天的 dsa_pivot_high 和 dsa_pivot_low:')
recent = df.iloc[-30:].copy()
for idx, (date, r) in enumerate(recent.iterrows()):
    if pd.notna(r.get('dsa_pivot_high')) or pd.notna(r.get('dsa_pivot_low')):
        print(f'{str(date)[:10]}: close={r["close"]:.2f}, high={r.get("dsa_pivot_high", 0):.2f}, low={r.get("dsa_pivot_low", 0):.2f}, d_dsa={r.get("dsa_pivot_pos_01", 0):.3f}')
