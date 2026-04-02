import sys, os, argparse, numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import pandas as pd

from batch_validate_v2 import fetch_kline
from atr_rope_with_factors_pytdx_plotly import ATRRopeEngine
from dynamic_swing_anchored_vwap import DSAConfig, dynamic_swing_anchored_vwap
from unified_pullback_factor_validator import detect_trend_state, rolling_trailing_pivots

ts_code = '002309.SZ'
df = fetch_kline(ts_code, 'd', 200)
print('=== fetch_kline result ===')
print('columns:', list(df.columns))
print('len:', len(df))
print(df[['open','high','low','close','vol']].tail(3).to_string())
df = df.rename(columns={'vol': 'volume'})
for col in ['open','high','low','close','volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print()

print('=== rope engine ===')
try:
    rope_engine = ATRRopeEngine(df.copy(), argparse.Namespace(
        len=14, multi=1.5, source='close', freq='d',
        show_ranges=True, show_atr_channel=True,
        show_break_markers=False, show_factor_panels=True,
    ))
    rope_engine.run()
    rope_df = rope_engine.df.copy()
    print('rope_df columns:', list(rope_df.columns)[:20])
    if 'rope_dir' in rope_df.columns:
        valid = rope_df['rope_dir'].dropna()
        print(f'rope_dir: valid count={len(valid)}, sample={valid.head(5).tolist()}')
    else:
        print('rope_dir MISSING from rope_df')
except Exception as e:
    print('rope engine error:', e)
    rope_df = df.copy()

print()
print('=== dsa ===')
try:
    dsa_cfg = DSAConfig(prd=50, baseAPT=20.0, useAdapt=False, volBias=10.0)
    dsa_vwap, dsa_dir, pivot_labels, _segments = dynamic_swing_anchored_vwap(df.copy(), dsa_cfg)
    print(f'dsa_dir dtype: {dsa_dir.dtype}')
    print(f'dsa_dir valid: {dsa_dir.dropna().shape}, sample: {dsa_dir.dropna().head(5).tolist()}')
except Exception as e:
    print('dsa error:', e)
    dsa_dir = pd.Series(0, index=df.index)

print()
print('=== detect_trend_state ===')
ts = detect_trend_state(df, 20)
print(f'result: {ts}')

print()
print('=== if we join rope_dir to df ===')
for col in ['rope', 'upper', 'lower', 'rope_dir']:
    if col in rope_df.columns:
        df[col] = rope_df[col].reindex(df.index).values
    else:
        df[col] = np.nan
print(f'After joining: rope_dir valid={df["rope_dir"].dropna().shape}')
print(f'df columns now: {[c for c in df.columns if "rope" in c or "dsa" in c]}')
ts2 = detect_trend_state(df, 20)
print(f'detect_trend_state after join: {ts2}')