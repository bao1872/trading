import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, '.')

from pytdx.hq import TdxHq_API
api = TdxHq_API()
api.connect("115.238.90.165", 7709)
klines = api.get_security_bars(category=9, market=0, code="000001", start=0, count=80)
api.disconnect()
df = pd.DataFrame(klines)
df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
df = df.set_index("datetime")
for col in ["open", "high", "low", "close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

from smc_factor_table import compute_smc_factors, _init_globals
_init_globals(df["high"].values, df["low"].values, list(df.index))
smc = compute_smc_factors(df, swing_size=20, internal_size=5)
result = df.join(smc)

sh = result[result["swing_high"].notna()][["swing_high", "swing_high_type"]].copy()
sl = result[result["swing_low"].notna()][["swing_low", "swing_low_type"]].copy()
print("=== Real data 000001.SH (80 bars) ===")
print(f"Swing highs: {len(sh)} (with HH/LH: {(sh['swing_high_type'] != '').sum()})")
print(sh.head(10).to_string())
print(f"\nSwing lows: {len(sl)} (with HL/LL: {(sl['swing_low_type'] != '').sum()})")
print(sl.head(10).to_string())

fv = result[result["fvg_valid"] == 1][["bullish_fvg_top", "bullish_fvg_bottom", "bearish_fvg_top", "bearish_fvg_bottom"]]
print(f"\nFVG valid bars: {len(fv)}")
if not fv.empty:
    print(fv.head(5).to_string())

bos = result[(result["bullish_bos_bar"] == 1) | (result["bearish_bos_bar"] == 1)]
print(f"\nBOS bars: {len(bos)}")
if not bos.empty:
    print(bos[["bullish_bos_bar", "bearish_bos_bar"]].head(5).to_string())

print("\nAll non-NaN SMC columns:")
for col in result.columns:
    nn = result[col].notna().sum()
    if nn > 0 and col not in ["open", "high", "low", "close"]:
        print(f"  {col}: {nn} non-NaN")