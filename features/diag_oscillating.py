import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, ".")

n = 100
highs = np.array([10.0 + 10.0 * abs(np.sin(i * 0.5)) for i in range(n)], dtype=float)
lows = highs - 0.5
df = pd.DataFrame({"open": highs, "high": highs, "low": lows, "close": highs},
                   index=pd.date_range("2024-01-01", periods=n, freq="D"))

from smc_factor_table import compute_smc_factors, _init_globals
_init_globals(df["high"].values, df["low"].values, list(df.index))

smc = compute_smc_factors(df, swing_size=10, internal_size=5)
result = df.join(smc)

sh = result[result["swing_high"].notna()][["swing_high", "swing_high_type"]].copy()
sh["type_str"] = sh["swing_high_type"].apply(lambda x: repr(str(x)))
print("Swing highs with types:")
print(sh.head(10).to_string())
print("\nUnique types:", sh["swing_high_type"].unique())
print(f"\nTotal SH: {len(sh)}, with type!=empty: {(sh['swing_high_type'] != '').sum()}")

sl = result[result["swing_low"].notna()][["swing_low", "swing_low_type"]].copy()
sl["type_str"] = sl["swing_low_type"].apply(lambda x: repr(str(x)))
print("\nSwing lows with types:")
print(sl.head(10).to_string())
print("\nUnique types:", sl["swing_low_type"].unique())
print(f"\nTotal SL: {len(sl)}, with type!=empty: {(sl['swing_low_type'] != '').sum()}")