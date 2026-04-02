# -*- coding: utf-8 -*-
"""
SMC 逐 bar 因子表生成模块（无未来函数版）

设计原则：
- No repainting：pivot/swing 结构一旦确认不再修改
- 逐 bar 状态机：每根 bar 独立计算当前状态
- 可独立运行，也可被 unified_pullback_factor_validator 引用

SMC 结构层核心逻辑来源：smc_luxalgo_pytdx_plotly.py
参考其 get_current_structure / leg / store_order_block / compute_fvg 的 Pine 兼容逻辑，
重构成逐 bar 表输出。

输出列（每根 bar）：
  swing_high / swing_low            : 当前活跃的 swing high/low 价格
  swing_high_type                   : HH/LH/None
  swing_low_type                    : HL/LL/None
  swing_high_bar_idx                : swing high 确认时的 bar 索引
  swing_low_bar_idx                 : swing low 确认时的 bar 索引
  swing_high_crossed                : 1=已向上突破当前 swing high
  swing_low_crossed                 : 1=已向下跌破当前 swing low
  trend_bias                        : 1=bullish, -1=bearish, 0=neutral
  bullish_ob_high / bullish_ob_low : 未失效的 bullish order block 价格带
  bearish_ob_high / bearish_ob_low : 未失效的 bearish order block 价格带
  swing_ob_valid                    : 1=有未失效的 swing OB
  internal_ob_valid                 : 1=有未失效的 internal OB
  bullish_fvg_top/bottom            : 未失效的 bullish FVG 价格带
  bearish_fvg_top/bottom            : 未失效的 bearish FVG 价格带
  fvg_valid                         : 1=有未失效的 FVG
  bullish_bos_bar / bullish_choch_bar : 最近一次 bullish BOS/CHOCH 的 bar 索引
  bearish_bos_bar / bearish_choch_bar : 最近一次 bearish BOS/CHOCH 的 bar 索引

Usage:
  python smc_factor_table.py --symbol 600547 --freq d --bars 200
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
for _p in (BASE_DIR, ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BEARISH, BULLISH, NEUTRAL = -1, 1, 0


@dataclass
class PivotRecord:
    level: float = np.nan
    bar_idx: int = -1
    crossed: bool = False
    bar_time: Optional[pd.Timestamp] = None


@dataclass
class OBRecord:
    high: float = np.nan
    low: float = np.nan
    bar_time: Optional[pd.Timestamp] = None
    bias: int = 0


@dataclass
class FVGRecord:
    top: float = np.nan
    bottom: float = np.nan
    bias: int = 0


def _rolling_high(arr: np.ndarray, i: int, size: int) -> float:
    start = i + 1
    end = min(i + size + 1, len(arr))
    if start < 0:
        start = 0
    if start >= end:
        return float(np.nan)
    return float(np.max(arr[start:end]))


def _rolling_low(arr: np.ndarray, i: int, size: int) -> float:
    start = i + 1
    end = min(i + size + 1, len(arr))
    if start < 0:
        start = 0
    if start >= end:
        return float(np.nan)
    return float(np.min(arr[start:end]))


def compute_smc_factors(
    df: pd.DataFrame,
    swing_size: int = 20,
    internal_size: int = 5,
    max_ob_count: int = 5,
    max_fvg_count: int = 5,
) -> pd.DataFrame:
    """计算 SMC 逐 bar 因子表，无未来函数。"""
    n = len(df)
    out = pd.DataFrame(index=df.index)

    for col in [
        "swing_high", "swing_low",
        "swing_high_type", "swing_low_type",
        "swing_high_bar_idx", "swing_low_bar_idx",
        "swing_high_crossed", "swing_low_crossed",
        "trend_bias",
        "internal_high", "internal_low",
        "internal_high_bar_idx", "internal_low_bar_idx",
        "bullish_ob_high", "bullish_ob_low",
        "bearish_ob_high", "bearish_ob_low",
        "swing_ob_valid", "internal_ob_valid",
        "bullish_fvg_top", "bullish_fvg_bottom",
        "bearish_fvg_top", "bearish_fvg_bottom",
        "fvg_valid",
        "bullish_bos_bar", "bullish_choch_bar",
        "bearish_bos_bar", "bearish_choch_bar",
    ]:
        out[col] = np.nan

    for col in [
        "swing_high_crossed", "swing_low_crossed",
        "trend_bias", "swing_ob_valid", "internal_ob_valid", "fvg_valid",
    ]:
        out[col] = 0

    for col in ["swing_high_type", "swing_low_type"]:
        out[col] = ""

    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    opens = df["open"].values.astype(float)

    swing_high = PivotRecord()
    swing_low = PivotRecord()
    internal_high = PivotRecord()
    internal_low = PivotRecord()

    swing_trend_bias = 0
    last_swing_high = np.nan
    last_swing_low = np.nan

    swing_obs: List[OBRecord] = []
    internal_obs: List[OBRecord] = []
    fvgs: List[FVGRecord] = []

    for i in range(n):
        out.at[out.index[i], "trend_bias"] = swing_trend_bias

        if i <= swing_size:
            out.at[out.index[i], "swing_high"] = np.nan
            out.at[out.index[i], "swing_low"] = np.nan
            continue

        if i <= internal_size:
            out.at[out.index[i], "internal_high"] = np.nan
            out.at[out.index[i], "internal_low"] = np.nan
            continue

        ref_swing = i - swing_size
        ref_internal = i - internal_size

        prev_swing_leg = 0
        if i > swing_size:
            prev_ref = ref_swing - 1
            prev_h_max = _rolling_high(highs, prev_ref, swing_size)
            prev_l_min = _rolling_low(lows, prev_ref, swing_size)
            prev_leg_h = highs[prev_ref] > prev_h_max
            prev_leg_l = lows[prev_ref] < prev_l_min
            if prev_leg_h and not prev_leg_l:
                prev_swing_leg = 1
            elif prev_leg_l and not prev_leg_h:
                prev_swing_leg = -1

        curr_h_max = _rolling_high(highs, ref_swing, swing_size)
        curr_l_min = _rolling_low(lows, ref_swing, swing_size)
        curr_leg_h = highs[ref_swing] > curr_h_max
        curr_leg_l = lows[ref_swing] < curr_l_min

        new_swing_high = curr_leg_h and not curr_leg_l
        new_swing_low = curr_leg_l and not curr_leg_h

        if new_swing_high:
            level = float(highs[ref_swing])
            if np.isfinite(last_swing_high):
                out.at[out.index[i], "swing_high_type"] = "HH" if level > last_swing_high else "LH"
            last_swing_high = level
            swing_high = PivotRecord(
                level=level,
                bar_idx=ref_swing,
                crossed=False,
                bar_time=df.index[ref_swing],
            )

        if new_swing_low:
            level = float(lows[ref_swing])
            if np.isfinite(last_swing_low):
                out.at[out.index[i], "swing_low_type"] = "HL" if level < last_swing_low else "LL"
            last_swing_low = level
            swing_low = PivotRecord(
                level=level,
                bar_idx=ref_swing,
                crossed=False,
                bar_time=df.index[ref_swing],
            )

        int_h_max = _rolling_high(highs, ref_internal, internal_size)
        int_l_min = _rolling_low(lows, ref_internal, internal_size)
        new_internal_high = highs[ref_internal] > int_h_max
        new_internal_low = lows[ref_internal] < int_l_min

        if new_internal_high:
            internal_high = PivotRecord(
                level=float(highs[ref_internal]),
                bar_idx=ref_internal,
                crossed=False,
                bar_time=df.index[ref_internal],
            )
        if new_internal_low:
            internal_low = PivotRecord(
                level=float(lows[ref_internal]),
                bar_idx=ref_internal,
                crossed=False,
                bar_time=df.index[ref_internal],
            )

        crossed_above_sh = (
            i > 0
            and not swing_high.crossed
            and np.isfinite(swing_high.level)
            and closes[i - 1] <= swing_high.level < closes[i]
        )
        crossed_below_sl = (
            i > 0
            and not swing_low.crossed
            and np.isfinite(swing_low.level)
            and closes[i - 1] >= swing_low.level > closes[i]
        )

        if crossed_above_sh:
            swing_high.crossed = True
            out.at[out.index[i], "swing_high_crossed"] = 1

        if crossed_below_sl:
            swing_low.crossed = True
            out.at[out.index[i], "swing_low_crossed"] = 1

        if crossed_above_sh and not swing_low.crossed:
            tag = "CHOCH" if swing_trend_bias == BEARISH else "BOS"
            if tag == "CHOCH":
                swing_trend_bias = BULLISH
                out.at[out.index[i], "bullish_choch_bar"] = i
            else:
                out.at[out.index[i], "bullish_bos_bar"] = i
            swing_low.crossed = True
            out.at[out.index[i], "swing_low_crossed"] = 1
            _store_ob(swing_low, i, BULLISH, swing_obs, max_ob_count)

        if crossed_below_sl and not swing_high.crossed:
            tag = "CHOCH" if swing_trend_bias == BULLISH else "BOS"
            if tag == "CHOCH":
                swing_trend_bias = BEARISH
                out.at[out.index[i], "bearish_choch_bar"] = i
            else:
                out.at[out.index[i], "bearish_bos_bar"] = i
            swing_high.crossed = True
            out.at[out.index[i], "swing_high_crossed"] = 1
            _store_ob(swing_high, i, BEARISH, swing_obs, max_ob_count)

        out.at[out.index[i], "swing_high"] = swing_high.level if np.isfinite(swing_high.level) else np.nan
        out.at[out.index[i], "swing_low"] = swing_low.level if np.isfinite(swing_low.level) else np.nan
        out.at[out.index[i], "swing_high_bar_idx"] = swing_high.bar_idx
        out.at[out.index[i], "swing_low_bar_idx"] = swing_low.bar_idx

        out.at[out.index[i], "internal_high"] = internal_high.level if np.isfinite(internal_high.level) else np.nan
        out.at[out.index[i], "internal_low"] = internal_low.level if np.isfinite(internal_low.level) else np.nan
        out.at[out.index[i], "internal_high_bar_idx"] = internal_high.bar_idx
        out.at[out.index[i], "internal_low_bar_idx"] = internal_low.bar_idx

        if new_swing_low:
            _store_ob(swing_low, i, BULLISH, swing_obs, max_ob_count)
        if new_swing_high:
            _store_ob(swing_high, i, BEARISH, swing_obs, max_ob_count)

        _invalidate_obs(swing_obs, highs[i], lows[i], closes[i])
        _invalidate_obs(internal_obs, highs[i], lows[i], closes[i])

        bull_ob = _best_ob(swing_obs, BULLISH)
        bear_ob = _best_ob(swing_obs, BEARISH)
        int_bull_ob = _best_ob(internal_obs, BULLISH)
        int_bear_ob = _best_ob(internal_obs, BEARISH)

        if bull_ob is not None:
            out.at[out.index[i], "bullish_ob_high"] = bull_ob.high
            out.at[out.index[i], "bullish_ob_low"] = bull_ob.low
            out.at[out.index[i], "swing_ob_valid"] = 1
        if bear_ob is not None:
            out.at[out.index[i], "bearish_ob_high"] = bear_ob.high
            out.at[out.index[i], "bearish_ob_low"] = bear_ob.low
        if int_bull_ob is not None or int_bear_ob is not None:
            out.at[out.index[i], "internal_ob_valid"] = 1

        _compute_fvg(i, df, fvgs, max_fvg_count, out)
        out.at[out.index[i], "fvg_valid"] = 1 if len(fvgs) > 0 else 0

    return out


def _store_ob(piv: PivotRecord, current_i: int, bias: int, obs_list: List[OBRecord], max_count: int):
    if piv.bar_idx < 0 or current_i <= piv.bar_idx + 1:
        return
    start = piv.bar_idx
    end = current_i
    if end <= start:
        return
    if bias == BEARISH:
        arr = highs_global[start:end]
        local_idx = int(np.argmax(arr))
    else:
        arr = lows_global[start:end]
        local_idx = int(np.argmin(arr))
    parsed_idx = start + local_idx
    ob = OBRecord(
        high=float(highs_global[parsed_idx]),
        low=float(lows_global[parsed_idx]),
        bar_time=times_global[parsed_idx] if parsed_idx < len(times_global) else None,
        bias=bias,
    )
    obs_list.insert(0, ob)
    if len(obs_list) > max_count:
        obs_list.pop()


highs_global = None
lows_global = None
times_global = None


def _init_globals(highs, lows, times):
    global highs_global, lows_global, times_global
    highs_global = highs
    lows_global = lows
    times_global = times


def _invalidate_obs(obs: List[OBRecord], high: float, low: float, close: float):
    for ob in obs:
        if np.isnan(ob.high) or np.isnan(ob.low):
            continue
        if ob.bias == BULLISH and close < ob.low:
            ob.high = np.nan
            ob.low = np.nan
        elif ob.bias == BEARISH and close > ob.high:
            ob.high = np.nan
            ob.low = np.nan


def _best_ob(obs: List[OBRecord], bias: int) -> Optional[OBRecord]:
    candidates = [
        ob for ob in obs
        if ob.bias == bias and np.isfinite(ob.high) and np.isfinite(ob.low)
    ]
    return candidates[0] if candidates else None


def _compute_fvg(i: int, df: pd.DataFrame, fvgs: List[FVGRecord], max_count: int, out: pd.DataFrame):
    if i < 3:
        return
    last_close = df["close"].iloc[i - 1]
    last2_high = df["high"].iloc[i - 2]
    last2_low = df["low"].iloc[i - 2]
    current_high = df["high"].iloc[i]
    current_low = df["low"].iloc[i]
    last_open = df["open"].iloc[i - 1]
    if last_open == 0:
        delta_pct = 0.0
    else:
        delta_pct = abs((last_close - last_open) / (last_open * 100.0))

    gap_found = False
    if current_low > last2_high and last_close > last2_high and delta_pct > 0:
        fvgs.insert(0, FVGRecord(top=float(current_low), bottom=float(last2_high), bias=BULLISH))
        gap_found = True
    elif current_high < last2_low and last_close < last2_low and delta_pct > 0:
        fvgs.insert(0, FVGRecord(top=float(current_high), bottom=float(last2_low), bias=BEARISH))
        gap_found = True

    if gap_found and len(fvgs) > max_count:
        fvgs.pop()

    close_i = df["close"].iloc[i]
    for fvg in fvgs:
        if np.isnan(fvg.top) or np.isnan(fvg.bottom):
            continue
        if fvg.bias == BULLISH and close_i < fvg.bottom:
            fvg.top = np.nan
            fvg.bottom = np.nan
        elif fvg.bias == BEARISH and close_i > fvg.top:
            fvg.top = np.nan
            fvg.bottom = np.nan

    bull_fvg = next((f for f in fvgs if f.bias == BULLISH and np.isfinite(f.bottom)), None)
    bear_fvg = next((f for f in fvgs if f.bias == BEARISH and np.isfinite(f.top)), None)
    if bull_fvg is not None:
        out.at[out.index[i], "bullish_fvg_top"] = bull_fvg.top
        out.at[out.index[i], "bullish_fvg_bottom"] = bull_fvg.bottom
    if bear_fvg is not None:
        out.at[out.index[i], "bearish_fvg_top"] = bear_fvg.top
        out.at[out.index[i], "bearish_fvg_bottom"] = bear_fvg.bottom


def fetch_data(symbol: str, freq: str, bars: int) -> pd.DataFrame:
    try:
        import pytdx
        from pytdx.hq import TdxHq_API
        api = TdxHq_API()
        api.connect("115.238.90.165", 7709)
        code_str = symbol.split(".")[0]
        mkt = 0 if symbol.endswith(".SH") else 1
        freq_map = {"d": 9, "w": 5, "m": 6, "5m": 0, "15m": 1, "30m": 2, "60m": 3}
        cat = freq_map.get(freq, 9)
        klines = api.get_security_bars(category=cat, market=mkt, code=code_str, start=0, count=bars)
        api.disconnect()
        if not klines:
            return pd.DataFrame()
        df = pd.DataFrame(klines)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df = df.rename(columns={"vol": "volume"})
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        print(f"pytdx fetch error: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMC 逐 bar 因子表")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--freq", default="d")
    parser.add_argument("--bars", type=int, default=200)
    parser.add_argument("--swing-size", type=int, default=20)
    parser.add_argument("--internal-size", type=int, default=5)
    parser.add_argument("--out", default="/tmp/smc_factors.csv")
    args = parser.parse_args()

    # Normalize symbol format to ts_code (e.g., 600547 -> 600547.SH)
    symbol = args.symbol
    if not symbol.endswith((".SH", ".SZ")):
        symbol = symbol + ".SH"

    df = fetch_data(symbol, args.freq, args.bars)
    if df.empty:
        print(f"获取数据失败: {symbol}")
        sys.exit(1)

    _init_globals(df["high"].values, df["low"].values, list(df.index))

    smc = compute_smc_factors(
        df,
        swing_size=args.swing_size,
        internal_size=args.internal_size,
    )

    result = df.join(smc)
    result.to_csv(args.out, encoding="utf-8-sig")
    print(f"[OK] SMC因子表已保存: {args.out}")
    print(f"行数: {len(result)}, 列数: {len(result.columns)}")
    print(f"列名: {list(result.columns)}")

    sv = result[result["swing_high"].notna()]
    if not sv.empty:
        print("\n=== 最近10个 swing pivot ===")
        print(sv[["swing_high", "swing_high_type", "swing_low", "swing_low_type", "trend_bias", "bullish_bos_bar", "bullish_choch_bar"]].tail(10).to_string())

    fv = result[result["fvg_valid"] == 1]
    if not fv.empty:
        print("\n=== 最近5个 FVG ===")
        print(fv[["bullish_fvg_top", "bullish_fvg_bottom", "bearish_fvg_top", "bearish_fvg_bottom"]].tail(5).to_string())

    obv = result[result["swing_ob_valid"] == 1]
    if not obv.empty:
        print("\n=== 最近5个 OB ===")
        print(obv[["bullish_ob_high", "bullish_ob_low", "bearish_ob_high", "bearish_ob_low"]].tail(5).to_string())