# -*- coding: utf-8 -*-
"""
因子计算统一入口 + 验证

Purpose:
    封装因子库核心计算函数的调用逻辑，提供统一入口，并验证与 factor_value_1d 的一致性。
    核心原则: 不重写任何因子计算逻辑，只引用 features/ 下的权威实现（SSOT）。

Pipeline Position:
    训练流水线工具模块（被 01_build_dataset.py import）。
    上游: features/ 下的权威因子实现
    下游: 01_build_dataset.py

Inputs:
    - DB: stock_k_data, stop_loss_selection
    - DB: factor_value_1d (验证模式)

Outputs:
    - DataFrame: 计算后的因子矩阵（返回给调用方）

How to Run:
    python stop_experiment/pipeline/compute_factors.py --verify --ts-codes 000001.SZ 600519.SH

Side Effects:
    - 验证模式会读取 factor_value_1d 表，不写入任何数据
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.factor_columns import (
    TREND_COLS, POSITION_COLS, MOMENTUM_COLS,
    VOLUME_COLS, RISK_COLS, RHYTHM_COLS, VSA_COLS,
    ALL_FEATURE_COLS,
)
from stop_experiment.pipeline.stop_config import DATASET_PATH, OUTPUT_DIR


def compute_stock_factors(df_kline: pd.DataFrame) -> pd.DataFrame:
    """
    对单只股票的K线数据计算全部因子库因子。

    Parameters
    ----------
    df_kline : pd.DataFrame
        需含列: open, high, low, close, volume。index 为 datetime。

    Returns
    -------
    pd.DataFrame
        index 与输入一致，列为 ALL_FEATURE_COLS 中属于因子库的列。
        绝对值列已被剔除，只保留可跨股票比较的因子。
    """
    from features.dsa_bbmacd_24factors_viewer import compute_dsa, compute_bbmacd, DSAConfig

    cfg = DSAConfig()

    # 1. compute_dsa → 趋势+位置+节奏因子
    dsa_result, _, _ = compute_dsa(df_kline, cfg)

    # 2. compute_bbmacd → 动量因子
    bb_result = compute_bbmacd(df_kline)

    # 3. volume_zscore → 量能因子
    vol_z_5, _, _ = volume_zscore(df_kline["volume"], 5)
    vol_z_10, _, _ = volume_zscore(df_kline["volume"], 10)
    vol_z_20, _, _ = volume_zscore(df_kline["volume"], 20)

    # 4. compute_24_factors → 合并部分派生因子（trend_align_momo, prev_pivot_code 等）
    #    直接调用 compute_24_factors 会把 dsa 和 bbmacd 的结果重新计算一遍，
    #    但它需要的列（dsa_dir, bbmacd_minus_avg 等）已经在 dsa_result 和 bb_result 中，
    #    所以我们手动拼接 compute_24_factors 需要的列来避免重复计算
    merged = df_kline.copy()
    for col in dsa_result.columns:
        merged[col] = dsa_result[col]
    for col in bb_result.columns:
        merged[col] = bb_result[col]

    # 手动计算 compute_24_factors 中的派生列（避免重新计算 dsa + bbmacd）
    pivot_code_map = {"HH": 2.0, "HL": 1.0, "LH": -1.0, "LL": -2.0}
    merged["prev_pivot_code"] = merged["prev_pivot_type"].map(pivot_code_map).astype(float)
    merged["ret_to_last_high_pct"] = merged["close"] / merged["last_confirmed_high"] - 1.0
    merged["ret_to_last_low_pct"] = merged["close"] / merged["last_confirmed_low"] - 1.0
    merged["price_vs_dsa_vwap_pct"] = merged["close"] / merged["DSA_VWAP"] - 1.0

    # trend_align_momo
    trend_align = np.zeros(len(merged), dtype=float)
    long_align = (merged["dsa_dir"] > 0) & (merged["bbmacd_minus_avg"] > 0)
    short_align = (merged["dsa_dir"] < 0) & (merged["bbmacd_minus_avg"] < 0)
    trend_align[long_align.to_numpy()] = 1.0
    trend_align[short_align.to_numpy()] = -1.0
    merged["trend_align_momo"] = trend_align

    # dsa_dir_age
    dir_series = merged["dsa_dir"]
    age = np.zeros(len(dir_series), dtype=int)
    current_age = 0
    for i in range(len(dir_series)):
        if i == 0 or dir_series.iloc[i] != dir_series.iloc[i - 1]:
            current_age = 0
        else:
            current_age += 1
        age[i] = current_age
    merged["dsa_dir_age"] = age

    # bbmacd_sign, bbmacd_slope_3_pct（归一化版本，可跨股比较）
    merged["bbmacd_sign"] = np.sign(merged["bbmacd"])
    merged["bbmacd_slope_3"] = merged["bbmacd"].diff(3)
    # bbmacd_slope_3_pct: bbmacd_slope_3 / close，归一化为百分比
    close_safe = merged["close"].replace(0, np.nan)
    merged["bbmacd_slope_3_pct"] = merged["bbmacd_slope_3"] / close_safe
    merged["bbmacd_slope_3_pct"] = merged["bbmacd_slope_3_pct"].replace([np.inf, -np.inf], np.nan)

    # current_pullback_from_stage_extreme_pct（compute_24_factors 中的逻辑）
    merged["current_pullback_from_stage_extreme_pct"] = _compute_current_pullback(merged)

    # 5. 量能因子
    merged["vol_zscore_5"] = vol_z_5
    merged["vol_zscore_10"] = vol_z_10
    merged["vol_zscore_20"] = vol_z_20
    merged["vol_ratio_10"] = df_kline["volume"] / df_kline["volume"].rolling(10, min_periods=1).mean()
    merged["days_since_vol_spike"] = _compute_days_since_vol_spike(vol_z_20, threshold=2.0)
    merged["vol_stage_cv"] = _compute_vol_stage_cv(df_kline["volume"], dsa_result["dsa_dir"])
    merged["vol_prev_stage_cv"] = merged["vol_stage_cv"].shift(20)
    merged["vol_cv_ratio"] = merged["vol_stage_cv"] / merged["vol_prev_stage_cv"].replace(0.0, np.nan)

    # 6. 风险因子
    merged["atr_pct"] = _compute_atr_pct(df_kline)
    merged["volatility_20d"] = _compute_volatility_20d(df_kline)
    merged["max_drawdown_60d"] = _compute_max_drawdown_60d(df_kline)
    merged["beta"] = _compute_beta(df_kline)

    # 7. liquidity_range_pos_01（来自 compute_dsa 的 dsa_pivot_pos_01 近似）
    #    更精确的计算需要 sell_stop_cluster / buy_stop_cluster
    #    但因子库注册时用的就是 dsa_pivot_pos_01，保持一致
    merged["liquidity_range_pos_01"] = merged["dsa_pivot_pos_01"]

    # 8. price_vol_coord（价量协同方向）
    merged["price_vol_coord"] = _compute_price_vol_coord(df_kline, merged["dsa_dir"])

    # 9. stage相关派生（runs 计算）
    merged["current_stage_amp_pct"], merged["current_stage_ret_pct"] = _compute_stage_metrics(merged)
    merged["prev_stage_amp_pct"] = merged["current_stage_amp_pct"].shift(1)

    # 10. VSA 量价因子（引用 factor_lib 权威实现，受 VSA_ENABLED 控制）
    from stop_experiment.pipeline.stop_config import VSA_ENABLED
    if VSA_ENABLED:
        from factor_lib.categories.quantity_price import compute_quantity_price_factors
        vsa_factors = compute_quantity_price_factors(df_kline)
        for col in vsa_factors.columns:
            merged[col] = vsa_factors[col]

    # 只保留需要的列
    factor_cols = [c for c in ALL_FEATURE_COLS if c in merged.columns]
    result = merged[factor_cols].copy()

    return result


# ==================== 辅助计算函数 ====================

def volume_zscore(vol: pd.Series, win: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """成交量Z-score（引用权威实现）"""
    from features.volume_zscore_plotly import volume_zscore as _vz
    return _vz(vol, win)


def _compute_days_since_vol_spike(vol_zscore_20: pd.Series, threshold: float = 2.0) -> pd.Series:
    """距上次放量（z-score > threshold）天数"""
    is_spike = vol_zscore_20 > threshold
    result = pd.Series(np.nan, index=vol_zscore_20.index, dtype=float)
    last_spike_idx = -1
    for i in range(len(is_spike)):
        if is_spike.iloc[i]:
            last_spike_idx = i
        if last_spike_idx >= 0:
            result.iloc[i] = float(i - last_spike_idx)
    return result


def _compute_vol_stage_cv(vol: pd.Series, dsa_dir: pd.Series, window: int = 20) -> pd.Series:
    """当前阶段量CV（20日滚动 std/mean）"""
    mu = vol.rolling(window, min_periods=1).mean()
    sd = vol.rolling(window, min_periods=1).std()
    return sd / mu.replace(0.0, np.nan)


def _compute_atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR百分比（与 factor_lib/categories/risk.py 一致）"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr / close * 100


def _compute_volatility_20d(df: pd.DataFrame) -> pd.Series:
    """20日收益率波动率（年化%，与 factor_lib/categories/risk.py 一致）"""
    returns = df["close"].pct_change()
    return returns.rolling(window=20, min_periods=1).std() * np.sqrt(252) * 100


def _compute_max_drawdown_60d(df: pd.DataFrame) -> pd.Series:
    """60日最大回撤（%，与 factor_lib/categories/risk.py 一致）"""
    close = df["close"]
    rolling_max = close.rolling(window=60, min_periods=1).max()
    drawdown = (close - rolling_max) / rolling_max * 100
    return drawdown.rolling(window=60, min_periods=1).min()


def _compute_beta(df: pd.DataFrame) -> pd.Series:
    """Beta系数（简化版，与 factor_lib/categories/risk.py 一致）"""
    returns = df["close"].pct_change()
    return returns.rolling(window=60, min_periods=1).std() * np.sqrt(252)


def _compute_price_vol_coord(df: pd.DataFrame, dsa_dir: pd.Series) -> pd.Series:
    """价量协同方向：1=价涨量增，-1=价涨量减或价跌量增"""
    price_up = df["close"].diff() > 0
    vol_up = df["volume"].diff() > 0
    coord = pd.Series(0, index=df.index, dtype=float)
    coord[price_up & vol_up] = 1.0
    coord[~price_up & ~vol_up] = -1.0
    return coord


def _compute_stage_metrics(merged: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """计算 current_stage_amp_pct 和 current_stage_ret_pct"""
    from features.dsa_bbmacd_24factors_viewer import _run_from_dir

    dir_out = merged["dsa_dir"].to_numpy(float)
    runs = _run_from_dir(dir_out)

    current_stage_amp = np.full(len(merged), np.nan)
    current_stage_ret = np.full(len(merged), np.nan)

    for st, ed, run_dir in runs:
        seg_high = float(np.nanmax(merged["high"].iloc[st:ed + 1]))
        seg_low = float(np.nanmin(merged["low"].iloc[st:ed + 1]))
        seg_amp = (seg_high - seg_low) / seg_low if np.isfinite(seg_low) and seg_low != 0 else np.nan
        current_stage_amp[st:ed + 1] = seg_amp

        start_close = float(merged["close"].iloc[st])
        if np.isfinite(start_close) and start_close != 0:
            current_stage_ret[st:ed + 1] = merged["close"].iloc[st:ed + 1] / start_close - 1.0

    return (
        pd.Series(current_stage_amp, index=merged.index),
        pd.Series(current_stage_ret, index=merged.index),
    )


def _compute_current_pullback(merged: pd.DataFrame) -> pd.Series:
    """计算 current_pullback_from_stage_extreme_pct"""
    from features.dsa_bbmacd_24factors_viewer import _run_from_dir

    dir_out = merged["dsa_dir"].to_numpy(float)
    runs = _run_from_dir(dir_out)
    current_pullback = np.full(len(merged), np.nan)

    for st, ed, run_dir in runs:
        if run_dir > 0:
            roll_high = merged["high"].iloc[st:ed + 1].cummax().to_numpy(dtype=float)
            curr_close = merged["close"].iloc[st:ed + 1].to_numpy(dtype=float)
            current_pullback[st:ed + 1] = curr_close / roll_high - 1.0
        else:
            roll_low = merged["low"].iloc[st:ed + 1].cummin().to_numpy(dtype=float)
            curr_close = merged["close"].iloc[st:ed + 1].to_numpy(dtype=float)
            current_pullback[st:ed + 1] = curr_close / roll_low - 1.0

    return pd.Series(current_pullback, index=merged.index)


def verify_against_db(df_factors: pd.DataFrame, ts_code: str, n_samples: int = 100) -> dict:
    """
    随机抽 n_samples 条与 factor_value_1d 对比验证一致性。

    Parameters
    ----------
    df_factors : pd.DataFrame
        compute_stock_factors 的输出，index 为 datetime
    ts_code : str
        股票代码，如 '000001.SZ'
    n_samples : int
        验证样本数

    Returns
    -------
    dict
        {factor_name: {max_abs_diff, mean_abs_diff, n_compared, pass}}
    """
    from datasource.database import get_engine
    from sqlalchemy import text

    engine = get_engine()
    results = {}

    # 取有 factor_value_1d 数据的日期范围
    with engine.connect() as conn:
        r = conn.execute(text("""
            SELECT MIN(as_of_date), MAX(as_of_date) FROM factor_value_1d
            WHERE ts_code = :ts_code
        """), {"ts_code": ts_code})
        row = r.fetchone()
        if row[0] is None:
            return {"error": f"factor_value_1d 中无 {ts_code} 数据"}

    # 对每个因子验证
    common_factors = [c for c in df_factors.columns if c in {
        "dsa_dir", "prev_pivot_code", "trend_align_momo", "dsa_dir_age",
        "dsa_pivot_pos_01", "price_vs_dsa_vwap_pct", "ret_to_last_high_pct",
        "ret_to_last_low_pct", "current_pullback_from_stage_extreme_pct",
        "bbmacd_band_pos_01", "bbmacd_bandwidth_zscore", "bbmacd_sign",
        "bbmacd_slope_3", "bbmacd_state", "bbmacd_cross_upper",
        "bbmacd_cross_lower", "bbmacd_cross_up_lower", "bbmacd_cross_down_upper",
        "vol_zscore_5", "vol_zscore_10", "vol_zscore_20", "vol_ratio_10",
        "atr_pct", "volatility_20d", "max_drawdown_60d", "beta",
        "current_stage_bars", "current_stage_amp_pct", "current_stage_ret_pct",
        "prev_stage_bars", "prev_stage_amp_pct",
    }]

    with engine.connect() as conn:
        for factor_name in common_factors:
            r = conn.execute(text("""
                SELECT as_of_date, factor_value FROM factor_value_1d
                WHERE ts_code = :ts_code AND factor_name = :factor_name
                ORDER BY RANDOM() LIMIT :n
            """), {"ts_code": ts_code, "factor_name": factor_name, "n": n_samples})
            db_rows = r.fetchall()

            if not db_rows:
                continue

            diffs = []
            n_compared = 0
            for db_date, db_val in db_rows:
                if db_val is None or not np.isfinite(db_val):
                    continue
                # 找 df_factors 中对应日期的值
                if db_date not in df_factors.index:
                    continue
                our_val = df_factors.loc[db_date, factor_name]
                if pd.isna(our_val) or not np.isfinite(our_val):
                    continue
                diffs.append(abs(float(our_val) - float(db_val)))
                n_compared += 1

            if diffs:
                max_diff = max(diffs)
                mean_diff = sum(diffs) / len(diffs)
                results[factor_name] = {
                    "max_abs_diff": max_diff,
                    "mean_abs_diff": mean_diff,
                    "n_compared": n_compared,
                    "pass": max_diff < 1e-4,
                }
            else:
                results[factor_name] = {
                    "max_abs_diff": None,
                    "mean_abs_diff": None,
                    "n_compared": 0,
                    "pass": None,
                }

    return results


# ==================== 自测入口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="因子计算验证")
    parser.add_argument("--verify", action="store_true", help="验证与factor_value_1d一致性")
    parser.add_argument("--ts-codes", nargs="+", default=["000001.SZ"], help="验证用的股票代码")
    parser.add_argument("--bars", type=int, default=300, help="取K线根数")
    args = parser.parse_args()

    from datasource.database import get_engine
    from sqlalchemy import text

    engine = get_engine()

    for ts_code in args.ts_codes:
        print(f"\n{'='*60}")
        print(f"验证 {ts_code}")
        print(f"{'='*60}")

        # 从 stock_k_data 取K线
        with engine.connect() as conn:
            df_k = pd.read_sql(text("""
                SELECT bar_time, open, high, low, close, volume
                FROM stock_k_data
                WHERE ts_code = :ts_code AND freq = 'd'
                ORDER BY bar_time DESC LIMIT :n
            """), conn, params={"ts_code": ts_code, "n": args.bars})
            df_k = df_k.sort_values("bar_time").set_index("bar_time")
            print(f"  K线: {len(df_k)} 根, {df_k.index[0]} ~ {df_k.index[-1]}")

        # 计算因子
        factors = compute_stock_factors(df_k)
        print(f"  因子列数: {len(factors.columns)}")
        print(f"  非空率: {factors.notna().mean().mean():.1%}")

        if args.verify:
            verify_result = verify_against_db(factors, ts_code)
            pass_count = sum(1 for v in verify_result.values() if isinstance(v, dict) and v.get("pass"))
            fail_count = sum(1 for v in verify_result.values() if isinstance(v, dict) and v.get("pass") is False)
            skip_count = sum(1 for v in verify_result.values() if isinstance(v, dict) and v.get("pass") is None)
            print(f"\n  验证结果: 通过={pass_count}, 失败={fail_count}, 无数据={skip_count}")

            for fname, info in sorted(verify_result.items()):
                if not isinstance(info, dict):
                    continue
                status = "PASS" if info.get("pass") else "FAIL" if info.get("pass") is False else "SKIP"
                if info.get("max_abs_diff") is not None:
                    print(f"    {fname}: {status} max_diff={info['max_abs_diff']:.6f} n={info['n_compared']}")
                else:
                    print(f"    {fname}: {status}")
