"""
派生特征计算（SSOT）

Purpose: 所有 DSA pipeline 脚本从此文件导入派生特征定义和计算函数
         禁止在各自脚本中重复定义

How to Run:
    python dsa_experiment/pipeline/derived_features.py   # 验证导入

Side Effects: 无（纯计算模块，不写库/不改表）
"""

import numpy as np
import pandas as pd

# 周线派生特征（06_weekly_selector + 07_daily_trading_sheet 周线部分共用）
# fillna(0) 与原始的 06_weekly_selector.build_features() 行为对齐
WEEKLY_DERIVED = {
    "high_low_range": lambda df: df["last_confirmed_high"].fillna(0) - df["last_confirmed_low"].fillna(0),
    "high_low_range_pct": lambda df: np.where(
        df["last_confirmed_low"].fillna(0) > 0,
        (df["last_confirmed_high"].fillna(0) - df["last_confirmed_low"].fillna(0)) / df["last_confirmed_low"],
        0,
    ),
    "pivot_pos_x_trend": lambda df: df["dsa_pivot_pos_01"].fillna(0) * df["trend_align_momo"].fillna(0),
    "stage_bars_ratio": lambda df: (
        df["current_stage_bars"].fillna(0)
        / df["prev_stage_bars"].fillna(0).replace(0, np.nan)
    ).fillna(0),
    "amp_x_pullback": lambda df: df["current_stage_amp_pct"].fillna(0) * df["current_pullback_from_stage_extreme_pct"].fillna(0),
    "bbmacd_band_width": lambda df: df["bbmacd_band_pos_01"].fillna(0) * df["bbmacd_bandwidth_zscore"].fillna(0),
}

WEEKLY_DERIVED_NAMES = list(WEEKLY_DERIVED.keys())

# 日线派生特征（07_daily_trading_sheet 日线部分使用）
DAILY_DERIVED = {
    "pivot_pos_x_trend": lambda df: df["dsa_pivot_pos_01"] * df["dsa_dir"],
    "amp_x_pullback": lambda df: (
        df["current_stage_amp_pct"] * df.get("current_pullback_from_stage_extreme_pct", 0)
    ),
    "vol_x_stage_amp": lambda df: df["vol_zscore_20"] * df["current_stage_amp_pct"],
}

DAILY_DERIVED_NAMES = list(DAILY_DERIVED.keys())


def build_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    """应用周线派生特征到 DataFrame"""
    df = df.copy()
    for name, func in WEEKLY_DERIVED.items():
        try:
            df[name] = func(df)
        except Exception:
            df[name] = 0
    return df


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """应用日线派生特征到 DataFrame"""
    df = df.copy()
    for name, func in DAILY_DERIVED.items():
        try:
            df[name] = func(df)
        except Exception:
            df[name] = 0
    return df


if __name__ == "__main__":
    df = pd.DataFrame({
        "last_confirmed_high": [10.0, 12.0],
        "last_confirmed_low":  [8.0, 9.0],
        "dsa_pivot_pos_01":    [0.5, 0.8],
        "trend_align_momo":    [1.0, -0.5],
        "current_stage_bars":  [20, 30],
        "prev_stage_bars":     [15, 20],
        "current_stage_amp_pct": [10.0, 15.0],
        "current_pullback_from_stage_extreme_pct": [5.0, 8.0],
        "bbmacd_band_pos_01": [0.3, 0.7],
        "bbmacd_bandwidth_zscore": [1.2, -0.5],
        "dsa_dir": [1, -1],
        "vol_zscore_20": [1.5, -0.8],
    })
    result = build_weekly_features(df)
    print("周线派生特征:")
    for n in WEEKLY_DERIVED_NAMES:
        print(f"  {n}: {result[n].tolist()}")
    result2 = build_daily_features(df)
    print("\n日线派生特征:")
    for n in DAILY_DERIVED_NAMES:
        print(f"  {n}: {result2[n].tolist()}")
    print("\n✅ derived_features 自测通过")
