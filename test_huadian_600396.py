# -*- coding: utf-8 -*-
"""
测试华电辽能(600396)的 DSA 计算，验证回看确认逻辑

Purpose
- 使用修改后的 compute_dsa 计算 600396 的 DSA 指标
- 检查 2026-03-02 的 dsa_pivot_pos_01 值
- 验证 C2 策略筛选条件

How to Run
    python test_huadian_600396.py
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "features"))

from features.merged_dsa_atr_rope_bb_factors import (
    DSAConfig,
    RopeConfig,
    compute_atr_rope,
    compute_bollinger,
    compute_dsa,
    fetch_kline_pytdx,
)


def test_huadian():
    symbol = "600396"
    print(f"=" * 70)
    print(f"测试股票: {symbol} 华电辽能")
    print(f"=" * 70)
    
    # 获取数据
    df = fetch_kline_pytdx(symbol, "d", 500)
    print(f"\n数据范围: {df.index[0]} ~ {df.index[-1]}, 共 {len(df)} 条")
    
    # 计算 DSA
    dsa_df, dsa_labels, dsa_segments = compute_dsa(df, DSAConfig(prd=50, base_apt=20.0))
    rope_df = compute_atr_rope(df, RopeConfig(length=14, multi=1.5))
    bb_df = compute_bollinger(df, length=20, mult=2.0, pct_lookback=120)
    
    # 合并数据
    merged = pd.concat([df, dsa_df, rope_df.drop(columns=df.columns, errors="ignore"), bb_df], axis=1)
    
    # 检查关键日期
    target_date = "2026-03-02"
    if target_date in merged.index.strftime("%Y-%m-%d"):
        row = merged.loc[merged.index.strftime("%Y-%m-%d") == target_date].iloc[0]
        print(f"\n【{target_date} 数据】")
        print(f"  收盘价: {row['close']:.2f}")
        print(f"  dsa_pivot_high: {row['dsa_pivot_high']:.2f}")
        print(f"  dsa_pivot_low: {row['dsa_pivot_low']:.2f}")
        print(f"  dsa_pivot_pos_01: {row['dsa_pivot_pos_01']:.4f}")
        print(f"  DSA_DIR: {row['DSA_DIR']}")
        print(f"  signed_vwap_dev_pct: {row['signed_vwap_dev_pct']:.4f}")
        print(f"  rope_dir: {row['rope_dir']}")
        print(f"  bars_since_dir_change: {row['bars_since_dir_change']}")
        print(f"  last_pivot_type: {row['last_pivot_type']}")
        
        # C2 策略条件检查
        print(f"\n【C2 策略条件检查】")
        c2_dsa_thr = 0.35
        c2_bars_thr = 3
        c2_vwap_thr = -1.0
        
        cond1 = row['dsa_pivot_pos_01'] <= c2_dsa_thr
        cond2 = row['rope_dir'] == 1
        cond3 = row['bars_since_dir_change'] <= c2_bars_thr
        cond4 = row['signed_vwap_dev_pct'] <= c2_vwap_thr
        
        print(f"  条件1: dsa_pivot_pos_01 ({row['dsa_pivot_pos_01']:.4f}) <= {c2_dsa_thr} -> {'✓' if cond1 else '✗'}")
        print(f"  条件2: rope_dir ({row['rope_dir']}) == 1 -> {'✓' if cond2 else '✗'}")
        print(f"  条件3: bars_since_dir_change ({row['bars_since_dir_change']}) <= {c2_bars_thr} -> {'✓' if cond3 else '✗'}")
        print(f"  条件4: signed_vwap_dev_pct ({row['signed_vwap_dev_pct']:.4f}) <= {c2_vwap_thr} -> {'✓' if cond4 else '✗'}")
        print(f"  总体: {'符合 C2 策略' if (cond1 and cond2 and cond3 and cond4) else '不符合 C2 策略'}")
    
    # 输出最近30天的关键数据
    print(f"\n【最近30天数据】")
    recent = merged.tail(30)[['close', 'dsa_pivot_high', 'dsa_pivot_low', 'dsa_pivot_pos_01', 
                               'DSA_DIR', 'signed_vwap_dev_pct', 'rope_dir', 
                               'bars_since_dir_change', 'last_pivot_type']]
    print(recent.to_string())
    
    # 输出 DSA 标签
    print(f"\n【DSA 回看确认标签】")
    for label in dsa_labels[-10:]:
        print(f"  {label['x'].strftime('%Y-%m-%d')}: {label['text']} @ {label['y']:.2f} (dir={label['dir']})")
    
    # 保存 CSV
    output_file = "/tmp/huadian_600396_test.csv"
    merged.to_csv(output_file, encoding="utf-8-sig")
    print(f"\n数据已保存到: {output_file}")


if __name__ == "__main__":
    test_huadian()
