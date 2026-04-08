# -*- coding: utf-8 -*-
"""
华电辽能(600396) DSA 特征修改后深度分析报告

Purpose
- 单独测试 600396 在修改后的 DSA 特征下的表现
- 分析 C2 策略信号触发情况
- 对比修改前后的差异

How to Run
    python test_huadian_analysis.py
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "features"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtrader"))

from features.merged_dsa_atr_rope_bb_factors import (
    DSAConfig,
    RopeConfig,
    compute_atr_rope,
    compute_bollinger,
    compute_dsa,
    fetch_kline_pytdx,
)


def analyze_huadian():
    symbol = "600396"
    print("=" * 80)
    print(f"华电辽能({symbol}) DSA 特征修改后深度分析")
    print("=" * 80)

    # 获取数据
    df = fetch_kline_pytdx(symbol, "d", 500)
    print(f"\n【数据概况】")
    print(f"  数据范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  总条数: {len(df)}")
    print(f"  当前价格: {df['close'].iloc[-1]:.2f}")

    # 计算因子
    dsa_df, dsa_labels, dsa_segments = compute_dsa(df, DSAConfig(prd=50, base_apt=20.0))
    rope_df = compute_atr_rope(df, RopeConfig(length=14, multi=1.5))
    bb_df = compute_bollinger(df, length=20, mult=2.0, pct_lookback=120)

    # 合并数据
    merged = pd.concat([df, dsa_df, rope_df.drop(columns=df.columns, errors="ignore"), bb_df], axis=1)

    # 添加未来收益
    for w in [5, 10, 20, 40, 60]:
        merged[f"ret_{w}"] = merged["close"].shift(-w) / merged["close"] - 1
        merged[f"max_dd_{w}"] = merged["close"].rolling(window=w, min_periods=1).apply(
            lambda x: (x.min() - x.iloc[0]) / x.iloc[0], raw=False
        ).shift(-w)

    # 分析关键日期 2026-03-02
    target_date = "2026-03-02"
    print(f"\n【关键日期分析: {target_date}】")

    if target_date in merged.index.strftime("%Y-%m-%d"):
        row = merged.loc[merged.index.strftime("%Y-%m-%d") == target_date].iloc[0]

        print(f"  收盘价: {row['close']:.2f}")
        print(f"  dsa_pivot_high: {row['dsa_pivot_high']:.2f} (回看确认的 HH)")
        print(f"  dsa_pivot_low: {row['dsa_pivot_low']:.2f} (回看确认的 HL)")
        print(f"  dsa_pivot_pos_01: {row['dsa_pivot_pos_01']:.4f}")
        print(f"  DSA_DIR: {row['DSA_DIR']}")
        print(f"  signed_vwap_dev_pct: {row['signed_vwap_dev_pct']:.4f}%")
        print(f"  rope_dir: {row['rope_dir']}")
        print(f"  bars_since_dir_change: {row['bars_since_dir_change']}")
        print(f"  last_pivot_type: {row['last_pivot_type']}")

        # C2 策略条件检查
        print(f"\n  【C2 策略条件检查】(dsa_thr=0.35, bars_thr=3, vwap_thr=-1.0)")
        c2_dsa_thr = 0.35
        c2_bars_thr = 3
        c2_vwap_thr = -1.0

        cond1 = row['dsa_pivot_pos_01'] <= c2_dsa_thr
        cond2 = row['rope_dir'] == 1
        cond3 = row['bars_since_dir_change'] <= c2_bars_thr
        cond4 = row['signed_vwap_dev_pct'] <= c2_vwap_thr

        print(f"    条件1: dsa_pivot_pos_01 ({row['dsa_pivot_pos_01']:.4f}) <= {c2_dsa_thr} -> {'✓ 通过' if cond1 else '✗ 失败'}")
        print(f"    条件2: rope_dir ({row['rope_dir']:.0f}) == 1 -> {'✓ 通过' if cond2 else '✗ 失败'}")
        print(f"    条件3: bars_since_dir_change ({row['bars_since_dir_change']:.0f}) <= {c2_bars_thr} -> {'✓ 通过' if cond3 else '✗ 失败'}")
        print(f"    条件4: signed_vwap_dev_pct ({row['signed_vwap_dev_pct']:.4f}%) <= {c2_vwap_thr} -> {'✓ 通过' if cond4 else '✗ 失败'}")
        print(f"    总体: {'✓✓✓ 符合 C2 策略 ✓✓✓' if (cond1 and cond2 and cond3 and cond4) else '✗✗✗ 不符合 C2 策略 ✗✗✗'}")

        # 未来收益
        print(f"\n  【未来收益】(从 {target_date} 起)")
        for w in [5, 10, 20, 40, 60]:
            ret_col = f"ret_{w}"
            if ret_col in row and pd.notna(row[ret_col]):
                print(f"    ret_{w}: {row[ret_col]*100:+.2f}%")

    # DSA 回看确认枢轴点列表
    print(f"\n【DSA 回看确认枢轴点列表】")
    print(f"{'日期':<12} {'类型':<6} {'价格':<8} {'方向':<6}")
    print("-" * 40)
    for label in dsa_labels:
        date_str = label['x'].strftime('%Y-%m-%d')
        print(f"{date_str:<12} {label['text']:<6} {label['y']:<8.2f} {'上升' if label['dir'] > 0 else '下降':<6}")

    # 分析最近 3 个月的数据
    print(f"\n【最近 3 个月数据】")
    recent_3m = merged.tail(90)[['close', 'dsa_pivot_high', 'dsa_pivot_low', 'dsa_pivot_pos_01',
                                  'DSA_DIR', 'signed_vwap_dev_pct', 'rope_dir',
                                  'bars_since_dir_change', 'last_pivot_type']]
    print(recent_3m.to_string())

    # C2 策略历史信号分析
    print(f"\n【C2 策略历史信号分析】(dsa_thr=0.35, bars_thr=3, vwap_thr=-1.0)")

    c2_mask = (
        (merged['dsa_pivot_pos_01'] <= 0.35) &
        (merged['rope_dir'] == 1) &
        (merged['bars_since_dir_change'] <= 3) &
        (merged['signed_vwap_dev_pct'] <= -1.0)
    )

    c2_signals = merged[c2_mask].copy()

    if len(c2_signals) > 0:
        print(f"  历史信号总数: {len(c2_signals)}")
        print(f"\n  {'日期':<12} {'收盘价':<8} {'dsa_pos':<10} {'ret_5':<10} {'ret_20':<10}")
        print("  " + "-" * 60)

        for idx, row in c2_signals.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            ret_5 = row.get('ret_5', np.nan)
            ret_20 = row.get('ret_20', np.nan)
            ret_5_str = f"{ret_5*100:+.2f}%" if pd.notna(ret_5) else "N/A"
            ret_20_str = f"{ret_20*100:+.2f}%" if pd.notna(ret_20) else "N/A"
            print(f"  {date_str:<12} {row['close']:<8.2f} {row['dsa_pivot_pos_01']:<10.4f} {ret_5_str:<10} {ret_20_str:<10}")

        # 统计收益
        print(f"\n  【信号收益统计】")
        for w in [5, 10, 20]:
            ret_col = f"ret_{w}"
            if ret_col in c2_signals.columns:
                valid_rets = c2_signals[ret_col].dropna()
                if len(valid_rets) > 0:
                    avg_ret = valid_rets.mean()
                    win_rate = (valid_rets > 0).mean()
                    print(f"    ret_{w}: 均值={avg_ret*100:+.2f}%, 胜率={win_rate*100:.1f}%")
    else:
        print(f"  历史无 C2 策略信号")

    # 保存详细数据
    output_file = "/tmp/huadian_600396_detailed.csv"
    merged.to_csv(output_file, encoding="utf-8-sig")
    print(f"\n【数据已保存】")
    print(f"  文件: {output_file}")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == "__main__":
    analyze_huadian()
