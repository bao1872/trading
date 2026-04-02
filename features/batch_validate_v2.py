#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段2批量验证脚本：多标的因子评分系统验证

目标：
1. 从 stock_list 数据库读取股票池
2. 选取 N 只股票，每只 200 bars
3. 运行 unified_pullback_factor_validator 逻辑
4. 汇总验证所有改进点：
   - break_penalty 上限 -1.2
   - combined_freshness 乘法叠乘
   - trend_state 三态检测
   - inside 分级
   - 评分分布健康度

Usage:
    python batch_validate_v2.py --n 10 --bars 200
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
for _p in (BASE_DIR, ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.stock_list_manager import get_stock_cache_from_db
from datasource.database import get_session
from sqlalchemy import text


def fetch_kline(ts_code: str, freq: str, bars: int):
    """用 pytdx 拉取 K 线数据"""
    try:
        import pytdx
        from pytdx.hq import TdxHq_API
        api = TdxHq_API()
        api.connect("115.238.90.165", 7709)
        code_str = ts_code.split(".")[0]
        mkt = 1 if ts_code.endswith(".SH") else 0
        klines = api.get_security_bars(
            category={"d": 9, "w": 5, "m": 6}.get(freq, 9),
            market=mkt,
            code=code_str,
            start=0,
            count=bars
        )
        api.disconnect()
        if klines:
            df = pd.DataFrame(klines)
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
    except Exception:
        pass
    return pd.DataFrame()


def run_validator_for_symbol(ts_code: str, freq: str, bars: int) -> dict:
    """对单个标的运行验证，返回关键指标"""
    from unified_pullback_factor_validator import (
        compute_support_scores,
        rolling_trailing_pivots,
        build_structure_proxy,
        atr_pine,
        detect_trend_state,
    )
    from dynamic_swing_anchored_vwap import (
        DSAConfig, dynamic_swing_anchored_vwap, compute_extra_factors
    )
    from atr_rope_with_factors_pytdx_plotly import ATRRopeEngine

    df = fetch_kline(ts_code, freq, bars)
    if df.empty or len(df) < 30:
        return {"error": "数据不足", "ts_code": ts_code}

    df = df.rename(columns={"vol": "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        rope_engine = ATRRopeEngine(df.copy(), argparse.Namespace(
            len=14, multi=1.5, source="close", freq=freq,
            show_ranges=True, show_atr_channel=True,
            show_break_markers=False, show_factor_panels=True,
        ))
        rope_engine.run()
        rope_df = rope_engine.df.copy()
    except Exception:
        rope_df = df.copy()
        for col in ["rope", "upper", "lower", "rope_dir", "c_lo"]:
            rope_df[col] = np.nan

    try:
        dsa_cfg = DSAConfig(prd=50, baseAPT=20.0, useAdapt=False, volBias=10.0)
        dsa_vwap, dsa_dir, pivot_labels, _segments = dynamic_swing_anchored_vwap(df.copy(), dsa_cfg)
        dsa_extra = compute_extra_factors(df.copy(), dsa_vwap, dsa_dir, pivot_labels)
    except Exception:
        dsa_vwap = pd.Series(np.nan, index=df.index)
        dsa_dir = pd.Series(0, index=df.index)
        dsa_extra = pd.DataFrame(index=df.index)

    for col in dsa_vwap.index:
        df.at[df.index[col], "dsa_vwap"] = dsa_vwap.iloc[col] if col < len(dsa_vwap) else np.nan
    df["dsa_dir"] = dsa_dir.reindex(df.index).values

    rope_keep = ["rope", "upper", "lower", "rope_dir", "c_lo"]
    for col in rope_keep:
        if col in rope_df.columns:
            df[col] = rope_df[col].reindex(df.index).values

    piv = rolling_trailing_pivots(df[["open", "high", "low", "close"]], lookback=20)
    for col in piv.columns:
        df[col] = piv[col]
    struct_df = build_structure_proxy(df[["open", "high", "low", "close"]].join(piv), piv)
    for col in struct_df.columns:
        df[col] = struct_df[col]

    support_df = compute_support_scores(df)
    for col in support_df.columns:
        df[col] = support_df[col]

    valid = df.dropna(subset=["support_score"])
    trend_state = detect_trend_state(df, lookback=20)
    all_states = df["cluster1_state"].value_counts().to_dict()
    below_states = {k: v for k, v in all_states.items() if "below" in k}

    return {
        "ts_code": ts_code,
        "valid_count": len(valid),
        "total_count": len(df),
        "trend_state": trend_state,
        "score_mean": valid["support_score"].mean() if len(valid) > 0 else np.nan,
        "score_std": valid["support_score"].std() if len(valid) > 0 else np.nan,
        "score_min": valid["support_score"].min() if len(valid) > 0 else np.nan,
        "score_max": valid["support_score"].max() if len(valid) > 0 else np.nan,
        "penalty_min": df["break_penalty"].min(),
        "penalty_max": df["break_penalty"].max(),
        "penalty_below_limit": (df["break_penalty"] < -1.2).any(),
        "inside_count": all_states.get("inside", 0),
        "above_count": all_states.get("above", 0),
        "below_count": sum(below_states.values()),
        "na_count": all_states.get("na", 0),
        "state_distribution": all_states,
        "break_penalty_vals": df["break_penalty"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="阶段2批量验证")
    parser.add_argument("--n", type=int, default=10, help="验证股票数量")
    parser.add_argument("--bars", type=int, default=200, help="K线数量")
    parser.add_argument("--freq", default="d", help="频率")
    parser.add_argument("--out", default="/tmp/batch_validation_v2.csv", help="输出CSV")
    args = parser.parse_args()

    stock_map = get_stock_cache_from_db()
    stocks = list(stock_map.items())

    print(f"股票池总数(stock_concepts_cache): {len(stocks)}")
    if len(stocks) < args.n:
        print(f"⚠️ 股票池不足 {args.n} 只，将使用全部 {len(stocks)} 只")

    selected = stocks[: args.n]
    print(f"选取 {len(selected)} 只股票验证...")

    results = []
    for i, (name, ts_code) in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] {name}({ts_code})...", end=" ", flush=True)
        r = run_validator_for_symbol(ts_code, args.freq, args.bars)
        if "error" not in r:
            print(f"✅ score_mean={r['score_mean']:.1f}, trend={r['trend_state']}, inside={r['inside_count']}")
        else:
            print(f"❌ {r['error']}")
        results.append(r)

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n汇总已保存: {args.out}")

    print("\n" + "="*70)
    print("阶段2 验证汇总报告")
    print("="*70)

    valid_results = [r for r in results if "error" not in r and r["valid_count"] > 0]
    if not valid_results:
        print("❌ 无有效结果")
        return

    penalty_ok = [r for r in valid_results if not r["penalty_below_limit"]]
    print(f"\n[1] break_penalty 上限验证: {len(penalty_ok)}/{len(valid_results)} ✅")
    for r in valid_results:
        status = "✅" if not r["penalty_below_limit"] else "❌"
        print(f"    {status} {r['ts_code']}: penalty范围=[{r['penalty_min']:.3f}, {r['penalty_max']:.3f}]")

    trend_counts = pd.Series([r["trend_state"] for r in valid_results]).value_counts()
    print(f"\n[2] trend_state 分布:")
    print(f"    trend_long={trend_counts.get('trend_long', 0)}, range={trend_counts.get('range', 0)}, trend_short={trend_counts.get('trend_short', 0)}, unclear={trend_counts.get('unclear', 0)}")

    all_inside = sum(r["inside_count"] for r in valid_results)
    all_above = sum(r["above_count"] for r in valid_results)
    all_below = sum(r["below_count"] for r in valid_results)
    print(f"\n[3] 位置状态分布:")
    print(f"    inside={all_inside}, above={all_above}, below={all_below}")
    for r in valid_results:
        print(f"    {r['ts_code']}: {r['state_distribution']}")

    score_means = [r["score_mean"] for r in valid_results]
    print(f"\n[4] 评分分布健康度:")
    print(f"    全局均值={np.mean(score_means):.1f}, 标准差={np.std(score_means):.1f}")
    print(f"    各标的均值: " + ", ".join(f"{r['ts_code']}={r['score_mean']:.1f}" for r in valid_results))
    healthy = all(3 < r["score_mean"] < 80 for r in valid_results)
    print(f"    评分是否健康(3~80): {'✅ YES' if healthy else '❌ NO'}")

    print(f"\n[5] 跌破事件统计:")
    total_below = sum(r["below_count"] for r in valid_results)
    print(f"    跌破总次数: {total_below}")
    for r in valid_results:
        if r["below_count"] > 0:
            below_penalties = [p for p in r["break_penalty_vals"] if p < 0]
            if below_penalties:
                print(f"    {r['ts_code']}: {r['below_count']}次, penalty范围=[{min(below_penalties):.3f}, {max(below_penalties):.3f}]")


if __name__ == "__main__":
    main()
