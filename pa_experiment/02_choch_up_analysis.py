#!/usr/bin/env python3
"""
CHoCH↑/↓ 场景专项分析

Purpose: 针对 CHoCH↑ 和 CHoCH↓ 事件，分析趋势状态区分力、成交量zscore与收益率关系
Inputs: pa_selection 表, stock_k_data 表
Outputs: pa_experiment/results/ 下的 CSV 文件
How to Run:
    python pa_experiment/02_choch_up_analysis.py
    python pa_experiment/02_choch_up_analysis.py --start-date 2024-01-01
Examples:
    python pa_experiment/02_choch_up_analysis.py
    python pa_experiment/02_choch_up_analysis.py --start-date 2025-01-01
Side Effects: 只写 CSV 文件，不写数据库
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from tabulate import tabulate

try:
    from config import DATABASE_URL
except ImportError:
    DATABASE_URL = os.environ.get("DATABASE_URL", "")

engine = create_engine(DATABASE_URL)

from stop_experiment.eval.filter_quality_evaluator import compute_future_metrics

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [3, 5, 10, 20]

# 成交量 zscore 5档分箱
VOL_ZSCORE_BINS = [
    ("<-1", lambda x: x < -1),
    ("-1~0", lambda x: (x >= -1) & (x < 0)),
    ("0~1", lambda x: (x >= 0) & (x < 1)),
    ("1~2", lambda x: (x >= 1) & (x < 2)),
    (">2", lambda x: x >= 2),
]


def load_and_enrich(start_date, end_date):
    """加载 CHoCH↑ 选股结果并计算未来收益和成交量 zscore"""
    # 加载选股结果（CHoCH↑ 和 CHoCH↓）
    sql = text("""
        SELECT selection_date, signal_date, ts_code, stock_name,
               evt_choch_up, evt_choch_down, pat_trend_state, change_pct, vol_zscore
        FROM pa_selection
        WHERE (evt_choch_up = True OR evt_choch_down = True)
          AND selection_date >= :start AND selection_date <= :end
        ORDER BY selection_date
    """)
    pa_df = pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})
    
    # 标记事件类型
    pa_df["event_type"] = ""
    pa_df.loc[pa_df["evt_choch_up"] == True, "event_type"] = "CHoCH↑"
    pa_df.loc[pa_df["evt_choch_down"] == True, "event_type"] = "CHoCH↓"
    # 如果同时触发两个事件，保留两个标签
    both = (pa_df["evt_choch_up"] == True) & (pa_df["evt_choch_down"] == True)
    pa_df.loc[both, "event_type"] = "CHoCH↑↓"
    
    print(f"CHoCH 记录: {len(pa_df)} (↑={int(pa_df['evt_choch_up'].sum())}, ↓={int(pa_df['evt_choch_down'].sum())})")

    if pa_df.empty:
        return pa_df

    # 加载价格 pivot
    print("加载价格数据...")
    ts_codes = pa_df["ts_code"].str[:6].unique().tolist()
    batch_size = 500
    all_dfs = []
    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i:i+batch_size]
        or_clause = " OR ".join([f"(ts_code = '{c}' OR ts_code = '{c}.SH' OR ts_code = '{c}.SZ')" for c in batch])
        sql = text(f"""
            SELECT bar_time, ts_code, open, high, low, close, volume
            FROM stock_k_data WHERE freq = 'd' AND ({or_clause})
            AND DATE(bar_time) >= :start AND DATE(bar_time) <= :end
            ORDER BY bar_time, ts_code
        """)
        all_dfs.append(pd.read_sql(sql, engine, params={"start": start_date, "end": end_date}))

    df = pd.concat(all_dfs, ignore_index=True)
    df["raw_code"] = df["ts_code"].str[:6]
    df["bar_time"] = pd.to_datetime(df["bar_time"]).dt.normalize()

    pivot = {}
    for col in ["close", "high", "low", "volume"]:
        pivot[col] = df.pivot_table(index="bar_time", columns="raw_code", values=col, aggfunc="last")
    print(f"价格数据: {pivot['close'].shape}")

    # 计算未来收益
    print("计算未来收益...")
    future_metrics = compute_future_metrics(pivot, HORIZONS)

    # 计算成交量 zscore
    vol_df = pivot["volume"].astype(float)
    vol_zscores = {}
    for w in [5, 10, 20]:
        mu = vol_df.rolling(w, min_periods=w).mean()
        sd = vol_df.rolling(w, min_periods=w).std(ddof=0)
        vol_zscores[f"vol_zscore_{w}"] = (vol_df - mu) / sd

    # 向量化 lookup
    pa_df["raw_code"] = pa_df["ts_code"].str[:6]
    pa_df["obs_date"] = pd.to_datetime(pa_df["selection_date"]).dt.normalize()

    for N in HORIZONS:
        for mn in ["return", "mfe", "mae"]:
            metric_df = future_metrics[N][mn]
            dates_arr = pa_df["obs_date"].values
            codes_arr = pa_df["raw_code"].values
            date_idx = metric_df.index.get_indexer(pd.to_datetime(dates_arr))
            code_idx = metric_df.columns.get_indexer(codes_arr)
            valid = (date_idx >= 0) & (code_idx >= 0)
            result = np.full(len(pa_df), np.nan)
            if valid.any():
                result[valid] = metric_df.values[date_idx[valid], code_idx[valid]]
            pa_df[f"{mn}_{N}"] = result

    for zscore_col, zscore_df in vol_zscores.items():
        dates_arr = pa_df["obs_date"].values
        codes_arr = pa_df["raw_code"].values
        date_idx = zscore_df.index.get_indexer(pd.to_datetime(dates_arr))
        code_idx = zscore_df.columns.get_indexer(codes_arr)
        valid = (date_idx >= 0) & (code_idx >= 0)
        result = np.full(len(pa_df), np.nan)
        if valid.any():
            result[valid] = zscore_df.values[date_idx[valid], code_idx[valid]]
        pa_df[zscore_col] = result

    return pa_df


def _stats(sub, h):
    col = f"return_{h}"
    v = sub[col].dropna()
    if len(v) == 0:
        return None
    return {
        "样本": len(v), "均值": f"{v.mean()*100:+.2f}%", "中位数": f"{v.median()*100:+.2f}%",
        "标准差": f"{v.std()*100:.2f}%", "偏度": f"{v.skew():.2f}",
        "胜率": f"{(v>0).mean()*100:.1f}%",
        "P25": f"{v.quantile(0.25)*100:+.2f}%", "P75": f"{v.quantile(0.75)*100:+.2f}%",
        "MFE": f"{sub[f'mfe_{h}'].dropna().mean()*100:+.2f}%",
        "MAE": f"{sub[f'mae_{h}'].dropna().mean()*100:+.2f}%",
        "均值r": v.mean(), "胜率r": (v > 0).mean(), "标准差r": v.std(),
    }


def analyze_trend(df):
    """趋势状态区分力分析（按事件类型分组）"""
    rows = []
    for evt in df["event_type"].unique():
        evt_df = df[df["event_type"] == evt]
        for trend_val, trend_name in [(1, "趋势↑"), (-1, "趋势↓")]:
            sub = evt_df[evt_df["pat_trend_state"] == trend_val]
            if sub.empty:
                continue
            for h in HORIZONS:
                s = _stats(sub, h)
                if s:
                    row = {"事件": evt, "趋势": trend_name, "持有期": f"{h}日"}
                    row.update(s)
                    rows.append(row)
    return pd.DataFrame(rows)


def analyze_vol_zscore(df, zscore_col="vol_zscore_20"):
    """成交量 zscore 5档分析（按事件类型分组）"""
    valid = df[zscore_col].notna()
    df_valid = df.loc[valid]
    rows = []
    for evt in df_valid["event_type"].unique():
        evt_df = df_valid[df_valid["event_type"] == evt]
        for bin_name, bin_cond in VOL_ZSCORE_BINS:
            mask = bin_cond(evt_df[zscore_col])
            sub = evt_df.loc[mask]
            if sub.empty:
                continue
            for h in HORIZONS:
                s = _stats(sub, h)
                if s:
                    row = {"事件": evt, "zscore档": bin_name, "zscore列": zscore_col, "持有期": f"{h}日"}
                    row.update(s)
                    rows.append(row)
    return pd.DataFrame(rows)


def analyze_trend_vol_cross(df, zscore_col="vol_zscore_20"):
    """趋势 × 成交量 交叉分析（按事件类型分组）"""
    valid = df[zscore_col].notna()
    df_valid = df.loc[valid]
    rows = []
    for evt in df_valid["event_type"].unique():
        evt_df = df_valid[df_valid["event_type"] == evt]
        for trend_val, trend_name in [(1, "趋势↑"), (-1, "趋势↓")]:
            trend_sub = evt_df[evt_df["pat_trend_state"] == trend_val]
            for bin_name, bin_cond in VOL_ZSCORE_BINS:
                mask = bin_cond(trend_sub[zscore_col])
                sub = trend_sub.loc[mask]
                if sub.empty:
                    continue
                for h in HORIZONS:
                    s = _stats(sub, h)
                    if s:
                        row = {"事件": evt, "趋势": trend_name, "zscore档": bin_name, "持有期": f"{h}日"}
                        row.update(s)
                        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="CHoCH↑ 场景专项分析")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-05-19")
    args = parser.parse_args()

    print("=" * 100)
    print("CHoCH↑/↓ 场景专项分析")
    print(f"  区间: {args.start_date} ~ {args.end_date}")
    print("=" * 100)

    # 加载并丰富数据
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    from datetime import timedelta
    extended_end = (end_dt + timedelta(days=35)).strftime("%Y-%m-%d")
    df = load_and_enrich(args.start_date, extended_end)

    if df.empty:
        print("无数据")
        return

    # 有效未来收益比例
    for h in HORIZONS:
        valid_rate = df[f"return_{h}"].notna().mean() * 100
        print(f"  {h}日收益: {valid_rate:.1f}% 有效")

    # 1. 趋势状态区分力
    print("\n" + "=" * 100)
    print("一、趋势状态区分力")
    print("=" * 100)
    trend_stats = analyze_trend(df)
    if not trend_stats.empty:
        print(tabulate(trend_stats[["事件", "趋势", "持有期", "样本", "均值", "中位数", "标准差", "偏度", "胜率", "P25", "P75", "MFE", "MAE"]],
                       headers="keys", tablefmt="grid", showindex=False))

    # 2. 成交量 zscore 分档（三个窗口）
    for w in [5, 10, 20]:
        zscore_col = f"vol_zscore_{w}"
        print(f"\n{'='*100}")
        print(f"二、成交量 zscore ({w}日) 分档收益")
        print("=" * 100)
        vol_stats = analyze_vol_zscore(df, zscore_col)
        if not vol_stats.empty:
            print(tabulate(vol_stats[["事件", "zscore档", "持有期", "样本", "均值", "中位数", "标准差", "偏度", "胜率", "P25", "P75"]],
                           headers="keys", tablefmt="grid", showindex=False))
            vol_stats.to_csv(RESULTS_DIR / f"choch_up_vol_zscore{w}_stats.csv", index=False, encoding="utf-8-sig")

    # 3. 趋势 × 成交量交叉（5日持有期汇总表，按事件分组）
    print(f"\n{'='*100}")
    print("三、趋势 × 成交量交叉分析（5日持有期）")
    print("=" * 100)
    cross_stats = analyze_trend_vol_cross(df, "vol_zscore_20")
    if not cross_stats.empty:
        cross_5d = cross_stats[cross_stats["持有期"] == "5日"]
        for evt in cross_5d["事件"].unique():
            evt_cross = cross_5d[cross_5d["事件"] == evt]
            print(f"\n--- {evt} ---")
            # 均值矩阵
            pivot_ret = evt_cross.pivot_table(index="趋势", columns="zscore档", values="均值r", aggfunc="first")
            bin_order = ["<-1", "-1~0", "0~1", "1~2", ">2"]
            pivot_ret = pivot_ret.reindex(columns=[c for c in bin_order if c in pivot_ret.columns])
            print("\n均值收益矩阵:")
            print(tabulate(pivot_ret.map(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "-"), headers="keys", tablefmt="grid"))

            # 胜率矩阵
            pivot_wr = evt_cross.pivot_table(index="趋势", columns="zscore档", values="胜率r", aggfunc="first")
            pivot_wr = pivot_wr.reindex(columns=[c for c in bin_order if c in pivot_wr.columns])
            print("\n胜率矩阵:")
            print(tabulate(pivot_wr.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"), headers="keys", tablefmt="grid"))

            # 样本量矩阵
            pivot_n = evt_cross.pivot_table(index="趋势", columns="zscore档", values="样本", aggfunc="first")
            pivot_n = pivot_n.reindex(columns=[c for c in bin_order if c in pivot_n.columns])
            print("\n样本量矩阵:")
            print(tabulate(pivot_n.map(lambda x: f"{int(x)}" if pd.notna(x) else "-"), headers="keys", tablefmt="grid"))

            # 标准差矩阵
            pivot_std = evt_cross.pivot_table(index="趋势", columns="zscore档", values="标准差r", aggfunc="first")
            pivot_std = pivot_std.reindex(columns=[c for c in bin_order if c in pivot_std.columns])
            print("\n收益标准差矩阵:")
            print(tabulate(pivot_std.map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-"), headers="keys", tablefmt="grid"))

    # 4. 全持有期交叉汇总
    print(f"\n{'='*100}")
    print("四、趋势 × 成交量交叉（全持有期均值收益）")
    print("=" * 100)
    if not cross_stats.empty:
        for evt in cross_stats["事件"].unique():
            print(f"\n--- {evt} ---")
            for h in HORIZONS:
                cross_h = cross_stats[(cross_stats["持有期"] == f"{h}日") & (cross_stats["事件"] == evt)]
                if not cross_h.empty:
                    pivot_h = cross_h.pivot_table(index="趋势", columns="zscore档", values="均值r", aggfunc="first")
                    bin_order = ["<-1", "-1~0", "0~1", "1~2", ">2"]
                    pivot_h = pivot_h.reindex(columns=[c for c in bin_order if c in pivot_h.columns])
                    print(f"\n{h}日持有期:")
                    print(tabulate(pivot_h.map(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "-"), headers="keys", tablefmt="grid"))

    # 保存
    trend_stats.to_csv(RESULTS_DIR / "choch_up_trend_stats.csv", index=False, encoding="utf-8-sig")
    cross_stats.to_csv(RESULTS_DIR / "choch_up_trend_vol_cross.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到 {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
