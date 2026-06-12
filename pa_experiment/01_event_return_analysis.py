#!/usr/bin/env python3
"""
PA事件收益率与成交量关系分析

Purpose: 分析PA选股结果中8类事件的3/5/10/20日持有期收益率分布，
         以及事件日成交量zscore(5/10/20日)与收益率的关系
Inputs: pa_selection 表, stock_k_data 表
Outputs: pa_experiment/results/ 下的 CSV 文件
How to Run:
    python pa_experiment/01_event_return_analysis.py
    python pa_experiment/01_event_return_analysis.py --start-date 2024-01-01
    python pa_experiment/01_event_return_analysis.py --start-date 2023-01-01 --end-date 2026-05-19
Examples:
    python pa_experiment/01_event_return_analysis.py
    python pa_experiment/01_event_return_analysis.py --start-date 2025-01-01
Side Effects: 只写 CSV 文件到 pa_experiment/results/，不写数据库

================================================================================
【分析维度】

1. 按事件类型分组：8类事件各自的收益率分布（均值/中位数/胜率/MFE/MAE）
2. 按成交量zscore分组：低(<0)/中(0~1)/高(>1) 三组，看不同成交量水平下收益率差异
3. 交叉分析：事件类型 × 成交量水平 的收益率矩阵

【核心计算】
- 未来收益：引用 stop_experiment/eval/filter_quality_evaluator.py 的 compute_future_metrics()
- 成交量zscore：向量化计算 vol_zscore_5/10/20
================================================================================
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

# 引用 SSOT 的未来收益计算函数
from stop_experiment.eval.filter_quality_evaluator import compute_future_metrics

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [3, 5, 10, 20]

EVENT_COLS = [
    ("evt_choch_up", "CHoCH↑"),
    ("evt_bos_up", "BoS↑"),
    ("evt_choch_down", "CHoCH↓"),
    ("evt_bos_down", "BoS↓"),
    ("evt_upper_liq_sweep", "扫高收回"),
    ("evt_lower_liq_sweep", "扫低收回"),
    ("evt_upper_sweep_fail_up", "扫高失败↑"),
    ("evt_lower_sweep_fail_down", "扫低失败↓"),
]

VOL_ZSCORE_BINS = [
    ("低(<0)", lambda x: x < 0),
    ("中(0~1)", lambda x: (x >= 0) & (x < 1)),
    ("高(>1)", lambda x: x >= 1),
]


def load_pa_selection(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """从 pa_selection 表加载选股结果"""
    conditions = []
    params = {}
    if start_date:
        conditions.append("selection_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append("selection_date <= :end_date")
        params["end_date"] = end_date

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    sql = text(f"""
        SELECT selection_date, signal_date, ts_code, stock_name,
               evt_choch_up, evt_bos_up, evt_choch_down, evt_bos_down,
               evt_upper_liq_sweep, evt_lower_liq_sweep,
               evt_upper_sweep_fail_up, evt_lower_sweep_fail_down,
               pat_trend_state, change_pct, vol_zscore
        FROM pa_selection
        WHERE {where_clause}
        ORDER BY selection_date, ts_code
    """)
    df = pd.read_sql(sql, engine, params=params)
    return df


def load_price_pivot(ts_codes: list, start_date: str, end_date: str) -> dict:
    """
    从 stock_k_data 加载价格数据并 pivot 成宽表

    Returns: dict with keys "close", "high", "low", "volume"
             每个值是 DataFrame(index=日期, columns=6位股票代码)
    """
    # 将 ts_code 转为6位纯数字
    raw_codes = [c[:6] if len(c) > 6 else c for c in ts_codes]
    unique_codes = list(set(raw_codes))

    # 构建查询条件（同时匹配 .SH 和 .SZ 后缀）
    code_conditions = []
    for code in unique_codes:
        code_conditions.append(f"ts_code = '{code}' OR ts_code = '{code}.SH' OR ts_code = '{code}.SZ'")

    # 分批查询避免 SQL 过长
    batch_size = 500
    all_dfs = []

    for i in range(0, len(code_conditions), batch_size):
        batch = code_conditions[i:i+batch_size]
        or_clause = " OR ".join([f"({c})" for c in batch])
        sql = text(f"""
            SELECT bar_time, ts_code, open, high, low, close, volume
            FROM stock_k_data
            WHERE freq = 'd' AND ({or_clause})
            AND DATE(bar_time) >= :start_date AND DATE(bar_time) <= :end_date
            ORDER BY bar_time, ts_code
        """)
        # end_date 需要往后推 max(HORIZONS) 天以计算未来收益
        params = {"start_date": start_date, "end_date": end_date}
        batch_df = pd.read_sql(sql, engine, params=params)
        all_dfs.append(batch_df)

    if not all_dfs:
        return {}

    df = pd.concat(all_dfs, ignore_index=True)
    if df.empty:
        return {}

    # ts_code 统一为6位纯数字
    df["raw_code"] = df["ts_code"].str[:6]
    df["bar_time"] = pd.to_datetime(df["bar_time"]).dt.normalize()

    # pivot
    pivot = {}
    for col in ["close", "high", "low", "volume"]:
        pivot[col] = df.pivot_table(index="bar_time", columns="raw_code", values=col, aggfunc="last")

    return pivot


def compute_vol_zscores(price_pivot: dict, windows: list = [5, 10, 20]) -> dict:
    """向量化计算成交量 zscore"""
    vol_df = price_pivot["volume"].astype(float)
    result = {}
    for w in windows:
        mu = vol_df.rolling(w, min_periods=w).mean()
        sd = vol_df.rolling(w, min_periods=w).std(ddof=0)
        result[f"vol_zscore_{w}"] = (vol_df - mu) / sd
    return result


def enrich_pa_data(pa_df: pd.DataFrame, price_pivot: dict, vol_zscores: dict) -> pd.DataFrame:
    """为选股结果添加未来收益和成交量 zscore"""
    pa_df = pa_df.copy()

    # 计算未来收益
    future_metrics = compute_future_metrics(price_pivot, HORIZONS)

    # 准备 lookup
    pa_df["raw_code"] = pa_df["ts_code"].str[:6]
    pa_df["obs_date"] = pd.to_datetime(pa_df["selection_date"]).dt.normalize()

    # 向量化 lookup 未来收益
    for N in HORIZONS:
        for metric_name in ["return", "mfe", "mae"]:
            metric_df = future_metrics[N][metric_name]
            col_name = f"{metric_name}_{N}"
            dates_arr = pa_df["obs_date"].values
            codes_arr = pa_df["raw_code"].values

            date_idx = metric_df.index.get_indexer(pd.to_datetime(dates_arr))
            code_idx = metric_df.columns.get_indexer(codes_arr)
            valid = (date_idx >= 0) & (code_idx >= 0)
            result = np.full(len(pa_df), np.nan)
            if valid.any():
                result[valid] = metric_df.values[date_idx[valid], code_idx[valid]]
            pa_df[col_name] = result

    # 向量化 lookup 成交量 zscore
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


def _group_stats(group_df: pd.DataFrame, horizon: int) -> dict:
    """计算单组在指定 horizon 下的统计指标"""
    ret_col = f"return_{horizon}"
    valid = group_df[ret_col].notna()
    n_valid = valid.sum()
    if n_valid == 0:
        return None

    ret = group_df.loc[valid, ret_col]
    mfe = group_df.loc[valid, f"mfe_{horizon}"]
    mae = group_df.loc[valid, f"mae_{horizon}"]

    return {
        "样本数": int(n_valid),
        "均值": f"{ret.mean()*100:.2f}%",
        "中位数": f"{ret.median()*100:.2f}%",
        "胜率": f"{(ret > 0).mean()*100:.1f}%",
        "MFE均值": f"{mfe.mean()*100:.2f}%",
        "MAE均值": f"{mae.mean()*100:.2f}%",
        "均值(raw)": ret.mean(),
        "中位数(raw)": ret.median(),
        "胜率(raw)": (ret > 0).mean(),
    }


def analyze_by_event(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """按事件类型分组统计收益率"""
    rows = []
    for evt_col, evt_name in EVENT_COLS:
        if evt_col not in enriched_df.columns:
            continue
        mask = enriched_df[evt_col] == True
        sub = enriched_df.loc[mask]
        if sub.empty:
            continue

        for h in HORIZONS:
            stats = _group_stats(sub, h)
            if stats:
                row = {"事件": evt_name, "持有期": f"{h}日"}
                row.update(stats)
                rows.append(row)

    return pd.DataFrame(rows)


def analyze_by_vol_zscore(enriched_df: pd.DataFrame, zscore_col: str = "vol_zscore_20") -> pd.DataFrame:
    """按成交量 zscore 分组统计收益率"""
    if zscore_col not in enriched_df.columns:
        print(f"  警告: {zscore_col} 列不存在")
        return pd.DataFrame()

    valid = enriched_df[zscore_col].notna()
    df_valid = enriched_df.loc[valid]

    rows = []
    for bin_name, bin_cond in VOL_ZSCORE_BINS:
        mask = bin_cond(df_valid[zscore_col])
        sub = df_valid.loc[mask]
        if sub.empty:
            continue

        for h in HORIZONS:
            stats = _group_stats(sub, h)
            if stats:
                row = {"成交量分组": bin_name, "zscore列": zscore_col, "持有期": f"{h}日"}
                row.update(stats)
                rows.append(row)

    return pd.DataFrame(rows)


def analyze_event_vol_cross(enriched_df: pd.DataFrame, zscore_col: str = "vol_zscore_20") -> pd.DataFrame:
    """事件类型 × 成交量水平 交叉分析"""
    if zscore_col not in enriched_df.columns:
        return pd.DataFrame()

    valid = enriched_df[zscore_col].notna()
    df_valid = enriched_df.loc[valid]

    rows = []
    for evt_col, evt_name in EVENT_COLS:
        if evt_col not in df_valid.columns:
            continue
        evt_mask = df_valid[evt_col] == True
        evt_sub = df_valid.loc[evt_mask]
        if evt_sub.empty:
            continue

        for bin_name, bin_cond in VOL_ZSCORE_BINS:
            mask = bin_cond(evt_sub[zscore_col])
            sub = evt_sub.loc[mask]
            if sub.empty:
                continue

            for h in HORIZONS:
                stats = _group_stats(sub, h)
                if stats:
                    row = {"事件": evt_name, "成交量分组": bin_name, "持有期": f"{h}日"}
                    row.update(stats)
                    rows.append(row)

    return pd.DataFrame(rows)


def print_summary(event_stats: pd.DataFrame, vol_stats: pd.DataFrame, cross_stats: pd.DataFrame):
    """打印统计表格"""
    print("\n" + "=" * 100)
    print("一、按事件类型分组的收益率统计")
    print("=" * 100)
    if not event_stats.empty:
        display_cols = ["事件", "持有期", "样本数", "均值", "中位数", "胜率", "MFE均值", "MAE均值"]
        display_cols = [c for c in display_cols if c in event_stats.columns]
        print(tabulate(event_stats[display_cols], headers="keys", tablefmt="grid", showindex=False))

    print("\n" + "=" * 100)
    print("二、按成交量 zscore 分组的收益率统计")
    print("=" * 100)
    if not vol_stats.empty:
        display_cols = ["成交量分组", "持有期", "样本数", "均值", "中位数", "胜率", "MFE均值", "MAE均值"]
        display_cols = [c for c in display_cols if c in vol_stats.columns]
        print(tabulate(vol_stats[display_cols], headers="keys", tablefmt="grid", showindex=False))

    print("\n" + "=" * 100)
    print("三、事件 × 成交量 交叉分析（5日持有期）")
    print("=" * 100)
    if not cross_stats.empty:
        # 只展示5日持有期的交叉表
        cross_5d = cross_stats[cross_stats["持有期"] == "5日"]
        if not cross_5d.empty:
            pivot_data = cross_5d.pivot_table(
                index="事件", columns="成交量分组",
                values="均值(raw)", aggfunc="first"
            )
            if not pivot_data.empty:
                pivot_display = pivot_data.map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")
                print(tabulate(pivot_display, headers="keys", tablefmt="grid"))

            # 胜率交叉表
            pivot_wr = cross_5d.pivot_table(
                index="事件", columns="成交量分组",
                values="胜率(raw)", aggfunc="first"
            )
            if not pivot_wr.empty:
                pivot_wr_display = pivot_wr.map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
                print("\n胜率交叉表（5日持有期）:")
                print(tabulate(pivot_wr_display, headers="keys", tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser(
        description="PA事件收益率与成交量关系分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-date", default="2023-01-01", help="分析开始日期 (默认 2023-01-01)")
    parser.add_argument("--end-date", default="2026-05-19", help="分析结束日期 (默认 2026-05-19)")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date

    print("=" * 100)
    print("PA事件收益率与成交量关系分析")
    print(f"  分析区间: {start_date} ~ {end_date}")
    print(f"  持有期: {HORIZONS} 日")
    print("=" * 100)

    # 1. 加载选股结果
    print("\n[1/4] 加载 PA 选股结果...")
    pa_df = load_pa_selection(start_date, end_date)
    print(f"  共 {len(pa_df)} 条记录, {pa_df['selection_date'].nunique()} 个交易日")

    if pa_df.empty:
        print("无选股结果数据，退出")
        return

    # 2. 加载价格数据
    print("\n[2/4] 加载价格数据...")
    ts_codes = pa_df["ts_code"].unique().tolist()
    # end_date 需要往后推 max(HORIZONS) 天以计算未来收益
    from datetime import timedelta
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    extended_end = (end_dt + timedelta(days=max(HORIZONS) + 10)).strftime("%Y-%m-%d")
    price_pivot = load_price_pivot(ts_codes, start_date, extended_end)
    if not price_pivot:
        print("无法加载价格数据，退出")
        return
    print(f"  价格数据: {price_pivot['close'].shape[0]} 个交易日, {price_pivot['close'].shape[1]} 只股票")

    # 3. 计算成交量 zscore 和未来收益
    print("\n[3/4] 计算成交量 zscore 和未来收益...")
    vol_zscores = compute_vol_zscores(price_pivot, windows=[5, 10, 20])
    enriched_df = enrich_pa_data(pa_df, price_pivot, vol_zscores)

    # 统计有效未来收益比例
    for h in HORIZONS:
        col = f"return_{h}"
        valid_rate = enriched_df[col].notna().mean() * 100
        print(f"  {h}日收益: {valid_rate:.1f}% 有效")

    # 4. 分析
    print("\n[4/4] 统计分析...")
    event_stats = analyze_by_event(enriched_df)

    # 对每个 zscore 窗口都做分析
    all_vol_stats = []
    for w in [5, 10, 20]:
        zscore_col = f"vol_zscore_{w}"
        vol_stats = analyze_by_vol_zscore(enriched_df, zscore_col)
        all_vol_stats.append(vol_stats)
    vol_stats_all = pd.concat(all_vol_stats, ignore_index=True)

    # 交叉分析（用20日zscore）
    cross_stats = analyze_event_vol_cross(enriched_df, "vol_zscore_20")

    # 打印结果
    print_summary(event_stats, vol_stats_all, cross_stats)

    # 保存 CSV
    event_stats.to_csv(RESULTS_DIR / "event_return_stats.csv", index=False, encoding="utf-8-sig")
    vol_stats_all.to_csv(RESULTS_DIR / "vol_zscore_return_stats.csv", index=False, encoding="utf-8-sig")
    cross_stats.to_csv(RESULTS_DIR / "event_vol_cross_stats.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到 {RESULTS_DIR}/")

    # 额外：对每个 zscore 窗口分别做交叉分析并保存
    for w in [5, 10, 20]:
        zscore_col = f"vol_zscore_{w}"
        cross = analyze_event_vol_cross(enriched_df, zscore_col)
        cross.to_csv(RESULTS_DIR / f"event_vol_cross_zscore{w}.csv", index=False, encoding="utf-8-sig")

    print("\n分析完成")


if __name__ == "__main__":
    main()
