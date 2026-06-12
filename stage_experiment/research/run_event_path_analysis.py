# -*- coding: utf-8 -*-
"""
stage_experiment/research/run_event_path_analysis.py - 端到端事件路径分析

Purpose: 生成每只股票每天的事件上下文特征和前向收益标签。

Public API:
    run_analysis(symbols, config=None, period='d', count=255) -> DataFrame

Inputs:
    symbols: 股票代码列表，如 ['000001', '600036']
    config: 配置字典或 yaml 路径
    period: K线周期，默认日线
    count: 拉取K线数量

Outputs: DataFrame with 事件上下文特征 + 前向收益标签
How to Run:
    python -m stage_experiment.research.run_event_path_analysis --symbols 000001 600036
Examples:
    python -m stage_experiment.research.run_event_path_analysis --symbols 000001
    python -m stage_experiment.research.run_event_path_analysis --symbols 000001 600036 --output result.parquet
Side Effects: 无写库/改表操作，可选输出 CSV/Parquet 文件
"""
import argparse
import sys
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from stage_experiment.core.event_groups import map_event_groups, load_config
from stage_experiment.core.event_density import compute_event_density
from stage_experiment.core.event_decay import compute_event_decay
from stage_experiment.core.event_sequence import compute_event_sequence
from stage_experiment.core.event_context import compute_event_context
from stage_experiment.core.path_score import compute_path_score


def _compute_forward_returns(close: pd.Series, periods: list) -> pd.DataFrame:
    result = pd.DataFrame(index=close.index)
    for n in periods:
        future_close = close.shift(-n)
        result[f"future_ret_{n}"] = future_close / close - 1.0
    return result


def _compute_forward_mdd(close: pd.Series, periods: list) -> pd.DataFrame:
    result = pd.DataFrame(index=close.index)
    close_vals = close.to_numpy(dtype=float)
    n_total = len(close_vals)
    for n in periods:
        mdd = np.full(n_total, np.nan)
        for i in range(n_total):
            end = min(i + n + 1, n_total)
            window = close_vals[i:end]
            if len(window) < 2:
                continue
            cummax = np.maximum.accumulate(window)
            drawdown = window / cummax - 1.0
            mdd[i] = drawdown.min()
        result[f"future_mdd_{n}"] = pd.Series(mdd, index=close.index)
    return result


def _process_single_symbol(
    symbol: str,
    df: pd.DataFrame,
    config: dict,
    period: str = "d",
) -> pd.DataFrame:
    from factor_lib import compute_panel_v2
    from event_lib import detect_panel

    factors_df = compute_panel_v2(df)
    events_df = detect_panel(factors_df)

    group_df = map_event_groups(events_df, config)
    density_df = compute_event_density(group_df, config)
    decay_df = compute_event_decay(group_df, config)
    sequence_df = compute_event_sequence(group_df, config)
    context_df = compute_event_context(decay_df, sequence_df, config)
    score_df = compute_path_score(context_df, group_df, config)

    forward_periods = config.get("forward_returns", [5, 10, 20, 40])
    close = df["close"] if "close" in df.columns else factors_df.get("close")
    fwd_ret = _compute_forward_returns(close, forward_periods)
    fwd_mdd = _compute_forward_mdd(close, [20, 40])

    out = pd.DataFrame(index=df.index)
    out["ts_code"] = symbol
    out["period"] = period

    for component in [group_df, density_df, decay_df, sequence_df, context_df, score_df, fwd_ret, fwd_mdd]:
        for col in component.columns:
            out[col] = component[col]

    group_defs = config.get("event_groups", {})
    all_event_names = set()
    for evt_list in group_defs.values():
        all_event_names.update(evt_list)
    for evt_name in sorted(all_event_names):
        if evt_name in events_df.columns:
            out[evt_name] = events_df[evt_name].fillna(0).astype(int)

    boundary_cols = [
        "stage_lower_boundary", "stage_mid_boundary", "stage_upper_boundary",
        "price_pos_in_stage_01", "dist_to_stage_lower_atr", "dist_to_stage_upper_atr",
    ]
    for col in boundary_cols:
        if col in factors_df.columns:
            out[col] = factors_df[col]

    return out


def run_analysis(
    symbols: list,
    config: dict = None,
    period: str = "d",
    count: int = 255,
) -> pd.DataFrame:
    """
    端到端生成每只股票每天的事件上下文特征。

    Args:
        symbols: 股票代码列表
        config: 配置字典，若为 None 则从默认 yaml 加载
        period: K线周期
        count: 拉取K线数量

    Returns:
        合并 DataFrame 含事件上下文特征 + 前向收益标签
    """
    if config is None:
        config = load_config()

    from datasource.pytdx_client import connect_pytdx, get_kline_data

    api = connect_pytdx()
    try:
        all_results = []
        for symbol in symbols:
            df = get_kline_data(api, symbol, period, count)
            if df.empty:
                print(f"⚠ {symbol}: 无数据，跳过")
                continue
            if len(df) < 60:
                print(f"⚠ {symbol}: 数据不足60条({len(df)})，跳过")
                continue
            print(f"✅ {symbol}: 获取 {len(df)} 条K线")
            try:
                result = _process_single_symbol(symbol, df, config, period=period)
                all_results.append(result)
            except Exception as e:
                print(f"❌ {symbol}: 处理失败 - {e}")
                raise
        if not all_results:
            return pd.DataFrame()
        return pd.concat(all_results, axis=0)
    finally:
        api.disconnect()


def main():
    parser = argparse.ArgumentParser(description="端到端事件路径分析")
    parser.add_argument("--symbols", nargs="+", required=True, help="股票代码列表")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（.csv 或 .parquet）")
    parser.add_argument("--count", type=int, default=255, help="拉取K线数量")
    args = parser.parse_args()

    result = run_analysis(args.symbols, count=args.count)

    if result.empty:
        print("无有效结果")
        sys.exit(1)

    print(f"\n=== 分析结果 ===")
    print(f"总行数: {len(result)}")
    print(f"总列数: {len(result.columns)}")
    print(f"列名: {list(result.columns)}")

    if args.output:
        if args.output.endswith(".parquet"):
            result.to_parquet(args.output, index=False)
        else:
            result.to_csv(args.output, index=False)
        print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
