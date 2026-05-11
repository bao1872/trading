#!/usr/bin/env python3
"""
每日推理生成器 — 唯一负责从原始数据构造当日候选 → 加载模型推理 → 生成 predictions/{T}.parquet。

用途：
    模型低频训练（按周/月），推理高频生成（每日盘后自动）。
    不依赖 candidate_with_scores.parquet（那是研究产物），不从研究表切一天。

用法：
    python -m stop_experiment.pipeline.07_generate_daily_predictions --date 2026-05-08
    python -m stop_experiment.pipeline.07_generate_daily_predictions --date 2026-05-08 --force

参数：
    --date   目标交易日 YYYY-MM-DD（不指定则自动取最新交易日）
    --force  强制覆盖已存在的预测账本

流程（6步）：
    Step 1: 解析目标交易日 T
    Step 2: 从 DB 构造 obs_date=T 的单日候选样本
    Step 3: 拼接因子库特征 + 派生特征
    Step 4: 加载已训练 final 模型（4个）
    Step 5: 推理 + score_stocks 打分
    Step 6: 写出 predictions/T.parquet

输入：
    - DB: stop_loss_selection (信号), stock_k_data (K线)
    - output/models_control/xxx_final.txt (4个已训练模型)

输出：
    - output/predictions/YYYY-MM-DD.parquet
      列: ts_code, signal_id, obs_date, obs_day, pred_sell_reg, pred_sell_cls,
           pred_buy_reg, pred_buy_cls, score

副作用：只读 DB + parquet，写 parquet（幂等）
"""

import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sqlalchemy import text

from datasource.database import get_engine
from stop_experiment.pipeline.stop_config import (
    OBS_DAYS, SELL_CLS_THRESHOLD, BUY_CLS_THRESHOLD,
    OUTPUT_DIR, MODELS_DIR, PREDICTIONS_DIR, MODEL_SPECS,
    PRODUCTION_PARAMS,
)
from stop_experiment.pipeline.factor_columns import (
    SLC_STATIC_COLS, ALL_FEATURE_COLS, META_COLS,
)
from stop_experiment.pipeline.compute_factors import compute_stock_factors
from stop_experiment.backtest.simple_backtest import score_stocks


CANDIDATE_OBS_DAYS = PRODUCTION_PARAMS["candidate_obs_days"]
SIGNAL_LOOKBACK_DAYS = OBS_DAYS + 5
OBS_COLS = ["ts_code", "signal_id", "obs_date", "obs_day",
            "pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls", "score"]


# ==================== Step 1: 日期解析 ====================

def _resolve_trade_date(args, engine):
    """确定目标交易日。--date 指定 → 使用；否则从 K 线取最新。"""
    if args.date:
        return pd.to_datetime(args.date)

    with engine.connect() as conn:
        sql = "SELECT MAX(bar_time) FROM stock_k_data WHERE freq = 'd'"
        result = conn.execute(text(sql)).scalar()
        if result is None:
            raise RuntimeError("无法获取最新交易日，stock_k_data 为空")
        latest = pd.to_datetime(result)
    print(f"  [信息] 未指定 --date，自动使用最新交易日: {latest.strftime('%Y-%m-%d')}")
    return latest


# ==================== Step 2: 构造单日候选样本 ====================

def _build_candidates(target_date, engine):
    """
    从 DB 构造 obs_date=T 的单日候选样本。
    复用 01_build_dataset 的函数，但只加载必要的信号和 K 线。
    返回 (df_day, kline_dict, msg)。
    """
    target_dt = pd.to_datetime(target_date) if isinstance(target_date, str) else target_date
    signal_start = (target_dt - pd.Timedelta(days=SIGNAL_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    signal_end = target_dt.strftime("%Y-%m-%d")

    # 读信号
    with engine.connect() as conn:
        sql = """
            SELECT id, selection_date, signal_date, ts_code, stock_name,
                   sell_stop_triggered, buy_stop_triggered,
                   sell_trigger_volume, buy_trigger_volume,
                   active_sell_cluster_count, active_buy_cluster_count,
                   sum_sells_active, sum_buys_active,
                   sell_trigger_max_vol_price, sell_stop_scale,
                   nearest_sell_stop_price, nearest_buy_stop_price,
                   dist_to_nearest_sell_stop_atr, dist_to_nearest_buy_stop_atr,
                   stop_cluster_ratio, change_pct, vol_zscore,
                   bbmacd_event, daily_bb_width_zscore,
                   last_event_type, last_event_volume, last_event_bars_ago
            FROM stop_loss_selection
            WHERE signal_date >= :sig_start AND signal_date <= :sig_end
            ORDER BY signal_date, ts_code
        """
        signals = pd.read_sql(text(sql), conn, params={
            "sig_start": signal_start, "sig_end": signal_end,
        })
    print(f"  [信号] {len(signals)} 条 (signal_date {signal_start} ~ {signal_end})")

    if signals.empty:
        return None, None, "当日无信号"

    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    signals["selection_date"] = pd.to_datetime(signals["selection_date"])

    ts_codes = sorted(signals["ts_code"].unique())
    min_signal = signals["signal_date"].min()
    max_signal = signals["signal_date"].max()

    # 读 K 线（扩展范围确保因子 lookback 与训练管线一致）
    # 训练时 batch_kline_start = min(selection_date) - 150天，最早信号约 2023-03
    # 即训练 K 线从约 2022-10 开始。为确保滚动因子有足够 warm-up，
    # 从 target_date - 1300天（约 3.5 年）开始加载，与训练口径对齐
    kline_start = (pd.Timestamp(target_dt) - pd.Timedelta(days=1300)).strftime("%Y-%m-%d")
    kline_end = (pd.Timestamp(max_signal) + pd.Timedelta(days=OBS_DAYS + 50)).strftime("%Y-%m-%d")

    kline_dict = {}
    with engine.connect() as conn:
        for ts_code in ts_codes:
            sql_k = """
                SELECT bar_time, open, high, low, close, volume
                FROM stock_k_data
                WHERE ts_code = :ts_code AND freq = 'd'
                  AND bar_time >= :start_date AND bar_time <= :end_date
                ORDER BY bar_time
            """
            df_k = pd.read_sql(text(sql_k), conn, params={
                "ts_code": ts_code, "start_date": kline_start, "end_date": kline_end,
            })
            if not df_k.empty:
                df_k["bar_time"] = pd.to_datetime(df_k["bar_time"])
                df_k = df_k.set_index("bar_time")
                kline_dict[ts_code] = df_k

    print(f"  [K线] {len(kline_dict)}/{len(ts_codes)} 只 ({kline_start} ~ {kline_end})")

    if not kline_dict:
        return None, None, "K线数据为空"

    # 展开观察期（复用 01 的核心逻辑）
    _mod01 = __import__("stop_experiment.pipeline.01_build_dataset", fromlist=["expand_observation_period"])
    expand_observation_period = _mod01.expand_observation_period
    df_expanded = expand_observation_period(signals, kline_dict)
    print(f"  [展开] {len(df_expanded)} 行 (所有 obs_day)")

    # 筛选 obs_date == T
    df_expanded["obs_date"] = pd.to_datetime(df_expanded["obs_date"])
    df_day = df_expanded[df_expanded["obs_date"] == target_dt].copy()
    print(f"  [筛选] obs_date={target_dt.strftime('%Y-%m-%d')}: {len(df_day)} 行")

    if df_day.empty:
        return None, kline_dict, "obs_date=T 无候选"

    return df_day, kline_dict, "OK"


# ==================== Step 3: 因子库 + 派生特征 ====================

def _merge_and_derive(df, kline_dict):
    """拼接因子库特征 → 计算派生特征。复用 01 的函数。"""
    _mod01b = __import__("stop_experiment.pipeline.01_build_dataset", fromlist=["merge_factor_lib_features", "build_derived_features"])
    merge_factor_lib_features = _mod01b.merge_factor_lib_features
    build_derived_features = _mod01b.build_derived_features
    df = merge_factor_lib_features(df, kline_dict)
    df = build_derived_features(df)
    return df


# ==================== Step 4-5: 模型推理 + 打分 ====================

def _predict_and_score(df):
    """加载 final 模型 → 推理 → score_stocks 打分。返回增加了预测列和 score 的 df。"""
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    if not feature_cols:
        raise RuntimeError(f"特征列为空，ALL_FEATURE_COLS 中无列在候选数据中。候选列: {list(df.columns[:20])}...")

    # 确保特征列均为数值类型（LightGBM 要求 int/float/bool）
    for c in feature_cols:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for model_name, spec in MODEL_SPECS.items():
        model_path = os.path.join(MODELS_DIR, f"{model_name}_final.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        model = lgb.Booster(model_file=model_path)
        df[f"pred_{model_name}"] = model.predict(df[feature_cols])

    df = score_stocks(df, strategy=PRODUCTION_PARAMS.get("strategy_default", "sell_score"))
    return df


# ==================== Step 6: 写出 ====================

def _save_predictions(df, target_date):
    """写出 predictions/T.parquet。"""
    target_dt = pd.to_datetime(target_date) if isinstance(target_date, str) else target_date
    date_str = target_dt.strftime("%Y-%m-%d")
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_path = os.path.join(PREDICTIONS_DIR, f"{date_str}.parquet")

    if "obs_day" in df.columns:
        before = len(df)
        df = df[df["obs_day"].isin(CANDIDATE_OBS_DAYS)].copy()
        if len(df) < before:
            print(f"  [过滤] obs_day∈{CANDIDATE_OBS_DAYS}: {before} → {len(df)} 行")

    available_cols = [c for c in OBS_COLS if c in df.columns]
    df_save = df[available_cols].copy()
    df_save.to_parquet(out_path, index=False)
    print(f"  [保存] {out_path} ({len(df_save)} 行, {len(available_cols)} 列)")
    return out_path


# ==================== 主入口 ====================

def generate_daily_predictions(target_date, force=False):
    """6 步日推理主流程。返回 (success, message, output_path)。"""
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    date_str = target_date.strftime("%Y-%m-%d")

    out_path = os.path.join(PREDICTIONS_DIR, f"{date_str}.parquet")
    if os.path.exists(out_path) and not force:
        existing = pd.read_parquet(out_path)
        return True, f"预测账本已存在 ({len(existing)} 行)，跳过。--force 可强制覆盖", out_path

    # 优先从 full_test_predictions 提取（确保与训练口径完全一致）
    ftp_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    if os.path.exists(ftp_path) and not force:
        ftp = pd.read_parquet(ftp_path)
        if "obs_date" in ftp.columns:
            ftp["obs_date"] = pd.to_datetime(ftp["obs_date"])
            day_data = ftp[ftp["obs_date"] == target_date].copy()
            if not day_data.empty:
                if "obs_day" in day_data.columns:
                    day_data = day_data[day_data["obs_day"].isin(CANDIDATE_OBS_DAYS)].copy()
                os.makedirs(PREDICTIONS_DIR, exist_ok=True)
                available_cols = [c for c in OBS_COLS if c in day_data.columns]
                day_save = day_data[available_cols].copy()
                day_save.to_parquet(out_path, index=False)
                print(f"  [提取] 从 full_test_predictions 提取 {date_str}: {len(day_save)} 行")
                return True, f"从 full_test_predictions 提取 ({len(day_save)} 行)", out_path

    print(f"\n{'='*60}")
    print(f"  07 单日推理: {date_str}")
    print(f"{'='*60}")

    engine = get_engine()

    try:
        # Step 2: 构造候选
        print(f"\n[Step 2] 构造单日候选样本...")
        df_day, kline_dict, msg2 = _build_candidates(target_date, engine)
        if df_day is None:
            # 无候选 → 生成空预测账本
            os.makedirs(PREDICTIONS_DIR, exist_ok=True)
            empty_df = pd.DataFrame(columns=OBS_COLS)
            empty_df.to_parquet(out_path, index=False)
            print(f"  [信息] {msg2}，生成空预测账本: {out_path}")
            return True, f"无候选 ({msg2})，空账本已生成", out_path

        # Step 3: 因子库 + 派生（复用 Step 2 的 kline_dict，不重复查询 DB）
        print(f"\n[Step 3] 拼接因子库 + 派生特征...")
        df_day = _merge_and_derive(df_day, kline_dict)

        # 过滤不可买入样本（与训练管线对齐）
        if "can_buy" in df_day.columns:
            before_cb = len(df_day)
            df_day = df_day[df_day["can_buy"] == 1].copy()
            if len(df_day) < before_cb:
                print(f"  [过滤] can_buy==1: {before_cb} → {len(df_day)} 行")

        # Step 4-5: 推理 + 打分
        print(f"\n[Step 4-5] 模型推理 + 打分...")
        df_day = _predict_and_score(df_day)

        # Step 6: 写出
        print(f"\n[Step 6] 写出预测账本...")
        path = _save_predictions(df_day, target_date)

        return True, f"预测账本已生成 ({len(df_day)} 行)", path

    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="每日推理生成器: 构造单日候选 → 模型推理 → 预测账本")
    parser.add_argument("--date", help="目标交易日 YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的预测账本")
    args = parser.parse_args()

    engine = get_engine()
    target_date = _resolve_trade_date(args, engine)
    engine.dispose()

    success, msg, path = generate_daily_predictions(target_date, force=args.force)
    print(f"\n{msg}")
    if success:
        print(f"  ✅ 完成: {path}")
    else:
        print(f"  ❌ 失败")
        sys.exit(1)


if __name__ == "__main__":
    main()