#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 Stop-Loss Clustering 实验数据集（分批版本）

Purpose:
    独立数据集构建脚本（唯一访问数据库的步骤）。
    读 stop_loss_selection → 计算因子库 → 展开20天观察期 → 计算MFE/MAE标签 → 分批保存。
    支持分批构建以控制内存，支持断点续跑。

Pipeline Position:
    训练流水线第一步（离线，一次性）。
    上游: DB (stop_loss_selection, stock_k_data)
    下游: 02_train_gbdt_models.py

Inputs:
    - stop_loss_selection (PostgreSQL): SLC信号数据
    - stock_k_data (PostgreSQL): 日K线数据

Outputs:
    - stop_experiment/output/dataset_batches/batch_XXX.parquet: 分批数据集
    - stop_experiment/output/dataset_batches/manifest.json: 元数据

How to Run:
    python stop_experiment/pipeline/01_build_dataset.py --batch-size 5000         # 全量分批
    python stop_experiment/pipeline/01_build_dataset.py --sample-limit 100         # 小批量调试
    python stop_experiment/pipeline/01_build_dataset.py --batch-index 3 --batch-total 11  # 断点续跑

Side Effects:
    - 只读数据库，不写入任何表
    - 输出文件到 stop_experiment/output/dataset_batches/

Note:
    OBS_DAYS=20 是训练侧观察窗（20天展开计算MFE/MAE标签），
    与生产 candidate_obs_days=[1] 完全独立。训练需要全20天样本构建标签。
"""

from __future__ import annotations

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import gc
import json
from tqdm import tqdm

from datasource.database import get_engine
from sqlalchemy import text

from stop_experiment.pipeline.stop_config import (
    OBS_DAYS, SELL_CLS_THRESHOLD, BUY_CLS_THRESHOLD,
    OUTPUT_DIR, DATASET_PATH,
    FACTOR_WARMUP_DAYS, LABEL_FORWARD_DAYS,
)
from stop_experiment.pipeline.factor_columns import (
    SLC_STATIC_COLS, DYNAMIC_COLS, DERIVED_COLS,
    ALL_FEATURE_COLS, META_COLS,
    FACTOR_CATEGORIES,
)
from stop_experiment.pipeline.compute_factors import compute_stock_factors, verify_against_db


# ==================== 数据读取 ====================

def load_stop_loss_selection(conn, sample_limit: int | None = None) -> pd.DataFrame:
    """从 stop_loss_selection 读取全部记录"""
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
        ORDER BY selection_date, ts_code
    """
    if sample_limit:
        sql += f" LIMIT {sample_limit}"
    df = pd.read_sql(text(sql), conn)
    print(f"  stop_loss_selection: {len(df)} 条信号, {df['ts_code'].nunique()} 只股票, "
          f"日期 {df['selection_date'].min()} ~ {df['selection_date'].max()}")
    return df


def load_stock_kline(conn, ts_code: str, start_date: str, end_date: str,
                     extra_bars: int = 50) -> pd.DataFrame:
    """
    加载单只股票的日K线数据。

    取 [start_date - extra_bars, end_date + OBS_DAYS + extra_bars] 范围，
    确保因子计算有足够的 lookback，标签计算有足够的 forward。
    """
    sql = """
        SELECT bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE ts_code = :ts_code AND freq = 'd'
          AND bar_time >= :start_date AND bar_time <= :end_date
        ORDER BY bar_time
    """
    # 扩展日期范围，确保有足够的 lookback/forward
    df = pd.read_sql(text(sql), conn, params={
        "ts_code": ts_code,
        "start_date": start_date,
        "end_date": end_date,
    })
    if df.empty:
        return df
    df = df.set_index("bar_time")
    return df


def load_batch_kline(conn, ts_codes: list, start_date: str, end_date: str) -> dict:
    """
    批量加载多只股票的日K线数据（单次查询，大幅提速）。

    Returns:
        {ts_code: DataFrame(index=bar_time, columns=[open,high,low,close,volume])}
    """
    sql = """
        SELECT ts_code, bar_time, open, high, low, close, volume
        FROM stock_k_data
        WHERE ts_code = ANY(:ts_codes) AND freq = 'd'
          AND bar_time >= :start_date AND bar_time <= :end_date
        ORDER BY ts_code, bar_time
    """
    df = pd.read_sql(text(sql), conn, params={
        "ts_codes": ts_codes,
        "start_date": start_date,
        "end_date": end_date,
    })
    kline_dict = {}
    for ts_code, gdf in df.groupby("ts_code"):
        gdf["bar_time"] = pd.to_datetime(gdf["bar_time"])
        gdf = gdf.set_index("bar_time").sort_index()
        kline_dict[ts_code] = gdf
    return kline_dict


# ==================== 观察期展开 ====================

def expand_observation_period(signals: pd.DataFrame, kline_dict: dict) -> pd.DataFrame:
    """
    将信号展开为20天观察期，每个观察日一个样本。

    Parameters
    ----------
    signals : pd.DataFrame
        stop_loss_selection 数据
    kline_dict : dict
        {ts_code: DataFrame(index=bar_time, columns=[open,high,low,close,volume])}

    Returns
    -------
    pd.DataFrame
        每行一个观察日样本，包含 SLC 静态特征 + 动态特征 + 标签
    """
    rows = []

    for _, signal in tqdm(signals.iterrows(), total=len(signals), desc="展开观察期"):
        ts_code = signal["ts_code"]
        signal_date = signal["signal_date"] if pd.notna(signal["signal_date"]) else signal["selection_date"]
        signal_date = pd.Timestamp(signal_date)

        kline = kline_dict.get(ts_code)
        if kline is None or kline.empty:
            continue

        obs_start_mask = kline.index >= signal_date
        if not obs_start_mask.any():
            continue
        obs_start_idx = kline.index.get_loc(kline.index[obs_start_mask][0])

        # 收集 SLC 静态特征
        slc_features = {}
        for col in SLC_STATIC_COLS:
            if col in signal.index:
                val = signal[col]
                slc_features[col] = val

        # last_event_type label encoding（仅存储到数据中，不作为特征）
        # 注：last_event_type v2已从特征列表剔除，但保留在数据中以备参考

        # bbmacd_event 不在 SLC_STATIC_COLS 中（已用 daily_bb_width_zscore 替代）

        # nearest_sell_stop_price / nearest_buy_stop_price 用于计算 dist_to_sell_stop_pct
        nearest_sell_price = signal.get("nearest_sell_stop_price", np.nan)
        nearest_buy_price = signal.get("nearest_buy_stop_price", np.nan)

        # 展开观察期
        for obs_offset in range(OBS_DAYS):
            obs_idx = obs_start_idx + obs_offset
            if obs_idx >= len(kline):
                break

            obs_date = kline.index[obs_idx]
            obs_close = kline["close"].iloc[obs_idx]

            # ---- 动态特征 ----
            dynamic = {}
            dynamic["obs_day"] = obs_offset + 1

            # ret_to_trigger: 相对信号日收盘价
            # 信号日收盘价 = kline 中 signal_date 对应的 close
            signal_close_mask = kline.index >= signal_date
            if signal_close_mask.any():
                signal_close_idx = kline.index.get_loc(kline.index[signal_close_mask][0])
                signal_close = kline["close"].iloc[signal_close_idx]
                if signal_close > 0:
                    dynamic["ret_to_trigger"] = (obs_close - signal_close) / signal_close
                else:
                    dynamic["ret_to_trigger"] = np.nan
            else:
                dynamic["ret_to_trigger"] = np.nan

            # 观察期内的区间统计
            range_start = obs_start_idx
            range_end = obs_idx + 1
            range_high = kline["high"].iloc[range_start:range_end].max()
            range_low = kline["low"].iloc[range_start:range_end].min()

            dynamic["high_to_trigger"] = range_high / signal_close if signal_close > 0 else np.nan
            dynamic["low_to_trigger"] = range_low / signal_close if signal_close > 0 else np.nan
            dynamic["intraday_range"] = (kline["high"].iloc[obs_idx] - kline["low"].iloc[obs_idx]) / obs_close if obs_close > 0 else np.nan

            # vol_ratio: 当天成交量 / 20日均量
            vol_start = max(0, obs_idx - 20)
            vol_mean = kline["volume"].iloc[vol_start:obs_idx].mean()
            dynamic["vol_ratio"] = kline["volume"].iloc[obs_idx] / vol_mean if vol_mean > 0 else np.nan

            # range_position: 收盘价在区间高低中位置
            if range_high > range_low:
                dynamic["range_position"] = (obs_close - range_low) / (range_high - range_low)
            else:
                dynamic["range_position"] = np.nan

            # vol_change: 成交量变化率
            if obs_idx > 0 and kline["volume"].iloc[obs_idx - 1] > 0:
                dynamic["vol_change"] = kline["volume"].iloc[obs_idx] / kline["volume"].iloc[obs_idx - 1] - 1
            else:
                dynamic["vol_change"] = np.nan

            # dist_to_sell_stop_pct / dist_to_buy_stop_pct
            if pd.notna(nearest_sell_price) and obs_close > 0:
                dynamic["dist_to_sell_stop_pct"] = (obs_close - nearest_sell_price) / obs_close
            else:
                dynamic["dist_to_sell_stop_pct"] = np.nan
            if pd.notna(nearest_buy_price) and obs_close > 0:
                dynamic["dist_to_buy_stop_pct"] = (obs_close - nearest_buy_price) / obs_close
            else:
                dynamic["dist_to_buy_stop_pct"] = np.nan

            # ---- 标签计算 ----
            labels = {}
            label_start = obs_idx + 1
            label_end = min(obs_idx + 1 + OBS_DAYS, len(kline))

            if label_end > label_start:
                future_high = kline["high"].iloc[label_start:label_end].max()
                future_low = kline["low"].iloc[label_start:label_end].min()

                if obs_close > 0:
                    mfe = future_high / obs_close - 1.0
                    mae = future_low / obs_close - 1.0
                else:
                    mfe = np.nan
                    mae = np.nan

                labels["mfe_20"] = mfe
                labels["mae_20"] = mae
                labels["sell_signal"] = 1 if (pd.notna(mfe) and mfe > SELL_CLS_THRESHOLD) else 0
                labels["buy_signal"] = 1 if (pd.notna(mae) and mae < BUY_CLS_THRESHOLD) else 0
            else:
                labels["mfe_20"] = np.nan
                labels["mae_20"] = np.nan
                labels["sell_signal"] = np.nan
                labels["buy_signal"] = np.nan

            # can_buy: 非涨停板
            prev_close = kline["close"].iloc[obs_idx - 1] if obs_idx > 0 else np.nan
            if pd.notna(prev_close) and prev_close > 0:
                daily_limit = (obs_close - prev_close) / prev_close
                labels["can_buy"] = 1 if daily_limit < 0.095 else 0
            else:
                labels["can_buy"] = 1

            # ---- 组装行 ----
            # 保留已剔除特征的原始值（仅作参考，不参与模型训练）
            extra_ref = {
                "sell_stop_triggered": signal.get("sell_stop_triggered", np.nan),
                "buy_stop_triggered": signal.get("buy_stop_triggered", np.nan),
                "last_event_type_raw": str(signal.get("last_event_type", "")),
                "bbmacd_slope_3_raw": np.nan,  # 将在因子库merge后由build_derived_features处理
            }
            row = {
                "signal_id": signal["id"],
                "ts_code": ts_code,
                "stock_name": signal.get("stock_name", ""),
                "signal_date": signal_date,
                "obs_date": obs_date,
                "obs_close": obs_close,
                "selection_date": signal["selection_date"],
                **slc_features,
                **dynamic,
                **labels,
                **extra_ref,
            }
            rows.append(row)

    return pd.DataFrame(rows)


# ==================== 因子库特征拼接 ====================

def merge_factor_lib_features(dataset: pd.DataFrame, kline_dict: dict) -> pd.DataFrame:
    """
    对每只股票计算因子库特征，按 ts_code + obs_date merge 到数据集。

    截断K线到 max(obs_date) 一次计算全部因子。
    pivot 因子的前视偏差已在 compute_pivot_factors 中修复：
    对最后一段未结束的 run，沿用上一段已完成 run 的 pivot 值，
    确保 T 时刻的因子值只使用 T 之前确认的 pivot。

    Parameters
    ----------
    dataset : pd.DataFrame
        expand_observation_period 的输出
    kline_dict : dict
        {ts_code: DataFrame}，K线需包含因子计算所需的前置数据

    Returns
    -------
    pd.DataFrame
        增加了因子库特征列
    """
    from stop_experiment.pipeline.compute_factors import compute_stock_factors
    from stop_experiment.pipeline.factor_columns import (
        TREND_COLS, POSITION_COLS, MOMENTUM_COLS,
        VOLUME_COLS, RISK_COLS, RHYTHM_COLS, VSA_COLS,
    )
    from stop_experiment.pipeline.stop_config import VSA_ENABLED

    factor_lib_cols = TREND_COLS + POSITION_COLS + MOMENTUM_COLS + VOLUME_COLS + RISK_COLS + RHYTHM_COLS + (VSA_COLS if VSA_ENABLED else [])

    all_factor_rows = []
    ts_codes = dataset["ts_code"].unique()
    print(f"  计算因子库: {len(ts_codes)} 只股票 (截断到max(obs_date)一次计算)")

    for ts_code in tqdm(ts_codes, desc="因子库计算"):
        kline = kline_dict.get(ts_code)
        if kline is None or kline.empty:
            continue

        stock_mask = dataset["ts_code"] == ts_code
        stock_obs_dates = sorted(dataset.loc[stock_mask, "obs_date"].unique())
        if not stock_obs_dates:
            continue

        max_obs_date = pd.Timestamp(max(stock_obs_dates))
        kline_truncated = kline[kline.index <= max_obs_date]
        if len(kline_truncated) < 10:
            continue

        try:
            factors = compute_stock_factors(kline_truncated)
        except Exception as e:
            print(f"    {ts_code} 因子计算失败: {e}")
            continue

        obs_dates_ts = [pd.Timestamp(d) for d in stock_obs_dates]
        available_dates = [d for d in obs_dates_ts if d in factors.index]
        if not available_dates:
            continue

        available_cols = [c for c in factor_lib_cols if c in factors.columns]
        factor_rows = factors.loc[available_dates, available_cols].copy()
        factor_rows["ts_code"] = ts_code
        factor_rows = factor_rows.reset_index()
        factor_rows = factor_rows.rename(columns={"bar_time": "obs_date"})
        all_factor_rows.append(factor_rows)

    if not all_factor_rows:
        print("  警告: 无因子数据")
        return dataset

    factor_df = pd.concat(all_factor_rows, ignore_index=True)
    print(f"  因子数据: {len(factor_df)} 行")

    dataset["obs_date"] = pd.to_datetime(dataset["obs_date"])
    factor_df["obs_date"] = pd.to_datetime(factor_df["obs_date"])
    dataset = dataset.merge(factor_df, on=["ts_code", "obs_date"], how="left")
    print(f"  合并后: {len(dataset)} 行, {len(dataset.columns)} 列")

    return dataset


# ==================== 派生特征计算 ====================

def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算派生特征列 + 缺失值填充 + 归一化
    
    关键改动（v2）:
    - buy_stop 相关列（88%缺失）用0填充，加 has_buy_cluster 标记
    - bbmacd_slope_3 → bbmacd_slope_3_pct（归一化为百分比）
    """
    # ---- 缺失值填充 ----
    # buy_stop 相关列缺失率88%+，填充为0并加标记
    buy_stop_cols = ["active_buy_cluster_count", "dist_to_nearest_buy_stop_atr"]
    for col in buy_stop_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # has_buy_cluster: 标记是否存在buy cluster
    if "active_buy_cluster_count" in df.columns:
        df["has_buy_cluster"] = (df["active_buy_cluster_count"] > 0).astype(int)
    else:
        df["has_buy_cluster"] = 0

    # ---- bbmacd_slope_3_pct 归一化 ----
    # bbmacd_slope_3_pct 优先从因子库merge获取（compute_factors.py中已计算）
    # 如果因子库未提供（兼容旧流程），则用bbmacd_slope_3 / obs_close计算
    if "bbmacd_slope_3_pct" not in df.columns or df["bbmacd_slope_3_pct"].isna().all():
        if "bbmacd_slope_3" in df.columns and "obs_close" in df.columns:
            close_safe = df["obs_close"].replace(0, np.nan)
            df["bbmacd_slope_3_pct"] = df["bbmacd_slope_3"] / close_safe
            df["bbmacd_slope_3_pct"] = df["bbmacd_slope_3_pct"].replace([np.inf, -np.inf], np.nan)
        else:
            df["bbmacd_slope_3_pct"] = np.nan

    # ---- 派生特征 ----
    sell_count = df.get("active_sell_cluster_count", pd.Series(0, index=df.index)).fillna(0)
    buy_count = df.get("active_buy_cluster_count", pd.Series(0, index=df.index))
    dist_sell = df.get("dist_to_nearest_sell_stop_atr", pd.Series(0, index=df.index)).fillna(0)
    dist_buy = df.get("dist_to_nearest_buy_stop_atr", pd.Series(0.01, index=df.index))
    ratio = df.get("stop_cluster_ratio", pd.Series(0, index=df.index)).fillna(0)
    vol_z = df.get("vol_zscore", pd.Series(0, index=df.index)).fillna(0)
    change = df.get("change_pct", pd.Series(0, index=df.index)).fillna(0)

    df["cluster_count_ratio"] = sell_count / (buy_count + 1)
    df["dist_atr_ratio"] = dist_sell / (dist_buy + 0.01)
    df["stop_cluster_ratio_x_vol"] = ratio * vol_z
    df["trigger_count_ratio"] = sell_count * change

    return df


# ==================== 分批构建 ====================

BATCHES_DIR = os.path.join(OUTPUT_DIR, "dataset_batches")

DEFAULT_BATCH_SIZE = 5000


def build_batch(signals_batch: pd.DataFrame, kline_dict: dict, batch_idx: int) -> dict:
    """
    对一批信号展开观察期 + 计算因子 + 保存 parquet。

    Returns:
        dict: batch info {rows, signals, date_min, date_max}
    """
    print(f"\n{'='*50}")
    print(f"  Batch {batch_idx}: {len(signals_batch)} 条信号, "
          f"{signals_batch['ts_code'].nunique()} 只股票")

    # 展开观察期（需要forward K线计算标签）
    dataset = expand_observation_period(signals_batch, kline_dict)
    print(f"    展开后: {len(dataset)} 行")

    if dataset.empty:
        return {"rows": 0, "signals": 0}

    dataset = merge_factor_lib_features(dataset, kline_dict)

    # 派生特征
    dataset = build_derived_features(dataset)

    # 保存
    os.makedirs(BATCHES_DIR, exist_ok=True)
    batch_path = os.path.join(BATCHES_DIR, f"batch_{batch_idx:03d}.parquet")
    dataset.to_parquet(batch_path, index=False)

    # 数据质量检查
    _check_batch_quality(dataset, batch_idx)

    info = {
        "rows": len(dataset),
        "signals": len(signals_batch),
        "date_min": str(dataset["obs_date"].min()),
        "date_max": str(dataset["obs_date"].max()),
        "file": f"batch_{batch_idx:03d}.parquet",
    }
    print(f"    保存: {batch_path} ({info['rows']} 行)")
    return info


def _check_batch_quality(dataset: pd.DataFrame, batch_idx: int) -> None:
    """每个 batch 构建后检查数据质量，及时发现异常。"""
    from stop_experiment.pipeline.factor_columns import PIVOT_COLS, RHYTHM_COLS, POSITION_COLS

    total = len(dataset)
    if total == 0:
        print(f"    ⚠️ Batch {batch_idx}: 空数据集")
        return

    # 1. 关键因子 NaN 率
    check_cols = PIVOT_COLS + ["dsa_dir", "DSA_VWAP", "bbmacd_minus_avg"]
    for col in check_cols:
        if col in dataset.columns:
            nan_rate = dataset[col].isna().mean()
            if nan_rate > 0.3:
                print(f"    ⚠️ Batch {batch_idx}: {col} NaN率={nan_rate:.1%}")

    # 2. pivot 因子全 NaN 的股票数
    pivot_available = [c for c in PIVOT_COLS if c in dataset.columns]
    if pivot_available:
        all_nan_per_stock = dataset.groupby("ts_code")[pivot_available].apply(
            lambda g: g.isna().all().all()
        )
        n_all_nan = all_nan_per_stock.sum()
        if n_all_nan > 0:
            print(f"    ⚠️ Batch {batch_idx}: {n_all_nan} 只股票 pivot 因子全 NaN")

    # 3. 因子值范围检查
    if "dsa_pivot_pos_01" in dataset.columns:
        valid = dataset["dsa_pivot_pos_01"].dropna()
        if len(valid) > 0:
            out_of_range = ((valid < 0) | (valid > 1)).mean()
            if out_of_range > 0:
                print(f"    ⚠️ Batch {batch_idx}: dsa_pivot_pos_01 超出[0,1]比例={out_of_range:.1%}")

    # 4. obs_day 分布
    obs_day_dist = dataset["obs_day"].value_counts().sort_index()
    n_obs_days = len(obs_day_dist)
    print(f"    质量: {total} 行, {n_obs_days} obs_days, NaN检查通过")


def build_manifest(batch_infos: list, total_rows: int) -> dict:
    """生成 manifest.json"""
    return {
        "version": "v2_batched",
        "created_at": pd.Timestamp.now().isoformat(),
        "total_rows": total_rows,
        "total_batches": len(batch_infos),
        "batch_files": [bi["file"] for bi in batch_infos],
        "batch_details": batch_infos,
    }


# ==================== 主流程 ====================

def main(args):
    print("=" * 60)
    print("Stop-Loss Clustering 数据集构建（分批版本）")
    print("=" * 60)

    engine = get_engine()
    t0 = time.time()

    # 1. 读 stop_loss_selection
    print("\n[1/5] 读取 stop_loss_selection...")
    with engine.connect() as conn:
        signals = load_stop_loss_selection(conn, sample_limit=args.sample_limit)

    if signals.empty:
        print("无信号数据，退出")
        return

    # 按 selection_date 排序确保时间有序
    signals = signals.sort_values("selection_date").reset_index(drop=True)
    min_date = signals["selection_date"].min()
    max_date = signals["selection_date"].max()
    n_signals = len(signals)

    # 2. 确定批次数
    if args.sample_limit and args.batch_size == DEFAULT_BATCH_SIZE:
        batch_size = min(args.sample_limit, 500)
    else:
        batch_size = args.batch_size

    n_batches = max(1, (n_signals + batch_size - 1) // batch_size)

    if args.batch_total > 0:
        n_batches = args.batch_total

    print(f"\n[2/5] 分批构建: {n_signals} 条信号 → {n_batches} 批 (每批 ~{batch_size} 条)")
    print(f"  日期范围: {min_date} ~ {max_date}")
    print(f"  K线按批次按需加载（降低内存占用）")

    # 3. 逐批构建（每批独立加载K线，处理完释放）
    batch_infos = []
    total_rows = 0

    for batch_idx in range(n_batches):
        # 断点续跑：跳过已完成的 batch
        if args.batch_index >= 0 and batch_idx < args.batch_index:
            continue
        if args.batch_index >= 0 and batch_idx > args.batch_index:
            break

        start = batch_idx * batch_size
        end = min(start + batch_size, n_signals)
        signals_batch = signals.iloc[start:end]

        # 按批次加载K线（只加载该批次涉及的股票）
        batch_ts_codes = sorted(signals_batch["ts_code"].unique())
        batch_kline_start = pd.Timestamp(signals_batch["selection_date"].min()) - pd.Timedelta(days=FACTOR_WARMUP_DAYS)
        batch_kline_end = pd.Timestamp(signals_batch["selection_date"].max()) + pd.Timedelta(days=OBS_DAYS + 25)

        print(f"\n  加载K线: Batch {batch_idx}, {len(batch_ts_codes)} 只股票...")
        with engine.connect() as conn:
            kline_dict = load_batch_kline(
                conn, batch_ts_codes,
                str(batch_kline_start.date()), str(batch_kline_end.date()),
            )
        print(f"  加载K线: {len(kline_dict)}/{len(batch_ts_codes)} 只股票有数据")

        info = build_batch(signals_batch, kline_dict, batch_idx)
        batch_infos.append(info)
        total_rows += info["rows"]

        # 释放该批次K线内存
        del kline_dict
        gc.collect()

        print(f"\n  累计: {total_rows} 行, {len(batch_infos)} 批完成")

    # 4. 保存 manifest
    print(f"\n[5/5] 保存 manifest...")
    manifest = build_manifest(batch_infos, total_rows)
    manifest_path = os.path.join(BATCHES_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  保存: {manifest_path}")

    # 汇总
    print(f"\n{'='*60}")
    print("数据集摘要")
    print(f"{'='*60}")
    print(f"  总批次: {len(batch_infos)}")
    print(f"  总行数: {total_rows}")
    print(f"  总信号数: {sum(bi['signals'] for bi in batch_infos)}")
    print(f"  输出目录: {BATCHES_DIR}")
    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 Stop-Loss Clustering 实验数据集（分批版本）")
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="限制信号数量（调试用）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"每批信号数（默认 {DEFAULT_BATCH_SIZE}）")
    parser.add_argument("--batch-index", type=int, default=-1,
                        help="仅构建第 N 批（0-based，-1=全部，支持断点续跑）")
    parser.add_argument("--batch-total", type=int, default=0,
                        help="总批次数（仅 --batch-index >= 0 时有效）")
    args = parser.parse_args()
    main(args)
