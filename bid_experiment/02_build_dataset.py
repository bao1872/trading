# -*- coding: utf-8 -*-
"""
竞价数据集构建脚本

Purpose:
    合并特征与标签，应用样本筛选，划分 train/val/test，输出 parquet 数据集

Inputs:
    - raw_data/{symbol}/: 原始竞价数据（由 compute_features / compute_labels 读取）
    - bid_config: DATASET_DIR, EMBARGO_DAYS, OBS_TRAIN_END, OBS_VAL_END, BUY_MIN_AUC_RET_ABS, BUY_MIN_AUC_AMOUNT_RATIO
    - feature_columns: ALL_FEATURE_COLS, META_COLS, LABEL_COLS

Outputs:
    - {output_dir}/{symbol}.parquet: 含特征+标签+split列的数据集

How to Run:
    python bid_experiment/02_build_dataset.py
    python bid_experiment/02_build_dataset.py --symbol 000001 --output-dir bid_experiment/output/dataset

Side Effects:
    - 写文件（parquet），不写数据库
"""

import argparse
import logging
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bid_experiment.bid_config import (
    DATASET_DIR,
    EMBARGO_DAYS,
    OBS_TRAIN_END,
    OBS_VAL_END,
    BUY_MIN_AUC_RET_ABS,
    BUY_MIN_AUC_AMOUNT_RATIO,
    RAW_DATA_DIR,
)
from bid_experiment.feature_columns import ALL_FEATURE_COLS, META_COLS, LABEL_COLS
from bid_experiment.compute_features import compute_features_for_stock
from bid_experiment.compute_labels import compute_labels_for_stock

logger = logging.getLogger(__name__)


def _assign_split(trade_date_int: int, train_end: pd.Timestamp, val_end: pd.Timestamp,
                  embargo_days: int) -> str:
    embargo_cutoff = train_end - timedelta(days=embargo_days)
    dt = pd.Timestamp(str(trade_date_int))
    if dt <= embargo_cutoff:
        return "train"
    if dt <= train_end:
        return "embargo"
    if dt <= val_end:
        return "val"
    return "test"


def build_dataset(symbol: str, raw_data_dir: str, output_dir: str) -> pd.DataFrame:
    """构建单只股票的数据集

    Args:
        symbol: 股票代码，如 '000001'
        raw_data_dir: 原始数据根目录
        output_dir: 数据集输出目录

    Returns:
        合并后的 DataFrame（含 split 列）
    """
    features_df = compute_features_for_stock(symbol, raw_data_dir)
    if features_df.empty:
        logger.warning(f"特征数据为空: {symbol}")
        return pd.DataFrame()

    labels_df = compute_labels_for_stock(symbol, raw_data_dir)
    if labels_df.empty:
        logger.warning(f"标签数据为空: {symbol}")
        return pd.DataFrame()

    merged = features_df.merge(labels_df, on=["stock_id", "trade_date"], how="inner")
    if merged.empty:
        logger.warning(f"特征与标签合并后为空: {symbol}")
        return pd.DataFrame()

    n_before = len(merged)

    buy_mask = merged["auc_ret_close"].abs() >= BUY_MIN_AUC_RET_ABS
    buy_mask = buy_mask & (merged["auc_amount"] > 0)
    sell_mask = merged["ret_5d"] > -0.05
    sample_mask = buy_mask | sell_mask
    merged = merged[sample_mask].reset_index(drop=True)

    n_after = len(merged)
    logger.info(f"样本筛选: {n_before} -> {n_after} (过滤 {n_before - n_after})")

    train_end = pd.Timestamp(OBS_TRAIN_END)
    val_end = pd.Timestamp(OBS_VAL_END)

    merged["split"] = merged["trade_date"].apply(
        lambda d: _assign_split(d, train_end, val_end, EMBARGO_DAYS)
    )
    merged = merged[merged["split"] != "embargo"].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{symbol}.parquet")
    merged.to_parquet(out_path, index=False)
    logger.info(f"数据集已保存: {out_path} ({len(merged)} 条)")

    _print_statistics(merged)

    return merged


def _print_statistics(df: pd.DataFrame) -> None:
    """打印数据集统计信息"""
    print(f"\n{'='*60}")
    print(f"数据集统计")
    print(f"{'='*60}")
    print(f"总样本数: {len(df)}")

    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        n = len(split_df)
        print(f"\n--- {split_name} ({n} 样本) ---")

        if n == 0:
            continue

        if "y_buy_now" in split_df.columns:
            pos = split_df["y_buy_now"].sum()
            print(f"  y_buy_now  正样本率: {pos/n:.4f} ({int(pos)}/{n})")

        if "y_sell_now" in split_df.columns:
            pos = split_df["y_sell_now"].sum()
            print(f"  y_sell_now 正样本率: {pos/n:.4f} ({int(pos)}/{n})")

        dates = split_df["trade_date"]
        date_min = str(dates.min())
        date_max = str(dates.max())
        print(f"  日期范围: {date_min} ~ {date_max}")

    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    if feature_cols:
        nan_counts = df[feature_cols].isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        print(f"\n--- 特征 NaN 概览 ---")
        if nan_cols.empty:
            print("  无 NaN")
        else:
            for col, cnt in nan_cols.items():
                print(f"  {col}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="竞价数据集构建脚本")
    parser.add_argument("--symbol", type=str, default="000001", help="股票代码")
    parser.add_argument("--output-dir", type=str, default=DATASET_DIR, help="输出目录")
    parser.add_argument("--raw-data-dir", type=str, default=RAW_DATA_DIR, help="原始数据目录")
    args = parser.parse_args()

    build_dataset(args.symbol, args.raw_data_dir, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
