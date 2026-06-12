# -*- coding: utf-8 -*-
"""
Purpose: 从分片 parquet 提取事件样本，计算路径标签，构建 GBDT v2 数据集（6个实验）
Inputs:  分片 parquet (freq + pivot_len 定位)
Outputs: results/gbdt/datasets/{experiment_name}.parquet
How to Run:
    python sr_experiment/08_build_gbdt_dataset.py --freq w --pivot-len 10
    python sr_experiment/08_build_gbdt_dataset.py --freq w --pivot-len 10 --experiment A1_trend_opp
Examples:
    python sr_experiment/08_build_gbdt_dataset.py --freq w --pivot-len 10
    python sr_experiment/08_build_gbdt_dataset.py --freq w --pivot-len 10 --experiment B_cluster_opp
Side Effects: 写 parquet 文件到 sr_experiment/results/gbdt/datasets/
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sr_experiment.gbdt_config import (
    BAD_BREAK_HORIZON,
    DATASETS_DIR,
    EXPERIMENT_SPECS,
    PATH_HORIZON,
    SL_PCT_B,
    SL_PCT_OPP,
    TP_PCT_B,
    TP_PCT_OPP,
    TRAIN_END,
    VAL_END,
)
from sr_experiment.gbdt_feature_columns import ALL_DATASET_COLS

_build_mod = importlib.import_module("sr_experiment.00_build_factor_panel_from_db")
iter_shards = _build_mod.iter_shards


def _compute_path_labels_group(group: pd.DataFrame) -> pd.DataFrame:
    """对单只股票的 DataFrame 计算路径标签（逐根前推判断先止盈/先止损）。"""
    close = group["close"].values.astype(float)
    low = group["low"].values.astype(float)
    active_support = group["active_support_ref"].values.astype(float) if "active_support_ref" in group.columns else np.full(len(group), np.nan)
    n = len(group)

    label_tp8 = np.zeros(n, dtype=np.int8)
    label_tp10 = np.zeros(n, dtype=np.int8)
    label_low_mdd = np.zeros(n, dtype=np.int8)
    label_bad_break = np.zeros(n, dtype=np.int8)
    tp_bar = np.full(n, np.nan, dtype=np.float32)
    sl_bar = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        event_close = close[i]
        if np.isnan(event_close) or event_close <= 0:
            continue

        tp_price_8 = event_close * (1 + TP_PCT_OPP)
        sl_price_6 = event_close * (1 - SL_PCT_OPP)
        tp_price_10 = event_close * (1 + TP_PCT_B)
        sl_price_8 = event_close * (1 - SL_PCT_B)
        support_level = active_support[i] if not np.isnan(active_support[i]) else low[i]
        low_mdd_limit = event_close * 0.92

        tp_hit_at = -1
        sl_hit_at = -1
        tp10_hit_at = -1
        sl8_hit_at = -1
        bad_break_at = -1
        all_above_mdd = True

        max_k = min(PATH_HORIZON, n - i - 1)
        for k in range(1, max_k + 1):
            fwd_close = close[i + k]
            fwd_low = low[i + k]

            if np.isnan(fwd_close):
                break

            if tp_hit_at < 0 and fwd_close >= tp_price_8:
                tp_hit_at = k
            if sl_hit_at < 0 and fwd_close <= sl_price_6:
                sl_hit_at = k
            if tp10_hit_at < 0 and fwd_close >= tp_price_10:
                tp10_hit_at = k
            if sl8_hit_at < 0 and fwd_close <= sl_price_8:
                sl8_hit_at = k
            if bad_break_at < 0 and k <= BAD_BREAK_HORIZON:
                if fwd_close < support_level:
                    bad_break_at = k
            if fwd_close < low_mdd_limit:
                all_above_mdd = False

        if tp_hit_at > 0 and (sl_hit_at < 0 or tp_hit_at < sl_hit_at):
            label_tp8[i] = 1
        if tp10_hit_at > 0 and (sl8_hit_at < 0 or tp10_hit_at < sl8_hit_at):
            label_tp10[i] = 1
        if all_above_mdd and max_k >= PATH_HORIZON:
            label_low_mdd[i] = 1
        if bad_break_at > 0:
            label_bad_break[i] = 1
        if tp_hit_at > 0:
            tp_bar[i] = tp_hit_at
        if sl_hit_at > 0:
            sl_bar[i] = sl_hit_at

    group = group.copy()
    group["label_tp8_sl6_20"] = label_tp8
    group["label_tp10_sl8_20"] = label_tp10
    group["label_low_mdd_20"] = label_low_mdd
    group["label_bad_break_10"] = label_bad_break
    group["tp_hit_bar"] = tp_bar
    group["sl_hit_bar"] = sl_bar
    return group


def _compute_path_labels(df: pd.DataFrame) -> pd.DataFrame:
    """按 ts_code 分组计算路径标签。"""
    if "ts_code" not in df.columns:
        return df
    df = df.sort_values(["ts_code", "bar_time"])
    parts = []
    for ts_code, group in df.groupby("ts_code"):
        parts.append(_compute_path_labels_group(group))
    return pd.concat(parts, ignore_index=True)


def build_datasets(freq: str, pivot_len: int, experiment_name: str | None = None):
    Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)

    cols_to_load = [c for c in ALL_DATASET_COLS if c != "ts_code"]
    if "ts_code" not in cols_to_load:
        cols_to_load.append("ts_code")

    print(f"加载分片数据: freq={freq}, pivot_len={pivot_len}")
    parts = []
    for shard_df in iter_shards(freq, pivot_len, columns=cols_to_load, shard_type="panel"):
        parts.append(shard_df)
    if not parts:
        print("无数据")
        return

    all_data = pd.concat(parts, ignore_index=True)
    print(f"总行数: {len(all_data)}")

    if "bar_time" in all_data.columns:
        all_data["bar_time"] = pd.to_datetime(all_data["bar_time"])

    print("计算路径标签...")
    all_data = _compute_path_labels(all_data)

    for label_col in ["label_tp8_sl6_20", "label_tp10_sl8_20", "label_low_mdd_20", "label_bad_break_10"]:
        if label_col in all_data.columns:
            rate = all_data[label_col].mean()
            print(f"  {label_col}: 正样本率={rate:.4f}")

    specs = EXPERIMENT_SPECS
    if experiment_name:
        specs = {k: v for k, v in specs.items() if k == experiment_name}

    for exp_name, spec in specs.items():
        print(f"\n构建实验 {exp_name}: {spec['description']}")
        mask = spec["sample_filter"](all_data)
        df = all_data.loc[mask].copy()
        if df.empty:
            print(f"  无样本，跳过")
            continue

        label_col = spec["label_name"]
        if label_col not in df.columns:
            print(f"  标签列 {label_col} 不存在，跳过")
            continue

        if "bar_time" in df.columns:
            df["split"] = "train"
            df.loc[df["bar_time"] > TRAIN_END, "split"] = "val"
            df.loc[df["bar_time"] > VAL_END, "split"] = "test"
        else:
            df["split"] = "train"

        split_counts = df["split"].value_counts().to_dict()
        pos_rate = df[label_col].mean()
        print(f"  样本数: {len(df)}, 正样本率: {pos_rate:.4f}")
        print(f"  切分: {split_counts}")

        out_path = Path(DATASETS_DIR) / f"{exp_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="构建 GBDT v2 数据集（6个实验）")
    parser.add_argument("--freq", type=str, default="w")
    parser.add_argument("--pivot-len", type=int, default=10)
    parser.add_argument("--experiment", type=str, default=None,
                        choices=list(EXPERIMENT_SPECS.keys()))
    args = parser.parse_args()

    build_datasets(args.freq, args.pivot_len, args.experiment)


if __name__ == "__main__":
    main()
