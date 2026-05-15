# -*- coding: utf-8 -*-
"""
Prediction Store 读写模块 — 预测制品的唯一真相源

Purpose:
    提供版本化的 prediction artifact 读写接口：
    - write: 写入 prediction_store/{profile}/{fv}/{mv}/{date}.parquet + manifest.json
    - read:  从 prediction_store 加载数据并构建 pred_lookup
    - 所有模式 (backtest/replay/live) 统一从此读取预测

Usage:
    from stop_experiment.registries.prediction_store import (
        write_prediction_artifact, read_prediction_store_range,
    )

Side Effects:
    - 写 parquet 文件 + manifest.json 到 output/prediction_store/
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from stop_experiment.registries import resolve_prediction_store_path, resolve_manifest_path

_STOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_prediction_artifact(
    profile_name: str,
    date: str,
    df: pd.DataFrame,
    prediction_source: str = "live",
) -> str:
    """写入预测制品到 prediction_store"""
    artifact_path = resolve_prediction_store_path(profile_name, date)
    _ensure_dir(artifact_path)
    df.to_parquet(artifact_path, index=False)

    manifest_path = resolve_manifest_path(profile_name)
    _ensure_dir(manifest_path)

    meta = {
        "prediction_job_id": f"pj_{datetime.now().strftime('%Y%m%d')}_{profile_name}_{date}",
        "profile_name": profile_name,
        "target_date": date,
        "prediction_source": prediction_source,
        "created_at": datetime.now().isoformat(),
        "file": os.path.basename(artifact_path),
    }

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    manifest.setdefault("artifacts", []).append(meta)
    manifest["last_updated"] = datetime.now().isoformat()

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    return artifact_path


def read_prediction_artifact(profile_name: str, date: str) -> pd.DataFrame | None:
    """读取单个预测制品"""
    path = resolve_prediction_store_path(profile_name, date)
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def read_prediction_store_range(
    profile_name: str,
    dates: list,
    score_col: str = "sell_score",
) -> tuple[pd.DataFrame, dict]:
    """从 prediction_store 加载指定日期范围的预测并构建 pred_lookup。

    生产口径 obs_day∈CANDIDATE_OBS_DAYS，同一 (ts_code, obs_date) 保留 obs_day 最小（最新信号）。
    pred_lookup 同时建立 (signal_id, obs_date) 和 (ts_code, obs_date) 两种 key，
    支持 find_exit_pred 的 ts_code fallback 查找。
    """
    from stop_experiment.pipeline.stop_config import CANDIDATE_OBS_DAYS

    all_dfs = []
    pred_lookup = {}
    import numpy as np

    for d in dates:
        ds = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        df = read_prediction_artifact(profile_name, ds)
        if df is None:
            continue

        if "obs_date" not in df.columns and "pred_date" in df.columns:
            df["obs_date"] = pd.to_datetime(df["pred_date"])
        elif "obs_date" not in df.columns:
            df["obs_date"] = pd.to_datetime(ds)
        if "obs_day" not in df.columns:
            df["obs_day"] = 1
        df = df[df["obs_day"].isin(CANDIDATE_OBS_DAYS)].copy()
        if "ts_code" in df.columns and "obs_date" in df.columns:
            df = df.sort_values("obs_day").drop_duplicates(subset=["ts_code", "obs_date"], keep="first")
        if "score" not in df.columns:
            from stop_experiment.backtest.simple_backtest import score_stocks
            df = score_stocks(df, score_col)
        all_dfs.append(df)

        for _, row in df.iterrows():
            pred_dict = {
                "pred_buy_cls": float(row.get("pred_buy_cls", np.nan)),
                "pred_sell_reg": float(row.get("pred_sell_reg", np.nan)),
                "pred_sell_cls": float(row.get("pred_sell_cls", np.nan)),
                "pred_buy_reg": float(row.get("pred_buy_reg", np.nan)),
                "composite_score": float(row.get("score", np.nan)),
            }
            sid_key = (int(row["signal_id"]), row["obs_date"])
            pred_lookup[sid_key] = pred_dict
            ts_code = row.get("ts_code")
            if ts_code:
                ts_key = (ts_code, row["obs_date"])
                pred_lookup[ts_key] = pred_dict

    if not all_dfs:
        return pd.DataFrame(), pred_lookup
    df_all = pd.concat(all_dfs, ignore_index=True)
    if "obs_date" in df_all.columns:
        df_all["obs_date"] = pd.to_datetime(df_all["obs_date"])
    return df_all, pred_lookup


def get_prediction_store_available_dates(profile_name: str) -> list:
    """获取 prediction_store 中已有的预测日期列表"""
    result = []
    from stop_experiment.registries import resolve_manifest_path
    manifest_path = resolve_manifest_path(profile_name)
    store_dir = os.path.dirname(manifest_path)
    if not os.path.exists(store_dir):
        return result
    for f in sorted(os.listdir(store_dir)):
        if f.endswith(".parquet"):
            result.append(f.replace(".parquet", ""))
    return result