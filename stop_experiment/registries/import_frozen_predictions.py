#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入 frozen predictions 到 Prediction Store

Purpose:
    将 output/full_test_predictions.parquet 按 obs_date 拆分为每日制品，
    写入 prediction_store/production/fv_202605_v1/mv_20260510_3y_v1/，
    并在 manifest.json 中标记 prediction_source=frozen。

Usage:
    python -m stop_experiment.registries.import_frozen_predictions
    python -m stop_experiment.registries.import_frozen_predictions --verify --sample 3

Side Effects:
    - 写入 prediction_store/ 下的 parquet 文件
    - 更新 manifest.json
    - 不修改 full_test_predictions.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stop_experiment.registries import resolve_prediction_store_path, resolve_manifest_path, load_profile
from stop_experiment.pipeline.stop_config import PRODUCTION_PARAMS, CANDIDATE_OBS_DAYS

_STOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FULL_TEST_PATH = os.path.join(_STOP_ROOT, "output", "full_test_predictions.parquet")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="导入 frozen predictions 到 Prediction Store")
    parser.add_argument("--verify", action="store_true", help="导入后抽样验证")
    parser.add_argument("--sample", type=int, default=3, help="验证抽样数量")
    parser.add_argument("--profile", type=str, default="production", help="目标 profile")
    args = parser.parse_args()

    if not os.path.exists(_FULL_TEST_PATH):
        print(f"❌ 文件不存在: {_FULL_TEST_PATH}")
        print("  请先运行 run_training.py 生成全量预测")
        sys.exit(1)

    print("=" * 70)
    print("导入 frozen predictions → Prediction Store")
    print(f"  源: {_FULL_TEST_PATH}")
    print(f"  目标 profile: {args.profile}")
    print("=" * 70)

    df_full = pd.read_parquet(_FULL_TEST_PATH)
    print(f"\n  加载: {len(df_full)} 行, {len(df_full.columns)} 列")

    if "obs_date" in df_full.columns:
        df_full["obs_date"] = pd.to_datetime(df_full["obs_date"])
    elif "pred_date" in df_full.columns:
        df_full["obs_date"] = pd.to_datetime(df_full["pred_date"])
    else:
        print("❌ 缺少 obs_date/pred_date 列，无法按日期拆分")
        sys.exit(1)

    if "obs_day" in df_full.columns:
        df_full = df_full[df_full["obs_day"].isin(CANDIDATE_OBS_DAYS)].copy()
        if "ts_code" in df_full.columns and "obs_date" in df_full.columns:
            df_full = df_full.sort_values("obs_day").drop_duplicates(subset=["ts_code", "obs_date"], keep="first")

    date_groups = df_full.groupby(df_full["obs_date"].dt.strftime("%Y-%m-%d"))
    dates = sorted(date_groups.groups.keys())
    print(f"  日期: {len(dates)} 天 ({dates[0]} ~ {dates[-1]})")

    manifest_path = resolve_manifest_path(args.profile)
    _ensure_dir(manifest_path)

    existing = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    imported = 0
    skipped = 0
    for date_str in dates:
        artifact_path = resolve_prediction_store_path(args.profile, date_str)
        if os.path.exists(artifact_path):
            skipped += 1
            continue

        _ensure_dir(artifact_path)
        df_day = date_groups.get_group(date_str).copy()
        df_day.to_parquet(artifact_path, index=False)
        imported += 1

        meta = {
            "prediction_job_id": f"pj_import_frozen_{date_str}",
            "profile_name": args.profile,
            "target_date": date_str,
            "prediction_source": "frozen",
            "created_at": datetime.now().isoformat(),
            "file": os.path.basename(artifact_path),
            "rows": len(df_day),
        }
        if "artifacts" not in existing:
            existing["artifacts"] = []
        existing["artifacts"].append(meta)

    if imported > 0:
        existing["last_updated"] = datetime.now().isoformat()
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  导入: {imported} 天, 跳过(已存在): {skipped} 天")

    if args.verify:
        print(f"\n{'='*70}")
        print(f"抽样验证 ({min(args.sample, len(dates))} / {len(dates)} 天)")
        import random
        random.seed(42)
        sample_dates = sorted(random.sample(dates, min(args.sample, len(dates))))

        all_ok = True
        for ds in sample_dates:
            artifact_path = resolve_prediction_store_path(args.profile, ds)
            original = date_groups.get_group(ds)
            imported_df = pd.read_parquet(artifact_path)

            same_rows = len(original) == len(imported_df)
            orig_cols = set(original.columns)
            imp_cols = set(imported_df.columns)
            same_cols = orig_cols == imp_cols

            status = "✅" if (same_rows and same_cols) else "❌"
            print(f"  {status} {ds}: {len(original)}→{len(imported_df)} 行, "
                  f"列{'一致' if same_cols else f'差异:{orig_cols.symmetric_difference(imp_cols)}'}")
            if not (same_rows and same_cols):
                all_ok = False

        if all_ok:
            print(f"\n  ✅ 全部抽样验证通过")
        else:
            print(f"\n  ❌ 部分抽样验证失败")

    print(f"\n✅ 完成")


if __name__ == "__main__":
    main()