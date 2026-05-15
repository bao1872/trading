#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线预测 vs 实时推理 一致性验证

Purpose:
    对比离线 full_test_predictions 与 07 实时推理的预测一致性，
    定位 K 线起始点、因子 warm-up、信号查询范围等差异根因。

Inputs:
    - stop_experiment/output/full_test_predictions.parquet (离线预测)
    - DB: stop_loss_selection, stock_k_data (实时推理数据源)
    - stop_experiment/output/models_control/*_final.txt (模型文件)

Outputs:
    - experiments/consistency_check/results/prediction_diff_{date}.csv (逐日预测差异)
    - experiments/consistency_check/results/feature_diff_{date}_{ts_code}.csv (特征差异)
    - experiments/consistency_check/results/consistency_summary.csv (汇总报告)

How to Run:
    python -m stop_experiment.experiments.consistency_check.01_compare_prediction_sources
    python -m stop_experiment.experiments.consistency_check.01_compare_prediction_sources --dates 2026-01-05 2026-02-02
    python -m stop_experiment.experiments.consistency_check.01_compare_prediction_sources --sample-stock 000001.SZ

Side Effects:
    - 只读 DB + parquet + 模型文件，不写入任何表
    - 结果仅写入 experiments/consistency_check/results/
"""

from __future__ import annotations

import sys
import os
import argparse
import logging
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR, OBS_VAL_END, CANDIDATE_OBS_DAYS,
)
from stop_experiment.pipeline.factor_columns import ALL_FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")

PRED_COLS = ["pred_sell_reg", "pred_sell_cls", "pred_buy_reg", "pred_buy_cls"]
DIFF_THRESHOLD = 1e-4
DEFAULT_TEST_DATES = [
    "2026-01-05", "2026-02-02", "2026-03-02", "2026-04-01", "2026-05-06",
]


def _import_mod07():
    mod = importlib.import_module("stop_experiment.pipeline.07_generate_daily_predictions")
    return mod


def load_offline_predictions(dates: list[str]) -> pd.DataFrame:
    ftp_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    if not os.path.exists(ftp_path):
        raise FileNotFoundError(f"离线预测文件不存在: {ftp_path}")
    ftp = pd.read_parquet(ftp_path)
    ftp["obs_date"] = pd.to_datetime(ftp["obs_date"])
    target_dates = pd.to_datetime(dates)
    df = ftp[ftp["obs_date"].isin(target_dates)].copy()
    if "obs_day" in df.columns:
        df = df[df["obs_day"].isin(CANDIDATE_OBS_DAYS)].copy()
        if "ts_code" in df.columns and "obs_date" in df.columns:
            df = df.sort_values("obs_day").drop_duplicates(subset=["ts_code", "obs_date"], keep="first")
    logger.info("离线预测: %d 行 (%s)", len(df), ", ".join(dates))
    return df


def run_realtime_inference(target_date: str, mod07) -> pd.DataFrame:
    from datasource.database import get_engine

    target_dt = pd.to_datetime(target_date)
    engine = get_engine()

    try:
        df_day, kline_dict, msg = mod07._build_candidates(target_dt, engine)
        if df_day is None:
            logger.warning("实时推理 %s: 无候选 (%s)", target_date, msg)
            return pd.DataFrame()
        df_day = mod07._merge_and_derive(df_day, kline_dict)
        df_day = mod07._predict_and_score(df_day)
        logger.info("实时推理 %s: %d 行", target_date, len(df_day))
        return df_day
    finally:
        engine.dispose()


def compare_predictions(offline: pd.DataFrame, realtime: pd.DataFrame, date_str: str) -> dict:
    merge_keys = ["signal_id", "obs_date"]
    for key in merge_keys:
        if key not in offline.columns or key not in realtime.columns:
            logger.warning("合并键 %s 缺失，跳过预测对比", key)
            return {}

    off = offline[offline["obs_date"] == pd.to_datetime(date_str)].copy()
    rt = realtime.copy()
    if "obs_date" in rt.columns:
        rt["obs_date"] = pd.to_datetime(rt["obs_date"])

    merged = off.merge(
        rt[merge_keys + PRED_COLS],
        on=merge_keys, how="inner", suffixes=("_off", "_rt"),
    )
    if merged.empty:
        logger.warning("日期 %s: 离线与实时无交集", date_str)
        return {}

    diff_stats = {"date": date_str, "n_matched": len(merged)}
    for col in PRED_COLS:
        off_col = f"{col}_off"
        rt_col = f"{col}_rt"
        if off_col not in merged.columns or rt_col not in merged.columns:
            diff_stats[f"{col}_diff_pct"] = np.nan
            continue
        diff = (merged[off_col] - merged[rt_col]).abs()
        n_diff = (diff > DIFF_THRESHOLD).sum()
        diff_stats[f"{col}_max_diff"] = diff.max()
        diff_stats[f"{col}_mean_diff"] = diff.mean()
        diff_stats[f"{col}_diff_pct"] = n_diff / len(merged) if len(merged) > 0 else 0.0

    diff_rows = []
    for col in PRED_COLS:
        off_col = f"{col}_off"
        rt_col = f"{col}_rt"
        if off_col not in merged.columns or rt_col not in merged.columns:
            continue
        diff = (merged[off_col] - merged[rt_col]).abs()
        mask = diff > DIFF_THRESHOLD
        if mask.any():
            sub = merged.loc[mask, merge_keys + [off_col, rt_col]].copy()
            sub["pred_col"] = col
            sub.columns = [c.replace("_off", "_offline").replace("_rt", "_realtime") if c not in merge_keys else c for c in sub.columns]
            diff_rows.append(sub)

    if diff_rows:
        diff_df = pd.concat(diff_rows, ignore_index=True)
        diff_path = os.path.join(RESULTS_DIR, f"prediction_diff_{date_str}.csv")
        diff_df.to_csv(diff_path, index=False)
        logger.info("预测差异详情: %s (%d 行)", diff_path, len(diff_df))

    return diff_stats


def compare_top10_overlap(offline: pd.DataFrame, realtime: pd.DataFrame, date_str: str) -> dict:
    off = offline[offline["obs_date"] == pd.to_datetime(date_str)].copy()
    rt = realtime.copy()
    if "obs_date" in rt.columns:
        rt["obs_date"] = pd.to_datetime(rt["obs_date"])
        rt = rt[rt["obs_date"] == pd.to_datetime(date_str)]

    if off.empty or rt.empty:
        return {"date": date_str, "top10_overlap": np.nan}

    off_top = set(off.nlargest(10, "pred_sell_reg")["ts_code"].tolist()) if "pred_sell_reg" in off.columns else set()
    rt_top = set(rt.nlargest(10, "pred_sell_reg")["ts_code"].tolist()) if "pred_sell_reg" in rt.columns else set()

    if not off_top and not rt_top:
        return {"date": date_str, "top10_overlap": np.nan}

    overlap = len(off_top & rt_top)
    union = len(off_top | rt_top)
    jaccard = overlap / union if union > 0 else 0.0

    return {
        "date": date_str,
        "top10_overlap_count": overlap,
        "top10_overlap_jaccard": jaccard,
        "offline_top10": sorted(off_top),
        "realtime_top10": sorted(rt_top),
    }


def compare_features(offline_full: pd.DataFrame, realtime: pd.DataFrame,
                     date_str: str, ts_code: str, mod07) -> pd.DataFrame:
    from datasource.database import get_engine

    rt = realtime.copy()
    if "obs_date" in rt.columns:
        rt["obs_date"] = pd.to_datetime(rt["obs_date"])
    rt_stock = rt[(rt["ts_code"] == ts_code)].copy()

    if rt_stock.empty:
        logger.warning("实时推理中无 %s 的数据", ts_code)
        return pd.DataFrame()

    feature_cols = [c for c in ALL_FEATURE_COLS if c in rt_stock.columns]
    if not feature_cols:
        logger.warning("实时推理数据中无特征列")
        return pd.DataFrame()

    ftp_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")
    if not os.path.exists(ftp_path):
        logger.warning("full_test_predictions.parquet 不存在，无法对比特征")
        return pd.DataFrame()

    scores_path = os.path.join(OUTPUT_DIR, "models_control", "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        logger.warning("candidate_with_scores.parquet 不存在，无法对比特征")
        return pd.DataFrame()

    offline_all = pd.read_parquet(scores_path)
    offline_all["obs_date"] = pd.to_datetime(offline_all["obs_date"])
    off_stock = offline_all[
        (offline_all["ts_code"] == ts_code) &
        (offline_all["obs_date"] == pd.to_datetime(date_str))
    ].copy()

    if off_stock.empty:
        logger.warning("离线数据中无 %s @ %s", ts_code, date_str)
        return pd.DataFrame()

    off_row = off_stock.iloc[0:1]
    rt_row = rt_stock.iloc[0:1]

    rows = []
    for col in feature_cols:
        off_val = off_row[col].values[0] if col in off_row.columns else np.nan
        rt_val = rt_row[col].values[0] if col in rt_row.columns else np.nan
        diff = abs(off_val - rt_val) if pd.notna(off_val) and pd.notna(rt_val) else np.nan
        rows.append({
            "feature": col,
            "offline_value": off_val,
            "realtime_value": rt_val,
            "abs_diff": diff,
            "exceeds_threshold": diff > DIFF_THRESHOLD if pd.notna(diff) else False,
        })

    feat_df = pd.DataFrame(rows)
    feat_path = os.path.join(RESULTS_DIR, f"feature_diff_{date_str}_{ts_code.replace('.', '_')}.csv")
    feat_df.to_csv(feat_path, index=False)
    logger.info("特征差异: %s (%d 列)", feat_path, len(feat_df))

    n_diff = feat_df["exceeds_threshold"].sum()
    logger.info("特征差异汇总: %d/%d 列超过阈值 %.1e", n_diff, len(feat_df), DIFF_THRESHOLD)

    return feat_df


def diagnose_root_cause(offline: pd.DataFrame, realtime: pd.DataFrame,
                        date_str: str, mod07) -> dict:
    from datasource.database import get_engine
    from stop_experiment.pipeline.stop_config import OBS_DAYS

    diagnosis = {"date": date_str}
    target_dt = pd.to_datetime(date_str)
    engine = get_engine()

    try:
        signal_lookback = OBS_DAYS + 5
        signal_start_rt = (target_dt - pd.Timedelta(days=signal_lookback)).strftime("%Y-%m-%d")
        signal_end_rt = target_dt.strftime("%Y-%m-%d")

        from sqlalchemy import text
        with engine.connect() as conn:
            rt_signal_count = pd.read_sql(
                text(
                    "SELECT COUNT(*) as cnt FROM stop_loss_selection "
                    "WHERE signal_date >= :sd AND signal_date <= :ed"
                ),
                conn, params={"sd": signal_start_rt, "ed": signal_end_rt},
            ).iloc[0]["cnt"]

        diagnosis["rt_signal_lookback_days"] = signal_lookback
        diagnosis["rt_signal_count"] = int(rt_signal_count)
        diagnosis["rt_signal_range"] = f"{signal_start_rt} ~ {signal_end_rt}"

        rt_kline_start = (target_dt - pd.Timedelta(days=1300)).strftime("%Y-%m-%d")
        diagnosis["rt_kline_start"] = rt_kline_start
        diagnosis["rt_kline_range_days"] = 1300

        scores_path = os.path.join(OUTPUT_DIR, "models_control", "candidate_with_scores.parquet")
        if os.path.exists(scores_path):
            offline_all = pd.read_parquet(scores_path)
            offline_all["obs_date"] = pd.to_datetime(offline_all["obs_date"])
            off_day = offline_all[offline_all["obs_date"] == target_dt]
            if not off_day.empty and "selection_date" in off_day.columns:
                off_min_sel = pd.to_datetime(off_day["selection_date"]).min()
                off_kline_start = (off_min_sel - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
                diagnosis["train_kline_start"] = off_kline_start
                diagnosis["train_kline_offset_days"] = 150
                diagnosis["kline_start_diff_days"] = (
                    pd.to_datetime(rt_kline_start) - pd.to_datetime(off_kline_start)
                ).days

        off_signal_ids = set(offline[offline["obs_date"] == target_dt]["signal_id"].tolist()) if "signal_id" in offline.columns else set()
        rt_signal_ids = set(realtime["signal_id"].tolist()) if "signal_id" in realtime.columns else set()
        diagnosis["offline_signal_count"] = len(off_signal_ids)
        diagnosis["realtime_signal_count"] = len(rt_signal_ids)
        diagnosis["signal_overlap"] = len(off_signal_ids & rt_signal_ids)
        diagnosis["signal_only_offline"] = len(off_signal_ids - rt_signal_ids)
        diagnosis["signal_only_realtime"] = len(rt_signal_ids - off_signal_ids)

    finally:
        engine.dispose()

    return diagnosis


def run_consistency_check(dates: list[str], sample_stock: str | None = None) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    mod07 = _import_mod07()

    logger.info("加载离线预测...")
    offline = load_offline_predictions(dates)

    all_pred_stats = []
    all_top10_stats = []
    all_diagnosis = []

    for date_str in dates:
        logger.info("=" * 60)
        logger.info("对比日期: %s", date_str)
        logger.info("=" * 60)

        rt = run_realtime_inference(date_str, mod07)
        if rt.empty:
            logger.warning("日期 %s: 实时推理无结果，跳过", date_str)
            continue

        pred_stats = compare_predictions(offline, rt, date_str)
        if pred_stats:
            all_pred_stats.append(pred_stats)
            logger.info(
                "预测差异: matched=%d, sell_reg_diff_pct=%.2f%%, sell_cls_diff_pct=%.2f%%",
                pred_stats.get("n_matched", 0),
                pred_stats.get("pred_sell_reg_diff_pct", 0) * 100,
                pred_stats.get("pred_sell_cls_diff_pct", 0) * 100,
            )

        top10_stats = compare_top10_overlap(offline, rt, date_str)
        all_top10_stats.append(top10_stats)
        logger.info(
            "Top10 重叠: count=%d, jaccard=%.3f",
            top10_stats.get("top10_overlap_count", 0),
            top10_stats.get("top10_overlap_jaccard", 0),
        )

        diagnosis = diagnose_root_cause(offline, rt, date_str, mod07)
        all_diagnosis.append(diagnosis)
        logger.info(
            "根因诊断: 信号重叠=%d, K线起始差异=%d天",
            diagnosis.get("signal_overlap", 0),
            diagnosis.get("kline_start_diff_days", "N/A"),
        )

        stock_for_feat = sample_stock
        if stock_for_feat is None and not rt.empty:
            stock_for_feat = rt.iloc[0]["ts_code"]
        if stock_for_feat:
            compare_features(offline, rt, date_str, stock_for_feat, mod07)

    summary_path = os.path.join(RESULTS_DIR, "consistency_summary.csv")
    if all_pred_stats:
        summary_df = pd.DataFrame(all_pred_stats)
        summary_df.to_csv(summary_path, index=False)
        logger.info("汇总报告: %s", summary_path)

    if all_diagnosis:
        diag_path = os.path.join(RESULTS_DIR, "diagnosis_summary.csv")
        diag_df = pd.DataFrame(all_diagnosis)
        diag_df.to_csv(diag_path, index=False)
        logger.info("根因诊断: %s", diag_path)

    return {
        "pred_stats": all_pred_stats,
        "top10_stats": all_top10_stats,
        "diagnosis": all_diagnosis,
    }


def main():
    parser = argparse.ArgumentParser(description="离线预测 vs 实时推理一致性验证")
    parser.add_argument(
        "--dates", nargs="+", default=DEFAULT_TEST_DATES,
        help=f"测试日期列表 (默认: {' '.join(DEFAULT_TEST_DATES)})",
    )
    parser.add_argument(
        "--sample-stock", default=None,
        help="特征对比用的股票代码 (默认取实时推理第一只)",
    )
    args = parser.parse_args()

    logger.info("一致性验证: dates=%s, sample_stock=%s", args.dates, args.sample_stock)
    results = run_consistency_check(args.dates, args.sample_stock)

    logger.info("=" * 60)
    logger.info("一致性验证完成")
    logger.info("=" * 60)

    if results["pred_stats"]:
        for stat in results["pred_stats"]:
            logger.info(
                "  %s: matched=%d, sell_reg_max_diff=%.6f, sell_reg_diff_pct=%.2f%%",
                stat["date"], stat.get("n_matched", 0),
                stat.get("pred_sell_reg_max_diff", 0),
                stat.get("pred_sell_reg_diff_pct", 0) * 100,
            )

    if results["diagnosis"]:
        any_signal_mismatch = any(d.get("signal_only_offline", 0) > 0 or d.get("signal_only_realtime", 0) > 0 for d in results["diagnosis"])
        any_kline_diff = any(d.get("kline_start_diff_days", 0) != 0 for d in results["diagnosis"] if "kline_start_diff_days" in d)
        if any_signal_mismatch:
            logger.info("  ⚠ 信号范围不一致，需检查 SIGNAL_LOOKBACK_DAYS 差异")
        if any_kline_diff:
            logger.info("  ⚠ K线起始点不一致，可能是因子 warm-up 差异的根因")
        if not any_signal_mismatch and not any_kline_diff:
            logger.info("  ✓ 信号范围和 K 线起始点一致")


if __name__ == "__main__":
    main()
