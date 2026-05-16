#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线预测 vs 实时推理 一致性验证

Purpose:
    对比离线 production_test_predictions 与 07 实时推理的预测一致性，
    检查候选池覆盖率、obs_day 一致性、预测差异、排序一致性，
    定位 K 线起始点、因子 warm-up、信号查询范围等差异根因。

Inputs:
    - stop_experiment/output/production_test_predictions.parquet (离线预测，优先)
    - stop_experiment/output/full_test_predictions.parquet (离线预测，回退)
    - DB: stop_loss_selection, stock_k_data (实时推理数据源)
    - stop_experiment/output/models_control/*_final.txt (模型文件)

Outputs:
    - experiments/consistency_check/results/consistency_summary.csv (汇总)
    - experiments/consistency_check/results/prediction_diff_{date}.csv (预测差异详情)
    - experiments/consistency_check/results/offline_only_{date}.csv (仅离线有)
    - experiments/consistency_check/results/realtime_only_{date}.csv (仅实时有)
    - experiments/consistency_check/results/obs_day_mismatch_{date}.csv (obs_day 不一致)
    - experiments/consistency_check/results/diagnosis_summary.csv (根因诊断)
    - experiments/consistency_check/results/verdict.json (判定结果)

How to Run:
    python -m stop_experiment.experiments.consistency_check.01_compare_prediction_sources
    python -m stop_experiment.experiments.consistency_check.01_compare_prediction_sources --dates 2026-01-05 2026-02-02
    python -m stop_experiment.experiments.consistency_check.01_compare_prediction_sources --dates 2026-01-05 --sample-stock 000001.SZ

Side Effects:
    - 只读 DB + parquet + 模型文件，不写入任何表
    - 结果仅写入 experiments/consistency_check/results/
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import logging
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from stop_experiment.pipeline.stop_config import (
    OUTPUT_DIR,
    OBS_VAL_END,
    OBS_DAYS,
    CANDIDATE_OBS_DAYS,
    FACTOR_WARMUP_DAYS,
    filter_production_candidates,
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
DEFAULT_TEST_DATES = [
    "2026-01-05", "2026-02-02", "2026-03-02", "2026-04-01", "2026-05-06",
]


def _import_mod07():
    return importlib.import_module("stop_experiment.pipeline.07_generate_daily_predictions")


def load_offline_predictions(dates: list[str]) -> pd.DataFrame:
    prod_path = os.path.join(OUTPUT_DIR, "production_test_predictions.parquet")
    full_path = os.path.join(OUTPUT_DIR, "full_test_predictions.parquet")

    if os.path.exists(prod_path):
        logger.info("读取 production_test_predictions.parquet")
        df = pd.read_parquet(prod_path)
    elif os.path.exists(full_path):
        logger.info("production_test_predictions.parquet 不存在，回退 full_test_predictions.parquet")
        df = pd.read_parquet(full_path)
    else:
        raise FileNotFoundError(
            f"离线预测文件不存在: {prod_path} 或 {full_path}"
        )

    df["obs_date"] = pd.to_datetime(df["obs_date"])
    target_dates = pd.to_datetime(dates)
    df = df[df["obs_date"].isin(target_dates)].copy()

    df = filter_production_candidates(df)

    logger.info("离线预测(生产口径): %d 行 (%s)", len(df), ", ".join(dates))
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
        df_day = filter_production_candidates(df_day)
        logger.info("实时推理 %s(生产口径): %d 行", target_date, len(df_day))
        return df_day
    finally:
        engine.dispose()


def _compute_pred_diff_stats(merged: pd.DataFrame) -> dict:
    stats = {}
    for col in PRED_COLS:
        off_col = f"{col}_off"
        rt_col = f"{col}_rt"
        if off_col not in merged.columns or rt_col not in merged.columns:
            for suffix in ["mean_abs_diff", "p95_abs_diff", "p99_abs_diff",
                           "max_abs_diff", "diff_gt_1e_6_rate", "diff_gt_1e_5_rate"]:
                stats[f"{col}_{suffix}"] = np.nan
            continue
        diff = (merged[off_col] - merged[rt_col]).abs()
        n = len(diff)
        stats[f"{col}_mean_abs_diff"] = diff.mean()
        stats[f"{col}_p95_abs_diff"] = diff.quantile(0.95)
        stats[f"{col}_p99_abs_diff"] = diff.quantile(0.99)
        stats[f"{col}_max_abs_diff"] = diff.max()
        stats[f"{col}_diff_gt_1e_6_rate"] = (diff > 1e-6).sum() / n if n > 0 else 0.0
        stats[f"{col}_diff_gt_1e_5_rate"] = (diff > 1e-5).sum() / n if n > 0 else 0.0
    return stats


def _compute_rank_consistency(merged: pd.DataFrame) -> dict:
    result = {}
    if "pred_sell_reg_off" not in merged.columns or "pred_sell_reg_rt" not in merged.columns:
        result["top10_jaccard"] = np.nan
        result["top20_jaccard"] = np.nan
        result["spearman_corr"] = np.nan
        return result

    off_sorted = merged.sort_values("pred_sell_reg_off", ascending=False)
    rt_sorted = merged.sort_values("pred_sell_reg_rt", ascending=False)

    off_top10 = set(off_sorted["ts_code"].head(10).tolist())
    rt_top10 = set(rt_sorted["ts_code"].head(10).tolist())
    off_top20 = set(off_sorted["ts_code"].head(20).tolist())
    rt_top20 = set(rt_sorted["ts_code"].head(20).tolist())

    def _jaccard(a: set, b: set) -> float:
        union = len(a | b)
        return len(a & b) / union if union > 0 else 0.0

    result["top10_jaccard"] = _jaccard(off_top10, rt_top10)
    result["top20_jaccard"] = _jaccard(off_top20, rt_top20)

    if len(merged) >= 3:
        corr, _ = scipy_stats.spearmanr(
            merged["pred_sell_reg_off"].values,
            merged["pred_sell_reg_rt"].values,
        )
        result["spearman_corr"] = corr
    else:
        result["spearman_corr"] = np.nan

    return result


def compare_predictions(offline: pd.DataFrame, realtime: pd.DataFrame, date_str: str) -> dict:
    merge_keys = ["ts_code", "obs_date"]
    for key in merge_keys:
        if key not in offline.columns or key not in realtime.columns:
            logger.warning("合并键 %s 缺失，跳过预测对比", key)
            return {}

    off = offline[offline["obs_date"] == pd.to_datetime(date_str)].copy()
    rt = realtime.copy()
    if "obs_date" in rt.columns:
        rt["obs_date"] = pd.to_datetime(rt["obs_date"])

    rt_merge_cols = merge_keys + PRED_COLS
    if "obs_day" in rt.columns:
        rt_merge_cols = rt_merge_cols + ["obs_day"]

    merged = off.merge(
        rt[rt_merge_cols],
        on=merge_keys, how="outer", indicator=True,
        suffixes=("_off", "_rt"),
    )

    offline_only = merged[merged["_merge"] == "left_only"].copy()
    realtime_only = merged[merged["_merge"] == "right_only"].copy()
    both = merged[merged["_merge"] == "both"].copy()

    offline_rows = len(off)
    realtime_rows = len(rt)
    matched_rows = len(both)
    offline_only_rows = len(offline_only)
    realtime_only_rows = len(realtime_only)
    offline_match_rate = matched_rows / offline_rows if offline_rows > 0 else 0.0
    realtime_match_rate = matched_rows / realtime_rows if realtime_rows > 0 else 0.0

    obs_day_diff_rate = 0.0
    obs_day_mismatch = pd.DataFrame()
    if "obs_day_off" in both.columns and "obs_day_rt" in both.columns:
        mask = both["obs_day_off"] != both["obs_day_rt"]
        obs_day_mismatch = both[mask].copy()
        obs_day_diff_rate = len(obs_day_mismatch) / matched_rows if matched_rows > 0 else 0.0
    elif "obs_day" in both.columns:
        obs_day_mismatch = pd.DataFrame()
        obs_day_diff_rate = 0.0

    summary = {
        "date": date_str,
        "offline_rows": offline_rows,
        "realtime_rows": realtime_rows,
        "matched_rows": matched_rows,
        "offline_only_rows": offline_only_rows,
        "realtime_only_rows": realtime_only_rows,
        "offline_match_rate": offline_match_rate,
        "realtime_match_rate": realtime_match_rate,
        "obs_day_diff_rate": obs_day_diff_rate,
    }

    if obs_day_diff_rate > 0:
        summary["obs_day_check"] = "FAIL"
    else:
        summary["obs_day_check"] = "PASS"

    if not both.empty:
        pred_stats = _compute_pred_diff_stats(both)
        summary.update(pred_stats)

        rank_stats = _compute_rank_consistency(both)
        summary.update(rank_stats)

        diff_rows = []
        for col in PRED_COLS:
            off_col = f"{col}_off"
            rt_col = f"{col}_rt"
            if off_col not in both.columns or rt_col not in both.columns:
                continue
            diff = (both[off_col] - both[rt_col]).abs()
            mask = diff > 1e-6
            if mask.any():
                sub = both.loc[mask, merge_keys + [off_col, rt_col]].copy()
                sub["pred_col"] = col
                sub = sub.rename(columns={
                    off_col: f"{col}_offline",
                    rt_col: f"{col}_realtime",
                })
                diff_rows.append(sub)

        if diff_rows:
            diff_df = pd.concat(diff_rows, ignore_index=True)
            diff_path = os.path.join(RESULTS_DIR, f"prediction_diff_{date_str}.csv")
            diff_df.to_csv(diff_path, index=False)
            logger.info("预测差异详情: %s (%d 行)", diff_path, len(diff_df))

    if not offline_only.empty:
        path = os.path.join(RESULTS_DIR, f"offline_only_{date_str}.csv")
        offline_only.to_csv(path, index=False)
        logger.info("仅离线有: %s (%d 行)", path, len(offline_only))

    if not realtime_only.empty:
        path = os.path.join(RESULTS_DIR, f"realtime_only_{date_str}.csv")
        realtime_only.to_csv(path, index=False)
        logger.info("仅实时有: %s (%d 行)", path, len(realtime_only))

    if not obs_day_mismatch.empty:
        path = os.path.join(RESULTS_DIR, f"obs_day_mismatch_{date_str}.csv")
        obs_day_mismatch.to_csv(path, index=False)
        logger.info("obs_day 不一致: %s (%d 行)", path, len(obs_day_mismatch))

    return summary


def diagnose_root_cause(offline: pd.DataFrame, realtime: pd.DataFrame,
                        date_str: str, mod07) -> dict:
    from datasource.database import get_engine

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

        rt_kline_start = (target_dt - pd.Timedelta(days=FACTOR_WARMUP_DAYS)).strftime("%Y-%m-%d")
        diagnosis["rt_kline_start"] = rt_kline_start
        diagnosis["rt_kline_range_days"] = FACTOR_WARMUP_DAYS

        scores_path = os.path.join(OUTPUT_DIR, "models_control", "candidate_with_scores.parquet")
        if os.path.exists(scores_path):
            offline_all = pd.read_parquet(scores_path)
            offline_all["obs_date"] = pd.to_datetime(offline_all["obs_date"])
            off_day = offline_all[offline_all["obs_date"] == target_dt]
            if not off_day.empty and "selection_date" in off_day.columns:
                off_min_sel = pd.to_datetime(off_day["selection_date"]).min()
                off_kline_start = (off_min_sel - pd.Timedelta(days=FACTOR_WARMUP_DAYS)).strftime("%Y-%m-%d")
                diagnosis["train_kline_start"] = off_kline_start
                diagnosis["train_kline_offset_days"] = FACTOR_WARMUP_DAYS
                diagnosis["kline_start_diff_days"] = (
                    pd.to_datetime(rt_kline_start) - pd.to_datetime(off_kline_start)
                ).days

        off_signal_ids = set(
            offline[offline["obs_date"] == target_dt]["signal_id"].tolist()
        ) if "signal_id" in offline.columns else set()
        rt_signal_ids = set(realtime["signal_id"].tolist()) if "signal_id" in realtime.columns else set()
        diagnosis["offline_signal_count"] = len(off_signal_ids)
        diagnosis["realtime_signal_count"] = len(rt_signal_ids)
        diagnosis["signal_overlap"] = len(off_signal_ids & rt_signal_ids)
        diagnosis["signal_only_offline"] = len(off_signal_ids - rt_signal_ids)
        diagnosis["signal_only_realtime"] = len(rt_signal_ids - off_signal_ids)

    finally:
        engine.dispose()

    return diagnosis


def _compute_verdict(all_summary: list[dict]) -> str:
    if not all_summary:
        return "FAIL_NO_DATA"

    for s in all_summary:
        offline_match_rate = s.get("offline_match_rate", 0)
        realtime_match_rate = s.get("realtime_match_rate", 0)
        obs_day_diff_rate = s.get("obs_day_diff_rate", 1)

        if offline_match_rate < 0.99 or realtime_match_rate < 0.99:
            return "FAIL_CANDIDATE_MISMATCH"

        if obs_day_diff_rate > 0:
            return "FAIL_OBS_DAY_MISMATCH"

    for s in all_summary:
        for col in PRED_COLS:
            p99 = s.get(f"{col}_p99_abs_diff", float("inf"))
            if pd.notna(p99) and p99 >= 1e-6:
                return "FAIL_PREDICTION_DIFF"

    for s in all_summary:
        top10_jaccard = s.get("top10_jaccard", 0)
        if pd.notna(top10_jaccard) and top10_jaccard < 0.9:
            return "FAIL_PREDICTION_DIFF"

    return "PASS"


def run_consistency_check(dates: list[str], sample_stock: str | None = None) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    mod07 = _import_mod07()

    logger.info("加载离线预测...")
    offline = load_offline_predictions(dates)

    all_summary = []
    all_diagnosis = []

    for date_str in dates:
        logger.info("=" * 60)
        logger.info("对比日期: %s", date_str)
        logger.info("=" * 60)

        rt = run_realtime_inference(date_str, mod07)
        if rt.empty:
            logger.warning("日期 %s: 实时推理无结果，跳过", date_str)
            continue

        summary = compare_predictions(offline, rt, date_str)
        if summary:
            all_summary.append(summary)
            logger.info(
                "候选池: offline=%d, realtime=%d, matched=%d, "
                "offline_match_rate=%.4f, realtime_match_rate=%.4f",
                summary.get("offline_rows", 0),
                summary.get("realtime_rows", 0),
                summary.get("matched_rows", 0),
                summary.get("offline_match_rate", 0),
                summary.get("realtime_match_rate", 0),
            )
            logger.info(
                "obs_day_diff_rate=%.4f, obs_day_check=%s",
                summary.get("obs_day_diff_rate", 0),
                summary.get("obs_day_check", "?"),
            )
            logger.info(
                "pred_sell_reg: mean_abs=%.2e, p99_abs=%.2e, max_abs=%.2e",
                summary.get("pred_sell_reg_mean_abs_diff", 0),
                summary.get("pred_sell_reg_p99_abs_diff", 0),
                summary.get("pred_sell_reg_max_abs_diff", 0),
            )
            logger.info(
                "排序一致性: top10_jaccard=%.3f, top20_jaccard=%.3f, spearman=%.4f",
                summary.get("top10_jaccard", 0),
                summary.get("top20_jaccard", 0),
                summary.get("spearman_corr", 0),
            )

        diagnosis = diagnose_root_cause(offline, rt, date_str, mod07)
        all_diagnosis.append(diagnosis)
        logger.info(
            "根因诊断: 信号重叠=%d, K线起始差异=%s天",
            diagnosis.get("signal_overlap", 0),
            diagnosis.get("kline_start_diff_days", "N/A"),
        )

    if all_summary:
        summary_path = os.path.join(RESULTS_DIR, "consistency_summary.csv")
        summary_df = pd.DataFrame(all_summary)
        summary_df.to_csv(summary_path, index=False)
        logger.info("汇总报告: %s", summary_path)

    if all_diagnosis:
        diag_path = os.path.join(RESULTS_DIR, "diagnosis_summary.csv")
        diag_df = pd.DataFrame(all_diagnosis)
        diag_df.to_csv(diag_path, index=False)
        logger.info("根因诊断: %s", diag_path)

    verdict = _compute_verdict(all_summary)
    verdict_data = {
        "verdict": verdict,
        "test_dates": dates,
        "n_dates_tested": len(all_summary),
        "details": all_summary,
    }
    verdict_path = os.path.join(RESULTS_DIR, "verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict_data, f, ensure_ascii=False, indent=2, default=str)
    logger.info("判定结果: %s → %s", verdict_path, verdict)

    return {
        "summary": all_summary,
        "diagnosis": all_diagnosis,
        "verdict": verdict,
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
    logger.info("一致性验证完成 — verdict: %s", results["verdict"])
    logger.info("=" * 60)

    for stat in results["summary"]:
        logger.info(
            "  %s: matched=%d, offline_match_rate=%.4f, realtime_match_rate=%.4f, "
            "obs_day_diff_rate=%.4f, sell_reg_p99=%.2e, top10_jaccard=%.3f",
            stat["date"],
            stat.get("matched_rows", 0),
            stat.get("offline_match_rate", 0),
            stat.get("realtime_match_rate", 0),
            stat.get("obs_day_diff_rate", 0),
            stat.get("pred_sell_reg_p99_abs_diff", 0),
            stat.get("top10_jaccard", 0),
        )

    if results["verdict"] == "PASS":
        logger.info("✓ 一致性验证通过")
    else:
        logger.info("✗ 一致性验证失败: %s", results["verdict"])


if __name__ == "__main__":
    main()
