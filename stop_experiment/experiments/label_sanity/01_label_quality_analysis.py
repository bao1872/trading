#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFE 标签质量验证: 检查 mfe_20 是否过于理想化，以及模型是否学到波动率而非可交易收益

Purpose:
    验证 mfe_20 标签的实盘可达性，量化"先大亏后回本"和"冲高但收盘没涨"的不可交易比例，
    分析波动率与 mfe_20 的关联，对比替代标签 good_trade 的统计特性。

Inputs:
    - stop_experiment/output/models_control/candidate_with_scores.parquet
      (含全量数据 + 4 模型预测列 + 标签列)
    - DB stock_k_data (K 线日线，用于计算 ret_5_close / ret_10_close / mdd_before_mfe)

Outputs:
    - experiments/label_sanity/results/label_quality_stats.csv
    - experiments/label_sanity/results/volatility_vs_mfe.csv
    - experiments/label_sanity/results/alternative_label_stats.csv
    - experiments/label_sanity/results/verdict.json

How to Run:
    python -m stop_experiment.experiments.label_sanity.01_label_quality_analysis
    python -m stop_experiment.experiments.label_sanity.01_label_quality_analysis --use-kline
    python -m stop_experiment.experiments.label_sanity.01_label_quality_analysis --use-kline --sample 500

Examples:
    # 快速模式: 用 mae_20 近似 mdd_before_mfe，不加载 K 线（默认）
    python -m stop_experiment.experiments.label_sanity.01_label_quality_analysis

    # 精确模式: 从 K 线计算 mdd_before_mfe / ret_5_close / ret_10_close
    python -m stop_experiment.experiments.label_sanity.01_label_quality_analysis --use-kline

    # 精确模式 + 采样: 只对 test 集前 500 条样本加载 K 线（调试用）
    python -m stop_experiment.experiments.label_sanity.01_label_quality_analysis --use-kline --sample 500

Side Effects:
    - 只读 candidate_with_scores.parquet 和 DB stock_k_data
    - 输出仅写入 experiments/label_sanity/results/
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from scipy import stats

from stop_experiment.pipeline.stop_config import MODELS_DIR, OBS_TRAIN_END, OBS_VAL_END

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NEED_COLS = [
    "obs_date", "ts_code", "obs_day", "obs_close",
    "mfe_20", "mae_20", "atr_pct", "volatility_20d",
    "sell_signal", "buy_signal",
    "pred_sell_reg", "pred_sell_cls",
]

MFE_THRESHOLD = 0.07
MAE_THRESHOLD = -0.07
GOOD_TRADE_RET_THRESHOLD = 0.05
GOOD_TRADE_MAE_THRESHOLD = -0.05


def load_scores(scores_path: str) -> pd.DataFrame:
    df = pd.read_parquet(scores_path, columns=NEED_COLS)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.dropna(subset=["mfe_20", "mae_20", "atr_pct"])
    return df


def split_test(df: pd.DataFrame) -> pd.DataFrame:
    val_end = pd.Timestamp(OBS_VAL_END)
    return df[df["obs_date"] > val_end].copy()


def load_kline_for_test(test_df: pd.DataFrame, sample_n: int | None = None) -> pd.DataFrame:
    from datasource.database import get_engine
    from sqlalchemy import text

    if sample_n is not None:
        test_df = test_df.head(sample_n)

    ts_codes = test_df["ts_code"].unique().tolist()
    obs_dates = test_df["obs_date"]

    date_min = (obs_dates.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    date_max = (obs_dates.max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    logger.info(f"加载 K 线: {len(ts_codes)} 只股票, {date_min} ~ {date_max}")

    engine = get_engine()
    chunks = []
    batch_size = 500
    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i:i + batch_size]
        placeholders = ", ".join([f":c{j}" for j in range(len(batch))])
        params = {f"c{j}": code for j, code in enumerate(batch)}
        sql = text(f"""
            SELECT ts_code, bar_time, open, high, low, close
            FROM stock_k_data
            WHERE freq = 'd'
              AND ts_code IN ({placeholders})
              AND bar_time >= :date_min
              AND bar_time <= :date_max
            ORDER BY ts_code, bar_time
        """)
        params["date_min"] = date_min
        params["date_max"] = date_max
        with engine.begin() as conn:
            chunk = pd.read_sql(sql, conn, params=params)
        chunks.append(chunk)
        logger.info(f"  K 线批次 {i // batch_size + 1}: {len(chunk)} 行")

    kline = pd.concat(chunks, ignore_index=True)
    kline["bar_time"] = pd.to_datetime(kline["bar_time"])
    logger.info(f"K 线总计: {len(kline)} 行, {kline['ts_code'].nunique()} 只股票")
    return kline


def compute_kline_metrics(test_df: pd.DataFrame, kline: pd.DataFrame) -> pd.DataFrame:
    kline_sorted = kline.sort_values(["ts_code", "bar_time"]).reset_index(drop=True)
    kline_sorted["bar_idx"] = kline_sorted.groupby("ts_code").cumcount()

    bar_time_idx = kline_sorted.set_index(["ts_code", "bar_time"])["bar_idx"]

    results = []
    grouped = test_df.groupby("ts_code")

    total = len(test_df)
    for ts_code, group in grouped:
        stock_kline = kline_sorted[kline_sorted["ts_code"] == ts_code].reset_index(drop=True)
        if len(stock_kline) == 0:
            continue

        bt_map = stock_kline.set_index("bar_time")["bar_idx"].to_dict()

        for _, row in group.iterrows():
            obs_date = row["obs_date"]
            if obs_date not in bt_map:
                continue
            obs_idx = bt_map[obs_date]
            obs_close = stock_kline.loc[obs_idx, "close"]
            n_bars = len(stock_kline)

            ret_5_close = np.nan
            if obs_idx + 5 < n_bars:
                ret_5_close = stock_kline.loc[obs_idx + 5, "close"] / obs_close - 1.0

            ret_10_close = np.nan
            if obs_idx + 10 < n_bars:
                ret_10_close = stock_kline.loc[obs_idx + 10, "close"] / obs_close - 1.0

            mdd_before_mfe = np.nan
            future_window = stock_kline.iloc[obs_idx + 1: obs_idx + 21]
            if len(future_window) > 0:
                cummax = future_window["high"].cummax()
                peak_idx = cummax.idxmax()
                peak_offset = peak_idx - obs_idx

                path_to_peak = stock_kline.iloc[obs_idx + 1: obs_idx + 1 + peak_offset]
                if len(path_to_peak) > 0:
                    running_max = path_to_peak["high"].cummax()
                    running_dd = path_to_peak["low"] / running_max - 1.0
                    mdd_before_mfe = running_dd.min()

            results.append({
                "ts_code": ts_code,
                "obs_date": obs_date,
                "obs_close": obs_close,
                "ret_5_close": ret_5_close,
                "ret_10_close": ret_10_close,
                "mdd_before_mfe": mdd_before_mfe,
            })

    result_df = pd.DataFrame(results)
    logger.info(f"K 线指标计算完成: {len(result_df)} / {total} 条样本有结果")
    return result_df


def analyze_label_quality(test_df: pd.DataFrame, kline_metrics: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    n_total = len(test_df)
    n_mfe_above = int((test_df["mfe_20"] > MFE_THRESHOLD).sum())
    pct_mfe_above = n_mfe_above / n_total if n_total > 0 else 0.0

    n_mfe_above_and_mae_deep = int(((test_df["mfe_20"] > MFE_THRESHOLD) & (test_df["mae_20"] < -0.05)).sum())
    pct_mfe_above_and_mae_deep = n_mfe_above_and_mae_deep / n_mfe_above if n_mfe_above > 0 else 0.0

    rows.append({
        "metric": "mfe_20>7% 比例",
        "value": pct_mfe_above,
        "n": n_mfe_above,
        "total": n_total,
        "method": "mae_approx",
    })
    rows.append({
        "metric": "mfe_20>7% 且 mae_20<-5% 比例(先大亏后回本)",
        "value": pct_mfe_above_and_mae_deep,
        "n": n_mfe_above_and_mae_deep,
        "total": n_mfe_above,
        "method": "mae_approx",
    })

    if kline_metrics is not None and len(kline_metrics) > 0:
        merged = test_df.merge(kline_metrics, on=["ts_code", "obs_date"], how="inner", suffixes=("", "_kline"))
        n_merged = len(merged)

        n_mfe_kline = int((merged["mfe_20"] > MFE_THRESHOLD).sum())

        has_mdd = merged["mdd_before_mfe"].notna()
        n_mdd_deep = int(((merged["mfe_20"] > MFE_THRESHOLD) & (merged["mdd_before_mfe"] < -0.05) & has_mdd).sum())
        n_mdd_valid = int(((merged["mfe_20"] > MFE_THRESHOLD) & has_mdd).sum())
        pct_mdd_deep = n_mdd_deep / n_mdd_valid if n_mdd_valid > 0 else 0.0

        rows.append({
            "metric": "mfe_20>7% 且 mdd_before_mfe<-5% 比例(精确)",
            "value": pct_mdd_deep,
            "n": n_mdd_deep,
            "total": n_mdd_valid,
            "method": "kline_exact",
        })

        has_ret10 = merged["ret_10_close"].notna()
        n_ret10_neg = int(((merged["mfe_20"] > MFE_THRESHOLD) & (merged["ret_10_close"] < 0) & has_ret10).sum())
        n_ret10_valid = int(((merged["mfe_20"] > MFE_THRESHOLD) & has_ret10).sum())
        pct_ret10_neg = n_ret10_neg / n_ret10_valid if n_ret10_valid > 0 else 0.0

        rows.append({
            "metric": "mfe_20>7% 但 ret_10_close<0 比例(冲高收盘没涨)",
            "value": pct_ret10_neg,
            "n": n_ret10_neg,
            "total": n_ret10_valid,
            "method": "kline_exact",
        })

        has_ret5 = merged["ret_5_close"].notna()
        avg_ret5 = merged.loc[has_ret5, "ret_5_close"].mean()
        avg_ret10 = merged.loc[has_ret10, "ret_10_close"].mean()
        avg_mfe = merged["mfe_20"].mean()

        rows.append({
            "metric": "test集平均 mfe_20",
            "value": avg_mfe,
            "n": n_merged,
            "total": n_merged,
            "method": "kline_exact",
        })
        rows.append({
            "metric": "test集平均 ret_5_close",
            "value": avg_ret5,
            "n": int(has_ret5.sum()),
            "total": n_merged,
            "method": "kline_exact",
        })
        rows.append({
            "metric": "test集平均 ret_10_close",
            "value": avg_ret10,
            "n": int(has_ret10.sum()),
            "total": n_merged,
            "method": "kline_exact",
        })

    return pd.DataFrame(rows)


def analyze_volatility_vs_mfe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    spearman_atr_mfe, p_atr_mfe = stats.spearmanr(df["atr_pct"], df["mfe_20"])
    rows.append({
        "metric": "atr_pct vs mfe_20 Spearman",
        "value": spearman_atr_mfe,
        "p_value": p_atr_mfe,
        "n": len(df),
    })

    spearman_vol_mfe, p_vol_mfe = stats.spearmanr(df["volatility_20d"], df["mfe_20"])
    rows.append({
        "metric": "volatility_20d vs mfe_20 Spearman",
        "value": spearman_vol_mfe,
        "p_value": p_vol_mfe,
        "n": len(df),
    })

    q33 = df["atr_pct"].quantile(1 / 3)
    q66 = df["atr_pct"].quantile(2 / 3)

    df_copy = df.copy()
    df_copy["atr_group"] = pd.cut(
        df_copy["atr_pct"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["low_atr", "mid_atr", "high_atr"],
    )

    for group_name, group_df in df_copy.groupby("atr_group", observed=True):
        rows.append({
            "metric": f"{group_name} mfe_20 均值",
            "value": group_df["mfe_20"].mean(),
            "p_value": np.nan,
            "n": len(group_df),
        })
        rows.append({
            "metric": f"{group_name} sell_signal 正类比例",
            "value": (group_df["sell_signal"] == 1).mean(),
            "p_value": np.nan,
            "n": len(group_df),
        })
        rows.append({
            "metric": f"{group_name} mae_20 均值",
            "value": group_df["mae_20"].mean(),
            "p_value": np.nan,
            "n": len(group_df),
        })

    high_atr_mfe = df_copy[df_copy["atr_group"] == "high_atr"]["mfe_20"].mean()
    low_atr_mfe = df_copy[df_copy["atr_group"] == "low_atr"]["mfe_20"].mean()
    rows.append({
        "metric": "高ATR/低ATR mfe_20 比值",
        "value": high_atr_mfe / low_atr_mfe if low_atr_mfe != 0 else np.nan,
        "p_value": np.nan,
        "n": len(df),
    })

    return pd.DataFrame(rows)


def analyze_alternative_label(df: pd.DataFrame, kline_metrics: pd.DataFrame | None) -> pd.DataFrame:
    rows = []

    sell_signal_rate = (df["sell_signal"] == 1).mean()
    rows.append({
        "metric": "sell_signal 正类比例",
        "value": sell_signal_rate,
        "n": len(df),
    })

    if kline_metrics is not None and len(kline_metrics) > 0:
        merged = df.merge(kline_metrics, on=["ts_code", "obs_date"], how="inner", suffixes=("", "_kline"))
        has_ret10 = merged["ret_10_close"].notna()
        valid = merged[has_ret10].copy()
        valid["good_trade"] = ((valid["ret_10_close"] > GOOD_TRADE_RET_THRESHOLD) & (valid["mae_20"] > GOOD_TRADE_MAE_THRESHOLD)).astype(int)
        good_trade_rate = valid["good_trade"].mean()

        rows.append({
            "metric": "good_trade 正类比例 (ret_10_close>5% 且 mae_20>-5%)",
            "value": good_trade_rate,
            "n": len(valid),
        })

        if valid["pred_sell_reg"].std() > 1e-10:
            ic_sell_vs_sell, _ = stats.spearmanr(valid["pred_sell_reg"], valid["sell_signal"])
            ic_sell_vs_good, _ = stats.spearmanr(valid["pred_sell_reg"], valid["good_trade"])
            rows.append({
                "metric": "pred_sell_reg vs sell_signal Spearman IC",
                "value": ic_sell_vs_sell,
                "n": len(valid),
            })
            rows.append({
                "metric": "pred_sell_reg vs good_trade Spearman IC",
                "value": ic_sell_vs_good,
                "n": len(valid),
            })

        daily_ic_sell = []
        daily_ic_good = []
        for date, day_group in valid.groupby("obs_date"):
            if len(day_group) < 10:
                continue
            if day_group["pred_sell_reg"].std() < 1e-10:
                continue
            if day_group["sell_signal"].std() < 1e-10:
                continue
            ic_s, _ = stats.spearmanr(day_group["pred_sell_reg"], day_group["sell_signal"])
            daily_ic_sell.append(ic_s)
            if day_group["good_trade"].std() < 1e-10:
                continue
            ic_g, _ = stats.spearmanr(day_group["pred_sell_reg"], day_group["good_trade"])
            daily_ic_good.append(ic_g)

        if daily_ic_sell:
            rows.append({
                "metric": "日均 IC: pred_sell_reg vs sell_signal",
                "value": np.mean(daily_ic_sell),
                "n": len(daily_ic_sell),
            })
        if daily_ic_good:
            rows.append({
                "metric": "日均 IC: pred_sell_reg vs good_trade",
                "value": np.mean(daily_ic_good),
                "n": len(daily_ic_good),
            })
    else:
        df_copy = df.copy()
        df_copy["good_trade_approx"] = ((df_copy["mfe_20"] > GOOD_TRADE_RET_THRESHOLD + 0.02) & (df_copy["mae_20"] > GOOD_TRADE_MAE_THRESHOLD)).astype(int)
        good_trade_approx_rate = df_copy["good_trade_approx"].mean()

        rows.append({
            "metric": "good_trade_approx 正类比例 (mfe_20>7% 且 mae_20>-5%, 无K线近似)",
            "value": good_trade_approx_rate,
            "n": len(df_copy),
        })

        if df_copy["pred_sell_reg"].std() > 1e-10:
            ic_sell, _ = stats.spearmanr(df_copy["pred_sell_reg"], df_copy["sell_signal"])
            ic_good, _ = stats.spearmanr(df_copy["pred_sell_reg"], df_copy["good_trade_approx"])
            rows.append({
                "metric": "pred_sell_reg vs sell_signal Spearman IC (全量)",
                "value": ic_sell,
                "n": len(df_copy),
            })
            rows.append({
                "metric": "pred_sell_reg vs good_trade_approx Spearman IC (全量)",
                "value": ic_good,
                "n": len(df_copy),
            })

        daily_ic_sell = []
        daily_ic_good = []
        for date, day_group in df_copy.groupby("obs_date"):
            if len(day_group) < 10:
                continue
            if day_group["pred_sell_reg"].std() < 1e-10:
                continue
            if day_group["sell_signal"].std() < 1e-10:
                continue
            ic_s, _ = stats.spearmanr(day_group["pred_sell_reg"], day_group["sell_signal"])
            daily_ic_sell.append(ic_s)
            if day_group["good_trade_approx"].std() < 1e-10:
                continue
            ic_g, _ = stats.spearmanr(day_group["pred_sell_reg"], day_group["good_trade_approx"])
            daily_ic_good.append(ic_g)

        if daily_ic_sell:
            rows.append({
                "metric": "日均 IC: pred_sell_reg vs sell_signal (全量)",
                "value": np.mean(daily_ic_sell),
                "n": len(daily_ic_sell),
            })
        if daily_ic_good:
            rows.append({
                "metric": "日均 IC: pred_sell_reg vs good_trade_approx (全量)",
                "value": np.mean(daily_ic_good),
                "n": len(daily_ic_good),
            })

    return pd.DataFrame(rows)


def build_verdict(
    label_stats: pd.DataFrame,
    vol_stats: pd.DataFrame,
    alt_stats: pd.DataFrame,
) -> dict:
    verdict = {
        "conclusion": "unknown",
        "label_idealization_risk": "unknown",
        "volatility_confusion_risk": "unknown",
        "alternative_label_viability": "unknown",
        "details": {},
    }

    mae_approx_row = label_stats[label_stats["metric"] == "mfe_20>7% 且 mae_20<-5% 比例(先大亏后回本)"]
    if len(mae_approx_row) > 0:
        pct = float(mae_approx_row["value"].iloc[0])
        verdict["details"]["pct_mfe_above_and_mae_deep"] = pct
        if pct > 0.40:
            verdict["label_idealization_risk"] = "high"
        elif pct > 0.25:
            verdict["label_idealization_risk"] = "medium"
        else:
            verdict["label_idealization_risk"] = "low"

    kline_row = label_stats[label_stats["metric"] == "mfe_20>7% 但 ret_10_close<0 比例(冲高收盘没涨)"]
    if len(kline_row) > 0:
        pct = float(kline_row["value"].iloc[0])
        verdict["details"]["pct_mfe_above_but_ret10_neg"] = pct
        if pct > 0.30:
            verdict["label_idealization_risk"] = "high"

    atr_spearman_row = vol_stats[vol_stats["metric"] == "atr_pct vs mfe_20 Spearman"]
    if len(atr_spearman_row) > 0:
        spearman_val = float(atr_spearman_row["value"].iloc[0])
        verdict["details"]["atr_mfe_spearman"] = spearman_val
        if spearman_val > 0.6:
            verdict["volatility_confusion_risk"] = "high"
        elif spearman_val > 0.4:
            verdict["volatility_confusion_risk"] = "medium"
        else:
            verdict["volatility_confusion_risk"] = "low"

    ratio_row = vol_stats[vol_stats["metric"] == "高ATR/低ATR mfe_20 比值"]
    if len(ratio_row) > 0:
        ratio = float(ratio_row["value"].iloc[0])
        verdict["details"]["high_low_atr_mfe_ratio"] = ratio
        if ratio > 3.0:
            verdict["volatility_confusion_risk"] = "high"

    ic_sell_row = alt_stats[alt_stats["metric"].str.contains("pred_sell_reg vs sell_signal")]
    ic_good_row = alt_stats[alt_stats["metric"].str.contains("pred_sell_reg vs good_trade")]

    if len(ic_sell_row) > 0 and len(ic_good_row) > 0:
        ic_sell = float(ic_sell_row["value"].iloc[0])
        ic_good = float(ic_good_row["value"].iloc[0])
        verdict["details"]["ic_sell_signal"] = ic_sell
        verdict["details"]["ic_good_trade"] = ic_good
        ic_ratio = ic_good / ic_sell if abs(ic_sell) > 1e-10 else 0.0
        verdict["details"]["ic_good_to_sell_ratio"] = ic_ratio
        if ic_ratio > 0.8:
            verdict["alternative_label_viability"] = "viable"
        elif ic_ratio > 0.5:
            verdict["alternative_label_viability"] = "partial"
        else:
            verdict["alternative_label_viability"] = "weak"

    risks = [verdict["label_idealization_risk"], verdict["volatility_confusion_risk"]]
    if "high" in risks:
        verdict["conclusion"] = "label_quality_concern"
    elif "medium" in risks:
        verdict["conclusion"] = "label_quality_moderate_risk"
    else:
        verdict["conclusion"] = "label_quality_acceptable"

    return verdict


def main():
    parser = argparse.ArgumentParser(description="MFE 标签质量验证分析")
    parser.add_argument("--use-kline", action="store_true", help="从 DB 加载 K 线计算精确指标（较慢）")
    parser.add_argument("--sample", type=int, default=None, help="K 线模式下的采样数量（调试用）")
    args = parser.parse_args()

    print("=" * 60)
    print("MFE 标签质量验证分析")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    scores_path = os.path.join(MODELS_DIR, "candidate_with_scores.parquet")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"{scores_path} 不存在")

    print("\n[1/5] 加载数据...")
    df = load_scores(scores_path)
    test_df = split_test(df)
    print(f"  全量: {len(df)} 行, test: {len(test_df)} 行")
    print(f"  test obs_date 范围: {test_df['obs_date'].min().date()} ~ {test_df['obs_date'].max().date()}")

    kline_metrics = None
    if args.use_kline:
        print("\n[2/5] 加载 K 线并计算精确指标...")
        kline = load_kline_for_test(test_df, sample_n=args.sample)
        kline_metrics = compute_kline_metrics(test_df, kline)
        if kline_metrics is not None and len(kline_metrics) > 0:
            print(f"  K 线指标: {len(kline_metrics)} 条样本")
            print(f"  ret_5_close 有效: {kline_metrics['ret_5_close'].notna().sum()}")
            print(f"  ret_10_close 有效: {kline_metrics['ret_10_close'].notna().sum()}")
            print(f"  mdd_before_mfe 有效: {kline_metrics['mdd_before_mfe'].notna().sum()}")
    else:
        print("\n[2/5] 跳过 K 线加载（使用 mae_20 近似），加 --use-kline 启用精确计算")

    print("\n[3/5] 标签质量分析...")
    label_stats = analyze_label_quality(test_df, kline_metrics)
    label_path = os.path.join(RESULTS_DIR, "label_quality_stats.csv")
    label_stats.to_csv(label_path, index=False)
    print(f"  保存: {label_path}")

    print(f"\n  {'指标':50s} {'值':>10s} {'样本数':>8s} {'总数':>8s} {'方法':12s}")
    for _, row in label_stats.iterrows():
        val_str = f"{row['value']:.4f}" if not np.isnan(row['value']) else "NaN"
        n_str = str(int(row['n'])) if not np.isnan(row['n']) else "-"
        total_str = str(int(row['total'])) if not np.isnan(row['total']) else "-"
        print(f"  {row['metric']:50s} {val_str:>10s} {n_str:>8s} {total_str:>8s} {row.get('method', ''):12s}")

    print("\n[4/5] 波动率 vs MFE 分析...")
    vol_stats = analyze_volatility_vs_mfe(test_df)
    vol_path = os.path.join(RESULTS_DIR, "volatility_vs_mfe.csv")
    vol_stats.to_csv(vol_path, index=False)
    print(f"  保存: {vol_path}")

    print(f"\n  {'指标':50s} {'值':>10s} {'样本数':>8s}")
    for _, row in vol_stats.iterrows():
        val_str = f"{row['value']:.4f}" if not np.isnan(row['value']) else "NaN"
        n_str = str(int(row['n'])) if not np.isnan(row['n']) else "-"
        print(f"  {row['metric']:50s} {val_str:>10s} {n_str:>8s}")

    print("\n[5/5] 替代标签对比 + 结论...")
    alt_stats = analyze_alternative_label(test_df, kline_metrics)
    alt_path = os.path.join(RESULTS_DIR, "alternative_label_stats.csv")
    alt_stats.to_csv(alt_path, index=False)
    print(f"  保存: {alt_path}")

    print(f"\n  {'指标':60s} {'值':>10s} {'样本数':>8s}")
    for _, row in alt_stats.iterrows():
        val_str = f"{row['value']:.4f}" if not np.isnan(row['value']) else "NaN"
        n_str = str(int(row['n'])) if not np.isnan(row['n']) else "-"
        print(f"  {row['metric']:60s} {val_str:>10s} {n_str:>8s}")

    verdict = build_verdict(label_stats, vol_stats, alt_stats)
    verdict_path = os.path.join(RESULTS_DIR, "verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print(f"  保存: {verdict_path}")

    print(f"\n{'=' * 60}")
    print("结论")
    print(f"{'=' * 60}")

    risk = verdict["label_idealization_risk"]
    if risk == "high":
        print(f"  ❌ 标签理想化风险高: mfe_20 大量不可交易信号")
    elif risk == "medium":
        print(f"  ⚠️  标签理想化风险中等: 部分信号不可交易")
    else:
        print(f"  ✅ 标签理想化风险低: mfe_20 多数可达")

    vol_risk = verdict["volatility_confusion_risk"]
    if vol_risk == "high":
        print(f"  ❌ 波动率混淆风险高: 模型可能选高波动票而非高收益票")
    elif vol_risk == "medium":
        print(f"  ⚠️  波动率混淆风险中等")
    else:
        print(f"  ✅ 波动率混淆风险低")

    viability = verdict["alternative_label_viability"]
    if viability == "viable":
        print(f"  ✅ 替代标签 good_trade 可行: IC 与 sell_signal 接近")
    elif viability == "partial":
        print(f"  ⚠️  替代标签 good_trade 部分可行: IC 有衰减")
    else:
        print(f"  ❌ 替代标签 good_trade 不可行: IC 衰减严重")

    print(f"\n  总判定: {verdict['conclusion']}")


if __name__ == "__main__":
    main()
