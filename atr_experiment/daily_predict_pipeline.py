#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATR GBDT 每日预测管道

Purpose: 合并选股结果+自选股，计算特征，执行两模型预测，写入 atr_rope_features 表
Inputs:  atr_rope_selection 表 + stock_watchlist 表 + stock_k_data 表
Outputs: atr_rope_features 表

How to Run:
    python atr_experiment/daily_predict_pipeline.py 2026-05-21
    python atr_experiment/daily_predict_pipeline.py 2026-05-21 --retrain

Examples:
    python atr_experiment/daily_predict_pipeline.py 2026-05-21
    python atr_experiment/daily_predict_pipeline.py 2026-05-21 --retrain

Side Effects: 写入 atr_rope_features 表（幂等：同一日期先删后插）
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from atr_experiment.atr_gbdt_utils import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES,
    PROFIT_THRESHOLD, REG_PARAMS, CLS_PARAMS, NUM_BOOST_ROUND,
    compute_derived_fields, enrich_with_future_metrics,
    prepare_features, load_selection_data,
)
from atr_experiment.daily_predict import (
    find_latest_model, load_cached_model, save_model, train_model,
)

OUTPUT_DIR = PROJECT_ROOT / "atr_experiment" / "output"
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


# ==================== 数据加载 ====================

def load_watchlist_codes() -> list:
    """从 stock_watchlist 表加载自选股代码列表"""
    from datasource.database import get_engine
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT ts_code, stock_name FROM stock_watchlist ORDER BY sort_order, added_date DESC"))
        rows = result.fetchall()
    engine.dispose()
    return [(row[0], row[1]) for row in rows]


def load_selection_features(selection_date: str) -> pd.DataFrame:
    """从 atr_rope_selection 加载当天选股结果的特征"""
    from datasource.database import get_engine
    engine = get_engine()
    sql = text("""
        SELECT ts_code, stock_name, regime, regime_strength,
               signal_type,
               rope_dev_pct, rope_dev_atr, range_width_pct,
               change_pct, vol_zscore, dsa_dir_bars,
               rope_dir1_pct, rope_dir_neg1_pct, range_pos_01,
               dsa_vwap_dev_pct, avg_amount_20d,
               bbmacd_event, low_rope_signal, low_vwap_signal,
               low_rope_dev_mean, low_rope_dev_std, low_rope_dev_today,
               low_vwap_dev_mean, low_vwap_dev_std, low_vwap_dev_today,
               rope_value
        FROM atr_rope_selection
        WHERE selection_date = :selection_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"selection_date": selection_date})
    engine.dispose()
    df["source"] = "selection"
    return df


def compute_watchlist_features(watchlist_codes: list, selection_date: str,
                               existing_codes: set) -> pd.DataFrame:
    """对不在选股结果中的自选股计算特征"""
    from selection.seletion_atr import compute_stock_features

    sel_date = pd.Timestamp(selection_date).date()
    results = []

    for ts_code, stock_name in watchlist_codes:
        if ts_code in existing_codes:
            continue
        try:
            features = compute_stock_features(ts_code, sel_date)
            if features is not None:
                features["stock_name"] = stock_name
                features["source"] = "watchlist"
                results.append(features)
        except Exception as e:
            logger.warning(f"计算自选股特征失败 {ts_code}: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


# ==================== 预测与标记 ====================

def predict_and_mark(df: pd.DataFrame, reg_model, cls_model,
                     top_k_list: list = None) -> pd.DataFrame:
    """双模型预测并标记交集（按source分组排名）"""
    if top_k_list is None:
        top_k_list = [5, 10]

    X_pred, available_features = prepare_features(df)
    df = df.copy()
    df["pred_return"] = reg_model.predict(X_pred)
    df["pred_prob"] = cls_model.predict(X_pred)
    df["combined_score"] = df["pred_return"] * df["pred_prob"]

    # 按 source 分组排名
    for source in df["source"].unique():
        mask = df["source"] == source
        df.loc[mask, "reg_rank"] = df.loc[mask, "pred_return"].rank(ascending=False, method="min").astype(int)
        df.loc[mask, "cls_rank"] = df.loc[mask, "pred_prob"].rank(ascending=False, method="min").astype(int)
        df.loc[mask, "combined_rank"] = df.loc[mask, "combined_score"].rank(ascending=False, method="min").astype(int)

        for k in top_k_list:
            reg_top = set(df.loc[mask].nlargest(k, "pred_return")["ts_code"].values)
            cls_top = set(df.loc[mask].nlargest(k, "pred_prob")["ts_code"].values)
            overlap = reg_top & cls_top

            df.loc[mask & df["ts_code"].isin(reg_top), f"in_reg_top{k}"] = True
            df.loc[mask & df["ts_code"].isin(cls_top), f"in_cls_top{k}"] = True
            df.loc[mask & df["ts_code"].isin(overlap), f"in_overlap_top{k}"] = True

    # 填充未标记的为False
    for k in top_k_list:
        for col in [f"in_reg_top{k}", f"in_cls_top{k}", f"in_overlap_top{k}"]:
            if col in df.columns:
                df[col] = df[col].fillna(False)

    return df


# ==================== 写入数据库 ====================

def save_to_database(df: pd.DataFrame, feature_date: str):
    """写入 atr_rope_features 表（幂等：先删后插）"""
    from datasource.database import get_engine
    engine = get_engine()

    # 选择要写入的列
    write_cols = [
        "feature_date", "ts_code", "stock_name", "source",
        "regime", "regime_strength", "signal_type",
        "rope_dev_pct", "rope_dev_atr",
        "range_width_pct", "lower_value", "upper_value",
        "change_pct", "vol_zscore", "dsa_dir_bars",
        "rope_dir1_pct", "rope_neg1_pct" if "rope_neg1_pct" in df.columns else "rope_dir_neg1_pct",
        "range_pos_01", "dsa_vwap_dev_pct", "avg_amount_20d",
        "bbmacd_event", "low_rope_signal", "low_vwap_signal",
        "low_rope_dev_mean_pct", "low_rope_dev_std_pct", "low_rope_dev_today_pct",
        "low_vwap_dev_mean_pct", "low_vwap_dev_std_pct", "low_vwap_dev_today_pct",
        "pred_return", "pred_prob", "combined_score",
        "reg_rank", "cls_rank", "combined_rank",
    ]
    for k in [5, 10]:
        write_cols.extend([f"in_reg_top{k}", f"in_cls_top{k}", f"in_overlap_top{k}"])

    # 只写入存在的列
    available_cols = [c for c in write_cols if c in df.columns]
    write_df = df[available_cols].copy()

    # 修正列名：rope_neg1_pct -> rope_dir_neg1_pct
    if "rope_neg1_pct" in write_df.columns:
        write_df.rename(columns={"rope_neg1_pct": "rope_dir_neg1_pct"}, inplace=True)

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM atr_rope_features WHERE feature_date = :d"), {"d": feature_date})
        conn.commit()

    write_df.to_sql("atr_rope_features", engine, if_exists="append", index=False)
    engine.dispose()
    print(f"写入 atr_rope_features: {len(write_df)} 条 (日期={feature_date})")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="ATR GBDT 每日预测管道")
    parser.add_argument("date", help="预测日期 (YYYY-MM-DD)")
    parser.add_argument("--retrain", action="store_true", help="强制重新训练模型")
    parser.add_argument("--horizon", type=int, default=10, help="预测期限 (默认: 10)")
    args = parser.parse_args()

    target_date = args.date
    horizon = args.horizon
    ret_col = f"return_{horizon}"
    profit_col = f"profit_{horizon}"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 1. 加载当天选股结果特征
    print(f"加载 {target_date} 选股结果特征...")
    sel_df = load_selection_features(target_date)
    print(f"  选股结果: {len(sel_df)} 条")
    existing_codes = set(sel_df["ts_code"].values) if not sel_df.empty else set()

    # 2. 加载自选股并计算特征
    print("加载自选股列表...")
    watchlist_codes = load_watchlist_codes()
    print(f"  自选股: {len(watchlist_codes)} 只")

    watch_df = compute_watchlist_features(watchlist_codes, target_date, existing_codes)
    print(f"  需计算特征的自选股: {len(watch_df)} 只")

    # 3. 合并两个股票池
    if not sel_df.empty:
        sel_df = compute_derived_fields(sel_df)
        # 确保数值列为float类型
        for col in NUMERIC_FEATURES:
            if col in sel_df.columns:
                sel_df[col] = pd.to_numeric(sel_df[col], errors="coerce")
    if not watch_df.empty:
        watch_df = compute_derived_fields(watch_df)
        for col in NUMERIC_FEATURES:
            if col in watch_df.columns:
                watch_df[col] = pd.to_numeric(watch_df[col], errors="coerce")

    combined_df = pd.concat([sel_df, watch_df], ignore_index=True)
    print(f"合并后: {len(combined_df)} 条 (选股={len(sel_df)}, 自选股={len(watch_df)})")

    if combined_df.empty:
        print("无数据可预测，退出")
        return

    # 4. 加载或训练模型
    model_version = None
    reg_model = None
    cls_model = None

    if not args.retrain:
        latest_reg = find_latest_model(horizon, "regression", before_date=target_date)
        latest_cls = find_latest_model(horizon, "classification", before_date=target_date)
        if latest_reg and latest_cls:
            model_version = min(latest_reg, latest_cls)
            print(f"使用缓存模型版本: {model_version}")

    if model_version and not args.retrain:
        reg_model = load_cached_model(model_version, horizon, "regression")
        cls_model = load_cached_model(model_version, horizon, "classification")

    if reg_model is None or cls_model is None:
        model_version = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        print("加载训练数据...")
        train_df = load_selection_data(model_version)
        train_df = compute_derived_fields(train_df)
        train_df = enrich_with_future_metrics(train_df, [horizon])
        train_df[profit_col] = (train_df[ret_col] > PROFIT_THRESHOLD).astype(int)

        version_dt = pd.Timestamp(model_version).date()
        dates = pd.to_datetime(train_df["selection_date"]).dt.date
        train_valid = train_df[dates <= version_dt].dropna(subset=[ret_col])

        print(f"训练集: {len(train_valid)} 条")

        if reg_model is None:
            print(f"训练回归模型 (target={ret_col})...")
            reg_model, _ = train_model(train_valid, ret_col, "regression")
            save_model(reg_model, model_version, horizon, "regression")

        if cls_model is None:
            train_cls = train_df[dates <= version_dt].dropna(subset=[profit_col])
            print(f"训练分类模型 (target={profit_col})...")
            cls_model, _ = train_model(train_cls, profit_col, "classification")
            save_model(cls_model, model_version, horizon, "classification")

    # 5. 预测并标记
    print("执行预测...")
    combined_df = predict_and_mark(combined_df, reg_model, cls_model)

    # 6. 写入数据库
    combined_df["feature_date"] = target_date
    save_to_database(combined_df, target_date)

    # 7. 输出摘要
    for source in ["selection", "watchlist"]:
        sub = combined_df[combined_df["source"] == source]
        if sub.empty:
            continue
        print(f"\n--- {source} 预测摘要 ---")
        top5 = sub.nlargest(5, "combined_score")
        for _, row in top5.iterrows():
            overlap_mark = "★" if row.get("in_overlap_top5", False) else ""
            print(f"  {row['ts_code']} {row.get('stock_name', '')} "
                  f"预测收益={row['pred_return']*100:.1f}% "
                  f">7%概率={row['pred_prob']*100:.1f}% "
                  f"综合={row['combined_score']:.4f} {overlap_mark}")

    print(f"\n预测管道完成！")


if __name__ == "__main__":
    main()
