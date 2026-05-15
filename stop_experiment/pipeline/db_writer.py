#!/usr/bin/env python3
"""
预测结果数据库写入模块

Purpose:
    将 07_generate_daily_predictions 产出的预测结果写入 stop_loss_predictions 表。
    使用 datasource.database.bulk_upsert 遵循项目 DB 写入 SSOT。

Inputs:
    - df: 包含预测结果的 DataFrame (07 的输出)
    - prediction_date, obs_date, model_version, profile 等元数据

Outputs:
    - 写入 PostgreSQL 表 stop_loss_predictions

Side Effects:
    - 写数据库 (UPSERT)
    - 建表 (如不存在)

Usage:
    from stop_experiment.pipeline.db_writer import save_predictions_to_db
    save_predictions_to_db(df, prediction_date="2026-05-08",
                           model_version="mv_20260512_retrain_v1",
                           profile="production")
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from sqlalchemy import text

from datasource.database import get_engine, bulk_upsert, table_exists


TABLE_NAME = "stop_loss_predictions"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    prediction_date DATE NOT NULL,
    obs_date DATE NOT NULL,
    obs_day INT NOT NULL DEFAULT 1,
    ts_code VARCHAR(20) NOT NULL,
    code VARCHAR(10) NOT NULL,
    stock_name VARCHAR(50),
    pred_sell_reg FLOAT,
    pred_sell_cls FLOAT,
    pred_buy_reg FLOAT,
    pred_buy_cls FLOAT,
    composite_score FLOAT,
    score FLOAT,
    signal_id BIGINT,
    sell_stop_triggered BOOLEAN,
    sell_stop_scale FLOAT,
    can_buy BOOLEAN DEFAULT TRUE,
    feature_version VARCHAR(50),
    model_version VARCHAR(50),
    profile VARCHAR(50) DEFAULT 'production',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(prediction_date, obs_date, ts_code, model_version, profile)
);

CREATE INDEX IF NOT EXISTS idx_sl_pred_date ON {TABLE_NAME}(prediction_date);
CREATE INDEX IF NOT EXISTS idx_sl_pred_obs_date ON {TABLE_NAME}(obs_date);
CREATE INDEX IF NOT EXISTS idx_sl_pred_ts_code ON {TABLE_NAME}(ts_code);
CREATE INDEX IF NOT EXISTS idx_sl_pred_model_version ON {TABLE_NAME}(model_version);
CREATE INDEX IF NOT EXISTS idx_sl_pred_composite ON {TABLE_NAME}(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_sl_pred_profile ON {TABLE_NAME}(profile);
"""

UNIQUE_KEYS = ["prediction_date", "obs_date", "ts_code", "model_version", "profile"]


def ensure_prediction_table():
    engine = get_engine()
    with engine.begin() as conn:
        if not table_exists(conn, TABLE_NAME):
            conn.execute(text(CREATE_TABLE_SQL))
            print(f"  [DB] 建表 {TABLE_NAME}")
        else:
            _migrate_unique_constraint(conn)
    engine.dispose()


def _migrate_unique_constraint(conn):
    """迁移 UNIQUE 约束：将 profile 加入唯一键。"""
    result = conn.execute(text(
        "SELECT conname FROM pg_constraint c "
        "JOIN pg_class t ON c.conrelid = t.oid "
        "WHERE t.relname = :table_name AND c.contype = 'u'"
    ), {"table_name": TABLE_NAME}).fetchall()

    old_constraint = None
    new_constraint = f"{TABLE_NAME}_pred_obs_code_model_profile_key"
    for (conname,) in result:
        if "mode_key" in conname and "profile" not in conname:
            old_constraint = conname
            break

    if old_constraint:
        try:
            conn.execute(text(f'ALTER TABLE {TABLE_NAME} DROP CONSTRAINT IF EXISTS {old_constraint}'))
            conn.execute(text(
                f'ALTER TABLE {TABLE_NAME} ADD CONSTRAINT {new_constraint} '
                f'UNIQUE (prediction_date, obs_date, ts_code, model_version, profile)'
            ))
            print(f"  [DB] 迁移 UNIQUE 约束: {old_constraint} → 含 profile")
        except Exception as e:
            print(f"  [DB] UNIQUE 约束迁移失败 (非致命): {e}")


def save_predictions_to_db(
    df: pd.DataFrame,
    prediction_date: str,
    model_version: str = "unknown",
    feature_version: str = "unknown",
    profile: str = "production",
) -> int:
    if df.empty:
        print(f"  [DB] 无数据可写入 {TABLE_NAME}")
        return 0

    ensure_prediction_table()

    pred_dt = pd.to_datetime(prediction_date)

    db_df = pd.DataFrame()
    db_df["prediction_date"] = df["obs_date"].values
    db_df["obs_date"] = df["obs_date"].values
    db_df["obs_day"] = df["obs_day"].values if "obs_day" in df.columns else 1

    if "ts_code" in df.columns:
        db_df["ts_code"] = df["ts_code"].values
        db_df["code"] = df["ts_code"].str.replace(r"\.(SZ|SH|BJ)", "", regex=True).values
    elif "code" in df.columns:
        db_df["code"] = df["code"].values
        db_df["ts_code"] = df["code"].values
    else:
        db_df["code"] = ""
        db_df["ts_code"] = ""

    db_df["stock_name"] = df["stock_name"].values if "stock_name" in df.columns else None
    db_df["pred_sell_reg"] = df["pred_sell_reg"].values if "pred_sell_reg" in df.columns else None
    db_df["pred_sell_cls"] = df["pred_sell_cls"].values if "pred_sell_cls" in df.columns else None
    db_df["pred_buy_reg"] = df["pred_buy_reg"].values if "pred_buy_reg" in df.columns else None
    db_df["pred_buy_cls"] = df["pred_buy_cls"].values if "pred_buy_cls" in df.columns else None
    db_df["composite_score"] = df["composite_score"].values if "composite_score" in df.columns else None
    db_df["score"] = df["score"].values if "score" in df.columns else None
    db_df["signal_id"] = df["signal_id"].values if "signal_id" in df.columns else None
    db_df["sell_stop_triggered"] = df["sell_stop_triggered"].values if "sell_stop_triggered" in df.columns else None
    db_df["sell_stop_scale"] = df["sell_stop_scale"].values if "sell_stop_scale" in df.columns else None
    db_df["can_buy"] = df["can_buy"].values if "can_buy" in df.columns else True
    if "can_buy" in db_df.columns:
        db_df["can_buy"] = db_df["can_buy"].astype(bool)
    db_df["feature_version"] = feature_version
    db_df["model_version"] = model_version
    db_df["profile"] = profile

    db_df["prediction_date"] = pd.to_datetime(db_df["prediction_date"]).dt.date
    db_df["obs_date"] = pd.to_datetime(db_df["obs_date"]).dt.date

    engine = get_engine()
    try:
        with engine.begin() as conn:
            n = bulk_upsert(conn, TABLE_NAME, db_df, UNIQUE_KEYS, auto_commit=False)
        print(f"  [DB] 写入 {TABLE_NAME}: {n} 条 (upsert, model={model_version})")
        return n
    except Exception as e:
        print(f"  [DB] 写入失败: {e}")
        raise
    finally:
        engine.dispose()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="预测结果数据库写入模块自测")
    parser.add_argument("--query-date", help="查询指定日期的预测记录")
    parser.add_argument("--count", action="store_true", help="统计总记录数")
    args = parser.parse_args()

    if args.query_date:
        engine = get_engine()
        with engine.connect() as conn:
            from datasource.database import query_sql
            result = query_sql(conn, f"""
                SELECT prediction_date, ts_code, code, stock_name,
                       pred_sell_reg, composite_score, score,
                       model_version, profile
                FROM {TABLE_NAME}
                WHERE prediction_date = :d
                ORDER BY composite_score DESC
                LIMIT 20
            """, {"d": args.query_date})
            print(f"\n=== {args.query_date} 预测记录 (Top 20) ===")
            print(result.to_string(index=False))
        engine.dispose()
    elif args.count:
        engine = get_engine()
        with engine.connect() as conn:
            from datasource.database import query_sql
            result = query_sql(conn, f"""
                SELECT prediction_date, model_version, COUNT(*) as cnt
                FROM {TABLE_NAME}
                GROUP BY prediction_date, model_version
                ORDER BY prediction_date
            """)
            print(f"\n=== {TABLE_NAME} 统计 ===")
            print(result.to_string(index=False))
        engine.dispose()
    else:
        print("用法: python -m stop_experiment.pipeline.db_writer --query-date 2026-05-08")
        print("      python -m stop_experiment.pipeline.db_writer --count")
