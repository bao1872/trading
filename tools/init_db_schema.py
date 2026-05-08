#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库 Schema 初始化脚本

Purpose: 创建因子库/事件库所需的数据库表
Inputs: 无（直接操作数据库）
Outputs: 数据库表创建结果
How to Run:
    python tools/init_db_schema.py
    python tools/init_db_schema.py --dry-run
Examples:
    python tools/init_db_schema.py
Side Effects: 创建/更新数据库表结构
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from datasource.database import get_engine, table_exists


# ====== 表定义 SQL ======

STOCK_K_DATA_SQL = """
CREATE TABLE IF NOT EXISTS stock_k_data (
    id          SERIAL PRIMARY KEY,
    ts_code     VARCHAR(20) NOT NULL,
    freq        VARCHAR(10) NOT NULL,
    bar_time    TIMESTAMP NOT NULL,
    open        FLOAT,
    high        FLOAT,
    low         FLOAT,
    close       FLOAT,
    volume      FLOAT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ts_code, freq, bar_time)
);
"""

STOCK_POOLS_SQL = """
CREATE TABLE IF NOT EXISTS stock_pools (
    id                BIGSERIAL PRIMARY KEY,
    ts_code           VARCHAR(20) NOT NULL UNIQUE,
    name              VARCHAR(50) NOT NULL,
    concepts          TEXT,
    popularity_rank   INTEGER,
    market_cap        FLOAT,
    total_market_cap  FLOAT,
    industry_l2       VARCHAR(100),
    industry_l3       VARCHAR(200),
    industry_pe       FLOAT,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INSTRUMENT_SNAPSHOT_SQL = """
CREATE TABLE IF NOT EXISTS instrument_snapshot (
    ts_code         VARCHAR(20) NOT NULL,
    trade_date      DATE NOT NULL,
    name            VARCHAR(50),
    industry_l1     VARCHAR(50),
    industry_l2     VARCHAR(50),
    industry_l3     VARCHAR(50),
    concepts        TEXT,
    market_cap      FLOAT,
    float_market_cap FLOAT,
    is_st           BOOLEAN DEFAULT FALSE,
    list_date       DATE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, trade_date)
);
"""

FACTOR_DEFINITION_SQL = """
CREATE TABLE IF NOT EXISTS factor_definition (
    factor_name     VARCHAR(50) NOT NULL PRIMARY KEY,
    factor_group    VARCHAR(30),
    freq_type_supported VARCHAR(20),
    description     TEXT,
    source_module   VARCHAR(100),
    source_function VARCHAR(100),
    direction       VARCHAR(10),
    is_core         BOOLEAN DEFAULT FALSE,
    is_active       BOOLEAN DEFAULT TRUE,
    max_lookback_bars INTEGER DEFAULT 120,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

EVENT_DEFINITION_SQL = """
CREATE TABLE IF NOT EXISTS event_definition (
    event_name      VARCHAR(50) NOT NULL PRIMARY KEY,
    event_group     VARCHAR(30),
    description     TEXT,
    required_factors TEXT,
    freq_type_supported VARCHAR(20),
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

EVENT_FACTOR_MAP_SQL = """
CREATE TABLE IF NOT EXISTS event_factor_map (
    event_name      VARCHAR(50) NOT NULL,
    factor_name     VARCHAR(50) NOT NULL,
    role            VARCHAR(20),
    PRIMARY KEY (event_name, factor_name)
);
"""

FACTOR_VALUE_SQL = """
CREATE TABLE IF NOT EXISTS factor_value (
    ts_code         VARCHAR(20) NOT NULL,
    as_of_date      DATE NOT NULL,
    freq            VARCHAR(5) NOT NULL,
    factor_name     VARCHAR(50) NOT NULL,
    factor_value    FLOAT,
    factor_version  VARCHAR(20) DEFAULT 'v1',
    source_table    VARCHAR(50) DEFAULT 'factor_lib',
    available_date  DATE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, as_of_date, freq, factor_name, factor_version)
) PARTITION BY LIST (ts_code);
"""

FACTOR_VALUE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_factor_lookup ON factor_value(ts_code, factor_name, freq, as_of_date);
CREATE INDEX IF NOT EXISTS idx_factor_date ON factor_value(as_of_date, freq, factor_name);
"""

EVENT_TRIGGER_SQL = """
CREATE TABLE IF NOT EXISTS event_trigger (
    ts_code         VARCHAR(20) NOT NULL,
    as_of_date      DATE NOT NULL,
    freq            VARCHAR(5) NOT NULL,
    event_name      VARCHAR(50) NOT NULL,
    triggered       BOOLEAN NOT NULL,
    event_strength  FLOAT,
    event_direction VARCHAR(10),
    event_version   VARCHAR(20) DEFAULT 'v1',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ts_code, as_of_date, freq, event_name, event_version)
) PARTITION BY LIST (ts_code);
"""

EVENT_TRIGGER_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_event_lookup ON event_trigger(ts_code, event_name, freq, as_of_date);
CREATE INDEX IF NOT EXISTS idx_event_date ON event_trigger(as_of_date, freq, event_name);
"""

MARKET_INDEX_BAR_SQL = """
CREATE TABLE IF NOT EXISTS market_index_bar (
    index_code      VARCHAR(20) NOT NULL,
    bar_time        TIMESTAMP NOT NULL,
    freq            VARCHAR(5) NOT NULL,
    open            FLOAT,
    high            FLOAT,
    low             FLOAT,
    close           FLOAT,
    volume          FLOAT,
    amount          FLOAT,
    PRIMARY KEY (index_code, bar_time, freq)
);
"""

ALL_TABLES = [
    ("stock_k_data", STOCK_K_DATA_SQL),
    ("stock_pools", STOCK_POOLS_SQL),
    ("instrument_snapshot", INSTRUMENT_SNAPSHOT_SQL),
    ("factor_definition", FACTOR_DEFINITION_SQL),
    ("event_definition", EVENT_DEFINITION_SQL),
    ("event_factor_map", EVENT_FACTOR_MAP_SQL),
    ("factor_value", FACTOR_VALUE_SQL),
    ("event_trigger", EVENT_TRIGGER_SQL),
    ("market_index_bar", MARKET_INDEX_BAR_SQL),
]

INDEX_SQLS = [
    ("stock_k_data", "CREATE INDEX IF NOT EXISTS idx_skd_code_freq_time ON stock_k_data(ts_code, freq, bar_time);"),
    ("stock_pools", "CREATE INDEX IF NOT EXISTS idx_pools_ts_code ON stock_pools(ts_code);"),
    ("factor_value", FACTOR_VALUE_INDEX_SQL),
    ("event_trigger", EVENT_TRIGGER_INDEX_SQL),
]


def init_schema(dry_run: bool = False):
    """初始化数据库 schema"""
    engine = get_engine()

    print("=" * 60)
    print("数据库 Schema 初始化")
    print("=" * 60)

    with engine.begin() as conn:
        for table_name, sql in ALL_TABLES:
            exists = table_exists(conn, table_name)
            if exists:
                print(f"  [SKIP] {table_name} 已存在")
                continue

            if dry_run:
                print(f"  [DRY-RUN] 将创建 {table_name}")
                continue

            try:
                conn.execute(text(sql))
                print(f"  [OK] {table_name} 创建成功")
            except Exception as e:
                print(f"  [FAIL] {table_name} 创建失败: {e}")
                raise

        # 创建索引
        for table_name, sql in INDEX_SQLS:
            if dry_run:
                print(f"  [DRY-RUN] 将创建 {table_name} 索引")
                continue
            try:
                conn.execute(text(sql))
                print(f"  [OK] {table_name} 索引创建成功")
            except Exception as e:
                print(f"  [WARN] {table_name} 索引创建失败: {e}")

    print("=" * 60)
    print("Schema 初始化完成")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="初始化数据库 Schema")
    parser.add_argument("--dry-run", action="store_true", help="试运行，不实际创建表")
    args = parser.parse_args()

    init_schema(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
