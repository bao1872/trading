# -*- coding: utf-8 -*-
"""
数据模型定义 - 统一管理数据库表结构

所有特征表的 DDL 定义集中在此文件，便于维护和查阅。
"""
from typing import Dict, List

TABLE_DEFINITIONS: Dict[str, str] = {}


K_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS stock_k_data (
    id SERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    freq VARCHAR(10) NOT NULL,
    bar_time TIMESTAMP NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, freq, bar_time)
);
CREATE INDEX IF NOT EXISTS idx_k_data_ts_code ON stock_k_data(ts_code);
CREATE INDEX IF NOT EXISTS idx_k_data_freq ON stock_k_data(freq);
CREATE INDEX IF NOT EXISTS idx_k_data_bar_time ON stock_k_data(bar_time);
"""
TABLE_DEFINITIONS["stock_k_data"] = K_DATA_TABLE


DIV_FEATURES_TABLE = """
CREATE TABLE IF NOT EXISTS stock_div_features (
    id SERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    freq VARCHAR(10) NOT NULL,
    bar_time TIMESTAMP NOT NULL,
    total_div_count INTEGER,
    macd_has_div INTEGER,
    macd_div_type INTEGER,
    macd_div_len INTEGER,
    macd_pos_reg INTEGER,
    macd_neg_reg INTEGER,
    macd_pos_hid INTEGER,
    macd_neg_hid INTEGER,
    hist_has_div INTEGER,
    hist_div_type INTEGER,
    hist_div_len INTEGER,
    hist_pos_reg INTEGER,
    hist_neg_reg INTEGER,
    hist_pos_hid INTEGER,
    hist_neg_hid INTEGER,
    obv_has_div INTEGER,
    obv_div_type INTEGER,
    obv_div_len INTEGER,
    obv_pos_reg INTEGER,
    obv_neg_reg INTEGER,
    obv_pos_hid INTEGER,
    obv_neg_hid INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, freq, bar_time)
);
CREATE INDEX IF NOT EXISTS idx_div_features_ts_code ON stock_div_features(ts_code);
CREATE INDEX IF NOT EXISTS idx_div_features_freq ON stock_div_features(freq);
CREATE INDEX IF NOT EXISTS idx_div_features_bar_time ON stock_div_features(bar_time);
"""
TABLE_DEFINITIONS["stock_div_features"] = DIV_FEATURES_TABLE


AMP_FEATURES_TABLE = """
CREATE TABLE IF NOT EXISTS stock_amp_features (
    id SERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    name VARCHAR(50),
    freq VARCHAR(10) NOT NULL,
    bar_time TIMESTAMP NOT NULL,
    window_len INTEGER,
    final_period INTEGER,
    pearson_r FLOAT,
    strength_pr FLOAT,
    bar_close FLOAT,
    bar_upper FLOAT,
    bar_lower FLOAT,
    close_pos_0_1 FLOAT,
    activity_pos_0_1 FLOAT,
    upper_ret_per_bar FLOAT,
    upper_total_ret FLOAT,
    lower_ret_per_bar FLOAT,
    lower_total_ret FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, freq, bar_time)
);
CREATE INDEX IF NOT EXISTS idx_amp_features_ts_code ON stock_amp_features(ts_code);
CREATE INDEX IF NOT EXISTS idx_amp_features_freq ON stock_amp_features(freq);
CREATE INDEX IF NOT EXISTS idx_amp_features_bar_time ON stock_amp_features(bar_time);
"""
TABLE_DEFINITIONS["stock_amp_features"] = AMP_FEATURES_TABLE


DIV_FEATURES_UPSERT_SQL = """
INSERT INTO stock_div_features 
(ts_code, freq, bar_time, total_div_count, macd_has_div, macd_div_type, macd_div_len,
 hist_has_div, hist_div_type, hist_div_len, obv_has_div, obv_div_type, obv_div_len)
VALUES (:ts_code, :freq, :bar_time, :total_div_count, :macd_has_div, :macd_div_type, :macd_div_len,
        :hist_has_div, :hist_div_type, :hist_div_len, :obv_has_div, :obv_div_type, :obv_div_len)
ON CONFLICT (ts_code, freq, bar_time) DO UPDATE SET
    total_div_count = EXCLUDED.total_div_count,
    macd_has_div = EXCLUDED.macd_has_div,
    macd_div_type = EXCLUDED.macd_div_type,
    macd_div_len = EXCLUDED.macd_div_len,
    hist_has_div = EXCLUDED.hist_has_div,
    hist_div_type = EXCLUDED.hist_div_type,
    hist_div_len = EXCLUDED.hist_div_len,
    obv_has_div = EXCLUDED.obv_has_div,
    obv_div_type = EXCLUDED.obv_div_type,
    obv_div_len = EXCLUDED.obv_div_len
"""


AMP_FEATURES_UPSERT_SQL = """
INSERT INTO stock_amp_features 
(ts_code, name, freq, bar_time, window_len, final_period, pearson_r, strength_pr,
 bar_close, bar_upper, bar_lower, close_pos_0_1, activity_pos_0_1,
 upper_ret_per_bar, upper_total_ret, lower_ret_per_bar, lower_total_ret)
VALUES (:ts_code, :name, :freq, :bar_time, :window_len, :final_period, :pearson_r, :strength_pr,
        :bar_close, :bar_upper, :bar_lower, :close_pos_0_1, :activity_pos_0_1,
        :upper_ret_per_bar, :upper_total_ret, :lower_ret_per_bar, :lower_total_ret)
ON CONFLICT (ts_code, freq, bar_time) DO UPDATE SET
    name = EXCLUDED.name,
    window_len = EXCLUDED.window_len,
    final_period = EXCLUDED.final_period,
    pearson_r = EXCLUDED.pearson_r,
    strength_pr = EXCLUDED.strength_pr,
    bar_close = EXCLUDED.bar_close,
    bar_upper = EXCLUDED.bar_upper,
    bar_lower = EXCLUDED.bar_lower,
    close_pos_0_1 = EXCLUDED.close_pos_0_1,
    activity_pos_0_1 = EXCLUDED.activity_pos_0_1,
    upper_ret_per_bar = EXCLUDED.upper_ret_per_bar,
    upper_total_ret = EXCLUDED.upper_total_ret,
    lower_ret_per_bar = EXCLUDED.lower_ret_per_bar,
    lower_total_ret = EXCLUDED.lower_total_ret
"""


POPULARITY_RANK_TABLE = """
CREATE TABLE IF NOT EXISTS stock_popularity_rank (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    ts_code VARCHAR(20) NOT NULL,
    name VARCHAR(50),
    rank INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_pop_rank_trade_date ON stock_popularity_rank(trade_date);
CREATE INDEX IF NOT EXISTS idx_pop_rank_ts_code ON stock_popularity_rank(ts_code);
CREATE INDEX IF NOT EXISTS idx_pop_rank_rank ON stock_popularity_rank(rank);
"""
TABLE_DEFINITIONS["stock_popularity_rank"] = POPULARITY_RANK_TABLE

STOCK_LIST_TABLE = """
CREATE TABLE IF NOT EXISTS stock_list (
    id SERIAL PRIMARY KEY,
    stock_name VARCHAR(50) NOT NULL UNIQUE,
    ts_code VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_stock_list_name ON stock_list(stock_name);
CREATE INDEX IF NOT EXISTS idx_stock_list_code ON stock_list(ts_code);
"""
TABLE_DEFINITIONS["stock_list"] = STOCK_LIST_TABLE


STOCK_CONCEPTS_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS stock_concepts_cache (
    id SERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(50) NOT NULL,
    concepts TEXT,
    popularity_rank INTEGER,
    market_cap FLOAT,
    total_market_cap FLOAT,
    industry VARCHAR(100),
    industry_pe FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_concepts_ts_code ON stock_concepts_cache(ts_code);
CREATE INDEX IF NOT EXISTS idx_concepts_name ON stock_concepts_cache(name);
CREATE INDEX IF NOT EXISTS idx_concepts_popularity ON stock_concepts_cache(popularity_rank);
CREATE INDEX IF NOT EXISTS idx_concepts_industry ON stock_concepts_cache(industry);
"""
TABLE_DEFINITIONS["stock_concepts_cache"] = STOCK_CONCEPTS_CACHE_TABLE


def get_table_names() -> List[str]:
    """获取所有表名列表"""
    return list(TABLE_DEFINITIONS.keys())


def get_create_sql(table_name: str) -> str:
    """获取指定表的创建 SQL"""
    if table_name not in TABLE_DEFINITIONS:
        raise ValueError(f"未知的表名：{table_name}")
    return TABLE_DEFINITIONS[table_name]
