# -*- coding: utf-8 -*-
"""
数据库表结构定义 - 统一管理数据库表结构（PostgreSQL）

所有特征表的 DDL 定义集中在此文件，便于维护和查阅。
"""
from typing import Dict, List

TABLE_DEFINITIONS: Dict[str, str] = {}


K_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS stock_k_data (
    id BIGSERIAL PRIMARY KEY,
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
    id BIGSERIAL PRIMARY KEY,
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


POPULARITY_RANK_TABLE = """
CREATE TABLE IF NOT EXISTS stock_popularity_rank (
    id BIGSERIAL PRIMARY KEY,
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
    id BIGSERIAL PRIMARY KEY,
    stock_name VARCHAR(50) NOT NULL UNIQUE,
    ts_code VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_stock_list_name ON stock_list(stock_name);
CREATE INDEX IF NOT EXISTS idx_stock_list_code ON stock_list(ts_code);
"""
TABLE_DEFINITIONS["stock_list"] = STOCK_LIST_TABLE


STOCK_POOLS_TABLE = """
CREATE TABLE IF NOT EXISTS stock_pools (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(50) NOT NULL,
    concepts TEXT,
    popularity_rank INTEGER,
    market_cap FLOAT,
    total_market_cap FLOAT,
    industry_l2 VARCHAR(100),
    industry_l3 VARCHAR(200),
    industry_pe FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pools_ts_code ON stock_pools(ts_code);
CREATE INDEX IF NOT EXISTS idx_pools_name ON stock_pools(name);
CREATE INDEX IF NOT EXISTS idx_pools_popularity ON stock_pools(popularity_rank);
CREATE INDEX IF NOT EXISTS idx_pools_industry_l2 ON stock_pools(industry_l2);
CREATE INDEX IF NOT EXISTS idx_pools_industry_l3 ON stock_pools(industry_l3);
"""
TABLE_DEFINITIONS["stock_pools"] = STOCK_POOLS_TABLE


STOCK_PICKER_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS stock_picker_results (
    id BIGSERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    stock_name VARCHAR(50) NOT NULL,
    ts_code VARCHAR(20) NOT NULL,
    amp_passed BOOLEAN,
    div_passed BOOLEAN,
    hist_passed BOOLEAN,
    all_passed BOOLEAN,
    amp_strength FLOAT,
    mid_slope FLOAT,
    upper_slope FLOAT,
    lower_slope FLOAT,
    close_pos FLOAT,
    channel_width FLOAT,
    period INTEGER,
    window_len INTEGER,
    history_peak_bar_idx INTEGER,
    history_peak_amp_strength FLOAT,
    history_peak_mid_slope FLOAT,
    history_peak_close_pos FLOAT,
    divergence_bar_idx INTEGER,
    divergence_indicator VARCHAR(20),
    divergence_type VARCHAR(50),
    divergence_distance FLOAT,
    divergence_age INTEGER,
    divergence_count INTEGER,
    limit_up_count_in_period INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_picker_trade_date ON stock_picker_results(trade_date);
CREATE INDEX IF NOT EXISTS idx_picker_ts_code ON stock_picker_results(ts_code);
CREATE INDEX IF NOT EXISTS idx_picker_all_passed ON stock_picker_results(all_passed);
CREATE INDEX IF NOT EXISTS idx_picker_amp_passed ON stock_picker_results(amp_passed);
"""
TABLE_DEFINITIONS["stock_picker_results"] = STOCK_PICKER_RESULTS_TABLE


MINUTE_FLOW_ANALYSIS_TABLE = """
CREATE TABLE IF NOT EXISTS stock_minute_flow_analysis (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    concepts TEXT,
    trade_date DATE NOT NULL,
    analysis_time TIMESTAMP NOT NULL,
    data_days INTEGER,
    volume_zscore FLOAT,
    flow_zscore FLOAT,
    current_volume FLOAT,
    current_flow FLOAT,
    history_volume_mean FLOAT,
    history_flow_mean FLOAT,
    history_volume_std FLOAT,
    history_flow_std FLOAT,
    history_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, analysis_time)
);
CREATE INDEX IF NOT EXISTS idx_minute_flow_ts_code ON stock_minute_flow_analysis(ts_code);
CREATE INDEX IF NOT EXISTS idx_minute_flow_trade_date ON stock_minute_flow_analysis(trade_date);
CREATE INDEX IF NOT EXISTS idx_minute_flow_analysis_time ON stock_minute_flow_analysis(analysis_time);
"""
TABLE_DEFINITIONS["stock_minute_flow_analysis"] = MINUTE_FLOW_ANALYSIS_TABLE


CONCEPT_FLOW_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS stock_concept_flow_stats (
    id BIGSERIAL PRIMARY KEY,
    analysis_time TIMESTAMP NOT NULL,
    concept VARCHAR(100) NOT NULL,
    stock_count INTEGER,
    total_flow FLOAT,
    avg_flow FLOAT,
    total_flow_zscore FLOAT,
    avg_flow_zscore FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(analysis_time, concept)
);
CREATE INDEX IF NOT EXISTS idx_concept_flow_analysis_time ON stock_concept_flow_stats(analysis_time);
CREATE INDEX IF NOT EXISTS idx_concept_flow_concept ON stock_concept_flow_stats(concept);
"""
TABLE_DEFINITIONS["stock_concept_flow_stats"] = CONCEPT_FLOW_STATS_TABLE


FINANCIAL_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS stock_financial_data (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    report_date VARCHAR(8) NOT NULL,
    营业收入 FLOAT,
    营业成本 FLOAT,
    营业利润 FLOAT,
    归属于母公司股东的净利润 FLOAT,
    扣除非经常性损益后的归属于母公司股东的净利润 FLOAT,
    销售费用 FLOAT,
    管理费用 FLOAT,
    研发费用 FLOAT,
    财务费用 FLOAT,
    投资收益 FLOAT,
    公允价值变动收益 FLOAT,
    信用减值损失 FLOAT,
    资产减值损失 FLOAT,
    经营活动产生的现金流量净额 FLOAT,
    应收账款 FLOAT,
    应收款项融资 FLOAT,
    预付款项 FLOAT,
    其他应收款 FLOAT,
    存货 FLOAT,
    合同资产 FLOAT,
    商誉 FLOAT,
    非经常性损益 FLOAT,
    应付账款 FLOAT,
    应收账款周转率 FLOAT,
    存货周转率 FLOAT,
    总资产周转率 FLOAT,
    货币资金 FLOAT,
    短期借款 FLOAT,
    一年内到期的非流动负债 FLOAT,
    长期借款 FLOAT,
    应付债券 FLOAT,
    负债合计 FLOAT,
    股东权益合计 FLOAT,
    资产负债率 FLOAT,
    流动比率 FLOAT,
    速动比率 FLOAT,
    现金比率 FLOAT,
    利息保障倍数 FLOAT,
    购建固定资产无形资产和其他长期资产支付的现金 FLOAT,
    固定资产 FLOAT,
    在建工程 FLOAT,
    无形资产 FLOAT,
    开发支出 FLOAT,
    固定资产周转率 FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, report_date)
);
CREATE INDEX IF NOT EXISTS idx_financial_ts_code ON stock_financial_data(ts_code);
CREATE INDEX IF NOT EXISTS idx_financial_report_date ON stock_financial_data(report_date);
CREATE INDEX IF NOT EXISTS idx_financial_stock_name ON stock_financial_data(stock_name);
"""
TABLE_DEFINITIONS["stock_financial_data"] = FINANCIAL_DATA_TABLE


FINANCIAL_SUMMARY_TABLE = """
CREATE TABLE IF NOT EXISTS stock_financial_summary (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    report_date VARCHAR(8) NOT NULL,
    营业总收入 FLOAT,
    营业收入 FLOAT,
    营业成本 FLOAT,
    营业利润 FLOAT,
    归母净利润 FLOAT,
    少数股东损益 FLOAT,
    EBIT FLOAT,
    经营活动现金流净额 FLOAT,
    资本开支 FLOAT,
    销售商品提供劳务收到的现金 FLOAT,
    总资产 FLOAT,
    应收账款 FLOAT,
    存货 FLOAT,
    应付账款 FLOAT,
    合同负债 FLOAT,
    EBITDA FLOAT,
    研发费用 FLOAT,
    期末现金 FLOAT,
    股东权益 FLOAT,
    查询问句 VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, report_date)
);
CREATE INDEX IF NOT EXISTS idx_summary_ts_code ON stock_financial_summary(ts_code);
CREATE INDEX IF NOT EXISTS idx_summary_report_date ON stock_financial_summary(report_date);
"""
TABLE_DEFINITIONS["stock_financial_summary"] = FINANCIAL_SUMMARY_TABLE


FINANCIAL_QUARTERLY_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS financial_quarterly_data (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    end_date VARCHAR(8) NOT NULL,
    report_type INTEGER,
    ann_date VARCHAR(8),
    f_ann_date VARCHAR(8),
    total_revenue FLOAT,
    revenue FLOAT,
    oper_cost FLOAT,
    operate_profit FLOAT,
    ebit FLOAT,
    ebitda FLOAT,
    n_income FLOAT,
    n_income_attr_p FLOAT,
    rd_exp FLOAT,
    n_cashflow_act FLOAT,
    free_cashflow FLOAT,
    c_fr_sale_sg FLOAT,
    c_pay_acq_const_fiolta FLOAT,
    total_assets FLOAT,
    accounts_receiv FLOAT,
    inventories FLOAT,
    accounts_pay FLOAT,
    contract_liab FLOAT,
    total_hldr_eqy_exc_min_int FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, end_date, report_type)
);
CREATE INDEX IF NOT EXISTS idx_financial_quarterly_ts_code ON financial_quarterly_data(ts_code);
CREATE INDEX IF NOT EXISTS idx_financial_quarterly_end_date ON financial_quarterly_data(end_date);
"""
TABLE_DEFINITIONS["financial_quarterly_data"] = FINANCIAL_QUARTERLY_DATA_TABLE


FINANCIAL_SCORE_POOL_TABLE = """
CREATE TABLE IF NOT EXISTS stock_financial_score_pool (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    report_date VARCHAR(8) NOT NULL,
    total_score FLOAT,
    规模与增长_score FLOAT,
    盈利能力_score FLOAT,
    利润质量_score FLOAT,
    现金创造能力_score FLOAT,
    资产效率与资金占用_score FLOAT,
    边际变化与持续性_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, report_date)
);
CREATE INDEX IF NOT EXISTS idx_score_pool_ts_code ON stock_financial_score_pool(ts_code);
CREATE INDEX IF NOT EXISTS idx_score_pool_report_date ON stock_financial_score_pool(report_date);
"""


SMC_PULLBACK_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS stock_smc_pullback_results (
    id BIGSERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    timeframe_set VARCHAR(20) NOT NULL DEFAULT 'd_60m_15m',
    passed BOOLEAN,
    score FLOAT,
    daily_recent_bullish_bos BOOLEAN,
    daily_recent_bos_bars_ago INTEGER,
    daily_higher_low_ok BOOLEAN,
    daily_prev_swing_low FLOAT,
    daily_current_swing_low FLOAT,
    daily_close FLOAT,
    daily_trend_bias INTEGER,
    daily_reason TEXT,
    intraday_has_displacement_up BOOLEAN,
    intraday_displacement_time VARCHAR(50),
    intraday_impulse_low FLOAT,
    intraday_impulse_high FLOAT,
    intraday_retrace_50 FLOAT,
    intraday_current_price FLOAT,
    intraday_in_discount BOOLEAN,
    intraday_above_key_swing_low BOOLEAN,
    intraday_reason TEXT,
    avg_amount_5d FLOAT,
    liq_passed BOOLEAN,
    liq_reason TEXT,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, symbol, timeframe_set)
);
CREATE INDEX IF NOT EXISTS idx_smc_pullback_trade_date ON stock_smc_pullback_results(trade_date);
CREATE INDEX IF NOT EXISTS idx_smc_pullback_symbol ON stock_smc_pullback_results(symbol);
CREATE INDEX IF NOT EXISTS idx_smc_pullback_passed ON stock_smc_pullback_results(passed);
CREATE INDEX IF NOT EXISTS idx_smc_pullback_score ON stock_smc_pullback_results(score);
CREATE INDEX IF NOT EXISTS idx_smc_pullback_timeframe ON stock_smc_pullback_results(timeframe_set);
"""
TABLE_DEFINITIONS["stock_smc_pullback_results"] = SMC_PULLBACK_RESULTS_TABLE


SELECTION_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS stock_selection_results (
    id BIGSERIAL PRIMARY KEY,
    batch_id VARCHAR(32) NOT NULL,
    selection_name VARCHAR(50) NOT NULL,
    selection_date DATE NOT NULL,
    ts_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    report_date VARCHAR(8) NOT NULL,
    total_score FLOAT,
    margin_score FLOAT,
    scale_growth_score FLOAT,
    profitability_score FLOAT,
    profit_quality_score FLOAT,
    cash_creation_score FLOAT,
    asset_efficiency_score FLOAT,
    q_rev_yoy_delta FLOAT,
    q_np_parent_yoy_delta FLOAT,
    trend_consistency FLOAT,
    ann_date VARCHAR(8),
    filter_condition JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(batch_id, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_selection_batch_id ON stock_selection_results(batch_id);
CREATE INDEX IF NOT EXISTS idx_selection_date ON stock_selection_results(selection_date);
CREATE INDEX IF NOT EXISTS idx_selection_ts_code ON stock_selection_results(ts_code);
CREATE INDEX IF NOT EXISTS idx_selection_report_date ON stock_selection_results(report_date);
CREATE INDEX IF NOT EXISTS idx_selection_margin_score ON stock_selection_results(margin_score);
"""
TABLE_DEFINITIONS["stock_selection_results"] = SELECTION_RESULTS_TABLE


SMC_REVERSAL_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS stock_smc_reversal_results (
    id BIGSERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    timeframe_set VARCHAR(20) NOT NULL DEFAULT 'd_60m_15m',
    passed BOOLEAN,
    score FLOAT,

    structure_recent_bullish_bos BOOLEAN,
    structure_recent_bos_bars_ago INTEGER,
    structure_higher_low_ok BOOLEAN,
    structure_prev_swing_low FLOAT,
    structure_current_swing_low FLOAT,
    structure_close FLOAT,
    structure_trend_bias INTEGER,
    structure_reason TEXT,

    pullback_has_sweep BOOLEAN,
    pullback_sweep_time VARCHAR(50),
    pullback_targeted_swing_low FLOAT,
    pullback_sweep_low FLOAT,
    pullback_reclaim_level FLOAT,
    pullback_reclaim_close FLOAT,
    pullback_sweep_depth_atr FLOAT,
    pullback_current_close FLOAT,
    pullback_reason TEXT,

    entry_has_reversal_confirmation BOOLEAN,
    entry_confirm_time VARCHAR(50),
    entry_confirm_break_level FLOAT,
    entry_confirm_close FLOAT,
    entry_confirm_high FLOAT,
    entry_current_close FLOAT,
    entry_current_low FLOAT,
    entry_reason TEXT,

    avg_amount_5d FLOAT,
    liq_passed BOOLEAN,
    liq_reason TEXT,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, symbol, timeframe_set)
);
CREATE INDEX IF NOT EXISTS idx_smc_reversal_trade_date ON stock_smc_reversal_results(trade_date);
CREATE INDEX IF NOT EXISTS idx_smc_reversal_symbol ON stock_smc_reversal_results(symbol);
CREATE INDEX IF NOT EXISTS idx_smc_reversal_passed ON stock_smc_reversal_results(passed);
CREATE INDEX IF NOT EXISTS idx_smc_reversal_score ON stock_smc_reversal_results(score);
CREATE INDEX IF NOT EXISTS idx_smc_reversal_timeframe ON stock_smc_reversal_results(timeframe_set);
"""
TABLE_DEFINITIONS["stock_smc_reversal_results"] = SMC_REVERSAL_RESULTS_TABLE


WEEKLY_VWAP_OB_PICKER_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS stock_weekly_vwap_ob_picker_results (
    id BIGSERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    stock_name VARCHAR(50) NOT NULL,
    ts_code VARCHAR(20) NOT NULL,
    weekly_bar_time TIMESTAMP,
    daily_bar_time TIMESTAMP,
    weekly_dir_age INTEGER,
    hard_cond_weekly_dir_up BOOLEAN,
    cond_weekly_near_vwap BOOLEAN,
    cond_weekly_bullish_ob BOOLEAN,
    cond_daily_bullish_ob BOOLEAN,
    cond_lhhh_body_hit BOOLEAN,
    lhhh_timeframe VARCHAR(5),
    cond_daily_macd_cross BOOLEAN,
    cond_weekly_in_support_channel BOOLEAN,
    cond_daily_in_support_channel BOOLEAN,
    selected BOOLEAN,
    hit_sources TEXT,
    weekly_close FLOAT,
    weekly_vwap FLOAT,
    weekly_dir INTEGER,
    weekly_vwap_dev_pct FLOAT,
    weekly_vwap_dev_pct_abs FLOAT,
    weekly_pivot_type VARCHAR(5),
    weekly_pivot_idx INTEGER,
    weekly_vwap_max_dev_pct FLOAT,
    weekly_vol_zscore_max FLOAT,
    weekly_ob_distance_pct FLOAT,
    daily_close FLOAT,
    daily_ob_distance_pct FLOAT,
    daily_internal_ob_distance_pct FLOAT,
    daily_swing_ob_distance_pct FLOAT,
    daily_dif FLOAT,
    daily_dea FLOAT,
    daily_macd FLOAT,
    weekly_nearest_support_hi FLOAT,
    weekly_nearest_support_lo FLOAT,
    daily_nearest_support_hi FLOAT,
    daily_nearest_support_lo FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trade_date, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_vwap_picker_trade_date ON stock_weekly_vwap_ob_picker_results(trade_date);
CREATE INDEX IF NOT EXISTS idx_vwap_picker_ts_code ON stock_weekly_vwap_ob_picker_results(ts_code);
CREATE INDEX IF NOT EXISTS idx_vwap_picker_selected ON stock_weekly_vwap_ob_picker_results(selected);
CREATE INDEX IF NOT EXISTS idx_vwap_picker_weekly_dir ON stock_weekly_vwap_ob_picker_results(weekly_dir);
"""
TABLE_DEFINITIONS["stock_weekly_vwap_ob_picker_results"] = WEEKLY_VWAP_OB_PICKER_RESULTS_TABLE


AMP_FEATURES_TABLE = """
CREATE TABLE IF NOT EXISTS stock_amp_features (
    id BIGSERIAL PRIMARY KEY,
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


STOCK_TOP10_HOLDERS_TUSHARE_TABLE = """
CREATE TABLE IF NOT EXISTS stock_top10_holders_tushare (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    report_date VARCHAR(8) NOT NULL,
    ann_date VARCHAR(8),
    holder_rank INTEGER NOT NULL,
    holder_name VARCHAR(200),
    holder_type VARCHAR(50),
    hold_amount FLOAT,
    hold_ratio FLOAT,
    hold_float_ratio FLOAT,
    hold_change FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, report_date, holder_rank)
);
CREATE INDEX IF NOT EXISTS idx_top10_tushare_ts_code ON stock_top10_holders_tushare(ts_code);
CREATE INDEX IF NOT EXISTS idx_top10_tushare_report_date ON stock_top10_holders_tushare(report_date);
CREATE INDEX IF NOT EXISTS idx_top10_tushare_ann_date ON stock_top10_holders_tushare(ann_date);
"""
TABLE_DEFINITIONS["stock_top10_holders_tushare"] = STOCK_TOP10_HOLDERS_TUSHARE_TABLE


BREAKOUT_DIR_TURN_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS breakout_dir_turn_events (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    name VARCHAR(50),
    event_time TIMESTAMP NOT NULL,
    freq VARCHAR(10) NOT NULL,
    breakout_quality_score FLOAT,
    breakout_quality_grade VARCHAR(10),
    score_trend_total FLOAT,
    score_candle_total FLOAT,
    score_volume_total FLOAT,
    score_freshness_total FLOAT,
    score_bg_rope_slope FLOAT,
    score_bg_dist_to_rope FLOAT,
    score_bg_consolidation FLOAT,
    score_candle_close_pos FLOAT,
    score_candle_body_to_range FLOAT,
    score_candle_upper_wick FLOAT,
    score_volume_vol_z FLOAT,
    score_volume_vol_record FLOAT,
    score_freshness_count INTEGER,
    score_freshness_cum_gain FLOAT,
    breakout_action VARCHAR(20),
    breakout_freshness_count INTEGER,
    breakout_freshness_cum_gain FLOAT,
    rope_slope_atr_5 FLOAT,
    dist_to_rope_atr FLOAT,
    consolidation_bars INTEGER,
    vol_zscore FLOAT,
    vol_record_days INTEGER,
    dir_turn_upper_price FLOAT,
    dir_turn_atr_raw FLOAT,
    dir_turn_tol_price FLOAT,
    dir_turn_band_low FLOAT,
    dir_turn_band_high FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, freq, event_time)
);
CREATE INDEX IF NOT EXISTS idx_dir_turn_ts_code ON breakout_dir_turn_events(ts_code);
CREATE INDEX IF NOT EXISTS idx_dir_turn_event_time ON breakout_dir_turn_events(event_time);
CREATE INDEX IF NOT EXISTS idx_dir_turn_freq ON breakout_dir_turn_events(freq);
"""
TABLE_DEFINITIONS["breakout_dir_turn_events"] = BREAKOUT_DIR_TURN_EVENTS_TABLE


BREAKOUT_PULLBACK_BUY_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS breakout_pullback_buy_events (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    name VARCHAR(50),
    buy_time TIMESTAMP NOT NULL,
    freq VARCHAR(10) NOT NULL,
    buy_type VARCHAR(20),
    source_breakout_time VARCHAR(50),
    source_breakout_index INTEGER,
    breakout_to_buy_bars INTEGER,
    breakout_quality_score FLOAT,
    score_trend_total FLOAT,
    score_candle_total FLOAT,
    score_volume_total FLOAT,
    score_freshness_total FLOAT,
    signal_note VARCHAR(100),
    pullback_touch_support_flag INTEGER,
    pullback_hhhl_seen_flag INTEGER,
    dir_turn_upper_price FLOAT,
    dir_turn_atr_raw FLOAT,
    dir_turn_tol_price FLOAT,
    dir_turn_band_low FLOAT,
    dir_turn_band_high FLOAT,
    rope FLOAT,
    close FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ts_code, freq, buy_time)
);
CREATE INDEX IF NOT EXISTS idx_pullback_ts_code ON breakout_pullback_buy_events(ts_code);
CREATE INDEX IF NOT EXISTS idx_pullback_buy_time ON breakout_pullback_buy_events(buy_time);
CREATE INDEX IF NOT EXISTS idx_pullback_freq ON breakout_pullback_buy_events(freq);
"""
TABLE_DEFINITIONS["breakout_pullback_buy_events"] = BREAKOUT_PULLBACK_BUY_EVENTS_TABLE


STOCK_ANOMALY_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS stock_anomaly_signals (
    id BIGSERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    ts_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    concepts TEXT,
    close_price FLOAT,
    amplitude FLOAT,
    volume_ratio FLOAT,
    turnover_rate FLOAT,
    abnormal_score FLOAT,
    zscore FLOAT,
    rank INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(snapshot_date, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_anomaly_snapshot ON stock_anomaly_signals(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_anomaly_ts_code ON stock_anomaly_signals(ts_code);
CREATE INDEX IF NOT EXISTS idx_anomaly_rank ON stock_anomaly_signals(rank);
"""
TABLE_DEFINITIONS["stock_anomaly_signals"] = STOCK_ANOMALY_SIGNALS_TABLE


STOCK_ANOMALY_SIGNALS_ROLLING_TABLE = """
CREATE TABLE IF NOT EXISTS stock_anomaly_signals_rolling (
    id BIGSERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    ts_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    concepts TEXT,
    close_price FLOAT,
    amplitude FLOAT,
    volume_ratio FLOAT,
    turnover_rate FLOAT,
    abnormal_score FLOAT,
    zscore FLOAT,
    rank INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(snapshot_date, ts_code)
);
CREATE INDEX IF NOT EXISTS idx_anomaly_rolling_snapshot ON stock_anomaly_signals_rolling(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_anomaly_rolling_ts_code ON stock_anomaly_signals_rolling(ts_code);
"""
TABLE_DEFINITIONS["stock_anomaly_signals_rolling"] = STOCK_ANOMALY_SIGNALS_ROLLING_TABLE


CONCEPT_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS concept_signals (
    id BIGSERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    concept VARCHAR(100) NOT NULL,
    stock_count INTEGER,
    avg_zscore FLOAT,
    strength FLOAT,
    concept_coverage FLOAT,
    total_zscore FLOAT,
    top_stocks TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(snapshot_date, concept)
);
CREATE INDEX IF NOT EXISTS idx_concept_signal_snapshot ON concept_signals(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_concept_signal_concept ON concept_signals(concept);
"""
TABLE_DEFINITIONS["concept_signals"] = CONCEPT_SIGNALS_TABLE


CONCEPT_SIGNALS_ROLLING_TABLE = """
CREATE TABLE IF NOT EXISTS concept_signals_rolling (
    id BIGSERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    concept VARCHAR(100) NOT NULL,
    stock_count INTEGER,
    avg_zscore FLOAT,
    strength FLOAT,
    concept_coverage FLOAT,
    total_zscore FLOAT,
    top_stocks TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(snapshot_date, concept)
);
CREATE INDEX IF NOT EXISTS idx_concept_rolling_snapshot ON concept_signals_rolling(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_concept_rolling_concept ON concept_signals_rolling(concept);
"""
TABLE_DEFINITIONS["concept_signals_rolling"] = CONCEPT_SIGNALS_ROLLING_TABLE


THEME_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS theme_signals (
    id BIGSERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    theme VARCHAR(200) NOT NULL,
    avg_zscore FLOAT,
    strength FLOAT,
    concept_coverage FLOAT,
    stock_count INTEGER,
    anomalous_concept_count INTEGER,
    total_concept_count INTEGER,
    total_zscore FLOAT,
    top_stocks TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(snapshot_date, theme)
);
CREATE INDEX IF NOT EXISTS idx_theme_signal_snapshot ON theme_signals(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_theme_signal_theme ON theme_signals(theme);
"""
TABLE_DEFINITIONS["theme_signals"] = THEME_SIGNALS_TABLE


THEME_SIGNALS_ROLLING_TABLE = """
CREATE TABLE IF NOT EXISTS theme_signals_rolling (
    id BIGSERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    theme VARCHAR(200) NOT NULL,
    avg_zscore FLOAT,
    strength FLOAT,
    concept_coverage FLOAT,
    stock_count INTEGER,
    anomalous_concept_count INTEGER,
    total_concept_count INTEGER,
    total_zscore FLOAT,
    top_stocks TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(snapshot_date, theme)
);
CREATE INDEX IF NOT EXISTS idx_theme_rolling_snapshot ON theme_signals_rolling(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_theme_rolling_theme ON theme_signals_rolling(theme);
"""
TABLE_DEFINITIONS["theme_signals_rolling"] = THEME_SIGNALS_ROLLING_TABLE


STOCK_WATCHLIST_TABLE = """
CREATE TABLE IF NOT EXISTS stock_watchlist (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL UNIQUE,
    stock_name VARCHAR(50),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_watchlist_ts_code ON stock_watchlist(ts_code);
"""
TABLE_DEFINITIONS["stock_watchlist"] = STOCK_WATCHLIST_TABLE


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


def get_table_names() -> List[str]:
    """获取所有表名列表"""
    return list(TABLE_DEFINITIONS.keys())


def get_create_sql(table_name: str) -> str:
    """获取指定表的创建 SQL（PostgreSQL）

    Args:
        table_name: 表名

    Returns:
        PostgreSQL 建表 SQL
    """
    if table_name not in TABLE_DEFINITIONS:
        raise ValueError(f"未知的表名：{table_name}")
    return TABLE_DEFINITIONS[table_name]