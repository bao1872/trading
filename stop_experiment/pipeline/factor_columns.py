# -*- coding: utf-8 -*-
"""
因子列名统一定义文件（SSOT）

Purpose:
    所有 stop_experiment 脚本统一从此文件导入因子列名，禁止在各自脚本中重复定义。

    设计原则:
    - 只保留可跨股票比较的因子：归一化/百分比/z-score/比率/计数/布尔
    - 剔除绝对值因子：绝对价格/绝对成交量/绝对波动值
    - 因子类别标记用于分组重要性分析

Pipeline Position:
    共享配置（被所有脚本 import）。
    上游: —
    下游: 01_build_dataset.py, compute_factors.py

Inputs:
    - 无（纯列名定义）

Outputs:
    - 无（被 import 使用）

How to Run:
    无需单独运行，被其他脚本 import

Side Effects:
    - 无
"""

# ==================== SLC静态因子（来自stop_loss_selection，已剔除绝对值） ====================
SLC_STATIC_COLS = [
    # 注：sell_stop_triggered / buy_stop_triggered 已剔除（gain=0）
    "active_sell_cluster_count",    # int, 活跃卖cluster数（计数，可比）
    "active_buy_cluster_count",     # int, 活跃买cluster数（计数，可比，88%缺失→0填充）
    "dist_to_nearest_sell_stop_atr",  # float, ATR归一化距离（可比）
    "dist_to_nearest_buy_stop_atr",   # float, ATR归一化距离（可比，88%缺失→0填充）
    "stop_cluster_ratio",           # float, sell/buy活跃量比（比率，可比）
    "change_pct",                   # float, 信号日涨跌幅（百分比，可比）
    "vol_zscore",                   # float, 信号日成交量Z-Score（z-score，可比）
    "daily_bb_width_zscore",        # float, 信号日布林带宽度Z-Score（z-score，可比）
    "last_event_bars_ago",          # int, 上次触发距今bar数（计数，可比）
    # 注：last_event_type 已剔除（低贡献+高基数类别）
    "has_buy_cluster",              # bool, 是否存在buy cluster（88%缺失标记列）
]

# ==================== 趋势类因子 ====================
TREND_COLS = [
    "dsa_dir",          # int, DSA趋势方向（1=上升，-1=下降）
    "prev_pivot_code",  # int, 前枢轴编码（HH=2,HL=1,LH=-1,LL=-2）
    "trend_align_momo", # int, 趋势-动量一致性（1=一致，-1=背离）
    "dsa_dir_age",      # int, DSA方向持续K线数
]

# ==================== 位置类因子 ====================
POSITION_COLS = [
    "dsa_pivot_pos_01",                        # float, DSA枢轴位置（0~1归一化）
    "price_vs_dsa_vwap_pct",                   # float, 价格相对DSA VWAP百分比
    "ret_to_last_high_pct",                    # float, 距前高回撤百分比
    "ret_to_last_low_pct",                     # float, 距前低反弹百分比
    "current_pullback_from_stage_extreme_pct", # float, 当前阶段极端回撤百分比
    "liquidity_range_pos_01",                  # float, 流动性区间位置（0~1归一化）
]

# ==================== 动量类因子（只保留可比的） ====================
MOMENTUM_COLS = [
    # 归一化/编码
    "bbmacd_band_pos_01",       # float, BBMACD布林带位置（0~1归一化）
    "bbmacd_bandwidth_zscore",  # float, BBMACD带宽Z-score
    "bbmacd_sign",              # int, BBMACD符号（1/0/-1）
    "bbmacd_slope_3_pct",       # float, BBMACD 3日斜率占价格百分比（归一化后可比）
    # 注：bbmacd_state 已剔除（低贡献+高基数类别）
    # 注：4个bbmacd_cross_* 已剔除（95%+为零，gain=0）
    # 注：bbmacd_slope_3 已替换为 bbmacd_slope_3_pct（绝对值不可跨股比较）
]

# ==================== 量能类因子 ====================
VOLUME_COLS = [
    "vol_zscore_5",        # float, 5日成交量Z-score
    "vol_zscore_10",       # float, 10日成交量Z-score
    "vol_zscore_20",       # float, 20日成交量Z-score
    "vol_ratio_10",        # float, 10日成交量比率
    "days_since_vol_spike", # int, 距上次放量天数
    "vol_stage_cv",        # float, 当前阶段量CV
    "vol_prev_stage_cv",   # float, 前一阶段量CV
    "vol_cv_ratio",        # float, CV比率
]

# ==================== 风险类因子 ====================
RISK_COLS = [
    "atr_pct",            # float, ATR占价格百分比（归一化）
    "volatility_20d",     # float, 20日波动率百分比
    "max_drawdown_60d",   # float, 60日最大回撤百分比
    "beta",               # float, Beta系数
]

# ==================== 节奏/协同类因子 ====================
RHYTHM_COLS = [
    "current_stage_bars",   # int, 当前阶段K线数
    "current_stage_amp_pct", # float, 当前阶段振幅百分比
    "current_stage_ret_pct", # float, 当前阶段收益百分比
    "prev_stage_bars",      # int, 前一阶段K线数
    "prev_stage_amp_pct",   # float, 前一阶段振幅百分比
    "price_vol_coord",      # int, 价量协同方向
]

# ==================== 动态特征（观察日实时计算） ====================
DYNAMIC_COLS = [
    "obs_day",               # int, 观察期第几天（1~20）
    "ret_to_trigger",        # float, 相对信号日涨跌幅
    "high_to_trigger",       # float, 观察期内最高价/信号日收盘价
    "low_to_trigger",        # float, 观察期内最低价/信号日收盘价
    "intraday_range",        # float, 当天日内振幅
    "vol_ratio",             # float, 当天成交量/20日均量
    "range_position",        # float, 收盘价在区间高低中位置
    "vol_change",            # float, 成交量变化率
    "dist_to_sell_stop_pct", # float, 当前价到卖止损距离(%)
    "dist_to_buy_stop_pct",  # float, 当前价到买止损距离(%)
]

# ==================== VSA 量价类因子 ====================
VSA_COLS = [
    "vsa_er_factor",          # float, VSA Effort-Rank - Result-Rank 核心 ER 因子
    "vsa_er_factor_ma",       # float, ER 因子 MA 平滑值
    "vsa_effort_rank",        # float, 量能努力等级 1-10
    "vsa_result_rank",        # float, 价差结果等级 1-10
    "vsa_vol_rank",           # float, 成交量滚动百分位排名 0-100
    "vsa_spread_rank",        # float, 价差滚动百分位排名 0-100
    "vsa_net_score",          # float, 多空背景净得分
    "vsa_strength_score",     # float, 多头背景得分
    "vsa_weakness_score",     # float, 空头背景得分
    "vsa_bull_score",         # float, 单根 K 线多头信号得分 (0/1/2/3)
    "vsa_bear_score",         # float, 单根 K 线空头信号得分 (0/1/2/3)
    "vsa_strong_move_z",      # float, 强移动 Z-score
]
DERIVED_COLS = [
    "cluster_count_ratio",       # active_sell / (active_buy + 1)
    "dist_atr_ratio",            # dist_sell_stop_atr / (dist_buy_stop_atr + 0.01)
    "stop_cluster_ratio_x_vol",  # stop_cluster_ratio * vol_zscore
    "trigger_count_ratio",       # active_sell_cluster_count * change_pct
]

# ==================== 标签列 ====================
REGRESSION_LABELS = ["mfe_20", "mae_20"]
CLASSIFICATION_LABELS = ["sell_signal", "buy_signal"]
ALL_LABELS = REGRESSION_LABELS + CLASSIFICATION_LABELS

# ==================== 元数据列 ====================
META_COLS = [
    "signal_id",          # stop_loss_selection 原表 id
    "ts_code",            # 股票代码
    "stock_name",         # 股票名称
    "signal_date",        # 信号触发日
    "obs_date",           # 观察日
    "obs_close",          # 观察日收盘价
    "can_buy",            # 是否可买入（非涨停）
    "selection_date",     # 选股日期（= signal_date）
]

# ==================== 因子类别标记（用于分组重要性分析） ====================
FACTOR_CATEGORIES = {
    "slc": SLC_STATIC_COLS,
    "trend": TREND_COLS,
    "position": POSITION_COLS,
    "momentum": MOMENTUM_COLS,
    "volume": VOLUME_COLS,
    "risk": RISK_COLS,
    "rhythm": RHYTHM_COLS,
    "dynamic": DYNAMIC_COLS,
    "derived": DERIVED_COLS,
    "vsa": VSA_COLS,
}

# ==================== 全部特征列 ====================
from stop_experiment.pipeline.stop_config import VSA_ENABLED

ALL_FEATURE_COLS = (
    SLC_STATIC_COLS + TREND_COLS + POSITION_COLS + MOMENTUM_COLS
    + VOLUME_COLS + RISK_COLS + RHYTHM_COLS + DYNAMIC_COLS + DERIVED_COLS
    + (VSA_COLS if VSA_ENABLED else [])
)
# ~69列含VSA (57基础 + 12 VSA)，默认57列（VSA_ENABLED=False）

# ==================== 剔除的不可比因子（仅作文档记录） ====================
EXCLUDED_ABSOLUTE_COLS = {
    # SLC绝对值因子
    "sell_trigger_volume": "绝对成交量",
    "buy_trigger_volume": "绝对成交量",
    "sum_sells_active": "绝对成交量",
    "sum_buys_active": "绝对成交量",
    "sell_trigger_max_vol_price": "绝对价格",
    "sell_stop_scale": "绝对价格×成交量",
    "nearest_sell_stop_price": "绝对价格",
    "nearest_buy_stop_price": "绝对价格",
    "last_event_volume": "绝对成交量",
    # 因子库绝对值因子
    "bbmacd": "绝对MACD值",
    "bbmacd_minus_avg": "绝对偏差值",
    "bbmacd_avg": "绝对信号线值",
    "bbmacd_std": "绝对标准差",
    "DSA_VWAP": "绝对价格",
    "m_rapida": "绝对移动均线",
    "m_lenta": "绝对移动均线",
    "dsa_atr": "绝对ATR值",
    "last_confirmed_high": "绝对价格",
    "last_confirmed_low": "绝对价格",
    "trendline_upper": "绝对价格",
    "trendline_lower": "绝对价格",
    "support_resistance_zones": "绝对价格",
    "liquidity_pools": "绝对价格",
    # v2 剔除：gain=0 或低贡献
    "bbmacd_slope_3": "绝对差分值，不可跨股比较（→ bbmacd_slope_3_pct）",
    "bbmacd_cross_upper": "95%+为零，gain=0",
    "bbmacd_cross_lower": "95%+为零，gain=0",
    "bbmacd_cross_up_lower": "95%+为零，gain=0",
    "bbmacd_cross_down_upper": "95%+为零，gain=0",
    "sell_stop_triggered": "gain=0",
    "buy_stop_triggered": "gain=0",
    "bbmacd_state": "低贡献+高基数类别编码",
    "last_event_type": "低贡献+高基数类别编码",
}
