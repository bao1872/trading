# market_structure_analysis/_config.py
# 集中管理模块内所有可配置阈值常量，避免散落不一致

# ——— 量能阈值 ———
VOL_ZSCORE_SPIKE_THRESHOLD = 2.0       # 成交量 zscore 放量/缩量判定阈值 (|zscore| > 2.0)
VOL_ZSCORE_PLAT_THRESHOLD = 0.5        # 平量判定区间 (-0.5 ~ 0.5)

# ——— Stop Cluster ———
NEAR_STOP_ATR_THRESHOLD = 1.0          # 止损簇"接近"判定，单位: ATR倍数

# ——— 标签 / 置信度 ———
V2_CONFIDENCE_THRESHOLD = 0.4          # v2 标签弱/强分界
CONFIDENCE_LOW_THRESHOLD = 0.3         # 低置信度告警线

# ——— 滚动窗口 ———
ZSCORE_WINDOW = 20                     # 滚动 zscore 回看窗口（交易日）

# ——— 最小样本 ———
MIN_INDUSTRY_SAMPLE = 20               # 行业组最小股票数（不足则排除出统计）
MIN_THEME_SAMPLE = 8                   # 题材/概念组最小股票数

# ——— 行业排名平滑 ———
SMOOTH_ALPHA = 0.6                     # 平滑权重: 当日得分 × alpha + 近3日均值 × (1-alpha)
SMOOTH_WINDOW = 3                      # 平滑回看窗口（交易日）
